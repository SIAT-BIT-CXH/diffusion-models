import torch
import torchvision
import wandb
import os
import math

import numpy as np

from models.ddpm import DDPM
from models.ema import ExponentialMovingAverage
from ddpm_sde import DDPM_SDE, EulerDiffEqSolver
from data_generator import DataGenerator

from ml_collections import ConfigDict
from typing import Optional, Union, Callable
from tqdm.auto import trange
from torch.nn import functional as F


class DiffusionRunner:
    def __init__(
            self,
            config: ConfigDict,
            eval: bool = False
    ):
        self.config = config
        self.eval = eval

        self.model = DDPM(config=config)
        self.sde = DDPM_SDE(config=config)
        self.diff_eq_solver = EulerDiffEqSolver(self.sde,
                                                self.calc_score,
                                                ode_sampling=config.training.ode_sampling)
        self.inverse_scaler = lambda x: torch.clip(127.5 * (x + 1), 0, 255)

        self.checkpoints_folder = config.training.checkpoints_folder
        if eval:
            self.ema = ExponentialMovingAverage(self.model.parameters(), config.model.ema_rate)
            self.restore_parameters()
            self.switch_to_ema()

        device = torch.device(self.config.device)
        self.device = device
        self.model.to(device)

    def restore_parameters(self, device: Optional[torch.device] = None) -> None:
        checkpoints_folder: str = self.checkpoints_folder
        if device is None:
            device = torch.device('cpu')
        model_ckpt = torch.load(checkpoints_folder + '/model.pth', map_location=device)
        self.model.load_state_dict(model_ckpt)

        ema_ckpt = torch.load(checkpoints_folder + '/ema.pth', map_location=device)
        self.ema.load_state_dict(ema_ckpt)

    def switch_to_ema(self) -> None:
        ema = self.ema
        score_model = self.model
        ema.store(score_model.parameters())
        ema.copy_to(score_model.parameters())

    def switch_back_from_ema(self) -> None:
        ema = self.ema
        score_model = self.model
        ema.restore(score_model.parameters())

    def set_optimizer(self) -> None:
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.config.optim.lr,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=self.config.optim.weight_decay
        )
        self.warmup = self.config.optim.linear_warmup
        self.grad_clip_norm = self.config.optim.grad_clip_norm
        self.optimizer = optimizer

        
    def calc_score(self, input_x: torch.Tensor, input_t: torch.Tensor, y=None) -> torch.Tensor:
        """
        calculate score w.r.t noisy X and t
        """
        #####
        
        eps = self.model(input_x, input_t)
        
        sigma = self.sde.marginal_std(input_t)

        score = - 1 * eps / sigma
        
        return score
    
    
    
    def sample_x_with_n(self, clean_x: torch.Tensor, t):
        
        eps = torch.randn_like(clean_x)
        x_mean, x_std = self.sde.marginal_prob(clean_x, t)
        

        return x_mean + x_std * eps, eps
    
    def sample_time(self, batch_size: int, eps: float = 1e-5):
        return torch.rand(batch_size, device = self.device) * (self.sde.T - eps) + eps
    
    
    def calc_loss(self, clean_x: torch.Tensor, eps: float = 1e-5) -> Union[float, torch.Tensor]:
        """
        Define score-matching MSE loss
        """
        ######
        batch_size = clean_x.shape[0]

        cur_t = self.sample_time(batch_size, eps)
        part_noise_x, eps = self.sample_x_with_n(clean_x, cur_t)
        
        part_loss = torch.sum(((eps - self.model(part_noise_x, cur_t))**2), (1, 2, 3))
        
        loss = torch.mean(part_loss)

        return loss

    def set_data_generator(self) -> None:
        self.datagen = DataGenerator(self.config)

    def manage_optimizer(self) -> None:
        self.lrs = []
        if self.warmup > 0 and self.step < self.warmup:
            for g in self.optimizer.param_groups:
                self.lrs += [g['lr']]
                g['lr'] = g['lr'] * float(self.step + 1) / self.warmup
        if self.grad_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                max_norm=self.grad_clip_norm
            )

    def restore_optimizer_state(self) -> None:
        if self.lrs:
            self.lrs = self.lrs[::-1]
            for g in self.optimizer.param_groups:
                g['lr'] = self.lrs.pop()

    def log_metric(self, metric_name: str, loader_name: str, value: Union[float, torch.Tensor, wandb.Image]):
        wandb.log({f'{metric_name}/{loader_name}': value}, step=self.step)

    def optimizer_step(self, loss: torch.Tensor) -> None:
        self.optimizer.zero_grad()
        loss.backward()

        self.manage_optimizer()
        self.log_metric('lr', 'train', self.optimizer.param_groups[0]['lr'])
        self.optimizer.step()
        self.ema.update(self.model.parameters())
        self.restore_optimizer_state()

    def validate(self) -> None:
        prev_mode= self.model.training

        self.model.eval()
        self.switch_to_ema()

        valid_loss = 0
        valid_count = 0
        with torch.no_grad():
            for (X, y) in self.datagen.valid_loader:
                X = X.to(self.device)
                loss = self.calc_loss(clean_x=X)
                valid_loss += loss.item() * X.size(0)
                valid_count += X.size(0)

        valid_loss = valid_loss / valid_count
        self.log_metric('loss', 'valid_loader', valid_loss)

        self.switch_back_from_ema()
        self.model.train(prev_mode)

    def train(self) -> None:
        self.set_optimizer()
        self.set_data_generator()
        train_generator = self.datagen.sample_train()
        self.step = 0

        wandb.init(project='sde', name='ddpm_cont')

        self.ema = ExponentialMovingAverage(self.model.parameters(), self.config.model.ema_rate)
        self.model.train()
        for iter_idx in trange(1, 1 + self.config.training.training_iters):
            self.step = iter_idx

            (X, y) = next(train_generator)
            X = X.to(self.device)
            loss = self.calc_loss(clean_x=X)
            self.log_metric('loss', 'train', loss.item())
            self.optimizer_step(loss)

            if iter_idx % self.config.training.snapshot_freq == 0:
                self.snapshot()

            if iter_idx % self.config.training.eval_freq == 0:
                self.validate()

            if iter_idx % self.config.training.checkpoint_freq == 0:
                self.save_checkpoint()

        self.model.eval()
        self.save_checkpoint()
        self.switch_to_ema()

    def save_checkpoint(self) -> None:
        if not os.path.exists(self.checkpoints_folder):
            os.makedirs(self.checkpoints_folder)
        torch.save(self.model.state_dict(), os.path.join(self.checkpoints_folder,
                                                               f'model.pth'))
        torch.save(self.ema.state_dict(), os.path.join(self.checkpoints_folder,
                                                       f'ema.pth'))
        torch.save(self.optimizer.state_dict(), os.path.join(self.checkpoints_folder,
                                                             f'opt.pth'))
        
    def reset_unconditional_sampling() -> None:
        self.diff_eq_solver = EulerDiffEqSolver(
            self.sde,
            self.calc_score,
            self.config.training.ode_sampling
        )

    def set_conditional_sampling(
            self,
            classifier_grad_fn: Callable[["NoisyImages", "T", "Labels"], "Scores"],
            T: float = 1.0
    ) -> None:
        def new_score_fn(x, t, y):
            """
            define posterior_score w.r.t T
            """
            ####
            
            cur_score = self.calc_score(x, t, y)

            posterior_score_T = cur_score + classifier_grad_fn(x, t, y) / T
            
            
            return posterior_score_T
        
        self.diff_eq_solver = EulerDiffEqSolver(
            self.sde,
            new_score_fn,
            self.config.training.ode_sampling
        )


    def set_classifier(self, classifier: torch.nn.Module, T: float = 1.0) -> None:
        self.classifier = classifier
        def classifier_grad_fn(x, t, y):
            """
            calculate likelihood_score with torch.autograd.grad
            """
            #####
            
            with torch.enable_grad():
                if not x.requires_grad == True:
                    x.requires_grad = True
                    
                x_ar = torch.arange(x.shape[0])
                log = self.classifier(x, t)[x_ar, y]
                
                likelihood_score = torch.autograd.grad(torch.sum(log), x)[0]
                
                
               
            return likelihood_score

        self.set_conditional_sampling(classifier_grad_fn, T=T)

    def sample_images(
            self, batch_size: int,
            eps:float = 1e-5,
            labels: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        shape = (
            batch_size,
            self.config.data.num_channels,
            self.config.data.image_size,
            self.config.data.image_size
        )
        device = torch.device(self.config.device)
        with torch.no_grad():
            """
            Implement cycle for Euler RSDE sampling w.r.t labels 
            labels = None if uncond. gen is used
            """
            #####
            
            pred_images = torch.randn(shape, device=device)
            circ = torch.linspace(self.sde.T, eps, self.sde.N, device=device)
            
            
            for t in circ:
                t = t.repeat(batch_size)
                pred_images, unn = self.diff_eq_solver.step(pred_images, t, labels)

            
                 
        return self.inverse_scaler(pred_images)

    def snapshot(self, labels: Optional[torch.Tensor] = None) -> None:
        prev_mode = self.model.training

        self.model.eval()
        self.switch_to_ema()

        images = self.sample_images(self.config.training.snapshot_batch_size, labels=labels).cpu()
        nrow = int(math.sqrt(self.config.training.snapshot_batch_size))
        grid = torchvision.utils.make_grid(images, nrow=nrow).permute(1, 2, 0)
        grid = grid.data.numpy().astype(np.uint8)
        self.log_metric('images', 'from_noise', wandb.Image(grid))

        self.switch_back_from_ema()
        self.model.train(prev_mode)

    def train_classifier(
            self,
            classifier: torch.nn.Module,
            classifier_optim: torch.optim.Optimizer,
            classifier_loss: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
            T: float = 10.0
    ) -> None:
        device = torch.device(self.config.device)
        self.device = device

        self.set_classifier(classifier, T=T)

        self.step = 0

        wandb.init(project='sde', name='noisy_classifier')

        def get_logits(X, y):
            
            t = self.sample_time(X.size(0)).to(device)

            """calc logits"""
            #########
            
            noise_x, unn = self.sample_x_with_n(X, t)
            
            pred_labels = self.classifier(noise_x, t)
            
            loss = classifier_loss(pred_labels, y)
            
            
            
            return loss, pred_labels

        self.set_data_generator()
        train_generator = self.datagen.sample_train()
        classifier.train()
        
        self.config.training.snapshot_batch_size = 100
        labels = np.tile(np.arange(10), (10, 1))
        labels = torch.Tensor(labels).to(device).long().view(-1)

        for iter_idx in trange(1, 1 + self.config.classifier.training_iters):
            self.step = iter_idx

            """
            train classifier
            """
            
            #########
            X, y = next(train_generator)
            X = X.to(self.device)
            y = y.to(self.device)

            loss, unn = get_logits(X, y)
            
            self.log_metric('cross_entropy', 'train', loss.item())

            classifier_optim.zero_grad()
            loss.backward()
            
            classifier_optim.step()
            
            
            

            if iter_idx % self.config.classifier.snapshot_freq == 0:
                self.snapshot(labels=labels)

            if iter_idx % self.config.classifier.eval_freq == 0:
                valid_loss = 0
                valid_accuracy = 0
                valid_count = 0
                classifier.eval()
                with torch.no_grad():
                    """
                    validate classifier
                    """
                    #######
                    
                    for (X, y) in self.datagen.valid_loader:
                        
                        X = X.to(self.device)
                        y = y.to(self.device)

                        loss, log = get_logits(X, y)
                        pred_labels = torch.argmax(log, dim=1)
                        
                        valid_count += X.size(0)
                        
                        valid_loss += loss.item() * X.size(0)
                        valid_accuracy += torch.sum((pred_labels == y))

                        
                    
                    
                    
                valid_loss = valid_loss / valid_count
                valid_accuracy = valid_accuracy / valid_count
                self.log_metric('cross_entropy', 'valid', valid_loss)
                self.log_metric('accuracy', 'valid', valid_accuracy)
                classifier.train()

            if iter_idx % self.config.classifier.checkpoint_freq == 0:
                torch.save(
                    classifier.state_dict(),
                    self.config.classifier.checkpoint_path
                )

        classifier.eval()
        torch.save(
            classifier.state_dict(),
            self.config.classifier.checkpoint_path
        )
