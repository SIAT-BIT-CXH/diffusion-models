import torch
import wandb

from default_mnist_config import create_default_mnist_config
from diffusion import DiffusionRunner
from models.classifier import ResNet, ResidualBlock, ConditionalResNet

import os

    #пишут что можно просто убрать
#os.environ['CUDA_VISIBLE_DEVICES'] = '3'


device = torch.device('cuda')
classifier_args = {
    "block": ResidualBlock,
    "layers": [2, 2, 2, 2]
}
model = ConditionalResNet(**classifier_args)
model.to(device)

optim = torch.optim.AdamW(model.parameters(), lr=1e-3)
loss_func = torch.nn.CrossEntropyLoss()

diffusion = DiffusionRunner(create_default_mnist_config(), eval=True)

diffusion.train_classifier(
    model,
    optim,
    loss_func,
    T=1.0
)
