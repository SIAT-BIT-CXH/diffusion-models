import torch
import numpy as np


class DDPM_SDE:
    def __init__(self, config):
        """Construct a Variance Preserving SDE.

        Args:
          beta_min: value of beta(0)
          beta_max: value of beta(1)
          N: number of discretization steps
        """
        self.N = config.sde.N
        self.beta_0 = config.sde.beta_min
        self.beta_1 = config.sde.beta_max

    @property
    def T(self):
        return 1

    def sde(self, x, t):
        """
        Calculate drift coeff. and diffusion coeff. in forward SDE
        """
        ######
        
        beta_t = (self.beta_1 - self.beta_0) * t + self.beta_0
        drift = -(1/2) * x *  beta_t.reshape(-1, 1, 1, 1)
        beta_sqrt = torch.sqrt(beta_t)
        diffusion = beta_sqrt.reshape(-1, 1, 1, 1)
        
        return drift, diffusion

    def marginal_prob(self, x_0, t):
        """
        Calculate marginal q(x_t|x_0)'s mean and std
        """
        #########
        
        integral = (1/2) * (self.beta_1 - self.beta_0) * t ** 2 + self.beta_0 * t
        mean = torch.exp(-(1/2) * integral).reshape(-1, 1, 1, 1) * x_0
        std = torch.sqrt(1 - torch.exp(-integral))
        std = std.reshape(-1, 1, 1, 1)
        return mean, std
    
    def marginal_std(self, t):
        """
        Calculate marginal q(x_t|x_0)'s std
        """
        intgr = (1/2) * (self.beta_1 - self.beta_0) * t**2 + self.beta_0 * t
        std = torch.sqrt(1 - torch.exp(-1 * intgr))
        std = std.reshape(-1, 1, 1, 1)
        return std

    def prior_sampling(self, shape):
        return torch.randn(*shape)

    def reverse(self, score_fn, ode_sampling=False):
        """Create the reverse-time SDE/ODE.
        Args:
          score_fn: A time-dependent score-based model that takes x and t and returns the score.
          ode_sampling: If `True`, create the reverse-time ODE used for probability flow sampling.
        """
        N = self.N
        T = self.T
        sde_fn = self.sde

        # Build the class for reverse-time SDE.
        class RSDE:
            def __init__(self):
                self.N = N
                self.ode_sampling = ode_sampling

            @property
            def T(self):
                return T

            def sde(self, x, t, y=None):
                """
                Create the drift and diffusion functions for the reverse SDE/ODE.
                
                
                y is here for class-conditional generation through score SDE/ODE
                """
                
                """
                Calculate drift and diffusion for reverse SDE/ODE
                
                
                ode_sampling - True -> reverse SDE
                ode_sampling - False -> reverse ODE
                """
                ###
                
                part_drift, part_diff = sde_fn(x, t)
                
                prod = score_fn(x, t, y)

                part_to_det = part_diff**2 * prod
                
                if self.ode_sampling: #reverce SDE and ODE
                    drift = part_drift - (1/2) * part_to_det
                    diffusion = torch.zeros_like(part_diff)
                else:
                    diffusion = part_diff
                    drift = part_drift - part_to_det
                
                
                return drift, diffusion

        return RSDE()


class EulerDiffEqSolver:
    def __init__(self, sde, score_fn, ode_sampling = False):
        self.sde = sde
        self.score_fn = score_fn
        self.ode_sampling = ode_sampling
        self.rsde = sde.reverse(score_fn, ode_sampling)

    def step(self, x, t, y=None):
        """
        Implement reverse SDE/ODE Euler solver
        """
        
        
        """
        x_mean = deterministic part
        x = x_mean + noise (yet another noise sampling)
        """
        ###
        cur_n = self.rsde.N
        
        step = -1/cur_n
        
        drift, diff = self.rsde.sde(x, t, y)

        x_mean = drift * step + x

        noise = torch.randn_like(x)
        
        x = x_mean + diff * noise * abs(step)**0.5
        
        return x, x_mean
