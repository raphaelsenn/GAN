"""
Author: Raphael Senn <raphaelsenn@gmx.de>
Initial coding: 2025-07-14
"""
import torch
import torch.nn as nn

from src.utils import EPSILON


class GeneratorLoss(nn.Module):
    """
    Implementation of the generator loss as described in the original GAN paper.

    Reference:
    Generative Adversarial Networks, Goodfellow et al. 2014
    https://arxiv.org/abs/1406.2661
    """  
    def __init__(self, maximize: bool=False) -> None:
        super().__init__() 
        self.maximize = maximize

    def forward(self, D_G_z: torch.Tensor) -> torch.Tensor:
        if self.maximize:
            return torch.log(D_G_z + EPSILON).mean()
        return torch.log(1 - D_G_z + EPSILON).mean()


class DiscriminatorLoss(nn.Module):
    """
    Implementation of the descriminator loss as described in the original gan paper.
    
    Reference:
    Generative Adversarial Networks, Goodfellow et al. 2014
    https://arxiv.org/abs/1406.2661

    NOTE: Im not stupid an know Binary cross-entropy loss.
    """ 
    def __init__(self) -> None:
        super().__init__()  
    
    def forward(self, D_x: torch.Tensor, D_G_z: torch.Tensor) -> torch.Tensor:
        return (torch.log(D_x + EPSILON) + torch.log(1 - D_G_z + EPSILON)).mean()