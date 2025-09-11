"""
Author: Raphael Senn <raphaelsenn@gmx.de>
Initial coding: 2025-07-14
"""
import torch
import torch.nn as nn

from src.utils import Maxout, EPSILON


class Generator(nn.Module):
    """
    Implementation of the MNIST-Generator from the original GAN paper.

    Reference:
    Generative Adversarial Networks, Goodfellow et al. 2014
    https://arxiv.org/abs/1406.2661
    """ 
    def __init__(
            self, 
            input_dim: int=100, 
            hidden_dim: int=1200, 
            output_dim: int=784,
            uniform_init: tuple[float, float] = (-0.005, 0.005)
        ) -> None:
        super().__init__()
        self.uniform_init = uniform_init
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(True),
            nn.Linear(hidden_dim, output_dim),
            nn.Sigmoid()
        )
        self._initialize_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
    
    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.uniform_(m.weight, self.uniform_init[0], self.uniform_init[1])
                if m.bias is not None:
                    nn.init.zeros_(m.bias)


class Discriminator(nn.Module):
    """
    Implementation of the MNIST-Discriminator from the original GAN paper.

    Reference:
    Generative Adversarial Networks, Goodfellow et al. 2014
    https://arxiv.org/abs/1406.2661
    """  
    def __init__(
            self, 
            input_dim: int=784,
            hidden_dim: int=240,
            num_pieces: int=5,
            input_p: float=0.2,
            hidden_p: float=0.5,
            uniform_init: tuple[float, float] = (-0.005, 0.005)
        ) -> None:
        super().__init__()
        self.uniform_init = uniform_init
        self.net = nn.Sequential(
            nn.Dropout(input_p), 
            Maxout(input_dim, hidden_dim, num_pieces),
            nn.Dropout(hidden_p),
            Maxout(hidden_dim, hidden_dim, num_pieces),
            nn.Dropout(hidden_p),
            nn.Linear(hidden_dim, 1), 
            nn.Sigmoid()
        )
        self._initialize_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.uniform_(m.weight, self.uniform_init[0], self.uniform_init[1])
                if m.bias is not None:
                    nn.init.zeros_(m.bias)