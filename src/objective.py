"""
Author: Raphael Senn <raphaelsenn@gmx.de>
Initial coding: 2025-07-14
"""
import torch
import torch.nn as nn


class GeneratorLoss(nn.Module):
    """
    Implementation of the (non-saturating) generator loss as described in the original GAN paper.

    NOTE: In this impl., we minimize -E[log(D(G(z)))] instead of maximize E[log(D(G(z)))].

    Objective:
    max_G E[log D(G(z))]    // original
    <=>
    min_G -E[log D(G(z))]
    <=> 
    min_G BCE(D(G(z)), 1)

    Reference:
    Generative Adversarial Networks, Goodfellow et al. 2014
    https://arxiv.org/abs/1406.2661
    """  
    def __init__(self) -> None:
        super().__init__() 
        self.bce_with_logits = nn.BCEWithLogitsLoss()

    def forward(self, D_G_z: torch.Tensor) -> torch.Tensor:
        return self.bce_with_logits(D_G_z, torch.ones_like(D_G_z, device=D_G_z.device))


class DiscriminatorLoss(nn.Module):
    """
    Implementation of the descriminator loss as described in the original gan paper.

    NOTE: In this impl., we minimize -E[log D(x)] - E_z[log (1 - D(G(z)))],
    instead of maxmize E[log D(x)] + E_z[log (1 - D(G(z)))]

    Objective:
    max_D E[log D(x)] + E_z[log (1 - D(G(z)))]          // original
    <=>
    min_D -(E[log E[D(x)] + E[log (1 - D(G(z)))])
    <=>
    min_D BCE(D(x), 1) + BCE(D(G(z)), 0)

    Reference:
    Generative Adversarial Networks, Goodfellow et al. 2014
    https://arxiv.org/abs/1406.2661
    """ 
    def __init__(self) -> None:
        super().__init__()  
        self.bce_with_logits = nn.BCEWithLogitsLoss()

    def forward(self, D_x: torch.Tensor, D_G_z: torch.Tensor) -> torch.Tensor:
        real = torch.ones_like(D_x, device=D_x.device)
        fake = torch.zeros_like(D_G_z, device=D_G_z.device)
        return self.bce_with_logits(D_x, real) + self.bce_with_logits(D_G_z, fake)