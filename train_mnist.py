"""
Author: Raphael Senn <raphaelsenn@gmx.de>
Initial coding: 2025-07-14
"""
import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from torchvision.transforms import transforms
from torchvision.datasets import MNIST

from src.gan_fc import (
    Generator,
    Discriminator,
)

from src.objective import (
    GeneratorLoss,
    DiscriminatorLoss
)

def train(
        generator: nn.Module,
        optimizer_g: Optimizer,
        criterion_g: nn.Module, 
        discriminator: nn.Module,
        optimizer_d: Optimizer,
        criterion_d: nn.Module,
        dataloader: DataLoader,
        epochs: int,
        device: torch.device,
        verbose: bool=True
    ) -> None:
    """
    Train a GAN using the given generator, discriminator, optimizers, and loss functions.
    """ 
    N_samples = len(dataset)
    for epoch in range(epochs):

        total_loss_g, total_loss_d = 0.0, 0.0
        for x, _ in dataloader:
            # --- Sampling: x ~ p_data, z ~ p_noise ---
            x = x.to(device)
            z = torch.rand(size=(x.shape[0], DIM_NOISE), device=device) * 2 - 1

            # --- Train Discriminator ---
            D_x = discriminator(x)
            D_G_z = discriminator(generator(z))
            optimizer_d.zero_grad()
            loss_d = criterion_d(D_x, D_G_z)
            loss_d.backward()
            optimizer_d.step()
            
            # --- Sampling: z ~ p_noise ---
            z = torch.rand(size=(x.shape[0], DIM_NOISE), device=device) * 2 - 1
            
            # --- Train Generator ---
            D_G_z = discriminator(generator(z))
            optimizer_g.zero_grad()
            loss_g = criterion_g(D_G_z)
            loss_g.backward()
            optimizer_g.step()

            total_loss_d += loss_d.item() * x.shape[0]
            total_loss_g += loss_g.item() * z.shape[0]

        if verbose: 
            print(
                f"epoch: {epoch}\t" 
                f"generator loss: {(total_loss_g/N_samples):.4f}\t"
                f"discriminator loss: {(total_loss_d/N_samples):.4f}"
            )


if __name__ == "__main__":
    # --- Settings and Hyperparameters ---
    DIM_NOISE = 100
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    VERBOSE = True
    EPOCHS = 100
    BATCH_SIZE = 128
    SHUFFLE = True
    LR_G = 0.002
    LR_D = 0.002
    BETAS_G = (0.5, 0.99)
    BETAS_D = (0.5, 0.99) 
    ROOT_DIR_MNIST = "./MNIST"

    # --- Loading the Dataset ---
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.view(-1))
    ])
    dataset = MNIST(ROOT_DIR_MNIST, train=True, download=True, transform=transform)
    dataloader = DataLoader(dataset, BATCH_SIZE, SHUFFLE)

    # --- Create generator and discriminator ---
    generator = Generator(DIM_NOISE).to(DEVICE)
    discriminator = Discriminator().to(DEVICE)
    criterion_g = GeneratorLoss(maximize=True)
    optimizer_g = torch.optim.Adam(generator.parameters(), LR_G, BETAS_G, maximize=True)
    criterion_d = DiscriminatorLoss()
    optimizer_d = torch.optim.Adam(discriminator.parameters(), LR_D, BETAS_D, maximize=True)
    
    # --- Start training ---
    train(
        generator, optimizer_g, criterion_g, 
        discriminator, optimizer_d, criterion_d,
        dataloader, EPOCHS, DEVICE, VERBOSE     
    )
    # --- Save trained models ---
    torch.save(generator.state_dict(), "generator_mnist.pth")
    torch.save(generator.state_dict(), "discriminator_mnist.pth")