"""
Author: Raphael Senn <raphaelsenn@gmx.de>
Initial coding: 2025-07-14
"""
import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader, Subset

from torchvision.transforms import transforms

from src.gan_celebfaces import (
    Generator,
    Discriminator,
)

from src.objective import (
    GeneratorLoss,
    DiscriminatorLoss
)

from src.dataset import CelebFaces


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
    SUBSET_SIZE_DATASET = 60000
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    VERBOSE = True
    EPOCHS = 100
    BATCH_SIZE = 100
    SHUFFLE = True
    LR_G = 0.0002
    LR_D = 0.0002
    BETAS_G = (0.5, 0.99)
    BETAS_D = (0.5, 0.99) 
    ROOT_DIR_CELEBFACES = "./celaba"
    DATA_FOLDER = "data"
    LANDMARS_FILE = "landmarks.csv"

    # --- Loading the Dataset ---
    transform = transforms.Compose([
        transforms.Grayscale(1),
        transforms.Resize(size=(96, 96)),
        transforms.CenterCrop(size=(48, 48)),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.view(-1))
    ])
    idx = torch.randperm(SUBSET_SIZE_DATASET)
    dataset = CelebFaces(ROOT_DIR_CELEBFACES, DATA_FOLDER, LANDMARS_FILE, transform)
    dataset = Subset(dataset, idx)
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
    torch.save(generator.state_dict(), 'generator_celebfaces.pth')
    torch.save(generator.state_dict(), 'discriminator_celebfaces.pth')