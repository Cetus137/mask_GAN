import torch
import torch.nn as nn
import torch.optim as optim
from models import Generator, Discriminator
from data_loader import get_data_loader
import torchvision.utils as vutils
from PIL import Image
import numpy as np
import os

def save_grayscale_tif(tensor, filename, nrow=8):
    """Save tensor as grayscale TIF file"""
    # Convert tensor to numpy and denormalize
    grid = vutils.make_grid(tensor, nrow=nrow, normalize=True, padding=2)
    ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
    
    # Convert RGB to grayscale if needed
    if ndarr.shape[2] == 3:
        ndarr = np.dot(ndarr[...,:3], [0.2989, 0.5870, 0.1140]).astype(np.uint8)
    elif ndarr.shape[2] == 1:
        ndarr = ndarr.squeeze(-1)
    
    # Save as grayscale TIF
    im = Image.fromarray(ndarr, mode='L')
    im.save(filename)

def train(data_dir, nz, nc, ngf, ndf, num_epochs, batch_size, image_size, lr, beta1, output_dir):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    dataloader = get_data_loader(data_dir, batch_size, image_size)
    
    netG = Generator(nz, ngf, nc).to(device)
    netD = Discriminator(nc, ndf).to(device)
    
    criterion = nn.BCELoss()
    
    fixed_noise = torch.randn(16, nz, 1, 1, device=device)  # Reduced for 256x256 images
    
    real_label = 1.
    fake_label = 0.
    
    optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))
    
    img_list = []
    G_losses = []
    D_losses = []
    iters = 0
    
    print("Starting Training Loop...")
    for epoch in range(num_epochs):
        for i, data in enumerate(dataloader, 0):
            
            # Update D network
            netD.zero_grad()
            real_cpu = data.to(device)
            b_size = real_cpu.size(0)
            label = torch.full((b_size,), real_label, dtype=torch.float, device=device)
            output = netD(real_cpu).view(-1)
            errD_real = criterion(output, label)
            errD_real.backward()
            D_x = output.mean().item()
            
            noise = torch.randn(b_size, nz, 1, 1, device=device)
            fake = netG(noise)
            label.fill_(fake_label)
            output = netD(fake.detach()).view(-1)
            errD_fake = criterion(output, label)
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            errD = errD_real + errD_fake
            optimizerD.step()
            
            # Update G network
            netG.zero_grad()
            label.fill_(real_label)
            output = netD(fake).view(-1)
            errG = criterion(output, label)
            errG.backward()
            D_G_z2 = output.mean().item()
            optimizerG.step()
            
            if i % 50 == 0:
                print(f'[{epoch}/{num_epochs}][{i}/{len(dataloader)}] Loss_D: {errD.item():.4f} Loss_G: {errG.item():.4f} D(x): {D_x:.4f} D(G(z)): {D_G_z1:.4f} / {D_G_z2:.4f}')
            
            G_losses.append(errG.item())
            D_losses.append(errD.item())
            
            if (iters % 500 == 0) or ((epoch == num_epochs-1) and (i == len(dataloader)-1)):
                with torch.no_grad():
                    fake = netG(fixed_noise).detach().cpu()
                img_list.append(vutils.make_grid(fake, padding=2, normalize=True))
                
            iters += 1
        
        # Save generated image at the end of each epoch
        with torch.no_grad():
            fake = netG(fixed_noise).detach().cpu()
        save_grayscale_tif(fake, f"{output_dir}/fake_samples_epoch_{epoch}.tif", nrow=4)