import torch
import torch.nn as nn
import torch.optim as optim
from models import Generator, Discriminator
from data_loader import get_data_loader
import torchvision.utils as vutils
from PIL import Image
import numpy as np
import os

def save_individual_tif_images(tensor, output_dir, epoch):
    """Save individual 256x256 grayscale TIF files"""
    for i, img in enumerate(tensor):
        # Convert single image tensor to numpy
        img_np = img.squeeze(0).cpu().numpy()  # Remove channel dimension if single channel
        
        # Denormalize from [-1, 1] to [0, 255]
        img_np = ((img_np + 1) * 127.5).clip(0, 255).astype(np.uint8)
        
        # Save as grayscale TIF
        im = Image.fromarray(img_np, mode='L')
        filename = f"{output_dir}/generated_epoch_{epoch:03d}_img_{i:02d}.tif"
        im.save(filename)
    
    print(f"Saved {len(tensor)} individual images for epoch {epoch}")

def train(data_dir, nz, nc, ngf, ndf, num_epochs, batch_size, image_size, lr, beta1, output_dir):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    dataloader = get_data_loader(data_dir, batch_size, image_size)
    
    # Print dataset information
    dataset_size = len(dataloader.dataset)
    num_batches = len(dataloader)
    print(f"Dataset size: {dataset_size} images")
    print(f"Number of batches: {num_batches}")
    print(f"Batch size: {batch_size}")
    
    if dataset_size < batch_size:
        print(f"Warning: Dataset size ({dataset_size}) is smaller than batch size ({batch_size})")
        print("Consider reducing batch size or adding more training data")
    
    netG = Generator(nz, ngf, nc).to(device)
    netD = Discriminator(nc, ndf).to(device)
    
    criterion = nn.BCELoss()
    
    fixed_noise = torch.randn(16, nz, 1, 1, device=device)  # Reduced for 256x256 images
    
    # Label smoothing to prevent discriminator from becoming too confident
    real_label = 0.9  # Instead of 1.0
    fake_label = 0.1  # Instead of 0.0
    
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
            
            real_labels = torch.full((b_size,), real_label, dtype=torch.float, device=device)
            output = netD(real_cpu).view(-1)
            errD_real = criterion(output, real_labels)
            errD_real.backward()
            D_x = output.mean().item()
            
            noise = torch.randn(b_size, nz, 1, 1, device=device)
            fake = netG(noise)
            
            fake_labels = torch.full((b_size,), fake_label, dtype=torch.float, device=device)
            output = netD(fake.detach()).view(-1)
            errD_fake = criterion(output, fake_labels)
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            errD = errD_real + errD_fake
            optimizerD.step()
            
            # Update G network (train generator more if discriminator is too strong)
            g_train_steps = 2 if D_x > 0.8 else 1  # Train G more if D is too confident
            
            for _ in range(g_train_steps):
                netG.zero_grad()
                # Generate new fake images for generator training
                noise = torch.randn(b_size, nz, 1, 1, device=device)
                fake = netG(noise)
                # Generator wants discriminator to think fake images are real
                gen_labels = torch.full((b_size,), real_label, dtype=torch.float, device=device)
                output = netD(fake).view(-1)
                errG = criterion(output, gen_labels)
                errG.backward()
                optimizerG.step()
            
            D_G_z2 = output.mean().item()
            
            if i % 50 == 0:
                print(f'[{epoch}/{num_epochs}][{i}/{len(dataloader)}] Loss_D: {errD.item():.4f} Loss_G: {errG.item():.4f} D(x): {D_x:.4f} D(G(z)): {D_G_z1:.4f} / {D_G_z2:.4f}')
            
            G_losses.append(errG.item())
            D_losses.append(errD.item())
            
            if (iters % 500 == 0) or ((epoch == num_epochs-1) and (i == len(dataloader)-1)):
                with torch.no_grad():
                    fake = netG(fixed_noise).detach().cpu()
                img_list.append(vutils.make_grid(fake, padding=2, normalize=True))
                
            iters += 1
        
        # Save generated images at the end of each epoch
        with torch.no_grad():
            fake = netG(fixed_noise).detach().cpu()
        save_individual_tif_images(fake, output_dir, epoch)