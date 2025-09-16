import torch
import torch.nn as nn
import torch.optim as optim
from models import Generator, Discriminator
from data_loader import get_data_loader
import torchvision.utils as vutils
from PIL import Image
import numpy as np
import os

def save_single_tif_image(tensor, output_dir, epoch):
    """Save single 256x256 grayscale TIF file"""
    # Take the first (and only) image from the tensor
    img = tensor[0]
    # Convert single image tensor to numpy
    img_np = img.squeeze(0).cpu().numpy()  # Remove channel dimension if single channel
    
    # Convert from [0, 1] to [0, 255] for saving
    img_np = (img_np * 255).clip(0, 255).astype(np.uint8)
    
    # Save as grayscale TIF
    im = Image.fromarray(img_np, mode='L')
    filename = f"{output_dir}/generated_epoch_{epoch:03d}.tif"
    im.save(filename)
    
    print(f"Saved generated image for epoch {epoch}")

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
    
    # Apply proper weight initialization
    from models import weights_init
    netG.apply(weights_init)
    netD.apply(weights_init)
    
    criterion = nn.BCELoss()
    
    fixed_noise = torch.randn(1, nz, 1, 1, device=device)  # Single image per epoch
    
    # Clean binary labels without smoothing
    real_label = 1.0
    fake_label = 0.0
    
    # Moderate rebalancing - find middle ground
    optimizerD = optim.Adam(netD.parameters(), lr=lr * 0.2, betas=(beta1, 0.999))  # Discriminator learns 5x slower
    optimizerG = optim.Adam(netG.parameters(), lr=lr * 2, betas=(beta1, 0.999))  # Generator learns 2x faster
    
    img_list = []
    G_losses = []
    D_losses = []
    iters = 0
    
    print("Starting Training Loop...")
    for epoch in range(num_epochs):
        for i, data in enumerate(dataloader, 0):
            
            real_cpu = data.to(device)
            b_size = real_cpu.size(0)
            
            # Update D network every 3 iterations - moderate limitation
            if i % 3 == 0:
                netD.zero_grad()
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
            else:
                # Still need to compute these for logging
                with torch.no_grad():
                    output = netD(real_cpu).view(-1)  
                    D_x = output.mean().item()
                    noise = torch.randn(b_size, nz, 1, 1, device=device)
                    fake = netG(noise)
                    output = netD(fake.detach()).view(-1)
                    D_G_z1 = output.mean().item()
                    errD = torch.tensor(0.0)
            
            # Train generator 2 times per iteration - moderate advantage
            for g_step in range(2):
                netG.zero_grad()
                noise = torch.randn(b_size, nz, 1, 1, device=device)
                fake = netG(noise)
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
        
        # Save generated image at the end of each epoch
        with torch.no_grad():
            fake = netG(fixed_noise).detach().cpu()
        save_single_tif_image(fake, output_dir, epoch)
        
        # Save model every 5 epochs
        if (epoch + 1) % 50 == 0:
            model_dir = os.path.join(output_dir, "models")
            os.makedirs(model_dir, exist_ok=True)
            torch.save({
                'epoch': epoch,
                'netG_state_dict': netG.state_dict(),
                'netD_state_dict': netD.state_dict(),
                'optimizerG_state_dict': optimizerG.state_dict(),
                'optimizerD_state_dict': optimizerD.state_dict(),
                'G_losses': G_losses,
                'D_losses': D_losses,
            }, f"{model_dir}/checkpoint_epoch_{epoch:03d}.pth")
            print(f"Saved model checkpoint at epoch {epoch}")