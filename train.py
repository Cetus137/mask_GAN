import torch
import torch.nn as nn
import torch.optim as optim
from models import Generator, Discriminator
from data_loader import get_data_loader
import torchvision.utils as vutils
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os

def save_comparison_image(fake_tensor, real_tensor, output_dir, epoch):
    """Save comparison figure with generated and real images side by side"""
    
    # Convert tensors to numpy
    fake_img = fake_tensor[0].squeeze(0).cpu().numpy()  # Remove channel dimension
    real_img = real_tensor[0].squeeze(0).cpu().numpy()  # Remove channel dimension
    
    # Create comparison figure
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    
    # Plot generated image
    axes[0].imshow(fake_img, cmap='gray', vmin=0, vmax=1)
    axes[0].set_title(f'Generated (Epoch {epoch})')
    axes[0].axis('off')
    
    # Plot real target image
    axes[1].imshow(real_img, cmap='gray', vmin=0, vmax=1)
    axes[1].set_title('Real Target')
    axes[1].axis('off')
    
    plt.tight_layout()
    
    # Save comparison figure
    filename = f"{output_dir}/comparison_epoch_{epoch:03d}.png"
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    
    # Also save individual TIF files
    fake_np = (fake_img * 255).clip(0, 255).astype(np.uint8)
    real_np = (real_img * 255).clip(0, 255).astype(np.uint8)
    
    Image.fromarray(fake_np, mode='L').save(f"{output_dir}/generated_epoch_{epoch:03d}.tif")
    Image.fromarray(real_np, mode='L').save(f"{output_dir}/target_epoch_{epoch:03d}.tif")
    
    print(f"Saved comparison image for epoch {epoch}")

def gradient_penalty(netD, real_data, fake_data, device, lambda_gp=10):
    """Calculate gradient penalty for WGAN-GP"""
    batch_size = real_data.size(0)
    alpha = torch.rand(batch_size, 1, 1, 1, device=device)
    alpha = alpha.expand_as(real_data)
    
    interpolated = alpha * real_data + (1 - alpha) * fake_data
    interpolated = interpolated.requires_grad_(True)
    
    # Calculate discriminator output for interpolated samples
    d_interpolated = netD(interpolated)
    
    # Calculate gradients
    gradients = torch.autograd.grad(
        outputs=d_interpolated,
        inputs=interpolated,
        grad_outputs=torch.ones_like(d_interpolated),
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]
    
    # Flatten gradients
    gradients = gradients.view(batch_size, -1)
    gradient_norm = gradients.norm(2, dim=1)
    penalty = lambda_gp * ((gradient_norm - 1) ** 2).mean()
    
    return penalty

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
    
    print(f"Generator parameters: {sum(p.numel() for p in netG.parameters()):,}")
    print(f"Discriminator parameters: {sum(p.numel() for p in netD.parameters()):,}")
    
    fixed_noise = torch.randn(1, nz, 1, 1, device=device)  # Single image per epoch
    
    # Traditional GAN parameters
    criterion = nn.BCELoss()  # Binary Cross Entropy Loss
    
    # Standard optimizers for traditional GAN
    optimizerD = optim.Adam(netD.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=0.0002, betas=(0.5, 0.999))
    
    img_list = []
    G_losses = []
    D_losses = []
    iters = 0
    
    print("Starting Traditional GAN Training Loop...")
    print(f"Using device: {device}")
    
    # Create labels for real and fake
    real_label = 1.
    fake_label = 0.
    
    for epoch in range(num_epochs):
        for i, data in enumerate(dataloader, 0):
            real_cpu = data.to(device)
            b_size = real_cpu.size(0)
            
            ############################
            # (1) Train Discriminator
            ############################
            netD.zero_grad()
            
            # Train with real data
            label = torch.full((b_size,), real_label, dtype=torch.float, device=device)
            real_output = netD(real_cpu).view(-1)
            errD_real = criterion(real_output, label)
            errD_real.backward()
            D_x = real_output.mean().item()
            
            # Train with fake data
            noise = torch.randn(b_size, nz, 1, 1, device=device)
            fake = netG(noise)
            label.fill_(fake_label)
            fake_output = netD(fake.detach()).view(-1)
            errD_fake = criterion(fake_output, label)
            errD_fake.backward()
            D_G_z1 = fake_output.mean().item()
            
            # Total discriminator loss
            errD = errD_real + errD_fake
            optimizerD.step()
            
            ############################
            # (2) Train Generator
            ############################
            netG.zero_grad()
            
            # We want generator to fool discriminator, so use real labels
            label.fill_(real_label)
            fake_output = netD(fake).view(-1)
            errG = criterion(fake_output, label)
            errG.backward()
            D_G_z2 = fake_output.mean().item()
            optimizerG.step()
            
            # Verbose logging every batch
            if i % 10 == 0:
                print(f'[{epoch:4d}/{num_epochs}][{i:3d}/{len(dataloader)}] '
                      f'Loss_D: {errD.item():8.4f} | Loss_G: {errG.item():8.4f} | '
                      f'D(x): {D_x:6.4f} | D(G(z)): {D_G_z1:6.4f}/{D_G_z2:6.4f}')
            
            G_losses.append(errG.item())
            D_losses.append(errD.item())
            
            if (iters % 500 == 0) or ((epoch == num_epochs-1) and (i == len(dataloader)-1)):
                with torch.no_grad():
                    fake = netG(fixed_noise).detach().cpu()
                img_list.append(vutils.make_grid(fake, padding=2, normalize=True))
                
            iters += 1
        
        # Epoch summary
        avg_D_loss = sum(D_losses[-len(dataloader):]) / len(dataloader) if D_losses else 0
        avg_G_loss = sum(G_losses[-len(dataloader):]) / len(dataloader) if G_losses else 0
        print(f"\n=== EPOCH {epoch+1}/{num_epochs} SUMMARY ===")
        print(f"Average D Loss: {avg_D_loss:.6f}")
        print(f"Average G Loss: {avg_G_loss:.6f}")
        print(f"Final D(x): {D_x:.6f} (want ~0.5)")
        print(f"Final D(G(z)): {D_G_z2:.6f} (want ~0.5)")
        print("=" * 40)
        
        # Save comparison image every 10 epochs
        if (epoch + 1) % 10 == 0 and len(dataloader) > 0:
            with torch.no_grad():
                fake = netG(fixed_noise).detach().cpu()
                # Get a real sample for comparison
                real_sample = next(iter(dataloader))  # Get first batch
                real_sample = real_sample.cpu()
            save_comparison_image(fake, real_sample, output_dir, epoch)
        
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