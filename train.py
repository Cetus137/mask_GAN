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
    
    # WGAN-GP parameters
    lambda_gp = 10
    n_critic = 5  # Train discriminator 5 times per generator update
    
    # Optimizers for WGAN-GP - using even lower learning rates for stability
    optimizerD = optim.Adam(netD.parameters(), lr=0.00005, betas=(0.0, 0.9))  # Lower lr for WGAN-GP
    optimizerG = optim.Adam(netG.parameters(), lr=0.00005, betas=(0.0, 0.9))  # Matching lr for balance
    
    img_list = []
    G_losses = []
    D_losses = []
    iters = 0
    
    print("Starting WGAN-GP Training Loop...")
    print(f"Using device: {device}")
    print(f"Lambda GP: {lambda_gp}, n_critic: {n_critic}")
    
    for epoch in range(num_epochs):
        for i, data in enumerate(dataloader, 0):
            real_cpu = data.to(device)
            b_size = real_cpu.size(0)
            
            ############################
            # (1) Train Discriminator/Critic
            ############################
            for _ in range(n_critic):
                netD.zero_grad()
                
                # Train with real data
                real_output = netD(real_cpu).view(-1)
                errD_real = -torch.mean(real_output)  # WGAN loss: -E[D(x)]
                
                # Train with fake data
                noise = torch.randn(b_size, nz, 1, 1, device=device)
                fake = netG(noise).detach()  # Detach to avoid training G
                fake_output = netD(fake).view(-1)
                errD_fake = torch.mean(fake_output)   # WGAN loss: E[D(G(z))]
                
                # Gradient penalty
                gp = gradient_penalty(netD, real_cpu, fake, device, lambda_gp)
                
                # Total discriminator loss
                errD = errD_real + errD_fake + gp
                errD.backward()
                # Gradient clipping for stability
                torch.nn.utils.clip_grad_norm_(netD.parameters(), max_norm=1.0)
                optimizerD.step()
            
            ############################
            # (2) Train Generator
            ############################
            netG.zero_grad()
            
            # Generate fake data
            noise = torch.randn(b_size, nz, 1, 1, device=device)
            fake = netG(noise)
            fake_output = netD(fake).view(-1)
            
            # WGAN Generator loss: -E[D(G(z))]
            errG_adv = -torch.mean(fake_output)
            
            # Add spatial regularization losses for better structure
            # Binary regularization - encourage 0 or 1 values
            binary_loss = torch.mean(4 * fake * (1 - fake))
            
            # Total variation loss for spatial smoothness
            tv_h = torch.mean(torch.abs(fake[:, :, :, 1:] - fake[:, :, :, :-1]))
            tv_v = torch.mean(torch.abs(fake[:, :, 1:, :] - fake[:, :, :-1, :]))
            tv_loss = tv_h + tv_v
            
            # Edge-preserving smoothness loss
            edge_loss = torch.mean(torch.abs(fake[:, :, 2:, :] - 2*fake[:, :, 1:-1, :] + fake[:, :, :-2, :]))
            edge_loss += torch.mean(torch.abs(fake[:, :, :, 2:] - 2*fake[:, :, :, 1:-1] + fake[:, :, :, :-2]))
            
            # Add hard thresholding loss - penalize values not close to 0 or 1
            threshold_loss = torch.mean(torch.minimum(fake, 1-fake))  # Minimized when fake is 0 or 1
            
            # Add entropy loss to discourage gray values
            eps = 1e-8
            entropy_loss = -torch.mean(fake * torch.log(fake + eps) + (1-fake) * torch.log(1-fake + eps))
            
            # Add connectivity loss to encourage coherent regions (shape-agnostic)
            # This encourages pixels to be similar to their neighbors
            connectivity_loss = torch.mean(torch.abs(fake[:, :, :-1, :] - fake[:, :, 1:, :])) + \
                              torch.mean(torch.abs(fake[:, :, :, :-1] - fake[:, :, :, 1:]))
            
            # Combined generator loss with much stronger binary constraints (no shape assumptions)
            errG = errG_adv + 3.0 * binary_loss + 0.1 * tv_loss + 0.1 * edge_loss + 2.0 * threshold_loss + 1.0 * entropy_loss + 0.3 * connectivity_loss
            
            errG.backward()
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(netG.parameters(), max_norm=1.0)
            optimizerG.step()
            
            # Calculate metrics for logging
            D_x = torch.mean(real_output).item()
            D_G_z = torch.mean(fake_output).item()
            
            # Verbose logging every batch
            if i % 10 == 0:
                print(f'[{epoch:4d}/{num_epochs}][{i:3d}/{len(dataloader)}] '
                      f'Loss_D: {errD.item():8.4f} | Loss_G: {errG.item():8.4f} | '
                      f'D(x): {D_x:6.4f} | D(G(z)): {D_G_z:6.4f} | '
                      f'GP: {gp.item():6.4f} | Binary: {binary_loss.item():6.4f} | '
                      f'TV: {tv_loss.item():6.4f} | Thresh: {threshold_loss.item():6.4f} | Entropy: {entropy_loss.item():6.4f}')
            
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
        print(f"Final D(G(z)): {D_G_z:.6f} (want ~0.5)")
        print("=" * 40)
        
        # Save generated image every 10 epochs
        if (epoch + 1) % 10 == 0:
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