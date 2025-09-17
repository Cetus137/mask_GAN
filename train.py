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

def gradient_penalty(netD, real_data, fake_data, device, lambda_gp=10):
    """Calculate gradient penalty for WGAN-GP"""
    batch_size = real_data.size(0)
    
    # Random weight term for interpolation between real and fake data
    alpha = torch.rand(batch_size, 1, 1, 1, device=device)
    
    # Get random interpolation between real and fake data
    interpolates = alpha * real_data + (1 - alpha) * fake_data
    interpolates = interpolates.to(device)
    interpolates.requires_grad_(True)
    
    # Get discriminator output for interpolated data
    disc_interpolates = netD(interpolates)
    
    # Get gradients with respect to interpolates
    gradients = torch.autograd.grad(
        outputs=disc_interpolates,
        inputs=interpolates,
        grad_outputs=torch.ones_like(disc_interpolates),
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]
    
    # Calculate gradient penalty
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * lambda_gp
    
    return gradient_penalty

def save_comparison_image(fake_tensor, real_tensor, output_dir, epoch, suffix=""):
    """Save comparison figure with generated and real images side by side"""
    
    # Convert tensors to numpy and handle [-1,1] range
    fake_img = fake_tensor[0].squeeze(0).cpu().numpy()  # Remove channel dimension
    real_img = real_tensor[0].squeeze(0).cpu().numpy()  # Remove channel dimension
    
    # Convert from [-1,1] to [0,1] for visualization
    fake_img = (fake_img + 1.0) / 2.0
    real_img = (real_img + 1.0) / 2.0
    
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
    suffix_str = f"_{suffix}" if suffix else ""
    filename = f"{output_dir}/comparison_epoch_{epoch:03d}{suffix_str}.png"
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    
    # Also save individual TIF files
    fake_np = (fake_img * 255).clip(0, 255).astype(np.uint8)
    real_np = (real_img * 255).clip(0, 255).astype(np.uint8)
    
    Image.fromarray(fake_np, mode='L').save(f"{output_dir}/generated_epoch_{epoch:03d}{suffix_str}.tif")
    Image.fromarray(real_np, mode='L').save(f"{output_dir}/target_epoch_{epoch:03d}{suffix_str}.tif")
    
    print(f"Saved comparison image for epoch {epoch}{suffix_str}")

def save_varied_samples(fake_tensor, output_dir, epoch):
    """Save a grid of varied generated samples to show diversity"""
    
    # Convert tensors to numpy and handle [-1,1] range
    num_samples = fake_tensor.shape[0]
    
    # Create a grid of samples
    fig, axes = plt.subplots(1, num_samples, figsize=(num_samples * 3, 3))
    if num_samples == 1:
        axes = [axes]  # Handle single sample case
    
    for i in range(num_samples):
        fake_img = fake_tensor[i].squeeze(0).cpu().numpy()  # Remove channel dimension
        
        # Convert from [-1,1] to [0,1] for visualization
        fake_img = (fake_img + 1.0) / 2.0
        
        axes[i].imshow(fake_img, cmap='gray', vmin=0, vmax=1)
        axes[i].set_title(f'Sample {i+1}')
        axes[i].axis('off')
        
        # Save individual TIF file
        fake_np = (fake_img * 255).clip(0, 255).astype(np.uint8)
        Image.fromarray(fake_np, mode='L').save(f"{output_dir}/varied_sample_{i+1}_epoch_{epoch:03d}.tif")
    
    plt.tight_layout()
    
    # Save grid figure
    filename = f"{output_dir}/varied_samples_epoch_{epoch:03d}.png"
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved {num_samples} varied samples for epoch {epoch}")

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

def train(data_dir, nz, nc, ngf, ndf, num_epochs, batch_size, image_size, lr, beta1, output_dir, resume_from=None):
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
    
    # Create multiple fixed noise vectors for consistent progress tracking
    fixed_noise = torch.randn(1, nz, 1, 1, device=device)  # Fixed for progress tracking
    num_samples = 4  # Generate multiple samples each epoch
    sample_noise = torch.randn(num_samples, nz, 1, 1, device=device)  # Different samples each time
    
    # WGAN-GP parameters - stable loss function
    lambda_gp = 10
    n_critic = 1  # Balanced training: 1:1 ratio for small dataset
    
    # WGAN-GP optimizers with lower learning rates
    optimizerD = optim.Adam(netD.parameters(), lr=0.0001, betas=(0.0, 0.9))
    optimizerG = optim.Adam(netG.parameters(), lr=0.0001, betas=(0.0, 0.9))
    
    img_list = []
    G_losses = []
    D_losses = []
    iters = 0
    start_epoch = 0
    
    # Load checkpoint if resuming training
    if resume_from is not None:
        if os.path.isfile(resume_from):
            print(f"Loading checkpoint from {resume_from}")
            checkpoint = torch.load(resume_from, map_location=device)
            
            # Load model states
            netG.load_state_dict(checkpoint['netG_state_dict'])
            netD.load_state_dict(checkpoint['netD_state_dict'])
            
            # Load optimizer states
            optimizerG.load_state_dict(checkpoint['optimizerG_state_dict'])
            optimizerD.load_state_dict(checkpoint['optimizerD_state_dict'])
            
            # Load training progress
            start_epoch = checkpoint['epoch'] + 1  # Start from next epoch
            if 'G_losses' in checkpoint:
                G_losses = checkpoint['G_losses']
            if 'D_losses' in checkpoint:
                D_losses = checkpoint['D_losses']
            if 'iters' in checkpoint:
                iters = checkpoint.get('iters', 0)
            
            print(f"Resuming training from epoch {start_epoch}")
            print(f"Loaded {len(G_losses)} generator loss values")
            print(f"Loaded {len(D_losses)} discriminator loss values")
        else:
            print(f"Warning: Checkpoint file {resume_from} not found. Starting fresh training.")
    
    print("Starting WGAN-GP Training Loop...")
    print(f"Using device: {device}")
    print(f"Lambda GP: {lambda_gp}, n_critic: {n_critic}")
    print(f"Starting from epoch: {start_epoch}")
    
    for epoch in range(start_epoch, num_epochs):
        for i, data in enumerate(dataloader, 0):
            real_cpu = data.to(device)
            b_size = real_cpu.size(0)
            
            ############################
            # (1) Train Discriminator/Critic n_critic times
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
            errG = -torch.mean(fake_output)
            errG.backward()
            optimizerG.step()
            
            # Calculate metrics for logging
            D_x = torch.mean(real_output).item()
            D_G_z2 = torch.mean(fake_output).item()
            
            # Verbose logging every batch
            if i % 10 == 0:
                print(f'[{epoch:4d}/{num_epochs}][{i:3d}/{len(dataloader)}] '
                      f'Loss_D: {errD.item():8.4f} | Loss_G: {errG.item():8.4f} | '
                      f'D(x): {D_x:6.4f} | D(G(z)): {D_G_z2:6.4f} | '
                      f'GP: {gp.item():6.4f}')
            
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
        
        # Save comparison images every 10 epochs
        if (epoch + 1) % 10 == 0 and len(dataloader) > 0:
            with torch.no_grad():
                # Option 1: Generate fixed sample for progress tracking (same input each time)
                fake_fixed = netG(fixed_noise).detach().cpu()
                
                # Option 2: Generate completely new random samples each epoch
                random_noise = torch.randn(num_samples, nz, 1, 1, device=device)
                fake_random = netG(random_noise).detach().cpu()
                
                # Get a real sample for comparison
                real_sample = next(iter(dataloader))  # Get first batch
                real_sample = real_sample.cpu()
                
            # Save fixed sample (for tracking training progress on same input)
            save_comparison_image(fake_fixed, real_sample, output_dir, epoch, suffix="fixed")
            
            # Save random samples (for showing diversity and preventing "always same" issue)
            save_varied_samples(fake_random, output_dir, epoch)
        
        # Save model every 20 epochs (more frequent for better resumability)
        if (epoch + 1) % 20 == 0:
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
                'iters': iters,
            }, f"{model_dir}/checkpoint_epoch_{epoch:03d}.pth")
            print(f"Saved model checkpoint at epoch {epoch}")
    
    # Save final checkpoint
    print("Training completed! Saving final checkpoint...")
    model_dir = os.path.join(output_dir, "models")
    os.makedirs(model_dir, exist_ok=True)
    final_checkpoint_path = f"{model_dir}/final_checkpoint.pth"
    torch.save({
        'epoch': num_epochs - 1,
        'netG_state_dict': netG.state_dict(),
        'netD_state_dict': netD.state_dict(),
        'optimizerG_state_dict': optimizerG.state_dict(),
        'optimizerD_state_dict': optimizerD.state_dict(),
        'G_losses': G_losses,
        'D_losses': D_losses,
        'iters': iters,
        'training_complete': True,
    }, final_checkpoint_path)
    print(f"Final checkpoint saved to: {final_checkpoint_path}")
    
    return netG, netD, G_losses, D_losses

def load_trained_model(checkpoint_path, nz=100, nc=1, ngf=64, ndf=64, device=None):
    """
    Load a trained GAN model from checkpoint for inference or continued training.
    
    Args:
        checkpoint_path: Path to the checkpoint file
        nz: Size of noise vector (should match training)
        nc: Number of channels (should match training) 
        ngf: Generator feature map size (should match training)
        ndf: Discriminator feature map size (should match training)
        device: Device to load model on (cuda/cpu)
    
    Returns:
        tuple: (generator, discriminator, training_info)
    """
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Initialize models with same architecture
    netG = Generator(nz, ngf, nc).to(device)
    netD = Discriminator(nc, ndf).to(device)
    
    # Load checkpoint
    print(f"Loading trained model from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Load model weights
    netG.load_state_dict(checkpoint['netG_state_dict'])
    netD.load_state_dict(checkpoint['netD_state_dict'])
    
    # Set to evaluation mode
    netG.eval()
    netD.eval()
    
    # Extract training info
    training_info = {
        'epoch': checkpoint.get('epoch', 'unknown'),
        'G_losses': checkpoint.get('G_losses', []),
        'D_losses': checkpoint.get('D_losses', []),
        'iters': checkpoint.get('iters', 0),
        'training_complete': checkpoint.get('training_complete', False)
    }
    
    print(f"Loaded model from epoch {training_info['epoch']}")
    print(f"Training was {'complete' if training_info['training_complete'] else 'incomplete'}")
    
    return netG, netD, training_info

def generate_samples(generator, num_samples=4, nz=100, device=None, output_dir=None):
    """
    Generate samples using a trained generator.
    
    Args:
        generator: Trained generator model
        num_samples: Number of samples to generate
        nz: Size of noise vector
        device: Device to use for generation
        output_dir: Optional directory to save samples
    
    Returns:
        torch.Tensor: Generated samples
    """
    if device is None:
        device = next(generator.parameters()).device
    
    generator.eval()
    with torch.no_grad():
        noise = torch.randn(num_samples, nz, 1, 1, device=device)
        samples = generator(noise)
    
    if output_dir:
        from utils import save_individual_tif_images, save_individual_float32_tif_images
        os.makedirs(output_dir, exist_ok=True)
        save_individual_tif_images(samples, output_dir, "inference_sample")
        save_individual_float32_tif_images(samples, output_dir, "inference_sample_float32")
        print(f"Saved {num_samples} generated samples to {output_dir}")
    
    return samples