import torch
import torchvision.utils as vutils
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

def save_individual_tif_images(tensor, output_dir, prefix="generated"):
    """Save individual 256x256 grayscale TIF files with continuous values"""
    for i, img in enumerate(tensor):
        # Convert single image tensor to numpy
        img_np = img.squeeze(0).cpu().numpy()  # Remove channel dimension if single channel
        
        # Convert from [-1, 1] range to [0, 255] for 8-bit TIF
        img_np = ((img_np + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
        
        # Save as grayscale TIF
        im = Image.fromarray(img_np, mode='L')
        filename = f"{output_dir}/{prefix}_img_{i:02d}.tif"
        im.save(filename)

def save_individual_float32_tif_images(tensor, output_dir, prefix="generated_float32"):
    """Save individual 256x256 32-bit float TIF files preserving full continuous value precision"""
    for i, img in enumerate(tensor):
        # Convert single image tensor to numpy
        img_np = img.squeeze(0).cpu().numpy().astype(np.float32)  # Remove channel dimension
        
        # Keep data in [-1, 1] range as 32-bit float values (no conversion)
        img_np = np.clip(img_np, -1.0, 1.0)
        
        # Save as 32-bit float TIF
        im = Image.fromarray(img_np, mode='F')
        filename = f"{output_dir}/{prefix}_img_{i:02d}.tif"
        im.save(filename)

def save_individual_normalized_float32_tif_images(tensor, output_dir, prefix="generated_normalized"):
    """Save individual 256x256 32-bit float TIF files normalized to [0,1] range"""
    for i, img in enumerate(tensor):
        # Convert single image tensor to numpy
        img_np = img.squeeze(0).cpu().numpy().astype(np.float32)  # Remove channel dimension
        
        # Convert from [-1, 1] to [0, 1] range for easier interpretation
        img_np = (img_np + 1.0) / 2.0
        img_np = np.clip(img_np, 0.0, 1.0)
        
        # Save as 32-bit float TIF
        im = Image.fromarray(img_np, mode='F')
        filename = f"{output_dir}/{prefix}_img_{i:02d}.tif"
        im.save(filename)

def save_image_grid(img_list, output_dir):
    """
    Saves individual images as grayscale TIFs.
    """
    if img_list:
        save_individual_tif_images(img_list[-1], output_dir, "gan_images")

def save_loss_plot(G_losses, D_losses, output_dir):
    """
    Saves a plot of generator and discriminator losses.
    """
    plt.figure(figsize=(10,5))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(G_losses,label="G")
    plt.plot(D_losses,label="D")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(f"{output_dir}/loss_curve.png")

def save_checkpoint(epoch, netG, netD, optimizerG, optimizerD, output_dir):
    """
    Saves a model checkpoint.
    """
    torch.save({
            'epoch': epoch,
            'netG_state_dict': netG.state_dict(),
            'netD_state_dict': netD.state_dict(),
            'optimizerG_state_dict': optimizerG.state_dict(),
            'optimizerD_state_dict': optimizerD.state_dict(),
            }, f"{output_dir}/checkpoint.pth")