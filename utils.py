import torch
import torchvision.utils as vutils
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

def save_individual_tif_images(tensor, output_dir, prefix="generated"):
    """Save individual 256x256 grayscale TIF files"""
    for i, img in enumerate(tensor):
        # Convert single image tensor to numpy
        img_np = img.squeeze(0).cpu().numpy()  # Remove channel dimension if single channel
        
        # Denormalize from [-1, 1] to [0, 255]
        img_np = ((img_np + 1) * 127.5).clip(0, 255).astype(np.uint8)
        
        # Save as grayscale TIF
        im = Image.fromarray(img_np, mode='L')
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