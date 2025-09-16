import torch
import torchvision.utils as vutils
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

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

def save_image_grid(img_list, output_dir):
    """
    Saves a grid of images as grayscale TIF.
    """
    if img_list:
        save_grayscale_tif(img_list[-1], f"{output_dir}/gan_images.tif")

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