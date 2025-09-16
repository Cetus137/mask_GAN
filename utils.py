import torch
import torchvision.utils as vutils
import matplotlib.pyplot as plt
import numpy as np

def save_image_grid(img_list, output_dir):
    """
    Saves a grid of images.
    """
    plt.figure(figsize=(8,8))
    plt.axis("off")
    plt.title("Generated Images")
    plt.imshow(np.transpose(vutils.make_grid(img_list[-1], padding=2, normalize=True).cpu(),(1,2,0)))
    plt.savefig(f"{output_dir}/gan_images.png")

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