import argparse
from train import train

def main():
    """
    Main function to run the GAN training process.
    """
    parser = argparse.ArgumentParser(description="Train a GAN to generate cell masks.")
    parser.add_argument('--data_dir', type=str, default='../data', help='Path to the dataset directory.')
    parser.add_argument('--output_dir', type=str, default='../output', help='Path to the output directory.')
    parser.add_argument('--nz', type=int, default=100, help='Size of the latent z vector.')
    parser.add_argument('--nc', type=int, default=1, help='Number of channels in the training images.')
    parser.add_argument('--ngf', type=int, default=64, help='Size of feature maps in the generator.')
    parser.add_argument('--ndf', type=int, default=64, help='Size of feature maps in the discriminator.')
    parser.add_argument('--num_epochs', type=int, default=200, help='Number of training epochs.')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size during training.')
    parser.add_argument('--image_size', type=int, default=256, help='Spatial size of training images.')
    parser.add_argument('--lr', type=float, default=0.0002, help='Learning rate for optimizers.')
    parser.add_argument('--beta1', type=float, default=0.5, help='Beta1 hyperparameter for Adam optimizers.')
    parser.add_argument('--resume_from', type=str, default=None, help='Path to checkpoint file to resume training from.')
    
    args = parser.parse_args()
    
    print("Starting GAN training process...")
    train(
        data_dir=args.data_dir,
        nz=args.nz,
        nc=args.nc,
        ngf=args.ngf,
        ndf=args.ndf,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        image_size=args.image_size,
        lr=args.lr,
        beta1=args.beta1,
        output_dir=args.output_dir,
        resume_from=args.resume_from
    )
    print("Training finished.")

if __name__ == "__main__":
    main()