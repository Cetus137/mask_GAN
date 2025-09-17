#!/usr/bin/env python3
"""
Inference script for generating samples from a trained Cell Continuous Value GAN.
"""

import argparse
import torch
import os
from train import load_trained_model, generate_samples

def main():
    parser = argparse.ArgumentParser(description="Generate samples from trained GAN model.")
    parser.add_argument('--checkpoint', type=str, required=True, 
                       help='Path to the checkpoint file (.pth)')
    parser.add_argument('--output_dir', type=str, default='../inference_output', 
                       help='Directory to save generated samples')
    parser.add_argument('--num_samples', type=int, default=8, 
                       help='Number of samples to generate')
    parser.add_argument('--nz', type=int, default=100, 
                       help='Size of noise vector (must match training)')
    parser.add_argument('--nc', type=int, default=1, 
                       help='Number of channels (must match training)')
    parser.add_argument('--ngf', type=int, default=64, 
                       help='Generator feature map size (must match training)')
    parser.add_argument('--ndf', type=int, default=64, 
                       help='Discriminator feature map size (must match training)')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("Loading trained GAN model...")
    
    # Load the trained model
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    generator, discriminator, training_info = load_trained_model(
        args.checkpoint, 
        nz=args.nz, 
        nc=args.nc, 
        ngf=args.ngf, 
        ndf=args.ndf, 
        device=device
    )
    
    print(f"\nModel Info:")
    print(f"- Trained for {training_info['epoch']} epochs")
    print(f"- Training complete: {training_info['training_complete']}")
    print(f"- Total G losses recorded: {len(training_info['G_losses'])}")
    print(f"- Total D losses recorded: {len(training_info['D_losses'])}")
    
    print(f"\nGenerating {args.num_samples} samples...")
    
    # Generate samples
    samples = generate_samples(
        generator, 
        num_samples=args.num_samples, 
        nz=args.nz, 
        device=device, 
        output_dir=args.output_dir
    )
    
    print(f"\nGeneration complete!")
    print(f"Generated samples saved to: {args.output_dir}")
    print(f"Sample shape: {samples.shape}")
    print(f"Sample value range: [{samples.min().item():.4f}, {samples.max().item():.4f}]")

if __name__ == "__main__":
    main()
