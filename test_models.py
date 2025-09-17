#!/usr/bin/env python3
"""
Test script to verify that the models work with 256x256 images.
"""

import torch
from models import Generator, Discriminator

def test_models():
    """Test the Generator and Discriminator with 256x256 images."""
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Model parameters
    nz = 100  # Size of latent vector
    nc = 1    # Number of channels (grayscale)
    ngf = 64  # Generator feature map size
    ndf = 64  # Discriminator feature map size
    
    # Initialize models
    netG = Generator(nz, ngf, nc).to(device)
    netD = Discriminator(nc, ndf).to(device)
    
    print("Models initialized successfully!")
    
    # Test Generator
    print("\nTesting Generator...")
    noise = torch.randn(4, nz, 1, 1, device=device)
    fake_images = netG(noise)
    print(f"Generator output shape: {fake_images.shape}")
    assert fake_images.shape == (4, 1, 256, 256), f"Expected (4, 1, 256, 256), got {fake_images.shape}"
    
    # Test continuous value range [-1,1]
    min_val = torch.min(fake_images).item()
    max_val = torch.max(fake_images).item()
    print(f"Generator output range: [{min_val:.4f}, {max_val:.4f}]")
    assert min_val >= -1.0 and max_val <= 1.0, f"Expected range [-1,1], got [{min_val}, {max_val}]"
    print("✓ Generator produces 256x256 continuous value maps in [-1,1] range")
    
    # Test Discriminator
    print("\nTesting Discriminator...")
    # Create realistic probability maps for testing
    real_images = torch.rand(4, 1, 256, 256, device=device)  # Random probabilities in [0,1]
    output = netD(real_images)
    print(f"Discriminator output shape: {output.shape}")
    assert len(output.shape) == 2 and output.shape[0] == 4, f"Expected batch dimension 4, got {output.shape}"
    print("✓ Discriminator processes 256x256 probability maps")
    
    # Test with fake images
    fake_output = netD(fake_images)
    print(f"Discriminator output for fake images shape: {fake_output.shape}")
    assert fake_output.shape == output.shape, "Discriminator output shapes should match"
    print("✓ Discriminator works with generated images")
    
    print("\n✅ All tests passed! Models are ready for 256x256 probability mask training.")

if __name__ == "__main__":
    test_models()
