import torch
import torch.nn as nn

def weights_init(m):
    """Initialize weights for DCGAN-style models"""
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

class Generator(nn.Module):
    def __init__(self, nz, ngf, nc):
        super(Generator, self).__init__()
        # Much simpler generator with ~1-2M parameters
        # Base channel count reduced to 64
        
        self.main = nn.Sequential(
            # Input: nz x 1 x 1 -> ngf*4 x 4 x 4
            nn.ConvTranspose2d(nz, ngf * 4, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            
            # State: ngf*4 x 4 x 4 -> ngf*2 x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            
            # State: ngf*2 x 8 x 8 -> ngf x 16 x 16
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            
            # State: ngf x 16 x 16 -> ngf//2 x 32 x 32
            nn.ConvTranspose2d(ngf, ngf // 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf // 2),
            nn.ReLU(True),
            
            # State: ngf//2 x 32 x 32 -> ngf//4 x 64 x 64
            nn.ConvTranspose2d(ngf // 2, ngf // 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf // 4),
            nn.ReLU(True),
            
            # State: ngf//4 x 64 x 64 -> ngf//8 x 128 x 128
            nn.ConvTranspose2d(ngf // 4, ngf // 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf // 8),
            nn.ReLU(True),
            
            # State: ngf//8 x 128 x 128 -> nc x 256 x 256
            nn.ConvTranspose2d(ngf // 8, nc, 4, 2, 1, bias=False),
            nn.Tanh()  # Output in [-1, 1] range for WGAN-GP
        )

    def forward(self, input):
        return self.main(input)

class Discriminator(nn.Module):
    def __init__(self, nc, ndf):
        super(Discriminator, self).__init__()
        # Much smaller discriminator to balance with 1M parameter generator
        # Target: ~500k-1M parameters to match generator
        
        self.main = nn.Sequential(
            # Input: nc x 256 x 256 -> ndf//2 x 128 x 128
            nn.Conv2d(nc, ndf // 2, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            
            # State: ndf//2 x 128 x 128 -> ndf x 64 x 64
            nn.Conv2d(ndf // 2, ndf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf),
            nn.LeakyReLU(0.2, inplace=True),
            
            # State: ndf x 64 x 64 -> ndf*2 x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            
            # State: ndf*2 x 32 x 32 -> ndf*2 x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            
            # State: ndf*2 x 16 x 16 -> ndf*2 x 8 x 8
            nn.Conv2d(ndf * 2, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            
            # State: ndf*2 x 8 x 8 -> 1 x 1 x 1
            nn.Conv2d(ndf * 2, 1, 8, 1, 0, bias=False)
            # No sigmoid for WGAN-GP (critic outputs raw scores)
        )

    def forward(self, input):
        return self.main(input)
