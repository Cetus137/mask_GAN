import torch.nn as nn
import torch

def weights_init(m):
    """Initialize weights according to DCGAN paper"""
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

class Generator(nn.Module):
    def __init__(self, nz, ngf, nc):
        super(Generator, self).__init__()
        
        # Initial dense layer for spatial structure
        self.initial = nn.Sequential(
            nn.ConvTranspose2d(nz, ngf * 16, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 16),
            nn.ReLU(True)
        )
        
        # Progressive upsampling with spatial attention
        self.layer1 = nn.Sequential(
            nn.ConvTranspose2d(ngf * 16, ngf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # Add spatial attention mechanism
            nn.Conv2d(ngf * 8, ngf * 8, 7, 1, 3, bias=False),  # Large kernel for spatial context
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True)
        )
        
        self.layer2 = nn.Sequential(
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            nn.Conv2d(ngf * 4, ngf * 4, 7, 1, 3, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True)
        )
        
        self.layer3 = nn.Sequential(
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            nn.Conv2d(ngf * 2, ngf * 2, 9, 1, 4, bias=False),  # Even larger kernel
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True)
        )
        
        self.layer4 = nn.Sequential(
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            nn.Conv2d(ngf, ngf, 9, 1, 4, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True)
        )
        
        self.layer5 = nn.Sequential(
            nn.ConvTranspose2d(ngf, ngf // 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf // 2),
            nn.ReLU(True),
            nn.Conv2d(ngf // 2, ngf // 2, 11, 1, 5, bias=False),  # Very large kernel for global context
            nn.BatchNorm2d(ngf // 2),
            nn.ReLU(True)
        )
        
        self.layer6 = nn.Sequential(
            nn.ConvTranspose2d(ngf // 2, ngf // 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf // 4),
            nn.ReLU(True),
            nn.Conv2d(ngf // 4, ngf // 4, 11, 1, 5, bias=False),
            nn.BatchNorm2d(ngf // 4),
            nn.ReLU(True)
        )
        
        # Multi-scale final processing
        self.final_large_scale = nn.Conv2d(ngf // 4, ngf // 8, 15, 1, 7, bias=False)  # Very large receptive field
        self.final_medium_scale = nn.Conv2d(ngf // 4, ngf // 8, 7, 1, 3, bias=False)
        self.final_small_scale = nn.Conv2d(ngf // 4, ngf // 8, 3, 1, 1, bias=False)
        
        self.final_combine = nn.Sequential(
            nn.BatchNorm2d(ngf // 8 * 3),  # 3 scales combined
            nn.ReLU(True),
            nn.Conv2d(ngf // 8 * 3, nc, 1, 1, 0, bias=False),  # 1x1 to combine features
            nn.Sigmoid()
        )

    def forward(self, input):
        x = self.initial(input)      # 4x4
        x = self.layer1(x)           # 8x8 with spatial context
        x = self.layer2(x)           # 16x16 with spatial context  
        x = self.layer3(x)           # 32x32 with large spatial context
        x = self.layer4(x)           # 64x64 with large spatial context
        x = self.layer5(x)           # 128x128 with very large spatial context
        x = self.layer6(x)           # 256x256 with very large spatial context
        
        # Multi-scale processing to capture different spatial scales
        large_scale = self.final_large_scale(x)    # Global/large structures
        medium_scale = self.final_medium_scale(x)  # Medium structures  
        small_scale = self.final_small_scale(x)    # Fine details
        
        # Combine all scales
        combined = torch.cat([large_scale, medium_scale, small_scale], dim=1)
        output = self.final_combine(combined)
        
        # Apply aggressive temperature scaling and hard thresholding
        temperature = 0.01  # Much lower temperature = much more binary
        output = torch.sigmoid((output - 0.5) / temperature)
        
        # Add progressive hard thresholding during training
        # This pushes values toward 0 or 1 more aggressively
        output = torch.where(output > 0.5, 
                           0.9 * output + 0.1,  # Push high values toward 1
                           0.9 * output)        # Push low values toward 0 
        return output

class Discriminator(nn.Module):
    def __init__(self, nc, ndf):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            # Input: nc x 256 x 256 -> ndf x 128 x 128
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # State size: ndf x 128 x 128 -> ndf*2 x 64 x 64
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # State size: ndf*2 x 64 x 64 -> ndf*4 x 32 x 32
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # State size: ndf*4 x 32 x 32 -> ndf*8 x 16 x 16
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # State size: ndf*8 x 16 x 16 -> ndf*16 x 8 x 8
            nn.Conv2d(ndf * 8, ndf * 16, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 16),
            nn.LeakyReLU(0.2, inplace=True),
            # State size: ndf*16 x 8 x 8 -> ndf*16 x 4 x 4
            nn.Conv2d(ndf * 16, ndf * 16, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 16),
            nn.LeakyReLU(0.2, inplace=True),
            # State size: ndf*16 x 4 x 4 -> 1 x 1 x 1 (no sigmoid for WGAN-GP)
            nn.Conv2d(ndf * 16, 1, 4, 1, 0, bias=False)
        )

    def forward(self, input):
        return self.main(input)