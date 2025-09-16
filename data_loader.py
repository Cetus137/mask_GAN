import os
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class CellMaskDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_files = [f for f in os.listdir(root_dir) if os.path.isfile(os.path.join(root_dir, f))]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.image_files[idx])
        # Load as numpy array to preserve uint16 integer values
        image = Image.open(img_path)
        image_array = np.array(image)
        
        # Convert back to PIL for transforms (ensure proper mode)
        if image_array.dtype == np.uint16:
            # Normalize uint16 to 0-255 range for PIL
            image_array = (image_array / image_array.max() * 255).astype(np.uint8)
        
        image = Image.fromarray(image_array, mode='L')
        
        if self.transform:
            image = self.transform(image)
        return image

def get_data_loader(data_dir, batch_size, image_size):
    def binarize_and_smooth(x):
        # Convert integer mask to binary (0 for background, 1 for any cell)
        binary = torch.where(x > 0.0, 1.0, 0.0)
        # Apply Gaussian smoothing
        binary = binary.unsqueeze(0)  # Add batch dim for conv
        smoothed = F.gaussian_blur(binary, kernel_size=5, sigma=1.0)
        return smoothed.squeeze(0)  # Remove batch dim
    
    transform = transforms.Compose([
        transforms.Resize((256, 256)),  # Always resize to 256x256
        transforms.ToTensor(),  # Converts to [0,1] range and normalizes
        transforms.Lambda(binarize_and_smooth)  # Binarize integer masks then smooth
    ])
    dataset = CellMaskDataset(root_dir=data_dir, transform=transform)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return data_loader