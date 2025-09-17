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
        # Load as numpy array to preserve continuous values
        image = Image.open(img_path)
        image_array = np.array(image, dtype=np.float32)
        
        # Handle different input formats and normalize to [0,1] for consistent processing
        if image_array.dtype == np.uint16:
            # Normalize uint16 to 0-1 range
            if image_array.max() > 1:
                image_array = image_array.astype(np.float32) / image_array.max()
            else:
                image_array = image_array.astype(np.float32)
        elif image_array.dtype == np.uint8:
            # Normalize uint8 to 0-1 range
            if image_array.max() > 1:
                image_array = image_array.astype(np.float32) / 255.0
            else:
                image_array = image_array.astype(np.float32)
        elif image_array.dtype in [np.float32, np.float64]:
            # For float inputs, normalize to [0,1] if values are outside this range
            image_array = image_array.astype(np.float32)
            if image_array.max() > 1.0 or image_array.min() < 0.0:
                # Normalize to [0,1] range
                image_array = (image_array - image_array.min()) / (image_array.max() - image_array.min())
        
        # Convert to PIL Image in 'F' mode (32-bit float)
        image = Image.fromarray(image_array, mode='F')
        
        if self.transform:
            image = self.transform(image)
        return image

def get_data_loader(data_dir, batch_size, image_size):
    def normalize_to_tanh_range(x):
        # Convert continuous values to [-1,1] range for Tanh generator output
        # Assumes input is normalized to [0,1] range after ToTensor()
        return x * 2.0 - 1.0
    
    transform = transforms.Compose([
        transforms.Resize((256, 256)),  # Always resize to 256x256
        transforms.ToTensor(),  # Converts to [0,1] range and normalizes
        transforms.Lambda(normalize_to_tanh_range)  # Normalize continuous values to [-1,1]
    ])
    dataset = CellMaskDataset(root_dir=data_dir, transform=transform)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return data_loader