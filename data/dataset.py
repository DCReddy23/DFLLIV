"""Dataset loaders for low-light image enhancement.

Provides PyTorch Dataset classes for:
- LOL (Low-Light) dataset
- LOL-v2 dataset
- Synthetic low-light pair generation
"""

import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import torchvision.transforms.functional as TF
from typing import Tuple, Optional, Callable
import random


class LOLDataset(Dataset):
    """LOL (Low-Light) Dataset loader.
    
    Dataset of paired low-light and normal-light images.
    
    Args:
        root_dir: Root directory containing 'low' and 'high' subdirectories
        split: 'train' or 'val' (default: 'train')
        crop_size: Size for random cropping (default: 256)
        augment: Whether to apply data augmentation (default: True)
        normalize: Whether to normalize images to [-1, 1] (default: False, keeps [0, 1])
    """
    
    def __init__(
        self,
        root_dir: str,
        split: str = 'train',
        crop_size: int = 256,
        augment: bool = True,
        normalize: bool = False
    ):
        self.root_dir = root_dir
        self.split = split
        self.crop_size = crop_size
        self.augment = augment and split == 'train'
        self.normalize = normalize
        
        # Paths to low and high light images
        self.low_dir = os.path.join(root_dir, 'low')
        self.high_dir = os.path.join(root_dir, 'high')
        
        if not os.path.exists(self.low_dir):
            raise ValueError(f"Low-light directory not found: {self.low_dir}")
        if not os.path.exists(self.high_dir):
            raise ValueError(f"Normal-light directory not found: {self.high_dir}")
        
        # Get list of image files
        self.low_images = sorted([
            f for f in os.listdir(self.low_dir)
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))
        ])
        
        if len(self.low_images) == 0:
            raise ValueError(f"No images found in {self.low_dir}")
        
        print(f"Loaded {len(self.low_images)} {split} image pairs from {root_dir}")
    
    def __len__(self) -> int:
        return len(self.low_images)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get a low-light and normal-light image pair.
        
        Args:
            idx: Index of the image pair
        
        Returns:
            Tuple of (low_light_image, normal_light_image) as tensors
        """
        # Load images
        low_name = self.low_images[idx]
        low_path = os.path.join(self.low_dir, low_name)
        
        # Try to find corresponding high-light image
        # LOL dataset sometimes has different naming conventions
        high_name = low_name
        high_path = os.path.join(self.high_dir, high_name)
        
        # If not found, try without suffix or with different suffix
        if not os.path.exists(high_path):
            base_name = os.path.splitext(low_name)[0]
            # Try common variations
            for ext in ['.png', '.jpg', '.jpeg', '.bmp']:
                high_path = os.path.join(self.high_dir, base_name + ext)
                if os.path.exists(high_path):
                    break
        
        if not os.path.exists(high_path):
            raise ValueError(f"Corresponding high-light image not found for {low_name}")
        
        # Load as PIL images
        low_img = Image.open(low_path).convert('RGB')
        high_img = Image.open(high_path).convert('RGB')
        
        # Apply transforms
        low_img, high_img = self._transform(low_img, high_img)
        
        return low_img, high_img
    
    def _transform(self, low_img: Image.Image, high_img: Image.Image) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply transformations to image pair.
        
        Args:
            low_img: Low-light PIL image
            high_img: Normal-light PIL image
        
        Returns:
            Transformed images as tensors
        """
        # Convert to tensor
        low_tensor = TF.to_tensor(low_img)
        high_tensor = TF.to_tensor(high_img)
        
        # Ensure same size
        if low_tensor.shape != high_tensor.shape:
            # Resize to match
            h, w = min(low_tensor.shape[1], high_tensor.shape[1]), min(low_tensor.shape[2], high_tensor.shape[2])
            low_tensor = TF.resize(low_tensor, [h, w])
            high_tensor = TF.resize(high_tensor, [h, w])
        
        # Random crop if augmenting and image is larger than crop size
        if self.augment:
            # Get crop parameters
            _, h, w = low_tensor.shape
            if h > self.crop_size and w > self.crop_size:
                i = random.randint(0, h - self.crop_size)
                j = random.randint(0, w - self.crop_size)
                low_tensor = TF.crop(low_tensor, i, j, self.crop_size, self.crop_size)
                high_tensor = TF.crop(high_tensor, i, j, self.crop_size, self.crop_size)
            
            # Random horizontal flip
            if random.random() > 0.5:
                low_tensor = TF.hflip(low_tensor)
                high_tensor = TF.hflip(high_tensor)
            
            # Random vertical flip
            if random.random() > 0.5:
                low_tensor = TF.vflip(low_tensor)
                high_tensor = TF.vflip(high_tensor)
            
            # Random rotation (90, 180, 270 degrees)
            if random.random() > 0.5:
                angle = random.choice([90, 180, 270])
                low_tensor = TF.rotate(low_tensor, angle)
                high_tensor = TF.rotate(high_tensor, angle)
        else:
            # Center crop for validation
            _, h, w = low_tensor.shape
            if h > self.crop_size or w > self.crop_size:
                low_tensor = TF.center_crop(low_tensor, self.crop_size)
                high_tensor = TF.center_crop(high_tensor, self.crop_size)
        
        # Normalize to [-1, 1] if requested
        if self.normalize:
            low_tensor = low_tensor * 2.0 - 1.0
            high_tensor = high_tensor * 2.0 - 1.0
        
        return low_tensor, high_tensor


class SyntheticLowLightDataset(Dataset):
    """Synthetic low-light dataset generator.
    
    Generates low-light images from normal-light images using:
    - Random gamma correction (gamma ∈ [2.0, 5.0])
    - Brightness reduction
    - Optional Gaussian/Poisson noise injection
    
    Args:
        image_dir: Directory containing normal-light images
        crop_size: Size for random cropping (default: 256)
        gamma_range: Range for gamma correction (default: (2.0, 5.0))
        brightness_range: Range for brightness reduction (default: (0.3, 0.7))
        add_noise: Whether to add noise (default: True)
        noise_std: Standard deviation of Gaussian noise (default: 0.02)
    """
    
    def __init__(
        self,
        image_dir: str,
        crop_size: int = 256,
        gamma_range: Tuple[float, float] = (2.0, 5.0),
        brightness_range: Tuple[float, float] = (0.3, 0.7),
        add_noise: bool = True,
        noise_std: float = 0.02
    ):
        self.image_dir = image_dir
        self.crop_size = crop_size
        self.gamma_range = gamma_range
        self.brightness_range = brightness_range
        self.add_noise = add_noise
        self.noise_std = noise_std
        
        # Get list of images
        self.images = sorted([
            f for f in os.listdir(image_dir)
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))
        ])
        
        if len(self.images) == 0:
            raise ValueError(f"No images found in {image_dir}")
        
        print(f"Loaded {len(self.images)} images for synthetic low-light generation")
    
    def __len__(self) -> int:
        return len(self.images)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate a synthetic low-light / normal-light pair.
        
        Args:
            idx: Index of the image
        
        Returns:
            Tuple of (low_light_image, normal_light_image) as tensors
        """
        # Load image
        img_path = os.path.join(self.image_dir, self.images[idx])
        img = Image.open(img_path).convert('RGB')
        
        # Convert to tensor and crop
        img_tensor = TF.to_tensor(img)
        
        # Random crop
        _, h, w = img_tensor.shape
        if h > self.crop_size and w > self.crop_size:
            i = random.randint(0, h - self.crop_size)
            j = random.randint(0, w - self.crop_size)
            img_tensor = TF.crop(img_tensor, i, j, self.crop_size, self.crop_size)
        else:
            img_tensor = TF.resize(img_tensor, [self.crop_size, self.crop_size])
        
        # Random horizontal flip
        if random.random() > 0.5:
            img_tensor = TF.hflip(img_tensor)
        
        # Generate low-light version
        low_light = self._generate_low_light(img_tensor.clone())
        
        return low_light, img_tensor
    
    def _generate_low_light(self, img: torch.Tensor) -> torch.Tensor:
        """Generate low-light version of an image.
        
        Args:
            img: Normal-light image tensor of shape (3, H, W) in range [0, 1]
        
        Returns:
            Low-light image tensor
        """
        # Random gamma correction
        gamma = random.uniform(*self.gamma_range)
        low_light = torch.pow(img, gamma)
        
        # Random brightness reduction
        brightness_factor = random.uniform(*self.brightness_range)
        low_light = low_light * brightness_factor
        
        # Add noise
        if self.add_noise:
            noise = torch.randn_like(low_light) * self.noise_std
            low_light = low_light + noise
        
        # Clip to valid range
        low_light = torch.clamp(low_light, 0.0, 1.0)
        
        return low_light


class CoordinateDatasetWrapper(Dataset):
    """Wrapper that adds coordinate grids to dataset samples.
    
    Wraps another dataset and adds normalized coordinate grids for
    neural field training.
    
    Args:
        base_dataset: Base dataset returning (low_img, high_img) pairs
        image_size: Size of images (assumes square images)
    """
    
    def __init__(self, base_dataset: Dataset, image_size: int = 256):
        self.base_dataset = base_dataset
        self.image_size = image_size
        
        # Pre-compute coordinate grid
        from models.coord_encoder import create_coordinate_grid
        self.coord_grid = create_coordinate_grid(image_size, image_size)
    
    def __len__(self) -> int:
        return len(self.base_dataset)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get sample with coordinates.
        
        Args:
            idx: Index
        
        Returns:
            Tuple of (low_light_image, normal_light_image, coordinates)
        """
        low_img, high_img = self.base_dataset[idx]
        
        return low_img, high_img, self.coord_grid


def get_dataloader(
    dataset_type: str,
    root_dir: str,
    batch_size: int = 8,
    split: str = 'train',
    num_workers: int = 4,
    **kwargs
) -> torch.utils.data.DataLoader:
    """Create a dataloader for the specified dataset.
    
    Args:
        dataset_type: Type of dataset ('LOL', 'LOL-v2', or 'synthetic')
        root_dir: Root directory of the dataset
        batch_size: Batch size
        split: 'train' or 'val'
        num_workers: Number of data loading workers
        **kwargs: Additional arguments for the dataset
    
    Returns:
        DataLoader instance
    """
    if dataset_type.upper() == 'LOL' or dataset_type.upper() == 'LOL-V2':
        dataset = LOLDataset(root_dir, split=split, **kwargs)
    elif dataset_type.lower() == 'synthetic':
        dataset = SyntheticLowLightDataset(root_dir, **kwargs)
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")
    
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == 'train'),
        num_workers=num_workers,
        pin_memory=True
    )
    
    return dataloader


if __name__ == "__main__":
    # Test dataset loading
    print("Testing LOLDataset...")
    print("Note: This test requires the LOL dataset to be downloaded.")
    print("If you see errors, make sure to run data/download_lol.sh first.")
    
    # Uncomment to test if dataset is available
    # dataset = LOLDataset('data/LOL/our485', split='train')
    # print(f"Dataset size: {len(dataset)}")
    # low, high = dataset[0]
    # print(f"Low-light shape: {low.shape}, range: [{low.min():.3f}, {low.max():.3f}]")
    # print(f"Normal-light shape: {high.shape}, range: [{high.min():.3f}, {high.max():.3f}]")
