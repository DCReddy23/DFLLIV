"""Visualization utilities for low-light image enhancement.

Provides functions for creating side-by-side comparisons, enhancement grids,
and training curve plots.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from typing import List, Optional, Union, Tuple
import os


def tensor_to_image(tensor: torch.Tensor) -> np.ndarray:
    """Convert a PyTorch tensor to a numpy image array.
    
    Args:
        tensor: Image tensor of shape (C, H, W) or (B, C, H, W)
    
    Returns:
        Numpy array of shape (H, W, C) or (B, H, W, C) in range [0, 255]
    """
    # Remove batch dimension if single image
    if tensor.dim() == 4 and tensor.shape[0] == 1:
        tensor = tensor.squeeze(0)
    
    # Convert to numpy and transpose
    img = tensor.detach().cpu().numpy()
    
    if img.ndim == 3:  # (C, H, W) -> (H, W, C)
        img = np.transpose(img, (1, 2, 0))
    elif img.ndim == 4:  # (B, C, H, W) -> (B, H, W, C)
        img = np.transpose(img, (0, 2, 3, 1))
    
    # Clip and scale to [0, 255]
    img = np.clip(img, 0, 1)
    img = (img * 255).astype(np.uint8)
    
    return img


def save_comparison(
    low_light: Union[torch.Tensor, np.ndarray],
    enhanced: Union[torch.Tensor, np.ndarray],
    ground_truth: Optional[Union[torch.Tensor, np.ndarray]] = None,
    save_path: str = 'comparison.png',
    titles: Optional[List[str]] = None
):
    """Save a side-by-side comparison of images.
    
    Args:
        low_light: Low-light input image
        enhanced: Enhanced output image
        ground_truth: Optional ground truth image
        save_path: Path to save the comparison image
        titles: Optional list of titles for each image
    """
    # Convert tensors to images
    if isinstance(low_light, torch.Tensor):
        low_light = tensor_to_image(low_light)
    if isinstance(enhanced, torch.Tensor):
        enhanced = tensor_to_image(enhanced)
    if ground_truth is not None and isinstance(ground_truth, torch.Tensor):
        ground_truth = tensor_to_image(ground_truth)
    
    # Setup figure
    num_images = 3 if ground_truth is not None else 2
    fig, axes = plt.subplots(1, num_images, figsize=(6 * num_images, 6))
    
    if num_images == 2:
        axes = [axes[0], axes[1]]
    
    # Default titles
    if titles is None:
        titles = ['Low-Light Input', 'Enhanced Output']
        if ground_truth is not None:
            titles.append('Ground Truth')
    
    # Plot images
    axes[0].imshow(low_light)
    axes[0].set_title(titles[0], fontsize=14)
    axes[0].axis('off')
    
    axes[1].imshow(enhanced)
    axes[1].set_title(titles[1], fontsize=14)
    axes[1].axis('off')
    
    if ground_truth is not None:
        axes[2].imshow(ground_truth)
        axes[2].set_title(titles[2], fontsize=14)
        axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved comparison to {save_path}")


def create_image_grid(
    images: List[Union[torch.Tensor, np.ndarray]],
    nrow: int = 4,
    padding: int = 2,
    save_path: Optional[str] = None
) -> np.ndarray:
    """Create a grid of images.
    
    Args:
        images: List of images (tensors or numpy arrays)
        nrow: Number of images per row
        padding: Padding between images in pixels
        save_path: Optional path to save the grid
    
    Returns:
        Grid image as numpy array
    """
    # Convert all to numpy
    np_images = []
    for img in images:
        if isinstance(img, torch.Tensor):
            img = tensor_to_image(img)
        np_images.append(img)
    
    # Get dimensions
    h, w = np_images[0].shape[:2]
    num_images = len(np_images)
    ncol = (num_images + nrow - 1) // nrow
    
    # Create grid
    grid_h = ncol * h + (ncol + 1) * padding
    grid_w = nrow * w + (nrow + 1) * padding
    grid = np.ones((grid_h, grid_w, 3), dtype=np.uint8) * 255
    
    # Fill grid
    for idx, img in enumerate(np_images):
        row = idx // nrow
        col = idx % nrow
        
        y_start = row * h + (row + 1) * padding
        x_start = col * w + (col + 1) * padding
        
        # Handle grayscale images
        if img.ndim == 2:
            img = np.stack([img] * 3, axis=-1)
        
        grid[y_start:y_start + h, x_start:x_start + w] = img
    
    # Save if path provided
    if save_path:
        Image.fromarray(grid).save(save_path)
        print(f"Saved grid to {save_path}")
    
    return grid


def plot_training_curves(
    metrics: dict,
    save_path: str = 'training_curves.png',
    title: str = 'Training Metrics'
):
    """Plot training curves.
    
    Args:
        metrics: Dictionary with metric names as keys and lists of values
        save_path: Path to save the plot
        title: Title for the plot
    """
    num_metrics = len(metrics)
    fig, axes = plt.subplots(1, num_metrics, figsize=(6 * num_metrics, 5))
    
    if num_metrics == 1:
        axes = [axes]
    
    for idx, (name, values) in enumerate(metrics.items()):
        axes[idx].plot(values, linewidth=2)
        axes[idx].set_xlabel('Epoch', fontsize=12)
        axes[idx].set_ylabel(name.upper(), fontsize=12)
        axes[idx].set_title(name.upper(), fontsize=14)
        axes[idx].grid(True, alpha=0.3)
    
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved training curves to {save_path}")


def visualize_enhancement_batch(
    low_light_batch: torch.Tensor,
    enhanced_batch: torch.Tensor,
    ground_truth_batch: Optional[torch.Tensor] = None,
    save_dir: str = 'visualizations',
    prefix: str = 'sample'
):
    """Visualize a batch of enhancements.
    
    Args:
        low_light_batch: Batch of low-light images (B, 3, H, W)
        enhanced_batch: Batch of enhanced images (B, 3, H, W)
        ground_truth_batch: Optional batch of ground truth images (B, 3, H, W)
        save_dir: Directory to save visualizations
        prefix: Prefix for saved files
    """
    os.makedirs(save_dir, exist_ok=True)
    
    batch_size = low_light_batch.shape[0]
    
    for i in range(batch_size):
        save_path = os.path.join(save_dir, f'{prefix}_{i}.png')
        
        gt = ground_truth_batch[i] if ground_truth_batch is not None else None
        
        save_comparison(
            low_light_batch[i],
            enhanced_batch[i],
            gt,
            save_path=save_path
        )


def create_enhancement_comparison_grid(
    low_light_images: List[torch.Tensor],
    enhanced_images: List[torch.Tensor],
    ground_truth_images: Optional[List[torch.Tensor]] = None,
    save_path: str = 'enhancement_grid.png',
    max_images: int = 8
):
    """Create a grid showing multiple enhancement results.
    
    Args:
        low_light_images: List of low-light images
        enhanced_images: List of enhanced images
        ground_truth_images: Optional list of ground truth images
        save_path: Path to save the grid
        max_images: Maximum number of images to include
    """
    num_images = min(len(low_light_images), max_images)
    num_cols = 3 if ground_truth_images is not None else 2
    
    fig, axes = plt.subplots(num_images, num_cols, figsize=(6 * num_cols, 6 * num_images))
    
    if num_images == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(num_images):
        # Low-light
        axes[i, 0].imshow(tensor_to_image(low_light_images[i]))
        if i == 0:
            axes[i, 0].set_title('Low-Light Input', fontsize=14)
        axes[i, 0].axis('off')
        
        # Enhanced
        axes[i, 1].imshow(tensor_to_image(enhanced_images[i]))
        if i == 0:
            axes[i, 1].set_title('Enhanced Output', fontsize=14)
        axes[i, 1].axis('off')
        
        # Ground truth
        if ground_truth_images is not None:
            axes[i, 2].imshow(tensor_to_image(ground_truth_images[i]))
            if i == 0:
                axes[i, 2].set_title('Ground Truth', fontsize=14)
            axes[i, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved enhancement grid to {save_path}")


if __name__ == "__main__":
    # Test visualization functions
    print("Testing visualization utilities...")
    
    # Create dummy images
    low_light = torch.rand(3, 256, 256) * 0.3  # Dark image
    enhanced = torch.rand(3, 256, 256)  # Enhanced image
    ground_truth = torch.rand(3, 256, 256)  # Ground truth
    
    # Test comparison
    save_comparison(low_light, enhanced, ground_truth, save_path='test_comparison.png')
    
    # Test grid
    images = [torch.rand(3, 128, 128) for _ in range(8)]
    create_image_grid(images, nrow=4, save_path='test_grid.png')
    
    # Test training curves
    metrics = {
        'loss': [1.0, 0.8, 0.6, 0.4, 0.3, 0.2],
        'psnr': [20, 22, 24, 26, 27, 28]
    }
    plot_training_curves(metrics, save_path='test_curves.png')
    
    print("Visualization tests completed!")
