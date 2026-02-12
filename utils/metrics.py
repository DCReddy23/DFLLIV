"""Metrics for evaluating image quality.

Provides PSNR, SSIM, and LPIPS metrics for low-light image enhancement evaluation.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Union, Tuple
import lpips
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def calculate_psnr(
    img1: Union[torch.Tensor, np.ndarray],
    img2: Union[torch.Tensor, np.ndarray],
    max_val: float = 1.0
) -> float:
    """Calculate Peak Signal-to-Noise Ratio (PSNR).
    
    Args:
        img1: First image (predicted/enhanced)
        img2: Second image (ground truth)
        max_val: Maximum possible pixel value (default: 1.0 for normalized images)
    
    Returns:
        PSNR value in dB
    """
    # Convert to numpy if needed
    if isinstance(img1, torch.Tensor):
        img1 = img1.detach().cpu().numpy()
    if isinstance(img2, torch.Tensor):
        img2 = img2.detach().cpu().numpy()
    
    # Handle batch dimension
    if img1.ndim == 4:  # (B, C, H, W)
        psnr_values = []
        for i in range(img1.shape[0]):
            # Transpose to (H, W, C) for skimage
            img1_hwc = np.transpose(img1[i], (1, 2, 0))
            img2_hwc = np.transpose(img2[i], (1, 2, 0))
            psnr = peak_signal_noise_ratio(img2_hwc, img1_hwc, data_range=max_val)
            psnr_values.append(psnr)
        return np.mean(psnr_values)
    else:  # Single image (C, H, W)
        img1_hwc = np.transpose(img1, (1, 2, 0))
        img2_hwc = np.transpose(img2, (1, 2, 0))
        return peak_signal_noise_ratio(img2_hwc, img1_hwc, data_range=max_val)


def calculate_ssim(
    img1: Union[torch.Tensor, np.ndarray],
    img2: Union[torch.Tensor, np.ndarray],
    max_val: float = 1.0,
    multichannel: bool = True
) -> float:
    """Calculate Structural Similarity Index (SSIM).
    
    Args:
        img1: First image (predicted/enhanced)
        img2: Second image (ground truth)
        max_val: Maximum possible pixel value (default: 1.0)
        multichannel: Whether to treat as multichannel image (default: True for RGB)
    
    Returns:
        SSIM value between 0 and 1
    """
    # Convert to numpy if needed
    if isinstance(img1, torch.Tensor):
        img1 = img1.detach().cpu().numpy()
    if isinstance(img2, torch.Tensor):
        img2 = img2.detach().cpu().numpy()
    
    # Handle batch dimension
    if img1.ndim == 4:  # (B, C, H, W)
        ssim_values = []
        for i in range(img1.shape[0]):
            # Transpose to (H, W, C) for skimage
            img1_hwc = np.transpose(img1[i], (1, 2, 0))
            img2_hwc = np.transpose(img2[i], (1, 2, 0))
            ssim = structural_similarity(
                img2_hwc, img1_hwc,
                data_range=max_val,
                channel_axis=2 if multichannel else None
            )
            ssim_values.append(ssim)
        return np.mean(ssim_values)
    else:  # Single image (C, H, W)
        img1_hwc = np.transpose(img1, (1, 2, 0))
        img2_hwc = np.transpose(img2, (1, 2, 0))
        return structural_similarity(
            img2_hwc, img1_hwc,
            data_range=max_val,
            channel_axis=2 if multichannel else None
        )


class LPIPSMetric:
    """Learned Perceptual Image Patch Similarity (LPIPS) metric.
    
    Uses a pretrained network to compute perceptual similarity.
    Lower values indicate more similar images.
    
    Args:
        net: Network type - 'alex', 'vgg', or 'squeeze' (default: 'alex')
        device: Device to run on (default: 'cuda' if available)
    """
    
    def __init__(self, net: str = 'alex', device: str = None):
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        self.device = device
        self.model = lpips.LPIPS(net=net).to(device)
        self.model.eval()
    
    def __call__(
        self,
        img1: torch.Tensor,
        img2: torch.Tensor,
        normalize: bool = False
    ) -> float:
        """Calculate LPIPS between two images.
        
        Args:
            img1: First image of shape (B, 3, H, W) or (3, H, W)
            img2: Second image of same shape
            normalize: Whether to normalize from [0, 1] to [-1, 1] (default: False)
        
        Returns:
            LPIPS distance (lower is better)
        """
        # Ensure batch dimension
        if img1.dim() == 3:
            img1 = img1.unsqueeze(0)
            img2 = img2.unsqueeze(0)
        
        # Move to device
        img1 = img1.to(self.device)
        img2 = img2.to(self.device)
        
        # Normalize if needed
        if normalize:
            img1 = img1 * 2.0 - 1.0
            img2 = img2 * 2.0 - 1.0
        
        # Compute LPIPS
        with torch.no_grad():
            distance = self.model(img1, img2)
        
        return distance.mean().item()


class MetricTracker:
    """Track multiple metrics during training/evaluation.
    
    Args:
        device: Device for LPIPS computation
    """
    
    def __init__(self, device: str = None):
        self.lpips_metric = LPIPSMetric(device=device)
        self.reset()
    
    def reset(self):
        """Reset all tracked metrics."""
        self.psnr_values = []
        self.ssim_values = []
        self.lpips_values = []
    
    def update(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        compute_lpips: bool = True
    ):
        """Update metrics with a new batch.
        
        Args:
            pred: Predicted images
            target: Ground truth images
            compute_lpips: Whether to compute LPIPS (slower, default: True)
        """
        # Compute PSNR
        psnr = calculate_psnr(pred, target)
        self.psnr_values.append(psnr)
        
        # Compute SSIM
        ssim = calculate_ssim(pred, target)
        self.ssim_values.append(ssim)
        
        # Compute LPIPS
        if compute_lpips:
            lpips_val = self.lpips_metric(pred, target, normalize=True)
            self.lpips_values.append(lpips_val)
    
    def compute(self) -> dict:
        """Compute average metrics.
        
        Returns:
            Dictionary with average PSNR, SSIM, and LPIPS
        """
        results = {
            'psnr': np.mean(self.psnr_values) if self.psnr_values else 0.0,
            'ssim': np.mean(self.ssim_values) if self.ssim_values else 0.0,
        }
        
        if self.lpips_values:
            results['lpips'] = np.mean(self.lpips_values)
        
        return results


if __name__ == "__main__":
    # Test metrics
    print("Testing metrics...")
    
    # Create dummy images
    img1 = torch.rand(2, 3, 256, 256)
    img2 = img1 + torch.randn_like(img1) * 0.1  # Add some noise
    img2 = torch.clamp(img2, 0, 1)
    
    # Test PSNR
    psnr = calculate_psnr(img1, img2)
    print(f"PSNR: {psnr:.2f} dB")
    
    # Test SSIM
    ssim = calculate_ssim(img1, img2)
    print(f"SSIM: {ssim:.4f}")
    
    # Test LPIPS
    print("Testing LPIPS...")
    lpips_metric = LPIPSMetric()
    lpips_val = lpips_metric(img1, img2, normalize=True)
    print(f"LPIPS: {lpips_val:.4f}")
    
    # Test MetricTracker
    print("\nTesting MetricTracker...")
    tracker = MetricTracker()
    tracker.update(img1, img2)
    results = tracker.compute()
    print(f"Results: PSNR={results['psnr']:.2f}, SSIM={results['ssim']:.4f}, LPIPS={results['lpips']:.4f}")
