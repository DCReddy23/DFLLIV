"""Condition Encoder for extracting features from low-light input images.

Uses a pretrained ResNet-18 backbone to extract a global conditioning vector
from the input low-light image.
"""

import torch
import torch.nn as nn
import torchvision.models as models
from typing import Optional


class ConditionEncoder(nn.Module):
    """CNN encoder for low-light image conditioning.
    
    Uses ResNet-18 pretrained on ImageNet as backbone, with a custom projection
    head to produce a conditioning vector of desired dimension.
    
    Args:
        condition_dim: Dimension of output conditioning vector (default: 256)
        pretrained: Whether to use ImageNet pretrained weights (default: True)
        freeze_backbone: Whether to freeze backbone weights (default: False)
    """
    
    def __init__(
        self,
        condition_dim: int = 256,
        pretrained: bool = True,
        freeze_backbone: bool = False
    ):
        super().__init__()
        self.condition_dim = condition_dim
        
        # Load ResNet-18 backbone
        resnet = models.resnet18(pretrained=pretrained)
        
        # Remove final FC layer to get features (512-dim for ResNet-18)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        self.feature_dim = 512  # ResNet-18 output dimension
        
        # Optionally freeze backbone
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # Projection head to condition_dim
        self.projection = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.feature_dim, condition_dim),
            nn.LayerNorm(condition_dim),
            nn.SiLU()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract conditioning vector from input image.
        
        Args:
            x: Input images of shape (B, 3, H, W), normalized to [0, 1] or [-1, 1]
        
        Returns:
            Conditioning vectors of shape (B, condition_dim)
        """
        # Extract features from backbone
        features = self.backbone(x)  # (B, 512, 1, 1)
        
        # Project to conditioning dimension
        condition_vector = self.projection(features)  # (B, condition_dim)
        
        return condition_vector


class LightweightConditionEncoder(nn.Module):
    """Lightweight CNN encoder as an alternative to ResNet.
    
    A simpler custom CNN for faster training and inference when
    computational resources are limited.
    
    Args:
        condition_dim: Dimension of output conditioning vector (default: 256)
        num_channels: Number of channels in convolutional layers (default: 64)
    """
    
    def __init__(
        self,
        condition_dim: int = 256,
        num_channels: int = 64
    ):
        super().__init__()
        self.condition_dim = condition_dim
        
        # Simple CNN with downsampling
        self.encoder = nn.Sequential(
            # Input: (B, 3, H, W)
            nn.Conv2d(3, num_channels, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(num_channels),
            nn.SiLU(),
            
            # (B, 64, H/2, W/2)
            nn.Conv2d(num_channels, num_channels * 2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(num_channels * 2),
            nn.SiLU(),
            
            # (B, 128, H/4, W/4)
            nn.Conv2d(num_channels * 2, num_channels * 4, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(num_channels * 4),
            nn.SiLU(),
            
            # (B, 256, H/8, W/8)
            nn.Conv2d(num_channels * 4, num_channels * 4, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(num_channels * 4),
            nn.SiLU(),
            
            # (B, 256, H/16, W/16)
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        # Projection head
        self.projection = nn.Sequential(
            nn.Flatten(),
            nn.Linear(num_channels * 4, condition_dim),
            nn.LayerNorm(condition_dim),
            nn.SiLU()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract conditioning vector from input image.
        
        Args:
            x: Input images of shape (B, 3, H, W)
        
        Returns:
            Conditioning vectors of shape (B, condition_dim)
        """
        features = self.encoder(x)
        condition_vector = self.projection(features)
        return condition_vector


if __name__ == "__main__":
    # Test the condition encoder
    print("Testing ConditionEncoder (ResNet-18 based)...")
    encoder = ConditionEncoder(condition_dim=256, pretrained=False)
    
    # Test with batch of images
    batch_images = torch.randn(4, 3, 256, 256)
    condition_vectors = encoder(batch_images)
    
    print(f"Input shape: {batch_images.shape}")
    print(f"Output shape: {condition_vectors.shape}")
    print(f"Number of parameters: {sum(p.numel() for p in encoder.parameters()):,}")
    
    print("\nTesting LightweightConditionEncoder...")
    lightweight_encoder = LightweightConditionEncoder(condition_dim=256)
    condition_vectors_light = lightweight_encoder(batch_images)
    
    print(f"Input shape: {batch_images.shape}")
    print(f"Output shape: {condition_vectors_light.shape}")
    print(f"Number of parameters: {sum(p.numel() for p in lightweight_encoder.parameters()):,}")
