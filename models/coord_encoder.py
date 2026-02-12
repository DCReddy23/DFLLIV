"""Coordinate Encoder with Fourier Feature Encoding.

This module implements Fourier feature encoding for pixel coordinates,
transforming (x, y) coordinates into high-dimensional sinusoidal features
as described in "Fourier Features Let Networks Learn High Frequency Functions".
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional


class CoordEncoder(nn.Module):
    """Fourier feature encoding for coordinates.
    
    Encodes 2D coordinates using sinusoidal positional encodings with multiple
    frequencies, enabling neural networks to better learn high-frequency details.
    
    Args:
        num_frequencies: Number of frequency bands for encoding (default: 128)
        max_freq_log2: Maximum frequency in log2 scale (default: 10)
        include_input: Whether to include the raw input coordinates (default: True)
    """
    
    def __init__(
        self,
        num_frequencies: int = 128,
        max_freq_log2: int = 10,
        include_input: bool = True
    ):
        super().__init__()
        self.num_frequencies = num_frequencies
        self.max_freq_log2 = max_freq_log2
        self.include_input = include_input
        
        # Create frequency bands (2^0, 2^1, ..., 2^max_freq_log2)
        freq_bands = 2.0 ** torch.linspace(0, max_freq_log2, num_frequencies)
        self.register_buffer('freq_bands', freq_bands)
        
        # Calculate output dimension
        # Each coord has num_frequencies * 2 (sin and cos) features
        # Plus optionally 2 for the raw input coordinates
        self.output_dim = num_frequencies * 2 * 2  # 2 coords * 2 functions
        if include_input:
            self.output_dim += 2
    
    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """Encode coordinates with Fourier features.
        
        Args:
            coords: Input coordinates of shape (B, N, 2) or (N, 2)
                   where N is number of points, coordinates are in range [0, 1]
        
        Returns:
            Encoded features of shape (B, N, output_dim) or (N, output_dim)
        """
        # Ensure coords are in correct shape
        if coords.dim() == 2:
            coords = coords.unsqueeze(0)  # Add batch dimension
            squeeze_output = True
        else:
            squeeze_output = False
        
        batch_size, num_points, _ = coords.shape
        
        # coords: (B, N, 2) -> (B, N, 1, 2)
        coords = coords.unsqueeze(2)
        
        # freq_bands: (F,) -> (1, 1, F, 1)
        freq_bands = self.freq_bands.view(1, 1, -1, 1)
        
        # Compute scaled coordinates: (B, N, F, 2)
        scaled_coords = coords * freq_bands * np.pi
        
        # Apply sin and cos: (B, N, F, 2) each
        sin_features = torch.sin(scaled_coords)
        cos_features = torch.cos(scaled_coords)
        
        # Flatten frequency and coordinate dimensions: (B, N, F*2*2)
        encoded = torch.cat([sin_features, cos_features], dim=2)
        encoded = encoded.reshape(batch_size, num_points, -1)
        
        # Optionally concatenate raw input
        if self.include_input:
            coords_2d = coords.squeeze(2)  # (B, N, 2)
            encoded = torch.cat([coords_2d, encoded], dim=-1)
        
        if squeeze_output:
            encoded = encoded.squeeze(0)
        
        return encoded


def create_coordinate_grid(
    height: int,
    width: int,
    device: Optional[torch.device] = None
) -> torch.Tensor:
    """Create a normalized coordinate grid for an image.
    
    Args:
        height: Image height in pixels
        width: Image width in pixels
        device: Device to create tensor on (default: CPU)
    
    Returns:
        Coordinate grid of shape (H, W, 2) with values in [0, 1]
    """
    # Create coordinate grids
    y_coords = torch.linspace(0, 1, height, device=device)
    x_coords = torch.linspace(0, 1, width, device=device)
    
    # Create meshgrid
    yy, xx = torch.meshgrid(y_coords, x_coords, indexing='ij')
    
    # Stack to get (H, W, 2)
    coords = torch.stack([xx, yy], dim=-1)
    
    return coords


if __name__ == "__main__":
    # Test the coordinate encoder
    encoder = CoordEncoder(num_frequencies=128)
    print(f"Output dimension: {encoder.output_dim}")
    
    # Test with single coordinate
    coord = torch.rand(100, 2)  # 100 random coordinates
    encoded = encoder(coord)
    print(f"Input shape: {coord.shape}")
    print(f"Output shape: {encoded.shape}")
    
    # Test with batch
    coords_batch = torch.rand(4, 100, 2)  # Batch of 4
    encoded_batch = encoder(coords_batch)
    print(f"Batch input shape: {coords_batch.shape}")
    print(f"Batch output shape: {encoded_batch.shape}")
    
    # Test coordinate grid
    grid = create_coordinate_grid(256, 256)
    print(f"Coordinate grid shape: {grid.shape}")
    print(f"Coordinate range: [{grid.min():.3f}, {grid.max():.3f}]")
