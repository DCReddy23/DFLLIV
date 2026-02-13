"""Diffusion Field MLP Model.

Core MLP network that takes Fourier-encoded coordinates, conditioning vector,
and timestep embedding to predict noise at each coordinate.
"""

import torch
import torch.nn as nn
import math
from typing import Optional


class TimestepEmbedding(nn.Module):
    """Sinusoidal timestep embedding.
    
    Converts integer timesteps to continuous embeddings using sinusoidal
    positional encodings, similar to transformers.
    
    Args:
        embedding_dim: Dimension of the embedding (default: 256)
        max_period: Maximum period for sinusoidal encoding (default: 10000)
    """
    
    def __init__(self, embedding_dim: int = 256, max_period: int = 10000):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.max_period = max_period
        
        # MLP to project embedding
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim * 4),
            nn.SiLU(),
            nn.Linear(embedding_dim * 4, embedding_dim)
        )
    
    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        """Generate timestep embeddings.
        
        Args:
            timesteps: Timestep indices of shape (B,) or (B, 1)
        
        Returns:
            Timestep embeddings of shape (B, embedding_dim)
        """
        # Ensure timesteps is 1D
        if timesteps.dim() > 1:
            timesteps = timesteps.squeeze(-1)
        
        half_dim = self.embedding_dim // 2
        emb = math.log(self.max_period) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=timesteps.device, dtype=torch.float32) * -emb)
        emb = timesteps.float()[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        
        # Project through MLP
        emb = self.mlp(emb)
        
        return emb


class DiffusionFieldMLP(nn.Module):
    """Core Diffusion Field MLP.
    
    Neural field that takes Fourier-encoded coordinates, conditioning vector,
    and timestep embedding to predict RGB noise at each coordinate.
    
    Inspired by NeRF architecture with skip connections and SiLU activations.
    
    Args:
        coord_dim: Dimension of coordinate encoding input
        condition_dim: Dimension of conditioning vector (default: 256)
        time_embed_dim: Dimension of timestep embedding (default: 256)
        hidden_dim: Hidden layer dimension (default: 256)
        num_layers: Number of MLP layers (default: 8)
        skip_connections: Layer indices for skip connections (default: [4])
        output_channels: Number of output channels (RGB) (default: 3)
    """
    
    def __init__(
        self,
        coord_dim: int,
        condition_dim: int = 256,
        time_embed_dim: int = 256,
        hidden_dim: int = 256,
        num_layers: int = 8,
        skip_connections: Optional[list] = None,
        output_channels: int = 3
    ):
        super().__init__()
        self.coord_dim = coord_dim
        self.condition_dim = condition_dim
        self.time_embed_dim = time_embed_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.skip_connections = skip_connections if skip_connections is not None else [4]
        self.output_channels = output_channels
        
        # Timestep embedding
        self.time_embed = TimestepEmbedding(time_embed_dim)
        
        # Input projection
        # Combines coord encoding + condition + time embedding + noisy pixel values
        input_dim = coord_dim + condition_dim + time_embed_dim + output_channels
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU()
        )
        
        # Hidden layers with skip connections
        self.layers = nn.ModuleList()
        for i in range(num_layers - 1):
            if i in self.skip_connections:
                # Skip connection from input
                layer_input_dim = hidden_dim + input_dim
            else:
                layer_input_dim = hidden_dim
            
            self.layers.append(nn.Sequential(
                nn.Linear(layer_input_dim, hidden_dim),
                nn.SiLU()
            ))
        
        # Output layer
        self.output_layer = nn.Linear(hidden_dim, output_channels)
    
    def forward(
        self,
        coords_encoded: torch.Tensor,
        condition: torch.Tensor,
        timesteps: torch.Tensor,
        noisy_pixels: torch.Tensor
    ) -> torch.Tensor:
        """Predict noise at given coordinates.
        
        Args:
            coords_encoded: Fourier-encoded coordinates of shape (B, N, coord_dim)
            condition: Conditioning vectors of shape (B, condition_dim)
            timesteps: Timestep indices of shape (B,)
            noisy_pixels: Noisy pixel values at coordinates of shape (B, N, 3)
        
        Returns:
            Predicted noise of shape (B, N, output_channels)
        """
        batch_size, num_points, _ = coords_encoded.shape
        
        # Embed timesteps
        time_emb = self.time_embed(timesteps)  # (B, time_embed_dim)
        
        # Expand condition and time_emb to match coordinate points
        condition_expanded = condition.unsqueeze(1).expand(-1, num_points, -1)  # (B, N, condition_dim)
        time_emb_expanded = time_emb.unsqueeze(1).expand(-1, num_points, -1)  # (B, N, time_embed_dim)
        
        # Concatenate all inputs including noisy pixels
        x = torch.cat([coords_encoded, condition_expanded, time_emb_expanded, noisy_pixels], dim=-1)  # (B, N, input_dim)
        
        # Store for skip connections
        input_features = x
        
        # Input layer
        x = self.input_layer(x)  # (B, N, hidden_dim)
        
        # Hidden layers with skip connections
        for i, layer in enumerate(self.layers):
            if i in self.skip_connections:
                x = torch.cat([x, input_features], dim=-1)
            x = layer(x)
        
        # Output layer
        noise_pred = self.output_layer(x)  # (B, N, output_channels)
        
        return noise_pred


class DiffusionFieldModel(nn.Module):
    """Complete Diffusion Field model combining all components.
    
    This is the full model that can be used for training and inference,
    combining coordinate encoding, condition encoding, and the diffusion field MLP.
    
    Args:
        fourier_frequencies: Number of Fourier frequencies for coord encoding (default: 128)
        condition_dim: Dimension of conditioning vector (default: 256)
        time_embed_dim: Dimension of timestep embedding (default: 256)
        hidden_dim: Hidden layer dimension (default: 256)
        num_layers: Number of MLP layers (default: 8)
        output_channels: Number of output channels (default: 3)
    """
    
    def __init__(
        self,
        fourier_frequencies: int = 128,
        condition_dim: int = 256,
        time_embed_dim: int = 256,
        hidden_dim: int = 256,
        num_layers: int = 8,
        output_channels: int = 3
    ):
        super().__init__()
        
        # Import here to avoid circular dependency
        from .coord_encoder import CoordEncoder
        from .condition_encoder import ConditionEncoder
        
        # Create submodules
        self.coord_encoder = CoordEncoder(num_frequencies=fourier_frequencies)
        self.condition_encoder = ConditionEncoder(condition_dim=condition_dim)
        
        # Diffusion field MLP
        self.diffusion_mlp = DiffusionFieldMLP(
            coord_dim=self.coord_encoder.output_dim,
            condition_dim=condition_dim,
            time_embed_dim=time_embed_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            output_channels=output_channels
        )
    
    def forward(
        self,
        low_light_image: torch.Tensor,
        coords: torch.Tensor,
        timesteps: torch.Tensor,
        noisy_image: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass through the full diffusion field model.
        
        Args:
            low_light_image: Low-light input image of shape (B, 3, H, W)
            coords: Normalized coordinates of shape (B, N, 2) or (H, W, 2)
            timesteps: Timestep indices of shape (B,)
            noisy_image: Noisy image at current timestep of shape (B, 3, H, W)
        
        Returns:
            Predicted noise at coordinates of shape (B, N, 3) or (B, H*W, 3)
        """
        # Extract condition from low-light image
        condition = self.condition_encoder(low_light_image)  # (B, condition_dim)
        
        # Handle different coordinate shapes
        if coords.dim() == 3 and coords.shape[0] != low_light_image.shape[0]:
            # coords is (H, W, 2), need to batch it
            coords = coords.reshape(-1, 2).unsqueeze(0)  # (1, H*W, 2)
            coords = coords.expand(low_light_image.shape[0], -1, -1)  # (B, H*W, 2)
        
        # Encode coordinates
        coords_encoded = self.coord_encoder(coords)  # (B, N, coord_dim)
        
        # Sample noisy pixel values at coordinates
        # noisy_image: (B, 3, H, W), coords: (B, N, 2) with values in [0,1]
        # Use grid_sample to sample pixels at coordinate locations
        batch_size = low_light_image.shape[0]
        num_points = coords_encoded.shape[1]
        
        # Convert coords from [0,1] to [-1,1] for grid_sample
        grid = coords * 2.0 - 1.0  # (B, N, 2)
        grid = grid.unsqueeze(1)  # (B, 1, N, 2)
        
        noisy_pixels = torch.nn.functional.grid_sample(
            noisy_image, grid, mode='bilinear', align_corners=True
        )  # (B, 3, 1, N)
        noisy_pixels = noisy_pixels.squeeze(2).permute(0, 2, 1)  # (B, N, 3)
        
        # Predict noise
        noise_pred = self.diffusion_mlp(coords_encoded, condition, timesteps, noisy_pixels)
        
        return noise_pred


if __name__ == "__main__":
    # Test the diffusion field model
    print("Testing TimestepEmbedding...")
    time_embed = TimestepEmbedding(embedding_dim=256)
    timesteps = torch.randint(0, 1000, (4,))
    time_emb = time_embed(timesteps)
    print(f"Timesteps: {timesteps}")
    print(f"Time embedding shape: {time_emb.shape}")
    
    print("\nTesting DiffusionFieldMLP...")
    mlp = DiffusionFieldMLP(
        coord_dim=514,  # 128 freq * 2 * 2 + 2 input
        condition_dim=256,
        time_embed_dim=256,
        hidden_dim=256,
        num_layers=8
    )
    
    coords_encoded = torch.randn(4, 100, 514)
    condition = torch.randn(4, 256)
    noisy_pixels = torch.randn(4, 100, 3)  # Add noisy pixels input
    noise_pred = mlp(coords_encoded, condition, timesteps, noisy_pixels)
    
    print(f"Coords encoded shape: {coords_encoded.shape}")
    print(f"Condition shape: {condition.shape}")
    print(f"Noise prediction shape: {noise_pred.shape}")
    print(f"Number of parameters: {sum(p.numel() for p in mlp.parameters()):,}")
    
    print("\nTesting Full DiffusionFieldModel...")
    model = DiffusionFieldModel(
        fourier_frequencies=128,
        condition_dim=256,
        hidden_dim=256,
        num_layers=8
    )
    
    low_light_img = torch.randn(2, 3, 256, 256)
    coords = torch.rand(256, 256, 2)
    timesteps = torch.randint(0, 1000, (2,))
    noisy_img = torch.randn(2, 3, 256, 256)  # Add noisy image input
    
    noise_pred_full = model(low_light_img, coords, timesteps, noisy_img)
    print(f"Low-light image shape: {low_light_img.shape}")
    print(f"Coords shape: {coords.shape}")
    print(f"Noise prediction shape: {noise_pred_full.shape}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
