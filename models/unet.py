"""UNet-based Diffusion Model for pixel-space diffusion.

An alternative to the diffusion field approach, this implements a standard
UNet architecture for conditional image diffusion.
"""

import torch
import torch.nn as nn
import math
from typing import Optional, List


class TimeEmbedding(nn.Module):
    """Time embedding for UNet blocks."""
    
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.SiLU(),
            nn.Linear(dim * 4, dim)
        )
    
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """Generate time embedding.
        
        Args:
            t: Timesteps of shape (B,)
        
        Returns:
            Time embeddings of shape (B, dim)
        """
        # Sinusoidal embedding
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device, dtype=torch.float32) * -emb)
        emb = t.float()[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        
        return self.mlp(emb)


class ResidualBlock(nn.Module):
    """Residual block with time embedding and optional conditioning.
    
    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        time_embed_dim: Dimension of time embedding
        dropout: Dropout rate (default: 0.0)
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_embed_dim: int,
        dropout: float = 0.0
    ):
        super().__init__()
        
        self.conv1 = nn.Sequential(
            nn.GroupNorm(32, in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        )
        
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_embed_dim, out_channels)
        )
        
        self.conv2 = nn.Sequential(
            nn.GroupNorm(32, out_channels),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        )
        
        if in_channels != out_channels:
            self.residual_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.residual_conv = nn.Identity()
    
    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor of shape (B, C, H, W)
            time_emb: Time embedding of shape (B, time_embed_dim)
        
        Returns:
            Output tensor of shape (B, out_channels, H, W)
        """
        h = self.conv1(x)
        
        # Add time embedding
        time_emb_proj = self.time_mlp(time_emb)[:, :, None, None]
        h = h + time_emb_proj
        
        h = self.conv2(h)
        
        return h + self.residual_conv(x)


class UNet(nn.Module):
    """Simplified UNet-based conditional diffusion model.
    
    Args:
        in_channels: Number of input channels (default: 3 for RGB)
        out_channels: Number of output channels (default: 3 for RGB)
        channels: Base number of channels (default: 128)
        channel_multipliers: Channel multipliers for each resolution (default: [1, 2, 4])
        num_res_blocks: Number of residual blocks per resolution (default: 2)
        dropout: Dropout rate (default: 0.0)
        condition_dim: Dimension of conditioning vector (default: 256)
    """
    
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        channels: int = 128,
        channel_multipliers: List[int] = [1, 2, 4],
        num_res_blocks: int = 2,
        dropout: float = 0.0,
        condition_dim: int = 256
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.channels = channels
        
        # Time embedding
        time_embed_dim = channels * 4
        self.time_embed = TimeEmbedding(time_embed_dim)
        
        # Condition encoder (lightweight version to avoid circular import)
        self.condition_encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.SiLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.SiLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(256, condition_dim),
            nn.SiLU()
        )
        
        # Condition projection
        self.condition_proj = nn.Linear(condition_dim, time_embed_dim)
        
        # Initial convolution
        self.conv_in = nn.Conv2d(in_channels, channels, kernel_size=3, padding=1)
        
        # Downsample path - track channels for skip connections
        self.down = nn.ModuleList()
        # skip_channels is used only during __init__ to track channel dims for architecture construction
        # Actual runtime skip connections are stored in 'hs' during forward()
        skip_channels_tracker = []  # Track channels for skip connections
        current_channels = channels
        skip_channels_tracker.append(current_channels)  # After conv_in
        
        for i, mult in enumerate(channel_multipliers):
            out_ch = channels * mult
            for _ in range(num_res_blocks):
                self.down.append(ResidualBlock(current_channels, out_ch, time_embed_dim, dropout))
                current_channels = out_ch
                skip_channels_tracker.append(current_channels)
            if i < len(channel_multipliers) - 1:
                self.down.append(nn.Conv2d(current_channels, current_channels, kernel_size=3, stride=2, padding=1))
                skip_channels_tracker.append(current_channels)
        
        # Middle
        self.mid = nn.ModuleList([
            ResidualBlock(current_channels, current_channels, time_embed_dim, dropout),
            ResidualBlock(current_channels, current_channels, time_embed_dim, dropout),
        ])
        
        # Upsample path - use tracked skip channels to build correct architecture
        self.up = nn.ModuleList()
        for i, mult in reversed(list(enumerate(channel_multipliers))):
            out_ch = channels * mult
            for j in range(num_res_blocks + 1):
                skip_ch = skip_channels_tracker.pop()
                self.up.append(ResidualBlock(current_channels + skip_ch, out_ch, time_embed_dim, dropout))
                current_channels = out_ch
            if i > 0:
                self.up.append(nn.ConvTranspose2d(current_channels, current_channels, kernel_size=4, stride=2, padding=1))
        
        # Output
        self.out = nn.Sequential(
            nn.GroupNorm(32, current_channels),
            nn.SiLU(),
            nn.Conv2d(current_channels, out_channels, kernel_size=3, padding=1)
        )
    
    def forward(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        low_light_condition: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Noisy input of shape (B, in_channels, H, W)
            timesteps: Timesteps of shape (B,)
            low_light_condition: Optional low-light image for conditioning (B, 3, H, W)
        
        Returns:
            Predicted noise of shape (B, out_channels, H, W)
        """
        # Time embedding
        time_emb = self.time_embed(timesteps)
        
        # Add condition if provided
        if low_light_condition is not None:
            cond_emb = self.condition_encoder(low_light_condition)
            cond_emb = self.condition_proj(cond_emb)
            time_emb = time_emb + cond_emb
        
        # Initial convolution
        h = self.conv_in(x)
        
        # Downsampling - store skip connections
        hs = [h]
        for module in self.down:
            if isinstance(module, ResidualBlock):
                h = module(h, time_emb)
            else:  # Downsample
                h = module(h)
            hs.append(h)
        
        # Middle
        for module in self.mid:
            h = module(h, time_emb)
        
        # Upsampling - use skip connections
        for module in self.up:
            if isinstance(module, ResidualBlock):
                h = torch.cat([h, hs.pop()], dim=1)
                h = module(h, time_emb)
            else:  # Upsample
                h = module(h)
        
        # Output
        return self.out(h)


if __name__ == "__main__":
    # Test the UNet model
    print("Testing UNet model...")
    
    model = UNet(
        in_channels=3,
        out_channels=3,
        channels=128,
        channel_multipliers=[1, 2, 4],
        num_res_blocks=2
    )
    
    x = torch.randn(2, 3, 256, 256)
    timesteps = torch.randint(0, 1000, (2,))
    low_light = torch.randn(2, 3, 256, 256)
    
    output = model(x, timesteps, low_light)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")

