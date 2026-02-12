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


class AttentionBlock(nn.Module):
    """Self-attention block for UNet.
    
    Args:
        channels: Number of channels
        num_heads: Number of attention heads (default: 4)
    """
    
    def __init__(self, channels: int, num_heads: int = 4):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        
        self.norm = nn.GroupNorm(32, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1)
        self.proj = nn.Conv2d(channels, channels, kernel_size=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor of shape (B, C, H, W)
        
        Returns:
            Output tensor of shape (B, C, H, W)
        """
        B, C, H, W = x.shape
        
        h = self.norm(x)
        qkv = self.qkv(h)
        
        # Reshape for multi-head attention
        qkv = qkv.reshape(B, 3, self.num_heads, C // self.num_heads, H * W)
        qkv = qkv.permute(1, 0, 2, 4, 3)  # (3, B, num_heads, H*W, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Attention
        attn = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(C // self.num_heads)
        attn = torch.softmax(attn, dim=-1)
        
        # Apply attention to values
        h = torch.matmul(attn, v)
        h = h.permute(0, 1, 3, 2).reshape(B, C, H, W)
        
        h = self.proj(h)
        
        return x + h


class Downsample(nn.Module):
    """Downsampling block."""
    
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample(nn.Module):
    """Upsampling block."""
    
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = nn.functional.interpolate(x, scale_factor=2, mode='nearest')
        return self.conv(x)


class UNet(nn.Module):
    """UNet-based conditional diffusion model.
    
    Args:
        in_channels: Number of input channels (default: 3 for RGB)
        out_channels: Number of output channels (default: 3 for RGB)
        channels: Base number of channels (default: 128)
        channel_multipliers: Channel multipliers for each resolution (default: [1, 2, 2, 4])
        num_res_blocks: Number of residual blocks per resolution (default: 2)
        attention_resolutions: Resolutions to apply attention at (default: [16])
        dropout: Dropout rate (default: 0.0)
        condition_dim: Dimension of conditioning vector (default: 256)
    """
    
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        channels: int = 128,
        channel_multipliers: List[int] = [1, 2, 2, 4],
        num_res_blocks: int = 2,
        attention_resolutions: List[int] = [16],
        dropout: float = 0.0,
        condition_dim: int = 256
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.channels = channels
        self.num_resolutions = len(channel_multipliers)
        
        # Time embedding
        time_embed_dim = channels * 4
        self.time_embed = TimeEmbedding(time_embed_dim)
        
        # Condition encoder (separate from diffusion field condition encoder)
        from .condition_encoder import LightweightConditionEncoder
        self.condition_encoder = LightweightConditionEncoder(condition_dim=condition_dim)
        
        # Condition projection
        self.condition_proj = nn.Linear(condition_dim, time_embed_dim)
        
        # Initial convolution
        self.conv_in = nn.Conv2d(in_channels, channels, kernel_size=3, padding=1)
        
        # Downsampling
        self.down_blocks = nn.ModuleList()
        current_channels = channels
        input_block_channels = [channels]
        
        for level, mult in enumerate(channel_multipliers):
            out_channels_level = channels * mult
            
            for _ in range(num_res_blocks):
                block = ResidualBlock(
                    current_channels,
                    out_channels_level,
                    time_embed_dim,
                    dropout
                )
                self.down_blocks.append(block)
                current_channels = out_channels_level
                input_block_channels.append(current_channels)
                
                # Add attention if at specified resolution
                # Note: We'll add attention based on level for simplicity
                if level >= 2:  # Add attention at lower resolutions
                    self.down_blocks.append(AttentionBlock(current_channels))
            
            # Downsample (except at last level)
            if level < self.num_resolutions - 1:
                self.down_blocks.append(Downsample(current_channels))
                input_block_channels.append(current_channels)
        
        # Middle
        self.middle_blocks = nn.ModuleList([
            ResidualBlock(current_channels, current_channels, time_embed_dim, dropout),
            AttentionBlock(current_channels),
            ResidualBlock(current_channels, current_channels, time_embed_dim, dropout)
        ])
        
        # Upsampling
        self.up_blocks = nn.ModuleList()
        
        for level, mult in reversed(list(enumerate(channel_multipliers))):
            out_channels_level = channels * mult
            
            for i in range(num_res_blocks + 1):
                # Skip connection from downsampling
                skip_channels = input_block_channels.pop()
                block = ResidualBlock(
                    current_channels + skip_channels,
                    out_channels_level,
                    time_embed_dim,
                    dropout
                )
                self.up_blocks.append(block)
                current_channels = out_channels_level
                
                # Add attention if at specified resolution
                if level >= 2:
                    self.up_blocks.append(AttentionBlock(current_channels))
                
                # Upsample (except at last block of last level)
                if level > 0 and i == num_res_blocks:
                    self.up_blocks.append(Upsample(current_channels))
        
        # Output
        self.out = nn.Sequential(
            nn.GroupNorm(32, current_channels),
            nn.SiLU(),
            nn.Conv2d(current_channels, self.out_channels, kernel_size=3, padding=1)
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
        
        # Downsampling with skip connections
        skip_connections = [h]
        for module in self.down_blocks:
            if isinstance(module, (ResidualBlock, AttentionBlock)):
                if isinstance(module, ResidualBlock):
                    h = module(h, time_emb)
                else:
                    h = module(h)
                skip_connections.append(h)
            elif isinstance(module, Downsample):
                h = module(h)
                skip_connections.append(h)
        
        # Middle
        for module in self.middle_blocks:
            if isinstance(module, ResidualBlock):
                h = module(h, time_emb)
            else:
                h = module(h)
        
        # Upsampling with skip connections
        for module in self.up_blocks:
            if isinstance(module, ResidualBlock):
                skip = skip_connections.pop()
                h = torch.cat([h, skip], dim=1)
                h = module(h, time_emb)
            elif isinstance(module, AttentionBlock):
                h = module(h)
            elif isinstance(module, Upsample):
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
        channel_multipliers=[1, 2, 2, 4],
        num_res_blocks=2
    )
    
    x = torch.randn(2, 3, 256, 256)
    timesteps = torch.randint(0, 1000, (2,))
    low_light = torch.randn(2, 3, 256, 256)
    
    output = model(x, timesteps, low_light)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
