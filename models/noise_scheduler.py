"""Noise Scheduler for Diffusion Models.

Implements noise scheduling for both DDPM and DDIM sampling, supporting
linear and cosine beta schedules.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Tuple, Dict


class NoiseScheduler:
    """Noise scheduler for diffusion models.
    
    Implements forward diffusion (adding noise) and reverse diffusion (denoising)
    with support for both DDPM and DDIM sampling strategies.
    
    Args:
        num_timesteps: Number of diffusion timesteps (default: 1000)
        beta_schedule: Type of beta schedule - 'linear' or 'cosine' (default: 'cosine')
        beta_start: Starting beta value for linear schedule (default: 0.0001)
        beta_end: Ending beta value for linear schedule (default: 0.02)
        device: Device to place tensors on (default: 'cpu')
    """
    
    def __init__(
        self,
        num_timesteps: int = 1000,
        beta_schedule: str = 'cosine',
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        device: str = 'cpu'
    ):
        self.num_timesteps = num_timesteps
        self.beta_schedule = beta_schedule
        self.device = device
        
        # Generate beta schedule
        if beta_schedule == 'linear':
            self.betas = torch.linspace(beta_start, beta_end, num_timesteps, device=device)
        elif beta_schedule == 'cosine':
            self.betas = self._cosine_beta_schedule(num_timesteps, device=device)
        else:
            raise ValueError(f"Unknown beta schedule: {beta_schedule}")
        
        # Pre-compute useful quantities
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = torch.cat([
            torch.ones(1, device=device),
            self.alphas_cumprod[:-1]
        ])
        
        # For reverse process
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        
        # For posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        # Clip to avoid division by zero at t=0
        self.posterior_variance = torch.clamp(self.posterior_variance, min=1e-20)
        
        self.posterior_log_variance_clipped = torch.log(self.posterior_variance)
        self.posterior_mean_coef1 = (
            self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (1.0 - self.alphas_cumprod)
        )
    
    def _cosine_beta_schedule(self, timesteps: int, device: str, s: float = 0.008) -> torch.Tensor:
        """Cosine schedule as proposed in 'Improved Denoising Diffusion Probabilistic Models'.
        
        Args:
            timesteps: Number of diffusion steps
            device: Device to create tensor on
            s: Small offset to prevent beta from being too small near t=0
        
        Returns:
            Beta values for each timestep
        """
        steps = timesteps + 1
        t = torch.linspace(0, timesteps, steps, device=device)
        alphas_cumprod = torch.cos(((t / timesteps) + s) / (1 + s) * np.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0.0001, 0.9999)
    
    def add_noise(
        self,
        x_start: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor
    ) -> torch.Tensor:
        """Add noise to clean images (forward diffusion).
        
        Implements: x_t = sqrt(alpha_cumprod_t) * x_0 + sqrt(1 - alpha_cumprod_t) * noise
        
        Args:
            x_start: Clean images of shape (B, C, H, W) or (B, N, C)
            noise: Noise tensor of same shape as x_start
            timesteps: Timestep indices of shape (B,)
        
        Returns:
            Noisy images at specified timesteps
        """
        sqrt_alpha_prod = self.sqrt_alphas_cumprod[timesteps]
        sqrt_one_minus_alpha_prod = self.sqrt_one_minus_alphas_cumprod[timesteps]
        
        # Reshape for broadcasting
        while len(sqrt_alpha_prod.shape) < len(x_start.shape):
            sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)
        
        noisy_images = sqrt_alpha_prod * x_start + sqrt_one_minus_alpha_prod * noise
        return noisy_images
    
    def sample_timesteps(self, batch_size: int) -> torch.Tensor:
        """Sample random timesteps for training.
        
        Args:
            batch_size: Number of timesteps to sample
        
        Returns:
            Random timestep indices of shape (batch_size,)
        """
        return torch.randint(0, self.num_timesteps, (batch_size,), device=self.device)
    
    def step(
        self,
        model_output: torch.Tensor,
        timestep: int,
        sample: torch.Tensor,
        predict_epsilon: bool = True
    ) -> torch.Tensor:
        """Perform one reverse diffusion step (DDPM).
        
        Args:
            model_output: Output from the model (predicted noise or x_0)
            timestep: Current timestep
            sample: Current noisy sample
            predict_epsilon: Whether model predicts noise (True) or x_0 (False)
        
        Returns:
            Denoised sample at timestep-1
        """
        t = timestep
        
        # Get parameters for this timestep
        alpha_t = self.alphas[t]
        alpha_cumprod_t = self.alphas_cumprod[t]
        beta_t = self.betas[t]
        
        # Compute predicted x_0
        if predict_epsilon:
            # model_output is epsilon
            pred_original_sample = (
                sample - self.sqrt_one_minus_alphas_cumprod[t] * model_output
            ) / self.sqrt_alphas_cumprod[t]
        else:
            # model_output is x_0
            pred_original_sample = model_output
        
        # Compute mean of posterior q(x_{t-1} | x_t, x_0)
        pred_prev_sample = (
            self.posterior_mean_coef1[t] * pred_original_sample +
            self.posterior_mean_coef2[t] * sample
        )
        
        # Add noise (except at t=0)
        if t > 0:
            noise = torch.randn_like(sample)
            variance = torch.sqrt(self.posterior_variance[t])
            # Reshape for broadcasting
            while len(variance.shape) < len(sample.shape):
                variance = variance.unsqueeze(-1)
            pred_prev_sample = pred_prev_sample + variance * noise
        
        return pred_prev_sample
    
    def ddim_step(
        self,
        model_output: torch.Tensor,
        timestep: int,
        sample: torch.Tensor,
        eta: float = 0.0,
        prev_timestep: Optional[int] = None
    ) -> torch.Tensor:
        """Perform one DDIM step (deterministic when eta=0).
        
        DDIM allows for faster sampling by skipping timesteps while maintaining
        quality. When eta=0, sampling is deterministic.
        
        Args:
            model_output: Predicted noise from the model
            timestep: Current timestep
            sample: Current noisy sample
            eta: Controls stochasticity (0=deterministic, 1=DDPM-like)
            prev_timestep: Previous timestep to jump to (for skipping steps)
        
        Returns:
            Denoised sample at prev_timestep
        """
        t = timestep
        prev_t = prev_timestep if prev_timestep is not None else t - 1
        
        # Get alpha values
        alpha_prod_t = self.alphas_cumprod[t]
        alpha_prod_t_prev = self.alphas_cumprod[prev_t] if prev_t >= 0 else torch.tensor(1.0, device=self.device)
        
        # Predict x_0
        pred_original_sample = (
            sample - torch.sqrt(1 - alpha_prod_t) * model_output
        ) / torch.sqrt(alpha_prod_t)
        
        # Compute variance
        variance = (1 - alpha_prod_t_prev) / (1 - alpha_prod_t) * (1 - alpha_prod_t / alpha_prod_t_prev)
        std_dev_t = eta * torch.sqrt(variance)
        
        # Compute direction pointing to x_t
        pred_sample_direction = torch.sqrt(1 - alpha_prod_t_prev - std_dev_t ** 2) * model_output
        
        # Compute x_{t-1}
        pred_prev_sample = torch.sqrt(alpha_prod_t_prev) * pred_original_sample + pred_sample_direction
        
        # Add noise if eta > 0
        if eta > 0 and prev_t >= 0:
            noise = torch.randn_like(sample)
            pred_prev_sample = pred_prev_sample + std_dev_t * noise
        
        return pred_prev_sample


if __name__ == "__main__":
    # Test the noise scheduler
    print("Testing NoiseScheduler...")
    
    # Test linear schedule
    scheduler_linear = NoiseScheduler(num_timesteps=1000, beta_schedule='linear')
    print(f"Linear schedule - Beta range: [{scheduler_linear.betas.min():.6f}, {scheduler_linear.betas.max():.6f}]")
    
    # Test cosine schedule
    scheduler_cosine = NoiseScheduler(num_timesteps=1000, beta_schedule='cosine')
    print(f"Cosine schedule - Beta range: [{scheduler_cosine.betas.min():.6f}, {scheduler_cosine.betas.max():.6f}]")
    
    # Test add_noise
    x_start = torch.randn(4, 3, 64, 64)
    noise = torch.randn_like(x_start)
    timesteps = scheduler_cosine.sample_timesteps(4)
    
    x_noisy = scheduler_cosine.add_noise(x_start, noise, timesteps)
    print(f"\nNoise addition:")
    print(f"  Clean image shape: {x_start.shape}")
    print(f"  Noisy image shape: {x_noisy.shape}")
    print(f"  Timesteps: {timesteps}")
    
    # Test DDPM step
    pred_noise = torch.randn_like(x_start)
    x_prev = scheduler_cosine.step(pred_noise, timestep=500, sample=x_noisy)
    print(f"\nDDPM step:")
    print(f"  Input shape: {x_noisy.shape}")
    print(f"  Output shape: {x_prev.shape}")
    
    # Test DDIM step
    x_prev_ddim = scheduler_cosine.ddim_step(pred_noise, timestep=500, sample=x_noisy, eta=0.0)
    print(f"\nDDIM step:")
    print(f"  Input shape: {x_noisy.shape}")
    print(f"  Output shape: {x_prev_ddim.shape}")
