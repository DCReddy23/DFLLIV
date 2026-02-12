"""Training script for low-light image enhancement using diffusion fields.

Supports:
- Mixed precision training (AMP)
- Gradient clipping
- Learning rate scheduling with warmup
- EMA of model weights
- Checkpointing
- TensorBoard / Weights & Biases logging
- Validation with PSNR/SSIM metrics
"""

import os
import argparse
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np
from typing import Dict, Optional
import lpips

from models import DiffusionFieldModel, UNet, NoiseScheduler
from data.dataset import get_dataloader
from utils.metrics import MetricTracker
from utils.visualization import save_comparison


class EMA:
    """Exponential Moving Average of model parameters."""
    
    def __init__(self, model: nn.Module, decay: float = 0.9999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        
        # Initialize shadow parameters
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
    
    def update(self):
        """Update EMA parameters."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()
    
    def apply_shadow(self):
        """Apply EMA parameters to model."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.backup[name] = param.data
                param.data = self.shadow[name]
    
    def restore(self):
        """Restore original parameters."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}


class Trainer:
    """Trainer for diffusion-based low-light enhancement."""
    
    def __init__(self, config: Dict, args: argparse.Namespace):
        self.config = config
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Create directories
        os.makedirs(config['training']['checkpoint_dir'], exist_ok=True)
        os.makedirs(config['training']['log_dir'], exist_ok=True)
        os.makedirs('outputs/train_vis', exist_ok=True)
        
        # Initialize model
        self.model = self._create_model()
        self.model.to(self.device)
        
        # Initialize noise scheduler
        self.noise_scheduler = NoiseScheduler(
            num_timesteps=config['diffusion']['num_timesteps'],
            beta_schedule=config['diffusion']['beta_schedule'],
            beta_start=config['diffusion']['beta_start'],
            beta_end=config['diffusion']['beta_end'],
            device=self.device
        )
        
        # Initialize optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config['training']['learning_rate'],
            weight_decay=config['training']['weight_decay']
        )
        
        # Initialize EMA
        self.ema = EMA(self.model, decay=config['training']['ema_decay'])
        
        # Learning rate scheduler
        self.scheduler = self._create_scheduler()
        
        # Mixed precision scaler
        self.scaler = GradScaler()
        
        # Perceptual loss (LPIPS)
        self.lpips_loss = lpips.LPIPS(net='alex').to(self.device)
        self.perceptual_weight = config['training']['perceptual_loss_weight']
        
        # TensorBoard
        self.writer = SummaryWriter(log_dir=config['training']['log_dir'])
        
        # Metrics
        self.metric_tracker = MetricTracker(device=self.device)
        
        # Training state
        self.epoch = 0
        self.global_step = 0
        self.best_psnr = 0.0
    
    def _create_model(self) -> nn.Module:
        """Create model based on config."""
        model_type = self.config['model']['type']
        
        if model_type == 'diffusion_field':
            model = DiffusionFieldModel(
                fourier_frequencies=self.config['model']['fourier_frequencies'],
                condition_dim=self.config['model']['condition_dim'],
                time_embed_dim=self.config['model']['time_embed_dim'],
                hidden_dim=self.config['model']['hidden_dim'],
                num_layers=self.config['model']['num_layers']
            )
        elif model_type == 'unet':
            model = UNet(
                in_channels=3,
                out_channels=3,
                channels=128,
                condition_dim=self.config['model']['condition_dim']
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        return model
    
    def _create_scheduler(self):
        """Create learning rate scheduler with warmup."""
        warmup_epochs = self.config['training']['warmup_epochs']
        total_epochs = self.config['training']['num_epochs']
        
        def lr_lambda(epoch):
            if epoch < warmup_epochs:
                return (epoch + 1) / warmup_epochs
            else:
                # Cosine annealing
                progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
                return 0.5 * (1.0 + np.cos(np.pi * progress))
        
        return optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
    
    def train_epoch(self, train_loader):
        """Train for one epoch."""
        self.model.train()
        epoch_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f'Epoch {self.epoch + 1}')
        for batch_idx, (low_light, normal_light) in enumerate(pbar):
            low_light = low_light.to(self.device)
            normal_light = normal_light.to(self.device)
            
            # Mixed precision training
            with autocast():
                loss = self._compute_loss(low_light, normal_light)
            
            # Backward pass
            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            
            # Gradient clipping
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config['training']['grad_clip']
            )
            
            # Optimizer step
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            # Update EMA
            self.ema.update()
            
            # Update metrics
            epoch_loss += loss.item()
            self.global_step += 1
            
            # Logging
            if self.global_step % 10 == 0:
                self.writer.add_scalar('train/loss', loss.item(), self.global_step)
                self.writer.add_scalar('train/lr', self.scheduler.get_last_lr()[0], self.global_step)
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        return epoch_loss / len(train_loader)
    
    def _compute_loss(self, low_light: torch.Tensor, normal_light: torch.Tensor) -> torch.Tensor:
        """Compute training loss."""
        batch_size = low_light.shape[0]
        
        # Sample random timesteps
        timesteps = self.noise_scheduler.sample_timesteps(batch_size)
        
        # Add noise to normal-light images
        noise = torch.randn_like(normal_light)
        noisy_images = self.noise_scheduler.add_noise(normal_light, noise, timesteps)
        
        # Predict noise
        if self.config['model']['type'] == 'unet':
            noise_pred = self.model(noisy_images, timesteps, low_light)
        else:
            # For diffusion field, we need coordinates
            from models.coord_encoder import create_coordinate_grid
            coords = create_coordinate_grid(
                normal_light.shape[2],
                normal_light.shape[3],
                device=self.device
            )
            noise_pred = self.model(low_light, coords, timesteps)
            # Reshape back to image space
            noise_pred = noise_pred.reshape(batch_size, normal_light.shape[2], normal_light.shape[3], 3)
            noise_pred = noise_pred.permute(0, 3, 1, 2)
        
        # MSE loss on noise prediction
        mse_loss = nn.functional.mse_loss(noise_pred, noise)
        
        # Optional perceptual loss
        if self.perceptual_weight > 0:
            # Compute predicted x_0
            alpha_t = self.noise_scheduler.sqrt_alphas_cumprod[timesteps]
            sigma_t = self.noise_scheduler.sqrt_one_minus_alphas_cumprod[timesteps]
            
            # Reshape for broadcasting
            alpha_t = alpha_t.view(-1, 1, 1, 1)
            sigma_t = sigma_t.view(-1, 1, 1, 1)
            
            pred_x0 = (noisy_images - sigma_t * noise_pred) / alpha_t
            pred_x0 = torch.clamp(pred_x0, -1, 1)
            
            # LPIPS expects [-1, 1] range
            normal_light_norm = normal_light * 2 - 1
            perceptual_loss = self.lpips_loss(pred_x0, normal_light_norm).mean()
            
            total_loss = mse_loss + self.perceptual_weight * perceptual_loss
        else:
            total_loss = mse_loss
        
        return total_loss
    
    @torch.no_grad()
    def validate(self, val_loader):
        """Validate the model."""
        self.model.eval()
        self.ema.apply_shadow()
        
        self.metric_tracker.reset()
        
        # Sample a few images for visualization
        vis_samples = []
        
        for batch_idx, (low_light, normal_light) in enumerate(tqdm(val_loader, desc='Validation')):
            low_light = low_light.to(self.device)
            normal_light = normal_light.to(self.device)
            
            # Generate enhanced images
            enhanced = self._sample(low_light)
            
            # Update metrics
            self.metric_tracker.update(enhanced, normal_light, compute_lpips=(batch_idx < 5))
            
            # Save some visualizations
            if batch_idx < 2:
                vis_samples.append((low_light[0], enhanced[0], normal_light[0]))
        
        # Compute average metrics
        metrics = self.metric_tracker.compute()
        
        # Log metrics
        self.writer.add_scalar('val/psnr', metrics['psnr'], self.epoch)
        self.writer.add_scalar('val/ssim', metrics['ssim'], self.epoch)
        if 'lpips' in metrics:
            self.writer.add_scalar('val/lpips', metrics['lpips'], self.epoch)
        
        # Save visualizations
        for idx, (low, enh, gt) in enumerate(vis_samples):
            save_comparison(
                low, enh, gt,
                save_path=f'outputs/train_vis/epoch_{self.epoch + 1}_sample_{idx}.png'
            )
        
        self.ema.restore()
        return metrics
    
    def _sample(self, low_light: torch.Tensor, num_steps: Optional[int] = None) -> torch.Tensor:
        """Sample enhanced images using DDIM."""
        if num_steps is None:
            num_steps = self.config['diffusion']['ddim_steps']
        
        batch_size = low_light.shape[0]
        image_size = low_light.shape[2:]
        
        # Start from random noise
        x = torch.randn(batch_size, 3, *image_size, device=self.device)
        
        # DDIM sampling
        timesteps = torch.linspace(
            self.config['diffusion']['num_timesteps'] - 1,
            0,
            num_steps,
            dtype=torch.long,
            device=self.device
        )
        
        for i, t in enumerate(tqdm(timesteps, desc='Sampling', leave=False)):
            t_batch = torch.full((batch_size,), t, device=self.device, dtype=torch.long)
            
            # Predict noise
            if self.config['model']['type'] == 'unet':
                noise_pred = self.model(x, t_batch, low_light)
            else:
                from models.coord_encoder import create_coordinate_grid
                coords = create_coordinate_grid(image_size[0], image_size[1], device=self.device)
                noise_pred = self.model(low_light, coords, t_batch)
                noise_pred = noise_pred.reshape(batch_size, image_size[0], image_size[1], 3)
                noise_pred = noise_pred.permute(0, 3, 1, 2)
            
            # DDIM step
            prev_t = timesteps[i + 1] if i < len(timesteps) - 1 else torch.tensor(-1, device=self.device)
            x = self.noise_scheduler.ddim_step(
                noise_pred, t.item(), x, eta=0.0,
                prev_timestep=prev_t.item() if prev_t >= 0 else None
            )
        
        # Clamp to [0, 1]
        x = torch.clamp(x, 0, 1)
        return x
    
    def save_checkpoint(self, is_best: bool = False):
        """Save checkpoint."""
        checkpoint = {
            'epoch': self.epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'ema_shadow': self.ema.shadow,
            'best_psnr': self.best_psnr,
            'config': self.config
        }
        
        # Save latest
        path = os.path.join(self.config['training']['checkpoint_dir'], 'latest.pth')
        torch.save(checkpoint, path)
        
        # Save best
        if is_best:
            path = os.path.join(self.config['training']['checkpoint_dir'], 'best.pth')
            torch.save(checkpoint, path)
        
        # Save periodic
        if (self.epoch + 1) % self.config['training']['save_every'] == 0:
            path = os.path.join(
                self.config['training']['checkpoint_dir'],
                f'checkpoint_epoch_{self.epoch + 1}.pth'
            )
            torch.save(checkpoint, path)
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.ema.shadow = checkpoint['ema_shadow']
        self.epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_psnr = checkpoint['best_psnr']
        print(f"Loaded checkpoint from epoch {self.epoch}")
    
    def train(self):
        """Main training loop."""
        # Create data loaders
        train_loader = get_dataloader(
            dataset_type=self.config['data']['dataset'],
            root_dir=self.config['data']['train_dir'],
            batch_size=self.config['training']['batch_size'],
            split='train',
            num_workers=self.config['data']['num_workers'],
            crop_size=self.config['data']['crop_size'],
            augment=self.config['data']['augment']
        )
        
        val_loader = get_dataloader(
            dataset_type=self.config['data']['dataset'],
            root_dir=self.config['data']['val_dir'],
            batch_size=self.config['inference']['batch_size'],
            split='val',
            num_workers=self.config['data']['num_workers'],
            crop_size=self.config['data']['crop_size'],
            augment=False
        )
        
        # Resume from checkpoint if specified
        if self.args.resume:
            self.load_checkpoint(self.args.resume)
        
        # Training loop
        num_epochs = self.config['training']['num_epochs']
        
        for epoch in range(self.epoch, num_epochs):
            self.epoch = epoch
            
            # Train
            train_loss = self.train_epoch(train_loader)
            print(f"Epoch {epoch + 1}/{num_epochs} - Train Loss: {train_loss:.4f}")
            
            # Update learning rate
            self.scheduler.step()
            
            # Validate
            if (epoch + 1) % self.config['training']['val_every'] == 0:
                metrics = self.validate(val_loader)
                print(f"Validation - PSNR: {metrics['psnr']:.2f}, SSIM: {metrics['ssim']:.4f}")
                
                # Check if best model
                is_best = metrics['psnr'] > self.best_psnr
                if is_best:
                    self.best_psnr = metrics['psnr']
                    print(f"New best PSNR: {self.best_psnr:.2f}")
                
                self.save_checkpoint(is_best=is_best)
            else:
                self.save_checkpoint(is_best=False)
        
        self.writer.close()
        print("Training completed!")


def main():
    parser = argparse.ArgumentParser(description='Train low-light enhancement model')
    parser.add_argument('--config', type=str, default='configs/default.yaml',
                        help='Path to config file')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Create trainer and train
    trainer = Trainer(config, args)
    trainer.train()


if __name__ == '__main__':
    main()
