"""Evaluation script for low-light image enhancement.

Computes PSNR, SSIM, and LPIPS metrics on test datasets.
Generates visual comparison grids and saves results to CSV/JSON.
"""

import os
import argparse
import yaml
import json
import csv
import torch
import numpy as np
from tqdm import tqdm
from typing import Dict, List

from models import DiffusionFieldModel, UNet, NoiseScheduler
from models.coord_encoder import create_coordinate_grid
from data.dataset import LOLDataset
from utils.metrics import calculate_psnr, calculate_ssim, LPIPSMetric
from utils.visualization import create_enhancement_comparison_grid


class Evaluator:
    """Evaluator for low-light enhancement models."""
    
    def __init__(
        self,
        checkpoint_path: str,
        config_path: str = None,
        device: str = None
    ):
        """Initialize evaluator.
        
        Args:
            checkpoint_path: Path to model checkpoint
            config_path: Optional path to config file
            device: Device to run on
        """
        self.device = torch.device(device if device else ('cuda' if torch.cuda.is_available() else 'cpu'))
        
        # Load checkpoint
        print(f"Loading checkpoint from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Load config
        if 'config' in checkpoint:
            self.config = checkpoint['config']
        elif config_path:
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
        else:
            raise ValueError("No config found in checkpoint and no config_path provided")
        
        # Initialize model
        self.model = self._create_model()
        
        # Load weights
        if 'ema_shadow' in checkpoint:
            state_dict = {}
            for name, param in self.model.named_parameters():
                if name in checkpoint['ema_shadow']:
                    state_dict[name] = checkpoint['ema_shadow'][name]
            self.model.load_state_dict(state_dict, strict=False)
        else:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        
        self.model.to(self.device)
        self.model.eval()
        
        # Initialize noise scheduler
        self.noise_scheduler = NoiseScheduler(
            num_timesteps=self.config['diffusion']['num_timesteps'],
            beta_schedule=self.config['diffusion']['beta_schedule'],
            beta_start=self.config['diffusion']['beta_start'],
            beta_end=self.config['diffusion']['beta_end'],
            device=self.device
        )
        
        # Initialize LPIPS metric
        self.lpips_metric = LPIPSMetric(device=self.device)
        
        print("Model loaded successfully!")
    
    def _create_model(self):
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
    
    @torch.no_grad()
    def enhance(
        self,
        low_light_image: torch.Tensor,
        num_steps: int = 50,
        eta: float = 0.0
    ) -> torch.Tensor:
        """Enhance a low-light image using DDIM sampling.
        
        Args:
            low_light_image: Input image tensor (B, 3, H, W)
            num_steps: Number of sampling steps
            eta: DDIM eta parameter
        
        Returns:
            Enhanced image tensor
        """
        batch_size = low_light_image.shape[0]
        image_size = low_light_image.shape[2:]
        
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
        
        for i, t in enumerate(timesteps):
            t_batch = torch.full((batch_size,), t, device=self.device, dtype=torch.long)
            
            # Predict noise
            if self.config['model']['type'] == 'unet':
                noise_pred = self.model(x, t_batch, low_light_image)
            else:
                coords = create_coordinate_grid(image_size[0], image_size[1], device=self.device)
                noise_pred = self.model(low_light_image, coords, t_batch)
                noise_pred = noise_pred.reshape(batch_size, image_size[0], image_size[1], 3)
                noise_pred = noise_pred.permute(0, 3, 1, 2)
            
            # DDIM step
            prev_t = timesteps[i + 1] if i < len(timesteps) - 1 else torch.tensor(-1, device=self.device)
            x = self.noise_scheduler.ddim_step(
                noise_pred, t.item(), x, eta=eta,
                prev_timestep=prev_t.item() if prev_t >= 0 else None
            )
        
        # Clamp to [0, 1]
        x = torch.clamp(x, 0, 1)
        return x
    
    def evaluate(
        self,
        dataset_dir: str,
        output_dir: str,
        num_steps: int = 50,
        save_images: bool = True,
        max_images: int = None
    ) -> Dict[str, float]:
        """Evaluate on a test dataset.
        
        Args:
            dataset_dir: Directory containing test dataset
            output_dir: Directory to save results
            num_steps: Number of sampling steps
            save_images: Whether to save enhanced images
            max_images: Maximum number of images to evaluate (None = all)
        
        Returns:
            Dictionary of average metrics
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Load dataset
        print(f"Loading dataset from {dataset_dir}...")
        dataset = LOLDataset(dataset_dir, split='val', crop_size=None, augment=False)
        
        if max_images:
            # Limit dataset size
            dataset.low_images = dataset.low_images[:max_images]
        
        print(f"Evaluating on {len(dataset)} images...")
        
        # Storage for metrics
        results = []
        low_light_samples = []
        enhanced_samples = []
        ground_truth_samples = []
        
        # Evaluate each image
        for idx in tqdm(range(len(dataset)), desc='Evaluating'):
            low_light, ground_truth = dataset[idx]
            
            # Add batch dimension and move to device
            low_light_batch = low_light.unsqueeze(0).to(self.device)
            ground_truth_batch = ground_truth.unsqueeze(0).to(self.device)
            
            # Enhance
            enhanced = self.enhance(low_light_batch, num_steps=num_steps)
            
            # Compute metrics
            psnr = calculate_psnr(enhanced, ground_truth_batch)
            ssim = calculate_ssim(enhanced, ground_truth_batch)
            lpips = self.lpips_metric(enhanced, ground_truth_batch, normalize=True)
            
            # Store results
            result = {
                'image_idx': idx,
                'image_name': dataset.low_images[idx],
                'psnr': psnr,
                'ssim': ssim,
                'lpips': lpips
            }
            results.append(result)
            
            # Save enhanced image
            if save_images:
                from utils.visualization import save_comparison
                save_path = os.path.join(
                    output_dir,
                    f'enhanced_{idx:03d}.png'
                )
                save_comparison(
                    low_light,
                    enhanced.squeeze(0),
                    ground_truth,
                    save_path=save_path
                )
            
            # Store samples for grid
            if idx < 8:
                low_light_samples.append(low_light)
                enhanced_samples.append(enhanced.squeeze(0))
                ground_truth_samples.append(ground_truth)
        
        # Compute average metrics
        avg_metrics = {
            'psnr': np.mean([r['psnr'] for r in results]),
            'ssim': np.mean([r['ssim'] for r in results]),
            'lpips': np.mean([r['lpips'] for r in results]),
            'psnr_std': np.std([r['psnr'] for r in results]),
            'ssim_std': np.std([r['ssim'] for r in results]),
            'lpips_std': np.std([r['lpips'] for r in results])
        }
        
        # Print results
        print("\n" + "=" * 50)
        print("Evaluation Results:")
        print("=" * 50)
        print(f"PSNR:  {avg_metrics['psnr']:.2f} ± {avg_metrics['psnr_std']:.2f} dB")
        print(f"SSIM:  {avg_metrics['ssim']:.4f} ± {avg_metrics['ssim_std']:.4f}")
        print(f"LPIPS: {avg_metrics['lpips']:.4f} ± {avg_metrics['lpips_std']:.4f}")
        print("=" * 50)
        
        # Save per-image results to CSV
        csv_path = os.path.join(output_dir, 'results.csv')
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
        print(f"\nPer-image results saved to {csv_path}")
        
        # Save average metrics to JSON
        json_path = os.path.join(output_dir, 'metrics.json')
        with open(json_path, 'w') as f:
            json.dump(avg_metrics, f, indent=4)
        print(f"Average metrics saved to {json_path}")
        
        # Create comparison grid
        if low_light_samples:
            grid_path = os.path.join(output_dir, 'comparison_grid.png')
            create_enhancement_comparison_grid(
                low_light_samples,
                enhanced_samples,
                ground_truth_samples,
                save_path=grid_path
            )
        
        return avg_metrics


def main():
    parser = argparse.ArgumentParser(description='Evaluate low-light enhancement model')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--config', type=str, default=None,
                        help='Path to config file')
    parser.add_argument('--dataset-dir', type=str, required=True,
                        help='Path to test dataset directory')
    parser.add_argument('--output-dir', type=str, default='results',
                        help='Directory to save results')
    parser.add_argument('--num-steps', type=int, default=50,
                        help='Number of sampling steps')
    parser.add_argument('--max-images', type=int, default=None,
                        help='Maximum number of images to evaluate')
    parser.add_argument('--no-save-images', action='store_true',
                        help='Do not save enhanced images')
    parser.add_argument('--device', type=str, default=None,
                        help='Device to use (cuda or cpu)')
    
    args = parser.parse_args()
    
    # Initialize evaluator
    evaluator = Evaluator(args.checkpoint, args.config, args.device)
    
    # Evaluate
    evaluator.evaluate(
        dataset_dir=args.dataset_dir,
        output_dir=args.output_dir,
        num_steps=args.num_steps,
        save_images=not args.no_save_images,
        max_images=args.max_images
    )


if __name__ == '__main__':
    main()
