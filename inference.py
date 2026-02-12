"""Inference script for low-light image enhancement.

Supports:
- Single image enhancement
- Batch inference on a directory
- DDPM and DDIM sampling
- Progress bars
- Side-by-side comparison outputs
"""

import os
import argparse
import yaml
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from typing import Optional
import torchvision.transforms.functional as TF

from models import DiffusionFieldModel, UNet, NoiseScheduler
from models.coord_encoder import create_coordinate_grid
from utils.visualization import save_comparison


class Enhancer:
    """Low-light image enhancer using diffusion models."""
    
    def __init__(
        self,
        checkpoint_path: str,
        config_path: Optional[str] = None,
        device: Optional[str] = None
    ):
        """Initialize enhancer.
        
        Args:
            checkpoint_path: Path to model checkpoint
            config_path: Optional path to config file (if not in checkpoint)
            device: Device to run on ('cuda' or 'cpu')
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
        
        # Load weights (handle both EMA and regular checkpoints)
        if 'ema_shadow' in checkpoint:
            # Load EMA weights
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
        
        print("Model loaded successfully!")
        print(f"Device: {self.device}")
    
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
        num_steps: Optional[int] = None,
        sampling_method: str = 'ddim',
        eta: float = 0.0
    ) -> torch.Tensor:
        """Enhance a low-light image.
        
        Args:
            low_light_image: Input image tensor of shape (1, 3, H, W) or (3, H, W)
            num_steps: Number of sampling steps (default: from config)
            sampling_method: 'ddim' or 'ddpm'
            eta: Noise level for DDIM (0.0 = deterministic)
        
        Returns:
            Enhanced image tensor of shape (1, 3, H, W) or (3, H, W)
        """
        # Ensure batch dimension
        if low_light_image.dim() == 3:
            low_light_image = low_light_image.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False
        
        low_light_image = low_light_image.to(self.device)
        batch_size = low_light_image.shape[0]
        image_size = low_light_image.shape[2:]
        
        # Set number of steps
        if num_steps is None:
            num_steps = self.config['inference']['sampling_steps']
        
        # Start from random noise
        x = torch.randn(batch_size, 3, *image_size, device=self.device)
        
        # Sampling
        if sampling_method == 'ddim':
            # DDIM sampling with skipped timesteps
            timesteps = torch.linspace(
                self.config['diffusion']['num_timesteps'] - 1,
                0,
                num_steps,
                dtype=torch.long,
                device=self.device
            )
            
            for i, t in enumerate(tqdm(timesteps, desc='DDIM Sampling', leave=False)):
                t_batch = torch.full((batch_size,), t, device=self.device, dtype=torch.long)
                
                # Predict noise
                noise_pred = self._predict_noise(x, low_light_image, t_batch)
                
                # DDIM step
                prev_t = timesteps[i + 1] if i < len(timesteps) - 1 else torch.tensor(-1, device=self.device)
                x = self.noise_scheduler.ddim_step(
                    noise_pred, t.item(), x, eta=eta,
                    prev_timestep=prev_t.item() if prev_t >= 0 else None
                )
        
        else:  # DDPM sampling
            for t in tqdm(range(self.config['diffusion']['num_timesteps'] - 1, -1, -1), desc='DDPM Sampling'):
                t_batch = torch.full((batch_size,), t, device=self.device, dtype=torch.long)
                
                # Predict noise
                noise_pred = self._predict_noise(x, low_light_image, t_batch)
                
                # DDPM step
                x = self.noise_scheduler.step(noise_pred, t, x)
        
        # Clamp to [0, 1]
        x = torch.clamp(x, 0, 1)
        
        if squeeze_output:
            x = x.squeeze(0)
        
        return x
    
    def _predict_noise(
        self,
        x: torch.Tensor,
        low_light: torch.Tensor,
        timesteps: torch.Tensor
    ) -> torch.Tensor:
        """Predict noise using the model."""
        if self.config['model']['type'] == 'unet':
            return self.model(x, timesteps, low_light)
        else:
            # Diffusion field
            batch_size = x.shape[0]
            image_size = x.shape[2:]
            coords = create_coordinate_grid(image_size[0], image_size[1], device=self.device)
            noise_pred = self.model(low_light, coords, timesteps)
            noise_pred = noise_pred.reshape(batch_size, image_size[0], image_size[1], 3)
            noise_pred = noise_pred.permute(0, 3, 1, 2)
            return noise_pred
    
    def enhance_image_file(
        self,
        image_path: str,
        output_path: Optional[str] = None,
        save_comparison: bool = True,
        **kwargs
    ) -> np.ndarray:
        """Enhance an image file.
        
        Args:
            image_path: Path to input image
            output_path: Path to save output (optional)
            save_comparison: Whether to save side-by-side comparison
            **kwargs: Additional arguments for enhance()
        
        Returns:
            Enhanced image as numpy array
        """
        # Load image
        img = Image.open(image_path).convert('RGB')
        img_tensor = TF.to_tensor(img).unsqueeze(0)
        
        # Enhance
        enhanced_tensor = self.enhance(img_tensor, **kwargs)
        
        # Convert to numpy
        enhanced_np = enhanced_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
        enhanced_np = (enhanced_np * 255).astype(np.uint8)
        
        # Save
        if output_path:
            if save_comparison:
                from utils.visualization import save_comparison as save_comp
                save_comp(img_tensor.squeeze(0), enhanced_tensor.squeeze(0), save_path=output_path)
            else:
                Image.fromarray(enhanced_np).save(output_path)
            print(f"Saved to {output_path}")
        
        return enhanced_np
    
    def enhance_directory(
        self,
        input_dir: str,
        output_dir: str,
        save_comparison: bool = True,
        **kwargs
    ):
        """Enhance all images in a directory.
        
        Args:
            input_dir: Directory containing input images
            output_dir: Directory to save output images
            save_comparison: Whether to save side-by-side comparisons
            **kwargs: Additional arguments for enhance()
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Get all image files
        image_files = [
            f for f in os.listdir(input_dir)
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))
        ]
        
        print(f"Found {len(image_files)} images in {input_dir}")
        
        # Process each image
        for img_file in tqdm(image_files, desc='Enhancing images'):
            input_path = os.path.join(input_dir, img_file)
            output_name = os.path.splitext(img_file)[0] + '_enhanced.png'
            output_path = os.path.join(output_dir, output_name)
            
            try:
                self.enhance_image_file(
                    input_path,
                    output_path,
                    save_comparison=save_comparison,
                    **kwargs
                )
            except Exception as e:
                print(f"Error processing {img_file}: {e}")
                continue
        
        print(f"Enhancement complete! Results saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='Enhance low-light images')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--config', type=str, default=None,
                        help='Path to config file (if not in checkpoint)')
    parser.add_argument('--input', type=str, required=True,
                        help='Input image or directory')
    parser.add_argument('--output', type=str, required=True,
                        help='Output image or directory')
    parser.add_argument('--sampling-method', type=str, default='ddim',
                        choices=['ddim', 'ddpm'],
                        help='Sampling method')
    parser.add_argument('--num-steps', type=int, default=50,
                        help='Number of sampling steps')
    parser.add_argument('--eta', type=float, default=0.0,
                        help='DDIM eta parameter (0.0 = deterministic)')
    parser.add_argument('--no-comparison', action='store_true',
                        help='Save only enhanced image, not comparison')
    parser.add_argument('--device', type=str, default=None,
                        help='Device to use (cuda or cpu)')
    
    args = parser.parse_args()
    
    # Initialize enhancer
    enhancer = Enhancer(args.checkpoint, args.config, args.device)
    
    # Check if input is file or directory
    if os.path.isfile(args.input):
        # Single image
        enhancer.enhance_image_file(
            args.input,
            args.output,
            save_comparison=not args.no_comparison,
            num_steps=args.num_steps,
            sampling_method=args.sampling_method,
            eta=args.eta
        )
    elif os.path.isdir(args.input):
        # Directory
        enhancer.enhance_directory(
            args.input,
            args.output,
            save_comparison=not args.no_comparison,
            num_steps=args.num_steps,
            sampling_method=args.sampling_method,
            eta=args.eta
        )
    else:
        print(f"Error: {args.input} is not a valid file or directory")


if __name__ == '__main__':
    main()
