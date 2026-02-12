# Google Colab Training Guide

This guide provides step-by-step instructions for training DFLLIV on Google Colab with the LOL dataset.

## 🎯 Quick Start

### Option 1: One-Click Launch (Recommended)

Click this badge to open the notebook directly in Google Colab:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/DCReddy23/DFLLIV/blob/main/notebooks/DFLLIV_Colab_Training.ipynb)

### Option 2: Manual Setup

1. Go to [Google Colab](https://colab.research.google.com)
2. File → Open Notebook → GitHub
3. Enter: `DCReddy23/DFLLIV`
4. Select: `notebooks/DFLLIV_Colab_Training.ipynb`

## 📋 Prerequisites

- **Google Account** - For Colab access
- **Google Drive** - For dataset and checkpoint storage (~2GB free space)
- **GPU Runtime** - Enable in Colab: Runtime → Change runtime type → GPU

## ⏱️ Training Time Estimates

| GPU Type | 100 Epochs | 500 Epochs |
|----------|------------|------------|
| T4 (Free) | 4-6 hours | 20-30 hours |
| V100 (Pro) | 2-3 hours | 10-15 hours |
| A100 (Pro+) | 1-2 hours | 5-8 hours |

**Note:** Colab free tier disconnects after ~12 hours. Use checkpointing to resume training.

## 📝 Step-by-Step Instructions

### 1. Enable GPU Runtime

Before running any code:
1. Click **Runtime** → **Change runtime type**
2. Select **GPU** under Hardware accelerator
3. Click **Save**

Verify GPU availability by running:
```python
!nvidia-smi
```

### 2. Mount Google Drive

The notebook will save checkpoints to your Google Drive for persistence:
```python
from google.colab import drive
drive.mount('/content/drive')
```

### 3. Download LOL Dataset

**Option A: Via Google Drive (Recommended)**
1. Download LOL dataset from [official link](https://drive.google.com/file/d/157bjO1_cFuSd0HWDUuAmcHRJDVyWpOxB/view)
2. Upload `LOLdataset.zip` to `MyDrive/DFLLIV/`
3. Run the dataset setup cell in the notebook

**Option B: Direct Download**
The notebook includes code to attempt direct download using `gdown`.

### 4. Configure Training

The default Colab configuration (`configs/colab.yaml`) is optimized for T4 GPU:

```yaml
training:
  batch_size: 4        # Reduced for memory efficiency
  num_epochs: 100      # For reasonable training time
  val_every: 5         # Validate every 5 epochs
  save_every: 25       # Save checkpoint every 25 epochs
```

**Adjust for your needs:**

- **Quick test (10 minutes):** 5-10 epochs
- **Demo training (1-2 hours):** 20-50 epochs  
- **Good results (4-6 hours):** 100 epochs
- **Best results (20-30 hours):** 300-500 epochs

### 5. Start Training

Run the training cell:
```bash
!python train.py --config configs/colab.yaml
```

**Monitor Progress:**
- Training loss and metrics appear in real-time
- TensorBoard available for detailed monitoring
- Validation runs every 5 epochs with visual samples

### 6. Resume Training (if disconnected)

If Colab disconnects, resume from the last checkpoint:
```bash
!python train.py --config configs/colab.yaml \
    --resume /content/drive/MyDrive/DFLLIV/checkpoints/latest.pth
```

### 7. Run Inference

After training, enhance test images:
```bash
!python inference.py \
    --checkpoint /content/drive/MyDrive/DFLLIV/checkpoints/best.pth \
    --input data/LOL/eval15/low \
    --output outputs/enhanced \
    --num-steps 50
```

### 8. Evaluate Results

Compute metrics on the test set:
```bash
!python evaluate.py \
    --checkpoint /content/drive/MyDrive/DFLLIV/checkpoints/best.pth \
    --dataset-dir data/LOL/eval15 \
    --output-dir results
```

## 🔧 Troubleshooting

### Out of Memory Error

**Problem:** `CUDA out of memory` error during training

**Solutions:**
1. Reduce batch size:
   ```python
   config['training']['batch_size'] = 2  # or even 1
   ```
2. Reduce crop size:
   ```python
   config['data']['crop_size'] = 128  # instead of 256
   ```
3. Use UNet instead of DiffusionField (smaller model):
   ```python
   config['model']['type'] = 'unet'
   ```

### Session Timeout

**Problem:** Colab disconnects after 12 hours (free tier)

**Solutions:**
1. Save checkpoints frequently (already configured)
2. Resume training from latest checkpoint
3. Upgrade to Colab Pro for longer sessions

### Dataset Download Issues

**Problem:** Automatic download fails

**Solutions:**
1. Download manually from [Google Drive](https://drive.google.com/file/d/157bjO1_cFuSd0HWDUuAmcHRJDVyWpOxB/view)
2. Upload to your Google Drive
3. Use the manual extraction code in the notebook

### Slow Training

**Problem:** Training seems slower than expected

**Check:**
1. GPU is enabled (not CPU): `!nvidia-smi`
2. Batch size isn't too small (use 4 or 8 if GPU allows)
3. `num_workers` is set to 2-4 in config

## 💡 Tips for Best Results

### 1. Start with Quick Test

Before long training, do a quick test:
```python
config['training']['num_epochs'] = 5
config['training']['val_every'] = 1
```

### 2. Monitor Training

Use TensorBoard to monitor:
- Training/validation loss curves
- PSNR/SSIM metrics
- Visual samples during validation

### 3. Experiment with Hyperparameters

- **Learning rate:** Try 5e-5 or 2e-4 if training is unstable
- **Batch size:** Increase to 8 if GPU memory allows
- **Model type:** Try UNet if DiffusionField is too slow

### 4. Save Results to Drive

All outputs automatically save to Google Drive:
- Checkpoints: `MyDrive/DFLLIV/checkpoints/`
- Logs: `MyDrive/DFLLIV/runs/`
- Outputs: `MyDrive/DFLLIV/outputs/`

### 5. Use Best Checkpoint

For inference, use `best.pth` (best validation PSNR) rather than `latest.pth`.

## 📊 Expected Results

After 100 epochs of training on LOL dataset:

- **PSNR:** ~18-22 dB (higher is better)
- **SSIM:** ~0.75-0.85 (closer to 1 is better)
- **LPIPS:** ~0.15-0.25 (lower is better)

After 500 epochs:
- **PSNR:** ~22-25 dB
- **SSIM:** ~0.80-0.90
- **LPIPS:** ~0.10-0.20

## 🔗 Useful Links

- [Main Repository](https://github.com/DCReddy23/DFLLIV)
- [Colab Notebook](https://colab.research.google.com/github/DCReddy23/DFLLIV/blob/main/notebooks/DFLLIV_Colab_Training.ipynb)
- [LOL Dataset](https://drive.google.com/file/d/157bjO1_cFuSd0HWDUuAmcHRJDVyWpOxB/view)
- [Project Documentation](https://github.com/DCReddy23/DFLLIV/blob/main/README.md)

## ❓ FAQ

**Q: Can I use Colab's free tier?**  
A: Yes! The notebook is optimized for free T4 GPU. Training 100 epochs takes 4-6 hours.

**Q: How do I save my work?**  
A: All checkpoints and results automatically save to your Google Drive.

**Q: What if Colab disconnects during training?**  
A: Resume with `--resume /content/drive/MyDrive/DFLLIV/checkpoints/latest.pth`

**Q: How much Google Drive space do I need?**  
A: ~2GB (500MB for dataset, ~1GB for checkpoints, ~500MB for outputs)

**Q: Can I train on my own images?**  
A: Yes! Use the synthetic dataset mode or prepare your own paired images.

**Q: How do I download my trained model?**  
A: Access it from Google Drive at `MyDrive/DFLLIV/checkpoints/best.pth`

## 🤝 Support

If you encounter issues:
1. Check this guide's Troubleshooting section
2. Review the [main README](https://github.com/DCReddy23/DFLLIV/blob/main/README.md)
3. Open an issue on [GitHub](https://github.com/DCReddy23/DFLLIV/issues)

---

**Happy Training! 🚀**
