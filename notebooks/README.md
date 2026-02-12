# DFLLIV Notebooks

This directory contains interactive Jupyter notebooks for training and experimenting with the DFLLIV model.

## 📓 Available Notebooks

### DFLLIV_Colab_Training.ipynb

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/DCReddy23/DFLLIV/blob/main/notebooks/DFLLIV_Colab_Training.ipynb)

**Complete Google Colab training notebook** - Train DFLLIV on the LOL dataset using Google Colab's free GPU.

**What's Included:**
- ✅ Environment setup and dependencies
- ✅ LOL dataset download and preparation  
- ✅ Training with T4 GPU (~4-6 hours for 100 epochs)
- ✅ Inference and visualization
- ✅ Model evaluation (PSNR, SSIM, LPIPS)
- ✅ Checkpoint management with Google Drive

**Best For:**
- First-time users wanting to try the model
- Training without local GPU setup
- Quick experimentation and prototyping
- Learning and educational purposes

## 🚀 Quick Start

### Option 1: Direct Launch (Easiest)

Click the "Open in Colab" badge above to launch the notebook directly.

### Option 2: From Google Colab

1. Go to [Google Colab](https://colab.research.google.com)
2. File → Open Notebook
3. Select "GitHub" tab
4. Enter repository: `DCReddy23/DFLLIV`
5. Choose: `notebooks/DFLLIV_Colab_Training.ipynb`

### Option 3: Local Jupyter

```bash
# Clone repository
git clone https://github.com/DCReddy23/DFLLIV.git
cd DFLLIV

# Install Jupyter
pip install jupyter

# Launch notebook
jupyter notebook notebooks/DFLLIV_Colab_Training.ipynb
```

**Note:** Local execution requires a CUDA GPU and all dependencies installed.

## 📚 Documentation

- **[COLAB_GUIDE.md](COLAB_GUIDE.md)** - Comprehensive guide for Google Colab training
- **[../README.md](../README.md)** - Main project documentation
- **[../configs/colab.yaml](../configs/colab.yaml)** - Colab-optimized configuration

## 💡 Tips

### First Time Users

1. Start with Google Colab (no setup required)
2. Follow the notebook cells in order
3. Enable GPU runtime in Colab
4. Allow 4-6 hours for training (100 epochs)

### Experienced Users

Customize the training:
- Adjust epochs in `configs/colab.yaml`
- Change batch size based on available GPU memory
- Try different model architectures (DiffusionField vs UNet)
- Experiment with hyperparameters

### For Best Results

- Train for 300-500 epochs (requires longer session or multiple runs)
- Use larger batch size (8-16) if GPU memory allows
- Enable mixed precision training (already configured)
- Save checkpoints to Google Drive for persistence

## ⏱️ Training Time Reference

| Configuration | GPU | Time (100 epochs) |
|--------------|-----|------------------|
| Colab Default | T4 (Free) | 4-6 hours |
| Colab Pro | V100 | 2-3 hours |
| Colab Pro+ | A100 | 1-2 hours |
| Local | RTX 3060 (6GB) | 6-8 hours |
| Local | RTX 3090 (24GB) | 2-3 hours |

## 🔧 Troubleshooting

### Common Issues

**1. "No GPU available"**
- Solution: Runtime → Change runtime type → Select GPU

**2. "Out of memory"**
- Solution: Reduce batch size to 2 or 1
- Or reduce crop_size to 128

**3. "Dataset download failed"**
- Solution: Download manually and upload to Google Drive
- See COLAB_GUIDE.md for instructions

**4. "Session disconnected"**
- Solution: Resume with latest checkpoint
- All checkpoints automatically save to Google Drive

## 📊 Expected Results

After 100 epochs of training:

**Metrics:**
- PSNR: ~18-22 dB
- SSIM: ~0.75-0.85  
- LPIPS: ~0.15-0.25

**Visual Quality:**
- Noticeable brightness enhancement
- Detail preservation
- Reduced noise

For better results, train longer (300-500 epochs).

## 🆘 Need Help?

1. Check [COLAB_GUIDE.md](COLAB_GUIDE.md) for detailed instructions
2. Review troubleshooting section above
3. Read main [README.md](../README.md)
4. Open an issue on [GitHub](https://github.com/DCReddy23/DFLLIV/issues)

## 🤝 Contributing

Want to add more notebooks? Contributions welcome!

Ideas for future notebooks:
- Advanced training techniques
- Custom dataset preparation
- Model architecture experiments
- Inference optimization
- Video enhancement

## 📝 License

All notebooks are licensed under MIT License, same as the main project.

---

**Ready to start? Click the Colab badge above! 🚀**
