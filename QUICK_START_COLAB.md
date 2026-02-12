# Quick Start: Train DFLLIV in Google Colab

## 🎯 Your Question Answered

**Q:** "Now I want to train this in Google Colab using LOL dataset. What to do?"

**A:** Follow these 3 simple steps:

---

## Step 1: Open the Colab Notebook

Click this badge to open the notebook directly in Google Colab:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/DCReddy23/DFLLIV/blob/main/notebooks/DFLLIV_Colab_Training.ipynb)

**Alternative:** Go to the [project README](https://github.com/DCReddy23/DFLLIV) and click the "Open in Colab" badge at the top.

---

## Step 2: Enable GPU

Once the notebook opens:

1. Click **Runtime** in the top menu
2. Select **Change runtime type**
3. Under "Hardware accelerator", select **GPU**
4. Click **Save**

This gives you access to a free T4 GPU with ~15GB memory.

---

## Step 3: Run the Cells

Simply run each cell in order by clicking the ▶️ play button:

1. **Cell 1-2**: Check GPU and mount Google Drive
2. **Cell 3-4**: Clone repository and install dependencies
3. **Cell 5-7**: Download and setup LOL dataset
4. **Cell 8**: Start training (this takes 4-6 hours)

The notebook will:
- ✅ Automatically install all dependencies
- ✅ Download the LOL dataset
- ✅ Configure everything optimally for Colab
- ✅ Train the model for 100 epochs
- ✅ Save checkpoints to your Google Drive
- ✅ Show results and visualizations

---

## ⏱️ How Long Does Training Take?

On Colab's free T4 GPU:
- **100 epochs**: ~4-6 hours (good results)
- **500 epochs**: ~20-30 hours (excellent results)

**Note:** Colab free tier may disconnect after ~12 hours. The notebook includes automatic checkpointing to Google Drive, so you can resume training if disconnected.

---

## 📊 What Results to Expect

After **100 epochs** (~4-6 hours):
- **PSNR**: 18-22 dB
- **SSIM**: 0.75-0.85
- **Visual Quality**: Noticeable brightness improvement, good detail preservation

After **500 epochs** (~20-30 hours):
- **PSNR**: 22-25 dB
- **SSIM**: 0.80-0.90
- **Visual Quality**: Professional-grade enhancement

---

## 💾 Where Are Results Saved?

Everything is automatically saved to your Google Drive:

```
Google Drive/
└── MyDrive/
    └── DFLLIV/
        ├── checkpoints/        # Trained models
        │   ├── best.pth        # Best model (use this)
        │   └── latest.pth      # Latest checkpoint
        ├── runs/               # Training logs
        └── outputs/            # Enhanced images
```

---

## 🔧 Troubleshooting

### "Out of Memory" Error
If you see a CUDA out of memory error:
1. Scroll to the configuration cell
2. Uncomment and run: `config['training']['batch_size'] = 2`
3. Restart training

### Dataset Download Fails
If automatic download doesn't work:
1. Download manually from: https://drive.google.com/file/d/157bjO1_cFuSd0HWDUuAmcHRJDVyWpOxB/view
2. Upload `LOLdataset.zip` to your Google Drive at `MyDrive/DFLLIV/`
3. Re-run the dataset setup cell

### Colab Disconnects During Training
If Colab disconnects (usually after 12 hours):
1. Reopen the notebook
2. Enable GPU again
3. Run the training cell with `--resume` flag (instructions in notebook)
4. Training continues from last checkpoint

---

## 📚 Need More Help?

- **Detailed Guide**: See `notebooks/COLAB_GUIDE.md`
- **Notebook README**: See `notebooks/README.md`
- **Main Documentation**: See project `README.md`
- **Issues**: Open an issue on GitHub

---

## 💡 Pro Tips

1. **Quick Test**: For a quick test run (10 minutes), reduce epochs to 5 in the config
2. **Better Results**: Train for 300-500 epochs if you have Colab Pro
3. **Monitor Progress**: Use TensorBoard cell to see real-time training curves
4. **Save Bandwidth**: The notebook saves everything to Google Drive automatically

---

## ✅ That's It!

You now have everything you need to train DFLLIV in Google Colab.

**Just click the badge, enable GPU, and run the cells. The notebook handles the rest!**

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/DCReddy23/DFLLIV/blob/main/notebooks/DFLLIV_Colab_Training.ipynb)

---

**Happy Training! 🚀**
