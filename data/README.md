# Dataset Documentation

This directory contains dataset loaders and download scripts for low-light image enhancement.

## Supported Datasets

### 1. LOL (Low-Light) Dataset

The LOL dataset contains 500 paired low-light and normal-light images:
- **Training set**: 485 pairs in `our485/`
- **Test set**: 15 pairs in `eval15/`

#### Download and Setup

Run the download script:
```bash
cd data
./download_lol.sh
```

**Note**: The LOL dataset requires manual download due to hosting restrictions.

1. Visit the Google Drive link: https://drive.google.com/file/d/157bjO1_cFuSd0HWDUuAmcHRJDVyWpOxB/view
2. Download `LOLdataset.zip`
3. Place it in the `data/` directory
4. Run `./download_lol.sh` again to extract

#### Expected Directory Structure

```
data/LOL/
├── our485/
│   ├── low/    # 485 low-light training images
│   └── high/   # 485 normal-light training images
└── eval15/
    ├── low/    # 15 low-light test images
    └── high/   # 15 normal-light test images
```

### 2. LOL-v2 Dataset

LOL-v2 is an extended version with both real and synthetic low-light image pairs (~1000+ pairs).

#### Download

1. Visit: https://github.com/flyywh/CVPR-2020-Semi-Low-Light
2. Download LOL-v2 (Real and Synthetic subsets)
3. Extract to `data/LOL-v2/`

#### Expected Directory Structure

```
data/LOL-v2/
├── Real_captured/
│   ├── Train/
│   │   ├── Low/
│   │   └── Normal/
│   └── Test/
│       ├── Low/
│       └── Normal/
└── Synthetic/
    ├── Train/
    │   ├── Low/
    │   └── Normal/
    └── Test/
        ├── Low/
        └── Normal/
```

### 3. Synthetic Low-Light Pairs

Generate synthetic low-light images from any well-lit image dataset (e.g., DIV2K, ImageNet).

The synthetic generator applies:
- Random gamma correction (γ ∈ [2.0, 5.0])
- Brightness reduction (factor ∈ [0.3, 0.7])
- Optional Gaussian/Poisson noise injection

#### Using Synthetic Data

```python
from data.dataset import SyntheticLowLightDataset

dataset = SyntheticLowLightDataset(
    image_dir='path/to/normal_images',
    crop_size=256,
    gamma_range=(2.0, 5.0),
    brightness_range=(0.3, 0.7),
    add_noise=True
)
```

## Using the Dataset Loaders

### LOL Dataset

```python
from data.dataset import LOLDataset, get_dataloader

# Create dataset
train_dataset = LOLDataset(
    root_dir='data/LOL/our485',
    split='train',
    crop_size=256,
    augment=True
)

# Create dataloader
train_loader = get_dataloader(
    dataset_type='LOL',
    root_dir='data/LOL/our485',
    batch_size=8,
    split='train',
    num_workers=4
)
```

### Data Augmentation

Training data augmentation includes:
- Random cropping (256×256)
- Random horizontal flips
- Random vertical flips
- Random 90° rotations

## Dataset Statistics

### LOL Dataset

| Subset | # Pairs | Image Size | Format |
|--------|---------|------------|--------|
| Train  | 485     | Variable   | PNG    |
| Test   | 15      | 400×600    | PNG    |

### File Size

- LOL dataset: ~500 MB (compressed)
- LOL-v2 dataset: ~2 GB (compressed)

## Citation

If you use the LOL dataset, please cite:

```bibtex
@inproceedings{wei2018deep,
  title={Deep retinex decomposition for low-light enhancement},
  author={Wei, Chen and Wang, Wenjing and Yang, Wenhan and Liu, Jiaying},
  booktitle={British Machine Vision Conference (BMVC)},
  year={2018}
}
```

For LOL-v2:

```bibtex
@inproceedings{yang2020fidelity,
  title={From fidelity to perceptual quality: A semi-supervised approach for low-light image enhancement},
  author={Yang, Wenhan and Wang, Wenjing and Huang, Haofeng and Wang, Shiqi and Liu, Jiaying},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={3063--3072},
  year={2020}
}
```
