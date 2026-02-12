#!/bin/bash

# Download script for LOL (Low-Light) Dataset
# This script downloads and extracts the LOL dataset for low-light image enhancement

set -e  # Exit on error

echo "====================================="
echo "LOL Dataset Download Script"
echo "====================================="

# Create data directory
mkdir -p data
cd data

# Check if LOL directory already exists
if [ -d "LOL" ]; then
    echo "LOL dataset directory already exists."
    read -p "Do you want to re-download? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Skipping download."
        exit 0
    fi
    rm -rf LOL
fi

echo "Downloading LOL dataset..."
echo "Note: The LOL dataset is hosted on various sources. This script uses a common mirror."

# LOL dataset is typically hosted on Google Drive or similar services
# We'll provide instructions for manual download

echo ""
echo "MANUAL DOWNLOAD REQUIRED:"
echo "========================="
echo "The LOL dataset needs to be downloaded manually due to hosting restrictions."
echo ""
echo "Please follow these steps:"
echo "1. Visit: https://drive.google.com/file/d/157bjO1_cFuSd0HWDUuAmcHRJDVyWpOxB/view"
echo "2. Download the 'LOLdataset.zip' file"
echo "3. Place it in the 'data/' directory: $(pwd)"
echo "4. Run this script again to extract"
echo ""

# Check if zip file exists
if [ -f "LOLdataset.zip" ]; then
    echo "Found LOLdataset.zip, extracting..."
    unzip -q LOLdataset.zip
    
    # Organize into expected structure
    if [ -d "lol_dataset" ] || [ -d "LOLdataset" ]; then
        # Different versions may have different names
        if [ -d "lol_dataset" ]; then
            mv lol_dataset LOL
        elif [ -d "LOLdataset" ]; then
            mv LOLdataset LOL
        fi
    fi
    
    # Check structure
    if [ -d "LOL/our485" ] && [ -d "LOL/eval15" ]; then
        echo "✓ Dataset extracted successfully!"
        echo ""
        echo "Dataset structure:"
        tree -L 2 LOL/ 2>/dev/null || ls -R LOL/
        echo ""
        echo "Training images: $(ls LOL/our485/low/ 2>/dev/null | wc -l)"
        echo "Test images: $(ls LOL/eval15/low/ 2>/dev/null | wc -l)"
    else
        echo "⚠ Warning: Dataset structure may not be as expected."
        echo "Please check the LOL directory structure."
    fi
    
    # Optionally remove zip file
    read -p "Remove LOLdataset.zip? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        rm LOLdataset.zip
        echo "Removed LOLdataset.zip"
    fi
else
    echo "LOLdataset.zip not found. Please download it manually first."
    exit 1
fi

echo ""
echo "====================================="
echo "Dataset setup complete!"
echo "====================================="
