# StyleGenie - Quick Start Guide

## Setup (First Time Only)

### 1. Activate Virtual Environment

```powershell
# Windows PowerShell
.\venv\Scripts\Activate.ps1

# You should see (venv) in your prompt
```

### 2. Verify Installation

```powershell
python -c "import torch; print(f'PyTorch {torch.__version__} installed successfully')"
```

## Usage

### Training a New Style Model

```powershell
# Basic training (uses default style image: edtaonisl.jpg)
python training_script.py

# Train with a different style image
python training_script.py --style_img_name mosaic.jpg

# Train with custom hyperparameters
python training_script.py --style_img_name candy.jpg --style_weight 5e5 --num_of_epochs 3

# Train without tensorboard
python training_script.py --disable_tensorboard

# Train with a subset for testing
python training_script.py --subset_size 1000 --num_of_epochs 1
```

**Note**: Training requires the MS COCO dataset. Download it first:
```powershell
python utils/resource_downloader.py --resource mscoco_dataset
```

### Stylizing Images

```powershell
# Stylize a single image (requires a trained model in models/binaries/)
python stylization_script.py --content_input taj_mahal.jpg --img_width 800

# Download pretrained models first if you don't have any
python utils/resource_downloader.py --resource pretrained_models

# Stylize with a specific model
python stylization_script.py --content_input lion.jpg --model_name mosaic_4e5_e2.pth

# Batch stylize all images in a directory
python stylization_script.py --content_input ./data/content-images/ --batch_size 5

# Verbose mode (shows model metadata)
python stylization_script.py --content_input taj_mahal.jpg --verbose
```

## Monitoring Training with TensorBoard

```powershell
# Start tensorboard (run in a separate terminal)
tensorboard --logdir=runs --samples_per_plugin images=50

# Open in browser
# Navigate to: http://localhost:6006/
```

## Common Parameters

### Training Script

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--style_img_name` | Style image filename (in `data/style-images/`) | `edtaonisl.jpg` |
| `--style_weight` | Weight for style loss | `4e5` |
| `--content_weight` | Weight for content loss | `1e0` |
| `--tv_weight` | Weight for total variation loss | `0` |
| `--num_of_epochs` | Number of training epochs | `2` |
| `--subset_size` | Number of MS COCO images to use | `None` (all) |
| `--enable_tensorboard` | Enable tensorboard logging | `False` (use flag to enable) |
| `--disable_tensorboard` | Disable tensorboard logging | `False` (use flag to disable) |
| `--checkpoint_freq` | Save checkpoint every N batches | `2000` |

### Stylization Script

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--content_input` | Image filename or directory | `taj_mahal.jpg` |
| `--model_name` | Model filename (in `models/binaries/`) | `mosaic_4e5_e2.pth` |
| `--img_width` | Output image width | `500` |
| `--batch_size` | Batch size for directory processing | `5` |
| `--verbose` | Print detailed information | `False` |

## Project Structure

```
StyleGenie/
├── venv/                      # Virtual environment (auto-generated)
├── data/
│   ├── content-images/        # Your input images
│   ├── style-images/          # Style images for training
│   ├── output-images/         # Stylized output images
│   └── mscoco/                # MS COCO training dataset
├── models/
│   ├── binaries/              # Trained model files (.pth)
│   ├── checkpoints/           # Training checkpoints
│   └── definitions/           # Model architecture definitions
├── utils/                     # Utility functions
├── training_script.py         # Main training script
├── stylization_script.py      # Main stylization script
├── requirements.txt           # Python dependencies
└── UPGRADE_NOTES.md           # Detailed upgrade information
```

## Tips & Tricks

### 1. Memory Issues
If you run out of memory:
- Reduce batch size: `--batch_size 2`
- Reduce image width: `--img_width 256`
- Use a subset for training: `--subset_size 10000`

### 2. Training Tips
- Start with small epochs to test: `--num_of_epochs 1`
- Adjust style weight to control stylization intensity (higher = more style)
- Use tensorboard to monitor loss curves
- Save checkpoints frequently for long training sessions

### 3. Best Results
- Use high-resolution style images (at least 512x512)
- Train for 2-3 epochs on full MS COCO dataset (~83k images)
- Experiment with style_weight between 1e5 and 1e6
- Keep content_weight at 1e0

## Troubleshooting

### "Module not found" error
Make sure the virtual environment is activated:
```powershell
.\venv\Scripts\Activate.ps1
```

### "MS COCO missing" error
Download the dataset:
```powershell
python utils/resource_downloader.py --resource mscoco_dataset
```

### "No models found" error
Download pretrained models or train your own:
```powershell
python utils/resource_downloader.py --resource pretrained_models
```

### CUDA out of memory
Your GPU ran out of memory. Reduce batch size or image size.

## Getting Help

1. Check [UPGRADE_NOTES.md](UPGRADE_NOTES.md) for detailed changes
2. Review the original [README.md](README.md) for more information
3. Check training/stylization parameters with `--help`:
   ```powershell
   python training_script.py --help
   python stylization_script.py --help
   ```

---

**Happy Styling! 🎨**

