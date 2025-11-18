# Anime character segmentation with UNet
![rr](https://user-images.githubusercontent.com/117014820/229329002-a16a5e17-7323-4f0e-898a-48f291ee6157.jpg)

Classes : [ background, hair, eye, mouth, face, skin, clothes, others ]

## Model
![1_Xp3R17gR8PL0SA0inUplcg](https://user-images.githubusercontent.com/117014820/229331131-181bbe04-259f-4649-926c-c8916a5508e3.jpg)

In: 3x512x512 -> Out: 8x512x512

It uses pretrained mobilenet_v2 as encoder

## Training

### Quick Start
```powershell
# Basic training (auto-detects GPU, uses 4 workers)
.\train_start.bat
```

### Advanced Options
```powershell
# Resume from checkpoint and save every 500 images
python -u train.py --resume --checkpoint_interval 500 --device cuda --num_workers 4

# Auto-compute class weights for imbalanced classes (recommended for rare classes like 'others')
python -u train.py --use_class_weights --device cuda

# Use pre-computed class weights from analysis
python -u scripts\compute_class_weights.py --mask_dir .\dataset\mask --output class_weights.json
python -u train.py --class_weights_file class_weights.json --device cuda

# Custom sample image for per-epoch visualization
python -u train.py --sample_image .\test_image\Hoshino.webp --device cuda

# CPU training (slower)
python -u train.py --device cpu --num_workers 0
```

### Handling Imbalanced Classes (Rare Accessories/Hats)

If some classes (like 'others' for hats/glasses) appear rarely:

1. **Analyze class distribution**:
```powershell
python -u scripts\compute_class_weights.py --mask_dir .\dataset\mask --output class_weights.json
```

2. **Train with automatic class weighting** (recommended):
```powershell
python -u train.py --use_class_weights --device cuda
```

3. **Or use pre-computed weights**:
```powershell
python -u train.py --class_weights_file class_weights.json --device cuda
```

This helps the model learn rare classes (hats hidden by hair, glasses not worn, etc.) by increasing their loss contribution.

### CLI Arguments

- `--device`: `cuda` or `cpu` (default: auto-detect)
- `--num_workers`: DataLoader worker processes (default: 4)
- `--sample_image`: Path to image for per-epoch visualization (saves to `./{epoch}epoch/`)
- `--checkpoint_interval`: Save checkpoint every N images (default: 500)
- `--resume`: Resume training from latest checkpoint
- `--use_class_weights`: Auto-compute class weights from training masks (helps with rare classes)
- `--class_weights_file`: Load pre-computed weights from JSON (output of `compute_class_weights.py`)
- `--ignore_index`: Class index to ignore in loss (default: -100)

### Performance Tips
- **GPU**: Use `--device cuda` with `--num_workers 4` (or higher on systems with many cores)
- **CPU**: Use `--num_workers 0` to avoid multiprocessing overhead
- **Checkpoints**: Periodic checkpoints (`checkpoint_{N}.pth`) are saved during training for recovery from interruptions

## References
[1] <i>Deep Learning Project — Drawing Anime Face with Simple Segmentation Mask <a href="https://medium.com/@steinsfu/drawing-anime-face-with-simple-segmentation-mask-ca955c62ce09">link</a></i>

[2] <i>pit-ray/Anime-Semantic-Segmentation-GAN <a href="https://github.com/pit-ray/Anime-Semantic-Segmentation-GAN">link</a></i>
