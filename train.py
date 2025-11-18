import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
from torchvision.utils import save_image
import numpy as np
import cv2 as cv
from PIL import Image
import os
import glob
import re
import csv
import json
import time

from network import UNet
from dataset import UNetDataset
from util import seg2img, img2seg
import argparse
from torchvision.transforms import InterpolationMode


def _get_device(explicit: str | None = None) -> torch.device:
    """Resolve torch.device from an optional string, falling back to CUDA if available.

    Mirrors the behavior used by `predict.py`.
    """
    if explicit:
        try:
            d = torch.device(explicit)
            if d.type == 'cuda' and not torch.cuda.is_available():
                return torch.device('cpu')
            return d
        except Exception:
            return torch.device('cpu')
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

####################################################################
# Constants and hyper parameters
DATA_PATH  = './dataset/output'
SEG_PATH = './dataset/mask'
MODEL_PATH = './model'
MODEL_NAME = 'UNet'
CHECKPOINT_PATH = './model/checkpoint.pth'
BASE_MODEL = ''  # 追加学習したい既存モデルのパス。未指定なら空文字のまま
LOG_CSV = os.path.join(MODEL_PATH, 'train_history.csv')
LOG_JSON = os.path.join(MODEL_PATH, 'train_history.json')

INPUT_LEN = 512

LEARNING_RATE = 0.0001
EPOCH = 50
START_EPOCH = 0

TRAIN_BATCH_SIZE = 8
VAL_BATCH_SIZE = 8
TEST_BATCH_SIZE = 1


class TrainDatasetWrapper(torch.utils.data.Dataset):
    """Wrap a Subset and apply deterministic augmentations to create multiple variants per sample.

    This wrapper multiplies the dataset size by `multiplier` (4 by default) and for each
    returned index maps it to a sample in the underlying `subset` and applies a deterministic
    augmentation chosen by the variant id.
    """
    def __init__(self, subset, base_dataset, multiplier=4):
        self.subset = subset
        self.base_dataset = base_dataset
        self.multiplier = multiplier
        # keep a local copy of subset indices to avoid changes
        self.indices = list(subset.indices) if hasattr(subset, 'indices') else list(range(len(subset)))

    def __len__(self):
        return len(self.indices) * self.multiplier

    def __getitem__(self, idx):
        # Import all dependencies at the start to avoid UnboundLocalError
        import random
        import numpy as np
        from PIL import Image as PILImage
        from torchvision.transforms import functional as TF
        from torchvision.transforms.functional import InterpolationMode
        
        # map virtual idx to underlying subset index and variant
        real_subset_idx = idx // self.multiplier
        aug_variant = idx % self.multiplier

        # get actual original dataset index from subset
        orig_idx = self.indices[real_subset_idx]

        # Retrieve original image/mask paths from base dataset
        img_path, seg_path = self.base_dataset.pairs[orig_idx]
        
        # Load image and mask
        img_pil = PILImage.open(img_path).convert('RGB')
        # use shared util.img2seg to convert mask image into class indices
        seg_np = img2seg(seg_path)
        seg_pil = PILImage.fromarray(seg_np.astype(np.uint8))

        # deterministic augmentations by variant
        if aug_variant >= 1:
            # Horizontal flip for variants 1 and 2
            if aug_variant in [1, 2]:
                img_pil = TF.hflip(img_pil)
                seg_pil = TF.hflip(seg_pil)
        if aug_variant == 2:
            # Rotation for variant 2
            angle = random.uniform(-15, 15)
            img_pil = TF.rotate(img_pil, angle, interpolation=InterpolationMode.BILINEAR)
            seg_pil = TF.rotate(seg_pil, angle, interpolation=InterpolationMode.NEAREST)
        if aug_variant == 3:
            img_pil = TF.vflip(img_pil)
            seg_pil = TF.vflip(seg_pil)
            translate = (random.randint(-20, 20), random.randint(-20, 20))
            scale = random.uniform(0.9, 1.1)
            shear = random.uniform(-5, 5)
            img_pil = TF.affine(img_pil, angle=0, translate=list(translate), scale=scale, shear=[shear], interpolation=InterpolationMode.BILINEAR)
            seg_pil = TF.affine(seg_pil, angle=0, translate=list(translate), scale=scale, shear=[shear], interpolation=InterpolationMode.NEAREST)

        # After geometric augment, apply image transform (same as base dataset) and prepare seg_tensor
        if self.base_dataset.transform is not None:
            img_tensor = self.base_dataset.transform(img_pil)
        else:
            from torchvision import transforms as _T
            img_tensor = _T.ToTensor()(img_pil)

        seg_tensor = torch.from_numpy(np.array(seg_pil).astype(np.int64))
        return img_tensor, seg_tensor

R_TRAIN = 0.88
R_VAL = 0.07
# Define transformer for images (ImageNet normalization + light color jitter)
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', help='device to use (cuda or cpu)')
    parser.add_argument('--num_workers', type=int, default=4, help='DataLoader num_workers')
    parser.add_argument('--sample_image', type=str, default='', help='Path to a sample image to run inference on every epoch and save result')
    args = parser.parse_args()
    
    device = _get_device(args.device)
    if device.type == 'cuda':
        # Enable CuDNN benchmark to select best conv algorithms; improves throughput for fixed input sizes (512x512)
        torch.backends.cudnn.benchmark = True
    
    transformer = transforms.Compose([
                transforms.Resize(INPUT_LEN),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
                transforms.RandomGrayscale(p=0.1),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ])
    # Deterministic transform for evaluation / saving sample outputs (no color jitter or randomness)
    eval_transform = transforms.Compose([
                transforms.Resize(INPUT_LEN),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ])
    # Load dataset (transform applies to images only; dataset keeps train_mode flag)
    total_dataset = UNetDataset(img_path=DATA_PATH, seg_path=SEG_PATH, transform=transformer, train_mode=False)
    # Split train, validation, test
    # Compute split sizes so that any rounding remainder goes to the training set
    len_total = len(total_dataset)
    R_TEST = 1.0 - R_TRAIN - R_VAL
    len_val = int(len_total * R_VAL)
    len_test = int(len_total * R_TEST)
    len_train = len_total - len_val - len_test
    train_dataset, validation_dataset, test_dataset = random_split(total_dataset, [len_train, len_val, len_test])
    # Enable train_mode for train_dataset by wrapping with augmentation-enabled dataset
    train_dataset_aug = TrainDatasetWrapper(train_dataset, total_dataset)
    # Build loaders
    # Do not drop the last (possibly smaller) batch so that all images are used per epoch
    train_loader = DataLoader(
        train_dataset_aug,
        batch_size=TRAIN_BATCH_SIZE,
        shuffle=True,
        drop_last=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == 'cuda')
    )
    val_loader = DataLoader(
        validation_dataset,
        batch_size=VAL_BATCH_SIZE,
        shuffle=True,
        drop_last=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == 'cuda')
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=TEST_BATCH_SIZE,
        shuffle=True,
        drop_last=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == 'cuda')
    )
    # Build Model :: In: 3x512x512 -> Out: 8x512x512
    model = UNet()
    model.to(device)
    
    optimizer = optim.Adam(model.parameters(), LEARNING_RATE, weight_decay=1e-5)
    # Cosine annealing with warm restarts: smooth decay and periodic restarts for better exploration
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-6)
    # Backup: reduce LR on plateau (validation loss)
    scheduler_plateau = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    criterion = nn.CrossEntropyLoss()
    
    # Prefer loading a BASE_MODEL for additional training if specified
    if BASE_MODEL and os.path.exists(BASE_MODEL):
        print(f'Loading base model from {BASE_MODEL} for additional training...')
        try:
            state = torch.load(BASE_MODEL, map_location=device)
            if isinstance(state, dict) and 'model_state_dict' in state:
                state = state['model_state_dict']
            try:
                model.load_state_dict(state)
            except RuntimeError as e:
                print(f'Warning: strict load failed ({e}). Retrying with strict=False...')
                missing_unexpected = model.load_state_dict(state, strict=False)
                try:
                    missing, unexpected = missing_unexpected  # PyTorch returns (missing, unexpected)
                    if missing:
                        print(f'  Missing keys: {missing}')
                    if unexpected:
                        print(f'  Unexpected keys: {unexpected}')
                except Exception:
                    pass
            START_EPOCH = 0
            print('Base model loaded. Starting additional training from epoch 0.')
        except Exception as e:
            print(f'Warning: Failed to load base model: {e}')
            print('Proceeding without base model (random initialization).')
            START_EPOCH = 0
    elif os.path.exists(CHECKPOINT_PATH):
        # Load checkpoint if exists (used when BASE_MODEL is not specified)
        print(f'Loading checkpoint from {CHECKPOINT_PATH}...')
        try:
            checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            if 'scheduler_plateau_state_dict' in checkpoint:
                scheduler_plateau.load_state_dict(checkpoint['scheduler_plateau_state_dict'])
            START_EPOCH = checkpoint['epoch'] + 1
            print(f'Resuming training from epoch {START_EPOCH}')
        except RuntimeError as e:
            print(f'Warning: Failed to load checkpoint: {e}')
            print('Model architecture mismatch detected. Starting training from scratch.')
            START_EPOCH = 0
    else:
        print('No checkpoint found. Starting training from scratch.')
        # Load pretrained model if available (only when starting fresh)
        if os.path.exists('save/UNet.pth'):
            try:
                model.load_state_dict(torch.load('save/UNet.pth'))
                print('Loaded pretrained model from save/UNet.pth')
            except RuntimeError as e:
                print(f'Warning: Failed to load pretrained model: {e}')
                print('Starting with randomly initialized weights.')
    
    # Detect per-epoch saved models (e.g. model/UNet_ep{n}.pth) and resume from the latest if newer
    try:
        ep_pattern = os.path.join(MODEL_PATH, MODEL_NAME + '_ep*.pth')
        ep_files = glob.glob(ep_pattern)
        max_epoch = -1
        max_fp = None
        for fp in ep_files:
            m = re.search(r'_ep(\d+)\.pth$', fp)
            if m:
                try:
                    e = int(m.group(1))
                except Exception:
                    continue
                if e > max_epoch:
                    max_epoch = e
                    max_fp = fp
        if max_epoch >= 0 and max_fp:
            # Only override START_EPOCH if this epoch is newer than current START_EPOCH
            if max_epoch + 1 > START_EPOCH:
                try:
                    print(f'Found saved epoch model: {max_fp} (epoch {max_epoch}). Loading and resuming from next epoch {max_epoch+1}...')
                    state = torch.load(max_fp, map_location=device)
                    if isinstance(state, dict) and 'model_state_dict' in state:
                        state = state['model_state_dict']
                    model.load_state_dict(state)
                    START_EPOCH = max_epoch + 1
                    print(f'Successfully loaded epoch {max_epoch}. START_EPOCH set to {START_EPOCH}.')
                except Exception as e:
                    print(f'Warning: failed to load epoch model {max_fp}: {e}')
    except Exception:
        pass
    
    def train(epoch):
        model.train()
        train_loss = 0
        # ensure log dir exists
        os.makedirs(MODEL_PATH, exist_ok=True)
        # init csv file with header if not exists
        if not os.path.exists(LOG_CSV):
            try:
                with open(LOG_CSV, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(['phase','epoch','batch_idx','processed','total_samples','loss','lr','timestamp'])
            except Exception:
                pass
        for batch_idx, data in enumerate(train_loader):
            img, seg_target = data
            img = img.to(device, non_blocking=True)
            seg_target = seg_target.to(device, non_blocking=True).long()
    
            optimizer.zero_grad()
    
            pred_seg = model(img)
            loss = criterion(pred_seg, seg_target)
    
            loss.backward()
            # accumulate total loss over samples (for correct per-sample average)
            batch_n = img.size(0)
            train_loss += loss.item() * batch_n
            optimizer.step()
                    
            # scheduler will step per-epoch (after validation)
            
            if batch_idx % 20 == 0:
                # use get_last_lr() to avoid deprecated get_lr() warning
                try:
                    lr = float(scheduler.get_last_lr()[0])
                except Exception:
                    lr = float(scheduler.get_lr()[0])
                # show processed samples count (use (batch_idx+1)*batch_n to include current batch)
                processed = (batch_idx + 1) * batch_n
                total = len(train_loader.dataset)
                percent = int(100. * processed / total) if total > 0 else 0
                print('Train Epoch: {:>6} [{:>6}/{:>6} ({:>2}%)]\tLoss: {:.6f}\t\t lr: {:.6f}'.format(
                    epoch, processed, total,
                    percent, loss.item(), lr))
        # compute average loss per sample
        avg_loss = train_loss / len(train_loader.dataset) if len(train_loader.dataset) > 0 else 0.0
        print('====> Epoch: {} Average loss: {:.8f}'.format(epoch, avg_loss))
    
    def save_checkpoint(epoch, model, optimizer, scheduler, scheduler_plateau, path):
        """Save training checkpoint"""
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'scheduler_plateau_state_dict': scheduler_plateau.state_dict(),
        }, path)
        print(f'Checkpoint saved at epoch {epoch}')
        
    def validation(epoch):
        model.eval()
        val_loss= 0
        with torch.no_grad():
            for data in val_loader:
                img, seg_target = data
                img = img.to(device, non_blocking=True)
                seg_target = seg_target.to(device, non_blocking=True)
                pred_seg = model(img)
    
                # sum up batch loss (per-sample)
                batch_n = img.size(0)
                val_loss += criterion(pred_seg, seg_target).item() * batch_n
                
            
        avg_val = val_loss / len(val_loader.dataset) if len(val_loader.dataset) > 0 else 0.0
        print('====> Test set loss: {:.8f}'.format(avg_val))
        # log validation loss + current lr
        try:
            current_lr = float(optimizer.param_groups[0]['lr'])
        except Exception:
            current_lr = 0.0
        try:
            with open(LOG_CSV, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['val', int(epoch), '', '', int(len(val_loader.dataset)), float(avg_val), float(current_lr), time.time()])
        except Exception:
            pass
        # also append to JSON (update last epoch entry if exists)
        try:
            if os.path.exists(LOG_JSON):
                with open(LOG_JSON, 'r', encoding='utf-8') as jf:
                    history = json.load(jf)
            else:
                history = {'epochs': []}
            if history.get('epochs'):
                history['epochs'][-1].update({'val_loss': float(avg_val), 'lr': float(current_lr)})
            else:
                history.setdefault('epochs', []).append({'epoch': int(epoch), 'val_loss': float(avg_val), 'lr': float(current_lr)})
            with open(LOG_JSON, 'w', encoding='utf-8') as jf:
                json.dump(history, jf, ensure_ascii=False, indent=2)
        except Exception:
            pass
        return avg_val
    
    
    def save_epoch_sample(epoch: int, image_path: str):
        """Run model inference on a single image and save a colorized segmentation into a folder named '<epoch+1>epoch'.
    
        Example: epoch=0 and image_path='./Hoshino.webp' -> './1epoch/Hoshino.png'
        """
        if not image_path:
            return
        try:
            # prepare output directory
            out_dir = os.path.join('.', f'{epoch+1}epoch')
            os.makedirs(out_dir, exist_ok=True)
    
            img_pil = Image.open(image_path).convert('RGB')
            inp = eval_transform(img_pil).unsqueeze(0).to(device)
            model.eval()
            with torch.no_grad():
                pred = model(inp)
                # convert to class indices
                pred_idx = pred.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
                vis = seg2img(pred_idx)
                # vis is expected as HxWx3 RGB uint8
                out_name = os.path.splitext(os.path.basename(image_path))[0] + '.png'
                out_path = os.path.join(out_dir, out_name)
                Image.fromarray(vis).save(out_path)
                print(f'Saved epoch sample to {out_path}')
        except Exception as e:
            print(f'Failed saving epoch sample for {image_path} at epoch {epoch}: {e}')
    
    for epoch in range(START_EPOCH, EPOCH):
        train(epoch)
        val_loss = validation(epoch)
        # Step CosineAnnealingWarmRestarts per-epoch
        scheduler.step()
        # Step ReduceLROnPlateau based on validation loss
        scheduler_plateau.step(val_loss)
        
        # Save checkpoint every epoch
        save_checkpoint(epoch, model, optimizer, scheduler, scheduler_plateau, CHECKPOINT_PATH)
        
        # Save model weights every epoch
        torch.save(model.state_dict(), MODEL_PATH+'/'+MODEL_NAME+f'_ep{epoch}'+'.pth')
        # If a sample image is specified, run inference and save a visualized segmentation for this epoch
        if getattr(args, 'sample_image', ''):
            save_epoch_sample(epoch, args.sample_image)
        
    
    torch.save(model.state_dict(), MODEL_PATH+'/'+MODEL_NAME+'.pth')
    
    with torch.no_grad():
        model.eval()
        for data in test_loader:
            img, seg_target = data
            img = img.to(device, non_blocking=True)
            seg_target = seg_target.to(device, non_blocking=True)
            pred_seg = model(img)
            
            pred_seg = pred_seg.cpu().numpy()
            for i in range(TEST_BATCH_SIZE):
                img = seg2img(np.moveaxis(pred_seg[i],0,2))
                cv.imwrite(f'result{i}.png',img)
            break