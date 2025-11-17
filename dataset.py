import glob
import os
from PIL import Image
from torch.utils.data import Dataset

from util import img2seg

class UNetDataset(Dataset):
    def __init__(self, img_path, seg_path, transform=None, train_mode=False):
            self.img_path = img_path
            self.seg_path = seg_path
            self.transform = transform
            self.train_mode = train_mode

            # 1) 取得する拡張子（大小どちらも）
            img_exts = ['.png', '.jpg', '.jpeg']
            img_exts_all = img_exts + [e.upper() for e in img_exts]
            seg_globs = ['*.png', '*.jpg', '*.jpeg', '*.PNG', '*.JPG', '*.JPEG']

            # 2) セグ画像の候補を収集
            seg_files = []
            for pat in seg_globs:
                seg_files.extend(glob.glob(os.path.join(self.seg_path, pat)))
            seg_files = sorted(seg_files)
            if not seg_files:
                raise FileNotFoundError(f"No segmentation files found in: {self.seg_path}")

            # 3) 画像/マスクの同名ペアのみ残す
            self.pairs = []  # list of tuples (img_path, seg_path)
            skipped = 0
            for sp in seg_files:
                base = os.path.basename(sp).rsplit('.', 1)[0]
                found_img = None
                for ext in img_exts_all:
                    candidate = os.path.join(self.img_path, base + ext)
                    if os.path.exists(candidate):
                        found_img = candidate
                        break
                if found_img is not None:
                    self.pairs.append((found_img, sp))
                else:
                    skipped += 1

            if not self.pairs:
                raise FileNotFoundError(
                    f"No paired image/mask files. Check names under '{self.img_path}' and '{self.seg_path}'.")

            if skipped:
                print(f"Warning: {skipped} mask(s) had no matching image and were skipped.")
            
    def __getitem__(self, idx): 
        # Support 4x data augmentation: treat idx as virtual index
        if self.train_mode:
            real_idx = idx // 4  # map to original sample
            aug_variant = idx % 4  # which augmentation variant (0-3)
        else:
            real_idx = idx
            aug_variant = 0
        
        img_path, seg_path = self.pairs[real_idx]
        try:
            seg = img2seg(seg_path)  # (H,W) class indices
        except Exception as e:
            raise RuntimeError(f"Failed to load segmentation file: {seg_path} -> {e}") from e

        try:
            img = Image.open(img_path).convert('RGB')
        except Exception as e:
            raise RuntimeError(f"Failed to load image file: {img_path} -> {e}") from e

        # Convert mask to PIL for paired geometric augmentation
        from PIL import Image as PILImage
        import torch, numpy as np
        seg_pil = PILImage.fromarray(seg.astype(np.uint8))

        # Apply paired geometric augmentations if train_mode
        if self.train_mode:
            import random
            import torchvision.transforms.functional as TF
            # Deterministic augmentation based on aug_variant
            # Variant 0: original
            # Variant 1: horizontal flip
            # Variant 2: horizontal flip + rotation
            # Variant 3: vertical flip + affine
            
            if aug_variant >= 1:
                # Horizontal flip for variants 1 and 2
                if aug_variant in [1, 2]:
                    img = TF.hflip(img)
                    seg_pil = TF.hflip(seg_pil)
            
            if aug_variant == 2:
                # Rotation for variant 2
                angle = random.uniform(-15, 15)
                img = TF.rotate(img, angle, interpolation=PILImage.BILINEAR)
                seg_pil = TF.rotate(seg_pil, angle, interpolation=PILImage.NEAREST)
            
            if aug_variant == 3:
                # Vertical flip
                img = TF.vflip(img)
                seg_pil = TF.vflip(seg_pil)
                # Affine
                translate = (random.randint(-20, 20), random.randint(-20, 20))
                scale = random.uniform(0.9, 1.1)
                shear = random.uniform(-5, 5)
                img = TF.affine(img, angle=0, translate=translate, scale=scale, shear=shear, interpolation=PILImage.BILINEAR)
                seg_pil = TF.affine(seg_pil, angle=0, translate=translate, scale=scale, shear=shear, interpolation=PILImage.NEAREST)

        # Apply image transform (ColorJitter, Normalize, etc.)
        if self.transform is not None:
            img_tensor = self.transform(img)
        else:
            # fallback minimal pipeline
            from torchvision import transforms as _T
            img_tensor = _T.ToTensor()(img)

        # Convert mask back to tensor (class indices)
        seg_tensor = torch.from_numpy(np.array(seg_pil).astype(np.int64))

        return img_tensor, seg_tensor
        
    def __len__(self): 
        if self.train_mode:
            return len(self.pairs) * 4  # 4x augmentation multiplier
        return len(self.pairs)
            