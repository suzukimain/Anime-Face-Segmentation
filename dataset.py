import glob
import os
from PIL import Image
from torch.utils.data import Dataset

from util import img2seg

class UNetDataset(Dataset):
    def __init__(self, img_path, seg_path, transform=None):
            self.img_path = img_path
            self.seg_path = seg_path
            self.transform = transform

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
        img_path, seg_path = self.pairs[idx]
        try:
            seg = img2seg(seg_path)
        except Exception as e:
            raise RuntimeError(f"Failed to load segmentation file: {seg_path} -> {e}") from e
        if self.transform is not None:
            seg_tensor = self.transform(seg)
        else:
            seg_tensor = None

        try:
            img = Image.open(img_path).convert('RGB')
        except Exception as e:
            raise RuntimeError(f"Failed to load image file: {img_path} -> {e}") from e
        if self.transform is not None:
            img_tensor = self.transform(img)
        else:
            img_tensor = None

        return img_tensor, seg_tensor
        
    def __len__(self): 
        return len(self.pairs)
            