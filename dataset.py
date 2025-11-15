import glob
import os
from PIL import Image
from torch.utils.data import Dataset

from util import img2seg

class UNetDataset(Dataset):
    def __init__(self, img_path, seg_path, transform=None):
            self.img_path = img_path
            self.seg_path = seg_path
            # collect segmentation files and sort for stable ordering
            self.seg_path_list = sorted(glob.glob(os.path.join(self.seg_path, '*.*')))
            if len(self.seg_path_list) == 0:
                raise FileNotFoundError(f"No segmentation files found in: {self.seg_path}")
            self.transform = transform
            
    def __getitem__(self, idx): 
        seg_path = self.seg_path_list[idx]
        try:
            seg = img2seg(seg_path)
        except Exception as e:
            raise RuntimeError(f"Failed to load segmentation file: {seg_path} -> {e}") from e
        if self.transform is not None:
            seg_tensor = self.transform(seg)
        else:
            seg_tensor = None
        
        file_name = os.path.basename(seg_path).rsplit('.')[0]
        # Support multiple image extensions for input images (.jpg, .jpeg, .png)
        exts = ['.jpg', '.jpeg', '.png']
        img_path = None
        for ext in exts:
            candidate = os.path.join(self.img_path, file_name + ext)
            if os.path.exists(candidate):
                img_path = candidate
                break
        # try uppercase extensions too (e.g. .JPG/.PNG)
        if img_path is None:
            for ext in [e.upper() for e in exts]:
                candidate = os.path.join(self.img_path, file_name + ext)
                if os.path.exists(candidate):
                    img_path = candidate
                    break
        if img_path is None:
            raise FileNotFoundError(f"Paired image for '{file_name}' not found in '{self.img_path}' (tried: {exts})")

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
        return len(self.seg_path_list)
            