import glob
import os
from PIL import Image
from torch.utils.data import Dataset

from util import img2seg

class UNetDataset(Dataset):
    def __init__(self, img_path, seg_path, transform=None):
            self.img_path = img_path
            self.seg_path = seg_path
            self.seg_path_list = glob.glob(self.seg_path + '/*.*')
            self.transform = transform
            
    def __getitem__(self, idx): 
        seg_path = self.seg_path_list[idx]
        seg = img2seg(seg_path)
        if self.transform is not None:
            seg_tensor = self.transform(seg)
        else:
            seg_tensor = None
        
        file_name = os.path.basename(seg_path).rsplit('.')[0]
        img_path = self.img_path+'/'+file_name+'.jpg'
        img = Image.open(img_path)
        if self.transform is not None:
            img_tensor = self.transform(img)
        else:
            img_tensor = None
            
        return img_tensor, seg_tensor
        
    def __len__(self): 
        return len(self.seg_path_list)
            