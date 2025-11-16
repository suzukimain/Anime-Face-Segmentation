import cv2 as cv
import numpy as np

COLOR_BACKGROUND = (255,255,0)   
COLOR_HAIR       = (0,255,255)   
COLOR_EYE        = (255,0,0)     
COLOR_MOUTH      = (255,255,255) 
COLOR_FACE       = (0,255,0)     
COLOR_BODY       = (0,0,255)     # Skin + Clothes combined

PALETTE = [
    COLOR_BACKGROUND,
    COLOR_HAIR,
    COLOR_EYE,
    COLOR_MOUTH,
    COLOR_FACE,
    COLOR_BODY
]


def img2seg(path):
    src = cv.imread(path)
    if src is None:
        raise FileNotFoundError(f"Segmentation image not found or unreadable: {path}")
    # Ensure the segmentation image has expected shape (H=512, W=512, C=3)
    if src.ndim != 3 or src.shape[2] != 3:
        raise ValueError(f"Segmentation image has unexpected channels: {path} shape={src.shape}")
    if src.shape[0] != 512 or src.shape[1] != 512:
        raise ValueError(f"Segmentation image has unexpected size: {path} shape={src.shape} (expected 512x512)")
    src = src.reshape(-1, 3)
    seg_list = []
    for color in PALETTE:
        seg_list.append(np.where(np.all(src==color, axis=1), 1.0, 0.0))
    dst = np.stack(seg_list,axis=1).reshape(512,512,6)
    
    return dst.astype(np.float32)

def seg2img(src):
    src = np.array(src)
    # Accept either (C,H,W) or (H,W,C). Normalize to (H,W,C)
    if src.ndim != 3:
        raise ValueError(f"seg2img: expected 3D array, got shape {src.shape}")
    if src.shape[0] == len(PALETTE):
        # (C,H,W) -> (H,W,C)
        src = np.moveaxis(src, 0, 2)
    elif src.shape[2] == len(PALETTE):
        # already (H,W,C)
        pass
    else:
        raise ValueError(f"seg2img: unexpected channel count {src.shape} (palette size {len(PALETTE)})")

    # src is now (H,W,C); pick class with highest score per pixel
    class_idx = np.argmax(src, axis=2)
    h, w = class_idx.shape
    dst = np.zeros((h, w, 3), dtype=np.uint8)
    for idx, color in enumerate(PALETTE):
        mask = (class_idx == idx)
        if np.any(mask):
            dst[mask] = color

    return dst