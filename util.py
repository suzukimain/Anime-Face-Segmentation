import cv2 as cv
import numpy as np

COLOR_BACKGROUND = (255,255,0)   
COLOR_HAIR       = (0,255,255)   
COLOR_EYE        = (255,0,0)     
COLOR_MOUTH      = (255,255,255) 
COLOR_FACE       = (0,255,0)     
COLOR_SKIN       = (0,0,255)     
COLOR_CLOTHES    = (128,128,128) 

PALETTE = [
    COLOR_BACKGROUND,
    COLOR_HAIR,
    COLOR_EYE,
    COLOR_MOUTH,
    COLOR_FACE,
    COLOR_SKIN,
    COLOR_CLOTHES
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
    h, w, _ = src.shape
    flat = src.reshape(-1, 3).astype(np.int16)
    pal = np.array(PALETTE, dtype=np.int16)  # (K,3)
    # Compute nearest palette color for every pixel to sanitize unexpected colors (e.g., glasses)
    # distance^2 over RGB
    d2 = ((flat[:, None, :] - pal[None, :, :]) ** 2).sum(axis=2)  # (N,K)
    idx = d2.argmin(axis=1).reshape(h, w).astype(np.int64)
    return idx

def seg2img(src):
    src = np.array(src)
    # Accept logits/probs (C,H,W) or (H,W,C), or class indices (H,W)
    if src.ndim == 2:
        class_idx = src.astype(np.int64)
    elif src.ndim == 3:
        if src.shape[0] == len(PALETTE):
            # (C,H,W) -> (H,W,C)
            src = np.moveaxis(src, 0, 2)
        if src.shape[2] != len(PALETTE):
            raise ValueError(f"seg2img: unexpected channel count {src.shape} (palette size {len(PALETTE)})")
        class_idx = np.argmax(src, axis=2)
    else:
        raise ValueError(f"seg2img: unexpected array shape {src.shape}")

    h, w = class_idx.shape
    dst = np.zeros((h, w, 3), dtype=np.uint8)
    for idx, color in enumerate(PALETTE):
        mask = (class_idx == idx)
        if np.any(mask):
            dst[mask] = color
    return dst