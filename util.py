import cv2 as cv
import numpy as np

# Dataset color mapping (RGB format but stored as BGR in OpenCV)
COLOR_BACKGROUND = (255, 255, 0)     # background - yellow
COLOR_HAIR       = (19, 69, 139)     # hair - dark brown
COLOR_FACE       = (144, 238, 144)   # face - light green
COLOR_CLOTHES    = (255, 0, 0)       # clothes - blue
COLOR_SKIN       = (180, 224, 255)   # skin - light peach
COLOR_EYE        = (0, 255, 255)     # eye - yellow
COLOR_MOUTH      = (0, 0, 255)       # mouth - red
COLOR_OTHERS     = (128, 128, 128)   # others - accessories, hats, glasses

PALETTE = [
    COLOR_BACKGROUND,
    COLOR_HAIR,
    COLOR_FACE,
    COLOR_CLOTHES,
    COLOR_SKIN,
    COLOR_EYE,
    COLOR_MOUTH,
    COLOR_OTHERS
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

def seg2img(src, num_classes: int = 8):
    """Convert segmentation logits/indices to colored image.
    
    Args:
        src: (C,H,W), (H,W,C), or (H,W) array
        num_classes: Expected number of segmentation classes
    Returns:
        (H,W,3) uint8 BGR image
    """
    src = np.array(src)
    # Accept logits/probs (C,H,W) or (H,W,C), or class indices (H,W)
    if src.ndim == 2:
        class_idx = src.astype(np.int64)
        detected_classes = int(class_idx.max()) + 1
    elif src.ndim == 3:
        # (C,H,W) format?
        if src.shape[0] <= 32 and src.shape[2] > 32:
            # Likely (C,H,W)
            detected_classes = src.shape[0]
            src = np.moveaxis(src, 0, 2)
        else:
            detected_classes = src.shape[2]
        class_idx = np.argmax(src, axis=2)
    else:
        raise ValueError(f"seg2img: unexpected array shape {src.shape}")

    # Use provided num_classes or detected
    n_colors = max(num_classes, detected_classes)
    # Generate palette: use predefined if available, else generate
    palette = list(PALETTE)
    while len(palette) < n_colors:
        # Generate additional colors
        np.random.seed(len(palette))
        palette.append(tuple(np.random.randint(0, 256, 3).tolist()))
    
    h, w = class_idx.shape
    dst = np.zeros((h, w, 3), dtype=np.uint8)
    for idx in range(n_colors):
        mask = (class_idx == idx)
        if np.any(mask):
            dst[mask] = palette[idx]
    return dst