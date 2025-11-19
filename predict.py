import os
import argparse
from typing import Optional, Union, Any, cast

import torch
from torchvision import transforms
import cv2 as cv
from PIL import Image

from network import UNet
from util import seg2img

__all__ = ["convert_img"]

def _get_device(explicit: Optional[str] = None) -> torch.device:
    """Resolve torch.device from an optional string, falling back to CUDA if available."""
    if explicit:
        try:
            d = torch.device(explicit)
            if d.type == "cuda" and not torch.cuda.is_available():
                # Fallback to CPU if CUDA requested but unavailable
                return torch.device("cpu")
            return d
        except Exception:
            return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def _detect_num_classes(state: dict) -> int:
    """Detect number of segmentation classes from checkpoint state_dict."""
    # Look for de_block0 final conv layer output channels
    candidates = [
        'de_block0.1.weight',
        'module.de_block0.1.weight',
        'de_block0.1.bias',
        'module.de_block0.1.bias'
    ]
    for key in candidates:
        if key in state:
            tensor = state[key]
            if 'weight' in key:
                # Conv2d weight shape: (out_ch, in_ch, k, k)
                return tensor.shape[0]
            elif 'bias' in key:
                # Conv2d bias shape: (out_ch,)
                return tensor.shape[0]
    # Fallback: default to 8
    print("Warning: Could not detect num_classes from checkpoint, defaulting to 8")
    return 8

def _load_model(model_path: str, device: torch.device) -> tuple[UNet, int]:
    """Load model from checkpoint and return (model, num_classes)."""
    checkpoint = torch.load(model_path, map_location=device)
    # Handle both checkpoint dict (with 'model_state_dict' or 'state_dict') and plain state_dict
    if isinstance(checkpoint, dict):
        if 'model_state_dict' in checkpoint:
            state = checkpoint['model_state_dict']
        elif 'state_dict' in checkpoint:
            state = checkpoint['state_dict']
        else:
            state = checkpoint
    else:
        state = checkpoint

    # Detect number of classes from checkpoint
    num_classes = _detect_num_classes(state)
    print(f"Detected {num_classes} classes from checkpoint")
    
    # Build model with detected number of classes
    model = UNet(num_classes=num_classes)
    model_state = model.state_dict()
    filtered_state = {}

    def find_matching_key(src_key: str):
        # Try exact key, then try with/without 'module.' prefix (DataParallel compat)
        if src_key in model_state:
            return src_key
        mod_pref = 'module.' + src_key
        if mod_pref in model_state:
            return mod_pref
        if src_key.startswith('module.'):
            stripped = src_key[len('module.'):]
            if stripped in model_state:
                return stripped
        return None

    for k, v in list(state.items()):
        match = find_matching_key(k)
        if match is None:
            # Key not present in target model; skip it.
            print(f"Skipping unmatched key in checkpoint: {k}")
            continue
        if model_state[match].shape != v.shape:
            print(f"Skipping key due to shape mismatch: {k} -> {match} | checkpoint: {tuple(v.shape)} != model: {tuple(model_state[match].shape)}")
            continue
        filtered_state[match] = v

    # Load filtered state dict (non-strict to allow missing keys)
    model.load_state_dict(filtered_state, strict=False)
    model.to(device)
    model.eval()
    return model, num_classes

def _to_pil_image(src: Union[str, Image.Image, Any]) -> Image.Image:
    """Accept path, PIL.Image, or numpy array and return a PIL RGB image."""
    if isinstance(src, Image.Image):
        pil = cast(Image.Image, src)
        return pil.convert("RGB")
    # ndarray-like (e.g., numpy) handling without importing numpy explicitly
    try:
        if hasattr(src, "__array_interface__") and hasattr(src, "shape"):
            # Assume channel-last
            if getattr(src, "ndim", 0) == 3 and getattr(src, "shape")[2] == 3:
                try:
                    arr_rgb = src[..., ::-1]  # BGR -> RGB  # type: ignore[index]
                except Exception:
                    arr_rgb = src  # type: ignore[assignment]
                return Image.fromarray(arr_rgb).convert("RGB")  # type: ignore[arg-type]
            return Image.fromarray(src).convert("RGB")  # type: ignore[arg-type]
    except Exception:
        pass
    # Fallback: treat as file path
    return Image.open(str(src)).convert("RGB")

def convert_img(
    src: Union[str, Image.Image, Any],
    model_path: str = "model/UNet.pth",
    device: Optional[str] = None,
    resize: int = 512,
    save_path: Optional[str] = None,
    save_dir: Optional[str] = None,
) -> Any:
    """
    Run segmentation and return a colorized result as a numpy array (H, W, 3) uint8.

    Parameters:
        src: Path to an image, a PIL.Image, or a numpy array (assumed BGR as used by OpenCV).
        model_path: Path to the pretrained U-Net weights.
        device: Optional device string (e.g., "cuda" or "cpu"); automatically chosen if None.
        resize: Side length to resize the image to (square) before inference.
        save_path: If provided, save the resulting image to this path (format inferred from the file extension).
        save_dir: (deprecated) If provided and `save_path` is None, save into this directory as `result.png` or `result_{n}.png`.

    Returns:
        numpy.ndarray: Color image of shape (resize, resize, 3) with dtype uint8.
    """
    # Resolve device and load model
    dev = _get_device(device)
    model, num_classes = _load_model(model_path, dev)

    # Preprocess image
    pil_img = _to_pil_image(src)
    transform = transforms.Compose([
        transforms.Resize(resize),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ])
    x = transform(pil_img).unsqueeze(0).to(dev)

    # Inference
    with torch.no_grad():
        seg = model(x).squeeze(0)

    # Postprocess to color image (numpy HWC uint8)
    result = seg2img(seg.detach().cpu().numpy(), num_classes=num_classes)  # (H, W, 3) uint8

    # Optional save: prefer explicit save_path, fallback to deprecated save_dir
    if save_path is not None:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        cv.imwrite(save_path, result)
    elif save_dir is not None:
        # Ensure directory exists
        os.makedirs(os.path.abspath(save_dir), exist_ok=True)
        base_path = os.path.join(save_dir, 'result.png')
        if not os.path.exists(base_path):
            out_path = base_path
        else:
            idx = 1
            while True:
                out_path = os.path.join(save_dir, f'result_{idx}.png')
                if not os.path.exists(out_path):
                    break
                idx += 1
        cv.imwrite(out_path, result)

    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--src_path', required=True, help='path of source image (square recommended)')
    parser.add_argument('--model_path', default='model/UNet.pth', help='path to load trained U-Net model')
    parser.add_argument('--save_path', help='file path to save the result image (e.g., out/result.png)')
    parser.add_argument('--save_dir', help='[deprecated] directory to save segmentation image')
    parser.add_argument('--device', help='device to use (cuda or cpu)')
    parser.add_argument('--size', type=int, default=512, help='resize length (square) before inference')
    args = parser.parse_args()

    save_path = args.save_path
    if save_path is None and args.save_dir:
        save_path = os.path.join(args.save_dir, 'result.png')

    _ = convert_img(
        src=args.src_path,
        model_path=args.model_path,
        device=args.device,
        resize=args.size,
        save_path=save_path,
    )



