# app/utils.py
import numpy as np
from PIL import Image
import torchvision.transforms as T
import torch

IMG_SIZE = 256

CLASS_COLORS = {
    0: (50,  50,  50),
    1: (0,   200,  0),
    2: (255, 230,  0),
    3: (255, 140,  0),
    4: (220,  30, 30),
}

CLASS_NAMES = ["Background", "No damage", "Minor damage", "Major damage", "Destroyed"]

TRANSFORM = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std= [0.229, 0.224, 0.225]),
])


def preprocess(pil_image: Image.Image, size: int = IMG_SIZE) -> torch.Tensor:
    """PIL RGB image → (1, 3, H, W) normalised tensor."""
    return TRANSFORM(pil_image.convert("RGB").resize((size, size))).unsqueeze(0)


def to_color_mask(pred_np: np.ndarray) -> np.ndarray:
    """(H, W) class-index array → (H, W, 3) uint8 RGB."""
    out = np.zeros((*pred_np.shape, 3), dtype=np.uint8)
    for cls, color in CLASS_COLORS.items():
        out[pred_np == cls] = color
    return out


def make_overlay(post_pil: Image.Image,
                 color_mask: np.ndarray,
                 alpha: float = 0.45,
                 size: int = IMG_SIZE) -> np.ndarray:
    """Blend post image with color mask."""
    post = np.array(post_pil.resize((size, size)))
    return (post * (1 - alpha) + color_mask * alpha).astype(np.uint8)


def damage_stats(pred_np: np.ndarray) -> dict:
    """Per-class pixel count and percentage."""
    total = pred_np.size
    return {
        name: {
            "count": int((pred_np == i).sum()),
            "pct"  : round(100 * (pred_np == i).sum() / total, 2),
        }
        for i, name in enumerate(CLASS_NAMES)
    }
