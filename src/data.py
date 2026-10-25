"""
Data: Load photo, and add Noise.

Paper-Reference:
- Modell: f = A*u + n, n ~ Gauss(0, σ²)  (Sec. 1 und Sec. 3.1).
  A = I
"""
from pathlib import Path
import cv2
import numpy as np

IMG_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def load_gray(path, max_side=None):
    """
    Normalize data to [0, 1] range, image to greyscale
    """
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(path)
    img = img.astype(np.float32) / 255.0
    # If img.shape > max side
    if max_side and max(img.shape) > max_side:
        s = max_side / max(img.shape)           # Calculate the scaling factor

        # Resize the image using scaling factor
        img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    return img


def load_folder(folder, max_side=None):
    """
    Read and sort the folder
    """
    files = sorted(p for p in Path(folder).iterdir() if p.suffix.lower() in IMG_EXT)
    if not files:
        raise FileNotFoundError(f"No photo in {folder}")
    return [(p.stem, load_gray(p, max_side)) for p in files]


def add_noise(img, sigma, seed):
    """
    Additives Gaussian Noise, σ in [0,1]-Skala
    All the methods see the exact same noisy images
    """
    # Fixed seed: default_rng(seed): return same numbers
    rng = np.random.default_rng(seed)
    return (img + rng.normal(0.0, sigma, img.shape)).astype(np.float32)


def sample_images():
    """Built-in test images"""
    from skimage import data, color, transform
    out = []
    for name in ["camera", "coins", "moon", "page", "text"]:
        out.append((name, np.asarray(getattr(data, name)(), dtype=np.float32) / 255.0))
    for name in ["astronaut", "chelsea", "coffee"]:
        g = color.rgb2gray(getattr(data, name)()).astype(np.float32)
        out.append((name, g))
    return [(n, transform.resize(g, (256, 256), anti_aliasing=True).astype(np.float32))
            for n, g in out]
