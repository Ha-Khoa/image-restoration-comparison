"""
METRIC
PSNR in dB like Table 1 (Sec. 4.2, 5).
SSIM not in paper, just an addition
"""
import numpy as np
from skimage.metrics import structural_similarity


def psnr(x, img):
    """
    PSNR = 10log10 (1/MSE)
    """
    x = np.clip(x, 0.0, 1.0)
    mse = np.mean((x.astype(np.float64) - img.astype(np.float64)) ** 2)
    return float(10 * np.log10(1.0 / max(mse, 1e-12)))


def ssim(x, img):
    return float(structural_similarity(np.clip(x, 0, 1), img, data_range=1.0))
