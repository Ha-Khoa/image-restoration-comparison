"""
Classical Denoising (Baselines).

Paper Reference:
- Equation (2): prox_R(b) = argmin_u 1/2||u-b||^2 + R(u) is itself a Entrauscher.
  For R = TV (Sec. 2, [34]) is exact TV-Denoising
  (hier Chambolle). Dasselbe Modul liefert später den TV-Prior im PnP-Code.
- Tab. 1 vergleicht NLM [3], BM3D [8] und DnCNN-S. NLM ist hier dabei,
  BM3D optional (nicht implementiert).
"""
import numpy as np
from scipy.ndimage import gaussian_filter, median_filter
from skimage.restoration import denoise_tv_chambolle, denoise_nl_means


def gaussian_denoise(x, s):
    return gaussian_filter(x, s)


def median_denoise(x, k):
    return median_filter(x, size=int(k))


def tv_denoise(x, w):
    return denoise_tv_chambolle(x, weight=w)


def nlm_denoise(x, h, sigma):
    # Patch 5x5, window 13x13 (fast_mode), h relative to σ
    return denoise_nl_means(x, h=h * sigma, sigma=sigma, patch_size=5,
                            patch_distance=6, fast_mode=True)


def make_methods(sigma):
    """Method -> List (Parameter, Function). Searching on Validation."""
    return {
        "Gaussian": [(s, lambda x, s=s: gaussian_denoise(x, s))
                     for s in [0.3, 0.4, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0]],
        "Median": [(k, lambda x, k=k: median_denoise(x, k)) for k in [3, 5, 7]],
        "TV": [(w, lambda x, w=w: tv_denoise(x, w))
               for w in np.geomspace(0.005, 0.6, 12)],
        "NLM": [(h, lambda x, h=h: nlm_denoise(x, h, sigma))
                for h in [0.6, 0.8, 1.0, 1.2]],
    }
