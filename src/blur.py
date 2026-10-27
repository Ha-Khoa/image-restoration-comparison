"""
Blue operator A for the deblurring experiments (Day 4).

Paper reference: the damaged image is f = A*u + n (Sec. 1, 3.1).

For deblurring (Sec. 5.2) the linear operator A is a convolution with a blur kernel,
folowed by Gaussian noise n.
Here A is appled with the FFT, which means the image is treated as periodic 
(the left edge wraps around to the right edge). The same assumption is used later when
the image is restored, so blurring and restoring match exactly.

The kernels below are my own choice. I do not have the kernels the paper use, so my numbers
are not comparable with the Table 3 of the paper
"""

import numpy as np

from src.data import add_noise
from src.run_day1 import SEED

def gaussian_kernel(width_in_pixels):
    """
    Normalized 2D Gaussian blur kernel. The width is the standard deviation in pixels.
    """

    # Determine kernel size based on standard deviation 
    radius = int(np.ceil(3 * width_in_pixels))
    offsets = np.arange(-radius, radius + 1)
    one_dimensional = np.exp(-(offsets ** 2) / (2 * width_in_pixels ** 2))
    kernel = np.outer(one_dimensional, one_dimensional)
    return kernel / kernel.sum()            # Normalize

class BlurOperator:
    """
    The operator A: blur an image by circular convolution with a kernel.

    The transfer function is the Fourier transform of the kernel, padded to the image size
    It depends on the image size, so it is computed once per size and kept
    """

    def __init__(self, kernel):
        if abs(kernel.sum() - 1.0) > 1e-6:
            raise ValueError("The blur kernel must sum to 1 (the blur must not change brightness)")
        self.kernel = kernel
        self._transfer_function_by_shape = {}

    def transfer_function(self, image_shape):
        if image_shape not in self._transfer_function_by_shape:
            kernel_height, kernel_width = self.kernel.shape
            image_height, image_width = image_shape
            if kernel_height > image_height or kernel_width > image_width:
                raise ValueError("The blur kernel is larger than the image")
            # Zero padding the smal kernel to match the full dimensions of the image
            padded_kernel = np.zeros(image_shape)
            padded_kernel[:kernel_height, :kernel_width] = self.kernel

            # Move the kernel centre to the top-left corner, so the blur does not shift the image.
            padded_kernel = np.roll(padded_kernel,
                                    shift=(-(kernel_height //2), -(kernel_width//2)),
                                    axis=(0,1))
            # Convert the padded/shifted kernel into frequency domain
            self._transfer_function_by_shape[image_shape] = np.fft.fft2(padded_kernel)

        return self._transfer_function_by_shape[image_shape]

    def apply(self, image):
        """
        Return A*u for a 2D image u.
        """
        transfer_function = self.transfer_function(image.shape)
        return np.real(np.fft.ifft2(np.fft.fft2(image) * transfer_function))

def degraded_set(images, blur_operator, noise_sigma, seed_offset):
    """
    For every (name, clean image): blur it, then add Gaussian noise

    Returns a list of (name, clean image, degraded image), the same layout as noisy_set in run_day1.py
    Seeds work as in Day 1: validation uses offset 1000, test uses offset 0.
    The noise is not clipped (dame as in data.py)
    """
    triples = []
    for index, (image_name, clean_image) in enumerate(images):
        blurred_image = blur_operator.apply(clean_image)
        degraded_image = add_noise(blurred_image, noise_sigma, SEED + seed_offset + index)
        triples.append((image_name, clean_image, degraded_image))

    return triples

        
            