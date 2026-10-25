"""
DnCNN denoising network

The network learns to predict noise insted of predicting the clean image directly

Paper reference: Sec. 4.2. The paper says its network is "similar to DnCNN-S"
    - 3x3 kernels, each followed by a ReLU
    - The input is a grayscale or a color image
"""

import numpy as np
import torch
from torch import nn

class DnCNN(nn.Module):
    def __init__(self, number_of_layers=17, number_of_channels=64, number_of_image_channels=1, use_batch_norm=True):
        super().__init__()
        if number_of_layers < 3:
            raise ValueError("number_of_layers must be at least 3 (first, middle, last)")

        layers = []

        # First layer: image -> Feature maps, followed by ReLU
        layers.append(nn.Conv2d(
                number_of_image_channels, 
                number_of_channels, 
                kernel_size=3, 
                padding=1, 
                bias=True)
            )
        layers.append(nn.ReLU(inplace=True))

        # Middle layers: feature maps -> feature maps.
        for _ in range(number_of_layers - 2):

            # Bias is not needed when BatchNorm follows the convolution
            layers.append(nn.Conv2d(
                number_of_channels,
                number_of_channels,
                kernel_size=3,
                padding=1,
                bias=not use_batch_norm
            ))

            # Normalize the feature values during training
            if use_batch_norm:
                layers.append(nn.BatchNorm2d(number_of_channels))
            layers.append(nn.ReLU(inplace=True))        # Apply ReLU after the convolution and optinal BatchNorm

        # Last layer: feature maps -> predicted noise (same shape as the image), no ReLU
        layers.append(nn.Conv2d(
            number_of_channels,
            number_of_image_channels,
            kernel_size=3,
            padding=1,
            bias=True)
        )    

        # Combine all layers into one network
        self.layers = nn.Sequential(*layers)
        self._initialize_weights()

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):    # check if current module is convolution layer or not
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")         # Kaiming initialization works well with ReLU
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        last_convolution = self.layers[-1]
        nn.init.zeros_(last_convolution.weight)
        nn.init.zeros_(last_convolution.bias)

    def predicted_noise(self, noisy_image):
        """
        Return the network's estimate of the noise in the image
        """
        return self.layers(noisy_image)

    def forward(self, noisy_image):
        """
        Return the denoised image: noisy image - the predicted noise
        Padding=1 -> Output has same height and width as the input
        """
        return noisy_image - self.predicted_noise(noisy_image)

def count_parameters(model):
    """
    Count all trainable and non_trainable model parameters
    """
    return sum(parameter.numel() for parameter in model.parameters())

def make_denoise_function(model, device):
    """
    Wrap a model as: 2D numpy umage in [0, 1]
    Same interface (NumPy) as the classical denoisers in classical.py
    The output is not clipped, clipping happens in the PSNR function
    """
    model.eval()

    def denoise(noisy_image):
        noisy_array = np.ascontiguousarray(noisy_image, dtype=np.float32)           # Convert Image to NumPy
        noisy_tensor = torch.from_numpy(noisy_array)[None, None].to(device)         # Convert numpy to tensor

        # Run model without gradient
        with torch.no_grad():
            denoised_tensor = model(noisy_tensor)
        return denoised_tensor[0, 0].cpu().numpy()          # Convert back to NumPy

    return denoise

def save_checkpoint(path, model, configuration):
    """
    Save the weights together with the settings needed to rebuild the model
    """
    torch.save({"model_state": model.state_dict(), "configuration": configuration}, path)
 
 
def load_checkpoint(path, device):
    """
    Rebuild the model from a checkpoint file. Returns (model, configuration)
    """

    # Load the saved weights and configuration onto the selected device
    checkpoint = torch.load(path, map_location=device)
    configuration = checkpoint["configuration"]

     # Rebuild the model using the original settings
    model = DnCNN(
        number_of_layers=configuration["number_of_layers"],
        number_of_channels=configuration["number_of_channels"],
        number_of_image_channels=configuration["number_of_image_channels"],
        use_batch_norm=configuration["use_batch_norm"],
    )
    # Restore the learned weights
    model.load_state_dict(checkpoint["model_state"])

    # Move the model to the selected device and enable evaluation mode
    model.to(device)
    model.eval()
    return model, configuration