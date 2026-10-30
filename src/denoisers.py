"""
Ready-made denoisers for plug-and-play

Every denoiser is a function "2D numpy image -> 2D numpy image",
the same format as the classical methods in classical.py and the network wrapper in model.py
"""

from src.classical import tv_denoise

def make_tv_denoiser(weight):
    """
    Total variation denoiser with a fixed weight (the setting 'w' of classical.py)
    """
    return lambda image: tv_denoise(image, weight)

def make_dncnn_denoiser(noise_sigma, checkpoint_folder="checkpoints", checkpoint_kind="best"):
    """
    DnCNN trained for one noise level (the files written by train.py)

    PyTorch is imported here and not at the top, so the total variation parts of the projects
    also run on a cimputer without PyTorch
    """
    from src.model import load_checkpoint, make_denoise_function
    from src.train import pick_device

    device = pick_device()
    checkpoint_path = f"{checkpoint_folder}/dncnn_sigma{noise_sigma:.2f}_{checkpoint_kind}.pt"
    model, _configuration = load_checkpoint(checkpoint_path, device)
    return make_denoise_function(model, device)