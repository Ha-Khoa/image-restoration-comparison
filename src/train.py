"""
Train DnCNN for one fixed noise level

Paper reference: Sec. 4.2
    - Training pipeline identical to [48]
    - Adam optimization
    - Train network for removeing Gaussian noise of a fixed standard deviation σ
Sec. 5: main network uses sigma =0.02

Run:
    python -m src.train --train_folder <path>/BSDS500/data/images/train \
        --val_folder data/BSDS500_val --sigma 0.02
"""

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from src.data import load_folder
from src.metrics import psnr
from src.model import DnCNN, count_parameters, make_denoise_function, save_checkpoint
from src.run_day1 import noisy_set

# Same offset as in run_day1.py, so the validation noise is identical there and here
VALIDATION_NOISE_OFFSET = 1000

def pick_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def sample_training_batch(train_images, batch_size, patch_size, sigma, numpy_generator, torch_generator):
    """
    Cut random patches out of the clean training umages and add fresh noise,
    Fresh noise is drawn at every step, so the network never sees the same noisy patch twice
    Noise is not clipped (same as in data.py)
    """

    # Randomly select an image and crop a patch for each batch item
    clean_patches = np.empty((batch_size, 1, patch_size, patch_size), dtype=np.float32)
    for batch_index in range(batch_size):
        image = train_images[numpy_generator.integers(len(train_images))]
        image_height, image_width = image.shape
        top = numpy_generator.integers(0, image_height - patch_size + 1)
        # Choose a valid top-left corner for the patch
        left = numpy_generator.integers(0, image_width - patch_size + 1)
        patch = image[top:top + patch_size, left:left + patch_size]

        # Random rotation (0, 90, 180, 270 degrees) and random horizontal flip
        patch = np.rot90(patch, numpy_generator.integers(4))
        if numpy_generator.integers(2) == 1:
            patch = np.fliplr(patch)
 
        clean_patches[batch_index, 0] = patch

    # Convert clean patches to a PyTorch tensor
    clean_tensor = torch.from_numpy(clean_patches)
    noise_tensor = torch.randn(clean_tensor.shape, generator=torch_generator) * sigma
    noisy_tensor = clean_tensor + noise_tensor

    return clean_tensor, noisy_tensor

def evaluate_on_validation(model, device, validation_triples):
    """
    Calculate the mean PSNR before and after denoising
    """

    # Create a function that accepts and returns NumPy images
    denoise_function = make_denoise_function(model, device)  # also sets model.eval()
    denoised_scores = []
    noisy_scores = []

    # Evaluate every validation image
    for _image_name, clean_image, noisy_image in validation_triples:
        denoised_scores.append(psnr(denoise_function(noisy_image), clean_image))
        noisy_scores.append(psnr(noisy_image, clean_image))

    # Switch back to training mode after validation
    model.train()
    return float(np.mean(denoised_scores)), float(np.mean(noisy_scores))

def main():
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument("--train_folder", required=True)
    argument_parser.add_argument("--val_folder", required=True)
    argument_parser.add_argument("--sigma", type=float, default=0.02)
    argument_parser.add_argument("--max_side", type=int, default=256)
    argument_parser.add_argument("--number_of_layers", type=int, default=17)
    argument_parser.add_argument("--number_of_channels", type=int, default=64)
    argument_parser.add_argument("--patch_size", type=int, default=40)
    argument_parser.add_argument("--batch_size", type=int, default=128)
    argument_parser.add_argument("--number_of_steps", type=int, default=3000)
    argument_parser.add_argument("--learning_rate", type=float, default=1e-3)
    argument_parser.add_argument("--minimum_learning_rate", type=float, default=1e-5)
    argument_parser.add_argument("--log_every", type=int, default=50)
    argument_parser.add_argument("--validate_every", type=int, default=250)
    argument_parser.add_argument("--seed", type=int, default=0)
    argument_parser.add_argument("--checkpoint_folder", default="checkpoints")
    argument_parser.add_argument("--results_folder", default="results")
    arguments = argument_parser.parse_args()
 
    numpy_generator = np.random.default_rng(arguments.seed)
    torch_generator = torch.Generator().manual_seed(arguments.seed)
    torch.manual_seed(arguments.seed)
    device = pick_device()
 
    # Training images are shrunk to the same maximum size as the test images, so the
    # network trains on the same kind of (slightly smoothed) images that it is tested on.
    train_images = [image for _name, image in load_folder(arguments.train_folder, arguments.max_side)]
    validation_images = load_folder(arguments.val_folder, arguments.max_side)
    validation_triples = noisy_set(validation_images, arguments.sigma, VALIDATION_NOISE_OFFSET)
    smallest_side = min(min(image.shape) for image in train_images)
    if smallest_side < arguments.patch_size:
        raise ValueError(f"patch_size {arguments.patch_size} is larger than the smallest image side {smallest_side}")
 
    model = DnCNN(number_of_layers=arguments.number_of_layers,
                  number_of_channels=arguments.number_of_channels).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=arguments.learning_rate)
    # The learning rate falls smoothly from learning_rate to minimum_learning_rate.
    learning_rate_schedule = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=arguments.number_of_steps, eta_min=arguments.minimum_learning_rate)
    # Mean squared error between denoised and clean patch. This equals the error between
    # predicted noise and true noise, because denoised = noisy - predicted noise.
    loss_function = nn.MSELoss()
 
    print(f"device={device}  layers={arguments.number_of_layers}  "
          f"channels={arguments.number_of_channels}  parameters={count_parameters(model):,}")
    print(f"train images={len(train_images)}  validation images={len(validation_images)}  "
          f"sigma={arguments.sigma}")
 
    Path(arguments.checkpoint_folder).mkdir(exist_ok=True)
    Path(arguments.results_folder).mkdir(exist_ok=True)
    sigma_text = f"{arguments.sigma:.2f}"
    best_checkpoint_path = Path(arguments.checkpoint_folder) / f"dncnn_sigma{sigma_text}_best.pt"
    last_checkpoint_path = Path(arguments.checkpoint_folder) / f"dncnn_sigma{sigma_text}_last.pt"
    configuration = {
        "number_of_layers": arguments.number_of_layers,
        "number_of_channels": arguments.number_of_channels,
        "number_of_image_channels": 1,
        "use_batch_norm": True,
        "sigma": arguments.sigma,
        "patch_size": arguments.patch_size,
        "batch_size": arguments.batch_size,
        "number_of_steps": arguments.number_of_steps,
        "learning_rate": arguments.learning_rate,
        "seed": arguments.seed,
    }
 
    log_rows = []
    best_validation_psnr = -np.inf
    recent_losses = []
    start_time = time.time()
    model.train()
 
    for step in range(1, arguments.number_of_steps + 1):
        clean_batch, noisy_batch = sample_training_batch(
            train_images, arguments.batch_size, arguments.patch_size, arguments.sigma,
            numpy_generator, torch_generator)
        clean_batch = clean_batch.to(device)
        noisy_batch = noisy_batch.to(device)
 
        denoised_batch = model(noisy_batch)
        loss = loss_function(denoised_batch, clean_batch)
 
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        learning_rate_schedule.step()
        recent_losses.append(loss.item())
 
        if step % arguments.log_every == 0:
            seconds_per_step = (time.time() - start_time) / step
            print(f"step {step:5d}  loss {np.mean(recent_losses):.6f}  "
                  f"{seconds_per_step:.2f} s/step", flush=True)
            log_rows.append({"step": step, "train_loss": float(np.mean(recent_losses)),
                             "validation_psnr": np.nan, "noisy_psnr": np.nan})
            recent_losses = []
 
        if step % arguments.validate_every == 0 or step == arguments.number_of_steps:
            validation_psnr, noisy_psnr = evaluate_on_validation(model, device, validation_triples)
            print(f"   validation PSNR {validation_psnr:.2f} dB  "
                  f"(noisy images without processing: {noisy_psnr:.2f} dB)", flush=True)
            log_rows.append({"step": step, "train_loss": np.nan,
                             "validation_psnr": validation_psnr, "noisy_psnr": noisy_psnr})
            if validation_psnr > best_validation_psnr:
                best_validation_psnr = validation_psnr
                save_checkpoint(best_checkpoint_path, model, configuration)
 
    save_checkpoint(last_checkpoint_path, model, configuration)
    log_path = Path(arguments.results_folder) / f"train_log_sigma{sigma_text}.csv"
    pd.DataFrame(log_rows).to_csv(log_path, index=False)
    print(f"best validation PSNR {best_validation_psnr:.2f} dB -> {best_checkpoint_path}")
    print(f"log saved to {log_path}")
 
 
if __name__ == "__main__":
    main()
 

