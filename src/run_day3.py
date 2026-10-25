"""
Measure the trained DnCNN networks on the TEST images and add them to day 1 table
Paper reference: Table 1
    Each noise level uses the network that was trained for exactly that noise level 
    ("each sigma requires an individually trained DnCNN-S", Sec. 4.2)

Usage:
    python -m src.run_day3 --test data/BSDS500_test --max_side 256

The test images and their noise are identical to Day 1 (same seed), so the DnCNN row is
measured on exactly the same noisy images as Gaussian, median, TV and NLM
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.data import load_folder
from src.metrics import psnr, ssim
from src.model import load_checkpoint, make_denoise_function
from src.run_day1 import SIGMAS, noisy_set
from src.train import pick_device

# Same offset as the test set in run_day1.py
TEST_NOISE_OFFSET = 0

# Row order of the final table
METHOD_ORDER = ["Noisy", "Gaussian", "Median", "TV", "NLM", "DnCNN"]

def main():
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument("--test", required=True, help="folder with the test images")
    argument_parser.add_argument("--max_side", type=int, default=256)
    argument_parser.add_argument("--checkpoint_folder", default="checkpoints")
    argument_parser.add_argument("--checkpoint_kind", choices=["best", "last"], default="best",
                                 help="'best' = checkpoint with the highest validation PSNR")
    argument_parser.add_argument("--day1_csv", default="results/day1_denoise.csv")
    argument_parser.add_argument("--output_csv", default="results/day3_denoise_with_dncnn.csv")
    arguments = argument_parser.parse_args()
 
    device = pick_device()
    test_images = load_folder(arguments.test, arguments.max_side)
    day1_table = pd.read_csv(arguments.day1_csv)
    # Remove old DnCNN rows (if this script is run twice) so rows are never duplicated.
    table_without_dncnn = day1_table[day1_table["method"] != "DnCNN"]
 
    dncnn_rows = []
    for sigma in SIGMAS:
        checkpoint_path = Path(arguments.checkpoint_folder) / \
            f"dncnn_sigma{sigma:.2f}_{arguments.checkpoint_kind}.pt"
        if not checkpoint_path.exists():
            print(f"sigma={sigma}: no checkpoint at {checkpoint_path}, skipped")
            continue
 
        model, configuration = load_checkpoint(checkpoint_path, device)
        denoise_function = make_denoise_function(model, device)
        test_triples = noisy_set(test_images, sigma, TEST_NOISE_OFFSET)
 
        psnr_scores = []
        ssim_scores = []
        for _image_name, clean_image, noisy_image in test_triples:
            denoised_image = denoise_function(noisy_image)
            psnr_scores.append(psnr(denoised_image, clean_image))
            ssim_scores.append(ssim(denoised_image, clean_image))
 
        print(f"sigma={sigma}: {checkpoint_path.name}  "
              f"(trained with sigma={configuration['sigma']}, "
              f"{configuration['number_of_steps']} steps)")
        dncnn_rows.append({
            "sigma": sigma,
            "method": "DnCNN",
            # For the classical methods this column holds the chosen setting. For the
            # network it holds the noise level the network was trained for.
            "param": configuration["sigma"],
            "psnr": float(np.mean(psnr_scores)),
            "ssim": float(np.mean(ssim_scores)),
        })
 
    if not dncnn_rows:
        raise SystemExit("No checkpoints found. Train the networks first (python -m src.train).")
 
    full_table = pd.concat([table_without_dncnn, pd.DataFrame(dncnn_rows)], ignore_index=True)
    full_table.to_csv(arguments.output_csv, index=False)
 
    psnr_table = full_table.pivot(index="method", columns="sigma", values="psnr")
    ssim_table = full_table.pivot(index="method", columns="sigma", values="ssim")
    ordered_methods = [method for method in METHOD_ORDER if method in psnr_table.index]
    print("\nPSNR in dB (test images):")
    print(psnr_table.loc[ordered_methods].round(2))
    print("\nSSIM (test images):")
    print(ssim_table.loc[ordered_methods].round(3))
 
    if "DnCNN" in psnr_table.index and "NLM" in psnr_table.index:
        print("\nDnCNN minus NLM in dB:")
        print((psnr_table.loc["DnCNN"] - psnr_table.loc["NLM"]).round(2))
    print(f"\nTable saved to {arguments.output_csv}")
 
 
if __name__ == "__main__":
    main()