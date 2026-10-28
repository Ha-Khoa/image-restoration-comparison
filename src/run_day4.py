"""
Day 4: classical deblurring baselines (blurred image, Wiener filter, Richardson-Lucy).

Paper reference: Sec. 5.2 and Table 3 (deblurring, PSNR per experiment), and Fig. 5 for the
noise levels. This script builds the table layout of Table 3 with my own experiments and
fills it with the classical methods. Day 5 adds plug-and-play with the network.

Usage (from the project root):
    python -m src.run_day4 --val data/BSDS500_val --test data/BSDS500_test --max_side 256

Same protocol as Day 1: the setting of each method (K, number of iterations) is chosen on
the validation images and measured on the test images. The test images are not used for tuning.
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # draw into a file, no window needed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.blur import BlurOperator, degraded_set, gaussian_kernel
from src.data import load_folder
from src.deblur_classical import make_deblur_methods
from src.metrics import psnr, ssim
from src.run_day1 import mean_psnr

# My own experiments (NOT the five experiments of the paper). The blur width is the standard
# deviation of the Gaussian kernel in pixels. The noise level 0.04 is the level that the
# caption of Fig. 5 in the paper gives for experiment a; 0.01 is the level given for
# experiment e. The blur kernel itself is my choice.
EXPERIMENTS = [
    {"label": "blur 1.6, noise 0.04", "blur_width": 1.6, "noise_sigma": 0.04},
    {"label": "blur 1.6, noise 0.01", "blur_width": 1.6, "noise_sigma": 0.01},
]

VALIDATION_SEED_OFFSET = 1000  # same offsets as in Day 1
TEST_SEED_OFFSET = 0


def main():
    # Read parameters from terminal
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument("--val", required=True, help="folder with validation images")
    argument_parser.add_argument("--test", required=True, help="folder with test images")
    argument_parser.add_argument("--max_side", type=int, default=256)
    argument_parser.add_argument("--results_folder", default="results")
    arguments = argument_parser.parse_args()

    validation_images = load_folder(arguments.val, arguments.max_side)
    test_images = load_folder(arguments.test, arguments.max_side)
    # Result folder
    Path(arguments.results_folder).mkdir(exist_ok=True)

    result_rows = []
    example_restorations = {}  # per experiment: images for the example figure

    # 2 experiments through 
    for experiment in EXPERIMENTS:
        # Gaussian Blur and operator
        blur_operator = BlurOperator(gaussian_kernel(experiment["blur_width"]))
        # Validation data
        validation_triples = degraded_set(validation_images, blur_operator,
                                          experiment["noise_sigma"], VALIDATION_SEED_OFFSET)
        # Test data
        test_triples = degraded_set(test_images, blur_operator,
                                    experiment["noise_sigma"], TEST_SEED_OFFSET)
        print(f"\n{experiment['label']}", flush=True)

        def add_result_row(method_name, setting_value, restoration_function):
            """
            Measure one method on all test images and store one row of the table
            """
            psnr_scores = []
            ssim_scores = []
            for _image_name, clean_image, degraded_image in test_triples:
                restored_image = restoration_function(degraded_image)  # computed once
                psnr_scores.append(psnr(restored_image, clean_image))
                ssim_scores.append(ssim(restored_image, clean_image))
            result_rows.append({
                "experiment": experiment["label"],
                "blur_width": experiment["blur_width"],
                "noise_sigma": experiment["noise_sigma"],
                "method": method_name,
                "param": setting_value,
                "psnr": float(np.mean(psnr_scores)),
                "ssim": float(np.mean(ssim_scores)),
            })

        # Reference row: the degraded image with nothing done to it.
        add_result_row("Blurred", "-", lambda degraded_image: degraded_image)
        example_restorations[experiment["label"]] = {
            "clean": test_triples[0][1], "Blurred": test_triples[0][2]}

        # Choose the best parameter on validation
        for method_name, candidates in make_deblur_methods(blur_operator).items():
            def score_on_validation(candidate):
                _setting_value, restoration_function = candidate
                return mean_psnr(restoration_function, validation_triples)
            # Highest PSNR
            best_setting, best_function = max(candidates, key=score_on_validation)
            # Evaluate on test
            add_result_row(method_name, round(float(best_setting), 5), best_function)
            # Save the restored images
            example_restorations[experiment["label"]][method_name] = best_function(test_triples[0][2])
            print(f"  {method_name}: chosen setting {best_setting:.5g}", flush=True)

    results_table = pd.DataFrame(result_rows)
    csv_path = Path(arguments.results_folder) / "day4_deblur.csv"
    results_table.to_csv(csv_path, index=False)

    # List of method
    method_order = ["Blurred", "Wiener", "Richardson-Lucy"]
    # PSNR Table
    psnr_table = results_table.pivot(index="method",
                                    columns="experiment",
                                    values="psnr")
    # SSIM Table
    ssim_table = results_table.pivot(index="method",
                                    columns="experiment",
                                    values="ssim")
    # print("PSNR table index:", psnr_table.index.tolist())
    # print("SSIM table index:", ssim_table.index.tolist())
    setting_table = results_table[results_table.method != "Blurred"].pivot(
        index="method", columns="experiment", values="param")
    print("\nPSNR in dB (test images):")
    print(psnr_table.loc[method_order].round(2))
    print("\nSSIM (test images):")
    print(ssim_table.loc[method_order].round(3))
    print("\nChosen settings (Wiener: K, Richardson-Lucy: number of iterations):")
    print(setting_table)

    # Example figure: first test image, one row per experiment.
    figure, axes = plt.subplots(len(EXPERIMENTS), 4, figsize=(12, 3.2 * len(EXPERIMENTS)), squeeze=False)
    for row_index, experiment in enumerate(EXPERIMENTS):
        images_of_experiment = example_restorations[experiment["label"]]
        clean_image = images_of_experiment["clean"]
        for column_index, key in enumerate(["clean", "Blurred", "Wiener", "Richardson-Lucy"]):
            axis = axes[row_index][column_index]
            axis.imshow(np.clip(images_of_experiment[key], 0, 1), cmap="gray", vmin=0, vmax=1)
            # Show PSNR on the image
            title = key if key == "clean" else f"{key}  {psnr(images_of_experiment[key], clean_image):.2f} dB"
            axis.set_title(title if column_index > 0 else f"clean ({experiment['label']})", fontsize=9)
            axis.axis("off")
    figure.tight_layout()
    # Save the image
    figure_path = Path(arguments.results_folder) / "day4_example.png"
    figure.savefig(figure_path, dpi=110)
    print(f"\nTable saved to {csv_path}, example figure saved to {figure_path}")


if __name__ == "__main__":
    main()