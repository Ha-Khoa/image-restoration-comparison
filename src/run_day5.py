"""Day 5: plug-and-play deblurring with ADMM, and the check of Remark 3.1.

Paper reference: Sec. 3.2 (Eq. 6 to 9), Remark 3.1, Sec. 3.3 (only alpha is tuned, gamma = 1),
Sec. 5.2 and Table 3 (deblurring). This script
  1. checks Remark 3.1: proximal gradient and ADMM should reach the same result;
  2. runs ADMM with a fixed denoiser (total variation and/or DnCNN) on the two deblurring
     experiments of Day 4 and adds the results to the Day 4 table.

Usage (from the project root, after run_day4 has been run):
    python -m src.run_day5 --val data/BSDS500_val --test data/BSDS500_test --max_side 256
    python -m src.run_day5 ... --denoisers tv            (without PyTorch, total variation only)

Same protocol as before: alpha is chosen on the validation images and measured on the test images.
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.blur import BlurOperator, degraded_set, gaussian_kernel
from src.data import load_folder
from src.deblur_classical import wiener_deblur
from src.denoisers import make_dncnn_denoiser, make_tv_denoiser
from src.metrics import psnr, ssim
from src.pnp import plug_and_play_admm, plug_and_play_proximal_gradient
from src.run_day1 import mean_psnr
from src.run_day4 import EXPERIMENTS, TEST_SEED_OFFSET, VALIDATION_SEED_OFFSET

# The paper stops after at most 30 iterations (Sec. 5). I use exactly 30, without a stopping rule.
NUMBER_OF_ITERATIONS = 30
# Values of alpha that are tried on the validation images (my choice; log-spaced).
ALPHA_GRID = np.geomspace(0.5, 500, 9)
# Weight of the total variation denoiser (my choice). After enough iterations only the ratio of
# this weight and alpha matters, so only alpha is tuned.
TV_DENOISER_WEIGHT = 0.05

# Remark 3.1 check: this many validation images, this many iterations, data weight 1 and step 1,
# the largest values for which the gradient step of proximal gradient is stable.
CHECK_NUMBER_OF_IMAGES = 5
CHECK_NUMBER_OF_ITERATIONS = 100


def build_denoisers(denoiser_names, network_sigmas):
    """List of (method name, denoiser function) for plug-and-play."""
    denoisers = []
    if "tv" in denoiser_names:
        denoisers.append(("PnP-TV", make_tv_denoiser(TV_DENOISER_WEIGHT)))
    if "dncnn" in denoiser_names:
        for network_sigma in network_sigmas:
            denoisers.append((f"PnP-DnCNN-{network_sigma:.2f}", make_dncnn_denoiser(network_sigma)))
    return denoisers


def check_remark_3_1(blur_operator, validation_triples, denoisers):
    """Remark 3.1: proximal gradient and ADMM have the same fixed points.

    Both are run with the same effective data weight (alpha = 1, step size 1, gamma = 1). If the
    iterations converge to the same point, the PSNR values and the images should be close.
    """
    print("Check of Remark 3.1 (alpha = 1, step size 1, gamma = 1, "
          f"{CHECK_NUMBER_OF_ITERATIONS} iterations, {CHECK_NUMBER_OF_IMAGES} validation images):")
    for method_name, denoiser in denoisers:
        psnr_by_admm, psnr_by_gradient, largest_difference = [], [], []
        for _image_name, clean_image, degraded_image in validation_triples[:CHECK_NUMBER_OF_IMAGES]:
            by_admm = plug_and_play_admm(degraded_image, blur_operator, denoiser, 1.0,
                                         CHECK_NUMBER_OF_ITERATIONS)
            by_gradient = plug_and_play_proximal_gradient(degraded_image, blur_operator, denoiser, 1.0,
                                                          1.0, CHECK_NUMBER_OF_ITERATIONS)
            psnr_by_admm.append(psnr(by_admm, clean_image))
            psnr_by_gradient.append(psnr(by_gradient, clean_image))
            largest_difference.append(np.abs(by_admm - by_gradient).max())
        print(f"  {method_name}: ADMM {np.mean(psnr_by_admm):.2f} dB, proximal gradient "
              f"{np.mean(psnr_by_gradient):.2f} dB, largest pixel difference {np.mean(largest_difference):.4f}")


def main():
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument("--val", required=True)
    argument_parser.add_argument("--test", required=True)
    argument_parser.add_argument("--max_side", type=int, default=256)
    argument_parser.add_argument("--denoisers", default="tv,dncnn", help="comma-separated: tv, dncnn")
    argument_parser.add_argument("--network_sigmas", default="0.02",
                                 help="comma-separated noise levels of the DnCNN networks to use")
    argument_parser.add_argument("--day4_csv", default="results/day4_deblur.csv")
    argument_parser.add_argument("--results_folder", default="results")
    argument_parser.add_argument("--skip_check", action="store_true", help="skip the Remark 3.1 check")
    arguments = argument_parser.parse_args()

    denoiser_names = [name.strip() for name in arguments.denoisers.split(",")]
    network_sigmas = [float(value) for value in arguments.network_sigmas.split(",")]
    denoisers = build_denoisers(denoiser_names, network_sigmas)
    validation_images = load_folder(arguments.val, arguments.max_side)
    test_images = load_folder(arguments.test, arguments.max_side)
    day4_table = pd.read_csv(arguments.day4_csv)
    Path(arguments.results_folder).mkdir(exist_ok=True)

    new_rows = []
    figure_images = {}        # per experiment: {panel name: image} for the example figure
    convergence_curves = {}   # per (experiment, method): PSNR after each iteration on one test image

    for experiment_index, experiment in enumerate(EXPERIMENTS):
        blur_operator = BlurOperator(gaussian_kernel(experiment["blur_width"]))
        validation_triples = degraded_set(validation_images, blur_operator,
                                          experiment["noise_sigma"], VALIDATION_SEED_OFFSET)
        test_triples = degraded_set(test_images, blur_operator,
                                    experiment["noise_sigma"], TEST_SEED_OFFSET)
        print(f"\n{experiment['label']}", flush=True)
        if experiment_index == 0 and not arguments.skip_check:
            check_remark_3_1(blur_operator, validation_triples, denoisers)

        _name, example_clean, example_degraded = test_triples[0]
        wiener_setting = day4_table[(day4_table.experiment == experiment["label"])
                                    & (day4_table.method == "Wiener")]["param"].astype(float).iloc[0]
        figure_images[experiment["label"]] = {
            "clean": example_clean, "Blurred": example_degraded,
            "Wiener": wiener_deblur(example_degraded, blur_operator, wiener_setting)}

        for method_name, denoiser in denoisers:
            def restore_with_alpha(degraded_image, alpha):
                return plug_and_play_admm(degraded_image, blur_operator, denoiser, alpha, NUMBER_OF_ITERATIONS)

            # Choose alpha on the validation images.
            validation_scores = []
            for alpha in ALPHA_GRID:
                restoration_function = lambda degraded_image, alpha=alpha: restore_with_alpha(degraded_image, alpha)
                validation_scores.append(mean_psnr(restoration_function, validation_triples))
            best_index = int(np.argmax(validation_scores))
            best_alpha = float(ALPHA_GRID[best_index])
            edge_note = "  (EDGE of the alpha grid: the best value may lie outside)" \
                if best_index in (0, len(ALPHA_GRID) - 1) else ""
            print(f"  {method_name}: best alpha {best_alpha:.3g}, validation PSNR "
                  f"{validation_scores[best_index]:.2f} dB{edge_note}", flush=True)

            # Measure on the test images, each image restored once.
            psnr_scores, ssim_scores = [], []
            for _image_name, clean_image, degraded_image in test_triples:
                restored_image = restore_with_alpha(degraded_image, best_alpha)
                psnr_scores.append(psnr(restored_image, clean_image))
                ssim_scores.append(ssim(restored_image, clean_image))
            new_rows.append({
                "experiment": experiment["label"], "blur_width": experiment["blur_width"],
                "noise_sigma": experiment["noise_sigma"], "method": method_name,
                "param": round(best_alpha, 4), "psnr": float(np.mean(psnr_scores)),
                "ssim": float(np.mean(ssim_scores))})

            # PSNR after every iteration on the first test image, to see whether ADMM converges.
            curve = []
            plug_and_play_admm(example_degraded, blur_operator, denoiser, best_alpha, NUMBER_OF_ITERATIONS,
                               iteration_callback=lambda _number, estimate: curve.append(psnr(estimate, example_clean)))
            convergence_curves[(experiment["label"], method_name)] = curve
            figure_images[experiment["label"]][method_name] = restore_with_alpha(example_degraded, best_alpha)

    # Add the new rows to the Day 4 table (old plug-and-play rows of the same methods are replaced).
    new_methods = {row["method"] for row in new_rows}
    full_table = pd.concat([day4_table[~day4_table.method.isin(new_methods)], pd.DataFrame(new_rows)],
                           ignore_index=True)
    csv_path = Path(arguments.results_folder) / "day5_deblur_with_pnp.csv"
    full_table.to_csv(csv_path, index=False)

    method_order = ["Blurred", "Wiener", "Richardson-Lucy"] + [name for name, _ in denoisers]
    method_order = [method for method in method_order if method in set(full_table.method)]
    print("\nPSNR in dB (test images):")
    print(full_table.pivot(index="method", columns="experiment", values="psnr").loc[method_order].round(2))
    print("\nSSIM (test images):")
    print(full_table.pivot(index="method", columns="experiment", values="ssim").loc[method_order].round(3))
    print("\nChosen settings (Wiener: K, Richardson-Lucy: iterations, plug-and-play: alpha):")
    print(full_table[full_table.method != "Blurred"].pivot(
        index="method", columns="experiment", values="param").loc[method_order[1:]])

    # Figure 1: example images. Figure 2: PSNR after each iteration.
    panel_names = ["clean", "Blurred", "Wiener"] + [name for name, _ in denoisers]
    figure, axes = plt.subplots(len(EXPERIMENTS), len(panel_names),
                                figsize=(3.0 * len(panel_names), 3.2 * len(EXPERIMENTS)), squeeze=False)
    for row_index, experiment in enumerate(EXPERIMENTS):
        images_of_experiment = figure_images[experiment["label"]]
        for column_index, panel_name in enumerate(panel_names):
            axis = axes[row_index][column_index]
            panel_image = images_of_experiment[panel_name]
            axis.imshow(np.clip(panel_image, 0, 1), cmap="gray", vmin=0, vmax=1)
            if panel_name == "clean":
                axis.set_title(f"clean ({experiment['label']})", fontsize=8)
            else:
                axis.set_title(f"{panel_name}  {psnr(panel_image, images_of_experiment['clean']):.2f} dB", fontsize=8)
            axis.axis("off")
    figure.tight_layout()
    figure.savefig(Path(arguments.results_folder) / "day5_example.png", dpi=110)

    figure, axes = plt.subplots(1, len(EXPERIMENTS), figsize=(5.5 * len(EXPERIMENTS), 3.6), squeeze=False)
    for column_index, experiment in enumerate(EXPERIMENTS):
        axis = axes[0][column_index]
        for method_name, _denoiser in denoisers:
            axis.plot(range(1, NUMBER_OF_ITERATIONS + 1), convergence_curves[(experiment["label"], method_name)],
                      label=method_name)
        axis.set_title(experiment["label"], fontsize=9)
        axis.set_xlabel("ADMM iteration")
        axis.set_ylabel("PSNR (dB), first test image")
        axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(Path(arguments.results_folder) / "day5_convergence.png", dpi=110)
    print(f"\nTable saved to {csv_path}; figures saved to {arguments.results_folder}/day5_example.png "
          "and day5_convergence.png")


if __name__ == "__main__":
    main()