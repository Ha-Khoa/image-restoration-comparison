"""Day 6: how the best data weight alpha depends on the noise level of the network, and an ablation.

Paper reference: Sec. 3.3 and Fig. 2. The paper trains denoising networks for different noise
levels sigma and finds, for each network, the alpha with the best PSNR. It reports that the best
alpha follows a parabola, alpha = p * sigma^2, and that the PSNR stays stable once sigma is large
enough. This script repeats that on a small scale with my three networks (sigma = 0.02, 0.05, 0.10)
and my two deblurring experiments.

Two studies:
  1. alpha-sigma study: for every network and experiment, alpha is searched on a grid on the
     validation images. Then the result is measured on the test images.
  2. ablation "decreasing sigma": instead of one fixed network, ADMM uses the network of
     sigma = 0.10 in the first third of the iterations, then 0.05, then 0.02. This idea comes from
     later work (IRCNN style), NOT from the paper. The paper keeps one fixed network.

Usage (from the project root; needs the three networks from train.py and src.run_day4):
    python -m src.run_day6 --val data/BSDS500_val --test data/BSDS500_test --max_side 256
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
from src.denoisers import make_dncnn_denoiser
from src.metrics import psnr, ssim
from src.pnp import plug_and_play_admm
from src.run_day1 import mean_psnr
from src.run_day4 import EXPERIMENTS, TEST_SEED_OFFSET, VALIDATION_SEED_OFFSET
from src.run_day5 import NUMBER_OF_ITERATIONS

# The best alpha grows with sigma^2 in the paper, so the grid must be wide: from sigma 0.02 to 0.10 the
# expected best alpha grows by a factor of 25. Log-spaced, my choice.
ALPHA_GRID = np.geomspace(0.5, 5000, 13)


def refine_best_alpha(alpha_grid, validation_scores):
    """Best alpha between the grid points.

    A parabola is fitted through the best grid point and its two neighbours (in log alpha), and its
    peak is taken. If the best point is at the edge of the grid, it is returned unchanged.
    """
    best_index = int(np.argmax(validation_scores))
    if best_index in (0, len(alpha_grid) - 1):
        return float(alpha_grid[best_index])
    log_alphas = np.log(alpha_grid[best_index - 1:best_index + 2])
    scores = np.asarray(validation_scores[best_index - 1:best_index + 2])
    curvature, slope, _offset = np.polyfit(log_alphas, scores, 2)
    if curvature >= 0:  # not a peak, keep the grid value
        return float(alpha_grid[best_index])
    return float(np.exp(-slope / (2 * curvature)))


def make_scheduled_denoiser(denoisers_in_order, iterations_per_stage):
    """A denoiser that switches network as the iterations go on.

    ADMM calls the denoiser exactly once per iteration, so counting calls counts iterations.
    A new scheduled denoiser must be made for every image (the counter starts at zero).
    """
    call_counter = {"count": 0}

    def denoise(image):
        stage_index = min(call_counter["count"] // iterations_per_stage, len(denoisers_in_order) - 1)
        call_counter["count"] += 1
        return denoisers_in_order[stage_index](image)

    return denoise


def choose_alpha_and_measure(restoration_function, alpha_grid, validation_triples, test_triples):
    """Search alpha on the validation images, then measure on the test images.

    restoration_function(degraded_image, alpha) returns the restored image.
    Returns (grid-best alpha, validation scores for the whole grid, mean test PSNR, mean test SSIM).
    """
    validation_scores = []
    for alpha in alpha_grid:
        function_for_alpha = lambda degraded_image, alpha=alpha: restoration_function(degraded_image, alpha)
        validation_scores.append(mean_psnr(function_for_alpha, validation_triples))
    best_alpha = float(alpha_grid[int(np.argmax(validation_scores))])
    psnr_scores, ssim_scores = [], []
    for _image_name, clean_image, degraded_image in test_triples:
        restored_image = restoration_function(degraded_image, best_alpha)  # each image restored once
        psnr_scores.append(psnr(restored_image, clean_image))
        ssim_scores.append(ssim(restored_image, clean_image))
    return best_alpha, validation_scores, float(np.mean(psnr_scores)), float(np.mean(ssim_scores))


def run_study(validation_images, test_images, network_sigmas, denoiser_factory,
              alpha_grid, results_folder):
    """Both studies. denoiser_factory(sigma) returns the denoiser for the network of that sigma."""
    denoisers = {sigma: denoiser_factory(sigma) for sigma in network_sigmas}
    sigmas_decreasing = sorted(network_sigmas, reverse=True)
    iterations_per_stage = NUMBER_OF_ITERATIONS // len(sigmas_decreasing)

    result_rows = []
    validation_curves = {}  # (experiment label, network sigma) -> validation PSNR for every alpha
    interpolated_alpha = {}  # (experiment label, network sigma) -> best alpha between grid points

    for experiment in EXPERIMENTS:
        blur_operator = BlurOperator(gaussian_kernel(experiment["blur_width"]))
        validation_triples = degraded_set(validation_images, blur_operator,
                                          experiment["noise_sigma"], VALIDATION_SEED_OFFSET)
        test_triples = degraded_set(test_images, blur_operator,
                                    experiment["noise_sigma"], TEST_SEED_OFFSET)
        print(f"\n{experiment['label']}", flush=True)

        # Study 1: one fixed network at a time.
        for network_sigma in sigmas_decreasing[::-1]:
            denoiser = denoisers[network_sigma]
            restoration_function = lambda degraded_image, alpha, denoiser=denoiser: plug_and_play_admm(
                degraded_image, blur_operator, denoiser, alpha, NUMBER_OF_ITERATIONS)
            best_alpha, validation_scores, test_psnr, test_ssim = choose_alpha_and_measure(
                restoration_function, alpha_grid, validation_triples, test_triples)
            edge_note = best_alpha in (alpha_grid[0], alpha_grid[-1])
            validation_curves[(experiment["label"], network_sigma)] = validation_scores
            interpolated_alpha[(experiment["label"], network_sigma)] = refine_best_alpha(alpha_grid, validation_scores)
            result_rows.append({
                "study": "fixed network", "experiment": experiment["label"], "network_sigma": network_sigma,
                "alpha": best_alpha, "alpha_interpolated": interpolated_alpha[(experiment["label"], network_sigma)],
                "alpha_at_grid_edge": edge_note, "validation_psnr": max(validation_scores),
                "test_psnr": test_psnr, "test_ssim": test_ssim})
            print(f"  network {network_sigma:.2f}: best alpha {best_alpha:.3g} "
                  f"(between grid points: {interpolated_alpha[(experiment['label'], network_sigma)]:.3g}), "
                  f"test PSNR {test_psnr:.2f} dB, SSIM {test_ssim:.3f}"
                  + ("   <-- EDGE of the alpha grid" if edge_note else ""), flush=True)

        # Study 2: network changes during the iterations (largest sigma first).
        def scheduled_restoration(degraded_image, alpha):
            scheduled_denoiser = make_scheduled_denoiser(
                [denoisers[sigma] for sigma in sigmas_decreasing], iterations_per_stage)
            return plug_and_play_admm(degraded_image, blur_operator, scheduled_denoiser, alpha,
                                      NUMBER_OF_ITERATIONS)

        best_alpha, validation_scores, test_psnr, test_ssim = choose_alpha_and_measure(
            scheduled_restoration, alpha_grid, validation_triples, test_triples)
        edge_note = best_alpha in (alpha_grid[0], alpha_grid[-1])
        result_rows.append({
            "study": "decreasing sigma", "experiment": experiment["label"], "network_sigma": np.nan,
            "alpha": best_alpha, "alpha_interpolated": refine_best_alpha(alpha_grid, validation_scores),
            "alpha_at_grid_edge": edge_note, "validation_psnr": max(validation_scores),
            "test_psnr": test_psnr, "test_ssim": test_ssim})
        print(f"  decreasing sigma {sigmas_decreasing}: best alpha {best_alpha:.3g}, test PSNR "
              f"{test_psnr:.2f} dB, SSIM {test_ssim:.3f}" + ("   <-- EDGE of the alpha grid" if edge_note else ""),
              flush=True)

    results_table = pd.DataFrame(result_rows)
    Path(results_folder).mkdir(exist_ok=True)
    results_table.to_csv(Path(results_folder) / "day6_alpha_sigma.csv", index=False)
    print_summary(results_table, interpolated_alpha, network_sigmas)
    make_figures(results_table, validation_curves, interpolated_alpha, alpha_grid, network_sigmas, results_folder)
    return results_table


def print_summary(results_table, interpolated_alpha, network_sigmas):
    fixed_rows = results_table[results_table.study == "fixed network"]
    print("\nTest PSNR in dB (fixed network, alpha chosen on validation images):")
    print(fixed_rows.pivot(index="network_sigma", columns="experiment", values="test_psnr").round(2))
    print("\nBest alpha (grid value):")
    print(fixed_rows.pivot(index="network_sigma", columns="experiment", values="alpha").round(3))

    print("\nFit alpha = p * sigma^2 (the relation of Fig. 2 in the paper), per experiment:")
    sigmas = np.array(sorted(network_sigmas))
    for experiment in EXPERIMENTS:
        alphas = np.array([interpolated_alpha[(experiment["label"], sigma)] for sigma in sigmas])
        # Least squares in log space: log alpha = log p + 2 log sigma  ->  log p is the mean of the differences.
        constant_p = float(np.exp(np.mean(np.log(alphas) - 2 * np.log(sigmas))))
        slope = float(np.polyfit(np.log(sigmas), np.log(alphas), 1)[0]) if len(sigmas) > 1 else float("nan")
        print(f"  {experiment['label']}: p = {constant_p:.3g}; slope of log alpha against log sigma = {slope:.2f} "
              "(the paper's parabola means slope 2)")

    print("\nAblation, test PSNR in dB:")
    ablation = results_table.copy()
    ablation["variant"] = np.where(ablation.study == "decreasing sigma", "decreasing sigma (0.10 > 0.05 > 0.02)",
                                   "fixed network " + ablation.network_sigma.map("{:.2f}".format))
    print(ablation.pivot(index="variant", columns="experiment", values="test_psnr").round(2))


def make_figures(results_table, validation_curves, interpolated_alpha, alpha_grid, network_sigmas, results_folder):
    sigmas = sorted(network_sigmas)
    # Figure 1: validation PSNR against alpha for each network.
    figure, axes = plt.subplots(1, len(EXPERIMENTS), figsize=(5.5 * len(EXPERIMENTS), 3.8), squeeze=False)
    for column_index, experiment in enumerate(EXPERIMENTS):
        axis = axes[0][column_index]
        for sigma in sigmas:
            axis.plot(alpha_grid, validation_curves[(experiment["label"], sigma)], marker="o", markersize=3,
                      label=f"network sigma {sigma:.2f}")
        axis.set_xscale("log")
        axis.set_xlabel("alpha (data weight)")
        axis.set_ylabel("PSNR on validation images (dB)")
        axis.set_title(experiment["label"], fontsize=9)
        axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(Path(results_folder) / "day6_alpha_curves.png", dpi=110)

    # Figure 2 (like Fig. 2 of the paper): best alpha against sigma, and PSNR against sigma.
    figure, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for experiment in EXPERIMENTS:
        alphas = np.array([interpolated_alpha[(experiment["label"], sigma)] for sigma in sigmas])
        constant_p = float(np.exp(np.mean(np.log(alphas) - 2 * np.log(np.array(sigmas)))))
        line = axes[0].plot(sigmas, alphas, marker="o", label=experiment["label"])[0]
        smooth_sigmas = np.linspace(min(sigmas), max(sigmas), 50)
        axes[0].plot(smooth_sigmas, constant_p * smooth_sigmas ** 2, linestyle="--", color=line.get_color(),
                     label=f"fit p*sigma^2, p={constant_p:.3g}")
        rows = results_table[(results_table.study == "fixed network") & (results_table.experiment == experiment["label"])]
        axes[1].plot(rows.network_sigma, rows.test_psnr, marker="o", label=experiment["label"])
    axes[0].set_xlabel("sigma of the network")
    axes[0].set_ylabel("best alpha (from validation images)")
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].legend(fontsize=7)
    axes[1].set_xlabel("sigma of the network")
    axes[1].set_ylabel("PSNR on test images (dB)")
    axes[1].legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(Path(results_folder) / "day6_alpha_vs_sigma.png", dpi=110)
    print(f"\nTable saved to {results_folder}/day6_alpha_sigma.csv, figures saved to "
          f"{results_folder}/day6_alpha_curves.png and day6_alpha_vs_sigma.png")


def main():
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument("--val", required=True)
    argument_parser.add_argument("--test", required=True)
    argument_parser.add_argument("--max_side", type=int, default=256)
    argument_parser.add_argument("--network_sigmas", default="0.02,0.05,0.10")
    argument_parser.add_argument("--number_of_validation_images", type=int, default=10,
                                 help="alpha is searched on this many validation images (to save time)")
    argument_parser.add_argument("--results_folder", default="results")
    arguments = argument_parser.parse_args()

    network_sigmas = [float(value) for value in arguments.network_sigmas.split(",")]
    all_validation_images = load_folder(arguments.val, arguments.max_side)
    # Evenly spaced subset of the validation images.
    subset_indices = np.linspace(0, len(all_validation_images) - 1, arguments.number_of_validation_images)
    validation_images = [all_validation_images[int(index)] for index in subset_indices.round()]
    test_images = load_folder(arguments.test, arguments.max_side)
    print(f"alpha is searched on {len(validation_images)} validation images, "
          f"measured on {len(test_images)} test images")
    run_study(validation_images, test_images, network_sigmas, make_dncnn_denoiser,
              ALPHA_GRID, arguments.results_folder)


if __name__ == "__main__":
    main()