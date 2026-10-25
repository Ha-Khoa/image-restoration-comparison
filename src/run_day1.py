"""
Tag 1: Rausch-Baselines. 
Run:
    python -m src.run_day1                    # Selftest with built-in photos
    python -m src.run_day1 --test data/Set12  --val data/BSDS500_val

Process
Step 1: Try with multiples param values 
    Example: NLM with 0.2, 0.5, 0.8, 1.0
Step 2: Evaluation on the validation set
    Calculate the PSNR for each value
Step 3: Choose the best param (highest PSNR)
Step 4: Evaluation on test set and print in the table
"""
import argparse
import numpy as np
import pandas as pd
from src.data import load_folder, sample_images, add_noise
from src.metrics import psnr, ssim
from src.classical import make_methods

SIGMAS = [0.02, 0.05, 0.10]   # Column of the Table 1
SEED = 0


def noisy_set(images, sigma, offset):
    """
    Parameter: cleaned image
    Return: noise, cleaned_image, noised_image
    """
    return [(n, g, add_noise(g, sigma, SEED + offset + i)) for i, (n, g) in enumerate(images)]


def mean_psnr(denoise_function, image_triples):
    scores = []
    for image_name, clean_image, noisy_image in image_triples:
        denoised_image = denoise_function(noisy_image)
        scores.append(psnr(denoised_image, clean_image))
    return float(np.mean(scores))


def main():
    """
    Run all the method on Test 
    Expected result: same number with the Table 1, Sec. 4.2, 5
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--test"); ap.add_argument("--val")
    ap.add_argument("--max_side", type=int, default=256)
    a = ap.parse_args()
    # Choose the data
    if a.test and a.val:
        test = load_folder(a.test, a.max_side); val = load_folder(a.val, a.max_side)
    else:
        imgs = sample_images(); val, test = imgs[:3], imgs[3:]
        print("[Selftest] : val=%d test=%d" % (len(val), len(test)))

    # Loop with sigma
    rows = []       # List of results
    for sigma in SIGMAS:

        # Create noisy validation and test images
        validation_triples = noisy_set(val, sigma, 1000)
        test_triples = noisy_set(test, sigma, 0)

        # Calculate PSNR and SSIM for noisy images without denoising
        rows.append(dict(
            sigma=sigma, 
            method="Noisy", 
            param="-",
            psnr=np.mean([psnr(y, g) for _, g, y in test_triples]),
            ssim=np.mean([ssim(y, g) for _, g, y in test_triples])))

        # Test each denoising method and its possible parameters
        for method_name, candidates in make_methods(sigma).items():    

            # Measure the average PSNR on the validation images        
            def score_on_validation(candidate):
                parameter, denoise_function = candidate
                return mean_psnr(denoise_function, validation_triples)

            # Choose the parameter with best validation score
            best_parameter, best_denoise_function = max(candidates, key=score_on_validation) 

            # Evaluate the best method on the test images
            rows.append(dict(sigma=sigma, method=method_name, param=round(float(best_parameter), 3),
                             psnr=np.mean([psnr(best_denoise_function(y), g) for _, g, y in test_triples]),
                             ssim=np.mean([ssim(best_denoise_function(y), g) for _, g, y in test_triples])))

    # Print the result
    df = pd.DataFrame(rows)
    df.to_csv("results/day1_denoise.csv", index=False)
    print(df.pivot(index="method", columns="sigma", values="psnr").round(2))
    print(df[df.method != "Noisy"].pivot(index="method", columns="sigma", values="param"))


if __name__ == "__main__":
    main()
