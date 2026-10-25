# image-restoration-comparison

## Einführung

Dieses Projekt ist eine kleine Lernübung. Ich lese das Paper *Learning Proximal Operators* (Meinhardt, Moeller, Hazirbas, Cremers, ICCV, 2017) und baue einen Teil davon selbst nach.

Das Paper zeigt, wie man ein neuronales Netz, das Bildrauschen entfernt, in ein Verfahren einbauen kann, das verschwommene Bilder wieder schärft.

Stand: Das Entfernen von Bildrauschen ist fertig. Ich vergleiche vier klassische Verfahren mit einem neuronalen Netz (DnCNN). Noch offen: das Schärfen verschwommener Bilder und die Komnination mit dem Verfahren aus dem Paper.

Wichtig: Das ist meine eigene, vereinfachte Version. Ich benutze andere Bilder und eigene Einstellungen. Meine Zahlen darf man deshalb nicht direkt mit den Tabellen im Paper vergleichen.

---

## What I did

1. Take clean photos and add random noise on purpose, so I know the perfect answer.
2. Remove the noise with four classical methods and with a neural network.
3. Measure how close each result is to the clean photo.

In the paper, the damaged image is `f = A·u + n` (`u` = clean image, `n` = noise). For plain
denoising, `A` does nothing, so it is `f = u + n`.

**Noise strength** is called sigma (σ). Image values are between 0 and 1. I use σ = 0.02
(weak), 0.05 and 0.10 (strong), the same levels as Table 1 in the paper.

**Methods**

- Gaussian filter: averages each pixel with its neighbours. Removes noise, but blurs edges.
- Median filter: takes the middle value of the neighbours.
- Total variation denoising: looks for an image that is close to the noisy one but has few
  jumps between neighbouring pixels.
- Non-local means: averages pixels whose surroundings look alike, even if they are far apart.
- DnCNN: a neural network that learns the noise from many examples (details below).

**Measures**

- PSNR (peak signal-to-noise ratio), in decibels (dB): how close the result is to the clean
  image. Higher is better. The paper uses it.
- SSIM (structural similarity), from 0 to 1: compares the structure of two images. 1 means
  identical. The paper does not use it. I added it.

**Choosing the settings.** 
Each classical method has one setting (for example, the width of
the Gaussian filter). I try several values on 25 validation images and keep the one with the
best PSNR. Then I measure on 25 different test images. So I never grade a method on the images
used to tune it. This split is my own choice. The paper does not describe it.

## Data

- BSDS500 (Berkeley Segmentation Dataset), taken from the GitHub copy `BIDS/BSDS500`. The
  original is on the website of the Berkeley computer vision group. The images are **not**
  in this repository.
  Link to the website: https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/resources.html 
- 25 validation images and 25 test images, evenly spaced over the sorted file list (no
  randomness). The lists are in `data/BSDS500_val/selected_images.txt` and
  `data/BSDS500_test/selected_images.txt`. For training the network I use the 200 `train` images.
- Images are grayscale with values from 0 to 1, shrunk so the longer side is at most 256 pixels.
  This makes them a little smoother than the standard images in the paper.
- Noise is not cut off at 0 and 1. Values are cut only when measuring.
- Random seeds are fixed. Every method sees the same noisy images. Validation and test images
  get different noise.

## The network (DnCNN)

- Paper (Section 4.2): 17 layers with 3×3 filters, each followed by a ReLU (a step that sets
  negative values to zero). Mine has 17 layers, 64 channels and 556,097 learned numbers.
- The paper does not give every detail, so I followed the original DnCNN paper (Zhang et al.,
  IEEE Transactions on Image Processing, 2017). I checked these three points against it:
  the first layer has a ReLU, the middle layers have batch normalization between the
  convolution and the ReLU (it keeps the values inside the network in a stable range), and the
  last layer has no ReLU. The network guesses the **noise**; the result is the noisy image
  minus the guessed noise (called residual learning). Like the original, I pad the images with
  zeros so the output has the same size as the input.
- Training: each step uses 128 random 40×40 patches of the training images, with random
  rotation, flipping and fresh noise. Optimizer: Adam, as in the paper. The learning rate
  falls from 0.001 to 0.00001. 6000 steps, about 32 minutes per network on a MacBook
  (Apple M1).
- The paper says it uses the training method of the original DnCNN, with Adam. I did not copy
  it exactly. Patch size (40×40) and batch size (128) are the same as in the original DnCNN.
  The rest differs: the original used 400 training images of 180×180 pixels and trained for
  50 epochs, which is about 80,000 steps by my reading. I use the 200 BSDS500 training images
  and only 6000 steps, so my networks are trained much more briefly. The learning rate and its
  schedule are my own choices.
- One network per noise strength, as in the paper (Section 4.2). Every 250 steps I measure
  PSNR on the validation images and keep the best network. Test images are never used for training.

## Results

### Test images (25 images)

PSNR in dB (higher is better):

| Method | σ = 0.02 | σ = 0.05 | σ = 0.10 |
|---|---|---|---|
| Noisy image (nothing done) | 34.03 | 26.14 | 20.26 |
| Gaussian filter | 35.00 | 29.17 | 26.84 |
| Median filter | 29.83 | 28.19 | 25.34 |
| Total variation | 36.47 | 31.02 | 27.74 |
| Non-local means | 36.64 | 31.07 | 27.85 |
| DnCNN | 37.79 | 32.50 | 29.06 |

SSIM (closer to 1 is better):

| Method | σ = 0.02 | σ = 0.05 | σ = 0.10 |
|---|---|---|---|
| Noisy image (nothing done) | 0.888 | 0.639 | 0.393 |
| Gaussian filter | 0.911 | 0.760 | 0.705 |
| Median filter | 0.857 | 0.766 | 0.599 |
| Total variation | 0.949 | 0.859 | 0.750 |
| Non-local means | 0.957 | 0.859 | 0.749 |
| DnCNN | 0.968 | 0.905 | 0.813 |

Chosen settings of the classical methods (Gaussian: width in pixels, median: window size,
total variation: weight, non-local means: factor h):

| Method | σ = 0.02 | σ = 0.05 | σ = 0.10 |
|---|---|---|---|
| Gaussian filter | 0.4 | 0.5 | 1.0 |
| Median filter | 3 | 3 | 3 |
| Total variation | 0.008 | 0.029 | 0.068 |
| Non-local means | 0.8 | 0.8 | 0.6 |

### Training of the network (validation images, not test images)

| σ | Noisy images (dB) | DnCNN after 6000 steps (dB) | Gain (dB) |
|---|---|---|---|
| 0.02 | 34.03 | 37.73 | 3.70 |
| 0.05 | 26.17 | 32.39 | 6.22 |
| 0.10 | 20.32 | 28.91 | 8.59 |

Most of the gain comes in the first 1000 steps. After step 3000 the PSNR rises only slowly.
These numbers are on validation images, so they cannot be compared with the table above.
The results on the test images (37.79, 32.50 and 29.06 dB) are close to these validation
results (37.73, 32.39 and 28.91 dB).

## What the results do and do not show

- **First check:** my "noisy image" numbers (34.03 / 26.14 / 20.26) are close to the "Noisy"
  column in Table 1 of the paper (33.99 / 26.06 / 20.14). So my noise and PSNR look right.
  The other numbers cannot be compared with the paper, because the images differ.
- **DnCNN is the best method at all three noise levels**, in PSNR and in SSIM. It is 1.15,
  1.42 and 1.21 dB above non-local means, the best classical method, at σ = 0.02, 0.05 and
  0.10. These gaps are much larger than the 0.17 dB between total variation and non-local
  means. In SSIM the gain over non-local means is small at σ = 0.02 (0.968 against 0.957)
  and larger at σ = 0.05 (0.905 against 0.859) and σ = 0.10 (0.813 against 0.749).
- Compared with the unprocessed noisy images, DnCNN gains 3.76, 6.36 and 8.80 dB. In Table 1
  of the paper, DnCNN-S gains about 3.8, 6.1 and 8.0 dB, so the gains are of similar size.
  The images are different, so this is not the same experiment.
- In Table 1 of the paper, DnCNN-S is 2.3, 2.4 and 1.7 dB above non-local means. My gaps are
  smaller. One possible reason is that my non-local means setting was tuned on validation
  images and is told the true noise strength, so it may be stronger than the one in the paper.
  Another is that my networks are trained much more briefly. I did not test either reason.
- Only 25 test images, one training run per network, no error ranges. Differences of a few
  tenths of a dB (for example total variation against non-local means, at most 0.17 dB) are
  not reliable.
- The median filter is poor for this kind of noise. At σ = 0.02 it is worse than doing nothing.
- Some chosen settings lie at the edge of the values I tried (median window 3, non-local means
  h = 0.6 at σ = 0.10), so the best value could be outside my range.
- Non-local means is told the true noise strength. That is a small advantage.
- BM3D, a strong classical method that the paper compares with, is not included.

## What went wrong first

My first training run (σ = 0.02) used a random start for the last layer. After about 2900
steps (the run stopped because my computer was switched off) the network only matched the
noisy images: 34.06 dB against 34.03 dB. It had learned to do nothing. After I set the last
layer to start at zero, a short test run of 600 steps already reached 35.71 dB. My guess is
that the random start made the network spend a long time just to get down to "do nothing".
I did not test other explanations, so this is a guess, not a proven fact.

## How to run

    python -m pip install -r requirements.txt

    # Choose the 25 validation and 25 test images from BSDS500
    python -m src.make_subsets --source <path>/BSDS500/data/images --count 25

    # Classical methods
    python -m src.run_day1 --val data/BSDS500_val --test data/BSDS500_test --max_side 256

    # Train the network for one noise strength (repeat for 0.05 and 0.10)
    python -m src.train --train_folder <path>/BSDS500/data/images/train \
        --val_folder data/BSDS500_val --sigma 0.02 --number_of_layers 17 --number_of_steps 6000

    # Measure the networks on the test images and add them to the table
    python -m src.run_day3 --test data/BSDS500_test --max_side 256

Results are saved in `results/`. Trained networks are in `checkpoints/`.

## Files

- `src/data.py`: loads images and adds noise
- `src/metrics.py`: PSNR and SSIM
- `src/classical.py`: the four classical methods
- `src/model.py`: the DnCNN network
- `src/train.py`: trains the network
- `src/run_day1.py`: experiment for the classical methods
- `src/run_day3.py`: measures the network on the test images
- `src/make_subsets.py`: chooses the validation and test images

## What comes next



## References

- T. Meinhardt, M. Moeller, C. Hazirbas, D. Cremers. *Learning Proximal Operators: Using
  Denoising Networks for Regularizing Inverse Imaging Problems.* ICCV 2017.
  arXiv: https://arxiv.org/abs/1704.03488, DOI: 10.1109/ICCV.2017.198
- K. Zhang, W. Zuo, Y. Chen, D. Meng, L. Zhang. *Beyond a Gaussian Denoiser: Residual Learning
  of Deep CNN for Image Denoising.* IEEE Transactions on Image Processing 26(7), 2017.
  arXiv: https://arxiv.org/abs/1608.03981
- BSDS500 (Berkeley Segmentation Dataset and Benchmark): see the Data section above.

This repository does not contain the papers or the images. Please use the links above.