"""
Classical debluring methods: Wiener filter and Richardson-Lucy

Paper reference: Sec. 1 (Equation 1): u = argmin H_f(Au) + R(u)
Both methods below solve a version of this problem for a blur operator A, without a neural network


"""

import numpy as np

def wiener_deblur(degraded_image, blur_operator, noise_to_signal_ratio):
    """
    Wiener filter with one constant noise-to-signal ratio K.

    In the Fourier domain: restored = conj(H) * degraded / (|H|^2 + K)
    solution of min_u 1/2 ||A u - f ||^ 2 + K/2 ||u||^2
    without K the division by tiny H would blow up the noise
    Larger K means a smoother result with less noise but also less sharpening
    """

    # Get the blur kernel matrix (H) in the frequency domain
    transfer_function = blur_operator.transfer_function(degraded_image.shape)
    degraded_spectrum = np.fft.fft2(degraded_image)

    # Wiener formula: Multiply by conj(H) and add K (noise_to_signal_ratio)
    restored_spectrum = (np.conj(transfer_function) * degraded_spectrum / (np.abs(transfer_function)**2 + noise_to_signal_ratio))

    # Convert back to spatial domain
    return np.real(np.fft.ifft2(restored_spectrum))

def richardson_lucy_deblur(degraded_image, blur_operator, number_of_iterations, minimum_value=1e-3):
    """
    Richard-Lucy iteration, started from the degraded image

    Each iteration will blur the current image to see if the simulated result matches the actual
    degraded image. If there are any discrepancies, it will apply a scaling factor to readjust 
    pixel structure

    estimate = estimate * A^T(degraded / A estimate)

    A^T is the blur with flipped kernel (in Fourier Domain is multiplication with conj(H))

    """
    # Get the blur kernel matrix (H) in frequncy domain
    transfer_function = blur_operator.transfer_function(degraded_image.shape)

    # Threshold
    observed_image = np.maximum(degraded_image, minimum_value)

    # Set the initial guess image as a copy of the blurry input
    estimate = observed_image.copy()

    # Iterations: more loops -> sharpness but amplify noise
    for _ in range(number_of_iterations):

        # Simulate blurring the current estimated clean image
        blurred_estimate = np.real(np.fft.ifft2(np.fft.fft2(estimate) * transfer_function))
        # Compute error ratio between actual blurry image
        ratio = observed_image / np.maximum(blurred_estimate, minimum_value)
        # Project the error back to the spatial domain using conj(H)
        correction = np.real(np.fft.ifft2(np.fft.fft2(ratio) * np.conj(transfer_function)))
        # Multiply the correction factor into estimate to sharpen it
        estimate = estimate * correction

    return estimate

def make_deblur_methods(blur_operator):
    """
    Candidate settings for each method, in the same formate as make_methods in classical.py

    Returns {method name: [(setting value, function), ...]}
    Every function takes a degraded image and returns the restored image. The grids are my own choice
    """

    return {
        "Wiener": [
            (noise_to_signal_ratio,
             lambda degraded_image, setting=noise_to_signal_ratio:
                wiener_deblur(degraded_image, blur_operator, setting))
             for noise_to_signal_ratio in np.geomspace(1e-4, 1.0, 13)
        ],
        "Richardson-Lucy": [
            (number_of_iterations,
             lambda degraded_image, setting=number_of_iterations:
                richardson_lucy_deblur(degraded_image, blur_operator, setting))
            for number_of_iterations in [1, 2, 3,5, 8, 12, 20, 30, 50]
        ],
    }