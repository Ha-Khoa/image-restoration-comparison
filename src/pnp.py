"""
Plug-and-play restoration with a fixed denoiser

Paper reference: Sec. 3.2 and Remark 3.1. The paper solves
    u = argmin H_f(Au) + R(u)
with algorithms that only need the proximal operator of R.
A proximal operator is a denoising step (Equation 2), so it is replaced by a fixed denoiser G
The data term is H_f(Au) = alpha/2 * ||Au - f||^2 (Sec. 3.3), where A is the blur operator and f the damaged/degraded image

Two algorithms from Sec. 3.2 are written here:
    - proximal gradient (Eq. 6), used as a quick check
    - ADMM (Eq. 7 to 9), which is the one used for the experiments
Remark 3.1 says that both have the same fixed points. The data step of ADMM is solved with the FFT (matching blur.py)

The denoiser is just an argument, so the same code runs with total variation, non-local means or DnCNN
"""
import numpy as np

def plug_and_play_proximal_gradient(
        degraded_image,
        blur_operator, 
        denoiser, 
        data_weight, 
        step_size, 
        number_of_iterations, 
        iteration_callback=None):
    """
    Proximal gradient with a denoiser (Equation 6 of the paper)

        u_next = G u - step_size * A^T * grad H_f(Au)
            - grad H_f(Au) = data_weight * (Au - f)
    
    So each iteration is: one gradient step on the data term, the one denoising step.
    A^T is multiplication with conj(H) in the Fourier domain
    """
    # Call the Blur operator A
    transfer_function = blur_operator.transfer_function(degraded_image.shape)
    # The gradient step is only stabe if step_size * data_weight * ||A||^2 <= 1
    # ||A||^2 is the largest value of |H|^2
    stability_number = step_size * data_weight * np.max(np.abs(transfer_function) ** 2)
    if stability_number > 1.0 + 1e-9:
        raise ValueError(f"Gradient step too large. Use a smaller step_size")
    degraded_spectrum = np.fft.fft2(degraded_image)         # Convert damaged image f = Au + n to F
    estimate = degraded_image.copy()
    for iteration_number in range(1, number_of_iterations + 1):
        # A^T (Au - f), computed in the Fourier domain -A * (A*U - F)
        residual_spectrum = np.conj(transfer_function) * (transfer_function * np.fft.fft2(estimate) - degraded_spectrum)
        # data_gradient = alpha * A^T (Au - f) with residual spectrum = A^T(Au-f)
        data_gradient = data_weight * np.real(np.fft.ifft2(residual_spectrum))
        # u = G(data_gradient)
        estimate = denoiser(estimate - step_size*data_gradient)
        if iteration_callback is not None:
            iteration_callback(iteration_number, estimate)

    return estimate

def plug_and_play_admm(
        degraded_image, 
        blur_operator, 
        denoiser, 
        data_weight, 
        number_of_iterations, 
        penalty=1.0, 
        iteration_callback=None):
    """
    ADMM with a denoiser (Equation 7 to 9 of the paper)
    The penalty is called gamma

    Variables: 
        - u: image
        - v: copy of the image that the denoiser works on
        - y: the "dual" variable that collects the disagreement between u and v

    One iteration, in the order used here:
        (8) v = G(u + y / gamma)                                denoising step (replaces prox of R)
        (7) u = prox_{(1/gamma) H_f(A)} (v - y/gamma)           data step
        (9) y = y + gamma * (u - v)                             update the disagreement

        Data step (7) with H_f(Au) = alpha/2 ||Au - f||^2 the proximal operator is the solution of
            (I + (alpha/gamma) A^T A) u = b + (alpha/gamma) A^T f, b = v - y /gamma
        A and A^T are diagonal in the Fourier domain, so this is a division per frequency:
            U = (B + (alpha/gamma) conj(H) F) / (1 + (alpha/gamma) |H|^2)
        
        Proposition 3.2 of the paper: gamma only rescales the data weight, so the paper fixes

        gamma = 1 and tunes only alpha. The same is done here
        Start values: u = degraded image, y = 0. 
        Return u after the last iteration
    """

    # transfer function Au with blur operator A  -> H = F(A)
    transfer_function = blur_operator.transfer_function (degraded_image.shape)
    # convert dammaged image f = Au + n -> F (Fourier spectrum) (F = F(f))
    degraded_spectrum = np.fft.fft2(degraded_image)
    # scale weight = alpha (data_weight)/penalty. If penalty = 1, then alpha / penalty = alpha
    scaled_weight = data_weight / penalty
    
    numerator_data_part = scaled_weight * np.conj(transfer_function) * degraded_spectrum
    denominator = 1.0 + scaled_weight * np.abs(transfer_function)**2

    estimate_u = degraded_image.copy()
    dual_y = np.zeros_like(degraded_image)
    for iteration_number in range(1, number_of_iterations + 1):
        denoised_v = denoiser(estimate_u + dual_y/penalty)                  # (8)
        shifted_v = denoised_v - dual_y/penalty
        estimate_u = np.real(np.fft.ifft2(
            (np.fft.fft2(shifted_v) + numerator_data_part) / denominator))  # (7)
        dual_y = dual_y + penalty * (estimate_u - denoised_v)               # (9)
        if iteration_callback is not None:
            iteration_callback(iteration_number, estimate_u)

    return estimate_u
        


