"""
Construct the imaginary-time spectral kernel for Gaussian basis functions using the forward and thermally reflected Laplace contributions.

The imaginary-time energy-current correlation is related to the real-frequency spectral density through a thermal kernel containing contributions at both $\tau$ and $\beta-\tau$. For the Gaussian basis representation, each of these contributions can be evaluated with the half-line Laplace transform from the previous step.

If

$$

I_j(c)=\int_0^\infty \exp\left[-\frac{(\omega-\mu_j)^2}{2s_j^2}\right]e^{-c\omega}\,d\omega,

$$

then the kernel multiplying amplitude $a_j$ at imaginary time $\tau_i$ is

$$

K_{ij}=\frac{1}{\pi}\left[I_j(\tau_i)+I_j(\beta-\tau_i)\right].

$$

This construction preserves the imaginary-time symmetry between $\tau$ and $\beta-\tau$.

Returns
-------
np.ndarray of shape (n_tau, n_basis) containing the imaginary-time Gaussian kernel
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def build_imaginary_time_kernel(
    tau: "np.ndarray",
    beta: float,
    centers: "np.ndarray",
    widths: "np.ndarray",
) -> "np.ndarray":
    """Construct the Gaussian imaginary-time kernel.

    Parameters
    ----------
    tau : np.ndarray
        One-dimensional finite array of imaginary times with shape
        (n_tau,), each satisfying 0 <= tau[i] <= beta.
    beta : float
        Positive inverse temperature.
    centers : np.ndarray
        One-dimensional finite array of nonnegative Gaussian centers with
        shape (n_basis,).
    widths : np.ndarray
        One-dimensional finite array of strictly positive Gaussian widths
        with shape (n_basis,).

    Returns
    -------
    kernel : np.ndarray
        Kernel matrix with shape (n_tau, n_basis).

    Raises
    ------
    ValueError
        If tau is not a nonempty one-dimensional finite array, if beta is
        not finite and strictly positive, or if any tau lies outside
        [0, beta]. Validation of centers and widths follows the transform
        requirements from the preceding step.
    """
    return kernel

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_build_imaginary_time_kernel(
    tau: "np.ndarray",
    beta: float,
    centers: "np.ndarray",
    widths: "np.ndarray",
) -> "np.ndarray":
    """Reference implementation."""
    tau = np.asarray(tau, dtype=float)

    if tau.ndim != 1 or tau.size == 0:
        raise ValueError("tau must be a nonempty one-dimensional array")
    if not np.all(np.isfinite(tau)):
        raise ValueError("tau must contain only finite values")
    if not np.isfinite(beta) or float(beta) <= 0.0:
        raise ValueError("beta must be finite and strictly positive")
    if np.any(tau < 0.0) or np.any(tau > float(beta)):
        raise ValueError("tau values must lie in [0, beta]")

    reflected = float(beta) - tau
    c_all = np.concatenate((tau, reflected))

    transforms = _oracle_compute_gaussian_laplace(
        c_all,
        centers,
        widths,
    )

    n_tau = tau.size
    forward = transforms[:n_tau]
    backward = transforms[n_tau:]

    return (forward + backward) / np.pi

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return comparisons with independent candidate/reference inputs."""
    return [
        {
            "setup": """import numpy as np
tau = np.array([0.075, 0.165, 0.285, 0.420, 0.570, 0.735], dtype=float)
beta = 1.5
centers = np.array([0.35, 0.80, 1.40, 2.20, 3.20, 4.50, 6.00], dtype=float)
widths = np.array([0.18, 0.25, 0.32, 0.42, 0.55, 0.72, 0.95], dtype=float)
""",
            "call": 'build_imaginary_time_kernel(tau.copy(), beta, centers.copy(), widths.copy())',
            "gold_call": '_oracle_build_imaginary_time_kernel(tau.copy(), beta, centers.copy(), widths.copy())',
        },
        {
            "setup": """import numpy as np
tau = np.array([0.0, 1.5], dtype=float)
beta = 1.5
centers = np.array([0.35, 1.40], dtype=float)
widths = np.array([0.18, 0.32], dtype=float)
""",
            "call": 'build_imaginary_time_kernel(tau.copy(), beta, centers.copy(), widths.copy())',
            "gold_call": '_oracle_build_imaginary_time_kernel(tau.copy(), beta, centers.copy(), widths.copy())',
        },
        {
            "setup": """import numpy as np
tau = np.array([0.75], dtype=float)
beta = 1.5
centers = np.array([0.0, 0.80, 3.20], dtype=float)
widths = np.array([0.20, 0.25, 0.55], dtype=float)
""",
            "call": 'build_imaginary_time_kernel(tau.copy(), beta, centers.copy(), widths.copy())',
            "gold_call": '_oracle_build_imaginary_time_kernel(tau.copy(), beta, centers.copy(), widths.copy())',
        },
        {
            "setup": """import numpy as np
tau = np.array([0.2, 1.6], dtype=float)
beta = 1.5
centers = np.array([0.35, 0.80], dtype=float)
widths = np.array([0.18, 0.25], dtype=float)

def run_model():
    try:
        build_imaginary_time_kernel(tau.copy(), beta, centers.copy(), widths.copy())
    except ValueError:
        return 1.0
    return 0.0

def run_gold():
    try:
        _oracle_build_imaginary_time_kernel(tau.copy(), beta, centers.copy(), widths.copy())
    except ValueError:
        return 1.0
    return 0.0
""",
            "call": 'run_model()',
            "gold_call": 'run_gold()',
        },
    ]
