"""
Evaluate the reconstructed additional spectral density at zero frequency from the fitted Gaussian amplitudes.

The dc thermal conductivity depends on the zero-frequency limit of the energy-current spectral density. After the positive Gaussian amplitudes have been reconstructed, the additional spectral contribution at $\omega=0$ can be evaluated directly from the Gaussian basis representation.

For amplitudes $a_j$, centers $\mu_j$, and widths $s_j$,

$$

\widetilde{\Lambda}(0)=\sum_{j=1}^{n_{\mathrm{basis}}}a_j\exp\left[-\frac{\mu_j^2}{2s_j^2}\right].

$$

Gaussian components centered far from zero frequency are exponentially suppressed in this quantity, even when they carry appreciable spectral weight at higher frequencies.

Returns
-------
float, the additional reconstructed spectral density at zero frequency
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def compute_zero_frequency_spectrum(
    amplitudes: "np.ndarray",
    centers: "np.ndarray",
    widths: "np.ndarray",
) -> float:
    """Evaluate the additional spectral density at zero frequency.

    Parameters
    ----------
    amplitudes : np.ndarray
        One-dimensional finite array of nonnegative Gaussian amplitudes with
        shape (n_basis,).
    centers : np.ndarray
        One-dimensional finite array of nonnegative Gaussian centers with
        shape (n_basis,).
    widths : np.ndarray
        One-dimensional finite array of strictly positive Gaussian widths
        with shape (n_basis,).

    Returns
    -------
    lambda_zero : float
        Additional reconstructed spectral density at zero frequency.

    Raises
    ------
    ValueError
        If an input is not a nonempty one-dimensional array, if the three
        arrays do not have the same shape, if any input is nonfinite, if any
        amplitude or center is negative, or if any width is not strictly
        positive.
    """
    return lambda_zero

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_compute_zero_frequency_spectrum(
    amplitudes: "np.ndarray",
    centers: "np.ndarray",
    widths: "np.ndarray",
) -> float:
    """Reference implementation."""
    amplitudes = np.asarray(amplitudes, dtype=float)
    centers = np.asarray(centers, dtype=float)
    widths = np.asarray(widths, dtype=float)

    if amplitudes.ndim != 1 or amplitudes.size == 0:
        raise ValueError(
            "amplitudes must be a nonempty one-dimensional array"
        )
    if centers.ndim != 1 or centers.size == 0:
        raise ValueError(
            "centers must be a nonempty one-dimensional array"
        )
    if widths.ndim != 1 or widths.size == 0:
        raise ValueError(
            "widths must be a nonempty one-dimensional array"
        )
    if amplitudes.shape != centers.shape or centers.shape != widths.shape:
        raise ValueError(
            "amplitudes, centers, and widths must have the same shape"
        )
    if not np.all(np.isfinite(amplitudes)):
        raise ValueError("amplitudes must contain only finite values")
    if not np.all(np.isfinite(centers)):
        raise ValueError("centers must contain only finite values")
    if not np.all(np.isfinite(widths)):
        raise ValueError("widths must contain only finite values")
    if np.any(amplitudes < 0.0):
        raise ValueError("amplitudes must be nonnegative")
    if np.any(centers < 0.0):
        raise ValueError("centers must be nonnegative")
    if np.any(widths <= 0.0):
        raise ValueError("widths must be strictly positive")

    zero_weights = np.exp(
        -(centers**2) / (2.0 * widths**2)
    )

    return float(amplitudes @ zero_weights)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return comparisons with independent candidate/reference inputs."""
    return [
        {
            "setup": """import numpy as np
amplitudes = np.array([0.012, 0.025, 0.018, 0.030], dtype=float)
centers = np.array([0.20, 0.90, 2.10, 4.00], dtype=float)
widths = np.array([0.15, 0.28, 0.50, 0.85], dtype=float)
""",
            "call": 'compute_zero_frequency_spectrum(amplitudes.copy(), centers.copy(), widths.copy())',
            "gold_call": '_oracle_compute_zero_frequency_spectrum(amplitudes.copy(), centers.copy(), widths.copy())',
        },
        {
            "setup": """import numpy as np
amplitudes = np.array([0.20, 0.35], dtype=float)
centers = np.array([0.0, 0.0], dtype=float)
widths = np.array([0.10, 2.00], dtype=float)
""",
            "call": 'compute_zero_frequency_spectrum(amplitudes.copy(), centers.copy(), widths.copy())',
            "gold_call": '_oracle_compute_zero_frequency_spectrum(amplitudes.copy(), centers.copy(), widths.copy())',
        },
        {
            "setup": """import numpy as np
amplitudes = np.array([1.0, 2.0, 3.0], dtype=float)
centers = np.array([2.0, 5.0, 9.0], dtype=float)
widths = np.array([0.20, 0.50, 0.80], dtype=float)
""",
            "call": 'compute_zero_frequency_spectrum(amplitudes.copy(), centers.copy(), widths.copy())',
            "gold_call": '_oracle_compute_zero_frequency_spectrum(amplitudes.copy(), centers.copy(), widths.copy())',
        },
        {
            "setup": """import numpy as np
amplitudes = np.array([0.10, 0.20], dtype=float)
centers = np.array([0.30, 0.80], dtype=float)
widths = np.array([0.15, 0.00], dtype=float)

def run_model():
    try:
        compute_zero_frequency_spectrum(
            amplitudes.copy(), centers.copy(), widths.copy()
        )
    except ValueError:
        return 1.0
    return 0.0

def run_gold():
    try:
        _oracle_compute_zero_frequency_spectrum(
            amplitudes.copy(), centers.copy(), widths.copy()
        )
    except ValueError:
        return 1.0
    return 0.0
""",
            "call": 'run_model()',
            "gold_call": 'run_gold()',
        },
    ]
