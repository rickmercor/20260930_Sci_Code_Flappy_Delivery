"""
Evaluate the half-line Laplace transform of Gaussian spectral basis functions for supplied nonnegative Laplace parameters.

The additional total-current spectral contribution is represented using Gaussian basis functions. Mapping this spectral contribution to imaginary time requires integrating each basis function against an exponential kernel. Because the physical spectral density is defined for nonnegative frequency, the required transform is evaluated over the half-line rather than over the full real axis.

For Gaussian center $\mu_j$, width $s_j$, and Laplace parameter $c_i\geq0$, define

$$

I_{ij}=\int_0^\infty \exp\left[-\frac{(\omega-\mu_j)^2}{2s_j^2}\right]e^{-c_i\omega}\,d\omega.

$$

The returned matrix contains $I_{ij}$ for every supplied $c_i$ and Gaussian basis function $j$.

Returns
-------
np.ndarray of shape (n_c, n_basis) containing the Gaussian half-line Laplace transforms
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def compute_gaussian_laplace(
    c: "np.ndarray",
    centers: "np.ndarray",
    widths: "np.ndarray",
) -> "np.ndarray":
    """Evaluate half-line Laplace transforms of Gaussian basis functions.

    Parameters
    ----------
    c : np.ndarray
        One-dimensional finite array of nonnegative Laplace parameters with
        shape (n_c,).
    centers : np.ndarray
        One-dimensional finite array of nonnegative Gaussian centers with
        shape (n_basis,).
    widths : np.ndarray
        One-dimensional finite array of strictly positive Gaussian widths
        with shape (n_basis,).

    Returns
    -------
    result : np.ndarray
        Transform matrix with shape (n_c, n_basis), where entry (i, j)
        is the half-line Laplace transform for c[i], centers[j], and
        widths[j].

    Raises
    ------
    ValueError
        If an input is not one-dimensional or nonempty, if centers and
        widths have different lengths, if any input is nonfinite, if any
        c or center is negative, or if any width is not strictly positive.
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
from scipy.special import erfc, erfcx


def _oracle_compute_gaussian_laplace(
    c: "np.ndarray",
    centers: "np.ndarray",
    widths: "np.ndarray",
) -> "np.ndarray":
    """Reference implementation."""
    c = np.asarray(c, dtype=float)
    centers = np.asarray(centers, dtype=float)
    widths = np.asarray(widths, dtype=float)

    if c.ndim != 1 or c.size == 0:
        raise ValueError("c must be a nonempty one-dimensional array")
    if centers.ndim != 1 or centers.size == 0:
        raise ValueError("centers must be a nonempty one-dimensional array")
    if widths.ndim != 1 or widths.size == 0:
        raise ValueError("widths must be a nonempty one-dimensional array")
    if centers.shape != widths.shape:
        raise ValueError("centers and widths must have the same shape")
    if not np.all(np.isfinite(c)):
        raise ValueError("c must contain only finite values")
    if not np.all(np.isfinite(centers)):
        raise ValueError("centers must contain only finite values")
    if not np.all(np.isfinite(widths)):
        raise ValueError("widths must contain only finite values")
    if np.any(c < 0.0):
        raise ValueError("c must be nonnegative")
    if np.any(centers < 0.0):
        raise ValueError("centers must be nonnegative")
    if np.any(widths <= 0.0):
        raise ValueError("widths must be strictly positive")

    c_col = c[:, None]
    mu_row = centers[None, :]
    s_row = widths[None, :]

    argument = (
        c_col * s_row**2 - mu_row
    ) / (np.sqrt(2.0) * s_row)

    mu_grid = np.broadcast_to(mu_row, argument.shape)
    s_grid = np.broadcast_to(s_row, argument.shape)
    c_grid = np.broadcast_to(c_col, argument.shape)

    scaled = np.empty_like(argument)
    positive = argument >= 0.0

    scaled[positive] = (
        np.exp(
            -(mu_grid[positive] ** 2)
            / (2.0 * s_grid[positive] ** 2)
        )
        * erfcx(argument[positive])
    )

    negative = ~positive

    scaled[negative] = (
        np.exp(
            -c_grid[negative] * mu_grid[negative]
            + 0.5 * (c_grid[negative] * s_grid[negative]) ** 2
        )
        * erfc(argument[negative])
    )

    return s_grid * np.sqrt(np.pi / 2.0) * scaled

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return comparisons with independent candidate/reference inputs."""
    return [
        {
            "setup": """import numpy as np
c = np.array([0.075, 0.735], dtype=float)
centers = np.array([0.35, 1.40, 6.00], dtype=float)
widths = np.array([0.18, 0.32, 0.95], dtype=float)
""",
            "call": 'compute_gaussian_laplace(c.copy(), centers.copy(), widths.copy())',
            "gold_call": '_oracle_compute_gaussian_laplace(c.copy(), centers.copy(), widths.copy())',
        },
        {
            "setup": """import numpy as np
c = np.array([0.0], dtype=float)
centers = np.array([0.0, 0.80], dtype=float)
widths = np.array([0.20, 0.25], dtype=float)
""",
            "call": 'compute_gaussian_laplace(c.copy(), centers.copy(), widths.copy())',
            "gold_call": '_oracle_compute_gaussian_laplace(c.copy(), centers.copy(), widths.copy())',
        },
        {
            "setup": """import numpy as np
c = np.array([0.10, 0.60, 1.40], dtype=float)
centers = np.array([0.20], dtype=float)
widths = np.array([0.05], dtype=float)
""",
            "call": 'compute_gaussian_laplace(c.copy(), centers.copy(), widths.copy())',
            "gold_call": '_oracle_compute_gaussian_laplace(c.copy(), centers.copy(), widths.copy())',
        },
        {
            "setup": """import numpy as np
c = np.array([0.2], dtype=float)
centers = np.array([0.4, 0.8], dtype=float)
widths = np.array([0.1, 0.0], dtype=float)

def run_model():
    try:
        compute_gaussian_laplace(c.copy(), centers.copy(), widths.copy())
    except ValueError:
        return 1.0
    return 0.0

def run_gold():
    try:
        _oracle_compute_gaussian_laplace(c.copy(), centers.copy(), widths.copy())
    except ValueError:
        return 1.0
    return 0.0
""",
            "call": 'run_model()',
            "gold_call": 'run_gold()',
        },
    ]
