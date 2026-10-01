"""
Zero-pad the density cube along its vertical (last) axis to emulate the aperiodic vertical convolution.

The discrete Fourier transform assumes the signal is one period of a periodic function, which is wrong in the vertical direction of a stratified disc with vacuum above and below. The standard remedy is zero-padding: the density is embedded in an enlarged vertical domain filled with zeros so that the circular (periodic) convolution performed by FFTs reproduces the aperiodic convolution on the physical cells. Because the truncated Green's function has support |z| <= L = alpha * Lz and is oscillatory in Fourier space, the vertical extent must be enlarged enough that (i) periodic images lie outside the kernel support and (ii) the analytic kernel transform is sampled finely enough; following the reference implementation the vertical size is quadrupled (pad_factor = 4). The two in-plane directions are genuinely (shear-)periodic and are not padded.

Returns
-------
np.ndarray of shape (Nx, Ny, pad_factor * Nz), float: the input density in the first Nz vertical slices and zeros elsewhere.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def zero_pad_density(rho: np.ndarray, pad_factor: int) -> np.ndarray:
    """Zero-pad a 3D density cube along its vertical (last) axis.

    Parameters
    ----------
    rho : np.ndarray
        Density of shape (Nx, Ny, Nz) with only finite values.
    pad_factor : int
        Integer enlargement factor of the vertical dimension (pad_factor
        >= 2). The padded cube has Nz_pad = pad_factor * Nz vertical cells.

    Returns
    -------
    rho_padded : np.ndarray
        Float array of shape (Nx, Ny, pad_factor * Nz) whose first Nz
        vertical slices equal rho and whose remaining slices are zero.

    Notes
    -----
    The evaluation sandbox may execute this function in isolation: include
    every import your implementation needs (for example
    ``import numpy as np``) inside the function body.
    """
    return np.zeros(np.asarray(rho).shape[:2] + (pad_factor * np.asarray(rho).shape[2],), dtype=float)  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_zero_pad_density(rho: np.ndarray, pad_factor: int) -> np.ndarray:
    # Local imports keep the oracle self-contained when the harness
    # executes it in isolation.
    import numpy as np

    rho = np.asarray(rho, dtype=float)
    if rho.ndim != 3 or min(rho.shape) < 1:
        raise ValueError("rho must be a 3D array of shape (Nx, Ny, Nz)")
    if not np.all(np.isfinite(rho)):
        raise ValueError("rho must contain only finite values")
    if not (isinstance(pad_factor, (int, np.integer)) and not isinstance(pad_factor, bool)
            and pad_factor >= 2):
        raise ValueError("pad_factor must be an integer >= 2")

    Nx, Ny, Nz = rho.shape
    rho_padded = np.zeros((Nx, Ny, int(pad_factor) * Nz), dtype=float)
    rho_padded[:, :, :Nz] = rho
    return rho_padded

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Valid: generic cube with quadrupling (normal scenario) ---
        {
            "setup": """import numpy as np
rng = np.random.default_rng(3)
rho = rng.normal(size=(4, 5, 6))
pad_factor = 4
""",
            "call": "zero_pad_density(rho, pad_factor)",
            "gold_call": "_oracle_zero_pad_density(rho, pad_factor)",
        },
        # --- Boundary: minimum allowed pad_factor = 2 ---
        {
            "setup": """import numpy as np
rho = np.arange(24, dtype=float).reshape(2, 3, 4)
pad_factor = 2
""",
            "call": "zero_pad_density(rho, pad_factor)",
            "gold_call": "_oracle_zero_pad_density(rho, pad_factor)",
        },
        # --- Edge: single vertical cell (Nz = 1) ---
        {
            "setup": """import numpy as np
rho = np.ones((3, 3, 1))
pad_factor = 4
""",
            "call": "zero_pad_density(rho, pad_factor)",
            "gold_call": "_oracle_zero_pad_density(rho, pad_factor)",
        },
        # --- Valid: benchmark configuration, 32^3 cube quadrupled vertically ---
        {
            "setup": """import numpy as np
N = 32
d = 6.0 / N
c = -3.0 + (np.arange(N) + 0.5) * d
X, Y, Z = np.meshgrid(c, c, c, indexing="ij")
k = 2.0 * np.pi / 6.0
rho = np.cos(k * X) * np.cos(k * Y) * np.exp(-0.5 * Z ** 2)
""",
            "call": "zero_pad_density(rho, 4)",
            "gold_call": "_oracle_zero_pad_density(rho, 4)",
        },
    ]
