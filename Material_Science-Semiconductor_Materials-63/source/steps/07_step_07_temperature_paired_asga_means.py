"""
Reduce the ordered As_Ga depth series to per-temperature means.

Finite-temperature defect levels are often summarized by averaging mid-gap depths within each temperature class before combining those class means into a single reported scalar.

Returns
-------
np.ndarray shape (n_temperatures,), per-temperature mean depths in eV
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def compute_temperature_paired_asga_means(
    ordered_depths: np.ndarray, n_per_temperature: int
) -> np.ndarray:
    """Return per-temperature means from an ordered As_Ga depth series.

    Parameters
    ----------
    ordered_depths : np.ndarray
        1D ordered As_Ga depths (T ascending blocks).
    n_per_temperature : int
        Number of As_Ga depths per temperature block.

    Returns
    -------
    paired_means : np.ndarray
        Shape (n_temperatures,), block-wise arithmetic means.
    """
    return np.asarray([], dtype=float)

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
def _oracle_compute_temperature_paired_asga_means(
    ordered_depths: np.ndarray, n_per_temperature: int
) -> np.ndarray:
    
    d = np.asarray(ordered_depths, dtype=float).reshape(-1)
    n = int(n_per_temperature)
    if n < 1:
        raise ValueError("n_per_temperature must be >= 1")
    if d.size < 1:
        raise ValueError("ordered_depths must be non-empty")
    if d.size % n != 0:
        raise ValueError("ordered_depths length must be divisible by n_per_temperature")
    if np.any(~np.isfinite(d)) or np.any(d < 0.0):
        raise ValueError("depths must be finite and non-negative")
    blocks = d.reshape(-1, n)
    return blocks.mean(axis=1).astype(float)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": "import numpy as np\nordered_depths = np.array([0.4099, 0.4298, 0.4562, 0.4317])\nn_per_temperature = 2\n",
            "call": "compute_temperature_paired_asga_means(ordered_depths, n_per_temperature)",
            "gold_call": "_oracle_compute_temperature_paired_asga_means(ordered_depths, n_per_temperature)",
        },
        {
            "setup": "import numpy as np\nordered_depths = np.array([0.5, 0.5])\nn_per_temperature = 2\n",
            "call": "compute_temperature_paired_asga_means(ordered_depths, n_per_temperature)",
            "gold_call": "_oracle_compute_temperature_paired_asga_means(ordered_depths, n_per_temperature)",
        },
        {
            "setup": "import numpy as np\nordered_depths = np.array([0.4, 0.2, 0.6])\nn_per_temperature = 1\n",
            "call": "compute_temperature_paired_asga_means(ordered_depths, n_per_temperature)",
            "gold_call": "_oracle_compute_temperature_paired_asga_means(ordered_depths, n_per_temperature)",
        },
    ]
