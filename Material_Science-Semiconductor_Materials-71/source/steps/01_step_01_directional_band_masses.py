"""
Recover directional carrier masses from near-edge nonparabolic band samples.

Near-edge Kane/hyperbolic samples convert measured (k, E) pairs into directional m*/m_e values that feed continuum ionization, polaron, and mobility descriptors.

Returns
-------
ndarray, shape (n,), directional effective masses in units of m_e
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def directional_band_masses(
    k_values: "np.ndarray",
    energy_values: "np.ndarray",
    alpha: float,
    hbar2_over_me: float = 7.6199642,
) -> "np.ndarray":
    """Return directional m*/m_e for each near-edge sample.

    Raises
    ------
    ValueError
        If inputs are mismatched or empty, alpha or hbar2_over_me is non-positive,
        any k or energy is non-positive, or the inversion denominator is non-positive.
    """
    return []

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_directional_band_masses(
    k_values: "np.ndarray",
    energy_values: "np.ndarray",
    alpha: float,
    hbar2_over_me: float = 7.6199642,
) -> "np.ndarray":
    k = np.asarray(k_values, dtype=float)
    e = np.asarray(energy_values, dtype=float)
    if k.ndim != 1 or e.ndim != 1 or k.shape != e.shape:
        raise ValueError("k_values and energy_values must be 1D arrays of equal length")
    if k.size == 0:
        raise ValueError("at least one band sample is required")
    if alpha <= 0:
        raise ValueError("alpha must be positive")
    if hbar2_over_me <= 0:
        raise ValueError("hbar2_over_me must be positive")
    if np.any(k <= 0):
        raise ValueError("all k_values must be positive")
    if np.any(e <= 0):
        raise ValueError("all energy_values must be positive")
    denom = (2.0 * alpha * e + 1.0) ** 2 - 1.0
    if np.any(denom <= 0):
        raise ValueError("inversion denominator must be positive")
    return (2.0 * alpha * hbar2_over_me * k**2) / denom

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": (
                "import numpy as np\n"
                "k_values = np.array([0.048, 0.052, 0.045])\n"
                "energy_values = np.array(["
                "0.0393135912340984, 0.024891847042767198, 0.02465676303235815])\n"
                "alpha = 0.38\n"
                "hbar2_over_me = 7.6199642\n"
            ),
            "call": "directional_band_masses(k_values, energy_values, alpha, hbar2_over_me)",
            "gold_call": "_oracle_directional_band_masses(k_values, energy_values, alpha, hbar2_over_me)",
        },
        {
            "setup": (
                "import numpy as np\n"
                "k_values = np.array([0.05])\n"
                "energy_values = np.array([0.01])\n"
                "alpha = 0.2\n"
                "hbar2_over_me = 7.6199642\n"
            ),
            "call": "directional_band_masses(k_values, energy_values, alpha, hbar2_over_me)",
            "gold_call": "_oracle_directional_band_masses(k_values, energy_values, alpha, hbar2_over_me)",
        },
        {
            "setup": (
                "import numpy as np\n"
                "k_values = np.array([1e-3, 2e-3, 3e-3])\n"
                "energy_values = np.array([1e-5, 2.5e-5, 4e-5])\n"
                "alpha = 0.5\n"
                "hbar2_over_me = 7.6199642\n"
            ),
            "call": "directional_band_masses(k_values, energy_values, alpha, hbar2_over_me)",
            "gold_call": "_oracle_directional_band_masses(k_values, energy_values, alpha, hbar2_over_me)",
        },
    ]
