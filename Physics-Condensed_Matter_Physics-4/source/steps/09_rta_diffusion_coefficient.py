"""
Compute the exciton diffusion coefficient of a given occupation of the moire mini-band states in the relaxation-time approximation.

When the occupation varies slowly in space and each state relaxes toward the local distribution with its own
scattering time, Fick's law holds with a diffusion tensor fixed by the squared group velocities, the scattering
times and the occupation. In an isotropic 2D system the scalar coefficient is half the trace of that tensor.
The scattering time of a state is the inverse of its total out-scattering rate. The coefficient is a weighted
average over the occupation, so an unnormalised occupation gives the same result. The group velocities enter
in nm/ps; report the coefficient in cm^2/s.

Returns
-------
d : float -- Diffusion coefficient in cm^2/s.
"""

import numpy as np
from scipy.linalg import expm

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def rta_diffusion_coefficient(v2: "np.ndarray", rates: "np.ndarray", occupation: "np.ndarray") -> float:
    """Return the relaxation-time-approximation diffusion coefficient.

    Parameters
    ----------
    v2 : np.ndarray
        Squared group-velocity magnitudes, nm^2/ps^2, S entries in the flattened state order.
    rates : np.ndarray
        (S, S) rates in 1/ps, rates[f, i] from state i into state f.
    occupation : np.ndarray
        Non-negative occupations, S entries, not necessarily normalised.

    Returns
    -------
    d : float
        Diffusion coefficient in cm^2/s.
    """
    return d

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_rta_diffusion_coefficient(v2: "np.ndarray", rates: "np.ndarray", occupation: "np.ndarray") -> float:
    v2 = np.asarray(v2, dtype=float).reshape(-1)
    n = np.asarray(occupation, dtype=float).reshape(-1)
    tau = 1.0 / rates.sum(axis=0)
    return float(0.5 * np.sum(v2 * tau * n) / np.sum(n) * 1e-2)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    base = """import numpy as np
rng = np.random.default_rng(6)
v2 = rng.uniform(0.0, 4.0e4, size=10)
r = rng.uniform(0.0, 0.5, size=(10, 10))
"""
    return [
        {"setup": base + "n = rng.uniform(0.0, 1.0, size=10)",
         "call": "rta_diffusion_coefficient(v2, r, n)",
         "gold_call": "_oracle_rta_diffusion_coefficient(v2, r, n)"},
        # a single occupied state
        {"setup": base + "n = np.zeros(10); n[7] = 3.0",
         "call": "rta_diffusion_coefficient(v2, r, n)",
         "gold_call": "_oracle_rta_diffusion_coefficient(v2, r, n)"},
        # thermal occupation of 2D-shaped inputs
        {"setup": base + "e = np.linspace(0.0, 9.0, 10)\nn = np.exp(-e / 0.8617)\nv2 = v2.reshape(5, 2)",
         "call": "rta_diffusion_coefficient(v2, r, n)",
         "gold_call": "_oracle_rta_diffusion_coefficient(v2, r, n)"},
    ]
