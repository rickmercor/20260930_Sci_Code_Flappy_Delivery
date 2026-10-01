"""
Evaluate the non-radiative recombination sink on every node from trap-assisted and Auger channels.

Carriers are also removed in pairs wherever holes and electrons coexist. In the transport layers the relevant channels are both non-radiative, since radiative recombination is confined to the emissive quantum-dot layer and does not act in the injection or transport layer.

The first channel is Shockley-Read-Hall recombination through defect states. Its rate has the excess of the carrier product over its equilibrium value, p n - ni^2, in the numerator and the denominator tau_n (p + ni) + tau_p (n + ni), which pairs each capture lifetime with the density of the opposite carrier plus the intrinsic density; the rate vanishes at equilibrium and changes sign under depletion. The second channel is Auger recombination, a three-body process whose released energy is taken up by a third carrier, so it adds (Cp p + Cn n)(p n - ni^2) with separate hole and electron capture probabilities. The total sink is the sum of the two.

The intrinsic density is a per-node quantity here: it differs between the two layers by twelve orders of magnitude, and a single global value would misplace the equilibrium point in one layer or the other.

Densities are in cm^-3, lifetimes in s, Auger probabilities in cm^6 s^-1 and the rate in cm^-3 s^-1.

Returns
-------
np.ndarray of shape (N,), the total recombination rate at every node in cm^-3 s^-1 as float64
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def recombination_rate(p: np.ndarray, n: np.ndarray, ni: np.ndarray, tau_p: float, tau_n: float,
                       Cp: float, Cn: float) -> np.ndarray:
    '''Total Shockley-Read-Hall plus Auger recombination rate at every node.

    Parameters
    ----------
    p, n : np.ndarray
        Arrays of shape (N,) of hole and electron densities in cm^-3.
    ni : np.ndarray
        Array of shape (N,) of intrinsic carrier densities in cm^-3, non-negative.
    tau_p, tau_n : float
        Hole and electron SRH lifetimes in s, positive.
    Cp, Cn : float
        Hole and electron Auger capture probabilities in cm^6 s^-1, non-negative.

    Returns
    -------
    U : np.ndarray
        Array of shape (N,) of recombination rates in cm^-3 s^-1 (negative for net generation).

    Raises
    ------
    ValueError
        If p, n and ni are not one-dimensional arrays of a common length, if a lifetime is not positive,
        if an Auger probability or any ni is negative, or if the SRH denominator vanishes at some node.
    '''
    return U

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_recombination_rate(p: np.ndarray, n: np.ndarray, ni: np.ndarray, tau_p: float, tau_n: float,
                               Cp: float, Cn: float) -> np.ndarray:
    p = np.array(p, dtype=float); n = np.array(n, dtype=float); ni = np.array(ni, dtype=float)
    if p.ndim != 1 or n.shape != p.shape or ni.shape != p.shape:
        raise ValueError("p, n and ni must be one-dimensional arrays of a common length")
    if not (tau_p > 0.0 and tau_n > 0.0):
        raise ValueError("lifetimes must be positive")
    if Cp < 0.0 or Cn < 0.0 or np.any(ni < 0.0):
        raise ValueError("Auger probabilities and intrinsic densities must be non-negative")
    denom = tau_n * (p + ni) + tau_p * (n + ni)
    if np.any(denom == 0.0):
        raise ValueError("the SRH denominator vanishes")
    excess = p * n - ni ** 2
    return excess / denom + (Cp * p + Cn * n) * excess

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        {
            "setup": """import numpy as np
p = np.concatenate([np.full(21, 2.2e19), np.full(80, 6.0e17)])
n = np.concatenate([np.full(21, 9.4552e-8), np.full(80, 2.5281e-29)])
ni = np.concatenate([np.full(21, 1.63e6), np.full(80, 1.59e-6)])""",
            "call": "recombination_rate(p, n, ni, 1.0e-7, 1.0e-7, 0.0, 0.0)",
            "gold_call": "_oracle_recombination_rate(p, n, ni, 1.0e-7, 1.0e-7, 0.0, 0.0)",
        },
        {
            "setup": """import numpy as np
p = np.array([4.0e10, 1.0e10, 2.0e9]); n = np.array([1.5e10, 3.0e10, 0.2e9]); ni = np.array([2.0e10, 1.0e10, 1.0e9])""",
            "call": "recombination_rate(p, n, ni, 2.0e-7, 5.0e-7, 1.0e-30, 3.0e-30)",
            "gold_call": "_oracle_recombination_rate(p, n, ni, 2.0e-7, 5.0e-7, 1.0e-30, 3.0e-30)",
        },
        {
            "setup": """import numpy as np
ni = np.array([1.63e6, 1.59e-6]); p = np.array([2.81e19, 1.0e17]); n = ni ** 2 / p""",
            "call": "recombination_rate(p, n, ni, 1.0e-7, 1.0e-7, 2.0e-31, 2.0e-31)",
            "gold_call": "_oracle_recombination_rate(p, n, ni, 1.0e-7, 1.0e-7, 2.0e-31, 2.0e-31)",
        },
        {
            "setup": """import numpy as np
p = np.array([1.0e18, 3.0e17]); n = np.array([1.0e18, 5.0e17]); ni = np.array([1.0e10, 1.0e10])""",
            "call": "recombination_rate(p, n, ni, 1.0e-7, 3.0e-7, 1.0e-30, 2.0e-30)",
            "gold_call": "_oracle_recombination_rate(p, n, ni, 1.0e-7, 3.0e-7, 1.0e-30, 2.0e-30)",
        },
        {
            "setup": """import numpy as np
def _probe(fn):
    hits = 0
    a = np.ones(2)
    for args in ((a, a, a, 0.0, 1.0, 0.0, 0.0), (a, a, a, 1.0, 1.0, -1.0, 0.0), (a, np.ones(3), a, 1.0, 1.0, 0.0, 0.0),
                 (np.zeros(2), np.zeros(2), np.zeros(2), 1.0, 1.0, 0.0, 0.0)):
        try:
            fn(*args)
        except ValueError:
            hits += 1
    return float(hits)""",
            "call": "_probe(recombination_rate)",
            "gold_call": "_probe(_oracle_recombination_rate)",
        },
    ]
