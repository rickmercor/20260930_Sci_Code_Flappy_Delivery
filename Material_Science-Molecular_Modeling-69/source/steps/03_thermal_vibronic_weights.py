"""
Compute thermally weighted displaced-oscillator overlaps.

The finite-temperature molecular vibronic model includes initially

occupied vibrational levels. Initial thermal normalization and the

final-state overlap sum have different roles.

Returns
-------
np.ndarray (nmax+1,mmax+1): thermal overlap weights
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def thermal_vibronic_weights(
    high: float,
    quantum: float,
    temperature: float,
    nmax: int,
    mmax: int,
) -> "np.ndarray":
    """Compute finite-temperature displaced-oscillator overlap weights.
    
    Parameters
    ----------
    high : float
        Nonnegative high-frequency reorganization energy in eV.
    quantum : float
        Positive effective oscillator energy in eV.
    temperature : float
        Positive temperature in K; k_B = 8.617333262145e-5 eV/K.
    nmax, mmax : int
        Nonnegative inclusive initial/final number-state cutoffs, not bool.
    
    Returns
    -------
    ndarray, shape (nmax+1, mmax+1)
        W[n,m] = p_n * |<m|D(sqrt(S))|n>|^2, with S=high/quantum.
        D is the real harmonic-oscillator displacement operator.
        Normalize p_n proportional to exp[-n*quantum/(k_B*T)] over
        n=0..nmax only. Do not renormalize the truncated final overlaps.
        At S=0 use Kronecker overlaps. Equivalent exact oscillator or
        Laguerre formulations are accepted.
    
    Raises
    ------
    ValueError
        For nonfinite scales, invalid signs or invalid cutoffs.
    """
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_thermal_vibronic_weights(
    high: float,
    quantum: float,
    temperature: float,
    nmax: int,
    mmax: int,
) -> "np.ndarray":
    import numpy as np
    from scipy.special import eval_genlaguerre, gammaln

    high = _positive(high, "high", zero=True)
    quantum, temperature = [
        _positive(v, "thermal scale") for v in (quantum, temperature)
    ]
    if any(
        isinstance(v, (bool, np.bool_))
        or not isinstance(v, (int, np.integer))
        or v < 0
        for v in (nmax, mmax)
    ):
        raise ValueError("cutoffs must be nonnegative integers")
    n = np.arange(nmax + 1)[:, None]
    m = np.arange(mmax + 1)[None, :]
    small, large = np.minimum(n, m), np.maximum(n, m)
    s = high / quantum
    if s == 0:
        overlap = (n == m).astype(float)
    else:
        overlap = np.exp(
            -s
            + (large - small) * np.log(s)
            + gammaln(small + 1)
            - gammaln(large + 1)
        )
        overlap *= eval_genlaguerre(small, large - small, s) ** 2
    thermal = np.exp(-n * quantum / (8.617333262145e-5 * temperature))
    thermal /= thermal.sum()
    return thermal * overlap

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    ("Return three scientific cases and one " "domain check.")
    return [
        {
            "setup": """
import numpy as np
from scipy.special import eval_genlaguerre, gammaln
""",
            "call": """
thermal_vibronic_weights(0.18, 0.060, 310.0, 12, 40)
""",
            "gold_call": """
_oracle_thermal_vibronic_weights(0.18, 0.060, 310.0, 12, 40)
""",
        },
        {
            "setup": """
import numpy as np
from scipy.special import eval_genlaguerre, gammaln
""",
            "call": """
thermal_vibronic_weights(0.0, 0.075, 500.0, 5, 3)
""",
            "gold_call": """
_oracle_thermal_vibronic_weights(0.0, 0.075, 500.0, 5, 3)
""",
        },
        {
            "setup": """
import numpy as np
from scipy.special import eval_genlaguerre, gammaln
""",
            "call": """
thermal_vibronic_weights(0.27, 0.040, 450.0, 8, 2)
""",
            "gold_call": """
_oracle_thermal_vibronic_weights(0.27, 0.040, 450.0, 8, 2)
""",
        },
        {
            "setup": """
import numpy as np
from scipy.special import eval_genlaguerre, gammaln


def check_domain(fn):
    try:
        fn(0.18, 0.060, 310.0, -1, 40)
    except ValueError:
        return 1
    return 0
""",
            "call": "check_domain(thermal_vibronic_weights)",
            "gold_call": "check_domain(_oracle_thermal_vibronic_weights)",
        },
    ]
