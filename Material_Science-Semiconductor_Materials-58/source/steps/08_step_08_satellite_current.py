"""
Step 08 - Integrated current of the satellite feature.

Integrated current of the satellite feature of a bilayer condensate.

Tunnelling out of the conduction-band layer at negative bias is possible only because condensed excitons put
conduction-band weight v_k^2 into the lower quasiparticle band. The current collected between the lower band edge
E_{0,-} and a bias eV = V below it, I(V) = int_V^{E_{0,-}} dV' dI/dV', therefore measures the condensate directly.
Because each momentum shell contributes once, the integral can be written as an integral over the momentum k up
to the value k(V) where E_{k,-} = V, and its V -> -infinity limit is fixed by the exciton density through a sum
rule; this step must reproduce that sum rule in the limit. Sign convention: the current
at negative bias is negative, so I(V) <= 0 with I(V = -infinity) = -4 pi n_ex in the units below.

Inputs: E_G (Ry*), d >= 0, r > 0 as before; V, a finite bias in Ry* or -numpy.inf. Output: I(V) in units of
G_0 Ry*/e as a float (0.0 for V >= E_{0,-}), converged to 1e-8 relative accuracy. Raises ValueError for invalid
inputs (V must be a real number or -inf; a finite value above the lower band edge gives 0.0), a non-converging
self-consistent iteration, or a non-monotonic lower band.

Returns
-------
float, the satellite current I(V) in units of G_0 Ry*/e (non-positive; 0.0 for V >= E_{0,-})
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np
from numpy.polynomial.legendre import leggauss


def satellite_current(EG: float, d: float, r: float, V: float) -> float:
    '''Current collected from the satellite feature between the lower band edge and the bias V.

    Parameters
    ----------
    EG : float
        Bare band gap E_G in Ry*, finite.
    d : float
        Interlayer distance in units of a_B*, >= 0.
    r : float
        Mass ratio m_v/m_c, > 0.
    V : float
        Bias energy eV in Ry* (measured from the middle of the bare gap), finite or -numpy.inf.

    Returns
    -------
    result : float
        I(V) = int_V^{E_{0,-}} (dI/dV') dV' of the conductance into the conduction-band layer, in units of
        G_0 Ry*/e, non-positive, 0.0 for V >= E_{0,-}, converged to 1e-8 relative accuracy.

    Raises
    ------
    ValueError
        If E_G is not finite, d is negative or not finite, r is not positive and finite, V is not a real number
        or -inf, the self-consistent iteration does not converge, or the lower band is not monotonic.
    '''
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
from numpy.polynomial.legendre import leggauss


def _oracle_satellite_current(EG: float, d: float, r: float, V: float) -> float:
    """Reference implementation: I(V) = -2 int_0^{k(V)} k v_k^2 dk (Eqs. 49 and 50 of the source)."""
    EG = _check_scalar(EG, "EG")
    d = _check_scalar(d, "d", nonneg=True)
    r = _check_scalar(r, "r", positive=True)
    if isinstance(V, bool) or not isinstance(V, (int, float, np.integer, np.floating)) or np.isnan(V) or V == np.inf:
        raise ValueError("V must be a real number or -inf")
    V = float(V)
    s = _hf(EG, d, r)
    if s["trivial"]:
        return 0.0
    fits = _band_curves(s)
    if V >= fits["Em"][0]:
        return 0.0
    if V == -np.inf:
        return float(-4 * np.pi * s["n"])
    kv = _k_of_bias(s, "Em", V)
    x, w = leggauss(64)
    kk = 0.5 * kv * (x + 1)
    v2 = _state_at(s, kk)[0]
    return float(-2 * np.sum(0.5 * kv * w * kk * v2))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Normal: the heterobilayer of the task, half a Rydberg below the lower edge ---
        {
            "setup": "import numpy as np\n"
                     "from scipy.special import ellipk, ellipe\n"
                     "from scipy.linalg import eigh\n"
                     "from scipy.optimize import brentq\n"
                     "from numpy.polynomial.legendre import leggauss\n",
            "call": "satellite_current(1.7815, 0.25, 2.0, -1.5)",
            "gold_call": "_oracle_satellite_current(1.7815, 0.25, 2.0, -1.5)",
            "tol": 1e-6,
        },
        # --- Normal: the whole satellite, which must obey the sum rule I(-inf) = -4 pi n_ex ---
        {
            "setup": "import numpy as np\n"
                     "from scipy.special import ellipk, ellipe\n"
                     "from scipy.linalg import eigh\n"
                     "from scipy.optimize import brentq\n"
                     "from numpy.polynomial.legendre import leggauss\n",
            "call": "satellite_current(1.7815, 0.25, 2.0, -np.inf)",
            "gold_call": "_oracle_satellite_current(1.7815, 0.25, 2.0, -np.inf)",
            "tol": 1e-6,
        },
        # --- Boundary: monolayer with a light valence band, bias far below the edge (most of the weight) ---
        {
            "setup": "import numpy as np\n"
                     "from scipy.special import ellipk, ellipe\n"
                     "from scipy.linalg import eigh\n"
                     "from scipy.optimize import brentq\n"
                     "from numpy.polynomial.legendre import leggauss\n",
            "call": "satellite_current(3.848585, 0.0, 0.5, -9.0)",
            "gold_call": "_oracle_satellite_current(3.848585, 0.0, 0.5, -9.0)",
            "tol": 1e-6,
        },
        # --- Edge: a bias inside the gap gives no current ---
        {
            "setup": "import numpy as np\n"
                     "from scipy.special import ellipk, ellipe\n"
                     "from scipy.linalg import eigh\n"
                     "from scipy.optimize import brentq\n"
                     "from numpy.polynomial.legendre import leggauss\n",
            "call": "satellite_current(1.7815, 0.25, 2.0, -0.4)",
            "gold_call": "_oracle_satellite_current(1.7815, 0.25, 2.0, -0.4)",
            "tol": 1e-12,
        },
        # --- Edge: without a condensate (gap above E_b) there is no satellite current at any bias ---
        {
            "setup": "import numpy as np\n"
                     "from scipy.special import ellipk, ellipe\n"
                     "from scipy.linalg import eigh\n"
                     "from scipy.optimize import brentq\n"
                     "from numpy.polynomial.legendre import leggauss\n",
            "call": "satellite_current(2.3, 0.25, 2.0, -3.0)",
            "gold_call": "_oracle_satellite_current(2.3, 0.25, 2.0, -3.0)",
            "tol": 1e-12,
        },
    ]
