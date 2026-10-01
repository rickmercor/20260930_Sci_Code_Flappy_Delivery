"""
Step 09 - Onset conductance of the satellite feature from the measured satellite weight (orchestrator).

Orchestrator: from the measured satellite weight to the conductance at the onset of the satellite feature.

A scanning tunnelling measurement on a heterobilayer condensate gives the integrated current of the satellite
feature, I_sat = I(V -> -infinity), in units of G_0 Ry*/e. The sum rule of the previous step converts it into the
local exciton density, the inversion of step 6 gives the bare gap that sustains that density, the self-consistent
solution of step 4 then fixes the condensate, and the conductance of step 7 evaluated as the bias approaches the
lower band edge from below gives the onset value of the satellite feature. The onset is finite: it is the ratio
of the conduction-band weight of the lower band at k = 0 to the magnitude of the k^2 coefficient of that band there, so it depends on the
pairing amplitude and on the Fock renormalisation of the heavier band at the same time. The pipeline must also
verify its own consistency: the bound state of step 2 must satisfy its eigen-equation under the exchange integral
of step 1, the compressibility of step 3 must give a first-order gap estimate within a few percent of the
self-consistent one, the density of step 5 must reproduce the target, and the full satellite current of step 8 at
the found gap must reproduce I_sat; any of these failing beyond its stated tolerance raises ValueError.

Inputs: I_sat < 0 (G_0 Ry*/e), d >= 0 (a_B*), r > 0. Output: the onset conductance in units of G_0 as a float,
converged to 1e-7 relative accuracy. Raises ValueError if I_sat is not a finite negative number, if d or r are
invalid, if the bands are not monotonic, or if an internal consistency check fails.

Returns
-------
float, the onset conductance of the satellite feature in units of G_0
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np
from scipy.interpolate import CubicSpline


def satellite_onset_conductance(I_sat: float, d: float, r: float) -> float:
    '''Conductance at the onset of the satellite feature, from the measured integrated satellite current.

    Parameters
    ----------
    I_sat : float
        Integrated satellite current I(V -> -infinity) in units of G_0 Ry*/e, finite and negative.
    d : float
        Interlayer distance in units of a_B*, >= 0.
    r : float
        Mass ratio m_v/m_c, > 0.

    Returns
    -------
    result : float
        Limit of the conductance into the conduction-band layer as the bias approaches the lower band edge from
        below, in units of G_0, converged to 1e-7 relative accuracy.

    Raises
    ------
    ValueError
        If I_sat is not a finite negative number, d is negative or not finite, r is not positive and finite, the
        quasiparticle bands are not monotonic, or an internal consistency check of the pipeline fails.
    '''
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
from scipy.interpolate import CubicSpline


def _oracle_satellite_onset_conductance(I_sat: float, d: float, r: float) -> float:
    """Reference implementation chaining the oracles of steps 1 to 8."""
    I_sat = _check_scalar(I_sat, "I_sat")
    if I_sat >= 0.0:
        raise ValueError("I_sat must be negative")
    d = _check_scalar(d, "d", nonneg=True)
    r = _check_scalar(r, "r", positive=True)
    n_ex = -I_sat / (4 * np.pi)
    # step 2 and step 1: bound state and its eigen-equation residual
    k_dense = np.concatenate([[0.0], np.logspace(-3, 2.5, 160)])
    ex = _oracle_exciton_state(d, k_dense)
    Eb, phi_d = ex[0], ex[1:]
    spline = CubicSpline(k_dense, phi_d, bc_type=((1, 0.0), (1, 0.0)))
    phi_call = lambda q: np.where(q <= k_dense[-1], spline(np.minimum(q, k_dense[-1])), phi_d[-1] * (k_dense[-1] / np.maximum(q, 1e-300)) ** 3)
    k_chk = np.array([0.5, 2.0])
    resid = _oracle_exchange_integral(phi_call, k_chk, d) / ((k_chk ** 2 + Eb) * spline(k_chk)) - 1
    if np.max(np.abs(resid)) > 1e-4:
        raise ValueError("bound state does not satisfy its eigen-equation")
    # step 3 and step 6: first-order gap estimate and the self-consistent inversion
    dmu = _oracle_inverse_compressibility(d)
    EG_first = Eb - dmu * n_ex
    EG = _oracle_gap_for_density(n_ex, d)
    if not abs(EG - EG_first) <= 0.05 * dmu * n_ex + 1e-12:
        raise ValueError("self-consistent gap is inconsistent with the first-order estimate")
    # step 5: density check
    if abs(_oracle_exciton_density(EG, d) - n_ex) > 1e-8 * n_ex:
        raise ValueError("density does not reproduce the target")
    # step 8: the full satellite current must reproduce the measurement
    if abs(_oracle_satellite_current(EG, d, r, -np.inf) - I_sat) > 1e-8 * abs(I_sat):
        raise ValueError("satellite sum rule is violated")
    # step 4 and step 7: onset conductance as the limit from inside the lower band
    st = _oracle_condensate_state(EG, d, r, np.array([0.0]))
    E0m = st[3, 0]
    onset = _oracle_averaged_conductance(EG, d, r, np.array([E0m]))[1, 0]
    return float(onset)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Normal: the measurement of the task ---
        {
            "setup": "import numpy as np\n"
                     "from scipy.special import ellipk, ellipe\n"
                     "from scipy.linalg import eigh\n"
                     "from scipy.optimize import brentq\n"
                     "from numpy.polynomial.legendre import leggauss\n"
                     "from scipy.interpolate import CubicSpline\n",
            "call": "satellite_onset_conductance(-0.04*np.pi, 0.25, 2.0)",
            "gold_call": "_oracle_satellite_onset_conductance(-0.04*np.pi, 0.25, 2.0)",
            "tol": 1e-6,
        },
        # --- Normal: half the exciton density, same bilayer ---
        {
            "setup": "import numpy as np\n"
                     "from scipy.special import ellipk, ellipe\n"
                     "from scipy.linalg import eigh\n"
                     "from scipy.optimize import brentq\n"
                     "from numpy.polynomial.legendre import leggauss\n"
                     "from scipy.interpolate import CubicSpline\n",
            "call": "satellite_onset_conductance(-0.02*np.pi, 0.25, 2.0)",
            "gold_call": "_oracle_satellite_onset_conductance(-0.02*np.pi, 0.25, 2.0)",
            "tol": 1e-6,
        },
        # --- Boundary: equal masses at half a Bohr radius ---
        {
            "setup": "import numpy as np\n"
                     "from scipy.special import ellipk, ellipe\n"
                     "from scipy.linalg import eigh\n"
                     "from scipy.optimize import brentq\n"
                     "from numpy.polynomial.legendre import leggauss\n"
                     "from scipy.interpolate import CubicSpline\n",
            "call": "satellite_onset_conductance(-0.04*np.pi, 0.5, 1.0)",
            "gold_call": "_oracle_satellite_onset_conductance(-0.04*np.pi, 0.5, 1.0)",
            "tol": 1e-6,
        },
        # --- Edge: a monolayer with a light valence band and a small condensate ---
        {
            "setup": "import numpy as np\n"
                     "from scipy.special import ellipk, ellipe\n"
                     "from scipy.linalg import eigh\n"
                     "from scipy.optimize import brentq\n"
                     "from numpy.polynomial.legendre import leggauss\n"
                     "from scipy.interpolate import CubicSpline\n",
            "call": "satellite_onset_conductance(-0.02*np.pi, 0.0, 0.5)",
            "gold_call": "_oracle_satellite_onset_conductance(-0.02*np.pi, 0.0, 0.5)",
            "tol": 1e-6,
        },
        # --- Invalid: a positive integrated current must raise ValueError (0 returned, 1 raised) ---
        {
            "setup": "import numpy as np\n"
                     "from scipy.special import ellipk, ellipe\n"
                     "from scipy.linalg import eigh\n"
                     "from scipy.optimize import brentq\n"
                     "from numpy.polynomial.legendre import leggauss\n"
                     "from scipy.interpolate import CubicSpline\n"
                     "def _probe(fn):\n"
                     "    try:\n"
                     "        fn(0.1, 0.25, 2.0)\n"
                     "    except ValueError:\n"
                     "        return 1\n"
                     "    return 0\n",
            "call": "_probe(satellite_onset_conductance)",
            "gold_call": "_probe(_oracle_satellite_onset_conductance)",
        },
    ]
