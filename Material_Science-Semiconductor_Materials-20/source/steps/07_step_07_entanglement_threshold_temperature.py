"""
Step 07 - Threshold temperature for a target concurrence (orchestrator).

Lattice temperature at which the photon-pair entanglement falls to a target value
(orchestrator).

Warming the lattice strengthens the phonon-induced decoherence of the biexciton cascade,
so the single-time-integrated concurrence C_bar(T) of step 06 decreases with the lattice
temperature over the range of interest. For a device specified by its dot and cavity
parameters and by its lattice, the highest operating temperature compatible with a
required entanglement is the temperature T* at which C_bar(T*) equals the target.

This step returns T* inside a given bracket [T_lo, T_hi] on which C_bar(T) - C_target
changes sign, with C_bar(T_lo) above the target and C_bar(T_hi) below it. The whole
pipeline enters every evaluation of C_bar: the spectral density (step 01), the phonon
propagator (step 02), the polaron response functions (step 03), the master equation
(step 04), the integrated pair correlations (step 05) and the concurrence (step 06).
T* must be located to within 1e-7 K.

Parameters are dot = [E_B (meV), delta (meV), g (meV), gamma (1/ps)] as in step 05 and
the lattice array of step 01.

Returns
-------
float, the lattice temperature T* in K at which C_bar(T*) equals C_target
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np
from scipy.optimize import brentq


def entanglement_threshold_temperature(C_target: float, T_lo: float, T_hi: float, dot: np.ndarray,
                                       lattice: np.ndarray) -> float:
    '''Temperature T* at which the single-time-integrated concurrence equals C_target.

    Parameters
    ----------
    C_target : float
        Target concurrence, 0 < C_target < 1.
    T_lo : float
        Lower end of the temperature bracket in K, 0 < T_lo < T_hi.
    T_hi : float
        Upper end of the temperature bracket in K.
    dot : np.ndarray
        Shape (4,): [E_B (meV), delta (meV), g (meV) > 0, gamma (1/ps) > 0].
    lattice : np.ndarray
        Shape (6,): [D_e (eV), D_h (eV), mass density (kg/m^3), c_s (m/s), a_e (nm), a_h (nm)].

    Returns
    -------
    T_star : float
        The temperature in K, T_lo < T_star < T_hi, at which C_bar(T_star) = C_target.

    Raises
    ------
    ValueError
        If C_target is not a finite number strictly between 0 and 1, if the bracket does
        not satisfy 0 < T_lo < T_hi, if C_bar(T_lo) is not above C_target or C_bar(T_hi)
        is not below it, or if dot or lattice is invalid (as in step 05).
    '''
    return T_star

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
from scipy.optimize import brentq


def _check_target(c_target, t_lo, t_hi):
    """Validate the target concurrence and the temperature bracket."""
    vals = []
    for name, v in (("C_target", c_target), ("T_lo", t_lo), ("T_hi", t_hi)):
        if isinstance(v, bool) or not isinstance(v, (int, float, np.integer, np.floating)):
            raise ValueError("%s must be a real number" % name)
        v = float(v)
        if not np.isfinite(v):
            raise ValueError("%s must be finite" % name)
        vals.append(v)
    c_target, t_lo, t_hi = vals
    if not 0.0 < c_target < 1.0:
        raise ValueError("C_target must lie strictly between 0 and 1")
    if not 0.0 < t_lo < t_hi:
        raise ValueError("the bracket must satisfy 0 < T_lo < T_hi")
    return c_target, t_lo, t_hi


def _concurrence_excess(temp, c_target, dot, lattice):
    """C_bar(temp) - c_target, evaluated with the reference pipeline."""
    return _oracle_integrated_concurrence(temp, dot, lattice) - c_target


def _oracle_entanglement_threshold_temperature(C_target: float, T_lo: float, T_hi: float, dot: np.ndarray,
                                               lattice: np.ndarray) -> float:
    c_target, t_lo, t_hi = _check_target(C_target, T_lo, T_hi)
    _check_dot(dot, need_loss=True)
    _check_lattice(lattice)

    f_lo = _concurrence_excess(t_lo, c_target, dot, lattice)
    f_hi = _concurrence_excess(t_hi, c_target, dot, lattice)
    if not (f_lo > 0.0 > f_hi):
        raise ValueError("C_target is not bracketed: need C_bar(T_lo) > C_target > C_bar(T_hi)")
    return float(brentq(_concurrence_excess, t_lo, t_hi, args=(c_target, dot, lattice),
                        xtol=1e-9, rtol=1e-13, maxiter=200))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Normal: benchmark device, target 0.95 in 4-40 K ---
        {
            "setup": "import numpy as np\n"
                     "from scipy.optimize import brentq\n"
                     "dot = np.array([1.2, 0.15, 0.2, 0.3])\n"
                     "lat = np.array([7.0, -3.5, 5370.0, 5110.0, 3.5, 3.5 / 1.15])\n",
            "call": "entanglement_threshold_temperature(0.95, 4.0, 40.0, dot.copy(), lat.copy())",
            "gold_call": "_oracle_entanglement_threshold_temperature(0.95, 4.0, 40.0, dot.copy(), lat.copy())",
            "tol": 1e-6,
        },
        # --- Boundary: the published parameter set, a target close to its low-temperature limit ---
        {
            "setup": "import numpy as np\n"
                     "from scipy.optimize import brentq\n"
                     "dot = np.array([1.5, 0.1, 0.1, 0.25])\n"
                     "lat = np.array([7.0, -3.5, 5370.0, 5110.0, 3.0, 3.0 / 1.15])\n",
            "call": "entanglement_threshold_temperature(0.985, 2.0, 30.0, dot.copy(), lat.copy())",
            "gold_call": "_oracle_entanglement_threshold_temperature(0.985, 2.0, 30.0, dot.copy(), lat.copy())",
            "tol": 1e-6,
        },
        # --- Edge: a large dot and strong coupling, wide bracket ---
        {
            "setup": "import numpy as np\n"
                     "from scipy.optimize import brentq\n"
                     "dot = np.array([1.0, 0.2, 0.3, 0.35])\n"
                     "lat = np.array([7.0, -3.5, 5370.0, 5110.0, 4.5, 4.5 / 1.15])\n",
            "call": "entanglement_threshold_temperature(0.94, 5.0, 60.0, dot.copy(), lat.copy())",
            "gold_call": "_oracle_entanglement_threshold_temperature(0.94, 5.0, 60.0, dot.copy(), lat.copy())",
            "tol": 1e-6,
        },
        # --- Invalid: target above the phonon-free limit, so it is not bracketed ---
        {
            "setup": "import numpy as np\n"
                     "from scipy.optimize import brentq\n"
                     "dot = np.array([1.2, 0.15, 0.2, 0.3])\n"
                     "lat = np.array([7.0, -3.5, 5370.0, 5110.0, 3.5, 3.5 / 1.15])\n"
                     "def run_model():\n"
                     "    try:\n"
                     "        entanglement_threshold_temperature(0.99, 4.0, 40.0, dot.copy(), lat.copy())\n"
                     "        return 0\n"
                     "    except ValueError:\n"
                     "        return 1\n"
                     "    except Exception:\n"
                     "        return 2\n"
                     "def run_gold():\n"
                     "    try:\n"
                     "        _oracle_entanglement_threshold_temperature(0.99, 4.0, 40.0, dot.copy(), lat.copy())\n"
                     "        return 0\n"
                     "    except ValueError:\n"
                     "        return 1\n"
                     "    except Exception:\n"
                     "        return 2\n",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
    ]
