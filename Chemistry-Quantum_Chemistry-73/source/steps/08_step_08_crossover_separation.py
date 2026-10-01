"""
Step 08: Crossover separation of localized density correction and SCF LSDA (orchestrator).

Separation beyond which density correction with a localized density beats self-consistent LSDA for stretched H2+ (orchestrator).

Near equilibrium the maximally localized orbital is a poor description of H2+ because it mixes in the ungerade state,
so evaluating the local functional on it gives a large positive binding-energy error. At large separation the splitting
between the two lowest states dies away exponentially while the self-consistent LSDA error grows toward a finite
negative limit, so the localized density wins. The crossover separation R_x is where the two absolute errors are equal.

The search scans separations on an evenly spaced grid from r_min to r_max inclusive and evaluates
g(R) = |error of LSDA on the localized orbital| - |error of self-consistent LSDA| from the previous step. The crossover is
located in the first scan interval [R_(k-1), R_k] with g(R_(k-1)) > 0 and g(R_k) <= 0, returning R_k if g(R_k) = 0, and
otherwise refined by Brent's method to an absolute tolerance of 10^-10 bohr.

Returns
-------
float, crossover separation R_x in bohr where the localized and self-consistent error magnitudes are equal
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def crossover_separation(softening: float, spacing: float, half_width: float, r_min: float, r_max: float, n_scan: int) -> float:
    '''Separation, refined in the first scan bracket, where the localized density-corrected error equals the SCF LSDA error in magnitude.

    Parameters
    ----------
    softening : float
        Softening length b > 0 of all interactions, in bohr.
    spacing : float
        Grid spacing h_x > 0 in bohr.
    half_width : float
        Half-width L of the grid in bohr, with 2L / h_x an integer.
    r_min : float
        Smallest separation of the scan in bohr, r_min > 0.
    r_max : float
        Largest separation of the scan in bohr, r_max > r_min.
    n_scan : int
        Number of evenly spaced scan separations, n_scan >= 2.

    Returns
    -------
    result : float
        Crossover separation R_x in bohr.

    Raises
    ------
    ValueError
        If 0 < r_min < r_max or n_scan >= 2 is violated, if g(r_min) <= 0 so that the scan starts past the crossover,
        or if g stays positive over the whole scan.
    '''
    return result  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
from scipy.optimize import brentq


def _crossover_gap(separation: float, softening: float, spacing: float, half_width: float) -> float:
    """Localized-orbital error magnitude minus self-consistent error magnitude at one separation."""
    err = _oracle_dissociation_errors(separation, softening, spacing, half_width)
    return abs(float(err[2])) - abs(float(err[0]))


def _oracle_crossover_separation(softening: float, spacing: float, half_width: float, r_min: float, r_max: float, n_scan: int) -> float:
    """Reference implementation."""
    import numpy as np
    from scipy.optimize import brentq
    if not (0.0 < r_min < r_max) or int(n_scan) < 2:
        raise ValueError("require 0 < r_min < r_max and n_scan >= 2")
    radii = np.linspace(r_min, r_max, int(n_scan))
    if _crossover_gap(radii[0], softening, spacing, half_width) <= 0.0:
        raise ValueError("the scan starts at or beyond the crossover")
    for k in range(1, radii.size):
        current = _crossover_gap(radii[k], softening, spacing, half_width)
        if current <= 0.0:
            return float(brentq(lambda r: _crossover_gap(r, softening, spacing, half_width), radii[k - 1], radii[k], xtol=1e-10))
    raise ValueError("no crossover within the scanned separations")

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Normal: unit softening on a coarse grid ---
        {
            "setup": "import numpy as np\n",
            "call": "crossover_separation(1.0, 0.1, 20.0, 3.0, 8.0, 6)",
            "gold_call": "_oracle_crossover_separation(1.0, 0.1, 20.0, 3.0, 8.0, 6)",
            "tol": 1e-7,
        },
        # --- Normal: a shorter softening and a finer grid ---
        {
            "setup": "import numpy as np\n",
            "call": "crossover_separation(0.75, 0.08, 16.0, 2.5, 8.0, 5)",
            "gold_call": "_oracle_crossover_separation(0.75, 0.08, 16.0, 2.5, 8.0, 5)",
            "tol": 1e-7,
        },
        # --- Boundary: only two scan points, so the whole range is a single bracket ---
        {
            "setup": "import numpy as np\n",
            "call": "crossover_separation(1.3, 0.1, 22.0, 3.0, 9.0, 2)",
            "gold_call": "_oracle_crossover_separation(1.3, 0.1, 22.0, 3.0, 9.0, 2)",
            "tol": 1e-7,
        },
        # --- Error: a scan that starts beyond the crossover must raise ValueError ---
        {
            "setup": "import numpy as np\n"
                     "def _probe(fn):\n"
                     "    try:\n"
                     "        fn(1.0, 0.1, 20.0, 6.5, 8.0, 3)\n"
                     "    except ValueError:\n"
                     "        return 1\n"
                     "    except Exception:\n"
                     "        return 2\n"
                     "    return 0\n",
            "call": "_probe(crossover_separation)",
            "gold_call": "_probe(_oracle_crossover_separation)",
        },
        # --- Error: a scan that ends before the crossover must raise ValueError ---
        {
            "setup": "import numpy as np\n"
                     "def _probe(fn):\n"
                     "    try:\n"
                     "        fn(1.0, 0.1, 20.0, 1.5, 3.5, 4)\n"
                     "    except ValueError:\n"
                     "        return 1\n"
                     "    except Exception:\n"
                     "        return 2\n"
                     "    return 0\n",
            "call": "_probe(crossover_separation)",
            "gold_call": "_probe(_oracle_crossover_separation)",
        },
    ]
