"""
Locate, for every pair of a reactant and a product vibronic level, the point on the bath coordinate at which that pair's transition conserves energy.

A golden-rule transition between two vibronic states is only allowed where their energies are equal, so the coupling that matters for a pair is the one at that geometry. The bath is a single coordinate q shared by every diabatic electronic state, and each state's energies ride on a harmonic well of the same force constant k centred on that state's own bath minimum:

E(j, xi; q) = E(j, xi) + (k / 2) * (q - q_j)**2.

The reactant well sits at q = 0 and the product well at q_II, the displacement that makes the reorganisation energy of the pair of wells equal to lambda, that is (k / 2) * q_II**2 = lambda, so q_II = (2 * lambda / k) ** 0.5. The proton potentials of the reactant and of the product do not change along q, so their vibronic energies at the bottom of their wells are the levels supplied here.

Setting E(I, mu; q) = E(II, nu; q) and using the same k on both sides makes the condition linear in q, with the single root

q*(mu, nu) = (lambda + E(II, nu) - E(I, mu)) / (k * q_II).

Each pair therefore has its own crossing point, and the nine pairs of a three-level problem sit at nine different geometries; the pair crosses (lambda + E(II, nu) - E(I, mu))**2 / (4 * lambda) above the reactant minimum, which is the familiar activation energy of a harmonic bath. The returned array is indexed by the reactant level first and the product level second.

Returns
-------
np.ndarray of shape (n_mu, n_nu), the bath coordinate of the crossing point of every reactant-product pair
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def crossing_points(reorganization: float, force_constant: float, e_reactant: np.ndarray,
                    e_product: np.ndarray) -> np.ndarray:
    '''Bath coordinate at which each reactant-product pair conserves energy.

    Parameters
    ----------
    reorganization : float
        Positive reorganisation energy lambda in eV.
    force_constant : float
        Positive bath force constant k in eV; the bath coordinate is
        dimensionless.
    e_reactant : np.ndarray
        (n_mu,) vibronic energies of the reactant in eV at the bottom of its
        bath well, ascending.
    e_product : np.ndarray
        (n_nu,) vibronic energies of the product in eV at the bottom of its
        bath well, ascending.

    Returns
    -------
    crossings : np.ndarray
        (n_mu, n_nu) bath coordinate of the crossing point of each pair.

    Raises
    ------
    ValueError
        If reorganization or force_constant is not positive and finite, or if
        either level array is not a non-empty finite one-dimensional array.
    '''
    return crossings  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _level_array(values, name):
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 1 or arr.size == 0 or not np.isfinite(arr).all():
        raise ValueError(f"{name} must be a non-empty finite one-dimensional array")
    return arr


def _bath_pair(reorganization, force_constant):
    lam = float(reorganization)
    k = float(force_constant)
    if not (np.isfinite(lam) and lam > 0.0):
        raise ValueError("reorganization must be positive and finite")
    if not (np.isfinite(k) and k > 0.0):
        raise ValueError("force_constant must be positive and finite")
    return lam, k, np.sqrt(2.0 * lam / k)


def _oracle_crossing_points(reorganization: float, force_constant: float,
                            e_reactant: np.ndarray, e_product: np.ndarray) -> np.ndarray:
    """Reference implementation."""
    lam, k, q_product = _bath_pair(reorganization, force_constant)
    er = _level_array(e_reactant, "e_reactant")
    ep = _level_array(e_product, "e_product")
    return (lam + ep[None, :] - er[:, None]) / (k * q_product)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        {
            "setup": """import numpy as np
# normal: three levels on each side of the five-state assembly
er = np.array([3.2458, 3.5748, 3.5826])
ep = np.array([3.1028, 3.4302, 3.4357])
""",
            "call": "crossing_points(0.280, 0.8, er, ep)",
            "gold_call": "_oracle_crossing_points(0.280, 0.8, er, ep)",
        },
        {
            "setup": """import numpy as np
# normal: a stiffer bath and a larger reorganisation energy move every crossing
er = np.array([3.2458, 3.5748, 3.5826])
ep = np.array([3.1028, 3.4302, 3.4357])
""",
            "call": "crossing_points(0.450, 1.5, er, ep)",
            "gold_call": "_oracle_crossing_points(0.450, 1.5, er, ep)",
        },
        {
            "setup": """import numpy as np
# boundary: an activationless pair, whose gap equals minus lambda, crosses at
# the reactant minimum
er = np.array([3.30])
ep = np.array([3.05])
""",
            "call": "crossing_points(0.25, 1.0, er, ep)",
            "gold_call": "_oracle_crossing_points(0.25, 1.0, er, ep)",
        },
        {
            "setup": """import numpy as np
# edge: ladders of different lengths, so the two axes are distinct, and an
# uphill pair, which crosses beyond the product well
er = np.array([3.20, 3.55])
ep = np.array([3.00, 3.40, 3.44])
""",
            "call": "crossing_points(0.31, 0.65, er, ep)",
            "gold_call": "_oracle_crossing_points(0.31, 0.65, er, ep)",
        },
        {
            "setup": """import numpy as np
# edge: a small reorganisation energy, where the crossings move far out along
# the bath coordinate
er = np.array([3.2458, 3.5748])
ep = np.array([3.1028, 3.4302])
""",
            "call": "crossing_points(0.06, 0.8, er, ep)",
            "gold_call": "_oracle_crossing_points(0.06, 0.8, er, ep)",
        },
        {
            "setup": """import numpy as np
er = np.array([3.20]); ep = np.array([3.00])
def run(f):
    try:
        f(0.0, 0.8, er, ep); return 0        # zero reorganisation energy
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run(crossing_points)",
            "gold_call": "run(_oracle_crossing_points)",
        },
        {
            "setup": """import numpy as np
er = np.array([3.20]); ep = np.array([3.00])
def run(f):
    try:
        f(0.28, -0.8, er, ep); return 0      # negative force constant
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run(crossing_points)",
            "gold_call": "run(_oracle_crossing_points)",
        },
        {
            "setup": """import numpy as np
er = np.array([3.20]); ep = np.array([])
def run(f):
    try:
        f(0.28, 0.8, er, ep); return 0       # empty product ladder
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run(crossing_points)",
            "gold_call": "run(_oracle_crossing_points)",
        },
    ]
