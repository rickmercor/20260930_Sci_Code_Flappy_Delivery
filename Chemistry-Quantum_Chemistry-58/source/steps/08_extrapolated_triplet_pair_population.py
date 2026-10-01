"""
Infinite-chain triplet-pair population of the dark state from an exact three-point power-law fit through the populations of three chain lengths.

The triplet-pair population of the dark state grows with chain length and is extrapolated to the infinite chain with the form P(N) = a N^(-alpha) + c, where N is the number of atoms and c is the limiting population. With three chain lengths the three parameters are determined exactly: the ratio of successive differences of P fixes alpha, after which a and c follow; alpha must be positive, so the fit exists only when the differences shrink faster than the logarithmic ratio of the chain lengths allows.

This step assembles the whole pipeline: the interaction matrix of each chain from its geometry, the dark state and the covalent triplet families of its subchains, the Loewdin-orthonormalised products and the summed populations, and finally the fit.

Returns
-------
float: the extrapolated infinite-chain triplet-pair population c of the dark state
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import itertools
import math
import types

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.optimize import brentq


def extrapolated_triplet_pair_population(bond_double: float, bond_single: float, bond_angle_deg: float, t0: float, delta: float, U: float, eps_r: float, chain_lengths: np.ndarray) -> float:
    '''Infinite-chain triplet-pair population from three chain lengths.

    Parameters
    ----------
    bond_double : float
        Length of the double bonds in angstrom, positive.
    bond_single : float
        Length of the single bonds in angstrom, positive.
    bond_angle_deg : float
        Bond angle in degrees, in (0, 180].
    t0 : float
        Mean hopping integral in eV, positive.
    delta : float
        Bond alternation, |delta| < 1, with the double bond first.
    U : float
        Hubbard parameter in eV, positive.
    eps_r : float
        Relative permittivity of the Ohno interaction, positive.
    chain_lengths : numpy.ndarray
        Three increasing even integers >= 4, the numbers of atoms of the chains.

    Returns
    -------
    c : float
        The constant of the exact fit P(N) = a N^(-alpha) + c through the three
        dark-state triplet-pair populations, as a native Python float.

    Raises
    ------
    ValueError
        If chain_lengths does not hold exactly three increasing even integers >= 4, a
        model parameter is invalid as in the earlier steps, or no positive alpha fits
        the three populations.
    '''
    return c

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import itertools
import math
import types

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.optimize import brentq


def _fit_three(nvals, pvals):
    n6, n8, n10 = nvals; p6, p8, p10 = pvals
    g = lambda a: (p6 - p8) * (n8 ** -a - n10 ** -a) - (p8 - p10) * (n6 ** -a - n8 ** -a)
    lo, hi = 1e-6, 60.0
    if g(lo) * g(hi) > 0:
        raise ValueError("no positive decay exponent fits the three populations")
    alpha = brentq(g, lo, hi, xtol=1e-14, rtol=1e-14, maxiter=500)
    a = (p8 - p10) / (n8 ** -alpha - n10 ** -alpha)
    c = p10 - a * n10 ** -alpha
    return alpha, a, c


def _oracle_extrapolated_triplet_pair_population(bond_double: float, bond_single: float, bond_angle_deg: float, t0: float, delta: float, U: float, eps_r: float, chain_lengths: np.ndarray) -> float:
    lens = np.asarray(chain_lengths)
    if lens.ndim != 1 or lens.size != 3:
        raise ValueError("chain_lengths must hold exactly three chain lengths")
    lens = [int(x) for x in lens]
    if any(x != lens_i for x, lens_i in zip(np.asarray(chain_lengths), lens)) or not (4 <= lens[0] < lens[1] < lens[2]):
        raise ValueError("chain_lengths must be three increasing even integers >= 4")
    pops = []
    for n in lens:
        V = _oracle_ohno_interaction_matrix(n, bond_double, bond_single, bond_angle_deg, U, eps_r)
        pops.append(float(_oracle_triplet_pair_populations(n, t0, delta, V, U).sum()))
    alpha, a, c = _fit_three(lens, pops)
    return float(c)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Normal: the benchmark, chains of 6, 8 and 10 atoms ---
        {
            "setup": """import numpy as np
""",
            "call": "extrapolated_triplet_pair_population(1.35, 1.46, 120.0, 2.5, 0.10, 4.0, 2.3, np.array([6, 8, 10]))",
            "gold_call": "_oracle_extrapolated_triplet_pair_population(1.35, 1.46, 120.0, 2.5, 0.10, 4.0, 2.3, np.array([6, 8, 10]))",
            "tol": 1e-07,
        },
        # --- Normal: the benchmark parameters on chains of 4, 6 and 8 atoms ---
        {
            "setup": """import numpy as np
""",
            "call": "extrapolated_triplet_pair_population(1.35, 1.46, 120.0, 2.5, 0.10, 4.0, 2.3, np.array([4, 6, 8]))",
            "gold_call": "_oracle_extrapolated_triplet_pair_population(1.35, 1.46, 120.0, 2.5, 0.10, 4.0, 2.3, np.array([4, 6, 8]))",
            "tol": 1e-07,
        },
        # --- Normal: the straight chain of the earlier parametrisation, chains of 4, 6 and 8 atoms ---
        {
            "setup": """import numpy as np
""",
            "call": "extrapolated_triplet_pair_population(1.40, 1.40, 180.0, 2.4, 1.0 / 12.0, 8.0, 2.0, np.array([4, 6, 8]))",
            "gold_call": "_oracle_extrapolated_triplet_pair_population(1.40, 1.40, 180.0, 2.4, 1.0 / 12.0, 8.0, 2.0, np.array([4, 6, 8]))",
            "tol": 1e-07,
        },
        # --- Edge: the strongly correlated benchmark geometry, chains of 4, 6 and 8 atoms ---
        {
            "setup": """import numpy as np
""",
            "call": "extrapolated_triplet_pair_population(1.35, 1.46, 120.0, 2.5, 0.10, 12.0, 2.3, np.array([4, 6, 8]))",
            "gold_call": "_oracle_extrapolated_triplet_pair_population(1.35, 1.46, 120.0, 2.5, 0.10, 12.0, 2.3, np.array([4, 6, 8]))",
            "tol": 1e-07,
        },
        # --- Invalid: an odd chain length ---
        {
            "setup": """import numpy as np
def run_model():
    try:
        extrapolated_triplet_pair_population(1.35, 1.46, 120.0, 2.5, 0.10, 4.0, 2.3, np.array([5, 6, 8]))
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_oracle():
    try:
        _oracle_extrapolated_triplet_pair_population(1.35, 1.46, 120.0, 2.5, 0.10, 4.0, 2.3, np.array([5, 6, 8]))
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_oracle()",
        },
        # --- Invalid: four chain lengths ---
        {
            "setup": """import numpy as np
def run_model():
    try:
        extrapolated_triplet_pair_population(1.35, 1.46, 120.0, 2.5, 0.10, 4.0, 2.3, np.array([4, 6, 8, 10]))
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_oracle():
    try:
        _oracle_extrapolated_triplet_pair_population(1.35, 1.46, 120.0, 2.5, 0.10, 4.0, 2.3, np.array([4, 6, 8, 10]))
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_oracle()",
        },
    ]
