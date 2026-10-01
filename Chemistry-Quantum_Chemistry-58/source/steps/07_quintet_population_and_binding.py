"""
Triplet-pair population of the lowest quintet and the triplet-pair binding energy of the dark state.

The lowest quintet is taken as the reference for two triplets that are not electronically bound, so its own triplet-pair population is the check that the basis of products captures a pair of triplets, and the difference between its energy and that of the dark state is the triplet-pair binding energy. For the quintet the projections are onto the same Loewdin-orthonormalised Sz = 0 products as for the singlet, using the Sz = 0 component of the lowest S = 2 state, and the population is 3/2 times the sum of the squared projections, the factor following from the composition of the Sz = 0 component of a quintet pair in terms of the products of the three triplet Sz components. For a four-atom chain the quintet population is exactly 1.

The binding energy is E(lowest quintet) - E(2(1)Ag), positive when the pair is bound.

Returns
-------
numpy.ndarray of shape (2,): the triplet-pair population of the lowest quintet and the binding energy E(1(5)Ag) - E(2(1)Ag) in eV
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


def quintet_population_and_binding(n_sites: int, t0: float, delta: float, V: np.ndarray, U: float) -> np.ndarray:
    '''Quintet triplet-pair population and triplet-pair binding energy.

    Parameters
    ----------
    n_sites : int
        Number of atoms N, an even integer >= 4.
    t0 : float
        Mean hopping integral in eV, positive.
    delta : float
        Bond alternation, |delta| < 1, with the double bond first.
    V : numpy.ndarray
        Symmetric (n_sites, n_sites) interaction matrix in eV with a zero diagonal.
    U : float
        Hubbard parameter in eV, positive.

    Returns
    -------
    result : numpy.ndarray
        Shape (2,): 3/2 times the sum of the squared projections of the Sz = 0 component
        of the lowest quintet onto the Loewdin-orthonormalised triplet-pair products, and
        E(lowest quintet) - E(2(1)Ag) in eV.

    Raises
    ------
    ValueError
        If n_sites is not an even integer >= 4, V is not a symmetric (n_sites, n_sites)
        array, t0 or U is not positive, |delta| >= 1, or the products are linearly
        dependent.
    '''
    return result

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


def _oracle_quintet_population_and_binding(n_sites: int, t0: float, delta: float, V: np.ndarray, U: float) -> np.ndarray:
    Vm = _validate_model(n_sites, t0, delta, V, U)
    if n_sites < 4:
        raise ValueError("n_sites must be at least 4 for a quintet state")
    d = _dark_states(n_sites, t0, delta, Vm, U)
    # Sz=0 component of the lowest quintet: apply S- twice to the Sz=2 eigenvector and normalise
    v1, s1 = _s_minus(d["vq"], d["q_sec"])
    v0, s0 = _s_minus(v1, s1)
    v0 = v0 / np.linalg.norm(v0)
    sec_f, B = _triplet_pair_basis(n_sites, t0, delta, Vm, U)
    pq = float(_lowdin_populations(B, v0, 1.5).sum())
    return np.array([pq, d["eq"] - d["w"][d["ag2"]]], dtype=float)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Normal: the ten-atom benchmark chain ---
        {
            "setup": """import numpy as np
V = np.array([[0.000000000000, 3.028724843957, 2.163107408042, 1.574142730777, 1.224092567900, 0.999724120164, 0.838156692309, 0.725480211265, 0.634743527644, 0.567792876722],
              [3.028724843957, 0.000000000000, 2.924747953395, 2.163107408042, 1.529921801556, 1.224092567900, 0.980582578949, 0.838156692309, 0.715161810280, 0.634743527644],
              [2.163107408042, 2.924747953395, 0.000000000000, 3.028724843957, 2.163107408042, 1.574142730777, 1.224092567900, 0.999724120164, 0.838156692309, 0.725480211265],
              [1.574142730777, 2.163107408042, 3.028724843957, 0.000000000000, 2.924747953395, 2.163107408042, 1.529921801556, 1.224092567900, 0.980582578949, 0.838156692309],
              [1.224092567900, 1.529921801556, 2.163107408042, 2.924747953395, 0.000000000000, 3.028724843957, 2.163107408042, 1.574142730777, 1.224092567900, 0.999724120164],
              [0.999724120164, 1.224092567900, 1.574142730777, 2.163107408042, 3.028724843957, 0.000000000000, 2.924747953395, 2.163107408042, 1.529921801556, 1.224092567900],
              [0.838156692309, 0.980582578949, 1.224092567900, 1.529921801556, 2.163107408042, 2.924747953395, 0.000000000000, 3.028724843957, 2.163107408042, 1.574142730777],
              [0.725480211265, 0.838156692309, 0.999724120164, 1.224092567900, 1.574142730777, 2.163107408042, 3.028724843957, 0.000000000000, 2.924747953395, 2.163107408042],
              [0.634743527644, 0.715161810280, 0.838156692309, 0.980582578949, 1.224092567900, 1.529921801556, 2.163107408042, 2.924747953395, 0.000000000000, 3.028724843957],
              [0.567792876722, 0.634743527644, 0.725480211265, 0.838156692309, 0.999724120164, 1.224092567900, 1.574142730777, 2.163107408042, 3.028724843957, 0.000000000000]])
n_sites = 10
t0 = 2.5
delta = 0.1
U = 4.0
""",
            "call": "quintet_population_and_binding(n_sites, t0, delta, V.copy(), U)",
            "gold_call": "_oracle_quintet_population_and_binding(n_sites, t0, delta, V.copy(), U)",
            "tol": 1e-07,
        },
        # --- Normal: the eight-atom benchmark chain ---
        {
            "setup": """import numpy as np
V = np.array([[0.000000000000, 3.028724843957, 2.163107408042, 1.574142730777, 1.224092567900, 0.999724120164, 0.838156692309, 0.725480211265],
              [3.028724843957, 0.000000000000, 2.924747953395, 2.163107408042, 1.529921801556, 1.224092567900, 0.980582578949, 0.838156692309],
              [2.163107408042, 2.924747953395, 0.000000000000, 3.028724843957, 2.163107408042, 1.574142730777, 1.224092567900, 0.999724120164],
              [1.574142730777, 2.163107408042, 3.028724843957, 0.000000000000, 2.924747953395, 2.163107408042, 1.529921801556, 1.224092567900],
              [1.224092567900, 1.529921801556, 2.163107408042, 2.924747953395, 0.000000000000, 3.028724843957, 2.163107408042, 1.574142730777],
              [0.999724120164, 1.224092567900, 1.574142730777, 2.163107408042, 3.028724843957, 0.000000000000, 2.924747953395, 2.163107408042],
              [0.838156692309, 0.980582578949, 1.224092567900, 1.529921801556, 2.163107408042, 2.924747953395, 0.000000000000, 3.028724843957],
              [0.725480211265, 0.838156692309, 0.999724120164, 1.224092567900, 1.574142730777, 2.163107408042, 3.028724843957, 0.000000000000]])
n_sites = 8
t0 = 2.5
delta = 0.1
U = 4.0
""",
            "call": "quintet_population_and_binding(n_sites, t0, delta, V.copy(), U)",
            "gold_call": "_oracle_quintet_population_and_binding(n_sites, t0, delta, V.copy(), U)",
            "tol": 1e-07,
        },
        # --- Boundary: the four-atom chain, whose quintet is exactly a pair of triplets ---
        {
            "setup": """import numpy as np
V = np.array([[0.000000000000, 3.028724843957, 2.163107408042, 1.574142730777],
              [3.028724843957, 0.000000000000, 2.924747953395, 2.163107408042],
              [2.163107408042, 2.924747953395, 0.000000000000, 3.028724843957],
              [1.574142730777, 2.163107408042, 3.028724843957, 0.000000000000]])
n_sites = 4
t0 = 2.5
delta = 0.1
U = 4.0
""",
            "call": "quintet_population_and_binding(n_sites, t0, delta, V.copy(), U)",
            "gold_call": "_oracle_quintet_population_and_binding(n_sites, t0, delta, V.copy(), U)",
            "tol": 1e-07,
        },
        # --- Normal: the strongly correlated six-atom chain ---
        {
            "setup": """import numpy as np
V = np.array([[0.000000000000, 4.325076507107, 2.514470076928, 1.695139627994, 1.278461106161, 1.028690941455],
              [4.325076507107, 0.000000000000, 4.037422412364, 2.514470076928, 1.640282944134, 1.278461106161],
              [2.514470076928, 4.037422412364, 0.000000000000, 4.325076507107, 2.514470076928, 1.695139627994],
              [1.695139627994, 2.514470076928, 4.325076507107, 0.000000000000, 4.037422412364, 2.514470076928],
              [1.278461106161, 1.640282944134, 2.514470076928, 4.037422412364, 0.000000000000, 4.325076507107],
              [1.028690941455, 1.278461106161, 1.695139627994, 2.514470076928, 4.325076507107, 0.000000000000]])
n_sites = 6
t0 = 2.5
delta = 0.1
U = 12.0
""",
            "call": "quintet_population_and_binding(n_sites, t0, delta, V.copy(), U)",
            "gold_call": "_oracle_quintet_population_and_binding(n_sites, t0, delta, V.copy(), U)",
            "tol": 1e-07,
        },
        # --- Normal: the eight-atom straight chain of the earlier parametrisation ---
        {
            "setup": """import numpy as np
V = np.array([[0.000000000000, 4.325422955697, 2.447611173136, 1.675898821515, 1.269166956406, 1.019964859237, 0.852089448374, 0.731463994105],
              [4.325422955697, 0.000000000000, 4.325422955697, 2.447611173136, 1.675898821515, 1.269166956406, 1.019964859237, 0.852089448374],
              [2.447611173136, 4.325422955697, 0.000000000000, 4.325422955697, 2.447611173136, 1.675898821515, 1.269166956406, 1.019964859237],
              [1.675898821515, 2.447611173136, 4.325422955697, 0.000000000000, 4.325422955697, 2.447611173136, 1.675898821515, 1.269166956406],
              [1.269166956406, 1.675898821515, 2.447611173136, 4.325422955697, 0.000000000000, 4.325422955697, 2.447611173136, 1.675898821515],
              [1.019964859237, 1.269166956406, 1.675898821515, 2.447611173136, 4.325422955697, 0.000000000000, 4.325422955697, 2.447611173136],
              [0.852089448374, 1.019964859237, 1.269166956406, 1.675898821515, 2.447611173136, 4.325422955697, 0.000000000000, 4.325422955697],
              [0.731463994105, 0.852089448374, 1.019964859237, 1.269166956406, 1.675898821515, 2.447611173136, 4.325422955697, 0.000000000000]])
n_sites = 8
t0 = 2.4
delta = 0.08333333333333333
U = 8.0
""",
            "call": "quintet_population_and_binding(n_sites, t0, delta, V.copy(), U)",
            "gold_call": "_oracle_quintet_population_and_binding(n_sites, t0, delta, V.copy(), U)",
            "tol": 1e-07,
        },
        # --- Edge: a six-atom chain whose two terminal atoms interact ten percent more weakly with the others ---
        {
            "setup": """import numpy as np
V = np.array([[0.000000000000, 2.725852359562, 1.946796667238, 1.416728457699, 1.101683311110, 0.899751708148],
              [2.725852359562, 0.000000000000, 2.924747953395, 2.163107408042, 1.529921801556, 1.101683311110],
              [1.946796667238, 2.924747953395, 0.000000000000, 3.028724843957, 2.163107408042, 1.416728457699],
              [1.416728457699, 2.163107408042, 3.028724843957, 0.000000000000, 2.924747953395, 1.946796667238],
              [1.101683311110, 1.529921801556, 2.163107408042, 2.924747953395, 0.000000000000, 2.725852359562],
              [0.899751708148, 1.101683311110, 1.416728457699, 1.946796667238, 2.725852359562, 0.000000000000]])
n_sites = 6
t0 = 2.5
delta = 0.1
U = 4.0
""",
            "call": "quintet_population_and_binding(n_sites, t0, delta, V.copy(), U)",
            "gold_call": "_oracle_quintet_population_and_binding(n_sites, t0, delta, V.copy(), U)",
            "tol": 1e-07,
        },
        # --- Invalid: a two-atom chain has no quintet ---
        {
            "setup": """import numpy as np
V = np.array([[0.000000000000, 3.028724843957],
              [3.028724843957, 0.000000000000]])
n_sites = 2
t0 = 2.5
delta = 0.1
U = 4.0
def run_model():
    try:
        quintet_population_and_binding(n_sites, t0, delta, V.copy(), U)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_oracle():
    try:
        _oracle_quintet_population_and_binding(n_sites, t0, delta, V.copy(), U)
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
