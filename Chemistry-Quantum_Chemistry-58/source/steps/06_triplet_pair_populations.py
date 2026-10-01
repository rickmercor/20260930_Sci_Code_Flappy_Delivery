"""
Triplet-pair population of the dark state 2(1)Ag resolved over the Loewdin-orthonormalised triplet-pair products.

The real-space triplet-pair population of a singlet is defined by projecting it onto the orthonormalised triplet-pair products. The raw products of the previous step are orthonormalised by Loewdin's symmetric procedure, which multiplies the set by the inverse square root of its overlap matrix and is the unique orthonormalisation that stays closest to the original products; each orthonormalised vector inherits the (m, j) label of its raw product. The population attributed to one basis element is three times the squared projection of the dark state onto it, the factor of three accounting for the two other Sz components of the spin-symmetrised singlet triplet pair, and the total population is the sum over the basis.

The dark state is the 2(1)Ag state defined in the step on dark-state energies, taken in the Sz = 0 sector. The per-element populations do not depend on the phases of the subchain eigenvectors or on the sign convention of the tensor products.

Returns
-------
numpy.ndarray of shape (N_d (N_d - 1)/2,): three times the squared projection of the 2(1)Ag state onto each Loewdin-orthonormalised triplet-pair product, ordered by m ascending then j ascending
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


def triplet_pair_populations(n_sites: int, t0: float, delta: float, V: np.ndarray, U: float) -> np.ndarray:
    '''Per-element triplet-pair populations of the dark state.

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
    populations : numpy.ndarray
        Shape (N_d (N_d - 1)/2,) with N_d = n_sites/2: for the Loewdin-orthonormalised
        products in the order m = 1..N_d - 1, j = 1..m, three times the squared projection
        of the 2(1)Ag state onto each; the sum is the triplet-pair population.

    Raises
    ------
    ValueError
        If n_sites is not an even integer >= 4, V is not a symmetric (n_sites, n_sites)
        array, t0 or U is not positive, |delta| >= 1, or the products are linearly
        dependent.
    '''
    return populations

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


def _lowdin_populations(B, psi, factor):
    S = B.T @ B
    ev, Uv = np.linalg.eigh(0.5 * (S + S.T))
    if ev.min() <= 1e-10:
        raise ValueError("the triplet-pair products are linearly dependent")
    Bo = B @ (Uv @ np.diag(ev ** -0.5) @ Uv.T)
    return factor * (Bo.T @ psi) ** 2


def _oracle_triplet_pair_populations(n_sites: int, t0: float, delta: float, V: np.ndarray, U: float) -> np.ndarray:
    Vm = _validate_model(n_sites, t0, delta, V, U)
    if n_sites < 4:
        raise ValueError("n_sites must be at least 4 to form a triplet pair")
    d = _dark_states(n_sites, t0, delta, Vm, U)
    sec_f, B = _triplet_pair_basis(n_sites, t0, delta, Vm, U)
    return _lowdin_populations(B, d["v"][:, d["ag2"]], 3.0)

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
            "call": "triplet_pair_populations(n_sites, t0, delta, V.copy(), U)",
            "gold_call": "_oracle_triplet_pair_populations(n_sites, t0, delta, V.copy(), U)",
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
            "call": "triplet_pair_populations(n_sites, t0, delta, V.copy(), U)",
            "gold_call": "_oracle_triplet_pair_populations(n_sites, t0, delta, V.copy(), U)",
            "tol": 1e-07,
        },
        # --- Edge: the four-atom chain close to the noninteracting limit ---
        {
            "setup": """import numpy as np
V = np.array([[0.000000000000, 0.009999981088, 0.009999924352, 0.009999829795],
              [0.009999981088, 0.000000000000, 0.009999981088, 0.009999924352],
              [0.009999924352, 0.009999981088, 0.000000000000, 0.009999981088],
              [0.009999829795, 0.009999924352, 0.009999981088, 0.000000000000]])
n_sites = 4
t0 = 2.4
delta = 0.08333333333333333
U = 0.01
""",
            "call": "triplet_pair_populations(n_sites, t0, delta, V.copy(), U)",
            "gold_call": "_oracle_triplet_pair_populations(n_sites, t0, delta, V.copy(), U)",
            "tol": 1e-07,
        },
        # --- Edge: the weakly screened eight-atom chain ---
        {
            "setup": """import numpy as np
V = np.array([[0.000000000000, 3.745220485711, 3.313404468934, 2.806363405152, 2.377964623387, 2.041983858636, 1.768394334390, 1.562024190845],
              [3.745220485711, 0.000000000000, 3.706654150805, 3.313404468934, 2.758192727283, 2.377964623387, 2.010962089284, 1.768394334390],
              [3.313404468934, 3.706654150805, 0.000000000000, 3.745220485711, 3.313404468934, 2.806363405152, 2.377964623387, 2.041983858636],
              [2.806363405152, 3.313404468934, 3.745220485711, 0.000000000000, 3.706654150805, 3.313404468934, 2.758192727283, 2.377964623387],
              [2.377964623387, 2.758192727283, 3.313404468934, 3.706654150805, 0.000000000000, 3.745220485711, 3.313404468934, 2.806363405152],
              [2.041983858636, 2.377964623387, 2.806363405152, 3.313404468934, 3.745220485711, 0.000000000000, 3.706654150805, 3.313404468934],
              [1.768394334390, 2.010962089284, 2.377964623387, 2.758192727283, 3.313404468934, 3.706654150805, 0.000000000000, 3.745220485711],
              [1.562024190845, 1.768394334390, 2.041983858636, 2.377964623387, 2.806363405152, 3.313404468934, 3.745220485711, 0.000000000000]])
n_sites = 8
t0 = 2.5
delta = 0.1
U = 4.0
""",
            "call": "triplet_pair_populations(n_sites, t0, delta, V.copy(), U)",
            "gold_call": "_oracle_triplet_pair_populations(n_sites, t0, delta, V.copy(), U)",
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
            "call": "triplet_pair_populations(n_sites, t0, delta, V.copy(), U)",
            "gold_call": "_oracle_triplet_pair_populations(n_sites, t0, delta, V.copy(), U)",
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
            "call": "triplet_pair_populations(n_sites, t0, delta, V.copy(), U)",
            "gold_call": "_oracle_triplet_pair_populations(n_sites, t0, delta, V.copy(), U)",
            "tol": 1e-07,
        },
        # --- Edge: an eight-atom chain whose two terminal atoms interact ten percent more weakly with the others ---
        {
            "setup": """import numpy as np
V = np.array([[0.000000000000, 2.725852359562, 1.946796667238, 1.416728457699, 1.101683311110, 0.899751708148, 0.754341023078, 0.652932190139],
              [2.725852359562, 0.000000000000, 2.924747953395, 2.163107408042, 1.529921801556, 1.224092567900, 0.980582578949, 0.754341023078],
              [1.946796667238, 2.924747953395, 0.000000000000, 3.028724843957, 2.163107408042, 1.574142730777, 1.224092567900, 0.899751708148],
              [1.416728457699, 2.163107408042, 3.028724843957, 0.000000000000, 2.924747953395, 2.163107408042, 1.529921801556, 1.101683311110],
              [1.101683311110, 1.529921801556, 2.163107408042, 2.924747953395, 0.000000000000, 3.028724843957, 2.163107408042, 1.416728457699],
              [0.899751708148, 1.224092567900, 1.574142730777, 2.163107408042, 3.028724843957, 0.000000000000, 2.924747953395, 1.946796667238],
              [0.754341023078, 0.980582578949, 1.224092567900, 1.529921801556, 2.163107408042, 2.924747953395, 0.000000000000, 2.725852359562],
              [0.652932190139, 0.754341023078, 0.899751708148, 1.101683311110, 1.416728457699, 1.946796667238, 2.725852359562, 0.000000000000]])
n_sites = 8
t0 = 2.5
delta = 0.1
U = 4.0
""",
            "call": "triplet_pair_populations(n_sites, t0, delta, V.copy(), U)",
            "gold_call": "_oracle_triplet_pair_populations(n_sites, t0, delta, V.copy(), U)",
            "tol": 1e-07,
        },
        # --- Invalid: a two-atom chain cannot be cut ---
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
        triplet_pair_populations(n_sites, t0, delta, V.copy(), U)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_oracle():
    try:
        _oracle_triplet_pair_populations(n_sites, t0, delta, V.copy(), U)
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
