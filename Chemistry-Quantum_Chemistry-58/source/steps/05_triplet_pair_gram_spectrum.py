"""
Eigenvalues of the overlap matrix of the raw triplet-pair tensor products built from the covalent triplets of every left and right subchain.

A chain of N_d = N/2 ethylene units can be cut after unit m into a left subchain of atoms 1 to 2m and a right subchain of atoms 2m + 1 to N, for every m from 1 to N_d - 1. Each subchain is an open chain of the same kind: its atoms keep their hopping integrals, so both subchains start with a double bond, and the interaction matrix of a subchain is the corresponding diagonal block of V (the leading 2m x 2m block for the left subchain, the trailing block for the right one). A triplet-pair product state is the tensor product of an Sz = 0 covalent triplet of the left subchain, T_j(m), the (j+1)-th lowest member of that subchain's covalent family with j = 0, ..., m - 1, and the lowest covalent triplet T_1(N_d - m) of the right subchain, written as a state of the full chain; the electrons of the two subchains occupy disjoint sets of atoms. There are N_d (N_d - 1)/2 such products, ordered by m ascending and then by j ascending.

The products are normalised but not mutually orthogonal, and their overlap (Gram) matrix carries the information needed to orthonormalise them. Its eigenvalues do not depend on the phases chosen for the subchain eigenvectors or on the sign convention of the tensor product; for a four-atom chain the basis has one element and the single eigenvalue is 1.

Returns
-------
numpy.ndarray of shape (N_d (N_d - 1)/2,): the eigenvalues of the overlap matrix of the raw tensor products, ascending
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


def triplet_pair_gram_spectrum(n_sites: int, t0: float, delta: float, V: np.ndarray, U: float) -> np.ndarray:
    '''Eigenvalues of the overlap matrix of the triplet-pair products.

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
    spectrum : numpy.ndarray
        Shape (N_d (N_d - 1)/2,) with N_d = n_sites/2: the eigenvalues, in ascending
        order, of the overlap matrix of the normalised products T_j(m) x T_1(N_d - m)
        for m = 1..N_d - 1 and j = 1..m.

    Raises
    ------
    ValueError
        If n_sites is not an even integer >= 4, V is not a symmetric (n_sites, n_sites)
        array, t0 or U is not positive, |delta| >= 1, or a subchain holds fewer covalent
        triplets than the construction needs.
    '''
    return spectrum

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


def _embed(psi_l, sec_l, psi_r, sec_r, sec_f):
    """Tensor product of a left-subchain state (sites 0..nl-1) and a right-subchain state (sites nl..n-1)
    as a vector of the full-chain sector (global sign immaterial for populations)."""
    nl = sec_l.n
    PL = psi_l.reshape(sec_l.nu, sec_l.nd); PR = psi_r.reshape(sec_r.nu, sec_r.nd)
    iu = np.array([[sec_f.iu[a | (b << nl)] for b in sec_r.su] for a in sec_l.su])
    idn = np.array([[sec_f.id[a | (b << nl)] for b in sec_r.sd] for a in sec_l.sd])
    out = np.zeros(sec_f.dim)
    idx = iu[:, :, None, None] * sec_f.nd + idn[None, None, :, :]
    out[idx.ravel()] = (PL[:, None, :, None] * PR[None, :, None, :]).ravel()
    return out


def _subchain_family(Vm, t0, delta, U, start, sites, members, cache):
    """Covalent triplet family of the subchain of `sites` atoms starting at atom index `start`, memoised in `cache`."""
    key = (start, sites, members)
    if key not in cache:
        cache[key] = _covalent_triplets(sites, t0, delta, Vm[start:start + sites, start:start + sites], U, members)
    return cache[key]


def _triplet_pair_basis(n_sites, t0, delta, V, U):
    """Raw (non-orthogonal) T0 x T0 products |T_j(m)> x |T_1(Nd-m)>, ordered by m = 1..Nd-1 and j = 1..m."""
    nd = n_sites // 2
    Vm = np.asarray(V, dtype=float)
    sec_f = _sector(n_sites, nd, nd)
    cache = {}
    cols = []
    for m in range(1, nd):
        sec_l, wl, vl, labl, faml = _subchain_family(Vm, t0, delta, U, 0, 2 * m, m, cache)
        sec_r, wr, vr, labr, famr = _subchain_family(Vm, t0, delta, U, 2 * m, 2 * (nd - m), 1, cache)
        for j in range(m):
            cols.append(_embed(vl[:, faml[j]], sec_l, vr[:, famr[0]], sec_r, sec_f))
    return sec_f, np.array(cols).T


def _oracle_triplet_pair_gram_spectrum(n_sites: int, t0: float, delta: float, V: np.ndarray, U: float) -> np.ndarray:
    Vm = _validate_model(n_sites, t0, delta, V, U)
    if n_sites < 4:
        raise ValueError("n_sites must be at least 4 to form a triplet pair")
    sec_f, B = _triplet_pair_basis(n_sites, t0, delta, Vm, U)
    S = B.T @ B
    return np.sort(np.linalg.eigvalsh(0.5 * (S + S.T)))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Normal: the ten products of the ten-atom benchmark chain ---
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
            "call": "triplet_pair_gram_spectrum(n_sites, t0, delta, V.copy(), U)",
            "gold_call": "_oracle_triplet_pair_gram_spectrum(n_sites, t0, delta, V.copy(), U)",
            "tol": 1e-08,
        },
        # --- Normal: the six products of the eight-atom benchmark chain ---
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
            "call": "triplet_pair_gram_spectrum(n_sites, t0, delta, V.copy(), U)",
            "gold_call": "_oracle_triplet_pair_gram_spectrum(n_sites, t0, delta, V.copy(), U)",
            "tol": 1e-08,
        },
        # --- Normal: the straight six-atom chain of the earlier parametrisation ---
        {
            "setup": """import numpy as np
V = np.array([[0.000000000000, 4.325422955697, 2.447611173136, 1.675898821515, 1.269166956406, 1.019964859237],
              [4.325422955697, 0.000000000000, 4.325422955697, 2.447611173136, 1.675898821515, 1.269166956406],
              [2.447611173136, 4.325422955697, 0.000000000000, 4.325422955697, 2.447611173136, 1.675898821515],
              [1.675898821515, 2.447611173136, 4.325422955697, 0.000000000000, 4.325422955697, 2.447611173136],
              [1.269166956406, 1.675898821515, 2.447611173136, 4.325422955697, 0.000000000000, 4.325422955697],
              [1.019964859237, 1.269166956406, 1.675898821515, 2.447611173136, 4.325422955697, 0.000000000000]])
n_sites = 6
t0 = 2.4
delta = 0.08333333333333333
U = 8.0
""",
            "call": "triplet_pair_gram_spectrum(n_sites, t0, delta, V.copy(), U)",
            "gold_call": "_oracle_triplet_pair_gram_spectrum(n_sites, t0, delta, V.copy(), U)",
            "tol": 1e-08,
        },
        # --- Normal: the strongly correlated eight-atom chain ---
        {
            "setup": """import numpy as np
V = np.array([[0.000000000000, 4.325076507107, 2.514470076928, 1.695139627994, 1.278461106161, 1.028690941455, 0.855007419245, 0.736325181976],
              [4.325076507107, 0.000000000000, 4.037422412364, 2.514470076928, 1.640282944134, 1.278461106161, 1.007871777646, 0.855007419245],
              [2.514470076928, 4.037422412364, 0.000000000000, 4.325076507107, 2.514470076928, 1.695139627994, 1.278461106161, 1.028690941455],
              [1.695139627994, 2.514470076928, 4.325076507107, 0.000000000000, 4.037422412364, 2.514470076928, 1.640282944134, 1.278461106161],
              [1.278461106161, 1.640282944134, 2.514470076928, 4.037422412364, 0.000000000000, 4.325076507107, 2.514470076928, 1.695139627994],
              [1.028690941455, 1.278461106161, 1.695139627994, 2.514470076928, 4.325076507107, 0.000000000000, 4.037422412364, 2.514470076928],
              [0.855007419245, 1.007871777646, 1.278461106161, 1.640282944134, 2.514470076928, 4.037422412364, 0.000000000000, 4.325076507107],
              [0.736325181976, 0.855007419245, 1.028690941455, 1.278461106161, 1.695139627994, 2.514470076928, 4.325076507107, 0.000000000000]])
n_sites = 8
t0 = 2.5
delta = 0.1
U = 12.0
""",
            "call": "triplet_pair_gram_spectrum(n_sites, t0, delta, V.copy(), U)",
            "gold_call": "_oracle_triplet_pair_gram_spectrum(n_sites, t0, delta, V.copy(), U)",
            "tol": 1e-08,
        },
        # --- Boundary: the four-atom chain, a single product with unit overlap ---
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
            "call": "triplet_pair_gram_spectrum(n_sites, t0, delta, V.copy(), U)",
            "gold_call": "_oracle_triplet_pair_gram_spectrum(n_sites, t0, delta, V.copy(), U)",
            "tol": 1e-08,
        },
        # --- Edge: an eight-atom chain whose two terminal atoms interact ten percent more weakly with the others, so the right subchain's interaction block is the mirror image of the left one's ---
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
            "call": "triplet_pair_gram_spectrum(n_sites, t0, delta, V.copy(), U)",
            "gold_call": "_oracle_triplet_pair_gram_spectrum(n_sites, t0, delta, V.copy(), U)",
            "tol": 1e-08,
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
        triplet_pair_gram_spectrum(n_sites, t0, delta, V.copy(), U)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_oracle():
    try:
        _oracle_triplet_pair_gram_spectrum(n_sites, t0, delta, V.copy(), U)
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
