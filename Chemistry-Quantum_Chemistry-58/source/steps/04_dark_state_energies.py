"""
Energies of the ground state, the lowest triplet, the covalent dark state 2(1)Ag and the lowest quintet of a chain.

The low-lying spectrum of a half-filled polyene chain is organised by three labels: the total spin, the parity under the spatial inversion that maps atom i to atom N + 1 - i, and the alternancy eigenvalue defined in the previous step. The ground state is an inversion-even singlet; the lowest triplet is inversion-odd; the dark state 2(1)Ag is the lowest covalent inversion-even singlet above the ground state, covalent meaning that its alternancy eigenvalue equals that of the ground state, and ionic inversion-even singlets may lie below it. The lowest quintet (S = 2) is the state used as the proxy for two electronically uncorrelated triplets; its energy can be taken from any of its Sz components.

The singlet and triplet states are identified in the Sz = 0 sector (n_up = n_down = N/2) through their spin and symmetry labels, since that sector contains a component of every spin multiplet. Total spin is measured by the expectation value of S^2, S(S+1).

Returns
-------
numpy.ndarray of shape (4,): energies in eV of the ground state, the lowest triplet, the dark state 2(1)Ag and the lowest quintet
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


def dark_state_energies(n_sites: int, t0: float, delta: float, V: np.ndarray, U: float) -> np.ndarray:
    '''Ground, lowest-triplet, dark-state and lowest-quintet energies of the chain.

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
    energies : numpy.ndarray
        Shape (4,): E(ground state), E(lowest triplet), E(2(1)Ag) with 2(1)Ag the lowest
        inversion-even singlet above the ground state whose alternancy eigenvalue equals
        that of the ground state, and E(lowest quintet), all in eV.

    Raises
    ------
    ValueError
        If n_sites is not an even integer >= 4, V is not a symmetric (n_sites, n_sites)
        array, t0 or U is not positive, or |delta| >= 1.
    '''
    return energies

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


def _s_minus(psi, sec):
    """S- psi, mapping (n_up, n_down) to (n_up-1, n_down+1)."""
    tgt = _sector(sec.n, sec.n_up - 1, sec.n_down + 1)
    P = psi.reshape(sec.nu, sec.nd)
    out = np.zeros((tgt.nu, tgt.nd))
    cross = (-1) ** tgt.n_up  # c_{i up} acts on the up block directly; c+_{i dn} passes the remaining up operators
    for i in range(sec.n):
        fu, tu, su = _site_maps(sec, tgt, 0, i, False)
        fd, td, sd = _site_maps(sec, tgt, 1, i, True)
        if fu.size == 0 or fd.size == 0:
            continue
        out[np.ix_(tu, td)] += cross * (su[:, None] * sd[None, :]) * P[np.ix_(fu, fd)]
    return out.ravel(), tgt


def _dark_states(n_sites, t0, delta, V, U):
    """Ground state, 2^1Ag- (lowest inversion-even excited singlet sharing the ground state alternancy eigenvalue) in the Sz=0 sector, and the lowest quintet via the Sz=2 sector."""
    sec = _sector(n_sites, n_sites // 2, n_sites // 2)
    k = min(sec.dim, 12)
    while True:
        sec, w, v, lab = _labelled_spectrum(n_sites, t0, delta, V, U, k)
        gs_sign = np.sign(lab[0, 3])
        sing_ag = [i for i in range(len(w)) if abs(lab[i, 1]) < 1e-6 and lab[i, 2] > 0.999 and np.sign(lab[i, 3]) == gs_sign]
        trip = [i for i in range(len(w)) if abs(lab[i, 1] - 2.0) < 1e-6]
        if len(sing_ag) >= 2 and trip:
            break
        if k >= sec.dim:
            raise ValueError("the Sz=0 sector does not contain a covalent inversion-even excited singlet")
        k = min(sec.dim, 2 * k)
    q_sec = _sector(n_sites, n_sites // 2 + 2, n_sites // 2 - 2)
    Hq = _hamiltonian(q_sec, t0, delta, V, U)
    wq, vq = _lowest(Hq, 1, q_sec.dim)
    return dict(sec=sec, w=w, v=v, lab=lab, gs=sing_ag[0], ag2=sing_ag[1], t1=trip[0], q_sec=q_sec, eq=float(wq[0]), vq=vq[:, 0])


def _oracle_dark_state_energies(n_sites: int, t0: float, delta: float, V: np.ndarray, U: float) -> np.ndarray:
    Vm = _validate_model(n_sites, t0, delta, V, U)
    if n_sites < 4:
        raise ValueError("n_sites must be at least 4 for a quintet state")
    d = _dark_states(n_sites, t0, delta, Vm, U)
    return np.array([d["w"][d["gs"]], d["w"][d["t1"]], d["w"][d["ag2"]], d["eq"]], dtype=float)

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
            "call": "dark_state_energies(n_sites, t0, delta, V.copy(), U)",
            "gold_call": "_oracle_dark_state_energies(n_sites, t0, delta, V.copy(), U)",
            "tol": 1e-08,
        },
        # --- Normal: the six-atom benchmark chain ---
        {
            "setup": """import numpy as np
V = np.array([[0.000000000000, 3.028724843957, 2.163107408042, 1.574142730777, 1.224092567900, 0.999724120164],
              [3.028724843957, 0.000000000000, 2.924747953395, 2.163107408042, 1.529921801556, 1.224092567900],
              [2.163107408042, 2.924747953395, 0.000000000000, 3.028724843957, 2.163107408042, 1.574142730777],
              [1.574142730777, 2.163107408042, 3.028724843957, 0.000000000000, 2.924747953395, 2.163107408042],
              [1.224092567900, 1.529921801556, 2.163107408042, 2.924747953395, 0.000000000000, 3.028724843957],
              [0.999724120164, 1.224092567900, 1.574142730777, 2.163107408042, 3.028724843957, 0.000000000000]])
n_sites = 6
t0 = 2.5
delta = 0.1
U = 4.0
""",
            "call": "dark_state_energies(n_sites, t0, delta, V.copy(), U)",
            "gold_call": "_oracle_dark_state_energies(n_sites, t0, delta, V.copy(), U)",
            "tol": 1e-08,
        },
        # --- Edge: the weakly screened eight-atom chain, where an ionic inversion-even singlet lies below the dark state ---
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
            "call": "dark_state_energies(n_sites, t0, delta, V.copy(), U)",
            "gold_call": "_oracle_dark_state_energies(n_sites, t0, delta, V.copy(), U)",
            "tol": 1e-08,
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
            "call": "dark_state_energies(n_sites, t0, delta, V.copy(), U)",
            "gold_call": "_oracle_dark_state_energies(n_sites, t0, delta, V.copy(), U)",
            "tol": 1e-08,
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
            "call": "dark_state_energies(n_sites, t0, delta, V.copy(), U)",
            "gold_call": "_oracle_dark_state_energies(n_sites, t0, delta, V.copy(), U)",
            "tol": 1e-08,
        },
        # --- Boundary: the four-atom chain close to the noninteracting limit ---
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
            "call": "dark_state_energies(n_sites, t0, delta, V.copy(), U)",
            "gold_call": "_oracle_dark_state_energies(n_sites, t0, delta, V.copy(), U)",
            "tol": 1e-08,
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
        dark_state_energies(n_sites, t0, delta, V.copy(), U)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_oracle():
    try:
        _oracle_dark_state_energies(n_sites, t0, delta, V.copy(), U)
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
