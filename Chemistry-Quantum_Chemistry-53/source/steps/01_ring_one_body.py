"""
The model is a ring of L localized orbitals (lattice sites) with nearest-neighbour hopping of amplitude t and periodic boundary conditions, so that site L-1 is bonded to site 0.

The model is a ring of L localized orbitals (lattice sites) with nearest-neighbour hopping of amplitude t and periodic boundary conditions, so that site L-1 is bonded to site 0. Every site i carries an external on-site potential v_ext[i]; a non-uniform choice of these potentials is what makes the density profile of the ring non-trivial. The one-electron part of the Hamiltonian reads

  h = -t sum_{i, sigma} (c+_{i sigma} c_{i+1 sigma} + h.c.) + sum_i v_ext[i] n_i,

with n_i = sum_sigma c+_{i sigma} c_{i sigma} and i + 1 taken modulo L. In the site basis this is the real symmetric L x L matrix with diagonal entries h[i, i] = v_ext[i] and off-diagonal entries h[i, i+1] = h[i+1, i] = -t (indices modulo L), all other entries being zero. The on-site repulsion U n_{i up} n_{i down} is not part of this matrix; it enters through the mean-field reference and the embedding clusters in the later steps.

Rings with fewer than three sites are not accepted: for L = 2 the two bonds of the periodic chain coincide and the hopping would be counted twice, so the model is not defined.

Returns
-------
np.ndarray of float with shape (L, L): the periodic ring one-electron matrix with v_ext on the diagonal and -t on the nearest-neighbour bonds.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def ring_one_body(L, t, v_ext):
    '''One-electron Hamiltonian matrix of a periodic non-uniform ring.

    Parameters
    ----------
    L : int
        Number of sites, >= 3.
    t : float
        Nearest-neighbour hopping amplitude (the matrix element is -t).
    v_ext : array_like of float, shape (L,)
        External on-site potential of every site.

    Returns
    -------
    h : np.ndarray of float, shape (L, L)
        Real symmetric one-electron matrix in the site basis: h[i, i] =
        v_ext[i], h[i, (i + 1) % L] = h[(i + 1) % L, i] = -t, zero elsewhere.
        Raises ValueError if L < 3, if v_ext does not have L entries, or if
        any input is not finite.
    '''
    return np.zeros((L, L), dtype=float)

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_ring_one_body(L, t, v_ext):
    if isinstance(L, bool) or not isinstance(L, (int, np.integer)):
        raise ValueError("L must be an integer")
    if L < 3:
        raise ValueError("the ring must have at least three sites")
    L = int(L)
    t = float(t)
    if not np.isfinite(t):
        raise ValueError("t must be finite")
    v = np.asarray(v_ext, dtype=float)
    if v.ndim != 1 or v.shape[0] != L:
        raise ValueError("v_ext must be a one-dimensional array with L entries")
    if not np.all(np.isfinite(v)):
        raise ValueError("v_ext must be finite")
    h = np.diag(v)
    for i in range(L):
        j = (i + 1) % L
        h[i, j] -= t
        h[j, i] -= t
    return h

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Normal: the task ring (six sites, t = 1, non-uniform potential) ---
        {
            "setup": "import numpy as np\nv = [-1.0, 2.0, -2.0, 3.0, -3.0, 1.0]\n",
            "call": "ring_one_body(6, 1.0, v)",
            "gold_call": "_oracle_ring_one_body(6, 1.0, v)",
        },
        # --- Normal: four sites with a weaker hopping and a different potential ---
        {
            "setup": "import numpy as np\nv = np.array([0.5, -0.5, 1.5, -1.5])\n",
            "call": "ring_one_body(4, 0.5, v)",
            "gold_call": "_oracle_ring_one_body(4, 0.5, v)",
        },
        # --- Boundary: the smallest admissible ring (three sites, all bonds distinct) ---
        {
            "setup": "import numpy as np\nv = [0.0, 1.0, -1.0]\n",
            "call": "ring_one_body(3, 1.0, v)",
            "gold_call": "_oracle_ring_one_body(3, 1.0, v)",
        },
        # --- Edge: zero hopping leaves only the on-site potentials ---
        {
            "setup": "import numpy as np\nv = [2.0, -2.0, 3.0, -3.0, 1.0]\n",
            "call": "ring_one_body(5, 0.0, v)",
            "gold_call": "_oracle_ring_one_body(5, 0.0, v)",
        },
        # --- Invalid: a two-site ring (the two periodic bonds coincide) ---
        {
            "setup": """import numpy as np
def run_model():
    try:
        ring_one_body(2, 1.0, [0.0, 0.0])
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_ring_one_body(2, 1.0, [0.0, 0.0])
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Invalid: potential vector with the wrong length ---
        {
            "setup": """import numpy as np
def run_model():
    try:
        ring_one_body(6, 1.0, [-1.0, 2.0, -2.0, 3.0])
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_ring_one_body(6, 1.0, [-1.0, 2.0, -2.0, 3.0])
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
    ]
