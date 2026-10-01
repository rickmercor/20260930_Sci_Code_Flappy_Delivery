"""
Assemble the Hatano-Nelson Hamiltonian matrix for a one-dimensional tight-binding

chain with non-reciprocal nearest-neighbour hoppings.



The first super-diagonal carries gamma * (1 + p) and the first sub-diagonal carries

gamma * (1 - p). Under periodic boundary conditions the bottom-left corner

H[N-1, 0] carries gamma * (1 + p) and the top-right corner H[0, N-1] carries

gamma * (1 - p); under open boundary conditions both corners stay zero.



Inputs

------

n_sites: int, number of chain sites N (>= 2)

gamma: float, hopping energy scale

p: float, non-reciprocity, |p| < 1

alpha_bc: float, boundary switch, 0.0 (OBC) or 1.0 (PBC)



Returns

-------

H: (n_sites, n_sites) complex ndarray



Raises

------

ValueError: if n_sites is below 2, gamma is not finite, |p| is not below 1, or alpha_bc is neither 0.0 nor 1.0

The Hatano-Nelson chain is the minimal lattice model of non-reciprocity. In the site

basis {|n>}, n = 1 .. N, it reads



    H = gamma * sum_{n=1}^{N-1} [ (1 + p) |n><n+1| + (1 - p) |n+1><n| ]

        + alpha_bc * gamma * [ (1 + p) |N><1| + (1 - p) |1><N| ].



The parameter p biases the two hopping directions against each other: the amplitude

for moving onto the left-hand neighbour is gamma(1 + p) and onto the right-hand

neighbour gamma(1 - p). Because those two weights differ, H is not equal to its own

conjugate transpose whenever p is non-zero, and the generated dynamics is non-unitary.

Setting p = 0 restores the ordinary Hermitian tight-binding chain.



At N = 2 the wrap bond coincides with the chain bond, so the two contributions

add and each off-diagonal entry is 2*gamma.



The boundary switch is not a cosmetic choice. With alpha_bc = 1 the chain closes into

a ring, the matrix becomes circulant, and its spectrum is a closed complex curve. With

alpha_bc = 0 the chain stays open, the matrix becomes tridiagonal Toeplitz, and the

spectrum collapses onto the real axis while the eigenvectors localise exponentially at

one end - the non-Hermitian skin effect. Every quantity computed downstream inherits

this distinction, because it is the location of the spectrum in the complex plane that

governs how the Chebyshev expansion behaves.

Returns
-------
np.ndarray, complex array of shape (n_sites, n_sites) holding the Hatano-Nelson Hamiltonian in the site basis
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def build_hatano_nelson(n_sites: int, gamma: float, p: float, alpha_bc: float) -> np.ndarray:
    '''Assemble the Hatano-Nelson Hamiltonian in the site basis.

    Parameters
    ----------
    n_sites : int
        Number of chain sites N, must be >= 2.
    gamma : float
        Hopping energy scale multiplying every hopping amplitude.
    p : float
        Non-reciprocity of the hoppings, must satisfy |p| < 1.
    alpha_bc : float
        Boundary-condition switch: 0.0 for open, 1.0 for periodic.

    Returns
    -------
    H : np.ndarray
        Complex array of shape (n_sites, n_sites) holding the Hamiltonian.

    Raises
    ------
    ValueError
        Raised if n_sites is below 2, gamma is not finite, |p| is not below 1, or alpha_bc is neither 0.0 nor 1.0.
    '''
    return H

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_build_hatano_nelson(n_sites: int, gamma: float, p: float, alpha_bc: float) -> np.ndarray:
    if not isinstance(n_sites, (int, np.integer)) or int(n_sites) < 2:
        raise ValueError("n_sites must be an integer >= 2")
    if not np.isfinite(float(gamma)):
        raise ValueError("gamma must be finite")
    if not np.isfinite(float(p)) or abs(float(p)) >= 1.0:
        raise ValueError("p must satisfy |p| < 1")
    if float(alpha_bc) not in (0.0, 1.0):
        raise ValueError("alpha_bc must be 0.0 (OBC) or 1.0 (PBC)")

    n = int(n_sites)
    g = float(gamma)
    pp = float(p)
    bc = float(alpha_bc)

    H = np.zeros((n, n), dtype=complex)
    idx = np.arange(n - 1)
    H[idx, idx + 1] = g * (1.0 + pp)
    H[idx + 1, idx] = g * (1.0 - pp)

    if bc != 0.0:
        # accumulate: on a two-site ring both wrap entries land on the chain bond
        H[n - 1, 0] += bc * g * (1.0 + pp)
        H[0, n - 1] += bc * g * (1.0 - pp)

    return H

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Normal: the task configuration, periodic boundary conditions ---
        {
            "setup": """import numpy as np
n_sites = 100
gamma = 0.85
p = 0.35
alpha_bc = 1.0
""",
            "call": "build_hatano_nelson(n_sites, gamma, p, alpha_bc)",
            "gold_call": "_oracle_build_hatano_nelson(n_sites, gamma, p, alpha_bc)",
        },
        # --- Normal: same parameters under open boundary conditions ---
        {
            "setup": """import numpy as np
n_sites = 100
gamma = 0.85
p = 0.35
alpha_bc = 0.0
""",
            "call": "build_hatano_nelson(n_sites, gamma, p, alpha_bc)",
            "gold_call": "_oracle_build_hatano_nelson(n_sites, gamma, p, alpha_bc)",
        },
        # --- Boundary: p = 0 recovers the Hermitian chain ---
        {
            "setup": """import numpy as np
n_sites = 8
gamma = 1.0
p = 0.0
alpha_bc = 1.0
""",
            "call": "build_hatano_nelson(n_sites, gamma, p, alpha_bc)",
            "gold_call": "_oracle_build_hatano_nelson(n_sites, gamma, p, alpha_bc)",
        },
        # --- Edge: smallest admissible chain, strong non-reciprocity ---
        {
            "setup": """import numpy as np
n_sites = 2
gamma = 0.5
p = 0.9
alpha_bc = 1.0
""",
            "call": "build_hatano_nelson(n_sites, gamma, p, alpha_bc)",
            "gold_call": "_oracle_build_hatano_nelson(n_sites, gamma, p, alpha_bc)",
        },
        # --- Invalid: |p| >= 1 ---
        {
            "setup": """import numpy as np
def run_model():
    _fn = build_hatano_nelson
    try:
        _fn(10, 1.0, 1.0, 1.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_oracle():
    _fn = _oracle_build_hatano_nelson
    try:
        _fn(10, 1.0, 1.0, 1.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_oracle()",
        },
        # --- Invalid: n_sites below 2 ---
        {
            "setup": """import numpy as np
def run_model():
    _fn = build_hatano_nelson
    try:
        _fn(1, 1.0, 0.2, 1.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_oracle():
    _fn = _oracle_build_hatano_nelson
    try:
        _fn(1, 1.0, 0.2, 1.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_oracle()",
        },
        # --- Invalid: alpha_bc outside {0, 1} ---
        {
            "setup": """import numpy as np
def run_model():
    _fn = build_hatano_nelson
    try:
        _fn(10, 1.0, 0.2, 0.5)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_oracle():
    _fn = _oracle_build_hatano_nelson
    try:
        _fn(10, 1.0, 0.2, 0.5)
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
