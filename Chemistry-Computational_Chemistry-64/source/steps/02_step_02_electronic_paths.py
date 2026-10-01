"""
Enumerate closed electronic paths supported by the complete dipole operator.

Insert electronic resolutions of the identity between dipoles. A path survives if every transition is supported by at least one term of the dipole operator; nuclear-coordinate dependence can activate a transition whose Condon matrix element vanishes

Returns
-------
return result
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def electronic_paths(mu0: np.ndarray, mu1: np.ndarray, vertices: int) -> np.ndarray:
    """Enumerate closed electronic paths supported by the complete dipole operator.
    
    Parameters
    ----------
    mu0 : complex ndarray, shape (N,N)
        Coordinate-independent dipole matrix.
    mu1 : complex ndarray, shape (F,N,N)
        Coefficients of a_f+a_f.dagger, without a 1/sqrt(2) factor.
    vertices : int
        Number of dipole insertions, 2 <= vertices <= 8; 1 <= N <= 4, 1 <= F <= 24.
    
    Returns
    -------
    result : integer ndarray, shape (P,vertices+1)
        Paths (p_0,...,p_vertices) with p_0=p_vertices=0, in lexicographic order.
        At every edge (a,b), at least one of mu0[a,b] or mu1[:,a,b] must be exactly
        nonzero. A zero Condon element does not remove an HT-supported edge.
        If P=0, return shape (0,vertices+1). No numerical cutoff is applied.
        Path indices follow the left-to-right operator product. Inputs are valid.
        Numeric outputs are tested with absolute and relative tolerances
        of 1e-9; integer outputs must be exact.
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import itertools
import numpy as np


def _oracle_electronic_paths(mu0: np.ndarray, mu1: np.ndarray, vertices: int) -> np.ndarray:
    mu0 = np.asarray(mu0)
    mu1 = np.asarray(mu1)
    support = (mu0 != 0) | np.any(mu1 != 0, axis=0)
    rows = []
    for inner in itertools.product(range(mu0.shape[0]), repeat=vertices-1):
        path = (0,) + inner + (0,)
        if all(support[path[k], path[k+1]] for k in range(vertices)):
            rows.append(path)
    return np.asarray(rows, dtype=int).reshape(-1, vertices+1)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [{'setup': 'from copy import deepcopy\n'
               'import numpy as np\n'
               'mu0=np.array([[0,1],[1,0.]])\n'
               'mu1=np.zeros((1,2,2))',
      'call': 'electronic_paths(*deepcopy((mu0, mu1, 4)))',
      'gold_call': '_oracle_electronic_paths(*deepcopy((mu0, mu1, 4)))',
      'name': 'legacy_1',
      'tol': 1e-09},
     {'setup': 'from copy import deepcopy\n'
               'import numpy as np\n'
               'mu0=np.zeros((3,3))\n'
               'mu1=np.array([[[0,1,0],[1,0,2],[0,2,0.]]])',
      'call': 'electronic_paths(*deepcopy((mu0, mu1, 6)))',
      'gold_call': '_oracle_electronic_paths(*deepcopy((mu0, mu1, 6)))',
      'name': 'legacy_2',
      'tol': 1e-09},
     {'setup': 'from copy import deepcopy\n'
               'import numpy as np\n'
               'mu0=np.zeros((2,2))\n'
               'mu1=np.zeros((2,2,2))',
      'call': 'electronic_paths(*deepcopy((mu0, mu1, 3)))',
      'gold_call': '_oracle_electronic_paths(*deepcopy((mu0, mu1, 3)))',
      'name': 'legacy_3',
      'tol': 1e-09},
     {'setup': 'from copy import deepcopy\n'
               'import numpy as np\n'
               'energies = np.array([0.0, 1.23, 2.17], dtype=float)\n'
               'omega = np.array([0.73, 1.11], dtype=float)\n'
               'displacements = np.array([[0.0, 0.0], [0.42, -0.31], [-0.36, 0.57]], dtype=float)\n'
               'mu0 = np.array([[0.12, 0.83, -0.21], [0.83, -0.17, 0.64], [-0.21, 0.64, 0.09]], '
               'dtype=float)\n'
               'mu1 = np.array([[[0.04, 0.19, 0.11], [0.19, -0.06, -0.14], [0.11, -0.14, 0.03]], '
               '[[-0.02, -0.13, 0.08], [-0.13, 0.05, 0.16], [0.08, 0.16, -0.04]]], dtype=float)\n'
               'waits = np.array([0.37, 0.82, 0.53, 1.14, 0.61], dtype=float)',
      'call': 'electronic_paths(*deepcopy((mu0, mu1, 6)))',
      'gold_call': '_oracle_electronic_paths(*deepcopy((mu0, mu1, 6)))',
      'name': 'legacy_4',
      'tol': 1e-09},
     {'name': 'support_only_in_mode_24',
      'setup': 'from copy import deepcopy\n'
               'import numpy as np\n'
               'mu0=np.zeros((2,2))\n'
               'mu1=np.zeros((24,2,2))\n'
               'mu1[23,0,1]=mu1[23,1,0]=0.4',
      'call': 'electronic_paths(*deepcopy((mu0, mu1, 6)))',
      'gold_call': '_oracle_electronic_paths(*deepcopy((mu0, mu1, 6)))',
      'tol': 1e-12}]
