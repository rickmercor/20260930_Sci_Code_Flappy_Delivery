"""
Obtain the lowest exciton eigenenergies while retaining physical degeneracies.

Obtain the lowest exciton eigenenergies while retaining physical degeneracies.

The Hermitian vertical exciton Hamiltonian defines a real spectrum. Count multiplicities when ordering the lowest levels.

Returns
-------
return result  # real ndarray, shape (count,)
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def exciton_levels(hamiltonian: np.ndarray, count: int = 3) -> np.ndarray:
    """Parameters
    ----------
    hamiltonian : finite Hermitian complex ndarray, shape (D,D)
        Exciton Hamiltonian in eV, D>=1.
    count : int, 1<=count<=D
        Number of lowest levels requested, including multiplicities.
    Returns
    -------
    real ndarray, shape (count,)
        Exciton energies in ascending order, in eV.
    Raises
    ------
    ValueError : malformed/nonfinite/non-Hermitian matrix or invalid count."""
    return np.zeros((count,))

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
from scipy.linalg import eigh

def _oracle_exciton_levels(hamiltonian, count=3):
    h = np.asarray(hamiltonian, dtype=complex)
    if h.ndim != 2 or h.shape[0] != h.shape[1] or not np.isfinite(h).all() or not isinstance(count,(int,np.integer)) or not 1 <= count <= len(h) or not np.allclose(h,h.conj().T,atol=1e-10):
        raise ValueError('finite Hermitian matrix and valid count required')
    return eigh(h,
                subset_by_index=[0, count-1], eigvals_only=True)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [{'setup': 'import numpy as np\nh=np.diag([4.,1.,3.,2.])\n', 'call': 'exciton_levels(h,3)', 'gold_call': '_oracle_exciton_levels(h,3)'}, {'setup': 'import numpy as np\nh=np.diag([1.,1.,2.,3.])\n', 'call': 'exciton_levels(h,2)', 'gold_call': '_oracle_exciton_levels(h,2)'}, {'setup': 'import numpy as np\nh=np.array([[1.,.3j],[-.3j,2.]])\n', 'call': 'exciton_levels(h,2)', 'gold_call': '_oracle_exciton_levels(h,2)'}, {'setup': 'import numpy as np\nh=np.array([[.1,-1.],[-1.,.2]])\n', 'call': 'exciton_levels(h,1)', 'gold_call': '_oracle_exciton_levels(h,1)'}, {'setup': 'import numpy as np\nh=np.array([[1.,1e-6],[1e-6,1.+1e-7]])\n', 'call': 'exciton_levels(h,2)', 'gold_call': '_oracle_exciton_levels(h,2)'}, {'setup': 'import numpy as np\nh=np.array([[1.,1.],[0.,2.]])\ndef _exception_code(fn):\n    try:\n        fn()\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\n', 'call': '_exception_code(lambda: exciton_levels(h,1))', 'gold_call': '_exception_code(lambda: _oracle_exciton_levels(h,1))'}]
