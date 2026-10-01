"""
Compute the random-phase-approximation correlation energy of a Hartree-Fock-stable determinant
from its orbital Hessian blocks A and B.

The RPA excitation energies omega are the eigenvalues of the non-Hermitian matrix [[A, B], [-B*,
-A*]]; when [[A, B], [B*, A*]] is positive definite they come in real pairs +omega, -omega. Take
the positive branch, the n eigenvalues with the largest real part.

The correlation energy uses the ring coupled-cluster-doubles normalisation, E_c = (1/4) [sum
over the positive branch of omega - Tr A], with the prefactor 1/4 rather than 1/2. Return it as
a float. The comparison needs about 1e-10 relative accuracy.

Returns
-------
Ec : float -- RPA correlation energy.
"""

import numpy as np
from scipy.linalg import expm

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def rpa_correlation(A: "np.ndarray", B: "np.ndarray") -> float:
    '''Compute the random-phase-approximation correlation energy of a Hartree-Fock-stable
    determinant from its orbital Hessian blocks A and B.

    Parameters
    ----------
    A : np.ndarray
        Hermitian orbital Hessian block of shape (n, n).
    B : np.ndarray
        Complex symmetric orbital Hessian block of shape (n, n); [[A, B], [B*, A*]] is positive definite.

    Returns
    -------
    Ec : float
        RPA correlation energy.
    '''
    return Ec

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_rpa_correlation(A: "np.ndarray", B: "np.ndarray") -> float:
    n = len(A)
    M = np.block([[A, B], [-np.conj(B), -np.conj(A)]])
    w = np.sort(np.linalg.eigvals(M).real)[n:]
    return float(0.25 * (w.sum() - np.trace(A).real))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # normal: complex six-dimensional blocks
        {
            "setup": 'import numpy as np\nn = 6\nk = np.arange(n)\nA = np.diag(3.0 + 0.5 * k) + 0.2 * np.cos(np.outer(k, k) + 0.3) + 0.15j * np.sin(np.subtract.outer(k, k))\nB = 0.25 * (np.cos(np.add.outer(k, k)) + 1j * np.sin(0.7 * np.add.outer(k, k) + 0.2))\nA_ref = A.copy()\nB_ref = B.copy()',
            "call": 'rpa_correlation(A, B)',
            "gold_call": '_oracle_rpa_correlation(A_ref, B_ref)',
        },
        # normal: smaller gaps and stronger coupling B
        {
            "setup": 'import numpy as np\nn = 5\nk = np.arange(n)\nA = np.diag(1.6 + 0.5 * k) + 0.2 * np.cos(np.outer(k, k) + 0.3) + 0.15j * np.sin(np.subtract.outer(k, k))\nB = 0.3 * (np.cos(np.add.outer(k, k)) + 1j * np.sin(0.7 * np.add.outer(k, k) + 0.2))\nA_ref = A.copy()\nB_ref = B.copy()',
            "call": 'rpa_correlation(A, B)',
            "gold_call": '_oracle_rpa_correlation(A_ref, B_ref)',
        },
        # boundary: weak coupling B, where E_c approaches zero quadratically
        {
            "setup": 'import numpy as np\nn = 4\nk = np.arange(n)\nA = np.diag(2.0 + 0.5 * k) + 0.2 * np.cos(np.outer(k, k) + 0.3) + 0.15j * np.sin(np.subtract.outer(k, k))\nB = 0.02 * (np.cos(np.add.outer(k, k)) + 1j * np.sin(0.7 * np.add.outer(k, k) + 0.2))\nA_ref = A.copy()\nB_ref = B.copy()',
            "call": 'rpa_correlation(A, B)',
            "gold_call": '_oracle_rpa_correlation(A_ref, B_ref)',
        },
        # edge: one-dimensional blocks
        {
            "setup": 'import numpy as np\nA = np.array([[1.3]])\nB = np.array([[0.4 - 0.5j]])\nA_ref = A.copy()\nB_ref = B.copy()',
            "call": 'rpa_correlation(A, B)',
            "gold_call": '_oracle_rpa_correlation(A_ref, B_ref)',
        },
        # normal: blocks of the stable Hartree-Fock solution of the four-site Heisenberg chain
        {
            "setup": 'import numpy as np\nA = np.array([[1.556004871874685, -4.024558464266192e-16, -2.220446049250313e-16, -0.2241858331693354], [-3.191891195797325e-16, 0.8318190387053597, -0.5, 0], [-2.220446049250313e-16, -0.5, 1.831819038705339, 5.273559366969494e-16], [-0.2241858331693354, -2.775557561562891e-17, 5.134781488891349e-16, 1.556004871874685]])\nB = np.array([[-3.456456776012108e-18, -6.938893903907228e-18, 1.023609380108312e-17, 0.2758141668306646], [6.938893903907228e-18, 1.369255057709524e-17, -0.2758141668306646, 1.369255057709522e-17], [-1.023609380108312e-17, -0.2758141668306647, -1.940888636146434e-17, -3.469446951953614e-17], [0.2758141668306647, -1.023609380108311e-17, -2.775557561562891e-17, 3.456456776012117e-18]])\nA_ref = A.copy()\nB_ref = B.copy()',
            "call": 'rpa_correlation(A, B)',
            "gold_call": '_oracle_rpa_correlation(A_ref, B_ref)',
        },
    ]
