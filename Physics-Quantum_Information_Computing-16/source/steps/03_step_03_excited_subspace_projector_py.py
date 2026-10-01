"""
Step 3: Projector onto the positive-energy subspace and kernel dimension.

P_perp projects onto the eigenvectors of a positive semidefinite operator with eigenvalues above the zero threshold, and the kernel dimension is returned alongside it. Applied to the (n-1)-site open chain, this gives the projector used in the certifiable formulation of the LTI SDP. At level n = 3, the two-site chain is h itself, so P_2_perp is the support projector of h.

Returns
-------
# tuple, (P_perp, dim_ker) with P_perp an ndarray projector and dim_ker a native Python int
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

# =============================================================================
# FUNCTION SIGNATURE
# =============================================================================
def excited_subspace_projector(H: np.ndarray, tol: float = 1e-10) -> tuple[np.ndarray, int]:
    '''Return the projector onto eigenvectors of H with eigenvalue > tol, and the kernel dimension.

    Parameters
    ----------
    H : np.ndarray
        Hermitian positive-semidefinite square matrix.
    tol : float
        Positive eigenvalue threshold.

    Returns
    -------
    result : tuple[np.ndarray, int]
        (P_perp, dim_ker): P_perp an orthogonal projector with the shape of H,
        dim_ker the number of eigenvalues <= tol as a native int.

    Raises
    ------
    ValueError
        If H is not a nonempty square Hermitian matrix, tol is not finite and > 0,
        or H has an eigenvalue below -tol.
    '''
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

# =============================================================================
# ORACLE SOLUTION
# =============================================================================
import numpy as np

def _oracle_excited_subspace_projector(
    H: np.ndarray,
    tol: float = 1e-10,
) -> tuple[np.ndarray, int]:
    import numpy as np

    H = np.asarray(H, dtype=complex)
    tol = float(tol)

    if H.ndim != 2 or H.shape[0] != H.shape[1] or H.shape[0] < 1:
        raise ValueError("H must be a nonempty square matrix")

    if not np.isfinite(tol) or tol <= 0.0:
        raise ValueError("tol must be finite and > 0")

    if not np.allclose(H, H.conj().T, rtol=0.0, atol=1e-10):
        raise ValueError("H must be Hermitian")

    w, V = np.linalg.eigh(0.5 * (H + H.conj().T))

    if w[0] < -tol:
        raise ValueError("H must be positive semidefinite within tol")

    U = V[:, w > tol]
    P = U @ U.conj().T if U.shape[1] else np.zeros_like(H)

    return 0.5 * (P + P.conj().T), int((w <= tol).sum())

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {"setup": "import numpy as np\nh,_=_oracle_build_local_interaction(0.347,0.783)\npack=lambda t: np.concatenate([np.concatenate([np.array([np.ndim(x),*np.shape(x)],dtype=complex), np.asarray(x,dtype=complex).ravel()]) for x in t])",
         "call": "pack(excited_subspace_projector(h))",
         "gold_call": "pack(_oracle_excited_subspace_projector(h))"},
        # boundary: 3-site open chain (27x27, kernel dimension 3)
        {"setup": "import numpy as np\nh,_=_oracle_build_local_interaction(0.347,0.783); I3=np.eye(3); H=np.kron(h,I3)+np.kron(I3,h)\npack=lambda t: np.concatenate([np.concatenate([np.array([np.ndim(x),*np.shape(x)],dtype=complex), np.asarray(x,dtype=complex).ravel()]) for x in t])",
         "call": "pack(excited_subspace_projector(H))",
         "gold_call": "pack(_oracle_excited_subspace_projector(H))"},
        # edge: zero matrix -> zero projector, full kernel
        {"setup": "import numpy as np\nH=np.zeros((4,4),dtype=complex)\npack=lambda t: np.concatenate([np.concatenate([np.array([np.ndim(x),*np.shape(x)],dtype=complex), np.asarray(x,dtype=complex).ravel()]) for x in t])",
         "call": "pack(excited_subspace_projector(H))",
         "gold_call": "pack(_oracle_excited_subspace_projector(H))"},
    ]
