"""
Recover a physical ERI tensor from an isometric THC representation.

An isometric tensor-hypercontraction representation expresses the physical four-index electron-repulsion tensor through a rectangular row-orthonormal map and an auxiliary interaction kernel. This reconstruction certifies the supplied ITHC representation before any walker propagation is performed.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def reconstruct_ithc_eri(
    isometry: 'np.ndarray',
    kernel: 'np.ndarray',
) -> 'np.ndarray':
    """Reconstruct the physical four-index ERI represented by ITHC.

    Parameters
    ----------
    isometry
        Real row-orthonormal array with shape
        ``(n_orbitals, n_auxiliary)``.
    kernel
        Real symmetric positive-semidefinite array with shape
        ``(n_auxiliary, n_auxiliary)``.

    Notes
    -----
    Return the tensor in ``(p, q, r, s)`` order using
    ``V[p,q,r,s] = sum_ab u[p,a] u[q,a] W[a,b] u[r,b] u[s,b]``.

    Returns
    -------
    np.ndarray
        Reconstructed ERI tensor with shape ``(n, n, n, n)``.

    Raises
    ------
    ValueError
        If the inputs have incompatible shapes, are nonfinite, or violate the
        stated isometry and kernel conditions.
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_reconstruct_ithc_eri(
    isometry: 'np.ndarray',
    kernel: 'np.ndarray',
) -> 'np.ndarray':
    import numpy as np

    u = np.asarray(isometry)
    W = np.asarray(kernel)
    if np.iscomplexobj(u) and np.any(np.abs(np.imag(u)) > 1e-13):
        raise ValueError("isometry must be real")
    u = np.asarray(np.real(u), dtype=float)
    W = np.asarray(W, dtype=float)
    if u.ndim != 2 or min(u.shape) == 0 or u.shape[0] > u.shape[1]:
        raise ValueError("isometry must have shape (n, n_aux) with n_aux >= n")
    if W.shape != (u.shape[1], u.shape[1]):
        raise ValueError("kernel has incompatible shape")
    if np.any(~np.isfinite(u)) or np.any(~np.isfinite(W)):
        raise ValueError("inputs must be finite")
    if not np.allclose(u @ u.T, np.eye(u.shape[0]), rtol=1e-11, atol=1e-11):
        raise ValueError("isometry rows must be orthonormal")
    if not np.allclose(W, W.T, rtol=1e-12, atol=1e-12):
        raise ValueError("kernel must be symmetric")
    if np.linalg.eigvalsh(W)[0] < -1e-11:
        raise ValueError("kernel must be positive semidefinite")
    return np.einsum(
        "pa,qa,ab,rb,sb->pqrs", u, u, W, u, u, optimize=True
    )

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": "import numpy as np\nrng=np.random.default_rng(11); q,r=np.linalg.qr(rng.normal(size=(6,4))); s=np.where(np.diag(r)>=0,1.,-1.); u=(q*s).T; c=rng.normal(size=(5,6)); W=c.T@c/5",
            "call": "reconstruct_ithc_eri(u,W)",
            "gold_call": "_oracle_reconstruct_ithc_eri(u,W)",
        },
        {
            "setup": "import numpy as np\nu=np.eye(3); W=np.array([[1.,.2,.1],[.2,.8,.05],[.1,.05,.6]])",
            "call": "reconstruct_ithc_eri(u,W)",
            "gold_call": "_oracle_reconstruct_ithc_eri(u,W)",
        },
        {
            "setup": "import numpy as np\nu=np.array([[.5,-.5,.5,.5]]); c=np.array([[.2,-.1,.3,.4],[.5,.2,-.2,.1]]); W=c.T@c",
            "call": "reconstruct_ithc_eri(u,W)",
            "gold_call": "_oracle_reconstruct_ithc_eri(u,W)",
        },
        {
            "setup": "import numpy as np\nu=np.array([[1.,0.,0.,0.],[0.,0.,1.,0.]]); W=np.diag([.7,0.,.4,0.])",
            "call": "reconstruct_ithc_eri(u,W)",
            "gold_call": "_oracle_reconstruct_ithc_eri(u,W)",
        },
    ]
