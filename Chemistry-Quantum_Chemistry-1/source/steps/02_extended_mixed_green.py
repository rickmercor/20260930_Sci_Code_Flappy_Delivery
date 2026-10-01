"""
Build the mixed one-body Green matrix in the ITHC enlarged basis.

The mixed one-particle Green matrix represents the transition density between a trial determinant and a walker. In a complex Slater representation, the placement of the transpose and complex conjugation determines the overlap orientation. The physical trial and walker are first mapped through the rectangular ITHC isometry before constructing this quantity.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def extended_mixed_green(
    trial: 'np.ndarray',
    walker: 'np.ndarray',
    isometry: 'np.ndarray',
) -> 'np.ndarray':
    """Evaluate the trial-walker mixed Green matrix in the ITHC space.

    Parameters
    ----------
    trial, walker
        Matching nonempty Slater matrices with shape
        ``(n_orbitals, n_electrons)``.
    isometry
        Real row-orthonormal array with shape
        ``(n_orbitals, n_auxiliary)``.

    Notes
    -----
    Set ``A_t = u.T @ trial`` and ``B_t = u.T @ walker``. With
    ``O = B_t.T @ A_t.conj()``, return
    ``A_t.conj() @ solve(O, B_t.T)``. The walker uses an ordinary transpose,
    not a Hermitian transpose.

    Returns
    -------
    np.ndarray
        Complex mixed Green matrix with shape
        ``(n_auxiliary, n_auxiliary)``.

    Raises
    ------
    ValueError
        If shapes, finiteness, row orthonormality, or overlap conditioning
        are invalid.
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_extended_mixed_green(
    trial: 'np.ndarray',
    walker: 'np.ndarray',
    isometry: 'np.ndarray',
) -> 'np.ndarray':
    import numpy as np

    A = np.asarray(trial, dtype=complex)
    B = np.asarray(walker, dtype=complex)
    u0 = np.asarray(isometry)
    if np.iscomplexobj(u0) and np.any(np.abs(np.imag(u0)) > 1e-13):
        raise ValueError("isometry must be real")
    u = np.asarray(np.real(u0), dtype=float)
    if A.ndim != 2 or B.shape != A.shape or min(A.shape) == 0:
        raise ValueError("trial and walker must be matching nonempty matrices")
    if u.ndim != 2 or u.shape[0] != A.shape[0]:
        raise ValueError("isometry has incompatible shape")
    if np.any(~np.isfinite(A)) or np.any(~np.isfinite(B)) or np.any(~np.isfinite(u)):
        raise ValueError("inputs must be finite")
    if not np.allclose(u @ u.T, np.eye(u.shape[0]), rtol=1e-11, atol=1e-11):
        raise ValueError("isometry rows must be orthonormal")

    At = u.T @ A
    Bt = u.T @ B
    overlap = Bt.T @ At.conj()
    if np.linalg.cond(overlap) > 1e12:
        raise ValueError("trial-walker overlap is singular")
    return At.conj() @ np.linalg.solve(overlap, Bt.T)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": "import numpy as np\nrng=np.random.default_rng(22); q,r=np.linalg.qr(rng.normal(size=(6,4))); u=(q*np.where(np.diag(r)>=0,1.,-1.)).T; A=np.linalg.qr(rng.normal(size=(4,2)))[0]; B=A+.13*rng.normal(size=(4,2))+.08j*rng.normal(size=(4,2))",
            "call": "extended_mixed_green(A,B,u)",
            "gold_call": "_oracle_extended_mixed_green(A,B,u)",
        },
        {
            "setup": "import numpy as np\nA=np.array([[1.],[0.],[0.]]); B=np.array([[1.+.2j],[.3],[.1j]]); u=np.eye(3)",
            "call": "extended_mixed_green(A,B,u)",
            "gold_call": "_oracle_extended_mixed_green(A,B,u)",
        },
        {
            "setup": "import numpy as np\nA=np.array([[1.],[0.]]); B=np.array([[2.],[1.j]]); u=np.array([[1.,0.,0.],[0.,2**-.5,2**-.5]])",
            "call": "extended_mixed_green(A,B,u)",
            "gold_call": "_oracle_extended_mixed_green(A,B,u)",
        },
        {
            "setup": "import numpy as np\nA=np.array([[1.,0.],[0.,1.],[0.,0.]],complex); B=np.array([[1.,.2j],[.1,1.],[-.2j,.3]],complex); u=np.array([[0.,1.,0.,0.],[0.,0.,0.,1.],[1.,0.,0.,0.]])",
            "call": "extended_mixed_green(A,B,u)",
            "gold_call": "_oracle_extended_mixed_green(A,B,u)",
        },
    ]
