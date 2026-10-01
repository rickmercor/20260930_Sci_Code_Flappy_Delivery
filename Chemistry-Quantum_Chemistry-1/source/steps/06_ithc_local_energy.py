"""
Contract physical and enlarged Green matrices into an ITHC local energy.

The mixed local energy combines the physical-basis one-electron contribution with the interaction contribution evaluated in the enlarged ITHC basis. Fermionic antisymmetry produces a direct occupation product minus an exchange contraction, with the diagonal pair contribution cancelling identically.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def ithc_local_energy(
    one_body: 'np.ndarray',
    kernel: 'np.ndarray',
    physical_green: 'np.ndarray',
    extended_green: 'np.ndarray',
) -> complex:
    """Evaluate the mixed local energy with the ITHC interaction.

    Parameters
    ----------
    one_body
        Hermitian physical-basis one-body matrix.
    kernel
        Real symmetric auxiliary-basis interaction kernel.
    physical_green
        Physical-basis mixed Green matrix.
    extended_green
        Auxiliary-basis mixed Green matrix.

    Notes
    -----
    Use
    ``E1 = sum_pq one_body[p,q]*physical_green[p,q]``.
    For ``n = diag(extended_green)``, use
    ``E2 = 0.5*sum_ab kernel[a,b]*(n[a]*n[b]
    - extended_green[a,b]*extended_green[b,a])``
    and return ``E1 + E2``.

    Returns
    -------
    complex
        Mixed local energy.

    Raises
    ------
    ValueError
        If shapes, finiteness, Hermiticity, or kernel symmetry are invalid.
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_ithc_local_energy(
    one_body: 'np.ndarray',
    kernel: 'np.ndarray',
    physical_green: 'np.ndarray',
    extended_green: 'np.ndarray',
) -> complex:
    import numpy as np

    h = np.asarray(one_body, dtype=complex)
    W = np.asarray(kernel, dtype=float)
    G = np.asarray(physical_green, dtype=complex)
    Gt = np.asarray(extended_green, dtype=complex)
    if h.ndim != 2 or h.shape[0] == 0 or h.shape[0] != h.shape[1]:
        raise ValueError("one_body must be a nonempty square matrix")
    if G.shape != h.shape:
        raise ValueError("physical_green has incompatible shape")
    if W.ndim != 2 or W.shape[0] == 0 or W.shape[0] != W.shape[1]:
        raise ValueError("kernel must be a nonempty square matrix")
    if Gt.shape != W.shape:
        raise ValueError("extended_green has incompatible shape")
    if any(np.any(~np.isfinite(v)) for v in (h, W, G, Gt)):
        raise ValueError("inputs must be finite")
    if not np.allclose(h, h.conj().T, rtol=1e-12, atol=1e-12):
        raise ValueError("one_body must be Hermitian")
    if not np.allclose(W, W.T, rtol=1e-12, atol=1e-12):
        raise ValueError("kernel must be symmetric")

    one_energy = np.einsum("pq,pq->", h, G)
    occupations = np.diag(Gt)
    pair_density = np.outer(occupations, occupations) - Gt * Gt.T
    off_diagonal = W.copy()
    np.fill_diagonal(off_diagonal, 0.0)
    two_energy = 0.5 * np.sum(off_diagonal * pair_density)
    return complex(one_energy + two_energy)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": "import numpy as np\nh=np.array([[-1.,.2],[.2,-.4]]); W=np.array([[.5,.3,.1],[.3,.7,-.2],[.1,-.2,.4]]); G=np.array([[.8+.1j,.2],[-.1j,.5-.1j]]); Gt=np.array([[.6+.1j,.1,.05j],[-.2,.5-.05j,.08],[.03j,-.04,.2]])",
            "call": "ithc_local_energy(h,W,G,Gt)",
            "gold_call": "_oracle_ithc_local_energy(h,W,G,Gt)",
        },
        {
            "setup": "import numpy as np\nh=np.array([[-.7]]); W=np.array([[5.,.2],[.2,3.]]); G=np.array([[1.]]); Gt=np.array([[.6,.1],[.2,.4]],complex)",
            "call": "ithc_local_energy(h,W,G,Gt)",
            "gold_call": "_oracle_ithc_local_energy(h,W,G,Gt)",
        },
        {
            "setup": "import numpy as np\nh=np.zeros((2,2)); W=np.diag([9.,4.,2.]); G=np.array([[.4,.1],[.2,.6]],complex); Gt=np.eye(3,dtype=complex)*.5",
            "call": "ithc_local_energy(h,W,G,Gt)",
            "gold_call": "_oracle_ithc_local_energy(h,W,G,Gt)",
        },
        {
            "setup": "import numpy as np\nh=np.array([[-1.,.15j],[-.15j,-.2]]); W=np.array([[.2,-.1],[-.1,.3]]); G=np.array([[.7,.2j],[-.1j,.4]]); Gt=np.array([[.55+.03j,.12],[-.08j,.45-.03j]])",
            "call": "ithc_local_energy(h,W,G,Gt)",
            "gold_call": "_oracle_ithc_local_energy(h,W,G,Gt)",
        },
    ]
