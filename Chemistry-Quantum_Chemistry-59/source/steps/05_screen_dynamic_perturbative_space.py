"""
Screen and canonically order the dynamic perturbative determinant space.

Dynamic heat-bath screening constructs an external perturbative space using both Hamiltonian couplings and the normalized amplitudes of the selected variational configurations. An external determinant is retained when at least one amplitude-weighted coupling reaches the inclusive threshold. Retained determinants are then ordered canonically by their alpha and beta bitmasks.

Returns
-------
A one-dimensional integer array containing the retained zero-based archive indices in canonical determinant order.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def screen_dynamic_perturbative_space(
    hamiltonian: 'np.ndarray',
    amplitudes: 'np.ndarray',
    determinants: 'np.ndarray',
    variational_indices: 'np.ndarray',
    eps1: float,
) -> 'np.ndarray':
    """Select external rows passing the amplitude-weighted heat-bath boundary.

    The selected amplitudes are normalized before screening. Archive rows
    outside the variational set are retained when their largest
    ``abs(H[a,i] * psi[i])`` is at least ``eps1``. Return retained archive
    indices in lexicographic ``(alpha_mask, beta_mask)`` order.

    Returns
    -------
    np.ndarray
        Zero-based retained archive indices in canonical determinant order.

    Raises
    ------
    ValueError
        If the matrix, amplitudes, determinants, indices or threshold are invalid.
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_screen_dynamic_perturbative_space(
    hamiltonian: 'np.ndarray',
    amplitudes: 'np.ndarray',
    determinants: 'np.ndarray',
    variational_indices: 'np.ndarray',
    eps1: float,
) -> 'np.ndarray':

    h = np.asarray(hamiltonian, dtype=float)
    psi = np.asarray(amplitudes, dtype=float)
    dets = np.asarray(determinants)
    v = np.asarray(variational_indices)

    if (
        h.ndim != 2
        or h.shape[0] == 0
        or h.shape[0] != h.shape[1]
    ):
        raise ValueError(
            "hamiltonian must be a nonempty square matrix"
        )

    n = h.shape[0]

    if (
        psi.shape != (n,)
        or dets.shape != (n, 2)
        or v.ndim != 1
        or v.size == 0
    ):
        raise ValueError("incompatible input shapes")

    if not np.issubdtype(
        dets.dtype,
        np.integer,
    ) or not np.issubdtype(
        v.dtype,
        np.integer,
    ):
        raise ValueError(
            "determinants and indices must be integral"
        )

    if np.any(dets < 0):
        raise ValueError(
            "determinant masks must be nonnegative"
        )

    if np.any(~np.isfinite(h)) or np.any(~np.isfinite(psi)):
        raise ValueError(
            "hamiltonian and amplitudes must be finite"
        )

    if not np.allclose(
        h,
        h.T,
        atol=2e-11,
        rtol=0.0,
    ):
        raise ValueError(
            "hamiltonian must be symmetric"
        )

    if len({tuple(map(int, d)) for d in dets}) != n:
        raise ValueError(
            "determinants must be unique"
        )

    if (
        np.any(v < 0)
        or np.any(v >= n)
        or np.unique(v).size != v.size
    ):
        raise ValueError(
            "variational indices must be unique and in range"
        )

    if not np.isfinite(eps1) or float(eps1) < 0.0:
        raise ValueError(
            "eps1 must be finite and nonnegative"
        )

    norm = float(np.linalg.norm(psi[v]))
    if norm == 0.0:
        raise ValueError(
            "selected amplitudes must have nonzero norm"
        )

    psi_v = psi[v] / norm

    is_external = np.ones(n, dtype=bool)
    is_external[v] = False
    external = np.flatnonzero(is_external)

    if external.size == 0:
        return np.empty(0, dtype=np.int64)

    score = np.max(
        np.abs(
            h[np.ix_(external, v)]
            * psi_v[None, :]
        ),
        axis=1,
    )

    keep = external[score >= float(eps1)]

    if keep.size == 0:
        return np.empty(0, dtype=np.int64)

    order = np.lexsort(
        (
            dets[keep, 1],
            dets[keep, 0],
        )
    )

    return keep[order].astype(np.int64)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": "import numpy as np\nh=np.array([[-1.,.2,.05,0.],[.2,-.7,.3,.08],[.05,.3,.4,.1],[0.,.08,.1,.9]]); psi=np.array([3.,4.,.2,.1]); d=np.array([[3,1],[1,3],[2,3],[3,2]],dtype=np.int64); v=np.array([0,1])",
            "call": "screen_dynamic_perturbative_space(h.copy(),psi.copy(),d.copy(),v.copy(),.12)",
            "gold_call": "_oracle_screen_dynamic_perturbative_space(h,psi,d,v,.12)",
        },
        {
            "setup": "import numpy as np\nh=np.array([[0.,.25],[.25,0.]]); psi=np.array([1.,0.]); d=np.array([[1,1],[2,1]],dtype=np.int64); v=np.array([0])",
            "call": "screen_dynamic_perturbative_space(h.copy(),psi.copy(),d.copy(),v.copy(),.25)",
            "gold_call": "_oracle_screen_dynamic_perturbative_space(h,psi,d,v,.25)",
        },
        {
            "setup": "import numpy as np\nh=np.diag([-1.,-.5,.2]); psi=np.array([1.,-2.,.3]); d=np.array([[2,3],[1,2],[1,1]],dtype=np.int64); v=np.array([0,1])",
            "call": "screen_dynamic_perturbative_space(h.copy(),psi.copy(),d.copy(),v.copy(),.001)",
            "gold_call": "_oracle_screen_dynamic_perturbative_space(h,psi,d,v,.001)",
        },
        {
            "setup": "import numpy as np\nh=np.array([[0.,0.,.4,.3],[0.,0.,.1,.2],[.4,.1,0.,0.],[.3,.2,0.,0.]]); psi=np.array([1.,1.,0.,0.]); d=np.array([[3,3],[2,2],[5,1],[1,5]],dtype=np.int64); v=np.array([0,1])",
            "call": "screen_dynamic_perturbative_space(h.copy(),psi.copy(),d.copy(),v.copy(),.05)",
            "gold_call": "_oracle_screen_dynamic_perturbative_space(h,psi,d,v,.05)",
        },
        {
            "setup": "import numpy as np\nh=np.array([[-1.,.2],[.2,-.5]]); psi=np.array([1.,-2.]); d=np.array([[1,1],[2,1]],dtype=np.int64); v=np.array([0,1])",
            "call": "screen_dynamic_perturbative_space(h.copy(),psi.copy(),d.copy(),v.copy(),.1)",
            "gold_call": "_oracle_screen_dynamic_perturbative_space(h,psi,d,v,.1)",
        },
        {
            "setup": "import numpy as np\nh=np.eye(2); psi=np.ones(2); d=np.array([[1,1],[2,1]],dtype=np.int64); v=np.array([0])\ndef check(fn):\n try: fn(h.copy(),psi.copy(),d.copy(),v.copy(),-.1)\n except ValueError: return 1\n except Exception: return 2\n return 0",
            "call": "check(screen_dynamic_perturbative_space)",
            "gold_call": "check(_oracle_screen_dynamic_perturbative_space)",
        },
        {
            "setup": "import numpy as np\nh=np.eye(2); psi=np.ones(2); d=np.array([[-1,1],[2,1]],dtype=np.int64); v=np.array([1])\ndef check(fn):\n try: fn(h.copy(),psi.copy(),d.copy(),v.copy(),.1)\n except ValueError: return 1\n except Exception: return 2\n return 0",
            "call": "check(screen_dynamic_perturbative_space)",
            "gold_call": "check(_oracle_screen_dynamic_perturbative_space)",
        },
    ]
