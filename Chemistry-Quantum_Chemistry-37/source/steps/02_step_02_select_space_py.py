"""
Implement select_space. State selection depends on coupling-to-energy-separation ratios. Secondary closure must include indirect couplings while excluding frozen states; exact degeneracies require selection without division.

State selection depends on coupling-to-energy-separation ratios. Secondary closure must include indirect couplings while excluding frozen states; exact degeneracies require selection without division. Frozen indices denote previously optimized references in the current sequential optimization order; this order is not assumed to match the exact Hamiltonian eigenvalue order.

Returns
-------
np.ndarray of shape (p,), sorted integer model-space indices.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def select_space(H: np.ndarray, e: np.ndarray, k: int,
                         rho: float = 0.4, enrich: float = 0.6) -> np.ndarray:
    """Select a candidate model space with secondary enrichment closure.

    Parameters
    ----------
    H : np.ndarray
        Finite real symmetric (n,n) Hamiltonian, n>=1; symmetry atol=1e-12.
    e : np.ndarray
        Finite real reference energies of shape (n,). These need not equal
        diag(H). Indices below k are frozen and cannot enter the model space.
    k : int
        Candidate index, 0<=k<n; booleans are not accepted as integers.
    rho, enrich : float
        Finite real scalars satisfying 0<rho<enrich. Starting with k, include
        every unfrozen j whose |H[k,j]/(e[k]-e[j])| exceeds rho, including
        exact degeneracies regardless of coupling. Enrich to closure using
        the same pair ratio and threshold enrich. All tests are strict >;
        exact degeneracies always qualify. Never divide a zero denominator.

    Returns
    -------
    result : np.ndarray
        Sorted, distinct integer indices of the selected space, including k.

    Raises
    ------
    ValueError
        If arrays are not convertible to finite real arrays of the stated
        shapes, H is not symmetric within atol=1e-12 and rtol=0, k is invalid,
        or thresholds are not finite real scalars with 0<rho<enrich.
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_select_space(H: np.ndarray, e: np.ndarray, k: int,
                         rho: float = 0.4, enrich: float = 0.6) -> np.ndarray:
    """Select a candidate model space with secondary enrichment closure.

    Parameters
    ----------
    H : np.ndarray
        Finite real symmetric (n,n) Hamiltonian, n>=1; symmetry atol=1e-12.
    e : np.ndarray
        Finite real reference energies of shape (n,). These need not equal
        diag(H). Indices below k are frozen and cannot enter the model space.
    k : int
        Candidate index, 0<=k<n; booleans are not accepted as integers.
    rho, enrich : float
        Finite real scalars satisfying 0<rho<enrich. Starting with k, include
        every unfrozen j whose |H[k,j]/(e[k]-e[j])| exceeds rho, including
        exact degeneracies regardless of coupling. Enrich to closure using
        the same pair ratio and threshold enrich. All tests are strict >;
        exact degeneracies always qualify. Never divide a zero denominator.

    Returns
    -------
    result : np.ndarray
        Sorted, distinct integer indices of the selected space, including k.

    Raises
    ------
    ValueError
        If arrays are not convertible to finite real arrays of the stated
        shapes, H is not symmetric within atol=1e-12 and rtol=0, k is invalid,
        or thresholds are not finite real scalars with 0<rho<enrich.
    """
    try:
        if np.iscomplexobj(H) or np.iscomplexobj(e):
            raise ValueError('complex arrays')
        H, e = np.asarray(H, float), np.asarray(e, float)
        t = np.asarray([rho, enrich])
        if t.shape != (2,) or np.iscomplexobj(t):
            raise ValueError('invalid thresholds')
        rho, enrich = np.asarray(t, float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError('invalid numeric inputs') from exc
    if H.ndim != 2 or H.shape[0] < 1 or H.shape[0] != H.shape[1]:
        raise ValueError('invalid H shape')
    n = H.shape[0]
    if e.shape != (n,) or not np.isfinite(H).all() or not np.isfinite(e).all():
        raise ValueError('invalid energies or entries')
    if not np.allclose(H, H.T, atol=1e-12, rtol=0):
        raise ValueError('asymmetric H')
    if isinstance(k, (bool, np.bool_)) or not isinstance(k, (int, np.integer)) or not 0 <= k < n:
        raise ValueError('invalid candidate')
    if not np.isfinite([rho, enrich]).all() or not 0 < rho < enrich:
        raise ValueError('invalid thresholds')
    d = np.abs(e[:,None] - e[None,:])
    r = np.full((n,n), np.inf)
    np.divide(np.abs(H), d, out=r, where=d != 0)
    np.fill_diagonal(r, 0)
    chosen = {k} | {j for j in range(k+1,n) if r[k,j] > rho}
    while True:
        expanded = chosen | {j for j in range(k,n) if j not in chosen
                             and any(r[i,j] > enrich for i in chosen)}
        if expanded == chosen:
            break
        chosen = expanded
    return np.array(sorted(chosen), dtype=int)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": """import numpy as np
e = np.array([0.0, 1.0, 3.0, 6.0])
H = np.diag(e)
H[0, 1] = H[1, 0] = 0.5
H[1, 2] = H[2, 1] = 1.4
H[2, 3] = H[3, 2] = 2.1
k = 0


def run(fn):
    return fn(H.copy(), e.copy(), k)
""",
            "call": 'run(select_space)',
            "gold_call": 'run(_oracle_select_space)',
        },
        {
            "setup": """import numpy as np
e = np.array([0.0, 1.0, 3.0])
H = np.diag(e)
H[0, 1] = H[1, 0] = 0.4
H[1, 2] = H[2, 1] = 1.2
k = 0


def run(fn):
    return fn(H.copy(), e.copy(), k)
""",
            "call": 'run(select_space)',
            "gold_call": 'run(_oracle_select_space)',
        },
        {
            "setup": """import numpy as np
e = np.array([0.0, 0.0, 0.0, 2.0])
H = np.diag(e)
k = 1


def run(fn):
    return fn(H.copy(), e.copy(), k)
""",
            "call": 'run(select_space)',
            "gold_call": 'run(_oracle_select_space)',
        },
        {
            "setup": """import numpy as np
H = np.array([[0.0, 1.0], [0.0, 2.0]])
e = np.array([0.0, 2.0])
k = 0


def run(fn):
    try:
        fn(H.copy(), e.copy(), k)
    except ValueError:
        return 1
    return 0
""",
            "call": 'run(select_space)',
            "gold_call": 'run(_oracle_select_space)',
        },
        {
            "setup": """import numpy as np
e = np.array([-8.0, 0.0, 1.0, 2.0, 3.0, 4.0])
H = np.diag(np.array([9.0, 8.0, 7.0, 6.0, 5.0, 4.0]))
for i, j, w in [(1, 5, 2.0), (5, 4, 0.7), (4, 3, 0.7), (3, 2, 0.7), (0, 1, 100.0)]:
    H[i, j] = H[j, i] = w


def run(fn):
    return fn(H.copy(), e.copy(), 1)
""",
            "call": 'run(select_space)',
            "gold_call": 'run(_oracle_select_space)',
        },
        {
            "setup": """import numpy as np
e = np.array([0.0, 0.0, 1.0, 2.0, 3.0])
H = np.diag(e)
H[0, 2] = H[2, 0] = 0.4
H[1, 3] = H[3, 1] = 1.2
H[1, 4] = H[4, 1] = 1.9


def run(fn):
    return fn(H.copy(), e.copy(), 0)
""",
            "call": 'run(select_space)',
            "gold_call": 'run(_oracle_select_space)',
        },
    ]
