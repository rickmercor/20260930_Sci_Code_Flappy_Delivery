"""
Implement ssrsbw_jet. Sequential state-specific optimization freezes previously accepted reference states but retains them as external perturbers. Each corrected energy is evaluated when its own reference is accepted.

Sequential state-specific optimization freezes previously accepted reference states but retains them as external perturbers. Each corrected energy is evaluated when its own reference is accepted.

Returns
-------
np.ndarray of shape (4,targets), sorted corrected energy jets.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def ssrsbw_jet(H: np.ndarray, targets: int = 3,
                       rho: float = 0.4, enrich: float = 0.6,
                       max_updates: int = 50) -> np.ndarray:
    """Optimize states sequentially and return corrected energy derivatives.

    Parameters
    ----------
    H : np.ndarray
        Finite real symmetric Hamiltonian jet (4,n,n), n>=1, in actual
        (value,a,b,ab) derivatives. Symmetry atol=1e-12, rtol=0.
        Initially e is the diagonal jet of H. Stably order the initial
        basis by e[0], preserving input order for ties.
    targets : int
        Number of targeted states, 1<=targets<=n, not bool.
    rho, enrich : float
        Finite real scalars 0<rho<enrich, used by select_space.
    max_updates : int
        Positive integer, not bool; maximum transformations per target.
        For each target, use select_space, effective_jet, eigensystem_jet
        and update_partition until the selected space is a singleton.
        Compute bw_root_jet immediately, then freeze the optimized reference
        (not its corrected energy). Frozen states remain external perturbers.
        Finally sort the stored corrected energy jets by their baseline.
        Derivatives describe the selected fixed branch; do not differentiate
        the discrete selection decisions and do not use finite differences.

    Returns
    -------
    result : np.ndarray
        Shape (4,targets): sorted corrected energies and control derivatives.

    Raises
    ------
    ValueError
        If input shape, finiteness, reality, symmetry, integer controls or
        thresholds violate the conditions above; a required P-Q denominator
        has magnitude <=1e-10; an effective spectrum has separation <=1e-10;
        a rotation jet fails orthogonality at atol=1e-9; a BW reference lies
        within 1e-10 of a pole or lacks exactly one root >1e-10 from finite
        interval bounds; a numerical eigensolver fails; calculations become
        nonfinite; more than max_updates transformations are needed for a
        target; or final corrected energies have separation <=1e-10.
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_ssrsbw_jet(H: np.ndarray, targets: int = 3,
                       rho: float = 0.4, enrich: float = 0.6,
                       max_updates: int = 50) -> np.ndarray:
    """Optimize states sequentially and return corrected energy derivatives.

    Parameters
    ----------
    H : np.ndarray
        Finite real symmetric Hamiltonian jet (4,n,n), n>=1, in actual
        (value,a,b,ab) derivatives. Symmetry atol=1e-12, rtol=0.
        Initially e is the diagonal jet of H. Stably order the initial
        basis by e[0], preserving input order for ties.
    targets : int
        Number of targeted states, 1<=targets<=n, not bool.
    rho, enrich : float
        Finite real scalars 0<rho<enrich, used by select_space.
    max_updates : int
        Positive integer, not bool; maximum transformations per target.
        For each target, use select_space, effective_jet, eigensystem_jet
        and update_partition until the selected space is a singleton.
        Compute bw_root_jet immediately, then freeze the optimized reference
        (not its corrected energy). Frozen states remain external perturbers.
        Finally sort the stored corrected energy jets by their baseline.
        Derivatives describe the selected fixed branch; do not differentiate
        the discrete selection decisions and do not use finite differences.

    Returns
    -------
    result : np.ndarray
        Shape (4,targets): sorted corrected energies and control derivatives.

    Raises
    ------
    ValueError
        If input shape, finiteness, reality, symmetry, integer controls or
        thresholds violate the conditions above; a required P-Q denominator
        has magnitude <=1e-10; an effective spectrum has separation <=1e-10;
        a rotation jet fails orthogonality at atol=1e-9; a BW reference lies
        within 1e-10 of a pole or lacks exactly one root >1e-10 from finite
        interval bounds; a numerical eigensolver fails; calculations become
        nonfinite; more than max_updates transformations are needed for a
        target; or final corrected energies have separation <=1e-10.
    """
    try:
        if np.iscomplexobj(H):
            raise ValueError('complex H')
        H = np.asarray(H,float).copy()
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError('invalid H') from exc
    if H.ndim != 3 or H.shape[0] != 4 or H.shape[1] < 1 or H.shape[1] != H.shape[2]:
        raise ValueError('invalid H shape')
    n = H.shape[1]
    if not np.isfinite(H).all() or not np.allclose(H,H.transpose(0,2,1),atol=1e-12,rtol=0):
        raise ValueError('invalid jet')
    if isinstance(targets,(bool,np.bool_)) or not isinstance(targets,(int,np.integer)) or not 1<=targets<=n:
        raise ValueError('invalid targets')
    if isinstance(max_updates,(bool,np.bool_)) or not isinstance(max_updates,(int,np.integer)) or max_updates<1:
        raise ValueError('invalid max_updates')
    e = np.array([np.diag(slot) for slot in H])
    order = np.argsort(e[0],kind='stable')
    H,e = H[:,order,:][:,:,order],e[:,order]
    corrected = []
    for k in range(targets):
        updates = 0
        while True:
            P = _oracle_select_space(H[0],e[0],k,rho,enrich)
            if len(P) == 1:
                break
            if updates == max_updates:
                raise ValueError('transformation limit exceeded')
            A = _oracle_effective_jet(H,e,P)
            L,V = _oracle_eigensystem_jet(A)
            H,e = _oracle_update_partition(H,e,P,L,V,frozen=k)
            updates += 1
        corrected.append(_oracle_bw_root_jet(H,e,k))
    result = np.stack(corrected,axis=1)
    result = result[:,np.argsort(result[0],kind='stable')]
    if np.any(np.diff(result[0])<=1e-10):
        raise ValueError('corrected spectrum not simple')
    return result

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": """import numpy as np
H = np.zeros((4, 4, 4))
H[0] = [[0, 3.2, -1, -1], [3.2, 0, -1.5, -1.5], [-1, -1.5, 3, 1.5], [-1, -1.5, 1.5, 6]]
H[1, 0, 1] = H[1, 1, 0] = 1
H[2, 2, 3] = H[2, 3, 2] = 1
targets = 3


def run(fn):
    return fn(H.copy(), targets=targets)
""",
            "call": 'run(ssrsbw_jet)',
            "gold_call": 'run(_oracle_ssrsbw_jet)',
        },
        {
            "setup": """import numpy as np
H = np.array([2.0, 0.3, -0.2, 0.5]).reshape(4, 1, 1)
targets = 1


def run(fn):
    return fn(H.copy(), targets=targets)
""",
            "call": 'run(ssrsbw_jet)',
            "gold_call": 'run(_oracle_ssrsbw_jet)',
        },
        {
            "setup": """import numpy as np
H = np.zeros((4, 3, 3))
H[0] = np.diag([2, -1, 4])
H[1] = np.diag([0.2, 0.3, 0.4])
H[2] = np.diag([0.1, -0.2, 0.5])
H[3] = np.diag([0.7, 0.8, 0.9])
targets = 3


def run(fn):
    return fn(H.copy(), targets=targets)
""",
            "call": 'run(ssrsbw_jet)',
            "gold_call": 'run(_oracle_ssrsbw_jet)',
        },
        {
            "setup": """import numpy as np
H = np.zeros((4, 4, 4))
H[0] = [[0, 3.2, -1, -1], [3.2, 0, -1.5, -1.5], [-1, -1.5, 3, 1.5], [-1, -1.5, 1.5, 6]]
H[1, 0, 1] = H[1, 1, 0] = 1
H[2, 2, 3] = H[2, 3, 2] = 1
targets = 3


def run(fn):
    try:
        fn(H.copy(), targets=targets, rho=0.6, enrich=0.4)
    except ValueError:
        return 1
    return 0
""",
            "call": 'run(ssrsbw_jet)',
            "gold_call": 'run(_oracle_ssrsbw_jet)',
        },
        {
            "setup": """import numpy as np
rng = np.random.default_rng(3712)
H = np.zeros((4, 4, 4))
H[0] = [[0, 3.2, -1, -1], [3.2, 0, -1.5, -1.5], [-1, -1.5, 3, 1.5], [-1, -1.5, 1.5, 6]]
X = rng.normal(size=(3, 4, 4))
H[1:] = 0.07 * (X + X.transpose(0, 2, 1))
H[1:, 1, 1] = H[1:, 0, 0]
p = np.array([2, 0, 3, 1])
H = H[:, p, :][:, :, p]


def run(fn):
    return fn(H.copy(), targets=3)
""",
            "call": 'run(ssrsbw_jet)',
            "gold_call": 'run(_oracle_ssrsbw_jet)',
        },
        {
            "setup": """import numpy as np
rng = np.random.default_rng(3713)
H = np.zeros((4, 4, 4))
H[0] = [[0, 3.2, -1, -1], [3.2, 0, -1.5, -1.5], [-1, -1.5, 3, 1.5], [-1, -1.5, 1.5, 6]]
X = rng.normal(size=(3, 4, 4))
H[1:] = 0.07 * (X + X.transpose(0, 2, 1))
H[1:, 1, 1] = H[1:, 0, 0]
p = np.array([3, 2, 1, 0])
H = H[:, p, :][:, :, p]


def run(fn):
    return fn(H.copy(), targets=3)
""",
            "call": 'run(ssrsbw_jet)',
            "gold_call": 'run(_oracle_ssrsbw_jet)',
        },
        {
            "setup": """import numpy as np
rng = np.random.default_rng(3714)
H = np.zeros((4, 6, 6))
H[0, :4, :4] = [[0, 3.2, -1, -1], [3.2, 0, -1.5, -1.5], [-1, -1.5, 3, 1.5], [-1, -1.5, 1.5, 6]]
H[0, 4, 4] = 10.0
H[0, 5, 5] = 14.0
H[0, :4, 4:] = [[0.12, -0.07], [-0.11, 0.09], [0.08, 0.1], [-0.06, 0.13]]
H[0, 4:, :4] = H[0, :4, 4:].T
H[0, 4, 5] = H[0, 5, 4] = 0.15
X = rng.normal(size=(3, 6, 6))
H[1:] = 0.04 * (X + X.transpose(0, 2, 1))
H[1:, 1, 1] = H[1:, 0, 0]


def run(fn):
    return fn(H.copy(), targets=4)
""",
            "call": 'run(ssrsbw_jet)',
            "gold_call": 'run(_oracle_ssrsbw_jet)',
        },
        {
            "setup": """import numpy as np
H = np.zeros((4, 4, 4))
H[0] = [[0, 3.2, -1, -1], [3.2, 0, -1.5, -1.5], [-1, -1.5, 3, 1.5], [-1, -1.5, 1.5, 6]]
H[1, 0, 1] = H[1, 1, 0] = 1
H[2, 2, 3] = H[2, 3, 2] = 1
targets = 3
H[1] *= 2.0
H[2] *= -3.0
H[3] *= -6.0
H += np.array([8.0, 0.3, -0.2, 0.7])[:, None, None] * np.eye(4)


def run(fn):
    return fn(H.copy(), targets=3)
""",
            "call": 'run(ssrsbw_jet)',
            "gold_call": 'run(_oracle_ssrsbw_jet)',
        },
        {
            "setup": """import numpy as np
H = np.zeros((4, 4, 4))
H[0] = [[0, 3.2, -1, -1], [3.2, 0, -1.5, -1.5], [-1, -1.5, 3, 1.5], [-1, -1.5, 1.5, 6]]
H[1, 0, 1] = H[1, 1, 0] = 1
H[2, 2, 3] = H[2, 3, 2] = 1
targets = 3
A, B = H[1].copy(), H[2].copy()
H[1] = 2.0*A - 0.4*B
H[2] = -0.3*A + 1.1*B
H[3] = 0.2*A - 0.5*B
H += np.array([5.0, 0.2, -0.1, 0.7])[:, None, None] * np.eye(4)


def run(fn):
    return fn(H.copy(), targets=3)
""",
            "call": 'run(ssrsbw_jet)',
            "gold_call": 'run(_oracle_ssrsbw_jet)',
        },
    ]
