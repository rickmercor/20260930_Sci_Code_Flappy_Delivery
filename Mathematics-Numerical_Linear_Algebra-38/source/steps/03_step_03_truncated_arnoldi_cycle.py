"""
Generate one adaptive Krylov decomposition using truncated orthogonalization and sketch-based conditioning.

Krylov methods build a sequence of vectors associated with repeated applications of an operator. In this benchmark, the basis is generated with limited orthogonalization and its numerical quality is monitored in a lower-dimensional representation so that the usable basis size can be selected adaptively.

Returns
-------
float, selected Krylov dimension represented as a native Python float
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def truncated_arnoldi_cycle(
    state: dict,
    t: int,
    tau: float,
    s0: int,
    eta: float,
    mmax: int,
) -> float:
    """Run one adaptive truncated-Arnoldi cycle.

    Parameters
    ----------
    state : dict
        Mutable pipeline state containing the operator, starting vector,
        sketch, and random-number generator.
    t : int
        Truncation parameter.
    tau : float
        Condition-number threshold.
    s0 : int
        Sketch rows added during growth.
    eta : float
        Sketch-growth control parameter.
    mmax : int
        Maximum Krylov dimension.

    Returns
    -------
    float
        Selected Krylov dimension.
    """
    result = 0.0
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _draw_sparse_sign_cycle(
    rng: np.random.Generator,
    rows: int,
    cols: int,
) -> np.ndarray:
    zeta = min(rows, 8)
    S = np.zeros((rows, cols), dtype=float)

    for j in range(cols):
        idx = rng.choice(rows, size=zeta, replace=False)
        signs = rng.choice(np.array([-1.0, 1.0]), size=zeta)
        S[idx, j] = signs / np.sqrt(zeta)

    return S


def _oracle_truncated_arnoldi_cycle(
    state: dict,
    t: int,
    tau: float,
    s0: int,
    eta: float,
    mmax: int,
) -> float:
    import numpy as np

    if not isinstance(state, dict):
        raise ValueError("state must be a dict")
    required = ("A", "b", "S", "rng")
    if any(key not in state for key in required):
        raise ValueError("state is missing required data")
    if not (isinstance(t, (int, np.integer)) and t >= 1):
        raise ValueError("t must be >= 1")
    if not (isinstance(tau, (int, float)) and float(tau) > 1.0):
        raise ValueError("tau must be > 1")
    if not (isinstance(s0, (int, np.integer)) and s0 >= 1):
        raise ValueError("s0 must be positive")
    if not (isinstance(eta, (int, float)) and float(eta) > 0.0):
        raise ValueError("eta must be positive")
    if not (isinstance(mmax, (int, np.integer)) and mmax >= 1):
        raise ValueError("mmax must be positive")

    A = np.asarray(state["A"], dtype=float)
    bstart = np.asarray(state["b"], dtype=float)
    S = np.asarray(state["S"], dtype=float)
    rng = state["rng"]

    beta = float(np.linalg.norm(bstart))
    if beta == 0.0:
        raise ValueError("starting vector must be nonzero")

    b1 = bstart / beta
    Bcols = [b1]

    s = S.shape[0]
    P = (S @ b1).reshape(-1, 1)

    H = np.zeros((mmax + 1, mmax), dtype=float)
    hsub = []

    m = mmax

    for j in range(1, mmax + 1):
        ej = A @ Bcols[j - 1]

        first_i = max(1, j - t + 1)

        for i in range(first_i, j + 1):
            hij = float(np.dot(Bcols[i - 1], ej))
            ej = ej - Bcols[i - 1] * hij
            H[i - 1, j - 1] = hij

        hj1j = float(np.linalg.norm(ej))
        if hj1j <= np.finfo(float).eps:
            raise ValueError("Arnoldi breakdown")

        H[j, j - 1] = hj1j
        hsub.append(hj1j)

        bj1 = ej / hj1j
        Bcols.append(bj1)

        P = np.column_stack((P, S @ bj1))

        if np.linalg.cond(P) > float(tau):
            m = j
            break

        if s < float(eta) * (j + 1):
            Sincr = _draw_sparse_sign_cycle(rng, s0, A.shape[0])
            S = np.vstack((S, Sincr))

            Bpartial = np.column_stack(Bcols)
            P = np.vstack((P, Sincr @ Bpartial))

            s += s0

        m = j

    Bm = np.column_stack(Bcols[:m])
    Hm = H[:m, :m]
    bmp1 = Bcols[m]

    state["cycle_Bm"] = Bm
    state["cycle_Hm"] = Hm
    state["cycle_bmp1"] = bmp1
    state["cycle_hmp1_m"] = float(hsub[m - 1])
    state["cycle_subdiag"] = np.asarray(hsub[:m - 1], dtype=float)
    state["S"] = S
    state["sketch_rows"] = S.shape[0]
    state["beta"] = beta

    return float(m)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": """import numpy as np
N = 6
state = {
    "A": np.diag(np.arange(1.0, N + 1)) + 0.05 * np.ones((N, N)),
    "b": np.ones(N),
    "S": _draw_sparse_sign_cycle(np.random.default_rng(2), 4, N),
    "rng": np.random.default_rng(1),
}
state["b"] /= np.linalg.norm(state["b"])
t = 1
tau = 1e6
s0 = 2
eta = 2.0
mmax = 4
""",
            "call": "truncated_arnoldi_cycle(state, t, tau, s0, eta, mmax)",
            "gold_call": "_oracle_truncated_arnoldi_cycle(state, t, tau, s0, eta, mmax)",
        },
        {
            "setup": """import numpy as np
N = 4
state = {
    "A": np.diag(np.arange(1.0, N + 1)),
    "b": np.ones(N),
    "S": np.eye(3, N),
    "rng": np.random.default_rng(3),
}
state["b"] /= np.linalg.norm(state["b"])
t = 1
tau = 1e8
s0 = 2
eta = 2.0
mmax = 1
""",
            "call": "truncated_arnoldi_cycle(state, t, tau, s0, eta, mmax)",
            "gold_call": "_oracle_truncated_arnoldi_cycle(state, t, tau, s0, eta, mmax)",
        },
        {
            "setup": """import numpy as np
N = 8
A = np.diag(np.geomspace(1.0, 1000.0, N))
b = np.ones(N)
b /= np.linalg.norm(b)
state = {
    "A": A,
    "b": b,
    "S": _draw_sparse_sign_cycle(np.random.default_rng(4), 4, N),
    "rng": np.random.default_rng(5),
}
t = 1
tau = 5.0
s0 = 3
eta = 2.0
mmax = 6
""",
            "call": "truncated_arnoldi_cycle(state, t, tau, s0, eta, mmax)",
            "gold_call": "_oracle_truncated_arnoldi_cycle(state, t, tau, s0, eta, mmax)",
        },
    ]
