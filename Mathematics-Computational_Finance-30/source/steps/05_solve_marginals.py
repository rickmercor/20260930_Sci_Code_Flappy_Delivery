"""
Assemble and solve the fitting program (SMP) that Section 4.1 of the source writes down, returning one marginal per expiry. The three maps are handed in: the per-expiry price maps from step 2 stacked along the first axis, the ordering map from step 3, and the per-expiry equality block from step 4 as a pair. The source states which discrepancy measure the objective minimises and how the quote weights enter it; that choice is what keeps the program in the class Section 4.1 names. Impose the cross-expiry ordering the source lists under 'Constraints', stated there in terms of step 3's output rather than of the marginals themselves. Non-negativity applies to every marginal. If the program does not solve, return an all-zero array of the correct shape rather than raising.

Section 4.1 sets the fit out as a program in a specific class, with the marginals as the decision variables and two families of slack quantities. The quoted mids are not assumed free of arbitrage, so the ordering constraint is generally active and the fit is not exact.

Returns
-------
A float64 array with one row per expiry and one column per ladder point.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def solve_marginals(price_matrices: "np.ndarray", ordering_map: "np.ndarray", constraint_block: "tuple[np.ndarray, np.ndarray]", mid_quotes: "np.ndarray", weights: "np.ndarray") -> "np.ndarray":
    """Fit one marginal per expiry to the quoted mids under the source's program.

    Parameters
    ----------
    price_matrices : numpy.ndarray
        Array of shape (M, R, N): the step 2 map of each of the M expiries, stacked along the first axis.
    ordering_map : numpy.ndarray
        The step 3 square map of shape (N, N).
    constraint_block : tuple of numpy.ndarray
        The step 4 pair (A, b): A of shape (rows, N) and b of shape (rows,), applied to every expiry.
    mid_quotes : numpy.ndarray
        Quoted mids of shape (M, N), one row per expiry over the full ladder; only the rows the price map prices are fitted.
    weights : numpy.ndarray
        Non-negative quote weights of shape (M, N), aligned with mid_quotes.

    Returns
    -------
    marginals : numpy.ndarray
        float64 array of shape (M, N): the fitted marginal of each expiry on the ladder, or all zeros if the program does not solve.

    Raises
    ------
    ValueError
        If price_matrices is not three-dimensional, if R does not equal the row count Eq. (31) prescribes for N, if ordering_map is not (N, N), if mid_quotes or weights is not (M, N), or if the constraint block does not have N columns.
    """
    return marginals

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
from scipy.optimize import linprog


def _oracle_solve_marginals(price_matrices: "np.ndarray", ordering_map: "np.ndarray", constraint_block: "tuple[np.ndarray, np.ndarray]", mid_quotes: "np.ndarray", weights: "np.ndarray") -> "np.ndarray":
    """L1 fit of the marginals under mass, unit mean and the cross-expiry ordering."""
    C = np.asarray(price_matrices, dtype=np.float64)
    U = np.asarray(ordering_map, dtype=np.float64)
    Q = np.asarray(mid_quotes, dtype=np.float64)
    W = np.asarray(weights, dtype=np.float64)
    if C.ndim != 3:
        raise ValueError("price_matrices must be stacked per expiry with shape (M, R, N)")
    M, R, N = C.shape
    if U.shape != (N, N):
        raise ValueError(f"ordering_map must be ({N}, {N}), got {U.shape}")
    if Q.shape != (M, N) or W.shape != (M, N):
        raise ValueError(f"mid_quotes and weights must both be ({M}, {N})")
    Aeq_blk, beq_blk = constraint_block
    Aeq_blk = np.asarray(Aeq_blk, dtype=np.float64)
    beq_blk = np.asarray(beq_blk, dtype=np.float64)
    if Aeq_blk.ndim != 2 or Aeq_blk.shape[1] != N:
        raise ValueError(f"constraint block must have {N} columns")
    if R != N - 2:
        raise ValueError("price_matrices must carry the interior row set")

    rows = np.arange(1, N - 1)
    nq, nt = M * N, M * R
    qs = lambda j: slice(j * N, (j + 1) * N)
    ts = lambda j: slice(nq + j * R, nq + (j + 1) * R)

    c = np.zeros(nq + nt, dtype=np.float64)
    c[nq:] = 1.0
    Aub, bub = [], []
    for j in range(M):
        for sgn in (1.0, -1.0):
            A = np.zeros((R, nq + nt))
            A[:, qs(j)] = sgn * (W[j, rows][:, None] * C[j])
            A[:, ts(j)] = -np.eye(R)
            Aub.append(A)
            bub.append(sgn * (W[j, rows] * Q[j, rows]))
    for j in range(1, M):
        A = np.zeros((N, nq + nt))
        A[:, qs(j - 1)] = U
        A[:, qs(j)] = -U
        Aub.append(A)
        bub.append(np.zeros(N))
    nr = Aeq_blk.shape[0]
    Aeq, beq = [], []
    for j in range(M):
        Z = np.zeros((nr, nq + nt))
        Z[:, qs(j)] = Aeq_blk
        Aeq.append(Z)
        beq.append(beq_blk)

    res = linprog(c, A_ub=np.vstack(Aub), b_ub=np.concatenate(bub),
                  A_eq=np.vstack(Aeq), b_eq=np.concatenate(beq),
                  bounds=[(0.0, None)] * (nq + nt), method="highs")
    if not res.success:
        return np.zeros((M, N), dtype=np.float64)
    return np.asarray(res.x[:nq].reshape(M, N), dtype=np.float64)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return the list of test case specifications."""
    return [
        {
            'setup': 'import numpy as np\nK = np.array([0.8, 0.9, 1.0, 1.1, 1.2])\nC = np.array([[[0.006690694657640911, 0.03931208647604689, 0.10929797755934978, 0.20143381909227975, 0.30015309052325445], [0.0007520560582488824, 0.009297977559349663, 0.04368009608449652, 0.11217456803829695, 0.20238976718097856], [5.4252703699081504e-05, 0.0014338190922797006, 0.012174568038296918, 0.048048105692946264, 0.11526797706991831]], [[0.01932332719706975, 0.05893131008281882, 0.12453012723932388, 0.2087597680252229, 0.30274953560141393], [0.005893308030281402, 0.02453012723932374, 0.06547923342535422, 0.12996949321921758, 0.21209268300484818], [0.0015386259056672028, 0.008759768025222825, 0.02996949321921738, 0.07202715676788962, 0.13558735190403703]]])\nU = np.maximum(K[None, :] - K[:, None], 0.0)\nB = (np.vstack([np.ones_like(K), K]), np.array([1.0, 1.0]))\nQ = np.array([[0.205, 0.118, 0.056, 0.020, 0.006], [0.215, 0.135, 0.079, 0.040, 0.017]])\nW = np.ones_like(Q)\n',
            'call': 'solve_marginals(C, U, B, Q, W)',
            'gold_call': '_oracle_solve_marginals(C, U, B, Q, W)',
        },
        {
            'setup': 'import numpy as np\nK = np.array([0.8, 0.9, 1.0, 1.1, 1.2])\nC = np.array([[[0.006690694657640911, 0.03931208647604689, 0.10929797755934978, 0.20143381909227975, 0.30015309052325445], [0.0007520560582488824, 0.009297977559349663, 0.04368009608449652, 0.11217456803829695, 0.20238976718097856], [5.4252703699081504e-05, 0.0014338190922797006, 0.012174568038296918, 0.048048105692946264, 0.11526797706991831]], [[0.01932332719706975, 0.05893131008281882, 0.12453012723932388, 0.2087597680252229, 0.30274953560141393], [0.005893308030281402, 0.02453012723932374, 0.06547923342535422, 0.12996949321921758, 0.21209268300484818], [0.0015386259056672028, 0.008759768025222825, 0.02996949321921738, 0.07202715676788962, 0.13558735190403703]]])\nU = np.maximum(K[None, :] - K[:, None], 0.0)\nB = (np.vstack([np.ones_like(K), K]), np.array([1.0, 1.0]))\nQ = np.array([[0.205, 0.118, 0.056, 0.020, 0.006], [0.215, 0.135, 0.079, 0.040, 0.017]])\nW = np.ones_like(Q)\ndef sums(m):\n    return (float(np.sum(m)), float(np.sum(m @ K)))\n',
            'call': 'sums(solve_marginals(C, U, B, Q, W))',
            'gold_call': 'sums(_oracle_solve_marginals(C, U, B, Q, W))',
        },
        {
            'setup': 'import numpy as np\nK = np.array([0.9, 1.0, 1.1])\nC = np.array([[[0.012529307753020108, 0.048829730703064456, 0.11602534329157677]]])\nU = np.maximum(K[None, :] - K[:, None], 0.0)\nB = (np.vstack([np.ones_like(K), K]), np.array([1.0, 1.0]))\nQ = np.array([[0.105, 0.045, 0.012]])\nW = np.ones_like(Q)\n',
            'call': 'solve_marginals(C, U, B, Q, W)',
            'gold_call': '_oracle_solve_marginals(C, U, B, Q, W)',
        },
        {
            'setup': 'import numpy as np\nK = np.array([0.8, 0.9, 1.0, 1.1, 1.2])\nC = np.array([[[0.006690694657640911, 0.03931208647604689, 0.10929797755934978, 0.20143381909227975, 0.30015309052325445], [0.0007520560582488824, 0.009297977559349663, 0.04368009608449652, 0.11217456803829695, 0.20238976718097856], [5.4252703699081504e-05, 0.0014338190922797006, 0.012174568038296918, 0.048048105692946264, 0.11526797706991831]], [[0.01932332719706975, 0.05893131008281882, 0.12453012723932388, 0.2087597680252229, 0.30274953560141393], [0.005893308030281402, 0.02453012723932374, 0.06547923342535422, 0.12996949321921758, 0.21209268300484818], [0.0015386259056672028, 0.008759768025222825, 0.02996949321921738, 0.07202715676788962, 0.13558735190403703]]])\nU = np.maximum(K[None, :] - K[:, None], 0.0)\nB = (np.vstack([np.ones_like(K), K]), np.array([1.0, 1.0]))\nQ = np.array([[0.205, 0.118, 0.056, 0.020, 0.006], [0.215, 0.135, 0.079, 0.040, 0.017]])\nW = np.ones_like(Q)\nW2 = np.array([[1.0, 2.0, 4.0, 2.0, 1.0], [1.0, 2.0, 4.0, 2.0, 1.0]])\n',
            'call': 'solve_marginals(C, U, B, Q, W2)',
            'gold_call': '_oracle_solve_marginals(C, U, B, Q, W2)',
        },
        {
            'setup': 'import numpy as np\nK = np.array([1.1, 1.2, 1.3])\nC = np.array([[[0.019730664431426292, 0.05859567684367739, 0.12360336800947058]]])\nU = np.maximum(K[None, :] - K[:, None], 0.0)\nB = (np.vstack([np.ones_like(K), K]), np.array([1.0, 1.0]))\nQ = np.array([[0.02, 0.01, 0.005]])\nW = np.ones_like(Q)\n',
            'call': 'solve_marginals(C, U, B, Q, W)',
            'gold_call': '_oracle_solve_marginals(C, U, B, Q, W)',
        },
        {
            'setup': 'import numpy as np\nK = np.array([0.8, 0.9, 1.0, 1.1, 1.2])\nC = np.array([[[0.006690694657640911, 0.03931208647604689, 0.10929797755934978, 0.20143381909227975, 0.30015309052325445], [0.0007520560582488824, 0.009297977559349663, 0.04368009608449652, 0.11217456803829695, 0.20238976718097856], [5.4252703699081504e-05, 0.0014338190922797006, 0.012174568038296918, 0.048048105692946264, 0.11526797706991831]], [[0.01932332719706975, 0.05893131008281882, 0.12453012723932388, 0.2087597680252229, 0.30274953560141393], [0.005893308030281402, 0.02453012723932374, 0.06547923342535422, 0.12996949321921758, 0.21209268300484818], [0.0015386259056672028, 0.008759768025222825, 0.02996949321921738, 0.07202715676788962, 0.13558735190403703]]])\nU = np.maximum(K[None, :] - K[:, None], 0.0)\nB = (np.vstack([np.ones_like(K), K]), np.array([1.0, 1.0]))\nQ = np.array([[0.205, 0.118, 0.056, 0.020, 0.006], [0.215, 0.135, 0.079, 0.040, 0.017]])\nW = np.ones_like(Q)\ndef run_model():\n    try:\n        solve_marginals(C, U, B, Q[:, :3], W)\n        return 0\n    except ValueError:\n        return 1\ndef run_oracle():\n    try:\n        _oracle_solve_marginals(C, U, B, Q[:, :3], W)\n        return 0\n    except ValueError:\n        return 1\n',
            'call': 'run_model()',
            'gold_call': 'run_oracle()',
        },
    ]
