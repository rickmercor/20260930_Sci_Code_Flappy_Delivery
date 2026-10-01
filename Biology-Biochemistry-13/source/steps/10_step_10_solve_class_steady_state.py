"""
Impose the conservation laws on a log-linear steady-state parametrization and return the positive steady state of the class.

A steady-state parametrization describes the whole positive steady-state variety, and the conserved totals select the steady state inside one stoichiometric compatibility class.

Returns
-------
np.ndarray: float positive steady state, shape (m,).
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np
def solve_class_steady_state(
    log_offset: np.ndarray,
    exponents: np.ndarray,
    conservation_matrix: np.ndarray,
    totals: np.ndarray,
) -> np.ndarray:
    """Return the positive steady state whose conserved totals equal ``totals``.

    Candidate states are ``x(z) = exp(log_offset + exponents @ z)`` with
    ``z`` in ``R**d``. Return the ``x(z)`` satisfying
    ``conservation_matrix @ x(z) = totals`` with a relative error below
    ``1e-12`` in every total; the inputs are such that exactly one such state
    exists. When ``d = 0`` the only candidate is ``exp(log_offset)``, which is
    returned without any condition being imposed.

    Parameters
    ----------
    log_offset : np.ndarray
        Float array with shape ``(m,)``.
    exponents : np.ndarray
        Float array with shape ``(m, d)``.
    conservation_matrix : np.ndarray
        Nonnegative weights with shape ``(d, m)``; every row has a positive
        entry.
    totals : np.ndarray
        Positive conserved totals with shape ``(d,)``.

    Returns
    -------
    np.ndarray
        Float array with shape ``(m,)``.

    Raises
    ------
    ValueError
        If the shapes are inconsistent (the number of conservation laws must
        equal ``d``; raise ValueError rather than letting a linear-algebra
        error propagate), if any input is non-finite, if a weight is
        negative or a row of weights is zero, if a total is not positive
        (zero included), or if no state meeting the tolerance is found.
    """
    return np.zeros(0, dtype=float)

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
def _oracle_solve_class_steady_state(
    log_offset: np.ndarray,
    exponents: np.ndarray,
    conservation_matrix: np.ndarray,
    totals: np.ndarray,
) -> np.ndarray:
    """Reference implementation (damped Newton on log-totals)."""
    import numpy as np

    offset = np.asarray(log_offset, dtype=float)
    powers = np.asarray(exponents, dtype=float)
    weights = np.asarray(conservation_matrix, dtype=float)
    target = np.asarray(totals, dtype=float)
    if offset.ndim != 1 or offset.size == 0:
        raise ValueError("log_offset must be a nonempty vector")
    species = offset.size
    if powers.ndim != 2 or powers.shape[0] != species:
        raise ValueError("exponents must have shape (m, d)")
    laws = powers.shape[1]
    if weights.size == 0:
        weights = np.zeros((0, species))
    if weights.shape != (laws, species) or target.shape != (laws,):
        raise ValueError("conservation data must have shapes (d, m) and (d,)")
    for values in (offset, powers, weights, target):
        if not np.all(np.isfinite(values)):
            raise ValueError("inputs must be finite")
    if np.any(weights < 0.0) or np.any(weights.sum(axis=1) <= 0.0) or np.any(target <= 0.0):
        raise ValueError("weights must be nonnegative with nonzero rows and totals positive")
    if laws == 0:
        return np.exp(offset)

    def evaluate(point):
        state = np.exp(offset + powers @ point)
        conserved = weights @ state
        return state, conserved, np.log(conserved) - np.log(target)

    point = np.zeros(laws)
    state, conserved, gap = evaluate(point)
    for _ in range(400):
        if np.max(np.abs(gap)) < 1e-14:
            break
        jacobian = (weights * state) @ powers / conserved[:, None]
        try:
            step = np.linalg.solve(jacobian, -gap)
        except np.linalg.LinAlgError:
            raise ValueError("singular Jacobian while imposing the conservation laws")
        damping, norm = 1.0, np.linalg.norm(gap)
        while True:
            trial = evaluate(point + damping * step)
            if np.all(np.isfinite(trial[2])) and np.linalg.norm(trial[2]) < (1.0 - 1e-4 * damping) * norm:
                break
            damping *= 0.5
            if damping < 1e-12:
                raise ValueError("the conservation laws could not be imposed")
        point = point + damping * step
        state, conserved, gap = trial
    if np.max(np.abs(conserved / target - 1.0)) > 1e-12:
        raise ValueError("the conservation laws could not be imposed")
    return state

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases() -> list[dict[str, str]]:
    """Return scalar-only test specifications."""
    reducer = (
        "import numpy as np\n"
        "def _fsig(a, n):\n"
        "    a = np.asarray(a, dtype=float)\n"
        "    if a.shape != (n,): return -1.0\n"
        "    w = np.cos(np.arange(n) + 1.0)\n"
        "    return float(np.sum(np.abs(a)) + np.sum(a * w))\n"
    )
    status = (
        "def _status(fn):\n"
        "    try:\n"
        "        fn()\n"
        "        return 0\n"
        "    except ValueError:\n"
        "        return 1\n"
        "    except Exception:\n"
        "        return 2\n"
    )
    binding = (
        "K = 2.7; T = np.array([1.9, 1.3])\n"
        "off = np.array([0.0, 0.0, np.log(K)])\n"
        "ex = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])\n"
        "W = np.array([[1.0, 0.0, 1.0], [0.0, 1.0, 1.0]])\n"
        "c = ((K * T.sum() + 1.0) - np.sqrt((K * T.sum() + 1.0) ** 2 - 4.0 * K * K * T[0] * T[1])) / (2.0 * K)\n"
    )
    relay = (
        "off = np.log(np.array([0.61, 1.34, 0.27, 0.88, 1.0, 2.05, 0.43, 1.0, 1.72]))\n"
        "ex = np.array([[1, 0], [1, 0], [1, 0], [1, 0], [1, -1], [0, 0], [1, 0], [1, 0], [0, 1]], dtype=float)\n"
        "W = np.array([[1, 1, 1, 1, 1, 0, 1, 1, 0], [0, 0, 1, 0, 0, 1, 1, 1, 1]], dtype=float)\n"
    )
    return [
        {
            "setup": reducer + binding,
            "call": "_fsig(solve_class_steady_state(off, ex, W, T), 3)",
            "gold_call": "_fsig([T[0] - c, T[1] - c, c], 3)",
        },
        {
            "setup": reducer + relay + "T = np.array([2.6, 3.4])\n",
            "call": "_fsig(solve_class_steady_state(off, ex, W, T), 9)",
            "gold_call": "_fsig(_oracle_solve_class_steady_state(off, ex, W, T), 9)",
        },
        {
            "setup": reducer + relay + "T = np.array([0.35, 5.2])\n",
            "call": "float(solve_class_steady_state(off, ex, W, T)[7])",
            "gold_call": "float(_oracle_solve_class_steady_state(off, ex, W, T)[7])",
        },
        {
            "setup": reducer + "off = np.log(np.array([1.92, 0.64, 1.78])); ex = np.zeros((3, 0)); W = np.zeros((0, 3)); T = np.zeros(0)\n",
            "call": "_fsig(solve_class_steady_state(off, ex, W, T), 3)",
            "gold_call": "_fsig([1.92, 0.64, 1.78], 3)",
        },
        {
            "setup": reducer + (
                "off = np.log(np.array([1.0, 0.5, 3.0, 0.2]))\n"
                "ex = np.array([[1.0], [2.0], [0.0], [1.0]])\n"
                "W = np.array([[1.0, 1.0, 0.0, 2.0]]); T = np.array([4.4])\n"
            ),
            "call": "_fsig(solve_class_steady_state(off, ex, W, T), 4)",
            "gold_call": "_fsig(_oracle_solve_class_steady_state(off, ex, W, T), 4)",
        },
        {
            "setup": reducer + status + binding + "T = np.array([1.9, 0.0])\n",
            "call": "_status(lambda: solve_class_steady_state(off, ex, W, T))",
            "gold_call": "_status(lambda: _oracle_solve_class_steady_state(off, ex, W, T))",
        },
        {
            "setup": reducer + status + binding + "W = W[:1]; T = T[:1]\n",
            "call": "_status(lambda: solve_class_steady_state(off, ex, W, T))",
            "gold_call": "_status(lambda: _oracle_solve_class_steady_state(off, ex, W, T))",
        },
    ]
