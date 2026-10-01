"""
Precompute, once and offline, the Gram matrix of the column blocks that span the approximate residual and its reduced Jacobian.

Because the lifted right-hand side is polynomial of degree two, every

full-order object the reduced Newton solver needs lies in the span of a fixed

set of columns built from the trial basis. Their inner products can therefore

be formed before the time march, and the online stage never touches the

full-order dimension.

Returns
-------
np.ndarray, the symmetric Gram matrix of shape (d, d) with d = 2 * n + n**2 + n_u * n + 1 + n_u.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

from numbers import Real

import math
import numpy as np


def hrf_operator_gram(
    trial_basis: np.ndarray,
    constant_operator: np.ndarray,
    linear_operator: np.ndarray,
    quadratic_operator: np.ndarray,
    input_operator: np.ndarray,
    bilinear_operator: np.ndarray,
) -> np.ndarray:
    """Form the offline Gram matrix of the reduced-space column blocks.

    Let Phi be the trial basis with N rows and n columns, let n_u be the number
    of inputs, and let C, A, F, B and N_op be the constant, linear, quadratic,
    input and bilinear operators of the polynomial system

        dx/dt = C + A x + F (x kron x) + B u + N_op (u kron x).

    Define the column-block matrix

        K = [ Phi | A Phi | F (Phi kron Phi) | N_op (I_{n_u} kron Phi) | C | B ]

    with the six blocks in that order and with C contributing a single column.
    Return the Gram matrix K^T K. The Kronecker layouts are numpy's, matching
    the column conventions of F and N_op.

    Parameters
    ----------
    trial_basis : np.ndarray
        Finite real array of shape (N, n) with N >= 1 and n >= 1.
    constant_operator : np.ndarray
        Finite real array of shape (N,).
    linear_operator : np.ndarray
        Finite real array of shape (N, N).
    quadratic_operator : np.ndarray
        Finite real array of shape (N, N**2).
    input_operator : np.ndarray
        Finite real array of shape (N, n_u) with n_u >= 1.
    bilinear_operator : np.ndarray
        Finite real array of shape (N, n_u * N).

    Returns
    -------
    gram : np.ndarray
        Float array of shape (d, d) with d = 2 * n + n**2 + n_u * n + 1 + n_u.

    Raises
    ------
    ValueError
        If any argument has the wrong rank, if the shapes are mutually
        inconsistent, or if any entry is not finite.
    """
    return np.empty((0, 0), dtype=float)

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_hrf_operator_gram(
    trial_basis: np.ndarray,
    constant_operator: np.ndarray,
    linear_operator: np.ndarray,
    quadratic_operator: np.ndarray,
    input_operator: np.ndarray,
    bilinear_operator: np.ndarray,
) -> np.ndarray:
    """Reference implementation."""
    from numbers import Real

    import math
    import numpy as np

    def _finite(name: str, values, ndim: int) -> np.ndarray:
        array = np.asarray(values, dtype=float)
        if array.ndim != ndim:
            raise ValueError(f"{name} must have ndim={ndim}")
        if not np.all(np.isfinite(array)):
            raise ValueError(f"{name} entries must be finite")
        return array

    basis = _finite("trial_basis", trial_basis, 2)
    constant = _finite("constant_operator", constant_operator, 1)
    linear = _finite("linear_operator", linear_operator, 2)
    quadratic = _finite("quadratic_operator", quadratic_operator, 2)
    inputs = _finite("input_operator", input_operator, 2)
    bilinear = _finite("bilinear_operator", bilinear_operator, 2)

    full, reduced = basis.shape
    if full < 1 or reduced < 1:
        raise ValueError("trial_basis must have positive extents")
    if constant.shape[0] != full:
        raise ValueError("constant_operator must have one entry per full-order row")
    if linear.shape != (full, full):
        raise ValueError("linear_operator must be square with the full-order size")
    if quadratic.shape != (full, full * full):
        raise ValueError("quadratic_operator must have full**2 columns")
    if inputs.shape[0] != full or inputs.shape[1] < 1:
        raise ValueError("input_operator must have one row per full-order row")
    n_inputs = inputs.shape[1]
    if bilinear.shape != (full, n_inputs * full):
        raise ValueError("bilinear_operator must have n_inputs * full columns")

    quadratic_columns = quadratic @ np.kron(basis, basis)
    bilinear_columns = bilinear @ np.kron(np.eye(n_inputs, dtype=float), basis)
    blocks = np.concatenate(
        [
            basis,
            linear @ basis,
            quadratic_columns,
            bilinear_columns,
            constant.reshape(full, 1),
            inputs,
        ],
        axis=1,
    )
    gram = blocks.T @ blocks
    return np.asarray(gram, dtype=float)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases() -> list[dict[str, str]]:
    """Return a list of test case specifications."""
    reducer = (
        "import numpy as np\n"
        "def _sig(value, scale):\n"
        "    a = np.asarray(value, dtype=float)\n"
        "    b = np.concatenate([np.asarray([a.ndim, *a.shape], dtype=float), a.ravel()])\n"
        "    k = np.arange(1.0, b.size + 1.0)\n"
        "    return float((np.sum(np.abs(b)) + np.sum(b * np.cos(k))) / scale)\n"
        "def _ops(full, red, n_u, seed):\n"
        "    rng = np.random.default_rng(seed)\n"
        "    basis, _ = np.linalg.qr(rng.standard_normal((full, red)))\n"
        "    c = rng.standard_normal(full)\n"
        "    a = rng.standard_normal((full, full))\n"
        "    f = np.zeros((full, full * full))\n"
        "    for r in range(full):\n"
        "        for p in range(full):\n"
        "            for q in range(p, full):\n"
        "                if (r + p + q) % 3 == 0:\n"
        "                    f[r, p * full + q] = rng.standard_normal()\n"
        "    b = rng.standard_normal((full, n_u))\n"
        "    nn = rng.standard_normal((full, n_u * full))\n"
        "    return basis, c, a, f, b, nn\n"
    )
    return [
        # Case 1: a small non-symmetric quadratic operator with two inputs.
        {
            "setup": reducer + "basis, c, a, f, b, nn = _ops(9, 3, 2, 4)\n",
            "call": "_sig(hrf_operator_gram(basis, c, a, f, b, nn), 1e2)",
            "gold_call": "_sig(_oracle_hrf_operator_gram(basis, c, a, f, b, nn), 1e2)",
        },
        # Case 2: a single reduced mode and a single input.
        {
            "setup": reducer + "basis, c, a, f, b, nn = _ops(6, 1, 1, 11)\n",
            "call": "_sig(hrf_operator_gram(basis, c, a, f, b, nn), 1e1)",
            "gold_call": "_sig(_oracle_hrf_operator_gram(basis, c, a, f, b, nn), 1e1)",
        },
        # Case 3: a wider reduced space, where the quadratic block dominates the
        #     Gram dimension.
        {
            "setup": reducer + "basis, c, a, f, b, nn = _ops(12, 4, 3, 21)\n",
            "call": "_sig(hrf_operator_gram(basis, c, a, f, b, nn), 1e3)",
            "gold_call": "_sig(_oracle_hrf_operator_gram(basis, c, a, f, b, nn), 1e3)",
        },
        # --- Decisive: the six blocks must appear in the stated order, so the
        #     named sub-blocks of the Gram matrix must reproduce the individual
        #     reduced operators. A permuted layout still yields a symmetric
        #     positive semi-definite matrix but fails here.
        {
            "setup": reducer + (
                "def layout():\n"
                "    basis, c, a, f, b, nn = _ops(9, 3, 2, 4)\n"
                "    g = hrf_operator_gram(basis, c, a, f, b, nn)\n"
                "    n, n_u = basis.shape[1], b.shape[1]\n"
                "    d = 2 * n + n * n + n_u * n + 1 + n_u\n"
                "    if g.shape != (d, d):\n"
                "        return -1\n"
                "    o = [0, n, 2 * n, 2 * n + n * n, 2 * n + n * n + n_u * n, d - n_u, d]\n"
                "    ok = int(np.max(np.abs(g[o[0]:o[1], o[0]:o[1]] - basis.T @ basis)) < 1e-10)\n"
                "    ok += 2 * int(np.max(np.abs(g[o[0]:o[1], o[1]:o[2]] - basis.T @ a @ basis)) < 1e-10)\n"
                "    ok += 4 * int(np.max(np.abs(g[o[0]:o[1], o[2]:o[3]]\n"
                "                                - basis.T @ (f @ np.kron(basis, basis)))) < 1e-10)\n"
                "    ok += 8 * int(np.max(np.abs(g[o[0]:o[1], o[3]:o[4]]\n"
                "                                - basis.T @ (nn @ np.kron(np.eye(n_u), basis)))) < 1e-10)\n"
                "    ok += 16 * int(np.max(np.abs(g[o[0]:o[1], o[4]] - basis.T @ c)) < 1e-10)\n"
                "    ok += 32 * int(np.max(np.abs(g[o[0]:o[1], o[5]:o[6]] - basis.T @ b)) < 1e-10)\n"
                "    return ok\n"
            ),
            "call": "layout()",
            "gold_call": "63",
        },
        # --- Decisive: the bilinear block must be contracted with the input
        #     identity on the left of the basis, so swapping the Kronecker
        #     factors changes the Gram matrix.
        {
            "setup": reducer + (
                "def bilinear_side():\n"
                "    basis, c, a, f, b, nn = _ops(8, 2, 2, 33)\n"
                "    g = hrf_operator_gram(basis, c, a, f, b, nn)\n"
                "    n, n_u = basis.shape[1], b.shape[1]\n"
                "    o3, o4 = 2 * n + n * n, 2 * n + n * n + n_u * n\n"
                "    right = basis.T @ (nn @ np.kron(np.eye(n_u), basis))\n"
                "    wrong = basis.T @ (nn @ np.kron(basis, np.eye(n_u)))\n"
                "    got = g[:n, o3:o4]\n"
                "    return (int(np.max(np.abs(got - right)) < 1e-10)\n"
                "            + 2 * int(np.max(np.abs(right - wrong)) > 1e-8))\n"
            ),
            "call": "bilinear_side()",
            "gold_call": "3",
        },
        # Case 6: invalid quadratic operator width.
        {
            "setup": reducer + (
                "basis, c, a, f, b, nn = _ops(6, 2, 2, 5)\n"
                "fbad = f[:, :-1]\n"
                "def run_model():\n"
                "    try:\n"
                "        hrf_operator_gram(basis, c, a, fbad, b, nn)\n"
                "        return 0\n"
                "    except ValueError:\n"
                "        return 1\n"
                "    except Exception:\n"
                "        return 2\n"
                "def run_gold():\n"
                "    try:\n"
                "        _oracle_hrf_operator_gram(basis, c, a, fbad, b, nn)\n"
                "        return 0\n"
                "    except ValueError:\n"
                "        return 1\n"
                "    except Exception:\n"
                "        return 2"
            ),
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # Case 7: invalid bilinear operator width.
        {
            "setup": reducer + (
                "basis, c, a, f, b, nn = _ops(6, 2, 2, 5)\n"
                "nbad = nn[:, :-2]\n"
                "def run_model():\n"
                "    try:\n"
                "        hrf_operator_gram(basis, c, a, f, b, nbad)\n"
                "        return 0\n"
                "    except ValueError:\n"
                "        return 1\n"
                "    except Exception:\n"
                "        return 2\n"
                "def run_gold():\n"
                "    try:\n"
                "        _oracle_hrf_operator_gram(basis, c, a, f, b, nbad)\n"
                "        return 0\n"
                "    except ValueError:\n"
                "        return 1\n"
                "    except Exception:\n"
                "        return 2"
            ),
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
    ]
