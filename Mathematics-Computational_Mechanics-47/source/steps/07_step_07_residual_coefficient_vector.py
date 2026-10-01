"""
Express the approximate residual of a linear multistep scheme as a coefficient vector against the precomputed column blocks.

Writing the residual in the offline column basis removes the full-order

dimension from the online stage because the residual is then represented by

quantities that depend only on the reduced state and the input.

Returns
-------
np.ndarray, the coefficient vector of shape (d,) with d = 2 * n + n**2 + n_u * n + 1 + n_u.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

from numbers import Real

import math
import numpy as np


def residual_coefficient_vector(
    reduced_history: np.ndarray,
    input_history: np.ndarray,
    time_step: float,
    alphas: np.ndarray,
    betas: np.ndarray,
) -> np.ndarray:
    """Return the coefficients of the approximate residual in the column basis.

    Let Phi be the trial basis, let the polynomial right-hand side be

        f(x, u) = C + A x + F (x kron x) + B u + N_op (u kron x),

    and let the residual of the linear multistep scheme at step m be

        r = sum_j alphas[j] * Phi xhat_{m-j}
            - time_step * sum_j betas[j] * f(Phi xhat_{m-j}, u_{m-j}),

    where row j of reduced_history holds xhat_{m-j} and row j of input_history
    holds u_{m-j}. Return the vector c for which r = K c, where

        K = [ Phi | A Phi | F (Phi kron Phi) | N_op (I_{n_u} kron Phi) | C | B ]

    is the same six-block matrix, in the same order and with the same numpy
    Kronecker layouts, whose Gram matrix is precomputed offline.

    Parameters
    ----------
    reduced_history : np.ndarray
        Finite real array of shape (tau + 1, n) with tau >= 0 and n >= 1.
    input_history : np.ndarray
        Finite real array of shape (tau + 1, n_u) with n_u >= 1.
    time_step : float
        Finite strictly positive step size.
    alphas : np.ndarray
        Finite real array of shape (tau + 1,) holding the state coefficients.
    betas : np.ndarray
        Finite real array of shape (tau + 1,) holding the rate coefficients.

    Returns
    -------
    coefficients : np.ndarray
        Float array of shape (d,) with d = 2 * n + n**2 + n_u * n + 1 + n_u.

    Raises
    ------
    ValueError
        If any array has the wrong rank, if the histories and coefficient
        arrays disagree in length, if any entry is not finite, or if time_step
        is not finite and positive.
    """
    return np.empty(0, dtype=float)

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_residual_coefficient_vector(
    reduced_history: np.ndarray,
    input_history: np.ndarray,
    time_step: float,
    alphas: np.ndarray,
    betas: np.ndarray,
) -> np.ndarray:
    """Reference implementation."""
    from numbers import Real

    import math
    import numpy as np

    def _is_number(value) -> bool:
        return isinstance(value, Real) and not isinstance(value, bool)

    def _finite(name: str, values, ndim: int) -> np.ndarray:
        array = np.asarray(values, dtype=float)
        if array.ndim != ndim:
            raise ValueError(f"{name} must have ndim={ndim}")
        if not np.all(np.isfinite(array)):
            raise ValueError(f"{name} entries must be finite")
        return array

    states = _finite("reduced_history", reduced_history, 2)
    controls = _finite("input_history", input_history, 2)
    state_weights = _finite("alphas", alphas, 1)
    rate_weights = _finite("betas", betas, 1)
    if states.shape[0] < 1 or states.shape[1] < 1:
        raise ValueError("reduced_history must have positive extents")
    if controls.shape[0] != states.shape[0] or controls.shape[1] < 1:
        raise ValueError("input_history must have one row per stored step")
    if state_weights.shape[0] != states.shape[0] or rate_weights.shape[0] != states.shape[0]:
        raise ValueError("alphas and betas must have one entry per stored step")
    if not _is_number(time_step) or not math.isfinite(float(time_step)) or float(time_step) <= 0.0:
        raise ValueError("time_step must be a finite positive real scalar")

    step = float(time_step)
    n_reduced = states.shape[1]
    n_inputs = controls.shape[1]
    quadratic = n_reduced * n_reduced
    bilinear = n_inputs * n_reduced
    total = 2 * n_reduced + quadratic + bilinear + 1 + n_inputs

    first = n_reduced
    second = 2 * n_reduced
    third = second + quadratic
    fourth = third + bilinear

    coefficients = np.zeros(total, dtype=float)
    for index in range(states.shape[0]):
        state = states[index]
        control = controls[index]
        weight = float(state_weights[index])
        rate = -step * float(rate_weights[index])
        coefficients[:first] += weight * state
        coefficients[first:second] += rate * state
        coefficients[second:third] += rate * np.kron(state, state)
        coefficients[third:fourth] += rate * np.kron(control, state)
        coefficients[fourth] += rate
        coefficients[fourth + 1:] += rate * control
    return coefficients

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
        "def _cols(ops):\n"
        "    basis, c, a, f, b, nn = ops\n"
        "    n_u = b.shape[1]\n"
        "    return np.concatenate([basis, a @ basis, f @ np.kron(basis, basis),\n"
        "                           nn @ np.kron(np.eye(n_u), basis),\n"
        "                           c.reshape(-1, 1), b], axis=1)\n"
    )
    return [
        # Case 1: backward Euler with two inputs.
        {
            "setup": reducer + (
                "hist = np.array([[0.4, -1.2, 0.7], [0.35, -1.1, 0.65]])\n"
                "uhist = np.array([[0.6, -0.2], [0.5, -0.15]])\n"
                "alphas = np.array([1.0, -1.0])\n"
                "betas = np.array([1.0, 0.0])\n"
                "dt = 0.01\n"
            ),
            "call": "_sig(residual_coefficient_vector(hist, uhist, dt, alphas, betas), 1e0)",
            "gold_call": "_sig(_oracle_residual_coefficient_vector(hist, uhist, dt, alphas, betas), 1e0)",
        },
        # Case 2: Crank-Nicolson, where the preceding level also contributes to
        #     the rate sum.
        {
            "setup": reducer + (
                "hist = np.array([[0.4, -1.2, 0.7], [0.35, -1.1, 0.65]])\n"
                "uhist = np.array([[0.6, -0.2], [0.5, -0.15]])\n"
                "alphas = np.array([1.0, -1.0])\n"
                "betas = np.array([0.5, 0.5])\n"
                "dt = 0.02\n"
            ),
            "call": "_sig(residual_coefficient_vector(hist, uhist, dt, alphas, betas), 1e0)",
            "gold_call": "_sig(_oracle_residual_coefficient_vector(hist, uhist, dt, alphas, betas), 1e0)",
        },
        # Case 3: a three-level scheme with a single reduced mode.
        {
            "setup": reducer + (
                "hist = np.array([[0.9], [0.8], [0.6]])\n"
                "uhist = np.array([[1.0], [0.5], [0.25]])\n"
                "alphas = np.array([1.5, -2.0, 0.5])\n"
                "betas = np.array([1.0, 0.0, 0.0])\n"
                "dt = 0.05\n"
            ),
            "call": "_sig(residual_coefficient_vector(hist, uhist, dt, alphas, betas), 1e0)",
            "gold_call": "_sig(_oracle_residual_coefficient_vector(hist, uhist, dt, alphas, betas), 1e0)",
        },
        # --- Decisive: contracting the coefficients with the column blocks must
        #     reproduce the residual assembled directly from the full-order
        #     operators, which pins the sign, the placement of the step size and
        #     the assignment of every term to its block.
        {
            "setup": reducer + (
                "def reconstruct():\n"
                "    ops = _ops(7, 3, 2, 23)\n"
                "    basis, c, a, f, b, nn = ops\n"
                "    cols = _cols(ops)\n"
                "    hist = np.array([[0.4, -1.2, 0.7], [0.35, -1.1, 0.65]])\n"
                "    uhist = np.array([[0.6, -0.2], [0.5, -0.15]])\n"
                "    alphas = np.array([1.0, -1.0])\n"
                "    betas = np.array([0.5, 0.5])\n"
                "    dt = 0.02\n"
                "    coeff = residual_coefficient_vector(hist, uhist, dt, alphas, betas)\n"
                "    direct = np.zeros(basis.shape[0])\n"
                "    for j in range(2):\n"
                "        st = basis @ hist[j]\n"
                "        u = uhist[j]\n"
                "        rate = c + a @ st + f @ np.kron(st, st) + b @ u + nn @ np.kron(u, st)\n"
                "        direct = direct + alphas[j] * st - dt * betas[j] * rate\n"
                "    return int(np.max(np.abs(cols @ coeff - direct)) < 1e-10)\n"
            ),
            "call": "reconstruct()",
            "gold_call": "1",
        },
        # --- Decisive: with every rate weight zero the residual is the pure
        #     state combination, so only the first block may be populated.
        {
            "setup": reducer + (
                "def explicit_split():\n"
                "    hist = np.array([[0.4, -1.2, 0.7], [0.35, -1.1, 0.65]])\n"
                "    uhist = np.array([[0.6, -0.2], [0.5, -0.15]])\n"
                "    coeff = residual_coefficient_vector(hist, uhist, 0.01,\n"
                "                                        np.array([1.0, -1.0]), np.array([0.0, 0.0]))\n"
                "    head = coeff[:3]\n"
                "    tail = coeff[3:]\n"
                "    want = hist[0] - hist[1]\n"
                "    return (int(np.max(np.abs(head - want)) < 1e-12)\n"
                "            + 2 * int(np.max(np.abs(tail)) < 1e-12))\n"
            ),
            "call": "explicit_split()",
            "gold_call": "3",
        },
        # Case 6: invalid history lengths.
        {
            "setup": reducer + (
                "hist = np.array([[0.4, -1.2], [0.35, -1.1]])\n"
                "uhist = np.array([[0.6, -0.2]])\n"
                "def run_model():\n"
                "    try:\n"
                "        residual_coefficient_vector(hist, uhist, 0.01, np.array([1.0, -1.0]), np.array([1.0, 0.0]))\n"
                "        return 0\n"
                "    except ValueError:\n"
                "        return 1\n"
                "    except Exception:\n"
                "        return 2\n"
                "def run_gold():\n"
                "    try:\n"
                "        _oracle_residual_coefficient_vector(hist, uhist, 0.01, np.array([1.0, -1.0]), np.array([1.0, 0.0]))\n"
                "        return 0\n"
                "    except ValueError:\n"
                "        return 1\n"
                "    except Exception:\n"
                "        return 2"
            ),
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # Case 7: invalid non-positive step size.
        {
            "setup": reducer + (
                "hist = np.array([[0.4, -1.2], [0.35, -1.1]])\n"
                "uhist = np.array([[0.6, -0.2], [0.5, -0.15]])\n"
                "def run_model():\n"
                "    try:\n"
                "        residual_coefficient_vector(hist, uhist, -0.01, np.array([1.0, -1.0]), np.array([1.0, 0.0]))\n"
                "        return 0\n"
                "    except ValueError:\n"
                "        return 1\n"
                "    except Exception:\n"
                "        return 2\n"
                "def run_gold():\n"
                "    try:\n"
                "        _oracle_residual_coefficient_vector(hist, uhist, -0.01, np.array([1.0, -1.0]), np.array([1.0, 0.0]))\n"
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
