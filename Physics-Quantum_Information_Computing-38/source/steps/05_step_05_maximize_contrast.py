"""
Refine each grid candidate by bounded Gauss-Newton ascent of the contrast and return the refined local maximizer with the largest contrast as the maximum-contrast estimate.

The estimator maximizes the contrast over a compact parameter box. Its positive-definite plug-in information supplies the Gauss-Newton ascent direction from first derivatives of the averaged drift; it need not equal the physical finite-mesh negative expected Hessian.

Returns
-------
np.ndarray: shape (2,), the maximum-contrast estimate (alpha_hat, beta_hat).
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def maximize_contrast(
    model_fn: "Callable[[float, float], np.ndarray]",
    starts: 'np.ndarray',
    alpha_bounds: tuple,
    beta_bounds: tuple,
    tolerance: float = 1e-10,
) -> 'np.ndarray':
    """Return the maximum-contrast point reached from the starting points.

    ``model_fn(alpha, beta)`` follows the contract of
    ``evaluate_contrast_model``: it returns
    ``[Phi, dPhi/dalpha, dPhi/dbeta, I_aa, I_ab, I_bb]`` with a symmetric
    positive-definite information matrix ``I``. From each start, clipped
    into the box, run bounded Gauss-Newton ascent: at the current point a
    coordinate is held fixed when it lies on a bound and its gradient
    component points out of the box, the free coordinates move along the
    direction ``d`` that solves ``I_ff d_f = g_f`` on the free block, the
    trial point ``theta + s d`` is clipped into the box, and ``s`` is halved
    from 1 until the contrast at the trial point is not below the current
    contrast. The ascent comes to rest at a local maximizer of the contrast
    in the box (every free coordinate has zero gradient there); locate it to
    within ``tolerance`` in each coordinate. Return the located maximizer
    with the largest contrast, the earliest start winning ties.

    Parameters
    ----------
    model_fn : callable
        Function of ``(alpha, beta)`` returning a finite length-6 array.
    starts : np.ndarray
        Finite array of shape ``(m, 2)`` with ``m >= 1`` starting points.
    alpha_bounds, beta_bounds : tuple
        ``(low, high)`` with finite ``low < high``.
    tolerance : float
        Positive accuracy of each located maximizer.

    Returns
    -------
    np.ndarray
        Float array ``[alpha_hat, beta_hat]`` of shape ``(2,)``.

    Raises
    ------
    ValueError
        If ``model_fn`` is not callable or returns anything other than a
        finite length-6 array, the information matrix on the free block is
        not positive definite, ``starts`` is not a finite ``(m, 2)`` array
        with ``m >= 1``, a bound pair is not two finite numbers with
        ``low < high``, or ``tolerance`` is not a finite positive number.
    """
    return theta_hat

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
def _oracle_maximize_contrast(
    model_fn: "Callable[[float, float], np.ndarray]",
    starts: 'np.ndarray',
    alpha_bounds: tuple,
    beta_bounds: tuple,
    tolerance: float = 1e-10,
) -> 'np.ndarray':
    """Reference implementation (bounded Gauss-Newton ascent, then Newton polish)."""
    import numpy as np

    def _is_number(value):
        return (not isinstance(value, bool)
                and isinstance(value, (int, float, np.integer, np.floating))
                and bool(np.isfinite(value)))

    def _is_function(value):
        return callable(value)

    def _pair(bounds, name):
        try:
            low, high = bounds
        except (TypeError, ValueError):
            raise ValueError(f"{name} must hold two numbers") from None
        if not (_is_number(low) and _is_number(high) and low < high):
            raise ValueError(f"{name} must be finite with low < high")
        return float(low), float(high)

    if not _is_function(model_fn):
        raise ValueError("model_fn must be callable")
    if not (_is_number(tolerance) and tolerance > 0.0):
        raise ValueError("tolerance must be a finite positive number")
    (a_lo, a_hi), (b_lo, b_hi) = _pair(alpha_bounds, "alpha_bounds"), _pair(beta_bounds, "beta_bounds")
    lo, hi = np.array([a_lo, b_lo]), np.array([a_hi, b_hi])
    points = np.asarray(starts, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2 or points.shape[0] == 0 or not np.all(np.isfinite(points)):
        raise ValueError("starts must be a finite (m, 2) array with m >= 1")

    def _eval(p):
        out = np.asarray(model_fn(float(p[0]), float(p[1])), dtype=float)
        if out.shape != (6,) or not np.all(np.isfinite(out)):
            raise ValueError("model_fn must return a finite length-6 array")
        return out[0], out[1:3].copy(), np.array([[out[3], out[4]], [out[4], out[5]]])

    def _free(p, grad):
        return np.flatnonzero(~(((p <= lo) & (grad < 0.0)) | ((p >= hi) & (grad > 0.0))))

    def _gauss_newton(p, grad, info):
        idx, move = _free(p, grad), np.zeros(2)
        if idx.size:
            block = info[np.ix_(idx, idx)]
            if not np.all(np.linalg.eigvalsh(block) > 0.0):
                raise ValueError("information matrix must be positive definite")
            move[idx] = np.linalg.solve(block, grad[idx])
        return move

    def _hessian(p, grad, idx):
        cols = []
        for k in idx:
            h = 1e-6 * max(1.0, abs(p[k]))
            e = np.zeros(2)
            e[k] = h
            if p[k] - h >= lo[k] and p[k] + h <= hi[k]:
                cols.append((_eval(p + e)[1] - _eval(p - e)[1]) / (2.0 * h))
            elif p[k] + h <= hi[k]:
                cols.append((_eval(p + e)[1] - grad) / h)
            else:
                cols.append((grad - _eval(p - e)[1]) / h)
        mat = np.array(cols).T[np.ix_(idx, range(len(idx)))]
        return 0.5 * (mat + mat.T)

    def _ascend(start):
        p = np.clip(start, lo, hi)
        for _ in range(5000):
            value, grad, info = _eval(p)
            move = _gauss_newton(p, grad, info)
            if np.max(np.abs(np.clip(p + move, lo, hi) - p)) <= 1e-7:
                break
            scale = 1.0
            while scale >= 1e-12:
                trial = np.clip(p + scale * move, lo, hi)
                if _eval(trial)[0] >= value:
                    break
                scale *= 0.5
            else:
                break
            p = trial
        # Gauss-Newton converges only linearly here; finish with Newton steps.
        for _ in range(100):
            value, grad, info = _eval(p)
            idx = _free(p, grad)
            if idx.size == 0:
                break
            hess = _hessian(p, grad, idx)
            move = np.zeros(2)
            if np.all(np.linalg.eigvalsh(hess) < 0.0):
                move[idx] = -np.linalg.solve(hess, grad[idx])
            else:
                move = _gauss_newton(p, grad, info)
            new = np.clip(p + move, lo, hi)
            shift, p = np.max(np.abs(new - p)), new
            if shift <= 0.01 * tolerance:
                break
        return p, _eval(p)[0]

    best_point, best_value = None, -np.inf
    for start in points:
        point, value = _ascend(start)
        if value > best_value:
            best_point, best_value = point, value
    return np.asarray(best_point, dtype=float)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases() -> list[dict[str, str]]:
    """Return scalar-only test specifications."""
    helpers = (
        "import numpy as np\n"
        "def _two_bumps(info):\n"
        "    def fn(a, b):\n"
        "        e1 = 2.0 * np.exp(-((a - 1.0) ** 2 + (b - 0.5) ** 2) / 0.5)\n"
        "        e2 = 1.5 * np.exp(-((a + 1.2) ** 2 + (b - 1.0) ** 2) / 0.3)\n"
        "        ga = -e1 * 2.0 * (a - 1.0) / 0.5 - e2 * 2.0 * (a + 1.2) / 0.3\n"
        "        gb = -e1 * 2.0 * (b - 0.5) / 0.5 - e2 * 2.0 * (b - 1.0) / 0.3\n"
        "        return np.array([e1 + e2, ga, gb, info[0], info[1], info[2]])\n"
        "    return fn\n"
        "def _ledge(a, b):\n"
        "    f = -(a - 0.4) ** 2 - 2.0 * (b + 0.2) ** 2 + 0.3 * a * b\n"
        "    return np.array([f, -2.0 * (a - 0.4) + 0.3 * b, -4.0 * (b + 0.2) + 0.3 * a, 2.0, -0.3, 4.0])\n"
        "def _bowl(a, b):\n"
        "    f = -(np.cosh(a - 0.3) + (b - 1.2) ** 2 + 0.4 * (a - 0.3) * (b - 1.2))\n"
        "    ga = -(np.sinh(a - 0.3) + 0.4 * (b - 1.2))\n"
        "    gb = -(2.0 * (b - 1.2) + 0.4 * (a - 0.3))\n"
        "    return np.array([f, ga, gb, 1.0, 0.0, 1.0])\n"
        "def _waves(a, b):\n"
        "    f = 1.2 * b * np.cos(a - 0.5) - 0.4 * b ** 2 + 0.15 * np.sin(2.0 * a)\n"
        "    ga = -1.2 * b * np.sin(a - 0.5) + 0.3 * np.cos(2.0 * a)\n"
        "    gb = 1.2 * np.cos(a - 0.5) - 0.8 * b\n"
        "    return np.array([f, ga, gb, 1.0 + b * b, 0.2, 1.0])\n"
        "def _bad(a, b):\n"
        "    return np.array([0.0, 1.0, 1.0, -1.0, 0.0, 1.0])\n"
        "def _psig(p):\n"
        "    p = np.asarray(p, dtype=float)\n"
        "    if p.shape != (2,):\n"
        "        return -1.0e9\n"
        "    return float(3.0 * p[0] - 2.0 * p[1] + p[0] * p[1])\n"
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
    return [
        {
            "setup": helpers,
            "call": "_psig(maximize_contrast(_two_bumps((2.0, 0.3, 1.5)), np.array([[-1.0, 0.9], [0.8, 0.6]]), (-3.0, 3.0), (0.0, 2.0), 1e-12))",
            "gold_call": "_psig(_oracle_maximize_contrast(_two_bumps((2.0, 0.3, 1.5)), np.array([[-1.0, 0.9], [0.8, 0.6]]), (-3.0, 3.0), (0.0, 2.0), 1e-12))",
        },
        {
            "setup": helpers,
            "call": "_psig(maximize_contrast(_two_bumps((0.5, 0.0, 0.5)), np.array([[-1.1, 1.0]]), (-3.0, 3.0), (0.0, 2.0), 1e-12))",
            "gold_call": "_psig(_oracle_maximize_contrast(_two_bumps((0.5, 0.0, 0.5)), np.array([[-1.1, 1.0]]), (-3.0, 3.0), (0.0, 2.0), 1e-12))",
        },
        {
            "setup": helpers,
            "call": "_psig(maximize_contrast(_ledge, np.array([[0.9, 0.8]]), (-1.0, 1.0), (0.0, 1.0), 1e-12))",
            "gold_call": "_psig(_oracle_maximize_contrast(_ledge, np.array([[0.9, 0.8]]), (-1.0, 1.0), (0.0, 1.0), 1e-12))",
        },
        {
            "setup": helpers,
            "call": "_psig(maximize_contrast(_bowl, np.array([[-2.0, 0.1]]), (-3.0, 3.0), (0.0, 3.0), 1e-12))",
            "gold_call": "_psig(_oracle_maximize_contrast(_bowl, np.array([[-2.0, 0.1]]), (-3.0, 3.0), (0.0, 3.0), 1e-12))",
        },
        {
            "setup": helpers,
            "call": "_psig(maximize_contrast(_waves, np.array([[-2.5, 0.4], [6.5, 1.6], [0.2, 2.9]]), (-4.0, 8.0), (0.05, 3.0), 1e-12))",
            "gold_call": "_psig(_oracle_maximize_contrast(_waves, np.array([[-2.5, 0.4], [6.5, 1.6], [0.2, 2.9]]), (-4.0, 8.0), (0.05, 3.0), 1e-12))",
        },
        {
            "setup": helpers,
            "call": "_psig(maximize_contrast(_bowl, np.array([[5.0, -1.0]]), (-1.0, 1.0), (0.5, 1.0), 1e-12))",
            "gold_call": "_psig(_oracle_maximize_contrast(_bowl, np.array([[5.0, -1.0]]), (-1.0, 1.0), (0.5, 1.0), 1e-12))",
        },
        {
            "setup": helpers + status,
            "call": "_status(lambda: maximize_contrast(_bowl, np.zeros((0, 2)), (-1.0, 1.0), (0.0, 1.0)))",
            "gold_call": "_status(lambda: _oracle_maximize_contrast(_bowl, np.zeros((0, 2)), (-1.0, 1.0), (0.0, 1.0)))",
        },
        {
            "setup": helpers + status,
            "call": "_status(lambda: maximize_contrast(_bad, np.array([[0.2, 0.3]]), (-1.0, 1.0), (0.0, 1.0)))",
            "gold_call": "_status(lambda: _oracle_maximize_contrast(_bad, np.array([[0.2, 0.3]]), (-1.0, 1.0), (0.0, 1.0)))",
        },
    ]
