"""
Evaluate the parametrised amplitude from its series coefficients and its pole product.

The parametrisation is the product of an explicit pole factor with a truncated power series in the conformal variable, so one evaluation rule serves every Riemann sheet the variable covers.

Returns
-------
np.ndarray: complex values of the parametrised amplitude, one per point.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def evaluate_series_form_factor(
    points: "np.ndarray",
    coefficients: "np.ndarray",
    pole_points: "np.ndarray",
    factor_fn: "Callable[..., np.ndarray]",
) -> "np.ndarray":
    """Return the parametrised amplitude at each point of the expansion variable.

    The parametrisation is

        ``F(w) = factor_fn(w, pole_points) * sum_k coefficients[k] * w**k``

    evaluated at every entry of ``points``. Any point of the expansion
    variable is accepted, inside or outside the unit disc.

    Parameters
    ----------
    points : np.ndarray
        One-dimensional non-empty array of finite complex points.
    coefficients : np.ndarray
        One-dimensional non-empty array of finite real coefficients, in
        order of increasing power.
    pole_points : np.ndarray
        One-dimensional array of pole images passed on to ``factor_fn``.
    factor_fn : callable
        Pole product following the contract of ``evaluate_pole_factor``.

    Returns
    -------
    np.ndarray
        Complex array with the same length as ``points``.

    Raises
    ------
    ValueError
        If ``points`` or ``coefficients`` is not a one-dimensional
        non-empty array of finite numbers, if ``coefficients`` is not real,
        if ``factor_fn`` is not callable or returns the wrong shape, or if
        the result is not finite.
    """
    return amplitude

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
def _oracle_evaluate_series_form_factor(
    points: "np.ndarray",
    coefficients: "np.ndarray",
    pole_points: "np.ndarray",
    factor_fn: "Callable[..., np.ndarray]",
) -> "np.ndarray":
    """Reference implementation (Horner evaluation times the pole product)."""
    import numpy as np

    def _is_function(value):
        return callable(value)

    if not _is_function(factor_fn):
        raise ValueError("factor_fn must be callable")
    try:
        grid = np.atleast_1d(np.asarray(points, dtype=complex))
        weights = np.atleast_1d(np.asarray(coefficients, dtype=complex))
    except (TypeError, ValueError):
        raise ValueError("points and coefficients must be arrays of numbers") from None
    if grid.ndim != 1 or grid.size < 1:
        raise ValueError("points must be a non-empty one-dimensional array")
    if weights.ndim != 1 or weights.size < 1:
        raise ValueError("coefficients must be a non-empty one-dimensional array")
    if not np.all(np.isfinite(grid.real) & np.isfinite(grid.imag)):
        raise ValueError("points must contain only finite numbers")
    if not np.all(np.isfinite(weights.real) & np.isfinite(weights.imag)):
        raise ValueError("coefficients must contain only finite numbers")
    if np.any(weights.imag != 0.0):
        raise ValueError("coefficients must be real")

    series = np.zeros(grid.size, dtype=complex)
    for weight in weights[::-1]:
        series = series * grid + weight

    factors = np.atleast_1d(np.asarray(factor_fn(grid, pole_points), dtype=complex))
    if factors.shape != grid.shape:
        raise ValueError("factor_fn must return one value per point")
    result = factors * series
    if not np.all(np.isfinite(result.real) & np.isfinite(result.imag)):
        raise ValueError("the parametrisation is not finite at a requested point")
    return result

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases() -> list[dict]:
    """Return the submitted cases with independent dependency chains and ordered outputs."""
    setups = ('import numpy as np\n'
     'def _scal(value):\n'
     '    arr = np.atleast_1d(np.asarray(value, dtype=complex)).ravel()\n'
     '    idx = np.arange(1.0, arr.size + 1.0)\n'
     '    return float(np.sum(np.abs(arr))\n'
     '                 + np.sum(arr.real * np.cos(idx))\n'
     '                 + np.sum(arr.imag * np.sin(idx)))\n'
     'POLES = np.array([-0.5572560361319145 - 0.44187389987867753j,\n'
     '                  -0.3692504531926694 - 0.8465459320922136j,\n'
     '                  -0.1917341775985101 - 1.2396284234993662j,\n'
     '                  -0.1791524758812030 - 2.4434970464164243j])\n'
     'COEF = np.array([3.523398244959, 5.019969089918, 4.718082454754,\n'
     '                 4.091297724846, 8.639350092120, -6.085466044387,\n'
     '                 -33.252445374530, -19.562798556883, -9.416653502022,\n'
     '                 42.325265871225])\n'
     'GRID = np.array([0.3189343369335683 + 0.0j,\n'
     '                 -0.3687232531791833 - 0.3027516319622959j,\n'
     '                 -0.2619056718872012 - 0.7058136487190447j])\n'
     'def _factor(points, poles):\n'
     '    return evaluate_pole_factor(points, poles)\n'
     'def _toy_factor(points, poles):\n'
     '    arr = np.asarray(points, dtype=complex)\n'
     '    return (1.0 + 0.5 * np.size(poles)) / (3.0 + arr)\n'
     '\n'
     'def _reference_factor(points, poles):\n'
     '    return _oracle_evaluate_pole_factor(points, poles)\n'
     '\n'
     'import copy\n'
     'def _independent_inputs(fn, *args, **kwargs):\n'
     '    return fn(*copy.deepcopy(args), **copy.deepcopy(kwargs))\n',
     'import numpy as np\n'
     'def _scal(value):\n'
     '    arr = np.atleast_1d(np.asarray(value, dtype=complex)).ravel()\n'
     '    idx = np.arange(1.0, arr.size + 1.0)\n'
     '    return float(np.sum(np.abs(arr))\n'
     '                 + np.sum(arr.real * np.cos(idx))\n'
     '                 + np.sum(arr.imag * np.sin(idx)))\n'
     'POLES = np.array([-0.5572560361319145 - 0.44187389987867753j,\n'
     '                  -0.3692504531926694 - 0.8465459320922136j,\n'
     '                  -0.1917341775985101 - 1.2396284234993662j,\n'
     '                  -0.1791524758812030 - 2.4434970464164243j])\n'
     'COEF = np.array([3.523398244959, 5.019969089918, 4.718082454754,\n'
     '                 4.091297724846, 8.639350092120, -6.085466044387,\n'
     '                 -33.252445374530, -19.562798556883, -9.416653502022,\n'
     '                 42.325265871225])\n'
     'GRID = np.array([0.3189343369335683 + 0.0j,\n'
     '                 -0.3687232531791833 - 0.3027516319622959j,\n'
     '                 -0.2619056718872012 - 0.7058136487190447j])\n'
     'def _factor(points, poles):\n'
     '    return evaluate_pole_factor(points, poles)\n'
     'def _toy_factor(points, poles):\n'
     '    arr = np.asarray(points, dtype=complex)\n'
     '    return (1.0 + 0.5 * np.size(poles)) / (3.0 + arr)\n'
     'def _status(fn):\n'
     '    try:\n'
     '        fn()\n'
     '        return 0\n'
     '    except ValueError:\n'
     '        return 1\n'
     '    except Exception:\n'
     '        return 2\n'
     'def _short_factor(points, poles):\n'
     '    return np.ones(1, dtype=complex)\n'
     '\n'
     'def _reference_factor(points, poles):\n'
     '    return _oracle_evaluate_pole_factor(points, poles)\n'
     '\n'
     'import copy\n'
     'def _independent_inputs(fn, *args, **kwargs):\n'
     '    return fn(*copy.deepcopy(args), **copy.deepcopy(kwargs))\n')
    return [
        {
            'setup': setups[0],
            'call': '_independent_inputs(evaluate_series_form_factor, GRID, COEF, POLES, _factor)',
            'gold_call': '_independent_inputs(_oracle_evaluate_series_form_factor, GRID, COEF, POLES, _reference_factor)',
        },
        {
            'setup': setups[0],
            'call': '_independent_inputs(evaluate_series_form_factor, np.array([1.0 + 0j, 0.95 + 0j]), COEF, POLES, _factor)',
            'gold_call': '_independent_inputs(_oracle_evaluate_series_form_factor, np.array([1.0 + 0j, 0.95 + 0j]), COEF, POLES, _reference_factor)',
        },
        {
            'setup': setups[0],
            'call': '_independent_inputs(evaluate_series_form_factor, np.array([1.8 - 1.1j, -2.0 + 0.3j]), COEF, POLES, _factor)',
            'gold_call': '_independent_inputs(_oracle_evaluate_series_form_factor, np.array([1.8 - 1.1j, -2.0 + 0.3j]), COEF, POLES, _reference_factor)',
        },
        {
            'setup': setups[0],
            'call': '_independent_inputs(evaluate_series_form_factor, GRID, np.array([2.5]), POLES, _factor)',
            'gold_call': '_independent_inputs(_oracle_evaluate_series_form_factor, GRID, np.array([2.5]), POLES, _reference_factor)',
        },
        {
            'setup': setups[0],
            'call': '_independent_inputs(evaluate_series_form_factor, np.array([0.0 + 0j]), COEF, POLES, _factor)',
            'gold_call': '_independent_inputs(_oracle_evaluate_series_form_factor, np.array([0.0 + 0j]), COEF, POLES, _reference_factor)',
        },
        {
            'setup': setups[0],
            'call': '_independent_inputs(evaluate_series_form_factor, GRID, COEF, POLES[:3], _toy_factor)',
            'gold_call': '_independent_inputs(_oracle_evaluate_series_form_factor, GRID, COEF, POLES[:3], _toy_factor)',
        },
        {
            'setup': setups[1],
            'call': '_status(lambda: _independent_inputs(evaluate_series_form_factor, GRID, np.array([1.0 + 0.5j, 2.0]), POLES, _factor))',
            'gold_call': '_status(lambda: _independent_inputs(_oracle_evaluate_series_form_factor, GRID, np.array([1.0 + 0.5j, 2.0]), POLES, _reference_factor))',
        },
        {
            'setup': setups[1],
            'call': '_status(lambda: _independent_inputs(evaluate_series_form_factor, np.array([], dtype=complex), COEF, POLES, _factor))',
            'gold_call': '_status(lambda: _independent_inputs(_oracle_evaluate_series_form_factor, np.array([], dtype=complex), COEF, POLES, _reference_factor))',
        },
        {
            'setup': setups[1],
            'call': '_status(lambda: _independent_inputs(evaluate_series_form_factor, GRID, COEF, POLES, _short_factor))',
            'gold_call': '_status(lambda: _independent_inputs(_oracle_evaluate_series_form_factor, GRID, COEF, POLES, _short_factor))',
        },
    ]
