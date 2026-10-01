"""
Fix the real coefficients of the truncated series from measured values of the amplitude and a prescribed high-energy fall-off.

Measured values below and above threshold and the known high-energy behaviour are linear conditions on the coefficients of a series in the conformal variable, so the truncation can be chosen to make them determine the series uniquely.

Returns
-------
np.ndarray: the real series coefficients in order of increasing power.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def solve_series_coefficients(
    real_points: "np.ndarray",
    real_values: "np.ndarray",
    complex_points: "np.ndarray",
    complex_values: "np.ndarray",
    pole_points: "np.ndarray",
    vanishing_order: int,
    factor_fn: "Callable[..., np.ndarray]",
) -> "np.ndarray":
    """Return the real series coefficients fixed by the measurements and fall-off.

    The parametrisation is ``F(w) = factor_fn(w, pole_points) * g(w)`` with
    ``g(w) = sum_k a_k w**k`` and every ``a_k`` real. Impose

    * ``F`` equal to ``real_values[j]`` at ``real_points[j]``, where both the
      point and the value are real, so each pair is one real condition;
    * ``F`` equal to ``complex_values[j]`` at ``complex_points[j]``, where
      the point and the value are complex, so each pair is two real
      conditions;
    * ``g`` vanishing to order ``vanishing_order`` at ``w = 1``, that is
      ``g`` and its first ``vanishing_order - 1`` derivatives vanish there.

    Truncate ``g`` at the degree for which these conditions determine the
    coefficients uniquely, and return them in order of increasing power.

    Parameters
    ----------
    real_points : np.ndarray
        One-dimensional array of real points, possibly empty.
    real_values : np.ndarray
        Real values of ``F`` at ``real_points``, same length.
    complex_points : np.ndarray
        One-dimensional array of complex points, possibly empty, none of
        them real.
    complex_values : np.ndarray
        Values of ``F`` at ``complex_points``, same length.
    pole_points : np.ndarray
        One-dimensional array of pole images passed on to ``factor_fn``.
    vanishing_order : int
        Number of vanishing conditions imposed at ``w = 1``, at least one.
    factor_fn : callable
        Pole product following the contract of ``evaluate_pole_factor``.

    Returns
    -------
    np.ndarray
        Real coefficients ``[a_0, ..., a_n]``.

    Raises
    ------
    ValueError
        If the arrays are not one-dimensional with matching lengths and
        finite entries, if a real point or value is not real, if a complex
        point is real, if ``vanishing_order`` is not a positive integer, if
        ``factor_fn`` is not callable or returns the wrong shape, if no
        condition is supplied at all, or if the resulting linear system is
        singular.
    """
    return coefficients

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
def _oracle_solve_series_coefficients(
    real_points: "np.ndarray",
    real_values: "np.ndarray",
    complex_points: "np.ndarray",
    complex_values: "np.ndarray",
    pole_points: "np.ndarray",
    vanishing_order: int,
    factor_fn: "Callable[..., np.ndarray]",
) -> "np.ndarray":
    """Reference implementation (square real linear system)."""
    import numpy as np

    def _as_line(value, label):
        try:
            arr = np.atleast_1d(np.asarray(value, dtype=complex))
        except (TypeError, ValueError):
            raise ValueError(f"{label} must be an array of numbers") from None
        if arr.ndim != 1:
            raise ValueError(f"{label} must be one-dimensional")
        if arr.size and not np.all(np.isfinite(arr.real) & np.isfinite(arr.imag)):
            raise ValueError(f"{label} must contain only finite numbers")
        return arr

    def _is_function(value):
        return callable(value)

    def _is_integer(value):
        return (not isinstance(value, bool)
                and isinstance(value, (int, np.integer)))

    if not _is_function(factor_fn):
        raise ValueError("factor_fn must be callable")
    if not _is_integer(vanishing_order):
        raise ValueError("vanishing_order must be an integer")
    order_count = int(vanishing_order)
    if order_count < 1:
        raise ValueError("vanishing_order must be at least one")

    flat_points = _as_line(real_points, "real_points")
    flat_values = _as_line(real_values, "real_values")
    wide_points = _as_line(complex_points, "complex_points")
    wide_values = _as_line(complex_values, "complex_values")
    if flat_points.size != flat_values.size:
        raise ValueError("real_points and real_values must have equal length")
    if wide_points.size != wide_values.size:
        raise ValueError("complex_points and complex_values must have equal length")
    if np.any(flat_points.imag != 0.0) or np.any(flat_values.imag != 0.0):
        raise ValueError("real_points and real_values must be real")
    if np.any(wide_points.imag == 0.0):
        raise ValueError("complex_points must lie off the real axis")

    width = flat_points.size + 2 * wide_points.size + order_count
    if flat_points.size + wide_points.size == 0:
        raise ValueError("at least one measurement is required")

    nodes = np.concatenate([flat_points, wide_points])
    factors = np.atleast_1d(np.asarray(factor_fn(nodes, pole_points), dtype=complex))
    if factors.shape != nodes.shape:
        raise ValueError("factor_fn must return one value per point")
    if np.any(factors == 0.0):
        raise ValueError("the pole product vanishes at a measurement point")

    powers = np.arange(width, dtype=float)
    rows, right = [], []
    for index in range(flat_points.size):
        basis = flat_points[index] ** powers
        rows.append(basis.real)
        right.append(float((flat_values[index] / factors[index]).real))
    for index in range(wide_points.size):
        basis = wide_points[index] ** powers
        target = wide_values[index] / factors[flat_points.size + index]
        rows.append(basis.real); right.append(float(target.real))
        rows.append(basis.imag); right.append(float(target.imag))
    for step in range(order_count):
        falling = np.ones(width, dtype=float)
        for shift in range(step):
            falling = falling * (powers - float(shift))
        rows.append(falling)
        right.append(0.0)

    matrix = np.array(rows, dtype=float)
    vector = np.array(right, dtype=float)
    try:
        coefficients = np.linalg.solve(matrix, vector)
    except np.linalg.LinAlgError:
        raise ValueError("the conditions do not determine the coefficients") from None
    if not np.all(np.isfinite(coefficients)):
        raise ValueError("the conditions do not determine the coefficients")
    return coefficients.astype(float)

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
     'FLAT = np.array([0.3189343369335683, 0.0393370733544467])\n'
     'FLATV = np.array([0.45, 0.80])\n'
     'WIDE = np.array([-0.3687232531791833 - 0.3027516319622959j,\n'
     '                 -0.2948067142222096 - 0.5202762302609606j,\n'
     '                 -0.2619056718872012 - 0.7058136487190447j])\n'
     'WIDEV = np.array([1.70 + 0.55j, 2.30 + 1.30j, 2.90 + 3.60j])\n'
     'def _factor(points, poles):\n'
     '    return evaluate_pole_factor(points, poles)\n'
     'def _toy_factor(points, poles):\n'
     '    arr = np.asarray(points, dtype=complex)\n'
     '    return 1.0 / (2.0 + np.sum(np.abs(np.asarray(poles))) + arr ** 2)\n'
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
     'FLAT = np.array([0.3189343369335683, 0.0393370733544467])\n'
     'FLATV = np.array([0.45, 0.80])\n'
     'WIDE = np.array([-0.3687232531791833 - 0.3027516319622959j,\n'
     '                 -0.2948067142222096 - 0.5202762302609606j,\n'
     '                 -0.2619056718872012 - 0.7058136487190447j])\n'
     'WIDEV = np.array([1.70 + 0.55j, 2.30 + 1.30j, 2.90 + 3.60j])\n'
     'def _factor(points, poles):\n'
     '    return evaluate_pole_factor(points, poles)\n'
     'def _toy_factor(points, poles):\n'
     '    arr = np.asarray(points, dtype=complex)\n'
     '    return 1.0 / (2.0 + np.sum(np.abs(np.asarray(poles))) + arr ** 2)\n'
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
            'call': '_independent_inputs(solve_series_coefficients, FLAT, FLATV, WIDE, WIDEV, POLES, 2, _factor)',
            'gold_call': '_independent_inputs(_oracle_solve_series_coefficients, FLAT, FLATV, WIDE, WIDEV, POLES, 2, _reference_factor)',
        },
        {
            'setup': setups[0],
            'call': '_independent_inputs(solve_series_coefficients, FLAT, FLATV, WIDE, WIDEV, POLES, 1, _factor)',
            'gold_call': '_independent_inputs(_oracle_solve_series_coefficients, FLAT, FLATV, WIDE, WIDEV, POLES, 1, _reference_factor)',
        },
        {
            'setup': setups[0],
            'call': '_independent_inputs(solve_series_coefficients, FLAT, FLATV, WIDE, WIDEV, POLES, 4, _factor)',
            'gold_call': '_independent_inputs(_oracle_solve_series_coefficients, FLAT, FLATV, WIDE, WIDEV, POLES, 4, _reference_factor)',
        },
        {
            'setup': setups[0],
            'call': '_independent_inputs(solve_series_coefficients, np.array([]), np.array([]), WIDE, WIDEV, POLES, 2, _factor)',
            'gold_call': '_independent_inputs(_oracle_solve_series_coefficients, np.array([]), np.array([]), WIDE, WIDEV, POLES, 2, _reference_factor)',
        },
        {
            'setup': setups[0],
            'call': '_independent_inputs(solve_series_coefficients, FLAT, FLATV, np.array([]), np.array([]), POLES, 3, _factor)',
            'gold_call': '_independent_inputs(_oracle_solve_series_coefficients, FLAT, FLATV, np.array([]), np.array([]), POLES, 3, _reference_factor)',
        },
        {
            'setup': setups[0],
            'call': '_independent_inputs(solve_series_coefficients, FLAT, FLATV, WIDE, WIDEV, POLES[:2], 2, _toy_factor)',
            'gold_call': '_independent_inputs(_oracle_solve_series_coefficients, FLAT, FLATV, WIDE, WIDEV, POLES[:2], 2, _toy_factor)',
        },
        {
            'setup': setups[0],
            'call': '_independent_inputs(solve_series_coefficients, np.array([0.1]), np.array([1.05]), WIDE[:1], WIDEV[:1], POLES, 1, _factor)',
            'gold_call': '_independent_inputs(_oracle_solve_series_coefficients, np.array([0.1]), np.array([1.05]), WIDE[:1], WIDEV[:1], POLES, 1, _reference_factor)',
        },
        {
            'setup': setups[1],
            'call': '_status(lambda: _independent_inputs(solve_series_coefficients, np.array([0.2, 0.2]), np.array([1.0, 2.0]), WIDE, WIDEV, POLES, 2, _factor))',
            'gold_call': '_status(lambda: _independent_inputs(_oracle_solve_series_coefficients, np.array([0.2, 0.2]), np.array([1.0, 2.0]), WIDE, WIDEV, POLES, 2, _reference_factor))',
        },
        {
            'setup': setups[1],
            'call': '_status(lambda: _independent_inputs(solve_series_coefficients, np.array([0.2 + 0.1j]), np.array([1.0]), WIDE, WIDEV, POLES, 2, _factor))',
            'gold_call': '_status(lambda: _independent_inputs(_oracle_solve_series_coefficients, np.array([0.2 + 0.1j]), np.array([1.0]), WIDE, WIDEV, POLES, 2, _reference_factor))',
        },
        {
            'setup': setups[1],
            'call': '_status(lambda: _independent_inputs(solve_series_coefficients, FLAT, FLATV, WIDE, WIDEV, POLES, 2, _short_factor))',
            'gold_call': '_status(lambda: _independent_inputs(_oracle_solve_series_coefficients, FLAT, FLATV, WIDE, WIDEV, POLES, 2, _short_factor))',
        },
        {
            'setup': setups[1],
            'call': '_status(lambda: _independent_inputs(solve_series_coefficients, FLAT, FLATV, WIDE, WIDEV, POLES, 0, _factor))',
            'gold_call': '_status(lambda: _independent_inputs(_oracle_solve_series_coefficients, FLAT, FLATV, WIDE, WIDEV, POLES, 0, _reference_factor))',
        },
    ]
