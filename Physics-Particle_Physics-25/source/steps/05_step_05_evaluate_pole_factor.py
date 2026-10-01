"""
Evaluate the product of pole factors that carries the resonances of the parametrisation in the expansion variable.

Once the resonance sheets lie inside the unit disc, each resonance can be written as an explicit conjugate pair of poles in the expansion variable instead of being left to the series.

Returns
-------
np.ndarray: complex pole-product values, one per requested point.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def evaluate_pole_factor(
    points: "np.ndarray",
    pole_points: "np.ndarray",
) -> "np.ndarray":
    """Return the resonance pole product at each requested point.

    With ``w_r`` the images of the resonance poles in the expansion
    variable, the product is

        ``P(w) = prod_r 1 / ((w - w_r) * (w - conj(w_r)))``

    so that every resonance contributes its pole and the mirror pole
    required for the parametrisation to be real where the variable is real.

    Parameters
    ----------
    points : np.ndarray
        One-dimensional array of finite complex points, at least one.
    pole_points : np.ndarray
        One-dimensional array of finite complex pole images, at least one,
        none of them real.

    Returns
    -------
    np.ndarray
        Complex array with the same length as ``points``.

    Raises
    ------
    ValueError
        If either argument is not a one-dimensional non-empty array of
        finite numbers, if any pole image is real, or if any point
        coincides with a pole or its mirror.
    """
    return product

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
def _oracle_evaluate_pole_factor(
    points: "np.ndarray",
    pole_points: "np.ndarray",
) -> "np.ndarray":
    """Reference implementation of the conjugate-pair pole product."""
    import numpy as np

    def _as_line(value, label):
        try:
            arr = np.asarray(value, dtype=complex)
        except (TypeError, ValueError):
            raise ValueError(f"{label} must be an array of numbers") from None
        arr = np.atleast_1d(arr)
        if arr.ndim != 1 or arr.size < 1:
            raise ValueError(f"{label} must be a non-empty one-dimensional array")
        if not np.all(np.isfinite(arr.real) & np.isfinite(arr.imag)):
            raise ValueError(f"{label} must contain only finite numbers")
        return arr

    grid = _as_line(points, "points")
    poles = _as_line(pole_points, "pole_points")
    if np.any(np.abs(poles.imag) <= 0.0):
        raise ValueError("pole images must lie off the real axis")

    total = np.ones(grid.size, dtype=complex)
    for pole in poles:
        gap = grid - pole
        mirror = grid - np.conj(pole)
        if np.any(gap == 0.0) or np.any(mirror == 0.0):
            raise ValueError("a requested point coincides with a pole")
        total = total / (gap * mirror)
    if not np.all(np.isfinite(total.real) & np.isfinite(total.imag)):
        raise ValueError("the pole product overflowed at a requested point")
    return total

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
     'GRID = np.array([0.3189343369335683 + 0.0j,\n'
     '                 0.0393370733544467 + 0.0j,\n'
     '                 -0.3687232531791833 - 0.3027516319622959j,\n'
     '                 -0.2948067142222096 - 0.5202762302609606j,\n'
     '                 -0.2619056718872012 - 0.7058136487190447j])\n'
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
     'GRID = np.array([0.3189343369335683 + 0.0j,\n'
     '                 0.0393370733544467 + 0.0j,\n'
     '                 -0.3687232531791833 - 0.3027516319622959j,\n'
     '                 -0.2948067142222096 - 0.5202762302609606j,\n'
     '                 -0.2619056718872012 - 0.7058136487190447j])\n'
     'def _status(fn):\n'
     '    try:\n'
     '        fn()\n'
     '        return 0\n'
     '    except ValueError:\n'
     '        return 1\n'
     '    except Exception:\n'
     '        return 2\n'
     '\n'
     'import copy\n'
     'def _independent_inputs(fn, *args, **kwargs):\n'
     '    return fn(*copy.deepcopy(args), **copy.deepcopy(kwargs))\n')
    return [
        {
            'setup': setups[0],
            'call': '_independent_inputs(evaluate_pole_factor, GRID, POLES)',
            'gold_call': '_independent_inputs(_oracle_evaluate_pole_factor, GRID, POLES)',
        },
        {
            'setup': setups[0],
            'call': '_independent_inputs(evaluate_pole_factor, np.array([1.0 + 0j]), POLES)',
            'gold_call': '_independent_inputs(_oracle_evaluate_pole_factor, np.array([1.0 + 0j]), POLES)',
        },
        {
            'setup': setups[0],
            'call': '_independent_inputs(evaluate_pole_factor, GRID, POLES[:1])',
            'gold_call': '_independent_inputs(_oracle_evaluate_pole_factor, GRID, POLES[:1])',
        },
        {
            'setup': setups[0],
            'call': '_independent_inputs(evaluate_pole_factor, np.array([1.6 - 0.9j, -2.3 + 0.4j]), POLES)',
            'gold_call': '_independent_inputs(_oracle_evaluate_pole_factor, np.array([1.6 - 0.9j, -2.3 + 0.4j]), POLES)',
        },
        {
            'setup': setups[0],
            'call': '_independent_inputs(evaluate_pole_factor, np.array([POLES[0] + 0.001]), POLES)',
            'gold_call': '_independent_inputs(_oracle_evaluate_pole_factor, np.array([POLES[0] + 0.001]), POLES)',
        },
        {
            'setup': setups[0],
            'call': '_independent_inputs(evaluate_pole_factor, np.conj(GRID), POLES)',
            'gold_call': '_independent_inputs(_oracle_evaluate_pole_factor, np.conj(GRID), POLES)',
        },
        {
            'setup': setups[1],
            'call': '_status(lambda: _independent_inputs(evaluate_pole_factor, np.array([np.conj(POLES[1])]), POLES))',
            'gold_call': '_status(lambda: _independent_inputs(_oracle_evaluate_pole_factor, np.array([np.conj(POLES[1])]), POLES))',
        },
        {
            'setup': setups[1],
            'call': '_status(lambda: _independent_inputs(evaluate_pole_factor, GRID, np.array([0.4 + 0j])))',
            'gold_call': '_status(lambda: _independent_inputs(_oracle_evaluate_pole_factor, GRID, np.array([0.4 + 0j])))',
        },
        {
            'setup': setups[1],
            'call': '_status(lambda: _independent_inputs(evaluate_pole_factor, GRID, np.array([], dtype=complex)))',
            'gold_call': '_status(lambda: _independent_inputs(_oracle_evaluate_pole_factor, GRID, np.array([], dtype=complex)))',
        },
    ]
