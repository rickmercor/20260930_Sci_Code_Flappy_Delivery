"""
Measure how fast the expansion variable approaches its value at infinite invariant mass, and with what coefficient.

The rate at which a conformal variable reaches the image of infinity controls how a prescribed high-energy fall-off of the amplitude translates into linear conditions on the expansion coefficients

Returns
-------
np.ndarray: two floats, the exponent p and the coefficient c of (1 - psi)**p -> c / abs(s).
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def solve_asymptotic_behaviour(
    variable_fn: "Callable[[float], complex]",
    probe: float = -1.0e10,
    ratio: float = 1.0e4,
) -> "np.ndarray":
    """Return the exponent and coefficient of the variable's approach to unity.

    ``variable_fn(s)`` returns the physical-sheet expansion variable at a
    real spacelike ``s``, where it is real and below one, and tends to ``1``
    as ``s`` tends to ``-inf``. There is a positive exponent ``p`` and a
    finite positive ``c`` with

        ``(1 - variable_fn(s)) ** p -> c / abs(s)``   as ``s -> -inf``.

    Estimate ``p`` from the two probes ``s = probe`` and
    ``s = probe * ratio``, and evaluate ``c`` at the farther of the two.

    Parameters
    ----------
    variable_fn : callable
        Physical-sheet expansion variable as a function of real ``s``.
    probe : float
        Finite negative probe point.
    ratio : float
        Finite factor larger than one giving the second, farther probe.

    Returns
    -------
    np.ndarray
        Array ``[p, c]`` of two floats.

    Raises
    ------
    ValueError
        If ``variable_fn`` is not callable, if ``probe`` is not a finite
        negative number, if ``ratio`` is not a finite number greater than
        one, if either probe value is not a finite real number below one, or
        if the estimated exponent is not finite and positive.
    """
    return behaviour

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
def _oracle_solve_asymptotic_behaviour(
    variable_fn: "Callable[[float], complex]",
    probe: float = -1.0e10,
    ratio: float = 1.0e4,
) -> "np.ndarray":
    """Reference implementation (two-point log-log slope)."""
    import numpy as np

    def _is_real(value):
        return (not isinstance(value, bool)
                and isinstance(value, (int, float, np.integer, np.floating))
                and bool(np.isfinite(value)))

    def _is_function(value):
        return callable(value)

    if not _is_function(variable_fn):
        raise ValueError("variable_fn must be callable")
    if not (_is_real(probe) and probe < 0.0):
        raise ValueError("probe must be a finite negative number")
    if not (_is_real(ratio) and ratio > 1.0):
        raise ValueError("ratio must be a finite number greater than one")

    near = float(probe)
    far = float(probe) * float(ratio)

    def _offset(point):
        value = complex(variable_fn(point))
        if not (np.isfinite(value.real) and np.isfinite(value.imag)):
            raise ValueError("the variable must be finite at the probe points")
        if abs(value.imag) > 1.0e-12:
            raise ValueError("the variable must be real at a spacelike probe")
        gap = value.real - 1.0
        if not (gap < 0.0):
            raise ValueError("the variable must stay below one at the probes")
        return -gap

    near_gap = _offset(near)
    far_gap = _offset(far)
    slope = np.log(near_gap / far_gap)
    if not (np.isfinite(slope) and slope > 0.0):
        raise ValueError("the probes do not resolve a decaying approach to unity")
    exponent = float(np.log(abs(far / near)) / slope)
    if not (np.isfinite(exponent) and exponent > 0.0):
        raise ValueError("the estimated exponent must be finite and positive")

    # (1 - psi)**p -> c / abs(s), read off at the farther probe.
    coefficient = float(abs(far) * far_gap ** exponent)
    if not np.isfinite(coefficient):
        raise ValueError("the estimated coefficient must be finite")
    return np.array([exponent, coefficient], dtype=float)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases() -> list[dict]:
    """Return the submitted cases with independent dependency chains and ordered outputs."""
    setups = ('import numpy as np\n'
     'SP = 4.0 * 0.13957 ** 2\n'
     'SIN = 4.0 * 0.493677 ** 2\n'
     'def _scal(value):\n'
     '    arr = np.atleast_1d(np.asarray(value, dtype=complex)).ravel()\n'
     '    idx = np.arange(1.0, arr.size + 1.0)\n'
     '    return float(np.sum(np.abs(arr))\n'
     '                 + np.sum(arr.real * np.cos(idx))\n'
     '                 + np.sum(arr.imag * np.sin(idx)))\n'
     'def _variable(s, lhc=0.0, origin=-0.30, sin=SIN):\n'
     '    return map_four_sheet_variable(\n'
     "        s, SP, sin, lhc, origin, '11',\n"
     '        map_two_threshold_variable, map_left_hand_cut)\n'
     'def _shifted(s):\n'
     '    return _variable(s, lhc=-0.05, origin=0.0)\n'
     'def _heavier(s):\n'
     '    return _variable(s, sin=4.0 * 0.5479 ** 2)\n'
     'def _toy(power, scale):\n'
     '    def fn(s):\n'
     '        return 1.0 - (scale / abs(s)) ** (1.0 / power)\n'
     '    return fn\n'
,
     'import numpy as np\n'
     'SP = 4.0 * 0.13957 ** 2\n'
     'SIN = 4.0 * 0.493677 ** 2\n'
     'def _scal(value):\n'
     '    arr = np.atleast_1d(np.asarray(value, dtype=complex)).ravel()\n'
     '    idx = np.arange(1.0, arr.size + 1.0)\n'
     '    return float(np.sum(np.abs(arr))\n'
     '                 + np.sum(arr.real * np.cos(idx))\n'
     '                 + np.sum(arr.imag * np.sin(idx)))\n'
     'def _variable(s, lhc=0.0, origin=-0.30, sin=SIN):\n'
     '    return map_four_sheet_variable(\n'
     "        s, SP, sin, lhc, origin, '11',\n"
     '        map_two_threshold_variable, map_left_hand_cut)\n'
     'def _shifted(s):\n'
     '    return _variable(s, lhc=-0.05, origin=0.0)\n'
     'def _heavier(s):\n'
     '    return _variable(s, sin=4.0 * 0.5479 ** 2)\n'
     'def _toy(power, scale):\n'
     '    def fn(s):\n'
     '        return 1.0 - (scale / abs(s)) ** (1.0 / power)\n'
     '    return fn\n'
     'def _status(fn):\n'
     '    try:\n'
     '        fn()\n'
     '        return 0\n'
     '    except ValueError:\n'
     '        return 1\n'
     '    except Exception:\n'
     '        return 2\n'
     'def _rising(s):\n'
     '    return 1.0 + 1.0 / abs(s)\n'
)
    return [
        {
            'setup': setups[0],
            'call': 'solve_asymptotic_behaviour(_variable)',
            'gold_call': '_oracle_solve_asymptotic_behaviour(lambda s: _oracle_map_four_sheet_variable(s, SP, SIN, 0.0, -0.30, \'11\', _oracle_map_two_threshold_variable, _oracle_map_left_hand_cut))',
            'tol': 1.0e-6,
        },
        {
            'setup': setups[0],
            'call': 'solve_asymptotic_behaviour(_shifted, -1000000000.0, 100000.0)',
            'gold_call': '_oracle_solve_asymptotic_behaviour(lambda s: _oracle_map_four_sheet_variable(s, SP, SIN, -0.05, 0.0, \'11\', _oracle_map_two_threshold_variable, _oracle_map_left_hand_cut), -1000000000.0, 100000.0)',
            'tol': 1.0e-6,
        },
        {
            'setup': setups[0],
            'call': 'solve_asymptotic_behaviour(_heavier)',
            'gold_call': '_oracle_solve_asymptotic_behaviour(lambda s: _oracle_map_four_sheet_variable(s, SP, 4.0 * 0.5479 ** 2, 0.0, -0.30, \'11\', _oracle_map_two_threshold_variable, _oracle_map_left_hand_cut))',
            'tol': 1.0e-6,
        },
        {
            'setup': setups[0],
            'call': 'solve_asymptotic_behaviour(_toy(3.0, 7.5), -100000000.0, 1000.0)',
            'gold_call': '_oracle_solve_asymptotic_behaviour(_toy(3.0, 7.5), -100000000.0, 1000.0)',
        },
        {
            'setup': setups[0],
            'call': 'solve_asymptotic_behaviour(_toy(1.5, 0.4), -10000000.0, 1000000.0)',
            'gold_call': '_oracle_solve_asymptotic_behaviour(_toy(1.5, 0.4), -10000000.0, 1000000.0)',
        },
        {
            'setup': setups[0],
            'call': 'solve_asymptotic_behaviour(_toy(2.0, 2.5), -1000000000.0, 100.0)',
            'gold_call': '_oracle_solve_asymptotic_behaviour(_toy(2.0, 2.5), -1000000000.0, 100.0)',
        },
        {
            'setup': setups[1],
            'call': '_status(lambda: solve_asymptotic_behaviour(_variable, 1.0e10, 1.0e4))',
            'gold_call': '_status(lambda: _oracle_solve_asymptotic_behaviour(lambda s: _oracle_map_four_sheet_variable(s, SP, SIN, 0.0, -0.30, \'11\', _oracle_map_two_threshold_variable, _oracle_map_left_hand_cut), 1.0e10, 1.0e4))',
        },
        {
            'setup': setups[1],
            'call': '_status(lambda: solve_asymptotic_behaviour(_variable, -1.0e10, 0.5))',
            'gold_call': '_status(lambda: _oracle_solve_asymptotic_behaviour(lambda s: _oracle_map_four_sheet_variable(s, SP, SIN, 0.0, -0.30, \'11\', _oracle_map_two_threshold_variable, _oracle_map_left_hand_cut), -1.0e10, 0.5))',
        },
        {
            'setup': setups[1],
            'call': '_status(lambda: solve_asymptotic_behaviour(_rising, -1.0e10, 1.0e4))',
            'gold_call': '_status(lambda: _oracle_solve_asymptotic_behaviour(_rising, -1.0e10, 1.0e4))',
        },
    ]
