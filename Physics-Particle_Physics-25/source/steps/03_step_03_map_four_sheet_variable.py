"""
Compose the two-threshold variable with the slit-disc map to obtain the expansion variable of the four-sheet parametrisation.

Composing the two maps leaves the physical sheet and its elastic partner inside the unit disc with no interior cut, so a power series in the composed variable can carry resonance poles as explicit factors.

Returns
-------
complex: the four-sheet expansion variable on the requested sheet.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def map_four_sheet_variable(
    s: complex,
    s_plus: float,
    s_inelastic: float,
    s_lhc: float,
    s_origin: float,
    sheet: str,
    threshold_fn: "Callable[..., complex]",
    lhc_fn: "Callable[..., complex]",
) -> complex:
    """Return the four-sheet expansion variable at ``s`` on one Riemann sheet.

    ``threshold_fn(s, s_plus, s_inelastic, sheet)`` follows the contract of
    ``map_two_threshold_variable`` and ``lhc_fn(point, cut_branch_point,
    origin_point)`` that of ``map_left_hand_cut``.

    The slit is fixed by the two reference points
    ``threshold_fn(s_lhc, s_plus, s_inelastic, "21")``, the image of the
    left-hand branch point on the sheet reached across the elastic cut, and
    ``threshold_fn(s_origin, s_plus, s_inelastic, "11")``, the image of the
    expansion point on the physical sheet; both are real, and their real
    parts are the arguments handed to ``lhc_fn``.

    ``lhc_fn`` accepts only arguments of modulus at most one. Where the
    two-threshold variable leaves the closed unit disc, return instead the
    value of the analytic continuation of the composed map through the unit
    circle, which maps the exterior of the disc onto itself.

    Parameters
    ----------
    s : complex
        Invariant mass squared; real values are accepted.
    s_plus : float
        Elastic threshold, finite and positive.
    s_inelastic : float
        Inelastic threshold, finite and larger than ``s_plus``.
    s_lhc : float
        Finite real point where the left-hand cut opens.
    s_origin : float
        Finite real expansion point, mapped to the origin on the physical
        sheet.
    sheet : str
        One of ``"11"``, ``"21"``, ``"22"`` and ``"12"``.
    threshold_fn : callable
        Two-threshold uniformising map.
    lhc_fn : callable
        Slit-disc conformal map.

    Returns
    -------
    complex
        The expansion variable on the requested sheet.

    Raises
    ------
    ValueError
        If ``threshold_fn`` or ``lhc_fn`` is not callable, if either
        reference point is not a finite real number, or if any stage of the
        composition rejects its argument or fails to return a finite number.
    """
    return 0j

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
def _oracle_map_four_sheet_variable(
    s: complex,
    s_plus: float,
    s_inelastic: float,
    s_lhc: float,
    s_origin: float,
    sheet: str,
    threshold_fn: "Callable[..., complex]",
    lhc_fn: "Callable[..., complex]",
) -> complex:
    """Reference implementation (composition with reflection outside the disc)."""
    import numpy as np

    def _finite(value):
        value = complex(value)
        return bool(np.isfinite(value.real) and np.isfinite(value.imag))

    def _is_function(value):
        return callable(value)

    if not _is_function(threshold_fn):
        raise ValueError("threshold_fn must be callable")
    if not _is_function(lhc_fn):
        raise ValueError("lhc_fn must be callable")

    edge = complex(threshold_fn(s_lhc, s_plus, s_inelastic, "21"))
    centre = complex(threshold_fn(s_origin, s_plus, s_inelastic, "11"))
    if not (_finite(edge) and _finite(centre)):
        raise ValueError("the reference points must be finite")
    if abs(edge.imag) > 1.0e-12 or abs(centre.imag) > 1.0e-12:
        raise ValueError("the reference points must be real")

    variable = complex(threshold_fn(s, s_plus, s_inelastic, sheet))
    if not _finite(variable):
        raise ValueError("the two-threshold variable must be finite")

    if abs(variable) <= 1.0:
        result = complex(lhc_fn(variable, edge.real, centre.real))
    else:
        # Reflection in the unit circle is the unique continuation that keeps
        # the boundary fixed, so the exterior value is the reciprocal of the
        # image of the reciprocal point.
        inner = complex(lhc_fn(1.0 / variable, edge.real, centre.real))
        if inner == 0.0:
            raise ValueError("the continued map is not finite at this s")
        result = 1.0 / inner
    if not _finite(result):
        raise ValueError("the composed map must return a finite value")
    return complex(result)

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
     'def _threshold(s, sp, sin, sheet):\n'
     '    return map_two_threshold_variable(s, sp, sin, sheet)\n'
     'def _lhc(point, edge, centre):\n'
     '    return map_left_hand_cut(point, edge, centre)\n'
     'def _toy_threshold(s, sp, sin, sheet):\n'
     '    base = (complex(s) - sp) / (sin + 3.0 * abs(s) + 1.0)\n'
     "    scale = {'11': 1.0, '21': -1.0, '22': 4.0, '12': -4.0}[sheet]\n"
     '    return scale * (0.3 + base)\n'
     'def _toy_lhc(point, edge, centre):\n'
     '    return (complex(point) - centre) / (1.0 - edge * complex(point))\n'
     '\n'
     'def _reference_threshold(s, sp, sin, sheet):\n'
     '    return _oracle_map_two_threshold_variable(s, sp, sin, sheet)\n'
     '\n'
     'def _reference_lhc(point, edge, centre):\n'
     '    return _oracle_map_left_hand_cut(point, edge, centre)\n',
     'import numpy as np\n'
     'SP = 4.0 * 0.13957 ** 2\n'
     'SIN = 4.0 * 0.493677 ** 2\n'
     'def _scal(value):\n'
     '    arr = np.atleast_1d(np.asarray(value, dtype=complex)).ravel()\n'
     '    idx = np.arange(1.0, arr.size + 1.0)\n'
     '    return float(np.sum(np.abs(arr))\n'
     '                 + np.sum(arr.real * np.cos(idx))\n'
     '                 + np.sum(arr.imag * np.sin(idx)))\n'
     'def _threshold(s, sp, sin, sheet):\n'
     '    return map_two_threshold_variable(s, sp, sin, sheet)\n'
     'def _lhc(point, edge, centre):\n'
     '    return map_left_hand_cut(point, edge, centre)\n'
     'def _toy_threshold(s, sp, sin, sheet):\n'
     '    base = (complex(s) - sp) / (sin + 3.0 * abs(s) + 1.0)\n'
     "    scale = {'11': 1.0, '21': -1.0, '22': 4.0, '12': -4.0}[sheet]\n"
     '    return scale * (0.3 + base)\n'
     'def _toy_lhc(point, edge, centre):\n'
     '    return (complex(point) - centre) / (1.0 - edge * complex(point))\n'
     'def _status(fn):\n'
     '    try:\n'
     '        fn()\n'
     '        return 0\n'
     '    except ValueError:\n'
     '        return 1\n'
     '    except Exception:\n'
     '        return 2\n'
     'def _bad_threshold(s, sp, sin, sheet):\n'
     "    return 0.4j if sheet == '21' else 0.2\n"
     '\n'
     'def _reference_threshold(s, sp, sin, sheet):\n'
     '    return _oracle_map_two_threshold_variable(s, sp, sin, sheet)\n'
     '\n'
     'def _reference_lhc(point, edge, centre):\n'
     '    return _oracle_map_left_hand_cut(point, edge, centre)\n')
    return [
        {
            'setup': setups[0],
            'call': "_scal(map_four_sheet_variable(-2.0, SP, SIN, 0.0, -0.30, '11', _threshold, _lhc))",
            'gold_call': "_scal(_oracle_map_four_sheet_variable(-2.0, SP, SIN, 0.0, -0.30, '11', _reference_threshold, _reference_lhc))",
        },
        {
            'setup': setups[0],
            'call': "_scal(map_four_sheet_variable(0.8, SP, SIN, 0.0, -0.30, '11', _threshold, _lhc))",
            'gold_call': "_scal(_oracle_map_four_sheet_variable(0.8, SP, SIN, 0.0, -0.30, '11', _reference_threshold, _reference_lhc))",
        },
        {
            'setup': setups[0],
            'call': "_scal(map_four_sheet_variable(0.1400030 - 0.2504040j, SP, SIN, 0.0, -0.30, '21', _threshold, _lhc))",
            'gold_call': "_scal(_oracle_map_four_sheet_variable(0.1400030 - 0.2504040j, SP, SIN, 0.0, -0.30, '21', _reference_threshold, _reference_lhc))",
        },
        {
            'setup': setups[0],
            'call': "_scal(map_four_sheet_variable(0.3675 - 0.49j, SP, SIN, 0.0, -0.30, '22', _threshold, _lhc))",
            'gold_call': "_scal(_oracle_map_four_sheet_variable(0.3675 - 0.49j, SP, SIN, 0.0, -0.30, '22', _reference_threshold, _reference_lhc))",
        },
        {
            'setup': setups[0],
            'call': "_scal(map_four_sheet_variable(0.9375800 - 0.1271355j, SP, SIN, 0.0, -0.30, '12', _threshold, _lhc))",
            'gold_call': "_scal(_oracle_map_four_sheet_variable(0.9375800 - 0.1271355j, SP, SIN, 0.0, -0.30, '12', _reference_threshold, _reference_lhc))",
        },
        {
            'setup': setups[0],
            'call': "_scal(map_four_sheet_variable(-0.30, SP, SIN, 0.0, -0.30, '11', _threshold, _lhc) + 0.7)",
            'gold_call': "_scal(_oracle_map_four_sheet_variable(-0.30, SP, SIN, 0.0, -0.30, '11', _reference_threshold, _reference_lhc) + 0.7)",
        },
        {
            'setup': setups[0],
            'call': "_scal(map_four_sheet_variable(0.45, SP, SIN, -0.05, 0.0, '11', _threshold, _lhc))",
            'gold_call': "_scal(_oracle_map_four_sheet_variable(0.45, SP, SIN, -0.05, 0.0, '11', _reference_threshold, _reference_lhc))",
        },
        {
            'setup': setups[0],
            'call': "_scal(map_four_sheet_variable(0.62, 0.11, 1.4, -0.2, 0.05, '22', _toy_threshold, _toy_lhc))",
            'gold_call': "_scal(_oracle_map_four_sheet_variable(0.62, 0.11, 1.4, -0.2, 0.05, '22', _toy_threshold, _toy_lhc))",
        },
        {
            'setup': setups[0],
            'call': "_scal(map_four_sheet_variable(-1.3, 0.09, 2.0, 0.4, -0.7, '11', _toy_threshold, _toy_lhc))",
            'gold_call': "_scal(_oracle_map_four_sheet_variable(-1.3, 0.09, 2.0, 0.4, -0.7, '11', _toy_threshold, _toy_lhc))",
        },
        {
            'setup': setups[1],
            'call': "_status(lambda: map_four_sheet_variable(0.5, SP, SIN, 0.0, -0.30, '11', 3.0, _lhc))",
            'gold_call': "_status(lambda: _oracle_map_four_sheet_variable(0.5, SP, SIN, 0.0, -0.30, '11', 3.0, _reference_lhc))",
        },
        {
            'setup': setups[1],
            'call': "_status(lambda: map_four_sheet_variable(0.5, SP, SIN, 0.0, -0.30, '11', _bad_threshold, _lhc))",
            'gold_call': "_status(lambda: _oracle_map_four_sheet_variable(0.5, SP, SIN, 0.0, -0.30, '11', _bad_threshold, _reference_lhc))",
        },
    ]
