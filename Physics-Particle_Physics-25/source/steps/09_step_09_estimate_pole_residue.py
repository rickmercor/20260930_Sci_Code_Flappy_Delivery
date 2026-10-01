"""
Compose every earlier step to obtain the modulus of the residue of the parametrised amplitude at a chosen resonance pole.

A parametrisation whose expansion variable reaches the resonance sheets lets the residue at a pole be read off directly from a fit performed on the physical axis.

Returns
-------
float: the modulus of the residue of the parametrised amplitude at the selected pole.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def estimate_pole_residue(
    s_plus: float = 4.0 * 0.13957 ** 2,
    s_inelastic: float = 4.0 * 0.493677 ** 2,
    s_lhc: float = 0.0,
    s_origin: float = -0.30,
    pole_roots: tuple = (0.462 - 0.271j, 0.9935 - 0.0285j, 0.9705 - 0.0655j, 0.700 - 0.350j),
    pole_sheets: tuple = ("21", "21", "22", "22"),
    real_nodes: tuple = (-2.00, -0.40),
    real_values: tuple = (0.45, 0.80),
    timelike_nodes: tuple = (0.20, 0.50, 0.80),
    timelike_values: tuple = (1.70 + 0.55j, 2.30 + 1.30j, 2.90 + 3.60j),
    target: int = 0,
    radius: float = 0.01,
    nodes: int = 128,
) -> float:
    """Return the modulus of the residue of the parametrised amplitude at one pole.

    The four-sheet expansion variable is built from the thresholds
    ``s_plus`` and ``s_inelastic``, from the left-hand branch point
    ``s_lhc`` on the sheet reached across the elastic cut, and from the
    expansion point ``s_origin``, which is sent to the origin on the
    physical sheet. Each entry of ``pole_roots`` is the square root of a
    pole position, carried by the sheet named in the matching entry of
    ``pole_sheets``, and these poles are implemented as explicit conjugate
    pairs of factors multiplying a truncated real power series.

    The coefficients are fixed by three kinds of condition: the values
    ``real_values`` taken at the real points ``real_nodes`` below the
    elastic threshold, the values ``timelike_values`` taken at the physical
    boundary values above it at ``timelike_nodes``, and the requirement
    that the parametrised amplitude fall off as one over the invariant mass
    squared at large spacelike argument. The series is truncated at the
    degree for which those conditions determine the coefficients uniquely.

    Return the modulus of the residue, in the units of ``s``, of the
    resulting amplitude continued to the sheet of the pole selected by
    ``target``, sampled on a circle of radius ``radius`` with ``nodes``
    points. The defaults reproduce the problem statement.

    Parameters
    ----------
    s_plus : float
        Elastic threshold, finite and positive.
    s_inelastic : float
        Inelastic threshold, finite and larger than ``s_plus``.
    s_lhc : float
        Finite real point where the left-hand cut opens.
    s_origin : float
        Finite real expansion point below ``s_plus``.
    pole_roots : tuple
        Non-empty tuple of finite complex square roots of pole positions.
    pole_sheets : tuple
        Sheet labels, same length as ``pole_roots``.
    real_nodes : tuple
        Real points below ``s_plus`` carrying real measurements.
    real_values : tuple
        Real measured values, same length as ``real_nodes``.
    timelike_nodes : tuple
        Real points above ``s_plus`` carrying complex measurements.
    timelike_values : tuple
        Measured values, same length as ``timelike_nodes``.
    target : int
        Index into ``pole_roots`` of the pole whose residue is returned.
    radius : float
        Finite positive radius of the sampling circle.
    nodes : int
        Number of sample points, at least two.

    Returns
    -------
    float
        The modulus of the residue at the selected pole.

    Raises
    ------
    ValueError
        If any argument is outside the stated domain, if the tuple lengths
        do not match, if ``target`` is not a valid index, or if any stage
        rejects its input.
    """
    return 0.0

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
def _oracle_estimate_pole_residue(
    s_plus: float = 4.0 * 0.13957 ** 2,
    s_inelastic: float = 4.0 * 0.493677 ** 2,
    s_lhc: float = 0.0,
    s_origin: float = -0.30,
    pole_roots: tuple = (0.462 - 0.271j, 0.9935 - 0.0285j, 0.9705 - 0.0655j, 0.700 - 0.350j),
    pole_sheets: tuple = ("21", "21", "22", "22"),
    real_nodes: tuple = (-2.00, -0.40),
    real_values: tuple = (0.45, 0.80),
    timelike_nodes: tuple = (0.20, 0.50, 0.80),
    timelike_values: tuple = (1.70 + 0.55j, 2.30 + 1.30j, 2.90 + 3.60j),
    target: int = 0,
    radius: float = 0.01,
    nodes: int = 128,
) -> float:
    """Reference orchestrator using only the reference function chain."""
    import numpy as np

    def _line(value, label):
        try:
            arr = np.atleast_1d(np.asarray(tuple(value), dtype=complex))
        except (TypeError, ValueError):
            raise ValueError(f"{label} must be a sequence of numbers") from None
        if arr.ndim != 1:
            raise ValueError(f"{label} must be one-dimensional")
        return arr

    roots = _line(pole_roots, "pole_roots")
    if roots.size < 1:
        raise ValueError("pole_roots must not be empty")
    sheets = tuple(pole_sheets)
    if len(sheets) != roots.size:
        raise ValueError("pole_sheets must match pole_roots in length")
    flat_nodes = _line(real_nodes, "real_nodes")
    flat_values = _line(real_values, "real_values")
    wide_nodes = _line(timelike_nodes, "timelike_nodes")
    wide_values = _line(timelike_values, "timelike_values")
    def _is_integer(value):
        return (not isinstance(value, bool)
                and isinstance(value, (int, np.integer)))

    if not _is_integer(target):
        raise ValueError("target must be an integer")
    if not 0 <= int(target) < roots.size:
        raise ValueError("target must index pole_roots")

    def _threshold(point, low, high, sheet):
        return _oracle_map_two_threshold_variable(point, low, high, sheet)

    def _slit(point, edge, centre):
        return _oracle_map_left_hand_cut(point, edge, centre)

    def _variable(point, sheet):
        return _oracle_map_four_sheet_variable(
            point, s_plus, s_inelastic, s_lhc, s_origin, sheet, _threshold, _slit
        )

    def _factor(points, poles):
        return _oracle_evaluate_pole_factor(points, poles)

    behaviour = _oracle_solve_asymptotic_behaviour(lambda point: _variable(point, "11"))
    exponent = float(np.asarray(behaviour, dtype=float).ravel()[0])
    vanishing = int(np.rint(exponent))
    if vanishing < 1:
        raise ValueError("the fall-off condition must impose at least one constraint")

    pole_points = np.array(
        [_variable(complex(root) ** 2, sheet) for root, sheet in zip(roots, sheets)],
        dtype=complex,
    )
    flat_points = np.array(
        [complex(_variable(complex(point).real, "11")).real for point in flat_nodes],
        dtype=float,
    )
    wide_points = np.array(
        [_variable(complex(point).real, "11") for point in wide_nodes], dtype=complex
    )
    coefficients = _oracle_solve_series_coefficients(
        flat_points,
        np.asarray(flat_values, dtype=complex).real.astype(float),
        wide_points,
        wide_values,
        pole_points,
        vanishing,
        _factor,
    )

    target_sheet = sheets[int(target)]

    def _amplitude(points):
        images = np.array(
            [_variable(complex(point), target_sheet) for point in np.atleast_1d(points)],
            dtype=complex,
        )
        return _oracle_evaluate_series_form_factor(images, coefficients, pole_points, _factor)

    residue = _oracle_extract_pole_residue(
        complex(roots[int(target)]) ** 2, _amplitude, radius, nodes
    )
    result = float(abs(residue))
    if not np.isfinite(result):
        raise ValueError("the residue modulus must be finite")
    return result

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases() -> list[dict[str, str]]:
    """Return scalar-only end-to-end test specifications."""
    common = "import numpy as np\n"
    status = (
        "import numpy as np\n"
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
        {   # the problem statement's own configuration
            "setup": common,
            "call": "estimate_pole_residue()",
            "gold_call": "_oracle_estimate_pole_residue()",
        },
        {   # the narrow resonance on the elastic partner sheet
            "setup": common,
            "call": "estimate_pole_residue(target=1, radius=0.006)",
            "gold_call": "_oracle_estimate_pole_residue(target=1, radius=0.006)",
        },
        {   # a pole on the sheet behind the inelastic cut
            "setup": common,
            "call": "estimate_pole_residue(target=2, radius=0.006, nodes=96)",
            "gold_call": "_oracle_estimate_pole_residue(target=2, radius=0.006, nodes=96)",
        },
        {   # a different expansion point moves nothing in the residue by much
            "setup": common,
            "call": "estimate_pole_residue(s_origin=-1.0)",
            "gold_call": "_oracle_estimate_pole_residue(s_origin=-1.0)",
        },
        {   # a heavier inelastic threshold and a shifted left-hand branch point
            "setup": common,
            "call": "estimate_pole_residue(s_inelastic=4.0 * 0.5479 ** 2, s_lhc=-0.05)",
            "gold_call": "_oracle_estimate_pole_residue(s_inelastic=4.0 * 0.5479 ** 2, s_lhc=-0.05)",
        },
        {   # three poles and a shorter measurement set
            "setup": common,
            "call": ("estimate_pole_residue(pole_roots=(0.462 - 0.271j, 0.9935 - 0.0285j, 0.700 - 0.350j),"
                     " pole_sheets=('21', '21', '22'), real_nodes=(-0.40,), real_values=(0.80,))"),
            "gold_call": ("_oracle_estimate_pole_residue(pole_roots=(0.462 - 0.271j, 0.9935 - 0.0285j, 0.700 - 0.350j),"
                          " pole_sheets=('21', '21', '22'), real_nodes=(-0.40,), real_values=(0.80,))"),
        },
        {   # invalid: target outside the pole list
            "setup": status,
            "call": "_status(lambda: estimate_pole_residue(target=7))",
            "gold_call": "_status(lambda: _oracle_estimate_pole_residue(target=7))",
        },
        {   # invalid: measurement values do not match their points
            "setup": status,
            "call": "_status(lambda: estimate_pole_residue(real_nodes=(-2.0, -0.4), real_values=(0.45,)))",
            "gold_call": "_status(lambda: _oracle_estimate_pole_residue(real_nodes=(-2.0, -0.4), real_values=(0.45,)))",
        },
        {   # invalid: thresholds in the wrong order
            "setup": status,
            "call": "_status(lambda: estimate_pole_residue(s_plus=1.2, s_inelastic=0.4))",
            "gold_call": "_status(lambda: _oracle_estimate_pole_residue(s_plus=1.2, s_inelastic=0.4))",
        },
    ]
