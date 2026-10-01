"""
Build the affine coefficients that replace each nonlinear confinement interval by a pair of half-spaces touching it at a chosen reference value, so that the confinement can be carried into a linear program.

The confinement of a detection parameter by its paired value is curved, and its lower and upper endpoints are respectively a convex and a concave function of that paired value. Replacing each endpoint by its tangent at a chosen reference value therefore widens the admissible region rather than narrowing it, which keeps any bound obtained from the widened region conservative no matter where the reference value is placed; how much is given away, however, depends entirely on that placement. The pair of affine functions returned here is exact at the reference value and relaxes away from it.

The four returned arrays have the broadcast shape of the inputs and are ordered as the slope and then the offset of the lower half-space, followed by the slope and then the offset of the upper half-space, each defined so that the lower half-space reads offset plus slope times the paired value, and likewise for the upper one. Reference values are first moved to lie at least `tangent_floor` inside the unit interval, which leaves the half-spaces conservative.

Returns
-------
lower_slope : np.ndarray — Slope of the lower half-space, of the broadcast shape.; lower_offset : np.ndarray — Offset of the lower half-space, of the broadcast shape.; upper_slope : np.ndarray — Slope of the upper half-space, of the broadcast shape.; upper_offset : np.ndarray — Offset of the upper half-space, of the broadcast shape.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def build_cs_tangent_coefficients(reference_values: "np.ndarray", overlaps: "np.ndarray", tangent_floor: float = 1e-12) -> "tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]":
    '''Build the affine relaxation of the confinement interval at a reference value.

    Parameters
    ----------
    reference_values : np.ndarray
        Reference values of the paired setting's parameter at which the
        relaxation is taken. Every entry in [0, 1]. Scalars are accepted.
    overlaps : np.ndarray
        Overlap parameter of the pair at the matching photon number. Every
        entry in (0, 1]. Broadcasts against `reference_values`.
    tangent_floor : float
        Distance from each end of the unit interval inside which a reference
        value is moved before the relaxation is taken. In (0, 0.5).

    Returns
    -------
    lower_slope : np.ndarray
        Slope of the lower half-space, of the broadcast shape.
    lower_offset : np.ndarray
        Offset of the lower half-space, of the broadcast shape.
    upper_slope : np.ndarray
        Slope of the upper half-space, of the broadcast shape.
    upper_offset : np.ndarray
        Offset of the upper half-space, of the broadcast shape.

    Raises
    ------
    ValueError
        If any entry of `reference_values` is outside [0, 1], if any entry of
        `overlaps` is outside (0, 1], if either input holds a non-finite value,
        if the two inputs do not broadcast against one another, or if
        `tangent_floor` is outside (0, 0.5).
    '''
    return lower_slope, lower_offset, upper_slope, upper_offset

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_build_cs_tangent_coefficients(reference_values: "np.ndarray", overlaps: "np.ndarray", tangent_floor: float = 1e-12) -> "tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]":
    if isinstance(tangent_floor, bool) or not isinstance(tangent_floor, (int, float, np.integer, np.floating)):
        raise ValueError("tangent_floor must be a real scalar")
    tangent_floor = float(tangent_floor)
    if not np.isfinite(tangent_floor) or tangent_floor <= 0.0 or tangent_floor >= 0.5:
        raise ValueError("tangent_floor must lie in (0, 0.5)")

    values = np.asarray(reference_values, dtype=float)
    weights = np.asarray(overlaps, dtype=float)
    if not np.all(np.isfinite(values)) or not np.all(np.isfinite(weights)):
        raise ValueError("inputs must be finite")
    if np.any(values < 0.0) or np.any(values > 1.0):
        raise ValueError("reference_values must lie in [0, 1]")
    if np.any(weights <= 0.0) or np.any(weights > 1.0):
        raise ValueError("overlaps must lie in (0, 1]")
    try:
        values, weights = np.broadcast_arrays(values, weights)
    except ValueError as exc:
        raise ValueError("reference_values and overlaps must broadcast") from exc

    anchor = np.clip(values, tangent_floor, 1.0 - tangent_floor)
    slack = 1.0 - weights
    lower_value, upper_value = _oracle_evaluate_cs_boundaries(anchor, weights)

    curvature = np.sqrt(weights * slack / (anchor * (1.0 - anchor)))
    tilt = (1.0 - 2.0 * anchor) * curvature
    lower_slope = np.where(anchor > slack, -1.0 + 2.0 * weights - tilt, 0.0)
    upper_slope = np.where(anchor < weights, -1.0 + 2.0 * weights + tilt, 0.0)

    lower_offset = lower_value - lower_slope * anchor
    upper_offset = upper_value - upper_slope * anchor
    return (np.asarray(lower_slope, dtype=float), np.asarray(lower_offset, dtype=float),
            np.asarray(upper_slope, dtype=float), np.asarray(upper_offset, dtype=float))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Retain scientific cases with independent invocation inputs."""
    from textwrap import dedent

    return [
        {
            'setup': dedent(
                """\
                from copy import deepcopy

                import numpy as np


                def _isolated(function, *arguments, **keywords):
                    arguments, keywords = deepcopy((arguments, keywords))
                    return function(*arguments, **keywords)


                values = (
                    np.array([1.4399999e-07, 0.1296920505, 0.2426, 0.3417,
                    0.428])
                )
                weights = (
                    np.array([0.996355331, 0.994285462, 0.990316263, 0.984394,
                    0.976596])
                )
                """
            ),
            'call': dedent(
                """\
                _isolated(build_cs_tangent_coefficients, values, weights)
                """
            ),
            'gold_call': dedent(
                """\
                (
                    _isolated(_oracle_build_cs_tangent_coefficients, values,
                    weights)
                )
                """
            ),
        },
        {
            'setup': dedent(
                """\
                from copy import deepcopy

                import numpy as np


                def _isolated(function, *arguments, **keywords):
                    arguments, keywords = deepcopy((arguments, keywords))
                    return function(*arguments, **keywords)


                values = (
                    np.array([7.19999e-08, 0.00082836, 0.0016478, 0.0024593])
                )
                weights = np.full(4, 0.9944841402)
                """
            ),
            'call': dedent(
                """\
                _isolated(build_cs_tangent_coefficients, values, weights)
                """
            ),
            'gold_call': dedent(
                """\
                (
                    _isolated(_oracle_build_cs_tangent_coefficients, values,
                    weights)
                )
                """
            ),
        },
        {
            'setup': dedent(
                """\
                from copy import deepcopy

                import numpy as np


                def _isolated(function, *arguments, **keywords):
                    arguments, keywords = deepcopy((arguments, keywords))
                    return function(*arguments, **keywords)


                values = np.array([0.02, 0.25, 0.5, 0.9])
                weights = np.ones(4)
                """
            ),
            'call': dedent(
                """\
                _isolated(build_cs_tangent_coefficients, values, weights)
                """
            ),
            'gold_call': dedent(
                """\
                (
                    _isolated(_oracle_build_cs_tangent_coefficients, values,
                    weights)
                )
                """
            ),
            'tol': 1e-12,
        },
        {
            'setup': dedent(
                """\
                from copy import deepcopy

                import numpy as np


                def _isolated(function, *arguments, **keywords):
                    arguments, keywords = deepcopy((arguments, keywords))
                    return function(*arguments, **keywords)


                values = np.array([0.0, 1.0])
                weights = np.array([0.97, 0.97])
                """
            ),
            'call': dedent(
                """\
                _isolated(build_cs_tangent_coefficients, values, weights)
                """
            ),
            'gold_call': dedent(
                """\
                (
                    _isolated(_oracle_build_cs_tangent_coefficients, values,
                    weights)
                )
                """
            ),
            'tol': 1e-09,
        },
        {
            'setup': dedent(
                """\
                from copy import deepcopy

                import numpy as np


                def _isolated(function, *arguments, **keywords):
                    arguments, keywords = deepcopy((arguments, keywords))
                    return function(*arguments, **keywords)


                weights = np.full(7, 0.88)
                values = (
                    np.array([0.0, 0.05, 0.1199999, 0.12, 0.1200001,
                    0.8799999, 0.92])
                )
                """
            ),
            'call': dedent(
                """\
                _isolated(build_cs_tangent_coefficients, values, weights)
                """
            ),
            'gold_call': dedent(
                """\
                (
                    _isolated(_oracle_build_cs_tangent_coefficients, values,
                    weights)
                )
                """
            ),
            'tol': 1e-09,
        },
        {
            'setup': dedent(
                """\
                from copy import deepcopy

                import numpy as np


                def _isolated(function, *arguments, **keywords):
                    arguments, keywords = deepcopy((arguments, keywords))
                    return function(*arguments, **keywords)


                def contact_residual(fn, boundary_fn):
                    values = np.array([0.03, 0.2, 0.55, 0.8])
                    weights = np.array([0.99, 0.95, 0.88, 0.7])
                    ls, lo, us, uo = _isolated(fn, values, weights)
                    lo_exact, hi_exact = (
                        _isolated(boundary_fn, values, weights)
                    )
                    return (
                        np.concatenate([lo + ls * values - lo_exact,
                        uo + us * values - hi_exact])
                    )
                """
            ),
            'call': dedent(
                """\
                (
                    contact_residual(build_cs_tangent_coefficients,
                    evaluate_cs_boundaries)
                )
                """
            ),
            'gold_call': dedent(
                """\
                (
                    contact_residual(_oracle_build_cs_tangent_coefficients,
                    _oracle_evaluate_cs_boundaries)
                )
                """
            ),
            'tol': 1e-12,
        },
        {
            'setup': dedent(
                """\
                from copy import deepcopy

                import numpy as np


                def _isolated(function, *arguments, **keywords):
                    arguments, keywords = deepcopy((arguments, keywords))
                    return function(*arguments, **keywords)


                def relaxation_margin(fn, boundary_fn):
                    grid = np.linspace(0.02, 0.98, 25)
                    ref = np.full(25, 0.4)
                    weights = np.full(25, 0.85)
                    ls, lo, us, uo = _isolated(fn, ref, weights)
                    lo_exact, hi_exact = _isolated(boundary_fn, grid, weights)
                    return (
                        np.array([float(np.max(lo + ls * grid - lo_exact)),
                        float(np.min(uo + us * grid - hi_exact))])
                    )
                """
            ),
            'call': dedent(
                """\
                (
                    relaxation_margin(build_cs_tangent_coefficients,
                    evaluate_cs_boundaries)
                )
                """
            ),
            'gold_call': dedent(
                """\
                (
                    relaxation_margin(_oracle_build_cs_tangent_coefficients,
                    _oracle_evaluate_cs_boundaries)
                )
                """
            ),
            'tol': 1e-12,
        },
        {
            'setup': dedent(
                """\
                from copy import deepcopy

                import numpy as np


                def _isolated(function, *arguments, **keywords):
                    arguments, keywords = deepcopy((arguments, keywords))
                    return function(*arguments, **keywords)


                values = 0.35
                weights = np.array([[0.6, 0.5], [0.4, 0.3]])
                """
            ),
            'call': dedent(
                """\
                _isolated(build_cs_tangent_coefficients, values, weights)
                """
            ),
            'gold_call': dedent(
                """\
                (
                    _isolated(_oracle_build_cs_tangent_coefficients, values,
                    weights)
                )
                """
            ),
        },
        {
            'setup': dedent(
                """\
                from copy import deepcopy

                import numpy as np


                def _isolated(function, *arguments, **keywords):
                    arguments, keywords = deepcopy((arguments, keywords))
                    return function(*arguments, **keywords)


                values = np.array([0.0, 1e-13, 0.5, 1.0])
                weights = np.full(4, 0.93)
                """
            ),
            'call': dedent(
                """\
                (
                    _isolated(build_cs_tangent_coefficients, values, weights,
                    1e-06)
                )
                """
            ),
            'gold_call': dedent(
                """\
                (
                    _isolated(_oracle_build_cs_tangent_coefficients, values,
                    weights, 1e-06)
                )
                """
            ),
            'tol': 1e-09,
        },
        {
            'setup': dedent(
                """\
                from copy import deepcopy

                import numpy as np


                def _isolated(function, *arguments, **keywords):
                    arguments, keywords = deepcopy((arguments, keywords))
                    return function(*arguments, **keywords)


                def run_model(fn, *args):
                    try:
                        _isolated(fn, *args)
                        return 0.0
                    except ValueError:
                        return 1.0


                args = (np.array([-1e-09, 0.5]), np.array([0.99, 0.99]))
                """
            ),
            'call': dedent(
                """\
                run_model(build_cs_tangent_coefficients, *args)
                """
            ),
            'gold_call': dedent(
                """\
                run_model(_oracle_build_cs_tangent_coefficients, *args)
                """
            ),
        },
        {
            'setup': dedent(
                """\
                from copy import deepcopy

                import numpy as np


                def _isolated(function, *arguments, **keywords):
                    arguments, keywords = deepcopy((arguments, keywords))
                    return function(*arguments, **keywords)


                def run_model(fn, *args):
                    try:
                        _isolated(fn, *args)
                        return 0.0
                    except ValueError:
                        return 1.0


                args = (np.array([0.5]), np.array([1.5]))
                """
            ),
            'call': dedent(
                """\
                run_model(build_cs_tangent_coefficients, *args)
                """
            ),
            'gold_call': dedent(
                """\
                run_model(_oracle_build_cs_tangent_coefficients, *args)
                """
            ),
        },
        {
            'setup': dedent(
                """\
                from copy import deepcopy

                import numpy as np


                def _isolated(function, *arguments, **keywords):
                    arguments, keywords = deepcopy((arguments, keywords))
                    return function(*arguments, **keywords)


                def run_model(fn, *args):
                    try:
                        _isolated(fn, *args)
                        return 0.0
                    except ValueError:
                        return 1.0


                args = (np.array([0.5]), np.array([0.9]), 0.5)
                """
            ),
            'call': dedent(
                """\
                run_model(build_cs_tangent_coefficients, *args)
                """
            ),
            'gold_call': dedent(
                """\
                run_model(_oracle_build_cs_tangent_coefficients, *args)
                """
            ),
        },
    ]
