"""
Evaluate the closed interval into which the detection parameter belonging to one nominal intensity setting is confined, given the value of the same parameter for the paired setting and the pair's photon-number-resolved overlap parameter.

At a fixed photon number, the conditional detection probability of a signal and the conditional error probability of a signal both behave as a probability attached to a nominal intensity setting, and the residual distinguishability of two settings limits how far the values attached to them can be driven apart. The narrower that residual distinguishability, the tighter the confinement, and when two settings become perfectly indistinguishable the interval closes onto the paired value itself. Because every such parameter is a probability, both reported endpoints always lie inside the unit interval and the lower never exceeds the upper.

Inputs broadcast against one another elementwise, and the two returned arrays have the broadcast shape, the first holding the lower endpoints and the second the upper endpoints.

Returns
-------
lower : np.ndarray — Lower endpoint of the confinement interval, of the broadcast shape.; upper : np.ndarray — Upper endpoint of the confinement interval, of the broadcast shape.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def evaluate_cs_boundaries(parameter_values: "np.ndarray", overlaps: "np.ndarray") -> "tuple[np.ndarray, np.ndarray]":
    '''Evaluate the confinement interval endpoints implied by one paired parameter value.

    Parameters
    ----------
    parameter_values : np.ndarray
        Values of the detection or error parameter attached to the paired
        nominal setting. Every entry in [0, 1]. Scalars are accepted.
    overlaps : np.ndarray
        Overlap parameter of the pair at the matching photon number. Every
        entry in (0, 1]. Broadcasts against `parameter_values`.

    Returns
    -------
    lower : np.ndarray
        Lower endpoint of the confinement interval, of the broadcast shape.
    upper : np.ndarray
        Upper endpoint of the confinement interval, of the broadcast shape.

    Raises
    ------
    ValueError
        If any entry of `parameter_values` is outside [0, 1], if any entry of
        `overlaps` is outside (0, 1], if either input holds a non-finite value,
        or if the two inputs do not broadcast against one another.
    '''
    return lower, upper

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_evaluate_cs_boundaries(parameter_values: "np.ndarray", overlaps: "np.ndarray") -> "tuple[np.ndarray, np.ndarray]":
    values = np.asarray(parameter_values, dtype=float)
    weights = np.asarray(overlaps, dtype=float)
    if not np.all(np.isfinite(values)) or not np.all(np.isfinite(weights)):
        raise ValueError("inputs must be finite")
    if np.any(values < 0.0) or np.any(values > 1.0):
        raise ValueError("parameter_values must lie in [0, 1]")
    if np.any(weights <= 0.0) or np.any(weights > 1.0):
        raise ValueError("overlaps must lie in (0, 1]")
    try:
        values, weights = np.broadcast_arrays(values, weights)
    except ValueError as exc:
        raise ValueError("parameter_values and overlaps must broadcast") from exc

    slack = 1.0 - weights
    centre = values + slack * (1.0 - 2.0 * values)
    radius = 2.0 * np.sqrt(np.clip(weights * slack * values * (1.0 - values), 0.0, None))

    lower = np.where(values > slack, centre - radius, 0.0)
    upper = np.where(values < weights, centre + radius, 1.0)
    return np.asarray(lower, dtype=float), np.asarray(upper, dtype=float)

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
                    np.array([0.0377879072, 0.1296920505, 0.0128856094, 0.5])
                )
                weights = np.full(4, 0.9944841402)
                """
            ),
            'call': dedent(
                """\
                _isolated(evaluate_cs_boundaries, values, weights)
                """
            ),
            'gold_call': dedent(
                """\
                _isolated(_oracle_evaluate_cs_boundaries, values, weights)
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


                values = np.full(6, 0.06035424227292)
                weights = (
                    np.array([0.996355331, 0.994285462, 0.990316263, 0.982436,
                    0.966921, 0.959127808])
                )
                """
            ),
            'call': dedent(
                """\
                _isolated(evaluate_cs_boundaries, values, weights)
                """
            ),
            'gold_call': dedent(
                """\
                _isolated(_oracle_evaluate_cs_boundaries, values, weights)
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


                values = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
                weights = np.ones(5)
                """
            ),
            'call': dedent(
                """\
                _isolated(evaluate_cs_boundaries, values, weights)
                """
            ),
            'gold_call': dedent(
                """\
                _isolated(_oracle_evaluate_cs_boundaries, values, weights)
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
                _isolated(evaluate_cs_boundaries, values, weights)
                """
            ),
            'gold_call': dedent(
                """\
                _isolated(_oracle_evaluate_cs_boundaries, values, weights)
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


                weights = np.full(9, 0.9)
                values = (
                    np.array([0.0, 0.05, 0.0999999, 0.1, 0.1000001, 0.5,
                    0.8999999, 0.9, 0.95])
                )
                """
            ),
            'call': dedent(
                """\
                _isolated(evaluate_cs_boundaries, values, weights)
                """
            ),
            'gold_call': dedent(
                """\
                _isolated(_oracle_evaluate_cs_boundaries, values, weights)
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


                values = np.array([1.4399999e-07, 1.311311907e-05, 0.001])
                weights = np.full(3, 0.996355330965)
                """
            ),
            'call': dedent(
                """\
                _isolated(evaluate_cs_boundaries, values, weights)
                """
            ),
            'gold_call': dedent(
                """\
                _isolated(_oracle_evaluate_cs_boundaries, values, weights)
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


                values = np.array([0.02, 0.2, 0.6])
                weights = np.array([0.55, 0.55, 0.55])
                """
            ),
            'call': dedent(
                """\
                _isolated(evaluate_cs_boundaries, values, weights)
                """
            ),
            'gold_call': dedent(
                """\
                _isolated(_oracle_evaluate_cs_boundaries, values, weights)
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


                values = 0.3
                weights = np.array([[0.999, 0.99], [0.95, 0.9]])
                """
            ),
            'call': dedent(
                """\
                _isolated(evaluate_cs_boundaries, values, weights)
                """
            ),
            'gold_call': dedent(
                """\
                _isolated(_oracle_evaluate_cs_boundaries, values, weights)
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


                def containment(fn):
                    values = np.linspace(0.0, 1.0, 21)
                    weights = np.full(21, 0.8)
                    lo, hi = _isolated(fn, values, weights)
                    return (
                        np.array([float(np.min(lo)), float(np.max(hi)),
                        float(np.min(hi - lo))])
                    )
                """
            ),
            'call': dedent(
                """\
                containment(evaluate_cs_boundaries)
                """
            ),
            'gold_call': dedent(
                """\
                containment(_oracle_evaluate_cs_boundaries)
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


                def run_model(fn, *args):
                    try:
                        _isolated(fn, *args)
                        return 0.0
                    except ValueError:
                        return 1.0


                args = (np.array([0.5, 1.2]), np.array([0.99, 0.99]))
                """
            ),
            'call': dedent(
                """\
                run_model(evaluate_cs_boundaries, *args)
                """
            ),
            'gold_call': dedent(
                """\
                run_model(_oracle_evaluate_cs_boundaries, *args)
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


                args = (np.array([0.5]), np.array([1.0001]))
                """
            ),
            'call': dedent(
                """\
                run_model(evaluate_cs_boundaries, *args)
                """
            ),
            'gold_call': dedent(
                """\
                run_model(_oracle_evaluate_cs_boundaries, *args)
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


                args = (np.array([0.5]), np.array([0.0]))
                """
            ),
            'call': dedent(
                """\
                run_model(evaluate_cs_boundaries, *args)
                """
            ),
            'gold_call': dedent(
                """\
                run_model(_oracle_evaluate_cs_boundaries, *args)
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


                args = (np.array([0.2, 0.3, 0.4]), np.array([0.99, 0.98]))
                """
            ),
            'call': dedent(
                """\
                run_model(evaluate_cs_boundaries, *args)
                """
            ),
            'gold_call': dedent(
                """\
                run_model(_oracle_evaluate_cs_boundaries, *args)
                """
            ),
        },
    ]
