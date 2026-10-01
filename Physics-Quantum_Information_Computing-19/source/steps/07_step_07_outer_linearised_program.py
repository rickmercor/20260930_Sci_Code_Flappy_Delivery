"""
Solve one linear program that bounds the single-photon detection or error parameter of the signal setting, using the sifted rates together with the affine relaxation of the confinement constraints taken at a supplied collection of reference points.

Because the photon number of a phase-randomised pulse is never measured, the photon-resolved parameters have to be squeezed between what the sifted rates permit and what the residual distinguishability of the settings permits. Every relaxation point supplied contributes its own pair of half-spaces for every ordered pair of settings and every photon number, and supplying several at once intersects all of them, so the admissible region can only shrink as points are added and the resulting bound can only improve. Any collection of points still yields a conservative bound, because each half-space contains the exact confinement.

The parameters are laid out with the nominal settings along the first axis, in the order supplied, and photon number along the second. The first nominal setting is the signal setting whose single-photon parameter is the objective, and the relaxation-point collection is indexed by its leading axis. The sifted rates supplied must be the ones conditioned on matching bases and the matching setting.

Returns
-------
objective : float — Optimal value of the signal setting's single-photon parameter.; parameters : np.ndarray — Shape (A, n_cut + 1). An optimal assignment of the photon-resolved parameters attaining `objective`.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def solve_outer_linearised_program(
    observed_rates: 'np.ndarray',
    reference_points: 'np.ndarray',
    intensities: 'np.ndarray',
    probabilities: 'np.ndarray',
    delta_max: float,
    correlation_range: int,
    n_cut: int,
    maximise: bool,
) -> 'tuple[float, np.ndarray]':
    '''Bound the signal single-photon parameter with one linear program.

    Parameters
    ----------
    observed_rates : np.ndarray
        Shape (A,). Sifted rate for each nominal setting, conditioned on
        matching bases and that setting. Every entry in [0, 1].
    reference_points : np.ndarray
        Shape (R, A, n_cut + 1) with R >= 1. Relaxation points; entry
        (r, a, n) is the reference value used for setting a at photon number n
        in the r-th point. Every entry in [0, 1].
    intensities : np.ndarray
        Shape (A,). Nominal mean photon numbers, all strictly positive. Entry
        zero is the signal setting.
    probabilities : np.ndarray
        Shape (A,). Selection probability of each nominal setting.
        Non-negative and summing to one.
    delta_max : float
        Maximum relative deviation of every actual mean photon number from its
        nominal setting, in [0, 1).
    correlation_range : int
        Memory span of the intensity correlations, in rounds. At least one.
    n_cut : int
        Photon-number cut-off. At least one.
    maximise : bool
        When False the signal setting's single-photon parameter is minimised,
        giving a lower bound; when True it is maximised, giving an upper bound.
        Required.

    Returns
    -------
    objective : float
        Optimal value of the signal setting's single-photon parameter.
    parameters : np.ndarray
        Shape (A, n_cut + 1). An optimal assignment of the photon-resolved
        parameters attaining `objective`.

    Notes
    -----
    Resolve the objective and affine constraints accurately enough for
    absolute errors below 1e-9. For HiGHS, primal and dual feasibility
    tolerances of 1e-10 with a common factor of 1000 applied to the
    objective and inequalities are sufficient for these instances.
    Undo the objective scaling before returning. Equivalent accurate
    LP implementations are accepted; the scaling changes no constraint.

    Raises
    ------
    ValueError
        If any array argument has the wrong shape or holds values outside its
        stated range, if `n_cut` is smaller than one, if `correlation_range` is
        smaller than one, if `delta_max` is negative or is not below one, or if
        the linear program is not solved to optimality.
    '''
    return objective, parameters

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
from scipy.optimize import linprog


def _oracle_solve_outer_linearised_program(
    observed_rates: "np.ndarray",
    reference_points: "np.ndarray",
    intensities: "np.ndarray",
    probabilities: "np.ndarray",
    delta_max: float,
    correlation_range: int,
    n_cut: int,
    maximise: bool,
) -> "tuple[float, np.ndarray]":
    if isinstance(n_cut, bool) or not isinstance(n_cut, (int, np.integer)):
        raise ValueError("n_cut must be an integer")
    n_cut = int(n_cut)
    if n_cut < 1:
        raise ValueError("n_cut must be at least one")
    if not isinstance(maximise, (bool, np.bool_)):
        raise ValueError("maximise must be a boolean")

    settings, weights = _check_intensity_family(
        intensities, probabilities
    )
    n_set = settings.size
    width = n_cut + 1
    rates = np.asarray(observed_rates, dtype=float)
    if rates.shape != (n_set,):
        raise ValueError("observed_rates must have shape (A,)")
    if (
        not np.all(np.isfinite(rates))
        or np.any(rates < 0.0)
        or np.any(rates > 1.0)
    ):
        raise ValueError("observed_rates must lie in [0, 1]")

    anchors = np.asarray(reference_points, dtype=float)
    if (
        anchors.ndim != 3
        or anchors.shape[1:] != (n_set, width)
        or anchors.shape[0] < 1
    ):
        raise ValueError(
            "reference_points must have shape (R, A, n_cut + 1), R >= 1"
        )
    if (
        not np.all(np.isfinite(anchors))
        or np.any(anchors < 0.0)
        or np.any(anchors > 1.0)
    ):
        raise ValueError("reference_points must lie in [0, 1]")

    rows, limits = [], []
    for index, setting in enumerate(settings):
        lower, upper, cut_mass = _oracle_compute_photon_number_bounds(
            float(setting), delta_max, n_cut
        )
        block = np.zeros((n_set, width))
        block[index] = lower
        rows.append(block.ravel())
        limits.append(float(rates[index]))
        block = np.zeros((n_set, width))
        block[index] = -upper
        rows.append(block.ravel())
        limits.append(float(cut_mass - rates[index]))

    for first in range(n_set):
        for second in range(n_set):
            if first == second:
                continue
            overlaps = _oracle_compute_cs_overlap_parameters(
                float(settings[first]), float(settings[second]),
                settings, weights, delta_max, correlation_range, n_cut,
            )
            for point in anchors:
                coefficients = _oracle_build_cs_tangent_coefficients(
                    point[first], overlaps
                )
                low_slope, low_offset, high_slope, high_offset = coefficients
                for photons in range(width):
                    block = np.zeros((n_set, width))
                    block[first, photons] = low_slope[photons]
                    block[second, photons] -= 1.0
                    rows.append(block.ravel())
                    limits.append(float(-low_offset[photons]))
                    block = np.zeros((n_set, width))
                    block[second, photons] += 1.0
                    block[first, photons] -= high_slope[photons]
                    rows.append(block.ravel())
                    limits.append(float(high_offset[photons]))

    cost = np.zeros(n_set * width)
    cost[1] = -1.0 if maximise else 1.0
    # Common positive scaling preserves the program and resolves tiny
    # affine residuals more accurately than default solver tolerances.
    solution = linprog(
        1000.0 * cost,
        A_ub=1000.0 * np.asarray(rows),
        b_ub=1000.0 * np.asarray(limits),
        bounds=[(0.0, 1.0)] * (n_set * width),
        method="highs",
        options={
            "primal_feasibility_tolerance": 1e-10,
            "dual_feasibility_tolerance": 1e-10,
        },
    )
    if not solution.success and solution.status == 4:
        solution = linprog(
            1000.0 * cost,
            A_ub=1000.0 * np.asarray(rows),
            b_ub=1000.0 * np.asarray(limits),
            bounds=[(0.0, 1.0)] * (n_set * width),
            method="highs-ipm",
            options={
                "primal_feasibility_tolerance": 1e-10,
                "dual_feasibility_tolerance": 1e-10,
            },
        )
    if not solution.success or solution.x is None:
        raise ValueError(
            "linear program not solved to optimality: " + solution.message
        )

    objective = float(cost @ solution.x)
    objective = -objective if maximise else objective
    return objective, np.asarray(solution.x, dtype=float).reshape(
        n_set, width
    )

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


                intensities = np.array([0.48, 0.1, 0.0001])
                probabilities = np.array([0.7, 0.18, 0.12])
                eta = 10 ** (-0.2 * 35.0 / 10.0) * 0.65
                no_dark = (1.0 - 7.2e-08) ** 2
                rates = 1.0 - no_dark * np.exp(-eta * intensities)
                n_cut = 10
                n = np.arange(n_cut + 1, dtype=float)
                silent = (1.0 - eta) ** n
                anchor = np.tile(1.0 - no_dark * silent, (3, 1))[None, :, :]
                args = (
                    (rates, anchor, intensities, probabilities, 0.001, 3,
                    n_cut, False)
                )
                """
            ),
            'call': dedent(
                """\
                _isolated(solve_outer_linearised_program, *args)[0]
                """
            ),
            'gold_call': dedent(
                """\
                _isolated(_oracle_solve_outer_linearised_program, *args)[0]
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


                intensities = np.array([0.48, 0.1, 0.0001])
                probabilities = np.array([0.7, 0.18, 0.12])
                eta = 10 ** (-0.2 * 35.0 / 10.0) * 0.65
                no_dark = (1.0 - 7.2e-08) ** 2
                rates = 1.0 - no_dark * np.exp(-eta * intensities)
                n_cut = 10
                n = np.arange(n_cut + 1, dtype=float)
                silent = (1.0 - eta) ** n
                anchor = np.tile(1.0 - no_dark * silent, (3, 1))[None, :, :]
                args = (
                    (rates, anchor, intensities, probabilities, 0.001, 3,
                    n_cut, True)
                )
                """
            ),
            'call': dedent(
                """\
                _isolated(solve_outer_linearised_program, *args)[0]
                """
            ),
            'gold_call': dedent(
                """\
                _isolated(_oracle_solve_outer_linearised_program, *args)[0]
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


                intensities = np.array([0.48, 0.1, 0.0001])
                probabilities = np.array([0.7, 0.18, 0.12])
                eta = 10 ** (-0.2 * 35.0 / 10.0) * 0.65
                no_dark = (1.0 - 7.2e-08) ** 2
                rates = 1.0 - no_dark * np.exp(-eta * intensities)
                n_cut = 10
                n = np.arange(n_cut + 1, dtype=float)
                silent = (1.0 - eta) ** n
                anchor = np.tile(1.0 - no_dark * silent, (3, 1))[None, :, :]


                def objective_consistency(fn):
                    value, table = (
                        _isolated(fn, rates, anchor, intensities,
                        probabilities, 0.001, 3, n_cut, False)
                    )
                    return (
                        np.array([value, table[0, 1], float(table.shape[0]),
                        float(table.shape[1])])
                    )
                """
            ),
            'call': dedent(
                """\
                objective_consistency(solve_outer_linearised_program)
                """
            ),
            'gold_call': dedent(
                """\
                objective_consistency(_oracle_solve_outer_linearised_program)
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


                intensities = np.array([0.48, 0.1, 0.0001])
                probabilities = np.array([0.7, 0.18, 0.12])
                eta = 10 ** (-0.2 * 35.0 / 10.0) * 0.65
                no_dark = (1.0 - 7.2e-08) ** 2
                rates = 1.0 - no_dark * np.exp(-eta * intensities)
                n_cut = 10
                n = np.arange(n_cut + 1, dtype=float)
                silent = (1.0 - eta) ** n
                anchor = np.tile(1.0 - no_dark * silent, (3, 1))[None, :, :]
                args = (
                    (rates, anchor, intensities, probabilities, 0.0, 3, n_cut,
                    False)
                )
                """
            ),
            'call': dedent(
                """\
                _isolated(solve_outer_linearised_program, *args)[0]
                """
            ),
            'gold_call': dedent(
                """\
                _isolated(_oracle_solve_outer_linearised_program, *args)[0]
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


                intensities = np.array([0.48, 0.1, 0.0001])
                probabilities = np.array([0.7, 0.18, 0.12])
                eta = 10 ** (-0.2 * 35.0 / 10.0) * 0.65
                no_dark = (1.0 - 7.2e-08) ** 2
                rates = 1.0 - no_dark * np.exp(-eta * intensities)
                n_cut = 10
                n = np.arange(n_cut + 1, dtype=float)
                silent = (1.0 - eta) ** n
                anchor = np.tile(1.0 - no_dark * silent, (3, 1))[None, :, :]
                args = (
                    (rates, anchor, intensities, probabilities, 0.001, 1,
                    n_cut, False)
                )
                """
            ),
            'call': dedent(
                """\
                _isolated(solve_outer_linearised_program, *args)[0]
                """
            ),
            'gold_call': dedent(
                """\
                _isolated(_oracle_solve_outer_linearised_program, *args)[0]
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


                intensities = np.array([0.48, 0.1, 0.0001])
                probabilities = np.array([0.7, 0.18, 0.12])
                eta = 0.1296920505
                rates = 1.0 - np.exp(-eta * intensities)
                anchor = np.full((1, 3, 2), 0.13)
                args = (
                    (rates, anchor, intensities, probabilities, 0.001, 3, 1,
                    False)
                )
                """
            ),
            'call': dedent(
                """\
                _isolated(solve_outer_linearised_program, *args)[0]
                """
            ),
            'gold_call': dedent(
                """\
                _isolated(_oracle_solve_outer_linearised_program, *args)[0]
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


                intensities = np.array([0.48, 0.1, 0.0001])
                probabilities = np.array([0.7, 0.18, 0.12])
                eta = 10 ** (-0.2 * 35.0 / 10.0) * 0.65
                no_dark = (1.0 - 7.2e-08) ** 2
                rates = 1.0 - no_dark * np.exp(-eta * intensities)
                n_cut = 10
                n = np.arange(n_cut + 1, dtype=float)
                silent = (1.0 - eta) ** n
                anchor = np.tile(1.0 - no_dark * silent, (3, 1))[None, :, :]
                args = (
                    (rates, anchor, intensities, probabilities, 0.001, 3,
                    n_cut, False)
                )
                second = np.full((3, n_cut + 1), 0.5)
                stacked = np.concatenate([anchor, second[None, :, :]], axis=0)
                args = (
                    (rates, stacked, intensities, probabilities, 0.001, 3,
                    n_cut, False)
                )
                """
            ),
            'call': dedent(
                """\
                _isolated(solve_outer_linearised_program, *args)[0]
                """
            ),
            'gold_call': dedent(
                """\
                _isolated(_oracle_solve_outer_linearised_program, *args)[0]
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


                intensities = np.array([0.48, 0.1, 0.0001])
                probabilities = np.array([0.7, 0.18, 0.12])
                eta = 10 ** (-0.2 * 35.0 / 10.0) * 0.65
                no_dark = (1.0 - 7.2e-08) ** 2
                rates = 1.0 - no_dark * np.exp(-eta * intensities)
                n_cut = 10
                n = np.arange(n_cut + 1, dtype=float)
                silent = (1.0 - eta) ** n
                anchor = np.tile(1.0 - no_dark * silent, (3, 1))[None, :, :]
                args = (
                    (rates, np.full((1, 3, n_cut + 1), 1e-09), intensities,
                    probabilities, 0.005, 3, n_cut, False)
                )
                """
            ),
            'call': dedent(
                """\
                _isolated(solve_outer_linearised_program, *args)[0]
                """
            ),
            'gold_call': dedent(
                """\
                _isolated(_oracle_solve_outer_linearised_program, *args)[0]
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


                intensities = np.array([0.48, 0.1, 0.0001])
                probabilities = np.array([0.7, 0.18, 0.12])
                eta = 10 ** (-0.2 * 35.0 / 10.0) * 0.65
                no_dark = (1.0 - 7.2e-08) ** 2
                rates = 1.0 - no_dark * np.exp(-eta * intensities)
                n_cut = 10
                n = np.arange(n_cut + 1, dtype=float)
                silent = (1.0 - eta) ** n
                anchor = np.tile(1.0 - no_dark * silent, (3, 1))[None, :, :]
                args = (
                    (rates, anchor, intensities, probabilities, 0.02, 5,
                    n_cut, False)
                )
                """
            ),
            'call': dedent(
                """\
                _isolated(solve_outer_linearised_program, *args)[0]
                """
            ),
            'gold_call': dedent(
                """\
                _isolated(_oracle_solve_outer_linearised_program, *args)[0]
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


                intensities = np.array([0.48, 0.1, 0.0001])
                probabilities = np.array([0.7, 0.18, 0.12])
                eta = 0.1296920505
                pd, dA = (7.2e-08, 0.08)
                no_dark = (1.0 - pd) ** 2
                imbalance = (
                    0.5 * (np.exp(-eta * intensities * np.cos(dA) ** 2) - np
                    .exp(-eta * intensities * np.sin(dA) ** 2))
                )
                rates = (
                    0.5 * pd ** 2 + pd * (1.0 - pd) * (1.0 + imbalance) +
                    no_dark * (0.5 + imbalance - 0.5 * np.exp(-eta
                    * intensities))
                )
                n_cut = 10
                n = np.arange(n_cut + 1, dtype=float)
                silent = (1.0 - eta) ** n
                aligned = (eta * np.cos(dA) ** 2 + 1.0 - eta) ** n - silent
                crossed = (eta * np.sin(dA) ** 2 + 1.0 - eta) ** n - silent
                both = 1.0 - silent - aligned - crossed
                ref = (
                    no_dark * (crossed + 0.5 * both) + pd * (1.0 - pd) * (0.5
                    * (crossed + both) + silent + crossed + 0.5 * (aligned +
                    both)) + 0.5 * pd ** 2
                )
                anchor = np.tile(ref, (3, 1))[None, :, :]
                args = (
                    (rates, anchor, intensities, probabilities, 0.001, 3,
                    n_cut, True)
                )
                """
            ),
            'call': dedent(
                """\
                _isolated(solve_outer_linearised_program, *args)[0]
                """
            ),
            'gold_call': dedent(
                """\
                _isolated(_oracle_solve_outer_linearised_program, *args)[0]
                """
            ),
            'tol': 1e-11,
        },
        {
            'setup': dedent(
                """\
                from copy import deepcopy

                import numpy as np


                def _isolated(function, *arguments, **keywords):
                    arguments, keywords = deepcopy((arguments, keywords))
                    return function(*arguments, **keywords)


                intensities = np.array([0.5, 0.0001])
                probabilities = np.array([0.85, 0.15])
                eta = 0.1296920505
                rates = 1.0 - np.exp(-eta * intensities)
                anchor = np.full((1, 2, 9), 0.12)
                args = (
                    (rates, anchor, intensities, probabilities, 0.001, 2, 8,
                    False)
                )
                """
            ),
            'call': dedent(
                """\
                _isolated(solve_outer_linearised_program, *args)[0]
                """
            ),
            'gold_call': dedent(
                """\
                _isolated(_oracle_solve_outer_linearised_program, *args)[0]
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


                intensities = np.array([0.48, 0.1, 0.0001])
                probabilities = np.array([0.7, 0.18, 0.12])
                eta = 10 ** (-0.2 * 35.0 / 10.0) * 0.65
                no_dark = (1.0 - 7.2e-08) ** 2
                rates = 1.0 - no_dark * np.exp(-eta * intensities)
                n_cut = 10
                n = np.arange(n_cut + 1, dtype=float)
                silent = (1.0 - eta) ** n
                anchor = np.tile(1.0 - no_dark * silent, (3, 1))[None, :, :]


                def run_model(fn, *args):
                    try:
                        _isolated(fn, *args)
                        return 0.0
                    except ValueError:
                        return 1.0


                args = (
                    (rates, np.full((1, 3, n_cut), 0.2), intensities,
                    probabilities, 0.001, 3, n_cut, False)
                )
                """
            ),
            'call': dedent(
                """\
                run_model(solve_outer_linearised_program, *args)
                """
            ),
            'gold_call': dedent(
                """\
                run_model(_oracle_solve_outer_linearised_program, *args)
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


                intensities = np.array([0.48, 0.1, 0.0001])
                probabilities = np.array([0.7, 0.18, 0.12])
                eta = 10 ** (-0.2 * 35.0 / 10.0) * 0.65
                no_dark = (1.0 - 7.2e-08) ** 2
                rates = 1.0 - no_dark * np.exp(-eta * intensities)
                n_cut = 10
                n = np.arange(n_cut + 1, dtype=float)
                silent = (1.0 - eta) ** n
                anchor = np.tile(1.0 - no_dark * silent, (3, 1))[None, :, :]


                def run_model(fn, *args):
                    try:
                        _isolated(fn, *args)
                        return 0.0
                    except ValueError:
                        return 1.0


                args = (
                    (rates, np.full((1, 3, 1), 0.2), intensities,
                    probabilities, 0.001, 3, 0, False)
                )
                """
            ),
            'call': dedent(
                """\
                run_model(solve_outer_linearised_program, *args)
                """
            ),
            'gold_call': dedent(
                """\
                run_model(_oracle_solve_outer_linearised_program, *args)
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


                intensities = np.array([0.48, 0.1, 0.0001])
                probabilities = np.array([0.7, 0.18, 0.12])
                eta = 10 ** (-0.2 * 35.0 / 10.0) * 0.65
                no_dark = (1.0 - 7.2e-08) ** 2
                rates = 1.0 - no_dark * np.exp(-eta * intensities)
                n_cut = 10
                n = np.arange(n_cut + 1, dtype=float)
                silent = (1.0 - eta) ** n
                anchor = np.tile(1.0 - no_dark * silent, (3, 1))[None, :, :]


                def run_model(fn, *args):
                    try:
                        _isolated(fn, *args)
                        return 0.0
                    except ValueError:
                        return 1.0


                args = (
                    (np.array([1.5, 0.01, 1e-05]), anchor, intensities,
                    probabilities, 0.001, 3, n_cut, False)
                )
                """
            ),
            'call': dedent(
                """\
                run_model(solve_outer_linearised_program, *args)
                """
            ),
            'gold_call': dedent(
                """\
                run_model(_oracle_solve_outer_linearised_program, *args)
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


                intensities = np.array([0.48, 0.1, 0.0001])
                probabilities = np.array([0.7, 0.18, 0.12])
                eta = 10 ** (-0.2 * 35.0 / 10.0) * 0.65
                no_dark = (1.0 - 7.2e-08) ** 2
                rates = 1.0 - no_dark * np.exp(-eta * intensities)
                n_cut = 10
                n = np.arange(n_cut + 1, dtype=float)
                silent = (1.0 - eta) ** n
                anchor = np.tile(1.0 - no_dark * silent, (3, 1))[None, :, :]


                def run_model(fn, *args):
                    try:
                        _isolated(fn, *args)
                        return 0.0
                    except ValueError:
                        return 1.0


                args = (
                    (rates, np.full((1, 3, n_cut + 1), 1.4), intensities,
                    probabilities, 0.001, 3, n_cut, False)
                )
                """
            ),
            'call': dedent(
                """\
                run_model(solve_outer_linearised_program, *args)
                """
            ),
            'gold_call': dedent(
                """\
                run_model(_oracle_solve_outer_linearised_program, *args)
                """
            ),
        },
    ]
