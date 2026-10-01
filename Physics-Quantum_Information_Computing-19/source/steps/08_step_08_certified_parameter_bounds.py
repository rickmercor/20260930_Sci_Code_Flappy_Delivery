"""
Resolve the exact Cauchy-Schwarz-constrained parameter-estimation problems, then return conservative bounds from one final LP per problem, anchored only at its numerically feasible solution.

For each observed-rate vector r, use variables V[a,n] in [0,1],

with settings along the first axis and n=0,...,n_cut. The lower and upper

photon probabilities and upper-end tail are those of Step 01. Impose

sum_n lower[a,n]*V[a,n] <= r[a] and

sum_n upper[a,n]*V[a,n] >= r[a]-tail[a].

For every ordered pair a != b and every n, impose the original nonlinear

bounds G_minus(V[a,n], tau[a,b,n]) <= V[b,n] <=

G_plus(V[a,n], tau[a,b,n]), using Steps 02 and 03.



Minimise V[0,1] for each detection-rate vector and maximise it for the

check-basis error-rate vector. The supplied photon-number references are

starting points, not substitutes for the resulting solutions. One valid

method successively accumulates outer tangents, checks relative objective

stability, and polishes a candidate under the exact inequalities. An

equivalent smooth parametrisation is V[a,n]=sin(theta[a,n])**2 with

theta in [0,pi/2]; each CS pair then requires

abs(theta[a,n]-theta[b,n]) <= acos(sqrt(tau[a,b,n])).

No particular nonlinear optimizer is required.



For budgets above one, check original-constraint violations <=1e-9,

then solve a new Step 07 LP using just the accepted candidate as its

single reference table. Return that LP objective and its absolute gap

from the candidate objective; require gap <=1e-8. These are numerical

feasibility and gap checks, not a claim of mathematically exact equality.

Raise ValueError if the budget is exhausted before all checks pass.



Retain the deliberate max_iterations=1 mode: solve at the supplied

channel reference and return that bound, with the absolute difference

from the reference's signal one-photon value as a diagnostic residual.

It makes no claim that the unrelaxed optimum has been reached.

Returns
-------
tuple[float, float, float, float, float, float]: key-basis detection lower bound, check-basis detection lower bound, check-basis error upper bound, followed by their candidate-to-final-LP absolute gaps. For max_iterations=1 the gaps use the supplied reference objectives and are diagnostics only.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def compute_certified_parameter_bounds(
    key_basis_rates: 'np.ndarray',
    check_basis_rates: 'np.ndarray',
    check_basis_error_rates: 'np.ndarray',
    yield_reference: 'np.ndarray',
    error_reference: 'np.ndarray',
    intensities: 'np.ndarray',
    probabilities: 'np.ndarray',
    delta_max: float,
    correlation_range: int,
    n_cut: int,
    max_iterations: int = 40,
    objective_rtol: float = 1e-12,
) -> 'tuple[float, float, float, float, float, float]':
    '''Certify bounds on the signal setting's single-photon parameters.

    Parameters
    ----------
    key_basis_rates : np.ndarray
        Shape (A,). Sifted detection rate per setting in the key basis.
    check_basis_rates : np.ndarray
        Shape (A,). Sifted detection rate per setting in the check basis.
    check_basis_error_rates : np.ndarray
        Shape (A,). Sifted error rate per setting in the check basis.
    yield_reference : np.ndarray
        Shape (n_cut + 1,). Starting reference values for the two detection
        programs, indexed by photon number. Every entry in [0, 1].
    error_reference : np.ndarray
        Shape (n_cut + 1,). Starting reference values for the error program,
        indexed by photon number. Every entry in [0, 1].
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
    max_iterations : int
        Positive refinement budget for each parameter-estimation problem.
        For the accumulated-tangent method, count its initial LP and each
        subsequent refinement LP, up to this limit. The final certifying
        LP and numerical work to obtain an exact-feasible candidate are
        additional. A value of one deliberately returns the initial
        channel-reference LP bound without claiming exact optimality.
    objective_rtol : float
        Strictly positive relative stopping tolerance. Compare adjacent
        refinement objectives v_old and v_new using
        abs(v_new-v_old) <= objective_rtol *
        max(abs(v_new), abs(v_old), np.finfo(float).tiny).
        For a budget above one, also require all original nonlinear and
        observed-rate inequalities within absolute 1e-9 and the gap
        between the candidate objective and its single-anchor LP bound
        at most 1e-8. Objective stability alone is insufficient.

    Returns
    -------
    key_yield_bound : float
        Lower bound from the final single-anchor LP in the key basis.
        Return this LP value, not the accumulated-refinement objective.
    check_yield_bound : float
        Lower bound on the same parameter in the check basis.
    error_bound : float
        Upper bound on the signal setting's single-photon error parameter in
        the check basis.
    key_yield_residual : float
        Absolute difference between the exact-feasible candidate's
        objective and the returned single-anchor LP bound. In one-pass
        mode, use the supplied reference's signal one-photon value;
        that residual is diagnostic and does not certify optimality.
    check_yield_residual : float
        The same residual for `check_yield_bound`.
    error_residual : float
        The same residual for `error_bound`.

    Raises
    ------
    ValueError
        If any array argument has the wrong shape or holds values outside its
        stated range, if `max_iterations` is smaller than one, if
        `objective_rtol` is not strictly positive, any LP is unsuccessful,
        or the budget above one is exhausted without meeting the relative,
        original-feasibility and final-gap conditions.
    '''
    return (key_yield_bound, check_yield_bound, error_bound,
            key_yield_residual, check_yield_residual, error_residual)

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
from scipy.optimize import LinearConstraint, minimize


def _oracle_compute_certified_parameter_bounds(
    key_basis_rates: "np.ndarray",
    check_basis_rates: "np.ndarray",
    check_basis_error_rates: "np.ndarray",
    yield_reference: "np.ndarray",
    error_reference: "np.ndarray",
    intensities: "np.ndarray",
    probabilities: "np.ndarray",
    delta_max: float,
    correlation_range: int,
    n_cut: int,
    max_iterations: int = 40,
    objective_rtol: float = 1e-12,
) -> "tuple[float, float, float, float, float, float]":
    if isinstance(max_iterations, bool) or not isinstance(
        max_iterations, (int, np.integer)
    ):
        raise ValueError("max_iterations must be an integer")
    max_iterations = int(max_iterations)
    if max_iterations < 1:
        raise ValueError("max_iterations must be at least one")
    if isinstance(objective_rtol, bool) or not isinstance(
        objective_rtol, (int, float, np.integer, np.floating)
    ):
        raise ValueError("objective_rtol must be a real scalar")
    objective_rtol = float(objective_rtol)
    if not np.isfinite(objective_rtol) or objective_rtol <= 0.0:
        raise ValueError("objective_rtol must be strictly positive")

    settings, weights = _check_intensity_family(
        intensities, probabilities
    )
    n_set = settings.size
    if isinstance(n_cut, bool) or not isinstance(n_cut, (int, np.integer)):
        raise ValueError("n_cut must be an integer")
    n_cut = int(n_cut)
    if n_cut < 1:
        raise ValueError("n_cut must be at least one")

    references = []
    for supplied in (yield_reference, error_reference):
        start = np.asarray(supplied, dtype=float)
        if start.shape != (n_cut + 1,):
            raise ValueError("reference shape must be (n_cut + 1,)")
        if (
            not np.all(np.isfinite(start))
            or np.any(start < 0.0)
            or np.any(start > 1.0)
        ):
            raise ValueError("reference arrays must lie in [0, 1]")
        references.append(np.tile(start, (n_set, 1)))
    yield_start, error_start = references

    poisson = [
        _oracle_compute_photon_number_bounds(
            float(setting), delta_max, n_cut
        )
        for setting in settings
    ]
    pairs = [
        (
            first,
            second,
            _oracle_compute_cs_overlap_parameters(
                float(settings[first]), float(settings[second]),
                settings, weights, delta_max, correlation_range, n_cut,
            ),
        )
        for first in range(n_set)
        for second in range(n_set)
        if first != second
    ]

    def _violation(rates, table):
        worst = 0.0
        for index, (lower, upper, tail) in enumerate(poisson):
            worst = max(
                worst,
                float(lower @ table[index] - rates[index]),
                float(rates[index] - tail - upper @ table[index]),
            )
        for first, second, overlaps in pairs:
            lower, upper = _oracle_evaluate_cs_boundaries(
                table[first], overlaps
            )
            worst = max(
                worst,
                float(np.max(lower - table[second])),
                float(np.max(table[second] - upper)),
            )
        return worst

    def _polish_exact_candidate(rates, table, value, maximise):
        # Y = sin(theta)^2 makes each exact CS pair equivalent to
        # |theta_i - theta_j| <= acos(sqrt(tau)).
        width = n_cut + 1
        cost = np.zeros(n_set * width)
        cost[1] = -1.0 if maximise else 1.0
        matrix = np.zeros((2 * n_set + 1, n_set * width))
        rhs = np.empty(2 * n_set + 1)
        for i, (lower, upper, tail) in enumerate(poisson):
            matrix[2 * i, i * width:(i + 1) * width] = lower
            matrix[2 * i + 1, i * width:(i + 1) * width] = -upper
            rhs[2 * i] = rates[i]
            rhs[2 * i + 1] = tail - rates[i]
        # Search near the accumulated outer bound, not an arbitrary
        # feasible point with a different objective.
        matrix[-1] = cost
        rhs[-1] = (-value if maximise else value) + 5e-9
        rows, caps = [], []
        for i, j, tau in pairs:
            if i >= j:
                continue
            angles = np.arccos(np.sqrt(tau))
            for n in range(width):
                row = np.zeros(n_set * width)
                row[i * width + n] = 1.0
                row[j * width + n] = -1.0
                rows.extend((row, -row))
                caps.extend((angles[n], angles[n]))
        linear = LinearConstraint(
            np.asarray(rows), -np.inf, np.asarray(caps)
        )

        def _rate_constraints(theta):
            return rhs - matrix @ np.sin(theta) ** 2

        def _rate_jacobian(theta):
            return -matrix * np.sin(2 * theta)[None, :]

        point = np.arcsin(np.sqrt(np.clip(table.ravel(), 0.0, 1.0)))
        fit = minimize(
            lambda theta: float(cost @ theta),
            point,
            jac=lambda theta: cost,
            bounds=[(0.0, np.pi / 2)] * len(point),
            method="SLSQP",
            constraints=[
                linear,
                {
                    "type": "ineq",
                    "fun": _rate_constraints,
                    "jac": _rate_jacobian,
                },
            ],
            options={"maxiter": 200, "ftol": 1e-13},
        )
        # The original inequalities and the final objective gap below
        # decide acceptance; optimizer status alone is not a certificate.
        return np.sin(fit.x).reshape(n_set, width) ** 2

    def _certify(rates, start, maximise):
        points = [np.clip(start, 0.0, 1.0)]
        value, table = _oracle_solve_outer_linearised_program(
            rates, np.asarray(points), settings, weights, delta_max,
            correlation_range, n_cut, maximise,
        )
        if max_iterations == 1:
            return float(value), abs(float(value) - float(start[0, 1]))
        for _ in range(1, max_iterations):
            points.append(np.clip(table, 0.0, 1.0))
            nxt, nxt_table = _oracle_solve_outer_linearised_program(
                rates, np.asarray(points), settings, weights, delta_max,
                correlation_range, n_cut, maximise,
            )
            relative_scale = max(
                abs(nxt), abs(value), np.finfo(float).tiny
            )
            settled = abs(nxt - value) <= objective_rtol * relative_scale
            value, table = nxt, np.clip(nxt_table, 0.0, 1.0)
            if not settled:
                continue
            if delta_max > 0.0:
                table = _polish_exact_candidate(
                    rates, table, value, maximise
                )
            if not np.all(np.isfinite(table)):
                continue
            if _violation(rates, table) > 1e-9:
                continue
            certified, _ = _oracle_solve_outer_linearised_program(
                rates, table[None, :, :], settings, weights, delta_max,
                correlation_range, n_cut, maximise,
            )
            residual = abs(float(table[0, 1]) - float(certified))
            if residual <= 1e-8:
                return float(certified), residual
        raise ValueError(
            "exact-program convergence and certification checks "
            "were not satisfied"
        )

    key_yield_bound, key_yield_residual = _certify(
        key_basis_rates, yield_start, False
    )
    if np.array_equal(key_basis_rates, check_basis_rates):
        check_yield_bound = key_yield_bound
        check_yield_residual = key_yield_residual
    else:
        check_yield_bound, check_yield_residual = _certify(
            check_basis_rates, yield_start, False
        )
    error_bound, error_residual = _certify(
        check_basis_error_rates, error_start, True
    )
    return (
        key_yield_bound, check_yield_bound, error_bound,
        key_yield_residual, check_yield_residual, error_residual,
    )

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Retain cases with isolated inputs and numerical checks."""
    from textwrap import dedent

    common = dedent(
        """\
        from copy import deepcopy

        import numpy as np


        def _isolated(function, *arguments, **keywords):
            arguments, keywords = deepcopy((arguments, keywords))
            return function(*arguments, **keywords)


        intensities = np.array([0.48, 0.1, 0.0001])
        probabilities = np.array([0.7, 0.18, 0.12])


        def model_inputs(eta, pd=7.2e-08, dA=0.08, n_cut=10):
            no_dark = (1.0 - pd) ** 2
            detected = 1.0 - no_dark * np.exp(-eta * intensities)
            imbalance = (
                0.5 * (np.exp(-eta * intensities * np.cos(dA) ** 2) -
                np.exp(-eta * intensities * np.sin(dA) ** 2))
            )
            errors = (
                0.5 * pd ** 2 + pd * (1.0 - pd) * (1.0 + imbalance) +
                no_dark * (0.5 + imbalance - 0.5 * np.exp(-eta
                * intensities))
            )
            n = np.arange(n_cut + 1, dtype=float)
            silent = (1.0 - eta) ** n
            aligned = (
                (eta * np.cos(dA) ** 2 + 1.0 - eta) ** n - silent
            )
            crossed = (
                (eta * np.sin(dA) ** 2 + 1.0 - eta) ** n - silent
            )
            both = 1.0 - silent - aligned - crossed
            error_ref = (
                no_dark * (crossed + 0.5 * both) + pd * (1.0 - pd) * (
                0.5 * (crossed + both) + silent + crossed + 0.5 * (
                aligned + both)) + 0.5 * pd ** 2
            )
            return (
                (detected, detected, errors, 1.0 - no_dark * silent,
                error_ref)
            )
        """
    )
    return [
        {
            "setup": common + dedent(
                """\

                eta = 10 ** (-0.2 * 35.0 / 10.0) * 0.65
                kz, kx, ke, yr, er = model_inputs(eta)
                args = (
                    (kz, kx, ke, yr, er, intensities, probabilities, 0.001, 3,
                    10)
                )
                """
            ),
            "call": (
                '_isolated(compute_certified_parameter_bounds, *args)[:3]'
            ),
            "gold_call": (
                '_isolated(_oracle_compute_certified_parameter_bounds, *args'
                ')[:3]'
            ),
            "tol": 1e-08,
        },
        {
            "setup": common + dedent(
                """\

                eta = 10 ** (-0.2 * 10.0 / 10.0) * 0.65
                kz, kx, ke, yr, er = model_inputs(eta)
                args = (
                    (kz, kx, ke, yr, er, intensities, probabilities, 0.001, 3,
                    10)
                )
                """
            ),
            "call": (
                '_isolated(compute_certified_parameter_bounds, *args)[:3]'
            ),
            "gold_call": (
                '_isolated(_oracle_compute_certified_parameter_bounds, *args'
                ')[:3]'
            ),
            "tol": 1e-08,
        },
        {
            "setup": common + dedent(
                """\

                eta = 10 ** (-0.2 * 35.0 / 10.0) * 0.65
                kz, kx, ke, yr, er = model_inputs(eta)


                def residual_scale(fn):
                    out = _isolated(
                        fn, kz, kx, ke, yr, er, intensities, probabilities,
                        0.001, 3, 10,
                    )
                    return np.array([
                        float(np.all(np.asarray(out[3:]) >= 0.0)),
                        float(np.all(np.asarray(out[3:]) <= 1e-8)),
                        float(out[0] > 0.0),
                        float(out[2] > 0.0),
                    ])
                """
            ),
            "call": (
                'residual_scale(compute_certified_parameter_bounds)'
            ),
            "gold_call": (
                'residual_scale(_oracle_compute_certified_parameter_bounds)'
            ),
        },
        {
            "setup": common + dedent(
                """\

                eta = 10 ** (-0.2 * 35.0 / 10.0) * 0.65
                kz, kx, ke, yr, er = model_inputs(eta)
                args = (
                    (kz, kx, ke, yr, er, intensities, probabilities, 0.001, 3,
                    10, 1)
                )
                """
            ),
            "call": (
                '_isolated(compute_certified_parameter_bounds, *args)[:3]'
            ),
            "gold_call": (
                '_isolated(_oracle_compute_certified_parameter_bounds, *args'
                ')[:3]'
            ),
            "tol": 1e-08,
        },
        {
            "setup": common + dedent(
                """\

                eta = 10 ** (-0.2 * 35.0 / 10.0) * 0.65
                kz, kx, ke, yr, er = model_inputs(eta)
                args = (
                    (kz, kx, ke, yr, er, intensities, probabilities, 0.0, 3,
                    10)
                )
                """
            ),
            "call": (
                '_isolated(compute_certified_parameter_bounds, *args)[:3]'
            ),
            "gold_call": (
                '_isolated(_oracle_compute_certified_parameter_bounds, *args'
                ')[:3]'
            ),
            "tol": 1e-08,
        },
        {
            "setup": common + dedent(
                """\

                eta = 10 ** (-0.2 * 35.0 / 10.0) * 0.65
                kz, kx, ke, yr, er = model_inputs(eta)
                args = (
                    (kz, kx, ke, yr, er, intensities, probabilities, 0.03, 4,
                    10)
                )
                """
            ),
            "call": (
                '_isolated(compute_certified_parameter_bounds, *args)[:3]'
            ),
            "gold_call": (
                '_isolated(_oracle_compute_certified_parameter_bounds, *args'
                ')[:3]'
            ),
            "tol": 1e-09,
        },
        {
            "setup": common + dedent(
                """\

                eta = 10 ** (-0.2 * 35.0 / 10.0) * 0.65
                kz, kx, ke, yr, er = model_inputs(eta)
                args = (
                    (kz, 0.92 * kx, ke, yr, er, intensities, probabilities,
                    0.001, 3, 10)
                )
                """
            ),
            "call": (
                '_isolated(compute_certified_parameter_bounds, *args)[:3]'
            ),
            "gold_call": (
                '_isolated(_oracle_compute_certified_parameter_bounds, *args'
                ')[:3]'
            ),
            "tol": 1e-08,
        },
        {
            "setup": common + dedent(
                """\

                eta = 10 ** (-0.2 * 35.0 / 10.0) * 0.65
                kz, kx, ke, yr, er = model_inputs(eta)
                args = (
                    (kz, kx, ke, yr, er, intensities, probabilities, 0.001, 3,
                    10, 40, 0.01)
                )
                """
            ),
            "call": (
                '_isolated(compute_certified_parameter_bounds, *args)[:3]'
            ),
            "gold_call": (
                '_isolated(_oracle_compute_certified_parameter_bounds, *args'
                ')[:3]'
            ),
            "tol": 1e-08,
        },
        {
            "setup": dedent(
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
                errs = 0.004 * rates
                n = np.arange(7, dtype=float)
                yr = 1.0 - (1.0 - eta) ** n
                er = np.full(7, 0.006)
                er[0] = 1e-07
                args = (
                    (rates, rates, errs, yr, er, intensities, probabilities,
                    0.001, 2, 6)
                )
                """
            ),
            "call": (
                '_isolated(compute_certified_parameter_bounds, *args)[:3]'
            ),
            "gold_call": (
                '_isolated(_oracle_compute_certified_parameter_bounds, *args'
                ')[:3]'
            ),
            "tol": 1e-08,
        },
        {
            "setup": common + dedent(
                """\

                def run_model(fn, *args, **kwargs):
                    try:
                        _isolated(fn, *args, **kwargs)
                        return 0.0
                    except ValueError:
                        return 1.0


                eta = 0.1296920505
                kz, kx, ke, yr, er = model_inputs(eta)
                args = (
                    (kz, kx, ke, yr, er, intensities, probabilities, 0.001, 3,
                    10, 0)
                )
                """
            ),
            "call": (
                'run_model(compute_certified_parameter_bounds, *args)'
            ),
            "gold_call": (
                'run_model(_oracle_compute_certified_parameter_bounds, *args'
                ')'
            ),
        },
        {
            "setup": common + dedent(
                """\

                def run_model(fn, *args, **kwargs):
                    try:
                        _isolated(fn, *args, **kwargs)
                        return 0.0
                    except ValueError:
                        return 1.0


                eta = 0.1296920505
                kz, kx, ke, yr, er = model_inputs(eta)
                args = (
                    (kz, kx, ke, yr, er, intensities, probabilities, 0.001, 3,
                    10, 40, 0.0)
                )
                """
            ),
            "call": (
                'run_model(compute_certified_parameter_bounds, *args)'
            ),
            "gold_call": (
                'run_model(_oracle_compute_certified_parameter_bounds, *args'
                ')'
            ),
        },
        {
            "setup": common + dedent(
                """\

                def run_model(fn, *args, **kwargs):
                    try:
                        _isolated(fn, *args, **kwargs)
                        return 0.0
                    except ValueError:
                        return 1.0


                eta = 0.1296920505
                kz, kx, ke, yr, er = model_inputs(eta)
                args = (
                    (kz, kx, ke, yr[:5], er, intensities, probabilities,
                    0.001, 3, 10)
                )
                """
            ),
            "call": (
                'run_model(compute_certified_parameter_bounds, *args)'
            ),
            "gold_call": (
                'run_model(_oracle_compute_certified_parameter_bounds, *args'
                ')'
            ),
        },
    ]
