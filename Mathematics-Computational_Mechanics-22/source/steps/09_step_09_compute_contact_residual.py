"""
Evaluate the certified condensed-contact pipeline through adaptive FGMRES.

Apply every preceding numerical contract in order: factor-preserving pressure

condensation, separated contact assembly, compliance-weighted response and

minimal budget-admissible selection, theorem-level spectral certification,

construction of each compact retained-coordinate state, block application of

every reduced inverse and of its adjoint, a field-of-values certificate for the

first right-preconditioned step, and finally a flexible acceptance cycle.  The

pressure contribution may itself be retained only approximately, as in the

source's nested extension: its compact Woodbury state then becomes the fixed

lower level beneath a varying contact state, while Appendix C certifies the

unchanged exact condensed operator. A scalar contact budget recovers the

original one-step path. A vector changes the retained contact space from one

iteration to the next, so the right-inverse actions must remain in schedule

order and the accepted space is genuinely flexible. Optional restart lengths

partition that schedule, and an optional nonzero guess changes every cycle's

fresh residual. The nonsymmetric correction is excluded from every selection

and reduction but retained in the complete operator. The returned quantity is

therefore the freshly recomputed normalized residual of the complete operator.

Returns
-------
one finite nonnegative float equal to the separately recomputed complete- operator residual at the end of the scheduled FGMRES cycle
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def compute_contact_residual(
    material_stiffness: np.ndarray,
    pressure_coupling: np.ndarray,
    pressure_block: np.ndarray,
    interaction_rows: np.ndarray,
    correction: np.ndarray,
    right_hand_side: np.ndarray,
    condition_budget: float | np.ndarray,
    pressure_condition_budget: float | None = None,
    cycle_lengths: np.ndarray | None = None,
    initial_guess: np.ndarray | None = None,
) -> float:
    r"""Chain steps 1--8 and return a nested restarted-FGMRES full residual.

    An implementation must call condense_volumetric_core,
    assemble_contact_operators, build_response_gramian,
    certify_preconditioned_spectrum, build_reduced_interaction_state,
    apply_reduced_inverse, certify_residual_bounds, and
    compute_directional_residual. Interpret a scalar condition_budget as a
    one-entry schedule and a one-dimensional array as an iteration-indexed
    schedule. For each entry independently, determine the smallest retained
    response rank whose Appendix-C robust condition ceiling is no larger than
    that entry, where a ceiling counts as no larger when it does not exceed the
    entry by more than 1e-12 times max(1, |entry|); rank zero is admissible and
    all numerically positive response modes must be retained if no smaller rank
    meets it. If
    pressure_condition_budget is supplied, first select the smallest pressure
    response rank whose exact posterior ceiling over material_stiffness meets
    that scalar budget under the same comparison tolerance, build its fixed
    compact state, and use the resulting
    pressure-surrogate core for every contact selection while retaining the
    exact condensed core as the Appendix-C reference. If the pressure budget is
    omitted, use the exact condensed core directly. For every selected contact
    projector recover an orthonormal basis of its dominant response space,
    certify that factor representation, build its compact state, and apply
    that state to the identity once forward and once as the adjoint. Confirm
    the adjoint action and preserve the compact states in schedule order.
    Certify every scheduled action with certify_residual_bounds. The first
    accepted history entry must also equal its closed-form certified reduction.
    Pass the ordered compact states, not pre-expanded dense inverses, to a
    flexible cycle whose requested length is the schedule length, together
    with the optional lower state, restart partition, and initial guess, and
    return its terminal history entry.

    Raises ValueError unless condition_budget is either a finite real scalar,
    not a bool, or a nonempty finite one-dimensional real array, not Boolean,
    with every entry at least one and length no larger than the displacement
    dimension; pressure_condition_budget is None or a finite real scalar, not
    a bool, at least one; under any preceding contract; or unless the
    scaled pressure rows reproduce the volumetric Gram matrix, the returned
    symmetric part matches the complete operator, the retained projector commutes with
    the response, the cutoff ratio lies in the unit interval, the spectral
    prediction agrees, every recovered basis spans the selected projector,
    every approximate-core certificate satisfies its five defining bounds,
    each compact state satisfies its two defining identities, each reduced
    preconditioner equals its own reassembly, every adjoint action is the
    transpose of its forward action, the first certified exact reduction matches the
    first accepted residual, and the terminal residual agrees with a separate
    fresh evaluation to rtol 1e-11 and atol 1e-13.

    Parameters
    ----------
    material_stiffness, pressure_coupling, pressure_block : np.ndarray
        Raw mixed finite-element core data.
    interaction_rows, correction : np.ndarray
        Ordered interaction factor and complete-operator correction.
    right_hand_side : np.ndarray
        Nonzero displacement-space right-hand side.
    condition_budget : float or np.ndarray
        One robust contact condition ceiling or a per-iteration schedule.
    pressure_condition_budget : float or None
        Optional exact pressure-level posterior condition ceiling.
    cycle_lengths : np.ndarray or None
        Optional positive restart partition summing to the schedule length.
    initial_guess : np.ndarray or None
        Optional finite displacement-space initial iterate.

    Returns
    -------
    float
        Finite dimensionless terminal normalized full residual.
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_compute_contact_residual(
    material_stiffness: np.ndarray,
    pressure_coupling: np.ndarray,
    pressure_block: np.ndarray,
    interaction_rows: np.ndarray,
    correction: np.ndarray,
    right_hand_side: np.ndarray,
    condition_budget: float | np.ndarray,
    pressure_condition_budget: float | None = None,
    cycle_lengths: np.ndarray | None = None,
    initial_guess: np.ndarray | None = None,
) -> float:
    """Chain all preceding oracle contracts and return the certified residual."""
    raw_budget = np.asarray(condition_budget)
    if raw_budget.dtype.kind not in {"i", "u", "f"} or raw_budget.ndim > 1:
        raise ValueError("condition_budget must be a scalar or real vector")
    try:
        if raw_budget.ndim == 0:
            budgets = np.array([float(raw_budget)], dtype=float)
        else:
            budgets = np.asarray(condition_budget, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("condition_budget must be real") from exc
    if budgets.size == 0 or not np.all(np.isfinite(budgets)):
        raise ValueError("condition_budget must be nonempty and finite")
    if np.any(budgets < 1.0):
        raise ValueError("every condition budget must be at least one")

    if pressure_condition_budget is None:
        pressure_budget = None
    else:
        if isinstance(pressure_condition_budget, (bool, np.bool_)):
            raise ValueError("pressure_condition_budget must be a real scalar")
        raw_pressure_budget = np.asarray(pressure_condition_budget)
        if raw_pressure_budget.ndim != 0 or raw_pressure_budget.dtype.kind not in {
            "i",
            "u",
            "f",
        }:
            raise ValueError("pressure_condition_budget must be a real scalar")
        pressure_budget = float(raw_pressure_budget)
        if not np.isfinite(pressure_budget) or pressure_budget < 1.0:
            raise ValueError(
                "pressure_condition_budget must be finite and at least one"
            )

    scaled_rows, volumetric, core = _oracle_condense_volumetric_core(
        material_stiffness,
        pressure_coupling,
        pressure_block,
    )
    _, _, full_operator, symmetric_part = _oracle_assemble_contact_operators(
        core,
        interaction_rows,
        correction,
    )
    n_dof = core.shape[0]
    if budgets.size > n_dof:
        raise ValueError("the budget schedule cannot exceed the displacement dimension")
    rows = np.asarray(interaction_rows, dtype=float)
    identity = np.eye(n_dof)

    def response_rank(levels: np.ndarray, n_rows: int) -> int:
        """Return the numerical response rank used by the selection contracts."""
        scale = max(1.0, float(levels[0])) if levels.size else 1.0
        tolerance = max(n_rows, n_dof, 1) * np.finfo(float).eps * scale
        return int(np.count_nonzero(levels > tolerance))

    def dominant_basis(response: np.ndarray, rank: int) -> np.ndarray:
        """Recover the dominant response basis without trusting level order."""
        eigenvalues, eigenvectors = np.linalg.eigh(response)
        order = np.argsort(eigenvalues)[::-1]
        return eigenvectors[:, order[:rank]]

    lower_rows = None
    lower_responses = None
    lower_reduced = None
    if pressure_budget is None:
        selection_core = core
    else:
        material = np.asarray(material_stiffness, dtype=float)
        pressure_response, pressure_levels, _, _ = _oracle_build_response_gramian(
            material,
            scaled_rows,
            0,
        )
        pressure_rank = response_rank(pressure_levels, scaled_rows.shape[0])
        pressure_selected = None
        pressure_tolerance = 1e-12 * max(1.0, abs(pressure_budget))
        for candidate_rank in range(pressure_rank + 1):
            candidate = _oracle_build_response_gramian(
                material,
                scaled_rows,
                candidate_rank,
            )
            candidate_basis = dominant_basis(
                candidate[0],
                candidate_rank,
            )
            candidate_certificate = _oracle_certify_preconditioned_spectrum(
                material,
                scaled_rows,
                candidate[1],
                None,
                candidate_basis,
            )
            if candidate_certificate[5] <= pressure_budget + pressure_tolerance:
                pressure_selected = (*candidate, candidate_basis, candidate_certificate)
                break
        if pressure_selected is None:
            raise ValueError("no pressure response rank meets its condition budget")
        (
            pressure_response,
            pressure_levels,
            pressure_projector,
            pressure_cutoff,
            pressure_basis,
            pressure_certificate,
        ) = pressure_selected
        (
            selection_core,
            lower_rows,
            lower_responses,
            lower_reduced,
        ) = _oracle_build_reduced_interaction_state(
            material,
            scaled_rows,
            pressure_basis,
        )
        lower_inverse = _oracle_apply_reduced_inverse(
            material,
            lower_rows,
            lower_responses,
            lower_reduced,
            identity,
            False,
        )
        lower_adjoint = _oracle_apply_reduced_inverse(
            material,
            lower_rows,
            lower_responses,
            lower_reduced,
            identity,
            True,
        )
        if not np.allclose(
            pressure_response @ pressure_projector,
            pressure_projector @ pressure_response,
            rtol=1e-9,
            atol=1e-11,
        ):
            raise ValueError("the pressure projector is not response-invariant")
        if not np.allclose(
            pressure_basis @ pressure_basis.T,
            pressure_projector,
            rtol=1e-9,
            atol=1e-11,
        ):
            raise ValueError("the pressure basis does not span its projector")
        if not 0.0 <= pressure_cutoff <= 1.0 + 1e-12:
            raise ValueError("the pressure cutoff ratio must lie in [0, 1]")
        if pressure_certificate[5] > pressure_budget + pressure_tolerance:
            raise ValueError("the pressure projector violates its condition budget")
        if not np.allclose(
            lower_adjoint,
            lower_inverse.T,
            rtol=1e-10,
            atol=1e-12,
        ):
            raise ValueError("the lower adjoint action is inconsistent")
        if not np.allclose(
            lower_inverse @ selection_core,
            identity,
            rtol=1e-9,
            atol=1e-11,
        ):
            raise ValueError("the lower compact action does not invert its core")
        expected_selection_core = material + scaled_rows.T @ (
            pressure_projector @ scaled_rows
        )
        if not np.allclose(
            selection_core,
            0.5 * (expected_selection_core + expected_selection_core.T),
            rtol=1e-11,
            atol=1e-13,
        ):
            raise ValueError("the pressure-surrogate core is inconsistent")

    _contact_response, contact_levels, _, _ = _oracle_build_response_gramian(
        selection_core,
        interaction_rows,
        0,
    )
    numerical_rank = response_rank(contact_levels, rows.shape[0])
    retained_row_states = []
    core_response_states = []
    reduced_matrix_states = []
    first_reduction = None
    first_estimate = None

    rhs = np.asarray(right_hand_side, dtype=float)
    if initial_guess is None:
        certification_rhs = rhs
    else:
        guess = np.asarray(initial_guess, dtype=float)
        if guess.shape != (n_dof,) or not np.all(np.isfinite(guess)):
            raise ValueError("initial_guess must be a finite vector of shape (n,)")
        certification_rhs = rhs - full_operator @ guess
        certification_scale = max(
            1.0,
            float(np.linalg.norm(rhs)),
            float(np.linalg.norm(full_operator, ord=2) * np.linalg.norm(guess)),
        )
        if float(np.linalg.norm(certification_rhs)) <= (
            100.0 * np.finfo(float).eps * certification_scale
        ):
            certification_rhs = rhs

    for iteration, budget in enumerate(budgets):
        budget_tolerance = 1e-12 * max(1.0, abs(float(budget)))
        selected = None
        for candidate_rank in range(numerical_rank + 1):
            candidate = _oracle_build_response_gramian(
                selection_core,
                interaction_rows,
                candidate_rank,
            )
            (
                _candidate_response,
                candidate_levels,
                candidate_projector,
                _candidate_ratio,
            ) = candidate
            candidate_basis = dominant_basis(candidate[0], candidate_rank)
            candidate_certificate = _oracle_certify_preconditioned_spectrum(
                selection_core,
                interaction_rows,
                candidate_levels,
                None,
                candidate_basis,
                core,
            )
            if candidate_certificate[6][3] <= budget + budget_tolerance:
                selected = (*candidate, candidate_basis, *candidate_certificate)
                break
        if selected is None:
            raise ValueError("no admissible response rank meets a condition budget")
        (
            response,
            levels,
            projector,
            cutoff_ratio,
            retained_basis,
            measured,
            predicted,
            certificate,
            condition_number,
            omitted_interaction,
            posterior_bound,
            robust_certificate,
        ) = selected
        (
            preconditioner,
            retained_rows,
            core_responses,
            reduced_matrix,
        ) = _oracle_build_reduced_interaction_state(
            selection_core,
            interaction_rows,
            retained_basis,
        )
        right_inverse = _oracle_apply_reduced_inverse(
            selection_core,
            retained_rows,
            core_responses,
            reduced_matrix,
            identity,
            False,
        )
        adjoint_inverse = _oracle_apply_reduced_inverse(
            selection_core,
            retained_rows,
            core_responses,
            reduced_matrix,
            identity,
            True,
        )
        reduction, coefficient, estimate, _, _ = _oracle_certify_residual_bounds(
            full_operator,
            right_inverse,
            certification_rhs,
        )

        if not np.allclose(
            response @ projector, projector @ response, rtol=1e-9, atol=1e-11
        ):
            raise ValueError("the retained projector is not response-invariant")
        if not np.allclose(
            retained_basis @ retained_basis.T,
            projector,
            rtol=1e-9,
            atol=1e-11,
        ):
            raise ValueError("the retained basis does not span the selected projector")
        if not 0.0 <= cutoff_ratio <= 1.0 + 1e-12:
            raise ValueError("the cutoff ratio must lie in [0, 1]")
        if not np.allclose(measured, predicted, rtol=1e-8, atol=1e-10):
            raise ValueError("the spectral certificate is inconsistent")
        if not np.isfinite(certificate) or certificate < 0.0 or certificate >= 1.0:
            raise ValueError("the omitted-response certificate is invalid")
        if not np.isfinite(condition_number) or condition_number < 1.0 - 1e-12:
            raise ValueError("the preconditioned condition number is invalid")
        if not np.isfinite(omitted_interaction) or omitted_interaction < 0.0:
            raise ValueError("the worst omitted interaction is invalid")
        if not np.isclose(
            posterior_bound, 1.0 + omitted_interaction, rtol=1e-12, atol=1e-14
        ):
            raise ValueError(
                "the posterior bound does not follow the omitted interaction"
            )
        if condition_number > posterior_bound + 1e-8:
            raise ValueError("the measured conditioning exceeds the posterior bound")
        if robust_certificate.shape != (5,) or not np.all(
            np.isfinite(robust_certificate)
        ):
            raise ValueError("the approximate-core certificate is invalid")
        c1, c2, delta, robust_bound, robust_condition = robust_certificate
        if c1 <= 0.0 or c2 < c1 or delta < 0.0:
            raise ValueError("the approximate-core constants are invalid")
        expected_robust_bound = max(c2 + delta, 1.0) / min(c1, 1.0)
        if not np.isclose(
            robust_bound,
            expected_robust_bound,
            rtol=1e-12,
            atol=1e-14,
        ):
            raise ValueError("the Appendix-C ceiling is inconsistent")
        if robust_condition < 1.0 - 1e-12 or robust_condition > robust_bound + 1e-8:
            raise ValueError("the robust measured conditioning is invalid")
        if robust_bound > budget + budget_tolerance:
            raise ValueError("the selected projector violates its robust budget")
        expected_preconditioner = selection_core + rows.T @ (projector @ rows)
        if not np.allclose(
            preconditioner,
            0.5 * (expected_preconditioner + expected_preconditioner.T),
            rtol=1e-11,
            atol=1e-13,
        ):
            raise ValueError("the reduced preconditioner is inconsistent")
        if not np.allclose(
            selection_core @ core_responses,
            retained_rows.T,
            rtol=1e-10,
            atol=1e-12,
        ):
            raise ValueError("the compact core responses are inconsistent")
        if not np.allclose(
            reduced_matrix,
            np.eye(retained_rows.shape[0]) + retained_rows @ core_responses,
            rtol=1e-10,
            atol=1e-12,
        ):
            raise ValueError("the compact reduced matrix is inconsistent")
        if not np.allclose(adjoint_inverse, right_inverse.T, rtol=1e-10, atol=1e-12):
            raise ValueError(
                "the adjoint action is not the transpose of the forward one"
            )
        if not np.allclose(
            right_inverse @ preconditioner, identity, rtol=1e-9, atol=1e-11
        ):
            raise ValueError("the reduced inverse does not invert the preconditioner")
        if not all(np.isfinite(value) for value in (reduction, coefficient, estimate)):
            raise ValueError("a residual certificate is invalid")
        if reduction > estimate + 1e-9:
            raise ValueError("a field-of-values certificate is invalid")

        retained_row_states.append(retained_rows)
        core_response_states.append(core_responses)
        reduced_matrix_states.append(reduced_matrix)
        if iteration == 0:
            first_reduction = float(reduction)
            first_estimate = float(estimate)

    iterate, history, coefficients, used = _oracle_compute_directional_residual(
        full_operator,
        right_hand_side,
        np.asarray(material_stiffness, dtype=float)
        if pressure_budget is not None
        else selection_core,
        tuple(retained_row_states),
        tuple(core_response_states),
        tuple(reduced_matrix_states),
        int(budgets.size),
        cycle_lengths,
        initial_guess,
        lower_rows,
        lower_responses,
        lower_reduced,
    )

    if not np.allclose(scaled_rows.T @ scaled_rows, volumetric, rtol=1e-11, atol=1e-13):
        raise ValueError("the scaled pressure rows do not reproduce the Gram matrix")
    if not np.allclose(
        symmetric_part,
        0.5 * (full_operator + full_operator.T),
        rtol=1e-12,
        atol=1e-14,
    ):
        raise ValueError("the symmetric part is inconsistent with the full operator")
    if used < 0 or used > budgets.size:
        raise ValueError("the flexible iteration count is invalid")
    if coefficients.shape != budgets.shape or not np.all(np.isfinite(coefficients)):
        raise ValueError("the flexible coefficients are invalid")
    fresh = float(np.linalg.norm(rhs - full_operator @ iterate) / np.linalg.norm(rhs))
    rho = float(history[-1])
    if not np.allclose(rho, fresh, rtol=1e-11, atol=1e-13):
        raise ValueError("the residual history is not a fresh full residual")
    if used:
        expected_first_reduction = history[0] * first_reduction
        expected_first_estimate = history[0] * first_estimate
        if not np.allclose(
            history[1],
            expected_first_reduction,
            rtol=1e-10,
            atol=1e-12,
        ):
            raise ValueError("the first certified reduction disagrees with the cycle")
        if history[1] > expected_first_estimate + 1e-9:
            raise ValueError("the first field-of-values certificate is invalid")
    return rho

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return scalar and scheduled integrations plus invalid budgets."""
    return [
        {
            "setup": """import numpy as np
K0 = np.array([[40,-1,.2,0,0],[-1,12,-.3,.1,0],[.2,-.3,2.5,-.2,.1],[0,.1,-.2,1.2,-.1],[0,0,.1,-.1,.8]], dtype=float)
B = np.array([[1,-1,0,0,.5],[0,.5,-1,1,0]], dtype=float)
D = np.diag([2.0, 1.5])
U = np.array([[8,0,0,0,0],[0,4,0,0,0],[0,0,2,.2,0],[0,0,.1,1.5,1.2]], dtype=float)
N = np.array([[.08,.02,0,0,0],[-.01,-.04,.03,0,0],[0,-.02,.05,.01,0],[0,0,-.03,-.02,.04],[.01,0,0,-.02,.03]])
b = np.array([1.0, -.5, .75, .2, -1.1])
kappa = 2.5
""",
            "call": "compute_contact_residual(K0, B, D, U, N, b, kappa)",
            "gold_call": "_oracle_compute_contact_residual(K0, B, D, U, N, b, kappa)",
        },
        {
            "setup": """import numpy as np
K0 = np.array([[4.0, -1.0], [-1.0, 3.0]])
B = np.array([[1.0, 0.5]])
D = np.array([[2.0]])
U = np.zeros((0, 2))
N = np.array([[0.1, 0.02], [-0.03, -0.05]])
b = np.array([1.0, -2.0])
kappa = 1.0
""",
            "call": "compute_contact_residual(K0, B, D, U, N, b, kappa)",
            "gold_call": "_oracle_compute_contact_residual(K0, B, D, U, N, b, kappa)",
        },
        {
            "setup": """import numpy as np
K0 = np.array([[6.0, -1.0, .2], [-1.0, 3.0, -.4], [.2, -.4, 1.5]])
B = np.array([[1.0, -.5, 0.0], [0.0, .2, 1.0]])
D = np.diag([2.0, 1.2])
U = np.array([[2.0, 0.0, 0.0], [0.0, 1.0, .5]])
N = np.array([[.03,.01,0.0],[-.02,-.01,.02],[0.0,-.015,.01]])
b = np.array([.5, -1.0, .75])
kappa = 1.0
""",
            "call": "compute_contact_residual(K0, B, D, U, N, b, kappa)",
            "gold_call": "_oracle_compute_contact_residual(K0, B, D, U, N, b, kappa)",
        },
        {
            "setup": """import numpy as np
K0 = np.diag([20.0,4.0,1.2,.7]) + np.array([[0,.2,0,0],[.2,0,-.1,0],[0,-.1,0,.05],[0,0,.05,0]])
B = np.array([[1.0,0.0,-.5,.2]])
D = np.array([[1.7]])
U = np.array([[5.0,0,0,0],[0,2.0,.2,0],[0,0,.5,1.8]])
N = np.array([[.02,0,0,0],[-.01,.01,.02,0],[0,-.02,-.01,.01],[.005,0,-.01,.02]])
b = np.array([1.0,-.4,.3,-.8])
kappa = 3.9525692497970013
""",
            "call": "compute_contact_residual(K0, B, D, U, N, b, kappa)",
            "gold_call": "_oracle_compute_contact_residual(K0, B, D, U, N, b, kappa)",
        },
        {
            "setup": """import numpy as np
K0 = np.array([[7.0,-.5,.2],[-.5,3.5,-.3],[.2,-.3,1.6]])
B = np.array([[1.0,-.4,.3],[0.0,.6,-1.0]])
D = np.array([[2.4, -0.8], [-0.8, 1.7]])
U = np.array([[3.0,0.0,.1],[0.0,1.4,0.0],[.2,0.0,.9]])
N = np.array([[.04,.01,-.01],[-.02,-.03,.02],[.01,-.015,.025]])
b = np.array([.9,-1.2,.4])
kappa = 1.33300863713649
""",
            "call": "compute_contact_residual(K0, B, D, U, N, b, kappa)",
            "gold_call": "_oracle_compute_contact_residual(K0, B, D, U, N, b, kappa)",
        },
        {
            "setup": """import numpy as np
K0 = np.array([[12.0,-.6,.15],[-.6,5.0,-.25],[.15,-.25,2.2]])
B = np.array([[.9,-.2,.4]])
D = np.array([[1.45]])
U = np.array([[4.0,0.0,0.0],[0.0,2.2,.1],[.3,0.0,1.1],[0.0,.5,1.6]])
N = np.array([[.03,-.01,.005],[.02,-.025,.015],[-.01,-.02,.02]])
b = np.array([-.7,1.1,.5])
kappa = 2.5985928395564457
""",
            "call": "compute_contact_residual(K0, B, D, U, N, b, kappa)",
            "gold_call": "_oracle_compute_contact_residual(K0, B, D, U, N, b, kappa)",
        },
        {
            "setup": """import numpy as np
K0 = np.eye(2)
B = np.array([[1.0, 0.0]])
D = np.array([[1.0]])
U = np.zeros((0, 2))
N = np.zeros((2, 2))
b = np.ones(2)
kappa = 0.9
def run_model():
    try:
        compute_contact_residual(K0, B, D, U, N, b, kappa)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_compute_contact_residual(K0, B, D, U, N, b, kappa)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        {
            "setup": """import numpy as np
K0 = np.array([[40,-1,.2,0,0],[-1,12,-.3,.1,0],[.2,-.3,2.5,-.2,.1],[0,.1,-.2,1.2,-.1],[0,0,.1,-.1,.8]], dtype=float)
B = np.array([[1,-1,0,0,.5],[0,.5,-1,1,0]], dtype=float)
D = np.diag([2.0, 1.5])
U = np.array([[8,0,0,0,0],[0,4,0,0,0],[0,0,2,.2,0],[0,0,.1,1.5,1.2]], dtype=float)
N = np.array([[.08,.02,0,0,0],[-.01,-.04,.03,0,0],[0,-.02,.05,.01,0],[0,0,-.03,-.02,.04],[.01,0,0,-.02,.03]])
b = np.array([1.0, -.5, .75, .2, -1.1])
kappa = np.array([4.0, 2.5, 1.1])
""",
            "call": "compute_contact_residual(K0, B, D, U, N, b, kappa)",
            "gold_call": "_oracle_compute_contact_residual(K0, B, D, U, N, b, kappa)",
        },
        {
            "setup": """import numpy as np
K0 = np.diag([20.0,4.0,1.2,.7]) + np.array([[0,.2,0,0],[.2,0,-.1,0],[0,-.1,0,.05],[0,0,.05,0]])
B = np.array([[1.0,0.0,-.5,.2]])
D = np.array([[1.7]])
U = np.array([[5.0,0,0,0],[0,2.0,.2,0],[0,0,.5,1.8]])
N = np.array([[.02,0,0,0],[-.01,.01,.02,0],[0,-.02,-.01,.01],[.005,0,-.01,.02]])
b = np.array([1.0,-.4,.3,-.8])
kappa = np.array([20.0, 3.9525692497970013, 1.0])
""",
            "call": "compute_contact_residual(K0, B, D, U, N, b, kappa)",
            "gold_call": "_oracle_compute_contact_residual(K0, B, D, U, N, b, kappa)",
        },
        {
            "setup": """import numpy as np
K0 = np.array([[4.0, -1.0], [-1.0, 3.0]])
B = np.array([[1.0, 0.5]])
D = np.array([[2.0]])
U = np.zeros((0, 2))
N = np.array([[0.1, 0.02], [-0.03, -0.05]])
b = np.array([1.0, -2.0])
kappa = np.array([1.0, 1.0])
""",
            "call": "compute_contact_residual(K0, B, D, U, N, b, kappa)",
            "gold_call": "_oracle_compute_contact_residual(K0, B, D, U, N, b, kappa)",
        },
        {
            "setup": """import numpy as np
K0 = np.eye(2)
B = np.array([[1.0, 0.0]])
D = np.array([[1.0]])
U = np.zeros((0, 2))
N = np.zeros((2, 2))
b = np.ones(2)
kappa = np.array([1.0, 1.0])
A = K0 + B.T @ np.linalg.solve(D, B) + U.T @ U + N
cycles = np.array([1, 1])
x0 = np.linalg.solve(A, b)
""",
            "call": "compute_contact_residual(K0, B, D, U, N, b, kappa, None, cycles, x0)",
            "gold_call": "_oracle_compute_contact_residual(K0, B, D, U, N, b, kappa, None, cycles, x0)",
        },
        {
            "setup": """import numpy as np
K0 = np.array([[40,-1,.2,0,0],[-1,12,-.3,.1,0],[.2,-.3,2.5,-.2,.1],[0,.1,-.2,1.2,-.1],[0,0,.1,-.1,.8]], dtype=float)
B = np.array([[1,-1,0,0,.5],[0,.5,-1,1,0]], dtype=float)
D = np.diag([2.0, 1.5])
U = np.array([[8,0,0,0,0],[0,4,0,0,0],[0,0,2,.2,0],[0,0,.1,1.5,1.2]], dtype=float)
N = np.array([[.08,.02,0,0,0],[-.01,-.04,.03,0,0],[0,-.02,.05,.01,0],[0,0,-.03,-.02,.04],[.01,0,0,-.02,.03]])
b = np.array([1.0, -.5, .75, .2, -1.1])
kappa = np.array([5.0, 3.0, 2.5, 1.3])
pressure_kappa = 1.5
cycles = np.array([2, 2])
x0 = np.array([.02, -.01, .03, 0.0, -.02])
""",
            "call": "compute_contact_residual(K0, B, D, U, N, b, kappa, pressure_kappa, cycles, x0)",
            "gold_call": "_oracle_compute_contact_residual(K0, B, D, U, N, b, kappa, pressure_kappa, cycles, x0)",
        },
        {
            "setup": """import numpy as np
K0 = np.array([[40,-1,.2,0,0],[-1,12,-.3,.1,0],[.2,-.3,2.5,-.2,.1],[0,.1,-.2,1.2,-.1],[0,0,.1,-.1,.8]], dtype=float)
B = np.array([[1,-1,0,0,.5],[0,.5,-1,1,0]], dtype=float)
D = np.diag([2.0, 1.5])
U = np.array([[8,0,0,0,0],[0,4,0,0,0],[0,0,2,.2,0],[0,0,.1,1.5,1.2]], dtype=float)
N = np.array([[.08,.02,0,0,0],[-.01,-.04,.03,0,0],[0,-.02,.05,.01,0],[0,0,-.03,-.02,.04],[.01,0,0,-.02,.03]])
b = np.array([1.0, -.5, .75, .2, -1.1])
kappa = np.array([6.0, 3.5])
pressure_kappa = 2.0
cycles = np.array([1, 1])
""",
            "call": "compute_contact_residual(K0, B, D, U, N, b, kappa, pressure_kappa, cycles)",
            "gold_call": "_oracle_compute_contact_residual(K0, B, D, U, N, b, kappa, pressure_kappa, cycles)",
        },
        {
            "setup": """import numpy as np
K0 = np.eye(2)
B = np.array([[1.0, 0.0]])
D = np.array([[1.0]])
U = np.zeros((0, 2))
N = np.zeros((2, 2))
b = np.ones(2)
kappa = 1.5
pressure_kappa = 2.0
def run_model():
    try:
        compute_contact_residual(K0, B, D, U, N, b, kappa, pressure_kappa)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_compute_contact_residual(K0, B, D, U, N, b, kappa, pressure_kappa)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        {
            "setup": """import numpy as np
K0 = np.eye(2)
B = np.array([[1.0, 0.0]])
D = np.array([[1.0]])
U = np.zeros((0, 2))
N = np.zeros((2, 2))
b = np.ones(2)
kappa = np.array([1.0, 1.0])
pressure_kappa = True
def run_model():
    try:
        compute_contact_residual(K0, B, D, U, N, b, kappa, pressure_kappa)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_compute_contact_residual(K0, B, D, U, N, b, kappa, pressure_kappa)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        {
            "setup": """import numpy as np
K0 = np.eye(2)
B = np.array([[1.0, 0.0]])
D = np.array([[1.0]])
U = np.zeros((0, 2))
N = np.zeros((2, 2))
b = np.ones(2)
kappa = np.zeros(0)
def run_model():
    try:
        compute_contact_residual(K0, B, D, U, N, b, kappa)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_compute_contact_residual(K0, B, D, U, N, b, kappa)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
    ]
