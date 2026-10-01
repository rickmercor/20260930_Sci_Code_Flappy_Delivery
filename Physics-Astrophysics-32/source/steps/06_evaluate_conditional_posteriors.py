"""
Analytic Cartesian-amplitude posterior for the spherical-22 parent.

Condition the complete 28-coordinate system from step 05 before selecting its parent block:



$$m_e=K_{ed}K_{dd}^{-1}d,\qquad

K_{e\mid d}=K_{ee}-K_{ed}K_{dd}^{-1}K_{de}.$$



For $y=(\Re h_{22},\Im h_{22})-m_e$, active design $X$ of dimension $d$, $K=K_{e\mid d}$, and the task-defined proper prior $a\sim N(0,s_a^2I_d)$, compute



$$F=X^TK^{-1}X+s_a^{-2}I_d,\qquad V=F^{-1},\qquad \bar a=VX^TK^{-1}y,$$



and the state score



$$\log Z_p=-\tfrac12[y^TK^{-1}y-\bar a^TF\bar a+\log|K|+\log|F|+d\log(s_a^2)].$$



Use $s_a=0.75$ by default and retain the prior-normalization term because the two parent models have different dimensions. Omit only the common $14\log(2\pi)$ data constant. Apply positive-definite operators through checked Cholesky factors. For each active complex mode report significance $1-\exp(-d^2/2)$ from its two-real-dimensional marginal mean and covariance. For the default hierarchy, conditioning gives $m_e=\rho\sqrt q\,d$ and $K_{e\mid d}=q(1-\rho^2)D$ before the parent coordinates are selected.

Returns
-------
`numpy.ndarray` of shape `(n,25)` with `(spin,mass,mean[4],covariance[16],log_Z_parent,S220,S320)`. The covariance is C-row-major. Inactive 320 mean, covariance rows and columns, and significance are exact zero. Every component is compared at `1e-9`.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np
def evaluate_conditional_posteriors(
    data: np.ndarray,
    templates: np.ndarray,
    covariances: np.ndarray,
    amplitude_prior_scale: float = 0.75,
) -> np.ndarray:
    """Evaluate the analytic mixed-parent amplitude posterior at every state.

    Parameters
    ----------
    data : numpy.ndarray, shape (7, 9)
        Finite ringdown record.
    templates : numpy.ndarray, shape (n, 18, 4)
        Step-04 parent designs and packed state metadata.
    covariances : numpy.ndarray, shape (n, 56, 56)
        Finite joint numerical-error covariances in step-05 `z=(e,d)` order.
    amplitude_prior_scale : float, default=0.75
        Finite positive standard deviation `s_a` of each active Cartesian
        amplitude coordinate. The same scale is used in both parent models.

    Returns
    -------
    numpy.ndarray, shape (n, 25)
        `(spin,mass,mean[4],covariance[16],log_Z_parent,S220,S320)` with the
        covariance flattened in C row-major order. Components are compared at
        tolerance `1e-9`.

    Raises
    ------
    ValueError
        If an input has an incompatible shape or nonfinite entry; state rows
        disagree; a supplied joint covariance is asymmetric, singular, or not
        positive definite; `amplitude_prior_scale` is nonpositive or nonfinite;
        a derived covariance cannot be factored; the
        active design is rank deficient; or an active posterior precision or
        two-dimensional marginal covariance is not positive definite.
    """
    return None
import numpy as np

# =============================================================================
# GOLD SOLUTION
# =============================================================================

r"""Residual-conditioned analytic posterior for the spherical-22 parent.

Let $e$ be the complete 28-coordinate highest-resolution numerical error and
$d$ the complete observed highest-minus-next-highest residual in step-05
order. Condition the full correlated system before selecting its parent block:

$$m_e=K_{ed}K_{dd}^{-1}d,\qquad
K_{e|d}=K_{ee}-K_{ed}K_{dd}^{-1}K_{de}.$$

For the conditioned real data vector $y-m_e$, design $X$, covariance
$K_{e|d}$, and task-defined proper Cartesian prior
$a\sim N(0,s_a^2I_d)$ on the $d=2$ or $d=4$ active coordinates, the posterior
is Gaussian:

$$F=X^TK^{-1}X+s_a^{-2}I_d,\qquad
V=F^{-1},\qquad \bar a=VX^TK^{-1}y.$$

The state score is the amplitude-marginalized log likelihood with common
state-independent constants omitted,

$$\log Z_p=-\tfrac12[y^TK^{-1}y-\bar a^TF\bar a
+\log|K|+\log|F|+d\log(s_a^2)].$$

The $d\log(s_a^2)$ prior-normalization term must be retained because the
220-only and mixed-parent models have different active dimensions. The default
uses $s_a=0.75$. Only the common $14\log(2\pi)$ data-normalization constant is
omitted.

All positive-definite applications use checked Cholesky factors. The complex
mode significance is $1-\exp(-d^2/2)$ using the two-real-dimensional
marginal mean and covariance.

Returns
-------
`numpy.ndarray` of shape `(n, 25)`: spin, mass, four amplitude-mean
coordinates, the row-major `4 by 4` posterior covariance, conditional parent
log score, and the 220 and 320 significances. In a 220-only fit the 320 mean,
covariance rows and columns, and significance are exact zero. Components are
compared at `1e-9`.
"""
import numpy as np
import numpy as np
_ALL_ERROR_INDICES = np.arange(28)
_ALL_RESOLUTION_INDICES = 28 + _ALL_ERROR_INDICES
_PARENT_COVARIANCE_INDICES = np.r_[0:7, 14:21]
def _posterior_cholesky(matrix: np.ndarray, label: str) -> np.ndarray:
    if not np.all(np.isfinite(matrix)) or not np.allclose(
        matrix, matrix.T, rtol=0.0, atol=1.0e-12
    ):
        raise ValueError(label + " must be finite and symmetric")
    try:
        return np.linalg.cholesky(matrix)
    except np.linalg.LinAlgError as exc:
        raise ValueError(label + " must be positive definite") from exc
def _posterior_factor_solve(factor: np.ndarray, right_hand_side: np.ndarray) -> np.ndarray:
    return np.linalg.solve(factor.T, np.linalg.solve(factor, right_hand_side))
def _condition_parent_error(
    joint_covariance: np.ndarray,
    resolution_observed: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    error_covariance = joint_covariance[:28, :28]
    cross_covariance = joint_covariance[
        np.ix_(_ALL_ERROR_INDICES, _ALL_RESOLUTION_INDICES)
    ]
    resolution_covariance = joint_covariance[
        np.ix_(_ALL_RESOLUTION_INDICES, _ALL_RESOLUTION_INDICES)
    ]
    resolution_factor = _posterior_cholesky(
        resolution_covariance, "parent resolution covariance"
    )
    conditional_mean = cross_covariance @ _posterior_factor_solve(
        resolution_factor, resolution_observed
    )
    conditional_covariance = error_covariance - cross_covariance @ (
        _posterior_factor_solve(resolution_factor, cross_covariance.T)
    )
    conditional_covariance = 0.5 * (
        conditional_covariance + conditional_covariance.T
    )
    _posterior_cholesky(conditional_covariance, "conditional numerical covariance")
    return (
        conditional_mean[_PARENT_COVARIANCE_INDICES],
        conditional_covariance[
            np.ix_(_PARENT_COVARIANCE_INDICES, _PARENT_COVARIANCE_INDICES)
        ],
    )
def _oracle_evaluate_conditional_posteriors(
    data: np.ndarray,
    templates: np.ndarray,
    covariances: np.ndarray,
    amplitude_prior_scale: float = 0.75,
) -> np.ndarray:
    record = np.asarray(data, dtype=float)
    design_pack = np.asarray(templates, dtype=float)
    covariance_pack = np.asarray(covariances, dtype=float)
    prior_scale = float(amplitude_prior_scale)
    if record.shape != (7, 9) or not np.all(np.isfinite(record)):
        raise ValueError("data must have finite shape (7, 9)")
    if design_pack.ndim != 3 or design_pack.shape[1:] != (18, 4) or design_pack.shape[0] == 0 or not np.all(np.isfinite(design_pack)):
        raise ValueError("templates must have nonempty finite shape (n, 18, 4)")
    if covariance_pack.shape != (design_pack.shape[0], 56, 56) or not np.all(np.isfinite(covariance_pack)):
        raise ValueError("covariances must have finite shape (n, 56, 56)")
    if np.any(design_pack[:, 14, 1] <= 0.0):
        raise ValueError("template masses must be positive")
    if (
        not np.isfinite(prior_scale)
        or prior_scale <= 0.0
        or prior_scale > np.sqrt(np.finfo(float).max)
    ):
        raise ValueError("amplitude_prior_scale must be finite, positive, and squareable")
    y_complex = record[:, 1] + 1j * record[:, 2]
    observed = np.concatenate((y_complex.real, y_complex.imag))
    resolution_observed = np.concatenate(
        (record[:, 5], record[:, 7], record[:, 6], record[:, 8])
    )
    output = np.zeros((design_pack.shape[0], 25), dtype=float)

    for index, (template, full_covariance) in enumerate(zip(design_pack, covariance_pack)):
        full_factor = _posterior_cholesky(full_covariance, "joint numerical covariance")
        del full_factor
        design = template[:14]
        first_active = np.linalg.norm(design[:, :2], axis=0) > 0.0
        mixed_active = np.linalg.norm(design[:, 2:4], axis=0) > 0.0
        if not np.all(first_active) or mixed_active[0] != mixed_active[1]:
            raise ValueError("active complex-mode columns must occur in real-imaginary pairs")
        active = np.array([0, 1, 2, 3] if np.all(mixed_active) else [0, 1])
        selected_design = design[:, active]
        if np.linalg.matrix_rank(selected_design) != active.size:
            raise ValueError("active parent design must have full column rank")
        error_mean, covariance = _condition_parent_error(
            full_covariance, resolution_observed
        )
        conditioned_observed = observed - error_mean
        covariance_factor = _posterior_cholesky(covariance, "parent numerical covariance")
        inverse_design = _posterior_factor_solve(covariance_factor, selected_design)
        inverse_observed = _posterior_factor_solve(
            covariance_factor, conditioned_observed
        )
        prior_variance = prior_scale * prior_scale
        fisher = selected_design.T @ inverse_design
        fisher += np.eye(active.size) / prior_variance
        fisher = 0.5 * (fisher + fisher.T)
        fisher_factor = _posterior_cholesky(
            fisher, "active amplitude posterior precision"
        )
        posterior_covariance = _posterior_factor_solve(
            fisher_factor, np.eye(active.size)
        )
        posterior_covariance = 0.5 * (posterior_covariance + posterior_covariance.T)
        _posterior_cholesky(posterior_covariance, "active posterior covariance")
        mean = posterior_covariance @ selected_design.T @ inverse_observed
        quadratic = conditioned_observed @ inverse_observed - mean @ fisher @ mean
        logdet_covariance = 2.0 * np.log(np.diag(covariance_factor)).sum()
        logdet_fisher = 2.0 * np.log(np.diag(fisher_factor)).sum()
        score = -0.5 * (
            quadratic
            + logdet_covariance
            + logdet_fisher
            + active.size * np.log(prior_variance)
        )

        full_mean = np.zeros(4, dtype=float)
        full_mean[active] = mean
        full_posterior_covariance = np.zeros((4, 4), dtype=float)
        full_posterior_covariance[np.ix_(active, active)] = posterior_covariance
        significances = np.zeros(2, dtype=float)
        for mode in range(active.size // 2):
            mode_slice = slice(2 * mode, 2 * mode + 2)
            marginal = posterior_covariance[mode_slice, mode_slice]
            marginal_factor = _posterior_cholesky(
                marginal, "complex-amplitude marginal covariance"
            )
            whitened = np.linalg.solve(marginal_factor, mean[mode_slice])
            significances[mode] = -np.expm1(-0.5 * (whitened @ whitened))

        output[index, :2] = template[14, :2]
        output[index, 2:6] = full_mean
        output[index, 6:22] = full_posterior_covariance.ravel(order="C")
        output[index, 22] = score
        output[index, 23:25] = np.clip(significances, 0.0, 1.0)
    if not np.all(np.isfinite(output)):
        raise ValueError("analytic posterior produced nonfinite output")
    return output

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            'setup': "def run_case(fn_under_test, fixture_load_ringdown_data, fixture_interpolate_spin_tables, fixture_synthesize_quadratic_response, fixture_build_linearized_templates, fixture_build_correlated_covariances):\n    import numpy as np\n    data = fixture_load_ringdown_data(0.91, 1.06)\n    tab = fixture_interpolate_spin_tables(np.array([0.74, 0.79, 0.84]), np.array([0.97, 1.0, 1.03]))\n    rsp = fixture_synthesize_quadratic_response(tab)\n    tpl = fixture_build_linearized_templates(data, tab, rsp, True)\n    freq = np.column_stack((tpl[:, 14, 2:4], tpl[:, 15, 2:4]))\n    cov = fixture_build_correlated_covariances(data, freq, 'anchor_gp', 0.86, 1.13)\n    return fn_under_test(data, tpl, cov, 0.61)",
            'call': 'run_case(evaluate_conditional_posteriors, load_ringdown_data, interpolate_spin_tables, synthesize_quadratic_response, build_linearized_templates, build_correlated_covariances)',
            'gold_call': 'run_case(_oracle_evaluate_conditional_posteriors, _oracle_load_ringdown_data, _oracle_interpolate_spin_tables, _oracle_synthesize_quadratic_response, _oracle_build_linearized_templates, _oracle_build_correlated_covariances)',
        },
        {
            'setup': "def run_case(fn_under_test, fixture_load_ringdown_data, fixture_interpolate_spin_tables, fixture_build_linearized_templates, fixture_build_correlated_covariances):\n    import numpy as np\n\n    def packed(fn):\n        out = fn()\n        pc = out[:, 6:22].reshape((-1, 4, 4))\n        return np.concatenate(([int(np.all(out[:, 4:6] == 0.0)), int(np.all(pc[:, 2:, :] == 0.0)), int(np.all(pc[:, :, 2:] == 0.0)), int(np.all(out[:, 24] == 0.0))], out.ravel()))\n    data = fixture_load_ringdown_data(1.08, 0.95)\n    tab = fixture_interpolate_spin_tables(np.array([0.71, 0.83]))\n    rsp = np.zeros((2, 4))\n    tpl = fixture_build_linearized_templates(data, tab, rsp, False)\n    freq = np.column_stack((tpl[:, 14, 2:4], tpl[:, 15, 2:4]))\n    cov = fixture_build_correlated_covariances(data, freq, 'generic_exponential')\n    return packed(lambda: fn_under_test(data, tpl, cov, 1.17))",
            'call': 'run_case(evaluate_conditional_posteriors, load_ringdown_data, interpolate_spin_tables, build_linearized_templates, build_correlated_covariances)',
            'gold_call': 'run_case(_oracle_evaluate_conditional_posteriors, _oracle_load_ringdown_data, _oracle_interpolate_spin_tables, _oracle_build_linearized_templates, _oracle_build_correlated_covariances)',
        },
        {
            'setup': "def run_case(fn_under_test, fixture_load_ringdown_data, fixture_interpolate_spin_tables, fixture_synthesize_quadratic_response, fixture_build_linearized_templates, fixture_build_correlated_covariances):\n    import numpy as np\n    data = fixture_load_ringdown_data(0.97, 1.02)\n    changed = data.copy()\n    changed[:, 5:7] = data[[3, 0, 6, 2, 5, 1, 4], 5:7]\n    tab = fixture_interpolate_spin_tables(np.array([0.73, 0.81]))\n    rsp = fixture_synthesize_quadratic_response(tab)\n    tpl = fixture_build_linearized_templates(data, tab, rsp, True)\n    freq = np.column_stack((tpl[:, 14, 2:4], tpl[:, 15, 2:4]))\n    cov = fixture_build_correlated_covariances(data, freq, 'anchor_gp')\n\n    def packed(fn):\n        base = fn(data)\n        altered = fn(changed)\n        return np.concatenate(([int(np.any(np.abs(base - altered) > 1e-08))], base.ravel(), altered.ravel()))\n    return packed(lambda record: fn_under_test(record, tpl, cov, 0.75))",
            'call': 'run_case(evaluate_conditional_posteriors, load_ringdown_data, interpolate_spin_tables, synthesize_quadratic_response, build_linearized_templates, build_correlated_covariances)',
            'gold_call': 'run_case(_oracle_evaluate_conditional_posteriors, _oracle_load_ringdown_data, _oracle_interpolate_spin_tables, _oracle_synthesize_quadratic_response, _oracle_build_linearized_templates, _oracle_build_correlated_covariances)',
        },
        {
            'setup': 'def run_case(fn_under_test, fixture_load_ringdown_data, fixture_interpolate_spin_tables, fixture_build_linearized_templates, fixture_build_correlated_covariances):\n    import numpy as np\n\n    def catches(fn):\n        try:\n            fn()\n        except ValueError:\n            return 1\n        except Exception:\n            return 2\n        return 0\n    data = fixture_load_ringdown_data()\n    tab = fixture_interpolate_spin_tables(np.array([0.76]))\n    rsp = np.zeros((1, 4))\n    tpl = fixture_build_linearized_templates(data, tab, rsp, True)\n    freq = np.column_stack((tpl[:, 14, 2:4], tpl[:, 15, 2:4]))\n    cov = fixture_build_correlated_covariances(data, freq)\n    negative = -np.eye(56)[None]\n    singular = np.zeros((1, 56, 56))\n    asymmetric = cov.copy()\n    asymmetric[0, 0, 1] = 0.001\n    bad_tpl = tpl.copy()\n    bad_tpl[0, :14, 3] = 0.0\n    rank_tpl = tpl.copy()\n    rank_tpl[0, :14, 3] = rank_tpl[0, :14, 2]\n    bad_data = data.copy()\n    bad_data[0, 1] = np.nan\n    extra_cov = np.repeat(cov, 2, axis=0)\n    return np.array([catches(lambda: fn_under_test(data, tpl, negative)), catches(lambda: fn_under_test(data, tpl, singular)), catches(lambda: fn_under_test(data, tpl, asymmetric)), catches(lambda: fn_under_test(data, bad_tpl, cov)), catches(lambda: fn_under_test(data, rank_tpl, cov)), catches(lambda: fn_under_test(bad_data, tpl, cov)), catches(lambda: fn_under_test(data, tpl, extra_cov)), catches(lambda: fn_under_test(data, np.zeros((1, 17, 4)), cov)), catches(lambda: fn_under_test(data, tpl, cov, 0.0)), catches(lambda: fn_under_test(data, tpl, cov, np.nan))])',
            'call': 'run_case(evaluate_conditional_posteriors, load_ringdown_data, interpolate_spin_tables, build_linearized_templates, build_correlated_covariances)',
            'gold_call': 'run_case(_oracle_evaluate_conditional_posteriors, _oracle_load_ringdown_data, _oracle_interpolate_spin_tables, _oracle_build_linearized_templates, _oracle_build_correlated_covariances)',
        },
    ]
