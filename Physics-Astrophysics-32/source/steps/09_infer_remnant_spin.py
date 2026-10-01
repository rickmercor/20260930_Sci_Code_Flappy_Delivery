"""
Final binary-black-hole remnant-spin inference.

Use spins `0.738,0.748,...,0.848` with a uniform discrete prior. Map the seven-point Gauss-Legendre rule to the uniform mass-ratio prior on `[0.96,1.04]`. The default calls every preceding step function with `error_model="anchor_gp"`, full residual conditioning, and the parent-conditioned child prediction from step 07.

Evaluate $M_0$ with only the 220 parent and self-coupled child and $M_1$ with both parents and both child responses. For each model evaluate the three coupled calibration regimes $(s_r,s_a,q,\rho,\gamma)=(0.75,0.55,0.40,0.82,0.10),(1.00,0.75,0.50,0.85,0.08),(1.25,1.20,0.48,0.82,0.30)$ with prior probabilities $p_r=(0.75,0.15,0.10)$. Here $s_r$ multiplies both responses, $s_a$ is the proper Cartesian amplitude prior scale, and $(q,\rho,\gamma)$ defines a separate covariance and conditional likelihood for that regime. Marginalize mass separately in all six cells defined by regime and model. At each spin combine the six log evidences with cell weight $p_r/2$, then normalize the twelve combined spin scores and return $\sum_k\chi_kW_k$. Do not factorize the supplied regime tuples, reuse one covariance for all regimes, or average separately normalized cell posteriors. $generic_exponential$ and $white_noise$ are alternate error models; $response_scale=0$ applies the zero response option, while the public $amplitude_prior_scale$ multiplies all three regime prior scales.

Returns
-------
Native finite Python `float`, compared at `1e-9`.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np
def infer_remnant_spin(
    response_scale: float = 1.0,
    error_model: str = "anchor_gp",
    amplitude_scale: float = 1.0,
    time_scale: float = 1.0,
    amplitude_prior_scale: float = 1.0,
) -> float:
    """Return the posterior mean remnant spin for the requested configuration.

    Parameters
    ----------
    response_scale : float, default=1.0
        Finite nonnegative multiplier on all three coupled-regime nonlinear
        response scales. The task answer uses one; zero applies the zero-
        response option in every regime.
    error_model : str, default="anchor_gp"
        One of `"anchor_gp"`, `"generic_exponential"`, or `"white_noise"`.
        The task answer uses `"anchor_gp"`.
    amplitude_scale : float, default=1.0
        Finite positive multiplier on all waveform and resolution-difference
        columns.
    time_scale : float, default=1.0
        Finite positive multiplier on all recorded times.
    amplitude_prior_scale : float, default=1.0
        Finite positive multiplier on all three coupled-regime Cartesian
        amplitude-prior standard deviations.

    Returns
    -------
    float
        Posterior mean dimensionless remnant spin after analytic amplitude,
        mass, coupled calibration-regime, and equal conditional parent-model
        marginalization of state scores that include residual-conditioned
        waveform likelihoods and the residual likelihood, compared at
        tolerance `1e-9`.

    Raises
    ------
    ValueError
        If response_scale is negative or nonfinite, a data scale or the
        amplitude-prior multiplier is not finite and positive, or error_model
        is unsupported.
        Errors raised by an earlier step retain their documented meaning.
    """
    return None
import numpy as np

# =============================================================================
# GOLD SOLUTION
# =============================================================================

r"""Final binary-black-hole remnant-spin inference.

Use spins `0.738, 0.748, ..., 0.848` with a uniform discrete prior and a
uniform remnant-mass-ratio prior on `[0.96, 1.04]`. Integrate the mass prior
with the seven-point Gauss-Legendre rule mapped to that interval and normalized
to unit total weight. The default path uses `error_model="anchor_gp"`, the
joint residual-conditioned numerical-error hierarchy, together with the full
correlated residual likelihood and parent-conditioned child prediction from
step 07.
Evaluate two parent-content hypotheses: `M0` contains only the 220 parent and
its self-coupled child, while `M1` contains the 220 and 320 parents and both
self and mixed child responses. Also marginalize three task-supplied coupled
calibration regimes `(response scale, amplitude-prior scale, q, rho, gamma)`
equal to `(0.75,0.55,0.40,0.82,0.10)`, `(1.00,0.75,0.50,0.85,0.08)`, and
`(1.25,1.20,0.48,0.82,0.30)`, with prior probabilities `0.75`, `0.15`, and
`0.10`. These are paired regimes, not independent parameter grids, and each
requires its own covariance and conditional likelihood. Within every regime,
`M0` and `M1` have equal conditional probability. Marginalize mass separately
in all six regime-model cells, combine each cell with prior weight `p_r/2`,
then normalize the spin posterior. Do not average already-normalized spin
posteriors. The optional scale arguments multiply the three response or
amplitude-prior entries and provide alternate configurations.

Returns
-------
A native finite `float`, the posterior mean remnant spin, compared at `1e-9`.
"""
import numpy as np
import numpy as np
_INFERENCE_SPINS = np.array([
    0.738, 0.748, 0.758, 0.768, 0.778, 0.788,
    0.798, 0.808, 0.818, 0.828, 0.838, 0.848,
])
_CALIBRATION_REGIMES = (
    (0.75, 0.75, 0.55, 0.40, 0.82, 0.10),
    (0.15, 1.00, 0.75, 0.50, 0.85, 0.08),
    (0.10, 1.25, 1.20, 0.48, 0.82, 0.30),
)
def _oracle_infer_remnant_spin(
    response_scale: float = 1.0,
    error_model: str = "anchor_gp",
    amplitude_scale: float = 1.0,
    time_scale: float = 1.0,
    amplitude_prior_scale: float = 1.0,
) -> float:
    scale = float(response_scale)
    if not np.isfinite(scale) or scale < 0.0:
        raise ValueError("response_scale must be finite and nonnegative")
    data = _oracle_load_ringdown_data(amplitude_scale, time_scale)
    mass_nodes_standard, mass_weights_standard = np.polynomial.legendre.leggauss(7)
    mass_nodes = 1.0 + 0.04 * mass_nodes_standard
    state_spins = np.repeat(_INFERENCE_SPINS, mass_nodes.size)
    state_masses = np.tile(mass_nodes, _INFERENCE_SPINS.size)
    mass_log_weights = np.tile(np.log(mass_weights_standard / 2.0), _INFERENCE_SPINS.size)
    table = _oracle_interpolate_spin_tables(state_spins, state_masses)
    response = _oracle_synthesize_quadratic_response(table)
    templates_m0 = _oracle_build_linearized_templates(data, table, response, False)
    templates_m1 = _oracle_build_linearized_templates(data, table, response, True)
    block_frequencies = np.column_stack((
        templates_m1[:, 14, 2:4], templates_m1[:, 15, 2:4]
    ))
    cell_rows = []
    cell_log_weights = []
    for (
        regime_weight,
        regime_response,
        regime_prior_scale,
        regime_q,
        regime_rho,
        regime_gamma,
    ) in _CALIBRATION_REGIMES:
        covariances = _oracle_build_correlated_covariances(
            data,
            block_frequencies,
            error_model,
            1.0,
            1.0,
            regime_q,
            regime_rho,
            regime_gamma,
        )
        for templates in (templates_m0, templates_m1):
            posteriors = _oracle_evaluate_conditional_posteriors(
                data,
                templates,
                covariances,
                amplitude_prior_scale * regime_prior_scale,
            )
            predictive_scores = _oracle_score_heldout_h64(
                data,
                templates,
                covariances,
                posteriors,
                scale * regime_response,
            )
            cell_rows.append(
                _oracle_normalize_spin_weights(
                    predictive_scores, mass_log_weights
                )
            )
            cell_log_weights.append(np.log(regime_weight / 2.0))
    reference_spins = cell_rows[0][:, 0]
    if any(
        not np.array_equal(rows[:, 0], reference_spins)
        for rows in cell_rows[1:]
    ):
        raise ValueError("regime-model spin grids must agree")
    cell_log_evidence = np.stack([rows[:, 1] for rows in cell_rows])
    weighted_log_evidence = cell_log_evidence + np.asarray(
        cell_log_weights
    )[:, None]
    cell_maximum = np.max(weighted_log_evidence, axis=0)
    combined_log_evidence = cell_maximum + np.log(
        np.sum(np.exp(weighted_log_evidence - cell_maximum), axis=0)
    )
    maximum = float(np.max(combined_log_evidence))
    spin_weights = np.exp(combined_log_evidence - maximum)
    spin_weights /= spin_weights.sum()
    answer = float(reference_spins @ spin_weights)
    if not np.isfinite(answer):
        raise ValueError("posterior mean is nonfinite")
    return answer

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Default, counterfactual, and direct invalid end-to-end calls."""
    return [
        {
            "setup": "response_scale=.63; error_model='generic_exponential'; amplitude_scale=.91; time_scale=1.06",
            "call": "infer_remnant_spin(response_scale,error_model,amplitude_scale,time_scale,.58)",
            "gold_call": "_oracle_infer_remnant_spin(response_scale,error_model,amplitude_scale,time_scale,.58)",
        },
        {
            "setup": "response_scale=0.; error_model='white_noise'; amplitude_scale=1.08; time_scale=.95",
            "call": "infer_remnant_spin(response_scale,error_model,amplitude_scale,time_scale,1.1)",
            "gold_call": "_oracle_infer_remnant_spin(response_scale,error_model,amplitude_scale,time_scale,1.1)",
        },
        {
            "setup": "import numpy as np\ndef catches(fn):\n try: fn()\n except ValueError: return 1\n except Exception: return 2\n return 0",
            "call": "np.array([catches(lambda: infer_remnant_spin(-1.)),catches(lambda: infer_remnant_spin(np.nan)),catches(lambda: infer_remnant_spin(1.,'matern')),catches(lambda: infer_remnant_spin(1.,'anchor_gp',0.,1.)),catches(lambda: infer_remnant_spin(1.,'anchor_gp',1.,np.inf)),catches(lambda: infer_remnant_spin(1.,'anchor_gp',1.,1.,0.)),catches(lambda: infer_remnant_spin(1.,'anchor_gp',1.,1.,np.nan))])",
            "gold_call": "np.array([catches(lambda: _oracle_infer_remnant_spin(-1.)),catches(lambda: _oracle_infer_remnant_spin(np.nan)),catches(lambda: _oracle_infer_remnant_spin(1.,'matern')),catches(lambda: _oracle_infer_remnant_spin(1.,'anchor_gp',0.,1.)),catches(lambda: _oracle_infer_remnant_spin(1.,'anchor_gp',1.,np.inf)),catches(lambda: _oracle_infer_remnant_spin(1.,'anchor_gp',1.,1.,0.)),catches(lambda: _oracle_infer_remnant_spin(1.,'anchor_gp',1.,1.,np.nan))])",
        },
    ]
