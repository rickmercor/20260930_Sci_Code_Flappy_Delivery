"""
Risk-load five diagnostic distributions independently and apply completeness, memory, and tolerance filters.

Use weighted population moments. Normalize only after forming each one-sided upper bound; do not combine diagnostics before uncertainty loading. `sigma_weights` must contain one finite strictly positive weight per sigma node and sum to one within absolute tolerance `1e-12`; otherwise raise `ValueError`.

Returns
-------
Return one real NumPy array of shape (number_of_candidates,13), with columns in this order: [desired_hz, centre_f_calc_hz, max_stride, peak_bound_normalized, zero_time_bound_normalized, nrmse_bound_normalized, spectral_bound_normalized, tail_bound_normalized, mean_tail, std_tail, tail_upper_bound, composite_bound, feasible].
"""

import numpy as np

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def uncertainty_risk_table(candidate_desired_hz, sigma_sampling_plans,
                           sigma_metrics, sigma_weights, risk_quantile,
                           peak_tolerance, zero_tolerance_s, rms_tolerance,
                           spectral_tolerance, tail_power_tolerance,
                           memory_limit_roundtrips):
    """Return an (number_of_candidates,13) float table with columns
    [desired_hz, centre_f_calc_hz, max_stride,
    peak_bound_normalized, zero_time_bound_normalized,
    nrmse_bound_normalized, spectral_bound_normalized,
    tail_bound_normalized, mean_tail, std_tail, tail_upper_bound,
    composite_bound, feasible] in that order. composite_bound is the largest of
    the five normalized bounds, and feasible is 1.0 or 0.0.
    """
    return np.empty((np.asarray(candidate_desired_hz).size, 13), dtype=float)

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_uncertainty_risk_table(candidate_desired_hz, sigma_sampling_plans,
                                   sigma_metrics, sigma_weights, risk_quantile,
                                   peak_tolerance, zero_tolerance_s, rms_tolerance,
                                   spectral_tolerance, tail_power_tolerance,
                                   memory_limit_roundtrips):
    """Propagate five separate metric distributions into a risk table."""
    import numpy as np
    candidates = np.asarray(candidate_desired_hz, dtype=float)
    plans = np.asarray(sigma_sampling_plans, dtype=float)
    metrics = np.asarray(sigma_metrics, dtype=float)
    weights = np.asarray(sigma_weights, dtype=float)
    if (candidates.ndim != 1 or candidates.size == 0
            or not np.all(np.isfinite(candidates)) or np.any(candidates <= 0)):
        raise ValueError("candidate frequencies must be a finite positive vector")
    if plans.ndim != 3 or plans.shape[1:] != (candidates.size, 6):
        raise ValueError("sigma_sampling_plans must have shape (H,C,6)")
    if metrics.shape != plans.shape[:2] + (5,) or weights.shape != (plans.shape[0],):
        raise ValueError("metric, weight, and plan shapes disagree")
    if (not np.all(np.isfinite(plans)) or not np.all(np.isfinite(metrics))
            or not np.all(np.isfinite(weights)) or np.any(metrics < 0)
            or np.any(weights <= 0)):
        raise ValueError("plans, metrics, and positive weights must be finite")
    if not np.isclose(np.sum(weights), 1.0, rtol=0.0, atol=1e-12):
        raise ValueError("sigma weights must sum to one")
    tolerances = np.asarray([peak_tolerance, zero_tolerance_s, rms_tolerance,
                             spectral_tolerance, tail_power_tolerance], dtype=float)
    try:
        risk_value = float(risk_quantile)
        memory_value = float(memory_limit_roundtrips)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("risk, tolerance, or memory input is invalid") from exc
    if (not np.isfinite(risk_value) or risk_value < 0
            or not np.isfinite(memory_value) or memory_value < 1
            or memory_value != int(memory_value)
            or np.any(tolerances <= 0) or not np.all(np.isfinite(tolerances))):
        raise ValueError("risk, tolerance, or memory input is invalid")
    memory_value = int(memory_value)
    mean_metric = np.einsum("h,hcm->cm", weights, metrics)
    metric_std = np.sqrt(np.einsum(
        "h,hcm->cm", weights, (metrics - mean_metric[None, :, :])**2
    ))
    upper_metric = mean_metric + risk_value * metric_std
    normalized_upper = upper_metric / tolerances[None, :]
    composite = np.max(normalized_upper, axis=1)
    max_stride = np.max(plans[:, :, 2], axis=0)
    any_partial = np.max(plans[:, :, 4], axis=0)
    rows = []
    for i, desired in enumerate(candidates):
        feasible = float(any_partial[i] == 0.0
                         and max_stride[i] <= memory_value
                         and composite[i] <= 1.0)
        rows.append([desired, plans[0, i, 0], max_stride[i],
                     normalized_upper[i, 0], normalized_upper[i, 1],
                     normalized_upper[i, 2], normalized_upper[i, 3],
                     normalized_upper[i, 4], mean_metric[i, 4],
                     metric_std[i, 4], upper_metric[i, 4], composite[i], feasible])
    return np.asarray(rows, dtype=float)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {"setup":'import numpy as np\ncandidate_desired_hz=np.array([100.,200.,300.])\nbase=np.array([[102.,1/102.,5.,1.,0.,.98],[198.,1/198.,3.,1.,0.,.99],[300.,1/300.,2.,1.,0.,1.]])\nsigma_sampling_plans=np.stack((base,base.copy(),base.copy())); sigma_sampling_plans[2,0,2]=7.\nsigma_metrics=np.array([[[.10,1e-6,.30,.08,.03],[.08,4e-7,.12,.04,.02],[.20,8e-7,.40,.09,.04]],[[.14,9e-7,.25,.10,.05],[.06,3e-7,.10,.03,.01],[.18,7e-7,.35,.08,.03]],[[.08,8e-7,.20,.06,.02],[.10,5e-7,.15,.05,.04],[.16,6e-7,.30,.07,.02]]])\nsigma_weights=np.array([.5,.25,.25]); risk_quantile=1.\npeak_tolerance=.20; zero_tolerance_s=1.5e-6; rms_tolerance=.55\nspectral_tolerance=.10; tail_power_tolerance=.05; memory_limit_roundtrips=6\n',"call":"uncertainty_risk_table(candidate_desired_hz,sigma_sampling_plans,sigma_metrics,sigma_weights,risk_quantile,peak_tolerance,zero_tolerance_s,rms_tolerance,spectral_tolerance,tail_power_tolerance,memory_limit_roundtrips)","gold_call":"_oracle_uncertainty_risk_table(candidate_desired_hz,sigma_sampling_plans,sigma_metrics,sigma_weights,risk_quantile,peak_tolerance,zero_tolerance_s,rms_tolerance,spectral_tolerance,tail_power_tolerance,memory_limit_roundtrips)"},
        {"setup":'import numpy as np\ncandidate_desired_hz=np.array([250.])\nsigma_sampling_plans=np.array([[[248.,1/248.,4.,1.,0.,.99]],[[252.,1/252.,4.,1.,0.,.99]],[[250.,1/250.,4.,1.,0.,1.]]])\nsigma_metrics=np.array([[[.08,4e-7,.15,.03,.015]],[[.10,5e-7,.18,.04,.020]],[[.09,6e-7,.16,.05,.025]]])\nsigma_weights=np.array([.5,.25,.25]); risk_quantile=1.2\npeak_tolerance=.3; zero_tolerance_s=2e-6; rms_tolerance=.6\nspectral_tolerance=.2; tail_power_tolerance=.08; memory_limit_roundtrips=4\n',"call":"uncertainty_risk_table(candidate_desired_hz,sigma_sampling_plans,sigma_metrics,sigma_weights,risk_quantile,peak_tolerance,zero_tolerance_s,rms_tolerance,spectral_tolerance,tail_power_tolerance,memory_limit_roundtrips)","gold_call":"_oracle_uncertainty_risk_table(candidate_desired_hz,sigma_sampling_plans,sigma_metrics,sigma_weights,risk_quantile,peak_tolerance,zero_tolerance_s,rms_tolerance,spectral_tolerance,tail_power_tolerance,memory_limit_roundtrips)"},
        {"setup":'import numpy as np\ncandidate_desired_hz=np.array([80.,160.])\nbase=np.array([[82.,1/82.,7.,1.,0.,.97],[158.,1/158.,8.,1.,0.,.98]])\nsigma_sampling_plans=np.stack((base,base.copy()))\nsigma_metrics=np.array([[[.01,1e-7,.02,.01,.005],[.02,2e-7,.03,.01,.006]],[[.02,2e-7,.03,.02,.007],[.03,3e-7,.04,.02,.008]]])\nsigma_weights=np.array([.5,.5]); risk_quantile=.5\npeak_tolerance=.5; zero_tolerance_s=5e-6; rms_tolerance=.8\nspectral_tolerance=.4; tail_power_tolerance=.2; memory_limit_roundtrips=5\n',"call":"uncertainty_risk_table(candidate_desired_hz,sigma_sampling_plans,sigma_metrics,sigma_weights,risk_quantile,peak_tolerance,zero_tolerance_s,rms_tolerance,spectral_tolerance,tail_power_tolerance,memory_limit_roundtrips)","gold_call":"_oracle_uncertainty_risk_table(candidate_desired_hz,sigma_sampling_plans,sigma_metrics,sigma_weights,risk_quantile,peak_tolerance,zero_tolerance_s,rms_tolerance,spectral_tolerance,tail_power_tolerance,memory_limit_roundtrips)"},
        {"setup":'import numpy as np\ncandidate_desired_hz=np.array([100.]); sigma_sampling_plans=np.array([[[100.,.01,2.,1.,0.,1.]],[[101.,1/101.,2.,1.,0.,.99]]])\nsigma_metrics=np.full((2,1,5),.01); sigma_weights=np.array([.6,.3]); risk_quantile=1.\npeak_tolerance=1.; zero_tolerance_s=1.; rms_tolerance=1.; spectral_tolerance=1.; tail_power_tolerance=1.; memory_limit_roundtrips=5\ndef _weight_sentinel(fn):\n    try:\n        fn(candidate_desired_hz,sigma_sampling_plans,sigma_metrics,sigma_weights,risk_quantile,peak_tolerance,zero_tolerance_s,rms_tolerance,spectral_tolerance,tail_power_tolerance,memory_limit_roundtrips)\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\n',"call":"_weight_sentinel(uncertainty_risk_table)","gold_call":"_weight_sentinel(_oracle_uncertainty_risk_table)"},
        {"setup":'import numpy as np\ncandidate_desired_hz=np.array([125.]); sigma_sampling_plans=np.array([[[125.,.008,3.,1.,0.,1.]],[[124.,1/124.,3.,1.,1.,.99]]])\nsigma_metrics=np.array([[[.05,1e-7,.05,.02,.01]],[[.06,2e-7,.06,.03,.02]]]); sigma_weights=np.array([.5,.5]); risk_quantile=0.\npeak_tolerance=1.; zero_tolerance_s=1e-5; rms_tolerance=1.; spectral_tolerance=1.; tail_power_tolerance=1.; memory_limit_roundtrips=3\n',"call":"float(uncertainty_risk_table(candidate_desired_hz,sigma_sampling_plans,sigma_metrics,sigma_weights,risk_quantile,peak_tolerance,zero_tolerance_s,rms_tolerance,spectral_tolerance,tail_power_tolerance,memory_limit_roundtrips)[0,12])","gold_call":"float(_oracle_uncertainty_risk_table(candidate_desired_hz,sigma_sampling_plans,sigma_metrics,sigma_weights,risk_quantile,peak_tolerance,zero_tolerance_s,rms_tolerance,spectral_tolerance,tail_power_tolerance,memory_limit_roundtrips)[0,12])"},
        {"setup":'import numpy as np\ncandidate_desired_hz=np.array([400.]); sigma_sampling_plans=np.array([[[400.,.0025,5.,1.,0.,1.]]])\nsigma_metrics=np.array([[[.2,2e-6,.5,.1,.04]]]); sigma_weights=np.array([1.]); risk_quantile=1.645\npeak_tolerance=.2; zero_tolerance_s=2e-6; rms_tolerance=.5; spectral_tolerance=.1; tail_power_tolerance=.04; memory_limit_roundtrips=5\n',"call":"uncertainty_risk_table(candidate_desired_hz,sigma_sampling_plans,sigma_metrics,sigma_weights,risk_quantile,peak_tolerance,zero_tolerance_s,rms_tolerance,spectral_tolerance,tail_power_tolerance,memory_limit_roundtrips)","gold_call":"_oracle_uncertainty_risk_table(candidate_desired_hz,sigma_sampling_plans,sigma_metrics,sigma_weights,risk_quantile,peak_tolerance,zero_tolerance_s,rms_tolerance,spectral_tolerance,tail_power_tolerance,memory_limit_roundtrips)"},
        {"setup":'import numpy as np\ncandidate_desired_hz=np.array([500.]); sigma_sampling_plans=np.array([[[490.,1/490.,2.,1.,0.,.98]],[[510.,1/510.,2.,1.,0.,.98]]])\nsigma_metrics=np.array([[[.10,1e-6,.20,.04,.02]],[[.30,3e-6,.40,.08,.06]]]); sigma_weights=np.array([.5,.5]); risk_quantile=0.\npeak_tolerance=.25; zero_tolerance_s=3e-6; rms_tolerance=.5; spectral_tolerance=.1; tail_power_tolerance=.1; memory_limit_roundtrips=2\n',"call":"uncertainty_risk_table(candidate_desired_hz,sigma_sampling_plans,sigma_metrics,sigma_weights,risk_quantile,peak_tolerance,zero_tolerance_s,rms_tolerance,spectral_tolerance,tail_power_tolerance,memory_limit_roundtrips)","gold_call":"_oracle_uncertainty_risk_table(candidate_desired_hz,sigma_sampling_plans,sigma_metrics,sigma_weights,risk_quantile,peak_tolerance,zero_tolerance_s,rms_tolerance,spectral_tolerance,tail_power_tolerance,memory_limit_roundtrips)"},
        {"setup":'import numpy as np\ncandidate_desired_hz=np.array([300.,100.,200.]); sigma_sampling_plans=np.array([[[290.,1/290.,2.,1.,0.,.96],[110.,1/110.,2.,1.,0.,.90],[205.,1/205.,2.,1.,0.,.975]]])\nsigma_metrics=np.array([[[.01,1e-7,.02,.01,.005],[.02,1e-7,.02,.01,.005],[.03,1e-7,.02,.01,.005]]]); sigma_weights=np.array([1.]); risk_quantile=0.\npeak_tolerance=.5; zero_tolerance_s=1e-5; rms_tolerance=.5; spectral_tolerance=.5; tail_power_tolerance=.5; memory_limit_roundtrips=2\n',"call":"uncertainty_risk_table(candidate_desired_hz,sigma_sampling_plans,sigma_metrics,sigma_weights,risk_quantile,peak_tolerance,zero_tolerance_s,rms_tolerance,spectral_tolerance,tail_power_tolerance,memory_limit_roundtrips)","gold_call":"_oracle_uncertainty_risk_table(candidate_desired_hz,sigma_sampling_plans,sigma_metrics,sigma_weights,risk_quantile,peak_tolerance,zero_tolerance_s,rms_tolerance,spectral_tolerance,tail_power_tolerance,memory_limit_roundtrips)"},
    ]
