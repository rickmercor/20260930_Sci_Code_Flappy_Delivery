"""
Run the complete pipeline and return the Frobenius norm of the selected-lag correlation gradient over all interior string images.

The full pathway-geometry sensitivity aggregates the analytic response of every interior image coordinate while holding endpoints fixed. This norm equals the maximum first-order observable increase over unit-Frobenius interior-string perturbations.

Returns
-------
float: one finite nonnegative native Python float containing the endpoint-constrained full pathway-geometry sensitivity norm at target_lag_step.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def binless_wham_geometry_sensitivity(
    points: "np.ndarray",
    strings: "np.ndarray",
    committor_traces: "np.ndarray",
    alpha: float,
    linear_coefficients: "np.ndarray",
    curvature_coefficients: "np.ndarray",
    orthogonal_scales: "np.ndarray",
    sample_counts: "np.ndarray",
    beta: float,
    lag_steps: "np.ndarray",
    tolerance: float,
    max_iterations: int,
    target_lag_step: int,
) -> float:
    r"""Return the full interior-string geometry sensitivity at one lag.

    Parameters
    ----------
    points : np.ndarray
        Finite pooled configurations with shape ``(n,d)``.
    strings : np.ndarray
        Finite pathway images with shape ``(w,m,d)``, where ``m >= 3``.
    committor_traces : np.ndarray
        Finite committor traces in ``[0,1]`` with shape ``(n,t)``.
    alpha : float
        Finite strictly positive string-kernel sharpness.
    linear_coefficients : np.ndarray
        Finite profile coefficients with shape ``(w,)``.
    curvature_coefficients : np.ndarray
        Finite curvature coefficients with shape ``(w,)``.
    orthogonal_scales : np.ndarray
        Finite nonnegative restraint scales with shape ``(w,)``.
    sample_counts : np.ndarray
        Finite positive pathway counts with shape ``(w,)``.
    beta : float
        Finite strictly positive inverse temperature.
    lag_steps : np.ndarray
        Nonempty valid one-dimensional integer-dtype lag array.
    tolerance : float
        Finite strictly positive inclusive convergence tolerance.
    max_iterations : int
        Positive integer update cap.
    target_lag_step : int
        Integer lag value occurring exactly once in lag_steps.

    Returns
    -------
    float
        Finite nonnegative native Python float equal to the Frobenius norm of
        the selected correlation gradient over interior string images.

    Raises
    ------
    ValueError
        If an upstream contract fails, fewer than three string images are
        supplied, the target lag is not an integer occurring exactly once, or
        the selected norm is nonfinite.
    RuntimeError
        If the WHAM offsets do not converge within max_iterations.

    Notes
    -----
    Use pathway 0 as the reporting reference and hold the first and last image
    of every pathway fixed when forming the final norm.
    """
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_binless_wham_geometry_sensitivity(
    points: "np.ndarray",
    strings: "np.ndarray",
    committor_traces: "np.ndarray",
    alpha: float,
    linear_coefficients: "np.ndarray",
    curvature_coefficients: "np.ndarray",
    orthogonal_scales: "np.ndarray",
    sample_counts: "np.ndarray",
    beta: float,
    lag_steps: "np.ndarray",
    tolerance: float,
    max_iterations: int,
    target_lag_step: int,
) -> float:
    strings_array = np.asarray(strings)
    if strings_array.ndim != 3 or strings_array.shape[1] < 3:
        raise ValueError("strings must contain at least three images")
    if isinstance(target_lag_step, (bool, np.bool_)) or not isinstance(target_lag_step, (int, np.integer)):
        raise ValueError("target_lag_step must be an integer")
    progress, progress_jacobian = _oracle_pcv_progress_geometry(
        points, strings, alpha
    )
    orthogonal, orthogonal_jacobian = _oracle_pcv_orthogonal_geometry(
        points, strings, alpha
    )
    bias_values, bias_jacobian = _oracle_pathway_bias_geometry(
        progress,
        progress_jacobian,
        orthogonal,
        orthogonal_jacobian,
        linear_coefficients,
        curvature_coefficients,
        orthogonal_scales,
    )
    offsets, _ = _oracle_solve_wham_offsets(
        bias_values,
        sample_counts,
        beta,
        tolerance,
        max_iterations,
        0,
    )
    _, _, correlation_gradient = _oracle_wham_geometry_gradient(
        bias_values,
        bias_jacobian,
        sample_counts,
        beta,
        offsets,
        committor_traces,
        lag_steps,
        0,
    )
    lag_array = np.asarray(lag_steps)
    matches = np.flatnonzero(lag_array == int(target_lag_step))
    if matches.size != 1:
        raise ValueError("target_lag_step must occur exactly once")
    interior_gradient = correlation_gradient[int(matches[0]), :, 1:-1, :]
    result = float(np.linalg.norm(interior_gradient))
    if not np.isfinite(result) or result < 0.0:
        raise ValueError("geometry sensitivity must be finite and nonnegative")
    return result

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases() -> list[dict[str, str]]:
    """Return workflow, permutation, zero-signal, and invalid-input cases."""
    call = "binless_wham_geometry_sensitivity(points.copy(),strings.copy(),traces.copy(),alpha,linear.copy(),curvature.copy(),scales.copy(),counts.copy(),beta,lags.copy(),tolerance,max_iterations,target)"
    gold_call = "_oracle_binless_wham_geometry_sensitivity(points.copy(),strings.copy(),traces.copy(),alpha,linear.copy(),curvature.copy(),scales.copy(),counts.copy(),beta,lags.copy(),tolerance,max_iterations,target)"
    base = """import numpy as np
strings=np.array([[[-1.,0.],[0.,.3],[1.,0.]],[[-1.,0.],[0.,-.25],[1.,0.]]])
points=np.array([[-.8,.05],[-.2,.25],[.3,-.1],[.85,.02]])
traces=np.array([[.05,.1,.2,.35,.55],[.2,.3,.45,.65,.8],[.4,.5,.62,.75,.9],[.7,.78,.86,.93,.98]])
alpha=10.0; linear=np.array([.4,-.2]); curvature=np.array([.8,1.1])
scales=np.array([1.5,2.0]); counts=np.array([2.0,2.0]); beta=1.2
lags=np.array([1,2],dtype=int); tolerance=1e-12; max_iterations=10000; target=2
"""
    return [
        {"setup": base, "call": call, "gold_call": gold_call},
        {
            "setup": base + "order=np.array([2,0,3,1]); points=points[order]; traces=traces[order]\n",
            "call": call,
            "gold_call": gold_call,
        },
        {
            "setup": base + "traces=np.full_like(traces,0.5)\n",
            "call": call,
            "gold_call": gold_call,
        },
        {
            "setup": base + "target=3\n" + "def model():\n    try: binless_wham_geometry_sensitivity(points.copy(),strings.copy(),traces.copy(),alpha,linear.copy(),curvature.copy(),scales.copy(),counts.copy(),beta,lags.copy(),tolerance,max_iterations,target); return 0\n    except ValueError: return 1\n    except Exception: return 2\ndef oracle():\n    try: _oracle_binless_wham_geometry_sensitivity(points.copy(),strings.copy(),traces.copy(),alpha,linear.copy(),curvature.copy(),scales.copy(),counts.copy(),beta,lags.copy(),tolerance,max_iterations,target); return 0\n    except ValueError: return 1\n    except Exception: return 2\n",
            "call": "model()",
            "gold_call": "oracle()",
        },
        {
            "setup": base + "tolerance=0.0\n" + "def model():\n    try: binless_wham_geometry_sensitivity(points.copy(),strings.copy(),traces.copy(),alpha,linear.copy(),curvature.copy(),scales.copy(),counts.copy(),beta,lags.copy(),tolerance,max_iterations,target); return 0\n    except ValueError: return 1\n    except Exception: return 2\ndef oracle():\n    try: _oracle_binless_wham_geometry_sensitivity(points.copy(),strings.copy(),traces.copy(),alpha,linear.copy(),curvature.copy(),scales.copy(),counts.copy(),beta,lags.copy(),tolerance,max_iterations,target); return 0\n    except ValueError: return 1\n    except Exception: return 2\n",
            "call": "model()",
            "gold_call": "oracle()",
        },
    ]
