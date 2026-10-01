"""
Implicitly differentiate the converged WHAM equations and normalized committor correlations with respect to every string-image coordinate.

A geometry perturbation changes pathway biases directly and relative offsets implicitly. Solving the gauge-fixed response system and propagating it through normalized weights yields complete offset and correlation gradients without finite differences

Returns
-------
tuple[np.ndarray, np.ndarray, np.ndarray]: offset gradients of shape (w, w, m, d), global correlations of shape (l,), and correlation gradients of shape (l, w, m, d), all float64 arrays.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def wham_geometry_gradient(
    bias_values: "np.ndarray",
    bias_jacobian: "np.ndarray",
    sample_counts: "np.ndarray",
    beta: float,
    free_energy_offsets: "np.ndarray",
    committor_traces: "np.ndarray",
    lag_steps: "np.ndarray",
    reference_index: int,
) -> tuple["np.ndarray", "np.ndarray", "np.ndarray"]:
    r"""Return offset and correlation gradients for all string coordinates.

    Parameters
    ----------
    bias_values : np.ndarray
        Finite bias matrix with shape ``(w, n)``.
    bias_jacobian : np.ndarray
        Finite local derivatives with shape ``(w, n, m, d)``; pathway ``i``
        entries differentiate ``bias_values[i]`` with respect to its own string.
    sample_counts : np.ndarray
        Finite positive pathway counts with shape ``(w,)``.
    beta : float
        Finite strictly positive inverse temperature.
    free_energy_offsets : np.ndarray
        Finite converged offsets with shape ``(w,)`` and a zero reference.
    committor_traces : np.ndarray
        Finite committor values in ``[0,1]`` with shape ``(n,t)``, ``t >= 2``.
    lag_steps : np.ndarray
        Nonempty one-dimensional array with integer dtype and entries satisfying
        ``1 <= lag < t``. A floating-point array is invalid even when every
        value is mathematically integral.
    reference_index : int
        In-range integer pathway index defining the fixed gauge.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        Float64 offset gradients with shape ``(w,w,m,d)``, correlations with
        shape ``(l,)``, and correlation gradients with shape ``(l,w,m,d)``.

    Raises
    ------
    ValueError
        If shapes or values violate the contract, counts or beta are not
        positive, the reference is invalid or nonzero, lag_steps lacks integer
        dtype or contains an invalid lag, the gauge-fixed system is singular,
        or an output is nonfinite.

    Notes
    -----
    Treat beta, committor traces, and sample counts as geometry-independent and
    differentiate the converged equations in the fixed reference gauge.
    """
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_wham_geometry_gradient(
    bias_values: "np.ndarray",
    bias_jacobian: "np.ndarray",
    sample_counts: "np.ndarray",
    beta: float,
    free_energy_offsets: "np.ndarray",
    committor_traces: "np.ndarray",
    lag_steps: "np.ndarray",
    reference_index: int,
) -> tuple["np.ndarray", "np.ndarray", "np.ndarray"]:
    bias_values = np.asarray(bias_values, dtype=np.float64)
    bias_jacobian = np.asarray(bias_jacobian, dtype=np.float64)
    sample_counts = np.asarray(sample_counts, dtype=np.float64)
    free_energy_offsets = np.asarray(free_energy_offsets, dtype=np.float64)
    committor_traces = np.asarray(committor_traces, dtype=np.float64)
    lag_steps = np.asarray(lag_steps)
    if bias_values.ndim != 2 or bias_values.shape[0] == 0 or bias_values.shape[1] == 0:
        raise ValueError("bias_values must have shape (w,n) with w,n > 0")
    w, n = bias_values.shape
    if bias_jacobian.ndim != 4 or bias_jacobian.shape[:2] != (w, n):
        raise ValueError("bias_jacobian must have shape (w,n,m,d)")
    m, d = bias_jacobian.shape[2:]
    if m < 2 or d < 1:
        raise ValueError("bias_jacobian requires m >= 2 and d >= 1")
    if sample_counts.shape != (w,) or free_energy_offsets.shape != (w,):
        raise ValueError("counts and offsets must have shape (w,)")
    arrays = (bias_values, bias_jacobian, sample_counts, free_energy_offsets)
    if not all(np.all(np.isfinite(values)) for values in arrays):
        raise ValueError("WHAM inputs must be finite")
    if np.any(sample_counts <= 0.0) or not np.isfinite(beta) or beta <= 0.0:
        raise ValueError("counts and beta must be finite and positive")
    if isinstance(reference_index, (bool, np.bool_)) or not isinstance(reference_index, (int, np.integer)):
        raise ValueError("reference_index must be an integer")
    reference_index = int(reference_index)
    if reference_index < 0 or reference_index >= w:
        raise ValueError("reference_index is out of range")
    if not np.isclose(free_energy_offsets[reference_index], 0.0, rtol=0.0, atol=1.0e-12):
        raise ValueError("the reference offset must be zero")
    if committor_traces.ndim != 2 or committor_traces.shape[0] != n or committor_traces.shape[1] < 2:
        raise ValueError("committor_traces must have shape (n,t) with t >= 2")
    if not np.all(np.isfinite(committor_traces)) or np.any(committor_traces < 0.0) or np.any(committor_traces > 1.0):
        raise ValueError("committor traces must be finite and lie in [0,1]")
    if lag_steps.ndim != 1 or lag_steps.size == 0 or not np.issubdtype(lag_steps.dtype, np.integer):
        raise ValueError("lag_steps must be a nonempty one-dimensional integer-dtype array")
    lag_steps = lag_steps.astype(int, copy=False)
    if np.any(lag_steps <= 0) or np.any(lag_steps >= committor_traces.shape[1]):
        raise ValueError("each lag must satisfy 1 <= lag < trace length")

    parameter_count = w * m * d
    bias_gradient = np.zeros((w, n, parameter_count), dtype=np.float64)
    for pathway in range(w):
        start = pathway * m * d
        stop = start + m * d
        bias_gradient[pathway, :, start:stop] = bias_jacobian[pathway].reshape(n, m * d)

    factors = _oracle_binless_reweighting_factors(
        bias_values, sample_counts, float(beta), free_energy_offsets
    )
    log_mixture = np.log(sample_counts)[:, None] - float(beta) * (
        bias_values - free_energy_offsets[:, None]
    )
    mixture_shift = np.max(log_mixture, axis=0, keepdims=True)
    mixture_probabilities = np.exp(log_mixture - mixture_shift)
    mixture_probabilities /= np.sum(mixture_probabilities, axis=0, keepdims=True)

    log_path = np.log(factors)[None, :] - float(beta) * bias_values
    path_shift = np.max(log_path, axis=1, keepdims=True)
    path_probabilities = np.exp(log_path - path_shift)
    path_probabilities /= np.sum(path_probabilities, axis=1, keepdims=True)

    active = np.array([i for i in range(w) if i != reference_index], dtype=int)
    offset_gradient = np.zeros((w, parameter_count), dtype=np.float64)
    if active.size:
        jacobian = np.empty((active.size, active.size), dtype=np.float64)
        for row, i in enumerate(active):
            for column, k in enumerate(active):
                jacobian[row, column] = (
                    (1.0 if i == k else 0.0)
                    - np.sum(
                        (path_probabilities[i] - path_probabilities[reference_index])
                        * mixture_probabilities[k]
                    )
                )
        mean_bias_gradient = np.einsum(
            "jn,jnp->np", mixture_probabilities, bias_gradient
        )
        direct_response = (
            np.einsum("in,np->ip", path_probabilities, mean_bias_gradient)
            - np.einsum("in,inp->ip", path_probabilities, bias_gradient)
        )
        right_hand_side = -(
            direct_response[active] - direct_response[reference_index]
        )
        try:
            offset_gradient[active] = np.linalg.solve(jacobian, right_hand_side)
        except np.linalg.LinAlgError as exc:
            raise ValueError("the gauge-fixed WHAM Jacobian is singular") from exc

    log_factor_gradient = float(beta) * np.einsum(
        "in,inp->np",
        mixture_probabilities,
        bias_gradient - offset_gradient[:, None, :],
    )
    conditional = np.empty((n, lag_steps.size), dtype=np.float64)
    for column, lag in enumerate(lag_steps):
        differences = committor_traces[:, lag:] - committor_traces[:, :-lag]
        conditional[:, column] = np.mean(differences * differences, axis=1)
    normalized_weights = factors / np.sum(factors)
    correlations = np.einsum("n,nl->l", normalized_weights, conditional)
    correlation_gradient = np.einsum(
        "n,nl,np->lp",
        normalized_weights,
        conditional - correlations[None, :],
        log_factor_gradient,
    )
    offset_gradient = offset_gradient.reshape(w, w, m, d)
    correlation_gradient = correlation_gradient.reshape(lag_steps.size, w, m, d)
    if not np.all(np.isfinite(offset_gradient)) or not np.all(np.isfinite(correlations)) or not np.all(np.isfinite(correlation_gradient)):
        raise ValueError("geometry-gradient outputs must be finite")
    return offset_gradient, correlations, correlation_gradient

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases() -> list[dict[str, str]]:
    """Return normal, one-pathway, zero-signal, and invalid-input cases."""
    call = "(lambda r:np.concatenate([r[0].ravel(),r[1].ravel(),r[2].ravel()]))(wham_geometry_gradient(bias.copy(),dbias.copy(),counts.copy(),beta,offsets.copy(),traces.copy(),lags.copy(),reference))"
    gold_call = "(lambda r:np.concatenate([r[0].ravel(),r[1].ravel(),r[2].ravel()]))(_oracle_wham_geometry_gradient(bias.copy(),dbias.copy(),counts.copy(),beta,offsets.copy(),traces.copy(),lags.copy(),reference))"
    return [
        {
            "setup": """import numpy as np
bias=np.array([[0.1,-0.2,0.4,0.0],[0.3,0.0,-0.1,0.2],[0.2,0.1,0.5,-0.3]])
dbias=np.arange(72,dtype=float).reshape(3,4,3,2)/200.0-0.1
counts=np.array([4.0,6.0,5.0]); beta=1.3
offsets=np.array([0.0,0.01364883603061084,0.0369752842074541])
traces=np.array([[0.0,0.2,0.5,0.9],[0.1,0.3,0.4,0.8],[0.2,0.25,0.55,0.85],[0.05,0.2,0.6,0.95]])
lags=np.array([1,2],dtype=int); reference=0
""",
            "call": call,
            "gold_call": gold_call,
        },
        {
            "setup": """import numpy as np
bias=np.array([[0.1,-0.2,0.4,0.0]]); dbias=np.arange(24,dtype=float).reshape(1,4,3,2)/100.0
counts=np.array([4.0]); beta=1.3; offsets=np.array([0.0])
traces=np.array([[0.0,0.2,0.5,0.9],[0.1,0.3,0.4,0.8],[0.2,0.25,0.55,0.85],[0.05,0.2,0.6,0.95]])
lags=np.array([1,2],dtype=int); reference=0
""",
            "call": call,
            "gold_call": gold_call,
        },
        {
            "setup": """import numpy as np
bias=np.array([[0.1,-0.2,0.4],[0.3,0.0,-0.1]]); dbias=np.ones((2,3,3,1))*0.2
counts=np.array([3.0,3.0]); beta=1.2; offsets=np.array([0.0,-0.030285149060821453])
traces=np.full((3,4),0.5); lags=np.array([1,2],dtype=int); reference=0
""",
            "call": call,
            "gold_call": gold_call,
        },
        {
            "setup": """import numpy as np
bias=np.array([[0.1,0.2],[0.3,0.4]]); dbias=np.ones((2,2,3,1))*0.1
counts=np.array([2.0,2.0]); beta=1.0; offsets=np.array([0.0,0.1])
traces=np.array([[0.1,0.2],[0.3,0.4]]); lags=np.array([1.0]); reference=0
def model():
    try: wham_geometry_gradient(bias.copy(),dbias.copy(),counts.copy(),beta,offsets.copy(),traces.copy(),lags.copy(),reference); return 0
    except ValueError: return 1
    except Exception: return 2
def oracle():
    try: _oracle_wham_geometry_gradient(bias.copy(),dbias.copy(),counts.copy(),beta,offsets.copy(),traces.copy(),lags.copy(),reference); return 0
    except ValueError: return 1
    except Exception: return 2
""",
            "call": "model()",
            "gold_call": "oracle()",
        },
        {
            "setup": """import numpy as np
bias=np.array([[0.1,0.2],[0.3,0.4]]); dbias=np.ones((2,2,2,1))*0.1
counts=np.array([2.0,2.0]); beta=1.0; offsets=np.array([0.0,0.1])
traces=np.array([[0.1,0.2],[0.3,0.4]]); lags=np.array([1],dtype=int); reference=0
dbias=dbias[:,:,:1,:]
def model():
    try: wham_geometry_gradient(bias.copy(),dbias.copy(),counts.copy(),beta,offsets.copy(),traces.copy(),lags.copy(),reference); return 0
    except ValueError: return 1
    except Exception: return 2
def oracle():
    try: _oracle_wham_geometry_gradient(bias.copy(),dbias.copy(),counts.copy(),beta,offsets.copy(),traces.copy(),lags.copy(),reference); return 0
    except ValueError: return 1
    except Exception: return 2
""",
            "call": "model()",
            "gold_call": "oracle()",
        },
    ]
