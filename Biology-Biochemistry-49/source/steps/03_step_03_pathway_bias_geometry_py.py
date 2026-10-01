"""
Combine the progress and orthogonal coordinates with the fixed bias convention to produce pathway biases and their geometry Jacobians

The converged bias subtracts the one-dimensional profile and adds the orthogonal restraint. Applying the chain rule to both contributions converts the path-coordinate Jacobians into the bias derivatives that enter the implicit WHAM response.

Returns
-------
tuple[np.ndarray, np.ndarray]: pathway bias values of shape (w, n) and local bias geometry derivatives of shape (w, n, m, d), both float64 arrays.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def pathway_bias_geometry(
    progress_values: "np.ndarray",
    progress_jacobian: "np.ndarray",
    orthogonal_values: "np.ndarray",
    orthogonal_jacobian: "np.ndarray",
    linear_coefficients: "np.ndarray",
    curvature_coefficients: "np.ndarray",
    orthogonal_scales: "np.ndarray",
) -> tuple["np.ndarray", "np.ndarray"]:
    r"""Return converged pathway biases and local geometry derivatives.

    Parameters
    ----------
    progress_values : np.ndarray
        Finite path progress values with shape ``(w, n)``.
    progress_jacobian : np.ndarray
        Finite local derivatives with shape ``(w, n, m, d)``.
    orthogonal_values : np.ndarray
        Finite orthogonal variables with shape ``(w, n)``.
    orthogonal_jacobian : np.ndarray
        Finite local derivatives with shape ``(w, n, m, d)``.
    linear_coefficients : np.ndarray
        Finite profile coefficients with shape ``(w,)``.
    curvature_coefficients : np.ndarray
        Finite curvature coefficients with shape ``(w,)``.
    orthogonal_scales : np.ndarray
        Finite nonnegative restraint scales with shape ``(w,)``.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Float64 bias values with shape ``(w, n)`` and local geometry
        derivatives with shape ``(w, n, m, d)``.

    Raises
    ------
    ValueError
        If shapes are incompatible, values are nonfinite, progress values lie
        outside ``[0,1]``, or an orthogonal scale is negative.

    Notes
    -----
    Use the supplied converged-bias convention with the profile subtracted and
    the quadratic orthogonal restraint added.
    """
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_pathway_bias_geometry(
    progress_values: "np.ndarray",
    progress_jacobian: "np.ndarray",
    orthogonal_values: "np.ndarray",
    orthogonal_jacobian: "np.ndarray",
    linear_coefficients: "np.ndarray",
    curvature_coefficients: "np.ndarray",
    orthogonal_scales: "np.ndarray",
) -> tuple["np.ndarray", "np.ndarray"]:
    progress_values = np.asarray(progress_values, dtype=np.float64)
    progress_jacobian = np.asarray(progress_jacobian, dtype=np.float64)
    orthogonal_values = np.asarray(orthogonal_values, dtype=np.float64)
    orthogonal_jacobian = np.asarray(orthogonal_jacobian, dtype=np.float64)
    linear_coefficients = np.asarray(linear_coefficients, dtype=np.float64)
    curvature_coefficients = np.asarray(curvature_coefficients, dtype=np.float64)
    orthogonal_scales = np.asarray(orthogonal_scales, dtype=np.float64)
    if progress_values.ndim != 2 or progress_values.shape[0] == 0 or progress_values.shape[1] == 0:
        raise ValueError("progress_values must have shape (w,n) with w,n > 0")
    w, n = progress_values.shape
    if orthogonal_values.shape != (w, n):
        raise ValueError("orthogonal_values must have shape (w,n)")
    if progress_jacobian.ndim != 4 or progress_jacobian.shape[:2] != (w, n):
        raise ValueError("progress_jacobian must have shape (w,n,m,d)")
    if progress_jacobian.shape[2] < 2 or progress_jacobian.shape[3] < 1:
        raise ValueError("the Jacobian requires at least two images and one dimension")
    if orthogonal_jacobian.shape != progress_jacobian.shape:
        raise ValueError("coordinate Jacobians must have matching shapes")
    for values in (linear_coefficients, curvature_coefficients, orthogonal_scales):
        if values.shape != (w,):
            raise ValueError("coefficient vectors must have shape (w,)")
    arrays = (
        progress_values, progress_jacobian, orthogonal_values,
        orthogonal_jacobian, linear_coefficients, curvature_coefficients,
        orthogonal_scales,
    )
    if not all(np.all(np.isfinite(values)) for values in arrays):
        raise ValueError("all bias inputs must be finite")
    if np.any(progress_values < 0.0) or np.any(progress_values > 1.0):
        raise ValueError("progress values must lie in [0,1]")
    if np.any(orthogonal_scales < 0.0):
        raise ValueError("orthogonal scales must be nonnegative")
    profile = (
        linear_coefficients[:, None] * progress_values
        + curvature_coefficients[:, None] * progress_values * (1.0 - progress_values)
    )
    bias_values = -profile + orthogonal_scales[:, None] * orthogonal_values * orthogonal_values
    profile_derivative = (
        linear_coefficients[:, None]
        + curvature_coefficients[:, None] * (1.0 - 2.0 * progress_values)
    )
    bias_jacobian = (
        -profile_derivative[:, :, None, None] * progress_jacobian
        + 2.0 * orthogonal_scales[:, None, None, None]
        * orthogonal_values[:, :, None, None] * orthogonal_jacobian
    )
    if not np.all(np.isfinite(bias_values)) or not np.all(np.isfinite(bias_jacobian)):
        raise ValueError("bias outputs must be finite")
    return bias_values, bias_jacobian

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases() -> list[dict[str, str]]:
    """Return normal, zero-profile, one-pathway, and invalid-input cases."""
    call = "(lambda r:np.concatenate([r[0].ravel(),r[1].ravel()]))(pathway_bias_geometry(s.copy(),ds.copy(),z.copy(),dz.copy(),linear.copy(),curvature.copy(),scales.copy()))"
    gold_call = "(lambda r:np.concatenate([r[0].ravel(),r[1].ravel()]))(_oracle_pathway_bias_geometry(s.copy(),ds.copy(),z.copy(),dz.copy(),linear.copy(),curvature.copy(),scales.copy()))"
    return [
        {
            "setup": """import numpy as np
s=np.array([[0.2,0.7],[0.4,0.8]]); z=np.array([[0.1,0.3],[0.2,0.4]])
ds=np.arange(24,dtype=float).reshape(2,2,3,2)/100.0
dz=np.arange(24,48,dtype=float).reshape(2,2,3,2)/120.0
linear=np.array([0.5,-0.2]); curvature=np.array([0.8,1.1]); scales=np.array([1.5,2.0])
""",
            "call": call,
            "gold_call": gold_call,
        },
        {
            "setup": """import numpy as np
s=np.array([[0.0,1.0]]); z=np.zeros((1,2)); ds=np.ones((1,2,2,1)); dz=np.zeros((1,2,2,1))
linear=np.zeros(1); curvature=np.zeros(1); scales=np.zeros(1)
""",
            "call": call,
            "gold_call": gold_call,
        },
        {
            "setup": """import numpy as np
s=np.array([[0.5]]); z=np.array([[-0.2]]); ds=np.array([[[[0.1],[0.2],[0.3]]]]); dz=np.array([[[[0.4],[0.5],[0.6]]]])
linear=np.array([0.7]); curvature=np.array([1.2]); scales=np.array([2.0])
""",
            "call": call,
            "gold_call": gold_call,
        },
        {
            "setup": """import numpy as np
s=np.array([[0.5]]); z=np.array([[0.2]]); ds=np.ones((1,1,2,1)); dz=np.ones((1,1,2,1))
linear=np.array([0.7]); curvature=np.array([1.2]); scales=np.array([-1.0])
def model():
    try: pathway_bias_geometry(s.copy(),ds.copy(),z.copy(),dz.copy(),linear.copy(),curvature.copy(),scales.copy()); return 0
    except ValueError: return 1
    except Exception: return 2
def oracle():
    try: _oracle_pathway_bias_geometry(s.copy(),ds.copy(),z.copy(),dz.copy(),linear.copy(),curvature.copy(),scales.copy()); return 0
    except ValueError: return 1
    except Exception: return 2
""",
            "call": "model()",
            "gold_call": "oracle()",
        },
        {
            "setup": """import numpy as np
s=np.array([[0.5]]); z=np.array([[0.2]]); ds=np.ones((1,1,2,1)); dz=np.ones((1,1,3,1))
linear=np.array([0.7]); curvature=np.array([1.2]); scales=np.array([1.0])
def model():
    try: pathway_bias_geometry(s.copy(),ds.copy(),z.copy(),dz.copy(),linear.copy(),curvature.copy(),scales.copy()); return 0
    except ValueError: return 1
    except Exception: return 2
def oracle():
    try: _oracle_pathway_bias_geometry(s.copy(),ds.copy(),z.copy(),dz.copy(),linear.copy(),curvature.copy(),scales.copy()); return 0
    except ValueError: return 1
    except Exception: return 2
""",
            "call": "model()",
            "gold_call": "oracle()",
        },
    ]
