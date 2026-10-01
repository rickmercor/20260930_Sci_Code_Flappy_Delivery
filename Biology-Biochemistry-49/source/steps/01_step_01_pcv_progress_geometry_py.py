"""
Compute the paper-specific soft path progress values and their analytic derivatives with respect to all string-image coordinates.

The progress coordinate assigns every string image a normalized exponential-distance weight. Differentiating this normalized kernel average provides the first geometry-response tensor needed to propagate pathway uncertainty into the reweighted committor observable

Returns
-------
tuple[np.ndarray, np.ndarray]: progress values of shape (w, n) and local progress-coordinate geometry derivatives of shape (w, n, m, d), both float64 arrays.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def pcv_progress_geometry(
    points: "np.ndarray",
    strings: "np.ndarray",
    alpha: float,
) -> tuple["np.ndarray", "np.ndarray"]:
    r"""Return soft progress coordinates and string-geometry derivatives.

    Parameters
    ----------
    points : np.ndarray
        Finite pooled configurations with shape ``(n, d)``.
    strings : np.ndarray
        Finite pathway images with shape ``(w, m, d)``, where ``m >= 2``.
    alpha : float
        Finite strictly positive kernel sharpness.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Float64 progress values with shape ``(w, n)`` and local image-coordinate
        derivatives with shape ``(w, n, m, d)``. The derivative entry
        ``[i,t,m,r]`` is with respect to ``strings[i,m,r]``.

    Raises
    ------
    ValueError
        If arrays have incompatible shapes, contain nonfinite values, there are
        fewer than two images, or alpha is not finite and strictly positive.

    Notes
    -----
    Use fractional image positions from zero to one and a stable normalized
    exponential-distance kernel over all images.
    """
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_pcv_progress_geometry(
    points: "np.ndarray",
    strings: "np.ndarray",
    alpha: float,
) -> tuple["np.ndarray", "np.ndarray"]:
    points = np.asarray(points, dtype=np.float64)
    strings = np.asarray(strings, dtype=np.float64)
    if points.ndim != 2 or points.shape[0] == 0 or points.shape[1] == 0:
        raise ValueError("points must have shape (n,d) with n,d > 0")
    if strings.ndim != 3 or strings.shape[0] == 0 or strings.shape[1] < 2:
        raise ValueError("strings must have shape (w,m,d) with w > 0 and m >= 2")
    if strings.shape[2] != points.shape[1]:
        raise ValueError("points and strings must have the same coordinate dimension")
    if not np.all(np.isfinite(points)) or not np.all(np.isfinite(strings)):
        raise ValueError("points and strings must be finite")
    if not np.isfinite(alpha) or alpha <= 0.0:
        raise ValueError("alpha must be finite and positive")
    difference = strings[:, None, :, :] - points[None, :, None, :]
    squared = np.sum(difference * difference, axis=-1)
    log_kernel = -float(alpha) * squared
    shift = np.max(log_kernel, axis=2, keepdims=True)
    kernel = np.exp(log_kernel - shift)
    rho = kernel / np.sum(kernel, axis=2, keepdims=True)
    fractions = np.linspace(0.0, 1.0, strings.shape[1], dtype=np.float64)
    progress = np.sum(rho * fractions[None, None, :], axis=2)
    jacobian = (
        -2.0 * float(alpha) * rho[:, :, :, None]
        * (fractions[None, None, :, None] - progress[:, :, None, None])
        * difference
    )
    if not np.all(np.isfinite(progress)) or not np.all(np.isfinite(jacobian)):
        raise ValueError("progress outputs must be finite")
    return progress, jacobian

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases() -> list[dict[str, str]]:
    """Return normal, two-image, sharp-kernel, and invalid-input cases."""
    flatten_model = "(lambda r:np.concatenate([r[0].ravel(),r[1].ravel()]))(pcv_progress_geometry(points.copy(),strings.copy(),alpha))"
    flatten_oracle = "(lambda r:np.concatenate([r[0].ravel(),r[1].ravel()]))(_oracle_pcv_progress_geometry(points.copy(),strings.copy(),alpha))"
    return [
        {
            "setup": """import numpy as np
points=np.array([[0.0,0.0],[0.5,0.2]])
strings=np.array([[[-1.0,0.0],[0.0,0.4],[1.0,0.0]],[[0.0,-1.0],[0.2,0.0],[0.0,1.0]]])
alpha=4.0
""",
            "call": flatten_model,
            "gold_call": flatten_oracle,
        },
        {
            "setup": """import numpy as np
points=np.array([[0.25]])
strings=np.array([[[0.0],[1.0]]])
alpha=2.0
""",
            "call": flatten_model,
            "gold_call": flatten_oracle,
        },
        {
            "setup": """import numpy as np
points=np.array([[9.5,0.0]])
strings=np.array([[[0.0,0.0],[9.0,0.0],[10.0,0.0]]])
alpha=500.0
""",
            "call": flatten_model,
            "gold_call": flatten_oracle,
        },
        {
            "setup": """import numpy as np
points=np.array([[0.0,0.0]]); strings=np.array([[[-1.0,0.0],[1.0,0.0]]]); alpha=0.0
def model():
    try: pcv_progress_geometry(points.copy(),strings.copy(),alpha); return 0
    except ValueError: return 1
    except Exception: return 2
def oracle():
    try: _oracle_pcv_progress_geometry(points.copy(),strings.copy(),alpha); return 0
    except ValueError: return 1
    except Exception: return 2
""",
            "call": "model()",
            "gold_call": "oracle()",
        },
        {
            "setup": """import numpy as np
points=np.array([[0.0,0.0]]); strings=np.array([[[-1.0],[1.0]]]); alpha=2.0
def model():
    try: pcv_progress_geometry(points.copy(),strings.copy(),alpha); return 0
    except ValueError: return 1
    except Exception: return 2
def oracle():
    try: _oracle_pcv_progress_geometry(points.copy(),strings.copy(),alpha); return 0
    except ValueError: return 1
    except Exception: return 2
""",
            "call": "model()",
            "gold_call": "oracle()",
        },
    ]
