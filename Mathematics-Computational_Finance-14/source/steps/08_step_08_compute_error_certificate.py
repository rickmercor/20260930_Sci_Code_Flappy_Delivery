"""
Form the constant-free a posteriori residual certificate.

The a posteriori estimate for the fully forward compound system controls the

aggregate state, value, and control error through the discretization scale and

the complete joint residual objective. Removing the unknown multiplicative

constant leaves the paper-defined computable certificate.

Inputs

------

loss_summary: Segment losses followed by their validated joint sum.

h: Positive uniform time step.

Returns

-------

certificate: Native float containing the constant-free error certificate.

Returns
-------
float, the constant-free a posteriori certificate as a native Python float
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def compute_error_certificate(loss_summary: np.ndarray, h: float) -> float:
    """Compute the constant-free certificate from the validated objective.

    Parameters
    ----------
    loss_summary : np.ndarray
        Segment losses followed by their joint sum.
    h : float
        Positive finite uniform step size.

    Raises
    ------
    ValueError
        If loss_summary is not a finite nonnegative vector with at least one
        component and a consistent final sum, or h is not finite and positive.

    Returns
    -------
    certificate : float
        Constant-free a posteriori certificate.
    """
    return certificate  # noqa: F821

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np  # noqa: E402, F811


def _oracle_compute_error_certificate(loss_summary: np.ndarray, h: float) -> float:
    """Reference implementation."""
    loss_summary = np.asarray(loss_summary, dtype=float)
    if loss_summary.ndim != 1 or loss_summary.size < 2:
        raise ValueError("loss_summary must contain component losses and their sum")
    if not np.all(np.isfinite(loss_summary)) or np.any(loss_summary < 0):
        raise ValueError("loss_summary must be finite and nonnegative")
    if not np.isclose(loss_summary[-1], np.sum(loss_summary[:-1]), rtol=1e-12, atol=1e-12):
        raise ValueError("the final loss_summary entry must equal the component sum")
    if not np.isfinite(h) or h <= 0:
        raise ValueError("h must be finite and positive")
    return float(h + loss_summary[-1])

# =============================================================================
# TEST CASES
# =============================================================================

import numpy as np  # noqa: E402, F811


def test_cases():
    """Return list of test case specifications."""
    return [
        {
            "setup": "loss_summary = np.array([0.5, 0.25, 0.75])\nh = 0.125\n",
            "call": "float(compute_error_certificate(loss_summary, h))",
            "gold_call": "float(_oracle_compute_error_certificate(loss_summary, h))",
        },
        {
            "setup": "loss_summary = np.array([0.0, 0.0])\nh = 0.25\n",
            "call": "float(compute_error_certificate(loss_summary, h))",
            "gold_call": "float(_oracle_compute_error_certificate(loss_summary, h))",
        },
        {
            "setup": """loss_summary = np.array([0.2, 0.3])
def run_model():
    try:
        compute_error_certificate(loss_summary, 0.1)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_oracle():
    try:
        _oracle_compute_error_certificate(loss_summary, 0.1)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_oracle()",
        },
    ]
