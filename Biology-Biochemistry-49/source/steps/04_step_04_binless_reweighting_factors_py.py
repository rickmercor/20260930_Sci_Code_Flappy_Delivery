"""
Evaluate the configuration-dependent reciprocal WHAM denominator for every pooled sample using stable log-sum-exp arithmetic.

Bin-less WHAM assigns each pooled configuration its own equilibrium factor from all pathway counts, biases, and relative offsets. Stable evaluation preserves the pointwise weighting when exponential terms differ substantially

Returns
-------
np.ndarray: finite positive float64 bin-less WHAM factors of shape (n,).
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def binless_reweighting_factors(
    bias_values: "np.ndarray",
    sample_counts: "np.ndarray",
    beta: float,
    free_energy_offsets: "np.ndarray",
) -> "np.ndarray":
    r"""Evaluate one pointwise bin-less WHAM factor per pooled configuration.

    Parameters
    ----------
    bias_values : np.ndarray
        Finite bias matrix with shape ``(w, n)``.
    sample_counts : np.ndarray
        Finite positive pathway counts with shape ``(w,)``.
    beta : float
        Finite positive inverse temperature.
    free_energy_offsets : np.ndarray
        Finite relative offsets with shape ``(w,)``.

    Returns
    -------
    np.ndarray
        Finite positive float64 factors with shape ``(n,)``.

    Raises
    ------
    ValueError
        If shapes or values violate the contract, counts or beta are not
        positive, or the resulting factors are not finite and positive.

    Notes
    -----
    Evaluate the reciprocal mixture denominator in log space.
    """
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_binless_reweighting_factors(
    bias_values: "np.ndarray",
    sample_counts: "np.ndarray",
    beta: float,
    free_energy_offsets: "np.ndarray",
) -> "np.ndarray":
    bias_values = np.asarray(bias_values, dtype=np.float64)
    sample_counts = np.asarray(sample_counts, dtype=np.float64)
    free_energy_offsets = np.asarray(free_energy_offsets, dtype=np.float64)
    if bias_values.ndim != 2 or bias_values.shape[0] == 0 or bias_values.shape[1] == 0:
        raise ValueError("bias_values must have shape (w, n) with w,n > 0")
    expected = (bias_values.shape[0],)
    if sample_counts.shape != expected or free_energy_offsets.shape != expected:
        raise ValueError("counts and offsets must have shape (w,)")
    if not np.all(np.isfinite(bias_values)) or not np.all(np.isfinite(sample_counts)):
        raise ValueError("bias values and counts must be finite")
    if not np.all(np.isfinite(free_energy_offsets)):
        raise ValueError("offsets must be finite")
    if np.any(sample_counts <= 0.0):
        raise ValueError("sample counts must be positive")
    if not np.isfinite(beta) or beta <= 0.0:
        raise ValueError("beta must be finite and positive")
    log_terms = (
        np.log(sample_counts)[:, None]
        - beta * (bias_values - free_energy_offsets[:, None])
    )
    maximum = np.max(log_terms, axis=0)
    log_denominator = maximum + np.log(
        np.sum(np.exp(log_terms - maximum[None, :]), axis=0)
    )
    result = np.exp(-log_denominator)
    if not np.all(np.isfinite(result)) or np.any(result <= 0.0):
        raise ValueError("reweighting factors are not finite and positive")
    return result

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases() -> list[dict[str, str]]:
    """Return five normal, gauge-shift, stability, and invalid-input cases."""

    return [
        {
            "setup": """import numpy as np
bias=np.array([[0.1,-0.2,0.4],[0.3,0.0,-0.1]])
counts=np.array([4.0,6.0]); beta=1.3; offsets=np.array([0.0,0.2])
""",
            "call": 'np.log(binless_reweighting_factors(bias.copy(),counts.copy(),beta,offsets.copy())).tolist()',
            "gold_call": 'np.log(_oracle_binless_reweighting_factors(bias.copy(),counts.copy(),beta,offsets.copy())).tolist()',
        },
        {
            "setup": """import numpy as np
bias=np.array([[0.1,-0.2],[0.3,0.0]])
counts=np.array([4.0,6.0]); beta=1.3; offsets=np.array([5.0,5.2])
""",
            "call": 'np.log(binless_reweighting_factors(bias.copy(),counts.copy(),beta,offsets.copy())).tolist()',
            "gold_call": 'np.log(_oracle_binless_reweighting_factors(bias.copy(),counts.copy(),beta,offsets.copy())).tolist()',
        },
        {
            "setup": """import numpy as np
bias=np.array([[-710.0,-709.0]], dtype=np.float64)
counts=np.array([2.0]); beta=1.0; offsets=np.array([0.0])
""",
            "call": 'np.log(binless_reweighting_factors(bias.copy(),counts.copy(),beta,offsets.copy())).tolist()',
            "gold_call": 'np.log(_oracle_binless_reweighting_factors(bias.copy(),counts.copy(),beta,offsets.copy())).tolist()',
        },
        {
            "setup": """import numpy as np
bias=np.array([[0.1,0.2],[0.3,0.4]]); counts=np.array([2.0,0.0]); offsets=np.zeros(2)
def model():
    try: binless_reweighting_factors(bias.copy(),counts.copy(),1.0,offsets.copy()); return 0
    except ValueError: return 1
    except Exception: return 2
def oracle():
    try: _oracle_binless_reweighting_factors(bias.copy(),counts.copy(),1.0,offsets.copy()); return 0
    except ValueError: return 1
    except Exception: return 2
""",
            "call": 'model()',
            "gold_call": 'oracle()',
        },
        {
            "setup": """import numpy as np
bias=np.array([[0.1,0.2]]); counts=np.array([2.0]); offsets=np.zeros(1)
def model():
    try: binless_reweighting_factors(bias.copy(),counts.copy(),0.0,offsets.copy()); return 0
    except ValueError: return 1
    except Exception: return 2
def oracle():
    try: _oracle_binless_reweighting_factors(bias.copy(),counts.copy(),0.0,offsets.copy()); return 0
    except ValueError: return 1
    except Exception: return 2
""",
            "call": 'model()',
            "gold_call": 'oracle()',
        },
    ]
