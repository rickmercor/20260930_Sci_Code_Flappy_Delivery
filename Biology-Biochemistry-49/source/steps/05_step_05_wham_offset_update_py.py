"""
Perform one simultaneous gauge-fixed relative free-energy update from the current WHAM factors.

The pointwise equilibrium distribution and relative free-energy constants are coupled. One simultaneous update evaluates every new offset from the same old vector and then fixes the selected reporting reference to zero.

Returns
-------
np.ndarray: updated gauge-fixed float64 relative offsets of shape (w,).
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def wham_offset_update(
    bias_values: "np.ndarray",
    sample_counts: "np.ndarray",
    beta: float,
    free_energy_offsets: "np.ndarray",
    reference_index: int,
) -> "np.ndarray":
    r"""Perform one simultaneous relative free-energy-offset update.

    Parameters
    ----------
    bias_values : np.ndarray
        Finite bias matrix with shape ``(w, n)``.
    sample_counts : np.ndarray
        Finite positive pathway counts with shape ``(w,)``.
    beta : float
        Finite positive inverse temperature.
    free_energy_offsets : np.ndarray
        Finite current offsets with shape ``(w,)``.
    reference_index : int
        Integer pathway index whose returned offset is fixed to zero.

    Returns
    -------
    np.ndarray
        Float64 updated relative offsets with shape ``(w,)``.

    Raises
    ------
    ValueError
        If an input violates the reweighting-factor contract or the reference
        index is not an in-range integer.

    Notes
    -----
    Evaluate every new offset from the same old offset vector.
    """
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_wham_offset_update(
    bias_values: "np.ndarray",
    sample_counts: "np.ndarray",
    beta: float,
    free_energy_offsets: "np.ndarray",
    reference_index: int,
) -> "np.ndarray":
    bias_values = np.asarray(bias_values, dtype=np.float64)
    if bias_values.ndim != 2 or bias_values.shape[0] == 0 or bias_values.shape[1] == 0:
        raise ValueError("bias_values must have shape (w, n) with w,n > 0")
    if isinstance(reference_index, (bool, np.bool_)) or not isinstance(
        reference_index, (int, np.integer)
    ):
        raise ValueError("reference_index must be an integer")
    reference_index = int(reference_index)
    if reference_index < 0 or reference_index >= bias_values.shape[0]:
        raise ValueError("reference_index is out of range")
    factors = _oracle_binless_reweighting_factors(
        bias_values, sample_counts, beta, free_energy_offsets
    )
    log_terms = -float(beta) * bias_values + np.log(factors)[None, :]
    maximum = np.max(log_terms, axis=1)
    log_sums = maximum + np.log(
        np.sum(np.exp(log_terms - maximum[:, None]), axis=1)
    )
    updated = -(log_sums - log_sums[reference_index]) / float(beta)
    updated[reference_index] = 0.0
    return updated

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases() -> list[dict[str, str]]:
    """Return five normal, reference, gauge, and invalid-input cases."""

    return [
        {
            "setup": """import numpy as np
bias=np.array([[0.1,-0.2,0.4],[0.3,0.0,-0.1],[0.2,0.1,0.5]])
counts=np.array([4.0,6.0,5.0]); beta=1.3; offsets=np.zeros(3); reference=0
""",
            "call": 'wham_offset_update(bias.copy(),counts.copy(),beta,offsets.copy(),reference).tolist()',
            "gold_call": '_oracle_wham_offset_update(bias.copy(),counts.copy(),beta,offsets.copy(),reference).tolist()',
        },
        {
            "setup": """import numpy as np
bias=np.array([[0.1,-0.2,0.4],[0.3,0.0,-0.1],[0.2,0.1,0.5]])
counts=np.array([4.0,6.0,5.0]); beta=1.3; offsets=np.array([0.2,-0.1,0.4]); reference=2
""",
            "call": 'wham_offset_update(bias.copy(),counts.copy(),beta,offsets.copy(),reference).tolist()',
            "gold_call": '_oracle_wham_offset_update(bias.copy(),counts.copy(),beta,offsets.copy(),reference).tolist()',
        },
        {
            "setup": """import numpy as np
bias=np.array([[0.1,-0.2],[0.3,0.0]])
counts=np.array([4.0,6.0]); beta=1.3; offsets=np.array([5.0,5.2]); reference=0
""",
            "call": 'wham_offset_update(bias.copy(),counts.copy(),beta,offsets.copy(),reference).tolist()',
            "gold_call": '_oracle_wham_offset_update(bias.copy(),counts.copy(),beta,offsets.copy(),reference).tolist()',
        },
        {
            "setup": """import numpy as np
bias=np.array([[0.1,0.2]]); counts=np.array([2.0]); offsets=np.zeros(1)
def model():
    try: wham_offset_update(bias.copy(),counts.copy(),1.0,offsets.copy(),1); return 0
    except ValueError: return 1
    except Exception: return 2
def oracle():
    try: _oracle_wham_offset_update(bias.copy(),counts.copy(),1.0,offsets.copy(),1); return 0
    except ValueError: return 1
    except Exception: return 2
""",
            "call": 'model()',
            "gold_call": 'oracle()',
        },
        {
            "setup": """import numpy as np
bias=np.array([0.1,0.2]); counts=np.array([2.0]); offsets=np.zeros(1)
def model():
    try: wham_offset_update(bias.copy(),counts.copy(),1.0,offsets.copy(),0); return 0
    except ValueError: return 1
    except Exception: return 2
def oracle():
    try: _oracle_wham_offset_update(bias.copy(),counts.copy(),1.0,offsets.copy(),0); return 0
    except ValueError: return 1
    except Exception: return 2
""",
            "call": 'model()',
            "gold_call": 'oracle()',
        },
    ]
