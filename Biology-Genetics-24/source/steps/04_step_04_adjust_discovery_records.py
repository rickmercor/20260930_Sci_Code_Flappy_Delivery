"""
Discovery records are selected without using held-out data. Within each split, expression resource, and association method, gene-wise p-values are adjusted by the Benjamini-Hochberg step-up procedure. A record is retained only when its adjusted value is strictly below the FDR threshold and its absolute effect meets the effect threshold.

Inputs

------

effects: Float array of shape (n_splits, n_resources, n_methods, n_genes).

p_values: Float array with the same shape.

fdr_threshold: Threshold in (0, 1].

effect_threshold: Non-negative absolute-effect threshold.

Returns

-------

fdr_values: Float array with the same shape as effects.

significant: Binary uint8 array with the same shape.

Returns
-------
tuple of float64 FDR values and uint8 significance indicators, both matching the input shape
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def adjust_discovery_records(
    effects: "np.ndarray",
    p_values: "np.ndarray",
    fdr_threshold: float,
    effect_threshold: float,
) -> "tuple[np.ndarray, np.ndarray]":
    """Adjust gene-wise p-values and identify discovery records.

    Parameters
    ----------
    effects : np.ndarray
        Effect array shaped (n_splits, n_resources, n_methods, n_genes).
    p_values : np.ndarray
        Nominal p-values with the same shape as `effects`.
    fdr_threshold : float
        Positive FDR cutoff no greater than one; comparison is strict.
    effect_threshold : float
        Finite non-negative threshold for the absolute effect.

    Raises
    ------
    ValueError
        If the arrays are not matching four-dimensional non-empty arrays,
        effects are non-finite, p-values lie outside [0, 1],
        `fdr_threshold` is not in (0, 1], or `effect_threshold` is negative or
        non-finite.

    Returns
    -------
    fdr_values : np.ndarray
        Benjamini-Hochberg adjusted values with the input shape.
    significant : np.ndarray
        Binary uint8 indicators of retained discovery records.
    """
    return discovery_results  # noqa: F821

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np  # noqa: E402, F811


def _benjamini_hochberg(values):
    """Return stable Benjamini-Hochberg adjusted values."""
    count = values.size
    order = np.argsort(values, kind="mergesort")
    ordered = values[order]
    adjusted = ordered * count / np.arange(1, count + 1, dtype=float)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.minimum(adjusted, 1.0)
    result = np.empty(count, dtype=float)
    result[order] = adjusted
    return result


def _oracle_adjust_discovery_records(
    effects: "np.ndarray",
    p_values: "np.ndarray",
    fdr_threshold: float,
    effect_threshold: float,
) -> "tuple[np.ndarray, np.ndarray]":
    """Reference implementation."""
    effects = np.asarray(effects, dtype=float)
    p_values = np.asarray(p_values, dtype=float)
    if effects.ndim != 4 or p_values.ndim != 4 or effects.shape != p_values.shape:
        raise ValueError("effects and p_values must be matching four-dimensional arrays")
    if any(size == 0 for size in effects.shape):
        raise ValueError("effects and p_values must be non-empty in every dimension")
    if not np.all(np.isfinite(effects)):
        raise ValueError("effects must be finite")
    if not np.all(np.isfinite(p_values)) or np.any(
        (p_values < 0.0) | (p_values > 1.0)
    ):
        raise ValueError("p_values must be finite and lie in [0, 1]")
    if not (
        isinstance(fdr_threshold, (int, float, np.integer, np.floating))
        and np.isfinite(fdr_threshold)
        and 0.0 < float(fdr_threshold) <= 1.0
    ):
        raise ValueError("fdr_threshold must lie in (0, 1]")
    if not (
        isinstance(effect_threshold, (int, float, np.integer, np.floating))
        and np.isfinite(effect_threshold)
        and float(effect_threshold) >= 0.0
    ):
        raise ValueError("effect_threshold must be finite and non-negative")

    fdr_values = np.empty_like(p_values)
    for index in np.ndindex(p_values.shape[:-1]):
        fdr_values[index] = _benjamini_hochberg(p_values[index])
    significant = (
        (fdr_values < float(fdr_threshold))
        & (np.abs(effects) >= float(effect_threshold))
    ).astype(np.uint8)
    return fdr_values, significant

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        {
            "setup": """import numpy as np
effects = np.array([[[[1.2, 0.6, 0.4, -2.0]]]])
p_values = np.array([[[[0.01, 0.04, 0.03, 0.2]]]])
""",
            "call": "[x.tolist() for x in adjust_discovery_records(effects, p_values, 0.1, 0.5)]",
            "gold_call": "[x.tolist() for x in _oracle_adjust_discovery_records(effects, p_values, 0.1, 0.5)]",
        },
        {
            "setup": """import numpy as np
effects = np.array([[[[0.0]]]])
p_values = np.array([[[[0.0]]]])
""",
            "call": "[x.tolist() for x in adjust_discovery_records(effects, p_values, 1.0, 0.0)]",
            "gold_call": "[x.tolist() for x in _oracle_adjust_discovery_records(effects, p_values, 1.0, 0.0)]",
        },
        {
            "setup": """import numpy as np
effects = np.zeros((1, 1, 1, 2))
p_values = np.array([[[[0.2, 1.1]]]])
def run_model():
    try:
        adjust_discovery_records(effects, p_values, 0.1, 0.5)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_oracle():
    try:
        _oracle_adjust_discovery_records(effects, p_values, 0.1, 0.5)
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
