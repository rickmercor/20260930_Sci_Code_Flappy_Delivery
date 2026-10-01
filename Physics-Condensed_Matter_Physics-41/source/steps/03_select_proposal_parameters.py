"""
Select the Gaussian proposal mean and variance from the current bead’s membership in the closed harmonic domain.

The mixed sampler uses the harmonic bridge when the current bead belongs to the closed domain $|x-x_\star|\le d$ and the free bridge otherwise. Equality is inside the harmonic domain. This step returns only the selected numerical mean and variance; the proposal-family label remains implicit in that pair.

Returns
-------
np.ndarray, a length-2 float array [selected_mean, selected_variance]
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import math

from numbers import Real

import numpy as np

def _finite_float(name: str, value: float) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real scalar")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result

def select_proposal_parameters(
    current: float,
    minimum: float,
    domain_radius: float,
    harmonic_mean: float,
    harmonic_variance: float,
    free_mean: float,
    free_variance: float,
) -> np.ndarray:
    """Select the state-dependent Gaussian proposal parameters.

    Parameters
    ----------
    current : float
        Current bead position.
    minimum : float
        Center of the harmonic domain.
    domain_radius : float
        Nonnegative radius of the closed harmonic domain.
    harmonic_mean, harmonic_variance : float
        Mean and positive variance of the harmonic bridge.
    free_mean, free_variance : float
        Mean and positive variance of the free bridge.

    Returns
    -------
    selected : numpy.ndarray
        Length-2 float array ``[selected_mean, selected_variance]``.

    Raises
    ------
    ValueError
        If any scalar input is not a finite real value; if
        ``domain_radius`` is negative; or if ``harmonic_variance`` or
        ``free_variance`` is not strictly positive.
    """
    return np.full(2, np.nan, dtype=float)

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_select_proposal_parameters(
    current: float,
    minimum: float,
    domain_radius: float,
    harmonic_mean: float,
    harmonic_variance: float,
    free_mean: float,
    free_variance: float,
) -> np.ndarray:
    """Reference implementation with function-local dependencies."""
    import math
    import numbers

    import numpy as np

    named_values = (
        ("current", current),
        ("minimum", minimum),
        ("domain_radius", domain_radius),
        ("harmonic_mean", harmonic_mean),
        ("harmonic_variance", harmonic_variance),
        ("free_mean", free_mean),
        ("free_variance", free_variance),
    )
    converted: dict[str, float] = {}
    for name, value in named_values:
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Real):
            raise ValueError(f"{name} must be a finite real scalar")
        value_f = float(value)
        if not math.isfinite(value_f):
            raise ValueError(f"{name} must be finite")
        converted[name] = value_f

    current_f = converted["current"]
    minimum_f = converted["minimum"]
    radius_f = converted["domain_radius"]
    harmonic_mean_f = converted["harmonic_mean"]
    harmonic_variance_f = converted["harmonic_variance"]
    free_mean_f = converted["free_mean"]
    free_variance_f = converted["free_variance"]
    if radius_f < 0.0:
        raise ValueError("domain_radius must be >= 0")
    if harmonic_variance_f <= 0.0 or free_variance_f <= 0.0:
        raise ValueError("proposal variances must be > 0")

    if abs(current_f - minimum_f) <= radius_f:
        return np.array([harmonic_mean_f, harmonic_variance_f], dtype=float)
    return np.array([free_mean_f, free_variance_f], dtype=float)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return normal, boundary, edge, and invalid test specifications."""
    return [
        {
            "setup": """current = 0.26\nminimum = -0.15\ndomain_radius = 0.44\nharmonic_mean = 0.074\nharmonic_variance = 0.099\nfree_mean = 0.125\nfree_variance = 0.113""",
            "call": "select_proposal_parameters(current, minimum, domain_radius, harmonic_mean, harmonic_variance, free_mean, free_variance)",
            "gold_call": "_oracle_select_proposal_parameters(current, minimum, domain_radius, harmonic_mean, harmonic_variance, free_mean, free_variance)",
        },
        {
            "setup": """current = 0.29\nminimum = -0.15\ndomain_radius = 0.44\nharmonic_mean = -1.0\nharmonic_variance = 0.2\nfree_mean = 2.0\nfree_variance = 0.3""",
            "call": "select_proposal_parameters(current, minimum, domain_radius, harmonic_mean, harmonic_variance, free_mean, free_variance)",
            "gold_call": "_oracle_select_proposal_parameters(current, minimum, domain_radius, harmonic_mean, harmonic_variance, free_mean, free_variance)",
        },
        {
            "setup": """current = -0.6\nminimum = -0.15\ndomain_radius = 0.44\nharmonic_mean = -1.0\nharmonic_variance = 0.2\nfree_mean = 2.0\nfree_variance = 0.3""",
            "call": "select_proposal_parameters(current, minimum, domain_radius, harmonic_mean, harmonic_variance, free_mean, free_variance)",
            "gold_call": "_oracle_select_proposal_parameters(current, minimum, domain_radius, harmonic_mean, harmonic_variance, free_mean, free_variance)",
        },
        {
            "setup": """current = 0.0\nminimum = 0.0\ndomain_radius = 0.0\nharmonic_mean = 1.0\nharmonic_variance = 0.1\nfree_mean = 2.0\nfree_variance = 0.2""",
            "call": "select_proposal_parameters(current, minimum, domain_radius, harmonic_mean, harmonic_variance, free_mean, free_variance)",
            "gold_call": "_oracle_select_proposal_parameters(current, minimum, domain_radius, harmonic_mean, harmonic_variance, free_mean, free_variance)",
        },
        {
            "setup": """current = 0.0\nminimum = 0.0\ndomain_radius = 1.0\nharmonic_mean = 0.0\nharmonic_variance = -1.0\nfree_mean = 0.0\nfree_variance = 1.0\ndef run_model():\n    try:\n        select_proposal_parameters(current, minimum, domain_radius, harmonic_mean, harmonic_variance, free_mean, free_variance)\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\ndef run_gold():\n    try:\n        _oracle_select_proposal_parameters(current, minimum, domain_radius, harmonic_mean, harmonic_variance, free_mean, free_variance)\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
    ]
