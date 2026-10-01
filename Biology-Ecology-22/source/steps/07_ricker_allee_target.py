"""
Return the target biomass (kg) for the next census predicted from the current female biomass (kg) by the discrete-time Ricker model with an Allee effect in the form the source adopts (its Eq. 2), with intrinsic growth rate gamma, Allee threshold allee_threshold and carrying capacity carrying_capacity (both in kg), applied to biomass rather than to counts. The target equals the current biomass when the current biomass lies exactly at the Allee threshold or at the carrying capacity, exceeds it in between and falls below it outside that range.

Ricker recruitment is the standard density-dependent spawner-recruit relation of fisheries science; adding an Allee threshold makes the empty state stable below the threshold and the carrying capacity stable above it, which is what decides whether a small founding population of an invader establishes. Applying it to spawning biomass rather than to spawner counts reflects that fecundity scales with body size.

Returns
-------
float, the target biomass for the next census in kilograms.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def ricker_allee_target(biomass: float, gamma: float, allee_threshold: float,
                        carrying_capacity: float) -> float:
    """Return the target biomass (kg) for the next census predicted from the current female biomass (kg) by the discrete-time Ricker model with an Allee effect in the form the source adopts (its Eq. 2), with intrinsic growth rate gamma, Allee threshold allee_threshold and carrying capacity carrying_capacity (both in kg), applied to biomass rather than to counts. The target equals the current biomass when the current biomass lies exactly at the Allee threshold or at the carrying capacity, exceeds it in between and falls below it outside that range.

    Parameters
    ----------
    biomass : float
        Current total female biomass in kg, nonnegative.
    gamma : float
        Positive intrinsic growth rate of the Ricker model.
    allee_threshold : float
        Allee threshold C in kg, with 0 < C < K.
    carrying_capacity : float
        Carrying capacity K in kg, above the Allee threshold.

    Returns
    -------
    target : float
        Target biomass for the next census in kg, as a native Python float.

    Raises
    ------
    ValueError
        If any input is not finite, biomass is negative, gamma is not positive, or the Allee threshold and carrying capacity do not satisfy 0 < C < K.
    """
    return target

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_ricker_allee_target(biomass: float, gamma: float, allee_threshold: float,
                                carrying_capacity: float) -> float:
    """Eq 2 on biomass: b_{t+1} = b exp( gamma (1 - b/K) (b - C)/K )."""
    b, g, C, K = float(biomass), float(gamma), float(allee_threshold), float(carrying_capacity)
    if not all(np.isfinite([b, g, C, K])):
        raise ValueError("all inputs must be finite")
    if b < 0.0:
        raise ValueError("biomass must be nonnegative")
    if not (0.0 < C < K):
        raise ValueError("the Allee threshold must satisfy 0 < C < K")
    if g <= 0.0:
        raise ValueError("gamma must be positive")
    # BOTH factors are scaled by K: (1 - b/K) and (b - C)/K. The authors' code writes
    # gamma*(1 - b/K)*(b - C) with gamma = 1.837/K, which is the same expression.
    return b * np.exp(g * (1.0 - b / K) * ((b - C) / K))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return the list of test case specifications."""
    return [
        {
            "setup": "import numpy as np\nw_adult = float(_oracle_length_to_weight(np.array([0.867]))[0])\nbiomass, gamma, allee_threshold, carrying_capacity = 223.792, 1.837, 10.0 * w_adult, 500.0 * w_adult\n",
            "call": "ricker_allee_target(biomass, gamma, allee_threshold, carrying_capacity)",
            "gold_call": "_oracle_ricker_allee_target(biomass, gamma, allee_threshold, carrying_capacity)",
        },
        {
            "setup": "import numpy as np\nw_adult = float(_oracle_length_to_weight(np.array([0.867]))[0])\nbiomass, gamma, allee_threshold, carrying_capacity = 10.0 * w_adult, 1.837, 10.0 * w_adult, 500.0 * w_adult\n",
            "call": "ricker_allee_target(biomass, gamma, allee_threshold, carrying_capacity)",
            "gold_call": "_oracle_ricker_allee_target(biomass, gamma, allee_threshold, carrying_capacity)",
        },
        {
            "setup": "import numpy as np\nw_adult = float(_oracle_length_to_weight(np.array([0.867]))[0])\nbiomass, gamma, allee_threshold, carrying_capacity = 5000.0, 1.837, 10.0 * w_adult, 500.0 * w_adult\n",
            "call": "ricker_allee_target(biomass, gamma, allee_threshold, carrying_capacity)",
            "gold_call": "_oracle_ricker_allee_target(biomass, gamma, allee_threshold, carrying_capacity)",
        },
        {
            "setup": "import numpy as np\ndef _fx_length_to_weight(lengths, intercept=1.02, slope=3.02):\n    zz = np.asarray(lengths, dtype=np.float64)\n    return 10.0 ** (float(intercept) + float(slope) * np.log10(zz))\nw_adult = float(_fx_length_to_weight(np.array([0.867]))[0])\nbiomass, gamma, allee_threshold, carrying_capacity = 223.792, 1.837, 500.0 * w_adult, 10.0 * w_adult\ndef run_model():\n    try:\n        ricker_allee_target(biomass, gamma, allee_threshold, carrying_capacity)\n        return 0\n    except ValueError:\n        return 1\ndef run_oracle():\n    try:\n        _oracle_ricker_allee_target(biomass, gamma, allee_threshold, carrying_capacity)\n        return 0\n    except ValueError:\n        return 1\n",
            "call": "run_model()",
            "gold_call": "run_oracle()",
        },
    ]
