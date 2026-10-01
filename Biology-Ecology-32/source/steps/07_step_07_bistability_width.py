"""
Measure the range of juvenile-consumption rates over which the role-reversal system has two alternative stable outcomes.

For some consumption rates both the prey-only community and a coexistence steady state are locally stable, and which one the system reaches depends on the initial populations. The length of this range summarises how strongly juvenile predation creates alternative stable states.

Returns
-------
float, the length of the set of g in [0, 1] with a stable prey-only state and a stable coexistence state
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def bistability_width(tau_star: float, params: dict) -> float:
    '''Length of the set of juvenile-consumption rates with two locally stable steady states.

    Parameters
    ----------
    tau_star : float
        Maturation age, with 1 <= tau_star <= 2.
    params : dict
        Model constants (keys as in net_reproductive_number).

    Returns
    -------
    width : float
        Total length of the set of g in [0, 1] for which, in the model defined in
        net_reproductive_number, the prey-only steady state is linearly stable and at least one
        coexistence steady state is linearly stable. Accurate to an absolute error of 1e-7.

    Raises
    ------
    ValueError
        If tau_star is not a finite number satisfying 1 <= tau_star <= 2, if params is not a
        dict containing finite numeric values for every required key, or if L or nu in params
        is not finite and positive.
    '''
    return width

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _pp_validate_tau_params(tau_star, params):
    required_keys = (
        "r", "a", "k", "b", "s", "zeta", "mu_M", "rho",
        "d_p", "b_p", "b_ep", "d_ep", "L", "nu",
    )

    if (
        not isinstance(tau_star, (int, float))
        or isinstance(tau_star, bool)
        or not np.isfinite(tau_star)
    ):
        raise ValueError("tau_star must be a finite number")

    if not (1.0 <= tau_star <= 2.0):
        raise ValueError("tau_star must satisfy 1 <= tau_star <= 2")

    if not isinstance(params, dict):
        raise ValueError("params must be a dict")

    for key in required_keys:
        if key not in params:
            raise ValueError(f"params is missing required key {key!r}")

        value = params[key]
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not np.isfinite(value)
        ):
            raise ValueError(
                f"params[{key!r}] must be a finite number"
            )

    if params["L"] <= 0.0:
        raise ValueError("params['L'] must be > 0")
    if params["nu"] <= 0.0:
        raise ValueError("params['nu'] must be > 0")


def _oracle_bistability_width(
    tau_star: float,
    params: dict,
) -> float:
    """Return the width of the validated bistable interval."""
    if not isinstance(params, dict):
        raise ValueError("params must be a dict")

    _pp_validate_tau_params(tau_star, params)

    invasion = _oracle_invasion_threshold(tau_star, params)
    stability_switch = _oracle_stability_switch_threshold(
        tau_star,
        params,
    )
    fold = _oracle_fold_threshold(tau_star, params)

    lower_edge = max(invasion, stability_switch)

    if fold <= lower_edge:
        return 0.0

    # Verify the physical branch at the midpoint of the proposed bistable
    # interval using the earlier reproduction, coexistence, and root steps.
    probe_g = 0.5 * (lower_edge + fold)

    states = _oracle_coexistence_steady_states(
        tau_star,
        probe_g,
        params,
    )
    if states.shape[0] == 0:
        return 0.0

    lower_state = states[0]

    reproductive_number = _oracle_net_reproductive_number(
        float(lower_state[0]),
        tau_star,
        probe_g,
        params,
    )

    rightmost_root = _oracle_rightmost_characteristic_root(
        lower_state,
        tau_star,
        probe_g,
        params,
    )

    # The selected coexistence state must satisfy the renewal condition and
    # be stable inside the claimed bistable interval.
    if (
        abs(reproductive_number - 1.0) > 1e-6
        or not np.isfinite(rightmost_root.real)
        or rightmost_root.real > 1e-6
    ):
        return 0.0

    return float(fold - lower_edge)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    params = """params = dict(r=0.4, a=0.1, k=0.3, b=0.8, s=0.2, zeta=10.0, mu_M=1.0, rho=5.0,
              d_p=0.4, b_p=0.05, b_ep=0.1, d_ep=0.1, L=30.0, nu=100.0)
"""
    return [
        # --- Normal: lower edge set by the stability switch, upper edge by the loss of coexistence ---
        {
            "setup": params,
            "call": "bistability_width(2.0, dict(params))",
            "gold_call": "_oracle_bistability_width(2.0, dict(params))",
            "tol": 1e-7,
        },
        # --- Normal: lower edge set by the invasion threshold, upper edge by the loss of coexistence ---
        {
            "setup": params,
            "call": "bistability_width(1.8, dict(params))",
            "gold_call": "_oracle_bistability_width(1.8, dict(params))",
            "tol": 1e-7,
        },
        # --- Boundary: coexistence persists to g = 1, so the range ends at 1 ---
        {
            "setup": params,
            "call": "bistability_width(1.0, dict(params))",
            "gold_call": "_oracle_bistability_width(1.0, dict(params))",
            "tol": 1e-7,
        },
    ]
