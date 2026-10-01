"""
Find the juvenile-consumption rate at which the low-prey coexistence steady state switches from oscillatory instability to stability.

When juvenile predators are rarely eaten, a long maturation delay destabilises coexistence and populations cycle. Stronger consumption of juveniles damps these cycles until the coexistence steady state with the smallest prey population becomes stable.

Returns
-------
float, the value of g at which the lowest-prey coexistence steady state changes linear stability
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def stability_switch_threshold(tau_star: float, params: dict) -> float:
    '''Juvenile-consumption rate at which the lowest-prey coexistence steady state changes linear stability.

    Parameters
    ----------
    tau_star : float
        Maturation age, with 1 <= tau_star <= 2.
    params : dict
        Model constants (keys as in net_reproductive_number).

    Returns
    -------
    g_switch : float
        For these maturation ages, the coexistence steady state with the smallest prey size of
        the model defined in net_reproductive_number exists for every g in [0, g_fold), with g_fold
        as returned by fold_threshold, and changes linear stability at exactly one value of g in
        that interval: unstable below it and stable above it. Return that value, accurate to an
        absolute error of 1e-7.

    Raises
    ------
    ValueError
        If tau_star is not a finite number satisfying 1 <= tau_star <= 2, if params is not a
        dict containing finite numeric values for every required key, or if L or nu in params
        is not finite and positive.
    '''
    return g_switch

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
from scipy.optimize import brentq


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


def _oracle_stability_switch_threshold(
    tau_star: float,
    params: dict,
) -> float:
    """Return the validated stability-switch threshold."""
    if not isinstance(params, dict):
        raise ValueError("params must be a dict")

    _pp_validate_tau_params(tau_star, params)

    p = params
    upper = _oracle_fold_threshold(tau_star, p)
    upper = 1.0 if upper >= 1.0 else upper - 1e-4

    tracked = {}

    def growth_rate(g):
        lower_state = _oracle_coexistence_steady_states(
            tau_star,
            g,
            p,
        )[0]

        if tracked:
            nearest_g = min(
                tracked,
                key=lambda known_g: abs(known_g - g),
            )

            def characteristic(values):
                return _pp_char_fun(
                    values,
                    lower_state,
                    tau_star,
                    g,
                    p,
                )

            root = complex(
                _pp_newton_many(
                    characteristic,
                    np.array([tracked[nearest_g]]),
                )[0]
            )

            if (
                abs(characteristic(np.array([root]))[0]) < 1e-10
                and abs(root - tracked[nearest_g]) < 0.05
            ):
                tracked[g] = root
                return root.real

        root = _oracle_rightmost_characteristic_root(
            lower_state,
            tau_star,
            g,
            p,
        )
        tracked[g] = root
        return root.real

    growth_rate(0.0)
    growth_rate(upper)

    return float(
        brentq(
            growth_rate,
            0.0,
            upper,
            xtol=1e-9,
        )
    )

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    params = """params = dict(r=0.4, a=0.1, k=0.3, b=0.8, s=0.2, zeta=10.0, mu_M=1.0, rho=5.0,
              d_p=0.4, b_p=0.05, b_ep=0.1, d_ep=0.1, L=30.0, nu=100.0)
"""
    return [
        # --- Normal: tau* = 1 ---
        {
            "setup": params,
            "call": "stability_switch_threshold(1.0, dict(params))",
            "gold_call": "_oracle_stability_switch_threshold(1.0, dict(params))",
            "tol": 1e-7,
        },
        # --- Normal: tau* = 1.5, no loss of coexistence states before g = 1 ---
        {
            "setup": params,
            "call": "stability_switch_threshold(1.5, dict(params))",
            "gold_call": "_oracle_stability_switch_threshold(1.5, dict(params))",
            "tol": 1e-7,
        },
        # --- Boundary: tau* = 2, where coexistence states vanish inside [0, 1] ---
        {
            "setup": params,
            "call": "stability_switch_threshold(2.0, dict(params))",
            "gold_call": "_oracle_stability_switch_threshold(2.0, dict(params))",
            "tol": 1e-7,
        },
    ]
