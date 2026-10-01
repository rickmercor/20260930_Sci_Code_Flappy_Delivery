"""
Find the juvenile-consumption rate above which predators can no longer invade a prey-only community.

Without predators the prey settles at its carrying capacity x = r / a. A small predator population introduced there either grows or dies out. Because prey eat juvenile predators, stronger juvenile consumption makes invasion harder, and beyond a threshold the prey-only community resists invasion.

Returns
-------
float, the value of g at which the prey-only steady state changes linear stability
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def invasion_threshold(tau_star: float, params: dict) -> float:
    '''Juvenile-consumption rate at which the prey-only steady state changes linear stability.

    Parameters
    ----------
    tau_star : float
        Maturation age, with 1 <= tau_star <= 2.
    params : dict
        Model constants (keys as in net_reproductive_number).

    Returns
    -------
    g_inv : float
        The value of g in [0, 1] such that the prey-only steady state x = r / a of the model
        defined in net_reproductive_number is linearly unstable for g < g_inv and linearly
        stable for g > g_inv (exactly one such value exists for these maturation ages).
        Accurate to an absolute error of 1e-8.

    Raises
    ------
    ValueError
        If tau_star is not a finite number satisfying 1 <= tau_star <= 2, if params is not a
        dict containing finite numeric values for every required key, or if L or nu in params
        is not finite and positive.
    '''
    return g_inv

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


def _oracle_invasion_threshold(
    tau_star: float,
    params: dict,
) -> float:
    """Return the validated juvenile-consumption invasion threshold."""
    if not isinstance(params, dict):
        raise ValueError("params must be a dict")

    _pp_validate_tau_params(tau_star, params)

    x_free = params["r"] / params["a"]
    quad = _pp_quad(tau_star, params)

    def invasion_residual(g):
        return (
            _pp_reproduction_on(
                quad,
                x_free,
                tau_star,
                g,
                params,
            )
            - 1.0
        )

    return float(
        brentq(
            invasion_residual,
            0.0,
            1.0,
            xtol=1e-13,
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
        # --- Normal: late maturation, low threshold ---
        {
            "setup": params,
            "call": "invasion_threshold(2.0, dict(params))",
            "gold_call": "_oracle_invasion_threshold(2.0, dict(params))",
            "tol": 1e-8,
        },
        # --- Normal: intermediate maturation age ---
        {
            "setup": params,
            "call": "invasion_threshold(1.5, dict(params))",
            "gold_call": "_oracle_invasion_threshold(1.5, dict(params))",
            "tol": 1e-8,
        },
        # --- Boundary: earliest maturation age in range, threshold close to 1 ---
        {
            "setup": params,
            "call": "invasion_threshold(1.0, dict(params))",
            "gold_call": "_oracle_invasion_threshold(1.0, dict(params))",
            "tol": 1e-8,
        },
    ]
