"""
Find every steady state of the age-structured role-reversal model in which prey and predators coexist.

A steady state is a time-independent solution (x*, u*(tau)). Besides the prey-only state x* = r / a without predators, the model can have one or several coexistence steady states with x* > 0 and a non-negative, not identically zero predator age density. Prey held above their carrying capacity by feeding on juvenile predators are possible, so coexistence states are not confined below r / a. Which long-term outcomes the model allows depends on how many coexistence states exist and where they lie.

Returns
-------
np.ndarray of shape (m, 4), rows [x*, u*(0), y1*, y2*] of all coexistence steady states sorted by increasing x* (m may be 0)
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def coexistence_steady_states(tau_star: float, g: float, params: dict) -> "np.ndarray":
    '''All coexistence steady states of the model defined in net_reproductive_number.

    Parameters
    ----------
    tau_star : float
        Maturation age, 0 < tau_star < L.
    g : float
        Consumption rate of juvenile predators by prey, g >= 0.
    params : dict
        Model constants (keys as in net_reproductive_number).

    Returns
    -------
    states : np.ndarray
        Float array of shape (m, 4), one row [x*, u*(0), y1*, y2*] per coexistence steady state
        with x* > 0 and u*(0) > 0 (prey sizes above r / a included),
        sorted by increasing x*, where u*(0) is the steady newborn density and y1*, y2* are the
        steady juvenile and adult predator numbers; m may be 0. Entries are accurate to a relative
        error of 1e-8.

    Raises
    ------
    ValueError
        If tau_star or g is not a finite number, if g < 0, if params is not a dict containing
        finite numeric values for every required key, if L or nu in params is not finite and
        positive, or if tau_star does not satisfy 0 < tau_star < L.
    '''
    return states

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
from scipy.optimize import brentq, minimize_scalar


def _pp_validate_common(tau_star, g, params):
    required_keys = (
        "r", "a", "k", "b", "s", "zeta", "mu_M", "rho",
        "d_p", "b_p", "b_ep", "d_ep", "L", "nu",
    )

    for name, value in (("tau_star", tau_star), ("g", g)):
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not np.isfinite(value)
        ):
            raise ValueError(f"{name} must be a finite number")

    if g < 0.0:
        raise ValueError("g must be >= 0")

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
            raise ValueError(f"params[{key!r}] must be a finite number")

    if params["L"] <= 0.0:
        raise ValueError("params['L'] must be > 0")
    if params["nu"] <= 0.0:
        raise ValueError("params['nu'] must be > 0")
    if not (0.0 < tau_star < params["L"]):
        raise ValueError("tau_star must satisfy 0 < tau_star < params['L']")


def _oracle_coexistence_steady_states(
    tau_star: float,
    g: float,
    params: dict,
) -> "np.ndarray":
    """Return all validated positive coexistence steady states."""
    if not isinstance(params, dict):
        raise ValueError("params must be a dict")

    _pp_validate_common(tau_star, g, params)

    p = params
    q = _pp_quad(tau_star, p)

    def excess(value):
        return _pp_reproduction_on(
            q,
            value,
            tau_star,
            g,
            p,
        ) - 1.0

    # Search to a prey size beyond which lifetime reproduction is below one.
    x_max = p["r"] / p["a"]
    if g > 0.0:
        while (
            x_max < 1e7
            and (p["k"] * x_max + 2.0 * p["b_p"])
            * p["L"]
            * np.exp(-g * x_max * tau_star)
            >= 1.0
        ):
            x_max *= 2.0
    else:
        # Without juvenile predation, reproduction increases with prey size.
        x_max = 1e3

    xs = np.unique(
        np.concatenate(
            [
                np.linspace(1e-9, p["r"] / p["a"], 201),
                np.geomspace(p["r"] / p["a"], x_max, 150),
            ]
        )
    )

    values = np.array([excess(value) for value in xs])
    brackets = [
        (xs[i], xs[i + 1])
        for i in range(len(xs) - 1)
        if values[i] * values[i + 1] < 0.0
    ]

    # A pair of roots can be hidden in one grid cell near an interior extremum.
    for i in range(1, len(xs) - 1):
        if (
            (values[i] - values[i - 1])
            * (values[i + 1] - values[i])
            < 0.0
        ):
            sign = 1.0 if values[i] > values[i - 1] else -1.0
            result = minimize_scalar(
                lambda value: -sign * excess(value),
                bounds=(xs[i - 1], xs[i + 1]),
                method="bounded",
                options={"xatol": 1e-13},
            )

            extremum_x = result.x
            extremum_value = excess(extremum_x)

            if (
                extremum_value * values[i - 1] < 0.0
                and extremum_value * values[i + 1] < 0.0
                and not any(
                    lower <= extremum_x <= upper
                    for lower, upper in brackets
                )
            ):
                brackets.extend(
                    [
                        (xs[i - 1], extremum_x),
                        (extremum_x, xs[i + 1]),
                    ]
                )

    rows = []
    for lower, upper in sorted(brackets):
        prey = brentq(
            excess,
            lower,
            upper,
            xtol=1e-15,
            rtol=1e-15,
        )

        survival = _pp_survival(
            prey,
            q["T"],
            tau_star,
            g,
            p,
        )

        juvenile_integral = q["W"] @ (survival * q["juv"])
        adult_integral = q["W"] @ (survival * ~q["juv"])

        newborn = (
            p["r"] - p["a"] * prey
        ) / (
            p["b"] * adult_integral
            - p["s"] * juvenile_integral
        )

        if newborn > 0.0:
            rows.append(
                [
                    prey,
                    newborn,
                    newborn * juvenile_integral,
                    newborn * adult_integral,
                ]
            )

    return np.array(rows, dtype=float).reshape(-1, 4)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    params = """params = dict(r=0.4, a=0.1, k=0.3, b=0.8, s=0.2, zeta=10.0, mu_M=1.0, rho=5.0,
              d_p=0.4, b_p=0.05, b_ep=0.1, d_ep=0.1, L=30.0, nu=100.0)
"""
    return [
        # --- Normal: a single coexistence steady state ---
        {
            "setup": params,
            "call": "coexistence_steady_states(1.0, 0.35, dict(params))",
            "gold_call": "_oracle_coexistence_steady_states(1.0, 0.35, dict(params))",
            "tol": 1e-8,
        },
        # --- Normal: two well-separated coexistence steady states ---
        {
            "setup": params,
            "call": "coexistence_steady_states(2.0, 0.6, dict(params))",
            "gold_call": "_oracle_coexistence_steady_states(2.0, 0.6, dict(params))",
            "tol": 1e-8,
        },
        # --- Edge: two coexistence steady states about to merge (prey sizes 0.857 and 0.883) ---
        {
            "setup": params,
            "call": "coexistence_steady_states(2.0, 0.7688, dict(params))",
            "gold_call": "_oracle_coexistence_steady_states(2.0, 0.7688, dict(params))",
            "tol": 1e-8,
        },
        # --- Edge: beyond the merge no coexistence steady state exists; the count is compared ---
        {
            "setup": params,
            "call": "float(len(coexistence_steady_states(2.0, 0.8, dict(params))))",
            "gold_call": "float(len(_oracle_coexistence_steady_states(2.0, 0.8, dict(params))))",
            "tol": 1e-12,
        },
        # --- Edge: weak juvenile predation admits a second state far above the prey carrying
        # capacity r / a = 4, where prey persist by feeding on juveniles (x* about 53.4) ---
        {
            "setup": params,
            "call": "coexistence_steady_states(2.0, 0.05, dict(params))",
            "gold_call": "_oracle_coexistence_steady_states(2.0, 0.05, dict(params))",
            "tol": 1e-8,
        },
    ]
