"""
Find the juvenile-consumption rate beyond which prey and predators can no longer coexist in any steady state.

As prey consume more juvenile predators, coexistence steady states can disappear. Past that point no steady state with predators remains.

Returns
-------
float, the smallest g in [0, 1] above which no coexistence steady state exists (1.0 if one exists at g = 1)
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def fold_threshold(tau_star: float, params: dict) -> float:
    """Smallest juvenile-consumption rate above which no coexistence steady state exists.

    Parameters
    ----------
    tau_star : float
        Maturation age, with 1 <= tau_star <= 2.
    params : dict
        Model constants, with the same required keys as
        ``net_reproductive_number``.

    Returns
    -------
    float
        The smallest ``g`` in [0, 1] such that the model has no coexistence
        steady state for any higher juvenile-consumption rate. Return 1.0 if
        coexistence still exists at g = 1. Accurate to an absolute error of
        1e-8.

    Raises
    ------
    ValueError
        If ``tau_star`` is not finite or is outside [1, 2], if ``params`` is
        not a dictionary containing every required finite numeric parameter,
        or if ``L`` or ``nu`` is not positive and finite.
    """
    return g_fold

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
from scipy.optimize import brentq, minimize_scalar


def _oracle_fold_threshold(
    tau_star: float,
    params: dict,
) -> float:
    """Return the validated fold threshold for coexistence."""
    if not isinstance(params, dict):
        raise ValueError("params must be a dict")

    _pp_validate_tau_params(tau_star, params)

    p = params

    if len(
        _oracle_coexistence_steady_states(
            tau_star,
            1.0,
            p,
        )
    ) > 0:
        return 1.0

    quad = _pp_quad(tau_star, p)
    x_max = p["r"] / p["a"]

    while (
        x_max < 1e7
        and (p["k"] * x_max + 2.0 * p["b_p"])
        * p["L"]
        * np.exp(-x_max * tau_star)
        >= 1.0
    ):
        x_max *= 2.0

    xs = np.unique(
        np.concatenate(
            [
                np.linspace(1e-6, p["r"] / p["a"], 301),
                np.geomspace(p["r"] / p["a"], x_max, 120),
            ]
        )
    )

    def peak_excess(g):
        values = np.array(
            [
                _pp_reproduction_on(
                    quad,
                    x,
                    tau_star,
                    g,
                    p,
                )
                for x in xs
            ]
        )

        index = int(np.argmax(values))
        result = minimize_scalar(
            lambda x: -_pp_reproduction_on(
                quad,
                x,
                tau_star,
                g,
                p,
            ),
            bounds=(
                xs[max(index - 1, 0)],
                xs[min(index + 1, len(xs) - 1)],
            ),
            method="bounded",
            options={"xatol": 1e-10},
        )

        return max(values[index], -result.fun) - 1.0

    return float(
        brentq(
            peak_excess,
            0.0,
            1.0,
            xtol=1e-10,
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
        # --- Normal: coexistence states vanish inside [0, 1] ---
        {
            "setup": params,
            "call": "fold_threshold(2.0, dict(params))",
            "gold_call": "_oracle_fold_threshold(2.0, dict(params))",
            "tol": 1e-8,
        },
        # --- Normal: a different maturation age, vanishing at a larger rate ---
        {
            "setup": params,
            "call": "fold_threshold(1.8, dict(params))",
            "gold_call": "_oracle_fold_threshold(1.8, dict(params))",
            "tol": 1e-8,
        },
        # --- Boundary: coexistence states still exist at g = 1 ---
        {
            "setup": params,
            "call": "fold_threshold(1.5, dict(params))",
            "gold_call": "_oracle_fold_threshold(1.5, dict(params))",
            "tol": 1e-8,
        },
    ]
