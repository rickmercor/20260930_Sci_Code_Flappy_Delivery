"""
Compute the long-time mean growth velocity of the copy, in nucleotides per second, from the converged cycle x and the site transfer entries Y: form the source's mean dwell time of the polymerase at each site of the period, which accounts for the time spent in both structural states, average it over the period and invert.

The velocity of a processive enzyme along a heterogeneous track is the inverse of the mean time it spends per site, including the time it dwells in states that do not advance the copy and the back-and-forth motion caused by the reverse reaction near stalling.

Returns
-------
float, the mean growth velocity v in nucleotides per second.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def mean_growth_velocity(x: "np.ndarray", Y: "np.ndarray") -> float:
    """Compute the long-time mean growth velocity of the copy, in nucleotides per second, from the converged cycle x and the site transfer entries Y: form the source's mean dwell time of the polymerase at each site of the period, which accounts for the time spent in both structural states, average it over the period and invert.

    Parameters
    ----------
    x : np.ndarray
        Converged cycle x[0..L] of the backward iteration.
    Y : np.ndarray
        Transfer entries of shape (L, 2, 2).

    Returns
    -------
    v : float
        Mean growth velocity in nt/s.

    Raises
    ------
    ValueError
        If Y is not (L, 2, 2), x does not have L + 1 entries, values are not finite, some x_l with l >= 1 is not positive, or some open-state column sum vanishes.
    """
    return v

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_mean_growth_velocity(x: "np.ndarray", Y: "np.ndarray") -> float:
    """Eqs. (40)-(41): v = 1 / < tau_l >, tau_l = (1 / x_l) (1 + R^F_l / R^E_l) averaged over the period."""
    xx = np.asarray(x, dtype=np.float64).ravel()
    YY = np.asarray(Y, dtype=np.float64)
    L = YY.shape[0] if YY.ndim == 3 else 0
    if YY.shape != (L, 2, 2) or L < 1 or xx.shape != (L + 1,):
        raise ValueError("Y must be (L, 2, 2) and x must have L + 1 entries")
    if not np.all(np.isfinite(xx)) or not np.all(np.isfinite(YY)) or np.any(xx[1:] <= 0.0):
        raise ValueError("x and Y must be finite with positive x_l")
    RE = YY[:, :, 0].sum(axis=1)
    RF = YY[:, :, 1].sum(axis=1)
    if np.any(RE <= 0.0):
        raise ValueError("R^E_l must be positive")
    tau = (1.0 / xx[1:]) * (1.0 + RF / RE)
    return float(1.0 / np.mean(tau))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return the list of test case specifications."""
    return [
        {
            "setup": "import numpy as np\nkin = np.array([[[2.07e-2, 170.0, 340.0, 6.2], [4.15e-4, 6500.0, 1.7, 293.0]],\n                [[3.92e-4, 6500.0, 1.7, 311.0], [9.3e-3, 170.0, 340.0, 2.4]]])\ntemplate = np.array([0, 0, 1, 1, 0, 1, 1, 1, 0, 0, 1, 0])\nconc = np.array([2.0e-8, 4.0e-9])\nppi, kp_const = 1.0e-4, 5.6e-3\nrates_m = reduced_transition_rates(kin, conc, ppi, kp_const)\nparams_m = backward_map_parameters(rates_m)\nx_m = backward_iteration(params_m, template)\nY_m = site_transfer_factors(x_m, params_m, rates_m, template)\nrates_o = _oracle_reduced_transition_rates(kin, conc, ppi, kp_const)\nparams_o = _oracle_backward_map_parameters(rates_o)\nx_o = _oracle_backward_iteration(params_o, template)\nY_o = _oracle_site_transfer_factors(x_o, params_o, rates_o, template)\n",
            "call": "mean_growth_velocity(x_m, Y_m)",
            "gold_call": "_oracle_mean_growth_velocity(x_o, Y_o)",
        },
        {
            "setup": "import numpy as np\nkin = np.array([[[2.07e-2, 170.0, 340.0, 6.2], [4.15e-4, 6500.0, 1.7, 293.0]],\n                [[3.92e-4, 6500.0, 1.7, 311.0], [9.3e-3, 170.0, 340.0, 2.4]]])\ntemplate = np.array([0, 0, 1, 1, 0, 1, 1, 1, 0, 0, 1, 0])\nconc = np.array([1.0, 1.0])\nppi, kp_const = 1.0e-4, 5.6e-3\nrates_m = reduced_transition_rates(kin, conc, ppi, kp_const)\nparams_m = backward_map_parameters(rates_m)\nx_m = backward_iteration(params_m, template)\nY_m = site_transfer_factors(x_m, params_m, rates_m, template)\nrates_o = _oracle_reduced_transition_rates(kin, conc, ppi, kp_const)\nparams_o = _oracle_backward_map_parameters(rates_o)\nx_o = _oracle_backward_iteration(params_o, template)\nY_o = _oracle_site_transfer_factors(x_o, params_o, rates_o, template)\n",
            "call": "mean_growth_velocity(x_m, Y_m)",
            "gold_call": "_oracle_mean_growth_velocity(x_o, Y_o)",
        },
        {
            "setup": "import numpy as np\nkin = np.array([[[1.0e-2, 100.0, 200.0, 5.0], [5.0e-4, 5000.0, 2.0, 250.0]],\n                [[4.0e-4, 5000.0, 2.0, 300.0], [1.0e-2, 100.0, 200.0, 3.0]]])\ntemplate = np.array([0, 1, 1, 0, 1])\nconc = np.array([1.0e-6, 1.0e-6])\nppi, kp_const = 5.0e-4, 1.0e-3\nrates_m = reduced_transition_rates(kin, conc, ppi, kp_const)\nparams_m = backward_map_parameters(rates_m)\nx_m = backward_iteration(params_m, template)\nY_m = site_transfer_factors(x_m, params_m, rates_m, template)\nrates_o = _oracle_reduced_transition_rates(kin, conc, ppi, kp_const)\nparams_o = _oracle_backward_map_parameters(rates_o)\nx_o = _oracle_backward_iteration(params_o, template)\nY_o = _oracle_site_transfer_factors(x_o, params_o, rates_o, template)\n",
            "call": "mean_growth_velocity(x_m, Y_m)",
            "gold_call": "_oracle_mean_growth_velocity(x_o, Y_o)",
        },
        {
            "setup": "import numpy as np\nkin = np.array([[[2.07e-2, 170.0, 340.0, 6.2], [4.15e-4, 6500.0, 1.7, 293.0]],\n                [[3.92e-4, 6500.0, 1.7, 311.0], [9.3e-3, 170.0, 340.0, 2.4]]])\ntemplate = np.array([0, 0, 1, 1, 0, 1, 1, 1, 0, 0, 1, 0])\nconc = np.array([2.0e-8, 4.0e-9])\nppi, kp_const = 1.0e-4, 5.6e-3\nrates_m = reduced_transition_rates(kin, conc, ppi, kp_const)\nparams_m = backward_map_parameters(rates_m)\nx_m = backward_iteration(params_m, template)\nY_m = site_transfer_factors(x_m, params_m, rates_m, template)\nrates_o = _oracle_reduced_transition_rates(kin, conc, ppi, kp_const)\nparams_o = _oracle_backward_map_parameters(rates_o)\nx_o = _oracle_backward_iteration(params_o, template)\nY_o = _oracle_site_transfer_factors(x_o, params_o, rates_o, template)\ndef run_model():\n    try:\n        mean_growth_velocity(x_m, Y_m[:-1])\n        return 0\n    except ValueError:\n        return 1\ndef run_oracle():\n    try:\n        _oracle_mean_growth_velocity(x_o, Y_o[:-1])\n        return 0\n    except ValueError:\n        return 1\n",
            "call": "run_model()",
            "gold_call": "run_oracle()",
        },
    ]
