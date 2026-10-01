"""
Compute the replication error probability: the long-time mean, over the sites of the period, of the probability that the copy unit at a site is the incorrect one (the same unit as the template unit), from the site transfer entries Y and the template; the local probability of each copy unit at a site follows from the open-state entries of that site.

The composition of the copy at each template position is a stationary property of the long-time solution and reflects both the kinetic discrimination at incorporation and the preferential removal of mismatched units by the reverse reaction.

Returns
-------
float, the error probability eta (mean fraction of incorrect pairs per site).
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def error_probability(Y: "np.ndarray", template: "np.ndarray") -> float:
    """Compute the replication error probability: the long-time mean, over the sites of the period, of the probability that the copy unit at a site is the incorrect one (the same unit as the template unit), from the site transfer entries Y and the template; the local probability of each copy unit at a site follows from the open-state entries of that site.

    Parameters
    ----------
    Y : np.ndarray
        Transfer entries of shape (L, 2, 2).
    template : np.ndarray
        One period of the template as integers 0 (A) and 1 (T).

    Returns
    -------
    eta : float
        Error probability per nucleotide.

    Raises
    ------
    ValueError
        If the template is invalid, Y is not a finite nonnegative (L, 2, 2) array, or some open-state column sum vanishes.
    """
    return eta

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _check_template(template):
    t = np.asarray(template).ravel()
    if t.size < 1 or t.dtype.kind not in "iu" or np.any((t != 0) & (t != 1)):
        raise ValueError("template must be a non-empty integer array of 0 (A) and 1 (T)")
    return t.astype(np.int64)


def _oracle_error_probability(Y: "np.ndarray", template: "np.ndarray") -> float:
    """Eqs. (43)-(44): eta = < Y^E_{n_l, l} / R^E_l >, the mean fraction of incorrect pairs (m = n)."""
    t = _check_template(template)
    YY = np.asarray(Y, dtype=np.float64)
    if YY.shape != (t.size, 2, 2) or not np.all(np.isfinite(YY)) or np.any(YY < 0.0):
        raise ValueError("Y must be a finite nonnegative (L, 2, 2) array matching the template")
    RE = YY[:, :, 0].sum(axis=1)
    if np.any(RE <= 0.0):
        raise ValueError("R^E_l must be positive")
    mu_wrong = YY[np.arange(t.size), t, 0] / RE
    return float(np.mean(mu_wrong))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return the list of test case specifications."""
    return [
        {
            "setup": "import numpy as np\nkin = np.array([[[2.07e-2, 170.0, 340.0, 6.2], [4.15e-4, 6500.0, 1.7, 293.0]],\n                [[3.92e-4, 6500.0, 1.7, 311.0], [9.3e-3, 170.0, 340.0, 2.4]]])\ntemplate = np.array([0, 0, 1, 1, 0, 1, 1, 1, 0, 0, 1, 0])\nconc = np.array([2.0e-8, 4.0e-9])\nppi, kp_const = 1.0e-4, 5.6e-3\nrates_m = reduced_transition_rates(kin, conc, ppi, kp_const)\nparams_m = backward_map_parameters(rates_m)\nx_m = backward_iteration(params_m, template)\nY_m = site_transfer_factors(x_m, params_m, rates_m, template)\nrates_o = _oracle_reduced_transition_rates(kin, conc, ppi, kp_const)\nparams_o = _oracle_backward_map_parameters(rates_o)\nx_o = _oracle_backward_iteration(params_o, template)\nY_o = _oracle_site_transfer_factors(x_o, params_o, rates_o, template)\n",
            "call": "error_probability(Y_m, template)",
            "gold_call": "_oracle_error_probability(Y_o, template)",
        },
        {
            "setup": "import numpy as np\nkin = np.array([[[2.07e-2, 170.0, 340.0, 6.2], [4.15e-4, 6500.0, 1.7, 293.0]],\n                [[3.92e-4, 6500.0, 1.7, 311.0], [9.3e-3, 170.0, 340.0, 2.4]]])\ntemplate = np.array([0, 0, 1, 1, 0, 1, 1, 1, 0, 0, 1, 0])\nconc = np.array([1.0, 1.0])\nppi, kp_const = 1.0e-4, 5.6e-3\nrates_m = reduced_transition_rates(kin, conc, ppi, kp_const)\nparams_m = backward_map_parameters(rates_m)\nx_m = backward_iteration(params_m, template)\nY_m = site_transfer_factors(x_m, params_m, rates_m, template)\nrates_o = _oracle_reduced_transition_rates(kin, conc, ppi, kp_const)\nparams_o = _oracle_backward_map_parameters(rates_o)\nx_o = _oracle_backward_iteration(params_o, template)\nY_o = _oracle_site_transfer_factors(x_o, params_o, rates_o, template)\n",
            "call": "error_probability(Y_m, template)",
            "gold_call": "_oracle_error_probability(Y_o, template)",
        },
        {
            "setup": "import numpy as np\nkin = np.array([[[1.0e-2, 100.0, 200.0, 5.0], [5.0e-4, 5000.0, 2.0, 250.0]],\n                [[4.0e-4, 5000.0, 2.0, 300.0], [1.0e-2, 100.0, 200.0, 3.0]]])\ntemplate = np.array([0, 1, 1, 0, 1])\nconc = np.array([1.0e-6, 1.0e-6])\nppi, kp_const = 5.0e-4, 1.0e-3\nrates_m = reduced_transition_rates(kin, conc, ppi, kp_const)\nparams_m = backward_map_parameters(rates_m)\nx_m = backward_iteration(params_m, template)\nY_m = site_transfer_factors(x_m, params_m, rates_m, template)\nrates_o = _oracle_reduced_transition_rates(kin, conc, ppi, kp_const)\nparams_o = _oracle_backward_map_parameters(rates_o)\nx_o = _oracle_backward_iteration(params_o, template)\nY_o = _oracle_site_transfer_factors(x_o, params_o, rates_o, template)\n",
            "call": "error_probability(Y_m, template)",
            "gold_call": "_oracle_error_probability(Y_o, template)",
        },
        {
            "setup": "import numpy as np\nkin = np.array([[[2.07e-2, 170.0, 340.0, 6.2], [4.15e-4, 6500.0, 1.7, 293.0]],\n                [[3.92e-4, 6500.0, 1.7, 311.0], [9.3e-3, 170.0, 340.0, 2.4]]])\ntemplate = np.array([0, 0, 1, 1, 0, 1, 1, 1, 0, 0, 1, 0])\nconc = np.array([2.0e-8, 4.0e-9])\nppi, kp_const = 1.0e-4, 5.6e-3\nrates_m = reduced_transition_rates(kin, conc, ppi, kp_const)\nparams_m = backward_map_parameters(rates_m)\nx_m = backward_iteration(params_m, template)\nY_m = site_transfer_factors(x_m, params_m, rates_m, template)\nrates_o = _oracle_reduced_transition_rates(kin, conc, ppi, kp_const)\nparams_o = _oracle_backward_map_parameters(rates_o)\nx_o = _oracle_backward_iteration(params_o, template)\nY_o = _oracle_site_transfer_factors(x_o, params_o, rates_o, template)\ndef run_model():\n    try:\n        error_probability(Y_m, np.array([0, 1]))\n        return 0\n    except ValueError:\n        return 1\ndef run_oracle():\n    try:\n        _oracle_error_probability(Y_o, np.array([0, 1]))\n        return 0\n    except ValueError:\n        return 1\n",
            "call": "run_model()",
            "gold_call": "run_oracle()",
        },
    ]
