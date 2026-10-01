"""
Compute, for every site l of the period and every copy unit m, the two entries of the source's reduced site transfer matrix: the open-state entry that weights the composition of the copy and the closed-state entry that weights the time spent closed, both from the converged cycle x, the effective parameters and the reduced rates. Return Y[l-1, m, s] with s = 0 for the open-state entry and s = 1 for the closed-state entry; the sums over m give the two forward factors of the site.

The forward iteration that propagates the long-time solution along the template multiplies site-by-site transfer matrices whose entries are ratios of effective rates to the local scale factor of the backward recursion. Because the scheme is strictly sequential these matrices have a single nonzero column, so two numbers per copy unit and site suffice.

Returns
-------
ndarray of float64, shape (L, 2, 2): Y[l-1, m, (open, closed)] transfer entries of site l.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def site_transfer_factors(x: "np.ndarray", params: "np.ndarray", rates: "np.ndarray",
                                  template: "np.ndarray") -> "np.ndarray":
    """Compute, for every site l of the period and every copy unit m, the two entries of the source's reduced site transfer matrix: the open-state entry that weights the composition of the copy and the closed-state entry that weights the time spent closed, both from the converged cycle x, the effective parameters and the reduced rates. Return Y[l-1, m, s] with s = 0 for the open-state entry and s = 1 for the closed-state entry; the sums over m give the two forward factors of the site.

    Parameters
    ----------
    x : np.ndarray
        Converged cycle x[0..L] of the backward iteration.
    params : np.ndarray
        Effective rates (alpha, beta) of shape (2, 2, 2, 2).
    rates : np.ndarray
        Reduced transition rates of shape (2, 2, 2, 4).
    template : np.ndarray
        One period of the template as integers 0 (A) and 1 (T).

    Returns
    -------
    Y : np.ndarray
        Transfer entries of shape (L, 2, 2).

    Raises
    ------
    ValueError
        If the template is invalid, x does not have L + 1 finite nonnegative entries, the array shapes do not match, or a denominator vanishes.
    """
    return Y

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _check_template(template):
    t = np.asarray(template).ravel()
    if t.size < 1 or t.dtype.kind not in "iu" or np.any((t != 0) & (t != 1)):
        raise ValueError("template must be a non-empty integer array of 0 (A) and 1 (T)")
    return t.astype(np.int64)


def _oracle_site_transfer_factors(x: "np.ndarray", params: "np.ndarray", rates: "np.ndarray",
                                  template: "np.ndarray") -> "np.ndarray":
    """Eq. (33): the entries Y^E_{m,l} and Y^F_{m,l} of the reduced 2x2 transfer matrices.

    Returns Y[l-1, m, s] for site l = 1..L, copy unit m and structural state s (0 = E, 1 = F):
    Y^E = alpha / (x_l + beta), Y^F = (alpha / w_pol) (x_l + w_depol) / (x_l + beta), with beta and
    w_depol taken at the doublet (n_l, n_{l+1}). Column sums R^E_l = sum_m Y^E and R^F_l = sum_m Y^F
    are the forward-iteration factors of eqs. (35)-(36).
    """
    t = _check_template(template)
    L = t.size
    xx = np.asarray(x, dtype=np.float64).ravel()
    if xx.shape != (L + 1,) or not np.all(np.isfinite(xx)) or np.any(xx < 0.0):
        raise ValueError("x must have L + 1 finite nonnegative entries")
    p = np.asarray(params, dtype=np.float64)
    r = np.asarray(rates, dtype=np.float64)
    if p.shape != (2, 2, 2, 2) or r.shape != (2, 2, 2, 4):
        raise ValueError("params must be (2, 2, 2, 2) and rates (2, 2, 2, 4)")
    Y = np.zeros((L, 2, 2), dtype=np.float64)
    for l in range(1, L + 1):
        n, nn = t[l - 1], t[l % L]
        for m in range(2):
            alpha, beta = p[m, n, nn]
            w_pol, w_dep = r[m, n, nn, 2], r[m, n, nn, 3]
            den = xx[l] + beta
            if den <= 0.0:
                raise ValueError("x_l + beta must be positive")
            Y[l - 1, m, 0] = alpha / den
            Y[l - 1, m, 1] = (alpha / w_pol) * (xx[l] + w_dep) / den
    return Y

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return the list of test case specifications."""
    return [
        {
            "setup": "import numpy as np\nkin = np.array([[[2.07e-2, 170.0, 340.0, 6.2], [4.15e-4, 6500.0, 1.7, 293.0]],\n                [[3.92e-4, 6500.0, 1.7, 311.0], [9.3e-3, 170.0, 340.0, 2.4]]])\ntemplate = np.array([0, 0, 1, 1, 0, 1, 1, 1, 0, 0, 1, 0])\nconc = np.array([2.0e-8, 4.0e-9])\nppi, kp_const = 1.0e-4, 5.6e-3\nrates_m = reduced_transition_rates(kin, conc, ppi, kp_const)\nparams_m = backward_map_parameters(rates_m)\nx_m = backward_iteration(params_m, template)\nrates_o = _oracle_reduced_transition_rates(kin, conc, ppi, kp_const)\nparams_o = _oracle_backward_map_parameters(rates_o)\nx_o = _oracle_backward_iteration(params_o, template)\n",
            "call": "np.asarray(site_transfer_factors(x_m, params_m, rates_m, template))",
            "gold_call": "np.asarray(_oracle_site_transfer_factors(x_o, params_o, rates_o, template))",
        },
        {
            "setup": "import numpy as np\nkin = np.array([[[2.07e-2, 170.0, 340.0, 6.2], [4.15e-4, 6500.0, 1.7, 293.0]],\n                [[3.92e-4, 6500.0, 1.7, 311.0], [9.3e-3, 170.0, 340.0, 2.4]]])\ntemplate = np.array([0, 0, 1, 1, 0, 1, 1, 1, 0, 0, 1, 0])\nconc = np.array([1.0, 1.0])\nppi, kp_const = 1.0e-4, 5.6e-3\nrates_m = reduced_transition_rates(kin, conc, ppi, kp_const)\nparams_m = backward_map_parameters(rates_m)\nx_m = backward_iteration(params_m, template)\nrates_o = _oracle_reduced_transition_rates(kin, conc, ppi, kp_const)\nparams_o = _oracle_backward_map_parameters(rates_o)\nx_o = _oracle_backward_iteration(params_o, template)\n",
            "call": "np.asarray(site_transfer_factors(x_m, params_m, rates_m, template))",
            "gold_call": "np.asarray(_oracle_site_transfer_factors(x_o, params_o, rates_o, template))",
        },
        {
            "setup": "import numpy as np\nkin = np.array([[[1.0e-2, 100.0, 200.0, 5.0], [5.0e-4, 5000.0, 2.0, 250.0]],\n                [[4.0e-4, 5000.0, 2.0, 300.0], [1.0e-2, 100.0, 200.0, 3.0]]])\ntemplate = np.array([0, 1, 1, 0, 1])\nconc = np.array([1.0e-6, 1.0e-6])\nppi, kp_const = 5.0e-4, 1.0e-3\nrates_m = reduced_transition_rates(kin, conc, ppi, kp_const)\nparams_m = backward_map_parameters(rates_m)\nx_m = backward_iteration(params_m, template)\nrates_o = _oracle_reduced_transition_rates(kin, conc, ppi, kp_const)\nparams_o = _oracle_backward_map_parameters(rates_o)\nx_o = _oracle_backward_iteration(params_o, template)\n",
            "call": "np.asarray(site_transfer_factors(x_m, params_m, rates_m, template))",
            "gold_call": "np.asarray(_oracle_site_transfer_factors(x_o, params_o, rates_o, template))",
        },
        {
            "setup": "import numpy as np\nkin = np.array([[[2.07e-2, 170.0, 340.0, 6.2], [4.15e-4, 6500.0, 1.7, 293.0]],\n                [[3.92e-4, 6500.0, 1.7, 311.0], [9.3e-3, 170.0, 340.0, 2.4]]])\ntemplate = np.array([0, 0, 1, 1, 0, 1, 1, 1, 0, 0, 1, 0])\nconc = np.array([2.0e-8, 4.0e-9])\nppi, kp_const = 1.0e-4, 5.6e-3\nrates_m = reduced_transition_rates(kin, conc, ppi, kp_const)\nparams_m = backward_map_parameters(rates_m)\nx_m = backward_iteration(params_m, template)\nrates_o = _oracle_reduced_transition_rates(kin, conc, ppi, kp_const)\nparams_o = _oracle_backward_map_parameters(rates_o)\nx_o = _oracle_backward_iteration(params_o, template)\ndef run_model():\n    try:\n        site_transfer_factors(x_m[:-1], params_m, rates_m, template)\n        return 0\n    except ValueError:\n        return 1\ndef run_oracle():\n    try:\n        _oracle_site_transfer_factors(x_o[:-1], params_o, rates_o, template)\n        return 0\n    except ValueError:\n        return 1\n",
            "call": "run_model()",
            "gold_call": "run_oracle()",
        },
    ]
