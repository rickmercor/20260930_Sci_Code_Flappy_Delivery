"""
Run the source's backward iterated map for the growth regime around the periodic template until its fixed cycle has converged: starting from unit value at the end of a period, step backward site by site through one period, close the period by identifying x_0 with x_L, and repeat loop after loop until the whole cycle changes by less than 1e-13 relatively (at most 100000 loops). Return x[0..L] with x[0] == x[L]; site l (1-based) copies template[l-1] and is followed by template[l % L].

In the growth regime the long-time solution of the copolymerization kinetics has a product structure along the template, and the site-dependent scale factors of that product obey a first-order recursion that runs against the direction of synthesis. On a periodic template the recursion has a periodic fixed cycle, reached by iterating around the period.

Returns
-------
ndarray of float64, shape (L + 1,), the converged periodic cycle x[0..L] of the backward map.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def backward_iteration(params: "np.ndarray", template: "np.ndarray") -> "np.ndarray":
    """Run the source's backward iterated map for the growth regime around the periodic template until its fixed cycle has converged: starting from unit value at the end of a period, step backward site by site through one period, close the period by identifying x_0 with x_L, and repeat loop after loop until the whole cycle changes by less than 1e-13 relatively (at most 100000 loops). Return x[0..L] with x[0] == x[L]; site l (1-based) copies template[l-1] and is followed by template[l % L].

    Parameters
    ----------
    params : np.ndarray
        Effective rates (alpha, beta) of shape (2, 2, 2, 2).
    template : np.ndarray
        One period of the template as integers 0 (A) and 1 (T).

    Returns
    -------
    x : np.ndarray
        Cycle values x[0..L] with x[0] == x[L].

    Raises
    ------
    ValueError
        If params is not (2, 2, 2, 2) finite nonnegative or template is not a nonempty 0/1 integer array.
    """
    return x

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _check_template(template):
    t = np.asarray(template).ravel()
    if t.size < 1 or t.dtype.kind not in "iu" or np.any((t != 0) & (t != 1)):
        raise ValueError("template must be a non-empty integer array of 0 (A) and 1 (T)")
    return t.astype(np.int64)


def _check_positive(x, name):
    if isinstance(x, bool) or not np.isfinite(x) or float(x) <= 0.0:
        raise ValueError(f"{name} must be a finite positive number")
    return float(x)


def _oracle_backward_iteration(params: "np.ndarray", template: "np.ndarray") -> "np.ndarray":
    """Eq. (29) run backward around the periodic template until the fixed cycle converges.

    x[l] for l = 0..L with x[0] == x[L] (periodic closure, eq. 39 regime); site l (1-based) has
    template unit template[l-1] and next unit template[l % L]. The map is
    x_{l-1} = x_l * sum_m alpha[m, n_l] / (x_l + beta[m, n_l, n_{l+1}]). Started from x_L = 1 and
    iterated loop after loop; converged when the whole cycle moves by less than 1e-13 relatively
    (at most 100000 loops).
    """
    p = np.asarray(params, dtype=np.float64)
    if p.shape != (2, 2, 2, 2) or not np.all(np.isfinite(p)) or np.any(p < 0.0):
        raise ValueError("params must be a (2, 2, 2, 2) array of finite nonnegative values")
    t = _check_template(template)
    max_loops, rtol = 100000, 1e-13          # fixed convergence settings of the method
    L = t.size
    alpha, beta = p[..., 0], p[..., 1]
    x = np.zeros(L + 1, dtype=np.float64)
    x[L] = 1.0
    for _ in range(int(max_loops)):
        prev = x.copy()
        for l in range(L, 0, -1):
            n, nn = t[l - 1], t[l % L]
            xl = x[l]
            x[l - 1] = xl * np.sum(alpha[:, n, nn] / (xl + beta[:, n, nn]))
        x[L] = x[0]
        if np.all(np.abs(x - prev) <= rtol * np.maximum(np.abs(x), 1e-300)):
            break
    return x

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return the list of test case specifications."""
    return [
        {
            "setup": "import numpy as np\nkin = np.array([[[2.07e-2, 170.0, 340.0, 6.2], [4.15e-4, 6500.0, 1.7, 293.0]],\n                [[3.92e-4, 6500.0, 1.7, 311.0], [9.3e-3, 170.0, 340.0, 2.4]]])\ntemplate = np.array([0, 0, 1, 1, 0, 1, 1, 1, 0, 0, 1, 0])\nconc = np.array([2.0e-8, 4.0e-9])\nppi, kp_const = 1.0e-4, 5.6e-3\nrates_m = reduced_transition_rates(kin, conc, ppi, kp_const)\nparams_m = backward_map_parameters(rates_m)\nrates_o = _oracle_reduced_transition_rates(kin, conc, ppi, kp_const)\nparams_o = _oracle_backward_map_parameters(rates_o)\n",
            "call": "np.asarray(backward_iteration(params_m, template))",
            "gold_call": "np.asarray(_oracle_backward_iteration(params_o, template))",
        },
        {
            "setup": "import numpy as np\nkin = np.array([[[2.07e-2, 170.0, 340.0, 6.2], [4.15e-4, 6500.0, 1.7, 293.0]],\n                [[3.92e-4, 6500.0, 1.7, 311.0], [9.3e-3, 170.0, 340.0, 2.4]]])\ntemplate = np.array([0, 0, 1, 1, 0, 1, 1, 1, 0, 0, 1, 0])\nconc = np.array([1.0, 1.0])\nppi, kp_const = 1.0e-4, 5.6e-3\nrates_m = reduced_transition_rates(kin, conc, ppi, kp_const)\nparams_m = backward_map_parameters(rates_m)\nrates_o = _oracle_reduced_transition_rates(kin, conc, ppi, kp_const)\nparams_o = _oracle_backward_map_parameters(rates_o)\n",
            "call": "np.asarray(backward_iteration(params_m, template))",
            "gold_call": "np.asarray(_oracle_backward_iteration(params_o, template))",
        },
        {
            "setup": "import numpy as np\nkin = np.array([[[1.0e-2, 100.0, 200.0, 5.0], [5.0e-4, 5000.0, 2.0, 250.0]],\n                [[4.0e-4, 5000.0, 2.0, 300.0], [1.0e-2, 100.0, 200.0, 3.0]]])\ntemplate = np.array([0, 1, 1, 0, 1])\nconc = np.array([1.0e-6, 1.0e-6])\nppi, kp_const = 5.0e-4, 1.0e-3\nrates_m = reduced_transition_rates(kin, conc, ppi, kp_const)\nparams_m = backward_map_parameters(rates_m)\nrates_o = _oracle_reduced_transition_rates(kin, conc, ppi, kp_const)\nparams_o = _oracle_backward_map_parameters(rates_o)\n",
            "call": "np.asarray(backward_iteration(params_m, template))",
            "gold_call": "np.asarray(_oracle_backward_iteration(params_o, template))",
        },
        {
            "setup": "import numpy as np\nkin = np.array([[[2.07e-2, 170.0, 340.0, 6.2], [4.15e-4, 6500.0, 1.7, 293.0]],\n                [[3.92e-4, 6500.0, 1.7, 311.0], [9.3e-3, 170.0, 340.0, 2.4]]])\ntemplate = np.array([0, 0, 1, 1, 0, 1, 1, 1, 0, 0, 1, 0])\nconc = np.array([2.0e-8, 4.0e-9])\nppi, kp_const = 1.0e-4, 5.6e-3\nrates_m = reduced_transition_rates(kin, conc, ppi, kp_const)\nparams_m = backward_map_parameters(rates_m)\nrates_o = _oracle_reduced_transition_rates(kin, conc, ppi, kp_const)\nparams_o = _oracle_backward_map_parameters(rates_o)\ndef run_model():\n    try:\n        backward_iteration(params_m, np.array([0, 2, 1]))\n        return 0\n    except ValueError:\n        return 1\ndef run_oracle():\n    try:\n        _oracle_backward_iteration(params_o, np.array([0, 2, 1]))\n        return 0\n    except ValueError:\n        return 1\n",
            "call": "run_model()",
            "gold_call": "run_oracle()",
        },
    ]
