"""
Find the onset of steady growth for the periodic template: the factor s by which both nucleotide concentrations must be multiplied for the copy to stop growing, i.e. the point where the long-time mean length changes from bounded to linearly increasing. Use the source's criterion for the onset, which compares site-wise effective backward and forward rates of the copy length evaluated at the onset itself; bracket the sign change by scanning s over decades from s = 1 and refine by bisection on log10(s) to 1e-12. The scale can be above 1 when the given concentrations already stall.

Template-directed copolymerization with a reverse reaction behaves like a random walk of the copy length in a sequence-dependent environment. Growth sets in when the accumulated bias of the walk along one period turns from backward to forward, a condition fixed by the geometric mean of the local backward-to-forward rate ratios rather than by any single site.

Returns
-------
float, the onset scale factor s at which the mean growth velocity vanishes.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def onset_of_growth_scale(kin: "np.ndarray", conc: "np.ndarray", ppi: float, kp_const: float,
                                  template: "np.ndarray") -> float:
    """Find the onset of steady growth for the periodic template: the factor s by which both nucleotide concentrations must be multiplied for the copy to stop growing, i.e. the point where the long-time mean length changes from bounded to linearly increasing. Use the source's criterion for the onset, which compares site-wise effective backward and forward rates of the copy length evaluated at the onset itself; bracket the sign change by scanning s over decades from s = 1 and refine by bisection on log10(s) to 1e-12. The scale can be above 1 when the given concentrations already stall.

    Parameters
    ----------
    kin : np.ndarray
        Kinetic table of shape (2, 2, 4).
    conc : np.ndarray
        Reference nucleotide concentrations ([dATP], [dTTP]) in M that s multiplies.
    ppi : float
        Pyrophosphate concentration in M.
    kp_const : float
        Pyrophosphorolysis constant K_P in M.
    template : np.ndarray
        One period of the template as integers 0 (A) and 1 (T).

    Returns
    -------
    s_onset : float
        Onset scale factor.

    Raises
    ------
    ValueError
        If any input is invalid as in the earlier steps, or no onset is found within 30 decades.
    """
    return s_onset

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _check_kin(kin):
    k = np.asarray(kin, dtype=np.float64)
    if k.shape != (2, 2, 4) or not np.all(np.isfinite(k)) or np.any(k <= 0.0):
        raise ValueError("kin must be a (2, 2, 4) array of finite positive constants")
    return k


def _check_conc(conc):
    c = np.asarray(conc, dtype=np.float64).ravel()
    if c.shape != (2,) or not np.all(np.isfinite(c)) or np.any(c <= 0.0):
        raise ValueError("conc must be two finite positive concentrations (dATP, dTTP)")
    return c


def _check_positive(x, name):
    if isinstance(x, bool) or not np.isfinite(x) or float(x) <= 0.0:
        raise ValueError(f"{name} must be a finite positive number")
    return float(x)


def _check_template(template):
    t = np.asarray(template).ravel()
    if t.size < 1 or t.dtype.kind not in "iu" or np.any((t != 0) & (t != 1)):
        raise ValueError("template must be a non-empty integer array of 0 (A) and 1 (T)")
    return t.astype(np.int64)


def _mean_log_backward_forward_ratio(kin, conc, ppi, kp_const, template, scale):
    """Appendix C eqs. (112)-(113) evaluated in the stationary limit x_l = 0 (the onset itself):
    < ln(b_l / a_l) > over the period at both concentrations multiplied by `scale`."""
    rates = _oracle_reduced_transition_rates(kin, np.asarray(conc, dtype=np.float64) * scale, ppi, kp_const)
    params = _oracle_backward_map_parameters(rates)
    t = np.asarray(template, dtype=np.int64)
    L = t.size
    Y = _oracle_site_transfer_factors(np.zeros(L + 1), params, rates, t)
    RE = Y[:, :, 0].sum(axis=1)
    RF = Y[:, :, 1].sum(axis=1)
    logs = np.zeros(L)
    for l in range(1, L + 1):
        n = t[l - 1]
        prev = l - 2 if l >= 2 else L - 1          # site l-1; periodic closure: site 0 == site L
        a = np.sum(rates[:, n, 0, 0]) / (1.0 + RF[prev] / RE[prev])
        b = np.sum(Y[l - 1, :, 1] * rates[:, n, 0, 1]) / (RE[l - 1] + RF[l - 1])
        logs[l - 1] = np.log(b / a)
    return float(np.mean(logs))


def _oracle_onset_of_growth_scale(kin: "np.ndarray", conc: "np.ndarray", ppi: float, kp_const: float,
                                  template: "np.ndarray") -> float:
    """Onset of steady growth (Appendix C, gamma -> 0): the factor s applied to both nucleotide
    concentrations at which < ln(b_l / a_l) > = 0 over the periodic template. The sign change is
    bracketed by scanning s over decades from s = 1 (downward if the given concentrations grow,
    upward if they stall) and located by bisection on log10(s) to 1e-12."""
    k = _check_kin(kin)
    c = _check_conc(conc)
    ppi = _check_positive(ppi, "ppi")
    kp_const = _check_positive(kp_const, "kp_const")
    t = _check_template(template)
    tol = 1e-12                               # fixed bisection tolerance on log10(s) of the method
    f = lambda ls: _mean_log_backward_forward_ratio(k, c, ppi, kp_const, t, 10.0 ** ls)
    f1 = f(0.0)
    step = -1.0 if f1 < 0.0 else 1.0                # growing at s = 1: onset lies below; else above
    lo = 0.0
    for _ in range(30):
        hi = lo + step
        if f(hi) * f1 < 0.0:
            break
        lo = hi
    else:
        raise ValueError("no onset of growth found within 30 decades of the given concentrations")
    lo, hi = (min(lo, hi), max(lo, hi))
    f_lo = f(lo)
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if f(mid) * f_lo > 0.0:
            lo, f_lo = mid, f(mid)
        else:
            hi = mid
        if hi - lo <= tol:
            break
    return float(10.0 ** (0.5 * (lo + hi)))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return the list of test case specifications."""
    return [
        {
            "setup": "import numpy as np\nkin = np.array([[[2.07e-2, 170.0, 340.0, 6.2], [4.15e-4, 6500.0, 1.7, 293.0]],\n                [[3.92e-4, 6500.0, 1.7, 311.0], [9.3e-3, 170.0, 340.0, 2.4]]])\ntemplate = np.array([0, 0, 1, 1, 0, 1, 1, 1, 0, 0, 1, 0])\nconc = np.array([2.0e-8, 4.0e-9])\nppi, kp_const = 1.0e-4, 5.6e-3\n",
            "call": "onset_of_growth_scale(kin, conc, ppi, kp_const, template)",
            "gold_call": "_oracle_onset_of_growth_scale(kin, conc, ppi, kp_const, template)",
        },
        {
            "setup": "import numpy as np\nkin = np.array([[[2.07e-2, 170.0, 340.0, 6.2], [4.15e-4, 6500.0, 1.7, 293.0]],\n                [[3.92e-4, 6500.0, 1.7, 311.0], [9.3e-3, 170.0, 340.0, 2.4]]])\ntemplate = np.array([0, 0, 1, 1, 0, 1, 1, 1, 0, 0, 1, 0])\nconc = np.array([1.0e-9, 2.0e-10])\nppi, kp_const = 1.0e-4, 5.6e-3\n",
            "call": "onset_of_growth_scale(kin, conc, ppi, kp_const, template)",
            "gold_call": "_oracle_onset_of_growth_scale(kin, conc, ppi, kp_const, template)",
        },
        {
            "setup": "import numpy as np\nkin = np.array([[[1.0e-2, 100.0, 200.0, 5.0], [5.0e-4, 5000.0, 2.0, 250.0]],\n                [[4.0e-4, 5000.0, 2.0, 300.0], [1.0e-2, 100.0, 200.0, 3.0]]])\ntemplate = np.array([0, 1, 1, 0, 1])\nconc = np.array([1.0e-6, 1.0e-6])\nppi, kp_const = 5.0e-4, 1.0e-3\n",
            "call": "onset_of_growth_scale(kin, conc, ppi, kp_const, template)",
            "gold_call": "_oracle_onset_of_growth_scale(kin, conc, ppi, kp_const, template)",
        },
        {
            "setup": "import numpy as np\nkin = np.array([[[2.07e-2, 170.0, 340.0, 6.2], [4.15e-4, 6500.0, 1.7, 293.0]],\n                [[3.92e-4, 6500.0, 1.7, 311.0], [9.3e-3, 170.0, 340.0, 2.4]]])\ntemplate = np.array([0, 0, 1, 1, 0, 1, 1, 1, 0, 0, 1, 0])\nconc = np.array([2.0e-8, 4.0e-9])\nppi, kp_const = 1.0e-4, 5.6e-3\ndef run_model():\n    try:\n        onset_of_growth_scale(kin, conc, -1.0e-4, kp_const, template)\n        return 0\n    except ValueError:\n        return 1\ndef run_oracle():\n    try:\n        _oracle_onset_of_growth_scale(kin, conc, -1.0e-4, kp_const, template)\n        return 0\n    except ValueError:\n        return 1\n",
            "call": "run_model()",
            "gold_call": "run_oracle()",
        },
    ]
