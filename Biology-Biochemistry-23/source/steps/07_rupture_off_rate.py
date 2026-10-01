"""
Return the rupture off-rate k (units of k_a) of a duplex whose master-equation generator T is given: propagate the probability vector from the fully closed duplex (index 0) as P(t) = exp(T t) P(0), sample the probability of the absorbing ruptured state (last index) at n_points equally spaced times from 0 to t_end inclusive, and fit the single-exponential law P_ss(t) = 1 - exp(-k t) by unweighted nonlinear least squares in k (initial guess 1 / t_end), iterating the fit to convergence with relative tolerances of 1e-14 on the parameter, the residual and the gradient (library defaults of about 1e-8 leave the last digits unconverged).

Experiments report rupture-time distributions and summarise them by a single off-rate, so the model is reduced the same way: the time course of the absorbing-state probability is fitted by a single exponential over a window comparable to the thermal dissociation time.

Returns
-------
float, the fitted off-rate k in units of k_a.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def rupture_off_rate(generator: "np.ndarray", t_end: float, n_points: int) -> float:
    """Return the rupture off-rate k (units of k_a) of a duplex whose master-equation generator T is given: propagate the probability vector from the fully closed duplex (index 0) as P(t) = exp(T t) P(0), sample the probability of the absorbing ruptured state (last index) at n_points equally spaced times from 0 to t_end inclusive, and fit the single-exponential law P_ss(t) = 1 - exp(-k t) by unweighted nonlinear least squares in k (initial guess 1 / t_end), iterating the fit to convergence with relative tolerances of 1e-14 on the parameter, the residual and the gradient (library defaults of about 1e-8 leave the last digits unconverged).

    Parameters
    ----------
    generator : np.ndarray
        Square generator matrix T of the master equation.
    t_end : float
        End of the sampling window in units of 1 / k_a (> 0).
    n_points : int
        Number of equally spaced sample times including 0 and t_end (>= 3).

    Returns
    -------
    k_off : float
        Fitted off-rate in units of k_a.

    Raises
    ------
    ValueError
        If generator is not a finite square matrix of size >= 2, t_end is not positive, or n_points < 3.
    """
    return k_off

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq, curve_fit


def _check_pos(x, name):
    if isinstance(x, bool) or not np.isfinite(x) or float(x) <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return float(x)


def _check_int(n, name, lo):
    if isinstance(n, bool) or int(n) != n or int(n) < lo:
        raise ValueError(f"{name} must be an integer >= {lo}")
    return int(n)


def _oracle_rupture_off_rate(generator: "np.ndarray", t_end: float, n_points: int) -> float:
    """Off-rate k (units of k_a) from a single-exponential fit P_ss(t) ~ 1 - exp(-k t), SI eq. (30) protocol.

    P(t) = expm(T t) P0 with P0 = full duplex (index 0); P_ss is the last component; sampled at n_points equally
    spaced times in [0, t_end] and fitted by unweighted nonlinear least squares with the single parameter k
    (initial guess 1 / t_end), iterated to convergence (parameter, residual and gradient tolerances 1e-14).
    """
    Tm = np.asarray(generator, dtype=float)
    if Tm.ndim != 2 or Tm.shape[0] != Tm.shape[1] or Tm.shape[0] < 2 or not np.all(np.isfinite(Tm)):
        raise ValueError("generator must be a finite square matrix of size >= 2")
    te = _check_pos(t_end, "t_end")
    npts = _check_int(n_points, "n_points", 3)
    w, V = np.linalg.eig(Tm)
    coef = np.linalg.solve(V, np.eye(Tm.shape[0])[:, 0])
    ts = np.linspace(0.0, te, npts)
    p_ss = np.real(V[-1, :] @ (np.exp(np.outer(w, ts)) * coef[:, None]))
    k, _ = curve_fit(lambda t, kk: 1.0 - np.exp(-kk * t), ts, p_ss, p0=[1.0 / te], xtol=1e-14, ftol=1e-14, gtol=1e-14)
    return float(k[0])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return the list of test case specifications."""
    return [
        {
            "setup": "import numpy as np\ngenerator = np.array([[-0.5, 0.0], [0.5, 0.0]])\nt_end, n_points = 10.0, 1000\n",
            "call": "rupture_off_rate(generator, t_end, n_points)",
            "gold_call": "_oracle_rupture_off_rate(generator, t_end, n_points)",
            "tol": 1e-08,
        },
        {
            "setup": "import numpy as np\ngenerator = _oracle_master_equation_generator(4, np.array([5.0, 0.2, 0.2, 0.25, 0.9, 0.95, 1.1]))\nt_end, n_points = 200.0, 500\n",
            "call": "rupture_off_rate(generator, t_end, n_points)",
            "gold_call": "_oracle_rupture_off_rate(generator, t_end, n_points)",
            "tol": 1e-08,
        },
        {
            "setup": "import numpy as np\ndh_stack, ds_stack, dh_non, ds_non = -7.6 * 4.184, -0.0920, 4.6 * 4.184, 0.0502\nlam_ss, l_ss = 0.77, 0.7\nn_bp = 9\ngenerator = _oracle_master_equation_generator(n_bp, _oracle_transition_rates(6.0, n_bp, 303.15, dh_stack, ds_stack, dh_non, ds_non, lam_ss, l_ss))\nt_end, n_points = 49385.0, 1000\n",
            "call": "rupture_off_rate(generator, t_end, n_points)",
            "gold_call": "_oracle_rupture_off_rate(generator, t_end, n_points)",
            "tol": 1e-08,
        },
        {
            "setup": "import numpy as np\ngenerator = np.array([[-0.5, 0.0], [0.5, 0.0]])\ndef run_model():\n    try:\n        rupture_off_rate(generator, 10.0, 2)\n        return 0\n    except ValueError:\n        return 1\ndef run_oracle():\n    try:\n        _oracle_rupture_off_rate(generator, 10.0, 2)\n        return 0\n    except ValueError:\n        return 1\n",
            "call": "run_model()",
            "gold_call": "run_oracle()",
        },
    ]
