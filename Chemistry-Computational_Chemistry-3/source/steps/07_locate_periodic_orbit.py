"""
Find a fixed point of the return map of the previous step, treating the separation and the radial momentum as independent unknowns so that closure is checked directly in both section coordinates. Return the converged separation and radial momentum together with the period.

Requiring both the separation and the radial momentum to return after one complete circuit gives a direct numerical closure test; converge the fixed point until both coordinates return to within 1e-10 or tighter. Integrate to a relative and absolute tolerance of 1e-10 or tighter; the orbit is strongly unstable and looser settings corrupt the quantities read off it.

Returns
-------
ndarray of shape (3,): the separation and radial momentum of the fixed point, and the period in the model time unit.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def locate_periodic_orbit(r_guess: float, pr_guess: float, energy: float, params: dict) -> "np.ndarray":
    """Find a fixed point of the return map of the previous step, treating the separation and the radial momentum as independent unknowns so that closure is checked directly in both section coordinates. Return the converged separation and radial momentum together with the period.

    Parameters
    ----------
    r_guess : float
        Starting separation for the iteration, in angstroms.
    pr_guess : float
        Starting radial momentum for the iteration.
    energy : float
        Total energy held fixed, in kcal/mol.
    params : dict
        Mapping carrying ix, iz, m, de, re, c1, c2, ve, alpha and b.

    Returns
    -------
    orbit : np.ndarray
        ndarray of shape (3,): the separation and radial momentum of the fixed point, and the period in the model time unit.

    Raises
    ------
    ValueError
        if no periodic orbit is found from the supplied guess, or if any evaluation of the return map is invalid.
    """
    return orbit

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
from scipy.integrate import solve_ivp


def _planar_field(y, params):
    """(r, theta, p_r, p_theta) with p_phi = 0; regular because the cot^2 term drops."""
    r, th, pr, pth = y
    ix, m = params["ix"], params["m"]
    de, re, c1, c2 = params["de"], params["re"], params["c1"], params["c2"]
    ve, alpha, b = params["ve"], params["alpha"], params["b"]
    s, c = np.sin(th), np.cos(th)
    a = _oracle_extract_radial_parameters(de, re, c1, c2)
    x = r / re
    dvch = (a[1] * np.exp(c1 * (1.0 - x)) * (-c1 / re)
            + a[2] * (-6.0) * x ** -7 / re + a[3] * (-4.0) * x ** -5 / re)
    v0 = ve * np.exp(-alpha * (r - re) ** 2)
    dv0 = v0 * (-2.0 * alpha * (r - re))
    return np.array([pr / m,
                     pth * (1.0 / ix + 1.0 / (m * r * r)),
                     pth * pth / (m * r ** 3) - dvch - dv0 * (s * s + b * s ** 3),
                     -v0 * (2.0 * s * c + 3.0 * b * s * s * c)], float)

def _planar_pth(r, th, pr, energy, params):
    ix, m = params["ix"], params["m"]
    s = np.sin(th)
    vch = _oracle_radial_potential(r, params["de"], params["re"], params["c1"], params["c2"])
    v0 = params["ve"] * np.exp(-params["alpha"] * (r - params["re"]) ** 2)
    rest = energy - pr * pr / (2.0 * m) - vch - v0 * (s * s + params["b"] * s ** 3)
    coef = 1.0 / (2.0 * ix) + 1.0 / (2.0 * m * r * r)
    if rest <= 0.0:
        raise ValueError("the requested energy is not attainable at this (r, p_r)")
    return np.sqrt(rest / coef)

def _oracle_locate_periodic_orbit(r_guess: float, pr_guess: float, energy: float, params: dict) -> "np.ndarray":
    x = np.array([float(r_guess), float(pr_guess)])
    for _ in range(60):
        out = _oracle_planar_return_map(x[0], x[1], energy, params)
        f = out[:2] - x
        if np.max(np.abs(f)) < 1e-12:
            return np.array([x[0], x[1], out[2]], float)
        J = np.empty((2, 2)); h = 1e-7
        for i in range(2):
            xp = x.copy(); xp[i] += h
            op = _oracle_planar_return_map(xp[0], xp[1], energy, params)
            J[:, i] = ((op[:2] - xp) - f) / h
        dx = np.linalg.solve(J, -f)
        lam = 1.0
        while lam > 1e-5:
            xn = x + lam * dx
            try:
                on = _oracle_planar_return_map(xn[0], xn[1], energy, params)
            except ValueError:
                lam *= 0.5; continue
            if np.max(np.abs(on[:2] - xn)) < np.max(np.abs(f)):
                break
            lam *= 0.5
        x = x + lam * dx
    raise ValueError("the Newton iteration did not converge to a periodic orbit")

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return the list of test-case specifications for this step."""
    return [
        {"tol": 1e-06, "setup": "import numpy as np",
         "call": "locate_periodic_orbit(3.65, 0.0, 0.5, dict(ix=2.373409, iz=4.746818, m=0.9445, de=47.0, re=1.1, c1=7.37, c2=1.61, ve=55.0, alpha=1.0, b=0.0))",
         "gold_call": "_oracle_locate_periodic_orbit(3.65, 0.0, 0.5, dict(ix=2.373409, iz=4.746818, m=0.9445, de=47.0, re=1.1, c1=7.37, c2=1.61, ve=55.0, alpha=1.0, b=0.0))"},   # normal
        {"tol": 1e-06, "setup": "import numpy as np",
         "call": "locate_periodic_orbit(3.63, 0.10, 0.5, dict(ix=2.373409, iz=4.746818, m=0.9445, de=47.0, re=1.1, c1=7.37, c2=1.61, ve=55.0, alpha=1.0, b=0.30))",
         "gold_call": "_oracle_locate_periodic_orbit(3.63, 0.10, 0.5, dict(ix=2.373409, iz=4.746818, m=0.9445, de=47.0, re=1.1, c1=7.37, c2=1.61, ve=55.0, alpha=1.0, b=0.30))"},   # normal
        {"tol": 1e-06, "setup": "import numpy as np",
         "call": "locate_periodic_orbit(3.66, 0.01, 0.5, dict(ix=2.373409, iz=4.746818, m=0.9445, de=47.0, re=1.1, c1=7.37, c2=1.61, ve=55.0, alpha=1.0, b=0.0))",
         "gold_call": "_oracle_locate_periodic_orbit(3.66, 0.01, 0.5, dict(ix=2.373409, iz=4.746818, m=0.9445, de=47.0, re=1.1, c1=7.37, c2=1.61, ve=55.0, alpha=1.0, b=0.0))"},   # boundary
        {"tol": 1e-06, "setup": "import numpy as np",
         "call": "locate_periodic_orbit(3.60, 0.17, 0.5, dict(ix=2.373409, iz=4.746818, m=0.9445, de=47.0, re=1.1, c1=7.37, c2=1.61, ve=55.0, alpha=1.0, b=0.45))",
         "gold_call": "_oracle_locate_periodic_orbit(3.60, 0.17, 0.5, dict(ix=2.373409, iz=4.746818, m=0.9445, de=47.0, re=1.1, c1=7.37, c2=1.61, ve=55.0, alpha=1.0, b=0.45))"},   # edge
        {"setup": "import numpy as np\ndef _c(fn):\n    try:\n        fn(0.5, 0.0, 0.5, dict(ix=2.373409, iz=4.746818, m=0.9445, de=47.0, re=1.1, c1=7.37, c2=1.61, ve=55.0, alpha=1.0, b=0.0))\n    except ValueError:\n        return 1\n    except Exception:\n        return 0\n    return 0",
         "call": "_c(locate_periodic_orbit)", "gold_call": "_c(_oracle_locate_periodic_orbit)"},   # exception contract -- a starting separation of 0.5 A sits deep inside the repulsive wall, far below the 3.55-3.75 A window where the orbit lives, so the return map has no real turning point there and the iteration cannot converge; do not replace this guess with a convergent one
    ]
