"""
Return the abbreviated action of the periodic orbit located from the supplied starting guess, that is the integral of the momenta against the coordinates taken once around the orbit. Evaluate it as the integral of twice the kinetic energy over one period.

The abbreviated action is the flux through the dividing surface the orbit spans, and it also distinguishes the wanted orbit from a wider companion branch of noticeably smaller action that a coarse search can land on instead. Integrate to a relative and absolute tolerance of 1e-10 or tighter; the orbit is strongly unstable and looser settings corrupt the quantities read off it.

Returns
-------
float: the abbreviated action of the orbit.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def orbit_action(r0: float, pr0: float, energy: float, params: dict) -> float:
    """Return the abbreviated action of the periodic orbit located from the supplied starting guess, that is the integral of the momenta against the coordinates taken once around the orbit. Evaluate it as the integral of twice the kinetic energy over one period.

    Parameters
    ----------
    r0 : float
        Positive starting separation for the iteration, in angstroms.
    pr0 : float
        Finite starting radial momentum for the iteration.
    energy : float
        Finite total energy held fixed, in kcal/mol.
    params : dict
        Mapping carrying ix, iz, m, de, re, c1, c2, ve, alpha and b.

    Returns
    -------
    action : float
        float: the abbreviated action of the orbit.

    Raises
    ------
    ValueError
        if the orbit cannot be located from the supplied guess.
    """
    return action

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

def _oracle_orbit_action(r0: float, pr0: float, energy: float, params: dict) -> float:
    if not np.isfinite(r0) or r0 <= 0.0:
        raise ValueError("r0 must be a positive finite number")
    if not np.isfinite(pr0):
        raise ValueError("pr0 must be finite")
    if not np.isfinite(energy):
        raise ValueError("energy must be finite")
    out = _oracle_locate_periodic_orbit(r0, pr0, energy, params)
    r, pr, T = out
    ix, m = params["ix"], params["m"]
    pth = _planar_pth(r, 0.0, pr, energy, params)
    def _aug(t, y, q):
        rr, th, p_r, p_th = y[:4]
        kin = p_th ** 2 * (1.0 / (2.0 * ix) + 1.0 / (2.0 * m * rr * rr)) + p_r * p_r / (2.0 * m)
        return np.concatenate([_planar_field(y[:4], q), [2.0 * kin]])
    s = solve_ivp(_aug, [0.0, T], [r, 0.0, pr, pth, 0.0], args=(params,),
                  method='DOP853', rtol=1e-12, atol=1e-12, max_step=0.5)
    return float(s.y[4, -1])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return the list of test-case specifications for this step."""
    return [
        {"tol": 1e-06, "setup": "import numpy as np",
         "call": "orbit_action(3.65, 0.0, 0.5, dict(ix=2.373409, iz=4.746818, m=0.9445, de=47.0, re=1.1, c1=7.37, c2=1.61, ve=55.0, alpha=1.0, b=0.0))",
         "gold_call": "_oracle_orbit_action(3.65, 0.0, 0.5, dict(ix=2.373409, iz=4.746818, m=0.9445, de=47.0, re=1.1, c1=7.37, c2=1.61, ve=55.0, alpha=1.0, b=0.0))"},   # normal
        {"tol": 1e-06, "setup": "import numpy as np",
         "call": "orbit_action(3.63, 0.10, 0.5, dict(ix=2.373409, iz=4.746818, m=0.9445, de=47.0, re=1.1, c1=7.37, c2=1.61, ve=55.0, alpha=1.0, b=0.30))",
         "gold_call": "_oracle_orbit_action(3.63, 0.10, 0.5, dict(ix=2.373409, iz=4.746818, m=0.9445, de=47.0, re=1.1, c1=7.37, c2=1.61, ve=55.0, alpha=1.0, b=0.30))"},   # normal
        {"tol": 1e-06, "setup": "import numpy as np",
         "call": "orbit_action(3.66, 0.0, 0.5, dict(ix=2.373409, iz=4.746818, m=0.9445, de=47.0, re=1.1, c1=7.37, c2=1.61, ve=55.0, alpha=1.0, b=0.0))",
         "gold_call": "_oracle_orbit_action(3.66, 0.0, 0.5, dict(ix=2.373409, iz=4.746818, m=0.9445, de=47.0, re=1.1, c1=7.37, c2=1.61, ve=55.0, alpha=1.0, b=0.0))"},   # boundary
        {"tol": 1e-06, "setup": "import numpy as np",
         "call": "orbit_action(3.60, 0.17, 0.5, dict(ix=2.373409, iz=4.746818, m=0.9445, de=47.0, re=1.1, c1=7.37, c2=1.61, ve=55.0, alpha=1.0, b=0.45))",
         "gold_call": "_oracle_orbit_action(3.60, 0.17, 0.5, dict(ix=2.373409, iz=4.746818, m=0.9445, de=47.0, re=1.1, c1=7.37, c2=1.61, ve=55.0, alpha=1.0, b=0.45))"},   # edge
        {"setup": "import numpy as np\ndef _c(fn):\n    try:\n        fn(0.5, 0.0, 0.5, dict(ix=2.373409, iz=4.746818, m=0.9445, de=47.0, re=1.1, c1=7.37, c2=1.61, ve=55.0, alpha=1.0, b=0.0))\n    except ValueError:\n        return 1\n    except Exception:\n        return 0\n    return 0",
         "call": "_c(orbit_action)", "gold_call": "_c(_oracle_orbit_action)"},   # exception contract -- same far guess as step 7: 0.5 A is inside the repulsive wall, so the orbit cannot be located and no action can be formed; do not replace this guess with a convergent one
    ]
