"""
The plane y = 0 with py = 0 is invariant under the flow. Working inside it, start from the point where the polar angle is zero (so the departing atom is on the positive z axis at separation r, with radial momentum pr), fix the remaining momentum from the requested total energy, and integrate until the polar angle has advanced by exactly two pi. Return the separation and radial momentum reached there, together with the elapsed time.

Inside the invariant plane the motion has two degrees of freedom. The orbit of interest circulates in the polar angle rather than librating, so a full turn of two pi is the natural section. Taking the section at zero polar angle rather than at a turning point keeps it transverse to the flow. The polar angle is measured within the plane from the positive z axis toward the positive x axis, so the atom leaves the axis with positive x-momentum (the positive root when that momentum is fixed from the energy); once the symmetry-breaking term is on, the two senses of circulation are not mirror images of each other, and the returned values refer to this sense. Integrate to a relative and absolute tolerance of 1e-10 or tighter; the orbit is strongly unstable and looser settings corrupt the quantities read off it.

Returns
-------
ndarray of shape (3,): the separation and radial momentum at the return, and the time taken in the model time unit.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def planar_return_map(r: float, pr: float, energy: float, params: dict) -> "np.ndarray":
    """The plane y = 0 with py = 0 is invariant under the flow. Working inside it, start from the point where the polar angle is zero (so the departing atom is on the positive z axis at separation r, with radial momentum pr), fix the remaining momentum from the requested total energy, and integrate until the polar angle has advanced by exactly two pi. Return the separation and radial momentum reached there, together with the elapsed time.

    Parameters
    ----------
    r : float
        Positive starting separation on the section, in angstroms.
    pr : float
        Finite starting radial momentum on the section.
    energy : float
        Total energy held fixed along the trajectory, in kcal/mol.
    params : dict
        Mapping carrying ix, iz, m, de, re, c1, c2, ve, alpha and b.

    Returns
    -------
    section_point : np.ndarray
        ndarray of shape (3,): the separation and radial momentum at the return, and the time taken in the model time unit.

    Raises
    ------
    ValueError
        if r is not positive and finite, if pr is not finite, if the requested energy is unattainable at that point, or if no full turn occurs within the time cap.
    """
    return section_point

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

def _oracle_planar_return_map(r: float, pr: float, energy: float, params: dict) -> "np.ndarray":
    if not np.isfinite(r) or r <= 0.0:
        raise ValueError("r must be a positive finite number")
    if not np.isfinite(pr):
        raise ValueError("pr must be finite")
    pth = _planar_pth(r, 0.0, pr, energy, params)
    ev = lambda t, y, q: y[1] - 2.0 * np.pi
    ev.terminal = True; ev.direction = 1.0
    s = solve_ivp(lambda t, y, q: _planar_field(y, q), [0.0, 30.0],
                  [r, 0.0, pr, pth], args=(params,), events=ev,
                  method='DOP853', rtol=1e-11, atol=1e-11)
    if not s.t_events[0].size:
        raise ValueError("the trajectory did not complete a full turn in theta")
    t = float(s.t_events[0][0]); y = s.y_events[0][0]
    return np.array([y[0], y[2], t], float)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return the list of test-case specifications for this step."""
    return [
        {"tol": 1e-06, "setup": "import numpy as np",
         "call": "planar_return_map(3.65072544, 0.0, 0.5, dict(ix=2.373409, iz=4.746818, m=0.9445, de=47.0, re=1.1, c1=7.37, c2=1.61, ve=55.0, alpha=1.0, b=0.0))",
         "gold_call": "_oracle_planar_return_map(3.65072544, 0.0, 0.5, dict(ix=2.373409, iz=4.746818, m=0.9445, de=47.0, re=1.1, c1=7.37, c2=1.61, ve=55.0, alpha=1.0, b=0.0))"},   # normal
        {"tol": 1e-06, "setup": "import numpy as np",
         "call": "planar_return_map(3.6, 0.05, 0.5, dict(ix=2.373409, iz=4.746818, m=0.9445, de=47.0, re=1.1, c1=7.37, c2=1.61, ve=55.0, alpha=1.0, b=0.0))",
         "gold_call": "_oracle_planar_return_map(3.6, 0.05, 0.5, dict(ix=2.373409, iz=4.746818, m=0.9445, de=47.0, re=1.1, c1=7.37, c2=1.61, ve=55.0, alpha=1.0, b=0.0))"},   # normal
        {"tol": 1e-06, "setup": "import numpy as np",
         "call": "planar_return_map(3.63193607, 0.10566843, 0.5, dict(ix=2.373409, iz=4.746818, m=0.9445, de=47.0, re=1.1, c1=7.37, c2=1.61, ve=55.0, alpha=1.0, b=0.30))",
         "gold_call": "_oracle_planar_return_map(3.63193607, 0.10566843, 0.5, dict(ix=2.373409, iz=4.746818, m=0.9445, de=47.0, re=1.1, c1=7.37, c2=1.61, ve=55.0, alpha=1.0, b=0.30))"},   # boundary
        {"tol": 1e-06, "setup": "import numpy as np",
         "call": "planar_return_map(3.5, 0.0, 1.1, dict(ix=2.373409, iz=4.746818, m=0.9445, de=47.0, re=1.1, c1=7.37, c2=1.61, ve=55.0, alpha=1.0, b=0.30))",
         "gold_call": "_oracle_planar_return_map(3.5, 0.0, 1.1, dict(ix=2.373409, iz=4.746818, m=0.9445, de=47.0, re=1.1, c1=7.37, c2=1.61, ve=55.0, alpha=1.0, b=0.30))"},   # edge
        {"setup": "import numpy as np\ndef _c(fn):\n    try:\n        fn(-3.6, 0.0, 0.5, dict(ix=2.373409, iz=4.746818, m=0.9445, de=47.0, re=1.1, c1=7.37, c2=1.61, ve=55.0, alpha=1.0, b=0.0))\n    except ValueError:\n        return 1\n    except Exception:\n        return 0\n    return 0",
         "call": "_c(planar_return_map)", "gold_call": "_c(_oracle_planar_return_map)"},   # exception contract
    ]
