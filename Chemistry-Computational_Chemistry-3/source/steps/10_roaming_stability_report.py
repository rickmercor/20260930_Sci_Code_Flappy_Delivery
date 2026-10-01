"""
Run the whole chain for the model at the requested total energy and symmetry-breaking strength. Use the published parameter set throughout: perpendicular moment of inertia 2.373409 and axial moment 4.746818 in unified atomic mass units times square angstroms, reduced mass 0.9445 in unified atomic mass units, and for the potential de = 47.0 kcal/mol, re = 1.1 angstroms, c1 = 7.37, c2 = 1.61, ve = 55.0 kcal/mol and alpha = 1.0 inverse square angstroms. Locate the orbit at zero symmetry-breaking and an energy of 0.5 by scanning starting separations between 3.55 and 3.75, taking the first separation from which the iteration converges, then follow it by continuation in n_steps equal increments first in the symmetry-breaking strength at that energy and then in the energy at the requested strength. At the end point compute the period, the abbreviated action, and the one-period matrix; split that matrix into the block acting on the out-of-plane pair (the second position component and its conjugate momentum) and the block acting on the four in-plane components. Return the trace of the out-of-plane block first, then the orbit data.

The continuation is what keeps the search on one branch: several unstable orbits coexist at the end point, and a scan that takes the first separation from which the iteration converges latches a wider companion branch, so the orbit is followed from the anchor instead. A single continuation increment overshoots onto that wide branch; two or more increments do not. The out-of-plane block is the one that decides whether a trajectory nudged off the invariant plane returns to it or leaves. The orbit is traversed in the sense of the return map, with the polar angle increasing from the positive z axis toward the positive x axis, which fixes the sign of the reported radial momentum. Integrate every trajectory and every variational system in this chain to a relative and absolute tolerance of 1e-10 or tighter: the orbit is strongly unstable and looser settings corrupt the multipliers.

Returns
-------
ndarray of shape (6,): the trace of the out-of-plane block, the separation and radial momentum of the orbit, its period, its abbreviated action, and the largest absolute in-plane multiplier.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def roaming_stability_report(energy: float, b: float, n_steps: int) -> "np.ndarray":
    """Run the whole chain for the model at the requested total energy and symmetry-breaking strength. Use the published parameter set throughout: perpendicular moment of inertia 2.373409 and axial moment 4.746818 in unified atomic mass units times square angstroms, reduced mass 0.9445 in unified atomic mass units, and for the potential de = 47.0 kcal/mol, re = 1.1 angstroms, c1 = 7.37, c2 = 1.61, ve = 55.0 kcal/mol and alpha = 1.0 inverse square angstroms. Locate the orbit at zero symmetry-breaking and an energy of 0.5 by scanning starting separations between 3.55 and 3.75, taking the first separation from which the iteration converges, then follow it by continuation in n_steps equal increments first in the symmetry-breaking strength at that energy and then in the energy at the requested strength. At the end point compute the period, the abbreviated action, and the one-period matrix; split that matrix into the block acting on the out-of-plane pair (the second position component and its conjugate momentum) and the block acting on the four in-plane components. Return the trace of the out-of-plane block first, then the orbit data.

    Parameters
    ----------
    energy : float
        Positive total energy at the end point, in kcal/mol.
    b : float
        Non-negative symmetry-breaking strength at the end point.
    n_steps : int
        Number of equal continuation increments, at least one.

    Returns
    -------
    report : np.ndarray
        ndarray of shape (6,): the trace of the out-of-plane block, the separation and radial momentum of the orbit, its period, its abbreviated action, and the largest absolute in-plane multiplier.

    Raises
    ------
    ValueError
        if the energy is not positive and finite, if b is negative or not finite, if n_steps is below one, if the anchor orbit cannot be located, or if any continuation increment fails to converge.
    """
    return report

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

def _oracle_roaming_stability_report(energy: float, b: float, n_steps: int) -> "np.ndarray":
    if not np.isfinite(energy) or energy <= 0.0:
        raise ValueError("energy must be a positive finite number")
    if not np.isfinite(b) or b < 0.0:
        raise ValueError("b must be a non-negative finite number")
    if int(n_steps) < 1:
        raise ValueError("n_steps must be at least one")
    ix = 2.373409
    base = dict(ix=ix, iz=2.0 * ix, m=0.9445, de=47.0, re=1.1,
                c1=7.37, c2=1.61, ve=55.0, alpha=1.0, b=0.0)
    # the FR1 seed at the anchor point E = 0.5, b = 0 is found by a bracketed scan
    seed = None
    for rr in np.linspace(3.55, 3.75, 41):
        try:
            out = _oracle_locate_periodic_orbit(rr, 0.0, 0.5, base)
        except ValueError:
            continue
        if 3.0 < out[0] < 3.9:
            seed = np.array([out[0], out[1]]); break
    if seed is None:
        raise ValueError("could not locate the FR1 anchor orbit")
    # continue first in b at E = 0.5, then in energy at the requested b
    ns = int(n_steps)
    for bb in np.linspace(0.0, float(b), ns + 1)[1:]:
        params = dict(base); params["b"] = float(bb)
        o = _oracle_locate_periodic_orbit(seed[0], seed[1], 0.5, params)
        seed = np.array([o[0], o[1]])
    params = dict(base); params["b"] = float(b)
    for ee in np.linspace(0.5, float(energy), ns + 1)[1:]:
        o = _oracle_locate_periodic_orbit(seed[0], seed[1], float(ee), params)
        seed = np.array([o[0], o[1]])
    orb = _oracle_locate_periodic_orbit(seed[0], seed[1], energy, params)
    r0, pr0, T = orb
    # the located point must be a genuine fixed point of the return map
    back = _oracle_planar_return_map(r0, pr0, energy, params)
    if abs(back[0] - r0) > 1e-8 or abs(back[1] - pr0) > 1e-8:
        raise ValueError("the continuation did not end on a periodic orbit")
    W = _oracle_orbit_action(r0, pr0, energy, params)
    # rebuild the Cartesian state and check it against the energy and the potentials
    coeff = _oracle_extract_radial_parameters(params["de"], params["re"], params["c1"], params["c2"])
    if not np.isfinite(coeff[0]):
        raise ValueError("the radial coefficients are not finite")
    pth = _planar_pth(r0, 0.0, pr0, energy, params)
    y0 = np.array([0.0, 0.0, r0, pth / r0, 0.0, pr0])
    vrad = _oracle_radial_potential(r0, params["de"], params["re"], params["c1"], params["c2"])
    vang = _oracle_angular_potential(y0[0], y0[1], y0[2], params["ve"], params["alpha"],
                                     params["re"], params["b"])
    if not np.isfinite(float(vrad) + float(vang)):
        raise ValueError("the potential is not finite on the orbit")
    if abs(_oracle_hamiltonian(y0, params) - energy) > 1e-8:
        raise ValueError("the reconstructed state does not sit at the requested energy")
    if abs(_oracle_vector_field(y0, params)[1]) > 1e-12:
        raise ValueError("the reaction plane is not invariant at the start of the orbit")
    M = _oracle_monodromy_matrix(y0, params, T)
    mt = M[np.ix_([1, 4], [1, 4])]
    inp = M[np.ix_([0, 2, 3, 5], [0, 2, 3, 5])]
    trace = float(np.trace(mt))
    lam_in = float(np.max(np.abs(np.linalg.eigvals(inp))))
    return np.array([trace, r0, pr0, T, W, lam_in], float)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return the list of test-case specifications for this step."""
    return [
        {"tol": 1e-06, "setup": "import numpy as np",
         "call": "roaming_stability_report(0.5, 0.0, 4)",
         "gold_call": "_oracle_roaming_stability_report(0.5, 0.0, 4)"},   # normal
        {"tol": 1e-06, "setup": "import numpy as np",
         "call": "roaming_stability_report(0.5, 0.30, 6)",
         "gold_call": "_oracle_roaming_stability_report(0.5, 0.30, 6)"},   # normal
        {"tol": 1e-06, "setup": "import numpy as np",
         "call": "roaming_stability_report(0.8, 0.20, 8)",
         "gold_call": "_oracle_roaming_stability_report(0.8, 0.20, 8)"},   # boundary
        {"tol": 1e-06, "setup": "import numpy as np",
         "call": "roaming_stability_report(1.10, 0.45, 12)",
         "gold_call": "_oracle_roaming_stability_report(1.10, 0.45, 12)"},   # edge
        {"setup": "import numpy as np\ndef _c(fn):\n    try:\n        fn(-0.5, 0.0, 4)\n    except ValueError:\n        return 1\n    except Exception:\n        return 0\n    return 0",
         "call": "_c(roaming_stability_report)", "gold_call": "_c(_oracle_roaming_stability_report)"},   # exception contract
    ]
