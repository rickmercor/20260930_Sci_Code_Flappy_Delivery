"""
Drive the fixed strain path of the configuration twice, once under each strength criterion, running the source's staggered update at every increment: resolve the point-local return mapping at the phase-field value carried in from the previous increment, build the consistent tangent, apply the source's irreversibility rule to the crack driving force, then update the phase field. Record one row per increment and per criterion. The last column is the norm of the difference between the actual stress increment and the increment the consistent tangent predicts.

The source solves displacement and phase field in a staggered loop, holding one field while the other is updated. Eq. (11) makes the crack driving force a running maximum so that the degradation can never be undone, which is what makes unloading behave differently from loading. The gap between the tangent's predicted stress increment and the delivered one measures how consistent the tangent of Eq. (30) actually is, which is the property the source claims for it.

Returns
-------
A (14, 8) float64 array. Rows 0 to 6 are the seven increments under the smooth criterion and rows 7 to 13 the same increments under the non-smooth criterion. The columns are [magnitude 1, magnitude 2, stress norm, potential value, driving force after irreversibility, phase field, tangent norm, tangent increment error].
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def cohesive_audit(load_scale: float) -> "np.ndarray":
    """Drive the fixed strain path of the configuration twice, once under each strength
    criterion, running the source's staggered update at every increment: resolve the point-
    local return mapping at the phase-field value carried in from the previous increment,
    build the consistent tangent, apply the source's irreversibility rule to the crack
    driving force, then update the phase field, and record one row per increment and per
    criterion.

    Args:
        load_scale: The single free parameter multiplying every strain increment of the
            configuration's fixed path, positive.

    Returns:
        A (14, 8) float64 array. Rows 0 to 6 are the seven increments under the smooth
        criterion and rows 7 to 13 the same increments under the non-smooth criterion. The
        columns are [magnitude 1, magnitude 2, stress norm, potential value, driving force
        after irreversibility, phase field, tangent norm, tangent increment error].

    Raises:
        ValueError: If load_scale is not a positive finite number.
    """
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _pack(t: "np.ndarray") -> "np.ndarray":
    """Tensor components [xx, yy, zz, yz, xz, xy] -> the source's six-component column."""
    r2 = np.sqrt(2.0)
    return np.array([t[0], t[1], t[2], r2 * t[3], r2 * t[4], r2 * t[5]], dtype=float)


def _configuration() -> "tuple[np.ndarray, np.ndarray, dict]":
    """The fixed strain path (tensor components), the supplied Laplacians and the material data."""
    deps = np.array([
        [ 6.0e-4, -2.0e-4, 0.0, 0.0,    0.0,  6.0e-4],
        [-1.8e-3, -1.1e-3, 0.0, 0.0,    0.0,  8.0e-4],
        [-6.0e-4, -4.0e-4, 0.0, 0.0,    0.0,  5.0e-4],
        [ 4.0e-4,  3.0e-4, 0.0, 0.0,    0.0, -1.1e-3],
        [ 1.3e-3,  9.0e-4, 0.0, 2.0e-4, 0.0,  4.0e-4],
        [ 9.0e-4,  7.0e-4, 0.0, 0.0,    0.0,  5.0e-4],
        [ 1.1e-3,  8.0e-4, 0.0, 0.0,    0.0,  6.0e-4],
    ])
    laps = np.array([0.0, 10.0, 20.0, 15.0, 40.0, 25.0, 30.0])
    pars = dict(E=200.0, nu=0.3, ft=0.150, fs=0.150, eps_ref=0.01,
                kappa=1.0e-3, kappa_t=1.0e-9, Gc=1.0e-4, ell=0.05)
    return deps, laps, pars


def _oracle_cohesive_audit(load_scale: float) -> "np.ndarray":
    if not np.isfinite(load_scale) or load_scale <= 0.0:
        raise ValueError("load_scale must be a positive finite number")
    deps, laps, P = _configuration()
    D_el = _oracle_elastic_operator(P["E"], P["nu"])[:6, :]
    rows = []
    for crit in ("smooth", "faceted"):
        eps = np.zeros(6); phi = 0.0; Fh = 0.0; sig_prev = np.zeros(6)
        for n in range(deps.shape[0]):
            d_eps = load_scale * _pack(deps[n])
            eps = eps + d_eps
            if crit == "smooth":
                r = _oracle_smooth_return_map(eps, phi, P["E"], P["nu"], P["ft"], P["fs"],
                                              P["eps_ref"], P["kappa"], P["kappa_t"])
                lam = np.array([r[0], 0.0]); act = np.array([r[1], 0.0])
                Fd = float(r[2]); sig = r[9:15]
                dirs = np.vstack([_oracle_smooth_direction_gradient(
                    eps, r[3:9], P["ft"], P["fs"], P["eps_ref"])[0], np.zeros(6)])
            else:
                r = _oracle_faceted_return_map(eps, phi, P["E"], P["nu"], P["ft"], P["fs"],
                                               P["kappa"], P["kappa_t"])
                lam = r[0:2]; act = r[2:4]; sig = r[11:17]
                dirs = _oracle_faceted_directions(eps)
                eta = r[5:11]
                # Fd of Eq. (17a) is positively homogeneous of degree one in eta,
                # so Euler's identity recovers it exactly from its own gradient.
                Fd = float(_oracle_faceted_potential_gradients(eta, P["ft"], P["fs"])[0] @ eta)
            if float(np.max(act)) <= 0.0:
                Dt = D_el
            else:
                Dt = _oracle_consistent_tangent(dirs, act, P["E"], P["nu"], P["kappa_t"])
            Fh = max(Fd, Fh)
            c2 = _oracle_degradation_state(phi, P["kappa"])[2]
            phi = _oracle_phase_field_update(Fh, P["Gc"], P["ell"], c2, laps[n])
            gap = float(np.linalg.norm((sig - sig_prev) - Dt @ d_eps))
            sig_prev = sig
            rows.append([lam[0], lam[1], float(np.linalg.norm(sig)), Fd, Fh, phi,
                         float(np.linalg.norm(Dt)), gap])
    return np.array(rows)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {'setup': 'import numpy as np\nload_scale = 1.0\n',
         'call': 'cohesive_audit(load_scale)',
         'gold_call': '_oracle_cohesive_audit(load_scale)'},
        {'setup': 'import numpy as np\nload_scale = 0.5\n',
         'call': 'cohesive_audit(load_scale)',
         'gold_call': '_oracle_cohesive_audit(load_scale)'},
        {'setup': 'import numpy as np\nload_scale = 1.5\n',
         'call': 'cohesive_audit(load_scale)',
         'gold_call': '_oracle_cohesive_audit(load_scale)'},
        {'setup': 'import numpy as np\n# invalid input: a zero load scale is not positive and must raise ValueError\nload_scale = 0.0\ndef _catches_value_error(fn):\n    try:\n        fn()\n    except ValueError:\n        return np.array([1.0])\n    return np.array([0.0])\n',
         'call': '_catches_value_error(lambda: cohesive_audit(load_scale))',
         'gold_call': '_catches_value_error(lambda: _oracle_cohesive_audit(load_scale))'},
    ]
