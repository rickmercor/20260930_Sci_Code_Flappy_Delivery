"""
Builds the smooth explicit predictor of one time step: it assembles the tridiagonal bulk-plus-secant-interface stiffness for the current damage state, predicts the displacement half a step ahead with no contact correction, assembles the cap-regime traction force on interfaces that are still below the crossover damage, and forms the contact-free velocity that later feeds the contact solve.

Smooth predictor and cap force at the predicted configuration.




Every interface contributes to the assembled system in one of two ways depending on its current damage: below the crossover damage it contributes no stiffness at all and instead an external-side traction pair that pulls its open faces together, while at or above the crossover damage it contributes an ordinary secant spring between its two duplicated faces, so the same interface can be a force source at one instant and a stiffness contributor at another as its damage grows past the crossover. The explicit predictor advances displacement using only the previous step's velocity and acceleration, ignoring the contact constraint entirely; evaluating the assembled stiffness and the cap force at this predicted, contact-free configuration gives a provisional acceleration that never itself becomes part of the stored state, because it only exists to build the contact-free velocity that the following contact solve corrects. No external load is ever applied after release, so the cap traction on interfaces still in their zero-stiffness regime is the only source term besides the bulk elastic restoring force.

Returns
-------
result : dict -- u_tilde, v_free, fcap, k_diag, k_up, delta_pred (see docstring).
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def predictor_forces(u, v, a, damage, dof_minus, dof_plus, element_dofs, mass, k_e: float, d_tilde: float, dt: float, alpha: float = 10.0, sigma_c: float = 262e6, Gc: float = 50.0, area: float = 6.45e-4) -> dict:
    """Assemble K(d), the smooth predictor, and the cap-regime force.

    Parameters
    ----------
    u, v, a : array_like
        Current displacement, velocity, and acceleration at every degree of
        freedom (same shape, length n_dof).
    damage : array_like
        Current damage of each interface, in [0, 1] (length n_if).
    dof_minus, dof_plus : array_like
        Minus-face and plus-face degree-of-freedom index of each interface
        (length n_if, integer).
    element_dofs : array_like
        (n_bulk, 2) integer array, the (left, right) degree-of-freedom index
        of every bulk element.
    mass : array_like
        Lumped mass at every degree of freedom (length n_dof, all > 0).
    k_e : float
        Bulk element spring constant (must be > 0).
    d_tilde : float
        Crossover damage separating the cap regime from the secant regime
        (must be in (0, 1)).
    dt : float
        Time step (must be > 0).
    alpha : float
        Interface stiffness cap ratio, dimensionless (must be > 0); the
        assembled secant spring at damage d is (1-d)/d * sigma_c/delta_c *
        area, with delta_c = 2*Gc/sigma_c.
    sigma_c : float
        Cohesive strength, in pascals (must be > 0).
    Gc : float
        Specific fracture energy, in J/m^2 (must be > 0).
    area : float
        Cross-sectional area, in m^2 (must be > 0).

    Returns
    -------
    result : dict
        u_tilde : ndarray, the predicted displacement (length n_dof).
        v_free : ndarray, the contact-free velocity (length n_dof).
        fcap : ndarray, the cap-regime traction force (length n_dof).
        k_diag : ndarray, diagonal of the assembled stiffness (length n_dof).
        k_up : ndarray, superdiagonal of the assembled stiffness (length
            n_dof - 1).
        delta_pred : ndarray, predicted opening of each interface (length
            n_if).

    Raises
    ------
    ValueError
        If u, v, and a do not share the same shape, if dt or k_e is not
        positive, or if alpha, sigma_c, or Gc is not a positive real number.
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _mulK(kdiag, kup, x):
    """Tridiagonal matrix-vector product from a (diagonal, superdiagonal)
    pair, exploiting the assembled stiffness's symmetric banded shape."""
    y = kdiag * x
    y[:-1] += kup * x[1:]
    y[1:] += kup * x[:-1]
    return y


def _oracle_predictor_forces(u, v, a, damage, dof_minus, dof_plus, element_dofs, mass, k_e: float, d_tilde: float, dt: float, alpha: float = 10.0, sigma_c: float = 262e6, Gc: float = 50.0, area: float = 6.45e-4) -> dict:
    """Reference implementation of predictor_forces."""
    u = np.asarray(u, dtype=float)
    v = np.asarray(v, dtype=float)
    a = np.asarray(a, dtype=float)
    if not (u.shape == v.shape == a.shape):
        raise ValueError("u, v, and a must share the same shape")
    if not (dt > 0.0 and k_e > 0.0):
        raise ValueError("dt and k_e must be positive")
    if not (alpha > 0.0 and sigma_c > 0.0 and Gc > 0.0):
        raise ValueError("alpha, sigma_c, and Gc must be positive")
    if not (0.0 < d_tilde < 1.0):
        raise ValueError("d_tilde must lie in (0, 1)")

    mass = np.asarray(mass, dtype=float)
    damage = np.asarray(damage, dtype=float)
    dof_minus = np.asarray(dof_minus, dtype=int)
    dof_plus = np.asarray(dof_plus, dtype=int)
    element_dofs = np.asarray(element_dofs, dtype=int)
    n_dof = u.shape[0]
    delta_c = 2.0 * Gc / sigma_c

    u_tilde = u + dt * v + 0.5 * dt * dt * a

    kdiag = np.zeros(n_dof)
    kup = np.zeros(n_dof - 1)
    for i, j in element_dofs:
        kdiag[i] += k_e
        kdiag[j] += k_e
        kup[i] += -k_e
    sec = (damage >= d_tilde) & (damage < 1.0)
    if sec.any():
        ks = (1.0 - damage[sec]) / damage[sec] * (sigma_c / delta_c) * area
        np.add.at(kdiag, dof_minus[sec], ks)
        np.add.at(kdiag, dof_plus[sec], ks)
        np.add.at(kup, dof_minus[sec], -ks)

    delta_pred = u_tilde[dof_plus] - u_tilde[dof_minus]
    fcap = np.zeros(n_dof)
    cap_sel = (delta_pred > 0.0) & (damage < d_tilde)
    if cap_sel.any():
        f = area * sigma_c * (1.0 - damage[cap_sel])
        np.add.at(fcap, dof_plus[cap_sel], -f)
        np.add.at(fcap, dof_minus[cap_sel], +f)

    K_ut = _mulK(kdiag, kup, u_tilde)
    v_free = v + 0.5 * dt * a + 0.5 * dt * (fcap - K_ut) / mass

    return {
        "u_tilde": u_tilde,
        "v_free": v_free,
        "fcap": fcap,
        "k_diag": kdiag,
        "k_up": kup,
        "delta_pred": delta_pred,
    }

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications.

    Uses a small 3-element, 1-interface unit fixture (unit masses and
    springs, benchmark cap-regime parameters) so cases stay mutation-cheap.
    >= 3 cases: signed component checks for a normal cap-regime opening
    step, a boundary case where the interface has already crossed into the
    secant regime, and two invalid-input edge cases (mismatched shapes,
    non-positive dt).
    """
    fixture = (
        "import numpy as np\n"
        "u0 = np.array([0.0, 0.0, 0.0, 0.0])\n"
        "v0 = np.array([-1.0, -0.1, 0.1, 1.0])\n"
        "a0 = np.zeros(4)\n"
        "dof_minus = np.array([1]); dof_plus = np.array([2])\n"
        "element_dofs = np.array([[0, 1], [2, 3]])\n"
        "mass = np.array([1.0, 1.0, 1.0, 1.0])\n"
        "k_e = 1.0; dt = 0.05\n"
    )
    return [
        {
            # normal: cap force acts positively on the minus face.
            "setup": fixture,
            "call": (
                "float(predictor_forces(u0, v0, a0, np.array([1e-6]), dof_minus, dof_plus, "
                "element_dofs, mass, k_e, 3.709110e-4, dt)['fcap'][1])"
            ),
            "gold_call": (
                "float(_oracle_predictor_forces(u0, v0, a0, np.array([1e-6]), dof_minus, dof_plus, "
                "element_dofs, mass, k_e, 3.709110e-4, dt)['fcap'][1])"
            ),
        },
        {
            # normal: cap force acts negatively on the plus face.
            "setup": fixture,
            "call": (
                "float(predictor_forces(u0, v0, a0, np.array([1e-6]), dof_minus, dof_plus, "
                "element_dofs, mass, k_e, 3.709110e-4, dt)['fcap'][2])"
            ),
            "gold_call": (
                "float(_oracle_predictor_forces(u0, v0, a0, np.array([1e-6]), dof_minus, dof_plus, "
                "element_dofs, mass, k_e, 3.709110e-4, dt)['fcap'][2])"
            ),
        },
        {
            # boundary: damage already past the crossover -- interface
            # contributes a secant spring instead of the cap force
            "setup": fixture,
            "call": (
                "float(predictor_forces(u0, v0, a0, np.array([0.5]), dof_minus, dof_plus, "
                "element_dofs, mass, k_e, 3.709110e-4, dt)['k_diag'][1]) "
                "+ float(np.sum(np.abs(predictor_forces(u0, v0, a0, np.array([0.5]), dof_minus, dof_plus, "
                "element_dofs, mass, k_e, 3.709110e-4, dt)['fcap'])))"
            ),
            "gold_call": (
                "float(_oracle_predictor_forces(u0, v0, a0, np.array([0.5]), dof_minus, dof_plus, "
                "element_dofs, mass, k_e, 3.709110e-4, dt)['k_diag'][1]) "
                "+ float(np.sum(np.abs(_oracle_predictor_forces(u0, v0, a0, np.array([0.5]), dof_minus, dof_plus, "
                "element_dofs, mass, k_e, 3.709110e-4, dt)['fcap'])))"
            ),
        },
        {
            # edge: mismatched u/v shapes are invalid
            "setup": fixture + """
def _guard(thunk):
    try:
        thunk()
        return 0
    except ValueError:
        return 2
    except Exception:
        return 1""",
            "call": "_guard(lambda: predictor_forces(u0, v0[:3], a0, np.array([1e-6]), dof_minus, dof_plus, element_dofs, mass, k_e, 3.709110e-4, dt))",
            "gold_call": "_guard(lambda: _oracle_predictor_forces(u0, v0[:3], a0, np.array([1e-6]), dof_minus, dof_plus, element_dofs, mass, k_e, 3.709110e-4, dt))",
        },
        {
            # edge: a non-positive time step is invalid
            "setup": fixture + """
def _guard(thunk):
    try:
        thunk()
        return 0
    except ValueError:
        return 2
    except Exception:
        return 1""",
            "call": "_guard(lambda: predictor_forces(u0, v0, a0, np.array([1e-6]), dof_minus, dof_plus, element_dofs, mass, k_e, 3.709110e-4, 0.0))",
            "gold_call": "_guard(lambda: _oracle_predictor_forces(u0, v0, a0, np.array([1e-6]), dof_minus, dof_plus, element_dofs, mass, k_e, 3.709110e-4, 0.0))",
        },
    ]
