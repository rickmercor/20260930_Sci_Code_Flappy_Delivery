"""
Folds the contact impulse into the state at the end of a time step: it turns the impulse into a velocity jump, corrects the predicted displacement with half of that jump, recomputes the acceleration at the corrected configuration, and forms the trapezoidal velocity update with the jump added on top.

Nonsmooth corrector: folding the contact impulse into the step.




The impulse solved for in the contact quadratic program is a velocity-level quantity, not a force, so it never enters the acceleration directly; instead it produces an instantaneous velocity jump at every degree of freedom the active interface pairs touch, obtained by scattering the impulse back through the same gap operator that built the contact problem and dividing by the lumped mass. Only half of that jump, scaled by the half-step trapezoidal weight, is needed to correct the predicted displacement, because the jump already represents the pairs' relative velocity being brought to rest and the displacement correction only has to account for the remaining half-step of travel at the corrected velocity. The acceleration stored at the end of the step is evaluated in equilibrium at this corrected displacement, never at the predicted one, and the stored velocity is the ordinary trapezoidal average of the old and new accelerations with the same velocity jump added on top, so the impulse's effect on the acceleration history is felt only indirectly, through the corrected configuration it produces.

Returns
-------
result : dict -- u_next, v_next, a_next, bv (see docstring).
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def nonsmooth_corrector(u_tilde, v, a, p, dof_minus, dof_plus, mass, k_diag, k_up, fcap, dt: float) -> dict:
    """Fold the contact impulse into the end-of-step state.

    Parameters
    ----------
    u_tilde : array_like
        Predicted displacement at every degree of freedom (length n_dof).
    v, a : array_like
        Velocity and acceleration at the start of the step (length n_dof
        each, same shape as u_tilde).
    p : array_like
        Normal impulse of every interface pair (length n_if; zero outside
        the active set).
    dof_minus, dof_plus : array_like
        Minus-face and plus-face degree-of-freedom index of each interface
        pair (length n_if, integer).
    mass : array_like
        Lumped mass at every degree of freedom (length n_dof, all > 0).
    k_diag : array_like
        Diagonal of the assembled stiffness (length n_dof).
    k_up : array_like
        Superdiagonal of the assembled stiffness (length n_dof - 1).
    fcap : array_like
        Cap-regime traction force at every degree of freedom (length
        n_dof).
    dt : float
        Time step (must be > 0).

    Returns
    -------
    result : dict
        u_next : ndarray, the corrected end-of-step displacement (length
            n_dof).
        v_next : ndarray, the end-of-step velocity (length n_dof).
        a_next : ndarray, the end-of-step acceleration (length n_dof).
        bv : ndarray, the velocity jump produced by the impulse (length
            n_dof).

    Raises
    ------
    ValueError
        If dt is not positive, or if u_tilde, v, and a do not share the
        same shape.
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


def _oracle_nonsmooth_corrector(u_tilde, v, a, p, dof_minus, dof_plus, mass, k_diag, k_up, fcap, dt: float) -> dict:
    """Reference implementation of nonsmooth_corrector."""
    u_tilde = np.asarray(u_tilde, dtype=float)
    v = np.asarray(v, dtype=float)
    a = np.asarray(a, dtype=float)
    if not (u_tilde.shape == v.shape == a.shape):
        raise ValueError("u_tilde, v, and a must share the same shape")
    if not (dt > 0.0):
        raise ValueError("dt must be positive")

    p = np.asarray(p, dtype=float)
    dof_minus = np.asarray(dof_minus, dtype=int)
    dof_plus = np.asarray(dof_plus, dtype=int)
    mass = np.asarray(mass, dtype=float)
    k_diag = np.asarray(k_diag, dtype=float)
    k_up = np.asarray(k_up, dtype=float)
    fcap = np.asarray(fcap, dtype=float)
    n_dof = u_tilde.shape[0]

    HTp = np.zeros(n_dof)
    np.add.at(HTp, dof_minus, -p)
    np.add.at(HTp, dof_plus, +p)
    bv = HTp / mass

    u_next = u_tilde + 0.5 * dt * bv
    a_next = (fcap - _mulK(k_diag, k_up, u_next)) / mass
    v_next = v + 0.5 * dt * (a + a_next) + bv

    return {"u_next": u_next, "v_next": v_next, "a_next": a_next, "bv": bv}

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications.

    Uses a small 4-dof unit fixture (unit masses, a mild bulk stiffness) so
    cases stay mutation-cheap. >= 3 cases: a normal scenario with a nonzero
    impulse on one pair with its jump and corrected state checked by
    component, a boundary case with zero impulse everywhere (should reduce
    to the smooth trapezoidal update with no jump), and two invalid-input
    edge cases (mismatched shapes, non-positive dt).
    """
    fixture = (
        "import numpy as np\n"
        "u_tilde = np.array([0.1, 0.2, 0.3, 0.4])\n"
        "v = np.array([-1.0, -0.1, 0.1, 1.0])\n"
        "a = np.array([0.0, 0.0, 0.0, 0.0])\n"
        "dof_minus = np.array([0, 2]); dof_plus = np.array([1, 3])\n"
        "mass = np.array([1.0, 1.0, 1.0, 1.0])\n"
        "k_diag = np.array([1.0, 1.0, 1.0, 1.0]); k_up = np.array([0.0, 0.0, 0.0])\n"
        "fcap = np.zeros(4)\n"
        "dt = 0.1\n"
    )
    return [
        {
            # normal: a nonzero impulse produces a signed minus-face jump.
            "setup": fixture,
            "call": (
                "float(nonsmooth_corrector(u_tilde, v, a, np.array([0.4, 0.0]), "
                "dof_minus, dof_plus, mass, k_diag, k_up, fcap, dt)['bv'][0])"
            ),
            "gold_call": (
                "float(_oracle_nonsmooth_corrector(u_tilde, v, a, np.array([0.4, 0.0]), "
                "dof_minus, dof_plus, mass, k_diag, k_up, fcap, dt)['bv'][0])"
            ),
        },
        {
            # normal: the same impulse corrects the minus-face displacement.
            "setup": fixture,
            "call": (
                "float(nonsmooth_corrector(u_tilde, v, a, np.array([0.4, 0.0]), "
                "dof_minus, dof_plus, mass, k_diag, k_up, fcap, dt)['u_next'][0])"
            ),
            "gold_call": (
                "float(_oracle_nonsmooth_corrector(u_tilde, v, a, np.array([0.4, 0.0]), "
                "dof_minus, dof_plus, mass, k_diag, k_up, fcap, dt)['u_next'][0])"
            ),
        },
        {
            # normal: the same impulse corrects the minus-face velocity.
            "setup": fixture,
            "call": (
                "float(nonsmooth_corrector(u_tilde, v, a, np.array([0.4, 0.0]), "
                "dof_minus, dof_plus, mass, k_diag, k_up, fcap, dt)['v_next'][0])"
            ),
            "gold_call": (
                "float(_oracle_nonsmooth_corrector(u_tilde, v, a, np.array([0.4, 0.0]), "
                "dof_minus, dof_plus, mass, k_diag, k_up, fcap, dt)['v_next'][0])"
            ),
        },
        {
            # boundary: zero impulse everywhere -- the jump vanishes and
            # u_next must equal u_tilde exactly.
            "setup": fixture,
            "call": (
                "float(np.max(np.abs(nonsmooth_corrector(u_tilde, v, a, np.array([0.0, 0.0]), "
                "dof_minus, dof_plus, mass, k_diag, k_up, fcap, dt)['u_next'] - u_tilde)))"
            ),
            "gold_call": (
                "float(np.max(np.abs(_oracle_nonsmooth_corrector(u_tilde, v, a, np.array([0.0, 0.0]), "
                "dof_minus, dof_plus, mass, k_diag, k_up, fcap, dt)['u_next'] - u_tilde)))"
            ),
        },
        {
            # edge: mismatched u_tilde/v shapes are invalid
            "setup": fixture + """
def _guard(thunk):
    try:
        thunk()
        return 0
    except ValueError:
        return 2
    except Exception:
        return 1""",
            "call": "_guard(lambda: nonsmooth_corrector(u_tilde, v[:3], a, np.array([0.0, 0.0]), dof_minus, dof_plus, mass, k_diag, k_up, fcap, dt))",
            "gold_call": "_guard(lambda: _oracle_nonsmooth_corrector(u_tilde, v[:3], a, np.array([0.0, 0.0]), dof_minus, dof_plus, mass, k_diag, k_up, fcap, dt))",
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
            "call": "_guard(lambda: nonsmooth_corrector(u_tilde, v, a, np.array([0.0, 0.0]), dof_minus, dof_plus, mass, k_diag, k_up, fcap, 0.0))",
            "gold_call": "_guard(lambda: _oracle_nonsmooth_corrector(u_tilde, v, a, np.array([0.0, 0.0]), dof_minus, dof_plus, mass, k_diag, k_up, fcap, 0.0))",
        },
    ]
