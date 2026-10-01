"""
Solves the velocity-level contact problem for one time step: it forms the gap and active set from the predicted displacement, assembles the modified Delassus operator that already accounts for the smooth dynamics' own stiffness coupling, and returns the normal impulse each active interface pair needs to satisfy the prescribed restitution relation.

Velocity-level contact solve via a convex quadratic program.




Once a pair of duplicated interface faces has closed to zero gap or less, an impulsive constraint force is the only way to keep the two faces from interpenetrating on the next half step, and the right notion of "the two faces' relative approach velocity" for that force to act on combines the contact-free velocity with the restitution-scaled velocity at the beginning of the step, because the impulse itself perturbs the acceleration through the stiffness that couples every degree of freedom together; the modified Delassus operator folds that stiffness coupling into the effective inertia the contact solve sees, so solving against it produces an impulse consistent with the actual half-step dynamics rather than a naively decoupled one. Whenever this operator is symmetric positive semidefinite, which the same stability bound used to size the time step also guarantees, enforcing the prescribed restitution relation at every active pair is equivalent to minimizing a convex quadratic in the impulse subject to nonnegativity, and an active-set method solves this by alternating between an equality solve on the currently unconstrained pairs and moving any activated or newly-violating pair between the free and blocked sets until every unconstrained pair's residual is nonnegative to within a velocity-scaled tolerance; the benchmark's zero restitution reduces this relation to perfectly inelastic contact.

Returns
-------
result : dict -- p, gap, active, residual (see docstring).
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def contact_qp_solve(dof_minus, dof_plus, mass, k_diag, k_up, u_tilde, v_free, v_n, dt: float, n_dof: int, e_restitution: float = 0.0) -> dict:
    """Solve the velocity-level contact QP for the active interface pairs.

    Parameters
    ----------
    dof_minus, dof_plus : array_like
        Minus-face and plus-face degree-of-freedom index of each interface
        pair (length n_if, integer).
    mass : array_like
        Lumped mass at every degree of freedom (length n_dof, all > 0).
    k_diag : array_like
        Diagonal of the assembled stiffness at the predicted configuration
        (length n_dof).
    k_up : array_like
        Superdiagonal of the assembled stiffness (length n_dof - 1).
    u_tilde : array_like
        Predicted displacement at every degree of freedom (length n_dof).
    v_free : array_like
        Contact-free velocity at every degree of freedom (length n_dof).
    v_n : array_like
        Velocity at the beginning of the current time step (length n_dof).
    dt : float
        Time step (must be > 0).
    n_dof : int
        Total number of degrees of freedom (must be >= 2).
    e_restitution : float
        Restitution coefficient, dimensionless (must be >= 0; 0 is
        perfectly inelastic contact).

    Returns
    -------
    result : dict
        p : ndarray, the normal impulse of every interface pair (length
            n_if; zero on pairs outside the active set).
        gap : ndarray, the signed opening of every interface pair at the
            predicted configuration (length n_if).
        active : ndarray, integer indices of the pairs in the active set.
        residual : float, the worst dual-feasibility residual among the
            inactive pairs at the solution (0.0 when the active set is
            empty).

    Raises
    ------
    ValueError
        If dt is not positive, if e_restitution is negative, or if n_dof is
        smaller than 2.
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


def _active_set_qp(W, b):
    """Solve the nonnegative QP and return only after validating its KKT conditions."""
    n = len(b)
    p = np.zeros(n)
    if n == 0:
        return p, 0.0
    free = b < -1e-12 * max(1.0, float(np.max(np.abs(b))))
    for _ in range(600):
        fidx = np.flatnonzero(free)
        candidate = np.zeros(n)
        if fidx.size:
            candidate[fidx] = np.linalg.solve(W[np.ix_(fidx, fidx)], -b[fidx])
        scale = max(1.0, float(np.max(np.abs(b))), float(np.max(np.abs(W @ p))))
        tolerance = 1e-12 * scale
        negative = candidate[fidx] < -tolerance
        if negative.any():
            direction = candidate - p
            ratios = p[fidx][negative] / -direction[fidx][negative]
            step = float(np.min(ratios))
            p += step * direction
            p[np.abs(p) <= tolerance] = 0.0
            blocking = fidx[negative][ratios <= step * (1.0 + 1e-12) + 1e-15]
            free[blocking] = False
            continue
        p = np.maximum(candidate, 0.0)
        g = W @ p + b
        scale = max(1.0, float(np.max(np.abs(b))), float(np.max(np.abs(W @ p))))
        tolerance = 1e-12 * scale
        bound = np.flatnonzero(~free)
        primal_ok = float(np.min(p)) >= -tolerance
        dual_ok = float(np.min(g)) >= -tolerance
        complementarity_ok = float(np.max(np.abs(p * g))) <= tolerance * max(
            1.0, float(np.max(np.abs(p)))
        )
        if primal_ok and dual_ok and complementarity_ok:
            return p, (float(np.min(g[bound])) if bound.size else 0.0)
        violated = bound[g[bound] < -tolerance]
        if violated.size:
            free[violated[np.argmin(g[violated])]] = True
            continue
        raise RuntimeError("active-set QP candidate failed KKT validation")
    raise RuntimeError("active-set QP did not converge within 600 iterations")


def _oracle_contact_qp_solve(dof_minus, dof_plus, mass, k_diag, k_up, u_tilde, v_free, v_n, dt: float, n_dof: int, e_restitution: float = 0.0) -> dict:
    """Reference implementation of contact_qp_solve."""
    if not (dt > 0.0):
        raise ValueError("dt must be positive")
    if e_restitution < 0.0:
        raise ValueError("e_restitution must be nonnegative")
    if not isinstance(n_dof, (int, np.integer)) or int(n_dof) < 2:
        raise ValueError("n_dof must be an integer >= 2")

    dof_minus = np.asarray(dof_minus, dtype=int)
    dof_plus = np.asarray(dof_plus, dtype=int)
    mass = np.asarray(mass, dtype=float)
    k_diag = np.asarray(k_diag, dtype=float)
    k_up = np.asarray(k_up, dtype=float)
    u_tilde = np.asarray(u_tilde, dtype=float)
    v_free = np.asarray(v_free, dtype=float)
    v_n = np.asarray(v_n, dtype=float)
    n_dof = int(n_dof)
    n_if = dof_minus.shape[0]

    H = np.zeros((n_if, n_dof))
    H[np.arange(n_if), dof_minus] = -1.0
    H[np.arange(n_if), dof_plus] = +1.0
    gap = H @ u_tilde
    active = np.flatnonzero(gap <= 0.0)

    p = np.zeros(n_if)
    residual = 0.0
    if active.size:
        HA = H[active]
        rhs_v = v_free + e_restitution * v_n
        b = HA @ rhs_v
        Z = HA / mass[None, :]
        KZ = np.empty_like(Z)
        for r in range(Z.shape[0]):
            KZ[r] = _mulK(k_diag, k_up, Z[r])
        W = Z @ HA.T - (dt * dt / 4.0) * (Z @ KZ.T)
        W = 0.5 * (W + W.T)
        p_active, residual = _active_set_qp(W, b)
        p[active] = p_active

    return {"p": p, "gap": gap, "active": active, "residual": residual}

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications.

    Uses a small 2-interface unit fixture (unit masses, a mild bulk
    stiffness) so cases stay mutation-cheap. >= 3 cases: a normal scenario
    with a closing pair (active) and an opening pair (inactive), a positive-
    restitution regression whose gold_call mirrors the call with the oracle
    name, a boundary case where every pair starts at exactly zero gap, and
    two invalid-input edge cases (non-positive dt, negative restitution).
    """
    fixture = (
        "import numpy as np\n"
        "dof_minus = np.array([0, 2]); dof_plus = np.array([1, 3])\n"
        "mass = np.array([1.0, 1.0, 1.0, 1.0])\n"
        "k_diag = np.array([1.0, 1.0, 1.0, 1.0]); k_up = np.array([0.0, 0.0, 0.0])\n"
        "v_n = np.array([0.1, -0.1, 0.4, -0.4])\n"
        "dt = 0.1\n"
    )
    coupled_fixture = (
        "import numpy as np\n"
        "dof_minus = np.array([0, 1]); dof_plus = np.array([1, 2])\n"
        "mass = 1.0 / np.array([1.0199, 0.49005, 1.0199])\n"
        "k_diag = np.zeros(3); k_up = np.zeros(2)\n"
        "u_tilde = np.zeros(3); v_free = np.array([0.0, -0.1, 0.4])\n"
        "v_n = np.zeros(3); dt = 0.1\n"
        "W = np.array([[1.50995, -0.49005], [-0.49005, 1.50995]])\n"
        "b = np.array([-0.1, 0.5])\n"
    )
    return [
        {
            # coupled-SPD regression: the second impulse binds while coupling
            # leaves a positive first impulse.
            "setup": coupled_fixture,
            "call": (
                "float(contact_qp_solve(dof_minus, dof_plus, mass, k_diag, k_up, "
                "u_tilde, v_free, v_n, dt, 3)['p'][0])"
            ),
            "gold_call": (
                "float(_oracle_contact_qp_solve(dof_minus, dof_plus, mass, k_diag, k_up, "
                "u_tilde, v_free, v_n, dt, 3)['p'][0])"
            ),
        },
        {
            # KKT primal feasibility is checked independently.
            "setup": coupled_fixture,
            "call": (
                "float(np.min(contact_qp_solve(dof_minus, dof_plus, mass, k_diag, k_up, "
                "u_tilde, v_free, v_n, dt, 3)['p']) >= -1e-12)"
            ),
            "gold_call": (
                "float(np.min(_oracle_contact_qp_solve(dof_minus, dof_plus, mass, k_diag, k_up, "
                "u_tilde, v_free, v_n, dt, 3)['p']) >= -1e-12)"
            ),
        },
        {
            # KKT dual feasibility is checked independently.
            "setup": coupled_fixture,
            "call": (
                "float(np.min(W @ contact_qp_solve(dof_minus, dof_plus, mass, k_diag, k_up, "
                "u_tilde, v_free, v_n, dt, 3)['p'] + b) >= -1e-12)"
            ),
            "gold_call": (
                "float(np.min(W @ _oracle_contact_qp_solve(dof_minus, dof_plus, mass, k_diag, k_up, "
                "u_tilde, v_free, v_n, dt, 3)['p'] + b) >= -1e-12)"
            ),
        },
        {
            # KKT complementarity is checked independently.
            "setup": coupled_fixture,
            "call": (
                "float(np.max(np.abs(contact_qp_solve(dof_minus, dof_plus, mass, k_diag, k_up, "
                "u_tilde, v_free, v_n, dt, 3)['p'] * (W @ contact_qp_solve(dof_minus, dof_plus, mass, "
                "k_diag, k_up, u_tilde, v_free, v_n, dt, 3)['p'] + b))) <= 1e-12)"
            ),
            "gold_call": (
                "float(np.max(np.abs(_oracle_contact_qp_solve(dof_minus, dof_plus, mass, k_diag, k_up, "
                "u_tilde, v_free, v_n, dt, 3)['p'] * (W @ _oracle_contact_qp_solve(dof_minus, dof_plus, mass, "
                "k_diag, k_up, u_tilde, v_free, v_n, dt, 3)['p'] + b))) <= 1e-12)"
            ),
        },
        {
            # regression: b = H(v_free + e*v_n), not (1+e)*H*v_free.
            "setup": (
                "import numpy as np\n"
                "dof_minus = np.array([0]); dof_plus = np.array([1])\n"
                "mass = np.ones(2); k_diag = np.zeros(2); k_up = np.zeros(1)\n"
                "u_tilde = np.zeros(2); v_free = np.array([0.5, -0.5])\n"
                "v_n = np.array([0.25, -0.25])"
            ),
            "call": (
                "float(contact_qp_solve(dof_minus, dof_plus, mass, k_diag, k_up, "
                "u_tilde, v_free, v_n=v_n, dt=0.1, n_dof=2, "
                "e_restitution=0.5)['p'][0])"
            ),
            "gold_call": (
                "float(_oracle_contact_qp_solve(dof_minus, dof_plus, mass, k_diag, k_up, "
                "u_tilde, v_free, v_n=v_n, dt=0.1, n_dof=2, "
                "e_restitution=0.5)['p'][0])"
            ),
        },
        {
            # normal: pair 0 is closing (gap <= 0, active), pair 1 is
            # opening (gap > 0, inactive) -- exposes the impulse sum and the
            # active-set size.
            "setup": fixture,
            "call": (
                "float(np.sum(contact_qp_solve(dof_minus, dof_plus, mass, k_diag, k_up, "
                "np.array([0.0, 0.0, 0.0, 1.0]), np.array([0.5, -0.5, -0.2, 0.2]), v_n, dt, 4)['p'])) "
                "+ float(contact_qp_solve(dof_minus, dof_plus, mass, k_diag, k_up, "
                "np.array([0.0, 0.0, 0.0, 1.0]), np.array([0.5, -0.5, -0.2, 0.2]), v_n, dt, 4)['active'].size)"
            ),
            "gold_call": (
                "float(np.sum(_oracle_contact_qp_solve(dof_minus, dof_plus, mass, k_diag, k_up, "
                "np.array([0.0, 0.0, 0.0, 1.0]), np.array([0.5, -0.5, -0.2, 0.2]), v_n, dt, 4)['p'])) "
                "+ float(_oracle_contact_qp_solve(dof_minus, dof_plus, mass, k_diag, k_up, "
                "np.array([0.0, 0.0, 0.0, 1.0]), np.array([0.5, -0.5, -0.2, 0.2]), v_n, dt, 4)['active'].size)"
            ),
        },
        {
            # boundary: both pairs start at exactly zero gap (active from
            # the first instant, as in the physical march's first step)
            "setup": fixture,
            "call": (
                "float(contact_qp_solve(dof_minus, dof_plus, mass, k_diag, k_up, "
                "np.zeros(4), np.array([0.3, -0.3, -0.1, 0.1]), v_n, dt, 4)['active'].size) "
                "+ float(np.max(np.abs(contact_qp_solve(dof_minus, dof_plus, mass, k_diag, k_up, "
                "np.zeros(4), np.array([0.3, -0.3, -0.1, 0.1]), v_n, dt, 4)['gap'])))"
            ),
            "gold_call": (
                "float(_oracle_contact_qp_solve(dof_minus, dof_plus, mass, k_diag, k_up, "
                "np.zeros(4), np.array([0.3, -0.3, -0.1, 0.1]), v_n, dt, 4)['active'].size) "
                "+ float(np.max(np.abs(_oracle_contact_qp_solve(dof_minus, dof_plus, mass, k_diag, k_up, "
                "np.zeros(4), np.array([0.3, -0.3, -0.1, 0.1]), v_n, dt, 4)['gap'])))"
            ),
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
            "call": "_guard(lambda: contact_qp_solve(dof_minus, dof_plus, mass, k_diag, k_up, np.zeros(4), np.zeros(4), v_n, 0.0, 4))",
            "gold_call": "_guard(lambda: _oracle_contact_qp_solve(dof_minus, dof_plus, mass, k_diag, k_up, np.zeros(4), np.zeros(4), v_n, 0.0, 4))",
        },
        {
            # edge: a negative restitution coefficient is invalid
            "setup": fixture + """
def _guard(thunk):
    try:
        thunk()
        return 0
    except ValueError:
        return 2
    except Exception:
        return 1""",
            "call": "_guard(lambda: contact_qp_solve(dof_minus, dof_plus, mass, k_diag, k_up, np.zeros(4), np.zeros(4), v_n, dt, 4, e_restitution=-0.1))",
            "gold_call": "_guard(lambda: _oracle_contact_qp_solve(dof_minus, dof_plus, mass, k_diag, k_up, np.zeros(4), np.zeros(4), v_n, dt, 4, e_restitution=-0.1))",
        },
    ]
