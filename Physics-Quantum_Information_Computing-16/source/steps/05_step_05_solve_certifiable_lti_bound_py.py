"""
Step 5: Solve the level-3 certifiable LTI SDP for one local interaction.

The SDP maximizes delta subject to q3(delta, Y) >= 0. Since q3 annihilates the three-site open-chain ground space by construction, the positive semidefinite cone is restricted to the orthogonal complement of that space, where the program is strictly feasible, and any gauge kernel of the map Y -> I (x) Z - Z (x) I is removed by a singular value decomposition. The program is solved with a self-contained primal log-det barrier method with damped Newton steps and a decreasing barrier parameter, so no external SDP solver is needed. The returned optimum is post-checked for numerical feasibility: the minimum eigenvalue of q3 on the complement must not fall below a floor tied to the solver tolerance. This is a numerical feasibility check at an active constraint, not the rigorous positive-margin certificate of the paper, which would lower delta until a strictly positive margin is obtained.

Returns
-------
# float, the certified level-3 LTI lower bound for x as a native Python float
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================
import numpy as np

def solve_certifiable_lti_bound(
    x: "np.ndarray",
    p2_perp: "np.ndarray",
    solver_tol: float = 1e-11,
) -> float:
    """Solve the level-3 certifiable LTI SDP and return its optimal value.

    Parameters
    ----------
    x : np.ndarray
        Hermitian positive-semidefinite local interaction of shape (9,9).
    p2_perp : np.ndarray
        Hermitian orthogonal projector of shape (9,9).
    solver_tol : float
        Positive finite barrier tolerance (the optimum is accurate to about this value).

    Returns
    -------
    delta : float
        Level-3 certifiable LTI lower bound as a native Python float.

    Raises
    ------
    ValueError
        If x is not Hermitian PSD of shape (9,9), p2_perp is not a Hermitian projector
        of shape (9,9), solver_tol is not finite and > 0, the barrier method cannot find a
        strictly feasible start, or the returned solution fails the numerical feasibility check.
    """
    return delta

# =============================================================================
# GOLD SOLUTION
# =============================================================================

# =============================================================================
# HELPERS
# =============================================================================
import numpy as np
def _realify(C: "np.ndarray") -> "np.ndarray":
    import numpy as np
    return np.block([[C.real, -C.imag], [C.imag, C.real]])

def _hermitian_basis_on_range(P: "np.ndarray") -> list:
    import numpy as np
    w, V = np.linalg.eigh(P); W = V[:, w > 0.5]; m = W.shape[1]; basis = []
    for a in range(m):
        for b in range(a, m):
            Z = np.zeros((m, m), dtype=complex)
            if a == b:
                Z[a, a] = 1.0; basis.append(W @ Z @ W.conj().T)
            else:
                Z[a, b] = Z[b, a] = 1/np.sqrt(2); basis.append(W @ Z @ W.conj().T)
                Z = np.zeros((m, m), dtype=complex); Z[a, b] = 1j/np.sqrt(2); Z[b, a] = -1j/np.sqrt(2)
                basis.append(W @ Z @ W.conj().T)
    return basis

def _barrier_sdp_max(c: "np.ndarray", C: "np.ndarray", A: "np.ndarray", z0: "np.ndarray", tol: float) -> "np.ndarray":
    """Maximize c.z subject to C + sum_i z_i A_i >= 0 (real symmetric), starting from a strictly feasible z0.
    Primal log-det barrier with damped Newton steps; stops when n*mu < tol.  Deterministic."""
    import numpy as np
    A = np.asarray(A, dtype=float); n = C.shape[0]; k = len(c)
    z = np.array(z0, dtype=float)
    Q = C + np.tensordot(z, A, axes=1)
    if np.linalg.eigvalsh(Q).min() <= 0.0:
        raise ValueError("barrier start is not strictly feasible")
    mu = max(1.0, abs(float(c @ z)))/n
    while True:
        for _ in range(100):
            L = np.linalg.cholesky(Q); Linv = np.linalg.inv(L); Qinv = Linv.T @ Linv
            QA = np.einsum('ij,kjl->kil', Qinv, A)
            g = -c - mu*np.einsum('kii->k', QA)
            H = mu*np.einsum('kij,lji->kl', QA, QA)
            H = 0.5*(H + H.T) + 1e-14*np.eye(k)
            d = -np.linalg.solve(H, g)
            if float(-g @ d) < 1e-13:
                break
            phi0 = -float(c @ z) - mu*2.0*np.log(np.diag(L)).sum()
            t = 1.0
            while True:
                zn = z + t*d; Qn = C + np.tensordot(zn, A, axes=1)
                try:
                    Ln = np.linalg.cholesky(Qn)
                    if -float(c @ zn) - mu*2.0*np.log(np.diag(Ln)).sum() <= phi0 + 0.25*t*float(g @ d):
                        break
                except np.linalg.LinAlgError:
                    pass
                t *= 0.5
                if t < 1e-12:
                    break
            z, Q = zn, Qn
        if n*mu < tol:
            return z
        mu *= 0.15

# =============================================================================
# ORACLE SOLUTION
# =============================================================================
import numpy as np

def _oracle_solve_certifiable_lti_bound(
    x: "np.ndarray",
    p2_perp: "np.ndarray",
    solver_tol: float = 1e-11,
) -> float:
    import numpy as np

    x = np.asarray(x, dtype=complex)
    P = np.asarray(p2_perp, dtype=complex)
    solver_tol = float(solver_tol)

    if x.shape != (9, 9) or not np.allclose(
        x, x.conj().T, atol=1e-10, rtol=0.0
    ):
        raise ValueError("x must be Hermitian with shape (9,9)")

    if np.linalg.eigvalsh(0.5 * (x + x.conj().T))[0] < -1e-10:
        raise ValueError("x must be positive semidefinite")

    if (
        P.shape != (9, 9)
        or not np.allclose(P, P.conj().T, atol=1e-10, rtol=0.0)
        or not np.allclose(P @ P, P, atol=1e-8, rtol=0.0)
    ):
        raise ValueError(
            "p2_perp must be a Hermitian projector with shape (9,9)"
        )

    if not np.isfinite(solver_tol) or solver_tol <= 0.0:
        raise ValueError("solver_tol must be finite and > 0")

    I3 = np.eye(3, dtype=complex)
    x1 = np.kron(x, I3)
    x2 = np.kron(I3, x)

    G0 = x1 @ x1 + x1 @ x2 + x2 @ x1

    e3, V3 = np.linalg.eigh(x1 + x2)
    Wp = V3[:, e3 > 1e-10]
    n = Wp.shape[1]

    red = lambda Op: _realify(Wp.conj().T @ Op @ Wp)

    basis = _hermitian_basis_on_range(P)
    B = np.array(
        [
            red(np.kron(I3, Yb) - np.kron(Yb, I3))
            for Yb in basis
        ]
    )

    C = red(G0)
    X1 = red(x1)

    # Phase I: find a strictly feasible point at delta = 0.
    A1 = np.concatenate([[-np.eye(2 * n)], B])
    c1 = np.zeros(1 + len(B))
    c1[0] = 1.0

    z1 = np.zeros(1 + len(B))
    z1[0] = float(np.linalg.eigvalsh(C).min()) - 1.0

    z1 = _barrier_sdp_max(c1, C, A1, z1, 1e-3)

    if z1[0] <= 0.0:
        raise ValueError("SDP has no strictly feasible point at delta = 0")

    # Phase II: maximize delta from the feasible Phase-I gauge.
    A2 = np.concatenate([[-X1], B])
    c2 = np.zeros(1 + len(B))
    c2[0] = 1.0

    z2 = np.concatenate([[0.0], z1[1:]])
    z2 = _barrier_sdp_max(c2, C, A2, z2, solver_tol)

    delta = float(z2[0])
    y = z2[1:]

    Y = sum(cc * Yb for cc, Yb in zip(y, basis))
    Y = 0.5 * (Y + Y.conj().T)

    q3 = _oracle_level3_generator(x, P, delta, Y)
    lmin = float(
        np.linalg.eigvalsh(Wp.conj().T @ q3 @ Wp)[0]
    )

    if lmin < -max(10.0 * solver_tol, 1e-10):
        raise ValueError("numerical feasibility check failed")

    return delta

# =============================================================================
# TEST CASES
# =============================================================================

# =============================================================================
# TEST CASES
# =============================================================================
def test_cases():
    return [
        # normal: target point, delta_h = 0.41547344
        {
            "setup": (
                "import numpy as np\n"
                "h,_=_oracle_build_local_interaction(0.347,0.783); "
                "P,_=_oracle_excited_subspace_projector(h)"
            ),
            "call": "solve_certifiable_lti_bound(h,P)",
            "gold_call": "_oracle_solve_certifiable_lti_bound(h,P)",
        },

        # boundary: undeformed Potts, optimum is exactly 3
        {
            "setup": (
                "import numpy as np\n"
                "h,_=_oracle_build_local_interaction(1.0,1.0); "
                "P,_=_oracle_excited_subspace_projector(h)"
            ),
            "call": "solve_certifiable_lti_bound(h,P)",
            "gold_call": "_oracle_solve_certifiable_lti_bound(h,P)",
        },

        # edge: projector interaction Pi = supp(h), optimum 0.04224334
        {
            "setup": (
                "import numpy as np\n"
                "h,_=_oracle_build_local_interaction(0.347,0.783); "
                "P,_=_oracle_excited_subspace_projector(h)"
            ),
            "call": "solve_certifiable_lti_bound(P,P)",
            "gold_call": "_oracle_solve_certifiable_lti_bound(P,P)",
        },
    ]
