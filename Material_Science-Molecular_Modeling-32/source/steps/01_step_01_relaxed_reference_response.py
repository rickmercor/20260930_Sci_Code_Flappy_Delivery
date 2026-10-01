"""
Compute an implicitly relaxed molecular reference and its second parameter response.

Let a parameter-dependent reduced molecular potential be

$$V_\lambda(r)=\tfrac12r^TA(\lambda)r-f(\lambda)^Tr+

\sum_j\left[\tfrac{a_j}{3}(c_j^Tr)^3+\tfrac{\beta_j(\lambda)}4(c_j^Tr)^4\right].$$

The symmetric matrices $A$, load $f$, and quartic coefficients $\beta$ have quadratic dependence on $\lambda$; $a$ and the channel matrix $C$ are fixed. Throughout this task, a three-layer jet contains raw derivatives $X^{[j]}=\partial_\lambda^jX|_0$, not Taylor coefficients divided by factorials.



The harmonic reference is evaluated on the locally stable stationary branch

$$g(r_*(\lambda),\lambda)=\nabla_rV_\lambda(r_*(\lambda))=0,\qquad

H_*(\lambda)=\nabla_r^2V_\lambda(r_*(\lambda)).$$

Implicit differentiation gives

$$H_*r_*'=-g_\lambda,$$

$$H_*r_*''=-g_{\lambda\lambda}-2g_{r\lambda}r_*'

-g_{rr}[r_*',r_*'].$$

All derivatives on the right are partial derivatives evaluated at the nominal stationary point. Thus evaluating the Hessian only at a fixed geometry omits the relaxation contribution. With $u=Cr$, the channel contribution to the gradient and Hessian is

$$g=A r-f+C^T(au^2+\beta u^3),\qquad

H=A+C^T\operatorname{diag}(2au+3\beta u^2)C,$$

where channel multiplication is componentwise. The returned Hessian derivatives must be total derivatives along $r_*(\lambda)$, including cubic and quartic contractions with both $r_*'$ and $r_*''$. This extends the paper's local harmonic reference to a smoothly changing physical model; the stationary solve and its response are task-defined.

Returns
-------
np.ndarray, shape (3,s,s+1), relaxed geometry and Hessian through second parameter order.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def relaxed_reference_response(
    A: "np.ndarray",
    C: "np.ndarray",
    cubic: "np.ndarray",
    beta: "np.ndarray",
    force: "np.ndarray",
    guess: "np.ndarray",
    tol: float,
    maxiter: int,
) -> "np.ndarray":
    r"""Compute an implicitly relaxed molecular reference and its second parameter response.

    Parameters
    ----------
    A : np.ndarray
        Shape (3,s,s), symmetric quadratic-stiffness derivative matrices.
    C : np.ndarray
        Shape (j,s), fixed displacement-channel matrix.
    cubic : np.ndarray
        Shape (j,), coefficients a multiplying channel cubes divided by 3.
    beta : np.ndarray
        Shape (3,j), quartic coefficient derivatives; quartic terms divided by 4.
    force : np.ndarray
        Shape (3,s), derivatives of the static linear load.
    guess : np.ndarray
        Shape (s,), starting point on the basin of the desired stable stationary root.
    tol : float
        Positive equilibrium infinity-norm residual tolerance, scaled by
        1+norm(force[0],inf).
    maxiter : int
        Positive maximum Newton updates. Use residual-decreasing backtracking.

    Returns
    -------
    result, np.ndarray
        Shape (3,s,s+1), column 0 contains r_star and its two derivatives;
        columns 1: contain the relaxed Hessian and its two total derivatives.
        The branch has nonsingular positive-definite nominal Hessian.

    Raises
    ------
    ValueError
        If the stationary solve or its residual-decreasing line search fails.

    All derivative arrays have leading axis (value, first derivative, second derivative) at lambda=0, with no factorial scaling. All data are real, finite, and conforming; inputs must be preserved.
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _jmul(a, b):
    return np.stack(
        (
            a[0] * b[0],
            a[1] * b[0] + a[0] * b[1],
            a[2] * b[0] + 2 * a[1] * b[1] + a[0] * b[2],
        )
    )


def _jmat(a, b):
    return np.stack(
        (
            a[0] @ b[0],
            a[1] @ b[0] + a[0] @ b[1],
            a[2] @ b[0] + 2 * a[1] @ b[1] + a[0] @ b[2],
        )
    )


def _jtranspose(a):
    return a.swapaxes(-1, -2)


def _jinverse(a):
    v = np.linalg.inv(a[0])
    first = -v @ a[1] @ v
    second = 2 * v @ a[1] @ v @ a[1] @ v - v @ a[2] @ v
    return np.stack((v, first, second))


def _gradient_hessian(r, A, C, cubic, beta, force):
    t = C @ r
    g = A @ r - force + C.T @ (cubic * t * t + beta * t**3)
    H = A + C.T @ ((2 * cubic * t + 3 * beta * t * t)[:, None] * C)
    return g, H


def _oracle_relaxed_reference_response(
    A: "np.ndarray",
    C: "np.ndarray",
    cubic: "np.ndarray",
    beta: "np.ndarray",
    force: "np.ndarray",
    guess: "np.ndarray",
    tol: float,
    maxiter: int,
) -> "np.ndarray":
    r = guess.copy()
    threshold = tol * (1 + np.linalg.norm(force[0], ord=np.inf))
    for iteration in range(maxiter + 1):
        g, H = _gradient_hessian(r, A[0], C, cubic, beta[0], force[0])
        residual = np.linalg.norm(g, ord=np.inf)
        if residual <= threshold:
            break
        if iteration == maxiter:
            raise ValueError("equilibrium solve failed")
        delta = np.linalg.solve(H, -g)
        step = 1.0
        for _ in range(30):
            trial = r + step * delta
            newg, _ = _gradient_hessian(trial, A[0], C, cubic, beta[0], force[0])
            if np.linalg.norm(newg, ord=np.inf) <= (1 - 1e-4 * step) * residual:
                r = trial
                break
            step *= 0.5
        else:
            raise ValueError("equilibrium line search failed")
    t = C @ r
    gl = A[1] @ r - force[1] + C.T @ (beta[1] * t**3)
    r1 = np.linalg.solve(H, -gl)
    t1 = C @ r1
    Hl = A[1] + C.T @ ((3 * beta[1] * t * t)[:, None] * C)
    gll = A[2] @ r - force[2] + C.T @ (beta[2] * t**3)
    r2 = np.linalg.solve(
        H, -gll - 2 * Hl @ r1 - C.T @ ((2 * cubic + 6 * beta[0] * t) * t1 * t1)
    )
    t2 = C @ r2
    H1 = A[1] + C.T @ (
        ((2 * cubic + 6 * beta[0] * t) * t1 + 3 * beta[1] * t * t)[:, None] * C
    )
    H2 = A[2] + C.T @ (
        (
            (2 * cubic + 6 * beta[0] * t) * t2
            + 6 * beta[0] * t1 * t1
            + 12 * beta[1] * t * t1
            + 3 * beta[2] * t * t
        )[:, None]
        * C
    )
    return np.concatenate(
        (np.stack((r, r1, r2))[:, :, None], np.stack((H, H1, H2))), axis=2
    )

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return coupled, boundary and response-stress cases."""
    return [
        {
            "setup": """import numpy as np
A = np.array([[[1.6246487603305786, 2.4386649010382797, -1.1861873256612212, 0.7048547179989777, 0.2650657512186243], [2.4386649010382797, 7.305024793388431, -2.6241521776486656, 1.3458344800334927, 1.6886346507388683], [-1.1861873256612212, -2.624152177648666, 5.9906921487603295, -0.3213074308698191, -2.1425023852020026], [0.7048547179989776, 1.3458344800334927, -0.3213074308698191, 12.217138429752065, 5.242804888834193], [0.26506575121862436, 1.6886346507388685, -2.1425023852020026, 5.242804888834193, 8.375008264462812]], [[0.08, 0.04, -0.04, 0.0, 0.04], [0.04, -0.04, 0.04, 0.04, 0.0], [-0.04, 0.04, 0.08, -0.04, 0.04], [0.0, 0.04, -0.04, 0.04, -0.04], [0.04, 0.0, 0.04, -0.04, -0.08]], [[0.015, -0.015, 0.0, 0.015, 0.0], [-0.015, 0.03, 0.015, 0.0, -0.015], [0.0, 0.015, -0.015, 0.015, 0.015], [0.015, 0.0, 0.015, 0.03, 0.0], [0.0, -0.015, 0.015, 0.0, 0.015]]], dtype=float)
C = np.array([[1.0, -1.0, 0.0, 0.0, 0.0], [0.0, 1.0, -1.0, 0.0, 0.0], [0.0, 0.0, 1.0, -1.0, 0.0], [0.0, 0.0, 0.0, 1.0, -1.0], [1.0, 0.0, 0.0, 0.0, 1.0], [1.0, -0.5, 0.25, -0.5, 1.0]], dtype=float)
cubic = np.array([0.12, -0.15, 0.08, -0.09, 0.1, -0.07], dtype=float)
beta = np.array([[0.65, 0.45, 0.55, 0.6, 0.5, 0.7], [0.22, -0.18, 0.12, 0.15, -0.14, 0.2], [0.08, 0.05, -0.04, 0.06, 0.03, -0.05]], dtype=float)
force = np.array([[0.035, -0.025, 0.04, -0.015, 0.02], [0.018, 0.01, -0.012, 0.015, -0.008], [-0.01, 0.012, 0.008, -0.006, 0.011]], dtype=float)
guess = np.array([0.1, -0.08, 0.06, -0.02, 0.05], dtype=float)
tol = 1e-12
maxiter = 40

def preserved(fn, *args):
    snapshots = [(x, x.copy()) for x in args if isinstance(x, np.ndarray)]
    result = fn(*args)
    for value, original in snapshots:
        assert np.array_equal(value, original), "inputs must be preserved"
    return result
""",
            "call": "preserved(relaxed_reference_response, A.copy(), C.copy(), cubic.copy(), beta.copy(), force.copy(), guess.copy(), tol, maxiter)",
            "gold_call": "preserved(_oracle_relaxed_reference_response, A.copy(), C.copy(), cubic.copy(), beta.copy(), force.copy(), guess.copy(), tol, maxiter)",
            "tol": 2e-09,
        },
        {
            "setup": """import numpy as np
A = np.array([[[1.6246487603305786, 2.4386649010382797, -1.1861873256612212, 0.7048547179989777, 0.2650657512186243], [2.4386649010382797, 7.305024793388431, -2.6241521776486656, 1.3458344800334927, 1.6886346507388683], [-1.1861873256612212, -2.624152177648666, 5.9906921487603295, -0.3213074308698191, -2.1425023852020026], [0.7048547179989776, 1.3458344800334927, -0.3213074308698191, 12.217138429752065, 5.242804888834193], [0.26506575121862436, 1.6886346507388685, -2.1425023852020026, 5.242804888834193, 8.375008264462812]], [[0.08, 0.04, -0.04, 0.0, 0.04], [0.04, -0.04, 0.04, 0.04, 0.0], [-0.04, 0.04, 0.08, -0.04, 0.04], [0.0, 0.04, -0.04, 0.04, -0.04], [0.04, 0.0, 0.04, -0.04, -0.08]], [[0.015, -0.015, 0.0, 0.015, 0.0], [-0.015, 0.03, 0.015, 0.0, -0.015], [0.0, 0.015, -0.015, 0.015, 0.015], [0.015, 0.0, 0.015, 0.03, 0.0], [0.0, -0.015, 0.015, 0.0, 0.015]]], dtype=float)
C = np.array([[1.0, -1.0, 0.0, 0.0, 0.0], [0.0, 1.0, -1.0, 0.0, 0.0], [0.0, 0.0, 1.0, -1.0, 0.0], [0.0, 0.0, 0.0, 1.0, -1.0], [1.0, 0.0, 0.0, 0.0, 1.0], [1.0, -0.5, 0.25, -0.5, 1.0]], dtype=float)
cubic = np.array([0.204, -0.255, 0.136, -0.153, 0.17, -0.11900000000000001], dtype=float)
beta = np.array([[0.65, 0.45, 0.55, 0.6, 0.5, 0.7], [-0.154, 0.126, -0.08399999999999999, -0.105, 0.098, -0.13999999999999999], [-0.055999999999999994, -0.034999999999999996, 0.027999999999999997, -0.041999999999999996, -0.020999999999999998, 0.034999999999999996]], dtype=float)
force = np.array([[0.049, -0.034999999999999996, 0.055999999999999994, -0.020999999999999998, 0.027999999999999997], [0.025199999999999997, 0.013999999999999999, -0.0168, 0.020999999999999998, -0.0112], [-0.013999999999999999, 0.0168, 0.0112, -0.0084, 0.015399999999999999]], dtype=float)
guess = np.array([0.2, -0.16, 0.12, -0.04, 0.1], dtype=float)
tol = 1e-12
maxiter = 40
""",
            "call": "relaxed_reference_response(A.copy(), C.copy(), cubic.copy(), beta.copy(), force.copy(), guess.copy(), tol, maxiter)",
            "gold_call": "_oracle_relaxed_reference_response(A.copy(), C.copy(), cubic.copy(), beta.copy(), force.copy(), guess.copy(), tol, maxiter)",
            "tol": 2e-09,
        },
        {
            "setup": """import numpy as np
A = np.array([[[1.6246487603305786, 2.4386649010382797, -1.1861873256612212, 0.7048547179989777, 0.2650657512186243], [2.4386649010382797, 7.305024793388431, -2.6241521776486656, 1.3458344800334927, 1.6886346507388683], [-1.1861873256612212, -2.624152177648666, 5.9906921487603295, -0.3213074308698191, -2.1425023852020026], [0.7048547179989776, 1.3458344800334927, -0.3213074308698191, 12.217138429752065, 5.242804888834193], [0.26506575121862436, 1.6886346507388685, -2.1425023852020026, 5.242804888834193, 8.375008264462812]], [[0.08, 0.04, -0.04, 0.0, 0.04], [0.04, -0.04, 0.04, 0.04, 0.0], [-0.04, 0.04, 0.08, -0.04, 0.04], [0.0, 0.04, -0.04, 0.04, -0.04], [0.04, 0.0, 0.04, -0.04, -0.08]], [[0.015, -0.015, 0.0, 0.015, 0.0], [-0.015, 0.03, 0.015, 0.0, -0.015], [0.0, 0.015, -0.015, 0.015, 0.015], [0.015, 0.0, 0.015, 0.03, 0.0], [0.0, -0.015, 0.015, 0.0, 0.015]]], dtype=float)
C = np.array([[1.0, -1.0, 0.0, 0.0, 0.0], [0.0, 1.0, -1.0, 0.0, 0.0], [0.0, 0.0, 1.0, -1.0, 0.0], [0.0, 0.0, 0.0, 1.0, -1.0], [1.0, 0.0, 0.0, 0.0, 1.0], [1.0, -0.5, 0.25, -0.5, 1.0]], dtype=float)
cubic = np.array([0.06, -0.075, 0.04, -0.045, 0.05, -0.035], dtype=float)
beta = np.array([[0.65, 0.45, 0.55, 0.6, 0.5, 0.7], [0.22, -0.18, 0.12, 0.15, -0.14, 0.2], [0.08, 0.05, -0.04, 0.06, 0.03, -0.05]], dtype=float)
force = np.array([[0.0, 0.0, 0.0, 0.0, 0.0], [0.018, 0.01, -0.012, 0.015, -0.008], [-0.01, 0.012, 0.008, -0.006, 0.011]], dtype=float)
guess = np.array([0.30000000000000004, -0.24, 0.18, -0.06, 0.15000000000000002], dtype=float)
tol = 1e-12
maxiter = 40
""",
            "call": "relaxed_reference_response(A.copy(), C.copy(), cubic.copy(), beta.copy(), force.copy(), guess.copy(), tol, maxiter)",
            "gold_call": "_oracle_relaxed_reference_response(A.copy(), C.copy(), cubic.copy(), beta.copy(), force.copy(), guess.copy(), tol, maxiter)",
            "tol": 2e-09,
        },
    ]
