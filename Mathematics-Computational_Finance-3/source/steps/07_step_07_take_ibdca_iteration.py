"""
Perform one safeguarded boosted portfolio-allocation iteration from two consecutive iterates.

Each update preserves the simplex constraints and uses the supplied algorithm controls consistently across the two portfolio blocks.

Returns
-------
The returned value is the next two-period allocation after the deterministic feasibility and descent checks.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def take_ibdca_iteration(probabilities: np.ndarray, returns: np.ndarray, w_prev: np.ndarray, c: np.ndarray, lambda1: float, lambda2: float, rho: float, alpha: float, gamma: float, tau: float, theta: float, nu: float, sigma: float, eta: float, bar_lambda: float, x_k: np.ndarray, x_km1: np.ndarray) -> np.ndarray:
    """Return the next allocation produced by one safeguarded boosted iteration.

    Parameters
    ----------
    probabilities : np.ndarray
        Scenario probabilities, shape (2, 6).
    returns : np.ndarray
        Scenario returns, shape (2, 6, 3).
    w_prev : np.ndarray
        Holding before period 1.
    c : np.ndarray
        Transaction-cost coefficients.
    lambda1, lambda2, rho, alpha, gamma, tau, theta, nu, sigma, eta, bar_lambda : float
        Model and algorithm controls. No arguments are optional.
    x_k : np.ndarray
        Current two-period allocation.
    x_km1 : np.ndarray
        Previous two-period allocation.

    Returns
    -------
    x_next : np.ndarray
        Next two-period allocation, shape (2, 3).

    Raises
    ------
    ValueError
        If the supplied arrays do not have the instance shapes or gamma violates
        the finite-scenario gap condition.
    """
    return x_next

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _project_simplex(v):
    v = np.asarray(v, dtype=float)
    if v.ndim != 1 or v.size == 0 or np.any(~np.isfinite(v)):
        raise ValueError("projection input must be a finite nonempty vector")

    u = np.sort(v)[::-1]
    cssv = np.cumsum(u) - 1.0
    idx = np.nonzero(
        u - cssv / (np.arange(v.size) + 1.0) > 0.0
    )[0][-1]

    return np.maximum(
        v - cssv[idx] / (idx + 1.0),
        0.0,
    )


def _phi(
    probabilities,
    returns,
    w_prev,
    c,
    lambda1,
    lambda2,
    rho,
    alpha,
    gamma,
    tau,
    x,
):
    p = np.asarray(probabilities, dtype=float)
    R = np.asarray(returns, dtype=float)
    wp = np.asarray(w_prev, dtype=float)
    cc = np.asarray(c, dtype=float)
    xx = np.asarray(x, dtype=float)

    mu = np.einsum("ts,tsn->tn", p, R)

    value = -sum(mu[t] @ xx[t] for t in range(2))
    value += float(lambda2) * float(np.dot(xx.ravel(), xx.ravel()))

    value += float(lambda1) * (
        cc @ np.abs(xx[0] - wp)
        + cc @ np.abs(xx[1] - xx[0])
    )

    for t in range(2):
        gap = _oracle_compute_scenario_gap(p[t], alpha)
        if not (0.0 < gamma < gap):
            raise ValueError(
                "gamma must satisfy the finite-scenario gap condition"
            )

        losses = _oracle_compute_loss_vector(R[t], xx[t])

        cvar_lower = _oracle_compute_cvar(
            losses, p[t], alpha - gamma
        )
        cvar_alpha = _oracle_compute_cvar(
            losses, p[t], alpha
        )

        terms = _oracle_build_dc_risk_terms(
            cvar_lower,
            cvar_alpha,
            alpha,
            gamma,
            tau,
        )

        value += float(rho) * (
            terms[4] - terms[1] * cvar_alpha
        )

    return float(value)


def _oracle_take_ibdca_iteration(
    probabilities: np.ndarray,
    returns: np.ndarray,
    w_prev: np.ndarray,
    c: np.ndarray,
    lambda1: float,
    lambda2: float,
    rho: float,
    alpha: float,
    gamma: float,
    tau: float,
    theta: float,
    nu: float,
    sigma: float,
    eta: float,
    bar_lambda: float,
    x_k: np.ndarray,
    x_km1: np.ndarray,
) -> np.ndarray:

    p = np.asarray(probabilities, dtype=float)
    R = np.asarray(returns, dtype=float)
    wp = np.asarray(w_prev, dtype=float)
    cc = np.asarray(c, dtype=float)
    x = np.asarray(x_k, dtype=float)
    xm1 = np.asarray(x_km1, dtype=float)

    if (
        p.shape != (2, 6)
        or R.shape != (2, 6, 3)
        or wp.shape != (3,)
        or cc.shape != (3,)
        or x.shape != (2, 3)
        or xm1.shape != (2, 3)
    ):
        raise ValueError("invalid shapes")

    # Inertial extrapolation followed by blockwise simplex projection.
    extrapolated = x + float(theta) * (x - xm1)

    w = np.vstack(
        [_project_simplex(extrapolated[t]) for t in range(2)]
    )

    # Objective safeguard.
    if (
        _phi(
            p, R, wp, cc,
            lambda1, lambda2, rho,
            alpha, gamma, tau, w
        )
        >
        _phi(
            p, R, wp, cc,
            lambda1, lambda2, rho,
            alpha, gamma, tau, x
        ) + 1e-14
    ):
        w = x.copy()

    # CVaR subgradients from the earlier sub-problem.
    B = (1.0 - float(alpha)) / float(gamma)

    subgradients = np.vstack([
        float(rho)
        * B
        * _oracle_compute_cvar_subgradient(
            R[t], p[t], w[t], alpha
        )
        for t in range(2)
    ])

    u = subgradients + float(nu) * w

    # Strongly-convex DCA subproblem from step 6.
    y = _oracle_solve_regularized_subproblem(
        p,
        R,
        wp,
        cc,
        lambda1,
        lambda2,
        rho,
        alpha,
        gamma,
        tau,
        nu,
        w,
        u.ravel(),
    )

    d = y - w

    # Maximum feasible extrapolation along d.
    lambda_max = np.inf

    for t in range(2):
        for i in range(3):
            if d[t, i] < 0.0:
                lambda_max = min(
                    lambda_max,
                    -w[t, i] / d[t, i],
                )

    lam = min(
        float(bar_lambda),
        float(lambda_max),
    )

    # Boosted Armijo line search.
    while lam > 1.0:
        trial = w + lam * d

        dnorm2 = float(np.dot(d.ravel(), d.ravel()))

        rhs = min(
            _phi(
                p, R, wp, cc,
                lambda1, lambda2, rho,
                alpha, gamma, tau, y
            ),
            _phi(
                p, R, wp, cc,
                lambda1, lambda2, rho,
                alpha, gamma, tau, w
            )
            - float(sigma) * lam * lam * dnorm2,
        )

        trial_value = _phi(
            p, R, wp, cc,
            lambda1, lambda2, rho,
            alpha, gamma, tau, trial
        )

        if trial_value <= rhs + 1e-12:
            break

        lam = max(
            1.0,
            float(eta) * lam,
        )

    return w + lam * d

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return three one-iteration configurations."""
    base = """import numpy as np
p=np.array([[0.05,0.10,0.15,0.20,0.22,0.28],[0.08,0.09,0.12,0.16,0.24,0.31]],float)
R=np.array([[[-0.15,0.05,0.03],[0.04,-0.13,0.02],[0.03,0.02,-0.12],[-0.05,-0.04,0.06],[0.08,0.07,0.09],[-0.02,0.03,-0.06]],[[ -0.12,0.04,0.02],[0.05,-0.11,0.03],[0.02,0.03,-0.10],[-0.04,-0.03,0.05],[0.09,0.08,0.10],[-0.03,0.02,-0.05]]],float)
w_prev=np.array([0.40,0.35,0.25],float)
x0=np.array([[0.45,0.30,0.25],[0.30,0.45,0.25]],float)
c=np.array([0.06,0.10,0.14],float)
alpha=0.65; tau=0.02; lambda1=0.07; lambda2=0.02; rho=6.0; gamma=0.005; theta=0.05; nu=0.04; sigma=0.004; eta=0.5; bar_lambda=1.2\nxm1=x0.copy()"""
    base2 = """import numpy as np
p=np.array([[0.05,0.10,0.15,0.20,0.22,0.28],[0.08,0.09,0.12,0.16,0.24,0.31]],float)
R=np.array([[[-0.15,0.05,0.03],[0.04,-0.13,0.02],[0.03,0.02,-0.12],[-0.05,-0.04,0.06],[0.08,0.07,0.09],[-0.02,0.03,-0.06]],[[ -0.12,0.04,0.02],[0.05,-0.11,0.03],[0.02,0.03,-0.10],[-0.04,-0.03,0.05],[0.09,0.08,0.10],[-0.03,0.02,-0.05]]],float)
w_prev=np.array([0.40,0.35,0.25],float)
x0=np.array([[0.45,0.30,0.25],[0.30,0.45,0.25]],float)
c=np.array([0.06,0.10,0.14],float)
alpha=0.65; tau=0.02; lambda1=0.07; lambda2=0.02; rho=6.0; gamma=0.005; theta=0.05; nu=0.04; sigma=0.004; eta=0.5; bar_lambda=1.2\nxm1=np.array([[0.40,0.35,0.25],[0.40,0.35,0.25]],float)"""
    base3 = """import numpy as np
p=np.array([[0.05,0.10,0.15,0.20,0.22,0.28],[0.08,0.09,0.12,0.16,0.24,0.31]],float)
R=np.array([[[-0.15,0.05,0.03],[0.04,-0.13,0.02],[0.03,0.02,-0.12],[-0.05,-0.04,0.06],[0.08,0.07,0.09],[-0.02,0.03,-0.06]],[[ -0.12,0.04,0.02],[0.05,-0.11,0.03],[0.02,0.03,-0.10],[-0.04,-0.03,0.05],[0.09,0.08,0.10],[-0.03,0.02,-0.05]]],float)
w_prev=np.array([0.40,0.35,0.25],float)
x0=np.array([[0.45,0.30,0.25],[0.30,0.45,0.25]],float)
c=np.array([0.06,0.10,0.14],float)
alpha=0.65; tau=0.02; lambda1=0.07; lambda2=0.02; rho=6.0; gamma=0.005; theta=0.05; nu=0.04; sigma=0.004; eta=0.5; bar_lambda=1.2\nxm1=np.array([[0.50,0.25,0.25],[0.25,0.50,0.25]],float); xk=np.array([[0.35,0.40,0.25],[0.35,0.40,0.25]],float)"""
    args="p,R,w_prev,c,lambda1,lambda2,rho,alpha,gamma,tau,theta,nu,sigma,eta,bar_lambda"
    return [
        {"setup": base, "call": f"tuple(take_ibdca_iteration({args},x0,xm1))", "gold_call": f"tuple(_oracle_take_ibdca_iteration({args},x0,xm1))", "tol": 1e-5},
        {"setup": base2, "call": f"tuple(take_ibdca_iteration({args},x0,xm1))", "gold_call": f"tuple(_oracle_take_ibdca_iteration({args},x0,xm1))", "tol": 1e-5},
        {"setup": base3, "call": f"tuple(take_ibdca_iteration({args},xk,xm1))", "gold_call": f"tuple(_oracle_take_ibdca_iteration({args},xk,xm1))", "tol": 1e-5},
        {"setup": base+"\nxk=np.array([[0.36525,0.38475,0.25],[0.315,0.435,0.25]],float); xm1=x0.copy()", "call": f"tuple(take_ibdca_iteration({args},xk,xm1))", "gold_call": f"tuple(_oracle_take_ibdca_iteration({args},xk,xm1))", "tol": 1e-5},
        {"setup": base+"\nxk=np.array([[0.329655,0.420345,0.25],[0.32805,0.42195,0.25]],float); xm1=np.array([[0.36525,0.38475,0.25],[0.315,0.435,0.25]],float)", "call": f"tuple(take_ibdca_iteration({args},xk,xm1))", "gold_call": f"tuple(_oracle_take_ibdca_iteration({args},xk,xm1))", "tol": 1e-5},
    ]
