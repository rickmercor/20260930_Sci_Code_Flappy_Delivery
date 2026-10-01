"""
Run the full deterministic portfolio-allocation procedure for the requested number of outer iterations and extract the final target component.

The terminal value is extracted from the final feasible allocation after the requested fixed number of updates.

Returns
-------
The result is the period-2, asset-1 weight of the terminal allocation.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def run_ibdca_target(probabilities: np.ndarray, returns: np.ndarray, w_prev: np.ndarray, c: np.ndarray, lambda1: float, lambda2: float, rho: float, alpha: float, gamma: float, tau: float, theta: float, nu: float, sigma: float, eta: float, bar_lambda: float, x0: np.ndarray, K: int) -> float:
    """Return the requested terminal allocation component after K outer iterations.

    Parameters
    ----------
    probabilities : np.ndarray
        Scenario probabilities, shape (2, 6).
    returns : np.ndarray
        Scenario returns, shape (2, 6, 3).
    w_prev : np.ndarray
        Previous holding before the first period.
    c : np.ndarray
        Transaction-cost coefficients.
    lambda1, lambda2, rho, alpha, gamma, tau, theta, nu, sigma, eta, bar_lambda : float
        Model, decomposition, inertia, and line-search controls.
    x0 : np.ndarray
        Initial allocation. The algorithm uses the same array as the x^{-1} state.
    K : int
        Number of outer iterations to execute; no early stopping is used.

    Returns
    -------
    value : float
        Period-2, asset-1 component of the terminal allocation.

    Raises
    ------
    ValueError
        If the supplied instance shapes are incompatible or K is negative.
    """
    return value

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_run_ibdca_target(
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
    x0: np.ndarray,
    K: int,
) -> float:

    p = np.asarray(probabilities, dtype=float)
    R = np.asarray(returns, dtype=float)
    wp = np.asarray(w_prev, dtype=float)
    cc = np.asarray(c, dtype=float)

    x = np.asarray(x0, dtype=float).copy()
    xm1 = np.asarray(x0, dtype=float).copy()

    k = int(K)

    if (
        p.shape != (2, 6)
        or R.shape != (2, 6, 3)
        or wp.shape != (3,)
        or cc.shape != (3,)
        or x.shape != (2, 3)
        or k < 0
    ):
        raise ValueError("invalid inputs")

    for _ in range(k):
        xn = _oracle_take_ibdca_iteration(
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
            theta,
            nu,
            sigma,
            eta,
            bar_lambda,
            x,
            xm1,
        )

        xm1, x = x, xn

    return float(x[1, 0])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return three distinct whole-pipeline configurations."""
    base = """import numpy as np
p=np.array([[0.05,0.10,0.15,0.20,0.22,0.28],[0.08,0.09,0.12,0.16,0.24,0.31]],float)
R=np.array([[[-0.15,0.05,0.03],[0.04,-0.13,0.02],[0.03,0.02,-0.12],[-0.05,-0.04,0.06],[0.08,0.07,0.09],[-0.02,0.03,-0.06]],[[ -0.12,0.04,0.02],[0.05,-0.11,0.03],[0.02,0.03,-0.10],[-0.04,-0.03,0.05],[0.09,0.08,0.10],[-0.03,0.02,-0.05]]],float)
w_prev=np.array([0.40,0.35,0.25],float)
x0=np.array([[0.45,0.30,0.25],[0.30,0.45,0.25]],float)
c=np.array([0.06,0.10,0.14],float)
alpha=0.65; tau=0.02; lambda1=0.07; lambda2=0.02; rho=6.0; gamma=0.005; theta=0.05; nu=0.04; sigma=0.004; eta=0.5; bar_lambda=1.2\nK=4"""
    return [
        {"setup": base, "call": "run_ibdca_target(p,R,w_prev,c,lambda1,lambda2,rho,alpha,gamma,tau,theta,nu,sigma,eta,bar_lambda,x0,K)", "gold_call": "_oracle_run_ibdca_target(p,R,w_prev,c,lambda1,lambda2,rho,alpha,gamma,tau,theta,nu,sigma,eta,bar_lambda,x0,K)", "tol": 1e-5},
        {"setup": base+"\nx_alt=np.array([[1/3,1/3,1/3],[1/3,1/3,1/3]],float); K=2", "call": "run_ibdca_target(p,R,w_prev,c,lambda1,lambda2,rho,alpha,gamma,tau,theta,nu,sigma,eta,bar_lambda,x_alt,K)", "gold_call": "_oracle_run_ibdca_target(p,R,w_prev,c,lambda1,lambda2,rho,alpha,gamma,tau,theta,nu,sigma,eta,bar_lambda,x_alt,K)", "tol": 1e-5},
        {"setup": base+"\nx_alt=np.array([[0.60,0.25,0.15],[0.20,0.55,0.25]],float); K=1", "call": "run_ibdca_target(p,R,w_prev,c,lambda1,lambda2,rho,alpha,gamma,tau,theta,nu,sigma,eta,bar_lambda,x_alt,K)", "gold_call": "_oracle_run_ibdca_target(p,R,w_prev,c,lambda1,lambda2,rho,alpha,gamma,tau,theta,nu,sigma,eta,bar_lambda,x_alt,K)", "tol": 1e-5},
    ]
