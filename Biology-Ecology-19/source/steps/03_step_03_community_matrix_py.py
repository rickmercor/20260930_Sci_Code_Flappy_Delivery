"""
Assemble the generalized-modeling community matrix from branching fractions, row timescales, and elasticities.

When the ODE is not named, the Jacobian at a steady state can still be written from biomass-flow shares and elasticities. Production, predation gain, mortality, and predation loss each have a place. Shared predators enter through the elasticity of the predator’s diet. Each row then gets its own turnover scale.

Returns
-------
np.ndarray, shape (N, N), community matrix J as float
"""

import math
import numpy
import scipy.special

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def community_matrix(
    A: "np.ndarray",
    phi: "np.ndarray",
    gamma: "np.ndarray",
    psi: "np.ndarray",
    mu: "np.ndarray",
    lam: "np.ndarray",
    alpha: "np.ndarray",
    rho: "np.ndarray",
    sigma: "np.ndarray",
    chi: "np.ndarray",
    beta: "np.ndarray",
) -> "np.ndarray":
    '''Return the generalized-modeling community matrix J.

    Parameters
    ----------
    A : np.ndarray
        Square 0-1 feeding matrix.
    phi, gamma, psi, mu, alpha, rho, sigma : np.ndarray
        Length-N elasticity or scale vectors.
    lam, chi, beta : np.ndarray
        Shape (N, N) elasticity and branching matrices.

    Returns
    -------
    J : np.ndarray
        Shape (N, N) Jacobian at the unknown steady state.

    Raises
    ------
    ValueError
        If any array shape does not match N or (N, N), if A is not 0-1,
        or if any input is not finite.
    '''
    return J

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_community_matrix(
    A: "np.ndarray",
    phi: "np.ndarray",
    gamma: "np.ndarray",
    psi: "np.ndarray",
    mu: "np.ndarray",
    lam: "np.ndarray",
    alpha: "np.ndarray",
    rho: "np.ndarray",
    sigma: "np.ndarray",
    chi: "np.ndarray",
    beta: "np.ndarray",
) -> "np.ndarray":
    A = np.asarray(A, dtype=float)
    phi = np.asarray(phi, dtype=float)
    gamma = np.asarray(gamma, dtype=float)
    psi = np.asarray(psi, dtype=float)
    mu = np.asarray(mu, dtype=float)
    lam = np.asarray(lam, dtype=float)
    alpha = np.asarray(alpha, dtype=float)
    rho = np.asarray(rho, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    chi = np.asarray(chi, dtype=float)
    beta = np.asarray(beta, dtype=float)
    if A.ndim != 2 or A.shape[0] != A.shape[1] or A.shape[0] < 1:
        raise ValueError("A must be a square 2D array with shape (N,N), N>=1")
    if np.any((A != 0.0) & (A != 1.0)):
        raise ValueError("A entries must be 0 or 1")
    N = A.shape[0]
    for name, v in [("phi", phi), ("gamma", gamma), ("psi", psi), ("mu", mu),
                    ("alpha", alpha), ("rho", rho), ("sigma", sigma)]:
        if v.shape != (N,) or not np.all(np.isfinite(v)):
            raise ValueError(name + " must be finite with shape (N,)")
    for name, m in [("lam", lam), ("chi", chi), ("beta", beta)]:
        if m.shape != (N, N) or not np.all(np.isfinite(m)):
            raise ValueError(name + " must be finite with shape (N, N)")
    J = np.zeros((N, N), dtype=float)
    for i in range(N):
        for j in range(N):
            if i == j:
                val = (1.0 - rho[i]) * phi[i]
                val += rho[i] * (gamma[i] * chi[i, i] * lam[i, i] + psi[i])
                val -= (1.0 - sigma[i]) * mu[i]
                val -= sigma[i] * np.sum(beta[:, i] * lam[:, i] * ((gamma - 1.0) * chi[:, i] + 1.0))
            else:
                val = 0.0
                if chi[i, j] != 0.0:
                    val += rho[i] * gamma[i] * chi[i, j] * lam[i, j]
                if beta[j, i] != 0.0:
                    val -= sigma[i] * beta[j, i] * psi[j]
                mask = (beta[:, i] != 0.0) & (chi[:, j] != 0.0)
                val -= sigma[i] * np.sum(beta[:, i] * lam[:, j] * (gamma - 1.0) * chi[:, j] * mask)
            J[i, j] = alpha[i] * val
    return J

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": """import numpy as np
A = np.array([[0,0,0],[0,0,0],[1,1,0]], dtype=float)
rho, sigma, chi, beta = _oracle_feeding_branching(A)
alpha = _oracle_row_timescales(A, 42.0)
phi = np.array([0.2, 0.5, 0.0]); gamma = np.array([1.0, 1.0, 1.2])
psi = np.array([1.0, 1.0, 0.8]); mu = np.array([1.5, 1.5, 1.8])
lam = np.ones((3, 3))
""",
            "call": "community_matrix(A, phi, gamma, psi, mu, lam, alpha, rho, sigma, chi, beta)",
            "gold_call": "_oracle_community_matrix(A, phi, gamma, psi, mu, lam, alpha, rho, sigma, chi, beta)",
        },
        {
            "setup": """import numpy as np
A = np.array([[0]], dtype=float)
rho, sigma, chi, beta = _oracle_feeding_branching(A)
alpha = _oracle_row_timescales(A, 42.0)
phi = np.array([0.3]); gamma = np.array([1.0]); psi = np.array([1.0]); mu = np.array([1.2])
lam = np.ones((1, 1))
""",
            "call": "community_matrix(A, phi, gamma, psi, mu, lam, alpha, rho, sigma, chi, beta)",
            "gold_call": "_oracle_community_matrix(A, phi, gamma, psi, mu, lam, alpha, rho, sigma, chi, beta)",
        },
        {
            "setup": """import numpy as np
A = np.array([[0,0,0,0,0,0],[0,0,0,0,0,0],[1,1,0,0,0,0],[0,1,0,0,0,0],[1,0,1,0,0,0],[0,1,0,1,1,0]], dtype=float)
rho, sigma, chi, beta = _oracle_feeding_branching(A)
alpha = _oracle_row_timescales(A, 42.0)
phi = np.array([0.22,0.38,0,0,0,0])
gamma = np.array([0.95,1.10,0.85,1.25,0.70,1.05])
psi = np.array([0.90,0.80,0.60,0.50,0.85,1.00])
mu = np.array([1.60,1.45,1.70,1.80,1.30,2.00])
lam = np.ones((6, 6)); lam[5, 4] = 0.75; lam[4, 0] = 1.20
""",
            "call": "community_matrix(A, phi, gamma, psi, mu, lam, alpha, rho, sigma, chi, beta)",
            "gold_call": "_oracle_community_matrix(A, phi, gamma, psi, mu, lam, alpha, rho, sigma, chi, beta)",
        },
    ]
