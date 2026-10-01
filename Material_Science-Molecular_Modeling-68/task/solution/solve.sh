#!/bin/bash
# Oracle: writes the gold solution so the verifier scores reward 1.0.
cat > /app/solution.py <<'SCICODE_GOLD_EOF'
import numpy as np
try:
    import scipy, scipy.linalg, scipy.sparse, scipy.optimize, scipy.spatial, scipy.integrate, scipy.stats, scipy.special, scipy.signal
except Exception:
    pass
import math

import numpy as np
def average_descriptor(G: np.ndarray) -> np.ndarray:
    """Reference implementation."""
    G = np.asarray(G, dtype=float)
    if G.ndim != 2:
        raise ValueError("G must be a 2D array of shape (N_atoms, D)")
    if G.shape[0] == 0:
        raise ValueError("G must contain at least one atom (N_atoms >= 1)")
    if not np.all(np.isfinite(G)):
        raise ValueError("G must contain only finite values")
    return G.mean(axis=0).astype(float)

import numpy as np

def compute_pca_basis(S_ref: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:

    """Reference implementation."""

    S_ref = np.asarray(S_ref, dtype=float)

    if S_ref.ndim != 2:

        raise ValueError("S_ref must be a 2D array of shape (N_ref, D)")

    if S_ref.shape[0] < 2:

        raise ValueError("S_ref must contain at least 2 reference rows")

    if not np.all(np.isfinite(S_ref)):

        raise ValueError("S_ref must contain only finite values")

    D = S_ref.shape[1]

    if not isinstance(k, (int, np.integer)) or not (1 <= k <= D):

        raise ValueError(f"k must be an integer with 1 <= k <= {D}")


    mu = S_ref.mean(axis=0)

    S_hat = S_ref - mu

    _, _, Vt = np.linalg.svd(S_hat, full_matrices=True)

    V = Vt.T

    V_k = V[:, :k]

    return mu.astype(float), V_k.astype(float)

import numpy as np
def project_to_cv(S: np.ndarray, mu: np.ndarray, V_k: np.ndarray) -> np.ndarray:
    """Reference implementation."""
    S = np.asarray(S, dtype=float)
    mu = np.asarray(mu, dtype=float)
    V_k = np.asarray(V_k, dtype=float)
    if S.ndim != 2:
        raise ValueError("S must be a 2D array of shape (N, D)")
    if mu.ndim != 1:
        raise ValueError("mu must be a 1D array of shape (D,)")
    if V_k.ndim != 2:
        raise ValueError("V_k must be a 2D array of shape (D, k)")
    if S.shape[1] != mu.shape[0] or mu.shape[0] != V_k.shape[0]:
        raise ValueError("S, mu, and V_k must have consistent descriptor dimension D")
    if not (np.all(np.isfinite(S)) and np.all(np.isfinite(mu)) and np.all(np.isfinite(V_k))):
        raise ValueError("S, mu, and V_k must contain only finite values")
    return ((S - mu) @ V_k).astype(float)

import numpy as np
def gaussian_kernel(s: np.ndarray, s_j: np.ndarray, sigma: float) -> float:
    """Reference implementation."""
    s = np.asarray(s, dtype=float)
    s_j = np.asarray(s_j, dtype=float)
    if s.ndim != 1 or s_j.ndim != 1:
        raise ValueError("s and s_j must be 1D arrays")
    if s.shape != s_j.shape:
        raise ValueError("s and s_j must have the same shape")
    if not np.all(np.isfinite(s)) or not np.all(np.isfinite(s_j)):
        raise ValueError("s and s_j must contain only finite values")
    if not (isinstance(sigma, (int, float)) and np.isfinite(sigma) and sigma > 0.0):
        raise ValueError("sigma must be a finite number > 0")

    k = s.shape[0]
    sigma2 = float(sigma) ** 2
    diff = s - s_j
    quad = np.dot(diff, diff) / sigma2
    norm_const = 1.0 / np.sqrt((sigma2 ** k) * (2 * np.pi) ** k)
    return float(norm_const * np.exp(-0.5 * quad))

import numpy as np
def density_estimate(s: np.ndarray, S_cv: np.ndarray, sigma: float) -> float:
    """Reference implementation."""
    s = np.asarray(s, dtype=float)
    S_cv = np.asarray(S_cv, dtype=float)
    if s.ndim != 1:
        raise ValueError("s must be a 1D array")
    if S_cv.ndim != 2:
        raise ValueError("S_cv must be a 2D array of shape (N_ref, k)")
    if S_cv.shape[0] == 0:
        raise ValueError("S_cv must contain at least one reference point")
    if s.shape[0] != S_cv.shape[1]:
        raise ValueError("s and S_cv must have matching CV dimension k")
    if not (isinstance(sigma, (int, float)) and np.isfinite(sigma) and sigma > 0.0):
        raise ValueError("sigma must be a finite number > 0")

    kernels = [gaussian_kernel(s, s_j, sigma) for s_j in S_cv]
    return float(np.mean(kernels))

import numpy as np
def normalization_constant(S_cv: np.ndarray, sigma: float) -> float:
    """Reference implementation. Leave-one-out: each center's density
    contribution excludes its own self-kernel term."""
    S_cv = np.asarray(S_cv, dtype=float)
    if S_cv.ndim != 2:
        raise ValueError("S_cv must be a 2D array of shape (N_ref, k)")
    if S_cv.shape[0] < 2:
        raise ValueError("S_cv must contain at least 2 reference points for leave-one-out")
    if not (isinstance(sigma, (int, float)) and np.isfinite(sigma) and sigma > 0.0):
        raise ValueError("sigma must be a finite number > 0")

    N = S_cv.shape[0]
    loo_vals = []
    for j in range(N):
        others = np.delete(S_cv, j, axis=0)
        d_j = np.mean([gaussian_kernel(S_cv[j], s_k, sigma) for s_k in others])
        loo_vals.append(d_j)
    return float(np.mean(loo_vals))

import numpy as np

def thermo_params(T: float, delta_E: float) -> tuple[float, float, float]:
    """Reference implementation."""
    _KB_EV_PER_K = 8.617333262e-5  # Boltzmann constant in eV/K
    if not (isinstance(T, (int, float)) and np.isfinite(T) and T > 0.0):
        raise ValueError("T must be a finite number > 0")
    if not (isinstance(delta_E, (int, float)) and np.isfinite(delta_E) and delta_E > 0.0):
        raise ValueError("delta_E must be a finite number > 0")

    beta = 1.0 / (_KB_EV_PER_K * float(T))
    gamma = beta * float(delta_E)
    if abs(gamma - 1.0) < 1e-9:
        raise ValueError("gamma must not be within 1e-9 of 1.0 (division by zero in epsilon)")
    epsilon = float(np.exp(-gamma / (gamma - 1.0)))
    return float(beta), float(gamma), epsilon

import numpy as np
def bias_potential(
    G: np.ndarray,
    S_ref: np.ndarray,
    k: int,
    sigma: float,
    T: float,
    delta_E: float,
) -> float:
    """Reference implementation. Chains ORACLE functions only."""
    s_prime = average_descriptor(G)
    mu, V_k = compute_pca_basis(S_ref, k)

    s_cv = project_to_cv(s_prime.reshape(1, -1), mu, V_k)[0]
    S_cv = project_to_cv(S_ref, mu, V_k)

    p_current = density_estimate(s_cv, S_cv, sigma)
    Z_n = normalization_constant(S_cv, sigma)
    beta, gamma, epsilon = thermo_params(T, delta_E)

    V_n = (gamma - 1.0) * (1.0 / beta) * np.log(p_current / Z_n + epsilon)
    return float(V_n)
SCICODE_GOLD_EOF
