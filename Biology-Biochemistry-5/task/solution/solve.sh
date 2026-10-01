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
from math import comb
from scipy.linalg import expm


def _check_theta(theta):
    th = np.asarray(theta, dtype=np.float64)
    if th.shape != (3,):
        raise ValueError("theta must be the three rate constants (theta_1, theta_2, theta_3)")
    if not np.all(np.isfinite(th)) or np.any(th <= 0.0):
        raise ValueError("every rate constant must be finite and strictly positive")
    return th


def _propensities(x, th):
    """lambda_j(x) for the three reactions of Table I at copy number x."""
    return np.array([th[0], th[1] * x, th[2] * x * (x - 1.0)], dtype=np.float64)


def truncated_generator(n_states: int, theta: "np.ndarray") -> "np.ndarray":
    """Eq. (7) restricted to the truncated state space S = {0, ..., N-1} of eq. (20)."""
    if isinstance(n_states, bool) or int(n_states) != n_states or int(n_states) < 2:
        raise ValueError("n_states must be an integer >= 2")
    N = int(n_states)
    th = _check_theta(theta)
    stoich = (1, -1, -2)
    L = np.zeros((N, N), dtype=np.float64)
    for x in range(N):
        lam = _propensities(float(x), th)
        for lj, s in zip(lam, stoich):
            # the diagonal carries the outflow of every reaction, including the R1 jump
            # from x = N-1 that leaves S: that lost mass is exactly the boundary input
            L[x, x] -= lj
            if 0 <= x + s < N:
                L[x, x + s] += lj
    return L

import numpy as np
from math import comb
from scipy.linalg import expm


def _check_theta(theta):
    th = np.asarray(theta, dtype=np.float64)
    if th.shape != (3,):
        raise ValueError("theta must be the three rate constants (theta_1, theta_2, theta_3)")
    if not np.all(np.isfinite(th)) or np.any(th <= 0.0):
        raise ValueError("every rate constant must be finite and strictly positive")
    return th


def boundary_input_column(n_states: int, theta: "np.ndarray") -> "np.ndarray":
    """Eq. (21): the column through which the boundary conditional moment drives q."""
    if isinstance(n_states, bool) or int(n_states) != n_states or int(n_states) < 2:
        raise ValueError("n_states must be an integer >= 2")
    N = int(n_states)
    th = _check_theta(theta)
    b = np.zeros(N, dtype=np.float64)
    # only R1 from x = N-1 reaches the boundary state x = N, with rate theta_1
    b[N - 1] = th[0]
    return b

import numpy as np
from math import comb
from scipy.linalg import expm


def _check_theta(theta):
    th = np.asarray(theta, dtype=np.float64)
    if th.shape != (3,):
        raise ValueError("theta must be the three rate constants (theta_1, theta_2, theta_3)")
    if not np.all(np.isfinite(th)) or np.any(th <= 0.0):
        raise ValueError("every rate constant must be finite and strictly positive")
    return th


def _shift_poly(p, k):
    """Multiply the polynomial coefficient vector p (x^0, x^1, x^2) by x^k, keeping degree <= 2."""
    out = np.zeros(3)
    for i, c in enumerate(p):
        if i + k <= 2:
            out[i + k] += c
    return out


def upper_input_polynomial(theta: "np.ndarray", mu: int) -> "np.ndarray":
    """Theorem 2, eq. (19): coefficients [c_0, c_1, c_2] of h+_mu(x) for mu in {1, 2}.

    First-order terms mu * s_j * lambda_j(x) * x^(mu-1) are summed over the
    NON-bimolecular reactions only (s_3 <= 0 makes the dropped term nonpositive);
    the |nu| >= 2 binomial terms C(mu, nu) s_j^nu lambda_j(x) x^(mu-nu) run over
    every reaction. Propensities are polynomials in x, so h+ is a polynomial.
    """
    th = _check_theta(theta)
    if isinstance(mu, bool) or int(mu) != mu or int(mu) not in (1, 2):
        raise ValueError("mu must be 1 or 2")
    mu = int(mu)
    th1, th2, th3 = th
    # lambda_j(x) as polynomial coefficient vectors [x^0, x^1, x^2]
    lam_poly = {1: np.array([th1, 0.0, 0.0]),
                2: np.array([0.0, th2, 0.0]),
                3: np.array([0.0, -th3, th3])}
    stoich = {1: 1.0, 2: -1.0, 3: -2.0}
    bimolecular = {3}

    h = np.zeros(3)
    for j in (1, 2, 3):
        if j not in bimolecular:
            h += mu * stoich[j] * _shift_poly(lam_poly[j], mu - 1)        # first-order, non-bimolecular
        for nu in range(2, mu + 1):                                # |nu| >= 2 terms, all reactions
            h += comb(mu, nu) * stoich[j] ** nu * _shift_poly(lam_poly[j], mu - nu)
    return h

import numpy as np
from math import comb
from scipy.linalg import expm


def input_bound_ode_matrix(coeffs_mu1: "np.ndarray", coeffs_mu2: "np.ndarray") -> "np.ndarray":
    """Eqs. (17), (22), (23): constant matrix of the closed LTI for z = [u_1+, u_2+, 1].

    Row mu reads d/dt u_mu+ = c_{mu,mu} u_mu+ + {c_{mu,nu}}^+ u_nu+ + {c_{mu,nu}}^- u_nu- + c_{mu,0},
    with u- = 0 (Sec. IV-A), so a NEGATIVE cross coefficient contributes nothing.
    """
    c1 = np.asarray(coeffs_mu1, dtype=np.float64)
    c2 = np.asarray(coeffs_mu2, dtype=np.float64)
    if c1.shape != (3,) or c2.shape != (3,):
        raise ValueError("each coefficient vector must have exactly three entries [c0, c1, c2]")
    if not (np.all(np.isfinite(c1)) and np.all(np.isfinite(c2))):
        raise ValueError("coefficients must be finite")
    if c1[2] != 0.0:
        raise ValueError("h+_1 must be affine in x (no x^2 term)")
    M = np.zeros((3, 3), dtype=np.float64)
    # u_1+: c_{1,1} u_1 + c_{1,0}
    M[0, 0] = c1[1]
    M[0, 2] = c1[0]
    # u_2+: c_{2,2} u_2 + [c_{2,1}]^+ u_1+ + [c_{2,1}]^- u_1- + c_{2,0};  u_1- = 0
    M[1, 1] = c2[2]
    M[1, 0] = max(c2[1], 0.0)
    M[1, 2] = c2[0]
    return M

import numpy as np
from math import comb
from scipy.linalg import expm


def bounding_output_at_time(generator: "np.ndarray", boundary_column: "np.ndarray",
                                     u_matrix: "np.ndarray", alpha: int, t: float) -> "np.ndarray":
    """Eqs. (12), (13): outputs [y+(t), y-(t)] of B+ and B- for f(x) = x^alpha, p_0 = e_0.

    B+ is integrated as one LTI on the augmented state [q, u_1+, u_2+, 1] with
    u_mu+(0) = N^mu (Lemma 2); B- uses u- = 0 so q- evolves under the generator alone.
    """
    L = np.asarray(generator, dtype=np.float64)
    b = np.asarray(boundary_column, dtype=np.float64)
    M = np.asarray(u_matrix, dtype=np.float64)
    if L.ndim != 2 or L.shape[0] != L.shape[1] or L.shape[0] < 2:
        raise ValueError("generator must be a square matrix of size >= 2")
    N = L.shape[0]
    if b.shape != (N,):
        raise ValueError("boundary_column must have one entry per truncated state")
    if M.shape != (3, 3):
        raise ValueError("u_matrix must be 3 x 3")
    if isinstance(alpha, bool) or int(alpha) != alpha or int(alpha) not in (1, 2):
        raise ValueError("alpha must be 1 or 2")
    if not np.isfinite(t) or t < 0.0:
        raise ValueError("t must be a finite nonnegative time")
    alpha = int(alpha)
    f0 = np.arange(N, dtype=np.float64) ** alpha
    A = np.zeros((N + 3, N + 3), dtype=np.float64)
    A[:N, :N] = L
    A[:N, N + alpha - 1] += b            # boundary input theta_1 u_alpha+(t) into row N-1
    A[N:, N:] = M
    z0 = np.concatenate([f0, [float(N), float(N) ** 2, 1.0]])
    y_plus = float((expm(A * float(t)) @ z0)[0])
    y_minus = float((expm(L * float(t)) @ f0)[0])
    return np.array([y_plus, y_minus], dtype=np.float64)

import numpy as np
from math import comb
from scipy.linalg import expm


def variance_bracket(y1_plus: float, y1_minus: float, y2_plus: float, y2_minus: float) -> "np.ndarray":
    """Sec. IV-A: guaranteed bracket [V-, V+] of V[X(t)] = E[X^2] - (E[X])^2 from the moment bounds.

    The upper variance bound pairs the UPPER second-moment bound with the LOWER mean bound,
    and the lower variance bound pairs the opposite pair.
    """
    vals = np.array([y1_plus, y1_minus, y2_plus, y2_minus], dtype=np.float64)
    if not np.all(np.isfinite(vals)):
        raise ValueError("moment bounds must be finite")
    if y1_plus < y1_minus or y2_plus < y2_minus:
        raise ValueError("each upper bound must be at least its lower bound")
    v_plus = y2_plus - y1_minus ** 2
    v_minus = y2_minus - y1_plus ** 2
    return np.array([v_plus, v_minus], dtype=np.float64)

import numpy as np
from math import comb
from scipy.linalg import expm


def _check_theta(theta):
    th = np.asarray(theta, dtype=np.float64)
    if th.shape != (3,):
        raise ValueError("theta must be the three rate constants (theta_1, theta_2, theta_3)")
    if not np.all(np.isfinite(th)) or np.any(th <= 0.0):
        raise ValueError("every rate constant must be finite and strictly positive")
    return th


def _check_n(n_states):
    if isinstance(n_states, bool) or int(n_states) != n_states or int(n_states) < 2:
        raise ValueError("n_states must be an integer >= 2")
    return int(n_states)


def variance_upper_bound(theta: "np.ndarray", n_states: int, t: float) -> float:
    """ORCHESTRATOR: the guaranteed upper bound V+(t) on V[X(t)] for X(0) = 0."""
    th = _check_theta(theta)
    N = _check_n(n_states)
    if not np.isfinite(t) or t < 0.0:
        raise ValueError("t must be a finite nonnegative time")
    L = truncated_generator(N, th)
    b = boundary_input_column(N, th)
    c1 = upper_input_polynomial(th, 1)
    c2 = upper_input_polynomial(th, 2)
    M = input_bound_ode_matrix(c1, c2)
    y1 = bounding_output_at_time(L, b, M, 1, t)
    y2 = bounding_output_at_time(L, b, M, 2, t)
    v = variance_bracket(float(y1[0]), float(y1[1]), float(y2[0]), float(y2[1]))
    return float(v[0])
SCICODE_GOLD_EOF
