#!/bin/bash
# Oracle: writes the gold solution so the verifier scores reward 1.0.
cat > /app/solution.py <<'SCICODE_GOLD_EOF'
import numpy as np
try:
    import scipy, scipy.linalg, scipy.sparse, scipy.optimize, scipy.spatial, scipy.integrate, scipy.stats, scipy.special, scipy.signal
except Exception:
    pass
import math

def parallel_rates(u0: float, u1: float, u2: float, w1: float, beta0: float, x: float, y: float) -> "np.ndarray":
    import numpy as np
    alpha0 = u0 * x
    alpha1 = u1 * y
    alpha2 = u2 / x
    beta1 = w1 / y
    keq = u1 / w1
    keq_star = alpha1 / beta1
    return np.array([alpha0, alpha1, alpha2, beta1, keq, keq_star], dtype=float)

def master_generator(u0: float, u1: float, u2: float, w0: float, w1: float, beta0: float, x: float, y: float, gamma1: float, gamma2: float) -> "np.ndarray":
    import numpy as np
    alpha0, alpha1, alpha2, beta1, _, _ = parallel_rates(u0, u1, u2, w1, beta0, x, y)
    q = np.zeros((6, 6), dtype=float)
    q[0,1] += u0;           q[0,2] += u1;          q[0,3] += gamma1
    q[1,0] += w0 + u2;      q[1,4] += gamma1
    q[2,0] += w1;           q[2,5] += gamma1
    q[3,4] += alpha0;       q[3,5] += alpha1;      q[3,0] += gamma2
    q[4,3] += beta0 + alpha2; q[4,1] += gamma2
    q[5,3] += beta1;        q[5,2] += gamma2
    q[np.diag_indices(6)] = -q.sum(axis=1)
    return q

def product_first_passage(generator: "np.typing.ArrayLike", u2: float, alpha2: float) -> "np.ndarray":
    import numpy as np
    q = np.asarray(generator, dtype=float)
    transient = q.copy()
    absorb = np.zeros((6, 2), dtype=float)
    transient[1,0] -= u2
    transient[4,3] -= alpha2
    absorb[1,0] = u2
    absorb[4,1] = alpha2
    times = np.linalg.solve(transient, -np.ones(6))
    probs = np.linalg.solve(transient, -absorb)
    return np.column_stack((times, probs))

def nonrenewal_turnover(first_passage: "np.typing.ArrayLike") -> "np.ndarray":
    import numpy as np
    fp = np.asarray(first_passage, dtype=float)
    p = fp[[0,3], 1:3]
    a = p.T - np.eye(2)
    a[-1,:] = 1.0
    b = np.array([0.0, 1.0])
    pi = np.linalg.solve(a, b)
    tau = float(pi @ fp[[0,3], 0])
    return np.array([tau, pi[0], pi[1], p[0,0], p[0,1], p[1,0], p[1,1]], dtype=float)

def static_turnover(u0: float, u1: float, u2: float, w0: float, w1: float) -> float:
    return float((1.0/u0 + w0/(u0*u2)) * (1.0 + u1/w1) + 1.0/u2)

def catalytic_efficiency(static_time: float, dynamic_time: float) -> float:
    return float(static_time / dynamic_time)

def stationary_distribution(generator: "np.typing.ArrayLike") -> "np.ndarray":
    import numpy as np
    q = np.asarray(generator, dtype=float)
    a = q.T.copy()
    a[-1,:] = 1.0
    b = np.zeros(q.shape[0], dtype=float)
    b[-1] = 1.0
    return np.linalg.solve(a, b)

import numpy as np

def cycle_dissipation(stationary: "np.typing.ArrayLike", u0: float, u1: float, u2: float, w0: float, w1: float, beta0: float, x: float, y: float, gamma1: float, gamma2: float) -> "np.ndarray":
    p = np.asarray(stationary, dtype=float)
    alpha0, alpha1, alpha2, beta1, _, _ = parallel_rates(u0, u1, u2, w1, beta0, x, y)
    j_sub = p[1] * gamma1 - p[4] * gamma2
    j_inh = p[2] * gamma1 - p[5] * gamma2
    a_sub = np.log((u0 * (beta0 + alpha2)) / ((w0 + u2) * alpha0))
    a_inh = np.log((u1 * beta1) / (w1 * alpha1))
    j_product = p[1] * u2 + p[4] * alpha2
    sigma = j_sub * a_sub + j_inh * a_inh
    delta_w = sigma / j_product
    return np.array([j_sub, j_inh, a_sub, a_inh, j_product, sigma, delta_w], dtype=float)

def energy_efficiency(delta_w: float, efficiency: float) -> float:
    return float(delta_w / (efficiency - 1.0))

def solve(data: "dict") -> float:
    rates = parallel_rates(data['u0'], data['u1'], data['u2'], data['w1'], data['beta0'], data['x'], data['y'])
    q = master_generator(data['u0'], data['u1'], data['u2'], data['w0'], data['w1'], data['beta0'], data['x'], data['y'], data['gamma1'], data['gamma2'])
    fp = product_first_passage(q, data['u2'], float(rates[2]))
    turnover = nonrenewal_turnover(fp)
    tau_static = static_turnover(data['u0'], data['u1'], data['u2'], data['w0'], data['w1'])
    ef = catalytic_efficiency(tau_static, float(turnover[0]))
    stationary = stationary_distribution(q)
    diss = cycle_dissipation(stationary, data['u0'], data['u1'], data['u2'], data['w0'], data['w1'], data['beta0'], data['x'], data['y'], data['gamma1'], data['gamma2'])
    return energy_efficiency(float(diss[-1]), ef)
SCICODE_GOLD_EOF
