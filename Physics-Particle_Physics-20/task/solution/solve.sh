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


def generate_channel_parameters(n_channels: int, seed: int):
    if isinstance(n_channels, bool) or not isinstance(n_channels, (int, np.integer)):
        raise ValueError("n_channels must be a positive integer")
    if n_channels < 1:
        raise ValueError("n_channels must be a positive integer")
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an integer")
    rng = np.random.default_rng(int(seed))
    u = rng.random((int(n_channels), 3))
    s = 0.8 + 4.2 * u[:, 0]
    b = 0.3 + 5.7 * u[:, 1]
    tau = 0.5 + 2.5 * u[:, 2]
    return np.column_stack([s, b, tau]).astype(float)

import numpy as np


def compute_asimov_counts(params):
    p = np.asarray(params, dtype=float)
    if p.ndim != 2 or p.shape[1] != 3:
        raise ValueError("params must be a 2-D array with 3 columns")
    if not np.all(np.isfinite(p)):
        raise ValueError("params must contain only finite values")
    if not np.all(p > 0.0):
        raise ValueError("all s, b, tau must be strictly positive")
    if np.any(p > 1.0e12):
        raise ValueError("all s, b, tau must not exceed 1e12")
    s, b, tau = p[:, 0], p[:, 1], p[:, 2]
    n = s + b
    m = tau * b
    return np.column_stack([n, m]).astype(float)

import math

import numpy as np


def compute_profiled_background(n: float, m: float, tau: float, s: float) -> float:
    vals = [n, m, tau, s]
    if not all(np.isfinite(float(v)) for v in vals):
        raise ValueError("all arguments must be finite")
    n, m, tau, s = (float(v) for v in vals)
    if n < 0.0 or m < 0.0:
        raise ValueError("n and m must be non-negative")
    if tau <= 0.0:
        raise ValueError("tau must be strictly positive")
    if s < 0.0:
        raise ValueError("s must be non-negative")
    if max(n, m, tau, s) > 1.0e12:
        raise ValueError("n, m, tau and s must not exceed 1e12")
    a = n + m - (1.0 + tau) * s
    c_root = 2.0 * math.sqrt(1.0 + tau) * math.sqrt(s) * math.sqrt(m)
    if c_root == 0.0:
        return float(a / (1.0 + tau)) if a > 0.0 else 0.0
    disc = math.hypot(a, c_root)
    num = (a + disc) if a >= 0.0 else (c_root * c_root) / (disc - a)
    return float(num / (2.0 * (1.0 + tau)))

import math

import numpy as np


def _poisson_deviance(a: float, e: float) -> float:
    if a == 0.0:
        return 2.0 * e
    x = (a - e) / e
    if abs(x) < 1.0e-4:
        return 2.0 * e * (x * x / 2.0 - x ** 3 / 6.0 + x ** 4 / 12.0 - x ** 5 / 20.0)
    return 2.0 * (a * math.log(a / e) - (a - e))


def compute_signed_root(n: float, m: float, tau: float) -> float:
    vals = [n, m, tau]
    if not all(np.isfinite(float(v)) for v in vals):
        raise ValueError("all arguments must be finite")
    n, m, tau = (float(v) for v in vals)
    if n <= 0.0 or m <= 0.0:
        raise ValueError("n and m must be strictly positive")
    if tau <= 0.0:
        raise ValueError("tau must be strictly positive")
    if max(n, m, tau) > 1.0e12:
        raise ValueError("n, m and tau must not exceed 1e12")
    b00 = compute_profiled_background(n, m, tau, 0.0)
    q = _poisson_deviance(n, b00) + _poisson_deviance(m, tau * b00)
    if q < 0.0:
        q = 0.0
    d = n - m / tau
    return 0.0 if d == 0.0 else float(math.copysign(math.sqrt(q), d))

import math

import numpy as np


def compute_auxiliary_statistic(n: float, m: float, tau: float) -> float:
    vals = [n, m, tau]
    if not all(np.isfinite(float(v)) for v in vals):
        raise ValueError("all arguments must be finite")
    n, m, tau = (float(v) for v in vals)
    if n <= 0.0 or m <= 0.0:
        raise ValueError("n and m must be strictly positive")
    if tau <= 0.0:
        raise ValueError("tau must be strictly positive")
    if max(n, m, tau) > 1.0e12:
        raise ValueError("n, m and tau must not exceed 1e12")
    scale = math.sqrt(n) * math.sqrt(m / (n + m))
    return float(scale * math.log(n * tau / m))

import numpy as np


def compute_corrected_significance(r0: float, u0: float) -> float:
    if not (np.isfinite(float(r0)) and np.isfinite(float(u0))):
        raise ValueError("r0 and u0 must be finite")
    r0, u0 = float(r0), float(u0)
    if r0 == 0.0 or u0 == 0.0:
        rstar = r0
    else:
        rstar = r0 + (1.0 / r0) * np.log(abs(u0 / r0))
    return float(max(0.0, rstar))

import numpy as np
from scipy import special


def compute_signal_probability(n: float, m: float, tau: float) -> float:
    vals = [n, m, tau]
    if not all(np.isfinite(float(v)) for v in vals):
        raise ValueError("all arguments must be finite")
    n, m, tau = (float(v) for v in vals)
    if n < 0.0 or m < 0.0:
        raise ValueError("n and m must be non-negative")
    if tau <= 0.0:
        raise ValueError("tau must be strictly positive")
    if max(n, m, tau) > 1.0e12:
        raise ValueError("n, m and tau must not exceed 1e12")
    sigma = 0.5
    x = tau / (1.0 + tau)
    a1 = m + 1.0 - sigma
    a2 = n - sigma + 1.0
    return float(special.betainc(a1, a2, x))

import numpy as np
from scipy import special


def compute_combined_sensitivity(n_channels: int, seed: int, threshold: float) -> float:
    if isinstance(n_channels, bool) or not isinstance(n_channels, (int, np.integer)):
        raise ValueError("n_channels must be a positive integer")
    if n_channels < 1:
        raise ValueError("n_channels must be a positive integer")
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an integer")
    t = float(threshold)
    if not np.isfinite(t) or not (0.0 < t < 1.0):
        raise ValueError("threshold must be finite and strictly between 0 and 1")
    params = generate_channel_parameters(n_channels, seed)
    counts = compute_asimov_counts(params)
    total = 0.0
    for i in range(params.shape[0]):
        tau_i = float(params[i, 2])
        n_i, m_i = float(counts[i, 0]), float(counts[i, 1])
        p_i = compute_signal_probability(n_i, m_i, tau_i)
        if p_i > t:
            r0_i = compute_signed_root(n_i, m_i, tau_i)
            u0_i = compute_auxiliary_statistic(n_i, m_i, tau_i)
            z_i = compute_corrected_significance(r0_i, u0_i)
            total += z_i * z_i
    return float(np.sqrt(total))
SCICODE_GOLD_EOF
