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
import math

def compton_kinematics_lo(E_gamma: float, m: float, alpha: float) -> tuple:
    E_gamma = float(E_gamma); m = float(m); alpha = float(alpha)
    if not (E_gamma > 0.0 and m > 0.0 and alpha > 0.0):
        raise ValueError("E_gamma, m and alpha must be strictly positive")
    s = m * m + 2.0 * m * E_gamma
    tau = m * m / s
    x = E_gamma / m
    u = 2.0 * x
    l = math.log1p(u)
    # The combination g(u) = u (1 + u/2) / (1 + u) - log1p(u) is O(u^3) and
    # cancels catastrophically for u << 1 (it costs ~3 digits per decade, e.g.
    # 32 % error at E_gamma = 10 keV(-ish) when evaluated directly).  For small
    # u it is summed from its convergent series
    #   g(u) = u^2 sum_{k>=1} (-1)^(k+1) u^k k / (2 (k + 2)),
    # which is exact to double precision; the closed form is used otherwise.
    if u < 0.5:
        term = u ** 3 / 6.0          # k = 1
        g = term
        k = 2
        while abs(term) > 1e-19 * abs(g) and k < 200:
            term = (-1.0) ** (k + 1) * u ** (k + 2) * k / (2.0 * (k + 2))
            g += term
            k += 1
    else:
        g = u * (1.0 + 0.5 * u) / (1.0 + u) - l
    sigma = 2.0 * math.pi * (alpha / m) ** 2 * ((1.0 + x) / x ** 3 * g
                                                + l / u - (1.0 + 3.0 * x) / (1.0 + u) ** 2)
    return float(s), float(tau), float(sigma)

import numpy as np
import math

def lo_high_energy_series(tau: float, s: float, alpha: float) -> float:
    tau = float(tau); s = float(s); alpha = float(alpha)
    if not (0.0 < tau <= 0.01) or not (s > 0.0) or not (alpha > 0.0):
        raise ValueError("require 0 < tau <= 0.01, s > 0, alpha > 0")
    L = math.log(1.0 / tau)
    # eq. (7), leading-order line
    poly = (2.0 * L + 1.0) + tau * (-6.0 * L + 17.0) + tau ** 2 * (-30.0 * L + 32.0) \
        + tau ** 3 * (-70.0 * L + 48.0) + tau ** 4 * (-126.0 * L + 64.0)
    return float(math.pi * alpha ** 2 / s * poly)

import numpy as np
import math

def nlo_high_energy_series(tau: float, s: float, alpha: float) -> float:
    tau = float(tau); s = float(s); alpha = float(alpha)
    if not (0.0 < tau <= 0.01) or not (s > 0.0) or not (alpha > 0.0):
        raise ValueError("require 0 < tau <= 0.01, s > 0, alpha > 0")
    L = math.log(1.0 / tau)
    # eq. (7), next-to-leading-order line
    poly = (L ** 3 / 3.0 - L ** 2 / 2.0 + 17.0 * L / 4.0 - 9.5016) \
        + tau * (2.0 * L ** 3 + 13.0 * L ** 2 - 36.5340 * L + 0.55139) \
        + tau ** 2 * (38.0 * L ** 3 / 3.0 + 151.0 * L ** 2 / 2.0 - 89.091 * L + 47.062) \
        + tau ** 3 * (157.0 * L ** 3 / 3.0 + 344.0 * L ** 2 / 3.0 - 220.07 * L + 149.73)
    return float(alpha ** 3 / s * poly)

import numpy as np
import math

def nnlo_high_energy_series(tau: float, s: float, alpha: float) -> float:
    tau = float(tau); s = float(s); alpha = float(alpha)
    if not (0.0 < tau <= 0.01) or not (s > 0.0) or not (alpha > 0.0):
        raise ValueError("require 0 < tau <= 0.01, s > 0, alpha > 0")
    L = math.log(1.0 / tau)
    # eq. (7), next-to-next-to-leading-order line
    poly = (L ** 5 / 48.0 - 0.149 * L ** 4 + 1.34 * L ** 3 - 1.17 * L ** 2 - 16.6 * L + 23.8) \
        + tau * (1.13 * L ** 5 - 6.54 * L ** 4 - 44.6 * L ** 3 + 442.0 * L ** 2 - 36.0 * L - 2.64e3)
    return float(alpha ** 4 / (math.pi * s) * poly)

import numpy as np
import math

def ll_ladder_term(n: int, tau: float, s: float, alpha: float) -> float:
    n = int(n); tau = float(tau); s = float(s); alpha = float(alpha)
    if n < 1 or not (0.0 < tau < 1.0) or not (s > 0.0) or not (alpha > 0.0):
        raise ValueError("require n >= 1, 0 < tau < 1, s > 0, alpha > 0")
    # eq. (12): sigma_LL^{N^{n-1}LO} = -2 alpha^{n+1} ln^{2n-1} tau / ((2 pi)^{n-2} s (n-1)! (n+1)!)
    lt = math.log(tau)
    return float(-2.0 * alpha ** (n + 1) * lt ** (2 * n - 1)
                 / ((2.0 * math.pi) ** (n - 2) * s * math.factorial(n - 1) * math.factorial(n + 1)))

import numpy as np
import math
from scipy.special import iv

def ll_resummed(tau: float, s: float, alpha: float, n_max: int) -> tuple:
    tau = float(tau); s = float(s); alpha = float(alpha); n_max = int(n_max)
    if not (0.0 < tau < 1.0) or not (s > 0.0) or not (alpha > 0.0) or n_max < 0:
        raise ValueError("require 0 < tau < 1, s > 0, alpha > 0, n_max >= 0")
    lt = math.log(tau)
    # eq. (13): sigma_LL = -(8 pi^2 alpha / (s ln tau)) I_2( sqrt(2 alpha/pi) ln tau )
    sigma_ll = -8.0 * math.pi ** 2 * alpha / (s * lt) * float(iv(2, math.sqrt(2.0 * alpha / math.pi) * lt))
    fixed = sum(ll_ladder_term(n, tau, s, alpha) for n in range(1, n_max + 1))
    return float(sigma_ll), float(sigma_ll - fixed)

import numpy as np

def compton_nnlo_ll(E_gamma: float, m: float, alpha: float) -> float:
    s, tau, _sigma_kn = compton_kinematics_lo(E_gamma, m, alpha)
    sigma = lo_high_energy_series(tau, s, alpha) \
        + nlo_high_energy_series(tau, s, alpha) \
        + nnlo_high_energy_series(tau, s, alpha)
    _sigma_ll, remainder = ll_resummed(tau, s, alpha, 3)
    sigma += remainder
    return float(sigma * 389.3793721)

import math
import numpy as np

def spectrum_averaged_cross_section(E_min: float, E_max: float, m: float, alpha: float, n_nodes: int) -> float:
    if not (E_min > 0.0 and E_max > E_min and int(n_nodes) >= 1):
        raise ValueError("require 0 < E_min < E_max and n_nodes >= 1")
    x, w = np.polynomial.legendre.leggauss(int(n_nodes))
    u_min, u_max = math.log(E_min), math.log(E_max)
    u = 0.5 * (u_max - u_min) * x + 0.5 * (u_max + u_min)
    total = 0.0
    for u_i, w_i in zip(u, w):
        total += float(w_i) * compton_nnlo_ll(math.exp(float(u_i)), m, alpha)
    return float(0.5 * total)
SCICODE_GOLD_EOF
