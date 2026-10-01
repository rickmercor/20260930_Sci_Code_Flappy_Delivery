#!/bin/bash
# Oracle: writes the gold solution so the verifier scores reward 1.0.
cat > /app/solution.py <<'SCICODE_GOLD_EOF'
import numpy as np
try:
    import scipy, scipy.linalg, scipy.sparse, scipy.optimize, scipy.spatial, scipy.integrate, scipy.stats, scipy.special, scipy.signal
except Exception:
    pass
import math

import math
import numpy as np

def lepton_kinematics(E_e: float, m_e: float, E_0: float) -> tuple:
    E_e = float(E_e); m_e = float(m_e); E_0 = float(E_0)
    if not (m_e > 0.0 and E_0 > m_e and m_e < E_e < E_0):
        raise ValueError("require m_e > 0, E_0 > m_e and m_e < E_e < E_0")
    p_e = math.sqrt((E_e - m_e) * (E_e + m_e))
    beta = p_e / E_e
    return float(beta), float(p_e), float(E_0 - E_e)

import math
import numpy as np
from scipy.special import loggamma

def fermi_function_msbar(beta: float, Z: int, mubar: float, m_e: float, alpha: float) -> float:
    beta = float(beta); Z = int(Z); mubar = float(mubar); m_e = float(m_e); alpha = float(alpha)
    if not (0.0 < beta < 1.0) or Z < 1 or mubar <= 0.0 or m_e <= 0.0 or not (0.0 < alpha * Z < 1.0):
        raise ValueError("require 0 < beta < 1, Z >= 1, mubar > 0, m_e > 0, 0 < alpha Z < 1")
    eta = math.sqrt(1.0 - (alpha * Z) ** 2)
    y = -Z * alpha / beta                       # positron emission
    p_e = beta * m_e / math.sqrt(1.0 - beta * beta)
    # |Gamma(eta + i y)|^2 e^{pi y} evaluated through log-gamma for stability
    log_abs_gamma_sq = 2.0 * loggamma(complex(eta, y)).real
    pref = 4.0 * eta / (1.0 + eta) ** 2 * 2.0 * (1.0 + eta) / math.gamma(2.0 * eta + 1.0) ** 2
    scale = (2.0 * p_e * math.exp(-np.euler_gamma) / mubar) ** (2.0 * (eta - 1.0))
    return float(pref * math.exp(log_abs_gamma_sq + math.pi * y) * scale)

import math
import numpy as np
from scipy.special import spence

def sirlin_g1(beta: float, Ebar: float, mubar: float, m_e: float) -> float:
    beta = float(beta); Ebar = float(Ebar); mubar = float(mubar); m_e = float(m_e)
    if not (0.0 < beta < 1.0) or Ebar <= 0.0 or mubar <= 0.0 or m_e <= 0.0:
        raise ValueError("require 0 < beta < 1, Ebar > 0, mubar > 0, m_e > 0")
    Li2 = lambda x: float(spence(1.0 - x))          # scipy's spence(1-x) is Li_2(x)
    E_e = m_e / math.sqrt((1.0 - beta) * (1.0 + beta))
    L_mu = math.log(mubar * mubar / (m_e * m_e))
    L = math.log((1.0 + beta) / (1.0 - beta))
    lm = math.log(m_e * m_e / (4.0 * Ebar * Ebar))
    r = Ebar / E_e
    val = (1.5 * L_mu
           - 4.0 / beta * (Li2(2.0 * beta / (1.0 + beta)) + 0.25 * L * L)
           + 2.0 * lm + 8.0 - 4.0 / 3.0 * r
           + L / beta * (-lm - 2.0 + beta * beta + r * r / 12.0 + 2.0 / 3.0 * r))
    return float(val / (2.0 * math.pi))

import math

from scipy.special import spence


def g2_alpha2z(beta: float, mubar: float, m_e: float) -> float:
    beta = float(beta)
    mubar = float(mubar)
    m_e = float(m_e)
    if not 0.0 < beta < 1.0 or mubar <= 0.0 or m_e <= 0.0:
        raise ValueError("require 0 < beta < 1, mubar > 0, m_e > 0")

    b = beta
    log_two = math.log(2.0)
    if 0.5 * m_e <= mubar <= 1.5 * m_e:
        log_scale = 2.0 * math.log1p((mubar - m_e) / m_e)
    else:
        log_scale = 2.0 * (math.log(mubar) - math.log(m_e))
    scale_term = (log_scale / 3.0) / b + 0.5 * log_scale

    if b < 0.02:
        # Positron expansion of Eq. (3.3), excluding the exact scale term.
        # Terms through b**7 avoid the cancellation of the closed form.
        coefficients = (
            37.0 / 8.0 - 4.0 * log_two,
            -4.0 / 3.0,
            2.0 * log_two / 3.0 - 239.0 / 288.0,
            -4.0 / 15.0,
            2.0 * log_two / 15.0 - 169.0 / 1152.0,
            -106.0 / 315.0,
            2.0 * log_two / 35.0 + 371.0 / 15360.0,
            -104.0 / 315.0,
        )
        regular = coefficients[-1]
        for coefficient in reversed(coefficients[:-1]):
            regular = coefficient + b * regular
        result = scale_term + regular
    else:
        q = (1.0 - b) / (1.0 + b)
        r = math.sqrt(q)
        log_ratio = math.log1p(b) - math.log1p(-b)
        # 1 - r, rationalized to avoid loss of significance.
        one_minus_r = 2.0 * b / ((1.0 + b) * (1.0 + r))

        # scipy.special.spence(z) = Li_2(1-z).
        li_q = float(spence(1.0 - q))
        li_r = float(spence(1.0 - r))
        li_q2 = float(spence(1.0 - q * q))
        braces_without_scale = math.fsum(
            (
                (9.0 - b * b) / (2.0 * b) * li_q,
                (b * b - 5.0) / b * li_r,
                -li_q2 / b,
                (3.0 - b * b) / (2.0 * b) * math.pi**2 / 6.0,
                -(4.0 - 5.0 * b + b**3)
                / (16.0 * b * b) * log_ratio**2,
                (b * b - 2.0) / (b * b) * math.log1p(r),
                -(2.0 * b * b + 2.0) / (b * b)
                * (math.log1p(b) - log_two),
                log_ratio / (12.0 * b * b)
                * (-6.0 + 10.0 * b + 3.0 * b * b + 3.0 * b**3),
                -4.0 / b,
                one_minus_r**3 * (1.0 + b)**2 / (144.0 * b**4)
                * (
                    r * (430.0 - 220.0 * b - 39.0 * b * b + 48.0 * b**3)
                    + 434.0 - 652.0 * b + 327.0 * b * b - 96.0 * b**3
                ),
            )
        )
        result = scale_term - braces_without_scale
    return float(result)

import math
import numpy as np

def spectrum_weight(p_e: float, Z: int, E_0: float, mubar: float, m_e: float, alpha: float) -> float:
    p_e = float(p_e); E_0 = float(E_0); m_e = float(m_e)
    if m_e <= 0.0 or E_0 <= m_e:
        raise ValueError("require m_e > 0 and E_0 > m_e")
    p_max = math.sqrt((E_0 - m_e) * (E_0 + m_e))
    if not (0.0 < p_e < p_max):
        raise ValueError("require 0 < p_e < sqrt(E_0^2 - m_e^2)")
    E_e = math.sqrt(p_e * p_e + m_e * m_e)
    beta, p_chk, Ebar = lepton_kinematics(E_e, m_e, E_0)      # Step 1
    Fbar = fermi_function_msbar(beta, Z, mubar, m_e, alpha)  # Step 2
    return float(p_chk * p_chk * Ebar * Ebar * Fbar)

import math
import numpy as np

def phase_space_average(Z: int, E_0: float, mubar: float, m_e: float, alpha: float, n_nodes: int) -> "np.ndarray":
    E_0 = float(E_0); m_e = float(m_e); n_nodes = int(n_nodes)
    if n_nodes < 1 or m_e <= 0.0 or E_0 <= m_e:
        raise ValueError("require n_nodes >= 1 and E_0 > m_e > 0")
    p_max = math.sqrt((E_0 - m_e) * (E_0 + m_e))
    x, w = np.polynomial.legendre.leggauss(n_nodes)
    norm = 0.0; s1 = 0.0; s2 = 0.0
    for x_i, w_i in zip(x, w):
        p = 0.5 * p_max * (float(x_i) + 1.0)
        wt = 0.5 * p_max * float(w_i) * spectrum_weight(p, Z, E_0, mubar, m_e, alpha)
        E_e = math.sqrt(p * p + m_e * m_e)
        beta, _p, Ebar = lepton_kinematics(E_e, m_e, E_0)   # Step 1
        norm += wt
        s1 += wt * sirlin_g1(beta, Ebar, mubar, m_e)
        s2 += wt * g2_alpha2z(beta, mubar, m_e)
    return np.array([norm, s1 / norm, s2 / norm], dtype=float)

import math
import numpy as np

def averaged_outer_correction(Q_EC: float, Z: int, m_e: float, alpha: float, n_nodes: int) -> float:
    Q_EC = float(Q_EC); m_e = float(m_e); alpha = float(alpha); Z = int(Z)
    if m_e <= 0.0 or Q_EC <= 2.0 * m_e:
        raise ValueError("require Q_EC > 2 m_e > 0")
    E_0 = Q_EC - m_e
    mubar = 2.0 * E_0 * math.exp(-1.0)
    _norm, gbar1, gbar2 = phase_space_average(Z, E_0, mubar, m_e, alpha, n_nodes)
    return float(alpha * gbar1 + alpha * alpha * Z * gbar2)
SCICODE_GOLD_EOF
