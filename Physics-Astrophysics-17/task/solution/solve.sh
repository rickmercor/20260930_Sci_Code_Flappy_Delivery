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


def _kmb_scalar(name, x, strict=True):
    """Finite real scalar: strictly positive when strict, otherwise non-negative."""
    if isinstance(x, (bool, np.bool_)) or np.ndim(x) != 0:
        raise ValueError(f"{name} must be a real scalar")
    try:
        v = float(x)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a real number") from None
    if not math.isfinite(v) or v < 0.0 or (strict and v == 0.0):
        raise ValueError(f"{name} must be finite and {'strictly positive' if strict else 'non-negative'}")
    return v


def _kmb_si_constants():
    """SI constants (elementary charge in C, proton mass in kg, vacuum permeability in H/m)."""
    return 1.602176634e-19, 1.67262192369e-27, 4.0e-7 * math.pi


def plasma_reference_scales(B0_nT: float, n_p_cm3: float, v_thp_kms: float) -> "np.ndarray":
    b_tesla = _kmb_scalar("B0_nT", B0_nT) * 1.0e-9
    n_si = _kmb_scalar("n_p_cm3", n_p_cm3) * 1.0e6
    v_th = _kmb_scalar("v_thp_kms", v_thp_kms)
    charge, m_p, mu_0 = _kmb_si_constants()
    omega_p = charge * b_tesla / m_p
    v_a = b_tesla / math.sqrt(mu_0 * n_si * m_p) / 1.0e3
    return np.array([omega_p, v_th / omega_p, v_a])

import math

import numpy as np
from numpy.typing import ArrayLike


def _kmb_finite(name, x):
    """Finite real scalar of either sign."""
    if isinstance(x, (bool, np.bool_)) or np.ndim(x) != 0:
        raise ValueError(f"{name} must be a real scalar")
    try:
        v = float(x)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a real number") from None
    if not math.isfinite(v):
        raise ValueError(f"{name} must be finite")
    return v


def _kmb_array(name, x):
    """Non-empty real array (a copy) whose entries are finite and strictly positive."""
    try:
        a = np.array(x, dtype=float)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be real") from None
    if a.size == 0 or not np.all(np.isfinite(a)) or np.any(a <= 0.0):
        raise ValueError(f"{name} must be non-empty, finite and strictly positive")
    return a


def perpendicular_velocity_amplitude(
    k_perp: "ArrayLike",
    B0_nT: float,
    v_A_kms: float,
    dB_ref_nT: float,
    k_ref: float,
    k_break: float,
    slope_large: float,
    slope_small: float,
) -> "np.ndarray":
    k = _kmb_array("k_perp", k_perp)
    b0 = _kmb_scalar("B0_nT", B0_nT)
    v_a = _kmb_scalar("v_A_kms", v_A_kms)
    amp = _kmb_scalar("dB_ref_nT", dB_ref_nT)
    k0 = _kmb_scalar("k_ref", k_ref)
    kb = _kmb_scalar("k_break", k_break)
    s_large = _kmb_finite("slope_large", slope_large)
    s_small = _kmb_finite("slope_small", slope_small)
    delta_b_nT = amp * (k / k0) ** (-s_large) * (1.0 + (k / kb) ** 2) ** (-0.5 * (s_small - s_large))
    return delta_b_nT / b0 * v_a

import math

import numpy as np
from numpy.typing import ArrayLike
from scipy.special import i0e


def _kmb_one_minus_gamma0(x):
    """1 - I0(x) exp(-x) for x >= 0, with a series below x = 1e-3 to avoid cancellation."""
    x = np.asarray(x, dtype=float)
    series = x * (1.0 - x * (0.75 - x * (5.0 / 12.0 - x * (35.0 / 192.0 - x * 21.0 / 320.0))))
    return np.where(x < 1.0e-3, series, 1.0 - i0e(x))


def electron_flow_factor(k_rho_p: "ArrayLike", te_over_tp: float) -> "np.ndarray":
    kr = _kmb_array("k_rho_p", k_rho_p)
    ztau = _kmb_scalar("te_over_tp", te_over_tp, strict=False)
    return kr * np.sqrt(0.5 * (ztau + 1.0 / _kmb_one_minus_gamma0(0.5 * kr * kr)))

import math

import numpy as np
from numpy.typing import ArrayLike
from scipy.special import i0e


def electric_amplitude_parameter(
    k_rho_p: "ArrayLike",
    delta_b_kms: "ArrayLike",
    v_thp_kms: float,
    te_over_tp: float,
) -> "np.ndarray":
    kr = _kmb_array("k_rho_p", k_rho_p)
    db = _kmb_array("delta_b_kms", delta_b_kms)
    v_th = _kmb_scalar("v_thp_kms", v_thp_kms)
    try:
        kr, db = np.broadcast_arrays(kr, db)
    except ValueError:
        raise ValueError("k_rho_p and delta_b_kms must broadcast together") from None
    alpha = electron_flow_factor(kr, te_over_tp)
    half_k2 = 0.5 * kr * kr
    return half_k2 / _kmb_one_minus_gamma0(half_k2) * db / (alpha * v_th)

import math

import numpy as np
from numpy.typing import ArrayLike
from scipy.special import i0e


def fluctuation_frequency_ratio(
    k_perp: "ArrayLike",
    delta_b_kms: "ArrayLike",
    omega_p: float,
    rho_p_km: float,
    te_over_tp: float,
) -> "np.ndarray":
    k = _kmb_array("k_perp", k_perp)
    db = _kmb_array("delta_b_kms", delta_b_kms)
    om = _kmb_scalar("omega_p", omega_p)
    rho = _kmb_scalar("rho_p_km", rho_p_km)
    try:
        k, db = np.broadcast_arrays(k, db)
    except ValueError:
        raise ValueError("k_perp and delta_b_kms must broadcast together") from None
    alpha = electron_flow_factor(k * rho, te_over_tp)
    return alpha * k * db / om

import math

import numpy as np
from numpy.typing import ArrayLike
from scipy.special import j1


def gyroaveraging_weight(k_rho: "ArrayLike") -> "np.ndarray":
    x = _kmb_array("k_rho", k_rho)
    return (j1(x) / x) ** 2

import math

import numpy as np
from scipy.special import i0e, j1


def proton_perpendicular_heating_rate(
    k_perp: float = 0.280,
    B0_nT: float = 180.0,
    n_p_cm3: float = 90.0,
    v_thp_kms: float = 92.0,
    te_over_tp: float = 2.0,
    dB_ref_nT: float = 12.0,
    k_ref: float = 1.0e-3,
    k_break: float = 0.150,
    slope_large: float = 1.0 / 3.0,
    slope_small: float = 0.70,
    c1: float = 0.75,
    c2: float = 0.34,
) -> float:
    k = _kmb_scalar("k_perp", k_perp)
    c1v = _kmb_scalar("c1", c1)
    c2v = _kmb_scalar("c2", c2)
    omega_p, rho_p, v_a = plasma_reference_scales(B0_nT, n_p_cm3, v_thp_kms)
    kk = np.array([k])
    delta_b = perpendicular_velocity_amplitude(kk, B0_nT, v_a, dB_ref_nT, k_ref, k_break,
                                                        slope_large, slope_small)
    k_rho = kk * rho_p
    epsilon = float(electric_amplitude_parameter(k_rho, delta_b, v_thp_kms, te_over_tp)[0])
    ratio = float(fluctuation_frequency_ratio(kk, delta_b, omega_p, rho_p, te_over_tp)[0])
    weight = float(gyroaveraging_weight(k_rho)[0])
    v_th = float(v_thp_kms) * 1.0e3
    return float(4.0 * c1v * omega_p * v_th * v_th * weight * epsilon * epsilon * ratio
                 * math.exp(-c2v / ratio))
SCICODE_GOLD_EOF
