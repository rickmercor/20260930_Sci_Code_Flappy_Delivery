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


def shear_strain_vector(e: "np.ndarray") -> "np.ndarray":
    """The shear-strain vector e_s = (e_yz, e_zx, e_xy) of a symmetric strain tensor."""
    e = np.asarray(e, dtype=np.float64)
    if e.shape != (3, 3):
        raise ValueError("strain tensor must be 3x3")
    if not np.all(np.isfinite(e)):
        raise ValueError("strain tensor must be finite")
    if not np.allclose(e, e.T, rtol=0.0, atol=1e-12):
        raise ValueError("strain tensor must be symmetric")
    return np.array([e[1, 2], e[2, 0], e[0, 1]], dtype=np.float64)

import numpy as np


def strained_band_gaps(delta_g: float, delta_soff: float, a: float, b: float, d_cv: float,
                               e: "np.ndarray", e_s: "np.ndarray") -> "np.ndarray":
    """The three strained gaps CB-LHB, CB-HHB, CB-SOB, as DECLARED in the prompt:
        D_LHB = D_g - (a + b/2) tr e - (3b/2) e_zz - d_cv |e_s|
        D_HHB = D_g - (a - b/2) tr e + (3b/2) e_zz + d_cv |e_s|
        D_SOB = D_g + D_soff - a tr e
    Returns array([D_LHB, D_HHB, D_SOB]).
    """
    if delta_g <= 0.0 or delta_soff <= 0.0:
        raise ValueError("unstrained gaps must be positive")
    if d_cv < 0.0:
        raise ValueError("d_cv must be nonnegative")
    e = np.asarray(e, dtype=np.float64)
    e_s = np.asarray(e_s, dtype=np.float64)
    if e.shape != (3, 3) or e_s.shape != (3,):
        raise ValueError("e must be 3x3 and e_s must have 3 components")
    tr = float(np.trace(e))
    ezz = float(e[2, 2])
    sn = float(np.linalg.norm(e_s))
    d_lhb = delta_g - (a + 0.5 * b) * tr - 1.5 * b * ezz - d_cv * sn
    d_hhb = delta_g - (a - 0.5 * b) * tr + 1.5 * b * ezz + d_cv * sn
    d_sob = delta_g + delta_soff - a * tr
    gaps = np.array([d_lhb, d_hhb, d_sob], dtype=np.float64)
    if np.any(gaps <= 0.0):
        raise ValueError("a strained gap closed; the effective model is not valid here")
    return gaps

import numpy as np


def strain_tensor_factor(e: "np.ndarray") -> "np.ndarray":
    """(A38): adj{1 - e} through FIRST order in the strain, O(e^2) discarded.

    adj{1-e} = [[1-e_yy-e_zz,  e_xy,        e_xz     ],
                [e_xy,         1-e_xx-e_zz, e_yz     ],
                [e_xz,         e_yz,        1-e_xx-e_yy]]
    The prompt declares the truncation; the source fixes which entries carry which strains.
    """
    e = np.asarray(e, dtype=np.float64)
    if e.shape != (3, 3):
        raise ValueError("strain tensor must be 3x3")
    if not np.all(np.isfinite(e)):
        raise ValueError("strain tensor must be finite")
    if not np.allclose(e, e.T, rtol=0.0, atol=1e-12):
        raise ValueError("strain tensor must be symmetric")
    exx, eyy, ezz = e[0, 0], e[1, 1], e[2, 2]
    exy, exz, eyz = e[0, 1], e[0, 2], e[1, 2]
    return np.array([[1.0 - eyy - ezz, exy, exz],
                     [exy, 1.0 - exx - ezz, eyz],
                     [exz, eyz, 1.0 - exx - eyy]], dtype=np.float64)

import numpy as np


def isotropic_g_tensor(P: float, adj: "np.ndarray", delta_lhb: float, delta_sob: float) -> "np.ndarray":
    """(A34) top: g_iso = g_e 1 - (2/3) (2 m_e P^2 / hbar^2) adj{1-e} (1/D_LHB - 1/D_SOB).

    With hbar^2/(2 m_e) = 0.0380998212 eV nm^2 as the unit, 2 m_e P^2 / hbar^2 = P^2 / 0.0380998212.
    """
    hb2_2me = 0.0380998212        # hbar^2 / (2 m_e)  [eV nm^2]   (given in the prompt)
    g_e = 2.0023193               # free-electron g-factor         (given in the prompt)
    if P <= 0.0:
        raise ValueError("P must be positive")
    if delta_lhb <= 0.0 or delta_sob <= 0.0:
        raise ValueError("gaps must be positive")
    adj = np.asarray(adj, dtype=np.float64)
    if adj.shape != (3, 3):
        raise ValueError("adj must be 3x3")
    kane = P * P / hb2_2me
    return g_e * np.eye(3) - (2.0 / 3.0) * kane * adj * (1.0 / delta_lhb - 1.0 / delta_sob)

import numpy as np


def anisotropic_g_tensor(P: float, adj: "np.ndarray", delta_lhb: float, delta_hhb: float) -> "np.ndarray":
    """(A34) bottom: g_ani = (2 m_e P^2 / hbar^2) adj{1-e} (1/D_LHB - 1/D_HHB).

    No 2/3 prefactor here, and the gap difference is LHB-HHB, so it vanishes when
    strain does not split the light and heavy holes.
    """
    hb2_2me = 0.0380998212        # hbar^2 / (2 m_e)  [eV nm^2]   (given in the prompt)
    if P <= 0.0:
        raise ValueError("P must be positive")
    if delta_lhb <= 0.0 or delta_hhb <= 0.0:
        raise ValueError("gaps must be positive")
    adj = np.asarray(adj, dtype=np.float64)
    if adj.shape != (3, 3):
        raise ValueError("adj must be 3x3")
    kane = P * P / hb2_2me
    return kane * adj * (1.0 / delta_lhb - 1.0 / delta_hhb)

import numpy as np


def effective_g_tensor(g_iso: "np.ndarray", g_ani: "np.ndarray") -> "np.ndarray":
    """(A28)-(A29): the two Zeeman terms are
           (mu_B/2) B . g_iso . sigma      and      (mu_B/2) B . g_ani . (0, 0, sigma_z)^T.
    Rewriting their sum as a single (mu_B/2) B . G . sigma gives
           G_jk = g_iso_jk + delta_kz g_ani_jz ,
    i.e. g_ani contributes ONLY the z-COLUMN of G (every B_j couples to sigma_z through
    g_ani_jz) and nothing else. G is in general not symmetric; that is allowed for a
    g-tensor, only G G^T is observable.
    """
    g_iso = np.asarray(g_iso, dtype=np.float64)
    g_ani = np.asarray(g_ani, dtype=np.float64)
    if g_iso.shape != (3, 3) or g_ani.shape != (3, 3):
        raise ValueError("both g tensors must be 3x3")
    g = g_iso.copy()
    g[:, 2] += g_ani[:, 2]
    return g

import numpy as np


def strain_g_anisotropy(delta_g: float, delta_soff: float, P: float, a: float, b: float, d_cv: float,
                                e: "np.ndarray") -> float:
    """ORCHESTRATOR. dg = G_zz - G_xx of the effective CB g-tensor under strain e.

    Calls every earlier step directly and uses each return value.
    """
    e = np.asarray(e, dtype=np.float64)
    if e.shape != (3, 3):
        raise ValueError("strain tensor must be 3x3")
    e_s = shear_strain_vector(e)
    gaps = strained_band_gaps(delta_g, delta_soff, a, b, d_cv, e, e_s)
    adj = strain_tensor_factor(e)
    g_iso = isotropic_g_tensor(P, adj, gaps[0], gaps[2])
    g_ani = anisotropic_g_tensor(P, adj, gaps[0], gaps[1])
    g = effective_g_tensor(g_iso, g_ani)
    return float(g[2, 2] - g[0, 0])
SCICODE_GOLD_EOF
