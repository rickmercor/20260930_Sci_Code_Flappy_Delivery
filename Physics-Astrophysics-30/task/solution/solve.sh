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


def evaluate_field(points: np.ndarray, region_params: dict) -> np.ndarray:
    P = np.asarray(points, dtype=float)
    if P.ndim != 2 or P.shape[1] != 3:
        raise ValueError("points must have shape (N, 3)")
    if not np.all(np.isfinite(P)):
        raise ValueError("points must contain only finite values")

    rp = region_params
    for key in ("B0", "hB", "aB", "z_foot", "p0", "up", "wp",
                "q0", "uq", "wq"):
        if key not in rp:
            raise ValueError(f"region_params missing key {key!r}")

    B0 = float(rp["B0"])
    hB = float(rp["hB"])
    aB = float(rp["aB"])
    if (not np.isfinite(B0) or not np.isfinite(hB)
            or B0 <= 0.0 or hB <= 0.0):
        raise ValueError("B0 and hB must be positive finite numbers")

    u3 = P[:, 2] - float(rp["z_foot"])
    base = 1.0 + u3 / hB
    if not np.all(np.isfinite(base)) or np.any(base <= 0.0):
        raise ValueError(
            "points lie outside the field domain "
            "(1 + u3/hB must be finite and positive)"
        )

    Bz = B0 * base ** (-aB)
    # div B = dBperp_x/dx + dBperp_y/dy + dBz/dz = t + dBz/du3 = 0
    t = (aB * B0 / hB) * base ** (-aB - 1.0)
    p = float(rp["p0"]) * np.exp(
        -((u3 - float(rp["up"])) / float(rp["wp"])) ** 2
    )
    q = float(rp["q0"]) * np.exp(
        -((u3 - float(rp["uq"])) / float(rp["wq"])) ** 2
    )

    B = np.empty_like(P)
    B[:, 0] = (0.5 * t + p) * P[:, 0] + q * P[:, 1]
    B[:, 1] = q * P[:, 0] + (0.5 * t - p) * P[:, 1]
    B[:, 2] = Bz
    return B

import numpy as np

def trace_field_line(region_params: dict, seed: np.ndarray, ell_max: float,
                           n_points: int) -> np.ndarray:
    s = np.asarray(seed, dtype=float)
    if s.shape != (3,):
        raise ValueError("seed must have shape (3,)")
    if not np.isfinite(float(ell_max)) or float(ell_max) <= 0.0:
        raise ValueError("ell_max must be a positive finite number")
    n = int(n_points)
    if n < 2:
        raise ValueError("n_points must be >= 2")

    n_sub = 8

    def bhat(x):
        B = evaluate_field(x[None, :], region_params)[0]
        return B / np.linalg.norm(B)

    dl = float(ell_max) / (n - 1) / n_sub
    out = np.empty((n, 3))
    out[0] = s
    x = s.copy()
    for i in range(1, n):
        for _ in range(n_sub):
            k1 = bhat(x)
            k2 = bhat(x + 0.5 * dl * k1)
            k3 = bhat(x + 0.5 * dl * k2)
            k4 = bhat(x + dl * k3)
            x = x + (dl / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        out[i] = x
    return out

import numpy as np


def plasma_background(
    positions: np.ndarray,
    bg_params: dict
) -> np.ndarray:
    P = np.asarray(positions, dtype=float)
    if P.ndim != 2 or P.shape[1] != 3:
        raise ValueError("positions must have shape (N, 3)")
    if not np.all(np.isfinite(P)):
        raise ValueError("positions must contain only finite values")

    bg = bg_params
    for key in ("rho0", "r0", "alpha_rho", "u0", "uinf", "lu"):
        if key not in bg:
            raise ValueError(f"bg_params missing key {key!r}")

    rho0 = float(bg["rho0"])
    r0 = float(bg["r0"])
    lu = float(bg["lu"])
    if (not np.isfinite(rho0) or not np.isfinite(r0)
            or not np.isfinite(lu)
            or rho0 <= 0.0 or r0 <= 0.0 or lu <= 0.0):
        raise ValueError("rho0, r0 and lu must be positive finite numbers")

    r = np.linalg.norm(P, axis=1)
    if not np.all(np.isfinite(r)) or np.any(r <= 0.0):
        raise ValueError(
            "positions must have finite positive heliocentric distance"
        )

    rho = rho0 * (r / r0) ** (-float(bg["alpha_rho"]))
    x = np.maximum(r - r0, 0.0)
    u_kms = (
        float(bg["u0"])
        + (float(bg["uinf"]) - float(bg["u0"]))
        * (1.0 - np.exp(-x / lu))
    )

    out = np.empty((len(P), 2))
    out[:, 0] = rho
    out[:, 1] = u_kms * 1.0e5
    return out

import numpy as np


def alfven_speed_gradient(
    bmag: np.ndarray,
    rho: np.ndarray,
    ds_cm: float
) -> np.ndarray:
    b = np.asarray(bmag, dtype=float)
    d = np.asarray(rho, dtype=float)

    if b.ndim != 1 or d.ndim != 1 or b.shape != d.shape:
        raise ValueError("bmag and rho must be 1-D arrays of the same length")
    if b.size < 3:
        raise ValueError("need at least 3 stations")
    if (not np.all(np.isfinite(b))
            or not np.all(np.isfinite(d))
            or np.any(b <= 0.0)
            or np.any(d <= 0.0)):
        raise ValueError(
            "bmag and rho must contain only finite positive values"
        )
    if not np.isfinite(float(ds_cm)) or float(ds_cm) <= 0.0:
        raise ValueError("ds_cm must be a positive finite number")

    va = b / np.sqrt(4.0 * np.pi * d)
    kva = np.gradient(np.log(va), float(ds_cm))

    out = np.empty((b.size, 2))
    out[:, 0] = va
    out[:, 1] = kva
    return out

import numpy as np

def deformation_rate(region_params: dict, positions: np.ndarray,
                           h_fd: float) -> np.ndarray:
    P = np.asarray(positions, dtype=float)
    if P.ndim != 2 or P.shape[1] != 3:
        raise ValueError("positions must have shape (N, 3)")
    h = float(h_fd)
    if not np.isfinite(h) or h <= 0.0:
        raise ValueError("h_fd must be a positive finite number")
    n = len(P)

    off = np.zeros((6, 3))
    for j in range(3):
        off[2 * j, j] = h
        off[2 * j + 1, j] = -h
    B = evaluate_field((P[:, None, :] + off[None, :, :]).reshape(-1, 3),
                             region_params).reshape(n, 6, 3)
    G = np.empty((n, 3, 3))
    for j in range(3):
        G[:, :, j] = (B[:, 2 * j] - B[:, 2 * j + 1]) / (2.0 * h)

    B0 = evaluate_field(P, region_params)
    bmag = np.linalg.norm(B0, axis=1)
    bh = B0 / bmag[:, None]
    proj = np.eye(3)[None] - bh[:, :, None] * bh[:, None, :]
    gb = np.einsum("nij,njk->nik", proj, G) / bmag[:, None, None]
    Gp = np.einsum("nij,njk->nik", gb, proj)

    # any orthonormal perpendicular pair; |S| does not depend on the choice
    e = np.tile(np.array([0.0, 0.0, 1.0]), (n, 1))
    flip = np.abs(np.einsum("ni,ni->n", e, bh)) > 0.9
    e[flip] = np.array([1.0, 0.0, 0.0])
    e1 = e - np.einsum("ni,ni->n", e, bh)[:, None] * bh
    e1 /= np.linalg.norm(e1, axis=1)[:, None]
    e2 = np.cross(bh, e1)
    E = np.stack([e1, e2], axis=1)
    M = np.einsum("nai,nij,nbj->nab", E, Gp, E)
    S = 0.5 * (M + M.transpose(0, 2, 1))
    S -= 0.5 * np.trace(M, axis1=1, axis2=2)[:, None, None] * np.eye(2)[None]
    return np.sqrt(np.einsum("nab,nab->n", S, S) / 2.0)

import numpy as np


def wave_amplitude(
    zp0_cms: float,
    va: np.ndarray,
    u: np.ndarray,
    damping: np.ndarray,
    s_cm: np.ndarray
) -> np.ndarray:
    z0 = float(zp0_cms)
    if not np.isfinite(z0) or z0 <= 0.0:
        raise ValueError("zp0_cms must be a positive finite number")

    va_ = np.asarray(va, dtype=float)
    u_ = np.asarray(u, dtype=float)
    dmp = np.asarray(damping, dtype=float)
    s = np.asarray(s_cm, dtype=float)

    if not (va_.ndim == u_.ndim == dmp.ndim == s.ndim == 1):
        raise ValueError("va, u, damping and s_cm must be 1-D arrays")
    if not (va_.shape == u_.shape == dmp.shape == s.shape):
        raise ValueError("va, u, damping and s_cm must have the same length")
    if va_.size == 0:
        raise ValueError("va, u, damping and s_cm must not be empty")
    if (not np.all(np.isfinite(va_))
            or not np.all(np.isfinite(u_))
            or not np.all(np.isfinite(dmp))
            or not np.all(np.isfinite(s))):
        raise ValueError(
            "va, u, damping and s_cm must contain only finite values"
        )
    if np.any(va_ <= 0.0) or np.any(u_ <= 0.0):
        raise ValueError("va and u must be positive")
    if s[0] != 0.0 or np.any(np.diff(s) <= 0.0):
        raise ValueError("s_cm must start at 0 and increase strictly")

    ma = u_ / va_
    g = np.sqrt(ma) + 1.0 / np.sqrt(ma)
    integral = np.concatenate([
        [0.0],
        np.cumsum(0.5 * (dmp[1:] + dmp[:-1]) * np.diff(s)),
    ])

    return z0 * (g[0] / g) * np.exp(-integral)

import numpy as np

def heating_rate(
    rho: np.ndarray,
    zp: np.ndarray,
    u: np.ndarray,
    va: np.ndarray,
    eta: np.ndarray
) -> np.ndarray:
    r = np.asarray(rho, dtype=float)
    z = np.asarray(zp, dtype=float)
    uu = np.asarray(u, dtype=float)
    v = np.asarray(va, dtype=float)
    e = np.asarray(eta, dtype=float)

    if not (r.ndim == z.ndim == uu.ndim == v.ndim == e.ndim == 1):
        raise ValueError("all inputs must be 1-D arrays")
    if not (r.shape == z.shape == uu.shape == v.shape == e.shape):
        raise ValueError("all inputs must have the same length")
    if (not np.all(np.isfinite(r))
            or not np.all(np.isfinite(z))
            or not np.all(np.isfinite(uu))
            or not np.all(np.isfinite(v))
            or not np.all(np.isfinite(e))):
        raise ValueError("all inputs must contain only finite values")
    if np.any(r <= 0.0) or np.any(v <= 0.0):
        raise ValueError("rho and va must be positive")
    if np.any(z < 0.0) or np.any(uu < 0.0) or np.any(e < 0.0):
        raise ValueError("zp, u and eta must be non-negative")

    return 0.25 * r * z ** 2 * (uu + v) * e

import numpy as np

def coronal_heating_summary(config: dict) -> float:
    for key in (
        "regions", "bg_params", "zp0_kms", "ell_max",
        "n_points", "h_fd", "r_sun_cm"
    ):
        if key not in config:
            raise ValueError(f"config missing key {key!r}")

    regions = config["regions"]
    if not isinstance(regions, list) or len(regions) == 0:
        raise ValueError("config['regions'] must be a non-empty list")

    zp0 = float(config["zp0_kms"])
    if not np.isfinite(zp0) or zp0 <= 0.0:
        raise ValueError("zp0_kms must be a positive finite number")

    n_points = int(config["n_points"])
    if n_points < 3:
        raise ValueError("n_points must be >= 3")

    rsun = float(config["r_sun_cm"])
    if not np.isfinite(rsun) or rsun <= 0.0:
        raise ValueError("r_sun_cm must be a positive finite number")

    ell_max = float(config["ell_max"])
    h_fd = float(config["h_fd"])
    bg = config["bg_params"]

    q_end = []
    for rp in regions:
        # Validate the region through Step 01 before accessing z_foot.
        evaluate_field(np.empty((0, 3), dtype=float), rp)

        seed = np.array([0.0, 0.0, float(rp["z_foot"])])
        pos = trace_field_line(rp, seed, ell_max, n_points)
        s_cm = np.linspace(0.0, ell_max, n_points) * rsun

        back = plasma_background(pos, bg)
        rho, u_cms = back[:, 0], back[:, 1]

        bmag = np.linalg.norm(
            evaluate_field(pos, rp),
            axis=1
        )
        alf = alfven_speed_gradient(
            bmag,
            rho,
            s_cm[1] - s_cm[0]
        )
        va, kva = alf[:, 0], alf[:, 1]

        smag = deformation_rate(rp, pos, h_fd) / rsun
        eta = 0.5 * np.abs(kva) + smag

        zp = wave_amplitude(
            zp0 * 1.0e5,
            va,
            u_cms,
            eta,
            s_cm
        )
        q = heating_rate(rho, zp, u_cms, va, eta)
        q_end.append(float(q[-1]))

    return float(np.log10(max(q_end)))
SCICODE_GOLD_EOF
