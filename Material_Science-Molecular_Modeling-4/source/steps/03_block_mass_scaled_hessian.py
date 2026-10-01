"""
Build the per-monomer mass-scaled Hessian blocks as exact analytic second derivatives of the bond, angle, and Lennard-Jones terms, assembled per block, symmetrized, and scaled by the inverse square-root masses on both sides. The grading tolerance is set so that finite-difference approximations of any order fail; only exact derivatives pass.

The block-diagonal truncation is the accelerated variant of the method: each block keeps the second derivatives of every interaction the monomer takes part in, including the intermonomer terms, with respect to that monomer's own nine coordinates, and only the cross-monomer blocks coupling different monomers' coordinates are dropped.

Returns
-------
float64 array (I, 9, 9): mass-scaled per-monomer Hessian blocks
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def block_mass_scaled_hessian(coords):
    """coords: float array (I, 3, 3).

    Returns float64 array (I, 9, 9): for each monomer the exact analytic
    Hessian of the total potential with respect to that monomer's own nine
    coordinates, symmetrized and scaled as M^-1/2 H M^-1/2. The blocks must
    be exact second derivatives; numerical differentiation of the gradient
    does not meet the grading tolerance."""
    return np.zeros((np.asarray(coords).shape[0], 9, 9))

# =============================================================================
# GOLD SOLUTION
# =============================================================================

"""Step 3: per-monomer mass-scaled Hessian blocks, exact analytic form."""

import numpy as np

_KB, _R0 = 800.0, 1.0
_KTH, _TH0 = 120.0, 1.9106332362490186
_EPS_B, _SIG_B = 1.2, 2.6
_EPS_A, _SIG_A = 0.05, 1.4
_MA, _MB = 1.0, 16.0


I3 = np.eye(3)


def _radial_hessian(d, fp, fpp):
    """Hessian of f(|d|) wrt the first endpoint: fpp*dh dh^T + (fp/r)(I - dh dh^T)."""
    r = np.linalg.norm(d)
    dh = d / r
    outer = np.outer(dh, dh)
    return fpp * outer + (fp / r) * (I3 - outer)


def _bond_hessian(A, B, kb, r0):
    """6x6 Hessian of kb/2 (|A-B| - r0)^2 over (A, B)."""
    d = A - B
    r = np.linalg.norm(d)
    Haa = _radial_hessian(d, kb * (r - r0), kb)
    H = np.zeros((6, 6))
    H[0:3, 0:3] = Haa
    H[3:6, 3:6] = Haa
    H[0:3, 3:6] = -Haa
    H[3:6, 0:3] = -Haa
    return H


def _lj_hessian_self(a, b, eps, sig):
    """3x3 Hessian of the LJ pair energy wrt the first atom only."""
    d = a - b
    r = np.linalg.norm(d)
    s6 = (sig / r) ** 6
    fp = 4.0 * eps * (-12.0 * s6 * s6 + 6.0 * s6) / r
    fpp = 4.0 * eps * (156.0 * s6 * s6 - 42.0 * s6) / (r * r)
    return _radial_hessian(d, fp, fpp)


def _angle_hessian(A1, B, A2, kth, th0):
    """9x9 Hessian of kth/2 (theta - th0)^2 over (A1, B, A2).

    Exact vector calculus of theta = acos(uh . wh): cosine gradients wrt the
    two arm tips, their derivative blocks, then the B rows and columns from
    translational invariance (the energy depends on A1-B and A2-B only)."""
    u = A1 - B
    w = A2 - B
    nu = np.linalg.norm(u)
    nw = np.linalg.norm(w)
    uh = u / nu
    wh = w / nw
    c = float(np.dot(uh, wh))
    s = np.sqrt(1.0 - c * c)
    th = np.arccos(c)

    dcd1 = (wh - c * uh) / nu
    dcd2 = (uh - c * wh) / nw
    g1 = -dcd1 / s
    g2 = -dcd2 / s

    Pu = (I3 - np.outer(uh, uh)) / nu
    Pw = (I3 - np.outer(wh, wh)) / nw

    D11 = (-(np.outer(uh, dcd1)) - c * Pu) / nu - np.outer(wh - c * uh, uh) / (nu * nu)
    D12 = (Pw - np.outer(uh, dcd2)) / nu
    D22 = (-(np.outer(wh, dcd2)) - c * Pw) / nw - np.outer(uh - c * wh, wh) / (nw * nw)

    def theta_block(Ddc, dca, dcb):
        return -Ddc / s - (c / (s ** 3)) * np.outer(dca, dcb)

    T11 = theta_block(D11, dcd1, dcd1)
    T12 = theta_block(D12, dcd1, dcd2)
    T22 = theta_block(D22, dcd2, dcd2)

    dl = th - th0
    H11 = kth * (dl * T11 + np.outer(g1, g1))
    H12 = kth * (dl * T12 + np.outer(g1, g2))
    H22 = kth * (dl * T22 + np.outer(g2, g2))
    H21 = H12.T

    H = np.zeros((9, 9))
    H[0:3, 0:3] = H11
    H[0:3, 6:9] = H12
    H[6:9, 0:3] = H21
    H[6:9, 6:9] = H22
    H[0:3, 3:6] = -(H11 + H12)
    H[6:9, 3:6] = -(H21 + H22)
    H[3:6, 0:3] = H[0:3, 3:6].T
    H[3:6, 6:9] = H[6:9, 3:6].T
    H[3:6, 3:6] = H11 + H12 + H21 + H22
    return H


def _oracle_block_mass_scaled_hessian(coords):
    r = np.asarray(coords, dtype=np.float64)
    if r.ndim != 3 or r.shape[1:] != (3, 3) or not np.all(np.isfinite(r)):
        raise ValueError("invalid coordinates")
    I = r.shape[0]
    H = np.zeros((I, 9, 9))
    for i in range(I):
        A1, B, A2 = r[i]
        hb1 = _bond_hessian(A1, B, _KB, _R0)
        H[i, 0:3, 0:3] += hb1[0:3, 0:3]; H[i, 0:3, 3:6] += hb1[0:3, 3:6]
        H[i, 3:6, 0:3] += hb1[3:6, 0:3]; H[i, 3:6, 3:6] += hb1[3:6, 3:6]
        hb2 = _bond_hessian(A2, B, _KB, _R0)
        H[i, 6:9, 6:9] += hb2[0:3, 0:3]; H[i, 6:9, 3:6] += hb2[0:3, 3:6]
        H[i, 3:6, 6:9] += hb2[3:6, 0:3]; H[i, 3:6, 3:6] += hb2[3:6, 3:6]
        H[i] += _angle_hessian(A1, B, A2, _KTH, _TH0)
    # intermonomer LJ curvature enters each block through the diagonal
    # atom terms of the pair Hessians
    for i in range(I):
        for j in range(I):
            if i == j:
                continue
            H[i, 3:6, 3:6] += _lj_hessian_self(r[i, 1], r[j, 1], _EPS_B, _SIG_B)
            for ai, sl in ((0, 0), (2, 6)):
                for aj in (0, 2):
                    H[i, sl:sl+3, sl:sl+3] += _lj_hessian_self(r[i, ai], r[j, aj], _EPS_A, _SIG_A)
    m = np.repeat([_MA, _MB, _MA], 3) ** -0.5
    S = np.outer(m, m)
    for i in range(I):
        H[i] = ((H[i] + H[i].T) / 2.0) * S
    return H

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {"setup": 'coords = np.array([[[1.033663, 0.108724, 0.16054], [0.112899, 0.073417, 0.022995], [-0.362965, 0.98136, 0.026921]], [[-4.294078, 0.010016, 0.0408], [-3.18449, -0.107037, -0.054749], [-2.905678, 0.672657, -0.601129]]])', "call": "block_mass_scaled_hessian(coords)", "gold_call": "_oracle_block_mass_scaled_hessian(coords)", "tol": 1e-09},
        {"setup": 'coords = np.array([[[0.271613, 0.085543, 0.706603], [0.091614, -0.023224, 0.292598], [-0.421447, 0.854928, -0.442858]], [[-2.334479, 0.617925, -0.334762], [-2.46615, -0.148027, 0.167377], [-1.249518, -0.608938, -0.400547]], [[0.536595, -1.348508, 0.009638], [0.008204, -2.137902, 0.017308], [0.136845, -2.611049, -1.123682]]])', "call": "block_mass_scaled_hessian(coords)", "gold_call": "_oracle_block_mass_scaled_hessian(coords)", "tol": 1e-09},
        {"setup": 'coords = np.array([[[1.15, 0.05, 0.1], [0.1, 0.05, 0.0], [-0.35, 1.05, 0.0]], [[-3.9, 0.0, 0.0], [-2.95, -0.1, 0.0], [-2.6, 0.75, -0.55]]])', "call": "block_mass_scaled_hessian(coords)", "gold_call": "_oracle_block_mass_scaled_hessian(coords)", "tol": 1e-09},
    ]
