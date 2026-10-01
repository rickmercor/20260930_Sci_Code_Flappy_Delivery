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


# ---------------------------------------------------------------- helpers
def _lattice():
    """D2Q9 velocities, the orthogonal moment transform, its inverse, and the
    isotropy weights of the nearest-neighbour interaction stencil."""
    E = np.array([[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1],
                  [1, 1], [-1, 1], [-1, -1], [1, -1]])
    M = np.array([[1, 1, 1, 1, 1, 1, 1, 1, 1],
                  [-4, -1, -1, -1, -1, 2, 2, 2, 2],
                  [4, -2, -2, -2, -2, 1, 1, 1, 1],
                  [0, 1, 0, -1, 0, 1, -1, -1, 1],
                  [0, -2, 0, 2, 0, 1, -1, -1, 1],
                  [0, 0, 1, 0, -1, 1, 1, -1, -1],
                  [0, 0, -2, 0, 2, 1, 1, -1, -1],
                  [0, 1, -1, 1, -1, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 1, -1, 1, -1]], dtype=float)
    WI = np.array([0.0, 1/3, 1/3, 1/3, 1/3, 1/12, 1/12, 1/12, 1/12])
    return E, M, np.linalg.inv(M), WI


def _phys():
    """Lattice sound speed squared, reference speed, time step, interaction strength."""
    return 1.0/3.0, 1.0, 1.0, -1.0


def _fluid():
    """van der Waals constants, the equation-of-state scaling factor, and the
    critical temperature and density that follow from them."""
    A, B, R, KEOS = 9.0/49.0, 2.0/21.0, 1.0, 1.0/16.0
    return A, B, R, KEOS, 8.0*A/(27.0*R*B), 1.0/(3.0*B)


def _relaxation(tau):
    """The MRT relaxation vector. s_0 = s_j = 1, s_e = s_eps = s_p = 1/tau, and
    s_q from (1/s_q - 1/2)(1/s_p - 1/2) = 1/12, which gives s_q = 9/7 at tau = 0.8."""
    sp = 1.0/tau
    sq = 1.0/((1.0/12.0)/(1.0/sp - 0.5) + 0.5)
    return np.array([1.0, sp, sp, 1.0, sq, 1.0, sq, sp, sp])[:, None]




def vdw_pressure(rho: np.ndarray, Tr: float) -> np.ndarray:
    A, B, R, KEOS, TCR, _ = _fluid()
    rho = np.asarray(rho, dtype=float)
    if np.any(rho <= 0.0):
        raise ValueError("density must be strictly positive")
    if np.any(B*rho >= 1.0):
        raise ValueError("b*rho must be < 1 for the van der Waals EOS")
    if not np.isfinite(Tr) or Tr <= 0.0:
        raise ValueError("reduced temperature must be positive and finite")
    return KEOS*(rho*R*(Tr*TCR)/(1.0 - B*rho) - A*rho**2)

import numpy as np


# ---------------------------------------------------------------- helpers
def _lattice():
    """D2Q9 velocities, the orthogonal moment transform, its inverse, and the
    isotropy weights of the nearest-neighbour interaction stencil."""
    E = np.array([[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1],
                  [1, 1], [-1, 1], [-1, -1], [1, -1]])
    M = np.array([[1, 1, 1, 1, 1, 1, 1, 1, 1],
                  [-4, -1, -1, -1, -1, 2, 2, 2, 2],
                  [4, -2, -2, -2, -2, 1, 1, 1, 1],
                  [0, 1, 0, -1, 0, 1, -1, -1, 1],
                  [0, -2, 0, 2, 0, 1, -1, -1, 1],
                  [0, 0, 1, 0, -1, 1, 1, -1, -1],
                  [0, 0, -2, 0, 2, 1, 1, -1, -1],
                  [0, 1, -1, 1, -1, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 1, -1, 1, -1]], dtype=float)
    WI = np.array([0.0, 1/3, 1/3, 1/3, 1/3, 1/12, 1/12, 1/12, 1/12])
    return E, M, np.linalg.inv(M), WI


def _phys():
    """Lattice sound speed squared, reference speed, time step, interaction strength."""
    return 1.0/3.0, 1.0, 1.0, -1.0


def _fluid():
    """van der Waals constants, the equation-of-state scaling factor, and the
    critical temperature and density that follow from them."""
    A, B, R, KEOS = 9.0/49.0, 2.0/21.0, 1.0, 1.0/16.0
    return A, B, R, KEOS, 8.0*A/(27.0*R*B), 1.0/(3.0*B)


def _relaxation(tau):
    """The MRT relaxation vector. s_0 = s_j = 1, s_e = s_eps = s_p = 1/tau, and
    s_q from (1/s_q - 1/2)(1/s_p - 1/2) = 1/12, which gives s_q = 9/7 at tau = 0.8."""
    sp = 1.0/tau
    sq = 1.0/((1.0/12.0)/(1.0/sp - 0.5) + 0.5)
    return np.array([1.0, sp, sp, 1.0, sq, 1.0, sq, sp, sp])[:, None]




def cohesion_field(rho: np.ndarray, Tr: float) -> np.ndarray:
    CS2, _, DT, G = _phys()
    rho = np.asarray(rho, dtype=float)
    arg = 2.0*(vdw_pressure(rho, Tr) - rho*CS2)/(G*DT**2)
    if np.any(arg < 0.0):
        raise ValueError("the squared cohesive field is negative: density out of range")
    return np.sqrt(arg)

import numpy as np


# ---------------------------------------------------------------- helpers
def _lattice():
    """D2Q9 velocities, the orthogonal moment transform, its inverse, and the
    isotropy weights of the nearest-neighbour interaction stencil."""
    E = np.array([[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1],
                  [1, 1], [-1, 1], [-1, -1], [1, -1]])
    M = np.array([[1, 1, 1, 1, 1, 1, 1, 1, 1],
                  [-4, -1, -1, -1, -1, 2, 2, 2, 2],
                  [4, -2, -2, -2, -2, 1, 1, 1, 1],
                  [0, 1, 0, -1, 0, 1, -1, -1, 1],
                  [0, -2, 0, 2, 0, 1, -1, -1, 1],
                  [0, 0, 1, 0, -1, 1, 1, -1, -1],
                  [0, 0, -2, 0, 2, 1, 1, -1, -1],
                  [0, 1, -1, 1, -1, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 1, -1, 1, -1]], dtype=float)
    WI = np.array([0.0, 1/3, 1/3, 1/3, 1/3, 1/12, 1/12, 1/12, 1/12])
    return E, M, np.linalg.inv(M), WI


def _phys():
    """Lattice sound speed squared, reference speed, time step, interaction strength."""
    return 1.0/3.0, 1.0, 1.0, -1.0


def _fluid():
    """van der Waals constants, the equation-of-state scaling factor, and the
    critical temperature and density that follow from them."""
    A, B, R, KEOS = 9.0/49.0, 2.0/21.0, 1.0, 1.0/16.0
    return A, B, R, KEOS, 8.0*A/(27.0*R*B), 1.0/(3.0*B)


def _relaxation(tau):
    """The MRT relaxation vector. s_0 = s_j = 1, s_e = s_eps = s_p = 1/tau, and
    s_q from (1/s_q - 1/2)(1/s_p - 1/2) = 1/12, which gives s_q = 9/7 at tau = 0.8."""
    sp = 1.0/tau
    sq = 1.0/((1.0/12.0)/(1.0/sp - 0.5) + 0.5)
    return np.array([1.0, sp, sp, 1.0, sq, 1.0, sq, sp, sp])[:, None]




def pairwise_force(psi: np.ndarray) -> np.ndarray:
    E, _, _, WI = _lattice()
    _, _, DT, G = _phys()
    psi = np.asarray(psi, dtype=float)
    if psi.ndim != 1 or psi.size < 3:
        raise ValueError("psi must be a 1-D array of length >= 3")
    if np.any(psi < 0.0):
        raise ValueError("the cohesive field must be non-negative")
    acc = np.zeros_like(psi)
    for i in range(1, 9):
        ey = E[i, 1]
        if ey != 0:
            acc += WI[i]*np.roll(psi, -ey)*ey
    return -G*psi*acc*DT

import numpy as np


# ---------------------------------------------------------------- helpers
def _lattice():
    """D2Q9 velocities, the orthogonal moment transform, its inverse, and the
    isotropy weights of the nearest-neighbour interaction stencil."""
    E = np.array([[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1],
                  [1, 1], [-1, 1], [-1, -1], [1, -1]])
    M = np.array([[1, 1, 1, 1, 1, 1, 1, 1, 1],
                  [-4, -1, -1, -1, -1, 2, 2, 2, 2],
                  [4, -2, -2, -2, -2, 1, 1, 1, 1],
                  [0, 1, 0, -1, 0, 1, -1, -1, 1],
                  [0, -2, 0, 2, 0, 1, -1, -1, 1],
                  [0, 0, 1, 0, -1, 1, 1, -1, -1],
                  [0, 0, -2, 0, 2, 1, 1, -1, -1],
                  [0, 1, -1, 1, -1, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 1, -1, 1, -1]], dtype=float)
    WI = np.array([0.0, 1/3, 1/3, 1/3, 1/3, 1/12, 1/12, 1/12, 1/12])
    return E, M, np.linalg.inv(M), WI


def _phys():
    """Lattice sound speed squared, reference speed, time step, interaction strength."""
    return 1.0/3.0, 1.0, 1.0, -1.0


def _fluid():
    """van der Waals constants, the equation-of-state scaling factor, and the
    critical temperature and density that follow from them."""
    A, B, R, KEOS = 9.0/49.0, 2.0/21.0, 1.0, 1.0/16.0
    return A, B, R, KEOS, 8.0*A/(27.0*R*B), 1.0/(3.0*B)


def _relaxation(tau):
    """The MRT relaxation vector. s_0 = s_j = 1, s_e = s_eps = s_p = 1/tau, and
    s_q from (1/s_q - 1/2)(1/s_p - 1/2) = 1/12, which gives s_q = 9/7 at tau = 0.8."""
    sp = 1.0/tau
    sq = 1.0/((1.0/12.0)/(1.0/sp - 0.5) + 0.5)
    return np.array([1.0, sp, sp, 1.0, sq, 1.0, sq, sp, sp])[:, None]




def equilibrium_moments(rho: np.ndarray, ux: np.ndarray,
                                uy: np.ndarray) -> np.ndarray:
    _, C, _, _ = _phys()
    rho = np.asarray(rho, dtype=float)
    ux = np.asarray(ux, dtype=float); uy = np.asarray(uy, dtype=float)
    if not (rho.shape == ux.shape == uy.shape):
        raise ValueError("rho, ux and uy must have the same shape")
    if np.any(rho <= 0.0):
        raise ValueError("density must be strictly positive")
    x, y = ux/C, uy/C
    u2 = x*x + y*y
    return np.stack([rho, -2*rho + 3*rho*u2, rho - 3*rho*u2,
                     rho*x, -rho*x, rho*y, -rho*y, rho*(x*x - y*y), rho*x*y])

import numpy as np


# ---------------------------------------------------------------- helpers
def _lattice():
    """D2Q9 velocities, the orthogonal moment transform, its inverse, and the
    isotropy weights of the nearest-neighbour interaction stencil."""
    E = np.array([[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1],
                  [1, 1], [-1, 1], [-1, -1], [1, -1]])
    M = np.array([[1, 1, 1, 1, 1, 1, 1, 1, 1],
                  [-4, -1, -1, -1, -1, 2, 2, 2, 2],
                  [4, -2, -2, -2, -2, 1, 1, 1, 1],
                  [0, 1, 0, -1, 0, 1, -1, -1, 1],
                  [0, -2, 0, 2, 0, 1, -1, -1, 1],
                  [0, 0, 1, 0, -1, 1, 1, -1, -1],
                  [0, 0, -2, 0, 2, 1, 1, -1, -1],
                  [0, 1, -1, 1, -1, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 1, -1, 1, -1]], dtype=float)
    WI = np.array([0.0, 1/3, 1/3, 1/3, 1/3, 1/12, 1/12, 1/12, 1/12])
    return E, M, np.linalg.inv(M), WI


def _phys():
    """Lattice sound speed squared, reference speed, time step, interaction strength."""
    return 1.0/3.0, 1.0, 1.0, -1.0


def _fluid():
    """van der Waals constants, the equation-of-state scaling factor, and the
    critical temperature and density that follow from them."""
    A, B, R, KEOS = 9.0/49.0, 2.0/21.0, 1.0, 1.0/16.0
    return A, B, R, KEOS, 8.0*A/(27.0*R*B), 1.0/(3.0*B)


def _relaxation(tau):
    """The MRT relaxation vector. s_0 = s_j = 1, s_e = s_eps = s_p = 1/tau, and
    s_q from (1/s_q - 1/2)(1/s_p - 1/2) = 1/12, which gives s_q = 9/7 at tau = 0.8."""
    sp = 1.0/tau
    sq = 1.0/((1.0/12.0)/(1.0/sp - 0.5) + 0.5)
    return np.array([1.0, sp, sp, 1.0, sq, 1.0, sq, sp, sp])[:, None]




def discrete_force_moments(Fx: np.ndarray, Fy: np.ndarray, ux: np.ndarray,
                                   uy: np.ndarray) -> np.ndarray:
    _, C, _, _ = _phys()
    Fx = np.asarray(Fx, dtype=float); Fy = np.asarray(Fy, dtype=float)
    ux = np.asarray(ux, dtype=float); uy = np.asarray(uy, dtype=float)
    if not (Fx.shape == Fy.shape == ux.shape == uy.shape):
        raise ValueError("all four inputs must have the same shape")
    fx, fy = Fx/C, Fy/C
    x, y = ux/C, uy/C
    Fu = fx*x + fy*y
    return np.stack([np.zeros_like(fx), 6*Fu, -6*Fu, fx, -fx, fy, -fy,
                     2*fx*x - 2*fy*y, fx*y + fy*x])

import numpy as np


# ---------------------------------------------------------------- helpers
def _lattice():
    """D2Q9 velocities, the orthogonal moment transform, its inverse, and the
    isotropy weights of the nearest-neighbour interaction stencil."""
    E = np.array([[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1],
                  [1, 1], [-1, 1], [-1, -1], [1, -1]])
    M = np.array([[1, 1, 1, 1, 1, 1, 1, 1, 1],
                  [-4, -1, -1, -1, -1, 2, 2, 2, 2],
                  [4, -2, -2, -2, -2, 1, 1, 1, 1],
                  [0, 1, 0, -1, 0, 1, -1, -1, 1],
                  [0, -2, 0, 2, 0, 1, -1, -1, 1],
                  [0, 0, 1, 0, -1, 1, 1, -1, -1],
                  [0, 0, -2, 0, 2, 1, 1, -1, -1],
                  [0, 1, -1, 1, -1, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 1, -1, 1, -1]], dtype=float)
    WI = np.array([0.0, 1/3, 1/3, 1/3, 1/3, 1/12, 1/12, 1/12, 1/12])
    return E, M, np.linalg.inv(M), WI


def _phys():
    """Lattice sound speed squared, reference speed, time step, interaction strength."""
    return 1.0/3.0, 1.0, 1.0, -1.0


def _fluid():
    """van der Waals constants, the equation-of-state scaling factor, and the
    critical temperature and density that follow from them."""
    A, B, R, KEOS = 9.0/49.0, 2.0/21.0, 1.0, 1.0/16.0
    return A, B, R, KEOS, 8.0*A/(27.0*R*B), 1.0/(3.0*B)


def _relaxation(tau):
    """The MRT relaxation vector. s_0 = s_j = 1, s_e = s_eps = s_p = 1/tau, and
    s_q from (1/s_q - 1/2)(1/s_p - 1/2) = 1/12, which gives s_q = 9/7 at tau = 0.8."""
    sp = 1.0/tau
    sq = 1.0/((1.0/12.0)/(1.0/sp - 0.5) + 0.5)
    return np.array([1.0, sp, sp, 1.0, sq, 1.0, sq, sp, sp])[:, None]




def improved_source_moments(Fix: np.ndarray, Fiy: np.ndarray, ux: np.ndarray,
                                    uy: np.ndarray, psi: np.ndarray,
                                    eps: float) -> np.ndarray:
    _, C, _, G = _phys()
    Fix = np.asarray(Fix, dtype=float); Fiy = np.asarray(Fiy, dtype=float)
    ux = np.asarray(ux, dtype=float); uy = np.asarray(uy, dtype=float)
    psi = np.asarray(psi, dtype=float)
    if not (Fix.shape == Fiy.shape == ux.shape == uy.shape == psi.shape):
        raise ValueError("all array inputs must have the same shape")
    if np.any(psi <= 0.0):
        raise ValueError("the pseudopotential must be strictly positive here")
    if not np.isfinite(eps):
        raise ValueError("eps must be finite")
    k1 = k2 = -eps/16.0
    den = G*psi**2*C**2
    Q1 = 3.0*(k1 + 2.0*k2)*(Fix**2 + Fiy**2)/den
    Q7 = k1*(Fix**2 - Fiy**2)/den
    Q8 = k1*(Fix*Fiy)/den
    a = (30.0*eps - 15.0)/16.0
    b = 3.0*eps/8.0
    Q4 = (a*Fix**2/den - b*Fiy**2/den)*(ux/C)
    Q6 = (a*Fiy**2/den - b*Fix**2/den)*(uy/C)
    Q2 = -Q1/2.0
    Z = np.zeros_like(Q1)
    return np.stack([Z, Q1, Q2, Z, Q4, Z, Q6, Q7, Q8])

import numpy as np


# ---------------------------------------------------------------- helpers
def _lattice():
    """D2Q9 velocities, the orthogonal moment transform, its inverse, and the
    isotropy weights of the nearest-neighbour interaction stencil."""
    E = np.array([[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1],
                  [1, 1], [-1, 1], [-1, -1], [1, -1]])
    M = np.array([[1, 1, 1, 1, 1, 1, 1, 1, 1],
                  [-4, -1, -1, -1, -1, 2, 2, 2, 2],
                  [4, -2, -2, -2, -2, 1, 1, 1, 1],
                  [0, 1, 0, -1, 0, 1, -1, -1, 1],
                  [0, -2, 0, 2, 0, 1, -1, -1, 1],
                  [0, 0, 1, 0, -1, 1, 1, -1, -1],
                  [0, 0, -2, 0, 2, 1, 1, -1, -1],
                  [0, 1, -1, 1, -1, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 1, -1, 1, -1]], dtype=float)
    WI = np.array([0.0, 1/3, 1/3, 1/3, 1/3, 1/12, 1/12, 1/12, 1/12])
    return E, M, np.linalg.inv(M), WI


def _phys():
    """Lattice sound speed squared, reference speed, time step, interaction strength."""
    return 1.0/3.0, 1.0, 1.0, -1.0


def _fluid():
    """van der Waals constants, the equation-of-state scaling factor, and the
    critical temperature and density that follow from them."""
    A, B, R, KEOS = 9.0/49.0, 2.0/21.0, 1.0, 1.0/16.0
    return A, B, R, KEOS, 8.0*A/(27.0*R*B), 1.0/(3.0*B)


def _relaxation(tau):
    """The MRT relaxation vector. s_0 = s_j = 1, s_e = s_eps = s_p = 1/tau, and
    s_q from (1/s_q - 1/2)(1/s_p - 1/2) = 1/12, which gives s_q = 9/7 at tau = 0.8."""
    sp = 1.0/tau
    sq = 1.0/((1.0/12.0)/(1.0/sp - 0.5) + 0.5)
    return np.array([1.0, sp, sp, 1.0, sq, 1.0, sq, sp, sp])[:, None]




def _original_source_moments(Fix, Fiy, psi, eps):
    """Huang and Wu's original third-order source term, Eq (11):
    Q_{m,4} = Q_{m,6} = 0 and Q_{m,2} = -Q_{m,1}."""
    _, C, _, G = _phys()
    Fix = np.asarray(Fix, dtype=float); Fiy = np.asarray(Fiy, dtype=float)
    psi = np.asarray(psi, dtype=float)
    if np.any(psi <= 0.0):
        raise ValueError("the pseudopotential must be strictly positive here")
    k1 = k2 = -eps/16.0
    den = G*psi**2*C**2
    Q1 = 3.0*(k1 + 2.0*k2)*(Fix**2 + Fiy**2)/den
    Q7 = k1*(Fix**2 - Fiy**2)/den
    Q8 = k1*(Fix*Fiy)/den
    Z = np.zeros_like(Q1)
    return np.stack([Z, Q1, -Q1, Z, Z, Z, Z, Q7, Q8])

def collide_stream(f: np.ndarray, Tr: float, eps: float, tau: float,
                           drive: float, improved: bool) -> np.ndarray:
    E, M, Minv, _ = _lattice()
    _, _, DT, _ = _phys()
    f = np.asarray(f, dtype=float)
    if f.ndim != 2 or f.shape[0] != 9:
        raise ValueError("f must have shape (9, N)")
    if tau <= 0.5:
        raise ValueError("tau must exceed 1/2 for a positive viscosity")
    N = f.shape[1]
    rho = np.maximum(f.sum(0), 1e-12)
    psi = cohesion_field(rho, Tr)
    Fiy = pairwise_force(psi)
    Fix = np.zeros(N)
    Fx, Fy = Fix + drive, Fiy
    ux = ((f*E[:, 0, None]).sum(0) + DT/2*Fx)/rho
    uy = ((f*E[:, 1, None]).sum(0) + DT/2*Fy)/rho
    ux[0] = ux[-1] = uy[0] = uy[-1] = 0.0
    S = _relaxation(tau)
    m = M @ f
    me = equilibrium_moments(rho, ux, uy)
    Fm = discrete_force_moments(Fx, Fy, ux, uy)
    if improved:
        Qm = improved_source_moments(Fix, Fiy, ux, uy, psi, eps)
    else:
        Qm = _original_source_moments(Fix, Fiy, psi, eps)
    fbar = Minv @ (m + DT*Fm - S*(m - me + DT/2*Fm) + S*Qm)
    out = np.empty_like(f)
    for i in range(9):
        out[i] = np.roll(fbar[i], E[i, 1])
    for i, o in ((2, 4), (5, 7), (6, 8)):
        out[i, 0] = fbar[o, 0]
    for i, o in ((4, 2), (7, 5), (8, 6)):
        out[i, -1] = fbar[o, -1]
    return out

import numpy as np


# ---------------------------------------------------------------- helpers
def _lattice():
    """D2Q9 velocities, the orthogonal moment transform, its inverse, and the
    isotropy weights of the nearest-neighbour interaction stencil."""
    E = np.array([[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1],
                  [1, 1], [-1, 1], [-1, -1], [1, -1]])
    M = np.array([[1, 1, 1, 1, 1, 1, 1, 1, 1],
                  [-4, -1, -1, -1, -1, 2, 2, 2, 2],
                  [4, -2, -2, -2, -2, 1, 1, 1, 1],
                  [0, 1, 0, -1, 0, 1, -1, -1, 1],
                  [0, -2, 0, 2, 0, 1, -1, -1, 1],
                  [0, 0, 1, 0, -1, 1, 1, -1, -1],
                  [0, 0, -2, 0, 2, 1, 1, -1, -1],
                  [0, 1, -1, 1, -1, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 1, -1, 1, -1]], dtype=float)
    WI = np.array([0.0, 1/3, 1/3, 1/3, 1/3, 1/12, 1/12, 1/12, 1/12])
    return E, M, np.linalg.inv(M), WI


def _phys():
    """Lattice sound speed squared, reference speed, time step, interaction strength."""
    return 1.0/3.0, 1.0, 1.0, -1.0


def _fluid():
    """van der Waals constants, the equation-of-state scaling factor, and the
    critical temperature and density that follow from them."""
    A, B, R, KEOS = 9.0/49.0, 2.0/21.0, 1.0, 1.0/16.0
    return A, B, R, KEOS, 8.0*A/(27.0*R*B), 1.0/(3.0*B)


def _relaxation(tau):
    """The MRT relaxation vector. s_0 = s_j = 1, s_e = s_eps = s_p = 1/tau, and
    s_q from (1/s_q - 1/2)(1/s_p - 1/2) = 1/12, which gives s_q = 9/7 at tau = 0.8."""
    sp = 1.0/tau
    sq = 1.0/((1.0/12.0)/(1.0/sp - 0.5) + 0.5)
    return np.array([1.0, sp, sp, 1.0, sq, 1.0, sq, sp, sp])[:, None]




def equilibrate_two_phase(N: int, Tr_target: float, eps: float, tau: float,
                                  n_stages: int, per_stage: int) -> np.ndarray:
    _, _, Minv, _ = _lattice()
    _, _, _, _, _, RHOCR = _fluid()
    if N < 16:
        raise ValueError("N must be at least 16")
    if n_stages < 1 or per_stage < 1:
        raise ValueError("n_stages and per_stage must be positive")
    y = np.arange(N, dtype=float)
    rho = RHOCR + 0.15*RHOCR*(np.tanh((y - N/4)/8.0) - np.tanh((y - 3*N/4)/8.0))
    f = Minv @ equilibrium_moments(rho, np.zeros(N), np.zeros(N))
    for Tr in ([Tr_target] if n_stages == 1
           else np.linspace(0.95, Tr_target, n_stages)):
        for _ in range(per_stage):
            f = collide_stream(f, Tr, eps, tau, 0.0, False)
    return f

import numpy as np


# ---------------------------------------------------------------- helpers
def _lattice():
    """D2Q9 velocities, the orthogonal moment transform, its inverse, and the
    isotropy weights of the nearest-neighbour interaction stencil."""
    E = np.array([[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1],
                  [1, 1], [-1, 1], [-1, -1], [1, -1]])
    M = np.array([[1, 1, 1, 1, 1, 1, 1, 1, 1],
                  [-4, -1, -1, -1, -1, 2, 2, 2, 2],
                  [4, -2, -2, -2, -2, 1, 1, 1, 1],
                  [0, 1, 0, -1, 0, 1, -1, -1, 1],
                  [0, -2, 0, 2, 0, 1, -1, -1, 1],
                  [0, 0, 1, 0, -1, 1, 1, -1, -1],
                  [0, 0, -2, 0, 2, 1, 1, -1, -1],
                  [0, 1, -1, 1, -1, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 1, -1, 1, -1]], dtype=float)
    WI = np.array([0.0, 1/3, 1/3, 1/3, 1/3, 1/12, 1/12, 1/12, 1/12])
    return E, M, np.linalg.inv(M), WI


def _phys():
    """Lattice sound speed squared, reference speed, time step, interaction strength."""
    return 1.0/3.0, 1.0, 1.0, -1.0


def _fluid():
    """van der Waals constants, the equation-of-state scaling factor, and the
    critical temperature and density that follow from them."""
    A, B, R, KEOS = 9.0/49.0, 2.0/21.0, 1.0, 1.0/16.0
    return A, B, R, KEOS, 8.0*A/(27.0*R*B), 1.0/(3.0*B)


def _relaxation(tau):
    """The MRT relaxation vector. s_0 = s_j = 1, s_e = s_eps = s_p = 1/tau, and
    s_q from (1/s_q - 1/2)(1/s_p - 1/2) = 1/12, which gives s_q = 9/7 at tau = 0.8."""
    sp = 1.0/tau
    sq = 1.0/((1.0/12.0)/(1.0/sp - 0.5) + 0.5)
    return np.array([1.0, sp, sp, 1.0, sq, 1.0, sq, sp, sp])[:, None]




def theory_velocity(rho: np.ndarray, tau: float, Fdri: float) -> np.ndarray:
    CS2, _, _, _ = _phys()
    rho = np.asarray(rho, dtype=float)
    if rho.ndim != 1 or rho.size < 3:
        raise ValueError("rho must be a 1-D array of length >= 3")
    if np.any(rho <= 0.0):
        raise ValueError("density must be strictly positive")
    if tau <= 0.5:
        raise ValueError("tau must exceed 1/2")
    N = rho.size
    mu = rho*CS2*(tau - 0.5)
    mh = 0.5*(mu[:-1] + mu[1:])
    A = np.zeros((N, N)); rhs = np.full(N, -float(Fdri))
    for j in range(1, N-1):
        A[j, j-1] = mh[j-1]; A[j, j+1] = mh[j]; A[j, j] = -(mh[j-1] + mh[j])
    A[0, 0] = A[-1, -1] = 1.0; rhs[0] = rhs[-1] = 0.0
    return np.linalg.solve(A, rhs)

import numpy as np


# ---------------------------------------------------------------- helpers
def _lattice():
    """D2Q9 velocities, the orthogonal moment transform, its inverse, and the
    isotropy weights of the nearest-neighbour interaction stencil."""
    E = np.array([[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1],
                  [1, 1], [-1, 1], [-1, -1], [1, -1]])
    M = np.array([[1, 1, 1, 1, 1, 1, 1, 1, 1],
                  [-4, -1, -1, -1, -1, 2, 2, 2, 2],
                  [4, -2, -2, -2, -2, 1, 1, 1, 1],
                  [0, 1, 0, -1, 0, 1, -1, -1, 1],
                  [0, -2, 0, 2, 0, 1, -1, -1, 1],
                  [0, 0, 1, 0, -1, 1, 1, -1, -1],
                  [0, 0, -2, 0, 2, 1, 1, -1, -1],
                  [0, 1, -1, 1, -1, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 1, -1, 1, -1]], dtype=float)
    WI = np.array([0.0, 1/3, 1/3, 1/3, 1/3, 1/12, 1/12, 1/12, 1/12])
    return E, M, np.linalg.inv(M), WI


def _phys():
    """Lattice sound speed squared, reference speed, time step, interaction strength."""
    return 1.0/3.0, 1.0, 1.0, -1.0


def _fluid():
    """van der Waals constants, the equation-of-state scaling factor, and the
    critical temperature and density that follow from them."""
    A, B, R, KEOS = 9.0/49.0, 2.0/21.0, 1.0, 1.0/16.0
    return A, B, R, KEOS, 8.0*A/(27.0*R*B), 1.0/(3.0*B)


def _relaxation(tau):
    """The MRT relaxation vector. s_0 = s_j = 1, s_e = s_eps = s_p = 1/tau, and
    s_q from (1/s_q - 1/2)(1/s_p - 1/2) = 1/12, which gives s_q = 9/7 at tau = 0.8."""
    sp = 1.0/tau
    sq = 1.0/((1.0/12.0)/(1.0/sp - 0.5) + 0.5)
    return np.array([1.0, sp, sp, 1.0, sq, 1.0, sq, sp, sp])[:, None]




def scheme_accuracy_audit(N: int, Tr: float, eps: float, tau: float,
                                  Fdri: float, n_stages: int, per_stage: int,
                                  drive_steps: int) -> float:
    E, _, _, _ = _lattice()
    CS2, _, DT, _ = _phys()
    if int(N) < 16:
        raise ValueError("N must be at least 16")
    if not (0.0 < Tr < 1.0):
        raise ValueError("the reduced temperature must lie strictly between 0 and 1")
    if tau <= 0.5:
        raise ValueError("tau must exceed 1/2 for a positive viscosity")
    if Fdri <= 0.0:
        raise ValueError("the driving force must be positive")
    if int(drive_steps) < 1 or int(n_stages) < 1 or int(per_stage) < 1:
        raise ValueError("step counts must be positive")

    f0 = equilibrate_two_phase(N, Tr, eps, tau, n_stages, per_stage)

    # Consistency gate over the settled interface: the pairwise force must balance
    # the gradient of the non-ideal part of the pressure. This exercises the early
    # steps on the state the rest of the audit is built from, and raises if the
    # chain is inconsistent rather than silently reporting a number.
    rho0 = f0.sum(0)
    p0 = vdw_pressure(rho0, Tr)
    psi0 = cohesion_field(rho0, Tr)
    Fi0 = pairwise_force(psi0)
    interior = slice(2, -2)
    resid = Fi0[interior] + np.gradient(p0 - rho0*CS2)[interior]
    scale = max(np.max(np.abs(Fi0[interior])), 1e-30)
    if np.max(np.abs(resid))/scale > 5e-2:
        raise ValueError("the settled interface does not satisfy the force balance")

    zero = np.zeros_like(rho0)
    me0 = equilibrium_moments(rho0, zero, zero)
    Fm0 = discrete_force_moments(zero, Fi0, zero, zero)
    Q0 = improved_source_moments(zero, Fi0, zero, zero, psi0, eps)
    if np.max(np.abs(Q0[4])) + np.max(np.abs(Q0[6])) > 1e-30:
        raise ValueError("the source term must vanish when the fluid is at rest")
    if abs(me0[0].sum() - rho0.sum()) > 1e-8*max(rho0.sum(), 1.0):
        raise ValueError("the equilibrium moments do not conserve mass")
    if np.max(np.abs(Fm0[3])) > 1e-30:
        raise ValueError("a wall-parallel force moment appeared with no driving force")

    out = {}
    for improved in (False, True):
        f = f0.copy()
        for _ in range(int(drive_steps)):
            f = collide_stream(f, Tr, eps, tau, Fdri, improved)
        rho = f.sum(0)
        ux = ((f*E[:, 0, None]).sum(0) + DT/2*Fdri)/rho
        ux[0] = ux[-1] = 0.0
        ut = theory_velocity(rho, tau, Fdri)
        out[improved] = float(np.linalg.norm(ux - ut)/np.linalg.norm(ut))
    return out[True]
SCICODE_GOLD_EOF
