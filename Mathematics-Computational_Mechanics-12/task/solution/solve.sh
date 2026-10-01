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



def _f64(a):
    np = __import__("numpy")
    return np.asarray(a, dtype=np.float64)

def _unit(v, what):
    np = __import__("numpy")
    v = _f64(v).ravel()
    if v.size != 3:
        raise ValueError("bad " + what)
    if not np.all(np.isfinite(v)):
        raise ValueError("non-finite " + what)
    nrm = float(np.linalg.norm(v))
    if nrm <= 0.0:
        raise ValueError("zero-length " + what)
    return v / nrm

def _mat3(a, what):
    np = __import__("numpy")
    a = _f64(a)
    if a.shape != (3, 3) or not np.all(np.isfinite(a)):
        raise ValueError("bad " + what)
    return a

def tdc_surface_frame(normal, t_tilde):
    """Tangential projector and the boundary triad."""
    np = __import__("numpy")
    n = _unit(normal, "normal")
    tt = _unit(t_tilde, "t_tilde")
    if abs(float(n @ tt)) > 1e-10:
        raise ValueError("boundary tangent must be orthogonal to normal")
    p = np.eye(3) - np.outer(n, n)          # Eq (2.3)
    m = np.cross(tt, n)                     # Eq (2.2), this order
    out = np.zeros((5, 3), dtype=np.float64)
    out[0:3, :] = p
    out[3, :] = m
    out[4, 0] = float(np.linalg.matrix_rank(p, tol=1e-12))
    out[4, 1] = float(np.max(np.abs(p @ p - p)))     # idempotent
    out[4, 2] = float(np.linalg.norm(p @ n))         # p n = 0
    return out

import numpy as np



def _f64(a):
    np = __import__("numpy")
    return np.asarray(a, dtype=np.float64)

def _unit(v, what):
    np = __import__("numpy")
    v = _f64(v).ravel()
    if v.size != 3:
        raise ValueError("bad " + what)
    if not np.all(np.isfinite(v)):
        raise ValueError("non-finite " + what)
    nrm = float(np.linalg.norm(v))
    if nrm <= 0.0:
        raise ValueError("zero-length " + what)
    return v / nrm

def _mat3(a, what):
    np = __import__("numpy")
    a = _f64(a)
    if a.shape != (3, 3) or not np.all(np.isfinite(a)):
        raise ValueError("bad " + what)
    return a

def tdc_fiber_direction(normal, grad_phi):
    """Fiber tangent from the level-set surface gradient."""
    np = __import__("numpy")
    n = _unit(normal, "normal")
    g = _f64(grad_phi).ravel()
    if g.size != 3 or not np.all(np.isfinite(g)):
        raise ValueError("bad grad_phi")
    p = np.eye(3) - np.outer(n, n)
    g_surf = p @ g                          # surface gradient is the tangential part
    t_star = np.cross(n, g_surf)            # Eq (2.6), n x gradGamma phi
    ns = float(np.linalg.norm(t_star))
    if ns <= 0.0:
        raise ValueError("degenerate fiber direction")
    t = t_star / ns                         # Eq (2.7)
    out = np.zeros((4, 3), dtype=np.float64)
    out[0, :] = g_surf
    out[1, :] = t_star
    out[2, :] = t
    out[3, 0] = ns
    out[3, 1] = float(np.dot(t, n))         # fiber is tangential: must vanish
    out[3, 2] = float(np.dot(t, g_surf))    # orthogonal to the surface gradient
    return out

import numpy as np



def _f64(a):
    np = __import__("numpy")
    return np.asarray(a, dtype=np.float64)

def _unit(v, what):
    np = __import__("numpy")
    v = _f64(v).ravel()
    if v.size != 3:
        raise ValueError("bad " + what)
    if not np.all(np.isfinite(v)):
        raise ValueError("non-finite " + what)
    nrm = float(np.linalg.norm(v))
    if nrm <= 0.0:
        raise ValueError("zero-length " + what)
    return v / nrm

def _mat3(a, what):
    np = __import__("numpy")
    a = _f64(a)
    if a.shape != (3, 3) or not np.all(np.isfinite(a)):
        raise ValueError("bad " + what)
    return a

def tdc_directional_gradient(grad_u, projector):
    """Directional surface gradient: the full gradient projected onto the tangent space."""
    np = __import__("numpy")
    if np.asarray(grad_u).shape != (3, 3):
        raise ValueError("gradient must have shape (3, 3)")
    G = _mat3(grad_u, "grad_u")
    P = _mat3(projector, "projector")
    return (G @ P).astype(np.float64)

import numpy as np



def _f64(a):
    np = __import__("numpy")
    return np.asarray(a, dtype=np.float64)

def _unit(v, what):
    np = __import__("numpy")
    v = _f64(v).ravel()
    if v.size != 3:
        raise ValueError("bad " + what)
    if not np.all(np.isfinite(v)):
        raise ValueError("non-finite " + what)
    nrm = float(np.linalg.norm(v))
    if nrm <= 0.0:
        raise ValueError("zero-length " + what)
    return v / nrm

def _mat3(a, what):
    np = __import__("numpy")
    a = _f64(a)
    if a.shape != (3, 3) or not np.all(np.isfinite(a)):
        raise ValueError("bad " + what)
    return a

def tdc_deformation_gradient(normal_X, normal_x, Grad_u, grad_u):
    """F from the UNDEFORMED projector, its inverse from the DEFORMED one."""
    np = __import__("numpy")
    if np.asarray(normal_X).size != 3:
        raise ValueError("normal_X must have three entries")
    N = _unit(normal_X, "normal_X")
    n = _unit(normal_x, "normal_x")
    P = np.eye(3) - np.outer(N, N)
    p = np.eye(3) - np.outer(n, n)
    GU = _mat3(Grad_u, "Grad_u")
    gu = _mat3(grad_u, "grad_u")
    F = P + tdc_directional_gradient(GU, P)          # Eq (3.2)
    Finv = p - tdc_directional_gradient(gu, p)       # Eq (3.3): formula, not matrix inversion
    out = np.zeros((12, 3), dtype=np.float64)
    out[0:3, :] = F
    out[3:6, :] = Finv
    out[6:9, :] = F.T                                 # Eq (3.4)
    out[9:12, :] = Finv.T                             # Eq (3.5)
    return out

import numpy as np



def _f64(a):
    np = __import__("numpy")
    return np.asarray(a, dtype=np.float64)

def _unit(v, what):
    np = __import__("numpy")
    v = _f64(v).ravel()
    if v.size != 3:
        raise ValueError("bad " + what)
    if not np.all(np.isfinite(v)):
        raise ValueError("non-finite " + what)
    nrm = float(np.linalg.norm(v))
    if nrm <= 0.0:
        raise ValueError("zero-length " + what)
    return v / nrm

def _mat3(a, what):
    np = __import__("numpy")
    a = _f64(a)
    if a.shape != (3, 3) or not np.all(np.isfinite(a)):
        raise ValueError("bad " + what)
    return a

def tdc_deformation_properties(F, Finv, P, p):
    """The four rank-two identities of the surface deformation gradient."""
    np = __import__("numpy")
    if np.asarray(F).shape != (3, 3):
        raise ValueError("F must have shape (3, 3)")
    F = _mat3(F, "F"); Fi = _mat3(Finv, "Finv")
    P = _mat3(P, "P"); pp = _mat3(p, "p")
    out = np.zeros(8, dtype=np.float64)
    out[0] = float(np.max(np.abs(Fi @ F - P)))        # (i)   Finv F = P
    out[1] = float(np.max(np.abs(F @ Fi - pp)))       # (ii)  F Finv = p
    out[2] = float(np.max(np.abs(pp @ F @ P - F)))    # (iii) F = p F P
    out[3] = float(np.max(np.abs(P @ Fi @ pp - Fi)))  # (iv)  Finv = P Finv p
    out[4] = float(np.linalg.matrix_rank(F, tol=1e-10))
    out[5] = float(np.linalg.matrix_rank(Fi, tol=1e-10))
    out[6] = float(np.abs(np.linalg.det(F)))          # singular: vanishes
    out[7] = float(np.max(np.abs(F @ Fi - np.eye(3))))  # NOT the identity
    return out

import numpy as np



def _f64(a):
    np = __import__("numpy")
    return np.asarray(a, dtype=np.float64)

def _unit(v, what):
    np = __import__("numpy")
    v = _f64(v).ravel()
    if v.size != 3:
        raise ValueError("bad " + what)
    if not np.all(np.isfinite(v)):
        raise ValueError("non-finite " + what)
    nrm = float(np.linalg.norm(v))
    if nrm <= 0.0:
        raise ValueError("zero-length " + what)
    return v / nrm

def _mat3(a, what):
    np = __import__("numpy")
    a = _f64(a)
    if a.shape != (3, 3) or not np.all(np.isfinite(a)):
        raise ValueError("bad " + what)
    return a

def tdc_strain_measures(F, thickness_T):
    """Right Cauchy-Green, principal stretches, incompressible third stretch, thickness."""
    np = __import__("numpy")
    F = _mat3(F, "F")
    T = float(thickness_T)
    if not np.isfinite(T) or T <= 0.0:
        raise ValueError("bad thickness")
    C = F.T @ F                                        # Eq (3.6)
    w, V = np.linalg.eigh(C)
    order = np.argsort(w)[::-1]                        # descending: two non-zero, one zero
    w = w[order]; V = V[:, order]
    l1 = float(np.sqrt(max(w[0], 0.0)))
    l2 = float(np.sqrt(max(w[1], 0.0)))
    if l1 <= 0.0 or l2 <= 0.0:
        raise ValueError("degenerate stretches")
    l3 = 1.0 / (l1 * l2)                               # incompressibility J = 1
    t_def = l3 * T                                     # deformed thickness
    Crec = (l1 ** 2) * np.outer(V[:, 0], V[:, 0]) + (l2 ** 2) * np.outer(V[:, 1], V[:, 1])
    out = np.zeros((7, 3), dtype=np.float64)
    out[0:3, :] = C
    out[3, :] = np.array([l1, l2, l3])
    out[4, :] = np.array([t_def, float(np.linalg.matrix_rank(C, tol=1e-10)), float(w[2])])
    eig_tol = 1e-12 * max(1.0, float(np.max(np.abs(w))))
    top = V[:, np.abs(w - w[0]) <= eig_tol]
    projector = top @ top.T
    for axis in range(3):
        direction = projector[:, axis]
        magnitude = float(np.linalg.norm(direction))
        if magnitude > 1e-12:
            direction = direction / magnitude
            break
    pivot = int(np.argmax(np.abs(direction)))
    if direction[pivot] < 0.0:
        direction = -direction
    out[5, :] = direction
    out[6, :] = np.array([float(np.max(np.abs(Crec - C))), l1 * l2 * l3, 0.0])
    return out

import numpy as np



def _f64(a):
    np = __import__("numpy")
    return np.asarray(a, dtype=np.float64)

def _unit(v, what):
    np = __import__("numpy")
    v = _f64(v).ravel()
    if v.size != 3:
        raise ValueError("bad " + what)
    if not np.all(np.isfinite(v)):
        raise ValueError("non-finite " + what)
    nrm = float(np.linalg.norm(v))
    if nrm <= 0.0:
        raise ValueError("zero-length " + what)
    return v / nrm

def _mat3(a, what):
    np = __import__("numpy")
    a = _f64(a)
    if a.shape != (3, 3) or not np.all(np.isfinite(a)):
        raise ValueError("bad " + what)
    return a

def tdc_audit(normal_X, normal_x, t_tilde, grad_phi, Grad_u, grad_u, thickness_T):
    """Integrates the six earlier steps."""
    np = __import__("numpy")
    if not np.isfinite(thickness_T) or thickness_T <= 0.0:
        raise ValueError("bad thickness")
    frame = tdc_surface_frame(normal_X, t_tilde)
    fib = tdc_fiber_direction(normal_X, grad_phi)
    dg = tdc_deformation_gradient(normal_X, normal_x, Grad_u, grad_u)
    F = dg[0:3, :]; Finv = dg[3:6, :]
    N = _unit(normal_X, "normal_X"); n = _unit(normal_x, "normal_x")
    P = frame[0:3, :]
    p = np.eye(3) - np.outer(n, n)
    props = tdc_deformation_properties(F, Finv, P, p)
    strain = tdc_strain_measures(F, thickness_T)
    dirG = tdc_directional_gradient(_mat3(Grad_u, "Grad_u"), P)   # explicit use of step 3
    out = np.zeros((6, 3), dtype=np.float64)
    out[0, :] = fib[2, :]                    # unit fiber tangent
    out[1, :] = strain[3, :]                 # principal stretches
    out[2, :] = props[0:3]                   # identities (i)-(iii) residuals
    out[3, :] = np.array([props[3], props[6], props[7]])
    out[4, :] = np.array([strain[4, 0], frame[4, 0],
                           float(np.max(np.abs(dirG - (F - P))))])
    out[5, :] = np.array([float(np.max(np.abs(F - p @ F @ P))), strain[6, 0], strain[6, 1]])
    return out

import numpy as np



def _f64(a):
    np = __import__("numpy")
    return np.asarray(a, dtype=np.float64)

def _unit(v, what):
    np = __import__("numpy")
    v = _f64(v).ravel()
    if v.size != 3:
        raise ValueError("bad " + what)
    if not np.all(np.isfinite(v)):
        raise ValueError("non-finite " + what)
    nrm = float(np.linalg.norm(v))
    if nrm <= 0.0:
        raise ValueError("zero-length " + what)
    return v / nrm

def _mat3(a, what):
    np = __import__("numpy")
    a = _f64(a)
    if a.shape != (3, 3) or not np.all(np.isfinite(a)):
        raise ValueError("bad " + what)
    return a

def _ogden(mu, alpha, what):
    np = __import__("numpy")
    mu = _f64(mu).ravel()
    alpha = _f64(alpha).ravel()
    if mu.size != alpha.size or mu.size == 0:
        raise ValueError("bad " + what)
    if not (np.all(np.isfinite(mu)) and np.all(np.isfinite(alpha))):
        raise ValueError("non-finite " + what)
    if np.any(alpha == 0.0):
        raise ValueError("zero exponent in " + what)
    return mu, alpha


def tdc_fiber_kinematics(normal_X, grad_phi, Grad_u):
    """Fiber projector, fiber deformation gradient and the fiber principal stretches."""
    np = __import__("numpy")
    N = _unit(normal_X, "normal_X")
    GU = _mat3(Grad_u, "Grad_u")
    T = tdc_fiber_direction(N, grad_phi)[2, :]      # unit fiber tangent
    PY = np.outer(T, T)                                     # fiber projector, rank one
    FY = PY + tdc_directional_gradient(GU, PY)      # Eq (4.1)
    CY = FY.T @ FY                                          # Eq (4.5), rank one
    w = np.sort(np.linalg.eigvalsh(CY))[::-1]
    if w[0] <= 0.0:
        raise ValueError("degenerate fiber stretch")
    l1 = float(np.sqrt(w[0]))                               # Eq (4.6) fiber stretch
    l23 = 1.0 / np.sqrt(l1)                                 # incompressible fibers, source rule
    out = np.zeros((6, 3), dtype=np.float64)
    out[0:3, :] = CY
    out[3, :] = np.array([l1, l23, l23])
    out[4, :] = np.array([float(np.linalg.matrix_rank(CY, tol=1e-12)), float(w[1]), float(w[2])])
    out[5, :] = T
    return out

import numpy as np



def _f64(a):
    np = __import__("numpy")
    return np.asarray(a, dtype=np.float64)

def _unit(v, what):
    np = __import__("numpy")
    v = _f64(v).ravel()
    if v.size != 3:
        raise ValueError("bad " + what)
    if not np.all(np.isfinite(v)):
        raise ValueError("non-finite " + what)
    nrm = float(np.linalg.norm(v))
    if nrm <= 0.0:
        raise ValueError("zero-length " + what)
    return v / nrm

def _mat3(a, what):
    np = __import__("numpy")
    a = _f64(a)
    if a.shape != (3, 3) or not np.all(np.isfinite(a)):
        raise ValueError("bad " + what)
    return a

def _ogden(mu, alpha, what):
    np = __import__("numpy")
    mu = _f64(mu).ravel()
    alpha = _f64(alpha).ravel()
    if mu.size != alpha.size or mu.size == 0:
        raise ValueError("bad " + what)
    if not (np.all(np.isfinite(mu)) and np.all(np.isfinite(alpha))):
        raise ValueError("non-finite " + what)
    if np.any(alpha == 0.0):
        raise ValueError("zero exponent in " + what)
    return mu, alpha


def tdc_strain_energies(lam_m, lam_f, mu_m, alpha_m, mu_f, alpha_f):
    """Membrane and fiber stored energy densities."""
    np = __import__("numpy")
    lm = _f64(lam_m).ravel()
    if lm.size != 2 or not np.all(np.isfinite(lm)) or np.any(lm <= 0.0):
        raise ValueError("bad membrane stretches")
    lf = float(lam_f)
    if not np.isfinite(lf) or lf <= 0.0:
        raise ValueError("bad fiber stretch")
    mm, am = _ogden(mu_m, alpha_m, "membrane law")
    mf, af = _ogden(mu_f, alpha_f, "fiber law")
    l1, l2 = float(lm[0]), float(lm[1])
    Wm = float(np.sum(mm / am * (l1 ** am + l2 ** am + (l1 * l2) ** (-am) - 3.0)))   # Eq (3.13)
    Wf = float(np.sum(mf / af * (lf ** af + 2.0 * lf ** (-af / 2.0) - 3.0)))          # Eq (4.12)
    dWm1 = float(np.sum(mm * (l1 ** (am - 1.0) - l1 ** (-am - 1.0) * l2 ** (-am))))
    dWf = float(np.sum(mf * (lf ** (af - 1.0) - lf ** (-af / 2.0 - 1.0))))            # Eq (4.13)
    out = np.zeros((2, 3), dtype=np.float64)
    out[0, :] = np.array([Wm, Wf, Wm + Wf])
    out[1, :] = np.array([dWm1, dWf, 0.0])
    return out

import numpy as np



def _f64(a):
    np = __import__("numpy")
    return np.asarray(a, dtype=np.float64)

def _unit(v, what):
    np = __import__("numpy")
    v = _f64(v).ravel()
    if v.size != 3:
        raise ValueError("bad " + what)
    if not np.all(np.isfinite(v)):
        raise ValueError("non-finite " + what)
    nrm = float(np.linalg.norm(v))
    if nrm <= 0.0:
        raise ValueError("zero-length " + what)
    return v / nrm

def _mat3(a, what):
    np = __import__("numpy")
    a = _f64(a)
    if a.shape != (3, 3) or not np.all(np.isfinite(a)):
        raise ValueError("bad " + what)
    return a

def _ogden(mu, alpha, what):
    np = __import__("numpy")
    mu = _f64(mu).ravel()
    alpha = _f64(alpha).ravel()
    if mu.size != alpha.size or mu.size == 0:
        raise ValueError("bad " + what)
    if not (np.all(np.isfinite(mu)) and np.all(np.isfinite(alpha))):
        raise ValueError("non-finite " + what)
    if np.any(alpha == 0.0):
        raise ValueError("zero exponent in " + what)
    return mu, alpha


def tdc_coupled_energy(normal_X, normal_x, t_tilde, grad_phi, Grad_u, grad_u,
                               thickness_T, mu_m, alpha_m, mu_f, alpha_f):
    """Total stored energy density of the coupled fiber-membrane model."""
    np = __import__("numpy")
    T = float(thickness_T)
    if not np.isfinite(T) or T <= 0.0:
        raise ValueError("bad thickness")
    N = _unit(normal_X, "normal_X")
    audit = tdc_audit(N, normal_x, t_tilde, grad_phi, Grad_u, grad_u, T)
    l1, l2 = float(audit[1, 0]), float(audit[1, 1])
    fk = tdc_fiber_kinematics(N, grad_phi, Grad_u)
    lf = float(fk[3, 0])
    en = tdc_strain_energies(np.array([l1, l2]), lf, mu_m, alpha_m, mu_f, alpha_f)
    Wm, Wf = float(en[0, 0]), float(en[0, 1])
    t_star_norm = float(tdc_fiber_direction(N, grad_phi)[3, 0])   # |T*|, Eq (2.6)
    A_tilde = T                                              # Sec 5: A~ = B T with B = 1
    Wtot = T * Wm + A_tilde * Wf * t_star_norm               # Eq (7.4)
    out = np.zeros((10, 3), dtype=np.float64)
    out[0, :] = np.array([l1, l2, lf])
    out[1, :] = np.array([Wm, Wf, Wtot])
    out[2, :] = fk[3, :]
    out[3, :] = np.array([T * Wm, A_tilde * Wf * t_star_norm, t_star_norm])
    out[4:10, :] = audit
    return out
SCICODE_GOLD_EOF
