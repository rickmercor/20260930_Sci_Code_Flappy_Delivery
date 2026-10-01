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

def spin_boson_potential(position: float, parameters: "np.ndarray") -> "np.ndarray":
    position = float(position)
    p = np.asarray(parameters, dtype=np.float64)
    if not np.isfinite(position) or p.shape != (6,) or not np.all(np.isfinite(p)):
        raise ValueError("position and six parameters must be finite")
    mass, omega, alpha, kappa, delta, spin_radius = p
    if mass <= 0.0 or omega <= 0.0 or delta <= 0.0 or spin_radius <= 0.0:
        raise ValueError("mass, omega, delta, and spin_radius must be positive")
    v0 = 0.5 * mass * omega * omega * position * position
    v1 = alpha + kappa * position
    v2 = -v1
    return np.array([v0, v1, v2, delta,
                     mass*omega*omega*position, kappa, -kappa, 0.0,
                     mass*omega*omega, 0.0, 0.0, 0.0], dtype=np.float64)

import numpy as np

def spin_generator(position: float, parameters: "np.ndarray") -> "np.ndarray":
    vals = spin_boson_potential(position, parameters)
    _, v1, v2, delta, _, v1r, v2r, deltar, *_ = vals
    h = np.array([2.0*delta, 0.0, v1-v2], dtype=np.float64)
    hr = np.array([2.0*deltar, 0.0, v1r-v2r], dtype=np.float64)
    hx, hy, hz = h
    w = 1j*np.array([[0.0,-hz,hy],[hz,0.0,-hx],[-hy,hx,0.0]], dtype=np.complex128)
    hxr, hyr, hzr = hr
    wr = 1j*np.array([[0.0,-hzr,hyr],[hzr,0.0,-hxr],[-hyr,hxr,0.0]], dtype=np.complex128)
    return np.concatenate((w.ravel(), wr.ravel(), h.astype(np.complex128), hr.astype(np.complex128)))

import numpy as np

def electronic_operators(position: float, timestep: float, parameters: "np.ndarray") -> "np.ndarray":
    timestep = float(timestep)
    if not np.isfinite(timestep) or timestep == 0.0:
        raise ValueError("timestep must be finite and nonzero")
    data = spin_generator(position, parameters)
    w = data[:9].reshape(3,3)
    eigvals, sw = np.linalg.eigh(w)
    phase = np.exp(-1j*eigvals*timestep)
    ups = np.empty(3, dtype=np.complex128)
    for i, lam in enumerate(eigvals):
        if lam == 0.0:
            ups[i] = timestep
        else:
            ups[i] = 1j*np.expm1(-1j*lam*timestep)/lam
    q = (sw*phase) @ sw.conj().T
    b = (sw*ups) @ sw.conj().T
    q = np.real_if_close(q, tol=1000).astype(np.complex128)
    b = np.real_if_close(b, tol=1000).astype(np.complex128)
    return np.concatenate((eigvals.astype(np.complex128), q.ravel(), b.ravel()))

import numpy as np

def coupled_flow(position: float, momentum: float, spin: "np.ndarray", timestep: float, parameters: "np.ndarray") -> "np.ndarray":
    position, momentum = float(position), float(momentum)
    spin = np.asarray(spin, dtype=np.float64)
    if not np.isfinite(position) or not np.isfinite(momentum) or spin.shape != (3,) or not np.all(np.isfinite(spin)):
        raise ValueError("position, momentum, and three spin components must be finite")
    vals = spin_boson_potential(position, parameters)
    ops = electronic_operators(position, timestep, parameters)
    q = ops[3:12].reshape(3,3)
    b = ops[12:21].reshape(3,3)
    hr = spin_generator(position, parameters)[21:24].real
    pnew = momentum - timestep*(vals[4] + 0.5*(vals[5]+vals[6])) - 0.5*np.dot(hr, b @ spin).real
    unew = (q @ spin).real
    return np.concatenate(([pnew], unew)).astype(np.float64)

import numpy as np

def spin_mint_step(canonical_state: "np.ndarray", timestep: float, parameters: "np.ndarray") -> "np.ndarray":
    z = np.asarray(canonical_state, dtype=np.float64)
    p = np.asarray(parameters, dtype=np.float64)
    if z.shape != (4,) or not np.all(np.isfinite(z)):
        raise ValueError("canonical_state must contain four finite values")
    spin_boson_potential(z[0], p)
    dt = float(timestep)
    if not np.isfinite(dt) or dt == 0.0:
        raise ValueError("timestep must be finite and nonzero")
    mass, *_, rs = p
    r, phi, mom, s = z
    if abs(s) >= rs:
        raise ValueError("canonical polar momentum must lie strictly inside the spin sphere")
    rho = np.sqrt(rs*rs-s*s)
    u = 2.0*np.array([rho*np.cos(phi), rho*np.sin(phi), s])
    rmid = r + 0.5*dt*mom/mass
    flow = coupled_flow(rmid, mom, u, dt, p)
    pnew, unew = flow[0], flow[1:]
    rnew = rmid + 0.5*dt*pnew/mass
    phiraw = np.arctan2(unew[1], unew[0])
    phinew = phi + np.arctan2(np.sin(phiraw-phi), np.cos(phiraw-phi))
    return np.array([rnew, phinew, pnew, 0.5*unew[2]], dtype=np.float64)

import numpy as np
from scipy.linalg import expm, expm_frechet

def canonical_tangent_step(canonical_state: "np.ndarray", timestep: float, parameters: "np.ndarray") -> "np.ndarray":
    z = np.asarray(canonical_state, dtype=np.float64)
    p = np.asarray(parameters, dtype=np.float64)
    znew = spin_mint_step(z, timestep, p)
    dt = float(timestep)
    mass, omega, _, _, _, rs = p
    r, phi, mom, s = z
    rho = np.sqrt(rs*rs-s*s)
    u = 2.0*np.array([rho*np.cos(phi), rho*np.sin(phi), s])
    du_dphi = 2.0*np.array([-rho*np.sin(phi), rho*np.cos(phi), 0.0])
    du_ds = 2.0*np.array([-s*np.cos(phi)/rho, -s*np.sin(phi)/rho, 1.0])

    # First half drift: x=[R,P,u_x,u_y,u_z].
    d1 = np.eye(5)
    d1[0,1] = 0.5*dt/mass
    rmid = r + 0.5*dt*mom/mass

    gen = spin_generator(rmid, p)
    w = gen[:9].reshape(3,3)
    wr = gen[9:18].reshape(3,3)
    a = -1j*w
    ar = -1j*wr
    q, qr = expm_frechet(a*dt, ar*dt, compute_expm=True)
    # Augmented exponential gives B=integral exp(A t)dt; its Frechet derivative gives B_R.
    k = np.zeros((6,6), dtype=np.complex128)
    kr = np.zeros((6,6), dtype=np.complex128)
    k[:3,:3] = a
    k[:3,3:] = np.eye(3)
    kr[:3,:3] = ar
    ek, ekr = expm_frechet(k*dt, kr*dt, compute_expm=True)
    b = ek[:3,3:]
    br = ekr[:3,3:]
    vals = spin_boson_potential(rmid, p)
    hr = gen[21:24].real
    hrr = np.zeros(3)
    u2 = (q @ u).real
    force_u = -0.5*(hr @ b).real
    force_r = -dt*vals[8] - 0.5*np.dot(hrr, b@u).real - 0.5*np.dot(hr, br@u).real
    c = np.eye(5)
    c[1,0] = force_r
    c[1,2:] = force_u
    c[2:,0] = (qr@u).real
    c[2:,2:] = q.real

    d2 = np.eye(5)
    d2[0,1] = 0.5*dt/mass
    tangent_x = d2 @ c @ d1

    # Canonical-to-spin and spin-to-canonical coordinate Jacobians.
    cin = np.zeros((5,4))
    cin[0,0] = 1.0
    cin[1,2] = 1.0
    cin[2:,1] = du_dphi
    cin[2:,3] = du_ds
    ux, uy = u2[0], u2[1]
    denom = ux*ux+uy*uy
    if denom <= 64.0*np.finfo(float).eps:
        raise ValueError("azimuth is singular at a spin pole")
    cout = np.zeros((4,5))
    cout[0,0] = 1.0
    cout[1,2] = -uy/denom
    cout[1,3] = ux/denom
    cout[2,1] = 1.0
    cout[3,4] = 0.5
    mstep = cout @ tangent_x @ cin
    return np.concatenate((znew, mstep.ravel())).astype(np.float64)

import numpy as np

def accumulate_trajectory(initial_state: "np.ndarray", timestep: float, n_steps: int, parameters: "np.ndarray") -> "np.ndarray":
    z = np.asarray(initial_state, dtype=np.float64).copy()
    p = np.asarray(parameters, dtype=np.float64)
    if isinstance(n_steps, bool) or int(n_steps) != n_steps or int(n_steps) <= 0:
        raise ValueError("n_steps must be a positive integer")
    n_steps = int(n_steps)
    rs = p[5]
    mtotal = np.eye(4)
    max_drift = 0.0
    for _ in range(n_steps):
        r, phi, mom, s = z
        if abs(s) >= rs:
            raise ValueError("canonical polar momentum must lie strictly inside the spin sphere")
        rho = np.sqrt(rs*rs-s*s)
        spin = 2.0*np.array([rho*np.cos(phi), rho*np.sin(phi), s])
        rmid = r + 0.5*float(timestep)*mom/p[0]
        propagated_spin = coupled_flow(rmid, mom, spin, timestep, p)[1:]
        max_drift = max(max_drift, abs(np.linalg.norm(propagated_spin)-2.0*rs))
        out = canonical_tangent_step(z, timestep, p)
        z = out[:4]
        mtotal = out[4:].reshape(4,4) @ mtotal
    return np.concatenate((z, mtotal.ravel(), [max_drift])).astype(np.float64)

import numpy as np

def stability_indicator(initial_state: "np.ndarray", timestep: float, n_steps: int, parameters: "np.ndarray") -> "np.ndarray":
    summary = accumulate_trajectory(initial_state, timestep, n_steps, parameters)
    zfinal = summary[:4]
    mtotal = summary[4:20].reshape(4,4)
    max_drift = summary[20]
    ident = np.eye(2)
    zero = np.zeros((2,2))
    jmat = np.block([[zero,ident],[-ident,zero]])
    sigma = np.linalg.svd(mtotal, compute_uv=False)[0]
    symplectic_defect = np.linalg.norm(mtotal.T@jmat@mtotal-jmat, ord='fro')
    determinant_defect = abs(np.linalg.det(mtotal)-1.0)
    total_time = abs(float(timestep))*int(n_steps)
    indicator = np.log(sigma)/total_time
    return np.concatenate((zfinal,[max_drift,sigma,symplectic_defect,determinant_defect,indicator])).astype(np.float64)
SCICODE_GOLD_EOF
