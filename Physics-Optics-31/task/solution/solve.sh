#!/bin/bash
# Oracle: writes the gold solution so the verifier scores reward 1.0.
cat > /app/solution.py <<'SCICODE_GOLD_EOF'
import numpy as np
try:
    import scipy, scipy.linalg, scipy.sparse, scipy.optimize, scipy.spatial, scipy.integrate, scipy.stats, scipy.special, scipy.signal
except Exception:
    pass
import math

from numbers import Integral, Real
from scipy.stats import norm, qmc
import numpy as np

def generate_weighted_sobol_nodes(
    q0: np.ndarray, p0: np.ndarray,
    gamma0: np.ndarray, gamma: np.ndarray,
    n_nodes: int, hbar: float = 1.0, seed: int = 23,
) -> np.ndarray:
    """Reference implementation."""
    q0, p0 = np.asarray(q0, float), np.asarray(p0, float)
    g0, g = np.asarray(gamma0, float), np.asarray(gamma, float)
    if q0.ndim != 1 or p0.shape != q0.shape or q0.size < 1:
        raise ValueError("q0 and p0 must be aligned vectors")
    d = q0.size
    if g0.shape != (d, d) or g.shape != (d, d):
        raise ValueError("width matrices must be d by d")
    if not np.all(np.isfinite(np.r_[q0, p0, g0.ravel(), g.ravel()])):
        raise ValueError("inputs must be finite")
    if not np.allclose(g0, g0.T) or not np.allclose(g, g.T):
        raise ValueError("width matrices must be symmetric")
    try:
        np.linalg.cholesky(g0); np.linalg.cholesky(g)
    except np.linalg.LinAlgError as exc:
        raise ValueError("width matrices must be positive definite") from exc
    if isinstance(n_nodes, bool) or not isinstance(n_nodes, Integral):
        raise ValueError("n_nodes must be a power-of-two integer")
    n = int(n_nodes)
    if n < 1 or n & (n - 1):
        raise ValueError("n_nodes must be a power of two")
    if isinstance(hbar, bool) or not isinstance(hbar, Real) or not np.isfinite(hbar) or hbar <= 0:
        raise ValueError("hbar must be positive and finite")
    if isinstance(seed, bool) or not isinstance(seed, Integral):
        raise ValueError("seed must be an integer")
    sigma = np.block([[np.linalg.inv(g0) + np.linalg.inv(g), np.zeros((d, d))],
                      [np.zeros((d, d)), g0 + g]])
    u = qmc.Sobol(2 * d, scramble=True, seed=int(seed)).random_base2(int(np.log2(n)))
    standard = norm.ppf(np.clip(u, np.finfo(float).eps, 1 - np.finfo(float).eps))
    return np.r_[q0, p0] + np.sqrt(float(hbar)) * standard @ np.linalg.cholesky(sigma).T

from numbers import Real
import numpy as np

def compute_gaussian_expansion_coefficients(
    nodes: np.ndarray, q0: np.ndarray, p0: np.ndarray,
    gamma0: np.ndarray, gamma: np.ndarray,
    hbar: float = 1.0,
) -> np.ndarray:
    """Reference implementation using the closed Gaussian overlap."""
    z = np.asarray(nodes, float)
    q0, p0 = np.asarray(q0, float), np.asarray(p0, float)
    g0, g = np.asarray(gamma0, float), np.asarray(gamma, float)
    if q0.ndim != 1 or q0.size < 1 or p0.shape != q0.shape:
        raise ValueError("q0 and p0 must be aligned nonempty vectors")
    d = q0.size
    if z.ndim != 2 or z.shape[1] != 2 * d or p0.shape != q0.shape or z.shape[0] < 1:
        raise ValueError("nodes and centers have incompatible shapes")
    if g0.shape != (d, d) or g.shape != (d, d):
        raise ValueError("width matrices must be d by d")
    if isinstance(hbar, bool) or not isinstance(hbar, Real) or not np.isfinite(hbar) or hbar <= 0:
        raise ValueError("hbar must be positive and finite")
    if not np.all(np.isfinite(np.r_[z.ravel(), q0, p0, g0.ravel(), g.ravel()])):
        raise ValueError("inputs must be finite")
    if not np.allclose(g0, g0.T) or not np.allclose(g, g.T):
        raise ValueError("width matrices must be symmetric")
    try:
        np.linalg.cholesky(g0); np.linalg.cholesky(g)
    except np.linalg.LinAlgError as exc:
        raise ValueError("width matrices must be positive definite") from exc
    h = float(hbar); n = z.shape[0]
    sigma = np.block([[np.linalg.inv(g0) + np.linalg.inv(g), np.zeros((d, d))],
                      [np.zeros((d, d)), g0 + g]])
    dz = z - np.r_[q0, p0]
    weight = np.exp(-np.einsum("ni,ij,nj->n", dz, np.linalg.inv(sigma), dz) / (2 * h))
    a = g0 + g
    inva = np.linalg.inv(a)
    pref = (np.linalg.det(g0) * np.linalg.det(g)) ** .25
    pref *= (2 * np.pi * h) ** (d / 2) / ((np.pi * h) ** (d / 2) * np.sqrt(np.linalg.det(a)))
    q, p = z[:, :d], z[:, d:]
    b = q @ g.T + q0 @ g0.T + 1j * (p0 - p)
    constant = (-.5 * np.einsum("ni,ij,nj->n", q, g, q)).astype(complex)
    constant += -.5 * q0 @ g0 @ q0 + 1j * np.einsum("ni,ni->n", p, q) - 1j * p0 @ q0
    exponent = (constant + .5 * np.einsum("ni,ij,nj->n", b, inva, b)) / h
    overlap = pref * np.exp(exponent)
    return overlap / (n * (2 * np.pi * h) ** d * weight)

from numbers import Real
import numpy as np
def evaluate_quasistatic_dipole(
    t: float, positions: np.ndarray,
    field_amplitude: float, radius: float,
    dielectric: complex, omega: float,
    charge: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reference implementation of Eqs. (2)-(3) and analytic derivatives."""
    r = np.asarray(positions, float)
    scalars = (t, field_amplitude, radius, omega, charge)
    if r.ndim < 1 or r.shape[-1] != 2 or not np.all(np.isfinite(r)):
        raise ValueError("positions must be finite with final dimension two")
    if any(isinstance(v, bool) or not isinstance(v, Real) or not np.isfinite(v) for v in scalars):
        raise ValueError("real scalar inputs must be finite")
    if float(radius) <= 0 or not np.isfinite(complex(dielectric)):
        raise ValueError("radius must be positive and dielectric finite")
    eps = complex(dielectric)
    if abs(eps + 1) == 0:
        raise ValueError("dielectric contrast is singular")
    x, y = r[..., 0], r[..., 1]
    s = x * x + y * y
    inside = s < float(radius) ** 2
    safe_s = np.where(inside, 1.0, s)
    contrast = abs((eps - 1) / (eps + 1))
    base = float(field_amplitude) * contrast
    k = base * float(radius) ** 2
    phi = np.where(inside, base * x, k * x / safe_s)
    grad = np.zeros(r.shape, float)
    grad[..., 0] = np.where(inside, base, k * (y * y - x * x) / safe_s ** 2)
    grad[..., 1] = np.where(inside, 0.0, -2 * k * x * y / safe_s ** 2)
    hess = np.zeros(r.shape[:-1] + (2, 2), float)
    hxx = 2 * k * x * (x * x - 3 * y * y) / safe_s ** 3
    hxy = 2 * k * y * (3 * x * x - y * y) / safe_s ** 3
    hess[..., 0, 0] = np.where(inside, 0.0, hxx)
    hess[..., 0, 1] = hess[..., 1, 0] = np.where(inside, 0.0, hxy)
    hess[..., 1, 1] = np.where(inside, 0.0, -hxx)
    factor = -float(charge) * np.cos(float(omega) * float(t))
    return factor * phi, factor * grad, factor * hess

from numbers import Real
import numpy as np
def initialize_hagedorn_state(
    nodes: np.ndarray, gamma: np.ndarray, hbar: float = 1.0
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Reference implementation of the paper's initial conditions."""
    z = np.asarray(nodes, float)
    g = np.asarray(gamma, float)
    if z.ndim != 2 or z.shape[0] < 1 or z.shape[1] < 2 or z.shape[1] % 2:
        raise ValueError("nodes must have shape (N,2*d)")
    d = z.shape[1] // 2
    if g.shape != (d, d) or not np.all(np.isfinite(np.r_[z.ravel(), g.ravel()])):
        raise ValueError("gamma must be finite and d by d")
    if not np.allclose(g, g.T):
        raise ValueError("gamma must be symmetric")
    values, vectors = np.linalg.eigh(g)
    if np.any(values <= 0):
        raise ValueError("gamma must be positive definite")
    if isinstance(hbar, bool) or not isinstance(hbar, Real) or not np.isfinite(hbar) or hbar <= 0:
        raise ValueError("hbar must be positive and finite")
    root = (vectors * np.sqrt(values)) @ vectors.T
    invroot = (vectors * (1 / np.sqrt(values))) @ vectors.T
    q, p = z[:, :d].copy(), z[:, d:].copy()
    qmat = np.broadcast_to(np.sqrt(float(hbar)) * invroot, (len(z), d, d)).copy()
    pmat = np.broadcast_to(1j * np.sqrt(float(hbar)) * root, (len(z), d, d)).copy()
    action = np.zeros(len(z), float)
    return q, p, qmat, pmat, action

from numbers import Integral, Real
import numpy as np
def propagate_hagedorn_state(
    q: np.ndarray, p: np.ndarray, Q: np.ndarray,
    P: np.ndarray, S: np.ndarray, dt: float, n_steps: int,
    mass: float, field_amplitude: float, radius: float,
    dielectric: complex, omega: float,
    charge: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Reference source-order propagation with local Hessian width dynamics."""
    q, p = np.array(q, float, copy=True), np.array(p, float, copy=True)
    Q, P = np.array(Q, complex, copy=True), np.array(P, complex, copy=True)
    S = np.array(S, float, copy=True)
    if q.ndim != 2 or q.shape[1] != 2 or p.shape != q.shape:
        raise ValueError("q and p must have shape (N,2)")
    if Q.shape != (len(q), 2, 2) or P.shape != Q.shape or S.shape != (len(q),):
        raise ValueError("Hagedorn arrays are not aligned")
    if not np.all(np.isfinite(np.r_[q.ravel(), p.ravel(), Q.real.ravel(), Q.imag.ravel(), P.real.ravel(), P.imag.ravel(), S])):
        raise ValueError("state must be finite")
    if isinstance(n_steps, bool) or not isinstance(n_steps, Integral) or n_steps < 0:
        raise ValueError("n_steps must be a nonnegative integer")
    if any(isinstance(v, bool) or not isinstance(v, Real) or not np.isfinite(v) for v in (dt, mass)):
        raise ValueError("dt and mass must be finite reals")
    if dt <= 0 or mass <= 0:
        raise ValueError("dt and mass must be positive")
    step = float(dt); m = float(mass)
    for k in range(int(n_steps)):
        t_mid = (k + .5) * step
        q += .5 * step * p / m
        Q += .5 * step * P / m
        value, grad, hess = evaluate_quasistatic_dipole(
            t_mid, q, field_amplitude, radius, dielectric, omega, charge
        )
        p -= .5 * step * grad
        P -= step * np.einsum("nij,njk->nik", hess, Q)
        S += step * (np.einsum("ni,ni->n", p, p) / (2 * m) - value)
        p -= .5 * step * grad
        q += .5 * step * p / m
        Q += .5 * step * P / m
    return q, p, Q, P, S

from numbers import Real
import numpy as np

def reconstruct_tgwp_momentum(
    x: np.ndarray, y: np.ndarray, coefficients: np.ndarray,
    q: np.ndarray, p: np.ndarray, Q: np.ndarray,
    P: np.ndarray, S: np.ndarray,
    hbar: float = 1.0,
) -> np.ndarray:
    """Reference closed-form Gaussian Fourier reconstruction."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    c = np.asarray(coefficients, complex)
    q, p, Q, P, S = map(np.asarray, (q, p, Q, P, S))
    if c.ndim != 1 or c.size < 1:
        raise ValueError("coefficients must be a nonempty vector")
    n = c.size
    if x.ndim != 1 or y.ndim != 1 or min(x.size, y.size) < 2:
        raise ValueError("x and y must be one-dimensional axes")
    if q.shape != (n, 2) or p.shape != q.shape or Q.shape != (n, 2, 2) or P.shape != Q.shape or S.shape != (n,):
        raise ValueError("Gaussian arrays are not aligned")
    for axis in (x, y):
        if not np.all(np.isfinite(axis)):
            raise ValueError("axes must be finite")
        spacing = np.diff(axis)
        if (not np.all(np.isfinite(spacing))
                or not (np.all(spacing > 0) or np.all(spacing < 0))
                or not np.allclose(spacing, spacing[0])):
            raise ValueError("axes must be strictly monotone and uniform with nonzero spacing")
    if isinstance(hbar, bool) or not isinstance(hbar, Real) or not np.isfinite(hbar) or hbar <= 0:
        raise ValueError("hbar must be positive and finite")
    if not np.all(np.isfinite(np.r_[c.real, c.imag, q.ravel(), p.ravel(), Q.real.ravel(), Q.imag.ravel(), P.real.ravel(), P.imag.ravel(), S])):
        raise ValueError("Gaussian arrays must be finite")
    try:
        invq = np.linalg.inv(Q)
        width = np.einsum("nij,njk->nik", P, invq)
        invwidth = np.linalg.inv(width)
    except np.linalg.LinAlgError as exc:
        raise ValueError("Q and the width matrices must be invertible") from exc
    detq = np.linalg.det(Q)
    detc = np.linalg.det(-1j * width)
    if (not np.all(np.isfinite(invq)) or not np.all(np.isfinite(invwidth))
            or not np.all(np.isfinite(detq * detc)) or np.any(detq * detc == 0)):
        raise ValueError("width matrices have invalid numerical inverses or determinants")
    kx = np.fft.fftshift(2 * np.pi * np.fft.fftfreq(len(x), d=np.diff(x)[0]))
    ky = np.fft.fftshift(2 * np.pi * np.fft.fftfreq(len(y), d=np.diff(y)[0]))
    gx, gy = np.meshgrid(kx, ky, indexing="ij")
    points = np.stack((gx.ravel(), gy.ravel()), axis=-1)
    delta = p[:, None, :] - float(hbar) * points[None, :, :]
    quad = np.einsum("nmi,nij,nmj->nm", delta, invwidth, delta)
    phase = np.exp(-.5j * quad / float(hbar) - 1j * np.einsum("ni,mi->nm", q, points) + 1j * S[:, None] / float(hbar))
    amplitude = np.pi ** (-.5) * (2 * np.pi * float(hbar)) / np.sqrt(detq * detc)
    wave = np.sum(c[:, None] * amplitude[:, None] * phase, axis=0).reshape(len(x), len(y))
    norm = np.sqrt(np.sum(np.abs(wave) ** 2) * abs(np.diff(kx)[0] * np.diff(ky)[0]))
    if not np.isfinite(norm) or norm <= 0:
        raise ValueError("momentum reconstruction has invalid norm")
    return wave / norm

from numbers import Real
import numpy as np

def initialize_grid_wavefunction(
    x: np.ndarray,
    y: np.ndarray,
    q0: np.ndarray,
    p0: np.ndarray,
    gamma0: np.ndarray,
    hbar: float = 1.0,
) -> np.ndarray:
    """Reference normalized Gaussian sampling."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    q0, p0, g0 = np.asarray(q0, float), np.asarray(p0, float), np.asarray(gamma0, float)
    if x.ndim != 1 or y.ndim != 1 or min(x.size, y.size) < 2:
        raise ValueError("x and y must be one-dimensional axes")
    if q0.shape != (2,) or p0.shape != (2,) or g0.shape != (2, 2):
        raise ValueError("center and width shapes are invalid")
    for axis in (x, y):
        if not np.all(np.isfinite(axis)):
            raise ValueError("axes must be finite")
        spacing = np.diff(axis)
        if (not np.all(np.isfinite(spacing))
                or not (np.all(spacing > 0) or np.all(spacing < 0))
                or not np.allclose(spacing, spacing[0])):
            raise ValueError("axes must be strictly monotone and uniform with nonzero spacing")
    if not np.allclose(g0, g0.T):
        raise ValueError("gamma0 must be symmetric")
    try:
        np.linalg.cholesky(g0)
    except np.linalg.LinAlgError as exc:
        raise ValueError("gamma0 must be positive definite") from exc
    if isinstance(hbar, bool) or not isinstance(hbar, Real) or not np.isfinite(hbar) or hbar <= 0:
        raise ValueError("hbar must be positive and finite")
    if not np.all(np.isfinite(np.r_[x, y, q0, p0, g0.ravel()])):
        raise ValueError("inputs must be finite")
    gx, gy = np.meshgrid(x, y, indexing="ij")
    dr = np.stack((gx - q0[0], gy - q0[1]), axis=-1)
    decay = np.einsum("...i,ij,...j->...", dr, g0, dr)
    phase = np.einsum("i,...i->...", p0, dr)
    pref = np.linalg.det(g0) ** .25 / (np.pi * float(hbar)) ** .5
    wave = pref * np.exp((-0.5 * decay + 1j * phase) / float(hbar))
    norm = np.sqrt(np.sum(np.abs(wave) ** 2) * abs(np.diff(x)[0] * np.diff(y)[0]))
    if not np.isfinite(norm) or norm <= 0:
        raise ValueError("sampled field has zero or non-finite norm")
    return wave / norm

from numbers import Integral, Real
import numpy as np

def propagate_split_step_reference(
    psi: np.ndarray, x: np.ndarray, y: np.ndarray,
    dt: float, n_steps: int, mass: float,
    field_amplitude: float, radius: float, dielectric: complex,
    omega: float, charge: float = 1.0,
    hbar: float = 1.0,
) -> np.ndarray:
    """Reference Fourier Strang propagator."""
    wave = np.array(psi, complex, copy=True)
    x, y = np.asarray(x, float), np.asarray(y, float)
    if x.ndim != 1 or y.ndim != 1 or wave.shape != (len(x), len(y)) or min(len(x), len(y)) < 2:
        raise ValueError("field and axes have incompatible shapes")
    for axis in (x, y):
        if not np.all(np.isfinite(axis)):
            raise ValueError("axes must be finite")
        spacing = np.diff(axis)
        if (not np.all(np.isfinite(spacing))
                or not (np.all(spacing > 0) or np.all(spacing < 0))
                or not np.allclose(spacing, spacing[0])):
            raise ValueError("axes must be strictly monotone and uniform with nonzero spacing")
    if isinstance(n_steps, bool) or not isinstance(n_steps, Integral) or n_steps < 0:
        raise ValueError("n_steps must be a nonnegative integer")
    if any(isinstance(v, bool) or not isinstance(v, Real) or not np.isfinite(v) for v in (dt, mass, hbar)):
        raise ValueError("dt, mass, and hbar must be finite reals")
    if dt <= 0 or mass <= 0 or hbar <= 0 or not np.all(np.isfinite(np.r_[wave.real.ravel(), wave.imag.ravel()])):
        raise ValueError("positive scalars and a finite field are required")
    dx, dy = np.diff(x)[0], np.diff(y)[0]
    input_norm = np.sqrt(np.sum(np.abs(wave) ** 2) * abs(dx * dy))
    if not np.isfinite(input_norm) or input_norm <= 0:
        raise ValueError("input field has zero or non-finite cell norm")
    kx = 2 * np.pi * np.fft.fftfreq(len(x), d=dx)
    ky = 2 * np.pi * np.fft.fftfreq(len(y), d=dy)
    kinetic = np.exp(-.5j * float(hbar) * float(dt) * (kx[:, None] ** 2 + ky[None, :] ** 2) / float(mass))
    gx, gy = np.meshgrid(x, y, indexing="ij")
    points = np.stack((gx, gy), axis=-1)
    for k in range(int(n_steps)):
        tmid = (k + .5) * float(dt)
        potential = evaluate_quasistatic_dipole(tmid, points, field_amplitude, radius, dielectric, omega, charge)[0]
        half = np.exp(-.5j * float(dt) * potential / float(hbar))
        wave *= half
        wave = np.fft.ifft2(np.fft.fft2(wave) * kinetic)
        wave *= half
    norm = np.sqrt(np.sum(np.abs(wave) ** 2) * abs(dx * dy))
    if not np.isfinite(norm) or norm <= 0:
        raise ValueError("propagated field has zero or non-finite cell norm")
    return wave / norm

from numbers import Real
import numpy as np

def compute_momentum_amplitude_error(
    meshfree_momentum: np.ndarray,
    reference_wave: np.ndarray,
    dx: float,
    dy: float,
) -> float:
    """Reference unit-normalized Fourier-amplitude shape comparison."""
    mesh = np.asarray(meshfree_momentum, complex)
    ref = np.asarray(reference_wave, complex)
    if mesh.ndim != 2 or mesh.shape != ref.shape or min(mesh.shape) < 2:
        raise ValueError("fields must be aligned two-dimensional arrays")
    if not np.all(np.isfinite(np.r_[mesh.real.ravel(), mesh.imag.ravel(), ref.real.ravel(), ref.imag.ravel()])):
        raise ValueError("fields must be finite")
    for spacing in (dx, dy):
        if isinstance(spacing, bool) or not isinstance(spacing, Real) or not np.isfinite(spacing) or spacing <= 0:
            raise ValueError("spacings must be positive finite reals")
    ref_k = float(dx) * float(dy) * np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(ref)))
    mesh_norm = np.linalg.norm(mesh)
    ref_norm = np.linalg.norm(ref_k)
    if not np.isfinite(mesh_norm) or mesh_norm <= 0:
        raise ValueError("mesh-free momentum field has zero or non-finite norm")
    if not np.isfinite(ref_norm) or ref_norm <= 0:
        raise ValueError("reference field has zero or non-finite Fourier norm")
    mesh_k = mesh / mesh_norm
    ref_k = ref_k / ref_norm
    denominator = np.linalg.norm(np.abs(ref_k))
    return float(np.linalg.norm(np.abs(mesh_k) - np.abs(ref_k)) / denominator)

from numbers import Integral, Real
import numpy as np

def run_meshfree_grid_benchmark(
    n_nodes: int = 32,
    nx: int = 48,
    ny: int = 40,
    dt: float = .02,
    n_steps: int = 40,
    field_amplitude: float = .12,
    gamma_scale: float = 4.0,
    seed: int = 23,
) -> float:
    """Reference orchestrator calling only prior oracle functions."""
    for value, name, lower in ((nx,"nx",8),(ny,"ny",8),(n_steps,"n_steps",0)):
        if isinstance(value, bool) or not isinstance(value, Integral) or int(value) < lower:
            raise ValueError(f"{name} has an invalid integer value")
    for value, name in ((dt,"dt"),(gamma_scale,"gamma_scale"),(field_amplitude,"field_amplitude")):
        if isinstance(value, bool) or not isinstance(value, Real) or not np.isfinite(value):
            raise ValueError(f"{name} must be finite")
    if dt <= 0 or gamma_scale <= 0:
        raise ValueError("dt and gamma_scale must be positive")
    q0 = np.array([-2.4, .65]); p0 = np.array([4.0, 0.0])
    gamma0 = np.diag([.8, 4.0]); gamma = float(gamma_scale) * gamma0
    x = np.linspace(-5.0, 3.0, int(nx), endpoint=True)
    y = np.linspace(-2.0, 2.0, int(ny), endpoint=True)
    probe = evaluate_quasistatic_dipole(0.0, np.array([[.4, 0.0]]), field_amplitude, .4, -24.061+1.5068j, 1.3)[0][0]
    contrast = abs((-24.061+1.5068j - 1) / (-24.061+1.5068j + 1))
    calibrated_field = -probe / (.4 * contrast)
    nodes = generate_weighted_sobol_nodes(q0, p0, gamma0, gamma, n_nodes, 1.0, seed)
    coeffs = compute_gaussian_expansion_coefficients(nodes, q0, p0, gamma0, gamma, 1.0)
    q, p, Q, P, S = initialize_hagedorn_state(nodes, gamma, 1.0)
    q, p, Q, P, S = propagate_hagedorn_state(q, p, Q, P, S, dt, n_steps, 1.0, calibrated_field, .4, -24.061+1.5068j, 1.3, 1.0)
    mesh = reconstruct_tgwp_momentum(x, y, coeffs, q, p, Q, P, S, 1.0)
    initial = initialize_grid_wavefunction(x, y, q0, p0, gamma0, 1.0)
    reference = propagate_split_step_reference(initial, x, y, dt, n_steps, 1.0, calibrated_field, .4, -24.061+1.5068j, 1.3, 1.0, 1.0)
    return compute_momentum_amplitude_error(mesh, reference, x[1]-x[0], y[1]-y[0])
SCICODE_GOLD_EOF
