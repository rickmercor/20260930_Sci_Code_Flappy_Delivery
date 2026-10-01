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
from scipy.special import erf

def isotropic_qmmm_damping(r_com, beta: float) -> np.ndarray:
    """Reference implementation of the isotropic QM-MM real-space damping envelope."""
    r = np.asarray(r_com, dtype=float)
    if not np.all(np.isfinite(r)):
        raise ValueError("r_com must be finite")
    if np.any(r < 0.0):
        raise ValueError("r_com must be non-negative (COM separations)")
    if not np.isfinite(beta) or float(beta) <= 0.0:
        raise ValueError("beta must be a finite strictly-positive number")

    x = float(beta) * r
    s = erf(x) - (2.0 / np.sqrt(np.pi)) * x * np.exp(-(x ** 2))
    # numerically S^G can leave [0, 1] by ~1e-16 near the end points
    s = np.clip(s, 0.0, 1.0)
    return np.asarray(s, dtype=float)

import numpy as np
from scipy.special import erf

_SLICES = {0: slice(0, 1), 1: slice(1, 4), 2: slice(4, 13), 3: slice(13, 40), 4: slice(40, 121)}
_TOTAL = 121

def _lambda_series(s: float, kmax: int = 4) -> np.ndarray:
    """Gaussian damping kernels [lambda_0, ..., lambda_kmax] at screened distance s = r/g."""

    s = float(s)
    erf_s = erf(s)
    gauss = np.exp(-s * s)
    two_over_sqrt_pi = 2.0 / np.sqrt(np.pi)
    lams = np.empty(kmax + 1, dtype=float)
    lams[0] = erf_s
    poly = 0.0        # running sum_{i=1..k} c_i s^{2i-1}
    c = 1.0           # c_1
    for k in range(1, kmax + 1):
        poly += c * s ** (2 * k - 1)
        lams[k] = erf_s - two_over_sqrt_pi * poly * gauss
        c *= 2.0 / (2.0 * k + 1.0)
    return lams


def _bare_tensor_set(r_vec) -> np.ndarray:
    """Undamped Cartesian interaction tensors T^(0..4) (gradients of 1/r), flattened (121,)."""
    r_vec = np.asarray(r_vec, dtype=float).reshape(3)
    r = float(np.sqrt(r_vec @ r_vec))
    if not np.isfinite(r) or r <= 0.0:
        raise ValueError("r_vec must be a non-zero finite 3-vector")
    d = np.eye(3)
    x = r_vec

    t0 = np.array([1.0 / r])
    t1 = -x / r ** 3
    t2 = (3.0 * np.outer(x, x) - r ** 2 * d) / r ** 5

    t3 = np.zeros((3, 3, 3))
    xxx = x[:, None, None] * x[None, :, None] * x[None, None, :]
    dd3 = (x[:, None, None] * d[None, :, :]
           + x[None, :, None] * d[:, None, :]
           + x[None, None, :] * d[:, :, None])
    t3 = -(15.0 * xxx) / r ** 7 + 3.0 * dd3 / r ** 5

    xxxx = xxx[..., None] * x[None, None, None, :]
    # six r r delta permutations
    rr_d = (
        np.einsum("a,b,cd->abcd", x, x, d)
        + np.einsum("a,c,bd->abcd", x, x, d)
        + np.einsum("a,d,bc->abcd", x, x, d)
        + np.einsum("b,c,ad->abcd", x, x, d)
        + np.einsum("b,d,ac->abcd", x, x, d)
        + np.einsum("c,d,ab->abcd", x, x, d)
    )
    dddd = (
        np.einsum("ab,cd->abcd", d, d)
        + np.einsum("ac,bd->abcd", d, d)
        + np.einsum("ad,bc->abcd", d, d)
    )
    t4 = 105.0 * xxxx / r ** 9 - 15.0 * rr_d / r ** 7 + 3.0 * dddd / r ** 5

    return np.concatenate([t0, t1.ravel(), t2.ravel(), t3.ravel(), t4.ravel()])


def _apply_lambda(bare: np.ndarray, lams: np.ndarray, r_vec) -> np.ndarray:
    """Rebuild the damped tensor set from the bare one by inserting lambda_k rank by rank."""
    r_vec = np.asarray(r_vec, dtype=float).reshape(3)
    r = float(np.sqrt(r_vec @ r_vec))
    d = np.eye(3)
    x = r_vec
    l0, l1, l2, l3, l4 = lams

    out = np.empty(_TOTAL, dtype=float)
    out[_SLICES[0]] = l0 / r
    out[_SLICES[1]] = (-x / r ** 3 * l1)
    t2 = 3.0 * np.outer(x, x) / r ** 5 * l2 - d / r ** 3 * l1
    out[_SLICES[2]] = t2.ravel()

    xxx = x[:, None, None] * x[None, :, None] * x[None, None, :]
    dd3 = (x[:, None, None] * d[None, :, :]
           + x[None, :, None] * d[:, None, :]
           + x[None, None, :] * d[:, :, None])
    t3 = -15.0 * xxx / r ** 7 * l3 + 3.0 * dd3 / r ** 5 * l2
    out[_SLICES[3]] = t3.ravel()

    xxxx = xxx[..., None] * x[None, None, None, :]
    rr_d = (
        np.einsum("a,b,cd->abcd", x, x, d)
        + np.einsum("a,c,bd->abcd", x, x, d)
        + np.einsum("a,d,bc->abcd", x, x, d)
        + np.einsum("b,c,ad->abcd", x, x, d)
        + np.einsum("b,d,ac->abcd", x, x, d)
        + np.einsum("c,d,ab->abcd", x, x, d)
    )
    dddd = (
        np.einsum("ab,cd->abcd", d, d)
        + np.einsum("ac,bd->abcd", d, d)
        + np.einsum("ad,bc->abcd", d, d)
    )
    t4 = 105.0 * xxxx / r ** 9 * l4 - 15.0 * rr_d / r ** 7 * l3 + 3.0 * dddd / r ** 5 * l2
    out[_SLICES[4]] = t4.ravel()
    return out

def gaussian_damped_tensors(r_vec, g=None) -> np.ndarray:
    """Reference damped-tensor builder (Gaussian screening recursion, ranks 0-4)."""
    _SLICES = {0: slice(0, 1), 1: slice(1, 4), 2: slice(4, 13), 3: slice(13, 40), 4: slice(40, 121)}
    _TOTAL = 121
    r_vec = np.asarray(r_vec, dtype=float).reshape(-1)
    if r_vec.size != 3 or not np.all(np.isfinite(r_vec)):
        raise ValueError("r_vec must be a finite 3-vector")
    r = float(np.sqrt(r_vec @ r_vec))
    if r <= 0.0:
        raise ValueError("r_vec must be non-zero (COM separation)")

    bare = _bare_tensor_set(r_vec)
    if g is None:
        return bare
    if not np.isfinite(g) or float(g) <= 0.0:
        raise ValueError("g must be a finite strictly-positive width (or None for undamped)")
    lams = _lambda_series(r / float(g), kmax=4)
    return _apply_lambda(bare, lams, r_vec)

import numpy as np

_SLICES = {0: slice(0, 1), 1: slice(1, 4), 2: slice(4, 13), 3: slice(13, 40), 4: slice(40, 121)}
_TOTAL = 121

def _bare_tensor_set(r_vec) -> np.ndarray:
    """Undamped Cartesian interaction tensors T^(0..4) (gradients of 1/r), flattened (121,)."""
    r_vec = np.asarray(r_vec, dtype=float).reshape(-1)
    if r_vec.size != 3 or not np.all(np.isfinite(r_vec)):
        raise ValueError("r_vec must be a finite 3-vector")
    r = float(np.sqrt(r_vec @ r_vec))
    if r <= 0.0:
        raise ValueError("r_vec must be non-zero")
    d = np.eye(3)
    x = r_vec

    t0 = np.array([1.0 / r])
    t1 = -x / r ** 3
    t2 = (3.0 * np.outer(x, x) - r ** 2 * d) / r ** 5

    xxx = x[:, None, None] * x[None, :, None] * x[None, None, :]
    dd3 = (x[:, None, None] * d[None, :, :]
           + x[None, :, None] * d[:, None, :]
           + x[None, None, :] * d[:, :, None])
    t3 = -(15.0 * xxx) / r ** 7 + 3.0 * dd3 / r ** 5

    xxxx = xxx[..., None] * x[None, None, None, :]
    rr_d = (
        np.einsum("a,b,cd->abcd", x, x, d)
        + np.einsum("a,c,bd->abcd", x, x, d)
        + np.einsum("a,d,bc->abcd", x, x, d)
        + np.einsum("b,c,ad->abcd", x, x, d)
        + np.einsum("b,d,ac->abcd", x, x, d)
        + np.einsum("c,d,ab->abcd", x, x, d)
    )
    dddd = (
        np.einsum("ab,cd->abcd", d, d)
        + np.einsum("ac,bd->abcd", d, d)
        + np.einsum("ad,bc->abcd", d, d)
    )
    t4 = 105.0 * xxxx / r ** 9 - 15.0 * rr_d / r ** 7 + 3.0 * dddd / r ** 5

    return np.concatenate([t0, t1.ravel(), t2.ravel(), t3.ravel(), t4.ravel()])


def _unpack_tensor_set(flat: np.ndarray):
    """Split a (121,) flat tensor set into (T0 scalar, T1 (3,), T2 (3,3), T3 (3,3,3), T4 (3,3,3,3))."""
    flat = np.asarray(flat, dtype=float).reshape(-1)
    if flat.size != _TOTAL:
        raise ValueError("tensor set must have length 121")
    t0 = float(flat[_SLICES[0]][0])
    t1 = flat[_SLICES[1]].reshape(3)
    t2 = flat[_SLICES[2]].reshape(3, 3)
    t3 = flat[_SLICES[3]].reshape(3, 3, 3)
    t4 = flat[_SLICES[4]].reshape(3, 3, 3, 3)
    return t0, t1, t2, t3, t4


def _as_quadrupole_matrix(theta) -> np.ndarray:
    """Coerce a quadrupole given as (3,3) or as [xx, yy, xy, xz, yz] into a (3,3) traceless matrix."""
    theta = np.asarray(theta, dtype=float)
    if theta.shape == (3, 3):
        q = 0.5 * (theta + theta.T)
    elif theta.shape == (5,):
        xx, yy, xy, xz, yz = theta
        q = np.array([[xx, xy, xz], [xy, yy, yz], [xz, yz, -(xx + yy)]], dtype=float)
    else:
        raise ValueError("theta must be (3,3) or a 5-vector [xx, yy, xy, xz, yz]")
    if abs(np.trace(q)) > 1e-7 * (1.0 + np.max(np.abs(q))):
        raise ValueError("quadrupole must be traceless")
    return q

def multipole_potential_field(r_vec, q: float, mu, theta, tensors=None, iso: float = 1.0) -> np.ndarray:
    """Reference multipole potential/field/field-gradient (ranks 0-2)."""
    r_vec = np.asarray(r_vec, dtype=float).reshape(-1)
    if r_vec.size != 3 or not np.all(np.isfinite(r_vec)):
        raise ValueError("r_vec must be a finite 3-vector")
    if float(r_vec @ r_vec) <= 0.0:
        raise ValueError("r_vec must be non-zero")
    mu = np.asarray(mu, dtype=float).reshape(-1)
    if mu.size != 3 or not np.all(np.isfinite(mu)):
        raise ValueError("mu must be a finite 3-vector")
    if not np.isfinite(q):
        raise ValueError("q must be finite")
    if not np.isfinite(iso):
        raise ValueError("iso must be finite")
    th = _as_quadrupole_matrix(theta)

    if tensors is None:
        flat = _bare_tensor_set(r_vec)
    else:
        flat = np.asarray(tensors, dtype=float).reshape(-1)
    t0, t1, t2, t3, t4 = _unpack_tensor_set(flat)

    phi = q * t0 - mu @ t1 + (1.0 / 3.0) * np.einsum("ab,ab->", th, t2)
    field = -q * t1 + mu @ t2 - (1.0 / 3.0) * np.einsum("bc,abc->a", th, t3)
    grad = -q * t2 + np.einsum("c,abc->ab", mu, t3) - (1.0 / 3.0) * np.einsum("cd,abcd->ab", th, t4)

    out = np.empty(13, dtype=float)
    out[0] = phi
    out[1:4] = field
    out[4:13] = grad.ravel()
    return float(iso) * out

import numpy as np

def _quad_to_matrix(theta) -> np.ndarray:
    """Coerce a quadrupole given as (3,3) or [xx, yy, xy, xz, yz] into a symmetric (3,3) matrix."""
    theta = np.asarray(theta, dtype=float)
    if theta.shape == (3, 3):
        return 0.5 * (theta + theta.T)
    if theta.shape == (5,):
        xx, yy, xy, xz, yz = theta
        return np.array([[xx, xy, xz], [xy, yy, yz], [xz, yz, -(xx + yy)]], dtype=float)
    raise ValueError("theta must be (3,3) or a 5-vector [xx, yy, xy, xz, yz]")

def translate_multipole(mu, theta, R, q: float = 0.0) -> np.ndarray:
    """Reference dipole-shift + quadrupole-shift to a new common origin."""
    mu = np.asarray(mu, dtype=float).reshape(-1)
    R = np.asarray(R, dtype=float).reshape(-1)
    if mu.size != 3 or R.size != 3:
        raise ValueError("mu and R must be 3-vectors")
    if not (np.all(np.isfinite(mu)) and np.all(np.isfinite(R))) or not np.isfinite(q):
        raise ValueError("mu, R and q must be finite")
    th = _quad_to_matrix(theta)
    q = float(q)
    Rx, Ry, Rz = R
    mx, my, mz = mu

    mu_new = mu - q * R

    txx = th[0, 0] - 2 * mx * Rx + my * Ry + mz * Rz + q * (Rx ** 2 - 0.5 * (Ry ** 2 + Rz ** 2))
    tyy = th[1, 1] - 2 * my * Ry + mx * Rx + mz * Rz + q * (Ry ** 2 - 0.5 * (Rx ** 2 + Rz ** 2))
    tzz = th[2, 2] - 2 * mz * Rz + mx * Rx + my * Ry + q * (Rz ** 2 - 0.5 * (Rx ** 2 + Ry ** 2))
    txy = th[0, 1] - 1.5 * (mx * Ry + my * Rx) + 1.5 * q * Rx * Ry
    txz = th[0, 2] - 1.5 * (mx * Rz + mz * Rx) + 1.5 * q * Rx * Rz
    tyz = th[1, 2] - 1.5 * (my * Rz + mz * Ry) + 1.5 * q * Ry * Rz

    th_new = np.array([[txx, txy, txz], [txy, tyy, tyz], [txz, tyz, tzz]], dtype=float)

    out = np.empty(12, dtype=float)
    out[:3] = mu_new
    out[3:] = th_new.ravel()
    return out

import numpy as np

_MIN_RESTARTS = 64

def _kmeans_pp_lloyd(X, k, rng, tol, max_iter=300):
    """One k-means++ seeding + Lloyd run. Returns (centres (k,d), labels (N,), inertia)."""
    n, d = X.shape

    # --- k-means++ initialisation (squared-distance / D^2 sampling) ---
    centres = np.empty((k, d), dtype=float)
    centres[0] = X[int(rng.integers(n))]
    d2 = np.sum((X - centres[0]) ** 2, axis=1)
    for c in range(1, k):
        total = float(d2.sum())
        if total <= 0.0:
            centres[c] = X[int(rng.integers(n))]
        else:
            r = float(rng.random()) * total
            cand = int(np.searchsorted(np.cumsum(d2), r, side="right"))
            cand = min(cand, n - 1)
            if d2[cand] == 0.0:
                cand = int(np.argmax(d2))
            centres[c] = X[cand]
        d2 = np.minimum(d2, np.sum((X - centres[c]) ** 2, axis=1))

    # --- Lloyd iteration ---
    labels = np.full(n, -1, dtype=int)
    for _ in range(int(max_iter)):
        dist2 = np.sum((X[:, None, :] - centres[None, :, :]) ** 2, axis=2)   # (N, k)
        new_labels = np.argmin(dist2, axis=1)
        new_centres = centres.copy()
        for c in range(k):
            m = new_labels == c
            if m.any():
                new_centres[c] = X[m].mean(axis=0)
            else:  # relocate an empty cluster onto its worst-fit point
                far = int(np.argmax(dist2[np.arange(n), new_labels]))
                new_centres[c] = X[far]
                new_labels[far] = c
        shift = float(np.sqrt(np.sum((new_centres - centres) ** 2, axis=1)).sum())
        stable = np.array_equal(new_labels, labels)
        centres, labels = new_centres, new_labels
        if stable and shift <= float(tol):
            break

    inertia = float(np.sum((X - centres[labels]) ** 2))
    return centres, labels, inertia


def _run_kmeans(X, k, random_state, n_init, tol):
    """k-means with many restarts; keeps the lowest-inertia clustering. Deterministic."""
    rng = np.random.default_rng(int(random_state))
    n_restarts = max(int(n_init), _MIN_RESTARTS)
    best = None
    for _ in range(n_restarts):
        centres, labels, inertia = _kmeans_pp_lloyd(X, int(k), rng, float(tol))
        if best is None or inertia < best[2] - 1e-12:
            best = (centres, labels, inertia)
    return best[0], best[1]

def kmeans_expansion_centers(xy, z, n_clusters: int = 3, random_state: int = 0,
                                     n_init: int = 10, tol: float = 1e-6) -> np.ndarray:
    """Reference K-means clustering of the pooled far-field image sites (numpy only)."""
    xy = np.asarray(xy, dtype=float)
    z = np.asarray(z, dtype=float).reshape(-1)
    if xy.ndim != 2 or xy.shape[1] != 2:
        raise ValueError("xy must have shape (N, 2)")
    n = xy.shape[0]
    if z.shape[0] != n:
        raise ValueError("z must have shape (N,)")
    if not (np.all(np.isfinite(xy)) and np.all(np.isfinite(z))):
        raise ValueError("inputs must be finite")
    if not isinstance(n_clusters, (int, np.integer)) or n_clusters < 1 or n_clusters > n:
        raise ValueError("n_clusters must be an integer in [1, N]")

    cxy, raw = _run_kmeans(xy, int(n_clusters), random_state, n_init, tol)

    centres = []
    for k in range(int(n_clusters)):
        m = raw == k
        cz = float(np.mean(z[m])) if np.any(m) else 0.0
        centres.append((float(cxy[k, 0]), float(cxy[k, 1]), cz, k))
    centres.sort(key=lambda c: (round(c[0], 10), round(c[1], 10)))
    old_to_new = {c[3]: i for i, c in enumerate(centres)}
    labels = np.array([old_to_new[int(r)] for r in raw], dtype=float)

    out = np.empty(3 * int(n_clusters) + n, dtype=float)
    for i, (cx, cy, cz, _) in enumerate(centres):
        out[3 * i: 3 * i + 3] = (cx, cy, cz)
    out[3 * int(n_clusters):] = labels
    return out

import numpy as np
from scipy.special import erf
 
def _detrace(m: np.ndarray) -> np.ndarray:
    """Traceless part of a (3,3) matrix (the 'traceless quadrupole response')."""
    m = np.asarray(m, dtype=float)
    return m - np.eye(3) * (np.trace(m) / 3.0)
 
 
def _quad_stack(theta, n) -> np.ndarray:
    """Coerce an (n,3,3) or (n,5) quadrupole stack to (n,3,3) symmetric matrices."""
    theta = np.asarray(theta, dtype=float)
    if theta.shape == (n, 3, 3):
        return 0.5 * (theta + np.transpose(theta, (0, 2, 1)))
    if theta.shape == (n, 5):
        xx, yy, xy, xz, yz = (theta[:, i] for i in range(5))
        out = np.zeros((n, 3, 3))
        out[:, 0, 0] = xx
        out[:, 1, 1] = yy
        out[:, 2, 2] = -(xx + yy)
        out[:, 0, 1] = out[:, 1, 0] = xy
        out[:, 0, 2] = out[:, 2, 0] = xz
        out[:, 1, 2] = out[:, 2, 1] = yz
        return out
    raise ValueError(f"quadrupole stack must be ({n},3,3) or ({n},5)")
 
 
def _contract(T2, T3, T4, mu, th):
    """q = 0 field (3,) and field gradient (3,3) at the target from a source (mu, th)."""
    V = mu @ T2 - (1.0 / 3.0) * np.einsum("bc,abc->a", th, T3)
    G = np.einsum("c,abc->ab", mu, T3) - (1.0 / 3.0) * np.einsum("cd,abcd->ab", th, T4)
    return V, G
 
 
def _target_field(v0, g0, terms, mm_mu, mm_th):
    """Total on-site field / gradient at one target for the current MM moments."""
    V = np.array(v0, dtype=float)
    G = np.array(g0, dtype=float)
    for (j, T2, T3, T4, iso) in terms:
        Vj, Gj = _contract(np.asarray(T2, float), np.asarray(T3, float), np.asarray(T4, float),
                           mm_mu[int(j)], mm_th[int(j)])
        V += iso * Vj
        G += iso * Gj
    return V, G
 
def mm_scf_energy(perm_mu, perm_th, v_const, g_const, mm_terms, alpha: float, C: float,
                          mix: float = 0.35, tol: float = 1e-8, max_iter: int = 20000) -> float:
    """Reference MM SCF + coupled-energy evaluation (pure linear-response solver)."""
    perm_mu = np.asarray(perm_mu, dtype=float)
    if perm_mu.shape != (4, 3):
        raise ValueError("perm_mu must have shape (4, 3)")
    perm_th = _quad_stack(perm_th, 4)
    v_const = np.asarray(v_const, dtype=float)
    g_const = np.asarray(g_const, dtype=float)
    if v_const.shape != (4, 3) or g_const.shape != (4, 3, 3):
        raise ValueError("v_const must be (4, 3) and g_const (4, 3, 3)")
    if not hasattr(mm_terms, "__len__") or len(mm_terms) != 4:
        raise ValueError("mm_terms must be a length-4 list (QM + 3 MM targets)")
    if not (0.0 < float(mix) < 1.0):
        raise ValueError("mix must lie strictly in (0, 1)")
    if not np.isfinite(tol) or float(tol) <= 0.0:
        raise ValueError("tol must be a finite positive number")
    if not np.isfinite(alpha) or not np.isfinite(C) or float(alpha) < 0.0:
        raise ValueError("alpha must be finite and non-negative, C finite")
 
    a, Cc, t = float(alpha), float(C), float(mix)
    mm_perm_mu = perm_mu[1:]        # (3, 3)
    mm_perm_th = perm_th[1:]        # (3, 3, 3)
 
    ind_mu = np.zeros((3, 3))
    ind_th = np.zeros((3, 3, 3))
    converged = False
    for _ in range(int(max_iter)):
        tot_mu = mm_perm_mu + ind_mu
        tot_th = mm_perm_th + ind_th
        fresh_mu = np.zeros((3, 3))
        fresh_th = np.zeros((3, 3, 3))
        for i in range(3):                 # MM targets are list entries 1..3
            V, G = _target_field(v_const[i + 1], g_const[i + 1], mm_terms[i + 1], tot_mu, tot_th)
            fresh_mu[i] = a * V
            fresh_th[i] = Cc * _detrace(G)
        new_mu = (1.0 - t) * fresh_mu + t * ind_mu
        new_th = (1.0 - t) * fresh_th + t * ind_th
        delta = max(np.max(np.abs(new_mu - ind_mu)), np.max(np.abs(new_th - ind_th)))
        ind_mu, ind_th = new_mu, new_th
        if delta < float(tol):
            converged = True
            break
    if not converged:
        raise RuntimeError("MM SCF did not converge within max_iter")
 
    tot_mu = mm_perm_mu + ind_mu
    tot_th = mm_perm_th + ind_th
    total = 0.0
    for tgt in range(4):
        V, G = _target_field(v_const[tgt], g_const[tgt], mm_terms[tgt], tot_mu, tot_th)
        total += perm_mu[tgt] @ V + (1.0 / 3.0) * np.einsum("ab,ab->", perm_th[tgt], G)
    return float(-0.5 * total)

import numpy as np
 
ANGSTROM_TO_BOHR = 1.8897261246
_TOTAL = 121
 
def _reference_system():
    """The prompt's geometry / moments / parameters in Angstrom and atomic units."""
    return dict(
        qm_pos=np.array([2.10, 2.10, 1.50]),
        qm_mu=np.array([0.10, 0.05, -0.60]),
        qm_theta=np.array([0.20, 0.15, -0.05, 0.10, -0.08]),   # xx, yy, xy, xz, yz
        mm_pos=np.array([[0.55, 0.40, 0.00],
                         [2.95, 1.35, 0.30],
                         [1.65, 3.10, -0.25]]),
        mm_mu=np.array([[0.42, -0.28, 0.09],
                        [-0.18, 0.52, -0.06],
                        [0.06, -0.38, 0.55]]),
        mm_theta=np.array([[0.75, -0.45, 0.22, -0.12, 0.05],
                           [-0.38, 0.85, -0.09, 0.18, -0.22],
                           [0.28, 0.12, 0.06, -0.28, 0.14]]),
        a1=np.array([4.20, 0.0, 0.0]),
        a2=np.array([0.0, 4.20, 0.0]),
        alpha=9.85, C=24.5,
    )
 
 
def _q2m(theta) -> np.ndarray:
    """Coerce a quadrupole given as (3,3) or [xx, yy, xy, xz, yz] to a traceless (3,3) matrix."""
    theta = np.asarray(theta, dtype=float)
    if theta.shape == (3, 3):
        return 0.5 * (theta + theta.T)
    xx, yy, xy, xz, yz = theta
    return np.array([[xx, xy, xz], [xy, yy, yz], [xz, yz, -(xx + yy)]], dtype=float)
 
 
def _far_field_sites(base_pos, base_mu, base_th, a1, a2, near_cells, far_window):
    """Pool the far-field periodic images of the four base sites (QM + 3 MM)."""
    near = {(int(a), int(b)) for a, b in near_cells}
    xy, z, mu, th = [], [], [], []
    for nx in far_window:
        for ny in far_window:
            if (nx, ny) in near:
                continue
            shift = nx * a1 + ny * a2
            for b in range(base_pos.shape[0]):
                p = base_pos[b] + shift
                xy.append(p[:2]); z.append(p[2]); mu.append(base_mu[b]); th.append(base_th[b])
    return np.array(xy), np.array(z), np.array(mu), np.array(th)
 
def coupled_qmmm_energy(beta_inv_angstrom: float = 0.291, g_angstrom: float = 0.32,
                                n_clusters: int = 3, mixing: float = 0.35, scf_tol: float = 1e-8) -> float:
    """Reference orchestrator: near/far split, then steps 5 -> 4 -> 1/2/3 -> 6."""
    if not np.isfinite(beta_inv_angstrom) or float(beta_inv_angstrom) <= 0.0:
        raise ValueError("beta_inv_angstrom must be finite and positive")
    if not np.isfinite(g_angstrom) or float(g_angstrom) <= 0.0:
        raise ValueError("g_angstrom must be finite and positive")
    if not isinstance(n_clusters, (int, np.integer)) or n_clusters < 1:
        raise ValueError("n_clusters must be a positive integer")
 
    B = ANGSTROM_TO_BOHR
    s = _reference_system()
    qm_pos = s["qm_pos"] * B
    mm_pos = s["mm_pos"] * B
    a1, a2 = s["a1"] * B, s["a2"] * B
    qm_mu = s["qm_mu"]
    qm_th = _q2m(s["qm_theta"])
    mm_mu = s["mm_mu"]
    mm_th = np.stack([_q2m(r) for r in s["mm_theta"]])
    alpha, C = s["alpha"], s["C"]
    beta = float(beta_inv_angstrom) / B
    g = float(g_angstrom) * B
    K = int(n_clusters)
 
    near_cells = [(nx, ny) for nx in (-1, 0, 1) for ny in (-1, 0, 1)]
    far_window = (-2, -1, 0, 1, 2)
 
    base_pos = np.vstack([qm_pos, mm_pos])
    base_mu = np.vstack([qm_mu, mm_mu])
    base_th = np.concatenate([qm_th[None, :, :], mm_th], axis=0)
 
    # ---- far field: step 5 (K-means clustering) then step 4 (origin shift) ----
    xy, z, fmu, fth = _far_field_sites(base_pos, base_mu, base_th, a1, a2, near_cells, far_window)
    packed = kmeans_expansion_centers(xy, z, n_clusters=K, random_state=0, n_init=10, tol=1e-6)
    cluster_pos = packed[:3 * K].reshape(K, 3)
    labels = packed[3 * K:].astype(int)
 
    cluster_mu = np.zeros((K, 3))
    cluster_th = np.zeros((K, 3, 3))
    for i in range(xy.shape[0]):
        k = labels[i]
        site = np.array([xy[i, 0], xy[i, 1], z[i]])
        shifted = np.asarray(translate_multipole(fmu[i], fth[i], cluster_pos[k] - site, q=0.0),
                             dtype=float).reshape(12)
        cluster_mu[k] += shifted[:3]
        cluster_th[k] += shifted[3:].reshape(3, 3)
 
    # ---- near field: periodic instances of the QM site and the 3 MM sites ----
    def _inst(bp):
        return [(bp + nx * a1 + ny * a2, nx, ny) for (nx, ny) in near_cells]
    qm_inst = _inst(qm_pos)
    mm_inst = [(p, nx, ny, j) for j in range(3) for (p, nx, ny) in _inst(mm_pos[j])]
 
    # ---- assemble the inputs for step 6 (steps 1, 2, 3) ----
    home = [qm_pos, mm_pos[0], mm_pos[1], mm_pos[2]]
    perm_mu = np.vstack([qm_mu, mm_mu])
    perm_th = np.concatenate([qm_th[None, :, :], mm_th], axis=0)
    v_const = np.zeros((4, 3))
    g_const = np.zeros((4, 3, 3))
    mm_terms = [[], [], [], []]
 
    for tgt in range(4):
        is_qm = (tgt == 0)
        tp = home[tgt]
 
        if not is_qm:
            for (p, nx, ny) in qm_inst:
                rvec = tp - p
                d = float(np.sqrt(rvec @ rvec))
                S = float(isotropic_qmmm_damping(np.array([d]), beta)[0])
                res = multipole_potential_field(rvec, 0.0, qm_mu, qm_th, tensors=None, iso=S)
                v_const[tgt] += res[1:4]
                g_const[tgt] += res[4:13].reshape(3, 3)
 
        for k in range(K):
            rvec = tp - cluster_pos[k]
            res = multipole_potential_field(rvec, 0.0, cluster_mu[k], cluster_th[k], tensors=None, iso=1.0)
            v_const[tgt] += res[1:4]
            g_const[tgt] += res[4:13].reshape(3, 3)
 
        for (p, nx, ny, j) in mm_inst:
            if (not is_qm) and j == tgt - 1 and nx == 0 and ny == 0:
                continue
            rvec = tp - p
            if is_qm:
                d = float(np.sqrt(rvec @ rvec))
                iso = float(isotropic_qmmm_damping(np.array([d]), beta)[0])
                flat = gaussian_damped_tensors(rvec, None)          # bare QM--MM tensor
            else:
                iso = 1.0
                flat = gaussian_damped_tensors(rvec, g)             # Gaussian-damped MM--MM tensor
            flat = np.asarray(flat, dtype=float)
            T2 = flat[4:13].reshape(3, 3)
            T3 = flat[13:40].reshape(3, 3, 3)
            T4 = flat[40:121].reshape(3, 3, 3, 3)
            mm_terms[tgt].append((j, T2, T3, T4, iso))
 
    # ---- step 6: MM SCF + converged coupled energy ----
    energy = mm_scf_energy(perm_mu, perm_th, v_const, g_const, mm_terms,
                           alpha, C, mix=float(mixing), tol=float(scf_tol))
    return float(energy)
SCICODE_GOLD_EOF
