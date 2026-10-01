#!/bin/bash
# Oracle: writes the gold solution so the verifier scores reward 1.0.
cat > /app/solution.py <<'SCICODE_GOLD_EOF'
import numpy as np
try:
    import scipy, scipy.linalg, scipy.sparse, scipy.optimize, scipy.spatial, scipy.integrate, scipy.stats, scipy.special, scipy.signal
except Exception:
    pass
import math

def cl_detector_coords(shape: "tuple", n_views: int, theta_deg: float, sod: float,
                               sdd: float, voxel_size: float, det_pixel: float,
                               det_size: int) -> "np.ndarray":
    """Detector coordinates (u, w) of every voxel centre, per view: rotational cone-beam CL."""
    import numpy as np
    nx, ny, nz = (int(s) for s in shape)
    theta = np.deg2rad(float(theta_deg))
    out = np.empty((2, int(n_views), nz, ny, nx), dtype=np.float64)
    zz, yy, xx = np.meshgrid((np.arange(nz) - nz / 2.0 + 0.5) * voxel_size,
                             (np.arange(ny) - ny / 2.0 + 0.5) * voxel_size,
                             (np.arange(nx) - nx / 2.0 + 0.5) * voxel_size, indexing="ij")
    pts = np.stack([xx, yy, zz], axis=-1)
    for v in range(int(n_views)):
        phi = 2.0 * np.pi * v / float(n_views)
        src = sod * np.array([np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi),
                              np.cos(theta)])
        nrm = src / np.linalg.norm(src)
        e1 = np.array([-np.sin(phi), np.cos(phi), 0.0])
        e2 = np.cross(nrm, e1)
        rel = pts - src
        t = sdd / (float(src @ nrm) - (pts @ nrm))
        hit = src + t[..., None] * rel
        out[0, v] = (hit @ e1) / det_pixel + det_size / 2.0
        out[1, v] = (hit @ e2) / det_pixel + det_size / 2.0
    return out

def cl_forward_project(volume: "np.ndarray", coords: "np.ndarray",
                               det_size: int) -> "np.ndarray":
    """System matrix A: bilinear deposition of every voxel at its perspective image."""
    import numpy as np
    f = np.asarray(volume, dtype=np.float64)
    c = np.asarray(coords, dtype=np.float64)
    n_views = c.shape[1]
    det = int(det_size)
    val = f.transpose(2, 1, 0)
    out = np.zeros((n_views, det, det), dtype=np.float64)
    for v in range(n_views):
        u, w = c[0, v], c[1, v]
        u0 = np.floor(u).astype(np.int64)
        w0 = np.floor(w).astype(np.int64)
        du, dw = u - u0, w - w0
        acc = np.zeros(det * det, dtype=np.float64)
        for ow, ou, wt in ((0, 0, (1.0 - du) * (1.0 - dw)), (0, 1, du * (1.0 - dw)),
                           (1, 0, (1.0 - du) * dw), (1, 1, du * dw)):
            iu = np.clip(u0 + ou, 0, det - 1)
            iw = np.clip(w0 + ow, 0, det - 1)
            np.add.at(acc, (iw * det + iu).ravel(), (val * wt).ravel())
        out[v] = acc.reshape(det, det)
    return out

def cl_back_project(projections: "np.ndarray", coords: "np.ndarray",
                            shape: "tuple") -> "np.ndarray":
    """Exact discrete transpose of step 2: the same bilinear weights, gathered."""
    import numpy as np
    g = np.asarray(projections, dtype=np.float64)
    c = np.asarray(coords, dtype=np.float64)
    nx, ny, nz = (int(s) for s in shape)
    det = g.shape[1]
    out = np.zeros((nz, ny, nx), dtype=np.float64)
    for v in range(c.shape[1]):
        u, w = c[0, v], c[1, v]
        u0 = np.floor(u).astype(np.int64)
        w0 = np.floor(w).astype(np.int64)
        du, dw = u - u0, w - w0
        for ow, ou, wt in ((0, 0, (1.0 - du) * (1.0 - dw)), (0, 1, du * (1.0 - dw)),
                           (1, 0, (1.0 - du) * dw), (1, 1, du * dw)):
            iu = np.clip(u0 + ou, 0, det - 1)
            iw = np.clip(w0 + ow, 0, det - 1)
            out += wt * g[v][iw, iu]
    return out.transpose(2, 1, 0)

def anisotropic_gradient(volume: "np.ndarray", axis: int,
                                 adjoint: bool = False) -> "np.ndarray":
    """Forward difference with periodic wrap along axis, or its exact transpose."""
    import numpy as np
    f = np.asarray(volume, dtype=np.float64)
    if int(axis) not in (0, 1, 2):
        raise ValueError("axis must be 0, 1 or 2")
    if adjoint:
        return np.roll(f, 1, axis=int(axis)) - f
    return np.roll(f, -1, axis=int(axis)) - f

def tensor_svt(tensor: "np.ndarray", threshold: float) -> "np.ndarray":
    """Algorithm 1: t-SVT. FFT along mode 3, SVT on the first half of the slices, conjugate rest."""
    import numpy as np
    Y = np.asarray(tensor, dtype=np.float64)
    if threshold < 0.0:
        raise ValueError("threshold must be non-negative")
    n3 = Y.shape[2]
    Yf = np.fft.fft(Y, axis=2)
    W = np.zeros_like(Yf)
    half = int(np.ceil((n3 + 1) / 2.0))
    for j in range(half):
        U, S, Vh = np.linalg.svd(Yf[:, :, j], full_matrices=False)
        W[:, :, j] = (U * np.maximum(S - threshold, 0.0)) @ Vh
    for j in range(half, n3):
        W[:, :, j] = np.conj(W[:, :, n3 - j])
    return np.real(np.fft.ifft(W, axis=2))

def operator_norm(coords: "np.ndarray", shape: "tuple", det_size: int,
                          n_power: int = 50) -> float:
    """L = ||K||_2 for K = [A; grad_x; grad_y; grad_z], by power iteration from the ones vector."""
    import numpy as np
    nx, ny, nz = (int(s) for s in shape)
    x = np.ones((nx, ny, nz), dtype=np.float64)
    norm = 0.0
    for _ in range(int(n_power)):
        y = cl_forward_project(x, coords, det_size)
        acc = cl_back_project(y, coords, shape)
        for axis in (0, 1, 2):
            acc = acc + anisotropic_gradient(
                anisotropic_gradient(x, axis), axis, adjoint=True)
        norm = float(np.linalg.norm(acc))
        x = acc / norm
    return float(np.sqrt(norm))

def agslr_cp_update(state: "tuple", projections: "np.ndarray", coords: "np.ndarray",
                            det_size: int, lambdas: "tuple", tau: float, sigma: float,
                            gamma: float = 1.0) -> tuple:
    """One iteration of Algorithm 3: dual ascent, t-SVT proximal step, printed extrapolation."""
    import numpy as np
    x, xbar, y, p, q, r = (np.asarray(a, dtype=np.float64) for a in state)
    g = np.asarray(projections, dtype=np.float64)
    l1, l2, l3, l4 = (float(v) for v in lambdas)
    if min(l1, l2, l3) <= 0.0:
        raise ValueError("lambda1, lambda2 and lambda3 must be positive")
    y_new = (y + sigma * (cl_forward_project(xbar, coords, det_size) - g)) / (1.0 + sigma)
    duals = []
    for dual, lam, axis in ((p, l1, 0), (q, l2, 1), (r, l3, 2)):
        cand = dual + sigma * anisotropic_gradient(xbar, axis)
        duals.append(lam * cand / np.maximum(lam, np.abs(cand)))
    p_new, q_new, r_new = duals
    acc = cl_back_project(y_new, coords, x.shape)
    for dual, axis in ((p_new, 0), (q_new, 1), (r_new, 2)):
        acc = acc + anisotropic_gradient(dual, axis, adjoint=True)
    x_new = tensor_svt(x - tau * acc, tau * l4)
    xbar_new = x + gamma * (x_new - x)
    return x_new, xbar_new, y_new, p_new, q_new, r_new

def run_agslr_cp(projections: "np.ndarray", shape: "tuple", n_views: int,
                         theta_deg: float, sod: float, sdd: float, voxel_size: float,
                         det_pixel: float, det_size: int, lambdas: "tuple",
                         n_iter: int = 30, n_power: int = 50) -> "np.ndarray":
    """Reconstruct from f = 0: geometry, operator norm, then n_iter AGSLR-CP iterations."""
    import numpy as np
    g = np.asarray(projections, dtype=np.float64)
    coords = cl_detector_coords(shape, n_views, theta_deg, sod, sdd, voxel_size,
                                        det_pixel, det_size)
    L = operator_norm(coords, shape, det_size, n_power)
    tau = sigma = 1.0 / L
    nx, ny, nz = (int(s) for s in shape)
    zero = np.zeros((nx, ny, nz), dtype=np.float64)
    state = (zero, zero.copy(), np.zeros_like(g), zero.copy(), zero.copy(), zero.copy())
    for _ in range(int(n_iter)):
        state = agslr_cp_update(state, g, coords, det_size, lambdas, tau, sigma, 1.0)
    return state[0]
SCICODE_GOLD_EOF
