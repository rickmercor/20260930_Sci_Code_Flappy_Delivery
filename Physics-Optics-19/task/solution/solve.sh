#!/bin/bash
# Oracle: writes the gold solution so the verifier scores reward 1.0.
cat > /app/solution.py <<'SCICODE_GOLD_EOF'
import numpy as np
try:
    import scipy, scipy.linalg, scipy.sparse, scipy.optimize, scipy.spatial, scipy.integrate, scipy.stats, scipy.special, scipy.signal
except Exception:
    pass
import math

def trace_sequential_rays(origins: "np.ndarray", directions: "np.ndarray", surfaces: "np.ndarray",
                                  n_object: float) -> tuple:
    """Real 3D ray trace through a sequence of rotationally symmetric aspheric surfaces."""
    import numpy as np
    o = np.array(origins, dtype=np.float64).reshape(-1, 3)
    d = np.array(directions, dtype=np.float64).reshape(-1, 3)
    d = d / np.linalg.norm(d, axis=1, keepdims=True)
    S = np.asarray(surfaces, dtype=np.float64).reshape(-1, 7)
    n_rays = o.shape[0]
    opl = np.zeros(n_rays)
    valid = np.ones(n_rays, dtype=bool)
    n1 = float(n_object)
    for zv, c, kap, a4, a6, n2, semi in S:
        with np.errstate(all="ignore"):
            t = (zv - o[:, 2]) / d[:, 2]
            for _ in range(100):
                p = o + t[:, None] * d
                r2 = p[:, 0] ** 2 + p[:, 1] ** 2
                arg = 1.0 - (1.0 + kap) * c * c * r2
                root = np.sqrt(arg)
                sag = c * r2 / (1.0 + root) + a4 * r2 ** 2 + a6 * r2 ** 3
                dsdr2 = 0.5 * c / root + 2.0 * a4 * r2 + 3.0 * a6 * r2 ** 2
                F = p[:, 2] - zv - sag
                dF = d[:, 2] - dsdr2 * 2.0 * (p[:, 0] * d[:, 0] + p[:, 1] * d[:, 1])
                step = F / dF
                t = t - step
                if np.all(~np.isfinite(step) | (np.abs(step) < 1e-15)):
                    break
            p = o + t[:, None] * d
            r2 = p[:, 0] ** 2 + p[:, 1] ** 2
            arg = 1.0 - (1.0 + kap) * c * c * r2
            root = np.sqrt(arg)
            dsdr2 = 0.5 * c / root + 2.0 * a4 * r2 + 3.0 * a6 * r2 ** 2
            nrm = np.stack([-2.0 * dsdr2 * p[:, 0], -2.0 * dsdr2 * p[:, 1], np.ones(n_rays)], axis=1)
            nrm /= np.linalg.norm(nrm, axis=1, keepdims=True)
            sag_final = c * r2 / (1.0 + root) + a4 * r2 ** 2 + a6 * r2 ** 3
            residual = p[:, 2] - zv - sag_final
            residual_scale = 1.0 + np.abs(p[:, 2]) + abs(zv) + np.abs(sag_final)
            on_surface = np.isfinite(residual) & (np.abs(residual) <= 32.0 * np.finfo(np.float64).eps * residual_scale)
            ok = np.isfinite(t) & on_surface & (arg >= 0.0) & (t > 0.0) & (r2 <= semi * semi)
            cos_i = -np.sum(nrm * d, axis=1)
            nrm = np.where(cos_i[:, None] < 0.0, -nrm, nrm)
            cos_i = np.abs(cos_i)
            mu = n1 / n2
            rad = 1.0 - mu * mu * (1.0 - cos_i * cos_i)
            ok &= rad >= 0.0
            cos_t = np.sqrt(np.maximum(rad, 0.0))
            d_new = mu * d + (mu * cos_i - cos_t)[:, None] * nrm
            d_new /= np.linalg.norm(d_new, axis=1, keepdims=True)
        valid &= ok
        opl = opl + n1 * t
        o, d = p, d_new
        n1 = n2
    o[~valid] = np.nan
    d[~valid] = np.nan
    opl[~valid] = np.nan
    return o, d, opl, valid

def paraxial_exit_pupil(surfaces: "np.ndarray", n_object: float, stop_z: float,
                                stop_radius: float) -> tuple:
    """Paraxial image of an object-side aperture stop through all surfaces."""
    import numpy as np
    S = np.asarray(surfaces, dtype=np.float64).reshape(-1, 7)
    M = np.eye(2)
    z_prev, n_prev = float(stop_z), float(n_object)
    for zv, c, _k, _a4, _a6, n2, _semi in S:
        M = np.array([[1.0, (zv - z_prev) / n_prev], [0.0, 1.0]]) @ M
        M = np.array([[1.0, 0.0], [-(n2 - n_prev) * c, 1.0]]) @ M
        z_prev, n_prev = zv, n2
    B = M[0, 1]
    D = M[1, 1]
    t = -n_prev * B / D
    return float(z_prev + t), float(abs(stop_radius / D))

def reference_sphere_field(surfaces: "np.ndarray", n_object: float, stop_z: float, stop_radius: float,
                                   tan_field: tuple, pupil_samples: "np.ndarray", wavelength: float,
                                   sensor_z: float) -> tuple:
    """Complex field on the reference sphere for one field direction."""
    import numpy as np
    S = np.asarray(surfaces, dtype=np.float64).reshape(-1, 7)
    tx, ty = float(tan_field[0]), float(tan_field[1])
    d0 = np.array([tx, ty, 1.0]) / np.sqrt(tx * tx + ty * ty + 1.0)
    n_img = S[-1, 5]
    cp, cd, copl, cval = trace_sequential_rays(np.array([[0.0, 0.0, stop_z]]), d0[None, :], S, n_object)
    if not cval[0]:
        raise ValueError("chief ray is vignetted")
    tc = (sensor_z - cp[0, 2]) / cd[0, 2]
    P = cp[0] + tc * cd[0]
    z_xp, _ = paraxial_exit_pupil(S, n_object, stop_z, stop_radius)
    E = np.array([0.0, 0.0, z_xp])
    R = float(np.linalg.norm(P - E))
    wc = cp[0] - P
    bc = wc @ cd[0]
    t_chief = -bc - np.sqrt(bc * bc - (wc @ wc - R * R))
    delta_chief = copl[0] + n_img * t_chief
    s = np.asarray(pupil_samples, dtype=np.float64).reshape(-1, 2) * stop_radius
    starts = np.column_stack([s, np.full(len(s), float(stop_z))])
    ref = n_object * (starts @ d0 - stop_z * d0[2])
    pos, dirs, opl, val = trace_sequential_rays(starts, np.tile(d0, (len(s), 1)), S, n_object)
    with np.errstate(invalid="ignore"):
        w = pos - P
        b = np.sum(w * dirs, axis=1)
        disc = b * b - (np.sum(w * w, axis=1) - R * R)
        t = -b - np.sqrt(disc)
        rho = pos + t[:, None] * dirs
        delta = opl + ref + n_img * t
    ok = val & (disc >= 0.0)
    k = 2.0 * np.pi / wavelength
    field = np.zeros(len(s), dtype=np.complex128)
    field[ok] = np.exp(1j * k * (delta[ok] - delta_chief))
    rho[~ok] = E
    normals = (P - rho) / R
    return rho, field, normals, P, R

def rayleigh_sommerfeld_psf(points: "np.ndarray", field: "np.ndarray", normals: "np.ndarray",
                                    pixel_x: "np.ndarray", pixel_y: "np.ndarray", sensor_z: float,
                                    wavelength: float) -> "np.ndarray":
    """Monte-Carlo Rayleigh-Sommerfeld intensity on a sensor grid (sensor in air)."""
    import numpy as np
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    v = np.asarray(field, dtype=np.complex128).reshape(-1)
    nrm = np.asarray(normals, dtype=np.float64).reshape(-1, 3)
    px = np.asarray(pixel_x, dtype=np.float64).reshape(-1)
    py = np.asarray(pixel_y, dtype=np.float64).reshape(-1)
    k = 2.0 * np.pi / wavelength
    N = len(v)
    out = np.empty((len(py), len(px)))
    for i, y in enumerate(py):
        rx = px[:, None] - pts[None, :, 0]
        ry = y - pts[None, :, 1]
        rz = sensor_z - pts[None, :, 2]
        r = np.sqrt(rx * rx + ry * ry + rz * rz)
        cos_t = (rx * nrm[None, :, 0] + ry * nrm[None, :, 1] + rz * nrm[None, :, 2]) / r
        U = np.sum(v[None, :] * np.exp(1j * k * r) / r * cos_t, axis=1)
        out[i] = np.abs(U) ** 2
    return out / (N * wavelength) ** 2

def chief_ray_landing(scene_shape: tuple, tan_bounds: tuple, surfaces: "np.ndarray", n_object: float,
                              stop_z: float, sensor_z: float) -> tuple:
    """Field-direction grid of the scene and the real chief-ray landing point of each sample."""
    import numpy as np
    hs, ws = int(scene_shape[0]), int(scene_shape[1])
    tx = np.linspace(tan_bounds[0], tan_bounds[1], ws)
    ty = np.linspace(tan_bounds[2], tan_bounds[3], hs)
    TX, TY = np.meshgrid(tx, ty)
    tan_grid = np.stack([TX, TY], axis=-1)
    d = np.column_stack([TX.ravel(), TY.ravel(), np.ones(hs * ws)])
    o = np.tile([0.0, 0.0, float(stop_z)], (hs * ws, 1))
    pos, dirs, _, _ = trace_sequential_rays(o, d, surfaces, n_object)
    t = (sensor_z - pos[:, 2]) / dirs[:, 2]
    land = pos[:, :2] + t[:, None] * dirs[:, :2]
    return tan_grid, land.reshape(hs, ws, 2)

def weighted_latent_images(scene: "np.ndarray", landing: "np.ndarray", node_rows: list, node_cols: list,
                                   pixel_pitch: float, sensor_shape: tuple,
                                   sensor_center: tuple) -> "np.ndarray":
    """Per-node weighted latent images: scene samples weighted by the node's
    interpolation weight, then deposited at their chief-ray landing pixel."""
    import numpy as np
    b = np.asarray(scene, dtype=np.float64)
    hs, ws = b.shape
    L = np.asarray(landing, dtype=np.float64)
    H, W = int(sensor_shape[0]), int(sensor_shape[1])
    weights_1d = []
    for nodes, n in ((np.asarray(node_rows, dtype=int), hs), (np.asarray(node_cols, dtype=int), ws)):
        if len(nodes) == 1:
            weights_1d.append(np.ones((1, n)))
            continue
        if nodes[0] != 0 or nodes[-1] != n - 1 or np.any(np.diff(nodes) <= 0):
            raise ValueError("nodes must increase strictly from 0 to n-1")
        idx = np.arange(n)
        weights_1d.append(np.stack([np.interp(idx, nodes, np.eye(len(nodes))[g]) for g in range(len(nodes))]))
    wy, wx = weights_1d
    with np.errstate(invalid="ignore"):
        col = np.floor((L[..., 0] - sensor_center[0]) / pixel_pitch + W / 2.0)
        row = np.floor((L[..., 1] - sensor_center[1]) / pixel_pitch + H / 2.0)
    inside = np.isfinite(col) & np.isfinite(row) & (col >= 0) & (col < W) & (row >= 0) & (row < H)
    r_i, c_i = row[inside].astype(int), col[inside].astype(int)
    out = np.zeros((wy.shape[0], wx.shape[0], H, W))
    for gy in range(wy.shape[0]):
        for gx in range(wx.shape[0]):
            vals = (b * wy[gy][:, None] * wx[gx][None, :])[inside]
            np.add.at(out[gy, gx], (r_i, c_i), vals)
    return out

def sum_of_convolutions(weighted: "np.ndarray", kernels: "np.ndarray") -> "np.ndarray":
    """Measurement as the sum over nodes of (weighted latent image) * (unit-sum kernel)."""
    import numpy as np
    from scipy.signal import fftconvolve
    Wt = np.asarray(weighted, dtype=np.float64)
    Kt = np.asarray(kernels, dtype=np.float64)
    gy, gx, H, W = Wt.shape
    K = Kt.shape[-1]
    if K % 2 == 0 or Kt.shape[-2] != K or Kt.shape[:2] != (gy, gx):
        raise ValueError("kernels must have shape (Gy, Gx, K, K) with K odd")
    out = np.zeros((H, W))
    for i in range(gy):
        for j in range(gx):
            h = Kt[i, j] / Kt[i, j].sum()
            out += fftconvolve(Wt[i, j], h, mode="same")
    return out

def render_measurement(scene: "np.ndarray", tan_bounds: tuple, surfaces: "np.ndarray", n_object: float,
                               stop_z: float, stop_radius: float, sensor_z: float, wavelength: float,
                               pixel_pitch: float, sensor_shape: tuple, sensor_center: tuple,
                               node_rows: list, node_cols: list, pupil_samples: "np.ndarray",
                               kernel_size: int) -> "np.ndarray":
    """Orchestrator: scene + lens -> interpolated wave-optics measurement."""
    import numpy as np
    b = np.asarray(scene, dtype=np.float64)
    tan_grid, landing = chief_ray_landing(b.shape, tan_bounds, surfaces, n_object, stop_z, sensor_z)
    offs = (np.arange(kernel_size) - (kernel_size - 1) / 2.0) * pixel_pitch
    kernels = np.zeros((len(node_rows), len(node_cols), kernel_size, kernel_size))
    for gy, r in enumerate(node_rows):
        for gx, c in enumerate(node_cols):
            pts, fld, nrm, P, _R = reference_sphere_field(
                surfaces, n_object, stop_z, stop_radius, tan_grid[r, c], pupil_samples, wavelength, sensor_z)
            kernels[gy, gx] = rayleigh_sommerfeld_psf(
                pts, fld, nrm, P[0] + offs, P[1] + offs, sensor_z, wavelength)
    weighted = weighted_latent_images(b, landing, node_rows, node_cols, pixel_pitch,
                                              sensor_shape, sensor_center)
    return sum_of_convolutions(weighted, kernels)
SCICODE_GOLD_EOF
