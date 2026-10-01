"""
Trace real rays through the sequential surface table. Return, for every ray, its intersection with the last surface, its unit direction after refraction there, the optical path length from its origin, and a validity flag. Solve aspheric intersections numerically to full double precision. A ray that misses a surface (no real intersection ahead of it), lands outside a semi-aperture or is totally internally reflected is invalid from then on, and its position, direction and path are NaN. Input directions need not be normalised; plane surfaces (curvature 0) and a non-air object medium must work.

The optical path ∫n ds becomes the wave's phase, so it must be accurate far below a wavelength: at 550 nm a path error of 10⁻¹² mm is already 10⁻⁸ rad.

Returns
-------
return positions, directions_out, opl, valid
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def trace_sequential_rays(origins: "np.ndarray", directions: "np.ndarray", surfaces: "np.ndarray",
                          n_object: float) -> tuple:
    """Trace rays through sequential rotationally symmetric aspheric surfaces.

    Args:
        origins: float (N, 3), start points [x, y, z] in mm, in the object medium before surface 1.
        directions: float (N, 3), ray directions (any length).
        surfaces: float (M, 7), rows [z_vertex, curvature, conic, a4, a6, n_after, semi_aperture]
            along +z (mm, 1/mm, -, 1/mm^3, 1/mm^5, -, mm).
        n_object: refractive index of the starting medium.

    Returns:
        tuple (positions (N, 3) at the last surface in mm, directions_out (N, 3) unit, opl (N,)
        optical path from the origin in mm, valid (N,) bool); invalid rows are NaN.
    """
    return positions, directions_out, opl, valid

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_trace_sequential_rays(origins: "np.ndarray", directions: "np.ndarray", surfaces: "np.ndarray",
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

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    c = """import numpy as np
def disk(n):
    i = np.arange(n) + 0.5
    r = np.sqrt(i / n); a = np.pi * (3 - np.sqrt(5)) * i
    return np.column_stack([r * np.cos(a), r * np.sin(a)])
def pack(out):
    pos, d, opl, ok = out
    ok = np.asarray(ok, dtype=bool)
    return np.concatenate([np.asarray(pos)[ok].ravel(), np.asarray(d)[ok].ravel(),
                           np.asarray(opl)[ok], ok.astype(float)])
ASPH = np.array([[0.0, 0.4, -0.8, 2e-3, -1e-4, 1.6, 1.2],
                 [1.2, -0.1, 1.5, -3e-3, 2e-4, 1.0, 1.5]])
"""
    return [
        {"setup": c + """plate = np.array([[0.0, 0.0, 0, 0, 0, 1.5, 5.0], [2.0, 0.0, 0, 0, 0, 1.0, 5.0]])
th = np.radians(25.0); tt = np.arcsin(np.sin(th) / 1.5)
o = np.array([[0.0, 0.0, -1.0]]); d = np.array([[2 * np.sin(th), 0.0, 2 * np.cos(th)]])
expected = np.concatenate([[np.tan(th) + 2 * np.tan(tt), 0.0, 2.0],
                           [np.sin(th), 0.0, np.cos(th)],
                           [1 / np.cos(th) + 1.5 * 2 / np.cos(tt)], [1.0]])
def check(out):
    got = pack(out)
    return bool(got.shape == expected.shape and np.allclose(got, expected, rtol=0, atol=1e-12))
""",  # 1.1
         "call": 'check(trace_sequential_rays(o, d, plate, 1.0))',
         "gold_call": 'check(_oracle_trace_sequential_rays(o, d, plate, 1.0))'},
        {"setup": c + """concave = np.array([[0.0, -0.5, 0, 0, 0, 1.7, 3.0]])
rng = np.random.default_rng(0)
d = np.column_stack([rng.uniform(-0.4, 0.4, (20, 2)), np.ones(20)])
d /= np.linalg.norm(d, axis=1, keepdims=True)
o = np.tile([0.0, 0.0, -2.0], (20, 1))
def check(out):
    _, dirs, opl, ok = out
    return bool(np.all(ok) and np.allclose(dirs, d, rtol=0, atol=1e-12)
                and np.allclose(opl, 2.0, rtol=0, atol=1e-12))
""",  # 1.2
         "call": 'check(trace_sequential_rays(o, d, concave, 1.0))',
         "gold_call": 'check(_oracle_trace_sequential_rays(o, d, concave, 1.0))'},
        {"setup": c + """n2, Rc = 1.5, 2.0
ell = np.array([[0.0, 1 / Rc, -(1 / n2) ** 2, 0, 0, n2, 1.5]])
F = np.array([0.0, 0.0, n2 * Rc / (n2 - 1)])
o = np.column_stack([disk(200) * 1.2, np.full(200, -1.0)])
d = np.tile([0.0, 0.0, 1.0], (200, 1))
def check(out):
    pos, dirs, opl, ok = out
    to_f = F - pos
    u = to_f / np.linalg.norm(to_f, axis=1, keepdims=True)
    cross = np.linalg.norm(np.cross(dirs, u), axis=1)
    tot = opl + n2 * np.linalg.norm(to_f, axis=1)
    return bool(np.all(ok)) and cross.max() < 1e-10 and np.ptp(tot) < 1e-10
""",  # 1.3
         "call": 'check(trace_sequential_rays(o, d, ell, 1.0))',
         "gold_call": 'check(_oracle_trace_sequential_rays(o, d, ell, 1.0))'},
        {"setup": c + """o = np.column_stack([disk(300) * 1.4, np.full(300, -0.5)])
d = np.column_stack([np.full(300, 0.15), np.full(300, -0.25), np.ones(300)])
""",  # 1.4
         "call": 'pack(trace_sequential_rays(o, d, ASPH, 1.0))',
         "gold_call": 'pack(_oracle_trace_sequential_rays(o, d, ASPH, 1.0))'},
        {"setup": c + """S = np.array([[0.0, -0.3, 0.5, 0, 0, 1.8, 3.0], [0.6, 0.0, 0, 0, 0, 1.0, 3.0]])
ang = np.radians(np.linspace(-40, 40, 33))
d = np.column_stack([np.zeros(33), np.sin(ang), np.cos(ang)])
o = np.tile([0.0, 0.1, -0.4], (33, 1))
""",  # 1.5
         "call": 'pack(trace_sequential_rays(o, d, S, 1.33))',
         "gold_call": 'pack(_oracle_trace_sequential_rays(o, d, S, 1.33))'},
        {"setup": c + """origins = np.array([[0.0, 0.0, -2.0]])
directions = np.array([[1.0, 0.0, 1.0]])
surfaces = np.array([[0.0, 1.0, -1.0, 0.0, 0.0, 1.5, 3.0]])
def check_miss(out):
    positions, directions_out, opl, valid = map(np.asarray, out)
    return bool(positions.shape == (1, 3) and directions_out.shape == (1, 3)
                and opl.shape == (1,) and valid.shape == (1,)
                and not valid[0] and np.isnan(positions).all()
                and np.isnan(directions_out).all() and np.isnan(opl).all())
""",  # 1.6
         "call": 'check_miss(trace_sequential_rays(origins, directions, surfaces, 1.0))',
         "gold_call": 'check_miss(_oracle_trace_sequential_rays(origins, directions, surfaces, 1.0))'},
    ]
