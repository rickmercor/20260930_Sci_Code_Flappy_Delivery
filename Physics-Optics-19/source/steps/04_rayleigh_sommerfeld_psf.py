"""
Propagate the sampled field to a rectangular grid of sensor points with the first Rayleigh–Sommerfeld integral evaluated as a Monte-Carlo sum over the samples, and return the intensity. The obliquity factor uses the given normal at each sample; the kernel keeps the exact point-to-point distance (no paraxial or far-field simplification); every sample carries the same weight (equal-area stop samples, no Jacobian). Normalisation (task convention; the source prints a different prefactor): for N equal-phase samples of nonnegative amplitude a_i on a sphere of radius R, the intensity at the sphere centre is (mean a_i)²/(λR)², so repeating every sample changes nothing. The sensor is in air; the field follows the Step 3 phase convention; the output is indexed [pixel_y, pixel_x].

A Fourier propagator needs a pupil grid fine enough for the local fringe frequency, which grows quickly with defocus and field; a set of scattered samples puts no grid on the pupil at all.

Returns
-------
return intensity
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def rayleigh_sommerfeld_psf(points: "np.ndarray", field: "np.ndarray", normals: "np.ndarray",
                            pixel_x: "np.ndarray", pixel_y: "np.ndarray", sensor_z: float,
                            wavelength: float) -> "np.ndarray":
    """Intensity on a sensor grid from scattered samples of a complex field.

    Args:
        points: float (N, 3), sample positions, mm.
        field: complex (N,), field at the samples.
        normals: float (N, 3), unit source-surface normals toward the sensor side.
        pixel_x: float (W,), sensor x coordinates, mm.
        pixel_y: float (H,), sensor y coordinates, mm.
        sensor_z: sensor plane position, mm.
        wavelength: vacuum wavelength, mm.

    Returns:
        float (H, W), intensity at (pixel_y[row], pixel_x[col], sensor_z).
    """
    return intensity

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_rayleigh_sommerfeld_psf(points: "np.ndarray", field: "np.ndarray", normals: "np.ndarray",
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

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    c = """import numpy as np
LAM = 5.5e-4
def cap(n, R, P, half_angle, axis=(0.0, 0.0, 1.0)):
    i = np.arange(n) + 0.5
    ct = 1 - (i / n) * (1 - np.cos(half_angle)); st = np.sqrt(1 - ct * ct)
    ph = np.pi * (3 - np.sqrt(5)) * i
    loc = np.column_stack([st * np.cos(ph), st * np.sin(ph), ct])
    a = np.asarray(axis, float) / np.linalg.norm(axis)
    v = np.cross([0.0, 0.0, 1.0], a); s = np.linalg.norm(v); c = a[2]
    Rm = np.eye(3)
    if s > 1e-15:
        vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        Rm = np.eye(3) + vx + vx @ vx * ((1 - c) / s ** 2)
    u = loc @ Rm.T
    return np.asarray(P, float) - R * u, u
"""
    return [
        {"setup": c + """P = np.array([0.1, -0.05, 4.0]); R = 3.5
pts, un = cap(500, R, P, 0.05)
amp = np.random.default_rng(3).uniform(0.2, 1.0, 500).astype(complex)
args = (pts, amp, un, np.array([P[0]]), np.array([P[1]]), P[2], LAM)
def check(h):
    h = np.asarray(h, dtype=float)
    return bool(h.shape == (1, 1) and abs(h[0, 0] * (LAM * R) ** 2 - amp.real.mean() ** 2) < 1e-12)
""",  # 4.1
         "call": 'check(rayleigh_sommerfeld_psf(*args))',
         "gold_call": 'check(_oracle_rayleigh_sommerfeld_psf(*args))'},
        {"setup": c + """R = 3.5; na = 0.02
pts, un = cap(20000, R, [0.0, 0.0, R], na)
r_airy = 0.61 * LAM / np.sin(na)
xs = np.linspace(0.0, 1.3 * r_airy, 131)
args = (pts, np.ones(20000, complex), un, xs, np.array([0.0]), R, LAM)
def check(h):
    line = np.asarray(h, dtype=float)[0, :120]
    return bool(abs(xs[np.argmin(line)] / r_airy - 1.0) < 0.02)
""",  # 4.2
         "call": 'check(rayleigh_sommerfeld_psf(*args))',
         "gold_call": 'check(_oracle_rayleigh_sommerfeld_psf(*args))'},
        {"setup": c + """pts, un = cap(400, 2.0, [0.3, 0.2, 3.0], 0.35, axis=(0.3, 0.2, 1.0))
v = np.exp(1j * np.random.default_rng(5).uniform(0, 0.8, 400))
args = (pts, v, un, np.linspace(0.2, 0.4, 9), np.linspace(0.1, 0.3, 7), 2.6, LAM)
""",  # 4.4
         "call": '(LAM * 2.0) ** 2 * rayleigh_sommerfeld_psf(*args)',
         "gold_call": '(LAM * 2.0) ** 2 * _oracle_rayleigh_sommerfeld_psf(*args)'},
        {"setup": c + """pts, un = cap(600, 5.0, [0.0, 0.0, 5.0], 0.25)
rho2 = (pts[:, 0] ** 2 + pts[:, 1] ** 2) / 1.25 ** 2
v = np.exp(1j * 2 * np.pi * 3.0 * (rho2 ** 2 + 0.4 * pts[:, 1] / 1.25))
args = (pts, v, un, np.linspace(-0.3, 0.3, 11), np.linspace(-0.2, 0.35, 12), 7.0, LAM)
""",  # 4.5
         "call": '1e2 * (LAM * 5.0) ** 2 * rayleigh_sommerfeld_psf(*args)',
         "gold_call": '1e2 * (LAM * 5.0) ** 2 * _oracle_rayleigh_sommerfeld_psf(*args)'},
    ]
