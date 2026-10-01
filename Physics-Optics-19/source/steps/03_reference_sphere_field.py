"""
For one field direction, return the complex field the lens delivers onto the exit-pupil reference sphere. That sphere passes through the paraxial exit-pupil centre of Step 2, and its radius is the distance from its centre to that point. Rays start at the stop points stop_radius·sample at z = stop_z, all travelling along the field direction, and are traced with Step 1; each is then continued in the last medium until it meets that sphere on the cap containing the exit-pupil centre, which may lie behind the last surface. Return each ray's point on the sphere and its complex amplitude, with phase increasing with optical path and the chief ray's own value on the sphere as the zero of phase: amplitude 1 for a ray that reaches the sphere, 0 for a ray lost anywhere, whose point is set to the exit-pupil centre. Also return unit sphere normals pointing to the centre, the centre and the radius. Must call trace_sequential_rays and paraxial_exit_pupil by name; raise ValueError if the chief ray is lost.

How many samples a propagation needs is set by how fast the sampled quantity varies across the pupil, not by the pupil's size.

Returns
-------
return points, field, normals, center, radius
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def reference_sphere_field(surfaces: "np.ndarray", n_object: float, stop_z: float, stop_radius: float,
                           tan_field: tuple, pupil_samples: "np.ndarray", wavelength: float,
                           sensor_z: float) -> tuple:
    """Complex field on the exit-pupil reference sphere for one field direction.

    Args:
        surfaces: float (M, 7) surface table (see Step 1).
        n_object: object-medium index (contains the stop).
        stop_z, stop_radius: stop plane position and radius, mm.
        tan_field: (tan_theta_x, tan_theta_y); rays travel along (tx, ty, 1) normalised.
        pupil_samples: float (N, 2), stop coordinates (x, y) in units of the stop radius; column 0 is x.
        wavelength: vacuum wavelength, mm.
        sensor_z: sensor plane position, mm (in the last medium).

    Returns:
        tuple (points (N, 3) on the sphere in mm, lost rays at the exit-pupil centre;
        field (N,) complex, amplitude*exp(1j*phase), chief phase 0, lost rays 0;
        normals (N, 3) unit, toward the centre; center (3,) sphere centre in mm;
        radius float in mm).

    Raises:
        ValueError: if the chief ray does not reach the sensor.
    """
    return points, field, normals, center, radius

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_reference_sphere_field(surfaces: "np.ndarray", n_object: float, stop_z: float, stop_radius: float,
                                   tan_field: tuple, pupil_samples: "np.ndarray", wavelength: float,
                                   sensor_z: float) -> tuple:
    """Complex field on the reference sphere for one field direction."""
    import numpy as np
    S = np.asarray(surfaces, dtype=np.float64).reshape(-1, 7)
    tx, ty = float(tan_field[0]), float(tan_field[1])
    d0 = np.array([tx, ty, 1.0]) / np.sqrt(tx * tx + ty * ty + 1.0)
    n_img = S[-1, 5]
    cp, cd, copl, cval = _oracle_trace_sequential_rays(np.array([[0.0, 0.0, stop_z]]), d0[None, :], S, n_object)
    if not cval[0]:
        raise ValueError("chief ray is vignetted")
    tc = (sensor_z - cp[0, 2]) / cd[0, 2]
    P = cp[0] + tc * cd[0]
    z_xp, _ = _oracle_paraxial_exit_pupil(S, n_object, stop_z, stop_radius)
    E = np.array([0.0, 0.0, z_xp])
    R = float(np.linalg.norm(P - E))
    wc = cp[0] - P
    bc = wc @ cd[0]
    t_chief = -bc - np.sqrt(bc * bc - (wc @ wc - R * R))
    delta_chief = copl[0] + n_img * t_chief
    s = np.asarray(pupil_samples, dtype=np.float64).reshape(-1, 2) * stop_radius
    starts = np.column_stack([s, np.full(len(s), float(stop_z))])
    ref = n_object * (starts @ d0 - stop_z * d0[2])
    pos, dirs, opl, val = _oracle_trace_sequential_rays(starts, np.tile(d0, (len(s), 1)), S, n_object)
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

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    c = """import numpy as np
LAM = 5.5e-4
SINGLET = np.array([[0.0, 1 / 3.0, 0, 0, 0, 1.5, 2.0], [1.0, -1 / 8.0, 0, 0, 0, 1.0, 2.0]])
def disk(n):
    i = np.arange(n) + 0.5
    r = np.sqrt(i / n); a = np.pi * (3 - np.sqrt(5)) * i
    return np.column_stack([r * np.cos(a), r * np.sin(a)])
def cplx(f):
    f = np.asarray(f, dtype=complex)
    return np.concatenate([f.real, f.imag])
def packed(out):
    return np.concatenate([np.ravel(out[0]), np.ravel(out[2]), cplx(out[1])])
t20 = np.tan(np.radians(20.0))
"""
    return [
        {"setup": c + """ell = np.array([[0.0, 0.5, -(1 / 1.5) ** 2, 0, 0, 1.5, 1.5]])
args = (ell, 1.0, -0.5, 0.3, (0.0, 0.0), disk(400), LAM, 6.0)
def check(out):
    pts, f, c, R = np.asarray(out[0]), np.asarray(out[1]), np.asarray(out[3]), float(out[4])
    return bool(np.max(np.abs(np.angle(f))) < 1e-8 and np.allclose(np.abs(f), 1.0)
                and np.allclose(c, [0.0, 0.0, 6.0], atol=1e-12)
                and np.allclose(np.linalg.norm(pts - c, axis=1), R, atol=1e-10))
""",  # 3.1
         "call": 'check(reference_sphere_field(*args))',
         "gold_call": 'check(_oracle_reference_sphere_field(*args))'},
        {"setup": c + """args = (SINGLET, 1.0, -0.2, 0.1, (0.0, t20), disk(50), LAM, 5.0)
def centre_radius(out):
    return np.concatenate([np.asarray(out[3], dtype=float), [float(out[4])]])
""",  # 3.2
         "call": 'centre_radius(reference_sphere_field(*args))',
         "gold_call": 'centre_radius(_oracle_reference_sphere_field(*args))'},
        {"setup": c + """args = (SINGLET, 1.0, -1.0, 0.2, (0.12, -0.2), disk(250), LAM, 5.4)
""",  # 3.4
         "call": 'packed(reference_sphere_field(*args))',
         "gold_call": 'packed(_oracle_reference_sphere_field(*args))'},
        {"setup": c + """S = SINGLET.copy(); S[0, 6] = 0.08
args = (S, 1.0, -0.2, 0.1, (0.0, t20), disk(300), LAM, 5.0)
""",  # 3.5
         "call": 'packed(reference_sphere_field(*args))',
         "gold_call": 'packed(_oracle_reference_sphere_field(*args))'},
        {"setup": c + """S = np.array([[0.0, 0.4, -0.8, 2e-3, -1e-4, 1.6, 1.2], [1.2, -0.1, 1.5, -3e-3, 2e-4, 1.45, 1.5]])
args = (S, 1.2, -0.3, 0.25, (0.05, 0.08), disk(200), 4.5e-4, 9.0)
""",  # 3.6
         "call": 'cplx(reference_sphere_field(*args)[1])',
         "gold_call": 'cplx(_oracle_reference_sphere_field(*args)[1])'},
    ]
