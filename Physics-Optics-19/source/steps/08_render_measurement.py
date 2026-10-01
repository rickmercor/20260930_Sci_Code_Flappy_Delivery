"""
Render the measurement of the scene through the lens, from the scene's field-direction bounds to the image on the sensor, using the five earlier steps. One PSF is rendered per node of the node_rows × node_cols sub-grid; every kernel is kernel_size × kernel_size with kernel_size odd, sampled at offsets spaced by the pixel pitch from that node's own sphere centre, with the centre index at zero offset. Must call chief_ray_landing, reference_sphere_field, rayleigh_sommerfeld_psf, weighted_latent_images and sum_of_convolutions by name, without inlined reimplementation.

Rendering every scene point with its own PSF is far too expensive, and a single convolution cannot reproduce an off-axis image.

Returns
-------
return measurement
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def render_measurement(scene: "np.ndarray", tan_bounds: tuple, surfaces: "np.ndarray", n_object: float,
                       stop_z: float, stop_radius: float, sensor_z: float, wavelength: float,
                       pixel_pitch: float, sensor_shape: tuple, sensor_center: tuple,
                       node_rows: list, node_cols: list, pupil_samples: "np.ndarray",
                       kernel_size: int) -> "np.ndarray":
    """Interpolated wave-optics measurement of an incoherent scene.

    Must call chief_ray_landing, reference_sphere_field, rayleigh_sommerfeld_psf,
    weighted_latent_images and sum_of_convolutions by name.

    Args:
        scene: float (Hs, Ws), intensity on the grid defined by tan_bounds (inclusive bounds).
        surfaces, n_object, stop_z, stop_radius, sensor_z, wavelength: lens, stop, sensor plane
            and vacuum wavelength as in Step 3 (mm); the sensor is in air.
        pixel_pitch, sensor_shape, sensor_center: pitch (mm), (H, W) and (x0, y0) (mm).
        node_rows, node_cols: PSF node indices into the scene grid (see Step 6).
        pupil_samples: float (N, 2), normalised stop coordinates (x, y) used for every PSF; column 0 is x.
        kernel_size: odd K, kernel size in pixels.

    Returns:
        float (H, W), measurement.
    """
    return measurement

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_render_measurement(scene: "np.ndarray", tan_bounds: tuple, surfaces: "np.ndarray", n_object: float,
                               stop_z: float, stop_radius: float, sensor_z: float, wavelength: float,
                               pixel_pitch: float, sensor_shape: tuple, sensor_center: tuple,
                               node_rows: list, node_cols: list, pupil_samples: "np.ndarray",
                               kernel_size: int) -> "np.ndarray":
    """Orchestrator: scene + lens -> interpolated wave-optics measurement."""
    import numpy as np
    b = np.asarray(scene, dtype=np.float64)
    tan_grid, landing = _oracle_chief_ray_landing(b.shape, tan_bounds, surfaces, n_object, stop_z, sensor_z)
    offs = (np.arange(kernel_size) - (kernel_size - 1) / 2.0) * pixel_pitch
    kernels = np.zeros((len(node_rows), len(node_cols), kernel_size, kernel_size))
    for gy, r in enumerate(node_rows):
        for gx, c in enumerate(node_cols):
            pts, fld, nrm, P, _R = _oracle_reference_sphere_field(
                surfaces, n_object, stop_z, stop_radius, tan_grid[r, c], pupil_samples, wavelength, sensor_z)
            kernels[gy, gx] = _oracle_rayleigh_sommerfeld_psf(
                pts, fld, nrm, P[0] + offs, P[1] + offs, sensor_z, wavelength)
    weighted = _oracle_weighted_latent_images(b, landing, node_rows, node_cols, pixel_pitch,
                                              sensor_shape, sensor_center)
    return _oracle_sum_of_convolutions(weighted, kernels)

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
TB = (-np.tan(np.radians(2)), np.tan(np.radians(2)), 0.0, np.tan(np.radians(10)))
cfg = dict(tan_bounds=TB, surfaces=SINGLET, n_object=1.0, stop_z=-0.2, stop_radius=0.3, sensor_z=5.0,
           wavelength=LAM, pixel_pitch=8e-3, sensor_shape=(120, 60), sensor_center=(0.0, 0.40),
           node_rows=[0, 16, 32], node_cols=[0, 12], pupil_samples=disk(300), kernel_size=15)
smooth = 1 + 0.5 * np.sin(np.arange(33)[:, None] / 5.0) * np.cos(np.arange(13)[None, :] / 3.0)
pt = np.zeros((33, 13)); pt[16, 12] = 1.0
"""
    return [
        {"setup": c,  # 8.1
         "call": 'np.asarray(render_measurement(pt, **cfg), dtype=float)',
         "gold_call": '_oracle_render_measurement(pt, **cfg)'},
        {"setup": c + """def energy(m, s):
    return float(np.asarray(m, dtype=float).sum() / s.sum())
""",  # 8.2
         "call": 'energy(render_measurement(smooth, **cfg), smooth)',
         "gold_call": 'energy(_oracle_render_measurement(smooth, **cfg), smooth)'},
        {"setup": c + """cfg1 = dict(cfg, node_rows=[20], node_cols=[6])
""",  # 8.4
         "call": 'np.asarray(render_measurement(smooth, **cfg1), dtype=float)',
         "gold_call": '_oracle_render_measurement(smooth, **cfg1)'},
        {"setup": c + """cfg5 = dict(cfg, tan_bounds=(0.05, 0.12, -0.1, 0.0), stop_radius=0.1, sensor_z=5.15,
            sensor_center=(0.43, -0.22), sensor_shape=(70, 60), node_rows=[0, 32], node_cols=[0, 5, 12])
sc5 = np.random.default_rng(9).random((33, 13))
""",  # 8.5
         "call": 'np.asarray(render_measurement(sc5, **cfg5), dtype=float)',
         "gold_call": '_oracle_render_measurement(sc5, **cfg5)'},
    ]
