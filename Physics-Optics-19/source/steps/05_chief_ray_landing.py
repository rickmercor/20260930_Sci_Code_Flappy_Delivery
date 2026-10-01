"""
Build the scene's field-direction grid and map every sample to the sensor with its real chief ray (from the stop centre along the sample's direction, traced with Step 1 and continued to the sensor plane). Column j has tanθx equal to the j-th of Ws equally spaced values from tx_min to tx_max, and row i has tanθy equal to the i-th of Hs values from ty_min to ty_max (inclusive). Must call trace_sequential_rays by name; lost chief rays give NaN landings.

It is the real chief ray, not the paraxial estimate f·tanθ, that carries the lens distortion.

Returns
-------
return tan_grid, landing
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def chief_ray_landing(scene_shape: tuple, tan_bounds: tuple, surfaces: "np.ndarray", n_object: float,
                      stop_z: float, sensor_z: float) -> tuple:
    """Field-direction grid of the scene and the real chief-ray landing of each sample.

    Args:
        scene_shape: (Hs, Ws).
        tan_bounds: (tx_min, tx_max, ty_min, ty_max), inclusive.
        surfaces: float (M, 7) surface table (see Step 1).
        n_object: object-medium index.
        stop_z: stop plane position, mm (chief rays start at (0, 0, stop_z)).
        sensor_z: sensor plane position, mm.

    Returns:
        tuple (tan_grid (Hs, Ws, 2) with [..., 0] = tan_theta_x and [..., 1] = tan_theta_y,
        landing (Hs, Ws, 2) sensor [x, y] in mm, NaN if lost).
    """
    return tan_grid, landing

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_chief_ray_landing(scene_shape: tuple, tan_bounds: tuple, surfaces: "np.ndarray", n_object: float,
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
    pos, dirs, _, _ = _oracle_trace_sequential_rays(o, d, surfaces, n_object)
    t = (sensor_z - pos[:, 2]) / dirs[:, 2]
    land = pos[:, :2] + t[:, None] * dirs[:, :2]
    return tan_grid, land.reshape(hs, ws, 2)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    c = """import numpy as np
SINGLET = np.array([[0.0, 1 / 3.0, 0, 0, 0, 1.5, 2.0], [1.0, -1 / 8.0, 0, 0, 0, 1.0, 2.0]])
efl = 1 / (0.5 * (1 / 3.0 + 1 / 8.0 - 0.5 * 1.0 / (1.5 * 3.0 * 8.0)))
"""
    return [
        {"setup": c + """args = ((5, 3), (-0.1, 0.1, -0.2, 0.3), SINGLET, 1.0, -0.2, 5.0)
expected = np.stack(np.meshgrid(np.linspace(-0.1, 0.1, 3), np.linspace(-0.2, 0.3, 5)), axis=-1)
def check(out):
    g = np.asarray(out[0], dtype=float)
    return bool(g.shape == expected.shape and np.allclose(g, expected, rtol=0, atol=1e-15))
""",  # 5.1
         "call": 'check(chief_ray_landing(*args))',
         "gold_call": 'check(_oracle_chief_ray_landing(*args))'},
        {"setup": c + """args = ((3, 3), (-1e-6, 1e-6, -1e-6, 1e-6), SINGLET, 1.0, -0.2, 5.0)
def check(out):
    L = np.asarray(out[1], dtype=float)
    return bool(abs(L[2, 2, 0] / 1e-6 - efl) < 1e-6 and abs(L[0, 2, 1] / -1e-6 - efl) < 1e-6
                and np.allclose(L[1, 1], 0.0, atol=1e-15))
""",  # 5.2
         "call": 'check(chief_ray_landing(*args))',
         "gold_call": 'check(_oracle_chief_ray_landing(*args))'},
        {"setup": c + """args = ((7, 4), (-0.2, 0.35, 0.1, 0.6), SINGLET, 1.0, -0.2, 5.0)
""",  # 5.3
         "call": 'np.asarray(chief_ray_landing(*args)[1], dtype=float)',
         "gold_call": '_oracle_chief_ray_landing(*args)[1]'},
        {"setup": c + """args = ((2, 2), (-0.1, 0.1, -0.1, 0.1), SINGLET, 1.0, -0.2, 5.0)
rot = lambda p: np.array([-p[1], p[0]])
def check(out):
    L = np.asarray(out[1], dtype=float)
    # grid points: [0,0]=(-t,-t) [0,1]=(t,-t) [1,0]=(-t,t) [1,1]=(t,t); rotation maps
    # (t,-t)->(t,t)->(-t,t)->(-t,-t)->(t,-t)
    pairs = [((0, 1), (1, 1)), ((1, 1), (1, 0)), ((1, 0), (0, 0)), ((0, 0), (0, 1))]
    return bool(all(np.allclose(rot(L[a]), L[b], rtol=0, atol=1e-13) for a, b in pairs)
                and abs(L[1, 1, 0]) > 0.3)
""",  # 5.4
         "call": 'check(chief_ray_landing(*args))',
         "gold_call": 'check(_oracle_chief_ray_landing(*args))'},
        {"setup": c + """S = np.array([[0.0, 0.4, -0.8, 2e-3, -1e-4, 1.6, 1.2], [1.2, -0.1, 1.5, -3e-3, 2e-4, 1.0, 1.5]])
args = ((4, 6), (-0.3, 0.1, -0.25, 0.25), S, 1.33, -0.6, 4.0)
""",  # 5.5
         "call": 'np.asarray(chief_ray_landing(*args)[1], dtype=float)',
         "gold_call": '_oracle_chief_ray_landing(*args)[1]'},
    ]
