"""
Return the axial position and radius of the exit pupil: the paraxial image of the aperture stop formed by all surfaces after it. The position is a signed coordinate on the same axis as the surface vertices and the radius a positive length; the pupil may be virtual, that is, lie before the last surface. Must handle a stop at the first vertex and a non-air object medium.

The exit pupil is the stop's conjugate in image space. This task uses one paraxial pupil for every field direction, so neither returned value depends on the field.

Returns
-------
return z_exit_pupil, exit_pupil_radius
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def paraxial_exit_pupil(surfaces: "np.ndarray", n_object: float, stop_z: float,
                        stop_radius: float) -> tuple:
    """Paraxial exit pupil: image of the aperture stop through every surface after it.

    Args:
        surfaces: float (M, 7) surface table (see Step 1).
        n_object: refractive index of the medium containing the stop.
        stop_z: stop plane position in mm (before surface 1).
        stop_radius: stop radius in mm.

    Returns:
        tuple (z_exit_pupil, exit_pupil_radius) of floats in mm; the radius is positive.
    """
    return z_exit_pupil, exit_pupil_radius

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_paraxial_exit_pupil(surfaces: "np.ndarray", n_object: float, stop_z: float,
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

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    c = """import numpy as np
one = np.array([[0.0, 0.5, 0, 0, 0, 1.5, 2.0]])
def near(expected, atol=1e-13):
    return lambda out: bool(np.allclose(np.array(out, dtype=float), expected, rtol=0, atol=atol))
"""
    return [
        {"setup": c + """sp = 1.5 / (0.5 * 0.5 - 1.0 / 0.5)
mag = (1.0 * sp) / (1.5 * -0.5)
check = near([sp, abs(mag) * 0.3])
""",  # 2.2
         "call": 'check(paraxial_exit_pupil(one, 1.0, -0.5, 0.3))',
         "gold_call": 'check(_oracle_paraxial_exit_pupil(one, 1.0, -0.5, 0.3))'},
        {"setup": c + """S = np.array([[0.0, 1 / 3.0, 0, 0, 0, 1.5, 2.0], [1.0, -1 / 8.0, 0, 0, 0, 1.0, 2.0]])
s_obj = -1.0
s_img = 1.0 / ((1.0 - 1.5) * (-1 / 8.0) + 1.5 / s_obj)
mag = (1.5 * s_img) / (1.0 * s_obj)
check = near([1.0 + s_img, abs(mag) * 0.25])
""",  # 2.3
         "call": 'check(paraxial_exit_pupil(S, 1.0, 0.0, 0.25))',
         "gold_call": 'check(_oracle_paraxial_exit_pupil(S, 1.0, 0.0, 0.25))'},
        {"setup": c + """S = np.array([[0.0, 0.3, 0.2, 1e-3, 0, 1.7, 2.0], [0.8, -0.1, 0, 0, 0, 1.2, 2.0],
              [1.5, 0.05, 0, 0, 0, 1.0, 2.0]])
""",  # 2.4
         "call": 'np.array(paraxial_exit_pupil(S, 1.33, -0.4, 0.2), dtype=float)',
         "gold_call": 'np.array(_oracle_paraxial_exit_pupil(S, 1.33, -0.4, 0.2), dtype=float)'},
        {"setup": c + """S = np.array([[0.0, -0.4, 0, 0, 0, 1.6, 2.0], [0.5, 0.1, 0, 0, 0, 1.0, 2.0],
              [1.5, 0.35, 0, 0, 0, 1.52, 2.0], [2.4, -0.2, 0, 0, 0, 1.0, 2.0]])
""",  # 2.5
         "call": 'np.array(paraxial_exit_pupil(S, 1.0, -1.0, 0.25), dtype=float)',
         "gold_call": 'np.array(_oracle_paraxial_exit_pupil(S, 1.0, -1.0, 0.25), dtype=float)'},
    ]
