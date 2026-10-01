"""
Run the whole construction on the configuration and report it. Build the reference set, place the nodes, take the orthogonal slices, and then for each contour level in turn build the conditional free energies, find the adaptive half-widths, smooth them into the envelope, project the profile through the tube, and evaluate the closed-form predictions at the local stiffnesses of the nodes, the transverse stiffness k(t) of the landscape at each node's curve parameter. Record one row per contour level, and put the run-level quantities in a leading row. The integral of the envelope along the progress coordinate is taken with the trapezoidal rule over the node grid, the abscissae being the uniform progress-grid values from 0 to 1 inclusive that define the nodes, and the reported total is the sum of those integrals over the contour levels.

The source validates the method in two ways at once, and this audit reproduces both on the same run: that the tube adapts its geometry by an order of magnitude between the open sites and the pinched saddles, and that the fraction of the orthogonal partition function it retains barely moves while it does so. The second is the theorem; the first is the reason the theorem is useful.

Returns
-------
A (len(contours) + 1, 9) float64 array. Row zero holds [reported total, soft-minimum sharpness, mean on-path orthogonal baseline at the nodes, smallest node stiffness, largest node stiffness, mean orthogonal coordinate of the reference configurations themselves, span of the progress coordinate over the reference configurations, 0, 0]. Each later row holds [contour level, contour level in kcal/mol, integral of the envelope along the progress coordinate, smallest envelope value, largest envelope value, mean retained fraction, closed-form retained fraction, mean ratio of the unsmoothed half-width to its harmonic estimate 2 Delta F*/k(t) (the half-width of a harmonic well of stiffness k(t) at the same excess, in the offset coordinate), span of the tube-restricted profile in kcal/mol].
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def tube_audit(n_images: int, n_stations: int, n_ortho: int, n_max: float, d_eff: int, smooth_window: int, dz_ref: float, contours: tuple) -> "np.ndarray":
    """Run the whole construction on the configuration and report it. Build the reference set, place
    the nodes, take the orthogonal slices, and then for each contour level in turn build the
    conditional free energies, find the adaptive half-widths, smooth them into the envelope, project
    the profile through the tube, and evaluate the closed-form predictions at the local stiffnesses
    of the nodes, the transverse stiffness k(t) of the landscape at each node's curve parameter.
    Record one row per contour level, and put the run-level quantities in a leading row. The
    integral of the envelope along the progress coordinate is taken with the trapezoidal rule over
    the node grid, the abscissae being the uniform progress-grid values from 0 to 1 inclusive that
    define the nodes, and the reported total is the sum of those integrals over the contour levels.

    Args:
        n_images: int, the number of reference configurations (at least 2).
        n_stations: int, the number of envelope nodes (at least 3), uniform in the progress
            coordinate from 0 to 1 inclusive.
        n_ortho: int, an odd number of at least 3, the number of uniformly spaced signed normal
            offsets over the sampled window, endpoints included.
        n_max: float, the positive half-extent of the sampled window in Angstrom (offsets run from
            -n_max to n_max).
        d_eff: int, the dimensionality of the collective-variable space (at least 2); the number of
            perpendicular directions is d_eff - 1.
        smooth_window: int, a positive odd number of nodes for the moving average of the half-width
            profile.
        dz_ref: float, the positive offset in Angstrom^2 at which the reference level of the
            conditional free energy is read.
        contours: sequence of positive floats, the contour levels in units of k_B T (at least one).

    Returns:
        A (len(contours) + 1, 9) float64 array. Row zero holds [reported total, soft-minimum
        sharpness, mean on-path orthogonal baseline at the nodes, smallest node stiffness, largest
        node stiffness, mean orthogonal coordinate of the reference configurations themselves, span
        of the progress coordinate over the reference configurations, 0, 0]. Each later row holds
        [contour level, contour level in kcal/mol, integral of the envelope along the progress
        coordinate, smallest envelope value, largest envelope value, mean retained fraction, closed-
        form retained fraction, mean ratio of the unsmoothed half-width to its harmonic estimate 2
        Delta F*/k(t) (the half-width of a harmonic well of stiffness k(t) at the same excess, in
        the offset coordinate), span of the tube-restricted profile in kcal/mol].

    Raises:
        ValueError: if contours is empty, or for any invalid argument of the underlying steps
        (n_images below 2, n_stations below 3, an even n_ortho, a non-positive n_max or dz_ref,
        d_eff below 2, an even smooth_window).
    """
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _K_B():
    return 0.0019872041

def _TEMP():
    return 300.0

def _KT():
    return _K_B()*_TEMP()

def _K_SOFT():
    return 40.0

def _K_STIFF():
    return 400.0

def _W_NECK():
    return 0.055

def _T_SAD1():
    return 0.30

def _T_SAD2():
    return 0.70

def _stiffness(t):
    t = np.asarray(t, dtype=float)
    return _K_SOFT() + (_K_STIFF()-_K_SOFT())*(np.exp(-((t-_T_SAD1())/_W_NECK())**2)
                                      + np.exp(-((t-_T_SAD2())/_W_NECK())**2))

def _lambda_sharpness(images):
    images = np.atleast_2d(np.asarray(images, dtype=float))
    if images.shape[0] < 2:
        raise ValueError("at least two reference images are needed")
    return 2.3/float(np.mean(np.sum(np.diff(images, axis=0)**2, axis=1)))

def _oracle_tube_audit(n_images: int, n_stations: int, n_ortho: int, n_max: float, d_eff: int, smooth_window: int, dz_ref: float, contours: tuple) -> "np.ndarray":
    contours = tuple(float(c) for c in contours)
    if len(contours) == 0:
        raise ValueError("at least one contour level is required")
    images = _oracle_reference_path(n_images)
    image_pcv = _oracle_pcv_coordinates(images, images)
    stations = _oracle_station_frame(images, n_stations)
    slices = _oracle_orthogonal_slices(stations, images, n_ortho, n_max)
    cond = _oracle_conditional_free_energy(slices, n_ortho, n_max, d_eff)
    s_grid = stations[:, 0]
    kappa = _stiffness(stations[:, 1])
    rows = []
    total = 0.0
    for c in contours:
        dfs = c*_KT()
        raw = _oracle_adaptive_half_width(cond, slices, dfs, d_eff, dz_ref)
        env = _oracle_smooth_envelope(raw, smooth_window)
        prof = _oracle_tube_projected_profile(slices, env, n_ortho, n_max)
        ref = _oracle_harmonic_tube_reference(d_eff, dfs, kappa)
        integral = float(np.trapezoid(env, s_grid))
        total += integral
        rows.append([c, dfs, integral, float(env.min()), float(env.max()),
                     float(prof[:, 1].mean()), float(ref[0, 0]),
                     float(np.mean(raw/(2.0*dfs/kappa))),
                     float(prof[:, 2].max()-prof[:, 2].min())])
    audit = np.array(rows, dtype=float)
    head = np.zeros((1, audit.shape[1]))
    head[0, 0] = total
    head[0, 1] = float(_lambda_sharpness(images))
    head[0, 2] = float(stations[:, 2].mean())
    head[0, 3] = float(kappa.min())
    head[0, 4] = float(kappa.max())
    head[0, 5] = float(image_pcv[:, 1].mean())
    head[0, 6] = float(image_pcv[:, 0].max() - image_pcv[:, 0].min())
    return np.vstack([head, audit])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {'setup': 'import numpy as np\nn_images = 150\nn_stations = 35\nn_ortho = 1801\nn_max = 0.32\nd_eff = 2\nsmooth_window = 3\ndz_ref = 8.0e-5\ncontours = (1.0, 2.0, 3.0, 4.0)\n',
         'call': 'tube_audit(n_images, n_stations, n_ortho, n_max, d_eff, smooth_window, dz_ref, contours)',
         'gold_call': '_oracle_tube_audit(n_images, n_stations, n_ortho, n_max, d_eff, smooth_window, dz_ref, contours)'},
        {'setup': 'import numpy as np\nn_images = 60\nn_stations = 21\nn_ortho = 1201\nn_max = 0.28\nd_eff = 2\nsmooth_window = 3\ndz_ref = 8.0e-5\ncontours = (2.0, 3.0)\n',
         'call': 'tube_audit(n_images, n_stations, n_ortho, n_max, d_eff, smooth_window, dz_ref, contours)',
         'gold_call': '_oracle_tube_audit(n_images, n_stations, n_ortho, n_max, d_eff, smooth_window, dz_ref, contours)'},
        {'setup': 'import numpy as np\nn_images = 120\nn_stations = 31\nn_ortho = 1601\nn_max = 0.34\nd_eff = 2\nsmooth_window = 5\ndz_ref = 8.0e-5\ncontours = (1.5,)\n',
         'call': 'tube_audit(n_images, n_stations, n_ortho, n_max, d_eff, smooth_window, dz_ref, contours)',
         'gold_call': '_oracle_tube_audit(n_images, n_stations, n_ortho, n_max, d_eff, smooth_window, dz_ref, contours)'},
    ]
