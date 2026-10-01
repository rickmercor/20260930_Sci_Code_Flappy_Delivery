"""
Place the nodes on which the confinement envelope will be represented. The source represents the envelope on a grid in the progress coordinate, so the nodes are the points of the guide curve whose on-path progress coordinate falls on a uniform grid spanning the whole path, endpoints included. Locate each node to machine precision by bisection on the guide-curve parameter rather than by interpolating a table, and pin the two end nodes to the two ends of the curve. For each node also report the on-path value of the orthogonal coordinate, which the source needs as a baseline because that value is not zero, together with the speed and the signed curvature of the guide curve there.

Everything downstream is a property of one orthogonal slice of the landscape, and the slices have to be anchored somewhere. Anchoring them in the progress coordinate rather than in the curve parameter is what makes the envelope a function of the variable the restraint actually reads. The local frame quantities are needed twice over: to erect the orthogonal direction, and to weight the orthogonal quadrature correctly on a curved route.

Returns
-------
An (n_stations, 5) float64 array with columns [progress coordinate, guide-curve parameter, on-path orthogonal coordinate, guide-curve speed, signed guide-curve curvature].
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def station_frame(images: "np.ndarray", n_stations: int) -> "np.ndarray":
    """Place the nodes on which the confinement envelope will be represented. The source represents the
    envelope on a grid in the progress coordinate, so the nodes are the points of the guide curve
    whose on-path progress coordinate falls on a uniform grid spanning the whole path, endpoints
    included. Locate each node to machine precision by bisection on the guide-curve parameter rather
    than by interpolating a table, and pin the two end nodes to the two ends of the curve. For each
    node also report the on-path value of the orthogonal coordinate, which the source needs as a
    baseline because that value is not zero, together with the speed and the signed curvature of the
    guide curve there.

    Args:
        images: array-like of shape (n_images, 2), the ordered reference configurations returned by
            reference_path.
        n_stations: int, the number of envelope nodes (at least 3), uniform in the progress
            coordinate from 0 to 1 inclusive.

    Returns:
        An (n_stations, 5) float64 array with columns [progress coordinate, guide-curve parameter,
        on-path orthogonal coordinate, guide-curve speed, signed guide-curve curvature].

    Raises:
        ValueError: if n_stations is smaller than 3, or if fewer than two reference configurations
        are given.
    """
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _PATH_AMP():
    return 0.15

def _guide(t):
    t = np.asarray(t, dtype=float)
    return np.stack([t, _PATH_AMP()*np.sin(np.pi*t)], -1)

def _guide_d1(t):
    t = np.asarray(t, dtype=float)
    return np.stack([np.ones_like(t), _PATH_AMP()*np.pi*np.cos(np.pi*t)], -1)

def _guide_d2(t):
    t = np.asarray(t, dtype=float)
    return np.stack([np.zeros_like(t), -_PATH_AMP()*np.pi**2*np.sin(np.pi*t)], -1)

def _local_frame(t):
    d1 = _guide_d1(t); speed = np.linalg.norm(d1, axis=-1)
    tang = d1/speed[..., None]
    normal = np.stack([-tang[..., 1], tang[..., 0]], -1)
    d2 = _guide_d2(t)
    curv = (d1[..., 0]*d2[..., 1] - d1[..., 1]*d2[..., 0])/speed**3
    return normal, speed, curv

def _oracle_station_frame(images: "np.ndarray", n_stations: int) -> "np.ndarray":
    n_stations = int(n_stations)
    if n_stations < 3:
        raise ValueError("n_stations must be at least 3")
    s_grid = np.linspace(0.0, 1.0, n_stations)

    def _progress(t):
        return _oracle_pcv_coordinates(_guide(np.atleast_1d(t)), images)[:, 0]

    lo = np.zeros(n_stations); hi = np.ones(n_stations)
    s_lo = _progress(lo); s_hi = _progress(hi)
    for _ in range(60):
        mid = 0.5*(lo + hi)
        s_mid = _progress(mid)
        left = s_mid < s_grid
        lo = np.where(left, mid, lo)
        hi = np.where(left, hi, mid)
    t_st = np.clip(0.5*(lo + hi), 0.0, 1.0)
    t_st[0] = 0.0
    t_st[-1] = 1.0
    on_path = _oracle_pcv_coordinates(_guide(t_st), images)
    _, speed, curv = _local_frame(t_st)
    return np.stack([s_grid, t_st, on_path[:, 1], speed, curv], 1)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {'setup': 'import numpy as np\nimages = _oracle_reference_path(60)\nn_stations = 11\n',
         'call': 'station_frame(images, n_stations)',
         'gold_call': '_oracle_station_frame(images, n_stations)'},
        {'setup': 'import numpy as np\nimages = _oracle_reference_path(200)\nn_stations = 41\n',
         'call': 'station_frame(images, n_stations)',
         'gold_call': '_oracle_station_frame(images, n_stations)'},
        {'setup': 'import numpy as np\nimages = _oracle_reference_path(24)\nn_stations = 9\n',
         'call': 'station_frame(images, n_stations)',
         'gold_call': '_oracle_station_frame(images, n_stations)'},
    ]
