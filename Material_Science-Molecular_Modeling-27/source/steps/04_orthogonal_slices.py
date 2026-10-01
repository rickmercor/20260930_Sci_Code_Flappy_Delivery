"""
Build the orthogonal slice of the ensemble at every node. Step off the guide curve along the unit normal on a uniform grid of signed offsets covering the sampling window symmetrically, endpoints included, map each offset to a configuration in the collective-variable plane, and report two numbers for it: the offset coordinate that the confinement acts on, which the source defines as the orthogonal coordinate measured from the on-path baseline of that node rather than from zero; and the equilibrium statistical weight of the configuration. The weight is the Boltzmann factor of the landscape at 300 K together with the volume element of the curved frame; shift the exponent by the global minimum of the landscape over the grid so the weights stay in range, which changes every weight by one common factor and cancels everywhere it is used.

The tube is a restraint on one number per configuration, and that number is not the raw orthogonal coordinate. For an arithmetic path variable the on-path value is a soft-minimum floor, not zero, so the offset has to be taken against that floor before any free-energy statement about it means anything. The volume element matters because the normal offsets of a curved route do not sweep equal areas.

Returns
-------
A (n_stations, n_ortho, 2) float64 array whose last axis holds [offset coordinate, statistical weight].
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def orthogonal_slices(stations: "np.ndarray", images: "np.ndarray", n_ortho: int, n_max: float) -> "np.ndarray":
    """Build the orthogonal slice of the ensemble at every node. Step off the guide curve along the
    unit normal on a uniform grid of signed offsets covering the sampling window symmetrically,
    endpoints included, map each offset to a configuration in the collective-variable plane, and
    report two numbers for it: the offset coordinate that the confinement acts on, which the source
    defines as the orthogonal coordinate measured from the on-path baseline of that node rather than
    from zero; and the equilibrium statistical weight of the configuration. The weight is the
    Boltzmann factor of the landscape at 300 K together with the volume element of the curved frame;
    shift the exponent by the global minimum of the landscape over the grid so the weights stay in
    range, which changes every weight by one common factor and cancels everywhere it is used.

    Args:
        stations: array-like of shape (n_stations, 5), the node table returned by station_frame.
        images: array-like of shape (n_images, 2), the ordered reference configurations returned by
            reference_path.
        n_ortho: int, an odd number of at least 3, the number of uniformly spaced signed normal
            offsets over the sampled window, endpoints included.
        n_max: float, the positive half-extent of the sampled window in Angstrom (offsets run from
            -n_max to n_max).

    Returns:
        A (n_stations, n_ortho, 2) float64 array whose last axis holds [offset coordinate,
        statistical weight].

    Raises:
        ValueError: if stations does not have five columns, if n_ortho is even or smaller than 3, or
        if n_max is not positive.
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

def _BETA():
    return 1.0/(_K_B()*_TEMP())

def _PATH_AMP():
    return 0.15

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

def _CUBIC():
    return 30.0

def _QUART():
    return 140.0

def _QUART_M():
    return 0.60

def _RIDGE():
    return 0.35

def _RIDGE_T():
    return 0.10

def _RIDGE_N():
    return 0.055

def _guide(t):
    t = np.asarray(t, dtype=float)
    return np.stack([t, _PATH_AMP()*np.sin(np.pi*t)], -1)

def _guide_d1(t):
    t = np.asarray(t, dtype=float)
    return np.stack([np.ones_like(t), _PATH_AMP()*np.pi*np.cos(np.pi*t)], -1)

def _guide_d2(t):
    t = np.asarray(t, dtype=float)
    return np.stack([np.zeros_like(t), -_PATH_AMP()*np.pi**2*np.sin(np.pi*t)], -1)

def _stiffness(t):
    t = np.asarray(t, dtype=float)
    return _K_SOFT() + (_K_STIFF()-_K_SOFT())*(np.exp(-((t-_T_SAD1())/_W_NECK())**2)
                                      + np.exp(-((t-_T_SAD2())/_W_NECK())**2))

def _along_path_energy(t):
    t = np.asarray(t, dtype=float)
    return (7.5*np.exp(-((t-_T_SAD1())/0.10)**2)
            + 6.2*np.exp(-((t-_T_SAD2())/0.10)**2)
            - 1.6*np.exp(-((t-0.50)/0.11)**2))

def _energy(t, n):
    t = np.asarray(t, dtype=float); n = np.asarray(n, dtype=float)
    return (_along_path_energy(t)
            + 0.5*_stiffness(t)*n**2
            + _CUBIC()*np.sin(2.0*np.pi*t)*n**3
            + _QUART()*(1.0+_QUART_M()*np.cos(2.0*np.pi*t))*n**4
            + _RIDGE()*np.exp(-((t-0.50)/_RIDGE_T())**2)*np.exp(-(n/_RIDGE_N())**2))

def _local_frame(t):
    d1 = _guide_d1(t); speed = np.linalg.norm(d1, axis=-1)
    tang = d1/speed[..., None]
    normal = np.stack([-tang[..., 1], tang[..., 0]], -1)
    d2 = _guide_d2(t)
    curv = (d1[..., 0]*d2[..., 1] - d1[..., 1]*d2[..., 0])/speed**3
    return normal, speed, curv

def _oracle_orthogonal_slices(stations: "np.ndarray", images: "np.ndarray", n_ortho: int, n_max: float) -> "np.ndarray":
    stations = np.atleast_2d(np.asarray(stations, dtype=float))
    n_ortho = int(n_ortho); n_max = float(n_max)
    if stations.shape[1] != 5:
        raise ValueError("stations must have five columns")
    if n_ortho < 3 or n_ortho % 2 == 0:
        raise ValueError("n_ortho must be an odd integer of at least 3")
    if n_max <= 0.0:
        raise ValueError("n_max must be positive")
    ng = np.linspace(-n_max, n_max, n_ortho)
    t_st = stations[:, 1]; z_floor = stations[:, 2]
    speed = stations[:, 3]; curv = stations[:, 4]
    normal, _, _ = _local_frame(t_st)
    xy = _guide(t_st)[:, None, :] + ng[None, :, None]*normal[:, None, :]
    z = _oracle_pcv_coordinates(xy.reshape(-1, 2), images)[:, 1]
    z = z.reshape(len(t_st), n_ortho)
    u = _energy(t_st[:, None], ng[None, :])
    w = np.exp(-_BETA()*(u - u.min()))*np.abs(1.0 - ng[None, :]*curv[:, None])*speed[:, None]
    return np.stack([z - z_floor[:, None], w], -1)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {'setup': 'import numpy as np\nimages = _oracle_reference_path(60)\nstations = _oracle_station_frame(images, 11)\nn_ortho = 801\nn_max = 0.34\n',
         'call': 'orthogonal_slices(stations, images, n_ortho, n_max)',
         'gold_call': '_oracle_orthogonal_slices(stations, images, n_ortho, n_max)'},
        {'setup': 'import numpy as np\nimages = _oracle_reference_path(200)\nstations = _oracle_station_frame(images, 21)\nn_ortho = 1201\nn_max = 0.25\n',
         'call': 'orthogonal_slices(stations, images, n_ortho, n_max)',
         'gold_call': '_oracle_orthogonal_slices(stations, images, n_ortho, n_max)'},
        {'setup': 'import numpy as np\nimages = _oracle_reference_path(200)\nstations = _oracle_station_frame(images, 41)\nn_ortho = 2001\nn_max = 0.34\n',
         'call': 'orthogonal_slices(stations, images, n_ortho, n_max)',
         'gold_call': '_oracle_orthogonal_slices(stations, images, n_ortho, n_max)'},
    ]
