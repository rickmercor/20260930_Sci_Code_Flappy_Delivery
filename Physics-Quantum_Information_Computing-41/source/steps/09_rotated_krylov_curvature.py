"""
Evaluate the complete fixed-decision noise-curvature experiment.

Combine deterministic measurement jets, physical-overlap repair, sequential noise-aware basis growth, variance-based angle selection and the response of the moving regularized subspace. The discrete dimension and angle are selected at zero and then held fixed; the retained projector is not held fixed.

Returns
-------
float, the unrounded second derivative at x=0 of the final aggregate physical ground energy, returned as a native Python float, with the selected dimension and grid angle held fixed while the retained spectral projector varies with x.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import math
from numbers import Integral, Real

import numpy as np
from scipy.linalg import eigh

def rotated_krylov_curvature(energies: 'np.ndarray', weights: 'np.ndarray', dt: float = 0.6, dimension: int = 6, batches: int = 4, noise_h: float = 0.035, noise_s: float = 0.065, cutoff: float = 0.08, angles: 'np.ndarray | tuple[float, ...]' = (0.0, 0.2, 0.45, 0.7, 0.95, 1.2, 1.4), gamma: float = 1.3, energy_tolerance: float = 1e-4) -> float:
    """Evaluate the complete fixed-decision noise-curvature experiment.
    
    Parameters
    ----------
    energies, weights : np.ndarray, shape (n,)
        Finite real levels and nonnegative spectral weights with positive finite total.
    dt : float, optional
        Nonnegative real-time spacing, default 0.6.
    dimension : int, optional
        Maximum number of Krylov columns in [3,32], default 6.
    batches : int, optional
        Equal-size batch count in [1,64], default 4.
    noise_h, noise_s : float, optional
        Nonnegative noise amplitudes, defaults 0.035 and 0.065.
    cutoff : float, optional
        Nonnegative absolute spectral cutoff, default 0.08.
    angles : np.ndarray, shape (m,), optional
        Nonempty ordered angle grid, default (0,0.2,0.45,0.7,0.95,1.2,1.4).
    gamma : float, optional
        Nonnegative adjacent-spread multiplier, default 1.3.
    energy_tolerance : float, optional
        Nonnegative absolute stopping threshold, default 1e-4.
    
    Returns
    -------
    result : float
        The unrounded second derivative at x=0 of the final aggregate physical ground energy.
    
    Raises
    ------
    ValueError
        If any preparation or selection parameter violates its stated domain, or any physical projection, thresholded solve, projector derivative or selected-energy derivative is undefined under the preceding contracts.
    
    Notes
    -----
    Call measurement_pencil_jets to construct the input jets. At dimensions 2,3,..., repair every raw overlap prefix with nearest_physical_overlap and compute the unrotated batch energies with rotated_ground_energy. Use batch_energy_moments and dimension_convergence on adjacent dimensions; the first success returns the current dimension, otherwise keep the maximum. Recompute the retained-dimension batch energies for all supplied angles and use select_variance_angle. Average the raw retained-dimension pencil jets uniformly, then repair the raw aggregate overlap once. Rotate the aggregate jets, obtain cutoff_projector_jet and evaluate projected_energy_jet. Return only its second derivative. All eight preceding public functions are required components. Use default repair tolerance 1e-13 and iteration budget 2000, derivative gap tolerance 1e-10 and inverse-map pole tolerance 1e-12. Every input is read-only; perform no I/O. For the derivative, freeze the selected dimension and grid angle, not the cutoff projector. Do not differentiate the discrete selection procedure.
    """
    return 0.0

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_rotated_krylov_curvature(energies: 'np.ndarray', weights: 'np.ndarray', dt: float = 0.6, dimension: int = 6, batches: int = 4, noise_h: float = 0.035, noise_s: float = 0.065, cutoff: float = 0.08, angles: 'np.ndarray | tuple[float, ...]' = (0.0, 0.2, 0.45, 0.7, 0.95, 1.2, 1.4), gamma: float = 1.3, energy_tolerance: float = 1e-4) -> float:
    import numpy as np
    from numbers import Integral, Real
    if isinstance(dimension, bool) or not isinstance(dimension, Integral) or dimension < 3:
        raise ValueError('The complete growth experiment requires at least three basis states.')
    for x in (gamma, energy_tolerance):
        if isinstance(x, bool) or not isinstance(x, Real) or not np.isfinite(x) or x < 0:
            raise ValueError('Convergence parameters must be finite and nonnegative.')
    try:
        theta_grid_raw = np.asarray(angles)
        if np.iscomplexobj(theta_grid_raw) and np.any(theta_grid_raw.imag != 0):
            raise ValueError('angles must be real-valued; nonzero imaginary parts are invalid.')
        theta_grid = np.asarray(theta_grid_raw.real if np.iscomplexobj(theta_grid_raw) else theta_grid_raw, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError('A real angle grid is required.') from exc
    if theta_grid.ndim != 1 or theta_grid.size == 0 or not np.all(np.isfinite(theta_grid)):
        raise ValueError('The angle grid must be a nonempty finite vector.')
    data = _oracle_measurement_pencil_jets(energies, weights, dt, dimension, batches, noise_h, noise_s)
    previous = None
    for size in range(2, dimension + 1):
        repaired = np.stack([_oracle_nearest_physical_overlap(data[0, 1, q, :size, :size]) for q in range(batches)])
        sample = np.array([_oracle_rotated_ground_energy(data[0, 0, q, :size, :size], repaired[q], 0.0, cutoff) for q in range(batches)])
        current = _oracle_batch_energy_moments(sample)
        if previous is not None:
            certificate = _oracle_dimension_convergence(previous, current, gamma, energy_tolerance)
            if certificate[2] == 1:
                break
        previous = current
    table = np.array([[_oracle_rotated_ground_energy(data[0, 0, q, :size, :size], repaired[q], theta, cutoff) for q in range(batches)] for theta in theta_grid])
    choice = _oracle_select_variance_angle(theta_grid, table)
    theta = float(choice[1])
    h = data[:, 0, :, :size, :size].mean(axis=1)
    s = np.zeros_like(h)
    s[0] = _oracle_nearest_physical_overlap(data[0, 1, :, :size, :size].mean(axis=0))
    c, t = np.cos(theta), np.sin(theta)
    numerator = c * h - t * s
    denominator = c * s + t * h
    projector = _oracle_cutoff_projector_jet(denominator, cutoff)
    result = _oracle_projected_energy_jet(numerator, denominator, projector, theta)
    return float(result[2])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return deterministic cases with oracle-independent input setup."""
    check_setup = """
def _checked(function, *args, _expect_exception=False, **kwargs):
    # Freeze the first call's inputs before invoking the candidate. Both calls
    # use fresh copies of that baseline even if setup variables later change.
    if not hasattr(_checked, "_baseline"):
        _checked._baseline = (
            tuple(x.copy() if isinstance(x, np.ndarray) else x for x in args),
            {k: x.copy() if isinstance(x, np.ndarray) else x for k, x in kwargs.items()},
        )
    base_args, base_kwargs = _checked._baseline
    copied_args = tuple(x.copy() if isinstance(x, np.ndarray) else x for x in base_args)
    copied_kwargs = {
        k: x.copy() if isinstance(x, np.ndarray) else x
        for k, x in base_kwargs.items()
    }
    watched = [x for x in copied_args if isinstance(x, np.ndarray)]
    watched.extend(x for x in copied_kwargs.values() if isinstance(x, np.ndarray))
    watched.extend(x for x in args if isinstance(x, np.ndarray))
    watched.extend(x for x in kwargs.values() if isinstance(x, np.ndarray))
    snapshots = [x.copy() for x in watched]

    try:
        result = function(*copied_args, **copied_kwargs)
    except ValueError:
        if not _expect_exception:
            raise
        result = 1
    except Exception:
        if not _expect_exception:
            raise
        result = 2
    else:
        if _expect_exception:
            result = 0

    preserved = all(
        after.shape == before.shape
        and after.dtype == before.dtype
        and np.array_equal(after, before, equal_nan=True)
        for after, before in zip(watched, snapshots)
    )
    # Pack values with shape, return-kind and preservation witnesses. Use
    # complex magnitudes so real/complex numeric arrays compare as before.
    value = np.asarray(result)
    if isinstance(result, np.ndarray) and value.dtype.kind in "uifc":
        kind = 1.0
    elif isinstance(result, (int, float, complex, np.number)) and not isinstance(result, (bool, np.bool_)):
        kind = 0.0
    else:
        kind = -1.0
    shape = np.array((value.ndim, *value.shape, kind), dtype=complex)
    numbers = value.ravel().astype(complex)
    return np.concatenate((shape, numbers, np.array([float(preserved)], dtype=complex)))
"""

    setup_1 = """import numpy as np
e = np.array([0.2, 0.8, 1.4, 2.1, 3.0])
w = np.array([0.26, 0.22, 0.2, 0.18, 0.14])
"""

    setup_2 = """import numpy as np
e = np.array([0.2, 0.8, 1.4, 2.1, 3.0])
w = np.array([0.26, 0.22, 0.2, 0.18, 0.14])
w[0] = -1
"""

    setup_3 = """import numpy as np
e = np.array([0.2, 0.8, 1.4, 2.1, 3.0])
w = np.array([0.26, 0.22, 0.2, 0.18, 0.14])
e = e.astype(complex)
e[0] += 1j
"""

    setup_4 = """import numpy as np
e = np.array([0.2, 0.8, 1.4, 2.1, 3.0])
w = np.array([0.26, 0.22, 0.2, 0.18, 0.14])
w = w.astype(complex)
w[0] += 1j
"""

    setup_5 = """import numpy as np
e = np.array([0.2, 0.8, 1.4, 2.1, 3.0])
w = np.array([0.26, 0.22, 0.2, 0.18, 0.14])
angles = np.array([1.2 + 1j])
"""

    cases = [
        # Case 1
        {
            'setup': setup_1,
            'call': '_checked(rotated_krylov_curvature, e,w)',
            'gold_call': '_checked(_oracle_rotated_krylov_curvature, e,w)',
        },
        # Case 2
        {
            'setup': setup_1,
            'call': '_checked(rotated_krylov_curvature, e,w,gamma=0.,energy_tolerance=0.)',
            'gold_call': '_checked(_oracle_rotated_krylov_curvature, e,w,gamma=0.,energy_tolerance=0.)',
        },
        # Case 3
        {
            'setup': setup_1,
            'call': '_checked(rotated_krylov_curvature, e,w,angles=np.array([0.]))',
            'gold_call': '_checked(_oracle_rotated_krylov_curvature, e,w,angles=np.array([0.]))',
        },
        # Case 4
        {
            'setup': setup_1,
            'call': '_checked(rotated_krylov_curvature, e,w,noise_s=.035)',
            'gold_call': '_checked(_oracle_rotated_krylov_curvature, e,w,noise_s=.035)',
        },
        # Case 5
        {
            'setup': setup_1,
            'call': '_checked(rotated_krylov_curvature, e,w,noise_h=0.)',
            'gold_call': '_checked(_oracle_rotated_krylov_curvature, e,w,noise_h=0.)',
        },
        # Case 6
        {
            'setup': setup_1,
            'call': '_checked(rotated_krylov_curvature, e,w,gamma=3.)',
            'gold_call': '_checked(_oracle_rotated_krylov_curvature, e,w,gamma=3.)',
        },
        # Case 7
        {
            'setup': setup_1,
            'call': '_checked(rotated_krylov_curvature, e,w,energy_tolerance=.1)',
            'gold_call': '_checked(_oracle_rotated_krylov_curvature, e,w,energy_tolerance=.1)',
        },
        # Case 8
        {
            'setup': setup_1,
            'call': '_checked(rotated_krylov_curvature, e,w,angles=np.array([1.4,.7,.2,0.]))',
            'gold_call': '_checked(_oracle_rotated_krylov_curvature, e,w,angles=np.array([1.4,.7,.2,0.]))',
        },
        # Case 9
        {
            'setup': setup_1,
            'call': '_checked(rotated_krylov_curvature, e,w,dimension=3)',
            'gold_call': '_checked(_oracle_rotated_krylov_curvature, e,w,dimension=3)',
        },
        # Case 10
        {
            'setup': setup_1,
            'call': '_checked(rotated_krylov_curvature, e,w,noise_s=0.)',
            'gold_call': '_checked(_oracle_rotated_krylov_curvature, e,w,noise_s=0.)',
        },
        # Case 11
        {
            'setup': setup_1,
            'call': '_checked(rotated_krylov_curvature, e,w,batches=1)',
            'gold_call': '_checked(_oracle_rotated_krylov_curvature, e,w,batches=1)',
        },
        # Case 12
        {
            'setup': setup_1,
            'call': '_checked(rotated_krylov_curvature, e, w, angles=np.array([]), _expect_exception=True)',
            'gold_call': '_checked(_oracle_rotated_krylov_curvature, e, w, angles=np.array([]), _expect_exception=True)',
        },
        # Case 13
        {
            'setup': setup_1,
            'call': '_checked(rotated_krylov_curvature, e, w, dimension=2, _expect_exception=True)',
            'gold_call': '_checked(_oracle_rotated_krylov_curvature, e, w, dimension=2, _expect_exception=True)',
        },
        # Case 14
        {
            'setup': setup_1,
            'call': '_checked(rotated_krylov_curvature, e, w, cutoff=100.0, _expect_exception=True)',
            'gold_call': '_checked(_oracle_rotated_krylov_curvature, e, w, cutoff=100.0, _expect_exception=True)',
        },
        # Case 15
        {
            'setup': setup_1,
            'call': '_checked(rotated_krylov_curvature, e, w, gamma=-1.0, _expect_exception=True)',
            'gold_call': '_checked(_oracle_rotated_krylov_curvature, e, w, gamma=-1.0, _expect_exception=True)',
        },
        # Case 16
        {
            'setup': setup_2,
            'call': '_checked(rotated_krylov_curvature, e, w, _expect_exception=True)',
            'gold_call': '_checked(_oracle_rotated_krylov_curvature, e, w, _expect_exception=True)',
        },
        # Case 17
        {
            'setup': setup_3,
            'call': '_checked(rotated_krylov_curvature, e, w, _expect_exception=True)',
            'gold_call': '_checked(_oracle_rotated_krylov_curvature, e, w, _expect_exception=True)',
        },
        # Case 18
        {
            'setup': setup_4,
            'call': '_checked(rotated_krylov_curvature, e, w, _expect_exception=True)',
            'gold_call': '_checked(_oracle_rotated_krylov_curvature, e, w, _expect_exception=True)',
        },
        # Case 19
        {
            'setup': setup_5,
            'call': '_checked(rotated_krylov_curvature, e, w, angles=angles, _expect_exception=True)',
            'gold_call': '_checked(_oracle_rotated_krylov_curvature, e, w, angles=angles, _expect_exception=True)',
        },
    ]
    for case in cases:
        case['setup'] += check_setup
    return cases
