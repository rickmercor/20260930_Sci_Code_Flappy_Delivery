"""
Normalize a vector velocity correlation and evaluate its generating sum

The thermal velocity scale fixes the zero-lag matrix. A matrix generating sum retains lag orientation and uses one complex grid for all components.

Returns
-------
Tuple (Y,z,F): real (n,d,d), complex (grid_size,), and complex     (grid_size,d,d) arrays, in that order. Y and F are dimensionless.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def correlation_transform(cvv: "np.ndarray", mass: float, thermal_energy: float, rho: float, grid_size: int) -> "tuple[np.ndarray, np.ndarray, np.ndarray]":
    """Normalize a vector velocity correlation and evaluate its generating sum.

    Parameters
    ----------
    cvv : real array (n,d,d)
        n >= 4, d >= 1. cvv[k] = <v(k*tau) v(0)^T>, in squared
        velocity units. Positive and negative lags are not symmetrized.
    mass, thermal_energy : positive finite real scalars
        Common probe mass and k_B*T in compatible reduced units.
    rho : finite real scalar > 1
    grid_size : even integer >= 8

    Contract
    --------
    Return the thermally normalized correlation sequence, the complex
    sampling grid and the one-sided matrix generating function
    F(z) = sum over k = 0..n-1 of Y[k] * z**(-(k+1)), evaluated at every
    grid point using every supplied lag, so that a mode
    Y[k] = Gamma * exp(lambda*tau*k) contributes Gamma / (z - exp(lambda*tau)).
    The normalized zero-lag matrix must equal the identity within maximum
    entry error 1e-8. Grid points are uniformly spaced counterclockwise on
    the circle of radius rho, beginning at the positive real axis. Preserve
    the orientation of each cross-correlation matrix.

    Returns
    -------
    result
        Tuple (Y,z,F): real (n,d,d), complex (grid_size,), and complex
        (grid_size,d,d) arrays, in that order. Y and F are dimensionless.

    Raises
    ------
    ValueError
        Complex/nonfinite real input, incompatible shape, n<4, d<1,
        nonpositive mass or thermal_energy, rho<=1, invalid grid_size,
        or Y[0] outside the stated tolerance.
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_correlation_transform(cvv: "np.ndarray", mass: float, thermal_energy: float, rho: float, grid_size: int) -> "tuple[np.ndarray, np.ndarray, np.ndarray]":
    """Normalize a vector velocity correlation and evaluate its generating sum.

    Parameters
    ----------
    cvv : real array (n,d,d)
        n >= 4, d >= 1. cvv[k] = <v(k*tau) v(0)^T>, in squared
        velocity units. Positive and negative lags are not symmetrized.
    mass, thermal_energy : positive finite real scalars
        Common probe mass and k_B*T in compatible reduced units.
    rho : finite real scalar > 1
    grid_size : even integer >= 8

    Contract
    --------
    Return the thermally normalized correlation sequence, the complex
    sampling grid and the source's one-sided matrix generating function
    evaluated on that grid using every supplied lag. The normalized
    zero-lag matrix must equal the identity within maximum entry error
    1e-8. Grid points are uniformly spaced counterclockwise on the circle
    of radius rho, beginning at the positive real axis. Preserve the
    orientation of each cross-correlation matrix.

    Returns
    -------
    result
        Tuple (Y,z,F): real (n,d,d), complex (grid_size,), and complex
        (grid_size,d,d) arrays, in that order. Y and F are dimensionless.

    Raises
    ------
    ValueError
        Complex/nonfinite real input, incompatible shape, n<4, d<1,
        nonpositive mass or thermal_energy, rho<=1, invalid grid_size,
        or Y[0] outside the stated tolerance.
    """
    import numpy as np
    if any((np.iscomplexobj(x) for x in [cvv, mass, thermal_energy, rho])):
        raise ValueError('real inputs required')
    c = np.asarray(cvv, dtype=float)
    if c.ndim != 3 or c.shape[0] < 4 or c.shape[1] < 1 or (c.shape[1] != c.shape[2]) or (not np.isfinite(c).all()):
        raise ValueError('invalid correlations')
    for value in [mass, thermal_energy, rho]:
        if np.ndim(value) != 0 or not np.isfinite(value):
            raise ValueError('invalid scalar')
    if mass <= 0 or thermal_energy <= 0 or rho <= 1:
        raise ValueError('scalar domain')
    if isinstance(grid_size, (bool, np.bool_)) or not isinstance(grid_size, (int, np.integer)) or grid_size < 8 or grid_size % 2:
        raise ValueError('even grid required')
    Y = c * (mass / thermal_energy)
    if np.max(np.abs(Y[0] - np.eye(c.shape[1]))) > 1e-08:
        raise ValueError('normalization')
    z = rho * np.exp(2j * np.pi * np.arange(grid_size) / grid_size)
    F = np.einsum('kab,lk->lab', Y, z[:, None] ** (-np.arange(len(Y)) - 1))
    return (Y, z, F)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return independent numerical test specifications."""
    common_0 = """import copy
import numpy as np
from scipy.linalg import expm, block_diag

def error_code(function, *args, **kwargs):
    try:
        function(*args, **kwargs)
    except ValueError:
        return 1
    except Exception:
        return 2
    return 0

def true_drift(case=0):
    if case == 2:
        return (np.array([[0.0, 1.4, 0.2], [-1.4, -0.8, 1.1], [-0.2, -1.1, -1.2]]), 1)
    B = np.array([[1.2, 0.25], [-0.35, 1.05], [0.7, -0.55], [0.4, 0.8]])
    J = np.array([[0, 1.3, -0.2, 0.4], [-1.3, 0, 0.65, -0.3], [0.2, -0.65, 0, 0.9], [-0.4, 0.3, -0.9, 0.0]])
    A = np.block([[np.zeros((2, 2)), B.T], [-B, J - np.diag([0.6, 1.1, 0.85, 1.4])]])
    return (A * (0.8 if case == 1 else 1.0), 2)

def stage_data(case=0):
    A, d = true_drift(case)
    tau = 0.4
    Y = np.array([expm(k * tau * A)[:d, :d] for k in range(128)])
    z = 1.4 * np.exp(2j * np.pi * np.arange(64) / 64)
    F = np.einsum('kab,lk->lab', Y, z[:, None] ** (-np.arange(len(Y)) - 1))
    rates, V = np.linalg.eig(A)
    W = np.linalg.inv(V)
    residues = np.array([np.outer(V[:d, j], W[j, :d]) for j in range(len(A))])
    return (A, d, Y, z, F, rates, residues)
'Deterministic synthetic probe data; not measurements reported in the article.'
A, d, Y, z, F, rates, gamma = stage_data()
"""
    common_1 = """import copy
import numpy as np
from scipy.linalg import expm, block_diag

def true_drift(case=0):
    if case == 2:
        return (np.array([[0.0, 1.4, 0.2], [-1.4, -0.8, 1.1], [-0.2, -1.1, -1.2]]), 1)
    B = np.array([[1.2, 0.25], [-0.35, 1.05], [0.7, -0.55], [0.4, 0.8]])
    J = np.array([[0, 1.3, -0.2, 0.4], [-1.3, 0, 0.65, -0.3], [0.2, -0.65, 0, 0.9], [-0.4, 0.3, -0.9, 0.0]])
    A = np.block([[np.zeros((2, 2)), B.T], [-B, J - np.diag([0.6, 1.1, 0.85, 1.4])]])
    return (A * (0.8 if case == 1 else 1.0), 2)

def stage_data(case=0):
    A, d = true_drift(case)
    tau = 0.4
    Y = np.array([expm(k * tau * A)[:d, :d] for k in range(128)])
    z = 1.4 * np.exp(2j * np.pi * np.arange(64) / 64)
    F = np.einsum('kab,lk->lab', Y, z[:, None] ** (-np.arange(len(Y)) - 1))
    rates, V = np.linalg.eig(A)
    W = np.linalg.inv(V)
    residues = np.array([np.outer(V[:d, j], W[j, :d]) for j in range(len(A))])
    return (A, d, Y, z, F, rates, residues)
'Deterministic synthetic probe data; not measurements reported in the article.'
"""
    return [
        {
            'setup': common_1 + """A, d, Y, z, F, rates, gamma = stage_data(0)
""",
            'call': 'np.concatenate([np.asarray(x).ravel() for x in correlation_transform(copy.deepcopy(Y / 5), copy.deepcopy(5.0), copy.deepcopy(1.0), copy.deepcopy(1.4), copy.deepcopy(64))])',
            'gold_call': 'np.concatenate([np.asarray(x).ravel() for x in _oracle_correlation_transform(copy.deepcopy(Y / 5), copy.deepcopy(5.0), copy.deepcopy(1.0), copy.deepcopy(1.4), copy.deepcopy(64))])',
        },
        {
            'setup': common_1 + """A, d, Y, z, F, rates, gamma = stage_data(1)
""",
            'call': 'np.concatenate([np.asarray(x).ravel() for x in correlation_transform(copy.deepcopy(Y / 5), copy.deepcopy(5.0), copy.deepcopy(1.0), copy.deepcopy(1.4), copy.deepcopy(64))])',
            'gold_call': 'np.concatenate([np.asarray(x).ravel() for x in _oracle_correlation_transform(copy.deepcopy(Y / 5), copy.deepcopy(5.0), copy.deepcopy(1.0), copy.deepcopy(1.4), copy.deepcopy(64))])',
        },
        {
            'setup': common_1 + """A, d, Y, z, F, rates, gamma = stage_data(2)
""",
            'call': 'np.concatenate([np.asarray(x).ravel() for x in correlation_transform(copy.deepcopy(Y / 5), copy.deepcopy(5.0), copy.deepcopy(1.0), copy.deepcopy(1.4), copy.deepcopy(64))])',
            'gold_call': 'np.concatenate([np.asarray(x).ravel() for x in _oracle_correlation_transform(copy.deepcopy(Y / 5), copy.deepcopy(5.0), copy.deepcopy(1.0), copy.deepcopy(1.4), copy.deepcopy(64))])',
        },
        {
            'setup': """import copy
import numpy as np
from scipy.linalg import expm, block_diag
'Deterministic synthetic probe data; not measurements reported in the article.'
Y = np.repeat(np.eye(1)[None], 4, axis=0)
""",
            'call': 'np.concatenate([np.asarray(x).ravel() for x in correlation_transform(copy.deepcopy(Y), copy.deepcopy(1.0), copy.deepcopy(1.0), copy.deepcopy(2.0), copy.deepcopy(8))])',
            'gold_call': 'np.concatenate([np.asarray(x).ravel() for x in _oracle_correlation_transform(copy.deepcopy(Y), copy.deepcopy(1.0), copy.deepcopy(1.0), copy.deepcopy(2.0), copy.deepcopy(8))])',
        },
        {
            'setup': common_0,
            'call': 'error_code(correlation_transform, copy.deepcopy(Y.astype(complex)), copy.deepcopy(1.0), copy.deepcopy(1.0), copy.deepcopy(1.4), copy.deepcopy(64))',
            'gold_call': 'error_code(_oracle_correlation_transform, copy.deepcopy(Y.astype(complex)), copy.deepcopy(1.0), copy.deepcopy(1.0), copy.deepcopy(1.4), copy.deepcopy(64))',
        },
        {
            'setup': common_0,
            'call': 'error_code(correlation_transform, copy.deepcopy(Y), copy.deepcopy(1.0), copy.deepcopy(1.0), copy.deepcopy(1.4), copy.deepcopy(63))',
            'gold_call': 'error_code(_oracle_correlation_transform, copy.deepcopy(Y), copy.deepcopy(1.0), copy.deepcopy(1.0), copy.deepcopy(1.4), copy.deepcopy(63))',
        },
        {
            'setup': common_0,
            'call': 'error_code(correlation_transform, copy.deepcopy(2 * Y), copy.deepcopy(1.0), copy.deepcopy(1.0), copy.deepcopy(1.4), copy.deepcopy(64))',
            'gold_call': 'error_code(_oracle_correlation_transform, copy.deepcopy(2 * Y), copy.deepcopy(1.0), copy.deepcopy(1.0), copy.deepcopy(1.4), copy.deepcopy(64))',
        },
    ]
