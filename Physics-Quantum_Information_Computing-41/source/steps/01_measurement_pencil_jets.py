"""
Construct the deterministic noisy real-time Krylov pencil and its first two derivatives.

Nonorthogonal time-evolved states define a generalized eigenproblem. A harmonic perturbation of measured Hamiltonian entries gives an exact second-order response input without drawing random numbers. All indices are zero based; lower triangles are Hermitian conjugates.

Returns
-------
np.ndarray, complex, shape (3, 2, batches, dimension, dimension), containing the noisy pencil and its first two derivatives at x=0; result[k, 0, q] is the kth Hamiltonian derivative for batch q, and result[k, 1, q] is the corresponding overlap derivative, with k=0 denoting the value. Derivatives are not divided by factorials.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import math
from numbers import Integral, Real

import numpy as np
from scipy.linalg import eigh

def measurement_pencil_jets(energies: 'np.ndarray', weights: 'np.ndarray', dt: float, dimension: int, batches: int, noise_h: float, noise_s: float) -> 'np.ndarray':
    """Construct the deterministic noisy real-time Krylov pencil and its first two derivatives.
    
    Parameters
    ----------
    energies : np.ndarray, shape (n,)
        Finite real energy levels, n >= 1, in the chosen energy unit.
    weights : np.ndarray, shape (n,)
        Finite nonnegative spectral weights with positive finite sum; normalize once.
    dt : float
        Finite nonnegative time spacing in reciprocal energy units.
    dimension : int
        Number of Krylov columns, between 2 and 32 inclusive; bool is invalid.
    batches : int
        Number of equal-size deterministic batches, between 1 and 64; bool is invalid.
    noise_h, noise_s : float
        Finite nonnegative Hamiltonian and overlap noise amplitudes.
    
    Returns
    -------
    result : np.ndarray, complex, shape (3, 2, batches, dimension, dimension)
        Axis 0 contains value, first derivative and second derivative at x=0.
        Axis 1 contains Hamiltonian and overlap, respectively.
    
    Raises
    ------
    ValueError
        If shapes, finite values, weight sum, scalar signs or integer ranges violate the stated contract.
    
    Notes
    -----
    Set V[a,j] = sqrt(w[a]) exp(-1j*energies[a]*j*dt), H=V^dagger diag(energies) V and S=V^dagger V. For batch q, the Hermitian arrays A, B, C have diagonal A[i,i]=cos((q+1)(i+1)), B[i,i]=sin((q+2)(i+1)), C[i,i]=0. For i<j, set z=1+j-i and A[i,j]=(sin((q+1)(i+j+2))+1j*cos((q+2)(j-i)))/z, B[i,j]=(cos((q+2)(i+j+2))+1j*sin((q+1)(j-i)))/z, C[i,j]=(cos((q+1)(i+j+2))+1j*sin((q+2)(j-i)))/z. The batch pencil is H_q(x)=H+noise_h*(cos(x)*A+sin(x)*B), S_q=S+noise_s*C. Return derivatives, not factorial-divided Taylor coefficients. Inputs are unchanged.
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_measurement_pencil_jets(energies: 'np.ndarray', weights: 'np.ndarray', dt: float, dimension: int, batches: int, noise_h: float, noise_s: float) -> 'np.ndarray':
    import numpy as np
    from numbers import Integral, Real
    try:
        e_raw = np.asarray(energies)
        if np.iscomplexobj(e_raw) and np.any(e_raw.imag != 0):
            raise ValueError('energies must be real-valued; nonzero imaginary parts are invalid.')
        e = np.asarray(e_raw.real if np.iscomplexobj(e_raw) else e_raw, dtype=float)
        w_raw = np.asarray(weights)
        if np.iscomplexobj(w_raw) and np.any(w_raw.imag != 0):
            raise ValueError('weights must be real-valued; nonzero imaginary parts are invalid.')
        w = np.asarray(w_raw.real if np.iscomplexobj(w_raw) else w_raw, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError('Real level energies and weights are required.') from exc
    if e.ndim != 1 or e.size < 1 or w.shape != e.shape or not np.all(np.isfinite(e)) or not np.all(np.isfinite(w)) or np.any(w < 0) or not 0 < w.sum() < np.inf:
        raise ValueError('Energies and nonnegative weights must be finite matching vectors with positive total weight.')
    for v in (dt, noise_h, noise_s):
        if isinstance(v, bool) or not isinstance(v, Real) or not np.isfinite(v) or v < 0:
            raise ValueError('The time spacing and noise scales must be finite and nonnegative.')
    for v, low, high in ((dimension, 2, 32), (batches, 1, 64)):
        if isinstance(v, bool) or not isinstance(v, Integral) or not low <= v <= high:
            raise ValueError('Dimension or batch count is outside its supported integer range.')
    w = w / w.sum()
    v = np.sqrt(w[:, None]) * np.exp(-1j * np.outer(e, np.arange(dimension) * dt))
    h = v.conj().T @ (e[:, None] * v)
    s = v.conj().T @ v
    result = np.zeros((3, 2, batches, dimension, dimension), dtype=complex)
    for q in range(batches):
        a = np.zeros_like(h)
        b = np.zeros_like(h)
        c = np.zeros_like(h)
        for i in range(dimension):
            a[i, i] = np.cos((q + 1) * (i + 1))
            b[i, i] = np.sin((q + 2) * (i + 1))
            for j in range(i + 1, dimension):
                z = 1 + j - i
                a[i, j] = (np.sin((q + 1) * (i + j + 2)) + 1j * np.cos((q + 2) * (j - i))) / z
                b[i, j] = (np.cos((q + 2) * (i + j + 2)) + 1j * np.sin((q + 1) * (j - i))) / z
                c[i, j] = (np.cos((q + 1) * (i + j + 2)) + 1j * np.sin((q + 2) * (j - i))) / z
                a[j, i] = a[i, j].conjugate()
                b[j, i] = b[i, j].conjugate()
                c[j, i] = c[i, j].conjugate()
        result[0, 0, q] = h + noise_h * a
        result[1, 0, q] = noise_h * b
        result[2, 0, q] = -noise_h * a
        result[0, 1, q] = s + noise_s * c
    return result

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
e = np.array([0.2, 0.8, 1.4])
w = np.array([0.3, 0.4, 0.3])
"""

    setup_2 = """import numpy as np
e = np.array([0.7])
w = np.array([3.0])
"""

    setup_3 = """import numpy as np
e = np.array([-2.0, 0.1, 3.0])
w = np.array([2.0, 0.0, 5.0])
"""

    setup_4 = """import numpy as np
e = np.array([0.2, 0.8])
w = np.array([0.5, 0.5])
"""

    setup_5 = """import numpy as np
e = np.array([0.2, 0.8])
w = np.array([0.0, 0.0])
"""

    setup_6 = """import numpy as np
e = np.array([0.2, np.nan])
w = np.array([0.5, 0.5])
"""

    setup_7 = """import numpy as np
e = np.array([0.2, 0.8, 1.4, 2.1, 3.0])
w = np.array([0.26, 0.22, 0.2, 0.18, 0.14])
e = e.astype(complex)
e[0] += 1j
"""

    setup_8 = """import numpy as np
e = np.array([0.2, 0.8, 1.4, 2.1, 3.0])
w = np.array([0.26, 0.22, 0.2, 0.18, 0.14])
w = w.astype(complex)
w[0] += 1j
"""

    cases = [
        # Case 1
        {
            'setup': setup_1,
            'call': '_checked(measurement_pencil_jets, e,w,.6,4,3,.035,.065)',
            'gold_call': '_checked(_oracle_measurement_pencil_jets, e,w,.6,4,3,.035,.065)',
        },
        # Case 2
        {
            'setup': setup_2,
            'call': '_checked(measurement_pencil_jets, e,w,0.,2,1,0.,0.)',
            'gold_call': '_checked(_oracle_measurement_pencil_jets, e,w,0.,2,1,0.,0.)',
        },
        # Case 3
        {
            'setup': setup_3,
            'call': '_checked(measurement_pencil_jets, e,w,.17,6,2,.2,.1)',
            'gold_call': '_checked(_oracle_measurement_pencil_jets, e,w,.17,6,2,.2,.1)',
        },
        # Case 4
        {
            'setup': setup_4,
            'call': '_checked(measurement_pencil_jets, e,w,.1,3,2,0.,.3)',
            'gold_call': '_checked(_oracle_measurement_pencil_jets, e,w,.1,3,2,0.,.3)',
        },
        # Case 5
        {
            'setup': setup_4,
            'call': '_checked(measurement_pencil_jets, e,w,.1,3,2,.3,0.)',
            'gold_call': '_checked(_oracle_measurement_pencil_jets, e,w,.1,3,2,.3,0.)',
        },
        # Case 6
        {
            'setup': setup_5,
            'call': '_checked(measurement_pencil_jets, e, w, 0.1, 3, 2, 0.1, 0.1, _expect_exception=True)',
            'gold_call': '_checked(_oracle_measurement_pencil_jets, e, w, 0.1, 3, 2, 0.1, 0.1, _expect_exception=True)',
        },
        # Case 7
        {
            'setup': setup_4,
            'call': '_checked(measurement_pencil_jets, e, w, 0.1, True, 2, 0.1, 0.1, _expect_exception=True)',
            'gold_call': '_checked(_oracle_measurement_pencil_jets, e, w, 0.1, True, 2, 0.1, 0.1, _expect_exception=True)',
        },
        # Case 8
        {
            'setup': setup_6,
            'call': '_checked(measurement_pencil_jets, e, w, 0.1, 3, 2, 0.1, 0.1, _expect_exception=True)',
            'gold_call': '_checked(_oracle_measurement_pencil_jets, e, w, 0.1, 3, 2, 0.1, 0.1, _expect_exception=True)',
        },
        # Case 9
        {
            'setup': setup_7,
            'call': '_checked(measurement_pencil_jets, e, w, 0.6, 3, 2, 0.035, 0.065, _expect_exception=True)',
            'gold_call': '_checked(_oracle_measurement_pencil_jets, e, w, 0.6, 3, 2, 0.035, 0.065, _expect_exception=True)',
        },
        # Case 10
        {
            'setup': setup_8,
            'call': '_checked(measurement_pencil_jets, e, w, 0.6, 3, 2, 0.035, 0.065, _expect_exception=True)',
            'gold_call': '_checked(_oracle_measurement_pencil_jets, e, w, 0.6, 3, 2, 0.035, 0.065, _expect_exception=True)',
        },
    ]
    for case in cases:
        case['setup'] += check_setup
    return cases
