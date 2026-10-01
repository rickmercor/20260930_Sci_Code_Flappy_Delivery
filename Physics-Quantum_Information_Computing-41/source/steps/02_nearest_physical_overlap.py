"""
Find the nearest physical Gram matrix of normalized states.

The rotation mixes the Hamiltonian with the overlap before thresholding, so physical overlap constraints must already hold. The Frobenius projection is onto the intersection of the Hermitian positive-semidefinite cone and the unit-diagonal affine set; a single eigenvalue clipping followed by diagonal rescaling is generally a different problem.

Returns
-------
np.ndarray, complex, shape (n, n), the Frobenius-nearest Hermitian positive-semidefinite overlap matrix with unit diagonal, to the requested numerical tolerance.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import math
from numbers import Integral, Real

import numpy as np
from scipy.linalg import eigh

def nearest_physical_overlap(overlap: 'np.ndarray', tolerance: float = 1e-13, max_iterations: int = 2000) -> 'np.ndarray':
    """Find the nearest physical Gram matrix of normalized states.
    
    Parameters
    ----------
    overlap : np.ndarray, shape (n, n)
        Nonempty finite Hermitian matrix; Hermiticity tolerance is 1e-10 absolute.
    tolerance : float, optional
        Convergence tolerance in (0, 1e-6], default 1e-13.
    max_iterations : int, optional
        Positive iteration budget, default 2000; bool is invalid.
    
    Returns
    -------
    result : np.ndarray, complex, shape (n, n)
        The Frobenius-nearest Hermitian positive-semidefinite matrix with unit diagonal,
        to the requested numerical tolerance.
    
    Raises
    ------
    ValueError
        If the input or convergence parameters violate the contract, or the constrained projection fails to converge within the budget.
    
    Notes
    -----
    Use the converged convex projection, preserving the input. For a bounded reproducible implementation, alternating cone and unit-diagonal projections with the cone correction retained may be used: start from the input and zero correction, project the corrected affine iterate onto the cone, update the correction, then impose unit diagonal. The stopping residual is the maximum of successive cone-iterate difference, successive affine-iterate difference and their mutual difference, divided by max(1, norm of the affine iterate). All norms are Frobenius norms. Return the affine iterate once this residual is at most tolerance. Equivalent converged convex solutions are accepted; do not add a final clipping or diagonal normalization that changes the minimizer.
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_nearest_physical_overlap(overlap: 'np.ndarray', tolerance: float = 1e-13, max_iterations: int = 2000) -> 'np.ndarray':
    import numpy as np
    from numbers import Integral, Real
    try:
        a = np.array(overlap, dtype=complex, copy=True)
    except (TypeError, ValueError) as exc:
        raise ValueError('A numerical overlap matrix is required.') from exc
    if a.ndim != 2 or a.shape[0] != a.shape[1] or a.shape[0] < 1 or not np.all(np.isfinite(a)) or not np.allclose(a, a.conj().T, atol=1e-10, rtol=0):
        raise ValueError('The overlap must be finite, square and Hermitian.')
    if isinstance(tolerance, bool) or not isinstance(tolerance, Real) or not np.isfinite(tolerance) or not 0 < tolerance <= 1e-6:
        raise ValueError('Tolerance must lie in (0, 1e-6].')
    if isinstance(max_iterations, bool) or not isinstance(max_iterations, Integral) or max_iterations < 1:
        raise ValueError('The iteration limit must be a positive integer.')
    y = (a + a.conj().T) / 2
    x = y.copy()
    correction = np.zeros_like(y)
    for _ in range(max_iterations):
        old_x, old_y = x, y
        r = y - correction
        values, vectors = np.linalg.eigh((r + r.conj().T) / 2)
        x = (vectors * np.maximum(values, 0)) @ vectors.conj().T
        correction = x - r
        y = x.copy()
        np.fill_diagonal(y, 1.0)
        scale = max(1.0, np.linalg.norm(y, 'fro'))
        residual = max(np.linalg.norm(x - old_x, 'fro'), np.linalg.norm(y - old_y, 'fro'), np.linalg.norm(y - x, 'fro')) / scale
        if residual <= tolerance:
            return (y + y.conj().T) / 2
    raise ValueError('Physical-overlap projection did not converge within the iteration limit.')

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
s = np.array([[1, 1.1 + 0.4j, 0.8], [1.1 - 0.4j, 1, -0.5j], [0.8, 0.5j, 1]], complex)
"""

    setup_2 = """import numpy as np
s = np.array([[2.5]], complex)
"""

    setup_3 = """import numpy as np
s = np.array([[1.0, 0.2j], [-0.2j, 1.0]])
"""

    setup_4 = """import numpy as np
s = np.array([[1, 1j], [-1j, 1]], complex)
"""

    setup_5 = """import numpy as np
s = np.array([[3, 1.5 + 0.8j], [1.5 - 0.8j, -0.4]], complex)
"""

    setup_6 = """import numpy as np
s = np.ones((2, 3))
"""

    setup_7 = """import numpy as np
s = np.array([[1.0, 0.4], [0.1, 1.0]])
"""

    setup_8 = """import numpy as np
s = np.array([[1.0, 1.5], [1.5, 1.0]])
"""

    setup_9 = """import numpy as np
s = np.eye(2)
"""

    cases = [
        # Case 1
        {
            'setup': setup_1,
            'call': '_checked(nearest_physical_overlap, s)',
            'gold_call': '_checked(_oracle_nearest_physical_overlap, s)',
        },
        # Case 2
        {
            'setup': setup_2,
            'call': '_checked(nearest_physical_overlap, s)',
            'gold_call': '_checked(_oracle_nearest_physical_overlap, s)',
        },
        # Case 3
        {
            'setup': setup_3,
            'call': '_checked(nearest_physical_overlap, s)',
            'gold_call': '_checked(_oracle_nearest_physical_overlap, s)',
        },
        # Case 4
        {
            'setup': setup_4,
            'call': '_checked(nearest_physical_overlap, s)',
            'gold_call': '_checked(_oracle_nearest_physical_overlap, s)',
        },
        # Case 5
        {
            'setup': setup_5,
            'call': '_checked(nearest_physical_overlap, s)',
            'gold_call': '_checked(_oracle_nearest_physical_overlap, s)',
        },
        # Case 6
        {
            'setup': setup_6,
            'call': '_checked(nearest_physical_overlap, s, _expect_exception=True)',
            'gold_call': '_checked(_oracle_nearest_physical_overlap, s, _expect_exception=True)',
        },
        # Case 7
        {
            'setup': setup_7,
            'call': '_checked(nearest_physical_overlap, s, _expect_exception=True)',
            'gold_call': '_checked(_oracle_nearest_physical_overlap, s, _expect_exception=True)',
        },
        # Case 8
        {
            'setup': setup_8,
            'call': '_checked(nearest_physical_overlap, s, max_iterations=0, _expect_exception=True)',
            'gold_call': '_checked(_oracle_nearest_physical_overlap, s, max_iterations=0, _expect_exception=True)',
        },
        # Case 9
        {
            'setup': setup_9,
            'call': '_checked(nearest_physical_overlap, s, tolerance=0.0, _expect_exception=True)',
            'gold_call': '_checked(_oracle_nearest_physical_overlap, s, tolerance=0.0, _expect_exception=True)',
        },
    ]
    for case in cases:
        case['setup'] += check_setup
    return cases
