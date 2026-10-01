"""
Select a rotation from the measurement-only variance objective.

Different rotations retain different noisy directions. The experimental heuristic compares equal-batch energy consistency across the supplied angle grid, without requiring a reference ground energy or maximizing retained rank.

Returns
-------
np.ndarray, float, shape (3,), containing the zero-based selected row index represented as a float, its angle in radians, and its population energy variance, in that order; the selected row minimizes population variance over the supplied grid, with the earliest supplied row winning an exact tie.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import math
from numbers import Integral, Real

import numpy as np
from scipy.linalg import eigh

def select_variance_angle(angles: 'np.ndarray', batch_energies: 'np.ndarray') -> 'np.ndarray':
    """Select a rotation from the measurement-only variance objective.
    
    Parameters
    ----------
    angles : np.ndarray, shape (m,)
        Nonempty finite real grid in its supplied order; sorting is not implied.
    batch_energies : np.ndarray, shape (m, b)
        Finite physical ground energies, with b>=1 equally weighted batches per angle.
    
    Returns
    -------
    result : np.ndarray, float, shape (3,)
        Zero-based selected row index, its angle, and its population variance.
    
    Raises
    ------
    ValueError
        If shapes, finite values or nonempty requirements fail, or a variance is not representable.
    
    Notes
    -----
    Select the global minimum population variance, with the earliest supplied row winning an exact tie. A single batch gives zero variance for every angle. Preserve both inputs.
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_select_variance_angle(angles: 'np.ndarray', batch_energies: 'np.ndarray') -> 'np.ndarray':
    import numpy as np
    try:
        a_raw = np.asarray(angles)
        if np.iscomplexobj(a_raw) and np.any(a_raw.imag != 0):
            raise ValueError('angles must be real-valued; nonzero imaginary parts are invalid.')
        a = np.asarray(a_raw.real if np.iscomplexobj(a_raw) else a_raw, dtype=float)
        e_raw = np.asarray(batch_energies)
        if np.iscomplexobj(e_raw) and np.any(e_raw.imag != 0):
            raise ValueError('batch_energies must be real-valued; nonzero imaginary parts are invalid.')
        e = np.asarray(e_raw.real if np.iscomplexobj(e_raw) else e_raw, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError('Real angles and energies are required.') from exc
    if a.ndim != 1 or a.size == 0 or e.ndim != 2 or e.shape[0] != a.size or e.shape[1] == 0 or not np.all(np.isfinite(a)) or not np.all(np.isfinite(e)):
        raise ValueError('Angles and batch-energy rows must be finite, nonempty and matched.')
    centered = e - np.mean(e, axis=1)[:, None]
    variances = np.mean(centered * centered, axis=1)
    if not np.all(np.isfinite(variances)):
        raise ValueError('Each population variance must be representable.')
    index = int(np.argmin(variances))
    return np.array([float(index), a[index], variances[index]], dtype=float)

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
a = np.array([0.7, 0.0, 0.4])
e = np.array([[1.0, 1.2, 0.8], [2.0, 2.0, 2.0], [0.0, 0.3, -0.3]])
"""

    setup_2 = """import numpy as np
a = np.array([1.2, 0.2, 0.0])
e = np.ones((3, 1))
"""

    setup_3 = """import numpy as np
a = np.array([0.3])
e = np.array([[2.0, 3.0]])
"""

    setup_4 = """import numpy as np
a = np.array([0.0, 0.3, 0.6, 0.9])
e = np.array([[-1.0, 1.0], [-0.1, 0.1], [-3.0, 3.0], [-2.0, 2.0]])
"""

    setup_5 = """import numpy as np
a = np.array([0.9, 0.1])
e = np.array([[1000000000.0 - 1, 1000000000.0 + 1], [1000000000.0 - 2, 1000000000.0 + 2]])
"""

    setup_6 = """import numpy as np
a = np.array([])
e = np.empty((0, 2))
"""

    setup_7 = """import numpy as np
a = np.array([0.0, 1.0])
e = np.ones((3, 2))
"""

    setup_8 = """import numpy as np
a = np.array([0 + 1j, 0.2])
e = np.array([[1.0, 2.0], [3.0, 4.0]])
"""

    setup_9 = """import numpy as np
a = np.array([0.0, 0.2])
e = np.array([[1 + 2j, 2.0], [3.0, 4.0]])
"""

    cases = [
        # Case 1
        {
            'setup': setup_1,
            'call': '_checked(select_variance_angle, a,e)',
            'gold_call': '_checked(_oracle_select_variance_angle, a,e)',
        },
        # Case 2
        {
            'setup': setup_2,
            'call': '_checked(select_variance_angle, a,e)',
            'gold_call': '_checked(_oracle_select_variance_angle, a,e)',
        },
        # Case 3
        {
            'setup': setup_3,
            'call': '_checked(select_variance_angle, a,e)',
            'gold_call': '_checked(_oracle_select_variance_angle, a,e)',
        },
        # Case 4
        {
            'setup': setup_4,
            'call': '_checked(select_variance_angle, a,e)',
            'gold_call': '_checked(_oracle_select_variance_angle, a,e)',
        },
        # Case 5
        {
            'setup': setup_5,
            'call': '_checked(select_variance_angle, a,e)',
            'gold_call': '_checked(_oracle_select_variance_angle, a,e)',
        },
        # Case 6
        {
            'setup': setup_6,
            'call': '_checked(select_variance_angle, a, e, _expect_exception=True)',
            'gold_call': '_checked(_oracle_select_variance_angle, a, e, _expect_exception=True)',
        },
        # Case 7
        {
            'setup': setup_7,
            'call': '_checked(select_variance_angle, a, e, _expect_exception=True)',
            'gold_call': '_checked(_oracle_select_variance_angle, a, e, _expect_exception=True)',
        },
        # Case 8
        {
            'setup': setup_8,
            'call': '_checked(select_variance_angle, a, e, _expect_exception=True)',
            'gold_call': '_checked(_oracle_select_variance_angle, a, e, _expect_exception=True)',
        },
        # Case 9
        {
            'setup': setup_9,
            'call': '_checked(select_variance_angle, a, e, _expect_exception=True)',
            'gold_call': '_checked(_oracle_select_variance_angle, a, e, _expect_exception=True)',
        },
    ]
    for case in cases:
        case['setup'] += check_setup
    return cases
