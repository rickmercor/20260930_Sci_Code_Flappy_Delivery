"""
Evaluate equal-batch population energy statistics.

The noise-aware growth and angle criteria use the spread of batch estimates, rather than the standard error of their mean. Centering before squaring preserves small physical spreads on top of a large common energy shift.

Returns
-------
np.ndarray, float, shape (3,), containing the equal-weight batch population mean, population standard deviation, and population variance, in that order.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import math
from numbers import Integral, Real

import numpy as np
from scipy.linalg import eigh

def batch_energy_moments(energies: 'np.ndarray') -> 'np.ndarray':
    """Evaluate equal-batch population energy statistics.
    
    Parameters
    ----------
    energies : np.ndarray, shape (b,)
        Nonempty finite real vector of equal-weight batch ground energies.
    
    Returns
    -------
    result : np.ndarray, float, shape (3,)
        Population mean, population standard deviation and population variance, in that order.
    
    Raises
    ------
    ValueError
        If energies is not a nonempty finite real vector or its population variance is not representable.
    
    Notes
    -----
    Use the population divisor b, including for b=1, whose dispersion is zero. Preserve the input.
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_batch_energy_moments(energies: 'np.ndarray') -> 'np.ndarray':
    import numpy as np
    try:
        e_raw = np.asarray(energies)
        if np.iscomplexobj(e_raw) and np.any(e_raw.imag != 0):
            raise ValueError('energies must be real-valued; nonzero imaginary parts are invalid.')
        e = np.asarray(e_raw.real if np.iscomplexobj(e_raw) else e_raw, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError('A real energy vector is required.') from exc
    if e.ndim != 1 or e.size == 0 or not np.all(np.isfinite(e)):
        raise ValueError('The batch energies must form a nonempty finite vector.')
    mean = float(np.mean(e))
    centered = e - mean
    variance = float(np.mean(centered * centered))
    if not np.isfinite(variance):
        raise ValueError('The population variance must be representable.')
    return np.array([mean, np.sqrt(variance), variance], dtype=float)

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
e = np.array([0.2, 0.5, -0.1, 1.2])
"""

    setup_2 = """import numpy as np
e = np.array([3.0])
"""

    setup_3 = """import numpy as np
e = np.array([7.0, 7.0, 7.0])
"""

    setup_4 = """import numpy as np
e = 1000000000.0 + np.array([-2.0, -1.0, 1.0, 2.0])
"""

    setup_5 = """import numpy as np
e = np.array([])
"""

    setup_6 = """import numpy as np
e = np.array([1.0, np.inf])
"""

    setup_7 = """import numpy as np
e = np.array([1 + 2j, 3 + 4j])
"""

    cases = [
        # Case 1
        {
            'setup': setup_1,
            'call': '_checked(batch_energy_moments, e)',
            'gold_call': '_checked(_oracle_batch_energy_moments, e)',
        },
        # Case 2
        {
            'setup': setup_2,
            'call': '_checked(batch_energy_moments, e)',
            'gold_call': '_checked(_oracle_batch_energy_moments, e)',
        },
        # Case 3
        {
            'setup': setup_3,
            'call': '_checked(batch_energy_moments, e)',
            'gold_call': '_checked(_oracle_batch_energy_moments, e)',
        },
        # Case 4
        {
            'setup': setup_4,
            'call': '_checked(batch_energy_moments, e)',
            'gold_call': '_checked(_oracle_batch_energy_moments, e)',
            'tol': 1e-07,
        },
        # Case 5
        {
            'setup': setup_5,
            'call': '_checked(batch_energy_moments, e, _expect_exception=True)',
            'gold_call': '_checked(_oracle_batch_energy_moments, e, _expect_exception=True)',
        },
        # Case 6
        {
            'setup': setup_6,
            'call': '_checked(batch_energy_moments, e, _expect_exception=True)',
            'gold_call': '_checked(_oracle_batch_energy_moments, e, _expect_exception=True)',
        },
        # Case 7
        {
            'setup': setup_7,
            'call': '_checked(batch_energy_moments, e, _expect_exception=True)',
            'gold_call': '_checked(_oracle_batch_energy_moments, e, _expect_exception=True)',
        },
    ]
    for case in cases:
        case['setup'] += check_setup
    return cases
