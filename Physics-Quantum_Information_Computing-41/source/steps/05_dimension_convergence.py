"""
Evaluate the noise-aware adjacent-dimension stopping certificate.

A small shift of the batch mean relative to the larger adjacent batch spread signals noise-limited Krylov growth. The test is made after adding the new state, so a successful certificate retains the current dimension.

Returns
-------
np.ndarray, float, shape (3,), containing the absolute change in adjacent batch means, the effective stopping threshold, and the stopping indicator, in that order; the indicator is 1.0 when the change is strictly below the threshold and 0.0 otherwise, including equality.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import math
from numbers import Integral, Real

import numpy as np
from scipy.linalg import eigh

def dimension_convergence(previous: 'np.ndarray', current: 'np.ndarray', gamma: float, energy_tolerance: float) -> 'np.ndarray':
    """Evaluate the noise-aware adjacent-dimension stopping certificate.
    
    Parameters
    ----------
    previous, current : np.ndarray, shape (3,)
        Consecutive population triples (mean, standard deviation, variance).
        Entries are finite; dispersion entries are nonnegative.
    gamma : float
        Finite nonnegative multiplier of the larger adjacent standard deviation.
    energy_tolerance : float
        Finite nonnegative independent absolute energy-change threshold.
    
    Returns
    -------
    result : np.ndarray, float, shape (3,)
        Absolute mean change, effective threshold, and a 0.0/1.0 stopping indicator.
    
    Raises
    ------
    ValueError
        If moment shapes, finite values, dispersion signs or parameter domains fail, or the threshold is not representable.
    
    Notes
    -----
    The effective threshold is max(energy_tolerance, gamma*max(previous standard deviation, current standard deviation)). Stop only when the absolute mean change is strictly smaller. Equality continues growth. Moment triples are inputs; no statistical consistency between their final two entries needs to be re-estimated.
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_dimension_convergence(previous: 'np.ndarray', current: 'np.ndarray', gamma: float, energy_tolerance: float) -> 'np.ndarray':
    import numpy as np
    from numbers import Real
    try:
        p_raw = np.asarray(previous)
        if np.iscomplexobj(p_raw) and np.any(p_raw.imag != 0):
            raise ValueError('previous must be real-valued; nonzero imaginary parts are invalid.')
        p = np.asarray(p_raw.real if np.iscomplexobj(p_raw) else p_raw, dtype=float)
        q_raw = np.asarray(current)
        if np.iscomplexobj(q_raw) and np.any(q_raw.imag != 0):
            raise ValueError('current must be real-valued; nonzero imaginary parts are invalid.')
        q = np.asarray(q_raw.real if np.iscomplexobj(q_raw) else q_raw, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError('Numerical moment triples are required.') from exc
    if p.shape != (3,) or q.shape != (3,) or not np.all(np.isfinite(p)) or not np.all(np.isfinite(q)) or np.any(p[1:] < 0) or np.any(q[1:] < 0):
        raise ValueError('Each moment triple must contain a finite mean and nonnegative dispersion entries.')
    for a in (gamma, energy_tolerance):
        if isinstance(a, bool) or not isinstance(a, Real) or not np.isfinite(a) or a < 0:
            raise ValueError('The convergence parameters must be finite and nonnegative.')
    difference = abs(float(q[0] - p[0]))
    threshold = max(float(energy_tolerance), float(gamma) * max(float(p[1]), float(q[1])))
    if not np.isfinite(threshold):
        raise ValueError('The convergence threshold must be representable.')
    return np.array([difference, threshold, float(difference < threshold)])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return deterministic cases with oracle-independent input setup."""
    exception_setup = """def _exception_code(function, *args, **kwargs):
    try:
        function(*args, **kwargs)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
"""

    setup_1 = """from copy import deepcopy
import numpy as np
p = np.array([0.0, 0.1, 0.01])
q = np.array([0.08, 0.05, 0.0025])
"""

    setup_2 = """from copy import deepcopy
import numpy as np
p = np.array([0.0, 0.1, 0.01])
q = np.array([0.2, 0.05, 0.0025])
"""

    setup_3 = """from copy import deepcopy
import numpy as np
p = np.array([0.0, 0.25, 0.0625])
q = np.array([0.25, 0.125, 0.015625])
"""

    setup_4 = """from copy import deepcopy
import numpy as np
p = np.array([0.0, 0.0, 0.0])
q = np.array([0.0, 0.0, 0.0])
"""

    setup_5 = """from copy import deepcopy
import numpy as np
p = np.array([0.0, 0.0, 0.0])
q = np.array([0.01, 0.0, 0.0])
"""

    setup_6 = """from copy import deepcopy
import numpy as np
p = np.array([0.0, 0.1, 0.01])
q = np.array([0.22, 0.25, 0.0625])
"""

    setup_7 = """from copy import deepcopy
import numpy as np
p = np.array([0.0, -0.1, 0.01])
q = np.zeros(3)
"""

    setup_8 = """from copy import deepcopy
import numpy as np
p = np.zeros(3)
q = np.zeros(3)
"""

    setup_9 = """from copy import deepcopy
import numpy as np
p = np.array([1 + 2j, 0.2, 0.04])
q = np.array([1.0, 0.2, 0.04])
"""

    setup_10 = """from copy import deepcopy
import numpy as np
p = np.array([1.0, 0.2, 0.04])
q = np.array([1 + 2j, 0.2, 0.04])
"""

    return [
        # Case 1
        {
            'setup': setup_1,
            'call': 'dimension_convergence(*deepcopy((p, q, 1.0, 0.0)))',
            'gold_call': '_oracle_dimension_convergence(*deepcopy((p, q, 1.0, 0.0)))',
        },
        # Case 2
        {
            'setup': setup_2,
            'call': 'dimension_convergence(*deepcopy((p, q, 1.0, 0.0)))',
            'gold_call': '_oracle_dimension_convergence(*deepcopy((p, q, 1.0, 0.0)))',
        },
        # Case 3
        {
            'setup': setup_3,
            'call': 'dimension_convergence(*deepcopy((p, q, 1.0, 0.0)))',
            'gold_call': '_oracle_dimension_convergence(*deepcopy((p, q, 1.0, 0.0)))',
        },
        # Case 4
        {
            'setup': setup_4,
            'call': 'dimension_convergence(*deepcopy((p, q, 0.0, 0.0)))',
            'gold_call': '_oracle_dimension_convergence(*deepcopy((p, q, 0.0, 0.0)))',
        },
        # Case 5
        {
            'setup': setup_5,
            'call': 'dimension_convergence(*deepcopy((p, q, 0.0, 0.02)))',
            'gold_call': '_oracle_dimension_convergence(*deepcopy((p, q, 0.0, 0.02)))',
        },
        # Case 6
        {
            'setup': setup_6,
            'call': 'dimension_convergence(*deepcopy((p, q, 1.0, 0.0)))',
            'gold_call': '_oracle_dimension_convergence(*deepcopy((p, q, 1.0, 0.0)))',
        },
        # Case 7
        {
            'setup': setup_7 + exception_setup,
            'call': '_exception_code(dimension_convergence, *deepcopy((p, q, 1.0, 0.0)))',
            'gold_call': '_exception_code(_oracle_dimension_convergence, *deepcopy((p, q, 1.0, 0.0)))',
        },
        # Case 8
        {
            'setup': setup_8 + exception_setup,
            'call': '_exception_code(dimension_convergence, *deepcopy((p, q, -1.0, 0.0)))',
            'gold_call': '_exception_code(_oracle_dimension_convergence, *deepcopy((p, q, -1.0, 0.0)))',
        },
        # Case 9
        {
            'setup': setup_9 + exception_setup,
            'call': '_exception_code(dimension_convergence, *deepcopy((p, q, 1.0, 0.0)))',
            'gold_call': '_exception_code(_oracle_dimension_convergence, *deepcopy((p, q, 1.0, 0.0)))',
        },
        # Case 10
        {
            'setup': setup_10 + exception_setup,
            'call': '_exception_code(dimension_convergence, *deepcopy((p, q, 1.0, 0.0)))',
            'gold_call': '_exception_code(_oracle_dimension_convergence, *deepcopy((p, q, 1.0, 0.0)))',
        },
    ]
