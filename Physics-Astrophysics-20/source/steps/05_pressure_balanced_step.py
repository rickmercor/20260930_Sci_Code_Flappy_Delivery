"""
One time step of the pressure-balanced slab.

One time step of the pressure-balanced slab.

The elements of the slab of step 04 evolve by the local physics of step 03 (radiative
sinks and firehose-regulated scattering at fixed density and field, under the schedule
of step 02) and by the reversible compression that keeps the slab pressure-balanced
(step 04). One time step of size dt from time t is the fixed splitting used throughout
this task: advance every element with step 03 from t to t + dt/2, map the result
through step 04 at fixed element masses and slab length, then advance every element
with step 03 from t + dt/2 to t + dt.

Returns
-------
The element densities and pressures after the step and the common total pressure of
its balance, stacked.

Returns
-------
The element densities and pressures after the step and the common total pressure of its balance, stacked.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def pressure_balanced_step(
    mass: ArrayLike,
    n: ArrayLike,
    p_perp: ArrayLike,
    p_par: ArrayLike,
    t: float,
    dt: float,
    L: float,
    beta0: float,
    theta0: float,
    chi: float,
    t_d: float,
    c_th: float = 1.4,
) -> np.ndarray:
    """Return the slab elements after one splitting time step.

    Parameters
    ----------
    mass, n, p_perp, p_par : array_like
        Element masses, densities and pressures at time ``t``, as in step 04.
    t : float
        Time at the start of the step (units 1/Omega0). Finite.
    dt : float
        Step size (units 1/Omega0). Finite and non-negative.
    L : float
        Slab length (step 04). Strictly positive.
    beta0 : float
        Reference plasma beta (step 01). Strictly positive.
    theta0 : float
        Reference temperature T0 / (m_e c^2) (step 01). Strictly positive.
    chi : float
        Reference product tau0 Omega0 (step 01). Strictly positive.
    t_d : float
        Onset time of the schedule of step 02 (units 1/Omega0). Finite.
    c_th : float, optional
        Firehose threshold constant of step 02. Strictly positive.

    Returns
    -------
    numpy.ndarray
        Array of shape (4, N). Rows 0, 1 and 2 hold n, P_perp and P_par of
        every element at ``t + dt``; row 3 holds, in every entry, the common
        total perpendicular pressure of the step-04 map taken inside the
        step. The field of an element is B = B0 n. For ``dt = 0`` the rows
        are the pressure-balanced state of the input.

    Raises
    ------
    ValueError
        For any input that step 04 rejects, if ``t`` or ``t_d`` is not a finite
        scalar, if ``dt`` is negative or not a finite scalar, or if ``chi`` or
        ``c_th`` is not a finite, strictly positive scalar.
    """
    return state  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import math

import numpy as np
from numpy.typing import ArrayLike


def _oracle_pressure_balanced_step(
    mass: ArrayLike,
    n: ArrayLike,
    p_perp: ArrayLike,
    p_par: ArrayLike,
    t: float,
    dt: float,
    L: float,
    beta0: float,
    theta0: float,
    chi: float,
    t_d: float,
    c_th: float = 1.4,
) -> np.ndarray:
    m, nn, p, q = _pb_state_args(mass, n, p_perp, p_par)
    t0 = _fr_finite("t", t)
    h = _fr_finite("dt", dt)
    if h < 0.0:
        raise ValueError("dt must be non-negative")
    length = _rs_scalar("L", L)
    beta0 = _rs_scalar("beta0", beta0)
    theta0 = _rs_scalar("theta0", theta0)
    chi = _rs_scalar("chi", chi)
    td = _fr_finite("t_d", t_d)
    cth = _rs_scalar("c_th", c_th)
    b0 = math.sqrt(2.0 * theta0 / beta0)
    half = 0.5 * h
    pq = _oracle_advance_uniform_cells(nn, b0 * nn, p, q, t0, t0 + half, beta0, theta0, chi, td, cth)
    n1, p1, q1, pi = _pb_project(m, nn, pq[0], pq[1], length, b0)
    pq = _oracle_advance_uniform_cells(n1, b0 * n1, p1, q1, t0 + half, t0 + h, beta0, theta0, chi, td, cth)
    return np.stack((n1, pq[0], pq[1], np.full(n1.size, pi)))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return cases with independent candidate and oracle argument graphs."""
    return [{'setup': 'import numpy as np\n'
               'b = np.array([0.72, 0.9, 1.1, 1.33, 1.05])\n'
               'm = b * 120.0\n'
               'p = 1000.0 * (1.0 + (1.0 - b * b) / 40.0)\n'
               'q = p * np.array([1.0, 1.01, 1.02, 1.03, 1.0])\n'
               'import copy\n'
               'def _independent_call(function, *args, **kwargs):\n'
               '    args, kwargs = copy.deepcopy((args, kwargs))\n'
               '    return function(*args, **kwargs)\n',
      'call': '_independent_call(pressure_balanced_step, m, b, p, q, 300.0, 80.0, 600.0, 40.0, 1000.0, '
              '2000.0, 340.0)',
      'gold_call': '_independent_call(_oracle_pressure_balanced_step, m, b, p, q, 300.0, 80.0, 600.0, '
                   '40.0, 1000.0, 2000.0, 340.0)'},
     {'setup': 'import numpy as np\n'
               'b = np.array([0.72, 0.9, 1.1, 1.33, 1.05])\n'
               'm = b * 120.0\n'
               'p = 1000.0 * (1.0 + (1.0 - b * b) / 40.0)\n'
               'q = p * np.array([1.0, 1.01, 1.02, 1.03, 1.0])\n'
               'import copy\n'
               'def _independent_call(function, *args, **kwargs):\n'
               '    args, kwargs = copy.deepcopy((args, kwargs))\n'
               '    return function(*args, **kwargs)\n',
      'call': '_independent_call(pressure_balanced_step, m, b, p, q, 500.0, 0.0, 600.0, 40.0, 1000.0, '
              '2000.0, 340.0)',
      'gold_call': '_independent_call(_oracle_pressure_balanced_step, m, b, p, q, 500.0, 0.0, 600.0, '
                   '40.0, 1000.0, 2000.0, 340.0)'},
     {'setup': 'import numpy as np\n'
               'n = np.array([0.6, 1.0, 1.7])\n'
               'm = np.array([0.3, 0.5, 0.9])\n'
               'p = np.array([8.0, 6.0, 3.0])\n'
               'q = np.array([9.5, 8.0, 3.2])\n'
               'import copy\n'
               'def _independent_call(function, *args, **kwargs):\n'
               '    args, kwargs = copy.deepcopy((args, kwargs))\n'
               '    return function(*args, **kwargs)\n',
      'call': '_independent_call(pressure_balanced_step, m, n, p, q, 30.0, 25.0, 1.4, 10.0, 10.0, '
              '200.0, 34.0)',
      'gold_call': '_independent_call(_oracle_pressure_balanced_step, m, n, p, q, 30.0, 25.0, 1.4, '
                   '10.0, 10.0, 200.0, 34.0)'},
     {'setup': 'import numpy as np\n'
               'def _probe_call():\n'
               '    try:\n'
               '        pressure_balanced_step(np.ones(3), np.ones(3), np.ones(3), np.ones(3), 0.0, '
               '-1.0, 3.0, 40.0, 1.0, 2000.0, 0.0)\n'
               '    except ValueError:\n'
               '        return 1\n'
               '    except Exception:\n'
               '        return 2\n'
               '    return 0\n'
               'def _probe_gold():\n'
               '    try:\n'
               '        _oracle_pressure_balanced_step(np.ones(3), np.ones(3), np.ones(3), np.ones(3), '
               '0.0, -1.0, 3.0, 40.0, 1.0, 2000.0, 0.0)\n'
               '    except ValueError:\n'
               '        return 1\n'
               '    except Exception:\n'
               '        return 2\n'
               '    return 0\n',
      'call': '_probe_call()',
      'gold_call': '_probe_gold()'}]
