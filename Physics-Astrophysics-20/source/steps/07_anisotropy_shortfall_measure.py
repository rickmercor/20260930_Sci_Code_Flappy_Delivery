"""
End-to-end benchmark value (final orchestrator step).

End-to-end benchmark value (final orchestrator step).

Evolve the benchmark slab of step 06 from its fixed two-mode seed to the final time and
report the anisotropy-shortfall measure Q: the width-weighted average over the slab of
the squared fractional distance of P_par - P_perp below the firehose threshold.
Elements held on (or above) the threshold contribute nothing; elements below it
contribute according to how far below it they sit.

Returns
-------
The benchmark anisotropy-shortfall measure Q.

Returns
-------
The benchmark anisotropy-shortfall measure Q.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def anisotropy_shortfall_measure(
    N: int = 512,
    t_end: float = 16000.0,
    L: float = 1.0,
    beta0: float = 80.0,
    theta0: float = 1000.0,
    chi: float = 2000.0,
    t_d: float = 340.0,
    c_th: float = 1.4,
    dt_max: float = 0.5,
) -> float:
    """Return the benchmark anisotropy-shortfall measure Q of the slab.

    The slab of step 06 is evolved from the fixed seed with rows
    (amplitude, k, phase) = (-0.28, 1, 0) and (0.05, 2, pi/3) to ``t_end``,
    and

        Q = (1/L) sum_i w_i [max(0, 1 - (P_par,i - P_perp,i) / (c_th B_i^2 / 2))]^2

    is returned, with w_i the element widths and B_i = B0 n_i the element
    fields of the final state.

    Parameters
    ----------
    N : int, optional
        Number of elements of step 06; at least 4.
    t_end : float, optional
        Final time (units 1/Omega0); non-negative.
    L : float, optional
        Slab length (any length unit). Strictly positive.
    beta0 : float, optional
        Reference plasma beta (step 01). Strictly positive.
    theta0 : float, optional
        Reference temperature T0 / (m_e c^2) (step 01). Strictly positive.
    chi : float, optional
        Reference product tau0 Omega0 (step 01). Strictly positive.
    t_d : float, optional
        Onset time of the schedule of step 02 (units 1/Omega0). Finite.
    c_th : float, optional
        Firehose threshold constant of step 02. Strictly positive.
    dt_max : float, optional
        Largest step size of step 06 (units 1/Omega0). Strictly positive.

    Returns
    -------
    float
        The anisotropy-shortfall measure Q (dimensionless, between 0 and 1).

    Raises
    ------
    ValueError
        For any argument that step 06 rejects, or if the final state fails one
        of these consistency checks against the earlier steps: the element
        widths add up to ``L`` within a relative 1e-9; mapping the state through
        step 04 changes no density by more than a relative 1e-2; a zero-length
        step of step 05 reproduces that mapped state within a relative 1e-12;
        the radiative sinks of step 01 remove energy (2 dP_perp + dP_par < 0) in
        every element; step 02 gives no scattering (rate exactly 0) in any
        element lying below the threshold by more than a relative 1e-10 at
        du_dx = 0; and a unit-time advance by step 03 does not increase
        2 P_perp + P_par in any element (relative slack 1e-12).
    """
    return 0.0

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import math

import numpy as np
from numpy.typing import ArrayLike


def _oracle_anisotropy_shortfall_measure(
    N: int = 512,
    t_end: float = 16000.0,
    L: float = 1.0,
    beta0: float = 80.0,
    theta0: float = 1000.0,
    chi: float = 2000.0,
    t_d: float = 340.0,
    c_th: float = 1.4,
    dt_max: float = 0.5,
) -> float:
    seed = np.array([[-0.28, 1.0, 0.0], [0.05, 2.0, math.pi / 3.0]])
    n, p, q, w = _oracle_evolve_pressure_balanced_slab(N, t_end, L, seed, beta0, theta0, chi, t_d,
                                                       c_th, dt_max)
    length = float(L)
    tf = float(t_end)
    if abs(float(np.sum(w)) - length) > 1e-9 * length:
        raise ValueError("consistency gate: the element widths no longer fill the slab")
    mass = n * w
    b0 = math.sqrt(2.0 * float(theta0) / float(beta0))
    B = b0 * n
    bal = _oracle_pressure_balanced_state(mass, n, p, q, L, beta0, theta0)
    if np.max(np.abs(bal[0] / n - 1.0)) > 1e-2:
        raise ValueError("consistency gate: the final state is far from pressure balance")
    zero = _oracle_pressure_balanced_step(mass, n, p, q, tf, 0.0, L, beta0, theta0, chi, t_d, c_th)
    if np.max(np.abs(zero[:3] / bal[:3] - 1.0)) > 1e-12:
        raise ValueError("consistency gate: a zero-length step differs from the balanced state")
    ap, aq = (0.9, 0.48) if tf >= float(t_d) else (16.0 / 15.0, 8.0 / 15.0)
    sinks = _oracle_radiative_pressure_sinks(n, B, p, q, ap, aq, beta0, theta0, chi)
    if not np.all(2.0 * sinks[0] + sinks[1] < 0.0):
        raise ValueError("consistency gate: radiation does not remove energy")
    thr = 0.5 * float(c_th) * B * B
    nu = _oracle_regulated_scattering_rate(n, B, p, q, 0.0, tf, beta0, theta0, chi, t_d, c_th)
    below = (q - p) < thr * (1.0 - 1e-10)
    if np.any(nu[below] != 0.0):
        raise ValueError("consistency gate: an element below the threshold is scattering")
    later = _oracle_advance_uniform_cells(n, B, p, q, tf, tf + 1.0, beta0, theta0, chi, t_d, c_th)
    if not np.all(2.0 * later[0] + later[1] <= (2.0 * p + q) * (1.0 + 1e-12)):
        raise ValueError("consistency gate: the local advance creates energy")
    return float(np.sum(w * np.maximum(0.0, 1.0 - (q - p) / thr) ** 2) / length)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return cases with independent candidate and oracle argument graphs."""
    return [{'setup': 'import numpy as np\n'
               'import copy\n'
               'def _independent_call(function, *args, **kwargs):\n'
               '    args, kwargs = copy.deepcopy((args, kwargs))\n'
               '    return function(*args, **kwargs)\n',
      'call': '_independent_call(anisotropy_shortfall_measure, N=16, t_end=16000.0, dt_max=8.0)',
      'gold_call': '_independent_call(_oracle_anisotropy_shortfall_measure, N=16, t_end=16000.0, '
                   'dt_max=8.0)'},
     {'setup': 'import numpy as np\n'
               'import copy\n'
               'def _independent_call(function, *args, **kwargs):\n'
               '    args, kwargs = copy.deepcopy((args, kwargs))\n'
               '    return function(*args, **kwargs)\n',
      'call': '_independent_call(anisotropy_shortfall_measure, N=8, t_end=60.0)',
      'gold_call': '_independent_call(_oracle_anisotropy_shortfall_measure, N=8, t_end=60.0)'},
     {'setup': 'import numpy as np\n'
               'import copy\n'
               'def _independent_call(function, *args, **kwargs):\n'
               '    args, kwargs = copy.deepcopy((args, kwargs))\n'
               '    return function(*args, **kwargs)\n',
      'call': '_independent_call(anisotropy_shortfall_measure, N=64, t_end=3000.0, beta0=15.0, '
              'theta0=40.0, chi=500.0, t_d=80.0, dt_max=2.0)',
      'gold_call': '_independent_call(_oracle_anisotropy_shortfall_measure, N=64, t_end=3000.0, '
                   'beta0=15.0, theta0=40.0, chi=500.0, t_d=80.0, dt_max=2.0)'},
     {'setup': 'import numpy as np\n'
               'def _probe_call():\n'
               '    try:\n'
               '        anisotropy_shortfall_measure(N=3, t_end=10.0)\n'
               '    except ValueError:\n'
               '        return 1\n'
               '    except Exception:\n'
               '        return 2\n'
               '    return 0\n'
               'def _probe_gold():\n'
               '    try:\n'
               '        _oracle_anisotropy_shortfall_measure(N=3, t_end=10.0)\n'
               '    except ValueError:\n'
               '        return 1\n'
               '    except Exception:\n'
               '        return 2\n'
               '    return 0\n',
      'call': '_probe_call()',
      'gold_call': '_probe_gold()'}]
