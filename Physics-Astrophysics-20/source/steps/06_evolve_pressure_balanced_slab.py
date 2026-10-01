"""
Evolution of the pressure-balanced radiating slab.

Evolution of the pressure-balanced radiating slab.

A periodic slab of the ultra-relativistic pair plasma of step 01 starts at rest,
isotropic and pressure-balanced, with its density and field modulated by a prescribed
seed. It is represented by N fluid elements of equal initial width and evolves by the
splitting step of step 05 under the schedule of step 02. The initial elements and the
time grid are fixed conventions of the task and are spelled out in the function
docstring.

Returns
-------
The element densities, pressures and widths at the end time, stacked.

Returns
-------
The element densities, pressures and widths at the end time, stacked.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def evolve_pressure_balanced_slab(
    N: int,
    t_end: float,
    L: float,
    seed: ArrayLike,
    beta0: float,
    theta0: float,
    chi: float,
    t_d: float,
    c_th: float = 1.4,
    dt_max: float = 0.5,
) -> np.ndarray:
    """Evolve the pressure-balanced slab and return its elements at ``t_end``.

    Initial elements: element i = 0, ..., N-1 has initial width L/N and centre
    x_i = (i + 1/2) L/N; b_i = 1 + sum_j amp_j cos(2 pi k_j x_i / L + phase_j)
    over the rows (amp_j, k_j, phase_j) of ``seed``; n_i = b_i,
    mass_i = n_i L/N and P_perp,i = P_par,i = P0 [1 + (1 - b_i^2)/beta0] with
    P0 = theta0. This state is first mapped through step 04.

    Time grid: the marks are t_d (only if 0 < t_d < t_end) and t_end, taken in
    increasing order from time 0. The interval [a, b] between consecutive marks
    is divided into k = ceil((b - a)/dt_max - 1e-9) equal steps; step j of the
    interval starts at a + j (b - a)/k and is taken with step 05.

    Parameters
    ----------
    N : int
        Number of elements; at least 4.
    t_end : float
        Final time (units 1/Omega0); non-negative. ``t_end = 0`` returns the
        mapped initial state.
    L : float
        Slab length (any length unit). Strictly positive.
    seed : array_like
        Array of shape (M, 3), M >= 1, with rows (amplitude, integer wavenumber
        index k, phase in radians).
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
    dt_max : float, optional
        Largest allowed step size (units 1/Omega0). Strictly positive.

    Returns
    -------
    numpy.ndarray
        Array of shape (4, N): rows n, P_perp, P_par and width mass/n of every
        element at ``t_end``, elements in their initial order (units of steps
        01 and 04).

    Raises
    ------
    ValueError
        If ``N`` is not an integer >= 4; if ``t_end`` is negative or not
        finite; if ``L``, ``beta0``, ``theta0``, ``chi``, ``c_th`` or ``dt_max``
        is not a finite, strictly positive scalar; if ``t_d`` is not finite; if
        ``seed`` is not an (M, 3) array of finite numbers with M >= 1 and
        integer wavenumber indices; or if the initial b or pressures are not
        strictly positive for every element.
    """
    return slab  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import math

import numpy as np
from numpy.typing import ArrayLike


def _oracle_evolve_pressure_balanced_slab(
    N: int,
    t_end: float,
    L: float,
    seed: ArrayLike,
    beta0: float,
    theta0: float,
    chi: float,
    t_d: float,
    c_th: float = 1.4,
    dt_max: float = 0.5,
) -> np.ndarray:
    if isinstance(N, (bool, np.bool_)) or np.ndim(N) != 0:
        raise ValueError("N must be an integer >= 4")
    try:
        nf = float(N)
    except (TypeError, ValueError):
        raise ValueError("N must be an integer >= 4") from None
    if not np.isfinite(nf) or nf != math.floor(nf) or nf < 4:
        raise ValueError("N must be an integer >= 4")
    ncell = int(nf)
    tf = _fr_finite("t_end", t_end)
    if tf < 0.0:
        raise ValueError("t_end must be non-negative")
    length = _rs_scalar("L", L)
    beta0 = _rs_scalar("beta0", beta0)
    theta0 = _rs_scalar("theta0", theta0)
    chi = _rs_scalar("chi", chi)
    td = _fr_finite("t_d", t_d)
    cth = _rs_scalar("c_th", c_th)
    dtm = _rs_scalar("dt_max", dt_max)
    try:
        sd = np.array(seed, dtype=float)
    except (TypeError, ValueError):
        raise ValueError("seed must be a numeric (M, 3) array") from None
    if sd.ndim != 2 or sd.shape[1] != 3 or sd.shape[0] < 1 or not np.all(np.isfinite(sd)):
        raise ValueError("seed must be a finite (M, 3) array with M >= 1")
    if np.any(sd[:, 1] != np.round(sd[:, 1])):
        raise ValueError("seed wavenumber indices must be integers")

    x = (np.arange(ncell, dtype=float) + 0.5) * (length / ncell)
    b = np.ones(ncell)
    for amp, k, ph in sd:
        b = b + amp * np.cos(2.0 * np.pi * k * x / length + ph)
    if np.any(b <= 0.0):
        raise ValueError("the seeded profile b must be positive for every element")
    p = theta0 * (1.0 + (1.0 - b * b) / beta0)
    if np.any(p <= 0.0):
        raise ValueError("the initial pressures must be positive for every element")
    n = b.copy()
    mass = n * (length / ncell)
    q = p.copy()
    b0 = math.sqrt(2.0 * theta0 / beta0)
    n, p, q, _ = _pb_project(mass, n, p, q, length, b0)

    marks = [td, tf] if 0.0 < td < tf else [tf]
    a = 0.0
    for mark in marks:
        span = mark - a
        if span > 0.0:
            k = int(math.ceil(span / dtm - 1e-9))
            h = span / k
            for j in range(k):
                st = _oracle_pressure_balanced_step(mass, n, p, q, a + j * h, h, length, beta0,
                                                    theta0, chi, td, cth)
                n, p, q = st[0], st[1], st[2]
        a = mark
    return np.stack((n, p, q, mass / n))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return cases with independent candidate and oracle argument graphs."""
    return [{'setup': 'import numpy as np\n'
               'seed = np.array([[-0.28, 1.0, 0.0], [0.05, 2.0, np.pi / 3.0]])\n'
               'import copy\n'
               'def _independent_call(function, *args, **kwargs):\n'
               '    args, kwargs = copy.deepcopy((args, kwargs))\n'
               '    return function(*args, **kwargs)\n',
      'call': '_independent_call(evolve_pressure_balanced_slab, 16, 3000.0, 600.0, seed, 40.0, 1000.0, '
              '2000.0, 340.0, dt_max=5.0)',
      'gold_call': '_independent_call(_oracle_evolve_pressure_balanced_slab, 16, 3000.0, 600.0, seed, '
                   '40.0, 1000.0, 2000.0, 340.0, dt_max=5.0)'},
     {'setup': 'import numpy as np\n'
               'seed = np.array([[-0.28, 1.0, 0.0], [0.05, 2.0, np.pi / 3.0]])\n'
               'import copy\n'
               'def _independent_call(function, *args, **kwargs):\n'
               '    args, kwargs = copy.deepcopy((args, kwargs))\n'
               '    return function(*args, **kwargs)\n',
      'call': '_independent_call(evolve_pressure_balanced_slab, 8, 0.0, 1.0, seed, 40.0, 1000.0, '
              '2000.0, 340.0)',
      'gold_call': '_independent_call(_oracle_evolve_pressure_balanced_slab, 8, 0.0, 1.0, seed, 40.0, '
                   '1000.0, 2000.0, 340.0)'},
     {'setup': 'import numpy as np\n'
               'seed = np.array([[0.2, 1.0, 0.4], [-0.1, 3.0, 1.0], [0.05, 2.0, 2.5]])\n'
               'import copy\n'
               'def _independent_call(function, *args, **kwargs):\n'
               '    args, kwargs = copy.deepcopy((args, kwargs))\n'
               '    return function(*args, **kwargs)\n',
      'call': '_independent_call(evolve_pressure_balanced_slab, 12, 1500.0, 1.0, seed, 10.0, 10.0, '
              '200.0, 34.0, c_th=1.4, dt_max=2.0)',
      'gold_call': '_independent_call(_oracle_evolve_pressure_balanced_slab, 12, 1500.0, 1.0, seed, '
                   '10.0, 10.0, 200.0, 34.0, c_th=1.4, dt_max=2.0)'},
     {'setup': 'import numpy as np\n'
               'def _probe_call():\n'
               '    try:\n'
               '        evolve_pressure_balanced_slab(3, 10.0, 1.0, np.array([[-0.2, 1.0, 0.0]]), '
               '40.0, 1.0, 2000.0, 5.0)\n'
               '    except ValueError:\n'
               '        return 1\n'
               '    except Exception:\n'
               '        return 2\n'
               '    return 0\n'
               'def _probe_gold():\n'
               '    try:\n'
               '        _oracle_evolve_pressure_balanced_slab(3, 10.0, 1.0, np.array([[-0.2, 1.0, '
               '0.0]]), 40.0, 1.0, 2000.0, 5.0)\n'
               '    except ValueError:\n'
               '        return 1\n'
               '    except Exception:\n'
               '        return 2\n'
               '    return 0\n',
      'call': '_probe_call()',
      'gold_call': '_probe_gold()'}]
