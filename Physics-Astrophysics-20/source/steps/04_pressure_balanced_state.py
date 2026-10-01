"""
The pressure-balanced state of a slab of fluid elements.

The pressure-balanced state of a slab of fluid elements.

The slab is described as N fluid elements of fixed mass that together fill a periodic
length L. Each element carries its own density n, its pressures P_perp and P_par and,
by flux freezing, the field B = B0 n (units of step 01, with n0 = 1). This step maps a
given element state to the state reached by compressing or expanding every element
perpendicular to the field without radiation or scattering, with the element masses
fixed, such that the total perpendicular pressure P_perp + B^2/2 is the same in every
element and the element widths mass/n add up to L. Along such a reversible
perpendicular compression P_perp is proportional to n^(8/5) and P_par to n^(4/5).

Returns
-------
The element densities and pressures of the pressure-balanced state and the common
total pressure, stacked.

Returns
-------
The element densities and pressures of the pressure-balanced state and the common total pressure, stacked.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def pressure_balanced_state(
    mass: ArrayLike,
    n: ArrayLike,
    p_perp: ArrayLike,
    p_par: ArrayLike,
    L: float,
    beta0: float,
    theta0: float,
) -> np.ndarray:
    """Return the pressure-balanced state of a slab of fluid elements.

    Parameters
    ----------
    mass : array_like
        Element masses (units n0 times the length unit of ``L``), shape (N,),
        N >= 1. Strictly positive.
    n : array_like
        Element densities (units n0), shape (N,). Strictly positive.
    p_perp : array_like
        Element perpendicular pressures (units n0 m_e c^2), shape (N,).
        Strictly positive.
    p_par : array_like
        Element parallel pressures (units n0 m_e c^2), shape (N,). Strictly
        positive.
    L : float
        Slab length, in the same length unit as ``mass / n``. Strictly
        positive.
    beta0 : float
        Reference plasma beta (step 01). Strictly positive.
    theta0 : float
        Reference temperature T0 / (m_e c^2) (step 01). Strictly positive.

    Returns
    -------
    numpy.ndarray
        Array of shape (4, N). Rows 0, 1 and 2 hold n, P_perp and P_par of
        every element in the pressure-balanced state; row 3 holds the common
        total perpendicular pressure P_perp + B^2/2 in every entry. The field
        of an element is B = B0 n with B0 the reference field of step 01. The
        balance and the length condition hold to a relative accuracy of 1e-12
        or better.

    Raises
    ------
    ValueError
        If ``mass``, ``n``, ``p_perp`` or ``p_par`` is not a one-dimensional
        array with at least one entry, if their lengths differ, if any entry
        is non-finite or non-positive, or if ``L``, ``beta0`` or ``theta0`` is
        not a finite, strictly positive scalar.
    """
    return state  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import math

import numpy as np
from numpy.typing import ArrayLike


def _pb_vec(name, x):
    """One-dimensional array of finite, strictly positive entries."""
    a = np.asarray(x, dtype=float)
    if a.ndim != 1 or a.size < 1:
        raise ValueError(f"{name} must be a one-dimensional array with at least one entry")
    if not np.all(np.isfinite(a)) or np.any(a <= 0.0):
        raise ValueError(f"{name} must be finite and strictly positive")
    return a


def _pb_factors(p, c, pi):
    """Compression factors s > 0 with p s^(8/5) + c s^2 = pi, by Newton from s = 1
    (the left side is convex and increasing in s, so the iterates stay positive)."""
    gp, _ = _fr_exponents()
    s = np.ones_like(p)
    fp = gp * p + 2.0 * c
    for _ in range(100):
        f = p * s ** gp + c * s * s - pi
        fp = gp * p * s ** (gp - 1.0) + 2.0 * c * s
        ds = f / fp
        s = s - ds
        if np.max(np.abs(ds) / s) < 1e-15:
            break
    return s, fp


def _pb_project(mass, n, p, q, length, b0):
    """Pressure-balanced state without validation; returns n, P_perp, P_par and the
    common total pressure."""
    gp, gq = _fr_exponents()
    c = 0.5 * (b0 * n) ** 2
    vol = mass / n
    pi = float(np.sum(vol * (p + c)) / length)
    for _ in range(200):
        s, fp = _pb_factors(p, c, pi)
        g = float(np.sum(vol / s)) - length
        dg = -float(np.sum(vol / (s * s) / fp))
        dpi = g / dg
        pi -= dpi
        if abs(dpi) <= 1e-15 * pi:
            break
    s, _ = _pb_factors(p, c, pi)
    return n * s, p * s ** gp, q * s ** gq, pi


def _pb_state_args(mass, n, p_perp, p_par):
    """Validated element arrays of equal length."""
    m = _pb_vec("mass", mass)
    nn = _pb_vec("n", n)
    p = _pb_vec("p_perp", p_perp)
    q = _pb_vec("p_par", p_par)
    if not (m.size == nn.size == p.size == q.size):
        raise ValueError("mass, n, p_perp and p_par must have the same length")
    return m, nn, p, q


def _oracle_pressure_balanced_state(
    mass: ArrayLike,
    n: ArrayLike,
    p_perp: ArrayLike,
    p_par: ArrayLike,
    L: float,
    beta0: float,
    theta0: float,
) -> np.ndarray:
    m, nn, p, q = _pb_state_args(mass, n, p_perp, p_par)
    if m.size < 1:
        raise ValueError("the slab needs at least one element")
    length = _rs_scalar("L", L)
    beta0 = _rs_scalar("beta0", beta0)
    theta0 = _rs_scalar("theta0", theta0)
    b0 = math.sqrt(2.0 * theta0 / beta0)
    n1, p1, q1, pi = _pb_project(m, nn, p, q, length, b0)
    return np.stack((n1, p1, q1, np.full(n1.size, pi)))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return cases with independent candidate and oracle argument graphs."""
    return [{'setup': 'import numpy as np\n'
               'm = np.array([0.5, 0.8, 1.0, 1.2, 0.9, 0.6])\n'
               'n = np.array([0.7, 0.9, 1.2, 1.5, 1.0, 0.8])\n'
               'p = np.array([900.0, 950.0, 850.0, 700.0, 1000.0, 1100.0])\n'
               'q = np.array([950.0, 1000.0, 900.0, 760.0, 1020.0, 1150.0])\n'
               'L = 1.1 * float(np.sum(m / n))\n'
               'import copy\n'
               'def _independent_call(function, *args, **kwargs):\n'
               '    args, kwargs = copy.deepcopy((args, kwargs))\n'
               '    return function(*args, **kwargs)\n',
      'call': '_independent_call(pressure_balanced_state, m, n, p, q, L, 40.0, 1000.0)',
      'gold_call': '_independent_call(_oracle_pressure_balanced_state, m, n, p, q, L, 40.0, 1000.0)'},
     {'setup': 'import numpy as np\n'
               'import copy\n'
               'def _independent_call(function, *args, **kwargs):\n'
               '    args, kwargs = copy.deepcopy((args, kwargs))\n'
               '    return function(*args, **kwargs)\n',
      'call': '_independent_call(pressure_balanced_state, np.ones(4), np.ones(4), np.full(4, 1000.0), '
              'np.full(4, 1000.0), 4.0, 40.0, 1000.0)',
      'gold_call': '_independent_call(_oracle_pressure_balanced_state, np.ones(4), np.ones(4), '
                   'np.full(4, 1000.0), np.full(4, 1000.0), 4.0, 40.0, 1000.0)'},
     {'setup': 'import numpy as np\n'
               'import copy\n'
               'def _independent_call(function, *args, **kwargs):\n'
               '    args, kwargs = copy.deepcopy((args, kwargs))\n'
               '    return function(*args, **kwargs)\n',
      'call': '_independent_call(pressure_balanced_state, np.array([1.0, 3.0]), np.array([0.5, 3.0]), '
              'np.array([5000.0, 20.0]), np.array([5200.0, 30.0]), 3.0, 10.0, 1000.0)',
      'gold_call': '_independent_call(_oracle_pressure_balanced_state, np.array([1.0, 3.0]), '
                   'np.array([0.5, 3.0]), np.array([5000.0, 20.0]), np.array([5200.0, 30.0]), 3.0, '
                   '10.0, 1000.0)'},
     {'setup': 'import numpy as np\n'
               'def _probe_call():\n'
               '    try:\n'
               '        pressure_balanced_state(np.array([1.0, 0.0]), np.ones(2), np.ones(2), '
               'np.ones(2), 2.0, 40.0, 1.0)\n'
               '    except ValueError:\n'
               '        return 1\n'
               '    except Exception:\n'
               '        return 2\n'
               '    return 0\n'
               'def _probe_gold():\n'
               '    try:\n'
               '        _oracle_pressure_balanced_state(np.array([1.0, 0.0]), np.ones(2), np.ones(2), '
               'np.ones(2), 2.0, 40.0, 1.0)\n'
               '    except ValueError:\n'
               '        return 1\n'
               '    except Exception:\n'
               '        return 2\n'
               '    return 0\n',
      'call': '_probe_call()',
      'gold_call': '_probe_gold()'}]
