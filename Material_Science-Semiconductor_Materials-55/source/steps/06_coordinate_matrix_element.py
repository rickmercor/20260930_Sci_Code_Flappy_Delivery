"""
Evaluate the squared generalised configuration-coordinate matrix element out of each thermally populated excited-state vibrational level into the energy-conserving ground-state level.

Once the lattice is warm the excited electronic state no longer sits only in its
zero-point vibrational level, and the nonradiative channel draws on every level m
of the excited-state ladder that is thermally populated. The operator that drives
the transition is the mass-weighted configuration coordinate itself, with its
origin at the relaxed ground-state geometry, taken between vibrational level m of
the excited electronic state and vibrational level n of the electronic ground
state. Both electronic states are harmonic wells of the same curvature offset by
delta_Q along that coordinate, so the two vibrational manifolds are the states of
one displaced oscillator pair, and every element follows from that algebra in
delta_Q, the coupling strength S and the two level indices alone. No numerical
overlap integral is needed.

Energy conservation later ties the two levels together as n = m + p, where the
level offset p is the transition energy measured in vibrational quanta and is in
general not an integer. For integer n the squared element is the Poisson weight
exp(-S) S^n / n! multiplied by a factor that is a polynomial in n. Continue the
element to real n by keeping that polynomial as it stands and replacing the
factorial with the gamma function. At m = 0 this reduces to the zero-temperature
element of the low-temperature theory.

The thermal sum reaches excited levels of a few hundred at high temperature, with
coupling strengths up to about ten. There the separate pieces of the element
overflow or cancel catastrophically if they are evaluated term by term, so the
returned values must stay accurate to about one part in 10^9, wherever the element
itself is representable in double precision, over excited levels up to 300 and
coupling strengths up to 10.

Returns
-------
numpy array shaped like excited_levels, the squared coordinate matrix element for each excited level in amu Angstrom^2.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def coordinate_matrix_element(delta_q: float, huang_rhys: float,
                              excited_levels: np.ndarray, level_offset: float) -> np.ndarray:
    '''Squared coordinate matrix element from excited levels m into ground levels m + p.

    Parameters
    ----------
    delta_q : float
        Mass-weighted coordinate offset in sqrt(amu) Angstrom. Must be positive.
    huang_rhys : float
        Dimensionless electron-phonon coupling strength S. Must be positive.
    excited_levels : numpy.ndarray of int
        Vibrational level indices m of the excited electronic state, each a
        non-negative integer.
    level_offset : float
        Real, non-negative offset p, so that each excited level m is paired with
        the ground-state level n = m + p continued to real values.

    Returns
    -------
    element_sq : numpy.ndarray
        |<chi_e,m| Q |chi_g,m+p>|^2 in amu Angstrom^2 for every m, with the same
        shape as excited_levels. The coordinate Q is measured from the relaxed
        ground-state geometry. Raises ValueError on non-finite input, on a
        non-positive delta_q or huang_rhys, on a negative level_offset, or when
        excited_levels holds anything other than non-negative integers.
    '''
    return element_sq

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _laguerre_ladder(top, alpha, x):
    # L_k^(alpha)(x) for k = 0 .. top by the three-term recurrence, for any real alpha.
    out = np.empty(top + 1)
    out[0] = 1.0
    if top >= 1:
        out[1] = 1.0 + alpha - x
    for k in range(1, top):
        out[k + 1] = ((2.0 * k + 1.0 + alpha - x) * out[k] - (k + alpha) * out[k - 1]) / (k + 1.0)
    return out


def _oracle_coordinate_matrix_element(delta_q: float, huang_rhys: float,
                                      excited_levels: np.ndarray, level_offset: float) -> np.ndarray:
    from math import lgamma
    dq = float(delta_q)
    s = float(huang_rhys)
    p = float(level_offset)
    m = np.asarray(excited_levels)
    if not (np.isfinite(dq) and np.isfinite(s) and np.isfinite(p)):
        raise ValueError("delta_q, huang_rhys and level_offset must be finite")
    if dq <= 0.0 or s <= 0.0:
        raise ValueError("delta_q and huang_rhys must be positive")
    if p < 0.0:
        raise ValueError("level_offset must be non-negative")
    if m.dtype.kind not in "iu" or np.any(m < 0):
        raise ValueError("excited_levels must be non-negative integers")
    mf = m.astype(float)
    top = int(m.max(initial=0)) + 1
    # The coordinate about the excited-state minimum is delta_q plus the two ladder terms,
    # so each element combines the overlaps from levels m - 1, m and m + 1. Every overlap
    # into level m + p is a Laguerre polynomial in S whose upper index depends on p alone,
    # so one recurrence per upper index serves all levels, and its coefficients are
    # polynomials in the ground-state level, which fixes the continuation to real values.
    lag_p = _laguerre_ladder(top, p, s)
    lag_up = _laguerre_ladder(top, p + 1.0, s)
    lag_dn = _laguerre_ladder(top, p - 1.0, s)
    below = np.where(m >= 1, lag_up[np.maximum(m - 1, 0)], 0.0)
    bracket = lag_p[m] + 0.5 * below + (mf + 1.0) / (2.0 * s) * lag_dn[m + 1]
    lgam = np.vectorize(lgamma, otypes=[float])
    n = mf + p
    log_env = -s + n * np.log(s) - lgam(n + 1.0) + lgam(mf + 1.0) - mf * np.log(s)
    with np.errstate(divide="ignore"):
        return np.exp(2.0 * np.log(dq) + log_env + 2.0 * np.log(np.abs(bracket)))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        {   # normal: the task centre, every level a thermal sum near 150 K draws on
            "setup": "import numpy as np\nm = np.arange(0, 31)\n",
            "call": "np.log(coordinate_matrix_element(1.020205375219, 3.902705291187, m.copy(), 24.289581799794))",
            "gold_call": "np.log(_oracle_coordinate_matrix_element(1.020205375219, 3.902705291187, m.copy(), 24.289581799794))",
        },
        {   # boundary: the zero-point level alone, which is the low-temperature element
            "setup": "import numpy as np\n",
            "call": "np.log(coordinate_matrix_element(1.02, 3.9, np.array([0]), 24.3))",
            "gold_call": "np.log(_oracle_coordinate_matrix_element(1.02, 3.9, np.array([0]), 24.3))",
        },
        {   # boundary: zero offset pairs each level with itself, the vertical diagonal
            "setup": "import numpy as np\nm = np.arange(0, 12)\n",
            "call": "coordinate_matrix_element(0.85, 2.2, m.copy(), 0.0)",
            "gold_call": "_oracle_coordinate_matrix_element(0.85, 2.2, m.copy(), 0.0)",
        },
        {   # edge: strong coupling and high levels, where term-by-term sums cancel
            "setup": "import numpy as np\nm = np.arange(0, 301, 25)\n",
            "call": "np.log(coordinate_matrix_element(1.4, 9.5, m.copy(), 12.7))",
            "gold_call": "np.log(_oracle_coordinate_matrix_element(1.4, 9.5, m.copy(), 12.7))",
        },
        {   # edge: non-contiguous high levels target narrow cancellation minima missed by a regular sparse ladder
            "setup": "import numpy as np\nm = np.array([49, 57, 128, 153, 167, 180, 195, 210, 225, 241, 257, 274, 291, 292])\n",
            "call": "np.log(coordinate_matrix_element(1.4, 9.5, m.copy(), 12.7))",
            "gold_call": "np.log(_oracle_coordinate_matrix_element(1.4, 9.5, m.copy(), 12.7))",
        },
        {   # edge: weak coupling with a large offset, deep in the Poisson tail
            "setup": "import numpy as np\nm = np.array([0, 1, 3, 10, 40, 90, 160])\n",
            "call": "np.log(coordinate_matrix_element(0.42, 0.35, m.copy(), 60.3))",
            "gold_call": "np.log(_oracle_coordinate_matrix_element(0.42, 0.35, m.copy(), 60.3))",
        },
        {   # normal: an intermediate coupling on a non-contiguous set of levels
            "setup": "import numpy as np\nm = np.array([2, 7, 19, 55, 120])\n",
            "call": "np.log(coordinate_matrix_element(1.15, 5.3, m.copy(), 6.45))",
            "gold_call": "np.log(_oracle_coordinate_matrix_element(1.15, 5.3, m.copy(), 6.45))",
        },
        {   # edge: a negative level offset must raise ValueError
            "setup": """import numpy as np
def probe(fn):
    try:
        fn(1.02, 3.9, np.array([0, 1]), -0.5)
    except ValueError:
        return 1
    except Exception:
        return 2
    return 0
""",
            "call": "probe(coordinate_matrix_element)",
            "gold_call": "probe(_oracle_coordinate_matrix_element)",
        },
    ]
