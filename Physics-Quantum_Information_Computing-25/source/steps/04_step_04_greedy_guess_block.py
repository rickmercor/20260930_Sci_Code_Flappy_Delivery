"""
Compute the exact correction profile of a diagonal block over an interval of affine weight calibration gains.

An uncertain calibration gain can change greedy decisions even though the measured syndrome is unchanged. Changes occur at objective crossings; block corrections need not improve monotonically with gain.

Returns
-------
tuple: (breakpoints, errors), where breakpoints is a tuple of reduced integer (numerator, positive_denominator) pairs covering gain_interval, and errors is an integer 0/1 array of shape (len(breakpoints) - 1, m_D + n_B), with one correction per open gain interval and adjacent identical corrections merged.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def greedy_guess_block(
    b_matrix: "np.ndarray",
    syndrome: "np.ndarray",
    weight_f: "np.ndarray",
    weight_g: "np.ndarray",
    max_iter: int,
    gain_interval: tuple,
) -> tuple:
    r"""Return the gain-dependent correction profile for one block $(I \mid B)$.

    Each weight row is the integer pair $(a, b)$ representing the
    affine objective coefficient $a + b\alpha$ at calibration gain $\alpha$.
    ``gain_interval`` is ``((lo_num, lo_den), (hi_num, hi_den))``, with
    positive denominators and $0 < \mathrm{lo} < \mathrm{hi}$. At each fixed
    gain use the specified monotone greedy decoder. Return its complete
    piecewise-constant correction profile, including every change on an open
    interval. Values at isolated breakpoints and interval endpoints do not
    contribute to the requested average and are excluded from the profile
    contract.

    The result is ``(breakpoints, errors)``. ``breakpoints`` is a tuple of
    reduced integer numerator/positive-denominator pairs in strictly
    increasing order, starting at $\mathrm{lo}$ and ending at $\mathrm{hi}$.
    Row $i$ of the integer 0/1 array ``errors`` applies between breakpoint
    $i$ and $i+1$. Adjacent rows must differ; merge adjacent intervals with
    the same full correction even when internal greedy decisions changed.
    Integer affine coefficients make the breakpoints rational. A sampled gain
    grid does not define this exact profile.

    For a fixed gain start from $g = 0$ and $f = s$, where $s$ = ``syndrome``.
    Form every candidate that sets one currently-zero bit of $g$ to one; bits
    are never cleared. Recompute $f = (B g + s) \bmod 2$ and the full weighted
    cost of $(f, g)$. Select the lowest-cost candidate, breaking equal
    costs by the lowest newly set $B$-column index. Accept only a strict
    improvement; otherwise stop. Stop also when no zero bits remain or
    after ``max_iter`` rounds. This rule applies throughout the gain interval.

    Parameters
    ----------
    b_matrix : np.ndarray
        Binary array $B$ of shape $(m_D, n_B)$, both dimensions positive.
    syndrome : np.ndarray
        Binary vector $s$ of length $m_D$.
    weight_f : np.ndarray
        Integer affine coefficients of shape $(m_D, 2)$.
    weight_g : np.ndarray
        Integer affine coefficients of shape $(n_B, 2)$.
    max_iter : int
        Non-negative round cap.
    gain_interval : tuple
        Two positive rational endpoints as defined above.

    Returns
    -------
    tuple
        Rational breakpoints and errors of shape $(N_{\mathrm{int}},\ m_D + n_B)$,
        where $N_{\mathrm{int}}$ is the number of intervals.

    Raises
    ------
    ValueError
        If a binary array, coefficient array, cap or interval violates
        the stated shape or domain.

    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
from fractions import Fraction
from bisect import bisect_right

def _gain_bounds(interval):
    try:
        pairs = tuple(tuple(pair) for pair in interval)
        if len(pairs) != 2 or any(len(pair) != 2 for pair in pairs):
            raise ValueError
        if any(isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, np.integer))
               for pair in pairs for v in pair):
            raise ValueError
        if any(pair[1] <= 0 for pair in pairs):
            raise ValueError
        lo, hi = (Fraction(int(a), int(b)) for a, b in pairs)
        if not 0 < lo < hi:
            raise ValueError
    except (TypeError, ValueError, ZeroDivisionError):
        raise ValueError("gain_interval must hold two increasing positive rational endpoints") from None
    return lo, hi

def _affine_array(values, rows):
    a = np.asarray(values)
    if a.shape != (rows, 2) or not np.issubdtype(a.dtype, np.integer):
        raise ValueError("weights must be integer arrays with columns (intercept, slope)")
    return a.copy()

def _profile_cost(coefficients, error):
    return tuple(sum(int(coefficients[j, k]) for j in np.flatnonzero(error)) for k in range(2))

def _right_key(cost, x):
    return cost[0] * x.denominator + cost[1] * x.numerator, cost[1]

def _crossing(x, end, lesser, greater):
    slope = lesser[1] - greater[1]
    if slope > 0:
        root = Fraction(greater[0] - lesser[0], slope)
        if x < root < end:
            end = root
    return end

def _profile_append(edges, errors, end, error):
    if errors and np.array_equal(errors[-1], error):
        edges[-1] = end
    else:
        errors.append(error.copy())
        edges.append(end)

def _profile_result(edges, errors):
    return tuple((int(x.numerator), int(x.denominator)) for x in edges), np.asarray(errors, dtype=np.int64)


def _oracle_greedy_guess_block(
    b_matrix: "np.ndarray",
    syndrome: "np.ndarray",
    weight_f: "np.ndarray",
    weight_g: "np.ndarray",
    max_iter: int,
    gain_interval: tuple,
) -> tuple:
    b = np.asarray(b_matrix)
    if b.ndim != 2 or min(b.shape) < 1 or not np.all((b == 0) | (b == 1)):
        raise ValueError("b_matrix must be a nonempty binary matrix")
    b = b.astype(np.int64)
    rows, cols = b.shape
    s = np.asarray(syndrome)
    if s.shape != (rows,) or not np.all((s == 0) | (s == 1)):
        raise ValueError("syndrome must be a binary vector matching B")
    s = s.astype(np.int64)
    coeff = np.vstack((_affine_array(weight_f, rows), _affine_array(weight_g, cols)))
    if isinstance(max_iter, (bool, np.bool_)) or not isinstance(max_iter, (int, np.integer)) or max_iter < 0:
        raise ValueError("max_iter must be a non-negative integer")
    lo, hi = _gain_bounds(gain_interval)
    def complete(g):
        e = np.concatenate(((b @ g + s) % 2, g))
        return e, _profile_cost(coeff, e)
    edges, errors, x = [lo], [], lo
    while x < hi:
        g = np.zeros(cols, dtype=np.int64)
        e, incumbent = complete(g)
        end = hi
        for _ in range(int(max_iter)):
            candidates = []
            for j in np.flatnonzero(g == 0):
                candidate = g.copy(); candidate[j] = 1
                ce, cost = complete(candidate)
                candidates.append((cost, int(j), candidate, ce))
            if not candidates:
                break
            best = min(candidates, key=lambda item: (_right_key(item[0], x), item[1]))
            cost, _, candidate, ce = best
            for other, _, _, _ in candidates:
                end = _crossing(x, end, cost, other)
            if _right_key(cost, x) < _right_key(incumbent, x):
                end = _crossing(x, end, cost, incumbent)
                g, e, incumbent = candidate, ce, cost
            else:
                end = _crossing(x, end, incumbent, cost)
                break
        _profile_append(edges, errors, end, e)
        x = end
    return _profile_result(edges, errors)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases() -> list[dict]:
    """Return explicit differential cases for Studio contract inspection."""
    return [{'setup': 'import numpy as np\n',
      'call': 'greedy_guess_block(np.array([[1,0,0,1],[1,1,0,0],[0,1,1,0],[0,0,1,1]]), np.array([1,1,0,1]), '
              'np.array([[40,0],[50,0],[30,0],[45,0]]), np.array([[0,32],[0,41],[0,25],[0,37]]), 3, ((1, 2), (3, '
              '2)))',
      'gold_call': '_oracle_greedy_guess_block(np.array([[1,0,0,1],[1,1,0,0],[0,1,1,0],[0,0,1,1]]), '
                   'np.array([1,1,0,1]), np.array([[40,0],[50,0],[30,0],[45,0]]), '
                   'np.array([[0,32],[0,41],[0,25],[0,37]]), 3, ((1, 2), (3, 2)))',
      'tol': 1e-12},
     {'setup': 'import numpy as np\n',
      'call': 'greedy_guess_block(np.array([[1,1],[1,0],[0,1]]), np.array([1,1,1]), '
              'np.array([[8,0],[5,0],[9,0]]), np.array([[0,7],[1,8]]), 2, ((1, 4), (2, 1)))',
      'gold_call': '_oracle_greedy_guess_block(np.array([[1,1],[1,0],[0,1]]), np.array([1,1,1]), '
                   'np.array([[8,0],[5,0],[9,0]]), np.array([[0,7],[1,8]]), 2, ((1, 4), (2, 1)))',
      'tol': 1e-12},
     {'setup': 'import numpy as np\n',
      'call': 'greedy_guess_block(np.array([[0,1,1,0],[0,1,0,0],[1,1,0,1],[0,0,0,1]]), np.array([1,1,0,1]), '
              'np.array([[157,0],[167,0],[139,0],[199,0]]), np.array([[0,144],[0,86],[0,22],[0,48]]), 4, ((3, '
              '4), (5, 4)))',
      'gold_call': '_oracle_greedy_guess_block(np.array([[0,1,1,0],[0,1,0,0],[1,1,0,1],[0,0,0,1]]), '
                   'np.array([1,1,0,1]), np.array([[157,0],[167,0],[139,0],[199,0]]), '
                   'np.array([[0,144],[0,86],[0,22],[0,48]]), 4, ((3, 4), (5, 4)))',
      'tol': 1e-12},
     {'setup': 'import numpy as np\n',
      'call': 'greedy_guess_block(np.array([[1,1]]), np.array([1]), np.array([[10,0]]), np.array([[1,3],[4,1]]), '
              '1, ((1, 1), (4, 1)))',
      'gold_call': '_oracle_greedy_guess_block(np.array([[1,1]]), np.array([1]), np.array([[10,0]]), '
                   'np.array([[1,3],[4,1]]), 1, ((1, 1), (4, 1)))',
      'tol': 1e-12},
     {'setup': 'import numpy as np\n',
      'call': 'greedy_guess_block(np.array([[1,1]]), np.array([1]), np.array([[10,0]]), np.array([[0,5],[0,5]]), '
              '2, ((1, 1), (3, 1)))',
      'gold_call': '_oracle_greedy_guess_block(np.array([[1,1]]), np.array([1]), np.array([[10,0]]), '
                   'np.array([[0,5],[0,5]]), 2, ((1, 1), (3, 1)))',
      'tol': 1e-12},
     {'setup': 'import numpy as np\n',
      'call': 'greedy_guess_block(np.array([[1]]), np.array([1]), np.array([[1000001,0]]), '
              'np.array([[0,1000000]]), 1, ((1, 1), (1000003, 1000000)))',
      'gold_call': '_oracle_greedy_guess_block(np.array([[1]]), np.array([1]), np.array([[1000001,0]]), '
                   'np.array([[0,1000000]]), 1, ((1, 1), (1000003, 1000000)))',
      'tol': 1e-12},
     {'setup': 'import numpy as np\n',
      'call': 'greedy_guess_block(np.array([[1,0],[0,1]]), np.array([1,0]), np.array([[3,2],[5,1]]), '
              'np.array([[4,-2],[3,1]]), 0, ((1, 2), (3, 2)))',
      'gold_call': '_oracle_greedy_guess_block(np.array([[1,0],[0,1]]), np.array([1,0]), '
                   'np.array([[3,2],[5,1]]), np.array([[4,-2],[3,1]]), 0, ((1, 2), (3, 2)))',
      'tol': 1e-12},
     {'setup': 'import numpy as np\n',
      'call': 'greedy_guess_block(np.array([[1,0],[0,1]]), np.array([0,0]), np.array([[3,0],[5,0]]), '
              'np.array([[0,4],[0,3]]), 3, ((1, 2), (3, 2)))',
      'gold_call': '_oracle_greedy_guess_block(np.array([[1,0],[0,1]]), np.array([0,0]), '
                   'np.array([[3,0],[5,0]]), np.array([[0,4],[0,3]]), 3, ((1, 2), (3, 2)))',
      'tol': 1e-12}]
