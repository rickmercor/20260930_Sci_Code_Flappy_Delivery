"""
Compute the tightest lower bound on the volume that the available moment data forces.

The same two sequences are supplied as before, together with a certified value $\bar g \ge \max_B g$ and the ambient dimension.

Consider again the family $(y_k - \alpha z_k)$ indexed by $\alpha \ge 0$. Previously the requirement was only that the sequence remain the moments of a non-negative measure. Now impose the stronger requirement that the measure be supported within the interval $[1, \bar g]$, and determine the extremal $\alpha$ compatible with that requirement. Return the corresponding bound on $\operatorname{vol}K$.

Return the extremal value the data determines, exactly as the construction yields it and without adjustment of any kind. This is the sole return path: a budget too small to form the weighted construction at all admits no answer and must raise rather than return a substitute value.

The result must be correct to full double precision across the whole range of budgets the function may be called with. The execution environment provides NumPy and the Python standard library. No other package is available, and an implementation that depends on one will not run.

Locating the residual is what distinguishes this bound from the previous one. The part of the pushforward that survives the subtraction lives entirely where $g > 1$, and since $g$ never exceeds $\bar g$ on the box, that residual is confined to the interval $[1, \bar g]$. Knowing merely that a measure is non-negative constrains the unknown scalar in one direction; knowing in addition where it lives constrains it in the other.

Support conditions of this kind are encoded multiplicatively. One selects a polynomial whose sign distinguishes the region where the residual is permitted to live from the region carrying the subtracted part, then requires that the sequence obtained by weighting the moments with that polynomial also be admissible. Weighting a moment sequence by a polynomial means forming combinations of shifted entries with the polynomial's coefficients, so the resulting matrix reaches further into the sequence than its side length alone would suggest.

The mechanics are otherwise as before: the condition is affine in $\alpha$, the subtracted part contributes a definite matrix, and the extremal value follows from dense symmetric linear algebra.

The dependence on $\bar g$ is one-sided and worth understanding. The companion bound involves no support information and is completely insensitive to how loosely the range was estimated. This one is not: enlarging $\bar g$ enlarges the interval, weakens the support constraint, and moves the bound away from the volume. A certified but loose estimate costs accuracy here and nowhere else, which is why a cheap estimate was acceptable at the outset.

The same numerical caution applies. The matrices involved are ill-conditioned in the side length and built from entries that grow geometrically in the index, so the working precision required grows with the budget. Fixed-precision arithmetic is adequate only over a limited range, beyond which it returns a plausible but incorrect value rather than failing visibly. The execution environment provides NumPy and the Python standard library. No other package is available, and an implementation that depends on one will not run.

Returns
-------
float: the bound on the volume of the unit sublevel set of g determined by the supplied data.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def compute_lower_bound(box_moments: np.ndarray, reference_moments: np.ndarray, gbar: float, dimension: int) -> float:
    """Compute the tightest lower bound on the volume forced by the moment data.

    Parameters
    ----------
    box_moments : np.ndarray
        Array of shape (K + 1,) and dtype float64 holding the pushforward
        moments [y_0, ..., y_K].
    reference_moments : np.ndarray
        Array of shape (K + 1,) and dtype float64 holding the constants
        [z_0, ..., z_K]. Must match box_moments in length.
    gbar : float
        A certified value satisfying gbar >= max over B of g. Must exceed 1.
    dimension : int
        The ambient dimension n.

    Returns
    -------
    float
        The bound on the volume of K = {x : g(x) <= 1} determined by the
        supplied data, returned exactly as the construction yields it.

    Raises
    ------
    ValueError
        If the moment sequences are empty, differ in length, or are not
        one-dimensional; if gbar does not exceed 1; if the budget is too
        small to form the weighted construction; or if the subtracted part
        fails to be positive definite.
    """
    return 0.0

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
from fractions import Fraction


def _oracle_compute_lower_bound(box_moments: np.ndarray, reference_moments: np.ndarray, gbar: float, dimension: int) -> float:
    """Reference implementation: exact rational bisection on the weighted affine matrix family."""
    y = np.asarray(box_moments, dtype=np.float64)
    z = np.asarray(reference_moments, dtype=np.float64)

    if y.ndim != 1 or z.ndim != 1:
        raise ValueError("moment sequences must be one-dimensional")
    if y.size != z.size or y.size == 0:
        raise ValueError("moment sequences must be non-empty and of equal length")

    gb = Fraction(float(gbar))
    if gb <= 1:
        raise ValueError("gbar must exceed 1")

    q = (y.size - 1) // 2
    if q == 0:
        raise ValueError("budget is too small to form the weighted construction")

    Y = [Fraction(float(v)) for v in y]
    Z = [Fraction(float(v)) for v in z]
    h = [-gb, gb + 1, Fraction(-1)]

    A = [[sum(h[k] * Y[i + j + k] for k in range(3)) for j in range(q)] for i in range(q)]
    B = [[-sum(h[k] * Z[i + j + k] for k in range(3)) for j in range(q)] for i in range(q)]

    def _h_pos_def(M):
        """Exact positive-definiteness test by symmetric Gaussian elimination."""
        n = len(M)
        m = [row[:] for row in M]
        for i in range(n):
            p = m[i][i]
            if p <= 0:
                return False
            inv = Fraction(1) / p
            for r in range(i + 1, n):
                f = m[r][i] * inv
                if f == 0:
                    continue
                for c in range(i, n):
                    m[r][c] -= f * m[i][c]
        return True

    if not _h_pos_def(B):
        raise ValueError("subtracted part is not positive definite")

    lo, hi = Fraction(-(2 ** 40)), Fraction(2 ** 40)
    for _ in range(140):
        mid = (lo + hi) / 2
        M = [[A[i][j] - mid * B[i][j] for j in range(q)] for i in range(q)]
        if _h_pos_def(M):
            lo = mid
        else:
            hi = mid

    return -float(2 ** int(dimension)) * float(lo)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Differential test cases: normal, boundary, and edge instances."""
    return [
        {
            # normal: the task instance at its budget k = 0..14, with the
            # certified estimate gbar = 8 produced upstream; the weighted
            # matrix reaches side 7 and the answer lies beyond the reach of
            # fixed-precision dense linear algebra
            "setup": "import numpy as np\nfrom fractions import Fraction\ncoef = [Fraction(2), Fraction(2), Fraction(2), Fraction(-2)]\nexpo = [(8, 0, 0), (0, 4, 0), (0, 0, 2), (4, 2, 0)]\npoly = {}\nfor c, e in zip(coef, expo):\n    poly[e] = poly.get(e, Fraction(0)) + c\ncur = {(0, 0, 0): Fraction(1)}\nys = []\nfor _ in range(15):\n    t = Fraction(0)\n    for a, c in cur.items():\n        if any(v % 2 for v in a):\n            continue\n        w = c\n        for v in a:\n            w /= (v + 1)\n        t += w\n    ys.append(float(t))\n    nxt = {}\n    for a, ca in cur.items():\n        for b, cb in poly.items():\n            k2 = tuple(p + q for p, q in zip(a, b))\n            nxt[k2] = nxt.get(k2, Fraction(0)) + ca * cb\n    cur = nxt\ny = np.array(ys)\nz = np.array([float(Fraction(7, 7 + 8 * k)) for k in range(15)])\ngbar = 8.0\nn = 3",
            "call": "compute_lower_bound(y, z, gbar, n)",
            "gold_call": "_oracle_compute_lower_bound(y, z, gbar, n)",
        },
        {
            # boundary: identical data with the exact maximum gbar = 4, which
            # tightens this bound while leaving the companion untouched
            "setup": "import numpy as np\nfrom fractions import Fraction\ncoef = [Fraction(2), Fraction(2), Fraction(2), Fraction(-2)]\nexpo = [(8, 0, 0), (0, 4, 0), (0, 0, 2), (4, 2, 0)]\npoly = {}\nfor c, e in zip(coef, expo):\n    poly[e] = poly.get(e, Fraction(0)) + c\ncur = {(0, 0, 0): Fraction(1)}\nys = []\nfor _ in range(15):\n    t = Fraction(0)\n    for a, c in cur.items():\n        if any(v % 2 for v in a):\n            continue\n        w = c\n        for v in a:\n            w /= (v + 1)\n        t += w\n    ys.append(float(t))\n    nxt = {}\n    for a, ca in cur.items():\n        for b, cb in poly.items():\n            k2 = tuple(p + q for p, q in zip(a, b))\n            nxt[k2] = nxt.get(k2, Fraction(0)) + ca * cb\n    cur = nxt\ny = np.array(ys)\nz = np.array([float(Fraction(7, 7 + 8 * k)) for k in range(15)])\ngbar = 4.0\nn = 3",
            "call": "compute_lower_bound(y, z, gbar, n)",
            "gold_call": "_oracle_compute_lower_bound(y, z, gbar, n)",
        },
        {
            # edge: a short budget k = 0..2, where the construction reduces
            # to a single scalar and the data determines an extremal value
            # that lies well away from the eventual bracket
            "setup": "import numpy as np\nfrom fractions import Fraction\ny = np.array([1.0, 52.0 / 45.0, 410288.0 / 208845.0])\nz = np.array([float(Fraction(7, 7 + 8 * k)) for k in range(3)])\ngbar = 8.0\nn = 3",
            "call": "compute_lower_bound(y, z, gbar, n)",
            "gold_call": "_oracle_compute_lower_bound(y, z, gbar, n)",
        },
        {
            # boundary: a reduced budget k = 0..6 at the same certified
            # estimate, confirming the endpoint moves monotonically with the
            # budget
            "setup": "import numpy as np\nfrom fractions import Fraction\ny = np.array([1.0, 52.0 / 45.0, 410288.0 / 208845.0, 45964096.0 / 11486475.0, 64170635008.0 / 6995263275.0, 4627392439985152.0 / 201624473375325.0, 5664835468095488.0 / 92204876128365.0])\nz = np.array([float(Fraction(7, 7 + 8 * k)) for k in range(7)])\ngbar = 8.0\nn = 3",
            "call": "compute_lower_bound(y, z, gbar, n)",
            "gold_call": "_oracle_compute_lower_bound(y, z, gbar, n)",
        },
        {
            # edge: a single moment, where the budget cannot support the
            # weighted construction and no value is defined
            "setup": "import numpy as np\ny = np.array([1.0])\nz = np.array([1.0])\ngbar = 8.0\nn = 3\ndef run_model():\n    try:\n        compute_lower_bound(y, z, gbar, n)\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\ndef run_gold():\n    try:\n        _oracle_compute_lower_bound(y, z, gbar, n)\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
    ]
