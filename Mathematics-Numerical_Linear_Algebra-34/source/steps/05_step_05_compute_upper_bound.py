"""
Compute the tightest upper bound on the volume that the available moment data forces.

Two sequences are supplied: the moments $y_0, \dots, y_K$ of the pushforward of the uniform measure on the box through $g$, and the constants $z_0, \dots, z_K$ relating moments of $g$ over $K$ to the volume of $K$.

Consider the family of sequences $(y_k - \alpha z_k)$ indexed by a scalar $\alpha \ge 0$. For small $\alpha$ such a sequence is the moment sequence of a non-negative measure; for large $\alpha$ it is not. Determine the largest $\alpha$ for which the truncated sequence remains admissible, using only the entries available within the supplied budget, and return the corresponding upper bound on $\operatorname{vol}K$.

The returned quantity bounds the volume of $K$ itself. The result must be correct to full double precision across the whole range of budgets the function may be called with, including budgets substantially larger than any single application requires.

The construction rests on a subtraction. The pushforward of the uniform measure on the box through $g$ decomposes into a part living where $g \le 1$, which is entirely determined by the unknown volume together with the constants $z_k$, and a residual part living where $g > 1$. Subtracting the first from the whole must leave something that is still a genuine non-negative measure, and that requirement constrains the unknown scalar from above. The largest admissible $\alpha$ is therefore an upper bound on the normalized volume, and the tightest one the data supports.

Admissibility of a truncated moment sequence is a classical question with a linear-algebraic answer. A finite sequence arises as the moments of a non-negative measure on the real line precisely when a certain symmetric matrix built from it, constant along antidiagonals, is positive semidefinite. Since that matrix depends affinely on $\alpha$ and the subtracted part contributes one that is strictly positive definite, the whole condition reduces to a scalar question about $\alpha$.

The size of the matrix is not free. Building it consumes entries of the sequence up to an index determined by its side length, so the budget imposes a ceiling; going beyond that ceiling would require data that is not present, while stopping short of it discards information and loosens the bound. The dependence is monotone in a strong sense: deleting the last row and column of a positive semidefinite matrix leaves a positive semidefinite matrix, so the admissible set of $\alpha$ can only shrink as the budget grows, and the bound can only improve.

Numerically this construction is delicate. Matrices of this type built from measures with smooth densities on a bounded interval are of Hilbert type, whose condition number grows geometrically in the side length, while the entries of the moment sequence themselves grow geometrically in the index. Both effects compound with the budget, and the working precision required to resolve the answer grows linearly in the side length. Fixed-precision dense linear algebra is adequate only over a limited range, beyond which it returns a plausible but incorrect value rather than failing visibly. The execution environment provides NumPy and the Python standard library. No other package is available, and an implementation that depends on one will not run.

Returns
-------
float: an upper bound on the volume of the unit sublevel set of g, given the moment budget.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def compute_upper_bound(box_moments: np.ndarray, reference_moments: np.ndarray, dimension: int) -> float:
    """Compute the tightest upper bound on the volume forced by the moment data.

    Parameters
    ----------
    box_moments : np.ndarray
        Array of shape (K + 1,) and dtype float64 holding the pushforward
        moments [y_0, ..., y_K].
    reference_moments : np.ndarray
        Array of shape (K + 1,) and dtype float64 holding the constants
        [z_0, ..., z_K]. Must match box_moments in length.
    dimension : int
        The ambient dimension n.

    Returns
    -------
    float
        An upper bound on the volume of K = {x : g(x) <= 1}, namely the
        largest admissible scalar rescaled to bound the volume itself.

    Raises
    ------
    ValueError
        If the moment sequences are empty, differ in length, or are not
        one-dimensional, or if the subtracted part fails to be positive
        definite.
    """
    return 0.0

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
from fractions import Fraction


def _oracle_compute_upper_bound(box_moments: np.ndarray, reference_moments: np.ndarray, dimension: int) -> float:
    """Reference implementation: exact rational bisection on the definiteness of the affine matrix family."""
    y = np.asarray(box_moments, dtype=np.float64)
    z = np.asarray(reference_moments, dtype=np.float64)

    if y.ndim != 1 or z.ndim != 1:
        raise ValueError("moment sequences must be one-dimensional")
    if y.size != z.size or y.size == 0:
        raise ValueError("moment sequences must be non-empty and of equal length")

    Y = [Fraction(float(v)) for v in y]
    Z = [Fraction(float(v)) for v in z]

    s = (len(Y) - 1) // 2 + 1
    A = [[Y[i + j] for j in range(s)] for i in range(s)]
    B = [[Z[i + j] for j in range(s)] for i in range(s)]

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

    lo, hi = Fraction(0), Fraction(2) ** 40
    for _ in range(120):
        mid = (lo + hi) / 2
        M = [[A[i][j] - mid * B[i][j] for j in range(s)] for i in range(s)]
        if _h_pos_def(M):
            lo = mid
        else:
            hi = mid

    return float(2 ** int(dimension)) * float(lo)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Differential test cases: normal, boundary, and edge instances."""
    return [
        {
            # normal: the task instance at its budget k = 0..14, where the
            # matrix reaches side 8 and both its conditioning and the growth
            # of the moment entries place the answer beyond the reach of
            # fixed-precision dense linear algebra; double-precision routes
            # return a range of plausible but incorrect values depending on
            # the driver used
            "setup": "import numpy as np\nfrom fractions import Fraction\ncoef = [Fraction(2), Fraction(2), Fraction(2), Fraction(-2)]\nexpo = [(8, 0, 0), (0, 4, 0), (0, 0, 2), (4, 2, 0)]\npoly = {}\nfor c, e in zip(coef, expo):\n    poly[e] = poly.get(e, Fraction(0)) + c\ncur = {(0, 0, 0): Fraction(1)}\nys = []\nfor _ in range(15):\n    t = Fraction(0)\n    for a, c in cur.items():\n        if any(v % 2 for v in a):\n            continue\n        w = c\n        for v in a:\n            w /= (v + 1)\n        t += w\n    ys.append(float(t))\n    nxt = {}\n    for a, ca in cur.items():\n        for b, cb in poly.items():\n            k2 = tuple(p + q for p, q in zip(a, b))\n            nxt[k2] = nxt.get(k2, Fraction(0)) + ca * cb\n    cur = nxt\ny = np.array(ys)\nz = np.array([float(Fraction(7, 7 + 8 * k)) for k in range(15)])\nn = 3",
            "call": "compute_upper_bound(y, z, n)",
            "gold_call": "_oracle_compute_upper_bound(y, z, n)",
        },
        {
            # boundary: the same data at a reduced budget k = 0..6, giving a
            # strictly looser bound and confirming monotonicity in the budget
            "setup": "import numpy as np\nfrom fractions import Fraction\ny = np.array([1.0, 52.0 / 45.0, 410288.0 / 208845.0, 45964096.0 / 11486475.0, 64170635008.0 / 6995263275.0, 4627392439985152.0 / 201624473375325.0, 5664835468095488.0 / 92204876128365.0])\nz = np.array([float(Fraction(7, 7 + 8 * k)) for k in range(7)])\nn = 3",
            "call": "compute_upper_bound(y, z, n)",
            "gold_call": "_oracle_compute_upper_bound(y, z, n)",
        },
        {
            # edge: an odd budget k = 0..5, where the final moment cannot be
            # used and the matrix must not be built one size too large
            "setup": "import numpy as np\nfrom fractions import Fraction\ny = np.array([1.0, 52.0 / 45.0, 410288.0 / 208845.0, 45964096.0 / 11486475.0, 64170635008.0 / 6995263275.0, 4627392439985152.0 / 201624473375325.0])\nz = np.array([float(Fraction(7, 7 + 8 * k)) for k in range(6)])\nn = 3",
            "call": "compute_upper_bound(y, z, n)",
            "gold_call": "_oracle_compute_upper_bound(y, z, n)",
        },
        {
            # edge: a single moment, giving the degenerate side-1 case in
            # which the bound reduces to the volume of the box itself
            "setup": "import numpy as np\ny = np.array([1.0])\nz = np.array([1.0])\nn = 3",
            "call": "compute_upper_bound(y, z, n)",
            "gold_call": "_oracle_compute_upper_bound(y, z, n)",
        },
    ]
