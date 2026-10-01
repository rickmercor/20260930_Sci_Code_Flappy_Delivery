"""
Fit an elementwise [4/4] Padé closure to the third-kernel Taylor coefficients. Return numerator and denominator coefficients in ascending powers.

Each matrix entry has its own scalar rational approximant. Normalize its denominator with q_0 = 1 and match the supplied series through degree eight. For rank-deficient matching systems, use the specified minimum-norm solution. A zero series has a zero numerator and denominator one.

Returns
-------
A real NumPy array of shape (2, 5, 4, 4). Index 0 contains numerator coefficients; index 1 contains denominator coefficients. Polynomial coefficients are in ascending powers.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def fit_closure(coefficients):
    """Fit the elementwise [4/4] third-kernel closure.

    Parameters
    ----------
    coefficients : array_like, real, shape (9, 4, 4)
        Taylor coefficients c_0 through c_8, not raw derivatives.

    Returns
    -------
    ndarray, real, shape (2, 5, 4, 4)
        Numerator and denominator coefficients in ascending powers.
        Every denominator has constant coefficient one.
        Solve coefficient matching by minimum-norm least squares
        with relative singular-value cutoff 1e-12.

    Raises
    ------
    ValueError
        If shape is incorrect, an entry is nonfinite, or a scalar
        matching system has maximum residual greater than
        1e-10*max(1, max(abs(c))) after the prescribed solve.
    """
    return np.zeros((2, 5, 4, 4))

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_fit_closure(coefficients):
    import numpy as np

    c = np.asarray(coefficients, dtype=float)

    if c.shape != (9, 4, 4) or not np.isfinite(c).all():
        raise ValueError(
            "Require nine finite 4 by 4 Taylor coefficients."
        )

    result = np.zeros((2, 5, 4, 4))
    result[1, 0] = 1.0

    for i in range(4):
        for j in range(4):
            a = c[:, i, j]

            matrix = np.array([
                [a[k - m] for m in range(1, 5)]
                for k in range(5, 9)
            ])

            rhs = -a[5:9]

            q = np.linalg.lstsq(
                matrix, rhs, rcond=1e-12
            )[0]

            scale = max(1.0, np.max(np.abs(a)))

            if np.max(np.abs(matrix @ q - rhs)) > 1e-10 * scale:
                raise ValueError(
                    "Inconsistent normalized Pade system."
                )

            result[1, 1:, i, j] = q

            result[0, :, i, j] = [
                a[k] + sum(
                    q[m - 1] * a[k - m]
                    for m in range(1, k + 1)
                )
                for k in range(5)
            ]

    return result

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": """import numpy as np
p = np.array([1., .2, -.3, .1, .07])
q = np.array([1., .3, .4, .1, .02])
s = np.zeros(9)
for n in range(9):
    s[n] = (p[n] if n < 5 else 0.) - sum(
        q[j] * s[n-j] for j in range(1, min(4, n) + 1)
    )
c = s[:, None, None] * np.arange(16.).reshape(1, 4, 4)
""",
            "call": "fit_closure(c)",
            "gold_call": "_oracle_fit_closure(c)",
        },
        {
            "setup": """import numpy as np
c = np.zeros((9, 4, 4))
""",
            "call": "fit_closure(c)",
            "gold_call": "_oracle_fit_closure(c)",
        },
        {
            "setup": """import numpy as np
c = np.zeros((9, 4, 4))
c[0] = np.eye(4)
c[3, 1, 2] = 2.
""",
            "call": "fit_closure(c)",
            "gold_call": "_oracle_fit_closure(c)",
        },
    ]
