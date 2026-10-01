"""
Integrate a matrix GQME with drift and a sampled causal memory kernel using nested trapezoidal quadrature and an implicit endpoint.

The correlation matrix obeys



$$

C'(t)=AC(t)+\int_0^t K(t-s)C(s)\,ds,

\qquad C(0)=I.

$$



Use the trapezoidal rule for the convolution at every grid point and for the outer time integral. The integral is zero at the initial time. Solve the unknown endpoint contributions together. Matrix products retain the displayed order.

Returns
-------
A real NumPy array of shape (N + 1, d, d), containing the correlation matrix at every grid point, including the initial identity matrix.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def propagate_correlation(kernel, drift, dt):
    """Integrate the drift-plus-memory GQME on a uniform grid.

    Parameters
    ----------
    kernel : array_like, real, shape (N + 1, d, d)
        K_1 sampled at times n*dt; N >= 1 and d >= 1.
    drift : array_like, real, shape (d, d)
        Constant left-acting drift A. It need not commute with K_1.
    dt : float
        Finite positive time step.

    Returns
    -------
    ndarray, real, shape (N + 1, d, d)
        Correlation matrices from the two trapezoidal quadratures
        in the background, with C(0) = identity. Retain both the
        initial drift derivative and the implicit convolution endpoint.

    Raises
    ------
    ValueError
        If shapes are incorrect, any entry is nonfinite, dt is
        nonfinite or nonpositive, or the implicit coefficient matrix
        has 2-norm condition number greater than 1e12.
    """
    import numpy as np
    return np.zeros_like(kernel, dtype=float)

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_propagate_correlation(kernel, drift, dt):
    import numpy as np

    k = np.asarray(kernel, dtype=float)
    a = np.asarray(drift, dtype=float)

    if (
        k.ndim != 3
        or len(k) < 2
        or k.shape[1] == 0
        or k.shape[1] != k.shape[2]
        or not np.isfinite(k).all()
        or a.shape != k.shape[1:]
        or not np.isfinite(a).all()
        or not np.isfinite(dt)
        or dt <= 0
    ):
        raise ValueError("Invalid kernel grid or time step.")

    identity = np.eye(k.shape[1])
    lhs = identity - dt * a / 2 - dt * dt * k[0] / 4

    if np.linalg.cond(lhs) > 1e12:
        raise ValueError(
            "Implicit endpoint matrix is ill-conditioned."
        )

    correlation = np.empty_like(k)
    correlation[0] = identity
    derivative = a.copy()

    for n in range(1, len(k)):
        interior = np.zeros_like(identity)

        for j in range(1, n):
            interior += k[n - j] @ correlation[j]

        known = dt * (k[n] / 2 + interior)

        correlation[n] = np.linalg.solve(
            lhs,
            correlation[n - 1]
            + dt * (derivative + known) / 2,
        )

        derivative = (
            a @ correlation[n]
            + known
            + dt * k[0] @ correlation[n] / 2
        )

    return correlation

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": """import numpy as np
k = np.array([
    [[-.4, .2], [-.1, -.3]],
    [[-.3, .1], [.05, -.2]],
    [[-.2, -.1], [.2, -.1]],
])
""",
            "call": "propagate_correlation(k, np.zeros((2, 2)), .2)",
            "gold_call": "_oracle_propagate_correlation(k, np.zeros((2, 2)), .2)",
        },
        {
            "setup": """import numpy as np
k = np.zeros((2, 2, 2))
""",
            "call": "propagate_correlation(k, np.zeros((2, 2)), .5)",
            "gold_call": "_oracle_propagate_correlation(k, np.zeros((2, 2)), .5)",
        },
        {
            "setup": """import numpy as np
k = np.array([
    [[0., 1.], [-2., 0.]],
    [[.2, -.3], [.4, -.1]],
])
""",
            "call": "propagate_correlation(k, np.zeros((2, 2)), .1)",
            "gold_call": "_oracle_propagate_correlation(k, np.zeros((2, 2)), .1)",
        },
        {
            "setup": """import numpy as np
k = np.array([
    [[.2, -.4], [.1, -.3]],
    [[.1, .2], [-.5, .4]],
    [[.3, -.1], [.2, .1]],
    [[.05, .2], [.15, -.1]],
])
a = np.array([[.1, -.7], [.3, -.2]])
""",
            "call": "propagate_correlation(k, a, .07)",
            "gold_call": "_oracle_propagate_correlation(k, a, .07)",
        },
        {
            "setup": """import numpy as np
k = np.zeros((8, 2, 2))
a = np.array([[.1, -.7], [.3, -.2]])
""",
            "call": "propagate_correlation(k, a, .07)",
            "gold_call": "_oracle_propagate_correlation(k, a, .07)",
        },
        {
            "setup": """import numpy as np
k = np.array([
    [[.7, -.2], [.1, .4]],
    [[-.3, .6], [.2, -.1]],
])
a = np.array([[.1, -.7], [.3, -.2]])
""",
            "call": "propagate_correlation(k, a, .05)",
            "gold_call": "_oracle_propagate_correlation(k, a, .05)",
        },
    ]
