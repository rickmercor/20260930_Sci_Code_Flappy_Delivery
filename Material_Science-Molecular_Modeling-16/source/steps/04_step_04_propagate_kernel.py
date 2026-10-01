"""
Propagate the first two auxiliary kernels with classical RK4 and an elementwise rational third-kernel closure, retaining nonzero drift.

Use levels one and two of the hierarchy defined in step 2. Initialize both levels from that step's operator definition. Evaluate the prescribed third-kernel closure at each RK4 stage; its denominator acts entrywise. Return the first kernel on the full grid.

Returns
-------
A real NumPy array of shape (steps + 1, 4, 4), containing K_1 at all uniform grid points, including zero.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def propagate_kernel(moments, closure, duration, steps):
    """Propagate two retained auxiliary kernels with nonzero drift.

    Parameters
    ----------
    moments : array_like, real, shape (13, 4, 4)
        Omega_0 through Omega_12. Use Omega_1 as the drift.
    closure : array_like, real, shape (2, 5, 4, 4)
        Numerator followed by denominator of K_3, ascending powers.
    duration : float
        Finite positive final time.
    steps : int
        Number of uniform classical RK4 intervals, from 1 to 4096.

    Returns
    -------
    ndarray, real, shape (steps + 1, 4, 4)
        K_1 at every grid point, including zero. Use the hierarchy
        and operator-defined initial values in step 2. Evaluate K_3
        at each RK4 stage; do not interpolate endpoint values.

    Raises
    ------
    ValueError
        If either array shape is incorrect, any entry is nonfinite,
        duration is nonfinite or nonpositive, steps is not an integer
        in [1, 4096], or any scalar closure denominator has magnitude
        at most 1e-10 at an RK4 stage time.
    """
    import numpy as np
    return np.zeros((steps + 1, 4, 4))

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_propagate_kernel(moments, closure, duration, steps):
    import numpy as np

    w = np.asarray(moments, dtype=float)
    pq = np.asarray(closure, dtype=float)

    if (
        w.shape != (13, 4, 4)
        or pq.shape != (2, 5, 4, 4)
        or not np.isfinite(w).all()
        or not np.isfinite(pq).all()
        or not np.isfinite(duration)
        or duration <= 0
        or not isinstance(steps, (int, np.integer))
        or not 1 <= steps <= 4096
    ):
        raise ValueError("Invalid kernel propagation inputs.")

    h = duration / steps
    times = np.arange(2 * steps + 1) * h / 2

    numerator = np.polynomial.polynomial.polyval(
        times, pq[0]
    ).transpose(2, 0, 1)

    denominator = np.polynomial.polynomial.polyval(
        times, pq[1]
    ).transpose(2, 0, 1)

    if np.min(np.abs(denominator)) <= 1e-10:
        raise ValueError(
            "Closure denominator vanishes at a stage time."
        )

    values = numerator / denominator

    x = w[2] - w[1] @ w[1]
    y = w[3] - w[1] @ w[2]

    result = np.empty((steps + 1, 4, 4))
    result[0] = x

    for n in range(steps):
        ax = y - x @ w[1]
        ay = values[2 * n] - x @ w[2]

        bx = y + h * ay / 2 - (x + h * ax / 2) @ w[1]
        by = (
            values[2 * n + 1]
            - (x + h * ax / 2) @ w[2]
        )

        cx = y + h * by / 2 - (x + h * bx / 2) @ w[1]
        cy = (
            values[2 * n + 1]
            - (x + h * bx / 2) @ w[2]
        )

        dx = y + h * cy - (x + h * cx) @ w[1]
        dy = (
            values[2 * n + 2]
            - (x + h * cx) @ w[2]
        )

        x = x + h * (ax + 2 * bx + 2 * cx + dx) / 6
        y = y + h * (ay + 2 * by + 2 * cy + dy) / 6

        result[n + 1] = x

    return result

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": """import numpy as np
w = np.zeros((13, 4, 4))
w[0] = np.eye(4)
w[2] = np.diag([0., -.4, -.7, -.2])
w[3, 1, 2] = .13
w[3, 2, 1] = -.1
pq = np.zeros((2, 5, 4, 4))
pq[1, 0] = 1.
pq[0, 0] = .2 * np.eye(4)
""",
            "call": "propagate_kernel(w, pq, .4, 8)",
            "gold_call": "_oracle_propagate_kernel(w, pq, .4, 8)",
        },
        {
            "setup": """import numpy as np
w = np.zeros((13, 4, 4))
w[0] = np.eye(4)
pq = np.zeros((2, 5, 4, 4))
pq[1, 0] = 1.
""",
            "call": "propagate_kernel(w, pq, 1., 1)",
            "gold_call": "_oracle_propagate_kernel(w, pq, 1., 1)",
        },
        {
            "setup": """import numpy as np
w = np.zeros((13, 4, 4))
w[0] = np.eye(4)
w[2, 1, 2] = -.3
w[2, 2, 1] = .2
pq = np.zeros((2, 5, 4, 4))
pq[1, 0] = 1.
pq[1, 2] = .2
pq[0, 0] = np.arange(16.).reshape(4, 4) / 30
""",
            "call": "propagate_kernel(w, pq, .5, 5)",
            "gold_call": "_oracle_propagate_kernel(w, pq, .5, 5)",
        },
        {
            "setup": """import numpy as np
rng = np.random.default_rng(1603)
w = rng.normal(0, .15, (13, 4, 4))
w[0] = np.eye(4)
pq = np.zeros((2, 5, 4, 4))
pq[1, 0] = 1.
pq[1, 1] = .4
pq[1, 2] = .2
pq[0, 0] = rng.normal(0, .2, (4, 4))
pq[0, 1] = rng.normal(0, .1, (4, 4))
""",
            "call": "propagate_kernel(w, pq, .3, 17)",
            "gold_call": "_oracle_propagate_kernel(w, pq, .3, 17)",
        },
        {
            "setup": """import numpy as np
a = np.array([
    [0., 0., 0., 0.],
    [0., 0., -.38, .14],
    [0., .38, 0., -.26],
    [0., -.14, .26, 0.],
])
w = np.array([
    np.linalg.matrix_power(a, n)
    for n in range(13)
])
pq = np.zeros((2, 5, 4, 4))
pq[1, 0] = 1.
""",
            "call": "propagate_kernel(w, pq, .8, 1)",
            "gold_call": "_oracle_propagate_kernel(w, pq, .8, 1)",
        },
    ]
