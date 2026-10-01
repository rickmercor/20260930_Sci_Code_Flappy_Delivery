"""
The row index labels the initial operator. Define



$$

PX=

\sum_j

\operatorname{Tr}[X(\phi_j\otimes\rho_b)]

(\phi_j\otimes I_b),

\qquad Q=1-P,

$$



$$

f_i(t)=e^{tQD}QD(\phi_i\otimes I_b),

$$



$$

K_{n,ij}(t)=

\operatorname{Tr}[D^n f_i(t)(\phi_j\otimes\rho_b)].

$$



Derive the initial kernels from this definition. The hierarchy is



$$

K_n'(t)=K_{n+1}(t)-K_1(t)\Omega_n.

$$



Differentiate it to obtain the requested Taylor coefficients. Inputs satisfying the signature are admissible moment arrays; no Hamiltonian-realizability test is required.

For zero projected drift, K_n(0) = Omega_(n+1). Repeated differentiation of K'_n = K_(n+1) - K_1 Omega_n gives the derivatives of K_3 at zero. Preserve the matrix-product order and divide each derivative of order m by m! before returning the coefficients.

Returns
-------
A real NumPy array of shape (9, 4, 4). Entry m contains the m-th derivative of K_3 at zero divided by m!, for m = 0,...,8.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def kernel_coefficients(moments):
    """Return third-kernel Taylor coefficients in the stated row convention.

    Parameters
    ----------
    moments : array_like, real, shape (13, 4, 4)
        Omega_0 through Omega_12. Omega_0 is identity within 1e-10
        entrywise. Omega_1 may be any finite matrix; retain it.

    Returns
    -------
    ndarray, real, shape (9, 4, 4)
        Entry m is the m-th derivative of K_3 at zero divided by m!,
        for m = 0,...,8, using the operator definition in the background.

    Raises
    ------
    ValueError
        If shape is incorrect, any entry is nonfinite, or Omega_0
        differs from identity by more than 1e-10 entrywise.
    """
    import numpy as np
    return np.zeros((9, 4, 4))

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_kernel_coefficients(moments):
    import math
    import numpy as np

    w = np.asarray(moments, dtype=float)

    if (
        w.shape != (13, 4, 4)
        or not np.isfinite(w).all()
        or not np.allclose(w[0], np.eye(4), atol=1e-10, rtol=0)
    ):
        raise ValueError(
            "Require thirteen finite moments with identity zeroth moment."
        )

    w = w.copy()

    previous = np.array([
        w[n + 1] - w[1] @ w[n]
        for n in range(1, 12)
    ])

    coefficients = [previous[2].copy()]

    for derivative_order in range(1, 9):
        current = np.array([
            previous[n] - previous[0] @ w[n]
            for n in range(1, len(previous))
        ])

        coefficients.append(
            current[2] / math.factorial(derivative_order)
        )

        previous = current

    return np.array(coefficients)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": """import numpy as np
w = np.zeros((13, 4, 4))
w[0] = np.eye(4)
b = np.arange(16.).reshape(4, 4) / 20
for n in range(2, 13):
    w[n] = np.roll(b, n, axis=0) / (n + 1)
""",
            "call": "kernel_coefficients(w)",
            "gold_call": "_oracle_kernel_coefficients(w)",
        },
        {
            "setup": """import numpy as np
w = np.zeros((13, 4, 4))
w[0] = np.eye(4)
""",
            "call": "kernel_coefficients(w)",
            "gold_call": "_oracle_kernel_coefficients(w)",
        },
        {
            "setup": """import numpy as np
w = np.zeros((13, 4, 4))
w[0] = np.eye(4)
w[12, 1, 2] = 40320.
""",
            "call": "kernel_coefficients(w)",
            "gold_call": "_oracle_kernel_coefficients(w)",
        },
        {
            "setup": """import numpy as np
rng = np.random.default_rng(1603)
w = rng.normal(0, .15, (13, 4, 4))
w[0] = np.eye(4)
""",
            "call": "kernel_coefficients(w)",
            "gold_call": "_oracle_kernel_coefficients(w)",
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
""",
            "call": "kernel_coefficients(w)",
            "gold_call": "_oracle_kernel_coefficients(w)",
        },
        {
            "setup": """import numpy as np
rng = np.random.default_rng(1603)
w = rng.normal(0, .15, (13, 4, 4))
w[0] = np.eye(4)
w[1] *= 1e-10
w[11] *= 1000
""",
            "call": "kernel_coefficients(w)",
            "gold_call": "_oracle_kernel_coefficients(w)",
        },
    ]
