"""
Compute the measured population by composing moments, third-kernel closure, kernel propagation and drift-aware correlation propagation. Compute the measured population by composing moments, third-kernel closure, kernel propagation and drift-aware correlation propagation.

Use moments through order 12 and the third-kernel [4/4] closure. Pass the first moment to the correlation propagator as its drift. Measure the last correlation matrix using the supplied Bloch vectors.

Returns
-------
A native Python float containing the population predicted by the complete discretized calculation.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def solve(h, hb, beta, initial, axis, duration, steps):
    """Return the population from the complete nonzero-drift calculation.

    Parameters
    ----------
    h : array_like, complex, shape (2*b, 2*b)
        Finite Hermitian total Hamiltonian, system-first ordering.
    hb : array_like, complex, shape (b, b)
        Finite Hermitian bath Hamiltonian; b >= 1.
    beta : float
        Finite nonnegative inverse temperature.
    initial : array_like, real, shape (3,)
        Finite Bloch vector of norm at most 1 + 1e-12.
    axis : array_like, real, shape (3,)
        Finite measurement axis with norm within 1e-12 of one.
    duration : float
        Finite positive final time.
    steps : int
        Number of uniform intervals, from 1 to 4096.

    Returns
    -------
    float
        Population from moments through order 12, elementwise [4/4]
        closure, classical RK4 for two retained kernels and nested
        trapezoidal correlation propagation with drift Omega_1.
        Compose the preceding six functions; do not propagate H directly.

    Raises
    ------
    ValueError
        For invalid input shapes or values as specified above, Hermitian
        defects above 1e-12 entrywise, Padé matching residual above
        1e-10*max(1,max(abs(c))) with rcond=1e-12, a stage denominator
        magnitude at most 1e-10, or implicit condition number above 1e12.
    """
    return 0.0

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_solve(h, hb, beta, initial, axis, duration, steps):
    moments = _oracle_compute_moments(
        h, hb, beta, 12
    )

    coefficients = _oracle_kernel_coefficients(
        moments
    )

    closure = _oracle_fit_closure(
        coefficients
    )

    kernel = _oracle_propagate_kernel(
        moments, closure, duration, steps
    )

    correlation = _oracle_propagate_correlation(
        kernel, moments[1], duration / steps
    )

    return _oracle_measure_population(
        correlation[-1], initial, axis
    )

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": """import numpy as np
x = np.array([[0., 1.], [1., 0.]])
y = np.array([[0., -1j], [1j, 0.]])
z = np.diag([1., -1.])
hb = np.diag([0., .7, 1.3])
bx = np.array([[0., .23, .11], [.23, 0., .17], [.11, .17, 0.]])
bz = np.array([[0., .13, -.19], [.13, 0., .07], [-.19, .07, 0.]])
hs = .13*x + .07*y + .19*z
h = (
    np.kron(hs, np.eye(3))
    + np.kron(np.eye(2), hb)
    + np.kron(x, bx)
    + np.kron(z, bz)
)
""",
            "call": "solve(h, hb, 1.4, [.3,-.4,.5], [0.,0.,1.], .8, 256)",
            "gold_call": "0.6757529792100742",
        },
        {
            "setup": """import numpy as np
x = np.array([[0., 1.], [1., 0.]])
y = np.array([[0., -1j], [1j, 0.]])
z = np.diag([1., -1.])
hb = np.diag([0., .7, 1.3])
hs = .13*x + .07*y + .19*z
h = np.kron(hs, np.eye(3)) + np.kron(np.eye(2), hb)
""",
            "call": "solve(h, hb, 1.4, [.3,-.4,.5], [0.,0.,1.], .8, 256)",
            "gold_call": "0.6874629736139558",
        },
        {
            "setup": """import numpy as np
x = np.array([[0., 1.], [1., 0.]])
y = np.array([[0., -1j], [1j, 0.]])
z = np.diag([1., -1.])
hb = np.diag([0., .7, 1.3])
bx = np.array([[0., .23, .11], [.23, 0., .17], [.11, .17, 0.]])
bz = np.array([[0., .13, -.19], [.13, 0., .07], [-.19, .07, 0.]])
hs = .13*x + .07*y + .19*z
h = (
    np.kron(hs, np.eye(3))
    + np.kron(np.eye(2), hb)
    + np.kron(x, bx)
    + np.kron(z, bz)
)
""",
            "call": "solve(h, hb, 0., [.3,-.4,.5], [0.,0.,1.], .8, 256)",
            "gold_call": "0.677538592857984",
        },
        {
            "setup": """import numpy as np
x = np.array([[0., 1.], [1., 0.]])
y = np.array([[0., -1j], [1j, 0.]])
z = np.diag([1., -1.])
hb = np.diag([0., .7, 1.3])
bx = np.array([[0., .23, .11], [.23, 0., .17], [.11, .17, 0.]])
bz = np.array([[0., .13, -.19], [.13, 0., .07], [-.19, .07, 0.]])
hs = .13*x + .07*y + .19*z
h = (
    np.kron(hs, np.eye(3))
    + np.kron(np.eye(2), hb)
    + np.kron(x, bx)
    + np.kron(z, bz)
)
""",
            "call": "solve(h, hb, 1.4, [.3,-.4,.5], [1.,0.,0.], .8, 256)",
            "gold_call": "0.7252340365486489",
        },
        {
            "setup": """import numpy as np
x = np.array([[0., 1.], [1., 0.]])
y = np.array([[0., -1j], [1j, 0.]])
z = np.diag([1., -1.])
hb = np.diag([0., .7, 1.3])
bx = np.array([[0., .23, .11], [.23, 0., .17], [.11, .17, 0.]])
bz = np.array([[0., .13, -.19], [.13, 0., .07], [-.19, .07, 0.]])
hs = .13*x + .07*y + .19*z
h = (
    np.kron(hs, np.eye(3))
    + np.kron(np.eye(2), hb)
    + np.kron(x, bx)
    + np.kron(z, bz)
)
h -= np.kron(hs, np.eye(3))
""",
            "call": "solve(h, hb, 1.4, [.3,-.4,.5], [0.,0.,1.], .8, 256)",
            "gold_call": "0.7327143985663065",
        },
        {
            "setup": """import numpy as np
x = np.array([[0., 1.], [1., 0.]])
y = np.array([[0., -1j], [1j, 0.]])
z = np.diag([1., -1.])
hb = np.diag([0., .7, 1.3])
bx = np.array([[0., .23, .11], [.23, 0., .17], [.11, .17, 0.]])
bz = np.array([[0., .13, -.19], [.13, 0., .07], [-.19, .07, 0.]])
hs = .13*x + .07*y + .19*z
h = (
    np.kron(hs, np.eye(3))
    + np.kron(np.eye(2), hb)
    + np.kron(x, bx)
    + np.kron(z, bz)
)
u = np.array([
    [1, 1j, 0],
    [1j, 1, 0],
    [0, 0, np.sqrt(2)],
]) / np.sqrt(2)
v = np.kron(np.eye(2), u)
h = v @ h @ v.conj().T
hb = u @ hb @ u.conj().T
""",
            "call": "solve(h, hb, 1.4, [.3,-.4,.5], [0.,0.,1.], .8, 256)",
            "gold_call": "0.6757529792100742",
        },
        {
            "setup": """import numpy as np
x = np.array([[0., 1.], [1., 0.]])
y = np.array([[0., -1j], [1j, 0.]])
z = np.diag([1., -1.])
hb = np.diag([0., .7, 1.3])
bx = np.array([[0., .23, .11], [.23, 0., .17], [.11, .17, 0.]])
bz = np.array([[0., .13, -.19], [.13, 0., .07], [-.19, .07, 0.]])
hs = .13*x + .07*y + .19*z
h = (
    np.kron(hs, np.eye(3))
    + np.kron(np.eye(2), hb)
    + np.kron(x, bx)
    + np.kron(z, bz)
)
h = h + 2*np.eye(6)
hb = hb + 2*np.eye(3)
""",
            "call": "solve(h, hb, 1.4, [.3,-.4,.5], [0.,0.,1.], .8, 256)",
            "gold_call": "0.6757529792100742",
        },
    ]
