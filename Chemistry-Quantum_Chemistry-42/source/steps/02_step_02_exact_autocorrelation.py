"""
Step 02: Exact autocorrelation by split-operator grid propagation.

Numerically exact wavepacket autocorrelation function after a vertical transition, by split-operator propagation on a grid.

In the time-dependent picture of electronic absorption at zero temperature and in the Condon approximation, the spectrum
is the Fourier transform of the autocorrelation function C(t) = <psi_0|psi_t>, where psi_0 is the vibrational ground
state of the initial electronic state and psi_t evolves under the vibrational Hamiltonian of the final electronic state.
For two vibrational coordinates the time-dependent Schrodinger equation can be solved essentially exactly on a Cartesian
grid, which provides the benchmark against which approximate semiclassical propagations are judged.

The initial electronic state is harmonic with its minimum at the origin. Its Hessian is R diag(omega_1^2, omega_2^2) R^T
with the rotation R = [[cos(theta), -sin(theta)], [sin(theta), cos(theta)]], so its normal modes are rotated by theta with
respect to the excited-state axes (a Duschinsky rotation). Its vibrational ground state is
    psi_0(q) = pi^(-1/2) (omega_1 omega_2)^(1/4) exp(-(1/2) q^T R diag(omega_1, omega_2) R^T q).
The final-state Hamiltonian is H = -(1/2) (d^2/dq1^2 + d^2/dq2^2) + V(q1, q2) with the anharmonic excited-state surface of
the previous step, in mass-weighted coordinates with unit masses and hbar = 1.

Returns
-------
numpy.ndarray of shape (n_steps + 1, 2), real and imaginary parts of the split-operator autocorrelation C(t_n)
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def exact_autocorrelation(displacement: float, surface_params: "np.ndarray", ground_params: "np.ndarray", q1_axis: "np.ndarray", q2_axis: "np.ndarray", time_step: float, n_steps: int) -> "np.ndarray":
    '''Autocorrelation C(t_n) = <psi_0|psi_(t_n)>, t_n = n time_step, from second-order split-operator propagation on a periodic grid.

    The discretization is fixed because the returned values depend on it: psi_0 is the analytic ground state sampled on the
    grid without renormalization; one step applies exp(-i V dt / 2), then the kinetic propagator exp(-i (k1^2 + k2^2) dt / 2)
    in the discrete Fourier representation with angular wavenumbers k = 2 pi m / (N h) of the unshifted FFT ordering, then
    exp(-i V dt / 2) again; and C(t_n) is the sum over grid points of conj(psi_0) psi_(t_n) times h1 h2, the product of
    the grid spacings.

    Parameters
    ----------
    displacement : float
        Stretch coordinate d of the excited-state minimum.
    surface_params : np.ndarray
        Array [omega_e, chi, omega_b, gamma, delta] of the excited-state surface.
    ground_params : np.ndarray
        Array [omega_1, omega_2, theta] with ground-state frequencies omega_1, omega_2 > 0 and Duschinsky angle theta in
        radians.
    q1_axis : np.ndarray
        Evenly spaced, increasing grid points of q1 (at least 4), treated as one period of a periodic grid.
    q2_axis : np.ndarray
        Evenly spaced, increasing grid points of q2 (at least 4), treated as one period of a periodic grid.
    time_step : float
        Time step dt > 0.
    n_steps : int
        Number of steps n_steps >= 0.

    Returns
    -------
    result : np.ndarray
        Array of shape (n_steps + 1, 2) whose row n is [Re C(t_n), Im C(t_n)].

    Raises
    ------
    ValueError
        If either axis is not one-dimensional, evenly spaced and increasing with at least 4 points, if time_step <= 0, if
        n_steps < 0, or if omega_1 or omega_2 is not strictly positive.
    '''
    return result  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_exact_autocorrelation(displacement: float, surface_params: "np.ndarray", ground_params: "np.ndarray", q1_axis: "np.ndarray", q2_axis: "np.ndarray", time_step: float, n_steps: int) -> "np.ndarray":
    """Reference implementation."""
    import numpy as np
    x1 = np.asarray(q1_axis, dtype=float)
    x2 = np.asarray(q2_axis, dtype=float)
    for ax in (x1, x2):
        if ax.ndim != 1 or ax.size < 4:
            raise ValueError("each axis must be one-dimensional with at least 4 points")
        steps = np.diff(ax)
        if steps[0] <= 0.0 or not np.allclose(steps, steps[0], rtol=1e-9, atol=0.0):
            raise ValueError("each axis must be evenly spaced and increasing")
    if not time_step > 0.0 or int(n_steps) < 0:
        raise ValueError("time_step must be positive and n_steps non-negative")
    w1, w2, theta = (float(v) for v in np.asarray(ground_params, dtype=float).ravel()[:3])
    if w1 <= 0.0 or w2 <= 0.0:
        raise ValueError("ground-state frequencies must be strictly positive")
    h1 = x1[1] - x1[0]
    h2 = x2[1] - x2[0]
    X1, X2 = np.meshgrid(x1, x2, indexing="ij")
    c, s = np.cos(theta), np.sin(theta)
    rot = np.array([[c, -s], [s, c]])
    width = rot @ np.diag([w1, w2]) @ rot.T
    quad = width[0, 0] * X1 * X1 + 2.0 * width[0, 1] * X1 * X2 + width[1, 1] * X2 * X2
    psi0 = np.pi ** -0.5 * (w1 * w2) ** 0.25 * np.exp(-0.5 * quad) + 0j
    pot = _oracle_excited_surface_derivatives(np.column_stack([X1.ravel(), X2.ravel()]), displacement, surface_params)[:, 0]
    half_v = np.exp(-0.5j * time_step * pot.reshape(X1.shape))
    k1 = 2.0 * np.pi * np.fft.fftfreq(x1.size, d=h1)
    k2 = 2.0 * np.pi * np.fft.fftfreq(x2.size, d=h2)
    kin = np.exp(-0.5j * time_step * (k1[:, None] ** 2 + k2[None, :] ** 2))
    weight = np.conj(psi0) * h1 * h2
    n = int(n_steps)
    out = np.empty((n + 1, 2))
    psi = psi0.copy()
    val = np.sum(weight * psi)
    out[0] = (val.real, val.imag)
    for i in range(n):
        psi = half_v * np.fft.ifft2(kin * np.fft.fft2(half_v * psi))
        val = np.sum(weight * psi)
        out[i + 1] = (val.real, val.imag)
    return out

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Normal: benchmark-like surface and Duschinsky angle, short propagation on a coarse grid ---
        {
            "setup": "import numpy as np\n"
                     "par = np.array([0.85, 0.02, 0.45, 0.3, 0.8])\n"
                     "gp = np.array([1.0, 0.5, np.deg2rad(20.0)])\n"
                     "a1 = np.linspace(-8.0, 16.0, 96, endpoint=False)\n"
                     "a2 = np.linspace(-8.0, 8.0, 32, endpoint=False)\n",
            "call": "exact_autocorrelation(2.0, par.copy(), gp.copy(), a1.copy(), a2.copy(), 0.05, 60)",
            "gold_call": "_oracle_exact_autocorrelation(2.0, par.copy(), gp.copy(), a1.copy(), a2.copy(), 0.05, 60)",
            "tol": 1e-9,
        },
        # --- Edge: negative Duschinsky angle, negative stretch-bend coupling and a stiffer, more anharmonic stretch ---
        {
            "setup": "import numpy as np\n"
                     "par = np.array([1.1, 0.04, 0.6, -0.4, -0.5])\n"
                     "gp = np.array([0.8, 0.9, -0.6])\n"
                     "a1 = np.linspace(-6.0, 14.0, 64, endpoint=False)\n"
                     "a2 = np.linspace(-7.0, 7.0, 48, endpoint=False)\n",
            "call": "exact_autocorrelation(1.3, par.copy(), gp.copy(), a1.copy(), a2.copy(), 0.08, 45)",
            "gold_call": "_oracle_exact_autocorrelation(1.3, par.copy(), gp.copy(), a1.copy(), a2.copy(), 0.08, 45)",
            "tol": 1e-9,
        },
        # --- Boundary: zero steps returns only C(0), the discrete norm of the sampled ground state ---
        {
            "setup": "import numpy as np\n"
                     "par = np.array([0.85, 0.02, 0.45, 0.3, 0.8])\n"
                     "gp = np.array([1.4, 0.3, 1.2])\n"
                     "a1 = np.linspace(-5.0, 5.0, 20, endpoint=False)\n"
                     "a2 = np.linspace(-9.0, 9.0, 24, endpoint=False)\n",
            "call": "exact_autocorrelation(0.5, par.copy(), gp.copy(), a1.copy(), a2.copy(), 0.1, 0)",
            "gold_call": "_oracle_exact_autocorrelation(0.5, par.copy(), gp.copy(), a1.copy(), a2.copy(), 0.1, 0)",
            "tol": 1e-12,
        },
        # --- Normal: undisplaced minimum with a displaced bend, long step on an elongated grid ---
        {
            "setup": "import numpy as np\n"
                     "par = np.array([0.7, 0.015, 0.3, 0.8, 1.2])\n"
                     "gp = np.array([0.9, 0.35, 0.3])\n"
                     "a1 = np.linspace(-9.0, 12.0, 70, endpoint=False)\n"
                     "a2 = np.linspace(-10.0, 12.0, 40, endpoint=False)\n",
            "call": "exact_autocorrelation(0.0, par.copy(), gp.copy(), a1.copy(), a2.copy(), 0.2, 40)",
            "gold_call": "_oracle_exact_autocorrelation(0.0, par.copy(), gp.copy(), a1.copy(), a2.copy(), 0.2, 40)",
            "tol": 1e-9,
        },
        # --- Error: an unevenly spaced axis must raise ValueError ---
        {
            "setup": "import numpy as np\n"
                     "def _probe(fn):\n"
                     "    try:\n"
                     "        fn(1.0, np.array([0.85, 0.02, 0.45, 0.3, 0.8]), np.array([1.0, 0.5, 0.2]),\n"
                     "           np.array([0.0, 1.0, 2.0, 3.5, 4.0]), np.linspace(-4.0, 4.0, 8, endpoint=False), 0.1, 3)\n"
                     "    except ValueError:\n"
                     "        return 1\n"
                     "    except Exception:\n"
                     "        return 2\n"
                     "    return 0\n",
            "call": "_probe(exact_autocorrelation)",
            "gold_call": "_probe(_oracle_exact_autocorrelation)",
        },
    ]
