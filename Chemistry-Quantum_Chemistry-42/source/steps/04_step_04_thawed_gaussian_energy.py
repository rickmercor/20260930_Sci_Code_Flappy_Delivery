"""
Step 04: True energy along a thawed Gaussian propagation.

True energy expectation value along a thawed Gaussian propagation with a local or a constant reference Hessian.

A thawed Gaussian propagated in an effective quadratic potential conserves the expectation value of that effective
Hamiltonian only when the effective potential derives from a fixed Hamiltonian function on the manifold of Gaussians. The
energy that matters physically is the expectation value of the true Hamiltonian in the propagated Gaussian. With a
constant, positive-definite reference Hessian the width oscillates boundedly, so in a bounded region of the surface this
energy only oscillates; when the Hessian is re-evaluated along the trajectory the width can grow without bound and the true
energy drifts. Monitoring it along the time series separates the two behaviours.

The expectation value of the Hamiltonian in a normalized Gaussian follows from its first and second moments. The kinetic
energy depends on the centre momentum and on the momentum spread, and the potential energy of the Morse stretch with the
stretch-dependent bend involves only Gaussian averages of exponentials of the coordinates times polynomials, which are
available in closed form. Coordinates are mass-weighted with unit masses and hbar = 1.

Returns
-------
numpy.ndarray of shape (n_steps + 1,), expectation value of the excited-state Hamiltonian in the thawed Gaussian at each step
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def thawed_gaussian_energy(displacement: float, surface_params: "np.ndarray", ground_params: "np.ndarray", hessian_mode: int, time_step: float, n_steps: int) -> "np.ndarray":
    '''Expectation value of the excited-state Hamiltonian -(1/2) Laplacian + V in the thawed Gaussian at every step.

    The Gaussian at step n is the one defined by the previous step: it starts as the vibrational ground state of the initial
    electronic state and is advanced by the same fourth-order composition of exact kinetic and potential flows with the same
    choice of K. The returned values are exact expectation values of the true Hamiltonian in those Gaussians, to rounding.

    Parameters
    ----------
    displacement : float
        Stretch coordinate d of the excited-state minimum.
    surface_params : np.ndarray
        Array [omega_e, chi, omega_b, gamma, delta] of the excited-state surface V.
    ground_params : np.ndarray
        Array [omega_1, omega_2, theta] with omega_1, omega_2 > 0 and theta in radians.
    hessian_mode : int
        Choice of K as in the previous step: 0 local harmonic, 1 excited-state minimum, 2 origin, 3 initial-state Hessian.
    time_step : float
        Time step dt > 0.
    n_steps : int
        Number of steps n_steps >= 0.

    Returns
    -------
    result : np.ndarray
        Array of shape (n_steps + 1,) with the energy expectation value E_n = <psi_n|H|psi_n>.

    Raises
    ------
    ValueError
        If hessian_mode is not 0, 1, 2 or 3, if time_step <= 0, if n_steps < 0, if omega_1 or omega_2 is not strictly
        positive, or if omega_e, chi or omega_b is not strictly positive.
    '''
    return result  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_thawed_gaussian_energy(displacement: float, surface_params: "np.ndarray", ground_params: "np.ndarray", hessian_mode: int, time_step: float, n_steps: int) -> "np.ndarray":
    """Reference implementation."""
    import numpy as np
    par = np.asarray(surface_params, dtype=float).ravel()
    if par.size != 5 or not np.all(np.isfinite(par)):
        raise ValueError("surface_params must hold five finite numbers")
    omega_e, chi, omega_b, gamma, delta = par
    if omega_e <= 0.0 or chi <= 0.0 or omega_b <= 0.0:
        raise ValueError("omega_e, chi and omega_b must be strictly positive")
    q, p, Q, P, S, roots = _gaussian_series(displacement, surface_params, ground_params, hessian_mode, time_step, n_steps)
    depth = omega_e / (4.0 * chi)
    a = np.sqrt(2.0 * omega_e * chi)
    sigma = 0.5 * np.real(Q @ np.conj(np.transpose(Q, (0, 2, 1))))
    pi_mom = 0.5 * np.real(P @ np.conj(np.transpose(P, (0, 2, 1))))
    kinetic = 0.5 * np.sum(p ** 2, axis=1) + 0.5 * np.trace(pi_mom, axis1=1, axis2=2)
    mx = q[:, 0] - float(displacement)
    my = q[:, 1] - delta
    s11 = sigma[:, 0, 0]
    s12 = sigma[:, 0, 1]
    s22 = sigma[:, 1, 1]
    # E[exp(-k x)] = exp(-k mu_x + k^2 s11 / 2); under that tilt y has mean mu_y - k s12 and variance s22
    e1 = np.exp(-a * mx + 0.5 * a * a * s11)
    e2 = np.exp(-2.0 * a * mx + 2.0 * a * a * s11)
    eg = np.exp(-gamma * mx + 0.5 * gamma * gamma * s11)
    potential = depth * (1.0 - 2.0 * e1 + e2) + 0.5 * omega_b * omega_b * eg * ((my - gamma * s12) ** 2 + s22)
    return kinetic + potential

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Normal: adiabatic reference Hessian on the benchmark-like surface, where the energy only oscillates ---
        {
            "setup": "import numpy as np\n"
                     "par = np.array([0.85, 0.02, 0.45, 0.3, 0.8])\n"
                     "gp = np.array([1.0, 0.5, np.deg2rad(20.0)])\n",
            "call": "thawed_gaussian_energy(2.2, par.copy(), gp.copy(), 1, 0.05, 400)",
            "gold_call": "_oracle_thawed_gaussian_energy(2.2, par.copy(), gp.copy(), 1, 0.05, 400)",
            "tol": 1e-9,
        },
        # --- Edge: local harmonic propagation, where the true energy runs away as the width grows ---
        {
            "setup": "import numpy as np\n"
                     "par = np.array([0.85, 0.02, 0.45, 0.3, 0.8])\n"
                     "gp = np.array([1.0, 0.5, np.deg2rad(20.0)])\n",
            "call": "thawed_gaussian_energy(2.6, par.copy(), gp.copy(), 0, 0.05, 560)",
            "gold_call": "_oracle_thawed_gaussian_energy(2.6, par.copy(), gp.copy(), 0, 0.05, 560)",
            "tol": 1e-8,
        },
        # --- Normal: vertical reference Hessian with a negative coupling and a large Duschinsky angle ---
        {
            "setup": "import numpy as np\n"
                     "par = np.array([1.1, 0.04, 0.6, -0.4, -0.5])\n"
                     "gp = np.array([0.6, 1.3, 1.1])\n",
            "call": "thawed_gaussian_energy(1.4, par.copy(), gp.copy(), 2, 0.04, 300)",
            "gold_call": "_oracle_thawed_gaussian_energy(1.4, par.copy(), gp.copy(), 2, 0.04, 300)",
            "tol": 1e-9,
        },
        # --- Normal: frozen initial-state Hessian with a stiff, strongly coupled bend ---
        {
            "setup": "import numpy as np\n"
                     "par = np.array([1.6, 0.03, 1.2, 0.25, 0.5])\n"
                     "gp = np.array([1.4, 1.5, -0.3])\n",
            "call": "thawed_gaussian_energy(1.7, par.copy(), gp.copy(), 3, 0.025, 600)",
            "gold_call": "_oracle_thawed_gaussian_energy(1.7, par.copy(), gp.copy(), 3, 0.025, 600)",
            "tol": 1e-9,
        },
        # --- Boundary: zero steps returns the energy of the initial ground state on the excited surface ---
        {
            "setup": "import numpy as np\n"
                     "par = np.array([0.6, 0.01, 0.35, 0.9, 1.9])\n"
                     "gp = np.array([0.6, 0.35, 0.4])\n",
            "call": "thawed_gaussian_energy(-0.7, par.copy(), gp.copy(), 1, 0.1, 0)",
            "gold_call": "_oracle_thawed_gaussian_energy(-0.7, par.copy(), gp.copy(), 1, 0.1, 0)",
            "tol": 1e-12,
        },
        # --- Error: a non-positive bend frequency must raise ValueError ---
        {
            "setup": "import numpy as np\n"
                     "def _probe(fn):\n"
                     "    try:\n"
                     "        fn(1.0, np.array([0.85, 0.02, 0.0, 0.3, 0.8]), np.array([1.0, 0.5, 0.2]), 1, 0.1, 3)\n"
                     "    except ValueError:\n"
                     "        return 1\n"
                     "    except Exception:\n"
                     "        return 2\n"
                     "    return 0\n",
            "call": "_probe(thawed_gaussian_energy)",
            "gold_call": "_probe(_oracle_thawed_gaussian_energy)",
        },
    ]
