"""
Implement compute_macroscopic_flux to evaluate the deterministic macroscopic
numerical flux H^{ma} at every periodic cell interface for the one-dimensional
single-group neutron transport equation with isotropic scattering.

Return the deterministic macroscopic numerical flux at every interface of the uniform periodic mesh for the source paper's one-dimensional, isotropic single-group model. The neutron speed is fixed at v = 1. Interface j separates cells (j-1) % N_x and j, consistently with step 01. Use the supplied scalar flux, scattering and total cross-sections, mesh width, time step and collision time.

Returns
-------
np.ndarray, macroscopic numerical flux H_ma at each periodic interface, shape (N_x,)
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def compute_macroscopic_flux(
    psi: np.ndarray,
    sigma_s: float,
    sigma: float,
    dx: float,
    dt: float,
    tau: float,
) -> np.ndarray:
    """
    Parameters
    ----------
    psi : numpy.ndarray
        Cell-averaged macroscopic scalar flux, shape (N_x,).
    sigma_s : float
        Macroscopic scattering cross-section.
    sigma : float
        Total macroscopic cross-section.
    dx : float
        Uniform cell width.
    dt : float
        Time step.
    tau : float
        Characteristic collision time, 1 / (v * Sigma).

    Returns
    -------
    H_ma : numpy.ndarray
        Macroscopic numerical flux at each periodic interface, shape (N_x,).

    Raises
    ------
    ValueError
        If dx, dt or tau is not positive.
    """
    return H_ma

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _compute_coefficients(dt: float, tau: float) -> tuple[float, float]:
    """Compute integral solution coefficients from the kinetic equation.

    Parameters
    ----------
    dt : float
        Time step size (dt > 0).
    tau : float
        Characteristic collision time (tau > 0).

    Returns
    -------
    coefficients : tuple[float, float]
        (c3, c5).
    """
    a0 = np.exp(-dt / tau)
    a1 = 1.0 - a0

    c3 = 2.0 * tau**2 * a1 / dt - tau * a0 - tau
    c5 = tau * a0 - tau**2 * a1 / dt

    return (c3, c5)



def _oracle_compute_macroscopic_flux(
    psi: np.ndarray,
    sigma_s: float,
    sigma: float,
    dx: float,
    dt: float,
    tau: float,
) -> np.ndarray:
    """Compute macroscopic numerical flux at periodic interfaces via Eq (3.37)."""
    if dx <= 0 or dt <= 0 or tau <= 0:
        raise ValueError("dx, dt and tau must be positive")
    N_x = len(psi)
    sigma_bar = sigma_s / (2.0 * sigma)
    S = sigma_bar * psi

    S_L, S_R = _oracle_reconstruct_interfaces(S, dx)

    c3, _ = _compute_coefficients(dt, tau)
    v = 1.0
    coeff = 2.0 * v * v * c3 / (3.0 * dx)

    H_ma = np.zeros(N_x)
    for j in range(N_x):
        left_cell = (j - 1) % N_x
        H_ma[j] = coeff * (S_L[j] - S[left_cell] + S[j] - S_R[j])

    return H_ma

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        # Normal: linear psi profile, sigma_t=11
        {
            "setup": """import numpy as np
dx = 0.01
N_x = 100
psi = np.linspace(0.5, 1.5, N_x)
sigma = 11.0
sigma_s = 10.0
tau = 1.0 / sigma
dt = 0.2 * dx / 1.0
""",
            "call": "np.round(compute_macroscopic_flux(psi.copy(), sigma_s, sigma, dx, dt, tau), 10)",
            "gold_call": "np.round(_oracle_compute_macroscopic_flux(psi.copy(), sigma_s, sigma, dx, dt, tau), 10)",
        },
        # Boundary: uniform psi, all fluxes should be zero
        {
            "setup": """import numpy as np
dx = 0.1
psi = np.full(10, 2.0)
sigma = 11.0
sigma_s = 10.0
tau = 1.0 / sigma
dt = 0.2 * dx
""",
            "call": "np.round(compute_macroscopic_flux(psi.copy(), sigma_s, sigma, dx, dt, tau), 12)",
            "gold_call": "np.round(_oracle_compute_macroscopic_flux(psi.copy(), sigma_s, sigma, dx, dt, tau), 12)",
        },
        # Edge: high Sigma (diffusion limit)
        {
            "setup": """import numpy as np
dx = 0.025
N_x = 40
psi = np.sin(np.pi * np.linspace(dx/2, 1-dx/2, N_x))
sigma = 1000.0
sigma_s = 999.0
tau = 1.0 / sigma
dt = 0.2 * dx / 1.0
""",
            "call": "np.round(compute_macroscopic_flux(psi.copy(), sigma_s, sigma, dx, dt, tau), 10)",
            "gold_call": "np.round(_oracle_compute_macroscopic_flux(psi.copy(), sigma_s, sigma, dx, dt, tau), 10)",
        },
    ]
