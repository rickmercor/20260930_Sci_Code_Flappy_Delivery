"""
Implement macroscopic_update to advance the macroscopic scalar flux by one
UGKWP time step, incorporating macroscopic, microscopic, and collisional flux
contributions along with the spatially varying external source.

Advance the scalar flux by one step on the uniform periodic mesh using the supplied deterministic and free-particle currents, the prior signed macroscopic remainder, cross-sections, and cell-constant external angular source. The neutron speed is fixed at v = 1. Interface j separates cells (j-1) % N_x and j; a current is positive in the increasing-x direction. Return the updated scalar flux without a positivity clamp.



Apply these two corrections to printing inconsistencies in the source paper: the external-source increment in Eq. 3.50 is Δt Q̂/τ, where Q̂ = 2q/sigma, as in Eqs. 3.9 and 4.7; and the source current in Eq. 3.25 retains the factor 1 − exp(−u/τ) from Eq. 3.17 inside its time integral. For the supplied cell-constant angular source this corrected source-interface current is H_q[j] = c1 (v/2) (q[j-1] − q[j])/sigma, with c1 = 1 − τ(1 − exp(−Δt/τ))/Δt. These corrections apply only to the external-source terms.

Returns
-------
np.ndarray, updated cell-averaged scalar flux psi^{n+1}, shape (N_x,)
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def macroscopic_update(
    psi: np.ndarray,
    H_ma: np.ndarray,
    H_mi_free: np.ndarray,
    psi_ma: np.ndarray,
    sigma: float,
    sigma_s: float,
    q_arr: np.ndarray,
    dx: float,
    dt: float,
    tau: float,
) -> np.ndarray:
    """
    Parameters
    ----------
    psi : numpy.ndarray
        Current cell-averaged scalar flux psi^n, shape (N_x,).
    H_ma : numpy.ndarray
        Macroscopic flux from Step 2, shape (N_x,).
    H_mi_free : numpy.ndarray
        Free-transport microscopic flux from Step 4, shape (N_x,).
    psi_ma : numpy.ndarray
        Macroscopic remainder psi - psi_mi formed at the end of the previous
        step (the full remainder, before the collisionless fraction
        exp(-dt / tau) is taken for resampling), shape (N_x,).
    sigma : float
        Total macroscopic cross-section.
    sigma_s : float
        Scattering cross-section.
    q_arr : numpy.ndarray
        Spatially varying external source, shape (N_x,).
    dx : float
        Cell width.
    dt : float
        Time step.
    tau : float
        Characteristic collision time.

    Returns
    -------
    psi_new : numpy.ndarray
        Updated scalar flux psi^{n+1}, shape (N_x,).

    Raises
    ------
    ValueError
        If dx, dt or tau is not positive.
    """
    return psi_new

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_macroscopic_update(
    psi: np.ndarray,
    H_ma: np.ndarray,
    H_mi_free: np.ndarray,
    psi_ma: np.ndarray,
    sigma: float,
    sigma_s: float,
    q_arr: np.ndarray,
    dx: float,
    dt: float,
    tau: float,
) -> np.ndarray:
    """Update macroscopic scalar flux with periodic collisional flux."""
    if dx <= 0 or dt <= 0 or tau <= 0:
        raise ValueError("dx, dt and tau must be positive")
    N_x = len(psi)
    v = 1.0

    _, c5 = _compute_coefficients(dt, tau)
    a0 = np.exp(-dt / tau)
    a1 = 1.0 - a0
    c4 = tau * a1 / dt

    psi_ma_L, psi_ma_R = _oracle_reconstruct_interfaces(psi_ma, dx)

    c5_term = c5 + 0.5 * dt * a0
    col_coeff = c5_term * v * v / (3.0 * dx)
    mom_coeff = (c4 - a0) * v / 4.0
    c1 = 1.0 - tau * a1 / dt

    H_ma_col = np.zeros(N_x)
    H_q = np.zeros(N_x)
    for j in range(N_x):
        left_cell = (j - 1) % N_x
        H_ma_col[j] = (
            col_coeff * (psi_ma_L[j] - psi_ma[left_cell] + psi_ma[j] - psi_ma_R[j])
            + mom_coeff * (psi_ma_L[j] - psi_ma_R[j])
        )
        H_q[j] = c1 * 0.5 * v * (q_arr[left_cell] - q_arr[j]) / sigma

    sigma_a = sigma - sigma_s
    alpha = 1.0 / (1.0 + v * dt * sigma_a)

    source = 2.0 * v * dt * q_arr

    psi_new = np.zeros(N_x)
    for i in range(N_x):
        j_right = (i + 1) % N_x
        j_left = i
        flux_div = (
            (H_ma[j_right] - H_ma[j_left])
            + (H_mi_free[j_right] - H_mi_free[j_left])
            + (H_ma_col[j_right] - H_ma_col[j_left])
            + (H_q[j_right] - H_q[j_left])
        )
        psi_new[i] = alpha * (psi[i] + source[i] - (dt / dx) * flux_div)

    return psi_new

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        # Nonzero microscopic interface currents redistribute scalar flux
        {
            "setup": """import numpy as np
dx = 0.25
sigma = 11.0
sigma_s = 10.0
tau = 1.0 / sigma
dt = 0.01
psi = np.ones(4)
psi_ma = np.zeros(4)
q_arr = np.zeros(4)
H_ma = np.zeros(4)
H_mi_free = np.array([0.0, 0.1, -0.05, 0.0])
""",
            "call": "np.round(macroscopic_update(psi.copy(), H_ma.copy(), H_mi_free.copy(), psi_ma.copy(), sigma, sigma_s, q_arr.copy(), dx, dt, tau), 10)",
            "gold_call": "np.round(_oracle_macroscopic_update(psi.copy(), H_ma.copy(), H_mi_free.copy(), psi_ma.copy(), sigma, sigma_s, q_arr.copy(), dx, dt, tau), 10)",
        },
        # Normal: absorbing medium (sigma_a=1), uniform psi, spatially varying source
        {
            "setup": """import numpy as np
N_x = 10
dx = 0.1
sigma = 11.0
sigma_s = 10.0
tau = 1.0 / sigma
dt = 0.2 * dx
psi = np.full(N_x, 1.0)
psi_ma = np.full(N_x, 0.5)
q_arr = np.zeros(N_x)
q_arr[3:7] = 0.5
H_ma = np.zeros(N_x)
H_mi_free = np.zeros(N_x)
""",
            "call": "np.round(macroscopic_update(psi.copy(), H_ma.copy(), H_mi_free.copy(), psi_ma.copy(), sigma, sigma_s, q_arr.copy(), dx, dt, tau), 10)",
            "gold_call": "np.round(_oracle_macroscopic_update(psi.copy(), H_ma.copy(), H_mi_free.copy(), psi_ma.copy(), sigma, sigma_s, q_arr.copy(), dx, dt, tau), 10)",
        },
        # Boundary: no source, no absorption (sigma_a=0, alpha=1)
        {
            "setup": """import numpy as np
N_x = 5
dx = 0.2
sigma = 10.0
sigma_s = 10.0
tau = 1.0 / sigma
dt = 0.01
psi = np.array([1.0, 2.0, 3.0, 2.0, 1.0])
psi_ma = np.array([0.5, 1.0, 1.5, 1.0, 0.5])
q_arr = np.zeros(N_x)
H_ma = np.zeros(N_x)
H_mi_free = np.zeros(N_x)
""",
            "call": "np.round(macroscopic_update(psi.copy(), H_ma.copy(), H_mi_free.copy(), psi_ma.copy(), sigma, sigma_s, q_arr.copy(), dx, dt, tau), 10)",
            "gold_call": "np.round(_oracle_macroscopic_update(psi.copy(), H_ma.copy(), H_mi_free.copy(), psi_ma.copy(), sigma, sigma_s, q_arr.copy(), dx, dt, tau), 10)",
        },
        # Edge: diffusion limit (large Sigma), gradient-driven transport
        {
            "setup": """import numpy as np
N_x = 20
dx = 0.05
sigma = 1000.0
sigma_s = 999.0
tau = 1.0 / sigma
dt = 0.2 * dx / 1.0
x_c = np.linspace(dx/2, 1-dx/2, N_x)
psi = np.sin(np.pi * x_c) + 1.0
psi_ma = 0.5 * psi
q_arr = np.full(N_x, 0.5)
H_ma = -(np.pi / (3.0 * sigma)) * np.cos(np.pi * np.arange(N_x) * dx)
H_mi_free = np.zeros(N_x)
""",
            "call": "np.round(macroscopic_update(psi.copy(), H_ma.copy(), H_mi_free.copy(), psi_ma.copy(), sigma, sigma_s, q_arr.copy(), dx, dt, tau), 8)",
            "gold_call": "np.round(_oracle_macroscopic_update(psi.copy(), H_ma.copy(), H_mi_free.copy(), psi_ma.copy(), sigma, sigma_s, q_arr.copy(), dx, dt, tau), 8)",
        },
    ]
