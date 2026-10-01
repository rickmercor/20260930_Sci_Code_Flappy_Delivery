"""
Implement solve_ugkwp_steady to compute the steady-state macroscopic scalar
flux for the one-dimensional neutron transport equation with isotropic
scattering and periodic boundary conditions using the UGKWP method.

The spatial domain is [0, 1) with periodic boundaries, neutron speed v = 1, and uniform cell width dx = 1 / N_x. Start from zero scalar flux and no particles. The target particle mass is m_e = dx / n_ppc, where n_ppc is the target number of particles per cell at scalar flux one. The macroscopic remainder is signed and must not be clamped; cells with nonpositive assigned mass generate no resampled particles. The running average contains the updated scalar flux from iterations k = avg_start, ..., max_iter - 1, giving exactly max_iter - avg_start samples.

Returns
-------
np.ndarray, time-averaged cell-centre scalar flux over iterations k >= avg_start, shape (N_x,)
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def solve_ugkwp_steady(
    sigma_s: float,
    sigma_a: float,
    q_arr: np.ndarray,
    N_x: int,
    CFL: float,
    n_ppc: int,
    max_iter: int,
    avg_start: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Parameters
    ----------
    sigma_s : float
        Scattering cross-section.
    sigma_a : float
        Absorption cross-section.
    q_arr : numpy.ndarray
        Spatially varying external source, shape (N_x,).
    N_x : int
        Number of spatial cells.
    CFL : float
        CFL number in (0, 1).
    n_ppc : int
        Target number of particles per cell.
    max_iter : int
        Maximum number of time steps.
    avg_start : int
        Iteration to begin Welford averaging.
    rng : numpy.random.Generator
        Random number generator.

    Returns
    -------
    psi_avg : numpy.ndarray
        Time-averaged macroscopic scalar flux at cell centers, shape (N_x,).

    Raises
    ------
    ValueError
        If N_x < 1, CFL is not positive, n_ppc < 1, or avg_start is not in [0, max_iter).

    Notes
    -----
    Preserve particle ordering for reproducibility. Existing surviving particles retain their relative order, followed by newly resampled particles in generation order. Whenever classification is performed, draw one uniform survival probability per classified particle using rng.random in particle order. Classification draws precede resampling draws in each iteration. Resampling draws are taken cell by cell in ascending order, with one position draw followed by one velocity draw per generated particle, as specified in step 06.
    """
    return psi_avg

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _bin_particles(x_p: np.ndarray, w_p: np.ndarray, n_particles: int, N_x: int, dx: float) -> np.ndarray:
    """Bin particle masses into cells."""
    psi_mi = np.zeros(N_x)
    for p in range(n_particles):
        cell = int(x_p[p] / dx) % N_x
        psi_mi[cell] += w_p[p] / dx
    return psi_mi



def _oracle_solve_ugkwp_steady(
    sigma_s: float, sigma_a: float, q_arr: np.ndarray,
    N_x: int, CFL: float, n_ppc: int,
    max_iter: int, avg_start: int, rng: np.random.Generator,
) -> np.ndarray:
    """Run UGKWP with periodic BC to steady state and return averaged scalar flux."""
    if N_x < 1 or CFL <= 0 or n_ppc < 1 or not 0 <= avg_start < max_iter:
        raise ValueError("invalid solver configuration")
    v = 1.0
    dx = 1.0 / N_x
    sigma_t = sigma_s + sigma_a
    tau = 1.0 / (v * sigma_t)
    dt = CFL * min(dx / v, 1.5 * dx**2 * sigma_t / v)
    m_e = dx / n_ppc

    psi = np.zeros(N_x)
    psi_ma = np.zeros(N_x)

    x_survive = np.empty(0)
    xi_survive = np.empty(0)
    w_survive = np.empty(0)
    x_resamp = np.empty(0)
    xi_resamp = np.empty(0)
    w_resamp = np.empty(0)

    psi_avg = np.zeros(N_x)
    n_avg = 0

    for iteration in range(max_iter):
        H_ma = _oracle_compute_macroscopic_flux(psi, sigma_s, sigma_t, dx, dt, tau)

        n_s = len(x_survive)
        if n_s > 0:
            t_f_s, is_cl_s = _oracle_classify_particles(tau, dt, rng, n_s)
        else:
            t_f_s = np.empty(0)
            is_cl_s = np.empty(0, dtype=bool)

        n_r = len(x_resamp)
        t_f_r = np.full(n_r, dt)

        x_p = np.concatenate((x_survive, x_resamp))
        xi_p = np.concatenate((xi_survive, xi_resamp))
        w_p = np.concatenate((w_survive, w_resamp))
        t_f = np.concatenate((t_f_s, t_f_r))
        is_collisionless = np.concatenate((is_cl_s, np.ones(n_r, dtype=bool)))

        if len(x_p) > 0:
            x_new, H_mi_free = _oracle_free_transport_step(
                x_p, xi_p, w_p, dx, N_x, dt, t_f
            )
            surviving = is_collisionless
            x_survive = x_new[surviving]
            xi_survive = xi_p[surviving]
            w_survive = w_p[surviving]
        else:
            H_mi_free = np.zeros(N_x)
            x_survive = np.empty(0)
            xi_survive = np.empty(0)
            w_survive = np.empty(0)

        psi_new = _oracle_macroscopic_update(
            psi, H_ma, H_mi_free, psi_ma,
            sigma_t, sigma_s, q_arr, dx, dt, tau,
        )

        psi_mi = _bin_particles(x_survive, w_survive, len(x_survive), N_x, dx)

        psi_ma = psi_new - psi_mi

        x_resamp, xi_resamp, w_resamp = _oracle_resample_particles(
            psi_ma, dt, tau, dx, m_e, rng,
        )

        psi = psi_new

        if iteration >= avg_start:
            n_avg += 1
            for i in range(N_x):
                psi_avg[i] += (psi[i] - psi_avg[i]) / n_avg

    return psi_avg

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    _setup = """import numpy as np
N_x = 40
dx = 1.0 / N_x
xc = np.array([(i + 0.5) * dx for i in range(N_x)])
q_arr = np.zeros(N_x)
for i in range(N_x):
    if 0.25 <= xc[i] < 0.75:
        q_arr[i] = 0.5
"""
    return [
        # Normal: sigma_s=10, sigma_a=1, time-averaged profile under the documented draw and ordering conventions
        {
            "setup": _setup + """
def run_model():
    rng = np.random.default_rng(42)
    psi = solve_ugkwp_steady(10.0, 1.0, q_arr.copy(), N_x, 0.2, 50, 4000, 2000, rng)
    return np.round(psi, 6)

def run_gold():
    rng = np.random.default_rng(42)
    psi = _oracle_solve_ugkwp_steady(10.0, 1.0, q_arr.copy(), N_x, 0.2, 50, 4000, 2000, rng)
    return np.round(psi, 6)
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # Boundary: diffusive regime (higher total cross-section) still symmetric about x=0.5
        {
            "setup": """import numpy as np
N_x = 40
dx = 1.0 / N_x
xc = np.array([(i + 0.5) * dx for i in range(N_x)])
q_arr = np.zeros(N_x)
for i in range(N_x):
    if 0.25 <= xc[i] < 0.75:
        q_arr[i] = 0.5

def run_model():
    rng = np.random.default_rng(42)
    psi = solve_ugkwp_steady(90.0, 10.0, q_arr.copy(), N_x, 0.2, 50, 4000, 2000, rng)
    sym_err = float(np.max(np.abs(psi - psi[::-1])) / (np.max(psi) + 1e-15))
    return np.concatenate(([float(sym_err < 0.1)], np.round(psi, 6)))

def run_gold():
    rng = np.random.default_rng(42)
    psi = _oracle_solve_ugkwp_steady(90.0, 10.0, q_arr.copy(), N_x, 0.2, 50, 4000, 2000, rng)
    sym_err = float(np.max(np.abs(psi - psi[::-1])) / (np.max(psi) + 1e-15))
    return np.concatenate(([float(sym_err < 0.1)], np.round(psi, 6)))
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # Edge: off-center narrow source - flux at the source should exceed flux far from it
        {
            "setup": """import numpy as np
N_x = 40
dx = 1.0 / N_x
xc = np.array([(i + 0.5) * dx for i in range(N_x)])
q_arr = np.zeros(N_x)
for i in range(N_x):
    if 0.05 <= xc[i] < 0.15:
        q_arr[i] = 0.5
src_idx = int(np.where(q_arr > 0)[0][0])
far_idx = (src_idx + N_x // 2) % N_x

def run_model():
    rng = np.random.default_rng(42)
    psi = solve_ugkwp_steady(10.0, 1.0, q_arr.copy(), N_x, 0.2, 50, 4000, 2000, rng)
    return np.concatenate(([float(psi[src_idx] > psi[far_idx])], np.round(psi, 6)))

def run_gold():
    rng = np.random.default_rng(42)
    psi = _oracle_solve_ugkwp_steady(10.0, 1.0, q_arr.copy(), N_x, 0.2, 50, 4000, 2000, rng)
    return np.concatenate(([float(psi[src_idx] > psi[far_idx])], np.round(psi, 6)))
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
    ]
