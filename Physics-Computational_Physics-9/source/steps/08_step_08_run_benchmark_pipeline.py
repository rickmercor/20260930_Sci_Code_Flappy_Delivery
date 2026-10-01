"""
Chain the sub-problem functions 01-07 end to end on the mixed-boundary benchmark and return the RMS relative-error norm.

This step chains the whole pipeline end to end for the mixed-boundary benchmark. The orchestrator (i) builds the cell-centered coordinate cube and the single-mode, vertically Gaussian source density with sub-problem 01, (ii) builds the shear-corrected in-plane wave-vectors at time t with sub-problem 02, (iii) evaluates the analytic Fourier-space Green's function with truncation half-width alpha*L on the padded vertical wavenumber grid with sub-problem 03, (iv) zero-pads the density vertically by pad_factor with sub-problem 04, (v) solves for the potential with the one-step 3D spectral convolution of sub-problem 05, (vi) evaluates the closed-form reference potential with sub-problem 06, and (vii) returns the RMS relative-error norm of sub-problem 07 after shifting each field so its minimum is 1. The result is a single deterministic scalar quantifying the solver accuracy.

Returns
-------
float: the RMS relative-error norm between the spectral and the analytic potential after min-to-1 shifting, as a native Python float.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def run_benchmark_pipeline(N: int, L: float, H: float, alpha: float = 1.1,
                           pad_factor: int = 4, G: float = 1.0,
                           Omega: float = 1.0, t: float = 0.0,
                           m_x: int = 1, m_y: int = 1) -> float:
    """Run the full spectral Poisson benchmark and return the error norm.

    Parameters
    ----------
    N : int
        Number of cells per dimension of the cubic box (N >= 4).
    L : float
        Box size in every dimension (L > 0).
    H : float
        Gaussian scale height of the vertical density profile (H > 0).
    alpha : float
        Green's function truncation factor, half-width alpha * L (alpha > 1).
    pad_factor : int
        Vertical zero-padding factor (integer, pad_factor >= 2 * alpha).
    G : float
        Gravitational constant (G > 0).
    Omega : float
        Orbital frequency of the shearing box (finite real number).
    t : float
        Time at which the wave-vectors are evaluated (finite real number).
    m_x : int
        Integer mode number of the density in x, kx = 2*pi*m_x/L.
    m_y : int
        Integer mode number of the density in y, ky = 2*pi*m_y/L (m_x and
        m_y must not both be zero).

    Returns
    -------
    error_norm : float
        The RMS relative-error norm between the spectral and the analytic
        potential after min-to-1 shifting, as a native Python float.

    Notes
    -----
    This is the final, orchestrating step: call the public functions of
    sub-problems 01-07 (``build_density_grid``, ``compute_shear_wavevectors``,
    ``greens_function_hat``, ``zero_pad_density``,
    ``spectral_convolution_potential``, ``analytic_layer_potential``,
    ``relative_error_rms``) and feed each returned value into the next, rather
    than reimplementing them. Include every import your implementation needs
    (for example ``import numpy as np``) inside the function body.
    """
    return 0.0  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_run_benchmark_pipeline(N: int, L: float, H: float, alpha: float = 1.1,
                                   pad_factor: int = 4, G: float = 1.0,
                                   Omega: float = 1.0, t: float = 0.0,
                                   m_x: int = 1, m_y: int = 1) -> float:
    import numpy as np

    # -- Validate the orchestrator inputs.
    if not (isinstance(N, (int, np.integer)) and not isinstance(N, bool) and N >= 4):
        raise ValueError("N must be an integer >= 4")
    if not (isinstance(L, (int, float)) and np.isfinite(L) and float(L) > 0.0):
        raise ValueError("L must be a finite number > 0")
    if not (isinstance(H, (int, float)) and np.isfinite(H) and float(H) > 0.0):
        raise ValueError("H must be a finite number > 0")
    if not (isinstance(alpha, (int, float)) and np.isfinite(alpha) and float(alpha) > 1.0):
        raise ValueError("alpha must be a finite number > 1")
    if not (isinstance(pad_factor, (int, np.integer)) and not isinstance(pad_factor, bool)
            and pad_factor >= 2.0 * float(alpha)):
        raise ValueError("pad_factor must be an integer >= 2 * alpha")
    if not (isinstance(G, (int, float)) and np.isfinite(G) and float(G) > 0.0):
        raise ValueError("G must be a finite number > 0")
    if not (isinstance(Omega, (int, float)) and np.isfinite(Omega)):
        raise ValueError("Omega must be a finite number")
    if not (isinstance(t, (int, float)) and np.isfinite(t)):
        raise ValueError("t must be a finite number")
    for name, val in (("m_x", m_x), ("m_y", m_y)):
        if not (isinstance(val, (int, np.integer)) and not isinstance(val, bool)):
            raise ValueError(f"{name} must be an integer")
    if int(m_x) == 0 and int(m_y) == 0:
        raise ValueError("m_x and m_y must not both be zero")

    N = int(N)
    L = float(L)
    H = float(H)
    alpha = float(alpha)
    pad_factor = int(pad_factor)
    G = float(G)
    d = L / N

    # -- Sub-problem 01: cell-centered grid and truncated Gaussian density.
    grid = _oracle_build_density_grid(N, L, H, m_x=int(m_x), m_y=int(m_y))
    x_3d, y_3d, z_3d, rho = grid[0], grid[1], grid[2], grid[3]
    kx_mode = 2.0 * np.pi * int(m_x) / L
    ky_mode = 2.0 * np.pi * int(m_y) / L

    # -- Sub-problem 02: shear-corrected in-plane wave-vectors.
    k_grids = _oracle_compute_shear_wavevectors(N, N, L, L, float(Omega), float(t))
    k_perp = np.sqrt(k_grids[0] ** 2 + k_grids[1] ** 2)[:, :, None]

    # -- Sub-problem 03: analytic Fourier-space kernel on the padded kz grid.
    kz = (2.0 * np.pi * np.fft.fftfreq(pad_factor * N, d=d))[None, None, :]
    greens_hat_grid = _oracle_greens_function_hat(k_perp, kz, alpha * L)

    # -- Sub-problem 04: vertical zero-padding of the density.
    rho_padded = _oracle_zero_pad_density(rho, pad_factor)

    # -- Sub-problem 05: one-step 3D spectral convolution on physical cells.
    phi_num = _oracle_spectral_convolution_potential(rho_padded, greens_hat_grid, N, G=G)

    # -- Sub-problem 06: closed-form reference potential.
    phi_ref = _oracle_analytic_layer_potential(x_3d, y_3d, z_3d, kx_mode, ky_mode, H, L, G=G)

    # -- Sub-problem 07: min-to-1 shifted RMS relative-error norm.
    return _oracle_relative_error_rms(phi_num, phi_ref)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Integration: thin layer, coarse grid (normal scenario) ---
        {
            "setup": """import numpy as np
N = 16
L = 6.0
H = 1.0
""",
            "call": "run_benchmark_pipeline(N, L, H)",
            "gold_call": "_oracle_run_benchmark_pipeline(N, L, H)",
        },
        # --- Integration: thick layer H = L / 2, stronger boundary truncation ---
        {
            "setup": """import numpy as np
N = 16
L = 6.0
H = 3.0
""",
            "call": "run_benchmark_pipeline(N, L, H)",
            "gold_call": "_oracle_run_benchmark_pipeline(N, L, H)",
        },
        # --- Integration: final-answer configuration N = 32, H = 1 ---
        {
            "setup": """import numpy as np
N = 32
L = 6.0
H = 1.0
""",
            "call": "run_benchmark_pipeline(N, L, H, alpha=1.1, pad_factor=4, G=1.0, Omega=1.0, t=0.0, m_x=1, m_y=1)",
            "gold_call": "_oracle_run_benchmark_pipeline(N, L, H, alpha=1.1, pad_factor=4, G=1.0, Omega=1.0, t=0.0, m_x=1, m_y=1)",
        },
        # --- Integration (boundary): t equal to one full shear period, offset wraps to 0 ---
        {
            "setup": """import numpy as np
N = 16
L = 6.0
H = 1.0
Omega = 1.0
t_wrap = L / (1.5 * Omega * L)
""",
            "call": "run_benchmark_pipeline(N, L, H, Omega=Omega, t=t_wrap)",
            "gold_call": "_oracle_run_benchmark_pipeline(N, L, H, Omega=Omega, t=t_wrap)",
        },
        # --- Integration (edge): higher in-plane mode numbers ---
        {
            "setup": """import numpy as np
N = 16
L = 6.0
H = 1.0
""",
            "call": "run_benchmark_pipeline(N, L, H, m_x=2, m_y=1)",
            "gold_call": "_oracle_run_benchmark_pipeline(N, L, H, m_x=2, m_y=1)",
        },
    ]
