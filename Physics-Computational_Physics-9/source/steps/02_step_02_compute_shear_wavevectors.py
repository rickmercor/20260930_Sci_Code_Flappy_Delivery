"""
Build the shear-corrected in-plane wave-vector grids of the fully periodic frame at time t.

In the shearing box approximation, the radial (x) boundaries of the local patch drift with the background Keplerian shear, so the box is strictly periodic in x only at discrete instants. Before a Fourier-space Poisson solve, the fields are remapped to the nearest fully periodic frame, and the coordinate transformation modifies the in-plane wave-vector: the azimuthal wavenumber leaks into the radial one. With shear rate q = 3/2 (Keplerian), the accumulated azimuthal boundary offset is Delta_y0(t) = mod(1.5 * Omega * Lx * t, Ly), and the wave-vector in the periodic frame becomes

    kx(t) = Kx + (Delta_y0(t) / Lx) * Ky,    ky = Ky,

where (Kx, Ky) are the standard periodic wavenumbers of the initial frame, Kx = 2*pi*n/Lx and Ky = 2*pi*m/Ly in FFT ordering. This step builds the in-plane wave-vector grids used by every subsequent Fourier-space operation.

Returns
-------
np.ndarray of shape (2, Nx, Ny), float: [0] is the shear-corrected kx(t) grid and [1] is the ky grid, both in FFT frequency ordering.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def compute_shear_wavevectors(Nx: int, Ny: int, Lx: float, Ly: float,
                              Omega: float, t: float) -> np.ndarray:
    """Build the shear-corrected in-plane wave-vector grids at time t.

    Parameters
    ----------
    Nx : int
        Number of grid cells in the x direction (Nx >= 1).
    Ny : int
        Number of grid cells in the y direction (Ny >= 1).
    Lx : float
        Box size in the x direction (Lx > 0).
    Ly : float
        Box size in the y direction (Ly > 0).
    Omega : float
        Orbital frequency of the shearing box (finite real number).
    t : float
        Time at which the wave-vectors are evaluated (finite real number).

    Returns
    -------
    k_grids : np.ndarray
        Array of shape (2, Nx, Ny). k_grids[0][i, j] is the shear-corrected
        radial wavenumber kx(t) for mode (i, j) and k_grids[1][i, j] is the
        azimuthal wavenumber ky, both in FFT frequency ordering.

    Notes
    -----
    The evaluation sandbox may execute this function in isolation: include
    every import your implementation needs (for example
    ``import numpy as np``) inside the function body.
    """
    return np.zeros((2, Nx, Ny), dtype=float)  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_compute_shear_wavevectors(Nx: int, Ny: int, Lx: float, Ly: float,
                                      Omega: float, t: float) -> np.ndarray:
    # Local imports keep the oracle self-contained when the harness
    # executes it in isolation.
    import numpy as np

    if not (isinstance(Nx, (int, np.integer)) and not isinstance(Nx, bool) and Nx >= 1):
        raise ValueError("Nx must be an integer >= 1")
    if not (isinstance(Ny, (int, np.integer)) and not isinstance(Ny, bool) and Ny >= 1):
        raise ValueError("Ny must be an integer >= 1")
    if not (isinstance(Lx, (int, float)) and np.isfinite(Lx) and float(Lx) > 0.0):
        raise ValueError("Lx must be a finite number > 0")
    if not (isinstance(Ly, (int, float)) and np.isfinite(Ly) and float(Ly) > 0.0):
        raise ValueError("Ly must be a finite number > 0")
    if not (isinstance(Omega, (int, float)) and np.isfinite(Omega)):
        raise ValueError("Omega must be a finite number")
    if not (isinstance(t, (int, float)) and np.isfinite(t)):
        raise ValueError("t must be a finite number")

    Lx = float(Lx)
    Ly = float(Ly)
    kx_base = 2.0 * np.pi * np.fft.fftfreq(Nx, d=Lx / Nx)
    ky_base = 2.0 * np.pi * np.fft.fftfreq(Ny, d=Ly / Ny)

    delta_y0 = np.mod(1.5 * float(Omega) * Lx * float(t), Ly)

    kx_2d = kx_base[:, None] + (delta_y0 / Lx) * ky_base[None, :]
    ky_2d = np.broadcast_to(ky_base[None, :], (Nx, Ny)).copy()

    return np.stack([kx_2d, ky_2d]).astype(float)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Valid: no shear offset at t = 0 (normal scenario) ---
        {
            "setup": """import numpy as np
Nx, Ny = 8, 6
Lx, Ly = 6.0, 4.0
Omega = 1.0
t = 0.0
""",
            "call": "compute_shear_wavevectors(Nx, Ny, Lx, Ly, Omega, t)",
            "gold_call": "_oracle_compute_shear_wavevectors(Nx, Ny, Lx, Ly, Omega, t)",
        },
        # --- Valid: generic time with nonzero azimuthal offset ---
        {
            "setup": """import numpy as np
Nx, Ny = 16, 16
Lx, Ly = 6.0, 6.0
Omega = 1.0
t = 0.7
""",
            "call": "compute_shear_wavevectors(Nx, Ny, Lx, Ly, Omega, t)",
            "gold_call": "_oracle_compute_shear_wavevectors(Nx, Ny, Lx, Ly, Omega, t)",
        },
        # --- Boundary: t equal to one full shear period, offset wraps to zero ---
        {
            "setup": """import numpy as np
Nx, Ny = 8, 8
Lx, Ly = 6.0, 6.0
Omega = 1.0
t = Ly / (1.5 * Omega * Lx)
""",
            "call": "compute_shear_wavevectors(Nx, Ny, Lx, Ly, Omega, t)",
            "gold_call": "_oracle_compute_shear_wavevectors(Nx, Ny, Lx, Ly, Omega, t)",
        },
        # --- Edge: single-cell dimensions (Nx = Ny = 1, only the zero mode) ---
        {
            "setup": """import numpy as np
Nx, Ny = 1, 1
Lx, Ly = 2.0, 3.0
Omega = 0.5
t = 1.3
""",
            "call": "compute_shear_wavevectors(Nx, Ny, Lx, Ly, Omega, t)",
            "gold_call": "_oracle_compute_shear_wavevectors(Nx, Ny, Lx, Ly, Omega, t)",
        },
    ]
