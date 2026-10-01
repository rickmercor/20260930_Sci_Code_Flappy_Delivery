"""
Build the cell-centered coordinate cube of the shearing box and evaluate the single-mode, vertically Gaussian source density on it.

Every quantity downstream of this step is sampled on a cell-centered (finite-volume) grid rather than a node-centered one: for a box of side L resolved by N cells the coordinates are x_i = -L/2 + (i + 1/2) * L/N, so the grid spans [-L/2, L/2] without ever sampling either face and, for even N, never samples the midplane z = 0. This convention is what a finite-volume magnetohydrodynamics code stores, and it matters here for two reasons: the half-cell offset shifts the sampled extrema of the potential away from the continuum ones, and it keeps the density strictly inside the box so that the source is compactly supported, which is the precondition for the truncated free-space kernel to be exact. The benchmark source is a single horizontal Fourier mode with a Gaussian vertical profile, rho = cos(kx*x) * cos(ky*y) * exp(-z^2 / (2*H^2)) with kx = 2*pi*m_x/L and ky = 2*pi*m_y/L, truncated to the box and vacuum outside it. Separability is deliberate: the horizontal factor is an exact eigenfunction of the in-plane Laplacian, so the three-dimensional Poisson problem collapses to a single one-dimensional screened problem in height whose solution can be written in closed form and used as the reference field.

Returns
-------
np.ndarray of shape (4, N, N, N), float: [0], [1], [2] are the cell-centered x, y, z coordinate cubes and [3] is the source density evaluated on them.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def build_density_grid(N: int, L: float, H: float, m_x: int = 1,
                       m_y: int = 1) -> np.ndarray:
    """Build the cell-centered grid and the single-mode Gaussian density.

    Parameters
    ----------
    N : int
        Number of cells per dimension of the cubic box (N >= 4).
    L : float
        Box size in every dimension (L > 0).
    H : float
        Gaussian scale height of the vertical density profile (H > 0).
    m_x : int
        Integer mode number of the density in x, kx = 2*pi*m_x/L.
    m_y : int
        Integer mode number of the density in y, ky = 2*pi*m_y/L (m_x and
        m_y must not both be zero).

    Returns
    -------
    grid : np.ndarray
        Array of shape (4, N, N, N). grid[0], grid[1], grid[2] are the
        cell-centered x, y, z coordinate cubes produced with "ij" indexing
        from x_i = -L/2 + (i + 1/2) * L/N, and grid[3] is the density
        cos(kx*x) * cos(ky*y) * exp(-0.5 * (z/H)**2) evaluated on them.

    Notes
    -----
    The evaluation sandbox may execute this function in isolation: include
    every import your implementation needs (for example
    ``import numpy as np``) inside the function body.
    """
    return np.zeros((4, N, N, N), dtype=float)  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_build_density_grid(N: int, L: float, H: float, m_x: int = 1,
                               m_y: int = 1) -> np.ndarray:
    # Local imports keep the oracle self-contained when the harness
    # executes it in isolation.
    import numpy as np

    if not (isinstance(N, (int, np.integer)) and not isinstance(N, bool) and N >= 4):
        raise ValueError("N must be an integer >= 4")
    if not (isinstance(L, (int, float)) and np.isfinite(L) and float(L) > 0.0):
        raise ValueError("L must be a finite number > 0")
    if not (isinstance(H, (int, float)) and np.isfinite(H) and float(H) > 0.0):
        raise ValueError("H must be a finite number > 0")
    for name, val in (("m_x", m_x), ("m_y", m_y)):
        if not (isinstance(val, (int, np.integer)) and not isinstance(val, bool)):
            raise ValueError(f"{name} must be an integer")
    if int(m_x) == 0 and int(m_y) == 0:
        raise ValueError("m_x and m_y must not both be zero")

    N = int(N)
    L = float(L)
    H = float(H)

    # -- Cell-centered coordinates: the first centre sits half a cell inside
    #    the lower face, so no sample lands on a box face.
    d = L / N
    coords = -L / 2.0 + (np.arange(N) + 0.5) * d
    x_3d, y_3d, z_3d = np.meshgrid(coords, coords, coords, indexing="ij")

    # -- Single in-plane Fourier mode with a truncated Gaussian vertical
    #    profile; vacuum outside the box is implied by the box-confined support.
    kx_mode = 2.0 * np.pi * int(m_x) / L
    ky_mode = 2.0 * np.pi * int(m_y) / L
    rho = np.cos(kx_mode * x_3d) * np.cos(ky_mode * y_3d) * np.exp(-0.5 * (z_3d / H) ** 2)

    return np.stack([x_3d, y_3d, z_3d, rho]).astype(float)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Valid: the benchmark configuration (normal scenario) ---
        {
            "setup": """import numpy as np
N = 32
L = 6.0
H = 1.0
""",
            "call": "build_density_grid(N, L, H)",
            "gold_call": "_oracle_build_density_grid(N, L, H)",
        },
        # --- Valid: coarse grid used by the convergence check ---
        {
            "setup": """import numpy as np
N = 16
L = 6.0
H = 1.0
""",
            "call": "build_density_grid(N, L, H, m_x=1, m_y=1)",
            "gold_call": "_oracle_build_density_grid(N, L, H, m_x=1, m_y=1)",
        },
        # --- Edge: thick layer H = L/2, density still appreciable at the edges ---
        {
            "setup": """import numpy as np
N = 8
L = 6.0
H = 3.0
""",
            "call": "build_density_grid(N, L, H)",
            "gold_call": "_oracle_build_density_grid(N, L, H)",
        },
        # --- Edge: anisotropic mode numbers, one of them zero ---
        {
            "setup": """import numpy as np
N = 8
L = 6.0
H = 1.0
""",
            "call": "build_density_grid(N, L, H, m_x=2, m_y=0)",
            "gold_call": "_oracle_build_density_grid(N, L, H, m_x=2, m_y=0)",
        },
        # --- Boundary: smallest admissible grid ---
        {
            "setup": """import numpy as np
N = 4
L = 1.0
H = 0.25
""",
            "call": "build_density_grid(N, L, H)",
            "gold_call": "_oracle_build_density_grid(N, L, H)",
        },
        # --- Pinned values: cell-centred convention on the benchmark grid ---
        # Expected values derived independently from x_i = -L/2 + (i+1/2)L/N,
        # not from the oracle: with L = 6 and N = 32 the extreme cell centres
        # are -+2.90625 (half a cell inside each face, so no sample lands on a
        # face), and the corner density is cos^2(2*pi*2.90625/6)*exp(-2.90625^2/2)
        # = 0.0145117.
        {
            "setup": """import numpy as np
EXPECTED = np.array([-2.90625, 2.90625, 0.0145117])
""",
            "call": ("np.round(np.array([build_density_grid(32, 6.0, 1.0)[0].min(),"
                     " build_density_grid(32, 6.0, 1.0)[0].max(),"
                     " build_density_grid(32, 6.0, 1.0)[3][0, 0, 0]], dtype=float), 7)"),
            "gold_call": "EXPECTED",
        },
    ]
