"""
Build the reciprocal-space grid of the periodic cell: the signed plane-wave wavevectors in the ordering a discrete Fourier transform produces, their squared magnitudes, and the Coulomb factor of the Hartree kernel with its divergent component removed.

A periodic cell of length $L$ discretised on $N_g$ points carries $N_g$ plane waves, and the ordering matters: a discrete Fourier transform indexes them $0, 1, \\dots$ up to the Nyquist mode and then **negatively**, so the second half of the array holds the negative wavevectors rather than the largest positive ones. Getting this wrong leaves the magnitudes $|G|$ monotonically increasing instead of symmetric about the Nyquist mode, which silently changes every kinetic energy and every Coulomb factor built from them.




The Coulomb interaction in reciprocal space is $4\\pi/|G|^2$, which diverges as $G \\to 0$. In a periodic cell that divergence is not physical: it is cancelled by the compensating uniform background of positive charge that makes the cell neutral overall, and the standard treatment is simply to set the $G = 0$ component of the Coulomb factor to zero. The source paper states exactly this convention when it bounds the kernel. A physical consequence worth holding on to is that the kernel then annihilates the mean of any density change, and since every density response in this problem integrates to zero, nothing physical is lost by the removal.




****--- Formulas ---****

With $j = 0, \\dots, N_g - 1$ the array index, the signed integer index and the wavevectors are




$$k_j = \\begin{cases} j & j < N_g/2 \\\\ j - N_g & j \\ge N_g/2 \\end{cases}, \\qquad G_j = \\frac{2\\pi k_j}{L}, \\qquad |G_j|^2 = G_j^2,$$




$$v(G_j) = \\frac{s}{N_g}\\cdot\\frac{4\\pi}{|G_j|^2} \\quad (|G_j|^2 > 10^{-12}), \\qquad v(G_j) = 0 \\quad (|G_j|^2 \\le 10^{-12}),$$




where $s$ is the supplied Coulomb scale. Return the three arrays stacked as the columns of one $(N_g, 3)$ array, in the order $G$, $|G|^2$, $v(G)$.

Returns
-------
`np.ndarray` of shape `(n_grid, 3)`, real: column 0 the signed wavevectors, column 1 their squares, column 2 the Coulomb factor whose first entry is exactly zero.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def plane_wave_grid(n_grid, cell_length, coulomb_scale):
    """n_grid: int >= 2, number of grid points. cell_length: float > 0, the cell length L.
    coulomb_scale: float, the scale s multiplying the Coulomb factor.
    Return an (n_grid, 3) real array whose columns are the signed wavevectors G, their
    squared magnitudes |G|^2, and the Coulomb factor v(G) with its G = 0 entry zeroed."""
    # Implement per the formulas above.
    grid = None
    return grid

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_plane_wave_grid(n_grid, cell_length, coulomb_scale):
    ng = int(n_grid)
    if ng != n_grid or ng < 2:
        raise ValueError("n_grid must be an integer >= 2")
    L = float(cell_length)
    if not np.isfinite(L) or L <= 0.0:
        raise ValueError("cell_length must be finite and positive")
    cs = float(coulomb_scale)
    if not np.isfinite(cs):
        raise ValueError("coulomb_scale must be finite")
    idx = np.arange(ng)
    kn = np.where(idx < ng // 2, idx, idx - ng).astype(float)
    G = 2.0 * np.pi * kn / L
    Gsq = G ** 2
    nz = Gsq > 1e-12
    v = np.zeros(ng, dtype=float)
    v[nz] = 4.0 * np.pi / Gsq[nz]
    v *= cs / ng
    out = np.zeros((ng, 3), dtype=float)
    out[:, 0] = G
    out[:, 1] = Gsq
    out[:, 2] = v
    return out

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of step test specifications (setup/call/gold_call)."""
    return [
        # normal: the main configuration of the problem statement.
        {"setup": "import numpy as np\n",
         "call": "plane_wave_grid(20, 8.0, 28.0)",
         "gold_call": "_oracle_plane_wave_grid(20, 8.0, 28.0)"},
        # normal: a second cell, different length and scale.
        {"setup": "import numpy as np\n",
         "call": "plane_wave_grid(8, 5.0, 6.0)",
         "gold_call": "_oracle_plane_wave_grid(8, 5.0, 6.0)"},
        # boundary: a third cell, so a hard-coded length fails.
        {"setup": "import numpy as np\n",
         "call": "plane_wave_grid(6, 4.0, 15.0)",
         "gold_call": "_oracle_plane_wave_grid(6, 4.0, 15.0)"},
        # edge: the Coulomb factor has its G = 0 entry removed exactly.
        {"setup": "import numpy as np\n",
         "call": "int(plane_wave_grid(12, 3.0, 9.0)[0, 2] == 0.0)",
         "gold_call": "1"},
        # edge: the wavevector ordering is signed, so the last entry is the smallest negative one.
        {"setup": "import numpy as np\n",
         "call": "int(np.allclose(plane_wave_grid(10, 4.0, 1.0)[9, 0], -2*np.pi/4.0, rtol=1e-12, atol=0))",
         "gold_call": "1"},
        # edge: the second column really is the square of the first.
        {"setup": "import numpy as np\n",
         "call": "int(np.allclose(plane_wave_grid(14, 6.0, 3.0)[:, 1], plane_wave_grid(14, 6.0, 3.0)[:, 0]**2, rtol=1e-12, atol=0))",
         "gold_call": "1"},
    ]
