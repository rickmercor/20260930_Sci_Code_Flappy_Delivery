"""
Solve ****one**** of the four decoupled constant-coefficient density subsystems of the scheme. After the sum-and-difference transformation and after the chemical potential has been eliminated, each subsystem is a single biharmonic-type equation in one unknown $\\varphi$, driven by two explicit right-hand sides $G_{13}$ and $G_{24}$ and carrying a sign $s=\\pm1$ in front of the shifted bi-Laplacian - $s=+1$ for a sum system, $s=-1$ for a difference system. Return the solution and the corresponding potential stacked along a new leading axis in the order $[\\varphi,\\ \\mu]$, so that the shape is `(2,) + G13.shape`. The equation contains one ****implicit nonlocal term****, and part of the task is to see that its value is available in closed form before the solve.

The nonlocal Lagrange multiplier that makes the $L^2$ gradient flow mass conserving subtracts the spatial mean of the chemical potential. Substituting the potential equation into the density equation therefore leaves a term proportional to the mean of the unknown, which looks implicit and global. It is neither, and seeing why is the point of this step. A term constant in space touches the zero Fourier mode alone, so the solve is still a mode-by-mode division; and the mean of the unknown is itself available in closed form ****before**** anything is solved, from testing the density equation against the constants, which annihilates the multiplier. Substituting that value makes the zero-mode equation an identity, so no special case is needed anywhere: assemble the right-hand side with the known mean, divide by the symbol, transform back. The symbol is where the care is needed. For the difference system its fourth-order part is negative over a disc of wavenumbers and vanishes on that disc's boundary, so its minimum over all wavenumbers is not where a first glance puts it, and positivity there is not automatic - it is what the lower bound on the stabilization constant, the one the source proves for the lower-boundedness of its energy, is there to guarantee. Establish that minimum and check it for this parameter set rather than assuming the solve is well posed.

****--- Formulas ---****

The equation this step inverts, its Fourier symbol, the assembly of its right-hand side from $G_{13}$, $G_{24}$ and the known mean, and the potential that goes with the solution are all the source's, reached from the reformulated system by converting the Crank-Nicolson averages to new-layer unknowns, summing and differencing the coupled pair, and eliminating the potential. Fetch them; do not reconstruct them by analogy. Fixed here, and graded: `sign` is $+1$ for a sum system and $-1$ for a difference system, and it multiplies the shifted bi-Laplacian built on the interspecies length scale $a_{12}$; the mean of $\\varphi$ is determined before the solve, not iterated for; the mobility and the stabilization constant that enter are the density ones, $\\mathcal M_\\phi$ and $S_\\phi$; and the return order is $[\\varphi,\\ \\mu]$.

Returns
-------
`np.ndarray` of shape `(2,) + G13.shape`, real and finite. The mean of the first entry is fixed by the data alone rather than by the solve, and is zero to round-off whenever the mean of `G13` is - a self-check worth running once you have worked out what that mean must be.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def pfc_split_solve(G13, G24, sign, cell, dt, params=None):
    """G13, G24: the two explicit right-hand sides, shape (Nx, Ny).
    sign: +1 for a sum system, -1 for a difference system.
    cell: domain edge lengths, a scalar or a length-2 sequence.
    dt: the time step. params: as in pfc_free_energy.
    Return the real array [phi, mu] of shape (2,) + G13.shape.
    Raise ValueError if G13 and G24 are not finite two-dimensional arrays
    of equal shape, if sign is not +1 or -1, if dt is not a finite
    positive scalar, or if the split operator is singular for the given
    parameters."""
    # Implement per the specification above.
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_pfc_split_solve(G13, G24, sign, cell, dt, params=None):
    p = _check_params(params)
    G13 = np.asarray(G13, float)
    G24 = np.asarray(G24, float)
    if G13.ndim != 2 or G24.shape != G13.shape:
        raise ValueError("G13 and G24 must be two-dimensional fields of equal shape")
    if not (np.all(np.isfinite(G13)) and np.all(np.isfinite(G24))):
        raise ValueError("the right-hand sides must be finite")
    s = float(sign)
    if s not in (1.0, -1.0):
        raise ValueError("sign must be +1 or -1")
    dt = float(dt)
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError("dt must be a finite positive scalar")
    L = _check_cell(cell, G13.shape)
    a12, Sp, Mp = p["a12"], p["S_phi"], p["M_phi"]
    k2 = _k2(G13.shape, L)
    n = G13.size

    mean13 = float(np.sum(G13)) / n
    mean24 = float(np.sum(G24)) / n
    pbar = Mp * dt * mean13
    rhs = (G13 - 0.5 * (G24 - mean24)
           + 0.5 * (Sp + 0.5 * s * a12 ** 2) * pbar)
    A = 1.0 / (Mp * dt) + 0.5 * k2 ** 2 + 0.5 * Sp + 0.25 * s * (a12 - k2) ** 2
    if np.any(np.abs(A) < 1.0e-12 * np.max(np.abs(A))):
        raise ValueError("the split operator is singular for this parameter set")
    psi = np.real(np.fft.ifft2(np.fft.fft2(rhs) / A))
    d = _oracle_pfc_spectral_derivatives(psi, L, a12)
    mu = d[3] + Sp * psi + 0.5 * s * d[4] + G24
    return np.stack([psi, mu])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of step test specifications (setup/call/gold_call)."""
    FLD = ("N = 24\n"
           "Lc = (32.0, 32.0)\n"
           "x = np.arange(N) * (Lc[0] / N)\n"
           "y = np.arange(N) * (Lc[1] / N)\n"
           "X, Y = np.meshgrid(x, y, indexing='ij')\n"
           "k = 2.0 * np.pi / 32.0\n"
           "phi1 = 0.4 * np.cos(3 * k * X) * np.sin(2 * k * Y) + 0.10\n"
           "phi2 = 0.3 * np.cos(2 * k * X) * np.cos(3 * k * Y) - 0.05\n"
           "Mf = np.stack([0.5 * np.sin(k * X) * np.sin(2 * k * Y),\n"
           "               0.4 * np.cos(2 * k * X) * np.cos(k * Y)])\n"
           "Hf = np.stack([np.sin(2 * k * X) * np.cos(k * Y), np.cos(k * Y)])\n")

    STIFF = "PS = {'gamma1': 1.0, 'gamma2': 1.0, 'eta1': -100.0, 'eta2': -100.0}\n"
    STATE = ("St, Hf = _pfc_initial_state((24, 24), (24.0, 24.0), 2.0e-3)\n")
    return [
        # normal: the SUM system, whose operator carries +1/4 of the shifted
        # bi-Laplacian, driven by a generic pair of right-hand sides.
        {"setup": "import numpy as np\n" + FLD +
                  "G13 = 3.0 * phi1 + 0.5 * phi2\nG24 = -2.0 * phi2 + 0.7 * phi1\n",
         "call": "pfc_split_solve(G13, G24, 1, Lc, 2.0e-3)",
         "gold_call": "_oracle_pfc_split_solve(G13, G24, 1, Lc, 2.0e-3)"},
        # normal: the DIFFERENCE system on the same data - the sign of the
        # shifted bi-Laplacian flips in both the operator and the potential.
        {"setup": "import numpy as np\n" + FLD +
                  "G13 = 3.0 * phi1 + 0.5 * phi2\nG24 = -2.0 * phi2 + 0.7 * phi1\n",
         "call": "pfc_split_solve(G13, G24, -1, Lc, 2.0e-3)",
         "gold_call": "_oracle_pfc_split_solve(G13, G24, -1, Lc, 2.0e-3)"},
        # boundary: a MEAN-FREE first right-hand side, so the mean of the answer
        # is exactly zero and the nonlocal correction contributes nothing.
        {"setup": "import numpy as np\n" + FLD +
                  "G13 = np.zeros_like(phi1)\nG24 = 4.0 * phi1 * phi2 + 1.5\n",
         "call": "pfc_split_solve(G13, G24, 1, Lc, 5.0e-3)",
         "gold_call": "_oracle_pfc_split_solve(G13, G24, 1, Lc, 5.0e-3)"},
        # boundary: a right-hand side with a large mean, which makes the whole
        # answer hinge on the nonlocal term; a rectangular box at the same time.
        {"setup": "import numpy as np\n"
                  "x = np.arange(20) * (24.0 / 20)\ny = np.arange(16) * (16.0 / 16)\n"
                  "X, Y = np.meshgrid(x, y, indexing='ij')\n"
                  "G13 = 5.0 + np.sin(2 * np.pi * X / 24.0)\n"
                  "G24 = -3.0 + 2.0 * np.cos(4 * np.pi * Y / 16.0)\n",
         "call": "pfc_split_solve(G13, G24, -1, (24.0, 16.0), 1.0e-2)",
         "gold_call": "_oracle_pfc_split_solve(G13, G24, -1, (24.0, 16.0), 1.0e-2)"},
        # edge: a very small time step, where 1/(M_phi*dt) dominates the operator
        # and the answer degenerates to M_phi*dt times the right-hand side.
        {"setup": "import numpy as np\n" + FLD +
                  "G13 = phi1 - phi2\nG24 = phi1 * phi2\n",
         "call": "pfc_split_solve(G13, G24, 1, Lc, 1.0e-6)",
         "gold_call": "_oracle_pfc_split_solve(G13, G24, 1, Lc, 1.0e-6)"},
        # edge: a large time step with an enlarged interspecies length scale and
        # a reduced stabiliser, the regime where the DIFFERENCE operator comes
        # closest to changing sign.
        {"setup": "import numpy as np\n" + FLD +
                  "PC = {'a12': 2.6, 'S_phi': 4.0}\nG13 = phi1\nG24 = phi2\n",
         "call": "pfc_split_solve(G13, G24, -1, Lc, 1.0, PC)",
         "gold_call": "_oracle_pfc_split_solve(G13, G24, -1, Lc, 1.0, PC)"},
    ]


STATE = ("St, Hf = _pfc_initial_state((24, 24), (24.0, 24.0), 2.0e-3)\n")
