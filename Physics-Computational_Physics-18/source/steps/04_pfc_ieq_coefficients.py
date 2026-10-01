"""
Evaluate the ****invariant-energy-quadratization auxiliary variable**** $U$ and the three coefficient functions $H_1$, $H_2$ and $\\boldsymbol R$ that multiply it in the reformulated chemical potentials. Return them stacked along a new leading axis in the order $[U,\\ H_1,\\ H_2,\\ R_x,\\ R_y]$, so that the shape is `(5,) + phi1.shape`. The radicand must be strictly positive: raise a `ValueError` if it is not, since a non-positive radicand means the shift constant $B$ was chosen too small for the state, and that is a modelling failure rather than a number to be returned.

Quadratization replaces a nonlinear energy by a quadratic one at the price of one extra unknown. Because the nonlinear energy density plus the two quadratic pieces that the reformulation moves under the root is bounded from below - which is precisely what the higher-order regularization terms guarantee - adding a large enough positive constant $B$ makes it positive, so its square root is a well-defined real field, and the reformulated energy is a sum of quadratic forms plus $\\|U\\|^2$. The three nonlinear variational derivatives then factor through $U$, each as a coefficient function times $U$. What sits under the root, what the three coefficient functions are, and where the two stabilization constants enter are the source's, and the problem statement directs you to fetch them: they are not reconstructible by inspection, because the stabilization constants appear twice with opposite signs - once under the root and once as explicit quadratic terms added back - so that they change nothing at the continuous level while supplying the discrete control the scheme's stability proof needs, and because the densities and the magnetization are not treated symmetrically there.

****--- Formulas ---****

The auxiliary variable, the radicand it is the root of, and the three coefficient functions are the source's; take them from the paper rather than guessing a plausible quadratization. Fixed here, and graded: $U$ is the ****positive**** root; the shift constant enters the radicand as $+B$ with the value given in the problem statement; the four objects are built from $N$, $N_1$, $N_2$ and $\\boldsymbol N_3$ of step 3 together with $\\phi_1$, $\\phi_2$ and $\\boldsymbol M$ themselves, and from nothing else; the return order is $[U,\\ H_1,\\ H_2,\\ R_x,\\ R_y]$; and if any grid value of the radicand is $\\le0$, raise `ValueError` rather than returning a `nan`.

Returns
-------
`np.ndarray` of shape `(5,) + phi1.shape`, real and finite, with $U>0$ everywhere. For the production shift constant $B=10^7$, $U$ is very nearly the constant $\\sqrt B\\approx 3162.3$ and the three coefficients are of order $10^{-3}$.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def pfc_ieq_coefficients(phi1, phi2, M, cell, params=None, H=None):
    """Arguments exactly as in pfc_free_energy.
    Return the real array [U, H1, H2, R_x, R_y] of shape (5,) + phi1.shape.
    Raise ValueError if the radicand is not strictly positive everywhere.
    Raise ValueError for mismatched or non-finite input fields, invalid
    domain lengths or parameters, or a non-finite IEQ radicand."""
    # Implement per the specification above.
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_pfc_ieq_coefficients(phi1, phi2, M, cell, params=None, H=None):
    p = _check_params(params)
    phi1 = np.asarray(phi1, float)
    phi2 = np.asarray(phi2, float)
    M = np.asarray(M, float)
    L = _check_cell(cell, phi1.shape)
    T = _oracle_pfc_nonlinear_terms(phi1, phi2, M, L, params, H)
    N, N1, N2 = T[0], T[1], T[2]
    N3 = T[3:5]
    m2 = M[0] ** 2 + M[1] ** 2
    rad = (0.5 * p["a1"] * phi1 ** 2 + 0.5 * p["a2"] * phi2 ** 2 + N
           - 0.5 * p["S_phi"] * phi1 ** 2 - 0.5 * p["S_phi"] * phi2 ** 2
           - 0.5 * p["S_m"] * m2 + p["B"])
    if not np.all(np.isfinite(rad)) or np.any(rad <= 0.0):
        raise ValueError("the IEQ radicand must be finite and strictly positive")
    U = np.sqrt(rad)
    H1 = (p["a1"] * phi1 - p["S_phi"] * phi1 + N1) / U
    H2 = (p["a2"] * phi2 - p["S_phi"] * phi2 + N2) / U
    R = (-p["S_m"] * M + N3) / U
    return np.stack([U, H1, H2, R[0], R[1]])


def _pfc_initial_fields(grid, cell):
    Nx, Ny = int(grid[0]), int(grid[1])
    Lx, Ly = float(cell[0]), float(cell[1])
    x = np.arange(Nx) * (Lx / Nx)
    y = np.arange(Ny) * (Ly / Ny)
    X, Y = np.meshgrid(x, y, indexing="ij")
    kx, ky = 2.0 * np.pi / Lx, 2.0 * np.pi / Ly
    phi1 = np.cos(8.0 * kx * X) * np.sin(8.0 * ky * Y)
    phi2 = np.cos(8.0 * kx * X) * np.cos(8.0 * ky * Y)
    M = np.stack([np.sin(2.0 * kx * X) * np.sin(2.0 * ky * Y),
                  np.cos(2.0 * kx * X) * np.cos(2.0 * ky * Y)])
    Hf = np.stack([np.sin(2.0 * kx * X) * np.cos(ky * Y),
                   np.cos(ky * Y)])
    return phi1, phi2, M, Hf


def _pfc_initial_state(grid, cell, dt, params=None):
    """Initial layer, the two backward ghost layers and Q^0 = 1."""
    p = _check_params(params)
    L = _check_cell(cell, (int(grid[0]), int(grid[1])))
    phi1, phi2, M, Hf = _pfc_initial_fields(grid, L)
    C = _oracle_pfc_ieq_coefficients(phi1, phi2, M, L, params, Hf)
    U, H1, H2 = C[0], C[1], C[2]
    R = C[3:5]
    d1 = _oracle_pfc_spectral_derivatives(phi1, L, p["a12"])
    d2 = _oracle_pfc_spectral_derivatives(phi2, L, p["a12"])
    lapM = np.stack([_oracle_pfc_spectral_derivatives(M[j], L)[2] for j in range(2)])
    mu1 = d1[3] + 2.0 * p["a1"] * d1[2] + 0.5 * d2[4] + p["S_phi"] * phi1 + H1 * U
    mu2 = d2[3] + 2.0 * p["a2"] * d2[2] + 0.5 * d1[4] + p["S_phi"] * phi2 + H2 * U
    mu3 = -p["omega0"] * lapM + p["S_m"] * M + R * U
    v1 = -p["M_phi"] * (mu1 - np.mean(mu1))
    v2 = -p["M_phi"] * (mu2 - np.mean(mu2))
    vM = -p["M_m"] * mu3
    vU = 0.5 * (H1 * v1 + H2 * v2 + R[0] * vM[0] + R[1] * vM[1])
    layer = np.stack([phi1, phi2, M[0], M[1], mu1, mu2, mu3[0], mu3[1], U])
    z = np.zeros_like(phi1)
    vel = np.stack([v1, v2, vM[0], vM[1], z, z, z, z, vU])
    return np.stack([layer, layer - dt * vel, layer - 2.0 * dt * vel]), Hf

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Original cases plus declared invalid-input contract checks."""
    return [{'setup': 'import numpy as np\n'
               'N = 24\n'
               'Lc = (32.0, 32.0)\n'
               'x = np.arange(N) * (Lc[0] / N)\n'
               'y = np.arange(N) * (Lc[1] / N)\n'
               "X, Y = np.meshgrid(x, y, indexing='ij')\n"
               'k = 2.0 * np.pi / 32.0\n'
               'phi1 = 0.4 * np.cos(3 * k * X) * np.sin(2 * k * Y) + 0.10\n'
               'phi2 = 0.3 * np.cos(2 * k * X) * np.cos(3 * k * Y) - 0.05\n'
               'Mf = np.stack([0.5 * np.sin(k * X) * np.sin(2 * k * Y),\n'
               '               0.4 * np.cos(2 * k * X) * np.cos(k * Y)])\n'
               'Hf = np.stack([np.sin(2 * k * X) * np.cos(k * Y), np.cos(k * Y)])\n',
      'call': 'pfc_ieq_coefficients(phi1, phi2, Mf, Lc, None, Hf)',
      'gold_call': '_oracle_pfc_ieq_coefficients(phi1, phi2, Mf, Lc, None, Hf)'},
     {'setup': 'import numpy as np\n'
               'N = 24\n'
               'Lc = (32.0, 32.0)\n'
               'x = np.arange(N) * (Lc[0] / N)\n'
               'y = np.arange(N) * (Lc[1] / N)\n'
               "X, Y = np.meshgrid(x, y, indexing='ij')\n"
               'k = 2.0 * np.pi / 32.0\n'
               'phi1 = 0.4 * np.cos(3 * k * X) * np.sin(2 * k * Y) + 0.10\n'
               'phi2 = 0.3 * np.cos(2 * k * X) * np.cos(3 * k * Y) - 0.05\n'
               'Mf = np.stack([0.5 * np.sin(k * X) * np.sin(2 * k * Y),\n'
               '               0.4 * np.cos(2 * k * X) * np.cos(k * Y)])\n'
               'Hf = np.stack([np.sin(2 * k * X) * np.cos(k * Y), np.cos(k * Y)])\n'
               "PB = {'B': 5.0}\n",
      'call': 'pfc_ieq_coefficients(phi1, phi2, Mf, Lc, PB, Hf)',
      'gold_call': '_oracle_pfc_ieq_coefficients(phi1, phi2, Mf, Lc, PB, Hf)'},
     {'setup': 'import numpy as np\n'
               'N = 24\n'
               'Lc = (32.0, 32.0)\n'
               'x = np.arange(N) * (Lc[0] / N)\n'
               'y = np.arange(N) * (Lc[1] / N)\n'
               "X, Y = np.meshgrid(x, y, indexing='ij')\n"
               'k = 2.0 * np.pi / 32.0\n'
               'phi1 = 0.4 * np.cos(3 * k * X) * np.sin(2 * k * Y) + 0.10\n'
               'phi2 = 0.3 * np.cos(2 * k * X) * np.cos(3 * k * Y) - 0.05\n'
               'Mf = np.stack([0.5 * np.sin(k * X) * np.sin(2 * k * Y),\n'
               '               0.4 * np.cos(2 * k * X) * np.cos(k * Y)])\n'
               'Hf = np.stack([np.sin(2 * k * X) * np.cos(k * Y), np.cos(k * Y)])\n'
               "PZ = {'S_phi': 0.0, 'S_m': 0.0, 'B': 20.0}\n",
      'call': 'pfc_ieq_coefficients(phi1, phi2, Mf, Lc, PZ, Hf)',
      'gold_call': '_oracle_pfc_ieq_coefficients(phi1, phi2, Mf, Lc, PZ, Hf)'},
     {'setup': 'import numpy as np\n'
               'N = 24\n'
               'Lc = (32.0, 32.0)\n'
               'x = np.arange(N) * (Lc[0] / N)\n'
               'y = np.arange(N) * (Lc[1] / N)\n'
               "X, Y = np.meshgrid(x, y, indexing='ij')\n"
               'k = 2.0 * np.pi / 32.0\n'
               'phi1 = 0.4 * np.cos(3 * k * X) * np.sin(2 * k * Y) + 0.10\n'
               'phi2 = 0.3 * np.cos(2 * k * X) * np.cos(3 * k * Y) - 0.05\n'
               'Mf = np.stack([0.5 * np.sin(k * X) * np.sin(2 * k * Y),\n'
               '               0.4 * np.cos(2 * k * X) * np.cos(k * Y)])\n'
               'Hf = np.stack([np.sin(2 * k * X) * np.cos(k * Y), np.cos(k * Y)])\n'
               "PA = {'a1': 0.6, 'a2': 1.7, 'B': 8.0}\n",
      'call': 'pfc_ieq_coefficients(phi1, phi2, Mf, Lc, PA, Hf)',
      'gold_call': '_oracle_pfc_ieq_coefficients(phi1, phi2, Mf, Lc, PA, Hf)'},
     {'setup': 'import numpy as np\n'
               'N = 24\n'
               'Lc = (32.0, 32.0)\n'
               'x = np.arange(N) * (Lc[0] / N)\n'
               'y = np.arange(N) * (Lc[1] / N)\n'
               "X, Y = np.meshgrid(x, y, indexing='ij')\n"
               'k = 2.0 * np.pi / 32.0\n'
               'phi1 = 0.4 * np.cos(3 * k * X) * np.sin(2 * k * Y) + 0.10\n'
               'phi2 = 0.3 * np.cos(2 * k * X) * np.cos(3 * k * Y) - 0.05\n'
               'Mf = np.stack([0.5 * np.sin(k * X) * np.sin(2 * k * Y),\n'
               '               0.4 * np.cos(2 * k * X) * np.cos(k * Y)])\n'
               'Hf = np.stack([np.sin(2 * k * X) * np.cos(k * Y), np.cos(k * Y)])\n'
               'Mb = 4.0 * Mf\n'
               "PN = {'B': 43.0, 'S_m': 40.0}\n",
      'call': 'pfc_ieq_coefficients(phi1, phi2, Mb, Lc, PN, Hf)',
      'gold_call': '_oracle_pfc_ieq_coefficients(phi1, phi2, Mb, Lc, PN, Hf)'},
     {'setup': 'import numpy as np\n'
               'phi1 = np.full((10, 14), -0.3)\n'
               'phi2 = np.full((10, 14), 0.4)\n'
               'Mf = np.stack([np.full((10, 14), 0.5), np.full((10, 14), -0.6)])\n'
               "PB = {'B': 12.0}\n",
      'call': 'pfc_ieq_coefficients(phi1, phi2, Mf, (15.0, 21.0), PB)',
      'gold_call': '_oracle_pfc_ieq_coefficients(phi1, phi2, Mf, (15.0, 21.0), PB)'},
     {'setup': 'import numpy as np\n'
               'N = 24\n'
               'Lc = (32.0, 32.0)\n'
               'x = np.arange(N) * (Lc[0] / N)\n'
               'y = np.arange(N) * (Lc[1] / N)\n'
               "X, Y = np.meshgrid(x, y, indexing='ij')\n"
               'k = 2.0 * np.pi / 32.0\n'
               'phi1 = 0.4 * np.cos(3 * k * X) * np.sin(2 * k * Y) + 0.10\n'
               'phi2 = 0.3 * np.cos(2 * k * X) * np.cos(3 * k * Y) - 0.05\n'
               'Mf = np.stack([0.5 * np.sin(k * X) * np.sin(2 * k * Y),\n'
               '               0.4 * np.cos(2 * k * X) * np.cos(k * Y)])\n'
               'Hf = np.stack([np.sin(2 * k * X) * np.cos(k * Y), np.cos(k * Y)])\n'
               'Mb = 4.0 * Mf\n'
               "PX = {'B': 42.0, 'S_m': 40.0}\n"
               'def _probe(f):\n'
               '    try:\n'
               '        f(phi1, phi2, Mb, Lc, PX, Hf)\n'
               '    except ValueError:\n'
               '        return 1.0\n'
               '    except Exception:\n'
               '        return 2.0\n'
               '    return 0.0\n',
      'call': '_probe(pfc_ieq_coefficients)',
      'gold_call': '_probe(_oracle_pfc_ieq_coefficients)'},
     {'setup': 'import numpy as np\n'
               'z = np.zeros((4, 4))\n'
               'M = np.zeros((2, 4, 4))\n'
               'H = np.zeros_like(M)\n'
               'H[0, 0, 0] = np.nan\n'
               'def _probe(f):\n'
               '    try:\n'
               '        f(z, z, M, (4.0, 4.0), H=H)\n'
               '    except ValueError:\n'
               '        return 1.0\n'
               '    except Exception:\n'
               '        return 2.0\n'
               '    return 0.0\n',
      'call': '_probe(pfc_ieq_coefficients)',
      'gold_call': '_probe(_oracle_pfc_ieq_coefficients)'},
     {'setup': 'import numpy as np\n'
               'z = np.zeros((4, 4))\n'
               'M = np.zeros((2, 4, 4))\n'
               'H = np.zeros_like(M)\n'
               'H[0, 0, 0] = np.inf\n'
               'def _probe(f):\n'
               '    try:\n'
               '        f(z, z, M, (4.0, 4.0), H=H)\n'
               '    except ValueError:\n'
               '        return 1.0\n'
               '    except Exception:\n'
               '        return 2.0\n'
               '    return 0.0\n',
      'call': '_probe(pfc_ieq_coefficients)',
      'gold_call': '_probe(_oracle_pfc_ieq_coefficients)'}]
