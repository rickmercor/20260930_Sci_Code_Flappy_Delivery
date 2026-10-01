"""
Evaluate the ****total free energy**** of the magnetic-coupled binary phase-field-crystal model on the grid: the sum of the binary PFC block, the Ginzburg-Landau magnetic block and the higher-order regularization block, integrated over the periodic domain. Return one float. Every integral is the plain grid sum times the cell area $h_xh_y$, and every spatial operator is the spectral one of step 1. This is the quantity the whole benchmark finally reports, and it is also the quantity the modified discrete energy of step 7 must reproduce at $t=0$, so an error here shows up twice.

The three blocks play different roles. The binary PFC block contains the two quartic operators $\\tfrac{\\phi_i}{2}(\\Delta+a_i)^2\\phi_i$, whose symbols vanish on a circle of wavenumbers and therefore select a periodic ground state, plus an interspecies term with its own length scale $a_{12}$, plus a double well in each density, plus a cubic **vacancy penalization** $\\tfrac{\\eta}{3}(|\\phi_i|^3-\\phi_i^3)$ that vanishes identically where $\\phi_i\\ge0$ and grows like $\\tfrac{2\\eta}{3}|\\phi_i|^3$ where $\\phi_i<0$, which is what keeps the densities positive, plus a coupling $\\tfrac{\\gamma_{12}}{2}\\phi_1^2\\phi_2^2$. The Ginzburg-Landau block is a standard magnetic double well with an exchange gradient term, a Zeeman term for the applied field, two terms that favour magnetic islands at atomic sites, and two magneto-elastic terms $-\\tfrac{\\eta_i}{2}(\\boldsymbol M\\cdot\\nabla\\phi_i)^2$ which, for the negative $\\eta_i$ used here, penalise magnetization aligned with a density gradient. The regularization block is what makes the whole functional bounded from below: the sixth-order term in $|\\boldsymbol M|$ dominates the quartic produced by the $\\gamma_i$ and $\\eta_i$ terms through $2ab\\le a^2+b^2$, and the quartic gradient terms do the same for the magneto-elastic ones. That lower bound is exactly what the quadratization of step 4 relies on.

**--- Formulas ---**

The three blocks $F_B$, $F_{GL}$ and $F_{Reg}$ are written out in the problem statement; use them exactly as given there, with the parameter values given there. Only the discrete reading is fixed here, and it is graded: each block is an energy **density** and the integral is the plain grid sum times the cell area $h_xh_y$; the interspecies term appears **once**, with the factor $\\tfrac12$, not once per species; and $|\\nabla\\boldsymbol M|^2=\\sum_{i,j}(\\partial_iM_j)^2$, $(\\boldsymbol M\\cdot\\nabla\\phi_i)^2$ and $|\\nabla\\phi_i|^4$ are formed pointwise from the first derivatives of step 1 and are never integrated by parts.

Returns
-------
`float`: the total free energy $E$ over the whole domain, finite. It is extensive - doubling the domain at fixed fields doubles it - and it is bounded from below for every parameter set with positive $\\theta,\\theta_1,\\theta_2$.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def pfc_free_energy(phi1, phi2, M, cell, params=None, H=None):
    """phi1, phi2: density fields of equal shape (Nx, Ny).
    M: magnetization, shape (2, Nx, Ny). H: applied field, same shape as M;
       None means no applied field.
    cell: domain edge lengths, a scalar or a length-2 sequence.
    params: dict of model-parameter overrides; None means the fixed set
       eps, M_phi, M_m, eta, gamma12, omega0, theta, theta1, theta2, alpha,
       beta, S_phi, S_m, a1, a2, a12, B, gamma1, gamma2, eta1, eta2.
    Return the total free energy of the model, a float.
    Raise ValueError for mismatched or non-finite fields, invalid domain
    lengths, or invalid model parameters. H=None means a zero applied field."""
    # Implement per the specification above.
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_pfc_free_energy(phi1, phi2, M, cell, params=None, H=None):
    p = _check_params(params)
    phi1 = np.asarray(phi1, float)
    phi2 = np.asarray(phi2, float)
    M = np.asarray(M, float)
    if phi1.ndim != 2 or phi2.shape != phi1.shape:
        raise ValueError("phi1 and phi2 must be two-dimensional fields of equal shape")
    if M.shape != (2,) + phi1.shape:
        raise ValueError("M must have shape (2,) + phi1.shape")
    L = _check_cell(cell, phi1.shape)
    H = np.zeros_like(M) if H is None else np.asarray(H, float)
    if H.shape != M.shape:
        raise ValueError("H must have the same shape as M")
    if not all(np.all(np.isfinite(v)) for v in (phi1, phi2, M, H)):
        raise ValueError("phi1, phi2, M and H must be finite")

    d1 = _oracle_pfc_spectral_derivatives(phi1, L, p["a1"])
    d2 = _oracle_pfc_spectral_derivatives(phi2, L, p["a2"])
    c12a = _oracle_pfc_spectral_derivatives(phi2, L, p["a12"])[4]
    g1 = d1[:2]
    g2 = d2[:2]
    m2 = M[0] ** 2 + M[1] ** 2
    mg1 = M[0] * g1[0] + M[1] * g1[1]
    mg2 = M[0] * g2[0] + M[1] * g2[1]
    gm2 = sum(_oracle_pfc_spectral_derivatives(M[j], L)[i] ** 2
              for j in range(2) for i in range(2))

    FB = (0.5 * phi1 * d1[4] + 0.5 * phi2 * d2[4] + 0.5 * phi1 * c12a
          + 0.25 * phi1 ** 4 - 0.5 * p["eps"] * phi1 ** 2
          + 0.25 * phi2 ** 4 - 0.5 * p["eps"] * phi2 ** 2
          + p["eta"] / 3.0 * (np.abs(phi1) ** 3 + np.abs(phi2) ** 3
                              - phi1 ** 3 - phi2 ** 3)
          + 0.5 * p["gamma12"] * phi1 ** 2 * phi2 ** 2)
    FGL = (0.5 * p["omega0"] * gm2 - 0.5 * p["alpha"] * m2
           + 0.25 * p["beta"] * m2 ** 2 - (M[0] * H[0] + M[1] * H[1])
           - p["gamma1"] * m2 * phi1 - p["gamma2"] * m2 * phi2
           - 0.5 * p["eta1"] * mg1 ** 2 - 0.5 * p["eta2"] * mg2 ** 2)
    FR = (p["theta"] / 6.0 * m2 ** 3
          + 0.25 * p["theta1"] * (g1[0] ** 2 + g1[1] ** 2) ** 2
          + 0.25 * p["theta2"] * (g2[0] ** 2 + g2[1] ** 2) ** 2)
    dv = np.prod(L) / phi1.size
    return float(np.sum(FB + FGL + FR) * dv)

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
      'call': 'pfc_free_energy(phi1, phi2, Mf, Lc, None, Hf)',
      'gold_call': '_oracle_pfc_free_energy(phi1, phi2, Mf, Lc, None, Hf)'},
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
               'Hf = np.stack([np.sin(2 * k * X) * np.cos(k * Y), np.cos(k * Y)])\n',
      'call': 'pfc_free_energy(phi1, phi2, Mf, Lc)',
      'gold_call': '_oracle_pfc_free_energy(phi1, phi2, Mf, Lc)'},
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
               "PS = {'gamma1': 1.0, 'gamma2': 1.0, 'eta1': -100.0, 'eta2': -100.0}\n",
      'call': 'pfc_free_energy(phi1, phi2, Mf, Lc, PS, Hf)',
      'gold_call': '_oracle_pfc_free_energy(phi1, phi2, Mf, Lc, PS, Hf)'},
     {'setup': 'import numpy as np\n'
               'x = np.arange(20) * (24.0 / 20)\n'
               'y = np.arange(16) * (16.0 / 16)\n'
               "X, Y = np.meshgrid(x, y, indexing='ij')\n"
               'phi1 = 0.5 * np.sin(2 * np.pi * X / 24.0)\n'
               'phi2 = 0.4 * np.cos(4 * np.pi * Y / 16.0) + 0.2\n'
               'Mf = np.stack([0.3 + 0 * X, 0.2 * np.sin(2 * np.pi * Y / 16.0)])\n'
               "PA = {'a1': 0.8, 'a2': 1.3, 'a12': 0.9}\n",
      'call': 'pfc_free_energy(phi1, phi2, Mf, (24.0, 16.0), PA)',
      'gold_call': '_oracle_pfc_free_energy(phi1, phi2, Mf, (24.0, 16.0), PA)'},
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
               'phi1 = -np.abs(phi1) - 0.2\n'
               'phi2 = -np.abs(phi2) - 0.3\n',
      'call': 'pfc_free_energy(phi1, phi2, Mf, Lc, None, Hf)',
      'gold_call': '_oracle_pfc_free_energy(phi1, phi2, Mf, Lc, None, Hf)'},
     {'setup': 'import numpy as np\n'
               'phi1 = np.full((12, 12), 0.35)\n'
               'phi2 = np.full((12, 12), -0.20)\n'
               'Mf = np.stack([np.full((12, 12), 0.6), np.full((12, 12), -0.4)])\n',
      'call': 'pfc_free_energy(phi1, phi2, Mf, 18.0)',
      'gold_call': '_oracle_pfc_free_energy(phi1, phi2, Mf, 18.0)'},
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
      'call': '_probe(pfc_free_energy)',
      'gold_call': '_probe(_oracle_pfc_free_energy)'},
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
      'call': '_probe(pfc_free_energy)',
      'gold_call': '_probe(_oracle_pfc_free_energy)'}]
