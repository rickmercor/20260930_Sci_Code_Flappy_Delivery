"""
Evaluate the ****modified discrete energy**** that the scheme dissipates - the quantity the source's discrete energy law bounds, which is the original free energy with its nonlinear part replaced by the squared norm of the auxiliary variable, plus the two quadratic stabilizer terms, plus the square of the nonlocal scalar, minus the two constants those replacements introduce, plus the two gradient-difference terms that the second-order time discretisation contributes. Return one float. The previous-layer densities are optional and default to the current ones, in which case the two gradient-difference terms vanish; that default is what the initial layer needs. ****Every squared gradient norm in this expression - the two $\\|\\nabla\\phi_i\\|^2$, the $\\|\\nabla\\boldsymbol M\\|^2$ and the two gradient-difference terms - is to be evaluated as $-(u,\\Delta u)$, not as the integral of the squared pointwise first derivatives.**** The two agree in the continuum and differ on the grid, and the choice is graded.

Quadratization buys unconditional stability at the price of dissipating a **modified** energy rather than the original one. At $t=0$ the auxiliary variable is the exact square root, so $\\|U^0\\|^2$ equals the integral of the nonlinear part plus $B|\\Omega|$, and the modified energy coincides with the original free energy to round-off; that agreement is the sharpest single check available on both expressions. As the run proceeds the auxiliary variable, which now obeys its own evolution equation, drifts away from the square root, and the two energies separate by an amount that measures the quadratization error. The identity is worth reading carefully in one place: subtracting $B|\\Omega|$ from $\\|U\\|^2$ is a cancellation of two numbers of order $B|\\Omega|$, which for $B=10^{7}$ on a domain of area $4096$ is $4.096\\times10^{10}$, so the result carries about ten significant figures rather than sixteen - a fact worth knowing before comparing the two energies. The $\\tfrac12|Q|^2-\\tfrac12$ pair is there for the same reason: it is zero at the exact value $Q=1$ and contributes only through the discrete drift. The gradient-norm convention is not cosmetic. A squared gradient norm has two discrete readings that agree in the continuum, $\\sum_i\\|\\partial_iu\\|^2$ and $-(u,\\Delta u)$, and on the grid they differ by exactly the Nyquist content of $u$, because the spectral first-derivative operator annihilates that mode of a real field while the Laplacian keeps it. The second reading is the one this energy needs, because the scheme itself is a Fourier-Galerkin discretisation in which the term carrying $\\Delta\\phi_i^{*}$ and the term carrying $(\\nabla\\phi_i^{*},\\nabla\\chi)$ have the same symbol at every mode; with that reading the source's discrete energy law holds as an exact identity, and with the other it is violated at the $10^{-4}$ level. The initial data carry no Nyquist content, so the two readings coincide there and the agreement with the original free energy is unaffected.

****--- Formulas ---****

The expression is the source's modified discrete energy - its Eq. (3.57) - and its terms and their coefficients are to be taken from there. In words, so that the interface is unambiguous: the original free energy with its nonlinear part replaced by the squared norm of the auxiliary variable, the two stabilizer quadratics added back, the square of the nonlocal scalar added with its own coefficient, the two constants that those replacements introduce subtracted, and the two gradient-difference terms in $\\phi_i^{n+1}-\\phi_i^{n}$ that the second-order time discretization contributes.

Fixed here, and graded: $|\\Omega|=L_xL_y$; every inner product and every norm is the grid sum times $h_xh_y$; the previous-layer densities default to the current ones, which makes the two gradient-difference terms vanish; and ****every squared gradient norm in the expression - the two $\\|\\nabla\\phi_i\\|^2$, the $\\|\\nabla\\boldsymbol M\\|^2$ and the two gradient-difference terms - is $\\|\\nabla u\\|^2:=-(u,\\Delta u)$****, not the integral of the squared pointwise first derivatives, so that $\\|\\nabla\\boldsymbol M\\|^2=-\\sum_j(M_j,\\Delta M_j)$. That last convention is not in the source and is graded: the two readings differ on the grid.

Returns
-------
`float`, finite. On the initial layer, where the auxiliary variable is the exact square root and the nonlocal scalar is one, it agrees with `pfc_free_energy` on the same data to about nine significant figures, the residual being the double-precision cancellation of $B|\\Omega|$ against $\\|U\\|^2$.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def pfc_modified_energy(phi1, phi2, M, U, q, cell, params=None,
                        phi1_prev=None, phi2_prev=None):
    """phi1, phi2, U: fields of shape (Nx, Ny). M: shape (2, Nx, Ny).
    q: the nonlocal scalar, a float. cell: domain edge lengths.
    params: as in pfc_free_energy.
    phi1_prev, phi2_prev: the previous-layer densities; None means equal to
       the current ones, so the two gradient-difference terms vanish.
    Return the modified discrete energy, a float.
    Raise ValueError for mismatched or non-finite fields (including optional
    previous densities), a non-finite or non-scalar q, invalid domain lengths,
    or invalid model parameters."""
    # Implement per the specification above.
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_pfc_modified_energy(phi1, phi2, M, U, q, cell, params=None,
                                phi1_prev=None, phi2_prev=None):
    p = _check_params(params)
    phi1 = np.asarray(phi1, float)
    phi2 = np.asarray(phi2, float)
    M = np.asarray(M, float)
    U = np.asarray(U, float)
    if phi1.ndim != 2 or phi2.shape != phi1.shape or U.shape != phi1.shape:
        raise ValueError("phi1, phi2 and U must be two-dimensional fields of equal shape")
    if M.shape != (2,) + phi1.shape:
        raise ValueError("M must have shape (2,) + phi1.shape")
    L = _check_cell(cell, phi1.shape)
    try:
        if np.ndim(q) != 0:
            raise ValueError("q must be a finite scalar")
        q = float(q)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("q must be a finite scalar") from exc
    if not np.isfinite(q):
        raise ValueError("q must be a finite scalar")
    p1p = phi1 if phi1_prev is None else np.asarray(phi1_prev, float)
    p2p = phi2 if phi2_prev is None else np.asarray(phi2_prev, float)
    if p1p.shape != phi1.shape or p2p.shape != phi1.shape:
        raise ValueError("the previous-layer fields must match phi1.shape")
    if not all(np.all(np.isfinite(v)) for v in (phi1, phi2, M, U, p1p, p2p)):
        raise ValueError("all current and previous fields must be finite")

    def gsq(u):
        """Squared gradient norm in the Galerkin sense, -(u, Laplacian u)."""
        return -_ip(u, _oracle_pfc_spectral_derivatives(u, L)[2], L)

    d1 = _oracle_pfc_spectral_derivatives(phi1, L, p["a12"])
    d2 = _oracle_pfc_spectral_derivatives(phi2, L, p["a12"])

    E = (0.5 * _ip(d1[2], d1[2], L) + 0.5 * _ip(d2[2], d2[2], L)
         + 0.5 * _ip(d1[4], phi2, L)
         - p["a1"] * gsq(phi1) - p["a2"] * gsq(phi2)
         + 0.5 * p["S_phi"] * _ip(phi1, phi1, L)
         + 0.5 * p["S_phi"] * _ip(phi2, phi2, L)
         + 0.5 * p["omega0"] * (gsq(M[0]) + gsq(M[1]))
         + 0.5 * p["S_m"] * (_ip(M[0], M[0], L) + _ip(M[1], M[1], L))
         + _ip(U, U, L) + 0.5 * q ** 2 - p["B"] * float(np.prod(L)) - 0.5
         + 0.5 * p["a1"] * gsq(phi1 - p1p) + 0.5 * p["a2"] * gsq(phi2 - p2p))
    return float(E)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Original cases plus declared invalid-input contract checks."""
    return [{'setup': 'import numpy as np\n'
               "PE = {'B': 100.0}\n"
               'St, Hf = _pfc_initial_state((24, 24), (24.0, 24.0), 2.0e-3, PE)\n'
               'U0 = St[0][8]\n',
      'call': 'pfc_modified_energy(St[0][0], St[0][1], St[0][2:4], U0, 1.0, (24.0, 24.0), PE)',
      'gold_call': '_oracle_pfc_modified_energy(St[0][0], St[0][1], St[0][2:4], U0, 1.0, (24.0, 24.0), '
                   'PE)'},
     {'setup': 'import numpy as np\n'
               "PE = {'B': 100.0}\n"
               'St, Hf = _pfc_initial_state((24, 24), (24.0, 24.0), 2.0e-3, PE)\n'
               'q = 1.0\n'
               'pr1, pr2 = St[0][0], St[0][1]\n'
               'o1 = _oracle_pfc_time_step(St, q, (24.0, 24.0), 2.0e-3, PE, Hf)\n'
               'q1 = float(o1[9].flat[0]); pr1_1, pr2_1 = St[0][0], St[0][1]; St1 = np.stack([o1[:9], St[0], St[1]])\n'
               'o2 = _oracle_pfc_time_step(St1, q1, (24.0, 24.0), 2.0e-3, PE, Hf)\n'
               'q2 = float(o2[9].flat[0]); pr1_2, pr2_2 = St1[0][0], St1[0][1]; St2 = np.stack([o2[:9], St1[0], St1[1]])\n'
               'o3 = _oracle_pfc_time_step(St2, q2, (24.0, 24.0), 2.0e-3, PE, Hf)\n'
               'q = float(o3[9].flat[0]); pr1, pr2 = St2[0][0], St2[0][1]; St = np.stack([o3[:9], St2[0], St2[1]])\n',
      'call': 'pfc_modified_energy(St[0][0], St[0][1], St[0][2:4], St[0][8], q, (24.0, 24.0), PE, pr1, '
              'pr2)',
      'gold_call': '_oracle_pfc_modified_energy(St[0][0], St[0][1], St[0][2:4], St[0][8], q, (24.0, '
                   '24.0), PE, pr1, pr2)'},
     {'setup': 'import numpy as np\n'
               "PE = {'B': 100.0}\n"
               'St, Hf = _pfc_initial_state((24, 24), (24.0, 24.0), 2.0e-3, PE)\n'
               'U0 = St[0][8]\n',
      'call': 'pfc_modified_energy(St[0][0], St[0][1], St[0][2:4], U0, -3.0, (24.0, 24.0), PE)',
      'gold_call': '_oracle_pfc_modified_energy(St[0][0], St[0][1], St[0][2:4], U0, -3.0, (24.0, '
                   '24.0), PE)'},
     {'setup': 'import numpy as np\n'
               "PB = {'B': 60.0}\n"
               'St, Hf = _pfc_initial_state((16, 24), (16.0, 24.0), 1.0e-3, PB)\n',
      'call': 'pfc_modified_energy(St[0][0], St[0][1], St[0][2:4], St[0][8], 1.0, (16.0, 24.0), PB)',
      'gold_call': '_oracle_pfc_modified_energy(St[0][0], St[0][1], St[0][2:4], St[0][8], 1.0, (16.0, '
                   '24.0), PB)'},
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
               "PA = {'a1': 0.6, 'a2': 1.7, 'B': 9.0}\n"
               'Uu = np.full_like(phi1, 3.0) + 0.2 * phi2\n'
               'q1 = 0.8 * phi1\n'
               'q2 = 1.4 * phi2\n',
      'call': 'pfc_modified_energy(phi1, phi2, Mf, Uu, 1.0, Lc, PA, q1, q2)',
      'gold_call': '_oracle_pfc_modified_energy(phi1, phi2, Mf, Uu, 1.0, Lc, PA, q1, q2)'},
     {'setup': 'import numpy as np\n'
               'phi1 = np.full((12, 12), 0.35)\n'
               'phi2 = np.full((12, 12), -0.20)\n'
               'Mf = np.stack([np.full((12, 12), 0.6), np.full((12, 12), -0.4)])\n'
               'Uu = np.full((12, 12), 2.0)\n'
               "PB = {'B': 3.0}\n",
      'call': 'pfc_modified_energy(phi1, phi2, Mf, Uu, 1.0, 18.0, PB)',
      'gold_call': '_oracle_pfc_modified_energy(phi1, phi2, Mf, Uu, 1.0, 18.0, PB)'},
     {'setup': 'import numpy as np\n'
               'xg = np.arange(16) * 1.0\n'
               'yg = np.arange(16) * 1.0\n'
               "X, Y = np.meshgrid(xg, yg, indexing='ij')\n"
               'phi1 = 0.6 * np.cos(np.pi * X) + 0.3 * np.sin(2 * np.pi * Y / 16.0)\n'
               'phi2 = 0.5 * np.cos(np.pi * Y) * np.cos(np.pi * X) - 0.2\n'
               'Mf = np.stack([0.4 * np.cos(np.pi * X), 0.3 * np.sin(np.pi * Y)])\n'
               'Uu = np.full_like(phi1, 4.0)\n'
               'q1 = 0.5 * phi1\n'
               'q2 = np.zeros_like(phi2)\n'
               "PB = {'B': 40.0}\n",
      'call': 'pfc_modified_energy(phi1, phi2, Mf, Uu, 1.0, 16.0, PB, q1, q2)',
      'gold_call': '_oracle_pfc_modified_energy(phi1, phi2, Mf, Uu, 1.0, 16.0, PB, q1, q2)'},
     {'setup': 'import numpy as np\n'
               'z = np.zeros((4, 4))\n'
               'M = np.zeros((2, 4, 4))\n'
               'U = np.full_like(z, np.nan)\n'
               'def _probe(f):\n'
               '    try:\n'
               '        f(z, z, M, U, 1.0, (4.0, 4.0))\n'
               '    except ValueError:\n'
               '        return 1.0\n'
               '    except Exception:\n'
               '        return 2.0\n'
               '    return 0.0\n',
      'call': '_probe(pfc_modified_energy)',
      'gold_call': '_probe(_oracle_pfc_modified_energy)'},
     {'setup': 'import numpy as np\n'
               'z = np.zeros((4, 4))\n'
               'M = np.zeros((2, 4, 4))\n'
               'def _probe(f):\n'
               '    try:\n'
               '        f(z, z, M, z, np.nan, (4.0, 4.0))\n'
               '    except ValueError:\n'
               '        return 1.0\n'
               '    except Exception:\n'
               '        return 2.0\n'
               '    return 0.0\n',
      'call': '_probe(pfc_modified_energy)',
      'gold_call': '_probe(_oracle_pfc_modified_energy)'},
     {'setup': 'import numpy as np\n'
               'z = np.zeros((4, 4))\n'
               'M = np.zeros((2, 4, 4))\n'
               'U = np.full_like(z, np.inf)\n'
               'def _probe(f):\n'
               '    try:\n'
               '        f(z, z, M, U, 1.0, (4.0, 4.0))\n'
               '    except ValueError:\n'
               '        return 1.0\n'
               '    except Exception:\n'
               '        return 2.0\n'
               '    return 0.0\n',
      'call': '_probe(pfc_modified_energy)',
      'gold_call': '_probe(_oracle_pfc_modified_energy)'},
     {'setup': 'import numpy as np\n'
               'z = np.zeros((4, 4))\n'
               'M = np.zeros((2, 4, 4))\n'
               'def _probe(f):\n'
               '    try:\n'
               '        f(z, z, M, z, np.inf, (4.0, 4.0))\n'
               '    except ValueError:\n'
               '        return 1.0\n'
               '    except Exception:\n'
               '        return 2.0\n'
               '    return 0.0\n',
      'call': '_probe(pfc_modified_energy)',
      'gold_call': '_probe(_oracle_pfc_modified_energy)'}]
