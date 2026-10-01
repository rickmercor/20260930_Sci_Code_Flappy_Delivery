"""
Evaluate the ****nonlinear part**** $N$ of the free-energy density - everything except the three blocks that are quadratic in derivatives of the fields - together with its three variational derivatives $N_1=\\delta N/\\delta\\phi_1$, $N_2=\\delta N/\\delta\\phi_2$ and $\\boldsymbol N_3=\\delta N/\\delta\\boldsymbol M$. Return them stacked along a new leading axis in the order $[N,\\ N_1,\\ N_2,\\ N_{3x},\\ N_{3y}]$, so that the shape is `(5,) + phi1.shape`. These four objects are what the quadratization of the next step is built from: $N$ sits under the square root and the three derivatives sit in the numerators of the three coefficient functions.

$N$ collects the double wells, the vacancy penalizations, the interspecies coupling, the whole Ginzburg-Landau block except its exchange gradient term, and the whole regularization block. Deriving its three variational derivatives is the work of this step, and two features make it more than routine differentiation. First, two of the terms depend on a density through $\\nabla\\phi_i$ rather than through $\\phi_i$ itself, so their variational derivatives are not pointwise derivatives; the magneto-elastic term in particular contributes to $N_i$ and to $\\boldsymbol N_3$ at once, the same term seen from its two sides, which is a useful check on your algebra. Second, the vacancy penalization is the one place where the naive derivative is wrong: its density is not smooth at zero, yet its variational derivative is continuous there and vanishes identically wherever the density is non-negative, which is exactly the behaviour that keeps the densities positive. Differentiating $|\\phi|^3$ as though it were $\\phi^3$ gets the non-negative side right and the negative side wrong.

****--- Formulas ---****

$N$ is what is left of the energy density of the problem statement after removing the three blocks that are quadratic in derivatives of the fields: the two terms $\\tfrac{\\phi_i}{2}(\\Delta+a_i)^2\\phi_i$, the interspecies term, and the exchange term $\\tfrac{\\omega_0}{2}|\\nabla\\boldsymbol M|^2$. Everything else stays in $N$. Derive $N_1$, $N_2$ and $\\boldsymbol N_3$ from that $N$ yourself: they are the variational derivatives of $\\int_\\Omega N$, so a term entering only through $\\nabla\\phi_i$ contributes $-\\nabla\\!\\cdot\\!\\big(\\partial N/\\partial\\nabla\\phi_i\\big)$ and not $\\partial N/\\partial\\phi_i$. Fixed here: every gradient and every divergence is spectral as in step 1, the divergence of $(v_x,v_y)$ being $\\partial_xv_x+\\partial_yv_y$; and the return order is $[N,\\ N_1,\\ N_2,\\ N_{3x},\\ N_{3y}]$.

Returns
-------
`np.ndarray` of shape `(5,) + phi1.shape`, real and finite. On a uniform state every divergence vanishes and the five entries reduce to closed-form polynomials in the densities and the magnetization.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def pfc_nonlinear_terms(phi1, phi2, M, cell, params=None, H=None):
    """Arguments exactly as in pfc_free_energy.
    Return the real array [N, dN/dphi1, dN/dphi2, (dN/dM)_x, (dN/dM)_y]
    of shape (5,) + phi1.shape.
    Raise ValueError for mismatched or non-finite fields, invalid domain
    lengths, or invalid model parameters. H=None means a zero applied field."""
    # Implement per the specification above.
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_pfc_nonlinear_terms(phi1, phi2, M, cell, params=None, H=None):
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

    g1 = _oracle_pfc_spectral_derivatives(phi1, L)[:2]
    g2 = _oracle_pfc_spectral_derivatives(phi2, L)[:2]
    m2 = M[0] ** 2 + M[1] ** 2
    mg1 = M[0] * g1[0] + M[1] * g1[1]
    mg2 = M[0] * g2[0] + M[1] * g2[1]
    s1 = g1[0] ** 2 + g1[1] ** 2
    s2 = g2[0] ** 2 + g2[1] ** 2

    def div(vx, vy):
        return (_oracle_pfc_spectral_derivatives(vx, L)[0]
                + _oracle_pfc_spectral_derivatives(vy, L)[1])

    N = (0.25 * phi1 ** 4 - 0.5 * p["eps"] * phi1 ** 2
         + 0.25 * phi2 ** 4 - 0.5 * p["eps"] * phi2 ** 2
         + p["eta"] / 3.0 * (np.abs(phi1) ** 3 + np.abs(phi2) ** 3
                             - phi1 ** 3 - phi2 ** 3)
         + 0.5 * p["gamma12"] * phi1 ** 2 * phi2 ** 2
         - 0.5 * p["alpha"] * m2 + 0.25 * p["beta"] * m2 ** 2
         - (M[0] * H[0] + M[1] * H[1])
         - p["gamma1"] * m2 * phi1 - p["gamma2"] * m2 * phi2
         - 0.5 * p["eta1"] * mg1 ** 2 - 0.5 * p["eta2"] * mg2 ** 2
         + p["theta"] / 6.0 * m2 ** 3
         + 0.25 * p["theta1"] * s1 ** 2 + 0.25 * p["theta2"] * s2 ** 2)

    N1 = (phi1 ** 3 - p["eps"] * phi1
          + p["eta"] * (np.abs(phi1) - phi1) * phi1
          + p["gamma12"] * phi1 * phi2 ** 2
          + p["eta1"] * div(mg1 * M[0], mg1 * M[1])
          - p["gamma1"] * m2
          - p["theta1"] * div(s1 * g1[0], s1 * g1[1]))
    N2 = (phi2 ** 3 - p["eps"] * phi2
          + p["eta"] * (np.abs(phi2) - phi2) * phi2
          + p["gamma12"] * phi1 ** 2 * phi2
          + p["eta2"] * div(mg2 * M[0], mg2 * M[1])
          - p["gamma2"] * m2
          - p["theta2"] * div(s2 * g2[0], s2 * g2[1]))
    N3 = (-p["alpha"] * M + p["beta"] * m2 * M - H
          - 2.0 * p["gamma1"] * M * phi1 - 2.0 * p["gamma2"] * M * phi2
          - p["eta1"] * mg1 * g1 - p["eta2"] * mg2 * g2
          + p["theta"] * m2 ** 2 * M)
    return np.stack([N, N1, N2, N3[0], N3[1]])

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
      'call': 'pfc_nonlinear_terms(phi1, phi2, Mf, Lc, None, Hf)',
      'gold_call': '_oracle_pfc_nonlinear_terms(phi1, phi2, Mf, Lc, None, Hf)'},
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
      'call': 'pfc_nonlinear_terms(phi1, phi2, Mf, Lc, PS, Hf)',
      'gold_call': '_oracle_pfc_nonlinear_terms(phi1, phi2, Mf, Lc, PS, Hf)'},
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
               "PR = {'theta': 0.5, 'theta1': 2.0, 'theta2': 3.0}\n",
      'call': 'pfc_nonlinear_terms(phi1, phi2, Mf, Lc, PR, Hf)',
      'gold_call': '_oracle_pfc_nonlinear_terms(phi1, phi2, Mf, Lc, PR, Hf)'},
     {'setup': 'import numpy as np\n'
               'x = np.arange(20) * (24.0 / 20)\n'
               'y = np.arange(16) * (16.0 / 16)\n'
               "X, Y = np.meshgrid(x, y, indexing='ij')\n"
               'phi1 = 0.6 * np.sin(2 * np.pi * X / 24.0) + 0.3\n'
               'phi2 = 0.2 * np.cos(4 * np.pi * Y / 16.0)\n'
               'Mf = np.stack([0.3 * np.cos(2 * np.pi * X / 24.0), 0.5 + 0 * X])\n'
               'Hf = np.stack([0.1 + 0 * X, -0.2 + 0 * X])\n'
               "PG = {'gamma12': 2.0, 'gamma1': 0.3, 'gamma2': -0.4}\n",
      'call': 'pfc_nonlinear_terms(phi1, phi2, Mf, (24.0, 16.0), PG, Hf)',
      'gold_call': '_oracle_pfc_nonlinear_terms(phi1, phi2, Mf, (24.0, 16.0), PG, Hf)'},
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
               'phi1 = phi1 - 0.10\n'
               'phi2 = phi2 + 0.05\n',
      'call': 'pfc_nonlinear_terms(phi1, phi2, Mf, Lc, None, Hf)',
      'gold_call': '_oracle_pfc_nonlinear_terms(phi1, phi2, Mf, Lc, None, Hf)'},
     {'setup': 'import numpy as np\n'
               'phi1 = np.full((12, 12), -0.45)\n'
               'phi2 = np.full((12, 12), 0.25)\n'
               'Mf = np.stack([np.full((12, 12), 0.7), np.full((12, 12), 0.2)])\n'
               'Hf = np.stack([np.full((12, 12), 0.3), np.zeros((12, 12))])\n',
      'call': 'pfc_nonlinear_terms(phi1, phi2, Mf, 18.0, None, Hf)',
      'gold_call': '_oracle_pfc_nonlinear_terms(phi1, phi2, Mf, 18.0, None, Hf)'},
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
      'call': '_probe(pfc_nonlinear_terms)',
      'gold_call': '_probe(_oracle_pfc_nonlinear_terms)'},
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
      'call': '_probe(pfc_nonlinear_terms)',
      'gold_call': '_probe(_oracle_pfc_nonlinear_terms)'}]
