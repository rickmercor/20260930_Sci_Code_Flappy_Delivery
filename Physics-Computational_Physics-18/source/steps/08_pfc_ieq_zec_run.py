"""
Run the whole benchmark. Build the initial data and the applied field from the closed form below; evaluate the auxiliary variable and the three coefficient functions on them; assemble the initial chemical potentials of the reformulated system with the nonlocal scalar at its initial value $Q^0=1$; build the two backward ghost layers from the exact initial velocities so that the second-order formulas apply from the very first step; advance $n_steps$ steps of size $dt$ with the step of step 6; and return the ****total free energy of the final layer divided by the area of the domain****. Setting `quantity` selects a diagnostic instead: `'energy0'` the initial free-energy density, `'emod'` and `'emod0'` the modified discrete energy density at the final and the initial layer, `'q'` the final nonlocal scalar, `'mass'` the total drift of the two spatial means over the run, and `'drift'` the sup-norm distance between the auxiliary variable carried by the scheme and the square root recomputed from the final fields.

This step is the whole workflow: the closed-form initial data and applied field, the consistent three-layer startup, the time loop of step 6, and the two energies of steps 2 and 7 wrapped around it. Three structural points govern it. First, the startup: the scheme reaches back two layers, and filling those layers with copies of the initial one makes the derivative extrapolation vanish, which is a first-order error that costs $0.0415$ in the answer; building them from the exact initial velocities instead - $\\psi^{-k}=\\psi^0-k\\,\\delta t\\,\\partial_t\\psi(0)$ - makes the same second-order formulas correct from the first step and needs the governing system's right-hand sides and the auxiliary variable's own evolution equation, both evaluated on the initial data. Second, the diagnostics are not decoration: the modified discrete energy must equal the original free energy at $t=0$, both energies must decrease at every step, and the two masses must be conserved to round-off - three checks a correct implementation gets for free and a wrong one fails visibly. Third, the nonlocal scalar is exactly one at the continuous level, so its computed value is a direct read-out of the scheme's consistency error, growing with the time step and shrinking under refinement.

****--- Formulas ---****

On $\\Omega=[0,L_x]\\times[0,L_y]$ with an $N_x\\times N_y$ grid, $x_i=iL_x/N_x$, $y_j=jL_y/N_y$ in `numpy.meshgrid(..., indexing='ij')` order and $\\kappa_x=2\\pi/L_x$, $\\kappa_y=2\\pi/L_y$, the instance of the problem statement is

$$\\phi_1^{0}=\\cos(8\\kappa_xx)\\sin(8\\kappa_yy),\\qquad \\phi_2^{0}=\\cos(8\\kappa_xx)\\cos(8\\kappa_yy),$$

$$\\boldsymbol M^{0}=\\big(\\sin(2\\kappa_xx)\\sin(2\\kappa_yy),\\ \\cos(2\\kappa_xx)\\cos(2\\kappa_yy)\\big),\\qquad \\boldsymbol H=\\big(\\sin(2\\kappa_xx)\\cos(\\kappa_yy),\\ \\cos(\\kappa_yy)\\big).$$

The rest is the source's. The initial chemical potentials are its ****reformulated**** ones - not the variational derivatives of the original energy - evaluated on that data with the nonlocal scalar at its initial value $Q^0=1$; fetch them. The initial velocities are the right-hand sides of the governing system of the problem statement evaluated on the initial data, together with the auxiliary variable's own evolution equation, which is likewise the source's. Fixed here, and graded: the two backward layers are $\\psi^{-k}=\\psi^{0}-k\\,\\delta t\\,\\partial_t\\psi(0)$ for $k=1,2$, applied to $\\phi_1,\\phi_2,\\boldsymbol M$ and the auxiliary variable, the ghost potentials never being referenced; $Q^0=1$; the loop is $n_steps$ calls of step 6; and the returned quantity is divided by $L_xL_y$, so that

$$\\text{answer}=\\frac{E\\big(\\phi_1^{N},\\phi_2^{N},\\boldsymbol M^{N}\\big)}{L_xL_y}.$$

Returns
-------
`float`: the free-energy density at the final time. On the production instance it decreases monotonically along the run; it equals the `'emod'` value at `n_steps=0` to about nine significant figures for every grid and parameter set, since the initial auxiliary variable is the exact square root; and the `'mass'` diagnostic is of order $10^{-17}$ for every time step, grid and parameter set.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def pfc_ieq_zec_run(n_steps=10, dt=2.0e-3, grid=(64, 64), cell=(64.0, 64.0), params=None, quantity="energy"):
    """n_steps: number of time steps. dt: the time step.
    grid: (Nx, Ny). cell: domain edge lengths (Lx, Ly).
    params: dict of model-parameter overrides; None means the fixed set.
    quantity: 'energy' (default, the final free-energy density), 'energy0',
       'emod', 'emod0', 'q', 'mass' or 'drift'.
    Return the requested scalar, a float.
    Raise ValueError if n_steps is not a non-negative integer, if dt is
    not a finite positive scalar, if either grid dimension is below four,
    or if quantity is not one of the names listed above.
    Integer-valued real scalars are accepted for n_steps and grid dimensions;
    other values, missing dimensions and non-finite values raise ValueError."""
    # Implement per the specification above.
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_pfc_ieq_zec_run(n_steps=10, dt=2.0e-3, grid=(64, 64), cell=(64.0, 64.0), params=None, quantity="energy"):
    try:
        if np.ndim(n_steps) != 0:
            raise ValueError("n_steps must be a non-negative integer")
        if isinstance(n_steps, (int, np.integer)):
            n_steps = int(n_steps)
        else:
            value = float(n_steps)
            if not np.isfinite(value) or not value.is_integer():
                raise ValueError("n_steps must be a non-negative integer")
            n_steps = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("n_steps must be a non-negative integer") from exc
    if n_steps < 0:
        raise ValueError("n_steps must be a non-negative integer")
    try:
        if np.ndim(dt) != 0:
            raise ValueError("dt must be a finite positive scalar")
        dt = float(dt)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("dt must be a finite positive scalar") from exc
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError("dt must be a finite positive scalar")
    if str(quantity) not in ("energy", "energy0", "emod", "emod0", "q",
                             "mass", "drift"):
        raise ValueError("quantity must be one of 'energy', 'energy0', 'emod', "
                         "'emod0', 'q', 'mass', 'drift'")
    try:
        dimensions = np.asarray(grid, dtype=float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("grid must contain two finite integer dimensions") from exc
    if (dimensions.shape != (2,) or not np.all(np.isfinite(dimensions))
            or np.any(dimensions < 4) or np.any(dimensions != np.floor(dimensions))):
        raise ValueError("grid must contain two integer dimensions of at least four")
    grid = (int(dimensions[0]), int(dimensions[1]))
    L = _check_cell(cell, grid)

    S, Hf = _pfc_initial_state(grid, L, dt, params)
    phi1, phi2 = S[0][0].copy(), S[0][1].copy()
    q = 1.0
    area = float(np.prod(L))
    if str(quantity) == "energy0":
        return _oracle_pfc_free_energy(phi1, phi2, S[0][2:4], L, params, Hf) / area
    if str(quantity) == "emod0":
        return _oracle_pfc_modified_energy(phi1, phi2, S[0][2:4], S[0][8], q, L,
                                           params) / area

    prev1, prev2 = phi1, phi2
    for _ in range(int(n_steps)):
        out = _oracle_pfc_time_step(S, q, L, dt, params, Hf)
        q = float(out[9].flat[0])
        prev1, prev2 = S[0][0], S[0][1]
        S = np.stack([out[:9], S[0], S[1]])

    f1, f2 = S[0][0], S[0][1]
    Mf, Uf = S[0][2:4], S[0][8]
    if str(quantity) == "emod":
        return _oracle_pfc_modified_energy(f1, f2, Mf, Uf, q, L, params,
                                           prev1, prev2) / area
    if str(quantity) == "q":
        return float(q)
    if str(quantity) == "mass":
        return float(abs(np.mean(f1) - np.mean(phi1))
                     + abs(np.mean(f2) - np.mean(phi2)))
    if str(quantity) == "drift":
        Ue = _oracle_pfc_ieq_coefficients(f1, f2, Mf, L, params, Hf)[0]
        return float(np.max(np.abs(Uf - Ue)))
    return _oracle_pfc_free_energy(f1, f2, Mf, L, params, Hf) / area

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Original cases plus declared invalid-input contract checks."""
    return [{'setup': 'import numpy as np\n',
      'call': 'pfc_ieq_zec_run()',
      'gold_call': '_oracle_pfc_ieq_zec_run()'},
     {'setup': 'import numpy as np\n',
      'call': 'pfc_ieq_zec_run(n_steps=0)',
      'gold_call': '_oracle_pfc_ieq_zec_run(n_steps=0)'},
     {'setup': 'import numpy as np\n',
      'call': "pfc_ieq_zec_run(quantity='emod')",
      'gold_call': "_oracle_pfc_ieq_zec_run(quantity='emod')"},
     {'setup': 'import numpy as np\n',
      'call': "pfc_ieq_zec_run(quantity='q')",
      'gold_call': "_oracle_pfc_ieq_zec_run(quantity='q')"},
     {'setup': 'import numpy as np\n',
      'call': 'pfc_ieq_zec_run(n_steps=5)',
      'gold_call': '_oracle_pfc_ieq_zec_run(n_steps=5)'},
     {'setup': 'import numpy as np\n',
      'call': 'pfc_ieq_zec_run(n_steps=20, dt=1.0e-3)',
      'gold_call': '_oracle_pfc_ieq_zec_run(n_steps=20, dt=1.0e-3)'},
     {'setup': 'import numpy as np\n',
      'call': 'pfc_ieq_zec_run(n_steps=4, grid=(48, 64), cell=(48.0, 64.0))',
      'gold_call': '_oracle_pfc_ieq_zec_run(n_steps=4, grid=(48, 64), cell=(48.0, 64.0))'},
     {'setup': 'import numpy as np\n'
               "PS = {'gamma1': 1.0, 'gamma2': 1.0, 'eta1': -100.0, 'eta2': -100.0}\n",
      'call': 'pfc_ieq_zec_run(n_steps=5, dt=1.0e-3, params=PS)',
      'gold_call': '_oracle_pfc_ieq_zec_run(n_steps=5, dt=1.0e-3, params=PS)'},
     {'setup': 'import numpy as np\n',
      'call': "pfc_ieq_zec_run(n_steps=6, dt=5.0e-3, quantity='mass')",
      'gold_call': "_oracle_pfc_ieq_zec_run(n_steps=6, dt=5.0e-3, quantity='mass')"},
     {'setup': 'import numpy as np\n'
               'def _probe(f):\n'
               '    try:\n'
               '        f(n_steps=None)\n'
               '    except ValueError:\n'
               '        return 1.0\n'
               '    except Exception:\n'
               '        return 2.0\n'
               '    return 0.0\n',
      'call': '_probe(pfc_ieq_zec_run)',
      'gold_call': '_probe(_oracle_pfc_ieq_zec_run)'},
     {'setup': 'import numpy as np\n'
               'def _probe(f):\n'
               '    try:\n'
               '        f(n_steps=np.nan)\n'
               '    except ValueError:\n'
               '        return 1.0\n'
               '    except Exception:\n'
               '        return 2.0\n'
               '    return 0.0\n',
      'call': '_probe(pfc_ieq_zec_run)',
      'gold_call': '_probe(_oracle_pfc_ieq_zec_run)'},
     {'setup': 'import numpy as np\n'
               'def _probe(f):\n'
               '    try:\n'
               '        f(n_steps=np.inf)\n'
               '    except ValueError:\n'
               '        return 1.0\n'
               '    except Exception:\n'
               '        return 2.0\n'
               '    return 0.0\n',
      'call': '_probe(pfc_ieq_zec_run)',
      'gold_call': '_probe(_oracle_pfc_ieq_zec_run)'},
     {'setup': 'import numpy as np\n'
               'def _probe(f):\n'
               '    try:\n'
               '        f(dt=None)\n'
               '    except ValueError:\n'
               '        return 1.0\n'
               '    except Exception:\n'
               '        return 2.0\n'
               '    return 0.0\n',
      'call': '_probe(pfc_ieq_zec_run)',
      'gold_call': '_probe(_oracle_pfc_ieq_zec_run)'},
     {'setup': 'import numpy as np\n'
               'def _probe(f):\n'
               '    try:\n'
               '        f(dt=np.inf)\n'
               '    except ValueError:\n'
               '        return 1.0\n'
               '    except Exception:\n'
               '        return 2.0\n'
               '    return 0.0\n',
      'call': '_probe(pfc_ieq_zec_run)',
      'gold_call': '_probe(_oracle_pfc_ieq_zec_run)'},
     {'setup': 'import numpy as np\n'
               'def _probe(f):\n'
               '    try:\n'
               '        f(grid=(4.5, 4))\n'
               '    except ValueError:\n'
               '        return 1.0\n'
               '    except Exception:\n'
               '        return 2.0\n'
               '    return 0.0\n',
      'call': '_probe(pfc_ieq_zec_run)',
      'gold_call': '_probe(_oracle_pfc_ieq_zec_run)'},
     {'setup': 'import numpy as np\n'
               'def _probe(f):\n'
               '    try:\n'
               '        f(grid=(4,))\n'
               '    except ValueError:\n'
               '        return 1.0\n'
               '    except Exception:\n'
               '        return 2.0\n'
               '    return 0.0\n',
      'call': '_probe(pfc_ieq_zec_run)',
      'gold_call': '_probe(_oracle_pfc_ieq_zec_run)'}]
