"""
Run the whole benchmark. Build the initial density from the closed form below on the requested grid and box; initialise the auxiliary variable as the exact square root of the shifted nonlinear energy of that initial density; advance $n_steps$ steps of size $dt$ with the step of step 6 and the coefficient family member of step 4. The energy of the final layer is ****assembled from two halves****: its quadratic block is read off the energy identity of step 8, evaluated on the layer the ****last**** step starts from, whose first entry is exactly that block; the nonlinear block, bulk and long-range together, comes from step 2. Return their sum ****divided by the area of the domain****. With $n_steps$ zero no step is taken, there is no identity to read, and the quadratic block is the initial one. Setting `quantity` selects a diagnostic instead: `'energy0'` the initial free-energy density, `'emod'` and `'emod0'` the modified discrete energy density of the final and of the initial layer, `'quad'` the quadratic block of the final layer divided by the area, `'r'` the final auxiliary variable, `'mass'` the relative drift of the spatial mean over the run, `'drift'` the absolute distance between the auxiliary variable carried by the scheme and the square root recomputed from the final density, and `'work'` and `'qform'` the stage-work term and the quadratic-form term that step 8 returns for that last step, each divided by the area and each zero when $n_steps$ is zero.

The initial density is a constant plus two smooth modes, chosen so that the mean is the constant exactly on every grid that resolves the modes, and so that one of the two modes sits inside the band where the symbol of the quadratic block is smaller than the temperature parameter and the other sits outside it, so the two relax at very different rates. The run therefore has genuine dynamics rather than uniform decay, and the free energy falls by more than a third over the prescribed interval. Every quantity offered here is a different projection of the same trajectory, so the set of them pins the pipeline from several sides at once: the initial energy tests the energy functional alone, the mass drift tests the zero mode of the stage solve alone, the auxiliary variable tests the scalar half of the scheme alone, and the modified energy tests the two halves against each other.




****--- Formulas ---****

With $\\kappa_x=2\\pi/L_x$, $\\kappa_y=2\\pi/L_y$, $x_i=i\\,L_x/N_x$ and $y_j=j\\,L_y/N_y$ for $i=0,\\dots,N_x-1$ and $j=0,\\dots,N_y-1$ on a `numpy.meshgrid(..., indexing='ij')` grid, $$\\phi^0(x,y)=0.15+0.30\\cos(6\\kappa_xx)\\cos(4\\kappa_yy)+0.20\\sin(2\\kappa_xx)\\sin(6\\kappa_yy),\\qquad r^0=\\sqrt{E_1(\\phi^0)+C_{SAV}}.$$ The reported default is $E(\\phi^{N})/|\\Omega|$ with $|\\Omega|=L_xL_y$ and $N=$ $n_steps$. The mass diagnostic is $|(\\phi^{N},1)_h-(\\phi^{0},1)_h|/|(\\phi^{0},1)_h|$.

Returns
-------
`float`: the free-energy density at the final time. On the production instance it decreases monotonically along the run; it equals the `'emod'` value at `n_steps=0` to round-off for every grid and parameter set, since the initial auxiliary variable is the exact square root; and the `'mass'` diagnostic is at or below $10^{-15}$ for every step size, grid and parameter set. Assembling the quadratic block from the identity rather than from a second evaluation of the energy functional changes nothing it should change: the two agree, and the `'qform'` diagnostic is never negative, whatever the step size, the grid, the parameters or the member of the coefficient family.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def mpfc_rksav_run(n_steps: int = 16, dt: float = 0.125, grid: tuple = (48, 64),
        cell: "float | tuple" = (32.0, 48.0), params: "dict | None" = None,
        coef: "dict | None" = None, quantity: str = "energy") -> float:
    """n_steps: number of time steps. dt: the time step.
    grid: (Nx, Ny), two integers; this step is two-dimensional.
    cell: domain edge lengths (Lx, Ly), a scalar or a pair.
    params: dict of model-parameter overrides; None means the fixed set.
    coef: dict of overrides for the five free constants of the
       coefficient family; None means this task's prescribed set.
    quantity: 'energy' (default, the final free-energy density),
       'energy0', 'emod', 'emod0', 'quad', 'r', 'mass', 'drift',
       'work' or 'qform'.
    Return the requested scalar, a float.
    Raise ValueError if n_steps is not a non-negative integer, if dt is
    not a finite positive scalar, if grid is not two integers each at
    least four, if quantity is not one of the names listed above, or if
    cell, params or coef is invalid."""
    # Implement per the specification above.
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _mpfc_initial_field(grid, cell):
    L = _check_cell(cell, (0, 0))
    x = np.arange(grid[0]) * (L[0] / grid[0])
    y = np.arange(grid[1]) * (L[1] / grid[1])
    X, Y = np.meshgrid(x, y, indexing="ij")
    kx = 2.0 * np.pi / L[0]
    ky = 2.0 * np.pi / L[1]
    return (0.15 + 0.30 * np.cos(6.0 * kx * X) * np.cos(4.0 * ky * Y)
            + 0.20 * np.sin(2.0 * kx * X) * np.sin(6.0 * ky * Y))


def _oracle_mpfc_rksav_run(n_steps: int = 16, dt: float = 0.125, grid: tuple = (48, 64),
        cell: "float | tuple" = (32.0, 48.0), params: "dict | None" = None,
        coef: "dict | None" = None, quantity: str = "energy") -> float:
    if isinstance(n_steps, bool) or not isinstance(n_steps, (int, np.integer)) or n_steps < 0:
        raise ValueError("n_steps must be a non-negative integer")
    dt = float(dt)
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError("dt must be a finite positive scalar")
    gr = np.atleast_1d(np.asarray(grid)).ravel()
    if gr.dtype.kind not in "buif" or not all(float(n).is_integer() for n in gr):
        raise ValueError("grid must be two integers, each at least four")
    g = tuple(int(n) for n in gr)
    if len(g) != 2 or min(g) < 4:
        raise ValueError("grid must be two integers, each at least four")
    if quantity not in ("energy", "energy0", "emod", "emod0", "quad", "r",
                        "mass", "drift", "work", "qform"):
        raise ValueError("unknown quantity %r" % (quantity,))
    L = _check_cell(cell, g)
    p = _check_params(params)
    c = _check_coef(coef)
    phi = _mpfc_initial_field(g, L)
    e0 = _oracle_mpfc_free_energy(phi, L, params)
    r = float(np.sqrt(e0[2] + p["C_sav"]))
    area = float(L[0] * L[1])
    m0 = _ip(phi, np.ones_like(phi), L)
    emod0 = _oracle_mpfc_modified_energy(phi, r, L, params)[0]
    nst = int(n_steps)
    ident = None
    for i in range(nst):
        # the last step is also taken through the energy identity of step 8,
        # whose first entry IS the quadratic block of the final layer, paired
        # with L, and whose remaining entries are the two terms the energy
        # theorem splits the change in that block into
        if i == nst - 1:
            ident = _oracle_mpfc_energy_identity(phi, r, dt, L, params, c)
        out = _oracle_mpfc_time_step(phi, r, dt, L, params, c)
        phi = out[0]
        r = float(out[1].flat[0])
    ef = _oracle_mpfc_free_energy(phi, L, params)
    # the quadratic half of the reported energy comes from the identity, the
    # nonlinear half from the energy functional; with no step taken there is no
    # identity to read and the quadratic half is the initial one
    e_quad = 0.5 * float(ident[0]) if ident is not None else ef[1]
    work = float(ident[2]) if ident is not None else 0.0
    qform = float(ident[3]) if ident is not None else 0.0
    md = _oracle_mpfc_modified_energy(phi, r, L, params)
    m1 = _ip(phi, np.ones_like(phi), L)
    table = dict(energy=(e_quad + ef[2]) / area, energy0=e0[0] / area,
                 emod=md[0] / area, emod0=emod0 / area, quad=e_quad / area,
                 r=r, mass=abs(m1 - m0) / abs(m0), drift=abs(md[3]),
                 work=work / area, qform=qform / area)
    return float(table[quantity])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of step test specifications (setup/call/gold_call)."""
    PSTIFF = "PS = {'eps': 0.9, 'alpha': 0.05, 'M': 1.0, 'C_sav': 50.0}\n"
    PE15 = "C15 = {'a11t': 1.0, 'a32t': 1.0, 'a33t': 1.0, 'a31': 1.0, 'a43': 1.0}\n"
    return [
        # normal: the production instance, the graded observable.
        {"setup": "import numpy as np\nimport copy\n_dup = copy.deepcopy\n",
         "call": "mpfc_rksav_run()",
         "gold_call": "_oracle_mpfc_rksav_run()"},
        # control: no time stepping at all.  A candidate whose step is a no-op,
        # or who reports the initial energy, returns this value for case 1.
        {"setup": "import numpy as np\nimport copy\n_dup = copy.deepcopy\n",
         "call": "mpfc_rksav_run(n_steps=0)",
         "gold_call": "_oracle_mpfc_rksav_run(n_steps=0)"},
        # control: the MODIFIED discrete energy of the same final layer, which
        # is what a candidate who confuses the two energies returns for case 1.
        {"setup": "import numpy as np\nimport copy\n_dup = copy.deepcopy\n",
         "call": "mpfc_rksav_run(quantity='emod')",
         "gold_call": "_oracle_mpfc_rksav_run(quantity='emod')"},
        # control: the auxiliary variable at the final time, which pins the
        # scalar half of the scheme independently of the field half.
        {"setup": "import numpy as np\nimport copy\n_dup = copy.deepcopy\n",
         "call": "mpfc_rksav_run(quantity='r')",
         "gold_call": "_oracle_mpfc_rksav_run(quantity='r')"},
        # boundary: half the final time at the same step size.
        {"setup": "import numpy as np\nimport copy\n_dup = copy.deepcopy\n",
         "call": "mpfc_rksav_run(n_steps=8)",
         "gold_call": "_oracle_mpfc_rksav_run(n_steps=8)"},
        # boundary: the same final time reached with twice as many half-sized
        # steps - a temporal refinement that moves the answer only slightly.
        {"setup": "import numpy as np\nimport copy\n_dup = copy.deepcopy\n",
         "call": "mpfc_rksav_run(n_steps=32, dt=0.0625)",
         "gold_call": "_oracle_mpfc_rksav_run(n_steps=32, dt=0.0625)"},
        # boundary: a SQUARE box on a square grid, where every hard-coded
        # rectangular factor of the production instance changes at once.
        {"setup": "import numpy as np\nimport copy\n_dup = copy.deepcopy\n",
         "call": "mpfc_rksav_run(n_steps=6, grid=(32, 32), cell=32.0)",
         "gold_call": "_oracle_mpfc_rksav_run(n_steps=6, grid=(32, 32), cell=32.0)"},
        # edge: the source's own displayed tableau instead of the prescribed
        # family member, which is a different third-order scheme.
        {"setup": "import numpy as np\nimport copy\n_dup = copy.deepcopy\n" + PE15,
         "call": "mpfc_rksav_run(n_steps=6, coef=_dup(C15))",
         "gold_call": "_oracle_mpfc_rksav_run(n_steps=6, coef=_dup(C15))"},
        # edge: a different material on a coarser grid and a larger step.
        {"setup": "import numpy as np\nimport copy\n_dup = copy.deepcopy\n" + PSTIFF,
         "call": "mpfc_rksav_run(n_steps=5, dt=0.25, grid=(32, 48), cell=(24.0, 36.0), params=_dup(PS))",
         "gold_call": "_oracle_mpfc_rksav_run(n_steps=5, dt=0.25, grid=(32, 48), cell=(24.0, 36.0), params=_dup(PS))"},
        # edge: mass conservation, which the scheme preserves exactly on the
        # grid whatever the step size, grid or parameters.
        {"setup": "import numpy as np\nimport copy\n_dup = copy.deepcopy\n",
         "call": "mpfc_rksav_run(n_steps=4, dt=1.0, quantity='mass')",
         "gold_call": "_oracle_mpfc_rksav_run(n_steps=4, dt=1.0, quantity='mass')"},
        # control: the stage-work term of the LAST step, which only the energy
        # identity of step 8 produces - a run that never calls it cannot
        # return this number at all.
        {"setup": "import numpy as np\nimport copy\n_dup = copy.deepcopy\n",
         "call": "mpfc_rksav_run(n_steps=3, quantity='work')",
         "gold_call": "_oracle_mpfc_rksav_run(n_steps=3, quantity='work')"},
        # control: the quadratic-form term of the same step, the one the
        # energy theorem needs to be non-negative.
        {"setup": "import numpy as np\nimport copy\n_dup = copy.deepcopy\n",
         "call": "mpfc_rksav_run(n_steps=2, dt=0.5, quantity='qform')",
         "gold_call": "_oracle_mpfc_rksav_run(n_steps=2, dt=0.5, quantity='qform')"},
        # edge: an unknown reported quantity, which must raise ValueError.
        {"setup": "import numpy as np\nimport copy\n_dup = copy.deepcopy\n"
                  "def _guard(f, *a, **k):\n"
                  "    try:\n"
                  "        f(*a, **k)\n"
                  "    except ValueError:\n"
                  "        return 1.0\n"
                  "    return 0.0\n",
         "call": "_guard(mpfc_rksav_run, n_steps=0, quantity='banana')",
         "gold_call": "_guard(_oracle_mpfc_rksav_run, n_steps=0, quantity='banana')"},
        # boundary: odd point counts on both axes, so neither axis carries a
        # Nyquist mode, run end to end for a few steps.
        {"setup": "import numpy as np\nimport copy\n_dup = copy.deepcopy\n",
         "call": "mpfc_rksav_run(n_steps=3, dt=0.1, grid=(21, 27), cell=(14.0, 18.0))",
         "gold_call": "_oracle_mpfc_rksav_run(n_steps=3, dt=0.1, grid=(21, 27), cell=(14.0, 18.0))"},
    ]
