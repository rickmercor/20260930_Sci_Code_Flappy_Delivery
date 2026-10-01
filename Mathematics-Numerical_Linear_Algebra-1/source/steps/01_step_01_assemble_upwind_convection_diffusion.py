"""
Assemble a conservative exponentially fitted convection-diffusion operator on nonuniform cells with periodic and total-flux Robin boundaries.

Exponential fitting makes the discrete operator nonsymmetric in its values and in its sparsity pattern, which is the setting where the columns of a sparse approximate inverse need individually chosen patterns.

Returns
-------
np.ndarray: dense operator $A = V^{-1/2} L V^{-1/2}$.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def assemble_upwind_convection_diffusion(velocity_x: "np.ndarray", velocity_y: "np.ndarray", *, fitted: dict = None) -> "np.ndarray":
    r"""Return the conservative fitted transport matrix of a cell grid.

    Assemble the conservative cell-centred operator
    $\nabla \cdot (\mathbf{v} u - D \nabla u) + \sigma u$ on a rectangular
    tensor grid, periodic in $y$ and with homogeneous total-flux Robin
    conditions at the $x$ ends. Return $A = V^{-1/2} L V^{-1/2}$, where $L$
    maps cell values to integrated outward flux balances plus
    $\sigma \, V \, u$ and $V$ is the diagonal cell volume. Unknown $(i, j)$,
    counted from zero, has index $j \cdot n_x + i$.

    ``velocity_x`` and ``velocity_y`` have shape $(n_y, n_x)$ and are
    cell-centre values. The required ``fitted`` dictionary contains:
      x_edges, y_edges: strictly increasing finite 1-D coordinate arrays
        of lengths $n_x + 1$ and $n_y + 1$, with $n_x, n_y \ge 1$;
      diffusion_x: positive finite $(n_y, n_x + 1)$ face diffusivities,
        including both $x$ boundary faces;
      diffusion_y: positive finite $(n_y, n_x)$ face diffusivities, entry
        $[j, i]$ belonging to the upper face of cell $(i, j)$, including the
        seam;
      reaction: nonnegative finite $(n_y, n_x)$ cell reaction coefficients;
      robin_left, robin_right: nonnegative length-$n_y$ arrays, allowing
        $+\infty$. An infinite value means zero Dirichlet trace. Zero means
        no flux.

    Cell centres are edge midpoints. Each face flux is the constant value
    $F = v_f u(s) - D_f \, du(s)/ds$ of the exact one-dimensional homogeneous
    transport equation on the segment joining the two adjacent centres, with
    their cell values as endpoint data. The segment crosses the $y$ seam by
    periodic continuation. Its speed $v_f$ is the linear interpolant of the
    adjacent normal velocity components at that face; $D_f$ is the supplied
    face diffusivity. Multiply $F$ by the face length and add its outward
    contribution to EACH adjoining cell. Distinct periodic faces must all be
    counted even if they join the same pair ($n_y = 2$); the two contributions
    cancel when both cells are the same ($n_y = 1$).

    At an $x$ boundary use the adjacent cell's velocity and the supplied
    boundary diffusivity on the half-cell segment from its centre to the
    boundary. Eliminate the boundary trace using outward TOTAL flux
    $F_{\mathrm{out}} = \rho \, u_{\mathrm{boundary}}$; $\rho = +\infty$
    prescribes $u_{\mathrm{boundary}} = 0$. This condition includes
    convection, so replacing it by a diffusive Robin condition defines a
    different matrix. Reaction is integrated exactly as a piecewise constant
    cell term. Both boundary faces count when $n_x = 1$.

    Use continuous zero-speed limits. Peclet numbers can have magnitude up to
    $10^{4}$ and can be as small as $10^{-14}$. Cell widths lie in
    $[10^{-6}, 10^{3}]$, diffusivities in $[10^{-100}, 10^{100}]$, and
    absolute velocities and reactions are at most $10^{100}$. Positive finite
    Robin values lie in $[10^{-250}, 10^{100}]$. All resulting matrix entries
    are representable finite floats. These numerical ranges are
    preconditions, not additional validation rules. Overflow or premature
    underflow in an intermediate exponential is not an acceptable result.
    Preserve each matrix entry to $10^{-10}$ relative to
    $\max(1, \text{that entry's absolute magnitude})$.

    Raise ValueError for a missing or nondictionary ``fitted`` value, missing
    required keys, wrong array dimensions or shapes, nonincreasing or
    nonfinite edges, nonfinite velocities, diffusion or reaction, nonpositive
    diffusion, negative reaction, or negative or NaN Robin coefficients.
    Other keys are ignored.

    Parameters
    ----------
    velocity_x : np.ndarray
        Float array of shape $(n_y, n_x)$: the $x$ velocity component at cell
        centres, indexed ``[j, i]``.
    velocity_y : np.ndarray
        Float array of shape $(n_y, n_x)$: the $y$ velocity component, same
        indexing.
    fitted : dict
        Required finite-volume data described above.

    Returns
    -------
    np.ndarray
        Dense float array of shape $(n_x n_y,\ n_x n_y)$.

    Raises
    ------
    ValueError
        If ``fitted`` is absent or is not a dictionary, if a required key is
        missing, or if any supplied array violates the shape, monotonicity,
        finiteness, positivity or sign conditions listed above.
    """
    return matrix

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
def _oracle_assemble_upwind_convection_diffusion(velocity_x: "np.ndarray", velocity_y: "np.ndarray", *, fitted: dict = None) -> "np.ndarray":
    """Reference implementation (node-by-node stencil assembly)."""
    import numpy as np

    def _fitted_operator():
        if not isinstance(fitted, dict):
            raise ValueError("fitted must be a dictionary")
        names = ("x_edges", "y_edges", "diffusion_x", "diffusion_y",
                 "reaction", "robin_left", "robin_right")
        if any(name not in fitted for name in names):
            raise ValueError("missing fitted-grid data")
        xe, ye, dx_face, dy_face, reaction, left, right = [
            np.asarray(fitted[name], dtype=float) for name in names]
        vx, vy = np.asarray(velocity_x, dtype=float), np.asarray(velocity_y, dtype=float)
        if (xe.ndim != 1 or ye.ndim != 1 or xe.size < 2 or ye.size < 2
                or not np.all(np.isfinite(xe)) or not np.all(np.isfinite(ye))
                or np.any(np.diff(xe) <= 0) or np.any(np.diff(ye) <= 0)):
            raise ValueError("edges must be finite strictly increasing vectors")
        nx, ny = xe.size - 1, ye.size - 1
        if (vx.shape != (ny, nx) or vy.shape != (ny, nx)
                or dx_face.shape != (ny, nx + 1) or dy_face.shape != (ny, nx)
                or reaction.shape != (ny, nx) or left.shape != (ny,) or right.shape != (ny,)):
            raise ValueError("inconsistent fitted-grid shapes")
        if not all(np.all(np.isfinite(a)) for a in (vx, vy, dx_face, dy_face, reaction)):
            raise ValueError("cell and diffusion data must be finite")
        if np.any(dx_face <= 0) or np.any(dy_face <= 0) or np.any(reaction < 0):
            raise ValueError("diffusion must be positive and reaction nonnegative")
        if any(np.any(np.isnan(a)) or np.any(a < 0) for a in (left, right)):
            raise ValueError("Robin coefficients must be nonnegative, allowing positive infinity")
        hx, hy = np.diff(xe), np.diff(ye)
        volume = hy[:, None] * hx[None, :]
        L = np.diag((reaction * volume).ravel())

        def _rates(diffusion, distance, velocity):
            conductance = diffusion / distance
            pe = velocity / conductance
            # c_plus*u_left - c_minus*u_right solves the constant-flux ODE.
            z = abs(pe)
            if z < 1e-4:
                even = 1.0 + z*z/12.0 - z**4/720.0 + z**6/30240.0
                low = conductance * (even - z/2.0)
            elif z > 50:
                low = np.exp(np.log(abs(velocity)) - z - np.log(-np.expm1(-z)))
            else:
                decay = np.exp(-z)
                low = abs(velocity) * decay / (-np.expm1(-z))
            high = low + abs(velocity)
            return (high, low) if velocity >= 0 else (low, high)

        def _face(a, b, diffusion, distance, velocity, area):
            if a == b:
                return  # Both contributions of a periodic self-face cancel exactly.
            outgoing, incoming = _rates(diffusion, distance, velocity)
            L[a, a] += area * outgoing
            L[a, b] -= area * incoming
            L[b, a] -= area * outgoing
            L[b, b] += area * incoming

        def _boundary(cell, diffusion, distance, outward_speed, area, rho):
            outgoing, incoming = _rates(diffusion, distance, outward_speed)
            if rho == 0:
                effective = 0.0
            elif np.isposinf(rho):
                effective = outgoing
            elif rho >= incoming:
                effective = outgoing / (1.0 + incoming/rho)
            else:
                effective = outgoing * (rho/incoming) / (1.0 + rho/incoming)
            L[cell, cell] += area * effective

        for j in range(ny):
            for i in range(nx - 1):
                distance = (hx[i] + hx[i + 1]) / 2.0
                speed = (hx[i + 1]*vx[j, i] + hx[i]*vx[j, i + 1]) / (hx[i] + hx[i + 1])
                _face(j*nx+i, j*nx+i+1, dx_face[j, i+1], distance, speed, hy[j])
            _boundary(j*nx, dx_face[j, 0], hx[0]/2.0, -vx[j, 0], hy[j], left[j])
            _boundary(j*nx+nx-1, dx_face[j, -1], hx[-1]/2.0, vx[j, -1], hy[j], right[j])
        for j in range(ny):
            above = (j + 1) % ny
            distance = (hy[j] + hy[above]) / 2.0
            for i in range(nx):
                speed = (hy[above]*vy[j, i] + hy[j]*vy[above, i]) / (hy[j] + hy[above])
                _face(j*nx+i, above*nx+i, dy_face[j, i], distance, speed, hx[i])
        root_volume = np.sqrt(volume.ravel())
        return L / root_volume[:, None] / root_volume[None, :]

    return _fitted_operator()

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases() -> list[dict]:
    """Original regression cases followed by fitted/admission coverage."""
    return [
        {
            'setup': "import numpy as np\ndef status(fn):\n    try:fn(np.ones((2,2)),np.ones((2,2)))\n    except ValueError:return 1\n    return 0\n",
            'call': 'status(assemble_upwind_convection_diffusion)',
            'gold_call': 'status(_oracle_assemble_upwind_convection_diffusion)',
        },
        {
            'setup': "import copy\nimport numpy as np\ndef fixture(nx,ny,seed):\n    rng=np.random.default_rng(seed)\n    hx=.2+rng.random(nx); hx/=hx.sum()\n    hy=.2+rng.random(ny); hy/=hy.sum()\n    g=dict(x_edges=np.r_[0.,np.cumsum(hx)], y_edges=np.r_[0.,np.cumsum(hy)],\n           diffusion_x=.4+rng.random((ny,nx+1)), diffusion_y=.3+rng.random((ny,nx)),\n           reaction=.2+rng.random((ny,nx)),robin_left=.1+rng.random(ny),robin_right=.2+rng.random(ny))\n    return rng.normal(size=(ny,nx))*12,rng.normal(size=(ny,nx))*9,g\ndef packed(fn):\n    A=np.asarray(fn(vx.copy(),vy.copy(),fitted=copy.deepcopy(g)),dtype=float)\n    n=vx.size\n    if A.shape!=(n,n): raise AssertionError('wrong fitted operator shape')\n    return A.ravel()\nvx,vy,g=fixture(4,3,71)\n",
            'call': 'packed(assemble_upwind_convection_diffusion)',
            'gold_call': 'packed(_oracle_assemble_upwind_convection_diffusion)',
            'tol': 1e-09,
        },
        {
            'setup': "import copy\nimport numpy as np\ndef fixture(nx,ny,seed):\n    rng=np.random.default_rng(seed)\n    hx=.2+rng.random(nx); hx/=hx.sum()\n    hy=.2+rng.random(ny); hy/=hy.sum()\n    g=dict(x_edges=np.r_[0.,np.cumsum(hx)], y_edges=np.r_[0.,np.cumsum(hy)],\n           diffusion_x=.4+rng.random((ny,nx+1)), diffusion_y=.3+rng.random((ny,nx)),\n           reaction=.2+rng.random((ny,nx)),robin_left=.1+rng.random(ny),robin_right=.2+rng.random(ny))\n    return rng.normal(size=(ny,nx))*12,rng.normal(size=(ny,nx))*9,g\ndef packed(fn):\n    A=np.asarray(fn(vx.copy(),vy.copy(),fitted=copy.deepcopy(g)),dtype=float)\n    n=vx.size\n    if A.shape!=(n,n): raise AssertionError('wrong fitted operator shape')\n    return A.ravel()\nvx,vy,g=fixture(3,2,72)\nvx[:]=0;vy[:]=0\n",
            'call': 'packed(assemble_upwind_convection_diffusion)',
            'gold_call': 'packed(_oracle_assemble_upwind_convection_diffusion)',
            'tol': 1e-09,
        },
        {
            'setup': "import copy\nimport numpy as np\ndef fixture(nx,ny,seed):\n    rng=np.random.default_rng(seed)\n    hx=.2+rng.random(nx); hx/=hx.sum()\n    hy=.2+rng.random(ny); hy/=hy.sum()\n    g=dict(x_edges=np.r_[0.,np.cumsum(hx)], y_edges=np.r_[0.,np.cumsum(hy)],\n           diffusion_x=.4+rng.random((ny,nx+1)), diffusion_y=.3+rng.random((ny,nx)),\n           reaction=.2+rng.random((ny,nx)),robin_left=.1+rng.random(ny),robin_right=.2+rng.random(ny))\n    return rng.normal(size=(ny,nx))*12,rng.normal(size=(ny,nx))*9,g\ndef packed(fn):\n    A=np.asarray(fn(vx.copy(),vy.copy(),fitted=copy.deepcopy(g)),dtype=float)\n    n=vx.size\n    if A.shape!=(n,n): raise AssertionError('wrong fitted operator shape')\n    return A.ravel()\nvx,vy,g=fixture(4,1,73)\nvy[:]=37\ng['robin_left'][:]=0;g['robin_right'][:]=np.inf\n",
            'call': 'packed(assemble_upwind_convection_diffusion)',
            'gold_call': 'packed(_oracle_assemble_upwind_convection_diffusion)',
            'tol': 1e-09,
        },
        {
            'setup': "import copy\nimport numpy as np\ndef fixture(nx,ny,seed):\n    rng=np.random.default_rng(seed)\n    hx=.2+rng.random(nx); hx/=hx.sum()\n    hy=.2+rng.random(ny); hy/=hy.sum()\n    g=dict(x_edges=np.r_[0.,np.cumsum(hx)], y_edges=np.r_[0.,np.cumsum(hy)],\n           diffusion_x=.4+rng.random((ny,nx+1)), diffusion_y=.3+rng.random((ny,nx)),\n           reaction=.2+rng.random((ny,nx)),robin_left=.1+rng.random(ny),robin_right=.2+rng.random(ny))\n    return rng.normal(size=(ny,nx))*12,rng.normal(size=(ny,nx))*9,g\ndef packed(fn):\n    A=np.asarray(fn(vx.copy(),vy.copy(),fitted=copy.deepcopy(g)),dtype=float)\n    n=vx.size\n    if A.shape!=(n,n): raise AssertionError('wrong fitted operator shape')\n    return A.ravel()\nvx,vy,g=fixture(1,4,74)\ng['robin_left']=np.array([0.,np.inf,.00001,10.])\n",
            'call': 'packed(assemble_upwind_convection_diffusion)',
            'gold_call': 'packed(_oracle_assemble_upwind_convection_diffusion)',
            'tol': 1e-09,
        },
        {
            'setup': "import copy\nimport numpy as np\ndef fixture(nx,ny,seed):\n    rng=np.random.default_rng(seed)\n    hx=.2+rng.random(nx); hx/=hx.sum()\n    hy=.2+rng.random(ny); hy/=hy.sum()\n    g=dict(x_edges=np.r_[0.,np.cumsum(hx)], y_edges=np.r_[0.,np.cumsum(hy)],\n           diffusion_x=.4+rng.random((ny,nx+1)), diffusion_y=.3+rng.random((ny,nx)),\n           reaction=.2+rng.random((ny,nx)),robin_left=.1+rng.random(ny),robin_right=.2+rng.random(ny))\n    return rng.normal(size=(ny,nx))*12,rng.normal(size=(ny,nx))*9,g\ndef packed(fn):\n    A=np.asarray(fn(vx.copy(),vy.copy(),fitted=copy.deepcopy(g)),dtype=float)\n    n=vx.size\n    if A.shape!=(n,n): raise AssertionError('wrong fitted operator shape')\n    return A.ravel()\nvx,vy,g=fixture(3,2,75)\ng['diffusion_y'][0]=.3;g['diffusion_y'][1]=2.\n",
            'call': 'packed(assemble_upwind_convection_diffusion)',
            'gold_call': 'packed(_oracle_assemble_upwind_convection_diffusion)',
            'tol': 1e-09,
        },
        {
            'setup': "import copy\nimport numpy as np\ndef fixture(nx,ny,seed):\n    rng=np.random.default_rng(seed)\n    hx=.2+rng.random(nx); hx/=hx.sum()\n    hy=.2+rng.random(ny); hy/=hy.sum()\n    g=dict(x_edges=np.r_[0.,np.cumsum(hx)], y_edges=np.r_[0.,np.cumsum(hy)],\n           diffusion_x=.4+rng.random((ny,nx+1)), diffusion_y=.3+rng.random((ny,nx)),\n           reaction=.2+rng.random((ny,nx)),robin_left=.1+rng.random(ny),robin_right=.2+rng.random(ny))\n    return rng.normal(size=(ny,nx))*12,rng.normal(size=(ny,nx))*9,g\ndef packed(fn):\n    A=np.asarray(fn(vx.copy(),vy.copy(),fitted=copy.deepcopy(g)),dtype=float)\n    n=vx.size\n    if A.shape!=(n,n): raise AssertionError('wrong fitted operator shape')\n    return A.ravel()\nvx=np.array([[10.,10.],[-10.,-10.]])\nvy=np.array([[12.,-12.],[12.,-12.]])\n_,_,g=fixture(2,2,76)\ng['x_edges']=np.array([0.,.4,1.]);g['y_edges']=np.array([0.,.7,1.])\ng['diffusion_x'][:]=.001;g['diffusion_y'][:]=.001\ng['robin_left']=np.array([0.,1e-200]);g['robin_right']=np.array([np.inf,.02])\n",
            'call': 'packed(assemble_upwind_convection_diffusion)',
            'gold_call': 'packed(_oracle_assemble_upwind_convection_diffusion)',
            'tol': 1e-09,
        },
        {
            'setup': "import copy\nimport numpy as np\ndef fixture(nx,ny,seed):\n    rng=np.random.default_rng(seed)\n    hx=.2+rng.random(nx); hx/=hx.sum()\n    hy=.2+rng.random(ny); hy/=hy.sum()\n    g=dict(x_edges=np.r_[0.,np.cumsum(hx)], y_edges=np.r_[0.,np.cumsum(hy)],\n           diffusion_x=.4+rng.random((ny,nx+1)), diffusion_y=.3+rng.random((ny,nx)),\n           reaction=.2+rng.random((ny,nx)),robin_left=.1+rng.random(ny),robin_right=.2+rng.random(ny))\n    return rng.normal(size=(ny,nx))*12,rng.normal(size=(ny,nx))*9,g\ndef packed(fn):\n    A=np.asarray(fn(vx.copy(),vy.copy(),fitted=copy.deepcopy(g)),dtype=float)\n    n=vx.size\n    if A.shape!=(n,n): raise AssertionError('wrong fitted operator shape')\n    return A.ravel()\nvx,vy,g=fixture(3,4,77)\nvx*=1e-15;vy*=1e-15\n",
            'call': 'packed(assemble_upwind_convection_diffusion)',
            'gold_call': 'packed(_oracle_assemble_upwind_convection_diffusion)',
            'tol': 1e-09,
        },
        {
            'setup': "import copy\nimport numpy as np\ndef fixture(nx,ny,seed):\n    rng=np.random.default_rng(seed)\n    hx=.2+rng.random(nx); hx/=hx.sum()\n    hy=.2+rng.random(ny); hy/=hy.sum()\n    g=dict(x_edges=np.r_[0.,np.cumsum(hx)], y_edges=np.r_[0.,np.cumsum(hy)],\n           diffusion_x=.4+rng.random((ny,nx+1)), diffusion_y=.3+rng.random((ny,nx)),\n           reaction=.2+rng.random((ny,nx)),robin_left=.1+rng.random(ny),robin_right=.2+rng.random(ny))\n    return rng.normal(size=(ny,nx))*12,rng.normal(size=(ny,nx))*9,g\ndef packed(fn):\n    A=np.asarray(fn(vx.copy(),vy.copy(),fitted=copy.deepcopy(g)),dtype=float)\n    n=vx.size\n    if A.shape!=(n,n): raise AssertionError('wrong fitted operator shape')\n    return A.ravel()\nvx,vy,g=fixture(2,3,78)\ng['reaction'][:]=0;g['robin_left'][:]=0;g['robin_right'][:]=0\n",
            'call': 'packed(assemble_upwind_convection_diffusion)',
            'gold_call': 'packed(_oracle_assemble_upwind_convection_diffusion)',
            'tol': 1e-09,
        },
        {
            'setup': "import copy\nimport numpy as np\ndef fixture(nx,ny,seed):\n    rng=np.random.default_rng(seed)\n    hx=.2+rng.random(nx); hx/=hx.sum()\n    hy=.2+rng.random(ny); hy/=hy.sum()\n    g=dict(x_edges=np.r_[0.,np.cumsum(hx)], y_edges=np.r_[0.,np.cumsum(hy)],\n           diffusion_x=.4+rng.random((ny,nx+1)), diffusion_y=.3+rng.random((ny,nx)),\n           reaction=.2+rng.random((ny,nx)),robin_left=.1+rng.random(ny),robin_right=.2+rng.random(ny))\n    return rng.normal(size=(ny,nx))*12,rng.normal(size=(ny,nx))*9,g\ndef packed(fn):\n    A=np.asarray(fn(vx.copy(),vy.copy(),fitted=copy.deepcopy(g)),dtype=float)\n    n=vx.size\n    if A.shape!=(n,n): raise AssertionError('wrong fitted operator shape')\n    return A.ravel()\nvx,vy,g=fixture(1,1,79)\nvx[:]=2.;vy[:]=50.;g['robin_left'][:]=np.inf;g['robin_right'][:]=1e-12\n",
            'call': 'packed(assemble_upwind_convection_diffusion)',
            'gold_call': 'packed(_oracle_assemble_upwind_convection_diffusion)',
            'tol': 1e-09,
        },
        {
            'setup': "import copy\nimport numpy as np\ndef fixture(nx,ny,seed):\n    rng=np.random.default_rng(seed)\n    hx=.2+rng.random(nx); hx/=hx.sum()\n    hy=.2+rng.random(ny); hy/=hy.sum()\n    g=dict(x_edges=np.r_[0.,np.cumsum(hx)], y_edges=np.r_[0.,np.cumsum(hy)],\n           diffusion_x=.4+rng.random((ny,nx+1)), diffusion_y=.3+rng.random((ny,nx)),\n           reaction=.2+rng.random((ny,nx)),robin_left=.1+rng.random(ny),robin_right=.2+rng.random(ny))\n    return rng.normal(size=(ny,nx))*12,rng.normal(size=(ny,nx))*9,g\ndef packed(fn):\n    A=np.asarray(fn(vx.copy(),vy.copy(),fitted=copy.deepcopy(g)),dtype=float)\n    n=vx.size\n    if A.shape!=(n,n): raise AssertionError('wrong fitted operator shape')\n    return A.ravel()\nvx,vy,g=fixture(3,2,80)\ndel g['reaction']\ndef status(fn):\n    try:fn(vx.copy(),vy.copy(),fitted=copy.deepcopy(g))\n    except ValueError:return 1\n    return 0\n",
            'call': 'status(assemble_upwind_convection_diffusion)',
            'gold_call': 'status(_oracle_assemble_upwind_convection_diffusion)',
        },
        {
            'setup': "import copy\nimport numpy as np\ndef fixture(nx,ny,seed):\n    rng=np.random.default_rng(seed)\n    hx=.2+rng.random(nx); hx/=hx.sum()\n    hy=.2+rng.random(ny); hy/=hy.sum()\n    g=dict(x_edges=np.r_[0.,np.cumsum(hx)], y_edges=np.r_[0.,np.cumsum(hy)],\n           diffusion_x=.4+rng.random((ny,nx+1)), diffusion_y=.3+rng.random((ny,nx)),\n           reaction=.2+rng.random((ny,nx)),robin_left=.1+rng.random(ny),robin_right=.2+rng.random(ny))\n    return rng.normal(size=(ny,nx))*12,rng.normal(size=(ny,nx))*9,g\ndef packed(fn):\n    A=np.asarray(fn(vx.copy(),vy.copy(),fitted=copy.deepcopy(g)),dtype=float)\n    n=vx.size\n    if A.shape!=(n,n): raise AssertionError('wrong fitted operator shape')\n    return A.ravel()\nvx,vy,g=fixture(3,2,80)\ng['diffusion_x'][0,0]=-1\ndef status(fn):\n    try:fn(vx.copy(),vy.copy(),fitted=copy.deepcopy(g))\n    except ValueError:return 1\n    return 0\n",
            'call': 'status(assemble_upwind_convection_diffusion)',
            'gold_call': 'status(_oracle_assemble_upwind_convection_diffusion)',
        },
        {
            'setup': "import copy\nimport numpy as np\ndef fixture(nx,ny,seed):\n    rng=np.random.default_rng(seed)\n    hx=.2+rng.random(nx); hx/=hx.sum()\n    hy=.2+rng.random(ny); hy/=hy.sum()\n    g=dict(x_edges=np.r_[0.,np.cumsum(hx)], y_edges=np.r_[0.,np.cumsum(hy)],\n           diffusion_x=.4+rng.random((ny,nx+1)), diffusion_y=.3+rng.random((ny,nx)),\n           reaction=.2+rng.random((ny,nx)),robin_left=.1+rng.random(ny),robin_right=.2+rng.random(ny))\n    return rng.normal(size=(ny,nx))*12,rng.normal(size=(ny,nx))*9,g\ndef packed(fn):\n    A=np.asarray(fn(vx.copy(),vy.copy(),fitted=copy.deepcopy(g)),dtype=float)\n    n=vx.size\n    if A.shape!=(n,n): raise AssertionError('wrong fitted operator shape')\n    return A.ravel()\nvx,vy,g=fixture(3,2,80)\ng['diffusion_y']=np.ones((1,1))\ndef status(fn):\n    try:fn(vx.copy(),vy.copy(),fitted=copy.deepcopy(g))\n    except ValueError:return 1\n    return 0\n",
            'call': 'status(assemble_upwind_convection_diffusion)',
            'gold_call': 'status(_oracle_assemble_upwind_convection_diffusion)',
        },
        {
            'setup': "import copy\nimport numpy as np\ndef fixture(nx,ny,seed):\n    rng=np.random.default_rng(seed)\n    hx=.2+rng.random(nx); hx/=hx.sum()\n    hy=.2+rng.random(ny); hy/=hy.sum()\n    g=dict(x_edges=np.r_[0.,np.cumsum(hx)], y_edges=np.r_[0.,np.cumsum(hy)],\n           diffusion_x=.4+rng.random((ny,nx+1)), diffusion_y=.3+rng.random((ny,nx)),\n           reaction=.2+rng.random((ny,nx)),robin_left=.1+rng.random(ny),robin_right=.2+rng.random(ny))\n    return rng.normal(size=(ny,nx))*12,rng.normal(size=(ny,nx))*9,g\ndef packed(fn):\n    A=np.asarray(fn(vx.copy(),vy.copy(),fitted=copy.deepcopy(g)),dtype=float)\n    n=vx.size\n    if A.shape!=(n,n): raise AssertionError('wrong fitted operator shape')\n    return A.ravel()\nvx,vy,g=fixture(3,2,80)\ng['robin_left'][0]=np.nan\ndef status(fn):\n    try:fn(vx.copy(),vy.copy(),fitted=copy.deepcopy(g))\n    except ValueError:return 1\n    return 0\n",
            'call': 'status(assemble_upwind_convection_diffusion)',
            'gold_call': 'status(_oracle_assemble_upwind_convection_diffusion)',
        },
        {
            'setup': "import copy\nimport numpy as np\ndef fixture(nx,ny,seed):\n    rng=np.random.default_rng(seed)\n    hx=.2+rng.random(nx); hx/=hx.sum()\n    hy=.2+rng.random(ny); hy/=hy.sum()\n    g=dict(x_edges=np.r_[0.,np.cumsum(hx)], y_edges=np.r_[0.,np.cumsum(hy)],\n           diffusion_x=.4+rng.random((ny,nx+1)), diffusion_y=.3+rng.random((ny,nx)),\n           reaction=.2+rng.random((ny,nx)),robin_left=.1+rng.random(ny),robin_right=.2+rng.random(ny))\n    return rng.normal(size=(ny,nx))*12,rng.normal(size=(ny,nx))*9,g\ndef packed(fn):\n    A=np.asarray(fn(vx.copy(),vy.copy(),fitted=copy.deepcopy(g)),dtype=float)\n    n=vx.size\n    if A.shape!=(n,n): raise AssertionError('wrong fitted operator shape')\n    return A.ravel()\nvx,vy,g=fixture(3,2,80)\ng=False\ndef status(fn):\n    try:fn(vx.copy(),vy.copy(),fitted=copy.deepcopy(g))\n    except ValueError:return 1\n    return 0\n",
            'call': 'status(assemble_upwind_convection_diffusion)',
            'gold_call': 'status(_oracle_assemble_upwind_convection_diffusion)',
        },
        {
            'setup': "import copy\nimport numpy as np\ndef fixture(nx,ny,seed):\n    rng=np.random.default_rng(seed)\n    hx=.2+rng.random(nx); hx/=hx.sum()\n    hy=.2+rng.random(ny); hy/=hy.sum()\n    g=dict(x_edges=np.r_[0.,np.cumsum(hx)], y_edges=np.r_[0.,np.cumsum(hy)],\n           diffusion_x=.4+rng.random((ny,nx+1)), diffusion_y=.3+rng.random((ny,nx)),\n           reaction=.2+rng.random((ny,nx)),robin_left=.1+rng.random(ny),robin_right=.2+rng.random(ny))\n    return rng.normal(size=(ny,nx))*12,rng.normal(size=(ny,nx))*9,g\ndef packed(fn):\n    A=np.asarray(fn(vx.copy(),vy.copy(),fitted=copy.deepcopy(g)),dtype=float)\n    n=vx.size\n    if A.shape!=(n,n): raise AssertionError('wrong fitted operator shape')\n    return A.ravel()\nvx,vy,g=fixture(1,1,91)\nvx[:]=-1e100;vy[:]=0\ng['x_edges']=np.array([0.,2.]);g['y_edges']=np.array([0.,1.])\ng['diffusion_x'][:]=1e100/750;g['diffusion_y'][:]=1.;g['reaction'][:]=0\ng['robin_left'][:]=1.9016849634749534e-226\ng['robin_right'][:]=0.0\n",
            'call': 'packed(assemble_upwind_convection_diffusion)',
            'gold_call': 'packed(_oracle_assemble_upwind_convection_diffusion)',
            'tol': 1e-09,
        },
        {
            'setup': "import copy\nimport numpy as np\ndef fixture(nx,ny,seed):\n    rng=np.random.default_rng(seed)\n    hx=.2+rng.random(nx); hx/=hx.sum()\n    hy=.2+rng.random(ny); hy/=hy.sum()\n    g=dict(x_edges=np.r_[0.,np.cumsum(hx)], y_edges=np.r_[0.,np.cumsum(hy)],\n           diffusion_x=.4+rng.random((ny,nx+1)), diffusion_y=.3+rng.random((ny,nx)),\n           reaction=.2+rng.random((ny,nx)),robin_left=.1+rng.random(ny),robin_right=.2+rng.random(ny))\n    return rng.normal(size=(ny,nx))*12,rng.normal(size=(ny,nx))*9,g\ndef packed(fn):\n    A=np.asarray(fn(vx.copy(),vy.copy(),fitted=copy.deepcopy(g)),dtype=float)\n    n=vx.size\n    if A.shape!=(n,n): raise AssertionError('wrong fitted operator shape')\n    return A.ravel()\nvx,vy,g=fixture(1,1,91)\nvx[:]=1e100;vy[:]=0\ng['x_edges']=np.array([0.,2.]);g['y_edges']=np.array([0.,1.])\ng['diffusion_x'][:]=1e100/750;g['diffusion_y'][:]=1.;g['reaction'][:]=0\ng['robin_left'][:]=0.0\ng['robin_right'][:]=1.9016849634749534e-226\n",
            'call': 'packed(assemble_upwind_convection_diffusion)',
            'gold_call': 'packed(_oracle_assemble_upwind_convection_diffusion)',
            'tol': 1e-09,
        },
    ]
