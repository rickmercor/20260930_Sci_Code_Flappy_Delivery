"""
Estimate the spectral norm L of the stacked operator built from the projector of step 2 and the three gradients of step 4, by the power iteration on the normal operator fixed in the background: start from the unnormalised volume whose entries are all 1 and run exactly n_power iterations of b = Bx, s = ||b||, x = b/s, returning sqrt(s) from the last iteration. The step sizes of the reconstruction are 1/L, so this value fixes them. The submitted function must import inside itself whatever it uses.

The scheme stacks the projector and the three gradients into one operator

$K$, and its convergence condition ties the product of the step sizes to $\lVert K\rVert_2^{-2}$. The

square of that norm is the largest eigenvalue of $K^{\mathsf T}K = A^{\mathsf T}A + \sum_i

\nabla_i^{\mathsf T}\nabla_i$, so a power iteration on the normal operator converges to it.

The recurrence and the returned quantity are fixed, because at a finite iteration count different

readings of "the estimate" do not agree. Start from the **unnormalised** volume of ones $x_0$, and

for each of the $n_{\text{power}}$ iterations compute $b = Bx$ with $B = A^{\mathsf T}A + \sum_i

\nabla_i^{\mathsf T}\nabla_i$, then $s = \lVert b\rVert_2$, then $x = b/s$. Return $\sqrt{s}$ from the

last iteration. This is the growth factor of the final product, not a Rayleigh quotient

$\sqrt{\langle x, Bx\rangle/\langle x, x\rangle}$ evaluated after the loop, and not $\lVert Kx\rVert/\lVert x\rVert$;

those converge to the same limit but differ at any finite $n_{\text{power}}$. Starting from a fixed

deterministic vector rather than a random one makes the estimate reproducible.

Returns
-------
return norm
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def operator_norm(coords: "np.ndarray", shape: "tuple", det_size: int,
                  n_power: int = 50) -> float:
    """Spectral norm of the stacked operator, by power iteration.

    Args:
        coords: float64 array of shape (2, n_views, nz, ny, nx) from step 1.
        shape: (nx, ny, nz) of the volume.
        det_size: number of detector pixels along each side.
        n_power: number of power iterations.

    Returns:
        float: sqrt(s) from the last iteration of the recurrence given in the scientific
            background, where s is the Euclidean norm of B x before renormalisation.
    """
    return norm

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_operator_norm(coords: "np.ndarray", shape: "tuple", det_size: int,
                          n_power: int = 50) -> float:
    """L = ||K||_2 for K = [A; grad_x; grad_y; grad_z], by power iteration from the ones vector."""
    import numpy as np
    nx, ny, nz = (int(s) for s in shape)
    x = np.ones((nx, ny, nz), dtype=np.float64)
    norm = 0.0
    for _ in range(int(n_power)):
        y = _oracle_cl_forward_project(x, coords, det_size)
        acc = _oracle_cl_back_project(y, coords, shape)
        for axis in (0, 1, 2):
            acc = acc + _oracle_anisotropic_gradient(
                _oracle_anisotropic_gradient(x, axis), axis, adjoint=True)
        norm = float(np.linalg.norm(acc))
        x = acc / norm
    return float(np.sqrt(norm))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    common = (
        "import numpy as np\n"
        "from copy import deepcopy\n"
        "def _invoke_copy(fn, *args, **kwargs):\n"
        "    args, kwargs = deepcopy((args, kwargs))\n"
        "    return fn(*args, **kwargs)\n"
        "SHAPE, NV, DET = (12, 12, 6), 6, 16\n"
        "GEO = dict(n_views=NV, theta_deg=45.0, sod=230.0, sdd=700.0, voxel_size=0.5,\n"
        "           det_pixel=1.0, det_size=DET)\n"
        "def ref_coords(shape, n_views, theta_deg, sod, sdd, voxel_size, det_pixel, det_size):\n"
        "    # independent reference geometry, written from the stated conventions\n"
        "    nx, ny, nz = shape; th = np.deg2rad(theta_deg)\n"
        "    out = np.empty((2, n_views, nz, ny, nx))\n"
        "    zz, yy, xx = np.meshgrid((np.arange(nz) - nz/2 + 0.5)*voxel_size,\n"
        "                             (np.arange(ny) - ny/2 + 0.5)*voxel_size,\n"
        "                             (np.arange(nx) - nx/2 + 0.5)*voxel_size, indexing='ij')\n"
        "    pts = np.stack([xx, yy, zz], -1)\n"
        "    for v in range(n_views):\n"
        "        phi = 2*np.pi*v/n_views\n"
        "        s = sod*np.array([np.sin(th)*np.cos(phi), np.sin(th)*np.sin(phi), np.cos(th)])\n"
        "        n = s/np.linalg.norm(s); e1 = np.array([-np.sin(phi), np.cos(phi), 0.0])\n"
        "        e2 = np.cross(n, e1); rel = pts - s\n"
        "        t = sdd/((s @ n) - (pts @ n)); hit = s + t[..., None]*rel\n"
        "        out[0, v] = (hit @ e1)/det_pixel + det_size/2\n"
        "        out[1, v] = (hit @ e2)/det_pixel + det_size/2\n"
        "    return out\n"
        "COORDS = ref_coords(SHAPE, **GEO)\n"
        "def ref_project(vol, coords, det):\n"
        "    val = np.asarray(vol).transpose(2, 1, 0)\n"
        "    out = np.zeros((coords.shape[1], det, det))\n"
        "    for v in range(coords.shape[1]):\n"
        "        u, w = coords[0, v], coords[1, v]\n"
        "        u0, w0 = np.floor(u).astype(int), np.floor(w).astype(int)\n"
        "        du, dw = u - u0, w - w0; acc = np.zeros(det*det)\n"
        "        for ow, ou, wt in ((0,0,(1-du)*(1-dw)), (0,1,du*(1-dw)), (1,0,(1-du)*dw), (1,1,du*dw)):\n"
        "            iu, iw = np.clip(u0+ou, 0, det-1), np.clip(w0+ow, 0, det-1)\n"
        "            np.add.at(acc, (iw*det+iu).ravel(), (val*wt).ravel())\n"
        "        out[v] = acc.reshape(det, det)\n"
        "    return out\n"
        "def phantom(seed=0, shape=SHAPE):\n"
        "    rng = np.random.default_rng(seed)\n"
        "    return np.abs(rng.standard_normal(shape))\n"
        "def near(expected, atol):\n"
        "    expected = np.asarray(expected, dtype=float)\n"
        "    return lambda out: bool(np.allclose(np.asarray(out, dtype=float), expected, rtol=0, atol=atol))\n"
        "def raises(thunk):\n"
        "    try:\n"
        "        thunk()\n"
        "    except ValueError:\n"
        "        return True\n"
        "    return False\n"
    )
    return [
        {   # 6.1 reference: the spectral norm for the task geometry
            "setup": common,
            "call": "_invoke_copy(operator_norm, COORDS, SHAPE, DET, 30)",
            "gold_call": "_invoke_copy(_oracle_operator_norm, COORDS, SHAPE, DET, 30)",
        },
        {   # 6.2 property: the estimate is settled, so two power counts agree to 1e-6
            "setup": common,
            "call": "_invoke_copy(operator_norm, COORDS, SHAPE, DET, 40)",
            "gold_call": "_invoke_copy(_oracle_operator_norm, COORDS, SHAPE, DET, 40)",
        },
        {   # 6.3 property: the norm of the stacked operator is at least that of the gradient block,
            #     whose squared norm is 12 for three periodic forward differences
            "setup": common,
            "call": "float(_invoke_copy(operator_norm, COORDS, SHAPE, DET, 30))**2",
            "gold_call": "float(_invoke_copy(_oracle_operator_norm, COORDS, SHAPE, DET, 30))**2",
        },
        {   # 6.4 edge: a coarser geometry with fewer views
            "setup": common + (
                "few = dict(GEO); few['n_views'] = 3\n"
                "C3 = ref_coords(SHAPE, **few)\n"
            ),
            "call": "_invoke_copy(operator_norm, C3, SHAPE, DET, 30)",
            "gold_call": "_invoke_copy(_oracle_operator_norm, C3, SHAPE, DET, 30)",
        },
    ]
