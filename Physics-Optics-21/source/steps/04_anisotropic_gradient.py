"""
Apply the forward difference along one axis with periodic wrap, or its exact transpose when adjoint is true. Axis 0 and 1 are the in-plane directions and axis 2 is the stacking direction; the three are applied with different weights by the reconstruction, which is what makes the regularisation anisotropic. Raise ValueError for any axis other than 0, 1 or 2. The submitted function must import inside itself whatever it uses.

With periodic wrap the forward difference along an axis is a circulant

operator, and its transpose is the backward difference with the opposite sign, $(\nabla^{\mathsf

T}p)_i = p_{i-1} - p_i$. The pair is needed because the gradient terms enter the objective through

their duals: the scheme ascends on $\nabla\bar f$ and descends on $\nabla^{\mathsf T}p$. Getting the

transpose wrong by a sign or a shift leaves an operator that is no longer adjoint, which shows up as a

drift in the objective rather than as an obvious failure.

Returns
-------
return result
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def anisotropic_gradient(volume: "np.ndarray", axis: int,
                         adjoint: bool = False) -> "np.ndarray":
    """Forward difference along an axis with periodic wrap, or its transpose.

    Args:
        volume: float64 array of shape (nx, ny, nz).
        axis: 0, 1 or 2.
        adjoint: if True, apply the transpose instead of the forward difference.

    Returns:
        np.ndarray: float64, same shape as the input.

    Raises:
        ValueError: if axis is not 0, 1 or 2.
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_anisotropic_gradient(volume: "np.ndarray", axis: int,
                                 adjoint: bool = False) -> "np.ndarray":
    """Forward difference with periodic wrap along axis, or its exact transpose."""
    import numpy as np
    f = np.asarray(volume, dtype=np.float64)
    if int(axis) not in (0, 1, 2):
        raise ValueError("axis must be 0, 1 or 2")
    if adjoint:
        return np.roll(f, 1, axis=int(axis)) - f
    return np.roll(f, -1, axis=int(axis)) - f

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
        {   # 4.1 analytic: a ramp along an axis has a constant forward difference except at the wrap
            "setup": common + (
                "n = SHAPE[0]; ramp = np.arange(n, dtype=float)[:, None, None]*np.ones(SHAPE)\n"
                "exp = np.ones(SHAPE); exp[-1] = 1 - n\n"
                "check = near(exp, 1e-12)\n"
            ),
            "call": "_invoke_copy(anisotropic_gradient, ramp, 0)",
            "gold_call": "_invoke_copy(_oracle_anisotropic_gradient, ramp, 0)",
        },
        {   # 4.2 analytic: the adjoint identity along the stacking direction
            "setup": common + (
                "rng = np.random.default_rng(8); u = rng.standard_normal(SHAPE); p = rng.standard_normal(SHAPE)\n"
                "gu = np.roll(u, -1, axis=2) - u\n"
                "lhs = float(np.sum(gu*p))\n"
                "ratio = lambda x: float(np.sum(u*np.asarray(x)))/lhs\n"
            ),
            "call": "ratio(_invoke_copy(anisotropic_gradient, p, 2, adjoint=True))",
            "gold_call": "ratio(_invoke_copy(_oracle_anisotropic_gradient, p, 2, adjoint=True))",
        },
        {   # 4.3 property: a volume constant along an axis has zero difference along it
            "setup": common + (
                "f = np.ones(SHAPE)*np.arange(SHAPE[1])[None, :, None]\n"
                "check = near(np.zeros(SHAPE), 1e-14)\n"
            ),
            "call": "_invoke_copy(anisotropic_gradient, f, 0) + 1.0",
            "gold_call": "_invoke_copy(_oracle_anisotropic_gradient, f, 0) + 1.0",
        },
        {   # 4.4 contract: an invalid axis raises ValueError
            "setup": common + "f = phantom(9)\n",
            "call": "raises(lambda: _invoke_copy(anisotropic_gradient, f, 3))",
            "gold_call": "raises(lambda: _invoke_copy(_oracle_anisotropic_gradient, f, 3))",
        },
    ]
