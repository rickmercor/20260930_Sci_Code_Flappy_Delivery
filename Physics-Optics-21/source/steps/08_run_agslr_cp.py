"""
Run the whole reconstruction: build the geometry with step 1, estimate the spectral norm with step 6, set both step sizes to its reciprocal, start from a zero image with zero duals, and apply n_iter iterations of step 7 with relaxation 1. Return the reconstructed volume. Must call the earlier step functions by name; no inlined reimplementation. The submitted function must import inside itself whatever it uses.

The reconstruction is the output of a fixed number of iterations, not of a

convergence test, so the iterate is a well-defined function of the data and the settings. Starting

from zero with zero duals is the initialisation the source method uses. Because the step sizes are

tied to the spectral norm of the stacked operator, the geometry, the projector and the gradients all

feed into them, and every part of the chain therefore influences the result.

Returns
-------
return reconstruction
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def run_agslr_cp(projections: "np.ndarray", shape: "tuple", n_views: int,
                 theta_deg: float, sod: float, sdd: float, voxel_size: float,
                 det_pixel: float, det_size: int, lambdas: "tuple",
                 n_iter: int = 30, n_power: int = 50) -> "np.ndarray":
    """Reconstruct a volume from laminographic projections.

    Args:
        projections: float64 array of shape (n_views, det_size, det_size).
        shape: (nx, ny, nz) of the volume.
        n_views: number of views.
        theta_deg: laminographic slope in degrees.
        sod: source-to-object distance, mm.
        sdd: source-to-detector distance, mm.
        voxel_size: isotropic voxel size, mm.
        det_pixel: detector pixel size, mm.
        det_size: number of detector pixels along each side.
        lambdas: (lambda1, lambda2, lambda3, lambda4).
        n_iter: number of iterations.
        n_power: number of power iterations for the spectral norm.

    Returns:
        np.ndarray: float64, shape (nx, ny, nz), the reconstruction.
    """
    return reconstruction

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_run_agslr_cp(projections: "np.ndarray", shape: "tuple", n_views: int,
                         theta_deg: float, sod: float, sdd: float, voxel_size: float,
                         det_pixel: float, det_size: int, lambdas: "tuple",
                         n_iter: int = 30, n_power: int = 50) -> "np.ndarray":
    """Reconstruct from f = 0: geometry, operator norm, then n_iter AGSLR-CP iterations."""
    import numpy as np
    g = np.asarray(projections, dtype=np.float64)
    coords = _oracle_cl_detector_coords(shape, n_views, theta_deg, sod, sdd, voxel_size,
                                        det_pixel, det_size)
    L = _oracle_operator_norm(coords, shape, det_size, n_power)
    tau = sigma = 1.0 / L
    nx, ny, nz = (int(s) for s in shape)
    zero = np.zeros((nx, ny, nz), dtype=np.float64)
    state = (zero, zero.copy(), np.zeros_like(g), zero.copy(), zero.copy(), zero.copy())
    for _ in range(int(n_iter)):
        state = _oracle_agslr_cp_update(state, g, coords, det_size, lambdas, tau, sigma, 1.0)
    return state[0]

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
    setup_data = (
        "f_true = phantom(19)\n"
        "g = ref_project(f_true, COORDS, DET)\n"
        "LAMS = (0.2, 0.2, 0.01, 40.0)\n"
        "rmse = lambda a, b: float(np.sqrt(np.mean((np.asarray(a) - b)**2)))\n"
    )
    return [
        {   # 8.1 boundary: zero iterations return the zero volume
            "setup": common + setup_data,
            "call": "_invoke_copy(run_agslr_cp, g, SHAPE, lambdas=LAMS, n_iter=0, n_power=20, **GEO) + 1.0",
            "gold_call": "_invoke_copy(_oracle_run_agslr_cp, g, SHAPE, lambdas=LAMS, n_iter=0, n_power=20, **GEO) + 1.0",
        },
        {   # 8.2 reference: five iterations on the small geometry
            "setup": common + setup_data,
            "call": "1.0e2*np.asarray(_invoke_copy(run_agslr_cp, g, SHAPE, lambdas=LAMS, n_iter=5, n_power=20, **GEO))",
            "gold_call": "1.0e2*np.asarray(_invoke_copy(_oracle_run_agslr_cp, g, SHAPE, lambdas=LAMS, n_iter=5, n_power=20, **GEO))",
        },
        {   # 8.3 property: iterating reduces the error against the truth
            "setup": common + setup_data,
            "call": "rmse(_invoke_copy(run_agslr_cp, g, SHAPE, lambdas=LAMS, n_iter=8, n_power=20, **GEO), f_true)",
            "gold_call": "rmse(_invoke_copy(_oracle_run_agslr_cp, g, SHAPE, lambdas=LAMS, n_iter=8, n_power=20, **GEO), f_true)",
        },
        {   # 8.4 property: a large low-rank weight drives the reconstruction to zero
            "setup": common + setup_data,
            "call": "float(np.max(np.abs(np.asarray(_invoke_copy(run_agslr_cp, g, SHAPE, lambdas=(0.2, 0.2, 0.01, 1.0e7), n_iter=3, n_power=20, **GEO))))) + 1.0",
            "gold_call": "float(np.max(np.abs(np.asarray(_invoke_copy(_oracle_run_agslr_cp, g, SHAPE, lambdas=(0.2, 0.2, 0.01, 1.0e7), n_iter=3, n_power=20, **GEO))))) + 1.0",
        },
        {   # 8.5 edge: a different slope changes the geometry and the reconstruction with it
            "setup": common + setup_data + "steep = dict(GEO); steep['theta_deg'] = 60.0\n",
            "call": "1.0e2*np.asarray(_invoke_copy(run_agslr_cp, g, SHAPE, lambdas=LAMS, n_iter=5, n_power=20, **steep))",
            "gold_call": "1.0e2*np.asarray(_invoke_copy(_oracle_run_agslr_cp, g, SHAPE, lambdas=LAMS, n_iter=5, n_power=20, **steep))",
        },
    ]
