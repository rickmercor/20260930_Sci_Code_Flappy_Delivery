"""
Apply the system matrix A: deposit every voxel's value bilinearly at its detector image, using the coordinates of step 1, and accumulate over all voxels. Contributions are clamped to the detector edges, so a voxel imaged outside the detector deposits on the border pixel rather than being discarded. The operator is linear in the volume for fixed geometry. The submitted function must import inside itself whatever it uses.

The system matrix of a scanner is defined by the discretisation chosen for

the rays, and this task fixes it as a deposition: the value of a voxel is spread over the four detector

pixels surrounding its image, with the bilinear weights $(1-d_u)(1-d_w)$, $d_u(1-d_w)$, $(1-d_u)d_w$

and $d_ud_w$, where $d_u$ and $d_w$ are the fractional parts of the image coordinates. Because those

four weights sum to one, the operator conserves the total deposited mass except where clamping moves a

contribution onto the border. Writing the projector this way, rather than as an interpolation that

reads from the volume, is what makes its exact transpose available in closed form.

Returns
-------
return projections
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def cl_forward_project(volume: "np.ndarray", coords: "np.ndarray",
                       det_size: int) -> "np.ndarray":
    """Project a volume onto the detector by bilinear deposition.

    Args:
        volume: float64 array of shape (nx, ny, nz).
        coords: float64 array of shape (2, n_views, nz, ny, nx) from step 1.
        det_size: number of detector pixels along each side.

    Returns:
        np.ndarray: float64, shape (n_views, det_size, det_size).
    """
    return projections

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_cl_forward_project(volume: "np.ndarray", coords: "np.ndarray",
                               det_size: int) -> "np.ndarray":
    """System matrix A: bilinear deposition of every voxel at its perspective image."""
    import numpy as np
    f = np.asarray(volume, dtype=np.float64)
    c = np.asarray(coords, dtype=np.float64)
    n_views = c.shape[1]
    det = int(det_size)
    val = f.transpose(2, 1, 0)
    out = np.zeros((n_views, det, det), dtype=np.float64)
    for v in range(n_views):
        u, w = c[0, v], c[1, v]
        u0 = np.floor(u).astype(np.int64)
        w0 = np.floor(w).astype(np.int64)
        du, dw = u - u0, w - w0
        acc = np.zeros(det * det, dtype=np.float64)
        for ow, ou, wt in ((0, 0, (1.0 - du) * (1.0 - dw)), (0, 1, du * (1.0 - dw)),
                           (1, 0, (1.0 - du) * dw), (1, 1, du * dw)):
            iu = np.clip(u0 + ou, 0, det - 1)
            iw = np.clip(w0 + ow, 0, det - 1)
            np.add.at(acc, (iw * det + iu).ravel(), (val * wt).ravel())
        out[v] = acc.reshape(det, det)
    return out

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
        {   # 2.1 analytic: the bilinear weights sum to one, so each view conserves the total mass
            "setup": common + (
                "f = phantom()\n"
            ),
            "call": "float(np.asarray(_invoke_copy(cl_forward_project, f, COORDS, DET)).sum()/(NV*f.sum()))",
            "gold_call": "float(np.asarray(_invoke_copy(_oracle_cl_forward_project, f, COORDS, DET)).sum()/(NV*f.sum()))",
        },
        {   # 2.2 reference: the projection against an independent deposition
            "setup": common + "f = phantom(1)\n",
            "call": "_invoke_copy(cl_forward_project, f, COORDS, DET)",
            "gold_call": "_invoke_copy(_oracle_cl_forward_project, f, COORDS, DET)",
        },
        {   # 2.3 property: linear in the volume
            "setup": common + (
                "a, b = phantom(2), phantom(3)\n"
            ),
            "call": "_invoke_copy(cl_forward_project, a + b, COORDS, DET)",
            "gold_call": "_invoke_copy(_oracle_cl_forward_project, a + b, COORDS, DET)",
        },
        {   # 2.4 boundary: a zero volume projects to zeros
            "setup": common + "z = np.zeros(SHAPE)\n",
            "call": "_invoke_copy(cl_forward_project, z, COORDS, DET) + 1.0",
            "gold_call": "_invoke_copy(_oracle_cl_forward_project, z, COORDS, DET) + 1.0",
        },
        {   # 2.5 edge: a single lit voxel deposits its value over at most four pixels per view
            "setup": common + (
                "f = np.zeros(SHAPE); f[7, 5, 2] = 2.5\n"
                "digest = lambda g: np.array([float(np.asarray(g).sum()),\n"
                "                             float(np.max(np.count_nonzero(np.asarray(g).reshape(NV, -1), axis=1)))])\n"
            ),
            "call": "digest(_invoke_copy(cl_forward_project, f, COORDS, DET))",
            "gold_call": "digest(_invoke_copy(_oracle_cl_forward_project, f, COORDS, DET))",
        },
    ]
