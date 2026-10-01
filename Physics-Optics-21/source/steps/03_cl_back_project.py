"""
Apply the exact discrete transpose of step 2 for the same geometry: read the projection values at the same four detector pixels, weight them with the same bilinear weights and accumulate into the voxel. This is the transpose, not an inverse and not a filtered back-projection; it must satisfy the inner-product identity with step 2 to machine precision for arbitrary inputs. The submitted function must import inside itself whatever it uses.

A primal-dual scheme needs the adjoint $A^{\mathsf T}$ of the operator it

uses, because the dual variables live on the detector and the primal variable lives in the volume. If

step 2 is written as a sparse matrix acting on the flattened volume, this step applies its transpose,

which turns the scatter of step 2 into a gather with identical weights. Any other choice, such as an

interpolation of the projections at the mapped coordinates or a back-projection with a reconstruction

filter, breaks the identity $\langle Au, v\rangle = \langle u, A^{\mathsf T}v\rangle$ and with it the

convergence of the iteration.

Returns
-------
return volume
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def cl_back_project(projections: "np.ndarray", coords: "np.ndarray",
                    shape: "tuple") -> "np.ndarray":
    """Apply the exact transpose of the projector of step 2.

    Args:
        projections: float64 array of shape (n_views, det_size, det_size).
        coords: float64 array of shape (2, n_views, nz, ny, nx) from step 1.
        shape: (nx, ny, nz) of the volume.

    Returns:
        np.ndarray: float64, shape (nx, ny, nz).
    """
    return volume

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_cl_back_project(projections: "np.ndarray", coords: "np.ndarray",
                            shape: "tuple") -> "np.ndarray":
    """Exact discrete transpose of step 2: the same bilinear weights, gathered."""
    import numpy as np
    g = np.asarray(projections, dtype=np.float64)
    c = np.asarray(coords, dtype=np.float64)
    nx, ny, nz = (int(s) for s in shape)
    det = g.shape[1]
    out = np.zeros((nz, ny, nx), dtype=np.float64)
    for v in range(c.shape[1]):
        u, w = c[0, v], c[1, v]
        u0 = np.floor(u).astype(np.int64)
        w0 = np.floor(w).astype(np.int64)
        du, dw = u - u0, w - w0
        for ow, ou, wt in ((0, 0, (1.0 - du) * (1.0 - dw)), (0, 1, du * (1.0 - dw)),
                           (1, 0, (1.0 - du) * dw), (1, 1, du * dw)):
            iu = np.clip(u0 + ou, 0, det - 1)
            iw = np.clip(w0 + ow, 0, det - 1)
            out += wt * g[v][iw, iu]
    return out.transpose(2, 1, 0)

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
        {   # 3.1 analytic: the inner-product identity with the projector, to machine precision
            "setup": common + (
                "u = phantom(4); v = np.random.default_rng(5).standard_normal((NV, DET, DET))\n"
                "lhs = float(np.sum(ref_project(u, COORDS, DET)*v))\n"
                "ratio = lambda x: float(np.sum(u*np.asarray(x)))/lhs\n"
            ),
            "call": "ratio(_invoke_copy(cl_back_project, v, COORDS, SHAPE))",
            "gold_call": "ratio(_invoke_copy(_oracle_cl_back_project, v, COORDS, SHAPE))",
        },
        {   # 3.2 reference: the back-projection of a single lit detector pixel
            "setup": common + (
                "g = np.zeros((NV, DET, DET)); g[0, 9, 7] = 1.0\n"
                "ref = np.zeros(SHAPE)\n"
                "u0 = np.floor(COORDS[0, 0]).astype(int); w0 = np.floor(COORDS[1, 0]).astype(int)\n"
                "du, dw = COORDS[0, 0] - u0, COORDS[1, 0] - w0\n"
                "acc = np.zeros((SHAPE[2], SHAPE[1], SHAPE[0]))\n"
                "for ow, ou, wt in ((0,0,(1-du)*(1-dw)), (0,1,du*(1-dw)), (1,0,(1-du)*dw), (1,1,du*dw)):\n"
                "    iu = np.clip(u0+ou, 0, DET-1); iw = np.clip(w0+ow, 0, DET-1)\n"
                "    acc += wt*g[0][iw, iu]\n"
                "check = near(acc.transpose(2, 1, 0), 1e-12)\n"
            ),
            "call": "1.0e2*np.asarray(_invoke_copy(cl_back_project, g, COORDS, SHAPE))",
            "gold_call": "1.0e2*np.asarray(_invoke_copy(_oracle_cl_back_project, g, COORDS, SHAPE))",
        },
        {   # 3.3 property: linear in the projections
            "setup": common + (
                "rng = np.random.default_rng(6)\n"
                "a = rng.standard_normal((NV, DET, DET)); b = rng.standard_normal((NV, DET, DET))\n"
                "u = phantom(7)\n"
                "la = float(np.sum(ref_project(u, COORDS, DET)*(a + b)))\n"
                "ratio = lambda x: float(np.sum(u*np.asarray(x)))/la\n"
            ),
            "call": "ratio(_invoke_copy(cl_back_project, a + b, COORDS, SHAPE))",
            "gold_call": "ratio(_invoke_copy(_oracle_cl_back_project, a + b, COORDS, SHAPE))",
        },
        {   # 3.4 boundary: zero projections give a zero volume
            "setup": common + "z = np.zeros((NV, DET, DET))\n",
            "call": "_invoke_copy(cl_back_project, z, COORDS, SHAPE) + 1.0",
            "gold_call": "_invoke_copy(_oracle_cl_back_project, z, COORDS, SHAPE) + 1.0",
        },
    ]
