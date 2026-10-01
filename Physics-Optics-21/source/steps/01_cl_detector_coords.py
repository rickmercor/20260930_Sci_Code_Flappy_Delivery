"""
Compute, for every view, the detector coordinates at which each voxel centre is imaged by the rotational cone-beam laminography geometry. Return one array holding the two coordinate maps for all views, so that the projector and its transpose consume exactly the same geometry. Voxel centres are offset by half a voxel from the grid corner, and the detector centre sits at index det_size / 2. The submitted function must import inside itself whatever it uses.

For view $v$ with $\phi_v = 2\pi v / n_{\text{views}}$ the source sits at

$S_v=\mathrm{sod}\,(\sin\theta\cos\phi_v,\ \sin\theta\sin\phi_v,\ \cos\theta)$, the detector plane is

perpendicular to $\hat S_v = S_v/\lVert S_v\rVert$ at distance $\mathrm{sdd}$ from the source, and the

in-plane axes are $e_1=(-\sin\phi_v,\cos\phi_v,0)$ and $e_2=\hat S_v\times e_1$. A voxel centre

$\mathbf r$ is imaged where the ray from $S_v$ through $\mathbf r$ meets that plane, which is a

perspective division, not an orthogonal projection: the scale factor depends on the depth of the voxel

along $\hat S_v$. The slope $\theta$ is what distinguishes laminography from tomography; at

$\theta = 90°$ the geometry degenerates to the circular tomographic one.

Returns
-------
return coords
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def cl_detector_coords(shape: "tuple", n_views: int, theta_deg: float, sod: float,
                       sdd: float, voxel_size: float, det_pixel: float,
                       det_size: int) -> "np.ndarray":
    """Detector coordinates of every voxel centre, for every view.

    Args:
        shape: (nx, ny, nz) of the volume, in voxels.
        n_views: number of views, equally spaced over a full turn.
        theta_deg: laminographic slope in degrees.
        sod: source-to-object distance, mm.
        sdd: source-to-detector distance, mm.
        voxel_size: isotropic voxel size, mm.
        det_pixel: detector pixel size, mm.
        det_size: number of detector pixels along each side.

    Returns:
        np.ndarray: float64, shape (2, n_views, nz, ny, nx); index 0 carries the coordinate
            along e1 and index 1 the coordinate along e2, both in detector pixels.
    """
    return coords

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_cl_detector_coords(shape: "tuple", n_views: int, theta_deg: float, sod: float,
                               sdd: float, voxel_size: float, det_pixel: float,
                               det_size: int) -> "np.ndarray":
    """Detector coordinates (u, w) of every voxel centre, per view: rotational cone-beam CL."""
    import numpy as np
    nx, ny, nz = (int(s) for s in shape)
    theta = np.deg2rad(float(theta_deg))
    out = np.empty((2, int(n_views), nz, ny, nx), dtype=np.float64)
    zz, yy, xx = np.meshgrid((np.arange(nz) - nz / 2.0 + 0.5) * voxel_size,
                             (np.arange(ny) - ny / 2.0 + 0.5) * voxel_size,
                             (np.arange(nx) - nx / 2.0 + 0.5) * voxel_size, indexing="ij")
    pts = np.stack([xx, yy, zz], axis=-1)
    for v in range(int(n_views)):
        phi = 2.0 * np.pi * v / float(n_views)
        src = sod * np.array([np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi),
                              np.cos(theta)])
        nrm = src / np.linalg.norm(src)
        e1 = np.array([-np.sin(phi), np.cos(phi), 0.0])
        e2 = np.cross(nrm, e1)
        rel = pts - src
        t = sdd / (float(src @ nrm) - (pts @ nrm))
        hit = src + t[..., None] * rel
        out[0, v] = (hit @ e1) / det_pixel + det_size / 2.0
        out[1, v] = (hit @ e2) / det_pixel + det_size / 2.0
    return out

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    common = (
        "import numpy as np\n"
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
        {   # 1.1 analytic: the voxel at the origin images at the detector centre in every view
            "setup": common + (
                "one = dict(GEO); one['det_size'] = DET\n"
            ),
            "call": "np.asarray(cl_detector_coords((1, 1, 1), **one))[:, :, 0, 0, 0]",
            "gold_call": "np.asarray(_oracle_cl_detector_coords((1, 1, 1), **one))[:, :, 0, 0, 0]",
        },
        {   # 1.2 reference: the full coordinate map against an independent implementation
            "setup": common,
            "call": "cl_detector_coords(SHAPE, **GEO)",
            "gold_call": "_oracle_cl_detector_coords(SHAPE, **GEO)",
        },
        {   # 1.3 property: halving the detector pixel doubles the offset from the centre
            "setup": common + (
                "fine = dict(GEO); fine['det_pixel'] = 0.5\n"
            ),
            "call": "cl_detector_coords(SHAPE, **fine)",
            "gold_call": "_oracle_cl_detector_coords(SHAPE, **fine)",
        },
        {   # 1.4 boundary: the tomographic limit theta = 90 degrees stays finite
            "setup": common + (
                "tomo = dict(GEO); tomo['theta_deg'] = 90.0\n"
            ),
            "call": "cl_detector_coords(SHAPE, **tomo)",
            "gold_call": "_oracle_cl_detector_coords(SHAPE, **tomo)",
        },
        {   # 1.5 edge: a non-cubic volume keeps the (2, n_views, nz, ny, nx) layout
            "setup": common,
            "call": "cl_detector_coords((8, 12, 4), **GEO)",
            "gold_call": "_oracle_cl_detector_coords((8, 12, 4), **GEO)",
        },
    ]
