"""
Apply the proximal mapping of the low-rank penalty used by the source method: the tensor singular value thresholding operator. It transforms the volume along the stacking direction, applies matrix singular value thresholding with the given threshold to the transformed slices, and returns the result to the spatial domain, taking the real part. Raise ValueError for a negative threshold. The submitted function must import inside itself whatever it uses.

Under the tensor-SVD induced by a transform along the third mode, the

tensor nuclear norm of a third-order tensor is the **average** of the nuclear norms of its

transformed frontal slices, $\lVert\mathcal X\rVert_*=\frac{1}{N_3}\sum_{j}\lVert\bar{\mathcal X}^{(j)}\rVert_*$,

equivalently the sum of the entries of the first **spatial** frontal slice of the singular tensor

$\mathcal S$. It is not the nuclear norm of the first transformed slice alone: for the tube

$[1,\,-1]$ that single slice vanishes while the norm is $1$. Its proximal mapping acts slice by slice

in the transformed domain, where each slice is an ordinary matrix: soft-threshold its singular values

at the given level and rebuild it. The source method

states this operator explicitly; note that the slices of a real tensor come in conjugate pairs, so the

work can be halved without changing the result, and that thresholding the untransformed slices is a

different operator with a different fixed point.

Returns
-------
return thresholded
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def tensor_svt(tensor: "np.ndarray", threshold: float) -> "np.ndarray":
    """Proximal mapping of the tensor nuclear norm.

    Args:
        tensor: float64 array of shape (n1, n2, n3).
        threshold: non-negative soft-thresholding level.

    Returns:
        np.ndarray: float64, same shape as the input.

    Raises:
        ValueError: if threshold is negative.
    """
    return thresholded

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_tensor_svt(tensor: "np.ndarray", threshold: float) -> "np.ndarray":
    """Algorithm 1: t-SVT. FFT along mode 3, SVT on the first half of the slices, conjugate rest."""
    import numpy as np
    Y = np.asarray(tensor, dtype=np.float64)
    if threshold < 0.0:
        raise ValueError("threshold must be non-negative")
    n3 = Y.shape[2]
    Yf = np.fft.fft(Y, axis=2)
    W = np.zeros_like(Yf)
    half = int(np.ceil((n3 + 1) / 2.0))
    for j in range(half):
        U, S, Vh = np.linalg.svd(Yf[:, :, j], full_matrices=False)
        W[:, :, j] = (U * np.maximum(S - threshold, 0.0)) @ Vh
    for j in range(half, n3):
        W[:, :, j] = np.conj(W[:, :, n3 - j])
    return np.real(np.fft.ifft(W, axis=2))

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
        {   # 5.1 boundary: a zero threshold returns the tensor unchanged
            "setup": common + "Y = phantom(10)\n",
            "call": "_invoke_copy(tensor_svt, Y, 0.0)",
            "gold_call": "_invoke_copy(_oracle_tensor_svt, Y, 0.0)",
        },
        {   # 5.2 boundary: a threshold above every singular value returns zeros
            "setup": common + "Y = phantom(11)\n",
            "call": "_invoke_copy(tensor_svt, Y, 1.0e6) + 1.0",
            "gold_call": "_invoke_copy(_oracle_tensor_svt, Y, 1.0e6) + 1.0",
        },
        {   # 5.3 analytic: for a tensor constant along the third mode the transform concentrates
            #     everything in the first slice, so the operator reduces to matrix thresholding there
            "setup": common + (
                "n1, n2, n3 = SHAPE\n"
                "M = np.random.default_rng(12).standard_normal((n1, n2))\n"
                "Y = np.repeat(M[:, :, None], n3, axis=2)\n"
                "U, S, Vh = np.linalg.svd(n3*M, full_matrices=False)\n"
                "ref = ((U*np.maximum(S - 1.5, 0.0)) @ Vh)/n3\n"
            ),
            "call": "1.0e2*np.asarray(_invoke_copy(tensor_svt, Y, 1.5))",
            "gold_call": "1.0e2*np.asarray(_invoke_copy(_oracle_tensor_svt, Y, 1.5))",
        },
        {   # 5.4 reference: a generic tensor at a threshold that bites, amplified to O(1)
            "setup": common + "Y = phantom(13)\n",
            "call": "1.0e2*_invoke_copy(tensor_svt, Y, 2.0)",
            "gold_call": "1.0e2*_invoke_copy(_oracle_tensor_svt, Y, 2.0)",
        },
        {   # 5.5 contract: a negative threshold raises ValueError
            "setup": common + "Y = phantom(14)\n",
            "call": "raises(lambda: _invoke_copy(tensor_svt, Y, -1.0))",
            "gold_call": "raises(lambda: _invoke_copy(_oracle_tensor_svt, Y, -1.0))",
        },
    ]
