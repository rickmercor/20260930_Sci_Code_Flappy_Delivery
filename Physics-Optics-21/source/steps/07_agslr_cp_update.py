"""
Apply one iteration of the reconstruction scheme to the current state. The state is the tuple (x, xbar, y, p, q, r) holding the image, the extrapolated image, the dual variable of the data term and the three dual variables of the gradient terms. Update the four duals with the proximal mappings of their conjugates, take the proximal step on the image with the low-rank operator of step 5, and form the new extrapolated image exactly as the source method prints it. Must call the earlier step functions by name; no inlined reimplementation. Raise ValueError if any of the first three regularisation weights is not positive. The submitted function must import inside itself whatever it uses.

The objective splits as $F(Kx)+G(x)$ with $F$ collecting the data term and

the three gradient penalties and $G$ the low-rank penalty. The conjugate of the quadratic data term

gives a dual update that is a damped combination of the old dual and the current residual, while the

conjugate of an $\ell_1$ penalty is the indicator of a box, whose proximal mapping clips the candidate

dual to the box radius: $\lambda\,c/\max(\lambda, |c|)$ componentwise. The primal step then moves

against the transposed duals and applies the proximal mapping of $G$. The relaxation that produces the

extrapolated image is the one printed by the source method, evaluated with its relaxation parameter

equal to 1.

Returns
-------
return updated
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def agslr_cp_update(state: "tuple", projections: "np.ndarray", coords: "np.ndarray",
                    det_size: int, lambdas: "tuple", tau: float, sigma: float,
                    gamma: float = 1.0) -> tuple:
    """One iteration of the reconstruction scheme.

    Args:
        state: (x, xbar, y, p, q, r); x, xbar, p, q, r are float64 arrays of shape
            (nx, ny, nz) and y is a float64 array shaped like the projections.
        projections: float64 array of shape (n_views, det_size, det_size).
        coords: float64 array of shape (2, n_views, nz, ny, nx) from step 1.
        det_size: number of detector pixels along each side.
        lambdas: (lambda1, lambda2, lambda3, lambda4), the three gradient weights and the
            low-rank weight.
        tau: primal step size.
        sigma: dual step size.
        gamma: relaxation parameter.

    Returns:
        tuple: the updated (x, xbar, y, p, q, r).

    Raises:
        ValueError: if lambda1, lambda2 or lambda3 is not positive.
    """
    return updated

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_agslr_cp_update(state: "tuple", projections: "np.ndarray", coords: "np.ndarray",
                            det_size: int, lambdas: "tuple", tau: float, sigma: float,
                            gamma: float = 1.0) -> tuple:
    """One iteration of Algorithm 3: dual ascent, t-SVT proximal step, printed extrapolation."""
    import numpy as np
    x, xbar, y, p, q, r = (np.asarray(a, dtype=np.float64) for a in state)
    g = np.asarray(projections, dtype=np.float64)
    l1, l2, l3, l4 = (float(v) for v in lambdas)
    if min(l1, l2, l3) <= 0.0:
        raise ValueError("lambda1, lambda2 and lambda3 must be positive")
    y_new = (y + sigma * (_oracle_cl_forward_project(xbar, coords, det_size) - g)) / (1.0 + sigma)
    duals = []
    for dual, lam, axis in ((p, l1, 0), (q, l2, 1), (r, l3, 2)):
        cand = dual + sigma * _oracle_anisotropic_gradient(xbar, axis)
        duals.append(lam * cand / np.maximum(lam, np.abs(cand)))
    p_new, q_new, r_new = duals
    acc = _oracle_cl_back_project(y_new, coords, x.shape)
    for dual, axis in ((p_new, 0), (q_new, 1), (r_new, 2)):
        acc = acc + _oracle_anisotropic_gradient(dual, axis, adjoint=True)
    x_new = _oracle_tensor_svt(x - tau * acc, tau * l4)
    xbar_new = x + gamma * (x_new - x)
    return x_new, xbar_new, y_new, p_new, q_new, r_new

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
    zero_state = (
        "z = np.zeros(SHAPE)\n"
        "g = ref_project(phantom(15), COORDS, DET)\n"
        "state0 = (z, z.copy(), np.zeros_like(g), z.copy(), z.copy(), z.copy())\n"
        "LAMS = (0.2, 0.2, 0.01, 40.0)\n"
        "TAU = SIG = 1.0/22.0\n"
    )
    return [
        {   # 7.1 reference: one update from the zero state, amplified to O(1)
            "setup": common + zero_state,
            "call": "1.0e2*np.stack([np.asarray(a) for a in _invoke_copy(agslr_cp_update, state0, g, COORDS, DET, LAMS, TAU, SIG)[:2]])",
            "gold_call": "1.0e2*np.stack([np.asarray(a) for a in _invoke_copy(_oracle_agslr_cp_update, state0, g, COORDS, DET, LAMS, TAU, SIG)[:2]])",
        },
        {   # 7.2 analytic: from the zero state the data dual is -sigma g / (1 + sigma)
            "setup": common + zero_state,
            "call": "1.0e2*np.asarray(_invoke_copy(agslr_cp_update, state0, g, COORDS, DET, LAMS, TAU, SIG)[2])",
            "gold_call": "1.0e2*np.asarray(_invoke_copy(_oracle_agslr_cp_update, state0, g, COORDS, DET, LAMS, TAU, SIG)[2])",
        },
        {   # 7.3 property: the three gradient duals stay inside their boxes
            "setup": common + zero_state + (
                "box = lambda s: np.array([float(np.max(np.abs(np.asarray(s[3])))),\n"
                "                          float(np.max(np.abs(np.asarray(s[4])))),\n"
                "                          float(np.max(np.abs(np.asarray(s[5]))))])\n"
            ),
            "call": "1.0e2*box(_invoke_copy(agslr_cp_update, state0, g, COORDS, DET, LAMS, TAU, SIG))",
            "gold_call": "1.0e2*box(_invoke_copy(_oracle_agslr_cp_update, state0, g, COORDS, DET, LAMS, TAU, SIG))",
        },
        {   # 7.4 property: with the printed relaxation at gamma = 1 the extrapolated image equals
            #     the new image, which a textbook extrapolation would not satisfy
            "setup": common + zero_state + (
                "gap = lambda s: 1.0e6*(np.asarray(s[1]) - np.asarray(s[0])) + 1.0\n"
            ),
            "call": "gap(_invoke_copy(agslr_cp_update, state0, g, COORDS, DET, LAMS, TAU, SIG))",
            "gold_call": "gap(_invoke_copy(_oracle_agslr_cp_update, state0, g, COORDS, DET, LAMS, TAU, SIG))",
        },
        {   # 7.5 reference: one update from a non-zero state, amplified to O(1)
            "setup": common + zero_state + (
                "rng = np.random.default_rng(16)\n"
                "s1 = (phantom(17), phantom(18), 0.1*rng.standard_normal(g.shape),\n"
                "      0.05*rng.standard_normal(SHAPE), 0.05*rng.standard_normal(SHAPE),\n"
                "      0.005*rng.standard_normal(SHAPE))\n"
            ),
            "call": "1.0e2*np.asarray(_invoke_copy(agslr_cp_update, s1, g, COORDS, DET, LAMS, TAU, SIG)[0])",
            "gold_call": "1.0e2*np.asarray(_invoke_copy(_oracle_agslr_cp_update, s1, g, COORDS, DET, LAMS, TAU, SIG)[0])",
        },
        {   # 7.6 contract: a non-positive gradient weight raises ValueError
            "setup": common + zero_state,
            "call": "raises(lambda: _invoke_copy(agslr_cp_update, state0, g, COORDS, DET, (0.0, 0.2, 0.01, 20.0), TAU, SIG))",
            "gold_call": "raises(lambda: _invoke_copy(_oracle_agslr_cp_update, state0, g, COORDS, DET, (0.0, 0.2, 0.01, 20.0), TAU, SIG))",
        },
    ]
