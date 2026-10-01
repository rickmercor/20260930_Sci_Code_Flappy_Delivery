"""
Report, at every node, the unconfined free-energy profile along the path, the fraction of the orthogonal partition function that the given tube retains, and the profile that a calculation inside that tube would return. The first is the marginal of the equilibrium density over the sampled orthogonal window, with the same grid spacing used as the measure. The third follows from the first two by the exact relation the source derives between them, which is a two-line consequence of the definitions of the marginal and of the restricted marginal; get its sign right.

This is the whole point of the construction. A tube that keeps the same fraction of the orthogonal partition function at every point of the path shifts the profile by a constant and leaves every free-energy difference along the path intact, while a tube whose retained fraction drifts moves barriers and basins relative to each other. The relation implemented here is what turns a statement about retained fractions into a statement about free energies.

Returns
-------
A (n_stations, 3) float64 array with columns [unconfined free energy in kcal/mol, retained fraction, tube-restricted free energy in kcal/mol].
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def tube_projected_profile(slices: "np.ndarray", z_max: "np.ndarray", n_ortho: int, n_max: float) -> "np.ndarray":
    """Report, at every node, the unconfined free-energy profile along the path, the fraction of the
    orthogonal partition function that the given tube retains, and the profile that a calculation
    inside that tube would return. The first is the marginal of the equilibrium density over the
    sampled orthogonal window, with the same grid spacing used as the measure. The third follows
    from the first two by the exact relation the source derives between them, which is a two-line
    consequence of the definitions of the marginal and of the restricted marginal; get its sign
    right.

    Args:
        slices: array-like of shape (n_stations, n_ortho, 2), the slices returned by
            orthogonal_slices.
        z_max: array-like of shape (n_stations,), the tube half-width at every node, in the offset
            coordinate.
        n_ortho: int, an odd number of at least 3, the number of uniformly spaced signed normal
            offsets over the sampled window, endpoints included.
        n_max: float, the positive half-extent of the sampled window in Angstrom (offsets run from
            -n_max to n_max).

    Returns:
        A (n_stations, 3) float64 array with columns [unconfined free energy in kcal/mol, retained
        fraction, tube-restricted free energy in kcal/mol].

    Raises:
        ValueError: if z_max does not carry one entry per station.
    """
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _K_B():
    return 0.0019872041

def _TEMP():
    return 300.0

def _BETA():
    return 1.0/(_K_B()*_TEMP())

def _oracle_tube_projected_profile(slices: "np.ndarray", z_max: "np.ndarray", n_ortho: int, n_max: float) -> "np.ndarray":
    slices = np.asarray(slices, dtype=float)
    z_max = np.asarray(z_max, dtype=float).ravel()
    n_ortho = int(n_ortho); n_max = float(n_max)
    if slices.shape[0] != z_max.size:
        raise ValueError("z_max must carry one entry per station")
    dn = 2.0*n_max/(n_ortho-1)
    out = np.empty((slices.shape[0], 3))
    for a in range(slices.shape[0]):
        dz = slices[a, :, 0]; w = slices[a, :, 1]
        total = float(w.sum())
        inside = float(w[dz <= z_max[a]].sum())
        p = inside/total
        f_marg = -np.log(total*dn)/_BETA()
        out[a] = (f_marg, p, f_marg - np.log(p)/_BETA())
    return out

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {'setup': 'import numpy as np\nimages = _oracle_reference_path(60)\nstations = _oracle_station_frame(images, 11)\nn_ortho = 801\nn_max = 0.34\ndef _fx_orthogonal_slices(stations, images, n_ortho, n_max):\n    def _fx_guide(t):\n        t = np.asarray(t, dtype=float)\n        return np.stack([t, 0.15*np.sin(np.pi*t)], -1)\n    def _fx_guide_d1(t):\n        t = np.asarray(t, dtype=float)\n        return np.stack([np.ones_like(t), 0.15*np.pi*np.cos(np.pi*t)], -1)\n    def _fx_local_frame(t):\n        d1 = _fx_guide_d1(t)\n        sp = np.linalg.norm(d1, axis=-1)\n        tang = d1/sp[..., None]\n        return np.stack([-tang[..., 1], tang[..., 0]], -1)\n    def _fx_lambda_sharpness(imgs):\n        imgs = np.atleast_2d(np.asarray(imgs, dtype=float))\n        return 2.3/float(np.mean(np.sum(np.diff(imgs, axis=0)**2, axis=1)))\n    def _fx_pcv_coordinates(points, imgs):\n        points = np.atleast_2d(np.asarray(points, dtype=float))\n        imgs = np.atleast_2d(np.asarray(imgs, dtype=float))\n        lam_sharp = _fx_lambda_sharpness(imgs)\n        M = imgs.shape[0]\n        idx = np.arange(M, dtype=float)\n        out = np.empty((points.shape[0], 2))\n        step = 60000\n        for a in range(0, points.shape[0], step):\n            p = points[a:a+step]\n            d = (p[:, None, 0]-imgs[None, :, 0])**2 + (p[:, None, 1]-imgs[None, :, 1])**2\n            dmin = d.min(1, keepdims=True)\n            w = np.exp(-lam_sharp*(d-dmin))\n            tot = w.sum(1)\n            out[a:a+step, 0] = (w @ idx)/tot/(M-1)\n            out[a:a+step, 1] = -(np.log(tot) - lam_sharp*dmin[:, 0])/lam_sharp\n        return out\n    def _fx_energy(t, n):\n        t = np.asarray(t, dtype=float); n = np.asarray(n, dtype=float)\n        k_soft, k_stiff, w_neck, t_sad1, t_sad2 = 40.0, 400.0, 0.055, 0.30, 0.70\n        stiffness = k_soft + (k_stiff-k_soft)*(np.exp(-((t-t_sad1)/w_neck)**2) + np.exp(-((t-t_sad2)/w_neck)**2))\n        along = (7.5*np.exp(-((t-t_sad1)/0.10)**2) + 6.2*np.exp(-((t-t_sad2)/0.10)**2) - 1.6*np.exp(-((t-0.50)/0.11)**2))\n        cubic, quart, quart_m, ridge, ridge_t, ridge_n = 30.0, 140.0, 0.60, 0.35, 0.10, 0.055\n        return (along + 0.5*stiffness*n**2 + cubic*np.sin(2.0*np.pi*t)*n**3 + quart*(1.0+quart_m*np.cos(2.0*np.pi*t))*n**4 + ridge*np.exp(-((t-0.50)/ridge_t)**2)*np.exp(-(n/ridge_n)**2))\n    stations = np.atleast_2d(np.asarray(stations, dtype=float))\n    n_ortho = int(n_ortho); n_max = float(n_max)\n    beta = 1.0/(0.0019872041*300.0)\n    ng = np.linspace(-n_max, n_max, n_ortho)\n    t_st = stations[:, 1]; z_floor = stations[:, 2]\n    speed = stations[:, 3]; curv = stations[:, 4]\n    normal = _fx_local_frame(t_st)\n    xy = _fx_guide(t_st)[:, None, :] + ng[None, :, None]*normal[:, None, :]\n    z = _fx_pcv_coordinates(xy.reshape(-1, 2), images)[:, 1]\n    z = z.reshape(len(t_st), n_ortho)\n    u = _fx_energy(t_st[:, None], ng[None, :])\n    w = np.exp(-beta*(u - u.min()))*np.abs(1.0 - ng[None, :]*curv[:, None])*speed[:, None]\n    return np.stack([z - z_floor[:, None], w], -1)\nslices = _fx_orthogonal_slices(stations, images, n_ortho, n_max)\ndef _fx_conditional_free_energy(slices, n_ortho, n_max, d_eff):\n    slices = np.asarray(slices, dtype=float)\n    n_ortho = int(n_ortho); n_max = float(n_max); d_eff = int(d_eff)\n    beta = 1.0/(0.0019872041*300.0)\n    d_perp = d_eff - 1\n    ng = np.linspace(-n_max, n_max, n_ortho)\n    pos = ng > 0.0; neg = ng < 0.0\n    npos = ng[pos]\n    out = np.empty((slices.shape[0], npos.size, 2))\n    for a in range(slices.shape[0]):\n        dz = slices[a, :, 0]; w = slices[a, :, 1]\n        folded = w[pos] + np.interp(-npos, ng[neg], w[neg])\n        slope = np.gradient(dz, ng)[pos]\n        dens = folded/np.abs(slope)\n        order = np.argsort(dz[pos])\n        dzp = dz[pos][order]; dens = dens[order]\n        with np.errstate(divide="ignore", invalid="ignore"):\n            fhat = -np.log(dens*np.maximum(dzp, 0.0)**((2.0-d_perp)/2.0))/beta\n        out[a, :, 0] = dzp\n        out[a, :, 1] = fhat\n    return out\ncond = _fx_conditional_free_energy(slices, n_ortho, n_max, 2)\nz_max = _oracle_smooth_envelope(_oracle_adaptive_half_width(cond, slices, 2.0*0.0019872041*300.0, 2, 8.0e-5), 3)\n', 'call': 'tube_projected_profile(slices, z_max, n_ortho, n_max)', 'gold_call': '_oracle_tube_projected_profile(slices, z_max, n_ortho, n_max)'},
        {'setup': 'import numpy as np\nimages = _oracle_reference_path(200)\nstations = _oracle_station_frame(images, 41)\nn_ortho = 2001\nn_max = 0.34\ndef _fx_orthogonal_slices(stations, images, n_ortho, n_max):\n    def _fx_guide(t):\n        t = np.asarray(t, dtype=float)\n        return np.stack([t, 0.15*np.sin(np.pi*t)], -1)\n    def _fx_guide_d1(t):\n        t = np.asarray(t, dtype=float)\n        return np.stack([np.ones_like(t), 0.15*np.pi*np.cos(np.pi*t)], -1)\n    def _fx_local_frame(t):\n        d1 = _fx_guide_d1(t)\n        sp = np.linalg.norm(d1, axis=-1)\n        tang = d1/sp[..., None]\n        return np.stack([-tang[..., 1], tang[..., 0]], -1)\n    def _fx_lambda_sharpness(imgs):\n        imgs = np.atleast_2d(np.asarray(imgs, dtype=float))\n        return 2.3/float(np.mean(np.sum(np.diff(imgs, axis=0)**2, axis=1)))\n    def _fx_pcv_coordinates(points, imgs):\n        points = np.atleast_2d(np.asarray(points, dtype=float))\n        imgs = np.atleast_2d(np.asarray(imgs, dtype=float))\n        lam_sharp = _fx_lambda_sharpness(imgs)\n        M = imgs.shape[0]\n        idx = np.arange(M, dtype=float)\n        out = np.empty((points.shape[0], 2))\n        step = 60000\n        for a in range(0, points.shape[0], step):\n            p = points[a:a+step]\n            d = (p[:, None, 0]-imgs[None, :, 0])**2 + (p[:, None, 1]-imgs[None, :, 1])**2\n            dmin = d.min(1, keepdims=True)\n            w = np.exp(-lam_sharp*(d-dmin))\n            tot = w.sum(1)\n            out[a:a+step, 0] = (w @ idx)/tot/(M-1)\n            out[a:a+step, 1] = -(np.log(tot) - lam_sharp*dmin[:, 0])/lam_sharp\n        return out\n    def _fx_energy(t, n):\n        t = np.asarray(t, dtype=float); n = np.asarray(n, dtype=float)\n        k_soft, k_stiff, w_neck, t_sad1, t_sad2 = 40.0, 400.0, 0.055, 0.30, 0.70\n        stiffness = k_soft + (k_stiff-k_soft)*(np.exp(-((t-t_sad1)/w_neck)**2) + np.exp(-((t-t_sad2)/w_neck)**2))\n        along = (7.5*np.exp(-((t-t_sad1)/0.10)**2) + 6.2*np.exp(-((t-t_sad2)/0.10)**2) - 1.6*np.exp(-((t-0.50)/0.11)**2))\n        cubic, quart, quart_m, ridge, ridge_t, ridge_n = 30.0, 140.0, 0.60, 0.35, 0.10, 0.055\n        return (along + 0.5*stiffness*n**2 + cubic*np.sin(2.0*np.pi*t)*n**3 + quart*(1.0+quart_m*np.cos(2.0*np.pi*t))*n**4 + ridge*np.exp(-((t-0.50)/ridge_t)**2)*np.exp(-(n/ridge_n)**2))\n    stations = np.atleast_2d(np.asarray(stations, dtype=float))\n    n_ortho = int(n_ortho); n_max = float(n_max)\n    beta = 1.0/(0.0019872041*300.0)\n    ng = np.linspace(-n_max, n_max, n_ortho)\n    t_st = stations[:, 1]; z_floor = stations[:, 2]\n    speed = stations[:, 3]; curv = stations[:, 4]\n    normal = _fx_local_frame(t_st)\n    xy = _fx_guide(t_st)[:, None, :] + ng[None, :, None]*normal[:, None, :]\n    z = _fx_pcv_coordinates(xy.reshape(-1, 2), images)[:, 1]\n    z = z.reshape(len(t_st), n_ortho)\n    u = _fx_energy(t_st[:, None], ng[None, :])\n    w = np.exp(-beta*(u - u.min()))*np.abs(1.0 - ng[None, :]*curv[:, None])*speed[:, None]\n    return np.stack([z - z_floor[:, None], w], -1)\nslices = _fx_orthogonal_slices(stations, images, n_ortho, n_max)\nz_max = np.full(41, 0.02)\n', 'call': 'tube_projected_profile(slices, z_max, n_ortho, n_max)', 'gold_call': '_oracle_tube_projected_profile(slices, z_max, n_ortho, n_max)'},
        {'setup': 'import numpy as np\nimages = _oracle_reference_path(200)\nstations = _oracle_station_frame(images, 21)\nn_ortho = 1201\nn_max = 0.25\ndef _fx_orthogonal_slices(stations, images, n_ortho, n_max):\n    def _fx_guide(t):\n        t = np.asarray(t, dtype=float)\n        return np.stack([t, 0.15*np.sin(np.pi*t)], -1)\n    def _fx_guide_d1(t):\n        t = np.asarray(t, dtype=float)\n        return np.stack([np.ones_like(t), 0.15*np.pi*np.cos(np.pi*t)], -1)\n    def _fx_local_frame(t):\n        d1 = _fx_guide_d1(t)\n        sp = np.linalg.norm(d1, axis=-1)\n        tang = d1/sp[..., None]\n        return np.stack([-tang[..., 1], tang[..., 0]], -1)\n    def _fx_lambda_sharpness(imgs):\n        imgs = np.atleast_2d(np.asarray(imgs, dtype=float))\n        return 2.3/float(np.mean(np.sum(np.diff(imgs, axis=0)**2, axis=1)))\n    def _fx_pcv_coordinates(points, imgs):\n        points = np.atleast_2d(np.asarray(points, dtype=float))\n        imgs = np.atleast_2d(np.asarray(imgs, dtype=float))\n        lam_sharp = _fx_lambda_sharpness(imgs)\n        M = imgs.shape[0]\n        idx = np.arange(M, dtype=float)\n        out = np.empty((points.shape[0], 2))\n        step = 60000\n        for a in range(0, points.shape[0], step):\n            p = points[a:a+step]\n            d = (p[:, None, 0]-imgs[None, :, 0])**2 + (p[:, None, 1]-imgs[None, :, 1])**2\n            dmin = d.min(1, keepdims=True)\n            w = np.exp(-lam_sharp*(d-dmin))\n            tot = w.sum(1)\n            out[a:a+step, 0] = (w @ idx)/tot/(M-1)\n            out[a:a+step, 1] = -(np.log(tot) - lam_sharp*dmin[:, 0])/lam_sharp\n        return out\n    def _fx_energy(t, n):\n        t = np.asarray(t, dtype=float); n = np.asarray(n, dtype=float)\n        k_soft, k_stiff, w_neck, t_sad1, t_sad2 = 40.0, 400.0, 0.055, 0.30, 0.70\n        stiffness = k_soft + (k_stiff-k_soft)*(np.exp(-((t-t_sad1)/w_neck)**2) + np.exp(-((t-t_sad2)/w_neck)**2))\n        along = (7.5*np.exp(-((t-t_sad1)/0.10)**2) + 6.2*np.exp(-((t-t_sad2)/0.10)**2) - 1.6*np.exp(-((t-0.50)/0.11)**2))\n        cubic, quart, quart_m, ridge, ridge_t, ridge_n = 30.0, 140.0, 0.60, 0.35, 0.10, 0.055\n        return (along + 0.5*stiffness*n**2 + cubic*np.sin(2.0*np.pi*t)*n**3 + quart*(1.0+quart_m*np.cos(2.0*np.pi*t))*n**4 + ridge*np.exp(-((t-0.50)/ridge_t)**2)*np.exp(-(n/ridge_n)**2))\n    stations = np.atleast_2d(np.asarray(stations, dtype=float))\n    n_ortho = int(n_ortho); n_max = float(n_max)\n    beta = 1.0/(0.0019872041*300.0)\n    ng = np.linspace(-n_max, n_max, n_ortho)\n    t_st = stations[:, 1]; z_floor = stations[:, 2]\n    speed = stations[:, 3]; curv = stations[:, 4]\n    normal = _fx_local_frame(t_st)\n    xy = _fx_guide(t_st)[:, None, :] + ng[None, :, None]*normal[:, None, :]\n    z = _fx_pcv_coordinates(xy.reshape(-1, 2), images)[:, 1]\n    z = z.reshape(len(t_st), n_ortho)\n    u = _fx_energy(t_st[:, None], ng[None, :])\n    w = np.exp(-beta*(u - u.min()))*np.abs(1.0 - ng[None, :]*curv[:, None])*speed[:, None]\n    return np.stack([z - z_floor[:, None], w], -1)\nslices = _fx_orthogonal_slices(stations, images, n_ortho, n_max)\ndef _fx_conditional_free_energy(slices, n_ortho, n_max, d_eff):\n    slices = np.asarray(slices, dtype=float)\n    n_ortho = int(n_ortho); n_max = float(n_max); d_eff = int(d_eff)\n    beta = 1.0/(0.0019872041*300.0)\n    d_perp = d_eff - 1\n    ng = np.linspace(-n_max, n_max, n_ortho)\n    pos = ng > 0.0; neg = ng < 0.0\n    npos = ng[pos]\n    out = np.empty((slices.shape[0], npos.size, 2))\n    for a in range(slices.shape[0]):\n        dz = slices[a, :, 0]; w = slices[a, :, 1]\n        folded = w[pos] + np.interp(-npos, ng[neg], w[neg])\n        slope = np.gradient(dz, ng)[pos]\n        dens = folded/np.abs(slope)\n        order = np.argsort(dz[pos])\n        dzp = dz[pos][order]; dens = dens[order]\n        with np.errstate(divide="ignore", invalid="ignore"):\n            fhat = -np.log(dens*np.maximum(dzp, 0.0)**((2.0-d_perp)/2.0))/beta\n        out[a, :, 0] = dzp\n        out[a, :, 1] = fhat\n    return out\ncond = _fx_conditional_free_energy(slices, n_ortho, n_max, 2)\nz_max = _oracle_smooth_envelope(_oracle_adaptive_half_width(cond, slices, 3.0*0.0019872041*300.0, 2, 8.0e-5), 3)\n', 'call': 'tube_projected_profile(slices, z_max, n_ortho, n_max)', 'gold_call': '_oracle_tube_projected_profile(slices, z_max, n_ortho, n_max)'},
    ]
