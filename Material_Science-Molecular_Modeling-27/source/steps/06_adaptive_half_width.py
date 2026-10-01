"""
Find the half-width of the adaptive tube at every node: the offset at which the conditional free energy first rises by the prescribed amount above its on-path reference level, which is the level the source measures the excess from. An arithmetic path variable has a soft-minimum core in which the on-path value is not numerically well defined, so read the reference level by linear interpolation of the conditional at the prescribed reference offset and scan outward from there; interpolate the crossing itself linearly between the two bracketing nodes of the offset grid. Where the prescribed rise is never reached inside the sampled window, set the half-width to twice the contour level in units of thermal energy, divided by d_perp (the number of perpendicular directions, one less than the dimensionality of the collective-variable space), times the mean of the offset coordinate over the entries of the slice whose offset coordinate is non-negative, weighted by their statistical weights; that is the harmonic limit for an orthogonal coordinate that is a squared displacement. Return one half-width per node, in the units of the offset coordinate.

This is the step that replaces a geometric guess by a thermodynamic criterion. The width is no longer a distance chosen in advance, it is wherever the local orthogonal free energy has climbed by a fixed amount above the path, so the tube opens where the landscape is flat and closes where it pinches without anyone tuning it. The fallback exists because in flat regions the climb may never be observed inside the window that was sampled; on this configuration it is never needed, and it is written down so that the behaviour is defined rather than left open.

Returns
-------
A (n_stations,) float64 array of tube half-widths in the offset coordinate.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def adaptive_half_width(cond: "np.ndarray", slices: "np.ndarray", delta_f_star: float, d_eff: int, dz_ref: float) -> "np.ndarray":
    """Find the half-width of the adaptive tube at every node: the offset at which the conditional free
    energy first rises by the prescribed amount above its on-path reference level, which is the
    level the source measures the excess from. An arithmetic path variable has a soft-minimum core
    in which the on-path value is not numerically well defined, so read the reference level by
    linear interpolation of the conditional at the prescribed reference offset and scan outward from
    there; interpolate the crossing itself linearly between the two bracketing nodes of the offset
    grid. Where the prescribed rise is never reached inside the sampled window, set the half-width
    to twice the contour level in units of thermal energy, divided by d_perp (the number of
    perpendicular directions, one less than the dimensionality of the collective-variable space),
    times the mean of the offset coordinate over the entries of the slice whose offset coordinate is
    non-negative, weighted by their statistical weights; that is the harmonic limit for an
    orthogonal coordinate that is a squared displacement. Return one half-width per node, in the
    units of the offset coordinate.

    Args:
        cond: array-like of shape (n_stations, (n_ortho - 1)//2, 2), the conditional free energies
            returned by conditional_free_energy.
        slices: array-like of shape (n_stations, n_ortho, 2), the slices returned by
            orthogonal_slices.
        delta_f_star: float, the positive contour level in kcal/mol (the level in units of k_B T
            times k_B T).
        d_eff: int, the dimensionality of the collective-variable space (at least 2); the number of
            perpendicular directions is d_eff - 1.
        dz_ref: float, the positive offset in Angstrom^2 at which the reference level of the
            conditional free energy is read.

    Returns:
        A (n_stations,) float64 array of tube half-widths in the offset coordinate.

    Raises:
        ValueError: if cond does not have shape (n_stations, n_half, 2), if delta_f_star or dz_ref
        is not positive, or if d_eff is smaller than 2.
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

def _oracle_adaptive_half_width(cond: "np.ndarray", slices: "np.ndarray", delta_f_star: float, d_eff: int, dz_ref: float) -> "np.ndarray":
    cond = np.asarray(cond, dtype=float); slices = np.asarray(slices, dtype=float)
    delta_f_star = float(delta_f_star); d_eff = int(d_eff); dz_ref = float(dz_ref)
    if cond.ndim != 3 or cond.shape[2] != 2:
        raise ValueError("cond must have shape (n_stations, n_half, 2)")
    if delta_f_star <= 0.0:
        raise ValueError("delta_f_star must be positive")
    if d_eff < 2:
        raise ValueError("d_eff must be at least 2")
    if dz_ref <= 0.0:
        raise ValueError("dz_ref must be positive")
    d_perp = d_eff - 1
    ns = cond.shape[0]
    out = np.empty(ns)
    for a in range(ns):
        dz = cond[a, :, 0]; f = cond[a, :, 1]
        ok = np.isfinite(f)
        good = np.where(ok)[0]
        f_ref = float(np.interp(dz_ref, dz[good], f[good]))
        excess = f - f_ref
        base = int(np.searchsorted(dz, dz_ref))
        found = np.nan
        for q in range(max(base-1, int(good[0])), dz.size-1):
            if ok[q] and ok[q+1] and excess[q] <= delta_f_star <= excess[q+1]:
                found = dz[q] + (delta_f_star-excess[q])/(excess[q+1]-excess[q])*(dz[q+1]-dz[q])
                break
        if not np.isfinite(found):
            dza = slices[a, :, 0]; wa = slices[a, :, 1]
            keep = dza >= 0.0
            mean = float(np.sum(wa[keep]*dza[keep])/np.sum(wa[keep]))
            found = (2.0*_BETA()*delta_f_star/d_perp)*mean
        out[a] = found
    return out

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {'setup': 'import numpy as np\nimages = _oracle_reference_path(60)\nstations = _oracle_station_frame(images, 11)\nn_ortho = 801\nn_max = 0.34\ndef _fx_orthogonal_slices(stations, images, n_ortho, n_max):\n    def _fx_guide(t):\n        t = np.asarray(t, dtype=float)\n        return np.stack([t, 0.15*np.sin(np.pi*t)], -1)\n    def _fx_guide_d1(t):\n        t = np.asarray(t, dtype=float)\n        return np.stack([np.ones_like(t), 0.15*np.pi*np.cos(np.pi*t)], -1)\n    def _fx_local_frame(t):\n        d1 = _fx_guide_d1(t)\n        sp = np.linalg.norm(d1, axis=-1)\n        tang = d1/sp[..., None]\n        return np.stack([-tang[..., 1], tang[..., 0]], -1)\n    def _fx_lambda_sharpness(imgs):\n        imgs = np.atleast_2d(np.asarray(imgs, dtype=float))\n        return 2.3/float(np.mean(np.sum(np.diff(imgs, axis=0)**2, axis=1)))\n    def _fx_pcv_coordinates(points, imgs):\n        points = np.atleast_2d(np.asarray(points, dtype=float))\n        imgs = np.atleast_2d(np.asarray(imgs, dtype=float))\n        lam_sharp = _fx_lambda_sharpness(imgs)\n        M = imgs.shape[0]\n        idx = np.arange(M, dtype=float)\n        out = np.empty((points.shape[0], 2))\n        step = 60000\n        for a in range(0, points.shape[0], step):\n            p = points[a:a+step]\n            d = (p[:, None, 0]-imgs[None, :, 0])**2 + (p[:, None, 1]-imgs[None, :, 1])**2\n            dmin = d.min(1, keepdims=True)\n            w = np.exp(-lam_sharp*(d-dmin))\n            tot = w.sum(1)\n            out[a:a+step, 0] = (w @ idx)/tot/(M-1)\n            out[a:a+step, 1] = -(np.log(tot) - lam_sharp*dmin[:, 0])/lam_sharp\n        return out\n    def _fx_energy(t, n):\n        t = np.asarray(t, dtype=float); n = np.asarray(n, dtype=float)\n        k_soft, k_stiff, w_neck, t_sad1, t_sad2 = 40.0, 400.0, 0.055, 0.30, 0.70\n        stiffness = k_soft + (k_stiff-k_soft)*(np.exp(-((t-t_sad1)/w_neck)**2) + np.exp(-((t-t_sad2)/w_neck)**2))\n        along = (7.5*np.exp(-((t-t_sad1)/0.10)**2) + 6.2*np.exp(-((t-t_sad2)/0.10)**2) - 1.6*np.exp(-((t-0.50)/0.11)**2))\n        cubic, quart, quart_m, ridge, ridge_t, ridge_n = 30.0, 140.0, 0.60, 0.35, 0.10, 0.055\n        return (along + 0.5*stiffness*n**2 + cubic*np.sin(2.0*np.pi*t)*n**3 + quart*(1.0+quart_m*np.cos(2.0*np.pi*t))*n**4 + ridge*np.exp(-((t-0.50)/ridge_t)**2)*np.exp(-(n/ridge_n)**2))\n    stations = np.atleast_2d(np.asarray(stations, dtype=float))\n    n_ortho = int(n_ortho); n_max = float(n_max)\n    beta = 1.0/(0.0019872041*300.0)\n    ng = np.linspace(-n_max, n_max, n_ortho)\n    t_st = stations[:, 1]; z_floor = stations[:, 2]\n    speed = stations[:, 3]; curv = stations[:, 4]\n    normal = _fx_local_frame(t_st)\n    xy = _fx_guide(t_st)[:, None, :] + ng[None, :, None]*normal[:, None, :]\n    z = _fx_pcv_coordinates(xy.reshape(-1, 2), images)[:, 1]\n    z = z.reshape(len(t_st), n_ortho)\n    u = _fx_energy(t_st[:, None], ng[None, :])\n    w = np.exp(-beta*(u - u.min()))*np.abs(1.0 - ng[None, :]*curv[:, None])*speed[:, None]\n    return np.stack([z - z_floor[:, None], w], -1)\nslices = _fx_orthogonal_slices(stations, images, n_ortho, n_max)\ndef _fx_conditional_free_energy(slices, n_ortho, n_max, d_eff):\n    slices = np.asarray(slices, dtype=float)\n    n_ortho = int(n_ortho); n_max = float(n_max); d_eff = int(d_eff)\n    beta = 1.0/(0.0019872041*300.0)\n    d_perp = d_eff - 1\n    ng = np.linspace(-n_max, n_max, n_ortho)\n    pos = ng > 0.0; neg = ng < 0.0\n    npos = ng[pos]\n    out = np.empty((slices.shape[0], npos.size, 2))\n    for a in range(slices.shape[0]):\n        dz = slices[a, :, 0]; w = slices[a, :, 1]\n        folded = w[pos] + np.interp(-npos, ng[neg], w[neg])\n        slope = np.gradient(dz, ng)[pos]\n        dens = folded/np.abs(slope)\n        order = np.argsort(dz[pos])\n        dzp = dz[pos][order]; dens = dens[order]\n        with np.errstate(divide="ignore", invalid="ignore"):\n            fhat = -np.log(dens*np.maximum(dzp, 0.0)**((2.0-d_perp)/2.0))/beta\n        out[a, :, 0] = dzp\n        out[a, :, 1] = fhat\n    return out\ncond = _fx_conditional_free_energy(slices, n_ortho, n_max, 2)\ndelta_f_star = 1.0*0.0019872041*300.0\nd_eff = 2\ndz_ref = 8.0e-5\n', 'call': 'adaptive_half_width(cond, slices, delta_f_star, d_eff, dz_ref)', 'gold_call': '_oracle_adaptive_half_width(cond, slices, delta_f_star, d_eff, dz_ref)'},
        {'setup': 'import numpy as np\nimages = _oracle_reference_path(60)\nstations = _oracle_station_frame(images, 11)\nn_ortho = 801\nn_max = 0.20\ndef _fx_orthogonal_slices(stations, images, n_ortho, n_max):\n    def _fx_guide(t):\n        t = np.asarray(t, dtype=float)\n        return np.stack([t, 0.15*np.sin(np.pi*t)], -1)\n    def _fx_guide_d1(t):\n        t = np.asarray(t, dtype=float)\n        return np.stack([np.ones_like(t), 0.15*np.pi*np.cos(np.pi*t)], -1)\n    def _fx_local_frame(t):\n        d1 = _fx_guide_d1(t)\n        sp = np.linalg.norm(d1, axis=-1)\n        tang = d1/sp[..., None]\n        return np.stack([-tang[..., 1], tang[..., 0]], -1)\n    def _fx_lambda_sharpness(imgs):\n        imgs = np.atleast_2d(np.asarray(imgs, dtype=float))\n        return 2.3/float(np.mean(np.sum(np.diff(imgs, axis=0)**2, axis=1)))\n    def _fx_pcv_coordinates(points, imgs):\n        points = np.atleast_2d(np.asarray(points, dtype=float))\n        imgs = np.atleast_2d(np.asarray(imgs, dtype=float))\n        lam_sharp = _fx_lambda_sharpness(imgs)\n        M = imgs.shape[0]\n        idx = np.arange(M, dtype=float)\n        out = np.empty((points.shape[0], 2))\n        step = 60000\n        for a in range(0, points.shape[0], step):\n            p = points[a:a+step]\n            d = (p[:, None, 0]-imgs[None, :, 0])**2 + (p[:, None, 1]-imgs[None, :, 1])**2\n            dmin = d.min(1, keepdims=True)\n            w = np.exp(-lam_sharp*(d-dmin))\n            tot = w.sum(1)\n            out[a:a+step, 0] = (w @ idx)/tot/(M-1)\n            out[a:a+step, 1] = -(np.log(tot) - lam_sharp*dmin[:, 0])/lam_sharp\n        return out\n    def _fx_energy(t, n):\n        t = np.asarray(t, dtype=float); n = np.asarray(n, dtype=float)\n        k_soft, k_stiff, w_neck, t_sad1, t_sad2 = 40.0, 400.0, 0.055, 0.30, 0.70\n        stiffness = k_soft + (k_stiff-k_soft)*(np.exp(-((t-t_sad1)/w_neck)**2) + np.exp(-((t-t_sad2)/w_neck)**2))\n        along = (7.5*np.exp(-((t-t_sad1)/0.10)**2) + 6.2*np.exp(-((t-t_sad2)/0.10)**2) - 1.6*np.exp(-((t-0.50)/0.11)**2))\n        cubic, quart, quart_m, ridge, ridge_t, ridge_n = 30.0, 140.0, 0.60, 0.35, 0.10, 0.055\n        return (along + 0.5*stiffness*n**2 + cubic*np.sin(2.0*np.pi*t)*n**3 + quart*(1.0+quart_m*np.cos(2.0*np.pi*t))*n**4 + ridge*np.exp(-((t-0.50)/ridge_t)**2)*np.exp(-(n/ridge_n)**2))\n    stations = np.atleast_2d(np.asarray(stations, dtype=float))\n    n_ortho = int(n_ortho); n_max = float(n_max)\n    beta = 1.0/(0.0019872041*300.0)\n    ng = np.linspace(-n_max, n_max, n_ortho)\n    t_st = stations[:, 1]; z_floor = stations[:, 2]\n    speed = stations[:, 3]; curv = stations[:, 4]\n    normal = _fx_local_frame(t_st)\n    xy = _fx_guide(t_st)[:, None, :] + ng[None, :, None]*normal[:, None, :]\n    z = _fx_pcv_coordinates(xy.reshape(-1, 2), images)[:, 1]\n    z = z.reshape(len(t_st), n_ortho)\n    u = _fx_energy(t_st[:, None], ng[None, :])\n    w = np.exp(-beta*(u - u.min()))*np.abs(1.0 - ng[None, :]*curv[:, None])*speed[:, None]\n    return np.stack([z - z_floor[:, None], w], -1)\nslices = _fx_orthogonal_slices(stations, images, n_ortho, n_max)\ndef _fx_conditional_free_energy(slices, n_ortho, n_max, d_eff):\n    slices = np.asarray(slices, dtype=float)\n    n_ortho = int(n_ortho); n_max = float(n_max); d_eff = int(d_eff)\n    beta = 1.0/(0.0019872041*300.0)\n    d_perp = d_eff - 1\n    ng = np.linspace(-n_max, n_max, n_ortho)\n    pos = ng > 0.0; neg = ng < 0.0\n    npos = ng[pos]\n    out = np.empty((slices.shape[0], npos.size, 2))\n    for a in range(slices.shape[0]):\n        dz = slices[a, :, 0]; w = slices[a, :, 1]\n        folded = w[pos] + np.interp(-npos, ng[neg], w[neg])\n        slope = np.gradient(dz, ng)[pos]\n        dens = folded/np.abs(slope)\n        order = np.argsort(dz[pos])\n        dzp = dz[pos][order]; dens = dens[order]\n        with np.errstate(divide="ignore", invalid="ignore"):\n            fhat = -np.log(dens*np.maximum(dzp, 0.0)**((2.0-d_perp)/2.0))/beta\n        out[a, :, 0] = dzp\n        out[a, :, 1] = fhat\n    return out\ncond = _fx_conditional_free_energy(slices, n_ortho, n_max, 2)\ndelta_f_star = 4.0*0.0019872041*300.0\nd_eff = 2\ndz_ref = 8.0e-5\n', 'call': 'adaptive_half_width(cond, slices, delta_f_star, d_eff, dz_ref)', 'gold_call': '_oracle_adaptive_half_width(cond, slices, delta_f_star, d_eff, dz_ref)'},
        {'setup': 'import numpy as np\nimages = _oracle_reference_path(200)\nstations = _oracle_station_frame(images, 41)\nn_ortho = 2001\nn_max = 0.34\ndef _fx_orthogonal_slices(stations, images, n_ortho, n_max):\n    def _fx_guide(t):\n        t = np.asarray(t, dtype=float)\n        return np.stack([t, 0.15*np.sin(np.pi*t)], -1)\n    def _fx_guide_d1(t):\n        t = np.asarray(t, dtype=float)\n        return np.stack([np.ones_like(t), 0.15*np.pi*np.cos(np.pi*t)], -1)\n    def _fx_local_frame(t):\n        d1 = _fx_guide_d1(t)\n        sp = np.linalg.norm(d1, axis=-1)\n        tang = d1/sp[..., None]\n        return np.stack([-tang[..., 1], tang[..., 0]], -1)\n    def _fx_lambda_sharpness(imgs):\n        imgs = np.atleast_2d(np.asarray(imgs, dtype=float))\n        return 2.3/float(np.mean(np.sum(np.diff(imgs, axis=0)**2, axis=1)))\n    def _fx_pcv_coordinates(points, imgs):\n        points = np.atleast_2d(np.asarray(points, dtype=float))\n        imgs = np.atleast_2d(np.asarray(imgs, dtype=float))\n        lam_sharp = _fx_lambda_sharpness(imgs)\n        M = imgs.shape[0]\n        idx = np.arange(M, dtype=float)\n        out = np.empty((points.shape[0], 2))\n        step = 60000\n        for a in range(0, points.shape[0], step):\n            p = points[a:a+step]\n            d = (p[:, None, 0]-imgs[None, :, 0])**2 + (p[:, None, 1]-imgs[None, :, 1])**2\n            dmin = d.min(1, keepdims=True)\n            w = np.exp(-lam_sharp*(d-dmin))\n            tot = w.sum(1)\n            out[a:a+step, 0] = (w @ idx)/tot/(M-1)\n            out[a:a+step, 1] = -(np.log(tot) - lam_sharp*dmin[:, 0])/lam_sharp\n        return out\n    def _fx_energy(t, n):\n        t = np.asarray(t, dtype=float); n = np.asarray(n, dtype=float)\n        k_soft, k_stiff, w_neck, t_sad1, t_sad2 = 40.0, 400.0, 0.055, 0.30, 0.70\n        stiffness = k_soft + (k_stiff-k_soft)*(np.exp(-((t-t_sad1)/w_neck)**2) + np.exp(-((t-t_sad2)/w_neck)**2))\n        along = (7.5*np.exp(-((t-t_sad1)/0.10)**2) + 6.2*np.exp(-((t-t_sad2)/0.10)**2) - 1.6*np.exp(-((t-0.50)/0.11)**2))\n        cubic, quart, quart_m, ridge, ridge_t, ridge_n = 30.0, 140.0, 0.60, 0.35, 0.10, 0.055\n        return (along + 0.5*stiffness*n**2 + cubic*np.sin(2.0*np.pi*t)*n**3 + quart*(1.0+quart_m*np.cos(2.0*np.pi*t))*n**4 + ridge*np.exp(-((t-0.50)/ridge_t)**2)*np.exp(-(n/ridge_n)**2))\n    stations = np.atleast_2d(np.asarray(stations, dtype=float))\n    n_ortho = int(n_ortho); n_max = float(n_max)\n    beta = 1.0/(0.0019872041*300.0)\n    ng = np.linspace(-n_max, n_max, n_ortho)\n    t_st = stations[:, 1]; z_floor = stations[:, 2]\n    speed = stations[:, 3]; curv = stations[:, 4]\n    normal = _fx_local_frame(t_st)\n    xy = _fx_guide(t_st)[:, None, :] + ng[None, :, None]*normal[:, None, :]\n    z = _fx_pcv_coordinates(xy.reshape(-1, 2), images)[:, 1]\n    z = z.reshape(len(t_st), n_ortho)\n    u = _fx_energy(t_st[:, None], ng[None, :])\n    w = np.exp(-beta*(u - u.min()))*np.abs(1.0 - ng[None, :]*curv[:, None])*speed[:, None]\n    return np.stack([z - z_floor[:, None], w], -1)\nslices = _fx_orthogonal_slices(stations, images, n_ortho, n_max)\ndef _fx_conditional_free_energy(slices, n_ortho, n_max, d_eff):\n    slices = np.asarray(slices, dtype=float)\n    n_ortho = int(n_ortho); n_max = float(n_max); d_eff = int(d_eff)\n    beta = 1.0/(0.0019872041*300.0)\n    d_perp = d_eff - 1\n    ng = np.linspace(-n_max, n_max, n_ortho)\n    pos = ng > 0.0; neg = ng < 0.0\n    npos = ng[pos]\n    out = np.empty((slices.shape[0], npos.size, 2))\n    for a in range(slices.shape[0]):\n        dz = slices[a, :, 0]; w = slices[a, :, 1]\n        folded = w[pos] + np.interp(-npos, ng[neg], w[neg])\n        slope = np.gradient(dz, ng)[pos]\n        dens = folded/np.abs(slope)\n        order = np.argsort(dz[pos])\n        dzp = dz[pos][order]; dens = dens[order]\n        with np.errstate(divide="ignore", invalid="ignore"):\n            fhat = -np.log(dens*np.maximum(dzp, 0.0)**((2.0-d_perp)/2.0))/beta\n        out[a, :, 0] = dzp\n        out[a, :, 1] = fhat\n    return out\ncond = _fx_conditional_free_energy(slices, n_ortho, n_max, 2)\ndelta_f_star = 3.0*0.0019872041*300.0\nd_eff = 2\ndz_ref = 8.0e-5\n', 'call': 'adaptive_half_width(cond, slices, delta_f_star, d_eff, dz_ref)', 'gold_call': '_oracle_adaptive_half_width(cond, slices, delta_f_star, d_eff, dz_ref)'},
    ]
