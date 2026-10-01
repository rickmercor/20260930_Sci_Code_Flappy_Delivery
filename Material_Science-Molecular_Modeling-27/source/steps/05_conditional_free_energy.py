"""
Turn each orthogonal slice into the conditional free energy of the offset coordinate that the contour condition is applied to. Three things have to happen and the source fixes all three. The offset coordinate cannot tell the two sides of the path apart, so the two branches of the slice have to be brought together before anything is measured: at every positive normal offset add the statistical weight of the configuration at the opposite offset (minus n) to the weight at plus n, and attach the sum to the offset coordinate of the positive branch. The density is wanted in the offset coordinate while the slice is sampled in the normal offset, so the change of variables between them has to be carried through: divide the folded weight by the magnitude of the derivative of the offset coordinate with respect to the normal offset along the positive branch, taken by centred finite differences on the offset grid. And a histogram in the offset coordinate carries a coordinate-volume contribution that the source removes with an explicit metric correction whose exponent is fixed by the effective perpendicular dimensionality of the problem, itself fixed by the dimensionality of the collective-variable space. Return the positive branch sorted by increasing offset, with the free energy on an absolute scale in kcal/mol.

The contour that defines the tube is a contour of the physical orthogonal free energy, not of whatever a raw histogram of the sampled coordinate happens to look like. Those two differ by a Jacobian, and on a path variable the difference is not a constant: it diverges at the path. The source derives the correction and states where it comes from; without it the contour is chased in the wrong variable and the tube it produces is not the one the theory is about.

Returns
-------
A (n_stations, (n_ortho - 1)//2, 2) float64 array whose last axis holds [offset coordinate, conditional free energy in kcal/mol], sorted by increasing offset within each station.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def conditional_free_energy(slices: "np.ndarray", n_ortho: int, n_max: float, d_eff: int) -> "np.ndarray":
    """Turn each orthogonal slice into the conditional free energy of the offset coordinate that the
    contour condition is applied to. Three things have to happen and the source fixes all three. The
    offset coordinate cannot tell the two sides of the path apart, so the two branches of the slice
    have to be brought together before anything is measured: at every positive normal offset add the
    statistical weight of the configuration at the opposite offset (minus n) to the weight at plus
    n, and attach the sum to the offset coordinate of the positive branch. The density is wanted in
    the offset coordinate while the slice is sampled in the normal offset, so the change of
    variables between them has to be carried through: divide the folded weight by the magnitude of
    the derivative of the offset coordinate with respect to the normal offset along the positive
    branch, taken by centred finite differences on the offset grid. And a histogram in the offset
    coordinate carries a coordinate-volume contribution that the source removes with an explicit
    metric correction whose exponent is fixed by the effective perpendicular dimensionality of the
    problem, itself fixed by the dimensionality of the collective-variable space. Return the
    positive branch sorted by increasing offset, with the free energy on an absolute scale in
    kcal/mol.

    Args:
        slices: array-like of shape (n_stations, n_ortho, 2), the slices returned by
            orthogonal_slices.
        n_ortho: int, an odd number of at least 3, the number of uniformly spaced signed normal
            offsets over the sampled window, endpoints included.
        n_max: float, the positive half-extent of the sampled window in Angstrom (offsets run from
            -n_max to n_max).
        d_eff: int, the dimensionality of the collective-variable space (at least 2); the number of
            perpendicular directions is d_eff - 1.

    Returns:
        A (n_stations, (n_ortho - 1)//2, 2) float64 array whose last axis holds [offset coordinate,
        conditional free energy in kcal/mol], sorted by increasing offset within each station.

    Raises:
        ValueError: if slices does not have shape (n_stations, n_ortho, 2), or if d_eff is smaller
        than 2.
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

def _oracle_conditional_free_energy(slices: "np.ndarray", n_ortho: int, n_max: float, d_eff: int) -> "np.ndarray":
    slices = np.asarray(slices, dtype=float)
    n_ortho = int(n_ortho); n_max = float(n_max); d_eff = int(d_eff)
    if slices.ndim != 3 or slices.shape[2] != 2:
        raise ValueError("slices must have shape (n_stations, n_ortho, 2)")
    if d_eff < 2:
        raise ValueError("d_eff must be at least 2")
    d_perp = d_eff - 1
    ng = np.linspace(-n_max, n_max, n_ortho)
    pos = ng > 0.0; neg = ng < 0.0
    npos = ng[pos]
    out = np.empty((slices.shape[0], npos.size, 2))
    for a in range(slices.shape[0]):
        dz = slices[a, :, 0]; w = slices[a, :, 1]
        folded = w[pos] + np.interp(-npos, ng[neg], w[neg])
        slope = np.gradient(dz, ng)[pos]
        dens = folded/np.abs(slope)
        order = np.argsort(dz[pos])
        dzp = dz[pos][order]; dens = dens[order]
        with np.errstate(divide="ignore", invalid="ignore"):
            fhat = -np.log(dens*np.maximum(dzp, 0.0)**((2.0-d_perp)/2.0))/_BETA()
        out[a, :, 0] = dzp
        out[a, :, 1] = fhat
    return out

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {'setup': 'import numpy as np\nimages = _oracle_reference_path(60)\nstations = _oracle_station_frame(images, 11)\nn_ortho = 801\nn_max = 0.34\ndef _fx_orthogonal_slices(stations, images, n_ortho, n_max):\n    def _fx_guide(t):\n        t = np.asarray(t, dtype=float)\n        return np.stack([t, 0.15*np.sin(np.pi*t)], -1)\n    def _fx_guide_d1(t):\n        t = np.asarray(t, dtype=float)\n        return np.stack([np.ones_like(t), 0.15*np.pi*np.cos(np.pi*t)], -1)\n    def _fx_local_frame(t):\n        d1 = _fx_guide_d1(t)\n        sp = np.linalg.norm(d1, axis=-1)\n        tang = d1/sp[..., None]\n        return np.stack([-tang[..., 1], tang[..., 0]], -1)\n    def _fx_lambda_sharpness(imgs):\n        imgs = np.atleast_2d(np.asarray(imgs, dtype=float))\n        return 2.3/float(np.mean(np.sum(np.diff(imgs, axis=0)**2, axis=1)))\n    def _fx_pcv_coordinates(points, imgs):\n        points = np.atleast_2d(np.asarray(points, dtype=float))\n        imgs = np.atleast_2d(np.asarray(imgs, dtype=float))\n        lam_sharp = _fx_lambda_sharpness(imgs)\n        M = imgs.shape[0]\n        idx = np.arange(M, dtype=float)\n        out = np.empty((points.shape[0], 2))\n        step = 60000\n        for a in range(0, points.shape[0], step):\n            p = points[a:a+step]\n            d = (p[:, None, 0]-imgs[None, :, 0])**2 + (p[:, None, 1]-imgs[None, :, 1])**2\n            dmin = d.min(1, keepdims=True)\n            w = np.exp(-lam_sharp*(d-dmin))\n            tot = w.sum(1)\n            out[a:a+step, 0] = (w @ idx)/tot/(M-1)\n            out[a:a+step, 1] = -(np.log(tot) - lam_sharp*dmin[:, 0])/lam_sharp\n        return out\n    def _fx_energy(t, n):\n        t = np.asarray(t, dtype=float); n = np.asarray(n, dtype=float)\n        k_soft, k_stiff, w_neck, t_sad1, t_sad2 = 40.0, 400.0, 0.055, 0.30, 0.70\n        stiffness = k_soft + (k_stiff-k_soft)*(np.exp(-((t-t_sad1)/w_neck)**2) + np.exp(-((t-t_sad2)/w_neck)**2))\n        along = (7.5*np.exp(-((t-t_sad1)/0.10)**2) + 6.2*np.exp(-((t-t_sad2)/0.10)**2) - 1.6*np.exp(-((t-0.50)/0.11)**2))\n        cubic, quart, quart_m, ridge, ridge_t, ridge_n = 30.0, 140.0, 0.60, 0.35, 0.10, 0.055\n        return (along + 0.5*stiffness*n**2 + cubic*np.sin(2.0*np.pi*t)*n**3 + quart*(1.0+quart_m*np.cos(2.0*np.pi*t))*n**4 + ridge*np.exp(-((t-0.50)/ridge_t)**2)*np.exp(-(n/ridge_n)**2))\n    stations = np.atleast_2d(np.asarray(stations, dtype=float))\n    n_ortho = int(n_ortho); n_max = float(n_max)\n    beta = 1.0/(0.0019872041*300.0)\n    ng = np.linspace(-n_max, n_max, n_ortho)\n    t_st = stations[:, 1]; z_floor = stations[:, 2]\n    speed = stations[:, 3]; curv = stations[:, 4]\n    normal = _fx_local_frame(t_st)\n    xy = _fx_guide(t_st)[:, None, :] + ng[None, :, None]*normal[:, None, :]\n    z = _fx_pcv_coordinates(xy.reshape(-1, 2), images)[:, 1]\n    z = z.reshape(len(t_st), n_ortho)\n    u = _fx_energy(t_st[:, None], ng[None, :])\n    w = np.exp(-beta*(u - u.min()))*np.abs(1.0 - ng[None, :]*curv[:, None])*speed[:, None]\n    return np.stack([z - z_floor[:, None], w], -1)\nslices = _fx_orthogonal_slices(stations, images, n_ortho, n_max)\nd_eff = 2\n', 'call': 'conditional_free_energy(slices, n_ortho, n_max, d_eff)', 'gold_call': '_oracle_conditional_free_energy(slices, n_ortho, n_max, d_eff)'},
        {'setup': 'import numpy as np\nimages = _oracle_reference_path(200)\nstations = _oracle_station_frame(images, 21)\nn_ortho = 1201\nn_max = 0.25\ndef _fx_orthogonal_slices(stations, images, n_ortho, n_max):\n    def _fx_guide(t):\n        t = np.asarray(t, dtype=float)\n        return np.stack([t, 0.15*np.sin(np.pi*t)], -1)\n    def _fx_guide_d1(t):\n        t = np.asarray(t, dtype=float)\n        return np.stack([np.ones_like(t), 0.15*np.pi*np.cos(np.pi*t)], -1)\n    def _fx_local_frame(t):\n        d1 = _fx_guide_d1(t)\n        sp = np.linalg.norm(d1, axis=-1)\n        tang = d1/sp[..., None]\n        return np.stack([-tang[..., 1], tang[..., 0]], -1)\n    def _fx_lambda_sharpness(imgs):\n        imgs = np.atleast_2d(np.asarray(imgs, dtype=float))\n        return 2.3/float(np.mean(np.sum(np.diff(imgs, axis=0)**2, axis=1)))\n    def _fx_pcv_coordinates(points, imgs):\n        points = np.atleast_2d(np.asarray(points, dtype=float))\n        imgs = np.atleast_2d(np.asarray(imgs, dtype=float))\n        lam_sharp = _fx_lambda_sharpness(imgs)\n        M = imgs.shape[0]\n        idx = np.arange(M, dtype=float)\n        out = np.empty((points.shape[0], 2))\n        step = 60000\n        for a in range(0, points.shape[0], step):\n            p = points[a:a+step]\n            d = (p[:, None, 0]-imgs[None, :, 0])**2 + (p[:, None, 1]-imgs[None, :, 1])**2\n            dmin = d.min(1, keepdims=True)\n            w = np.exp(-lam_sharp*(d-dmin))\n            tot = w.sum(1)\n            out[a:a+step, 0] = (w @ idx)/tot/(M-1)\n            out[a:a+step, 1] = -(np.log(tot) - lam_sharp*dmin[:, 0])/lam_sharp\n        return out\n    def _fx_energy(t, n):\n        t = np.asarray(t, dtype=float); n = np.asarray(n, dtype=float)\n        k_soft, k_stiff, w_neck, t_sad1, t_sad2 = 40.0, 400.0, 0.055, 0.30, 0.70\n        stiffness = k_soft + (k_stiff-k_soft)*(np.exp(-((t-t_sad1)/w_neck)**2) + np.exp(-((t-t_sad2)/w_neck)**2))\n        along = (7.5*np.exp(-((t-t_sad1)/0.10)**2) + 6.2*np.exp(-((t-t_sad2)/0.10)**2) - 1.6*np.exp(-((t-0.50)/0.11)**2))\n        cubic, quart, quart_m, ridge, ridge_t, ridge_n = 30.0, 140.0, 0.60, 0.35, 0.10, 0.055\n        return (along + 0.5*stiffness*n**2 + cubic*np.sin(2.0*np.pi*t)*n**3 + quart*(1.0+quart_m*np.cos(2.0*np.pi*t))*n**4 + ridge*np.exp(-((t-0.50)/ridge_t)**2)*np.exp(-(n/ridge_n)**2))\n    stations = np.atleast_2d(np.asarray(stations, dtype=float))\n    n_ortho = int(n_ortho); n_max = float(n_max)\n    beta = 1.0/(0.0019872041*300.0)\n    ng = np.linspace(-n_max, n_max, n_ortho)\n    t_st = stations[:, 1]; z_floor = stations[:, 2]\n    speed = stations[:, 3]; curv = stations[:, 4]\n    normal = _fx_local_frame(t_st)\n    xy = _fx_guide(t_st)[:, None, :] + ng[None, :, None]*normal[:, None, :]\n    z = _fx_pcv_coordinates(xy.reshape(-1, 2), images)[:, 1]\n    z = z.reshape(len(t_st), n_ortho)\n    u = _fx_energy(t_st[:, None], ng[None, :])\n    w = np.exp(-beta*(u - u.min()))*np.abs(1.0 - ng[None, :]*curv[:, None])*speed[:, None]\n    return np.stack([z - z_floor[:, None], w], -1)\nslices = _fx_orthogonal_slices(stations, images, n_ortho, n_max)\nd_eff = 2\n', 'call': 'conditional_free_energy(slices, n_ortho, n_max, d_eff)', 'gold_call': '_oracle_conditional_free_energy(slices, n_ortho, n_max, d_eff)'},
        {'setup': 'import numpy as np\nimages = _oracle_reference_path(200)\nstations = _oracle_station_frame(images, 41)\nn_ortho = 2001\nn_max = 0.34\ndef _fx_orthogonal_slices(stations, images, n_ortho, n_max):\n    def _fx_guide(t):\n        t = np.asarray(t, dtype=float)\n        return np.stack([t, 0.15*np.sin(np.pi*t)], -1)\n    def _fx_guide_d1(t):\n        t = np.asarray(t, dtype=float)\n        return np.stack([np.ones_like(t), 0.15*np.pi*np.cos(np.pi*t)], -1)\n    def _fx_local_frame(t):\n        d1 = _fx_guide_d1(t)\n        sp = np.linalg.norm(d1, axis=-1)\n        tang = d1/sp[..., None]\n        return np.stack([-tang[..., 1], tang[..., 0]], -1)\n    def _fx_lambda_sharpness(imgs):\n        imgs = np.atleast_2d(np.asarray(imgs, dtype=float))\n        return 2.3/float(np.mean(np.sum(np.diff(imgs, axis=0)**2, axis=1)))\n    def _fx_pcv_coordinates(points, imgs):\n        points = np.atleast_2d(np.asarray(points, dtype=float))\n        imgs = np.atleast_2d(np.asarray(imgs, dtype=float))\n        lam_sharp = _fx_lambda_sharpness(imgs)\n        M = imgs.shape[0]\n        idx = np.arange(M, dtype=float)\n        out = np.empty((points.shape[0], 2))\n        step = 60000\n        for a in range(0, points.shape[0], step):\n            p = points[a:a+step]\n            d = (p[:, None, 0]-imgs[None, :, 0])**2 + (p[:, None, 1]-imgs[None, :, 1])**2\n            dmin = d.min(1, keepdims=True)\n            w = np.exp(-lam_sharp*(d-dmin))\n            tot = w.sum(1)\n            out[a:a+step, 0] = (w @ idx)/tot/(M-1)\n            out[a:a+step, 1] = -(np.log(tot) - lam_sharp*dmin[:, 0])/lam_sharp\n        return out\n    def _fx_energy(t, n):\n        t = np.asarray(t, dtype=float); n = np.asarray(n, dtype=float)\n        k_soft, k_stiff, w_neck, t_sad1, t_sad2 = 40.0, 400.0, 0.055, 0.30, 0.70\n        stiffness = k_soft + (k_stiff-k_soft)*(np.exp(-((t-t_sad1)/w_neck)**2) + np.exp(-((t-t_sad2)/w_neck)**2))\n        along = (7.5*np.exp(-((t-t_sad1)/0.10)**2) + 6.2*np.exp(-((t-t_sad2)/0.10)**2) - 1.6*np.exp(-((t-0.50)/0.11)**2))\n        cubic, quart, quart_m, ridge, ridge_t, ridge_n = 30.0, 140.0, 0.60, 0.35, 0.10, 0.055\n        return (along + 0.5*stiffness*n**2 + cubic*np.sin(2.0*np.pi*t)*n**3 + quart*(1.0+quart_m*np.cos(2.0*np.pi*t))*n**4 + ridge*np.exp(-((t-0.50)/ridge_t)**2)*np.exp(-(n/ridge_n)**2))\n    stations = np.atleast_2d(np.asarray(stations, dtype=float))\n    n_ortho = int(n_ortho); n_max = float(n_max)\n    beta = 1.0/(0.0019872041*300.0)\n    ng = np.linspace(-n_max, n_max, n_ortho)\n    t_st = stations[:, 1]; z_floor = stations[:, 2]\n    speed = stations[:, 3]; curv = stations[:, 4]\n    normal = _fx_local_frame(t_st)\n    xy = _fx_guide(t_st)[:, None, :] + ng[None, :, None]*normal[:, None, :]\n    z = _fx_pcv_coordinates(xy.reshape(-1, 2), images)[:, 1]\n    z = z.reshape(len(t_st), n_ortho)\n    u = _fx_energy(t_st[:, None], ng[None, :])\n    w = np.exp(-beta*(u - u.min()))*np.abs(1.0 - ng[None, :]*curv[:, None])*speed[:, None]\n    return np.stack([z - z_floor[:, None], w], -1)\nslices = _fx_orthogonal_slices(stations, images, n_ortho, n_max)\nd_eff = 3\n', 'call': 'conditional_free_energy(slices, n_ortho, n_max, d_eff)', 'gold_call': '_oracle_conditional_free_energy(slices, n_ortho, n_max, d_eff)'},
    ]
