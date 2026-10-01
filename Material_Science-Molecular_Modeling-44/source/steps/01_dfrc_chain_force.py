"""
Determine the chain force transmitted by a polymer chain held at a prescribed end-to-end distance with the deformable freely rotating chain (dFRC) model of the elasticity source (its Eqs. 7-9): the chain is represented by a uniform, non-fluctuating bond length l and bond-vector angle phi (the angle between successive bond vectors) chosen to minimize the Helmholtz free energy Psi(l, phi; r) = N v_str(l) + (N - 1) v_ben(phi) + Psi_ent(r; l, phi) at the prescribed end-to-end distance r = stretch_ratio * R_0, where R_0 = N l_e cos(phi_e / 2) is the end-to-end length of the undeformed zigzag, v_str(l) = D_e [1 - exp(-a (l - l_e))]^2 is the Morse stretching energy and v_ben(phi) = k_phi (phi - phi_e)^2 / 2. Psi_ent is the entropic free energy of a freely rotating chain with bond length l and bond angle phi, whose contour length is R_max = N l cos(phi / 2) and whose Kuhn length is l_k = 2 l cos(phi / 2) / (1 - cos phi); with the relative extension r* = r / R_max (0 < r* < 1) and eta = L^-1(r*) the exact inverse of the Langevin function L(x) = coth x - 1/x, the source's explicit force-extension relation (its Eq. 8) is f = (k_B T / l_k) { eta + (1/2) r*^2 / (1 - r*)^2 [1 - r*^(l_k / l - 1)] }, and its closed-form free energy (its Eq. 9) is Psi_ent = k_B T (R_max / l_k) [ r* eta + ln(eta / sinh eta) + (1 + r* (1 - r*)) / (2 (1 - r*)) + ln(1 - r*) - (1/2) B(r*; 2 + l_k / l, -1) ], where B(x; a, b) = int_0^x t^(a - 1) (1 - t)^(b - 1) dt is the incomplete beta function (here b = -1, integrand t^(a - 1) (1 - t)^-2, a = 2 + l_k / l). Eq. 9 is the integral of Eq. 8 over r plus an additive constant that depends on l and phi (through R_max / l_k and l_k / l); that constant must be kept, because it moves the optimum: integrating Eq. 8 from r = 0 instead does not give the source's free energy. Evaluate the inverse Langevin function exactly (no Pade or other approximant) and the incomplete beta function to a relative accuracy of 1e-10 or better for r* close to 1. The chain force is f = dPsi_ent / dr at the optimum, i.e. Eq. 8 evaluated at (l*, phi*). Return the optimal bond length in units of l_e, the optimal bond-vector angle in radians and the reduced chain force f l_e / D_e. When beta_kphi = 0 (freely jointed bonds) the angle is undetermined: use the freely jointed reduction l_k = l, R_max = N l, for which the correction term of Eq. 8 vanishes and Eq. 9 reduces to Psi_ent = N k_B T [r* eta + ln(eta / sinh eta)] up to an l-independent constant; optimize l alone and return 0.0 for the angle. The optimum is unique in the physical domain (r* < 1); converge it to 1e-12 in both variables (bond length in l_e, angle in radians). Finite-difference gradients cannot reach that accuracy next to the r* -> 1 singularity of Psi_ent; use the stationarity conditions with analytic derivatives: Psi_ent depends on l only through r*, so dPsi_ent/dl = -(r / l) f with f from Eq. 8, and dPsi_ent/dphi involves dB(r*; a, -1)/da = int_0^r* t^(a - 1) ln t (1 - t)^-2 dt. Reduced units: lengths in units of the equilibrium bond length l_e, energies in units of the dissociation energy D_e, so beta_de = D_e/(k_B T), a_le = a l_e and f_red = f l_e / D_e; angles in radians.

Single-chain force-extension models with deformable bonds map the stretch of a chain segment in a loaded network onto the force carried by its backbone. A freely rotating chain with effective, stretch-dependent bond length and bond angle captures both the entropic elasticity at low force and the energetic stiffening by bond stretching and bond-angle opening at high force, and its force at a prescribed extension is the input a bond-scission calculation needs.

Returns
-------
tuple of three floats (l_star, phi_star, f_red): optimal bond length in l_e, optimal bond-vector angle in radians (0.0 in the freely jointed case), reduced chain force f l_e / D_e.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def dfrc_chain_force(stretch_ratio: float, n_bonds: int, beta_de: float, a_le: float, beta_kphi: float,
                             phi_e: float) -> tuple:
    """Determine the chain force transmitted by a polymer chain held at a prescribed end-to-end distance with the deformable freely rotating chain (dFRC) model of the elasticity source (its Eqs. 7-9): the chain is represented by a uniform, non-fluctuating bond length l and bond-vector angle phi (the angle between successive bond vectors) chosen to minimize the Helmholtz free energy Psi(l, phi; r) = N v_str(l) + (N - 1) v_ben(phi) + Psi_ent(r; l, phi) at the prescribed end-to-end distance r = stretch_ratio * R_0, where R_0 = N l_e cos(phi_e / 2) is the end-to-end length of the undeformed zigzag, v_str(l) = D_e [1 - exp(-a (l - l_e))]^2 is the Morse stretching energy and v_ben(phi) = k_phi (phi - phi_e)^2 / 2. Psi_ent is the entropic free energy of a freely rotating chain with bond length l and bond angle phi, whose contour length is R_max = N l cos(phi / 2) and whose Kuhn length is l_k = 2 l cos(phi / 2) / (1 - cos phi); with the relative extension r* = r / R_max (0 < r* < 1) and eta = L^-1(r*) the exact inverse of the Langevin function L(x) = coth x - 1/x, the source's explicit force-extension relation (its Eq. 8) is f = (k_B T / l_k) { eta + (1/2) r*^2 / (1 - r*)^2 [1 - r*^(l_k / l - 1)] }, and its closed-form free energy (its Eq. 9) is Psi_ent = k_B T (R_max / l_k) [ r* eta + ln(eta / sinh eta) + (1 + r* (1 - r*)) / (2 (1 - r*)) + ln(1 - r*) - (1/2) B(r*; 2 + l_k / l, -1) ], where B(x; a, b) = int_0^x t^(a - 1) (1 - t)^(b - 1) dt is the incomplete beta function (here b = -1, integrand t^(a - 1) (1 - t)^-2, a = 2 + l_k / l). Eq. 9 is the integral of Eq. 8 over r plus an additive constant that depends on l and phi (through R_max / l_k and l_k / l); that constant must be kept, because it moves the optimum: integrating Eq. 8 from r = 0 instead does not give the source's free energy. Evaluate the inverse Langevin function exactly (no Pade or other approximant) and the incomplete beta function to a relative accuracy of 1e-10 or better for r* close to 1. The chain force is f = dPsi_ent / dr at the optimum, i.e. Eq. 8 evaluated at (l*, phi*). Return the optimal bond length in units of l_e, the optimal bond-vector angle in radians and the reduced chain force f l_e / D_e. When beta_kphi = 0 (freely jointed bonds) the angle is undetermined: use the freely jointed reduction l_k = l, R_max = N l, for which the correction term of Eq. 8 vanishes and Eq. 9 reduces to Psi_ent = N k_B T [r* eta + ln(eta / sinh eta)] up to an l-independent constant; optimize l alone and return 0.0 for the angle. The optimum is unique in the physical domain (r* < 1); converge it to 1e-12 in both variables (bond length in l_e, angle in radians). Finite-difference gradients cannot reach that accuracy next to the r* -> 1 singularity of Psi_ent; use the stationarity conditions with analytic derivatives: Psi_ent depends on l only through r*, so dPsi_ent/dl = -(r / l) f with f from Eq. 8, and dPsi_ent/dphi involves dB(r*; a, -1)/da = int_0^r* t^(a - 1) ln t (1 - t)^-2 dt. Reduced units: lengths in units of the equilibrium bond length l_e, energies in units of the dissociation energy D_e, so beta_de = D_e/(k_B T), a_le = a l_e and f_red = f l_e / D_e; angles in radians.

    Parameters
    ----------
    stretch_ratio : float
        Prescribed end-to-end distance divided by R_0 = N l_e cos(phi_e / 2) (positive).
    n_bonds : int
        Number of bonds N (>= 2).
    beta_de : float
        Reduced dissociation energy D_e / (k_B T).
    a_le : float
        Reduced Morse range parameter a l_e.
    beta_kphi : float
        Bending stiffness in units of k_B T per rad^2 (non-negative).
    phi_e : float
        Equilibrium angle between successive bond vectors in radians, strictly inside (0, pi).

    Returns
    -------
    result : tuple
        (l_star, phi_star, f_red) as native floats.

    Raises
    ------
    ValueError
        If stretch_ratio, beta_de or a_le is not finite positive, n_bonds is not an integer >= 2, beta_kphi is negative or not finite, phi_e is outside (0, pi), or no physical optimum exists for the supplied stretch ratio.
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
from scipy.optimize import brentq, minimize


def _check_positive(name, value):
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be a finite positive number")


def _morse(l, a_le):
    e = np.exp(-a_le * (l - 1.0))
    return (1.0 - e) ** 2


def _morse_prime(l, a_le):
    e = np.exp(-a_le * (l - 1.0))
    return 2.0 * a_le * e * (1.0 - e)


def _langevin_inverse(y):
    """Inverse Langevin function L^-1(y) for 0 < y < 1 by root bracketing."""
    hi = 10.0 / (1.0 - y) + 10.0
    return brentq(lambda b: 1.0 / np.tanh(b) - 1.0 / b - y, 1e-9, hi, xtol=1e-14, rtol=4.0 * np.finfo(float).eps)


def _frc_geometry(l, phi, n_bonds):
    """Contour length R_max = N l cos(phi/2) and Kuhn length l_k = 2 l cos(phi/2) / (1 - cos phi) of a FRC."""
    return n_bonds * l * np.cos(0.5 * phi), 2.0 * l * np.cos(0.5 * phi) / (1.0 - np.cos(phi))


def _incomplete_beta_minus_one(x, a, n_nodes=80):
    """B(x; a, -1) = int_0^x t^(a-1) (1-t)^-2 dt for 0 < x < 1, by parts: x^a/(1-x) - (a-1) int_0^x t^(a-1)/(1-t) dt,
    with the remaining log-singular integral split as -ln(1-x) - int_0^x (1 - t^(a-1))/(1-t) dt (smooth, Gauss-Legendre)."""
    xs, ws = np.polynomial.legendre.leggauss(n_nodes)
    t = 0.5 * x * (xs + 1.0)
    smooth = np.sum(0.5 * x * ws * (1.0 - t ** (a - 1.0)) / (1.0 - t))
    return x ** a / (1.0 - x) - (a - 1.0) * (-np.log1p(-x) - smooth)


def _incomplete_beta_minus_one_da(x, a, n_nodes=80):
    """d/da B(x; a, -1) = int_0^x t^(a-1) ln t (1-t)^-2 dt for 0 < x < 1, by parts: x^(a-1) ln x / (1-x) - int_0^x g(t)/(1-t) dt
    with g(t) = t^(a-2) [(a-1) ln t + 1], g(1) = 1, the log-singular remainder split as -ln(1-x) + int_0^x (g(t) - 1)/(1-t) dt."""
    xs, ws = np.polynomial.legendre.leggauss(n_nodes)
    t = 0.5 * x * (xs + 1.0)
    g = t ** (a - 2.0) * ((a - 1.0) * np.log(t) + 1.0)
    smooth = np.sum(0.5 * x * ws * (g - 1.0) / (1.0 - t))
    return x ** (a - 1.0) * np.log(x) / (1.0 - x) - (-np.log1p(-x) + smooth)


def _frc_force(r, l, phi, n_bonds, k_t):
    """Explicit FRC force-extension relation of the elasticity source (its Eq. 8), energies in D_e."""
    r_max, l_k = _frc_geometry(l, phi, n_bonds)
    rs = r / r_max
    b = _langevin_inverse(rs)
    return k_t / l_k * (b + 0.5 * rs ** 2 / (1.0 - rs) ** 2 * (1.0 - rs ** (l_k / l - 1.0)))


def _frc_free_energy(r, l, phi, n_bonds, k_t):
    """Entropic free energy of the FRC (elasticity source, Eq. 9); 1e6 outside the physical domain."""
    r_max, l_k = _frc_geometry(l, phi, n_bonds)
    rs = r / r_max
    if not (1e-9 < rs < 1.0 - 1e-7):
        return 1e6
    b = _langevin_inverse(rs)
    log_sinh = b + np.log1p(-np.exp(-2.0 * b)) - np.log(2.0)
    inc_beta = _incomplete_beta_minus_one(rs, 2.0 + l_k / l)                    # B(rs; 2 + l_k/l, -1)
    return k_t * r_max / l_k * (rs * b + np.log(b) - log_sinh + (1.0 + rs * (1.0 - rs)) / (2.0 * (1.0 - rs))
                                + np.log(1.0 - rs) - 0.5 * inc_beta)


def _dfrc_total(p, r, n_bonds, beta_de, a_le, beta_kphi, phi_e):
    """Helmholtz free energy Psi(l, phi; r) of the dFRC with Morse bonds and harmonic bending, in D_e."""
    l, phi = float(p[0]), float(p[1])
    if l <= 0.5 or l > 3.0 or phi <= 0.05 or phi >= np.pi - 0.05:
        return 1e6
    k_t = 1.0 / beta_de
    return (n_bonds * _morse(l, a_le) + (n_bonds - 1) * 0.5 * beta_kphi * k_t * (phi - phi_e) ** 2
            + _frc_free_energy(r, l, phi, n_bonds, k_t))


def _dfrc_gradient(p, r, n_bonds, beta_de, a_le, beta_kphi, phi_e):
    """Analytic gradient (dPsi/dl, dPsi/dphi) of the dFRC free energy, in D_e per l_e and D_e per radian. Psi_ent depends on
    l only through r* = r / R_max, so dPsi_ent/dl = -(r / l) f with f from Eq. 8; the phi-derivative also needs
    d(l_k/l)/dphi and the derivative of the incomplete beta function with respect to its first parameter."""
    l, phi = float(p[0]), float(p[1])
    k_t = 1.0 / beta_de
    r_max, l_k = _frc_geometry(l, phi, n_bonds)
    x = r / r_max
    if l <= 0.5 or l > 3.0 or phi <= 0.05 or phi >= np.pi - 0.05 or not (1e-9 < x < 1.0 - 1e-7):
        return [1e6, 1e6]
    eta = _langevin_inverse(x)
    a = 2.0 + l_k / l                                                          # incomplete-beta parameter, a function of phi only
    g_x = eta + 0.5 * x ** 2 / (1.0 - x) ** 2 * (1.0 - x ** (a - 3.0))          # dG/dr* = f l_k / (k_B T), Eq. 8
    f = k_t / l_k * g_x
    log_sinh = eta + np.log1p(-np.exp(-2.0 * eta)) - np.log(2.0)
    g_val = (x * eta + np.log(eta) - log_sinh + (1.0 + x * (1.0 - x)) / (2.0 * (1.0 - x)) + np.log1p(-x)
             - 0.5 * _incomplete_beta_minus_one(x, a))                          # Psi_ent = k_B T (R_max / l_k) G
    g_a = -0.5 * _incomplete_beta_minus_one_da(x, a)
    prefac, dprefac = 0.5 * n_bonds * (1.0 - np.cos(phi)), 0.5 * n_bonds * np.sin(phi)   # R_max / l_k and its phi-derivative
    dx_dphi = 0.5 * x * np.tan(0.5 * phi)
    c, sn = np.cos(0.5 * phi), np.sin(0.5 * phi)
    da_dphi = (-sn * (1.0 - np.cos(phi)) - 2.0 * c * np.sin(phi)) / (1.0 - np.cos(phi)) ** 2
    dpsi_dl = n_bonds * _morse_prime(l, a_le) - (r / l) * f
    dpsi_dphi = ((n_bonds - 1) * beta_kphi * k_t * (phi - phi_e)
                 + k_t * (dprefac * g_val + prefac * (g_x * dx_dphi + g_a * da_dphi)))
    return [dpsi_dl, dpsi_dphi]


def _newton_polish(gradient, p0, args, n_iter=12, h=1e-6):
    """Newton iterations on a stationarity system with a central-difference Jacobian of the analytic gradient; returns the
    point and the size of the last step (a converged optimum has a last step far below 1e-12)."""
    p = np.array(p0, dtype=float)
    step = np.inf
    for _ in range(n_iter):
        g0 = np.array(gradient(p, *args))
        jac = np.zeros((p.size, p.size))
        for k in range(p.size):
            d = np.zeros(p.size)
            d[k] = h
            jac[:, k] = (np.array(gradient(p + d, *args)) - np.array(gradient(p - d, *args))) / (2.0 * h)
        delta = np.linalg.solve(jac, -g0)
        p = p + delta
        step = float(np.max(np.abs(delta)))
        if step < 1e-15:
            break
    return p, step


def _fjc_total(p, r, n_bonds, beta_de, a_le):
    """Free energy of the extensible freely jointed reduction of the dFRC (l_k = l, R_max = N l), in D_e."""
    l = float(p[0]) if np.ndim(p) else float(p)
    rs = r / (n_bonds * l)
    if l <= 0.5 or l > 3.0 or not (1e-9 < rs < 1.0 - 1e-7):
        return 1e6
    b = _langevin_inverse(rs)
    log_sinh = b + np.log1p(-np.exp(-2.0 * b)) - np.log(2.0)
    return n_bonds * _morse(l, a_le) + n_bonds / beta_de * (rs * b + np.log(b) - log_sinh)


def _fjc_gradient(p, r, n_bonds, beta_de, a_le):
    """Analytic dPsi/dl of the freely jointed reduction: dPsi_ent/dl = -(r / l) f with f = k_B T eta / l."""
    l = float(p[0]) if np.ndim(p) else float(p)
    rs = r / (n_bonds * l)
    if l <= 0.5 or l > 3.0 or not (1e-9 < rs < 1.0 - 1e-7):
        return [1e6]
    eta = _langevin_inverse(rs)
    return [n_bonds * _morse_prime(l, a_le) - (r / l) * eta / (beta_de * l)]


def _oracle_dfrc_chain_force(stretch_ratio: float, n_bonds: int, beta_de: float, a_le: float, beta_kphi: float,
                             phi_e: float) -> tuple:
    for name, value in (("stretch_ratio", stretch_ratio), ("beta_de", beta_de), ("a_le", a_le)):
        _check_positive(name, value)
    if not isinstance(n_bonds, (int, np.integer)) or n_bonds < 2:
        raise ValueError("n_bonds must be an integer >= 2")
    if not np.isfinite(beta_kphi) or beta_kphi < 0.0:
        raise ValueError("beta_kphi must be finite and non-negative")
    if not np.isfinite(phi_e) or phi_e <= 0.0 or phi_e >= np.pi:
        raise ValueError("phi_e must lie strictly inside (0, pi)")
    k_t = 1.0 / beta_de
    r = stretch_ratio * n_bonds * np.cos(0.5 * phi_e)                      # r in units of l_e, R_0 = N l_e cos(phi_e/2)
    if beta_kphi == 0.0:                                                      # freely jointed limit: l_k = l, R_max = N l
        fj_args = (r, int(n_bonds), beta_de, a_le)
        res = minimize(_fjc_total, x0=[max(1.0, r / n_bonds * 1.001)], args=fj_args, method="Nelder-Mead",
                       options=dict(xatol=1e-7, fatol=1e-12, maxiter=4000))
        sol, step = _newton_polish(_fjc_gradient, res.x, fj_args)
        if not step < 1e-12:
            raise ValueError("the freely jointed optimum did not converge to 1e-12")
        l_star = float(sol[0])
        rs = r / (n_bonds * l_star)
        return l_star, 0.0, float(k_t / l_star * _langevin_inverse(rs))
    args = (r, int(n_bonds), beta_de, a_le, beta_kphi, phi_e)
    start = [max(1.0, 1.1 * stretch_ratio), phi_e]                             # feasible start (r* < 1)
    res = minimize(_dfrc_total, x0=start, args=args, method="Nelder-Mead",
                   options=dict(xatol=1e-7, fatol=1e-12, maxiter=4000))          # locate the basin
    sol, step = _newton_polish(_dfrc_gradient, res.x, args)                    # Newton on the analytic gradient: 1e-12 in both variables
    if not step < 1e-12:
        raise ValueError("the dFRC optimum did not converge to 1e-12 in both variables")
    l_star, phi_star = float(sol[0]), float(sol[1])
    if _dfrc_total(sol, *args) >= 1e5:
        raise ValueError("no physical dFRC optimum for the supplied stretch ratio")
    return l_star, phi_star, float(_frc_force(r, l_star, phi_star, int(n_bonds), k_t))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return the list of test case specifications."""
    return [
        {
            "setup": "import numpy as np\nstretch_ratio, n_bonds = 1.22, 12\nbeta_de, a_le, beta_kphi = 279.19347375490736, 2.1487249999999998, 123.53693528978204\nphi_e = np.deg2rad(70.5)\n",
            "call": "dfrc_chain_force(stretch_ratio, n_bonds, beta_de, a_le, beta_kphi, phi_e)",
            "gold_call": "_oracle_dfrc_chain_force(stretch_ratio, n_bonds, beta_de, a_le, beta_kphi, phi_e)",
            "tol": 1e-09,
        },
        {
            "setup": "import numpy as np\nstretch_ratio, n_bonds = 0.9, 12\nbeta_de, a_le, beta_kphi = 279.19347375490736, 2.1487249999999998, 123.53693528978204\nphi_e = np.deg2rad(70.5)\n",
            "call": "dfrc_chain_force(stretch_ratio, n_bonds, beta_de, a_le, beta_kphi, phi_e)",
            "gold_call": "_oracle_dfrc_chain_force(stretch_ratio, n_bonds, beta_de, a_le, beta_kphi, phi_e)",
            "tol": 1e-09,
        },
        {
            "setup": "import numpy as np\nstretch_ratio, n_bonds = 1.40, 12\nbeta_de, a_le, beta_kphi = 279.19347375490736, 2.1487249999999998, 123.53693528978204\nphi_e = np.deg2rad(70.5)\n",
            "call": "dfrc_chain_force(stretch_ratio, n_bonds, beta_de, a_le, beta_kphi, phi_e)",
            "gold_call": "_oracle_dfrc_chain_force(stretch_ratio, n_bonds, beta_de, a_le, beta_kphi, phi_e)",
            "tol": 1e-09,
        },
        {
            "setup": "import numpy as np\nstretch_ratio, n_bonds = 1.02, 12\nbeta_de, a_le, beta_kphi = 279.19347375490736, 2.1487249999999998, 123.53693528978204\nphi_e = np.deg2rad(70.5)\nbeta_kphi = 0.0\n",
            "call": "dfrc_chain_force(stretch_ratio, n_bonds, beta_de, a_le, beta_kphi, phi_e)",
            "gold_call": "_oracle_dfrc_chain_force(stretch_ratio, n_bonds, beta_de, a_le, beta_kphi, phi_e)",
            "tol": 1e-09,
        },
        {
            "setup": "import numpy as np\nstretch_ratio, n_bonds = 1.22, 12\nbeta_de, a_le, beta_kphi = 279.19347375490736, 2.1487249999999998, 123.53693528978204\nphi_e = np.deg2rad(70.5)\ndef run_model():\n    try:\n        dfrc_chain_force(stretch_ratio, 1, beta_de, a_le, beta_kphi, phi_e)\n        return 0\n    except ValueError:\n        return 1\ndef run_oracle():\n    try:\n        _oracle_dfrc_chain_force(stretch_ratio, 1, beta_de, a_le, beta_kphi, phi_e)\n        return 0\n    except ValueError:\n        return 1\n",
            "call": "run_model()",
            "gold_call": "run_oracle()",
        },
    ]
