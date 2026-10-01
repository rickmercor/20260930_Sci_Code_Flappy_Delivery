"""
Locate the forward bias at which the injected-carrier model stops being accurate. Define the deviation delta(V) = 100 (C_analytic(V) / C_exact(V) - 1) in percent, with the closed-form capacitance of the injected-carrier model and the exact drift-diffusion capacitance on the grid of N intervals. Starting from V = 0, march in steps of march_step (V) until delta first reaches the level (percent), then locate the crossing delta(V) = level inside that interval to 1e-12 V (bisection or any bracketing root finder; the deviation grows monotonically with the bias in that interval). Return the threshold voltage V_thr, the remaining effective built-in potential V_bi(V_thr) - V_thr, and the bulk-field parameter z = q [V_bi(V_thr) - V_thr] / (2 kT) at the threshold (the ratio of the bulk potential drop to 2 kT/q that controls the overlap of the two accumulation regions). Raise ValueError if level or march_step is not positive, if the deviation is already above the level at zero bias, or if it never reaches the level before V_bi,0 - 4 kT/q.

The source states that its analytical expressions are valid for voltages a few kT/q below V_bi(V) - 2 eta kT/q; the exact reference locates the actual limit for each device and expresses it through the bulk-field parameter, whose exponential tails set the accuracy of the majority-carrier-only description of the accumulation regions.

Returns
-------
A (3,) float64 array [V_thr in V, V_bi(V_thr) - V_thr in V, z].
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def validity_threshold(level: float, intervals: int, par: "np.ndarray", march_step: float) -> "np.ndarray":
    """Locate the forward bias at which the injected-carrier model stops being accurate. A (3,)
    float64 array [V_thr in V, V_bi(V_thr) - V_thr in V, z].
    Parameters
    ----------
    level : float
        Deviation level in percent that defines the threshold, positive.
    intervals : int
        Number N of grid intervals, an even integer of at least 4.
    par : np.ndarray
        The (14,) parameter vector returned by the parameter step.
    march_step : float
        Bias step in V used to march up from zero bias, positive.

    Raises
    ------
    ValueError
        If par does not have 14 entries, if level or march_step is not
        positive, if N is not an even integer of at least 4, if the deviation
        already exceeds the level at zero bias, or if it never reaches the
        level below the built-in potential.
    """
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


_Q = 1.602176634e-19

_KB = 1.380649e-23

_E0 = 8.8541878128e-12

def _bernoulli(x):
    """B(x) = x / (exp(x) - 1) with a series branch near zero."""
    x = np.asarray(x, dtype=float); out = np.empty_like(x); s = np.abs(x) < 1e-8
    out[s] = 1.0 - x[s] / 2.0 + x[s] ** 2 / 12.0
    xb = x[~s]; out[~s] = xb / np.expm1(xb)
    return out

def _dbernoulli(x):
    """dB/dx with a series branch near zero."""
    x = np.asarray(x, dtype=float); out = np.empty_like(x); s = np.abs(x) < 1e-5
    out[s] = -0.5 + x[s] / 6.0
    xb = x[~s]; ex = np.expm1(xb); out[~s] = (ex - xb * (ex + 1.0)) / ex ** 2
    return out

def _block_tridiagonal_solve(A, B, C, r):
    """Solve the block tridiagonal system with diagonal blocks A[i], super-diagonal B[i] (row i to i+1)
    and sub-diagonal C[i] (row i to i-1); r has shape (M, k). Block Thomas algorithm."""
    M = r.shape[0]
    Ap = A.copy(); rp = r.copy()
    for i in range(1, M):
        L = np.linalg.solve(Ap[i - 1].T, C[i].T).T
        Ap[i] = Ap[i] - L @ B[i - 1]
        rp[i] = rp[i] - L @ rp[i - 1]
    x = np.empty_like(rp)
    x[M - 1] = np.linalg.solve(Ap[M - 1], rp[M - 1])
    for i in range(M - 2, -1, -1):
        x[i] = np.linalg.solve(Ap[i], rp[i] - B[i] @ x[i + 1])
    return x

def _dd_unpack(state, prm, V):
    """Full node arrays (psi in V, n and p in 1/m^3) from the interior state (M, 3) = [psi/VT, ln n, ln p]."""
    VT, Vbi0, n_an, p_an, n_cat, p_cat = prm["VT"], prm["Vbi0"], prm["n_an"], prm["p_an"], prm["n_cat"], prm["p_cat"]
    psi = np.concatenate([[0.0], state[:, 0] * VT, [Vbi0 - V]])
    n = np.concatenate([[n_an], np.exp(state[:, 1]), [n_cat]])
    p = np.concatenate([[p_an], np.exp(state[:, 2]), [p_cat]])
    return psi, n, p

def _dd_fluxes(psi, n, p, prm, h):
    dl = np.diff(psi) / prm["VT"]; Bp = _bernoulli(dl); Bm = _bernoulli(-dl)
    Jn = _Q * prm["Dn"] / h * (n[1:] * Bp - n[:-1] * Bm)
    Jp = _Q * prm["Dp"] / h * (p[:-1] * Bp - p[1:] * Bm)
    return Jn, Jp

def _dd_residual(state, prm, V, h):
    """Scaled residuals (M, 3): Poisson in units of VT, continuity per carrier at each node."""
    psi, n, p = _dd_unpack(state, prm, V); VT = prm["VT"]; ee = prm["ee"]
    Jn, Jp = _dd_fluxes(psi, n, p, prm, h)
    R = prm["gamma"] * (n * p - prm["ni2"])
    Fpsi = ((psi[2:] - 2.0 * psi[1:-1] + psi[:-2]) / h ** 2 + _Q / ee * (p[1:-1] - n[1:-1])) * h ** 2 / VT
    Fn = ((Jn[1:] - Jn[:-1]) / h - _Q * R[1:-1]) * h ** 2 / (_Q * prm["Dn"] * n[1:-1])
    Fp = ((Jp[1:] - Jp[:-1]) / h + _Q * R[1:-1]) * h ** 2 / (_Q * prm["Dp"] * p[1:-1])
    return np.stack([Fpsi, Fn, Fp], axis=1)

def _dd_jacobian(state, prm, V, h):
    """Blocks (M,3,3) of the Jacobian of the scaled residual with respect to [psi/VT, ln n, ln p], plus dF/dV (M,3)."""
    psi, n, p = _dd_unpack(state, prm, V); VT = prm["VT"]; ee = prm["ee"]; M = state.shape[0]; N = M + 1
    dl = np.diff(psi) / VT; Bp = _bernoulli(dl); Bm = _bernoulli(-dl); dBp = _dbernoulli(dl); dBm = -_dbernoulli(-dl)
    c = _Q * h ** 2 / (ee * VT); an = h ** 2 * prm["gamma"] / prm["Dn"]; ap = h ** 2 * prm["gamma"] / prm["Dp"]
    i = np.arange(1, N); ni = n[i]; pi_ = p[i]; nip = n[i + 1]; nim = n[i - 1]; pip = p[i + 1]; pim = p[i - 1]
    A = np.zeros((M, 3, 3)); B = np.zeros((M, 3, 3)); C = np.zeros((M, 3, 3)); dV = np.zeros((M, 3))
    # Poisson row: (psi_{i+1} - 2 psi_i + psi_{i-1})/VT + c (p_i - n_i); unknown psi/VT
    A[:, 0, 0] = -2.0; A[:, 0, 1] = -c * ni; A[:, 0, 2] = c * pi_; B[:, 0, 0] = 1.0; C[:, 0, 0] = 1.0
    dV[-1, 0] = -1.0 / VT
    # electron row divided by n_i
    fn = (nip * Bp[i] - ni * Bm[i]) - (ni * Bp[i - 1] - nim * Bm[i - 1]) - an * (ni * pi_ - prm["ni2"])
    A[:, 1, 1] = (-Bm[i] - Bp[i - 1]) - an * pi_ - fn / ni
    A[:, 1, 2] = -an * pi_
    B[:, 1, 1] = Bp[i] * nip / ni; C[:, 1, 1] = Bm[i - 1] * nim / ni
    dli = (nip * dBp[i] - ni * dBm[i]) / ni; dlim = -(ni * dBp[i - 1] - nim * dBm[i - 1]) / ni
    A[:, 1, 0] = -dli + dlim; B[:, 1, 0] = dli; C[:, 1, 0] = -dlim
    dV[-1, 1] = -dli[-1] / VT
    # hole row divided by p_i
    fp = (pi_ * Bp[i] - pip * Bm[i]) - (pim * Bp[i - 1] - pi_ * Bm[i - 1]) + ap * (ni * pi_ - prm["ni2"])
    A[:, 2, 2] = (Bp[i] + Bm[i - 1]) + ap * ni - fp / pi_
    A[:, 2, 1] = ap * ni
    B[:, 2, 2] = -Bm[i] * pip / pi_; C[:, 2, 2] = -Bp[i - 1] * pim / pi_
    dpi = (pi_ * dBp[i] - pip * dBm[i]) / pi_; dpim = -(pim * dBp[i - 1] - pi_ * dBm[i - 1]) / pi_
    A[:, 2, 0] = -dpi + dpim; B[:, 2, 0] = dpi; C[:, 2, 0] = -dpim
    dV[-1, 2] = -dpi[-1] / VT
    return A, B, C, dV

def _dd_params(par):
    """Dictionary of SI quantities from the parameter vector of the first step."""
    VT, Vbi0, cgeo, lam_an, lam_cat, ln_pan, ln_ncat, ln_nan, ln_pcat, gam, d_nm, eps, mun, mup = par
    d = d_nm * 1e-9
    return dict(VT=VT, Vbi0=Vbi0, d=d, ee=eps * _E0, Cgeo=eps * _E0 / d, p_an=np.exp(ln_pan), n_cat=np.exp(ln_ncat),
                n_an=np.exp(ln_nan), p_cat=np.exp(ln_pcat), ni2=np.exp(ln_nan + ln_pan), gamma=gam * 1e-18,
                Dn=mun * VT, Dp=mup * VT, lam_an=lam_an * 1e-9, lam_cat=lam_cat * 1e-9)

def _dd_equilibrium(prm, N):
    """Zero-bias state: Poisson-Boltzmann solution (Newton with a line search) in the state layout."""
    VT = prm["VT"]; d = prm["d"]; h = d / N; M = N - 1; c = _Q * h ** 2 / (prm["ee"] * VT)
    psi = np.linspace(0.0, prm["Vbi0"], N + 1)
    def res(ps):
        n = prm["n_an"] * np.exp(ps / VT); p = prm["p_an"] * np.exp(-ps / VT)
        return (ps[2:] - 2.0 * ps[1:-1] + ps[:-2]) / VT + c * (p[1:-1] - n[1:-1]), n, p
    for it in range(300):
        F, n, p = res(psi); nrm = np.max(np.abs(F))
        if nrm < 1e-14: break
        A = ((-2.0 - c * (p[1:-1] + n[1:-1])) / VT).reshape(M, 1, 1)
        Bq = np.full((M, 1, 1), 1.0 / VT); Cq = np.full((M, 1, 1), 1.0 / VT)
        dpsi = _block_tridiagonal_solve(A, Bq, Cq, -F.reshape(M, 1))[:, 0]
        lam = 1.0
        for _ in range(60):
            pt = psi.copy(); pt[1:-1] += lam * dpsi
            with np.errstate(over="ignore", invalid="ignore"):
                Ft = res(pt)[0]
            if np.all(np.isfinite(Ft)) and np.max(np.abs(Ft)) < nrm: break
            lam *= 0.5
        psi = pt
    n = prm["n_an"] * np.exp(psi / VT); p = prm["p_an"] * np.exp(-psi / VT)
    return np.stack([psi[1:-1] / VT, np.log(n[1:-1]), np.log(p[1:-1])], axis=1)

def _dd_newton(state, prm, V, h, tol=1e-13, maxit=200):
    U = state.copy()
    for it in range(maxit):
        F = _dd_residual(U, prm, V, h); nrm = np.max(np.abs(F))
        if nrm < tol: return U, nrm
        A, B, C, _ = _dd_jacobian(U, prm, V, h)
        dU = _block_tridiagonal_solve(A, B, C, -F); lam = 1.0
        for _ in range(60):
            Ut = U + lam * dU
            with np.errstate(over="ignore", invalid="ignore"):
                Ft = _dd_residual(Ut, prm, V, h)
            if np.all(np.isfinite(Ft)) and np.max(np.abs(Ft)) < nrm: break
            lam *= 0.5
        U = Ut
    return U, nrm

def _dd_continuation(prm, N, V, state0=None, V0=0.0, dvmax=0.05):
    """Steady state at bias V by continuation in steps of at most dvmax from (V0, state0) (equilibrium if None)."""
    h = prm["d"] / N
    U = _dd_equilibrium(prm, N) if state0 is None else state0.copy()
    nsteps = max(1, int(np.ceil(abs(V - V0) / dvmax - 1e-12)))
    for k in range(1, nsteps + 1):
        Vk = V0 + (V - V0) * k / nsteps
        U, nrm = _dd_newton(U, prm, Vk, h)
        if not (nrm < 1e-12):
            raise ValueError("drift-diffusion solve did not converge at V = %g" % Vk)
    return U

def _dd_charge_capacitance_current(state, prm, V, N):
    """Displaced charge (Ramo-Shockley form), its exact derivative dQ/dV and the current density (SI)."""
    h = prm["d"] / N; d = prm["d"]; M = N - 1
    psi, n, p = _dd_unpack(state, prm, V)
    x = np.arange(N + 1) * h; tw = np.full(N + 1, h); tw[0] = tw[-1] = 0.5 * h
    Q = prm["Cgeo"] * (V - prm["Vbi0"]) + _Q * np.sum(tw * ((x / d) * p + (1.0 - x / d) * n))
    A, B, C, dV = _dd_jacobian(state, prm, V, h)
    dUdV = _block_tridiagonal_solve(A, B, C, -dV)
    xi = np.arange(1, N) * h
    dQ = np.zeros((M, 3)); dQ[:, 1] = _Q * h * (1.0 - xi / d) * n[1:-1]; dQ[:, 2] = _Q * h * (xi / d) * p[1:-1]
    Cap = float(np.sum(dQ * dUdV) + prm["Cgeo"])
    Jn, Jp = _dd_fluxes(psi, n, p, prm, h)
    return Q, Cap, float((Jn + Jp)[N // 2]), psi, n, p

def _oracle_validity_threshold(level: float, intervals: int, par: "np.ndarray", march_step: float) -> "np.ndarray":
    lev = float(level); N = int(intervals); par = np.asarray(par, dtype=float).ravel(); dv = float(march_step)
    if par.size != 14:
        raise ValueError("par must be the 14-entry parameter vector")
    if not (lev > 0.0) or not (dv > 0.0):
        raise ValueError("level and march step must be positive")
    if N < 4 or N != intervals or N % 2 != 0:
        raise ValueError("intervals must be an even integer of at least 4")
    prm = _dd_params(par); h = prm["d"] / N
    def deviation(V, U0, V0):
        Us = _dd_continuation(prm, N, V, state0=U0, V0=V0, dvmax=dv)
        Q, Cap, J, psi, n, p = _dd_charge_capacitance_current(Us, prm, V, N)
        ca = _oracle_analytic_capacitance(V, par)[0] * prm["Cgeo"]
        return 100.0 * (ca / Cap - 1.0), Us, J
    U = _dd_equilibrium(prm, N); U, nrm = _dd_newton(U, prm, 0.0, h)
    dev0, U, J = deviation(0.0, U, 0.0)
    if dev0 >= lev:
        raise ValueError("deviation already above the level at zero bias")
    Va = 0.0; Ua = U
    while True:
        Vn = Va + dv
        if not (Vn < prm["Vbi0"] - 4.0 * prm["VT"]):
            raise ValueError("deviation never reaches the level below the built-in potential")
        devn, Un, J = deviation(Vn, Ua, Va)
        if devn >= lev:
            a, b = Va, Vn
            break
        Va, Ua = Vn, Un
    for it in range(200):
        m = 0.5 * (a + b)
        devm, Um, Jm = deviation(m, Ua, a)
        if devm < lev:
            a, Ua = m, Um
        else:
            b = m
        if b - a < 1e-12:
            break
    Vt = 0.5 * (a + b)
    devt, Ut, Jt = deviation(Vt, Ua, a)
    Vb = _oracle_effective_built_in_potential(Vt, par)[0]
    z = _Q * (Vb - Vt) / (2.0 * _KB * (prm["VT"] * _Q / _KB))
    return np.array([Vt, Vb - Vt, z], dtype=float)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {'setup': 'import numpy as np\npar = diode_parameters(100.0, 3.5, 300.0, 1.0e26, 1.0e26, 1.5, 0.02, 0.02, 1.0e-8, 5.0e-9, 0.1)\npar_gold = _oracle_diode_parameters(100.0, 3.5, 300.0, 1.0e26, 1.0e26, 1.5, 0.02, 0.02, 1.0e-8, 5.0e-9, 0.1)\nlevel = 1.0\nintervals = 200\nmarch_step = 0.05\n',
         'call': 'validity_threshold(level, intervals, par, march_step)',
         'gold_call': '_oracle_validity_threshold(level, intervals, par_gold, march_step)'},
        {'setup': 'import numpy as np\npar = diode_parameters(100.0, 3.5, 300.0, 1.0e26, 1.0e26, 1.5, 0.02, 0.50, 1.0e-8, 5.0e-9, 0.1)\npar_gold = _oracle_diode_parameters(100.0, 3.5, 300.0, 1.0e26, 1.0e26, 1.5, 0.02, 0.50, 1.0e-8, 5.0e-9, 0.1)\nlevel = 0.5\nintervals = 200\nmarch_step = 0.1\n',
         'call': 'validity_threshold(level, intervals, par, march_step)',
         'gold_call': '_oracle_validity_threshold(level, intervals, par_gold, march_step)'},
        {'setup': 'import numpy as np\npar = diode_parameters(80.0, 3.0, 320.0, 5.0e25, 2.0e26, 1.7, 0.10, 0.15, 2.0e-8, 1.0e-8, 0.05)\npar_gold = _oracle_diode_parameters(80.0, 3.0, 320.0, 5.0e25, 2.0e26, 1.7, 0.10, 0.15, 2.0e-8, 1.0e-8, 0.05)\nlevel = 2.0\nintervals = 300\nmarch_step = 0.05\n',
         'call': 'validity_threshold(level, intervals, par, march_step)',
         'gold_call': '_oracle_validity_threshold(level, intervals, par_gold, march_step)'},
    ]
