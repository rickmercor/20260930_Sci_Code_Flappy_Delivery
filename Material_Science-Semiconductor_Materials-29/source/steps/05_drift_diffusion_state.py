"""
Solve the exact steady-state drift-diffusion model of the same diode at the voltage V on a uniform grid of N intervals (nodes x_i = i d / N, i = 0..N) and return the converged state. The unknowns are the electrostatic potential psi, the electron density n and the hole density p. Boundary values: psi(0) = 0, psi(d) = V_bi,0 - V, and the Boltzmann contact densities n_an, p_an, n_cat and p_cat of the parameter step. The discrete nonlinear system is solved to machine precision, with every scaled residual below 1e-12; the converged discrete solution is unique. Return, per node, psi q / kT, ln n and ln p, with densities in 1/m^3. Raise ValueError if par does not have 14 entries, if N is not an even integer of at least 4, or if V is not below V_bi,0.

The source validates its analytical framework against numerical drift-diffusion simulations at low frequency, where the capacitance is not limited by transport; the steady-state drift-diffusion model with Boltzmann contacts is the reference against which the injected-carrier approximations (majority carrier only in each accumulation region, uniform bulk field) can be measured exactly.

Returns
-------
An (N + 1, 3) float64 array with columns [psi q / kT, ln n, ln p] at the nodes x_i = i d / N.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def drift_diffusion_state(voltage: float, intervals: int, par: "np.ndarray") -> "np.ndarray":
    """Solve the exact steady-state drift-diffusion model of the same diode at the voltage V on
    a uniform grid of N intervals (nodes x_i = i d / N, i = 0..N) and return the converged
    state. An (N + 1, 3) float64 array with columns [psi q / kT, ln n, ln p] at the nodes
    x_i = i d / N.
    Parameters
    ----------
    voltage : float
        Applied bias V in V, below the nominal built-in potential V_bi,0.
    intervals : int
        Number N of grid intervals, an even integer of at least 4.
    par : np.ndarray
        The (14,) parameter vector returned by the parameter step.

    Raises
    ------
    ValueError
        If par does not have 14 entries, if N is not an even integer of at
        least 4, or if V is not below V_bi,0.
    """
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


_Q = 1.602176634e-19

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

def _oracle_drift_diffusion_state(voltage: float, intervals: int, par: "np.ndarray") -> "np.ndarray":
    par = np.asarray(par, dtype=float).ravel(); N = int(intervals); V = float(voltage)
    if par.size != 14:
        raise ValueError("par must be the 14-entry parameter vector")
    if N < 4 or N != intervals or N % 2 != 0:
        raise ValueError("intervals must be an even integer of at least 4")
    prm = _dd_params(par)
    if not (V < prm["Vbi0"]):
        raise ValueError("bias must lie below the nominal built-in potential")
    U = _dd_continuation(prm, N, V)
    psi, n, p = _dd_unpack(U, prm, V)
    return np.stack([psi / prm["VT"], np.log(n), np.log(p)], axis=1)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {'setup': 'import numpy as np\npar = diode_parameters(100.0, 3.5, 300.0, 1.0e26, 1.0e26, 1.5, 0.02, 0.02, 1.0e-8, 5.0e-9, 0.1)\npar_gold = _oracle_diode_parameters(100.0, 3.5, 300.0, 1.0e26, 1.0e26, 1.5, 0.02, 0.02, 1.0e-8, 5.0e-9, 0.1)\nvoltage = 0.0\nintervals = 200\n',
         'call': 'drift_diffusion_state(voltage, intervals, par)',
         'gold_call': '_oracle_drift_diffusion_state(voltage, intervals, par_gold)'},
        {'setup': 'import numpy as np\npar = diode_parameters(100.0, 3.5, 300.0, 1.0e26, 1.0e26, 1.5, 0.02, 0.02, 1.0e-8, 5.0e-9, 0.1)\npar_gold = _oracle_diode_parameters(100.0, 3.5, 300.0, 1.0e26, 1.0e26, 1.5, 0.02, 0.02, 1.0e-8, 5.0e-9, 0.1)\nvoltage = 0.7\nintervals = 400\n',
         'call': 'drift_diffusion_state(voltage, intervals, par)',
         'gold_call': '_oracle_drift_diffusion_state(voltage, intervals, par_gold)'},
        {'setup': 'import numpy as np\npar = diode_parameters(100.0, 3.5, 300.0, 1.0e26, 1.0e26, 1.5, 0.02, 0.50, 1.0e-8, 5.0e-9, 0.1)\npar_gold = _oracle_diode_parameters(100.0, 3.5, 300.0, 1.0e26, 1.0e26, 1.5, 0.02, 0.50, 1.0e-8, 5.0e-9, 0.1)\nvoltage = -0.4\nintervals = 200\n',
         'call': 'drift_diffusion_state(voltage, intervals, par)',
         'gold_call': '_oracle_drift_diffusion_state(voltage, intervals, par_gold)'},
        {'setup': 'import numpy as np\npar = diode_parameters(80.0, 3.0, 320.0, 5.0e25, 2.0e26, 1.7, 0.10, 0.15, 2.0e-8, 1.0e-8, 0.05)\npar_gold = _oracle_diode_parameters(80.0, 3.0, 320.0, 5.0e25, 2.0e26, 1.7, 0.10, 0.15, 2.0e-8, 1.0e-8, 0.05)\nvoltage = 0.5\nintervals = 300\n',
         'call': 'drift_diffusion_state(voltage, intervals, par)',
         'gold_call': '_oracle_drift_diffusion_state(voltage, intervals, par_gold)'},
    ]
