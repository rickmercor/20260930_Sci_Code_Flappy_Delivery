"""
Apply the source's built-in potential extraction protocol to the exact model and derive the first-order expansion it rests on. Compute the exact capacitance at each of the fit voltages (strictly increasing, below V_bi,0) from converged drift-diffusion states on the grid of N intervals, form y(V) = (Delta C / C_geo)^-1 with Delta C = C - C_geo, and fit the straight line y = S (V* - V) by ordinary least squares in V. Read the contact factor and the built-in potential the way the source does for its limiting cases (both contacts ohmic or one ohmic and one non-injecting): eta_ext = q C(0) / (2 S kT C_geo) and V_bi,ext = V* + S^-1 [1 + C(0) / C_geo], with C(0) the exact zero-bias capacitance. Then derive the exact first-order expansion of the analytic inverse excess capacitance of the injected-carrier model about V = 0 for arbitrary contacts, y(V) = y(0) - S_lin V + ..., using dV_bi/dV = 1 - C/C_geo and the voltage dependence of both contact factors, and return S_lin and V*_lin = y(0) / S_lin, together with the source's slope factor f_S = eta^2 - 6 (eta + 1/eta) + 12 evaluated at eta(0), which the source gives for devices with at least one ohmic contact. Raise ValueError if fewer than three fit voltages are given, if they are not strictly increasing or not below V_bi,0, if par does not have 14 entries, or if N is not an even integer of at least 4.

The source extracts the built-in potential from a linear fit of the inverse relative excess capacitance near zero bias, reading the contact factor from the slope; the reading assumes the slope factor of its limiting cases, and both the finite fit window and contacts that are only nearly ohmic bias the extracted built-in potential with respect to the effective built-in potential at zero bias.

Returns
-------
An (8,) float64 array [S in 1/V, V* in V, eta_ext, V_bi,ext in V, V_bi,ext - V_bi(0) in V, S_lin in 1/V, V*_lin in V, f_S].
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def built_in_extraction(fit_voltages: "np.ndarray", intervals: int, par: "np.ndarray") -> "np.ndarray":
    """Apply the source's built-in potential extraction protocol to the exact model and derive
    the first-order expansion it rests on. An (8,) float64 array [S in 1/V, V* in V,
    eta_ext, V_bi,ext in V, V_bi,ext - V_bi(0) in V, S_lin in 1/V, V*_lin in V, f_S].
    Parameters
    ----------
    fit_voltages : np.ndarray
        At least three strictly increasing bias values in V at which the
        inverse excess capacitance is fitted, all below V_bi,0.
    intervals : int
        Number N of grid intervals, an even integer of at least 4.
    par : np.ndarray
        The (14,) parameter vector returned by the parameter step.

    Raises
    ------
    ValueError
        If fit_voltages is not at least three strictly increasing values, if
        par does not have 14 entries, if N is not an even integer of at least
        4, or if any fit voltage is not below V_bi,0.
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

def _oracle_built_in_extraction(fit_voltages: "np.ndarray", intervals: int, par: "np.ndarray") -> "np.ndarray":
    vs = np.atleast_1d(np.asarray(fit_voltages, dtype=float)); par = np.asarray(par, dtype=float).ravel(); N = int(intervals)
    if vs.ndim != 1 or vs.size < 3 or np.any(np.diff(vs) <= 0.0):
        raise ValueError("fit voltages must be at least three strictly increasing values")
    if par.size != 14:
        raise ValueError("par must be the 14-entry parameter vector")
    if N < 4 or N != intervals or N % 2 != 0:
        raise ValueError("intervals must be an even integer of at least 4")
    prm = _dd_params(par); VT = par[0]
    if not (vs[-1] < prm["Vbi0"]):
        raise ValueError("fit voltages must lie below the nominal built-in potential")
    U0 = _dd_equilibrium(prm, N); U0, nrm = _dd_newton(U0, prm, 0.0, prm["d"] / N)
    c0 = _dd_charge_capacitance_current(U0, prm, 0.0, N)[1] / prm["Cgeo"]
    y = np.empty(vs.size)
    for sign in (1.0, -1.0):                                   # march away from equilibrium in both directions
        idx = [k for k in range(vs.size) if sign * vs[k] > 0.0] if sign > 0 else [k for k in range(vs.size) if vs[k] < 0.0][::-1]
        U, Vc = U0, 0.0
        for k in idx:
            U = _dd_continuation(prm, N, vs[k], state0=U, V0=Vc); Vc = vs[k]
            y[k] = 1.0 / (_dd_charge_capacitance_current(U, prm, vs[k], N)[1] / prm["Cgeo"] - 1.0)
    for k in range(vs.size):
        if vs[k] == 0.0:
            y[k] = 1.0 / (c0 - 1.0)
    A = np.vstack([np.ones_like(vs), vs]).T
    coef = np.linalg.lstsq(A, y, rcond=None)[0]
    intercept, slope = coef[0], coef[1]
    S = -slope; Vstar = intercept / S
    eta_ext = c0 / (2.0 * S * VT)                                   # the source's limiting-case reading of the slope
    vbi_ext = Vstar + (1.0 + c0) / S                                # the source's built-in potential formula V_bi = V* + [1 + C(0)/C_geo] / S
    # exact first-order expansion of the analytic inverse excess capacitance about V = 0 (arbitrary contacts)
    Vb0, ea, ec, eb, wa, wc, xs = _oracle_effective_built_in_potential(0.0, par); eta0 = ea + ec
    ca0 = _oracle_analytic_capacitance(0.0, par)[0]
    y0 = (Vb0 - 2.0 * eta0 * VT) / (2.0 * eta0 * VT)
    deta = ca0 * (ea * (2.0 - ea) * (1.0 - ea) + ec * (2.0 - ec) * (1.0 - ec)) / Vb0      # d eta / dV at V = 0
    dy = -(ca0 / eta0 + Vb0 * deta / eta0 ** 2) / (2.0 * VT)                          # d y / dV at V = 0
    S_lin = -dy; Vstar_lin = y0 / S_lin
    fS = eta0 ** 2 - 6.0 * (eta0 + 1.0 / eta0) + 12.0                                   # the source's one-ohmic-contact factor
    return np.array([S, Vstar, eta_ext, vbi_ext, vbi_ext - Vb0, S_lin, Vstar_lin, fS], dtype=float)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {'setup': 'import numpy as np\npar = diode_parameters(100.0, 3.5, 300.0, 1.0e26, 1.0e26, 1.5, 0.02, 0.02, 1.0e-8, 5.0e-9, 0.1)\npar_gold = _oracle_diode_parameters(100.0, 3.5, 300.0, 1.0e26, 1.0e26, 1.5, 0.02, 0.02, 1.0e-8, 5.0e-9, 0.1)\nfit_voltages = np.array([-0.2, -0.1, 0.0, 0.1, 0.2])\nintervals = 200\n',
         'call': 'built_in_extraction(fit_voltages, intervals, par)',
         'gold_call': '_oracle_built_in_extraction(fit_voltages, intervals, par_gold)'},
        {'setup': 'import numpy as np\npar = diode_parameters(100.0, 3.5, 300.0, 1.0e26, 1.0e26, 1.5, 0.02, 0.50, 1.0e-8, 5.0e-9, 0.1)\npar_gold = _oracle_diode_parameters(100.0, 3.5, 300.0, 1.0e26, 1.0e26, 1.5, 0.02, 0.50, 1.0e-8, 5.0e-9, 0.1)\nfit_voltages = np.array([-0.3, -0.15, 0.0, 0.15])\nintervals = 200\n',
         'call': 'built_in_extraction(fit_voltages, intervals, par)',
         'gold_call': '_oracle_built_in_extraction(fit_voltages, intervals, par_gold)'},
        {'setup': 'import numpy as np\npar = diode_parameters(80.0, 3.0, 320.0, 5.0e25, 2.0e26, 1.7, 0.10, 0.15, 2.0e-8, 1.0e-8, 0.05)\npar_gold = _oracle_diode_parameters(80.0, 3.0, 320.0, 5.0e25, 2.0e26, 1.7, 0.10, 0.15, 2.0e-8, 1.0e-8, 0.05)\nfit_voltages = np.array([-0.1, 0.0, 0.1])\nintervals = 300\n',
         'call': 'built_in_extraction(fit_voltages, intervals, par)',
         'gold_call': '_oracle_built_in_extraction(fit_voltages, intervals, par_gold)'},
    ]
