"""
From a converged drift-diffusion state at the voltage V, compute the charge displaced through the external circuit and its exact derivative with respect to V, evaluating any integral over the layer with the trapezoidal rule on the grid. Return Q / C_geo in V, the capacitance C / C_geo, and the ratio of Q to eps eps0 times the field at the middle of the layer, taken as (psi_{N/2+1} - psi_{N/2}) / h with the sign of E = -psi'. The capacitance must be the exact derivative of the discrete steady state, not a finite difference of Q between separately converged states. Raise ValueError if the state is not an (N + 1, 3) array with the contact potentials of this bias or if its discrete residuals exceed 1e-9.

A low-frequency capacitance measurement senses the charge that flows through the external circuit when the voltage changes; for injected carriers this charge is the electrode charge induced by each carrier, weighted by its position between the plates, and it reduces to eps eps0 times the bulk field when the accumulation regions are thin. Differentiating the discrete solution exactly avoids the loss of precision of finite differences between two independently converged nonlinear solves.

Returns
-------
A (3,) float64 array [Q / C_geo in V, C / C_geo, Q / (eps eps0 E_mid)].
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def exact_capacitance(state: "np.ndarray", voltage: float, par: "np.ndarray") -> "np.ndarray":
    """From a converged drift-diffusion state at the voltage V, evaluate the charge displaced
    through the external circuit and its exact derivative with respect to V. The displaced
    charge per unit area is Q = C_geo (V - V_bi,0) + q times the integral over the layer of
    [(x / d) p(x) + (1 - x / d) n(x)] (the Ramo-Shockley weighting of the injected carriers
    by their distance from the contact that injected them, which is equivalent to the
    source's expression in terms of the electrode charges and the mean carrier densities);
    evaluate the integral with the trapezoidal rule on the grid. A (3,) float64 array [Q /
    C_geo in V, C / C_geo, Q / (eps eps0 E_mid)].
    Parameters
    ----------
    state : np.ndarray
        The converged (N + 1, 3) drift-diffusion state at this bias.
    voltage : float
        Applied bias V in V at which the state was converged.
    par : np.ndarray
        The (14,) parameter vector returned by the parameter step.

    Raises
    ------
    ValueError
        If state is not an (N + 1, 3) array, if it does not carry the contact
        potentials of this bias, or if its discrete residuals exceed 1e-9.
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

def _oracle_exact_capacitance(state: "np.ndarray", voltage: float, par: "np.ndarray") -> "np.ndarray":
    st = np.asarray(state, dtype=float); par = np.asarray(par, dtype=float).ravel(); V = float(voltage)
    if st.ndim != 2 or st.shape[1] != 3 or st.shape[0] < 5:
        raise ValueError("state must be an (N + 1, 3) array")
    N = st.shape[0] - 1; prm = _dd_params(par)
    if abs(st[0, 0]) > 1e-9 or abs(st[-1, 0] * prm["VT"] - (prm["Vbi0"] - V)) > 1e-9:
        raise ValueError("state does not satisfy the contact potentials at this bias")
    U = st[1:-1].copy()
    F = _dd_residual(U, prm, V, prm["d"] / N)
    if np.max(np.abs(F)) > 1e-9:
        raise ValueError("state is not a converged steady state")
    Q, Cap, J, psi, n, p = _dd_charge_capacitance_current(U, prm, V, N)
    Emid = -(psi[N // 2 + 1] - psi[N // 2]) / (prm["d"] / N)
    return np.array([Q / prm["Cgeo"], Cap / prm["Cgeo"], Q / (prm["ee"] * Emid)], dtype=float)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {'setup': 'import numpy as np\npar = diode_parameters(100.0, 3.5, 300.0, 1.0e26, 1.0e26, 1.5, 0.02, 0.02, 1.0e-8, 5.0e-9, 0.1)\npar_gold = _oracle_diode_parameters(100.0, 3.5, 300.0, 1.0e26, 1.0e26, 1.5, 0.02, 0.02, 1.0e-8, 5.0e-9, 0.1)\nvoltage = 0.0\nintervals = 200\nstate = drift_diffusion_state(voltage, intervals, par)\nstate_gold = _oracle_drift_diffusion_state(voltage, intervals, par_gold)\n',
         'call': 'exact_capacitance(state, voltage, par)',
         'gold_call': '_oracle_exact_capacitance(state_gold, voltage, par_gold)'},
        {'setup': 'import numpy as np\npar = diode_parameters(100.0, 3.5, 300.0, 1.0e26, 1.0e26, 1.5, 0.02, 0.02, 1.0e-8, 5.0e-9, 0.1)\npar_gold = _oracle_diode_parameters(100.0, 3.5, 300.0, 1.0e26, 1.0e26, 1.5, 0.02, 0.02, 1.0e-8, 5.0e-9, 0.1)\nvoltage = 0.7\nintervals = 400\nstate = drift_diffusion_state(voltage, intervals, par)\nstate_gold = _oracle_drift_diffusion_state(voltage, intervals, par_gold)\n',
         'call': 'exact_capacitance(state, voltage, par)',
         'gold_call': '_oracle_exact_capacitance(state_gold, voltage, par_gold)'},
        {'setup': 'import numpy as np\npar = diode_parameters(100.0, 3.5, 300.0, 1.0e26, 1.0e26, 1.5, 0.02, 0.50, 1.0e-8, 5.0e-9, 0.1)\npar_gold = _oracle_diode_parameters(100.0, 3.5, 300.0, 1.0e26, 1.0e26, 1.5, 0.02, 0.50, 1.0e-8, 5.0e-9, 0.1)\nvoltage = -0.4\nintervals = 200\nstate = drift_diffusion_state(voltage, intervals, par)\nstate_gold = _oracle_drift_diffusion_state(voltage, intervals, par_gold)\n',
         'call': 'exact_capacitance(state, voltage, par)',
         'gold_call': '_oracle_exact_capacitance(state_gold, voltage, par_gold)'},
        {'setup': 'import numpy as np\npar = diode_parameters(80.0, 3.0, 320.0, 5.0e25, 2.0e26, 1.7, 0.10, 0.15, 2.0e-8, 1.0e-8, 0.05)\npar_gold = _oracle_diode_parameters(80.0, 3.0, 320.0, 5.0e25, 2.0e26, 1.7, 0.10, 0.15, 2.0e-8, 1.0e-8, 0.05)\nvoltage = 0.5\nintervals = 300\nstate = drift_diffusion_state(voltage, intervals, par)\nstate_gold = _oracle_drift_diffusion_state(voltage, intervals, par_gold)\n',
         'call': 'exact_capacitance(state, voltage, par)',
         'gold_call': '_oracle_exact_capacitance(state_gold, voltage, par_gold)'},
    ]
