"""
Returns the minimum Freidlin-Wentzell action of a fluctuation path that carries the strike to the burnout boundary at a prescribed time, with the terminal Hamiltonian and the terminal carrier population of the optimal path.

Noise-induced subthreshold burnout is a rare event whose probability is controlled by the cheapest fluctuation path; its action is the quantity that the source's Monte Carlo ensembles cannot resolve at small noise.

Returns
-------
A float64 array of shape (3,).
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def seb_instanton_action(params: "np.ndarray", ell: float, s_f: float) -> "np.ndarray":
    r"""Returns the minimum Freidlin-Wentzell action of a fluctuation path that carries the strike to the burnout boundary at a prescribed time, with the terminal Hamiltonian and the terminal carrier population of the optimal path.

    The source's reduced electrothermal model of single-event burnout in a SiC power MOSFET has two collective
    variables of the ion-strike sensitive region, the excess carrier population $n(s)$ and the normalised temperature
    $\Theta(s) = (T - T_0)/(T_c - T_0)$, in the dimensionless time $s = t/\tau_{loss}$. The ion strike injects
    carriers through the pulse $g(s;\ell) = \ell\,(2\pi)^{-1/2}\sigma_s^{-1}\exp[-(s - s_0)^2/(2\sigma_s^2)]$ of
    ionisation strength $\ell$; the coarse-grained field is $e(n) = b\,[1 + \eta_n\,n/(n + n_s)]$; the avalanche
    factor is $f(e, \Theta) = A_f\exp[-B_f(1 + a_\Theta\Theta)/e]$; the deterministic dynamics is $dn/ds = g(s;\ell) +
    [f(e(n), \Theta) - 1]\,n$ and $d\Theta/ds = \kappa\,e(n)\,n - r\,\Theta$ from $n(0) = \Theta(0) = 0$. Burnout is
    the first passage of $\Theta$ to the absorbing boundary $\Theta = 1$ within the observation window $0 \le s \le
    s_c$. The eleven model parameters are passed as one array in the fixed order $[s_0, \sigma_s, b, \eta_n, n_s,
    \kappa, r, A_f, B_f, a_\Theta, s_c]$. In the stochastic model the two variables obey the Ito equations $dn = [g +
    (f - 1)n]\,ds + \sigma_n\,n\,dW_n$ and $d\Theta = [\kappa e n - r\Theta]\,ds + \sigma_\Theta\,dW_\Theta$ with
    independent Wiener processes and the amplitudes linked by $\sigma_n = \sigma$, $\sigma_\Theta = 0.35\,\sigma$.

    For weak noise the probability of a rare boundary crossing is governed by the Freidlin-Wentzell action of the most
    probable path. With the diffusion of the stochastic model written as $\sigma^2 D(n)$, $D = \mathrm{diag}(n^2,
    0.35^2)$, the action of a path $x(s) = (n, \Theta)$ is $S[x] = \tfrac12\int_0^{s_f}(\dot x - A)^T D^{-1}(\dot x -
    A)\,ds$ with $A$ the deterministic drift, so that the crossing probability behaves as $\exp(-S/\sigma^2)$. This
    step returns the minimum of $S$ over paths that start at $(0, 0)$ at $s = 0$ and reach $\Theta = 1$ exactly at $s
    = s_f$ with the carrier endpoint free, together with the value of the Hamiltonian $H = p\cdot A + \tfrac12 p^T D
    p$ of the optimal path at $s_f$ (which equals $-dS/ds_f$) and the carrier population of the optimal path at $s_f$.
    The minimiser is a solution of the Hamiltonian two-point boundary value problem; the action must be accurate to
    $10^{-9}$.

    Args:
        params: array of eleven floats, the model parameters.
        ell: non-negative float, the ionisation strength.
        s_f: float, the crossing time, in $(s_0, s_c]$.

    Returns:
        A numpy float64 array of shape $(3,)$: $S(s_f)$, $H(s_f)$, $n(s_f)$.

    Raises:
        ValueError: on invalid params or ell, if s_f is not in $(s_0, s_c]$, or if the boundary value problem does not
        converge.
    """
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np, math
from scipy.integrate import solve_ivp, solve_bvp
from scipy.optimize import brentq, minimize_scalar
from scipy.special import erf

def _fin(x, name):
    try: v = float(x)
    except Exception: raise ValueError(f"{name} must be a real number")
    if not np.isfinite(v): raise ValueError(f"{name} must be finite")
    return v
def _pos(x, name):
    v = _fin(x, name)
    if v <= 0.0: raise ValueError(f"{name} must be positive")
    return v
def _nonneg(x, name):
    v = _fin(x, name)
    if v < 0.0: raise ValueError(f"{name} must be non-negative")
    return v
def _params(p):
    """[s0, sigma_s, b, eta_n, n_s, kappa, r, A_f, B_f, a_Theta, s_c] validated"""
    a = np.asarray(p, dtype=np.float64).ravel()
    if a.size != 11 or not np.all(np.isfinite(a)): raise ValueError("params must hold eleven finite numbers")
    s0, sg, b, eta, ns, kappa, r, Af, Bf, aT, sc = [float(v) for v in a]
    if sg <= 0.0 or b <= 0.0 or ns <= 0.0 or kappa <= 0.0 or r <= 0.0 or Af <= 0.0 or Bf <= 0.0 or sc <= 0.0: raise ValueError("pulse width, field, saturation scale, energy injection, relaxation, avalanche scales and window must be positive")
    if s0 < 0.0 or aT < 0.0: raise ValueError("pulse centre and thermal coefficient must be non-negative")
    if eta <= -1.0: raise ValueError("the field-redistribution strength must exceed -1")
    if sc <= s0: raise ValueError("the observation window must contain the ionisation pulse")
    return dict(s0=s0, sg=sg, b=b, eta=eta, ns=ns, kappa=kappa, r=r, Af=Af, Bf=Bf, aT=aT, sc=sc)
def _times(s_eval, sc):
    t = np.asarray(s_eval, dtype=np.float64).ravel()
    if t.size == 0 or not np.all(np.isfinite(t)) or np.any(t < 0.0) or np.any(t > sc): raise ValueError("evaluation times must lie in [0, s_c]")
    return t

def _source(P, s, ell):
    return ell / (math.sqrt(2.0 * math.pi) * P['sg']) * math.exp(-0.5 * ((s - P['s0']) / P['sg']) ** 2)
def _field(P, n):
    return P['b'] * (1.0 + P['eta'] * n / (n + P['ns']))
def _dfield(P, n):
    return P['b'] * P['eta'] * P['ns'] / (n + P['ns']) ** 2
def _aval(P, e, th):
    return P['Af'] * math.exp(-P['Bf'] * (1.0 + P['aT'] * th) / e)
def _drift(P, s, n, th, ell):
    e = _field(P, n); f = _aval(P, e, th)
    return _source(P, s, ell) + (f - 1.0) * n, P['kappa'] * e * n - P['r'] * th, P['kappa'] * e * n
def _jac(P, s, n, th, ell):
    e = _field(P, n); f = _aval(P, e, th); de = _dfield(P, n)
    dfde = f * P['Bf'] * (1.0 + P['aT'] * th) / e ** 2; dfdt = -f * P['Bf'] * P['aT'] / e
    return np.array([[(f - 1.0) + n * dfde * de, n * dfdt], [P['kappa'] * (e + n * de), -P['r']]])

def _trajectory(P, ell, s_end, rtol=1e-12):
    """deterministic n, Theta and the accumulated deposited work up to s_end (dense output); raises if the carrier
    population leaves the reduced model's range"""
    def _rhs(s, y):
        if y[0] > 1e6: raise ValueError("carrier population beyond the reduced model's range")
        return list(_drift(P, s, y[0], y[1], ell))
    sol = solve_ivp(_rhs, (0.0, s_end), [0.0, 0.0, 0.0], method='DOP853', rtol=rtol, atol=1e-15, dense_output=True, max_step=min(0.2, P['sg']))
    if not sol.success: raise ValueError("the deterministic response did not converge")
    return sol

def _peak(sol, a, b):
    """maximum of Theta on [a, b] from the dense solution: grid scan plus bounded refinement; the maximum may sit
    at the end of the window"""
    ss = np.linspace(a, b, 2001); th = sol.sol(ss)[1]
    i = int(np.argmax(th))
    if i == ss.size - 1:
        return float(th[-1]), float(b)
    if i == 0:
        return float(th[0]), float(a)
    r = minimize_scalar(lambda s: -sol.sol(s)[1], bounds=(ss[i - 1], ss[i + 1]), method='bounded', options={'xatol': 1e-13, 'maxiter': 500})
    return float(-r.fun), float(r.x)

def _carrier_peak(sol, a, b):
    """maximum of n on [a, b] from the dense solution: grid scan plus bounded refinement"""
    ss = np.linspace(a, b, 2001); nn = sol.sol(ss)[0]
    i = int(np.argmax(nn))
    if i == 0 or i == ss.size - 1:
        return float(nn[i])
    r = minimize_scalar(lambda s: -sol.sol(s)[0], bounds=(ss[i - 1], ss[i + 1]), method='bounded', options={'xatol': 1e-13, 'maxiter': 500})
    return float(-r.fun)

def _indicators(P, ell):
    sol = _trajectory(P, ell, P['sc'])
    lth, smax = _peak(sol, 0.0, P['sc'])
    y = sol.sol(P['sc'])
    return float(y[2]), lth, smax, _carrier_peak(sol, 0.0, P['sc']), sol

def _window_excess(P, ell):
    """Lambda_th - 1, with a carrier blow-up counted as certain burnout"""
    try:
        return _indicators(P, ell)[1] - 1.0
    except ValueError as e:
        if "reduced model" in str(e): return 1.0
        raise

def _threshold(P, lo=1e-3, hi=None):
    """ionisation strength at which the window maximum of Theta equals one (the source's Lambda_th = 1 boundary)"""
    _h = lambda ell: _window_excess(P, ell)
    if hi is None:
        hi = 1.0
        while _h(hi) < 0.0:
            hi *= 2.0
            if hi > 1e4: raise ValueError("no burnout threshold below an ionisation strength of 1e4")
    if _h(lo) > 0.0: raise ValueError("burnout already at the smallest ionisation strength")
    return brentq(_h, lo, hi, xtol=1e-13, rtol=1e-15, maxiter=200)

def _unconstrained_peak(P, ell, horizon=None):
    """the first maximum of Theta after the pulse when the trajectory is followed beyond the window; if the
    temperature is still rising at the horizon the peak is reported as infinite (a runaway strike)"""
    horizon = horizon or (P['sc'] + 40.0)
    try:
        sol = _trajectory(P, ell, horizon)
    except ValueError as e:
        if "reduced model" in str(e): return math.inf, math.nan
        raise
    ss = np.linspace(P['s0'], horizon, 4001); th = sol.sol(ss)[1]
    # the first local maximum after the pulse centre: the first grid point that is not exceeded by its successor
    rising = np.nonzero(th[1:] < th[:-1])[0]
    if rising.size == 0: return math.inf, math.nan
    i = int(rising[0])
    r = minimize_scalar(lambda s: -sol.sol(s)[1], bounds=(ss[max(i - 1, 0)], ss[i + 1]), method='bounded', options={'xatol': 1e-13, 'maxiter': 500})
    return float(-r.fun), float(r.x)

def _peak_threshold(P):
    """ionisation strength at which the unconstrained peak equals one; the window threshold is an upper bracket
    because the unconstrained peak is never below the window maximum"""
    def _h(ell):
        pk = _unconstrained_peak(P, ell)[0]
        return 1.0 if math.isinf(pk) else pk - 1.0
    hi = _threshold(P); lo = 1e-3
    if _h(lo) > 0.0: raise ValueError("burnout already at the smallest ionisation strength")
    hh = _h(hi)
    if hh < -1e-9: raise ValueError("the window threshold does not bracket the peak threshold")
    ell = hi if hh <= 0.0 else brentq(_h, lo, hi, xtol=1e-13, rtol=1e-15, maxiter=200)
    pk, spk = _unconstrained_peak(P, ell)
    if not math.isfinite(spk): raise ValueError("the temperature has not peaked within the extended horizon")
    return ell, spk

def _feedback_boundary(P, ell, r):
    """reference avalanche factor F = f(b, 0) at which the window maximum of Theta equals one, varying A_f at fixed
    B_f and a_Theta, for the given relaxation strength"""
    Q = dict(P); Q['r'] = r
    def _h(Af):
        R = dict(Q); R['Af'] = Af
        return _window_excess(R, ell)
    lo, hi = 1e-3, Q['Af']
    while _h(hi) < 0.0:
        hi *= 2.0
        if hi > 1e4: raise ValueError("no feedback boundary below an avalanche scale of 1e4")
    while _h(lo) > 0.0:
        lo *= 0.5
        if lo < 1e-9: raise ValueError("burnout persists at vanishing avalanche feedback")
    Af = brentq(_h, lo, hi, xtol=1e-13, rtol=1e-15, maxiter=200)
    return Af * math.exp(-Q['Bf'] / Q['b'])

def _lna(P, ell, s_end, rtol=1e-12):
    """deterministic trajectory and the linear-noise covariance per unit sigma^2 with the Ito diffusion
    diag(n^2, 0.35^2): dSigma/ds = J Sigma + Sigma J^T + D"""
    def _rhs(s, y):
        n, th = y[0], y[1]
        if n > 1e6: raise ValueError("carrier population beyond the reduced model's range")
        An, AT, dW = _drift(P, s, n, th, ell); J = _jac(P, s, n, th, ell)
        S = np.array([[y[3], y[4]], [y[4], y[5]]])
        dS = J @ S + S @ J.T + np.diag([n * n, 0.35 ** 2])
        return [An, AT, dW, dS[0, 0], dS[0, 1], dS[1, 1]]
    sol = solve_ivp(_rhs, (0.0, s_end), [0.0] * 6, method='DOP853', rtol=rtol, atol=1e-15, dense_output=True, max_step=min(0.2, P['sg']))
    if not sol.success: raise ValueError("the linear-noise covariance did not converge")
    return sol

def _gaussian_margin(P, ell):
    """the largest standardised margin z = (Theta - 1)/sqrt(Sigma_hat_ThetaTheta) over the window, its time, and
    the linear-noise solution"""
    sol = _lna(P, ell, P['sc'])
    def _z(s):
        y = sol.sol(s)
        return -np.inf if y[5] <= 0.0 else (y[1] - 1.0) / math.sqrt(y[5])
    ss = np.linspace(P['s0'], P['sc'], 2001); zz = np.array([_z(s) for s in ss])
    i = int(np.argmax(zz))
    if i == ss.size - 1: return float(zz[-1]), float(P['sc']), sol
    if i == 0: return float(zz[0]), float(ss[0]), sol
    r = minimize_scalar(lambda s: -_z(s), bounds=(ss[i - 1], ss[i + 1]), method='bounded', options={'xatol': 1e-13, 'maxiter': 500})
    return float(-r.fun), float(r.x), sol

def _Phi(x):
    return 0.5 * (1.0 + erf(x / math.sqrt(2.0)))

def _gaussian_burnout(P, ell, sigma):
    zmax, smax, sol = _gaussian_margin(P, ell)
    return _Phi(zmax / sigma), smax, zmax

def _gaussian_width(P, sigma):
    def _h(q):
        return lambda ell: _gaussian_burnout(P, ell, sigma)[0] - q
    lo = 1e-3; hi = 1.0
    while _h(0.9)(hi) < 0.0:
        hi *= 2.0
        if hi > 1e4: raise ValueError("the burnout probability never reaches 0.9")
    if _h(0.1)(lo) > 0.0: raise ValueError("the burnout probability exceeds 0.1 at the smallest ionisation strength")
    l1 = brentq(_h(0.1), lo, hi, xtol=1e-13, rtol=1e-15, maxiter=200)
    l9 = brentq(_h(0.9), l1, hi, xtol=1e-13, rtol=1e-15, maxiter=200)
    return l1, l9, l9 - l1

def _ham_rhs(P, s, Y, ell):
    n, th, pn, pt = Y
    An, AT, dW = _drift(P, s, n, th, ell); J = _jac(P, s, n, th, ell)
    return [An + n * n * pn, AT + 0.35 ** 2 * pt, -(J[0, 0] * pn + J[1, 0] * pt) - n * pn * pn, -(J[0, 1] * pn + J[1, 1] * pt)]

def _instanton(P, ell, sf, tol=1e-10):
    """minimum Freidlin-Wentzell action of a path from (0, 0) at s = 0 to Theta = 1 at s = sf (free carrier
    endpoint), diffusion diag(n^2, 0.35^2): Hamiltonian two-point boundary value problem"""
    if sf <= P['s0']: raise ValueError("the crossing time must lie beyond the ionisation pulse centre")
    det = _trajectory(P, ell, sf)
    ss = np.linspace(0.0, sf, 401); d = det.sol(ss)
    Y0 = np.zeros((4, ss.size)); Y0[0] = d[0]; Y0[1] = d[1]; Y0[3] = 0.05
    def _fun(s, Y):
        out = np.empty_like(Y)
        for i in range(s.size): out[:, i] = _ham_rhs(P, s[i], Y[:, i], ell)
        return out
    def _bc(Ya, Yb): return np.array([Ya[0], Ya[1], Yb[1] - 1.0, Yb[2]])
    res = solve_bvp(_fun, _bc, ss, Y0, tol=tol, max_nodes=400000, verbose=0)
    if res.status != 0: raise ValueError("the instanton boundary value problem did not converge")
    # Gauss-Legendre quadrature of the Lagrangian on the converged mesh
    x = res.x; gx, gw = np.polynomial.legendre.leggauss(6)
    S = 0.0
    for a, b in zip(x[:-1], x[1:]):
        sm = 0.5 * (a + b) + 0.5 * (b - a) * gx; Y = res.sol(sm)
        L = 0.5 * (Y[2] ** 2 * Y[0] ** 2 + 0.35 ** 2 * Y[3] ** 2)
        S += 0.5 * (b - a) * float(np.dot(gw, L))
    n, th, pn, pt = res.sol(sf)
    An, AT, dW = _drift(P, sf, n, th, ell)
    H = pn * An + pt * AT + 0.5 * (n * n * pn * pn + 0.35 ** 2 * pt * pt)
    return float(S), float(H), float(n)

def _instanton_free(P, ell, sf0, tol=1e-10):
    """free crossing time: the Hamiltonian system on tau = s/sf in [0, 1] with sf an unknown parameter fixed by the
    transversality condition H(sf) = 0"""
    det = _trajectory(P, ell, sf0)
    tt = np.linspace(0.0, 1.0, 401); d = det.sol(tt * sf0)
    Y0 = np.zeros((4, tt.size)); Y0[0] = d[0]; Y0[1] = d[1]; Y0[3] = 0.05
    def _fun(t, Y, p):
        sf = p[0]; out = np.empty_like(Y)
        for i in range(t.size): out[:, i] = sf * np.asarray(_ham_rhs(P, sf * t[i], Y[:, i], ell))
        return out
    def _bc(Ya, Yb, p):
        sf = p[0]; n, th, pn, pt = Yb
        An, AT, dW = _drift(P, sf, n, th, ell)
        H = pn * An + pt * AT + 0.5 * (n * n * pn * pn + 0.35 ** 2 * pt * pt)
        return np.array([Ya[0], Ya[1], Yb[1] - 1.0, Yb[2], H])
    res = solve_bvp(_fun, _bc, tt, Y0, p=np.array([sf0]), tol=tol, max_nodes=400000, verbose=0)
    if res.status != 0: raise ValueError("the free-time instanton boundary value problem did not converge")
    sf = float(res.p[0])
    x = res.x; gx, gw = np.polynomial.legendre.leggauss(6); S = 0.0
    for a, b in zip(x[:-1], x[1:]):
        tm = 0.5 * (a + b) + 0.5 * (b - a) * gx; Y = res.sol(tm)
        L = 0.5 * (Y[2] ** 2 * Y[0] ** 2 + 0.35 ** 2 * Y[3] ** 2)
        S += 0.5 * (b - a) * sf * float(np.dot(gw, L))
    return float(S), sf

def _subthreshold_rate(P, ell):
    """S* = min over crossing times sf in (s0, sc] of the fixed-time action; the terminal Hamiltonian equals -dS/dsf,
    so the minimum sits at the window end when H(sc) >= 0 and otherwise at the interior zero of H (transversality)"""
    sc = P['sc']
    S_end, H_end, n_end = _instanton(P, ell, sc)
    if H_end >= 0.0:
        return S_end, sc, H_end
    peak, speak = _unconstrained_peak(P, ell, horizon=sc)
    S, sf = _instanton_free(P, ell, min(max(speak, P['s0'] + 0.5), sc - 1e-6))
    if not (P['s0'] < sf < sc): raise ValueError("the crossing-time optimum left the window")
    return S, sf, 0.0

def _oracle_seb_instanton_action(params: "np.ndarray", ell: float, s_f: float) -> "np.ndarray":
    P = _params(params); l = _nonneg(ell, "ell"); sf = _fin(s_f, "s_f")
    if not (P['s0'] < sf <= P['sc']): raise ValueError("the crossing time must lie in (s0, sc]")
    out = np.array(_instanton(P, l, sf), dtype=np.float64)
    if not np.all(np.isfinite(out)): raise ValueError("non-finite action")
    return out

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {"setup":"import numpy as np\nparams = np.array([2.0, 0.18, 1.20, 0.35, 1.0, 0.16, 0.18, 5.5, 2.2, 0.12, 12.0])\n","call":"seb_instanton_action(params.copy(), 0.70, 12.0)","gold_call":"_oracle_seb_instanton_action(params.copy(), 0.70, 12.0)","tol":1e-7},
        {"setup":"import numpy as np\nparams = np.array([2.0, 0.18, 1.20, 0.35, 1.0, 0.16, 0.18, 5.5, 2.2, 0.12, 12.0])\n","call":"seb_instanton_action(params.copy(), 0.70, 10.0)","gold_call":"_oracle_seb_instanton_action(params.copy(), 0.70, 10.0)","tol":1e-7},
        {"setup":"import numpy as np\nparams = np.array([2.0, 0.18, 1.20, 0.35, 1.0, 0.16, 0.18, 5.5, 2.2, 0.12, 20.0])\n","call":"seb_instanton_action(params.copy(), 0.72, 15.0)","gold_call":"_oracle_seb_instanton_action(params.copy(), 0.72, 15.0)","tol":1e-7},
        {"setup":"import numpy as np\nparams = np.array([2.0, 0.18, 1.20, 0.35, 1.0, 0.16, 0.18, 5.5, 2.2, 0.12, 12.0])\n# boundary: a strike above the deterministic threshold, where the boundary is reached with no fluctuation cost at the deterministic crossing time and the action is positive only because the crossing is forced earlier\n","call":"seb_instanton_action(params.copy(), 0.85, 8.0)","gold_call":"_oracle_seb_instanton_action(params.copy(), 0.85, 8.0)","tol":1e-7},
        {"setup":"import numpy as np\nparams = np.array([2.0, 0.18, 1.20, 0.35, 1.0, 0.16, 0.18, 5.5, 2.2, 0.12, 12.0])\n# invalid input: a crossing time before the pulse centre must raise ValueError\ndef _catches_value_error(fn):\n    try:\n        fn()\n    except ValueError:\n        return np.array([1.0])\n    return np.array([0.0])\n","call":"_catches_value_error(lambda: seb_instanton_action(params.copy(), 0.70, 1.0))","gold_call":"_catches_value_error(lambda: _oracle_seb_instanton_action(params.copy(), 0.70, 1.0))"},
    ]
