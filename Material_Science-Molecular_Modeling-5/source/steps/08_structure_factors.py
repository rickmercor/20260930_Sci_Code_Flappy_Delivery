"""
Return the partial structure factors S++(q) and S+-(q) and the charge-charge structure factor SZZ(q) at wavenumbers q in (0, 20]. The total correlation functions are H_ij = h_ij + (1 + h_ij) T_ij with the Mayer functions of step 4 and the corrections of step 7, and they equal minus one inside the hard core. With mole fractions one half, S_ij(q) is delta_ij/2 plus rho/4 times the two-dimensional Fourier transform of H_ij, including the hard-core disk, and SZZ is the charge-weighted combination of the partial structure factors normalised so that it tends to one for an ideal gas of point charges. The transforms must be accurate to 1e-8 in absolute terms, which requires the correlation functions on a radial grid extending far enough for the screened tails to have decayed. Reject invalid parameters and wavenumbers outside (0, 20].

The partial structure factors are what scattering or simulation measures and what the source compares with Monte Carlo data; the charge structure factor tests how well the approximate theory screens, since an exact theory of a conducting fluid would make it vanish quadratically at long wavelength. The first-order scheme retains a finite value there, which is one of its measurable imperfections.

Returns
-------
A (3, n) float64 array whose rows are S++, S+- and SZZ at the n wavenumbers.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def structure_factors(q: "np.ndarray", sigma: float, Gamma: float, rho: float) -> "np.ndarray":
    """Return the partial structure factors S++(q) and S+-(q) and the charge-charge structure
    factor SZZ(q) at wavenumbers q in (0, 20]. The total correlation functions are H_ij = h_ij +
    (1 + h_ij) T_ij with the Mayer functions of step 4 and the corrections of step 7, and they
    equal minus one inside the hard core. With mole fractions one half, S_ij(q) is delta_ij/2
    plus rho/4 times the two-dimensional Fourier transform of H_ij, including the hard-core
    disk, and SZZ is the charge-weighted combination of the partial structure factors normalised
    so that it tends to one for an ideal gas of point charges. The transforms must be accurate
    to 1e-8 in absolute terms, which requires the correlation functions on a radial grid
    extending far enough for the screened tails to have decayed. Reject invalid parameters and
    wavenumbers outside (0, 20].

    Args:
        q: array-like of shape (n,) of finite wavenumbers in (0, 20], in units of the inverse disk diameter.
        sigma: non-negative finite float, the splitting length in units of the disk diameter (sigma = 0 is the
            Debye-Hueckel limit of the scheme).
        Gamma: positive finite float, the Coulomb coupling.
        rho: positive finite float, the total reduced ion density rho a^2 (both species together).

    Returns:
        A numpy float64 array of shape (3, n) whose rows are S++(q), S+-(q) and SZZ(q) at the n wavenumbers:
        S_ij = delta_ij/2 + (rho/4) Hbar_ij(q) with H_ij = h_ij + (1 + h_ij) T_ij (equal to -1 inside the hard
        core) and SZZ = 1 + (rho/2)[Hbar++ - Hbar+-]; accurate to 1e-8 in absolute terms.

    Raises:
        ValueError: if q is not a one-dimensional array of finite values inside (0, 20]; if sigma is negative
            or not finite; or if Gamma or rho is not a positive finite number.
    """
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
from scipy.special import kve, k0, k1, j0, j1
from scipy.optimize import brentq


def _check_state(Gamma, rho):
    Gamma = float(Gamma); rho = float(rho)
    if not (Gamma > 0.0) or not np.isfinite(Gamma):
        raise ValueError("Gamma must be positive")
    if not (rho > 0.0) or not np.isfinite(rho):
        raise ValueError("rho must be positive")
    return Gamma, rho


def _check_sigma(sigma):
    sigma = float(sigma)
    if not (sigma >= 0.0) or not np.isfinite(sigma):
        raise ValueError("sigma must be non-negative")
    return sigma


def _kv(n, z):
    z = np.asarray(z, dtype=complex)
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        out = kve(n, z) * np.exp(-z)
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def _S(q, sigma):
    x = (sigma * q) ** 2
    return 1.0 + x + x ** 2 + x ** 3 + x ** 4


def _gl_partial_fractions(sigma, kap2):
    """Poles of 1/(x S(x) + kap2) in x = q^2: residues B_m and b_m = sqrt(-x_m) with Re b_m > 0."""
    if sigma == 0.0:
        return np.array([1.0 + 0j]), np.array([np.sqrt(kap2) + 0j])
    coef = np.array([kap2, 1.0, sigma ** 2, sigma ** 4, sigma ** 6, sigma ** 8], dtype=complex)
    x = np.polynomial.polynomial.polyroots(coef)
    B = 1.0 / np.polynomial.polynomial.polyval(x, np.polynomial.polynomial.polyder(coef))
    b = np.sqrt(-x)
    b = np.where(b.real < 0.0, -b, b)
    return B, b


def _gl_closed(u, sigma, Gamma, kap2):
    """Gl(u) = Gamma int_0^inf q J0(qu)/(q^2 S + kap2) dq = Gamma sum_m B_m K0(b_m u)."""
    B, b = _gl_partial_fractions(sigma, kap2)
    u = np.asarray(u, dtype=float)
    z = np.multiply.outer(u, b)
    out = Gamma * np.real(np.sum(B * _kv(0, z), axis=-1))
    zero = (u == 0.0)
    if np.any(zero):
        out = np.where(zero, -Gamma * np.real(np.sum(B * np.log(b))), out)
    return out


def _filter_poles():
    """The four non-real fifth roots of unity omega_k, the pole directions beta_k = sqrt(-omega_k) with
    positive real part and the residues c_k = 1/(omega_k P'(omega_k)) of 1 - 1/S in x = (sigma q)^2."""
    omega = np.exp(2j * np.pi * np.arange(1, 5) / 5.0)
    beta = np.exp(1j * np.pi * (np.arange(1, 5) / 5.0 - 0.5))
    ck = 1.0 / (omega * (1.0 + 2.0 * omega + 3.0 * omega ** 2 + 4.0 * omega ** 3))
    return beta, ck


def _vs_closed(u, sigma, Gamma):
    """vs(u) = Gamma int_0^inf (S-1)/(q S) J0(qu) dq = -Gamma sum_k c_k K0(beta_k u/sigma)."""
    u = np.asarray(u, dtype=float)
    if sigma == 0.0:
        return np.zeros_like(u)
    beta, ck = _filter_poles()
    z = np.multiply.outer(u / sigma, beta)
    return -Gamma * np.real(np.sum(ck * _kv(0, z), axis=-1))


def _w_total(u, sigma, Gamma, rho):
    kap2 = 2.0 * np.pi * Gamma * rho
    return _vs_closed(u, sigma, Gamma) + _gl_closed(u, sigma, Gamma, kap2)


def _leggauss(n):
    """Gauss-Legendre nodes and weights on [-1, 1], rebuilt on every call (no module-level cache)."""
    return np.polynomial.legendre.leggauss(n)


def _gl_nodes(a, b, xw):
    x, w = xw
    return 0.5 * (b - a) * x + 0.5 * (b + a), 0.5 * (b - a) * w


def _panel_edges(a, b, width, ratio=1.12, far=6.0):
    if b - a <= 0.0:
        return np.array([a, b])
    if b <= far:
        m = max(1, int(np.ceil((b - a) / width)))
        return np.linspace(a, b, m + 1)
    e = [a]
    while e[-1] < far - 1e-12 and e[-1] < b:
        e.append(min(e[-1] + width, far, b))
    w = width
    while e[-1] < b - 1e-12:
        w *= ratio
        e.append(min(e[-1] + w, b))
    return np.array(e)


def _panel_nodes(edges, n):
    xw = _leggauss(n); xs = []; ws = []
    for a, b in zip(edges[:-1], edges[1:]):
        x, w = _gl_nodes(a, b, xw); xs.append(x); ws.append(w)
    return np.concatenate(xs), np.concatenate(ws)


def _umax(Gamma, rho):
    kap = np.sqrt(2.0 * np.pi * Gamma * rho)
    return 1.0 + max(30.0, 45.0 / kap)


def _rcut(Gamma, rho, sigma):
    """Radial cutoff of the correlation integrals: the distance at which the slowest exponential tail of the
    kernels (the screened poles b_m of Gl and the filter poles beta_k/sigma of vs) has decayed by 13 decades,
    never beyond _umax."""
    kap2 = 2.0 * np.pi * Gamma * rho
    B, b = _gl_partial_fractions(sigma, kap2)
    lam = float(np.min(b.real))
    if sigma > 0.0:
        beta, _ = _filter_poles()
        lam = min(lam, float(np.min(beta.real)) / sigma)
    return min(_umax(Gamma, rho), 1.0 + 13.0 * np.log(10.0) / lam)


def _w_spline(sigma, Gamma, rho):
    from scipy.interpolate import CubicSpline
    U = _umax(Gamma, rho) + 3.0
    ug = np.linspace(1e-6, U, int(U / 0.002) + 1)
    return CubicSpline(ug, _w_total(ug, sigma, Gamma, rho))


def _hankel_forward(q, u, wu, f):
    """2 pi int u J0(q u) f(u) du on the nodes (u, wu), for an array q."""
    q = np.atleast_1d(np.asarray(q, dtype=float))
    out = np.empty_like(q)
    for i in range(0, len(q), 64):
        qq = q[i:i + 64]
        out[i:i + 64] = 2.0 * np.pi * np.sum(j0(np.outer(qq, u)) * (wu * u * f), axis=1)
    return out


def _chord_A(r, r1, g, g1, xw):
    """A_g(r, r1) = int_0^{2 pi} g(d(phi)) dphi with d^2 = r^2 + r1^2 - 2 r r1 cos(phi), g(d) = 0 for d < 1,
    written as 4 int g(d) d dd / sqrt((d^2 - dmin^2)(dmax^2 - d^2)) with the endpoint singularities removed;
    xw holds the Gauss-Legendre nodes and weights of the angular quadrature."""
    x, w = xw
    th = 0.5 * np.pi * (x + 1.0); wth = 0.5 * np.pi * w
    th2 = 0.25 * np.pi * (x + 1.0); wth2 = 0.25 * np.pi * w
    r1 = np.asarray(r1, dtype=float)
    dmin = np.abs(r - r1); dmax = r + r1
    out = np.zeros_like(r1)
    m1 = dmin >= 1.0
    if m1.any():
        xmin = dmin[m1] ** 2; xmax = dmax[m1] ** 2
        xmid = 0.5 * (xmin + xmax); xh = 0.5 * (xmax - xmin)
        xx = xmid[:, None] - xh[:, None] * np.cos(th)[None, :]
        out[m1] = 2.0 * np.sum(wth * g(np.sqrt(xx)), axis=1)
    m2 = (dmin < 1.0) & (dmax > 1.0)
    if m2.any():
        xmin = dmin[m2] ** 2; xmax = dmax[m2] ** 2
        a = xmax - 1.0; eps = 1.0 - xmin
        c = np.cos(th2)[None, :]
        xx = xmax[:, None] - a[:, None] * np.sin(th2)[None, :] ** 2
        D = np.sqrt(a[:, None] * c ** 2 + eps[:, None])
        integ = np.sum(wth2 * (g(np.sqrt(xx)) - g1) * c / D, axis=1)
        sing = g1 * np.arcsin(1.0 / np.sqrt(1.0 + eps / a)) / np.sqrt(a)
        out[m2] = 4.0 * np.sqrt(a) * (sing + integ)
    return out


def _graded_panels(a, b, kink_lo, kink_hi, width, xw):
    edges = _panel_edges(a, b, width); m = len(edges) - 1
    x, w = xw; xs = []; ws = []
    for i, (p, q) in enumerate(zip(edges[:-1], edges[1:])):
        L = q - p
        if i == 0 and kink_lo:
            t = 0.5 * (x + 1.0); wt = 0.5 * w
            xs.append(p + L * t * t); ws.append(wt * 2.0 * L * t)
        elif i == m - 1 and kink_hi:
            t = 0.5 * (x + 1.0); wt = 0.5 * w
            xs.append(q - L * t * t); ws.append(wt * 2.0 * L * t)
        else:
            xs.append(0.5 * L * x + 0.5 * (p + q)); ws.append(0.5 * L * w)
    return np.concatenate(xs), np.concatenate(ws)


def _theta_conv_k(r, g, g1, xw_out, xw_in, width=0.25):
    """(theta * k)(r), r >= 1: int_0^1 r1 dr1 A_g(r, r1); theta = unit-disc indicator, k = g on d >= 1;
    xw_out and xw_in are the Gauss-Legendre nodes of the outer (r1) and inner (angular) quadratures."""
    if r >= 2.0:
        r1, w = _graded_panels(0.0, 1.0, False, False, width, xw_out)
        return float(np.sum(w * r1 * _chord_A(r, r1, g, g1, xw_in)))
    rk = r - 1.0; tot = 0.0
    if rk > 0.0:
        r1, w = _graded_panels(0.0, rk, False, False, width, xw_out); tot += np.sum(w * r1 * _chord_A(r, r1, g, g1, xw_in))
    r1, w = _graded_panels(rk, 1.0, True, False, width, xw_out); tot += np.sum(w * r1 * _chord_A(r, r1, g, g1, xw_in))
    return float(tot)


def _k_conv_k(r, fa, gb, gb1, rcut, xw_out, xw_in, width=0.25):
    """(k_a * k_b)(r) = int_1^rcut r1 k_a(r1) A_b(r, r1) dr1, sqrt cusps at r1 = r -/+ 1 handled."""
    pts = sorted(set([1.0] + [x for x in (r - 1.0, r + 1.0) if 1.0 < x < rcut] + [rcut]))
    tot = 0.0
    for p, q in zip(pts[:-1], pts[1:]):
        klo = abs(p - (r - 1.0)) < 1e-14
        khi = abs(q - (r + 1.0)) < 1e-14
        r1, w = _graded_panels(p, q, klo, khi, width, xw_out)
        tot += np.sum(w * r1 * fa(r1) * _chord_A(r, r1, gb, gb1, xw_in))
    return float(tot)


def _lens(r):
    r = np.asarray(r, dtype=float); out = np.zeros_like(r); m = r < 2.0
    out[m] = 2.0 * np.arccos(r[m] / 2.0) - 0.5 * r[m] * np.sqrt(4.0 - r[m] ** 2)
    return out


def _glgl(r, sigma, Gamma, rho):
    """(Gl * Gl)(r) = int dq q/(2 pi) J0(q r) Glbar(q)^2 (smooth, fast-decaying integrand)."""
    kap2 = 2.0 * np.pi * Gamma * rho
    qmax = max(60.0, 40.0 / max(sigma, 0.05))
    q, w = _panel_nodes(np.linspace(0.0, qmax, int(qmax / 0.05) + 1), 8)
    gb = (2.0 * np.pi * Gamma / (q * q * _S(q, sigma) + kap2)) ** 2
    r = np.atleast_1d(np.asarray(r, dtype=float)); out = np.empty_like(r)
    for i in range(0, len(r), 64):
        rr = r[i:i + 64]
        out[i:i + 64] = np.sum(w * q * gb * j0(np.outer(rr, q)), axis=1) / (2.0 * np.pi)
    return out


def _T_values(r, sigma, Gamma, rho, wspl):
    r = np.atleast_1d(np.asarray(r, dtype=float))
    kp = lambda d: np.exp(wspl(d)) - 1.0      # h+- for d >= 1
    km = lambda d: np.exp(-wspl(d)) - 1.0     # h++ for d >= 1
    kp1 = float(kp(1.0)); km1 = float(km(1.0))
    U = _rcut(Gamma, rho, sigma)
    L = _lens(r)
    xo = _leggauss(16); xi = _leggauss(32)
    tkp = np.array([_theta_conv_k(ri, kp, kp1, xo, xi) for ri in r])
    tkm = np.array([_theta_conv_k(ri, km, km1, xo, xi) for ri in r])
    kmm = np.array([_k_conv_k(ri, km, km, km1, U, xo, xi) for ri in r])
    kpp = np.array([_k_conv_k(ri, kp, kp, kp1, U, xo, xi) for ri in r])
    kmp = np.array([_k_conv_k(ri, km, kp, kp1, U, xo, xi) for ri in r])
    gg = _glgl(r, sigma, Gamma, rho)
    hpp = L - 2.0 * tkm + kmm          # h++ * h++
    hmm = L - 2.0 * tkp + kpp          # h+- * h+-
    hpm = L - tkp - tkm + kmp          # h++ * h+-
    Tpp = 0.5 * rho * (hpp + hmm - 2.0 * gg)
    Tpm = rho * (hpm + gg)
    return Tpp, Tpm


def _correlation_table(sigma, Gamma, rho):
    """Nodes r in [1, rcut] with h, T and H: graded Gauss-Legendre panels on [1, 2] and [2, rcut] whose
    quadratic substitutions on both sides of r = 2 absorb the (2 - r)^(3/2) cusp of the lens area."""
    wspl = _w_spline(sigma, Gamma, rho)
    U = _rcut(Gamma, rho, sigma)
    xw = _leggauss(16)
    r1, w1 = _graded_panels(1.0, 2.0, False, True, 0.5, xw)
    r2, w2 = _graded_panels(2.0, U, True, False, 0.5, xw)
    r = np.concatenate([r1, r2]); w = np.concatenate([w1, w2])
    wr = wspl(r); hpm = np.exp(wr) - 1.0; hpp = np.exp(-wr) - 1.0
    Tpp, Tpm = _T_values(r, sigma, Gamma, rho, wspl)
    Hpm = hpm + (hpm + 1.0) * Tpm; Hpp = hpp + (hpp + 1.0) * Tpp
    return dict(r=r, w=w, hpm=hpm, hpp=hpp, Tpp=Tpp, Tpm=Tpm, Hpm=Hpm, Hpp=Hpp, wspl=wspl)


def _oracle_structure_factors(q: "np.ndarray", sigma: float, Gamma: float, rho: float) -> "np.ndarray":
    sigma = _check_sigma(sigma); Gamma, rho = _check_state(Gamma, rho)
    q = np.atleast_1d(np.asarray(q, dtype=float))
    if q.ndim != 1 or np.any(q <= 0.0) or np.any(q > 20.0) or not np.all(np.isfinite(q)):
        raise ValueError("q must lie in (0, 20]")
    tab = _correlation_table(sigma, Gamma, rho)
    core = -2.0 * np.pi * j1(q) / q
    Hpp = core + _hankel_forward(q, tab["r"], tab["w"], tab["Hpp"])
    Hpm = core + _hankel_forward(q, tab["r"], tab["w"], tab["Hpm"])
    Spp = 0.5 + 0.25 * rho * Hpp
    Spm = 0.25 * rho * Hpm
    Szz = 1.0 + 0.5 * rho * (Hpp - Hpm)
    return np.vstack([Spp, Spm, Szz])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        {'setup': 'import numpy as np\nGamma, rho = 2.0, 0.10\nsigma = _oracle_splitting_length(Gamma, rho)\nq = np.array([0.5, 2.0, 5.0])\n',
         'call': 'structure_factors(q, sigma, Gamma, rho)',
         'gold_call': '_oracle_structure_factors(q, sigma, Gamma, rho)',
         'tol': 1e-08},
        {'setup': 'import numpy as np\nGamma, rho = 1.25, 0.15\nsigma = _oracle_splitting_length(Gamma, rho)\nq = np.array([0.25, 1.0, 3.0, 10.0])\n',
         'call': 'structure_factors(q, sigma, Gamma, rho)',
         'gold_call': '_oracle_structure_factors(q, sigma, Gamma, rho)',
         'tol': 1e-08},
        {'setup': 'import numpy as np\nGamma, rho = 4.0, 0.20\nsigma = 0.6\nq = np.array([1.5, 4.0])\n# boundary: the strongest coupling of the test set\n',
         'call': 'structure_factors(q, sigma, Gamma, rho)',
         'gold_call': '_oracle_structure_factors(q, sigma, Gamma, rho)',
         'tol': 1e-08},
        {'setup': 'import numpy as np\nGamma, rho = 0.5, 0.05\nsigma = 1.1\nq = np.array([0.5, 20.0])\n# edge: the largest admissible wavenumber q = 20\n',
         'call': 'structure_factors(q, sigma, Gamma, rho)',
         'gold_call': '_oracle_structure_factors(q, sigma, Gamma, rho)',
         'tol': 1e-08},
        {'setup': 'import numpy as np\nGamma, rho = 2.0, 0.10\nsigma = 0.7\nq = np.array([0.5, 25.0])\n# invalid input: a wavenumber above 20 must raise ValueError\ndef _catches_value_error(fn):\n    try:\n        fn()\n    except ValueError:\n        return np.array([1.0])\n    return np.array([0.0])\n',
         'call': '_catches_value_error(lambda: structure_factors(q, sigma, Gamma, rho))',
         'gold_call': '_catches_value_error(lambda: _oracle_structure_factors(q, sigma, Gamma, rho))'},
    ]
