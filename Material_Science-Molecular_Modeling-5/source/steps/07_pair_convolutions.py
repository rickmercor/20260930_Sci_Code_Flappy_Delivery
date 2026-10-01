"""
Return the first-order correlation corrections T++(r) and T+-(r) at the distances r >= 1 for a symmetric 1:1 electrolyte in which each species has number density rho/2. For species i and j, T_ij is the sum over the third species n of n_n times the two-dimensional convolution of the Mayer functions h_in and h_nj, minus q_i q_j q_n^2 n_n times the convolution of the screened kernel Gl with itself, where the convolution of two radial functions f and g is the integral over the plane of f(r') g(|r - r'|) and the Mayer functions include the hard-core disk on which they equal minus one. The result must be accurate to 1e-9 in absolute terms; the convolutions of functions that jump at the contact distance have to be evaluated with the geometry of the overlapping disks made explicit, not by a truncated Fourier quadrature. Reject invalid parameters and distances below one or non-finite.

Beyond the Gaussian level, the first correction to the pair distribution is the chain of two Mayer functions through an intermediate ion, summed over the species of that ion with its density, from which the ring already counted by the screened kernel is subtracted so that nothing is double counted. Two hard disks that both exclude a third ion overlap on a lens whose area is elementary, and it is that lens, together with the parts of the convolution outside it, that sets the size of the correction at contact.

Returns
-------
A (2, n) float64 array whose first row is T++ and whose second row is T+- at the n distances.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def pair_convolutions(r: "np.ndarray", sigma: float, Gamma: float, rho: float) -> "np.ndarray":
    """Return the first-order correlation corrections T++(r) and T+-(r) at the distances r >= 1 for
    a symmetric 1:1 electrolyte in which each species has number density rho/2. For species i
    and j, T_ij is the sum over the third species n of n_n times the two-dimensional convolution
    of the Mayer functions h_in and h_nj, minus q_i q_j q_n^2 n_n times the convolution of the
    screened kernel Gl with itself, where the convolution of two radial functions f and g is the
    integral over the plane of f(r') g(|r - r'|) and the Mayer functions include the hard-core
    disk on which they equal minus one. The result must be accurate to 1e-9 in absolute terms;
    the convolutions of functions that jump at the contact distance have to be evaluated with
    the geometry of the overlapping disks made explicit, not by a truncated Fourier quadrature.
    Reject invalid parameters and distances below one or non-finite.

    Args:
        r: array-like of shape (n,) of finite distances r >= 1 in units of the disk diameter.
        sigma: non-negative finite float, the splitting length in units of the disk diameter (sigma = 0 is the
            Debye-Hueckel limit of the scheme).
        Gamma: positive finite float, the Coulomb coupling.
        rho: positive finite float, the total reduced ion density rho a^2 (both species together).

    Returns:
        A numpy float64 array of shape (2, n): the first row is T++(r) = (rho/2)[h++ * h++ + h+- * h+- -
        2 Gl * Gl] and the second row T+-(r) = rho [h++ * h+- + Gl * Gl], with * the two-dimensional
        convolution and the Mayer functions equal to -1 inside the hard core, at the n distances, accurate to
        1e-9 in absolute terms.

    Raises:
        ValueError: if r is not a one-dimensional array of finite values with every entry >= 1; if sigma is
            negative or not finite; or if Gamma or rho is not a positive finite number.
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


def _oracle_pair_convolutions(r: "np.ndarray", sigma: float, Gamma: float, rho: float) -> "np.ndarray":
    sigma = _check_sigma(sigma); Gamma, rho = _check_state(Gamma, rho)
    r = np.atleast_1d(np.asarray(r, dtype=float))
    if r.ndim != 1 or np.any(r < 1.0) or not np.all(np.isfinite(r)):
        raise ValueError("r must be >= 1")
    wspl = _w_spline(sigma, Gamma, rho)
    Tpp, Tpm = _T_values(r, sigma, Gamma, rho, wspl)
    return np.vstack([Tpp, Tpm])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        {'setup': 'import numpy as np\nGamma, rho = 2.0, 0.10\nsigma = _oracle_splitting_length(Gamma, rho)\nr = np.array([1.0, 1.5, 2.0, 2.5, 4.0])\n',
         'call': 'pair_convolutions(r, sigma, Gamma, rho)',
         'gold_call': '_oracle_pair_convolutions(r, sigma, Gamma, rho)',
         'tol': 1e-08},
        {'setup': 'import numpy as np\nGamma, rho = 1.25, 0.15\nsigma = _oracle_splitting_length(Gamma, rho)\nr = np.array([1.0, 3.0, 6.0])\n',
         'call': 'pair_convolutions(r, sigma, Gamma, rho)',
         'gold_call': '_oracle_pair_convolutions(r, sigma, Gamma, rho)',
         'tol': 1e-08},
        {'setup': 'import numpy as np\nGamma, rho = 4.0, 0.20\nsigma = 0.6\nr = np.array([1.0, 1.25, 2.0])\n# boundary: contact and the kink distance r = 2 where the exclusion disks stop overlapping\n',
         'call': 'pair_convolutions(r, sigma, Gamma, rho)',
         'gold_call': '_oracle_pair_convolutions(r, sigma, Gamma, rho)',
         'tol': 1e-08},
        {'setup': 'import numpy as np\nGamma, rho = 0.5, 0.05\nsigma = 1.1\nr = np.array([1.0, 2.0, 5.0])\n# edge: a splitting length above the disk diameter\n',
         'call': 'pair_convolutions(r, sigma, Gamma, rho)',
         'gold_call': '_oracle_pair_convolutions(r, sigma, Gamma, rho)',
         'tol': 1e-08},
        {'setup': 'import numpy as np\nGamma, rho = 2.0, 0.10\nsigma = 0.7\nr = np.array([0.5, 2.0])\n# invalid input: a distance inside the hard core must raise ValueError\ndef _catches_value_error(fn):\n    try:\n        fn()\n    except ValueError:\n        return np.array([1.0])\n    return np.array([0.0])\n',
         'call': '_catches_value_error(lambda: pair_convolutions(r, sigma, Gamma, rho))',
         'gold_call': '_catches_value_error(lambda: _oracle_pair_convolutions(r, sigma, Gamma, rho))'},
    ]
