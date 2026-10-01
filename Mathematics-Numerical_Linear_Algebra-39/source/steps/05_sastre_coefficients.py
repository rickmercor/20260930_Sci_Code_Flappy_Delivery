"""
Evaluates the real coefficients of the source's three-multiplication evaluation scheme for a degree-eight polynomial.

The evaluation scheme is what makes degree eight affordable at three matrix multiplications; its coefficient system is closed by one extra equation so that the coefficients stay real for every member of the family.

Returns
-------
A float64 array of shape (10,).
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def sastre_coefficients(b: "np.ndarray") -> "np.ndarray":
    r"""Evaluates the real coefficients of the source's three-multiplication evaluation scheme for a degree-eight polynomial.

    The source computes the matrix step function of a symmetric matrix $X_0$ with spectrum in $[0, \lambda_{lumo}]
    \cup [\lambda_{homo}, 1]$ by the recursive expansion $X_i = p_i(X_{i-1})$ with degree-eight component polynomials.
    Each component polynomial is a member $p_{(L, R; a, b)}$ of the SP8 family, $L + R = 7$, written in the source's
    parametrisation $p(x) = s_8 \sum_{k=0}^{7} (-1)^k e_k(s)\, x^{8-k}/(8-k) + s_9$ with $e_k$ the elementary
    symmetric polynomials of the stationary points $s_1, \dots, s_7$, so that $p'(x) = s_8 \prod_k (x - s_k)$. For
    construction bounds $0 \leq a < b \leq 1$ the member with $L$ stationary points in $[0, a]$ and $R$ in $[b, 1]$ is
    the unique solution of the equioscillatory conditions $p(s_k) = 0$ for odd $k \leq L$ and $p(s_k) = p(a)$ for even
    $k \leq L$; $p(s_{L+k}) = 1$ for odd $k \leq R$ and $p(s_{L+k}) = p(b)$ for even $k \leq R$; $p(0) = 0$ if $L$ is
    even and $p(0) = p(a)$ if $L$ is odd; $p(1) = 1$ if $R$ is even and $p(1) = p(b)$ if $R$ is odd; with the ordering
    $0 \leq s_L \leq \dots \leq s_1 \leq a < b \leq s_{L+1} \leq \dots \leq s_7 \leq 1$. At a bound sitting at its
    limit the conditions of that side are replaced by the modified ones: for $a = 0$, $s_k = 0$ for $k \leq L$ and
    $p(0) = 0$; for $b = 1$, $s_{L+k} = 1$ for $k \leq R$ and $p(1) = 1$; with both limits the member is the
    closed-form polynomial $p_{(L,R;0,1)}$ with $p' \propto x^L (x - 1)^R$. The expansion (the source's Algorithm 5.1)
    carries inner bounds $\lambda^{in}_{lumo} \leq \lambda^{in}_{homo}$ and outer bounds $\lambda^{out}_{lumo} \leq
    \lambda^{in}_{lumo}$, $\lambda^{in}_{homo} \leq \lambda^{out}_{homo}$ on the two eigenvalues, with the threshold
    $\kappa = 0.01$: the acceleration bounds are $a = \lambda^{out}_{lumo}$ if $\lambda^{out}_{lumo} > \kappa$ and $a
    = 0$ otherwise, and $b = \lambda^{out}_{homo}$ if $\lambda^{out}_{homo} < 1 - \kappa$ and $b = 1$ otherwise; the
    family is built for $(a, b)$; the member applied at iteration $i$ maximises $p(\lambda^{in}_{homo}) -
    p(\lambda^{in}_{lumo})$ over $L = 0, \dots, 7$ unless both inner bounds lie within $\kappa$ of their limits
    ($\lambda^{in}_{lumo} < \kappa$ and $1 - \kappa < \lambda^{in}_{homo}$), in which case $L = 3$ at odd and $L = 4$
    at even iterations; all four bounds are then mapped through the applied polynomial. The sign test $\mathrm{Tr}[X_i
    - X_i^2] \leq 0$ ends the expansion after any iteration; the parameter-free bound test is applied only once both
    acceleration bounds sit at their limits. The synthetic configurations are $16 \times 16$: from the source's Table
    1 energies (in the order $\lambda_{min}$, $\lambda^{out}_{homo}$, $\lambda^{in}_{homo}$, $\lambda^{in}_{lumo}$,
    $\lambda^{out}_{lumo}$, $\lambda_{max}$) the source's normalisation maps an energy $e$ to $(\lambda_{max} -
    e)/(\lambda_{max} - \lambda_{min})$; the spectrum places $\lambda_{lumo}$ at the midpoint of
    $[\lambda^{out}_{lumo}, \lambda^{in}_{lumo}]$ and $\lambda_{homo}$ at the midpoint of $[\lambda^{in}_{homo},
    \lambda^{out}_{homo}]$, with $n_{occ} = \mathrm{round}(16(1 - \mu))$, $\mu = (\lambda^{in}_{lumo} +
    \lambda^{in}_{homo})/2$, equidistant eigenvalues on $[\lambda_{homo}, 1]$ ($n_{occ}$ of them, endpoints included)
    and $16 - n_{occ}$ on $[0, \lambda_{lumo}]$, assembled on the orthonormal frame $Q$ of the QR factorisation of the
    integer matrix $M_{ij} = (((i+1)(j+2)(i+j+3) + a_{seed}(i+4)(j+5)) \bmod 47) - 23$, $i, j = 0..15$, each column of
    $Q$ sign-fixed so that its first entry of modulus above $10^{-12}$ is positive, and $X_0 =
    Q\,\mathrm{diag}(\lambda)\,Q^T$ symmetrised.

    This step evaluates the source's real evaluation coefficients of the three-multiplication scheme: $f_4 = b_8$,
    $c_1 = b_7/(2 f_4)$, $t_2 = b_6/f_4 - c_1^2$, $t_1 = b_5/f_4 - c_1 t_2$, $d_0 = (1 - t_2^2 + 4 b_4/f_4 - 4 c_1
    t_1)/4$, $e_2 = (t_2 + 1)/2$, $d_2 = (t_2 - 1)/2$, $e_1 = c_1 d_0 + t_1 e_2 - b_3/f_4$, $d_1 = t_1 - e_1$, $f_2 =
    b_2 - f_4 (d_0 e_2 + d_1 e_1)$, $f_1 = b_1 - f_4 d_0 e_1$, $f_0 = b_0$.

    Args:
        b: array of shape (9,), the monomial coefficients $b_0, \dots, b_8$ of a degree-eight polynomial with $b_8$
        nonzero.

    Returns:
        A numpy float64 array of shape $(10,)$: $c_1, d_0, d_1, d_2, e_1, e_2, f_0, f_1, f_2, f_4$ in this order.

    Raises:
        ValueError: if b does not have shape (9,) or its leading coefficient is zero.
    """
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np, math
from math import comb


def _kappa():
    """the source's deactivation threshold of the acceleration (Algorithm 5.1)"""
    return 0.01

def _dim():
    """the order of the synthetic matrices"""
    return 16

def _check_index(l):
    if isinstance(l, bool) or not isinstance(l, (int, np.integer)) or not (0 <= int(l) <= 7):
        raise ValueError("l must be an integer in 0..7")
    return int(l)

def _table(l):
    """the source's tabulated closed-form members for l = 4..7 (Table 2), monomial order b0..b8"""
    t = {4: [0, 0, 0, 0, 0, 56, -140, 120, -35], 5: [0, 0, 0, 0, 0, 0, 28, -48, 21], 6: [0, 0, 0, 0, 0, 0, 0, 8, -7], 7: [0, 0, 0, 0, 0, 0, 0, 0, 1]}
    return np.array(t[l], dtype=np.float64)

def _horner(b, x):
    y = 0.0
    for k in range(8, -1, -1):
        y = y * x + b[k]
    return y

def _stop_constant(l):
    return [np.inf, 28.0, 56.0, 82.0, 82.0, 56.0, 28.0, np.inf][l]

def _stop_order(l):
    return [1, 2, 3, 4, 4, 3, 2, 1][l]

def _esym(st):
    """elementary symmetric polynomials e_0..e_7 of the seven stationary points"""
    e = np.zeros(8); e[0] = 1.0
    for v in st:
        e[1:] = e[1:] + v * e[:-1]
    return e

def _pcoef(st):
    """monomial coefficients of P(x) = int_0^x prod_k (t - s_k) dt, the parametrisation (4.1) with unit scale and zero offset"""
    e = _esym(st); c = np.zeros(9)
    for k in range(8):
        c[8 - k] = ((-1.0) ** k) * e[k] / (8 - k)
    return c

def _pdiff(st, x1, x0):
    """int_{x0}^{x1} prod_k (t - s_k) dt by the four-point Gauss rule, exact for the degree-seven integrand and free of cancellation"""
    g = np.array([-0.8611363115940526, -0.3399810435848563, 0.3399810435848563, 0.8611363115940526])
    w = np.array([0.3478548451374538, 0.6521451548625461, 0.6521451548625461, 0.3478548451374538])
    m = 0.5 * (x1 + x0); h = 0.5 * (x1 - x0); t = m + h * g
    f = np.ones(4)
    for s in st:
        f = f * (t - s)
    return h * float(np.dot(w, f))

def _check_bounds(a, b):
    try:
        a = float(a); b = float(b)
    except (TypeError, ValueError):
        raise ValueError("bounds must be real numbers")
    if not (math.isfinite(a) and math.isfinite(b)) or a < 0.0 or b > 1.0 or not (a < b):
        raise ValueError("bounds must satisfy 0 <= lam_lumo < lam_homo <= 1")
    return a, b

def _alternation_sets(st, L, a, b):
    """lower and upper alternation points of the left interval and upper and lower points of the right interval"""
    R = 7 - L
    lo = ([0.0] if L % 2 == 0 else []) + [st[k - 1] for k in range(1, L + 1) if k % 2 == 1]
    up = [a] + [st[k - 1] for k in range(1, L + 1) if k % 2 == 0] + ([0.0] if L % 2 == 1 else [])
    ru = [st[L + k - 1] for k in range(1, R + 1) if k % 2 == 1] + ([1.0] if R % 2 == 0 else [])
    rl = [b] + [st[L + k - 1] for k in range(1, R + 1) if k % 2 == 0] + ([1.0] if R % 2 == 1 else [])
    return lo, up, ru, rl

def _free_index(L, a, b):
    fl = [] if a == 0.0 else list(range(L)); fr = [] if b == 1.0 else list(range(L, 7))
    return fl + fr

def _full_st(free, L, a, b):
    st = np.zeros(7)
    if b == 1.0: st[L:] = 1.0
    st[_free_index(L, a, b)] = free
    return st

def _residual_free(free, L, a, b):
    """scale-free form of the equioscillatory conditions: equal values of P on each alternation set, normalised by the amplitude of the interval"""
    R = 7 - L; st = _full_st(free, L, a, b)
    lo, up, ru, rl = _alternation_sets(st, L, a, b); res = []
    if a > 0.0 and L > 0:
        amp = _pdiff(st, a, st[0])
        for x in lo[1:]: res.append(_pdiff(st, x, lo[0]) / amp)
        for x in up[1:]: res.append(_pdiff(st, x, up[0]) / amp)
    if b < 1.0 and R > 0:
        amp = _pdiff(st, st[L], b)
        for x in ru[1:]: res.append(_pdiff(st, x, ru[0]) / amp)
        for x in rl[1:]: res.append(_pdiff(st, x, rl[0]) / amp)
    return np.array(res)

def _strict_ordered(st, L, a, b):
    """the ordering (4.7) with strict inequalities on the free stationary points"""
    R = 7 - L
    if L > 0 and a > 0.0:
        if not (st[0] < a and st[L - 1] > 0.0): return False
        if L > 1 and not np.all(np.diff(st[:L]) < 0.0): return False
    if R > 0 and b < 1.0:
        if not (st[L] > b and st[6] < 1.0): return False
        if R > 1 and not np.all(np.diff(st[L:]) > 0.0): return False
    return True

def _assemble(free, L, a, b):
    """the nine parameters: stationary points, scale s8 and offset s9 fixed by p = 0 at the lower-left level and p = 1 at the upper-right level"""
    st = _full_st(free, L, a, b)
    lo, up, ru, rl = _alternation_sets(st, L, a, b)
    xl = 0.0 if a == 0.0 else lo[0]
    xr = 1.0 if b == 1.0 else ru[0]
    s8 = 1.0 / _pdiff(st, xr, xl); s9 = -s8 * _pdiff(st, xl, 0.0)
    return np.concatenate([st, [s8, s9]])

def _jacobian(free, L, a, b):
    n = free.size; f0 = _residual_free(free, L, a, b); J = np.zeros((f0.size, n)); h = 1e-7
    for j in range(n):
        d = np.zeros(n); d[j] = h
        J[:, j] = (_residual_free(free + d, L, a, b) - _residual_free(free - d, L, a, b)) / (2.0 * h)
    return J

def _newton_free(L, a, b, z0):
    """damped Newton iteration on the scale-free system that keeps the ordering (4.7)"""
    z = z0.copy(); F = _residual_free(z, L, a, b); nf = float(np.max(np.abs(F)))
    for it in range(60):
        if nf < 1e-12: break
        try:
            dz = np.linalg.solve(_jacobian(z, L, a, b), -F)
        except np.linalg.LinAlgError:
            return z, False
        lam = 1.0; accepted = False
        while lam > 1e-4:
            zn = z + lam * dz
            if _strict_ordered(_full_st(zn, L, a, b), L, a, b):
                Fn = _residual_free(zn, L, a, b); nfn = float(np.max(np.abs(Fn)))
                if nfn < nf or lam < 0.1:
                    z, F, nf = zn, Fn, nfn; accepted = True; break
            lam *= 0.5
        if not accepted: return z, False
    for it in range(3):
        try:
            dz = np.linalg.solve(_jacobian(z, L, a, b), -_residual_free(z, L, a, b))
        except np.linalg.LinAlgError:
            break
        zn = z + dz
        if _strict_ordered(_full_st(zn, L, a, b), L, a, b) and np.max(np.abs(_residual_free(zn, L, a, b))) <= np.max(np.abs(_residual_free(z, L, a, b))): z = zn
    nf = float(np.max(np.abs(_residual_free(z, L, a, b))))
    return z, (nf < 1e-11 and _strict_ordered(_full_st(z, L, a, b), L, a, b))

def _guess(L, a, b, kind, power):
    """Chebyshev extrema (kind 0) or Chebyshev zeros (kind 1) on each interval, the relative positions raised to a power, as the initial stationary points"""
    R = 7 - L; st = np.zeros(7)
    if L > 0 and a > 0.0:
        k = np.arange(1, L + 1)
        u = 0.5 * (1.0 - np.cos(np.pi * k / (L + 1))) if kind == 0 else 0.5 * (1.0 - np.cos(np.pi * (2 * k - 1) / (2 * L + 2)))
        st[:L] = (a * u ** power)[::-1]
    if R > 0 and b < 1.0:
        k = np.arange(1, R + 1)
        u = 0.5 * (1.0 - np.cos(np.pi * k / (R + 1))) if kind == 0 else 0.5 * (1.0 - np.cos(np.pi * (2 * k - 1) / (2 * R + 2)))
        st[L:] = b + (1.0 - b) * (1.0 - (1.0 - u) ** power)
    return st[_free_index(L, a, b)]

def _solve_equioscillatory(L, a, b):
    """the unique solution of the equioscillatory conditions with the ordering (4.7), from deterministic Chebyshev starts of increasing compression"""
    if len(_free_index(L, a, b)) == 0:
        return _assemble(np.zeros(0), L, a, b)
    for power in (1.0, 1.5, 0.75, 2.0, 0.5, 3.0):
        for kind in (0, 1):
            z, ok = _newton_free(L, a, b, _guess(L, a, b, kind, power))
            if ok: return _assemble(z, L, a, b)
    raise ValueError("the equioscillatory conditions could not be solved for L=%d on (%g, %g)" % (L, a, b))

def _coefficients(s):
    c = _pcoef(s[:7]) * s[7]; c[0] = c[0] + s[8]
    return c

def _transform(energies):
    """the source's normalisation (2.2): an eigenvalue e of H maps to (lmax - e)/(lmax - lmin); Table 1 order lmin, out_homo, in_homo, in_lumo, out_lumo, lmax"""
    e = np.asarray(energies, dtype=np.float64).ravel()
    if e.size != 6 or not np.all(np.isfinite(e)): raise ValueError("energies must hold six finite values")
    lmin, oh, ih, il, ol, lmax = e
    if not (lmin < oh <= ih < il <= ol < lmax): raise ValueError("energies must be ordered lmin < out_homo <= in_homo < in_lumo <= out_lumo < lmax")
    T = (lmax - np.array([il, ih, ol, oh])) / (lmax - lmin)
    return T   # in_lumo, in_homo, out_lumo, out_homo on [0, 1]

def _build(bounds, seed):
    """synthetic 16 by 16 matrix consistent with the bounds: the extreme eigenvalues of each interval at the midpoints of the bound intervals, the rest equidistant, on the sign-fixed QR frame of the mod-47 integer constructor"""
    n = _dim(); inl, inh, outl, outh = bounds
    lam_lumo = 0.5 * (outl + inl); lam_homo = 0.5 * (inh + outh)
    mu = 0.5 * (inl + inh); nocc = int(round(n * (1.0 - mu)))
    lam = np.concatenate([np.linspace(0.0, lam_lumo, n - nocc), np.linspace(lam_homo, 1.0, nocc)])
    M = np.empty((n, n))
    for i in range(n):
        for j in range(n):
            M[i, j] = (((i + 1) * (j + 2) * (i + j + 3) + seed * (i + 4) * (j + 5)) % 47) - 23
    Q, _r = np.linalg.qr(M)
    for c in range(n):
        col = Q[:, c]
        for v in col:
            if abs(v) > 1e-12:
                if v < 0:
                    Q[:, c] = -col
                break
    X0 = Q @ np.diag(lam) @ Q.T
    return (X0 + X0.T) / 2.0, nocc

def _oracle_sastre_coefficients(b: "np.ndarray") -> "np.ndarray":
    bb = np.asarray(b, dtype=np.float64)
    if bb.shape != (9,) or bb[8] == 0.0:
        raise ValueError("need 9 monomial coefficients with b8 nonzero")
    f4 = bb[8]
    c1 = bb[7] / (2.0 * f4)
    t2 = bb[6] / f4 - c1 ** 2
    t1 = bb[5] / f4 - c1 * t2
    d0 = 0.25 * (1.0 - t2 ** 2 + 4.0 * bb[4] / f4 - 4.0 * c1 * t1)
    e2 = 0.5 * (t2 + 1.0)
    d2 = 0.5 * (t2 - 1.0)
    e1 = c1 * d0 + t1 * e2 - bb[3] / f4
    d1 = t1 - e1
    f2 = bb[2] - f4 * (d0 * e2 + d1 * e1)
    f1 = bb[1] - f4 * d0 * e1
    f0 = bb[0]
    return np.array([c1, d0, d1, d2, e1, e2, f0, f1, f2, f4], dtype=np.float64)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {"setup":"import numpy as np\nb = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 56.0, -140.0, 120.0, -35.0])\n","call":"sastre_coefficients(b)","gold_call":"_oracle_sastre_coefficients(b)","tol":1e-10},
        {"setup":"import numpy as np\nb = np.array([0.265751390362448, -18.144270417601728, 406.2979779085488, -3461.0224569760653, 14442.745122636843, -32614.753854260664, 40621.59851616709, -26211.140158434806, 6835.153371986298])\n","call":"sastre_coefficients(b)","gold_call":"_oracle_sastre_coefficients(b)","tol":1e-6},
        {"setup":"import numpy as np\nb = np.array([0.0, 0.0, 0.0, 56.0, -210.0, 336.0, -280.0, 120.0, -21.0])\n","call":"sastre_coefficients(b)","gold_call":"_oracle_sastre_coefficients(b)","tol":1e-10},
        {"setup":"import numpy as np\n# boundary: the pure monomial x^8, whose scheme coefficients reduce to the closing equation alone\nb = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])\n","call":"sastre_coefficients(b)","gold_call":"_oracle_sastre_coefficients(b)","tol":1e-10},
        {"setup":"import numpy as np\n# invalid input: a vanishing leading coefficient must raise ValueError\ndef _catches_value_error(fn):\n    try:\n        fn()\n    except ValueError:\n        return np.array([1.0])\n    return np.array([0.0])\n","call":"_catches_value_error(lambda: sastre_coefficients(np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])))","gold_call":"_catches_value_error(lambda: _oracle_sastre_coefficients(np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])))"},
    ]
