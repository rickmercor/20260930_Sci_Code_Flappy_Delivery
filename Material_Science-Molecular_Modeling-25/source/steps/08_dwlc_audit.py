"""
Builds a 27-regime physical response tensor and reduces it to a source-dependent robustness audit.

The canonical record exercises all seven scientific steps. The regime tensor then tests coupled changes in stiffness, anharmonicity and tension, so the final score depends on the source-derived closure, inverse reading, extension, correlation and nematic definitions rather than on one state-point lookup.

Returns
-------
A float64 array of shape (15,).
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def dwlc_audit(lp_over_a: float, c: float, f: float, Ns: int, n: int) -> "np.ndarray":
    r"""lp_over_a, c, f, Ns, n: as in the marginals step.
    Build the canonical eight-entry physical record from all seven earlier public functions. Let
    im = Ns//2 and h = n//4. The record is (B_pi, P, K_h, l_1, l_h, X, R, Q): antiparallel bond
    energy, mid-chain peak density times n, closure exponent at the (h,h) corner with signed
    indices r=s=h, endpoint apparent persistence values, reduced extension, RMS end-to-end
    distance divided by the contour length Ns*a, and mid-chain nematic order.

    Next evaluate 27 regimes in lexicographic order over lp_over_a*(0.5,1,2), c-axis
    (0,c,2*c+0.1), and f-axis (f-0.8,f,f+0.8). If c is zero, use (0,0.1,0.3). For each regime
    form a response row (X,Q,P,K_h,l_1,l_h). Scale each column by sqrt(mean(column**2)) and center
    the scaled columns. Let s be the descending singular values and
    H = -sum(p*log(p))/log(6), p=s**2/sum(s**2). Reshape the scaled uncentered response to
    (3,3,3,6), contract the first three axes with q=(1,-2,1)/sqrt(6), and call its Euclidean norm M.
    Let W=sum((r+j/10)*response[r-1,j-1]) for one-based r=1..27 and j=1..6. Let D be the NumPy
    default-linear 0.9 quantile minus 0.1 quantile of l_h-l_1. Define
    J=(H+M+abs(D)/(1+abs(D)))/(3+abs(Q)).

    Returns a numpy float64 array of shape (15,) containing the canonical record followed by
    (W,s[0],s[-1],H,M,D,J).
    Raises:
        ValueError: on a non-finite input, a non-positive lp_over_a, a negative c, invalid n,
        fewer than four segments, or a non-finite or degenerate audit reduction.
    """
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _fin(x, name):
    v = float(x)
    if not np.isfinite(v):
        raise ValueError("%s must be finite" % name)
    return v


def _nonneg(x, name):
    v = _fin(x, name)
    if v < 0.0:
        raise ValueError("%s must be non-negative" % name)
    return v


def _int(x, name, lo=0):
    if isinstance(x, bool) or not np.isscalar(x):
        raise ValueError("%s must be an integer scalar" % name)
    v = float(x)
    if not np.isfinite(v) or v != int(v):
        raise ValueError("%s must be an integer" % name)
    v = int(v)
    if v < lo:
        raise ValueError("%s must be at least %d" % (name, lo))
    return v


def _grid(n):
    eps = 2.0 * np.pi / n
    return eps, np.arange(n) * eps


def _signed(n):
    return ((np.arange(n) + n // 2) % n) - n // 2


def _bend(n):
    """Bend angle of every ordered pair of directions, reduced to the smaller of the two arcs."""
    eps = 2.0 * np.pi / n
    r = np.arange(n)
    d = (r[:, None] - r[None, :]).astype(np.float64)
    d = (d + n // 2) % n - n // 2
    return d * eps


def _bond_energy(lp_over_a, c, n):
    """Reduced bending energy of one bond. The harmonic part is the continuum wormlike-chain
    energy on the discretization, with the plane relation between stiffness and persistence
    length folded in; the anharmonic factor multiplies it by one plus c times the squared bend
    angle, so c is measured in inverse squared radians and c = 0 restores the harmonic bond."""
    D = _bend(n)
    return 0.5 * (lp_over_a / 2.0) * D * D * (1.0 + c * D * D)


def _fb(lp_over_a, c, f, Ns, n):
    """Forward and backward transfer vectors of the open chain: the tension acts on each of the
    Ns segments, the bending on each of the Ns-1 bonds joining them."""
    W = np.exp(-_bond_energy(lp_over_a, c, n))
    _, th = _grid(n)
    s = np.exp(f * np.cos(th))
    fwd = np.empty((Ns, n), dtype=np.float64)
    bwd = np.empty((Ns, n), dtype=np.float64)
    fwd[0] = s
    for i in range(1, Ns):
        fwd[i] = (W.T @ fwd[i - 1]) * s
    bwd[Ns - 1] = 1.0
    for i in range(Ns - 2, -1, -1):
        bwd[i] = W @ (s * bwd[i + 1])
    Z = float(fwd[Ns - 1].sum())
    if not np.isfinite(Z) or Z <= 0.0:
        raise ValueError("partition function underflowed or overflowed")
    return W, s, fwd, bwd, Z


def _marginals(lp_over_a, c, f, Ns, n):
    _, _, fwd, bwd, _ = _fb(lp_over_a, c, f, Ns, n)
    m = fwd * bwd
    tot = m.sum(axis=1, keepdims=True)
    if not np.all(np.isfinite(tot)) or np.any(tot <= 0.0):
        raise ValueError("a site marginal failed to normalize")
    return m / tot


def _log_pair(lp_over_a, c, f, Ns, n, bond):
    """Log of the unnormalized neighbouring pair weight on bond (bond, bond+1). Working in logs
    keeps a stiff chain from underflowing, and the transfer messages cancel identically when the
    connected combination is formed."""
    W, s, fwd, bwd, _ = _fb(lp_over_a, c, f, Ns, n)
    B = _bond_energy(lp_over_a, c, n)
    g = s * bwd[bond + 1]
    lf = fwd[bond]
    if np.any(lf <= 0.0) or np.any(g <= 0.0):
        raise ValueError("a transfer weight underflowed to zero")
    return np.log(lf)[:, None] - B + np.log(g)[None, :]


def _block(n):
    """The central block of signed indices on which the connected combination is unambiguous."""
    rs = _signed(n)
    h = n // 4
    order = np.argsort(rs)
    sel = order[np.abs(rs[order]) <= h]
    return sel, np.sort(rs[np.abs(rs) <= h]), int(np.where(rs == 0)[0][0]), h


def _closure(lp_over_a, c, f, Ns, n, bond):
    L = _log_pair(lp_over_a, c, f, Ns, n, bond)
    sel, idx, ref, _ = _block(n)
    K = (L[np.ix_(sel, sel)] + L[ref, ref]) - (L[sel, ref][:, None] + L[ref, sel][None, :])
    if not np.all(np.isfinite(K)):
        raise ValueError("non-finite closure exponent")
    return K, idx


def _apparent(lp_over_a, c, f, Ns, n, bond):
    """Read the connected combination as if the bond were harmonic. For a harmonic bond the
    source's relation gives a single coupling, so dividing by the product of the two signed
    indices returns it; for any other bond the result varies with the bend angle, and that
    variation is what the harmonic reading misses."""
    K, idx = _closure(lp_over_a, c, f, Ns, n, bond)
    eps = 2.0 * np.pi / n
    kmax = int(idx.max())
    out = np.empty(kmax, dtype=np.float64)
    for k in range(1, kmax + 1):
        ip = int(np.where(idx == k)[0][0])
        im = int(np.where(idx == -k)[0][0])
        out[k - 1] = 2.0 * (K[ip, im] / (-(k * k))) / (eps * eps)
    if not np.all(np.isfinite(out)):
        raise ValueError("non-finite apparent persistence length")
    return out


def _corr(lp_over_a, c, f, Ns, n, i0):
    W, s, fwd, bwd, Z = _fb(lp_over_a, c, f, Ns, n)
    _, th = _grid(n)
    u = np.exp(1j * th)
    out = np.empty(Ns, dtype=np.float64)
    out[i0] = 1.0
    v = (fwd[i0] * u).astype(np.complex128)
    for j in range(i0 + 1, Ns):
        v = (W.T @ v) * s
        out[j] = float(np.real(np.sum(v * np.conj(u) * bwd[j])) / Z)
    v = (bwd[i0] * u).astype(np.complex128)
    for j in range(i0 - 1, -1, -1):
        v = W @ (v * s)
        out[j] = float(np.real(np.sum(v * np.conj(u) * fwd[j])) / Z)
    return out

def _oracle_dwlc_audit(lp_over_a: float, c: float, f: float, Ns: int, n: int) -> "np.ndarray":
    lp = _fin(lp_over_a, "lp_over_a")
    if lp <= 0.0:
        raise ValueError("lp_over_a must be positive")
    cc = _nonneg(c, "c"); ff = _fin(f, "f")
    nn = _int(n, "n", 8)
    if nn % 4:
        raise ValueError("n must be divisible by four")
    ns = _int(Ns, "Ns", 4)
    im = ns // 2
    h = nn // 4

    B = _oracle_dwlc_bond_energy(lp, cc, nn)
    m = _oracle_dwlc_marginals(lp, cc, ff, ns, nn)
    K = _oracle_dwlc_closure_exponent(lp, cc, ff, ns, nn, im)
    ap = _oracle_dwlc_apparent_persistence(lp, cc, ff, ns, nn, im)
    ex = _oracle_dwlc_extension(lp, cc, ff, ns, nn)
    nem = _oracle_dwlc_nematic_profile(lp, cc, ff, ns, nn)
    r2 = 0.0
    for i in range(ns):
        r2 += float(np.sum(_oracle_dwlc_tangent_correlation(lp, cc, ff, ns, nn, i)))
    if r2 <= 0.0 or not np.isfinite(r2):
        raise ValueError("non-positive mean square end to end distance")
    rms = np.sqrt(r2) / ns
    if rms * rms < ex[ns] * ex[ns] - 1e-12:
        raise ValueError("mean square size is below squared mean extension")
    if cc == 0.0 and abs(float(ap[0]) - lp) > 1e-8 * max(1.0, lp):
        raise ValueError("harmonic inverse reading did not recover persistence")
    base = np.array([B[0, nn//2], np.max(m[im])*nn, K[-1,-1], ap[0], ap[h-1],
                     ex[ns], rms, nem[im]], dtype=np.float64)

    lps = lp * np.array([0.5, 1.0, 2.0], dtype=np.float64)
    cs = np.array([0.0, cc, 2.0*cc + 0.1], dtype=np.float64) if cc > 0.0 else np.array([0.0, 0.1, 0.3])
    fs = np.array([ff - 0.8, ff, ff + 0.8], dtype=np.float64)
    rows = []
    for lv in lps:
        for cv in cs:
            for fv in fs:
                mm = _oracle_dwlc_marginals(lv, cv, fv, ns, nn)
                kk = _oracle_dwlc_closure_exponent(lv, cv, fv, ns, nn, im)
                aa = _oracle_dwlc_apparent_persistence(lv, cv, fv, ns, nn, im)
                xx = _oracle_dwlc_extension(lv, cv, fv, ns, nn)[ns]
                qq = _oracle_dwlc_nematic_profile(lv, cv, fv, ns, nn)[im]
                rows.append([xx, qq, np.max(mm[im])*nn, kk[-1,-1], aa[0], aa[-1]])
    response = np.asarray(rows, dtype=np.float64)
    scale = np.sqrt(np.mean(response*response, axis=0))
    if np.any(scale <= 0.0) or not np.all(np.isfinite(scale)):
        raise ValueError("degenerate response scale")
    z = response / scale
    zc = z - np.mean(z, axis=0)
    s = np.linalg.svd(zc, compute_uv=False)
    p = s*s / np.sum(s*s)
    H = -float(np.sum(p*np.log(p))) / np.log(6.0)
    tensor = z.reshape(3,3,3,6)
    q = np.array([1.0,-2.0,1.0], dtype=np.float64) / np.sqrt(6.0)
    mixed = np.einsum("i,j,k,ijkm->m", q, q, q, tensor)
    M = float(np.linalg.norm(mixed))
    weights = np.arange(1.0,28.0)[:,None] + np.arange(1.0,7.0)[None,:]/10.0
    W = float(np.sum(weights*response))
    gaps = response[:,5] - response[:,4]
    D = float(np.quantile(gaps,0.9) - np.quantile(gaps,0.1))
    J = (H + M + abs(D)/(1.0+abs(D))) / (3.0 + abs(float(base[7])))
    out = np.concatenate([base, np.array([W,s[0],s[-1],H,M,D,J])])
    if not np.all(np.isfinite(out)):
        raise ValueError("non-finite audit record")
    return out

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {"setup":"import numpy as np\n","call":"dwlc_audit(2.2,0.2,0.8,32,96)","gold_call":"_oracle_dwlc_audit(2.2,0.2,0.8,32,96)"},
        {"setup":"import numpy as np\n","call":"dwlc_audit(1.0,0.6,1.0,20,64)","gold_call":"_oracle_dwlc_audit(1.0,0.6,1.0,20,64)"},
        {"setup":"import numpy as np\n","call":"dwlc_audit(0.5,1.0,2.5,16,48)","gold_call":"_oracle_dwlc_audit(0.5,1.0,2.5,16,48)"},
        {"setup":"import numpy as np\n# boundary: a harmonic bond, where the backward reading must return the input persistence length\n","call":"dwlc_audit(2.0,0.0,1.0,12,48)","gold_call":"_oracle_dwlc_audit(2.0,0.0,1.0,12,48)"},
        {"setup":"import numpy as np\n# invalid input: fewer than four segments leaves no mid chain bond to audit\ndef _probe(fn):\n    try:\n        fn(1.0,0.1,1.0,3,48)\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\n","call":"_probe(dwlc_audit)","gold_call":"_probe(_oracle_dwlc_audit)"},
    ]
