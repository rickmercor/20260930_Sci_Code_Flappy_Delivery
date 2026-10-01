"""
Measures the connected part of the neighbouring pair distribution on the central block of directions.

The source's exact bond-local closure distinguishes the local interaction from chain-end and external-field effects.

Returns
-------
A float64 array of shape (m, m) with m = 2*(n//4)+1.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def dwlc_closure_exponent(lp_over_a: float, c: float, f: float, Ns: int, n: int, bond: int) -> "np.ndarray":
    r"""lp_over_a, c, f, Ns, n: as in the marginals step. bond: integer in $0 \le \mathrm{bond}
    \le N_s - 2$, the bond joining segments bond and bond+1.

    Let $C^{rs}$ be the joint probability that segment bond points along direction r and segment
    bond+1 along direction s. Compute the source's exact bond-local connected closure exponent
    from this neighboring-pair distribution, using direction 0 as the reference state. The
    result must remain numerically finite for the tested stiff-chain cases.

    Return it on the central block of signed indices $|r_\sigma| \le n/4$ and
    $|s_\sigma| \le n/4$, where the signed index $r_\sigma$ runs from $-n/2$ to $n/2 - 1$ and the
    bend angle of a pair is unambiguous.

    Returns a numpy float64 array of shape $(m, m)$ with $m = 2\lfloor n/4\rfloor + 1$, rows and
    columns ordered by increasing signed index. It is symmetric and vanishes on the row and the
    column of the reference state.

    Raises:
        ValueError: on a non-finite input, a non-positive lp_over_a, a negative c, an invalid n, a
            non-integer or smaller than 2 value of Ns, or a bond index outside its range.
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

def _oracle_dwlc_closure_exponent(lp_over_a: float, c: float, f: float, Ns: int, n: int, bond: int) -> "np.ndarray":
    lp = _fin(lp_over_a, "lp_over_a")
    if lp <= 0.0:
        raise ValueError("lp_over_a must be positive")
    cc = _nonneg(c, "c"); ff = _fin(f, "f")
    nn = _int(n, "n", 8)
    if nn % 4:
        raise ValueError("n must be divisible by four")
    ns = _int(Ns, "Ns", 2)
    bd = _int(bond, "bond", 0)
    if bd > ns - 2:
        raise ValueError("bond must be at most Ns-2")
    K, _ = _closure(lp, cc, ff, ns, nn, bd)
    return np.asarray(K, dtype=np.float64)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {"setup":"import numpy as np\n","call":"dwlc_closure_exponent(2.2,0.2,0.8,32,96,16)","gold_call":"_oracle_dwlc_closure_exponent(2.2,0.2,0.8,32,96,16)"},
        {"setup":"import numpy as np\n","call":"dwlc_closure_exponent(1.0,0.0,1.0,20,64,0)","gold_call":"_oracle_dwlc_closure_exponent(1.0,0.0,1.0,20,64,0)"},
        {"setup":"import numpy as np\n","call":"dwlc_closure_exponent(0.5,1.2,-2.0,12,48,7)","gold_call":"_oracle_dwlc_closure_exponent(0.5,1.2,-2.0,12,48,7)"},
        {"setup":"import numpy as np\n# boundary: the last bond of the chain, where the backward weight is trivial\n","call":"dwlc_closure_exponent(2.0,0.3,1.0,10,32,8)","gold_call":"_oracle_dwlc_closure_exponent(2.0,0.3,1.0,10,32,8)"},
        {"setup":"import numpy as np\n# invalid input: a bond index past the last bond\ndef _probe(fn):\n    try:\n        fn(1.0,0.1,1.0,10,32,9)\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\n","call":"_probe(dwlc_closure_exponent)","gold_call":"_probe(_oracle_dwlc_closure_exponent)"},
    ]
