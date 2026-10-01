#!/bin/bash
# Oracle: writes the gold solution so the verifier scores reward 1.0.
cat > /app/solution.py <<'SCICODE_GOLD_EOF'
import numpy as np
try:
    import scipy, scipy.linalg, scipy.sparse, scipy.optimize, scipy.spatial, scipy.integrate, scipy.stats, scipy.special, scipy.signal
except Exception:
    pass
import math

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

def dwlc_bond_energy(lp_over_a: float, c: float, n: int) -> "np.ndarray":
    lp = _fin(lp_over_a, "lp_over_a")
    if lp <= 0.0:
        raise ValueError("lp_over_a must be positive")
    cc = _nonneg(c, "c")
    nn = _int(n, "n", 8)
    if nn % 4:
        raise ValueError("n must be divisible by four")
    B = _bond_energy(lp, cc, nn)
    if not np.all(np.isfinite(B)):
        raise ValueError("non-finite bond energy")
    return np.asarray(B, dtype=np.float64)

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

def dwlc_marginals(lp_over_a: float, c: float, f: float, Ns: int, n: int) -> "np.ndarray":
    lp = _fin(lp_over_a, "lp_over_a")
    if lp <= 0.0:
        raise ValueError("lp_over_a must be positive")
    cc = _nonneg(c, "c"); ff = _fin(f, "f")
    nn = _int(n, "n", 8)
    if nn % 4:
        raise ValueError("n must be divisible by four")
    ns = _int(Ns, "Ns", 2)
    m = _marginals(lp, cc, ff, ns, nn)
    if not np.all(np.isfinite(m)):
        raise ValueError("non-finite angular density")
    return np.asarray(m, dtype=np.float64)

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

def dwlc_extension(lp_over_a: float, c: float, f: float, Ns: int, n: int) -> "np.ndarray":
    lp = _fin(lp_over_a, "lp_over_a")
    if lp <= 0.0:
        raise ValueError("lp_over_a must be positive")
    cc = _nonneg(c, "c"); ff = _fin(f, "f")
    nn = _int(n, "n", 8)
    if nn % 4:
        raise ValueError("n must be divisible by four")
    ns = _int(Ns, "Ns", 2)
    m = _marginals(lp, cc, ff, ns, nn)
    _, th = _grid(nn)
    p = m @ np.cos(th)
    out = np.empty(ns + 1, dtype=np.float64)
    out[:ns] = p
    out[ns] = float(np.mean(p))
    if not np.all(np.isfinite(out)):
        raise ValueError("non-finite extension")
    return out

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

def dwlc_closure_exponent(lp_over_a: float, c: float, f: float, Ns: int, n: int, bond: int) -> "np.ndarray":
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

def dwlc_apparent_persistence(lp_over_a: float, c: float, f: float, Ns: int, n: int, bond: int) -> "np.ndarray":
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
    out = _apparent(lp, cc, ff, ns, nn, bd)
    if np.any(out <= 0.0):
        raise ValueError("non-positive apparent persistence length")
    return np.asarray(out, dtype=np.float64)

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

def dwlc_tangent_correlation(lp_over_a: float, c: float, f: float, Ns: int, n: int, i0: int) -> "np.ndarray":
    lp = _fin(lp_over_a, "lp_over_a")
    if lp <= 0.0:
        raise ValueError("lp_over_a must be positive")
    cc = _nonneg(c, "c"); ff = _fin(f, "f")
    nn = _int(n, "n", 8)
    if nn % 4:
        raise ValueError("n must be divisible by four")
    ns = _int(Ns, "Ns", 2)
    k = _int(i0, "i0", 0)
    if k > ns - 1:
        raise ValueError("i0 must be at most Ns-1")
    out = _corr(lp, cc, ff, ns, nn, k)
    if not np.all(np.isfinite(out)) or np.any(np.abs(out) > 1.0 + 1e-9):
        raise ValueError("tangent correlation out of range")
    return np.asarray(out, dtype=np.float64)

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

def dwlc_nematic_profile(lp_over_a: float, c: float, f: float, Ns: int, n: int) -> "np.ndarray":
    lp = _fin(lp_over_a, "lp_over_a")
    if lp <= 0.0:
        raise ValueError("lp_over_a must be positive")
    cc = _nonneg(c, "c"); ff = _fin(f, "f")
    nn = _int(n, "n", 8)
    if nn % 4:
        raise ValueError("n must be divisible by four")
    ns = _int(Ns, "Ns", 2)
    m = _marginals(lp, cc, ff, ns, nn)
    _, th = _grid(nn)
    out = m @ np.cos(2.0 * th)
    if not np.all(np.isfinite(out)) or np.any(np.abs(out) > 1.0 + 1e-9):
        raise ValueError("nematic order parameter out of range")
    return np.asarray(out, dtype=np.float64)

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

def dwlc_audit(lp_over_a: float, c: float, f: float, Ns: int, n: int) -> "np.ndarray":
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

    B = dwlc_bond_energy(lp, cc, nn)
    m = dwlc_marginals(lp, cc, ff, ns, nn)
    K = dwlc_closure_exponent(lp, cc, ff, ns, nn, im)
    ap = dwlc_apparent_persistence(lp, cc, ff, ns, nn, im)
    ex = dwlc_extension(lp, cc, ff, ns, nn)
    nem = dwlc_nematic_profile(lp, cc, ff, ns, nn)
    r2 = 0.0
    for i in range(ns):
        r2 += float(np.sum(dwlc_tangent_correlation(lp, cc, ff, ns, nn, i)))
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
                mm = dwlc_marginals(lv, cv, fv, ns, nn)
                kk = dwlc_closure_exponent(lv, cv, fv, ns, nn, im)
                aa = dwlc_apparent_persistence(lv, cv, fv, ns, nn, im)
                xx = dwlc_extension(lv, cv, fv, ns, nn)[ns]
                qq = dwlc_nematic_profile(lv, cv, fv, ns, nn)[im]
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
SCICODE_GOLD_EOF
