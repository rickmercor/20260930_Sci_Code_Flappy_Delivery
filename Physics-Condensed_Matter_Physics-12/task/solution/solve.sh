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


def _bench():
    """Benchmark configuration: beta, U, dmu, hyb, Nf, Nb."""
    return 1.0, 5.0, 1.0, 0.5, 8, 7


def _nf_int(Nf):
    """Fermionic integer labels n, so that nu = (2n+1) pi / beta."""
    return np.arange(-(Nf//2), Nf - Nf//2)


def _nb_int(Nb):
    """Bosonic integer labels m, so that omega = 2 m pi / beta."""
    return np.arange(-(Nb//2), Nb - Nb//2)


def _fidx(Nf, n):
    """Fermionic integer label -> array position, -1 when outside the box."""
    lo = -(Nf//2)
    pos = n - lo
    return np.where((pos >= 0) & (pos < Nf), pos, -1)


def _bidx(Nb, m):
    lo = -(Nb//2)
    pos = m - lo
    return np.where((pos >= 0) & (pos < Nb), pos, -1)


def _gather(A, i, j, k):
    """A[i, j, k] with any out-of-box index giving zero."""
    ok = (i >= 0) & (j >= 0) & (k >= 0)
    out = np.zeros(np.broadcast(i, j, k).shape, dtype=complex)
    if ok.any():
        out[ok] = A[i[ok], j[ok], k[ok]]
    return out


def _shift_indices(Nf, Nb):
    """The three index maps the parquet decomposition needs, precomputed once."""
    n = _nf_int(Nf)[:, None, None]
    npr = _nf_int(Nf)[None, :, None]
    m = _nb_int(Nb)[None, None, :]
    shape = (Nf, Nf, Nb)
    bc = lambda x: np.broadcast_to(x, shape)
    A = (bc(_fidx(Nf, n)), bc(_fidx(Nf, n + m)), bc(_bidx(Nb, npr - n)))
    B = (bc(_fidx(Nf, n)), bc(_fidx(Nf, npr)), bc(_bidx(Nb, -m - n - npr - 1)))
    C = (bc(_fidx(Nf, n)), bc(_fidx(Nf, -npr - m - 1)), bc(_bidx(Nb, npr - n)))
    return A, B, C


def _unpack(state, Nf, Nb):
    L = Nf*Nf*Nb
    Phi = np.stack([state[i*L:(i+1)*L].reshape(Nf, Nf, Nb) for i in range(4)])
    return Phi, state[4*L:].copy()


def _pack(Phi, G):
    return np.concatenate([Phi[0].ravel(), Phi[1].ravel(), Phi[2].ravel(), Phi[3].ravel(), G])




def matsubara_grids(beta: float, Nf: int, Nb: int) -> 'np.ndarray':
    if not np.isfinite(beta) or beta <= 0.0:
        raise ValueError("the inverse temperature must be positive and finite")
    if Nf < 2 or Nb < 1:
        raise ValueError("the boxes must hold at least two fermionic and one bosonic frequency")
    nu = (2.0*_nf_int(Nf) + 1.0)*np.pi/beta
    om = 2.0*_nb_int(Nb)*np.pi/beta
    return np.concatenate([nu, om])

import numpy as np


def _bench():
    """Benchmark configuration: beta, U, dmu, hyb, Nf, Nb."""
    return 1.0, 5.0, 1.0, 0.5, 8, 7


def _nf_int(Nf):
    """Fermionic integer labels n, so that nu = (2n+1) pi / beta."""
    return np.arange(-(Nf//2), Nf - Nf//2)


def _nb_int(Nb):
    """Bosonic integer labels m, so that omega = 2 m pi / beta."""
    return np.arange(-(Nb//2), Nb - Nb//2)


def _fidx(Nf, n):
    """Fermionic integer label -> array position, -1 when outside the box."""
    lo = -(Nf//2)
    pos = n - lo
    return np.where((pos >= 0) & (pos < Nf), pos, -1)


def _bidx(Nb, m):
    lo = -(Nb//2)
    pos = m - lo
    return np.where((pos >= 0) & (pos < Nb), pos, -1)


def _gather(A, i, j, k):
    """A[i, j, k] with any out-of-box index giving zero."""
    ok = (i >= 0) & (j >= 0) & (k >= 0)
    out = np.zeros(np.broadcast(i, j, k).shape, dtype=complex)
    if ok.any():
        out[ok] = A[i[ok], j[ok], k[ok]]
    return out


def _shift_indices(Nf, Nb):
    """The three index maps the parquet decomposition needs, precomputed once."""
    n = _nf_int(Nf)[:, None, None]
    npr = _nf_int(Nf)[None, :, None]
    m = _nb_int(Nb)[None, None, :]
    shape = (Nf, Nf, Nb)
    bc = lambda x: np.broadcast_to(x, shape)
    A = (bc(_fidx(Nf, n)), bc(_fidx(Nf, n + m)), bc(_bidx(Nb, npr - n)))
    B = (bc(_fidx(Nf, n)), bc(_fidx(Nf, npr)), bc(_bidx(Nb, -m - n - npr - 1)))
    C = (bc(_fidx(Nf, n)), bc(_fidx(Nf, -npr - m - 1)), bc(_bidx(Nb, npr - n)))
    return A, B, C


def _unpack(state, Nf, Nb):
    L = Nf*Nf*Nb
    Phi = np.stack([state[i*L:(i+1)*L].reshape(Nf, Nf, Nb) for i in range(4)])
    return Phi, state[4*L:].copy()


def _pack(Phi, G):
    return np.concatenate([Phi[0].ravel(), Phi[1].ravel(), Phi[2].ravel(), Phi[3].ravel(), G])




def bare_propagator(beta: float, Nf: int, U: float, dmu: float, hyb: float) -> 'np.ndarray':
    if not np.isfinite([beta, U, dmu, hyb]).all():
        raise ValueError("the model parameters must be finite")
    if beta <= 0.0:
        raise ValueError("the inverse temperature must be positive")
    if hyb < 0.0:
        raise ValueError("the hybridisation strength must not be negative")
    nu = matsubara_grids(beta, Nf, 1)[:Nf]
    return 1.0/(1j*nu + dmu + 0.5*U + 1j*hyb*np.sign(nu))

import numpy as np


def _bench():
    """Benchmark configuration: beta, U, dmu, hyb, Nf, Nb."""
    return 1.0, 5.0, 1.0, 0.5, 8, 7


def _nf_int(Nf):
    """Fermionic integer labels n, so that nu = (2n+1) pi / beta."""
    return np.arange(-(Nf//2), Nf - Nf//2)


def _nb_int(Nb):
    """Bosonic integer labels m, so that omega = 2 m pi / beta."""
    return np.arange(-(Nb//2), Nb - Nb//2)


def _fidx(Nf, n):
    """Fermionic integer label -> array position, -1 when outside the box."""
    lo = -(Nf//2)
    pos = n - lo
    return np.where((pos >= 0) & (pos < Nf), pos, -1)


def _bidx(Nb, m):
    lo = -(Nb//2)
    pos = m - lo
    return np.where((pos >= 0) & (pos < Nb), pos, -1)


def _gather(A, i, j, k):
    """A[i, j, k] with any out-of-box index giving zero."""
    ok = (i >= 0) & (j >= 0) & (k >= 0)
    out = np.zeros(np.broadcast(i, j, k).shape, dtype=complex)
    if ok.any():
        out[ok] = A[i[ok], j[ok], k[ok]]
    return out


def _shift_indices(Nf, Nb):
    """The three index maps the parquet decomposition needs, precomputed once."""
    n = _nf_int(Nf)[:, None, None]
    npr = _nf_int(Nf)[None, :, None]
    m = _nb_int(Nb)[None, None, :]
    shape = (Nf, Nf, Nb)
    bc = lambda x: np.broadcast_to(x, shape)
    A = (bc(_fidx(Nf, n)), bc(_fidx(Nf, n + m)), bc(_bidx(Nb, npr - n)))
    B = (bc(_fidx(Nf, n)), bc(_fidx(Nf, npr)), bc(_bidx(Nb, -m - n - npr - 1)))
    C = (bc(_fidx(Nf, n)), bc(_fidx(Nf, -npr - m - 1)), bc(_bidx(Nb, npr - n)))
    return A, B, C


def _unpack(state, Nf, Nb):
    L = Nf*Nf*Nb
    Phi = np.stack([state[i*L:(i+1)*L].reshape(Nf, Nf, Nb) for i in range(4)])
    return Phi, state[4*L:].copy()


def _pack(Phi, G):
    return np.concatenate([Phi[0].ravel(), Phi[1].ravel(), Phi[2].ravel(), Phi[3].ravel(), G])




def bare_lambda(U: float) -> 'np.ndarray':
    if not np.isfinite(U):
        raise ValueError("the interaction must be finite")
    return np.array([U, -U, 2.0*U, 0.0], dtype=complex)

import numpy as np


def _bench():
    """Benchmark configuration: beta, U, dmu, hyb, Nf, Nb."""
    return 1.0, 5.0, 1.0, 0.5, 8, 7


def _nf_int(Nf):
    """Fermionic integer labels n, so that nu = (2n+1) pi / beta."""
    return np.arange(-(Nf//2), Nf - Nf//2)


def _nb_int(Nb):
    """Bosonic integer labels m, so that omega = 2 m pi / beta."""
    return np.arange(-(Nb//2), Nb - Nb//2)


def _fidx(Nf, n):
    """Fermionic integer label -> array position, -1 when outside the box."""
    lo = -(Nf//2)
    pos = n - lo
    return np.where((pos >= 0) & (pos < Nf), pos, -1)


def _bidx(Nb, m):
    lo = -(Nb//2)
    pos = m - lo
    return np.where((pos >= 0) & (pos < Nb), pos, -1)


def _gather(A, i, j, k):
    """A[i, j, k] with any out-of-box index giving zero."""
    ok = (i >= 0) & (j >= 0) & (k >= 0)
    out = np.zeros(np.broadcast(i, j, k).shape, dtype=complex)
    if ok.any():
        out[ok] = A[i[ok], j[ok], k[ok]]
    return out


def _shift_indices(Nf, Nb):
    """The three index maps the parquet decomposition needs, precomputed once."""
    n = _nf_int(Nf)[:, None, None]
    npr = _nf_int(Nf)[None, :, None]
    m = _nb_int(Nb)[None, None, :]
    shape = (Nf, Nf, Nb)
    bc = lambda x: np.broadcast_to(x, shape)
    A = (bc(_fidx(Nf, n)), bc(_fidx(Nf, n + m)), bc(_bidx(Nb, npr - n)))
    B = (bc(_fidx(Nf, n)), bc(_fidx(Nf, npr)), bc(_bidx(Nb, -m - n - npr - 1)))
    C = (bc(_fidx(Nf, n)), bc(_fidx(Nf, -npr - m - 1)), bc(_bidx(Nb, npr - n)))
    return A, B, C


def _unpack(state, Nf, Nb):
    L = Nf*Nf*Nb
    Phi = np.stack([state[i*L:(i+1)*L].reshape(Nf, Nf, Nb) for i in range(4)])
    return Phi, state[4*L:].copy()


def _pack(Phi, G):
    return np.concatenate([Phi[0].ravel(), Phi[1].ravel(), Phi[2].ravel(), Phi[3].ravel(), G])




def irreducible_vertices(Phi: 'np.ndarray', Lam: 'np.ndarray') -> 'np.ndarray':
    Phi = np.asarray(Phi, dtype=complex)
    if Phi.ndim != 4 or Phi.shape[0] != 4:
        raise ValueError("Phi must have shape (4, Nf, Nf, Nb) ordered d, m, s, t")
    Lam = np.asarray(Lam, dtype=complex)
    if Lam.shape != (4,):
        raise ValueError("Lambda must hold the four channel constants d, m, s, t")
    Nf, Nb = Phi.shape[1], Phi.shape[3]
    if Phi.shape[2] != Nf:
        raise ValueError("the two fermionic axes of Phi must have equal length")
    A, B, C = _shift_indices(Nf, Nb)
    dA, mA = _gather(Phi[0], *A), _gather(Phi[1], *A)
    sB, tB = _gather(Phi[2], *B), _gather(Phi[3], *B)
    dB, mB = _gather(Phi[0], *B), _gather(Phi[1], *B)
    dC, mC = _gather(Phi[0], *C), _gather(Phi[1], *C)
    Gam = np.empty_like(Phi)
    Gam[0] = Lam[0] - 0.5*dA - 1.5*mA + 0.5*sB + 1.5*tB
    Gam[1] = Lam[1] - 0.5*dA + 0.5*mA - 0.5*sB + 0.5*tB
    Gam[2] = Lam[2] + 0.5*dB - 1.5*mB + 0.5*dC - 1.5*mC
    Gam[3] = Lam[3] + 0.5*dB + 0.5*mB - 0.5*dC - 0.5*mC
    return Gam

import numpy as np


def _bench():
    """Benchmark configuration: beta, U, dmu, hyb, Nf, Nb."""
    return 1.0, 5.0, 1.0, 0.5, 8, 7


def _nf_int(Nf):
    """Fermionic integer labels n, so that nu = (2n+1) pi / beta."""
    return np.arange(-(Nf//2), Nf - Nf//2)


def _nb_int(Nb):
    """Bosonic integer labels m, so that omega = 2 m pi / beta."""
    return np.arange(-(Nb//2), Nb - Nb//2)


def _fidx(Nf, n):
    """Fermionic integer label -> array position, -1 when outside the box."""
    lo = -(Nf//2)
    pos = n - lo
    return np.where((pos >= 0) & (pos < Nf), pos, -1)


def _bidx(Nb, m):
    lo = -(Nb//2)
    pos = m - lo
    return np.where((pos >= 0) & (pos < Nb), pos, -1)


def _gather(A, i, j, k):
    """A[i, j, k] with any out-of-box index giving zero."""
    ok = (i >= 0) & (j >= 0) & (k >= 0)
    out = np.zeros(np.broadcast(i, j, k).shape, dtype=complex)
    if ok.any():
        out[ok] = A[i[ok], j[ok], k[ok]]
    return out


def _shift_indices(Nf, Nb):
    """The three index maps the parquet decomposition needs, precomputed once."""
    n = _nf_int(Nf)[:, None, None]
    npr = _nf_int(Nf)[None, :, None]
    m = _nb_int(Nb)[None, None, :]
    shape = (Nf, Nf, Nb)
    bc = lambda x: np.broadcast_to(x, shape)
    A = (bc(_fidx(Nf, n)), bc(_fidx(Nf, n + m)), bc(_bidx(Nb, npr - n)))
    B = (bc(_fidx(Nf, n)), bc(_fidx(Nf, npr)), bc(_bidx(Nb, -m - n - npr - 1)))
    C = (bc(_fidx(Nf, n)), bc(_fidx(Nf, -npr - m - 1)), bc(_bidx(Nb, npr - n)))
    return A, B, C


def _unpack(state, Nf, Nb):
    L = Nf*Nf*Nb
    Phi = np.stack([state[i*L:(i+1)*L].reshape(Nf, Nf, Nb) for i in range(4)])
    return Phi, state[4*L:].copy()


def _pack(Phi, G):
    return np.concatenate([Phi[0].ravel(), Phi[1].ravel(), Phi[2].ravel(), Phi[3].ravel(), G])




def pair_bubbles(G: 'np.ndarray', Nb: int) -> 'np.ndarray':
    G = np.asarray(G, dtype=complex)
    if G.ndim != 1:
        raise ValueError("the Green's function must be a one-dimensional array over nu")
    if Nb < 1:
        raise ValueError("the bosonic box must hold at least one frequency")
    Nf = G.size
    nf, nb = _nf_int(Nf), _nb_int(Nb)
    out = np.zeros((2, Nf, Nb), dtype=complex)
    rows = np.arange(Nf)
    for k, m in enumerate(nb):
        jph = _fidx(Nf, nf + m)
        ok = jph >= 0
        out[0, ok, k] = G[rows[ok]]*G[jph[ok]]
        jpp = _fidx(Nf, -nf - m - 1)
        ok = jpp >= 0
        out[1, ok, k] = G[rows[ok]]*G[jpp[ok]]
    return out

import numpy as np


def _bench():
    """Benchmark configuration: beta, U, dmu, hyb, Nf, Nb."""
    return 1.0, 5.0, 1.0, 0.5, 8, 7


def _nf_int(Nf):
    """Fermionic integer labels n, so that nu = (2n+1) pi / beta."""
    return np.arange(-(Nf//2), Nf - Nf//2)


def _nb_int(Nb):
    """Bosonic integer labels m, so that omega = 2 m pi / beta."""
    return np.arange(-(Nb//2), Nb - Nb//2)


def _fidx(Nf, n):
    """Fermionic integer label -> array position, -1 when outside the box."""
    lo = -(Nf//2)
    pos = n - lo
    return np.where((pos >= 0) & (pos < Nf), pos, -1)


def _bidx(Nb, m):
    lo = -(Nb//2)
    pos = m - lo
    return np.where((pos >= 0) & (pos < Nb), pos, -1)


def _gather(A, i, j, k):
    """A[i, j, k] with any out-of-box index giving zero."""
    ok = (i >= 0) & (j >= 0) & (k >= 0)
    out = np.zeros(np.broadcast(i, j, k).shape, dtype=complex)
    if ok.any():
        out[ok] = A[i[ok], j[ok], k[ok]]
    return out


def _shift_indices(Nf, Nb):
    """The three index maps the parquet decomposition needs, precomputed once."""
    n = _nf_int(Nf)[:, None, None]
    npr = _nf_int(Nf)[None, :, None]
    m = _nb_int(Nb)[None, None, :]
    shape = (Nf, Nf, Nb)
    bc = lambda x: np.broadcast_to(x, shape)
    A = (bc(_fidx(Nf, n)), bc(_fidx(Nf, n + m)), bc(_bidx(Nb, npr - n)))
    B = (bc(_fidx(Nf, n)), bc(_fidx(Nf, npr)), bc(_bidx(Nb, -m - n - npr - 1)))
    C = (bc(_fidx(Nf, n)), bc(_fidx(Nf, -npr - m - 1)), bc(_bidx(Nb, npr - n)))
    return A, B, C


def _unpack(state, Nf, Nb):
    L = Nf*Nf*Nb
    Phi = np.stack([state[i*L:(i+1)*L].reshape(Nf, Nf, Nb) for i in range(4)])
    return Phi, state[4*L:].copy()


def _pack(Phi, G):
    return np.concatenate([Phi[0].ravel(), Phi[1].ravel(), Phi[2].ravel(), Phi[3].ravel(), G])




def self_energy(Phi: 'np.ndarray', Gam: 'np.ndarray', G: 'np.ndarray',
                        beta: float, U: float) -> 'np.ndarray':
    Phi = np.asarray(Phi, dtype=complex); Gam = np.asarray(Gam, dtype=complex)
    G = np.asarray(G, dtype=complex)
    if Phi.shape != Gam.shape:
        raise ValueError("Phi and Gamma must have the same shape")
    if beta <= 0.0:
        raise ValueError("the inverse temperature must be positive")
    Nf, Nb = G.size, Phi.shape[3]
    nu = (2.0*_nf_int(Nf) + 1.0)*np.pi/beta
    # (1/beta) sum_nu G(nu) e^{i nu 0^+} = n/2. G falls off as 1/(i nu); that tail is subtracted
    # frequency by frequency and its exact sum, one half, is added back. On a box symmetric about
    # zero the subtracted tail itself sums to zero, so it is the added half that carries the
    # contact contribution -- dropping it leaves only the regular part, 0.0704 at the benchmark, and
    # an eightfold-too-small static self-energy (0.351877 in place of 2.851877).
    dens = np.sum(G - 1.0/(1j*nu))/beta + 0.5
    bub = pair_bubbles(G, Nb)[0]
    nf, nb = _nf_int(Nf), _nb_int(Nb)
    acc = np.zeros(Nf, dtype=complex)
    rows = np.arange(Nf)
    for k, m in enumerate(nb):
        j = _fidx(Nf, nf + m)
        ok = j >= 0
        gshift = np.zeros(Nf, dtype=complex)
        gshift[ok] = G[j[ok]]
        F = Phi[0][:, :, k] + Gam[0][:, :, k] - Phi[1][:, :, k] - Gam[1][:, :, k]
        acc += (F @ bub[:, k])*gshift
    return U*dens*np.ones(Nf, dtype=complex) - (U/(2.0*beta**2))*acc

import numpy as np


def _bench():
    """Benchmark configuration: beta, U, dmu, hyb, Nf, Nb."""
    return 1.0, 5.0, 1.0, 0.5, 8, 7


def _nf_int(Nf):
    """Fermionic integer labels n, so that nu = (2n+1) pi / beta."""
    return np.arange(-(Nf//2), Nf - Nf//2)


def _nb_int(Nb):
    """Bosonic integer labels m, so that omega = 2 m pi / beta."""
    return np.arange(-(Nb//2), Nb - Nb//2)


def _fidx(Nf, n):
    """Fermionic integer label -> array position, -1 when outside the box."""
    lo = -(Nf//2)
    pos = n - lo
    return np.where((pos >= 0) & (pos < Nf), pos, -1)


def _bidx(Nb, m):
    lo = -(Nb//2)
    pos = m - lo
    return np.where((pos >= 0) & (pos < Nb), pos, -1)


def _gather(A, i, j, k):
    """A[i, j, k] with any out-of-box index giving zero."""
    ok = (i >= 0) & (j >= 0) & (k >= 0)
    out = np.zeros(np.broadcast(i, j, k).shape, dtype=complex)
    if ok.any():
        out[ok] = A[i[ok], j[ok], k[ok]]
    return out


def _shift_indices(Nf, Nb):
    """The three index maps the parquet decomposition needs, precomputed once."""
    n = _nf_int(Nf)[:, None, None]
    npr = _nf_int(Nf)[None, :, None]
    m = _nb_int(Nb)[None, None, :]
    shape = (Nf, Nf, Nb)
    bc = lambda x: np.broadcast_to(x, shape)
    A = (bc(_fidx(Nf, n)), bc(_fidx(Nf, n + m)), bc(_bidx(Nb, npr - n)))
    B = (bc(_fidx(Nf, n)), bc(_fidx(Nf, npr)), bc(_bidx(Nb, -m - n - npr - 1)))
    C = (bc(_fidx(Nf, n)), bc(_fidx(Nf, -npr - m - 1)), bc(_bidx(Nb, npr - n)))
    return A, B, C


def _unpack(state, Nf, Nb):
    L = Nf*Nf*Nb
    Phi = np.stack([state[i*L:(i+1)*L].reshape(Nf, Nf, Nb) for i in range(4)])
    return Phi, state[4*L:].copy()


def _pack(Phi, G):
    return np.concatenate([Phi[0].ravel(), Phi[1].ravel(), Phi[2].ravel(), Phi[3].ravel(), G])




def parquet_map(state: 'np.ndarray', Lam: 'np.ndarray', beta: float, U: float,
                        dmu: float, hyb: float, Nf: int, Nb: int) -> 'np.ndarray':
    state = np.asarray(state, dtype=complex)
    if state.size != 4*Nf*Nf*Nb + Nf:
        raise ValueError("the state must hold four vertex blocks and one Green's function")
    Phi, G = _unpack(state, Nf, Nb)
    Gam = irreducible_vertices(Phi, Lam)
    bub = pair_bubbles(G, Nb)
    new = np.empty_like(Phi)
    for k in range(Nb):
        new[0][:, :, k] = (Gam[0][:, :, k]*bub[0][None, :, k]) @ (Phi[0][:, :, k] + Gam[0][:, :, k])
        new[1][:, :, k] = (Gam[1][:, :, k]*bub[0][None, :, k]) @ (Phi[1][:, :, k] + Gam[1][:, :, k])
        new[2][:, :, k] = -0.5*((Phi[2][:, :, k] + Gam[2][:, :, k])*bub[1][None, :, k]) @ Gam[2][:, :, k]
        new[3][:, :, k] = +0.5*((Phi[3][:, :, k] + Gam[3][:, :, k])*bub[1][None, :, k]) @ Gam[3][:, :, k]
    new = new/beta
    Sig = self_energy(Phi, Gam, G, beta, U)
    G0 = bare_propagator(beta, Nf, U, dmu, hyb)
    return _pack(new, 1.0/(1.0/G0 - Sig))

import numpy as np


def _bench():
    """Benchmark configuration: beta, U, dmu, hyb, Nf, Nb."""
    return 1.0, 5.0, 1.0, 0.5, 8, 7


def _nf_int(Nf):
    """Fermionic integer labels n, so that nu = (2n+1) pi / beta."""
    return np.arange(-(Nf//2), Nf - Nf//2)


def _nb_int(Nb):
    """Bosonic integer labels m, so that omega = 2 m pi / beta."""
    return np.arange(-(Nb//2), Nb - Nb//2)


def _fidx(Nf, n):
    """Fermionic integer label -> array position, -1 when outside the box."""
    lo = -(Nf//2)
    pos = n - lo
    return np.where((pos >= 0) & (pos < Nf), pos, -1)


def _bidx(Nb, m):
    lo = -(Nb//2)
    pos = m - lo
    return np.where((pos >= 0) & (pos < Nb), pos, -1)


def _gather(A, i, j, k):
    """A[i, j, k] with any out-of-box index giving zero."""
    ok = (i >= 0) & (j >= 0) & (k >= 0)
    out = np.zeros(np.broadcast(i, j, k).shape, dtype=complex)
    if ok.any():
        out[ok] = A[i[ok], j[ok], k[ok]]
    return out


def _shift_indices(Nf, Nb):
    """The three index maps the parquet decomposition needs, precomputed once."""
    n = _nf_int(Nf)[:, None, None]
    npr = _nf_int(Nf)[None, :, None]
    m = _nb_int(Nb)[None, None, :]
    shape = (Nf, Nf, Nb)
    bc = lambda x: np.broadcast_to(x, shape)
    A = (bc(_fidx(Nf, n)), bc(_fidx(Nf, n + m)), bc(_bidx(Nb, npr - n)))
    B = (bc(_fidx(Nf, n)), bc(_fidx(Nf, npr)), bc(_bidx(Nb, -m - n - npr - 1)))
    C = (bc(_fidx(Nf, n)), bc(_fidx(Nf, -npr - m - 1)), bc(_bidx(Nb, npr - n)))
    return A, B, C


def _unpack(state, Nf, Nb):
    L = Nf*Nf*Nb
    Phi = np.stack([state[i*L:(i+1)*L].reshape(Nf, Nf, Nb) for i in range(4)])
    return Phi, state[4*L:].copy()


def _pack(Phi, G):
    return np.concatenate([Phi[0].ravel(), Phi[1].ravel(), Phi[2].ravel(), Phi[3].ravel(), G])




def relax_parquet(state0: 'np.ndarray', Lam: 'np.ndarray', beta: float, U: float,
                          dmu: float, hyb: float, Nf: int, Nb: int, p: float) -> 'np.ndarray':
    if not (0.0 < p <= 1.0):
        raise ValueError("the damping parameter must lie in (0, 1]")
    v = np.asarray(state0, dtype=complex).copy()
    for _ in range(20000):
        nv = p*parquet_map(v, Lam, beta, U, dmu, hyb, Nf, Nb) + (1.0 - p)*v
        if not np.all(np.isfinite(nv)):
            raise ValueError("the damped iteration diverged")
        err = np.max(np.abs(nv - v))/max(1.0, np.max(np.abs(nv)))
        v = nv
        if err < 1e-13:
            return v
    raise ValueError("the damped iteration did not converge within the iteration budget")

import numpy as np


def _bench():
    """Benchmark configuration: beta, U, dmu, hyb, Nf, Nb."""
    return 1.0, 5.0, 1.0, 0.5, 8, 7


def _nf_int(Nf):
    """Fermionic integer labels n, so that nu = (2n+1) pi / beta."""
    return np.arange(-(Nf//2), Nf - Nf//2)


def _nb_int(Nb):
    """Bosonic integer labels m, so that omega = 2 m pi / beta."""
    return np.arange(-(Nb//2), Nb - Nb//2)


def _fidx(Nf, n):
    """Fermionic integer label -> array position, -1 when outside the box."""
    lo = -(Nf//2)
    pos = n - lo
    return np.where((pos >= 0) & (pos < Nf), pos, -1)


def _bidx(Nb, m):
    lo = -(Nb//2)
    pos = m - lo
    return np.where((pos >= 0) & (pos < Nb), pos, -1)


def _gather(A, i, j, k):
    """A[i, j, k] with any out-of-box index giving zero."""
    ok = (i >= 0) & (j >= 0) & (k >= 0)
    out = np.zeros(np.broadcast(i, j, k).shape, dtype=complex)
    if ok.any():
        out[ok] = A[i[ok], j[ok], k[ok]]
    return out


def _shift_indices(Nf, Nb):
    """The three index maps the parquet decomposition needs, precomputed once."""
    n = _nf_int(Nf)[:, None, None]
    npr = _nf_int(Nf)[None, :, None]
    m = _nb_int(Nb)[None, None, :]
    shape = (Nf, Nf, Nb)
    bc = lambda x: np.broadcast_to(x, shape)
    A = (bc(_fidx(Nf, n)), bc(_fidx(Nf, n + m)), bc(_bidx(Nb, npr - n)))
    B = (bc(_fidx(Nf, n)), bc(_fidx(Nf, npr)), bc(_bidx(Nb, -m - n - npr - 1)))
    C = (bc(_fidx(Nf, n)), bc(_fidx(Nf, -npr - m - 1)), bc(_bidx(Nb, npr - n)))
    return A, B, C


def _unpack(state, Nf, Nb):
    L = Nf*Nf*Nb
    Phi = np.stack([state[i*L:(i+1)*L].reshape(Nf, Nf, Nb) for i in range(4)])
    return Phi, state[4*L:].copy()


def _pack(Phi, G):
    return np.concatenate([Phi[0].ravel(), Phi[1].ravel(), Phi[2].ravel(), Phi[3].ravel(), G])




def gamma_from_full_vertex(F: 'np.ndarray', G: 'np.ndarray', beta: float, Nb: int) -> 'np.ndarray':
    F = np.asarray(F, dtype=complex); G = np.asarray(G, dtype=complex)
    if F.ndim != 4 or F.shape[0] != 4:
        raise ValueError("the full vertex must carry the four channels first")
    if beta <= 0.0:
        raise ValueError("the inverse temperature must be positive")
    Nf = G.size
    if F.shape[1:] != (Nf, Nf, Nb):
        raise ValueError("the full vertex and the propagator disagree about the box")
    bub = pair_bubbles(G, Nb)
    I = np.eye(Nf, dtype=complex)
    # batched over the bosonic index: (Nb, Nf, Nf) matrices in the two fermionic legs
    Dph = bub[0].T[:, :, None]*I            # diag(chi_ph) per bosonic frequency
    Dpp = bub[1].T[:, :, None]*I
    Gam = np.empty_like(F)
    for r in (0, 1):                        # Phi_r = (1/beta) Gamma_r chi_ph F_r
        Fr = np.moveaxis(F[r], 2, 0)
        Gam[r] = np.moveaxis(Fr @ np.linalg.inv(I + (Dph @ Fr)/beta), 0, 2)
    Fs = np.moveaxis(F[2], 2, 0)            # Phi_s = -(1/2beta) F_s chi_pp Gamma_s
    Gam[2] = np.moveaxis(np.linalg.solve(I - (Fs @ Dpp)/(2.0*beta), Fs), 0, 2)
    Ft = np.moveaxis(F[3], 2, 0)            # Phi_t = +(1/2beta) F_t chi_pp Gamma_t
    Gam[3] = np.moveaxis(np.linalg.solve(I + (Ft @ Dpp)/(2.0*beta), Ft), 0, 2)
    return Gam

import numpy as np


def _bench():
    """Benchmark configuration: beta, U, dmu, hyb, Nf, Nb."""
    return 1.0, 5.0, 1.0, 0.5, 8, 7


def _nf_int(Nf):
    """Fermionic integer labels n, so that nu = (2n+1) pi / beta."""
    return np.arange(-(Nf//2), Nf - Nf//2)


def _nb_int(Nb):
    """Bosonic integer labels m, so that omega = 2 m pi / beta."""
    return np.arange(-(Nb//2), Nb - Nb//2)


def _fidx(Nf, n):
    """Fermionic integer label -> array position, -1 when outside the box."""
    lo = -(Nf//2)
    pos = n - lo
    return np.where((pos >= 0) & (pos < Nf), pos, -1)


def _bidx(Nb, m):
    lo = -(Nb//2)
    pos = m - lo
    return np.where((pos >= 0) & (pos < Nb), pos, -1)


def _gather(A, i, j, k):
    """A[i, j, k] with any out-of-box index giving zero."""
    ok = (i >= 0) & (j >= 0) & (k >= 0)
    out = np.zeros(np.broadcast(i, j, k).shape, dtype=complex)
    if ok.any():
        out[ok] = A[i[ok], j[ok], k[ok]]
    return out


def _shift_indices(Nf, Nb):
    """The three index maps the parquet decomposition needs, precomputed once."""
    n = _nf_int(Nf)[:, None, None]
    npr = _nf_int(Nf)[None, :, None]
    m = _nb_int(Nb)[None, None, :]
    shape = (Nf, Nf, Nb)
    bc = lambda x: np.broadcast_to(x, shape)
    A = (bc(_fidx(Nf, n)), bc(_fidx(Nf, n + m)), bc(_bidx(Nb, npr - n)))
    B = (bc(_fidx(Nf, n)), bc(_fidx(Nf, npr)), bc(_bidx(Nb, -m - n - npr - 1)))
    C = (bc(_fidx(Nf, n)), bc(_fidx(Nf, -npr - m - 1)), bc(_bidx(Nb, npr - n)))
    return A, B, C


def _unpack(state, Nf, Nb):
    L = Nf*Nf*Nb
    Phi = np.stack([state[i*L:(i+1)*L].reshape(Nf, Nf, Nb) for i in range(4)])
    return Phi, state[4*L:].copy()


def _pack(Phi, G):
    return np.concatenate([Phi[0].ravel(), Phi[1].ravel(), Phi[2].ravel(), Phi[3].ravel(), G])




def strong_coupling_map(state: 'np.ndarray', Lam: 'np.ndarray', beta: float, U: float,
                                dmu: float, hyb: float, Nf: int, Nb: int) -> 'np.ndarray':
    state = np.asarray(state, dtype=complex)
    if state.size != 4*Nf*Nf*Nb + Nf:
        raise ValueError("the state must hold four vertex blocks and one Green's function")
    F, G = _unpack(state, Nf, Nb)
    Gam = gamma_from_full_vertex(F, G, beta, Nb)
    Phi = F - Gam
    Gtil = irreducible_vertices(Phi, Lam)
    # The equation of motion is a functional of whatever this sweep iterates, which here is the FULL
    # vertex. Phi + Gam is the state's own F; Phi + Gtil is not, away from the fixed point. The two
    # coincide at the fixed point, so this choice is invisible to a residual check and visible only
    # in the Jacobian -- which is precisely the quantity being asked for.
    Sig = self_energy(Phi, Gam, G, beta, U)
    G0 = bare_propagator(beta, Nf, U, dmu, hyb)
    return _pack(Gtil + Phi, 1.0/(1.0/G0 - Sig))

import numpy as np


def _bench():
    """Benchmark configuration: beta, U, dmu, hyb, Nf, Nb."""
    return 1.0, 5.0, 1.0, 0.5, 8, 7


def _nf_int(Nf):
    """Fermionic integer labels n, so that nu = (2n+1) pi / beta."""
    return np.arange(-(Nf//2), Nf - Nf//2)


def _nb_int(Nb):
    """Bosonic integer labels m, so that omega = 2 m pi / beta."""
    return np.arange(-(Nb//2), Nb - Nb//2)


def _fidx(Nf, n):
    """Fermionic integer label -> array position, -1 when outside the box."""
    lo = -(Nf//2)
    pos = n - lo
    return np.where((pos >= 0) & (pos < Nf), pos, -1)


def _bidx(Nb, m):
    lo = -(Nb//2)
    pos = m - lo
    return np.where((pos >= 0) & (pos < Nb), pos, -1)


def _gather(A, i, j, k):
    """A[i, j, k] with any out-of-box index giving zero."""
    ok = (i >= 0) & (j >= 0) & (k >= 0)
    out = np.zeros(np.broadcast(i, j, k).shape, dtype=complex)
    if ok.any():
        out[ok] = A[i[ok], j[ok], k[ok]]
    return out


def _shift_indices(Nf, Nb):
    """The three index maps the parquet decomposition needs, precomputed once."""
    n = _nf_int(Nf)[:, None, None]
    npr = _nf_int(Nf)[None, :, None]
    m = _nb_int(Nb)[None, None, :]
    shape = (Nf, Nf, Nb)
    bc = lambda x: np.broadcast_to(x, shape)
    A = (bc(_fidx(Nf, n)), bc(_fidx(Nf, n + m)), bc(_bidx(Nb, npr - n)))
    B = (bc(_fidx(Nf, n)), bc(_fidx(Nf, npr)), bc(_bidx(Nb, -m - n - npr - 1)))
    C = (bc(_fidx(Nf, n)), bc(_fidx(Nf, -npr - m - 1)), bc(_bidx(Nb, npr - n)))
    return A, B, C


def _unpack(state, Nf, Nb):
    L = Nf*Nf*Nb
    Phi = np.stack([state[i*L:(i+1)*L].reshape(Nf, Nf, Nb) for i in range(4)])
    return Phi, state[4*L:].copy()


def _pack(Phi, G):
    return np.concatenate([Phi[0].ravel(), Phi[1].ravel(), Phi[2].ravel(), Phi[3].ravel(), G])




def stability_spectrum(state: 'np.ndarray', Lam: 'np.ndarray', beta: float, U: float,
                               dmu: float, hyb: float, Nf: int, Nb: int,
                               scheme: str) -> 'np.ndarray':
    state = np.asarray(state, dtype=complex)
    n = state.size
    if n != 4*Nf*Nf*Nb + Nf:
        raise ValueError("the state has the wrong length for this box")
    if scheme == "reducible":
        sweep = parquet_map
    elif scheme == "full":
        sweep = strong_coupling_map
    else:
        raise ValueError("the scheme must be either 'reducible' or 'full'")
    h = 1e-5
    J = np.empty((n, n), dtype=complex)
    for j in range(n):
        e = np.zeros(n, dtype=complex); e[j] = h
        J[:, j] = (sweep(state + e, Lam, beta, U, dmu, hyb, Nf, Nb)
                   - sweep(state - e, Lam, beta, U, dmu, hyb, Nf, Nb))/(2.0*h)
    return np.linalg.eigvals(np.eye(n) - J)

import numpy as np


def _bench():
    """Benchmark configuration: beta, U, dmu, hyb, Nf, Nb."""
    return 1.0, 5.0, 1.0, 0.5, 8, 7


def _nf_int(Nf):
    """Fermionic integer labels n, so that nu = (2n+1) pi / beta."""
    return np.arange(-(Nf//2), Nf - Nf//2)


def _nb_int(Nb):
    """Bosonic integer labels m, so that omega = 2 m pi / beta."""
    return np.arange(-(Nb//2), Nb - Nb//2)


def _fidx(Nf, n):
    """Fermionic integer label -> array position, -1 when outside the box."""
    lo = -(Nf//2)
    pos = n - lo
    return np.where((pos >= 0) & (pos < Nf), pos, -1)


def _bidx(Nb, m):
    lo = -(Nb//2)
    pos = m - lo
    return np.where((pos >= 0) & (pos < Nb), pos, -1)


def _gather(A, i, j, k):
    """A[i, j, k] with any out-of-box index giving zero."""
    ok = (i >= 0) & (j >= 0) & (k >= 0)
    out = np.zeros(np.broadcast(i, j, k).shape, dtype=complex)
    if ok.any():
        out[ok] = A[i[ok], j[ok], k[ok]]
    return out


def _shift_indices(Nf, Nb):
    """The three index maps the parquet decomposition needs, precomputed once."""
    n = _nf_int(Nf)[:, None, None]
    npr = _nf_int(Nf)[None, :, None]
    m = _nb_int(Nb)[None, None, :]
    shape = (Nf, Nf, Nb)
    bc = lambda x: np.broadcast_to(x, shape)
    A = (bc(_fidx(Nf, n)), bc(_fidx(Nf, n + m)), bc(_bidx(Nb, npr - n)))
    B = (bc(_fidx(Nf, n)), bc(_fidx(Nf, npr)), bc(_bidx(Nb, -m - n - npr - 1)))
    C = (bc(_fidx(Nf, n)), bc(_fidx(Nf, -npr - m - 1)), bc(_bidx(Nb, npr - n)))
    return A, B, C


def _unpack(state, Nf, Nb):
    L = Nf*Nf*Nb
    Phi = np.stack([state[i*L:(i+1)*L].reshape(Nf, Nf, Nb) for i in range(4)])
    return Phi, state[4*L:].copy()


def _pack(Phi, G):
    return np.concatenate([Phi[0].ravel(), Phi[1].ravel(), Phi[2].ravel(), Phi[3].ravel(), G])




def minimal_damping_audit(c: float, p_solve: float) -> float:
    if not np.isfinite(c) or not (0.0 < c < 1.0):
        raise ValueError("the safety factor must lie strictly between zero and one")
    if not (0.0 < p_solve <= 1.0):
        raise ValueError("the damping used to reach the fixed point must lie in (0, 1]")
    beta, U, dmu, hyb, Nf, Nb = _bench()

    grids = matsubara_grids(beta, Nf, Nb)
    if grids.size != Nf + Nb:
        raise ValueError("the grid step must return the fermionic then the bosonic frequencies")
    if abs(grids[:Nf][Nf//2] - np.pi/beta) > 1e-12:
        raise ValueError("the fermionic grid is not centred on +- pi/beta")

    G0 = bare_propagator(beta, Nf, U, dmu, hyb)
    Lam = bare_lambda(U)
    if abs(Lam[0] + Lam[1]) > 1e-12:
        raise ValueError("the density and magnetic bare vertices must cancel")

    zero = np.zeros((4, Nf, Nf, Nb), dtype=complex)
    if np.abs(irreducible_vertices(zero, Lam) - Lam[:, None, None, None]).max() > 1e-12:
        raise ValueError("with no reducible vertex, Gamma must reduce to Lambda")
    bub = pair_bubbles(G0, Nb)
    if bub.shape != (2, Nf, Nb):
        raise ValueError("the bubbles must carry the ph channel first and the pp channel second")

    state = relax_parquet(_pack(zero, G0), Lam, beta, U, dmu, hyb, Nf, Nb, p_solve)
    if np.max(np.abs(parquet_map(state, Lam, beta, U, dmu, hyb, Nf, Nb) - state)) > 1e-9:
        raise ValueError("the returned state is not a fixed point of the parquet map")

    # The converged propagator must satisfy the Dyson equation against the self-energy built from
    # its own converged vertices. This reaches the self-energy step directly rather than only
    # through the sweep, so a broken self-energy cannot hide behind a converged fixed point.
    Phi_fin, G_fin = _unpack(state, Nf, Nb)
    Gam_fin = irreducible_vertices(Phi_fin, Lam)
    Sig = self_energy(Phi_fin, Gam_fin, G_fin, beta, U)
    if np.abs((1.0/G0 - 1.0/G_fin) - Sig).max() > 1e-9:
        raise ValueError("the converged propagator does not satisfy the Dyson equation with the "
                         "self-energy of its own vertices")

    lam_r = stability_spectrum(state, Lam, beta, U, dmu, hyb, Nf, Nb, "reducible")

    # The same physical fixed point, re-expressed in the full vertices F_r = Phi_r + Gamma_r. The
    # strong-coupling sweep inverts the Bethe-Salpeter equations instead of summing them, so it is
    # a different map with the same fixed point -- which is exactly what makes it a check on both.
    F_fin = Phi_fin + Gam_fin
    fstate = _pack(F_fin, G_fin)
    if np.max(np.abs(strong_coupling_map(fstate, Lam, beta, U, dmu, hyb, Nf, Nb)
                     - fstate)) > 1e-9:
        raise ValueError("the full-vertex state is not a fixed point of the strong-coupling sweep, "
                         "so the two schemes do not share the physical fixed point")
    if np.abs(gamma_from_full_vertex(F_fin, G_fin, beta, Nb) - Gam_fin).max() > 1e-9:
        raise ValueError("inverting the Bethe-Salpeter equations does not return the irreducible "
                         "vertices the fixed point was built from")
    lam_f = stability_spectrum(fstate, Lam, beta, U, dmu, hyb, Nf, Nb, "full")

    bounds = []
    for lam in (lam_r, lam_f):
        if lam.real.min() <= 0.0:
            raise ValueError("an eigenvalue has non-positive real part; damping cannot stabilise "
                             "this fixed point and the minimal-damping formula does not apply")
        bounds.append(np.min(c*2.0*np.abs(lam.real)/np.abs(lam)**2))
    return float(min(1.0, min(bounds)))
SCICODE_GOLD_EOF
