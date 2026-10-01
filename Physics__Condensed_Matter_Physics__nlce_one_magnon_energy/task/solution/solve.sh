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

import functools

import numpy as np

def _nlce_default_device():
    return dict(J=0.8, n_max=8, k=1.2566370614359172)

def _nlce_real(x, name):
    v = float(x)
    if not np.isfinite(v):
        raise ValueError(name + " must be finite")
    return v

def _nlce_params(dev, keys):
    return [dev[k] for k in keys]

@functools.lru_cache(maxsize=None)
def _nlce_chain(m, J):
    """H = -sum_i sigma^z_i - J sum_i sigma^x_i sigma^x_{i+1} on an open chain of m sites."""
    sx = np.array([[0.0, 1.0], [1.0, 0.0]])
    sz = np.array([[1.0, 0.0], [0.0, -1.0]])

    def _op(site, o):
        out = np.array([[1.0]])
        for q in range(m):
            out = np.kron(out, o if q == site else np.eye(2))
        return out

    H = np.zeros((2 ** m, 2 ** m))
    for i in range(m):
        H -= _op(i, sz)
    for i in range(m - 1):
        H -= J * (_op(i, sx) @ _op(i + 1, sx))
    H.setflags(write=False)
    return H


def _nlce_do_hamiltonian_element(a, b, m, dev):
    """Matrix element H[a, b] of the open transverse-field Ising chain."""
    m = int(_nlce_real(m, "m"))
    a, b = int(_nlce_real(a, "a")), int(_nlce_real(b, "b"))
    if not (0 <= a < 2 ** m and 0 <= b < 2 ** m):
        raise ValueError("basis index out of range")
    return float(_nlce_chain(m, _nlce_real(_nlce_params(dev, ("J",))[0], "J"))[a, b])


def _nlce_flip_rows(m):
    """Product-basis indices of the m single-spin-flip states, site 0 first."""
    return [1 << (m - 1 - i) for i in range(m)]

def hamiltonian_element(a: int, b: int, m: int, dev: dict) -> float:
    """Matrix element H[a, b] of the open transverse-field Ising chain."""
    return _nlce_do_hamiltonian_element(a, b, m, dev)

import functools

import numpy as np

def _nlce_default_device():
    return dict(J=0.8, n_max=8, k=1.2566370614359172)

def _nlce_real(x, name):
    v = float(x)
    if not np.isfinite(v):
        raise ValueError(name + " must be finite")
    return v

def _nlce_params(dev, keys):
    return [dev[k] for k in keys]

@functools.lru_cache(maxsize=None)
def _nlce_chain(m, J):
    """H = -sum_i sigma^z_i - J sum_i sigma^x_i sigma^x_{i+1} on an open chain of m sites."""
    sx = np.array([[0.0, 1.0], [1.0, 0.0]])
    sz = np.array([[1.0, 0.0], [0.0, -1.0]])

    def _op(site, o):
        out = np.array([[1.0]])
        for q in range(m):
            out = np.kron(out, o if q == site else np.eye(2))
        return out

    H = np.zeros((2 ** m, 2 ** m))
    for i in range(m):
        H -= _op(i, sz)
    for i in range(m - 1):
        H -= J * (_op(i, sx) @ _op(i + 1, sx))
    H.setflags(write=False)
    return H


def _nlce_do_hamiltonian_element(a, b, m, dev):
    """Matrix element H[a, b] of the open transverse-field Ising chain."""
    m = int(_nlce_real(m, "m"))
    a, b = int(_nlce_real(a, "a")), int(_nlce_real(b, "b"))
    if not (0 <= a < 2 ** m and 0 <= b < 2 ** m):
        raise ValueError("basis index out of range")
    return float(_nlce_chain(m, _nlce_real(_nlce_params(dev, ("J",))[0], "J"))[a, b])


def _nlce_flip_rows(m):
    """Product-basis indices of the m single-spin-flip states, site 0 first."""
    return [1 << (m - 1 - i) for i in range(m)]

@functools.lru_cache(maxsize=None)
def _nlce_eig(m, J):
    """Spectrum and eigenvectors of the m-site chain, cached read-only by (m, J)."""
    lam, S = np.linalg.eigh(_nlce_chain(m, J))
    lam.setflags(write=False)
    S.setflags(write=False)
    return lam, S


def _nlce_spectrum(m, dev):
    """Spectrum of the m-site chain, whose Hamiltonian must agree with step 01's elements."""
    J = _nlce_real(_nlce_params(dev, ("J",))[0], "J")
    m = int(m)
    if m < 1:
        raise ValueError("chain length must be at least 1")
    _nlce_bind()
    H = _nlce_chain(m, J)
    probe = [(0, 0), (1, 1), (0, 3)] if m >= 2 else [(0, 0), (1, 1)]
    for a, b in probe:                                                      # step 01
        if abs(hamiltonian_element(a, b, m, dev) - H[a, b]) > 1e-12:
            raise ValueError("the chain does not have the matrix elements step 01 reports")
    return _nlce_eig(m, J)


def _nlce_subset(m, dev):
    """The m eigenstates that deform the target block least, Eq. (39)."""
    m = int(m)
    lam, S = _nlce_spectrum(m, dev)
    J = _nlce_real(_nlce_params(dev, ("J",))[0], "J")
    return _nlce_select(m, J), lam, S


@functools.lru_cache(maxsize=None)
def _nlce_select(m, J):
    """Eq. (39) argmin, cached by (m, J). Seed with the m eigenstates of largest weight on the
    single-flip subspace, then exchange one member at a time for any eigenstate with weight there
    while that lowers the restricted deformation. Checked against exhaustive search over the
    highest-weight candidates on every chain and coupling the task uses."""
    _lam, S = _nlce_eig(m, J)
    rows = _nlce_flip_rows(m)
    p = (S[rows, :] ** 2).sum(axis=0)
    order = np.argsort(-p, kind="stable")
    sub = sorted(order[:m].tolist())
    pool = [int(a) for a in order if p[a] > 1e-12]

    def _dev_of(cols):
        sg = np.linalg.svd(S[np.ix_(rows, cols)], compute_uv=False)
        return float(np.sqrt(np.sum((sg - 1.0) ** 2)))

    best = _dev_of(sub)
    improved = True
    while improved:
        improved = False
        for pos in range(m):
            for c in pool:
                if c in sub:
                    continue
                trial = sorted(sub[:pos] + [c] + sub[pos + 1:])
                d = _dev_of(trial)
                if d < best - 1e-12:
                    best, sub, improved = d, trial, True
    return tuple(sub)


def _nlce_svd_block(m, dev):
    m = int(m)
    sub, lam, S = _nlce_subset(m, dev)
    U, sg, Vh = np.linalg.svd(S[np.ix_(_nlce_flip_rows(m), list(sub))])
    return U, sg, Vh, np.diag(lam[list(sub)]), lam


def _nlce_bind():
    """Bind an earlier step's oracle to this file's copy only when the grader has not supplied
    it; setdefault never overrides a real one, so the chain always runs through the oracles
    actually present."""
    g = globals()
    for name in ("hamiltonian_element", "selection_deviation", "effective_element", "cluster_quantity",
                 "cluster_weight", "nlce_hopping"):
        if "_nlce_do_" + name in g:
            g.setdefault("" + name, g["_nlce_do_" + name])


def _nlce_do_selection_deviation(m, dev):
    """||T_11 - 1|| of the selected set; T_11 = U Sigma U^dag, Eq. (13)."""
    m = int(_nlce_real(m, "m"))
    _U, sg, _Vh, _L, _lam = _nlce_svd_block(m, dev)
    return float(np.sqrt(np.sum((sg - 1.0) ** 2)))

def selection_deviation(m: int, dev: dict) -> float:
    """Frobenius deviation of the selected target block from the identity."""
    return _nlce_do_selection_deviation(m, dev)

import functools

import numpy as np

def _nlce_default_device():
    return dict(J=0.8, n_max=8, k=1.2566370614359172)

def _nlce_real(x, name):
    v = float(x)
    if not np.isfinite(v):
        raise ValueError(name + " must be finite")
    return v

def _nlce_params(dev, keys):
    return [dev[k] for k in keys]

@functools.lru_cache(maxsize=None)
def _nlce_chain(m, J):
    """H = -sum_i sigma^z_i - J sum_i sigma^x_i sigma^x_{i+1} on an open chain of m sites."""
    sx = np.array([[0.0, 1.0], [1.0, 0.0]])
    sz = np.array([[1.0, 0.0], [0.0, -1.0]])

    def _op(site, o):
        out = np.array([[1.0]])
        for q in range(m):
            out = np.kron(out, o if q == site else np.eye(2))
        return out

    H = np.zeros((2 ** m, 2 ** m))
    for i in range(m):
        H -= _op(i, sz)
    for i in range(m - 1):
        H -= J * (_op(i, sx) @ _op(i + 1, sx))
    H.setflags(write=False)
    return H


def _nlce_do_hamiltonian_element(a, b, m, dev):
    """Matrix element H[a, b] of the open transverse-field Ising chain."""
    m = int(_nlce_real(m, "m"))
    a, b = int(_nlce_real(a, "a")), int(_nlce_real(b, "b"))
    if not (0 <= a < 2 ** m and 0 <= b < 2 ** m):
        raise ValueError("basis index out of range")
    return float(_nlce_chain(m, _nlce_real(_nlce_params(dev, ("J",))[0], "J"))[a, b])


def _nlce_flip_rows(m):
    """Product-basis indices of the m single-spin-flip states, site 0 first."""
    return [1 << (m - 1 - i) for i in range(m)]

@functools.lru_cache(maxsize=None)
def _nlce_eig(m, J):
    """Spectrum and eigenvectors of the m-site chain, cached read-only by (m, J)."""
    lam, S = np.linalg.eigh(_nlce_chain(m, J))
    lam.setflags(write=False)
    S.setflags(write=False)
    return lam, S


def _nlce_spectrum(m, dev):
    """Spectrum of the m-site chain, whose Hamiltonian must agree with step 01's elements."""
    J = _nlce_real(_nlce_params(dev, ("J",))[0], "J")
    m = int(m)
    if m < 1:
        raise ValueError("chain length must be at least 1")
    _nlce_bind()
    H = _nlce_chain(m, J)
    probe = [(0, 0), (1, 1), (0, 3)] if m >= 2 else [(0, 0), (1, 1)]
    for a, b in probe:                                                      # step 01
        if abs(hamiltonian_element(a, b, m, dev) - H[a, b]) > 1e-12:
            raise ValueError("the chain does not have the matrix elements step 01 reports")
    return _nlce_eig(m, J)


def _nlce_subset(m, dev):
    """The m eigenstates that deform the target block least, Eq. (39)."""
    m = int(m)
    lam, S = _nlce_spectrum(m, dev)
    J = _nlce_real(_nlce_params(dev, ("J",))[0], "J")
    return _nlce_select(m, J), lam, S


@functools.lru_cache(maxsize=None)
def _nlce_select(m, J):
    """Eq. (39) argmin, cached by (m, J). Seed with the m eigenstates of largest weight on the
    single-flip subspace, then exchange one member at a time for any eigenstate with weight there
    while that lowers the restricted deformation. Checked against exhaustive search over the
    highest-weight candidates on every chain and coupling the task uses."""
    _lam, S = _nlce_eig(m, J)
    rows = _nlce_flip_rows(m)
    p = (S[rows, :] ** 2).sum(axis=0)
    order = np.argsort(-p, kind="stable")
    sub = sorted(order[:m].tolist())
    pool = [int(a) for a in order if p[a] > 1e-12]

    def _dev_of(cols):
        sg = np.linalg.svd(S[np.ix_(rows, cols)], compute_uv=False)
        return float(np.sqrt(np.sum((sg - 1.0) ** 2)))

    best = _dev_of(sub)
    improved = True
    while improved:
        improved = False
        for pos in range(m):
            for c in pool:
                if c in sub:
                    continue
                trial = sorted(sub[:pos] + [c] + sub[pos + 1:])
                d = _dev_of(trial)
                if d < best - 1e-12:
                    best, sub, improved = d, trial, True
    return tuple(sub)


def _nlce_svd_block(m, dev):
    m = int(m)
    sub, lam, S = _nlce_subset(m, dev)
    U, sg, Vh = np.linalg.svd(S[np.ix_(_nlce_flip_rows(m), list(sub))])
    return U, sg, Vh, np.diag(lam[list(sub)]), lam


def _nlce_bind():
    """Bind an earlier step's oracle to this file's copy only when the grader has not supplied
    it; setdefault never overrides a real one, so the chain always runs through the oracles
    actually present."""
    g = globals()
    for name in ("hamiltonian_element", "selection_deviation", "effective_element", "cluster_quantity",
                 "cluster_weight", "nlce_hopping"):
        if "_nlce_do_" + name in g:
            g.setdefault("" + name, g["_nlce_do_" + name])


def _nlce_do_selection_deviation(m, dev):
    """||T_11 - 1|| of the selected set; T_11 = U Sigma U^dag, Eq. (13)."""
    m = int(_nlce_real(m, "m"))
    _U, sg, _Vh, _L, _lam = _nlce_svd_block(m, dev)
    return float(np.sqrt(np.sum((sg - 1.0) ** 2)))

def _nlce_do_effective_element(i, j, m, dev):
    """[H_eff]_ij by Theorem 1, Eq. (14), on the set whose deformation step 02 reports."""
    m = int(_nlce_real(m, "m"))
    i, j = int(_nlce_real(i, "i")), int(_nlce_real(j, "j"))
    if not (0 <= i < m and 0 <= j < m):
        raise ValueError("site index out of range")
    U, sg, Vh, Lam, _lam = _nlce_svd_block(m, dev)
    d = float(np.sqrt(np.sum((sg - 1.0) ** 2)))
    if abs(selection_deviation(m, dev) - d) > 1e-9 * (1.0 + d):     # step 02
        raise ValueError("the selected set does not have the deformation step 02 reports")
    P = U @ Vh
    return float((P @ Lam @ P.conj().T)[i, j])

def effective_element(i: int, j: int, m: int, dev: dict) -> float:
    """Element of the one-magnon effective Hamiltonian."""
    _nlce_bind()
    return _nlce_do_effective_element(i, j, m, dev)

import functools

import numpy as np

def _nlce_default_device():
    return dict(J=0.8, n_max=8, k=1.2566370614359172)

def _nlce_real(x, name):
    v = float(x)
    if not np.isfinite(v):
        raise ValueError(name + " must be finite")
    return v

def _nlce_params(dev, keys):
    return [dev[k] for k in keys]

@functools.lru_cache(maxsize=None)
def _nlce_chain(m, J):
    """H = -sum_i sigma^z_i - J sum_i sigma^x_i sigma^x_{i+1} on an open chain of m sites."""
    sx = np.array([[0.0, 1.0], [1.0, 0.0]])
    sz = np.array([[1.0, 0.0], [0.0, -1.0]])

    def _op(site, o):
        out = np.array([[1.0]])
        for q in range(m):
            out = np.kron(out, o if q == site else np.eye(2))
        return out

    H = np.zeros((2 ** m, 2 ** m))
    for i in range(m):
        H -= _op(i, sz)
    for i in range(m - 1):
        H -= J * (_op(i, sx) @ _op(i + 1, sx))
    H.setflags(write=False)
    return H


def _nlce_do_hamiltonian_element(a, b, m, dev):
    """Matrix element H[a, b] of the open transverse-field Ising chain."""
    m = int(_nlce_real(m, "m"))
    a, b = int(_nlce_real(a, "a")), int(_nlce_real(b, "b"))
    if not (0 <= a < 2 ** m and 0 <= b < 2 ** m):
        raise ValueError("basis index out of range")
    return float(_nlce_chain(m, _nlce_real(_nlce_params(dev, ("J",))[0], "J"))[a, b])


def _nlce_flip_rows(m):
    """Product-basis indices of the m single-spin-flip states, site 0 first."""
    return [1 << (m - 1 - i) for i in range(m)]

@functools.lru_cache(maxsize=None)
def _nlce_eig(m, J):
    """Spectrum and eigenvectors of the m-site chain, cached read-only by (m, J)."""
    lam, S = np.linalg.eigh(_nlce_chain(m, J))
    lam.setflags(write=False)
    S.setflags(write=False)
    return lam, S


def _nlce_spectrum(m, dev):
    """Spectrum of the m-site chain, whose Hamiltonian must agree with step 01's elements."""
    J = _nlce_real(_nlce_params(dev, ("J",))[0], "J")
    m = int(m)
    if m < 1:
        raise ValueError("chain length must be at least 1")
    _nlce_bind()
    H = _nlce_chain(m, J)
    probe = [(0, 0), (1, 1), (0, 3)] if m >= 2 else [(0, 0), (1, 1)]
    for a, b in probe:                                                      # step 01
        if abs(hamiltonian_element(a, b, m, dev) - H[a, b]) > 1e-12:
            raise ValueError("the chain does not have the matrix elements step 01 reports")
    return _nlce_eig(m, J)


def _nlce_subset(m, dev):
    """The m eigenstates that deform the target block least, Eq. (39)."""
    m = int(m)
    lam, S = _nlce_spectrum(m, dev)
    J = _nlce_real(_nlce_params(dev, ("J",))[0], "J")
    return _nlce_select(m, J), lam, S


@functools.lru_cache(maxsize=None)
def _nlce_select(m, J):
    """Eq. (39) argmin, cached by (m, J). Seed with the m eigenstates of largest weight on the
    single-flip subspace, then exchange one member at a time for any eigenstate with weight there
    while that lowers the restricted deformation. Checked against exhaustive search over the
    highest-weight candidates on every chain and coupling the task uses."""
    _lam, S = _nlce_eig(m, J)
    rows = _nlce_flip_rows(m)
    p = (S[rows, :] ** 2).sum(axis=0)
    order = np.argsort(-p, kind="stable")
    sub = sorted(order[:m].tolist())
    pool = [int(a) for a in order if p[a] > 1e-12]

    def _dev_of(cols):
        sg = np.linalg.svd(S[np.ix_(rows, cols)], compute_uv=False)
        return float(np.sqrt(np.sum((sg - 1.0) ** 2)))

    best = _dev_of(sub)
    improved = True
    while improved:
        improved = False
        for pos in range(m):
            for c in pool:
                if c in sub:
                    continue
                trial = sorted(sub[:pos] + [c] + sub[pos + 1:])
                d = _dev_of(trial)
                if d < best - 1e-12:
                    best, sub, improved = d, trial, True
    return tuple(sub)


def _nlce_svd_block(m, dev):
    m = int(m)
    sub, lam, S = _nlce_subset(m, dev)
    U, sg, Vh = np.linalg.svd(S[np.ix_(_nlce_flip_rows(m), list(sub))])
    return U, sg, Vh, np.diag(lam[list(sub)]), lam


def _nlce_bind():
    """Bind an earlier step's oracle to this file's copy only when the grader has not supplied
    it; setdefault never overrides a real one, so the chain always runs through the oracles
    actually present."""
    g = globals()
    for name in ("hamiltonian_element", "selection_deviation", "effective_element", "cluster_quantity",
                 "cluster_weight", "nlce_hopping"):
        if "_nlce_do_" + name in g:
            g.setdefault("" + name, g["_nlce_do_" + name])


def _nlce_do_selection_deviation(m, dev):
    """||T_11 - 1|| of the selected set; T_11 = U Sigma U^dag, Eq. (13)."""
    m = int(_nlce_real(m, "m"))
    _U, sg, _Vh, _L, _lam = _nlce_svd_block(m, dev)
    return float(np.sqrt(np.sum((sg - 1.0) ** 2)))

def _nlce_do_effective_element(i, j, m, dev):
    """[H_eff]_ij by Theorem 1, Eq. (14), on the set whose deformation step 02 reports."""
    m = int(_nlce_real(m, "m"))
    i, j = int(_nlce_real(i, "i")), int(_nlce_real(j, "j"))
    if not (0 <= i < m and 0 <= j < m):
        raise ValueError("site index out of range")
    U, sg, Vh, Lam, _lam = _nlce_svd_block(m, dev)
    d = float(np.sqrt(np.sum((sg - 1.0) ** 2)))
    if abs(selection_deviation(m, dev) - d) > 1e-9 * (1.0 + d):     # step 02
        raise ValueError("the selected set does not have the deformation step 02 reports")
    P = U @ Vh
    return float((P @ Lam @ P.conj().T)[i, j])

def _nlce_do_cluster_quantity(m, r, dev):
    """Distance-r pair sum of the step-03 elements; for r = 0 the ground-state energy is
    removed m times so excitation energies stay cluster additive."""
    m, r = int(_nlce_real(m, "m")), int(_nlce_real(r, "r"))
    if r < 0:
        raise ValueError("separation must be non-negative")
    if m <= r:
        return 0.0
    s = sum(effective_element(i, i + r, m, dev) for i in range(m - r))    # step 03
    if r == 0:
        s -= m * float(_nlce_spectrum(m, dev)[0][0])
    return float(s)

def cluster_quantity(m: int, r: int, dev: dict) -> float:
    """Distance-r pair sum of the effective Hamiltonian on the m-site chain."""
    _nlce_bind()
    return _nlce_do_cluster_quantity(m, r, dev)

import functools

import numpy as np

def _nlce_default_device():
    return dict(J=0.8, n_max=8, k=1.2566370614359172)

def _nlce_real(x, name):
    v = float(x)
    if not np.isfinite(v):
        raise ValueError(name + " must be finite")
    return v

def _nlce_params(dev, keys):
    return [dev[k] for k in keys]

@functools.lru_cache(maxsize=None)
def _nlce_chain(m, J):
    """H = -sum_i sigma^z_i - J sum_i sigma^x_i sigma^x_{i+1} on an open chain of m sites."""
    sx = np.array([[0.0, 1.0], [1.0, 0.0]])
    sz = np.array([[1.0, 0.0], [0.0, -1.0]])

    def _op(site, o):
        out = np.array([[1.0]])
        for q in range(m):
            out = np.kron(out, o if q == site else np.eye(2))
        return out

    H = np.zeros((2 ** m, 2 ** m))
    for i in range(m):
        H -= _op(i, sz)
    for i in range(m - 1):
        H -= J * (_op(i, sx) @ _op(i + 1, sx))
    H.setflags(write=False)
    return H


def _nlce_do_hamiltonian_element(a, b, m, dev):
    """Matrix element H[a, b] of the open transverse-field Ising chain."""
    m = int(_nlce_real(m, "m"))
    a, b = int(_nlce_real(a, "a")), int(_nlce_real(b, "b"))
    if not (0 <= a < 2 ** m and 0 <= b < 2 ** m):
        raise ValueError("basis index out of range")
    return float(_nlce_chain(m, _nlce_real(_nlce_params(dev, ("J",))[0], "J"))[a, b])


def _nlce_flip_rows(m):
    """Product-basis indices of the m single-spin-flip states, site 0 first."""
    return [1 << (m - 1 - i) for i in range(m)]

@functools.lru_cache(maxsize=None)
def _nlce_eig(m, J):
    """Spectrum and eigenvectors of the m-site chain, cached read-only by (m, J)."""
    lam, S = np.linalg.eigh(_nlce_chain(m, J))
    lam.setflags(write=False)
    S.setflags(write=False)
    return lam, S


def _nlce_spectrum(m, dev):
    """Spectrum of the m-site chain, whose Hamiltonian must agree with step 01's elements."""
    J = _nlce_real(_nlce_params(dev, ("J",))[0], "J")
    m = int(m)
    if m < 1:
        raise ValueError("chain length must be at least 1")
    _nlce_bind()
    H = _nlce_chain(m, J)
    probe = [(0, 0), (1, 1), (0, 3)] if m >= 2 else [(0, 0), (1, 1)]
    for a, b in probe:                                                      # step 01
        if abs(hamiltonian_element(a, b, m, dev) - H[a, b]) > 1e-12:
            raise ValueError("the chain does not have the matrix elements step 01 reports")
    return _nlce_eig(m, J)


def _nlce_subset(m, dev):
    """The m eigenstates that deform the target block least, Eq. (39)."""
    m = int(m)
    lam, S = _nlce_spectrum(m, dev)
    J = _nlce_real(_nlce_params(dev, ("J",))[0], "J")
    return _nlce_select(m, J), lam, S


@functools.lru_cache(maxsize=None)
def _nlce_select(m, J):
    """Eq. (39) argmin, cached by (m, J). Seed with the m eigenstates of largest weight on the
    single-flip subspace, then exchange one member at a time for any eigenstate with weight there
    while that lowers the restricted deformation. Checked against exhaustive search over the
    highest-weight candidates on every chain and coupling the task uses."""
    _lam, S = _nlce_eig(m, J)
    rows = _nlce_flip_rows(m)
    p = (S[rows, :] ** 2).sum(axis=0)
    order = np.argsort(-p, kind="stable")
    sub = sorted(order[:m].tolist())
    pool = [int(a) for a in order if p[a] > 1e-12]

    def _dev_of(cols):
        sg = np.linalg.svd(S[np.ix_(rows, cols)], compute_uv=False)
        return float(np.sqrt(np.sum((sg - 1.0) ** 2)))

    best = _dev_of(sub)
    improved = True
    while improved:
        improved = False
        for pos in range(m):
            for c in pool:
                if c in sub:
                    continue
                trial = sorted(sub[:pos] + [c] + sub[pos + 1:])
                d = _dev_of(trial)
                if d < best - 1e-12:
                    best, sub, improved = d, trial, True
    return tuple(sub)


def _nlce_svd_block(m, dev):
    m = int(m)
    sub, lam, S = _nlce_subset(m, dev)
    U, sg, Vh = np.linalg.svd(S[np.ix_(_nlce_flip_rows(m), list(sub))])
    return U, sg, Vh, np.diag(lam[list(sub)]), lam


def _nlce_bind():
    """Bind an earlier step's oracle to this file's copy only when the grader has not supplied
    it; setdefault never overrides a real one, so the chain always runs through the oracles
    actually present."""
    g = globals()
    for name in ("hamiltonian_element", "selection_deviation", "effective_element", "cluster_quantity",
                 "cluster_weight", "nlce_hopping"):
        if "_nlce_do_" + name in g:
            g.setdefault("" + name, g["_nlce_do_" + name])


def _nlce_do_selection_deviation(m, dev):
    """||T_11 - 1|| of the selected set; T_11 = U Sigma U^dag, Eq. (13)."""
    m = int(_nlce_real(m, "m"))
    _U, sg, _Vh, _L, _lam = _nlce_svd_block(m, dev)
    return float(np.sqrt(np.sum((sg - 1.0) ** 2)))

def _nlce_do_effective_element(i, j, m, dev):
    """[H_eff]_ij by Theorem 1, Eq. (14), on the set whose deformation step 02 reports."""
    m = int(_nlce_real(m, "m"))
    i, j = int(_nlce_real(i, "i")), int(_nlce_real(j, "j"))
    if not (0 <= i < m and 0 <= j < m):
        raise ValueError("site index out of range")
    U, sg, Vh, Lam, _lam = _nlce_svd_block(m, dev)
    d = float(np.sqrt(np.sum((sg - 1.0) ** 2)))
    if abs(selection_deviation(m, dev) - d) > 1e-9 * (1.0 + d):     # step 02
        raise ValueError("the selected set does not have the deformation step 02 reports")
    P = U @ Vh
    return float((P @ Lam @ P.conj().T)[i, j])

def _nlce_do_cluster_quantity(m, r, dev):
    """Distance-r pair sum of the step-03 elements; for r = 0 the ground-state energy is
    removed m times so excitation energies stay cluster additive."""
    m, r = int(_nlce_real(m, "m")), int(_nlce_real(r, "r"))
    if r < 0:
        raise ValueError("separation must be non-negative")
    if m <= r:
        return 0.0
    s = sum(effective_element(i, i + r, m, dev) for i in range(m - r))    # step 03
    if r == 0:
        s -= m * float(_nlce_spectrum(m, dev)[0][0])
    return float(s)

def _nlce_do_cluster_weight(m, r, dev):
    """W_(r,m) = T_(r,m) - sum_s (m - s + 1) W_(r,s), Eq. (41), from the step-04 pair sums."""
    m, r = int(_nlce_real(m, "m")), int(_nlce_real(r, "r"))
    if m < 1:
        raise ValueError("chain length must be at least 1")
    W = {}
    for q in range(1, m + 1):
        w = cluster_quantity(q, r, dev)                              # step 04
        for s in range(1, q):
            w -= (q - s + 1) * W[s]
        W[q] = w
    return float(W[m])

def cluster_weight(m: int, r: int, dev: dict) -> float:
    """Linked-cluster weight of the m-site chain at separation r."""
    _nlce_bind()
    return _nlce_do_cluster_weight(m, r, dev)

import functools

import numpy as np

def _nlce_default_device():
    return dict(J=0.8, n_max=8, k=1.2566370614359172)

def _nlce_real(x, name):
    v = float(x)
    if not np.isfinite(v):
        raise ValueError(name + " must be finite")
    return v

def _nlce_params(dev, keys):
    return [dev[k] for k in keys]

@functools.lru_cache(maxsize=None)
def _nlce_chain(m, J):
    """H = -sum_i sigma^z_i - J sum_i sigma^x_i sigma^x_{i+1} on an open chain of m sites."""
    sx = np.array([[0.0, 1.0], [1.0, 0.0]])
    sz = np.array([[1.0, 0.0], [0.0, -1.0]])

    def _op(site, o):
        out = np.array([[1.0]])
        for q in range(m):
            out = np.kron(out, o if q == site else np.eye(2))
        return out

    H = np.zeros((2 ** m, 2 ** m))
    for i in range(m):
        H -= _op(i, sz)
    for i in range(m - 1):
        H -= J * (_op(i, sx) @ _op(i + 1, sx))
    H.setflags(write=False)
    return H


def _nlce_do_hamiltonian_element(a, b, m, dev):
    """Matrix element H[a, b] of the open transverse-field Ising chain."""
    m = int(_nlce_real(m, "m"))
    a, b = int(_nlce_real(a, "a")), int(_nlce_real(b, "b"))
    if not (0 <= a < 2 ** m and 0 <= b < 2 ** m):
        raise ValueError("basis index out of range")
    return float(_nlce_chain(m, _nlce_real(_nlce_params(dev, ("J",))[0], "J"))[a, b])


def _nlce_flip_rows(m):
    """Product-basis indices of the m single-spin-flip states, site 0 first."""
    return [1 << (m - 1 - i) for i in range(m)]

@functools.lru_cache(maxsize=None)
def _nlce_eig(m, J):
    """Spectrum and eigenvectors of the m-site chain, cached read-only by (m, J)."""
    lam, S = np.linalg.eigh(_nlce_chain(m, J))
    lam.setflags(write=False)
    S.setflags(write=False)
    return lam, S


def _nlce_spectrum(m, dev):
    """Spectrum of the m-site chain, whose Hamiltonian must agree with step 01's elements."""
    J = _nlce_real(_nlce_params(dev, ("J",))[0], "J")
    m = int(m)
    if m < 1:
        raise ValueError("chain length must be at least 1")
    _nlce_bind()
    H = _nlce_chain(m, J)
    probe = [(0, 0), (1, 1), (0, 3)] if m >= 2 else [(0, 0), (1, 1)]
    for a, b in probe:                                                      # step 01
        if abs(hamiltonian_element(a, b, m, dev) - H[a, b]) > 1e-12:
            raise ValueError("the chain does not have the matrix elements step 01 reports")
    return _nlce_eig(m, J)


def _nlce_subset(m, dev):
    """The m eigenstates that deform the target block least, Eq. (39)."""
    m = int(m)
    lam, S = _nlce_spectrum(m, dev)
    J = _nlce_real(_nlce_params(dev, ("J",))[0], "J")
    return _nlce_select(m, J), lam, S


@functools.lru_cache(maxsize=None)
def _nlce_select(m, J):
    """Eq. (39) argmin, cached by (m, J). Seed with the m eigenstates of largest weight on the
    single-flip subspace, then exchange one member at a time for any eigenstate with weight there
    while that lowers the restricted deformation. Checked against exhaustive search over the
    highest-weight candidates on every chain and coupling the task uses."""
    _lam, S = _nlce_eig(m, J)
    rows = _nlce_flip_rows(m)
    p = (S[rows, :] ** 2).sum(axis=0)
    order = np.argsort(-p, kind="stable")
    sub = sorted(order[:m].tolist())
    pool = [int(a) for a in order if p[a] > 1e-12]

    def _dev_of(cols):
        sg = np.linalg.svd(S[np.ix_(rows, cols)], compute_uv=False)
        return float(np.sqrt(np.sum((sg - 1.0) ** 2)))

    best = _dev_of(sub)
    improved = True
    while improved:
        improved = False
        for pos in range(m):
            for c in pool:
                if c in sub:
                    continue
                trial = sorted(sub[:pos] + [c] + sub[pos + 1:])
                d = _dev_of(trial)
                if d < best - 1e-12:
                    best, sub, improved = d, trial, True
    return tuple(sub)


def _nlce_svd_block(m, dev):
    m = int(m)
    sub, lam, S = _nlce_subset(m, dev)
    U, sg, Vh = np.linalg.svd(S[np.ix_(_nlce_flip_rows(m), list(sub))])
    return U, sg, Vh, np.diag(lam[list(sub)]), lam


def _nlce_bind():
    """Bind an earlier step's oracle to this file's copy only when the grader has not supplied
    it; setdefault never overrides a real one, so the chain always runs through the oracles
    actually present."""
    g = globals()
    for name in ("hamiltonian_element", "selection_deviation", "effective_element", "cluster_quantity",
                 "cluster_weight", "nlce_hopping"):
        if "_nlce_do_" + name in g:
            g.setdefault("" + name, g["_nlce_do_" + name])


def _nlce_do_selection_deviation(m, dev):
    """||T_11 - 1|| of the selected set; T_11 = U Sigma U^dag, Eq. (13)."""
    m = int(_nlce_real(m, "m"))
    _U, sg, _Vh, _L, _lam = _nlce_svd_block(m, dev)
    return float(np.sqrt(np.sum((sg - 1.0) ** 2)))

def _nlce_do_effective_element(i, j, m, dev):
    """[H_eff]_ij by Theorem 1, Eq. (14), on the set whose deformation step 02 reports."""
    m = int(_nlce_real(m, "m"))
    i, j = int(_nlce_real(i, "i")), int(_nlce_real(j, "j"))
    if not (0 <= i < m and 0 <= j < m):
        raise ValueError("site index out of range")
    U, sg, Vh, Lam, _lam = _nlce_svd_block(m, dev)
    d = float(np.sqrt(np.sum((sg - 1.0) ** 2)))
    if abs(selection_deviation(m, dev) - d) > 1e-9 * (1.0 + d):     # step 02
        raise ValueError("the selected set does not have the deformation step 02 reports")
    P = U @ Vh
    return float((P @ Lam @ P.conj().T)[i, j])

def _nlce_do_cluster_quantity(m, r, dev):
    """Distance-r pair sum of the step-03 elements; for r = 0 the ground-state energy is
    removed m times so excitation energies stay cluster additive."""
    m, r = int(_nlce_real(m, "m")), int(_nlce_real(r, "r"))
    if r < 0:
        raise ValueError("separation must be non-negative")
    if m <= r:
        return 0.0
    s = sum(effective_element(i, i + r, m, dev) for i in range(m - r))    # step 03
    if r == 0:
        s -= m * float(_nlce_spectrum(m, dev)[0][0])
    return float(s)

def _nlce_do_cluster_weight(m, r, dev):
    """W_(r,m) = T_(r,m) - sum_s (m - s + 1) W_(r,s), Eq. (41), from the step-04 pair sums."""
    m, r = int(_nlce_real(m, "m")), int(_nlce_real(r, "r"))
    if m < 1:
        raise ValueError("chain length must be at least 1")
    W = {}
    for q in range(1, m + 1):
        w = cluster_quantity(q, r, dev)                              # step 04
        for s in range(1, q):
            w -= (q - s + 1) * W[s]
        W[q] = w
    return float(W[m])

def _nlce_do_nlce_hopping(r, dev):
    """t_r = sum_m W_(r,m) up to n_max, Eq. (40), from the step-05 weights."""
    r = int(_nlce_real(r, "r"))
    n_max = int(_nlce_real(_nlce_params(dev, ("n_max",))[0], "n_max"))
    if n_max < 1:
        raise ValueError("n_max must be at least 1")
    return float(sum(cluster_weight(m, r, dev) for m in range(1, n_max + 1)))   # step 05

def nlce_hopping(r: int, dev: dict) -> float:
    """Order-n_max hopping amplitude at separation r."""
    _nlce_bind()
    return _nlce_do_nlce_hopping(r, dev)

import functools

import numpy as np

def _nlce_default_device():
    return dict(J=0.8, n_max=8, k=1.2566370614359172)

def _nlce_real(x, name):
    v = float(x)
    if not np.isfinite(v):
        raise ValueError(name + " must be finite")
    return v

def _nlce_params(dev, keys):
    return [dev[k] for k in keys]

@functools.lru_cache(maxsize=None)
def _nlce_chain(m, J):
    """H = -sum_i sigma^z_i - J sum_i sigma^x_i sigma^x_{i+1} on an open chain of m sites."""
    sx = np.array([[0.0, 1.0], [1.0, 0.0]])
    sz = np.array([[1.0, 0.0], [0.0, -1.0]])

    def _op(site, o):
        out = np.array([[1.0]])
        for q in range(m):
            out = np.kron(out, o if q == site else np.eye(2))
        return out

    H = np.zeros((2 ** m, 2 ** m))
    for i in range(m):
        H -= _op(i, sz)
    for i in range(m - 1):
        H -= J * (_op(i, sx) @ _op(i + 1, sx))
    H.setflags(write=False)
    return H


def _nlce_do_hamiltonian_element(a, b, m, dev):
    """Matrix element H[a, b] of the open transverse-field Ising chain."""
    m = int(_nlce_real(m, "m"))
    a, b = int(_nlce_real(a, "a")), int(_nlce_real(b, "b"))
    if not (0 <= a < 2 ** m and 0 <= b < 2 ** m):
        raise ValueError("basis index out of range")
    return float(_nlce_chain(m, _nlce_real(_nlce_params(dev, ("J",))[0], "J"))[a, b])


def _nlce_flip_rows(m):
    """Product-basis indices of the m single-spin-flip states, site 0 first."""
    return [1 << (m - 1 - i) for i in range(m)]

@functools.lru_cache(maxsize=None)
def _nlce_eig(m, J):
    """Spectrum and eigenvectors of the m-site chain, cached read-only by (m, J)."""
    lam, S = np.linalg.eigh(_nlce_chain(m, J))
    lam.setflags(write=False)
    S.setflags(write=False)
    return lam, S


def _nlce_spectrum(m, dev):
    """Spectrum of the m-site chain, whose Hamiltonian must agree with step 01's elements."""
    J = _nlce_real(_nlce_params(dev, ("J",))[0], "J")
    m = int(m)
    if m < 1:
        raise ValueError("chain length must be at least 1")
    _nlce_bind()
    H = _nlce_chain(m, J)
    probe = [(0, 0), (1, 1), (0, 3)] if m >= 2 else [(0, 0), (1, 1)]
    for a, b in probe:                                                      # step 01
        if abs(hamiltonian_element(a, b, m, dev) - H[a, b]) > 1e-12:
            raise ValueError("the chain does not have the matrix elements step 01 reports")
    return _nlce_eig(m, J)


def _nlce_subset(m, dev):
    """The m eigenstates that deform the target block least, Eq. (39)."""
    m = int(m)
    lam, S = _nlce_spectrum(m, dev)
    J = _nlce_real(_nlce_params(dev, ("J",))[0], "J")
    return _nlce_select(m, J), lam, S


@functools.lru_cache(maxsize=None)
def _nlce_select(m, J):
    """Eq. (39) argmin, cached by (m, J). Seed with the m eigenstates of largest weight on the
    single-flip subspace, then exchange one member at a time for any eigenstate with weight there
    while that lowers the restricted deformation. Checked against exhaustive search over the
    highest-weight candidates on every chain and coupling the task uses."""
    _lam, S = _nlce_eig(m, J)
    rows = _nlce_flip_rows(m)
    p = (S[rows, :] ** 2).sum(axis=0)
    order = np.argsort(-p, kind="stable")
    sub = sorted(order[:m].tolist())
    pool = [int(a) for a in order if p[a] > 1e-12]

    def _dev_of(cols):
        sg = np.linalg.svd(S[np.ix_(rows, cols)], compute_uv=False)
        return float(np.sqrt(np.sum((sg - 1.0) ** 2)))

    best = _dev_of(sub)
    improved = True
    while improved:
        improved = False
        for pos in range(m):
            for c in pool:
                if c in sub:
                    continue
                trial = sorted(sub[:pos] + [c] + sub[pos + 1:])
                d = _dev_of(trial)
                if d < best - 1e-12:
                    best, sub, improved = d, trial, True
    return tuple(sub)


def _nlce_svd_block(m, dev):
    m = int(m)
    sub, lam, S = _nlce_subset(m, dev)
    U, sg, Vh = np.linalg.svd(S[np.ix_(_nlce_flip_rows(m), list(sub))])
    return U, sg, Vh, np.diag(lam[list(sub)]), lam


def _nlce_bind():
    """Bind an earlier step's oracle to this file's copy only when the grader has not supplied
    it; setdefault never overrides a real one, so the chain always runs through the oracles
    actually present."""
    g = globals()
    for name in ("hamiltonian_element", "selection_deviation", "effective_element", "cluster_quantity",
                 "cluster_weight", "nlce_hopping"):
        if "_nlce_do_" + name in g:
            g.setdefault("" + name, g["_nlce_do_" + name])


def _nlce_do_selection_deviation(m, dev):
    """||T_11 - 1|| of the selected set; T_11 = U Sigma U^dag, Eq. (13)."""
    m = int(_nlce_real(m, "m"))
    _U, sg, _Vh, _L, _lam = _nlce_svd_block(m, dev)
    return float(np.sqrt(np.sum((sg - 1.0) ** 2)))

def _nlce_do_effective_element(i, j, m, dev):
    """[H_eff]_ij by Theorem 1, Eq. (14), on the set whose deformation step 02 reports."""
    m = int(_nlce_real(m, "m"))
    i, j = int(_nlce_real(i, "i")), int(_nlce_real(j, "j"))
    if not (0 <= i < m and 0 <= j < m):
        raise ValueError("site index out of range")
    U, sg, Vh, Lam, _lam = _nlce_svd_block(m, dev)
    d = float(np.sqrt(np.sum((sg - 1.0) ** 2)))
    if abs(selection_deviation(m, dev) - d) > 1e-9 * (1.0 + d):     # step 02
        raise ValueError("the selected set does not have the deformation step 02 reports")
    P = U @ Vh
    return float((P @ Lam @ P.conj().T)[i, j])

def _nlce_do_cluster_quantity(m, r, dev):
    """Distance-r pair sum of the step-03 elements; for r = 0 the ground-state energy is
    removed m times so excitation energies stay cluster additive."""
    m, r = int(_nlce_real(m, "m")), int(_nlce_real(r, "r"))
    if r < 0:
        raise ValueError("separation must be non-negative")
    if m <= r:
        return 0.0
    s = sum(effective_element(i, i + r, m, dev) for i in range(m - r))    # step 03
    if r == 0:
        s -= m * float(_nlce_spectrum(m, dev)[0][0])
    return float(s)

def _nlce_do_cluster_weight(m, r, dev):
    """W_(r,m) = T_(r,m) - sum_s (m - s + 1) W_(r,s), Eq. (41), from the step-04 pair sums."""
    m, r = int(_nlce_real(m, "m")), int(_nlce_real(r, "r"))
    if m < 1:
        raise ValueError("chain length must be at least 1")
    W = {}
    for q in range(1, m + 1):
        w = cluster_quantity(q, r, dev)                              # step 04
        for s in range(1, q):
            w -= (q - s + 1) * W[s]
        W[q] = w
    return float(W[m])

def _nlce_do_nlce_hopping(r, dev):
    """t_r = sum_m W_(r,m) up to n_max, Eq. (40), from the step-05 weights."""
    r = int(_nlce_real(r, "r"))
    n_max = int(_nlce_real(_nlce_params(dev, ("n_max",))[0], "n_max"))
    if n_max < 1:
        raise ValueError("n_max must be at least 1")
    return float(sum(cluster_weight(m, r, dev) for m in range(1, n_max + 1)))   # step 05

def one_magnon_energy(dev: dict) -> float:
    """One-magnon energy at the reported wavevector, from the order-n_max expansion."""
    _nlce_bind()
    n_max = int(_nlce_real(_nlce_params(dev, ("n_max",))[0], "n_max"))
    k = _nlce_real(_nlce_params(dev, ("k",))[0], "k")
    if n_max < 1:
        raise ValueError("n_max must be at least 1")
    t = [nlce_hopping(r, dev) for r in range(n_max)]           # step 06
    return float(t[0] + 2.0 * sum(t[r] * np.cos(k * r) for r in range(1, n_max)))
SCICODE_GOLD_EOF
