"""
Strip the subchain contributions from a chain's pair sum.
Returns the linked-cluster weight, dimensionless.

The linked-cluster weight of the $m$-site chain at separation $r$ is its pair sum $T_{(r,m)}$ from the previous step less the weights of all of its connected subclusters, in the sense of the standard numerical linked-cluster expansion. A subcluster is a set of sites, so a subchain of a given length placed at a different position inside the chain is a different subcluster. Weights are defined recursively from the smallest cluster upwards. Return the weight of the $m$-site chain at separation $r$.

Returns
-------
float: the linked-cluster weight $W_{(r,m)}$ of the $m$-site chain at separation $r$.
"""

import numpy as np

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def cluster_weight(m: int, r: int, dev: dict) -> float:
    r"""Strip the subchain contributions from a chain's pair sum.

    Returns the linked-cluster weight, dimensionless.

    Parameters
    ----------
    m : int
        Number of sites in the chain, $m\ge1$.
    r : int
        Separation in sites, $r\ge0$.
    dev : dict
        Parameter dict. Key read here: ``J``, the transverse-field Ising
        coupling, $0\le J<1$.

    Returns
    -------
    weight : float
        The linked-cluster weight $W_{(r,m)}$ of the $m$-site chain at
        separation $r$.

    Raises
    ------
    ValueError
        If $m$ or $r$ is not finite, if $m<1$, if $r<0$, or if $J$ is not
        finite while $r<m$.
    """
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

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
        if abs(_oracle_hamiltonian_element(a, b, m, dev) - H[a, b]) > 1e-12:
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
            g.setdefault("_oracle_" + name, g["_nlce_do_" + name])


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
    if abs(_oracle_selection_deviation(m, dev) - d) > 1e-9 * (1.0 + d):     # step 02
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
    s = sum(_oracle_effective_element(i, i + r, m, dev) for i in range(m - r))    # step 03
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
        w = _oracle_cluster_quantity(q, r, dev)                              # step 04
        for s in range(1, q):
            w -= (q - s + 1) * W[s]
        W[q] = w
    return float(W[m])

def _oracle_cluster_weight(m: int, r: int, dev: dict) -> float:
    """Linked-cluster weight of the m-site chain at separation r."""
    _nlce_bind()
    return _nlce_do_cluster_weight(m, r, dev)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        # normal: a five-site chain at next-nearest-neighbour separation
        {"setup": 'import numpy as np; dev = _nlce_default_device()',
         "call": 'cluster_weight(5, 2, dev)',
         "gold_call": '_oracle_cluster_weight(5, 2, dev)'},
        # boundary: the first chain that supports separation three
        {"setup": 'import numpy as np; dev = _nlce_default_device()',
         "call": 'cluster_weight(4, 3, dev)',
         "gold_call": '_oracle_cluster_weight(4, 3, dev)'},
        # normal: the diagonal channel
        {"setup": 'import numpy as np; dev = _nlce_default_device()',
         "call": 'cluster_weight(4, 0, dev)',
         "gold_call": '_oracle_cluster_weight(4, 0, dev)'},
        # edge: the single-site chain has no subchains to remove
        {"setup": 'import numpy as np; dev = _nlce_default_device()',
         "call": 'cluster_weight(1, 0, dev)',
         "gold_call": '_oracle_cluster_weight(1, 0, dev)'},
    ]
