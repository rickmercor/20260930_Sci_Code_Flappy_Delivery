"""
Choose the eigenstates that deform the target block least and report that deformation.
Returns the Frobenius deviation, dimensionless.

A block diagonalisation of the chain Hamiltonian $H$ with respect to a chosen set of $m$ eigenstates is a unitary $T$ that sends the $m$ single-flip basis states into the span of those eigenstates, so that $T^\dagger HT$ has no elements linking the one-magnon block to the rest of the Hilbert space. Such $T$ are not unique. Write $T_{11}$ for the $m$-by-$m$ matrix whose element in row $j$ and column $i$ is the overlap of single-flip state $j$ with $T$ applied to single-flip state $i$, with sites in order. For a given set, among all its block-diagonalising transformations take the one whose $T_{11}$ is closest to the identity in the Frobenius norm, and call that distance, $\|T_{11}-1\|_F$, the deformation of the set. The selected set is the set of $m$ eigenstates whose deformation is smallest over all sets of $m$ eigenstates. Return that smallest deformation.



The chains tested have up to $8$ sites, so the spectrum has up to $256$ eigenstates and there are more than $10^{14}$ sets of $8$ of them: an exhaustive search over all sets, or over all sets drawn from one symmetry sector, never finishes, and the later steps reuse this selection on every chain. Each call is expected to finish in well under a second.

Returns
-------
float: the Frobenius deviation $\|T_{11}-1\|_F$ of the chosen set.
"""

import numpy as np

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def selection_deviation(m: int, dev: dict) -> float:
    r"""Choose the eigenstates that deform the target block least and report that deformation.

    Returns the Frobenius deviation, dimensionless.

    Parameters
    ----------
    m : int
        Number of sites in the chain, $m\ge1$.
    dev : dict
        Parameter dict. Key read here: ``J``, the transverse-field Ising
        coupling, $J\ge0$ (the selection is defined beyond the gapped range
        $0\le J<1$ used by the expansion).

    Returns
    -------
    deviation : float
        The Frobenius deviation $\lVert T_{11}-1\rVert_F$ of the chosen set.

    Raises
    ------
    ValueError
        If $m$ or $J$ is not finite, or if $m<1$.
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

def _oracle_selection_deviation(m: int, dev: dict) -> float:
    """Frobenius deviation of the selected target block from the identity."""
    return _nlce_do_selection_deviation(m, dev)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        # normal: the chain used for the expansion
        {"setup": 'import numpy as np; dev = _nlce_default_device()',
         "call": 'selection_deviation(8, dev)',
         "gold_call": '_oracle_selection_deviation(8, dev)'},
        # boundary: the shortest chain with an interior site
        {"setup": 'import numpy as np; dev = _nlce_default_device()',
         "call": 'selection_deviation(3, dev)',
         "gold_call": '_oracle_selection_deviation(3, dev)'},
        # normal: weaker coupling deforms the block less
        {"setup": 'import numpy as np; dev = dict(_nlce_default_device(), J=0.2)',
         "call": 'selection_deviation(6, dev)',
         "gold_call": '_oracle_selection_deviation(6, dev)'},
        # edge: near the critical coupling the deformation is largest
        {"setup": 'import numpy as np; dev = dict(_nlce_default_device(), J=0.95)',
         "call": 'selection_deviation(6, dev)',
         "gold_call": '_oracle_selection_deviation(6, dev)'},
        # edge: beyond the critical coupling, where the largest-weight states are not the least-deforming set
        {"setup": 'import numpy as np; dev = dict(_nlce_default_device(), J=1.5)',
         "call": 'selection_deviation(6, dev)',
         "gold_call": '_oracle_selection_deviation(6, dev)'},
        # edge: the longest chain beyond the critical coupling
        {"setup": 'import numpy as np; dev = dict(_nlce_default_device(), J=1.3)',
         "call": 'selection_deviation(8, dev)',
         "gold_call": '_oracle_selection_deviation(8, dev)'},
        # edge: near the critical coupling the longest chain keeps a level above the lowest ones
        {"setup": 'import numpy as np; dev = dict(_nlce_default_device(), J=0.95)',
         "call": 'selection_deviation(8, dev)',
         "gold_call": '_oracle_selection_deviation(8, dev)'},
    ]
