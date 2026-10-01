"""
Isolates the shape dependent boundary term together with the finite size correction, as the finite lattice sum minus the Ewald bulk potential.

Separating the shape independent periodic bulk from the finite-minus-bulk remainder makes conditional convergence tractable. The remainder contains both the asymptotic boundary contribution and a finite size correction.

Returns
-------
A float64 array of shape (1,).
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def clc_boundary_term(cell_edges: "np.ndarray", basis_pos: "np.ndarray", basis_q: "np.ndarray", ref_index: int, px: int, py: int, pz: int, spherical: bool, alpha: float, n_real: int, n_rec: int) -> "np.ndarray":
    r"""All arguments as in the direct potential and Ewald steps.

    The finite lattice sum of a given shape and size differs from the shape independent bulk potential
    by a boundary term that depends on the shape but not the size and a finite size correction that
    vanishes as the crystal grows. Return their sum, that is the finite lattice sum of this shape
    and size minus the Ewald bulk potential of the same cell and reference ion.

    For a neutral cell, a vanishing total dipole $\mathbf{M}=\sum_j q_j\mathbf{r}_j$ eliminates
    the asymptotic boundary contribution to the total cell energy, but need not eliminate the
    boundary contribution to an individual site potential. Return the finite-minus-bulk site
    potential as specified; do not replace it by zero solely because $\mathbf{M}=\mathbf{0}$.

    Returns a numpy float64 array of shape $(1,)$.

    Raises:
        ValueError: any condition either underlying step rejects, or a non-finite result.
    """
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
from scipy.special import erfc


def _fin(x, name):
    v = float(x)
    if not np.isfinite(v):
        raise ValueError("%s must be finite" % name)
    return v


def _pos(x, name):
    v = _fin(x, name)
    if v <= 0.0:
        raise ValueError("%s must be positive" % name)
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


def _edges(cell_edges):
    L = np.asarray(cell_edges, dtype=np.float64)
    if L.shape != (3,):
        raise ValueError("cell_edges must have shape (3,)")
    if not np.all(np.isfinite(L)) or np.any(L <= 0.0):
        raise ValueError("every cell edge must be finite and positive")
    return L


def _cell(basis_pos, basis_q):
    """Validate a unit cell given in reduced coordinates with a neutral charge set."""
    p = np.asarray(basis_pos, dtype=np.float64)
    q = np.asarray(basis_q, dtype=np.float64)
    if p.ndim != 2 or p.shape[1] != 3 or p.shape[0] < 2:
        raise ValueError("basis_pos must have shape (m, 3) with m at least 2")
    if q.ndim != 1 or q.shape[0] != p.shape[0]:
        raise ValueError("basis_q must have one charge per basis position")
    if not np.all(np.isfinite(p)) or not np.all(np.isfinite(q)):
        raise ValueError("non-finite basis")
    if np.any(p < 0.0) or np.any(p >= 1.0):
        raise ValueError("reduced basis positions must lie in [0, 1)")
    if abs(float(q.sum())) > 1e-12:
        raise ValueError("the unit cell must be charge neutral")
    d = np.linalg.norm(p[:, None, :] - p[None, :, :], axis=2)
    np.fill_diagonal(d, np.inf)
    if np.min(d) < 1e-9:
        raise ValueError("two basis positions coincide")
    return p, q


def _offsets(px, py, pz, spherical):
    """Integer lattice vectors of the finite crystal, origin removed, lexicographic in
    (n_x, n_y, n_z). The sphere, when asked for, is applied to the integer indices."""
    g = np.stack(np.meshgrid(np.arange(-px, px + 1), np.arange(-py, py + 1),
                             np.arange(-pz, pz + 1), indexing="ij"), axis=-1).reshape(-1, 3).astype(np.float64)
    g = g[np.any(g != 0.0, axis=1)]
    if spherical:
        g = g[np.linalg.norm(g, axis=1) <= float(min(px, py, pz)) + 1e-12]
    return g[np.lexsort((g[:, 2], g[:, 1], g[:, 0]))]


def _direct(L, p, q, ref, off):
    """Finite lattice sum at basis ion `ref`. Reduced coordinates are carried to Cartesian by
    the cell edges; each non central cell enters as its charges minus the same charges at that
    cell's origin."""
    r0 = p[ref] * L
    nl = off * L
    dn = np.linalg.norm(nl, axis=1)
    tot = 0.0
    for j in range(p.shape[0]):
        rj = p[j] * L
        if j != ref:
            d0 = np.linalg.norm(rj - r0)
            if d0 < 1e-12:
                raise ValueError("two basis ions coincide")
            tot += q[j] / d0
        dj = np.linalg.norm(nl + (rj - r0), axis=1)
        if np.any(dj < 1e-12):
            raise ValueError("an image coincides with the reference ion")
        tot += q[j] * np.sum(1.0 / dj - 1.0 / dn)
    if not np.isfinite(tot):
        raise ValueError("non-finite lattice sum")
    return float(tot)


def _ewald(L, p, q, ref, alpha, n_real, n_rec):
    """Tinfoil Ewald potential at basis ion `ref` of the orthorhombic periodic lattice."""
    V = float(np.prod(L))
    r0 = p[ref] * L
    a = np.arange(-n_real, n_real + 1)
    sh = np.stack(np.meshgrid(a, a, a, indexing="ij"), axis=-1).reshape(-1, 3).astype(np.float64) * L
    tot = 0.0
    for j in range(p.shape[0]):
        d = np.linalg.norm(p[j] * L + sh - r0, axis=1)
        m = d > 1e-12
        tot += q[j] * np.sum(erfc(alpha * d[m]) / d[m])
    b = np.arange(-n_rec, n_rec + 1)
    bb = np.stack(np.meshgrid(b, b, b, indexing="ij"), axis=-1).reshape(-1, 3).astype(np.float64)
    bb = bb[np.any(bb != 0.0, axis=1)]
    ks = 2.0 * np.pi * bb / L
    k2 = np.sum(ks * ks, axis=1)
    S = np.sum(q[:, None] * np.exp(1j * (ks @ (p * L).T).T), axis=0)
    tot += float(np.real(np.sum(4.0 * np.pi / V * np.exp(-k2 / (4.0 * alpha ** 2)) / k2 * S * np.exp(-1j * (ks @ r0)))))
    tot -= 2.0 * alpha / np.sqrt(np.pi) * q[ref]
    if not np.isfinite(tot):
        raise ValueError("non-finite Ewald potential")
    return float(tot)


def _nearest(L, p, ref, span=2):
    """Distance from the reference ion to the nearest other ion of the crystal, searching every
    basis ion in every cell with |n_i| <= span, the reference ion itself excluded."""
    r0 = p[ref] * L
    a = np.arange(-span, span + 1)
    sh = np.stack(np.meshgrid(a, a, a, indexing="ij"), axis=-1).reshape(-1, 3).astype(np.float64) * L
    best = np.inf
    for j in range(p.shape[0]):
        d = np.linalg.norm(p[j] * L + sh - r0, axis=1)
        d = d[d > 1e-12]
        if d.size:
            best = min(best, float(np.min(d)))
    if not np.isfinite(best) or best <= 0.0:
        raise ValueError("no neighbouring ion found")
    return best

def _oracle_clc_boundary_term(cell_edges: "np.ndarray", basis_pos: "np.ndarray", basis_q: "np.ndarray", ref_index: int, px: int, py: int, pz: int, spherical: bool, alpha: float, n_real: int, n_rec: int) -> "np.ndarray":
    if _pos(alpha, "alpha") <= 0.0:
        raise ValueError("alpha must be positive")
    d = _oracle_clc_direct_potential(cell_edges, basis_pos, basis_q, ref_index, px, py, pz, spherical)
    e = _oracle_clc_ewald_potential(cell_edges, basis_pos, basis_q, ref_index, alpha, n_real, n_rec)
    v = float(d[0] - e[0])
    if not np.isfinite(v):
        raise ValueError("non-finite boundary term")
    return np.array([v], dtype=np.float64)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {"tol":1e-9,"setup":"import numpy as np\n","call":"clc_boundary_term(np.array([1.,1,1]),np.array([[0.,0,0],[0.5,0,0]]),np.array([1.,-1]),1,16,16,4,False,4.0,3,8)","gold_call":"_oracle_clc_boundary_term(np.array([1.,1,1]),np.array([[0.,0,0],[0.5,0,0]]),np.array([1.,-1]),1,16,16,4,False,4.0,3,8)"},
        {"tol":1e-9,"setup":"import numpy as np\n","call":"clc_boundary_term(np.array([1.,1.3,0.7]),np.array([[0.,0,0],[0.5,0,0]]),np.array([1.,-1]),1,14,14,14,False,4.0,3,9)","gold_call":"_oracle_clc_boundary_term(np.array([1.,1.3,0.7]),np.array([[0.,0,0],[0.5,0,0]]),np.array([1.,-1]),1,14,14,14,False,4.0,3,9)"},
        {"tol":1e-9,"setup":"import numpy as np\n","call":"clc_boundary_term(np.array([1.,1,1.6]),np.array([[0.,0,0],[0.5,0,0]]),np.array([1.,-1]),1,10,10,20,False,4.0,3,8)","gold_call":"_oracle_clc_boundary_term(np.array([1.,1,1.6]),np.array([[0.,0,0],[0.5,0,0]]),np.array([1.,-1]),1,10,10,20,False,4.0,3,8)"},
        {"tol":1e-9,"setup":"import numpy as np\n# boundary: this particular rocksalt basis cancels its site-boundary potential\n","call":"clc_boundary_term(np.array([1.,1,1]),np.array([[0.,0,0],[0.5,0.5,0],[0.5,0,0.5],[0,0.5,0.5],[0.5,0.5,0.5],[0.5,0,0],[0,0.5,0],[0,0,0.5]]),np.array([1.,1,1,1,-1,-1,-1,-1]),0,8,8,8,False,4.0,3,8)","gold_call":"_oracle_clc_boundary_term(np.array([1.,1,1]),np.array([[0.,0,0],[0.5,0.5,0],[0.5,0,0.5],[0,0.5,0.5],[0.5,0.5,0.5],[0.5,0,0],[0,0.5,0],[0,0,0.5]]),np.array([1.,1,1,1,-1,-1,-1,-1]),0,8,8,8,False,4.0,3,8)"},
        {"tol":1e-9,"setup":"import numpy as np\n# invalid input: a reference index past the last basis ion\ndef _probe(fn):\n    try:\n        fn(np.array([1.,1,1]),np.array([[0.,0,0],[0.5,0,0]]),np.array([1.,-1]),2,6,6,6,False,4.0,3,8)\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\n","call":"_probe(clc_boundary_term)","gold_call":"_probe(_oracle_clc_boundary_term)"},
        {"tol":1e-09,"setup":"import numpy as np\n# neutral zero-dipole basis with a nonzero site-boundary potential\n","call":"clc_boundary_term(np.array([1.,1.,1.]),np.array([[0.25,0.5,0.5],[0.5,0.5,0.5],[0.75,0.5,0.5]]),np.array([1.,-2.,1.]),1,16,16,16,False,4.0,3,9)","gold_call":"_oracle_clc_boundary_term(np.array([1.,1.,1.]),np.array([[0.25,0.5,0.5],[0.5,0.5,0.5],[0.75,0.5,0.5]]),np.array([1.,-2.,1.]),1,16,16,16,False,4.0,3,9)"},
    ]
