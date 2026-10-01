"""
Evaluates the finite Coulomb lattice sum at one ion of an orthorhombic unit cell over a finite crystal of given shape and size.

This is the conditionally convergent object. Pair each distant charge against the same charge at the translated reference site in that cell, which is the cell origin in reference-centered coordinates, so every distant cell enters as a neutral group.

Returns
-------
A float64 array of shape (1,).
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def clc_direct_potential(cell_edges: "np.ndarray", basis_pos: "np.ndarray", basis_q: "np.ndarray", ref_index: int, px: int, py: int, pz: int, spherical: bool) -> "np.ndarray":
    r"""cell_edges: array of shape (3,) of positive floats, the edge lengths of the orthorhombic
    unit cell along x, y and z. basis_pos: array of shape (m, 3) with m at least two, the
    positions of the ions in reduced coordinates, each component in $[0, 1)$. basis_q: array of
    shape (m,), their charges, summing to zero. ref_index: integer in $0 \le \mathrm{ref} < m$.
    px, py, pz, spherical: as in the offsets step.

    A reduced coordinate is carried to a Cartesian one by multiplying it componentwise by the cell
    edges, and a cell index $\mathbf{n}$ sits at the Cartesian point $\mathbf{n}\odot\mathbf{L}$.

    The potential at the reference ion has two parts. The other ions of the central cell contribute
    $q_j/|\mathbf{r}_j - \mathbf{r}_{\mathrm{ref}}|$ in Cartesian distance. Every other cell
    contributes, for each basis ion $j$, the difference
    $q_j(1/|\mathbf{n}\odot\mathbf{L} + \mathbf{r}_j - \mathbf{r}_{\mathrm{ref}}|
    - 1/|\mathbf{n}\odot\mathbf{L}|)$, that is the charge at its own site minus the same charge at
    the translated reference site $\mathbf{r}_{\mathrm{ref}} + \mathbf{n}\odot\mathbf{L}$.
    This site is the cell origin in reference-centered coordinates, at displacement
    $\mathbf{n}\odot\mathbf{L}$ from the reference ion. Apply the subtraction for every basis
    ion including the reference one.

    Returns a numpy float64 array of shape $(1,)$.

    Raises:
        ValueError: on cell_edges of the wrong shape or with a non-positive entry, a non-finite
            basis, fewer than two ions, a mismatched charge array, a reduced position outside
            $[0, 1)$, a non-neutral cell, two coincident positions, a ref_index outside its range,
            an invalid px, py or pz, or a non-finite result.
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

def _oracle_clc_direct_potential(cell_edges: "np.ndarray", basis_pos: "np.ndarray", basis_q: "np.ndarray", ref_index: int, px: int, py: int, pz: int, spherical: bool) -> "np.ndarray":
    L = _edges(cell_edges)
    p, q = _cell(basis_pos, basis_q)
    ref = _int(ref_index, "ref_index", 0)
    if ref >= p.shape[0]:
        raise ValueError("ref_index must be smaller than the number of basis ions")
    a = _int(px, "px", 1); b = _int(py, "py", 1); c = _int(pz, "pz", 1)
    off = _offsets(a, b, c, bool(spherical))
    if off.shape[0] == 0:
        raise ValueError("the finite crystal contains no cells besides the central one")
    return np.array([_direct(L, p, q, ref, off)], dtype=np.float64)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {"tol":1e-9,"setup":"import numpy as np\n","call":"clc_direct_potential(np.array([1.,1,1]),np.array([[0.,0,0],[0.5,0,0]]),np.array([1.,-1]),1,12,12,12,False)","gold_call":"_oracle_clc_direct_potential(np.array([1.,1,1]),np.array([[0.,0,0],[0.5,0,0]]),np.array([1.,-1]),1,12,12,12,False)"},
        {"tol":1e-9,"setup":"import numpy as np\n","call":"clc_direct_potential(np.array([1.,1.3,0.7]),np.array([[0.,0,0],[0.5,0,0]]),np.array([1.,-1]),1,10,10,10,False)","gold_call":"_oracle_clc_direct_potential(np.array([1.,1.3,0.7]),np.array([[0.,0,0],[0.5,0,0]]),np.array([1.,-1]),1,10,10,10,False)"},
        {"tol":1e-9,"setup":"import numpy as np\n","call":"clc_direct_potential(np.array([1.,1,1]),np.array([[0.,0,0],[0.5,0.5,0],[0.5,0,0.5],[0,0.5,0.5],[0.5,0.5,0.5],[0.5,0,0],[0,0.5,0],[0,0,0.5]]),np.array([1.,1,1,1,-1,-1,-1,-1]),0,5,5,5,False)","gold_call":"_oracle_clc_direct_potential(np.array([1.,1,1]),np.array([[0.,0,0],[0.5,0.5,0],[0.5,0,0.5],[0,0.5,0.5],[0.5,0.5,0.5],[0.5,0,0],[0,0.5,0],[0,0,0.5]]),np.array([1.,1,1,1,-1,-1,-1,-1]),0,5,5,5,False)"},
        {"tol":1e-9,"setup":"import numpy as np\n# boundary: a crystal one cell thick along z, where the shape is a plate\n","call":"clc_direct_potential(np.array([1.,1,1.6]),np.array([[0.,0,0],[0.5,0,0]]),np.array([1.,-1]),1,10,10,1,False)","gold_call":"_oracle_clc_direct_potential(np.array([1.,1,1.6]),np.array([[0.,0,0],[0.5,0,0]]),np.array([1.,-1]),1,10,10,1,False)"},
        {"tol":1e-9,"setup":"import numpy as np\n# invalid input: a cell edge of zero has no Cartesian embedding\ndef _probe(fn):\n    try:\n        fn(np.array([1.,0,1]),np.array([[0.,0,0],[0.5,0,0]]),np.array([1.,-1]),1,4,4,4,False)\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\n","call":"_probe(clc_direct_potential)","gold_call":"_probe(_oracle_clc_direct_potential)"},
    ]
