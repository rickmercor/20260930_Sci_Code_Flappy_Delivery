"""
Assemble only over the exterior elements, and only the phases, nothing else: the material is the background everywhere outside the resonator, so no element carries the resonator parameters. Using the resonator moduli here is the single largest error available in this problem, and it scales the answer by the reciprocal square root of the stiffness contrast.

Having built the matrix over all n_side cubed nodes, extract the three blocks the condensation needs, indexing degrees of freedom as node index times three plus component, with the free and surface degree of freedom lists each sorted ascending. Return the free to free block, the free to surface block and the surface to surface block. Nodes strictly inside the resonator never appear.

Returns
-------
dict, holding the sparse blocks K_ff, K_fc and K_cc, and the index arrays free_dofs and surface_dofs.

Assemble the stiffness of the background occupying the cell outside the resonator, subject to the quasi-periodicity that Floquet-Bloch theory imposes on the cell. A displacement field in this setting is alpha-quasi-periodic, meaning

$$u(x + L n) = exp(i alpha . n) u(x)$$

for every integer translation n of the lattice, with the quasi-momentum alpha running over the Brillouin zone from minus pi to pi in each direction. The consequence for assembly is that the cell has only n_side cubed independent nodes, and whenever an element reaches past the far face of the cell the corner it reaches is not a new unknown but a known multiple of the node it wraps onto.

Concretely, for element with index triple (i, j, k) and corner offset (a, b, c), the corner sits at (i + a, j + b, k + c). Reduce each component modulo n_side to find the node that carries the unknown, and record the wrap vector w, whose entries are one where the sum reached n_side and zero otherwise. The corner value is then exp(i alpha . w) times the unknown. Substituting into the element energy, the contribution of that element to the reduced matrix entry joining corner p to corner q is

$$conj(exp(i alpha . w_p)) * element_stiffness[p, q] * exp(i alpha . w_q),$$

one corner carrying the conjugate phase and the other the plain one. What matters is that the two are conjugate to each other, which is what makes the assembled matrix Hermitian. The prompt fixes the displayed placement of the conjugate. Reversing both phases would conjugate the whole matrix, equivalently replacing alpha by minus alpha, and is a useful spectral cross-check rather than the delivered convention. Applying only one of the two phases, or omitting them altogether, is the error to guard against: it destroys the Hermitian symmetry or silently produces the ordinary periodic matrix belonging to alpha equal to zero.

Returns
-------
dict, holding the sparse blocks K_ff, K_fc and K_cc, and the index arrays free_dofs and surface_dofs.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def assemble_bloch_exterior(
    elements: np.ndarray,
    surface_nodes: np.ndarray,
    free_nodes: np.ndarray,
    element_stiffness: np.ndarray,
    corner_signs: np.ndarray,
    n_side: int,
    alpha: tuple,
) -> dict:
    """Assemble the quasi-periodic background stiffness outside the resonator and split it into blocks.

    Parameters
    ----------
    elements : np.ndarray
        Index triples of the exterior elements, shape (n_exterior, 3).
    surface_nodes : np.ndarray
        Flat indices of the resonator surface nodes, sorted ascending.
    free_nodes : np.ndarray
        Flat indices of the free nodes, sorted ascending.
    element_stiffness : np.ndarray
        Twenty-four by twenty-four element stiffness.
    corner_signs : np.ndarray
        Eight corner offsets of shape (8, 3).
    n_side : int
        Elements along each edge of the cell.
    alpha : tuple
        Three finite floats giving the quasi-momentum, not all zero.

    Returns
    -------
    dict
        Under the keys K_ff, K_fc, K_cc, free_dofs and surface_dofs.

    Raises
    ------
    ValueError
        When elements is not a two-dimensional integer array of three columns, when surface_nodes or free_nodes is not a non-empty one-dimensional integer array of indices inside the cell, when element_stiffness is not twenty-four by twenty-four, when corner_signs is not eight by three, when n_side is not an integer of four or more, when alpha does not hold exactly three finite floats, or when alpha is the zero vector.
    """
    return

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
import scipy.sparse as sp


def _checked_alpha(alpha):
    """Return the quasi-momentum as a float array once it is known to be admissible."""
    a = np.asarray(alpha, dtype=np.float64).ravel()
    if a.size != 3:
        raise ValueError("alpha must hold exactly three components")
    if not np.all(np.isfinite(a)):
        raise ValueError("every component of alpha must be finite")
    if not np.any(a != 0.0):
        raise ValueError("alpha must not be the zero vector")
    return a


def _oracle_assemble_bloch_exterior(
    elements: np.ndarray,
    surface_nodes: np.ndarray,
    free_nodes: np.ndarray,
    element_stiffness: np.ndarray,
    corner_signs: np.ndarray,
    n_side: int,
    alpha: tuple,
) -> dict:
    """Reference implementation."""
    el = np.asarray(elements)
    if el.ndim != 2 or el.shape[1] != 3 or not np.issubdtype(el.dtype, np.integer):
        raise ValueError("elements must be an integer array of three columns")
    Ke = np.asarray(element_stiffness, dtype=np.float64)
    if Ke.shape != (24, 24):
        raise ValueError("element_stiffness must be twenty-four by twenty-four")
    signs = np.asarray(corner_signs)
    if signs.shape != (8, 3):
        raise ValueError("corner_signs must be eight by three")
    if isinstance(n_side, bool) or not isinstance(n_side, (int, np.integer)) or int(n_side) < 4:
        raise ValueError("n_side must be an integer of four or more")
    n_side = int(n_side)
    a = _checked_alpha(alpha)

    offs = signs.astype(np.int64)
    reach = el[:, None, :] + offs[None, :, :]                 # (ne, 8, 3)
    wrap = reach // n_side
    node = ((reach[..., 0] % n_side) * n_side * n_side
            + (reach[..., 1] % n_side) * n_side
            + (reach[..., 2] % n_side))
    phase = np.exp(1j * (wrap @ a))                           # (ne, 8)

    dof = (3 * node[:, :, None] + np.arange(3)[None, None, :]).reshape(len(el), 24)
    ph24 = np.repeat(phase, 3, axis=1)
    factor = np.conj(ph24)[:, :, None] * ph24[:, None, :]
    vals = (Ke[None, :, :] * factor).ravel()
    rows = np.repeat(dof, 24, axis=1).ravel()
    cols = np.tile(dof, (1, 24)).ravel()
    ndof = 3 * n_side ** 3
    K = sp.coo_matrix((vals, (rows, cols)), shape=(ndof, ndof)).tocsr()

    con = np.asarray(surface_nodes)
    fre = np.asarray(free_nodes)
    for name, arr in (("surface_nodes", con), ("free_nodes", fre)):
        if arr.ndim != 1 or arr.size == 0 or not np.issubdtype(arr.dtype, np.integer):
            raise ValueError("%s must be a non-empty one-dimensional integer array" % name)
        if arr.min() < 0 or arr.max() >= n_side ** 3:
            raise ValueError("%s holds an index outside the cell" % name)
    con = con.astype(np.int64)
    fre = fre.astype(np.int64)
    cdof = np.sort(np.concatenate([3 * con + d for d in range(3)]))
    fdof = np.sort(np.concatenate([3 * fre + d for d in range(3)]))
    return {
        "K_ff": K[fdof][:, fdof].tocsr(),
        "K_fc": K[fdof][:, cdof].tocsr(),
        "K_cc": K[cdof][:, cdof].tocsr(),
        "free_dofs": fdof,
        "surface_dofs": cdof,
    }

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": r"""import copy as _input_copy
def _invoke_with_fresh_inputs(fn, *args, **kwargs):
    # deep-copied arguments keep candidate, reference and repeated calls independent
    args, kwargs = _input_copy.deepcopy((args, kwargs))
    return fn(*args, **kwargs)

import numpy as np
def build(n_side, spans, alpha, lam=1.5e6, mu=5.0e5, L=0.02):
    span = np.array(spans, dtype=np.int64); lo = (n_side - span) // 2; hi = lo + span
    g = np.arange(n_side, dtype=np.int64)
    ii, jj, kk = np.meshgrid(g, g, g, indexing="ij")
    cells = np.stack([ii.ravel(), jj.ravel(), kk.ravel()], 1)
    el = cells[~np.all((cells >= lo) & (cells < hi), axis=1)]
    rs = [np.arange(lo[d], hi[d] + 1, dtype=np.int64) for d in range(3)]
    aa, bb, cc = np.meshgrid(rs[0], rs[1], rs[2], indexing="ij")
    pts = np.stack([aa.ravel(), bb.ravel(), cc.ravel()], 1)
    on = np.any((pts == lo) | (pts == hi), axis=1)
    flat = lambda p: (p[..., 0] % n_side) * n_side * n_side + (p[..., 1] % n_side) * n_side + p[..., 2] % n_side
    surf = np.unique(flat(pts[on]))
    offs = np.array([(x, y, z) for x in (0, 1) for y in (0, 1) for z in (0, 1)], dtype=np.int64)
    touched = np.zeros(n_side ** 3, bool); touched[flat(el[:, None, :] + offs[None, :, :]).ravel()] = True
    every = np.arange(n_side ** 3, dtype=np.int64)
    free = every[touched & ~np.isin(every, surf)]
    h = L / n_side
    s = offs.astype(float) * 2.0 - 1.0
    C = np.zeros((6, 6)); C[:3, :3] = lam
    C[0, 0] = C[1, 1] = C[2, 2] = lam + 2.0 * mu; C[3, 3] = C[4, 4] = C[5, 5] = mu
    q = 1.0 / np.sqrt(3.0); Ke = np.zeros((24, 24))
    for xi in (-q, q):
        for et in (-q, q):
            for ze in (-q, q):
                dN = np.empty((8, 3))
                dN[:, 0] = 0.125 * s[:, 0] * (1 + s[:, 1] * et) * (1 + s[:, 2] * ze)
                dN[:, 1] = 0.125 * (1 + s[:, 0] * xi) * s[:, 1] * (1 + s[:, 2] * ze)
                dN[:, 2] = 0.125 * (1 + s[:, 0] * xi) * (1 + s[:, 1] * et) * s[:, 2]
                gg = dN * (2.0 / h); B = np.zeros((6, 24))
                B[0, 0::3] = gg[:, 0]; B[1, 1::3] = gg[:, 1]; B[2, 2::3] = gg[:, 2]
                B[3, 1::3] = gg[:, 2]; B[3, 2::3] = gg[:, 1]
                B[4, 0::3] = gg[:, 2]; B[4, 2::3] = gg[:, 0]
                B[5, 0::3] = gg[:, 1]; B[5, 1::3] = gg[:, 0]
                Ke += B.T @ C @ B * (h / 2.0) ** 3
    return el, surf, free, 0.5 * (Ke + Ke.T), offs, n_side, alpha
def digest(out):
    A = out["K_ff"]; B = out["K_fc"]; C = out["K_cc"]
    herm = float(abs(A - A.getH()).max()) if A.nnz else 0.0
    d = A.diagonal()
    # the entrywise magnitudes and the diagonal are phase blind, so the digest must read the
    # real and imaginary parts: a matrix assembled with no phases at all agrees with the correct
    # one on shape, on the diagonal and on abs(K_fc), and differs only here
    return (A.shape, B.shape, C.shape, round(herm, 6),
            round(float(np.abs(d.imag).max()), 10), int(d.real.min() > 0),
            round(float(d.real.sum()), 4), round(float(abs(B).sum()), 4),
            round(float(A.real.sum()), 4), round(float(np.abs(A.imag).sum()), 4),
            # all three blocks, so that scaling one of them on its own is visible
            round(float(C.diagonal().real.sum()), 4), round(float(abs(C).sum()), 4),
            round(float(abs(B).max()), 6), round(float(abs(C).max()), 6),
            int(out["free_dofs"][0]), int(out["surface_dofs"][-1]))
def flat(x):
    # flatten to a tuple of plain numeric terminals, which is what AutoQC compares
    if isinstance(x, (tuple, list)):
        return tuple(v for e in x for v in flat(e))
    if hasattr(x, "tolist"):
        return flat(x.tolist())
    return (int(x),) if isinstance(x, bool) else (x,)
""",
            'call': 'flat(digest(_invoke_with_fresh_inputs(assemble_bloch_exterior, *build(10, (4, 3, 2), (np.pi, 0.4, 0.2)))))',
            'gold_call': 'flat(digest(_invoke_with_fresh_inputs(_oracle_assemble_bloch_exterior, *build(10, (4, 3, 2), (np.pi, 0.4, 0.2)))))',
        },
        {
            "setup": r"""import copy as _input_copy
def _invoke_with_fresh_inputs(fn, *args, **kwargs):
    # deep-copied arguments keep candidate, reference and repeated calls independent
    args, kwargs = _input_copy.deepcopy((args, kwargs))
    return fn(*args, **kwargs)

import numpy as np
def mini(alpha):
    n_side = 6; spans = (2, 2, 2)
    span = np.array(spans, dtype=np.int64); lo = (n_side - span) // 2; hi = lo + span
    g = np.arange(n_side, dtype=np.int64)
    ii, jj, kk = np.meshgrid(g, g, g, indexing="ij")
    cells = np.stack([ii.ravel(), jj.ravel(), kk.ravel()], 1)
    el = cells[~np.all((cells >= lo) & (cells < hi), axis=1)]
    rs = [np.arange(lo[d], hi[d] + 1, dtype=np.int64) for d in range(3)]
    aa, bb, cc = np.meshgrid(rs[0], rs[1], rs[2], indexing="ij")
    pts = np.stack([aa.ravel(), bb.ravel(), cc.ravel()], 1)
    on = np.any((pts == lo) | (pts == hi), axis=1)
    flat = lambda p: (p[..., 0] % n_side) * n_side * n_side + (p[..., 1] % n_side) * n_side + p[..., 2] % n_side
    surf = np.unique(flat(pts[on]))
    offs = np.array([(x, y, z) for x in (0, 1) for y in (0, 1) for z in (0, 1)], dtype=np.int64)
    touched = np.zeros(n_side ** 3, bool); touched[flat(el[:, None, :] + offs[None, :, :]).ravel()] = True
    every = np.arange(n_side ** 3, dtype=np.int64)
    free = every[touched & ~np.isin(every, surf)]
    Ke = np.eye(24)
    return el, surf, free, Ke, offs, n_side, alpha
def probe(fn, alpha):
    out = _invoke_with_fresh_inputs(fn, *mini(alpha)); A = out["K_ff"]
    # a flag, not the value: the defect sits at roundoff and moves with the accumulation order,
    # so pinning its decimals would reject a conformant assembly
    return (A.shape[0], int(float(abs(A - A.getH()).max()) <= 1e-10 * max(float(abs(A).max()), 1.0)),
            round(float(A.diagonal().real.sum()), 6),
            round(float(A.real.sum()), 6), round(float(np.abs(A.imag).sum()), 6))
def flat(x):
    # flatten to a tuple of plain numeric terminals, which is what AutoQC compares
    if isinstance(x, (tuple, list)):
        return tuple(v for e in x for v in flat(e))
    if hasattr(x, "tolist"):
        return flat(x.tolist())
    return (int(x),) if isinstance(x, bool) else (x,)
""",
            'call': 'flat((probe(assemble_bloch_exterior, (np.pi, 0.0, 0.0)), probe(assemble_bloch_exterior, (0.3, -1.1, 2.2))))',
            'gold_call': 'flat((probe(_oracle_assemble_bloch_exterior, (np.pi, 0.0, 0.0)), probe(_oracle_assemble_bloch_exterior, (0.3, -1.1, 2.2))))',
        },
        {
            "setup": r"""import copy as _input_copy
def _invoke_with_fresh_inputs(fn, *args, **kwargs):
    # deep-copied arguments keep candidate, reference and repeated calls independent
    args, kwargs = _input_copy.deepcopy((args, kwargs))
    return fn(*args, **kwargs)

import numpy as np
EL = np.array([[0, 0, 0]], dtype=np.int64)
SURF = np.array([0], dtype=np.int64)
FREE = np.array([1], dtype=np.int64)
KE = np.eye(24)
OFF = np.array([(x, y, z) for x in (0, 1) for y in (0, 1) for z in (0, 1)], dtype=np.int64)
def verdict(fn, el=EL, ke=KE, off=OFF, n=6, al=(1.0, 0.0, 0.0), surf=SURF):
    try:
        _invoke_with_fresh_inputs(fn, el, surf, FREE, ke, off, n, al)
    except ValueError:
        return 1
    except Exception:
        return 2
    return 0
def flat(x):
    # flatten to a tuple of plain numeric terminals, which is what AutoQC compares
    if isinstance(x, (tuple, list)):
        return tuple(v for e in x for v in flat(e))
    if hasattr(x, "tolist"):
        return flat(x.tolist())
    return (int(x),) if isinstance(x, bool) else (x,)
def verdicts(fn):
    return flat((verdict(fn, al=(0.0, 0.0, 0.0)), verdict(fn, al=(1.0, 2.0)), verdict(fn, al=(1.0, float('nan'), 0.0)), verdict(fn, ke=np.eye(12)), verdict(fn, off=OFF[:4]), verdict(fn, n=2), verdict(fn, el=EL.astype(float)), verdict(fn, surf=np.array([[0, 1], [2, 3]])), verdict(fn, surf=np.array([10 ** 6])), verdict(fn, n=True), verdict(fn)))
""",
            'call': 'verdicts(assemble_bloch_exterior)',
            'gold_call': 'verdicts(_oracle_assemble_bloch_exterior)',
        },
    ]
