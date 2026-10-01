"""
The mass matrix is the consistent one: the element mass is the density times the integral over the element of the product of the trilinear shape functions, evaluated with the same two-point Gauss rule in each direction as the stiffness, which is exact for that integrand. Assemble it with the same degree of freedom ordering as the stiffness. Every element of the grid carries both a stiffness and a mass, the interior ones scaled as above.

Both matrices are assembled with the same quasi-periodic rule as the exterior stiffness: a corner that wraps across a cell face carries the Bloch phase of the wrap, conjugated on the outgoing side and plain on the incoming, so that both matrices are Hermitian. When every Bloch phase is real the assembled matrices are real and may be kept so. The resonator is placed as the partition placed it, its low corner at element index n_side minus span, integer-divided by two, in each direction.

Returns
-------
dict, holding the sparse Hermitian matrices K_full and M_full over all three n_side cubed degrees of freedom, the count n_inside of elements inside the resonator, and total_mass, the mass of the whole cell.

Assemble the full two-phase Bloch pencil of the cell at the stated finite contrast, both stiffness and mass, over every element of the grid.

Everything before this stage works in the leading-order limit of small contrast, where the resonator moves as a rigid body and only the background is discretised. This stage keeps the contrast finite. The resonator is a second elastic phase in its own right: inside it the Lame parameters are those of the background divided by delta and the density is that of the background divided by eps, exactly as the configuration states, and it is discretised on the same grid as the background with the same elements. The stiffness of an element is linear in its two Lame parameters, so the interior element stiffness is the exterior one divided by delta; nothing else about the element changes.

Returns
-------
dict, holding the sparse Hermitian matrices K_full and M_full over all three n_side cubed degrees of freedom, the count n_inside of elements inside the resonator, and total_mass, the mass of the whole cell.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def assemble_full_bloch_pencil(
    n_side: int,
    spans: tuple,
    element_stiffness: np.ndarray,
    corner_signs: np.ndarray,
    h: float,
    rho: float,
    delta: float,
    eps: float,
    alpha: tuple,
) -> dict:
    """Assemble the finite-contrast two-phase stiffness and consistent mass of the whole cell at one quasi-momentum.

    Parameters
    ----------
    n_side : int
        Elements along each edge of the cell, at least four.
    spans : tuple
        Three integers giving the elements spanned by the resonator along each axis.
    element_stiffness : np.ndarray
        The (24, 24) background element stiffness in newton per metre.
    corner_signs : np.ndarray
        The (8, 3) binary corner offsets, in the prescribed order.
    h : float
        Element edge in metre, above zero.
    rho : float
        Background density in kilogram per cubic metre, above zero.
    delta : float
        Reciprocal stiffness contrast, above zero.
    eps : float
        Reciprocal density contrast, above zero.
    alpha : tuple
        The quasi-momentum, three finite values, dimensionless.

    Returns
    -------
    dict
        Under the keys K_full, M_full, n_inside and total_mass.
        K_full and M_full are scipy sparse matrices of shape (3 n_side cubed, 3 n_side cubed),
        Hermitian, with degree of freedom index equal to three times the flat node index plus
        the component, the flat node index being (i times n_side plus j) times n_side plus k
        after periodic wrapping. M_full is the consistent mass. n_inside is the number of
        elements inside the resonator. total_mass is rho times the exterior volume plus rho
        over eps times the resonator volume, in kilogram.

    Raises
    ------
    ValueError
        When n_side is not an integer of four or more, when spans does not hold three integers each at least one and at most n_side minus two, when element_stiffness is not (24, 24) or corner_signs is not (8, 3), when h, rho, delta or eps is not finite and above zero, or when alpha does not hold exactly three finite values.
    """
    return

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
import scipy.sparse as sp


def _hex_consistent_mass(h, offs):
    """Element mass for unit density on a cube of edge h, two-point Gauss in each direction."""
    q = 1.0 / np.sqrt(3.0)
    s = offs.astype(np.float64) * 2.0 - 1.0
    Me = np.zeros((24, 24))
    for xi in (-q, q):
        for et in (-q, q):
            for ze in (-q, q):
                Nn = 0.125 * (1.0 + s[:, 0] * xi) * (1.0 + s[:, 1] * et) * (1.0 + s[:, 2] * ze)
                Nm = np.zeros((3, 24))
                for c in range(3):
                    Nm[c, c::3] = Nn
                Me += (Nm.T @ Nm) * (h ** 3 / 8.0)
    return Me


def _bloch_assemble(cells, scale, E24, offs, n_side, alpha, real):
    """Quasi-periodic assembly of one 24 by 24 element matrix over the listed elements."""
    a = np.asarray(alpha, dtype=np.float64)
    reach = cells[:, None, :] + offs[None, :, :]
    wrap = reach // n_side
    node = ((reach[..., 0] % n_side) * n_side * n_side
            + (reach[..., 1] % n_side) * n_side
            + (reach[..., 2] % n_side))
    phase = np.exp(1j * (wrap @ a))
    if real:
        phase = phase.real
    dof = (3 * node[:, :, None] + np.arange(3)[None, None, :]).reshape(len(cells), 24)
    ph24 = np.repeat(phase, 3, axis=1)
    factor = np.conj(ph24)[:, :, None] * ph24[:, None, :] * scale[:, None, None]
    vals = (E24[None, :, :] * factor).ravel()
    rows = np.repeat(dof, 24, axis=1).ravel()
    cols = np.tile(dof, (1, 24)).ravel()
    n = 3 * n_side ** 3
    A = sp.coo_matrix((vals, (rows, cols)), shape=(n, n)).tocsc()
    return 0.5 * (A + A.conj().T)


def _oracle_assemble_full_bloch_pencil(
    n_side: int,
    spans: tuple,
    element_stiffness: np.ndarray,
    corner_signs: np.ndarray,
    h: float,
    rho: float,
    delta: float,
    eps: float,
    alpha: tuple,
) -> dict:
    """Reference implementation."""
    if isinstance(n_side, bool) or not isinstance(n_side, (int, np.integer)) or int(n_side) < 4:
        raise ValueError("n_side must be an integer of four or more")
    n_side = int(n_side)
    try:
        span = tuple(spans)
    except TypeError:
        raise ValueError("spans must hold exactly three integers")
    if len(span) != 3 or any(isinstance(v, bool) or not isinstance(v, (int, np.integer)) for v in span):
        raise ValueError("spans must hold exactly three integers")
    span = np.array([int(v) for v in span], dtype=np.int64)
    if np.any(span < 1) or np.any(span > n_side - 2):
        raise ValueError("spans must lie in [1, n_side - 2]")
    Ke = np.asarray(element_stiffness, dtype=np.float64)
    if Ke.shape != (24, 24):
        raise ValueError("element_stiffness must be twenty-four by twenty-four")
    offs = np.asarray(corner_signs)
    if offs.shape != (8, 3):
        raise ValueError("corner_signs must be eight by three")
    offs = offs.astype(np.int64)
    for name, v in (("h", h), ("rho", rho), ("delta", delta), ("eps", eps)):
        v = float(v)
        if not np.isfinite(v) or v <= 0.0:
            raise ValueError("%s must be finite and above zero" % name)
    h = float(h); rho = float(rho); delta = float(delta); eps = float(eps)
    try:
        a = np.asarray(alpha, dtype=np.float64).ravel()
    except (TypeError, ValueError):
        raise ValueError("alpha must hold exactly three finite values")
    if a.size != 3 or not np.all(np.isfinite(a)):
        raise ValueError("alpha must hold exactly three finite values")

    g = np.arange(n_side, dtype=np.int64)
    ii, jj, kk = np.meshgrid(g, g, g, indexing="ij")
    cells = np.stack([ii.ravel(), jj.ravel(), kk.ravel()], 1)
    lo = (n_side - span) // 2
    hi = lo + span
    inside = np.all((cells >= lo) & (cells < hi), axis=1)
    real = bool(np.allclose(np.sin(a), 0.0, atol=1e-12))
    kscale = np.where(inside, 1.0 / delta, 1.0)
    mscale = np.where(inside, rho / eps, rho)
    K = _bloch_assemble(cells, kscale, Ke, offs, n_side, a, real)
    M = _bloch_assemble(cells, mscale, _hex_consistent_mass(h, offs), offs, n_side, a, real)
    n_in = int(inside.sum())
    total_mass = rho * (len(cells) - n_in) * h ** 3 + (rho / eps) * n_in * h ** 3
    return {"K_full": K, "M_full": M, "n_inside": n_in, "total_mass": float(total_mass)}

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

import numpy as np, scipy.sparse as sp
def _elem(lam, mu, h):
    # one trilinear brick of edge h: eight corner offsets and the exact two-point Gauss stiffness
    offs = np.array([(x, y, z) for x in (0, 1) for y in (0, 1) for z in (0, 1)], dtype=np.int64)
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
                gg = dN * (2.0 / h); Bm = np.zeros((6, 24))
                Bm[0, 0::3] = gg[:, 0]; Bm[1, 1::3] = gg[:, 1]; Bm[2, 2::3] = gg[:, 2]
                Bm[3, 1::3] = gg[:, 2]; Bm[3, 2::3] = gg[:, 1]
                Bm[4, 0::3] = gg[:, 2]; Bm[4, 2::3] = gg[:, 0]
                Bm[5, 0::3] = gg[:, 1]; Bm[5, 1::3] = gg[:, 0]
                Ke += Bm.T @ C @ Bm * (h / 2.0) ** 3
    return {"element_stiffness": 0.5 * (Ke + Ke.T), "corner_signs": offs}
# a small cell at a quasi-momentum whose phases are all real
KE = _elem(1.5e6, 5.0e5, 0.002)
def digest(out):
    K = out["K_full"]; M = out["M_full"]
    return (tuple(int(n) for n in K.shape), tuple(int(n) for n in M.shape),
            int(K.nnz > 0), int(M.nnz > 0),
            # Hermitian to rounding, and with no imaginary part beyond rounding when every
            # phase is real, whichever dtype the candidate chose to carry
            int(bool(abs(K - K.conj().T).max() <= 1e-12 * abs(K).max())),
            int(bool(abs(M - M.conj().T).max() <= 1e-12 * abs(M).max())),
            int(bool((not np.iscomplexobj(K.data)) or abs(K.data.imag).max() <= 1e-12 * abs(K).max())),
            round(float(K.diagonal().real.sum()), 3), round(float(M.diagonal().real.sum()), 12),
            int(out["n_inside"]), round(float(out["total_mass"]), 12))
def flat(x):
    # flatten to a tuple of plain numeric terminals, which is what AutoQC compares
    if isinstance(x, (tuple, list)):
        return tuple(v for e in x for v in flat(e))
    if hasattr(x, "tolist"):
        return flat(x.tolist())
    return (int(x),) if isinstance(x, bool) else (x,)
""",
            "call": "flat(digest(_invoke_with_fresh_inputs(assemble_full_bloch_pencil, 10, (4, 3, 2), KE['element_stiffness'], KE['corner_signs'], 0.002, 1200.0, 1.0e-2, 5.0e-3, (np.pi, 0.0, 0.0))))",
            "gold_call": "flat(digest(_invoke_with_fresh_inputs(_oracle_assemble_full_bloch_pencil, 10, (4, 3, 2), KE['element_stiffness'], KE['corner_signs'], 0.002, 1200.0, 1.0e-2, 5.0e-3, (np.pi, 0.0, 0.0))))",
        },
        {
            "setup": r"""import copy as _input_copy
def _invoke_with_fresh_inputs(fn, *args, **kwargs):
    # deep-copied arguments keep candidate, reference and repeated calls independent
    args, kwargs = _input_copy.deepcopy((args, kwargs))
    return fn(*args, **kwargs)

import numpy as np, scipy.sparse as sp
def _elem(lam, mu, h):
    # one trilinear brick of edge h: eight corner offsets and the exact two-point Gauss stiffness
    offs = np.array([(x, y, z) for x in (0, 1) for y in (0, 1) for z in (0, 1)], dtype=np.int64)
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
                gg = dN * (2.0 / h); Bm = np.zeros((6, 24))
                Bm[0, 0::3] = gg[:, 0]; Bm[1, 1::3] = gg[:, 1]; Bm[2, 2::3] = gg[:, 2]
                Bm[3, 1::3] = gg[:, 2]; Bm[3, 2::3] = gg[:, 1]
                Bm[4, 0::3] = gg[:, 2]; Bm[4, 2::3] = gg[:, 0]
                Bm[5, 0::3] = gg[:, 1]; Bm[5, 1::3] = gg[:, 0]
                Ke += Bm.T @ C @ Bm * (h / 2.0) ** 3
    return {"element_stiffness": 0.5 * (Ke + Ke.T), "corner_signs": offs}
# a generic quasi-momentum, so the phases are complex, and a unit background at unit contrast
KE = _elem(1.0, 0.5, 0.125)
def digest(out):
    K = out["K_full"]; M = out["M_full"]
    return (tuple(int(n) for n in K.shape),
            int(bool(abs(K - K.conj().T).max() <= 1e-12 * abs(K).max())),
            int(bool(abs(M - M.conj().T).max() <= 1e-12 * abs(M).max())),
            # genuinely complex here: the phases are not all real
            int(bool(np.iscomplexobj(K.data) and abs(K.data.imag).max() > 1e-12 * abs(K).max())),
            round(float(K.diagonal().real.sum()), 6), round(float(M.diagonal().real.sum()), 12),
            round(float(abs(K.data).max()), 6),
            int(out["n_inside"]), round(float(out["total_mass"]), 12))
def flat(x):
    # flatten to a tuple of plain numeric terminals, which is what AutoQC compares
    if isinstance(x, (tuple, list)):
        return tuple(v for e in x for v in flat(e))
    if hasattr(x, "tolist"):
        return flat(x.tolist())
    return (int(x),) if isinstance(x, bool) else (x,)
""",
            "call": "flat((digest(_invoke_with_fresh_inputs(assemble_full_bloch_pencil, 8, (2, 4, 4), KE['element_stiffness'], KE['corner_signs'], 0.125, 1.0, 1.0, 1.0, (0.3, -1.1, 2.2))), digest(_invoke_with_fresh_inputs(assemble_full_bloch_pencil, 8, (3, 3, 3), KE['element_stiffness'], KE['corner_signs'], 0.125, 2.0, 0.5, 0.25, (0.0, 0.0, 0.7)))))",
            "gold_call": "flat((digest(_invoke_with_fresh_inputs(_oracle_assemble_full_bloch_pencil, 8, (2, 4, 4), KE['element_stiffness'], KE['corner_signs'], 0.125, 1.0, 1.0, 1.0, (0.3, -1.1, 2.2))), digest(_invoke_with_fresh_inputs(_oracle_assemble_full_bloch_pencil, 8, (3, 3, 3), KE['element_stiffness'], KE['corner_signs'], 0.125, 2.0, 0.5, 0.25, (0.0, 0.0, 0.7)))))",
        },
        {
            "setup": r"""import copy as _input_copy
def _invoke_with_fresh_inputs(fn, *args, **kwargs):
    # deep-copied arguments keep candidate, reference and repeated calls independent
    args, kwargs = _input_copy.deepcopy((args, kwargs))
    return fn(*args, **kwargs)

import numpy as np, scipy.sparse as sp
def _elem(lam, mu, h):
    # one trilinear brick of edge h: eight corner offsets and the exact two-point Gauss stiffness
    offs = np.array([(x, y, z) for x in (0, 1) for y in (0, 1) for z in (0, 1)], dtype=np.int64)
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
                gg = dN * (2.0 / h); Bm = np.zeros((6, 24))
                Bm[0, 0::3] = gg[:, 0]; Bm[1, 1::3] = gg[:, 1]; Bm[2, 2::3] = gg[:, 2]
                Bm[3, 1::3] = gg[:, 2]; Bm[3, 2::3] = gg[:, 1]
                Bm[4, 0::3] = gg[:, 2]; Bm[4, 2::3] = gg[:, 0]
                Bm[5, 0::3] = gg[:, 1]; Bm[5, 1::3] = gg[:, 0]
                Ke += Bm.T @ C @ Bm * (h / 2.0) ** 3
    return {"element_stiffness": 0.5 * (Ke + Ke.T), "corner_signs": offs}
KE = _elem(1.0, 0.5, 0.125)
KS, CS = KE["element_stiffness"], KE["corner_signs"]
def verdict(fn, n=8, sp_=(2, 2, 2), ks=None, cs=None, h=0.125, r=1.0, d=1.0, e=1.0, al=(0.3, 0.2, 0.1)):
    try:
        _invoke_with_fresh_inputs(fn, n, sp_, KS if ks is None else ks, CS if cs is None else cs, h, r, d, e, al)
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
    return flat((verdict(fn, n=3), verdict(fn, sp_=(2, 2)), verdict(fn, sp_=(2, 7, 2)), verdict(fn, ks=np.eye(6)), verdict(fn, cs=np.zeros((4, 3), dtype=np.int64)), verdict(fn, h=0.0), verdict(fn, r=-1.0), verdict(fn, d=0.0), verdict(fn, e=float('nan')), verdict(fn, al=(0.1, 0.2)), verdict(fn)))
""",
            'call': 'verdicts(assemble_full_bloch_pencil)',
            'gold_call': 'verdicts(_oracle_assemble_full_bloch_pencil)',
        },
    ]
