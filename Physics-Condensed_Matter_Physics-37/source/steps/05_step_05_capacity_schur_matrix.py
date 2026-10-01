"""
What the map has to be tested against is the second thing this stage needs. In the static limit the interior problem is a Lame system with a traction-free surface, whose solutions are the six rigid motions of the resonator, three translations and three rotations. All six are retained here. The source's own reduction keeps only the three constant fields; that restriction is not a consequence of the quasi-periodicity, since a centred rotation multiplied by a cutoff that vanishes before the cell faces is an admissible quasi-periodic field with zero strain and zero traction on the resonator surface, and dropping it is what this stage undoes. The matrix is therefore six by six, built by testing the map against all six motions:

$$Q[i, j] = - integral over the resonator surface of (map applied to g_i) . g_j.$$

Collect the six motions as the columns of a matrix E and form Q as E conjugate-transposed times S times E. Column c for c in zero to two is the constant field e_c, one in component c at every surface node and zero elsewhere. Column three plus a is the rotation about axis a through the point reference_point supplied to this stage: at a surface node of position x, it is the vector e_a cross (x - reference_point), taken component by component. The minus sign in the definition is already accounted for: the outward normal of the region outside the resonator points into the resonator on that surface, so the condensed form returns the positive quantity directly, and Q comes out as the elastic energy stored outside by the field that equals g_i on the surface.

The first three columns are dimensionless and the last three carry metre, so the six diagonal entries of Q do not share a unit and its six eigenvalues are not a spectrum. Converting them into frequencies is the next stage's problem.

Two properties follow and both must be checked, because they are the only independent confirmation available that the assembly and the phases are right. Q is Hermitian, and Q is positive definite. Reject a matrix whose departure from Hermitian symmetry is large relative to its own diagonal, entry by entry rather than against the largest entry alone, since the two blocks differ by six orders of magnitude and an absolute measure would not see an asymmetric rotational block; reject rather than symmetrising it away, since symmetrising unconditionally conceals exactly the assembly error the check exists to catch; then symmetrise the accepted matrix so that the eigenvalues the next stage takes are real by construction rather than by luck.

Solve with the conjugate gradient method preconditioned by the diagonal, which is available because K_ff is Hermitian positive definite, iterating until the residual falls to rel_tol times the norm of the right-hand side, and verifying that against a freshly evaluated residual with an order of magnitude of slack, since at a tolerance near the roundoff floor the two differ by a few per cent. Accumulate every inner product as a plain summation over the elementwise product rather than through a threaded dot product, so the result does not shift when the linear algebra library changes how many threads it uses. The iteration counts are part of what this stage returns and must be the counts actually taken, so a direct factorisation is not a conformant implementation of this step even though it would reach the same residual; the prompt's tolerance wording governs the pipeline, this docstring governs this step.

Returns
-------
dict, holding the 6 by 6 capacity, its eigenvalues, the hermitian_defect and the six iteration counts.

This is the stage that carries the physics. The quasi-periodic Dirichlet-to-Neumann map sends a displacement prescribed on the resonator surface to the traction the background outside exerts in response: solve the static Lame system in the cell outside the resonator with that displacement as Dirichlet data and quasi-periodic conditions on the cell faces, then read the traction off the surface. Condensing the assembled exterior stiffness onto the surface degrees of freedom realises exactly that map, because eliminating the free degrees of freedom is the discrete form of solving the exterior problem. The Schur complement

$$S = K_cc - K_cf * inverse(K_ff) * K_fc$$

is therefore the discrete Dirichlet-to-Neumann map, with K_cf the conjugate transpose of K_fc.

Returns
-------
dict, holding the 6 by 6 capacity, its eigenvalues, the hermitian_defect and the six iteration counts.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def capacity_schur_matrix(
    K_ff: "scipy.sparse.spmatrix",
    K_fc: "scipy.sparse.spmatrix",
    K_cc: "scipy.sparse.spmatrix",
    surface_dofs: np.ndarray,
    n_side: int,
    h: float,
    reference_point: np.ndarray,
    rel_tol: float,
) -> dict:
    """Condense the exterior stiffness onto the resonator surface and test it against the six rigid motions of the resonator.

    Parameters
    ----------
    K_ff : scipy.sparse.spmatrix
        Free to free block, Hermitian positive definite.
    K_fc : scipy.sparse.spmatrix
        Free to surface block.
    K_cc : scipy.sparse.spmatrix
        Surface to surface block.
    surface_dofs : np.ndarray
        Degree of freedom indices of the surface block.
    n_side : int
        Elements along each edge of the cell, used to place the surface nodes.
    h : float
        Element edge in metre.
    reference_point : np.ndarray
        The point about which the rigid rotations are taken, in metre, shape (3,).
    rel_tol : float
        Relative residual at which the iteration stops.

    Returns
    -------
    dict
        Under the keys capacity, eigenvalues, hermitian_defect and iterations.
        hermitian_defect is the departure from Hermitian symmetry of the UNSYMMETRISED
        condensed matrix, scaled entrywise by the square roots of the corresponding diagonal
        entries, so it is dimensionless and treats the translation and rotation blocks alike;
        an unscaled absolute defect would be blind to a gross asymmetry in the rotational
        block, whose entries are of order one against 1e5 for the translational block. Reject
        when it exceeds 1e-6, and symmetrise only after that check has passed.
        capacity is the (6, 6) Hermitian positive definite matrix and eigenvalues
        its six eigenvalues ascending.
        iterations is a length-six sequence of actual diagonal-preconditioned
        conjugate-gradient iteration counts, ordered by the six boundary fields
        below. Obtain these counts from the six right-hand sides in
        K_ff X = -K_fc E.
        The six columns of E are the six rigid motions of the resonator sampled at the
        surface degrees of freedom, in the order: the three unit Cartesian translations,
        then the three unit rotations about the axes through reference_point. Rotations
        carry radian, so the first three columns are dimensionless and the last three
        carry metre.

    Raises
    ------
    ValueError
        When the three blocks do not have consistent shapes, when surface_dofs does not match the surface block, when n_side is not an integer of four or more, when h is not finite and above zero, when reference_point does not hold exactly three finite values, when rel_tol is not finite and above zero, when the iteration fails to reach rel_tol, when the condensed matrix is not Hermitian to working accuracy, or when the resulting matrix fails to be positive definite.
    """
    return

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
import scipy.sparse as sp


def _inner(a, b):
    """Hermitian inner product by plain summation, so the value does not depend on BLAS threading."""
    return np.sum(np.conj(a) * b)


def _pcg(A, B, rel_tol, max_iter=200000):
    """Jacobi-preconditioned conjugate gradient for a Hermitian positive definite A, column by column."""
    tol = rel_tol
    diag = A.diagonal().real
    if not np.all(np.isfinite(diag)):
        raise ValueError("K_ff has a diagonal entry that is not finite")
    if np.any(diag <= 0.0):
        raise ValueError("K_ff has a non-positive diagonal entry")
    prec = 1.0 / diag
    out = np.zeros_like(B)
    counts = []
    for col in range(B.shape[1]):
        rhs = B[:, col]
        x = np.zeros_like(rhs)
        r = rhs - A @ x
        z = prec * r
        p = z.copy()
        rz = _inner(r, z)
        target = rel_tol * np.sqrt(_inner(rhs, rhs).real)
        steps = 0
        while steps < max_iter:
            if np.sqrt(_inner(r, r).real) <= target:
                break
            Ap = A @ p
            curvature = _inner(p, Ap)
            if curvature == 0.0 or rz == 0.0:
                break
            step = rz / curvature
            x = x + step * p
            r = r - step * Ap
            z = prec * r
            rz_next = _inner(r, z)
            p = z + (rz_next / rz) * p
            rz = rz_next
            steps += 1
        true_residual = np.sqrt(_inner(rhs - A @ x, rhs - A @ x).real)
        if true_residual > 10.0 * target:
            raise ValueError(
                "conjugate gradient did not reach rel_tol: residual %.3e against %.3e" % (true_residual / max(np.sqrt(_inner(rhs, rhs).real), 1e-300), tol))
        out[:, col] = x
        counts.append(steps)
    return out, counts


def _rigid_boundary_fields(cdof, n_side, h, reference_point):
    """The six rigid motions of the resonator sampled at the surface degrees of freedom.

    Columns 0 to 2 are the unit Cartesian translations; column 3 + a is the rotation about the
    axis a through the centroid, holding the relevant component of e_a cross r.
    """
    node = cdof // 3
    comp = cdof % 3
    i1 = node // (n_side * n_side)
    i2 = (node // n_side) % n_side
    i3 = node % n_side
    r = (np.stack([i1, i2, i3], 1).astype(np.float64) * h
         - np.asarray(reference_point, dtype=np.float64)[None, :])
    E = np.zeros((cdof.size, 6), dtype=np.complex128)
    for c in range(3):
        E[comp == c, c] = 1.0
    for a in range(3):
        for c in range(3):
            if a == c:
                continue
            b = 3 - a - c
            sign = 1.0 if (c, a, b) in ((0, 1, 2), (1, 2, 0), (2, 0, 1)) else -1.0
            sel = comp == c
            E[sel, 3 + a] = sign * r[sel, b]
    return E


def _oracle_capacity_schur_matrix(
    K_ff: "scipy.sparse.spmatrix",
    K_fc: "scipy.sparse.spmatrix",
    K_cc: "scipy.sparse.spmatrix",
    surface_dofs: np.ndarray,
    n_side: int,
    h: float,
    reference_point: np.ndarray,
    rel_tol: float,
) -> dict:
    """Reference implementation."""
    tol = float(rel_tol)
    if not np.isfinite(tol) or tol <= 0.0:
        raise ValueError("rel_tol must be finite and above zero")
    A = sp.csr_matrix(K_ff)
    Bfc = sp.csr_matrix(K_fc)
    Ccc = sp.csr_matrix(K_cc)
    if A.shape[0] != A.shape[1]:
        raise ValueError("K_ff must be square")
    if Bfc.shape[0] != A.shape[0]:
        raise ValueError("K_fc must have one row per free degree of freedom")
    if Ccc.shape[0] != Ccc.shape[1] or Ccc.shape[0] != Bfc.shape[1]:
        raise ValueError("K_cc must be square and match K_fc")
    cdof = np.asarray(surface_dofs)
    if cdof.ndim != 1 or not np.issubdtype(cdof.dtype, np.integer):
        raise ValueError("surface_dofs must be a one-dimensional integer array")
    cdof = cdof.astype(np.int64)
    if cdof.size != Ccc.shape[0]:
        raise ValueError("surface_dofs must match the surface block")

    if int(n_side) != n_side or int(n_side) < 4:
        raise ValueError("n_side must be an integer of four or more")
    hh = float(h)
    if not np.isfinite(hh) or hh <= 0.0:
        raise ValueError("h must be finite and above zero")
    ctr = np.asarray(reference_point, dtype=np.float64).ravel()
    if ctr.size != 3 or not np.all(np.isfinite(ctr)):
        raise ValueError("reference_point must hold exactly three finite values")

    E = _rigid_boundary_fields(cdof, int(n_side), hh, ctr)

    lifted, counts = _pcg(A, -(Bfc @ E), tol)
    raw = E.conj().T @ (Ccc @ E) + E.conj().T @ (Bfc.conj().T @ lifted)
    dg = np.sqrt(np.abs(np.diag(raw)))
    dg = np.where(dg > 0.0, dg, 1.0)
    defect = float((np.abs(raw - raw.conj().T) / np.outer(dg, dg)).max())
    if defect > 1.0e-6:
        raise ValueError(
            "condensed matrix not Hermitian: entrywise defect %.3e against 1.0e-06" % (defect,))
    Q = 0.5 * (raw + raw.conj().T)
    w = np.linalg.eigvalsh(Q)
    if not np.all(w > 0.0):
        raise ValueError("the capacity matrix must be positive definite")
    return {
        "capacity": Q,
        "eigenvalues": w,
        "hermitian_defect": defect,
        "iterations": counts,
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

import numpy as np, scipy.sparse as sp
def cell(n_side, spans, alpha, lam=1.5e6, mu=5.0e5, L=0.02):
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
    h = L / n_side; s = offs.astype(float) * 2.0 - 1.0
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
    Ke = 0.5 * (Ke + Ke.T)
    reach = el[:, None, :] + offs[None, :, :]; wrap = reach // n_side
    node = flat(reach); a = np.asarray(alpha, float); ph = np.exp(1j * (wrap @ a))
    dof = (3 * node[:, :, None] + np.arange(3)[None, None, :]).reshape(len(el), 24)
    p24 = np.repeat(ph, 3, axis=1); fac = np.conj(p24)[:, :, None] * p24[:, None, :]
    vals = (Ke[None, :, :] * fac).ravel()
    rows = np.repeat(dof, 24, axis=1).ravel(); cols = np.tile(dof, (1, 24)).ravel()
    K = sp.coo_matrix((vals, (rows, cols)), shape=(3 * n_side ** 3,) * 2).tocsr()
    cd = np.sort(np.concatenate([3 * surf + d for d in range(3)]))
    fd = np.sort(np.concatenate([3 * free + d for d in range(3)]))
    centre = (lo.astype(np.float64) + span / 2.0) * h
    return K[fd][:, cd], K[fd][:, fd].tocsr(), K[cd][:, cd].tocsr(), cd, n_side, h, centre
def digest(out):
    Q = out["capacity"]; w = out["eigenvalues"]
    # flags, not values: both defects sit at the roundoff floor of a matrix whose scale is about
    # 6.4e4, so that floor is near 1e-11 and pinning decimals below it would reject a conformant
    # accumulation that differs from this one only in the order it sums the Schur complement
    return (tuple(round(float(v), 5) for v in w),
            int(float(out["hermitian_defect"]) <= 1e-6),
            # exact, not a threshold: any genuine symmetrisation of the Schur complement makes
            # Q Hermitian to the bit, verified across the permitted solver band, whereas returning
            # the raw condensation leaves a defect near 1e-8 that a scaled threshold would admit
            int(bool(np.all(Q == Q.conj().T))),
            int(np.all(w > 0)), round(float(np.trace(Q).real), 5),
            round(float(np.abs(Q.imag).max()), 6) + 0.0,
            # a direct solve returning a dummy count, or an unpreconditioned iteration, shows here;
            # the band is wide so that any faithful preconditioned run passes
            int(all(1 <= c <= 5000 for c in out["iterations"])), len(out["iterations"]),
            # entrywise magnitudes, not signed entries: the sign of a rotation column is a gauge,
            # a similarity by diag(+-1) that leaves the pencil and every frequency alone, so the
            # signed coupling entries would grade a convention rather than the physics; the
            # magnitudes still catch a wrong column, which the eigenvalues alone would not
            tuple(round(abs(float(Q[i, j].real)), 5) + 0.0 for i in range(6) for j in range(6)),
            tuple(round(abs(float(Q[i, j].imag)), 5) + 0.0 for i in range(6) for j in range(6)))
Kfc, Kff, Kcc, CD, NS, HH, CTR = cell(10, (4, 3, 2), (np.pi, 0.4, 0.2))
def flat(x):
    # flatten to a tuple of plain numeric terminals, which is what AutoQC compares
    if isinstance(x, (tuple, list)):
        return tuple(v for e in x for v in flat(e))
    if hasattr(x, "tolist"):
        return flat(x.tolist())
    return (int(x),) if isinstance(x, bool) else (x,)
""",
            'call': 'flat(digest(_invoke_with_fresh_inputs(capacity_schur_matrix, Kff, Kfc, Kcc, CD, NS, HH, CTR, 1e-12)))',
            'gold_call': 'flat(digest(_invoke_with_fresh_inputs(_oracle_capacity_schur_matrix, Kff, Kfc, Kcc, CD, NS, HH, CTR, 1e-12)))',
        },
        {
            "setup": r"""import copy as _input_copy
def _invoke_with_fresh_inputs(fn, *args, **kwargs):
    # deep-copied arguments keep candidate, reference and repeated calls independent
    args, kwargs = _input_copy.deepcopy((args, kwargs))
    return fn(*args, **kwargs)

import numpy as np, scipy.sparse as sp
# two further exterior cells: a three-degree-of-freedom stand-in cannot carry six independent
# rigid motions, so this case exercises the same path on real geometry instead
def cell(n_side, spans, alpha, lam=1.5e6, mu=5.0e5, L=0.02):
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
    h = L / n_side; s = offs.astype(float) * 2.0 - 1.0
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
    Ke = 0.5 * (Ke + Ke.T)
    reach = el[:, None, :] + offs[None, :, :]; wrap = reach // n_side
    node = flat(reach); a = np.asarray(alpha, float); ph = np.exp(1j * (wrap @ a))
    dof = (3 * node[:, :, None] + np.arange(3)[None, None, :]).reshape(len(el), 24)
    p24 = np.repeat(ph, 3, axis=1); fac = np.conj(p24)[:, :, None] * p24[:, None, :]
    vals = (Ke[None, :, :] * fac).ravel()
    rows = np.repeat(dof, 24, axis=1).ravel(); cols = np.tile(dof, (1, 24)).ravel()
    K = sp.coo_matrix((vals, (rows, cols)), shape=(3 * n_side ** 3,) * 2).tocsr()
    cd = np.sort(np.concatenate([3 * surf + d for d in range(3)]))
    fd = np.sort(np.concatenate([3 * free + d for d in range(3)]))
    centre = (lo.astype(np.float64) + span / 2.0) * h
    return K[fd][:, cd], K[fd][:, fd].tocsr(), K[cd][:, cd].tocsr(), cd, n_side, h, centre
def digest(out):
    return (tuple(round(float(v), 8) for v in out["eigenvalues"]),
            # the defect sits at roundoff and moves with the solver, so check the contract the
            # docstring states rather than pinning a value no two implementations share
            int(float(out["hermitian_defect"]) <= 1e-6),
            int(all(c >= 1 for c in out["iterations"])), len(out["iterations"]),
            int(out["capacity"].shape[0]), int(out["capacity"].shape[1]))
A1, F1, C1, D1, N1, H1, T1 = cell(8, (2, 2, 2), (0.3, -1.1, 2.2))
A2, F2, C2, D2, N2, H2, T2 = cell(8, (4, 2, 2), (np.pi, np.pi, 0.0))
def flat(x):
    # flatten to a tuple of plain numeric terminals, which is what AutoQC compares
    if isinstance(x, (tuple, list)):
        return tuple(v for e in x for v in flat(e))
    if hasattr(x, "tolist"):
        return flat(x.tolist())
    return (int(x),) if isinstance(x, bool) else (x,)
""",
            'call': 'flat((digest(_invoke_with_fresh_inputs(capacity_schur_matrix, F1, A1, C1, D1, N1, H1, T1, 1e-13)), digest(_invoke_with_fresh_inputs(capacity_schur_matrix, F2, A2, C2, D2, N2, H2, T2, 1e-13))))',
            'gold_call': 'flat((digest(_invoke_with_fresh_inputs(_oracle_capacity_schur_matrix, F1, A1, C1, D1, N1, H1, T1, 1e-13)), digest(_invoke_with_fresh_inputs(_oracle_capacity_schur_matrix, F2, A2, C2, D2, N2, H2, T2, 1e-13))))',
        },
        {
            "setup": r"""import copy as _input_copy
def _invoke_with_fresh_inputs(fn, *args, **kwargs):
    # deep-copied arguments keep candidate, reference and repeated calls independent
    args, kwargs = _input_copy.deepcopy((args, kwargs))
    return fn(*args, **kwargs)

import numpy as np, scipy.sparse as sp
A = sp.csr_matrix(np.eye(6, dtype=complex) * 4.0)
F = sp.csr_matrix(np.ones((6, 3), dtype=complex) * -0.5)
C = sp.csr_matrix(np.eye(3, dtype=complex) * 3.0)
D = np.arange(3, dtype=np.int64)
NEG = sp.csr_matrix(np.diag([1.0, -1.0, 1.0, 1.0, 1.0, 1.0]).astype(complex))
# the two rejections the Raises clause promises and that nothing else reached: a surface block
# that condenses to a matrix far from Hermitian, and one that condenses to an indefinite matrix
ZERO = sp.csr_matrix(np.zeros((6, 3), dtype=complex))
SKEW = sp.csr_matrix(np.array([[3.0, 2.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 3.0]], dtype=complex))
INDEF = sp.csr_matrix(np.diag([1.0, -1.0, 1.0]).astype(complex))
def cell(n_side, spans, alpha, lam=1.5e6, mu=5.0e5, L=0.02):
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
    h = L / n_side; s = offs.astype(float) * 2.0 - 1.0
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
    Ke = 0.5 * (Ke + Ke.T)
    reach = el[:, None, :] + offs[None, :, :]; wrap = reach // n_side
    node = flat(reach); a = np.asarray(alpha, float); ph = np.exp(1j * (wrap @ a))
    dof = (3 * node[:, :, None] + np.arange(3)[None, None, :]).reshape(len(el), 24)
    p24 = np.repeat(ph, 3, axis=1); fac = np.conj(p24)[:, :, None] * p24[:, None, :]
    vals = (Ke[None, :, :] * fac).ravel()
    rows = np.repeat(dof, 24, axis=1).ravel(); cols = np.tile(dof, (1, 24)).ravel()
    K = sp.coo_matrix((vals, (rows, cols)), shape=(3 * n_side ** 3,) * 2).tocsr()
    cd = np.sort(np.concatenate([3 * surf + d for d in range(3)]))
    fd = np.sort(np.concatenate([3 * free + d for d in range(3)]))
    centre = (lo.astype(np.float64) + span / 2.0) * h
    return K[fd][:, cd], K[fd][:, fd].tocsr(), K[cd][:, cd].tocsr(), cd, n_side, h, centre
# a genuine small cell for the success case: three surface degrees of freedom cannot span six
# independent rigid motions, so the synthetic fixtures below can only exercise the error paths
GF, GA, GC, GD, GN, GH, GT = cell(6, (2, 2, 2), (0.4, 1.1, -0.7))
def verdict(fn, a=GA, f=GF, c=GC, d=GD, ns=GN, hh=GH, ct=GT, t=1e-12):
    try:
        _invoke_with_fresh_inputs(fn, a, f, c, d, ns, hh, ct, t)
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
    return flat((verdict(fn, t=0.0), verdict(fn, t=float('nan')), verdict(fn, a=NEG, f=F, c=C, d=D), verdict(fn, a=A, f=sp.csr_matrix(np.ones((5, 3), dtype=complex)), c=C, d=D), verdict(fn, a=A, f=F, c=C, d=np.arange(4, dtype=np.int64)), verdict(fn, a=A, f=ZERO, c=SKEW, d=D), verdict(fn, a=A, f=ZERO, c=INDEF, d=D), verdict(fn, ns=3), verdict(fn, hh=0.0), verdict(fn, ct=np.zeros(2)), verdict(fn)))
""",
            'call': 'verdicts(capacity_schur_matrix)',
            'gold_call': 'verdicts(_oracle_capacity_schur_matrix)',
        },
    ]
