"""
The pencil is large and sparse, and the frequencies wanted sit at the bottom of a spectrum that reaches into the megahertz, so the eigenvalues are to be found by a method suited to that: a shift-and-invert iteration about a low shift, or any other iteration that returns the smallest eigenvalues of a Hermitian positive definite pencil, factorising the sparse matrix once and iterating on the factor. Whatever iteration is used, each returned pair must be checked against the pencil directly, by forming the residual of the eigenvalue equation and requiring it to be small relative to the stiffness term; an iteration that has not converged is rejected rather than reported. When the matrices are real, keep the arithmetic real.

Returns
-------
dict, holding the count lowest eigenfrequencies of the pencil ascending, in hertz and in radian per second, the relative residual of each, and the count.

Extract the lowest eigenfrequencies of the finite-contrast pencil.

The pencil assembled at the previous stage is the discrete form of the full transmission problem at the stated finite contrast, with no reduction to rigid motions anywhere in it. Its eigenvalues are the squared angular frequencies of the Bloch modes of the two-phase cell at that quasi-momentum, and its lowest ones are the modes in which the resonator carries almost all of the motion. What this stage returns is the lowest few of them, in hertz and ascending, each solved to a stated relative residual so that the value can be compared against a leading-order figure at a hundredth of a hertz.

Returns
-------
dict, holding the count lowest eigenfrequencies of the pencil ascending, in hertz and in radian per second, the relative residual of each, and the count.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def full_pencil_spectrum(
    K_full: "scipy.sparse.spmatrix",
    M_full: "scipy.sparse.spmatrix",
    count: int,
    rel_tol: float,
) -> dict:
    """Return the lowest eigenfrequencies of the Hermitian pencil K_full v = omega squared M_full v.

    Parameters
    ----------
    K_full : scipy.sparse.spmatrix
        The finite-contrast stiffness of the whole cell, Hermitian positive definite.
    M_full : scipy.sparse.spmatrix
        The consistent mass of the whole cell, Hermitian positive definite, same shape.
    count : int
        How many of the lowest eigenfrequencies to return, at least one and below the order.
    rel_tol : float
        Bound on the relative residual of each returned pair, finite and above zero.

    Returns
    -------
    dict
        Under the keys hertz, angular, residuals and count. hertz and angular are float arrays
        of shape (count,), ascending, the eigenfrequencies in hertz and in radian per second.
        residuals holds, for each returned pair, the norm of K_full v minus omega squared
        M_full v divided by the norm of K_full v, each at most rel_tol. count is the number
        returned.

    Raises
    ------
    ValueError
        When K_full or M_full is not square, when their shapes differ, when either departs from Hermitian symmetry by more than one part in 1e-9 of its largest entry, when count is not an integer between one and the order minus one, when rel_tol is not finite and above zero, when an eigenvalue comes out non-positive, or when any returned pair fails its residual bound.
    """
    return

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


def _oracle_full_pencil_spectrum(
    K_full: "scipy.sparse.spmatrix",
    M_full: "scipy.sparse.spmatrix",
    count: int,
    rel_tol: float,
) -> dict:
    """Reference implementation."""
    if not sp.issparse(K_full) or not sp.issparse(M_full):
        raise ValueError("K_full and M_full must be sparse matrices")
    K = K_full.tocsc()
    M = M_full.tocsc()
    if K.shape[0] != K.shape[1] or M.shape[0] != M.shape[1]:
        raise ValueError("K_full and M_full must be square")
    if K.shape != M.shape:
        raise ValueError("K_full and M_full must have the same shape")
    n = K.shape[0]
    for name, A in (("K_full", K), ("M_full", M)):
        top = float(abs(A).max()) if A.nnz else 0.0
        if top == 0.0 or float(abs(A - A.conj().T).max()) > 1.0e-9 * top:
            raise ValueError("%s must be Hermitian" % name)
    if isinstance(count, bool) or not isinstance(count, (int, np.integer)) or not 1 <= int(count) < n:
        raise ValueError("count must lie in [1, order - 1]")
    count = int(count)
    tol = float(rel_tol)
    if not np.isfinite(tol) or tol <= 0.0:
        raise ValueError("rel_tol must be finite and above zero")

    real = not (np.iscomplexobj(K.data) and np.abs(K.data.imag).max() > 0.0) \
        and not (np.iscomplexobj(M.data) and np.abs(M.data.imag).max() > 0.0)
    if real:
        K = K.real.tocsc(); M = M.real.tocsc()
    sigma = -1.0
    v0 = np.ones(n, dtype=np.float64 if real else np.complex128)
    # a few pairs beyond the ones wanted, then keep the lowest: at a quasi-momentum whose phases
    # are all real the cell is symmetric about the resonator centroid and the all-ones start
    # vector lies in one parity sector, so a block of exactly count pairs can converge without
    # ever reaching a mode of the opposite symmetry and silently report a higher eigenvalue
    k = min(count + 4, n - 1)
    w, V = spla.eigsh(K, k=k, M=M, sigma=sigma, which="LM", v0=v0, tol=1e-12)
    order = np.argsort(w.real)[:count]
    w = w.real[order]; V = V[:, order]
    if not np.all(w > 0.0):
        raise ValueError("every eigenvalue of the pencil must be positive")
    res = np.empty(count)
    for j in range(count):
        v = V[:, j]
        kv = K @ v
        r = kv - w[j] * (M @ v)
        res[j] = float(np.sqrt(np.sum(np.abs(r) ** 2).real) / np.sqrt(np.sum(np.abs(kv) ** 2).real))
    if np.any(res > tol):
        raise ValueError("an eigenpair did not reach rel_tol: worst residual %.3e" % float(res.max()))
    ang = np.sqrt(w)
    return {"hertz": ang / (2.0 * np.pi), "angular": ang, "residuals": res, "count": count}

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
def _pencil(n_side, spans, Ke, offs, h, rho, delta, eps, alpha):
    # the two-phase Bloch pencil of the cell, assembled here from closed forms: interior elements
    # carry Ke / delta and rho / eps, the consistent brick mass is h^3 / 27 halved once per
    # coordinate in which two corners differ, and each wrapped corner carries exp(i alpha . wrap)
    span = np.array(spans, dtype=np.int64); lo = (n_side - span) // 2; hi = lo + span
    g = np.arange(n_side, dtype=np.int64)
    ii, jj, kk = np.meshgrid(g, g, g, indexing="ij")
    cells = np.stack([ii.ravel(), jj.ravel(), kk.ravel()], 1)
    inside = np.all((cells >= lo) & (cells < hi), axis=1)
    kscale = np.where(inside, 1.0 / delta, 1.0); mscale = np.where(inside, rho / eps, rho)
    dab = (offs[:, None, :] != offs[None, :, :]).sum(-1)
    Me = np.kron(h ** 3 / 27.0 * 0.5 ** dab, np.eye(3))
    reach = cells[:, None, :] + offs[None, :, :]; wrap = reach // n_side
    node = (reach[..., 0] % n_side) * n_side * n_side + (reach[..., 1] % n_side) * n_side + reach[..., 2] % n_side
    a = np.asarray(alpha, dtype=float); ph = np.exp(1j * (wrap @ a))
    if np.allclose(np.sin(a), 0.0):
        ph = ph.real
    dof = (3 * node[:, :, None] + np.arange(3)[None, None, :]).reshape(len(cells), 24)
    p24 = np.repeat(ph, 3, axis=1); fac = np.conj(p24)[:, :, None] * p24[:, None, :]
    rows = np.repeat(dof, 24, axis=1).ravel(); cols = np.tile(dof, (1, 24)).ravel()
    n = 3 * n_side ** 3
    K = sp.coo_matrix(((Ke[None, :, :] * fac * kscale[:, None, None]).ravel(), (rows, cols)), shape=(n, n)).tocsc()
    M = sp.coo_matrix(((Me[None, :, :] * fac * mscale[:, None, None]).ravel(), (rows, cols)), shape=(n, n)).tocsc()
    return {"K_full": 0.5 * (K + K.conj().T), "M_full": 0.5 * (M + M.conj().T)}
# the small cell at the real-phase quasi-momentum, its pencil from the previous stage
KE = _elem(1.5e6, 5.0e5, 0.002)
P = _pencil(10, (4, 3, 2), KE["element_stiffness"], KE["corner_signs"], 0.002, 1200.0, 1.0e-2, 5.0e-3, (np.pi, 0.0, 0.0))
def digest(out):
    hz = np.asarray(out["hertz"], dtype=float)
    return (tuple(round(float(v), 5) for v in hz),
            tuple(round(float(v), 4) for v in np.asarray(out["angular"], dtype=float)),
            int(bool(np.all(np.diff(hz) >= 0.0))), int(out["count"]),
            # flags, not values: the residuals sit at the solver floor and move with it,
            # but a residual of exactly zero is a residual that was never computed
            int(bool(np.all((np.asarray(out["residuals"], dtype=float) > 0.0) & (np.asarray(out["residuals"], dtype=float) <= 1e-8)))),
            int(bool(np.all(hz > 0.0))))
def flat(x):
    # flatten to a tuple of plain numeric terminals, which is what AutoQC compares
    if isinstance(x, (tuple, list)):
        return tuple(v for e in x for v in flat(e))
    if hasattr(x, "tolist"):
        return flat(x.tolist())
    return (int(x),) if isinstance(x, bool) else (x,)
""",
            "call": "flat(digest(_invoke_with_fresh_inputs(full_pencil_spectrum, P['K_full'], P['M_full'], 8, 1e-8)))",
            "gold_call": "flat(digest(_invoke_with_fresh_inputs(_oracle_full_pencil_spectrum, P['K_full'], P['M_full'], 8, 1e-8)))",
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
def _pencil(n_side, spans, Ke, offs, h, rho, delta, eps, alpha):
    # the two-phase Bloch pencil of the cell, assembled here from closed forms: interior elements
    # carry Ke / delta and rho / eps, the consistent brick mass is h^3 / 27 halved once per
    # coordinate in which two corners differ, and each wrapped corner carries exp(i alpha . wrap)
    span = np.array(spans, dtype=np.int64); lo = (n_side - span) // 2; hi = lo + span
    g = np.arange(n_side, dtype=np.int64)
    ii, jj, kk = np.meshgrid(g, g, g, indexing="ij")
    cells = np.stack([ii.ravel(), jj.ravel(), kk.ravel()], 1)
    inside = np.all((cells >= lo) & (cells < hi), axis=1)
    kscale = np.where(inside, 1.0 / delta, 1.0); mscale = np.where(inside, rho / eps, rho)
    dab = (offs[:, None, :] != offs[None, :, :]).sum(-1)
    Me = np.kron(h ** 3 / 27.0 * 0.5 ** dab, np.eye(3))
    reach = cells[:, None, :] + offs[None, :, :]; wrap = reach // n_side
    node = (reach[..., 0] % n_side) * n_side * n_side + (reach[..., 1] % n_side) * n_side + reach[..., 2] % n_side
    a = np.asarray(alpha, dtype=float); ph = np.exp(1j * (wrap @ a))
    if np.allclose(np.sin(a), 0.0):
        ph = ph.real
    dof = (3 * node[:, :, None] + np.arange(3)[None, None, :]).reshape(len(cells), 24)
    p24 = np.repeat(ph, 3, axis=1); fac = np.conj(p24)[:, :, None] * p24[:, None, :]
    rows = np.repeat(dof, 24, axis=1).ravel(); cols = np.tile(dof, (1, 24)).ravel()
    n = 3 * n_side ** 3
    K = sp.coo_matrix(((Ke[None, :, :] * fac * kscale[:, None, None]).ravel(), (rows, cols)), shape=(n, n)).tocsc()
    M = sp.coo_matrix(((Me[None, :, :] * fac * mscale[:, None, None]).ravel(), (rows, cols)), shape=(n, n)).tocsc()
    return {"K_full": 0.5 * (K + K.conj().T), "M_full": 0.5 * (M + M.conj().T)}
# complex phases, and a second contrast: the lowest six must scale as the square root of eps
KE = _elem(1.5e6, 5.0e5, 0.0025)
P1 = _pencil(8, (2, 4, 4), KE["element_stiffness"], KE["corner_signs"], 0.0025, 1200.0, 1.0e-2, 5.0e-3, (0.3, -1.1, 2.2))
P2 = _pencil(8, (2, 4, 4), KE["element_stiffness"], KE["corner_signs"], 0.0025, 1200.0, 2.5e-3, 1.25e-3, (0.3, -1.1, 2.2))
def digest(out):
    hz = np.asarray(out["hertz"], dtype=float)
    return (tuple(round(float(v), 5) for v in hz), int(bool(np.all(np.diff(hz) >= 0.0))), int(out["count"]),
            int(bool(np.all((np.asarray(out["residuals"], dtype=float) > 0.0) & (np.asarray(out["residuals"], dtype=float) <= 1e-8)))))
def flat(x):
    # flatten to a tuple of plain numeric terminals, which is what AutoQC compares
    if isinstance(x, (tuple, list)):
        return tuple(v for e in x for v in flat(e))
    if hasattr(x, "tolist"):
        return flat(x.tolist())
    return (int(x),) if isinstance(x, bool) else (x,)
""",
            "call": "flat((digest(_invoke_with_fresh_inputs(full_pencil_spectrum, P1['K_full'], P1['M_full'], 6, 1e-8)), digest(_invoke_with_fresh_inputs(full_pencil_spectrum, P2['K_full'], P2['M_full'], 6, 1e-8))))",
            "gold_call": "flat((digest(_invoke_with_fresh_inputs(_oracle_full_pencil_spectrum, P1['K_full'], P1['M_full'], 6, 1e-8)), digest(_invoke_with_fresh_inputs(_oracle_full_pencil_spectrum, P2['K_full'], P2['M_full'], 6, 1e-8))))",
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
def _pencil(n_side, spans, Ke, offs, h, rho, delta, eps, alpha):
    # the two-phase Bloch pencil of the cell, assembled here from closed forms: interior elements
    # carry Ke / delta and rho / eps, the consistent brick mass is h^3 / 27 halved once per
    # coordinate in which two corners differ, and each wrapped corner carries exp(i alpha . wrap)
    span = np.array(spans, dtype=np.int64); lo = (n_side - span) // 2; hi = lo + span
    g = np.arange(n_side, dtype=np.int64)
    ii, jj, kk = np.meshgrid(g, g, g, indexing="ij")
    cells = np.stack([ii.ravel(), jj.ravel(), kk.ravel()], 1)
    inside = np.all((cells >= lo) & (cells < hi), axis=1)
    kscale = np.where(inside, 1.0 / delta, 1.0); mscale = np.where(inside, rho / eps, rho)
    dab = (offs[:, None, :] != offs[None, :, :]).sum(-1)
    Me = np.kron(h ** 3 / 27.0 * 0.5 ** dab, np.eye(3))
    reach = cells[:, None, :] + offs[None, :, :]; wrap = reach // n_side
    node = (reach[..., 0] % n_side) * n_side * n_side + (reach[..., 1] % n_side) * n_side + reach[..., 2] % n_side
    a = np.asarray(alpha, dtype=float); ph = np.exp(1j * (wrap @ a))
    if np.allclose(np.sin(a), 0.0):
        ph = ph.real
    dof = (3 * node[:, :, None] + np.arange(3)[None, None, :]).reshape(len(cells), 24)
    p24 = np.repeat(ph, 3, axis=1); fac = np.conj(p24)[:, :, None] * p24[:, None, :]
    rows = np.repeat(dof, 24, axis=1).ravel(); cols = np.tile(dof, (1, 24)).ravel()
    n = 3 * n_side ** 3
    K = sp.coo_matrix(((Ke[None, :, :] * fac * kscale[:, None, None]).ravel(), (rows, cols)), shape=(n, n)).tocsc()
    M = sp.coo_matrix(((Me[None, :, :] * fac * mscale[:, None, None]).ravel(), (rows, cols)), shape=(n, n)).tocsc()
    return {"K_full": 0.5 * (K + K.conj().T), "M_full": 0.5 * (M + M.conj().T)}
KE = _elem(1.0, 0.5, 0.125)
P = _pencil(6, (2, 2, 2), KE["element_stiffness"], KE["corner_signs"], 0.125, 1.0, 0.5, 0.5, (0.9, 0.0, 0.0))
GK, GM = P["K_full"], P["M_full"]
SKEW = GK.tolil(); SKEW[0, 1] = SKEW[0, 1] + 1.0e-3 * abs(GK).max(); SKEW = SKEW.tocsc()
def verdict(fn, k=GK, m=GM, c=4, t=1e-8):
    try:
        _invoke_with_fresh_inputs(fn, k, m, c, t)
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
    return flat((verdict(fn, k=GK[:, :10]), verdict(fn, m=GM[:100, :100]), verdict(fn, k=SKEW), verdict(fn, c=0), verdict(fn, c=GK.shape[0]), verdict(fn, t=0.0), verdict(fn, t=float('inf')), verdict(fn, t=1e-25), verdict(fn)))
""",
            'call': 'verdicts(full_pencil_spectrum)',
            'gold_call': 'verdicts(_oracle_full_pencil_spectrum)',
        },
    ]
