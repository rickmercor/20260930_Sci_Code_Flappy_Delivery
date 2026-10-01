"""
Runs the whole chain on the benchmark configuration and returns the audit of the finite Coulomb lattice sum of a sheared sample.

The audit assembles the macroscopic potential of the prescribed proportions, the boundary and bulk parts, the finite sum and its scaled residual, the caesium chloride reference, the two finite-size coefficients of the sample (the multipole prediction and the measured one) with their difference, the same difference measured from the second-order term alone and evaluated analytically from the termination of the sample, and the controls in which the construction is exact: the cube, and the box with perpendicular edges of the same nominal lengths as the sample, whose boundary term is also compared with the arctangent closed form of a rectangular prism.

Returns
-------
A numpy float64 array of shape (18,) holding the audit of the finite Coulomb lattice sum, its first entry the termination contribution.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def lattice_audit(r: "np.ndarray", E: "np.ndarray", p: int, p_bulk: int, sizes: "np.ndarray", l: float) -> "np.ndarray":
    """Runs the whole chain on the benchmark configuration and returns the audit of the finite Coulomb lattice sum of a sheared sample.

    Args:
        r: array-like of shape (3,), the displacement of the positive charge from the negative one, in the
            same length units as l; must be non-zero.
        E: array-like of shape (3, 3) of integer values with non-zero determinant, whose columns are the
            edge directions of the sample.
        p: positive integer, the size index used for the reported bulk extraction, the finite sum, the
            scaled residual and the Madelung check.
        p_bulk: positive integer, the size index of the cubic extraction that fixes the macroscopic value
            inside the finite-size fits.
        sizes: array-like of at least three distinct positive integers, the size indices summed in the fits.
        l: positive float, the cubic lattice constant.

    Returns:
        A numpy float64 array of shape (18,) holding, in this order: [0] the termination contribution, the
        measured leading coefficient minus the multipole prediction, for the sample; [1] the measured leading
        coefficient; [2] the multipole prediction; [3] the (2p+1)^(-2) coefficient of the second-order sums
        of the sample; [4] the macroscopic potential of the sample; [5] its boundary term; [6] the bulk part
        from the cube at size index p; [7] the finite sum of the sample at size index p; [8] that sum minus
        the macroscopic value, multiplied by (2p+1)^2; [9] the caesium chloride Madelung constant at size
        index p; [10] the cube's multipole prediction minus its closed form; [11] the measured coefficient
        minus the multipole prediction for the box whose edge lengths are the absolute diagonal entries of E
        (the cube when that diagonal is singular); [12] that box's boundary term minus the arctangent closed
        form -(4/V) [x^2 arctan(bc/(ad)) + y^2 arctan(ac/(bd)) + z^2 arctan(ab/(cd))] with d the space
        diagonal; [13] the (2p+1)^(-4) coefficient of the sample's residual fit; [14] the number of cells of
        the sample at size index p, origin excluded; [15] the termination contribution evaluated analytically
        from the Euler-Maclaurin expansion of the sample's second-order sum; [16] entry [15] minus entry [0];
        [17] the analytic termination coefficient of the box of entry [11], which vanishes. Potentials and
        coefficients are in inverse length units, the reciprocal of the unit in which r and l are given (the
        absolute potential, which for l = 1 coincides with units of 1/l).

    Raises:
        ValueError: if any argument is invalid as in the earlier steps (E must have the sheared-box form of the
            previous step), or if the macroscopic potential comes out non-positive.
    """
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _real(v, name):
    if isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, float, np.integer, np.floating)):
        raise ValueError("invalid " + name)
    v = float(v)
    if not np.isfinite(v):
        raise ValueError("invalid " + name)
    return v


def _pos_float(v, name):
    v = _real(v, name)
    if v <= 0.0:
        raise ValueError("invalid " + name)
    return v


def _nonneg_int(v, name):
    if isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, np.integer)) or v < 0:
        raise ValueError("invalid " + name)
    return int(v)


def _pos_int(v, name):
    v = _nonneg_int(v, name)
    if v < 1:
        raise ValueError("invalid " + name)
    return v


def _vec3(r, name):
    a = np.asarray(r, dtype=np.float64)
    if a.shape != (3,) or not np.all(np.isfinite(a)):
        raise ValueError("invalid " + name)
    return a


def _edges(E, name):
    """3x3 matrix of integer values (any numeric dtype) whose COLUMNS are the edge directions."""
    A = np.asarray(E, dtype=np.float64)
    if A.shape != (3, 3) or not np.all(np.isfinite(A)):
        raise ValueError("invalid " + name)
    if not np.allclose(A, np.round(A), atol=1e-9):
        raise ValueError("invalid " + name)
    A = np.round(A)
    if abs(float(np.linalg.det(A))) < 0.5:
        raise ValueError("invalid " + name)
    return A


def _lattice_E(p, A):
    """Lattice vectors of the sample {n : max_i |(A^-1 n)_i| <= (2p+1)/2}, origin excluded."""
    K = (2 * p + 1) / 2.0
    Ainv = np.linalg.inv(A)
    ext = np.abs(A) @ np.array([K, K, K])
    rng = [np.arange(-int(np.floor(e)), int(np.floor(e)) + 1, dtype=np.float64) for e in ext]
    I, J, L = np.meshgrid(rng[0], rng[1], rng[2], indexing="ij")
    n = np.stack([I.ravel(), J.ravel(), L.ravel()], axis=1)
    u = np.einsum("ij,kj->ik", n, Ainv)
    n = n[np.max(np.abs(u), axis=1) <= K + 1e-9]
    nn = np.sqrt(np.einsum("ij,ij->i", n, n))
    keep = nn > 0.0
    return n[keep], nn[keep]


def _pair_sum_E(r, p, A, l):
    """Finite lattice sum nu(r, p | sample A) for one +/- pair per cell, in inverse length units."""
    n, nn = _lattice_E(p, A)
    rr = np.asarray(r, dtype=np.float64) / l
    d1 = np.sqrt(np.einsum("ij,ij->i", n + rr, n + rr))
    return (1.0 / np.sqrt(rr @ rr) + float(np.sum(1.0 / d1 - 1.0 / nn))) / l


def _face_grid(A, i, sgn, m):
    """Gauss-Legendre nodes/weights on the face u_i = sgn/2 of the unit sample spanned by A.

    Returns the points x (m*m, 3), the weights (m*m,) that include the face area, and the
    outward unit normal.
    """
    a = [A[:, k] for k in range(3)]
    j, k = [q for q in range(3) if q != i]
    nvec = np.cross(a[j], a[k])
    area = float(np.sqrt(nvec @ nvec))
    nhat = nvec / area
    if float(nhat @ a[i]) < 0.0:
        nhat = -nhat
    nhat = sgn * nhat
    g, w = np.polynomial.legendre.leggauss(m)
    g = 0.5 * g
    w = 0.5 * w
    S, T = np.meshgrid(g, g, indexing="ij")
    WS, WT = np.meshgrid(w, w, indexing="ij")
    x = (sgn * 0.5) * a[i][None, :] + S.ravel()[:, None] * a[j][None, :] + T.ravel()[:, None] * a[k][None, :]
    return x, (WS * WT).ravel() * area, nhat


def _boundary_tensor(A):
    """D_ij = surface integral over the unit sample of n_i x_j / |x|^3 (trace 4 pi)."""
    D = np.zeros((3, 3))
    for i in range(3):
        for sgn in (1.0, -1.0):
            x, w, nh = _face_grid(A, i, sgn, 96)
            x3 = np.sqrt(np.einsum("ij,ij->i", x, x)) ** 3
            D += np.einsum("i,ij->j", w / x3, x)[None, :] * nh[:, None]
    return D


def _multipole_coefficient(A, r, V):
    """K^-2 coefficient of the continuum hexadecapole exterior integral, sample spanned by A.

    -(1/V) int_ext r^4 P4(cos) / |x|^5 d^3x over the exterior of the size-K sample (boundary at
    K l R(xhat)) equals t2 / K^2 with t2 = -(1/(2 V l^2)) sum_faces int h(xhat) (xhat.n) / |x|^4 dS
    over the faces of the unit-size sample, h = r^4 P4(rhat.xhat).
    """
    rr = float(r @ r)
    l = V ** (1.0 / 3.0)
    tot = 0.0
    for i in range(3):
        for sgn in (1.0, -1.0):
            x, w, nh = _face_grid(A, i, sgn, 96)
            nx = np.sqrt(np.einsum("ij,ij->i", x, x))
            xh = x / nx[:, None]
            c = np.einsum("ij,j->i", xh, r) / np.sqrt(rr)
            h = rr ** 2 * (35.0 * c ** 4 - 30.0 * c ** 2 + 3.0) / 8.0
            tot += float(np.sum(w * h * np.einsum("ij,j->i", xh, nh) / nx ** 4))
    # the size-K sample of a lattice of constant l has its boundary at K * l * R(xhat): one factor 1/l^2
    # from the radial integral on top of the 1/V of the cell density, i.e. r^4 / l^5 overall
    return -tot / (2.0 * V * l * l)


def _sizes(sizes, name):
    s = np.asarray(sizes)
    if s.ndim != 1 or s.shape[0] < 3 or not np.all(np.isfinite(s.astype(np.float64))):
        raise ValueError("invalid " + name)
    if not np.all(s == np.round(s)) or np.any(s < 1) or len(set(int(v) for v in s)) != s.shape[0]:
        raise ValueError("invalid " + name)
    return [int(v) for v in s]


def _quadratic_sum_E(r, p, A, l):
    """Lattice sum over the sample of the second-order (dipole-dipole) multipole term, in inverse length units."""
    n, nn = _lattice_E(p, A)
    rr = np.asarray(r, dtype=np.float64) / l
    rn = np.einsum("ij,j->i", n, rr)
    return float(np.sum((3.0 * rn ** 2 - float(rr @ rr) * nn ** 2) / (2.0 * nn ** 5))) / l


def _sheared_family(E, name):
    """Edge matrix of the form [[a, b, 0], [0, c, 0], [0, 0, d]] with a, c, d odd positive integers."""
    A = _edges(E, name)
    a, b, c, d = float(A[0, 0]), float(A[0, 1]), float(A[1, 1]), float(A[2, 2])
    off = np.array([A[0, 2], A[1, 0], A[1, 2], A[2, 0], A[2, 1]])
    if np.any(off != 0.0) or min(a, c, d) <= 0.0 or int(a) % 2 == 0 or int(c) % 2 == 0 or int(d) % 2 == 0:
        raise ValueError("invalid " + name)
    return a, b, c, d


def _g2(x, r):
    """Second-order (dipole-dipole) term of the multipole series, [3 (r.x)^2 - r^2 x^2] / (2 |x|^5)."""
    rr = float(r @ r)
    rx = np.einsum("...k,k->...", x, r)
    x2 = np.einsum("...k,...k->...", x, x)
    return (3.0 * rx * rx - rr * x2) / (2.0 * x2 ** 2.5)


def _grad_g2(x, r):
    rr = float(r @ r)
    rx = np.einsum("...k,k->...", x, r)
    x2 = np.einsum("...k,...k->...", x, x)
    t1 = (3.0 * rx[..., None] * r[None, :] - rr * x) / x2[..., None] ** 2.5
    t2 = 5.0 * (3.0 * rx * rx - rr * x2)[..., None] * x / (2.0 * x2[..., None] ** 3.5)
    return t1 - t2


def _gl_nodes(n, lo, hi):
    x, w = np.polynomial.legendre.leggauss(n)
    return 0.5 * (hi - lo) * x + 0.5 * (hi + lo), 0.5 * (hi - lo) * w


def _em_q2(a, b, c, d, r, n=160):
    """K^-2 coefficient of the second-order lattice sum minus its limit, unit-scale sheared box.

    Iterated Euler-Maclaurin over n_x (limits (b/c) n_y +- a/2, boundary phase period c), then n_y
    and n_z (midplane faces): midpoint face terms, the phase-averaged second-order term of the
    sheared faces, the edge term from the n_y dependence of the n_x limits, and the summation-by-
    parts boundary term of the zero-mean first-order weight.
    """
    s = b / c
    y, wy = _gl_nodes(n, -c / 2, c / 2)
    z, wz = _gl_nodes(n, -d / 2, d / 2)
    u, wu = _gl_nodes(n, -0.5, 0.5)
    term1 = 0.0
    for sy in (1.0, -1.0):
        yy = sy * c / 2
        X = s * yy + a * u[:, None] + 0.0 * z[None, :]
        Z = 0.0 * u[:, None] + z[None, :]
        P = np.stack([X, np.full_like(X, yy), Z], axis=-1)
        term1 += sy * float(np.sum((a * wu)[:, None] * wz[None, :] * _grad_g2(P, r)[..., 1]))
    for sz in (1.0, -1.0):
        zz = sz * d / 2
        X = s * y[None, :] + a * u[:, None]
        Y = 0.0 * u[:, None] + y[None, :]
        P = np.stack([X, Y, np.full_like(X, zz)], axis=-1)
        term1 += sz * float(np.sum((a * wu)[:, None] * wy[None, :] * _grad_g2(P, r)[..., 2]))
    term1 *= -1.0 / 24.0
    cc = int(round(c))
    k = np.arange(cc)
    phi = np.mod(int(round(b)) * k / cc + 0.5, 1.0)
    b1 = phi - 0.5
    b2mean = float(np.mean(phi * phi - phi + 1.0 / 6.0))
    Y2, Z2 = np.meshgrid(y, z, indexing="ij")
    W2 = wy[:, None] * wz[None, :]
    Pp = np.stack([s * Y2 + a / 2, Y2, Z2], axis=-1)
    Pm = np.stack([s * Y2 - a / 2, Y2, Z2], axis=-1)
    term2 = 0.5 * b2mean * float(np.sum(W2 * (_grad_g2(Pp, r)[..., 0] - _grad_g2(Pm, r)[..., 0])))

    def _f1(yy):
        Pp = np.stack([np.full_like(z, s * yy + a / 2), np.full_like(z, yy), z], axis=-1)
        Pm = np.stack([np.full_like(z, s * yy - a / 2), np.full_like(z, yy), z], axis=-1)
        return float(np.sum(wz * (_g2(Pp, r) - _g2(Pm, r))))
    f_top, f_bot = _f1(c / 2), _f1(-c / 2)
    term3 = -(1.0 / 24.0) * s * (f_top - f_bot)
    start = int((-(cc - 1) // 2) % cc)
    wbar = float(np.mean(np.cumsum(b1[(start + np.arange(cc)) % cc])))
    term4 = wbar * (f_top - f_bot)
    return term1 + term2 + term3 + term4


def _oracle_lattice_audit(r: "np.ndarray", E: "np.ndarray", p: int, p_bulk: int, sizes: "np.ndarray", l: float) -> "np.ndarray":
    r = _vec3(r, "r")
    A = _edges(E, "E")
    p = _pos_int(p, "p")
    p_bulk = _pos_int(p_bulk, "p_bulk")
    ps = _sizes(sizes, "sizes")
    l = _pos_float(l, "l")
    V = l ** 3
    inf = _oracle_infinite_crystal_potential(r, A, p, l)
    fit = _oracle_finite_size_fit(r, A, ps, p_bulk, l)
    mp = _oracle_multipole_coefficient(r, A, V)
    s2 = fit[0] - mp[0]
    mad = _oracle_madelung_constant(p)
    # control: a sample with mutually perpendicular edges of the same nominal lengths
    B = np.diag(np.abs(np.diag(A)))
    if abs(float(np.linalg.det(B))) < 0.5:
        B = np.eye(3)
    fitb = _oracle_finite_size_fit(r, B, ps, p_bulk, l)
    mpb = _oracle_multipole_coefficient(r, B, V)
    nbb = _oracle_boundary_term(r, B, V)[0]
    a, b, c = (float(B[0, 0]), float(B[1, 1]), float(B[2, 2]))
    d = np.sqrt(a * a + b * b + c * c)
    x, y, z = r / l
    arctan = -(4.0 / l) * (x * x * np.arctan(b * c / (a * d)) + y * y * np.arctan(a * c / (b * d))
                          + z * z * np.arctan(a * b / (c * d)))
    cen = _oracle_lattice_census(p, A)
    term = _oracle_termination_coefficient(r, A, l)
    return np.array([
        s2,               # 0 termination contribution: actual minus multipole coefficient
        fit[0],           # 1 actual leading coefficient c2 (even-power fit)
        mp[0],            # 2 multipole (continuum hexadecapole) coefficient t2
        fit[2],           # 3 K^-2 coefficient of the quadratic-term lattice sum (independent route to s2)
        inf[0],           # 4 macroscopic potential of the sample
        inf[2],           # 5 boundary term of the sample
        inf[1],           # 6 bulk term from the cube at size p
        inf[3],           # 7 finite lattice sum of the sample at size p
        inf[4],           # 8 scaled residual at size p
        mad[0],           # 9 CsCl Madelung constant at size p
        mp[3],            # 10 cube: multipole construction minus closed form (should vanish)
        fitb[0] - mpb[0], # 11 control box: actual minus multipole coefficient (should be negligible)
        nbb - arctan,     # 12 control box: boundary term minus arctangent closed form (should vanish)
        fit[1],           # 13 K^-4 coefficient of the sample's residual
        cen[0],           # 14 number of cells of the sample at size p, origin excluded
        term[0],          # 15 termination contribution from the Euler-Maclaurin evaluation (analytic)
        term[0] - s2,     # 16 analytic minus fitted termination contribution (fit truncation, should be small)
        term[1],          # 17 analytic termination coefficient of the control box (should vanish)
    ], dtype=np.float64)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        {'setup': 'import numpy as np\nE = np.array([[7, 3, 0], [0, 5, 0], [0, 0, 3]])\n',
         'call': 'lattice_audit(np.array([0.37, 0.21, 0.13]), E, 8, 12, np.array([6, 8, 10, 12]), 1.0)',
         'gold_call': '_oracle_lattice_audit(np.array([0.37, 0.21, 0.13]), E, 8, 12, np.array([6, 8, 10, 12]), 1.0)',
         'tol': 1e-08},
        {'setup': 'import numpy as np\n# the mirror image of the benchmark shear, sheared the other way\nE = np.array([[7, -3, 0], [0, 5, 0], [0, 0, 3]])\n',
         'call': 'lattice_audit(np.array([0.37, 0.21, 0.13]), E, 6, 8, np.array([4, 6, 8]), 1.0)',
         'gold_call': '_oracle_lattice_audit(np.array([0.37, 0.21, 0.13]), E, 6, 8, np.array([4, 6, 8]), 1.0)',
         'tol': 1e-08},
        {'setup': 'import numpy as np\n# boundary: a box with perpendicular edges, where the termination contribution should be negligible\n',
         'call': 'lattice_audit(np.array([0.5, 0.0, 0.0]), np.diag([3, 1, 1]), 5, 6, np.array([3, 4, 6]), 1.0)',
         'gold_call': '_oracle_lattice_audit(np.array([0.5, 0.0, 0.0]), np.diag([3, 1, 1]), 5, 6, np.array([3, 4, 6]), 1.0)',
         'tol': 1e-08},
        {'setup': 'import numpy as np\n# invalid input: a zero bulk size index must raise ValueError\nE = np.array([[7, 3, 0], [0, 5, 0], [0, 0, 3]])\ndef _catches_value_error(fn):\n    try:\n        fn()\n    except ValueError:\n        return np.array([1.0])\n    return np.array([0.0])\n',
         'call': '_catches_value_error(lambda: lattice_audit(np.array([0.37, 0.21, 0.13]), E, 8, 0, np.array([6, 8, 10]), 1.0))',
         'gold_call': '_catches_value_error(lambda: _oracle_lattice_audit(np.array([0.37, 0.21, 0.13]), E, 8, 0, np.array([6, 8, 10]), 1.0))'},
    ]
