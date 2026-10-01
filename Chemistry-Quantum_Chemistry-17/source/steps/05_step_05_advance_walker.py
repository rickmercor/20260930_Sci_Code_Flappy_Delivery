"""
Advance an unrestricted walker through one force-biased imaginary-time step with a selected DOCI trial, retaining mixed estimators and the determinant normalization.

The trial uses paired configurations in a fixed orthonormal spatial orbital frame, while the spin determinants evolve independently in the Hamiltonian basis. Singular component overlaps can contribute to mixed matrix elements. The trial force sets a complex Gaussian shift, and normal ordering fixes the one-body part of the propagator. QR normalization changes the determinant overlap through its triangular factors; this factor and the Gaussian correction remain separate from the normalized mixed energy.

Returns
-------
return q_alpha, q_beta, old_overlap, new_overlap, old_energy, new_energy, old_force, log_gaussian, gauge
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def advance_walker(occupations: "np.ndarray", coefficients: "np.ndarray", phi_alpha: "np.ndarray", phi_beta: "np.ndarray", h: "np.ndarray", factors: "np.ndarray", field: "np.ndarray", dt: float, trial_orbitals: "np.ndarray | None" = None) -> "tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]":
    """Return the propagated walker and the quantities for its weight update.

    The Hamiltonian spatial basis is orthonormal. The trial ket is the sum
    of the listed paired determinants with the supplied complex coefficients.
    Each determinant occupies columns U[:, occupations[d]] in both spins;
    alpha creation operators precede beta creation operators, with increasing
    occupied frame indices within each spin. U is fixed and defaults to the
    identity. No orbital or coefficient optimization is requested.

    The Hamiltonian is
    H = sum_{pq,s} h[p,q] a^dagger[p,s] a[q,s]
        + (1/2) sum_{l,pqrs,sigma,tau} L[l,p,q] L[l,r,s]
          a^dagger[p,sigma] a^dagger[r,tau] a[s,tau] a[q,sigma],
    without a nuclear constant. Define
    v_l = i sum_{pq,s} L[l,p,q] a^dagger[p,s] a[q,s].
    For any walker |W>, O(W)=<Psi_T|W>, E(W)=<Psi_T|H|W>/O(W),
    and f_l(W)=<Psi_T|v_l|W>/O(W). Only the summed trial overlap
    is an estimator denominator; zero overlaps of individual components
    can coexist with nonzero one- or two-body matrix elements.

    The force-biased one-body propagator is
    B = exp(-dt*h0/2)
        @ exp(i*sqrt(dt)*sum_l (field[l]-bar_x[l])*L[l])
        @ exp(-dt*h0/2),
    with h0=h-(1/2)*sum_l L[l]@L[l] and
    bar_x=-sqrt(dt)*f(Phi) evaluated on the input walker.
    For each spin, B@phi=Q@R is the thin QR factorization whose R diagonal
    is real positive. |B Phi>=gauge*|Q_alpha Q_beta>, with
    gauge=det(R_alpha)*det(R_beta). No weight update is performed here.

    Parameters
    ----------
    occupations : integer ndarray, shape (nd,k)
        Distinct, strictly increasing paired occupation tuples with indices
        in [0,n), 1 <= k <= min(n,6), 1 <= n <= 28, and 1 <= nd <= 32.
        A selected subset in any row order is permitted.
    coefficients : real ndarray, shape (nd,2)
        Packed complex ket coefficients, not necessarily normalized.
    phi_alpha, phi_beta : real ndarrays, shape (n,k,2)
        Packed complex full-column-rank walker orbitals. The spin sectors
        are independent and need not be orthonormal or paired.
    h : real ndarray, shape (n,n)
        Symmetric one-electron integrals in Eh.
    factors : real ndarray, shape (r,n,n)
        Symmetric two-electron factors in sqrt(Eh), 1 <= r <= 8.
    field : real ndarray, shape (r,)
        Prescribed standard-normal coordinates, one per factor.
    dt : float
        Nonnegative imaginary-time step in inverse Eh. At dt=0 the
        propagator is the identity, but the input walker is still normalized.
    trial_orbitals : real ndarray, shape (n,n,2), or None
        Packed complex unitary matrix U in the Hamiltonian basis.
        U^dagger U=I is a precondition. None means U=I. Complex conjugation
        of this ket frame and the coefficients defines the trial bra.
        All inputs are finite and must not be modified. Packed arrays have
        final axis [real, imaginary]. NumPy and SciPy are available;
        earlier public functions may be called.

    Returns
    -------
    q_alpha, q_beta : real ndarrays, shape (n,k,2)
        Packed normalized spin orbitals Q with the positive-diagonal QR gauge.
    old_overlap, new_overlap : real ndarrays, shape (2,)
        Packed O(Phi) and O(Q), respectively. new_overlap excludes gauge.
    old_energy, new_energy : real ndarrays, shape (2,)
        Packed E(Phi) and E(Q) in Eh, retaining their imaginary parts.
    old_force : real ndarray, shape (r,2)
        Packed f(Phi) in sqrt(Eh), before multiplication by -sqrt(dt).
    log_gaussian : real ndarray, shape (2,)
        Packed field dot bar_x - (bar_x dot bar_x)/2. Both dots are bilinear.
    gauge : real ndarray, shape (2,)
        Packed det(R_alpha)*det(R_beta) for the positive-diagonal convention.

    Raises
    ------
    ValueError
        If packed shapes or trial/walker/Hamiltonian/field dimensions
        disagree, the dimensional bounds fail, an occupation row is invalid,
        h or a factor is not symmetric, or dt<0. Also raised if the magnitude
        of either total trial overlap is <=1e-12, or any propagated thin-QR
        diagonal magnitude is <=1e-12.
    """
    return q_alpha, q_beta, old_overlap, new_overlap, old_energy, new_energy, old_force, log_gaussian, gauge

# =============================================================================
# GOLD SOLUTION
# =============================================================================

from itertools import combinations
import numpy as np
from scipy.linalg import expm


def _spin_operator_numerators(overlap_matrices, operator_matrices):
    """Return det(S), first derivatives, and factor second derivatives.

    S has shape (nd,k,k). X has shape (nd,1+r,k,k), h first, factors next.
    The second result is d/dt det(S+t X)|0 for each operator. The final
    result is d^2/dt^2 det(S+t X)|0 for each two-electron factor. These
    polynomial derivatives are well-defined at singular S.
    """
    s = overlap_matrices
    x = operator_matrices
    k = s.shape[-1]
    det_s = np.linalg.det(s)
    if k == 1:
        first = x[..., 0, 0].copy()
        return det_s, first, np.zeros_like(first[:, 1:])
    idx = np.arange(k)
    keep = np.array([idx[idx != i] for i in idx], dtype=int)
    minors = s[:, keep[:, None, :, None], keep[None, :, None, :]]
    signs = (-1.0) ** (idx[:, None] + idx[None, :])
    cofactors = signs * np.linalg.det(minors)
    first = np.einsum('dij,dlij->dl', cofactors, x)
    pairs = np.array(list(combinations(range(k), 2)), dtype=int)
    i, j = pairs.T
    keep2 = np.array([idx[(idx != a) & (idx != b)] for a, b in pairs])
    if k == 2:
        second_cofactors = np.ones((len(s), 1, 1), dtype=complex)
    else:
        minors2 = s[:, keep2[:, None, :, None], keep2[None, :, None, :]]
        signs2 = (-1.0) ** (i[:, None] + j[:, None] + i[None, :] + j[None, :])
        second_cofactors = signs2 * np.linalg.det(minors2)
    y = x[:, 1:]
    wedge = (y[:, :, i[:, None], i[None, :]] * y[:, :, j[:, None], j[None, :]]
             - y[:, :, i[:, None], j[None, :]] * y[:, :, j[:, None], i[None, :]])
    second = 2.0 * np.einsum('dij,dlij->dl', second_cofactors, wedge)
    return det_s, first, second


def _mixed_estimators(occupations: "np.ndarray", coefficients: "np.ndarray", phi_alpha: "np.ndarray", phi_beta: "np.ndarray", h: "np.ndarray", factors: "np.ndarray", trial_orbitals: "np.ndarray | None" = None) -> "tuple[np.ndarray, np.ndarray, np.ndarray]":
    occ = np.asarray(occupations)
    cp = np.asarray(coefficients, dtype=float)
    pa = np.asarray(phi_alpha, dtype=float)
    pb = np.asarray(phi_beta, dtype=float)
    h = np.asarray(h, dtype=float)
    factors = np.asarray(factors, dtype=float)
    if h.ndim != 2 or h.shape[0] != h.shape[1] or not 1 <= len(h) <= 28:
        raise ValueError('Require a square one-electron matrix with n <= 28.')
    n = len(h)
    if (occ.ndim != 2 or not 1 <= len(occ) <= 32
            or not np.issubdtype(occ.dtype, np.integer)
            or not 1 <= occ.shape[1] <= min(n, 6)
            or np.any(occ < 0) or np.any(occ >= n)
            or np.any(np.diff(occ, axis=1) <= 0)
            or len(np.unique(occ, axis=0)) != len(occ)):
        raise ValueError('Invalid paired occupations or selected-trial dimensions.')
    if (cp.shape != (len(occ), 2) or pa.shape != (n, occ.shape[1], 2)
            or pb.shape != pa.shape or factors.ndim != 3
            or factors.shape[1:] != (n, n) or not 1 <= len(factors) <= 8):
        raise ValueError('Incompatible trial, walker, or factor shapes.')
    if not np.allclose(h, h.T) or not np.allclose(factors, factors.transpose(0, 2, 1)):
        raise ValueError('Hamiltonian matrices must be symmetric.')
    coeff = cp[:, 0] + 1j * cp[:, 1]
    a = pa[..., 0] + 1j * pa[..., 1]
    b = pb[..., 0] + 1j * pb[..., 1]
    if trial_orbitals is None:
        u = np.eye(n, dtype=complex)
    else:
        up = np.asarray(trial_orbitals, dtype=float)
        if up.shape != (n, n, 2):
            raise ValueError('Require packed trial orbitals with shape (n,n,2).')
        u = up[..., 0] + 1j * up[..., 1]
    udag = u.conj().T
    ops = np.concatenate((h[None], factors), axis=0)
    xa = (udag @ (ops @ a))[:, occ, :].transpose(1, 0, 2, 3)
    xb = (udag @ (ops @ b))[:, occ, :].transpose(1, 0, 2, 3)
    sa, first_a, second_a = _spin_operator_numerators((udag @ a)[occ, :], xa)
    sb, first_b, second_b = _spin_operator_numerators((udag @ b)[occ, :], xb)
    ca = coeff.conjugate()
    overlap = np.dot(ca, sa * sb)
    if abs(overlap) <= 1e-12:
        raise ValueError('Total trial overlap must exceed 1e-12.')
    linear_a, linear_b = first_a[:, 1:], first_b[:, 1:]
    one = sb * first_a[:, 0] + sa * first_b[:, 0]
    two = (sb[:, None] * second_a + sa[:, None] * second_b
           + 2.0 * linear_a * linear_b)
    energy = np.dot(ca, one + 0.5 * np.sum(two, axis=1)) / overlap
    force = 1j * np.einsum('d,dl->l', ca,
                          sb[:, None] * linear_a + sa[:, None] * linear_b) / overlap
    return (np.array([overlap.real, overlap.imag]),
            np.array([energy.real, energy.imag]),
            np.stack((force.real, force.imag), axis=-1))


def _oracle_advance_walker(occupations: "np.ndarray", coefficients: "np.ndarray", phi_alpha: "np.ndarray", phi_beta: "np.ndarray", h: "np.ndarray", factors: "np.ndarray", field: "np.ndarray", dt: float, trial_orbitals: "np.ndarray | None" = None) -> "tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]":
    if dt < 0:
        raise ValueError('Require a nonnegative imaginary-time step.')
    old_overlap, old_energy, old_force = _mixed_estimators(
        occupations, coefficients, phi_alpha, phi_beta, h, factors, trial_orbitals)
    h = np.asarray(h, dtype=float)
    factors = np.asarray(factors, dtype=float)
    field = np.asarray(field, dtype=float)
    if field.shape != (len(factors),):
        raise ValueError('Require one real field coordinate per factor.')
    force = old_force[:, 0] + 1j*old_force[:, 1]
    shift = -np.sqrt(dt)*force
    h0 = h - 0.5*np.einsum('lpq,lqr->pr', factors, factors)
    half = expm(-0.5*dt*h0)
    middle = expm(1j*np.sqrt(dt)*np.einsum('l,lpq->pq', field-shift, factors))
    propagator = half @ middle @ half
    log_gaussian = np.dot(field, shift) - 0.5*np.dot(shift, shift)
    normalized = []
    gauge = 1.0 + 0.0j
    for packed in (phi_alpha, phi_beta):
        packed = np.asarray(packed, dtype=float)
        phi = packed[..., 0] + 1j*packed[..., 1]
        q, r = np.linalg.qr(propagator @ phi, mode='reduced')
        diagonal = np.diag(r)
        if np.min(np.abs(diagonal)) <= 1e-12:
            raise ValueError('Propagated QR diagonal must exceed 1e-12.')
        phases = diagonal/np.abs(diagonal)
        q = q*phases[None, :]
        r = phases.conjugate()[:, None]*r
        gauge *= np.prod(np.diag(r))
        normalized.append(np.stack((q.real, q.imag), axis=-1))
    q_alpha, q_beta = normalized
    new_overlap, new_energy, _ = _mixed_estimators(
        occupations, coefficients, q_alpha, q_beta, h, factors, trial_orbitals)
    return (q_alpha, q_beta, old_overlap, new_overlap, old_energy, new_energy,
            old_force, np.array([log_gaussian.real, log_gaussian.imag]),
            np.array([gauge.real, gauge.imag]))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return normal, boundary, and edge comparisons against the oracle."""
    checks = """
def _checked(fn, *args):
    copied = [arg.copy() if isinstance(arg, np.ndarray) else arg for arg in args]
    before = [arg.copy() if isinstance(arg, np.ndarray) else arg for arg in copied]
    try:
        result = fn(*copied)
    finally:
        for original, current in zip(before, copied):
            if isinstance(original, np.ndarray) and not np.array_equal(original, current):
                raise AssertionError('Input arrays must not be modified')
    expected_shapes = (np.asarray(args[2]).shape, np.asarray(args[3]).shape,
                       (2,), (2,), (2,), (2,), (np.asarray(args[5]).shape[0], 2),
                       (2,), (2,))
    if not isinstance(result, (tuple, list)) or len(result) != len(expected_shapes):
        raise AssertionError('Return structure does not match the documented tuple')
    arrays = [np.asarray(value, dtype=float) for value in result]
    if any(value.shape != shape for value, shape in zip(arrays, expected_shapes)):
        raise AssertionError('Return shapes do not match the documented contract')
    if not all(np.all(np.isfinite(value)) for value in arrays):
        raise AssertionError('Returned values must be finite')
    return np.concatenate([value.ravel() for value in arrays])

def _raises(fn, *args):
    try:
        fn(*args)
    except ValueError:
        return 1
    raise AssertionError('Expected ValueError')
"""
    setup_0 = """import numpy as np

H = np.array([
    [-1.084145, 0.12, -0.08, 0.05],
    [0.12, -0.885553, 0.11, -0.06],
    [-0.08, 0.11, -0.763931, 0.09],
    [0.05, -0.06, 0.09, -0.742439],
], dtype=float)
L = np.array([
    [[0.58, 0.196, -0.084, 0.112],
     [0.196, 0.43, 0.224, -0.056],
     [-0.084, 0.224, 0.36, 0.168],
     [0.112, -0.056, 0.168, 0.29]],
    [[0.16, -0.308, 0.112, 0.084],
     [-0.308, -0.22, 0.140, 0.196],
     [0.112, 0.140, 0.19, -0.224],
     [0.084, 0.196, -0.224, -0.14]],
    [[-0.12, 0.084, 0.252, -0.140],
     [0.084, 0.17, -0.168, 0.056],
     [0.252, -0.168, 0.08, 0.308],
     [-0.140, 0.056, 0.308, -0.10]],
], dtype=float)
BASES = (
    np.array([[1.0, 0.07], [-0.04, 0.92],
              [0.18, -0.14], [-0.12, 0.21]], dtype=float),
    np.array([[0.95, -0.03], [0.06, 1.02],
              [-0.13, 0.17], [0.16, 0.09]], dtype=float),
)
WALKERS = np.empty((4, 2, 4, 2, 2), dtype=float)
for w in range(4):
    for spin, base in enumerate(BASES):
        for p in range(4):
            for j in range(2):
                z = (base[p, j]
                     + 0.035*w*np.cos((p + 1)*(j + 2))
                     + 0.020j*(w + spin)*np.sin((p + 2)*(j + 1)))
                WALKERS[w, spin, p, j] = (z.real, z.imag)
WEIGHTS = np.array([1.0, 0.8, 1.2, 0.9], dtype=float)
FIELDS = np.empty((7, 4, 3), dtype=float)
for t in range(7):
    for w in range(4):
        for ell in range(3):
            FIELDS[t, w, ell] = (
                0.85*np.sin((t + 1)*(w + 2)*(ell + 1))
                + 0.35*np.cos((t + 2)*(ell + 2) + w)
            )
NPAIR = 2
DT = 0.08
ENERGY_SHIFT = -2.1
ARGS = (H, L, NPAIR, WALKERS, WEIGHTS, FIELDS, DT, ENERGY_SHIFT)


def pack(z):
    z = np.asarray(z, dtype=complex)
    return np.stack((z.real, z.imag), axis=-1)

def _raises(fn, *args):
    try:
        fn(*args)
    except ValueError:
        return 1
    raise AssertionError('Expected ValueError')

OCC = np.array([[0,1],[0,2],[0,3],[1,2],[1,3],[2,3]], dtype=int)
C = pack(np.array([0.6641087358712602, 0.044491721035112175,
                   -0.3682035944352817, -0.31448486148478577,
                   0.055587573647215255, 0.5651685414959503]))
"""
    setup_1 = """import numpy as np

H = np.array([
    [-1.084145, 0.12, -0.08, 0.05],
    [0.12, -0.885553, 0.11, -0.06],
    [-0.08, 0.11, -0.763931, 0.09],
    [0.05, -0.06, 0.09, -0.742439],
], dtype=float)
L = np.array([
    [[0.58, 0.196, -0.084, 0.112],
     [0.196, 0.43, 0.224, -0.056],
     [-0.084, 0.224, 0.36, 0.168],
     [0.112, -0.056, 0.168, 0.29]],
    [[0.16, -0.308, 0.112, 0.084],
     [-0.308, -0.22, 0.140, 0.196],
     [0.112, 0.140, 0.19, -0.224],
     [0.084, 0.196, -0.224, -0.14]],
    [[-0.12, 0.084, 0.252, -0.140],
     [0.084, 0.17, -0.168, 0.056],
     [0.252, -0.168, 0.08, 0.308],
     [-0.140, 0.056, 0.308, -0.10]],
], dtype=float)
BASES = (
    np.array([[1.0, 0.07], [-0.04, 0.92],
              [0.18, -0.14], [-0.12, 0.21]], dtype=float),
    np.array([[0.95, -0.03], [0.06, 1.02],
              [-0.13, 0.17], [0.16, 0.09]], dtype=float),
)
WALKERS = np.empty((4, 2, 4, 2, 2), dtype=float)
for w in range(4):
    for spin, base in enumerate(BASES):
        for p in range(4):
            for j in range(2):
                z = (base[p, j]
                     + 0.035*w*np.cos((p + 1)*(j + 2))
                     + 0.020j*(w + spin)*np.sin((p + 2)*(j + 1)))
                WALKERS[w, spin, p, j] = (z.real, z.imag)
WEIGHTS = np.array([1.0, 0.8, 1.2, 0.9], dtype=float)
FIELDS = np.empty((7, 4, 3), dtype=float)
for t in range(7):
    for w in range(4):
        for ell in range(3):
            FIELDS[t, w, ell] = (
                0.85*np.sin((t + 1)*(w + 2)*(ell + 1))
                + 0.35*np.cos((t + 2)*(ell + 2) + w)
            )
NPAIR = 2
DT = 0.08
ENERGY_SHIFT = -2.1
ARGS = (H, L, NPAIR, WALKERS, WEIGHTS, FIELDS, DT, ENERGY_SHIFT)


def pack(z):
    z = np.asarray(z, dtype=complex)
    return np.stack((z.real, z.imag), axis=-1)

def _raises(fn, *args):
    try:
        fn(*args)
    except ValueError:
        return 1
    raise AssertionError('Expected ValueError')

OCC = np.array([[0,1],[0,2],[0,3],[1,2],[1,3],[2,3]], dtype=int)
C = pack(np.array([0.6641087358712602, 0.044491721035112175,
                   -0.3682035944352817, -0.31448486148478577,
                   0.055587573647215255, 0.5651685414959503]))

CC=pack((C[:,0]+1j*np.linspace(-.2,.3,len(C)))*np.exp(.61j))
"""
    setup_2 = """import numpy as np

H = np.array([
    [-1.084145, 0.12, -0.08, 0.05],
    [0.12, -0.885553, 0.11, -0.06],
    [-0.08, 0.11, -0.763931, 0.09],
    [0.05, -0.06, 0.09, -0.742439],
], dtype=float)
L = np.array([
    [[0.58, 0.196, -0.084, 0.112],
     [0.196, 0.43, 0.224, -0.056],
     [-0.084, 0.224, 0.36, 0.168],
     [0.112, -0.056, 0.168, 0.29]],
    [[0.16, -0.308, 0.112, 0.084],
     [-0.308, -0.22, 0.140, 0.196],
     [0.112, 0.140, 0.19, -0.224],
     [0.084, 0.196, -0.224, -0.14]],
    [[-0.12, 0.084, 0.252, -0.140],
     [0.084, 0.17, -0.168, 0.056],
     [0.252, -0.168, 0.08, 0.308],
     [-0.140, 0.056, 0.308, -0.10]],
], dtype=float)
BASES = (
    np.array([[1.0, 0.07], [-0.04, 0.92],
              [0.18, -0.14], [-0.12, 0.21]], dtype=float),
    np.array([[0.95, -0.03], [0.06, 1.02],
              [-0.13, 0.17], [0.16, 0.09]], dtype=float),
)
WALKERS = np.empty((4, 2, 4, 2, 2), dtype=float)
for w in range(4):
    for spin, base in enumerate(BASES):
        for p in range(4):
            for j in range(2):
                z = (base[p, j]
                     + 0.035*w*np.cos((p + 1)*(j + 2))
                     + 0.020j*(w + spin)*np.sin((p + 2)*(j + 1)))
                WALKERS[w, spin, p, j] = (z.real, z.imag)
WEIGHTS = np.array([1.0, 0.8, 1.2, 0.9], dtype=float)
FIELDS = np.empty((7, 4, 3), dtype=float)
for t in range(7):
    for w in range(4):
        for ell in range(3):
            FIELDS[t, w, ell] = (
                0.85*np.sin((t + 1)*(w + 2)*(ell + 1))
                + 0.35*np.cos((t + 2)*(ell + 2) + w)
            )
NPAIR = 2
DT = 0.08
ENERGY_SHIFT = -2.1
ARGS = (H, L, NPAIR, WALKERS, WEIGHTS, FIELDS, DT, ENERGY_SHIFT)


def pack(z):
    z = np.asarray(z, dtype=complex)
    return np.stack((z.real, z.imag), axis=-1)

def _raises(fn, *args):
    try:
        fn(*args)
    except ValueError:
        return 1
    raise AssertionError('Expected ValueError')

OCC = np.array([[0,1],[0,2],[0,3],[1,2],[1,3],[2,3]], dtype=int)
C = pack(np.array([0.6641087358712602, 0.044491721035112175,
                   -0.3682035944352817, -0.31448486148478577,
                   0.055587573647215255, 0.5651685414959503]))

from itertools import combinations
OCC3 = np.array(list(combinations(range(5), 3)), dtype=int)
C3 = pack(np.array([0.7, -0.2+0.1j, 0.15, 0.12j, -0.05,
                    0.21-0.08j, 0.03, -0.13j, 0.08, 0.06]))
A3 = np.array([[1.0+0.1j,0.2,-0.1j], [0.1,0.9-0.1j,0.15],
               [-0.2j,0.05,1.1], [0.12,-0.14j,0.08],
               [-0.07j,0.16,-0.09]], dtype=complex)
B3 = np.array([[0.9,0.1j,0.13], [-0.1,1.05+0.15j,-0.07j],
               [0.2j,0.1,0.85-0.1j], [-0.16,0.12,0.18j],
               [0.14j,-0.05,0.11]], dtype=complex)
GRID = np.arange(25, dtype=float).reshape(5,5)
H3 = np.diag([-1.2,-0.95,-0.75,-0.5,-0.3])
H3 += 0.04*(np.sin(GRID)+np.sin(GRID).T)
L3 = np.array([0.08*(np.cos(GRID)+np.cos(GRID).T)+0.2*np.eye(5),
               0.06*(np.sin(0.7*GRID)+np.sin(0.7*GRID).T)-0.1*np.eye(5)])
GA = np.array([[1.1+0.2j,0.2,-0.1j],[0.1j,0.9-0.1j,0.15],
               [0.0,-0.2j,1.05]], dtype=complex)
GB = np.array([[0.8-0.1j,-0.15j,0.2],[0.1,1.2+0.1j,0.0],
               [0.05j,0.1,0.95-0.2j]], dtype=complex)
"""
    setup_3 = """import numpy as np

def pack(z):
    z = np.asarray(z, dtype=complex)
    return np.stack((z.real, z.imag), axis=-1)

E = np.eye(6, dtype=complex)
GRID = np.arange(36, dtype=float).reshape(6, 6)
H = np.diag([-1.3, -1.0, -0.8, -0.55, -0.35, -0.1])
H += 0.07*(np.sin(GRID)+np.sin(GRID).T)
L = np.array([0.12*(np.cos(GRID)+np.cos(GRID).T)+0.2*np.eye(6),
              0.09*(np.sin(0.7*GRID)+np.sin(0.7*GRID).T)-0.1*np.eye(6)])
GA = np.array([[1.1+0.2j, 0.2, -0.1j], [0.1j, 0.9-0.1j, 0.15],
               [0.0, -0.2j, 1.05]], dtype=complex)
GB = np.array([[0.8-0.1j, -0.15j, 0.2], [0.1, 1.2+0.1j, 0.0],
               [0.05j, 0.1, 0.95-0.2j]], dtype=complex)
C = pack(np.array([0.5+0.2j, 0.7-0.1j]))
OCC = np.array([[0,1,2], [0,1,3]])
A = E[:, :3]
B = E[:, :3].copy()
B[:, 2] += 0.3*E[:, 3]
"""
    setup_4 = """import numpy as np

def pack(z):
    z = np.asarray(z, dtype=complex)
    return np.stack((z.real, z.imag), axis=-1)

E = np.eye(6, dtype=complex)
GRID = np.arange(36, dtype=float).reshape(6, 6)
H = np.diag([-1.3, -1.0, -0.8, -0.55, -0.35, -0.1])
H += 0.07*(np.sin(GRID)+np.sin(GRID).T)
L = np.array([0.12*(np.cos(GRID)+np.cos(GRID).T)+0.2*np.eye(6),
              0.09*(np.sin(0.7*GRID)+np.sin(0.7*GRID).T)-0.1*np.eye(6)])
GA = np.array([[1.1+0.2j, 0.2, -0.1j], [0.1j, 0.9-0.1j, 0.15],
               [0.0, -0.2j, 1.05]], dtype=complex)
GB = np.array([[0.8-0.1j, -0.15j, 0.2], [0.1, 1.2+0.1j, 0.0],
               [0.05j, 0.1, 0.95-0.2j]], dtype=complex)
C = pack(np.array([0.5+0.2j, 0.7-0.1j]))
OCC = np.array([[0,1,2], [0,3,4]])
A = E[:, [0,3,4]]
B = E[:, :3].copy()
B[:, 1] += 0.3*E[:, 3]
B[:, 2] += 0.4*E[:, 4]
"""
    setup_5 = """import numpy as np

def pack(z):
    z = np.asarray(z, dtype=complex)
    return np.stack((z.real, z.imag), axis=-1)

E = np.eye(6, dtype=complex)
GRID = np.arange(36, dtype=float).reshape(6, 6)
H = np.diag([-1.3, -1.0, -0.8, -0.55, -0.35, -0.1])
H += 0.07*(np.sin(GRID)+np.sin(GRID).T)
L = np.array([0.12*(np.cos(GRID)+np.cos(GRID).T)+0.2*np.eye(6),
              0.09*(np.sin(0.7*GRID)+np.sin(0.7*GRID).T)-0.1*np.eye(6)])
GA = np.array([[1.1+0.2j, 0.2, -0.1j], [0.1j, 0.9-0.1j, 0.15],
               [0.0, -0.2j, 1.05]], dtype=complex)
GB = np.array([[0.8-0.1j, -0.15j, 0.2], [0.1, 1.2+0.1j, 0.0],
               [0.05j, 0.1, 0.95-0.2j]], dtype=complex)
C = pack(np.array([0.5+0.2j, 0.7-0.1j]))
OCC = np.array([[0,1,2], [0,1,3]])
A = E[:, :3]
B = E[:, :3]
"""
    setup_6 = """import numpy as np

def pack(z):
    z = np.asarray(z, dtype=complex)
    return np.stack((z.real, z.imag), axis=-1)

E = np.eye(6, dtype=complex)
GRID = np.arange(36, dtype=float).reshape(6, 6)
H = np.diag([-1.3, -1.0, -0.8, -0.55, -0.35, -0.1])
H += 0.07*(np.sin(GRID)+np.sin(GRID).T)
L = np.array([0.12*(np.cos(GRID)+np.cos(GRID).T)+0.2*np.eye(6),
              0.09*(np.sin(0.7*GRID)+np.sin(0.7*GRID).T)-0.1*np.eye(6)])
GA = np.array([[1.1+0.2j, 0.2, -0.1j], [0.1j, 0.9-0.1j, 0.15],
               [0.0, -0.2j, 1.05]], dtype=complex)
GB = np.array([[0.8-0.1j, -0.15j, 0.2], [0.1, 1.2+0.1j, 0.0],
               [0.05j, 0.1, 0.95-0.2j]], dtype=complex)
C = pack(np.array([0.5+0.2j, 0.7-0.1j]))
OCC = np.array([[0,1,2], [0,3,4]])
A = E[:, [0,3,4]]
B = E[:, [0,1,4]].copy()
B[:, 1] += 0.3*E[:, 3]
B[:, 2] += 0.4*E[:, 5]
"""
    setup_7 = """import numpy as np

def pack(z):
    z = np.asarray(z, dtype=complex)
    return np.stack((z.real, z.imag), axis=-1)

E = np.eye(6, dtype=complex)
GRID = np.arange(36, dtype=float).reshape(6, 6)
H = np.diag([-1.3, -1.0, -0.8, -0.55, -0.35, -0.1])
H += 0.07*(np.sin(GRID)+np.sin(GRID).T)
L = np.array([0.12*(np.cos(GRID)+np.cos(GRID).T)+0.2*np.eye(6),
              0.09*(np.sin(0.7*GRID)+np.sin(0.7*GRID).T)-0.1*np.eye(6)])
GA = np.array([[1.1+0.2j, 0.2, -0.1j], [0.1j, 0.9-0.1j, 0.15],
               [0.0, -0.2j, 1.05]], dtype=complex)
GB = np.array([[0.8-0.1j, -0.15j, 0.2], [0.1, 1.2+0.1j, 0.0],
               [0.05j, 0.1, 0.95-0.2j]], dtype=complex)
C = pack(np.array([0.5+0.2j, 0.7-0.1j]))
OCC = np.array([[0,1,2], [0,3,4]])
A = E[:, [0,3,4]]
B = E[:, [0,3,4]]
"""
    setup_8 = """import numpy as np

def pack(z):
    z = np.asarray(z, dtype=complex)
    return np.stack((z.real, z.imag), axis=-1)

E = np.eye(6, dtype=complex)
GRID = np.arange(36, dtype=float).reshape(6, 6)
H = np.diag([-1.3, -1.0, -0.8, -0.55, -0.35, -0.1])
H += 0.07*(np.sin(GRID)+np.sin(GRID).T)
L = np.array([0.12*(np.cos(GRID)+np.cos(GRID).T)+0.2*np.eye(6),
              0.09*(np.sin(0.7*GRID)+np.sin(0.7*GRID).T)-0.1*np.eye(6)])
GA = np.array([[1.1+0.2j, 0.2, -0.1j], [0.1j, 0.9-0.1j, 0.15],
               [0.0, -0.2j, 1.05]], dtype=complex)
GB = np.array([[0.8-0.1j, -0.15j, 0.2], [0.1, 1.2+0.1j, 0.0],
               [0.05j, 0.1, 0.95-0.2j]], dtype=complex)
C = pack(np.array([0.5+0.2j, 0.7-0.1j]))
OCC = np.array([[0,3,4], [0,1,2]])
C = C[::-1].copy()
A = E[:, [0,3,4]]
B = E[:, :3].copy()
B[:, 1] += 0.3*E[:, 3]
B[:, 2] += 0.4*E[:, 4]
"""
    setup_9 = """import numpy as np

def pack(z):
    z = np.asarray(z, dtype=complex)
    return np.stack((z.real, z.imag), axis=-1)

E = np.eye(6, dtype=complex)
g = np.arange(36, dtype=float).reshape(6, 6)
U, _ = np.linalg.qr(np.eye(6) + 0.23*np.sin(g + 0.4)
                    + 0.19j*np.cos(0.7*g + 0.2))
H = np.diag([-1.3, -1.0, -0.8, -0.55, -0.35, -0.1])
H += 0.07*(np.sin(g) + np.sin(g).T)
L = np.array([0.12*(np.cos(g) + np.cos(g).T) + 0.2*np.eye(6),
              0.09*(np.sin(0.7*g) + np.sin(0.7*g).T) - 0.1*np.eye(6)])
GA = np.array([[1.1+0.2j, 0.2, -0.1j], [0.1j, 0.9-0.1j, 0.15],
               [0.0, -0.2j, 1.05]], dtype=complex)
GB = np.array([[0.8-0.1j, -0.15j, 0.2], [0.1, 1.2+0.1j, 0.0],
               [0.05j, 0.1, 0.95-0.2j]], dtype=complex)
C = pack(np.array([0.5+0.2j, 0.7-0.1j]))

from itertools import combinations
OCC = np.array(list(combinations(range(6), 3)), dtype=int)
C = pack(np.array([0.8 if d == 0 else 0.12*np.cos(d) + 0.09j*np.sin(2*d)
                   for d in range(len(OCC))]))
gk = np.arange(18, dtype=float).reshape(6, 3)
A = (E[:, :3] + 0.08*np.sin(gk + 0.3) + 0.06j*np.cos(gk)) @ GA
B = (E[:, :3] + 0.11*np.cos(gk + 0.7) + 0.04j*np.sin(0.8*gk)) @ GB
"""
    setup_10 = """import numpy as np

def pack(z):
    z = np.asarray(z, dtype=complex)
    return np.stack((z.real, z.imag), axis=-1)

E = np.eye(6, dtype=complex)
g = np.arange(36, dtype=float).reshape(6, 6)
U, _ = np.linalg.qr(np.eye(6) + 0.23*np.sin(g + 0.4)
                    + 0.19j*np.cos(0.7*g + 0.2))
H = np.diag([-1.3, -1.0, -0.8, -0.55, -0.35, -0.1])
H += 0.07*(np.sin(g) + np.sin(g).T)
L = np.array([0.12*(np.cos(g) + np.cos(g).T) + 0.2*np.eye(6),
              0.09*(np.sin(0.7*g) + np.sin(0.7*g).T) - 0.1*np.eye(6)])
GA = np.array([[1.1+0.2j, 0.2, -0.1j], [0.1j, 0.9-0.1j, 0.15],
               [0.0, -0.2j, 1.05]], dtype=complex)
GB = np.array([[0.8-0.1j, -0.15j, 0.2], [0.1, 1.2+0.1j, 0.0],
               [0.05j, 0.1, 0.95-0.2j]], dtype=complex)
C = pack(np.array([0.5+0.2j, 0.7-0.1j]))

OCC = np.array([[0,1,2], [0,1,3]], dtype=int)
A = U @ E[:, :3] @ GA
B = U @ E[:, :3] @ GB
"""
    setup_11 = """import numpy as np

def pack(z):
    z = np.asarray(z, dtype=complex)
    return np.stack((z.real, z.imag), axis=-1)

E = np.eye(6, dtype=complex)
g = np.arange(36, dtype=float).reshape(6, 6)
U, _ = np.linalg.qr(np.eye(6) + 0.23*np.sin(g + 0.4)
                    + 0.19j*np.cos(0.7*g + 0.2))
H = np.diag([-1.3, -1.0, -0.8, -0.55, -0.35, -0.1])
H += 0.07*(np.sin(g) + np.sin(g).T)
L = np.array([0.12*(np.cos(g) + np.cos(g).T) + 0.2*np.eye(6),
              0.09*(np.sin(0.7*g) + np.sin(0.7*g).T) - 0.1*np.eye(6)])
GA = np.array([[1.1+0.2j, 0.2, -0.1j], [0.1j, 0.9-0.1j, 0.15],
               [0.0, -0.2j, 1.05]], dtype=complex)
GB = np.array([[0.8-0.1j, -0.15j, 0.2], [0.1, 1.2+0.1j, 0.0],
               [0.05j, 0.1, 0.95-0.2j]], dtype=complex)
C = pack(np.array([0.5+0.2j, 0.7-0.1j]))

OCC = np.array([[0,1,2], [0,3,4]], dtype=int)
AA = E[:, [0,3,4]].copy()
BB = E[:, :3].copy()
BB[:, 1] += 0.3*E[:, 3]
BB[:, 2] += 0.4*E[:, 4]
A = U @ AA @ GA
B = U @ BB @ GB
"""
    setup_12 = """import numpy as np

def pack(z):
    z = np.asarray(z, dtype=complex)
    return np.stack((z.real, z.imag), axis=-1)

E = np.eye(6, dtype=complex)
g = np.arange(36, dtype=float).reshape(6, 6)
U, _ = np.linalg.qr(np.eye(6) + 0.23*np.sin(g + 0.4)
                    + 0.19j*np.cos(0.7*g + 0.2))
H = np.diag([-1.3, -1.0, -0.8, -0.55, -0.35, -0.1])
H += 0.07*(np.sin(g) + np.sin(g).T)
L = np.array([0.12*(np.cos(g) + np.cos(g).T) + 0.2*np.eye(6),
              0.09*(np.sin(0.7*g) + np.sin(0.7*g).T) - 0.1*np.eye(6)])
GA = np.array([[1.1+0.2j, 0.2, -0.1j], [0.1j, 0.9-0.1j, 0.15],
               [0.0, -0.2j, 1.05]], dtype=complex)
GB = np.array([[0.8-0.1j, -0.15j, 0.2], [0.1, 1.2+0.1j, 0.0],
               [0.05j, 0.1, 0.95-0.2j]], dtype=complex)
C = pack(np.array([0.5+0.2j, 0.7-0.1j]))

OCC = np.array([[0,1,2], [0,3,4]], dtype=int)
AA = E[:, [0,3,4]].copy()
BB = E[:, :3].copy()
BB[:, 1] += 0.3*E[:, 3]
BB[:, 2] += 0.4*E[:, 4]
A = U @ AA @ GA
B = U @ BB @ GB

AA[:, 1] += 2e-7*E[:, 1]
AA[:, 2] += 3e-7*E[:, 2]
A = U @ AA @ GA
"""
    setup_13 = """import numpy as np

def pack(z):
    z = np.asarray(z, dtype=complex)
    return np.stack((z.real, z.imag), axis=-1)

n, k = 28, 6
E = np.eye(n, dtype=complex)
q = np.arange(n, dtype=float)
U = np.exp(2j*np.pi*q[:,None]*q[None,:]/n)/np.sqrt(n)
OCC = np.array([list(range(6))]
               + [list(range(5)) + [p] for p in range(6,28)]
               + [list(range(4)) + [6,p] for p in range(7,16)], dtype=int)
d = np.arange(len(OCC), dtype=float)
C = pack(np.where(d == 0, 0.8, 0.045*np.cos(d) + 0.03j*np.sin(2*d)))
g = np.arange(n*n, dtype=float).reshape(n,n)
gk = np.arange(n*k, dtype=float).reshape(n,k)
H = np.diag(np.linspace(-1.25,-0.2,n))
H += 0.014*(np.sin(0.7*g)+np.sin(0.7*g).T)
L = np.array([0.013*(np.cos((ell+1)*g/17)+np.cos((ell+1)*g/17).T)
              + (0.08-0.013*ell)*np.eye(n) for ell in range(8)])
AA = E[:, :k] + 0.025*np.sin(gk + 0.4) + 0.017j*np.cos(0.7*gk)
BB = E[:, :k] + 0.03*np.cos(gk + 0.7) + 0.013j*np.sin(0.9*gk)
GA = np.eye(k) + 0.018j*np.arange(k*k).reshape(k,k)/k
GB = np.eye(k) + 0.02*np.sin(np.arange(k*k).reshape(k,k))
A = U @ AA @ GA
B = U @ BB @ GB
"""
    setup_14 = """import numpy as np

def pack(z):
    z = np.asarray(z, dtype=complex)
    return np.stack((z.real, z.imag), axis=-1)

E = np.eye(6, dtype=complex)
g = np.arange(36, dtype=float).reshape(6, 6)
U, _ = np.linalg.qr(np.eye(6) + 0.23*np.sin(g + 0.4)
                    + 0.19j*np.cos(0.7*g + 0.2))
H = np.diag([-1.3, -1.0, -0.8, -0.55, -0.35, -0.1])
H += 0.07*(np.sin(g) + np.sin(g).T)
L = np.array([0.12*(np.cos(g) + np.cos(g).T) + 0.2*np.eye(6),
              0.09*(np.sin(0.7*g) + np.sin(0.7*g).T) - 0.1*np.eye(6)])
GA = np.array([[1.1+0.2j, 0.2, -0.1j], [0.1j, 0.9-0.1j, 0.15],
               [0.0, -0.2j, 1.05]], dtype=complex)
GB = np.array([[0.8-0.1j, -0.15j, 0.2], [0.1, 1.2+0.1j, 0.0],
               [0.05j, 0.1, 0.95-0.2j]], dtype=complex)
C = pack(np.array([0.5+0.2j, 0.7-0.1j]))

OCC = np.array([[0], [2], [4]], dtype=int)
C = pack(np.array([0.8+0.2j, -0.25j, 0.31-0.1j]))
A = np.array([[1.0], [0.11j], [-0.2], [0.15-0.07j], [0.3j], [-0.12]])
B = np.array([[0.9+0.2j], [-0.14], [0.25j], [0.17], [-0.16j], [0.21]])
"""
    setup_15 = """import numpy as np

def pack(z):
    z = np.asarray(z, dtype=complex)
    return np.stack((z.real, z.imag), axis=-1)

OCC = np.array([[0]], dtype=int)
C = pack(np.array([1.0]))
A = pack(np.array([[1.0], [0.0]]))
B = A.copy()
H = 50.0*np.eye(2)
L = np.zeros((1, 2, 2))
FIELD = np.zeros(1)

"""
    setup_16 = """import numpy as np

def pack(z):
    z = np.asarray(z, dtype=complex)
    return np.stack((z.real, z.imag), axis=-1)

OCC = np.array([[0]], dtype=int)
C = pack(np.array([1.0]))
A = pack(np.array([[1.0], [0.0]]))
B = A.copy()
H = 0.5*np.eye(2)
L = np.array([[[0.0, 1.0], [1.0, 0.0]]])
FIELD = np.array([np.pi/2])

"""
    return [
        # Preserved estimator fixture 1, zero-time propagation
        {
            "setup": setup_0 + checks,
            "call": '_checked(advance_walker, OCC, C, WALKERS[0, 0], WALKERS[0, 1], H, L, np.zeros(len(L), dtype=float), 0.0)',
            "gold_call": '_checked(_oracle_advance_walker, OCC, C, WALKERS[0, 0], WALKERS[0, 1], H, L, np.zeros(len(L), dtype=float), 0.0)',
            "tol": 2e-08,
        },
        # Preserved estimator fixture 2, zero-time propagation
        {
            "setup": setup_0 + checks,
            "call": '_checked(advance_walker, OCC, C, pack(np.eye(4)[:, :2]), pack(np.eye(4)[:, :2]), H, L, np.zeros(len(L), dtype=float), 0.0)',
            "gold_call": '_checked(_oracle_advance_walker, OCC, C, pack(np.eye(4)[:, :2]), pack(np.eye(4)[:, :2]), H, L, np.zeros(len(L), dtype=float), 0.0)',
            "tol": 2e-08,
        },
        # Preserved estimator fixture 3, zero-time propagation
        {
            "setup": setup_1 + checks,
            "call": '_checked(advance_walker, OCC, CC, WALKERS[2, 0], WALKERS[1, 1], H, L, np.zeros(len(L), dtype=float), 0.0)',
            "gold_call": '_checked(_oracle_advance_walker, OCC, CC, WALKERS[2, 0], WALKERS[1, 1], H, L, np.zeros(len(L), dtype=float), 0.0)',
            "tol": 2e-08,
        },
        # Preserved estimator fixture 4, zero-time propagation
        {
            "setup": setup_0 + checks,
            "call": '_checked(advance_walker, np.array([[0, 1]]), pack(np.array([0.3 + 0.2j])), pack(np.array([[1.0, 0.2j], [0.1, 0.7]])), pack(np.array([[0.9, 0.3], [-0.2j, 1.1]])), H[:2, :2], L[:, :2, :2], np.zeros(len(L[:, :2, :2]), dtype=float), 0.0)',
            "gold_call": '_checked(_oracle_advance_walker, np.array([[0, 1]]), pack(np.array([0.3 + 0.2j])), pack(np.array([[1.0, 0.2j], [0.1, 0.7]])), pack(np.array([[0.9, 0.3], [-0.2j, 1.1]])), H[:2, :2], L[:, :2, :2], np.zeros(len(L[:, :2, :2]), dtype=float), 0.0)',
            "tol": 2e-08,
        },
        # Preserved estimator fixture 5, zero-time propagation
        {
            "setup": setup_0 + checks,
            "call": '_raises(_checked, advance_walker, OCC, np.zeros_like(C), WALKERS[0, 0], WALKERS[0, 1], H, L, np.zeros(len(L), dtype=float), 0.0)',
            "gold_call": '_raises(_checked, _oracle_advance_walker, OCC, np.zeros_like(C), WALKERS[0, 0], WALKERS[0, 1], H, L, np.zeros(len(L), dtype=float), 0.0)',
            "tol": 2e-08,
        },
        # Preserved estimator fixture 6, zero-time propagation
        {
            "setup": setup_2 + checks,
            "call": '_checked(advance_walker, OCC3, C3, pack(A3), pack(B3), H3, L3, np.zeros(len(L3), dtype=float), 0.0)',
            "gold_call": '_checked(_oracle_advance_walker, OCC3, C3, pack(A3), pack(B3), H3, L3, np.zeros(len(L3), dtype=float), 0.0)',
            "tol": 2e-08,
        },
        # Preserved estimator fixture 7, zero-time propagation
        {
            "setup": setup_2 + checks,
            "call": '_checked(advance_walker, OCC3, C3, pack(np.eye(5, 3)), pack(np.eye(5, 3)), H3, L3, np.zeros(len(L3), dtype=float), 0.0)',
            "gold_call": '_checked(_oracle_advance_walker, OCC3, C3, pack(np.eye(5, 3)), pack(np.eye(5, 3)), H3, L3, np.zeros(len(L3), dtype=float), 0.0)',
            "tol": 2e-08,
        },
        # Preserved estimator fixture 8, zero-time propagation
        {
            "setup": setup_2 + checks,
            "call": '_checked(advance_walker, OCC3, C3, pack(A3 @ GA), pack(B3 @ GB), H3, L3, np.zeros(len(L3), dtype=float), 0.0)',
            "gold_call": '_checked(_oracle_advance_walker, OCC3, C3, pack(A3 @ GA), pack(B3 @ GB), H3, L3, np.zeros(len(L3), dtype=float), 0.0)',
            "tol": 2e-08,
        },
        # Preserved estimator fixture 9, zero-time propagation
        {
            "setup": setup_3 + checks,
            "call": '_checked(advance_walker, OCC, C, pack(A @ GA), pack(B @ GB), H, L, np.zeros(len(L), dtype=float), 0.0)',
            "gold_call": '_checked(_oracle_advance_walker, OCC, C, pack(A @ GA), pack(B @ GB), H, L, np.zeros(len(L), dtype=float), 0.0)',
            "tol": 2e-08,
        },
        # Preserved estimator fixture 10, zero-time propagation
        {
            "setup": setup_4 + checks,
            "call": '_checked(advance_walker, OCC, C, pack(A @ GA), pack(B @ GB), H, L, np.zeros(len(L), dtype=float), 0.0)',
            "gold_call": '_checked(_oracle_advance_walker, OCC, C, pack(A @ GA), pack(B @ GB), H, L, np.zeros(len(L), dtype=float), 0.0)',
            "tol": 2e-08,
        },
        # Preserved estimator fixture 11, zero-time propagation
        {
            "setup": setup_5 + checks,
            "call": '_checked(advance_walker, OCC, C, pack(A @ GA), pack(B @ GB), H, L, np.zeros(len(L), dtype=float), 0.0)',
            "gold_call": '_checked(_oracle_advance_walker, OCC, C, pack(A @ GA), pack(B @ GB), H, L, np.zeros(len(L), dtype=float), 0.0)',
            "tol": 2e-08,
        },
        # Preserved estimator fixture 12, zero-time propagation
        {
            "setup": setup_6 + checks,
            "call": '_checked(advance_walker, OCC, C, pack(A @ GA), pack(B @ GB), H, L, np.zeros(len(L), dtype=float), 0.0)',
            "gold_call": '_checked(_oracle_advance_walker, OCC, C, pack(A @ GA), pack(B @ GB), H, L, np.zeros(len(L), dtype=float), 0.0)',
            "tol": 2e-08,
        },
        # Preserved estimator fixture 13, zero-time propagation
        {
            "setup": setup_7 + checks,
            "call": '_checked(advance_walker, OCC, C, pack(A @ GA), pack(B @ GB), H, L, np.zeros(len(L), dtype=float), 0.0)',
            "gold_call": '_checked(_oracle_advance_walker, OCC, C, pack(A @ GA), pack(B @ GB), H, L, np.zeros(len(L), dtype=float), 0.0)',
            "tol": 2e-08,
        },
        # Preserved estimator fixture 14, zero-time propagation
        {
            "setup": setup_8 + checks,
            "call": '_checked(advance_walker, OCC, C, pack(A @ GA), pack(B @ GB), H, L, np.zeros(len(L), dtype=float), 0.0)',
            "gold_call": '_checked(_oracle_advance_walker, OCC, C, pack(A @ GA), pack(B @ GB), H, L, np.zeros(len(L), dtype=float), 0.0)',
            "tol": 2e-08,
        },
        # Preserved estimator fixture 15, zero-time propagation
        {
            "setup": setup_9 + checks,
            "call": '_checked(advance_walker, OCC, C, pack(A), pack(B), H, L, np.zeros(len(L), dtype=float), 0.0, pack(U))',
            "gold_call": '_checked(_oracle_advance_walker, OCC, C, pack(A), pack(B), H, L, np.zeros(len(L), dtype=float), 0.0, pack(U))',
            "tol": 2e-08,
        },
        # Preserved estimator fixture 16, zero-time propagation
        {
            "setup": setup_10 + checks,
            "call": '_checked(advance_walker, OCC, C, pack(A), pack(B), H, L, np.zeros(len(L), dtype=float), 0.0, pack(U))',
            "gold_call": '_checked(_oracle_advance_walker, OCC, C, pack(A), pack(B), H, L, np.zeros(len(L), dtype=float), 0.0, pack(U))',
            "tol": 2e-08,
        },
        # Preserved estimator fixture 17, zero-time propagation
        {
            "setup": setup_11 + checks,
            "call": '_checked(advance_walker, OCC, C, pack(A), pack(B), H, L, np.zeros(len(L), dtype=float), 0.0, pack(U))',
            "gold_call": '_checked(_oracle_advance_walker, OCC, C, pack(A), pack(B), H, L, np.zeros(len(L), dtype=float), 0.0, pack(U))',
            "tol": 2e-08,
        },
        # Preserved estimator fixture 18, zero-time propagation
        {
            "setup": setup_12 + checks,
            "call": '_checked(advance_walker, OCC, C, pack(A), pack(B), H, L, np.zeros(len(L), dtype=float), 0.0, pack(U))',
            "gold_call": '_checked(_oracle_advance_walker, OCC, C, pack(A), pack(B), H, L, np.zeros(len(L), dtype=float), 0.0, pack(U))',
            "tol": 2e-08,
        },
        # Preserved estimator fixture 19, zero-time propagation
        {
            "setup": setup_13 + checks,
            "call": '_checked(advance_walker, OCC, C, pack(A), pack(B), H, L, np.zeros(len(L), dtype=float), 0.0, pack(U))',
            "gold_call": '_checked(_oracle_advance_walker, OCC, C, pack(A), pack(B), H, L, np.zeros(len(L), dtype=float), 0.0, pack(U))',
            "tol": 2e-08,
        },
        # Preserved estimator fixture 20, zero-time propagation
        {
            "setup": setup_14 + checks,
            "call": '_checked(advance_walker, OCC, C, pack(A), pack(B), H, L, np.zeros(len(L), dtype=float), 0.0, pack(U))',
            "gold_call": '_checked(_oracle_advance_walker, OCC, C, pack(A), pack(B), H, L, np.zeros(len(L), dtype=float), 0.0, pack(U))',
            "tol": 2e-08,
        },
        # Noncommuting benchmark propagation
        {
            "setup": setup_0 + checks,
            "call": '_checked(advance_walker, OCC, C, WALKERS[0, 0], WALKERS[0, 1], H, L, FIELDS[0, 0], 0.08)',
            "gold_call": '_checked(_oracle_advance_walker, OCC, C, WALKERS[0, 0], WALKERS[0, 1], H, L, FIELDS[0, 0], 0.08)',
            "tol": 2e-08,
        },
        # Complex frame and rank-one overlaps in both spins
        {
            "setup": setup_10 + checks,
            "call": '_checked(advance_walker, OCC, C, pack(A), pack(B), H, L, np.array([0.61, -0.43]), 0.047, pack(U))',
            "gold_call": '_checked(_oracle_advance_walker, OCC, C, pack(A), pack(B), H, L, np.array([0.61, -0.43]), 0.047, pack(U))',
            "tol": 2e-08,
        },
        # Complex frame, rank-two overlap, and nonnormalized input gauge
        {
            "setup": setup_11 + checks,
            "call": '_checked(advance_walker, OCC, C, pack(A), pack(B), H, L, np.array([-0.38, 0.72]), 0.035, pack(U))',
            "gold_call": '_checked(_oracle_advance_walker, OCC, C, pack(A), pack(B), H, L, np.array([-0.38, 0.72]), 0.035, pack(U))',
            "tol": 2e-08,
        },
        # Twelve-electron selected trial in 28 orbitals
        {
            "setup": setup_13 + checks,
            "call": '_checked(advance_walker, OCC, C, pack(A), pack(B), H, L, 0.6 * np.sin(np.arange(8) + 0.4), 0.025, pack(U))',
            "gold_call": '_checked(_oracle_advance_walker, OCC, C, pack(A), pack(B), H, L, 0.6 * np.sin(np.arange(8) + 0.4), 0.025, pack(U))',
            "tol": 2e-08,
        },
        # Negative imaginary-time step
        {
            "setup": setup_0 + checks,
            "call": '_raises(_checked, advance_walker, OCC, C, WALKERS[0, 0], WALKERS[0, 1], H, L, FIELDS[0, 0], -0.1)',
            "gold_call": '_raises(_checked, _oracle_advance_walker, OCC, C, WALKERS[0, 0], WALKERS[0, 1], H, L, FIELDS[0, 0], -0.1)',
            "tol": 2e-08,
        },
        # Propagated QR norm falls below the documented threshold
        {
            "setup": setup_15 + checks,
            "call": '_raises(_checked, advance_walker, OCC, C, A, B, H, L, FIELD, 1.0)',
            "gold_call": '_raises(_checked, _oracle_advance_walker, OCC, C, A, B, H, L, FIELD, 1.0)',
            "tol": 2e-08,
        },
        # Nonzero old overlap but zero normalized propagated overlap
        {
            "setup": setup_16 + checks,
            "call": '_raises(_checked, advance_walker, OCC, C, A, B, H, L, FIELD, 1.0)',
            "gold_call": '_raises(_checked, _oracle_advance_walker, OCC, C, A, B, H, L, FIELD, 1.0)',
            "tol": 2e-08,
        },
    ]
