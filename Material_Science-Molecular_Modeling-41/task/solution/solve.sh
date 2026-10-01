#!/bin/bash
# Oracle: writes the gold solution so the verifier scores reward 1.0.
cat > /app/solution.py <<'SCICODE_GOLD_EOF'
import numpy as np
try:
    import scipy, scipy.linalg, scipy.sparse, scipy.optimize, scipy.spatial, scipy.integrate, scipy.stats, scipy.special, scipy.signal
except Exception:
    pass
import math

def correlation_transform(cvv: "np.ndarray", mass: float, thermal_energy: float, rho: float, grid_size: int) -> "tuple[np.ndarray, np.ndarray, np.ndarray]":
    """Normalize a vector velocity correlation and evaluate its generating sum.

    Parameters
    ----------
    cvv : real array (n,d,d)
        n >= 4, d >= 1. cvv[k] = <v(k*tau) v(0)^T>, in squared
        velocity units. Positive and negative lags are not symmetrized.
    mass, thermal_energy : positive finite real scalars
        Common probe mass and k_B*T in compatible reduced units.
    rho : finite real scalar > 1
    grid_size : even integer >= 8

    Contract
    --------
    Return the thermally normalized correlation sequence, the complex
    sampling grid and the source's one-sided matrix generating function
    evaluated on that grid using every supplied lag. The normalized
    zero-lag matrix must equal the identity within maximum entry error
    1e-8. Grid points are uniformly spaced counterclockwise on the circle
    of radius rho, beginning at the positive real axis. Preserve the
    orientation of each cross-correlation matrix.

    Returns
    -------
    result
        Tuple (Y,z,F): real (n,d,d), complex (grid_size,), and complex
        (grid_size,d,d) arrays, in that order. Y and F are dimensionless.

    Raises
    ------
    ValueError
        Complex/nonfinite real input, incompatible shape, n<4, d<1,
        nonpositive mass or thermal_energy, rho<=1, invalid grid_size,
        or Y[0] outside the stated tolerance.
    """
    import numpy as np
    if any((np.iscomplexobj(x) for x in [cvv, mass, thermal_energy, rho])):
        raise ValueError('real inputs required')
    c = np.asarray(cvv, dtype=float)
    if c.ndim != 3 or c.shape[0] < 4 or c.shape[1] < 1 or (c.shape[1] != c.shape[2]) or (not np.isfinite(c).all()):
        raise ValueError('invalid correlations')
    for value in [mass, thermal_energy, rho]:
        if np.ndim(value) != 0 or not np.isfinite(value):
            raise ValueError('invalid scalar')
    if mass <= 0 or thermal_energy <= 0 or rho <= 1:
        raise ValueError('scalar domain')
    if isinstance(grid_size, (bool, np.bool_)) or not isinstance(grid_size, (int, np.integer)) or grid_size < 8 or grid_size % 2:
        raise ValueError('even grid required')
    Y = c * (mass / thermal_energy)
    if np.max(np.abs(Y[0] - np.eye(c.shape[1]))) > 1e-08:
        raise ValueError('normalization')
    z = rho * np.exp(2j * np.pi * np.arange(grid_size) / grid_size)
    F = np.einsum('kab,lk->lab', Y, z[:, None] ** (-np.arange(len(Y)) - 1))
    return (Y, z, F)

def shared_rational_fit(z: "np.ndarray", values: "np.ndarray", tolerance: float, max_support: int) -> "tuple[np.ndarray, np.ndarray]":
    """Fit matrix samples using a common scalar barycentric denominator.

    Parameters
    ----------
    z : complex array (g,)
        Counterclockwise uniform circle rho*exp(2*pi*i*l/g), even g>=8,
        rho>1. Circle agreement tolerance is 1e-10 in absolute entries.
    values : complex array (g,d,d), d>=1
        Conjugate symmetry values[(-l)%g]=conj(values[l]), tolerance 1e-9.
    tolerance : finite real scalar > 0
        Maximum Frobenius residual allowed over the entire grid.
    max_support : integer, 2<=max_support<=g-2
        Maximum number of support points, including conjugate partners.

    Contract
    --------
    Use the source's matrix-valued rational fit with a shared denominator.
    Initial support indices are [0,g//2]. Support and weights are
    conjugate paired, and real supports have real weights. Normalize
    the weight vector to Euclidean norm one; a common sign is immaterial.
    The convergence measure is the largest Frobenius residual on the grid.
    Nonsupport conjugate orbits are scored by their mean residual. Among
    orbits within 1e-12*max(1,largest_score), choose the smallest index
    in 0,...,g//2, inserted before its distinct conjugate partner.
    Any minimizing weights are accepted through the fitted function.

    Returns
    -------
    result
        Tuple (indices,weights): integer (p,) and complex (p,) arrays in
        support insertion order. 2<=p<=max_support, ||weights||_2=1.

    Raises
    ------
    ValueError
        Nonfinite inputs; invalid shapes, circle, conjugacy, tolerance or
        support budget; or residual tolerance not reached within budget.
    """
    import numpy as np
    z = np.asarray(z, dtype=complex)
    v = np.asarray(values, dtype=complex)
    if z.ndim != 1 or len(z) < 8 or len(z) % 2 or (v.ndim != 3) or (v.shape[0] != len(z)) or (v.shape[1] < 1) or (v.shape[1] != v.shape[2]) or (not np.isfinite(z).all()) or (not np.isfinite(v).all()):
        raise ValueError('samples')
    g = len(z)
    rho = z[0].real
    if rho <= 1 or np.max(np.abs(z - rho * np.exp(2j * np.pi * np.arange(g) / g))) > 1e-10:
        raise ValueError('circle')
    if np.max(np.abs(v[-np.arange(g) % g] - v.conj())) > 1e-09:
        raise ValueError('conjugacy')
    if np.iscomplexobj(tolerance) or np.ndim(tolerance) != 0 or (not np.isfinite(tolerance)) or (tolerance <= 0):
        raise ValueError('tolerance')
    if isinstance(max_support, (bool, np.bool_)) or not isinstance(max_support, (int, np.integer)) or (not 2 <= max_support <= g - 2):
        raise ValueError('budget')
    s = [0, g // 2]
    while True:
        rest = np.array([l for l in range(g) if l not in s])
        p = len(s)
        U = np.zeros((p, p), complex)
        seen = set()
        col = 0
        for j, k in enumerate(s):
            if j in seen:
                continue
            partner = s.index(-k % g)
            if partner == j:
                U[j, col] = 1
                col += 1
            else:
                U[j, col] = U[partner, col] = 1 / np.sqrt(2)
                U[j, col + 1] = 1j / np.sqrt(2)
                U[partner, col + 1] = -1j / np.sqrt(2)
                seen.add(partner)
                col += 2
            seen.add(j)
        L = ((v[rest, None] - v[np.array(s)][None]) / (z[rest, None, None, None] - z[np.array(s)][None, :, None, None])).transpose(0, 2, 3, 1).reshape(-1, p)
        LU = L @ U
        _, _, vh = np.linalg.svd(np.vstack([LU.real, LU.imag]), full_matrices=False)
        w = U @ vh[-1]
        C = 1 / (z[rest, None] - z[np.array(s)][None])
        with np.errstate(divide='ignore', invalid='ignore'):
            fitted = np.einsum('ij,jab->iab', C * w, v[s]) / (C @ w)[:, None, None]
        scores = np.zeros(g)
        scores[rest] = np.linalg.norm(v[rest] - fitted, axis=(1, 2))
        scores[~np.isfinite(scores)] = np.inf
        if np.max(scores) <= tolerance:
            return (np.array(s, dtype=int), w)
        orbits = [k for k in range(g // 2 + 1) if k not in s]
        score = np.array([(scores[k] + scores[-k % g]) / 2 for k in orbits])
        largest = np.max(score)
        if np.isinf(largest):
            k = orbits[int(np.flatnonzero(np.isinf(score))[0])]
        else:
            k = orbits[int(np.flatnonzero(score >= largest - 1e-12 * max(1.0, largest))[0])]
        new = [k] if k == -k % g else [k, -k % g]
        if len(s) + len(new) > max_support:
            raise ValueError('budget exhausted')
        s.extend(new)

def continuous_poles(support: "np.ndarray", weights: "np.ndarray", tau: float) -> "np.ndarray":
    """Extract stable continuous exponents of a barycentric denominator.

    Parameters
    ----------
    support, weights : finite complex arrays (p,), p>=2
        Distinct support points and nonzero weights; same shape.
    tau : positive finite real scalar
        Correlation sampling interval in reduced time units.

    Contract
    --------
    Return the stable continuous exponents of the supplied denominator.
    Retain discrete poles only when 1e-12<abs(z)<1-1e-10; discard negative
    real poles. Real means |Im(z)|<=1e-8; conjugate pairing tolerance is
    1e-7. Average each pair with the conjugate of its partner.
    Use the principal logarithm and the open band |Im(lambda)|<pi/tau.
    Return real rates first, in increasing real part; then conjugate
    pairs sorted by the positive member's (real part, imaginary part),
    with the positive member first. Rates have inverse-time units.

    Returns
    -------
    result
        Complex array (r,) of stable rates in the specified order, r>=1.

    Raises
    ------
    ValueError
        Invalid shape, nonfinite values, duplicate support within 1e-12,
        zero weight, invalid tau, no retained poles, or unpaired pole.
    """
    import numpy as np
    from scipy.linalg import eig
    s = np.asarray(support, complex)
    w = np.asarray(weights, complex)
    if s.ndim != 1 or len(s) < 2 or w.shape != s.shape or (not np.isfinite(s).all()) or (not np.isfinite(w).all()) or np.any(abs(w) == 0):
        raise ValueError('support and weights')
    if np.any(np.abs(s[:, None] - s[None, :] + np.eye(len(s))) < 1e-12):
        raise ValueError('duplicate support')
    if np.iscomplexobj(tau) or np.ndim(tau) != 0 or (not np.isfinite(tau)) or (tau <= 0):
        raise ValueError('tau')
    p = len(s)
    M = np.zeros((p + 1, p + 1), complex)
    M[0, 1:] = w
    M[1:, 0] = 1
    M[1:, 1:] = np.diag(s)
    N = np.eye(p + 1)
    N[0, 0] = 0
    z = eig(M, N, right=False)
    z = z[np.isfinite(z) & (abs(z) > 1e-12) & (abs(z) < 1 - 1e-10)]
    real = []
    positive = []
    used = set()
    for i, a in enumerate(z):
        if i in used:
            continue
        if abs(a.imag) <= 1e-08:
            if a.real > 0:
                real.append(np.log(a.real) / tau)
            used.add(i)
            continue
        choices = [j for j in range(len(z)) if j != i and j not in used]
        if not choices:
            raise ValueError('unpaired pole')
        j = min(choices, key=lambda k: abs(z[k] - a.conjugate()))
        if abs(z[j] - a.conjugate()) > 1e-07:
            raise ValueError('unpaired pole')
        a = (a + z[j].conjugate()) / 2
        if a.imag < 0:
            a = a.conjugate()
        positive.append(np.log(a) / tau)
        used.update([i, j])
    out = list(sorted(real))
    for a in sorted(positive, key=lambda x: (x.real, x.imag)):
        out.extend([a, a.conjugate()])
    if not out:
        raise ValueError('no stable poles')
    return np.array(out, complex)

def constrained_residues(Y: "np.ndarray", tau: float, rates: "np.ndarray") -> "np.ndarray":
    """Fit matrix exponential residues with inertial short-time constraints.

    Parameters
    ----------
    Y : finite real array (n,d,d), n>=4, d>=1
    tau : positive finite real scalar
    rates : finite complex array (r,), r>=1
        Distinct stable rates, conjugate closed, with |Im(rate)|<pi/tau.
        Real means |Im(rate)|<=1e-8. Pair and distinctness tolerance 1e-7.

    Contract
    --------
    Return the least-squares matrix amplitudes of the source's inertial
    exponential reconstruction, using every supplied lag and matrix
    entry with equal weight. The fitted curve has identity zero-lag
    covariance, zero initial slope and symmetric curvature at zero.
    Real rates have real residues and paired rates conjugate residues.
    The relative null-space rank threshold is 1e-12. Individual residue
    matrices need not be symmetric or positive.

    Returns
    -------
    result
        Complex array (r,d,d), ordered exactly as rates. Dimensionless.
        On exact-data fixtures, tests compare the sampled curve with Y
        (max abs error 2e-7) and check all three constraints
        (max abs residual 1e-7). Noisy-data fits compare the minimizer
        with rtol=2e-8 and atol=2e-8.

    Raises
    ------
    ValueError
        Complex Y, invalid shapes or nonfinite data, invalid tau, unstable,
        repeated, unpaired or out-of-band rates; inconsistent constraints
        (maximum constraint residual >1e-8); or a rank-deficient constrained fit.
    """
    import numpy as np
    from scipy.linalg import null_space
    if np.iscomplexobj(Y):
        raise ValueError('real Y')
    y = np.asarray(Y, float)
    lam = np.asarray(rates, complex)
    if y.ndim != 3 or len(y) < 4 or y.shape[1] < 1 or (y.shape[1] != y.shape[2]) or (not np.isfinite(y).all()) or (lam.ndim != 1) or (not len(lam)) or (not np.isfinite(lam).all()):
        raise ValueError('data')
    if np.iscomplexobj(tau) or np.ndim(tau) != 0 or (not np.isfinite(tau)) or (tau <= 0):
        raise ValueError('tau')
    if np.any(lam.real >= 0) or np.any(abs(lam.imag) >= np.pi / tau):
        raise ValueError('rates domain')
    if len(lam) > 1 and np.min(abs(lam[:, None] - lam[None, :] + np.eye(len(lam)) * 10000000000.0)) < 1e-07:
        raise ValueError('repeated rates')
    basis = []
    groups = []
    used = set()
    for i, a in enumerate(lam):
        if i in used:
            continue
        if abs(a.imag) <= 1e-08:
            basis.append((a.real, 0))
            groups.append((i, None))
            used.add(i)
        elif a.imag > 0:
            j = int(np.argmin(abs(lam - a.conjugate())))
            if j == i or abs(lam[j] - a.conjugate()) > 1e-07:
                raise ValueError('unpaired rates')
            basis.extend([(a, 1), (a, 2)])
            groups.append((i, j))
            used.update([i, j])
    if len(used) != len(lam):
        raise ValueError('unpaired rates')
    t = np.arange(len(y)) * tau
    columns = []
    ders = []
    for a, kind in basis:
        e = np.exp(t * a)
        columns.append(e.real if kind == 0 else 2 * e.real if kind == 1 else -2 * e.imag)
        ders.append([float(np.real(a ** k)) if kind == 0 else 2 * (a ** k).real if kind == 1 else -2 * (a ** k).imag for k in range(3)])
    X = np.array(columns).T
    der = np.array(ders).T
    p = len(basis)
    d = y.shape[1]
    design = np.kron(X, np.eye(d * d))
    C = []
    target = []
    for order in [0, 1]:
        for a in range(d):
            for b in range(d):
                row = np.zeros((p, d, d))
                row[:, a, b] = der[order]
                C.append(row.ravel())
                target.append(float(order == 0 and a == b))
    for a in range(d):
        for b in range(a + 1, d):
            row = np.zeros((p, d, d))
            row[:, a, b] = der[2]
            row[:, b, a] = -der[2]
            C.append(row.ravel())
            target.append(0.0)
    C = np.array(C)
    target = np.array(target)
    x0 = np.linalg.lstsq(C, target, rcond=1e-12)[0]
    if np.max(abs(C @ x0 - target)) > 1e-08:
        raise ValueError('inconsistent constraints')
    Z = null_space(C, rcond=1e-12)
    if Z.shape[1]:
        u, _, rank, _ = np.linalg.lstsq(design @ Z, y.ravel() - design @ x0, rcond=1e-12)
        if rank < Z.shape[1]:
            raise ValueError('unidentifiable fit')
        x0 = x0 + Z @ u
    coeff = x0.reshape(p, d, d)
    gamma = np.zeros((len(lam), d, d), complex)
    k = 0
    for i, j in groups:
        if j is None:
            gamma[i] = coeff[k]
            k += 1
        else:
            gamma[i] = coeff[k] + 1j * coeff[k + 1]
            gamma[j] = gamma[i].conj()
            k += 2
    return gamma

def real_velocity_embedding(rates: "np.ndarray", residues: "np.ndarray", rank_tolerance: float) -> "np.ndarray":
    """Construct a real minimal residue realization with velocity first.

    Parameters
    ----------
    rates : finite complex array (r,), stable and conjugate closed
    residues : finite complex array (r,d,d)
        Conjugate closed with rates, maximum entry tolerance 1e-7.
        Real rates have real residues. sum residues=I within 1e-7.
    rank_tolerance : finite real scalar in (0,1)

    Contract
    --------
    Return a real residue realization with the normalized velocities
    first and the retained auxiliary coordinates following them.
    Retain each residue singular value s strictly above
    rank_tolerance*max(1,s_max). Conjugate pairs use real rotation blocks
    with the positive-imaginary representative first; equivalent real
    hidden-coordinate bases are accepted. The retained zero-lag
    correlation must agree with the identity within 1e-6.
    Tests compare the correlation and basis identities, not individual
    entries in an arbitrary hidden-coordinate basis.

    Returns
    -------
    result
        Real drift matrix A, shape (q,q), q>d, inverse-time units.
        Its correlation is exp(A*t)[:d,:d].

    Raises
    ------
    ValueError
        Invalid shapes, nonfinite inputs, unstable/unpaired/repeated
        rates (pair tolerance 1e-7), inconsistent residue conjugacy or
        normalization, invalid rank_tolerance, retained q<=d, or singular X.
    """
    import numpy as np
    from scipy.linalg import block_diag, null_space
    lam = np.asarray(rates, complex)
    gamma = np.asarray(residues, complex)
    if lam.ndim != 1 or not len(lam) or gamma.ndim != 3 or (gamma.shape[0] != len(lam)) or (gamma.shape[1] < 1) or (gamma.shape[1] != gamma.shape[2]) or (not np.isfinite(lam).all()) or (not np.isfinite(gamma).all()) or np.any(lam.real >= 0):
        raise ValueError('residues')
    if np.iscomplexobj(rank_tolerance) or np.ndim(rank_tolerance) != 0 or (not np.isfinite(rank_tolerance)) or (not 0 < rank_tolerance < 1):
        raise ValueError('rank tolerance')
    if len(lam) > 1 and np.min(abs(lam[:, None] - lam[None, :] + np.eye(len(lam)) * 10000000000.0)) < 1e-07:
        raise ValueError('repeated rates')
    d = gamma.shape[1]
    if np.max(abs(gamma.sum(axis=0) - np.eye(d))) > 1e-07:
        raise ValueError('normalization')
    blocks = []
    outputs = []
    inputs = []
    seen = set()
    for i, a in enumerate(lam):
        if i in seen:
            continue
        if abs(a.imag) <= 1e-08:
            if np.max(abs(gamma[i].imag)) > 1e-07:
                raise ValueError('real residue')
            g = gamma[i].real
            pair = False
            seen.add(i)
        else:
            if a.imag < 0:
                continue
            j = int(np.argmin(abs(lam - a.conjugate())))
            if i == j or abs(lam[j] - a.conjugate()) > 1e-07 or np.max(abs(gamma[j] - gamma[i].conj())) > 1e-07:
                raise ValueError('conjugacy')
            g = gamma[i]
            pair = True
            seen.update([i, j])
        u, s, vh = np.linalg.svd(g)
        keep = s > rank_tolerance * max(1.0, s[0])
        k = int(keep.sum())
        if not k:
            continue
        C = u[:, keep] * np.sqrt(s[keep])
        B = np.sqrt(s[keep])[:, None] * vh[keep]
        if not pair:
            blocks.append(a.real * np.eye(k))
            outputs.append(C.real)
            inputs.append(B.real)
        else:
            blocks.append(np.block([[a.real * np.eye(k), -a.imag * np.eye(k)], [a.imag * np.eye(k), a.real * np.eye(k)]]))
            outputs.append(np.concatenate([2 * C.real, -2 * C.imag], axis=1))
            inputs.append(np.concatenate([B.real, B.imag], axis=0))
    if len(seen) != len(lam) or not blocks:
        raise ValueError('unpaired or empty realization')
    J = block_diag(*blocks)
    Cbar = np.concatenate(outputs, axis=1)
    Bbar = np.concatenate(inputs, axis=0)
    if len(J) <= d or np.max(abs(Cbar @ Bbar - np.eye(d))) > 1e-06:
        raise ValueError('retained normalization')
    X = np.vstack([Cbar, null_space(Bbar.T, rcond=1e-12).T])
    try:
        A = np.linalg.solve(X.T, (X @ J).T).T
    except np.linalg.LinAlgError as e:
        raise ValueError('singular realization') from e
    return A

def singular_memory_reduction(A: "np.ndarray", dimension: int) -> "tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]":
    """Reduce a purely inertial embedding to a regular auxiliary noise problem.

    Parameters
    ----------
    A : finite real drift array (q,q), q>=2*d
    dimension : integer d>=1
        Number of leading normalized velocity coordinates.

    Contract
    --------
    Return the source's singular-to-regular auxiliary reduction of an
    inertial realization. The leading velocity block must be zero within
    1e-7; all drift eigenvalues have real part below -1e-10. The induced
    zero-time memory must be symmetric within 1e-7 and positive definite
    with minimum eigenvalue above 1e-10. Use its symmetric positive
    square root as the coordinate convention and an orthonormal basis
    for the remaining null space. The symmetric part D1+D1.T of the
    reduced direct-damping block must have minimum eigenvalue >1e-10.
    Equivalent null-space bases are accepted through reduction identities.

    Returns
    -------
    result
        Tuple (A1,B1,C1,D1,X). With h=q-2*d and a=q-d, the shapes are
        (h,h),(h,d),(h,d),(d,d),(a,a). All arrays are real; h may be zero.
        Hidden bases may differ, provided the defining identities hold.

    Raises
    ------
    ValueError
        Complex/nonfinite A, invalid square shape or dimension, q<2*d,
        nonzero velocity block, unstable A, nonsymmetric or nonpositive S,
        singular X, or nonpositive R1 under the stated tolerances.
    """
    import numpy as np
    from scipy.linalg import null_space
    if np.iscomplexobj(A):
        raise ValueError('real drift')
    A = np.asarray(A, float)
    if isinstance(dimension, (bool, np.bool_)) or not isinstance(dimension, (int, np.integer)) or dimension < 1:
        raise ValueError('dimension')
    d = dimension
    if A.ndim != 2 or A.shape[0] != A.shape[1] or len(A) < 2 * d or (not np.isfinite(A).all()):
        raise ValueError('drift shape')
    if np.max(abs(A[:d, :d])) > 1e-07 or np.max(np.linalg.eigvals(A).real) >= -1e-10:
        raise ValueError('inertial stable drift')
    B = A[:d, d:].T
    C = -A[d:, :d]
    A0 = A[d:, d:]
    S = B.T @ C
    if np.max(abs(S - S.T)) > 1e-07:
        raise ValueError('curvature symmetry')
    s, U = np.linalg.eigh((S + S.T) / 2)
    if np.min(s) <= 1e-10:
        raise ValueError('curvature positivity')
    invroot = U / np.sqrt(s) @ U.T
    X = np.vstack([invroot @ B.T, null_space(C.T, rcond=1e-12).T])
    try:
        V = np.linalg.solve(X.T, (X @ A0).T).T
    except np.linalg.LinAlgError as e:
        raise ValueError('singular deflation') from e
    D = -V[:d, :d]
    if np.min(np.linalg.eigvalsh(D + D.T)) <= 1e-10:
        raise ValueError('regular damping')
    return (V[d:, d:], V[:d, d:].T, -V[d:, :d], D, X)

def regular_lure(A0: "np.ndarray", B: "np.ndarray", C: "np.ndarray", D: "np.ndarray") -> "tuple[np.ndarray, np.ndarray, np.ndarray]":
    """Solve the stabilizing regular Lur'e covariance completion.

    Parameters
    ----------
    A0 : finite real array (h,h), h>=0
    B, C : finite real arrays (h,d), d>=1
    D : finite real array (d,d)

    Contract
    --------
    Return the positive stationary covariance and shared-noise factors
    of the source's regular auxiliary problem, selecting its stabilizing
    covariance branch. The symmetric direct-damping part must have
    minimum eigenvalue >1e-10. K uses the lower-Cholesky convention.
    Stability means eigenvalue real parts below -1e-10. Reject Hamiltonian
    eigenvalues within 1e-10 of the imaginary axis, covariance minimum
    eigenvalue <=1e-10, or Riccati Frobenius residual greater than
    1e-7*max(1,||S||_F). For h=0 return empty S and L, together with K.

    Returns
    -------
    result
        Tuple (S,L,K): real arrays (h,h),(h,d),(d,d). S is dimensionless;
        L,K have inverse-square-root-time units in the rescaled state.

    Raises
    ------
    ValueError
        Complex/nonfinite input, incompatible shapes, nonpositive R,
        missing stable h-dimensional graph, singular U, imaginary-axis
        Hamiltonian spectrum, nonpositive covariance, unstable completed
        Riccati branch, or excessive residual under the stated thresholds.
    """
    import numpy as np
    from scipy.linalg import schur
    if any((np.iscomplexobj(x) for x in [A0, B, C, D])):
        raise ValueError('real matrices')
    A0, B, C, D = [np.asarray(x, float) for x in [A0, B, C, D]]
    if A0.ndim != 2 or A0.shape[0] != A0.shape[1] or D.ndim != 2 or (D.shape[0] < 1) or (D.shape[0] != D.shape[1]) or (B.shape != (len(A0), len(D))) or (C.shape != B.shape) or (not all((np.isfinite(x).all() for x in [A0, B, C, D]))):
        raise ValueError('shapes')
    R = D + D.T
    if np.min(np.linalg.eigvalsh(R)) <= 1e-10:
        raise ValueError('R positivity')
    K = np.linalg.cholesky(R)
    h = len(A0)
    if h == 0:
        return (np.empty((0, 0)), np.empty((0, len(D))), K)
    P = A0 - C @ np.linalg.solve(R, B.T)
    G = B @ np.linalg.solve(R, B.T)
    Q = C @ np.linalg.solve(R, C.T)
    H = np.block([[P.T, G], [-Q, -P]])
    if np.min(abs(np.linalg.eigvals(H).real)) <= 1e-10:
        raise ValueError('imaginary axis')
    _, U, count = schur(H, output='real', sort=lambda a, b: a < 0)
    if count != h:
        raise ValueError('stable graph dimension')
    try:
        S = np.linalg.solve(U[:h, :h].T, U[h:, :h].T).T
    except np.linalg.LinAlgError as e:
        raise ValueError('singular graph') from e
    S = (S + S.T) / 2
    if np.min(np.linalg.eigvalsh(S)) <= 1e-10 or np.max(np.linalg.eigvals(P + S @ G).real) >= -1e-10:
        raise ValueError('Riccati branch')
    if np.linalg.norm(P @ S + S @ P.T + S @ G @ S + Q) > 1e-07 * max(1.0, np.linalg.norm(S)):
        raise ValueError('Riccati residual')
    L = np.linalg.solve(K, (C - S @ B).T).T
    return (S, L, K)

def complete_discrete_embedding(A: "np.ndarray", dimension: int, X: "np.ndarray", S1: "np.ndarray", L1: "np.ndarray", K1: "np.ndarray", intervals: "np.ndarray") -> "tuple[np.ndarray, np.ndarray, np.ndarray]":
    """Complete the stationary covariance and integrate the embedded noise.

    Parameters
    ----------
    A : finite real stable drift (q,q), q>=2*d
    dimension : integer d>=1
    X : finite real invertible auxiliary transformation (q-d,q-d)
    S1, L1, K1 : finite real arrays (q-2*d,q-2*d),(q-2*d,d),(d,d)
        The covariance and noise factors from the reduced Lur'e problem.
    intervals : finite real array (j,), j>=0, each entry in (0,3]
        Time increments between successive held-out measurements.

    Contract
    --------
    Return the stationary covariance in the original state coordinates
    and exact finite-interval transition and process-noise covariances.
    Keep all coupled noise contributions. Sigma is symmetric within
    1e-7 and has minimum eigenvalue >1e-10; drift eigenvalue real parts
    are below -1e-10. The continuous stationarity residual has Frobenius
    norm at most 1e-6*max(1,||Sigma||_F). Return symmetric process-noise
    covariances with minimum eigenvalue no smaller than -1e-8.
    Exact covariance identities or exact integration are acceptable;
    no diagonal regularizer is part of the model.

    Returns
    -------
    result
        Tuple (Sigma,T,Q): real (q,q), (j,q,q), (j,q,q) arrays. Sigma
        and Q are covariances of the dimensionless state. For j=0, the
        last two arrays retain shape (0,q,q).

    Raises
    ------
    ValueError
        Complex/nonfinite input, incompatible shapes, invalid dimension
        or intervals, singular X, nonpositive/nonsymmetric covariance,
        unstable A, failed stationary Lyapunov identity, or a transition
        covariance with minimum eigenvalue below -1e-8.
    """
    import numpy as np
    from scipy.linalg import block_diag, expm
    if any((np.iscomplexobj(x) for x in [A, X, S1, L1, K1, intervals])):
        raise ValueError('real input')
    A, X, S1, L1, K1, dt = [np.asarray(x, float) for x in [A, X, S1, L1, K1, intervals]]
    if isinstance(dimension, (bool, np.bool_)) or not isinstance(dimension, (int, np.integer)) or dimension < 1:
        raise ValueError('dimension')
    d = dimension
    if A.ndim != 2 or A.shape[0] != A.shape[1] or len(A) < 2 * d:
        raise ValueError('drift')
    q = len(A)
    a = q - d
    h = q - 2 * d
    if X.shape != (a, a) or S1.shape != (h, h) or L1.shape != (h, d) or (K1.shape != (d, d)) or (dt.ndim != 1) or (not all((np.isfinite(x).all() for x in [A, X, S1, L1, K1, dt]))) or np.any(dt <= 0) or np.any(dt > 3):
        raise ValueError('input shapes or intervals')
    try:
        Xi = np.linalg.solve(X, np.eye(a))
    except np.linalg.LinAlgError as e:
        raise ValueError('singular X') from e
    Sigma = block_diag(np.eye(d), Xi @ block_diag(np.eye(d), S1) @ Xi.T)
    noise = np.vstack([np.zeros((d, d)), Xi @ np.vstack([K1, L1])])
    if np.max(abs(Sigma - Sigma.T)) > 1e-07 or np.min(np.linalg.eigvalsh((Sigma + Sigma.T) / 2)) <= 1e-10 or np.max(np.linalg.eigvals(A).real) >= -1e-10:
        raise ValueError('stationary covariance')
    Sigma = (Sigma + Sigma.T) / 2
    N = noise @ noise.T
    if np.linalg.norm(A @ Sigma + Sigma @ A.T + N) > 1e-06 * max(1.0, np.linalg.norm(Sigma)):
        raise ValueError('stationary noise')
    aug = np.block([[A, N], [np.zeros_like(A), -A.T]])
    transitions = np.empty((len(dt), q, q))
    covariances = np.empty_like(transitions)
    for k, interval in enumerate(dt):
        V = expm(interval * aug)
        T = V[:q, :q]
        Q = V[:q, q:] @ T.T
        Q = (Q + Q.T) / 2
        if np.min(np.linalg.eigvalsh(Q)) < -1e-08:
            raise ValueError('noise covariance')
        transitions[k] = T
        covariances[k] = Q
    return (Sigma, transitions, covariances)

def velocity_innovation_score(Sigma: "np.ndarray", transitions: "np.ndarray", noise_covariances: "np.ndarray", observations: "np.ndarray", observation_covariance: "np.ndarray") -> float:
    """Evaluate the stationary Gaussian innovation likelihood of probe velocities.

    Parameters
    ----------
    Sigma : finite real stationary covariance (q,q), positive definite
    transitions, noise_covariances : finite real arrays (m-1,q,q)
    observations : finite real array (m,d), m>=1, 1<=d<=q
        Dimensionless velocities v*sqrt(mass/(k_B*T)).
    observation_covariance : finite real symmetric positive definite (d,d)
        Independent measurement-error covariance in those same units.

    Contract
    --------
    Evaluate the complete joint stationary Gaussian likelihood of all
    measurements of the first d state coordinates, with zero mean and
    initial state covariance Sigma. Use each supplied transition and
    process covariance for its corresponding interval and include the
    independent measurement covariance, the initial observation and
    Gaussian normalization constants. Divide by m*d. Equivalent dense
    joint-covariance and sequential-conditioning calculations are valid.
    Symmetry tolerance is 1e-7; positive definite means minimum >1e-10;
    process covariances may be semidefinite down to -1e-8.

    Returns
    -------
    result
        Finite Python float, negative log likelihood in nats per scalar
        normalized velocity. Negative values are allowed for a density.

    Raises
    ------
    ValueError
        Complex/nonfinite input, incompatible shapes, nonsymmetric or
        nonpositive Sigma/R, nonsymmetric or non-semidefinite Q under
        the stated tolerances, or a nonpositive innovation covariance.
    """
    import numpy as np
    if any((np.iscomplexobj(x) for x in [Sigma, transitions, noise_covariances, observations, observation_covariance])):
        raise ValueError('real input')
    S0, T, Q, y, R = [np.asarray(x, float) for x in [Sigma, transitions, noise_covariances, observations, observation_covariance]]
    if y.ndim != 2 or len(y) < 1 or y.shape[1] < 1 or (S0.ndim != 2) or (S0.shape[0] != S0.shape[1]):
        raise ValueError('data shapes')
    m, d = y.shape
    q = len(S0)
    if d > q or R.shape != (d, d) or T.shape != (m - 1, q, q) or (Q.shape != T.shape) or (not all((np.isfinite(x).all() for x in [S0, T, Q, y, R]))):
        raise ValueError('data shapes')
    for x in [S0, R]:
        if np.max(abs(x - x.T)) > 1e-07 or np.min(np.linalg.eigvalsh((x + x.T) / 2)) <= 1e-10:
            raise ValueError('positive covariance')
    for x in Q:
        if np.max(abs(x - x.T)) > 1e-07 or np.min(np.linalg.eigvalsh((x + x.T) / 2)) < -1e-08:
            raise ValueError('transition noise')
    mean = np.zeros(q)
    P = S0.copy()
    total = 0.0
    for k, row in enumerate(y):
        residual = row - mean[:d]
        S = P[:d, :d] + R
        S = (S + S.T) / 2
        try:
            L = np.linalg.cholesky(S)
        except np.linalg.LinAlgError as e:
            raise ValueError('innovation covariance') from e
        u = np.linalg.solve(L, residual)
        total += 0.5 * (d * np.log(2 * np.pi) + 2 * np.log(np.diag(L)).sum() + u @ u)
        gain = np.linalg.solve(L.T, np.linalg.solve(L, P[:, :d].T)).T
        mean = mean + gain @ residual
        P = P - gain @ S @ gain.T
        P = (P + P.T) / 2
        if k < m - 1:
            mean = T[k] @ mean
            P = T[k] @ P @ T[k].T + Q[k]
            P = (P + P.T) / 2
    return float(total / (m * d))

def molecular_memory_score(cvv: "np.ndarray", tau: float, mass: float, thermal_energy: float, observations: "np.ndarray", intervals: "np.ndarray", observation_covariance: "np.ndarray", rho: float = 1.4, grid_size: int = 64, fit_tolerance: float = 0.001, max_support: int = 18, rank_tolerance: float = 1e-07) -> float:
    """Reconstruct an inertial vector memory model and score a held-out trajectory.

    Parameters
    ----------
    cvv : real (n,d,d), n>=4, normalized zero-lag mass*cvv[0]/k_B*T=I
    tau : positive correlation sample spacing
    mass, thermal_energy : positive mass and k_B*T
    observations : real (m,d), m>=1, physical velocities
    intervals : real (m-1,), increments in (0,3] between observations
    observation_covariance : real positive definite (d,d)
        Measurement covariance of normalized velocities, already scaled.
    rho=1.4, grid_size=64, fit_tolerance=0.001, max_support=18,
    rank_tolerance=1e-7 : numerical parameters
        Their domains and conventions are defined in the earlier steps.

    Definition
    ----------
    Execute correlation_transform, shared_rational_fit, continuous_poles,
    constrained_residues, real_velocity_embedding, singular_memory_reduction,
    regular_lure, complete_discrete_embedding, and velocity_innovation_score,
    in that order. Multiply observations by sqrt(mass/thermal_energy).
    The likelihood is a density in normalized velocity coordinates; do
    not add a change-of-units Jacobian. Preserve every cross correlation.
    Inputs must admit the stable positive-real inertial realization with
    q>=2*d specified by those steps. All their rejection conditions apply.
    Final score comparisons use absolute tolerance 2e-6. A numerically
    equivalent realization is accepted. The fitted parameters, covariance,
    and noise must be derived from these inputs, without stored answers.
    Import required packages inside the function.

    Returns
    -------
    result
        Finite Python float: stationary negative log likelihood per scalar
        normalized velocity, in nats.

    Raises
    ------
    ValueError
        Any invalid input or failed admissibility condition declared by
        an earlier step; in addition, observation dimension must equal the
        cvv dimension and intervals must have length len(observations)-1.
    """
    import numpy as np
    Y, z, F = correlation_transform(cvv, mass, thermal_energy, rho, grid_size)
    if np.iscomplexobj(observations):
        raise ValueError('real observations')
    obs = np.asarray(observations, float)
    if obs.ndim != 2 or len(obs) < 1 or obs.shape[1] != Y.shape[1] or (np.shape(intervals) != (len(obs) - 1,)):
        raise ValueError('observations')
    indices, w = shared_rational_fit(z, F, fit_tolerance, max_support)
    rates = continuous_poles(z[indices], w, tau)
    residues = constrained_residues(Y, tau, rates)
    A = real_velocity_embedding(rates, residues, rank_tolerance)
    d = Y.shape[1]
    A1, B1, C1, D1, X = singular_memory_reduction(A, d)
    S1, L1, K1 = regular_lure(A1, B1, C1, D1)
    Sigma, T, Q = complete_discrete_embedding(A, d, X, S1, L1, K1, intervals)
    return velocity_innovation_score(Sigma, T, Q, obs * np.sqrt(mass / thermal_energy), observation_covariance)
SCICODE_GOLD_EOF
