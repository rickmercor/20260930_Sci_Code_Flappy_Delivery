#!/bin/bash
# Oracle: writes the gold solution so the verifier scores reward 1.0.
cat > /app/solution.py <<'SCICODE_GOLD_EOF'
import numpy as np
try:
    import scipy, scipy.linalg, scipy.sparse, scipy.optimize, scipy.spatial, scipy.integrate, scipy.stats, scipy.special, scipy.signal
except Exception:
    pass
import math

def band_structure(momentum, parameters):
    """Diagonalize the periodic two-orbital Bloch Hamiltonian and its derivatives.

    momentum is finite real (nk,2), nk>=1, in inverse lattice units.
    parameters is finite real (7,)=(m,bx,by,a,tx,ty,tz).
    H=dx*sigma_x+dy*sigma_y+dz*sigma_z, where
    dx=a+tx*cos(kx)+0.19*cos(ky), dy=ty*sin(kx)+tz*sin(ky),
    dz=m+bx*(1-cos(kx))+by*(1-cos(ky)). Energy unit is fixed by parameters;
    lattice constant and hbar are 1. Use ascending energies (valence,conduction).
    dH/dkx uses (-tx*sin(kx),ty*cos(kx),bx*sin(kx));
    dH/dky uses (-0.19*sin(ky),tz*cos(ky),by*sin(ky)).

    Returns
    -------
    result
        Tuple (energies real (nk,2), eigenvectors complex (nk,2,2) in columns,
        H complex (nk,2,2), dH complex (nk,2,2,2), derivative axis second).
        Eigenvector phases are arbitrary; only physical invariants are specified.

    Raises
    ------
    ValueError
        For non-real/nonfinite inputs, wrong shapes, or a sampled gap <=1e-10.
    """
    import numpy as np
    try:
        if np.iscomplexobj(momentum) or np.iscomplexobj(parameters):
            raise ValueError('real inputs')
        k = np.asarray(momentum, dtype=float)
        p = np.asarray(parameters, dtype=float)
    except (TypeError, ValueError) as e:
        raise ValueError('numeric inputs') from e
    if k.ndim != 2 or k.shape[1] != 2 or len(k) == 0 or (p.shape != (7,)) or (not np.all(np.isfinite(k))) or (not np.all(np.isfinite(p))):
        raise ValueError('shape or finite inputs')
    x, y = k.T
    m, bx, by, a, tx, ty, tz = p

    def matrix(dx, dy, dz):
        h = np.empty((len(k), 2, 2), dtype=complex)
        h[:, 0, 0] = dz
        h[:, 1, 1] = -dz
        h[:, 0, 1] = dx - 1j * dy
        h[:, 1, 0] = dx + 1j * dy
        return h
    h = matrix(a + tx * np.cos(x) + 0.19 * np.cos(y), ty * np.sin(x) + tz * np.sin(y), m + bx * (1 - np.cos(x)) + by * (1 - np.cos(y)))
    dh = np.stack([matrix(-tx * np.sin(x), ty * np.cos(x), bx * np.sin(x)), matrix(-0.19 * np.sin(y), tz * np.cos(y), by * np.sin(y))], axis=1)
    e, u = np.linalg.eigh(h)
    if np.any(e[:, 1] - e[:, 0] <= 1e-10):
        raise ValueError('closed gap')
    return (e, u, h, dh)

def density_vertices(left, right, transfer, reciprocal, positions):
    """Evaluate point-orbital density vertices in the cell-periodic orbital gauge.

    left and right are finite complex (n,orb) coefficient arrays with n,orb>=1.
    transfer is finite real (2,), reciprocal finite real (ng,2), ng>=1,
    positions finite real (orb,2). All coefficients use the same orbital basis.
    I_G=sum_l conj(left_l)*right_l*exp[-i*(transfer+G) dot positions_l].
    Coefficients need not be normalized. Lattice constant=1.

    Returns
    -------
    result
        Complex (n,ng) array of density vertices.

    Raises
    ------
    ValueError
        For wrong shapes, nonfinite inputs, or non-real geometric arrays.
    """
    import numpy as np
    try:
        l = np.asarray(left, dtype=complex)
        r = np.asarray(right, dtype=complex)
        if any((np.iscomplexobj(v) for v in (transfer, reciprocal, positions))):
            raise ValueError('real geometry')
        q, g, t = [np.asarray(v, dtype=float) for v in (transfer, reciprocal, positions)]
    except (TypeError, ValueError) as e:
        raise ValueError('numeric inputs') from e
    if l.ndim != 2 or 0 in l.shape or r.shape != l.shape or (q.shape != (2,)) or (g.ndim != 2) or (g.shape[1] != 2) or (len(g) == 0) or (t.shape != (l.shape[1], 2)) or any((not np.all(np.isfinite(v)) for v in (l, r, q, g, t))):
        raise ValueError('shapes or finite inputs')
    phase = np.exp(-1j * (q[None, :] + g) @ t.T)
    return l.conj() * r @ phase.T

def static_polarizability(momentum, transfer, reciprocal, positions, parameters, spin=2.0):
    """Compute the static independent-particle polarizability, including both band directions.

    momentum/parameters follow band_structure; transfer, reciprocal and
    positions follow density_vertices, with positions shape (2,2).
    spin is a finite positive real scalar. At zero temperature f_v=1,f_c=0.
    Evaluate bands at k and k+q without folding q or shifting reciprocal indices.
    chi_GG'=spin/nk * sum_{k,n!=m} [(f_n-f_m)/(E_nk-E_m,k+q)]
    * I_G(nk,mk+q)*conj(I_G'(nk,mk+q)). Include (n,m)=(0,1),(1,0).
    Cell area=1. Do not replace this matrix by its head or use a factor-two
    shortcut for the two transition directions.

    Returns
    -------
    result
        Complex Hermitian (ng,ng) polarizability in inverse energy units.

    Raises
    ------
    ValueError
        For invalid delegated inputs, nonpositive/non-real/nonfinite spin,
        or an interband energy denominator with absolute value <=1e-10.
    """
    import numpy as np
    try:
        if np.iscomplexobj(spin):
            raise ValueError('real spin')
        s = np.asarray(spin, dtype=float)
        q = np.asarray(transfer, dtype=float)
    except (TypeError, ValueError) as e:
        raise ValueError('numeric inputs') from e
    if s.ndim != 0 or not np.isfinite(s) or s <= 0 or (q.shape != (2,)) or np.iscomplexobj(transfer) or (not np.all(np.isfinite(q))):
        raise ValueError('spin or transfer')
    e, u, _, _ = band_structure(momentum, parameters)
    ep, up, _, _ = band_structure(np.asarray(momentum, dtype=float) + q, parameters)
    chi = None
    for n, m, sign in [(0, 1, 1.0), (1, 0, -1.0)]:
        vertex = density_vertices(u[:, :, n], up[:, :, m], q, reciprocal, positions)
        den = e[:, n] - ep[:, m]
        if np.any(abs(den) <= 1e-10):
            raise ValueError('singular transition')
        term = np.einsum('k,kg,kh->gh', sign / den, vertex, vertex.conj())
        chi = term if chi is None else chi + term
    return float(s) * chi / len(e)

def screened_interaction(transfer, reciprocal, polarizability, coupling):
    """Invert the symmetrized microscopic dielectric matrix at nonsingular momentum.

    transfer is finite real (2,), reciprocal finite real (ng,2), ng>=1.
    polarizability is finite Hermitian complex (ng,ng), atol=1e-11,rtol=0.
    coupling>0 is finite real. Cell area=1; v_G=2*pi*coupling/|q+G|.
    E=I-sqrt(v)*chi*sqrt(v), W=sqrt(v)*E^-1*sqrt(v).
    Require every |q+G|>1e-12 and E positive definite. The first reciprocal
    vector is the head for callers using a macroscopic dielectric function.
    Return E^-1 as well as W; epsilon_M=1/(E^-1)[0,0], not E[0,0].

    Returns
    -------
    result
        Tuple (W complex (ng,ng), inverse_dielectric complex (ng,ng)).

    Raises
    ------
    ValueError
        For wrong shapes, nonfinite inputs, non-real geometry/coupling,
        nonpositive coupling, non-Hermitian chi, singular q+G, or nonpositive E.
    """
    import numpy as np
    try:
        if any((np.iscomplexobj(v) for v in (transfer, reciprocal, coupling))):
            raise ValueError('real inputs')
        q, g, c = [np.asarray(v, dtype=float) for v in (transfer, reciprocal, coupling)]
        chi = np.asarray(polarizability, dtype=complex)
    except (TypeError, ValueError) as e:
        raise ValueError('numeric inputs') from e
    if q.shape != (2,) or g.ndim != 2 or g.shape[1] != 2 or (len(g) == 0) or (c.ndim != 0) or (chi.shape != (len(g), len(g))) or any((not np.all(np.isfinite(v)) for v in (q, g, c, chi))) or (c <= 0) or (not np.allclose(chi, chi.conj().T, atol=1e-11, rtol=0)):
        raise ValueError('shape or bounds')
    norm = np.linalg.norm(q + g, axis=1)
    if np.any(norm <= 1e-12):
        raise ValueError('Coulomb singularity')
    root = np.sqrt(2 * np.pi * float(c) / norm)
    eps = np.eye(len(g)) - root[:, None] * chi * root[None, :]
    try:
        l = np.linalg.cholesky(eps)
        inv = np.linalg.solve(l.conj().T, np.linalg.solve(l, np.eye(len(g))))
    except np.linalg.LinAlgError as e:
        raise ValueError('dielectric not positive') from e
    return (root[:, None] * inv * root[None, :], inv)

def gamma_interaction(momentum, reciprocal, positions, parameters, coupling, spin, radial_order, angular_order):
    """Regularize the zero-transfer cell by a circular microscopic head average.

    Inputs follow static_polarizability and screened_interaction. reciprocal[0]
    must be zero to atol=1e-14 and all other vectors nonzero (>1e-12).
    radial_order>=2 and angular_order>=4 are integers. For nk k-points,
    replace their reciprocal cell of area (2*pi)^2/nk by a circle of radius
    R=sqrt(4*pi/nk). Set W00_bar=(1/pi/R^2)*int_0^R r dr int_0^{2pi} W00(q)dtheta.
    Use radial_order Gauss-Legendre nodes on [0,R] and equally spaced angles
    theta_j=2*pi*j/angular_order. At each node recompute the full chi and W.
    At q=0 use chi's nonzero-G body to obtain the screened body. Set wings
    W0G=WG0=0 for G!=0. This benchmark uses a direct circular average, not a
    fitted linear small-q dielectric approximation.

    Returns
    -------
    result
        Complex Hermitian (ng,ng) regularized zero-transfer interaction.

    Raises
    ------
    ValueError
        For invalid delegated inputs, invalid integer quadrature orders,
        a nonzero first reciprocal vector, or an additional zero reciprocal vector.
    """
    import numpy as np
    if any((isinstance(x, (bool, np.bool_)) or not isinstance(x, (int, np.integer)) for x in (radial_order, angular_order))) or radial_order < 2 or angular_order < 4:
        raise ValueError('quadrature orders')
    chi = static_polarizability(momentum, [0.0, 0.0], reciprocal, positions, parameters, spin)
    g = np.asarray(reciprocal, dtype=float)
    if np.linalg.norm(g[0]) > 1e-14 or np.any(np.linalg.norm(g[1:], axis=1) <= 1e-12):
        raise ValueError('head index')
    radius = np.sqrt(4 * np.pi / len(momentum))
    x, w = np.polynomial.legendre.leggauss(int(radial_order))
    r = radius * (x + 1) / 2
    wr = radius * w / 2
    head = 0.0
    for ri, wi in zip(r, wr):
        for j in range(angular_order):
            theta = 2 * np.pi * j / angular_order
            q = ri * np.array([np.cos(theta), np.sin(theta)])
            response = static_polarizability(momentum, q, g, positions, parameters, spin)
            screened, _ = screened_interaction(q, g, response, coupling)
            head += 2 * ri * wi * float(screened[0, 0].real) / (radius ** 2 * angular_order)
    out = np.zeros_like(chi)
    out[0, 0] = head
    if len(g) > 1:
        out[1:, 1:] = screened_interaction([0.0, 0.0], g[1:], chi[1:, 1:], coupling)[0]
    return out

def exciton_hamiltonian(momentum, reciprocal, positions, parameters, coupling, spin, radial_order, angular_order):
    """Assemble the direct Tamm-Dancoff exciton Hamiltonian on the supplied k-grid.

    Inputs follow gamma_interaction. Require distinct k-points, no coincident
    nonzero-transfer q+G singularities, and an inversion-symmetric reciprocal
    set containing zero first. k differences are unwrapped; do not fold them.
    Band order is v=0,c=1. For q=k_i-k_j define
    a_G=sum_l conj(C_c,i,l)*C_c,j,l*exp[+i*(q+G) dot tau_l],
    b_G=sum_l conj(C_v,i,l)*C_v,j,l*exp[+i*(q+G) dot tau_l].
    D_ij=(a @ W(q) @ conj(b))/nk; H_ij=(Ec_i-Ev_i)*delta_ij-D_ij.
    At i=j use gamma_interaction; elsewhere recompute full static RPA and W.
    Cache equal transfers using component rounding to 13 decimal places;
    evaluate a cache entry at its first unrounded transfer. Omit exchange.
    Check Hermiticity (atol=2e-10,rtol=0); then symmetrize roundoff only.

    Returns
    -------
    result
        Complex Hermitian (nk,nk) direct exciton Hamiltonian, in energy units.
        Band phases may change entries; compare gauge-invariant quantities.

    Raises
    ------
    ValueError
        For invalid delegated inputs, duplicate k-points at 13-decimal precision,
        an inversion-asymmetric/duplicate reciprocal set, or failed Hermiticity.
    """
    import numpy as np
    e, u, _, _ = band_structure(momentum, parameters)
    k = np.asarray(momentum, dtype=float)
    g = np.asarray(reciprocal, dtype=float)
    zero = gamma_interaction(k, g, positions, parameters, coupling, spin, radial_order, angular_order)
    gs = {tuple(v) for v in np.round(g, 13)}
    if len(gs) != len(g) or any((tuple(-v) not in gs for v in np.round(g, 13))):
        raise ValueError('reciprocal inversion')
    if len({tuple(v) for v in np.round(k, 13)}) != len(k):
        raise ValueError('duplicate momentum')
    h = np.diag(e[:, 1] - e[:, 0]).astype(complex)
    cache = {}
    n = len(k)
    for i in range(n):
        for j in range(n):
            q = k[i] - k[j]
            if i == j:
                w = zero
            else:
                key = tuple(np.round(q, 13))
                if key not in cache:
                    chi = static_polarizability(k, q, g, positions, parameters, spin)
                    cache[key] = screened_interaction(q, g, chi, coupling)[0]
                w = cache[key]
            a = density_vertices(u[i:i + 1, :, 1], u[j:j + 1, :, 1], -q, -g, positions)[0]
            b = density_vertices(u[i:i + 1, :, 0], u[j:j + 1, :, 0], -q, -g, positions)[0]
            h[i, j] -= a @ w @ b.conj() / n
    if not np.allclose(h, h.conj().T, atol=2e-10, rtol=0):
        raise ValueError('BSE Hermiticity')
    return (h + h.conj().T) / 2

def optical_dipole(momentum, positions, parameters, polarization):
    """Compute interband dipoles with the intracell position contribution to velocity.

    momentum and parameters follow band_structure. positions is finite real
    (2,2); polarization is a finite nonzero complex (2,) vector, normalized
    internally by its Euclidean norm. In the point-orbital, cell-periodic gauge,
    v_alpha=dH/dk_alpha-i*[diag(tau_alpha),H], with hbar=1.
    d_k=C_c,k^dagger*(sum_alpha polarization_alpha*v_alpha)*C_v,k/(Ec_k-Ev_k).
    The omitted common phase i does not affect oscillator strengths.

    Returns
    -------
    result
        Complex (nk,) dipole vector in lattice-length units.

    Raises
    ------
    ValueError
        For invalid delegated inputs, non-real/nonfinite/wrong-shaped positions,
        or nonfinite, zero or wrong-shaped polarization.
    """
    import numpy as np
    e, u, h, dh = band_structure(momentum, parameters)
    try:
        if np.iscomplexobj(positions):
            raise ValueError('real positions')
        pos = np.asarray(positions, dtype=float)
        pol = np.asarray(polarization, dtype=complex)
    except (TypeError, ValueError) as ex:
        raise ValueError('numeric inputs') from ex
    if pos.shape != (2, 2) or pol.shape != (2,) or (not np.all(np.isfinite(pos))) or (not np.all(np.isfinite(pol))) or (np.linalg.norm(pol) == 0):
        raise ValueError('positions or polarization')
    pol = pol / np.linalg.norm(pol)
    v = np.zeros_like(h)
    for axis in range(2):
        comm = (pos[:, axis, None] - pos[None, :, axis]) * h
        v += pol[axis] * (dh[:, axis] - 1j * comm)
    return np.einsum('ki,kij,kj->k', u[:, :, 1].conj(), v, u[:, :, 0]) / (e[:, 1] - e[:, 0])

def spectral_fraction(hamiltonian, dipole, window, broadening):
    """Integrate the normalized exciton oscillator measure over an energy window.

    hamiltonian is finite Hermitian complex (n,n), n>=1; dipole is finite
    nonzero complex (n,). window=(lo,hi) is finite real with lo<hi;
    broadening>0 is finite real. Let H A_s=E_s A_s, ||A_s||=1,
    f_s=|A_s^dagger dipole|^2. Use the normalized Lorentzian of half-width
    broadening on the entire real energy line. The window fraction is
    sum_s f_s*[atan((hi-E_s)/b)-atan((lo-E_s)/b)]/(pi*sum_s f_s).
    Keep complex conjugation and sum incoherently over different eigenstates.

    Returns
    -------
    result
        Float oscillator fraction in [0,1] up to roundoff.

    Raises
    ------
    ValueError
        For wrong shapes, nonfinite inputs, non-Hermitian H (atol=2e-10,
        rtol=0), zero dipole, complex/invalid window or nonpositive broadening.
    """
    import numpy as np
    try:
        h = np.asarray(hamiltonian, dtype=complex)
        d = np.asarray(dipole, dtype=complex)
        if np.iscomplexobj(window) or np.iscomplexobj(broadening):
            raise ValueError('real window')
        w = np.asarray(window, dtype=float)
        b = np.asarray(broadening, dtype=float)
    except (TypeError, ValueError) as e:
        raise ValueError('numeric inputs') from e
    if h.ndim != 2 or len(h) == 0 or h.shape[1] != len(h) or (d.shape != (len(h),)) or (w.shape != (2,)) or (b.ndim != 0) or any((not np.all(np.isfinite(v)) for v in (h, d, w, b))) or (w[0] >= w[1]) or (b <= 0) or (np.linalg.norm(d) == 0) or (not np.allclose(h, h.conj().T, atol=2e-10, rtol=0)):
        raise ValueError('bounds or Hermiticity')
    e, a = np.linalg.eigh(h)
    f = abs(a.conj().T @ d) ** 2
    integrals = (np.arctan((w[1] - e) / b) - np.arctan((w[0] - e) / b)) / np.pi
    return float(f @ integrals / f.sum())

def solve(data):
    """Return the exciton oscillator fraction for the supplied semiconductor model.

    data is the make_inputs() dictionary with keys momentum, reciprocal,
    positions, parameters, coupling, spin, radial_order, angular_order,
    polarization, window, broadening. Every value follows the preceding
    contracts. Assemble the direct microscopic-screened Tamm-Dancoff matrix,
    compute the position-corrected dipole, and integrate the oscillator measure.
    The finite k/G grids and circle quadrature are fixed benchmark definitions.

    Returns
    -------
    result
        Float normalized oscillator fraction in the prescribed energy window.

    Raises
    ------
    ValueError
        For a non-dictionary input, absent required key, or any invalid delegated input.
    """
    keys = 'momentum reciprocal positions parameters coupling spin radial_order angular_order polarization window broadening'.split()
    if not isinstance(data, dict) or any((k not in data for k in keys)):
        raise ValueError('missing data')
    h = exciton_hamiltonian(*[data[k] for k in keys[:8]])
    d = optical_dipole(data['momentum'], data['positions'], data['parameters'], data['polarization'])
    return spectral_fraction(h, d, data['window'], data['broadening'])
SCICODE_GOLD_EOF
