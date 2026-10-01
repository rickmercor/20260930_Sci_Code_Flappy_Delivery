#!/bin/bash
# Oracle: writes the gold solution so the verifier scores reward 1.0.
cat > /app/solution.py <<'SCICODE_GOLD_EOF'
import numpy as np
try:
    import scipy, scipy.linalg, scipy.sparse, scipy.optimize, scipy.spatial, scipy.integrate, scipy.stats, scipy.special, scipy.signal
except Exception:
    pass
import math

import numpy as np

def assemble_t3_hamiltonian(n_links: int, n_colors: int, coupling: float,
                                    mass: float) -> "np.ndarray":
    import numpy as np

    if (isinstance(n_links, (bool, np.bool_, np.timedelta64, np.datetime64))
            or not isinstance(n_links, (int, np.integer))):
        raise ValueError('n_links must be an integer')
    if (isinstance(n_colors, (bool, np.bool_, np.timedelta64, np.datetime64))
            or not isinstance(n_colors, (int, np.integer))):
        raise ValueError('n_colors must be an integer')
    L = int(n_links)
    nc_i = int(n_colors)
    if L not in (4, 6):
        raise ValueError('n_links must be 4 or 6')
    if not 3 <= nc_i <= 64:
        raise ValueError('n_colors must be between 3 and 64')

    def _finite_real(value, name):
        if (isinstance(value, (bool, np.bool_, np.timedelta64, np.datetime64))
                or not isinstance(value, (int, float, np.integer, np.floating))):
            raise ValueError(f'{name} must be a finite real scalar')
        try:
            result = float(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f'{name} must be a finite real scalar') from exc
        if not np.isfinite(result):
            raise ValueError(f'{name} must be a finite real scalar')
        return result

    nc = float(nc_i)
    g = _finite_real(coupling, 'coupling')
    m = _finite_real(mass, 'mass')
    # 0 = singlet, 1 = N, 2 = Nbar
    I3 = np.eye(3, dtype=float)
    P1 = np.diag([1.0, 0.0, 0.0])
    PN = np.diag([0.0, 1.0, 0.0])
    PNb = np.diag([0.0, 0.0, 1.0])
    ZN = I3 - 2.0 * PN
    ZNb = I3 - 2.0 * PNb
    X1N = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    X1Nb = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])

    def _embed(op, site):
        parts = [I3] * L
        parts[site % L] = op
        out = parts[0]
        for part in parts[1:]:
            out = np.kron(out, part)
        return out

    def _embed3(left, mid, right, j):
        ops = [I3] * L
        ops[(j - 1) % L] = left
        ops[j % L] = mid
        ops[(j + 1) % L] = right
        out = ops[0]
        for part in ops[1:]:
            out = np.kron(out, part)
        return out

    dim = 3 ** L
    hamiltonian = np.zeros((dim, dim), dtype=float)
    elec_pre = -(g * ((nc * nc - 1.0) / (8.0 * nc))) * g
    root = np.sqrt
    leading = 0.5 * root(nc)
    joining = 0.5 / root(nc)
    adjacent = 0.5 * (root(nc - 1.0) - root(nc))
    between = 0.5 * ((nc - 1.0) / root(nc) - root(nc))
    if not np.all(np.isfinite([elec_pre, leading, joining, adjacent, between])):
        raise ValueError('parameters must produce finite Hamiltonian coefficients')
    # Combine local terms before embedding to avoid cancelling large mass terms.
    local_diagonal = elec_pre * (ZN + ZNb) - m * (ZN - ZNb)
    for j in range(L):
        hamiltonian += _embed(local_diagonal, j)
        hamiltonian += leading * _embed3(P1 + PN, X1N, P1 + PN, j)
        hamiltonian += joining * _embed3(PN, X1Nb, PN, j)
        hamiltonian += adjacent * _embed3(P1, X1N, PN, j)
        hamiltonian += adjacent * _embed3(PN, X1N, P1, j)
        hamiltonian += between * _embed3(PN, X1N, PN, j)
    if not np.all(np.isfinite(hamiltonian)):
        raise ValueError('parameters must produce finite Hamiltonian entries')
    return hamiltonian

import numpy as np

def assemble_t1_hamiltonian(n_links: int, n_colors: int, coupling: float,
                                    mass: float) -> "np.ndarray":
    import numpy as np

    if (isinstance(n_links, (bool, np.bool_, np.timedelta64, np.datetime64))
            or not isinstance(n_links, (int, np.integer))):
        raise ValueError('n_links must be an integer')
    if (isinstance(n_colors, (bool, np.bool_, np.timedelta64, np.datetime64))
            or not isinstance(n_colors, (int, np.integer))):
        raise ValueError('n_colors must be an integer')
    L = int(n_links)
    nc_i = int(n_colors)
    if L not in (4, 6):
        raise ValueError('n_links must be 4 or 6')
    if not 3 <= nc_i <= 64:
        raise ValueError('n_colors must be between 3 and 64')

    def _finite_real(value, name):
        if (isinstance(value, (bool, np.bool_, np.timedelta64, np.datetime64))
                or not isinstance(value, (int, float, np.integer, np.floating))):
            raise ValueError(f'{name} must be a finite real scalar')
        try:
            result = float(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f'{name} must be a finite real scalar') from exc
        if not np.isfinite(result):
            raise ValueError(f'{name} must be a finite real scalar')
        return result

    nc = float(nc_i)
    g = _finite_real(coupling, 'coupling')
    m = _finite_real(mass, 'mass')
    I = np.eye(2, dtype=float)
    Z = np.diag([1.0, -1.0])
    X = np.array([[0.0, 1.0], [1.0, 0.0]])
    P0 = np.diag([1.0, 0.0])

    def _embed(op, site):
        parts = [I] * L
        parts[site % L] = op
        out = parts[0]
        for p in parts[1:]:
            out = np.kron(out, p)
        return out

    def _embed3(left, mid, right, j):
        ops = [I] * L
        ops[(j - 1) % L] = left
        ops[j % L] = mid
        ops[(j + 1) % L] = right
        out = ops[0]
        for p in ops[1:]:
            out = np.kron(out, p)
        return out

    H = np.zeros((2 ** L, 2 ** L), dtype=float)
    elec_pre = -(g * ((nc * nc - 1.0) / (8.0 * nc))) * g
    a = 0.5 * np.sqrt(nc)
    if not np.all(np.isfinite([elec_pre, a])):
        raise ValueError('parameters must produce finite Hamiltonian coefficients')
    for j in range(L):
        H += (elec_pre - m) * _embed(Z, j)
        H += a * _embed3(P0, X, P0, j)
    if not np.all(np.isfinite(H)):
        raise ValueError('parameters must produce finite Hamiltonian entries')
    return H

import numpy as np

def vacuum_connected_indices(hamiltonian: object,
                                     tolerance: float) -> "np.ndarray":
    import numpy as np

    try:
        raw = np.asarray(hamiltonian)
        if raw.dtype.kind not in 'iuf':
            raise ValueError('hamiltonian must be a real numeric array')
        H = raw.astype(float, copy=False)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError('hamiltonian must be a finite real square matrix') from exc
    if H.ndim != 2 or H.shape[0] != H.shape[1]:
        raise ValueError('hamiltonian must be square')
    if H.shape[0] < 1:
        raise ValueError('hamiltonian must have at least one row')
    if not np.all(np.isfinite(H)):
        raise ValueError('hamiltonian entries must be finite')
    if (isinstance(tolerance, (bool, np.bool_, np.timedelta64, np.datetime64))
            or not isinstance(tolerance, (int, float, np.integer, np.floating))):
        raise ValueError('tolerance must be a finite real scalar')
    try:
        tol = float(tolerance)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError('tolerance must be a finite real scalar') from exc
    if not np.isfinite(tol) or tol < 0.0:
        raise ValueError('tolerance must be finite and at least 0')
    adjacency = np.abs(H) > tol
    np.fill_diagonal(adjacency, False)
    adjacency = adjacency | adjacency.T
    seen = {0}
    stack = [0]
    while stack:
        i = stack.pop()
        for j in np.nonzero(adjacency[i])[0]:
            j = int(j)
            if j not in seen:
                seen.add(j)
                stack.append(j)
    return np.array(sorted(seen), dtype=int)

import numpy as np

def lowest_spectral_gap(hamiltonian: object) -> float:
    import numpy as np

    try:
        raw = np.asarray(hamiltonian)
        if raw.dtype.kind not in 'iuf':
            raise ValueError('hamiltonian must be a real numeric array')
        H = raw.astype(float, copy=False)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError('hamiltonian must be a finite real square matrix') from exc
    if H.ndim != 2 or H.shape[0] != H.shape[1] or H.shape[0] < 2:
        raise ValueError('hamiltonian must be a square matrix with at least two rows')
    if not np.all(np.isfinite(H)):
        raise ValueError('hamiltonian entries must be finite')
    with np.errstate(over='ignore', invalid='ignore'):
        same_sign = np.signbit(H) == np.signbit(H.T)
        symmetric = np.where(same_sign, H + 0.5 * (H.T - H),
                             0.5 * (H + H.T))
    if not np.all(np.isfinite(symmetric)):
        raise ValueError('symmetrized hamiltonian entries must be finite')
    try:
        ev = np.sort(np.linalg.eigvalsh(symmetric))
    except np.linalg.LinAlgError as exc:
        raise ValueError('hamiltonian could not be diagonalized') from exc
    with np.errstate(over='ignore', invalid='ignore'):
        gap = ev[1] - ev[0]
    if not np.isfinite(gap):
        raise ValueError('hamiltonian must produce a finite spectral gap')
    return float(gap)

import numpy as np

def subspace_gap(hamiltonian: object, indices: object) -> float:
    import numpy as np

    try:
        raw_h = np.asarray(hamiltonian)
        if raw_h.dtype.kind not in 'iuf':
            raise ValueError('hamiltonian must be a real numeric array')
        H = raw_h.astype(float, copy=False)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError('hamiltonian must be a finite real square matrix') from exc
    if H.ndim != 2 or H.shape[0] != H.shape[1]:
        raise ValueError('hamiltonian must be square')
    if not np.all(np.isfinite(H)):
        raise ValueError('hamiltonian entries must be finite')
    try:
        idx = np.asarray(indices)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError('indices must be a one-dimensional integer array') from exc
    if idx.ndim != 1:
        raise ValueError('indices must be one dimensional')
    if idx.dtype.kind not in 'iu':
        raise ValueError('indices must contain integers')
    if idx.size < 2:
        raise ValueError('indices must hold at least two entries')
    if np.any(idx[1:] <= idx[:-1]):
        raise ValueError('indices must be strictly increasing')
    if idx.min() < 0 or idx.max() >= H.shape[0]:
        raise ValueError('indices must be valid rows of the hamiltonian')
    idx = idx.astype(np.intp, copy=False)
    block = H[np.ix_(idx, idx)]
    return float(lowest_spectral_gap(block))

import numpy as np

def isolated_meson_energy(n_colors: int, coupling: float, mass: float) -> float:
    import numpy as np

    if (isinstance(n_colors, (bool, np.bool_, np.timedelta64, np.datetime64))
            or not isinstance(n_colors, (int, np.integer))):
        raise ValueError('n_colors must be an integer')
    nc_i = int(n_colors)
    if nc_i < 2:
        raise ValueError('n_colors must be at least 2')

    def _finite_real(value, name):
        if (isinstance(value, (bool, np.bool_, np.timedelta64, np.datetime64))
                or not isinstance(value, (int, float, np.integer, np.floating))):
            raise ValueError(f'{name} must be a finite real scalar')
        try:
            result = float(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f'{name} must be a finite real scalar') from exc
        if not np.isfinite(result):
            raise ValueError(f'{name} must be a finite real scalar')
        return result

    try:
        nc = float(nc_i)
    except OverflowError as exc:
        raise ValueError('n_colors is too large to evaluate') from exc
    if not np.isfinite(nc):
        raise ValueError('n_colors is too large to evaluate')
    g = _finite_real(coupling, 'coupling')
    m = _finite_real(mass, 'mass')
    with np.errstate(over='ignore', invalid='ignore'):
        c2 = 0.5 * (nc - 1.0 / nc)
        energy = 2.0 * m + (g * (0.5 * c2)) * g
        if not np.isfinite(energy):
            # Rescale only overflowing sums so subnormal masses are preserved.
            energy = 4.0 * (0.5 * m + (g * (0.125 * c2)) * g)
    if not np.isfinite(energy):
        raise ValueError('inputs must produce a finite isolated-meson energy')
    return float(energy)

import numpy as np

def compute_truncation_gap_index(n_links: int, n_colors: int, coupling: float,
                                         mass: float) -> float:
    import numpy as np

    if (isinstance(n_links, (bool, np.bool_, np.timedelta64, np.datetime64))
            or not isinstance(n_links, (int, np.integer))):
        raise ValueError('n_links must be an integer')
    if (isinstance(n_colors, (bool, np.bool_, np.timedelta64, np.datetime64))
            or not isinstance(n_colors, (int, np.integer))):
        raise ValueError('n_colors must be an integer')
    if int(n_links) not in (4, 6):
        raise ValueError('n_links must be 4 or 6')
    if not 3 <= int(n_colors) <= 64:
        raise ValueError('n_colors must be between 3 and 64')
    for value, name in ((coupling, 'coupling'), (mass, 'mass')):
        if (isinstance(value, (bool, np.bool_, np.timedelta64, np.datetime64))
                or not isinstance(value, (int, float, np.integer, np.floating))):
            raise ValueError(f'{name} must be a finite real scalar')
        try:
            finite = np.isfinite(float(value))
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f'{name} must be a finite real scalar') from exc
        if not finite:
            raise ValueError(f'{name} must be a finite real scalar')

    pair = isolated_meson_energy(n_colors, coupling, mass)
    if pair == 0.0:
        raise ValueError('isolated meson energy must not be zero')
    h3 = assemble_t3_hamiltonian(n_links, n_colors, coupling, mass)
    h1 = assemble_t1_hamiltonian(n_links, n_colors, coupling, mass)
    block3 = vacuum_connected_indices(h3, 1e-12)
    block1 = vacuum_connected_indices(h1, 1e-12)
    gap3 = subspace_gap(h3, block3)
    # Compute the T1 gap from its principal vacuum block.
    gap1 = lowest_spectral_gap(h1[np.ix_(block1, block1)])
    with np.errstate(over='ignore', divide='ignore', invalid='ignore'):
        result = (gap3 - gap1) / pair
    if not np.isfinite(result):
        raise ValueError('parameters must produce a finite truncation gap index')
    return float(result)
SCICODE_GOLD_EOF
