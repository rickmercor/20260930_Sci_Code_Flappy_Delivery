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

def _spin_operators():
    """Quasispin I (I=1) and hole spin S (S=1/2) in the ordered product basis."""
    root2 = np.sqrt(2.0)
    ix = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=float) / root2
    iy = np.array([[0, -1j, 0], [1j, 0, -1j], [0, 1j, 0]]) / root2
    iz = np.diag([1.0, 0.0, -1.0]).astype(complex)
    sx = np.array([[0, 1], [1, 0]], dtype=complex) / 2.0
    sy = np.array([[0, -1j], [1j, 0]]) / 2.0
    sz = np.diag([1.0, -1.0]).astype(complex) / 2.0
    e3, e2 = np.eye(3), np.eye(2)
    return ([np.kron(m, e2) for m in (ix, iy, iz)],
            [np.kron(e3, m) for m in (sx, sy, sz)])

def spin_orbit_hamiltonian(delta: float) -> "np.ndarray":
    d = float(delta)
    if not np.isfinite(d) or d < 0.0:
        raise ValueError("delta must be finite and non-negative")
    iop, sop = _spin_operators()
    i_dot_s = sum(iop[a] @ sop[a] for a in range(3))
    return np.real((2.0 / 3.0) * d * (np.eye(6) + i_dot_s))

import numpy as np

def _spin_operators():
    """Quasispin I (I=1) and hole spin S (S=1/2) in the ordered product basis."""
    root2 = np.sqrt(2.0)
    ix = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=float) / root2
    iy = np.array([[0, -1j, 0], [1j, 0, -1j], [0, 1j, 0]]) / root2
    iz = np.diag([1.0, 0.0, -1.0]).astype(complex)
    sx = np.array([[0, 1], [1, 0]], dtype=complex) / 2.0
    sy = np.array([[0, -1j], [1j, 0]]) / 2.0
    sz = np.diag([1.0, -1.0]).astype(complex) / 2.0
    e3, e2 = np.eye(3), np.eye(2)
    return ([np.kron(m, e2) for m in (ix, iy, iz)],
            [np.kron(e3, m) for m in (sx, sy, sz)])

def luttinger_kohn_matrices(gamma1: float, gamma2: float, gamma3: float, eta1: float, eta2: float, eta3: float) -> "np.ndarray":
    vals = [float(v) for v in (gamma1, gamma2, gamma3, eta1, eta2, eta3)]
    if not all(np.isfinite(v) for v in vals):
        raise ValueError("all six band parameters must be finite")
    g1, g2, g3, n1, n2, n3 = vals
    iop, sop = _spin_operators()
    i_dot_s = sum(iop[a] @ sop[a] for a in range(3))
    blocks = [(g1 + 4.0 * g2) * np.eye(6) + 2.0 * (n1 + 2.0 * n2) * i_dot_s]
    for a in range(3):
        blocks.append(-6.0 * g2 * (iop[a] @ iop[a]) - 12.0 * n2 * (iop[a] @ sop[a]))
    for a, b in ((0, 1), (1, 2), (2, 0)):
        anti = (iop[a] @ iop[b] + iop[b] @ iop[a]) / 2.0
        blocks.append(-12.0 * g3 * anti - 12.0 * n3 * (iop[a] @ sop[b] + iop[b] @ sop[a]))
    return np.stack(blocks)

import numpy as np

def momentum_matrix_elements(width: float, num_modes: int) -> "np.ndarray":
    lw = float(width)
    if not np.isfinite(lw) or lw <= 0.0:
        raise ValueError("width must be finite and strictly positive")
    if isinstance(num_modes, bool) or not isinstance(num_modes, (int, np.integer)) or int(num_modes) < 1:
        raise ValueError("num_modes must be an integer >= 1")
    m = int(num_modes)
    d = np.zeros((m, m))
    for n in range(1, m + 1):
        for p in range(1, m + 1):
            if (n + p) % 2 == 0 or n % 2 == 0:
                continue
            d[n - 1, p - 1] = 4.0 * n * p / (lw * (n * n - p * p)) * (-1) ** (((n - p - 1) // 2) % 2)
    return -1j * (d - d.T)

import numpy as np

def hole_hamiltonian(width: float, k_inplane: float, direction: tuple, num_modes: int, params: dict) -> "np.ndarray":
    lw = float(width)
    if not np.isfinite(lw) or lw <= 0.0:
        raise ValueError("width must be finite and strictly positive")
    kk = float(k_inplane)
    if not np.isfinite(kk) or kk < 0.0:
        raise ValueError("k_inplane must be finite and non-negative")
    try:
        unit = np.asarray(direction, dtype=float)
    except Exception as exc:
        raise ValueError("direction must be a finite nonzero two-vector") from exc
    if unit.shape != (2,) or not np.all(np.isfinite(unit)):
        raise ValueError("direction must be a finite nonzero two-vector")
    norm = float(np.linalg.norm(unit))
    if norm == 0.0:
        raise ValueError("direction must be a finite nonzero two-vector")
    unit = unit / norm
    if isinstance(num_modes, bool) or not isinstance(num_modes, (int, np.integer)) or int(num_modes) < 1:
        raise ValueError("num_modes must be an integer >= 1")
    if not isinstance(params, dict):
        raise ValueError("params must be a dict")
    for key in ("gamma1", "gamma2", "gamma3", "eta1", "eta2", "eta3", "delta"):
        if key not in params:
            raise ValueError("params is missing the key " + key)
        try:
            value = float(params[key])
        except Exception as exc:
            raise ValueError("params[" + key + "] must be finite") from exc
        if not np.isfinite(value):
            raise ValueError("params[" + key + "] must be finite")
    if float(params["delta"]) <= 0.0:
        raise ValueError("params['delta'] must be strictly positive")

    m = int(num_modes)
    lk = luttinger_kohn_matrices(
        params["gamma1"], params["gamma2"], params["gamma3"],
        params["eta1"], params["eta2"], params["eta3"],
    )
    h_so = spin_orbit_hamiltonian(float(params["delta"]))
    pz = momentum_matrix_elements(lw, m)
    pz2 = np.diag([(n * np.pi / lw) ** 2 for n in range(1, m + 1)]).astype(complex)
    eye_m = np.eye(m, dtype=complex)
    kx, ky = kk * unit

    ham = np.kron((kx * kx + ky * ky) * eye_m + pz2, lk[0].astype(complex))
    ham += np.kron(kx * kx * eye_m, lk[1].astype(complex))
    ham += np.kron(ky * ky * eye_m, lk[2].astype(complex))
    ham += np.kron(pz2, lk[3].astype(complex))
    ham += np.kron(kx * ky * eye_m, lk[4].astype(complex))
    ham += np.kron(ky * pz, lk[5].astype(complex))
    ham += np.kron(kx * pz, lk[6].astype(complex))
    ham = 38.0998212 * ham + np.kron(eye_m, h_so.astype(complex))
    return 0.5 * (ham + ham.conj().T)

import numpy as np

def tracked_topmost_subband_path(width: float, k_values: "np.ndarray", direction: tuple, num_modes: int, params: dict) -> "np.ndarray":
    try:
        path = np.asarray(k_values, dtype=float)
    except Exception as exc:
        raise ValueError("k_values must be a finite one-dimensional path") from exc
    if path.ndim != 1 or path.size == 0 or not np.all(np.isfinite(path)):
        raise ValueError("k_values must be a finite one-dimensional path")
    if path[0] != 0.0 or np.any(path < 0.0) or np.any(np.diff(path) <= 0.0):
        raise ValueError("k_values must start at zero and then increase strictly")

    first_ham = hole_hamiltonian(width, float(path[0]), direction, num_modes, params)
    splitting = float(params["delta"])
    levels, states = np.linalg.eigh(spin_orbit_hamiltonian(splitting))
    upper = states[:, levels > 0.5 * splitting]
    internal_green = upper @ upper.conj().T
    full_green = np.kron(np.eye(int(num_modes)), internal_green)

    rows = []
    previous_projector = None
    for point, kk in enumerate(path):
        ham = first_ham if point == 0 else hole_hamiltonian(
            width, float(kk), direction, num_modes, params,
        )
        energies, vectors = np.linalg.eigh(ham)
        pair_count = energies.size // 2

        if previous_projector is None:
            selected = 0
        else:
            scores = np.empty(pair_count, dtype=float)
            for pair_index in range(pair_count):
                candidate = vectors[:, 2 * pair_index:2 * pair_index + 2]
                candidate_projector = candidate @ candidate.conj().T
                scores[pair_index] = float(np.real(np.trace(
                    previous_projector @ candidate_projector
                )))
            selected = int(np.argmax(scores))

        pair = vectors[:, 2 * selected:2 * selected + 2]
        previous_projector = pair @ pair.conj().T
        restricted = pair.conj().T @ full_green @ pair
        restricted = 0.5 * (restricted + restricted.conj().T)
        green = np.linalg.eigvalsh(restricted)
        lo = 2 * selected
        rows.append([
            0.5 * float(energies[lo] + energies[lo + 1]),
            float(abs(energies[lo + 1] - energies[lo])),
            float(np.real(green[0])),
            float(np.real(green[1])),
            float(selected),
        ])
    return np.asarray(rows, dtype=float)

import numpy as np
from scipy.optimize import brentq

def mixing_crossover_wavevector(width: float, direction: tuple, num_modes: int, params: dict, k_max: float) -> float:
    top = float(k_max)
    if not np.isfinite(top) or top <= 0.0:
        raise ValueError("k_max must be finite and strictly positive")

    grid = np.linspace(0.0, top, 121)
    tracked = tracked_topmost_subband_path(
        width, grid, direction, num_modes, params,
    )
    zone_centre = float(tracked[0, 0])
    stiffness = 38.0998212 * float(params["gamma1"])
    residual = tracked[:, 0] - (zone_centre + stiffness * grid * grid)

    bracket = None
    for index in range(1, grid.size - 1):
        if residual[index] * residual[index + 1] <= 0.0:
            bracket = (index, index + 1)
            break
    if bracket is None:
        raise ValueError("no tracked nonzero crossing below k_max")

    left, right = bracket
    pair_index = int(round(float(tracked[left, 4])))

    def _residual(kk):
        ham = hole_hamiltonian(width, float(kk), direction, num_modes, params)
        energies = np.linalg.eigvalsh(ham)
        lo = 2 * pair_index
        exact = 0.5 * float(energies[lo] + energies[lo + 1])
        return exact - (zone_centre + stiffness * float(kk) ** 2)

    return float(brentq(
        _residual, float(grid[left]), float(grid[right]),
        xtol=1e-13, rtol=8.881784197001252e-16,
    ))

import numpy as np

def green_admixture_at_crossover(width: float, direction: tuple, num_modes: int, params: dict, k_max: float) -> float:
    crossing = mixing_crossover_wavevector(
        width, direction, num_modes, params, k_max,
    )
    path = np.linspace(0.0, float(crossing), 121)
    tracked = tracked_topmost_subband_path(
        width, path, direction, num_modes, params,
    )
    value = 0.5 * float(tracked[-1, 2] + tracked[-1, 3])
    return float(np.clip(value, 0.0, 1.0))
SCICODE_GOLD_EOF
