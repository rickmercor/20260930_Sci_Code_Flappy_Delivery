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

def build_transmutation_matrix(flux: float, sigma_gamma: "np.ndarray", sigma_f: "np.ndarray", fission_yields: "np.ndarray") -> "np.ndarray":
    """Reference implementation."""
    if flux < 0:
        raise ValueError("flux must be non-negative")
    m = len(sigma_gamma)
    T = np.zeros((m, m))

    # Total absorption = capture + fission (convert barns to cm²)
    sigma_a = (sigma_gamma + sigma_f) * 1e-24

    for i in range(m):
        # Diagonal: destruction by neutron absorption
        T[i, i] = -flux * sigma_a[i]

        for j in range(m):
            if j == i:
                continue
            # Fission yield production: flux * y_{j->i} * sigma_f_j (barn -> cm²)
            T[i, j] += flux * fission_yields[j, i] * sigma_f[j] * 1e-24

    return T

import numpy as np

def build_cell_matrix(T: "np.ndarray", decay_constants: "np.ndarray", branching_ratios: "np.ndarray", removal_rates: "np.ndarray") -> "np.ndarray":
    """Reference implementation."""
    if np.any(np.asarray(decay_constants) < 0):
        raise ValueError("decay constants must be non-negative")
    m = T.shape[0]
    A = T.copy()

    for i in range(m):
        # Diagonal: decay loss and engineered removal
        A[i, i] -= decay_constants[i] + removal_rates[i]

        for j in range(m):
            if j == i:
                continue
            # Decay production: b_{j->i} * lambda_j
            A[i, j] += branching_ratios[j, i] * decay_constants[j]

    return A

import numpy as np

def build_flow_coupling(n_cells: int, m: int, volumes: "np.ndarray", Q: float, flow_fractions: "np.ndarray") -> tuple[list, list]:
    """Reference implementation."""
    if n_cells < 1 or Q < 0 or np.any(np.asarray(volumes) <= 0):
        raise ValueError("need n_cells >= 1, Q >= 0 and positive volumes")
    outflow_diags = []
    inflow_blocks = [[np.zeros((m, m)) for _ in range(n_cells)] for _ in range(n_cells)]

    for k in range(n_cells):
        # Total outflow fraction from cell k
        total_outflow_frac = np.sum(flow_fractions[k, :])
        # Outflow diagonal: -(Q/V_k) * total_outflow_frac * I
        outflow_diags.append(-Q / volumes[k] * total_outflow_frac * np.eye(m))

        for l in range(n_cells):
            if flow_fractions[k, l] > 0.0:
                # Inflow to cell l from cell k: +(Q/V_l) * f_{k->l} * I
                inflow_blocks[k][l] = Q / volumes[l] * flow_fractions[k, l] * np.eye(m)

    return outflow_diags, inflow_blocks

import numpy as np

def assemble_system_matrix(cell_matrices: list, outflow_diags: list, inflow_blocks: list) -> "np.ndarray":
    """Reference implementation."""
    if len(cell_matrices) == 0:
        raise ValueError("cell_matrices must be non-empty")
    n = len(cell_matrices)
    m = cell_matrices[0].shape[0]
    size = m * n
    R = np.zeros((size, size))

    for k in range(n):
        # Diagonal block: cell matrix + outflow
        row_start = k * m
        row_end = (k + 1) * m
        R[row_start:row_end, row_start:row_end] = cell_matrices[k] + outflow_diags[k]

        # Add all inflows, including flow returning to the same cell.
        for j in range(n):
            col_start = j * m
            col_end = (j + 1) * m
            # inflow_blocks[j][k] = flow FROM cell j TO cell k
            R[row_start:row_end, col_start:col_end] += inflow_blocks[j][k]

    return R

import numpy as np

def _cram16_action(A, v):
    """Ordinary PFD action; Pusa (2012), Eq. (5) and Table 2."""
    THETA = np.array([
     -1.0843917078696988026e1 + 1.9277446167181652284e1j,
     -5.2649713434426468895 + 1.6220221473167927305e1j,
      5.9481522689511774808 + 3.5874573620183222829j,
      3.5091036084149180974 + 8.4361989858843750826j,
      6.4161776990994341923 + 1.1941223933701386874j,
      1.4193758971856659786 + 1.0925363484496722585e1j,
      4.9931747377179963991 + 5.9968817136039422260j,
     -1.4139284624888862114 + 1.3497725698892745389e1j,
    ])
    ALPHA = np.array([
     -5.0901521865224915650e-7 - 2.4220017652852287970e-5j,
      2.1151742182466030907e-4 + 4.3892969647380673918e-3j,
      1.1339775178483930527e2 + 1.0194721704215856450e2j,
      1.5059585270023467528e1 - 5.7514052776421819979j,
     -6.4500878025539646595e1 - 2.2459440762652096056e2j,
     -1.4793007113557999718 + 1.7686588323782937906j,
     -6.2518392463207918892e1 - 1.1190391094283228480e1j,
      4.1023136835410021273e-2 - 1.5743466173455468191e-1j,
    ])
    ALPHA0 = 2.1248537104952237488e-16
    identity = np.eye(A.shape[0])
    terms = [np.real(alpha * np.linalg.solve(A - theta * identity, v))
             for theta, alpha in zip(THETA, ALPHA)]
    return ALPHA0 * v + 2.0 * np.sum(terms, axis=0)

def solve_timestep(R: "np.ndarray", dt: float, N: "np.ndarray", S: "np.ndarray") -> "np.ndarray":
    """Reference implementation."""
    if dt <= 0:
        raise ValueError("dt must be positive")
    if not np.any(S != 0.0):
        return _cram16_action(R * dt, N)

    # A constant final coordinate integrates the source without cancellation.
    size = R.shape[0]
    augmented = np.zeros((size + 1, size + 1))
    augmented[:size, :size] = R
    augmented[:size, size] = S
    return _cram16_action(augmented * dt, np.r_[N, 1.0])[:size]

import numpy as np

def run_msre_depletion(n_cells: int, m: int, fluxes: "np.ndarray", volumes: "np.ndarray", Q: float, flow_fractions: "np.ndarray", sigma_gamma: "np.ndarray", sigma_f: "np.ndarray", fission_yields: "np.ndarray", decay_constants: "np.ndarray", branching_ratios: "np.ndarray", removal_rates: "np.ndarray", addition_rates: "np.ndarray", N_0: "np.ndarray", total_time: float, n_steps: int) -> float:
    """Reference implementation."""
    if n_steps < 1 or total_time < 0:
        raise ValueError("need n_steps >= 1 and total_time >= 0")
    cell_matrices = []
    for c in range(n_cells):
        T = build_transmutation_matrix(
            fluxes[c], sigma_gamma, sigma_f, fission_yields)
        A = build_cell_matrix(
            T, decay_constants, branching_ratios, removal_rates[c])
        cell_matrices.append(A)

    outflow_diags, inflow_blocks = build_flow_coupling(
        n_cells, m, volumes, Q, flow_fractions)
    R = assemble_system_matrix(cell_matrices, outflow_diags, inflow_blocks)

    N = N_0.copy() if len(N_0) == n_cells * m else np.tile(N_0, n_cells)
    if total_time == 0:
        return float(N[3])
    S = addition_rates.flatten()

    dt = total_time / n_steps
    for _ in range(n_steps):
        N = solve_timestep(R, dt, N, S)

    return N[3]

import numpy as np
from scipy.optimize import minimize_scalar

def find_optimal_flow(n_cells: int, m: int, fluxes: "np.ndarray", volumes: "np.ndarray", flow_fractions: "np.ndarray", sigma_gamma: "np.ndarray", sigma_f: "np.ndarray", fission_yields: "np.ndarray", decay_constants: "np.ndarray", branching_ratios: "np.ndarray", removal_rates: "np.ndarray", addition_rates: "np.ndarray", N_0: "np.ndarray", total_time: float, n_steps: int, Q_min: float, Q_max: float) -> float:
    """Reference implementation."""

    if Q_min > Q_max:
        raise ValueError("Q_min must not exceed Q_max")
    def _objective(Q):
        return -run_msre_depletion(
            n_cells, m, fluxes, volumes, Q, flow_fractions,
            sigma_gamma, sigma_f, fission_yields,
            decay_constants, branching_ratios, removal_rates,
            addition_rates,
            N_0, total_time, n_steps)

    result = minimize_scalar(_objective, bounds=(Q_min, Q_max), method='bounded')
    candidates = [float(Q_min), float(result.x), float(Q_max)]
    Q_opt = min(candidates, key=lambda Q: (_objective(Q), Q))

    return float(f"{Q_opt:.4g}")
SCICODE_GOLD_EOF
