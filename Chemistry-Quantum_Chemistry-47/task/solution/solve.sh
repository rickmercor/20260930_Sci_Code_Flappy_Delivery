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

def build_model_matrices(positions: "np.ndarray", types: "np.ndarray", params: dict) -> "np.ndarray":
    pos = np.asarray(positions, dtype=float)
    typ = np.asarray(types, dtype=int)
    if pos.ndim != 1 or typ.ndim != 1 or pos.shape[0] != typ.shape[0]:
        raise ValueError("positions and types must be 1-D arrays of the same length")
    if typ.size and (typ.min() < 0 or typ.max() > 1):
        raise ValueError("types must be 0 or 1")
    alpha = np.asarray(params["onsite_energy"], dtype=float)
    u_onsite = np.asarray(params["onsite_repulsion"], dtype=float)
    kappa = float(params["kappa"])
    sigma = float(params["sigma"])
    n = pos.shape[0]
    R = np.abs(pos[:, None] - pos[None, :])
    off = ~np.eye(n, dtype=bool)
    S = np.where(off, np.exp(-R ** 2 / (2.0 * sigma ** 2)), 1.0)
    a = 2.0 / (u_onsite[typ][:, None] + u_onsite[typ][None, :])
    gamma = 1.0 / np.sqrt(R ** 2 + a ** 2)
    hdiag = alpha[typ] - np.sum(gamma * off, axis=1)
    H = np.diag(hdiag) + kappa * S * off * 0.5 * (hdiag[:, None] + hdiag[None, :])
    return np.stack([S, H, gamma])

import numpy as np

def mean_field_energy(H: "np.ndarray", gamma: "np.ndarray", P: "np.ndarray", positions: "np.ndarray", params: dict) -> float:
    H = np.asarray(H, dtype=float)
    gamma = np.asarray(gamma, dtype=float)
    P = np.asarray(P, dtype=float)
    pos = np.asarray(positions, dtype=float)
    n = H.shape[0]
    if H.shape != (n, n) or gamma.shape != (n, n) or P.shape != (n, n) or pos.shape != (n,):
        raise ValueError("H, gamma and P must be (n, n) and positions (n,)")
    F = _build_fock(H, gamma, P)
    return float(0.5 * np.sum(P * (H + F)) + _core_repulsion(pos, params))

def _build_fock(H: "np.ndarray", gamma: "np.ndarray", P: "np.ndarray") -> "np.ndarray":
    return H + np.diag(gamma @ np.diag(P)) - 0.5 * P * gamma

def _core_repulsion(positions: "np.ndarray", params: dict) -> float:
    pos = np.asarray(positions, dtype=float)
    strength = float(params["core_repulsion_strength"])
    rho = float(params["core_repulsion_range"])
    iu = np.triu_indices(pos.shape[0], 1)
    R = np.abs(pos[:, None] - pos[None, :])[iu]
    return float(np.sum(1.0 / R + strength * np.exp(-R / rho)))

import numpy as np
from scipy.linalg import eigh

def scf_reference_energy(positions: "np.ndarray", types: "np.ndarray", params: dict, tol: float = 1e-10, max_iter: int = 500, damping: float = 0.5) -> float:
    pos = np.asarray(positions, dtype=float)
    n = pos.shape[0]
    if n % 2:
        raise ValueError("the model needs an even number of atoms")
    S, H, gamma = build_model_matrices(pos, types, params)
    P = np.zeros((n, n))
    for _ in range(int(max_iter)):
        F = _build_fock(H, gamma, P)
        P_new = _occupied_density(F, S, n)
        if np.max(np.abs(P_new - P)) < tol:
            return mean_field_energy(H, gamma, P_new, pos, params)
        P = (1.0 - damping) * P + damping * P_new
    raise ValueError("SCF did not converge within max_iter iterations")

def _occupied_density(M: "np.ndarray", S: "np.ndarray", n_electrons: int) -> "np.ndarray":
    if n_electrons % 2:
        raise ValueError("closed-shell density needs an even electron count")
    _, C = eigh(np.asarray(M, dtype=float), np.asarray(S, dtype=float))
    occ = C[:, : n_electrons // 2]
    return 2.0 * occ @ occ.T

def _converged_scf(positions: "np.ndarray", types: "np.ndarray", params: dict, tol: float = 1e-12, max_iter: int = 5000) -> tuple:
    """Overlap, core Hamiltonian, Ohno matrix, density, orbital energies and coefficients at convergence."""
    pos = np.asarray(positions, dtype=float)
    n = pos.shape[0]
    if n % 2:
        raise ValueError("the model needs an even number of atoms")
    S, H, gamma = build_model_matrices(pos, types, params)
    P = np.zeros((n, n))
    for _ in range(int(max_iter)):
        F = _build_fock(H, gamma, P)
        w, C = eigh(F, S)
        occ = C[:, : n // 2]
        P_new = 2.0 * occ @ occ.T
        if np.max(np.abs(P_new - P)) < tol:
            return S, H, gamma, P_new, w, C
        P = 0.5 * (P + P_new)
    raise ValueError("SCF did not converge within max_iter iterations")

import numpy as np

def scf_energy_gradient(positions: "np.ndarray", types: "np.ndarray", params: dict) -> "np.ndarray":
    pos = np.asarray(positions, dtype=float)
    typ = np.asarray(types, dtype=int)
    n = pos.shape[0]
    S, H, gamma, P, w, C = _converged_scf(pos, typ, params)
    kappa = float(params["kappa"])
    sigma = float(params["sigma"])
    strength = float(params["core_repulsion_strength"])
    rho = float(params["core_repulsion_range"])
    occ = C[:, : n // 2]
    weighted = 2.0 * (occ * w[: n // 2]) @ occ.T
    R = np.abs(pos[:, None] - pos[None, :])
    sign = np.sign(pos[:, None] - pos[None, :])
    off = ~np.eye(n, dtype=bool)
    hdiag = np.diag(H)
    p = np.diag(P)
    pair_weight = np.outer(p, p) - 0.5 * P ** 2
    iu = np.triu_indices(n, 1)
    core_force = -1.0 / R[iu] ** 2 - strength / rho * np.exp(-R[iu] / rho)
    gradient = np.zeros(n)
    for a in range(n):
        dR = np.zeros((n, n))
        dR[a, :] = sign[a, :]
        dR[:, a] = -sign[:, a]
        dS = S * (-R / sigma ** 2) * dR * off
        dgamma = -R * gamma ** 3 * dR
        dhdiag = -np.sum(dgamma * off, axis=1)
        dH = np.diag(dhdiag) + kappa * off * (dS * 0.5 * (hdiag[:, None] + hdiag[None, :])
                                              + S * 0.5 * (dhdiag[:, None] + dhdiag[None, :]))
        one_electron = np.sum(P * dH)
        two_electron = 0.5 * np.sum(pair_weight * dgamma)
        pulay = -np.sum(weighted * dS)
        cores = np.sum(core_force * dR[iu])
        gradient[a] = one_electron + two_electron + pulay + cores
    return gradient

import numpy as np

def scf_energy_hessian(positions: "np.ndarray", types: "np.ndarray", params: dict) -> "np.ndarray":
    pos = np.asarray(positions, dtype=float)
    typ = np.asarray(types, dtype=int)
    n = pos.shape[0]
    S, H, gamma, P, eps, C = _converged_scf(pos, typ, params)
    kappa = float(params["kappa"])
    sigma = float(params["sigma"])
    strength = float(params["core_repulsion_strength"])
    rho = float(params["core_repulsion_range"])
    nocc = n // 2
    occ = C[:, :nocc]
    weighted = 2.0 * (occ * eps[:nocc]) @ occ.T
    R = np.abs(pos[:, None] - pos[None, :])
    sign = np.sign(pos[:, None] - pos[None, :])
    off = ~np.eye(n, dtype=bool)
    hdiag = np.diag(H)
    p = np.diag(P)
    pair_weight = np.outer(p, p) - 0.5 * P ** 2
    S1 = S * (-R / sigma ** 2) * off
    S2 = S * (R ** 2 / sigma ** 4 - 1.0 / sigma ** 2) * off
    G1 = -R * gamma ** 3 * off
    G2 = (-gamma ** 3 + 3.0 * R ** 2 * gamma ** 5) * off
    iu = np.triu_indices(n, 1)
    core_second = 2.0 / R[iu] ** 3 + strength / rho ** 2 * np.exp(-R[iu] / rho)
    steps = []
    for a in range(n):
        d = np.zeros((n, n))
        d[a, :] = sign[a, :]
        d[:, a] = -sign[:, a]
        steps.append(d)
    firsts = []
    for a in range(n):
        dS = S1 * steps[a]
        dG = G1 * steps[a]
        dh = -np.sum(dG * off, axis=1)
        dH = np.diag(dh) + kappa * off * (dS * 0.5 * (hdiag[:, None] + hdiag[None, :])
                                          + S * 0.5 * (dh[:, None] + dh[None, :]))
        firsts.append((dS, dG, dh, dH + _two_electron_part(dG, P)))
    responses = [_orbital_response(S, gamma, C, eps, nocc, firsts[b][0], firsts[b][3]) for b in range(n)]
    hessian = np.zeros((n, n))
    for a in range(n):
        dSa, dGa, dha, dFa = firsts[a]
        for b in range(n):
            dSb, dGb, dhb, dFb = firsts[b]
            dd = steps[a] * steps[b]
            S_ab = S2 * dd
            G_ab = G2 * dd
            h_ab = -np.sum(G_ab * off, axis=1)
            H_ab = np.diag(h_ab) + kappa * off * (S_ab * 0.5 * (hdiag[:, None] + hdiag[None, :])
                                                  + dSa * 0.5 * (dhb[:, None] + dhb[None, :])
                                                  + dSb * 0.5 * (dha[:, None] + dha[None, :])
                                                  + S * 0.5 * (h_ab[:, None] + h_ab[None, :]))
            explicit = (np.sum(P * H_ab) + 0.5 * np.sum(pair_weight * G_ab) - np.sum(weighted * S_ab)
                        + np.sum(core_second * steps[a][iu] * steps[b][iu]))
            P_b, W_b = responses[b]
            hessian[a, b] = explicit + np.sum(P_b * dFa) - np.sum(W_b * dSa)
    return hessian

def _two_electron_part(gamma: "np.ndarray", P: "np.ndarray") -> "np.ndarray":
    return np.diag(gamma @ np.diag(P)) - 0.5 * P * gamma

def _orbital_response(S, gamma, C, eps, nocc, dS, dF_explicit):
    """First-order density and energy-weighted density for one displacement, from coupled-perturbed SCF."""
    n = C.shape[0]
    Sm = C.T @ dS @ C
    Fm = C.T @ dF_explicit @ C
    P_occ = np.zeros((n, n))
    for i in range(nocc):
        for j in range(nocc):
            P_occ -= 2.0 * Sm[i, j] * np.outer(C[:, j], C[:, i])
    pairs = [(a, i) for a in range(nocc, n) for i in range(nocc)]
    m = len(pairs)

    def _density(u):
        out = P_occ.copy()
        for k, (a, i) in enumerate(pairs):
            out += 2.0 * u[k] * (np.outer(C[:, a], C[:, i]) + np.outer(C[:, i], C[:, a]))
        return out

    base = C.T @ _two_electron_part(gamma, P_occ) @ C
    rhs = np.array([eps[i] * Sm[a, i] - Fm[a, i] - base[a, i] for a, i in pairs])
    matrix = np.zeros((m, m))
    for col in range(m):
        unit = np.zeros(m)
        unit[col] = 1.0
        coupling = C.T @ _two_electron_part(gamma, _density(unit) - P_occ) @ C
        for k, (a, i) in enumerate(pairs):
            matrix[k, col] = (eps[a] - eps[i]) * (1.0 if k == col else 0.0) + coupling[a, i]
    u = np.linalg.solve(matrix, rhs) if m else np.zeros(0)
    P_b = _density(u)
    F_total = C.T @ (dF_explicit + _two_electron_part(gamma, P_b)) @ C
    U = np.zeros((n, n))
    for i in range(nocc):
        for q in range(n):
            U[q, i] = -0.5 * Sm[i, i] if q == i else (eps[i] * Sm[q, i] - F_total[q, i]) / (eps[q] - eps[i])
    eps_b = np.array([F_total[i, i] - eps[i] * Sm[i, i] for i in range(nocc)])
    C_b = C @ U
    W_b = np.zeros((n, n))
    for i in range(nocc):
        W_b += 2.0 * (eps_b[i] * np.outer(C[:, i], C[:, i])
                      + eps[i] * (np.outer(C_b[:, i], C[:, i]) + np.outer(C[:, i], C_b[:, i])))
    return P_b, W_b

import numpy as np
from scipy.optimize import minimize

def equilibrium_bonds(types: "np.ndarray", start_bonds: "np.ndarray", params: dict) -> "np.ndarray":
    typ = np.asarray(types, dtype=int)
    b0 = np.asarray(start_bonds, dtype=float)
    if typ.shape[0] % 2:
        raise ValueError("the model needs an even number of atoms")
    if b0.shape != (typ.shape[0] - 1,) or np.any(b0 <= 0.0):
        raise ValueError("start_bonds must hold one positive length per consecutive pair")
    energy = lambda x: scf_reference_energy(_bonds_to_positions(np.exp(x)), typ, params, 1e-12, 20000, 0.3)
    gradient = lambda x: _bond_gradient(np.exp(x), typ, params) * np.exp(x)
    result = minimize(energy, np.log(b0), jac=gradient, method="BFGS", options={"gtol": 1e-11, "maxiter": 500})
    bonds = np.exp(np.asarray(result.x, dtype=float))
    for _ in range(20):
        g = _bond_gradient(bonds, typ, params)
        if np.max(np.abs(g)) < 1e-13:
            break
        step = np.linalg.solve(_bond_hessian(bonds, typ, params), g)
        while np.any(bonds - step <= 0.0):
            step = 0.5 * step
        bonds = bonds - step
    return bonds

def _bonds_to_positions(bonds: "np.ndarray") -> "np.ndarray":
    return np.concatenate([[0.0], np.cumsum(np.asarray(bonds, dtype=float))])

def _bond_gradient(bonds: "np.ndarray", types: "np.ndarray", params: dict) -> "np.ndarray":
    g = scf_energy_gradient(_bonds_to_positions(bonds), types, params)
    return np.array([g[k + 1:].sum() for k in range(len(bonds))])

def _bond_hessian(bonds: "np.ndarray", types: "np.ndarray", params: dict) -> "np.ndarray":
    hess = scf_energy_hessian(_bonds_to_positions(bonds), types, params)
    k = len(bonds)
    return np.array([[hess[i + 1:, j + 1:].sum() for j in range(k)] for i in range(k)])

import numpy as np

def normal_modes(types: "np.ndarray", eq_bonds: "np.ndarray", params: dict, step: float = 1e-4) -> "np.ndarray":
    typ = np.asarray(types, dtype=int)
    bonds = np.asarray(eq_bonds, dtype=float)
    if typ.shape[0] % 2:
        raise ValueError("the model needs an even number of atoms")
    if bonds.shape != (typ.shape[0] - 1,):
        raise ValueError("eq_bonds must hold one length per consecutive pair")
    hess = _bond_hessian_by_differences(bonds, typ, params, float(step))
    values, vectors = np.linalg.eigh(hess)
    if np.any(values <= 0.0):
        raise ValueError("the curvature matrix is not positive definite")
    modes = vectors.T.copy()
    for row in modes:
        if row[np.argmax(np.abs(row))] < 0.0:
            row *= -1.0
    return np.vstack([np.sqrt(values), modes])

def _bond_hessian_by_differences(bonds: "np.ndarray", types: "np.ndarray", params: dict, step: float) -> "np.ndarray":
    bonds = np.asarray(bonds, dtype=float)
    k = bonds.shape[0]
    hess = np.zeros((k, k))
    for col in range(k):
        shift = np.zeros(k)
        shift[col] = step
        hess[:, col] = (_bond_gradient(bonds + shift, types, params)
                        - _bond_gradient(bonds - shift, types, params)) / (2.0 * step)
    return 0.5 * (hess + hess.T)

import numpy as np

def displace_along_modes(eq_bonds: "np.ndarray", modes: "np.ndarray", omegas: "np.ndarray", eps: float, direction: "np.ndarray") -> "np.ndarray":
    b0 = np.asarray(eq_bonds, dtype=float)
    V = np.asarray(modes, dtype=float)
    w = np.asarray(omegas, dtype=float)
    u = np.asarray(direction, dtype=float)
    k = b0.shape[0]
    if V.shape != (k, k) or w.shape != (k,) or u.shape != (k,):
        raise ValueError("modes must be (K, K) and omegas, direction (K,)")
    if eps < 0.0:
        raise ValueError("eps must be non-negative")
    norm = np.linalg.norm(u)
    if norm == 0.0:
        raise ValueError("direction must be nonzero")
    radius = np.sqrt(2.0 * float(eps))
    q = (u / norm) * radius / w
    bonds = b0 + V.T @ q
    return np.concatenate([[0.0], np.cumsum(bonds)])

import numpy as np

def apply_workspace_program(program: "np.ndarray", S: "np.ndarray", H: "np.ndarray", gamma: "np.ndarray") -> "np.ndarray":
    S = np.asarray(S, dtype=float)
    H = np.asarray(H, dtype=float)
    gamma = np.asarray(gamma, dtype=float)
    n = H.shape[0]
    if S.shape != (n, n) or H.shape != (n, n) or gamma.shape != (n, n):
        raise ValueError("S, H and gamma must share one square shape")
    ops = np.asarray(program).reshape(-1)
    if ops.size and (not np.issubdtype(ops.dtype, np.integer) or ops.min() < 0 or ops.max() > 6):
        raise ValueError("program entries must be integers in 0..6")
    S_inv = np.linalg.inv(S)
    atomic_field = _build_fock(np.zeros((n, n)), gamma, np.eye(n))
    M = H.copy()
    for op in ops:
        if op == 1:
            M = M + H
        elif op == 2:
            M = M + S
        elif op == 3:
            M = M @ S_inv @ M
        elif op == 4:
            M = M + atomic_field
        elif op == 5:
            M = M * S
        elif op == 6:
            M = 0.5 * M
    return M

import numpy as np

def program_energy(program: "np.ndarray", positions: "np.ndarray", types: "np.ndarray", params: dict) -> float:
    pos = np.asarray(positions, dtype=float)
    n = pos.shape[0]
    if n % 2:
        raise ValueError("the model needs an even number of atoms")
    S, H, gamma = build_model_matrices(pos, types, params)
    M = apply_workspace_program(program, S, H, gamma)
    P = _occupied_density(M, S, n)
    return mean_field_energy(H, gamma, P, pos, params)

import numpy as np

def fit_shifted_loss(pred: list, ref: list, ref_shifts: "np.ndarray", counts: "np.ndarray") -> "np.ndarray":
    D = np.asarray(ref_shifts, dtype=float)
    N = np.asarray(counts, dtype=float)
    n_mol = D.shape[0]
    if len(pred) != n_mol or len(ref) != n_mol or N.ndim != 2 or N.shape[0] != n_mol:
        raise ValueError("pred, ref, ref_shifts and counts must describe the same molecules")
    rows, targets = [], []
    for i in range(n_mol):
        p = np.asarray(pred[i], dtype=float).reshape(-1)
        r = np.asarray(ref[i], dtype=float).reshape(-1)
        if p.shape != r.shape:
            raise ValueError("pred and ref of one molecule must have the same length")
        for pj, rj in zip(p, r):
            rows.append(N[i])
            targets.append(pj - (rj - D[i]))
    if not rows:
        raise ValueError("at least one configuration is required")
    A = np.array(rows)
    y = np.array(targets)
    if np.linalg.matrix_rank(A) < N.shape[1]:
        raise ValueError("counts must have full column rank to determine every shift")
    d, *_ = np.linalg.lstsq(A, y, rcond=None)
    residual = y - A @ d
    loss = float(np.sqrt(np.mean(residual ** 2)))
    return np.concatenate([[loss], d])

import numpy as np

def anneal_program(molecules: list, params: dict, n_ops: int, n_functions: int, n_iter: int, t_init: float, seed: int) -> "np.ndarray":
    n_ops, n_functions, n_iter = int(n_ops), int(n_functions), int(n_iter)
    if n_functions < 1:
        raise ValueError("n_functions must be at least 1")
    if n_ops < 1 or n_ops > 7:
        raise ValueError("n_ops must be in 1..7")
    rng = np.random.default_rng(int(seed))
    program = rng.integers(0, n_ops, size=n_functions)
    current = float(_training_loss(program, molecules, params)[0])
    best_loss, best_program = current, program.copy()
    for k in range(n_iter):
        temperature = float(t_init) * (1.0 - k / n_iter)
        r = min(int(rng.integers(1, 4)), n_functions)
        positions = rng.choice(n_functions, size=r, replace=False)
        candidate = program.copy()
        candidate[positions] = rng.integers(0, n_ops, size=r)
        cand_loss = float(_training_loss(candidate, molecules, params)[0])
        if cand_loss <= current or rng.random() < np.exp(-(cand_loss - current) / temperature):
            program, current = candidate, cand_loss
            if current < best_loss:
                best_loss, best_program = current, program.copy()
    return np.concatenate([[best_loss], best_program.astype(float)])

def _training_loss(program: "np.ndarray", molecules: list, params: dict) -> "np.ndarray":
    """Shifted loss and fitted shifts of one program over the training molecules."""
    n_types = len(params["onsite_energy"])
    pred, ref, shifts, counts = [], [], [], []
    for mol in molecules:
        types = np.asarray(mol["types"], dtype=int)
        pred.append(np.array([program_energy(program, pos, types, params) for pos in mol["positions"]]))
        ref.append(np.asarray(mol["ref_energies"], dtype=float))
        shifts.append(float(mol["ref_shift"]))
        counts.append(np.bincount(types, minlength=n_types))
    return fit_shifted_loss(pred, ref, np.array(shifts), np.array(counts))

import numpy as np

def pips_holdout_rmse(train_specs: list, holdout_spec: dict, params: dict, e_max: float, n_functions: int, n_iter: int, t_init: float, seed: int) -> float:
    alpha = np.asarray(params["onsite_energy"], dtype=float)
    n_types = alpha.shape[0]
    molecules = []
    for spec in train_specs:
        types = np.asarray(spec["types"], dtype=int)
        positions, energies, e_eq = _sampled_geometries(spec, params)
        keep = energies - e_eq <= e_max
        if not np.any(keep):
            continue
        molecules.append({"types": types, "positions": positions[keep], "ref_energies": energies[keep],
                          "ref_shift": float(np.sum(alpha[types]))})
    if not molecules:
        raise ValueError("no training geometry lies within e_max of equilibrium")
    result = anneal_program(molecules, params, 7, n_functions, n_iter, t_init, seed)
    program = result[1:].astype(int)
    shifts = _training_loss(program, molecules, params)[1:]
    types = np.asarray(holdout_spec["types"], dtype=int)
    positions, energies, _ = _sampled_geometries(holdout_spec, params)
    predicted = np.array([program_energy(program, pos, types, params) for pos in positions])
    d_bar = float(np.bincount(types, minlength=n_types) @ shifts)
    d_ref = float(np.sum(alpha[types]))
    residual = (predicted - d_bar) - (energies - d_ref)
    return float(np.sqrt(np.mean(residual ** 2)))

def _sampled_geometries(spec: dict, params: dict) -> tuple:
    """Sampled positions and their SCF energies, plus the equilibrium SCF energy."""
    types = np.asarray(spec["types"], dtype=int)
    bonds = equilibrium_bonds(types, spec["start_bonds"], params)
    modes = normal_modes(types, bonds, params)
    omegas, vectors = modes[0], modes[1:]
    positions, energies = [], []
    for eps, direction in spec["samples"]:
        pos = displace_along_modes(bonds, vectors, omegas, eps, direction)
        positions.append(pos)
        energies.append(scf_reference_energy(pos, types, params))
    e_eq = scf_reference_energy(_bonds_to_positions(bonds), types, params)
    return np.array(positions), np.array(energies), e_eq
SCICODE_GOLD_EOF
