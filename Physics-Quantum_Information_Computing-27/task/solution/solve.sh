#!/bin/bash
# Oracle: writes the gold solution so the verifier scores reward 1.0.
cat > /app/solution.py <<'SCICODE_GOLD_EOF'
import numpy as np
try:
    import scipy, scipy.linalg, scipy.sparse, scipy.optimize, scipy.spatial, scipy.integrate, scipy.stats, scipy.special, scipy.signal
except Exception:
    pass
import math

def build_ising_gibbs(couplings: "np.ndarray", fields: "np.ndarray", beta: float) -> "np.ndarray":
    import numpy as np
    from numbers import Real

    j = np.asarray(couplings)
    h = np.asarray(fields)
    if (j.ndim != 1 or h.shape != j.shape or not 2 <= j.size <= 8
            or np.iscomplexobj(j) or np.iscomplexobj(h)):
        raise ValueError("Use equally sized real vectors with 2 through 8 entries.")
    j, h = j.astype(float), h.astype(float)
    if (not np.all(np.isfinite(j)) or not np.all(np.isfinite(h))
            or not isinstance(beta, Real) or not np.isfinite(beta) or beta < 0):
        raise ValueError("Couplings and fields must be finite; beta must be finite and nonnegative.")
    n = j.size
    d = 1 << n
    if beta == 0:
        return np.eye(d, dtype=complex) / d
    basis = np.arange(d)
    spins = 1 - 2 * ((basis[:, None] >> np.arange(n - 1, -1, -1)) & 1)
    diagonal = -np.sum(j * spins * np.roll(spins, -1, axis=1), axis=1)
    hamiltonian = np.diag(diagonal)
    for site in range(n):
        hamiltonian[basis, basis ^ (1 << (n - 1 - site))] -= h[site]
    energies, vectors = np.linalg.eigh(hamiltonian)
    with np.errstate(over="ignore"):
        weights = np.exp(-float(beta) * (energies - energies[0]))
    weights /= weights.sum()
    return np.asarray((vectors * weights) @ vectors.T, dtype=complex)

def contract_triangle_state(links: "np.ndarray", unitaries: "np.ndarray") -> "np.ndarray":
    import numpy as np

    s = np.asarray(links, dtype=complex)
    u = np.asarray(unitaries, dtype=complex)
    if (s.shape != (3, 4) or u.shape != (3, 4, 4)
            or not np.all(np.isfinite(s)) or not np.all(np.isfinite(u))):
        raise ValueError("Use finite link vectors (3,4) and local matrices (3,4,4).")
    if (not np.allclose(np.sum(abs(s)**2, axis=1), 1.0, atol=1e-10, rtol=0)
            or not np.allclose(u.conj().transpose(0, 2, 1) @ u, np.eye(4), atol=1e-10, rtol=0)):
        raise ValueError("Links must be normalized and local matrices unitary.")
    tensor = np.einsum("ab,cd,ef->afbcde", s[0].reshape(2, 2),
                       s[1].reshape(2, 2), s[2].reshape(2, 2)).reshape(4, 4, 4)
    return np.einsum("ia,jb,kc,abc->ijk", u[0], u[1], u[2], tensor).reshape(64)

def optimize_resource_link(residual: "np.ndarray", links: "np.ndarray", unitaries: "np.ndarray", edge: int) -> "np.ndarray":
    import numpy as np
    from numbers import Integral

    r = np.asarray(residual, dtype=complex)
    if (r.shape != (64, 64) or not np.all(np.isfinite(r))
            or not np.allclose(r, r.conj().T, atol=1e-10, rtol=0)):
        raise ValueError("Residual must be a finite Hermitian (64,64) matrix.")
    if not isinstance(edge, Integral) or isinstance(edge, (bool, np.bool_)) or not 0 <= edge < 3:
        raise ValueError("Edge must be an integer in {0,1,2}.")
    contract_triangle_state(links, unitaries)
    s = np.asarray(links, dtype=complex)
    embedding = np.empty((64, 4), dtype=complex)
    for component in range(4):
        trial = s.copy()
        trial[edge] = np.eye(4)[component]
        embedding[:, component] = contract_triangle_state(trial, unitaries)
    effective = embedding.conj().T @ r @ embedding
    effective = (effective + effective.conj().T) / 2
    values, vectors = np.linalg.eigh(effective)
    cluster = values >= values[-1] - 1e-12 * max(1.0, float(np.max(abs(values))))
    space = vectors[:, cluster]
    updated = space @ (space.conj().T @ s[edge])
    norm = np.linalg.norm(updated)
    if norm <= 1e-12:
        for component in range(4):
            updated = space @ space[component].conj()
            norm = np.linalg.norm(updated)
            if norm > 1e-12:
                break
    updated /= norm
    first = int(np.flatnonzero(abs(updated) > 1e-12)[0])
    updated *= np.exp(-1j * np.angle(updated[first]))
    return updated

def optimize_local_rotation(residual: "np.ndarray", links: "np.ndarray", unitaries: "np.ndarray", party: int, generator: "np.ndarray") -> "np.ndarray":
    import numpy as np
    from numbers import Integral

    r = np.asarray(residual, dtype=complex)
    g = np.asarray(generator, dtype=complex)
    if (r.shape != (64, 64) or not np.all(np.isfinite(r))
            or not np.allclose(r, r.conj().T, atol=1e-10, rtol=0)):
        raise ValueError("Residual must be a finite Hermitian (64,64) matrix.")
    if (g.shape != (4, 4) or not np.all(np.isfinite(g))
            or not np.allclose(g, g.conj().T, atol=1e-10, rtol=0)
            or not np.allclose(g @ g, np.eye(4), atol=1e-10, rtol=0)):
        raise ValueError("Generator must be a finite Hermitian involution of shape (4,4).")
    if not isinstance(party, Integral) or isinstance(party, (bool, np.bool_)) or not 0 <= party < 3:
        raise ValueError("Party must be an integer in {0,1,2}.")
    psi = contract_triangle_state(links, unitaries)
    u = np.asarray(unitaries, dtype=complex)
    trial = u.copy()
    trial[party] = g @ u[party]
    phi = contract_triangle_state(links, trial)
    a = float(np.vdot(psi, r @ psi).real)
    b = float(np.vdot(phi, r @ phi).real)
    coherence = np.vdot(psi, r @ phi)
    cosine_coefficient = (a - b) / 2
    sine_coefficient = float(coherence.imag)
    if np.hypot(cosine_coefficient, sine_coefficient) <= 1e-14 * max(1.0, abs(a), abs(b)):
        theta = 0.0
    else:
        theta = float(0.5 * np.arctan2(sine_coefficient, cosine_coefficient))
        if theta >= np.pi / 2:
            theta -= np.pi
    return (np.cos(theta) * np.eye(4) - 1j * np.sin(theta) * g) @ u[party]

def project_network_memory(target: "np.ndarray", anchor: "np.ndarray", memory: "np.ndarray") -> "np.ndarray":
    import numpy as np
    from itertools import combinations

    rho = np.asarray(target, dtype=complex)
    current = np.asarray(anchor, dtype=complex)
    atoms = np.asarray(memory, dtype=complex)
    if (rho.ndim != 2 or rho.shape[0] != rho.shape[1] or rho.shape[0] < 2
            or current.shape != rho.shape or atoms.ndim != 3
            or atoms.shape[1:] != rho.shape or len(atoms) > 5):
        raise ValueError("Use same-sized square states and zero through five memory states.")
    vertices = np.concatenate((current[None], atoms), axis=0)
    for matrix in [rho, *vertices]:
        if (not np.all(np.isfinite(matrix))
                or not np.allclose(matrix, matrix.conj().T, atol=1e-10, rtol=0)
                or not np.isclose(np.trace(matrix), 1, atol=1e-10, rtol=0)
                or np.linalg.eigvalsh(matrix)[0] < -1e-10):
            raise ValueError("Inputs must be density matrices within absolute tolerance 1e-10.")
    if len(atoms) == 0:
        return current.copy()
    flat = vertices.reshape(len(vertices), -1)
    design = np.concatenate((flat.real, flat.imag), axis=1).T
    goal = np.concatenate((rho.ravel().real, rho.ravel().imag))
    best_error = np.inf
    best_state = current.copy()
    for count in range(1, len(vertices) + 1):
        for face in combinations(range(len(vertices)), count):
            base = design[:, face[0]]
            if count == 1:
                weights = np.ones(1)
            else:
                differences = design[:, face[1:]] - base[:, None]
                coefficients = np.linalg.lstsq(differences, goal - base, rcond=1e-13)[0]
                weights = np.r_[1 - coefficients.sum(), coefficients]
            if np.min(weights) < -1e-11:
                continue
            weights = np.maximum(weights, 0)
            weights /= weights.sum()
            displacement = design[:, face] @ weights - goal
            error = float(np.dot(displacement, displacement))
            if error < best_error:
                best_error = error
                best_state = np.tensordot(weights, vertices[list(face)], axes=1)
    return best_state

def admit_network_atom(target: "np.ndarray", anchor: "np.ndarray", memory: "np.ndarray", candidate: "np.ndarray", capacity: int) -> "np.ndarray":
    import numpy as np
    from numbers import Integral

    rho, current, atom = (np.asarray(v, dtype=complex) for v in (target, anchor, candidate))
    bank = np.asarray(memory, dtype=complex)
    if not isinstance(capacity, Integral) or isinstance(capacity, (bool, np.bool_)) or not 1 <= capacity <= 5:
        raise ValueError("Capacity must be an integer from 1 through 5.")
    if (rho.ndim != 2 or rho.shape[0] != rho.shape[1] or rho.shape[0] < 2
            or current.shape != rho.shape or atom.shape != rho.shape
            or bank.ndim != 3 or bank.shape[1:] != rho.shape or len(bank) > capacity):
        raise ValueError("Density-matrix and memory shapes must agree.")
    for matrix in [rho, current, atom, *bank]:
        if (not np.all(np.isfinite(matrix))
                or not np.allclose(matrix, matrix.conj().T, atol=1e-10, rtol=0)
                or not np.isclose(np.trace(matrix), 1, atol=1e-10, rtol=0)
                or np.linalg.eigvalsh(matrix)[0] < -1e-10):
            raise ValueError("Inputs must be density matrices within absolute tolerance 1e-10.")
    score = float(np.vdot(rho - current, atom - current).real)
    if score <= 0:
        return bank.copy()
    return np.concatenate((bank, atom[None]), axis=0)[-capacity:].copy()

def certify_network_noise(target: "np.ndarray", approximant: "np.ndarray") -> float:
    import numpy as np

    rho = np.asarray(target, dtype=complex)
    sigma = np.asarray(approximant, dtype=complex)
    if rho.ndim != 2 or rho.shape[0] != rho.shape[1] or rho.shape[0] < 2 or sigma.shape != rho.shape:
        raise ValueError("Use square same-sized density matrices with dimension at least two.")
    for matrix in (rho, sigma):
        if (not np.all(np.isfinite(matrix))
                or not np.allclose(matrix, matrix.conj().T, atol=1e-10, rtol=0)
                or not np.isclose(np.trace(matrix), 1, atol=1e-10, rtol=0)
                or np.linalg.eigvalsh(matrix)[0] < -1e-10):
            raise ValueError("Inputs must be density matrices within absolute tolerance 1e-10.")
    delta = float(np.linalg.norm(rho - sigma))
    dimension = rho.shape[0]
    radius = 1.0 / np.sqrt(float(dimension) * (dimension - 1))
    return float(delta / (delta + radius))

def network_noise_certificate(couplings: "np.ndarray", fields: "np.ndarray", beta: float, seed_ids: "np.ndarray", sweep_counts: "np.ndarray", capacity: int) -> float:
    import numpy as np
    from numbers import Integral

    j, h = np.asarray(couplings), np.asarray(fields)
    ids, sweeps = np.asarray(seed_ids), np.asarray(sweep_counts)
    if j.shape != (6,) or h.shape != (6,):
        raise ValueError("The triangle benchmark requires six couplings and six fields.")
    if (ids.ndim != 1 or sweeps.shape != ids.shape or not np.issubdtype(ids.dtype, np.integer)
            or not np.issubdtype(sweeps.dtype, np.integer) or np.any(ids < 0) or np.any(sweeps < 0)
            or np.any(ids > 1000) or np.any(sweeps > 20)):
        raise ValueError("Use integer seed IDs in [0,1000] and matching sweep counts in [0,20].")
    if not isinstance(capacity, Integral) or isinstance(capacity, (bool, np.bool_)) or not 1 <= capacity <= 5:
        raise ValueError("Capacity must be an integer from 1 through 5.")
    rho = build_ising_gibbs(j, h, beta)
    current = np.eye(64, dtype=complex) / 64
    memory = np.empty((0, 64, 64), dtype=complex)
    x = np.array([[0, 1], [1, 0]], dtype=complex)
    y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    z = np.diag([1.0, -1.0])
    p = np.kron(x, y)
    q = np.kron(z, np.eye(2))
    for k, count in zip(ids, sweeps):
        residual = rho - current
        links = np.array([[1, (k + 1 + e) / 10 + 1j * (e + 1) / 7,
                           (2 * e - k) / 9 - 1j / 5, 0.8 - 1j * (k + e) / 13]
                          for e in range(3)], dtype=complex)
        links /= np.linalg.norm(links, axis=1)[:, None]
        unitaries = np.empty((3, 4, 4), dtype=complex)
        for party in range(3):
            alpha = (k + party + 1) * np.pi / 13
            eta = (2 * k - party + 2) * np.pi / 17
            unitaries[party] = ((np.cos(alpha) * np.eye(4) - 1j * np.sin(alpha) * p)
                                @ (np.cos(eta) * np.eye(4) - 1j * np.sin(eta) * q))
        for _ in range(int(count)):
            for edge in range(3):
                links[edge] = optimize_resource_link(residual, links, unitaries, edge)
            for party in range(3):
                for generator in (p, q):
                    unitaries[party] = optimize_local_rotation(
                        residual, links, unitaries, party, generator)
        psi = contract_triangle_state(links, unitaries)
        candidate = np.outer(psi, psi.conj())
        memory = admit_network_atom(rho, current, memory, candidate, capacity)
        current = project_network_memory(rho, current, memory)
    return certify_network_noise(rho, current)
SCICODE_GOLD_EOF
