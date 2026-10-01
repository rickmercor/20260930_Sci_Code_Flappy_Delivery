#!/bin/bash
# Oracle: writes the gold solution so the verifier scores reward 1.0.
cat > /app/solution.py <<'SCICODE_GOLD_EOF'
import numpy as np
try:
    import scipy, scipy.linalg, scipy.sparse, scipy.optimize, scipy.spatial, scipy.integrate, scipy.stats, scipy.special, scipy.signal
except Exception:
    pass
import math

def potential_jet(q, origin, powers, coefficients):
    import numpy as np
    q = np.asarray(q, dtype=float)
    origin = np.asarray(origin, dtype=float)
    raw = np.asarray(powers)
    coefficients = np.asarray(coefficients, dtype=float)
    if (q.ndim != 1 or q.size == 0 or origin.shape != q.shape
            or raw.ndim != 2 or raw.shape[1] != q.size
            or coefficients.shape != (raw.shape[0],)
            or not all(np.all(np.isfinite(a)) for a in (q, origin, raw, coefficients))
            or np.any(raw < 0) or np.any(raw != np.floor(raw))):
        raise ValueError("Invalid polynomial arrays")
    powers = raw.astype(int)
    x = q - origin
    d = q.size
    value = 0.0
    gradient = np.zeros(d)
    hessian = np.zeros((d, d))
    for alpha, weight in zip(powers, coefficients):
        value += weight * np.prod(x ** alpha)
        for i in range(d):
            if alpha[i] == 0:
                continue
            beta = alpha.copy()
            beta[i] -= 1
            gradient[i] += weight * alpha[i] * np.prod(x ** beta)
            for j in range(d):
                if beta[j] == 0:
                    continue
                gamma = beta.copy()
                gamma[j] -= 1
                hessian[i, j] += weight * alpha[i] * beta[j] * np.prod(x ** gamma)
    return float(value), gradient, hessian

def initial_frame(mass, ground_hessian, hbar):
    import numpy as np

    mass = np.asarray(mass, dtype=float)
    ground_hessian = np.asarray(ground_hessian, dtype=float)
    if (mass.ndim != 2 or mass.shape[0] == 0 or mass.shape[0] != mass.shape[1]
            or ground_hessian.shape != mass.shape
            or not all(np.all(np.isfinite(a)) for a in (mass, ground_hessian))
            or not np.allclose(mass, mass.T, atol=1e-12, rtol=0)
            or not np.allclose(ground_hessian, ground_hessian.T, atol=1e-12, rtol=0)
            or not np.isfinite(hbar) or hbar <= 0):
        raise ValueError("Invalid mass, Hessian, or hbar")

    values, vectors = np.linalg.eigh(mass)
    if np.min(values) <= 0:
        raise ValueError("Mass must be positive definite")
    root_mass = (vectors * values ** 0.5) @ vectors.T
    inverse_root_mass = (vectors * values ** -0.5) @ vectors.T

    dynamical = inverse_root_mass @ ground_hessian @ inverse_root_mass
    values, vectors = np.linalg.eigh(dynamical)
    if np.min(values) <= 0:
        raise ValueError("Hessian must be positive definite")
    frequencies = (vectors * values ** 0.5) @ vectors.T

    precision = root_mass @ frequencies @ root_mass
    values, vectors = np.linalg.eigh(precision)
    if np.min(values) <= 0:
        raise ValueError("Precision must be positive definite")
    Q0 = (vectors * values ** -0.5) @ vectors.T
    P0 = 1j * ((vectors * values ** 0.5) @ vectors.T)
    E0 = 0.5 * hbar * np.trace(frequencies)
    return Q0, P0, float(E0)

def centre_action(mass, q0, p0, origin, powers, coefficients, dt, nsteps):
    import numpy as np
    mass = np.asarray(mass, dtype=float)
    q = np.asarray(q0, dtype=float).copy()
    p = np.asarray(p0, dtype=float).copy()
    if (q.ndim != 1 or q.size == 0 or p.shape != q.shape
            or mass.shape != (q.size, q.size)
            or not all(np.all(np.isfinite(a)) for a in (mass, q, p))
            or not np.allclose(mass, mass.T, atol=1e-12, rtol=0)
            or np.min(np.linalg.eigvalsh(mass)) <= 0
            or not np.isfinite(dt) or dt == 0
            or not isinstance(nsteps, (int, np.integer)) or nsteps < 0):
        raise ValueError("Invalid trajectory inputs")
    potential_jet(q, origin, powers, coefficients)
    inverse_mass = np.linalg.inv(mass)
    qs = np.empty((nsteps + 1, q.size))
    ps = np.empty_like(qs)
    actions = np.empty(nsteps + 1)
    qs[0], ps[0], actions[0] = q, p, 0.0
    action = 0.0
    gamma = 1.0 / (2.0 - 2.0 ** (1.0 / 3.0))
    for step in range(nsteps):
        for scale in (gamma, 1.0 - 2.0 * gamma, gamma):
            h = dt * scale
            value, gradient, _ = potential_jet(q, origin, powers, coefficients)
            p -= 0.5 * h * gradient
            action -= 0.5 * h * value
            action += 0.5 * h * (p @ inverse_mass @ p)
            q += h * (inverse_mass @ p)
            value, gradient, _ = potential_jet(q, origin, powers, coefficients)
            p -= 0.5 * h * gradient
            action -= 0.5 * h * value
        qs[step + 1], ps[step + 1], actions[step + 1] = q, p, action
    return qs, ps, actions

def width_path(mass, reference_hessian, q0, p0, times):
    import numpy as np
    from scipy.linalg import expm
    mass = np.asarray(mass, dtype=float)
    reference_hessian = np.asarray(reference_hessian, dtype=float)
    q0 = np.asarray(q0, dtype=complex)
    p0 = np.asarray(p0, dtype=complex)
    times = np.asarray(times, dtype=float)
    if (mass.ndim != 2 or mass.shape[0] == 0 or mass.shape[0] != mass.shape[1]
            or any(a.shape != mass.shape for a in (reference_hessian, q0, p0))
            or times.ndim != 1 or times.size == 0 or times[0] != 0
            or (times.size > 1 and not (np.all(np.diff(times) > 0) or np.all(np.diff(times) < 0)))
            or not all(np.all(np.isfinite(a)) for a in (mass, reference_hessian, q0, p0, times))
            or not np.allclose(mass, mass.T, atol=1e-12, rtol=0)
            or not np.allclose(reference_hessian, reference_hessian.T, atol=1e-12, rtol=0)
            or np.min(np.linalg.eigvalsh(mass)) <= 0
            or not np.allclose(q0.imag, 0, atol=1e-12, rtol=0)
            or not np.allclose(q0, q0.T, atol=1e-12, rtol=0)
            or np.min(np.linalg.eigvalsh(q0.real)) <= 0):
        raise ValueError("Invalid width inputs")
    d = mass.shape[0]
    if (not np.allclose(q0.T @ p0 - p0.T @ q0, 0, atol=1e-9, rtol=0)
            or not np.allclose(q0.conj().T @ p0 - p0.conj().T @ q0,
                               2j * np.eye(d), atol=1e-9, rtol=0)):
        raise ValueError("Initial frame is not canonical")
    generator = np.block([[np.zeros((d, d)), np.linalg.inv(mass)],
                          [-reference_hessian, np.zeros((d, d))]])
    initial = np.vstack((q0, p0))
    frames = np.stack([expm(t * generator) @ initial for t in times])
    qs, ps = frames[:, :d], frames[:, d:]
    signs, logabs = np.linalg.slogdet(qs)
    phases = np.unwrap(np.angle(signs))
    phases -= phases[0]
    logs = logabs + 1j * phases
    return qs, ps, logs

def dipole_coefficients(q0, powers, coefficients, hbar):
    import numpy as np
    from itertools import product
    q0 = np.asarray(q0, dtype=float)
    raw = np.asarray(powers)
    coefficients = np.asarray(coefficients, dtype=complex)
    if (q0.ndim != 2 or q0.shape[0] == 0 or q0.shape[0] != q0.shape[1]
            or raw.ndim != 2 or raw.shape[0] == 0 or raw.shape[1] != q0.shape[0]
            or coefficients.shape != (raw.shape[0], 3)
            or not all(np.all(np.isfinite(a)) for a in (q0, raw, coefficients))
            or np.any(raw < 0) or np.any(raw != np.floor(raw))
            or not np.allclose(q0, q0.T, atol=1e-12, rtol=0)
            or np.min(np.linalg.eigvalsh(q0)) <= 0
            or not np.isfinite(hbar) or hbar <= 0):
        raise ValueError("Invalid dipole inputs")
    powers = raw.astype(int)
    degree = int(np.max(powers.sum(axis=1)))
    if degree > 6:
        raise ValueError("Dipole degree exceeds six")
    d = q0.shape[0]
    labels = np.array(sorted(
        (k for k in product(range(degree + 1), repeat=d) if sum(k) <= degree),
        key=lambda k: (sum(k), k),
    ), dtype=int)
    index = {tuple(k): i for i, k in enumerate(labels)}
    output = np.zeros((len(labels), 3), dtype=complex)
    scale = np.sqrt(hbar / 2.0) * q0
    vacuum = (0,) * d
    for alpha, polar_coefficients in zip(powers, coefficients):
        state = {vacuum: 1.0}
        for axis, exponent in enumerate(alpha):
            for _ in range(exponent):
                updated = {}
                for k, amplitude in state.items():
                    for mode in range(d):
                        up = list(k)
                        up[mode] += 1
                        up = tuple(up)
                        updated[up] = updated.get(up, 0.0) + amplitude * scale[axis, mode] * np.sqrt(k[mode] + 1)
                        if k[mode]:
                            down = list(k)
                            down[mode] -= 1
                            down = tuple(down)
                            updated[down] = updated.get(down, 0.0) + amplitude * scale[axis, mode] * np.sqrt(k[mode])
                state = updated
        for k, amplitude in state.items():
            output[index[k]] += amplitude * polar_coefficients
    return labels, output

def overlap_generator(q_bra, p_bra, Q_bra, P_bra, S_bra, logdet_bra,
                              q_ket, p_ket, Q_ket, P_ket, S_ket, logdet_ket, hbar):
    import numpy as np
    q_bra, p_bra, q_ket, p_ket = [np.asarray(a, dtype=float) for a in (q_bra, p_bra, q_ket, p_ket)]
    Q_bra, P_bra, Q_ket, P_ket = [np.asarray(a, dtype=complex) for a in (Q_bra, P_bra, Q_ket, P_ket)]
    d = q_bra.size
    if (q_bra.shape != (d,) or d == 0
            or any(a.shape != (d,) for a in (p_bra, q_ket, p_ket))
            or any(a.shape != (d, d) for a in (Q_bra, P_bra, Q_ket, P_ket))
            or not all(np.all(np.isfinite(a)) for a in
                       (q_bra, p_bra, q_ket, p_ket, Q_bra, P_bra, Q_ket, P_ket,
                        S_bra, S_ket, logdet_bra, logdet_ket, hbar))
            or hbar <= 0):
        raise ValueError("Invalid Gaussian inputs")
    for Q, P, logdet in ((Q_bra, P_bra, logdet_bra), (Q_ket, P_ket, logdet_ket)):
        if (not np.allclose(Q.T @ P - P.T @ Q, 0, atol=1e-8, rtol=0)
                or not np.allclose(Q.conj().T @ P - P.conj().T @ Q,
                                   2j * np.eye(d), atol=1e-8, rtol=0)
                or abs(np.exp(logdet) - np.linalg.det(Q)) > 1e-8 * max(1.0, abs(np.linalg.det(Q)))):
            raise ValueError("Noncanonical frame or inconsistent determinant lift")
    inverse_bra = np.linalg.inv(Q_bra).conj()
    inverse_ket = np.linalg.inv(Q_ket)
    A_bra = (P_bra @ np.linalg.inv(Q_bra)).conj()
    A_ket = P_ket @ inverse_ket
    precision = -1j * (A_ket - A_bra) / hbar
    linear = 1j * (-A_ket @ q_ket + A_bra @ q_bra + p_ket - p_bra) / hbar
    constant = 1j * (0.5 * q_ket @ A_ket @ q_ket - 0.5 * q_bra @ A_bra @ q_bra
                     - p_ket @ q_ket + p_bra @ q_bra + S_ket - S_bra) / hbar
    rows = np.sqrt(2.0 / hbar) * np.vstack((inverse_bra, inverse_ket))
    shift = -np.concatenate((rows[:d] @ q_bra, rows[d:] @ q_ket))
    B = rows @ np.linalg.solve(precision, rows.T)
    B[:d, :d] -= inverse_bra @ Q_bra
    B[d:, d:] -= inverse_ket @ Q_ket.conj()
    b = shift + rows @ np.linalg.solve(precision, linear)
    # Each eigenvalue of this accretive matrix has positive real part.
    # Summing their principal logarithms keeps the matrix square-root branch.
    logdet_precision = np.sum(np.log(np.linalg.eigvals(precision)))
    log_prefactor = (0.5 * d * np.log(2.0 / hbar)
                     - 0.5 * (np.conj(logdet_bra) + logdet_ket + logdet_precision)
                     + constant + 0.5 * linear @ np.linalg.solve(precision, linear))
    overlap = complex(np.exp(log_prefactor))
    return 0.5 * (B + B.T), b, overlap

def fock_overlap(B, b, vacuum_overlap, bra_labels, ket_labels):
    import math
    import numpy as np
    B = np.asarray(B, dtype=complex)
    b = np.asarray(b, dtype=complex)
    left = np.asarray(bra_labels)
    right = np.asarray(ket_labels)
    if (B.ndim != 2 or B.shape[0] == 0 or B.shape[0] != B.shape[1] or B.shape[0] % 2
            or b.shape != (B.shape[0],)
            or left.ndim != 2 or right.ndim != 2
            or left.shape[1] != B.shape[0] // 2 or right.shape[1] != left.shape[1]
            or not all(np.all(np.isfinite(a)) for a in (B, b, vacuum_overlap, left, right))
            or not np.allclose(B, B.T, atol=1e-10, rtol=0)
            or np.any(left < 0) or np.any(right < 0)
            or np.any(left != np.floor(left)) or np.any(right != np.floor(right))
            or (left.size and np.max(left.sum(axis=1)) > 8)
            or (right.size and np.max(right.sum(axis=1)) > 8)):
        raise ValueError("Invalid generating function or occupation labels")
    left, right = left.astype(int), right.astype(int)

    zero = (0,) * len(b)
    required = {zero}
    stack = [tuple(k) + tuple(ell) for k in left for ell in right]
    while stack:
        alpha = stack.pop()
        if alpha in required:
            continue
        required.add(alpha)
        i = next(j for j, count in enumerate(alpha) if count)
        beta = list(alpha)
        beta[i] -= 1
        stack.append(tuple(beta))
        for j, count in enumerate(beta):
            if count:
                gamma = beta.copy()
                gamma[j] -= 1
                stack.append(tuple(gamma))
    moments = {zero: 1.0 + 0j}
    for alpha in sorted(required, key=lambda k: (sum(k), k)):
        if alpha == zero:
            continue
        i = next(j for j, count in enumerate(alpha) if count)
        beta = list(alpha)
        beta[i] -= 1
        value = b[i] * moments[tuple(beta)]
        for j, count in enumerate(beta):
            if count:
                gamma = beta.copy()
                gamma[j] -= 1
                value += B[i, j] * count * moments[tuple(gamma)]
        moments[alpha] = value
    output = np.empty((len(left), len(right)), dtype=complex)
    for row, k in enumerate(left):
        for column, ell in enumerate(right):
            alpha = tuple(k) + tuple(ell)
            denominator = math.sqrt(math.prod(math.factorial(int(n)) for n in alpha))
            output[row, column] = vacuum_overlap * moments[alpha] / denominator
    return output

def correlation_path(qs, ps, actions, Qs, Ps, logdets, labels, coefficients, hbar):
    import numpy as np
    qs, ps, actions = [np.asarray(a, dtype=float) for a in (qs, ps, actions)]
    Qs, Ps, logdets, coefficients = [np.asarray(a, dtype=complex) for a in (Qs, Ps, logdets, coefficients)]
    labels = np.asarray(labels)
    if (qs.ndim != 2 or qs.shape[0] == 0 or qs.shape[1] == 0 or ps.shape != qs.shape
            or actions.shape != (qs.shape[0],) or logdets.shape != actions.shape
            or Qs.shape != (qs.shape[0], qs.shape[1], qs.shape[1]) or Ps.shape != Qs.shape
            or labels.ndim != 2 or labels.shape[1] != qs.shape[1]
            or coefficients.shape != (labels.shape[0], 3)
            or not all(np.all(np.isfinite(a)) for a in (qs, ps, actions, Qs, Ps, logdets, labels, coefficients))
            or not np.isfinite(hbar) or hbar <= 0):
        raise ValueError("Invalid correlation inputs")
    answer = np.empty(len(qs), dtype=complex)
    initial = (qs[0], ps[0], Qs[0], Ps[0], actions[0], logdets[0])
    for t in range(len(qs)):
        current = (qs[t], ps[t], Qs[t], Ps[t], actions[t], logdets[t])
        B, b, overlap = overlap_generator(*initial, *current, hbar)
        matrix = fock_overlap(B, b, overlap, labels, labels)
        answer[t] = np.einsum("ia,ij,ja->", coefficients.conj(), matrix, coefficients) / 3.0
    return answer

def solve(data):
    import numpy as np

    required = {
        "mass",
        "ground_hessian",
        "q_initial",
        "origin",
        "potential_powers",
        "potential_coefficients",
        "q_reference",
        "dipole_powers",
        "dipole_coefficients",
        "hbar",
        "dt",
        "nsteps",
        "omega",
        "eta",
    }

    if not isinstance(data, dict) or not required.issubset(data):
        raise ValueError("Missing benchmark fields")

    hbar = data["hbar"]
    dt = data["dt"]
    nsteps = data["nsteps"]
    omega = data["omega"]
    eta = data["eta"]

    if (
        not all(np.isfinite(v) for v in (hbar, dt, omega, eta))
        or hbar <= 0
        or dt <= 0
        or eta < 0
        or not isinstance(nsteps, (int, np.integer))
        or nsteps < 2
        or nsteps % 2 != 0
    ):
        raise ValueError("Invalid spectral grid")

    Q0, P0, energy = initial_frame(
        data["mass"],
        data["ground_hessian"],
        hbar,
    )

    _, _, reference_hessian = potential_jet(
        data["q_reference"],
        data["origin"],
        data["potential_powers"],
        data["potential_coefficients"],
    )

    qs, ps, actions = centre_action(
        data["mass"],
        data["q_initial"],
        np.zeros(Q0.shape[0]),
        data["origin"],
        data["potential_powers"],
        data["potential_coefficients"],
        dt,
        nsteps,
    )

    times = dt * np.arange(nsteps + 1)

    Qs, Ps, logdets = width_path(
        data["mass"],
        reference_hessian,
        Q0,
        P0,
        times,
    )

    labels, coefficients = dipole_coefficients(
        Q0,
        data["dipole_powers"],
        data["dipole_coefficients"],
        hbar,
    )

    correlation = correlation_path(
        qs,
        ps,
        actions,
        Qs,
        Ps,
        logdets,
        labels,
        coefficients,
        hbar,
    )

    norm = float(correlation[0].real)
    if norm <= 0:
        raise ValueError("The initial dipole state has zero norm")

    integrand = correlation * np.exp(
        1j * (omega + energy / hbar) * times
        - eta * times ** 2
    )

    weights = np.ones(nsteps + 1)
    weights[1:-1:2] = 4.0
    weights[2:-1:2] = 2.0

    return float(
        (dt / 3.0)
        * np.real(weights @ integrand)
        / (np.pi * norm)
    )
SCICODE_GOLD_EOF
