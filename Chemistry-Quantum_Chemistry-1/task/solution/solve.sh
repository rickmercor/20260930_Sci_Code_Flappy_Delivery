#!/bin/bash
# Oracle: writes the gold solution so the verifier scores reward 1.0.
cat > /app/solution.py <<'SCICODE_GOLD_EOF'
import numpy as np
try:
    import scipy, scipy.linalg, scipy.sparse, scipy.optimize, scipy.spatial, scipy.integrate, scipy.stats, scipy.special, scipy.signal
except Exception:
    pass
import math

def reconstruct_ithc_eri(
    isometry: 'np.ndarray',
    kernel: 'np.ndarray',
) -> 'np.ndarray':
    import numpy as np

    u = np.asarray(isometry)
    W = np.asarray(kernel)
    if np.iscomplexobj(u) and np.any(np.abs(np.imag(u)) > 1e-13):
        raise ValueError("isometry must be real")
    u = np.asarray(np.real(u), dtype=float)
    W = np.asarray(W, dtype=float)
    if u.ndim != 2 or min(u.shape) == 0 or u.shape[0] > u.shape[1]:
        raise ValueError("isometry must have shape (n, n_aux) with n_aux >= n")
    if W.shape != (u.shape[1], u.shape[1]):
        raise ValueError("kernel has incompatible shape")
    if np.any(~np.isfinite(u)) or np.any(~np.isfinite(W)):
        raise ValueError("inputs must be finite")
    if not np.allclose(u @ u.T, np.eye(u.shape[0]), rtol=1e-11, atol=1e-11):
        raise ValueError("isometry rows must be orthonormal")
    if not np.allclose(W, W.T, rtol=1e-12, atol=1e-12):
        raise ValueError("kernel must be symmetric")
    if np.linalg.eigvalsh(W)[0] < -1e-11:
        raise ValueError("kernel must be positive semidefinite")
    return np.einsum(
        "pa,qa,ab,rb,sb->pqrs", u, u, W, u, u, optimize=True
    )

def extended_mixed_green(
    trial: 'np.ndarray',
    walker: 'np.ndarray',
    isometry: 'np.ndarray',
) -> 'np.ndarray':
    import numpy as np

    A = np.asarray(trial, dtype=complex)
    B = np.asarray(walker, dtype=complex)
    u0 = np.asarray(isometry)
    if np.iscomplexobj(u0) and np.any(np.abs(np.imag(u0)) > 1e-13):
        raise ValueError("isometry must be real")
    u = np.asarray(np.real(u0), dtype=float)
    if A.ndim != 2 or B.shape != A.shape or min(A.shape) == 0:
        raise ValueError("trial and walker must be matching nonempty matrices")
    if u.ndim != 2 or u.shape[0] != A.shape[0]:
        raise ValueError("isometry has incompatible shape")
    if np.any(~np.isfinite(A)) or np.any(~np.isfinite(B)) or np.any(~np.isfinite(u)):
        raise ValueError("inputs must be finite")
    if not np.allclose(u @ u.T, np.eye(u.shape[0]), rtol=1e-11, atol=1e-11):
        raise ValueError("isometry rows must be orthonormal")

    At = u.T @ A
    Bt = u.T @ B
    overlap = Bt.T @ At.conj()
    if np.linalg.cond(overlap) > 1e12:
        raise ValueError("trial-walker overlap is singular")
    return At.conj() @ np.linalg.solve(overlap, Bt.T)

def ithc_force_bias(
    extended_green: 'np.ndarray',
    channels: 'np.ndarray',
    time_step: float,
) -> 'np.ndarray':
    import numpy as np

    G = np.asarray(extended_green, dtype=complex)
    w = np.asarray(channels, dtype=float)
    dt = float(time_step)
    if G.ndim != 2 or G.shape[0] == 0 or G.shape[0] != G.shape[1]:
        raise ValueError("extended_green must be a nonempty square matrix")
    if w.ndim != 2 or w.shape[1] != G.shape[0] or w.shape[0] == 0:
        raise ValueError("channels have incompatible shape")
    if not np.isfinite(dt) or dt < 0:
        raise ValueError("time_step must be finite and nonnegative")
    if np.any(~np.isfinite(G)) or np.any(~np.isfinite(w)):
        raise ValueError("inputs must be finite")
    auxiliary_operators = 1.0j * w
    return -np.sqrt(dt) * (auxiliary_operators @ np.diag(G))

def propagate_ithc_walker(
    half_propagated_walker: 'np.ndarray',
    half_one_body: 'np.ndarray',
    isometry: 'np.ndarray',
    channels: 'np.ndarray',
    auxiliary_field: 'np.ndarray',
    force_bias: 'np.ndarray',
    time_step: float,
) -> 'np.ndarray':
    import numpy as np

    B = np.asarray(half_propagated_walker, dtype=complex)
    H = np.asarray(half_one_body, dtype=complex)
    u = np.asarray(isometry, dtype=float)
    w = np.asarray(channels, dtype=float)
    x = np.asarray(auxiliary_field, dtype=float)
    xb = np.asarray(force_bias, dtype=complex)
    dt = float(time_step)
    if B.ndim != 2 or min(B.shape) == 0:
        raise ValueError("walker must be a nonempty matrix")
    if H.shape != (B.shape[0], B.shape[0]):
        raise ValueError("half_one_body has incompatible shape")
    if u.ndim != 2 or u.shape[0] != B.shape[0]:
        raise ValueError("isometry has incompatible shape")
    if w.ndim != 2 or w.shape[1] != u.shape[1]:
        raise ValueError("channels have incompatible shape")
    if x.shape != (w.shape[0],) or xb.shape != x.shape:
        raise ValueError("field and force bias have incompatible shape")
    if not np.isfinite(dt) or dt < 0:
        raise ValueError("time_step must be finite and nonnegative")
    if any(np.any(~np.isfinite(v)) for v in (B, H, u, w, x, xb)):
        raise ValueError("inputs must be finite")
    if not np.allclose(u @ u.T, np.eye(u.shape[0]), rtol=1e-11, atol=1e-11):
        raise ValueError("isometry rows must be orthonormal")

    extended = u.T @ B
    exponent = np.sqrt(complex(-dt)) * ((x - xb) @ w)
    with np.errstate(over="ignore", invalid="ignore"):
        extended = np.exp(exponent)[:, None] * extended
        propagated = H @ (u @ extended)
    if np.any(~np.isfinite(propagated)):
        raise ValueError("propagation exceeds the floating-point range")
    return propagated

def phaseless_importance(
    trial: 'np.ndarray',
    old_walker: 'np.ndarray',
    new_walker: 'np.ndarray',
    auxiliary_field: 'np.ndarray',
    force_bias: 'np.ndarray',
) -> float:
    import numpy as np

    A = np.asarray(trial, dtype=complex)
    B0 = np.asarray(old_walker, dtype=complex)
    B1 = np.asarray(new_walker, dtype=complex)
    x = np.asarray(auxiliary_field, dtype=float)
    xb = np.asarray(force_bias, dtype=complex)
    if (
        A.ndim != 2
        or min(A.shape) == 0
        or B0.shape != A.shape
        or B1.shape != A.shape
    ):
        raise ValueError("trial and walkers must be matching nonempty matrices")
    if x.ndim != 1 or xb.shape != x.shape or x.size == 0:
        raise ValueError("field and force bias must be matching vectors")
    if any(np.any(~np.isfinite(v)) for v in (A, B0, B1, x, xb)):
        raise ValueError("inputs must be finite")

    old_phase, old_logabs = np.linalg.slogdet(A.conj().T @ B0)
    if old_phase == 0:
        raise ValueError("old trial-walker overlap is zero")

    new_phase, new_logabs = np.linalg.slogdet(A.conj().T @ B1)
    with np.errstate(over="ignore", invalid="ignore"):
        force_factor = np.dot(x, xb) - 0.5 * np.dot(xb, xb)
    if not np.isfinite(force_factor):
        raise ValueError("importance magnitude exceeds the floating-point range")
    log_magnitude = new_logabs - old_logabs + float(np.real(force_factor))
    if new_phase == 0:
        magnitude = 0.0
        ratio_phase = 0.0j
    else:
        if not np.isfinite(log_magnitude) or log_magnitude > np.log(np.finfo(float).max):
            raise ValueError("importance magnitude exceeds the floating-point range")
        if log_magnitude < np.log(np.nextafter(0.0, 1.0)):
            magnitude = 0.0
        else:
            magnitude = float(np.exp(log_magnitude))
        ratio_phase = new_phase / old_phase
    cosine_gate = max(0.0, float(np.cos(np.angle(ratio_phase))))
    return float(magnitude * cosine_gate)

def ithc_local_energy(
    one_body: 'np.ndarray',
    kernel: 'np.ndarray',
    physical_green: 'np.ndarray',
    extended_green: 'np.ndarray',
) -> complex:
    import numpy as np

    h = np.asarray(one_body, dtype=complex)
    W = np.asarray(kernel, dtype=float)
    G = np.asarray(physical_green, dtype=complex)
    Gt = np.asarray(extended_green, dtype=complex)
    if h.ndim != 2 or h.shape[0] == 0 or h.shape[0] != h.shape[1]:
        raise ValueError("one_body must be a nonempty square matrix")
    if G.shape != h.shape:
        raise ValueError("physical_green has incompatible shape")
    if W.ndim != 2 or W.shape[0] == 0 or W.shape[0] != W.shape[1]:
        raise ValueError("kernel must be a nonempty square matrix")
    if Gt.shape != W.shape:
        raise ValueError("extended_green has incompatible shape")
    if any(np.any(~np.isfinite(v)) for v in (h, W, G, Gt)):
        raise ValueError("inputs must be finite")
    if not np.allclose(h, h.conj().T, rtol=1e-12, atol=1e-12):
        raise ValueError("one_body must be Hermitian")
    if not np.allclose(W, W.T, rtol=1e-12, atol=1e-12):
        raise ValueError("kernel must be symmetric")

    one_energy = np.einsum("pq,pq->", h, G)
    occupations = np.diag(Gt)
    pair_density = np.outer(occupations, occupations) - Gt * Gt.T
    off_diagonal = W.copy()
    np.fill_diagonal(off_diagonal, 0.0)
    two_energy = 0.5 * np.sum(off_diagonal * pair_density)
    return complex(one_energy + two_energy)

def phaseless_mixed_energy(
    local_energies: 'np.ndarray',
    walker_weights: 'np.ndarray',
) -> float:
    import numpy as np

    energies = np.asarray(local_energies, dtype=complex)
    weights = np.asarray(walker_weights, dtype=float)
    if (
        energies.ndim != 1
        or energies.size == 0
        or weights.shape != energies.shape
    ):
        raise ValueError("energies and weights must be matching nonempty vectors")
    if np.any(~np.isfinite(energies)) or np.any(~np.isfinite(weights)):
        raise ValueError("inputs must be finite")
    if np.any(weights < 0) or not np.any(weights > 0):
        raise ValueError("weights must be nonnegative with positive sum")

    scaled_weights = weights / np.max(weights)
    normalized = scaled_weights / np.sum(scaled_weights)
    return float(np.dot(normalized, np.real(energies)))

def ithc_afqmc_projected_energy(
    one_body: 'np.ndarray',
    half_one_body: 'np.ndarray',
    isometry: 'np.ndarray',
    kernel: 'np.ndarray',
    channels: 'np.ndarray',
    eri_reference: 'np.ndarray',
    trial: 'np.ndarray',
    walkers: 'np.ndarray',
    walker_weights: 'np.ndarray',
    auxiliary_fields: 'np.ndarray',
    time_step: float,
    eri_tolerance: float,
) -> float:
    import numpy as np

    supplied = (
        one_body,
        half_one_body,
        isometry,
        kernel,
        channels,
        eri_reference,
        trial,
        walkers,
        walker_weights,
        auxiliary_fields,
        time_step,
        eri_tolerance,
    )
    if any(value is None for value in supplied):
        raise ValueError("all inputs are required")

    h = np.asarray(one_body, dtype=complex)
    half = np.asarray(half_one_body, dtype=complex)
    u = np.asarray(isometry, dtype=float)
    W = np.asarray(kernel, dtype=float)
    w = np.asarray(channels, dtype=float)
    Vref = np.asarray(eri_reference, dtype=float)
    A = np.asarray(trial, dtype=complex)
    batch = np.asarray(walkers, dtype=complex)
    weights = np.asarray(walker_weights, dtype=float)
    fields = np.asarray(auxiliary_fields, dtype=float)
    dt = float(time_step)
    tol = float(eri_tolerance)

    if batch.ndim != 3 or batch.shape[1:] != A.shape or batch.shape[0] == 0:
        raise ValueError("walkers have incompatible shape")
    if weights.shape != (batch.shape[0],):
        raise ValueError("walker_weights have incompatible shape")
    if fields.shape != (batch.shape[0], w.shape[0]):
        raise ValueError("auxiliary_fields have incompatible shape")
    if h.shape != (A.shape[0], A.shape[0]) or half.shape != h.shape:
        raise ValueError("one-body matrices have incompatible shape")
    if (
        w.ndim != 2
        or W.shape != (u.shape[1], u.shape[1])
        or w.shape[1] != u.shape[1]
    ):
        raise ValueError("ITHC channel shapes are incompatible")
    if not np.isfinite(tol) or tol < 0:
        raise ValueError("eri_tolerance must be finite and nonnegative")
    if not np.allclose(w.T @ w, W, rtol=1e-10, atol=1e-12):
        raise ValueError("channels do not factor the kernel")

    reconstructed = reconstruct_ithc_eri(u, W)
    if Vref.shape != reconstructed.shape:
        raise ValueError("eri_reference has incompatible shape")
    scale = max(1.0, float(np.linalg.norm(Vref)))
    if np.linalg.norm(reconstructed - Vref) > tol * scale:
        raise ValueError("ITHC factors fail the ERI certificate")

    local_energies = []
    importance = []
    for index in range(batch.shape[0]):
        old_walker = batch[index]
        half_walker = half @ old_walker

        green_before = extended_mixed_green(A, half_walker, u)
        bias = ithc_force_bias(green_before, w, dt)

        new_walker = propagate_ithc_walker(
            half_walker,
            half,
            u,
            w,
            fields[index],
            bias,
            dt,
        )

        importance.append(
            phaseless_importance(
                A,
                old_walker,
                new_walker,
                fields[index],
                bias,
            )
        )

        green_extended = extended_mixed_green(A, new_walker, u)
        green_physical = u @ green_extended @ u.T

        local_energies.append(
            ithc_local_energy(
                h,
                W,
                green_physical,
                green_extended,
            )
        )

    updated_weights = weights * np.asarray(importance, dtype=float)

    return phaseless_mixed_energy(
        np.asarray(local_energies, dtype=complex),
        updated_weights,
    )


def _make_ithc_fixture(
    seed,
    n_orbitals,
    n_auxiliary,
    n_electrons,
    n_walkers,
    time_step,
):
    import numpy as np

    rng = np.random.default_rng(seed)

    q, r = np.linalg.qr(
        rng.normal(size=(n_auxiliary, n_orbitals))
    )
    q = q * np.where(np.diag(r) >= 0.0, 1.0, -1.0)
    u = q.T

    channels = rng.normal(
        scale=0.48,
        size=(n_auxiliary, n_auxiliary),
    )
    channels /= np.sqrt(n_auxiliary)
    kernel = channels.T @ channels

    raw_h = rng.normal(
        scale=0.18,
        size=(n_orbitals, n_orbitals),
    )
    one_body = (
        0.5 * (raw_h + raw_h.T)
        - 0.55 * np.eye(n_orbitals)
    )

    eigvals, eigvecs = np.linalg.eigh(one_body)
    half = (
        eigvecs * np.exp(-0.5 * time_step * eigvals)
    ) @ eigvecs.T

    trial, trial_r = np.linalg.qr(
        rng.normal(size=(n_orbitals, n_electrons))
    )
    trial = trial * np.where(
        np.diag(trial_r) >= 0.0,
        1.0,
        -1.0,
    )

    walkers = []
    for _ in range(n_walkers):
        candidate = (
            trial
            + 0.11 * rng.normal(size=trial.shape)
            + 0.07j * rng.normal(size=trial.shape)
        )
        walker, walker_r = np.linalg.qr(candidate)
        diagonal = np.diag(walker_r)
        phase = np.where(
            np.abs(diagonal) > 0.0,
            diagonal / np.abs(diagonal),
            1.0,
        )
        walkers.append(walker * phase)

    walkers = np.asarray(walkers)
    weights = 0.4 + rng.random(n_walkers)
    fields = rng.normal(
        scale=0.7,
        size=(n_walkers, n_auxiliary),
    )

    eri_reference = np.einsum(
        "pa,qa,ab,rb,sb->pqrs",
        u,
        u,
        kernel,
        u,
        u,
        optimize=True,
    )

    return (
        one_body,
        half,
        u,
        kernel,
        channels,
        eri_reference,
        trial,
        walkers,
        weights,
        fields,
        time_step,
        1e-11,
    )
SCICODE_GOLD_EOF
