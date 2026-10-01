#!/bin/bash
# Oracle: writes the gold solution so the verifier scores reward 1.0.
cat > /app/solution.py <<'SCICODE_GOLD_EOF'
import numpy as np
try:
    import scipy, scipy.linalg, scipy.sparse, scipy.optimize, scipy.spatial, scipy.integrate, scipy.stats, scipy.special, scipy.signal
except Exception:
    pass
import math

def compute_promoter_occupancy(switch_rates: np.ndarray, transcription_rates: np.ndarray) -> np.ndarray:
    """Reference implementation (left null vector of the promoter generator)."""
    import numpy as np

    silent = np.asarray(switch_rates, dtype=float)
    firing = np.asarray(transcription_rates, dtype=float)
    if silent.ndim != 2 or silent.shape[0] != silent.shape[1] or silent.shape[0] < 1:
        raise ValueError("switch_rates must be a square (N, N) array with N >= 1")
    if firing.shape != silent.shape:
        raise ValueError("transcription_rates must have the shape of switch_rates")
    if not (np.all(np.isfinite(silent)) and np.all(np.isfinite(firing))):
        raise ValueError("rates must be finite")
    if np.any(silent < 0.0) or np.any(firing < 0.0):
        raise ValueError("rates must be nonnegative")
    if np.any(np.diag(silent) != 0.0):
        raise ValueError("switch_rates must have a zero diagonal")
    n = silent.shape[0]
    # Only state-changing events enter the promoter generator.  Diagonal
    # transcription events release a transcript but leave the promoter state
    # unchanged, so they cancel from the promoter-state balance exactly.
    state_change_rates = silent + firing
    np.fill_diagonal(state_change_rates, 0.0)
    generator = state_change_rates.copy()
    np.fill_diagonal(generator, -state_change_rates.sum(axis=1))

    if n == 1:
        return np.ones(1, dtype=float)

    # A finite chain has one stationary distribution exactly when it has one
    # closed communicating class.  Check that graph property directly instead
    # of deciding it from an ill-conditioned numerical rank threshold.
    reach = (state_change_rates > 0.0) | np.eye(n, dtype=bool)
    for k in range(n):
        reach |= reach[:, k, None] & reach[None, k, :]
    assigned = np.zeros(n, dtype=bool)
    closed_classes = 0
    for i in range(n):
        if assigned[i]:
            continue
        component = reach[i] & reach[:, i]
        assigned |= component
        if not np.any((state_change_rates[component] > 0.0)[:, ~component]):
            closed_classes += 1
    if closed_classes != 1:
        raise ValueError("the promoter chain has no unique stationary distribution")

    # Uniform rate rescaling cannot change a stationary distribution.  Scale
    # the balance equations before replacing one dependent equation by the
    # normalization condition, so very slow but otherwise ordinary chains do
    # not lose their state-balance information next to the unit-sum row.
    rate_scale = float(np.max(np.abs(generator)))
    scaled_generator = generator / rate_scale
    system = scaled_generator.T.copy()
    system[-1, :] = 1.0
    rhs = np.zeros(n)
    rhs[-1] = 1.0
    occupancy = np.linalg.solve(system, rhs)
    return occupancy.astype(float)

import numpy as np
def compute_burst_limit_fano(
    switch_rates: np.ndarray,
    transcription_rates: np.ndarray,
    burst_size: float,
    protein_decay: float,
) -> float:
    """Reference implementation (first two protein binomial moments of the burst model)."""
    import numpy as np

    for value in (burst_size, protein_decay):
        if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
            raise ValueError("burst_size and protein_decay must be real numbers")
        if not (np.isfinite(value) and value > 0.0):
            raise ValueError("burst_size and protein_decay must be positive and finite")
    beta, delta = float(burst_size), float(protein_decay)
    occupancy = compute_promoter_occupancy(switch_rates, transcription_rates)
    silent = np.asarray(switch_rates, dtype=float)
    firing = np.asarray(transcription_rates, dtype=float)
    generator = silent + firing
    np.fill_diagonal(generator, 0.0)
    np.fill_diagonal(generator, -generator.sum(axis=1))
    n = generator.shape[0]
    output = firing @ np.ones(n)
    s1 = float(occupancy @ output)
    if s1 <= 0.0:
        raise ValueError("the promoter never transcribes at stationarity")
    # Protein memory of the promoter state enters through the resolvent at the protein decay rate.
    t_delta = float(occupancy @ firing @ np.linalg.solve(delta * np.eye(n) - generator, output))
    b1 = beta * s1 / delta
    b2 = beta**2 / (2.0 * delta) * (s1 + t_delta)
    return float(1.0 + 2.0 * b2 / b1 - b1)

import numpy as np
def compute_protein_mean_and_fano(
    switch_rates: np.ndarray,
    transcription_rates: np.ndarray,
    mrna_decay: float,
    translation: float,
    protein_decay: float,
) -> np.ndarray:
    """Reference implementation (first two coarse-grained protein binomial moments)."""
    import numpy as np

    for value in (mrna_decay, translation, protein_decay):
        if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
            raise ValueError("scalar rates must be real numbers")
        if not (np.isfinite(value) and value > 0.0):
            raise ValueError("scalar rates must be positive and finite")
    u, v, delta = float(mrna_decay), float(translation), float(protein_decay)
    occupancy = compute_promoter_occupancy(switch_rates, transcription_rates)
    silent = np.asarray(switch_rates, dtype=float)
    firing = np.asarray(transcription_rates, dtype=float)
    generator = silent + firing
    np.fill_diagonal(generator, 0.0)
    np.fill_diagonal(generator, -generator.sum(axis=1))
    n = generator.shape[0]
    identity = np.eye(n)
    output = firing @ np.ones(n)
    s1 = float(occupancy @ output)
    if s1 <= 0.0:
        raise ValueError("the promoter never transcribes at stationarity")
    # Resolvents of the promoter generator act on the per-state transcription output.
    after_u = np.linalg.solve(u * identity - generator, output)
    after_ud = np.linalg.solve(u * identity - generator, np.linalg.solve(delta * identity - generator, output))
    t_u = float(occupancy @ firing @ after_u)
    t_ud = float(occupancy @ firing @ after_ud)
    b01 = v * s1 / (u * delta)
    b02 = (v**2 / (2.0 * u * delta * (u + delta)) * s1
           + v**2 / (2.0 * delta * (u + delta)) * t_ud
           + v**2 / (2.0 * u * delta * (u + delta)) * t_u)
    fano = 1.0 + 2.0 * b02 / b01 - b01
    return np.array([b01, fano], dtype=float)

import numpy as np
def infer_mrna_decay_rate(
    switch_rates: np.ndarray,
    transcription_rates: np.ndarray,
    protein_decay: float,
    burst_size: float,
    protein_fano: float,
) -> float:
    """Reference implementation (bracketed root of the exact Fano factor in log u)."""
    import numpy as np
    from scipy.optimize import brentq

    if isinstance(protein_fano, bool) or not isinstance(protein_fano, (int, float, np.integer, np.floating)):
        raise ValueError("protein_fano must be a real number")
    if not np.isfinite(protein_fano):
        raise ValueError("protein_fano must be finite")
    target = float(protein_fano)
    # The burst model is the infinite-decay limit at fixed burst size and bounds the
    # attainable Fano factor from above; an arbitrarily stable transcript approaches 1.
    ceiling = compute_burst_limit_fano(switch_rates, transcription_rates, burst_size, protein_decay)
    if not (1.0 < target < ceiling):
        raise ValueError("no positive transcript decay rate reproduces protein_fano")
    beta, delta = float(burst_size), float(protein_decay)

    def _fano_gap(log_u):
        u = float(np.exp(log_u))
        moments = compute_protein_mean_and_fano(
            switch_rates, transcription_rates, u, beta * u, delta)
        return float(moments[1]) - target

    low = high = float(np.log(delta))
    for _ in range(200):
        if _fano_gap(low) < 0.0:
            break
        low -= np.log(4.0)
    else:
        raise ValueError("no positive transcript decay rate reproduces protein_fano")
    for _ in range(200):
        if _fano_gap(high) > 0.0:
            break
        high += np.log(4.0)
    else:
        raise ValueError("no positive transcript decay rate reproduces protein_fano")
    root = brentq(_fano_gap, low, high, xtol=1.0e-14, rtol=1.0e-15, maxiter=500)
    return float(np.exp(root))

import numpy as np
def compute_binomial_moment_table(
    switch_rates: np.ndarray,
    transcription_rates: np.ndarray,
    mrna_decay: float,
    translation: float,
    protein_decay: float,
    max_layer: int,
) -> np.ndarray:
    """Reference implementation (layer-by-layer matrix hierarchy, coarse-grained)."""
    import numpy as np

    for value in (mrna_decay, translation, protein_decay):
        if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
            raise ValueError("scalar rates must be real numbers")
        if not (np.isfinite(value) and value > 0.0):
            raise ValueError("scalar rates must be positive and finite")
    if isinstance(max_layer, bool) or not isinstance(max_layer, (int, np.integer)) or max_layer < 0:
        raise ValueError("max_layer must be an integer of at least 0")
    u, v, delta, top = float(mrna_decay), float(translation), float(protein_decay), int(max_layer)
    occupancy = compute_promoter_occupancy(switch_rates, transcription_rates)
    firing = np.asarray(transcription_rates, dtype=float)
    generator = np.asarray(switch_rates, dtype=float) + firing
    np.fill_diagonal(generator, 0.0)
    np.fill_diagonal(generator, -generator.sum(axis=1))
    n = generator.shape[0]
    identity = np.eye(n)
    ones = np.ones(n)
    table = np.zeros((top + 1, top + 1))
    table[0, 0] = 1.0
    previous = [identity]  # matrix-form moments of the layer below, indexed by p
    for layer in range(1, top + 1):
        current = [None] * (layer + 1)
        # Within a layer the moments are solved from the largest p downwards.
        for p in range(layer, -1, -1):
            q = layer - p
            rhs = np.zeros((n, n))
            if p >= 1:
                rhs -= previous[p - 1] @ firing
            if q >= 1:
                rhs -= v * p * previous[p] + v * (p + 1) * current[p + 1]
            system = generator - (u * p + delta * q) * identity
            current[p] = np.linalg.solve(system.T, rhs.T).T
            table[p, q] = occupancy @ current[p] @ ones
        previous = current
    return table

import numpy as np
def compute_transcript_conditioned_protein_statistics(moment_table: np.ndarray, mrna_count: int) -> np.ndarray:
    """Reference implementation (alternating binomial-moment inversion in the transcript index)."""
    import math
    import numpy as np

    table = np.asarray(moment_table, dtype=float)
    if table.ndim != 2 or table.shape[0] != table.shape[1] or not np.all(np.isfinite(table)):
        raise ValueError("moment_table must be a square two-dimensional array of finite numbers")
    top = table.shape[0] - 1
    if isinstance(mrna_count, bool) or not isinstance(mrna_count, (int, np.integer)):
        raise ValueError("mrna_count must be an integer")
    m = int(mrna_count)
    if m < 0 or m > top - 2:
        raise ValueError("mrna_count must satisfy 0 <= mrna_count <= L - 2")
    slice_moments = []
    for q in range(3):
        terms = []
        for p in range(m, top - q + 1):
            value = table[p, q]
            if value == 0.0:
                continue
            # Binomial weights are formed in the log domain to keep large orders finite.
            log_weight = math.lgamma(p + 1) - math.lgamma(m + 1) - math.lgamma(p - m + 1)
            sign = -1.0 if (p - m) % 2 else 1.0
            terms.append(sign * math.copysign(math.exp(log_weight + math.log(abs(value))), value))
        slice_moments.append(math.fsum(terms))
    probability, first, second = slice_moments
    if not (probability > 0.0 and first > 0.0):
        raise ValueError("the transcript slice has no probability or no protein")
    mean = first / probability
    fano = (2.0 * second / probability + mean - mean * mean) / mean
    return np.array([probability, mean, fano], dtype=float)

import numpy as np
def predict_transcript_free_protein_mean(
    switch_rates: np.ndarray,
    transcription_rates: np.ndarray,
    stable_half_life: float,
    stable_mean: float,
    destabilized_half_life: float,
    reported_burst_size: float,
    max_layer: int,
) -> float:
    """Reference orchestrator using only the reference function chain."""
    import numpy as np

    for value in (stable_half_life, stable_mean, destabilized_half_life, reported_burst_size):
        if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
            raise ValueError("half-lives, stable_mean and reported_burst_size must be real numbers")
        if not (np.isfinite(value) and value > 0.0):
            raise ValueError("half-lives, stable_mean and reported_burst_size must be positive and finite")
    if isinstance(max_layer, bool) or not isinstance(max_layer, (int, np.integer)) or max_layer < 2:
        raise ValueError("max_layer must be an integer of at least 2")
    stable_decay = float(np.log(2.0) / stable_half_life)
    destabilized_decay = float(np.log(2.0) / destabilized_half_life)
    occupancy = compute_promoter_occupancy(switch_rates, transcription_rates)
    firing = np.asarray(transcription_rates, dtype=float)
    transcript_output = float(occupancy @ firing @ np.ones(firing.shape[0]))
    if transcript_output <= 0.0:
        raise ValueError("the promoter never transcribes at stationarity")
    # The protein mean fixes the proteins made per transcript, identically in both descriptions.
    burst_size = float(stable_mean) * stable_decay / transcript_output
    matched_fano = compute_burst_limit_fano(
        switch_rates, transcription_rates, float(reported_burst_size), destabilized_decay)
    decay = infer_mrna_decay_rate(
        switch_rates, transcription_rates, destabilized_decay, burst_size, matched_fano)
    translation = burst_size * decay
    check = compute_protein_mean_and_fano(
        switch_rates, transcription_rates, decay, translation, destabilized_decay)
    if not np.isclose(check[1], matched_fano, rtol=1.0e-9, atol=0.0):
        raise ValueError("the calibrated rates do not reproduce the matched Fano factor")
    table = compute_binomial_moment_table(
        switch_rates, transcription_rates, decay, translation, stable_decay, max_layer)
    statistics = compute_transcript_conditioned_protein_statistics(table, 0)
    return float(statistics[1])
SCICODE_GOLD_EOF
