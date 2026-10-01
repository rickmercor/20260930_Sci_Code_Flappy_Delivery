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

def _validate_rna_grammar(sequence, rule_weights, unpaired, pair_factors):
    if not isinstance(sequence, str) or not sequence or set(sequence) - set("ACGU"):
        raise ValueError("sequence must be a non-empty string over A, C, G, U")
    w, u, pf = (np.asarray(a, dtype=float) for a in (rule_weights, unpaired, pair_factors))
    if w.shape != (5,) or u.shape != (2, 4) or pf.shape != (4, 4):
        raise ValueError("grammar arrays have the wrong shape")
    allowed = np.zeros((4, 4), dtype=bool)
    allowed[[0, 3, 1, 2, 2, 3], [3, 0, 2, 1, 3, 2]] = True
    if any(not np.all(np.isfinite(a)) or np.any(a < 0.0) for a in (w, u, pf)) \
            or w[4] <= 0.0 or np.any(pf[~allowed] != 0.0):
        raise ValueError("factors must be finite and non-negative, t_E positive, and disallowed pairs zero")
    x = np.fromiter(("ACGU".index(ch) for ch in sequence), dtype=int, count=len(sequence))
    return x, w, u, pf

def _log_factor(values):
    values = np.asarray(values, dtype=float)
    result = np.full(values.shape, -np.inf, dtype=float)
    positive = values > 0.0
    result[positive] = np.log(values[positive])
    return result

def _logsumexp(values):
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return -np.inf
    maximum = float(np.max(values))
    if np.isneginf(maximum):
        return -np.inf
    return maximum + float(np.log(np.sum(np.exp(values - maximum))))

def compute_log_inside_table(sequence: str, rule_weights: "np.ndarray",
                                     unpaired: "np.ndarray", pair_factors: "np.ndarray") -> "np.ndarray":
    x, w, u, pf = _validate_rna_grammar(sequence, rule_weights, unpaired, pair_factors)
    n = len(x)
    lw, lu, lpf = (_log_factor(a) for a in (w, u, pf))
    table = np.full((n + 1, n + 1), -np.inf, dtype=float)
    np.fill_diagonal(table, lw[4])
    for span in range(1, n + 1):
        for i in range(n - span + 1):
            k = i + span
            terms = [
                lw[1] + lu[0, x[i]] + table[i + 1, k],
                lw[2] + lu[1, x[k - 1]] + table[i, k - 1],
            ]
            if span >= 2:
                terms.append(lw[0] + lpf[x[i], x[k - 1]] + table[i + 1, k - 1])
                split_terms = table[i, i + 1:k] + table[i + 1:k, k]
                terms.append(lw[3] + _logsumexp(split_terms))
            table[i, k] = _logsumexp(terms)
    return table

import numpy as np

def compute_log_outside_table(sequence: str, rule_weights: "np.ndarray", unpaired: "np.ndarray",
                                      pair_factors: "np.ndarray", log_inside: "np.ndarray") -> "np.ndarray":
    x, w, u, pf = _validate_rna_grammar(sequence, rule_weights, unpaired, pair_factors)
    n = len(x)
    table = np.asarray(log_inside, dtype=float)
    if table.shape != (n + 1, n + 1) or np.any(np.isnan(table)) or np.any(np.isposinf(table)) \
            or not np.isfinite(table[0, n]):
        raise ValueError("log_inside must have the right shape, no NaN or positive infinity, and a finite root")
    lw, lu, lpf = (_log_factor(a) for a in (w, u, pf))
    outside = np.full((n + 1, n + 1), -np.inf, dtype=float)
    outside[0, n] = 0.0
    for span in range(n, 0, -1):
        for i in range(n - span + 1):
            k = i + span
            parent = outside[i, k]
            if np.isneginf(parent):
                continue
            outside[i + 1, k] = np.logaddexp(
                outside[i + 1, k], parent + lw[1] + lu[0, x[i]]
            )
            outside[i, k - 1] = np.logaddexp(
                outside[i, k - 1], parent + lw[2] + lu[1, x[k - 1]]
            )
            if span >= 2:
                outside[i + 1, k - 1] = np.logaddexp(
                    outside[i + 1, k - 1], parent + lw[0] + lpf[x[i], x[k - 1]]
                )
                split = np.arange(i + 1, k)
                outside[i, split] = np.logaddexp(
                    outside[i, split], parent + lw[3] + table[split, k]
                )
                outside[split, k] = np.logaddexp(
                    outside[split, k], parent + lw[3] + table[i, split]
                )
    return outside

import numpy as np

def evaluate_single_replacement_channels(sequence: str, rule_weights: "np.ndarray",
                                                 unpaired: "np.ndarray", pair_factors: "np.ndarray",
                                                 log_inside: "np.ndarray",
                                                 log_outside: "np.ndarray") -> "np.ndarray":
    x, w, u, pf = _validate_rna_grammar(sequence, rule_weights, unpaired, pair_factors)
    n = len(x)
    inside, outside = (np.asarray(a, dtype=float) for a in (log_inside, log_outside))
    if any(a.shape != (n + 1, n + 1) or np.any(np.isnan(a)) or np.any(np.isposinf(a))
           for a in (inside, outside)) or not np.isfinite(inside[0, n]):
        raise ValueError("log tables must have the right shape, no NaN or positive infinity, and a finite root")
    lw, lu, lpf = (_log_factor(a) for a in (w, u, pf))
    result = np.full((n, 4, 4), -np.inf, dtype=float)
    for p in range(n):
        right_ends = np.arange(p + 2, n + 1)
        left_starts = np.arange(0, p)
        right_bounds = np.arange(p + 1, n + 1)
        left_bounds = np.arange(0, p + 1)
        p5_core = outside[p, right_ends] + lw[0] + inside[p + 1, right_ends - 1]
        p3_core = outside[left_starts, p + 1] + lw[0] + inside[left_starts + 1, p]
        left_core = outside[p, right_bounds] + inside[p + 1, right_bounds]
        right_core = outside[left_bounds, p + 1] + inside[left_bounds, p]
        for candidate in range(4):
            result[p, candidate, 0] = _logsumexp(
                p5_core + lpf[candidate, x[right_ends - 1]]
            )
            result[p, candidate, 1] = _logsumexp(
                p3_core + lpf[x[left_starts], candidate]
            )
            result[p, candidate, 2] = lw[1] + lu[0, candidate] + _logsumexp(left_core)
            result[p, candidate, 3] = lw[2] + lu[1, candidate] + _logsumexp(right_core)
    return result

import numpy as np

def rank_single_substitutions(sequence: str, channel_logs: "np.ndarray",
                                      minimum_gap: int) -> "np.ndarray":
    if not isinstance(sequence, str) or not sequence or set(sequence) - set("ACGU"):
        raise ValueError("sequence must be a non-empty RNA string")
    logs = np.asarray(channel_logs, dtype=float)
    n = len(sequence)
    if logs.shape != (n, 4, 4) or np.any(np.isnan(logs)) or np.any(np.isposinf(logs)):
        raise ValueError("channel_logs must have shape (n, 4, 4) with no NaN or positive infinity")
    if isinstance(minimum_gap, bool) or not isinstance(minimum_gap, (int, np.integer)) or minimum_gap < 1:
        raise ValueError("minimum_gap must be a positive integer")
    endpoints = np.logaddexp.reduce(logs, axis=2)
    x = np.fromiter(("ACGU".index(ch) for ch in sequence), dtype=int, count=n)
    reference_entries = endpoints[np.arange(n), x]
    if not np.all(np.isfinite(reference_entries)) \
            or not np.allclose(reference_entries, reference_entries[0], rtol=0.0, atol=1e-10):
        raise ValueError("unchanged-base endpoints must share one finite reference log weight")
    relative_logs = endpoints - reference_entries[0]
    scores = np.empty_like(relative_logs)
    increasing = relative_logs >= 0.0
    scores[increasing] = np.expm1(relative_logs[increasing])
    scores[~increasing] = -np.expm1(relative_logs[~increasing])
    if not np.all(np.isfinite(scores)):
        raise ValueError("relative substitution scores must be finite")
    scores[np.arange(n), x] = -1.0
    p, c = np.unravel_index(np.argmax(scores), scores.shape)
    separated = scores.copy()
    separated[max(0, p - int(minimum_gap) + 1):p + int(minimum_gap), :] = -1.0
    if separated.max() < 0.0:
        raise ValueError("no substitution lies at least minimum_gap positions from the strongest one")
    q, d = np.unravel_index(np.argmax(separated), separated.shape)
    return np.array([scores[p, c], p, c, scores[q, d], q, d], dtype=float)

import numpy as np

def evaluate_ordered_double_endpoints(sequence: str, rule_weights: "np.ndarray",
                                              unpaired: "np.ndarray", pair_factors: "np.ndarray",
                                              log_inside: "np.ndarray", channel_logs: "np.ndarray",
                                              position: int, nucleotide: str,
                                              partner_position: int,
                                              partner_nucleotide: str) -> "np.ndarray":
    x, w, u, pf = _validate_rna_grammar(sequence, rule_weights, unpaired, pair_factors)
    n = len(x)
    inside = np.asarray(log_inside, dtype=float)
    channels = np.asarray(channel_logs, dtype=float)
    if inside.shape != (n + 1, n + 1) or channels.shape != (n, 4, 4) \
            or np.any(np.isnan(inside)) or np.any(np.isposinf(inside)) \
            or np.any(np.isnan(channels)) or np.any(np.isposinf(channels)) \
            or not np.isfinite(inside[0, n]):
        raise ValueError("reference log outputs have the wrong shape or invalid values")
    for site, base in ((position, nucleotide), (partner_position, partner_nucleotide)):
        if isinstance(site, bool) or not isinstance(site, (int, np.integer)) \
                or not 0 <= site < n or not isinstance(base, str) or len(base) != 1 \
                or base not in "ACGU" or base == sequence[int(site)]:
            raise ValueError("each replacement must change a valid position to one RNA base")
    if int(position) == int(partner_position):
        raise ValueError("the two positions must be distinct")

    p, q = int(position), int(partner_position)
    c, d = "ACGU".index(nucleotide), "ACGU".index(partner_nucleotide)
    xp = x[p]
    totals = np.logaddexp.reduce(channels, axis=2)
    unchanged = totals[np.arange(n), x]
    if not np.all(np.isfinite(unchanged)) \
            or not np.allclose(unchanged, inside[0, n], rtol=0.0, atol=1e-9):
        raise ValueError("reference inside and replacement-channel outputs are inconsistent")

    q_background = sequence[:q] + partner_nucleotide + sequence[q + 1:]
    q_inside = compute_log_inside_table(q_background, w, u, pf)
    q_outside = compute_log_outside_table(q_background, w, u, pf, q_inside)
    q_channels = evaluate_single_replacement_channels(
        q_background, w, u, pf, q_inside, q_outside
    )
    q_totals = np.logaddexp.reduce(q_channels, axis=2)
    endpoint_logs = np.array([
        [inside[0, n], totals[p, c], totals[q, d], q_totals[p, c]],
        [channels[p, xp, 0], channels[p, c, 0], q_channels[p, xp, 0], q_channels[p, c, 0]],
    ], dtype=float)
    if np.any(endpoint_logs[1] > endpoint_logs[0] + 1e-9):
        raise ValueError("an event endpoint cannot exceed its corresponding total endpoint")
    return endpoint_logs

import numpy as np

def _signed_log_difference(positive_logs: "np.ndarray",
                           negative_logs: "np.ndarray") -> float:
    """Return sum(exp(positive_logs)) minus sum(exp(negative_logs)) stably."""
    positive_values = list(np.asarray(positive_logs, dtype=float).ravel())
    negative_values = list(np.asarray(negative_logs, dtype=float).ravel())
    unmatched_positive = []
    for value in positive_values:
        match = next((i for i, other in enumerate(negative_values) if value == other), None)
        if match is None:
            unmatched_positive.append(value)
        else:
            negative_values.pop(match)
    if not unmatched_positive and not negative_values:
        return 0.0
    positive = -np.inf if not unmatched_positive else float(
        np.logaddexp.reduce(np.asarray(unmatched_positive, dtype=float))
    )
    negative = -np.inf if not negative_values else float(
        np.logaddexp.reduce(np.asarray(negative_values, dtype=float))
    )
    if positive == negative:
        return 0.0
    larger, smaller = (positive, negative) if positive > negative else (negative, positive)
    log_magnitude = larger + float(np.log(-np.expm1(smaller - larger)))
    if log_magnitude > np.log(np.finfo(float).max):
        return np.inf if positive > negative else -np.inf
    magnitude = float(np.exp(log_magnitude))
    return magnitude if positive > negative else -magnitude

def compute_epistatic_pairing_metrics(log_endpoints: "np.ndarray") -> "np.ndarray":
    logs = np.asarray(log_endpoints, dtype=float)
    if logs.shape != (2, 4) or not np.all(np.isfinite(logs[0])) \
            or np.any(np.isnan(logs[1])) or np.any(np.isposinf(logs[1])) \
            or np.any(logs[1] > logs[0] + 1e-10):
        raise ValueError("log_endpoints has an invalid shape, total, or event endpoint")
    relative = logs[0] - logs[0, 0]
    interaction = _signed_log_difference(
        np.array([relative[3], 0.0]), np.array([relative[1], relative[2]])
    )
    share = 0.0 if np.isneginf(logs[1, 3]) else float(np.exp(logs[1, 3] - logs[0, 3]))
    if not np.isfinite(interaction) or not np.isfinite(share) or not 0.0 <= share <= 1.0 + 1e-12:
        raise ValueError("metrics must be finite and the event share must lie in [0, 1]")
    return np.array([interaction, min(1.0, share)], dtype=float)

import numpy as np
def compute_epistatic_pairing_share(
    length: int = 1500,
    seed: int = 20260912,
    minimum_gap: int = 200,
    rule_weights: tuple = (0.7, 0.44, 0.36, 0.24, 0.8),
    unpaired: tuple = ((0.28, 0.19, 0.27, 0.26), (0.24, 0.26, 0.21, 0.29)),
    pair_factors: tuple = ((0.0, 0.0, 0.0, 1.1), (0.0, 0.0, 1.9, 0.0), (0.0, 1.9, 0.0, 0.5), (1.1, 0.0, 0.7, 0.0)),
) -> float:
    import numpy as np
    for value, floor in ((length, 2), (seed, 0)):
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < floor:
            raise ValueError("length must be an integer >= 2 and seed a non-negative integer")
    grammar = (rule_weights, unpaired, pair_factors)
    sequence = "".join("ACGU"[v] for v in np.random.default_rng(int(seed)).integers(0, 4, int(length)))
    log_inside = compute_log_inside_table(sequence, *grammar)
    log_outside = compute_log_outside_table(sequence, *grammar, log_inside)
    channels = evaluate_single_replacement_channels(
        sequence, *grammar, log_inside, log_outside
    )
    ranked = rank_single_substitutions(sequence, channels, minimum_gap)
    p, c, q, d = int(ranked[1]), "ACGU"[int(ranked[2])], int(ranked[4]), "ACGU"[int(ranked[5])]
    log_endpoints = evaluate_ordered_double_endpoints(
        sequence, *grammar, log_inside, channels, p, c, q, d
    )
    metrics = compute_epistatic_pairing_metrics(log_endpoints)
    return float(metrics[1])
SCICODE_GOLD_EOF
