#!/bin/bash
# Oracle: writes the gold solution so the verifier scores reward 1.0.
cat > /app/solution.py <<'SCICODE_GOLD_EOF'
import numpy as np
try:
    import scipy, scipy.linalg, scipy.sparse, scipy.optimize, scipy.spatial, scipy.integrate, scipy.stats, scipy.special, scipy.signal
except Exception:
    pass
import math

import math
from numbers import Integral, Real


def compute_link_acceptance(
    beta: float,
    local_weight: float,
    local_count: int,
    move: int,
) -> float:
    import math
    from numbers import Integral, Real

    if isinstance(beta, bool) or not isinstance(beta, Real) or not math.isfinite(float(beta)) or float(beta) <= 0.0:
        raise ValueError("beta must be a finite real scalar > 0")
    if isinstance(local_weight, bool) or not isinstance(local_weight, Real) or not math.isfinite(float(local_weight)) or float(local_weight) <= 0.0:
        raise ValueError("local_weight must be a finite real scalar > 0")
    if isinstance(local_count, bool) or not isinstance(local_count, Integral) or int(local_count) < 0:
        raise ValueError("local_count must be a nonnegative integer")
    if isinstance(move, bool) or not isinstance(move, Integral) or int(move) not in (-1, 1):
        raise ValueError("move must be +1 or -1")
    n = int(local_count)
    if int(move) == -1 and n == 0:
        raise ValueError("cannot remove a link from zero count")
    raw = float(beta) * float(local_weight) / float(n + 1) if int(move) == 1 else float(n) / (float(beta) * float(local_weight))
    return float(min(1.0, raw))

import math
from numbers import Integral, Real

def apply_link_move(
    counts: np.ndarray,
    location: int,
    move: int,
    acceptance: float,
    uniform: float,
) -> np.ndarray:
    import math
    from numbers import Integral, Real
    import numpy as np

    raw = np.asarray(counts)
    if raw.ndim != 1 or raw.size < 1 or not np.issubdtype(raw.dtype, np.integer) or np.issubdtype(raw.dtype, np.bool_) or np.any(raw < 0):
        raise ValueError("counts must be a nonempty one-dimensional nonnegative integer array")
    if isinstance(location, bool) or not isinstance(location, Integral) or not (0 <= int(location) < raw.size):
        raise ValueError("location is out of range")
    if isinstance(move, bool) or not isinstance(move, Integral) or int(move) not in (-1, 1):
        raise ValueError("move must be +1 or -1")
    for name, value in (("acceptance", acceptance), ("uniform", uniform)):
        if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(float(value)) or not (0.0 <= float(value) <= 1.0):
            raise ValueError(f"{name} must be finite and in [0,1]")
    loc = int(location)
    if int(move) == -1 and int(raw[loc]) == 0:
        raise ValueError("cannot remove a link from zero count")
    updated = raw.astype(int, copy=True)
    accepted = int(float(uniform) < float(acceptance))
    if accepted:
        updated[loc] += int(move)
    out = np.empty(updated.size + 1, dtype=int)
    out[:-1] = updated
    out[-1] = accepted
    return out

from numbers import Integral

import numpy as np


def compute_ssm_projection_amplitudes(
    state_words: np.ndarray,
    lattice_size: int,
) -> np.ndarray:
    from numbers import Integral
    import numpy as np

    if (
        isinstance(lattice_size, bool)
        or not isinstance(lattice_size, Integral)
        or int(lattice_size) < 2
        or int(lattice_size) > 6
        or int(lattice_size) % 2 != 0
    ):
        raise ValueError("lattice_size must be an even integer in [2,6]")
    L = int(lattice_size)
    raw = np.asarray(state_words)
    if (
        raw.ndim != 1
        or raw.size < 1
        or not np.issubdtype(raw.dtype, np.integer)
        or np.issubdtype(raw.dtype, np.bool_)
        or np.any(raw < 0)
    ):
        raise ValueError("state_words must be a nonempty 1D nonnegative integer array")
    limit = 1 << (L * L)
    if any(int(word) >= limit for word in raw):
        raise ValueError("a state word has a bit outside the lattice")

    diagonal = []
    for family in (0, 1):
        for y in range(L):
            for x in range(L):
                if x % 2 != 0 or (x + y) % 2 != family:
                    continue
                jx = (x + 1) % L if family == 0 else (x - 1) % L
                jy = (y + 1) % L
                diagonal.append((x + L * y, jx + L * jy, x, y))
    horizontal = []
    for y in range(L):
        for x in range(L):
            if (x + y) % 2 != 0:
                continue
            jx = (x + 1) % L if x % 2 == 0 else (x - 1) % L
            horizontal.append((x + L * y, jx + L * y, x, y))

    n_diagonal = L * L // 2
    out = np.zeros((raw.size, 4 + n_diagonal), dtype=float)
    for row, encoded in enumerate(raw):
        word = int(encoded)
        spins = np.asarray(
            [-1 if (word >> site) & 1 else 1 for site in range(L * L)],
            dtype=int,
        )

        single_site = 1.0
        for site in range(L * L):
            x = site % L
            y = site // L
            if (x + y) % 2 == 1:
                single_site *= -1.0 if spins[site] == 1 else 1.0
        out[row, 0] = single_site

        diagonal_singlets = []
        diagonal_compatible = True
        diagonal_singlet_product = 1.0
        for i, j, x, y in diagonal:
            if spins[i] == spins[j]:
                diagonal_compatible = False
                break
            phase = -1.0 if (x + y) % 2 else 1.0
            local_singlet = phase * (1.0 if spins[i] == 1 else -1.0)
            diagonal_singlets.append(local_singlet)
            diagonal_singlet_product *= local_singlet
        if diagonal_compatible:
            out[row, 1] = 1.0
            for dimer, local_singlet in enumerate(diagonal_singlets):
                out[row, 4 + dimer] = diagonal_singlet_product / local_singlet

        horizontal_compatible = True
        horizontal_singlet_product = 1.0
        for i, j, _x, _y in horizontal:
            if spins[i] == spins[j]:
                horizontal_compatible = False
                break
            horizontal_singlet_product *= 1.0 if spins[i] == 1 else -1.0
        if horizontal_compatible:
            out[row, 2] = horizontal_singlet_product
            out[row, 3] = 1.0
    return out

import math
from numbers import Integral, Real

import numpy as np


def compute_pdms_contribution(
    sign: int,
    final_amplitudes: np.ndarray,
    initial_amplitudes: np.ndarray,
    total_links: int,
    beta: float,
) -> np.ndarray:
    import math
    from numbers import Integral, Real
    import numpy as np

    if isinstance(sign, bool) or not isinstance(sign, Integral) or int(sign) not in (-1, 1):
        raise ValueError("sign must be -1 or +1")
    f = np.asarray(final_amplitudes, dtype=float)
    i = np.asarray(initial_amplitudes, dtype=float)
    if f.ndim != 1 or i.ndim != 1 or f.size < 1 or f.shape != i.shape or not np.all(np.isfinite(f)) or not np.all(np.isfinite(i)):
        raise ValueError("amplitude vectors must be matched nonempty finite one-dimensional arrays")
    if isinstance(total_links, bool) or not isinstance(total_links, Integral) or int(total_links) < 0:
        raise ValueError("total_links must be a nonnegative integer")
    if isinstance(beta, bool) or not isinstance(beta, Real) or not math.isfinite(float(beta)) or float(beta) <= 0.0:
        raise ValueError("beta must be a finite real scalar > 0")
    density = float(int(sign)) * np.outer(f, i)
    shifted = -(float(int(total_links)) / float(beta)) * density
    return np.stack((density, shifted), axis=0)

import numpy as np


def average_sector_blocks(sample_terms: np.ndarray) -> np.ndarray:
    import numpy as np

    terms = np.asarray(sample_terms, dtype=float)
    if (
        terms.ndim != 6
        or terms.shape[0] != 2
        or terms.shape[1] != 2
        or terms.shape[2] < 3
        or terms.shape[3] < 1
        or terms.shape[4] < 1
        or terms.shape[4] != terms.shape[5]
        or not np.all(np.isfinite(terms))
    ):
        raise ValueError(
            "sample_terms must have finite shape (2,2,blocks>=3,samples>=1,d,d)"
        )
    means = np.mean(terms, axis=3)
    return 0.5 * (means + np.swapaxes(means, -1, -2))

import math
from numbers import Integral, Real

import numpy as np


def compute_bootstrap_rank_pair_scan(
    block_matrices: np.ndarray,
    energy_shift: float,
    plus_dimension: int,
    candidate_pairs: np.ndarray,
    relative_cutoffs: np.ndarray,
    n_bootstrap: int,
    bootstrap_seed: int,
) -> np.ndarray:
    import math
    from numbers import Integral, Real
    import numpy as np

    blocks = np.asarray(block_matrices, dtype=float)
    if (
        blocks.ndim != 5
        or blocks.shape[0] != 2
        or blocks.shape[1] != 2
        or blocks.shape[2] < 3
        or blocks.shape[3] < 3
        or blocks.shape[3] != blocks.shape[4]
        or not np.all(np.isfinite(blocks))
    ):
        raise ValueError("block_matrices must have finite shape (2,2,blocks>=3,d,d), d>=3")
    d = int(blocks.shape[3])
    if (
        isinstance(plus_dimension, bool)
        or not isinstance(plus_dimension, Integral)
        or not (2 <= int(plus_dimension) <= d)
    ):
        raise ValueError("plus_dimension must be an integer in [2,d]")
    p_dim = int(plus_dimension)

    raw_pairs = np.asarray(candidate_pairs)
    if (
        raw_pairs.ndim != 2
        or raw_pairs.shape[0] < 1
        or raw_pairs.shape[1] != 2
        or not np.issubdtype(raw_pairs.dtype, np.integer)
        or np.issubdtype(raw_pairs.dtype, np.bool_)
    ):
        raise ValueError("candidate_pairs must have nonempty integer shape (candidates,2)")
    pairs = raw_pairs.astype(int, copy=False)
    if len({(int(a), int(b)) for a, b in pairs}) != pairs.shape[0]:
        raise ValueError("candidate_pairs must not contain duplicates")
    if np.any(pairs[:, 0] < 2) or np.any(pairs[:, 0] > p_dim):
        raise ValueError("a Q=+1 candidate rank is out of range")
    if np.any(pairs[:, 1] < 2) or np.any(pairs[:, 1] > d):
        raise ValueError("a Q=-1 candidate rank is out of range")

    cuts = np.asarray(relative_cutoffs, dtype=float)
    if cuts.shape != (2,) or not np.all(np.isfinite(cuts)) or np.any(cuts < 0.0) or np.any(cuts >= 1.0):
        raise ValueError("relative_cutoffs must contain two finite values in [0,1)")
    if isinstance(energy_shift, bool) or not isinstance(energy_shift, Real) or not math.isfinite(float(energy_shift)):
        raise ValueError("energy_shift must be a finite real scalar")
    if isinstance(n_bootstrap, bool) or not isinstance(n_bootstrap, Integral) or int(n_bootstrap) < 5:
        raise ValueError("n_bootstrap must be an integer >= 5")
    if isinstance(bootstrap_seed, bool) or not isinstance(bootstrap_seed, Integral) or int(bootstrap_seed) < 0:
        raise ValueError("bootstrap_seed must be a nonnegative integer")

    def _retained_spectrum(z_matrix, e_matrix, rank, cutoff, enforce_cutoff):
        z = 0.5 * (z_matrix + z_matrix.T)
        e = 0.5 * (e_matrix + e_matrix.T)
        eigenvalues, eigenvectors = np.linalg.eigh(z)
        if not np.all(np.isfinite(eigenvalues)) or eigenvalues[-1] <= 0.0:
            return None
        retained = eigenvalues[-int(rank):]
        if np.any(retained <= 0.0):
            return None
        if enforce_cutoff and np.any(retained <= float(cutoff) * eigenvalues[-1]):
            return None
        u = eigenvectors[:, -int(rank):]
        scales = np.sqrt(retained)
        h_eff = (u.T @ e @ u) / (scales[:, None] * scales[None, :])
        h_eff = 0.5 * (h_eff + h_eff.T) + float(energy_shift) * np.eye(int(rank))
        energies = np.linalg.eigvalsh(h_eff)
        if not np.all(np.isfinite(energies)):
            return None
        return energies

    block_count = int(blocks.shape[2])
    indices = np.random.default_rng(int(bootstrap_seed)).integers(
        0, block_count, size=(int(n_bootstrap), block_count)
    )
    central_plus_z = np.mean(blocks[0, 0, :, :p_dim, :p_dim], axis=0)
    central_plus_e = np.mean(blocks[0, 1, :, :p_dim, :p_dim], axis=0)
    central_minus_z = np.mean(blocks[1, 0], axis=0)
    central_minus_e = np.mean(blocks[1, 1], axis=0)
    max_float = np.finfo(float).max
    scan = np.empty((pairs.shape[0], 7), dtype=float)

    for row_index, (rank_plus, rank_minus) in enumerate(pairs):
        central_plus = _retained_spectrum(
            central_plus_z, central_plus_e, int(rank_plus), cuts[0], False
        )
        central_minus = _retained_spectrum(
            central_minus_z, central_minus_e, int(rank_minus), cuts[1], False
        )
        if central_plus is None or central_minus is None:
            raise ValueError("a central retained density spectrum is not positive")
        central_gap = float(central_minus[0] - central_plus[0])
        if not math.isfinite(central_gap) or central_gap <= 0.0:
            raise ValueError("a central candidate does not have a positive finite intersector gap")

        valid_values = []
        for bootstrap_row in indices:
            plus_z = np.mean(blocks[0, 0, bootstrap_row, :p_dim, :p_dim], axis=0)
            plus_e = np.mean(blocks[0, 1, bootstrap_row, :p_dim, :p_dim], axis=0)
            minus_z = np.mean(blocks[1, 0, bootstrap_row], axis=0)
            minus_e = np.mean(blocks[1, 1, bootstrap_row], axis=0)
            plus_energies = _retained_spectrum(
                plus_z, plus_e, int(rank_plus), cuts[0], True
            )
            minus_energies = _retained_spectrum(
                minus_z, minus_e, int(rank_minus), cuts[1], True
            )
            if plus_energies is None or minus_energies is None:
                continue
            gap = float(minus_energies[0] - plus_energies[0])
            if math.isfinite(gap) and gap > 0.0:
                valid_values.append(gap)

        valid = np.asarray(valid_values, dtype=float)
        fraction = float(valid.size) / float(int(n_bootstrap))
        if valid.size == 0:
            spread = max_float
            bootstrap_mean = max_float
            corrected = max_float
        else:
            bootstrap_mean = float(np.mean(valid))
            corrected = float(2.0 * central_gap - bootstrap_mean)
            logs = np.log(valid)
            center = np.median(logs)
            spread = float(1.4826 * np.median(np.abs(logs - center)))
        scan[row_index] = (
            float(rank_plus),
            float(rank_minus),
            fraction,
            spread,
            central_gap,
            bootstrap_mean,
            corrected,
        )
    return scan

import math
from numbers import Real

import numpy as np


def select_stable_bias_corrected_gap(
    rank_pair_scan: np.ndarray,
    min_valid_fraction: float,
    max_log_mad: float,
) -> float:
    import math
    from numbers import Real
    import numpy as np

    scan = np.asarray(rank_pair_scan, dtype=float)
    if scan.ndim != 2 or scan.shape[0] < 1 or scan.shape[1] != 7 or not np.all(np.isfinite(scan)):
        raise ValueError("rank_pair_scan must have finite shape (candidates,7)")
    if (
        isinstance(min_valid_fraction, bool)
        or not isinstance(min_valid_fraction, Real)
        or not math.isfinite(float(min_valid_fraction))
        or not (0.0 < float(min_valid_fraction) <= 1.0)
    ):
        raise ValueError("min_valid_fraction must lie in (0,1]")
    if (
        isinstance(max_log_mad, bool)
        or not isinstance(max_log_mad, Real)
        or not math.isfinite(float(max_log_mad))
        or float(max_log_mad) < 0.0
    ):
        raise ValueError("max_log_mad must be finite and nonnegative")

    labels = scan[:, :2]
    rounded = np.rint(labels)
    if np.any(labels != rounded) or np.any(rounded < 2):
        raise ValueError("candidate rank labels must be integer-valued and at least two")
    if len({(int(a), int(b)) for a, b in rounded}) != scan.shape[0]:
        raise ValueError("candidate rank pairs must be unique")
    if np.any(scan[:, 2] < 0.0) or np.any(scan[:, 2] > 1.0):
        raise ValueError("valid fractions must lie in [0,1]")
    if np.any(scan[:, 3] < 0.0) or np.any(scan[:, 4] <= 0.0):
        raise ValueError("spreads must be nonnegative and central gaps positive")

    max_float = np.finfo(float).max
    for row in scan:
        corrected = float(row[6])
        if (
            row[2] >= float(min_valid_fraction)
            and row[3] <= float(max_log_mad)
            and corrected > 0.0
            and corrected < max_float
        ):
            return corrected
    raise ValueError("no candidate rank pair satisfies the stability thresholds")

import math
from numbers import Integral, Real

import numpy as np


def run_ssm_bootstrap_gap(
    beta: float,
    coupling_j: float,
    coupling_j_prime: float,
    initial_counts: np.ndarray,
    locations: np.ndarray,
    moves: np.ndarray,
    local_weights: np.ndarray,
    uniforms: np.ndarray,
    state_words: np.ndarray,
    lattice_size: int,
    relative_cutoffs: np.ndarray,
    n_bootstrap: int,
    bootstrap_seed: int,
    min_valid_fraction: float,
    max_log_mad: float,
) -> float:
    import math
    from numbers import Integral, Real
    import numpy as np

    if (
        isinstance(lattice_size, bool)
        or not isinstance(lattice_size, Integral)
        or int(lattice_size) < 4
        or int(lattice_size) > 6
        or int(lattice_size) % 2 != 0
    ):
        raise ValueError("lattice_size must be an even integer in [4,6]")
    L = int(lattice_size)
    for name, value, positive in (
        ("coupling_j", coupling_j, True),
        ("coupling_j_prime", coupling_j_prime, False),
    ):
        if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(float(value)):
            raise ValueError(f"{name} must be a finite real scalar")
        if (positive and float(value) <= 0.0) or (not positive and float(value) < 0.0):
            raise ValueError(f"{name} is outside its antiferromagnetic domain")

    counts0 = np.asarray(initial_counts)
    loc = np.asarray(locations)
    mov = np.asarray(moves)
    weights = np.asarray(local_weights, dtype=float)
    draws = np.asarray(uniforms, dtype=float)
    if (
        counts0.ndim != 3
        or counts0.shape[0] != 2
        or counts0.shape[1] < 3
        or counts0.shape[2] < 1
        or not np.issubdtype(counts0.dtype, np.integer)
        or np.issubdtype(counts0.dtype, np.bool_)
        or np.any(counts0 < 0)
    ):
        raise ValueError("initial_counts must have nonnegative integer shape (2,blocks>=3,locations>=1)")
    if (
        loc.ndim != 3
        or loc.shape[0] != 2
        or loc.shape[1] != counts0.shape[1]
        or loc.shape[2] < 161
        or mov.shape != loc.shape
        or weights.shape != loc.shape
        or draws.shape != loc.shape
    ):
        raise ValueError("proposal arrays must share shape (2,blocks,samples>=161)")
    if (
        not np.issubdtype(loc.dtype, np.integer)
        or np.issubdtype(loc.dtype, np.bool_)
        or not np.issubdtype(mov.dtype, np.integer)
        or np.issubdtype(mov.dtype, np.bool_)
        or np.any(loc < 0)
        or np.any(loc >= counts0.shape[2])
        or np.any((mov != 1) & (mov != -1))
        or not np.all(np.isfinite(weights))
        or np.any(weights <= 0.0)
        or not np.all(np.isfinite(draws))
        or np.any(draws < 0.0)
        or np.any(draws > 1.0)
    ):
        raise ValueError("proposal arrays contain an invalid location, move, weight, or uniform")

    words = np.asarray(state_words)
    if (
        words.ndim != 1
        or words.size < 4
        or words.size % 2 != 0
        or not np.issubdtype(words.dtype, np.integer)
        or np.issubdtype(words.dtype, np.bool_)
    ):
        raise ValueError("state_words must be a nonempty even-length integer vector")

    amplitudes = compute_ssm_projection_amplitudes(words, L)
    minus_dimension = L * L // 2
    matrix_dimension = max(4, minus_dimension)
    sectors, blocks, samples = loc.shape
    terms = np.zeros(
        (2, 2, blocks, samples, matrix_dimension, matrix_dimension),
        dtype=float,
    )
    state_moduli = (int(words.size), int(words.size // 2))

    for sector in range(sectors):
        modulus = state_moduli[sector]
        if modulus < 2:
            raise ValueError("each sector needs at least two addressable state words")
        for block in range(blocks):
            current = counts0[sector, block].astype(int, copy=True)
            accepted_so_far = 0
            for proposal in range(samples):
                location = int(loc[sector, block, proposal])
                acceptance = compute_link_acceptance(
                    beta,
                    float(weights[sector, block, proposal]),
                    int(current[location]),
                    int(mov[sector, block, proposal]),
                )
                packed = apply_link_move(
                    current,
                    location,
                    int(mov[sector, block, proposal]),
                    acceptance,
                    float(draws[sector, block, proposal]),
                )
                current = packed[:-1].astype(int, copy=True)
                accepted_so_far += int(packed[-1])
                total_links = int(np.sum(current))

                final_index = (
                    17 * proposal
                    + 11 * block
                    + 3 * total_links
                    + accepted_so_far
                    + 13 * sector
                ) % modulus
                if proposal < 160:
                    initial_index = final_index
                    sign = 1
                else:
                    initial_index = (
                        29 * proposal
                        + 7 * block
                        + 5 * total_links
                        + 2 * accepted_so_far
                        + 11
                        + 17 * sector
                    ) % modulus
                    if initial_index == final_index:
                        initial_index = (
                            initial_index
                            + 1
                            + ((total_links + accepted_so_far) % (modulus - 1))
                        ) % modulus
                    hash_value = (
                        3 * block
                        + 5 * proposal
                        + total_links
                        + 2 * accepted_so_far
                        + 7 * sector
                    ) % 13
                    sign = -1 if hash_value in (0, 3) else 1

                if sector == 0:
                    final_vector = amplitudes[final_index, :4]
                    initial_vector = amplitudes[initial_index, :4]
                    dimension = 4
                else:
                    final_vector = amplitudes[final_index, 4:]
                    initial_vector = amplitudes[initial_index, 4:]
                    dimension = minus_dimension
                contribution = compute_pdms_contribution(
                    sign,
                    final_vector,
                    initial_vector,
                    total_links,
                    beta,
                )
                terms[sector, :, block, proposal, :dimension, :dimension] = contribution

    block_matrices = average_sector_blocks(terms)
    energy_shift = (
        float(coupling_j_prime) / 2.0 + float(coupling_j) / 8.0
    ) * float(L * L)
    candidate_pairs = np.asarray(
        sorted(
            (
                (rank_plus, rank_minus)
                for rank_plus in (4, 3, 2)
                for rank_minus in range(minus_dimension, 2, -1)
            ),
            key=lambda pair: (-(pair[0] + pair[1]), -pair[1], -pair[0]),
        ),
        dtype=int,
    )
    scan = compute_bootstrap_rank_pair_scan(
        block_matrices,
        energy_shift,
        4,
        candidate_pairs,
        relative_cutoffs,
        n_bootstrap,
        bootstrap_seed,
    )
    return select_stable_bias_corrected_gap(
        scan,
        min_valid_fraction,
        max_log_mad,
    )
SCICODE_GOLD_EOF
