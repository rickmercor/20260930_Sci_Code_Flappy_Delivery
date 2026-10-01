"""
Run the reduction over a candidate ensemble of consecutive seeds, resolve a multiway near-maximin pool by a robust optimally matched shadow-ensemble medoid, and return the single basis-sensitive scalar endpoint.

The calculation composes the polar factor, the seeded subspace sample, the real basis, the half-size block reduction, the Hermitian eigensolve, the real recovery, and the two weighted basis checksums, and it repeats that composition once for every candidate seed. One candidate supplies the reported spectrum and checksums; the ensemble as a whole supplies the spread of $C_Q$, which is what makes the endpoint a measure of basis sensitivity rather than of one arbitrary draw. When at least three candidates have nearly equal worst QR pivots, each sampling regime is audited with five independent shadow reductions. Robustly standardized conditioning and basis-checksum profiles are compared by optimal lane matching, and the medoid regime is selected before deterministic log-volume, pivot, and index tie-breaks. With one-based $j$ and $C_Q^{(k)}$ the second checksum of candidate $k$, the endpoint is



$$

S=\sum_{j=1}^{n}j\sigma_j+10^{-3}C_Z+10^{-4}C_Q+10^{-2}\Bigl(\max_kC_Q^{(k)}-\min_kC_Q^{(k)}\Bigr).

$$



Because every term after the first depends on the constructed bases, clean but incorrect sign, phase, eigenspace gauge, random-stream partition, robust scaling, or matching choices change the final scalar.

Returns
-------
one finite float equal to sum(j * sigma_j) + 1e-3 * C_Z + 1e-4 * C_Q + 1e-2 * (max_k C_Q_k - min_k C_Q_k) for the robust near-maximin medoid candidate; ValueError when the documented input contract fails
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def compute_basis_sensitive_skew_scalar(
    A: np.ndarray,
    seed: int = 260812153,
    candidates: int = 5,
) -> float:
    """Return the basis-sensitive scalar from the candidate ensemble.

    Raises ``ValueError`` unless every one of the following holds: ``candidates``
    is an integer, not a bool, of at least two; ``seed`` is an integer, not a
    bool; every validation rule of the seven earlier stages passes on the arrays
    this function derives for each candidate, so a singular ``A`` such as
    ``numpy.zeros((4, 4))`` is rejected rather than reduced; and the assembled
    scalar is finite.

    Parameters
    ----------
    A : np.ndarray
        Finite nonsingular real skew-symmetric array of even order.
    seed : int, optional
        Seed of the first candidate.  Candidate ``k`` uses ``seed + k``.
    candidates : int, optional
        Number of consecutive seeds in the ensemble.

    Returns
    -------
    float
        One finite scalar.  Every candidate runs the full reduction and yields a
        score ``min(abs(R[j, j]))`` from step two together with ``C_Z``, ``C_Q``
        and ``sigma``.  Let ``s_max`` be the largest score and retain every
        candidate with score at least ``0.95*s_max``.  If the pool has at least
        three members, create five independent shadow seeds per original
        candidate by applying ``SeedSequence`` to the two 32-bit words of
        ``seed``, then ``order``, ``candidates``, ``0x4e4c4131``, and
        ``0x34534844``; call ``spawn(candidates*5)`` in candidate-major order and
        draw one ``uint64`` from each child.  Run the complete seven-stage
        reduction on every shadow seed and describe it by
        ``[min(abs(diag(R))), sum(log(abs(diag(R)))), C_Z, C_Q]``.  Normalize
        all pooled shadow rows componentwise by the median and by
        ``max(MAD, 128*eps*max(1,max(abs(column))))``.  The distance between two
        original candidates is the minimum, over all ``5!`` lane permutations,
        of the mean Euclidean matched-row distance.  Select the candidate whose
        sum of distances to the other pool members is smallest; values within
        ``256*eps*max(1,max(abs(distance_sums)))`` tie.  For one- or two-member
        pools this medoid stage is vacuous.  Resolve any remaining tie by
        maximizing ``sum(log(abs(R[j,j])))``, then the original score, with
        respective tolerances ``64*eps*max(1,abs(best_log_volume))`` and
        ``64*eps*max(1,s_max)``, and finally take the earliest index.  With that
        candidate's values the result is
        ``sum(j * sigma_j) + 1e-3 * C_Z + 1e-4 * C_Q``
        ``+ 1e-2 * (max_k C_Q_k - min_k C_Q_k)``.
    """
    return result  # noqa: F821 - required model stub

# =============================================================================
# GOLD SOLUTION
# =============================================================================

from itertools import permutations

import numpy as np


_NLA14_SHADOW_LANES = 5
_NLA14_MATCHINGS = tuple(permutations(range(_NLA14_SHADOW_LANES)))


def _nla14_run_candidate(A, polar, candidate_seed):
    """Run all seven earlier stages and return one candidate record."""
    order = polar.shape[0]
    n = order // 2
    packed = _oracle_sample_positive_imaginary_basis(polar, candidate_seed)
    diagonal = packed[order].real
    real_basis = _oracle_build_real_symplectic_basis(packed[:order])
    blocks = _oracle_compute_skew_hamiltonian_blocks(A, real_basis)
    eigensystem = _oracle_solve_half_size_hermitian_problem(blocks)
    spectral_data = _oracle_recover_real_spectral_decomposition(
        real_basis,
        eigensystem,
    )
    checksums = _oracle_compute_basis_checksums(
        real_basis,
        spectral_data[:order],
    )
    return (
        float(diagonal.min()),
        float(np.log(diagonal).sum()),
        float(checksums[0]),
        float(checksums[1]),
        spectral_data[order, :n].copy(),
    )


def _nla14_shadow_seed_grid(seed, order, candidates):
    """Create platform-stable independent uint64 streams for the shadow audit."""
    mask = (1 << 32) - 1
    entropy = [
        int(seed) & mask,
        (int(seed) >> 32) & mask,
        int(order) & mask,
        int(candidates) & mask,
        0x4E4C4131,
        0x34534844,
    ]
    root = np.random.SeedSequence(entropy)
    children = root.spawn(int(candidates) * _NLA14_SHADOW_LANES)
    seeds = np.empty(
        (int(candidates), _NLA14_SHADOW_LANES),
        dtype=np.uint64,
    )
    for flat_index, child in enumerate(children):
        seeds.flat[flat_index] = child.generate_state(1, dtype=np.uint64)[0]
    return seeds


def _nla14_matching_distance(left, right):
    """Return the minimum mean Euclidean cost over all five-lane matchings."""
    best = np.inf
    for matching in _NLA14_MATCHINGS:
        cost = 0.0
        for lane, other_lane in enumerate(matching):
            cost += float(np.linalg.norm(left[lane] - right[other_lane]))
        best = min(best, cost)
    return float(best / _NLA14_SHADOW_LANES)


def _nla14_select_medoid(A, polar, seed, candidates, pool, records):
    """Select a robust finalist from a near-maximin pool."""
    pool = np.asarray(pool, dtype=int)
    if pool.size >= 3:
        seeds = _nla14_shadow_seed_grid(seed, polar.shape[0], candidates)
        raw = np.empty((pool.size, _NLA14_SHADOW_LANES, 4), dtype=float)
        for local_index, candidate_index in enumerate(pool):
            for lane in range(_NLA14_SHADOW_LANES):
                record = _nla14_run_candidate(
                    A,
                    polar,
                    int(seeds[candidate_index, lane]),
                )
                raw[local_index, lane] = record[:4]

        flattened = raw.reshape(-1, raw.shape[-1])
        center = np.median(flattened, axis=0)
        mad = np.median(np.abs(flattened - center), axis=0)
        magnitude = np.maximum(1.0, np.max(np.abs(flattened), axis=0))
        scale = np.maximum(mad, 128.0 * np.finfo(float).eps * magnitude)
        profiles = (raw - center) / scale

        distances = np.zeros((pool.size, pool.size), dtype=float)
        for left in range(pool.size):
            for right in range(left + 1, pool.size):
                distance = _nla14_matching_distance(
                    profiles[left],
                    profiles[right],
                )
                distances[left, right] = distance
                distances[right, left] = distance
        eccentricities = distances.sum(axis=1)
        best_eccentricity = float(eccentricities.min())
        eccentricity_tolerance = (
            256.0
            * np.finfo(float).eps
            * max(1.0, float(np.max(np.abs(eccentricities))))
        )
        pool = pool[
            eccentricities <= best_eccentricity + eccentricity_tolerance
        ]

    log_volumes = np.array([records[index][1] for index in pool])
    best_volume = float(log_volumes.max())
    volume_tolerance = 64.0 * np.finfo(float).eps * max(1.0, abs(best_volume))
    pool = pool[log_volumes >= best_volume - volume_tolerance]

    scores = np.array([records[index][0] for index in pool])
    finalist_score = float(scores.max())
    best_score = max(record[0] for record in records)
    score_tolerance = 64.0 * np.finfo(float).eps * max(1.0, best_score)
    return int(pool[np.flatnonzero(scores >= finalist_score - score_tolerance)[0]])


def _oracle_compute_basis_sensitive_skew_scalar(
    A: np.ndarray,
    seed: int = 260812153,
    candidates: int = 5,
) -> float:
    """Reference candidate-ensemble calculation composed from the seven earlier oracles."""
    if isinstance(candidates, (bool, np.bool_)) or not isinstance(
        candidates, (int, np.integer)
    ):
        raise ValueError("candidates must be an integer")
    if int(candidates) < 2:
        raise ValueError("candidates must be at least two")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an integer")

    polar = _oracle_compute_skew_polar_factor(A)
    order = polar.shape[0]
    n = order // 2

    records = []
    for index in range(int(candidates)):
        packed = _oracle_sample_positive_imaginary_basis(
            polar,
            int(seed) + index,
        )
        diagonal = packed[order].real
        real_basis = _oracle_build_real_symplectic_basis(packed[:order])
        blocks = _oracle_compute_skew_hamiltonian_blocks(A, real_basis)
        eigensystem = _oracle_solve_half_size_hermitian_problem(blocks)
        spectral_data = _oracle_recover_real_spectral_decomposition(
            real_basis,
            eigensystem,
        )
        checksums = _oracle_compute_basis_checksums(
            real_basis,
            spectral_data[:order],
        )
        records.append(
            (
                float(diagonal.min()),
                float(np.log(diagonal).sum()),
                float(checksums[0]),
                float(checksums[1]),
                spectral_data[order, :n].copy(),
            )
        )

    scores = np.array([record[0] for record in records])
    z_sums = np.array([record[2] for record in records])
    q_sums = np.array([record[3] for record in records])
    best_score = float(scores.max())
    pool = np.flatnonzero(scores >= 0.95 * best_score)
    selected = _nla14_select_medoid(
        A,
        polar,
        int(seed),
        int(candidates),
        pool,
        records,
    )
    spread = float(q_sums.max() - q_sums.min())
    indices = np.arange(1, n + 1, dtype=float)
    result = float(
        indices @ records[selected][4]
        + 1e-3 * z_sums[selected]
        + 1e-4 * q_sums[selected]
        + 1e-2 * spread
    )
    if not np.isfinite(result):
        raise ValueError("the final scalar is not finite")
    return result

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return pipelines with clustered spectra and candidate tournaments."""
    return [
        {
            "setup": """import numpy as np
sigma = np.diag([3.0, 0.5])
A = np.block([[np.zeros((2,2)), -sigma], [sigma, np.zeros((2,2))]])
seed = 7
""",
            "call": "compute_basis_sensitive_skew_scalar(A, seed)",
            "gold_call": "_oracle_compute_basis_sensitive_skew_scalar(A, seed)",
        },
        {
            "setup": """import numpy as np
values = np.array([5.0, 2.0, 0.4, 0.05])
S = np.block([[np.zeros((4,4)), -np.diag(values)], [np.diag(values), np.zeros((4,4))]])
Q0 = np.eye(8)
for p, q, theta in [(0,5,0.37),(2,7,-0.52),(1,4,0.81),(3,6,-0.28),(0,2,0.44)]:
    G = np.eye(8)
    c, s = np.cos(theta), np.sin(theta)
    G[p,p] = G[q,q] = c
    G[p,q], G[q,p] = -s, s
    Q0 = Q0 @ G
A = Q0 @ S @ Q0.T
seed = 314159
""",
            "call": "compute_basis_sensitive_skew_scalar(A, seed)",
            "gold_call": "_oracle_compute_basis_sensitive_skew_scalar(A, seed)",
        },
        {
            "setup": """import numpy as np
values = np.array([100.0, 1.0, 0.01])
S = np.block([[np.zeros((3,3)), -np.diag(values)], [np.diag(values), np.zeros((3,3))]])
Q0 = np.eye(6)
for p, q, theta in [(0,4,0.23),(1,5,-0.67),(2,3,0.49),(0,1,-0.31)]:
    G = np.eye(6)
    c, s = np.cos(theta), np.sin(theta)
    G[p,p] = G[q,q] = c
    G[p,q], G[q,p] = -s, s
    Q0 = Q0 @ G
A = Q0 @ S @ Q0.T
seed = 123
""",
            "call": "compute_basis_sensitive_skew_scalar(A, seed)",
            "gold_call": "_oracle_compute_basis_sensitive_skew_scalar(A, seed)",
        },
        {
            "setup": """import numpy as np
values = np.array([6.0, 2.5, 0.8, 0.09, 0.004])
S = np.block([[np.zeros((5,5)), -np.diag(values)], [np.diag(values), np.zeros((5,5))]])
Q0 = np.eye(10)
for p, q, theta in [(0,6,0.41),(1,8,-0.33),(2,5,0.77),(3,9,-0.58),(4,7,0.19),(0,3,-0.64),(5,8,0.28)]:
    G = np.eye(10)
    c, s = np.cos(theta), np.sin(theta)
    G[p,p] = G[q,q] = c
    G[p,q], G[q,p] = -s, s
    Q0 = Q0 @ G
A = Q0 @ S @ Q0.T
A = 0.5 * (A - A.T)
seed = 4242
""",
            "call": "compute_basis_sensitive_skew_scalar(A, seed)",
            "gold_call": "_oracle_compute_basis_sensitive_skew_scalar(A, seed)",
        },
        {
            "setup": """import numpy as np
values = np.array([6.0, 6.0, 1.2, 0.08])
S = np.block([[np.zeros((4,4)), -np.diag(values)], [np.diag(values), np.zeros((4,4))]])
Q0 = np.eye(8)
for p, q, theta in [(0,5,0.37),(1,6,-0.52),(2,7,0.81),(3,4,-0.28),(0,2,0.44),(5,7,-0.33)]:
    G = np.eye(8)
    c, s = np.cos(theta), np.sin(theta)
    G[p,p] = G[q,q] = c
    G[p,q], G[q,p] = -s, s
    Q0 = Q0 @ G
A = Q0 @ S @ Q0.T
A = 0.5 * (A - A.T)
seed = 271828
""",
            "call": "compute_basis_sensitive_skew_scalar(A, seed)",
            "gold_call": "_oracle_compute_basis_sensitive_skew_scalar(A, seed)",
        },
        {
            "setup": """import numpy as np
sigma = np.diag([3.0, 0.5])
A = np.block([[np.zeros((2,2)), -sigma], [sigma, np.zeros((2,2))]])
seed = 1
""",
            "call": "compute_basis_sensitive_skew_scalar(A, seed)",
            "gold_call": "_oracle_compute_basis_sensitive_skew_scalar(A, seed)",
        },
        {
            "setup": """import numpy as np
sigma = np.diag([3.0, 0.5])
A = np.block([[np.zeros((2,2)), -sigma], [sigma, np.zeros((2,2))]])
seed = 110
candidates = 9
""",
            "call": "compute_basis_sensitive_skew_scalar(A, seed, candidates)",
            "gold_call": "_oracle_compute_basis_sensitive_skew_scalar(A, seed, candidates)",
        },
        {
            "setup": """import numpy as np
values = np.array([5.0, 2.0, 0.4, 0.05])
S = np.block([[np.zeros((4,4)), -np.diag(values)], [np.diag(values), np.zeros((4,4))]])
Q0 = np.eye(8)
for p, q, theta in [(0,5,0.37),(2,7,-0.52),(1,4,0.81),(3,6,-0.28),(0,2,0.44)]:
    G = np.eye(8)
    c, s = np.cos(theta), np.sin(theta)
    G[p,p] = G[q,q] = c
    G[p,q], G[q,p] = -s, s
    Q0 = Q0 @ G
A = Q0 @ S @ Q0.T
seed = 78
candidates = 9
""",
            "call": "compute_basis_sensitive_skew_scalar(A, seed, candidates)",
            "gold_call": "_oracle_compute_basis_sensitive_skew_scalar(A, seed, candidates)",
        },
        {
            "setup": """import numpy as np
A = np.zeros((3, 3))
def run_model():
    try:
        compute_basis_sensitive_skew_scalar(A)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_compute_basis_sensitive_skew_scalar(A)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        {
            "setup": """import numpy as np
A = np.zeros((4, 4))
def run_model():
    try:
        compute_basis_sensitive_skew_scalar(A)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_compute_basis_sensitive_skew_scalar(A)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
    ]
