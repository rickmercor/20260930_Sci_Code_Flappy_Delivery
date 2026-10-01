"""
Run the complete two-sector projected-density benchmark on a periodic Shastry-Sutherland lattice. Generate state-dependent sample pairs after each strict link decision, construct the Q=+1 and Q=-1 block matrices, scan ordered retained-rank pairs with a common bootstrap table, and return the first stable pair's additive bootstrap-bias-corrected intersector gap.

For sector 0, sample indices range over all supplied state words and use the first four projection coordinates. For sector 1, indices range over the first half of the state words and use the one-triplet coordinates. After each proposal, let T be the post-decision total count and a the cumulative accepted count. Set f=(17*p+11*b+3*T+a+13*s) mod M_s. For p<160 use i=f and sign +1; otherwise use i=(29*p+7*b+5*T+2*a+11+17*s) mod M_s, replace equality by i=(i+1+((T+a) mod (M_s-1))) mod M_s, and set the sign negative exactly when (3*b+5*p+T+2*a+7*s) mod 13 is 0 or 3.

Use the Hamiltonian shift `$E0=(J_prime/2+J/8)*L**2$`. Candidate ranks are `$r_plus in (4,3,2)$` and `$r_minus=L**2//2,...,3$`; order pairs by decreasing `$r_plus+r_minus$`, then decreasing `$r_minus$`, then decreasing `$r_plus$`. A pair is stable when its valid fraction and log-MAD meet the supplied inclusive thresholds, and selection stops at the first such pair.

The public implementation must call the seven preceding public functions—`$compute_link_acceptance$`, `$apply_link_move$`, `$compute_ssm_projection_amplitudes$`, `$compute_pdms_contribution$`, `$average_sector_blocks$`, `$compute_bootstrap_rank_pair_scan$`, and `$select_stable_bias_corrected_gap$`—rather than reimplementing their operations locally.

Returns
-------
float, the positive bootstrap-bias-corrected Q=-1 minus Q=+1 projected gap
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

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
    """Run the complete deterministic two-sector gap extraction.

    Parameters
    ----------
    beta : float
        Strictly positive inverse temperature.
    coupling_j : float
        Strictly positive diagonal-dimer coupling.
    coupling_j_prime : float
        Finite nonnegative nearest-neighbor coupling.
    initial_counts : np.ndarray
        Nonnegative integer array of shape ``(2, blocks, locations)``.
    locations, moves : np.ndarray
        Matched integer arrays of shape ``(2, blocks, samples)`` with at
        least 161 samples; moves are ``+1`` or ``-1``.
    local_weights, uniforms : np.ndarray
        Matched finite float arrays; weights are positive and uniforms lie
        in ``[0, 1]``.
    state_words : np.ndarray
        Even-length encoded state array. Sector 0 uses all rows, and sector
        1 uses the first half.
    lattice_size : int
        Even periodic linear size in ``[4, 6]``.
    relative_cutoffs : np.ndarray
        Two strict relative density-mode cutoffs for Q=+1 and Q=-1.
    n_bootstrap, bootstrap_seed : int
        Bootstrap replicate count and nonnegative RNG seed.
    min_valid_fraction, max_log_mad : float
        Inclusive pair-stability thresholds.

    Returns
    -------
    gap : float
        Positive finite additive bootstrap-bias-corrected intersector gap.

    Raises
    ------
    ValueError
        If an input is invalid, a delegated step rejects its input, or no
        ordered rank pair is stable.
    """
    return gap

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import math
from numbers import Integral, Real

import numpy as np


def _oracle_run_ssm_bootstrap_gap(
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

    amplitudes = _oracle_compute_ssm_projection_amplitudes(words, L)
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
                acceptance = _oracle_compute_link_acceptance(
                    beta,
                    float(weights[sector, block, proposal]),
                    int(current[location]),
                    int(mov[sector, block, proposal]),
                )
                packed = _oracle_apply_link_move(
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
                contribution = _oracle_compute_pdms_contribution(
                    sign,
                    final_vector,
                    initial_vector,
                    total_links,
                    beta,
                )
                terms[sector, :, block, proposal, :dimension, :dimension] = contribution

    block_matrices = _oracle_average_sector_blocks(terms)
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
    scan = _oracle_compute_bootstrap_rank_pair_scan(
        block_matrices,
        energy_shift,
        4,
        candidate_pairs,
        relative_cutoffs,
        n_bootstrap,
        bootstrap_seed,
    )
    return _oracle_select_stable_bias_corrected_gap(
        scan,
        min_valid_fraction,
        max_log_mad,
    )

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    base = """import numpy as np
beta=6.0
coupling_j=1.0
coupling_j_prime=0.5
L=4
B=24
P=192
Q=8
# Paper-convention dimer covers used only to generate deterministic state words.
diag=[]
for family in (0,1):
 for y in range(L):
  for x in range(L):
   if x%2==0 and (x+y)%2==family:
    jx=(x+1)%L if family==0 else (x-1)%L
    diag.append((x+L*y,jx+L*((y+1)%L)))
horiz=[]
for y in range(L):
 for x in range(L):
  if (x+y)%2==0:
   jx=(x+1)%L if x%2==0 else (x-1)%L
   horiz.append((x+L*y,jx+L*y))
def make_word(code,cover):
 word=0
 for a,(i,j) in enumerate(cover):
  word |= 1 << (i if ((code>>a)&1) else j)
 return word
state_words=np.array([make_word((73*k+11)%256,diag) for k in range(64)]+[make_word((109*k+7)%256,horiz) for k in range(64)],dtype=np.int64)
initial_counts=np.empty((2,B,Q),dtype=int)
locations=np.empty((2,B,P),dtype=int)
moves=np.empty_like(locations)
local_weights=np.empty((2,B,P),dtype=float)
uniforms=np.empty_like(local_weights)
for s in range(2):
 base_count=7-s
 for b in range(B):
  for q in range(Q):
   initial_counts[s,b,q]=base_count+((5*b+3*q+2+4*s)%7)
  for p in range(P):
   locations[s,b,p]=(7*p+3*b+2*s+(p//9))%Q
   moves[s,b,p]=1 if ((11*p+7*b+5*s+3)%17)<9 else -1
   local_weights[s,b,p]=0.28+0.06*((13*p+5*b+7*s+4)%29)
   uniforms[s,b,p]=(((97*p+53*b+113*s+31)%1543)+0.5)/1543.0
relative_cutoffs=np.array([0.25,0.50],dtype=float)
n_bootstrap=257
bootstrap_seed=447
min_valid_fraction=0.95
max_log_mad=0.125"""
    call = "run_ssm_bootstrap_gap(beta,coupling_j,coupling_j_prime,initial_counts.copy(),locations.copy(),moves.copy(),local_weights.copy(),uniforms.copy(),state_words.copy(),L,relative_cutoffs.copy(),n_bootstrap,bootstrap_seed,min_valid_fraction,max_log_mad)"
    gold = "_oracle_run_ssm_bootstrap_gap(beta,coupling_j,coupling_j_prime,initial_counts.copy(),locations.copy(),moves.copy(),local_weights.copy(),uniforms.copy(),state_words.copy(),L,relative_cutoffs.copy(),n_bootstrap,bootstrap_seed,min_valid_fraction,max_log_mad)"
    bad_words = base + """
state_words=state_words[:-1]
def run_model():
 try:
  """ + call + """
  return 0
 except ValueError:
  return 1
 except Exception:
  return 2
def run_gold():
 try:
  """ + gold + """
  return 0
 except ValueError:
  return 1
 except Exception:
  return 2"""
    no_stable = base + """
max_log_mad=0.01
def run_model():
 try:
  """ + call + """
  return 0
 except ValueError:
  return 1
 except Exception:
  return 2
def run_gold():
 try:
  """ + gold + """
  return 0
 except ValueError:
  return 1
 except Exception:
  return 2"""
    bad_shape = base + """
locations=locations[:,:,:160]
moves=moves[:,:,:160]
local_weights=local_weights[:,:,:160]
uniforms=uniforms[:,:,:160]
def run_model():
 try:
  """ + call + """
  return 0
 except ValueError:
  return 1
 except Exception:
  return 2
def run_gold():
 try:
  """ + gold + """
  return 0
 except ValueError:
  return 1
 except Exception:
  return 2"""
    return [
        {"setup": base, "call": call, "gold_call": gold},
        {"setup": base + "\nmax_log_mad=0.13", "call": call, "gold_call": gold},
        {"setup": base + "\nuniforms[0,0,0]=1.0", "call": call, "gold_call": gold},
        {"setup": bad_words, "call": "run_model()", "gold_call": "run_gold()"},
        {"setup": no_stable, "call": "run_model()", "gold_call": "run_gold()"},
        {"setup": bad_shape, "call": "run_model()", "gold_call": "run_gold()"},
    ]
