# Physics-Condensed_Matter_Physics-7

## Background

Projected thermal-density methods provide a Monte Carlo route to low-lying spectra without reconstructing the exponentially large many-body Hamiltonian. Instead of sampling only a scalar partition function, they estimate density and energy matrices inside physically chosen projection subspaces, so a reduced generalized eigenvalue problem yields the accessible thermal energies.
For frustrated magnets, signs and finite statistics can make some projected-density eigenmodes indistinguishable from zero. A reliable analysis therefore has to preserve the sign-restored matrix estimators, use symmetry-informed projection sectors, remove unresolved density directions from both matrices through one common subspace, and quantify whether the resulting intersector gap is stable under resampling. The present task combines these ideas for two symmetry sectors of a periodic dimerized spin lattice.

## Problem

Finite-statistics projected thermal matrices can reveal the singlet-to-triplet gap of a frustrated dimer magnet, but sign fluctuations and unresolved density modes require a symmetry-resolved stability analysis rather than one direct matrix inversion. Compute the additive first-order bootstrap-bias-corrected gap $\Delta_{\mathrm{bc}}$ between the lowest $Q=-1$ and $Q=+1$ projected thermal energies for the deterministic periodic $L=4$ instance below, in units of $J$, using IEEE-754 binary64 and no intermediate rounding. Use the continuous-time insertion/removal balance, sign-restored projected-density and shifted-energy estimators, the oriented unnormalized dimer-product projection states in the two $Q$ sectors, cancellation of the common sampling normalization, the source Hamiltonian shift, and a common zero-mode-free reduction of both projected matrices; the exact source-specific identities are not otherwise supplied.

```python
import numpy as np
L=4; beta=6.0; J=1.0; Jp=0.5
B=24; P=192; Q=8
diag=[]
for family in (0,1):
    for y in range(L):
        for x in range(L):
            if x%2==0 and (x+y)%2==family:
                jx=(x+1)%L if family==0 else (x-1)%L
                diag.append((x+L*y,jx+L*((y+1)%L),x,y))
horiz=[]
for y in range(L):
    for x in range(L):
        if (x+y)%2==0:
            jx=(x+1)%L if x%2==0 else (x-1)%L
            horiz.append((x+L*y,jx+L*y,x,y))
def make_word(code,cover):
    word=0
    for a,(i,j,_,_) in enumerate(cover):
        word |= 1 << (i if ((code>>a)&1) else j)
    return word
state_words=np.array(
    [make_word((73*k+11)%256,diag) for k in range(64)]
    +[make_word((109*k+7)%256,horiz) for k in range(64)],dtype=np.int64)
initial_counts=np.empty((2,B,Q),dtype=np.int64)
locations=np.empty((2,B,P),dtype=np.int64)
moves=np.empty_like(locations)
local_weights=np.empty((2,B,P),dtype=float)
uniforms=np.empty_like(local_weights)
for s in range(2):
    for b in range(B):
        for q in range(Q):
            initial_counts[s,b,q]=7-s+((5*b+3*q+2+4*s)%7)
        for p in range(P):
            locations[s,b,p]=(7*p+3*b+2*s+p//9)%Q
            moves[s,b,p]=1 if ((11*p+7*b+5*s+3)%17)<9 else -1
            local_weights[s,b,p]=0.28+0.06*((13*p+5*b+7*s+4)%29)
            uniforms[s,b,p]=(((97*p+53*b+113*s+31)%1543)+0.5)/1543.0
relative_cutoffs=np.array([0.25,0.50])
n_bootstrap=257; bootstrap_seed=447
min_valid_fraction=0.95; max_log_mad=0.125
```

Site $x+Ly$ is spin up when its word bit is 0 and spin down when it is 1; the diagonal cover is ordered by all family-A anchors and then all family-B anchors in the loop order above, the horizontal cover uses its loop order, the $Q=+1$ coordinate order is the alternating single-site product followed by diagonal-triplet, horizontal-singlet, and horizontal-triplet products, and the $Q=-1$ coordinates place one triplet on each diagonal dimer in cover order with singlets on the others. Reset counts independently for every sector and block, use each selected location's pre-move count, set $d=\mathbf 1[u<A]$ as the acceptance indicator, mutate the selected count immediately only when $d=1$, and after each decision let $T$ be the post-decision total count and $a$ the cumulative accepted count; sector $s=0$ uses all 128 state rows and its first four amplitudes, while $s=1$ uses rows 0 through 63 and the eight one-triplet amplitudes. For modulus $M_s=(128,64)_s$, set $f=(17p+11b+3T+a+13s)\bmod M_s$; for $p<160$ use $(i,\mathcal S)=(f,+1)$, while for $p\ge160$ first set $i=(29p+7b+5T+2a+11+17s)\bmod M_s$, replace equality by $i\leftarrow[i+1+(T+a)\bmod(M_s-1)]\bmod M_s$, and set $\mathcal S=-1$ exactly when $(3b+5p+T+2a+7s)\bmod13\in\{0,3\}$. Average the 192 signed matrix samples into 24 symmetric block matrices per sector, use one common table `np.random.default_rng(447).integers(0,24,size=(257,24))`, and assess rank pairs $(r_+,r_-)$ with $r_+\in(4,3,2)$ and $r_-\in(8,7,\ldots,3)$ ordered by decreasing $r_++r_-$, then decreasing $r_-$, then decreasing $r_+$; a bootstrap replicate is valid only if, in each sector $s$, every retained eigenvalue is strictly positive and satisfies $\lambda_j>c_s\lambda_{\max}(Z_s^{(b)})$, where $c_s$ is `relative_cutoffs[s]` and $\lambda_{\max}(Z_s^{(b)})$ is the largest eigenvalue of that same replicate's sector density matrix, and if its lowest intersector gap $\Delta_b$ is finite and positive. Define $f$ as the valid fraction, $s=1.4826\,\operatorname{median}(|\log\Delta_b-\operatorname{median}(\log\Delta_b)|)$, call a pair stable when $f\ge0.95$ and $s\le0.125$, choose the first stable pair, and return $\Delta_{\mathrm{bc}}=2\Delta_{\mathrm{full}}-\operatorname{mean}(\Delta_b)$ over its valid replicates. Use these reporting checkpoints: the four $Q=+1$ amplitudes and zero-based $Q=-1$ one-triplet coordinate 2 at `state_words[0]`; $C_s=\sum_{b=0}^{23}(b+1)a_{s,b}$, where $a_{s,b}$ is the final accepted-move count in block $b$ of sector $s$; the central $(0,1)$ entries of $Z_+,\widetilde E_+,Z_-,\widetilde E_-$ in that order. Report each sector's retained density eigenvalues in ascending order.

In `<reasoning>`, first retrieve and apply four source-only checks: (i) for the source's $L=4$, $(J,J')=(1,0.5)$ $P_+$ subspace, report the four Table XI overlaps of its nondegenerate $E_6$ eigenstate in the ordered projection-coordinate basis, treating a common overall sign reversal as equivalent; (ii) for the same lattice and coupling, report the four ordered $P_+$ projected thermal energies listed in Table V at $\beta=2$; (iii) in the source's $L=8$ Heisenberg-limit analysis, report the three ordered $P_+$ projected thermal energies listed in Table VI at $\beta=5$ after the unresolved fourth projection state is removed; and (iv) report the two ordered $P_+$ projected thermal energies listed there at $\beta=15$ in the retained two-state subspace. Include the source's parenthesized uncertainties for all three projected-energy rows. Identify the paper's $P_+$ subspace with the task's $Q=+1$ sector where applicable; treat these source-only results as interpretation and convergence checks rather than inputs to the seeded $\beta=6$ calculation; then state the five method identities and report the requested projection-amplitude checkpoints, the two weighted acceptance checksums, four central matrix entries, the complete ordered list of higher-priority zero-support rank pairs, the decisive $(3,6)$ and $(2,6)$ diagnostics, the selected ranks and retained density eigenvalues, the two lowest central energies, the central gap, the bootstrap-mean gap, and the final bias-corrected gap to about 12 significant figures without dumping matrices or per-proposal paths.

Output Format Requirements:
Emit <final_answer> immediately, then <reasoning>. Do not write a long derivation before the tags.
You must emit exactly one finite decimal inside <final_answer>...</final_answer>, even if the value is approximate or you are unsure.
Rules:
- The tags are required. Do not omit them or leave them empty.
- The tagged value must be a finite decimal (examples: 0.4847, 0.22, 1.05). Not NaN, not Inf, not a fraction string, not a vector, not prose.
- Put only that one number between the tags. No units, no words, no extra lines.
Keep <reasoning> short (a few hundred words). Show only the few scalars that determine the final number.
Do not paste the input matrices, full coefficient vectors, per-iteration paths, or per-fold candidate tables.

## Output format

```
## Output format

Scientific reasoning wrapped in <reasoning>...</reasoning> tags. Include enough intermediate calculations to justify the deterministic computation, without turning the response into a general pipeline summary. A single final numeric answer wrapped in <final_answer>...</final_answer> tags.
```

## Your task

Implement **all 8 functions** below into a single file at `/app/solution.py`. Requirements:

- Use the EXACT function name and signature shown for each step.
- Define every function at module scope (importable).
- Do NOT include tests, example usage, prints, or a `__main__` block.
- Available libraries: numpy, scipy, sympy, pandas, scikit-learn, networkx.

Start your file with these imports:

```python
import numpy as np
```

### Step 1

compute_link_acceptance

Goal
----
Compute the acceptance probability for one continuous-time diagonal-link insertion or removal. Use the pre-move local count and return a native float in the closed interval [0, 1].

```python
def compute_link_acceptance(
    beta: float,
    local_weight: float,
    local_count: int,
    move: int,
) -> float:
    """Compute a continuous-time link-move acceptance probability.

    Parameters
    ----------
    beta : float
        Strictly positive inverse temperature.
    local_weight : float
        Strictly positive magnitude of the selected diagonal matrix element.
    local_count : int
        Nonnegative selected-location count before the proposal.
    move : int
        ``+1`` for insertion or ``-1`` for removal.

    Returns
    -------
    acceptance : float
        Acceptance probability in ``[0, 1]``.

    Raises
    ------
    ValueError
        If an input is non-finite or outside its domain, or if removal is
        requested from zero count.
    """
    return acceptance
```

### Step 2

apply_link_move

Goal
----
Apply one supplied strict Metropolis decision to an independent copy of a local link-count vector. The returned numeric array contains the updated counts followed by the 0/1 decision.

```python
import numpy as np

def apply_link_move(
    counts: np.ndarray,
    location: int,
    move: int,
    acceptance: float,
    uniform: float,
) -> np.ndarray:
    """Apply one accepted or rejected link-count mutation.

    Parameters
    ----------
    counts : np.ndarray
        Nonempty one-dimensional nonnegative integer count vector.
    location : int
        Zero-based selected location.
    move : int
        ``+1`` for insertion or ``-1`` for removal.
    acceptance : float
        Acceptance probability in ``[0, 1]``.
    uniform : float
        Supplied deviate in ``[0, 1]``; acceptance is strictly ``uniform < acceptance``.

    Returns
    -------
    packed_result : np.ndarray
        Integer vector of length ``counts.size + 1``. The first entries are an
        independent updated count vector and the last is the decision flag.

    Raises
    ------
    ValueError
        If the count vector or a scalar argument is invalid, or removal is
        proposed at a zero-count location.
    """
    return packed_result
```

### Step 3

compute_ssm_projection_amplitudes

Goal
----
Construct overlaps of spin-$z$ basis words with the two projection sectors used for the periodic Shastry–Sutherland lattice. Bit $x+Ly$ is zero for spin up and one for spin down. The visible helper fixes the oriented diagonal and horizontal dimer covers.

```python
import numpy as np

def compute_ssm_projection_amplitudes(
    state_words: np.ndarray,
    lattice_size: int,
) -> np.ndarray:
    """Compute the two Shastry-Sutherland projection-sector amplitudes.

    Parameters
    ----------
    state_words : np.ndarray
        Nonempty one-dimensional nonnegative integer array. Bit
        ``x + lattice_size*y`` is zero for spin up and one for spin down.
    lattice_size : int
        Even periodic linear size in ``[2, 6]``.

    Returns
    -------
    amplitudes : np.ndarray
        Float array with one row per state word. The first four columns are,
        in order, the alternating single-site Q=+1 product, the product of
        diagonal triplets, the product of horizontal singlets, and the
        product of horizontal triplets. The remaining columns contain the
        Q=-1 states with one diagonal triplet, in diagonal-cover order, and
        singlets on every other diagonal dimer.
        The alternating single-site product is unnormalized: its local state
        is (|up> + |down>) for x+y even and (|down> - |up>) for x+y odd.
        Its amplitude contributes a factor -1 at every odd-sublattice site
        whose word bit is 0, and +1 otherwise.

    Raises
    ------
    ValueError
        If the lattice size or encoded words are invalid.
    """
    return amplitudes
```

### Step 4

compute_pdms_contribution

Goal
----
Construct the signed normalized projected-density and shifted-energy contribution from one post-decision configuration. The two matrix contributions share the same oriented boundary-amplitude outer product.

```python
import numpy as np

def compute_pdms_contribution(
    sign: int,
    final_amplitudes: np.ndarray,
    initial_amplitudes: np.ndarray,
    total_links: int,
    beta: float,
) -> np.ndarray:
    """Compute one signed projected-matrix sample.

    Parameters
    ----------
    sign : int
        Path sign, either ``-1`` or ``+1``.
    final_amplitudes, initial_amplitudes : np.ndarray
        Matched nonempty one-dimensional finite projection-amplitude vectors.
    total_links : int
        Nonnegative total number of links after the proposal decision.
    beta : float
        Strictly positive finite inverse temperature.

    Returns
    -------
    contributions : np.ndarray
        Float array of shape ``(2, d, d)``. Entry 0 is the density contribution;
        entry 1 is the shifted-energy contribution.

    Raises
    ------
    ValueError
        If a sign, vector, count, or inverse temperature is invalid.
    """
    return contributions
```

### Step 5

average_sector_blocks

Goal
----
Average equal-weight signed projected-matrix samples within every independent block and take the symmetric part of each block mean while preserving both symmetry sectors and both matrix channels.

```python
import numpy as np

def average_sector_blocks(sample_terms: np.ndarray) -> np.ndarray:
    """Average and symmetrize two-sector matrix samples by block.

    Parameters
    ----------
    sample_terms : np.ndarray
        Finite float array of shape ``(2, 2, blocks, samples, d, d)``. The
        first axis labels the two sectors and the second labels density then
        shifted energy. At least three blocks and one sample are required.

    Returns
    -------
    block_matrices : np.ndarray
        Float array of shape ``(2, 2, blocks, d, d)`` containing symmetric
        equal-weight block means.

    Raises
    ------
    ValueError
        If the shape is invalid or any entry is non-finite.
    """
    return block_matrices
```

### Step 6

compute_bootstrap_rank_pair_scan

Goal
----
Evaluate an ordered list of retained-rank pairs with one common nonparametric block-bootstrap table. Each row records support, logarithmic robust spread, the full-record intersector gap, its valid-bootstrap mean, and the additive first-order bootstrap bias correction.

```python
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
    """Compute a shared-bootstrap stability scan over sector-rank pairs.

    Parameters
    ----------
    block_matrices : np.ndarray
        Finite array of shape ``(2, 2, blocks, d, d)``. Sector 0 occupies
        the leading ``plus_dimension`` coordinates; sector 1 occupies all
        ``d`` coordinates. Matrix channel 0 is density and channel 1 is
        shifted energy. There must be at least three blocks and d >= 3.
    energy_shift : float
        Finite additive energy shift applied to both generalized spectra.
    plus_dimension : int
        Integer dimension of the leading Q=+1 matrix block, in [2, d].
    candidate_pairs : np.ndarray
        Nonempty integer array of shape ``(candidates, 2)`` containing
        ``(r_plus, r_minus)`` in the exact order to be assessed, with no
        duplicate pair. Require 2 <= r_plus <= plus_dimension and
        2 <= r_minus <= d.
    relative_cutoffs : np.ndarray
        Two finite values in ``[0, 1)`` for Q=+1 and Q=-1 bootstrap spectra.
        A retained bootstrap mode must obey both ``lambda > 0`` and
        ``lambda > cutoff*lambda_max``. For the full-record central spectra,
        retained modes must be strictly positive; relative cutoffs do not apply.
    n_bootstrap : int
        Number of common block-bootstrap replicates, at least five.
    bootstrap_seed : int
        Nonnegative seed for ``np.random.default_rng``.

    Returns
    -------
    scan : np.ndarray
        Float array of shape ``(candidates, 7)`` with columns
        ``r_plus, r_minus, valid_fraction, log_mad, central_gap,
        bootstrap_mean, bias_corrected_gap``. For every nonempty set of
        valid bootstrap gaps, including one or two gaps, compute log_mad
        with the stated median formula. If no replicate is valid, log_mad,
        bootstrap_mean, and bias_corrected_gap use the largest finite
        binary64 value as a sentinel; valid_fraction is zero and central_gap
        retains its full-record value.

    Raises
    ------
    ValueError
        If an input is invalid, a candidate is duplicated or out of range,
        or a central candidate spectrum does not yield a positive finite
        intersector gap.
    """
    return scan
```

### Step 7

select_stable_bias_corrected_gap

Goal
----
Select the first ordered rank pair whose bootstrap support and robust logarithmic spread satisfy the inclusive stability thresholds, then return its positive additive bootstrap-bias-corrected intersector gap.

```python
import numpy as np

def select_stable_bias_corrected_gap(
    rank_pair_scan: np.ndarray,
    min_valid_fraction: float,
    max_log_mad: float,
) -> float:
    """Return the bias-corrected gap from the first stable scan row.

    Parameters
    ----------
    rank_pair_scan : np.ndarray
        Finite float array of shape ``(candidates, 7)`` with the column order
        documented by ``compute_bootstrap_rank_pair_scan``. Candidate pairs
        must be unique positive integer-valued labels.
    min_valid_fraction : float
        Inclusive support threshold in ``(0, 1]``.
    max_log_mad : float
        Inclusive finite nonnegative robust-spread threshold.

    Returns
    -------
    gap : float
        Positive finite bias-corrected intersector gap from the first stable
        candidate row.

    Raises
    ------
    ValueError
        If the scan or thresholds are invalid or no row is stable.
    """
    return gap
```

### Step 8

run_ssm_bootstrap_gap

Goal
----
Run the complete two-sector projected-density benchmark on a periodic Shastry-Sutherland lattice. Generate state-dependent sample pairs after each strict link decision, construct the Q=+1 and Q=-1 block matrices, scan ordered retained-rank pairs with a common bootstrap table, and return the first stable pair's additive bootstrap-bias-corrected intersector gap.

```python
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
```
