# Biology-Biochemistry-36

## Background

Random-mutagenesis libraries are usually summarised by a low-order expansion of a sequence score in the mutation rate, on the assumption that a few terms describe the library across the rates actually used at the bench. When the score is structure-aware it is a partition function summing many competing parses, and the library response of its logarithm is nonlinear, so the rate at which a truncated description stops being quantitatively adequate is not evident from the expansion itself.

This task fixes one library and one scoring grammar and asks for that rate: the mutation rate at which a fourth-order Taylor description of the log response departs from the exact response by a fixed relative margin.

## Problem

I am designing an independent random-mutagenesis library around the RNA `GGACAUCGAGUCCUAG` (5′ to 3′): at mutation rate μ, each position retains its reference nucleotide with probability 1 − μ and otherwise becomes each of the other three nucleotides with probability μ/3. Score a sequence by the partition function Z of an ambiguous one-nonterminal grammar whose rules are `S -> a S b`, `S -> a S`, `S -> S a`, `S -> S S`, and `S -> epsilon`, with respective rule weights `t_P`, `t_L`, `t_R`, `t_B`, and `t_E`; the pair rule may be used only when at least three nucleotides lie between its emitted endpoints, `S -> S S` splits into two nonempty subsequences, `S -> epsilon` is the only derivation of an empty subsequence, and distinct parse trees count separately. The local factors are `t_P e_P(a,b)`, `t_L e_L(a)`, `t_R e_R(a)`, `t_B`, and `t_E`, with rule weights `(0.32, 0.24, 0.18, 0.14, 0.70)`, left emissions `e_L(A,C,G,U) = (0.33, 0.21, 0.26, 0.20)`, right emissions `e_R(A,C,G,U) = (0.19, 0.30, 0.22, 0.29)`, and nonzero ordered pair emissions `e_P(A,U) = 1.15`, `e_P(U,A) = 0.85`, `e_P(G,C) = 1.75`, `e_P(C,G) = 1.40`, `e_P(G,U) = 0.50`, and `e_P(U,G) = 0.65`. Whenever a pair emitting positions `i,j` directly encloses a pair emitting `i+1,j−1`, multiply by an additional stacking factor 2.2, 1.5, 1.3, or 0.8 according as the ordered outer/inner pair classes are Watson–Crick/Watson–Crick, Watson–Crick/G–U, G–U/Watson–Crick, or G–U/G–U; no additional factor is used if either pair is outside those classes. Define `A(μ) = E_μ[Z] / Z(reference)`, and let `G_4(μ)` be the degree-four Taylor polynomial about zero of the nonlinear observable `log A(μ)`. Using the published exact finite-substitution framework to obtain the library mean without enumerating its 4^16 sequences or sampling, report to ten significant figures the unique mutation rate μ_star in `[0.045, 0.070]` satisfying

\[
\frac{\left|\log A(\mu_\star)-G_4(\mu_\star)\right|}{\left|\log A(\mu_\star)\right|}=0.30.
\]

In `<reasoning>`, give `Z(reference)`, the coefficients through order four of `A(μ)` and `log A(μ)`, the relative remainder at both bracket endpoints, and a concise source-grounded justification of the modelling choices behind the library mean and the reported series.

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

01_compute_stacked_inside_weights

Goal
----
Compute the total and pair-class-resolved inside weights of every subsequence of an RNA under a five-rule stochastic grammar with a minimum enclosed loop and nearest-neighbour stacking.

```python
def compute_stacked_inside_weights(
    sequence: "np.ndarray",
    rule_weights: "np.ndarray",
    left_emission: "np.ndarray",
    right_emission: "np.ndarray",
    pair_emission: "np.ndarray",
    stacking_factors: "np.ndarray",
    min_loop: int,
) -> "np.ndarray":
    """Return total and pair-class-resolved inside weights of all subsequences.

    Nucleotides are coded A = 0, C = 1, G = 2, U = 3 and positions are
    1-based. ``rule_weights`` is ``(t_P, t_L, t_R, t_B, t_E)``. The grammar
    has one nonterminal ``S`` and five rules: ``S -> a S b`` emits paired
    endpoints with factor ``t_P * pair_emission[a, b]`` and is allowed only
    when at least ``min_loop`` nucleotides lie between them; ``S -> a S`` and
    ``S -> S a`` emit one unpaired endpoint with factors
    ``t_L * left_emission[a]`` and ``t_R * right_emission[a]``;
    ``S -> S S`` splits into two nonempty strings with factor ``t_B``; and
    ``S -> epsilon`` is the sole empty derivation with factor ``t_E``.
    Distinct parse trees count separately.

    Pair-topped states have three classes: neutral/other, Watson–Crick (A-U,
    U-A, G-C, C-G), and G–U wobble (G-U, U-G). A neutral outer or inner pair
    contributes stack factor 1. Otherwise use ``stacking_factors[k]``, where
    ``k = 0, 1, 2, 3`` for ordered outer/inner classes WC/WC, WC/GU, GU/WC,
    and GU/GU. Only a directly pair-topped interior receives this factor.

    Channel 0 of the result is the total inside weight. Channels 1, 2 and 3
    are respectively the neutral, Watson–Crick and G–U pair-topped weights.
    Entry ``[0, i, j]`` covers ``x_i ... x_j`` for ``1 <= i <= L + 1`` and
    ``i - 1 <= j <= L``; ``j = i - 1`` is empty. Every other entry is zero,
    and ``result[0, 1, L]`` is the partition function. Consequently, for a
    nonempty span, channel 0 includes the sum of channels 1 through 3 plus
    all non-pair-rooted derivations.

    Parameters
    ----------
    sequence : np.ndarray
        Integer nucleotide codes, shape ``(L,)`` with ``L >= 1``.
    rule_weights : np.ndarray
        Finite nonnegative ``(t_P, t_L, t_R, t_B, t_E)``, shape ``(5,)``.
    left_emission, right_emission : np.ndarray
        Finite nonnegative factors indexed by nucleotide, shape ``(4,)``.
    pair_emission : np.ndarray
        Finite nonnegative factors indexed ``[5' nucleotide, 3' nucleotide]``,
        shape ``(4, 4)``; noncanonical entries may be nonzero.
    stacking_factors : np.ndarray
        Finite nonnegative stacking factors, shape ``(4,)``.
    min_loop : int
        Nonnegative minimum number of nucleotides enclosed by a pair.

    Returns
    -------
    np.ndarray
        Float array with shape ``(4, L + 2, L + 1)``.

    Raises
    ------
    ValueError
        If ``sequence`` is not a nonempty one-dimensional array of integer
        codes in ``0..3``, if a factor array has the wrong shape or holds a
        negative or non-finite value, or if ``min_loop`` is not a nonnegative
        integer (booleans are rejected).
    """
    return weights
```

### Step 2

02_build_mutation_profile_polynomials

Goal
----
Lift an independent RNA substitution library into a multivariate affine probability profile whose variables label disjoint position groups.

```python
def build_mutation_profile_polynomials(
    sequence: "np.ndarray",
    mutation_kernel: "np.ndarray",
    position_groups: "np.ndarray",
) -> "np.ndarray":
    """Return grouped multivariate nucleotide-probability coefficients.

    Nucleotides are coded A = 0, C = 1, G = 2 and U = 3. Row ``x`` of
    ``mutation_kernel`` is the conditional replacement distribution when a
    reference nucleotide ``x`` mutates; its diagonal is zero and each row
    sums to one. ``position_groups[i]`` assigns position ``i`` to one of
    ``G`` mutation variables. Group labels must be contiguous ``0..G-1``,
    every label must occur, and ``1 <= G <= 4``.

    With ``g = position_groups[i]``, site ``i`` has nucleotide ``a`` with
    probability

    ``P[i,a](mu_0,...,mu_{G-1}) = (1-mu_g) 1[a=sequence[i]]
                                      + mu_g mutation_kernel[sequence[i],a]``.

    Return the coefficient tensor in increasing degree along every mutation
    variable. Its shape is ``(L, 4) + (2,) * G``. For each site, only the
    all-zero coefficient index and the index with degree one on its assigned
    group axis may be nonzero. Setting every ``mu_g = mu`` recovers the
    original independent one-rate mutation library.

    Parameters
    ----------
    sequence : np.ndarray
        Nonempty one-dimensional integer array with entries in ``0..3``.
    mutation_kernel : np.ndarray
        Finite nonnegative row-stochastic array of shape ``(4, 4)`` with a
        zero diagonal.
    position_groups : np.ndarray
        One-dimensional integer group labels of shape ``(L,)`` satisfying
        the contiguous-label conditions above; booleans are rejected.

    Returns
    -------
    np.ndarray
        Float coefficient tensor of shape ``(L, 4) + (2,) * G``.

    Raises
    ------
    ValueError
        If the sequence, mutation kernel or position-group assignment violates
        the conditions above.
    """
    return profile
```

### Step 3

03_compute_profile_partition_polynomial

Goal
----
Compute the exact multivariate mutation-rate polynomial of a stacked RNA grammar averaged over a grouped independent substitution profile.

```python
def compute_profile_partition_polynomial(
    profile_polynomials: "np.ndarray",
    position_groups: "np.ndarray",
    rule_weights: "np.ndarray",
    left_emission: "np.ndarray",
    right_emission: "np.ndarray",
    pair_emission: "np.ndarray",
    stacking_factors: "np.ndarray",
    min_loop: int,
) -> "np.ndarray":
    """Return the exact grouped profile-mean partition polynomial.

    For ``G`` contiguous position groups, ``profile_polynomials`` has shape
    ``(L, 4) + (2,) * G`` and follows the grouped affine contract of
    ``build_mutation_profile_polynomials``. Site ``i`` may depend only on the
    variable selected by ``position_groups[i]``: only its all-zero coefficient
    and its degree-one coefficient on that group axis may be nonzero. Its four
    nucleotide polynomials sum identically to one and are nonnegative at both
    endpoints of their active variable.

    The grammar and stack classes are those of
    ``compute_stacked_inside_weights``. A pair may enclose at least
    ``min_loop`` nucleotides, and a stack factor applies only when the interior
    derivation is directly pair-topped. Since an inner pair's nucleotide
    identities determine both its emission and the adjacent stack, those
    shared endpoint probabilities must be combined jointly. Retaining both
    endpoint identities is exact; an algebraically equivalent sufficient
    context is also valid.

    Polynomial products are multidimensional convolutions. If group ``g``
    contains ``n_g`` positions, return a coefficient grid of shape
    ``(n_0 + 1, ..., n_{G-1} + 1)``. Entry ``r`` is the coefficient of
    ``prod_g mu_g**r_g``. No coefficient may be discarded within those bounds.

    Parameters
    ----------
    profile_polynomials : np.ndarray
        Finite grouped coefficients with shape ``(L, 4) + (2,) * G``, where
        ``L >= 1`` and ``1 <= G <= 4``.
    position_groups : np.ndarray
        Contiguous integer labels ``0..G-1`` of shape ``(L,)``, with every
        label present.
    rule_weights : np.ndarray
        Finite nonnegative ``(t_P, t_L, t_R, t_B, t_E)``, shape ``(5,)``.
    left_emission, right_emission : np.ndarray
        Finite nonnegative factors indexed by nucleotide, shape ``(4,)``.
    pair_emission : np.ndarray
        Finite nonnegative factors indexed ``[5' nucleotide, 3' nucleotide]``,
        shape ``(4, 4)``.
    stacking_factors : np.ndarray
        Finite nonnegative values for WC/WC, WC/GU, GU/WC and GU/GU,
        shape ``(4,)``; a neutral pair on either side gives factor one.
    min_loop : int
        Nonnegative minimum number of nucleotides enclosed by a pair.

    Returns
    -------
    np.ndarray
        Multivariate float coefficient grid with shape determined by the group
        population counts.

    Raises
    ------
    ValueError
        If an input violates any condition above.
    """
    return coefficients
```

### Step 4

04_normalize_order_coefficients

Goal
----
Restrict a grouped mutation polynomial to a common mutation rate and normalize it by the independently computed reference partition function.

```python
def normalize_order_coefficients(
    fixed_inside_weights: "np.ndarray",
    library_coefficients: "np.ndarray",
    relative_tolerance: float = 1e-10,
) -> "np.ndarray":
    """Diagonalize grouped coefficients and normalize by the reference value.

    ``fixed_inside_weights`` is the class-resolved result of
    ``compute_stacked_inside_weights`` for a length-``L`` reference; its
    partition function is entry ``[0, 1, L]``. ``library_coefficients`` is
    the one- to four-dimensional grid returned by
    ``compute_profile_partition_polynomial``. If its shape is
    ``(n_0 + 1, ..., n_{G-1} + 1)``, require ``sum_g n_g = L``.

    Restrict the grouped polynomial ``F(mu_0,...,mu_{G-1})`` to the diagonal
    ``mu_0 = ... = mu_{G-1} = mu``. Thus every grid entry at multi-index
    ``(r_0,...,r_{G-1})`` is accumulated into univariate coefficient
    ``r_0 + ... + r_{G-1}``. Before normalization, require the all-zero grid
    entry to agree with the fixed partition function within
    ``relative_tolerance``.

    Parameters
    ----------
    fixed_inside_weights : np.ndarray
        Finite array of shape ``(4, L + 2, L + 1)`` with ``L >= 1``.
    library_coefficients : np.ndarray
        Finite coefficient grid with one through four nonempty axes and total
        degree capacity ``sum(shape[g] - 1) = L``.
    relative_tolerance : float
        Finite nonnegative relative tolerance for constant-term agreement.

    Returns
    -------
    np.ndarray
        Normalized diagonal coefficients of shape ``(L + 1,)`` in increasing
        total substitution order.

    Raises
    ------
    ValueError
        If shapes or values are invalid, the reference partition function is
        not positive, or the two constant terms do not agree.
    """
    return normalized
```

### Step 5

05_compute_log_series_coefficients

Goal
----
Transform an exact normalized partition polynomial into the Taylor coefficients of its log observable.

```python
def compute_log_series_coefficients(
    normalized_coefficients: "np.ndarray",
    max_order: int,
) -> "np.ndarray":
    """Return the Taylor coefficients of ``log(A(mu))`` through ``max_order``.

    ``normalized_coefficients`` stores the increasing-order coefficients of
    a finite polynomial ``A(mu)`` with a strictly positive constant term.
    Return ``g[0:max_order+1]`` such that the formal power series of
    ``log(A(mu))`` is ``sum(g[r] * mu**r)`` through the requested order.
    The requested order may not exceed the supplied polynomial degree.

    Parameters
    ----------
    normalized_coefficients : np.ndarray
        Finite one-dimensional coefficient vector with positive first entry
        and at least two elements.
    max_order : int
        Integer in ``1..len(normalized_coefficients)-1``; booleans are
        rejected.

    Returns
    -------
    np.ndarray
        Float vector of shape ``(max_order + 1,)`` in increasing power order.

    Raises
    ------
    ValueError
        If the coefficient vector or order violates the conditions above.
    """
    return log_coefficients
```

### Step 6

06_evaluate_log_remainder_share

Goal
----
Evaluate how much of a log partition-function change is omitted by a finite substitution-order series.

```python
def evaluate_log_remainder_share(
    normalized_coefficients: "np.ndarray",
    log_coefficients: "np.ndarray",
    mutation_rate: float,
) -> float:
    """Return the relative absolute remainder of a truncated log series.

    Let ``A(mu)`` be the polynomial stored in increasing power order by
    ``normalized_coefficients`` and let ``G_k(mu)`` be the polynomial stored
    by ``log_coefficients``.  The constant terms must represent a normalized
    observable: ``A(0) = 1`` and ``G_k(0) = 0``.  Return

    ``abs(log(A(mu)) - G_k(mu)) / abs(log(A(mu)))``.

    Parameters
    ----------
    normalized_coefficients : np.ndarray
        Finite one-dimensional vector with at least two entries and constant
        coefficient one to absolute tolerance ``1e-10``.
    log_coefficients : np.ndarray
        Finite one-dimensional vector with between two and
        ``len(normalized_coefficients)`` entries and zero constant coefficient
        to absolute tolerance ``1e-10``.
    mutation_rate : float
        Finite scalar in ``[0, 1]``.

    Returns
    -------
    float
        Nonnegative relative absolute remainder.

    Raises
    ------
    ValueError
        If an input is invalid, ``A(mu)`` is not positive, or its logarithm
        is zero at the requested rate.
    """
    return share
```

### Step 7

07_locate_log_remainder_threshold

Goal
----
Locate a mutation rate at which omitted log-partition substitution orders reach a prescribed share.

```python
def locate_log_remainder_threshold(
    normalized_coefficients: "np.ndarray",
    log_coefficients: "np.ndarray",
    target_share: float,
    lower_rate: float,
    upper_rate: float,
    rate_tolerance: float = 1e-12,
    max_iterations: int = 200,
) -> float:
    """Return the bracketed crossing of a relative log-remainder target.

    Use ``evaluate_log_remainder_share`` to define
    ``f(mu) = remainder_share(mu) - target_share``.  The caller guarantees
    that the closed interval contains exactly one crossing; its endpoint
    values must have opposite signs or one must be exactly zero.  Use a
    bracketing method and stop once the bracket width is no larger than
    ``rate_tolerance``.  Return the midpoint of the final bracket, except
    that an exact endpoint crossing is returned directly.

    Parameters
    ----------
    normalized_coefficients, log_coefficients : np.ndarray
        Inputs accepted by ``evaluate_log_remainder_share``.
    target_share : float
        Finite nonnegative target.
    lower_rate, upper_rate : float
        Finite rates satisfying ``0 < lower_rate < upper_rate <= 1``.
    rate_tolerance : float
        Finite positive absolute bracket-width tolerance.
    max_iterations : int
        Positive integer iteration cap; booleans are rejected.

    Returns
    -------
    float
        Mutation rate of the unique bracketed crossing.

    Raises
    ------
    ValueError
        If scalar controls are invalid, endpoint evaluation fails, the target
        is not bracketed, or the iteration cap is reached first.
    """
    return mutation_rate
```

### Step 8

08_run_log_nonadditivity_threshold

Goal
----
Compose class-resolved fixed-sequence, grouped profile-polynomial and nonlinear-series calculations into a mutation-rate design threshold.

```python
def run_log_nonadditivity_threshold(
    sequence: "np.ndarray | None" = None,
    mutation_kernel: "np.ndarray | None" = None,
    rule_weights: "np.ndarray | None" = None,
    left_emission: "np.ndarray | None" = None,
    right_emission: "np.ndarray | None" = None,
    pair_emission: "np.ndarray | None" = None,
    stacking_factors: "np.ndarray | None" = None,
    min_loop: "int | None" = None,
    log_order: "int | None" = None,
    target_share: "float | None" = None,
    lower_rate: "float | None" = None,
    upper_rate: "float | None" = None,
    rate_tolerance: float = 1e-12,
) -> float:
    """Return the mutation rate at the requested log-remainder threshold.

    Group valid input positions by their reference nucleotide code, with
    contiguous group labels following the sorted observed codes, and compose
    in order,
    ``build_mutation_profile_polynomials``,
    ``compute_stacked_inside_weights``,
    ``compute_profile_partition_polynomial``,
    ``normalize_order_coefficients``,
    ``compute_log_series_coefficients`` and
    ``locate_log_remainder_threshold``. The grouped polynomial is restricted
    to the common-rate diagonal during normalization, so this internal lift
    leaves the requested one-rate library unchanged. Their validation rules
    apply.

    Any argument left as ``None`` takes its benchmark value: sequence
    ``GGACAUCGAGUCCUAG``; a uniform conditional replacement among the other
    three nucleotides; rule weights ``(0.32, 0.24, 0.18, 0.14, 0.70)``;
    left emissions ``(0.33, 0.21, 0.26, 0.20)``; right emissions
    ``(0.19, 0.30, 0.22, 0.29)``; pair emissions A-U 1.15, U-A 0.85,
    G-C 1.75, C-G 1.40, G-U 0.50 and U-G 0.65 with all others zero;
    stacking factors ``(2.2, 1.5, 1.3, 0.8)``; ``min_loop = 3``;
    ``log_order = 4``; ``target_share = 0.30``; and the unique-root bracket
    ``[0.045, 0.070]``.

    Parameters
    ----------
    sequence, mutation_kernel : np.ndarray or None
        Reference nucleotide codes and the conditional-replacement kernel.
    rule_weights, left_emission, right_emission : np.ndarray or None
        Grammar-rule and single-emission weights.
    pair_emission, stacking_factors : np.ndarray or None
        Ordered pair emissions and the four ordered canonical stack factors.
    min_loop, log_order : int or None
        Minimum paired-loop length and retained logarithmic series order.
    target_share, lower_rate, upper_rate : float or None
        Relative omitted-order target and its unique-crossing bracket.
    rate_tolerance : float
        Finite positive absolute bracket-width tolerance.

    Returns
    -------
    float
        Mutation rate at the unique crossing in the supplied bracket.

    Raises
    ------
    ValueError
        If an explicit or default input violates an upstream contract, if the
        logarithmic-order or solver controls are invalid, or if the requested
        target is not bracketed by ``lower_rate`` and ``upper_rate``.
    """
    return mutation_rate
```
