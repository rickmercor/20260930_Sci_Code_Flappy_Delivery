# Biology-Biochemistry-8

## Background

Probabilistic analysis of biological sequences rests on dynamic programs that sum weights over hidden objects built from local factors: state paths in hidden Markov models, alignment paths in pairwise alignment ensembles, and parse trees or secondary structures in RNA grammars and partition-function models. Their forward and backward, or inside and outside, quantities give posterior probabilities and expected feature counts, and they are routinely reused for sensitivity analysis. Mutational scanning and sequence design ask a different question, namely what happens when observed nucleotides are replaced by finite amounts, and whether the effects of several substitutions combine additively, which bears on RNA stability and on epistasis between sites.

## Problem

I am scanning point substitutions in a long RNA under a toy stochastic grammar, and I want exact ensemble weights rather than first-order estimates. The grammar has one nonterminal and five productions, each contributing its rule weight times any emission factor: a pair production emitting a 5′ base and a 3′ base around the nonterminal (t_P = 0.70 times the ordered pair factor), a left-unpaired production emitting one base before it (t_L = 0.44 times e_L), a right-unpaired production emitting one base after it (t_R = 0.36 times e_R), a bifurcation into two nonterminals (t_B = 0.24) and an empty production (t_E = 0.80). In A, C, G, U order the emission factors are e_L = (0.28, 0.19, 0.27, 0.26) and e_R = (0.24, 0.26, 0.21, 0.29), and the ordered pair factors (5′ base, 3′ base) are 1.1 for A-U and U-A, 1.9 for C-G and G-C, 0.5 for G-U, 0.7 for U-G and zero for every other ordered pair. The weight of a sequence is the summed weight of all its parse trees, where the empty production derives only an empty stretch, each bifurcation child derives a non-empty stretch, and a pair production may join any two positions with no minimum loop. My reference RNA has 1,500 bases, read 5′ to 3′ with positions numbered from 1, base i being A, C, G or U as entry i of numpy.random.default_rng(20260912).integers(0, 4, 1500) is 0, 1, 2 or 3, and Z denotes its weight. I rank all 4,500 single substitutions by the absolute change in weight divided by Z, and I take the top-ranked substitution, at position p, together with the top-ranked substitution at a position q with |q − p| ≥ 200. For the double mutant carrying both, the interaction is its weight minus the weights of the two single mutants plus Z. I want the fraction of the double mutant's weight carried by parse trees in which position p is emitted as the 5′ base of a pair production. In your reasoning, state Z, the two chosen substitutions with their normalized changes and the interaction divided by Z, and tell me what these results imply about predicting substitution effects from derivatives of the weight computed on the reference sequence, both in this grammar and in nearest-neighbour thermodynamic models.

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

Implement **all 7 functions** below into a single file at `/app/solution.py`. Requirements:

- Use the EXACT function name and signature shown for each step.
- Define every function at module scope (importable).
- Do NOT include tests, example usage, prints, or a `__main__` block.
- Available libraries: numpy, scipy, sympy, pandas, scikit-learn, networkx.

Start your file with these imports:

```python
import numpy as np
```

### Step 1

01_compute_log_inside_table

Goal
----
Tabulate log ensemble weights for every interval of an RNA sequence under the five-production grammar.

```python
def compute_log_inside_table(sequence: str, rule_weights: "np.ndarray",
                             unpaired: "np.ndarray", pair_factors: "np.ndarray") -> "np.ndarray":
    """Return the natural logarithm of every interval's inside weight.

    The grammar is the five-production grammar in the problem statement. Array
    layouts are ``(t_P, t_L, t_R, t_B, t_E)``, left/right unpaired factors in
    A, C, G, U order, and ordered pair factors with the 5' base as the row.
    Entry ``[i, k]`` represents ``sequence[i:k]``. Empty intervals have weight
    ``t_E``; bifurcation children are non-empty; a pair may join the endpoints
    of any interval of length at least two. Store ``-inf`` for a zero weight and
    for entries below the diagonal.

    Parameters
    ----------
    sequence : str
        Non-empty RNA over A, C, G, U.
    rule_weights : np.ndarray
        Non-negative finite array of shape (5,), with positive t_E.
    unpaired : np.ndarray
        Non-negative finite array of shape (2, 4), left then right.
    pair_factors : np.ndarray
        Non-negative finite array of shape (4, 4), with nonzero entries only
        for A-U, U-A, C-G, G-C, G-U and U-G.

    Returns
    -------
    np.ndarray
        Float log-inside table of shape (n + 1, n + 1).

    Raises
    ------
    ValueError
        If the sequence or a grammar array violates the contract above.
    """
    return log_inside  # placeholder
```

### Step 2

02_compute_log_outside_table

Goal
----
Tabulate log outside adjoints for every interval of the five-production RNA grammar.

```python
def compute_log_outside_table(sequence: str, rule_weights: "np.ndarray", unpaired: "np.ndarray",
                              pair_factors: "np.ndarray", log_inside: "np.ndarray") -> "np.ndarray":
    """Return natural logarithms of the outside adjoints.

    ``log_inside`` follows ``compute_log_inside_table``. Entry ``[i, k]`` is
    the logarithm of the derivative of the root weight with respect to the
    linear inside weight of ``sequence[i:k]``. The root entry is therefore
    zero. Store ``-inf`` for a zero adjoint and below the diagonal. Reverse
    propagation follows the same empty, pair and non-empty-bifurcation
    conventions as the inside table.

    Parameters
    ----------
    sequence : str
        Non-empty RNA over A, C, G, U.
    rule_weights, unpaired, pair_factors : np.ndarray
        Grammar arrays in the layouts accepted by ``compute_log_inside_table``.
    log_inside : np.ndarray
        Log-inside table of shape (n + 1, n + 1), with finite root weight.

    Returns
    -------
    np.ndarray
        Float log-outside table of shape (n + 1, n + 1).

    Raises
    ------
    ValueError
        If the grammar is invalid or ``log_inside`` has the wrong shape,
        contains NaN or positive infinity, or has a non-finite root entry.
    """
    return log_outside  # placeholder
```

### Step 3

03_evaluate_single_replacement_channels

Goal
----
Evaluate every finite one-base replacement and resolve its exact weight among the four emission roles.

```python
def evaluate_single_replacement_channels(sequence: str, rule_weights: "np.ndarray",
                                         unpaired: "np.ndarray", pair_factors: "np.ndarray",
                                         log_inside: "np.ndarray",
                                         log_outside: "np.ndarray") -> "np.ndarray":
    """Return log weights for all candidates and the four emission roles.

    For each 0-based position and candidate base in A, C, G, U order, resolve
    the exact finite-replacement endpoint into parse trees where that position
    is emitted as the 5' base of a pair, the 3' base of a pair, left-unpaired,
    or right-unpaired, in that axis-2 order. The four classes are mutually
    exclusive and exhaustive; log-summing them gives the complete mutant
    weight. Use the reference log tables from the first two steps. Store
    ``-inf`` when a role has zero weight.

    Parameters
    ----------
    sequence : str
        Non-empty RNA over A, C, G, U.
    rule_weights, unpaired, pair_factors : np.ndarray
        Grammar arrays in the layouts accepted by ``compute_log_inside_table``.
    log_inside, log_outside : np.ndarray
        Reference log tables of shape (n + 1, n + 1).

    Returns
    -------
    np.ndarray
        Float array of shape (n, 4, 4): position, candidate base, role.

    Raises
    ------
    ValueError
        If the grammar is invalid, a log table has the wrong shape, contains
        NaN or positive infinity, or the reference root weight is non-finite.
    """
    return channel_logs  # placeholder
```

### Step 4

04_rank_single_substitutions

Goal
----
Rank exact single substitutions from their role-resolved log weights and select the strongest separated pair.

```python
def rank_single_substitutions(sequence: str, channel_logs: "np.ndarray",
                              minimum_gap: int) -> "np.ndarray":
    """Return the strongest substitution and the strongest separated one.

    ``channel_logs[p, c, r]`` contains the four role log weights from
    ``evaluate_single_replacement_channels``. Log-sum the role axis to obtain
    each exact mutant endpoint. The endpoint for the unchanged base must give
    one common finite reference weight. Score every changed base by its
    absolute weight change divided by that reference and return
    ``[top_score, p, c, gap_score, q, d]``. Positions and bases are 0-based and
    A, C, G, U ordered. Ties go to the smaller position, then earlier base.

    Returns
    -------
    np.ndarray
        Float array ``[top_score, p, c, gap_score, q, d]``.

    Raises
    ------
    ValueError
        If the sequence is invalid, ``channel_logs`` has the wrong shape or
        contains NaN or positive infinity, the unchanged endpoints disagree,
        ``minimum_gap`` is not a positive integer, a score is non-finite, or
        no substitution is sufficiently separated.
    """
    return ranked  # placeholder
```

### Step 5

05_evaluate_ordered_double_endpoints

Goal
----
Evaluate total and event-specific endpoint weights for a selected two-site substitution in an ordered mutant background.

```python
def evaluate_ordered_double_endpoints(sequence: str, rule_weights: "np.ndarray",
                                      unpaired: "np.ndarray", pair_factors: "np.ndarray",
                                      log_inside: "np.ndarray", channel_logs: "np.ndarray",
                                      position: int, nucleotide: str,
                                      partner_position: int,
                                      partner_nucleotide: str) -> "np.ndarray":
    """Return log endpoint weights for a selected two-site replacement.

    Let p = ``position`` receive c = ``nucleotide`` and q =
    ``partner_position`` receive d = ``partner_nucleotide``. Row 0 contains
    natural logs of total weights for [reference, p-only, q-only, double]. Row
    1 contains the corresponding logs of the unnormalized event weight in
    which p is the 5' base of a pair. Evaluate the double endpoint by exact
    one-site recombination at p in the background already carrying q to d.
    ``log_inside`` and ``channel_logs`` are reference outputs from steps 1 and
    3. Store ``-inf`` for a zero event weight.

    Returns
    -------
    np.ndarray
        Float array of shape (2, 4), with endpoint columns ordered as above.

    Raises
    ------
    ValueError
        If the grammar or reference outputs are invalid, positions are not
        distinct integers in range, or a replacement is not a one-character
        RNA base different from the base already present.
    """
    return log_endpoints  # placeholder
```

### Step 6

06_compute_epistatic_pairing_metrics

Goal
----
Convert total and event-specific endpoint log weights into the normalized finite interaction and double-mutant pairing share.

```python
def compute_epistatic_pairing_metrics(log_endpoints: "np.ndarray") -> "np.ndarray":
    """Return the normalized interaction and double-mutant event share.

    ``log_endpoints`` has shape (2, 4) and the layout returned by
    ``evaluate_ordered_double_endpoints``. From total endpoints
    ``[Z, Z_p, Z_q, Z_pq]``, compute ``(Z_pq - Z_p - Z_q + Z) / Z``.
    Compute the pairing share as the double-mutant event endpoint divided by
    ``Z_pq``. Event endpoints may be ``-inf`` for zero weight and must not
    exceed their corresponding total endpoints.

    Returns
    -------
    np.ndarray
        Float array ``[normalized_interaction, pairing_share]``.

    Raises
    ------
    ValueError
        If the input has the wrong shape, a total endpoint is non-finite, an
        event endpoint is NaN or positive infinity, an event exceeds its total,
        or either returned metric is non-finite or the share lies outside [0, 1].
    """
    return metrics  # placeholder
```

### Step 7

07_compute_epistatic_pairing_share

Goal
----
Compose every earlier step to rank all single substitutions of a long reference RNA and evaluate the selected double mutant through stable total-weight and pairing-event log endpoints. Orchestrator: Yes, it builds the log inside and outside tables, evaluates role-resolved single-replacement endpoints, ranks substitutions, evaluates ordered two-site total and event endpoints, and converts them to the requested share.

```python
def compute_epistatic_pairing_share(
    length: int = 1500,
    seed: int = 20260912,
    minimum_gap: int = 200,
    rule_weights: tuple = (0.7, 0.44, 0.36, 0.24, 0.8),
    unpaired: tuple = ((0.28, 0.19, 0.27, 0.26), (0.24, 0.26, 0.21, 0.29)),
    pair_factors: tuple = ((0.0, 0.0, 0.0, 1.1), (0.0, 0.0, 1.9, 0.0), (0.0, 1.9, 0.0, 0.5), (1.1, 0.0, 0.7, 0.0)),
) -> float:
    """Return the 5' pairing share of the strongest substitution's site in the double mutant.

    The reference is the ``length``-base string whose base i is 'ACGU'[v_i]
    for v = numpy.random.default_rng(seed).integers(0, 4, length). Rank its
    single substitutions as in ``rank_single_substitutions`` with
    ``minimum_gap``, giving (p, c) and (q, d), and form the double mutant y
    carrying both. Evaluate the ordered finite endpoints in log space and
    return the summed weight of the parse trees of y in which position p is
    emitted as the 5' base of a pair production, divided by the ensemble
    weight of y. Grammar arguments follow ``compute_log_inside_table``
    (array-likes are accepted); the defaults reproduce the problem statement.

    Returns
    -------
    float
        Share in [0, 1].

    Raises
    ------
    ValueError
        If ``length`` is not an integer >= 2, ``seed`` is not a non-negative
        integer, or ``minimum_gap`` or the grammar is invalid as in the steps above.
    """
    return 0.0  # placeholder
```
