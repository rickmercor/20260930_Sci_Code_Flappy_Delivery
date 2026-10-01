# Physics-Quantum_Information_Computing-40

## Background

Quantum error-correction decoders often work with a sparse binary detector model whose columns represent possible fault mechanisms and whose rows represent detector events. Belief propagation exchanges local likelihood messages on the associated Tanner graph and can be effective when the graph is sparse, but short cycles and nearly degenerate explanations of a syndrome can make the iterative messages oscillate or fail to converge.

A bounded search can complement local message passing by fixing a small number of uncertain variables, re-running inference on the reduced graph, and retaining only a limited number of candidate partial assignments. Such a design trades additional computation for a broader exploration of competing explanations while still avoiding exhaustive enumeration of all binary error vectors.

## Problem

Sparse detector-error models for quantum error correction can be decoded by combining min-sum belief propagation with a bounded tree search that fixes unreliable error nodes and retains only the most promising partial assignments. For this deterministic benchmark, use the supplied source's Appendix A/B decoder on the binary detector matrix $H$, error probabilities $p$, and syndrome $s$ below, with $(\mathrm{max\_rounds},\mathrm{beam\_widths},\mathrm{initial\_iters},\mathrm{iters\_per\_round},\mathrm{num\_results})=(6,(1,2,4,8),3,4,3)$; the decoder output for each beam width must be a valid correction satisfying $H\hat e=s\pmod 2$.

$$
H=\begin{pmatrix}
0&1&0&0&1&0&0&0&1&1&0&0&0&0&1&0&0&1&0&0\\
1&1&0&0&0&0&1&0&0&0&0&1&1&0&0&0&0&1&0&0\\
0&1&0&1&0&0&0&1&0&0&0&0&1&0&0&1&0&0&1&0\\
0&1&0&0&0&1&0&0&1&0&0&1&0&1&0&1&0&0&0&0\\
0&0&0&1&0&0&0&0&0&1&0&0&0&1&0&0&1&1&1&0\\
0&0&0&0&0&0&0&0&0&1&1&0&1&0&0&0&1&1&0&1\\
1&0&0&0&0&0&1&0&1&0&0&0&0&0&0&1&1&0&1&0\\
0&1&0&0&1&0&1&0&0&0&0&1&0&0&0&0&0&1&0&1\\
0&1&1&0&0&0&0&1&1&1&0&0&0&0&0&1&0&0&0&0\\
0&1&0&0&1&1&0&0&0&1&0&0&1&0&1&0&0&0&0&0\\
0&1&1&0&0&1&0&0&0&0&0&0&0&0&0&1&0&1&0&1
\end{pmatrix},
$$
$$
p=(0.099321675569,0.055717560406,0.074255307877,0.070190994539,0.151229358583,0.145537983086,0.035252957104,0.085929973968,0.086129312934,0.159482638903,0.153475804526,0.075766077846,0.144920993860,0.135244812128,0.140972421803,0.131736174287,0.114588720979,0.098089955076,0.082017411514,0.155506630793),
$$
$$
s=(1,0,1,0,1,1,0,1,0,1,1).
$$

Use row-major Tanner-edge order, $\operatorname{sign}(0)=+1$, the smallest node index for every equal-reliability node tie, child value $0$ before $1$, and stable input order for equal path scores; only unique valid corrections count toward $\mathrm{num\_results}$, and if the round budget ends before three unique results are collected, select the minimum-weight valid correction found. Before the source-specific bounded search, explicitly report the first three prior LLRs $\log((1-p_j)/p_j)$, the detector-row degree vector, the syndrome Hamming weight, and the total prior-LLR sum; then identify from the source the cumulative-LLR reliability rule, masked-BP warm start and syndrome adjustment, normalized path score and pruning order, and multi-result minimum-weight termination rule.

For each $b\in(1,2,4,8)$, name the five diagnostics as $d_b=(W_b,h_b,r_b,R_b,C_b)$, where $W_b$ is the selected correction weight, $h_b$ its Hamming weight, $r_b$ the number of branching rounds used, $R_b$ the number of unique valid results found on return, and $C_b$ the number of child paths expanded. Form
$$
g_b=\left(\frac{W_b}{\sum_j\log((1-p_j)/p_j)},\frac{h_b}{20},\frac{r_b}{6},\frac{R_b}{3},\frac{C_b}{12b}\right),
$$
and compute $L=\|g_2-g_1\|_2+\|g_4-g_2\|_2+\|g_8-g_4\|_2$ without intermediate rounding; the reasoning must also state the initial BP hard-decision support and seed branching index, the two first-round child scores and next branching indices, all four $d_b$, the selected correction supports, all four $g_b$, and the three path increments. Report $L$ to exactly 12 decimal places.

## Output Format Requirements
Output `<final_answer>` first with exactly one finite decimal value, followed by `<reasoning>` containing the derivation and requested diagnostics.

## Output format

```
## Output format

Scientific reasoning wrapped in <reasoning>...</reasoning> tags. Include enough intermediate calculations to justify the deterministic computation, without turning the response into a general pipeline summary. A single final numeric answer wrapped in <final_answer>...</final_answer> tags.
```

## Your task

Implement **all 9 functions** below into a single file at `/app/solution.py`. Requirements:

- Use the EXACT function name and signature shown for each step.
- Define every function at module scope (importable).
- Do NOT include tests, example usage, prints, or a `__main__` block.
- Available libraries: numpy, scipy, sympy, pandas, scikit-learn, networkx.

Start your file with these imports:

```python
import numpy as np
```

### Step 1

01_min_sum_bp_trace.py

Goal
----
Run standard min-sum BP and return posterior LLRs, hard decisions, final warm-start edge messages, and convergence status for every executed iteration.

```python
def min_sum_bp_trace(H: "np.ndarray", p: "np.ndarray", syndrome: "np.ndarray", max_iters: int) -> "np.ndarray":
    '''Return the standard min-sum BP trace in deterministic row-major edge order.

    Parameters
    ----------
    H : np.ndarray
        Binary parity-check matrix of shape (M,N); each detector row has degree at least two.
    p : np.ndarray
        Error-source probabilities of shape (N,), all strictly between 0 and 0.5.
    syndrome : np.ndarray
        Binary syndrome vector of shape (M,).
    max_iters : int
        Positive maximum number of BP iterations.

    Returns
    -------
    trace : np.ndarray
        Float array of shape (T, 2*N + E + 1), T <= max_iters, where E is the
        number of nonzero entries in H. Columns are posterior LLRs, hard bits,
        error-to-detector messages in row-major (detector,error) edge order,
        and a success flag. Stop at the first syndrome match. Use sign(0)=+1
        and hard bit 0 for posterior>0, otherwise 1.
    '''
    return trace
```

### Step 2

02_seed_beam_state.py

Goal
----
Convert the initial BP trace into the single numerical seed path used by the bounded search.

```python
def seed_beam_state(H: "np.ndarray", p: "np.ndarray", syndrome: "np.ndarray", initial_iters: int) -> "np.ndarray":
    '''Return the seed-path state built from the initial BP run.

    Parameters
    ----------
    H : np.ndarray
        Binary parity-check matrix of shape (M,N).
    p : np.ndarray
        Error probabilities of shape (N,).
    syndrome : np.ndarray
        Binary syndrome of shape (M,).
    initial_iters : int
        Positive BP iteration cap.

    Returns
    -------
    state : np.ndarray
        Float vector of length 4+2*N+E with layout
        [success, iters, score, next_pos, mask_values(N), edge_msgs(E), correction(N)].
        mask_values are -1 for every unmasked node, score is 0, and next_pos is
        the smallest index minimizing the absolute cumulative posterior LLR over
        the iterations actually executed.
    '''
    return state
```

### Step 3

03_masked_bp_trace.py

Goal
----
Run min-sum BP after applying fixed-node masks, source-defined syndrome modification, and parent-message warm start.

```python
def masked_bp_trace(
    H: "np.ndarray",
    p: "np.ndarray",
    syndrome: "np.ndarray",
    edge_msgs: "np.ndarray",
    mask_values: "np.ndarray",
    max_iters: int,
) -> "np.ndarray":
    """Return the warm-started masked-BP trace.

    Parameters
    ----------
    H : np.ndarray
        Binary parity-check matrix of shape (M, N).
    p : np.ndarray
        Error probabilities of shape (N,).
    syndrome : np.ndarray
        Original binary syndrome of shape (M,).
    edge_msgs : np.ndarray
        Warm-start error-to-detector messages for all E row-major Tanner
        edges. These values initialize the message state before iteration.
    mask_values : np.ndarray
        Integer-like vector of length N with -1 for unmasked nodes and
        0 or 1 for fixed nodes. Every computed detector-to-error message
        has at least one other unmasked neighbor for the supplied inputs.
    max_iters : int
        Positive masked-BP iteration cap.

    Returns
    -------
    trace : np.ndarray
        Float array of shape (T, 2*N + E + 1), where T <= max_iters.
        Columns contain posterior LLRs, hard bits, error-to-detector
        messages in row-major edge order, and a success flag, respectively.
        Masked posterior LLRs and masked hard bits are zero in every row.
        Message slots incident to masked error nodes retain their input
        edge_msgs values unchanged in every row; those slots do not enter
        any BP update. Unmasked message slots contain the updated messages.
        For each fixed-one node, flip its adjacent syndrome bits before
        iteration. Ignore messages to or from masked nodes. Stop at the
        first match to that modified syndrome. Use sign(0)=+1 and set an
        unmasked hard bit to 0 for positive posterior LLR, otherwise to 1.
    """
    return trace
```

### Step 4

04_expand_beam_state.py

Goal
----
Branch one active path at its selected node, evaluate fixed values zero and one, and compute each child's next branch and reliability score.

```python
def expand_beam_state(
    H: "np.ndarray",
    p: "np.ndarray",
    syndrome: "np.ndarray",
    state: "np.ndarray",
    iters_per_round: int,
) -> "np.ndarray":
    """Return the two child path states in fixed-value order 0 then 1.

    Parameters
    ----------
    H : np.ndarray
        Binary parity-check matrix of shape (M, N).
    p : np.ndarray
        Error probabilities of shape (N,).
    syndrome : np.ndarray
        Original binary syndrome.
    state : np.ndarray
        Parent state in the layout returned by seed_beam_state:
        [success, iters, score, next_pos, mask_values(N), edge_msgs(E),
        correction(N)]. The selected branch node is unmasked.
    iters_per_round : int
        Positive masked-BP iteration cap for each child.

    Returns
    -------
    children : np.ndarray
        Float array of shape (2, 4 + 2*N + E), using the common state
        layout. Each child fixes the parent's next_pos to its branch value
        and starts masked BP from the parent's edge messages. Message slots
        incident to any masked node, including the new branch node, retain
        their parent edge-message values unchanged.
        next_pos is the smallest-index unmasked node minimizing the
        absolute cumulative posterior LLR over the executed iterations.
        score is the sum of those magnitudes over unmasked nodes divided
        by the number of masked-BP iterations actually executed.
        If no unmasked nodes remain, next_pos is -1 and score is zero.
        On success, correction is the full valid correction with all fixed
        mask values restored. On failure, correction contains the final
        masked-BP hard bits, including zero at each masked position.
    """
    return children
```

### Step 5

05_prune_beam_states.py

Goal
----
Keep at most the requested beam width of highest-reliability candidate paths.

```python
def prune_beam_states(states: "np.ndarray", beam_width: int) -> "np.ndarray":
    '''Keep the highest-scoring states using stable descending order.

    Parameters
    ----------
    states : np.ndarray
        Float array of shape (P,D) in the common path-state layout; column 2 is score.
    beam_width : int
        Positive maximum number of rows to retain.

    Returns
    -------
    pruned : np.ndarray
        At most beam_width rows sorted by decreasing score. Equal scores preserve
        their original input order.
    '''
    return pruned
```

### Step 6

06_minimum_weight_correction.py

Goal
----
Deduplicate valid corrections and select the member with minimum Bernoulli log-likelihood weight.

```python
def minimum_weight_correction(corrections: "np.ndarray", p: "np.ndarray") -> "np.ndarray":
    '''Return the minimum-weight unique correction.

    Parameters
    ----------
    corrections : np.ndarray
        Binary array of shape (R,N), R>=1; duplicate rows are allowed.
    p : np.ndarray
        Error probabilities of shape (N,), all strictly between 0 and 0.5.

    Returns
    -------
    result : np.ndarray
        Float vector [minimum_weight, correction_bits...]. Weight is the sum over
        selected bits of log((1-p_j)/p_j). Duplicates are removed preserving first
        appearance; exact weight ties choose the earliest remaining row.
    '''
    return result
```

### Step 7

07_beam_search_decode.py

Goal
----
Execute the complete source search for one syndrome, including initialization, masked branching, descending-score exploration, pruning, unique-result collection, and weighted selection.

```python
def beam_search_decode(H: "np.ndarray", p: "np.ndarray", syndrome: "np.ndarray", max_rounds: int, beam_width: int, initial_iters: int, iters_per_round: int, num_results: int) -> "np.ndarray":
    '''Decode one syndrome with bounded BP-guided branching.

    Parameters
    ----------
    H : np.ndarray
        Binary parity-check matrix of shape (M,N).
    p : np.ndarray
        Error probabilities of shape (N,).
    syndrome : np.ndarray
        Binary syndrome of shape (M,).
    max_rounds : int
        Positive maximum number of masked branching rounds.
    beam_width : int
        Positive number of active paths retained after each round.
    initial_iters : int
        Positive initial standard-BP iteration cap.
    iters_per_round : int
        Positive masked-BP iteration cap per child.
    num_results : int
        Positive target number of unique valid corrections.

    Returns
    -------
    result : np.ndarray
        Float vector [weight, rounds_used, unique_results_found,
        children_expanded, correction_bits...]. Paths are explored in stable
        decreasing-score order; each path branches value 0 then 1. Stop as soon
        as num_results unique valid corrections have been collected and choose
        their minimum-weight member. If max_rounds is exhausted first, choose
        the minimum-weight member among all unique valid corrections found.
        Unless collecting a correction triggers this overall return, retain
        every evaluated child for the next round's score-based pruning,
        including successful children and children yielding duplicate
        corrections. A successful child may therefore be expanded again.

    Raises
    ------
    ValueError
        If no valid correction is found within the search budget.
    '''
    return result
```

### Step 8

08_beam_search_profile.py

Goal
----
Evaluate the full decoder across the ordered beam widths and summarize correction quality and search work.

```python
def beam_search_profile(H: "np.ndarray", p: "np.ndarray", syndrome: "np.ndarray", beam_widths: "np.ndarray", max_rounds: int, initial_iters: int, iters_per_round: int, num_results: int) -> "np.ndarray":
    '''Return one five-component decoder profile per beam width.

    Parameters
    ----------
    H, p, syndrome : np.ndarray
        Detector model, error probabilities, and binary syndrome.
    beam_widths : np.ndarray
        Positive integer-like beam widths in the requested order.
    max_rounds, initial_iters, iters_per_round, num_results : int
        Decoder controls passed unchanged to beam_search_decode.

    Returns
    -------
    profile : np.ndarray
        Float array of shape (B,5). Each row is
        [selected_weight, correction_hamming_weight, rounds_used,
        unique_results_found, children_expanded].
    '''
    return profile
```

### Step 9

09_cumulative_beam_profile_path.py

Goal
----
Normalize the four beam-width profiles and sum the Euclidean distances between consecutive profiles.

```python
def cumulative_beam_profile_path(H: "np.ndarray", p: "np.ndarray", syndrome: "np.ndarray", beam_widths: "np.ndarray", max_rounds: int, initial_iters: int, iters_per_round: int, num_results: int) -> float:
    '''Return the cumulative Euclidean path through normalized beam profiles.

    Parameters
    ----------
    H, p, syndrome : np.ndarray
        Detector model, probabilities, and binary syndrome.
    beam_widths : np.ndarray
        Ordered positive beam widths.
    max_rounds, initial_iters, iters_per_round, num_results : int
        Decoder parameters.

    Returns
    -------
    value : float
        For each width b and profile row (W,h,r,R,C), form
        g_b=(W/sum_j log((1-p_j)/p_j), h/N, r/max_rounds,
        R/num_results, C/(2*b*max_rounds)). Return the sum of Euclidean
        distances between consecutive g_b rows, without intermediate rounding.
    '''
    return value
```
