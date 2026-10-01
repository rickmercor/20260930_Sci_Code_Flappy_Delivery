# Physics-Quantum_Information_Computing-29

## Background

A binary detector-error matrix records which observed parities each independent fault flips. Local fault probabilities determine prior log odds, while cycles couple the messages used to infer a correction. A finite conditional search ranks incomplete histories using an internal statistic rather than solving a global likelihood optimization.

Calibration uncertainty can change both the numerical messages and the selected histories. The requested mean squared change concerns the statistic of a fresh run at each uncertain log-odds value, not a continuation of the decoder state across parameter values. The given finite input model is synthetic and requires no experimental dataset. The source defines the conditional ranking rule; the uniform uncertainty model defines how the resulting statistic is averaged.

## Problem

Quantum error-correction decoders must remain reliable when calibrated fault log odds vary slightly, since conditional decisions on short cycles can change discontinuously. Consider the finite-width quantum decoder introduced in a July 2026 journal article that ranks incomplete conditional fault histories before a valid correction is available; the inputs are a binary detector matrix, an uncertain prior log-odds vector and a syndrome, and the requested output is the mean squared change of its terminal internal ranking statistic under the specified calibration uncertainty.

Let R(x) be the largest retained statistic after the fixed branching schedule, with a fresh independent decoder initialization for each x, and compute the uniform expectation of (R(x)-R(0)) squared over the stated interval, not the square of the difference of expectations. Use the article's mathematical unscaled flooding min-sum equations and full branching pseudocode rather than released implementation optimizations: converged children remain eligible for retention and later expansion, every unmasked fault is eligible for selection and syndrome stopping occurs after each complete update.

All indexing starts at zero, fault-selection ties favor the smaller index, children are explored in value order zero then one and equal-score paths are ordered lexicographically by their sequences of (fault index, fixed value) pairs. Treat x and the supplied log odds as exact real inputs with p_j=1/(1+exp(ell_j(x))); zero posterior log odds decodes as one, and values at isolated parameter boundaries do not affect the expectation. In the short reasoning, state the reliability and ranking definitions used and report R(0) and the separate conditional mean squared changes for x<0 and x>0 without listing paths or parameter partitions.
number_of_detectors = 7
number_of_faults = 21
column_offsets = [(0,1,2),(0,1,3),(0,2,4)]
matrix_rule = H[i,7*a+b] equals 1 exactly when i is congruent to b+c modulo 7 for some c in column_offsets[a], for a=0,1,2 and b=0,...,6; all other entries are zero
base_log_odds = [29,47,23,61,37,53,31,43,59,19,41,67,17,71,73,79,83,89,97,101,103]/16
uncertain_log_odds = ell_12(x)=base_log_odds[12]+x; every other ell_j(x)=base_log_odds[j]
uncertainty_distribution = uniform on [-1/16,1/16]
syndrome = [1,0,0,1,1,0,0]
initial_iterations = 4
iterations_per_branch = 3
branching_rounds = 3
retained_population_limit = 2
solution_collection_limit = 64 distinct binary corrections
round_limit_behavior = stop after pruning the third population even if the collection limit was not reached
rounding = six digits after the decimal point
Your final answer must be a single number: the mean squared terminal-statistic change.

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
Output Format Requirements:
Emit <final_answer> immediately, then <reasoning>. Do not write a long derivation before the tags.
You must emit exactly one finite decimal inside <final_answer>...</final_answer>, even if the value is approximate or you are unsure.
Rules:
- The tags are required. Do not omit them or leave them empty.
- The tagged value must be a finite decimal (examples: 0.4847, 0.22, 1.05). Not NaN, not Inf, not a fraction string, not a vector, not prose.
- Put only that one number between the tags. No units, no words, no extra lines.
Keep <reasoning> short (a few hundred words). Show only the few scalars that determine the final number.
Do not paste the input matrices, full coefficient vectors, per-iteration paths, or per-fold candidate tables.
```

## Your task

Implement **all 10 functions** below into a single file at `/app/solution.py`. Requirements:

- Use the EXACT function name and signature shown for each step.
- Define every function at module scope (importable).
- Do NOT include tests, example usage, prints, or a `__main__` block.
- Available libraries: numpy, scipy, sympy, pandas, scikit-learn, networkx.

Start your file with these imports:

```python
import numpy as np
```

### Step 1

01_detector_matrix

Goal
----
Construct the binary detector matrix from circulant column supports. For modulus m and an integer array offsets of shape (b,k), column a*m+q contains ones at rows (q+offsets[a,t]) modulo m. Require 3<=m<=16, 1<=b<=4, 2<=k<=m, b*m<=64 and distinct offsets in each row, with each offset in [0,m-1]. Return an integer array of shape (m,b*m). These are synthetic construction bounds, not a fitted physical regime. Invalid inputs raise ValueError.

```python
import numpy as np


def detector_matrix(modulus, offsets):
    """Construct circulant fault supports.

    Parameters
    ----------
    modulus : int
        Nonboolean integer in [3,16].
    offsets : array_like
        Integer-valued support table with the bounds stated above.

    Returns
    -------
    numpy.ndarray
        Binary detector matrix.

    Raises
    ------
    ValueError
        If construction bounds or distinct-support conditions fail.
    """
    return None
```

### Step 2

02_residual_syndrome

Goal
----
Return the residual syndrome after fixing selected binary faults. XOR the original syndrome with each column of H whose fixed value is one. H is a binary array with shape (m,n), 1<=m<=32 and 2<=n<=64. The binary syndrome has length m and fixed has length n with entries -1, 0 or 1; -1 denotes an unmasked fault. Every detector must retain at least two unmasked neighbors. These synthetic domain restrictions keep every later min-sum reduction nonempty. Real numeric array inputs, including zero-imaginary complex arrays, are accepted. Invalid shape, finiteness, value or live-degree conditions raise ValueError. Return a length-m integer array without mutating inputs.

```python
import numpy as np


def residual_syndrome(H, syndrome, fixed):
    """Condition binary detector parity on fixed faults.

    Parameters
    ----------
    H : array_like
        Binary matrix with the stated synthetic size limits.
    syndrome : array_like
        Original binary detector outcomes.
    fixed : array_like
        Mask values -1, 0 or 1.

    Returns
    -------
    numpy.ndarray
        Residual binary syndrome.

    Raises
    ------
    ValueError
        If any domain or live-degree condition fails.
    """
    return None
```

### Step 3

03_masked_run

Goal
----
Run warm-started masked flooding min-sum. Obtain the residual syndrome from residual_syndrome(H,syndrome,fixed). On each live edge, the check message is (-1) to the residual bit power times the product of signs and minimum absolute value of the other live incoming fault messages; sign(0)=0. A live posterior is its prior plus incoming check messages. Its outgoing message excludes the recipient's check message. Accumulate signed posteriors starting at zero for this call only. Complete all outgoing updates before testing the residual syndrome, using live decisions posterior<=0 and zeros in masked positions. Stop after the first match or max_iters, never before iteration one. Return (messages,posterior_sum,iterations,correction,success), with fixed values restored even on failure, masked edge messages unchanged and nonedges zero. The matrix, syndrome and mask satisfy the residual-syndrome domain. Priors have length n and magnitude<=16, messages have shape (m,n), magnitude<=1e18 and zero nonedges; all are finite real numeric arrays, including zero-imaginary complex arrays. Treat input binary64 values as exact real inputs to the recurrence, without intermediate rounding that changes a sign, minimum or convergence decision. Return binary64 arrays with absolute error<=1e-9 plus 1e-12 times the reference magnitude in each entry. This includes small residuals after cancellation of large warm messages. max_iters is a nonboolean integer or integer-valued real scalar in [1,32]. Invalid inputs raise ValueError. Do not mutate inputs.

```python
import numpy as np


def masked_run(H, prior, syndrome, messages, fixed, max_iters):
    """Run one conditional min-sum trajectory.

    Parameters
    ----------
    H : array_like
        Binary detector matrix, 1<=m<=32 and 2<=n<=64.
    prior : array_like
        Finite real log odds, shape (n,), magnitude at most 16.
    syndrome : array_like
        Original binary syndrome, shape (m,).
    messages : array_like
        Incoming fault-to-detector snapshot, shape (m,n), bounded as above.
    fixed : array_like
        Mask vector of -1, 0 or 1 leaving at least two live faults per detector.
    max_iters : int
        Nonboolean integer-valued iteration budget in [1,32].

    Returns
    -------
    tuple
        Final messages, signed posterior sum, executed count, full correction
        and integer success flag. Array shapes are (m,n), (n,) and (n,).

    Raises
    ------
    ValueError
        If any stated domain condition fails.
    """
    return None
```

### Step 4

04_path_statistics

Goal
----
Return the next branching index, predictive path score and unnormalized reliability sum from one conditional run. For the unmasked index set U, select the smallest index minimizing abs(posterior_sum[j]); compute A=sum over U of abs(posterior_sum[j]) and score=A/iterations. The accumulator contains signed posteriors from the current run only, not a sum of magnitudes or a cumulative path history. posterior_sum is a finite real length-n array with magnitude<=1e12 and 2<=n<=64. fixed is a length-n array in {-1,0,1}, at least one entry is -1 and masked accumulator entries are zero. iterations is a nonboolean integer-valued real scalar in [1,32]. Return three native numeric scalars (index,score,A). Invalid inputs raise ValueError.

```python
import numpy as np


def path_statistics(posterior_sum, fixed, iterations):
    """Summarize one run's predictive reliability.

    Parameters
    ----------
    posterior_sum : array_like
        Signed within-run posterior sums, with the bounds above.
    fixed : array_like
        Mask values in {-1,0,1}.
    iterations : int
        Executed iteration count in [1,32].

    Returns
    -------
    tuple
        Selected index, normalized score and absolute reliability sum.

    Raises
    ------
    ValueError
        If the numerical or masking contract is violated.
    """
    return None
```

### Step 5

05_rank_paths

Goal
----
Return indices of the highest-scoring paths in deterministic order. Sort scores descending, break equal scores by the lexicographic sequence of (fault index,fixed value) pairs in each path and break identical keys by original candidate order. Keep at most width indices. scores is a finite nonnegative length-c array bounded by 1e12, with 1<=c<=64. paths has shape (c,d,2), where 0<=d<=6; entries are integer-valued, fault indices are in [0,63], values are binary and no index repeats inside a path. width is a nonboolean integer in [1,32]. Return an integer vector of length min(c,width). Invalid inputs raise ValueError; inputs are not mutated.

```python
import numpy as np


def rank_paths(scores, paths, width):
    """Rank candidate histories by score then lexicographic path.

    Parameters
    ----------
    scores : array_like
        Finite nonnegative path scores.
    paths : array_like
        Candidate history array of shape (c,d,2).
    width : int
        Maximum retained population in [1,32].

    Returns
    -------
    numpy.ndarray
        Retained indices in exploration order.

    Raises
    ------
    ValueError
        If any score, path or width constraint fails.
    """
    return None
```

### Step 6

06_branch_trajectory

Goal
----
Return the retained population of a finite predictive-reliability branching decoder. Initialize fault-to-detector messages to H times the prior vector, run masked_run(H,prior,syndrome,messages,all_unmasked,initial_iters) and select the initial node with path_statistics(). Each round expands retained paths in their current order, assigning their selected node zero then one. Each child copies its parent's final messages, extends its own fixed mask and runs masked_run() for inner_iters. Its next node and score come from path_statistics() on that call's accumulator and actual iterations. Converged children stay eligible. Rank children with rank_paths() and keep width entries. Track distinct successful full corrections. Return (scores,paths,initial_node,leading_iterations,leading_sum,result_count), with scores descending and paths shaped (population,rounds,2). H and prior satisfy masked_run's domain; H must have at least rounds+2 ones per row and n>rounds. rounds is a nonboolean integer in [1,6], width in [1,8] and both iteration budgets are nonboolean integer-valued scalars in [1,32]. These bounds allow at most 63 conditional calls, so a 64-distinct-solution collection limit never terminates this benchmark before its round limit. Invalid domains or upstream magnitude failures raise ValueError. Do not mutate inputs.

```python
import numpy as np


def branch_trajectory(H, prior, syndrome, initial_iters=12, inner_iters=8, rounds=6, width=4):
    """Run the bounded predictive branching trajectory.

    Parameters
    ----------
    H : array_like
        Binary detector matrix with row degree at least rounds+2.
    prior : array_like
        Real log odds within the conditional-run domain.
    syndrome : array_like
        Binary original detector outcomes.
    initial_iters : int
        Root iteration budget in [1,32].
    inner_iters : int
        Child iteration budget in [1,32].
    rounds : int
        Nonboolean number of branching rounds in [1,6].
    width : int
        Nonboolean retained width in [1,8].

    Returns
    -------
    tuple
        Scores, path pairs, initial node, leading run count, leading absolute
        reliability sum and distinct correction count.

    Raises
    ------
    ValueError
        If any stated domain or upstream calculation bound fails.
    """
    return None
```

### Step 7

07_terminal_score

Goal
----
Return the largest retained predictive path score for a circulant detector fixture. Compose detector_matrix(modulus,offsets) with branch_trajectory() using the supplied log odds and syndrome, then return its first retained score as a native float. The modulus and offset domain is that of detector_matrix(), and all remaining inputs obey branch_trajectory(), including row degree at least rounds+2, n>rounds, finite prior magnitude<=16, nonboolean rounds in [1,6], width in [1,8] and nonboolean integer-valued iteration budgets in [1,32]. The synthetic domain keeps the distinct-result collection limit 64 unreachable. Every upstream domain failure raises ValueError. Return the unrounded score and do not mutate inputs.

```python
import numpy as np


def terminal_score(modulus, offsets, prior, syndrome, initial_iters=12, inner_iters=8, rounds=6, width=4):
    """Compose the detector construction and finite branching trajectory.

    Parameters
    ----------
    modulus : int
        Circulant modulus in [3,16].
    offsets : array_like
        Distinct within-row circulant support offsets.
    prior : array_like
        Exact input log odds in fault-column order.
    syndrome : array_like
        Binary detector outcomes.
    initial_iters : int
        Root iteration budget in [1,32].
    inner_iters : int
        Child iteration budget in [1,32].
    rounds : int
        Branching depth in [1,6].
    width : int
        Retained population limit in [1,8].

    Returns
    -------
    float
        Largest terminal predictive reliability score, unrounded.

    Raises
    ------
    ValueError
        If any upstream construction or numerical bound fails.
    """
    return None
```

### Step 8

08_trace_score_cell

Goal
----
Return the exact affine terminal score and its executed-comparison cell for uncertain priors base+direction*x. H is binary shape(m,n), 1<=m<=8, 2<=n<=24, n>rounds and row degree>=rounds+2. base and direction have length n, are finite real binary64 inputs interpreted exactly and have magnitudes<=16 and<=4. syndrome is binary length m. sample is an integer pair[numerator,positive_denominator]; bounds has two such rows and satisfies -1<=lower<sample<upper<=1. initial_iters and inner_iters are nonboolean integers in[1,6], rounds in[1,3] and width in[1,2]. Return four reduced rational pairs for cell_lower,cell_upper,intercept,slope, with positive denominators. Invalid domains or a nonconstant comparison expression vanishing at sample raise ValueError. Identically zero expressions return sign zero without narrowing. Do not mutate inputs.

```python
import numpy as np


def trace_score_cell(H, base, direction, syndrome, sample, bounds, initial_iters=4, inner_iters=3, rounds=3, width=2):
    """Return a rational affine score cell for conditional decoding.

    Parameters
    ----------
    H : array_like
        Binary incidence within the stated size and live-degree bounds.
    base : array_like
        Base log odds, interpreted as exact binary64 values.
    direction : array_like
        Affine log-odds perturbation coefficients.
    syndrome : array_like
        Binary original detector syndrome.
    sample : array_like
        Exact rational interior point as an integer numerator-denominator pair.
    bounds : array_like
        Two rational endpoint pairs in [-1,1].
    initial_iters : int
        Initial run budget in [1,6].
    inner_iters : int
        Child run budget in [1,6].
    rounds : int
        Branching depth in [1,3].
    width : int
        Retained population width in [1,2].

    Returns
    -------
    list
        Four reduced rational pairs encoding cell endpoints and affine score.

    Raises
    ------
    ValueError
        If the domain fails or a nonconstant queried expression is zero at sample.
    """
    return None
```

### Step 9

09_cell_squared_integral

Goal
----
Integrate a squared affine deviation over a rational interval. cell is a four-by-two integer table encoding lower endpoint a, upper endpoint b, intercept u and slope v as reduced or unreduced numerator/positive-denominator pairs. Require -1<=a<b<=1, |u|<=1e12 and |v|<=1e12. reference is a finite nonboolean real scalar with magnitude<=1e12 and is interpreted as its exact binary64 value. Return the native float integral of (u+v*x-reference)^2 on [a,b]. Compute this as (b-a)*(z_a^2+z_a*z_b+z_b^2)/3 using exact rational arithmetic before the final float conversion, where z_a=u+v*a-reference and z_b=u+v*b-reference. Invalid inputs raise ValueError.

```python
import numpy as np


def cell_squared_integral(cell, reference):
    """Integrate the squared deviation on one affine cell.

    Parameters
    ----------
    cell : array_like
        Four rational pairs for lower, upper, intercept and slope.
    reference : float
        Reference score within the stated finite bound.

    Returns
    -------
    float
        Integral over the cell, without probability normalization.

    Raises
    ------
    ValueError
        If rational shapes, denominator signs or numerical bounds fail.
    """
    return None
```

### Step 10

10_uncertainty_response

Goal
----
Compute the uniform mean squared terminal-score change for one uncertain prior. Build H with detector_matrix(), evaluate reference=terminal_score() on the base priors using the supplied schedule and partition the uncertainty interval into exact affine comparison cells using trace_score_cell(). Sum cell_squared_integral(cell,reference) and divide by interval width. Return(mean_squared_change,reference,negative_conditional_mean,positive_conditional_mean) as native floats. lower and upper are integer rational pairs with -1<=lower<0<upper<=1; uncertain_index is a nonboolean integer in [0,n-1]. The direction is one at uncertain_index and zero elsewhere. Use trace_score_cell's domain: m<=8,n<=24,|base|<=16, initial_iters and inner_iters in[1,6], rounds in[1,3], width in[1,2], row degree>=rounds+2. Construction and binary-syndrome limits also apply. Require at most 4096 trace cells; if a new cell cannot be found by testing l+(h-l)/d for d=2,...,257 on an uncovered interval(l,h), raise ValueError. A sample at a nonconstant equality is skipped. Subtract the exact returned rational cell from the uncovered interval, split integration at zero and continue until covered. Return only after full interval coverage. Upstream failures or cap exhaustion raise ValueError.

```python
import numpy as np


def uncertainty_response(modulus, offsets, base, syndrome, uncertain_index=12, lower=(-1,16), upper=(1,16), initial_iters=4, inner_iters=3, rounds=3, width=2):
    """Integrate squared score changes under a uniform prior perturbation.

    Parameters
    ----------
    modulus : int
        Circulant modulus in [3,8].
    offsets : array_like
        Support offsets satisfying construction and trace-cell size bounds.
    base : array_like
        Base log odds in fault-column order.
    syndrome : array_like
        Binary detector outcomes.
    uncertain_index : int
        Index whose log odds are shifted by the uncertainty variable.
    lower : array_like
        Negative rational lower endpoint as an integer pair.
    upper : array_like
        Positive rational upper endpoint as an integer pair.
    initial_iters : int
        Root budget in [1,6].
    inner_iters : int
        Child budget in [1,6].
    rounds : int
        Branching depth in [1,3].
    width : int
        Retained width in [1,2].

    Returns
    -------
    tuple
        Mean squared change, base score and two half-interval conditional means.

    Raises
    ------
    ValueError
        For unsupported inputs, numerical bounds or partition-cap exhaustion.
    """
    return None
```
