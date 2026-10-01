# Physics-Quantum_Information_Computing-18

## Background

Quantum-error decoders infer logical classes from detector outcomes while many fault histories remain indistinguishable at the detector level. Correlated physical noise couples the bit and phase components of those histories. Sampling one component can supply empirical information for reweighting the other, but finite samples carry both temporal dependence and estimation uncertainty.

A completed nonlocal sampling update may require a variable number of elementary proposals. A decoder's logical decision can therefore be correlated with its computational cost, especially when empirical information changes the later transition rules. Small exact models can resolve this dependence and distinguish variability in work from variability in the reported logical class. Retaining the sampled configurations across successive feedback passes makes the relevant history richer than a collection of marginal probabilities.

## Problem

Finite-history feedback can couple quantum-decoder decisions to computational work, so consider a synthetic paired-detector benchmark with dimensionless depolarizing strength $p=0.24$ at seven paired fault locations and the following binary detector matrices, fixed reference chains, measured syndromes, logical map and edge correspondence:

$$
H_Z=\begin{pmatrix}
1&1&1&0&0&0&1\\
1&0&0&1&1&0&0\\
0&1&0&1&0&1&0\\
0&0&1&0&1&1&0
\end{pmatrix},\qquad
H_X=\begin{pmatrix}
1&1&0&1&0&0&0\\
1&0&1&0&1&1&0\\
0&1&1&0&0&0&1\\
0&0&0&1&1&0&0
\end{pmatrix}.
$$

The fixed reference chains, observed syndromes, final logical map, and edge correspondence are

$$
M_Z=(0,1,0,0,0,0,1),\quad s_Z=(0,0,1,0)^{\mathsf T},\qquad
M_X=(1,0,0,0,0,1,0),\quad s_X=(1,0,0,0)^{\mathsf T},
$$

$$
L_X=\begin{pmatrix}1&0&0&0&0&0&0\\0&0&1&0&0&0&0\end{pmatrix},\qquad
f=(3,6,1,5,0,4,2),
$$

Here zero-based Z-edge $e$ pairs with X-edge $f_e$, binary maps act modulo two, the physical error represented by relative cycle $C$ is $C\oplus M$, and physical logical word $(b_0,b_1)$ has integer label $b_0+2b_1$.
Use the conditional cycle ensemble with one shared added boundary vertex for each graph's one-detector faults and fixed-tail, single-mobile-defect updates: choose the tail uniformly, propose an incident neighbor uniformly, use degree-corrected Metropolis acceptance, and stop on the first return of the head to its tail after at least one proposal, so an initial rejection completes an update while subsequent rejections retain the open state.
Apply phases $Z_0\to X_1\to Z_2\to X_3$ with sample counts $(2,2,2,5)$ and completed-update spacings $(2,2,1,2)$; both relative cycles initially equal zero, each sector retains its last terminal cycle when the other sector is sampled, and the empirical occupancy counters restart for each phase.
Start $Z_0$ with the depolarizing-channel marginal prior, then freeze each next phase's priors from the preceding phase's realized physical single-edge empirical marginals using channel-conditional reweighting, the forward map $f$ for Z-to-X feedback and its inverse for X-to-Z feedback, always using the supplied fixed reference chain.
Record only after the specified number of completed updates, retain the chain between records, and report the most frequent physical logical label among the five final X records, breaking ties by the smallest integer label.
Let $W$ count every attempted elementary edge proposal across all four phases, including accepted and rejected trials, and compute the conditional skewness of $W$ given that the reported label is two:

$$
\gamma_2=\frac{\mathbb E[(W-\mu_2)^3\mid\widehat g=2]}
{\operatorname{Var}(W\mid\widehat g=2)^{3/2}},\qquad
\mu_2=\mathbb E[W\mid\widehat g=2].
$$

Evaluate the finite stochastic process without Monte Carlo, excursion-length truncation, equilibrium substitution, independent-sample replacement or premature averaging of feedback; justify the reference and channel transformations, the complete-excursion work law, the jointly retained history information and the physical-label decision, and report the conditioning-event probability, conditional mean, conditional variance and final report-law mass residual in the reasoning.
Return $\gamma_2$ as one finite decimal with absolute error at most $10^{-8}$.

Output Format Requirements:
Emit <final_answer> immediately, then <reasoning>. Do not write a long
derivation before the tags.
You must emit exactly one finite decimal inside
<final_answer>...</final_answer>, even if the value is approximate or you
are unsure.
Rules:
- The tags are required. Do not omit them or leave them empty.
- The tagged value must be a finite decimal (examples: 0.4847, 0.22, 1.05).
Not NaN, not Inf, not a fraction string, not a vector, not prose.
- Put only that one number between the tags. No units, no words, no extra
lines.
Keep <reasoning> short (a few hundred words). Show only the few scalars that
determine the final number.
Do not paste the input matrices, full coefficient vectors, per-iteration
paths, or per-fold candidate tables.

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

01_construct_physical_cycle_ensemble

Goal
----
Construct the ordered physical error configurations represented by closed relative cycles.

```python
def construct_physical_cycle_ensemble(
    detectors: "np.ndarray", reference: "np.ndarray"
) -> "np.ndarray":
    r"""Construct the ordered physical error configurations represented by closed relative cycles.

    Parameters
    ----------
    detectors : np.ndarray
        Binary matrix $H$ of shape $(m,E)$, $1\le E\le8$. Each column has
        support one or two. Adding a shared boundary vertex when needed must
        give a simple connected graph with $2\le n\le6$ and cycle rank
        $0\le E-n+1\le3$.
    reference : np.ndarray
        Binary reference chain $M$ of shape $(E,)$ in detector-column order.

    Returns
    -------
    physical_errors : np.ndarray
        Binary integer array of shape $(2^{E-n+1},E)$, in ascending relative
        cycle-mask order. Row zero is the supplied reference chain.

    Raises
    ------
    ValueError
        If binary data, shapes, column supports, augmented connectivity,
        simple-graph conditions, cycle rank, or size bounds are invalid.

    Notes
    -----
    All probabilities and counts are dimensionless. Random choices are
    integrated out; no seed, external data, or sampling approximation is used.
    """
    return
```

### Step 2

02_build_oriented_transition

Goal
----
Construct the elementary transition matrix for a fixed tail in the conditional cycle ensemble.

```python
def build_oriented_transition(
    detectors: "np.ndarray",
    probabilities: "np.ndarray",
    reference: "np.ndarray",
    tail: int,
) -> "np.ndarray":
    r"""Construct the elementary transition matrix for a fixed tail in the conditional cycle ensemble.

    Parameters
    ----------
    detectors : np.ndarray
        Binary matrix $H$ of shape $(m,E)$, $1\le E\le8$. Each column has
        support one or two. Adding a shared boundary vertex when needed must
        give a simple connected graph with $2\le n\le6$ and cycle rank
        $0\le E-n+1\le3$.
    reference : np.ndarray
        Binary reference chain $M$ of shape $(E,)$ in detector-column order.
    probabilities : np.ndarray
        Finite edge probabilities $q$ of shape $(E,)$ with $0<q_e<1$.
    tail : int
        Fixed augmented vertex index $0\le t<n$.

    Returns
    -------
    transition : np.ndarray
        Row-stochastic float matrix of shape $(nk,nk)$ with $k=2^{E-n+1}$,
        using the state ordering specified in the scientific background.

    Raises
    ------
    ValueError
        If graph/reference constraints fail, probabilities are mismatched,
        nonfinite or outside the open unit interval, or tail is invalid.

    Notes
    -----
    All probabilities and counts are dimensionless. Random choices are
    integrated out; no seed, external data, or sampling approximation is used.
    """
    return
```

### Step 3

03_compute_work_recording_kernel

Goal
----
Compute the completed-update recording kernel with work coefficients through order three.

```python
def compute_work_recording_kernel(
    detectors: "np.ndarray",
    probabilities: "np.ndarray",
    reference: "np.ndarray",
    spacing: int,
) -> "np.ndarray":
    r"""Compute the completed-update recording kernel with work coefficients through order three.

    Parameters
    ----------
    detectors : np.ndarray
        Binary matrix $H$ of shape $(m,E)$, $1\le E\le8$. Each column has
        support one or two. Adding a shared boundary vertex when needed must
        give a simple connected graph with $2\le n\le6$ and cycle rank
        $0\le E-n+1\le3$.
    reference : np.ndarray
        Binary reference chain $M$ of shape $(E,)$ in detector-column order.
    probabilities : np.ndarray
        Physical priors of shape $(E,)$ with finite $0<q_e<1$.
    spacing : int
        Number of completed updates before recording, $1\le a\le3$.

    Returns
    -------
    coefficients : np.ndarray
        Nonnegative array of shape $(k,k,4)$ in relative-mask order, where
        $k=2^{E-n+1}$ and the last index is binomial work order $r$.
        The order-zero matrix is row stochastic.

    Raises
    ------
    ValueError
        If graph/reference constraints, probabilities or spacing are invalid,
        or the first-return systems have no numerically finite moments.

    Notes
    -----
    All probabilities and counts are dimensionless. Random choices are
    integrated out; no seed, external data, or sampling approximation is used.
    """
    return
```

### Step 4

04_compute_count_work_transfer

Goal
----
Compute joint physical count, terminal-state and work coefficients for every initial cycle.

```python
def compute_count_work_transfer(
    recording_kernel: "np.ndarray", physical_errors: "np.ndarray", n_samples: int
) -> "np.ndarray":
    r"""Compute joint physical count, terminal-state and work coefficients for every initial cycle.

    Parameters
    ----------
    recording_kernel : np.ndarray
        Nonnegative finite array $(k,k,4)$ of binomial work coefficients;
        its order-zero rows sum to one.
    physical_errors : np.ndarray
        Distinct binary physical rows $x_i$ of shape $(k,E)$, $1\le k,E\le8$.
        Row order agrees with the recording kernel.
    n_samples : int
        Number of recorded samples, $1\le N\le3$.

    Returns
    -------
    transfer : np.ndarray
        Table of shape $(J,E+1+4k)$. Columns are the $E$ counts, terminal
        index $j$, then $\mathbb E[\binom{W}{r}\mathbf1_{\{c,j\}}\mid i]$
        for $i=0,\ldots,k-1$ and $r=0,1,2,3$, in that nested order.

    Raises
    ------
    ValueError
        If physical rows, coefficient dimensions/nonnegativity, zeroth-order
        stochasticity or the sample count violate the stated contract.

    Notes
    -----
    All probabilities and counts are dimensionless. Random choices are
    integrated out; no seed, external data, or sampling approximation is used.
    """
    return
```

### Step 5

05_transfer_retained_phase

Goal
----
Transfer a joint retained-state and accumulated-work law through one empirical-feedback phase.

```python
def transfer_retained_phase(
    incoming: "np.ndarray",
    source_samples: int,
    channel_probability: float,
    edge_map: "np.ndarray",
    target_detectors: "np.ndarray",
    target_reference: "np.ndarray",
    passive_states: int,
    target_samples: int,
    target_spacing: int,
) -> "np.ndarray":
    r"""Transfer a joint retained-state and accumulated-work law through one empirical-feedback phase.

    Parameters
    ----------
    incoming : np.ndarray
        Nonnegative finite table $(J,E+6)$ with unique integer keys
        $(b,i,c_0,\ldots,c_{E-1})$, then four binomial work coefficients.
        State bounds are $0\le b<B$, $0\le i<k$, and $0\le c_e\le N$.
        Zeroth coefficients sum to one; rows may be in any order.
    source_samples : int
        Previous phase's sample count $1\le N\le3$.
    channel_probability : float
        Dimensionless depolarizing strength $0<p<1$.
    edge_map : np.ndarray
        Integer permutation of length $E$ mapping source edge to target edge.
    target_detectors : np.ndarray
        Binary matchable detector matrix with $E\le8$, whose augmented graph
        is simple and connected, has two to six vertices and cycle rank at most three.
    target_reference : np.ndarray
        Binary target reference chain of length $E$; cycle rows use relative-mask order.
    passive_states : int
        Number $B$ of inactive-chain cycles, $1\le B\le8$.
    target_samples : int
        Number of new physical observations, between one and three.
    target_spacing : int
        Completed updates preceding each new record, between one and three.

    Returns
    -------
    outgoing : np.ndarray
        Nonnegative table $(J',E+6)$ sorted by the keys $(j,b,d)$.
        Its last four columns contain total-work coefficients, without
        normalizing higher orders or resetting either retained state.

    Raises
    ------
    ValueError
        If any incoming key, dimension, coefficient, probability normalization,
        graph, channel, permutation or sampling parameter is invalid.

    Notes
    -----
    All probabilities and counts are dimensionless. Random choices are
    integrated out; no seed, external data, or sampling approximation is used.
    """
    return
```

### Step 6

06_compute_report_work_transfer

Goal
----
Compute joint physical plurality-report and work coefficients from every initial cycle.

```python
def compute_report_work_transfer(
    recording_kernel: "np.ndarray",
    physical_errors: "np.ndarray",
    logical: "np.ndarray",
    n_samples: int,
) -> "np.ndarray":
    r"""Compute joint physical plurality-report and work coefficients from every initial cycle.

    Parameters
    ----------
    recording_kernel : np.ndarray
        Finite nonnegative binomial-work kernel of shape $(k,k,4)$,
        with a row-stochastic zeroth coefficient.
    physical_errors : np.ndarray
        Distinct binary physical configurations $(k,E)$, $1\le k,E\le8$,
        ordered consistently with the kernel.
    logical : np.ndarray
        Binary map $(r,E)$, $1\le r\le2$, with least significant bit first.
    n_samples : int
        Number of correlated records, $1\le N\le5$.

    Returns
    -------
    report_coefficients : np.ndarray
        Array $(k,2^r,4)$ with entry $(i,g,s)$ equal to
        $\mathbb E[\binom{W}{s}\mathbf1_{\{\widehat g=g\}}\mid i]$.
        Keep the full label range, including unattainable labels.

    Raises
    ------
    ValueError
        If physical rows, logical map, coefficient kernel, stochasticity
        or sample count violates its stated contract.

    Notes
    -----
    All probabilities and counts are dimensionless. Random choices are
    integrated out; no seed, external data, or sampling approximation is used.
    """
    return
```

### Step 7

07_compute_adaptive_work_moments

Goal
----
Compute the physical report law and work coefficients of the retained-state feedback decoder.

```python
def compute_adaptive_work_moments(
    detectors_z: "np.ndarray",
    detectors_x: "np.ndarray",
    reference_z: "np.ndarray",
    reference_x: "np.ndarray",
    syndrome_z: "np.ndarray",
    syndrome_x: "np.ndarray",
    logical_x: "np.ndarray",
    channel_probability: float,
    edge_map: "np.ndarray",
    sample_counts: "np.ndarray",
    spacings: "np.ndarray",
) -> "np.ndarray":
    r"""Compute the physical report law and work coefficients of the retained-state feedback decoder.

    Parameters
    ----------
    detectors_z, detectors_x : np.ndarray
        Binary paired detector matrices with the same edge count $1\le E\le8$.
        Each augmented graph is simple, connected, has two to six vertices
        and cycle rank zero through three; columns have support one or two.
    reference_z, reference_x : np.ndarray
        Binary length-$E$ reference chains in the corresponding edge orders.
    syndrome_z, syndrome_x : np.ndarray
        Binary detector vectors satisfying $H_ZM_Z=s_Z$ and $H_XM_X=s_X$
        modulo two; their lengths match the corresponding detector row counts.
    logical_x : np.ndarray
        Binary final-X logical map $(r,E)$, $1\le r\le2$; row zero is the
        least significant physical-label bit.
    channel_probability : float
        Finite dimensionless depolarizing strength $0<p<1$.
    edge_map : np.ndarray
        Integer permutation $f$ from Z-edge indices to paired X-edge indices.
    sample_counts : np.ndarray
        Four positive integers for $Z_0,X_1,Z_2,X_3$; the first three are
        at most three and the last is at most five.
    spacings : np.ndarray
        Four integers in $[1,3]$, counting completed updates before each record.

    Returns
    -------
    report_coefficients : np.ndarray
        Array $(2^r,4)$ whose entry $(g,s)$ is
        $\mathbb E[\binom{W}{s}\mathbf1_{\{\widehat g=g\}}]$ for the
        total attempted-proposal count $W$. Order-zero entries sum to one.

    Raises
    ------
    ValueError
        If any model, syndrome/reference relation, channel, permutation,
        schedule, or intermediate transition/moment system is invalid.

    Notes
    -----
    All probabilities and counts are dimensionless. Random choices are
    integrated out; no seed, external data, or sampling approximation is used.
    """
    return
```

### Step 8

08_compute_conditional_work_skewness

Goal
----
Compute conditional work skewness for a reported physical logical class.

```python
def compute_conditional_work_skewness(
    detectors_z: "np.ndarray",
    detectors_x: "np.ndarray",
    reference_z: "np.ndarray",
    reference_x: "np.ndarray",
    syndrome_z: "np.ndarray",
    syndrome_x: "np.ndarray",
    logical_x: "np.ndarray",
    channel_probability: float,
    edge_map: "np.ndarray",
    sample_counts: "np.ndarray",
    spacings: "np.ndarray",
    target_label: int,
) -> float:
    r"""Compute conditional work skewness for a reported physical logical class.

    Parameters
    ----------
    detectors_z, detectors_x : np.ndarray
        Binary paired detector matrices with the same edge count $1\le E\le8$.
        Each augmented graph is simple, connected, has two to six vertices
        and cycle rank zero through three; columns have support one or two.
    reference_z, reference_x : np.ndarray
        Binary length-$E$ reference chains in the corresponding edge orders.
    syndrome_z, syndrome_x : np.ndarray
        Binary detector vectors satisfying $H_ZM_Z=s_Z$ and $H_XM_X=s_X$
        modulo two; their lengths match the corresponding detector row counts.
    logical_x : np.ndarray
        Binary final-X logical map $(r,E)$, $1\le r\le2$; row zero is the
        least significant physical-label bit.
    channel_probability : float
        Finite dimensionless depolarizing strength $0<p<1$.
    edge_map : np.ndarray
        Integer permutation $f$ from Z-edge indices to paired X-edge indices.
    sample_counts : np.ndarray
        Four positive integers for $Z_0,X_1,Z_2,X_3$; the first three are
        at most three and the last is at most five.
    spacings : np.ndarray
        Four integers in $[1,3]$, counting completed updates before each record.
    target_label : int
        Physical integer label in $[0,2^r)$ on which work is conditioned.

    Returns
    -------
    skewness : float
        One finite dimensionless conditional third central moment divided by
        conditional variance to the power three halves.

    Raises
    ------
    ValueError
        If any model or schedule condition fails, the target label is invalid,
        its event has zero probability, or the conditional variance is not
        positive and finite. A nonfinite final statistic also raises ValueError.

    Notes
    -----
    All probabilities and counts are dimensionless. Random choices are
    integrated out; no seed, external data, or sampling approximation is used.
    """
    return
```
