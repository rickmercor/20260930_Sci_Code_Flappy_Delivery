# Physics-Optics-28

## Background

Gaussian optical elements offer reproducible control over continuous-variable light, while useful non-Gaussian resource states are often prepared conditionally. In a layered source, a classical measurement record and a surviving quantum state have different physical roles: the former selects later settings, while the latter remains part of the optical apparatus. Production probability and conditional state quality are therefore distinct experimental figures of merit. Vacuum-environment attenuation is a coherent open-system process because the unobserved environment is traced rather than measured, so surviving optical coherences can remain relevant. Finite detection efficiency and optical attenuation matter even when every prepared input is pure, and comparisons between source architectures depend on which optical resources remain available to each architecture.

## Problem

## Setup

A two-layer optical source uses photon-number measurements and outcome-dependent Gaussian optics to prepare a squeezed odd cat, with the following task-defined preparation and hardware losses.

## Inputs

All phases are in radians, squeezing parameters and probabilities are dimensionless, and the accepted records are first count 1 followed by second count 2, or first count 2 followed by second count 1.

| Initial vacuum mode | Squeezing magnitude | Phase |
|---|:---:|:---:|
| 0 | 0.50 | 0.18 |
| 1 | 0.41 | 2.72 |
| 2 | 0.46 | 0.39 |

| Preparation mixer, in temporal order | Angle | Phase |
|---|:---:|:---:|
| Modes (0,1) | 0.63 | 0.27 |
| Modes (1,2) | 0.51 | 1.03 |
| Modes (0,2) | 0.37 | -0.42 |

| Intensity efficiency | Value |
|---|:---:|
| First number detector | 0.99 |
| Memory of original mode 1 | 0.98 |
| Memory of original mode 2 | 0.95 |
| Second number detector | 0.99 |
| Final signal transmission | 0.99 |

Target: \(S(0.5,0)|C_-(\sqrt{6})\rangle\), with \(|C_-(\alpha)\rangle\) proportional to \(|\alpha\rangle-|-\alpha\rangle\)

## Physical model

The three vacua are squeezed and undergo the listed preparation mixers before mode 0 is destructively measured, after which original modes 1 and 2 pass through their respective memories and are relabeled as retained modes 0 and 1.

Each first count selects a Gaussian operation \(G\) on these two retained modes, with independent controls \(0\leq\theta_a,\theta_b\leq\pi/2\), \(0\leq r_0,r_1\leq0.5\), and unrestricted periodic phases:

\[
G=B(\theta_b,\varphi_b)\,[S(r_0,\phi_0)\otimes S(r_1,\phi_1)]\,B(\theta_a,\varphi_a)
\]

All losses couple to independent vacuum environments, detectors have no dark counts, other records are discarded, and after \(G\) retained mode 1 is destructively measured while retained mode 0 undergoes final signal loss.

Gate conventions, with \(a,b\) ordered as the indicated mode pair:

\[
S(r,\phi)=\exp\!\left[\tfrac12\left(r e^{-i\phi}a^2-r e^{i\phi}a^{\dagger2}\right)\right],\qquad
B(\theta,\varphi)=\exp\!\left[\theta\left(e^{i\varphi}a^\dagger b-e^{-i\varphi}ab^\dagger\right)\right]
\]

If \(\sigma_1,\sigma_2\) are the unnormalized final signal states for the accepted records, define \(P=\operatorname{tr}(\sigma_1+\sigma_2)\), \(F=\langle\psi_{\rm target}|(\sigma_1+\sigma_2)|\psi_{\rm target}\rangle/P\), and the source reward \(R=F+P\), with zero reward when \(P=0\).

## Task

**Compute the maximum adaptive reward \(R_{\rm ad}^{\star}\).**

In the short reasoning, report the total accepted-record success probability
\(P^\star\) and pooled conditional fidelity \(F^\star\) at the same maximizing
adaptive instrument, so that the reported numbers satisfy
\(R_{\rm ad}^{\star}=P^\star+F^\star\). Preserve each accepted record's
unnormalized incident-attempt weight through final loss, pool the two final
signal states, normalize only once when evaluating \(F^\star\), and briefly
state whether a fixed-policy finite-Fock reevaluation at the maximizing
controls agrees with the reported \(P^\star\), \(F^\star\), and
\(R_{\rm ad}^{\star}\) within the joint error budget; also characterize what
remains in the retained modes after the first count and how that count selects
the later Gaussian controller.

Using the primary source only for its reported odd-cat comparison, state whether
that comparison retains an adaptive advantage under 10% loss without loss-aware
redesign and explain the source's stated mechanism for the result. For the
prompt-defined independent-vacuum attenuation, also state with a brief physical
reason whether the untruncated channel at strictly intermediate transmission
admits a Kraus decomposition in which every Kraus operator has operator rank one.

## Numerical conventions

Use the finite Fock model \(n=0,\ldots,27\) in each mode with tensor indices \(|i,j,k\rangle\mapsto28^2i+28j+k\) and \(|i,j\rangle\mapsto28i+j\), project the gate generators before exponentiation, normalize the truncated odd-cat vector before applying its squeezer, preserve unnormalized record weights, and achieve one joint absolute error budget of 0.0004 for \(P^\star\), \(F^\star\), and \(R_{\rm ad}^{\star}\) relative to their true defined values, accepting equivalent maximizing instruments without prescribing optimizer coordinates.

## Output format

```
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

gaussian gate

Goal
----
Construct a finite-Fock squeezing or beam-mixer Gaussian unitary and optionally apply it to amplitude columns.

```python
def gaussian_gate(d: int, kind: str, parameters: object, columns: object = None) -> object:
    """Return a finite-Fock Gaussian unitary as a complex ndarray.

    d is an integer in [2, 40], excluding bool. kind is 'squeeze' or 'mix'.
    parameters is a length-2 real numeric sequence, not complex or string data.
    For squeeze it is (r, phi), with 0 <= r <= 1 and any finite phase.
    For mix it is (theta, phi), with 0 <= theta <= pi/2 and any finite phase.
    S = exp((conj(z)*a*a-z*a.H*a.H)/2), z=r*exp(1j*phi).
    B = exp(theta*(exp(1j*phi)*a.H tensor a-exp(-1j*phi)*a tensor a.H)).
    a[n-1,n]=sqrt(n) on 0,...,d-1. The tensor order is |i,j> -> i*d+j.
    Return shape (d,d) for S or (d*d,d*d) for B. Generators are projected
    before exponentiation. Alternatively columns is a finite numeric real or
    complex matrix (dimension,rank), rank>=1, with squared Frobenius norm at
    most 1+1e-10, and the return is U@columns with that same shape. The two
    forms are equivalent. Each cell must agree to relative 1e-9 or absolute
    1e-12. Raise ValueError for any violation or nonfinite input.
    Returns
    -------
    Return `(d,d)` for S or `(d*d,d*d)` for B, or the same shape as supplied amplitude columns for U@columns. Every cell is a matrix element or amplitude.
    """
    return None
```

### Step 2

attenuate

Goal
----
Apply quantum-limited vacuum attenuation to one selected mode of a one- or two-mode density matrix.

```python
def attenuate(rho: object, eta: float, mode: int = 0, modes: int = 1) -> object:
    """Return the complex density matrix after single-mode vacuum attenuation.

    rho is a finite real/complex numeric square matrix of size d**modes, where
    d is in [2,40]. modes is integer 1 or 2 and mode is an integer in
    [0,modes-1], excluding booleans. Tensor order is lexicographic in mode
    occupations. eta is a finite real numeric scalar in [0,1], not complex,
    boolean or string data. rho obeys the shared physicality convention:
    max(abs(rho-rho.H))<=1e-10, trace real part in [-1e-10,1+1e-10], and
    Hermitian-part minimum eigenvalue >=-1e-10. Interpret it as C(H), the
    disclosed trace-preserving PSD canonicalization, before propagation.
    K_l[n-l,n]=sqrt(comb(n,l)*(1-eta)**l*eta**(n-l)); act with K_l only on
    mode, sum K_l C(H) K_l.H, and retain the original record mass, not unit
    probability. Return the same shape, including coherences. All cells must
    agree to relative 1e-9 or absolute 1e-12. Raise ValueError for any invalid
    kind, shape, discrete argument, nonfinite value or physical/domain failure.
    Returns
    -------
    Return the same square density shape as the input, preserving its physical trace. All coherences and both tensor axes are included in the output budget.
    """
    return None
```

### Step 3

herald

Goal
----
Condition on an inefficient photon-number-resolving count and trace out the detected mode while preserving unnormalized branch weight.

```python
def herald(state: object, count: int, efficiency: float, detected_mode: int = 0, factorized: bool = False, modes: int = 2) -> object:
    """Return the unnormalized retained density matrix after a PNR record.

    modes is integer 2 or 3, detected_mode is integer in [0,modes-1], and
    count is a nonnegative integer, all excluding booleans. Local dimension
    d is in [2,40] and tensor order is lexicographic. factorized must be bool
    or numpy.bool_. If False, state is a numeric physical (d**modes,d**modes)
    matrix interpreted as shared C(H). If True, state is a finite real/complex
    numeric array (d**modes,rank), rank>=1, representing state@state.H with
    squared Frobenius norm <=1+1e-10. No particular factor representation is
    required. efficiency is finite real numeric in [0,1], not complex/string/
    bool. Detection has diagonal effect comb(n,count)*efficiency**count*
    (1-efficiency)**(n-count) for n>=count and zero otherwise. Trace out only
    detected_mode, keeping the other modes in their original order and all
    their coherences. Return shape (d**(modes-1),d**(modes-1)), not a normalized
    state. Counts >=d and zero-probability outcomes return zero. Cells agree to
    relative 1e-9 or absolute 1e-12. Raise ValueError for every kind, shape,
    finite-data, physicality or discrete/continuous domain violation.
    Returns
    -------
    Return `(d**(modes-1),d**(modes-1))`, in the surviving modes' original order. The trace includes the incoming branch probability. Impossible counts give the zero matrix, not an exception, even at zero efficiency.
    """
    return None
```

### Step 4

retained states

Goal
----
Prepare the squeezed three-mode source, herald the first count, and apply both memory losses to the retained pair.

```python
def retained_states(d: int, squeezers: object, mixers: object, counts: object, efficiencies: object) -> object:
    """Return the first-record ensemble after both retained-mode memories.

    d is integer in [2,40], excluding bool. squeezers and mixers are finite
    real numeric (3,2) arrays, not complex/string/bool data. Each squeezer row
    is (r,phase), 0<=r<=1. Rows act on initial vacuum modes 0,1,2. Each mixer
    row is (theta,phase), 0<=theta<=pi/2, in temporal order (0,1),(1,2),(0,2).
    All phases are unrestricted finite radians. counts contains one or two
    DISTINCT nonnegative integer first records, excluding booleans, in return
    order. efficiencies is a finite real numeric length-3 array in [0,1]:
    first detector, memory on original mode 1, memory on original mode 2.
    Apply the given preparation, destructively measure original mode 0, then
    apply the two local attenuation channels. Return complex array
    (len(counts),d*d,d*d), unnormalized, with retained order |i,j> -> i*d+j
    for original modes 1,2. Zero-probability records remain zero rows. Each
    density cell agrees to relative 1e-9 or absolute 1e-12. Raise ValueError
    for invalid kinds, shapes, nonfinite entries, duplicate counts or any
    stated discrete/continuous domain violation.
    Returns
    -------
    Return complex shape `(len(counts),d*d,d*d)` in first-count order. Every density slice has its own physical probability in its trace. Zero-probability records remain full zero slices, with no padding or omitted cells.
    """
    return None
```

### Step 5

feedforward state

Goal
----
Apply one count-selected two-mode mixer-squeezers-mixer Gaussian controller while preserving branch mass.

```python
def feedforward_state(state: object, controls: object, factorized: bool = False) -> object:
    """Apply one two-mode Gaussian controller, preserving physical branch mass.

    controls is a finite real numeric length-8 array in the order
    (theta_a,phase_a,r0,phi0,r1,phi1,theta_b,phase_b). Angles are in [0,pi/2],
    both squeezing magnitudes in [0,0.5], and all phases are unrestricted.
    Complex, string and boolean controls are invalid. The operation is
    G=B_b (S0 tensor S1) B_a with the shared finite-Fock gate conventions.
    factorized is bool or numpy.bool_. If False, state is a physical numeric
    (d*d,d*d) matrix, d in [2,40], interpreted as shared C(H), and the result
    is G C(H) G.H of that shape. If True, state is finite numeric real/complex
    amplitude columns (d*d,rank), rank>=1, squared Frobenius norm<=1+1e-10,
    and the result is G@state with the same shape. No factorization convention
    is imposed. The modes retain their order and there is no detector or
    additional loss in this operation. Exact zero stays zero. Preserve the
    input mass, including accepted roundoff and small positive mass, not unit
    probability. Every returned cell agrees to relative 1e-9 or absolute
    1e-12. Raise ValueError for all invalid kinds, shapes, finite-data,
    physicality, flag or control-domain conditions.
    Returns
    -------
    For density input return `G C(H) G.H` of shape `(d*d,d*d)`. For amplitude columns return G@state with the original shape, rank and physical amplitude. An exact zero remains zero without normalizing by zero.
    """
    return None
```

### Step 6

policy statistics

Goal
----
Compute each labeled record's joint success probability and target overlap after its controller, second herald, and final loss.

```python
def policy_statistics(states: object, target: object, controls: object, counts: object, efficiencies: object, factorized: bool = False) -> object:
    """Return the (P_k,M_k) array for one or two labeled first records.

    factorized is bool or numpy.bool_. With False, states is finite numeric
    real/complex shape (b,d*d,d*d), b=1 or 2, d in [2,40], containing physical
    matrices interpreted individually by shared C(H). Their total real trace
    is <=1+1e-10. With True, states has shape (b,d*d,rank), rank>=1, and is an
    ensemble of amplitude factors with total squared norm<=1+1e-10. Zero
    columns may pad different factor ranks. No factor convention is required.
    target is a finite numeric real/complex length-d vector with squared norm
    within 1e-10 of one. controls is a real numeric (b,8) array of controllers
    with feedforward_state domains. counts has b nonnegative integers,
    excluding booleans, specifying SECOND counts paired in the same order as
    states. Repeated second counts are allowed. efficiencies is a finite real
    numeric length-2 array in [0,1]: second detector, final signal transmission.
    Apply each G, destructively measure retained mode 1, then attenuate mode 0.
    Row k is (trace(sigma_k),target.H@sigma_k@target), including first-record
    mass. Both columns are real and zero when that record has zero success.
    Never average normalized branches equally. Cells agree to relative 1e-9
    or absolute 1e-12. Raise ValueError for every kind, shape, finite-data,
    physicality, normalization, total-mass, flag or stated domain violation.
    Returns
    -------
    Return real shape `(b,2)`, row k exactly `(P_k,M_k)`. Both cells are zero for a zero-success branch. All 2*b cells are specified, with no unused padding.
    """
    return None
```

### Step 7

optimal reward

Goal
----
Optimize the pooled source reward over common or adaptive Gaussian controllers with the supplied inline-squeezer bound.

```python
def optimal_reward(states: object, target: object, counts: object, efficiencies: object, rmax: float, adaptive: bool) -> float:
    """Return the optimal physical pooled reward as one real scalar.

    states, target, counts and efficiencies have the density-input domains
    of policy_statistics. rmax is a finite real numeric scalar in [0,0.5],
    not complex/string/bool. adaptive is bool or numpy.bool_. Each G has
    both mixing angles in [0,pi/2], both squeezing magnitudes in [0,rmax],
    and unrestricted periodic phases. A common G is required for every
    record if adaptive is False, otherwise the G settings may differ.
    Interpret accepted states with shared deterministic C(H) BEFORE any
    normalized fidelity or optimization, preserving their physical masses.
    For each policy, sum the P_k and M_k columns and use R=P+M/P when P>0,
    or zero at exactly P=0. No positive-probability floor is permitted.
    The optimum is the supremum if it is approached only at zero success.
    Return only the value, not coordinates. The absolute error from the true
    bounded-domain optimum, including iterative and evaluation errors
    together, must be <=0.0004. No optimizer, seed, coordinates or valid
    representation is prescribed. Raise ValueError for all inherited invalid
    input conditions, invalid rmax or non-boolean adaptive.
    Returns
    -------
    Return one finite real optimal value with total absolute error <=0.0004. No policy, seed, optimizer prescription, certificate format or iteration history is returned. Tests observe the scalar budget directly.
    """
    return None
```

### Step 8

source reward

Goal
----
Return the optimized pooled reward for one declared optical-source configuration.

```python
def source_reward(configuration: str = "benchmark") -> float:
    """Return one finite real optimal reward for the named source configuration.

    configuration is a string in {"benchmark", "vacuum", "common", "adaptive",
    "rare_detector"}; it defaults to "benchmark". Other values or types raise
    ValueError. Each configuration is completely specified below. Angles and
    phases are in radians; mixer rows act on (0,1),(1,2),(0,2) in that order.

    "benchmark": d=28; squeezers=((.50,.18),(.41,2.72),(.46,.39));
    mixers=((.63,.27),(.51,1.03),(.37,-.42)); first_counts=(1,2);
    second_counts=(2,1); efficiencies=(.99,.98,.95,.99,.99);
    target_spec=(sqrt(6),.5,0); rmax=.5; adaptive=True.

    "vacuum": d=3; all three squeezer rows=(0,0);
    mixers=((.31,.2),(.27,-.4),(.19,.7)); first_counts=(1,);
    second_counts=(0,); efficiencies=(.96,.91,.93,.89,.94);
    target_spec=(1.1,.2,.1); rmax=.18; adaptive=True.

    "common" and "adaptive": d=4;
    squeezers=((.27,.1),(.18,-.5),(.33,.7));
    mixers=((.34,-.1),(.22,.6),(.41,-.4)); first_counts=(0,1);
    second_counts=(1,0); efficiencies=(.91,.88,.93,.90,.95);
    target_spec=(1.3,.22,.1); rmax=.15. adaptive is False for "common"
    and True for "adaptive".

    "rare_detector": d=3 with the benchmark squeezer and mixer arrays;
    first_counts=(1,); second_counts=(2,); efficiencies=(1,1,1,1e-200,1);
    target_spec=(.9,0,0); rmax=0; adaptive=True.

    The five efficiencies are first detector, the two retained memories,
    second detector, final signal loss. target_spec=(alpha,r,phase) means
    the normalized truncated odd coherent-state superposition at real
    alpha, followed by the projected single-mode squeeze (r,phase).
    The controller and pooled reward have the optimal_reward conventions.
    Preserve first-record weights relative to incident preparation attempts.
    A positive success probability has no floor, even if its final numeric
    representation underflows. Use the same shared C(H) state interpretation.

    Returns
    -------
    One finite real value within 0.0004 of the configured bounded-domain
    optimum, including both evaluation and optimization errors.
    """
    return None
```
