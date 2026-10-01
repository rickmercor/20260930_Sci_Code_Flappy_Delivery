# Biology-Biochemistry-9

## Background

Metabolic mass balance usually admits several internal flux configurations, even when uptake and secretion are measured. Reaction-level expression evidence can distinguish those configurations, but kinetic direction restrictions and physiological energy measurements describe different aspects of the biochemical state.

In this task, a candidate denotes an expression profile and a scenario denotes an independently applied expression perturbation. The physiological measurements describe allowable concentrations and uncertainty in standard reaction energies. The requested quantity is a dimensionless measure of the weakest delivery through a selected route across the complete perturbation panel. All inputs are synthetic; the module is a benchmark abstraction rather than a reconstruction of a named organism.

Notation: net reaction flux follows the orientation in the reaction table; a directional pair is ordered forward, reverse; concentration logarithms use a 1 mol L^-1 reference; energy quantities are in kJ mol^-1.

## Problem

A synthetic biochemical network has six transcriptomic candidates and six measured expression scenarios per candidate, with fixed exchange demands and forward-only evidence for three internal reactions. Infer each phenotype-specific steady state using the recent transcriptome-weighted bidirectional entropy treatment that preserves kinetically imposed net directionality, then assess its energy estimates by the treatment's physiological variability analysis. A scenario is calibrated when its physiological region is nonempty and every active internal reaction's inferred energy lies no farther than the stated excess limit outside that reaction's direction-conditioned physiological interval. A candidate is eligible only if all six scenarios are calibrated, and its robust performance is the smallest fraction of the G-export demand carried by R10 across those scenarios. Determine the largest eligible robust performance, with the given tie rules.

The tables below define synthetic inputs in a common normalized flux/expression unit; standard reaction energies are independent uncertain quantities, while concentrations and co-assay contrasts define a joint physiological region for each candidate. Scenario factors apply independently to the original candidate weights, and the same candidate-specific physiological measurements apply in every scenario. Use the table conventions and report enough precision to keep absolute error below 2e-6 in the final fraction.

In the short reasoning, identify the governing source insights and give this compact certificate: the balance rank; the selected candidate's P01/R03 net flux, P00/R12 directional pair, P02/R10 net flux and P00/R07 physiological interval; the first infeasible candidate/scenario in row-major order and that candidate's P00/R03 lower physiological endpoint; the largest finite interval-excess event, with its reaction, inferred energy and physiological interval; the eligible candidates; the runner-up's robust score; and the selected candidate and limiting scenario.

## Numerical configuration

- Metabolite order: A, B, C, D, E, F, G
- Reaction order: R01 through R14; unit substrate and product stoichiometry
- Stoichiometric convention: products minus reactants; S v = b
- Unspecified net directions: free
- Reaction and candidate labels: identifiers

| Reaction | Oriented conversion | Forward-only evidence |
| --- | --- | --- |
| R01 | A to B | No |
| R02 | A to C | No |
| R03 | C to B | Yes |
| R04 | B to D | No |
| R05 | C to D | No |
| R06 | C to E | No |
| R07 | D to E | No |
| R08 | D to F | No |
| R09 | E to F | No |
| R10 | E to G | No |
| R11 | F to G | No |
| R12 | G to B | Yes |
| R13 | F to C | Yes |
| R14 | A to G | No |

| Metabolite | A | B | C | D | E | F | G |
| --- | --- | --- | --- | --- | --- | --- | --- |
| b | -10 | 0 | 0 | 0 | 0 | 2 | 8 |

- Exchange interpretation: input at A; output at F and G
- Expression-weight scope: the 14 internal reactions

### Original reaction weights

| Candidate | R01 | R02 | R03 | R04 | R05 | R06 | R07 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| C01 | 4.8 | 13.1 | 2.4 | 9.7 | 3.9 | 12.6 | 2.1 |
| C02 | 8.4 | 10.2 | 6.3 | 5.1 | 9.8 | 11.3 | 7.4 |
| C03 | 12.2 | 7.6 | 3.1 | 13.4 | 6.8 | 8.9 | 4.2 |
| C04 | 6.1 | 14.8 | 9.2 | 6.3 | 12.1 | 15.7 | 3.6 |
| C05 | 10.7 | 9.3 | 4.6 | 11.9 | 5.2 | 10.8 | 8.1 |
| C06 | 7.9 | 12.4 | 7.1 | 8.6 | 7.7 | 13.9 | 5.5 |

| Candidate | R08 | R09 | R10 | R11 | R12 | R13 | R14 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| C01 | 4.3 | 7.2 | 16.5 | 5.4 | 8.1 | 3.6 | 1.1 |
| C02 | 3.6 | 4.9 | 14.2 | 6.7 | 4.5 | 9.3 | 1.7 |
| C03 | 7.1 | 2.7 | 18.4 | 3.9 | 6.8 | 5.1 | 2.4 |
| C04 | 5.8 | 10.4 | 13.7 | 7.8 | 3.2 | 8.6 | 0.9 |
| C05 | 6.4 | 3.8 | 15.9 | 4.7 | 7.3 | 2.9 | 1.4 |
| C06 | 4.9 | 6.3 | 17.1 | 5.9 | 5.6 | 6.2 | 1.9 |

### Expression scenarios

- Omitted expression factors: exactly 1
- P00: original state

| Scenario | Non-unit reaction factors |
| --- | --- |
| P00 | None |
| P01 | R01: 0.22; R09: 1.80 |
| P02 | R04: 0.28; R14: 1.50 |
| P03 | R06: 0.55; R11: 1.40 |
| P04 | R07: 0.18; R10: 0.75 |
| P05 | R02: 0.35; R04: 1.60 |

### Physiological inputs and decision constants

- Concentration coordinates x_A through x_G: natural logarithms relative to 1 mol L^-1
- Co-assay convention: inclusive interval for the displayed linear contrast of x

| Quantity | Value |
| --- | --- |
| Standard reaction-energy means | 0 kJ mol^-1 for each of R01-R14 |
| Standard reaction-energy standard deviations | 0.04 kJ mol^-1 for each of R01-R14 |
| Confidence level for standard energies | The source's 99% interval convention |
| x bounds | [-10.5, -4.5] for every metabolite and candidate |
| Co-assay H1 | x_B - x_C |
| Co-assay H2 | x_G - x_E |
| Co-assay H3 | x_A - x_B + x_D - x_E |
| Gas constant R | 0.00831446261815324 kJ mol^-1 K^-1 |
| Temperature | 298.15 K |
| Active net-flux threshold | Absolute net flux strictly greater than 1e-7 |
| Minimum driving force for an active reaction | 0.001 kJ mol^-1 |
| Allowed interval excess | 0.10 kJ mol^-1, inclusive |
| Robust-score normalization | G-export demand, 8 flux units |
| Candidate tie rule | Within 1e-10 of the largest eligible score, choose the lowest candidate label |
| Limiting-scenario tie rule | Within 1e-10 of the selected candidate's minimum score, choose the lowest scenario label |

| Candidate | H1 interval | H2 interval | H3 interval |
| --- | --- | --- | --- |
| C01 | [0.18, 0.42] | [-6, 6] | [-12, 12] |
| C02 | [-6, 6] | [-6, 6] | [-12, 12] |
| C03 | [-6, 6] | [-6, 6] | [-12, 12] |
| C04 | [-6, 6] | [-0.58, -0.22] | [-12, 12] |
| C05 | [-6, 6] | [-6, 6] | [2.25, 2.90] |
| C06 | [-6, 6] | [-6, 6] | [-12, 12] |

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

01_perturbed_weights.py

Goal
----
Construct the reaction expression panel.

```python
def perturbed_weights(weights: "np.ndarray", factors: "np.ndarray") -> "np.ndarray":
    """
    weights is a positive finite (C,N) array and factors is a positive finite (P,N)
    array. C, P and N are positive. Return the aligned (C,P,N) reaction-weight panel.
    Each factor is relative to the unperturbed row. Invalid shapes, nonpositive values,
    nonfinite values or product overflow raise ValueError.
    """
    return result
```

### Step 2

02_balance_rows.py

Goal
----
Retain the independent mass balance equations.

```python
def balance_rows(stoichiometry: "np.ndarray", demand: "np.ndarray") -> "np.ndarray":
    """
    stoichiometry is a finite nonempty (M,N) array, with products positive and reactants
    negative. demand is a finite length-M vector in S v = demand. Scan rows in input
    order and retain a row exactly when it increases rank, using an absolute singular-
    value threshold of 1e-10. Return retained rows with demand as the last column, shape
    (rank,N+1). Raise ValueError for invalid inputs or inconsistent equations at that
    rank threshold.
    """
    return result
```

### Step 3

03_infer_net_panel.py

Goal
----
Infer the transcriptome specific net flux panel.

```python
def infer_net_panel(
    balance: "np.ndarray", weights: "np.ndarray", irreversible: "np.ndarray"
) -> "np.ndarray":
    """
    balance is a finite (r,N+1) independent-row system; its last column is demand.
    weights is positive finite (C,P,N). irreversible is a unique integer vector of zero-
    based reaction indices, of length at most six. These indices carry forward-only
    biological evidence; all remaining net directions are unspecified. Use the source-
    matched transcriptomic maximum-entropy treatment: keep a strictly positive forward
    and reverse component f_j and r_j for every reaction, whose difference f_j - r_j is
    that reaction's net flux, and maximise -sum_j [f_j ln(f_j / g_j) + r_j ln(r_j / g_j)]
    with g_j the supplied reaction weight, subject to the balance rows and to
    f_j - r_j >= 0 on the reactions named by irreversible. Return finite signed net fluxes,
    shape (C,P,N). Raise ValueError for invalid inputs or an infeasible state. Inputs
    use the common numerical units declared in the main problem. Numerical acceptance
    uses absolute tolerance 2e-6.

    An unresolved stationary solver raises RuntimeError."""
    return result
```

### Step 4

04_directional_panel.py

Goal
----
Recover the directional flux components.

```python
def directional_panel(nets: "np.ndarray", weights: "np.ndarray") -> "np.ndarray":
    """
    nets and weights are same-shaped nonempty finite arrays; weights is strictly
    positive. Recover the source-matched directional pair for each net value and weight.
    The added last axis is [forward, reverse]; all earlier axes are unchanged. Raise
    ValueError for invalid inputs or unrepresentable positive directional values.
    """
    return result
```

### Step 5

05_energy_panel.py

Goal
----
Evaluate the directional energy quantities.

```python
def energy_panel(
    directional: "np.ndarray", gas_constant: float, temperature: float
) -> "np.ndarray":
    """
    directional is a positive finite nonempty array whose last axis is [forward,
    reverse]. gas_constant is positive and finite in kJ mol^-1 K^-1; temperature is
    positive and finite in K. Return the source directional energy statistic, in
    kJ mol^-1, with the last pair axis removed. The pipeline compares this statistic
    only at active net reactions. Its value at a blocked net reaction does not
    determine that reaction's physiological Gibbs energy. Invalid inputs raise
    ValueError.
    """
    return result
```

### Step 6

06_active_directions.py

Goal
----
Identify active reaction directions.

```python
def active_directions(
    nets: "np.ndarray", activity_threshold: float
) -> "np.ndarray":
    """
    nets is any nonempty finite array. activity_threshold is positive and finite in net-
    flux units. Return a same-shaped numeric array with +1 or -1 for positive or
    negative active coordinates, and 0 when the absolute net value is at most the
    threshold. Invalid inputs raise ValueError.
    """
    return result
```

### Step 7

07_physiological_ranges.py

Goal
----
Calculate physiological energy intervals.

```python
def physiological_ranges(
    stoichiometry: "np.ndarray",
    directions: "np.ndarray",
    standard_mean: "np.ndarray",
    standard_sd: "np.ndarray",
    log_bounds: "np.ndarray",
    contrasts: "np.ndarray",
    contrast_bounds: "np.ndarray",
    gas_constant: float,
    temperature: float,
    driving_floor: float,
) -> "np.ndarray":
    """
    Let stoichiometry have shape (M,N), directions have shape (C,P,N) with entries
    -1,0,+1, and standard_mean and standard_sd have length N in kJ mol^-1. Standard
    uncertainties use the source 99% interval convention; standard_sd is nonnegative.
    log_bounds has shape (C,M,2), and bounds natural logarithms of concentrations
    relative to 1 mol L^-1. contrasts has shape (H,M); contrast_bounds has shape
    (C,H,2). Every last pair is [lower,upper], including equality bounds. Apply the
    source physiological variability analysis, adding the supplied contrast
    inequalities. R, T and driving_floor are positive finite quantities in kJ mol^-1
    K^-1, K and kJ mol^-1. Return shape (C,P,1+2N): feasible flag, N lower endpoints, N
    upper endpoints. If the joint physiological region is empty, return an all-zero row.
    Otherwise the flag is 1 and endpoints are computed for every reaction. Invalid
    shapes, indices, signs, bounds or nonfinite data raise ValueError; an unresolved
    numerical solver status raises RuntimeError.
    """
    return result
```

### Step 8

08_compatibility_panel.py

Goal
----
Calibration compares active reaction estimates with their physiological variability intervals. A scenario is acceptable only when its physiological state space exists and its largest active interval excess meets the stated limit.

```python
def compatibility_panel(
    energies: "np.ndarray",
    directions: "np.ndarray",
    ranges: "np.ndarray",
    allowed_excess: float,
) -> "np.ndarray":
    """
    energies and directions have shape (C,P,N); directions contains -1,0,+1. ranges has
    shape (C,P,1+2N) in the preceding range convention. allowed_excess is finite and
    nonnegative in kJ mol^-1. Return (C,P,2) rows [compatible flag, largest active
    distance outside a range]. Compatibility uses an inclusive excess bound and the
    feasibility flag. The excess is 0 for infeasible rows or rows with no active
    coordinates; the feasibility flag still controls compatibility. Invalid data raise
    ValueError.
    """
    return result
```

### Step 9

09_select_robust_candidate.py

Goal
----
Select the robust candidate.

```python
def select_robust_candidate(
    nets: "np.ndarray",
    compatibility: "np.ndarray",
    target: int,
    normalization: float,
) -> "np.ndarray":
    """
    nets is finite (C,P,N), compatibility is finite (C,P,2) with binary flags in its
    first component, target is a zero-based integer reaction index, and normalization is
    positive finite. A candidate is eligible when every scenario is compatible. Its
    score is its minimum signed target net flux divided by normalization. Select the
    largest eligible score; candidates within 1e-10 of the largest score tie, with the
    smallest row index selected. Within the selected candidate, scenarios within 1e-10
    of its minimum score tie, with the smallest scenario index selected. Return one
    vector: selected candidate index, selected limiting scenario index, selected score,
    C eligibility flags, C scores (including ineligible rows). Invalid inputs or no
    eligible candidate raise ValueError.
    """
    return result
```

### Step 10

10_resolve_metabolic_panel.py

Goal
----
Resolve the complete metabolic panel.

```python
def resolve_metabolic_panel(data: dict | None = None) -> float:
    """
    data is None for the numerical fixture in the main problem, or a dictionary with
    exactly these keys: S, b, weights, factors, irreversible, standard_mean,
    standard_sd, log_bounds, contrasts, contrast_bounds, R, T, tau, epsilon, excess,
    target, normalization. Their shapes, units and valid domains are the corresponding
    inputs in steps 1-9. Compose all preceding public functions in order and use their
    returned values. Return the selected robust score as one float. Invalid inputs or no
    eligible candidate raise ValueError. The default raw-input constructor is supplied
    in this item; it contains no precomputed answers. Numerical acceptance uses absolute
    tolerance 2e-6. Unresolved numerical solver failures propagate as RuntimeError.
    """
    return result
```
