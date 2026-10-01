# Biology-Biochemistry-21

## Background

Cells redistribute enzyme mass when nutrient supply changes, so a high-growth proteome cannot be scaled uniformly and still represent the same biochemical state. The double-graded treatment used here separates pathway-specific kinetic pressure from sector-specific restriction and supplies linked enzyme and metabolite responses.

Fluxes are then inferred from reaction capacity rather than from stoichiometry alone. Producing flux sums act as metabolic context for apparent enzyme saturation, while mass balance and irreversible capacity bounds keep the fitted state physically interpretable. The final thermodynamic test asks whether one biochemical uncertainty state can support the whole nutrient panel; fitting each state separately would answer a weaker question.

All measurements in this task are synthetic. Reaction fluxes, allocations and concentrations are normalized quantities; energy entries use a common consistent unit. A robust score measures the weakest product yield after cost, not the largest product flux in one favorable state.

## Problem

A synthetic eight-reaction carbon-conversion module has six enzyme-allocation designs and four nutrient states; infer each design's condition-specific enzyme abundances with the recent double-graded growth-response treatment, fit its fluxes with the recent flux-sum/logit capacity treatment, test all four states against one thermodynamic uncertainty vector shared by that design, retain only designs whose worst kinetic residual and shared thermodynamic maximum meet their limits, and determine the largest eligible smallest-state R06 product yield after design cost, using the stated tie rule.

The order of every row and column is fixed by the tables; design multipliers modify the original high-growth allocation and are normalized once before any nutrient-state response is applied, the four nutrient states use the same normalized design allocation, kinetic constants, regression coefficients and thermodynamic uncertainty vector, and concentrations are obtained from the double-graded metabolite response before the thermodynamic calculation.

Use the source definitions exactly: the growth-response treatment uses its kinetic factor, mass-weighted network average, enzyme response and metabolite response; the capacity treatment uses producing flux sums, its logit saturation model and its capacity-residual fit; the benchmark adds the stated quadratic regularizer; and the task-specific thermodynamic bridge is applied after both source methods.

For the kinetic fit, enforce the displayed steady-state parameterization and all bounds, run all three stated starts, retain the successful feasible result with the smallest objective and use the lexicographically smallest independent-variable vector for objectives within 1e-12; for the thermodynamic minimax, run all five stated starts, retain the successful feasible result with the smallest maximum and use the lexicographically smallest uncertainty vector for maxima within 1e-12.

The tables contain synthetic normalized biochemical measurements; candidate, reaction and nutrient-state labels are identifiers; both eligibility limits are inclusive, while thermodynamic activity requires flux strictly greater than the activity threshold.
In the short reasoning, identify the two source methods and the task-specific thermodynamic bridge, and report this compact certificate: the six mass-weighted kinetic averages; the six worst kinetic residuals and C03's kinetic-gate outcome at the inclusive 0.595 limit; the six shared thermodynamic maxima; C02's largest independently fitted thermodynamic maximum; the six robust yields and six cost-normalized scores; the eligible designs and C06's pass/fail status for each of the inclusive 0.595 worst-residual limit and inclusive 0.35 shared-thermodynamic-maximum limit; C06's four R06 fluxes and its limiting-yield state; the shared-thermodynamic limiting state and reaction for C06; the runner-up, winning margin and selected design; and the winner obtained when each of the shared-thermodynamic gate, kinetic-residual gate and cost normalization is separately removed.

## Network and fixed order

| Reaction | Conversion |
| --- | --- |
| R01 | external carbon to A |
| R02 | A to B |
| R03 | A to C |
| R04 | B to D |
| R05 | C to D |
| R06 | D to product |
| R07 | B to waste |
| R08 | C to waste |

Stoichiometric convention: products minus reactants. All eight reactions are irreversible.

| Metabolite | R01 | R02 | R03 | R04 | R05 | R06 | R07 | R08 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| A | 1 | -1 | -1 | 0 | 0 | 0 | 0 | 0 |
| B | 0 | 1 | 0 | -1 | 0 | 0 | -1 | 0 |
| C | 0 | 0 | 1 | 0 | -1 | 0 | 0 | -1 |
| D | 0 | 0 | 0 | 1 | 1 | -1 | 0 | 0 |

At uptake u and independent variables x = (v_R02, v_R04, v_R05), use

v = [u, x1, u-x1, x2, x3, x2+x3, x1-x2, u-x1-x3].

Require 0 <= v_j <= E_j kmax_j for every reaction.

## High-growth kinetic inputs

| Reaction | Original allocation | Affinity K | Turnover kappa | Pathway weight B | Sector | kmax | Standard energy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| R01 | 0.160 | 0.180 | 54 | 1.00 | S0 | 11.0 | -9.3 |
| R02 | 0.130 | 0.055 | 31 | 0.82 | S0 | 12.5 | -3.0 |
| R03 | 0.120 | 0.092 | 47 | 0.82 | S0 | 12.0 | -2.5 |
| R04 | 0.110 | 0.041 | 28 | 0.73 | S2 | 13.5 | -2.2 |
| R05 | 0.105 | 0.077 | 39 | 0.73 | S2 | 13.0 | -1.8 |
| R06 | 0.150 | 0.032 | 22 | 0.95 | S2 | 10.5 | -20.3 |
| R07 | 0.115 | 0.068 | 18 | 0.61 | S1 | 8.5 | -15.3 |
| R08 | 0.110 | 0.049 | 25 | 0.61 | S1 | 8.0 | -20.0 |

## Design multipliers

| Design | R01 | R02 | R03 | R04 | R05 | R06 | R07 | R08 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| C01 | 1.02 | 1.16 | 0.91 | 1.08 | 0.93 | 1.05 | 0.88 | 1.12 |
| C02 | 0.96 | 1.04 | 1.12 | 0.94 | 1.10 | 1.08 | 1.03 | 0.91 |
| C03 | 1.05 | 0.91 | 1.18 | 1.13 | 0.90 | 0.97 | 1.08 | 0.94 |
| C04 | 0.93 | 1.14 | 0.98 | 1.02 | 1.12 | 1.04 | 0.92 | 1.07 |
| C05 | 1.01 | 1.08 | 1.05 | 1.10 | 1.02 | 1.09 | 0.95 | 0.96 |
| C06 | 1.07 | 0.97 | 1.09 | 0.96 | 1.06 | 1.02 | 1.10 | 0.90 |

| Design | S0 restriction | S1 restriction | S2 restriction | Energy shift | Cost |
| --- | ---: | ---: | ---: | ---: | ---: |
| C01 | 1.00 | 0.05 | 0.88 | 0.00 | 1.0000 |
| C02 | 0.94 | 0.08 | 1.02 | 0.00 | 1.0000 |
| C03 | 1.08 | 0.02 | 0.91 | -0.12 | 0.9400 |
| C04 | 0.89 | 0.12 | 1.10 | 0.00 | 1.0000 |
| C05 | 0.98 | 0.04 | 0.95 | 0.00 | 1.0575 |
| C06 | 1.04 | 0.06 | 0.86 | 0.00 | 0.9732 |

## Nutrient states and metabolites

| State | Growth ratio | R01 uptake |
| --- | ---: | ---: |
| P00 | 1.00 | 0.82 |
| P01 | 0.78 | 0.70 |
| P02 | 0.55 | 0.57 |
| P03 | 0.34 | 0.46 |

High-growth metabolome-to-proteome mass fraction: 0.12.

| Metabolite | High-growth concentration | Sector | L coordinate 1 | L coordinate 2 |
| --- | ---: | ---: | ---: | ---: |
| A | 0.0060 | S0 | 0.60 | 0.10 |
| B | 0.0038 | S0 | -0.25 | 0.55 |
| C | 0.0044 | S0 | 0.20 | -0.50 |
| D | 0.0029 | S2 | -0.45 | -0.15 |

## Flux-sum/logit coefficients

For reaction j, use the source producing-flux-sum/logit model with intercept beta_j and the four coefficients ordered A, B, C, D.

| Reaction | beta | A | B | C | D |
| --- | ---: | ---: | ---: | ---: | ---: |
| R01 | -0.30 | 0.35 | -0.12 | 0.08 | 0.04 |
| R02 | -0.10 | 0.16 | 0.25 | -0.18 | 0.09 |
| R03 | -0.22 | 0.18 | -0.14 | 0.27 | 0.07 |
| R04 | 0.05 | -0.05 | 0.22 | 0.03 | 0.28 |
| R05 | -0.08 | 0.02 | -0.03 | 0.24 | 0.31 |
| R06 | -0.18 | 0.08 | 0.04 | 0.06 | 0.38 |
| R07 | 0.12 | -0.09 | 0.21 | -0.02 | 0.05 |
| R08 | 0.09 | 0.03 | -0.05 | 0.19 | 0.02 |

The regularized flux-fit weight is 0.012.

## Numerical rules for the flux fit

For each design and state, minimize the squared capacity-residual objective plus the task-specific quadratic flux penalty w sum_j v_j^2 over x. Use SciPy SLSQP with ftol = 1e-13 and maxiter = 2000. Use these starts in order:

| Start | x1 | x2 | x3 |
| --- | ---: | ---: | ---: |
| 1 | 0.50u | 0.25u | 0.25u |
| 2 | 0.70u | 0.50u | 0.15u |
| 3 | 0.30u | 0.12u | 0.50u |

Discard a result if SLSQP reports failure or any flux or capacity inequality is violated by more than 2e-8. The state residual used for eligibility is the root mean square of the eight unregularized capacity residuals.

## Shared thermodynamic bridge

For design d, state s and reaction j, use

corrected_energy_dsj = standard_energy_j + energy_shift_d + RT (S^T log concentration_ds)_j + driving_force + RT (S^T L delta_d)_j.

One delta_d is shared across all four states of design d and must satisfy delta_d,1^2 + delta_d,2^2 <= 1. Minimize t_d subject to corrected_energy_dsj <= t_d for every active reaction in every state. Use SLSQP with ftol = 1e-13 and maxiter = 2000 from delta = (0,0), (1,0), (-1,0), (0,1), (0,-1); initialize t at the largest corrected energy for that delta. Discard a result if SLSQP reports failure or an inequality is violated by more than 2e-8. The limiting state and reaction are the lowest (state, reaction) pair among active reactions whose corrected energy is within 1e-9 of the shared maximum.

| Constant | Value |
| --- | ---: |
| RT | 2.4789570296 |
| Driving-force addition | 0.08 |
| Active-flux threshold | 1e-8 |
| Worst RMS residual limit | 0.595 |
| Shared thermodynamic maximum limit | 0.35 |
| Design-score tie tolerance | 1e-10 |

The robust yield of a design is min_s(v_R06,ds / uptake_s), its score is robust yield divided by design cost, the lowest design label wins among eligible scores within the tie tolerance of the maximum, and the runner-up uses the same rule after removing the winner.

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

Kinetic Factors

Goal
----
Calculate source-defined kinetic factors for the enzyme panel.

```python
def kinetic_factors(
    affinity: "np.ndarray",
    turnover: "np.ndarray",
    pathway_weight: "np.ndarray",
) -> "np.ndarray":
    """Calculate kinetic factors.

    Parameters
    ----------
    affinity, turnover, pathway_weight
        Finite positive one-dimensional arrays with identical shape.

    Returns
    -------
    np.ndarray
        A float64 vector with one kinetic factor per reaction.

    Raises
    ------
    ValueError
        If an input is not one-dimensional, finite and positive, or if the
        three shapes differ.
    """
    return result
```

### Step 2

Growth Response

Goal
----
Calculate double-graded enzyme and metabolite responses.

```python
def growth_response(
    phi_high: "np.ndarray",
    kinetic_factor: "np.ndarray",
    reaction_sector: "np.ndarray",
    metabolite_sector: "np.ndarray",
    restriction: "np.ndarray",
    growth_ratios: "np.ndarray",
    metabolome_fraction: float,
) -> "np.ndarray":
    """Calculate double-graded growth responses.

    Parameters
    ----------
    phi_high, kinetic_factor
        Aligned finite positive reaction vectors. ``phi_high`` must sum to one.
    reaction_sector, metabolite_sector
        Zero-based sector labels for reactions and metabolites.
    restriction
        Finite nonnegative restriction factor for each sector.
    growth_ratios
        Finite one-dimensional values in the closed interval [0, 1].
    metabolome_fraction
        Positive finite high-growth metabolome-to-proteome mass fraction.

    Returns
    -------
    np.ndarray
        c-bar, row-major enzyme responses, and row-major metabolite responses.

    Raises
    ------
    ValueError
        If shapes, labels, normalization, ranges, or finite-value requirements
        are violated, or if the mass-weighted kinetic average is not positive.
    """
    return result
```

### Step 3

Enzyme Profiles

Goal
----
Construct condition-specific enzyme abundances for one design.

```python
def enzyme_profiles(
    base_phi: "np.ndarray",
    design_multiplier: "np.ndarray",
    affinity: "np.ndarray",
    turnover: "np.ndarray",
    pathway_weight: "np.ndarray",
    reaction_sector: "np.ndarray",
    metabolite_sector: "np.ndarray",
    restriction: "np.ndarray",
    growth_ratios: "np.ndarray",
    metabolome_fraction: float,
) -> "np.ndarray":
    """Construct condition-specific enzyme abundances.

    Parameters
    ----------
    base_phi, design_multiplier, affinity, turnover, pathway_weight
        Aligned finite positive reaction vectors.
    reaction_sector, metabolite_sector, restriction, growth_ratios,
    metabolome_fraction
        Inputs following the growth-response contract.

    Returns
    -------
    np.ndarray
        A float64 condition-by-reaction enzyme-abundance matrix.

    Raises
    ------
    ValueError
        If the reaction vectors are misaligned or nonpositive, or if any
        downstream kinetic-factor or growth-response requirement is violated.
    """
    return result
```

### Step 4

Flux Sums

Goal
----
Calculate metabolite flux-sums from an irreversible flux vector.

```python
def flux_sums(stoichiometry: "np.ndarray", flux: "np.ndarray") -> "np.ndarray":
    """Calculate metabolite flux-sums.

    Parameters
    ----------
    stoichiometry
        A finite metabolite-by-reaction matrix using products minus reactants.
    flux
        A finite nonnegative reaction vector aligned with the matrix columns.

    Returns
    -------
    np.ndarray
        One float64 flux-sum per metabolite.

    Raises
    ------
    ValueError
        If the matrix and vector are misaligned, empty or non-finite, or if a
        flux is negative.
    """
    return result
```

### Step 5

Kineflux State

Goal
----
Infer one steady flux state with the flux-sum/logit capacity model.

```python
def kineflux_state(
    uptake: float,
    enzyme_abundance: "np.ndarray",
    apparent_rate: "np.ndarray",
    intercept: "np.ndarray",
    coefficients: "np.ndarray",
    stoichiometry: "np.ndarray",
    weight: float,
) -> "np.ndarray":
    """Infer one condition-specific steady flux state.

    Parameters
    ----------
    uptake
        Positive finite flux fixed on R01.
    enzyme_abundance, apparent_rate, intercept
        Finite reaction vectors of length eight. Enzyme abundances and
        apparent rates must be positive.
    coefficients
        Finite 8-by-4 flux-sum coefficient matrix.
    stoichiometry
        The finite 4-by-8 matrix for the stated branched network.
    weight
        Finite nonnegative quadratic flux penalty.

    Returns
    -------
    np.ndarray
        R01--R08 fluxes, R01--R08 saturation fractions, minimized objective,
        RMS capacity residual, and minimum capacity margin.

    Raises
    ------
    ValueError
        If an input violates the stated shape, range, topology or finite-value
        requirements, or if no feasible converged state is found.
    """
    return result
```

### Step 6

Shared Thermodynamic Fit

Goal
----
Fit one thermodynamic uncertainty state across all conditions.

```python
def shared_thermodynamic_fit(
    fluxes: "np.ndarray",
    log_concentrations: "np.ndarray",
    stoichiometry: "np.ndarray",
    standard_energy: "np.ndarray",
    uncertainty_map: "np.ndarray",
    gas_temperature: float,
    driving_force: float,
    design_shift: float,
    active_threshold: float,
) -> "np.ndarray":
    """Minimize the worst corrected energy with shared uncertainty.

    Parameters
    ----------
    fluxes
        Finite nonnegative condition-by-reaction flux matrix.
    log_concentrations
        Finite condition-by-metabolite natural-log concentrations.
    stoichiometry
        Finite metabolite-by-reaction matrix aligned to both inputs.
    standard_energy
        Finite standard-energy vector aligned to reactions.
    uncertainty_map
        Finite metabolite-by-coordinate matrix. The shared coordinate vector
        has Euclidean norm at most one.
    gas_temperature
        Positive finite value of RT.
    driving_force, design_shift
        Finite scalar additions to every corrected reaction energy.
    active_threshold
        Finite nonnegative threshold; flux strictly above it is active.

    Returns
    -------
    np.ndarray
        Worst corrected energy, fitted uncertainty coordinates, limiting
        condition index, and limiting reaction index.

    Raises
    ------
    ValueError
        If shapes, ranges or finite-value requirements are violated, no active
        reaction exists, or the constrained minimax fit does not converge.
    """
    return result
```

### Step 7

Design Record

Goal
----
Evaluate every condition for one biochemical design.

```python
def design_record(data: dict, design_index: int) -> "np.ndarray":
    """Evaluate one design across all conditions.

    Parameters
    ----------
    data
        Mapping containing the numerical arrays and scalars named in the task:
        affinity, turnover, pathway_weight, base_phi, reaction_sector,
        metabolite_sector, design_multiplier, restriction, growth_ratios,
        metabolome_fraction, uptake, apparent_rate, intercept, coefficients,
        stoichiometry, objective_weight, rho_high, standard_energy,
        uncertainty_map, gas_temperature, driving_force, design_shift,
        active_threshold, residual_limit, thermo_limit, and cost.
    design_index
        Zero-based integer row of the design arrays.

    Returns
    -------
    np.ndarray
        Eligibility indicator; cost-normalized score; c-bar; worst RMS
        residual; shared thermodynamic maximum; robust product yield; limiting
        thermodynamic condition and reaction; four R06 condition fluxes; and
        the limiting-yield condition index.

    Raises
    ------
    ValueError
        If keys are missing, the design index is invalid, design or condition
        arrays are misaligned, cost or uptake is nonpositive, or any composed
        step rejects its inputs.
    """
    return result
```

### Step 8

Evaluate Panel

Goal
----
Evaluate every biochemical design in row order.

```python
def evaluate_panel(data: dict) -> "np.ndarray":
    """Evaluate the complete design panel.

    Parameters
    ----------
    data
        Mapping following the design-record contract, with one row per design
        in both ``design_multiplier`` and ``restriction``.

    Returns
    -------
    np.ndarray
        A float64 design-by-thirteen matrix in the original design order.

    Raises
    ------
    ValueError
        If the mapping is missing design arrays, has no designs, has
        inconsistent design-row counts, or a composed evaluation fails.
    """
    return result
```

### Step 9

Select Design

Goal
----
Select the best eligible design and its eligible runner-up.

```python
def select_design(panel: "np.ndarray", tie_tolerance: float) -> "np.ndarray":
    """Select eligible designs by cost-normalized robust score.

    Parameters
    ----------
    panel
        Finite design-by-thirteen matrix. Column zero is eligibility and
        column one is score.
    tie_tolerance
        Finite nonnegative tolerance. Among scores within this distance of a
        maximum, the lowest design index is selected.

    Returns
    -------
    np.ndarray
        Winner index, winner score, runner-up index, and runner-up score. If
        only one design is eligible, both runner-up entries are -1.

    Raises
    ------
    ValueError
        If panel shape, eligibility indicators, scores or tolerance are
        invalid, or if no design is eligible.
    """
    return result
```

### Step 10

Run Benchmark

Goal
----
Run the complete biochemical design benchmark.

```python
def run_benchmark(data: dict, tie_tolerance: float) -> float:
    """Return the winning eligible design score.

    Parameters
    ----------
    data
        Mapping following the complete panel contract.
    tie_tolerance
        Finite nonnegative design-score tie tolerance.

    Returns
    -------
    float
        The winning eligible cost-normalized robust product yield.

    Raises
    ------
    ValueError
        If panel evaluation or design selection rejects its inputs.
    """
    return result
```
