# Biology-Biochemistry-18

## Background

A reactant pool contains protonation species whose relative abundances depend on the compartment state. Transport can select one chemical species from that pool, while the imported or exported material re-equilibrates with the destination pool. Consequently, reactant-level chemical balancing may require fractional buffered-proton coefficients, and electrical balance is a separate physical constraint.

Expression-informed directional entropy defines a phenotype without a prescribed biomass objective. A constrained zero net flux can coexist with positive forward and reverse directional variables. Physiological calibration compares the resulting energy estimates with a model containing concentrations and shared formation-energy uncertainty; shared factors encode coupling across reactions and conditions. The four-condition calibration and the performance rule in this task are stipulated extensions of the source methods.

## Problem

A buffered two-compartment biocatalytic module has six alternative expression designs and four independently imposed conditions. Infer each design's directional-entropy phenotype using reaction-attuned internal-reaction weights, reactant-level transformed thermodynamics, source-constrained organic balances, membrane-current balance, and the stated one-way evidence. Resolve transported species by the source method's compartment convention, subject to the supplied experimental override, and use the supplied species energies as already transformed at the stated pH. After phenotype inference, calibrate its active directional-ratio energies against one simultaneous physiological model per design: the intrinsic formation-error factors are shared across all four conditions, whereas concentrations may differ between conditions. Define the calibration radius as the smallest nonnegative uniform absolute energy mismatch compatible with the physiological constraints and the stated Euclidean uncertainty ball. A design is eligible when that radius is at most the declared limit; select the eligible design with the largest worst-condition fraction of terminal extracellular production carried by R11. Report that fraction as one dimensionless scalar to eight decimal places, with absolute answer tolerance 0.000002. In the reasoning, identify the source relations used and provide the compact scientific audit quantities listed below.

Model contract

| Item | Value or convention |
| --- | --- |
| Compartment labels | e = exterior; c = interior |
| Organic reactant order | A_e, B_e, C_e, D_e, A_c, B_c, C_c, D_c |
| Reaction order | R01 through R14 |
| Stoichiometry | Products positive, reactants negative; each listed organic coefficient has magnitude one |
| Chemical abstraction | A, B, C and D share a conserved non-hydrogen skeleton; each reactant pools the two listed protonation species |
| Temperature and constants | T = 298.15 K; R = 0.00831446261815324 kJ mol^-1 K^-1; F = 96.48533212 kJ mol^-1 V^-1 |
| Organic steady-state demand | (-10, 0, 0, 8, 0, 0, 0, 2) in the stated reactant order |
| Buffering | pH is clamped; free-proton rows are balanced chemically but exchanged with buffers, with no additional proton steady-state equation |
| Electrical condition | Zero net membrane charge transfer; positive transfer means e to c |
| Direction evidence | Net flux is nonnegative for R08, R11, R13 and R14; all other net fluxes have unrestricted sign |
| Directional variables | Both directions are strictly positive, including reactions whose constrained net flux is zero |
| Entropy scope | All fourteen internal reactions; original expression weights have no total-sum normalization |
| Order of stages | The physiological energy constraints apply to calibration after entropy-based phenotype inference |
| Concentration coordinate | x_i = ln(c_i / 1 mol L^-1), where c_i is the total reactant-pool concentration |
| Concentration bounds | -10.5 <= x_i <= -4.5 for every organic reactant and condition |
| Transported free protons | k is the number of additional free H+ ions transported along the reaction arrow, separate from protons bound to the transported organic species |
| Electrochemical sign convention | Transport from s to d contributes -k RT ln(10)(pH_d-pH_s) + q F(psi_d-psi_s), with q the total charge carried along that arrow |
| Transformed energies | Use the supplied species energies directly; the electrochemical transport term is additional; buffered H+ and the charge pseudo-reactant have no organic concentration coordinate |
| Formation uncertainty | Add L_i m to each pooled formation energy; the same two-vector m applies to both compartments and all four conditions of one design; each alternative design has its own calibration |
| Uncertainty set | m_1^2 + m_2^2 <= 0.36; this is a stipulated deterministic set |
| Activity | A reaction is active exactly when absolute net flux is greater than 0.0000001, recomputed separately for each condition |
| Physiological driving force | For each active reaction, sign(net flux) times physiological reaction energy <= -0.001 kJ mol^-1; inactive reactions impose no driving-force or mismatch constraint |
| Mismatch objective | Minimize the maximum absolute difference between inferred directional-ratio energy and physiological energy over all active reactions in all four conditions |
| Eligibility | Calibration radius <= 3.0 kJ mol^-1; an empty joint physiological region is ineligible |
| Performance | Minimum over the four conditions of net R11 flux divided by 8 |
| Ties | Scores within 0.0000000001 are tied; choose the lowest candidate number, then the lowest limiting-condition number |
| Scenario application | Multiply each original weight by that condition's factor; unlisted factors are one; condition rows are independent |
| Units of synthetic flux and weight | One common fixed flux unit; the logarithms in the entropy use dimensionless directional-flux/weight ratios |

Reactions and transport evidence

| Reaction | Organic transformation | Transport evidence |
| --- | --- | --- |
| R01 | A_e -> A_c | Default species; k = 0 |
| R02 | A_e -> A_c | Default species; k = 1 |
| R03 | B_e -> B_c | Default species; k = 0 |
| R04 | C_c -> C_e | Default species; k = 0 |
| R05 | D_c -> D_e | Experimentally transported species 1; k = 0 |
| R06 | A_c -> B_c | No membrane crossing |
| R07 | A_c -> C_c | No membrane crossing |
| R08 | B_c -> C_c | No membrane crossing |
| R09 | B_c -> D_c | No membrane crossing |
| R10 | C_c -> D_c | No membrane crossing |
| R11 | C_e -> D_e | No membrane crossing |
| R12 | B_e -> C_e | No membrane crossing |
| R13 | A_e -> B_e | No membrane crossing |
| R14 | No organic transformation | H+_e -> H+_c; k = 1 |

Species identities

| Species index | Bound protons | Charge |
| --- | --- | --- |
| 0 | 0 | -1 |
| 1 | 1 | 0 |

Conditions

| Condition | pH_e | pH_c | psi_e (V) | psi_c (V) |
| --- | --- | --- | --- | --- |
| P00 | 6.0 | 6.8 | 0.0 | 0.038 |
| P01 | 6.4 | 7.3 | 0.0 | 0.044 |
| P02 | 6.2 | 7.0 | 0.0 | 0.04 |
| P03 | 6.5 | 7.6 | 0.0 | 0.057 |

Already transformed species formation energies (kJ mol^-1)

| Condition | Compartment | Reactant | Species 0 | Species 1 |
| --- | --- | --- | --- | --- |
| P00 | e | A | 0.000000 | -1.712403 |
| P00 | e | B | -2.000000 | -8.278810 |
| P00 | e | C | -1.000000 | -4.995607 |
| P00 | e | D | -4.000000 | -13.132815 |
| P00 | c | A | 0.000000 | 2.854005 |
| P00 | c | B | -2.000000 | -3.712403 |
| P00 | c | C | -1.000000 | -0.429199 |
| P00 | c | D | -4.000000 | -8.566408 |
| P01 | e | A | 0.000000 | 0.570801 |
| P01 | e | B | -2.000000 | -5.995607 |
| P01 | e | C | -1.000000 | -2.712403 |
| P01 | e | D | -4.000000 | -10.849611 |
| P01 | c | A | 0.000000 | 5.708010 |
| P01 | c | B | -2.000000 | -0.858398 |
| P01 | c | C | -1.000000 | 2.424806 |
| P01 | c | D | -4.000000 | -5.712403 |
| P02 | e | A | 0.000000 | -0.570801 |
| P02 | e | B | -2.000000 | -7.137209 |
| P02 | e | C | -1.000000 | -3.854005 |
| P02 | e | D | -4.000000 | -11.991213 |
| P02 | c | A | 0.000000 | 3.995607 |
| P02 | c | B | -2.000000 | -2.570801 |
| P02 | c | C | -1.000000 | 0.712403 |
| P02 | c | D | -4.000000 | -7.424806 |
| P03 | e | A | 0.000000 | 1.141602 |
| P03 | e | B | -2.000000 | -5.424806 |
| P03 | e | C | -1.000000 | -2.141602 |
| P03 | e | D | -4.000000 | -10.278810 |
| P03 | c | A | 0.000000 | 7.420412 |
| P03 | c | B | -2.000000 | 0.854005 |
| P03 | c | C | -1.000000 | 4.137209 |
| P03 | c | D | -4.000000 | -4.000000 |

Original reaction weights

| Design | R01 | R02 | R03 | R04 | R05 | R06 | R07 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| C01 | 5.32 | 9.35 | 14.2 | 16.73 | 12.15 | 11.54 | 12.67 |
| C02 | 12.4 | 11.31 | 10.31 | 14.13 | 11.86 | 10.59 | 12.43 |
| C03 | 7.7 | 13.65 | 15.06 | 6.44 | 6.92 | 9.54 | 6.51 |
| C04 | 2.87 | 12.67 | 15.81 | 16.62 | 4.19 | 8.77 | 9.63 |
| C05 | 9.83 | 9.45 | 9.87 | 9.71 | 10.31 | 2.23 | 17.36 |
| C06 | 17.22 | 8.89 | 5.97 | 3.19 | 7.9 | 4.59 | 4.2 |

| Design | R08 | R09 | R10 | R11 | R12 | R13 | R14 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| C01 | 7.33 | 5.78 | 6.68 | 13.78 | 9.64 | 17.74 | 0.04 |
| C02 | 14.05 | 17.7 | 15.02 | 16.57 | 11.48 | 4.84 | 0.08 |
| C03 | 10.02 | 9.02 | 9.07 | 17.85 | 2.42 | 7.25 | 0.025 |
| C04 | 9.24 | 17.8 | 13.36 | 14.85 | 9.08 | 7.47 | 0.12 |
| C05 | 9.88 | 15.8 | 9.29 | 15.28 | 13.85 | 11.52 | 0.055 |
| C06 | 4.63 | 14.57 | 9.47 | 17.93 | 10.34 | 17 | 0.035 |

Independent weight changes

| Condition | Changes |
| --- | --- |
| P00 | All factors 1 |
| P01 | R01: 0.31; R09: 1.55 |
| P02 | R04: 0.37; R14: 1.8 |
| P03 | R06: 0.42; R11: 1.65 |

Formation-error loadings (kJ mol^-1)

| Reactant in either compartment | Factor 1 | Factor 2 |
| --- | --- | --- |
| A | 1.6 | -0.8 |
| B | -1.2 | 0.8 |
| C | 0.8 | 1.8 |
| D | -1.0 | -1.2 |

Co-assay coordinates

| Coordinate | Definition |
| --- | --- |
| H1 | x_Ac - x_Bc |
| H2 | x_Cc - x_Dc |
| H3 | x_Be - x_Bc - x_Ce + x_Cc |

Co-assay intervals shared by all alternative designs

| Condition | H1 lower | H1 upper | H2 lower | H2 upper | H3 lower | H3 upper |
| --- | --- | --- | --- | --- | --- | --- |
| P00 | 0.707 | 1.067 | -0.433 | -0.073 | 1.421 | 1.821 |
| P01 | 0.437 | 0.797 | 0.146 | 0.506 | 0.625 | 1.025 |
| P02 | -0.496 | -0.136 | -0.327 | 0.033 | 1.604 | 2.004 |
| P03 | -0.660 | -0.300 | -0.817 | -0.457 | 1.197 | 1.597 |

Compact audit in the reasoning

Provide the interior B pooled formation energy in P00; the species index transported by R03 in P01; the buffered interior proton coefficient of R03 in P00; the R06/R07 reaction-energy error inner product from the shared loadings; the independent organic-plus-charge balance rank in P01; the C05/P02 net R11 flux; the C04/P00 inferred R14 energy and physiological R14 energy; all six joint calibration radii; the eligible designs; and the selected design's limiting condition.

Output format

Start with `<final_answer>` containing only the finite decimal scalar and `</final_answer>`. Follow it with `<reasoning>` containing the scientific reasoning and requested audit quantities, and close with `</reasoning>`. Include nothing after the closing reasoning tag.

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

01_reactant_state

Goal
----
Compute each pooled reactant state from the already transformed species energies.

```python
def reactant_state(species_energies: "np.ndarray", species_h: "np.ndarray",
                   species_z: "np.ndarray", rt: float) -> "np.ndarray":
    """
    species_energies has shape (P,2,M,J), in kJ/mol; axes are condition,
    compartment (exterior then interior), reactant and species. species_h and
    species_z are length-J proton counts and charges shared by all reactants.
    rt is the positive thermal energy in kJ/mol. Return shape (P,2,M,3+J),
    with trailing entries pooled formation energy, mean bound-proton count,
    mean charge, then the J fractions in supplied order. P,M,J are positive.
    Finite aligned data are required; invalid shapes, nonfinite data or
    nonpositive rt raise ValueError. Tied energies have equal probabilities.
    """
    return result
```

### Step 2

02_transport_state

Goal
----
Resolve the transported species and free-proton count for every transport record and condition.

```python
def transport_state(reactants: "np.ndarray", transports: "np.ndarray",
                    species_h: "np.ndarray", species_z: "np.ndarray") -> "np.ndarray":
    """
    reactants is the (P,2,M,3+J) output contract of step 1. species_h and
    species_z have length J. transports has shape (T,6), integer columns
    (reaction index, reactant index, source compartment, destination
    compartment, extra free-proton count, species override). Compartments
    are 0 exterior and 1 interior; source and destination differ. Reactant
    index -1 denotes pure proton transport; override -1 denotes the source
    default. Reaction indices are distinct nonnegative integers. Counts
    are nonnegative. Return (P,T,4): transported bound-proton count,
    transported organic charge, extra free-proton count, selected species
    index. Pure proton rows use (0,0,k,-1). T may be zero. Resolve equal
    maximum abundances by the lowest species index. Invalid dimensions,
    noninteger records or out-of-range species/compartment indices raise
    ValueError. Other inputs satisfy the preceding step contracts.
    """
    return result
```

### Step 3

03_balanced_reactions

Goal
----
Construct organic, buffered-proton and membrane-charge coefficients for each condition.

```python
def balanced_reactions(stoich: "np.ndarray", reactants: "np.ndarray",
                       transports: "np.ndarray", transported: "np.ndarray") -> "np.ndarray":
    """
    stoich is the (2M,N) organic matrix, exterior reactants followed by
    interior reactants. reactants and transports follow steps 1 and 2.
    transported has shape (P,T,4) from step 2. Return (P,2M+3,N), rows
    organic, buffered exterior H+, buffered interior H+, membrane charge.
    The charge coefficient is positive for exterior-to-interior transfer.
    Chemical reactions without transport records remain compartment-local.
    The proton rows encode re-equilibration plus explicitly transported
    protons; they are buffered and are not additional steady-state rows.
    Inputs obey chemical skeleton conservation and aligned species counts.
    Shape mismatches, nonfinite stoichiometry/transport data and reaction
    indices outside the matrix raise ValueError. Numerical zeros below
    1e-13 may be returned as exact zero.
    """
    return result
```

### Step 4

04_energy_operator

Goal
----
Form each reaction energy as an affine function of concentrations and intrinsic formation errors.

```python
def energy_operator(stoich: "np.ndarray", reactants: "np.ndarray",
                    balanced: "np.ndarray", transports: "np.ndarray",
                    conditions: "np.ndarray", loadings: "np.ndarray",
                    rt: float, faraday: float) -> "np.ndarray":
    """
    stoich, reactants, balanced and transports follow preceding steps.
    conditions has shape (P,2,2), trailing columns pH and electrical
    potential in volts. loadings has shape (2M,K), kJ/mol per intrinsic
    factor; its rows follow organic reactant order. rt and faraday are
    positive finite constants in kJ/mol and kJ/mol/V. Return (P,N,1+2M+K):
    constant energy, the 2M concentration coefficients, then K intrinsic
    factor coefficients. Thus one packed row represents a+B*x+Q*m.
    Use the stated electrochemical signs and organic concentration
    coordinates. K may be zero. Invalid aligned energy dimensions,
    nonfinite condition/loading data or nonpositive constants raise
    ValueError; other inputs satisfy preceding contracts.
    """
    return result
```

### Step 5

05_phenotype_panel

Goal
----
Infer the unique net-flux optimum for every design and independent condition.

```python
def phenotype_panel(balanced: "np.ndarray", demand: "np.ndarray",

                    weights: "np.ndarray", factors: "np.ndarray",

                    oneway: "np.ndarray") -> "np.ndarray":

    """

    balanced has shape (P,M+3,N), with M organic rows, two buffered H+

    rows and one charge row. demand is length M; the charge demand is

    zero. weights is positive finite (C,N), factors positive finite

    (P,N); multiply factors independently into the original weights.

    oneway lists distinct zero-based net-nonnegative reaction indices.

    Return net fluxes (C,P,N) for the source directional-entropy optimum:

    with effective weights w = weights * factors, maximise

    -sum_j [f_j ln(f_j / w_j) + r_j ln(r_j / w_j)] over strictly positive

    forward and reverse variables whose differences f_j - r_j are the net

    fluxes, subject to the balance rows and the one-way constraints.

    All other net fluxes are unconstrained in sign; no normalization or

    extra linear flux reward is added. Finite aligned arrays and feasible

    balances are required; tested shape/value violations or infeasibility

    raise ValueError. An unresolved numerical solve raises RuntimeError.

    Duplicate conservation rows are allowed. Absolute accuracy 0.000002

    is sufficient; numerical fluxes below 2e-10 may be set to zero.

    """

    return result
```

### Step 6

06_directional_state

Goal
----
Recover the entropy-consistent positive directional pair and its energy estimate.

```python
def directional_state(net: "np.ndarray", weights: "np.ndarray",
                      factors: "np.ndarray", rt: float) -> "np.ndarray":
    """
    net has shape (C,P,N); weights is positive finite (C,N) and factors
    positive finite (P,N). They are the original weights and independent
    condition factors used in phenotype inference. rt is positive and
    finite in kJ/mol. Return (C,P,N,3), trailing entries forward flux,
    reverse flux and inferred energy in kJ/mol. Both directional fluxes
    are positive, even at net zero. Invalid aligned shapes, nonfinite
    values or nonpositive weights, factors or rt raise ValueError. Valid
    inputs are in a range where directional outputs are representable.
    """
    return result
```

### Step 7

07_calibration_system

Goal
----
Construct the joint calibration half-spaces with shared intrinsic factors and condition-specific concentrations.

```python
def calibration_system(operator: "np.ndarray", net: "np.ndarray",
                       energy: "np.ndarray", assay: "np.ndarray",
                       intervals: "np.ndarray", x_bounds: "np.ndarray",
                       activity: float, drive: float) -> "np.ndarray":
    """
    operator has shape (P,N,1+M+K), packed as step 4; net and energy
    are (P,N) for one design. assay is (H,M); intervals is (P,H,2);
    x_bounds is (P,M,2), with ordered lower/upper bounds. activity is
    nonnegative and drive positive. Return augmented half-spaces [D|d]
    for D*y<=d, variables y=(x_0,...,x_(P-1),m,delta); each x has M
    entries and the single shared m has K. Columns number P*M+K+2.
    For each condition in order, visit active reactions in ascending
    index: driving-force row, upper mismatch row, lower mismatch row;
    then assay upper/lower pairs in supplied order; then concentration
    upper/lower pairs in reactant order. Append -delta<=0 last. Only
    abs(net)>activity is active; equality is inactive. Do not encode
    the intrinsic uncertainty ball here. H and K may be zero. Invalid
    alignment, nonfinite data, reversed bounds or invalid thresholds
    raise ValueError. This explicit row order is the numerical contract.
    """
    return result
```

### Step 8

08_joint_energy_radius

Goal
----
Find the least simultaneous mismatch over the shared Euclidean uncertainty region.

```python
def joint_energy_radius(system: "np.ndarray", n_concentrations: int,
                        n_factors: int, radius: float) -> float:
    """
    system is a finite augmented [D|d] array as in step 7, with variables
    (concentrations, shared factors, delta). n_concentrations and n_factors
    are nonnegative integer counts; the last variable is delta>=0. radius
    is finite and nonnegative, constraining the Euclidean norm of all
    n_factors together. Return the minimum delta, or positive infinity
    if the feasible region is empty. Bounded concentration domains are
    encoded in system. Zero factors or zero radius are allowed. Invalid
    dimensions, nonfinite data or negative radius raise ValueError.
    Unresolved solver status or optimality gap raises RuntimeError.
    Finite answers are accepted within absolute tolerance 0.000002.
    """
    return result
```

### Step 9

09_robust_delivery

Goal
----
Apply the calibration limit and choose the largest worst-condition delivery fraction.

```python
def robust_delivery(net: "np.ndarray", calibration: "np.ndarray", route: int,
                    demand: float, limit: float, tie: float) -> "np.ndarray":
    """
    net is finite (C,P,N); calibration is length C, nonnegative with
    positive infinity allowed for infeasibility. route is a zero-based
    reaction index; demand is a positive denominator. limit and tie are
    finite and nonnegative. Eligible means calibration<=limit. The
    score is min_condition(net[route]/demand). Return length three:
    selected score, zero-based candidate index, zero-based limiting
    condition index, all numerical. Candidate scores within tie of the
    maximum are tied; choose the lowest candidate index. Limiting
    condition scores within tie of that candidate minimum are tied;
    choose the lowest condition index. Invalid values/alignment or no
    eligible design raise ValueError.
    """
    return result
```

### Step 10

10_resolve_transport_panel

Goal
----
Compose all preceding scientific operations into the requested robust production fraction

```python
def resolve_transport_panel(species_energies: "np.ndarray", weights: "np.ndarray",
                            assay_intervals: "np.ndarray", radius: float,
                            limit: float, model: dict) -> float:
    """
    species_energies is finite (4,2,4,2), weights positive finite (C,14),
    and assay_intervals finite ordered (4,3,2). They use the main problem
    axes and units. radius>=0 is the shared intrinsic-factor norm bound;
    limit>=0 is the calibration eligibility limit in kJ/mol. model is a dictionary with keys s, transports, conditions, factors,
    loadings, h, z, assay, demand, oneway, x_bounds, rt, faraday, activity
    and drive. These are the aligned inputs of the preceding public
    functions: s is organic stoichiometry; h and z are species properties.
    The model uses four conditions, eight organic pools and two intrinsic
    factors. All required numerical inputs are explicit arguments. Compose
    the preceding public functions and use every returned scientific
    result. Return the selected robust fraction as a finite float. Invalid
    inputs, infeasible phenotype or no eligible design raise ValueError.
    Unresolved numerical failures propagate as RuntimeError. The main
    benchmark uses radius=0.6, limit=3.0. Absolute accuracy 0.000002 suffices.
    """
    return result
```
