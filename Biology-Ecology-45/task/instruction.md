# Biology-Ecology-45

## Background

Process-based agroecosystem models represent how weather, soil water, vegetation, chemical partitioning, and transformation jointly determine the environmental fate of a plant-protection product. A useful pesticide-fate extension must distinguish the parent compound from a metabolite, represent the large sorbed pool of a strongly adsorbing molecule, and still permit short rainfall events to move a small dissolved fraction rapidly through preferential pathways.
This task isolates those coupled mechanisms from the much larger host model. It uses a fixed spatial basis, a short deterministic forcing sequence, and explicit state bookkeeping so the benchmark is reproducible without the original agricultural-model code or field datasets. The final standardized response-sensitivity norm tests the interaction of all pathways rather than the transcription of one equation.

## Problem

A daily agroecosystem fate model has been extended to represent the coupled behavior of a strongly sorbing parent herbicide and its soil metabolite. Use the cited 2026 article and its Supplementary Material to recover the environmental degradation-rate correction, nonlinear Freundlich partition, rainfall-dependent canopy wash-off, preferential macropore routing, and metabolite-formation rules. Implement those source methods in a deterministic seven-day, three-layer benchmark and compute the standardized full-pathway response sensitivity defined below.
 
Use a 1 m² basis throughout: masses are mg m$^{-2}$, water depth in mm equals water volume in L m$^{-2}$, aqueous concentration is mg L$^{-1}$, soil mass is kg m$^{-2}$, and $K_f$ is interpreted consistently with $C_s=K_f C_w^n$. Close each layer's phase partition by solving $M=V_w C_w+M_s K_f C_w^n$ for its unique nonnegative $C_w$. The tabulated glyphosate $K_f=38.37$ and $n=1.28$ are the Valdobbiadene values from Supplementary Table S2; the AMPA partition arrays below are benchmark configuration inputs rather than claims about that table.
 
Resolve the source's implementation ambiguities by using these declared conventions. Test the degradation branch with Celsius temperature, but convert $T$ to Kelvin inside the Arrhenius exponent; cap the Walker moisture ratio at one. Treat the right-hand side of the source's decreasing-exponential wash-off equation as post-event canopy residue and transfer its complement to the top layer; when rain, LAI, or ground cover is zero, set intercepted rain to zero. Within each day, apply canopy wash-off, then foliar decay, add the washed mass to the top layer, dissipate pre-existing soil glyphosate and AMPA, add newly formed AMPA only after that day's pre-existing AMPA dissipation, solve phase partitioning, and finally route preferential bypass. Cap bypassed solute at the dissolved mass available in the top layer, redistribute the $CRK_{ad}$ fraction across all layers in proportion to layer thickness, and count the remainder as cumulative export.
 
Use IEEE-754 float64 arithmetic, natural exponentials, and no rounding between operations. Solve every phase closure by bisection on the bracket from $0$ to $M/V_w$, halving it until its width is at most $10^{-13}\max(1, \text{hi})$ or 300 iterations have been taken, and return the midpoint of the final bracket; a solve that reaches 300 iterations without meeting that test is an error. Use this configuration:
 
~~~python
forcing = {
    "temperature_c": np.array([
        [18.0,16.5,15.0],[20.0,18.2,16.4],[22.0,19.5,17.0],
        [19.0,17.5,16.0],[24.0,21.0,18.5],[21.0,19.0,17.2],
        [17.0,16.0,15.2],
    ], dtype=np.float64),
    "moisture": np.array([
        [0.19,0.24,0.29],[0.20,0.25,0.30],[0.28,0.30,0.32],
        [0.24,0.28,0.31],[0.33,0.34,0.36],[0.27,0.31,0.34],
        [0.25,0.29,0.33],
    ], dtype=np.float64),
    "rainfall": np.array([0.0,6.0,18.0,0.0,32.0,4.0,15.0]),
    "leaf_area_index": np.array([2.20,2.18,2.15,2.12,2.08,2.04,2.00]),
    "ground_cover_fraction": np.array([0.72,0.71,0.70,0.69,0.68,0.67,0.66]),
    "retention_current": np.array([35.0,38.0,52.0,44.0,68.0,48.0,57.0]),
    "retention_maximum": np.full(7,80.0),
}
profile = {
    "layer_thickness": np.array([1.0,4.0,10.0]),
    "soil_mass": np.array([13.0,52.0,130.0]),
    "field_capacity": np.array([0.31,0.33,0.35]),
    "glyphosate_kf": np.full(3,38.37),
    "glyphosate_exponent": np.full(3,1.28),
    "ampa_kf": np.array([22.0,26.0,31.0]),
    "ampa_exponent": np.array([1.16,1.18,1.20]),
}
parameters = {
    "application_mass": 144.0, "k_ref": 0.165,
    "activation_energy": 54000.0, "gas_constant": 8.314,
    "reference_temperature_k": 293.15, "moisture_exponent": 0.7,
    "solubility_g_l": 12.0, "foliar_half_life_days": 10.60,
    "transformation_fraction": 0.30, "movement_fraction": 0.50,
    "adsorption_fraction": 0.08, "interception_alpha": 0.25,
    "glyphosate_kf_scale": 1.0, "glyphosate_exponent_scale": 1.0,
    "ampa_kf_scale": 1.0, "ampa_exponent_scale": 1.0,
}
~~~
 
Initialize canopy glyphosate to the first day's ground-cover fraction times the application mass and top-layer glyphosate to the complement; all other pools and cumulative exports start at zero. After seven days, form $y=[P_{fol},G_1,G_2,G_3,A_1,A_2,A_3,E_G,E_A]$ in that exact order. For $p=[k_{ref},s_{K_f,G},s_{n,G},s_{K_f,A},s_{n,A},Sol,t_{1/2,fol},CRK_{mov},CRK_{ad},TRSF]$, in that order, use $h=10^{-4}$ and define $S_{ij}=[y_i(p_j(1+h))-y_i(p_j(1-h))]/[2h\,y_i(p)]$.
 
Return $\lVert S\rVert_F$. In <reasoning>, first give four concise source-method statements: (1) the temperature-moisture degradation correction, (2) the Freundlich relation, (3) the canopy interception/wash-off construction, and (4) the preferential-flow and AMPA-formation constructions. Then report the first-day three-layer degradation-rate vector; the diagnostic Freundlich root for $M=40.32$, $V_w=1.9$, $M_s=13$, $K_f=38.37$, and $n=1.28$; the day-2 canopy output generated from the day-1 remaining canopy mass and the second forcing row; the nine baseline response values; the ten column norms of $S$ in parameter order; their descending ranking; the $9\times10$ matrix dimensions; the final Frobenius norm; and one benchmark-specific sentence identifying the strongest and weakest columns. Report all numeric values to at least 12 significant figures.
 
Output Format Requirements: Emit <final_answer> immediately, then <reasoning>. Do not write a long derivation before the tags. You must emit exactly one finite decimal number inside <final_answer>...</final_answer>, even if the value is approximate or you are unsure. Do not place units, words, LaTeX, boxed notation, lists, ranges, or multiple values inside <final_answer>. Show only the requested source-method relations, vectors, ranking, scalar, and brief interpretation in <reasoning>...</reasoning>.

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

01_step_01_environmental_degradation_rate.py

Goal
----
Compute the source's temperature- and moisture-adjusted first-order degradation coefficient for every soil layer. A correct result converts positive Celsius temperatures to Kelvin only inside the Arrhenius term, caps moisture at field capacity, and returns zero at nonpositive Celsius temperature; a failure changes dissipation and every downstream parent/metabolite mass.

```python
def environmental_degradation_rate(
    temperature_c: "np.ndarray",
    moisture: "np.ndarray",
    field_capacity: "np.ndarray",
    k_ref: float,
    activation_energy: float,
    gas_constant: float,
    reference_temperature_k: float,
    moisture_exponent: float,
) -> "np.ndarray":
    """Return layerwise environment-adjusted daily degradation rates.
 
    Parameters
    ----------
    temperature_c, moisture, field_capacity : np.ndarray
        Finite one-dimensional arrays with the same nonempty shape. Moisture
        is nonnegative and field capacity is strictly positive.
    k_ref : float
        Finite nonnegative reference rate in d^-1.
    activation_energy : float
        Finite nonnegative activation energy in J mol^-1.
    gas_constant : float
        Finite positive gas constant in J mol^-1 K^-1.
    reference_temperature_k : float
        Finite positive reference temperature in kelvin.
    moisture_exponent : float
        Finite nonnegative Walker exponent.
 
    Returns
    -------
    rates : np.ndarray
        Daily rate coefficients with shape (L,).
 
    Raises
    ------
    ValueError
        If shapes, finiteness, or stated value ranges are violated.
    """
    return rates
```

### Step 2

02_step_02_freundlich_aqueous_concentration.py

Goal
----
Solve the nonlinear one-layer mass closure for the unique nonnegative aqueous concentration. A passing implementation preserves the declared water and soil mass basis and respects the convergence contract; an incorrect phase balance changes the dissolved mass available for preferential transport.

```python
def freundlich_aqueous_concentration(
    total_mass: float,
    water_volume: float,
    soil_mass: float,
    freundlich_coefficient: float,
    freundlich_exponent: float,
    tolerance: float,
    max_iterations: int,
) -> float:
    """Solve the nonlinear Freundlich phase-mass closure.
 
    Parameters are finite. Total and soil masses and the coefficient are
    nonnegative; water volume, exponent, and tolerance are strictly positive.
    max_iterations must be a positive integer; bool and float are invalid.
 
    Returns
    -------
    concentration : float
        Unique nonnegative aqueous concentration in mg L^-1.
 
    Raises
    ------
    ValueError
        If finiteness, stated ranges, or the integer contract is violated.
    RuntimeError
        If bisection does not converge within max_iterations.
    """
    return concentration
```

### Step 3

03_step_03_canopy_washoff.py

Goal
----
Compute canopy residue, soil transfer, foliar dissipation, and intercepted rainfall for one day. Passing behavior uses the declared mass-conserving interpretation of the source's decreasing exponential and handles degenerate rainfall or canopy geometry without division by zero; failure corrupts the soil input and canopy mass balance.

```python
def canopy_washoff(
    canopy_mass: float,
    rainfall: float,
    leaf_area_index: float,
    ground_cover_fraction: float,
    solubility_g_l: float,
    foliar_half_life_days: float,
    interception_alpha: float,
) -> "np.ndarray":
    """Apply one day's canopy wash-off followed by foliar decay.
 
    All values are finite. Mass, rain, and LAI are nonnegative; solubility,
    half-life, and alpha are positive; ground cover lies in [0,1]. The source
    exponential is treated as post-wash residue and its complement as transfer.
 
    Returns
    -------
    result : np.ndarray
        [remaining canopy mass, soil transfer, foliar loss, intercepted rain].
 
    Raises
    ------
    ValueError
        If finiteness or any stated range is violated.
    """
    return result
```

### Step 4

04_step_04_preferential_bypass.py

Goal
----
Route top-layer dissolved solute through a rainfall-driven macropore event. A correct output caps removal at available dissolved mass, deposits the adsorbed fraction by layer thickness, and exports the remainder; a failing result violates solute conservation or the paper's preferential-flow construction.

```python
def preferential_bypass(
    aqueous_concentration: float,
    available_dissolved_mass: float,
    rainfall: float,
    retention_current: float,
    retention_maximum: float,
    movement_fraction: float,
    adsorption_fraction: float,
    layer_thickness: "np.ndarray",
) -> "np.ndarray":
    """Route one solute through a preferential-flow event.
 
    Returns
    -------
    routing : np.ndarray
        [preferential water, bypass mass, L deposited masses, exported mass].
 
    Raises
    ------
    ValueError
        If array shape, finiteness, nonnegativity, retention ordering, or the
        [0,1] fraction contracts are violated.
    """
    return routing
```

### Step 5

05_step_05_daily_glyphosate_ampa_update.py

Goal
----
Advance all canopy, layered parent/metabolite, and export pools through one declared daily event sequence. Passing behavior reuses the earlier process functions, delays new AMPA until after existing-AMPA dissipation, and preserves nonnegative mass bookkeeping; failure changes the coupled state even when individual formulas are correct.

```python
def daily_glyphosate_ampa_update(
    state: "np.ndarray",
    temperature_c: "np.ndarray",
    moisture: "np.ndarray",
    rainfall: float,
    leaf_area_index: float,
    ground_cover_fraction: float,
    retention_current: float,
    retention_maximum: float,
    profile: dict,
    parameters: dict,
) -> "np.ndarray":
    """Advance the coupled parent--metabolite state by one day.
 
    State order is [canopy GLY, L soil GLY, L soil AMPA, cumulative GLY
    export, cumulative AMPA export]. Dictionaries use the prompt keys and are
    not mutated.
 
    Returns
    -------
    next_state : np.ndarray
        Nonnegative state with shape (2L+3,) in the same order.
 
    Raises
    ------
    ValueError
        If state/profile/daily shapes, finiteness, water volume, or an upstream
        process contract is invalid.
    """
    return next_state
```

### Step 6

06_step_06_simulate_glyphosate_ampa_fate.py

Goal
----
Initialize the application and run the daily update across all forcing rows. A correct simulation validates every configuration key and preserves the exact state order without mutating inputs; a failure invalidates the endpoint used by every sensitivity calculation.

```python
def simulate_glyphosate_ampa_fate(
    forcing: dict,
    profile: dict,
    parameters: dict,
) -> "np.ndarray":
    """Run the deterministic coupled fate simulation through all days.
 
    Returns
    -------
    final_state : np.ndarray
        Shape (2L+3,) in the declared state order.
 
    Raises
    ------
    ValueError
        If required keys, shapes, finiteness, cover bounds, application mass,
        or an upstream daily contract is invalid.
    """
    return final_state
```

### Step 7

07_step_07_pathway_sensitivity_matrix.py

Goal
----
Compute the response-by-parameter matrix from paired relative perturbations of the complete multi-day simulation. Passing behavior perturbs one positive parameter at a time, uses the unperturbed response for standardization, and rejects noninteger response indices; failure can look plausible while changing the final norm materially.

```python
def pathway_sensitivity_matrix(
    forcing: dict,
    profile: dict,
    parameters: dict,
    sensitive_keys: list[str],
    response_indices: "np.ndarray",
    relative_step: float,
) -> "np.ndarray":
    """Return the standardized full-pipeline response sensitivity matrix.
 
    sensitive_keys is a nonempty list of unique positive parameter names.
    response_indices is a nonempty 1D integer array of unique valid state
    indices whose baseline values are positive.
 
    Returns
    -------
    sensitivities : np.ndarray
        Shape (R,P), preserving response and parameter order.
 
    Raises
    ------
    ValueError
        If the step, keys, indices, baselines, fractions, or upstream
        simulation inputs violate their contracts. Float indices are invalid.
    """
    return sensitivities
```

### Step 8

08_step_08_epic_fate_sensitivity_norm.py

Goal
----
Orchestrate the complete sensitivity calculation and return its Frobenius norm. A passing implementation delegates to the sensitivity-matrix function and reduces every matrix entry exactly once; a failing result misses a pathway, response, or standardization and cannot reproduce the independent literal integration targets.

```python
def epic_fate_sensitivity_norm(
    forcing: dict,
    profile: dict,
    parameters: dict,
    sensitive_keys: list[str],
    response_indices: "np.ndarray",
    relative_step: float,
) -> float:
    """Return the aggregate standardized sensitivity norm.
 
    Returns
    -------
    norm : float
        Native Python float equal to the matrix Frobenius norm.
 
    Raises
    ------
    ValueError
        If any sensitivity-matrix or upstream simulation contract is invalid.
    """
    return norm
```
