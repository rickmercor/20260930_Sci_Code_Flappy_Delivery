# Biology-Ecology-18

## Background

The task joins three ecological calculations that are normally used separately. The first measures niche breadth and overlap after removing focal species without changing the complete-table occupancy count. The second reconstructs a disturbed community trajectory from reference trajectories and carries the signed forecast residual into species-level exposure. The third asks whether each management plan remains feasible, globally stable, and above its reserve under a shared two-dimensional climate disturbance.
The difficulty lies in the hand-offs. A circular exclusion changes the breadth and overlap matrix. A state-level PETRA weight changes the selected forecast and the signed exposure. A local-stability shortcut, independent disturbance box, or endpoint-only effort search then changes the management winner. These are not harmless numerical substitutions: every one enters several downstream matrices before the final plan comparison.
The complete pipeline therefore checks method recovery, numerical implementation, and ecological interpretation together. All ordering rules, strict comparisons, tie rules, and array layouts are binding.

## Problem

Coupled niche, trajectory, and climate-robust management score
Write the ten functions in the sub-problems and use the complete chain to choose one management plan for the six-species community below. The calculation joins three published ecological methods. None of the three parts may be replaced by a circular niche calculation, an ordinary nearest-neighbour forecast, a local stability check, an independent disturbance box, or an endpoint-only plan comparison.
The resource table is handled noncircularly. For the breadth of species i, calculate the resource-state factors from the table with species i removed, then apply those factors to the original row of species i. For the overlap of species i and j, calculate the factors after removing both species, then apply them to the two original rows. In every exclusion keep r_prime, the number of occupied resource states, fixed at its value in the complete table. Use the modified exponential factors from the source paper, the paper's single-sum state contributions, the Shannon-type standardized breadth, and the information-theoretic overlap. Use k=10000.
The disturbed trajectory is handled with PETRA-EDR. Rank individual reference states by Bray-Curtis dissimilarity, preserve trajectory identity, use each selected trajectory's smallest target-to-state dissimilarity for its kernel weight, and stop a sweep before the first shift with fewer than min_pts usable neighbours. Calibrate the five candidates by MPD, choose the first exact minimum, and forecast from the pre-disturbance state. Distance to a forecast means the smallest Bray-Curtis dissimilarity to any state on the forecast. The signed assessment residual is (assessment - closest_forecast_state) / sum(assessment + closest_forecast_state). Multiply it by guild_to_species to obtain the six-species exposure vector.
Use the noncircular breadths, pair overlaps, PETRA resilience profile, and exposure vector to alter competition, self-regulation, disturbance loading, and reserve requirements. The exact coupling rules are given below. For each plan and condition, convert competition to regulation coordinates by dividing column j by that plan's slope d_j. Use the finite-community canonical decomposition, the uniform-forcing reachable-subspace feasibility threshold, and the symmetric-part global stability boundary. The lower effort is the larger of minimum_effort and buffer times every condition-specific feasibility and stability threshold.
At a feasible effort a, solve the equilibrium and its two correlated disturbance responses in canonical coordinates. The two disturbance coordinates share one Euclidean unit disk. For population i, the admissible radius is (x_i - reserve_i) / ||response_i||_2; do not replace the row norm with a componentwise maximum. A plan's climate radius is the minimum over all populations and conditions.
Optimize effort continuously on the full closed interval [lower, upper]. The unnormalized utility is
radius(a) + gain*log1p(a) - penalty*a.
Divide its global maximum by the plan cost. Crossings between limiting populations and interior stationary points are allowed. Do not compare only the two interval endpoints. An unavailable plan has no score. Select the largest nonnegative score; values within 1e-9 tie in favour of the earliest zero-based plan.
Fixed coupling rules
Let beta be the six noncircular breadths, Gamma the 6 by 6 noncircular overlap matrix, e the six-species PETRA exposure, and (R,A,Q,N) the resistance, amplitude, recovery, and net-change profile. For condition s=0,1,2, set
tau_s = c0 + c1*[N,A,Q]_s
B_s[i,j] = B0_s[i,j] * max(0.08, 1 + tau_s*Gamma[i,j] + c2*(e_i-e_j)), with a zero diagonal,
D[c,i] = D0[c,i]*(1+c3*(1-beta_i))*(1+c4*abs(e_i)),
U[s,i,0] = U0[s,i,0] + c5*e_i,
U[s,i,1] = U0[s,i,1] - 0.7*c5*e_i,
reserve_i = reserve0_i*(1+c6*abs(e_i))*(1+c7*(1-R)).
coefficients = [1.5, 6.0, 7.0, 0.8, 3.0, 5.0, 6.0, 1.5].
Numerical instance
resource_matrix = [[7, 39, 11, 9, 0, 6, 0], [39, 0, 25, 0, 31, 4, 0], [0, 0, 0, 21, 37, 10, 0], [0, 0, 0, 0, 14, 0, 0], [0, 0, 0, 37, 0, 0, 0], [0, 38, 0, 0, 0, 0, 0]]
k = 10000.0

reference_trajectories = [[[62, 26, 8, 3, 1], [56, 28, 10, 4, 2], [48, 31, 12, 6, 3], [38, 32, 17, 8, 5], [28, 30, 22, 12, 8], [21, 25, 25, 19, 10], [17, 20, 26, 23, 14], [13, 18, 23, 26, 20]], [[65, 24, 7, 3, 1], [57, 28, 9, 4, 2], [49, 31, 12, 5, 3], [39, 31, 17, 8, 5], [29, 30, 21, 12, 8], [21, 26, 25, 17, 11], [16, 21, 26, 22, 15], [12, 18, 24, 26, 20]], [[60, 28, 8, 3, 1], [53, 30, 11, 4, 2], [46, 32, 13, 6, 3], [37, 31, 18, 9, 5], [29, 28, 22, 13, 8], [22, 24, 24, 19, 11], [18, 20, 25, 23, 15], [14, 18, 23, 25, 20]], [[66, 23, 7, 3, 1], [58, 27, 9, 4, 2], [49, 30, 12, 6, 3], [39, 30, 17, 9, 5], [29, 29, 20, 14, 8], [21, 25, 24, 19, 11], [16, 21, 25, 22, 16], [11, 19, 24, 25, 21]], [[59, 27, 10, 3, 1], [53, 29, 12, 4, 2], [45, 32, 14, 6, 3], [37, 32, 17, 9, 5], [29, 29, 22, 12, 8], [23, 24, 24, 18, 11], [18, 20, 26, 21, 15], [15, 16, 24, 25, 20]], [[63, 27, 6, 3, 1], [56, 29, 9, 4, 2], [47, 32, 12, 6, 3], [38, 32, 17, 8, 5], [28, 30, 21, 13, 8], [21, 25, 25, 18, 11], [17, 20, 26, 22, 15], [13, 17, 24, 26, 20]]]
calibration_targets = [[53, 30, 10, 5, 2], [36, 31, 18, 10, 5], [20, 24, 25, 19, 12]]
disturbed_trajectory = [[56, 28, 10, 4, 2], [46, 31, 14, 6, 3], [18, 17, 18, 25, 22], [20, 20, 20, 23, 17], [22, 22, 21, 20, 15], [20, 23, 23, 19, 15]]
candidates = [(4, 0.42, 3, 'linear', 1.0), (5, 0.42, 3, 'exponential', 2.0), (6, 0.42, 4, 'hyperbolic', 3.0), (7, 0.42, 4, 'spherical', 1.0), (8, 0.42, 5, 'power', 2.0)]
max_steps = 4
state_indices = [1, 2, 4]  # pre-disturbance, disturbance, assessment
guild_to_species = [[1.0, 0.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 0.0, 1.0], [0.25, -0.15, 0.35, -0.2, 0.4]]

base_competition = [[[0.0, 0.515, 0.858, 1.099, 0.88, 0.244], [0.218, 0.0, 0.792, 0.794, 0.891, 0.134], [1.02, 0.121, 0.0, 0.869, 0.803, 0.199], [0.396, 0.345, 0.452, 0.0, 0.859, 0.543], [0.876, 0.526, 0.397, 0.614, 0.0, 0.387], [0.398, 0.691, 0.511, 0.179, 0.331, 0.0]], [[0.0, 0.685, 1.068, 1.013, 0.914, 0.3], [0.205, 0.0, 0.669, 0.87, 0.757, 0.15], [1.302, 0.093, 0.0, 0.783, 0.697, 0.25], [0.292, 0.416, 0.471, 0.0, 0.952, 0.72], [0.828, 0.54, 0.443, 0.606, 0.0, 0.406], [0.372, 0.636, 0.487, 0.133, 0.421, 0.0]], [[0.0, 0.628, 1.112, 1.418, 1.106, 0.223], [0.226, 0.0, 1.133, 0.992, 1.014, 0.177], [1.352, 0.158, 0.0, 1.025, 0.93, 0.232], [0.337, 0.409, 0.347, 0.0, 1.136, 0.366], [1.115, 0.593, 0.438, 0.666, 0.0, 0.469], [0.465, 0.774, 0.403, 0.219, 0.304, 0.0]]]
base_designs = [[0.614, 1.117, 0.817, 1.164, 1.254, 0.973], [1.459, 0.848, 1.019, 0.854, 1.278, 0.643], [0.595, 1.436, 0.855, 0.827, 0.9, 0.781], [0.922, 1.044, 0.85, 1.174, 1.472, 0.922], [1.222, 1.284, 0.928, 0.63, 1.255, 0.553], [1.044, 1.404, 0.966, 0.579, 0.665, 0.716]]
base_loadings = [[[0.306, 0.292], [-0.313, 0.23], [-0.395, 0.32], [0.382, 0.177], [0.361, 0.427], [-0.416, -0.493]], [[-0.198, 0.303], [-0.129, 0.568], [-0.205, -0.096], [0.252, 0.326], [-0.297, 0.106], [-0.056, -0.095]], [[-0.008, 0.031], [-0.332, 0.017], [0.395, 0.785], [0.069, 0.69], [0.083, -0.445], [-0.045, -0.223]]]
base_reserves = [0.032, 0.045, 0.036, 0.027, 0.041, 0.034]
upper_efforts = [8.0, 7.6, 8.4, 7.9, 8.2, 7.7]
plan_costs = [1.0, 1.0, 1.0, 0.97672, 1.0, 1.0]
plan_policy = [[3.0, 0.05], [3.0, 0.05], [3.0, 0.05], [3.0, 0.05], [3.0, 0.05], [3.0, 0.05]]  # columns are gain, penalty
buffer = 1.2
minimum_effort = 0.2

Use natural logarithms, float64, numpy.linalg.solve, numpy.linalg.eigvals, and numpy.linalg.eigvalsh. Do not round an intermediate value.

Output Format Requirements:
Emit <final_answer> immediately, then <reasoning>. Do not write a long derivation before the tags.
You must emit exactly one finite decimal inside <final_answer>...</final_answer>, even if the value is approximate or you are unsure.
Rules:
- The tags are required. Do not omit them or leave them empty.
- The tagged value must be a finite decimal (examples: 0.4847, 0.22, 1.05). Not NaN, not Inf, not a fraction string, not a vector, not prose. Report the selected plan's normalized score to ten decimal places.
- Put only that one number between the tags. No units, no words, no extra lines. Do not put a plan name, unit, interval, or other text inside the tag.
Keep <reasoning> short (a few hundred words). Report the six noncircular breadths, the complete overlap matrix, all five MPD values and the selected candidate, the resilience profile and exposure, every plan's feasibility thresholds, stability boundaries and lower effort, every plan's optimum effort, climate radius, raw utility and normalized score, the selected plan, and the closest rejected score.
Do not include code or repeat the input arrays in the reasoning.

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

Calculate the noncircular niche profile

Goal
----
Calculate the noncircular niche profile.

```python
def noncircular_niche_profile(resource_matrix: "np.ndarray", k: float) -> "np.ndarray":
    """Calculate the noncircular niche profile.

    Returns
    -------
    A float64 vector containing six breadths followed by the row-major 6 by 6 overlap matrix.

    Raises
    ------
    ValueError
        If an input violates the stated contract.
    """
    return result
```

### Step 2

Calibrate the PETRA forecast

Goal
----
Calibrate the PETRA forecast.

```python
def petra_calibration(reference: "np.ndarray", calibration_targets: "np.ndarray", candidates: list[tuple[int, float, int, str, float]], max_steps: int) -> "np.ndarray":
    """Calibrate the PETRA forecast.

    Returns
    -------
    A float64 vector containing the selected zero-based candidate followed by every MPD score.

    Raises
    ------
    ValueError
        If an input violates the stated contract.
    """
    return result
```

### Step 3

Build the PETRA resilience and exposure signal.

Goal
----
Build the PETRA resilience and exposure signal.

```python
def petra_residual_signal(reference: "np.ndarray", disturbed: "np.ndarray", candidate: tuple[int, float, int, str, float], max_steps: int, indices: "np.ndarray", guild_map: "np.ndarray") -> "np.ndarray":
    """Build the PETRA resilience and exposure signal.

    Returns
    -------
    A float64 vector containing four resilience measures followed by six signed species exposures.

    Raises
    ------
    ValueError
        If an input violates the stated contract.
    """
    return result
```

### Step 4

Assemble the coupled ecological system.

Goal
----
Assemble the coupled ecological system.

```python
def assemble_coupled_system(base_competition: "np.ndarray", base_designs: "np.ndarray", base_loadings: "np.ndarray", base_reserves: "np.ndarray", niche_profile: "np.ndarray", petra_signal: "np.ndarray", coefficients: "np.ndarray") -> "np.ndarray":
    """Assemble the coupled ecological system.

    Returns
    -------
    One float64 vector formed by flattening and concatenating competition, designs, loadings, and reserves in that order.

    Raises
    ------
    ValueError
        If an input violates the stated contract.
    """
    return result
```

### Step 5

Calculate feasibility and stability certificates.

Goal
----
Calculate feasibility and stability certificates.

```python
def plan_certificates(competition: "np.ndarray", designs: "np.ndarray", buffer: float, minimum_effort: float) -> "np.ndarray":
    """Calculate feasibility and stability certificates.

    Returns
    -------
    A float64 matrix with three feasibility thresholds, three stability boundaries, and the lower effort per plan.

    Raises
    ------
    ValueError
        If an input violates the stated contract.
    """
    return result
```

### Step 6

Calculate correlated disturbance radii.

Goal
----
Calculate correlated disturbance radii.

```python
def population_radius_panel(competition: "np.ndarray", slopes: "np.ndarray", loadings: "np.ndarray", reserves: "np.ndarray", effort: float) -> "np.ndarray":
    """Calculate correlated disturbance radii.

    Returns
    -------
    A float64 condition-by-population matrix of correlated-disk radii.

    Raises
    ------
    ValueError
        If an input violates the stated contract.
    """
    return result
```

### Step 7

Find a plan's continuous effort optimum.

Goal
----
Find a plan's continuous effort optimum.

```python
def continuous_plan_optimum(canonical_panel: "np.ndarray", slopes: "np.ndarray", loadings: "np.ndarray", reserves: "np.ndarray", interval: "np.ndarray", policy: "np.ndarray") -> "np.ndarray":
    """Find a plan's continuous effort optimum.

    Returns
    -------
    A float64 vector `(effort, climate_radius, raw_utility)`.

    Raises
    ------
    ValueError
        If an input violates the stated contract.
    """
    return result
```

### Step 8

Score every management plan.

Goal
----
Score every management plan.

```python
def management_score_vector(competition: "np.ndarray", designs: "np.ndarray", loadings: "np.ndarray", reserves: "np.ndarray", upper_efforts: "np.ndarray", costs: "np.ndarray", penalties: "np.ndarray", buffer: float, minimum_effort: float) -> "np.ndarray":
    """Score every management plan.

    Returns
    -------
    A float64 matrix with lower effort, optimum effort, radius, raw utility, and normalized score per plan.

    Raises
    ------
    ValueError
        If an input violates the stated contract.
    """
    return result
```

### Step 9

Select the robust management plan.

Goal
----
Select the robust management plan.

```python
def select_management_plan(score_table: "np.ndarray", tie_tolerance: float) -> "np.ndarray":
    """Select the robust management plan.

    Returns
    -------
    A float64 vector `(zero_based_plan, normalized_score)`, or `(-1, 0)` when none is available.

    Raises
    ------
    ValueError
        If an input violates the stated contract.
    """
    return result
```

### Step 10

Run the complete coupled pipeline.

Goal
----
Run the complete coupled pipeline.

```python
def select_coupled_management(resource_matrix: "np.ndarray", k: float, reference: "np.ndarray", calibration_targets: "np.ndarray", disturbed: "np.ndarray", candidates: list[tuple[int, float, int, str, float]], max_steps: int, indices: "np.ndarray", guild_map: "np.ndarray", base_competition: "np.ndarray", base_designs: "np.ndarray", base_loadings: "np.ndarray", base_reserves: "np.ndarray", upper_efforts: "np.ndarray", costs: "np.ndarray", penalties: "np.ndarray", coefficients: "np.ndarray", buffer: float, minimum_effort: float) -> float:
    """Run the complete coupled pipeline.

    Returns
    -------
    One float equal to the selected plan's cost-normalized robust utility.

    Raises
    ------
    ValueError
        If an input violates the stated contract.
    """
    return result
```
