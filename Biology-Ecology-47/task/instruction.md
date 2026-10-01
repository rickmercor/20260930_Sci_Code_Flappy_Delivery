# Biology-Ecology-47

## Background

The resource calculation uses modified resource-state weights and keeps the complete-table occupied-state count during focal exclusions. PETRA-EDR then calibrates a trajectory forecast with trajectory-level distances, within-trajectory movement, strict neighbourhood boundaries, and the published resilience signs. The resulting niche and trajectory quantities alter competition, self-regulation, environmental loadings, reserves, pulse contraction, and climate covariance.
The coexistence certificate and the periodic risk calculation answer different questions. The first gives source-defined finite-community feasibility and symmetric-part stability bounds. The second checks the ordered two-season map and propagates correlated innovation through a discrete Lyapunov equation. The optimizer must therefore preserve feasibility, cycle stability, covariance, active condition changes, and cross-plan regret at the same effort. Several plausible shortcuts produce smooth numerical answers but select a different plan.

## Problem

Implement the ten public functions in the sub-problems and use the complete chain to select one management plan for the six-species community below. The calculation combines noncircular niche geometry, PETRA-EDR calibration, finite-community coexistence certificates, a two-season Floquet map, correlated climate innovations, stationary cycle covariance, continuous effort optimization, and cross-plan regret control. Use the source-defined ecological methods where named and the task-defined coupling rules exactly; circular niche weights, reduced occupancy counts, state-level PETRA weights, continuous-time stability alone, diagonal climate covariance, reversed seasonal products, endpoint-only effort searches, scenario averaging, and cost normalization before optimization are not substitutes.
For each plan and climate condition, let M(a) = a*diag(d) + B and solve M(a)x = 1. Form dry and wet Jacobians J0 = -diag(x*s0)M and J1 = -diag(x*s1)M, where s0[i] = 0.78 + 0.55*abs(e[i]) + 0.08*c and s1[i] = 1.22 - 0.35*abs(e[i]) - 0.05*c for zero-based climate condition c. With the supplied durations, use E0 = expm(J0*t0), E1 = expm(J1*t1), P = diag(exp(-(pulse_decay + contraction_gain*a))), and the ordered cycle map F = E1@P@E0.
Let G0 = diag(x)@U and G1 = diag(x)@(U@R_c), where R_c = [[0.82+0.03*c,-0.21],[0.17,0.91-0.02*c]]. Set Q0 = G0@Sigma_c@G0.T + diag(nu_c*x*x), Q1 = G1@Sigma_c@G1.T + diag((1.15-0.08*c)*nu_c*x*x), and Qcycle = E1@P@Q0@P.T@E1.T + Q1. A condition is usable only when every equilibrium density exceeds its reserve and the spectral radius of F is strictly below stability_limit; for a usable condition solve C = F@C@F.T + Qcycle and take the security radius as min_i (x_i-reserve_i)/(chi_radius*sqrt(C_ii)).
The five policy columns are gain, effort penalty, covariance penalty, Floquet penalty, and curvature. At effort a the condition score is [security - covariance_penalty*log1p(trace(C)/(x dot x)) - Floquet_penalty*rho(F) + gain*log1p(a) - effort_penalty*a - curvature*(a-6.1)^2]/cost. For each plan maximize the minimum condition score on its closed certificate interval using the stated 129-point scan and bounded local refinements, then compute each condition's regret against the best plan at its own common optimum, average the two largest regrets, and subtract regret_weight times that tail regret from the plan's robust score; reject a plan when the tail regret exceeds regret_limit.
Select the largest finite decision score, treating scores within tie_tolerance as tied and choosing the earliest zero-based plan. Use float64 throughout, retain the listed order of species, conditions, candidates, and plans, and do not round intermediate values. In the reasoning report the noncircular breadths and overlap matrix, all calibration MPDs and the selected PETRA candidate, the resilience profile and signed exposure, all plan certificate lower bounds, every plan's optimized effort and three condition scores, the robust scores, tail regrets, decision scores, selected plan, and the endpoint-only and scenario-mean near misses.
Fixed coupling rules
Write the eleven coefficient entries as tau0, tau1, chi, db, de, du, dl, dr, dp, dc, and di. Let breadth be the first six entries of the niche profile, overlap its remaining 36 entries reshaped to 6 by 6, and let resistance, amplitude, recovery, net change, and the six signed species exposures be the PETRA output. For zero-based climate condition c, set tau[c] = tau0 + tau1*[net change, amplitude, recovery][c] and direction[i,j] = exposure[i] - exposure[j]. Calculate
B[c,i,j] = base_competition[c,i,j]*max(1 + tau[c]*overlap[i,j] + chi*direction[i,j], 0.08),
then set every diagonal entry of B to zero. For plan p and species i, set
D[p,i] = base_designs[p,i]*(1 + db*(1-breadth[i]))*(1 + de*abs(exposure[i])).
Copy base_loadings to U, then add du*exposure[i] to U[c,i,0] and subtract 0.7*du*exposure[i] from U[c,i,1]. Set
reserve[i] = base_reserves[i]*(1 + dl*abs(exposure[i]))*(1 + dr*(1-resistance)),
pulse[p,i] = pulse_decay_base[p,i]*(1 + dp*(1-breadth[i]))*(1 + 0.5*dp*abs(exposure[i])),
and nu[c,i] = idiosyncratic_noise_base[c,i]*(1 + di*abs(exposure[i])). Add dc*[net change, -amplitude, recovery][c] to climate_covariance_base[c,0,1] and copy that upper off-diagonal to [c,1,0]. The adjusted climate matrices must remain positive definite. Pack the arrays in this exact order: B, D, U, reserve, pulse, climate covariance, nu, each flattened in C order.
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
base_reserves = [0.012, 0.015, 0.013, 0.010, 0.014, 0.012]
upper_efforts = [8.0, 7.6, 8.4, 7.9, 8.2, 7.7]
plan_costs = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
plan_policy = [[1.8, 0.10, 0.12, 0.18, 0.06], [1.8, 0.10, 0.12, 0.18, 0.06], [1.8, 0.10, 0.12, 0.18, 0.06], [1.8, 0.10, 0.12, 0.18, 0.06], [1.8, 0.10, 0.12, 0.18, 0.06], [1.8, 0.10, 0.12, 0.18, 0.06]]  # gain, effort penalty, covariance penalty, Floquet penalty, curvature
buffer = 1.2
minimum_effort = 0.2
pulse_decay_base = [[0.36, 0.41, 0.34, 0.44, 0.39, 0.33], [0.43, 0.31, 0.40, 0.35, 0.46, 0.29], [0.32, 0.47, 0.37, 0.30, 0.41, 0.38], [0.39, 0.35, 0.29, 0.46, 0.33, 0.42], [0.45, 0.38, 0.43, 0.28, 0.36, 0.31], [0.34, 0.44, 0.32, 0.40, 0.30, 0.47]]
climate_covariance_base = [[[1.0, 0.72], [0.72, 1.25]], [[0.85, -0.63], [-0.63, 1.35]], [[1.30, 0.81], [0.81, 0.95]]]
idiosyncratic_noise_base = [[0.0018, 0.0022, 0.0016, 0.0025, 0.0019, 0.0021], [0.0021, 0.0017, 0.0024, 0.0018, 0.0023, 0.0016], [0.0016, 0.0025, 0.0020, 0.0022, 0.0017, 0.0024]]
season_durations = [[0.43, 0.57], [0.51, 0.49], [0.61, 0.39]]
contraction_gain = 0.045
stability_limit = 0.91
chi_radius = 2.447746830680816
regret_weight = 0.06
regret_limit = 0.75
tie_tolerance = 1.0e-9
coefficients = [1.5, 6.0, 7.0, 0.8, 3.0, 5.0, 6.0, 1.5, 0.55, 0.9, 0.45]
Output format
Start with the final answer and then give the reasoning. Put exactly one finite decimal number inside the final-answer tag and report the selected plan's decision score to ten decimal places. Put no plan name, unit, interval, or other text inside the tag. Put no text outside the two tags.
<final_answer>NUMBER</final_answer>
<reasoning>
REASONING
</reasoning>

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

Noncircular Niche Profile

Goal
----
Calculate the modified noncircular niche profile.

```python
def noncircular_niche_profile(resource_matrix: "np.ndarray", k: float) -> "np.ndarray":
    """Calculate the noncircular niche profile.
 
    Returns
    -------
    A float64 vector containing s breadths followed by the row-major s by s overlap matrix.
 
    Raises
    ------
    ValueError
        If an input violates the stated contract.
    """
    return result
```

### Step 2

Petra Calibration

Goal
----
Calibrate the PETRA-EDR forecast.

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

Petra Residual Signal

Goal
----
Calculate PETRA-EDR resilience and signed exposure.

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

Assemble Periodic System

Goal
----
Assemble the trajectory-conditioned periodic community system.

```python
def assemble_periodic_system(base_competition: "np.ndarray", base_designs: "np.ndarray", base_loadings: "np.ndarray", base_reserves: "np.ndarray", pulse_decay_base: "np.ndarray", climate_covariance_base: "np.ndarray", idiosyncratic_noise_base: "np.ndarray", niche_profile: "np.ndarray", petra_signal: "np.ndarray", coefficients: "np.ndarray") -> "np.ndarray":
    """Assemble the trajectory-conditioned periodic community arrays in the stated packed order.
 
    Returns
    -------
    The numerical object stated in the step contract.
 
    Raises
    ------
    ValueError
        If an input violates the stated contract.
    """
    return result
```

### Step 5

Plan Certificates

Goal
----
Calculate finite-community feasibility and stability certificates.

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

Periodic Risk Panel

Goal
----
Calculate the periodic risk panel for one management plan.

```python
def periodic_risk_panel(competition: "np.ndarray", slope: "np.ndarray", loadings: "np.ndarray", reserves: "np.ndarray", pulse_decay: "np.ndarray", climate_covariance: "np.ndarray", idiosyncratic_noise: "np.ndarray", durations: "np.ndarray", exposure: "np.ndarray", effort: float, contraction_gain: float, stability_limit: float, chi_radius: float) -> "np.ndarray":
    """Calculate equilibrium, Floquet, covariance, and security diagnostics for one plan.
 
    Returns
    -------
    The numerical object stated in the step contract.
 
    Raises
    ------
    ValueError
        If an input violates the stated contract.
    """
    return result
```

### Step 7

Continuous Plan Profile

Goal
----
Optimize one plan under all periodic climate conditions.

```python
def continuous_plan_profile(competition: "np.ndarray", slope: "np.ndarray", loadings: "np.ndarray", reserves: "np.ndarray", pulse_decay: "np.ndarray", climate_covariance: "np.ndarray", idiosyncratic_noise: "np.ndarray", durations: "np.ndarray", exposure: "np.ndarray", interval: "np.ndarray", policy: "np.ndarray", cost: float, contraction_gain: float, stability_limit: float, chi_radius: float) -> "np.ndarray":
    """Optimize one plan on its closed certificate interval and return the common-effort profile.
 
    Returns
    -------
    The numerical object stated in the step contract.
 
    Raises
    ------
    ValueError
        If an input violates the stated contract.
    """
    return result
```

### Step 8

Management Profile Table

Goal
----
Calculate the optimized profile table for every plan.

```python
def management_profile_table(competition: "np.ndarray", slopes: "np.ndarray", loadings: "np.ndarray", reserves: "np.ndarray", pulse_decay: "np.ndarray", climate_covariance: "np.ndarray", idiosyncratic_noise: "np.ndarray", durations: "np.ndarray", exposure: "np.ndarray", certificates: "np.ndarray", upper_efforts: "np.ndarray", policies: "np.ndarray", costs: "np.ndarray", contraction_gain: float, stability_limit: float, chi_radius: float) -> "np.ndarray":
    """Calculate the continuous common-effort profile for every management plan.
 
    Returns
    -------
    The numerical object stated in the step contract.
 
    Raises
    ------
    ValueError
        If an input violates the stated contract.
    """
    return result
```

### Step 9

Select Regret Controlled Plan

Goal
----
Select the regret-controlled management plan.

```python
def select_regret_controlled_plan(profile_table: "np.ndarray", regret_weight: float, regret_limit: float, tie_tolerance: float) -> "np.ndarray":
    """Apply cross-plan tail regret and return the selected summary followed by the complete plan audit.
 
    Returns
    -------
    The numerical object stated in the step contract.
 
    Raises
    ------
    ValueError
        If an input violates the stated contract.
    """
    return result
```

### Step 10

Select Periodic Management

Goal
----
Run the complete periodic-disturbance management pipeline.

```python
def select_periodic_management(resource_matrix: "np.ndarray", k: float, reference: "np.ndarray", calibration_targets: "np.ndarray", disturbed: "np.ndarray", candidates: list[tuple[int, float, int, str, float]], max_steps: int, indices: "np.ndarray", guild_map: "np.ndarray", base_competition: "np.ndarray", base_designs: "np.ndarray", base_loadings: "np.ndarray", base_reserves: "np.ndarray", pulse_decay_base: "np.ndarray", climate_covariance_base: "np.ndarray", idiosyncratic_noise_base: "np.ndarray", durations: "np.ndarray", upper_efforts: "np.ndarray", costs: "np.ndarray", policies: "np.ndarray", coefficients: "np.ndarray", buffer: float, minimum_effort: float, contraction_gain: float, stability_limit: float, chi_radius: float, regret_weight: float, regret_limit: float, tie_tolerance: float) -> float:
    """Run the full ecology pipeline and return the selected regret-controlled decision score.
 
    Returns
    -------
    The numerical object stated in the step contract.
 
    Raises
    ------
    ValueError
        If an input violates the stated contract.
    """
    return result
```
