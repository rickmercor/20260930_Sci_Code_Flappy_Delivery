# taguchi_mixed_robust_20260827

## Background

This task evaluates a fixed numerical archive through competing mathematical conventions and a shared-parameter consistency test. The supplied coefficients define an author-created surrogate instance, not a direct numerical reproduction of the cited study. Source selection determines the conventions; the stated data and admissibility rules determine the numerical comparison.

## Problem

Using the primary study cited in Browsing Sources, identify the zero-based source-consistent row in each archived formulation family and determine the runner-up-minus-winner closure gap for the supplied joint nonlinear-fiber design.

| Family | Candidate 0 | Candidate 1 | Candidate 2 | Candidate 3 |
|---|---|---|---|---|
| DDF profile `q(z)=beta2(z)/beta2(0)` | `1+(C0*a1)z+(C0*a2*z)^2/2+C0*(a3*z)^3/6` | `1+C0*(a1*z+(a2*z)^2/2+(a3*z)^3/6)` | `1+(C0*a1)z+(C0*a2*z)^2/2+(C0*a3*z)^3/6` | `1+C0*(a1*z+a2*z^2/2+a3*z^3/6)` |
| response from `[dN_i,dP,dT]` | `sqrt(dP^2+dT^2)` | `max_i(|dN_i|,|dP|,|dT|)` | `sqrt(mean(dN_i^2))` | `sqrt(mean(dN_i^2)+dP^2+dT^2)` |
| factor-level selection | settings of the single run minimizing `f` | factorwise maxima of level-mean `-20*log10(f)` | factorwise minima of level-mean `f` | factorwise maxima of level-median `-20*log10(f)` |
| guiding-center multiplier `m(G)` | `(G-1)/(G*ln(G))` | `G*(G-1)/ln(G)` | `G*ln(G)/(G-1)` | `ln(G)/(G-1)` |

Use the source-defined three-level initialization and factor-level update without intermediate rounding. For reproducible OA indexing, enumerate `(a,b)` lexicographically for `a,b in {0,1,2}` and use columns `[a,b,(a+b) mod 3,(a+2b) mod 3]`; a two-factor campaign uses the first two columns.

- DDF archive: four iterations; factors `[A1,A2,A3,C0]` with `ai=10^Ai`; initialization intervals `Ai:[-1.9,-0.8]`, `C0:[-1.25,-0.75]`; rates `[0.50,0.65,0.75]`; training locations `[0.75,2.25,3.75,5.25,6.75,8.25,9.75] km`; validation folds `[[1.5,4.5,7.5],[3,6,9]] km`; reference losses `[0.244,0.281] dB/km`; per-run response equal to the worst-reference RMS order error induced by the candidate profile relative to `q_ref(z)=exp[-alpha*z]`, where `alpha=alpha_dB*ln(10)/10` and `N=sqrt(q_ref/q)`; final nested orders `o=1,2,3`, admissible only when positive on `z=0,0.05,...,10 km`, with the lowest order within 3% of the best validation `E_D,o` retained; `alpha_D=-sum(z*ln(q_o(z)))/sum(z^2)` over both folds.

- Guiding-center archive: five iterations for every rate, span, and target; factors `[G,P_fac]` with initialization interval `[1,10]` for each factor; `P_launch=P_fac*P0`; `beta2=-7650 fs^2/m`, `gamma=1.3 W^-1 km^-1`, intensity-`sech^2` FWHM `50 ps`, loss `0.2611 dB/km`, spans `[0.17,0.24]*105.5 km`; `P0` and each theoretical state recovered from the paper and selected multiplier; separate targets `N*=1` and `N_T`; trial coordinates `x=G/G_T-1`, `y=P_fac/P_fac,T-1`, `u=[1,x,y,xy,x^2,y^2]`; `u@B_j` ordered as four signed deviations `N_i-N_T` and the relative terminal power and width errors, from which the target-specific order residuals are reconstructed before applying the selected response.

```text
B_0 = [[ 0.026,-0.021, 0.017,-0.014, 0.010,-0.008],
       [ 0.470, 0.390, 0.540, 0.440, 0.260,-0.200],
       [ 0.310, 0.370, 0.270, 0.340, 0.320, 0.250],
       [ 0.130,-0.070, 0.090,-0.050,-0.100, 0.080],
       [ 0.070, 0.060, 0.080, 0.050, 0.040, 0.010],
       [-0.030,-0.050,-0.020,-0.040, 0.010, 0.030]]
B_1 = [[ 0.034,-0.028, 0.023,-0.017, 0.014,-0.011],
       [ 0.560, 0.460, 0.630, 0.500, 0.310,-0.240],
       [ 0.360, 0.440, 0.320, 0.400, 0.380, 0.300],
       [ 0.100,-0.090, 0.120,-0.070,-0.120, 0.100],
       [ 0.090, 0.040, 0.060, 0.070, 0.060,-0.010],
       [-0.050,-0.070,-0.040,-0.060,-0.010, 0.050]]
```

- Closure: at each rate's final centers, `E_1` and `E_T` are the worse-span responses for the unit and theoretical targets; `alpha_G=sum(L_j*ln(G_j))/sum(L_j^2)` and `epsilon_P=max_j |P_fac,j/m(G_j)-1|` use the theoretical-target centers; set

`J_r=max(E_D/tau_D,E_1/tau_1,E_T/tau_T,|alpha_D-alpha_G|/tau_alpha,epsilon_P/tau_P)`

for `[tau_D,tau_1,tau_T,tau_alpha,tau_P]=[0.05,0.24,0.06,0.002,0.10]`, rank by `(J_r,r)`, and return the second-smallest score minus the smallest score.

- Reported certificate: in `<reasoning>`, give the convention quartet and the source signature that distinguishes each selected row: the physically convergent DDF factor combinations, the role of the two terminal guiding-center terms, the complete factorwise selection-and-recentering operation, and the coupled periodic-amplifier gain/launch relation. Also give retained-order, joint-score, and zero-based active-channel vectors for rates `[0.50,0.65,0.75]` in that order, with channels `[DDF,unit,theory,shared-loss,P0]`; for each rate, the pair `(best nested E_D,cubic E_D)`; winning `alpha_D` and `alpha_G`; and the rate and response of the DDF-only minimizer. Numerical diagnostics use absolute tolerance `1e-8`.

Reproducibility conventions:
- The stated factor intervals are used only to initialize levels; subsequent recentering and contraction are not clipped to those intervals.
- Exact single-run ties select the lowest OA row index. Exact factorwise ties select the lowest level index. Zero response has positive-infinite SNR; means and medians use the extended-real convention, so a mean containing positive infinity is positive infinity. For an even-sized median, average the two middle values; finite plus positive infinity is positive infinity. No positive floor is substituted for zero.
- A DDF training trial with any nonpositive profile value receives response 1000000.0. For each nested validation order, any nonpositive profile value receives score 1000000.0; otherwise E_D,o is the maximum over all fold/reference pairs of their location-wise RMS order error. Dense-grid positivity is checked separately. Among dense-feasible orders, let best be the smallest score and select the lowest order with E_D,o <= best*(1+0.03)+1e-15. If none is dense-feasible, the campaign is inadmissible. An order with the sentinel score remains eligible if it is dense-feasible and satisfies that same comparison.
- Exact ties between normalized closure channels select the lowest channel index. A selected DDF profile must be positive at every validation location. Final theoretical-target gains and launch factors must be positive and finite. Inferred fitted loss slopes may be signed. Do not clip these slopes or introduce a gain floor. An inadmissible or numerically unrepresentable alternative API instance raises ValueError; the stated default instance is admissible.

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

taguchi_initial_levels

Goal
----
Construct interior factor levels and their common difference.

```python
import numpy as np

def taguchi_initial_levels(v_min: float, v_max: float, s: int=3) -> np.ndarray:
    """Return interior levels followed by their level difference.

Parameters
----------
v_min, v_max : float
    Strict lower and upper factor bounds.
s : int
    Positive number of interior levels.

Returns
-------
out : numpy.ndarray
    Float array of shape `(s+1,)` ordered as
    `[level_1, ..., level_s, level_difference]`.

Conventions
-----------
Require finite v_min < v_max and a positive integer s. The returned positive spacing must be representable as float64. These intervals initialize levels; this function imposes no subsequent campaign clipping.

Raises
------
ValueError
    If the stated input conditions, representability conditions or admissibility conditions fail."""
    return np.zeros(s + 1, dtype=float)
```

### Step 2

prime_strength2_oa

Goal
----
Construct the canonical prime-level strength-two orthogonal array.

```python
import numpy as np

def prime_strength2_oa(q: int, k: int) -> np.ndarray:
    """Return the canonical `OA(q^2,k,q,2)` for prime `q`.

Parameters
----------
q : int
    Prime number of levels.
k : int
    Number of factors, satisfying `2 <= k <= q+1`.

Returns
-------
oa : numpy.ndarray
    Integer array of shape `(q*q, k)`. Rows are lexicographic in
    `(a,b)`; columns are `[a,b,a+b,a+2b,...]` modulo `q`.

Conventions
-----------
The signature returns an integer placeholder with shape (q*q,k); it does not implement the OA construction."""
    return np.zeros((q * q, k), dtype=int)
```

### Step 3

sech_soliton_peak_power_mw

Goal
----
Evaluate first-order intensity-sech-squared peak-power normalization.

```python
import numpy as np

def sech_soliton_peak_power_mw(beta2_fs2_per_m: float, gamma_w_inv_km: float, fwhm_ps: float) -> float:
    """Return the first-order `sech` soliton peak power in mW.

Parameters
----------
beta2_fs2_per_m : float
    Group-velocity dispersion in `fs^2/m`; its magnitude is used.
gamma_w_inv_km : float
    Positive nonlinear coefficient in `W^-1 km^-1`.
fwhm_ps : float
    Positive intensity FWHM in ps.

Returns
-------
power_mw : float
    First-order-soliton peak power in mW.

Conventions
-----------
All inputs must be finite; gamma and FWHM must be positive. Zero dispersion returns exactly zero. Otherwise the returned power must be finite and strictly positive in float64.

Raises
------
ValueError
    If the stated input conditions, representability conditions or admissibility conditions fail."""
    return 0.0
```

### Step 4

guiding_center_theory_candidates

Goal
----
Evaluate four archived guiding-center launch-state conventions.

```python
import numpy as np

def guiding_center_theory_candidates(alpha_db_per_km: float, span_km: float, p0_mw: float) -> np.ndarray:
    """Return four candidate guiding-center launch states.

Parameters
----------
alpha_db_per_km : float
    Nonnegative power attenuation in dB/km.
span_km : float
    Nonnegative amplifier spacing in km.
p0_mw : float
    Positive first-order-soliton peak power in mW.

Returns
-------
states : numpy.ndarray
    Float array of shape `(4,4)`. Rows follow the declared candidate order;
    columns are `[integrated_loss_exponent,gain,launch_power_mW,target_order]`, where
    `integrated_loss_exponent = alpha_db_per_km*span_km*ln(10)/10` is dimensionless.

Conventions
-----------
All inputs must be finite. Loss and span are nonnegative; p0_mw is positive. At zero integrated loss, every multiplier, gain and order equals one. All gains, launch powers and target orders must be finite and strictly positive in float64.

Raises
------
ValueError
    If the stated input conditions, representability conditions or admissibility conditions fail."""
    return np.zeros((4, 4), dtype=float)
```

### Step 5

guiding_center_response_candidates

Goal
----
Evaluate four archived guiding-center response conventions.

```python
import numpy as np

def guiding_center_response_candidates(order_residuals: np.ndarray, power_error: float, width_error: float) -> np.ndarray:
    """Return the four candidate dimensionless responses.

Parameters
----------
order_residuals : numpy.ndarray
    Nonempty one-dimensional residuals `N_i-N_target`.
power_error, width_error : float
    Finite relative final-versus-launch errors.

Returns
-------
responses : numpy.ndarray
    Float array of shape `(4,)` ordered as `[terminal_RSS,
    componentwise_maximum,order_RMS,full_RSS]`.

Conventions
-----------
All residuals must be finite. Evaluate the order RMS without avoidable overflow from squaring. Zero residuals give zero response. Every returned component must be finite in float64.

Raises
------
ValueError
    If the stated input conditions, representability conditions or admissibility conditions fail."""
    return np.zeros(4, dtype=float)
```

### Step 6

dispersion_profile_candidates

Goal
----
Evaluate four archived cubic DDF-profile conventions.

```python
import numpy as np

def dispersion_profile_candidates(log_a: np.ndarray, c0: float, z_km: np.ndarray) -> np.ndarray:
    """Return four normalized candidate profiles.

Parameters
----------
log_a : numpy.ndarray
    Length-3 base-10 logarithmic factors `[A1,A2,A3]`.
c0 : float
    Finite signed coefficient.
z_km : numpy.ndarray
    Nonempty one-dimensional nonnegative distances in km.

Returns
-------
profiles : numpy.ndarray
    Float array of shape `(4,len(z_km))`; rows follow candidate order
    `0,1,2,3` from the task table.

Conventions
-----------
All inputs must be finite. If c0 is zero, every profile equals one for any finite log_a and distance. At zero distance, every profile equals one. Negative finite profiles are returned without clipping; admissibility is decided by callers. Every returned component must be finite in float64.

Raises
------
ValueError
    If the stated input conditions, representability conditions or admissibility conditions fail."""
    return np.zeros((4, np.asarray(z_km).size), dtype=float)
```

### Step 7

taguchi_protocol_candidates

Goal
----
Evaluate four archived factor-level selection protocols.

```python
import numpy as np

def taguchi_protocol_candidates(levels: np.ndarray, oa: np.ndarray, responses: np.ndarray, level_differences: np.ndarray, reduction_rate: float) -> np.ndarray:
    """Return contracted updates for four candidate protocols.

Parameters
----------
levels : numpy.ndarray
    Float array of shape `(K,s)` in ascending level order.
oa : numpy.ndarray
    Integer array of shape `(M,K)` with zero-based levels.
responses : numpy.ndarray
    Length-`M` finite nonnegative smaller-is-better responses.
level_differences : numpy.ndarray
    Length-`K` positive current spacings.
reduction_rate : float
    Finite contraction rate in `(0,1)`.

Returns
-------
updates : numpy.ndarray
    Float array of shape `(4,K,s+3)`. Rows follow the task's candidate
    order; for each factor, columns are
    `[selected_index,center,new_spacing,new_levels...]`.

Conventions
-----------
Require K>=1, s>=2, M>=1, strictly ascending finite levels, valid integer OA indices with every factor level represented, finite nonnegative responses and finite positive spacings. Use candidate order: single-run minimum response, maximum level-mean SNR, minimum level-mean response, maximum level-median SNR. SNR=-20*log10(f) for f>0 and positive infinity for f=0. A mean containing positive infinity is positive infinity; use the usual ordered median, averaging the two middle values for even counts, with finite plus positive infinity equal to positive infinity. Exact single-run ties choose the lowest OA row index; exact factorwise ties choose the lowest level index. Recenter each factor on its selected old level, multiply its spacing by reduction_rate, and use offsets arange(s)-(s-1)/2. Do not clip new levels to initialization intervals. Return finite float64 updates and strictly positive representable contracted spacings.

Raises
------
ValueError
    If the stated input conditions, representability conditions or admissibility conditions fail."""
    return np.zeros((4, np.asarray(levels).shape[0], np.asarray(levels).shape[1] + 3), dtype=float)
```

### Step 8

ddf_truncation_selector

Goal
----
Compare nested DDF profiles across folds and reference losses.

```python
import numpy as np

def ddf_truncation_selector(log_a: np.ndarray, c0: float, validation_z_km: np.ndarray, references: np.ndarray, dense_z_km: np.ndarray, relative_tolerance: float=0.03) -> np.ndarray:
    """Return robust nested-profile and identifiability diagnostics.

Parameters
----------
log_a : numpy.ndarray
    Length-3 base-10 logarithmic factors `[A1,A2,A3]`.
c0 : float
    Finite nonzero signed coefficient.
validation_z_km : numpy.ndarray
    Float array of shape `(F,Z)` containing validation locations by fold.
references : numpy.ndarray
    Strictly positive reference profiles of shape `(F,R,Z)`.
dense_z_km : numpy.ndarray
    Nonempty one-dimensional nonnegative feasibility grid.
relative_tolerance : float
    Nonnegative fractional score tolerance for choosing the lowest order.

Returns
-------
out : numpy.ndarray
    Float array of shape `(8,)` ordered as `[three_worst_order_RMS_scores,
    three_dense_profile_minima,selected_order,selected_score]`.

Conventions
-----------
For each nested order, a validation profile with any value <=0 receives score exactly 1000000.0; this is a sentinel, not an RMS. Otherwise calculate sqrt(mean((1-sqrt(reference/profile))**2)) over locations separately for every fold/reference pair, then take the maximum. Dense minima are calculated separately for each order. Dense feasibility means minimum>0; the validation sentinel alone does not remove a dense-feasible order. Let best be the minimum score among dense-feasible orders. Select the lowest order whose score is <= best*(1+relative_tolerance)+1e-15 and whose dense minimum is positive. The selected order is one-based. No dense-feasible order raises ValueError. Inputs and returned diagnostics must be finite; grids are nonnegative, c0 is nonzero, references positive and tolerance nonnegative.

Raises
------
ValueError
    If the stated input conditions, representability conditions or admissibility conditions fail."""
    return np.zeros(8, dtype=float)
```

### Step 9

cross_example_closure_score

Goal
----
Assemble the normalized shared-loss closure score.

```python
import numpy as np

def cross_example_closure_score(ddf_error: float, unit_order_response: float, theory_order_response: float, alpha_ddf: float, alpha_guiding: float, p0_relative_errors: np.ndarray, tolerances: np.ndarray) -> np.ndarray:
    """Return component errors and their normalized minimax score.

Parameters
----------
ddf_error, unit_order_response, theory_order_response : float
    Finite nonnegative archived response values.
alpha_ddf, alpha_guiding : float
    Finite inferred power-loss coefficients in `km^-1`.
p0_relative_errors : numpy.ndarray
    Nonempty one-dimensional relative launch-normalization errors.
tolerances : numpy.ndarray
    Five positive scales in DDF, unit-order, theoretical-order,
    shared-loss, and launch-normalization order.

Returns
-------
out : numpy.ndarray
    Shape `(12,)`, ordered as the six raw diagnostics
    `[DDF,unit,theory,alpha_DDF,alpha_guiding,max_abs_P0_error]`,
    the five normalized channels, and the joint maximum.

Conventions
-----------
Inferred alpha_ddf and alpha_guiding may have either sign; do not clip or reject finite negative slopes. Only the first three response arguments must be nonnegative. The loss channel is abs(alpha_ddf-alpha_guiding)/tolerances[3]. All inputs and returned diagnostics must be finite, and tolerances must be positive.

Raises
------
ValueError
    If the stated input conditions, representability conditions or admissibility conditions fail."""
    return np.zeros(12, dtype=float)
```

### Step 10

cross_example_taguchi_gap

Goal
----
Resolve the joint design and its runner-up score gap by composing every earlier public subproblem.

```python
import numpy as np

def cross_example_taguchi_gap(reduction_candidates: np.ndarray=None, surrogate_tensor: np.ndarray=None, closure_tolerances: np.ndarray=None, ddf_iterations: int=4, guiding_iterations: int=5) -> float:
    """Return the runner-up-minus-winner normalized closure score.

Parameters
----------
reduction_candidates : numpy.ndarray
    Three distinct rates in `(0,1)`; `None` selects the task instance.
surrogate_tensor : numpy.ndarray
    Shape `(2,6,6)`; span, polynomial-basis row, then four order
    residuals and two terminal residuals. `None` selects the task archive.
closure_tolerances : numpy.ndarray
    Five positive normalization scales. `None` selects the task scales.
ddf_iterations, guiding_iterations : int
    Positive modified-Taguchi iteration counts.

Returns
-------
gap : float
    Finite nonnegative difference between the two smallest joint scores.

Notes
-----
The submitted orchestrator must call and combine every earlier public subproblem
function; local reimplementations do not satisfy this integration contract.

Conventions
-----------
Use initialization-only intervals: no subsequent clipping. Use Step 7's exact-zero SNR and lowest-index tie rules. A finite surrogate tensor, including the zero tensor, is accepted for evaluation but does not guarantee a physically admissible final state. Raise ValueError if there is no positive dense-grid DDF order, if the selected DDF profile is nonpositive on a validation fold, if either final theoretical-target gain or launch factor is nonpositive or nonfinite, or if required diagnostics cannot be represented finitely. Inferred loss slopes may be signed. The default all-zero tensor with the default rates and iteration counts raises ValueError because deterministic recentering produces an inadmissible final theoretical state. No fallback clipping or gain floor is applied.

Raises
------
ValueError
    If the stated input conditions, representability conditions or admissibility conditions fail."""
    return 0.0
```
