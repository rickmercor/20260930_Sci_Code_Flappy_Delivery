# Biology-Genetics-11

## Background

All indices are zero-based. There are J=12 aligned SNPs, R=4 traits, N=260 individuals and at most L=5 additive single-effect components. The supplied decimals define exact benchmark inputs after binary64 parsing; do not round intermediate values. Traits are marginally standardized. The target sufficient statistics are Sxx=N*LD and the supplied Sxy; they refer to the same centered individuals and aligned SNPs. Residual covariance is fixed after training, so no Y-transpose-Y statistic or residual-covariance refitting is needed.

The matrix weak_z contains marginal Z scores from a separate genome-wide panel, for the source's standardized-trait weak-association estimator with cutoff 2. The matrix training_z contains one previously selected lead Z vector from each of 24 distinct training regions. These rows are noisy observations of a zero-centered multivariate Gaussian covariance mixture; every row has the residual correlation matrix inferred from weak_z as its observation-noise covariance. The target-region crossproducts are separate from both training panels.

There are four initial covariance patterns U0 and initial pattern weights w0. Keep the pattern means at zero, perform exactly 11 simultaneous EM updates of both pattern covariances and weights under the noisy lead-score model, then add 0.035 times the identity to every resulting covariance exactly once. Preserve pattern order and do not prune, split, merge, rescale or otherwise normalize the learned patterns. This fixed iteration count and terminal ridge are benchmark settings. Freeze this learned prior throughout target-region fitting.

A latent regional component selects a SNP with probabilities pi; conditional on covariance-pattern index k its effect vector has covariance scale*U[k]. All other SNP effect rows in that component are zero. Patterns have probabilities given by the learned weights; pi remains fixed. For each component update, choose its scale from scale_grid by the source's single-effect marginal-evidence criterion; this finite grid restricts the source optimization. Choose the smaller scale on an exact tie. Zero is an exact-null scale.

Initialize every component mean at zero, every component's SNP probabilities at pi, and all scales at zero. A full sweep updates components 0 through 4 sequentially using the newest available states. Stop at the first completed sweep for which the maximum absolute change in any unconditional component-mean entry is at most 1e-10 AND each scale equals its value at the beginning of that sweep. The cap is 400 completed sweeps. These scheduling and stopping choices are part of the deterministic benchmark.

Use per-component 90% credible sets, selecting the shortest descending-probability prefix whose mass reaches 0.90; break probability ties by lower SNP index. A singleton has purity one; other sets use the minimum absolute pairwise LD. Retain a set only if its component has positive scale and its purity is at least 0.95. For identical memberships retain the earliest otherwise-retainable component. Evaluate the source's component average local false sign rate for trait 2, using its definition of conditional strict positive/negative sign events, including any exact point masses.

The follow-up policy admits components whose set is retained and whose source-defined average local false sign rate in trait 2 is strictly below 0.00305. Candidate SNPs are the union of those memberships. Choose the candidate with the largest priority, breaking priority ties by lower SNP index. Report that SNP's fitted cross-trait PIP considering all components with nonzero effect variance, including components whose sets were discarded. If there is no candidate, report zero. This follow-up policy is a stated experimental rule, not a claim that the source prescribes these thresholds or proves causality.

Input arrays (row-major numerical literals)

```text
weak_z = [
  [0.23369346, -0.17508269, 0.06051106, -0.17571872],
  [0.5574564, 0.216615, -0.24548148, -1.25876482],
  [0.29207995, -0.07604982, 0.13597037, 0.02179003],
  [0.31454556, 0.24806547, 1.2027268, 0.25558816],
  [1.234773, -0.72577051, 0.61452995, 1.4043428],
  [0.89320975, 0.62461436, 0.39441824, -0.29833234],
  [-0.67850287, -0.7811954, 0.20298489, -0.40410839],
  [0.47105548, -0.35859318, 0.70417471, 0.61777345],
  [0.31650999, 0.10839269, 1.11845043, 0.78117726],
  [0.08601763, 0.19120808, 2.40700805, -0.33585941],
  [1.0696446, -1.19357739, -0.63545274, 0.34783781],
  [1.45572038, 0.58859816, 0.34534166, 0.15991665],
  [0.30241791, 0.248883, 0.25201724, -0.19422137],
  [-0.44702599, -0.46474153, 1.01260085, -0.3895243],
  [-0.15975772, 0.1443687, -0.17328929, -1.59613842],
  [-0.71721432, -0.75458848, 0.49131386, -0.0814158],
  [-0.24525783, -0.95268689, 0.37606064, 0.7097958],
  [0.53551596, -0.18438758, -0.6098825, 0.10954793],
  [-0.32238921, -0.56706352, 1.57465401, -0.06945547],
  [0.97139008, 0.39623585, 1.35388691, 0.69873086],
  [0.73958338, -0.55207714, 0.17845305, 0.61330469],
  [1.68517507, 0.09831533, -0.66618546, 0.23241195],
  [0.19891176, -1.29053898, 0.74465425, 1.16487517],
  [0.52031932, -0.85516461, -0.29560088, -0.368894],
  [-0.22576051, -1.09686947, 0.90247834, 1.06838975],
  [0.50561193, -1.37682951, 0.75609522, 1.3048096],
  [0.52635556, -0.34564791, -0.07097181, -0.73103425],
  [-0.11131806, -0.947197, 1.32356564, 0.99906139],
  [2.0, 0.17, -0.92, 1.27],
  [-0.38, 2.0, 0.21, -0.58],
  [2.0000001, -0.36, 0.47, 0.22],
  [0.41, -2.13, -0.86, 0.7]
]
```

```text
training_z = [
  [7.67461249, 4.50370261, -3.33374916, 2.40864863],
  [-0.91201661, -1.39220683, -1.78414752, -1.0237524],
  [-5.07287546, 7.34435508, -1.80228958, -3.82025334],
  [3.0228341, 1.61667546, -1.06216727, 0.75524767],
  [0.37587909, -0.69807186, -6.01167057, -5.6340842],
  [0.37121449, -3.54766673, 0.57004122, 0.93108219],
  [1.84531289, 0.10051162, -2.36609214, -0.88434184],
  [0.14292546, -0.02874323, -1.82589409, -0.9961944],
  [-0.60806141, 1.64157308, 0.98707004, 0.10554153],
  [2.28789836, 1.51053694, -0.48648416, 0.89669287],
  [-0.55937565, 0.5521669, -0.88121476, -1.7997869],
  [-0.95755953, 4.29617859, -1.01913501, -2.46273758],
  [-2.17345991, -1.71648581, 1.59372576, 0.81841489],
  [-0.48380852, 1.49787207, 8.06982785, 3.62422554],
  [-2.96015023, -1.1914807, 0.66165324, -0.56402064],
  [-13.3607319, -10.63554522, 7.26736079, -1.15895513],
  [0.90994597, 0.18908875, -4.21755112, -1.82832497],
  [-8.92667943, 12.47838203, 0.03974972, -3.9671226],
  [-4.9979793, -2.98671412, 3.48140437, -0.03289327],
  [-0.79127896, 0.49932967, 4.24053583, 1.56782391],
  [0.9371515, 1.63589498, 0.74797691, -0.2721742],
  [-4.79978519, -2.71287958, 1.97553532, 0.30052748],
  [-0.42010035, 1.02819801, 4.93030649, 1.52992072],
  [-2.11568392, 3.67147555, -2.00599132, -1.72878754]
]
```

```text
w0 = [0.19, 0.29, 0.25, 0.27]
```

```text
U0 = [
  [
    [3.0, 0.0, 0.0, 0.0],
    [0.0, 3.0, 0.0, 0.0],
    [0.0, 0.0, 3.0, 0.0],
    [0.0, 0.0, 0.0, 3.0]
  ],
  [
    [9.0, 6.3, -3.6, 0.9],
    [6.3, 4.409999999999999, -2.5199999999999996, 0.6299999999999999],
    [-3.6, -2.5199999999999996, 1.4400000000000004, -0.3600000000000001],
    [0.9, 0.6299999999999999, -0.3600000000000001, 0.09000000000000002]
  ],
  [
    [0.09000000000000002, 0.09000000000000002, 0.9, 0.675],
    [0.09000000000000002, 0.09000000000000002, 0.9, 0.675],
    [0.9, 0.9, 9.0, 6.75],
    [0.675, 0.675, 6.75, 5.0625]
  ],
  [
    [4.409999999999999, -6.3, 0.6299999999999999, 1.89],
    [-6.3, 9.0, -0.9, -2.6999999999999997],
    [0.6299999999999999, -0.9, 0.09000000000000002, 0.27],
    [1.89, -2.6999999999999997, 0.27, 0.8099999999999999]
  ]
]
```

```text
LD = [
  [1.0, 0.9624016853, 0.982994134, -0.9367008002, -0.1549603195, -0.1343416388, -0.0856391198, 0.0215722064, -0.0532113133, -0.1297015262, -0.0929629668, 0.0012328472],
  [0.9624016853, 1.0, 0.9665989536, -0.9237008624, 0.0063547851, 0.0157556146, 0.0617283831, -0.1356476005, -0.0494374166, -0.1420311988, -0.0857520843, 0.0030843899],
  [0.982994134, 0.9665989536, 1.0, -0.9367395405, -0.1291866096, -0.1078089725, -0.0509489781, -0.0090535248, -0.0771618757, -0.1527682709, -0.1197227049, 0.0396415371],
  [-0.9367008002, -0.9237008624, -0.9367395405, 1.0, 0.0548391595, 0.0845219751, 0.0179279726, 0.0080277382, -0.0828610775, 0.000682965, -0.0267282793, 0.1025455211],
  [-0.1549603195, 0.0063547851, -0.1291866096, 0.0548391595, 1.0, 0.9620689156, 0.9690855895, -0.9400966729, 0.1283796117, 0.0788675909, 0.1579587958, -0.0643645113],
  [-0.1343416388, 0.0157556146, -0.1078089725, 0.0845219751, 0.9620689156, 1.0, 0.9784304063, -0.9715354636, -0.0509565442, -0.0938981856, -0.0123310634, 0.1056408715],
  [-0.0856391198, 0.0617283831, -0.0509489781, 0.0179279726, 0.9690855895, 0.9784304063, 1.0, -0.9671188217, 0.0149765021, -0.0291586015, 0.0468405, 0.0524225559],
  [0.0215722064, -0.1356476005, -0.0090535248, 0.0080277382, -0.9400966729, -0.9715354636, -0.9671188217, 1.0, 0.098462341, 0.1519624181, 0.0626836825, -0.1440434006],
  [-0.0532113133, -0.0494374166, -0.0771618757, -0.0828610775, 0.1283796117, -0.0509565442, 0.0149765021, 0.098462341, 1.0, 0.9823595636, 0.9821528195, -0.9735318416],
  [-0.1297015262, -0.1420311988, -0.1527682709, 0.000682965, 0.0788675909, -0.0938981856, -0.0291586015, 0.1519624181, 0.9823595636, 1.0, 0.9802654295, -0.963857128],
  [-0.0929629668, -0.0857520843, -0.1197227049, -0.0267282793, 0.1579587958, -0.0123310634, 0.0468405, 0.0626836825, 0.9821528195, 0.9802654295, 1.0, -0.9733576402],
  [0.0012328472, 0.0030843899, 0.0396415371, 0.1025455211, -0.0643645113, 0.1056408715, 0.0524225559, -0.1440434006, -0.9735318416, -0.963857128, -0.9733576402, 1.0]
]
```

```text
Sxy = [
  [63.99713982, 60.61974615, -23.81527662, -0.37684009],
  [66.34052179, 69.39992997, -15.48671729, 5.09042181],
  [63.0656513, 63.10850911, -21.90542803, 1.02722627],
  [-63.34281803, -53.27649494, 10.04210753, -10.75814384],
  [2.30897011, 0.90152724, 57.47734466, 40.09619819],
  [-4.31239715, 10.09198489, 52.69777165, 35.24530897],
  [2.26658642, 6.77309635, 53.80317386, 38.32902079],
  [-2.41148369, -22.58857363, -48.06588866, -32.70099071],
  [29.41978111, -74.24660579, 28.26568588, 31.38603285],
  [23.31188248, -84.63016257, 26.91923598, 31.65744875],
  [26.47365183, -78.84141882, 29.96852426, 33.12368745],
  [-32.9164472, 71.61355047, -23.50159663, -30.31338756]
]
```

```text
pi = [0.04381618148539182, 0.11532366726936899, 0.09359136698972634, 0.07190017342217968, 0.10020933604174, 0.11305396105761956, 0.07286298065943514, 0.08249620312581035, 0.05625957076462742, 0.11301507715716524, 0.07721552184433922, 0.06025596018259638]
```

```text
scale_grid = [0.0, 0.00012, 0.00025, 0.0005, 0.0009, 0.0015, 0.0024, 0.0038, 0.006, 0.0095, 0.015, 0.024, 0.038, 0.06, 0.095]
```

```text
priority = [73.0, 61.0, 89.0, 97.0, 92.0, 54.0, 87.0, 76.0, 91.0, 66.0, 95.0, 83.0]
```

## Problem

A follow-up experiment must choose one SNP from a joint fine-mapping analysis of four correlated quantitative traits, using the recent source method that combines multiple multivariate single-variant effects with a learned cross-trait covariance mixture. Identify that source and determine the cross-trait posterior probability of a nonzero effect for the SNP selected by the supplied follow-up policy. Infer the residual correlation and learned prior from the two supplied training panels before fitting the target region, using the complete numerical specification below. The target is the fitted-model probability for the policy-selected SNP, with an exact-null component interpreted as zero effect.

In <reasoning>, identify the source method and DOI and explain its weak-score residual estimator, noisy covariance learning, exact sufficient-statistics equivalence and component-level trait-sign summary. Report the weak-row count and residual correlation entry V[0,3], the first-update weight of prior pattern 0, the final prior weights and trace of regularized prior pattern 0. Give the terminal scales, first terminal sweep and the preceding/terminal maximum component-mean changes. State each positive-scale component's credible-set membership, achieved coverage, minimum purity and average local false sign rate in trait 2, then identify the eligible components and selected SNP. Justify its PIP using its terminal component selection probabilities; keep the explanation to the few values needed for these decisions.

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

01_estimate_residual_correlation

Goal
----
Obtain the source method's weak-association residual correlation estimate from aligned, marginally standardized Z scores, with the supplied weak-association cutoff.

```python
import numpy as np


def estimate_residual_correlation(weak_z, cutoff) -> np.ndarray:
    """Obtain the source method's weak-association residual correlation estimate from aligned, marginally standardized Z scores, with the supplied weak-association cutoff.

    Return R x R positive-definite correlation matrix."""
    # Placeholder only; implement the operation described above.
    return np.zeros((np.shape(weak_z)[1], np.shape(weak_z)[1]), dtype=float)
```

### Step 2

02_update_covariance_mixture

Goal
----
Carry out one simultaneous expectation/maximization update of the zero-centered prior mixture under the source's noisy lead-score training model.

```python
import numpy as np


def update_covariance_mixture(training_z, residual_covariance, prior) -> np.ndarray:
    """Carry out one simultaneous expectation/maximization update of the zero-centered prior mixture under the source's noisy lead-score training model.

    Return Updated K x (1+R*R) packed prior."""
    # Placeholder only; implement the operation described above.
    return np.zeros_like(prior, dtype=float)
```

### Step 3

03_learn_covariance_prior

Goal
----
Fit the source's zero-centered noisy training mixture for exactly iterations updates, then apply the supplied terminal covariance ridge.

```python
import numpy as np


def learn_covariance_prior(
    training_z, residual_covariance, initial_prior, iterations, ridge,
) -> np.ndarray:
    """Fit the source's zero-centered noisy training mixture for exactly iterations updates, then apply the supplied terminal covariance ridge.

    Return Final K x (1+R*R) packed prior."""
    # Placeholder only; implement the operation described above.
    return np.zeros_like(initial_prior, dtype=float)
```

### Step 4

04_fit_single_effect

Goal
----
Determine the fixed-scale single-effect posterior, retaining SNP, covariance-pattern, and trait-sign uncertainty.

```python
import numpy as np


def fit_single_effect(
    information, residual_crossproduct, residual_covariance, prior, variant_prior,
    scale,
) -> np.ndarray:
    """Determine the fixed-scale single-effect posterior, retaining SNP, covariance-pattern, and trait-sign uncertainty.

    Return Packed (J+1) x (2R+3) posterior state."""
    # Placeholder only; implement the operation described above.
    return np.zeros(
        (np.size(information) + 1, 2 * np.shape(residual_crossproduct)[1] + 3),
        dtype=float,
    )
```

### Step 5

05_select_effect_scale

Goal
----
Select the component scale by the source's marginal-evidence objective over a supplied finite grid and return its posterior.

```python
import numpy as np


def select_effect_scale(
    information, residual_crossproduct, residual_covariance, prior, variant_prior,
    scale_grid, initial_candidate=None,
) -> np.ndarray:
    """Select the component scale by the source's marginal-evidence objective over a supplied finite grid and return its posterior.

    Return Selected packed (J+1) x (2R+3) posterior state."""
    # Placeholder only; implement the operation described above.
    return np.zeros(
        (np.size(information) + 1, 2 * np.shape(residual_crossproduct)[1] + 3),
        dtype=float,
    )
```

### Step 6

06_fit_additive_effects

Goal
----
Fit the source model with effects latent single-effect components using the supplied exact sufficient statistics and fixed learned prior.

```python
import numpy as np


def fit_additive_effects(
    genotype_crossproduct, genotype_trait_crossproduct, residual_covariance, prior,
    variant_prior, scale_grid, effects, tolerance, max_sweeps,
    first_component_state=None,
) -> np.ndarray:
    """Fit the source model with effects latent single-effect components using the supplied exact sufficient statistics and fixed learned prior.

    Return L x (J+1) x (2R+3) terminal state."""
    # Placeholder only; implement the operation described above.
    return np.zeros(
        (int(effects), np.shape(genotype_trait_crossproduct)[0] + 1,
         2 * np.shape(genotype_trait_crossproduct)[1] + 3),
        dtype=float,
    )
```

### Step 7

07_resolve_credible_sets

Goal
----
Determine source-style per-component credible sets and retain those meeting the supplied scale, purity, and deduplication policy.

```python
import numpy as np


def resolve_credible_sets(state, ld, coverage, min_purity) -> np.ndarray:
    """Determine source-style per-component credible sets and retain those meeting the supplied scale, purity, and deduplication policy.

    Return L x (J+3) numeric credible-set diagnostics."""
    # Placeholder only; implement the operation described above.
    return np.zeros((np.shape(state)[0], np.shape(ld)[0] + 3), dtype=float)
```

### Step 8

08_average_trait_sign_uncertainty

Goal
----
Compute the source-defined component average local false sign rate for each trait.

```python
import numpy as np


def average_trait_sign_uncertainty(state) -> np.ndarray:
    """Compute the source-defined component average local false sign rate for each trait.

    Return L x R component average local false sign rates."""
    # Placeholder only; implement the operation described above.
    return np.zeros(
        (np.shape(state)[0], (np.shape(state)[2] - 3) // 2),
        dtype=float,
    )
```

### Step 9

09_prioritize_variant

Goal
----
Select a follow-up SNP under the declared credible-set and trait-sign policy and report its overall probability of a nonzero effect in any trait

```python
import numpy as np


def prioritize_variant(
    state, credible_sets, average_lfsr, trait_index, sign_threshold, priority,
) -> np.ndarray:
    """Select a follow-up SNP under the declared credible-set and trait-sign policy and report its overall probability of a nonzero effect in any trait.

    Return Length-2 numeric vector [selected zero-based SNP index, cross-trait PIP]."""
    # Placeholder only; implement the operation described above.
    return np.zeros(2, dtype=float)
```

### Step 10

10_compute_prioritized_pip

Goal
----
Orchestrate the complete source-grounded training, regional fit, uncertainty summary, and follow-up policy; return the one requested probability.

```python
import numpy as np


def compute_prioritized_pip(
    weak_z, training_z, initial_prior, genotype_crossproduct,
    genotype_trait_crossproduct, variant_prior, scale_grid, priority, effects,
    ed_iterations, ridge, weak_cutoff, coverage, min_purity, trait_index,
    sign_threshold, tolerance, max_sweeps,
) -> float:
    """Orchestrate the complete source-grounded training, regional fit, uncertainty summary, and follow-up policy; return the one requested probability.

    Return One finite float in [0,1]."""
    # Placeholder only; implement the operation described above.
    return 0.0
```
