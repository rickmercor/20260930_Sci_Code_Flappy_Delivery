# Biology-Genetics-14

## Background

Cis-expression models predict a gene's regulated expression from nearby variants. Reference panels contain relatively few expression-profiled individuals compared with the number of correlated candidate variants, so local effects require strong regularization without discarding diffuse genetic signal.

Hierarchical Bayesian mixed models address this tension by combining structured sparsity over linkage-disequilibrium neighborhoods with a dense polygenic component. Genomic annotations can further alter prior support for individual variants within regions that remain eligible for nonzero effects.

Posterior inference yields SNP-level weight distributions, regional inclusion summaries, and dense background estimates. Held-out expression prediction and downstream transcriptome-wide association analyses evaluate whether these weights capture transferable cis-regulatory signal.

## Problem

Genotype-based cis-expression prediction must separate a small number of local regulatory effects from a dense genetic background while accounting for linkage disequilibrium. Consider a hierarchical Bayesian mixed model in which an LD-block gate controls annotation-informed SNP spike-and-slab effects and the dense component is represented in a fixed genetic-relationship-matrix eigenbasis.

Evaluate a reduced fixed-hyperparameter posterior trace for one synthetic gene. The block decision must use the method's approximate collapsed evidence, active blocks must use its sequential SNP scan, and the annotation coefficients must be refreshed from the active-block subset while the dense effect is updated in the rotated basis.

Your task is to solve one concrete deterministic example of this pipeline. Use the following configuration:

- `X_rot = [[1.3416407864998738, 1.3416407864998738, 1.3416407864998738, 1.3416407864998738, 1.3416407864998738, 1.3416407864998738], [1.0488088481701516, -1.0488088481701516, 1.0488088481701516, -1.0488088481701516, 1.0488088481701516, -1.0488088481701516], [1.02469507659596, 1.02469507659596, -1.02469507659596, -1.02469507659596, 0.0, 0.0], [0.6123724356957945, -0.6123724356957945, -0.6123724356957945, 0.6123724356957945, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]`
- `y_rot = [1.3, 0.4, -0.9, 1.0, -0.5]`
- `eigenvalues = [1.8, 1.1, 0.7, 0.25, 0.0]`
- `positions_bp = [999000, 1000000, 1050000, 1200000, 1700000, 2200000]`
- `tss_bp = 1000000`
- `block_indices = [[0, 1, 2], [3, 4, 5]]`
- `X_test = [[0.6, 0.55, -0.2, 0.9, -0.4, 0.82], [-0.3, -0.28, 1.1, -0.5, 0.7, -0.47]]`
- `n_sweeps = 3`
- `sigma2 = 0.55`, `eta_beta = 0.85`, `eta_g = 0.65`, and `pi_block = 0.38`, all fixed across sweeps
- `alpha_init = -1.55` and `kappa_init = -0.9`
- `alpha_prior_mean = -2.9444389791664403`, `kappa_prior_mean = 0.0`, `alpha_prior_var = 4.0`, and `kappa_prior_var = 4.0`
- `alpha_step = 0.3` and `kappa_step = 0.5`
- `u_block = [[0.15, 0.72], [0.58, 0.22], [0.3088, 0.63]]`
- `u_snp = [[0.12, 0.62, 0.18, 0.77, 0.18, 0.54], [0.43, 0.08, 0.71, 0.29, 0.66, 0.15], [0.21, 0.47, 0.05, 0.81, 0.34, 0.58]]`
- `z_beta = [[0.35, -0.8, 1.1, -0.25, 0.6, -1.2], [-0.4, 0.95, -0.55, 0.7, -1.05, 0.2], [1.25, -0.3, 0.45, -0.9, 0.15, 0.85]]`
- `z_g = [[0.2, -0.7, 1.1, 0.3, -0.4], [-0.6, 0.5, -0.2, 1.2, 0.1], [0.8, -0.1, 0.4, -0.5, 1.0]]`
- `mh_proposal_normals = [[[0.4, -0.6], [-1.1, 0.3], [0.7, -0.2], [0.2, 0.9]], [[-0.5, 0.8], [0.9, -1.0], [-0.3, 0.4], [1.2, -0.7]], [[0.6, 0.5], [-0.8, -0.4], [0.1, 1.1], [-1.0, 0.2]]]`
- `mh_uniforms = [[0.99, 0.95, 0.4, 0.97], [0.95, 0.2, 0.99, 0.7], [0.85, 0.95, 0.8, 0.99]]`
- Initialize `beta`, `gamma`, all block indicators, and the rotated dense effect to zero. Consume each random-variate table by sweep; SNP entries retain their original indices, and the four annotation proposals in each row are considered in listed order.

Carry the state through all three sweeps without variance-scale or block-prior updates, average the three post-sweep sparse-effect vectors, and use only those posterior mean cis weights for held-out prediction. Your final answer must be a single number: the predicted expression for the first held-out individual.

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

Implement **all 7 functions** below into a single file at `/app/solution.py`. Requirements:

- Use the EXACT function name and signature shown for each step.
- Define every function at module scope (importable).
- Do NOT include tests, example usage, prints, or a `__main__` block.
- Available libraries: numpy, scipy, sympy, pandas, scikit-learn, networkx.

Start your file with these imports:

```python
import numpy as np
```

### Step 1

01_compute_annotation_probabilities

Goal
----
Construct the transcription-start-site-informed SNP inclusion prior.

Regulatory variants nearer a gene's transcription start site can receive larger

prior support for a nonzero cis effect. For SNP j, the annotation is the absolute

distance a_j = |position_j - tss| / 10^6 in megabases, the prior log-odds are

ell_j = alpha + kappa * a_j, and the inclusion probability is

pi_j = 1 / (1 + exp(-ell_j)). Alpha controls the baseline log-odds at the TSS,

while kappa controls how those odds vary with distance. The base-pair to

megabase conversion is part of the model convention, and the logistic transform

is evaluated stably so extreme finite coefficients approach zero or one without

overflow.

Inputs

------

positions_bp : one-dimensional SNP positions in base pairs

tss_bp : transcription start site position in base pairs

alpha : annotation-prior intercept

kappa : annotation-prior distance coefficient

Returns

-------

distances_mb : absolute SNP-to-TSS distances in megabases

inclusion_probabilities : annotation-informed probabilities pi_j

```python
import numpy as np


def compute_annotation_probabilities(
    positions_bp: np.ndarray,
    tss_bp: float,
    alpha: float,
    kappa: float,
) -> np.ndarray:
    """Compute TSS distances and annotation-informed inclusion probabilities.

    Parameters
    ----------
    positions_bp : np.ndarray
        One-dimensional SNP positions in base pairs.
    tss_bp : float
        Transcription start site position in base pairs.
    alpha : float
        Logistic-prior intercept.
    kappa : float
        Logistic-prior TSS-distance coefficient.

    Raises
    ------
    ValueError
        If positions are empty or not a finite one-dimensional array, or if a
        scalar parameter is not finite.

    Returns
    -------
    result : np.ndarray
        Float64 array of shape (2, p). Row 0 is `distances_mb`; row 1 is
        `inclusion_probabilities`.
    """
    return result  # noqa: F821
```

### Step 2

02_compute_collapsed_block_probability

Goal
----
Evaluate the approximate collapsed posterior gate for one LD block.

The block indicator zeta_b first decides whether every sparse effect in block b

is zero or whether its SNP-level spike-and-slab model is active. Given the common

block-removed residual r_-b, integrating a Gaussian slab with variance

sigma2 * eta_beta gives v_j = [x_j^T x_j / sigma2 +

1 / (sigma2 * eta_beta)]^-1, mu_j = v_j x_j^T r_-b / sigma2, and

logBF_j = 0.5 log(v_j) - 0.5 log(sigma2 * eta_beta) + mu_j^2 / (2 v_j).

For this coarse block comparison, the within-block Gram matrix is diagonalized,

so Delta_b = sum_j log[(1 - pi_j) + pi_j exp(logBF_j)] is formed with the same

r_-b for every SNP. The posterior block probability is

sigmoid(logit(pi_block) + Delta_b). This diagonal approximation belongs only to

the block gate; an active block is subsequently handled by an LD-aware

sequential scan.

Inputs

------

X_rot : rotated genotype matrix with one column per SNP

block_indices : ordered SNP indices belonging to block b

residual_without_block : common residual r_-b used by every block SNP

sigma2, eta_beta : residual variance and slab-to-residual variance ratio

inclusion_probabilities : SNP prior probabilities pi_j

pi_block : prior probability that block b is active

Returns

-------

log_evidence_increment : approximate collapsed increment Delta_b

block_probability : posterior probability that zeta_b equals one

log_bayes_factors : per-SNP integrated log Bayes factors

```python
import numpy as np


def compute_collapsed_block_probability(
    X_rot: np.ndarray,
    block_indices: np.ndarray,
    residual_without_block: np.ndarray,
    sigma2: float,
    eta_beta: float,
    inclusion_probabilities: np.ndarray,
    pi_block: float,
) -> np.ndarray:
    """Compute approximate collapsed evidence and the block-on probability.

    Parameters
    ----------
    X_rot : np.ndarray
        Rotated genotype matrix with shape (n, p).
    block_indices : np.ndarray
        Unique SNP indices in one LD block.
    residual_without_block : np.ndarray
        Residual after restoring the current block contribution.
    sigma2 : float
        Positive residual variance.
    eta_beta : float
        Positive slab-to-residual variance ratio.
    inclusion_probabilities : np.ndarray
        SNP prior probabilities with shape (p,), all strictly between zero and one.
    pi_block : float
        Block prior probability strictly between zero and one.

    Raises
    ------
    ValueError
        If dimensions, indices, variances, or probabilities are invalid.

    Returns
    -------
    result : np.ndarray
        Float64 vector whose first two entries are the evidence increment and
        block-on probability, followed by the per-SNP log Bayes factors.
    """
    return result  # noqa: F821
```

### Step 3

03_scan_active_block

Goal
----
Sample SNP effects sequentially within an active LD block.

Conditional on zeta_b = 1, the SNP indicators and effects are updated by a

coordinate-wise Gibbs scan in block order. The running residual is

r = y_rot - X_rot beta - g_rot. Before visiting SNP j, its old contribution

x_j beta_j is restored to r; the conditional slab variance v_j, mean mu_j, and

log Bayes factor are then computed from that restored residual. The posterior

inclusion log-odds equal logit(pi_j) + logBF_j. If the supplied uniform passes

this gate, beta_j = mu_j + sqrt(v_j) z_j and the new contribution is immediately

subtracted from r; otherwise beta_j and gamma_j are set to zero. Because later

SNPs see the effects sampled earlier in the same pass, this update preserves the

local dependence induced by linkage disequilibrium and is not equivalent to a

simultaneous diagonal scan.

Inputs

------

X_rot : rotated genotype matrix

block_indices : ordered indices of the active block

residual, beta, gamma : current residual and sparse state

sigma2, eta_beta : residual variance and slab variance ratio

inclusion_probabilities : annotation-informed probabilities pi_j

u_gate, z_normal : fixed uniform and standard-normal variates in block order

Returns

-------

residual : residual after the complete block scan

beta, gamma : updated sparse effects and inclusion indicators

scan_probabilities : conditional inclusion probabilities in visitation order

```python
import numpy as np


def scan_active_block(
    X_rot: np.ndarray,
    block_indices: np.ndarray,
    residual: np.ndarray,
    beta: np.ndarray,
    gamma: np.ndarray,
    sigma2: float,
    eta_beta: float,
    inclusion_probabilities: np.ndarray,
    u_gate: np.ndarray,
    z_normal: np.ndarray,
) -> np.ndarray:
    """Perform one sequential spike-and-slab scan of an active LD block.

    Parameters
    ----------
    X_rot : np.ndarray
        Rotated genotype matrix with shape (n, p).
    block_indices : np.ndarray
        Ordered unique SNP indices for the active block.
    residual : np.ndarray
        Current residual y_rot - X_rot @ beta - g_rot.
    beta : np.ndarray
        Current sparse-effect vector with shape (p,).
    gamma : np.ndarray
        Current binary inclusion vector with shape (p,).
    sigma2 : float
        Positive residual variance.
    eta_beta : float
        Positive slab-to-residual variance ratio.
    inclusion_probabilities : np.ndarray
        Per-SNP prior probabilities with shape (p,).
    u_gate : np.ndarray
        Uniform variates in block order.
    z_normal : np.ndarray
        Standard-normal variates in block order.

    Raises
    ------
    ValueError
        If dimensions, parameters, or supplied variates are invalid, or if
        beta[j] is nonzero for any index at which gamma[j] is false.

    Returns
    -------
    result : np.ndarray
        Float64 vector containing `residual` (first n entries), `beta` (next p),
        numeric `gamma` indicators (next p), and `scan_probabilities` (remaining
        entries).
    """
    return result  # noqa: F821
```

### Step 4

04_sample_polygenic_background

Goal
----
Sample the dense polygenic background in the GRM eigenbasis.

The mixed model separates sparse cis effects from a dense component

g ~ N(0, sigma2 * eta_g * K), where K is the genetic relationship matrix.

Rotating into the eigenbasis of K diagonalizes the Gaussian conditional. For

eigenvalue lambda_i and cis residual e_i = y_rot_i - (X_rot beta)_i, define

s_i = eta_g * lambda_i / (1 + eta_g * lambda_i). The conditional mean is

s_i e_i, the conditional variance is sigma2 * s_i, and a supplied standard

normal variate gives g_rot_i = s_i e_i + sqrt(sigma2 * s_i) z_i. A zero

eigenvalue has zero shrinkage and variance, so its dense-effect component is

deterministically zero regardless of z_i.

Inputs

------

y_rot, X_rot : expression and genotype data in the GRM eigenbasis

beta : current sparse cis-effect vector

eigenvalues : nonnegative eigenvalues lambda_i of the GRM

sigma2, eta_g : residual variance and polygenic-to-residual variance ratio

z_normal : fixed standard-normal variates for the rotated components

Returns

-------

g_rot : sampled dense effect in the eigenbasis

shrinkage : componentwise coefficients s_i

conditional_variance : componentwise Gaussian variances sigma2 * s_i

```python
import numpy as np


def sample_polygenic_background(
    y_rot: np.ndarray,
    X_rot: np.ndarray,
    beta: np.ndarray,
    eigenvalues: np.ndarray,
    sigma2: float,
    eta_g: float,
    z_normal: np.ndarray,
) -> np.ndarray:
    """Sample the dense polygenic effect componentwise in the rotated basis.

    Parameters
    ----------
    y_rot : np.ndarray
        Rotated expression vector with shape (n,).
    X_rot : np.ndarray
        Rotated genotype matrix with shape (n, p).
    beta : np.ndarray
        Sparse-effect vector with shape (p,).
    eigenvalues : np.ndarray
        Nonnegative GRM eigenvalues with shape (n,).
    sigma2 : float
        Positive residual variance.
    eta_g : float
        Positive polygenic-to-residual variance ratio.
    z_normal : np.ndarray
        Standard-normal variates with shape (n,).

    Raises
    ------
    ValueError
        If dimensions, eigenvalues, variances, or supplied variates are invalid.

    Returns
    -------
    result : np.ndarray
        Float64 array of shape (3, n). Rows contain the dense sample, shrinkage
        coefficients, and conditional variances, respectively.
    """
    return result  # noqa: F821
```

### Step 5

05_update_annotation_prior

Goal
----
Update the coefficients of the TSS-informed SNP inclusion prior.

For SNP j, ell_j = alpha + kappa * a_j and pi_j = sigmoid(ell_j), where a_j is

its TSS distance in megabases. Conditional on the current block state, only SNPs

inside active blocks contribute Bernoulli observations to the annotation

likelihood; zeros forced by inactive blocks are structural and are not evidence

about alpha or kappa. The log target is the active-set sum

gamma_j ell_j - log(1 + exp(ell_j)) plus independent Gaussian log priors for

alpha and kappa. Each proposal adds alpha_step * z_alpha and

kappa_step * z_kappa to the latest accepted state, and it is accepted when the

log of its supplied uniform is below the change in log posterior. Updating the

current state after every acceptance is essential because the proposal row is a

Metropolis path, not a collection of comparisons with the sweep's initial

coefficients.

Inputs

------

alpha, kappa : current annotation-prior coefficients

distances_mb, gamma : SNP annotations and inclusion indicators

zeta, block_indices : block activity and the SNP partition

proposal_normals, proposal_uniforms : fixed random-walk and acceptance variates

alpha_prior_mean, kappa_prior_mean : Gaussian prior centers

alpha_prior_var, kappa_prior_var : Gaussian prior variances

alpha_step, kappa_step : random-walk scales

Returns

-------

alpha, kappa : coefficients after the ordered proposal sequence

accepted : Boolean acceptance indicators in proposal order

log_posterior : log target at the final coefficient state

```python
import numpy as np


def update_annotation_prior(
    alpha: float,
    kappa: float,
    distances_mb: np.ndarray,
    gamma: np.ndarray,
    zeta: np.ndarray,
    block_indices: list,
    proposal_normals: np.ndarray,
    proposal_uniforms: np.ndarray,
    alpha_prior_mean: float,
    kappa_prior_mean: float,
    alpha_prior_var: float,
    kappa_prior_var: float,
    alpha_step: float,
    kappa_step: float,
) -> np.ndarray:
    """Apply ordered random-walk Metropolis updates to annotation coefficients.

    Parameters
    ----------
    alpha, kappa : float
        Current logistic-prior coefficients.
    distances_mb : np.ndarray
        Nonnegative TSS distances with shape (p,).
    gamma : np.ndarray
        Binary SNP inclusion vector with shape (p,).
    zeta : np.ndarray
        Binary block activity vector.
    block_indices : list
        Ordered list of disjoint integer SNP-index arrays.
    proposal_normals : np.ndarray
        Standard-normal proposal increments with shape (q, 2).
    proposal_uniforms : np.ndarray
        Uniform acceptance variates with shape (q,).
    alpha_prior_mean, kappa_prior_mean : float
        Gaussian prior means.
    alpha_prior_var, kappa_prior_var : float
        Positive Gaussian prior variances.
    alpha_step, kappa_step : float
        Positive proposal scales.

    Raises
    ------
    ValueError
        If state dimensions, block membership, priors, or variates are invalid.

    Returns
    -------
    result : np.ndarray
        Float64 vector containing updated alpha, updated kappa, q numeric
        acceptance indicators, and the final log posterior, in that order.
    """
    return result  # noqa: F821
```

### Step 6

06_summarize_sparse_prediction

Goal
----
Convert retained sparse-effect draws into transferable cis predictions.

If beta^(m) is the sparse SNP-effect vector retained after sweep m, its posterior

mean is beta_mean = M^-1 sum_m beta^(m). Held-out genotype row X_test[i] is

scored as X_test[i]^T beta_mean, or X_test beta_mean for the complete panel. The dense polygenic term

used while fitting the reference samples is not added to this prediction: it is

sample specific and has no held-out value without a separate train-test kernel

construction. The resulting sparse-cis weights are therefore the transferable

quantity used for individual-level prediction and downstream TWAS-style linear

scores.

Inputs

------

beta_samples : retained sparse-effect matrix with one sweep per row

X_test : held-out genotype matrix with matching SNP columns

Returns

-------

beta_mean : componentwise posterior mean SNP weights

predictions : sparse-cis predictions for all held-out rows

first_prediction : prediction for the first held-out individual

```python
import numpy as np


def summarize_sparse_prediction(
    beta_samples: np.ndarray,
    X_test: np.ndarray,
) -> np.ndarray:
    """Average sparse effects and form held-out sparse-cis predictions.

    Parameters
    ----------
    beta_samples : np.ndarray
        Retained sparse-effect samples with shape (m, p).
    X_test : np.ndarray
        Held-out genotype matrix with shape (n_test, p).

    Raises
    ------
    ValueError
        If either input is empty, nonfinite, not two dimensional, or has an
        incompatible SNP dimension.

    Returns
    -------
    result : np.ndarray
        Float64 vector containing the p posterior-mean weights, all n_test
        predictions, and the first prediction as its final entry.
    """
    return result  # noqa: F821
```

### Step 7

07_run_full_pipeline

Goal
----
Assemble the fixed-hyperparameter block-sparse mixed-model trace.

The rotated model is y_rot = X_rot beta + g_rot + epsilon. Sparse cis effects

beta are governed hierarchically by block indicators zeta_b and

annotation-informed SNP indicators gamma_j, while g_rot represents the dense polygenic

background in the eigenbasis of the genetic relationship matrix. Each sweep

first evaluates TSS-based SNP priors, then visits every LD block using a

collapsed block gate and, when active, a sequential within-block SNP scan. The

dense Gaussian effect is sampled after the sparse pass, followed by the ordered

Metropolis update of the annotation coefficients. The supplied uniform and

normal tables replace stochastic draws and make the trace deterministic. The

variance ratios and block prior remain fixed, post-sweep beta vectors are

retained, and their componentwise mean defines the sparse-cis predictor

X_test beta_mean.

Inputs

------

X_rot, y_rot, eigenvalues : rotated training data and GRM spectrum

positions_bp, tss_bp, block_indices : genomic annotations and LD partition

X_test, n_sweeps : held-out genotypes and trace length

u_block, u_snp, z_beta, z_g : fixed block, SNP, slab, and dense variates

mh_proposal_normals, mh_uniforms : fixed annotation-update variates

sigma2, eta_beta, eta_g, pi_block : fixed model hyperparameters

alpha_init, kappa_init : initial annotation coefficients

alpha_prior_mean, kappa_prior_mean : annotation prior centers

alpha_prior_var, kappa_prior_var : annotation prior variances

alpha_step, kappa_step : annotation random-walk scales

Returns

-------

first_prediction : sparse-cis predicted expression for the first held-out row

```python
import numpy as np
def run_full_pipeline(
    X_rot: np.ndarray,
    y_rot: np.ndarray,
    eigenvalues: np.ndarray,
    positions_bp: np.ndarray,
    tss_bp: float,
    block_indices: list,
    X_test: np.ndarray,
    n_sweeps: int,
    u_block: np.ndarray,
    u_snp: np.ndarray,
    z_beta: np.ndarray,
    z_g: np.ndarray,
    mh_proposal_normals: np.ndarray,
    mh_uniforms: np.ndarray,
    sigma2: float,
    eta_beta: float,
    eta_g: float,
    pi_block: float,
    alpha_init: float,
    kappa_init: float,
    alpha_prior_mean: float,
    kappa_prior_mean: float,
    alpha_prior_var: float,
    kappa_prior_var: float,
    alpha_step: float,
    kappa_step: float,
) -> float:
    """Run the fixed-hyperparameter trace and return the first test prediction.

    Parameters
    ----------
    X_rot, y_rot, eigenvalues : np.ndarray
        Rotated training quantities and GRM eigenvalues.
    positions_bp : np.ndarray
        SNP genomic positions in base pairs.
    tss_bp : float
        Transcription start site position in base pairs.
    block_indices : list
        Ordered disjoint arrays that partition the SNP indices.
    X_test : np.ndarray
        Held-out genotype matrix.
    n_sweeps : int
        Positive number of deterministic sweeps.
    u_block, u_snp, z_beta, z_g : np.ndarray
        Supplied block, SNP, slab, and dense-effect variates by sweep.
    mh_proposal_normals, mh_uniforms : np.ndarray
        Supplied annotation proposal and acceptance variates by sweep.
    sigma2, eta_beta, eta_g : float
        Fixed positive residual, slab-ratio, and polygenic-ratio parameters.
    pi_block : float
        Fixed block inclusion prior probability.
    alpha_init, kappa_init : float
        Initial annotation coefficients.
    alpha_prior_mean, kappa_prior_mean : float
        Annotation prior means.
    alpha_prior_var, kappa_prior_var : float
        Positive annotation prior variances.
    alpha_step, kappa_step : float
        Positive annotation proposal scales.

    Raises
    ------
    ValueError
        If the sweep count or random-variate table dimensions are invalid.

    Returns
    -------
    first_prediction : float
        Sparse-cis predicted expression for the first held-out individual.
    """
    return first_prediction  # noqa: F821
```
