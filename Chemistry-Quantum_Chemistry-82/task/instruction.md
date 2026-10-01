# Chemistry-Quantum_Chemistry-82

## Background

A molecular first hyperpolarizability is a rank-three response tensor. Its contraction with an incident-field direction produces a vector whose length and orientation vary across the unit sphere. Comparing an approximate basis representation with a numerically converged reference therefore requires global, signed, and Cartesian-resolved diagnostics. The supplied fixture is a compact numerical benchmark: its tensor generator, equal-weight spherical design, six basis labels, deterministic mixture initialization, regularization, and final scalar reduction are disclosed conventions rather than claims about production electronic-structure software. The paper-dependent decisions are deliberately left to be recovered from the source and then applied to those fixed arrays.

## Problem

Directional errors in molecular first hyperpolarizabilities can remain substantial even when a scalar tensor error looks small. Compute a deterministic second-harmonic-generation convergence audit for the synthetic ensemble below, recovering the source's frequency prescription, effective-response representation, reference normalization, signed projection regions, component analysis, and Gaussian-mixture/BIC interpretation. This fixture replaces the source's real molecules and Lebedev quadrature with reproducible tensors and equal-weight directions; its six indexed perturbation families are synthetic, not six measured basis sets. Use dimensionless error fractions throughout, not percentages. Report the frequency vector and R, P, N, A, C, G, H, B, D in the reasoning, explain the normalization, sign regions and component anisotropy, and identify the source's most consistently converged basis family, the reported effect of double augmentation, and the role of core-polarization functions for second-row systems. The final output is J rounded to six decimal places.

## Fixed numerical fixture

All indices start at zero and all calculations use float64. Set seed = 82026, M = 68 molecules, n = 86 directions, omega_max = 0.36 atomic units, and maximum mixture count = 4. Recover the five input frequencies from the source.

For j = 0,...,n-1, let phi = (1+sqrt(5))/2, z_j = 1-2(j+0.5)/n, rho_j = sqrt(1-z_j^2), and a_j = 2*pi*j/phi. Normalize each vector (rho_j*cos(a_j), rho_j*sin(a_j), z_j) to unit length and use w_j = 4*pi/n. With numpy.random.default_rng(seed), first draw the complete reference array from Normal(0,0.42), then the complete mode array from Normal(0,1), both with shape (M,3,3,3) in row-major order. Separately symmetrize the final two axes of each array by averaging with its transpose. For molecule m, add 1.05+0.06m, -0.72+0.025m, and 0.48+0.035(m mod 5) to reference entries (0,0,0), (1,1,1), and (2,2,2), respectively; divide each mode tensor by its Frobenius norm.

Set b_bias = (0.24,0.075,0.12,0.026,0.052,0.006), b_anis = (0.19,0.105,0.12,0.065,0.07,0.028), g_anis = (0.38,0.42,0.72,2.25), and g = m mod 4. The six-by-four sign matrix is

S = ((1,-0.85,1,0.15), (0.75,-0.65,-0.45,0.05), (0.85,-0.72,0.8,0.10), (0.55,-0.52,-0.35,0.02), (0.68,-0.60,0.55,0.08), (0.34,-0.28,-0.22,0.01)).

At frequency index f, set x_f = omega_f/omega_max, t_f = 1+0.24*x_f+0.11*x_f^2, beta_ref(m,f) = t_f*reference_m+0.035*f*mode_m, and

beta_basis(m,b,f) = [1+b_bias_b*S_(b,g)*(1+0.09*sin((m+1)*(f+1)))]*beta_ref(m,f) + b_anis_b*g_anis_g*(0.58+0.07*f)*mode_m.

Apply the source's directional error definitions to each tensor pair. Use its common reference scale in total and Cartesian component errors. Benchmark anisotropy is the largest normalized Cartesian RMS error divided by the smallest. Use the signed projection only to partition directions. Within each nonempty sign region, compute the quadrature-weighted RMS norm of the full directional error vector and divide by the common spherical mean reference magnitude from Eq. 29. A sign region with zero quadrature weight contributes zero RMS error; directions with exactly zero signed projection belong to neither region.

## Mixture convention

For each molecule, form twelve features by alternating the frequency-mean positive-region and negative-region normalized directional-error RMS values for each family. Fit full-covariance Gaussian mixtures to these unstandardized features for q = 1,...,4 with covariance regularizer 1e-6. Initialize the first mean with the row farthest in squared Euclidean distance from the feature mean, then choose each additional row by maximizing its minimum squared distance to previously chosen rows, excluding selected rows. Break ties at the lowest row index. Initial mixture weights are 1/q; every initial covariance is the population feature covariance plus 1e-6 times the identity.

At each iteration, evaluate Gaussian log densities and log-sum-exp responsibilities, retain that log likelihood and those responsibilities, then perform the M-step. Add 1e-15 to each responsibility column sum before dividing to obtain mixture weights, means, and weighted covariances; add the regularizer to each updated covariance. Stop after this M-step when the absolute change from the previous log likelihood is at most 1e-10*(1+absolute previous log likelihood), or after 250 iterations. Use the retained likelihood and responsibilities from the last E-step for BIC and entropy, without an additional E-step. Select the lowest BIC, breaking a tie at the smallest q. Use BIC = -2*log likelihood + p*log(M), where p = (q-1)+12*q+78*q. H is mean posterior entropy using natural logarithms and responsibilities clipped to [1e-300,1]; B is the second-smallest BIC minus the smallest, and G is the selected q. Relabel components by lexicographically ascending mean vectors only when returning labels.

## Diagnostics and scalar reduction

R, P, N, and A are unweighted means over all molecules, families, and frequencies of relative total error, positive projection RMS, negative projection RMS, and anisotropy, respectively. C is relative total error at molecule floor(M/2), family 3, frequency index 2. If F is the M-by-12 feature matrix, K is the sum of (12*m+c+1)*F_(m,c) over all m and c. D is mean relative total error at the last frequency minus its mean at frequency zero. Compute

J = R+0.35*(P+N)+0.025*A+0.02*H+0.0005*B+0.1*absolute(D)+0.005*G.

Also report the selected mixture's free-parameter count p in the reasoning.

Output Format Requirements: Emit <final_answer> immediately, then <reasoning>. Do not write a long derivation before the tags. You must emit exactly one finite decimal inside <final_answer>...</final_answer>, even if the value is approximate or you are unsure. Rules: - The tags are required. Do not omit them or leave them empty. - The tagged value must be a finite decimal (examples: 0.4847, 0.22, 1.05). Not NaN, not Inf, not a fraction string, not a vector, not prose. - Put only that one number between the tags. No units, no words, no extra lines. Keep <reasoning> concise (a few hundred words). Report every scalar and explanation explicitly requested above, while omitting unrequested bulk data. Do not paste the input matrices, full coefficient vectors, per-iteration paths, or per-fold candidate tables.

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

build_shg_frequency_grid

Goal
----
Use the source's second-harmonic-generation frequency prescription. For a positive first-excitation estimate omega_max, evaluate the five integer indices from zero through four at the source denominator eight. Return the numerical frequency vector.

```python
def build_shg_frequency_grid(omega_max: float = 0.36) -> np.ndarray:
    """Return the five source-prescribed SHG input frequencies.

    Parameters
    ----------
    omega_max : float
        Positive finite first-excitation estimate.
    Returns
    -------
    np.ndarray
        Five frequencies in ascending order.
    Raises
    ------
    ValueError
        If omega_max is not positive and finite.
    """
    return None
```

### Step 2

compute_effective_hyperpolarizability

Goal
----
Apply the source's unit-sphere representation. For each incident unit direction u, contract the final two tensor axes with u twice so the kth output component is the sum of beta[k,i,j] u[i] u[j]. Preserve the input Cartesian frame.

```python
def compute_effective_hyperpolarizability(beta: np.ndarray, directions: np.ndarray) -> np.ndarray:
    """Map a rank-three hyperpolarizability tensor onto unit directions.

    Parameters
    ----------
    beta : np.ndarray
        Finite tensor with shape (3,3,3).
    directions : np.ndarray
        At least four Cartesian unit directions with shape (n,3).
    Returns
    -------
    np.ndarray
        Effective response vectors with shape (n,3).
    Raises
    ------
    ValueError
        If a shape, finiteness, count, or unit-norm contract fails.
    """
    return None
```

### Step 3

compute_relative_rms_total

Goal
----
Use the source's relative RMS total error. Form the effective basis and reference vector fields, take the 4*pi-normalized spherical RMS norm of their difference, and divide by the spherical mean magnitude of the reference vector field from source equation 29. Report dimensionless fractions, without percentage scaling. If the row-major directional error array is e[n,k], return its checksum sum over n and k of (3n+k+1)e[n,k] together with the ratio and reference scale.

```python
def compute_relative_rms_total(beta_basis: np.ndarray, beta_mra: np.ndarray, directions: np.ndarray, weights: np.ndarray) -> tuple:
    """Compute the source-normalized spherical RMS total error.

    Parameters
    ----------
    beta_basis, beta_mra : np.ndarray
        Approximate and reference tensors, each with shape (3,3,3).
    directions : np.ndarray
        Unit directions with shape (n,3).
    weights : np.ndarray
        Positive quadrature weights of length n that sum to 4*pi.
    Returns
    -------
    tuple
        Relative RMS error, quadrature-weighted spherical mean magnitude of the MRA effective-response vectors, and directional-error checksum.
    Raises
    ------
    ValueError
        If quadrature or tensor contracts fail or the reference norm vanishes.
    """
    return None
```

### Step 4

compute_signed_projection_metrics

Goal
----
Apply the source's signed projection diagnostic. Project the basis effective vector onto the reference effective vector, normalize by the squared reference norm, subtract one, and partition directions by sign. Within each nonempty sign region, compute the quadrature-weighted RMS norm of the full directional error vector and divide by the common spherical mean magnitude of the reference effective-response vectors from source equation 29. If s[n] is the signed projection error, also return the checksum sum_n (n+1)s[n] and each sign region's quadrature weight divided by 4*pi; an empty sign region contributes zero RMS error.

```python
def compute_signed_projection_metrics(beta_basis: np.ndarray, beta_mra: np.ndarray, directions: np.ndarray, weights: np.ndarray) -> tuple:
    """Compute sign-partitioned directional-error metrics on the unit sphere.

    Parameters
    ----------
    beta_basis, beta_mra : np.ndarray
        Approximate and reference tensors, each with shape (3,3,3).
    directions : np.ndarray
        Unit directions with shape (n,3).
    weights : np.ndarray
        Positive quadrature weights of length n that sum to 4*pi.
    Returns
    -------
    tuple
        Positive-region and negative-region normalized vector-error RMS values,
        signed checksum, positive weight fraction, and negative weight fraction.
    Raises
    ------
    ValueError
        If quadrature or tensor contracts fail or a sampled reference response vanishes.
    """
    return None
```

### Step 5

compute_component_rms_anisotropy

Goal
----
Use the source's component-wise RMS analysis. Compute one spherical RMS error c[k] for each Cartesian output component, normalize each by the common spherical mean MRA vector magnitude from source equation 29, then report the three-vector, its maximum-to-minimum ratio, and the checksum c[0]+2c[1]+3c[2]. Report dimensionless fractions, without percentage scaling.

```python
def compute_component_rms_anisotropy(beta_basis: np.ndarray, beta_mra: np.ndarray, directions: np.ndarray, weights: np.ndarray) -> tuple:
    """Compute Cartesian RMS errors and their anisotropy.

    Parameters
    ----------
    beta_basis, beta_mra : np.ndarray
        Approximate and reference tensors, each with shape (3,3,3).
    directions : np.ndarray
        Unit directions with shape (n,3).
    weights : np.ndarray
        Positive quadrature weights of length n that sum to 4*pi.
    Returns
    -------
    tuple
        Three normalized component RMS errors, max-to-min ratio, and component checksum.
    Raises
    ------
    ValueError
        If quadrature or tensor contracts fail or a required norm vanishes.
    """
    return None
```

### Step 6

select_convergence_clusters

Goal
----
Follow the source's Gaussian-mixture clustering with BIC model selection. For each candidate count q from one through max_clusters, fit a full-covariance mixture by standard EM and use BIC=-2 log L+p log n with p=(q-1)+qd+qd(d+1)/2. The benchmark starts with the point farthest from the grand mean and repeatedly adds the point maximizing its minimum squared distance to selected means; all components start with the biased empirical covariance plus regularization times the identity and equal weights. Stop after 250 iterations or when the log-likelihood change is at most 1e-10 times one plus the previous magnitude, choose the smallest-BIC candidate, and relabel components by lexicographically sorted means. Return the selected count, numeric labels, BIC vector, mean posterior entropy, and the second-smallest minus smallest BIC.



Retain the likelihood and responsibilities from the last E-step, before its M-step, for BIC and entropy; do not run an additional E-step. Add 1e-15 to the component responsibility sums in each M-step. Farthest-point ties use the lowest row index, and BIC ties use the smallest component count.

```python
def select_convergence_clusters(features: np.ndarray, max_clusters: int = 4, regularization: float = 1e-6) -> tuple:
    """Select a full-covariance Gaussian mixture by BIC.

    Parameters
    ----------
    features : np.ndarray
        Finite nonconstant feature matrix with at least eight rows and two columns.
    max_clusters : int
        Largest candidate count, from two through min(4,n-1).
    regularization : float
        Positive covariance diagonal regularizer.
    Returns
    -------
    tuple
        Selected count, numeric labels, BIC vector, mean posterior entropy, and BIC margin.
    Raises
    ------
    ValueError
        If shape, variability, candidate-count, or regularization contracts fail.
    """
    return None
```

### Step 7

compute_directional_basis_audit

Goal
----
Construct the fixture with default_rng(seed). For n=direction_count and j=0,...,n-1, set phi=(1+sqrt(5))/2, z=1-2(j+0.5)/n, rho=sqrt(1-z^2), azimuth=2pij/phi, normalize [rho cos(azimuth),rho sin(azimuth),z], and assign weight 4pi/n. Draw reference then mode arrays of shape (molecule_count,3,3,3) from normal distributions with scales 0.42 and 1.0, symmetrize their final two axes, add 1.05+0.06m, -0.72+0.025m, and 0.48+0.035(m mod 5) to reference entries [0,0,0], [1,1,1], and [2,2,2], and Frobenius-normalize each mode. Use base_bias=[0.24,0.075,0.12,0.026,0.052,0.006], anis_scale=[0.19,0.105,0.12,0.065,0.07,0.028], group_anisotropy=[0.38,0.42,0.72,2.25], groups m mod 4, and the row-major group_sign matrix [[1,-0.85,1,0.15],[0.75,-0.65,-0.45,0.05],[0.85,-0.72,0.8,0.10],[0.55,-0.52,-0.35,0.02],[0.68,-0.60,0.55,0.08],[0.34,-0.28,-0.22,0.01]]. At frequency index f, let x = omega[f]/omega_max, scale = 1 + 0.24 × x + 0.11 × x^2, beta_mra = scale × reference + 0.035 × f × mode, bias = base_bias[b] × group_sign[b,group], modulation = 1 + 0.09 × sin((m+1)(f+1)), and beta_basis = (1 + bias × modulation) × beta_mra + anis_scale[b] × group_anisotropy[group] × (0.58 + 0.07 × f) × mode. Call Steps 1 through 5 for every molecule, family, and frequency; set the twelve features to frequency means of alternating positive-region and negative-region normalized vector-error RMS values; and call Step 6. Define R, P, N, and A as full-cube means, C as relative[molecule_count//2,3,2], K as the row-major weighted feature checksum, D as the last-frequency mean relative error minus the static mean, and G, H, B from Step 6. Return J and all diagnostics, where J=R+0.35(P+N)+0.025A+0.02H+0.0005B+0.1|D|+0.005G.

```python
def compute_directional_basis_audit(seed: int = 82026, molecule_count: int = 68, direction_count: int = 86, omega_max: float = 0.36, max_clusters: int = 4) -> tuple:
    """Run the complete deterministic directional basis audit.

    Parameters
    ----------
    seed : int
        Seed for the disclosed tensor fixture.
    molecule_count : int
        Number of synthetic molecules, at least eight.
    direction_count : int
        Number of equal-weight spherical directions, at least 24.
    omega_max : float
        Positive first-excitation estimate.
    max_clusters : int
        Largest mixture count accepted by Step 6.
    Returns
    -------
    tuple
        J, ten scalar diagnostics, frequency vector, error cube, feature matrix, labels, and BIC vector.
    Raises
    ------
    ValueError
        If a top-level input violates its documented contract.
    """
    return None
```
