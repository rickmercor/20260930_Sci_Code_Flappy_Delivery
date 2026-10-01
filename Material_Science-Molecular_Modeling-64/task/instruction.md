# Material_Science-Molecular_Modeling-64

## Background

Let lambda=[0,0.25,0.5,0.75,1] and x=[0.15,0.30,0.45,0.60,0.75,0.90,1.00]. The synchronized energy-gap blocks have shape (C,B,L) and are constructed exactly (ordinary float64 arithmetic) as

`gap_blocks = gap_means[:,None,:] + scales[:,None,None]*(H @ R.T)[None,:,:]`,

where

```python
gap_means = np.array([
  [-0.046395623,-0.059551873,-0.067520623,-0.070301873,-0.067895623],
  [-0.119592843,-0.133405343,-0.141842843,-0.144905343,-0.142592843],
  [-0.204877463,-0.219346213,-0.228252463,-0.231596213,-0.229377463],
  [-0.299644351,-0.314769351,-0.324144351,-0.327769351,-0.325644351],
  [-0.408949543,-0.424730793,-0.434574543,-0.438480793,-0.436449543],
  [-0.530757393,-0.547194893,-0.557507393,-0.561694893,-0.559757393],
  [-0.617605610,-0.634480610,-0.645105610,-0.649480610,-0.647605610]])
scales = 0.0015*(1+0.08*np.arange(7))
H = np.array([[1,1,1,1,1],[-1,1,-1,1,-1],[1,-1,-1,1,1],
              [-1,-1,1,1,-1],[1,1,1,-1,-1],[-1,1,-1,-1,1],
              [1,-1,-1,-1,-1],[-1,-1,1,-1,1]], dtype=float)
R = np.array([[1,0,0,0,0],[0.35,0.93675,0,0,0],
              [0.15,0.25,0.95656,0,0],[0.08,0.12,0.30,0.94234,0],
              [0.05,0.10,0.16,0.28,0.94048]])
```

Each entry is an energy-gap observation in eV/atom; its orientation and the physical meaning of the x=0.60, lambda=1 observation follow from the source. Rows are synchronized across lambda.

T=520 K, k_B=8.617333262145e-5 eV/K, m_A=22.98976928 u, and m_B=6.94 u. The solution volumes are [39.00,36.70,34.35,32.40,30.45,28.75,27.80] A^3/atom. At P*=0 GPa, V_A=V0=41.552 A^3/atom, B0=5.143 GPa, and B0'=3.943; use 64-point Gauss-Legendre quadrature and 0.006241509074460763 eV/(GPa A^3).

The finite-size calibration is x_fs=[0.12,0.28,0.46,0.64,0.82,1.00], r_fs=[0.00185,0.00062,-0.00048,-0.00131,-0.00037,0.00091] eV/atom, sigma_grid=[0.0008,0.0012,0.0016,0.0020,0.0025,0.0032] eV/atom, length_grid=[0.06,0.12,0.20,0.32,0.50,0.80], and noise_sigma=0.00035 eV/atom. Profile the zero-mean exponential-kernel likelihood, breaking exact ties by amplitude then length. Its pre-endmember covariance and the synchronized sampling covariance undergo the source-consistent response transformation.

The statistical configuration uses correlated quadratic TI, the paper-consistent analytic reference and pressure terms, a shared-endmember response for both sampling and finite-size covariance, and Redlich-Kister coefficient counts p in {2,3,4}. Rank supported models by `AICc=chi2+2*p+2*p*(p+1)/(n-p-1)` and retain physically feasible candidates within 4 of the best feasible value.

A physically feasible candidate has exactly one lower curvature zero on 257 inclusive nodes over [0.05,0.49] and retains an enclosing common tangent for the nominal coefficients and the ordered unit-Cholesky perturbations `c,c+L[:,0],c-L[:,0],...`. Tangents use 9 starts per side on [0.01,0.49] and [0.51,0.99], tolerance 1e-10, residual at most 1e-9 eV/atom, and lexicographic selection by `(alpha,-beta,residual,slope)`.

For each supported candidate, the one-sided profile certificate is the unique crossing below x_sp of `q(x)-1.645*sqrt(b2(x).T@C@b2(x))`, bracketed on 513 inclusive nodes and bisected to width 1e-12. The active-set stress certificate refits every estimated quantity, first selecting the synchronized-row removal with the smallest profile certificate and then the smallest feasible conditional second-removal certificate; preserve original one-based labels and break exact ties by the smaller label.

## Problem

Determine the conservative Na-rich liquid spinodal certificate at 520 K for the synthetic Li-Na-like observations below. Use the cited source to resolve two consequential forks: whether the auxiliary substituted species carries the reference or solute mass, and whether zero target pressure removes the volume leg or retains pure-reference EOS work; apply the source-consistent branch rather than merely naming it.

Infer the alchemical coupling convention used for forces and Monte Carlo acceptance from that same source. Compute the uncertainty-qualified active-set result under the numerical configuration in the scientific background.

In <reasoning>, first give one compact sentence stating the source-resolved auxiliary-species construction, propagated-force convention, Monte Carlo acceptance energy, zero-pressure EOS treatment, and how correlated quadratic ATI combines with the analytic mass and pressure contributions. Then give a compact numerical audit containing the complete-data profile bound, the two original synchronized-block labels controlling the stress certificate, the bound after the first removal, the selected nominal root and nonlinear profile half-width, and the unrounded selected certificate. Do not provide a candidate table or narrate the implementation sequence.

Use float64 arithmetic. Report diagnostic compositions to at least five decimal places and round only the tagged result to four decimals.
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
Output Format Requirements:
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

gap_block_statistics

Goal
----
Estimate synchronized lambda-window means and their covariance of the mean.

```python
import numpy as np

def gap_block_statistics(block_gaps):
    """Estimate the lambda-window mean gap and its correlated uncertainty.

    Parameters
    ----------
    block_gaps : array-like, shape (B,L)
        B synchronized block estimates of U1-U0 in eV/atom at L ordered
        lambda windows; require L>=3 and B>=L+1.

    Returns
    -------
    numpy.ndarray, shape (L+1,L)
        Row 0 contains the L mean gaps in input-window order. Rows 1..L
        contain the LxL sample covariance matrix of those means, in the same
        row/column order, in (eV/atom)^2.
    """
    return None
```

### Step 2

quadratic_gls_ti

Goal
----
Fit and integrate the quadratic alchemical energy-gap curve with correlated uncertainty.

```python
import numpy as np

def quadratic_gls_ti(lambdas, statistics):
    """Integrate a quadratic alchemical gap fitted with its full covariance.

    Returns a float64 ndarray of length 3 ordered as
    [F_TI_eV_per_atom, standard_error_eV_per_atom,
    quadratic_second_derivative_eV_per_atom].
    """
    return None
```

### Step 3

alchemical_mass_correction

Goal
----
Evaluate the analytic auxiliary-mass and configurational Helmholtz correction.

```python
import numpy as np

def alchemical_mass_correction(x_solute, temperature_k, mass_reference_u, mass_solute_u):
    """Evaluate the auxiliary-system mass and ideal-mixing correction.

    Returns a float64 ndarray of length 3 ordered as
    [F_mass, ideal_configurational_term, kinetic_mass_term], all in eV/atom.
    """
    return None
```

### Step 4

birch_murnaghan_pv

Goal
----
Integrate the pure-reference EOS to obtain the finite-volume Gibbs correction.

```python
import numpy as np

def birch_murnaghan_pv(volume_solution_a3, volume_reference_a3,
                       pressure_target_gpa, bulk_modulus_gpa,
                       bulk_derivative, volume_zero_a3,
                       quadrature_order=64):
    """Evaluate the third-order Birch-Murnaghan EOS correction.

    Returns a float64 ndarray of length 3 ordered as
    [F_PV_eV_per_atom, P_at_solution_volume_GPa,
    P_at_reference_volume_minus_target_GPa].
    """
    return None
```

### Step 5

mixing_curve_covariance

Goal
----
Construct excess-mixing responses and shared-endmember sampling covariance.

```python
import numpy as np

def mixing_curve_covariance(compositions, gibbs_certificates, temperature_k):
    """Build excess-mixing responses with shared-endmember covariance.

    `compositions` is strictly increasing and ends at x=1. Each row of
    `gibbs_certificates` is ordered [Delta_G, standard_error, F_TI, F_mass,
    F_PV]. For C compositions, returns a float64 ndarray of shape
    (C-1,C+1). Column 0 is the increasing interior composition x, column 1
    is the excess response in eV/atom, and columns 2..C are its (C-1)x(C-1)
    covariance matrix in the same x row/column order, in (eV/atom)^2.
    """
    return None
```

### Step 6

augment_finite_size_covariance

Goal
----
Calibrate and add a composition-correlated finite-size uncertainty model.

```python
import numpy as np

def augment_finite_size_covariance(mixing_certificate,
                                   calibration_compositions,
                                   calibration_residuals_ev,
                                   sigma_grid_ev,
                                   length_grid,
                                   noise_sigma_ev):
    """Profile and add finite-size uncertainty correlated across composition.

    `mixing_certificate` has shape (N,N+2), with x in column 0, excess
    response in column 1, and the NxN sampling covariance in columns 2..N+1.
    The paired-cell residuals at `calibration_compositions` are scored for
    every Cartesian-product pair in the strictly increasing positive sigma
    and length grids using log(det(C))+r.T@solve(C,r), where
    C=sigma^2*exp(-|xa-xb|/length)+noise_sigma_ev^2*I. Select the lexicographic
    minimum (objective,sigma,length). Return a float64 ndarray of shape
    (N,N+4): the original x and response, the augmented NxN covariance, then
    the selected sigma and length repeated down the final two columns.
    """
    return None
```

### Step 7

fit_redlich_kister

Goal
----
Fit and physically screen correlated Redlich-Kister model orders.

```python
import numpy as np

def fit_redlich_kister(augmented_certificate, temperature_k,
                       bracket=(0.05, 0.49), grid_points=257):
    """Fit and physically screen a Redlich-Kister order ensemble.

    `augmented_certificate` has shape (N,N+4). Fit coefficient counts 2, 3,
    and 4 by correlated GLS, compute AICc, and mark a candidate feasible only
    when an odd `grid_points` scan finds exactly one strict sign change of the
    full ideal-plus-excess second derivative inside `bracket`; refine that
    root to 1e-12 by bisection. Returns shape (3,24), rows ordered by counts
    [2,3,4], and columns [count,AICc,feasible,root,L0..L3,
    flattened_row_major_4x4_covariance], zero-padding unused coefficients and
    covariance entries. An infeasible root uses the numeric sentinel 0.0 and
    must never enter the supported set.
    """
    return None
```

### Step 8

common_tangent_screen

Goal
----
Solve nominal and covariance-robust liquid-liquid coexistence certificates.

```python
import numpy as np

def common_tangent_screen(rk_ensemble, temperature_k,
                          left_interval=(0.01,0.49),
                          right_interval=(0.51,0.99),
                          starts_per_axis=9, tolerance=1.0e-10,
                          sigma_radius=1.0):
    """Screen nominal and coefficient-uncertain coexistence consistency.

    Input is (3,24) from `fit_redlich_kister`. Solve nominal common tangents
    from the evenly spaced start-grid product. Use the exact 2x2 Newton
    Jacobian and first in-bounds halving among powers 0..13; select by smaller
    alpha, larger beta, residual, then slope. Require residual<=10*tolerance
    and alpha<stored_root<0.5<beta. For C=LL.T, then process c and ordered
    c+sigma_radius*L[:,j], c-sigma_radius*L[:,j]. Recompute each 257-node root
    and tangent by continuation from the preceding accepted solution. Every
    point must retain one enclosed root; the envelope is max(alpha),min(beta).

    Returns float64 shape (3,36): the original 24 columns followed by
    [phase_feasible,x_alpha,x_beta,tangent_slope,max_residual,width,
    robust_feasible,robust_alpha,robust_beta,robust_worst_residual,
    robust_min_root_margin,robust_max_root_shift]. Infeasible robust
    diagnostics use numeric zeros.
    """
    return None
```

### Step 9

spinodal_lower_confidence

Goal
----
Select a nonlinear profile-curvature lower bound.

```python
import numpy as np

def spinodal_lower_confidence(rk_ensemble, temperature_k,
                              z_score=1.645, delta_aicc=4.0):
    """Select a profile bound across covariance-robust feasible RK orders.

    Input is the (3,36) output of `common_tangent_screen`. Retain triple-
    feasible models within `delta_aicc` of the best feasible AICc. Evaluate
    q(x)-z_score*sqrt(b2(x).T@Cov@b2(x)) on 513 inclusive nodes from 0.05 to
    the stored root; require one strict crossing and bisect to 1e-12. Here q
    is full curvature and b2 is the active RK second-derivative basis. Choose
    the smallest crossing, then smaller coefficient count. Return float64(7):
    [profile_lower_bound,nominal_root,profile_half_width,coefficient_count,
    Delta_AICc_from_best,third_derivative_at_root,local_delta_standard_error].
    """
    return None
```

### Step 10

ati_spinodal_pipeline

Goal
----
Compose ATI inference and select a conditional two-row stability bound.

```python
import numpy as np

def ati_spinodal_pipeline(gap_blocks, lambdas, compositions,
                          solution_volumes_a3, temperature_k,
                          mass_reference_u, mass_solute_u,
                          volume_reference_a3, pressure_target_gpa,
                          bulk_modulus_gpa, bulk_derivative,
                          volume_zero_a3, calibration_compositions,
                          calibration_residuals_ev, sigma_grid_ev,
                          length_grid, noise_sigma_ev,
                          monitor_index=3, z_score=1.645,
                          delta_aicc=4.0, coexistence_sigma_radius=1.0):
    """Compose ATI inference and a sequential synchronized-block stress test.

    Refit all single synchronized-row deletions and select the smallest bound,
    then condition on that deletion and refit every remaining second deletion.
    Exclude a second candidate if any stage raises ValueError, including loss
    of positive-definite block covariance. Select the smallest surviving pair
    bound; ties use the smaller original one-based row number at each round.
    Return float64(19): [pair_bound,complete_bound,first_deletion_bound,
    first_row,second_row,valid_second_count,pair_root,pair_profile_half_width,
    pair_coefficient_count,robust_alpha,robust_beta,local_delta_standard_error,
    first_minus_pair,complete_minus_pair,original_block_count]. Every fold
    independently re-estimates all covariances and coefficients. The last four
    entries are [unconditional_runner_up_row,unconditional_runner_up_bound,
    runner_up_conditional_feasible,local_delta_lower_bound].
    """
    return None
```
