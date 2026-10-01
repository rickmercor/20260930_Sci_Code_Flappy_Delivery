# Physics-Computational_Physics-17

## Problem

Consider a free open chain of 12 monomers in three dimensions, indexed 1 through 12, whose active conformational dynamics is described at the Gaussian second-moment level with reciprocal, conformation-preaveraged Rotne-Prager-Yamakawa mobility. In dimensionless units, the self-mobility is 1, the monomer radius is a=0.5, neighboring monomers are joined by Hookean springs of stiffness 0.5, and every unordered pair, including neighbors, contributes the contact energy \((2\pi a^2/3)^{-3/2}\exp[-3|\mathbf r_i-\mathbf r_j|^2/(2a^2)]\).

The independent, zero-mean Gaussian force noises have covariance \(\langle\eta_i^\alpha(t)\eta_j^\beta(t')\rangle=C_{ij}\delta_{\alpha\beta}\delta(t-t')/3\), where C is diagonal with C_33=4 and all other diagonal entries 1; there is no confinement or other interaction. An independent zero-mean Gaussian spatial velocity field has covariance \(\langle\xi^\alpha(\mathbf r,t)\xi^\beta(\mathbf r',t')\rangle=(4/3)\delta_{\alpha\beta}\exp[-|\mathbf r-\mathbf r'|^2/(2(1.5)^2)]\delta(t-t')\). Monomer j also experiences \(\boldsymbol\chi_j=-\sum_{i\ne j}B_{ji}\nabla_{\mathbf r_j}[\exp(-|\mathbf r_j-\mathbf r_i|/\ell)/|\mathbf r_j-\mathbf r_i|]\), with ell=2, B_9,3=-3, B_3,9=1 and every other B entry zero, and the screened response includes the finite-monomer-size regularization. At time zero the eleven bond vectors are independent, isotropic, zero-mean Gaussian vectors of mean squared length 1, the center of mass is zero in each initial realization, and all the specified interactions and noises act for t>=0.

For this transient Gaussian description, determine \(Q=\langle[\mathbf r_3(2)-\mathbf r_2(2)]\cdot[\mathbf r_9(1)-\mathbf r_8(1)]\rangle-\langle[\mathbf r_9(2)-\mathbf r_8(2)]\cdot[\mathbf r_3(1)-\mathbf r_2(1)]\rangle\), where brackets denote the ensemble cross-moments of that description and the initial squared bond length sets the unit. Give Q with absolute error at most 0.001.

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

01_contact_response

Goal
----
Return the Gaussian-averaged contact force response matrix for a normalized isotropic three-dimensional Gaussian pair potential with squared width radius**2/3 and amplitude strength. The input is a symmetric nonnegative squared-separation matrix with zero diagonal; radius is positive and strength nonnegative. Return minus one third of the Gaussian-averaged trace of the pair Hessian, so a repulsive kernel gives negative off-diagonal entries. Diagonal response ensures zero row sums, with self interactions excluded. Raise ValueError for invalid inputs.

```python
def contact_response(squared_separations, radius, strength):
    """Return the Gaussian-averaged contact force response matrix for a normalized isotropic three-dimensional Gaussian pair potential with squared width radius**2/3 and amplitude strength. The input is a symmetric nonnegative squared-separation matrix with zero diagonal; radius is positive and strength nonnegative. Return minus one third of the Gaussian-averaged trace of the pair Hessian, so a repulsive kernel gives negative off-diagonal entries. Diagonal response ensures zero row sums, with self interactions excluded. Raise ValueError for invalid inputs."""
    return
```

### Step 2

02_averaged_mobility

Goal
----
Return the isotropic conformation average of reciprocal Rotne-Prager-Yamakawa mobility in three dimensions, normalized to the specified self_mobility. Inputs are a symmetric nonnegative squared-separation matrix with zero diagonal and positive finite radius and self_mobility. Include overlapping-bead separations and the zero-separation limit. Raise ValueError for invalid inputs.

```python
def averaged_mobility(squared_separations, radius, self_mobility):
    """Return the isotropic conformation average of reciprocal Rotne-Prager-Yamakawa mobility in three dimensions, normalized to the specified self_mobility. Inputs are a symmetric nonnegative squared-separation matrix with zero diagonal and positive finite radius and self_mobility. Include overlapping-bead separations and the zero-separation limit. Raise ValueError for invalid inputs."""
    return
```

### Step 3

03_screened_response

Goal
----
Return the isotropic Gaussian-averaged force response of nonreciprocal screened Coulomb interactions: force on j is minus the sum of B[j,i] times the gradient with respect to r_j of exp(-distance/ell)/distance. The squared separation in the averaged response is regularized by adding radius**2. couplings is a finite square matrix with zero diagonal; its first index is the responding bead. Exclude self forces and impose zero row sums. radius and screening_length are positive. Raise ValueError for invalid inputs.

```python
def screened_response(squared_separations, radius, screening_length, couplings):
    """Return the isotropic Gaussian-averaged force response of nonreciprocal screened Coulomb interactions: force on j is minus the sum of B[j,i] times the gradient with respect to r_j of exp(-distance/ell)/distance. The squared separation in the averaged response is regularized by adding radius**2. couplings is a finite square matrix with zero diagonal; its first index is the responding bead. Exclude self forces and impose zero row sums. radius and screening_length are positive. Raise ValueError for invalid inputs."""
    return
```

### Step 4

04_averaged_spatial_covariance

Goal
----
Return the conformation-averaged dot-product covariance of an independent spatial Gaussian velocity field. Its Cartesian covariance at positions separated by r is amplitude/3 times the identity times exp(-r*r/(2*length**2)), with unit temporal delta correlation. The isotropic Gaussian displacement has mean squared length given by squared_separations. This is a symmetric finite nonnegative square matrix with zero diagonal; amplitude is finite nonnegative and length finite positive. Include self correlations. Raise ValueError for invalid inputs.

```python
def averaged_spatial_covariance(squared_separations, amplitude, length):
    """Return the spatial velocity covariance averaged over Gaussian conformations; the matrix contains dot-product, not single-component, covariances. Raise ValueError for invalid inputs."""
    return
```

### Step 5

05_drift_and_noise

Goal
----
Return a numeric array of shape (2,n,n) containing the preaveraged drift and the monomer force-noise contribution to the dot-product covariance injection for the isotropic Gaussian polymer. mobility is symmetric; spring, contact, chemical are force-response matrices; force_covariance uses the convention that Cartesian noise covariance is force_covariance/3. Every input must be a same-size finite square matrix. Raise ValueError for invalid inputs.

```python
def drift_and_noise(mobility, spring, contact, chemical, force_covariance):
    """Return a numeric array of shape (2,n,n) containing the preaveraged drift and the monomer force-noise contribution to the dot-product covariance injection for the isotropic Gaussian polymer. mobility is symmetric; spring, contact, chemical are force-response matrices; force_covariance uses the convention that Cartesian noise covariance is force_covariance/3. Every input must be a same-size finite square matrix. Raise ValueError for invalid inputs."""
    return
```

### Step 6

06_evolve_covariance

Goal
----
Return the equal-time dot-product covariance at end_time, starting at time zero. coefficient_function(X) returns a (2,n,n) array containing the current preaveraged drift and covariance injection for a supplied covariance X. Use the continuous Gaussian covariance dynamics with relative numerical accuracy 1e-9 or better and absolute accuracy 1e-11 or better. The initial covariance is finite symmetric positive semidefinite; end_time is finite and nonnegative. Raise ValueError for invalid inputs.

```python
def evolve_covariance(initial_covariance, end_time, coefficient_function):
    """Return the equal-time dot-product covariance at end_time, starting at time zero. coefficient_function(X) returns a (2,n,n) array containing the current preaveraged drift and covariance injection for a supplied covariance X. Use the continuous Gaussian covariance dynamics with relative numerical accuracy 1e-9 or better and absolute accuracy 1e-11 or better. The initial covariance is finite symmetric positive semidefinite; end_time is finite and nonnegative. Raise ValueError for invalid inputs."""
    return
```

### Step 7

07_evolve_cross_covariance

Goal
----
Return the unequal-time dot-product cross-covariance with the earlier time held fixed. earlier_covariance is the equal-time covariance at that earlier time. coefficient_function(X) returns the current (drift, covariance injection) as a numeric (2,n,n) array. The same-time covariance continues evolving during lag, and the unequal-time covariance follows the causal Gaussian dynamics. Return only the final unequal-time matrix, with relative accuracy 1e-9 or better and absolute accuracy 1e-11 or better. lag is finite and nonnegative and earlier_covariance is finite symmetric positive semidefinite. Raise ValueError for invalid inputs.

```python
def evolve_cross_covariance(earlier_covariance, lag, coefficient_function):
    """Return the unequal-time dot-product cross-covariance with the earlier time held fixed. earlier_covariance is the equal-time covariance at that earlier time. coefficient_function(X) returns the current (drift, covariance injection) as a numeric (2,n,n) array. The same-time covariance continues evolving during lag, and the unequal-time covariance follows the causal Gaussian dynamics. Return only the final unequal-time matrix, with relative accuracy 1e-9 or better and absolute accuracy 1e-11 or better. lag is finite and nonnegative and earlier_covariance is finite symmetric positive semidefinite. Raise ValueError for invalid inputs."""
    return
```

### Step 8

08_correlation_asymmetry

Goal
----
Final orchestrator: compute Q for the twelve-bead task configuration. activity replaces C[3,3] in one-based notation; attraction replaces B[9,3]; all other parameters are those of the task. Initial independent bonds have unit mean squared length, radius 0.5, spring stiffness 0.5, contact strength 1, self-mobility 1, screening length 2, B[3,9]=1, earlier time 1 and later time 2. An independent spatial Gaussian velocity field has amplitude 4 and correlation length 1.5; its Cartesian covariance is amplitude/3 times exp(-r*r/(2*length**2)) times the identity and temporal delta function. activity is finite nonnegative and attraction finite. Raise ValueError for invalid inputs.

```python
def correlation_asymmetry(activity=4.0, attraction=-3.0):
    """Final orchestrator: compute Q for the twelve-bead task configuration. activity replaces C[3,3] in one-based notation; attraction replaces B[9,3]; all other parameters are those of the task. Initial independent bonds have unit mean squared length, radius 0.5, spring stiffness 0.5, contact strength 1, self-mobility 1, screening length 2, B[3,9]=1, earlier time 1 and later time 2. An independent spatial Gaussian velocity field has amplitude 4 and correlation length 1.5; its Cartesian covariance is amplitude/3 times exp(-r*r/(2*length**2)) times the identity and temporal delta function. activity is finite nonnegative and attraction finite. Raise ValueError for invalid inputs."""
    return
```
