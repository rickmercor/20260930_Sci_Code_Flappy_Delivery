# Material_Science-Molecular_Modeling-4

## Background

Molecular clusters and macromolecules separate naturally into stiff intramolecular vibrations and slow collective motion. Coarse-grained models exploit this by replacing the fast modes with an effective free energy on the slow coordinates, which shrinks the space to be sampled and lets simulations reach equilibrium with far fewer steps. The crudest route, rigid-body coarse-graining, freezes each molecular fragment at a fixed shape; it is cheap but visibly distorts heat capacities and pair distributions at low temperature. A recent refinement instead relaxes each fragment along its stiff subspace before adding a harmonic free-energy correction from the local curvatures, recovering near exact thermodynamic consistency with the all-atom reference at comparable cost. The construction rests on a small number of precise linear-algebra conventions, and its accuracy claims depend on following them exactly. This task implements and audits that pipeline for clusters of flexible triatomic model monomers, comparing the relaxed construction against the fixed-shape baseline it was designed to replace.

## Problem

Coarse-grained models replace fast intramolecular motion with an effective potential on the slow degrees of freedom. The subspace harmonic relaxation (SHR) method, which reformulates and accelerates an earlier rigidification approach, constructs this coarse-grained free energy directly from the all-atom potential: each group of atoms is relaxed along the subspace the paper selects by a fixed number of Newton type updates built from a per-block pseudoinverse of the mass-scaled Hessian, and a harmonic free-energy correction is added from the selected mode frequencies. Your job is to implement the SHR pipeline exactly as the paper specifies it and audit it against the conventional fixed-geometry rigidification on the supplied clusters. The load-bearing algorithmic choices, which subspace enters the pseudoinverse, how that pseudoinverse is treated across the update iterations, and where the frequencies entering the free-energy correction are evaluated, are the paper's; recover them from the source. This problem is calibrated so those choices change the reported numbers by amounts far beyond the grading tolerance.

The system is a cluster of I flexible triatomic A-B-A monomers in three dimensions, reduced units, hbar = kB = 1. Atom order per monomer is [A1, B, A2]; coordinates have shape (I, 3, 3) as (monomer, atom, xyz). Masses are mA = 1.0 and mB = 16.0. The all-atom potential is: harmonic bonds B-A1 and B-A2 with kb = 800.0 and r0 = 1.0; a harmonic angle A1-B-A2 with kth = 120.0 and th0 = 1.9106332362490186 radians; and intermonomer Lennard-Jones interactions between every pair of distinct monomers, B with B using epsB = 1.2 and sigB = 2.6, and every A with every A using epsA = 0.05 and sigA = 1.4. There are no A-B intermonomer terms and no interactions within a monomer beyond the bonds and the angle. Each monomer carries Li = 3 selected intramolecular modes, and per-monomer Hessian blocks are the exact analytic second derivatives of the total potential with respect to that monomer's own nine coordinates, symmetrized, then scaled by the inverse square roots of the atomic masses on both sides; the blocks must be exact, and numerical differentiation of the gradient does not meet the grading tolerance.

Implement seven functions. cluster_energy(coords) and cluster_gradient(coords) return the potential and its exact analytic gradient. block_mass_scaled_hessian(coords) returns the (I, 9, 9) mass-scaled blocks defined above. reference_configuration(coords) maps each monomer to the isolated-monomer internal minimum in its own frame; the frame contract is: origin at B, x along the normalized bisector of the B to A1 and B to A2 unit vectors, z along their normalized cross product, y completing the right-handed frame, with A1 placed at angle +th0/2 from x toward y and A2 at -th0/2, both at bond length r0. If the sine of the bond angle, the norm of the cross product of the two bond vectors divided by the product of their lengths, is below 1e-8, the plane normal is instead the Cartesian basis vector with the smallest absolute dot product with x, smallest index on ties, orthogonalized against x and normalized; a zero-length bond vector or a degenerate bisector raises ValueError. shr_relax(coords, num_iters) performs exactly num_iters updates of the paper's iteration with no tolerance exit and returns the final configuration; the update rule, its mass scaling, and the treatment of the pseudoinverse across iterations follow the source. harmonic_thermo(V, omegas, T) returns [classical free energy, quantum free energy, total classical positional mode variance, total quantum positional mode variance] with the exact forms the paper gives for the free energies and for the thermal mode variances of its backmapping construction. shr_audit(configs, num_iters, T), the final step, must be assembled by calling the earlier functions and returns one float64 row per configuration with twenty columns: potential energy at the reference-manifold point, potential energy at the relaxed point, classical coarse-grained free energy, quantum coarse-grained free energy, classical fixed-geometry free energy obtained with no relaxation, the difference of columns two and four (zero indexed: column 2 minus column 4), smallest and largest selected mode frequency entering the harmonic correction, with the selection, the evaluation point, and the frequency definition all following the source, total classical and total quantum backmapping variance, displacement norms of the first and second updates (zero when fewer updates are taken), a flag that the relaxed potential energy is strictly below the reference value, and a flag that the classical coarse-grained free energy is strictly below the fixed-geometry value. Columns fourteen to seventeen, zero indexed, are the backmapping covariance audit summed over every monomer of the configuration, each contributing its own B atom: the largest eigenvalue of the quantum backmapping covariance of that monomer's B atom at the relaxed point, built from the monomer's selected modes as the sum over modes of the quantum mode variance times the outer product of the mass-unscaled eigenvector components on the B atom; that covariance's anisotropy, largest minus smallest eigenvalue over the trace; the signed alignment of its principal axis with the monomer plane normal, where the principal eigenvector is expressed in the monomer's local frame at the relaxed point and flipped so that its first component with absolute value above 1e-8 in the basis order x, y, z is positive, then projected on z; and the ratio of the quantum to the classical covariance trace; each of the four columns reports the sum of the per-monomer values. The penultimate column is the signed frame twist summed over every monomer: stack the frame vectors x, y, z of the frame contract as the rows of a matrix at the reference point and at the relaxed point, form the relative rotation as the relaxed-point matrix times the transpose of the reference-point matrix, extract the axis vector from half the differences of the off-diagonal antisymmetric entries, take the angle from the two-argument arctangent of the axis norm and half the trace minus one, and set the sign positive when the axis has nonnegative dot product with the reference-point z axis. The final column is the source's minimum-energy-manifold residual at the relaxed point, summed over every monomer: apply the pseudoinverse the source prescribes to the mass-scaled gradient and take the Euclidean norm of the result for each monomer, with the pseudoinverse, its evaluation point, and the placement of the mass scaling all following the source's definition of that manifold. With num_iters = 0 the audit must reproduce the conventional fixed-geometry rigidification identically.

Evaluate shr_audit on the five configurations listed below, in this order, with num_iters = 2 and T = 0.35. They have shapes (3,3,3), (4,3,3), (4,3,3), (5,3,3), (2,3,3) and are given as nested JSON, one monomer per line:

```json
[
 [
  [[1.033663, 0.108724, 0.16054], [0.112899, 0.073417, 0.022995], [-0.362965, 0.98136, 0.026921]],
  [[-4.294078, 0.010016, 0.0408], [-3.18449, -0.107037, -0.054749], [-2.905678, 0.672657, -0.601129]],
  [[-0.937176, -3.241764, 0.301703], [0.003554, -3.168499, -0.057999], [0.319276, -4.039705, -0.194912]]
 ],
 [
  [[0.271613, 0.085543, 0.706603], [0.091614, -0.023224, 0.292598], [-0.421447, 0.854928, -0.442858]],
  [[-2.334479, 0.617925, -0.334762], [-2.46615, -0.148027, 0.167377], [-1.249518, -0.608938, -0.400547]],
  [[0.536595, -1.348508, 0.009638], [0.008204, -2.137902, 0.017308], [0.136845, -2.611049, -1.123682]],
  [[-0.349612, -0.40276, -1.453091], [0.236177, 0.140533, -2.300676], [-0.612218, 0.189077, -2.936322]]
 ],
 [
  [[0.78118, 0.6165, -0.461584], [0.007969, 0.095973, 0.140201], [-0.161187, 0.192377, 0.902405]],
  [[-3.347254, -0.707507, -0.702641], [-2.704236, -0.004769, 0.031733], [-2.151768, -0.663321, 0.680483]],
  [[-0.905117, -2.782927, 0.189408], [-0.0752, -2.911689, 0.048541], [0.539791, -2.353142, 0.77435]],
  [[-0.561311, 0.065038, -3.446814], [-0.033451, -0.029028, -2.777209], [-0.374238, 0.424573, -2.065868]]
 ],
 [
  [[0.202603, 0.389625, 0.851976], [-0.026597, -0.025761, -0.019083], [0.092733, 0.775856, -0.650099]],
  [[-3.067929, 0.397487, 0.903875], [-3.146453, 0.004114, 0.04078], [-2.330466, -0.432468, -0.296686]],
  [[-0.082103, -4.183185, -0.326684], [-0.005234, -3.206419, -0.004281], [0.810262, -2.781893, -0.358398]],
  [[0.823373, 0.527608, -3.439266], [-0.018188, 0.008185, -3.21766], [-0.63673, 0.61384, -2.658203]],
  [[0.116225, 0.872441, 2.80348], [-0.057454, -0.028002, 3.193326], [0.362577, -0.638241, 2.644926]]
 ],
 [
  [[-0.01695, 0.578721, 0.791478], [-0.031673, 0.012649, -0.009394], [0.023738, 0.584845, -0.859335]],
  [[1.92156, 0.555159, 0.809247], [1.931787, 0.015884, -0.030533], [1.90437, 0.566946, -0.841134]]
 ]
]
```

 All outputs are float64, finite, and deterministic: two runs on identical inputs must agree exactly. Validate inputs and raise ValueError on malformed coordinates, nonpositive temperature, an invalid iteration count, or a nonpositive selected eigenvalue. As the final answer, report the sum over the five configurations of the classical SHR minus fixed-geometry free-energy difference (the sixth column) to six significant figures.

In your reasoning report the conventions you used for the subspace selection, the treatment of the pseudoinverse across the iterations, the point at which the harmonic frequencies are evaluated, the classical and quantum free-energy forms, the two backmapping variance forms, the zero-iteration identity, and the manifold residual, and give numerically the five per-configuration classical-minus-fixed-geometry gaps, which are the scalars that determine the final number.

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

cluster_energy

Goal
----
Compute the total potential energy of a cluster of flexible A-B-A monomers: harmonic B-A bonds and the harmonic A1-B-A2 angle inside every monomer, plus intermonomer Lennard-Jones interactions, B with B and every A with every A, using the parameter values fixed in the problem statement. Validate the coordinate array and raise ValueError on malformed input.

```python
def cluster_energy(coords):
    """coords: float array (I, 3, 3), atom order [A1, B, A2] per monomer,
    laid out (monomer, atom, xyz).

    Returns float: the total potential energy of the cluster."""
    return 0.0
```

### Step 2

cluster_gradient

Goal
----
Compute the exact analytic gradient of the cluster potential with respect to every coordinate, distributing bond and angle forces correctly among the three atoms of each monomer and applying action-reaction between Lennard-Jones partners. Same validation as the energy.

```python
import numpy as np


def cluster_gradient(coords):
    """coords: float array (I, 3, 3).

    Returns float64 array (I, 3, 3): the exact analytic gradient of the
    total potential with respect to every coordinate."""
    return np.zeros_like(np.asarray(coords, dtype=np.float64))
```

### Step 3

block_mass_scaled_hessian

Goal
----
Build the per-monomer mass-scaled Hessian blocks as exact analytic second derivatives of the bond, angle, and Lennard-Jones terms, assembled per block, symmetrized, and scaled by the inverse square-root masses on both sides. The grading tolerance is set so that finite-difference approximations of any order fail; only exact derivatives pass.

```python
import numpy as np


def block_mass_scaled_hessian(coords):
    """coords: float array (I, 3, 3).

    Returns float64 array (I, 9, 9): for each monomer the exact analytic
    Hessian of the total potential with respect to that monomer's own nine
    coordinates, symmetrized and scaled as M^-1/2 H M^-1/2. The blocks must
    be exact second derivatives; numerical differentiation of the gradient
    does not meet the grading tolerance."""
    return np.zeros((np.asarray(coords).shape[0], 9, 9))
```

### Step 4

reference_configuration

Goal
----
Rebuild every monomer at the isolated-monomer internal minimum inside its own frame using the exact frame contract from the problem statement: origin at B, x along the normalized bond bisector, z along the normalized cross product of the two bond vectors, y completing the right-handed frame, A1 at plus half the equilibrium angle from x toward y and A2 at minus half, both at the equilibrium bond length. Handle the near-collinear degeneracy exactly as the statement prescribes: when the sine of the bond angle falls below the stated threshold the plane normal comes from the deterministic basis-vector fallback, and zero-length bonds or a degenerate bisector raise ValueError.

```python
import numpy as np


def reference_configuration(coords):
    """coords: float array (I, 3, 3).

    Returns float64 array (I, 3, 3): every monomer rebuilt at the
    isolated-monomer internal minimum in its own frame per the frame
    contract in the problem statement."""
    return np.zeros_like(np.asarray(coords, dtype=np.float64))
```

### Step 5

shr_relax

Goal
----
Perform exactly num_iters updates of the paper’s subspace relaxation iteration, with the mass scaling and the treatment of the pseudoinverse across iterations following the source, and no tolerance-based exit. Validate the iteration count.

```python
import numpy as np


def shr_relax(coords, num_iters):
    """coords: float array (I, 3, 3) starting configuration; num_iters:
    exact number of relaxation updates, validated.

    Returns float64 array (I, 3, 3): the relaxed configuration after
    exactly num_iters updates of the source iteration."""
    return np.zeros_like(np.asarray(coords, dtype=np.float64))
```

### Step 6

harmonic_thermo

Goal
----
From a potential energy value, the selected frequencies, and a temperature, compute the classical and quantum coarse-grained free energies and the total classical and quantum positional mode variances of the paper’s backmapping construction, using the exact forms the paper gives. Validate the inputs.

```python
import numpy as np


def harmonic_thermo(V, omegas, T):
    """V: float potential energy; omegas: float array (L,) of selected
    frequencies; T: temperature, reduced units with hbar = kB = 1.

    Returns float64 array (4,): [classical free energy, quantum free
    energy, total classical positional mode variance, total quantum
    positional mode variance], forms per the source paper."""
    return np.zeros(4)
```

### Step 7

shr_audit

Goal
----
The final orchestrator. For each supplied configuration, chain the earlier sub-problem functions to produce the twenty declared columns. Columns 0 to 13 are the reference and relaxed potential energies, classical and quantum coarse-grained free energies, the fixed-geometry free energy from zero relaxation, the classical gap, smallest and largest selected frequency, total classical and quantum backmapping variances, the first and second update displacement norms, and the two strict-inequality flags. With num_iters equal to zero it must reproduce the fixed-geometry rigidification identically. Columns 14 to 17 are the backmapping covariance audit summed over every monomer of the configuration, each monomer contributing its own B atom at the relaxed point: the largest eigenvalue of the quantum covariance, its anisotropy, the signed alignment of its principal axis with that monomer's plane normal, and the quantum to classical trace ratio, with the principal-axis sign fixed by the declared frame and flip convention. Column 19 is the source’s Eq 6 minimum-energy-manifold residual at the relaxed point, summed over every monomer, with the stiff-subspace pseudoinverse and the mass scaling applied exactly as the source defines that manifold condition. Column 18 is the signed frame twist between the reference and relaxed points, also summed over every monomer, under the declared row-stacking, product-order, axis-extraction, and sign conventions.

```python
import numpy as np


def shr_audit(configs, num_iters, T):
    """configs: sequence of float arrays (I_k, 3, 3); num_iters: exact
    relaxation iteration count; T: temperature.

    Returns float64 array (n, 20): one row per configuration with the
    twenty declared audit columns, assembled by calling the earlier
    sub-problem functions. Columns 14 to 17 are the backmapping covariance
    audit summed over every monomer of the configuration, each monomer
    contributing its own B atom at the relaxed point: largest
    eigenvalue of the quantum covariance, its anisotropy, the signed
    alignment of its principal axis with that monomer plane normal under the
    declared sign convention, and the quantum to classical trace ratio. Column 18 is the signed frame twist
    between the reference and relaxed points, also summed over every monomer,
    under the declared row-stacking, product-order, axis-extraction, and
    sign conventions."""
    return np.zeros((len(configs), 20))
```
