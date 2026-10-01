# Material_Science-Molecular_Modeling-48

## Background

*Embedded-atom-method potentials.* In the EAM the energy of a metal is a sum of pairwise terms and a per-atom embedding energy that is a nonlinear function of the local electron density, itself a pairwise sum. The nonlinearity in $\rho$ is what lets a short-ranged model reproduce the elastic anisotropy and the surface energies of a metal, and it is why the force on atom $i$ is not a sum of pair forces: differentiating $\sum_k F(\rho_k)$ produces, for each neighbour $j$, a term carrying $F'(\rho_i) + F'(\rho_j)$, because atom $i$ moving changes both its own density and its neighbour's.

*Elastic constants from a lattice sum.* For a centrosymmetric crystal under homogeneous strain no internal relaxation is required, so the elastic constants are second derivatives of the energy per atom with respect to the strain tensor, divided by the volume per atom. The pure axial strain gives $c_{11}$ directly; a biaxial strain gives $c_{11}+c_{12}$; a pure shear gives $c_{44}$, with the usual factor of two between the tensor and engineering shear strains.

*Fisher information and the Cramer-Rao bound.* The Fisher information matrix of a parametric statistical model measures how sharply the likelihood of data distinguishes nearby parameter values. Its inverse lower-bounds the covariance of any unbiased estimator, so a large FIM in a given parameter direction means that direction is well constrained by the data. For independent data the FIM is additive, and each datum's contribution is scaled by its inverse variance - which is exactly what a weight in a weighted least-squares loss is. Because the FIM depends on the model and the measurement conditions but not on the measured values, it can be computed *before* the data are acquired, which is what makes optimal experimental design possible.

*Optimal design by matching, not by maximising.* Classical optimal design maximises a scalar summary of the FIM - its determinant, its trace, its smallest eigenvalue. Those summaries are blind to *which* parameter directions the intended prediction actually needs. An alternative is to compare two information matrices directly, through the positive-semidefinite ordering, so that the geometry of both is respected: the data are required to supply at least as much information as the prediction demands, in every direction at once. Once such a condition is imposed, the propagated prediction uncertainty is automatically bounded by the target, for a reason that is worth working out explicitly.

*Identifiability.* A parameter that does not affect any of the modelled observables contributes an exact zero row and column to the corresponding information matrix. A piecewise model makes this state-dependent: parameters that appear only in a branch the data never visit are invisible to those data. Whether a target's information lives inside the subspace the data can constrain is exactly the question of whether the matching condition can be satisfied at all.

## Problem

An interatomic potential fitted to a large, redundant training set is expensive to build and gives no guarantee about the precision of the properties it will be used to predict. A published active learning strategy repairs this by *matching information*: rather than minimising parameter uncertainty indiscriminately, it requires that the Fisher information supplied by the selected training data be at least as large - as a matrix inequality on the parameter space - as the information needed to predict a set of target properties to prescribed uncertainties. This task asks you to **evaluate the guarantee that condition buys**, exactly, on a fully specified deterministic instance built around an embedded-atom-method potential for body-centred-cubic tantalum. Nothing is trained and nothing is random.

*The potential.* The total energy of a configuration of $N$ atoms is

$$E \;=\; \sum_{i<j}\phi(r_{ij}) \;+\; \sum_{i=1}^{N}F(\rho_i), \qquad \rho_i \;=\; \sum_{j\neq i}f(r_{ij}),$$

with the pair potential, the electron-density function and the piecewise embedding function

$$\phi(r) = \frac{A\exp[-\alpha(r/r_e-1)]}{1+(r/r_e-\kappa)^{20}} - \frac{B\exp[-\beta(r/r_e-1)]}{1+(r/r_e-\lambda)^{20}}, \qquad f(r) = \frac{f_e\exp[-\beta(r/r_e-1)]}{1+(r/r_e-\lambda)^{20}},$$

$$F(\rho) = \begin{cases} \sum_{i=0}^{3}F_{ni}\big(\tfrac{\rho}{\rho_n}-1\big)^{i}, & \rho < \rho_n,\ \ \rho_n = 0.85\rho_e,\\[4pt] \sum_{i=0}^{3}F_{i}\big(\tfrac{\rho}{\rho_e}-1\big)^{i}, & \rho_n \le \rho < \rho_0,\ \ \rho_0 = 1.15\rho_e,\\[4pt] F_e\Big[1-\eta\log\big(\tfrac{\rho}{\rho_s}\big)\Big]\big(\tfrac{\rho}{\rho_s}\big)^{\eta}, & \rho_0 \le \rho. \end{cases}$$

The twenty numerical parameters are fixed throughout, in this order,

```
index  0 r_e      1 f_e      2 rho_e    3 rho_s    4 alpha    5 beta
       6 A        7 B        8 kappa    9 lambda
      10 F_n0    11 F_n1    12 F_n2    13 F_n3
      14 F_0     15 F_1     16 F_2     17 F_3     18 eta     19 F_e

 2.860082   3.086341  33.787168  33.787168   8.489528   4.527748
 0.611679   1.032101   0.176977   0.353954
-5.103845  -0.405524   1.112997  -3.585325
-5.14       0.0        1.640098   0.221375   0.848843  -5.141526
```

with energies in eV and lengths in angstrom. Both $\phi$ and $f$ are **hard-truncated**: they and their derivatives are taken to be exactly $0$ for $r \ge r_c$, with $r_c = 6.0$ A, and no shifting of any kind is applied. All sums run over periodic images: for a box of edge lengths $(L_x,L_y,L_z)$, every image separation shorter than $r_c$ contributes, not only the minimum image.

*Parameters that vary.* Only **seven** of the twenty are treated as adjustable - indices $0, 5, 6, 7, 8, 18, 19$, that is $r_e$, $\beta$, $A$, $B$, $\kappa$, $\eta$ and $F_e$ - and they are handled in a **logarithmic** parameterisation, $\theta_j = \log|p_j|$ at fixed sign. Every parameter derivative below is a derivative with respect to $\theta$, taken by the central difference $\big[g(p_j e^{h}) - g(p_j e^{-h})\big]/(2h)$ with $h = 5\times10^{-3}$, all other parameters held fixed. Note that one of the seven is negative, so the perturbation is multiplicative, not additive in $\log p$.

*Target properties.* The quantities whose precision is being guaranteed are the **five indicator properties of the BCC crystal that the source paper adopts as its quantities of interest**, each paired with the **target uncertainty the paper assigns it**. The five target uncertainties are **not given here and must be fetched from the paper**; the answer depends on the ratios among them, so guessing them up to an overall factor does not help. So that the graded number is reproducible, the conventions for computing the properties themselves are fixed here. The equilibrium lattice constant $a_0$ is the stationary point of the energy per atom of the ideal BCC lattice; the reference implementation locates it by Newton iteration on the analytic $dE/da$ starting from $3.30$ A with a central-difference second derivative of step $10^{-4}$, but the stationary point is determined to machine precision and any accurate root finder reproduces it. The volume per atom is $a_0^3/2$. Elastic constants are second derivatives of the energy per atom with respect to homogeneous strain at $a_0$, evaluated by the three-point stencil $\big[E(\varepsilon)-2E(0)+E(-\varepsilon)\big]/h_\varepsilon^2$ with $h_\varepsilon = 10^{-2}$, applied to the deformation $\boldsymbol r \mapsto (\mathcal I + \boldsymbol\varepsilon)\boldsymbol r$ for the strain patterns $\varepsilon_{xx}=e$; $\varepsilon_{xx}=\varepsilon_{yy}=e$; and $\varepsilon_{yz}=\varepsilon_{zy}=e/2$; divided by the volume per atom and converted with $1\,\mathrm{eV/A^3} = 160.21766208$ GPa. No internal relaxation is needed, the lattice being centrosymmetric.

*Candidate training data.* The pool is twelve periodic configurations, $m = 0,\dots,11$. Configuration $m$ is a $2\times2\times2$ BCC supercell of lattice constant $a_m = 3.05 + 0.06\,(m \bmod 8)$ A in a cube of edge $L_m = 2a_m$, so $N = 16$ atoms, built by taking the eight cube corners $(i,j,k)a_m$ with $i,j,k \in \{0,1\}$ in `numpy.meshgrid(..., indexing='ij')` order followed by the eight body centres $(i+\tfrac12,j+\tfrac12,k+\tfrac12)a_m$ in the same order, then displacing atom $n = 0,\dots,15$ of that list by

$$\boldsymbol u_n = \mathcal A_m\Big[\sin(1.7n + 0.9m + 0.3),\; \sin(2.3n + 1.4m + 1.1),\; \sin(3.1n + 2.2m + 1.9)\Big], \qquad \mathcal A_m = 0.08 + 0.04\,(m \bmod 4)\ \text{A},$$

and wrapping the result back into $[0, L_m)$. Each configuration supplies **one** configuration-energy datum and **one per-atom force datum for each of its sixteen atoms** (all three Cartesian components of an atom's force sharing that atom's single weight), so the pool holds $12$ energy data and $192$ force data.

*Weighting.* Follow the source paper's own simplest weighting scheme: **all configuration-energy residuals share one weight $w_E$ and all per-atom force residuals share one weight $w_F$**, with the ratio fixed at $w_E/w_F = 10^{-2}$. Under that scheme the information-matching condition determines the two weights completely, because it fixes their common scale; use the **smallest** scale that satisfies it, which is what the paper's weight-optimisation objective selects.

*What to report.* With those weights in force, propagate the resulting parameter information back to the five indicator properties to obtain a predicted uncertainty $\sigma_n$ for each, and **report the sum over the five indicator properties of $\sigma_n/\delta_n$**, where $\delta_n$ is that property's target uncertainty. **Find the source paper and take from it**: the definition of the Fisher information matrix and how the information of independent data accumulates; how that definition reduces to a computable expression for the weighted least-squares energy-and-force loss the paper trains with, and hence what a single per-atom force datum and a single configuration-energy datum each contribute; how the information required by a target property with a prescribed uncertainty is expressed as a matrix; the exact form of the information-matching condition relating the two, and the objective that is minimised over the data weights subject to it; the target uncertainty the paper assigns each of the five indicator properties, and why it adopts those five rather than the property it ultimately cares about; and the formula that turns the total information matrix into a prediction uncertainty for one property. Do not substitute a plausible reconstruction: the graded number depends on the matrix - not scalar - form of the matching condition, on the inverse *square* of each target uncertainty, on the logarithmic parameterisation and on the quadratic form that appears in the propagation formula.

State in your reasoning: the five indicator properties you identified and the five target uncertainties you fetched; the five computed property values; the expression you used for the FIM of one force datum and of one energy datum, and for the FIM of a target property; the rank of the target information matrix and which parameters lie in its null space, with the physical reason; the common weight scale the matching condition selects and the value of the weight-optimisation objective it implies; the five predicted uncertainties and the five ratios $\sigma_n/\delta_n$; whether every ratio is below one and why that had to be so; and the two values the same pipeline returns when the pool is restricted to configuration energies alone and to per-atom forces alone.

Output Format Requirements: Emit `<final_answer>` immediately, then `<reasoning>`. Do not write a long derivation before the tags. You must emit exactly one finite decimal inside `<final_answer>...</final_answer>`, even if the value is approximate or you are unsure. Rules:

- The tags are required. Do not omit them or leave them empty.
- The tagged value must be a finite decimal (examples: `0.4847`, `12.6`, `1.05`). Not NaN, not Inf, not a fraction string, not a vector, not prose.
- Put only that one number between the tags. No units, no words, no extra lines. Keep `<reasoning>` short (a few hundred words): the diagnostics listed above are its required content and must all appear, and beyond them show only the few scalars that determine the final number. Do not paste the atomic coordinates, the full Jacobians, the Fisher matrices, or per-configuration tables.

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

eam_pair_and_density

Goal
----
Evaluate the embedded-atom pair potential $\\phi(r)$ and the electron-density function $f(r)$ of the source paper's EAM parameterisation, ****together with their first derivatives**** with respect to $r$, at an array of interatomic distances. Return all four stacked along a new leading axis, in the order $[\\phi,\\, f,\\, \\phi',\\, f']$, so that the shape is `(4,) + r.shape`. Both functions are ****hard-truncated****: every one of the four is exactly $0$ where $r \\ge r_{\\mathrm{cut}}$, with no shifting and no smoothing. The derivatives are graded because the forces of step 3 and the equilibrium lattice constant of step 4 are built from them analytically, not by differencing.

```python
import numpy as np

def eam_pair_and_density(r, params, r_cut=6.0):
    """r: array of interatomic distances (any shape, all >= 0).
    params: length-20 EAM parameter vector in the order
      (r_e, f_e, rho_e, rho_s, alpha, beta, A, B, kappa, lambda,
       F_n0..F_n3, F_0..F_3, eta, F_e).
    r_cut: hard cutoff; every returned quantity is exactly 0 where r >= r_cut.
    Return the real array [phi, f, dphi/dr, df/dr] of shape (4,) + r.shape."""
    # Implement per the formulas above.
    return None
```

### Step 2

eam_embedding

Goal
----
Evaluate the ****piecewise**** embedding function $F(\\rho)$ of the same parameterisation, and its first derivative $F'(\\rho)$, at an array of local electron densities. Return the two stacked along a new leading axis, in the order $[F,\\, F']$, so that the shape is `(2,) + rho.shape`. The three branches are selected by $\\rho < \\rho_n$, $\\rho_n \\le \\rho < \\rho_0$ and $\\rho_0 \\le \\rho$, with $\\rho_n = 0.85\\rho_e$ and $\\rho_0 = 1.15\\rho_e$; the half-open convention at both knots is graded. ****All three branches matter****: the third one, the only place where $\\eta$ and $F_e$ appear, is what makes those two parameters identifiable at all, and a candidate who implements only the polynomial branches will find later steps failing with a singular information matrix rather than returning a wrong number.

```python
import numpy as np

def eam_embedding(rho, params):
    """rho: array of local electron densities (any shape, all >= 0).
    params: the same length-20 EAM parameter vector as in step 1.
    Return the real array [F, dF/drho] of shape (2,) + rho.shape."""
    # Implement per the formulas above.
    return None
```

### Step 3

eam_energy_forces

Goal
----
Evaluate the total energy and ****all per-atom forces**** of a periodic configuration under the EAM potential of steps 1 and 2, and return them as one flat array, the energy first and then the forces in atom-major, Cartesian-minor order: $[E, F_0x, F_0y, F_0z, F_1x, ...]$, of length $3N+1$. The box is orthorhombic, given either as a scalar edge length or as a length-3 vector, and the sums run over ****all periodic images within the cutoff****, not the minimum image alone - at the box sizes used here $L < 2r_{\\mathrm{cut}}$, so an atom interacts with more than one image of some neighbours and with images of itself. The forces are graded as analytic derivatives, and their embedding contribution is the point of this step.

```python
import numpy as np

def eam_energy_forces(positions, cell, params, r_cut=6.0):
    """positions: real (N, 3) array of atomic positions.
    cell: float, or length-3 sequence (Lx, Ly, Lz), the orthorhombic box edges.
    params: the same length-20 EAM parameter vector as in step 1.
    r_cut: the hard cutoff of step 1.
    Return the real 1-D array [E, F_0x, F_0y, F_0z, F_1x, ...] of length 3N + 1."""
    # Implement per the formulas above.
    return None
```

### Step 4

bcc_indicator_properties

Goal
----
Return the ****five indicator properties of the BCC crystal that the source paper adopts as its quantities of interest****, computed from the potential itself, in the order $[a_0,\\; E_{\\mathrm{coh}},\\; c_{11},\\; c_{12},\\; c_{44}]$ - the equilibrium lattice constant in angstrom, the cohesive energy in eV per atom, and the three cubic elastic constants in GPa. ****Nothing beyond the conventions below is derived for you:**** set up the ideal BCC lattice sum, locate its energy minimum, and obtain the elastic constants as second derivatives of the energy per atom with respect to homogeneous strain. The crystal is centrosymmetric, so no internal relaxation is needed.

```python
import numpy as np

def bcc_indicator_properties(params, r_cut=6.0, a_start=3.30, h_a=1e-4,
                             n_newton=8, h_strain=1e-2):
    """params: the same length-20 EAM parameter vector as in step 1.
    r_cut: the hard cutoff. a_start, h_a, n_newton: the Newton iteration for the
    equilibrium lattice constant. h_strain: the strain step of the second-difference
    stencil for the elastic constants.
    Return the real length-5 array [a0 (A), E_coh (eV/atom), c11, c12, c44 (GPa)]."""
    # Implement per the principle above.
    return None
```

### Step 5

log_parameter_jacobian

Goal
----
Return the Jacobian of an arbitrary vector-valued model output with respect to the ****logarithms**** of a chosen subset of the potential's parameters, by central differences. Column $j$ of the result is $\\partial(\\text{output})/\\partial\\theta_j$ with $\\theta_j = \\log|p_j|$ at fixed sign, evaluated as $\\big[\\mathrm{func}(p\\ \\text{with}\\ p_j \\to p_je^{h}) - \\mathrm{func}(p\\ \\text{with}\\ p_j \\to p_je^{-h})\\big]/(2h)$, every other entry of `params` held fixed. ****The perturbation is multiplicative, not additive in $\\log p$****, which is the only formulation that works for the one master parameter that is negative; taking a logarithm of it directly is a domain error. The output of `func` is flattened to one dimension before differencing, so a scalar-valued `func` gives a Jacobian with one row.

```python
import numpy as np

def log_parameter_jacobian(func, params, master_idx=(0, 5, 6, 7, 8, 18, 19), h=5e-3):
    """func: callable taking a length-20 parameter vector and returning a scalar or an
    array of model outputs.
    params: the length-20 EAM parameter vector to differentiate about.
    master_idx: the indices of params to differentiate with respect to, in order.
    h: the central-difference step in the logarithm of each parameter.
    Return the real (n_out, n_master) Jacobian with respect to log|p_j|."""
    # Implement per the formulas above.
    return None
```

### Step 6

data_fisher_matrices

Goal
----
Build the two ****pooled Fisher information matrices**** of the candidate training data: one accumulated over the configuration-energy data, one over the per-atom force data. Return them stacked as an array of shape $(2, P, P)$ with $P$ the number of master parameters, energy first. ****The per-datum Fisher information is not given: derive it**** from the definition in the source paper for the Gaussian model implied by the paper's weighted sum of squared energy and force residuals, and note that the three Cartesian components of one atom's force share that atom's single weight, so an atom is one datum, not three. The matrices returned here are **unweighted** pools - the weights are the common scales that step 7 determines - so each is the plain sum of its data's per-datum matrices.

```python
import numpy as np

def data_fisher_matrices(positions_list, cells, params,
                         master_idx=(0, 5, 6, 7, 8, 18, 19), h=5e-3, r_cut=6.0):
    """positions_list: sequence of (N_m, 3) position arrays, one per configuration.
    cells: sequence of box specifications, one per configuration (each a scalar or a
    length-3 vector); a single scalar is broadcast to every configuration.
    params: the length-20 EAM parameter vector to evaluate the information at.
    master_idx, h: passed through to the parameter Jacobian.
    r_cut: the hard cutoff.
    Return the real array np.stack([I_E, I_F]) of shape (2, P, P), P = len(master_idx)."""
    # Implement per the principle above.
    return None
```

### Step 7

minimal_information_scale

Goal
----
Given a pooled data information matrix and the Jacobian of the target quantities of interest with their target uncertainties, return the ****smallest common weight scale $t \\ge 0$ for which the source paper's information-matching condition holds****. ****Neither the target's information matrix nor the matching condition is given: derive both**** - the first from how a scalar prediction with a prescribed uncertainty translates into an information requirement, the second from the paper. Then reduce the condition to something computable: with a single free scale it is not a semidefinite program but a symmetric-definite generalised eigenvalue problem, and the answer is a single eigenvalue. Raise a `ValueError` if the pooled matrix is singular, because then no finite common weight can satisfy the condition in the deficient directions.

```python
import numpy as np

def minimal_information_scale(I_pool, qoi_jacobian, delta):
    """I_pool: symmetric positive definite (P, P) pooled data information matrix.
    qoi_jacobian: (Q, P) Jacobian of the Q quantities of interest with respect to the
    same P parameters.
    delta: length-Q sequence of target uncertainties, in the units of the QoIs.
    Return the smallest common weight scale t >= 0 satisfying the matching condition,
    a float. Raise ValueError if I_pool is singular."""
    # Implement per the principle above.
    return None
```

### Step 8

alim_uncertainty_ratio_sum

Goal
----
Run the whole benchmark. Build the pool of $n_conf$ candidate configurations from the closed form below; evaluate the pooled energy and force information matrices at the fixed EAM parameters; combine them into one pooled matrix using the fixed ratio of the energy weight to the force weight; evaluate the Jacobian of the five indicator properties; find the smallest common weight scale that satisfies the information-matching condition; form the total information matrix at that scale; ****propagate it back to a predicted uncertainty for each of the five indicator properties**** using the source paper's propagation formula; and return the ****sum of the five ratios of predicted to target uncertainty****. Setting `data='E'` restricts the pool to configuration energies and `data='F'` to per-atom forces; $delta=None$ means the source paper's own target uncertainties for the five indicator properties, in the order of step 4.

```python
import numpy as np

def alim_uncertainty_ratio_sum(params=None, n_conf=12, n_cell=2, data="EF",
                               w_ratio=1.0e-2, r_cut=6.0, h=5.0e-3, delta=None,
                               master_idx=(0, 5, 6, 7, 8, 18, 19),
                               a_start=3.30, h_a=1.0e-4, n_newton=8, h_strain=1.0e-2):
    """params: length-20 EAM parameter vector; None means the fixed tantalum set.
    n_conf, n_cell: number of candidate configurations and the supercell repeat.
    data: 'E', 'F' or 'EF' - which data types enter the pooled information.
    w_ratio: the ratio of the common energy weight to the common force weight.
    r_cut: the hard cutoff. h: the parameter-Jacobian step.
    delta: the five target uncertainties; None means the source paper's values.
    master_idx: the adjustable parameter indices.
    a_start, h_a, n_newton, h_strain: passed to the indicator-property step.
    Return the sum over the five indicator properties of sigma_n / delta_n, a float."""
    # Implement per the principle above.
    return None
```
