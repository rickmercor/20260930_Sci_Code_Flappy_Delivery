# Mathematics-Computational_Mechanics-51

## Background

Fibre-based materials such as paper, felt and many porous or biological media are, mechanically, networks of slender beams meeting at rigid junctions. Resolving every fibre in three dimensions is prohibitive, so the geometry is abstracted to a spatial network: a graph embedded in space carrying one-dimensional elastodynamic equations along its edges, coupled at its nodes by continuity and balance conditions. Two features make such networks awkward to discretise. Segment lengths vary enormously, so an explicit time integrator inherits a severe step-size restriction from the shortest fibre; and each edge is parametrised in its own direction, so quantities meeting at a junction arrive with edge-dependent orientations that have to be reconciled before they can be balanced. A hybridizable discontinuous Galerkin formulation addresses both difficulties structurally, by reorganising the unknowns so that the algebra solved at each step reflects the graph rather than the fibres, and by admitting an implicit discretisation in time. Whatever a scheme of this kind preserves exactly is the sharpest available check on an implementation, because it separates a result that merely looks plausible from one that respects the structure of the underlying problem.

## Problem

Fibre-based materials such as paper, felt and many porous media are, mechanically, networks of slender beams meeting at rigid junctions. Resolving them in three dimensions is prohibitive, so they are modelled as a spatial network: one-dimensional elastodynamic equations along each edge, coupled at the nodes by algebraic conditions. The source formulates the shear-deformable beam network in a hybrid dual mixed form and discretises it with a hybridizable discontinuous Galerkin method, combined with an implicit time discretisation. Your job is to implement that scheme exactly as the source specifies it and audit its behaviour on the fixed network family below. The construction rests on a handful of conventions that the source states and that a plausible alternative reading would get wrong; they are not reproduced here, and recovering them from the literature is the substance of the problem. The problem is calibrated so that departing from those choices changes the reported numbers by amounts far beyond the grading tolerance.

Each configuration is a network of n_nodes nodes with integer parameter a. Node j sits at [((3j+a) mod 7) - 3, ((5j+2a) mod 11) - 5, ((2j+3a) mod 13) - 6] divided by 3. Node j is joined to node j+1 for every j below n_nodes-1, and to node j+3 for every j below n_nodes-3; edges are stored in that order and each is parametrised from its lower-indexed to its higher-indexed endpoint. Nodes 0 and n_nodes-1 are held fixed with prescribed nodal values of zero, and there are no distributed loads. Edge k carries the four coefficient blocks I + c*outer(w,w), where w is the normalisation of [((k+1) mod 3)+1, ((k+2) mod 4)+1, ((k+3) mod 5)+1] and c is 0.6, 0.9, 0.4 and 0.7 for the block of the force resultants, the moment resultants, the displacement and the rotation respectively. The three configurations are (n_nodes, a) = (7, 3), (6, 5) and (8, 2).

Fix the following numerical conventions so the results are reproducible. Use the Legendre polynomial basis of degree at most 3 in the arclength parameter on each edge, taken component by component so that the degree-of-freedom index is component*(p+1) + mode, and integrate with Gauss-Legendre quadrature on 2p+6 points, which is exact for every integrand here. Take the stabilisation parameter equal to 1.0 on every edge. Advance 24 steps of size 0.02. The initial primal field on edge k is the L2 projection onto that polynomial space of the function whose component c at arclength x is sin((c+1)*x/2 + 0.3*(k+1))/(c+2), and the initial velocity field is the projection of cos((c+2)*x/3 + 0.2*(k+1))/(c+3). Solve every linear system directly; no iterative solver, tolerance or preconditioner enters the reported numbers.

Compute, for each configuration, the discrete energy of the initial state and of the state after the declared number of steps; the Euclidean norm of the converged nodal unknowns at the final time and the summed norms over the free nodes; and the two constants of the source's spectral equivalence between the operator actually solved at each step and a combination of two forms posed directly on the graph, smallest first. Take the answer to be the sum, over the three configurations, of the summed norm of the full nodal value over the free nodes at the final time.

Every reported quantity must be float64, finite and deterministic: two runs on identical inputs must agree exactly. In your reasoning report, quote the RELATIVE agreement between each configuration's initial and final discrete energy as a number rather than describing it qualitatively, and say why the agreement is exact here rather than merely close. Give the ratio of the two spectral equivalence constants for each configuration and say what that ratio buys, how the step size enters it, and what the size of the system solved at each step depends on. State whether the operator solved at each step is symmetric positive definite. Also state, in ONE sentence each, the source conventions your numbers actually depend on, covering the nodal coupling of the network, the beam constitutive relation, the numerical flux, the discrete energy, the initial state, the time discretisation, the weighting of the two graph forms, the combination of those forms the spectral equivalence is stated for, and the representation that exhibits the condensed node operator as symmetric positive definite. For each of those, name the choice the source makes and say what an obvious alternative reading would have been; a convention that no plausible alternative competes with is not worth a sentence. Report the required values as a compact list of numbers and each convention as a single sentence, not as a derivation; done that way the whole report fits the length the output format allows. As the final answer, report that sum to six significant figures.

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

Rules:
- The tags are required. Do not omit them or leave them empty.
- The tagged value must be a finite decimal (examples: 12.3456, -0.802, 45.9). Not NaN, not Inf, not a fraction string, not a vector, not prose.
- Put only that one number between the tags. No units, no words, no extra lines.
- Budget your response so that the <final_answer> tag is always reached and closed. If you are running long, stop the intermediate work and emit the answer.

Keep the reasoning under roughly 900 words, naming the source's conventions briefly rather than deriving them, and do not tabulate basis functions, element matrices, per-edge coefficient vectors, or any per-degree-of-freedom values.
```

## Your task

Implement **all 9 functions** below into a single file at `/app/solution.py`. Requirements:

- Use the EXACT function name and signature shown for each step.
- Define every function at module scope (importable).
- Do NOT include tests, example usage, prints, or a `__main__` block.
- Available libraries: numpy, scipy, sympy, pandas, scikit-learn, networkx.

Start your file with these imports:

```python
import numpy as np
```

### Step 1

edge_frames

Goal
----
Given the node coordinates of a spatial network and its list of edges, return one row per edge holding, in order, the three components of that edge's unit tangent vector, the edge length, and then one entry per node giving the source's orientation sign of that edge with respect to that node, or zero when the node is not an endpoint of the edge. The sign convention is the source's and is fixed by the direction of the edge tangent relative to the node; recover it from the paper. Raise ValueError on an edge of zero length.

```python
import numpy as np


def edge_frames(nodes, edges):
    """nodes: (n_nodes, 3) coordinates; edges: list of (a, b) node index pairs.
    Returns (E, 4 + n_nodes) float64 with the unit tangent in columns 0-2, the
    length in column 3, and the orientation sign of the edge with respect to
    every node in columns 4 onwards (zero for non-endpoints)."""
    return np.zeros((len(edges), 4 + len(nodes)))
```

### Step 2

timoshenko_operators

Goal
----
Assemble the block operators of the source's beam equations of motion for one edge. Return a stack of three 6x6 matrices: first the block-diagonal coefficient matrix pairing the force resultants with the moment resultants, second the block-diagonal coefficient matrix pairing the displacement with the rotation, and third the source's tangent-coupling operator, the linear map built from the edge tangent that couples the two three-component halves of the state in the way the source's equations of motion require. Its structure is the source's convention, so recover it from the paper rather than assuming a symmetric or diagonal form.

```python
import numpy as np


def timoshenko_operators(i_hat, Cn, Cm, Cu, Cr):
    """i_hat: (3,) unit tangent; Cn, Cm, Cu, Cr: (3, 3) symmetric positive
    definite blocks. Returns (3, 6, 6) float64: the two block-diagonal
    coefficient matrices followed by the tangent-coupling operator."""
    return np.zeros((3, 6, 6))
```

### Step 3

hdg_local_matrices

Goal
----
Build the five edge matrices of the source's hybrid dual mixed formulation on a single edge, using the Legendre polynomial basis of degree at most p in the arclength parameter, taken component by component so that the degree-of-freedom index is component*(p+1) + mode. Return a stack of five N x N matrices in this order: the matrix of the source's dual bilinear form, the matrix of the source's mixed form coupling the dual and primal variables, the plain mass matrix, the matrix of the source's auxiliary bilinear form used by the first-order-in-time formulation, and the matrix that collects the two endpoint traces. The mixed form contains a derivative contribution together with the tangent coupling of the previous sub-problem; its exact composition is the source's. Integrate exactly with Gauss-Legendre quadrature on 2p+6 points. Raise ValueError unless p >= 0 and the length is positive.

```python
import numpy as np


def hdg_local_matrices(p, L, i_hat, Cn, Cm, Cu, Cr):
    """p: polynomial degree; L: edge length; i_hat: (3,) unit tangent;
    Cn, Cm, Cu, Cr: (3, 3) coefficient blocks. Returns (5, N, N) float64 with
    N = 6*(p+1), stacked in the order described above."""
    return np.zeros((5, 6 * (p + 1), 6 * (p + 1)))
```

### Step 4

hdg_numerical_flux

Goal
----
Evaluate the source's HDG numerical flux at both endpoints of one edge. Row 0 is the value at the start node of the edge and row 1 the value at the end node. The flux combines the endpoint trace of the dual variable, carried with that endpoint's orientation sign, and a stabilisation contribution proportional to the mismatch between the endpoint trace of the primal variable and the hybrid nodal value there. The exact combination, including which variable the stabilisation penalises, is the source's; recover it from the paper. The 12-vector of hybrid values holds the start-node value in its first six entries and the end-node value in the last six.

```python
import numpy as np


def hdg_numerical_flux(q_edge, y_edge, lam12, p, L, tau):
    """q_edge, y_edge: (6*(p+1),) coefficient vectors of the dual and primal
    variables on the edge; lam12: (12,) hybrid nodal values; p, L: degree and
    length; tau: stabilisation parameter. Returns (2, 6) float64."""
    return np.zeros((2, 6))
```

### Step 5

edge_local_solver

Goal
----
Reconstruct the three interior fields of one edge from the hybrid nodal data, by solving the edge-local block system of the source's fully discrete scheme. Return them stacked as rows in the order dual, primal, auxiliary. The block system couples the dual bilinear form, the mixed form and its transpose, the stabilised trace term and the two mass couplings that carry the time discretisation; its exact block structure and the way the hybrid data enters the two right-hand sides are the source's. The 12-vector of hybrid values holds the start-node value in its first six entries and the end-node value in the last six, and nu0 and nuL are that edge's orientation signs at the start and end nodes. Raise ValueError unless dt and tau are positive. Assemble by calling the earlier sub-problem functions.

```python
import numpy as np


def edge_local_solver(p, L, i_hat, Cn, Cm, Cu, Cr, tau, dt, nu0, nuL, lam12, rhs_y, rhs_z):
    """p, L: degree and edge length; i_hat: (3,) tangent; Cn, Cm, Cu, Cr: (3, 3)
    blocks; tau: stabilisation; dt: time step; nu0, nuL: orientation signs at the
    two endpoints; lam12: (12,) hybrid nodal values; rhs_y, rhs_z: (6*(p+1),)
    right-hand sides. Returns (3, 6*(p+1)) float64 stacked as dual, primal,
    auxiliary."""
    return np.zeros((3, 6 * (p + 1)))
```

### Step 6

condensed_node_system

Goal
----
Eliminate every edge interior and assemble the source's condensed system on the free nodes, returning the operator with the load vector appended as a final column. Free nodes are those not in the Dirichlet set, ordered by increasing node index, with the six components of a node occupying consecutive rows. The condensed operator is obtained by substituting the edge-local reconstructions into the source's nodal condition, which is imposed on the numerical flux rather than on the dual variable directly, summed over the edges incident to each free node with each edge's own orientation sign. Contributions of Dirichlet nodes move to the load vector. The resulting operator is symmetric positive definite. Assemble by calling the earlier sub-problem functions.

```python
import numpy as np


def condensed_node_system(nodes, edges, dirichlet, coeffs, p, tau, dt, rhs_y, rhs_z, lamD):
    """nodes: (n_nodes, 3); edges: list of (a, b); dirichlet: set of node indices;
    coeffs: per-edge [Cn, Cm, Cu, Cr]; p, tau, dt as before; rhs_y, rhs_z: lists of
    (6*(p+1),) per-edge right-hand sides; lamD: dict of prescribed nodal values.
    Returns (6*nf, 6*nf + 1) float64."""
    return np.zeros((6, 7))
```

### Step 7

network_time_step

Goal
----
Advance the network state by one step of the source's energy-conservative implicit time discretisation and return the hybrid nodal values on the free nodes at the new time level. The two per-edge right-hand sides carry the previous dual, primal and auxiliary fields together with the previous hybrid nodal state, each with the weight the source's scheme gives it; the previous hybrid state is supplied per edge as a 12-vector, start node first. How the source treats quantities that sit between two time levels, and the weights that follow from it, must be recovered from the paper, and they are what make the discrete energy exactly conserved rather than merely bounded. Assemble by calling the earlier sub-problem functions.

```python
import numpy as np


def network_time_step(nodes, edges, dirichlet, coeffs, p, tau, dt, Q, Y, Z, lam_prev, lamD):
    """Q, Y, Z: lists of (6*(p+1),) per-edge dual, primal and auxiliary fields at
    the previous time level; lam_prev: list of per-edge (12,) hybrid values at the
    previous level; other arguments as before. Returns (6*nf,) float64."""
    return np.zeros(6)
```

### Step 8

graph_forms

Goal
----
Assemble the source's two graph-level bilinear forms on the free nodes and return them stacked, the mass-type form first and the weighted graph Laplacian-type form second. Both are built by summing a contribution from every edge of the network, each written in terms of the values the nodal function takes at that edge's two endpoints and weighted by that edge's length. One form sees those two endpoint values separately, the other sees only their difference, and the two carry opposite powers of the edge length together with a common numerical prefactor. The exact weights, the prefactor and which power goes with which form are the source's convention; recover them from the paper rather than assuming an unweighted graph Laplacian. Free nodes are those not in the Dirichlet set, ordered by increasing node index, with the six components of a node occupying consecutive rows; endpoints that are Dirichlet nodes contribute nothing. Raise ValueError on an edge of zero length.

```python
import numpy as np


def graph_forms(nodes, edges, dirichlet):
    """nodes: (n_nodes, 3) coordinates; edges: list of (a, b) node index pairs;
    dirichlet: set of fixed node indices. Returns (2, 6*nf, 6*nf) float64 with the
    mass-type form first and the weighted graph Laplacian-type form second."""
    return np.zeros((2, 6, 6))
```

### Step 9

beam_network_audit

Goal
----
Run the whole scheme on the three declared network configurations and return one row each. A configuration with n_nodes nodes and integer parameter a places node j at [((3j+a) mod 7) - 3, ((5j+2a) mod 11) - 5, ((2j+3a) mod 13) - 6] divided by 3, joins j to j+1 for every j below n_nodes-1 and j to j+3 for every j below n_nodes-3, and holds nodes 0 and n_nodes-1 fixed with prescribed nodal values of zero. Edge k carries the coefficient blocks I + c outer(w, w) with w the normalisation of [((k+1) mod 3)+1, ((k+2) mod 4)+1, ((k+3) mod 5)+1] and c equal to 0.6, 0.9, 0.4 and 0.7 for the four blocks in the order used by the earlier sub-problems. The three configurations are (n_nodes, a) = (7, 3), (6, 5) and (8, 2), all at polynomial degree 3, stabilisation 1.0 and 24 steps of size 0.02*dt_scale. The initial primal field on edge k is the L2 projection onto the polynomial space of the function whose component c at arclength x is sin((c+1)x/2 + 0.3(k+1))/(c+2), and the initial velocity field is the projection of cos((c+2)x/3 + 0.2(k+1))/(c+3); the initial auxiliary field is obtained from that velocity as the source's first-order formulation prescribes, and the initial dual and hybrid states are NOT free: the source fixes them by its initial constitutive relation together with its nodal condition, and that system has to be solved before the first step. Each row has SEVEN entries: [initial discrete energy, final discrete energy, Euclidean norm of the final free-node hybrid vector, sum over free nodes of the norm of the displacement half of each nodal value, sum over free nodes of the norm of the full nodal value, and then the two constants of the source's spectral equivalence between the condensed node operator and a combination of the two graph forms of the previous sub-problem, ordered SMALLEST FIRST. Those two constants are the extremal generalised eigenvalues of the condensed operator with respect to that combination; the source scales one of the two graph forms by a power of the time step before combining them, and which form and which power is the source's convention, so recover it from the paper. Evaluate the condensed operator at the declared time step with vanishing right-hand sides. Raise ValueError unless dt_scale is positive and finite. Assemble by calling the earlier sub-problem functions, every one of them.

```python
import numpy as np


def beam_network_audit(dt_scale):
    """dt_scale: positive multiplier on the declared time step. Returns (3, 7)
    float64 with one row per configuration as described above."""
    return np.zeros((3, 7))
```
