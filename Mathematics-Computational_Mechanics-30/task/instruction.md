# Mathematics-Computational_Mechanics-30

## Background

Peridynamics is a nonlocal reformulation of solid mechanics in which each material point
interacts with every point inside a ball of radius delta, the horizon, through bonds. In the
bond-based variant the force in a bond depends only on that bond's own stretch, and the
discrete equations at a node are sums over its family, the set of neighbours whose cells
fall inside the horizon, with each neighbour carrying a quadrature volume. The nonlocal
analogues of the gradient and of the strain energy are likewise sums over the family,
weighted by a kernel that decays with bond length as an inverse power of the distance.

When a node sits within one horizon of a free surface, part of its ball lies outside the
body and the family is truncated. The truncated sums no longer reproduce the values an
untruncated neighbourhood would give, so the effective stiffness near boundaries and corners
is reduced. This surface effect is a discretization artefact rather than a physical
property, and a long line of corrections exists for it, from volume and fictitious-node
methods to energy-based amplification of individual bonds.

The correction studied here is different in kind. Every node is given a single scalar
influence weight, and the weights are chosen so that a small set of discrete operators
evaluated at that node matches a reference in the least-squares sense. Within a
neighbourhood the weight field is represented in a polynomial basis, so the unknowns are the
polynomial coefficients rather than one number per neighbour, and the fitted weight at the
node itself is read off from that polynomial. The construction is purely geometric: it
depends on the lattice and on the kernel, not on the material model or the loading, and
the weights are computed once, before any load is applied.

The task uses a uniform square lattice with a horizon of four spacings, a complete quadratic
basis in coordinates scaled by the horizon, and the bond-based kernel exponent q = 3. The
quantity reported is the optimized weight at the corner node of the lattice, where the
family is cut down to a single quadrant. Standard numpy linear algebra suffices; no mesh
library is needed.

## Problem

Bond-based peridynamics replaces the stress divergence of continuum mechanics with a sum
over a neighbourhood of bonds of radius delta. Near a free surface that neighbourhood is
truncated, so the discrete derivative and energy operators at boundary nodes no longer
match their full-neighbourhood limits, and the model softens artificially there. A recent
correction assigns each node a scalar influence weight, lets each bond carry an effective
weight derived from the two nodes it joins, and determines the weights so that, at every
node, the discrete operators reproduce a reference as closely as possible in the
least-squares sense.

Adopt that correction exactly as the source prescribes it: how the weight field is
represented within a neighbourhood, how the fitted weight enters each operator row, what
reference the operators are matched against and how that reference is evaluated, how the
matching conditions are stacked and scaled, how the quadrature volume of each neighbour is
assigned, and how the per-node fits are turned into the nodal weights. Represent the
weight field in the complete quadratic monomial basis with coordinates scaled by the
horizon, use the same monomials as test functions, take the bond-based kernel exponent
q = 3, use the Euclidean norm wherever a row of a linear system is normalized, and solve
every least-squares problem without Tikhonov regularization.

Work on a uniform square lattice of 21 by 11 nodes with spacing dx = 0.0625, horizon
delta = 4 dx, and the family of a node being every other node at distance at most delta,
nodes at exactly distance delta included. Consider the node at the origin corner, which
sees only the quadrant of its neighbourhood that lies inside the body.

Report the optimized influence weight of that corner node. Report, as evidence that the chain was
executed: the number of neighbours in the corner node's family; the full-ball x-derivative target and the x-lifted energy target for the
linear test function x/delta; and the optimized weights of the midpoint node of the long free edge and of a fully surrounded
interior node. State whether the corner weight depends on the lattice spacing and why.

State the conventions you adopted and justify each from the source, and state what, according to
the source, the fitted weights correct and what they leave unchanged.

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

neighbor_family

Goal
----
Return the discrete family of node i on a uniform lattice of spacing dx: every other node whose distance from node i is positive and at most delta, nodes at exactly distance delta included, decided exactly (on the integer lattice offsets, not on rounded floating-point distances). Return an (n_F, 5) array whose columns are the neighbour's node index (stored as a float), the bond vector components xi_x and xi_y, the bond length, and the quadrature volume V_j assigned to that neighbour. Assign the volume the way the source's numerical study does on a uniform grid; note that it names its quadrature choice. Order rows by increasing node index. Raise ValueError if coords is not an (N, 2) array with N >= 2, if i is not a valid index, if delta or dx is not positive, if coords do not lie on a lattice of spacing dx, or if the node has no neighbours inside the horizon.

```python
import numpy as np


def neighbor_family(coords, i, delta, dx):
    """ndarray of float64 with shape (n_F, 5): columns node index, xi_x, xi_y, bond length, V_j."""
    return np.zeros((1, 5), dtype=np.float64)
```

### Step 2

scaled_monomial_basis

Goal
----
Evaluate the complete quadratic monomial basis of Eq (49) at each row of the (n, 2) array xi of bond vectors, with both coordinates scaled by the horizon delta before any power is taken. Return an (n, 6) array in the source's ordering: constant, x, y, x squared, xy, y squared. Raise ValueError if xi is not a non-empty (n, 2) array or if delta is not positive.

```python
import numpy as np


def scaled_monomial_basis(xi, delta):
    """ndarray of float64 with shape (n, 6)."""
    return np.zeros((np.shape(xi)[0], 6), dtype=np.float64)
```

### Step 3

derivative_block

Goal
----
Assemble the derivative-operator matrices of Eqs (56) and (57) for the x and y directions and stack them as a (2 n_T, n_p) array, x rows first, where the n_T = n_p - 1 test functions are the non-constant monomials of the basis in basis order (the constant test gives an identically zero row and is omitted). family is the (n_F, 5) array of step 1, basis the (n_F, n_p) basis evaluated at the neighbours, basis0 the basis evaluated at the node itself. Each row is a sum over the family of the source's per-bond basis factor, times the bond component divided by the bond length to the power q, times the test function increment between neighbour and node, times V_j. Raise ValueError if family is not (n_F, 5), if basis is not (n_F, n_p) with basis0 of length n_p, or if delta or q is not positive.

```python
import numpy as np


def derivative_block(family, basis, basis0, delta, q):
    """ndarray of float64 with shape (2 n_T, n_p)."""
    return np.zeros((2 * (np.shape(basis)[1] - 1), np.shape(basis)[1]), dtype=np.float64)
```

### Step 4

energy_block

Goal
----
Assemble the energy-operator matrix of Eq (59) for the vector test functions formed by multiplying each non-constant scalar test by the two Cartesian unit vectors, and return it as a (2 n_T, n_p) array, x-lifted tests first, tests in basis order. Each row is a sum over the family of the source's per-bond basis factor, times the square of the test increment times the bond component, divided by the bond length to the power q, times V_j; leave out any constant prefactor common to a row and its target. Raise ValueError under the same conditions as the derivative block.

```python
import numpy as np


def energy_block(family, basis, basis0, delta, q):
    """ndarray of float64 with shape (2 n_T, n_p)."""
    return np.zeros((2 * (np.shape(basis)[1] - 1), np.shape(basis)[1]), dtype=np.float64)
```

### Step 5

full_ball_targets

Goal
----
Return the target vector the source fits against: the derivative operators of Eq (60) and the energy operator, each applied to the five non-constant scaled monomial test functions at the node, evaluated over the reference neighbourhood the source prescribes for the targets, and concatenated as x-derivative targets, y-derivative targets, x-lifted energy targets, y-lifted energy targets, in that order, giving a length-20 array. Evaluate every entry exactly, with the same prefactor convention as the energy block. Raise ValueError if delta or q is not positive, or if an entry is not finite.

```python
import numpy as np


def full_ball_targets(delta, q):
    """ndarray of float64 with shape (20,)."""
    return np.zeros(20, dtype=np.float64)
```

### Step 6

local_coupling_row

Goal
----
Solve the regularized least-squares fit of Eqs (64) to (68) at one node and return the relation it yields between this node's weight and its neighbours' weights. Stack the collocation block (the basis at the neighbours, whose targets are the neighbours' own weights and are therefore left symbolic), the derivative block and the energy block into one matrix; scale every row of that matrix, together with its target, to unit Euclidean norm; form the Tikhonov-regularized generalized inverse with parameter lam; and read off the row that gives the weight field's value at the node. Return a length n_F + 1 array whose first entry is the contribution of the supplied targets and whose remaining entries are the coefficients multiplying the weights of the n_F neighbours, in family order. Raise ValueError if the blocks are not 2-D with a common basis dimension, if the target length does not match the derivative and energy rows, or if lam is negative.

```python
import numpy as np


def local_coupling_row(basis, m_deriv, m_energy, targets, lam):
    """ndarray of float64 with shape (n_F + 1,): the target term, then one coefficient per neighbour."""
    return np.zeros(np.shape(basis)[0] + 1, dtype=np.float64)
```

### Step 7

corner_influence_weight

Goal
----
Orchestrator. Build the nx-by-ny uniform lattice of spacing dx with its origin node at index 0 and the horizon set to m times dx. For every node of the lattice build its family, evaluate the scaled basis at its neighbours and at the node, assemble the derivative and energy blocks with exponent q, and obtain its coupling row against the target vector with parameter lam; assemble those rows into the global linear system of Eq (69) that determines all nodal weights at once, solve it directly, and return the optimized influence weight of the origin corner node. Call the earlier step functions rather than reimplementing any of them. Raise ValueError if nx or ny is not an integer of at least 2, if dx or q is not positive, if lam is negative, if m is not a positive integer, or if the horizon exceeds what the grid extent can hold.

```python
import numpy as np


def corner_influence_weight(nx, ny, dx, m, q, lam):
    """float, the optimized influence weight of the corner node."""
    return 0.0
```
