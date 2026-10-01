# Mathematics-Computational_Mechanics-5

## Background

Simulating a fault or a fracture inside a geomechanical model means enforcing contact across a surface whose two sides are meshed independently. The mortar method does this weakly, with a set of Lagrange multipliers on one side of the interface standing for the interface tractions. Which pair of discrete spaces to use for the displacements and the multipliers is not a free choice: the pair has to satisfy an inf-sup condition for the resulting saddle point problem to be solvable at all.

Pairing trilinear displacements with vector piecewise constant multipliers on the non-mortar side is attractive for practical reasons that matter in geological applications. Dirichlet conditions can be imposed on the boundary of the fault without locally modifying the multiplier basis, and the solvability problems that arise where several interfaces intersect are disposed of automatically. That pair, however, is not inf-sup stable, and the Schur complement of the resulting system acquires spurious modes that show up as oscillating traction profiles.

The source of this task repairs the pair by adding a term built on the jump of the multiplier across the interior edges of the non-mortar interface, in the spirit of pressure-jump stabilization for the Stokes problem. The contribution of that term must be scaled: too small and the spurious modes survive, too large and the constraint itself is distorted. The source derives the scaling from a local analysis on a patch of cells around each interface node, extending a macroelement argument from the conforming case to the non-conforming one, so that the strength of the stabilisation follows from the discretisation itself and no user selected parameter enters anywhere.

This task implements that construction on a fixed pair of blocks meeting at a planar non-conforming interface, and audits it the two ways the source does: by tracking the inf-sup constant as the interface is refined at a fixed grid ratio, and by checking that a constant traction still crosses the interface exactly.

## Problem

Enforcing contact across a fault whose two sides are meshed independently is usually done with a mortar method and a set of Lagrange multipliers standing for the interface tractions. Pairing piecewise linear displacements with piecewise constant multipliers on the non-mortar side is attractive in practice, because Dirichlet data can be applied on the fault boundary without touching the multiplier basis and cross points of intersecting interfaces need no special treatment, but that pair violates the inf-sup condition and the resulting saddle point problem is unstable. A recent formulation repairs it by adding a term built on the jump of the multiplier across the interior edges of the non-mortar interface, and derives the scaling of that term from a local analysis on a patch around each interface node, so that no user selected parameter is introduced anywhere. Your job is to implement that construction exactly as the source specifies it and to audit it on the fixed configuration below.

The load-bearing choices are the source's, and you are expected to recover them from the paper: which part of the gathered patch stiffness is kept when the local approximation is formed, how the tensor that scales an edge contribution is built from the two faces meeting at that edge, the structure with which an edge contribution enters the global operator, what happens when the sweep reaches the same edge from more than one patch, the mesh size and the multiplier metric that appear in the source's matrix characterisation of the inf-sup constant, the sign with which the stabilising operator enters the saddle point system, and the fact that the operator coupling the two interface grids has to be integrated as the source intends rather than approximated. The audit is calibrated so that departing from any one of these changes the reported number by far more than the grading tolerance.

Geometry and discretisation, all quantities dimensionless. Two blocks meet at the planar interface z = 0. Block 1 occupies the unit square in x and y and spans z from -depth to 0; block 2 occupies the same square and spans z from 0 to +depth. Each block is meshed with n by n trilinear hexahedra in plan and a single element through its thickness. Within a block, number the nodes with x fastest, then y, then z, and give each node three consecutive degrees of freedom ordered x, y, z. Block 1 is the non-mortar side: the multipliers are vector valued and piecewise constant, one per interface face of block 1, and those faces are numbered with x fastest. The interface is planar with a constant normal along z, so the projection the mortar method needs reduces to the in-plane identity. The material is isotropic linear elastic with Young's modulus E = 1 and Poisson ratio nu = 0.

Three interface grid pairs are audited, given as (n for block 1, n for block 2): (4, 2), (6, 3) and (8, 4). Every pair holds the ratio of the two interface mesh sizes fixed, and the non-mortar side is always the finer of the two.

The patch used to derive the local scaling is defined as follows, and you should take this definition as given rather than inferring it. For an internal node of the mortar interface grid, the patch consists of the mortar interface faces that touch that node, together with every non-mortar interface face whose area overlaps the region those mortar faces cover. An interior edge of the patch is one shared by two non-mortar faces that both belong to the patch and that are neighbours across a coordinate direction. Within a patch, order the displacement degrees of freedom with the non-mortar side first and then the mortar side, each in ascending global index.

Two boundary condition settings are used. In the clamped setting, every degree of freedom on the far face of block 1 at z = -depth and on the far face of block 2 at z = +depth is fixed, and the inf-sup constant is evaluated there. In the loaded setting, only the far face of block 1 is fixed, block 2 carries no displacement constraint, and a surface traction is applied on the top face of block 2. Each top face contributes to each of its four corner nodes a load equal to one quarter of h squared times the traction evaluated at that node, h being the in-plane element size of block 2. The graded traction is (0.30 y, 0.20 x, -(1 + 0.5 x + 0.25 y)); the uniform traction is (0, 0, -1).

For each of the three grid pairs, assemble both blocks, both interface operators and the stabilising operator, then record a row of ten numbers: the inf-sup constant of the clamped setting computed without the stabilising term, the same constant computed with it, the Frobenius norm of the stabilising operator, and then from the loaded setting the mean, the smallest and the largest normal multiplier under the graded traction, the mean magnitude of the tangential multiplier under the graded traction, the largest departure of the normal multiplier from its own mean under the uniform traction, and finally two diagnostics of the patch built around the first internal node of the mortar interface grid, namely the number of non-mortar faces it contains and the Frobenius norm of the local approximation formed on it.

Evaluate the audit at depth = 1.0. All quantities are float64, finite and deterministic: two runs on identical input must agree exactly. As the final answer, report the sum over the three grid pairs of the inf-sup constant computed with the stabilising term, to six significant figures.

In your reasoning, state the conventions you adopted and justify each of them from the source. Report as numerical results the three inf-sup constants computed without the stabilising term, the Frobenius norm of the stabilising operator for the coarsest grid pair, the largest normal multiplier of the finest grid pair under the graded traction, the largest departure of the normal multiplier from its mean under the uniform traction, and the Frobenius norm of the local approximation on the reference patch of the coarsest grid pair.

The quantities listed above are the reported result, not intermediate bulk output. Give them
as a compact list of labelled values inside the reasoning section, and state each convention in a
sentence or two. A short list of exactly those values is not the kind of per-iteration or
per-candidate output the format note below asks you to leave out, and a response that reports them
compactly is both complete and within the length the note asks for.

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

hex_stiffness

Goal
----
Build the element stiffness matrix of one trilinear hexahedron of the displacement space the source uses, Eq. (14a), for an isotropic linear elastic solid. The element is a rectangular box aligned with the axes. Use the tensor-product Gauss rule that integrates this element exactly, and order the twenty four degrees of freedom node by node with the three components contiguous within each node. Take the eight local nodes in the counter-clockwise order (-1,-1,-1), (1,-1,-1), (1,1,-1), (-1,1,-1) around the lower face, followed by the same four corners on the upper face. That local ordering belongs to the element itself and is not the lexicographic ordering used to number the nodes of a block.

```python
import numpy as np


def hex_stiffness(E, nu, hx, hy, hz):
    """Build the element stiffness matrix of one trilinear hexahedron of the displacement space
    the source uses, Eq. (14a), for an isotropic linear elastic solid. A (24, 24) float64
    symmetric element stiffness matrix."""
    return np.zeros((24, 24))
```

### Step 2

block_stiffness

Goal
----
Assemble the global stiffness matrix of one of the two subdomains. The block occupies the unit square in x and y, is meshed with n by n elements in plan and a single element through the thickness, and every element is the box whose stiffness matrix is supplied. Number the nodes with x fastest, then y, then z, and give each node three consecutive degrees of freedom. When an element is gathered into the global matrix its eight nodes are taken in the element's own counter-clockwise local order, not in the global lexicographic order.

```python
import numpy as np


def block_stiffness(n, Ke):
    """Assemble the global stiffness matrix of one of the two subdomains. A (3*N, 3*N) float64
    symmetric stiffness matrix, N being the number of nodes of the block."""
    return np.zeros((6, 6))
```

### Step 3

mortar_mass

Goal
----
Assemble the first of the two interface operators of Eq. (11), pairing every multiplier basis function with the displacement basis functions of the non-mortar side over the interface. The multipliers are the vector piecewise constants of Eq. (14b), one per non-mortar interface face; the displacements are the traces of the trilinear block functions. Each of the three components pairs only with its own component.

```python
import numpy as np


def mortar_mass(n1):
    """Assemble the first of the two interface operators of Eq. (11), pairing every multiplier
    basis function with the displacement basis functions of the non-mortar side over the
    interface. A (3*n1*n1, 3*N1) float64 array, N1 being the number of nodes of the non-
    mortar block."""
    return np.zeros((3, 3))
```

### Step 4

mortar_coupling

Goal
----
Assemble the second interface operator of Eq. (11), pairing every multiplier with the displacement basis functions of the mortar side after they have been mapped onto the non-mortar interface. The two interface grids do not match, so the integrand is discontinuous over a non-mortar face; the source names this the major algorithmic challenge of the method and it must be integrated as the source intends rather than approximated.

```python
import numpy as np


def mortar_coupling(n1, n2):
    """Assemble the second interface operator of Eq. (11), pairing every multiplier with the
    displacement basis functions of the mortar side after they have been mapped onto the
    non-mortar interface. A (3*n1*n1, 3*N2) float64 array, N2 being the number of nodes of
    the mortar block."""
    return np.zeros((3, 3))
```

### Step 5

macroelement_masks

Goal
----
Identify the local patch the source builds around one internal node of the mortar interface grid, as the configuration defines it: the mortar faces touching that node, together with every non-mortar face whose area overlaps the region they cover. Return the membership of both sets as a single indicator vector, mortar faces after non-mortar faces, each face numbered with the x index fastest.

```python
import numpy as np


def macroelement_masks(n1, n2, i2, j2):
    """Identify the local patch the source builds around one internal node of the mortar
    interface grid, as the configuration defines it: the mortar faces touching that node,
    together with every non-mortar face whose area overlaps the region they cover. A (n1*n1
    + n2*n2,) float64 indicator vector holding 1.0 for a member face and 0.0 otherwise."""
    return np.zeros(2)
```

### Step 6

local_scaling

Goal
----
Form the local approximation of Eq. (19) on the patch of one mortar node. Gather the two blocks' stiffness entries for the patch displacement degrees of freedom and the two interface operators for its multiplier degrees of freedom, then combine them into an operator on the patch multiplier space. The source states exactly which part of the gathered stiffness it retains and why, immediately below the equation, and that choice is what fixes the magnitude of everything downstream.

```python
import numpy as np


def local_scaling(n1, n2, i2, j2, A1, A2, D, M):
    """Form the local approximation of Eq. (19) on the patch of one mortar node. A (3*m, 3*m)
    float64 symmetric array on the patch multiplier space, m being the number of non-mortar
    faces in the patch."""
    return np.zeros((3, 3))
```

### Step 7

stabilization_matrix

Goal
----
Assemble the global stabilising operator of Eq. (16) by the procedure of Algorithm 1. Sweep the internal nodes of the mortar interface grid; on each patch form the local approximation and derive from it the second order tensor that scales the contribution of every internal edge of the patch, per Eq. (20); then place that contribution into the global operator through the jump structure the bilinear form implies. Take an internal edge to be one shared by two non-mortar faces that both belong to the patch and that are neighbours across a coordinate direction. The sweep visits some edges more than once, and Algorithm 1 says what to do about that.

```python
import numpy as np


def stabilization_matrix(n1, n2, A1, A2, D, M):
    """Assemble the global stabilising operator of Eq. (16) by the procedure of Algorithm 1. A
    (3*n1*n1, 3*n1*n1) float64 symmetric positive semi-definite array."""
    return np.zeros((3, 3))
```

### Step 8

infsup_constant

Goal
----
Evaluate the discrete inf-sup constant of the saddle point problem through the characterisation the source derives, Eq. (24). Build the constrained operator on the multiplier space from the given stiffness and constraint operators, add the given stabilising operator to it, and extract the constant from the extreme generalised eigenvalue against the given scaled multiplier metric. Return the constant itself, not its square, and never a negative number.

```python
import numpy as np


def infsup_constant(A, B, Q, H, h):
    """Evaluate the discrete inf-sup constant of the saddle point problem through the
    characterisation the source derives, Eq. (24). A Python float, the inf-sup constant."""
    return 0.0
```

### Step 9

interface_tractions

Goal
----
Solve the stabilised saddle point system of Eq. (17) for the interface multipliers under the given load, and return only the multipliers. The stabilising operator enters the constraint block of the system, and it enters with the sign that Eq. (17) gives it. There is no load on the constraint equations.

```python
import numpy as np


def interface_tractions(A, B, H, f):
    """Solve the stabilised saddle point system of Eq. (17) for the interface multipliers under
    the given load, and return only the multipliers. A (3*n1*n1,) float64 array of interface
    multiplier values, three components per non-mortar face."""
    return np.zeros(3)
```

### Step 10

mortar_audit

Goal
----
Run the whole audit over the three interface grid pairs of the configuration. For each pair assemble both blocks, both interface operators and the stabilising operator, then evaluate the inf-sup constant twice, once without the stabilising term and once with it, in the clamped setting of the configuration; and solve the loaded setting twice, once under the graded surface traction and once under the uniform one. Record one row per grid pair. The mesh size and the multiplier metric that enter the inf-sup characterisation are the ones the source's own derivation uses.

```python
import numpy as np


def mortar_audit(depth):
    """Run the whole audit over the three interface grid pairs of the configuration. A (3, 10)
    float64 array, one row per grid pair, with columns [inf-sup constant without
    stabilisation, inf-sup constant with stabilisation, Frobenius norm of the stabilising
    operator, mean normal multiplier under the graded load, smallest normal multiplier,
    largest normal multiplier, mean tangential multiplier magnitude, largest departure of
    the normal multiplier from its mean under the uniform load, number of non-mortar faces
    in the patch of the first internal mortar node, Frobenius norm of the local
    approximation formed on that same patch]."""
    return np.zeros((3, 10))
```
