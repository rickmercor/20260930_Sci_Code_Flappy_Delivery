# Material_Science-Molecular_Modeling-14

## Background

Twisted trilayer graphene with two independent twist angles has no supercell: its tight-binding model couples momenta on a four-dimensional reciprocal lattice. Transforming the model exactly into momentum space and truncating the scattering channels around the Dirac cones turns an infinite incommensurate problem into a small dense matrix whose spectral density is read off with a damped Chebyshev expansion. On a fixed small trilayer family the whole construction can be audited: the degree-of-freedom counts witness the truncation criteria and the momentum local density of states depends on every convention in the pipeline.

## Problem

Twisted trilayer graphene with two independent small twist angles is double-incommensurate: no supercell exists, and its tight-binding model couples momenta on a four-dimensional reciprocal lattice. The source paper's momentum space algorithm makes this computable: it transforms the tight-binding model exactly into reciprocal space, cuts the four-dimensional lattice down with a two-stage truncation adapted to the Dirac cones, and evaluates the momentum local density of states with a damped Chebyshev expansion. Your job is to implement this pipeline exactly as the source specifies it and audit it on the fixed trilayer family below. The load-bearing algorithmic choices, the base lattice conventions of the monolayer, the reduction map applied to shifted momenta and the choice of reference Dirac point in the window criterion, which distances the two truncations test, the Bloch phase convention of the intralayer sum and its real-space truncation, which layer pairs couple and under which matching condition on the third layer's reciprocal vector, at which combined momentum the interlayer coupling is evaluated and which normalization, phase factors and smooth taper multiply it, and the exact damping coefficients and reconstruction of the kernel polynomial approximation, are the source's; recover them from the paper. The problem is calibrated so that departing from those choices changes the reported numbers by amounts far beyond the grading tolerance.

The lattice constant is a = 1.42 sqrt(3), and the two configurations use angle triples (-0.06, 0, 0.11) and (-0.09, 0, 0.07) radians with per-configuration truncation radii (W, L) = (1.15, 5.0) and (1.05, 4.6). The base momentum is the middle layer's first Dirac point in configuration 1 and its second Dirac point in configuration 2, plus the offset (0.03, 0.02) in both. Integer reciprocal coordinates are scanned over [-3, 3]^2. The declared intralayer hopping functions, indexed by integer Bravais steps (n1, n2) from sublattice A to B or within a sublattice, are: t1 = -1 on the A-to-B steps (0,0), (-1,0), (-1,1); t2 = 0.15 on the same-sublattice steps (1,0), (0,1), (1,-1) and their negatives; t3 = 0.08 on the A-to-B steps (0,1), (-2,1), (0,-1); and t4 = 0.05 on the A-to-B steps (2,-1), (-2,0), (1,1); each A-to-B entry carries its Hermitian partner. The real-space truncation radius is tau = 3.0 with transition width delta = 0.6. The interlayer coupling function is h(xi) = 0.045 exp(-0.1 |xi|^2). The kernel polynomial order is P = 64 with spectral scaling s = 0.25, and the audit energies are (-0.5, 0.0, 0.5).

Implement eight functions. layer_geometry(thetas) returns the (3, 14) packed geometry: for each layer the counterclockwise rotation of the source's base Bravais matrix, reciprocal matrix, both Dirac points and B-sublattice offset, packed as [A(4), B(4), K(2), K'(2), tau_B(2)] with matrices flattened row-major; it raises ValueError on invalid input. cell_reduce(X, B) reduces a batch of momenta into a layer's reciprocal unit cell exactly by the source's reduction map. wl_dof(q, geom, W, L, nmax) enumerates the surviving reciprocal degrees of freedom as (n, 5) int64 rows [j, nk1, nk2, nl1, nl2], pairing each layer with the other two in increasing layer order, applying the source's L criterion to the reciprocal vectors and the source's window criterion to the prepared shifted momentum against the reference Dirac point chosen as the source prescribes, rows ordered by layer then ascending lexicographic integer coordinates; it raises ValueError on nonpositive radii. bump_gtau(r, tau, delta) evaluates the source's smoothly truncated compactly supported bump on an array of radii, including both plateau regions and the source's transition profile. intralayer_block(qeff, Aj, tauB, tau) returns the (2, 2, 2) Hermitian intralayer Bloch block (real part in [..., 0], imaginary in [..., 1]) from the declared hopping functions with the source's phase convention and real-space truncation. assemble_hamiltonian(q, dofs, geom) returns the (2n, 2n, 2) Hermitian reciprocal Hamiltonian, two sublattice rows per degree of freedom in order: intralayer blocks on the diagonal at the shifted momenta, and interlayer couplings exactly as the source specifies them. kpm_ldos(Hri, dofs, Elist, P, s) returns the (nE,) momentum local density of states at the base momentum via the source's damped Chebyshev approximation on the zero-reciprocal-vector entries, averaged as the source defines it; it raises ValueError on an invalid order. ttg_audit(P), the final step, must be assembled by calling the earlier functions: it builds both declared configurations, runs the pipeline on each, and returns a float64 array (2, 7) with columns: the degree-of-freedom count; the Hamiltonian dimension; the Hermiticity residual of the assembled matrix; the momentum LDoS at the three declared energies; and the largest eigenvalue.

Evaluate ttg_audit with P = 64. All outputs are float64, finite, and deterministic: two runs on identical inputs must agree exactly. As the final answer, report the sum over the two configurations of the momentum LDoS at energy zero (the fifth column) to six significant figures.

Output Format Requirements:
Scientific reasoning wrapped in <reasoning>...</reasoning> tags. Include enough intermediate calculations to justify the deterministic computation, without turning the response into a general pipeline summary. A single final numeric answer wrapped in <final_answer>...</final_answer> tags.
Rules:
- The tags are required. Do not omit them or leave them empty.
- The tagged value must be a finite decimal (examples: 0.4847, 2.13, 1.05). Not NaN, not Inf, not a fraction string, not a vector, not prose.
- Put only that one number between the tags. No units, no words, no extra lines.
Keep <reasoning> short (a few hundred words). As numerical results, report only the audit row entries declared above for each of the two configurations, plus the final summed value.
Do not paste the degree-of-freedom tables, Hamiltonian blocks, or per-moment Chebyshev data. The two-by-two base lattice matrices may be quoted.

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

layer_geometry

Goal
----
Build the rotated geometry of the three graphene layers for a given angle triple: per layer, the Bravais matrix, the reciprocal matrix, the two Dirac points and the B-sublattice offset, packed row-wise as [A (4, row-major), B (4, row-major), K (2), K' (2), tau_B (2)], using the source's base lattice conventions with lattice constant a = 1.42 sqrt(3).

```python
import numpy as np


def layer_geometry(thetas):
    """thetas: (3,) rotation angles in radians. Returns (3, 14) float64: for
    each layer j the counterclockwise rotation by thetas[j] applied to the
    source's base Bravais matrix, reciprocal matrix, both Dirac points and
    the B-sublattice offset, packed as [A(4), B(4), K(2), K'(2), tau_B(2)]
    with matrices flattened row-major. Raises ValueError on a wrong shape or
    nonfinite input."""
    return np.zeros((3, 14))
```

### Step 2

cell_reduce

Goal
----
Reduce a batch of momentum vectors into the reciprocal unit cell of a layer exactly by the source's reduction map for its truncation criterion.

```python
import numpy as np


def cell_reduce(X, B):
    """X: (n, 2) momentum vectors (a single vector may be passed as (2,));
    B: the layer reciprocal matrix, passed flattened (4,) row-major or as
    (2, 2). Returns (n, 2) float64: each vector reduced into the layer's
    reciprocal unit cell exactly by the source's reduction map."""
    return np.atleast_2d(np.asarray(X, dtype=np.float64))
```

### Step 3

wl_dof

Goal
----
Enumerate the reciprocal degrees of freedom that survive the source's W and L truncations for a base momentum q: for each layer j the pairs of reciprocal vectors of the other two layers (in increasing layer order), with integer coordinates scanned over [-nmax, nmax]^2 in ascending lexicographic order, layer-major. Which distances the two criteria test, how the shifted momentum is prepared before the W test, and how the reference Dirac point is chosen, follow the source.

```python
import numpy as np


def wl_dof(q, geom, W, L, nmax):
    """q: (2,) base momentum; geom: (3, 14) packed layer geometry; W, L:
    truncation radii; nmax: integer scan bound. Returns (n_dof, 5) int64
    rows [j, nk1, nk2, nl1, nl2]: for layer j the integer coordinates of the
    reciprocal vectors of the other two layers in increasing layer order,
    keeping exactly the pairs the source's L and W criteria admit, rows
    ordered by layer then ascending lexicographic integer coordinates.
    Raises ValueError on nonpositive radii."""
    return np.zeros((1, 5), dtype=np.int64)
```

### Step 4

bump_gtau

Goal
----
Evaluate the source's smoothly truncated compactly supported bump function of width delta and cutoff tau on an array of radii, exactly as the source defines it, including both plateau regions.

```python
import numpy as np


def bump_gtau(r, tau, delta):
    """r: array of nonnegative radii; tau: cutoff; delta: transition width.
    Returns an array of the same shape: the source's smoothly truncated bump
    at each radius, equal to one deep inside the support, zero at and beyond
    the cutoff, with the source's transition profile in between."""
    return np.ones_like(np.asarray(r, dtype=np.float64))
```

### Step 5

intralayer_block

Goal
----
Evaluate the 2x2 intralayer Bloch block of one layer at an effective momentum, from the declared hopping functions: amplitude t1 = -1 on the Bravais steps (0,0), (-1,0), (-1,1) from sublattice A to B; t2 = 0.15 on the steps (1,0), (0,1), (1,-1) and their negatives within each sublattice; t3 = 0.08 on the A-to-B steps (0,1), (-2,1), (0,-1); t4 = 0.05 on the A-to-B steps (2,-1), (-2,0), (1,1); each with its Hermitian partner. The Bloch phase convention and the real-space truncation of the hop sum at radius tau follow the source. The last axis stacks real and imaginary parts.

```python
import numpy as np


def intralayer_block(qeff, Aj, tauB, tau):
    """qeff: (2,) effective momentum; Aj: layer Bravais matrix, flattened
    (4,) row-major or (2, 2); tauB: (2,) B-sublattice offset; tau: real-space
    truncation radius. Returns (2, 2, 2) float64: the Hermitian intralayer
    Bloch block from the declared hopping shells, phases and truncation
    exactly as the source specifies, real part in [..., 0] and imaginary part
    in [..., 1]."""
    return np.zeros((2, 2, 2))
```

### Step 6

assemble_hamiltonian

Goal
----
Assemble the Hermitian reciprocal Hamiltonian on the truncated degrees of freedom: intralayer blocks on the diagonal at the shifted momenta, and interlayer couplings only between the layer pairs the source admits, subject to the source's matching condition on the third layer's reciprocal vector, with the source's normalization and sublattice phase factors, the declared interlayer coupling h(xi) = 0.045 exp(-0.1 |xi|^2) evaluated at the momentum the source prescribes, and tapered by the source's bump with tau = 3.0, delta = 0.6. The last axis stacks real and imaginary parts.

```python
import numpy as np


def assemble_hamiltonian(q, dofs, geom):
    """q: (2,) base momentum; dofs: (n, 5) int64 degrees of freedom from the
    truncation step; geom: (3, 14) packed geometry. Returns (2n, 2n, 2)
    float64: the Hermitian reciprocal Hamiltonian with two sublattice rows
    per degree of freedom in order, assembled exactly as the source
    specifies, real part in [..., 0] and imaginary part in [..., 1]."""
    return np.zeros((2 * np.asarray(dofs).shape[0], 2 * np.asarray(dofs).shape[0], 2))
```

### Step 7

kpm_ldos

Goal
----
Evaluate the momentum local density of states at the base momentum through the source's kernel polynomial approximation of the delta function at polynomial order P and scaling s: Chebyshev moments of the scaled Hamiltonian on the zero-reciprocal-vector unit vectors of each layer, damped by the source's kernel coefficients, reconstructed with the source's normalization, and averaged over layers and sublattices as the source defines the momentum LDoS.

```python
import numpy as np


def kpm_ldos(Hri, dofs, Elist, P, s):
    """Hri: (2n, 2n, 2) stacked Hamiltonian; dofs: (n, 5) degrees of freedom;
    Elist: (nE,) energies; P: polynomial order; s: spectral scaling with
    s * spectrum inside (-1, 1). Returns (nE,) float64: the momentum LDoS at
    each energy, computed with the source's damped Chebyshev approximation
    on the zero-reciprocal-vector entries and averaged as the source
    prescribes. Raises ValueError on an invalid order."""
    return np.zeros(np.asarray(Elist).shape[0])
```

### Step 8

ttg_audit

Goal
----
Run the full pipeline on the two declared trilayer configurations and report per configuration: the number of surviving reciprocal degrees of freedom; the Hamiltonian dimension; the Hermiticity residual of the assembled matrix; the momentum LDoS at the three declared energies at polynomial order P; and the largest eigenvalue of the assembled Hamiltonian. Assemble by calling the earlier sub-problem functions.

```python
import numpy as np


def ttg_audit(P):
    """P: Chebyshev polynomial order. Builds the two declared trilayer
    configurations, runs the truncation, assembly and kernel polynomial
    pipeline on each with the declared parameters, and returns a float64
    array (2, 7) with columns: degree-of-freedom count; Hamiltonian
    dimension; Hermiticity residual; the momentum LDoS at the three declared
    energies; and the largest eigenvalue. Assembled by calling the earlier
    sub-problem functions. Raises ValueError on an invalid order."""
    return np.zeros((2, 7))
```
