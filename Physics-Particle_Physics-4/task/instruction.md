# Physics-Particle_Physics-4

## Background

Hamiltonian lattice gauge theory describes spatial gauge fields and their conjugate colour-electric momenta subject to Gauss's law. The source paper removes local gauge redundancy using a maximal tree on an open cubic lattice. Its gauge-fixing construction, boundary derivatives, and remaining physical variables determine the nonlocal interaction to be evaluated here. The reduction atlas records link metadata, oriented incidence, and descendant indicators for that tree. The paper implements the nonlocal inverse by QSVT and digitizes each physical amplitude in the field-amplitude basis; the Chebyshev singlet Green and the Pauli-Z register operators are numerical representations of those constructions. The response matrices express the reduced source in physical-momentum coordinates, while the temporal field describes the dependent gauge component. These arrays are numerical representations chosen for this task. The deterministic fields and residual SU(3) frame specify one numerical source. Applying a common Frobenius normalization preserves its relative colour amplitudes. The reported colour-row energies depend on the chosen frame; their sum is invariant under a common global colour rotation. The benchmark asks for this sum after eliminating the temporal field according to the source paper.

## Problem

Implement the eight ordered Python functions below and use their composed result to report the
nonlocal interaction energy of the source paper's maximal-tree-gauge SU(3) Yang-Mills
Hamiltonian for one fixed reduced non-Abelian Gauss source in a residual global colour frame.

Use a cubic open lattice with L=5, spacing a=0.37, coupling g=0.91, and frame index j=2.
Recover the gauge hierarchy, Gauss kernel, one-sided endpoint rules, and retained physical
variables from the source paper. The fields, normalization, and array conventions below
define this benchmark. Functions taking side_length use a Python int or NumPy integer with
2 <= L <= 5 (bool excluded); lattice spacing must be finite and positive. Flatten
(n1,n2,n3) as `n1+L*(n2+L*n3)` and write `V=L**3`.

Step 01 returns a rooted atlas of shape (V-1, 6+2V) with columns
(family, tail, head, parent_link, head_depth, subtree_size), then V incidence entries
(-1 at tail, +1 at head), then V descendant indicators. Direction codes are 3,2,1.
Root at the origin, orient links away from it, sort families by descending direction
then n1-fastest tail; parent_link is the atlas row entering the tail or -1 at the
origin; head_depth counts links from the origin; subtree_size counts the head and
its descendants.

Step 02 returns the paper's QSVT singlet Green of Chebyshev degree d together
with the field-amplitude digitization of one gauge register: Pauli-Z weights
of A and the real matrices (A, Re Pi, Im Pi). Use d=5, K=4 and Amax=1 when
composing the energy.

Step 04 returns the paper's three positive kernel terms unsummed as (3,V,V) in
direction order (3,2,1); K is their sum. Retain the zero mode. K_plus is the
Moore-Penrose inverse on the site-singlet subspace.

Use the paper's physical forward links: a direction-i tail coordinate runs through
L-2 and transverse coordinates through L-1, after the paper's stratum restrictions,
ordered by increasing direction then n1-fastest tail, with colour r=0,...,7 fastest.
At a physical direction-I tail set X=(n1+0.5)/L, Y=(n2+0.5)/L, Z=(n3+0.5)/L, with
I=0 for n1 and I=1 for n2. For colour r=0,...,7,

    A_I^r = 0.41*sin(pi*((r+1)*X + 0.17*(I+1)*Z))
             + 0.23*cos(pi*(((r mod 3)+1)*Y - 0.13*(I+1)*X)),
    P_I^r = cos(pi*(((r mod 4)+1)*Z + 0.11*(I+1)*Y))
             + 0.37*sin(pi*((r+2)*Y - 0.19*(I+1)*X)).

Let lambda[0],...,lambda[7] be Gell-Mann lambda_1,...,lambda_8, with
Tr(lambda[r] lambda[s])=2 delta_rs and
[lambda[r],lambda[s]]=2i f[r,s,t] lambda[t]. For frame j start from
theta=(0.29,-0.17,0.23,0.11,-0.19,0.13,-0.07,0.31), roll it right by `2*j`
entries, and multiply odd 0-based entries by `(-1)**j`. Set
H=sum_r theta_r lambda[r]/2, U=exp(iH) from the Hermitian spectrum, and
R[a,b]=Re Tr(lambda[a] U lambda[b] U^dagger)/2; replace A and P by R@A and R@P,
including at the fixed j=2.

Step 03 returns p together with the paper's reduced Gauss response. Source rows are
`b*V+n` and momentum columns `8*ell+r`. M0 and M1 are the Abelian and connection
pieces of the endnote's left covariant divergence of the physical momenta
(Equations (34)-(35)), so the unprojected source is `(M0+g*M1)@p`.
Step 08 reshapes that source to eight colour rows, subtracts each row mean, and
applies one common scale s so the Frobenius norm is `sqrt(sum_b(1+0.1*b)**2)`.
Step 07 takes a kernel and a charge array of shape (C,N) with N=V and returns the
zero-mean temporal field in the same layout. Return unrounded values; apply
round(value,6) only when writing final_answer.

In reasoning, give a short certificate stating the gauge hierarchy, the supported kernel terms, the one-sided endpoint rule and endpoint diagonal, and the retained physical families. Give each fact with its supporting equation, figure, or table location in the source paper. Identify the Hamiltonian term being evaluated, citing its equation and prefactor. Report the colour-row energies E_0 and E_7 in frame j=2 and their unrounded total E, each to at least ten significant figures (relative tolerance 1e-9). Confirm E independently through the Green quadratic, the temporal inner product, the tree-flow energy, and the physical-momentum quadratic; each equals E. These are evaluations of the paper's nonlocal interaction on the normalized source. Round the final answer to six decimal places using round(E,6). Omit auxiliary counts, spectra, maxima, full matrices, link lists, and field arrays.

Implement:

1. build_maximal_tree_reduction_atlas(side_length)
2. assemble_qsvt_field_amplitude_objects(side_length, lattice_spacing,
   polynomial_degree, n_qubits, a_max, tree_atlas)
3. build_reduced_nonabelian_gauss_response(side_length, lattice_spacing, frame_index,
   tree_atlas)
4. assemble_maximal_tree_gauss_terms(side_length, lattice_spacing, tree_atlas)
5. invert_gauss_kernel_on_singlet(kernel)
6. build_ordered_tree_charge_flow_map(side_length, lattice_spacing, tree_atlas)
7. solve_temporal_gauge_field(kernel, charges)
8. maximal_tree_nonlocal_energy(side_length, lattice_spacing)

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

01_build_maximal_tree_reduction_atlas

Goal
----
Build a rooted maximal-tree reduction atlas for the source paper's gauge hierarchy.

```python
import numpy as np

def build_maximal_tree_reduction_atlas(side_length) -> np.ndarray:
    '''Rooted incidence-and-subtree atlas for the paper's maximal tree.

    Parameters
    ----------
    side_length : int
        Number L of sites per open-lattice direction, 2 <= L <= 5.

    Returns
    -------
    atlas : np.ndarray, shape (L**3 - 1, 6 + 2*L**3), float
        Link metadata, oriented incidence rows, and descendant-subtree indicators
        in the source hierarchy's ordered n3/n2/n1 families.

    Raises
    ------
    ValueError
        If side_length is not a Python int or NumPy integer in [2, 5]
        (bool excluded).
    '''
    return np.zeros((int(side_length) ** 3 - 1, 6 + 2 * int(side_length) ** 3))
```

### Step 2

02_assemble_qsvt_field_amplitude_objects

Goal
----
Assemble the QSVT singlet Green and field-amplitude digitization.

```python
import numpy as np

def assemble_qsvt_field_amplitude_objects(
    side_length,
    lattice_spacing,
    polynomial_degree,
    n_qubits,
    a_max,
    tree_atlas,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    '''QSVT singlet Green and field-amplitude A, Pi digitization.

    Parameters
    ----------
    side_length : int
        Number L of sites per cubic direction, 2 <= L <= 5.
    lattice_spacing : float
        Positive lattice spacing a.
    polynomial_degree : int
        Chebyshev degree d, 3 <= d <= 16.
    n_qubits : int
        Number K of field-amplitude qubits, 3 <= K <= 6.
    a_max : float
        Positive field-amplitude cutoff Amax.
    tree_atlas : array-like, shape (L**3 - 1, 6 + 2*L**3)
        Rooted maximal-tree atlas returned by Step 01.

    Returns
    -------
    green : np.ndarray, shape (L**3, L**3), float
        Lifted degree-d QSVT singlet Green.
    z_weights : np.ndarray, shape (n_qubits,), float
        Pauli-Z coefficients of A, qubit J the 2**J bit of lambda.
    operators : np.ndarray, shape (3, 2**n_qubits, 2**n_qubits), float
        Real A, real part of Pi, and imaginary part of Pi.

    Raises
    ------
    ValueError
        If any integer argument is not a Python int or NumPy integer in its
        stated range (bool excluded); if lattice_spacing or a_max is not
        finite and positive; or if the atlas has the wrong shape or
        nonfinite entries.
    '''
    return np.empty((0, 0)), np.empty(0), np.empty((3, 0, 0))
```

### Step 3

03_build_reduced_nonabelian_gauss_response

Goal
----
Build the reduced non-Abelian Gauss response and interaction pullback.

```python
import numpy as np

def build_reduced_nonabelian_gauss_response(
    side_length, lattice_spacing, frame_index, tree_atlas
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    '''Reduced Gauss response, nonlocal coefficients, and physical momenta.

    Parameters
    ----------
    side_length : int
        Number L of sites per cubic direction, 2 <= L <= 5.
    lattice_spacing : float
        Positive lattice spacing a.
    frame_index : int
        Residual global SU(3) frame label in {0, 1, 2}; bool is excluded.
    tree_atlas : array-like, shape (L**3 - 1, 6 + 2*L**3)
        Rooted maximal-tree atlas returned by Step 01.

    Returns
    -------
    response : np.ndarray, shape (2, 8*L**3, 8*Nphys), float
        Abelian and connection pieces M0, M1 of the reduced Gauss response.
    coefficients : np.ndarray, shape (3, 8*Nphys, 8*Nphys), float
        Symmetric g**0, g**1, and g**2 nonlocal interaction coefficients.
    momenta : np.ndarray, shape (8*Nphys,), float
        Rotated physical-link momenta, where Nphys=(L-1)**2*(2*L+1).

    Raises
    ------
    ValueError
        If side_length is not a Python int or NumPy integer in [2, 5]
        (bool excluded), or frame_index is not a Python int or NumPy integer
        in {0, 1, 2} (bool excluded); if lattice_spacing is not finite and
        positive; or if the atlas has the wrong shape, nonfinite entries,
        or an invalid incidence tree.
    '''
    return np.empty((2, 0, 0)), np.empty((3, 0, 0)), np.empty(0)
```

### Step 4

04_assemble_maximal_tree_gauss_terms

Goal
----
Assemble the three ordered Gauss-kernel terms from the tree atlas.

```python
import numpy as np

def assemble_maximal_tree_gauss_terms(
    side_length, lattice_spacing, tree_atlas
) -> np.ndarray:
    '''Ordered derivative terms in the maximal-tree-gauge Gauss kernel.

    Parameters
    ----------
    side_length : int
        Number L of sites on each side of the open cubic lattice, 2 <= L <= 5.
    lattice_spacing : float
        Positive lattice spacing.
    tree_atlas : array-like, shape (L**3 - 1, 6 + 2*L**3)
        Rooted maximal-tree atlas returned by Step 01.

    Returns
    -------
    terms : np.ndarray, shape (3, L**3, L**3), float
        The three positive source-kernel terms, ordered n3, n2, n1.

    Raises
    ------
    ValueError
        If a lattice argument is invalid, or the atlas has the wrong shape or
        nonfinite entries.
    '''
    return np.zeros((3, int(side_length) ** 3, int(side_length) ** 3))
```

### Step 5

05_invert_gauss_kernel_on_singlet

Goal
----
Invert the singular Gauss kernel only on the global singlet subspace.

```python
import numpy as np

def invert_gauss_kernel_on_singlet(kernel) -> np.ndarray:
    '''Green matrix for the positive Gauss kernel with only its constant mode removed.

    Parameters
    ----------
    kernel : array-like of shape (N, N)
        Finite real symmetric kernel on N >= 2 sites with one constant zero mode.
        On its orthogonal complement, lambda_min > 1e-11 * lambda_max > 0.

    Returns
    -------
    green : np.ndarray, shape (N, N), float
        Symmetric Moore-Penrose inverse of Keff with zero row and column sums.

    Raises
    ------
    ValueError
        If kernel violates the shape, realness, finiteness, symmetry, constant-mode,
        or relative spectral-cutoff conditions stated above and in the background.
    '''
    return np.zeros_like(np.asarray(kernel, dtype=float))
```

### Step 6

06_build_ordered_tree_charge_flow_map

Goal
----
Extract the rooted charge-to-link flow map from the reduction atlas.

```python
import numpy as np

def build_ordered_tree_charge_flow_map(
    side_length, lattice_spacing, tree_atlas
) -> np.ndarray:
    '''Rooted charge-to-link reconstruction map for the ordered maximal tree.

    Parameters
    ----------
    side_length : int
        Number L of sites per open-lattice direction, 2 <= L <= 5.
    lattice_spacing : float
        Positive lattice spacing.
    tree_atlas : array-like, shape (L**3 - 1, 6 + 2*L**3)
        Rooted maximal-tree atlas returned by Step 01.

    Returns
    -------
    flow_map : np.ndarray, shape (L**3 - 1, L**3), float
        Ordered map whose rows are +a on each oriented link's descendant subtree.

    Raises
    ------
    ValueError
        If a lattice argument is invalid, or the atlas has the wrong shape or
        nonfinite entries.
    '''
    return np.zeros((int(side_length) ** 3 - 1, int(side_length) ** 3))
```

### Step 7

07_solve_temporal_gauge_field

Goal
----
Solve the eliminated temporal gauge field for globally neutral colour sources.

```python
import numpy as np

def solve_temporal_gauge_field(kernel, charges) -> np.ndarray:
    '''Zero-mean temporal field solving K A0 + Q = 0 in every colour channel.

    Parameters
    ----------
    kernel : array-like of shape (N, N)
        Finite real symmetric kernel on N >= 2 sites with one constant zero mode.
        On its orthogonal complement, lambda_min > 1e-11 * lambda_max > 0.
    charges : array-like of shape (C, N)
        Finite real globally neutral source in each of C >= 1 colour rows.

    Returns
    -------
    temporal_field : np.ndarray, shape (C, N), float
        Unique zero-mean solution rows satisfying Keff A0^a = -Qeff^a.

    Raises
    ------
    ValueError
        If the inputs violate the shape, realness, finiteness, symmetry, constant-mode,
        relative spectral-cutoff, or source-neutrality conditions stated above and
        in the background.
    '''
    return np.zeros_like(np.asarray(charges, dtype=float))
```

### Step 8

08_maximal_tree_nonlocal_energy

Goal
----
Evaluate the gauge-fixed nonlocal Yang-Mills interaction.

```python
import numpy as np

def maximal_tree_nonlocal_energy(side_length, lattice_spacing) -> float:
    '''Nonlocal interaction energy for the fixed eight-colour maximal-tree source.

    Parameters
    ----------
    side_length : int
        Number L of sites in each open-lattice direction, 2 <= L <= 5.
    lattice_spacing : float
        Finite positive lattice spacing a.

    Returns
    -------
    energy : float
        Unrounded positive total 0.5*sum_b q_b.T@K_plus@q_b.

    Raises
    ------
    ValueError
        If side_length is not a Python int or NumPy integer in [2, 5]
        (bool excluded), or lattice_spacing is not finite and positive.
    RuntimeError
        If any of the six independently represented reconstruction,
        tree, kernel, flow, field, and energy identities disagree beyond
        the stated 1e-5 tolerance, or if the reduced response gives a
        nonfinite or zero source norm.
    '''
    return 0.0
```
