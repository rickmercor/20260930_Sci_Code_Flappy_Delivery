# Discovering a matrix functional that predicts Hartree-Fock energies without self-consistency

## Background

Self-consistent field theory is built around a fixed point: the Fock matrix is a functional of the one-particle density matrix, the density matrix is constructed from the occupied eigenvectors of the Fock matrix, and the pair is iterated to convergence. Every step of that loop costs a full contraction of the two-electron repulsion tensor, and the loop is what makes a potential energy surface expensive to map.

An alternative is to ask for a matrix whose eigenvectors are close to the self-consistent ones but whose construction never touches a density. Such a matrix can be assembled from the overlap, kinetic and nuclear-attraction matrices, from purely geometric quantities such as the distances between the atoms carrying each pair of basis functions, and from fixed slices of the repulsion tensor, which unlike the Coulomb and exchange matrices require no density to evaluate. Once it is built, a single generalised diagonalisation yields orbitals, and the ordinary Hartree-Fock energy expression can be evaluated at the resulting density.

Nothing constrains such a matrix to resemble a Fock matrix, and there is no variational principle guiding its construction, so the question of which combination of available matrices works best is an empirical one. It is naturally posed as a search over sequences of elementary operations drawn from a fixed catalogue, scored by how well the resulting energies track a reference potential energy curve. The Hartree-Fock energy is evaluated from the resulting density, so a uniform shift of the workspace eigenvalues leaves that energy unchanged. Fitting additive constants is a separate modeling choice that removes geometry-independent bias and makes the objective measure agreement in curve shape.

## Problem

A restricted Hartree-Fock calculation cycles because the Fock matrix depends on the density and
the density comes from diagonalising the Fock matrix. This task replaces that cycle with a single
matrix expression, built only from quantities fixed by the geometry and diagonalised once: a
workspace matrix starts as the core Hamiltonian, is transformed by a short sequence of operations
drawn from a fixed catalogue, and the sequence is found by search.

## The catalogue

Eleven matrices are available as operands, in this order:

`S`, `T`, `V`, `S^-1/2`, `D`, `1/D`, `S^1/2`, `(mm|nn)`, `(mn|mn)`, `(mn|nn)`, `S^-1`

`S`, `T` and `V` are the overlap, kinetic-energy and nuclear-attraction matrices in an STO-3G
basis. A later step keeps only the lower triangle of a matrix, so the order of the basis functions is
part of the specification: the first atom named sits at the origin and the second on the positive
z axis at the given separation; functions run atom by atom in that order and, within an atom,
every s shell in increasing principal quantum number then every p shell, an s shell contributing
one function and a p shell three ordered x, y, z, each contraction normalised as a whole so the
diagonal of `S` is one. `D` holds the distance in bohr between the atoms carrying each pair of basis functions,
and `1/D` its element-wise reciprocal, taken to be zero where `D` vanishes. The last three are
slices of the two-electron integral tensor: `(mm|nn)`, `(mn|mn)` and `(mn|nn)`.

Operations are numbered 0 to 130 and act on the workspace matrix `M`:

| ids | operation |
|---|---|
| 0-15 | `M + c`, `M - c`, `M * c`, `M / c`, with `c` running over `0.5, 1, 3, 4` within each block |
| 16-59 | element-wise `M + A`, `M - A`, `M * A`, `M / A` for the eleven operands in order |
| 60-70 | the matrix product `M A` |
| 71-114 | element-wise `M + A^-1`, `M - A^-1`, `M * A^-1`, `M / A^-1`, the matrix inverse |
| 115-125 | the matrix product `M A^-1` |
| 126-130 | element-wise `exp M`, `ln M`, `M^0.5`, `M^2`, `exp(-M)` |

After the sequence has been applied, the workspace matrix is made symmetric by keeping its lower
triangle and reflecting it: `M <- L + L^T - diag(L)` with `L` the lower triangle including the
diagonal. Orbitals come from the single generalised eigenproblem `M c = S c e`, solved once and
never iterated. The number of doubly occupied orbitals is `n_occ = (Za + Zb)/2`, half the sum of
the two nuclear charges; the density is built from the `n_occ` lowest-lying orbitals, and the
energy is the ordinary Hartree-Fock expression at that density with the nuclear repulsion
`Enuc = Za*Zb / R_bohr` added. Separations are given in angstrom and converted by dividing by
0.52917721092 angstrom per bohr; the first atom sits at the origin and the second on the positive
z axis at that separation.

## What is being fitted, and to what

The training set is given in full on this page: three diatomics on a uniform bond-length grid in
steps of 0.05 Angstrom, LiCl over 1.35 to 3.0 Angstrom, NaCl over 1.6 to 4.0 Angstrom and LiF over
1.05 to 3.0 Angstrom. From each of the 34, 49 and 40 geometries, 15 are kept, at indices
`floor(i(N-1)/14)` for `i = 0..14`, giving 45 geometries in all. The reference for every geometry is
its RHF/STO-3G total energy.

Each predicted curve is allowed one additive constant per molecule, written as a sum of one free
parameter per element it contains and shared across molecules, chosen to minimise the squared
residuals. Take a molecule's constant to be the sum of the `e_atom` offsets of its two elements plus the mean, over that
molecule's own geometries, of predicted energy minus reference energy, and subtract it from the
predicted curve; the reference curve has the bare sum of the two `e_atom` offsets subtracted from
it in the same way. If you report the constants under a different but stated convention, say which
one. The figure of merit is the root mean square residual over all 45 geometries, in kcal/mol at
627.5094740631 kcal/mol per hartree.

## The search, and the answer

Consider every sequence of exactly **two** operations whose first is one of the 22 matrix products
(ids 60-70 and 115-125) and whose second is any of the 131. A sequence is discarded if at any of
the 45 geometries an operation needs a matrix inverse that does not exist, or produces a
non-finite entry, or leaves the eigenvalue problem unsolvable.

Report the smallest figure of merit any surviving sequence attains, in kcal/mol to seven
significant figures. Also state the two operation ids of the sequence that attains
it, the figure of merit of the next best sequence, approximately how many of the 2882 candidates survive at every geometry (entries that vanish by symmetry may be computed as exact zeros or as round-off, which moves the count within roughly 2030 to 2150), the three additive constants the fit assigns to LiCl, NaCl and LiF, and what the
same search returns if the workspace matrix is instead made symmetric by averaging it with its
own transpose. Show the chain that produces your
number rather than the number alone: the single diagonalisation, the occupied count, the energy
expression, the part the basis ordering plays, and how the additive constants are fitted.

Identify the source work this method comes from, and use it rather than list it: say how its own search over programs differs from the exhaustive enumeration prescribed here, including how it changes a program from one step to the next, and how the loss it minimises is defined. State whether the figure of merit you obtain on the fixture is defined as the same quantity as the training error the source reports for its best diatomic program.

## The fixture, in full

Everything needed to evaluate a candidate is given here; nothing has to be looked up.

Nuclear charges `Z`, in atomic units:

```
  Li 3, F 9, Na 11, Cl 17
```

Per-element reference offsets `e_atom`, in hartree. They only size the shared fit:

```
  Li  -7.315525754
  F   -97.98651118
  Na  -159.6682151
  Cl  -454.5421021
```

STO-3G shells. Each line is one shell: its angular momentum (0 for s, 1 for p), then its
three exponents, then its three contraction coefficients.

```
  Li
    l=0  exp 16.119575  2.9362007  0.7946505  coef 0.15432897  0.53532814  0.44463454
    l=0  exp 0.6362897  0.1478601  0.0480887  coef -0.09996723  0.39951283  0.70011547
    l=1  exp 0.6362897  0.1478601  0.0480887  coef 0.15591627  0.60768372  0.39195739
  F
    l=0  exp 166.67913  30.360812  8.2168207  coef 0.15432897  0.53532814  0.44463454
    l=0  exp 6.4648032  1.5022812  0.4885885  coef -0.09996723  0.39951283  0.70011547
    l=1  exp 6.4648032  1.5022812  0.4885885  coef 0.15591627  0.60768372  0.39195739
  Na
    l=0  exp 250.77243  45.678511  12.362388  coef 0.1543289673  0.5353281423  0.4446345422
    l=0  exp 12.040193  2.7978819  0.909958  coef -0.09996722919  0.3995128261  0.7001154689
    l=0  exp 1.4787406  0.4125649  0.1614751  coef -0.219620369  0.2255954336  0.900398426
    l=1  exp 12.040193  2.7978819  0.909958  coef 0.155916275  0.6076837186  0.3919573931
    l=1  exp 1.4787406  0.4125649  0.1614751  coef 0.01058760429  0.5951670053  0.462001012
  Cl
    l=0  exp 601.3456136  109.5358542  29.64467686  coef 0.1543289673  0.5353281423  0.4446345422
    l=0  exp 38.96041889  9.053563477  2.944499834  coef -0.09996722919  0.3995128261  0.7001154689
    l=0  exp 2.129386495  0.5940934274  0.232524141  coef -0.219620369  0.2255954336  0.900398426
    l=1  exp 38.96041889  9.053563477  2.944499834  coef 0.155916275  0.6076837186  0.3919573931
    l=1  exp 2.129386495  0.5940934274  0.232524141  coef 0.01058760429  0.5951670053  0.462001012
```

The 45 geometries, in order, with their RHF/STO-3G reference total energies in hartree.
The separation `R` is in angstrom.

```
   #  pair      R        E_ref
   0  Li-Cl    1.35   -461.7610730219
   1  Li-Cl    1.45   -461.8612125106
   2  Li-Cl    1.55   -461.9236061627
   3  Li-Cl    1.70   -461.9725902441
   4  Li-Cl    1.80   -461.9868405148
   5  Li-Cl    1.90   -461.9920679602
   6  Li-Cl    2.05   -461.9893587642
   7  Li-Cl    2.15   -461.9832176856
   8  Li-Cl    2.25   -461.9749903687
   9  Li-Cl    2.40   -461.9603311846
  10  Li-Cl    2.50   -461.949702674
  11  Li-Cl    2.60   -461.9387468321
  12  Li-Cl    2.75   -461.9221173921
  13  Li-Cl    2.85   -461.9110889403
  14  Li-Cl    3.00   -461.8948558357
  15  Na-Cl    1.60   -614.2842090293
  16  Na-Cl    1.75   -614.4171868917
  17  Na-Cl    1.90   -614.4869669188
  18  Na-Cl    2.10   -614.5240688284
  19  Na-Cl    2.25   -614.5283625367
  20  Na-Cl    2.45   -614.5179565663
  21  Na-Cl    2.60   -614.5040728887
  22  Na-Cl    2.80   -614.4828018837
  23  Na-Cl    2.95   -614.4669787728
  24  Na-Cl    3.10   -614.4523751539
  25  Na-Cl    3.30   -614.4355673044
  26  Na-Cl    3.45   -614.4250567953
  27  Na-Cl    3.65   -614.413516074
  28  Na-Cl    3.80   -614.4063611664
  29  Na-Cl    4.00   -614.3983040498
  30  Li-F     1.05   -105.1985475057
  31  Li-F     1.15   -105.3033146229
  32  Li-F     1.30   -105.3646469892
  33  Li-F     1.45   -105.3720937952
  34  Li-F     1.60   -105.3580221283
  35  Li-F     1.70   -105.3437290008
  36  Li-F     1.85   -105.3197027692
  37  Li-F     2.00   -105.2953633858
  38  Li-F     2.15   -105.2722529659
  39  Li-F     2.30   -105.2507039473
  40  Li-F     2.40   -105.2370300807
  41  Li-F     2.55   -105.2172754948
  42  Li-F     2.70   -105.1983289947
  43  Li-F     2.85   -105.1802933322
  44  Li-F     3.00   -105.1633707317
```
Output Format Requirements:
Scientific reasoning wrapped in <reasoning>...</reasoning> tags. Include enough intermediate calculations to justify the deterministic computation, without turning the response into a general pipeline summary. A single final numeric answer wrapped in <final_answer>...</final_answer> tags.

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

basis_table

Goal
----
Build the table of contracted atomic orbitals for a diatomic molecule. SPECIFICATION: symbols names the two atoms. basis_data maps an element symbol to a list of shells; each shell is a tuple whose first entry is the angular momentum, 0 for s and 1 for p, whose second entry is a list of exactly three Gaussian exponents and whose third entry is a list of exactly three contraction coefficients. Orbitals are generated by walking the two atoms in the order given and, within each atom, its shells in the order given; an s shell contributes one orbital with cartesian powers (0, 0, 0) and a p shell contributes three, with powers (1, 0, 0), (0, 1, 0) and (0, 0, 1) in that order. For an orbital with powers (lx, ly, lz) and total power L, each coefficient is first multiplied by the primitive normalisation, which is two times the exponent over pi, raised to the power three quarters, times four times the exponent raised to the power L over two, divided by the square root of the product of the double factorials of 2*lx-1, 2*ly-1 and 2*lz-1, where the double factorial of -1 is one. The whole contraction is then divided by the square root of its own self-overlap, which is the sum over both primitives of the product of the two scaled coefficients times pi over the exponent sum raised to the power three halves, times that same product of double factorials, divided by twice the exponent sum raised to the power L. The result is a real array with one row per orbital and ten columns, in this order: the index of the atom carrying the orbital, which is 0 or 1; the three cartesian powers; the three exponents; and the three normalised contraction coefficients.

```python
def basis_table(symbols: "list", basis_data: "dict") -> "np.ndarray":
    """Build the table of contracted atomic orbitals for a diatomic molecule. SPECIFICATION: symbols names the two atoms. basis_data maps an element symbol to a list of shells; each shell is a tuple whose first entry is the angular momentum, 0 for s and 1 for p, whose second entry is a list of exactly three Gaussian exponents and whose third entry is a list of exactly three contraction coefficients. Orbitals are generated by walking the two atoms in the order given and, within each atom, its shells in the order given; an s shell contributes one orbital with cartesian powers (0, 0, 0) and a p shell contributes three, with powers (1, 0, 0), (0, 1, 0) and (0, 0, 1) in that order. For an orbital with powers (lx, ly, lz) and total power L, each coefficient is first multiplied by the primitive normalisation, which is two times the exponent over pi, raised to the power three quarters, times four times the exponent raised to the power L over two, divided by the square root of the product of the double factorials of 2*lx-1, 2*ly-1 and 2*lz-1, where the double factorial of -1 is one. The whole contraction is then divided by the square root of its own self-overlap, which is the sum over both primitives of the product of the two scaled coefficients times pi over the exponent sum raised to the power three halves, times that same product of double factorials, divided by twice the exponent sum raised to the power L. The result is a real array with one row per orbital and ten columns, in this order: the index of the atom carrying the orbital, which is 0 or 1; the three cartesian powers; the three exponents; and the three normalised contraction coefficients.

    Parameters
    ----------
    symbols : list of str, length 2
        The two element symbols, in the order the orbitals are laid out.
    basis_data : dict
        Maps an element symbol to its list of shells. Each shell is a tuple
        (l, exponents, coefficients): l is 0 for an s shell and 1 for a p shell,
        and exponents and coefficients are each a list of exactly three floats.

    Returns
    -------
    numpy.ndarray of shape (n, 10) and real dtype.

    Raises
    ------
    ValueError: if symbols does not name exactly two atoms, if basis_data has no entry for one of them, or if any shell does not hold exactly three primitives.
    """
    return [[0.0]]  # placeholder
```

### Step 2

core_integrals

Goal
----
Assemble the one-electron matrices over the contracted basis. SPECIFICATION: table is the array returned by the previous step, coords holds the two nuclear positions in bohr with shape (2, 3), and charges holds their two nuclear charges in the same order. Element (i, j) of the overlap matrix is the integral of the product of orbitals i and j; of the kinetic matrix, the integral of orbital i against minus one half of the Laplacian of orbital j; and of the nuclear attraction matrix, the integral of the product of orbitals i and j against the sum over both nuclei of minus the charge divided by the distance to that nucleus. All three are real and symmetric. The result is a real array of shape (3, n, n) holding, in this order, the overlap matrix, the kinetic matrix and the nuclear attraction matrix.

```python
def core_integrals(table: "np.ndarray", coords: "np.ndarray", charges: "list") -> "np.ndarray":
    """Assemble the one-electron matrices over the contracted basis. SPECIFICATION: table is the array returned by the previous step, coords holds the two nuclear positions in bohr with shape (2, 3), and charges holds their two nuclear charges in the same order. Element (i, j) of the overlap matrix is the integral of the product of orbitals i and j; of the kinetic matrix, the integral of orbital i against minus one half of the Laplacian of orbital j; and of the nuclear attraction matrix, the integral of the product of orbitals i and j against the sum over both nuclei of minus the charge divided by the distance to that nucleus. All three are real and symmetric. The result is a real array of shape (3, n, n) holding, in this order, the overlap matrix, the kinetic matrix and the nuclear attraction matrix.

    Parameters
    ----------
    table : numpy.ndarray of shape (n, 10) and real dtype
        The contracted-orbital table, one row per orbital.
    coords : numpy.ndarray of shape (2, 3) and real dtype
        Cartesian nuclear positions in bohr, one row per atom.
    charges : list of float, length 2
        Nuclear charges, in the same atom order as coords.

    Returns
    -------
    numpy.ndarray of shape (3, n, n) and real dtype.

    Raises
    ------
    ValueError: if the basis table is empty or does not have ten columns, or if coords does not have shape (2, 3), or if charges does not have length two.
    """
    return [[[0.0]]]  # placeholder
```

### Step 3

electron_repulsion

Goal
----
Assemble the two-electron repulsion tensor over the contracted basis. SPECIFICATION: table and coords are as in the previous step. Element (m, n, l, s) is the integral over both electron coordinates of the product of orbitals m and n evaluated at the first electron, times the product of orbitals l and s evaluated at the second, divided by the distance between the two electrons. The result is a real array of shape (n, n, n, n).

```python
def electron_repulsion(table: "np.ndarray", coords: "np.ndarray") -> "np.ndarray":
    """Assemble the two-electron repulsion tensor over the contracted basis. SPECIFICATION: table and coords are as in the previous step. Element (m, n, l, s) is the integral over both electron coordinates of the product of orbitals m and n evaluated at the first electron, times the product of orbitals l and s evaluated at the second, divided by the distance between the two electrons. The result is a real array of shape (n, n, n, n).

    Parameters
    ----------
    table : numpy.ndarray of shape (n, 10) and real dtype
        The contracted-orbital table, one row per orbital.
    coords : numpy.ndarray of shape (2, 3) and real dtype
        Cartesian nuclear positions in bohr, one row per atom.

    Returns
    -------
    numpy.ndarray of shape (n, n, n, n) and real dtype.

    Raises
    ------
    ValueError: if the basis table is empty or does not have ten columns, or if coords does not have shape (2, 3).
    """
    return [[[[0.0]]]]  # placeholder
```

### Step 4

descriptor_matrices

Goal
----
Build the four geometry and repulsion descriptors the search draws on. SPECIFICATION: table and coords are as before and eri is the tensor returned by the previous step. The distance matrix holds, at (i, j), the distance in bohr between the nucleus carrying orbital i and the nucleus carrying orbital j, so its entries are zero whenever the two orbitals sit on the same atom. The three repulsion slices are, at (m, n), the tensor elements (m, m, n, n), then (m, n, m, n), then (m, n, n, n). The result is a real array of shape (4, n, n) holding, in this order, the distance matrix and those three slices.

```python
def descriptor_matrices(table: "np.ndarray", coords: "np.ndarray", eri: "np.ndarray") -> "np.ndarray":
    """Build the four geometry and repulsion descriptors the search draws on. SPECIFICATION: table and coords are as before and eri is the tensor returned by the previous step. The distance matrix holds, at (i, j), the distance in bohr between the nucleus carrying orbital i and the nucleus carrying orbital j, so its entries are zero whenever the two orbitals sit on the same atom. The three repulsion slices are, at (m, n), the tensor elements (m, m, n, n), then (m, n, m, n), then (m, n, n, n). The result is a real array of shape (4, n, n) holding, in this order, the distance matrix and those three slices.

    Parameters
    ----------
    table : numpy.ndarray of shape (n, 10) and real dtype
        The contracted-orbital table, one row per orbital.
    coords : numpy.ndarray of shape (2, 3) and real dtype
        Cartesian nuclear positions in bohr, one row per atom.
    eri : numpy.ndarray of shape (n, n, n, n) and real dtype
        The two-electron repulsion tensor in chemists' notation.

    Returns
    -------
    numpy.ndarray of shape (4, n, n) and real dtype.

    Raises
    ------
    ValueError: if the basis table does not have ten columns, or if eri does not have shape (n, n, n, n) for the n rows of the table.
    """
    return [[[0.0]]]  # placeholder
```

### Step 5

operand_table

Goal
----
Collect the eleven matrices the operations are allowed to use, in their fixed order. SPECIFICATION: core is the array of shape (3, n, n) returned by the one-electron step and descriptors the array of shape (4, n, n) returned by the previous step. The result is a real array of shape (11, n, n) whose entries are, in this order: the overlap matrix; the kinetic matrix; the nuclear attraction matrix; the inverse square root of the overlap matrix; the distance matrix; the element-wise reciprocal of the distance matrix, defined to be zero at every position where the distance itself is zero; the square root of the overlap matrix; and then the three repulsion slices in the order they arrive, followed by the matrix inverse of the overlap matrix. Both the square root and the inverse square root are the symmetric ones, obtained by diagonalising the overlap matrix and raising its eigenvalues to the power one half or minus one half while keeping its eigenvectors.

```python
def operand_table(core: "np.ndarray", descriptors: "np.ndarray") -> "np.ndarray":
    """Collect the eleven matrices the operations are allowed to use, in their fixed order. SPECIFICATION: core is the array of shape (3, n, n) returned by the one-electron step and descriptors the array of shape (4, n, n) returned by the previous step. The result is a real array of shape (11, n, n) whose entries are, in this order: the overlap matrix; the kinetic matrix; the nuclear attraction matrix; the inverse square root of the overlap matrix; the distance matrix; the element-wise reciprocal of the distance matrix, defined to be zero at every position where the distance itself is zero; the square root of the overlap matrix; and then the three repulsion slices in the order they arrive, followed by the matrix inverse of the overlap matrix. Both the square root and the inverse square root are the symmetric ones, obtained by diagonalising the overlap matrix and raising its eigenvalues to the power one half or minus one half while keeping its eigenvectors.

    Parameters
    ----------
    core : numpy.ndarray of shape (3, n, n) and real dtype
        The overlap, kinetic-energy and nuclear-attraction matrices, in that order.
    descriptors : numpy.ndarray of shape (4, n, n) and real dtype
        The distance matrix and the three repulsion slices (mm|nn), (mn|mn) and (mn|nn),
        in that order.

    Returns
    -------
    numpy.ndarray of shape (11, n, n) and real dtype.

    Raises
    ------
    ValueError: if core does not have shape (3, n, n), if descriptors does not have shape (4, n, n), or if the overlap matrix is singular or not positive definite.
    """
    return [[[0.0]]]  # placeholder
```

### Step 6

apply_operation

Goal
----
Apply one catalogue operation to the workspace matrix. SPECIFICATION: operands is the array of shape (11, n, n) from the previous step and A(k) denotes its k-th entry, counting from zero. Operations are numbered as follows. Ids 0 to 15 combine M with a constant c, adding for ids 0 to 3, subtracting for 4 to 7, multiplying for 8 to 11 and dividing for 12 to 15, with c running over 0.5, 1, 3 and 4 in that order inside each block of four. Ids 16 to 59 combine M element-wise with A(k), adding for 16 to 26, subtracting for 27 to 37, multiplying for 38 to 48 and dividing for 49 to 59, with k running from 0 to 10 in that order inside each block of eleven. Ids 60 to 70 replace M by the matrix product of M with A(k), with k running from 0 to 10. Ids 71 to 114 repeat the four element-wise combinations but against the matrix inverse of A(k), in the same arrangement, so 71 to 81 add, 82 to 92 subtract, 93 to 103 multiply and 104 to 114 divide. Ids 115 to 125 replace M by the matrix product of M with the matrix inverse of A(k). Id 126 applies the exponential to every element, 127 the natural logarithm, 128 raises every element to the power one half, 129 squares every element, and 130 applies the exponential of the negative of every element. Division and the element-wise functions act entry by entry; only the products named as matrix products, and the inverses, are matrix operations. The result is the transformed matrix, of the same shape as M.

```python
def apply_operation(M: "np.ndarray", op_id: "int", operands: "np.ndarray") -> "np.ndarray":
    """Apply one catalogue operation to the workspace matrix. SPECIFICATION: operands is the array of shape (11, n, n) from the previous step and A(k) denotes its k-th entry, counting from zero. Operations are numbered as follows. Ids 0 to 15 combine M with a constant c, adding for ids 0 to 3, subtracting for 4 to 7, multiplying for 8 to 11 and dividing for 12 to 15, with c running over 0.5, 1, 3 and 4 in that order inside each block of four. Ids 16 to 59 combine M element-wise with A(k), adding for 16 to 26, subtracting for 27 to 37, multiplying for 38 to 48 and dividing for 49 to 59, with k running from 0 to 10 in that order inside each block of eleven. Ids 60 to 70 replace M by the matrix product of M with A(k), with k running from 0 to 10. Ids 71 to 114 repeat the four element-wise combinations but against the matrix inverse of A(k), in the same arrangement, so 71 to 81 add, 82 to 92 subtract, 93 to 103 multiply and 104 to 114 divide. Ids 115 to 125 replace M by the matrix product of M with the matrix inverse of A(k). Id 126 applies the exponential to every element, 127 the natural logarithm, 128 raises every element to the power one half, 129 squares every element, and 130 applies the exponential of the negative of every element. Division and the element-wise functions act entry by entry; only the products named as matrix products, and the inverses, are matrix operations. The result is the transformed matrix, of the same shape as M.

    Parameters
    ----------
    M : numpy.ndarray of shape (n, n) and real dtype
        The workspace matrix the operation acts on.
    op_id : int
        The operation id, an integer from 0 to 130 inclusive.
    operands : numpy.ndarray of shape (11, n, n) and real dtype
        The eleven operand matrices, in the order fixed by the operand table.

    Returns
    -------
    numpy.ndarray of shape (n, n) and real dtype.

    Raises
    ------
    ValueError: if op_id is outside 0 to 130, if M is not square, if operands does not have shape (11, n, n) matching M, if an operation needs the inverse of a matrix that has none, or if the result holds a non-finite entry.
    """
    return [[0.0]]  # placeholder
```

### Step 7

workspace_matrix

Goal
----
Run a whole program and return the symmetrised workspace matrix. SPECIFICATION: the workspace matrix starts as hcore and each entry of program, an integer operation id, is applied to it in order using the rules of the previous step. Once every operation has been applied the matrix is symmetrised by taking L to be its lower triangle including the diagonal and replacing it by L plus the transpose of L minus the diagonal of L, so that the final matrix agrees with the unsymmetrised one on and below the diagonal and is symmetric. An empty program leaves the workspace matrix at hcore before that symmetrisation. The result is a real symmetric array with the same shape as hcore.

```python
def workspace_matrix(hcore: "np.ndarray", program: "list", operands: "np.ndarray") -> "np.ndarray":
    """Run a whole program and return the symmetrised workspace matrix. SPECIFICATION: the workspace matrix starts as hcore and each entry of program, an integer operation id, is applied to it in order using the rules of the previous step. Once every operation has been applied the matrix is symmetrised by taking L to be its lower triangle including the diagonal and replacing it by L plus the transpose of L minus the diagonal of L, so that the final matrix agrees with the unsymmetrised one on and below the diagonal and is symmetric. An empty program leaves the workspace matrix at hcore before that symmetrisation. The result is a real symmetric array with the same shape as hcore.

    Parameters
    ----------
    hcore : numpy.ndarray of shape (n, n) and real dtype
        The core Hamiltonian, which the workspace matrix starts as.
    program : list of int
        The operation ids to apply, in order. May be empty.
    operands : numpy.ndarray of shape (11, n, n) and real dtype
        The eleven operand matrices, in the order fixed by the operand table.

    Returns
    -------
    numpy.ndarray of shape (n, n) and real dtype.

    Raises
    ------
    ValueError: if hcore is not square, or if any operation in program fails for the reasons listed for the previous step.
    """
    return [[0.0]]  # placeholder
```

### Step 8

program_energy

Goal
----
Read an electronic energy out of a workspace matrix. SPECIFICATION: solve the generalised eigenvalue problem in which M plays the role of the effective Hamiltonian and S the metric, that is find the coefficients and eigenvalues satisfying M c = S c e. Order the eigenvalues ascending and keep the n_occ eigenvectors with the smallest ones. The density matrix is twice the product of that block of coefficients with its own transpose. The Coulomb matrix has element (m, n) equal to the sum over l and s of the repulsion element (m, n, l, s) times the density element (l, s), and the exchange matrix has element (m, n) equal to the sum over l and s of the repulsion element (m, l, s, n) times the density element (l, s). The Fock matrix is hcore plus the Coulomb matrix minus one half of the exchange matrix. Return one half of the sum over all elements of the density matrix times the sum of hcore and the Fock matrix. Nuclear repulsion is not included.

```python
def program_energy(M: "np.ndarray", S: "np.ndarray", hcore: "np.ndarray", eri: "np.ndarray", n_occ: "int") -> "float":
    """Read an electronic energy out of a workspace matrix. SPECIFICATION: solve the generalised eigenvalue problem in which M plays the role of the effective Hamiltonian and S the metric, that is find the coefficients and eigenvalues satisfying M c = S c e. Order the eigenvalues ascending and keep the n_occ eigenvectors with the smallest ones. The density matrix is twice the product of that block of coefficients with its own transpose. The Coulomb matrix has element (m, n) equal to the sum over l and s of the repulsion element (m, n, l, s) times the density element (l, s), and the exchange matrix has element (m, n) equal to the sum over l and s of the repulsion element (m, l, s, n) times the density element (l, s). The Fock matrix is hcore plus the Coulomb matrix minus one half of the exchange matrix. Return one half of the sum over all elements of the density matrix times the sum of hcore and the Fock matrix. Nuclear repulsion is not included.

    Parameters
    ----------
    M : numpy.ndarray of shape (n, n) and real dtype
        The symmetrised workspace matrix whose orbitals are used.
    S : numpy.ndarray of shape (n, n) and real dtype
        The overlap matrix.
    hcore : numpy.ndarray of shape (n, n) and real dtype
        The core Hamiltonian.
    eri : numpy.ndarray of shape (n, n, n, n) and real dtype
        The two-electron repulsion tensor in chemists' notation.
    n_occ : int
        The number of doubly occupied orbitals, at least 1 and at most n.

    Returns
    -------
    float.

    Raises
    ------
    ValueError: if n_occ is not between one and the size of the basis, if the matrix shapes are inconsistent, if the generalised eigenvalue problem cannot be solved, or if the energy is not finite.
    """
    return 0.0  # placeholder
```

### Step 9

target_function

Goal
----
Score a set of predicted energies against the reference curve. SPECIFICATION: species is a sequence of pairs of element symbols, one per geometry, e_pips the predicted total energies and e_ref the reference total energies, both in hartree and in the same order, and e_atom maps an element symbol to its reference atomic energy. For each distinct pair in species, the reference shift is the sum of the two atomic energies of that pair, and the predicted shift is that same sum plus the mean, over just the geometries of that pair, of the predicted energy minus the reference energy. The residual at a geometry is the predicted energy minus its predicted shift, minus the quantity the reference energy minus its reference shift. Return the square root of the mean of the squared residuals taken over every geometry of every pair, multiplied by 627.5094740631 to convert from hartree to kcal/mol.

```python
def target_function(species: "list", e_pips: "list", e_ref: "list", e_atom: "dict") -> "float":
    """Score a set of predicted energies against the reference curve. SPECIFICATION: species is a sequence of pairs of element symbols, one per geometry, e_pips the predicted total energies and e_ref the reference total energies, both in hartree and in the same order, and e_atom maps an element symbol to its reference atomic energy. For each distinct pair in species, the reference shift is the sum of the two atomic energies of that pair, and the predicted shift is that same sum plus the mean, over just the geometries of that pair, of the predicted energy minus the reference energy. The residual at a geometry is the predicted energy minus its predicted shift, minus the quantity the reference energy minus its reference shift. Return the square root of the mean of the squared residuals taken over every geometry of every pair, multiplied by 627.5094740631 to convert from hartree to kcal/mol.

    Parameters
    ----------
    species : list of tuple of str
        One pair of element symbols per geometry.
    e_pips : list of float
        The predicted total energy at each geometry, in hartree.
    e_ref : list of float
        The reference total energy at each geometry, in hartree. Same length as e_pips.
    e_atom : dict
        Maps an element symbol to a float, used only to size the shared fit.

    Returns
    -------
    float.

    Raises
    ------
    ValueError: if species, e_pips and e_ref do not share one non-zero length, if any energy is not finite, or if e_atom has no entry for an element that appears.
    """
    return 0.0  # placeholder
```

### Step 10

best_two_operation_program

Goal
----
Search every two-operation program and report the best one. SPECIFICATION: species is a sequence of pairs of element symbols, one per geometry, bond_lengths the corresponding separations in angstrom, and e_ref the reference total energies in hartree; e_atom and basis_data are as in the earlier steps and charges maps an element symbol to its nuclear charge. At each geometry the first atom sits at the origin and the second on the positive z axis, with the separation converted to bohr by dividing by 0.52917721092; the number of occupied orbitals is half the sum of the two nuclear charges, and the nuclear repulsion is the product of the two charges divided by the separation in bohr. The total predicted energy at a geometry is the electronic energy of the workspace matrix plus that nuclear repulsion. The candidates are every ordered pair of operation ids whose first entry is one of the twenty two matrix products, that is ids 60 to 70 and 115 to 125, and whose second entry is any id from 0 to 130. A candidate is discarded if at any geometry an operation fails, the eigenvalue problem cannot be solved, or the energy is not finite. Among the candidates that survive at every geometry, choose the one with the smallest score, breaking a tie by the smaller first id and then by the smaller second id. Return that smallest score in kcal/mol.

```python
def best_two_operation_program(species: "list", bond_lengths: "list", e_ref: "list", e_atom: "dict",
                               basis_data: "dict", charges: "dict") -> "float":
    """Search every two-operation program and report the best one. SPECIFICATION: species is a sequence of pairs of element symbols, one per geometry, bond_lengths the corresponding separations in angstrom, and e_ref the reference total energies in hartree; e_atom and basis_data are as in the earlier steps and charges maps an element symbol to its nuclear charge. At each geometry the first atom sits at the origin and the second on the positive z axis, with the separation converted to bohr by dividing by 0.52917721092; the number of occupied orbitals is half the sum of the two nuclear charges, and the nuclear repulsion is the product of the two charges divided by the separation in bohr. The total predicted energy at a geometry is the electronic energy of the workspace matrix plus that nuclear repulsion. The candidates are every ordered pair of operation ids whose first entry is one of the twenty two matrix products, that is ids 60 to 70 and 115 to 125, and whose second entry is any id from 0 to 130. A candidate is discarded if at any geometry an operation fails, the eigenvalue problem cannot be solved, or the energy is not finite. Among the candidates that survive at every geometry, choose the one with the smallest score, breaking a tie by the smaller first id and then by the smaller second id. Return that smallest score in kcal/mol.

    Parameters
    ----------
    species : list of tuple of str
        One pair of element symbols per geometry.
    bond_lengths : list of float
        The separation at each geometry, in angstrom. Same length as species.
    e_ref : list of float
        The reference total energy at each geometry, in hartree. Same length as species.
    e_atom : dict
        Maps an element symbol to a float, used only to size the shared fit.
    basis_data : dict
        Maps an element symbol to its list of shells, as in the first step.
    charges : dict
        Maps an element symbol to its nuclear charge as a float.

    Returns
    -------
    float.

    Raises
    ------
    ValueError: if species, bond_lengths and e_ref do not share one non-zero length, if any bond length is not positive and finite, if a molecule does not have an even number of electrons, or if no candidate program survives at every geometry.
    """
    return 0.0  # placeholder
```
