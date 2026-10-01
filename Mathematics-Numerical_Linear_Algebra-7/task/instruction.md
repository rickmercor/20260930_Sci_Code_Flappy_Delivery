# Mathematics-Numerical_Linear_Algebra-7

## Background

Space-time variational methods treat time like a spatial variable: the wave equation is
discretised on the whole space-time cylinder at once, and after a Fourier expansion in the
spatial eigenfunctions of the Laplacian the analysis reduces to a family of one-dimensional
problems in time, one per spatial eigenvalue mu. In the isogeometric version the time
discretisation uses B-splines of degree p whose smoothness across the mesh is chosen
independently through the regularity r, obtained by repeating each interior knot p - r
times. The scheme is a Petrov-Galerkin one: the trial space carries the initial condition and
the test space the end condition, so the two index ranges are shifted against each other
and the system matrices are not symmetric.

Away from the two ends of the time interval the system matrix is block Toeplitz with
blocks of size p - r: every block row is a translate of the previous one, because the mesh
is uniform and the splines are translates of each other. Such matrices are finite sections
of an infinite block Toeplitz operator whose symbol is a matrix polynomial in a variable on
the unit circle, and a classical theory relates the behaviour of the condition numbers of the
sections, as the number of blocks grows, to the zeros of the determinant of that symbol:
zeros on the unit circle make the operator non-Fredholm and the growth at most
polynomial, while zeros off the circle can drive exponential growth at a rate set by the
zero nearest to the circle. The mesh-scaled parameter rho = mu h^2 moves the zeros, so
the parameter axis splits into ranges of polynomial and of exponential growth separated by
thresholds that are algebraic numbers determined by the spline space alone.

In the exponential regime the smallest singular value of a section decays geometrically
with the number of blocks and soon falls below the rounding level of double precision
arithmetic, after which any floating-point factorisation reports noise of the order of the
unit roundoff times the norm. The entries of the sections are rational numbers, however, so
the inverse can be computed exactly and the condition numbers recovered to full accuracy
at any size. Maximal-regularity splines have their own stability theory, with a mesh
condition on rho given in closed form through values of the Riemann zeta function.

## Problem

A recent paper studies how the spectral condition numbers of the system matrices of space-time isogeometric discretisations of the linear wave equation grow with the number of time elements, by reducing them to finite sections of block Toeplitz operators with matrix-polynomial symbols; it treats the spline spaces (p, r) = (2, 0), (3, 0) and (3, 1) and shows that the parameter ranges of exponential growth are cut out by the zeros of the symbol determinant. Recover the construction from the source and extend it to the space of degree p = 4 and regularity r = 2, which the source does not treat.

The setting, in the source's conventions: on the uniform mesh of unit elements covering [0, nmesh], the spline space of degree p and regularity r is spanned by the B-splines of the open knot vector in which 0 and nmesh appear p + 1 times and every interior integer knot appears p - r times (Cox-de Boor recursion, no normalisation), indexed from 0 in knot order. The trial functions are those with indices 1 to n, the test functions those with indices 0 to n - 1, n = nmesh (p - r) + r, and with unit elements the mesh-independent matrices are M[l, j] = integral of phi_j phi_{l-1} and B[l, j] = integral of phi_j' phi_{l-1}', l, j = 1, ..., n. The wave scheme matrix is K(rho) = -B + rho M with rho = mu h^2. Away from the boundary K is block Toeplitz with blocks of size N = p - r, the block partition starting at the first row and column (rows and columns 1 to N form the first block); with the block at block offset d = block column index minus block row index written A_{-d}, the symbol is A_rho(t) = sum over d of (-B_d + rho M_d) t^{-d}, S_rho(t) = t^k A_rho(t) with k the largest positive offset, and the objects of interest are the pure block Toeplitz sections T_n(A_rho) = [A_{i-j}] of n block rows, the zeros of det S_rho(t) counted inside, on and outside the unit circle (zeros at the origin count as inside), the values of rho at which that count changes, and the spectral condition numbers kappa_2(T_n(A_rho)) of the sections, which in the exponential regime exceed the reciprocal of the double precision unit roundoff by many orders of magnitude and therefore cannot be read off a floating-point factorisation of the section.

Work with (p, r) = (4, 2) at rho = 60. Report: the complete list of positive thresholds of rho at which the zero location type of det S_rho changes, to six significant figures; the type at rho = 60 as the triple (inside, on, outside) and the moduli of the inside zero closest to the unit circle and of the outside zero closest to it; the resulting asymptotic growth factor of the condition numbers per block row; the base-10 logarithm of kappa_2(T_60(A_60)), the section with 60 block rows (120 rows and columns), to six significant figures, together with its largest singular value and the base-10 logarithm of its smallest singular value; the same base-10 logarithms of the condition number for the sections with 40 and 50 block rows and of the smallest singular value for 40 block rows; the ratios kappa_2(T_60)/kappa_2(T_59) and kappa_2(T_41)/kappa_2(T_40); the (1, 1) entry of the two-block section; and, as checks of the construction, the thresholds the same chain returns for the source's own cases (2, 0), (3, 0) and (3, 1), which must reproduce the values the source proves. State which arithmetic produced the condition numbers to the required accuracy and why a double precision singular value decomposition of the section cannot; the width of the narrow exponential window of the (4, 2) scheme below rho = 10 and the growth factor per block inside it at rho = 9.875. The final answer is log10 kappa_2(T_60(A_60)) for (p, r) = (4, 2), the exact value, not a double precision estimate and not the value for a section of another size.

Your reasoning should also report, as evidence that the chain was executed: the three (4, 2) thresholds; the type triple at rho = 60 with its two moduli and the asymptotic growth factor; the largest singular value and log10 of the smallest singular value of T_60; log10 kappa_2 of T_40 and T_50 and log10 of the smallest singular value of T_40; the two consecutive-section ratios; the (1, 1) entry of the two-block section; the source's thresholds for (2, 0), (3, 0) and (3, 1) as recomputed; the window width and the growth factor at rho = 9.875; and, from the source, the type it assigns to the (3, 1) symbol for rho in (42, infinity) and why it analyses pure block Toeplitz extensions when r > 0, the value of its maximal-regularity mesh condition (the closed form in the Riemann zeta function) for p = 3 and for p = 4 with a comparison against the (4, 2) thresholds, and the matrix sizes shown in its condition-number plots together with what their levelling off in the exponential ranges signifies.

## Output format

```
## Output format

Scientific reasoning wrapped in <reasoning>...</reasoning> tags. Include enough intermediate calculations to justify the deterministic computation, without turning the response into a general pipeline summary. A single final numeric answer wrapped in <final_answer>...</final_answer> tags.
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

spline_basis_values

Goal
----
Return the values of the B-spline basis of degree p and regularity r on the uniform mesh of nmesh unit elements covering [0, nmesh], at the given evaluation points. The basis is generated by the Cox-de Boor recursion on the open knot vector in which the two end knots 0 and nmesh appear p + 1 times and each interior integer knot appears p - r times, so that the splines are C^r across the interior knots; no other normalisation is applied. The basis functions are indexed from 0 in knot order and there are nmesh (p - r) + r + 1 of them (the number of knots, nmesh (p - r) + r + p + 2, minus p + 1). Points that coincide with an interior knot belong to the element on their right; the right end point belongs to the last element. Reject an invalid degree or regularity, nmesh < 1, and points outside [0, nmesh].

```python
# =============================================================================

def spline_basis_values(p, r, nmesh, tvals):
    """Return the values of the B-spline basis of degree p and regularity r on the uniform mesh
    of nmesh unit elements covering [0, nmesh], at the given evaluation points.

    Args:
        p (int): polynomial degree of the spline space, at least 1.
        r (int): regularity of the spline space, an integer with 0 <= r <= p - 1.
        nmesh (int): number of uniform mesh elements of unit width on [0, nmesh], at
            least 1.
        tvals (array_like of float): one-dimensional evaluation points in [0, nmesh].

    Returns:
        numpy.ndarray of shape (nfun, len(tvals)) with the values of every basis
        function at every point, where nfun = nmesh (p - r) + r + 1 (the open knot
        vector has nmesh (p - r) + r + p + 2 entries).

    Raises:
        ValueError: if p < 1 or r is outside 0 <= r <= p - 1.
        ValueError: if nmesh < 1 or a point lies outside [0, nmesh] or is not finite.
    """
    return None
```

### Step 2

petrov_galerkin_matrices

Goal
----
Return the two n x n Petrov-Galerkin matrices of the space-time scheme on the mesh of the previous step, with n = nmesh (p - r) + r. The trial space is spanned by the basis functions with indices 1 to n, which vanish at t = 0, and the test space by the functions with indices 0 to n - 1, which vanish at t = nmesh; the mass matrix has entries M[l, j] = integral over [0, nmesh] of phi_j times phi_{l-1}, and the stiffness matrix has entries B[l, j] = integral of the first derivatives of the same two functions, for l, j = 1, ..., n, both computed exactly. With unit elements these are already the mesh-independent scaled matrices of the source, whose entries are rational numbers. Reject invalid p, r or nmesh < 2.

```python
def petrov_galerkin_matrices(p, r, nmesh):
    """Return the two n x n Petrov-Galerkin matrices of the space-time scheme on the mesh of
    the previous step, with n = nmesh (p - r) + r.

    Args:
        p (int): polynomial degree of the spline space, at least 1.
        r (int): regularity of the spline space, an integer with 0 <= r <= p - 1.
        nmesh (int): number of uniform unit elements, at least 2.

    Returns:
        numpy.ndarray of shape (2, n, n) with n = nmesh (p - r) + r: the mass matrix in
        [0] and the stiffness matrix in [1].

    Raises:
        ValueError: if p < 1 or r is outside 0 <= r <= p - 1.
        ValueError: if nmesh < 2.
    """
    return None
```

### Step 3

interior_symbol_blocks

Goal
----
Return the N x N blocks, N = p - r, that define the block Toeplitz structure of the two matrices of the previous step away from the boundaries: partition rows and columns into consecutive blocks of size N starting at the first row and column (rows and columns 0 to N - 1 form block 0), and for a block row deep inside the matrix return the mass block and the stiffness block found at block offset d from the diagonal, where d is the block column index minus the block row index, for d from -p to p, in an array indexed by d + p, with zero blocks where the offset is empty. The blocks must be independent of the block row chosen and of nmesh once the mesh is large enough for the interior to exist; the last r rows and columns of the matrices are boundary perturbations and are not part of the pattern. Reject invalid p or r.

```python
def interior_symbol_blocks(p, r):
    """Return the N x N blocks, N = p - r, that define the block Toeplitz structure of the two
    matrices of the previous step away from the boundaries: partition rows and columns into
    consecutive blocks of size N starting at the first row and column (rows and columns 0 to
    N - 1 form block 0), and for a block row deep inside the matrix return the mass block
    and the stiffness block found at block offset d from the diagonal, where d is the block
    column index minus the block row index, for d from -p to p, in an array indexed by d +
    p, with zero blocks where the offset is empty.

    Args:
        p (int): polynomial degree of the spline space, at least 1.
        r (int): regularity of the spline space, an integer with 0 <= r <= p - 1.

    Returns:
        numpy.ndarray of shape (2p + 1, 2, N, N), N = p - r: entry [d + p, 0] is the
        mass block and [d + p, 1] the stiffness block at block offset d = block column
        index minus block row index (blocks of size N counted from row and column 0),
        zero when absent.

    Raises:
        ValueError: if p < 1 or r is outside 0 <= r <= p - 1.
    """
    return None
```

### Step 4

symbol_determinant

Goal
----
Return the coefficients, highest power first, of the scalar polynomial det S_rho(t), where S_rho is the matrix polynomial symbol of the block Toeplitz matrices K = -B + rho M built from the interior blocks of the previous step. With the block at offset d written A_{-d}, the Laurent symbol is A_rho(t) = sum over d of (-B_d + rho M_d) t^{-d}; S_rho(t) = t^k A_rho(t) with k the largest positive offset present, so that S_rho is a genuine matrix polynomial, and its determinant is taken exactly. Reject invalid p, r or a negative rho.

```python
def symbol_determinant(p, r, rho):
    """Return the coefficients, highest power first, of the scalar polynomial det S_rho(t),
    where S_rho is the matrix polynomial symbol of the block Toeplitz matrices K = -B + rho
    M built from the interior blocks of the previous step.

    Args:
        p (int): polynomial degree of the spline space, at least 1.
        r (int): regularity of the spline space, an integer with 0 <= r <= p - 1.
        rho (float or Fraction): non-negative mesh-scaled parameter rho = mu h^2 of the
            wave scheme.

    Returns:
        numpy.ndarray of shape (deg + 1,) with the coefficients of det S_rho(t), highest
        power first.

    Raises:
        ValueError: if p < 1 or r is outside 0 <= r <= p - 1.
        ValueError: if rho is negative.
    """
    return None
```

### Step 5

zero_location_type

Goal
----
Classify the zeros of det S_rho(t) from the previous step with respect to the unit circle: return the number of zeros strictly inside (zeros at the origin included), on, and strictly outside the circle, followed by the largest modulus among the inside zeros and the smallest modulus among the outside zeros, each 0 if that group is empty. Zeros on the circle must be identified exactly, not by a floating tolerance on the modulus: after removing the factor t^j from a zero of multiplicity j at the origin the determinant is self-reciprocal, so the substitution y = t + 1/t reduces it to a polynomial in y of half the degree whose real roots in [-2, 2] correspond to pairs of zeros on the circle and whose other roots correspond to pairs inside and outside. Reject invalid p, r or a negative rho.

```python
def zero_location_type(p, r, rho):
    """Classify the zeros of det S_rho(t) from the previous step with respect to the unit
    circle: return the number of zeros strictly inside (zeros at the origin included), on,
    and strictly outside the circle, followed by the largest modulus among the inside zeros
    and the smallest modulus among the outside zeros, each 0 if that group is empty.

    Args:
        p (int): polynomial degree of the spline space, at least 1.
        r (int): regularity of the spline space, an integer with 0 <= r <= p - 1.
        rho (float or Fraction): non-negative parameter of the wave scheme.

    Returns:
        numpy.ndarray of shape (5,): numbers of zeros strictly inside, on and strictly
        outside the unit circle, the largest modulus among the inside zeros and the
        smallest modulus among the outside zeros (0 when a group is empty).

    Raises:
        ValueError: if p < 1 or r is outside 0 <= r <= p - 1.
        ValueError: if rho is negative.
    """
    return None
```

### Step 6

growth_thresholds

Goal
----
Return, in increasing order, every positive value of rho at which the zero location type of the previous step changes, to at least ten significant figures. Such a change happens only where a zero of the determinant crosses the unit circle, that is where a real root of the reduced polynomial in y crosses 2 or -2, or where a pair of real roots of the reduced polynomial turns complex; the candidate values are the positive roots of the exact polynomials in rho obtained by evaluating the reduced polynomial at y = 2 and at y = -2 and by its discriminant, and a candidate is kept only if the type really differs on the two sides of it. Reject invalid p or r.

```python
def growth_thresholds(p, r):
    """Return, in increasing order, every positive value of rho at which the zero location type
    of the previous step changes, to at least ten significant figures.

    Args:
        p (int): polynomial degree of the spline space, at least 1.
        r (int): regularity of the spline space, an integer with 0 <= r <= p - 1.

    Returns:
        numpy.ndarray of the positive values of rho, in increasing order, at which the
        zero location type changes.

    Raises:
        ValueError: if p < 1 or r is outside 0 <= r <= p - 1.
    """
    return None
```

### Step 7

block_toeplitz_section

Goal
----
Return the leading nblocks x nblocks block section of the infinite block Toeplitz matrix generated by the symbol of the wave scheme, that is the matrix whose block at block position (i, j) is -B_d + rho M_d with d = j - i taken from the interior blocks of step 3, and zero where that offset is absent; this is the pure block Toeplitz matrix whose finite sections the source analyses, without the boundary perturbations of the actual Petrov-Galerkin matrix. Reject invalid p, r, a negative rho or nblocks < 1.

```python
def block_toeplitz_section(p, r, rho, nblocks):
    """Return the leading nblocks x nblocks block section of the infinite block Toeplitz matrix
    generated by the symbol of the wave scheme, that is the matrix whose block at block
    position (i, j) is -B_d + rho M_d with d = j - i taken from the interior blocks of step
    3, and zero where that offset is absent; this is the pure block Toeplitz matrix whose
    finite sections the source analyses, without the boundary perturbations of the actual
    Petrov-Galerkin matrix.

    Args:
        p (int): polynomial degree of the spline space, at least 1.
        r (int): regularity of the spline space, an integer with 0 <= r <= p - 1.
        rho (float or Fraction): non-negative parameter of the wave scheme.
        nblocks (int): number of block rows and block columns, at least 1.

    Returns:
        numpy.ndarray of shape (N nblocks, N nblocks), N = p - r, the finite section of
        the block Toeplitz operator.

    Raises:
        ValueError: if p < 1 or r is outside 0 <= r <= p - 1.
        ValueError: if rho is negative or nblocks < 1.
    """
    return None
```

### Step 8

exact_condition_number

Goal
----
Return the base-10 logarithm of the spectral condition number of the block Toeplitz section of the previous step, together with its largest singular value and the base-10 logarithm of its smallest singular value. The condition numbers of interest exceed the reciprocal of the double precision unit roundoff by many orders of magnitude, so a floating-point singular value decomposition of the section cannot deliver them: the reciprocal of the smallest singular value has to be obtained as the spectral norm of the exactly computed inverse, whose rational entries are only converted to floating point at the end, or by an equivalent exact or extended-precision route, and the result must be accurate to a relative error of 1e-10. Reject invalid p, r, a negative rho, nblocks < 1, and a singular section.

```python
def exact_condition_number(p, r, rho, nblocks):
    """Return the base-10 logarithm of the spectral condition number of the block Toeplitz
    section of the previous step, together with its largest singular value and the base-10
    logarithm of its smallest singular value.

    Args:
        p (int): polynomial degree of the spline space, at least 1.
        r (int): regularity of the spline space, an integer with 0 <= r <= p - 1.
        rho (float or Fraction): non-negative parameter of the wave scheme.
        nblocks (int): number of blocks of the section, at least 1.

    Returns:
        numpy.ndarray of shape (3,): log10 of the spectral condition number, the largest
        singular value, and log10 of the smallest singular value of the section.

    Raises:
        ValueError: if p < 1 or r is outside 0 <= r <= p - 1.
        ValueError: if rho is negative or nblocks < 1.
        RuntimeError: if the section is singular.
    """
    return None
```

### Step 9

growth_factors

Goal
----
Return, for every block count nb in nb_list, the ratio of the spectral condition number of the section with nb + 1 blocks to that of the section with nb blocks, each condition number computed as in the previous step. Reject invalid p, r, a negative rho or a block count below 1.

```python
def growth_factors(p, r, rho, nb_list):
    """Return, for every block count nb in nb_list, the ratio of the spectral condition number
    of the section with nb + 1 blocks to that of the section with nb blocks, each condition
    number computed as in the previous step.

    Args:
        p (int): polynomial degree of the spline space, at least 1.
        r (int): regularity of the spline space, an integer with 0 <= r <= p - 1.
        rho (float or Fraction): non-negative parameter of the wave scheme.
        nb_list (array_like of int): block counts, each at least 1.

    Returns:
        numpy.ndarray of shape (len(nb_list),) with the ratios of the condition numbers
        of the sections with nb + 1 and nb blocks.

    Raises:
        ValueError: if p < 1 or r is outside 0 <= r <= p - 1.
        ValueError: if rho is negative or a block count is below 1.
    """
    return None
```

### Step 10

iga_conditioning_audit

Goal
----
Assemble the conditioning audit of the wave scheme with degree p and regularity r at parameter rho, calling the earlier steps. Row 0: the first three growth thresholds of (p, r) in increasing order (zero when fewer exist) and the number of thresholds. Row 1: the numbers of zeros of det S_rho inside, on and outside the unit circle, and the asymptotic growth factor per block, the reciprocal of the largest inside modulus (zero when there is none). Row 2: log10 of the spectral condition number of the section with nb_final blocks, its largest singular value, log10 of its smallest singular value, and the ratio of the condition numbers of the sections with nb_final and nb_final - 1 blocks. Row 3: the largest threshold of the (2, 0) scheme, the largest threshold of the (3, 0) scheme, the smallest threshold of the (3, 1) scheme, all recomputed by the same chain as a check against the source, and the (1, 1) entry of the section with two blocks of the (p, r) scheme at rho. Raise an error if the (2, 0) chain does not return exactly three thresholds ending at 60, or if any entry is not finite. Reject invalid p, r, a negative rho or nb_final < 3.

```python
def iga_conditioning_audit(p, r, rho, nb_final):
    """Assemble the conditioning audit of the wave scheme with degree p and regularity r at
    parameter rho, calling the earlier steps.

    Args:
        p (int): polynomial degree of the spline space, at least 1.
        r (int): regularity of the spline space, an integer with 0 <= r <= p - 1.
        rho (float or Fraction): non-negative parameter of the wave scheme.
        nb_final (int): number of blocks of the reported section, at least 3.

    Returns:
        numpy.ndarray of shape (4, 4), the audit described in the docstring, with log10
        of the condition number of the reported section in row 2, column 0.

    Raises:
        ValueError: if p < 1 or r is outside 0 <= r <= p - 1.
        ValueError: if rho is negative or nb_final < 3.
        RuntimeError: if the source's (2, 0) thresholds are not reproduced or the audit
            is not finite.
    """
    return None
```
