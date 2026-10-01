# Chemistry-Computational_Chemistry-18

## Background

Simulations of condensed matter under periodic boundary conditions rest on a sum that does not converge absolutely. The electrostatic potential at an ion of a periodic crystal is accumulated over an infinite array of replicated cells, and because the Coulomb interaction falls off as the inverse distance while the number of cells in a shell grows as the square of it, the terms of the series do not decay fast enough for the sum to be independent of the order in which they are taken. A conditionally convergent series has no value until an order of summation is prescribed.

The physically meaningful way to prescribe that order is geometric: one names the shape of the macroscopic sample and lets it grow with its proportions held fixed. Different shapes then define different limits for the same lattice, and the difference between them is not an artefact of the arithmetic but a real statement about long-ranged interactions, which allow the outermost surface of a sample to retain influence over its centre no matter how large the sample becomes.

Once the geometry is fixed, the potential separates into parts of quite different character. One part belongs to the lattice alone and is blind to the shape and the size of the sample. A second is controlled entirely by the proportions of the outer surface and survives the macroscopic limit, because the surface charges of a polarized sample act on the centre through a field that does not weaken as the sample grows. A third reflects only the incompleteness of a finite sample and disappears as it grows, at a rate that depends on how the terms of a multipole expansion of the omitted cells happen to cancel for the unit cell in use.

That last point has a practical consequence which is easy to miss. Two unit cells describing the same crystal may differ in which of the leading corrections vanish, so the same structure can appear to converge rapidly under one description of its repeating unit and slowly under another. Judging a summation scheme therefore means asking which contributions it removes explicitly and which it leaves to cancel by accident of the cell chosen.

## Problem

A crystal assembled by repeating a neutral unit cell has no single well-defined electrostatic potential at one of its ions until the shape of the macroscopic sample is named. The Coulomb interaction weakens as the inverse distance while the number of cells at a given distance grows as its square, so the lattice sum converges only conditionally: its value depends on the order in which the cells are accumulated, and that order is fixed by the geometry of the sample rather than by the lattice. Samples of different shape built on the same lattice therefore reach different limits, and the difference survives however large they become.

Work with a simple cubic lattice of constant l = 1 in which every unit cell carries one positive and one negative unit charge: the negative charge sits at the cell origin and the positive charge is displaced from it by r = (0.37, 0.21, 0.13), in units of the lattice constant. The quantity of interest throughout is the electrostatic potential experienced by the negative ion of the central cell, in units where the Coulomb prefactor is one, so that a pair of unit charges at separation d contributes 1/d. Summing over a finite sample means adding, for every cell of it, the attraction to the displaced positive charge and the repulsion from the negative charge at that cell's origin, the central cell contributing only its own displaced partner.

Two sample shapes are used. Both are parallelepipeds centred on the central cell and grown at fixed proportions through a size index p: a cell with lattice vector n belongs to the sample when the solution u of E u = n has every component within (2p+1)/2 in magnitude, where the columns of the matrix E are the three edge directions. The first sample is the cube, E = I. The second is oblique, with edge directions (7, 0, 0), (3, 5, 0) and (0, 0, 3) as the columns of E, so that two of its three edges are not perpendicular; it holds 105 (2p+1)^3 cells.

As a sample grows at fixed proportions the sum separates into three parts: a bulk contribution which is the same for every sample shape and size; a contribution fixed entirely by the proportions of the sample's outer surface, independent of its absolute size; and a part that vanishes as the sample grows. The surface-fixed part is the interaction of the central dipole with the uniform polarization of the sample, one dipole per cell: nu_b(r|Omega) = (1/2V) * integral over Omega of d^3x (r . grad)^2 (1/|x|), where V is the volume of the unit cell (here V = l^3 = 1), Omega is the region the sample occupies, and the value does not depend on the size of the sample. For a sample with three equal perpendicular edges it equals -2 pi (r . r) / (3 V). The vanishing part is the leading size-dependent term obtained by expanding the contribution of every cell omitted from the finite sample in multipoles about the origin and replacing the sum over the omitted cells by an integral over the region outside the sample, one cell per volume V; for the cube this construction gives [24 (r . r)^2 - 40 (x^4 + y^4 + z^4)] / [9 sqrt(3) (2p+1)^2 l^5], where x, y and z are the components of r.

Obtain the bulk contribution from the cubic sample at size index p = 8 by removing the surface-fixed part and the vanishing part, and check the construction by taking the displacement (0.5, 0.5, 0.5) instead: expressed in units of the nearest-neighbour separation, the bulk contribution must then reproduce the Madelung constant of the caesium chloride structure, 1.76267477307098. Determine the surface-fixed part belonging to the oblique sample and the macroscopic limit of its potential. Then examine how the direct sum over the oblique sample approaches that limit. Determine the coefficient of (2p+1)^(-2) in the vanishing part as the multipole construction described above predicts it for the oblique sample, and determine the actual coefficient, defined as the limit of (2p+1)^2 [nu(r, p | oblique) - nu(r, infinity | oblique)] as p grows. The final answer is the amount by which the actual coefficient differs from the multipole prediction, actual minus predicted, to three significant figures. If the two differ, explain where the difference comes from, and support the explanation with a control calculation in which the effect you identify should be absent and with a direct evaluation of that effect.

Your reasoning should also report, as evidence that the source was read and the chain was executed: the closed form available in the literature for the surface-fixed part of a sample whose three edges are mutually perpendicular, together with its agreement with your general evaluation for edge lengths 7, 5 and 3; the most general sample shape for which such a closed form is available and whether the oblique sample is covered by it; the sign the surface-fixed part can take for any sample shape, and what its average over the orientations of the displacement equals; what the bulk contribution corresponds to in the standard treatment of periodic electrostatics and how the surface-fixed part appears in that treatment; the surface-fixed part of the oblique sample and its macroscopic potential, both to six significant figures; the Madelung check; the two coefficients to three significant figures; and the direction from which the direct sums approach the macroscopic value.

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

Implement **all 11 functions** below into a single file at `/app/solution.py`. Requirements:

- Use the EXACT function name and signature shown for each step.
- Define every function at module scope (importable).
- Do NOT include tests, example usage, prints, or a `__main__` block.
- Available libraries: numpy, scipy, sympy, pandas, scikit-learn, networkx.

Start your file with these imports:

```python
import numpy as np
```

### Step 1

lattice_census

Goal
----
Enumerates the lattice vectors of a finite crystal of a given size index and edge matrix and reports its census.

```python
def lattice_census(p: int, E: "np.ndarray") -> "np.ndarray":
    """Enumerates the lattice vectors of a finite crystal of a given size index and edge matrix and reports its census.

    Args:
        p: positive integer, the size index of the sample; the census needs at least one cell besides the
            central one, so p = 0 is rejected.
        E: array-like of shape (3, 3) holding integer values in any numeric dtype, with non-zero
            determinant, whose COLUMNS are the edge directions of the sample; a cell n belongs to the sample
            when every component of the solution u of E u = n satisfies |u_i| <= (2p+1)/2.

    Returns:
        A numpy float64 array of shape (6,): the number of lattice vectors of the sample excluding the origin,
        the cell count |det E| (2p+1)^3 predicted by the volume of the sample (origin included), the largest
        vector norm, the sum of the inverse norms, and the largest absolute cell index along x and along z.

    Raises:
        ValueError: if p is not a positive integer; or if E is not a 3x3 array of integer values with
            non-zero determinant.
    """
    return None
```

### Step 2

finite_lattice_sum

Goal
----
Evaluates the electrostatic potential at the reference ion of a finite crystal of oppositely charged pairs.

```python
def finite_lattice_sum(r: "np.ndarray", p: int, E: "np.ndarray", l: float) -> "np.ndarray":
    """Evaluates the electrostatic potential at the reference ion of a finite crystal of oppositely charged pairs.

    Args:
        r: array-like of shape (3,), the displacement of the positive charge from the negative one, an
            absolute length in the same units as l, not a multiple of the lattice constant (with l = 2 and
            r = (0.25, 0, 0) the pair separation is 0.25 and the displacement is one eighth of the lattice
            constant); must be non-zero.
        p: non-negative integer, the size index of the sample; p = 0 is the central cell alone.
        E: array-like of shape (3, 3) of integer values with non-zero determinant, whose columns are the
            edge directions of the sample, with the membership rule of the previous step.
        l: positive float, the cubic lattice constant.

    Returns:
        A numpy float64 array of shape (3,), in inverse length units, the reciprocal of the unit in which r and l are given (the absolute potential, which for l = 1 coincides with units of 1/l): the sum for size index p, the sum for size index p-1
        (equal to the first entry when p is zero), and their difference.

    Raises:
        ValueError: if r is not a finite real vector of length three or has zero length; if p is not a
            non-negative integer; if E is not a 3x3 array of integer values with non-zero determinant; or if l
            is not a positive finite real number.
    """
    return None
```

### Step 3

boundary_term

Goal
----
Evaluates the non-periodic boundary contribution that the shape of a macroscopic crystal imposes on the reference ion.

```python
def boundary_term(r: "np.ndarray", E: "np.ndarray", V: float) -> "np.ndarray":
    """Evaluates the non-periodic boundary contribution that the shape of a macroscopic crystal imposes on the reference ion.

    Args:
        r: array-like of shape (3,), the displacement of the positive charge from the negative one, in the
            same length units as the cube root of V.
        E: array-like of shape (3, 3) of integer values with non-zero determinant, whose columns are the
            edge directions of the sample; only the proportions of the parallelepiped they span matter.
        V: positive float, the volume of the unit cell.

    Returns:
        A numpy float64 array of shape (3,), in inverse length units, the reciprocal of the unit in which r is
        given: the boundary term for the sample spanned by the columns of E, the value -2 pi (r . r)/(3V) of a
        sample with three equal perpendicular edges, and their difference.

    Raises:
        ValueError: if r is not a finite real vector of length three; if E is not a 3x3 array of integer
            values with non-zero determinant; if V is not a positive finite real number; or if the boundary
            term comes out positive.
    """
    return None
```

### Step 4

finite_size_term

Goal
----
Evaluates the leading correction that separates a finite cubic crystal from the macroscopic limit of the same shape.

```python
def finite_size_term(r: "np.ndarray", p: int, l: float) -> "np.ndarray":
    """Evaluates the leading correction that separates a finite cubic crystal from the macroscopic limit of the same shape.

    Args:
        r: array-like of shape (3,), the displacement of the positive charge from the negative one, in the
            same length units as l.
        p: non-negative integer, the size index of the cubic sample.
        l: positive float, the cubic lattice constant.

    Returns:
        A numpy float64 array of shape (4,): the correction in inverse length units, the reciprocal of the unit in which r and l are given (the absolute potential, which for l = 1 coincides with units of 1/l), the correction multiplied by
        (2p+1)^2, the squared length of the displacement, and the sum of the fourth powers of its components.

    Raises:
        ValueError: if r is not a finite real vector of length three, if p is not a non-negative integer,
            or if l is not a positive finite real number.
    """
    return None
```

### Step 5

bulk_pair_potential

Goal
----
Extracts the shape-independent bulk pair potential from the finite lattice sum of a cubic sample.

```python
def bulk_pair_potential(r: "np.ndarray", p: int, l: float) -> "np.ndarray":
    """Extracts the shape-independent bulk pair potential from the finite lattice sum of a cubic sample.

    Args:
        r: array-like of shape (3,), the displacement of the positive charge from the negative one, in the
            same length units as l; must be non-zero.
        p: positive integer, the size index of the cubic sample.
        l: positive float, the cubic lattice constant.

    Returns:
        A numpy float64 array of shape (5,), in inverse length units, the reciprocal of the unit in which r and l are given (the absolute potential, which for l = 1 coincides with units of 1/l): the bulk pair potential, the finite lattice sum
        of the cubic sample it came from, the boundary term removed, the finite-size term removed, and the
        change in the bulk pair potential when the size index is reduced by one.

    Raises:
        ValueError: if r is not a finite real vector of length three or has zero length, if p is not a
            positive integer, or if l is not a positive finite real number.
    """
    return None
```

### Step 6

madelung_constant

Goal
----
Evaluates the Madelung constant of the caesium chloride structure from the corrected direct summation.

```python
def madelung_constant(p: int) -> "np.ndarray":
    """Evaluates the Madelung constant of the caesium chloride structure from the corrected direct summation.

    Args:
        p: positive integer, the size index of the cubic sample used for the extraction.

    Returns:
        A numpy float64 array of shape (5,), all in units of the nearest-neighbour separation: the Madelung
        constant, its deviation from the reference value 1.76267477307098, and the three signed contributions
        that add up to the constant, namely the finite lattice sum, the negative of the boundary term (a
        positive number here) and the negative of the finite-size term.

    Raises:
        ValueError: if p is not a positive integer.
    """
    return None
```

### Step 7

infinite_crystal_potential

Goal
----
Evaluates the potential of a macroscopic crystal of prescribed proportions from the bulk and boundary parts and compares the finite sum of the same shape with it.

```python
def infinite_crystal_potential(r: "np.ndarray", E: "np.ndarray", p: int, l: float) -> "np.ndarray":
    """Evaluates the potential of a macroscopic crystal of prescribed proportions from the bulk and boundary parts and compares the finite sum of the same shape with it.

    Args:
        r: array-like of shape (3,), the displacement of the positive charge from the negative one, in the
            same length units as l; must be non-zero.
        E: array-like of shape (3, 3) of integer values with non-zero determinant, whose columns are the
            edge directions of the sample and so fix the proportions of the macroscopic sample.
        p: positive integer, the size index used both for the bulk extraction from the cubic sample and for
            the finite sum of the prescribed shape.
        l: positive float, the cubic lattice constant.

    Returns:
        A numpy float64 array of shape (5,), in inverse length units, the reciprocal of the unit in which r and l are given (the absolute potential, which for l = 1 coincides with units of 1/l): the macroscopic potential
        for the prescribed proportions, the bulk part, the boundary part of that shape, the finite lattice
        sum of the same shape at size index p, and the difference between that sum and the macroscopic value
        multiplied by (2p+1)^2.

    Raises:
        ValueError: if r is not a finite real vector of length three or has zero length; if E is not a 3x3
            array of integer values with non-zero determinant; if p is not a positive integer; if l is not a
            positive finite real number; or if the macroscopic potential comes out non-positive.
    """
    return None
```

### Step 8

multipole_coefficient

Goal
----
Evaluates the coefficient of the leading finite-size term that the multipole construction predicts for a sample of prescribed proportions.

```python
def multipole_coefficient(r: "np.ndarray", E: "np.ndarray", V: float) -> "np.ndarray":
    """Evaluates the coefficient of the leading finite-size term that the multipole construction predicts for a sample of prescribed proportions.

    Args:
        r: array-like of shape (3,), the displacement of the positive charge from the negative one, in the
            same length units as the cube root of V.
        E: array-like of shape (3, 3) of integer values with non-zero determinant, whose columns are the
            edge directions of the sample; only the proportions matter.
        V: positive float, the volume of the unit cell.

    Returns:
        A numpy float64 array of shape (4,), in inverse length units, the reciprocal of the unit in which r is given: the predicted coefficient
        of (2p+1)^(-2) for the sample spanned by the columns of E, the same construction applied to the cube,
        the closed-form cube coefficient of the finite-size step (its correction multiplied by (2p+1)^2), and
        the difference of the last two.

    Raises:
        ValueError: if r is not a finite real vector of length three; if E is not a 3x3 array of integer
            values with non-zero determinant; or if V is not a positive finite real number.
    """
    return None
```

### Step 9

finite_size_fit

Goal
----
Measures the actual leading finite-size coefficient of a sample from its direct sums at several sizes, and the corresponding coefficient of the second-order multipole term alone.

```python
def finite_size_fit(r: "np.ndarray", E: "np.ndarray", sizes: "np.ndarray", p_bulk: int, l: float) -> "np.ndarray":
    """Measures the actual leading finite-size coefficient of a sample from its direct sums at several sizes, and the corresponding coefficient of the second-order multipole term alone.

    Args:
        r: array-like of shape (3,), the displacement of the positive charge from the negative one, in the
            same length units as l; must be non-zero.
        E: array-like of shape (3, 3) of integer values with non-zero determinant, whose columns are the
            edge directions of the sample.
        sizes: array-like of at least three distinct positive integers, the size indices at which the
            sample is summed.
        p_bulk: positive integer, the size index of the cubic sample from which the bulk part of the
            macroscopic value is extracted.
        l: positive float, the cubic lattice constant.

    Returns:
        A numpy float64 array of shape (5,), in inverse length units, the reciprocal of the unit in which r and l are given (the absolute potential, which for l = 1 coincides with units of 1/l): the least-squares coefficients a and b of
        a (2p+1)^(-2) + b (2p+1)^(-4) fitted to the residuals, finite sum minus macroscopic value, over the
        given sizes; the least-squares coefficients q2 and q4 of q0 + q2 (2p+1)^(-2) + q4 (2p+1)^(-4)
        fitted to the second-order lattice sums over the same sizes; and q0.

    Raises:
        ValueError: if r is not a finite real vector of length three or has zero length; if E is not a 3x3
            array of integer values with non-zero determinant; if sizes holds fewer than three entries or any
            entry that is not a distinct positive integer; if p_bulk is not a positive integer; or if l is
            not a positive finite real number.
    """
    return None
```

### Step 10

termination_coefficient

Goal
----
Evaluates analytically the coefficient of the leading finite-size term that the termination of a sheared sample adds to the multipole construction.

```python
def termination_coefficient(r: "np.ndarray", E: "np.ndarray", l: float) -> "np.ndarray":
    """Evaluates analytically the coefficient of the leading finite-size term that the termination of a sheared sample adds to the multipole construction.

    Args:
        r: array-like of shape (3,), the displacement of the positive charge from the negative one, in the
            same length units as l; must be non-zero.
        E: array-like of shape (3, 3) of integer values of the form [[a, b, 0], [0, c, 0], [0, 0, d]] with a,
            c and d odd positive integers and b any integer (b = 0 is a box); any other edge matrix is
            rejected.
        l: positive float, the cubic lattice constant.

    Returns:
        A numpy float64 array of shape (2,), in inverse length units, the reciprocal of the unit in which r
        and l are given: the (2p+1)^(-2) coefficient of the second-order lattice sum over the sample minus
        its limit, from the Euler-Maclaurin evaluation, and the same quantity for the box with edge lengths
        a, c, d (b = 0), which vanishes.

    Raises:
        ValueError: if r is not a finite real vector of length three or has zero length; if E is not a 3x3
            array of integer values of the stated form with odd positive a, c, d; or if l is not a positive
            finite real number.
    """
    return None
```

### Step 11

lattice_audit

Goal
----
Runs the whole chain on the benchmark configuration and returns the audit of the finite Coulomb lattice sum of a sheared sample.

```python
def lattice_audit(r: "np.ndarray", E: "np.ndarray", p: int, p_bulk: int, sizes: "np.ndarray", l: float) -> "np.ndarray":
    """Runs the whole chain on the benchmark configuration and returns the audit of the finite Coulomb lattice sum of a sheared sample.

    Args:
        r: array-like of shape (3,), the displacement of the positive charge from the negative one, in the
            same length units as l; must be non-zero.
        E: array-like of shape (3, 3) of integer values with non-zero determinant, whose columns are the
            edge directions of the sample.
        p: positive integer, the size index used for the reported bulk extraction, the finite sum, the
            scaled residual and the Madelung check.
        p_bulk: positive integer, the size index of the cubic extraction that fixes the macroscopic value
            inside the finite-size fits.
        sizes: array-like of at least three distinct positive integers, the size indices summed in the fits.
        l: positive float, the cubic lattice constant.

    Returns:
        A numpy float64 array of shape (18,) holding, in this order: [0] the termination contribution, the
        measured leading coefficient minus the multipole prediction, for the sample; [1] the measured leading
        coefficient; [2] the multipole prediction; [3] the (2p+1)^(-2) coefficient of the second-order sums
        of the sample; [4] the macroscopic potential of the sample; [5] its boundary term; [6] the bulk part
        from the cube at size index p; [7] the finite sum of the sample at size index p; [8] that sum minus
        the macroscopic value, multiplied by (2p+1)^2; [9] the caesium chloride Madelung constant at size
        index p; [10] the cube's multipole prediction minus its closed form; [11] the measured coefficient
        minus the multipole prediction for the box whose edge lengths are the absolute diagonal entries of E
        (the cube when that diagonal is singular); [12] that box's boundary term minus the arctangent closed
        form -(4/V) [x^2 arctan(bc/(ad)) + y^2 arctan(ac/(bd)) + z^2 arctan(ab/(cd))] with d the space
        diagonal; [13] the (2p+1)^(-4) coefficient of the sample's residual fit; [14] the number of cells of
        the sample at size index p, origin excluded; [15] the termination contribution evaluated analytically
        from the Euler-Maclaurin expansion of the sample's second-order sum; [16] entry [15] minus entry [0];
        [17] the analytic termination coefficient of the box of entry [11], which vanishes. Potentials and
        coefficients are in inverse length units, the reciprocal of the unit in which r and l are given (the
        absolute potential, which for l = 1 coincides with units of 1/l).

    Raises:
        ValueError: if any argument is invalid as in the earlier steps (E must have the sheared-box form of the
            previous step), or if the macroscopic potential comes out non-positive.
    """
    return None
```
