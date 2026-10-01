# Chemistry-Quantum_Chemistry-9

## Problem

A global hybrid density functional mixes one fixed fraction of exact exchange into a semilocal approximation everywhere in space, while a local hybrid lets that fraction vary from point to point, so the admixture stays small where the semilocal functional already works and grows where an electron's own charge is smeared over a region it should not occupy. Recent work proposes a minimally empirical local hybrid of this kind, built on a gradient-corrected semilocal exchange functional and on a measure of how far a model exchange hole falls short of holding a whole electron; its two empirical parameters were fixed by minimising weighted errors over a large thermochemical benchmark, so they cannot be derived and can only be read off and used.

Consider a lithium atom carrying a charge of one half, the fragment a singly ionised lithium dimer dissociates into, described by two spherically symmetric real orbitals that are functions of the radius alone. The first is built from exp(-4.70 r) and exp(-2.45 r), each of those two scaled on its own so that its square integrates to one over all space and only then combined with weights 0.15 and 0.90, with the combination scaled the same way at the end; the second is r exp(-0.66 r), first made orthogonal to that scaled first orbital under the integral over all space and only then scaled the same way. Put one electron in the first orbital and one half of an electron in the second for the majority spin, and one electron in the first orbital and nothing in the second for the minority spin, and work throughout in hartree atomic units.

Evaluate the exact-exchange admixture energy of that local hybrid for this density: summed over the two spins, the integral over all space of the position-dependent mixing fraction times the exact exchange energy density of that spin minus its semilocal exchange energy density. Take the mixing fraction, the semilocal functional it is built on, and both empirical parameters exactly as the source defines them, refitting nothing, and cap at one any effective hole normalisation that inverting the model hole would place above one, capping it before the two spin channels are combined with each other. Refine the radial quadrature until the answer stops moving in its sixth significant figure, and report that admixture energy in hartree.

State the conventions you adopted at each point where the framework leaves a choice open, justifying each from the source literature, and in your reasoning also report five further quantities for the same density: the number of electrons in each of the two spin channels, the exact exchange energy of the atom in hartree, the semilocal exchange energy of the atom in hartree, the part of the admixture energy that the majority-spin channel contributes on its own in hartree, and the admixture energy divided by the difference between those two exchange energies.

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

radial_quadrature

Goal
----
Build a quadrature for integrals over the radial half line. Take the n_nodes Gauss-Legendre nodes and weights of the interval from minus one to one, map them onto the open unit interval by replacing each node t with (t+1)/2 and halving each weight, and send a mapped node u to the radius scale*u/(1-u). The quadrature weight belonging to that radius is the halved Gauss-Legendre weight times scale/(1-u)**2, so that the sum over nodes of weight times f(radius) approximates the integral of f from zero to infinity. Keep the nodes in the order numpy's Gauss-Legendre routine returns them, which puts the smallest radius first.

```python
def radial_quadrature(n_nodes: int, scale: float) -> "np.ndarray":
    '''Build a quadrature for integrals over the radial half line. Take the n_nodes Gauss-Legendre nodes and weights of the interval from minus one to one, map them onto the open unit interval by replacing each node t with (t+1)/2 and halving each weight, and send a mapped node u to the radius scale*u/(1-u). The quadrature weight belonging to that radius is the halved Gauss-Legendre weight times scale/(1-u)**2, so that the sum over nodes of weight times f(radius) approximates the integral of f from zero to infinity. Keep the nodes in the order numpy's Gauss-Legendre routine returns them, which puts the smallest radius first.

    Parameters
    ----------
    n_nodes : int
        Number of Gauss-Legendre nodes, at least one.
    scale : float
        Positive length in bohr setting where the mapped nodes concentrate.

    Returns
    -------
    grid : np.ndarray
        ndarray of shape (2, n_nodes): row 0 the radii in bohr, row 1 the quadrature weights in bohr.

    Raises
    ------
    ValueError
        if n_nodes is smaller than one, or if scale is not positive.
    '''
    return grid  # placeholder
```

### Step 2

orthonormal_orbitals

Goal
----
Build two orthonormal radial orbitals out of Slater primitives and return them as one padded coefficient table. A primitive carrying label one is sqrt(z**3/pi)*exp(-z*r) and a primitive carrying label two is sqrt(z**5/(3*pi))*r*exp(-z*r), each already scaled so that the integral of its square over all space is one. Define the overlap of two radial functions as the integral from zero to infinity of four pi r**2 times their product. The inner orbital is the sum of the label-one primitives built from inner_exponents weighted by inner_coefficients, divided by the square root of its own overlap with itself. The outer orbital starts as the single label-two primitive built from outer_exponent, then has the inner orbital subtracted from it once, weighted by the overlap of that starting primitive with the already-scaled inner orbital, and is only then divided by the square root of its own overlap with itself. The inner orbital is never modified, so the order of those two operations is part of the contract. Return an array whose first index selects the orbital with the inner one first, whose second index runs over terms, and whose third index holds the primitive label, the exponent and the coefficient in that order. List the inner orbital's terms in the order of inner_exponents, list the outer orbital's label-two term first and its label-one terms after it in that same order, and pad the inner orbital with trailing rows of all zeros so both orbitals carry the same number of terms.

```python
def orthonormal_orbitals(inner_exponents: "np.ndarray", inner_coefficients: "np.ndarray", outer_exponent: float) -> "np.ndarray":
    '''Build two orthonormal radial orbitals out of Slater primitives and return them as one padded coefficient table. A primitive carrying label one is sqrt(z**3/pi)*exp(-z*r) and a primitive carrying label two is sqrt(z**5/(3*pi))*r*exp(-z*r), each already scaled so that the integral of its square over all space is one. Define the overlap of two radial functions as the integral from zero to infinity of four pi r**2 times their product. The inner orbital is the sum of the label-one primitives built from inner_exponents weighted by inner_coefficients, divided by the square root of its own overlap with itself. The outer orbital starts as the single label-two primitive built from outer_exponent, then has the inner orbital subtracted from it once, weighted by the overlap of that starting primitive with the already-scaled inner orbital, and is only then divided by the square root of its own overlap with itself. The inner orbital is never modified, so the order of those two operations is part of the contract. Return an array whose first index selects the orbital with the inner one first, whose second index runs over terms, and whose third index holds the primitive label, the exponent and the coefficient in that order. List the inner orbital's terms in the order of inner_exponents, list the outer orbital's label-two term first and its label-one terms after it in that same order, and pad the inner orbital with trailing rows of all zeros so both orbitals carry the same number of terms.

    Parameters
    ----------
    inner_exponents : np.ndarray
        Positive Slater exponents of the inner orbital's label-one primitives, in inverse bohr.
    inner_coefficients : np.ndarray
        Weights of those primitives, same length as inner_exponents.
    outer_exponent : float
        Positive Slater exponent of the outer orbital's label-two primitive, in inverse bohr.

    Returns
    -------
    table : np.ndarray
        ndarray of shape (2, len(inner_exponents)+1, 3): orbital, term, then primitive label, Slater exponent and coefficient.

    Raises
    ------
    ValueError
        if inner_exponents and inner_coefficients have different lengths or are empty, or if any Slater exponent is not positive.
    '''
    return table  # placeholder
```

### Step 3

spin_channel_fields

Goal
----
Evaluate the five radial fields that one spin channel contributes, at each radius. A row (p, z, w) of an orbital table contributes w times the primitive it names, where label p equal to one means sqrt(z**3/pi)*exp(-z*r) and label p equal to two means sqrt(z**5/(3*pi))*r*exp(-z*r); a row whose coefficient is zero is padding and contributes nothing, whatever its other entries. Write phi_k for the k-th orbital of the table, taken as that sum over its rows, n_k for its occupation and a prime for d/dr. The density is the sum over k of n_k phi_k**2. The gradient is the sum over k of 2 n_k phi_k phi_k'. The Laplacian is the sum over k of 2 n_k times (phi_k (phi_k'' + 2 phi_k'/r) + phi_k'**2). The orbital kinetic density is the sum over k of n_k phi_k'**2, carrying no factor of one half. The fifth field is the model-hole curvature: the Laplacian minus twice (the orbital kinetic density minus a quarter of the gradient squared divided by the density), the whole difference divided by six. Return the five fields as rows in exactly that order.

```python
def spin_channel_fields(orbitals: "np.ndarray", occupations: "np.ndarray", radii: "np.ndarray") -> "np.ndarray":
    '''Evaluate the five radial fields that one spin channel contributes, at each radius. A row (p, z, w) of an orbital table contributes w times the primitive it names, where label p equal to one means sqrt(z**3/pi)*exp(-z*r) and label p equal to two means sqrt(z**5/(3*pi))*r*exp(-z*r); a row whose coefficient is zero is padding and contributes nothing, whatever its other entries. Write phi_k for the k-th orbital of the table, taken as that sum over its rows, n_k for its occupation and a prime for d/dr. The density is the sum over k of n_k phi_k**2. The gradient is the sum over k of 2 n_k phi_k phi_k'. The Laplacian is the sum over k of 2 n_k times (phi_k (phi_k'' + 2 phi_k'/r) + phi_k'**2). The orbital kinetic density is the sum over k of n_k phi_k'**2, carrying no factor of one half. The fifth field is the model-hole curvature: the Laplacian minus twice (the orbital kinetic density minus a quarter of the gradient squared divided by the density), the whole difference divided by six. Return the five fields as rows in exactly that order.

    Parameters
    ----------
    orbitals : np.ndarray
        Padded coefficient table of shape (n_orbitals, n_terms, 3); each row is primitive label, exponent, coefficient.
    occupations : np.ndarray
        Occupation of each orbital, each between zero and one.
    radii : np.ndarray
        Strictly positive radii in bohr at which to evaluate.

    Returns
    -------
    fields : np.ndarray
        ndarray of shape (5, len(radii)): density, gradient, Laplacian, orbital kinetic density and model-hole curvature, in hartree atomic units.

    Raises
    ------
    ValueError
        if orbitals does not have shape (n_orbitals, n_terms, 3), if orbitals and occupations describe different numbers of orbitals, if any occupation lies outside zero to one, or if any radius is not positive.
    '''
    return fields  # placeholder
```

### Step 4

b86b_exchange_density

Goal
----
Return the gradient-corrected semilocal exchange energy density of one spin channel, evaluated pointwise. Write rho for the density and x for the absolute gradient divided by rho**(4/3). The value is minus (3/2)*(3/(4*pi))**(1/3) times rho**(4/3), minus 0.00375 times rho**(4/3) times x**2 divided by (1 + 0.007*x**2)**(4/5). The channel enters on its own and the two channels are summed only later, so no spin-scaling factor is applied here.

```python
def b86b_exchange_density(density: "np.ndarray", gradient: "np.ndarray") -> "np.ndarray":
    '''Return the gradient-corrected semilocal exchange energy density of one spin channel, evaluated pointwise. Write rho for the density and x for the absolute gradient divided by rho**(4/3). The value is minus (3/2)*(3/(4*pi))**(1/3) times rho**(4/3), minus 0.00375 times rho**(4/3) times x**2 divided by (1 + 0.007*x**2)**(4/5). The channel enters on its own and the two channels are summed only later, so no spin-scaling factor is applied here.

    Parameters
    ----------
    density : np.ndarray
        Strictly positive spin density in inverse bohr cubed.
    gradient : np.ndarray
        Radial derivative of that density, same length.

    Returns
    -------
    energy_density : np.ndarray
        ndarray of shape (len(density),): the semilocal exchange energy density of that spin channel, in hartree per bohr cubed.

    Raises
    ------
    ValueError
        if density and gradient have different lengths, or if any density value is not positive.
    '''
    return energy_density  # placeholder
```

### Step 5

exact_exchange_density

Goal
----
Return the exact exchange energy density of one spin channel, in the gauge that takes the integrand as it stands with no function added to it. A row (p, z, w) of an orbital table contributes w times the primitive it names, where label p equal to one means sqrt(z**3/pi)*exp(-z*r) and label p equal to two means sqrt(z**5/(3*pi))*r*exp(-z*r); a row whose coefficient is zero is padding and contributes nothing, whatever its other entries. The value at radius r is minus one half of the sum over every ordered pair (i, j) of orbitals of n_i n_j phi_i(r) phi_j(r) times the electrostatic potential at r of the product density phi_i phi_j. Fractional occupations enter as the plain product n_i n_j and not as its square root, which is what makes a single fractionally occupied orbital cancel its own classical repulsion exactly. Every product of two of these orbitals is spherically symmetric, and the potential at radius r of a spherical density f is four pi times the quantity (the integral from zero to r of s**2 f(s) ds, divided by r, plus the integral from r to infinity of s f(s) ds).

```python
def exact_exchange_density(orbitals: "np.ndarray", occupations: "np.ndarray", radii: "np.ndarray") -> "np.ndarray":
    '''Return the exact exchange energy density of one spin channel, in the gauge that takes the integrand as it stands with no function added to it. A row (p, z, w) of an orbital table contributes w times the primitive it names, where label p equal to one means sqrt(z**3/pi)*exp(-z*r) and label p equal to two means sqrt(z**5/(3*pi))*r*exp(-z*r); a row whose coefficient is zero is padding and contributes nothing, whatever its other entries. The value at radius r is minus one half of the sum over every ordered pair (i, j) of orbitals of n_i n_j phi_i(r) phi_j(r) times the electrostatic potential at r of the product density phi_i phi_j. Fractional occupations enter as the plain product n_i n_j and not as its square root, which is what makes a single fractionally occupied orbital cancel its own classical repulsion exactly. Every product of two of these orbitals is spherically symmetric, and the potential at radius r of a spherical density f is four pi times the quantity (the integral from zero to r of s**2 f(s) ds, divided by r, plus the integral from r to infinity of s f(s) ds).

    Parameters
    ----------
    orbitals : np.ndarray
        Padded coefficient table of shape (n_orbitals, n_terms, 3), as returned by the orbital step.
    occupations : np.ndarray
        Occupation of each orbital, each between zero and one.
    radii : np.ndarray
        Strictly positive radii in bohr at which to evaluate.

    Returns
    -------
    energy_density : np.ndarray
        ndarray of shape (len(radii),): the exact exchange energy density of that spin channel, in hartree per bohr cubed.

    Raises
    ------
    ValueError
        if orbitals does not have shape (n_orbitals, n_terms, 3), if orbitals and occupations describe different numbers of orbitals, if any occupation lies outside zero to one, or if any radius is not positive.
    '''
    return energy_density  # placeholder
```

### Step 6

effective_hole_normalisation

Goal
----
Invert a model exchange hole pointwise to find the normalisation it would need in order to reproduce a given exact exchange energy density. Define first the model potential P(rho, Q, N) at the reference point of a hole of normalisation N sitting in a density rho with curvature Q: solve x*exp(-2*x/3)/(x-2) = (2/3)*(pi*rho/N)**(2/3)*rho/Q for x, taking the unique root larger than two when the right-hand side is positive and the unique root between zero and two when it is negative; then set alpha = (8*pi*rho*exp(x)/N)**(1/3) and b = x/alpha, and take P = -N*(1 - exp(-x) - x*exp(-x)/2)/b. The target at each point is twice the exact exchange energy density divided by the density. Solve P(rho, Q, N) equals that target for N, searching only normalisations between zero and one, and return one wherever no normalisation at or below one reaches the target. P is monotone in N over that interval, so the solution is unique wherever one exists, and the returned values therefore never exceed one.

```python
def effective_hole_normalisation(density: "np.ndarray", curvature: "np.ndarray", exact_exchange_energy_density: "np.ndarray") -> "np.ndarray":
    '''Invert a model exchange hole pointwise to find the normalisation it would need in order to reproduce a given exact exchange energy density. Define first the model potential P(rho, Q, N) at the reference point of a hole of normalisation N sitting in a density rho with curvature Q: solve x*exp(-2*x/3)/(x-2) = (2/3)*(pi*rho/N)**(2/3)*rho/Q for x, taking the unique root larger than two when the right-hand side is positive and the unique root between zero and two when it is negative; then set alpha = (8*pi*rho*exp(x)/N)**(1/3) and b = x/alpha, and take P = -N*(1 - exp(-x) - x*exp(-x)/2)/b. The target at each point is twice the exact exchange energy density divided by the density. Solve P(rho, Q, N) equals that target for N, searching only normalisations between zero and one, and return one wherever no normalisation at or below one reaches the target. P is monotone in N over that interval, so the solution is unique wherever one exists, and the returned values therefore never exceed one.

    Parameters
    ----------
    density : np.ndarray
        Strictly positive spin density in inverse bohr cubed.
    curvature : np.ndarray
        Model-hole curvature of the same channel, same length; zero is a valid value.
    exact_exchange_energy_density : np.ndarray
        Strictly negative exact exchange energy density of the same channel, same length.

    Returns
    -------
    normalisation : np.ndarray
        ndarray of shape (len(density),): the effective hole normalisation at each point, dimensionless and never above one.

    Raises
    ------
    ValueError
        if density, curvature and exact_exchange_energy_density do not all have the same length, if any density value is not positive, or if any exact exchange energy density is not negative. A curvature of exactly zero is valid and is handled as the limiting case, not rejected.
    '''
    return normalisation  # placeholder
```

### Step 7

xc_hole_normalisation

Goal
----
Combine the two spin channels' effective hole normalisations into the effective exchange-correlation hole normalisation of the same-spin channel, pointwise. Write a for the same-spin normalisation and b for the other-spin one. Take f as the least of three numbers: (1-a)/b, (1-b)/a, and one; treat a ratio whose denominator is zero as larger than the other two so that it never wins. Return a + f*b. The function is not symmetric in its two arguments, so calling it for the other channel means exchanging them.

```python
def xc_hole_normalisation(same_spin_norm: "np.ndarray", other_spin_norm: "np.ndarray") -> "np.ndarray":
    '''Combine the two spin channels' effective hole normalisations into the effective exchange-correlation hole normalisation of the same-spin channel, pointwise. Write a for the same-spin normalisation and b for the other-spin one. Take f as the least of three numbers: (1-a)/b, (1-b)/a, and one; treat a ratio whose denominator is zero as larger than the other two so that it never wins. Return a + f*b. The function is not symmetric in its two arguments, so calling it for the other channel means exchanging them.

    Parameters
    ----------
    same_spin_norm : np.ndarray
        Effective hole normalisation of the channel being combined, each between zero and one.
    other_spin_norm : np.ndarray
        Effective hole normalisation of the opposite channel, same length, each between zero and one.

    Returns
    -------
    normalisation : np.ndarray
        ndarray of shape (len(same_spin_norm),): the effective exchange-correlation hole normalisation of the same-spin channel, dimensionless.

    Raises
    ------
    ValueError
        if the two arrays have different lengths, or if any value lies outside zero to one.
    '''
    return normalisation  # placeholder
```

### Step 8

local_mixing_function

Goal
----
Return the position-dependent fraction of exact exchange the local hybrid mixes in, evaluated pointwise. Write rho for the density, eps for the semilocal exchange energy density of the same channel, and n for the exchange-correlation hole normalisation of the same channel. Form the length z as the absolute value of rho divided by eps. Form the dimensionless scaling factor s as b_param times the square of the sine of pi times n, plus one, so that s equals one wherever n is a whole number and rises to b_param plus one halfway between two whole numbers. The mixing fraction at each point is the error function of c_param times s times z, so the scaling factor multiplies the argument of the error function and not the function itself. Return one value per point, each between zero and one.

```python
def local_mixing_function(density: "np.ndarray", b86b_energy_density: "np.ndarray", xc_hole_norm: "np.ndarray", c_param: float, b_param: float) -> "np.ndarray":
    '''Return the position-dependent fraction of exact exchange the local hybrid mixes in, evaluated pointwise. Write rho for the density, eps for the semilocal exchange energy density of the same channel, and n for the exchange-correlation hole normalisation of the same channel. Form the length z as the absolute value of rho divided by eps. Form the dimensionless scaling factor s as b_param times the square of the sine of pi times n, plus one, so that s equals one wherever n is a whole number and rises to b_param plus one halfway between two whole numbers. The mixing fraction at each point is the error function of c_param times s times z, so the scaling factor multiplies the argument of the error function and not the function itself. Return one value per point, each between zero and one.

    Parameters
    ----------
    density : np.ndarray
        Strictly positive spin density in inverse bohr cubed.
    b86b_energy_density : np.ndarray
        Non-zero semilocal exchange energy density of the same channel, same length.
    xc_hole_norm : np.ndarray
        Effective exchange-correlation hole normalisation of the same channel, same length.
    c_param : float
        Positive coefficient multiplying the length inside the mixing function.
    b_param : float
        Non-negative constant scaling the hole-normalisation factor.

    Returns
    -------
    mixing : np.ndarray
        ndarray of shape (len(density),): the local exact-exchange mixing fraction, dimensionless and between zero and one.

    Raises
    ------
    ValueError
        if density, b86b_energy_density and xc_hole_norm do not all have the same length, if any density value is not positive, if any semilocal exchange energy density is zero, if c_param is not positive, or if b_param is negative.
    '''
    return mixing  # placeholder
```

### Step 9

channel_admixture

Goal
----
Return one spin channel's contribution to the exact-exchange admixture energy. It is four pi times the quadrature sum over the supplied nodes of weight times radius squared times the mixing fraction times (the exact exchange energy density minus the semilocal one) at that node. The difference is taken in that order, exact first, and the four pi times radius squared is the spherical measure that turns a radial quadrature into an integral over all space. Return a plain float.

```python
def channel_admixture(mixing: "np.ndarray", exact_density: "np.ndarray", semilocal_density: "np.ndarray", radii: "np.ndarray", weights: "np.ndarray") -> float:
    '''Return one spin channel's contribution to the exact-exchange admixture energy. It is four pi times the quadrature sum over the supplied nodes of weight times radius squared times the mixing fraction times (the exact exchange energy density minus the semilocal one) at that node. The difference is taken in that order, exact first, and the four pi times radius squared is the spherical measure that turns a radial quadrature into an integral over all space. Return a plain float.

    Parameters
    ----------
    mixing : np.ndarray
        Local exact-exchange mixing fraction at each node, each between zero and one.
    exact_density : np.ndarray
        Exact exchange energy density at each node, same length.
    semilocal_density : np.ndarray
        Semilocal exchange energy density at each node, same length.
    radii : np.ndarray
        Strictly positive radii in bohr, same length.
    weights : np.ndarray
        Radial quadrature weights in bohr, same length.

    Returns
    -------
    contribution : float
        float: that channel's contribution to the admixture energy, in hartree.

    Raises
    ------
    ValueError
        if mixing, the two energy densities, radii and weights do not all have the same length, if any radius is not positive, or if any mixing fraction lies outside zero to one.
    '''
    return contribution  # placeholder
```

### Step 10

lhnz_exchange_correction

Goal
----
Assemble the whole chain and return the local hybrid's exact-exchange admixture energy for a spherical, spin-polarised atom, summed over both spin channels. The majority channel holds one electron in the inner orbital and outer_occupation of an electron in the outer one; the minority channel holds one electron in the inner orbital alone. A channel is evaluated only at the radii where its own density exceeds 1e-14, and each channel is integrated over its own surviving radii. Both channels' effective hole normalisations must be formed before either channel's exchange-correlation hole normalisation is taken, and at a radius a channel did not survive its effective hole normalisation is taken to be one. Return the total as a plain float.

```python
def lhnz_exchange_correction(inner_exponents: "np.ndarray", inner_coefficients: "np.ndarray", outer_exponent: float, outer_occupation: float, c_param: float, b_param: float, n_nodes: int, scale: float) -> float:
    '''Assemble the whole chain and return the local hybrid's exact-exchange admixture energy for a spherical, spin-polarised atom, summed over both spin channels. The majority channel holds one electron in the inner orbital and outer_occupation of an electron in the outer one; the minority channel holds one electron in the inner orbital alone. A channel is evaluated only at the radii where its own density exceeds 1e-14, and each channel is integrated over its own surviving radii. Both channels' effective hole normalisations must be formed before either channel's exchange-correlation hole normalisation is taken, and at a radius a channel did not survive its effective hole normalisation is taken to be one. Return the total as a plain float.

    Parameters
    ----------
    inner_exponents : np.ndarray
        Positive Slater exponents of the inner orbital, in inverse bohr.
    inner_coefficients : np.ndarray
        Weights of those primitives, same length as inner_exponents.
    outer_exponent : float
        Positive Slater exponent of the outer orbital, in inverse bohr.
    outer_occupation : float
        Occupation of the outer orbital in the majority channel, between zero and one.
    c_param : float
        Positive coefficient multiplying the length inside the mixing function.
    b_param : float
        Non-negative constant scaling the hole-normalisation factor.
    n_nodes : int
        Number of radial quadrature nodes, at least one.
    scale : float
        Positive quadrature length scale in bohr.

    Returns
    -------
    admixture : float
        float: the exact-exchange admixture energy of the atom, in hartree.

    Raises
    ------
    ValueError
        if outer_occupation lies outside zero to one, or if any value it forwards to an earlier step is invalid.
    '''
    return admixture  # placeholder
```
