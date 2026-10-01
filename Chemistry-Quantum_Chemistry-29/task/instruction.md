# Chemistry-Quantum_Chemistry-29

## Background

In Kohn-Sham density functional theory the many-electron problem is replaced by independent electrons moving in a local, multiplicative effective potential, and the exchange-correlation part of that potential carries all of the many-body physics. Several important methods work backwards from a result to that potential: Kohn-Sham inversion looks for the potential that reproduces a given density, subsystem embedding needs the potential that couples one fragment to the rest, and optimized effective potential methods find the best local potential for an orbital-dependent energy. In practical molecular calculations all of these quantities live in a finite atom-centred basis set, so a local potential is often available only as a matrix of its integrals with pairs of basis functions, and the real-space potential has to be recovered from that matrix.

That recovery is a notoriously awkward inverse problem. A finite matrix holds far fewer numbers than a function of position, so infinitely many potentials share the same matrix, and the natural building blocks for an expansion, the products of pairs of basis functions, become nearly or exactly linearly dependent as the basis grows. Enlarging the basis therefore improves the uniqueness of the answer while ruining the numerical conditioning of the equations that determine it. Traditional remedies choose well-conditioned subsets by singular value truncation or pivoted Cholesky decomposition, or use special basis sets whose products are all linearly independent, but these choices are made on numerical grounds rather than on how much each building block matters for the potential being recovered.

The local spin density approximation is a convenient proving ground. Its potentials are explicit functions of the two spin densities, so for any self-consistent calculation the exact potential of each spin is known pointwise and the quality of a reconstruction can be judged in real space, and reference energies of atoms computed on numerical grids without any basis-set error are available for checking a basis-set calculation. In an atom with an unpaired electron the two spin potentials differ, most strongly in the outer region where one spin dominates, and how the correlation functional interpolates between unpolarized and fully polarized electron gases decides the minority-spin potential there. Atomic local-density potentials also have a well-known weakness far from the nucleus, where they decay exponentially instead of like -1/r, which is the origin of the large gap between the highest occupied orbital energy and the ionization energy.

## Problem

A local potential that is known only through its matrix in a finite basis, as happens in Kohn-Sham inversion, density embedding and optimized effective potential methods, cannot simply be expanded in products of basis functions and solved for, because those products are nearly or exactly linearly dependent and the linear system is hopelessly ill-conditioned. A recent treatment turns the dependencies into an asset: it builds the potential from a subset of products added one at a time in order of importance, so that every matrix element whose product depends linearly on the chosen ones is reproduced automatically. Here that reconstruction is tested on a potential known exactly in real space, the minority-spin exchange-correlation potential of the open-shell lithium atom in the complete-basis-set limit, and the output is the real-space L2 error of the reconstructed potential.

Treat Li (Z = 3, 1s2 2s1, two up-spin and one down-spin electron) in spin-unrestricted Kohn-Sham theory in atomic units, nonrelativistically with a point nucleus, with the local spin density approximation made of Slater exchange with its exact spin scaling and the spin-polarized Vosko-Wilk-Nusair correlation energy (VWN5, built from their paramagnetic, ferromagnetic and spin-stiffness fits). Solve the radial Kohn-Sham equations numerically without any basis set, converged to the complete-basis-set limit, and take v_xc,down(r), the down-spin exchange-correlation potential of the self-consistent spin densities, evaluated wherever the density is positive without any density cutoff. Form its matrix V_ij = <f_i | v_xc,down | f_j>, with every integral taken over all space, in the basis of the 16 normalized s-type Gaussians f_k(r) = (2 alpha_k / pi)^(3/4) exp(-alpha_k r^2) with alpha_k = 0.025 * 2.3^(k-1) bohr^-2, k = 1, ..., 16.

Reconstruct the down-spin potential from V with that treatment's importance-based selection of basis-function products, expanding v_R in the plain products f_k f_l (k <= l) of the normalized basis functions, ranking pairs at every iteration, including the first, by the unnormalized residual |V^R_ij - V_ij| between the matrix of the current v_R and V, and stopping when the largest residual over all pairs i <= j falls below eps. Give as the final answer ||v_R - v_xc,down|| = (integral over all space of (v_R - v_xc,down)^2 d^3r)^(1/2), in hartree bohr^(3/2), for eps = 1e-6 hartree, to at least four significant figures.

In your reasoning also give the down-spin density at r = 1 bohr and the down-spin potential at r = 3 bohr; the pair selected first and its matrix element; the number of selected products at eps = 1e-6; the down-spin L2 error at eps = 1e-5; the L2 error at eps = 1e-6 when the up-spin potential is reconstructed in the same way from its own matrix; and the first ionization energy of lithium in eV from the basis-free total energies of Li and Li+ in the same approximation. Compare with experiment: give the measured first ionization energy of lithium tabulated in the NIST Atomic Spectra Database in cm^-1 together with the uncertainty that database lists beside it, the error of the computed ionization energy relative to that measurement in eV, how many times that error exceeds the tabulated uncertainty, and by how much minus the up-spin 2s orbital energy falls short of the measured value, in eV. Report the density, the potential and the ionization energies to at least five significant figures, the tabulated measurement and its uncertainty with every digit the database prints, and every other scalar to at least four significant figures.

Output Format Requirements:
Emit <final_answer> immediately, then <reasoning>. Do not write a long derivation before the tags.
You must emit exactly one finite decimal inside <final_answer>...</final_answer>, even if the value is approximate or you are unsure.
Rules:
- The tags are required. Do not omit them or leave them empty.
- The tagged value must be a finite decimal (examples: 0.4847, 0.22, 1.05). Not NaN, not Inf, not a fraction string, not a vector, not prose.
- Put only that one number between the tags. No units, no words, no extra lines.
Show the scalars the problem statement asks for, together with the few others that determine the final number.
Do not paste the input matrices, full coefficient vectors, per-iteration paths, or per-fold candidate tables.

## Output format

```
## Output format

Scientific reasoning wrapped in <reasoning>...</reasoning> tags. Include enough intermediate calculations to justify the deterministic computation, without turning the response into a general pipeline summary. A single final numeric answer wrapped in <final_answer>...</final_answer> tags.
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

01_lsd_exchange_correlation

Goal
----
Step 01: Spin-polarized LSD exchange-correlation energy and potentials (Slater plus VWN5). Spin-polarized local density approximation for exchange and correlation.

```python
def lsd_exchange_correlation(rho_up: "np.ndarray", rho_down: "np.ndarray") -> "np.ndarray":
    '''Exchange-correlation energy per electron and spin potentials of the LSD approximation (Slater exchange plus VWN5).

    Parameters
    ----------
    rho_up : np.ndarray
        Up-spin electron density in bohr^-3, any shape, every entry finite and >= 0.
    rho_down : np.ndarray
        Down-spin electron density in bohr^-3, same shape as rho_up, every entry finite and >= 0.

    Returns
    -------
    result : np.ndarray
        Array of shape (3,) + rho_up.shape and dtype float in hartree. result[0] is the exchange-correlation energy per
        electron eps_xc(rho_up, rho_down), and result[1] and result[2] are the up-spin and down-spin potentials
        v_xc,sigma = d(rho eps_xc)/d rho_sigma with rho = rho_up + rho_down. Exchange is Slater exchange with exact spin
        scaling; correlation is the spin-polarized Vosko-Wilk-Nusair (VWN5) form built from the paramagnetic and
        ferromagnetic Ceperley-Alder fits and the spin-stiffness fit. Where rho is 0 all three entries are 0; every
        positive total density, including fully polarized points where one spin density is 0, is evaluated from the
        functional itself.

    Raises
    ------
    ValueError
        If the two arrays differ in shape or any entry is negative or not finite.
    '''
    return result
```

### Step 2

02_radial_lsd_spin_densities

Goal
----
Step 02: Basis-free self-consistent LSD spin densities of an s-electron atom by a radial solver. Basis-free self-consistent spin densities of a spherical atom or ion in the local spin density approximation.

```python
def radial_lsd_spin_densities(Z: float, n_up: int, n_down: int, radii: "np.ndarray") -> "np.ndarray":
    '''Converged basis-free LSD spin densities of an atom or ion with only s electrons, at given distances.

    Parameters
    ----------
    Z : float
        Nuclear charge in units of e, finite and > 0.
    n_up : int
        Number of occupied up-spin s orbitals, 1 or 2.
    n_down : int
        Number of occupied down-spin s orbitals, integer with 0 <= n_down <= n_up; the system has n_up + n_down <= Z
        electrons.
    radii : np.ndarray
        1-D array of finite distances r > 0 from the nucleus in bohr.

    Returns
    -------
    result : np.ndarray
        Array of shape (2, len(radii)) in bohr^-3: row 0 is the up-spin and row 1 the down-spin density of the self-
        consistent, spin-unrestricted Kohn-Sham solution in the complete-basis-set limit. Each spin channel has the
        nonrelativistic kinetic energy, the attraction -Z/r to a point nucleus, the Hartree potential of the total
        density and its own exchange-correlation potential from lsd_exchange_correlation (no density cutoff), and its
        n_sigma lowest s states are singly occupied. The densities are converged to a relative accuracy of 1e-8 wherever
        they exceed 1e-10 bohr^-3.

    Raises
    ------
    ValueError
        If Z is not finite or not positive, n_up or n_down is not an integer in the ranges above, or radii contains a
        value that is not finite and positive.
    '''
    return result
```

### Step 3

03_xc_potential_matrices

Goal
----
Step 03: Basis-set matrices of the basis-free spin-resolved exchange-correlation potentials. Matrix representations of the basis-free spin-resolved exchange-correlation potentials in a Gaussian basis.

```python
def xc_potential_matrices(Z: float, n_up: int, n_down: int, exponents: "np.ndarray") -> "np.ndarray":
    '''Matrices of the basis-free LSD exchange-correlation potentials of both spins in a normalized s-Gaussian basis.

    Parameters
    ----------
    Z : float
        Nuclear charge in units of e, as in radial_lsd_spin_densities.
    n_up : int
        Number of occupied up-spin s orbitals, as in radial_lsd_spin_densities.
    n_down : int
        Number of occupied down-spin s orbitals, as in radial_lsd_spin_densities.
    exponents : np.ndarray
        1-D array of K distinct, finite, positive exponents alpha_k in bohr^-2 defining the normalized basis functions
        f_k(r) = (2 alpha_k / pi)^(3/4) exp(-alpha_k r^2), in the order given.

    Returns
    -------
    result : np.ndarray
        Array of shape (2, K, K) in hartree, each block symmetric: result[s][k, l] is the integral over all space of
        f_k(r) v_xc,s(r) f_l(r), with s = 0 for the up-spin and s = 1 for the down-spin potential of
        lsd_exchange_correlation evaluated at the basis-free self-consistent spin densities of
        radial_lsd_spin_densities, with no density cutoff. Each element is converged to better than 1e-9 hartree.

    Raises
    ------
    ValueError
        Under the conditions of radial_lsd_spin_densities, or if the exponents are not a 1-D array of distinct finite
        positive numbers.
    '''
    return result
```

### Step 4

04_product_overlap_matrix

Goal
----
Step 04: Overlap matrix of basis-function products. Overlap matrix of the products of pairs of basis functions.

```python
def product_overlap_matrix(exponents: "np.ndarray") -> "np.ndarray":
    '''Overlap matrix of the unnormalized products of normalized s Gaussians.

    Parameters
    ----------
    exponents : np.ndarray
        1-D array of K distinct, finite, positive exponents alpha_k in bohr^-2 defining the normalized basis functions
        f_k(r) = (2 alpha_k / pi)^(3/4) exp(-alpha_k r^2).

    Returns
    -------
    result : np.ndarray
        Symmetric array of shape (M, M), M = K(K + 1)/2, with element (p, q) equal to the integral over all space of
        f_i(r) f_j(r) f_k(r) f_l(r), where p labels the pair (i, j) and q the pair (k, l). Pairs with i <= j are ordered
        row by row: (0, 0), (0, 1), ..., (0, K-1), (1, 1), (1, 2), ..., (K-1, K-1). The products are not renormalized.

    Raises
    ------
    ValueError
        If the exponents are not a 1-D array of distinct finite positive numbers.
    '''
    return result
```

### Step 5

05_importance_selected_products

Goal
----
Step 05: Importance-based selection of basis-function products. Importance-based selection of basis-function products that reproduce a potential matrix.

```python
def importance_selected_products(V: "np.ndarray", exponents: "np.ndarray", eps: float) -> "np.ndarray":
    '''Products of basis-function pairs chosen one at a time by the largest unreproduced matrix element.

    Parameters
    ----------
    V : np.ndarray
        Symmetric (K, K) matrix in hartree of a local potential v in the normalized basis f_k of product_overlap_matrix,
        V_ij = <f_i | v | f_j>.
    exponents : np.ndarray
        1-D array of the K distinct positive exponents of that basis, in bohr^-2.
    eps : float
        Convergence threshold in hartree, finite and > 0.

    Returns
    -------
    result : np.ndarray
        Integer array of shape (M_R, 2): the selected pairs (i, j), i <= j, zero-based, in the order selected. With the
        selected set R, the trial potential v_R(r) = sum over (k, l) in R of b_kl f_k(r) f_l(r) uses plain, unnormalized
        products, and its coefficients b solve exactly the square system sum over (k, l) in R of
        <f_i f_j | f_k f_l> b_kl = V_ij for all (i, j) in R, so its matrix V^R reproduces V on every selected pair.
        Starting from an empty R (v_R = 0), each iteration computes the residuals |V^R_ij - V_ij| over all pairs i <= j;
        if the largest residual is below eps the selection stops, otherwise the pair with the largest residual is added
        (the first in the row-by-row pair order of product_overlap_matrix if several are equal) and b is recomputed.
        Exactly one pair is added per iteration, and the selection also stops once every pair has been selected. If
        the largest element of |V| is already below eps the result has shape (0, 2).

    Raises
    ------
    ValueError
        If V is not a finite symmetric (K, K) array matching the exponents, the exponents are invalid, or eps is not
        finite and positive.
    '''
    return result
```

### Step 6

06_reconstructed_potential

Goal
----
Step 06: Reconstructed real-space potential. Real-space potential reconstructed from its matrix with importance-selected products.

```python
def reconstructed_potential(V: "np.ndarray", exponents: "np.ndarray", eps: float, radii: "np.ndarray") -> "np.ndarray":
    '''Reconstructed local potential v_R(r) at given distances from the nucleus.

    Parameters
    ----------
    V : np.ndarray
        Symmetric (K, K) potential matrix in hartree, as in importance_selected_products.
    exponents : np.ndarray
        1-D array of the K distinct positive exponents of the normalized s-Gaussian basis, in bohr^-2.
    eps : float
        Selection threshold in hartree, finite and > 0, as in importance_selected_products.
    radii : np.ndarray
        1-D array of finite distances r >= 0 in bohr.

    Returns
    -------
    result : np.ndarray
        Array with the shape of radii, in hartree: v_R(r) = sum over (k, l) in R of b_kl f_k(r) f_l(r), with R the pairs
        selected by importance_selected_products for (V, exponents, eps) and b the coefficients that reproduce V exactly
        on those pairs. It is 0 everywhere when no pair is selected.

    Raises
    ------
    ValueError
        Under the conditions of importance_selected_products, or if radii contains a negative or non-finite value.
    '''
    return result
```

### Step 7

07_reconstruction_l2_error

Goal
----
Step 07: L2 error of the reconstructed spin-resolved exchange-correlation potential (orchestrator). Real-space L2 error of a spin-resolved exchange-correlation potential reconstructed from its basis-set matrix (orchestrator).

```python
def reconstruction_l2_error(Z: float, n_up: int, n_down: int, exponents: "np.ndarray", spin: int, eps: float) -> float:
    '''L2 error of the importance-selected reconstruction of a self-consistent LSD exchange-correlation potential.

    Parameters
    ----------
    Z : float
        Nuclear charge in units of e, as in radial_lsd_spin_densities.
    n_up : int
        Number of occupied up-spin s orbitals, as in radial_lsd_spin_densities.
    n_down : int
        Number of occupied down-spin s orbitals, as in radial_lsd_spin_densities.
    exponents : np.ndarray
        1-D array of K distinct positive exponents in bohr^-2 of the normalized s-Gaussian basis.
    spin : int
        0 for the up-spin and 1 for the down-spin exchange-correlation potential.
    eps : float
        Selection threshold in hartree, finite and > 0, as in importance_selected_products.

    Returns
    -------
    result : float
        ||v_R - v_xc,s|| = (integral over all space of (v_R(r) - v_xc,s(r))^2 d^3r)^(1/2) in hartree bohr^(3/2), as a
        Python float, where v_xc,s is the LSD potential of spin s at the basis-free self-consistent spin densities
        (radial_lsd_spin_densities and lsd_exchange_correlation, no density cutoff) and v_R is the reconstruction of
        reconstructed_potential from the matrix of spin s from xc_potential_matrices, with the same basis and threshold.
        Converged to better than 1e-7.

    Raises
    ------
    ValueError
        Under the conditions of radial_lsd_spin_densities, xc_potential_matrices or importance_selected_products, or if spin is not 0 or 1.
    '''
    return result
```
