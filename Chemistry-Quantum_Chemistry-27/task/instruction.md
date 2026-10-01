# Chemistry-Quantum_Chemistry-27

## Background

Strongly correlated electrons, as found in bond breaking, transition-metal chemistry and lattice models of correlated materials, are poorly described by wave functions organized around a single reference determinant. A productive alternative organizes the many-electron space by seniority, the number of singly occupied levels once every spin orbital has been assigned a partner. Wave functions restricted to low seniority capture much of the strong correlation. Exact configuration interaction within a seniority sector still has a dimension that grows combinatorially with the number of levels and electrons, although approximate treatments of such wave functions (for example pair-coupled-cluster or mean-field decouplings of the sector) can reach polynomial cost. Their quality depends on the choice of orbitals, which are therefore optimized variationally together with the configuration-interaction coefficients.
 
The Hubbard model, in which electrons hop between neighbouring lattice sites and pay an energy penalty for double occupancy of a site, is a standard testbed for such methods because its exact ground state can be computed for small lattices and because its correlation strength is tuned by a single ratio of the interaction to the hopping. As the interaction grows, electrons localize with one electron per site, so a description in terms of singly occupied levels becomes natural; away from half filling, some levels are empty or doubly occupied while others remain singly occupied. Orbital optimization in such constrained wave functions generally has several stationary points, and breaking spin symmetry in the orbitals can matter qualitatively.

## Problem

Zero-seniority wave functions (doubly occupied configuration interaction) describe strong electron correlation well, but other seniority sectors are much less explored, although the maximal-seniority sector has the same underlying algebraic structure. A recent configuration-interaction approach assigns a fixed local seniority to every paired level: a set of pairing levels is restricted to seniority zero, and a set of spin levels to seniority one. The Hamiltonian is replaced by its local-seniority-conserving part, written with the so(4) generators of each level, and the orbitals are optimized variationally. The method takes one- and two-electron integrals, a seniority partition and electron counts, and returns an orbital-optimized variational energy whose accuracy is judged against full configuration interaction.
 
The benchmark is a periodic Hubbard ring of $M=8$ sites with $H=-t\sum_{i,\sigma}(c^\dagger_{i\sigma}c_{i+1,\sigma}+c^\dagger_{i+1,\sigma}c_{i\sigma})+U\sum_i n_{i\uparrow}n_{i\downarrow}$ (site $i+1$ taken modulo 8), with $t=1$, $U=4$ and $N_\uparrow=N_\downarrow=3$. Use the source paper's fixed-local-seniority CI at local seniority $\Omega=4$: levels 1–4 are pairing levels holding one electron pair, and levels 5–8 are spin levels holding two up and two down electrons. Use the source's local-seniority-conserving Hamiltonian, its coefficient definitions and its default orbital model (the one in which the pairing levels share one set of spatial orbitals for both spins while the spin levels are spin-unrestricted), with real orbitals. The fixed-local-seniority CI energy is the global minimum of the lowest sector eigenvalue over all orbitals allowed by that model. Measure correlation from the closed-shell restricted determinant $\Phi_{\mathrm{ref}}$ that places both spins in the three lowest eigenvectors of the hopping matrix, $E_{\mathrm{ref}}=\langle\Phi_{\mathrm{ref}}|H|\Phi_{\mathrm{ref}}\rangle$, and compute the exact energy $E_{\mathrm{FCI}}$ in the $N_\uparrow=N_\downarrow=3$ space.
 
Methodology at a high level: build the sector of product states (pair configurations times spin configurations), transform the Hubbard integrals to the unrestricted orbitals, assemble and diagonalize the local-seniority-conserving Hamiltonian in the sector, and minimize the lowest eigenvalue over the orbital parameters. The energy surface is not convex, so the global minimum must be established rather than assumed from one local optimization. Then compare with $E_{\mathrm{ref}}$ and $E_{\mathrm{FCI}}$.
 
In the reasoning, report: the dimensions of the full CI space and of the fixed-local-seniority CI sector; the number of independent orbital parameters in the orbital model; $E_{\mathrm{ref}}$; $E_{\mathrm{FCI}}$; the simplified form that the coefficients $W_{pq}$, $B_{pq}$ and $X_{pq}$ take for the Hubbard interaction; why the coupling between pairing-level charge and spin-level spin does not vanish here; why the spin levels must be spin-unrestricted, shown with the two-site Hubbard model, together with the orbital-optimized fixed-local-seniority CI energy of the two-site dimer at maximal seniority ($t=1$, $U=3$); the global-minimum fixed-local-seniority CI energy $E_{\mathrm{sector}}$ and the energy of the lowest competing local minimum; the fixed-local-seniority CI error $E_{\mathrm{sector}}-E_{\mathrm{FCI}}$; the correlation energies $E_{\mathrm{ref}}-E_{\mathrm{FCI}}$ and $E_{\mathrm{ref}}-E_{\mathrm{sector}}$; and the energy increase when the optimized fixed-local-seniority CI coefficient matrix $C_{\Gamma\Lambda}$ (pair configurations by spin configurations) is truncated to its largest singular value. Your final answer must be a single number: the recovered fraction of the correlation energy, $f=(E_{\mathrm{ref}}-E_{\mathrm{sector}})/(E_{\mathrm{ref}}-E_{\mathrm{FCI}})$.

Output Format Requirements:
Emit <final_answer>...</final_answer> first, followed by <reasoning>...</reasoning>. Put exactly one finite decimal and no units or prose inside <final_answer>. In <reasoning>, include enough intermediate calculations and all scientific diagnostics explicitly requested in the prompt to justify the deterministic computation, without turning the response into a general pipeline summary.

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

01_unrestricted_orbitals.py

Goal
----
Build the restricted-pairing, unrestricted-spin orbital coefficient matrices of a seniority eigenstate configuration interaction wave function from antisymmetric rotation parameters.

```python
def unrestricted_orbitals(x_params: "np.ndarray", y_params: "np.ndarray", n_levels: int, n_spin_levels: int) -> "np.ndarray":
    '''Return the spin-up and spin-down orbital coefficient matrices.
 
    Let M = n_levels and k = n_spin_levels. The antisymmetric generator X (M x M) has
    strict upper triangle filled row by row from x_params, in the order of
    numpy.triu_indices(M, 1), and X = A - A^T where A holds those entries. The generator
    Y (M x M) is zero except for its last k rows and columns, whose k x k block is built
    from y_params in the same way (order of numpy.triu_indices(k, 1)). Then
 
        C_up   = expm(X),
        C_down = expm(X) @ expm(Y),
 
    where column p of each matrix is the p-th orbital expanded in the M basis functions.
    Levels 0..M-k-1 are the pairing levels and levels M-k..M-1 the spin levels.
 
    Parameters
    ----------
    x_params : np.ndarray
        1-D array of M(M-1)/2 real rotation parameters.
    y_params : np.ndarray
        1-D array of k(k-1)/2 real rotation parameters (empty when k < 2).
    n_levels : int
        Number of spatial levels M >= 1.
    n_spin_levels : int
        Number of spin levels k with 0 <= k <= M.
 
    Returns
    -------
    orbitals : np.ndarray
        Array of shape (2, M, M): orbitals[0] = C_up and orbitals[1] = C_down.
 
    Raises
    ------
    ValueError
        If M < 1, k is outside [0, M], or a parameter array has the wrong length.
    '''
    return orbitals
```

### Step 2

02_seniority_coefficients.py

Goal
----
Transform a Hubbard Hamiltonian to unrestricted orbitals and extract the coefficients of its local-seniority-conserving part.

```python
def seniority_coefficients(hopping: "np.ndarray", onsite_U: float, orbitals: "np.ndarray") -> "np.ndarray":
    '''Return the coefficients of the local-seniority-conserving Hubbard Hamiltonian.

    Pair spin-up and spin-down orbitals with the same level index and use the unrestricted
    orbital basis stored in ``orbitals``. Column ``p`` of ``orbitals[0]`` is the spin-up
    orbital for level ``p`` and column ``p`` of ``orbitals[1]`` is the corresponding
    spin-down orbital. For the site-basis Hubbard Hamiltonian the scalar constant is E0 = 0
    and is not returned.

    Parameters
    ----------
    hopping : np.ndarray
        Real symmetric M x M one-electron matrix in the site basis.
    onsite_U : float
        On-site Hubbard interaction strength.
    orbitals : np.ndarray
        Real array of shape (2, M, M) containing the spin-up and spin-down orbital matrices.

    Returns
    -------
    coefficients : np.ndarray
        Array of shape (7, M, M), ordered as
        [diag(eps), diag(B), L, W, Bz, Kud, X] for the Hamiltonian written in the step
        background. The first two entries are diagonal matrices; W, Bz, Kud and X have
        zero diagonals, while L includes its diagonal terms.

    Raises
    ------
    ValueError
        If hopping is not square and symmetric or orbitals does not have shape (2, M, M).
    '''
    return coefficients
```

### Step 3

03_seniority_sector_basis.py

Goal
----
Enumerate the product states of a fixed-local-seniority sector: pair configurations on the pairing levels times spin configurations on the singly occupied levels.

```python
def seniority_sector_basis(n_pairing_levels: int, n_pairs: int, n_spin_levels: int, n_up: int) -> "np.ndarray":
    '''Return the ordered product-state basis of a fixed-local-seniority sector.
 
    Levels 0..P-1 are pairing levels (P = n_pairing_levels) and levels P..P+k-1 are spin
    levels (k = n_spin_levels). Each basis state is a row of length P + k with entries
        2  : doubly occupied pairing level,   0 : empty pairing level,
        1  : spin level holding an up electron,   -1 : spin level holding a down electron.
    Exactly n_pairs pairing levels are doubly occupied and exactly n_up spin levels hold
    up electrons. Rows are ordered with the pair configuration as the outer index and
    the spin configuration as the inner index; each is enumerated in the order of
    itertools.combinations over the occupied pairing levels (0..P-1) and over the up-spin
    levels (numbered 0..k-1 within the spin block), respectively.
 
    Parameters
    ----------
    n_pairing_levels : int
        P >= 0.
    n_pairs : int
        Number of electron pairs, 0 <= n_pairs <= P.
    n_spin_levels : int
        k >= 0.
    n_up : int
        Number of up electrons on the spin levels, 0 <= n_up <= k.
 
    Returns
    -------
    basis : np.ndarray
        Integer array of shape (C(P, n_pairs) * C(k, n_up), P + k).
 
    Raises
    ------
    ValueError
        If any count is negative, n_pairs > P, n_up > k, or P + k = 0.
    '''
    return basis
```

### Step 4

04_seci_hamiltonian.py

Goal
----
Build the matrix of the local-seniority-conserving (so(4)) Hamiltonian in a fixed-local-seniority product-state basis.

```python
def seci_hamiltonian(coefficients: "np.ndarray", sector_basis: "np.ndarray") -> "np.ndarray":
    '''Return the SECI Hamiltonian matrix in the given product-state basis.
 
    coefficients = [diag(eps), diag(B), L, W, Bz, Kud, X] (shape (7, M, M)); L and Kud
    are symmetric. Each row of sector_basis encodes one product state with entries 2
    (doubly occupied pairing level), 0 (empty pairing level), 1 (up spin level) or -1
    (down spin level). For a state, N_p = 2, 0, 1, 1 and S^z_p = 0, 0, 1/2, -1/2 for the
    four entry values. With E0 = 0 the Hamiltonian is
 
        H = sum_p eps_p N_p + sum_p B_p S^z_p + sum_{p,q} L_pq P+_p P_q
            + (1/4) sum_{p!=q} W_pq N_p N_q + sum_{p!=q} Bz_pq S^z_p S^z_q
            + sum_{p!=q} X_pq N_p S^z_q - sum_{p!=q} Kud_pq S+_p S-_q .
 
    Diagonal elements collect all N and S^z terms plus L_pp for every doubly occupied
    level. Off-diagonal elements use matrix elements +1 for the product-state
    transitions: P+_p P_q (p != q) maps a state with level q doubly occupied and level p
    empty to the state with the pair moved to p, contributing +L_pq; S+_p S-_q (p != q)
    maps a state with p down and q up to the state with p up and q down, contributing
    -Kud_pq. Element H[i, j] is the coefficient of basis state i in H applied to basis
    state j.
 
    Parameters
    ----------
    coefficients : np.ndarray
        Array of shape (7, M, M).
    sector_basis : np.ndarray
        Integer array of shape (D, M) with entries in {2, 0, 1, -1}; every state reached
        by the transitions above must be in the basis.
 
    Returns
    -------
    hamiltonian : np.ndarray
        Real symmetric D x D matrix.
 
    Raises
    ------
    ValueError
        If the shapes are inconsistent, a basis entry is not in {2, 0, 1, -1}, or a
        transition leads to a state that is not in the basis.
    '''
    return hamiltonian
```

### Step 5

05_seci_energy_gradient.py

Goal
----
Evaluate the seniority eigenstate configuration interaction energy of a Hubbard model for given orbital-rotation parameters, together with its exact gradient with respect to those parameters.

```python
def seci_energy_gradient(x_params: "np.ndarray", y_params: "np.ndarray", hopping: "np.ndarray", onsite_U: float, n_pairs: int, n_spin_levels: int) -> "np.ndarray":
    '''Return the SECI energy and its gradient with respect to the orbital parameters.
 
    Let M = hopping.shape[0] and k = n_spin_levels (k even). The orbitals are
    C_up = expm(X), C_dn = expm(X) expm(Y), with X built from x_params (order of
    numpy.triu_indices(M, 1), X = A - A^T) and Y zero except its last k x k block built
    from y_params (order of numpy.triu_indices(k, 1)). Levels 0..M-k-1 are pairing
    levels holding n_pairs pairs; levels M-k..M-1 are spin levels with k/2 up and k/2
    down electrons (S^z = 0). The energy E(x, y) is the lowest eigenvalue of the
    local-seniority-conserving part of the Hubbard Hamiltonian
        H = sum_{ij,s} hopping[i,j] c+_{is} c_{js} + U sum_i n_{i up} n_{i down}
    restricted to this sector (equivalently, of the full H projected onto the sector).
    The gradient is the exact derivative of E with respect to every entry of x_params
    and y_params.
 
    Parameters
    ----------
    x_params : np.ndarray
        M(M-1)/2 rotation parameters for X.
    y_params : np.ndarray
        k(k-1)/2 rotation parameters for Y.
    hopping : np.ndarray
        Real symmetric M x M site-basis one-electron matrix.
    onsite_U : float
        On-site repulsion U.
    n_pairs : int
        Number of electron pairs on the pairing levels (0 <= n_pairs <= M - k).
    n_spin_levels : int
        Even number k of spin levels, 0 <= k <= M.
 
    Returns
    -------
    result : np.ndarray
        Array of length 1 + M(M-1)/2 + k(k-1)/2: [E, dE/dx_1, ..., dE/dy_last]. The
        energy must be accurate to 1e-10 and each gradient component to 1e-7.
 
    Raises
    ------
    ValueError
        If k is odd or outside [0, M], n_pairs is outside [0, M - k], a parameter array
        has the wrong length, or the lowest eigenvalue is degenerate (gap below 1e-9),
        so that the gradient is undefined.
    '''
    return result
```

### Step 6

06_optimize_seci_orbitals.py

Goal
----
Find the variationally optimal seniority eigenstate configuration interaction energy of a Hubbard model by minimizing over all orbital rotations.

```python
def optimize_seci_orbitals(hopping: "np.ndarray", onsite_U: float, n_pairs: int, n_spin_levels: int, seed: int = 20260927) -> float:
    '''Return the orbital-optimized SECI energy (global minimum over orbital rotations).
 
    With M = hopping.shape[0], k = n_spin_levels (even), pairing levels 0..M-k-1 holding
    n_pairs pairs and spin levels M-k..M-1 holding k/2 up and k/2 down electrons, let
    E(x, y) be the lowest eigenvalue of the local-seniority-conserving part of the
    Hubbard Hamiltonian
        H = sum_{ij,s} hopping[i,j] c+_{is} c_{js} + U sum_i n_{i up} n_{i down}
    in this sector, for orbitals C_up = expm(X) and C_dn = expm(X) expm(Y), where X is a
    real antisymmetric M x M matrix and Y a real antisymmetric matrix that is zero
    outside its last k x k block. Return
        E_SECI = min over all real antisymmetric X and block-restricted Y of E(x, y),
    the global minimum, accurate to 1e-8. The minimizing orbitals are not unique and are
    not returned. At the reported minimum the gradient norm is below 1e-6. The energy
    and gradient at any (x, y) are those of seci_energy_gradient. Random starting points,
    if used, are drawn from numpy.random.default_rng(seed); the returned global minimum
    must not depend on the seed.
 
    Parameters
    ----------
    hopping : np.ndarray
        Real symmetric M x M site-basis one-electron matrix.
    onsite_U : float
        On-site repulsion U.
    n_pairs : int
        Number of electron pairs on the pairing levels (0 <= n_pairs <= M - k).
    n_spin_levels : int
        Even number k of spin levels, 0 <= k <= M.
    seed : int, optional
        Seed for the random starting points of the global search (default 20260927).
 
    Returns
    -------
    energy : float
        Global minimum of the SECI energy.
 
    Raises
    ------
    ValueError
        If k is odd or outside [0, M], or n_pairs is outside [0, M - k].
    '''
    return energy
```

### Step 7

07_hubbard_fci_energy.py

Goal
----
Compute the exact (full configuration interaction) ground-state energy of a Hubbard model in a fixed particle-number and spin sector.

```python
def hubbard_fci_energy(hopping: "np.ndarray", onsite_U: float, n_up: int, n_down: int) -> float:
    '''Return the lowest eigenvalue of a Hubbard Hamiltonian at fixed (N_up, N_down).
 
    H = sum_{ij,s} hopping[i,j] c+_{is} c_{js} + U sum_i n_{i up} n_{i down}, diagonalized
    in the space of all determinants with n_up up and n_down down electrons on
    M = hopping.shape[0] sites. The result must be accurate to 1e-10 (any exact or
    iterative eigensolver may be used).
 
    Parameters
    ----------
    hopping : np.ndarray
        Real symmetric M x M one-electron matrix.
    onsite_U : float
        On-site interaction U.
    n_up : int
        Number of up electrons, 0 <= n_up <= M.
    n_down : int
        Number of down electrons, 0 <= n_down <= M.
 
    Returns
    -------
    energy : float
        Ground-state energy in the (n_up, n_down) sector.
 
    Raises
    ------
    ValueError
        If hopping is not square and symmetric or an electron count is outside [0, M].
    '''
    return energy
```

### Step 8

08_seci_correlation_fraction.py

Goal
----
Compute the fraction of the Hubbard-ring correlation energy recovered by orbital-optimized seniority eigenstate configuration interaction at a chosen local seniority.

```python
def seci_correlation_fraction(n_sites: int, hopping_t: float, onsite_U: float, n_electrons: int, n_spin_levels: int, seed: int = 20260927) -> float:
    '''Return the correlation-energy fraction recovered by orbital-optimized SECI.
 
    System: periodic Hubbard ring of M = n_sites sites with hopping[i, i+1 mod M] =
    hopping[i+1 mod M, i] = -hopping_t (all other entries zero), on-site repulsion U, and
    n_electrons electrons with S^z = 0 (n_up = n_down = n_electrons / 2).
      * E_ref: energy of the closed-shell restricted determinant whose up and down
        electrons both occupy the n_electrons/2 lowest eigenvectors phi_i of the hopping
        matrix, E_ref = 2 sum_i eps_i + U sum_j rho_j^2 with rho_j = sum_i phi_i(j)^2. The
        shell must be closed (gap between the highest occupied and lowest unoccupied
        one-electron energies larger than 1e-9).
      * E_SECI: global minimum over orbital rotations C_up = expm(X), C_dn = expm(X)
        expm(Y) of the lowest eigenvalue of the local-seniority-conserving Hamiltonian in
        the sector with k = n_spin_levels spin levels (k/2 up, k/2 down; Y confined to
        their block) and (n_electrons - k)/2 pairs on the M - k pairing levels.
        E_SECI is the value returned by optimize_seci_orbitals(hopping, U,
        (n_electrons - k)/2, k, seed).
      * E_FCI: exact ground-state energy with n_up = n_down = n_electrons / 2.
    Return f = (E_ref - E_SECI) / (E_ref - E_FCI), accurate to 1e-7.
 
    Parameters
    ----------
    n_sites : int
        Number of ring sites M >= 3.
    hopping_t : float
        Hopping amplitude t > 0.
    onsite_U : float
        On-site repulsion U > 0.
    n_electrons : int
        Even electron number with 0 < n_electrons < 2 M. A completely filled ring
        (n_electrons = 2 M) is excluded: it has a single determinant, so
        E_ref = E_SECI = E_FCI and f = 0/0 is undefined.
    n_spin_levels : int
        Even k with 0 <= k <= M and 0 <= (n_electrons - k)/2 <= M - k.
    seed : int, optional
        Seed forwarded to optimize_seci_orbitals for its random starting points
        (default 20260927); f must not depend on it.
 
    Returns
    -------
    fraction : float
        Recovered fraction of the correlation energy.
 
    Raises
    ------
    ValueError
        If the inputs violate the conditions above (including a completely filled
        ring) or the reference shell is open.
    '''
    return fraction
```
