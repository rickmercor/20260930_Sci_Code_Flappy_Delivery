# Chemistry-Quantum_Chemistry-26

## Background

Ionization potentials and electron affinities are among the most directly measurable electronic properties of molecules. In many-body theory they appear as quasiparticle energies, the poles of the one-particle Green's function, which is linked to a mean-field reference through the Dyson equation and a self-energy that collects the correlation effects beyond the mean field. The second-order self-energy is the lowest-order correlated approximation beyond Hartree–Fock and is widely used as an inexpensive route to these energies.

Because the self-energy depends on the orbitals and orbital energies used to build it, self-consistent variants of second-order Green's-function theory are attractive, but they are known to be numerically fragile for many molecules. Spin-component scaling, in which same-spin and opposite-spin electron-pair contributions are weighted separately, is a common empirical way to improve second-order correlation treatments.

## Problem

A recent refinement of quasiparticle self-consistent second-order Green's function theory stabilizes the method with a flow-equation-based regularization. Apply this method to a linear chain of ten hydrogen atoms with a uniform spacing of 2.0 bohr between neighbouring atoms, using the spin-component-scaled parameter set its developers recommend for combined ionization-potential and electron-affinity accuracy. Use the STO-3G basis without density fitting, a closed-shell restricted Hartree–Fock reference, and correlate all electrons. Only NumPy and SciPy are available for the calculation; no quantum-chemistry package such as PySCF can be used. Report the converged quasiparticle energy of the highest occupied molecular orbital, in hartree, as a single number with at least seven significant figures.
For the numerical interfaces, total molecular charge defaults to $0$, and the quasiparticle iteration defaults to a residual tolerance of $10^{-9}$ hartree and at most 200 Fock-matrix builds; the residual is the largest absolute element of the total quasiparticle Fock matrix in the current orbital basis minus the diagonal matrix of the current orbital energies. The end-to-end benchmark uses a residual tolerance of $10^{-10}$ hartree and at most 300 Fock-matrix builds.

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

compute_one_electron_integrals

Goal
----
Compute the overlap, kinetic-energy and nuclear-attraction matrices of a molecule over contracted Cartesian s- and p-type Gaussian basis functions.

```python
def compute_one_electron_integrals(coords: "np.ndarray", nuclear_charges: "np.ndarray", shells: list) -> "np.ndarray":
    '''Overlap, kinetic-energy and nuclear-attraction matrices over contracted s/p Gaussians.

    Parameters
    ----------
    coords : np.ndarray
        Nuclear positions in bohr, shape (n_atoms, 3).
    nuclear_charges : np.ndarray
        Nuclear charges Z_A (atomic units), shape (n_atoms,). They enter only the
        nuclear-attraction matrix.
    shells : list of dict
        Basis shells in basis-function order. Each dict has the keys
        ``"center"`` (int, row index into ``coords``), ``"l"`` (int, 0 for s or 1 for p),
        ``"exponents"`` (sequence of positive floats, bohr^-2) and ``"coefficients"``
        (sequence of floats of the same length). The coefficients multiply normalised
        primitive Gaussians; every contracted function is then renormalised to unit
        self-overlap. An s shell contributes one basis function and a p shell three,
        ordered x, y, z. An SP shell of a basis-set library is supplied as a separate
        s shell and p shell with their own coefficients.

    Returns
    -------
    ints : np.ndarray
        Array of shape (3, nbf, nbf) with ints[0] the overlap matrix S, ints[1] the
        kinetic-energy matrix T and ints[2] the nuclear-attraction matrix V (hartree),
        which contains the attraction of an electron to every nucleus. All three are real
        symmetric; nbf is the
        total number of basis functions.

    Raises
    ------
    ValueError
        If ``coords`` is not of shape (n_atoms, 3), the charges do not match the atoms,
        a shell has l other than 0 or 1, an invalid centre index, or empty or
        mismatched exponent and coefficient lists, or a non-positive exponent.
    '''
    return ints
```

### Step 2

compute_electron_repulsion_integrals

Goal
----
Compute the full four-index tensor of electron-repulsion integrals, in chemists' notation, over contracted Cartesian s- and p-type Gaussian basis functions.

```python
def compute_electron_repulsion_integrals(coords: "np.ndarray", shells: list) -> "np.ndarray":
    '''Electron-repulsion integrals (mn|ls) over contracted s/p Gaussians, chemists' notation.

    Parameters
    ----------
    coords : np.ndarray
        Nuclear positions (shell centres) in bohr, shape (n_atoms, 3).
    shells : list of dict
        Basis shells in basis-function order, in the same format as for the one-electron
        integrals: keys ``"center"`` (int), ``"l"`` (0 or 1), ``"exponents"`` and
        ``"coefficients"`` (equal-length sequences). The coefficients multiply normalised
        primitives, and each contracted function is renormalised to unit self-overlap.
        An s shell gives one function and a p shell three, ordered x, y, z.

    Returns
    -------
    eri : np.ndarray
        Array of shape (nbf, nbf, nbf, nbf) in hartree, with
        eri[m, n, l, s] holding (mn|ls), the Coulomb repulsion between the charge
        distribution of the basis-function pair (m, n) and that of the pair (l, s).

    Raises
    ------
    ValueError
        If ``coords`` is not of shape (n_atoms, 3), or a shell has l other than 0 or 1,
        an invalid centre index, empty or mismatched exponent and coefficient lists, or a
        non-positive exponent.
    '''
    return eri
```

### Step 3

solve_restricted_hartree_fock

Goal
----
Solve the closed-shell restricted Hartree-Fock (Roothaan-Hall) equations self-consistently and return the total energy, the canonical orbital energies and the orbital coefficients.

```python
def solve_restricted_hartree_fock(S: "np.ndarray", h: "np.ndarray", eri: "np.ndarray", n_occ: int, e_nuc: float) -> tuple:
    '''Closed-shell restricted Hartree-Fock ground state with aufbau occupation.

    Parameters
    ----------
    S : np.ndarray
        AO overlap matrix, shape (nbf, nbf), symmetric positive definite.
    h : np.ndarray
        AO core Hamiltonian (kinetic plus nuclear attraction), shape (nbf, nbf), hartree.
    eri : np.ndarray
        AO electron-repulsion integrals in chemists' notation, eri[m, n, l, s] holding (mn|ls),
        shape (nbf, nbf, nbf, nbf), with the eightfold symmetry of real orbitals.
    n_occ : int
        Number of doubly occupied spatial orbitals, 1 <= n_occ <= nbf.
    e_nuc : float
        Nuclear repulsion energy (hartree), added to the electronic energy.

    Returns
    -------
    result : tuple
        ``(energy, eps, C)``: ``energy`` is the converged total RHF energy (float,
        hartree, including ``e_nuc``); ``eps`` is the array of the nbf canonical orbital
        energies in ascending order, shape (nbf,); ``C`` is the (nbf, nbf) array whose
        column k holds the AO coefficients of the orbital with energy eps[k], normalised
        to be orthonormal in the overlap metric. The n_occ lowest orbitals are occupied. The self-consistent
        field is started from the core-Hamiltonian guess and converged at least to energy
        changes below 1e-10 hartree and a largest element of the orbital gradient
        (the commutator of the Fock and density matrices, in an orthonormalised basis)
        below 1e-8.

    Raises
    ------
    ValueError
        If the array shapes are inconsistent, S is not positive definite, or n_occ is
        outside [1, nbf].
    RuntimeError
        If the self-consistent field does not converge.
    '''
    return energy, eps, C
```

### Step 4

compute_flow_regularized_weight

Goal
----
Evaluate, elementwise, the flow-parameter-dependent energy-denominator weight that multiplies each two-electron-integral product in the static self-energy of the flow-regularised quasiparticle self-consistent second-order Green's-function method (the source method).

```python
def compute_flow_regularized_weight(x: "np.ndarray", y: "np.ndarray", s: float) -> "np.ndarray":
    '''Flow-regularised energy-denominator weight of the source method's static self-energy.

    The weight is the complete factor that depends on the energy denominators and on s
    and multiplies a product of two-electron integrals in the source method's
    spin-integrated (spatial-orbital) static self-energy. It is taken exactly as it
    appears there, with no numerical prefactor added or removed.

    Parameters
    ----------
    x : np.ndarray
        Energy denominators (hartree) evaluated at the quasiparticle energy of the row
        orbital p. Any shape broadcastable with ``y``.
    y : np.ndarray
        Energy denominators (hartree) evaluated at the quasiparticle energy of the column
        orbital q. Any shape broadcastable with ``x``.
    s : float
        Flow parameter in hartree^-2, finite and non-negative.

    Returns
    -------
    w : np.ndarray
        Float array of the broadcast shape of ``x`` and ``y`` containing the weight
        elementwise, in hartree^-1. Elements with x = y = 0 take the continuous limit,
        0, for every s; the whole array is 0 when s = 0. No element may be NaN or
        infinite for finite inputs.

    Raises
    ------
    ValueError
        If ``s`` is negative or not finite.
    '''
    return w
```

### Step 5

compute_static_self_energy

Goal
----
Build the flow-regularised second-order static self-energy matrix of the source method for a closed-shell reference, in its spin-integrated, spin-component-scaled form over spatial molecular orbitals.

```python
def compute_static_self_energy(eps: "np.ndarray", eri_mo: "np.ndarray", n_occ: int, s: float, c_ss: float, c_os: float) -> "np.ndarray":
    '''Flow-regularised, spin-component-scaled static second-order self-energy (MO basis).

    Implements the source method's static self-energy for a closed-shell reference in
    spatial orbitals: the spin integration of its spin-orbital expression, with the
    same-spin and opposite-spin pair contributions scaled by ``c_ss`` and ``c_os`` and
    the energy denominators formed from ``eps`` as in the source method. All orbitals
    are correlated (no frozen core).

    Parameters
    ----------
    eps : np.ndarray
        Orbital (quasiparticle) energies in hartree, shape (n,). Orbitals 0 .. n_occ-1
        are doubly occupied and n_occ .. n-1 are virtual.
    eri_mo : np.ndarray
        Two-electron integrals over the n real spatial orbitals in chemists' notation,
        eri_mo[p, q, r, s] holding (pq|rs), shape (n, n, n, n), with eightfold permutational
        symmetry.
    n_occ : int
        Number of doubly occupied orbitals, 1 <= n_occ <= n.
    s : float
        Flow parameter in hartree^-2, finite and non-negative.
    c_ss : float
        Scaling factor of the same-spin contributions.
    c_os : float
        Scaling factor of the opposite-spin contributions.

    Returns
    -------
    sigma : np.ndarray
        Static self-energy matrix F_pq(s) in hartree over all n orbitals (occupied and
        virtual), shape (n, n). It vanishes for s = 0 and when there are no virtual
        orbitals.

    Raises
    ------
    ValueError
        If the shapes of ``eps`` and ``eri_mo`` are inconsistent, n_occ lies outside
        [1, n], or ``s`` is negative or not finite.
    '''
    return sigma
```

### Step 6

build_quasiparticle_fock

Goal
----
Assemble the total quasiparticle Fock matrix of the source method in the atomic-orbital basis from the current orbitals and quasiparticle energies.

```python
def build_quasiparticle_fock(S: "np.ndarray", h: "np.ndarray", eri: "np.ndarray", C: "np.ndarray", eps: "np.ndarray", n_occ: int, s: float, c_ss: float, c_os: float) -> "np.ndarray":
    '''Total quasiparticle Fock matrix (Hartree-Fock plus static self-energy) in the AO basis.

    Parameters
    ----------
    S : np.ndarray
        AO overlap matrix, shape (nbf, nbf).
    h : np.ndarray
        AO core Hamiltonian, shape (nbf, nbf), hartree.
    eri : np.ndarray
        AO electron-repulsion integrals in chemists' notation, eri[m, n, l, s] holding (mn|ls),
        shape (nbf, nbf, nbf, nbf).
    C : np.ndarray
        Current orbital coefficients, shape (nbf, nbf); the columns are orbitals,
        orthonormal in the overlap metric. Columns 0 .. n_occ-1 are the doubly occupied orbitals.
    eps : np.ndarray
        Current quasiparticle energies of the columns of ``C`` (hartree), shape (nbf,).
    n_occ : int
        Number of doubly occupied orbitals, 1 <= n_occ <= nbf.
    s : float
        Flow parameter of the source method (hartree^-2), finite and non-negative.
    c_ss : float
        Same-spin scaling factor of the self-energy.
    c_os : float
        Opposite-spin scaling factor of the self-energy.

    Returns
    -------
    fock : np.ndarray
        AO-basis matrix of shape (nbf, nbf), in hartree, of the source method's total
        quasiparticle operator: the closed-shell Hartree-Fock Fock operator of the aufbau
        density of the first n_occ columns of ``C``, plus the source method's
        flow-regularised static self-energy evaluated in the orbitals ``C`` with the
        energies ``eps`` (all orbitals correlated).

    Raises
    ------
    ValueError
        If the array shapes are inconsistent, n_occ lies outside [1, nbf], or ``s`` is
        negative or not finite.
    '''
    return fock
```

### Step 7

iterate_quasiparticle_self_consistency

Goal
----
Iterate the source method's quasiparticle equations to self-consistency and return the converged quasiparticle energies and orbitals.

```python
def iterate_quasiparticle_self_consistency(S: "np.ndarray", h: "np.ndarray", eri: "np.ndarray", n_occ: int, C0: "np.ndarray", eps0: "np.ndarray", s: float, c_ss: float, c_os: float, conv_tol: float = 1e-9, max_iter: int = 200) -> tuple:
    '''Converge the source method's quasiparticle self-consistency cycle.

    Parameters
    ----------
    S : np.ndarray
        AO overlap matrix, shape (nbf, nbf).
    h : np.ndarray
        AO core Hamiltonian, shape (nbf, nbf), hartree.
    eri : np.ndarray
        AO electron-repulsion integrals, chemists' notation (mn|ls), shape (nbf,) * 4.
    n_occ : int
        Number of doubly occupied orbitals, 1 <= n_occ <= nbf.
    C0 : np.ndarray
        Starting orbital coefficients, shape (nbf, nbf), orthonormal in the overlap
        metric, with columns ordered by ascending ``eps0`` (typically the restricted Hartree-Fock solution).
    eps0 : np.ndarray
        Starting orbital energies (hartree), shape (nbf,), ascending.
    s : float
        Flow parameter of the source method (hartree^-2), finite and non-negative.
    c_ss : float
        Same-spin scaling factor of the self-energy.
    c_os : float
        Opposite-spin scaling factor of the self-energy.
    conv_tol : float
        Convergence threshold (hartree): the cycle stops at the first iterate (C, eps)
        for which the total quasiparticle Fock matrix, expressed in the orbitals C,
        deviates from the diagonal matrix of eps by less than ``conv_tol`` in every
        element, where the total quasiparticle Fock matrix is
        (Hartree-Fock matrix of the aufbau density of C plus the back-transformed static
        self-energy built from C and eps).
    max_iter : int
        Maximum number of Fock-matrix builds.

    Returns
    -------
    result : tuple
        ``(eps, C)``: the converged quasiparticle energies in ascending order, shape
        (nbf,), and the corresponding orbital coefficients, shape (nbf, nbf),
        orthonormal in the overlap metric. The occupied quasiparticle orbitals are the first n_occ columns.

    Raises
    ------
    ValueError
        If the array shapes are inconsistent, n_occ lies outside [1, nbf], or ``s`` is
        negative or not finite.
    RuntimeError
        If the cycle does not converge within ``max_iter`` iterations.
    '''
    return eps, C
```

### Step 8

compute_homo_quasiparticle_energy

Goal
----
Compute the self-consistent highest-occupied quasiparticle energy of a closed-shell molecule with one of the source method's published parametrisations, starting from the atomic geometry and basis.

```python
def compute_homo_quasiparticle_energy(coords: "np.ndarray", nuclear_charges: "np.ndarray", shells: list, variant: str, charge: int = 0) -> float:
    '''Converged HOMO quasiparticle energy of the source method for a closed-shell molecule.

    Parameters
    ----------
    coords : np.ndarray
        Nuclear positions in bohr, shape (n_atoms, 3).
    nuclear_charges : np.ndarray
        Nuclear charges Z_A, shape (n_atoms,).
    shells : list of dict
        Basis shells in the format of the one-electron-integral step (keys ``"center"``,
        ``"l"`` in {0, 1}, ``"exponents"``, ``"coefficients"``; coefficients of normalised
        primitives, contracted functions renormalised, p functions ordered x, y, z).
    variant : str
        Which of the source method's published parametrisations to use: ``"plain"`` for
        the unscaled parametrisation, ``"scs"`` for the spin-component-scaled one and
        ``"sos"`` for the scaled-opposite-spin one.
    charge : int
        Total molecular charge; the number of electrons is sum(Z_A) - charge.

    Returns
    -------
    homo : float
        The quasiparticle energy (hartree) of the highest occupied orbital, n_occ =
        (sum(Z_A) - charge) / 2, taken directly from the converged quasiparticle
        self-consistent cycle. The reference is the closed-shell restricted Hartree-Fock
        solution (core-Hamiltonian guess, aufbau occupation) that starts the cycle. All
        electrons are correlated, the integrals are exact (no density fitting), and the
        cycle is converged until the total quasiparticle Fock matrix in the current
        orbitals deviates from diagonal form by less than 1e-10 hartree.

    Raises
    ------
    ValueError
        If ``variant`` is not one of "plain", "scs" or "sos", the electron count is odd
        or not positive, or the basis has fewer functions than occupied orbitals.
    '''
    return homo
```
