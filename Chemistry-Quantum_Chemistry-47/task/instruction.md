# Chemistry-Quantum_Chemistry-47

## Background

In Hartree-Fock and Kohn-Sham theory the orbitals of a closed shell molecule are the generalized eigenvectors of a Fock matrix that depends on those same orbitals through the density. The Roothaan equations are therefore solved by iteration: a guess density gives a Fock matrix, diagonalizing it gives a new density, and the cycle repeats until nothing changes. The converged density fixes the total energy, and the number of cycles needed dominates the cost of routine calculations. Because the converged energy is stationary with respect to the orbitals, its derivatives with respect to nuclear positions can be written without solving for how the orbitals respond, which is what makes geometry optimization and harmonic frequency analysis practical.

Symbolic regression and program synthesis search over compositions of elementary operations rather than over continuous weights. Applied to matrix-based electronic structure, the candidates are short programs of matrix operations acting on a workspace matrix in the atomic-orbital basis, and a program is judged by how well the energies obtained from its output reproduce reference calculations on a training set. Such programs contain no adjustable floating-point parameters, so transferability is tested by applying the same program, unchanged, to molecules outside the training set that are built from the same elements.

Training geometries for potential energy surface work are commonly produced by displacing a molecule along its normal modes. In the harmonic approximation the energy above equilibrium is a weighted sum of squared mode displacements, so a geometry can be characterized by an excess energy and a direction in mode space. Energies of different molecules are placed on a common scale by referring them to their separated atoms, which removes the large geometry-independent atomic contributions and leaves the binding energies that matter chemically.

## Problem

Hartree-Fock and Kohn-Sham calculations spend most of their effort iterating the Roothaan equations until the orbitals stop changing. A recent method skips the iteration. It takes a short program of matrix operations, applies it to a starting guess matrix, and diagonalizes the result once; the eigenvectors of that "workspace matrix" are used as the orbital coefficients. The program is found by a random search, scored against converged reference energies for a few small training molecules, and the winner is then used, unchanged, on other molecules made of the same elements.

Your job is to carry out this discovery procedure on the small model described below and report how well the program you find does on a molecule it never saw during training.

The model. Atoms sit on a line. Each atom has one basis function, one electron and a core charge of one, so a molecule with n atoms (n is always even) is a closed shell with n/2 doubly occupied orbitals. There are two element types, 0 and 1, with on-site energies of -1.0 and -1.6 and on-site repulsions of 0.9 and 1.3. The overlap between two different sites is a Gaussian in their separation with standard deviation 1.1; a site overlaps perfectly with itself. Two sites interact through the Ohno formula: the interaction is the inverse square root of the squared separation plus a squared length, and that length is fixed for each pair so that two sites at zero separation would interact with the average of their two on-site repulsions. A site's diagonal core-Hamiltonian element is its on-site energy lowered by its Ohno interaction with every other core. The off-diagonal core-Hamiltonian elements follow the Wolfsberg-Helmholz rule of extended Hueckel theory with a proportionality constant of 1.75. Two-electron integrals obey zero differential overlap: an integral is nonzero only when each electron's two orbital indices coincide, and its value is then the Ohno interaction between the two sites. With these integrals, the Fock matrix and the electronic energy are the ordinary closed-shell expressions of Roothaan theory. The cores repel as unit point charges through Coulomb's law, plus a short-range exponential wall of height 6.0 and decay length 0.45. Reference energies are fully converged self-consistent solutions of this model, with the density converged to 1e-10. An isolated atom is a single electron on a single site, so its energy is nothing more than its one-electron energy.

The molecules. Positions are described by bond lengths, the distances between consecutive atoms, with the first atom at the origin. The equilibrium geometry of a molecule is the minimum of its converged reference energy in bond-length coordinates, found by starting from every bond equal to 1.2. Its normal modes are the eigenvectors of the curvature matrix of that energy in bond-length coordinates, with unit masses, and each mode's frequency is the square root of the corresponding eigenvalue. Modes are numbered in order of increasing frequency, and each mode vector is normalized to unit length with its sign chosen so that its largest-magnitude component is positive. Molecules are written as their sequence of element types.

Training molecule 1: (0, 0). Training molecule 2: (1, 1). Training molecule 3: (0, 1, 0, 1). Hold-out molecule: (0, 1).

The sampled geometries. Each geometry is specified by an excess energy above equilibrium in the local quadratic model and a direction. It is the point on the constant-energy shell of that quadratic model that lies along the given direction, expressed in the scaled normal-mode coordinates used by the sampling scheme of the source method. Only the orientation of the direction matters, not its length, and the direction's components refer to the modes in the numbering above.

Molecule 1: (0, (1)), (0.005, (1)), (0.012, (-1)), (0.019, (1)), (0.019, (-1)).
Molecule 2: (0, (1)), (0.008, (-1)), (0.015, (1)), (0.019, (-1)), (0.004, (1)).
Molecule 3: (0, (1, 0, 0)), (0.006, (1, 1, 1)), (0.012, (2, -1, 0.5)), (0.018, (-1, 0, 1)), (0.019, (0, -3, -4)).
Hold-out: (0, (1)), (0.006, (1)), (0.012, (-1)), (0.019, (1)), (0.019, (-1)).

Keep a training geometry only if its reference energy lies no more than 0.02 above the reference energy of the equilibrium geometry. All five hold-out geometries are used.

The program search. The workspace matrix starts as the core Hamiltonian, and a program of four operations from the following library is applied in order: 0, leave the matrix as it is; 1, add the core Hamiltonian; 2, add the overlap matrix; 3, replace the matrix by itself times the inverse overlap times itself; 4, add the two-electron part of the Fock matrix evaluated for the density that puts exactly one electron on every site; 5, multiply the matrix elementwise by the overlap; 6, halve the matrix. A program predicts the energy of a geometry by solving the generalized eigenproblem of its final workspace matrix with the overlap once, occupying the n/2 lowest eigenvectors, and evaluating the model's energy expression for that density.

Energies are compared on a shifted scale. Every reference energy is measured from the sum of the isolated-atom energies of the molecule's atoms. Every program prediction is shifted by a molecule-specific constant made of one free contribution per element type, counted once for each atom of that type; the two contributions are chosen afresh for each candidate program so that its loss is as small as possible. The loss of a program is the root-mean-square difference between shifted predictions and shifted references over all retained training geometries of all training molecules.

The search is simulated annealing driven by a single NumPy default generator seeded with 7. The starting program is one call drawing four uniform integers over the seven library indices. Each of the 200 iterations then draws, in this order, an integer r uniform from 1 to 3, a choice of r distinct positions among the four, and r uniform library indices to write into those positions. A candidate whose loss is not larger than the current loss is accepted outright; otherwise one uniform draw on the unit interval is compared with the Metropolis factor, the exponential of minus the loss increase over the temperature, and the candidate is accepted when the draw is smaller. The temperature at iteration k, counting from zero, is 0.05 times (1 - k/200). The best program encountered, including the starting one, is kept.

Finally, apply the best program to the five hold-out geometries and report, as the final answer, the root-mean-square difference between the shifted program predictions and the shifted reference energies of the hold-out molecule, in the model's energy units, together with the intermediate results and conclusions that determine it.

Output Format Requirements:
Scientific reasoning wrapped in <reasoning>...</reasoning> tags. Include enough intermediate calculations to justify the deterministic computation, without turning the response into a general pipeline summary. A single final numeric answer wrapped in <final_answer>...</final_answer> tags.

## Output format

```
## Output format

Scientific reasoning wrapped in <reasoning>...</reasoning> tags. Include enough intermediate calculations to justify the deterministic computation, without turning the response into a general pipeline summary. A single final numeric answer wrapped in <final_answer>...</final_answer> tags.
```

## Your task

Implement **all 13 functions** below into a single file at `/app/solution.py`. Requirements:

- Use the EXACT function name and signature shown for each step.
- Define every function at module scope (importable).
- Do NOT include tests, example usage, prints, or a `__main__` block.
- Available libraries: numpy, scipy, sympy, pandas, scikit-learn, networkx.

Start your file with these imports:

```python
import numpy as np
```

### Step 1

build_model_matrices

Goal
----
Set up the overlap, core Hamiltonian and site-site Coulomb matrices of the site model.

```python
def build_model_matrices(positions: "np.ndarray", types: "np.ndarray", params: dict) -> "np.ndarray":
    """Return the overlap, core-Hamiltonian and Coulomb matrices stacked together.

    Parameters
    ----------
    positions : np.ndarray
        Shape (n,), atom positions along the line.
    types : np.ndarray
        Shape (n,), element type of each atom, 0 or 1.
    params : dict
        Model constants: ``onsite_energy`` (two floats, one per type),
        ``onsite_repulsion`` (two floats, one per type), ``kappa`` (the
        Wolfsberg-Helmholz factor) and ``sigma`` (the standard deviation of
        the Gaussian overlap). Other keys are ignored.

    Returns
    -------
    matrices : np.ndarray
        Shape (3, n, n) float array holding, in this order, the overlap S (unit
        diagonal, Gaussian off-diagonal with standard deviation sigma in the
        separation), the core Hamiltonian H (diagonal: on-site energy minus the
        sum of the Ohno interactions with all other sites; off-diagonal:
        kappa times the overlap times the mean of the two diagonal entries) and
        the Ohno interaction matrix Gamma (on-site repulsion on the diagonal;
        for distinct sites the inverse of the root of the squared separation
        plus the squared Ohno length, the Ohno length being twice the inverse
        of the summed on-site repulsions of the pair).

    Raises
    ------
    ValueError
        If positions and types are not one-dimensional arrays of the same
        length, or if a type is not 0 or 1.
    """
    return matrices
```

### Step 2

mean_field_energy

Goal
----
Total mean-field energy of the site model for a given closed-shell density matrix.

```python
def mean_field_energy(H: "np.ndarray", gamma: "np.ndarray", P: "np.ndarray", positions: "np.ndarray", params: dict) -> float:
    """Return the total mean-field energy of the density matrix P.

    Parameters
    ----------
    H : np.ndarray
        Shape (n, n) core Hamiltonian.
    gamma : np.ndarray
        Shape (n, n) Ohno interaction matrix.
    P : np.ndarray
        Shape (n, n) symmetric closed-shell density matrix.
    positions : np.ndarray
        Shape (n,) atom positions, used only for the core-core repulsion.
    params : dict
        Model constants; uses ``core_repulsion_strength`` (the height of the
        exponential wall) and ``core_repulsion_range`` (its decay length).

    Returns
    -------
    energy : float
        The closed-shell electronic energy of Roothaan theory evaluated with
        the zero-differential-overlap Fock matrix built from P, plus the
        core-core repulsion, which for every pair of atoms is the inverse
        separation plus the wall height times a decaying exponential of the
        separation over the decay length. Returned as a native Python float.

    Raises
    ------
    ValueError
        If H, gamma and P are not square matrices of one common size, or the
        number of positions differs from that size.
    """
    return energy
```

### Step 3

scf_reference_energy

Goal
----
Converged self-consistent-field reference energy of a molecule of the site model.

```python
def scf_reference_energy(positions: "np.ndarray", types: "np.ndarray", params: dict, tol: float = 1e-10, max_iter: int = 500, damping: float = 0.5) -> float:
    """Return the converged closed-shell SCF total energy.

    Parameters
    ----------
    positions : np.ndarray
        Shape (n,) atom positions; n must be even.
    types : np.ndarray
        Shape (n,) element types, 0 or 1.
    params : dict
        Model constants used by the matrix builder and the energy evaluation.
    tol : float
        Convergence threshold on the largest absolute change of a density
        element between one diagonalization and the next.
    max_iter : int
        Maximum number of diagonalizations.
    damping : float
        Fraction of the new density mixed into the iterate after a
        non-converged diagonalization; the rest is the previous iterate.

    Returns
    -------
    energy : float
        Total energy of the converged density as a native Python float. Each
        iteration builds the Fock matrix from the current density, solves the
        generalized eigenproblem with the overlap, forms the new density from
        the n/2 lowest eigenvectors and stops when the largest change is below
        tol, returning the energy of that new density; otherwise the damped
        mixture becomes the next iterate. The result does not depend on the
        damping value.

    Raises
    ------
    ValueError
        If the atom count is odd or convergence is not reached within
        max_iter diagonalizations.
    """
    return energy
```

### Step 4

scf_energy_gradient

Goal
----
Analytic first derivative of the converged SCF energy with respect to every atom position.

```python
def scf_energy_gradient(positions: "np.ndarray", types: "np.ndarray", params: dict) -> "np.ndarray":
    """Return the exact gradient of the converged SCF total energy.

    Parameters
    ----------
    positions : np.ndarray
        Shape (n,) atom positions; n is even.
    types : np.ndarray
        Shape (n,) element types, 0 or 1.
    params : dict
        Model constants.

    Returns
    -------
    gradient : np.ndarray
        Shape (n,) float array whose entry a is the derivative of the converged
        SCF total energy (the energy the reference-energy routine returns) with
        respect to the position of atom a, holding the other atoms fixed. The
        derivative is the analytic one, evaluated at the SCF solution converged
        until the largest change of a density element between successive
        diagonalizations is below 1e-12, and is checked to 1e-11 in every
        component.

    Raises
    ------
    ValueError
        If the atom count is odd, or if positions and types disagree in length.
    """
    return gradient
```

### Step 5

scf_energy_hessian

Goal
----
Analytic second derivatives of the converged SCF energy with respect to the atom positions.

```python
def scf_energy_hessian(positions: "np.ndarray", types: "np.ndarray", params: dict) -> "np.ndarray":
    """Return the exact second-derivative matrix of the converged SCF total energy.

    Parameters
    ----------
    positions : np.ndarray
        Shape (n,) atom positions; n is even, and no two atoms coincide.
    types : np.ndarray
        Shape (n,) element types, 0 or 1.
    params : dict
        Model constants.

    Returns
    -------
    hessian : np.ndarray
        Shape (n, n) float array whose entry (a, b) is the second derivative of
        the converged SCF total energy (the energy the reference-energy routine
        returns) with respect to the positions of atoms a and b. The matrix is
        the analytic one, including the response of the orbitals to the
        displacement, evaluated at the SCF solution converged until the largest
        change of a density element between successive diagonalizations is
        below 1e-12, and is checked to 1e-11 in every entry; it is symmetric to
        that precision.

    Raises
    ------
    ValueError
        If the atom count is odd, or if positions and types disagree in length.
    """
    return hessian
```

### Step 6

equilibrium_bonds

Goal
----
Equilibrium bond lengths of a chain molecule of the site model.

```python
def equilibrium_bonds(types: "np.ndarray", start_bonds: "np.ndarray", params: dict) -> "np.ndarray":
    """Return the bond lengths that minimize the converged SCF total energy.

    Parameters
    ----------
    types : np.ndarray
        Shape (K + 1,) element types of the chain, 0 or 1; K + 1 is even.
    start_bonds : np.ndarray
        Shape (K,) positive starting bond lengths, anywhere between roughly
        half and twice the equilibrium values. Every molecule of this task has
        exactly one minimum with all bonds positive, and it must be reached
        from any such start.
    params : dict
        Model constants.

    Returns
    -------
    bonds : np.ndarray
        Shape (K,) float bond lengths at which every component of the energy
        gradient with respect to the bond lengths has magnitude below 1e-10,
        all bonds being positive. The bond-length gradient is obtained from the
        position gradient: the derivative with respect to bond k is the sum of
        the position derivatives of all atoms that follow bond k.

    Raises
    ------
    ValueError
        If the atom count is odd, if start_bonds does not have K entries, or if
        any starting bond is not positive.
    """
    return bonds
```

### Step 7

normal_modes

Goal
----
Harmonic frequencies and normal modes of a chain molecule in bond-length coordinates.

```python
def normal_modes(types: "np.ndarray", eq_bonds: "np.ndarray", params: dict, step: float = 1e-4) -> "np.ndarray":
    """Return the harmonic frequencies and unit normal-mode vectors.

    Parameters
    ----------
    types : np.ndarray
        Shape (K + 1,) element types of the chain, 0 or 1; K + 1 is even.
    eq_bonds : np.ndarray
        Shape (K,) equilibrium bond lengths.
    params : dict
        Model constants.
    step : float
        Finite-difference step for the curvature matrix.

    Returns
    -------
    modes : np.ndarray
        Shape (K + 1, K) float array. Row 0 holds the K harmonic frequencies
        in increasing order, each the square root of an eigenvalue of the
        curvature matrix. Row k (1-based) holds the unit-length eigenvector of
        the k-th smallest eigenvalue in bond-length coordinates, with its sign
        chosen so that the component of largest magnitude is positive. The
        curvature matrix entry (i, j) is the central difference of bond-gradient
        component i between bond j displaced by +step and by -step, divided by
        twice the step, and the matrix is symmetrized as half the sum of itself
        and its transpose before diagonalization. All eigenvalues are positive
        at an equilibrium geometry.

    Raises
    ------
    ValueError
        If the atom count is odd, eq_bonds does not have K entries, or any
        eigenvalue of the curvature matrix is not positive.
    """
    return modes
```

### Step 8

displace_along_modes

Goal
----
Generate a displaced chain geometry from a normal-mode sample on a quadratic energy shell.

```python
def displace_along_modes(eq_bonds: "np.ndarray", modes: "np.ndarray", omegas: "np.ndarray", eps: float, direction: "np.ndarray") -> "np.ndarray":
    """Return atom positions displaced along normal modes to quadratic energy eps.

    Parameters
    ----------
    eq_bonds : np.ndarray
        Shape (K,) equilibrium bond lengths between consecutive atoms of a
        chain of K + 1 atoms.
    modes : np.ndarray
        Shape (K, K) array whose row k is the unit normal-mode vector k in
        bond-length coordinates.
    omegas : np.ndarray
        Shape (K,) positive harmonic frequencies of the modes.
    eps : float
        Non-negative excess energy of the sample in the local quadratic model.
    direction : np.ndarray
        Shape (K,) nonzero direction of the sample in the coordinates obtained
        by scaling each mode displacement with its frequency; only its
        orientation matters.

    Returns
    -------
    positions : np.ndarray
        Shape (K + 1,) float positions with the first atom at 0 and consecutive
        atoms separated by the displaced bond lengths, which are the equilibrium
        bonds plus each mode vector times its mode displacement, where the mode
        displacements are the point on the quadratic energy shell at eps that
        lies along the given direction in the frequency-scaled coordinates.

    Raises
    ------
    ValueError
        If eps is negative, the direction has zero length, or the array shapes
        are inconsistent.
    """
    return positions
```

### Step 9

apply_workspace_program

Goal
----
Apply a program of matrix primitives to the workspace matrix that approximates the Fock matrix.

```python
def apply_workspace_program(program: "np.ndarray", S: "np.ndarray", H: "np.ndarray", gamma: "np.ndarray") -> "np.ndarray":
    """Return the workspace matrix produced by applying the program to H.

    Parameters
    ----------
    program : np.ndarray
        Shape (N_f,) integer array of primitive indices applied in order.
        The library is: 0, leave M unchanged; 1, add the core Hamiltonian;
        2, add the overlap; 3, replace M by the product M times the inverse
        overlap times M; 4, add the two-electron part of the Fock matrix
        evaluated for the density with one electron on every site (the
        identity matrix); 5, multiply M elementwise by the overlap; 6, halve M.
        An empty program returns H.
    S : np.ndarray
        Shape (n, n) symmetric positive definite overlap matrix.
    H : np.ndarray
        Shape (n, n) core Hamiltonian.
    gamma : np.ndarray
        Shape (n, n) screened Coulomb matrix.

    Returns
    -------
    M : np.ndarray
        Shape (n, n) float workspace matrix after the last primitive.

    Raises
    ------
    ValueError
        If any program entry is not an integer in 0..6, or S, H and gamma are
        not square matrices of one common shape.
    """
    return M
```

### Step 10

program_energy

Goal
----
Predict the total energy of a molecule from a workspace program with a single diagonalization.

```python
def program_energy(program: "np.ndarray", positions: "np.ndarray", types: "np.ndarray", params: dict) -> float:
    """Return the one-shot total energy predicted by a program.

    Parameters
    ----------
    program : np.ndarray
        Shape (N_f,) integer primitive indices in 0..6, applied in order to
        the core Hamiltonian as in the workspace-program library.
    positions : np.ndarray
        Shape (n,) atom positions; n is even.
    types : np.ndarray
        Shape (n,) element types, 0 or 1.
    params : dict
        Model constants.

    Returns
    -------
    energy : float
        The total mean-field energy of the density built from the n/2 lowest
        generalized eigenvectors of the final workspace matrix, using the
        Fock matrix rebuilt from that density, as a native Python float.

    Raises
    ------
    ValueError
        If the atom count is odd, the program contains an index outside 0..6,
        or positions and types disagree in length.
    """
    return energy
```

### Step 11

fit_shifted_loss

Goal
----
Evaluate the shifted root-mean-square loss of a program with optimized per-element energy shifts.

```python
def fit_shifted_loss(pred: list, ref: list, ref_shifts: "np.ndarray", counts: "np.ndarray") -> "np.ndarray":
    """Return the minimized shifted RMSE loss and the fitted per-element shifts.

    Parameters
    ----------
    pred : list
        Length N_m list; entry i is a shape (N_i,) array of program-predicted
        total energies for the retained configurations of molecule i.
    ref : list
        Length N_m list; entry i is a shape (N_i,) array of reference total
        energies for the same configurations, in the same order.
    ref_shifts : np.ndarray
        Shape (N_m,) reference shifts D_i (independent-atom energies).
    counts : np.ndarray
        Shape (N_m, T) integer numbers of atoms of each of the T element
        types in each molecule; must have full column rank T.

    Returns
    -------
    result : np.ndarray
        Shape (1 + T,) float array whose first entry is the smallest
        achievable loss and whose remaining entries are the per-element
        contributions that achieve it. The loss is the root of the mean, over
        all configurations of all molecules, of the squared difference between
        the prediction minus the molecule's assembled shift and the reference
        minus the molecule's independent-atom energy.

    Raises
    ------
    ValueError
        If the list lengths or per-molecule array lengths disagree, if
        ``counts`` does not have full column rank, or if no configuration is
        supplied.
    """
    return result
```

### Step 12

anneal_program

Goal
----
Search the program space by simulated annealing to minimize the shifted training loss.

```python
def anneal_program(molecules: list, params: dict, n_ops: int, n_functions: int, n_iter: int, t_init: float, seed: int) -> "np.ndarray":
    """Return the best loss and program found by simulated annealing.

    Parameters
    ----------
    molecules : list
        Training molecules; each entry is a dict with keys ``positions``
        (shape (N_i, n_i) array, one retained configuration per row),
        ``types`` (shape (n_i,) element types 0 or 1), ``ref_energies``
        (shape (N_i,) reference total energies of those rows) and
        ``ref_shift`` (float independent-atom reference energy D_i). The
        atom-type counts n_{it} are derived from ``types`` for the T element
        types listed in ``params['onsite_energy']``.
    params : dict
        Model constants.
    n_ops : int
        Number of usable primitives: candidate entries are drawn from
        0 .. n_ops - 1 of the library (n_ops <= 7).
    n_functions : int
        Program length N_f >= 1.
    n_iter : int
        Number of annealing iterations.
    t_init : float
        Initial artificial temperature; iteration k (0-based) uses
        T_k = t_init * (1 - k / n_iter).
    seed : int
        Seed of the NumPy generator ``np.random.default_rng(seed)`` that
        drives every random draw, in this order: the initial program
        ``rng.integers(0, n_ops, size=n_functions)``; then per iteration
        ``r = rng.integers(1, 4)`` capped at n_functions, the positions
        ``rng.choice(n_functions, size=r, replace=False)``, the replacement
        primitives ``rng.integers(0, n_ops, size=r)``, and, only when the
        candidate loss exceeds the current loss, one ``rng.random()`` that
        accepts the candidate if it is below exp(-(loss_cand - loss_cur) / T_k).
        A candidate whose loss does not exceed the current loss is accepted
        without a draw.

    Returns
    -------
    result : np.ndarray
        Shape (1 + n_functions,) float array [best_loss, op_1, ..., op_Nf]
        where best_loss is the smallest shifted RMSE loss encountered (initial
        program included) and op_k are the primitive indices of the first
        program that reached it.

    Raises
    ------
    ValueError
        If n_functions < 1, n_ops is outside 1..7, or the training molecules
        cannot determine every per-element shift.
    """
    return result
```

### Step 13

pips_holdout_rmse

Goal
----
Complete workflow: optimize geometries, sample, train a workspace program and score it on a hold-out molecule.

```python
def pips_holdout_rmse(train_specs: list, holdout_spec: dict, params: dict, e_max: float, n_functions: int, n_iter: int, t_init: float, seed: int) -> float:
    """Return the hold-out RMSE of the annealed program with transferred shifts.

    Parameters
    ----------
    train_specs : list
        Training molecules; each entry is a dict with keys ``types`` (shape
        (n,) element types 0 or 1), ``start_bonds`` (shape (n - 1,) starting
        bond lengths for the geometry optimization) and ``samples`` (list of
        (eps, direction) pairs for the normal-mode sampler, directions given
        in the frequency-scaled coordinates of the computed modes, ordered
        by increasing frequency).
    holdout_spec : dict
        Same keys for the hold-out molecule; all of its samples are used.
    params : dict
        Model constants, including ``onsite_energy`` whose entries are the
        isolated-atom energies per element type.
    e_max : float
        A training geometry is retained only if its reference energy minus
        the reference energy of the equilibrium geometry is at most e_max.
    n_functions : int
        Program length for the search over the full seven-primitive library.
    n_iter : int
        Number of annealing iterations.
    t_init : float
        Initial annealing temperature.
    seed : int
        Seed of the annealing generator.

    Returns
    -------
    rmse : float
        Root of the mean squared difference, over the hold-out geometries,
        between the program prediction minus the hold-out shift and the
        reference SCF energy minus the hold-out independent-atom energy. The
        hold-out shift is assembled from the per-element contributions fitted
        on the retained training geometries for the best annealed program.
        Native Python float.

    Raises
    ------
    ValueError
        If no training geometry is retained, if the training molecules cannot
        determine every per-element contribution, or if any molecule has an
        odd atom count.
    """
    return rmse
```
