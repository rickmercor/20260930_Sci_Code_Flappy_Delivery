# Rigidification penalty of a cyclic water tetramer

## Background

Simulating a molecular system is limited from below by its fastest motion. A covalent stretch oscillates on a timescale a hundred times shorter than the rearrangements that actually interest anyone studying a liquid, a cluster or a protein, yet it sets the integration step and consumes most of the computational effort while contributing almost nothing to the phenomena under study. Coarse-graining is the standard response: replace groups of atoms by a smaller number of effective sites and simulate the reduced system instead. The bottom-up branch of that field insists that the reduced model be derived from the all-atom one rather than fitted to experiment, so that it inherits a defensible connection to the underlying physics — through force matching, relative entropy minimisation, or the formal machinery of the multiscale coarse-graining theory. The price is that the object one obtains is not a potential energy but a free energy: the eliminated coordinates have been integrated out, so they leave behind an entropy, and the resulting surface depends on temperature.

The crudest and most widely used reduction is simply to freeze the fast coordinates. Rigid water models, which fix the O–H distances and the H–O–H angle at values fitted to bulk properties, have underpinned condensed-phase simulation for half a century, and rigid-body treatments of molecular clusters follow the same logic. The compromise is that a frozen monomer cannot respond to its surroundings. A water molecule donating a hydrogen bond really does lengthen and soften the donating O–H bond, by an amount that depends on the strength of that bond; a model that holds the geometry fixed discards this, and with it the mechanism by which the local environment feeds back on the intramolecular force constants. An intermediate position has been available for some time in the form of elastic network and local harmonic approximations, which keep the fast modes but describe them quadratically. Pushing that idea further gives a rigidification scheme in which, for each configuration of the rigid units, the fast coordinates are relaxed to mechanical equilibrium and then charged the harmonic free energy of their own normal modes — an approach that turns out to reproduce all-atom thermodynamics to within about a wavenumber for small water and ammonia clusters.

Accuracy of that kind is worth little if it costs more than the calculation it replaces, and in its original form it did. Every evaluation of the coarse-grained free energy required a constrained minimisation over the fast subspace followed by a normal-mode analysis, and diagonalising the full second-derivative matrix of an *N*-atom system scales as the cube of the number of coordinates. A single coarse-grained energy therefore cost far more than a single all-atom energy, inverting the entire purpose of the exercise. Two economies recover the position. Replacing the iterative minimisation by a fixed, very small number of Newton steps exploits the fact that the surface along the stiff directions is nearly quadratic, so one step is exact for a harmonic system and two suffice in practice. Truncating the second-derivative matrix to its diagonal blocks, one per rigid unit, replaces a single large diagonalisation by many tiny ones — the same building-block strategy long used to extract low-frequency modes of macromolecules — and is justified because the modes being priced are localised on individual units while the discarded couplings are precisely the slow, intermolecular ones the coarse-grained coordinates already describe explicitly.

What makes the intramolecular modes worth this much care, particularly for water, is that they are the modes carrying the zero-point energy. An O–H stretch near 3800 cm⁻¹ has a vibrational level spacing more than twenty times the thermal energy at room temperature, so its thermal occupation is negligible, but it does contribute several kcal/mol of zero-point energy — and that contribution is linear in the frequency rather than logarithmic, so it responds far more sharply to a shift in force constant than any classical estimate would suggest. Lower-frequency bends have small but nonzero occupations that must still be retained when a final result is graded to six significant figures. Empirical water potentials designed for path-integral simulation, in which nuclear quantum effects are treated explicitly, are built with anharmonic intramolecular terms for exactly this reason. A coarse-grained model that misplaces the intramolecular frequencies therefore misplaces the zero-point energy, and does so in a way that is invisible to any purely classical validation. The same harmonic parameters that expose this also provide the remedy for a separate deficiency: because the reduced configurations carry no intramolecular fluctuation at all, distributions of bond lengths and angles evaluated on them collapse to spikes, and reinstating a Gaussian spread along the retained modes recovers all-atom structural observables from a coarse-grained trajectory at essentially no cost.

## Problem

Bottom-up coarse-graining of a molecular cluster onto a set of rigid monomers replaces the fast intramolecular degrees of freedom by a free energy, and the fidelity of the resulting model is decided entirely by what is done with those degrees of freedom before they are integrated out. Freezing every monomer at one fixed reference shape is the conventional treatment; letting the stiff coordinates settle to mechanical equilibrium at each coarse-grained configuration, and then charging the local harmonic free energy of the relaxed modes, is a recent alternative that recovers near-exact thermodynamic consistency at a cost competitive with the underlying all-atom model. What the conventional treatment costs is not part of the rigidification theory, and it is what you are asked to compute.

The all-atom system is a cyclic $(\mathrm{H_2O})_4$ cluster in vacuum described by the q-TIP4P/F flexible water potential, with every intramolecular term and every intermolecular pair retained and no cutoff. Atoms are ordered O, H, H within each monomer, and the Cartesian coordinates in angstrom are

| atom | $x$ | $y$ | $z$ |
| --- | --- | --- | --- |
| O$_1$ |   1.965757 |   0.000000 |   0.000000 |
| H$_1$ |   1.276254 |   0.689503 |   0.097837 |
| H$_1$ |   2.425271 |   0.000868 |   0.837181 |
| O$_2$ |   0.000000 |   1.965757 |   0.000000 |
| H$_2$ |  -0.680176 |   1.285581 |  -0.077118 |
| H$_2$ |   0.004437 |   2.439238 |  -0.852304 |
| O$_3$ |  -1.965757 |   0.000000 |   0.000000 |
| H$_3$ |  -1.270755 |  -0.695002 |   0.118515 |
| H$_3$ |  -2.428412 |   0.000015 |   0.841160 |
| O$_4$ |   0.000000 |  -1.965757 |   0.000000 |
| H$_4$ |   0.684659 |  -1.281098 |  -0.058165 |
| H$_4$ |   0.015394 |  -2.427969 |  -0.869683 |

Take $m_\mathrm{O} = 15.9994$ u and $m_\mathrm{H} = 1.00794$ u, work at $T = 250$ K, and use two relaxation iterations.

Write $F_\mathrm{fix}$ for the coarse-grained free energy of the unrelaxed treatment and $F_\mathrm{rel}$ for that of the relaxed treatment, both evaluated for the single coarse-grained configuration — the monomer positions and orientations — carried by the cluster above, and each charged the local harmonic free energy of its own stiff modes. Use the position-orientation map of Paper I: for each water monomer retain the oxygen (the first and heaviest atom) as its position; take the local y-axis along the vector from that oxygen to the midpoint of its two hydrogens, the local x-axis along the component of H2-H1 perpendicular to y, and complete a right-handed frame. Build the unrelaxed configuration by placing the isolated-monomer minimum at those retained oxygen positions and Jacobi-frame orientations. Their difference, made intensive as

$$\Lambda = \frac{F_\mathrm{fix} - F_\mathrm{rel}}{L\,k_\mathrm{B}T}$$

with $L$ the total number of stiff degrees of freedom in the cluster, is the rigidification penalty, and it depends on whether those modes are treated as classical or as quantum oscillators. Your final answer must be a single number: the ratio $\Lambda_\mathrm{quantum}/\Lambda_\mathrm{classical}$ of the penalty obtained with the quantum harmonic free energy to the penalty obtained in the classical limit; quote it to six significant figures. Second derivatives are to be taken as central differences of the analytic gradient, with the displacement refined until those six figures no longer move. The few scalars that determine that number, and which you must therefore state, are these: the surface parameters and the reference monomer geometry you adopt; the number of stiff degrees of freedom you count; the potential energy of each configuration; the mode spectrum of a monomer block at each, and the factor converting those mass-scaled eigenvalues to absolute frequencies; how far the relaxation moves the atoms; how the stretch and bend level spacings compare with the thermal energy; the vibrational contribution to each of the two free-energy differences, including the quantum occupation correction, the frequency-ratio sum behind the classical one, and the share of each difference that its vibrational term supplies; and the two penalties separately. State also how a classical canonical all-atom ensemble is regenerated from a relaxed coarse-grained configuration for negligible additional cost.


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
## Output format

Scientific reasoning wrapped in <reasoning>...</reasoning> tags. Include enough intermediate calculations to justify the deterministic computation, without turning the response into a general pipeline summary. A single final numeric answer wrapped in <final_answer>...</final_answer> tags.
```

## Your task

Implement **all 12 functions** below into a single file at `/app/solution.py`. Requirements:

- Use the EXACT function name and signature shown for each step.
- Define every function at module scope (importable).
- Do NOT include tests, example usage, prints, or a `__main__` block.
- Available libraries: numpy, scipy, sympy, pandas, scikit-learn, networkx.

Start your file with these imports:

```python
import numpy as np
```

### Step 1

01_compute_monomer_reference_geometry

Goal
----
Minimize the intramolecular potential of a single isolated water monomer and return the resulting oxygen-centred Cartesian geometry in the Jacobi body frame of Paper I.

```python
import numpy as np

def compute_monomer_reference_geometry(d_r: float = 116.09, alpha_r: float = 2.287,
                                       r_eq: float = 0.9419, k_theta: float = 87.85,
                                       theta_eq_deg: float = 107.4) -> np.ndarray:
    """Return the isolated-monomer minimum geometry in its Jacobi body frame.

    Parameters
    ----------
    d_r : float
        Dissociation energy of the quartic Morse O-H stretch in kcal/mol
        (d_r > 0).
    alpha_r : float
        Morse range parameter of the O-H stretch in inverse angstrom
        (alpha_r > 0).
    r_eq : float
        Equilibrium O-H bond length in angstrom (r_eq > 0).
    k_theta : float
        Harmonic force constant of the H-O-H bend in kcal/(mol rad**2)
        (k_theta > 0).
    theta_eq_deg : float
        Equilibrium H-O-H angle in degrees, strictly between 0 and 180.

    Returns
    -------
    geometry : np.ndarray
        Array of shape (3, 3) in angstrom holding the O, H and H positions of
        the energy-minimized isolated monomer, with the oxygen at the origin,
        the H-O-H bisector along +y and the projected H2-H1 vector along +x.

    Raises
    ------
    ValueError
        If a stretch or bend parameter is non-finite or non-positive, if
        ``theta_eq_deg`` is outside the open interval (0, 180), or if the
        isolated-monomer minimization cannot produce a physical geometry.
    """
    return geometry  # placeholder
```

### Step 2

02_compute_potential_energy

Goal
----
Evaluate the all-atom q-TIP4P/F potential energy of a water cluster.

```python
import numpy as np

def compute_potential_energy(coords: np.ndarray, d_r: float = 116.09,
                             alpha_r: float = 2.287, r_eq: float = 0.9419,
                             k_theta: float = 87.85, theta_eq_deg: float = 107.4,
                             epsilon: float = 0.1852, sigma: float = 3.1589,
                             q_m: float = 1.1128, gamma: float = 0.73612,
                             coulomb_constant: float = 332.06371) -> float:
    """Evaluate the q-TIP4P/F potential energy of a water cluster.

    Parameters
    ----------
    coords : np.ndarray
        Array of shape (3 * n_monomers, 3) in angstrom holding the atomic
        positions, ordered O, H, H within each monomer.
    d_r : float
        Dissociation energy of the quartic Morse O-H stretch in kcal/mol
        (d_r > 0).
    alpha_r : float
        Morse range parameter of the O-H stretch in inverse angstrom
        (alpha_r > 0).
    r_eq : float
        Equilibrium O-H bond length in angstrom (r_eq > 0).
    k_theta : float
        Harmonic force constant of the H-O-H bend in kcal/(mol rad**2)
        (k_theta > 0).
    theta_eq_deg : float
        Equilibrium H-O-H angle in degrees, strictly between 0 and 180.
    epsilon : float
        Oxygen-oxygen Lennard-Jones well depth in kcal/mol (epsilon > 0).
    sigma : float
        Oxygen-oxygen Lennard-Jones diameter in angstrom (sigma > 0).
    q_m : float
        Magnitude of the negative charge on the massless site, in elementary
        charges; each hydrogen carries half of it with the opposite sign
        (q_m > 0).
    gamma : float
        Fraction defining the position of the massless charge site along the
        bisector, strictly between 0 and 1.
    coulomb_constant : float
        Electrostatic prefactor in kcal angstrom /(mol e**2)
        (coulomb_constant > 0).

    Returns
    -------
    energy : float
        Total potential energy of the cluster in kcal/mol, as a native Python
        float.

    Raises
    ------
    ValueError
        If ``coords`` has the wrong shape, does not contain complete O-H-H
        monomers, or contains non-finite entries; if a positive surface
        parameter is invalid; if ``theta_eq_deg`` or ``gamma`` lies outside
        its stated open interval; or if a required intersite distance is zero.
    """
    return energy  # placeholder
```

### Step 3

03_compute_potential_gradient

Goal
----
Evaluate the analytic gradient of the all-atom q-TIP4P/F potential with respect to every atomic coordinate.

```python
import numpy as np

def compute_potential_gradient(coords: np.ndarray, d_r: float = 116.09,
                               alpha_r: float = 2.287, r_eq: float = 0.9419,
                               k_theta: float = 87.85, theta_eq_deg: float = 107.4,
                               epsilon: float = 0.1852, sigma: float = 3.1589,
                               q_m: float = 1.1128, gamma: float = 0.73612,
                               coulomb_constant: float = 332.06371) -> np.ndarray:
    """Evaluate the analytic Cartesian gradient of the q-TIP4P/F potential.

    Parameters
    ----------
    coords : np.ndarray
        Array of shape (3 * n_monomers, 3) in angstrom holding the atomic
        positions, ordered O, H, H within each monomer.
    d_r : float
        Dissociation energy of the quartic Morse O-H stretch in kcal/mol
        (d_r > 0).
    alpha_r : float
        Morse range parameter of the O-H stretch in inverse angstrom
        (alpha_r > 0).
    r_eq : float
        Equilibrium O-H bond length in angstrom (r_eq > 0).
    k_theta : float
        Harmonic force constant of the H-O-H bend in kcal/(mol rad**2)
        (k_theta > 0).
    theta_eq_deg : float
        Equilibrium H-O-H angle in degrees, strictly between 0 and 180.
    epsilon : float
        Oxygen-oxygen Lennard-Jones well depth in kcal/mol (epsilon > 0).
    sigma : float
        Oxygen-oxygen Lennard-Jones diameter in angstrom (sigma > 0).
    q_m : float
        Magnitude of the negative charge on the massless site, in elementary
        charges; each hydrogen carries half of it with the opposite sign
        (q_m > 0).
    gamma : float
        Fraction defining the position of the massless charge site along the
        bisector, strictly between 0 and 1.
    coulomb_constant : float
        Electrostatic prefactor in kcal angstrom /(mol e**2)
        (coulomb_constant > 0).

    Returns
    -------
    gradient : np.ndarray
        Array with the same shape as ``coords`` holding the derivative of the
        energy with respect to every atomic coordinate, in
        kcal/(mol angstrom).

    Raises
    ------
    ValueError
        If ``coords`` has the wrong shape, does not contain complete O-H-H
        monomers, or contains non-finite entries; if a positive surface
        parameter is invalid; if ``theta_eq_deg`` or ``gamma`` lies outside
        its stated open interval; or if a required intersite distance is zero.
    """
    return gradient  # placeholder
```

### Step 4

04_build_reference_configuration

Goal
----
Rebuild every monomer of a cluster at the isolated-monomer reference geometry while preserving its centre of mass and its orientation, producing the zeroth-order point of the reference manifold.

```python
import numpy as np

def build_reference_configuration(coords: np.ndarray,
                                  monomer_geometry: np.ndarray) -> np.ndarray:
    """Place the reference monomer geometry onto every monomer of a cluster.

    Parameters
    ----------
    coords : np.ndarray
        Array of shape (3 * n_monomers, 3) in angstrom holding the atomic
        positions of the input cluster, ordered O, H, H within each monomer.
    monomer_geometry : np.ndarray
        Array of shape (3, 3) in angstrom holding the reference monomer
        geometry. Its first row is the oxygen position; its Jacobi frame
        defines the body-frame orientation.

    Returns
    -------
    reference_coords : np.ndarray
        Array with the same shape as ``coords`` in angstrom, holding the
        cluster with every monomer rebuilt at the reference geometry in its
        original position and orientation.

    Raises
    ------
    ValueError
        If either coordinate array has the wrong shape or non-finite entries,
        if ``coords`` does not contain complete three-atom monomers, or if a
        monomer has a degenerate bisector or projected H2-H1 Jacobi axis.
    """
    return reference_coords  # placeholder
```

### Step 5

05_compute_block_hessian

Goal
----
Build the mass-scaled second-derivative matrix of the all-atom potential truncated to one diagonal block per monomer.

```python
import numpy as np

def compute_block_hessian(coords: np.ndarray, atom_masses: np.ndarray,
                          gradient_fn, step: float = 1.0e-5) -> np.ndarray:
    """Build the mass-scaled block-diagonal Hessian of the cluster potential.

    Parameters
    ----------
    coords : np.ndarray
        Array of shape (3 * n_monomers, 3) in angstrom holding the atomic
        positions, ordered O, H, H within each monomer.
    atom_masses : np.ndarray
        Array of shape (3 * n_monomers,) in unified atomic mass units holding
        the mass of every atom, in the same order as ``coords``. All entries
        must be strictly positive.
    gradient_fn : callable
        Analytic gradient of the potential being differentiated: it maps an
        array of positions of the same shape as ``coords`` to the derivative of
        the energy with respect to every atomic coordinate, an array of that
        same shape. For the cluster of this task it is the function built in
        sub-problem 03, called with its default q-TIP4P/F parameters.
    step : float
        Central-difference displacement in angstrom used to differentiate the
        analytic gradient (step > 0).

    Returns
    -------
    blocks : np.ndarray
        Array of shape (n_monomers, 9, 9) holding one symmetric mass-scaled
        second-derivative block per monomer, in kcal/(mol angstrom**2 u). The
        row and column order within a block is the atom-major flattening
        O(x, y, z), H(x, y, z), H(x, y, z).

    Raises
    ------
    ValueError
        If ``coords`` has the wrong shape, atom count, or non-finite entries;
        if ``atom_masses`` has a mismatched shape or any non-finite or
        non-positive entry; if ``gradient_fn`` is not callable; or if ``step``
        is non-finite or non-positive.
    """
    return blocks  # placeholder
```

### Step 6

06_build_subspace_pseudoinverse

Goal
----
Invert each mass-scaled Hessian block on the subspace spanned by its stiffest eigenvectors only, producing the operator that projects a gradient onto the fast degrees of freedom.

```python
import numpy as np

def build_subspace_pseudoinverse(blocks: np.ndarray, n_stiff: int = 3) -> np.ndarray:
    """Invert each Hessian block on the span of its stiffest eigenvectors.

    Parameters
    ----------
    blocks : np.ndarray
        Array of shape (n_monomers, n_dim, n_dim) holding one symmetric
        mass-scaled second-derivative block per monomer.
    n_stiff : int
        Number of stiff degrees of freedom retained per block, at least 1 and
        at most n_dim.

    Returns
    -------
    pseudoinverse : np.ndarray
        Array with the same shape as ``blocks`` holding the subspace
        pseudoinverse of each block, in mol angstrom**2 u/kcal.

    Raises
    ------
    ValueError
        If ``blocks`` is empty, non-finite, non-square, or non-symmetric; if
        ``n_stiff`` is not an integer between one and the block dimension; or
        if any retained stiff eigenvalue is non-positive.
    """
    return pseudoinverse  # placeholder
```

### Step 7

07_relax_subspace_newton_raphson

Goal
----
Drive the monomers to mechanical equilibrium along their stiff degrees of freedom with a fixed number of Newton-Raphson steps preconditioned by the subspace pseudoinverse.

```python
import numpy as np

def relax_subspace_newton_raphson(coords: np.ndarray, pseudoinverse: np.ndarray,
                                  atom_masses: np.ndarray, gradient_fn,
                                  n_iterations: int = 2) -> np.ndarray:
    """Relax a cluster along its stiff subspace by fixed-count Newton steps.

    Parameters
    ----------
    coords : np.ndarray
        Array of shape (3 * n_monomers, 3) in angstrom holding the starting
        configuration, ordered O, H, H within each monomer.
    pseudoinverse : np.ndarray
        Array of shape (n_monomers, 9, 9) holding the stiff-subspace
        pseudoinverse of each mass-scaled Hessian block, built once at the
        starting configuration and held fixed throughout.
    atom_masses : np.ndarray
        Array of shape (3 * n_monomers,) in unified atomic mass units holding
        the mass of every atom, in the same order as ``coords``. All entries
        must be strictly positive.
    gradient_fn : callable
        Analytic gradient of the potential being relaxed on: it maps an array
        of positions of the same shape as ``coords`` to the derivative of the
        energy with respect to every atomic coordinate, an array of that same
        shape. It must be the gradient of the same potential whose curvature
        produced ``pseudoinverse``. For the cluster of this task it is the
        function built in sub-problem 03, called with its default q-TIP4P/F
        parameters.
    n_iterations : int
        Number of Newton-Raphson iterations to perform (n_iterations >= 0);
        zero returns the starting configuration unchanged.

    Returns
    -------
    relaxed_coords : np.ndarray
        Array with the same shape as ``coords`` in angstrom, holding the
        configuration after the prescribed number of subspace Newton steps.

    Raises
    ------
    ValueError
        If the coordinate, pseudoinverse, or mass arrays have incompatible
        shapes or non-finite entries; if any mass is non-positive; if
        ``gradient_fn`` is not callable; or if ``n_iterations`` is not a
        non-negative integer.
    """
    return relaxed_coords  # placeholder
```

### Step 8

08_compute_stiff_mode_frequencies

Goal
----
Extract the stiff normal-mode angular frequencies from each mass-scaled Hessian block.

```python
import numpy as np


def compute_stiff_mode_frequencies(blocks: np.ndarray, n_stiff: int = 3) -> np.ndarray:
    """Return the stiff angular frequencies of each mass-scaled Hessian block.

    Parameters
    ----------
    blocks : np.ndarray
        Array of shape (n_monomers, n_dim, n_dim) holding one symmetric
        mass-scaled second-derivative block per monomer.
    n_stiff : int
        Number of stiff degrees of freedom retained per block, at least 1 and
        at most n_dim.

    Returns
    -------
    frequencies : np.ndarray
        Array of shape (n_monomers, n_stiff) holding the angular frequency of
        each retained mode in (kcal/(mol angstrom**2 u))**(1/2), ordered from
        stiffest to softest within each monomer.

    Raises
    ------
    ValueError
        If ``blocks`` is empty, non-finite, non-square, or non-symmetric; if
        ``n_stiff`` is not an integer between one and the block dimension; or
        if any retained stiff eigenvalue is non-positive.
    """
    return frequencies  # placeholder
```

### Step 9

09_compute_stiff_mode_vectors

Goal
----
Extract the mass-scaled eigenvectors of the stiff normal modes from each Hessian block, under a fixed sign convention.

```python
import numpy as np

def compute_stiff_mode_vectors(blocks: np.ndarray, n_stiff: int = 3) -> np.ndarray:
    """Return the stiff mode vectors of each mass-scaled Hessian block.

    Parameters
    ----------
    blocks : np.ndarray
        Array of shape (n_monomers, n_dim, n_dim) holding one symmetric
        mass-scaled second-derivative block per monomer.
    n_stiff : int
        Number of stiff degrees of freedom retained per block, at least 1 and
        at most n_dim.

    Returns
    -------
    modes : np.ndarray
        Array of shape (n_monomers, n_dim, n_stiff) whose columns are the
        orthonormal mass-scaled eigenvectors of the retained stiff modes,
        ordered from stiffest to softest within each monomer. Each column is
        sign-fixed so that its entry of largest magnitude is positive, ties
        being resolved in favour of the lowest index.

    Raises
    ------
    ValueError
        If ``blocks`` is empty, non-finite, non-square, or non-symmetric; if
        ``n_stiff`` is not an integer between one and the block dimension; or
        if any retained stiff eigenvalue is non-positive.
    """
    return modes  # placeholder
```

### Step 10

10_compute_cg_free_energy

Goal
----
Add the classical harmonic free energy of the stiff modes to the potential energy of a configuration to obtain the coarse-grained free energy at a given temperature.

```python
import numpy as np

def compute_cg_free_energy(potential_energy: float, frequencies: np.ndarray,
                           temperature: float = 250.0,
                           quantum: bool = False) -> float:
    """Return the coarse-grained free energy of a configuration.

    Parameters
    ----------
    potential_energy : float
        All-atom potential energy of the configuration in kcal/mol.
    frequencies : np.ndarray
        Array of shape (n_monomers, n_stiff) holding the stiff angular
        frequencies in (kcal/(mol angstrom**2 u))**(1/2). All entries must be
        strictly positive.
    temperature : float
        Absolute temperature in kelvin (temperature > 0).
    quantum : bool
        When False, charge each mode the classical harmonic free energy; when
        True, charge the quantum harmonic free energy instead.

    Returns
    -------
    free_energy : float
        Coarse-grained free energy in kcal/mol, as a native Python float,
        equal to the potential energy plus the harmonic free energy of every
        retained stiff mode in the selected regime.

    Raises
    ------
    ValueError
        If ``potential_energy`` is non-finite; if ``frequencies`` has the
        wrong shape or any non-finite or non-positive entry; if ``temperature``
        is non-finite or non-positive; or if ``quantum`` is not boolean.
    """
    return free_energy  # placeholder
```

### Step 11

11_backmap_thermal_configuration

Goal
----
Regenerate a canonical all-atom configuration from a relaxed coarse-grained configuration by adding harmonic thermal displacements along the stiff modes.

```python
import numpy as np

def backmap_thermal_configuration(coords: np.ndarray, frequencies: np.ndarray,
                                  modes: np.ndarray, atom_masses: np.ndarray,
                                  temperature: float = 250.0,
                                  seed: int = 0) -> np.ndarray:
    """Add harmonic thermal displacements to a relaxed configuration.

    Random numbers are drawn from ``np.random.default_rng(seed)``, taking one
    call of size ``n_stiff`` per monomer in increasing monomer order.

    Parameters
    ----------
    coords : np.ndarray
        Array of shape (3 * n_monomers, 3) in angstrom holding the relaxed
        configuration, ordered O, H, H within each monomer.
    frequencies : np.ndarray
        Array of shape (n_monomers, n_stiff) holding the stiff angular
        frequencies in (kcal/(mol angstrom**2 u))**(1/2). All entries must be
        strictly positive.
    modes : np.ndarray
        Array of shape (n_monomers, 9, n_stiff) whose columns are the
        orthonormal mass-scaled mode vectors matching ``frequencies``.
    atom_masses : np.ndarray
        Array of shape (3 * n_monomers,) in unified atomic mass units holding
        the mass of every atom, in the same order as ``coords``. All entries
        must be strictly positive.
    temperature : float
        Absolute temperature in kelvin (temperature > 0).
    seed : int
        Seed of the random generator; the result must be reproducible.

    Returns
    -------
    sampled_coords : np.ndarray
        Array with the same shape as ``coords`` in angstrom, holding one
        all-atom configuration drawn from the classical harmonic ensemble
        about the relaxed configuration.

    Raises
    ------
    ValueError
        If the coordinate, frequency, mode, or mass arrays have incompatible
        shapes or non-finite entries; if a frequency or mass is non-positive;
        if ``temperature`` is non-finite or non-positive; or if ``seed`` is
        not a non-negative integer.
    """
    return sampled_coords  # placeholder
```

### Step 12

12_run_shr_pipeline

Goal
----
Chain the sub-problem functions 01-11 end to end on the cyclic water tetramer and return the ratio of its quantum to its classical dimensionless rigidification penalty.

```python
import numpy as np

def run_shr_pipeline(coords: np.ndarray = None, temperature: float = 250.0,
                     n_iterations: int = 2, n_stiff: int = 3,
                     mass_o: float = 15.9994, mass_h: float = 1.00794,
                     hessian_step: float = 1.0e-5) -> float:
    """Run the full rigidification-penalty measurement on a water cluster.

    Parameters
    ----------
    coords : np.ndarray or None
        Array of shape (3 * n_monomers, 3) in angstrom holding the all-atom
        cluster, ordered O, H, H within each monomer. ``None`` selects the
        benchmark cyclic tetramer of the problem statement.
    temperature : float
        Absolute temperature in kelvin (temperature > 0).
    n_iterations : int
        Number of subspace Newton-Raphson iterations used by the relaxed
        treatment (n_iterations >= 1; zero would make the two treatments
        identical and the reported ratio undefined).
    n_stiff : int
        Number of stiff intramolecular degrees of freedom per monomer, at
        least 1 and at most 9.
    mass_o : float
        Oxygen mass in unified atomic mass units (mass_o > 0).
    mass_h : float
        Hydrogen mass in unified atomic mass units (mass_h > 0).
    hessian_step : float
        Central-difference displacement in angstrom used for the block
        Hessians (hessian_step > 0).

    Returns
    -------
    amplification : float
        Dimensionless ratio of the quantum rigidification penalty to the
        classical one, each penalty being the free-energy excess of the frozen
        treatment over the relaxed one per stiff degree of freedom in units of
        the thermal energy, as a native Python float.

    Raises
    ------
    ValueError
        If ``coords`` has the wrong shape, atom count, or non-finite entries;
        if the temperature, masses, or Hessian step are non-finite or
        non-positive; if ``n_iterations`` is negative or ``n_stiff`` is not an
        integer from 1 through 9; if reconstruction violates its contract; or
        if zero iterations makes the classical penalty and ratio undefined.
    """
    return amplification  # placeholder
```
