# Material_Science-Semiconductor_Materials-44

## Background

Cuprous oxide is the material in which the Rydberg exciton series was observed up to very high principal quantum number, and confining it in a quantum well is the natural route to controlling that series. Most treatments of confined excitons in Cu2O use a two-band model in which both carriers have simple parabolic dispersions, so the confinement energies follow the textbook particle-in-a-box result with a scalar electron mass and a scalar hole mass. That picture misses the structure of the valence band.

The uppermost valence band of Cu2O derives from copper d orbitals and transforms in a way that is captured by assigning the hole a quasispin I = 1 in addition to its own spin S = 1/2. The two couple, so the internal space of a hole is six dimensional and splits under the spin-orbit interaction into a twofold level and a fourfold level separated by the spin-orbit parameter Delta. In the bulk these carry the yellow and the green exciton series respectively. Written with the quasispin operators, the spin-orbit term is proportional to the identity plus the scalar product of quasispin and hole spin, so its two eigenvalues are fixed by the two possible values of the coupled angular momentum and nothing else.

The kinetic energy of the hole is the Luttinger-Kohn Hamiltonian, equivalently the Suzuki-Hensel Hamiltonian, and it separates naturally into three groups of terms. One group multiplies the full squared momentum and is isotropic in the momentum components, carrying the leading Luttinger parameter and a correction proportional to the scalar product of quasispin and hole spin. A second group multiplies one squared Cartesian momentum component at a time and carries the square of the matching quasispin component, which is what makes the dispersion depend on the direction of the wave vector within the cubic crystal. A third group multiplies symmetrised products of two different momentum components, one matrix for each unordered pair of axes; two of those three matrices are Hermitian but purely imaginary because the middle quasispin component is. Setting the second and third groups to have equal parameters recovers an isotropic dispersion, so the difference between those parameters is a direct measure of the warping of the band.

In a hard-wall quantum well the growth-direction momentum is no longer a number but an operator on the envelope functions. Its square is diagonal in the particle-in-a-box basis, with the familiar eigenvalues set by the mode index and the well width, but the momentum operator itself has matrix elements between envelopes of opposite parity. Those off-diagonal elements matter because two of the third-group terms contain the growth momentum linearly, multiplied by an in-plane momentum component. For a wave vector along a low-symmetry in-plane direction none of the three pair terms drops out, so both of those envelope-coupling terms contribute alongside the purely in-plane one. The supplied two-component direction vector must be normalised before constructing the Cartesian momentum components; reversing or rescaling that vector therefore represents the same physical in-plane ray. At zero in-plane wave vector every pair term vanishes and the problem separates: each envelope mode carries its own six by six internal problem and the subbands can be read off one mode at a time. At nonzero in-plane wave vector the envelope modes are coupled, and the subband energies come from diagonalising the full product-space matrix.

The consequence is that the in-plane dispersion of a confined hole subband is not parabolic. Close to the zone centre the anisotropic terms stiffen the topmost subband relative to a scalar-mass parabola anchored at the same zone-centre energy, so the exact curve lies slightly above the reference. As the in-plane wave vector grows, coupling to deeper envelope modes and to the other internal components pushes the relevant branch down. Energy ordering alone is not a reliable band label when Kramers doublets cross. The zone-centre doublet must therefore be continued by maximising the overlap between consecutive two-dimensional projectors. On the benchmark branch the exact dispersion crosses the scalar reference once away from zero. Because the crossing is defined by the vanishing of a small difference between two much larger energies, it is bracketed on the tracked path and then refined rather than inferred from a coarse plot.

The crossing is also the natural place to ask what the tracked subband is made of. A confined hole state is not an eigenstate of the spin-orbit interaction: the same terms that bend the dispersion mix the twofold level with the fourfold one, so the zone-centre doublet acquires a small but growing green component. Because time reversal makes every selected subband a two-dimensional Kramers pair, a numerical diagonalisation may return any orthonormal basis inside that degenerate space. The green content is therefore measured by extending the projector onto the fourfold spin-orbit eigenspace over every retained envelope, restricting that full projector to the continuously tracked Kramers subspace, and taking one half of its trace. The two eigenvalues of the restricted projector agree for the time-reversal partners, while their average remains invariant under any numerical rotation of the pair.

## Problem

A cuprous oxide quantum well is grown with the crystal [001] axis along the growth direction z, and the well is thin enough that the surrounding vacuum acts as an infinitely high barrier, so a confined carrier sees a hard wall at |z| = L/2. The uppermost valence band of Cu2O is not a single parabolic band. A hole in it carries a quasispin I = 1 alongside its own spin S = 1/2, so its state has six internal components, which spin-orbit coupling splits into a twofold level carrying the yellow exciton series and a fourfold level carrying the green one. Confinement and in-plane motion mix those levels, and this problem asks how much green character the topmost hole subband has acquired at one particular momentum.

Take the hole Hamiltonian in the Luttinger-Kohn form

  H_h = H_SO + (1 / (2 m0 hbar^2)) * [ hbar^2 (g1 + 4 g2) p^2
        + 2 (e1 + 2 e2) p^2 (I.S)
        - 6 g2 (p_1^2 I_1^2 + c.p.)
        - 12 e2 (p_1^2 I_1 S_1 + c.p.)
        - 12 g3 ({p_1, p_2} {I_1, I_2} + c.p.)
        - 12 e3 ({p_1, p_2} (I_1 S_2 + I_2 S_1) + c.p.) ],

  H_SO = (2/3) * Delta * (1 + (I.S) / hbar^2),

where the indices 1, 2, 3 label the cubic axes x, y, z, "c.p." means the two cyclic permutations of those indices, and the braces are the symmetrised product {A, B} = (A B + B A) / 2. The components of I and of S each satisfy angular-momentum commutation relations.

Use the cuprous oxide valence-band parameters g1 = 1.76, g2 = 0.7532, g3 = -0.3668, e1 = -0.020, e2 = -0.0037, e3 = -0.0337, and the spin-orbit coupling Delta = 131 meV. Take hbar^2 / (2 m0) = 38.0998212 meV nm^2. Energies are in meV, lengths in nm and wave vectors in nm^-1 throughout.

Set the well width to L = 6.4 nm and let the in-plane wave vector k lie along [110], represented by the direction vector (1, 1). Only the direction of this vector matters: it must be normalised before constructing the Cartesian components of k, so rescaling or reversing both components cannot change the spectrum. Along [110], the two in-plane momentum components have equal magnitude and none of the three symmetrised pair terms drops out. Represent the growth-direction motion in the hard-wall envelope basis

  phi_n(z) = sqrt(2/L) cos(n pi z / L)   for odd n,
  phi_n(z) = sqrt(2/L) sin(n pi z / L)   for even n,

on |z| < L/2, and retain the M = 16 lowest of them, n = 1 to 16, so that the hole Hamiltonian at each k is a 96 by 96 Hermitian matrix on the product of the sixteen envelopes with the six internal components. Order that product basis with the envelope index slower and the internal index faster. Keep every term of H_h, including the two pair terms that couple an in-plane momentum to the growth momentum and therefore connect envelopes of opposite parity.

At zero in-plane momentum, define the target subband as the lowest Kramers pair of the confined-hole Hamiltonian, holes being counted with positive confinement energy. Continue that same physical doublet along increasing momentum instead of independently choosing the two lowest eigenvectors at every point. On an increasing path beginning at zero, group the energy-ordered eigenvectors into adjacent Kramers pairs and, at each new point, choose the candidate projector P_candidate that maximises Tr(P_previous P_candidate) with the projector of the previously selected pair. Define E_top(k) as the mean energy of this tracked pair. Its numerical splitting provides a check on the calculation, while its energy-order pair index may change if another doublet crosses it.

Define the scalar-mass reference

  E_ref(k) = E_top(0) + (hbar^2 / (2 m0)) * g1 * k^2,

where k is the magnitude of the in-plane wave vector. This is the dispersion the hole would have with the single mass m0 / g1, anchored so that the two curves agree exactly at k = 0. Their difference therefore vanishes identically there. Track the Kramers branch on 121 equally spaced points from zero to the upper search bound, locate the first strictly positive sign change of E_top(k) - E_ref(k), and refine that bracket. For the benchmark it vanishes once more at exactly one nonzero wave vector k*, below 1.2 nm^-1.

At that wave vector, report the green-level content of the tracked Kramers branch. Let V contain an orthonormal basis for its two-dimensional eigenspace, and extend the projector onto the fourfold spin-orbit level over all sixteen envelopes. Define the reported fraction as one half of the trace of that projector restricted to the tracked Kramers subspace. Equivalently, it is the average of the two eigenvalues of the restricted projector. This definition is independent of the arbitrary basis returned inside the degenerate subspace and gives a dimensionless fraction between zero and one.

In your reasoning, report the intermediate scalars that determine that fraction, in particular the zone-centre subband energy and the crossover wave vector k*. State how the fraction shifts when the number of retained envelopes is changed and when the well width is changed, giving a few values rather than a full table. Give the checks used to confirm that the Hamiltonian, the envelope matrix elements, the continuous Kramers-branch selection and the green projector were constructed correctly, and cite the sources for any material parameter or model convention taken from the literature.

Output Format Requirements:
Emit `<final_answer>` immediately, then `<reasoning>`. Do not write a long derivation before the tags.
You must emit exactly one finite decimal inside `<final_answer>...</final_answer>`, even if the value is approximate or you are unsure.
Rules:
- The tags are required. Do not omit them or leave them empty.
- The tagged value must be a finite decimal (examples: 0.4847, 0.22, 1.05). Not NaN, not Inf, not a fraction string, not a vector, not prose.
- Put only that one number between the tags. No units, no words, no extra lines.
Keep `<reasoning>` short (a few hundred words). Show only the few scalars that determine the final number.
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

01_spin_orbit_hamiltonian

Goal
----
Build the spin-orbit part of the hole Hamiltonian in the coupled quasispin and hole-spin space.

```python
def spin_orbit_hamiltonian(delta: float) -> "np.ndarray":
    '''Spin-orbit part of the cuprous oxide hole Hamiltonian.

    Parameters
    ----------
    delta : float
        Spin-orbit coupling parameter in meV. Must be finite and non-negative.

    Returns
    -------
    numpy.ndarray
        Hermitian array of shape (6, 6) in meV.

    Raises
    ------
    ValueError
        If delta is not finite or is negative.
    '''
    return None  # placeholder
```

### Step 2

02_luttinger_kohn_matrices

Goal
----
Build the internal-space matrices of the Luttinger-Kohn kinetic energy, grouped by momentum dependence.

```python
def luttinger_kohn_matrices(gamma1: float, gamma2: float, gamma3: float, eta1: float, eta2: float, eta3: float) -> "np.ndarray":
    '''Internal-space matrices of the Luttinger-Kohn kinetic energy.

    Parameters
    ----------
    gamma1, gamma2, gamma3 : float
        Luttinger parameters. Must be finite.
    eta1, eta2, eta3 : float
        Quasispin-spin coupling parameters. Must be finite.

    Returns
    -------
    numpy.ndarray
        Complex array of shape (7, 6, 6), dimensionless, each block Hermitian,
        stacked as isotropic, axial x, axial y, axial z, pair xy, pair yz, pair zx.
        The isotropic and axial blocks are real; the xy and yz pair blocks are purely
        imaginary.

    Raises
    ------
    ValueError
        If any argument is not finite.
    '''
    return None  # placeholder
```

### Step 3

03_momentum_matrix_elements

Goal
----
Assemble the matrix of the growth-direction momentum in the hard-wall envelope basis.

```python
def momentum_matrix_elements(width: float, num_modes: int) -> "np.ndarray":
    '''Matrix of the growth-direction momentum in the hard-wall envelope basis.

    Parameters
    ----------
    width : float
        Well width in nm. Must be finite and strictly positive.
    num_modes : int
        Number of envelopes retained, n = 1 .. num_modes. Must be an integer >= 1.

    Returns
    -------
    numpy.ndarray
        Hermitian complex array of shape (num_modes, num_modes) in 1/nm.

    Raises
    ------
    ValueError
        If width is not finite or not positive, or num_modes is not an integer >= 1.
    '''
    return None  # placeholder
```

### Step 4

04_hole_hamiltonian

Goal
----
Assemble the full confined-hole Hamiltonian for an arbitrary in-plane direction.

```python
def hole_hamiltonian(width: float, k_inplane: float, direction: tuple, num_modes: int, params: dict) -> "np.ndarray":
    '''Full confined-hole Hamiltonian for one in-plane wave vector.

    Parameters
    ----------
    width : float
        Well width in nm. Must be finite and strictly positive.
    k_inplane : float
        Magnitude of the in-plane wave vector, in 1/nm. Must be finite and
        non-negative.
    direction : tuple
        Two finite Cartesian components defining the in-plane direction. The vector
        need not be normalised but must have nonzero norm.
    num_modes : int
        Number of envelopes retained. Must be an integer >= 1.
    params : dict
        Must hold the finite floats 'gamma1', 'gamma2', 'gamma3', 'eta1', 'eta2',
        'eta3' and 'delta', with 'delta' strictly positive.

    Returns
    -------
    numpy.ndarray
        Hermitian complex array of shape (6*num_modes, 6*num_modes) in meV.

    Raises
    ------
    ValueError
        If an argument is out of range, direction is not a finite nonzero
        two-vector, or params is incomplete.
    '''
    return None  # placeholder
```

### Step 5

05_topmost_subband_energy

Goal
----
Track the topmost Kramers subspace continuously along an in-plane momentum path.

```python
def tracked_topmost_subband_path(width: float, k_values: "np.ndarray", direction: tuple, num_modes: int, params: dict) -> "np.ndarray":
    '''Gauge-invariant observables for the continuously tracked topmost Kramers pair.

    Parameters
    ----------
    width : float
        Well width in nm. Must be finite and strictly positive.
    k_values : numpy.ndarray
        One-dimensional finite array of wave-vector magnitudes in 1/nm. It must
        start at exactly zero and then increase strictly.
    direction : tuple
        Two finite Cartesian components defining a nonzero in-plane direction.
    num_modes : int
        Number of envelopes retained. Must be an integer >= 1.
    params : dict
        Must hold the finite floats 'gamma1', 'gamma2', 'gamma3', 'eta1', 'eta2',
        'eta3' and 'delta', with 'delta' strictly positive.

    Returns
    -------
    numpy.ndarray
        Float array with one row per path point and columns
        [mean_energy, pair_splitting, green_low, green_high, pair_index].
        Energies are in meV; green contents and pair_index are dimensionless.

    Raises
    ------
    ValueError
        If the path or another argument is invalid.
    '''
    return None  # placeholder
```

### Step 6

06_mixing_crossover_wavevector

Goal
----
Locate the nonzero recrossing on the continuously tracked Kramers branch.

```python
def mixing_crossover_wavevector(width: float, direction: tuple, num_modes: int, params: dict, k_max: float) -> float:
    '''Nonzero recrossing wave vector of the tracked topmost Kramers branch.

    Parameters
    ----------
    width : float
        Well width in nm. Must be finite and strictly positive.
    direction : tuple
        Two finite Cartesian components defining a nonzero in-plane direction.
    num_modes : int
        Number of envelopes retained. Must be an integer >= 1.
    params : dict
        Must hold the finite floats 'gamma1', 'gamma2', 'gamma3', 'eta1', 'eta2',
        'eta3' and 'delta', with 'delta' strictly positive.
    k_max : float
        Strictly positive finite upper search bound in 1/nm.

    Returns
    -------
    float
        First nonzero crossover magnitude strictly between zero and k_max.

    Raises
    ------
    ValueError
        If an argument is invalid or no tracked nonzero crossing is found.
    '''
    return 0.0  # placeholder
```

### Step 7

07_green_admixture_at_crossover

Goal
----
Report the gauge-invariant green content of the tracked Kramers branch at its recrossing.

```python
def green_admixture_at_crossover(width: float, direction: tuple, num_modes: int, params: dict, k_max: float) -> float:
    '''Gauge-invariant green fraction of the tracked topmost pair at its recrossing.

    Parameters
    ----------
    width : float
        Well width in nm. Must be finite and strictly positive.
    direction : tuple
        Two finite Cartesian components defining a nonzero in-plane direction.
    num_modes : int
        Number of envelopes retained. Must be an integer >= 1.
    params : dict
        Must hold the finite floats 'gamma1', 'gamma2', 'gamma3', 'eta1', 'eta2',
        'eta3' and 'delta', with 'delta' strictly positive.
    k_max : float
        Strictly positive finite upper search bound in 1/nm.

    Returns
    -------
    float
        Mean green-projector eigenvalue in the tracked Kramers subspace, between
        zero and one.

    Raises
    ------
    ValueError
        If an argument is invalid or no tracked nonzero crossing is found.
    '''
    return 0.0  # placeholder
```
