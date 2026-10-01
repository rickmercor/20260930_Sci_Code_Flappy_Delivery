# Chemistry-Quantum_Chemistry-64

## Background

Attosecond and few-femtosecond light sources now probe electrons on their own time scale, and the simulations that interpret these experiments solve the time-dependent Schrödinger equation for the electrons of a molecule driven by an explicit laser field. Real-time electronic-structure methods do this by propagating a parametrised wave function step by step in time, and quantities such as the induced dipole moment, absorption spectra, transient pump-probe signals and charge migration follow from the propagated state.

Coupled-cluster theory is the most accurate widely used family of such methods for molecules near their equilibrium structure. In its time-dependent form the state is not a single normalised wave function: an exponential cluster ket and a separately parametrised bra are propagated together under a time-dependent bivariational principle, and physical observables are computed as their bra-ket expectation values. This construction keeps the method size-extensive and allows systematic improvement through the truncation level of the cluster operator, but it also means that the amplitudes carry no direct probabilistic meaning, which makes a chemical reading of the propagated state less direct than for a normalised wave function.

Chemical interpretation of electron dynamics is usually phrased in terms of orbital excitations: which occupied orbital loses an electron, which virtual orbital receives it, and which excited configurations are populated after the pulse. For stationary excited states this information comes from the eigenvectors of equation-of-motion or response calculations, which have to be solved in addition to the real-time simulation. Extracting it directly from the propagated state is attractive because a single broadband simulation already contains the response of every excited state the pulse can reach, including configurations that are dipole-forbidden from the ground state and are populated only through sequences of field interactions or through electron correlation. Such dark configurations are central to stimulated Raman processes and to the ultrafast redistribution of electronic population in strong fields.

Two practical aspects shape every real-time simulation of this kind. Laser pulses are represented by carrier waves under smooth envelopes, and envelopes with strictly finite support make the start and end of the matter-field interaction unambiguous. The equations of motion are stiff enough that explicit integrators need small, fixed time steps, and truncated coupled-cluster dynamics can become unstable when the field depletes the reference configuration too strongly, so the field strength and duration must be chosen with the stability of the method in mind.

## Problem

Real-time coupled-cluster simulations can follow the electrons of a molecule through an intense, few-femtosecond laser pulse with high accuracy, but the propagated amplitudes do not say in chemical terms which electronic configurations the pulse populates. A recent analysis closes this gap by assigning every Slater determinant a configuration weight at every instant, built from both the ket and the independently propagated bra of the bivariational time-dependent coupled-cluster state, so that populations can be followed in real time, including those of determinants that the field cannot reach directly from the reference because the dipole transition to them is forbidden by symmetry. The calculation below takes a molecular geometry, a Gaussian basis and a laser pulse, and returns the change that the pulse leaves in the weight of such a dark determinant.

Simulate a linear chain of four hydrogen atoms with the following configuration, in atomic units throughout:

- Geometry: the atoms lie on the z axis at z = -2.7, -0.9, 0.9 and 2.7 bohr (equal spacing of 1.8 bohr, centred on the origin).
- Basis on every atom: the 6-31G hydrogen basis, s functions only. The inner function is contracted from primitives with exponents 18.7311370, 2.8253944 and 0.6401217 bohr^-2 and coefficients 0.03349460, 0.23472695 and 0.81375733 (for normalised primitives); the outer function is a single primitive with exponent 0.1612778 bohr^-2. Normalise every primitive and rescale each contracted function to unit self-overlap.
- Reference: the lowest-energy closed-shell restricted Hartree-Fock determinant, with its two lowest canonical orbitals doubly occupied. All later quantities use its canonical orbitals, and every electron and every orbital is correlated.
- Ground state: the coupled-cluster singles and doubles (CCSD) ground state together with its de-excitation (Lambda) amplitudes, which define the bra <Phi0|(1 + Lambda) exp(-T) of the state. Converge both sets of ground-state equations until every residual is below 1e-10.
- Field: a z-polarised pulse in the length gauge. The electrons carry charge -1, so the field adds E(t) times the sum of the electronic z coordinates to the Hamiltonian, with E(t) = 0.02 cos(0.38 t) cos^10(pi t / 60) for |t| <= 30 and E(t) = 0 otherwise.
- Dynamics: starting from the CCSD ground state at t = -30, propagate both the cluster and the de-excitation amplitudes of the bivariational time-dependent CCSD state through the whole pulse to t = +30, using the classical fourth-order Runge-Kutta method with a fixed step of 0.05 (1200 steps) and the field evaluated at each stage time. The phase of the unit operator in T does not need to be propagated.

The target determinants promote one electron from the lowest canonical orbital (HOMO-1) to the lowest virtual orbital (LUMO). Both orbitals are symmetric under inversion through the centre of the chain, so this excitation is dipole dark. Take their configuration weight as the time-dependent coupled-cluster analysis defines it for a bra-ket pair, summed over the two spin orientations of the promoted electron, and evaluate it for the CCSD ground state and for the state at t = +30.

The final answer is the weight at t = +30 minus the ground-state weight, to at least eight decimal places. In the reasoning, write out the definition you use for the configuration weight of a determinant, the equation of motion you propagate for the de-excitation amplitudes, and the expression of the bra coefficient of a singly excited determinant in terms of the amplitudes, and show the following scalars, each as a single value to at least six decimal places (energies in hartree): the restricted Hartree-Fock total energy and the CCSD correlation energy; for the CCSD ground state, the reference weight, the total weight of all double excitations and the summed HOMO-1 to LUMO weight, the last to at least eight decimal places; at t = +30, the reference weight, the total weight of all double excitations and the summed weight of the HOMO to LUMO excitation; and the smallest reference weight on the 1201 times of the propagation grid together with the time at which it occurs.

Using external literature beyond the supplied article, also explain why the conventional linear coupled-cluster bra can give ill-behaved configuration weights for noninteracting subsystems; report the published noninteracting-helium example that demonstrates the size-consistency limitation of the reference weight; and state the Hamiltonian-structure and integrator-stability conclusions reported for bivariational time-dependent coupled-cluster propagation.

Output Format Requirements:
Emit <final_answer> immediately, then <reasoning>. Do not write a long derivation before the tags.
You must emit exactly one finite decimal inside <final_answer>...</final_answer>, even if the value is approximate or you are unsure.
Rules:
- The tags are required. Do not omit them or leave them empty.
- The tagged value must be a finite decimal (examples: 0.4847, 0.22, 1.05). Not NaN, not Inf, not a fraction string, not a vector, not prose.
- Put only that one number between the tags. No units, no words, no extra lines.
Keep <reasoning> short (a few hundred words). Show only the few scalars that determine the final number. Do not paste the input matrices, full coefficient vectors, per-iteration paths, or per-fold candidate tables.

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

01_s_type_one_electron_integrals

Goal
----
Step 01 - One-electron and z-coordinate integrals over contracted s-type Gaussians.

```python
import numpy as np
from scipy.special import erf


def s_type_one_electron_integrals(coords: np.ndarray, charges: np.ndarray, exponents: list, coefficients: list) -> np.ndarray:
    '''Overlap, kinetic, nuclear attraction and z-coordinate matrices over contracted s functions.

    Parameters
    ----------
    coords : np.ndarray
        Nuclear positions in bohr, shape (n_atoms, 3). Every atom carries the
        same list of shells.
    charges : np.ndarray
        Nuclear charges, shape (n_atoms,), each non-negative.
    exponents : list
        One 1D array of primitive exponents (bohr^-2) per contracted shell.
    coefficients : list
        One 1D array of contraction coefficients per shell, referring to
        normalised primitives, in the same order as exponents.

    Returns
    -------
    integrals : np.ndarray
        Array of shape (4, n_bf, n_bf) holding the overlap matrix S, the
        kinetic energy matrix T, the nuclear attraction matrix V and the matrix
        Z of the electron's z coordinate (bohr), in that order, with
        n_bf = n_atoms * len(exponents).

    Raises
    ------
    ValueError
        If coords is not a finite (n_atoms, 3) array, if charges does not have
        shape (n_atoms,) or holds a negative or non-finite value, if exponents
        and coefficients are empty or differ in length, if any shell has
        mismatched or empty arrays, or if any exponent is not positive.
    '''
    return integrals
```

### Step 2

02_s_type_electron_repulsion

Goal
----
Step 02 - Electron repulsion integrals over contracted s-type Gaussians.

```python
import numpy as np


def s_type_electron_repulsion(coords: np.ndarray, exponents: list, coefficients: list) -> np.ndarray:
    '''Four-index electron repulsion integrals (ij|kl) over contracted s functions.

    Parameters
    ----------
    coords : np.ndarray
        Nuclear positions in bohr, shape (n_atoms, 3). Every atom carries the
        same list of shells.
    exponents : list
        One 1D array of primitive exponents (bohr^-2) per contracted shell.
    coefficients : list
        One 1D array of contraction coefficients per shell, referring to
        normalised primitives, in the same order as exponents.

    Returns
    -------
    eri : np.ndarray
        Array of shape (n_bf, n_bf, n_bf, n_bf) in hartree, eri[i, j, k, l] =
        (ij|kl) in chemist's notation, n_bf = n_atoms * len(exponents).

    Raises
    ------
    ValueError
        If coords is not a finite (n_atoms, 3) array, if exponents and
        coefficients are empty or differ in length, if any shell has
        mismatched or empty arrays, or if any exponent is not positive.
    '''
    return eri
```

### Step 3

03_rhf_canonical_orbitals

Goal
----
Step 03 - Canonical closed-shell RHF orbitals.

```python
import numpy as np


def rhf_canonical_orbitals(coords: np.ndarray, charges: np.ndarray, exponents: list, coefficients: list, n_occ: int) -> np.ndarray:
    '''Canonical closed-shell RHF orbital energies and coefficients.

    Parameters
    ----------
    coords : np.ndarray
        Nuclear positions in bohr, shape (n_atoms, 3).
    charges : np.ndarray
        Nuclear charges, shape (n_atoms,).
    exponents : list
        One 1D array of primitive exponents per contracted s shell, the same
        shells on every atom.
    coefficients : list
        One 1D array of contraction coefficients per shell (normalised
        primitives), in the same order as exponents.
    n_occ : int
        Number of doubly occupied orbitals, 1 <= n_occ <= n_bf.

    Returns
    -------
    orbitals : np.ndarray
        Array of shape (n_bf + 1, n_bf). Row 0 holds the orbital energies in
        ascending order (hartree); rows 1 to n_bf hold the coefficient matrix C,
        column k being orbital k, normalised so that C^T S C = 1 and signed so
        that the first coefficient of magnitude above 1e-8 in each column is
        positive.

    Raises
    ------
    ValueError
        If the geometry or basis is invalid (the conditions of the integral
        steps), if n_occ is not an integer in [1, n_bf], if the overlap matrix
        is not positive definite, or if the self-consistent field does not
        converge within 1000 iterations.
    '''
    return orbitals
```

### Step 4

04_ccsd_lambda_ground_state

Goal
----
Step 04 - Ground-state CCSD cluster and Lambda amplitudes in spin orbitals.

```python
import numpy as np


def ccsd_lambda_ground_state(fock: np.ndarray, eri_as: np.ndarray, n_electrons: int) -> np.ndarray:
    '''Converged ground-state CCSD cluster and de-excitation amplitudes, packed in one vector.

    Parameters
    ----------
    fock : np.ndarray
        Spin-orbital Fock matrix of the reference determinant, shape (n, n),
        real and symmetric, not necessarily diagonal, hartree.
    eri_as : np.ndarray
        Antisymmetrised two-electron integrals g[p,q,r,s] = <pq||rs>, shape
        (n, n, n, n), hartree.
    n_electrons : int
        Number of electrons; the reference occupies spin orbitals
        0 .. n_electrons - 1.

    Returns
    -------
    amplitudes : np.ndarray
        Real vector [t1, t2, l1, l2] (each flattened in C order, doubles as full
        antisymmetric (o, o, v, v) arrays) of length 2 (o v + o^2 v^2), where
        o = n_electrons and v = n - o.

    Raises
    ------
    ValueError
        If fock is not a finite symmetric square matrix of size at least 2
        (tolerance 1e-10), if eri_as is not a finite (n, n, n, n) array that is
        antisymmetric within each index pair and symmetric under exchange of
        the pairs (tolerance 1e-10), if n_electrons is not an integer in
        [1, n - 1], if a Jacobi denominator D_i^a, or D_ij^ab with i != j and
        a != b, is smaller than 1e-8 in magnitude, or if either set of
        equations fails to converge within 500 sweeps.
    '''
    return amplitudes
```

### Step 5

05_tdccsd_time_derivative

Goal
----
Step 05 - Time derivative of the bivariational TD-CCSD amplitudes in a field.

```python
import numpy as np


def tdccsd_time_derivative(fock: np.ndarray, dipole: np.ndarray, eri_as: np.ndarray, n_electrons: int, amplitudes: np.ndarray, field: float) -> np.ndarray:
    '''Time derivative of the packed TD-CCSD amplitudes [t1, t2, l1, l2] in a z-polarised field.

    Parameters
    ----------
    fock : np.ndarray
        Field-free spin-orbital Fock matrix of the reference, shape (n, n),
        real and symmetric, hartree.
    dipole : np.ndarray
        Spin-orbital matrix of the electronic z coordinate, shape (n, n), real
        and symmetric, bohr.
    eri_as : np.ndarray
        Antisymmetrised two-electron integrals <pq||rs>, shape (n, n, n, n).
    n_electrons : int
        Number of electrons; the reference occupies spin orbitals
        0 .. n_electrons - 1.
    amplitudes : np.ndarray
        Packed amplitude vector [t1, t2, l1, l2], real or complex, length
        L = 2 (o v + o^2 v^2) with o = n_electrons and v = n - o.
    field : float
        Instantaneous electric field strength E(t) in atomic units.

    Returns
    -------
    derivative : np.ndarray
        Real vector of length 2 L: [Re(dy/dt), Im(dy/dt)] for the packed
        amplitude vector y, in atomic units of inverse time.

    Raises
    ------
    ValueError
        If fock, eri_as or n_electrons violate the conditions of the
        ground-state step, if dipole is not a finite symmetric (n, n) matrix
        (tolerance 1e-10), if amplitudes is not a finite 1D vector of length L,
        or if field is not a finite real number.
    '''
    return derivative
```

### Step 6

06_tdccsd_propagate

Goal
----
Step 06 - Runge-Kutta propagation of TD-CCSD through a trigonometric-envelope pulse.

```python
import numpy as np


def tdccsd_propagate(fock: np.ndarray, dipole: np.ndarray, eri_as: np.ndarray, n_electrons: int, amplitudes: np.ndarray, e0: float, omega: float, t_center: float, t_foot: float, n_env: int, t_start: float, dt: float, n_steps: int) -> np.ndarray:
    '''Propagate the packed TD-CCSD amplitudes with fourth-order Runge-Kutta through a trigonometric-envelope pulse.

    Parameters
    ----------
    fock : np.ndarray
        Field-free spin-orbital Fock matrix of the reference, shape (n, n).
    dipole : np.ndarray
        Spin-orbital matrix of the electronic z coordinate, shape (n, n).
    eri_as : np.ndarray
        Antisymmetrised two-electron integrals <pq||rs>, shape (n, n, n, n).
    n_electrons : int
        Number of electrons (occupied spin orbitals 0 .. n_electrons - 1).
    amplitudes : np.ndarray
        Initial packed vector [t1, t2, l1, l2], real or complex, length L.
    e0 : float
        Peak field amplitude in atomic units (any finite real number).
    omega : float
        Carrier angular frequency in hartree (finite, non-negative).
    t_center : float
        Centre of the envelope in atomic units of time.
    t_foot : float
        Foot-to-foot duration of the envelope, strictly positive.
    n_env : int
        Positive integer exponent of the cosine envelope.
    t_start : float
        Time of the initial vector.
    dt : float
        Strictly positive Runge-Kutta step.
    n_steps : int
        Non-negative number of steps.

    Returns
    -------
    final : np.ndarray
        Real vector of length 2 L: real parts of the packed amplitudes at
        t_start + n_steps dt followed by their imaginary parts.

    Raises
    ------
    ValueError
        If the Hamiltonian, dipole or amplitude inputs violate the conditions
        of the previous step, if e0, omega, t_center or t_start is not finite
        or omega is negative, if t_foot or dt is not a finite positive number,
        if n_env is not a positive integer, or if n_steps is not a non-negative
        integer.
    '''
    return final
```

### Step 7

07_configuration_weights

Goal
----
Step 07 - Configuration weights of a coupled-cluster bra-ket pair.

```python
import numpy as np


def configuration_weights(amplitudes: np.ndarray, n_electrons: int, n_spin_orbitals: int) -> np.ndarray:
    '''Reference, single and double configuration weights of a CCSD bra-ket pair.

    Parameters
    ----------
    amplitudes : np.ndarray
        Packed vector [t1, t2, l1, l2], real or complex, of length
        2 (o v + o^2 v^2) with o = n_electrons and v = n_spin_orbitals - o;
        the doubles arrays must be antisymmetric in (i, j) and in (a, b).
    n_electrons : int
        Number of occupied spin orbitals o, 1 <= o < n_spin_orbitals.
    n_spin_orbitals : int
        Total number of spin orbitals, at least 2.

    Returns
    -------
    weights : np.ndarray
        Real vector of length 1 + o v + (o (o - 1) / 2) (v (v - 1) / 2):
        [W_0, W_i^a in C order, W_ij^ab for i < j, a < b].

    Raises
    ------
    ValueError
        If n_spin_orbitals is not an integer of at least 2, if n_electrons is
        not an integer in [1, n_spin_orbitals - 1], if amplitudes is not a
        finite 1D vector of the stated length, or if either doubles array is
        not antisymmetric in (i, j) and in (a, b) within 1e-10.
    '''
    return weights
```

### Step 8

08_dark_weight_change

Goal
----
Step 08 - Pulse-induced change in the weight of a dark determinant of a hydrogen chain (orchestrator).

```python
import numpy as np


def dark_weight_change(n_atoms: int, spacing: float, exponents: list, coefficients: list, e0: float, omega: float, t_foot: float, n_env: int, dt: float, occ: int, vir: int) -> float:
    '''Change of the spin-summed configuration weight of the excitation occ -> vir across a laser pulse.

    Parameters
    ----------
    n_atoms : int
        Even number of hydrogen atoms in the chain, at least 2.
    spacing : float
        Nearest-neighbour distance in bohr, strictly positive.
    exponents : list
        One 1D array of primitive exponents per contracted s shell (the same
        shells on every atom).
    coefficients : list
        One 1D array of contraction coefficients per shell (normalised
        primitives), in the same order as exponents.
    e0 : float
        Peak field amplitude in atomic units.
    omega : float
        Carrier angular frequency in hartree, non-negative.
    t_foot : float
        Foot-to-foot duration of the envelope in atomic units of time; the pulse
        is centred at time 0.
    n_env : int
        Positive integer exponent of the cosine envelope.
    dt : float
        Runge-Kutta step; t_foot / dt must be an integer within a relative
        tolerance of 1e-9.
    occ : int
        Occupied canonical spatial orbital, 0 <= occ < n_atoms / 2.
    vir : int
        Virtual canonical spatial orbital, n_atoms / 2 <= vir < n_bf.

    Returns
    -------
    delta_weight : float
        Weight of the two spin-orbital determinants occ -> vir at t = t_foot / 2
        minus their weight in the coupled-cluster ground state (dimensionless).

    Raises
    ------
    ValueError
        If n_atoms is not an even integer of at least 2, if spacing is not a
        finite positive number, if the basis or pulse parameters violate the
        conditions of the earlier steps, if t_foot / dt is not an integer, or if
        occ or vir is out of its range.
    '''
    return delta_weight
```
