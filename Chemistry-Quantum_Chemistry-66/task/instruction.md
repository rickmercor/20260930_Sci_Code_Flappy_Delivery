# Chemistry-Quantum_Chemistry-66

## Background

# Scientific background

## Green's functions, self-energy, and quasiparticle energies

The one-body Green's function of a many-electron system describes how easy
it is to add one electron to a given orbital, or to remove one electron from
it. Its poles are the true, correlated electron addition and removal
energies of the system. These energies differ from the plain Hartree-Fock
orbital energies because of electron correlation. The Dyson equation
connects the true Green's function to a simpler reference Green's function
through the self-energy. The self-energy is an operator that collects every
correlation effect that is not already present in the reference. To find a
quasiparticle energy, you find a pole of the true Green's function. In
practice, this means that you solve an equation that contains the
self-energy, evaluated at an energy close to the reference orbital energy.

## The GW approximation

The GW approximation builds the self-energy from the product of the Green's
function (G) and the dynamically screened Coulomb interaction (W). Screening
means that the bare electron-electron repulsion is replaced by an effective
interaction that depends on frequency. This effective interaction accounts
for how the surrounding electrons rearrange in response to a perturbation.
GW captures a large part of the correlation that Hartree-Fock theory misses,
at a moderate computational cost. It is the standard starting point for the
calculation of ionization energies and electron affinities in molecules and
solids. Because GW has a good and well-tested track record, later
refinements are usually judged by two questions. The first is how much they
improve on GW. The second is whether they preserve the desirable
mathematical properties of the self-energy. One such property is that the
self-energy can be written as a sum over discrete states with positive
spectral weight.

## The random-phase approximation and screening

The screened interaction used in GW is built from a response function. This
response function is most commonly approximated at the level of the
random-phase approximation (RPA). The direct RPA keeps only the classical
(Hartree-type) part of the electron-electron interaction in this response
and leaves out the exchange part. This is the standard choice for GW-type
screening. When you solve the RPA equations for a system, you obtain a set
of collective excitation energies. Each excitation energy has an associated
pair of amplitudes. These amplitudes describe how the excitation mixes
occupied and virtual orbitals. The excitation energies and amplitudes are
then combined with the bare Coulomb integrals to produce an effective
integral. This effective integral couples any two orbitals to any given
collective excitation. It is the basic building block from which GW and its
extensions are constructed.

## Going beyond GW: vertex corrections

GW itself includes only the lowest-order coupling between an electron and
the density fluctuations of the surrounding electrons. This is called a
bare vertex. Real systems benefit from higher-order corrections that
include a dressed vertex. This is especially true for systems with
multireference character or strong correlation, and for cases where
satellite (shake-up) structures matter. One systematic family of such
corrections keeps terms up to second order in the screened interaction. The
names of these corrections are sometimes built from how many powers of the
screened interaction W and the bare Coulomb interaction appear in the
corresponding self-energy diagrams. These corrections improve accuracy. But
they can break desirable mathematical properties of the self-energy that
plain GW has. One example is the property that the self-energy can be
written as a discrete sum of positive spectral weights. This is equivalent
to the self-energy corresponding to a well-defined Hermitian effective
Hamiltonian.

## The algebraic-diagrammatic construction

The algebraic-diagrammatic construction (ADC) is a general and systematic
way to turn an approximate, frequency-dependent self-energy into an
ordinary Hermitian eigenvalue problem that does not depend on frequency.
This is done order by order. The idea is to identify, order by order in the
interaction, which additional intermediate configurations must be included
as extra rows and columns of an effective matrix. An intermediate
configuration is an excited determinant relative to a chosen orbital. The
construction also identifies the diagonal energies of these configurations,
their couplings to the reference orbital, and their couplings to one
another. The result is that the eigenvalues of this matrix reproduce the
poles of the target self-energy exactly through that order. ADC has a long
history of application directly to the bare Coulomb interaction. This gives
a hierarchy of methods that has been used for ionization energies and
electron affinities. ADC can equally be applied to a self-energy that is
itself built from the screened interaction, such as GW and its
vertex-corrected extensions. The application of ADC to an already-screened
self-energy, instead of to the bare interaction, is a comparatively recent
direction. It produces a self-energy that goes beyond plain GW in accuracy
and also keeps the convenient matrix structure with positive spectral
weights.

## Diagonal approximation and quasiparticle weight

The full self-energy matrix over every pair of orbitals is expensive to
build. A common and well-established simplification is the diagonal
approximation. It is used, for example, whenever you want a one-shot
quasiparticle energy for a single orbital of interest. In this
approximation, the self-energy connects an orbital only to itself. The
reference part of the effective matrix then reduces to a single row and
column. The space of intermediate configurations, built from all occupied
and virtual orbitals, is unchanged. This turns the problem into the search
for one eigenvalue of a smaller matrix. The correct eigenvalue is the one
whose eigenvector has the largest overlap with the bare reference orbital.
This overlap is called the quasiparticle weight. A value close to 1
indicates a well-defined, single-particle-like excitation, and not a
strongly mixed satellite state.

## Toy active spaces for method development

These theories are all built from the same small set of ingredients. These
are the orbital energies, the screened and bare two-electron integrals, and
the collective excitation energies and amplitudes. Because of this, the
theories can be tested and validated on a small, self-contained active space
of a few orbitals. The active space can use real molecular integrals from an
inexpensive quantum chemistry calculation, without a model of the full
electronic structure of a large molecule. Such a toy active space is only
meaningful if it keeps enough distinct, non-degenerate collective
excitations for every term of the theory's working equations to matter. An
active space that is too small can accidentally reduce every sum to a single
surviving term. This hides index and sign mistakes instead of revealing them.

## Problem

Take a small active space of four molecular orbitals from a restricted
Hartree-Fock calculation on a water molecule. The basis is STO-3G. The
oxygen atom is at (0, 0, 0.1173) angstrom and the two hydrogen atoms are at
(0, ±0.7572, -0.4692) angstrom. Orbitals 0 and 1 are occupied. Orbitals 2
and 3 are virtual. Treat this active space as the entire system. Do not add
any other orbital, and do not treat any other electron as present. Each
spatial orbital p gives two spin-orbitals with the same orbital energy. The
alpha spin-orbital has index 2p and the beta spin-orbital has index 2p+1. A
two-electron Coulomb integral between spin-orbitals is equal to the matching
spatial-orbital integral when the two bra spins are the same and the two ket
spins are the same. In all other cases the integral is zero.

Orbital energies (hartree):
eps(0) = -0.45302164
eps(1) = -0.39123623
eps(2) = 0.60517186
eps(3) = 0.74159807

The two-electron Coulomb integrals over the spatial orbitals use chemist
notation (pq|rs). The table below gives every nonzero combination of p, q,
r, s in {0, 1, 2, 3}. All values are in hartree. A combination that is not
listed has the value 0.0.

```
(0,0,0,0)=0.78263693  (0,0,0,2)=-0.12138634  (0,0,1,1)=0.72887765
(0,0,2,0)=-0.12138634 (0,0,2,2)=0.54907338   (0,0,3,3)=0.60821194
(0,1,0,1)=0.05591042  (0,1,1,0)=0.05591042   (0,1,1,2)=0.00174051
(0,1,2,1)=0.00174051  (0,2,0,0)=-0.12138634  (0,2,0,2)=0.06874080
(0,2,1,1)=-0.11631017 (0,2,2,0)=0.06874080   (0,2,2,2)=-0.04459738
(0,2,3,3)=-0.04154608 (0,3,0,3)=0.06897688   (0,3,2,3)=0.05791860
(0,3,3,0)=0.06897688  (0,3,3,2)=0.05791860   (1,0,0,1)=0.05591042
(1,0,1,0)=0.05591042  (1,0,1,2)=0.00174051   (1,0,2,1)=0.00174051
(1,1,0,0)=0.72887765  (1,1,0,2)=-0.11631017  (1,1,1,1)=0.88015909
(1,1,2,0)=-0.11631017 (1,1,2,2)=0.58894806   (1,1,3,3)=0.62498500
(1,2,0,1)=0.00174051  (1,2,1,0)=0.00174051   (1,2,1,2)=0.03858992
(1,2,2,1)=0.03858992  (1,3,1,3)=0.02435073   (1,3,3,1)=0.02435073
(2,0,0,0)=-0.12138634 (2,0,0,2)=0.06874080   (2,0,1,1)=-0.11631017
(2,0,2,0)=0.06874080  (2,0,2,2)=-0.04459738  (2,0,3,3)=-0.04154608
(2,1,0,1)=0.00174051  (2,1,1,0)=0.00174051   (2,1,1,2)=0.03858992
(2,1,2,1)=0.03858992  (2,2,0,0)=0.54907338   (2,2,0,2)=-0.04459738
(2,2,1,1)=0.58894806  (2,2,2,0)=-0.04459738  (2,2,2,2)=0.59713088
(2,2,3,3)=0.56630978  (2,3,0,3)=0.05791860   (2,3,2,3)=0.11530640
(2,3,3,0)=0.05791860  (2,3,3,2)=0.11530640   (3,0,0,3)=0.06897688
(3,0,2,3)=0.05791860  (3,0,3,0)=0.06897688   (3,0,3,2)=0.05791860
(3,1,1,3)=0.02435073  (3,1,3,1)=0.02435073   (3,2,0,3)=0.05791860
(3,2,2,3)=0.11530640  (3,2,3,0)=0.05791860   (3,2,3,2)=0.11530640
(3,3,0,0)=0.60821194  (3,3,0,2)=-0.04154608  (3,3,1,1)=0.62498500
(3,3,2,0)=-0.04154608 (3,3,2,2)=0.56630978   (3,3,3,3)=0.61951549
```

Solve the closed-shell direct random-phase approximation for this active
space in the spin-orbital basis. Use Hartree-kernel screening only, with no
exchange kernel. This gives the collective excitation energies and their
forward and backward amplitudes. Use them to build the standard GW effective
integrals. Then build the self-energy hierarchy of the algebraic-diagrammatic
construction (ADC) applied to the GW self-energy. The hierarchy has four
increasing levels:

1. Plain one-shot GW.
2. The next level, which is exactly equivalent to a previously published
   positive-semidefinite GW+2SOSEX self-energy.
3. The level that adds the ADC(3) diagonal correction to the G3W2
   self-energy.
4. The full ADC treatment of the complete G3W2 self-energy. This level
   introduces a new class of configuration, built from two simultaneous
   collective excitations at once.

Use the paper's own diagonal approximation at every level. In this
approximation, the reference orbital contributes only through its own row
and column.

Compute the ionization potential of spin-orbital 3 at each of the four
levels. Report the ionization potential of spin-orbital 3 at the full level
of this hierarchy. Spin-orbital 3 is the beta spin of orbital 1, the highest
occupied orbital. The ionization potential is the positive energy required
to remove that electron. It is equal to the negative of the quasiparticle
eigenvalue whose eigenvector has the largest overlap with the bare reference
orbital. Give the answer in hartree, rounded to four decimal places. In
your reasoning, also report the quasiparticle weight of that root at the
full level, that is the squared overlap of its eigenvector with the bare
reference orbital, rounded to two decimal places.

Output Format Requirements:
Emit <final_answer> immediately, then <reasoning>. Do not write a long
derivation before the tags.
You must emit exactly one finite decimal inside
<final_answer>...</final_answer>, even if the value is approximate or you are
unsure.

Rules:

- The tags are required. Do not omit them or leave them empty.
- The tagged value must be a finite decimal (examples: 0.4847, 0.22, 1.05).
  Not NaN, not Inf, not a fraction string, not a vector, not prose.
- Put only that one number between the tags. No units, no words, no extra
  lines.
- Do not paste the input matrices, full coefficient vectors, per-iteration
  paths, or per-fold candidate tables.

## Output format

```
## Output format

Scientific reasoning wrapped in <reasoning>...</reasoning> tags. Include enough intermediate calculations to justify the deterministic computation, without turning the response into a general pipeline summary. A single final numeric answer wrapped in <final_answer>...</final_answer> tags.
```

## Your task

Implement **all 9 functions** below into a single file at `/app/solution.py`. Requirements:

- Use the EXACT function name and signature shown for each step.
- Define every function at module scope (importable).
- Do NOT include tests, example usage, prints, or a `__main__` block.
- Available libraries: numpy, scipy, sympy, pandas, scikit-learn, networkx.

Start your file with these imports:

```python
import numpy as np
```

### Step 1

01_Symmetric eigensystem

Goal
----
Compute every eigenvalue and eigenvector of a real

symmetric matrix. This is a standard routine from numerical linear

algebra. Later steps use it as a building block for the larger,

paper-specific calculations. It is not itself part of the physics being

modeled.

```python
def symmetric_eigensystem(matrix: list[list[float]]) -> tuple[list[float], list[list[float]]]:
    """
    Compute the eigenvalues and eigenvectors of a real symmetric matrix.

    Args:
        matrix: an n-by-n real symmetric matrix, as a list of n rows,
            each a list of n floats. matrix[i][j] == matrix[j][i] for
            every i, j (up to floating-point noise in the input).

    Conventions:
        - Eigenvalues are returned sorted in ascending order.
        - Eigenvectors are returned as a list of n vectors (each a list
          of n floats), in the same order as the eigenvalues. This
          means that eigenvectors[k] is a unit-norm eigenvector for
          eigenvalues[k].
        - Each eigenvector is normalized to unit Euclidean norm. Its
          overall sign is fixed so that its single largest-magnitude
          component is positive. If two components are tied for largest
          magnitude to within 1e-14, the one with the smaller index
          decides the sign.

    """
    return ([], [])
```

### Step 2

02_Direct RPA excitation energies

Goal
----
Solve the closed-shell direct RPA response problem in

the spin-orbital basis, with the Hartree kernel only. Include every

same-spin occupied-to-virtual excitation. Return the excitation

energies together with their forward (X) and backward (Y) amplitudes.

```python
def direct_rpa(
    eps: list[float],
    chem_spatial: dict[tuple[int, int, int, int], float],
    occ: list[int],
    vir: list[int],
) -> tuple[list[float], list[list[float]], list[list[float]]]:
    """
    Solve the direct RPA problem for a set of active spin-orbitals.

    Args:
        eps: orbital energy of each active spin-orbital, indexed 0..n-1.
            Spin-orbital 2*p is the alpha member of spatial orbital p,
            and spin-orbital 2*p+1 is the beta member. Both have the
            same energy.
        chem_spatial: two-electron Coulomb integrals over the active
            spatial orbitals, in chemist notation (pq|rs), keyed by the
            4-tuple (p, q, r, s) of spatial-orbital indices. A key that
            is absent has value 0.0. The corresponding spin-orbital
            integral (p', q', r', s') is chem_spatial[(p'//2, q'//2,
            r'//2, s'//2)] if p', q' have the same spin and r', s' have
            the same spin. In all other cases it is 0.0.
        occ: the occupied active spin-orbital indices.
        vir: the virtual active spin-orbital indices.

    Conventions:
        - The occupied-virtual pair list pairs every occupied
          spin-orbital with every virtual spin-orbital of the same spin
          (the same spin-orbital index parity). The order is occ[0]
          with every same-spin member of vir, then occ[1] with every
          same-spin member of vir, and so on. The order of occ and vir
          is preserved as given.
        - Excitation energies are returned sorted in ascending order,
          together with their amplitudes in the same order.
        - Each amplitude pair (X, Y) for a given excitation is
          normalized so that sum(X[k]**2) - sum(Y[k]**2) == 1. The
          overall sign is fixed by the eigenvector sign convention of
          the symmetric eigensystem step. That convention is applied to
          the Hermitian matrix (A-B)^(1/2) (A+B) (A-B)^(1/2), whose
          eigenvalues are the squared excitation energies. It is not
          applied to (A-B) itself, whose own eigenvectors do not
          determine this sign.

    """
    return ([], [], [])
```

### Step 3

03_GW effective integrals

Goal
----
Combine the direct-RPA excitation amplitudes with the

bare Coulomb integrals to form the GW effective integral between every

pair of active spin-orbitals and every excitation. This quantity is the

basic coupling unit from which every later step is built.

```python
def gw_effective_integrals(
    eps: list[float],
    chem_spatial: dict[tuple[int, int, int, int], float],
    occ: list[int],
    vir: list[int],
    omega: list[float],
    X: list[list[float]],
    Y: list[list[float]],
) -> list[list[list[float]]]:
    """
    Compute the GW effective integral for every active spin-orbital
    pair and every excitation.

    Args:
        eps: orbital energy of each active spin-orbital.
        chem_spatial: two-electron Coulomb integrals over the active
            spatial orbitals, as in the direct-RPA step. A key that is
            absent has value 0.0. The spin-orbital physicist-notation
            integral <pq|rs> equals chem_spatial[(p//2, r//2, q//2,
            s//2)] if p, r have the same spin and q, s have the same
            spin. In all other cases it is 0.0.
        occ, vir: the occupied and virtual active spin-orbital indices.
        omega: excitation energies, in the same order as X and Y.
        X, Y: forward and backward amplitudes for each excitation nu.
            Each X[nu] (and each Y[nu]) is a list over the
            occupied-virtual pairs. The pairs are built by pairing every
            occupied spin-orbital with every same-spin virtual
            spin-orbital, occ[0] first, in the order occ and vir are
            given.

    """
    return []
```

### Step 4

04_ADC-GW effective Hamiltonian (tier 1)

Goal
----
Assemble the smallest effective Hamiltonian of the

algebraic-diagrammatic construction (ADC) hierarchy used in this task.

It has one reference row and column for a single probed orbital, plus

every two-hole-one-particle and two-particle-one-hole configuration

built from the active space. This tier reproduces plain one-shot GW

exactly. The paper states that it "does not introduce any additional

terms" beyond conventional GW.

```python
def adc_gw_tier1_matrix(
    eps: list[float],
    occ: list[int],
    vir: list[int],
    omega: list[float],
    M: list[list[list[float]]],
    probe: int,
) -> list[list[float]]:
    """
    Assemble the tier-1 (ADC-GW) effective Hamiltonian for one probed
    orbital, in the diagonal approximation. In this approximation the
    reference block is reduced to the single row and column for the
    probed orbital. The paper uses this approximation for every result
    in its own results table.

    Args:
        eps: orbital energy of each active spin-orbital.
        occ, vir: the occupied and virtual active spin-orbital indices.
        omega: excitation energies.
        M: the GW effective integrals from the previous step, M[p][q][nu].
        probe: the active spin-orbital index whose quasiparticle energy
            is sought.

    Conventions:
        - Row and column 0 are the reference (the probed orbital
          itself).
        - The next len(occ)*len(omega) rows and columns are the
          two-hole-one-particle configurations. They are ordered by
          occupied orbital first (in the order occ is given) and by
          excitation index second (in the order omega is given).
        - The remaining len(vir)*len(omega) rows and columns are the
          two-particle-one-hole configurations. They are ordered by
          virtual orbital first (in the order vir is given) and by
          excitation index second.
        - The assembled matrix is symmetric (real and Hermitian).

    """
    return []
```

### Step 5

05_ADC-2SOSEX effective Hamiltonian (tier 2)

Goal
----
Add a second coupling term to the tier-1 (ADC-GW)

matrix. The paper states that, with this term included, the resulting

scheme is exactly equivalent to a self-energy already published by a

different group (GW+2SOSEX made positive semi-definite). This term is

therefore not new physics. It is still a paper-specific formula that

must be implemented correctly: two sums with similar-looking factors

but different energy denominators.

```python
def adc_2sosex_matrix(
    eps: list[float],
    chem_spatial: dict[tuple[int, int, int, int], float],
    occ: list[int],
    vir: list[int],
    omega: list[float],
    M: list[list[list[float]]],
    probe: int,
) -> list[list[float]]:
    """
    Assemble the tier-2 (ADC-2SOSEX) effective Hamiltonian for one
    probed orbital, by adding a second coupling contribution to the
    tier-1 matrix from the previous step.

    Args:
        eps, occ, vir, omega, M, probe: as in the tier-1 step.
        chem_spatial: bare two-electron Coulomb integrals over the
            active spatial orbitals, as in earlier steps.

    Conventions:
        - The row and column layout is the same as in the tier-1
          matrix.
        - The diagonal entries (the unperturbed configuration energies)
          are unchanged from tier 1.
        - Some terms would require a division by an energy denominator
          that is extremely close to zero (magnitude below 1e-9) while
          every GW effective integral the term multiplies is itself
          negligible (magnitude below 1e-9). Such a term contributes
          exactly 0.0 to the sum. When both parts of this condition
          hold, the term is an exact cancellation, and it must not be
          evaluated as a division.

    """
    return []
```

### Step 6

06_ADC(3)-G3W2 effective Hamiltonian (tier 3)

Goal
----
Add the first new correction of this paper to the

tier-2 matrix. This is a symmetrized diagonal-block correction. It

couples different two-hole-one-particle configurations to one another,

and separately it couples different two-particle-one-hole

configurations to one another. It does not only couple configurations

to the reference orbital.

```python
def adc3_g3w2_matrix(
    eps: list[float],
    chem_spatial: dict[tuple[int, int, int, int], float],
    occ: list[int],
    vir: list[int],
    omega: list[float],
    M: list[list[list[float]]],
    probe: int,
) -> list[list[float]]:
    """
    Assemble the tier-3 (ADC(3)-G3W2) effective Hamiltonian for one
    probed orbital, by adding a diagonal-block correction to the tier-2
    matrix from the previous step.

    Args:
        eps, chem_spatial, occ, vir, omega, M, probe: as in the tier-1
            and tier-2 steps. chem_spatial must be forwarded to the
            tier-2 step so that it can build its own correction. The
            new correction of this step is built only from the GW
            effective integrals and the orbital and excitation
            energies.

    Conventions:
        - The row and column layout is the same as in the tier-1 and
          tier-2 matrices.
        - The correction of this step can couple two different
          two-hole-one-particle configurations to each other, not only
          a configuration to itself. Separately, it can couple two
          different two-particle-one-hole configurations to each other.
          It never couples a two-hole-one-particle configuration to a
          two-particle-one-hole configuration.
        - A term contributes exactly 0.0 if at least one GW effective
          integral factor in its numerator has magnitude below 1e-9
          and at least one of its energy denominators has magnitude
          below 1e-9.

    """
    return []
```

### Step 7

07_ADC-G3W2 third coupling term (tier 4, part 1)

Goal
----
Add the third and most complex coupling correction

between the reference orbital and every two-hole-one-particle or

two-particle-one-hole configuration. This correction, together with the

new block of configurations added in the next step, completes the full

theory.

```python
def adc_g3w2_third_coupling(
    eps: list[float],
    chem_spatial: dict[tuple[int, int, int, int], float],
    occ: list[int],
    vir: list[int],
    omega: list[float],
    M: list[list[list[float]]],
    probe: int,
) -> list[list[float]]:
    """
    Assemble the effective Hamiltonian for one probed orbital after
    adding the tier-4 third coupling correction to the tier-3 matrix
    from the previous step. This step does not yet add the new block of
    configurations from the following step. It only updates the
    coupling between the reference and the configurations that are
    already present.

    Args:
        eps, chem_spatial, occ, vir, omega, M, probe: as in the earlier
            tiers. chem_spatial must be forwarded down to the tier-2
            step.

    Conventions:
        - The row and column layout and the matrix size are the same as
          in the earlier tiers.
        - A term contributes exactly 0.0 if at least one GW effective
          integral factor in its numerator has magnitude below 1e-9
          and at least one of its energy denominators has magnitude
          below 1e-9.

    """
    return []
```

### Step 8

08_Three-hole-two-particle block (tier 4, part 2)

Goal
----
Extend the effective Hamiltonian with a new class of

configuration. Each new configuration is one hole (or particle) orbital

together with two simultaneous excitations at once. The new

configurations couple back to the ordinary two-hole-one-particle (or

two-particle-one-hole) configurations. This block completes the full

ADC-G3W2 theory.

```python
def three_hole_two_particle_augmented_matrix(
    eps: list[float],
    chem_spatial: dict[tuple[int, int, int, int], float],
    occ: list[int],
    vir: list[int],
    omega: list[float],
    M: list[list[list[float]]],
    probe: int,
) -> list[list[float]]:
    """
    Build the full tier-4 (ADC-G3W2) effective Hamiltonian for one
    probed orbital. Start from the matrix of the previous step, which
    holds the reference plus every two-hole-one-particle and
    two-particle-one-hole configuration, with the tier-4 third coupling
    term already included. Then append the new configurations, each
    built from one hole (or particle) orbital together with two
    excitation indices at once.

    Args:
        eps, chem_spatial, occ, vir, omega, M, probe: as in the earlier
            tiers. chem_spatial must be forwarded down to the tier-2
            step.

    Conventions:
        - The first 1 + len(occ)*len(omega) + len(vir)*len(omega) rows
          and columns are exactly the matrix from the previous step,
          unchanged.
        - The next len(occ)*len(omega)*len(omega) rows and columns are
          the new hole-type configurations. There is one for each
          occupied orbital (in the order occ is given) and each ordered
          pair of excitation indices (nu, mu). nu is the outer loop and
          mu is the inner loop. Both range over the full excitation
          list in the order omega is given. nu and mu may be equal.
        - The remaining len(vir)*len(omega)*len(omega) rows and columns
          are the corresponding particle-type configurations, built in
          the same way from vir.
        - A new hole-type configuration never couples to a
          two-particle-one-hole configuration. A new particle-type
          configuration never couples to a two-hole-one-particle
          configuration. Beyond that, the source defines exactly which
          pairs of configurations are coupled and which are not.
        - A term contributes exactly 0.0 if at least one GW effective
          integral factor in its numerator has magnitude below 1e-9
          and at least one of its energy denominators has magnitude
          below 1e-9.

    """
    return []
```

### Step 9

09_ADC-G3W2 ionization potential

Goal
----
Combine every earlier step into the full pipeline and

report the final answer. The final answer is the ionization potential

of one named orbital at the full ADC-G3W2 level of theory.

```python
def adc_g3w2_ionization_potential(
    eps: list[float],
    chem_spatial: dict[tuple[int, int, int, int], float],
    occ: list[int],
    vir: list[int],
    probe: int,
) -> float:
    """
    Compute the ionization potential of one active spin-orbital at the
    full ADC-G3W2 level.

    Args:
        eps: orbital energy of each active spin-orbital.
        chem_spatial: bare two-electron Coulomb integrals over the
            active spatial orbitals, as in the earlier steps.
        occ, vir: the occupied and virtual active spin-orbital indices.
        probe: the active spin-orbital index whose ionization potential
            is sought.

    Conventions:
        - The ionization potential is reported as a positive number,
          the energy required to remove the electron. It is the
          negative of the quasiparticle eigenvalue whose eigenvector
          has the largest overlap (squared component) with the bare
          reference orbital. It is not simply the eigenvalue
          numerically closest to the starting orbital energy.

    """
    return 0.0
```
