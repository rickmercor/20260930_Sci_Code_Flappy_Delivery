# Chemistry-Quantum_Chemistry-52

## Background

## Scientific Background

Photoemission and inverse photoemission measure what happens to a molecule when one electron is added or taken away, and the theoretical object that describes those processes is the one-particle Green's function, whose poles are the electron addition and removal energies, and the weight carried by each pole says how much of a clean one-electron process that transition really is. The GW approximation has been the workhorse for computing those poles for four decades, because it dresses the propagator with an interaction that is screened by the response of all the other electrons rather than with the bare Coulomb repulsion, and that screening is what brings computed ionisation energies and band gaps into agreement with experiment. Its weakness is structural rather than numerical. GW is built on a single Slater determinant, so wherever several electronic configurations carry comparable weight — a stretched bond, a transition-metal centre, a biradical, a molecule caught midway through a reaction — the reference itself is wrong before any screening is applied, and the satellite structure that carries the most interesting physics is the first thing to be lost. The satellites are exactly the features that a single-determinant starting point cannot produce, because they correspond to an electron removal accompanied by a simultaneous excitation of the remaining electrons, and their spectral weight is drawn away from the main quasiparticle peak rather than added to it.

The broader effort this task belongs to is the construction of many-body propagator methods that keep the diagrammatic structure and the favourable scaling of GW while replacing its single-determinant reference by a genuinely correlated one. The technical difficulty is that almost every ingredient of the standard derivation assumes a mean-field vacuum. Once the zeroth-order Hamiltonian retains the full two-electron interaction inside a chosen active space, the zeroth-order Green's function stops being a set of orbital energies and becomes a Lehmann sum over exact states of two neighbouring particle-number sectors; the polarisation propagator that provides the screening stops being a particle-hole object and acquires several physically distinct kinds of excitation at once; the Dyson equation itself has to be generalised, because the residual interaction can no longer be inserted between two free propagators in the usual way. Each of those generalisations has more than one defensible reading, and the readings differ in what they predict, so the design choices are load-bearing rather than cosmetic.

Small, exactly solvable model Hamiltonians are the standard testbed for this kind of development, because they allow the whole construction to be driven end to end and compared against an exact diagonalisation of the same operator, and because the active space, the electron count and the strength of the interaction can be tuned independently until each design choice actually changes the answer. Working in a general spin-orbital basis rather than in spatial orbitals with a spin block structure is deliberate in that setting, because it removes the symmetry-induced degeneracies that would otherwise make individual poles ambiguous, and it keeps the formalism in the indices the derivation is written in. The quantities such a testbed is meant to expose are the ones a converged production calculation would hide — which screening channels carry weight, how much of the spectral weight of a satellite survives the dressing, and whether a given simplification of the self-energy changes a number in the fourth significant digit or in the first.

## Problem

The GW approximation describes electron addition and removal by dressing a one-particle propagator with a dynamically screened interaction, but it is built on a single determinant and degrades badly when several electronic configurations carry comparable weight. A recent construction lifts that restriction by adopting an interacting zeroth-order Hamiltonian that keeps the full two-electron interaction inside a chosen active space and demotes everything outside it to a weak residual perturbation, then rebuilding the screened interaction and the self-energy diagrammatically on top of that interacting reference. Its inputs are a set of one- and two-electron integrals, a partition of the spin orbitals into occupied inactive, active and unoccupied inactive sets, and an electron count; its output is the set of electron-removal and electron-addition energies of the system together with their spectral weights.

Four stages carry the calculation and each must be taken in the source's own terms, starting from a zeroth-order Hamiltonian partitioned so that the inactive orbitals enter quadratically through orbital energies obtained from the Fock-like operators the source defines, while the active orbitals carry an effective one-electron operator and their own full two-electron interaction; diagonalising the active part in the neutral particle-number sector and in both singly charged sectors supplies the zeroth-order states, their charged excitation energies and the transition density matrices the later stages consume. The residual interaction is then screened at the multi-reference random-phase-approximation level, which collects the several distinct kinds of screening the source enumerates into one paired eigenvalue problem, solved with the usual metric normalisation and contracted back against the residual interaction to give the screened couplings. Those couplings, together with the Lehmann representation of the interacting zeroth-order Green's function, are combined into the one-shot self-energy the source derives, and the generalised Dyson equation then delivers the interacting Green's function whose poles and residues are wanted.

Carry out every stage as the source prescribes, rather than substituting single-reference GW, a bare random-phase approximation, or a plain configuration-interaction spectrum at any point. The system is a twelve-spin-orbital model in atomic units, built from twelve normalised spherical Gaussian functions that carry exponents 0.42 + 0.06p and centres (1.35 cos 2.4p, 1.35 sin 2.4p, 0.62p) in bohr for p = 0 to 11, with the angle in radians. The one-electron operator is the kinetic energy together with the Coulomb attraction to point charges of magnitude 0.55 fixed at all twelve of those centres, and the two-electron integrals are the ordinary Coulomb repulsion integrals over the same twelve functions. Orthonormalise those twelve functions by symmetric Löwdin orthogonalisation, keeping the input order, and carry both sets of integrals into that orthonormal basis, and treat the twelve resulting functions as twelve spin orbitals of one general two-body Hamiltonian. Spin orbitals 4 and 5 are the occupied inactive orbitals, spin orbitals 1, 2, 3, 6, 7 and 8 are the active orbitals, and spin orbitals 0, 9, 10 and 11 are the unoccupied inactive orbitals, and the system holds five electrons, three of them in the active space, with the zeroth-order reference being the lowest state of the partitioned zeroth-order Hamiltonian in that five-electron space. Collect every pole of the resulting interacting Green's function together with its spectral weight, taken as the trace of the residue matrix there, then keep only the poles at negative frequency, order those by decreasing spectral weight, and report the frequency of the third one in electronvolts, with 1 hartree = 27.211386245988 eV, to at least six significant digits.

## Output format

```
## Output format

Scientific reasoning wrapped in <reasoning>...</reasoning> tags. Include enough intermediate calculations to justify the deterministic computation, without turning the response into a general pipeline summary. A single final numeric answer wrapped in <final_answer>...</final_answer> tags.
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

Orthonormal One Electron Element

Goal
----
One-electron integral of the twelve-Gaussian model in its Loewdin-orthonormal basis.

```python
def orthonormal_one_electron_element(n_basis: int, p: int, q: int) -> float:
    """One element of the Loewdin-orthonormal one-electron Hamiltonian of the model.

    Parameters
    ----------
    n_basis : int
        Number of spherical Gaussian primitives in the helix, 1 <= n_basis <= 24.  The
        frozen model system uses n_basis = 12.
    p : int
        Row index of the requested element, 0 <= p < n_basis.
    q : int
        Column index of the requested element, 0 <= q < n_basis.

    Returns
    -------
    float
        h[p, q] = (X^T (T + V) X)[p, q] in hartree, with X = S^{-1/2} the symmetric
        orthonormaliser of the primitive overlap matrix, T the kinetic-energy matrix and
        V the attraction to point charges of magnitude 0.55 at all n_basis centres.
    """
    return 0.0
```

### Step 2

Orthonormal Two Electron Element

Goal
----
Two-electron integral of the twelve-Gaussian model in its Loewdin-orthonormal basis.

```python
def orthonormal_two_electron_element(n_basis: int, p: int, q: int, r: int, s: int) -> float:
    """One physicists'-notation two-electron integral of the orthonormalised model.

    Parameters
    ----------
    n_basis : int
        Number of spherical Gaussian primitives in the helix, 1 <= n_basis <= 16.  The
        frozen model system uses n_basis = 12.
    p, q, r, s : int
        Indices of the requested element, each in the range 0 to n_basis - 1.

    Returns
    -------
    float
        <pq|rs> in hartree, i.e. the chemists' integral (pr|qs) over the Loewdin
        orthonormal basis X = S^{-1/2} of the primitive helix, with the four primitive
        indices contracted against the first index of X.
    """
    return 0.0
```

### Step 3

Active Effective One Electron Element

Goal
----
Active-space effective one-electron operator of the Dyall zeroth-order Hamiltonian.

```python
def active_effective_one_electron_element(h: np.ndarray, g: np.ndarray, core: list[int], act: list[int],
        x: int, y: int) -> float:
    """One element of the active-space effective one-electron operator heff.

    Parameters
    ----------
    h : array_like, shape (n, n)
        One-electron Hamiltonian in an orthonormal spin-orbital basis, in hartree.
    g : array_like, shape (n, n, n, n)
        Two-electron integrals in physicists' notation, g[p, q, r, s] = <pq|rs>.
    core : sequence of int
        Occupied inactive (core) spin-orbital indices.
    act : sequence of int
        Active spin-orbital indices; must be disjoint from core.
    x : int
        Row spin orbital of the requested element; must appear in act.
    y : int
        Column spin orbital of the requested element; must appear in act.

    Returns
    -------
    float
        heff[x, y] = h[x, y] + sum over the core spin orbitals k of <xk||yk>, in hartree.
    """
    return 0.0
```

### Step 4

Dyall Inactive Orbital Energy

Goal
----
Inactive orbital energies of the Dyall zeroth-order Hamiltonian.

```python
def dyall_inactive_orbital_energy(h: np.ndarray, g: np.ndarray, core: list[int], act: list[int],
        vir: list[int], n_act_elec: int, index: int) -> float:
    """One inactive orbital energy of the Dyall zeroth-order Hamiltonian.

    Parameters
    ----------
    h : array_like, shape (n, n)
        One-electron Hamiltonian in an orthonormal spin-orbital basis, in hartree.
    g : array_like, shape (n, n, n, n)
        Two-electron integrals in physicists' notation, g[p, q, r, s] = <pq|rs>.
    core : sequence of int
        Occupied inactive spin-orbital indices.
    act : sequence of int
        Active spin-orbital indices.
    vir : sequence of int
        Unoccupied inactive spin-orbital indices.  core, act and vir together must
        partition range(n) exactly once.
    n_act_elec : int
        Number of electrons in the active space.
    index : int
        Position in the concatenated list of inactive orbital energies, core block first
        in ascending order, then virtual block in ascending order.

    Returns
    -------
    float
        The requested orbital energy in hartree.
    """
    return 0.0
```

### Step 5

Active Reference Energy

Goal
----
Reference energy of the interacting active space of the Dyall partition.

```python
def active_reference_energy(h: np.ndarray, g: np.ndarray, core: list[int], act: list[int],
        n_act_elec: int) -> float:
    """Energy of the interacting active-space reference of the Dyall partition.

    Parameters
    ----------
    h : array_like, shape (n, n)
        One-electron Hamiltonian in an orthonormal spin-orbital basis, in hartree.
    g : array_like, shape (n, n, n, n)
        Two-electron integrals in physicists' notation, g[p, q, r, s] = <pq|rs>.
    core : sequence of int
        Occupied inactive spin-orbital indices.
    act : sequence of int
        Active spin-orbital indices; must be disjoint from core.
    n_act_elec : int
        Number of electrons in the active space.

    Returns
    -------
    float
        The lowest eigenvalue, in hartree, of the active-space configuration interaction
        problem built from heff and the all-active two-electron block.
    """
    return 0.0
```

### Step 6

Charged Channel Energy

Goal
----
Charged-sector channel energies of the zeroth-order Green's function.

```python
def charged_channel_energy(h: np.ndarray, g: np.ndarray, core: list[int], act: list[int],
        vir: list[int], n_elec: int, n_act_elec: int, branch: int,
        index: int) -> float:
    """One retained charged-sector excitation energy of the Dyall zeroth-order Hamiltonian.

    Parameters
    ----------
    h : array_like, shape (n, n)
        One-electron Hamiltonian in an orthonormal spin-orbital basis, in hartree.
    g : array_like, shape (n, n, n, n)
        Two-electron integrals in physicists' notation, g[p, q, r, s] = <pq|rs>.
    core, act, vir : sequence of int
        Occupied inactive, active and unoccupied inactive spin-orbital indices; together
        they must partition range(n) exactly once.
    n_elec : int
        Total number of electrons, equal to n_act_elec plus len(core).
    n_act_elec : int
        Number of electrons in the active space.
    branch : int
        +1 for the electron-addition (N + 1) sector, -1 for the electron-removal (N - 1)
        sector.
    index : int
        Position in the retained channel list of that branch, ordered by increasing
        sector eigenvalue.

    Returns
    -------
    float
        E^{N+1}_m - E^N_0 for branch +1, or E^{N-1}_m - E^N_0 for branch -1, in hartree.
    """
    return 0.0
```

### Step 7

Mrrpa Excitation Energy

Goal
----
Excitation energies of the multi-reference random-phase approximation.

```python
def mrrpa_excitation_energy(h: np.ndarray, g: np.ndarray, core: list[int], act: list[int],
        vir: list[int], n_elec: int, n_act_elec: int, index: int) -> float:
    """One excitation energy of the multi-reference random-phase approximation.

    Parameters
    ----------
    h : array_like, shape (n, n)
        One-electron Hamiltonian in an orthonormal spin-orbital basis, in hartree.
    g : array_like, shape (n, n, n, n)
        Two-electron integrals in physicists' notation, g[p, q, r, s] = <pq|rs>.
    core, act, vir : sequence of int
        Occupied inactive, active and unoccupied inactive spin-orbital indices; together
        they must partition range(n) exactly once.
    n_elec : int
        Total number of electrons, equal to n_act_elec plus len(core).
    n_act_elec : int
        Number of electrons in the active space.
    index : int
        Position in the ascending MR-RPA spectrum, 0 for the lowest mode.

    Returns
    -------
    float
        Omega[index] in hartree, the positive branch of the paired MR-RPA eigenvalue
        problem built on the four screening blocks of the residual interaction.
    """
    return 0.0
```

### Step 8

Static Screened Interaction Element

Goal
----
Static limit of the screened interaction built from the MR-RPA couplings.

```python
def static_screened_interaction_element(h: np.ndarray, g: np.ndarray, core: list[int], act: list[int],
        vir: list[int], n_elec: int, n_act_elec: int, p: int,
        r: int) -> float:
    """One element of the statically screened residual interaction.

    Parameters
    ----------
    h : array_like, shape (n, n)
        One-electron Hamiltonian in an orthonormal spin-orbital basis, in hartree.
    g : array_like, shape (n, n, n, n)
        Two-electron integrals in physicists' notation, g[p, q, r, s] = <pq|rs>.
    core, act, vir : sequence of int
        Occupied inactive, active and unoccupied inactive spin-orbital indices; together
        they must partition range(n) exactly once.
    n_elec : int
        Total number of electrons, equal to n_act_elec plus len(core).
    n_act_elec : int
        Number of electrons in the active space.
    p : int
        First spin-orbital index of the requested element, 0 <= p < n.
    r : int
        Second spin-orbital index of the requested element, 0 <= r < n.

    Returns
    -------
    float
        W[p, r] = 2 sum_I M[p, r, I]^2 / Omega_I in hartree, with M the screened couplings
        of the MR-RPA modes and Omega their excitation energies.
    """
    return 0.0
```

### Step 9

Static Self Energy Norm

Goal
----
Frobenius norm of the static part of the multi-reference self-energy.

```python
def static_self_energy_norm(h: np.ndarray, g: np.ndarray, core: list[int], act: list[int],
        vir: list[int], n_elec: int, n_act_elec: int) -> float:
    """Frobenius norm of the static part of the multi-reference self-energy.

    Parameters
    ----------
    h : array_like, shape (n, n)
        One-electron Hamiltonian in an orthonormal spin-orbital basis, in hartree.
    g : array_like, shape (n, n, n, n)
        Two-electron integrals in physicists' notation, g[p, q, r, s] = <pq|rs>.
    core, act, vir : sequence of int
        Occupied inactive, active and unoccupied inactive spin-orbital indices; together
        they must partition range(n) exactly once.
    n_elec : int
        Total number of electrons, equal to n_act_elec plus len(core).
    n_act_elec : int
        Number of electrons in the active space.

    Returns
    -------
    float
        The Frobenius norm, in hartree, of Sigma_static = u + vbar contracted with the
        generalised one-body density of the zeroth-order reference.
    """
    return 0.0
```

### Step 10

Mrgw Satellite Energy

Goal
----
Multi-reference GW satellite energy: the closed pipeline and the final answer.

```python
def mrgw_satellite_energy(n_basis: int | None = None, core: list[int] | None = None,
        act: list[int] | None = None, vir: list[int] | None = None,
        n_elec: int | None = None, n_act_elec: int | None = None,
        rank: int | None = None) -> float:
    """Frequency of a selected pole of the multi-reference GW Green's function, in eV.

    Parameters
    ----------
    n_basis : int, optional
        Number of spherical Gaussian primitives in the helix, 2 <= n_basis <= 12.  Leave
        every model argument unset to use the frozen twelve-spin-orbital model.
    core, act, vir : sequence of int, optional
        Occupied inactive, active and unoccupied inactive spin-orbital indices; together
        they must partition range(n_basis) exactly once.
    n_elec : int, optional
        Total number of electrons, equal to n_act_elec plus len(core).
    n_act_elec : int, optional
        Number of electrons in the active space.
    rank : int, optional
        Position in the weight ordering of the negative-frequency poles, counting from
        one.  Defaults to 3, the satellite the frozen model asks for.

    Returns
    -------
    float
        The frequency in electronvolts of the rank-th heaviest pole of the interacting
        Green's function among those at negative frequency.  With no arguments this is
        the multi-reference GW satellite energy of the frozen model.
    """
    return 0.0
```
