# Chemistry-Quantum_Chemistry-58

## Background

Linear polyenes and the carotenoids built on them are the textbook systems of pi-electron correlation. Their absorption spectrum is dominated by a bright, dipole-allowed singlet, yet the lowest excited singlet is a dark state of the same symmetry as the ground state, and its very existence below the bright state is a failure of any one-electron picture: it appears only when doubly excited configurations are allowed to mix, and it is best understood as a strongly correlated, largely covalent excitation in which the electrons stay one per atom and only their spins are rearranged. Semi-empirical pi-electron Hamiltonians of the Pariser-Parr-Pople type, with a short-range Hubbard repulsion and a screened long-range Coulomb tail, reproduce the ordering of these states and have been solved essentially exactly for short chains by full configuration interaction and for long chains by the density matrix renormalisation group.

A recurring interpretation of the dark state is that it is a pair of triplet excitations bound together on the same chain. The idea matters beyond spectroscopy because singlet fission, the conversion of one photoexcited singlet into two triplets, requires an intermediate of exactly this kind, and whether a covalent dark state can serve as that intermediate depends on how much of it really is a triplet pair, how strongly the pair is bound, and how both quantities depend on the Coulomb interaction and on the chain length. Quantifying the triplet-pair character requires a definition that is formulated in real space, so that it corresponds to triplets localised on separate parts of the molecule, and that respects the spin and particle-hole symmetries which organise the excitations of half-filled bipartite chains into covalent and ionic families.

## Problem

The lowest excited singlet of a linear polyene, the dipole-forbidden 2(1)Ag state, has long been described as a bound pair of triplet excitations, and a recent treatment makes that description quantitative: it defines a real-space triplet-pair population by projecting the state onto tensor products of triplet eigenstates of the two subchains into which the molecule can be cut, and extrapolates the population to the infinite chain. Your task is to carry out that analysis for a Pariser-Parr-Pople model of short polyenes and report the extrapolated triplet-pair population of the dark state.

The pi electrons of an open chain of N atoms (N even, one electron per atom) are described by H = -sum over bonds k = 1..N-1 and spins of t_k (c+(k,s) c(k+1,s) + h.c.) + U sum over atoms i of (n(i,up) - 1/2)(n(i,down) - 1/2) + sum over pairs i < j of V_ij (n_i - 1)(n_j - 1), where bond k joins atoms k and k+1 and carries t_k = t0 (1 + delta) for odd k (the double bonds) and t0 (1 - delta) for even k, with t0 = 2.5 eV, delta = 0.10 and U = 4.0 eV, and the Ohno interaction V_ij = U / sqrt(1 + (U eps r_ij / 14.397)^2) with U in eV, the through-space distance r_ij in angstrom and eps = 2.3. The atoms lie on a planar zigzag with double bonds of 1.35 angstrom, single bonds of 1.46 angstrom and a bond angle of 120 degrees at every atom.

Label every eigenstate by its total spin, by its parity under the spatial inversion that maps atom i to atom N + 1 - i, and by its eigenvalue under the alternancy transformation that replaces every creation operator c+(i,s) by (-1)^i c(i,s) and maps the empty chain onto the completely filled one; call a singlet covalent when its alternancy eigenvalue equals that of the ground state, and a triplet covalent when its eigenvalue equals that of the lowest triplet. The dark state 2(1)Ag is the lowest covalent singlet above the ground state that shares the ground state's inversion eigenvalue. Cut the chain of N_d = N/2 ethylene units into a left subchain of m units and a right subchain of N_d - m units for every m from 1 to N_d - 1, each subchain being an open chain of the same kind whose atoms keep their hopping integrals and their mutual Ohno interactions; let T_j(m) be the Sz = 0 component of the j-th lowest covalent triplet of the m-unit subchain, form the N_d (N_d - 1)/2 tensor products T_j(m) x T_1(N_d - m) with j = 1, ..., m, orthonormalise them by Loewdin's symmetric procedure, and take the triplet-pair population of a singlet as three times the sum of its squared projections onto that basis, the factor that the spin-symmetrised singlet pair requires; for the Sz = 0 component of the lowest quintet the corresponding factor is 3/2.

Compute the dark-state population P(N) for N = 6, 8 and 10, fit P(N) = a N^(-alpha) + c exactly through the three values, and report c as the final answer. In the reasoning give, to six significant figures, for the ten-atom chain the energies in eV of the ground state, the lowest triplet, the dark state and the lowest quintet, the smallest and the largest eigenvalue of the overlap matrix of the ten raw tensor products, the dark-state population resolved by the position of the cut (the sum over j for each m), the triplet-pair population of the lowest quintet and the triplet-pair binding energy E(1(5)Ag) - E(2(1)Ag); give P(6), P(8), P(10) and alpha; explain the origin of the factor of three, state whether the total population depends on the choice of orthonormalisation, and give as a check the number of products, the quintet population and the dark-state population of the four-atom chain; and state two results of the earlier work on which this definition rests: the total squared overlap that the direct-product analysis with triplets on the two halves of a twelve-atom chain assigned to the vertical 2(1)Ag state, and the energy by which the triplet pair and the charge-transfer exciton stabilise each other at resonance in the effective two-particle model of the dark state when both hoppings have magnitude 1 eV and their nearest-neighbour coupling is 3 eV.

Output Format Requirements:
Emit <final_answer> immediately, then <reasoning>. Do not write a long derivation before the tags.
You must emit exactly one finite decimal inside <final_answer>...</final_answer>, even if the value is approximate or you are unsure.
Rules:
- The tags are required. Do not omit them or leave them empty.
- The tagged value must be a finite decimal (examples: 0.4847, 0.22, 1.05). Not NaN, not Inf, not a fraction string, not a vector, not prose.
- Put only that one number between the tags. No units, no words, no extra lines.
Keep <reasoning> short (a few hundred words). Give the values requested above and only the few other scalars that determine the final number. Do not paste the input matrices, per-state lists or per-iteration paths.

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

ohno_interaction_matrix

Goal
----
Through-space Ohno interaction matrix of a planar zigzag polyene chain built from its bond lengths and bond angle.

```python
import itertools
import math
import types

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.optimize import brentq


def ohno_interaction_matrix(n_sites: int, bond_double: float, bond_single: float, bond_angle_deg: float, U: float, eps_r: float) -> np.ndarray:
    '''Ohno interaction matrix of a planar zigzag chain.

    Parameters
    ----------
    n_sites : int
        Number of carbon atoms N, an even integer >= 2.
    bond_double : float
        Length of the double bonds (bonds 1, 3, 5, ...) in angstrom, positive.
    bond_single : float
        Length of the single bonds (bonds 2, 4, 6, ...) in angstrom, positive.
    bond_angle_deg : float
        Bond angle at every atom in degrees, in (0, 180]; 180 gives a straight chain.
    U : float
        Hubbard parameter in eV, positive.
    eps_r : float
        Relative permittivity of the Ohno form, positive.

    Returns
    -------
    V : numpy.ndarray
        Shape (n_sites, n_sites), V[i, j] = U / sqrt(1 + (U eps_r r_ij / 14.397)^2) for
        i != j with r_ij the through-space distance in angstrom, and V[i, i] = 0.

    Raises
    ------
    ValueError
        If n_sites is not an even integer >= 2, a bond length is not positive,
        bond_angle_deg is outside (0, 180], or U or eps_r is not positive.
    '''
    return V
```

### Step 2

sector_eigenvalues

Goal
----
Lowest eigenvalues of the Pariser-Parr-Pople Hamiltonian in a sector of fixed numbers of spin-up and spin-down electrons.

```python
import itertools
import math
import types

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.optimize import brentq


def sector_eigenvalues(n_sites: int, t0: float, delta: float, V: np.ndarray, U: float, n_up: int, n_down: int, n_states: int) -> np.ndarray:
    '''Lowest eigenvalues of the PPP chain in the (n_up, n_down) sector.

    Parameters
    ----------
    n_sites : int
        Number of atoms N, an even integer >= 2.
    t0 : float
        Mean hopping integral in eV, positive.
    delta : float
        Bond alternation, |delta| < 1; bond k carries t0 (1 + delta) for odd k and t0 (1 - delta) for even k.
    V : numpy.ndarray
        Symmetric (n_sites, n_sites) interaction matrix in eV with a zero diagonal.
    U : float
        Hubbard parameter in eV, positive.
    n_up : int
        Number of spin-up electrons, between 0 and n_sites.
    n_down : int
        Number of spin-down electrons, between 0 and n_sites.
    n_states : int
        Number of eigenvalues wanted, between 1 and the sector dimension.

    Returns
    -------
    energies : numpy.ndarray
        Shape (n_states,), the lowest eigenvalues in ascending order, in eV.

    Raises
    ------
    ValueError
        If n_sites is not an even integer >= 2, V is not a symmetric (n_sites, n_sites)
        array, t0 or U is not positive, |delta| >= 1, n_up or n_down is outside
        [0, n_sites], or n_states is outside [1, sector dimension].
    '''
    return energies
```

### Step 3

covalent_triplet_family

Goal
----
Energies and covalent weights of the lowest members of the covalent triplet family of a chain, selected by their alternancy eigenvalue.

```python
import itertools
import math
import types

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.optimize import brentq


def covalent_triplet_family(n_sites: int, t0: float, delta: float, V: np.ndarray, U: float, n_members: int) -> np.ndarray:
    '''Lowest covalent triplets of the chain in the Sz = 0 sector.

    Parameters
    ----------
    n_sites : int
        Number of atoms N, an even integer >= 2.
    t0 : float
        Mean hopping integral in eV, positive.
    delta : float
        Bond alternation, |delta| < 1, with the double bond first.
    V : numpy.ndarray
        Symmetric (n_sites, n_sites) interaction matrix in eV with a zero diagonal.
    U : float
        Hubbard parameter in eV, positive.
    n_members : int
        Number of covalent triplets wanted, a positive integer.

    Returns
    -------
    family : numpy.ndarray
        Shape (n_members, 2). Row j holds the energy in eV of the (j+1)-th lowest triplet
        whose alternancy eigenvalue equals that of the lowest triplet, and the covalent
        weight of that state (total weight of the determinants with every atom singly
        occupied). Rows are in ascending energy.

    Raises
    ------
    ValueError
        If n_sites is not an even integer >= 2, V is not a symmetric (n_sites, n_sites)
        array, t0 or U is not positive, |delta| >= 1, n_members is not a positive
        integer, or the sector holds fewer than n_members covalent triplets.
    '''
    return family
```

### Step 4

dark_state_energies

Goal
----
Energies of the ground state, the lowest triplet, the covalent dark state 2(1)Ag and the lowest quintet of a chain.

```python
import itertools
import math
import types

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.optimize import brentq


def dark_state_energies(n_sites: int, t0: float, delta: float, V: np.ndarray, U: float) -> np.ndarray:
    '''Ground, lowest-triplet, dark-state and lowest-quintet energies of the chain.

    Parameters
    ----------
    n_sites : int
        Number of atoms N, an even integer >= 4.
    t0 : float
        Mean hopping integral in eV, positive.
    delta : float
        Bond alternation, |delta| < 1, with the double bond first.
    V : numpy.ndarray
        Symmetric (n_sites, n_sites) interaction matrix in eV with a zero diagonal.
    U : float
        Hubbard parameter in eV, positive.

    Returns
    -------
    energies : numpy.ndarray
        Shape (4,): E(ground state), E(lowest triplet), E(2(1)Ag) with 2(1)Ag the lowest
        inversion-even singlet above the ground state whose alternancy eigenvalue equals
        that of the ground state, and E(lowest quintet), all in eV.

    Raises
    ------
    ValueError
        If n_sites is not an even integer >= 4, V is not a symmetric (n_sites, n_sites)
        array, t0 or U is not positive, or |delta| >= 1.
    '''
    return energies
```

### Step 5

triplet_pair_gram_spectrum

Goal
----
Eigenvalues of the overlap matrix of the raw triplet-pair tensor products built from the covalent triplets of every left and right subchain.

```python
import itertools
import math
import types

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.optimize import brentq


def triplet_pair_gram_spectrum(n_sites: int, t0: float, delta: float, V: np.ndarray, U: float) -> np.ndarray:
    '''Eigenvalues of the overlap matrix of the triplet-pair products.

    Parameters
    ----------
    n_sites : int
        Number of atoms N, an even integer >= 4.
    t0 : float
        Mean hopping integral in eV, positive.
    delta : float
        Bond alternation, |delta| < 1, with the double bond first.
    V : numpy.ndarray
        Symmetric (n_sites, n_sites) interaction matrix in eV with a zero diagonal.
    U : float
        Hubbard parameter in eV, positive.

    Returns
    -------
    spectrum : numpy.ndarray
        Shape (N_d (N_d - 1)/2,) with N_d = n_sites/2: the eigenvalues, in ascending
        order, of the overlap matrix of the normalised products T_j(m) x T_1(N_d - m)
        for m = 1..N_d - 1 and j = 1..m.

    Raises
    ------
    ValueError
        If n_sites is not an even integer >= 4, V is not a symmetric (n_sites, n_sites)
        array, t0 or U is not positive, |delta| >= 1, or a subchain holds fewer covalent
        triplets than the construction needs.
    '''
    return spectrum
```

### Step 6

triplet_pair_populations

Goal
----
Triplet-pair population of the dark state 2(1)Ag resolved over the Loewdin-orthonormalised triplet-pair products.

```python
import itertools
import math
import types

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.optimize import brentq


def triplet_pair_populations(n_sites: int, t0: float, delta: float, V: np.ndarray, U: float) -> np.ndarray:
    '''Per-element triplet-pair populations of the dark state.

    Parameters
    ----------
    n_sites : int
        Number of atoms N, an even integer >= 4.
    t0 : float
        Mean hopping integral in eV, positive.
    delta : float
        Bond alternation, |delta| < 1, with the double bond first.
    V : numpy.ndarray
        Symmetric (n_sites, n_sites) interaction matrix in eV with a zero diagonal.
    U : float
        Hubbard parameter in eV, positive.

    Returns
    -------
    populations : numpy.ndarray
        Shape (N_d (N_d - 1)/2,) with N_d = n_sites/2: for the Loewdin-orthonormalised
        products in the order m = 1..N_d - 1, j = 1..m, three times the squared projection
        of the 2(1)Ag state onto each; the sum is the triplet-pair population.

    Raises
    ------
    ValueError
        If n_sites is not an even integer >= 4, V is not a symmetric (n_sites, n_sites)
        array, t0 or U is not positive, |delta| >= 1, or the products are linearly
        dependent.
    '''
    return populations
```

### Step 7

quintet_population_and_binding

Goal
----
Triplet-pair population of the lowest quintet and the triplet-pair binding energy of the dark state.

```python
import itertools
import math
import types

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.optimize import brentq


def quintet_population_and_binding(n_sites: int, t0: float, delta: float, V: np.ndarray, U: float) -> np.ndarray:
    '''Quintet triplet-pair population and triplet-pair binding energy.

    Parameters
    ----------
    n_sites : int
        Number of atoms N, an even integer >= 4.
    t0 : float
        Mean hopping integral in eV, positive.
    delta : float
        Bond alternation, |delta| < 1, with the double bond first.
    V : numpy.ndarray
        Symmetric (n_sites, n_sites) interaction matrix in eV with a zero diagonal.
    U : float
        Hubbard parameter in eV, positive.

    Returns
    -------
    result : numpy.ndarray
        Shape (2,): 3/2 times the sum of the squared projections of the Sz = 0 component
        of the lowest quintet onto the Loewdin-orthonormalised triplet-pair products, and
        E(lowest quintet) - E(2(1)Ag) in eV.

    Raises
    ------
    ValueError
        If n_sites is not an even integer >= 4, V is not a symmetric (n_sites, n_sites)
        array, t0 or U is not positive, |delta| >= 1, or the products are linearly
        dependent.
    '''
    return result
```

### Step 8

extrapolated_triplet_pair_population

Goal
----
Infinite-chain triplet-pair population of the dark state from an exact three-point power-law fit through the populations of three chain lengths.

```python
import itertools
import math
import types

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.optimize import brentq


def extrapolated_triplet_pair_population(bond_double: float, bond_single: float, bond_angle_deg: float, t0: float, delta: float, U: float, eps_r: float, chain_lengths: np.ndarray) -> float:
    '''Infinite-chain triplet-pair population from three chain lengths.

    Parameters
    ----------
    bond_double : float
        Length of the double bonds in angstrom, positive.
    bond_single : float
        Length of the single bonds in angstrom, positive.
    bond_angle_deg : float
        Bond angle in degrees, in (0, 180].
    t0 : float
        Mean hopping integral in eV, positive.
    delta : float
        Bond alternation, |delta| < 1, with the double bond first.
    U : float
        Hubbard parameter in eV, positive.
    eps_r : float
        Relative permittivity of the Ohno interaction, positive.
    chain_lengths : numpy.ndarray
        Three increasing even integers >= 4, the numbers of atoms of the chains.

    Returns
    -------
    c : float
        The constant of the exact fit P(N) = a N^(-alpha) + c through the three
        dark-state triplet-pair populations, as a native Python float.

    Raises
    ------
    ValueError
        If chain_lengths does not hold exactly three increasing even integers >= 4, a
        model parameter is invalid as in the earlier steps, or no positive alpha fits
        the three populations.
    '''
    return c
```
