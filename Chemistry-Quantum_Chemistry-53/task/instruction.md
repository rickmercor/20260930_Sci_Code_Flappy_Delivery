# Chemistry-Quantum_Chemistry-53

## Background

Strongly correlated electrons remain one of the hard problems of electronic structure theory. Wave-function methods that treat correlation systematically scale exponentially with system size, while density functional theory is cheap but depends on approximate exchange-correlation functionals that become unreliable precisely where correlation is strong. Quantum embedding methods sit between the two: they cut the system into small fragments that are treated with a high-level method and reconnect the fragments through a low-level description of the rest of the system. Density matrix embedding theory (DMET) is the best known representative. It starts from a single Slater determinant for the whole system, builds for each fragment a compact bath of orbitals from the one-electron reduced density matrix of that determinant, projects the full Hamiltonian onto the fragment-plus-bath cluster, solves the cluster exactly, and feeds the result back into the mean-field description through a correlation potential and a chemical potential. The density-only flavour of DMET, called density embedding theory (DET), uses just the diagonal of the density matrix (the site occupations) in its self-consistency loop, which makes it look very much like a lattice version of Kohn-Sham theory.

A recent line of work has been pursuing that analogy seriously. It showed that a Householder transformation gives an explicit single bath orbital for every embedded site directly from the reference density matrix, and that the DET self-consistency can be recast as an in-principle exact density functional theory formulated on the lattice of localized orbitals. In that formulation the quantity that drives the embedding is a local (site diagonal) Hartree-exchange-correlation potential, the analogue of the Kohn-Sham potential, and the chemical potential that DMET attaches to the impurity is no longer a single global Lagrange multiplier but a site-dependent functional of that local potential. The practical method derived from these ideas, in which that local potential is the basic variable of the embedding, works well when electron correlation is strong and local, but its impurity chemical potential expression was derived in that limit and the method fails badly in weakly correlated situations, for molecular systems near equilibrium geometry in particular.

The work that this task is built on generalizes that method in the sense of generalized Kohn-Sham theory. Instead of a non-interacting reference with a fully local potential, the reference determinant is the ground state of a one-electron Hamiltonian that contains the non-local Hartree-Fock exchange potential (100 percent exact exchange) plus a local correlation potential, and only that correlation potential is optimized. The generalized method is benchmarked on small lattice models with non-uniform potentials and on short hydrogen chains in a minimal basis, for which exact full configuration interaction results are available. The comparisons show that the generalized method keeps the accuracy of its predecessor at strong coupling, remains numerically stable, and removes the predecessor's failure at weak coupling, while reproducing the exact density profile better than density embedding with a global chemical potential. The wider aim of this line of work is an exact density-functional embedding theory for molecules in which the functional is never written down explicitly but is evaluated through the embedding clusters themselves.

## Problem

Quantum embedding methods such as density matrix embedding theory treat strong local electron correlation by cutting a large system into small impurity-plus-bath clusters that are solved exactly and tied together through a mean-field reference. In the density-only variant of that approach (density embedding theory), each localized orbital is embedded in turn into a single bath orbital built from the reference one-electron density matrix, the full Hamiltonian is projected onto the cluster, and one global chemical potential shared by all clusters fixes the total electron number. A recently proposed embedding theory, formulated in the language of generalized Kohn-Sham density functional theory, reformulates this scheme: the reference determinant is the ground state of a one-electron Hamiltonian that contains the full (100 percent) Hartree-Fock exchange potential together with a local correlation potential carrying one value per site; every embedded orbital receives its own impurity chemical potential, which the theory expresses, in the strongly correlated limit, through that correlation potential and the bath orbital of the site; and the correlation potential is adjusted until the occupation of every site in the reference determinant coincides with the occupation of the same site in its own correlated cluster. Your task is to carry this self-consistent embedding through for a half-filled non-uniform Hubbard ring and to report the converged impurity chemical potential of one site.

The ring has six sites with periodic boundary conditions, nearest-neighbour hopping amplitude t, an on-site repulsion U n_{i up} n_{i down} on every site, and external on-site potentials v_ext that make the density profile non-uniform; it holds six electrons in a closed-shell (spin-restricted) treatment. Each site i is embedded singly: its bath is the one orbital that the Householder (equivalently Schmidt) construction extracts from the reference density matrix, the cluster Hamiltonian is the full Hamiltonian projected onto the determinants in which the remaining occupied reference orbitals (the core) stay doubly occupied while two electrons are distributed over the impurity and bath orbitals, and the impurity chemical potential of the cluster multiplies the impurity occupation operator and is subtracted from that projected Hamiltonian. The cluster is solved by exact diagonalization, the reference problem is solved self-consistently for every trial correlation potential (the Hartree-Fock potential, the bath orbitals and the cores all respond to it), and the correlation potential is started from zero, where the reference is the restricted Hartree-Fock determinant.

Use exactly this configuration:

- sites indexed 0 to 5 around the ring, site 5 bonded to site 0
- t = 1 (all energies in units of t), U = 7
- v_ext = (-1, 2, -2, 3, -3, 1) on sites 0 to 5
- six electrons, three doubly occupied reference orbitals
- spin-summed site occupations n_i = 2 gamma_ii are the matched densities
- convergence: the Euclidean norm of the site-by-site density mismatch below 1e-8

Report the impurity chemical potential mu_imp of site 2 (the site with v_ext = -2) at self-consistency, in units of t, with the sign convention that the cluster Hamiltonian of site 2 is the projected Hamiltonian minus mu_imp times the impurity occupation operator. Your final answer must be a single number: mu_imp of site 2. The few scalars that determine and validate this number are site-resolved, so report them explicitly in your reasoning: the restricted Hartree-Fock starting occupations, the starting density mismatch, the converged correlation potential, the converged per-spin site occupations together with their exact full-configuration-interaction counterparts, the bath coefficients of site 2, the impurity chemical potentials of all six sites, and the parameters, occupation and ground level of the converged site-2 cluster. Justify briefly the embedding choices your calculation makes, in particular which Hamiltonian the cluster sees, how each cluster's chemical potential is fixed and why the density mapping is well posed. These requested values are the scalars the output rules below ask you to show; the vectors those rules exclude are input matrices, orbital coefficient matrices and per-iteration histories.

Output Format Requirements:
Emit <final_answer> immediately, then <reasoning>. Do not write a long
derivation before the tags.
You must emit exactly one finite decimal inside
<final_answer>...</final_answer>, even if the value is approximate or you
are unsure.
Rules:
- The tags are required. Do not omit them or leave them empty.
- The tagged value must be a finite decimal (examples: 0.4847, 0.22, 1.05).
Not NaN, not Inf, not a fraction string, not a vector, not prose.
- Put only that one number between the tags. No units, no words, no extra
lines.
Keep <reasoning> short (a few hundred words). Show only the few scalars that
determine the final number.
Do not paste the input matrices, full coefficient vectors, per-iteration
paths, or per-fold candidate tables.

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

ring_one_body

Goal
----
The model is a ring of L localized orbitals (lattice sites) with nearest-neighbour hopping of amplitude t and periodic boundary conditions, so that site L-1 is bonded to site 0.

```python
import numpy as np


def ring_one_body(L, t, v_ext):
    '''One-electron Hamiltonian matrix of a periodic non-uniform ring.

    Parameters
    ----------
    L : int
        Number of sites, >= 3.
    t : float
        Nearest-neighbour hopping amplitude (the matrix element is -t).
    v_ext : array_like of float, shape (L,)
        External on-site potential of every site.

    Returns
    -------
    h : np.ndarray of float, shape (L, L)
        Real symmetric one-electron matrix in the site basis: h[i, i] =
        v_ext[i], h[i, (i + 1) % L] = h[(i + 1) % L, i] = -t, zero elsewhere.
        Raises ValueError if L < 3, if v_ext does not have L entries, or if
        any input is not finite.
    '''
    return np.zeros((L, L), dtype=float)
```

### Step 2

gks_density_matrix

Goal
----
The full-size reference system of the embedding is a generalized Kohn-Sham (gKS) determinant: a single closed-shell Slater determinant that is the ground state of    h_gKS = h + v_Hx[gamma] + sum_i v_c[i] n_i,  where h is the one-electron matrix of the ring, v_Hx[gamma] is the Hartree-Fock (100 percent exact exchange) potential built from the reference one-electron reduced density matrix gamma itself, and v_c is a local (site diagonal) correlation potential that is an external input of this step (it is optimized in a later step).

```python
import numpy as np


def gks_density_matrix(h, U, v_c, n_elec):
    '''Self-consistent closed-shell gKS one-electron reduced density matrix.

    Parameters
    ----------
    h : array_like of float, shape (L, L)
        Real symmetric one-electron matrix of the ring.
    U : float
        On-site repulsion; the Hartree-Fock potential is diag(U gamma_ii).
    v_c : array_like of float, shape (L,)
        Local correlation potential added to the diagonal of the Fock matrix.
    n_elec : int
        Even number of electrons, 2 <= n_elec <= 2L - 2; the lowest
        n_elec / 2 gKS orbitals are doubly occupied.

    Returns
    -------
    gamma : np.ndarray of float, shape (L, L)
        Converged per-spin density matrix gamma = C_occ C_occ^T (idempotent,
        trace n_elec / 2), converged so that the commutator F gamma - gamma F
        is below 1e-10 in absolute value. Raises ValueError for a non-square
        or non-symmetric h, a v_c of the wrong length, an odd or out-of-range
        n_elec, or a converged Fock matrix whose highest occupied and lowest
        unoccupied levels are degenerate (gap below 1e-8).
    '''
    return np.zeros_like(np.asarray(h, dtype=float))
```

### Step 3

bath_orbital

Goal
----
In single-orbital density matrix embedding, every localized orbital chi_i (site i) is embedded in turn into a quantum bath made of one delocalized orbital b^(i).

```python
import numpy as np


def bath_orbital(gamma, site):
    '''Single bath orbital of site `site` built from the reference 1-RDM.

    Parameters
    ----------
    gamma : array_like of float, shape (L, L)
        Real symmetric per-spin one-electron reduced density matrix of the
        reference determinant.
    site : int
        Index of the embedded orbital, 0 <= site < L.

    Returns
    -------
    b : np.ndarray of float, shape (L,)
        Site-basis coefficients of the normalized bath orbital: b[site] = 0,
        b[j] proportional to gamma[site, j] for j != site, sum_j b[j]^2 = 1.
        Raises ValueError if gamma is not square and symmetric, if site is
        out of range, or if the off-diagonal part of row `site` vanishes
        (norm below 1e-12).
    '''
    return np.zeros(np.asarray(gamma).shape[0], dtype=float)
```

### Step 4

cluster_hamiltonian

Goal
----
In the interacting-bath formulation of density matrix embedding, the cluster Hamiltonian of the embedded site i is the exact projection of the full Hamiltonian H onto the space spanned by the determinants    |Phi_core Phi_alpha>,  where Phi_core is the product of the occupied reference (gKS) orbitals that have no overlap with the impurity (they are also orthogonal to the bath, because the impurity plus bath space is invariant under gamma), and Phi_alpha runs over all determinants that distribute two electrons among the impurity orbital I = chi_i and the bath orbital B = b^(i).

```python
import numpy as np


def cluster_hamiltonian(h, U, gamma, site):
    '''Projected two-electron cluster Hamiltonian of the embedded site.

    Parameters
    ----------
    h : array_like of float, shape (L, L)
        Real symmetric one-electron matrix of the ring.
    U : float
        On-site repulsion.
    gamma : array_like of float, shape (L, L)
        Idempotent per-spin reference density matrix (the bath is built from
        its row `site`, and its projection on the orthogonal complement of the
        impurity plus bath space defines the frozen core).
    site : int
        Index of the embedded orbital, 0 <= site < L.

    Returns
    -------
    h_cl : np.ndarray of float, shape (4, 4)
        Matrix of the projected Hamiltonian (without chemical potential term
        and without the constant core energy) in the S_z = 0 basis
        |I up, I down>, |I up, B down>, |B up, I down>, |B up, B down>.
        Raises ValueError if gamma is not symmetric and idempotent (to 1e-6),
        if the cluster does not hold exactly one electron per spin, if site
        is out of range, or if no bath orbital exists for the site.
    '''
    return np.zeros((4, 4), dtype=float)
```

### Step 5

impurity_chemical_potential

Goal
----
Projecting the Hamiltonian onto an embedding cluster is exact only for a non-interacting or Hartree-Fock state.

```python
import numpy as np


def impurity_chemical_potential(b, v_c):
    '''gLPFET impurity chemical potential of one embedding cluster.

    Parameters
    ----------
    b : array_like of float, shape (L,)
        Normalized bath orbital of the embedded site in the site basis
        (zero on the embedded site itself).
    v_c : array_like of float, shape (L,)
        Local correlation potential of the gKS reference, one value per site.

    Returns
    -------
    mu : float
        Impurity chemical potential mu_imp of the cluster according to the
        gLPFET ansatz (the quantity subtracted, multiplied by the impurity
        occupation operator, from the projected cluster Hamiltonian).
        Raises ValueError if b and v_c do not have the same length or are
        not finite.
    '''
    return 0.0
```

### Step 6

cluster_site_density

Goal
----
The embedding cluster of site i is described by the Hamiltonian    H^(i) = P^(i) H P^(i) - mu_imp^(i) n_I,  where the first term is the projected cluster Hamiltonian of the previous steps (a 4 x 4 matrix in the S_z = 0 basis |I up, I down>, |I up, B down>, |B up, I down>, |B up, B down>) and n_I = n_{I up} + n_{I down} counts the electrons on the impurity orbital.

```python
import numpy as np


def cluster_site_density(h_cl, mu):
    '''Correlated impurity occupation of one embedding cluster.

    Parameters
    ----------
    h_cl : array_like of float, shape (4, 4)
        Real symmetric projected cluster Hamiltonian in the basis
        |I up, I down>, |I up, B down>, |B up, I down>, |B up, B down>.
    mu : float
        Impurity chemical potential; the cluster Hamiltonian solved is
        h_cl - mu * diag(2, 1, 1, 0).

    Returns
    -------
    n_cl : float
        Spin-summed occupation of the impurity orbital in the ground state
        of the cluster, between 0 and 2. Raises ValueError if h_cl is not a
        finite symmetric 4 x 4 matrix, if mu is not finite, or if the ground
        level of the cluster is degenerate (gap below 1e-10).
    '''
    return 0.0
```

### Step 7

density_mismatch

Goal
----
The self-consistency condition of gLPFET is a set of local density constraints: for every site i, the occupation of site i in the gKS reference determinant must equal the correlated occupation of that site in its own embedding cluster,    <n_i>_gKS = <n_i>_H^(i)   for all i,  both being spin-summed densities (n_i^gKS = 2 gamma_ii).

```python
import numpy as np


def density_mismatch(h, U, v_c, n_elec):
    '''Site-resolved gLPFET density mismatch for a given correlation potential.

    Parameters
    ----------
    h : array_like of float, shape (L, L)
        Real symmetric one-electron matrix of the ring.
    U : float
        On-site repulsion.
    v_c : array_like of float, shape (L,)
        Local correlation potential of the gKS reference.
    n_elec : int
        Even number of electrons, 2 <= n_elec <= 2L - 2.

    Returns
    -------
    r : np.ndarray of float, shape (L,)
        r[i] = n_i^cl - n_i^gKS, the spin-summed correlated occupation of
        site i from its embedding cluster minus the gKS occupation of the
        same site. Raises ValueError for invalid h, U, v_c or n_elec (same
        rules as the reference and cluster steps).
    '''
    return np.zeros(np.asarray(h).shape[0], dtype=float)
```

### Step 8

solve_correlation_potential

Goal
----
The outer loop of gLPFET determines the local correlation potential v_c from the local density constraints: the L unknowns v_c[0..L-1] are fixed by the L equations    n_i^cl(v_c) - n_i^gKS(v_c) = 0,   i = 0..L-1.

```python
import numpy as np


def solve_correlation_potential(h, U, n_elec, tol=1e-10):
    '''Self-consistent gLPFET correlation potential of the ring.

    Parameters
    ----------
    h : array_like of float, shape (L, L)
        Real symmetric one-electron matrix of the ring.
    U : float
        On-site repulsion.
    n_elec : int
        Even number of electrons, 2 <= n_elec <= 2L - 2.
    tol : float
        Required upper bound on the converged mismatch norm Delta.

    Returns
    -------
    v_c : np.ndarray of float, shape (L,)
        Local correlation potential for which the cluster densities and the
        gKS densities agree on every site (mismatch norm below tol), obtained
        from the starting point v_c = 0. Raises ValueError for invalid inputs
        (same rules as the reference step) and RuntimeError if the density
        mapping cannot be converged.
    '''
    return np.zeros(np.asarray(h).shape[0], dtype=float)
```

### Step 9

hubbard_fci_reference

Goal
----
Construct the exact closed-shell FCI benchmark and return its energy, full spin-summed one-particle density matrix, site-resolved double occupations, and exact static density-response matrix.

```python
import numpy as np


def hubbard_fci_reference(h, U, n_elec):
    '''Exact closed-shell reference for a finite Hubbard Hamiltonian.

    Parameters
    ----------
    h : array_like of float, shape (L, L)
        Finite real symmetric one-electron matrix in the site basis.
    U : float
        On-site interaction in U n_{i up} n_{i down}.
    n_elec : int
        Even number of electrons, 2 <= n_elec <= 2L - 2. The calculation
        uses N_up = N_down = n_elec / 2 and supports 2 <= L <= 6.

    Returns
    -------
    result : np.ndarray of float, shape (1 + 2*L*L + L,)
        result[0] is the exact ground-state energy. The next L*L entries are
        the row-major spin-summed one-particle density matrix gamma[p,q] =
        sum_sigma <a^dagger_(p,sigma) a_(q,sigma)>. The following L entries
        are <n_(i,up) n_(i,down)>. The final L*L entries are the row-major
        static response d<n_i>/d<v_j>. Raises ValueError for invalid inputs
        or a degenerate ground state with a gap below 1e-10.
    '''
    return np.zeros(
        1 + 2 * np.asarray(h).shape[0] ** 2 + np.asarray(h).shape[0],
        dtype=float,
    )
```

### Step 10

glpfet_impurity_potential

Goal
----
Run the complete gLPFET workflow, validate the exact reduced-density and static-response FCI reference, and return the converged impurity chemical potential of the requested site.

```python
import numpy as np


def glpfet_impurity_potential(L, t, v_ext, U, n_elec, site):
    '''Converged gLPFET impurity chemical potential of one site of the ring.

    Chains every earlier step: ring_one_body, solve_correlation_potential,
    then density_mismatch at the converged potential (the density mapping must
    hold to 1e-8), gks_density_matrix, hubbard_fci_reference (including its
    reduced-density energy closure and static-response invariants),
    bath_orbital and impurity_chemical_potential for the requested site, and
    finally cluster_hamiltonian and cluster_site_density to rebuild the site's
    cluster and confirm that its occupation matches the reference density.

    Parameters
    ----------
    L : int
        Number of sites, 3 <= L <= 6.
    t : float
        Nearest-neighbour hopping amplitude.
    v_ext : array_like of float, shape (L,)
        External on-site potential.
    U : float
        On-site repulsion.
    n_elec : int
        Even number of electrons, 2 <= n_elec <= 2L - 2.
    site : int
        Index of the embedded site whose impurity chemical potential is
        requested, 0 <= site < L.

    Returns
    -------
    mu : float
        The gLPFET impurity chemical potential mu_imp of the requested site
        at self-consistency. Raises ValueError for invalid inputs.
    '''
    return 0.0
```
