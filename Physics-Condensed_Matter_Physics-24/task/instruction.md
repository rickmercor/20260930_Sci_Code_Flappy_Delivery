# Physics-Condensed_Matter_Physics-24

## Problem

Electrons occupying different orbitals of the same transition-metal ion are not independent: the Coulomb repulsion and Hund's coupling that act between them build correlations that no single-orbital picture captures, and quantifying how strongly two orbitals are entangled has become a way to characterize correlated metals and their approach to localization. The difficulty is that an orbital is not a qubit, since each one carries four local configurations rather than two, and any entanglement measure must therefore say precisely which configurations it retains and how it treats the rest. This calculation takes a two-orbital correlated impurity coupled to a discrete bath, forms its thermal state by exact diagonalization, and returns a single number measuring the entanglement between the two correlated orbitals.

The impurity carries two orbitals split by a crystal field, with intra-orbital repulsion $U$, Hund's coupling $J$, inter-orbital repulsion $U' = U - 2J$, and the spin-flip and pair-hopping terms that make the interaction rotationally invariant in orbital space. Each orbital couples to its own bath level through a hybridization with no inter-orbital component, so the orbitals communicate only through the local Coulomb terms. The full Hamiltonian is diagonalized exactly in the eight-mode Fock space of dimension 256, and the thermal density matrix at inverse temperature $\beta$ follows from its complete spectrum. The entanglement between the two correlated orbitals is then read from a small set of expectation values in that thermal state, with the pair of orbitals reduced to an effective pair of two-level systems. Identifying which local configurations that reduction retains, which expectation values enter, and how they combine into the measure, is the substance of this problem.

Work in the grand canonical ensemble with the chemical potential fixed at $\mu = (3U - 5J)/2$ and place both bath levels at zero energy. The two impurity orbitals are split symmetrically about their common centre, so that orbital 1 sits at $+\Delta/2 - \mu$ and orbital 2 at $-\Delta/2 - \mu$, and each orbital $\alpha$ hybridizes with its own bath level with amplitude $V_\alpha$. Measure every energy in the same units and report the entanglement measure in the normalization that its own definition carries. The configuration is

U = 2.5
J = 0.5
Delta = 1.2
V_1 = 0.60
V_2 = 0.35
beta = 8.0
n_bath_per_orbital = 1
n_orbitals = 2

Your final answer must be a single number: the orbital entanglement measure of the two correlated orbitals in this thermal state.

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

01_fock_space_operators

Goal
----
Construct the fermionic creation and annihilation matrices for a finite Fock space. The input fixes the number of modes and the returned arrays hold the real matrix elements of every creation operator together with its corresponding annihilation operator. Invalid input raises ValueError: n_modes must be an integer between 1 and 16.

```python
import numpy as np


def fock_space_operators(n_modes):
    """Build Jordan-Wigner operators in the occupation-number basis.

    Parameters
    ----------
    n_modes : int
        Number of fermionic modes, from 1 through 16.

    Returns
    -------
    c_dag : numpy.ndarray
        Real creation-operator array with shape
        ``(n_modes, 2**n_modes, 2**n_modes)``.
    c : numpy.ndarray
        Real annihilation-operator array with shape
        ``(n_modes, 2**n_modes, 2**n_modes)``.
    
    Raises
    ------
    ValueError
        If ``n_modes`` is not an integer, or lies outside 1 through 16.
    """
    return None
```

### Step 2

02_anderson_impurity_hamiltonian

Goal
----
Assemble the two-orbital Anderson impurity Hamiltonian in the many-body basis. The scalar interactions, bath arrays and system sizes determine a real symmetric matrix whose entries are many-body transition amplitudes and occupation energies in the Fock basis. Invalid input raises ValueError: the two size arguments must be positive integers, both bath arrays must have shape (n_orb, n_bath_per_orb) and the four scalar energies must be finite.

```python
import numpy as np


def anderson_impurity_hamiltonian(U, J, delta, mu, eps_bath, V_bath, n_orb, n_bath_per_orb):
    """Build the Anderson impurity Hamiltonian in the many-body basis.

    Parameters
    ----------
    U : float
        Intra-orbital Coulomb repulsion.
    J : float
        Hund coupling in the Kanamori interaction.
    delta : float
        Crystal-field splitting between the two impurity orbitals.
    mu : float
        Chemical potential shifting both impurity orbital energies.
    eps_bath : array_like
        Bath energies with shape ``(n_orb, n_bath_per_orb)``.
    V_bath : array_like
        Orbital-preserving impurity-bath hybridizations with shape
        ``(n_orb, n_bath_per_orb)``.
    n_orb : int
        Number of impurity orbitals.
    n_bath_per_orb : int
        Number of bath levels attached to each orbital.

    Returns
    -------
    H : numpy.ndarray
        Real symmetric many-body Hamiltonian with shape
        ``(2**n_modes, 2**n_modes)``, where
        ``n_modes = n_orb * 2 * (1 + n_bath_per_orb)``.
    
    Raises
    ------
    ValueError
        If ``n_orb`` or ``n_bath_per_orb`` is not a positive integer, if the
        bath arrays do not both have shape ``(n_orb, n_bath_per_orb)``, or if
        ``U``, ``J``, ``delta`` or ``mu`` is not a finite scalar.
    """
    return None
```

### Step 3

03_thermal_density_matrix

Goal
----
Construct the normalized thermal density matrix of a finite many-body Hamiltonian at inverse temperature beta. The returned matrix is real symmetric with unit trace, and every subsequent expectation value in this calculation is a trace against it. Invalid input raises ValueError: H must be a two-dimensional square symmetric array and beta must be positive and finite.

```python
import numpy as np


def thermal_density_matrix(H, beta):
    """Construct the normalized grand canonical thermal density matrix.

    Parameters
    ----------
    H : numpy.ndarray
        Real symmetric many-body Hamiltonian with shape ``(dim, dim)``.
    beta : float
        Positive finite inverse temperature.

    Returns
    -------
    rho : numpy.ndarray
        Real symmetric thermal density matrix with shape ``(dim, dim)`` and
        unit trace.
    
    Raises
    ------
    ValueError
        If ``H`` is not a two-dimensional square symmetric array, or if
        ``beta`` is not a positive finite scalar.
    """
    return None
```

### Step 4

04_retained_configuration_weights

Goal
----
Return the weights of four retained local configurations of a two-orbital impurity, ordered as a doubly occupied first orbital with an empty second, an up electron on the first with a down electron on the second, a down electron on the first with an up electron on the second, and an empty first orbital with a doubly occupied second. Invalid input raises ValueError: the two size arguments must be positive integers and rho must be square with the full Fock-space dimension.

```python
import numpy as np


def retained_configuration_weights(rho, n_orb, n_bath_per_orb):
    """Evaluate the four retained local configuration weights.

    Parameters
    ----------
    rho : array_like
        Trace-one thermal density matrix in the full Fock space.
    n_orb : int
        Number of impurity orbitals represented in the Fock space.
    n_bath_per_orb : int
        Number of bath levels attached to each impurity orbital.

    Returns
    -------
    weights : numpy.ndarray
        Four floating-point weights ordered as the first-orbital double
        occupation, the two opposite-spin separate occupations and the
        second-orbital double occupation.
    
    Raises
    ------
    ValueError
        If ``n_orb`` or ``n_bath_per_orb`` is not a positive integer, or if
        ``rho`` is not a square matrix of the full Fock-space dimension.
    """
    return None
```

### Step 5

05_spin_exchange_correlator

Goal
----
Return the magnitude of the spin-exchange coherence between the two impurity orbitals, evaluated in the supplied thermal state. Invalid input raises ValueError: the two size arguments must be positive integers and rho must be square with the full Fock-space dimension.

```python
import numpy as np


def spin_exchange_correlator(rho, n_orb, n_bath_per_orb):
    """Evaluate the magnitude of the interorbital spin-exchange correlator.

    Parameters
    ----------
    rho : array_like
        Trace-one thermal density matrix in the full Fock space.
    n_orb : int
        Number of impurity orbitals represented in the Fock space.
    n_bath_per_orb : int
        Number of bath levels attached to each impurity orbital.

    Returns
    -------
    magnitude : float
        Absolute value of the thermal spin-exchange expectation value.
    
    Raises
    ------
    ValueError
        If ``n_orb`` or ``n_bath_per_orb`` is not a positive integer, or if
        ``rho`` is not a square matrix of the full Fock-space dimension.
    """
    return None
```

### Step 6

06_pair_transfer_coherence

Goal
----
Return the magnitude of the coherence that transfers an electron pair from the second impurity orbital to the first, evaluated in the supplied thermal state. Invalid input raises ValueError: the two size arguments must be positive integers and rho must be square with the full Fock-space dimension.

```python
import numpy as np


def pair_transfer_coherence(rho, n_orb, n_bath_per_orb):
    """Compute the magnitude of the impurity pair-transfer coherence.

    Parameters
    ----------
    rho : array_like
        Many-body density matrix in the occupation-number basis.
    n_orb : int
        Number of impurity orbitals.
    n_bath_per_orb : int
        Number of bath levels attached to each orbital.

    Returns
    -------
    value : float
        Magnitude of the coherence that transfers an impurity pair from
        orbital one to orbital zero.
    
    Raises
    ------
    ValueError
        If ``n_orb`` or ``n_bath_per_orb`` is not a positive integer, or if
        ``rho`` is not a square matrix of the full Fock-space dimension.
    """
    return None
```

### Step 7

07_orbital_concurrence

Goal
----
Return the orbital entanglement measure built from four retained configuration weights and the magnitude of the surviving coherence between them. Invalid input raises ValueError: weights must hold four finite nonnegative numbers and z_magnitude must be a finite nonnegative scalar.

```python
import numpy as np


def orbital_concurrence(weights, z_magnitude):
    """Compute the spin-exchange branch of the orbital entanglement measure.

    Parameters
    ----------
    weights : array_like
        Four unnormalized populations ordered as ``[u_plus, w_1, w_2,
        u_minus]``.
    z_magnitude : float
        Magnitude of the spin-exchange coherence.

    Returns
    -------
    value : float
        Orbital entanglement measure from the spin-exchange branch.
    
    Raises
    ------
    ValueError
        If ``weights`` is not a one-dimensional sequence of four finite
        nonnegative numbers, or if ``z_magnitude`` is not a finite
        nonnegative scalar.
    """
    return None
```

### Step 8

08_run_pipeline

Goal
----
Orbital entanglement of a two-orbital Kanamori impurity at finite temperature.

```python
import numpy as np


def run_pipeline(U, J, delta, V_1, V_2, beta):
    """Compute the orbital entanglement measure of the two correlated orbitals.

    Parameters
    ----------
    U : float
        Intra-orbital Coulomb repulsion.
    J : float
        Hund's coupling. The inter-orbital repulsion is U - 2 J.
    delta : float
        Crystal-field splitting between the two impurity orbitals.
    V_1 : float
        Hybridization amplitude of the first orbital with its bath level.
    V_2 : float
        Hybridization amplitude of the second orbital with its bath level.
    beta : float
        Inverse temperature of the thermal state.

    Returns
    -------
    float
        Orbital entanglement measure of the two correlated orbitals.
    
    Raises
    ------
    ValueError
        If any argument is not finite, or if ``beta`` is not positive.
    """
    return None
```
