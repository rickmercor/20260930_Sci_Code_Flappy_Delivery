# Physics-Condensed_Matter_Physics-52

## Background

Finite quantum impurity models provide controlled laboratories for studying how local interactions reshape single-particle excitations.  When orbitals mix coherently, Green's functions, self-energies, and hybridization functions are matrix valued, so their off-diagonal phases can carry physical information that is lost in channel-by-channel scalar treatments.

Discrete spectral representations make causality visible through real poles and positive-semidefinite matrix residues.  They also permit interacting response functions to be coupled back into another finite impurity problem without analytic continuation, while spectral sum rules and eigenvalue bounds provide stringent numerical checks.  A regularized low-energy response is useful when a finite spectrum has poles close to the chemical potential, where an unregularized derivative would be singular or poorly conditioned.

## Problem

A finite two-orbital Anderson impurity with coherent orbital mixing is to be propagated through one bounded exact discrete self-consistency update, retaining the full matrix structure throughout.  In the occupation-bit convention, impurity modes are 0 and 1, the two bath modes follow, and the Hamiltonian is $H=d^\dagger E_{\mathrm{imp}}d+c^\dagger E_{\mathrm{bath}}c+c^\dagger Vd+d^\dagger V^\dagger c+U n_0n_1$.

Use
$$
E_{\mathrm{imp}}=\begin{pmatrix}-0.973779&0.112032-0.097339i\\0.112032+0.097339i&-1.300553\end{pmatrix},\quad
E_{\mathrm{bath}}=\begin{pmatrix}-1.084493&-0.148375-0.048758i\\-0.148375+0.048758i&1.029384\end{pmatrix},
$$
$$
V=\begin{pmatrix}-0.049439-0.310296i&-0.118014+0.213365i\\0.009953+0.122430i&-0.019628+0.285367i\end{pmatrix},\qquad U=2.367874.
$$
The initial zero-temperature canonical solve uses particle number $N=2$, and the feedback solve uses $N=4$.

Apply the causal full-matrix symmetric equation-of-motion estimator in its doubled-propagator form, reconstruct its discrete self-energy exactly in pole space, and insert that matrix self-energy into the exact discrete hybridization feedback map applied directly to the supplied bath block $(E_{\mathrm{bath}},V)$, with bath modes 2 and 3 carrying the impurity orbital labels 0 and 1.  Realize the entire causal feedback spectrum as an unreduced auxiliary bath, perform the second interacting canonical solve, and apply the method's regularized matrix quasiparticle response with regulator $\lambda=0.04$; do not diagonalize the two physical channels into independent scalar problems.  Treat energies as ground-state-referenced Lehmann energies and coalesce only pole groups whose span is at most $10^{-10}$; omit a coalesced Lehmann group only when the trace of its summed residue is at most $10^{-12}$.  In each later positive-semidefinite metric or residue eigendecomposition, retain an eigenvalue $r_k$ only when $r_k>2\times10^{-11}\max(1,\max_\ell|r_\ell|)$, and omit a reconstructed self-energy pole group only when the trace of its summed residue is at most $2\times10^{-11}$.

For the normalized probe $u=(\sqrt{0.6},e^{0.73i}\sqrt{0.4})^T$, compute the real scalar $u^\dagger Z_\lambda u$ after the second solve and report it rounded once, at the end, to ten digits after the decimal point.

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

01_build_noncommuting_impurity_model

Goal
----
Build the finite many-body representation of a two-component impurity coupled to matrix-valued bath sites.

```python
def build_noncommuting_impurity_model(
    impurity_energy: "np.ndarray",
    bath_energies: "np.ndarray",
    bath_couplings: "np.ndarray",
    interaction_u: float,
) -> "tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]":
    """Return the Hamiltonians, annihilators, and particle-number labels.

    Parameters
    ----------
    impurity_energy : np.ndarray
        Finite Hermitian array of shape ``(2, 2)``.
    bath_energies : np.ndarray
        Finite Hermitian bath blocks of shape ``(M, 2, 2)``; ``M`` may be zero.
    bath_couplings : np.ndarray
        Finite impurity-to-bath blocks of shape ``(M, 2, 2)``.  With impurity
        column ``a0`` and bath column ``ai``, block ``i`` contributes
        ``a0^H bath_couplings[i]^H ai + ai^H bath_couplings[i] a0``.
    interaction_u : float
        Finite real coefficient of the impurity interaction ``n_0 n_1``.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        The full Hamiltonian and interaction Hamiltonian, each of shape
        ``(2**L, 2**L)``, all annihilation matrices with shape
        ``(L, 2**L, 2**L)``, and the integer particle number of every basis
        state with shape ``(2**L,)``.  Here ``L = 2*(M+1)``.  The occupation-bit
        basis uses ``(state >> p) & 1`` for mode ``p``; impurity modes are 0 and
        1, bath modes are site-major, and fermionic signs count occupied modes
        with lower indices.

    Raises
    ------
    ValueError
        If shapes, finiteness, Hermiticity, or the reality of ``interaction_u``
        violate the contract.

    Notes
    -----
    Input arrays are not required to be preserved by implementations.
    """
    return hamiltonian, interaction_hamiltonian, annihilators, occupations
```

### Step 2

02_compute_augmented_matrix_lehmann_spectrum

Goal
----
Compute the complete zero-temperature discrete spectrum of the doubled impurity propagator.

```python
def compute_augmented_matrix_lehmann_spectrum(
    hamiltonian: "np.ndarray",
    interaction_hamiltonian: "np.ndarray",
    annihilators: "np.ndarray",
    occupations: "np.ndarray",
    particle_number: int,
) -> "tuple[float, np.ndarray, np.ndarray]":
    """Return the canonical ground energy and augmented Lehmann spectrum.

    Parameters
    ----------
    hamiltonian : np.ndarray
        Finite Hermitian many-body Hamiltonian of shape ``(D, D)``.
    interaction_hamiltonian : np.ndarray
        Matching finite Hermitian interaction Hamiltonian.
    annihilators : np.ndarray
        Mode-resolved annihilation matrices of shape ``(L, D, D)`` in the
        occupation-bit convention of the preceding model construction.
    occupations : np.ndarray
        Integer particle-number labels of shape ``(D,)``.
    particle_number : int
        Canonical reference sector with a unique lowest state and both adjacent
        particle sectors available.

    Returns
    -------
    tuple[float, np.ndarray, np.ndarray]
        The ground energy, real poles of shape ``(P,)``, and Hermitian
        positive-semidefinite residues of shape ``(P, 4, 4)`` for the complete
        zero-temperature doubled propagator.  Component order is the two
        physical impurity channels followed by their two auxiliary channels.
        Pole order and equivalent decompositions within an exactly degenerate
        pole subspace do not change the represented propagator.

    Raises
    ------
    ValueError
        If dimensions, finiteness, Hermiticity, occupation labels, particle
        sector, canonical algebra, or reference-state uniqueness violate the
        contract.

    Notes
    -----
    Input arrays are not required to be preserved by implementations.
    Pole groups with span at most ``1e-10`` are coalesced, and a group whose
    summed-residue trace is at most ``1e-12`` is numerically null and omitted.
    """
    return ground_energy, poles, residues
```

### Step 3

03_recover_causal_matrix_self_energy

Goal
----
Recover a causal matrix-valued discrete self-energy from an augmented Lehmann spectrum.

```python
def recover_causal_matrix_self_energy(
    augmented_poles: "np.ndarray",
    augmented_residues: "np.ndarray",
) -> "tuple[np.ndarray, np.ndarray, np.ndarray]":
    """Return the static matrix and causal dynamical self-energy spectrum.

    Parameters
    ----------
    augmented_poles : np.ndarray
        Finite real poles of shape ``(P,)``.
    augmented_residues : np.ndarray
        Hermitian positive-semidefinite residues of shape ``(P, 2*N, 2*N)``
        for a complete normalized doubled propagator, ordered with physical
        channels first and matching auxiliary channels second.  The physical
        zeroth moment is the ``N``-dimensional identity.  Either the complete
        doubled norm is positive definite or the auxiliary sector is identically
        zero.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        The Hermitian static contribution of shape ``(N, N)``, real dynamical
        poles of shape ``(Q,)``, and Hermitian positive-semidefinite dynamical
        residues of shape ``(Q, N, N)``.  Ordering and equivalent splitting of
        exactly degenerate residues do not change the represented self-energy.
        An identically zero auxiliary sector returns empty dynamical arrays.

    Raises
    ------
    ValueError
        If shapes, finiteness, reality, positivity, normalization, parity of the
        doubled dimension, or the required spectral metric violate the contract.

    Notes
    -----
    Input arrays are not required to be preserved by implementations.
    For each positive-semidefinite metric or residue with eigenvalues ``r``,
    numerical rank retains ``r[k] > 2e-11 * max(1, max(abs(r)))``.  A
    reconstructed pole group whose summed-residue trace is at most ``2e-11``
    is numerically null and omitted.
    """
    return self_energy_static, self_energy_poles, self_energy_residues
```

### Step 4

04_update_exact_matrix_hybridization

Goal
----
Apply the exact discrete matrix feedback map to a bath and a causal self-energy spectrum.

```python
def update_exact_matrix_hybridization(
    bath_energies: "np.ndarray",
    bath_couplings: "np.ndarray",
    self_energy_static: "np.ndarray",
    self_energy_poles: "np.ndarray",
    self_energy_residues: "np.ndarray",
) -> "tuple[np.ndarray, np.ndarray]":
    """Return the unreduced causal spectrum of the feedback hybridization.

    Parameters
    ----------
    bath_energies : np.ndarray
        Finite Hermitian blocks of shape ``(M, N, N)``.
    bath_couplings : np.ndarray
        Finite coupling blocks of matching shape in the convention where the
        original hybridization is represented by these bath blocks.
    self_energy_static : np.ndarray
        Finite Hermitian static matrix of shape ``(N, N)`` in the same fixed
        orbital basis as the bath blocks.
    self_energy_poles : np.ndarray
        Finite real dynamical poles of shape ``(Q,)``.
    self_energy_residues : np.ndarray
        Hermitian positive-semidefinite residues of shape ``(Q, N, N)``.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Real poles of shape ``(R,)`` and Hermitian positive-semidefinite
        residues of shape ``(R, N, N)`` for the exact discrete feedback
        hybridization.  No approximate pole reduction is applied.  Pole order
        and equivalent decompositions inside exactly degenerate subspaces do
        not change the represented function.

    Raises
    ------
    ValueError
        If dimensions, finiteness, reality, Hermiticity, or positivity violate
        the contract.

    Notes
    -----
    Input arrays are not required to be preserved by implementations.
    For each positive-semidefinite residue with eigenvalues ``r``, numerical
    rank retains ``r[k] > 2e-11 * max(1, max(abs(r)))``.  A resulting pole
    group whose summed-residue trace is at most ``2e-11`` is numerically null
    and omitted.
    """
    return hybridization_poles, hybridization_residues
```

### Step 5

05_solve_exact_feedback_impurity

Goal
----
Realize the feedback hybridization exactly and solve the resulting interacting impurity in a fixed particle sector.

```python
def solve_exact_feedback_impurity(
    impurity_energy: "np.ndarray",
    hybridization_poles: "np.ndarray",
    hybridization_residues: "np.ndarray",
    interaction_u: float,
    particle_number: int,
) -> "tuple[float, np.ndarray, np.ndarray]":
    """Return the second ground energy and augmented Lehmann spectrum.

    Parameters
    ----------
    impurity_energy : np.ndarray
        Finite Hermitian array of shape ``(2, 2)``.
    hybridization_poles : np.ndarray
        Finite real poles of shape ``(R,)`` for an unreduced causal
        hybridization.
    hybridization_residues : np.ndarray
        Hermitian positive-semidefinite residues of shape ``(R, 2, 2)``.
        Equivalent rank decompositions at a degenerate pole describe the same
        bath.
    interaction_u : float
        Finite real coefficient of the impurity interaction ``n_0 n_1``.
    particle_number : int
        Canonical reference sector of the realized finite model, with a unique
        lowest state and both adjacent sectors available.

    Returns
    -------
    tuple[float, np.ndarray, np.ndarray]
        The second canonical ground energy, real poles of shape ``(P,)``, and
        Hermitian positive-semidefinite residues of shape ``(P, 4, 4)`` for the
        complete doubled impurity propagator.  Components are ordered as the
        two physical channels followed by their two auxiliary channels.

    Raises
    ------
    ValueError
        If dimensions, finiteness, Hermiticity, positivity, reality, particle
        sector, or reference-state uniqueness violate the contract.

    Notes
    -----
    Input arrays are not required to be preserved by implementations.
    For each positive-semidefinite hybridization residue with eigenvalues
    ``r``, numerical rank retains
    ``r[k] > 2e-11 * max(1, max(abs(r)))``.  Output Lehmann pole groups with
    span at most ``1e-10`` are coalesced, and a group whose summed-residue
    trace is at most ``1e-12`` is numerically null and omitted.
    """
    return ground_energy, poles, residues
```

### Step 6

06_certify_regularized_matrix_response

Goal
----
Recover and certify the regularized low-energy matrix response of the second interacting solution.

```python
def certify_regularized_matrix_response(
    augmented_poles: "np.ndarray",
    augmented_residues: "np.ndarray",
    regulator: float,
) -> "tuple[np.ndarray, np.ndarray]":
    """Return the regularized response matrix and its spectral certificate.

    Parameters
    ----------
    augmented_poles : np.ndarray
        Finite real poles of shape ``(P,)`` for the second complete doubled
        propagator.
    augmented_residues : np.ndarray
        Hermitian positive-semidefinite residues of shape ``(P, 2*N, 2*N)``,
        ordered with physical channels first and matching auxiliary channels
        second.
    regulator : float
        Positive finite energy regulator used by the source-defined
        low-energy response.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        The Hermitian regularized response matrix of shape ``(N, N)`` and a
        five-entry real certificate containing, in order, the traces of the
        zeroth and first dynamical self-energy moments, the minimum residue
        eigenvalue, and the minimum and maximum response eigenvalues.  The
        minimum residue eigenvalue is zero for an empty dynamical spectrum.

    Raises
    ------
    ValueError
        If ``regulator`` is not positive and finite, if the augmented spectrum
        violates the preceding self-energy contract, or if the recovered
        response is nonfinite or non-Hermitian.

    Notes
    -----
    Input arrays are not required to be preserved by implementations.
    """
    return response_matrix, certificate
```

### Step 7

07_bounded_matrix_two_solve_weight

Goal
----
Compose the bounded two-solve matrix feedback calculation into one deterministic projected quasiparticle weight.

```python
def bounded_matrix_two_solve_weight(
    impurity_energy: "np.ndarray",
    bath_energies: "np.ndarray",
    bath_couplings: "np.ndarray",
    interaction_u: float,
    first_particle_number: int,
    second_particle_number: int,
    regulator: float,
    probe: "np.ndarray",
) -> float:
    """Return the rounded probe projection after one exact feedback update.

    Parameters
    ----------
    impurity_energy : np.ndarray
        Finite Hermitian impurity matrix of shape ``(2, 2)``.
    bath_energies : np.ndarray
        Finite Hermitian bath blocks of shape ``(M, 2, 2)``.
    bath_couplings : np.ndarray
        Finite coupling blocks of matching shape.
    interaction_u : float
        Finite real coefficient of the impurity interaction ``n_0 n_1``.
    first_particle_number : int
        Canonical particle number for the initial interacting solve.
    second_particle_number : int
        Canonical particle number for the exact feedback solve.
    regulator : float
        Positive finite energy regulator for the second low-energy response.
    probe : np.ndarray
        Finite normalized complex vector of shape ``(2,)``.

    Returns
    -------
    float
        The real scalar ``probe^H Z probe`` from the second regularized response
        matrix, rounded once to ten digits after completing the two-solve
        calculation.

    Raises
    ------
    ValueError
        If the probe is not finite and normalized, or if an upstream contract
        required by the supplied inputs is violated.

    Notes
    -----
    Input arrays are not required to be preserved by implementations.
    """
    return projected_weight
```
