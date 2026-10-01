# Chemistry-Quantum_Chemistry-59

## Background

Neural quantum states use nonlinear models to represent fermionic wavefunctions. In molecular Fock space, stochastic optimization can struggle when important configurations occupy sparse, separated regions of the probability distribution.

Deterministic selected-configuration approaches replace part of that sampling with exact evaluation on finite determinant sets. They can compare alternative energy objectives while retaining a compact representation of the many-electron state.

Their accuracy depends on stable support selection, consistent Hamiltonian evaluation, and controlled recovery of correlation beyond the current support. Diagnostics that separate imperfect optimization inside the selected space from contributions outside it help identify which approximation limits a calculation.

## Problem

A deterministic neural-quantum-state snapshot is available on a finite determinant archive; execute the NumPy block once and quantify what signed percentage of its complete second-order correction comes from the unsaturated residual inside the selected variational support rather than from the screened external space, using one-based archive positions only in the requested audit.

```python
import itertools
import numpy as np

rng=np.random.default_rng(26090410)
n_orb,n_alpha,n_beta,n_target=5,2,2,28
n_so=2*n_orb

h_spatial=rng.normal(scale=.18,size=(n_orb,n_orb))
h_spatial=(h_spatial+h_spatial.T)/2
h_spatial[np.diag_indices(n_orb)]-=np.linspace(1.55,.25,n_orb)
h1=np.zeros((n_so,n_so),dtype=float)
for p in range(n_so):
    for q in range(n_so):
        if p%2==q%2:
            h1[p,q]=h_spatial[p//2,q//2]

pairs=[(p,q) for p in range(n_so) for q in range(p+1,n_so)]
pair_block=rng.normal(scale=.055,size=(len(pairs),len(pairs)))
pair_block=(pair_block+pair_block.T)/2
spin_count=np.array([(p&1)+(q&1) for p,q in pairs])
pair_block*=spin_count[:,None]==spin_count[None,:]
g2=np.zeros((n_so,n_so,n_so,n_so),dtype=float)
for a,(p,q) in enumerate(pairs):
    for b,(r,s) in enumerate(pairs):
        value=pair_block[a,b]
        g2[p,q,r,s]=value
        g2[q,p,r,s]=-value
        g2[p,q,s,r]=-value
        g2[q,p,s,r]=value

alpha_masks=[sum(1<<i for i in occ)
             for occ in itertools.combinations(range(n_orb),n_alpha)]
beta_masks=[sum(1<<i for i in occ)
            for occ in itertools.combinations(range(n_orb),n_beta)]
determinants=np.array(
    list(itertools.product(alpha_masks,beta_masks)),
    dtype=np.int64
)
determinants=determinants[
    rng.permutation(len(determinants))[:n_target]
]

orbital_diag=np.diag(h_spatial)
one_body_diag=np.empty(n_target,dtype=float)
for k,(alpha,beta) in enumerate(determinants):
    occupied=[
        i for i in range(n_orb)
        if (int(alpha)>>i)&1
    ]
    occupied += [
        i for i in range(n_orb)
        if (int(beta)>>i)&1
    ]
    one_body_diag[k]=np.sum(orbital_diag[occupied])

logabs=-1.15*(one_body_diag-np.min(one_body_diag))
logabs+=rng.normal(scale=.22,size=n_target)
signs=rng.choice(np.array([-1.,1.]),size=n_target)
signs[np.argmax(logabs)]=1.

mass_fraction=.86
min_k,max_k=6,11
eps1=.012
e_nuc=1.3275
```

Each determinant row stores alpha and beta spatial-orbital bitmasks, while `h1` and the real antisymmetrized `g2` tensor use the source-defined spin-orbital ordering. Use the primary article and pinned implementation to reconstruct their deterministic configuration-selection, Hamiltonian, screening, energy-diagnostic, and internal/external Epstein-Nesbet workflow for the supplied snapshot. Apply `mass_fraction`, `min_k`, and `max_k` to the complete archive; resolve exact probability ties by original archive order, a benchmark convention supplied here rather than inferred from the repository. After the perturbative determinant set is selected, form each external PT2 residual from the complete `H_PV` row over the selected support; do not reapply `eps1` to individual `H_ai*c_i` contributions. Return `100*delta_internal/(delta_internal+delta_external)` as one finite decimal, preserving the signs of both corrections and without clipping or taking absolute values.

In `<reasoning>`, cite the sources; briefly report the article's Li2O convergence result, and reconcile its main-text H2O and N2 dissociation accuracy statements with the Supporting Information energy tables at the same support sizes.Also identify the source conventions governing determinant excitation phases across spin channels and same-spin double excitations, the ordering and deduplication of deferred external determinants before indexed couplings are constructed, and the normalization and external-block treatment used for the proxy energy. Then compactly report the trace and Hermiticity residual of the complete-archive Hamiltonian, the selected one-based support labels, the cumulative masses before and after the last selected determinant, the screened external labels, the smallest retained and largest rejected screening scores, the five-energy ledger E_var, E_asym, E_proxy, E_diag and delta_opt = E_var - E_diag, both PT2 channels, their sum, and the corrected total energy. Briefly explain why the signed share can lie outside the interval from zero to one hundred and why nuclear repulsion is excluded from the electronic PT2 denominators. Report floating values to at least ten decimal places; the requested audit vectors are permitted by the brevity instruction and must not be replaced by input tensors.

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

apply_fermion_string

Goal
----
Apply an ordered fermionic operator string in factorized spin channels.

```python
def apply_fermion_string(
    determinant: 'np.ndarray',
    annihilators: 'np.ndarray',
    creators: 'np.ndarray',
    n_orb: int,
) -> 'np.ndarray':
    """Apply annihilation and creation operators in chronological action order.

    A determinant stores separate alpha and beta spatial-orbital masks. Input
    operator labels use alpha ``2*i`` and beta ``2*i+1`` only as an integral-
    tensor lookup convention. Fermionic parity is accumulated independently
    inside each spin mask: occupied orbitals of the opposite spin never enter
    a parity count. The entries of ``annihilators`` act first from left to
    right, followed by ``creators`` from left to right.

    Returns
    -------
    np.ndarray
        ``[alpha_mask, beta_mask, phase]``. A Pauli-forbidden action returns
        ``[-1, -1, 0]``.

    Raises
    ------
    ValueError
        If masks, orbital indices, dimensions or dtypes are invalid.
    """
    return result
```

### Step 2

build_selected_hamiltonian

Goal
----
Construct a determinant-space electronic Hamiltonian.

```python
def build_selected_hamiltonian(
    determinants: 'np.ndarray',
    h1: 'np.ndarray',
    g2: 'np.ndarray',
    n_orb: int,
) -> 'np.ndarray':
    """Build the electronic Hamiltonian on the supplied determinant archive.

    ``g2[p,q,r,s]`` contains real antisymmetrized spin-orbital integrals. Apply
    the source Slater--Condon cases for diagonal, single and double
    excitations. Excitation phase must be obtained with the factorized-spin
    operator action from step 1; excitations above degree two have zero matrix
    element. Retain the supplied determinant order.

    Returns
    -------
    np.ndarray
        Real symmetric Hamiltonian with shape ``(n,n)``.

        Raises
    ------
    ValueError
        If determinant or integral shapes, symmetries or values are invalid, or
        if the same determinant appears more than once.
    """
    return result
```

### Step 3

normalized_configuration_probabilities

Goal
----
Convert signed log-amplitudes into stable normalized probabilities.

```python
def normalized_configuration_probabilities(
    signs: 'np.ndarray',
    logabs: 'np.ndarray',
) -> 'np.ndarray':
    """Return normalized squared amplitudes without losing dynamic range.

    Returns
    -------
    np.ndarray
        One probability per configuration, in input order.

    Raises
    ------
    ValueError
        If arrays are empty, incompatible, non-finite or contain invalid signs.
    """
    return result
```

### Step 4

select_cumulative_support

Goal
----
Select the minimal probability-ranked support reaching a mass target.

```python
def select_cumulative_support(
    probabilities: 'np.ndarray',
    fraction: float,
    min_k: int,
    max_k: int,
) -> 'np.ndarray':
    """Apply cumulative-mass selection with deterministic input-order ties.

    Returns
    -------
    np.ndarray
        Selected zero-based indices in decreasing-probability order.

    Raises
    ------
    ValueError
        If probabilities or selector bounds are invalid.
    """
    return result
```

### Step 5

screen_dynamic_perturbative_space

Goal
----
Screen and canonically order the dynamic perturbative determinant space.

```python
def screen_dynamic_perturbative_space(
    hamiltonian: 'np.ndarray',
    amplitudes: 'np.ndarray',
    determinants: 'np.ndarray',
    variational_indices: 'np.ndarray',
    eps1: float,
) -> 'np.ndarray':
    """Select external rows passing the amplitude-weighted heat-bath boundary.

    The selected amplitudes are normalized before screening. Archive rows
    outside the variational set are retained when their largest
    ``abs(H[a,i] * psi[i])`` is at least ``eps1``. Return retained archive
    indices in lexicographic ``(alpha_mask, beta_mask)`` order.

    Returns
    -------
    np.ndarray
        Zero-based retained archive indices in canonical determinant order.

    Raises
    ------
    ValueError
        If the matrix, amplitudes, determinants, indices or threshold are invalid.
    """
    return result
```

### Step 6

deterministic_energy_diagnostics

Goal
----
Evaluate deterministic variational, asymmetric, proxy and diagonal energies.

```python
def deterministic_energy_diagnostics(
    hamiltonian: 'np.ndarray',
    amplitudes: 'np.ndarray',
    variational_indices: 'np.ndarray',
    perturbative_indices: 'np.ndarray',
) -> 'np.ndarray':
    """Return the five-energy diagnostic ledger for selected configuration sets.

    The ledger is ``[E_var, E_asym, E_proxy, E_diag, delta_opt]``. ``E_var``
    and ``E_asym`` use normalization on the variational support. ``E_proxy``
    uses normalization on the concatenated target and replaces the external-
    external block by its diagonal. ``E_diag`` is the lowest eigenvalue of the
    variational block and ``delta_opt = E_var - E_diag``.

    Returns
    -------
    np.ndarray
        Five finite diagnostic values in the stated order.

    Raises
    ------
    ValueError
        If shapes, index sets, symmetry, values or normalization are invalid.
    """
    return result
```

### Step 7

decomposed_epstein_nesbet_pt2

Goal
----
Decompose the Epstein–Nesbet correction into internal and external terms.

```python
def decomposed_epstein_nesbet_pt2(
    hamiltonian: 'np.ndarray',
    amplitudes: 'np.ndarray',
    variational_indices: 'np.ndarray',
    perturbative_indices: 'np.ndarray',
    reference_total_energy: float,
    nuclear_repulsion: float,
) -> 'np.ndarray':
    """Evaluate internal-residual and screened-external PT2 corrections.

    The selected amplitudes are normalized on the variational support. For
    every supplied perturbative index, form the external residual from the
    complete Hamiltonian row over the selected support; do not apply another
    elementwise screening threshold. Convert the total reference energy to
    its electronic value before forming either Epstein-Nesbet denominator.

    Returns
    -------
    np.ndarray
        ``[delta_internal, delta_external, delta_total, n_external]``.

    Raises
    ------
    ValueError
        If inputs are incompatible, non-finite, non-Hermitian or singular.
    """
    return result
```

### Step 8

nqs_internal_pt2_share

Goal
----
Orchestrate deterministic support construction and PT2 error attribution.

```python
def nqs_internal_pt2_share(
    determinants: 'np.ndarray' = None,
    h1: 'np.ndarray' = None,
    g2: 'np.ndarray' = None,
    signs: 'np.ndarray' = None,
    logabs: 'np.ndarray' = None,
    mass_fraction: float = None,
    eps1: float = None,
    min_k: int = None,
    max_k: int = None,
    nuclear_repulsion: float = None,
) -> float:
    """Run steps 1--7 and return the internal share of total PT2 in percent.

    With no arguments, run the benchmark fixture from the problem statement.
    Otherwise every argument is required.

    Returns
    -------
    float
        ``100 * delta_internal / (delta_internal + delta_external)``.

    Raises
    ------
    ValueError
        If an earlier step rejects an input or the total correction is zero.
    """
    return result
```
