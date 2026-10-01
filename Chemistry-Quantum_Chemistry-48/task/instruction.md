# Chemistry-Quantum_Chemistry-48

## Background

Each determinant in the expansion carries its own orbital frame. The fixture starts from orbitals that are orthonormal within each determinant, while no orthogonality is imposed between different determinants; EIDOS itself permits arbitrary linearly independent orbitals within a determinant. Direct orbital coefficients are useful in this setting because, once all creation operators except one are held fixed, the many-electron state is linear in the coefficients of the remaining orbital.

Removing one orbital leaves an \((n-1)\)-electron hole determinant. The missing orbital can be reinserted in any of the \(m\) spin-orbital directions, so the coefficients from all determinants form a vector of length \(N_Dm\). Its Hamiltonian expectation and norm are quadratic forms, and minimizing their quotient gives a generalized eigenvalue problem.

The corresponding norm matrix has an exact kernel. Reinserting an orbital already occupied by its hole state vanishes by antisymmetry, accounting for \(N_D(n-1)\) redundant directions. Those directions are removed before whitening; treating them as small positive eigenvalues would enlarge the variational space numerically.

Before the update, the orbital labels carry a separate gauge freedom. Left multiplication of a determinant's orbital rows by a matrix in \(SL(n)\) leaves that determinant unchanged while altering the direction presented as row zero. The supplied candidates are different choices of this gauge. Their energies are close enough that a wrong fermionic sign, two-electron factor, frame orientation, or null-space treatment changes the ordering.

## Problem

EIDOS treats a nonorthogonal determinant expansion as linear in one exposed orbital per determinant, so its energy is quadratic; that orbital nevertheless depends on a state-preserving electron-label gauge. In the equal-weight three-determinant fixture below ($m=8,n=4$), select the lowest-energy gauge among ten candidates; positions are zero-based and the returned gauge index is one-based.

Using the identified primary article and the archived `cqsl/detopt-v0.22` release, briefly justify the exposed-orbital reduction and describe how the release chooses and mixes the orbital to optimize before constructing the effective matrices.

Generate the input by running this NumPy fixture in order and keeping its QR column signs.

```python
import numpy as np

rng=np.random.default_rng(26082026)
m,n,n_det=8,4,3
A=rng.normal(size=(m,m))
one_body=.11*(A+A.T)+np.diag(np.linspace(-1.35,1.10,m))
P=[(p,q) for p in range(m) for q in range(p+1,m)]
B=rng.normal(size=(len(P),len(P)))
G=.035*(B+B.T)+np.diag(np.linspace(.22,.58,len(P)))
antisym_two_body=np.zeros((m,m,m,m))
for a,(p,q) in enumerate(P):
    for b,(r,s) in enumerate(P):
        value=G[a,b]
        antisym_two_body[p,q,r,s],antisym_two_body[q,p,r,s],antisym_two_body[p,q,s,r],antisym_two_body[q,p,s,r]=value,-value,-value,value
q0,_=np.linalg.qr(rng.normal(size=(m,n)))
q1,_=np.linalg.qr(q0+.018*rng.normal(size=(m,n)))
q2,_=np.linalg.qr(rng.normal(size=(m,n)))
determinant_orbitals=np.stack([q0.T,q1.T,q2.T])

def frame(k):
    r=np.random.default_rng(48151623+k)
    z=10.0**r.uniform(-1.2,.25)
    out=[]
    for _ in range(n_det):
        M=np.eye(n)
        for _ in range(7):
            i,j=r.choice(n,size=2,replace=False)
            S=np.eye(n)
            S[i,j]=r.normal(scale=z)
            M=S@M
        out.append(M)
    return np.stack(out)

mixing_candidates=np.stack([
    frame(k)
    for k in (402,1895,1839,1009,3817,3383,4790,3572,1958,1503)
])
close_shear=np.eye(n)
close_shear[3,0]=.23
mixing_candidates[4,2]=close_shear@mixing_candidates[4,2]
nuclear_energy=1.70
metric_cutoff=1e-13
```

Use \(\hat H=E_{\rm nuc}+\sum_{pq}h_{pq}c_p^\dagger c_q+\tfrac14\sum_{pqrs}\bar v_{pqrs}c_p^\dagger c_q^\dagger c_s c_r\), taking `antisym_two_body[p,q,r,s]` as \(\bar v_{pqrs}\), and sort every fixed-particle basis by occupation bitstring. For each gauge, left-multiply the orbital rows, remove transformed row zero, form the hole-state Hamiltonian/norm pencil, and take its lowest positive-metric variational root. Remove the \(N_D(n-1)\) structural zero modes without jitter and retain \(s>(10^{-13})s_{\max}\). Rank in supplied order, breaking only exact ties by the earlier candidate, and return the winning one-based index. In the audit, give sector and invariance checks, metric ranks and ratios, energy ledger and margin, and winners after transposing gauges, removing the last row, or changing the two-electron prefactor to \(1/2\).

Output Format Requirements:
Emit <final_answer> immediately, then <reasoning>. Do not write a long derivation before the tags.
You must emit exactly one finite decimal inside <final_answer>...</final_answer>, even if the value is approximate or you are unsure.
Rules:
- The tags are required. Do not omit them or leave them empty.
- The tagged value must be a finite decimal. Not NaN, not Inf, not a fraction, not a vector, and not prose.
- Put only that one number between the tags. No units, words, or extra lines.
- Keep <reasoning> concise, but include the requested source conclusions and complete audit: the sector and invariance checks, all ten metric ratios and updated energies, the energy margin, and the three counterfactual winners.
- Do not include full matrices, state vectors, code, or iterative traces.

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

Enumerate the canonical occupation bitstrings for one fixed electron-number sector.

Goal
----
Enumerate the canonical occupation bitstrings for one fixed electron-number sector.

```python
def enumerate_fock_states(n_orbitals: int, n_electrons: int) -> np.ndarray:
    """Enumerate a fixed-particle Fock sector.

    Parameters
    ----------
    n_orbitals : int
        Number of spin orbitals, between 1 and 62.
    n_electrons : int
        Particle count between zero and n_orbitals.

    Returns
    -------
    np.ndarray
        Increasing int64 occupation bitstrings.
    """
    return result
```

### Step 2

expand_slater_determinant

Goal
----
Expand an ordered set of orbital rows into amplitudes over a supplied occupation basis.

```python
def expand_slater_determinant(
    orbital_rows: np.ndarray,
    basis_states: np.ndarray,
) -> np.ndarray:
    """Expand an ordered orbital product in an occupation basis.

    Parameters
    ----------
    orbital_rows : np.ndarray
        Real orbital coefficients with shape (n_electrons, n_orbitals).
    basis_states : np.ndarray
        Distinct bitstrings in the matching particle sector.

    Returns
    -------
    np.ndarray
        Determinant amplitudes in basis_states order.
    """
    return result
```

### Step 3

build_active_space_hamiltonian

Goal
----
Construct the electronic Hamiltonian in a fixed occupation sector.

```python
def build_active_space_hamiltonian(
    one_body: np.ndarray,
    antisym_two_body: np.ndarray,
    basis_states: np.ndarray,
    nuclear_energy: float,
) -> np.ndarray:
    """Build a fixed-sector spin-orbital Hamiltonian.

    Parameters
    ----------
    one_body : np.ndarray
        Symmetric one-electron matrix with shape (m, m).
    antisym_two_body : np.ndarray
        Antisymmetrized integrals v[p,q,r,s] with shape (m,m,m,m).
    basis_states : np.ndarray
        Distinct bitstrings sharing one electron count.
    nuclear_energy : float
        Scalar additive nuclear energy.

    Returns
    -------
    np.ndarray
        Symmetric sector Hamiltonian in basis_states order.
    """
    return result
```

### Step 4

transform_orbital_frames

Goal
----
Apply one determinant-one electron-label transformation to every determinant.

```python
def transform_orbital_frames(
    determinant_orbitals: np.ndarray,
    mixing_matrices: np.ndarray,
) -> np.ndarray:
    """Transform determinant orbital rows by determinant-one matrices.

    Parameters
    ----------
    determinant_orbitals : np.ndarray
        Orbital rows with shape
        (n_det, n_electrons, n_orbitals).
    mixing_matrices : np.ndarray
        One SL(n) matrix per determinant.

    Returns
    -------
    np.ndarray
        Left-transformed orbital rows with the input shape.
    """
    return result
```

### Step 5

assemble_hole_lift

Goal
----
Lift every hole determinant through all creation operators into the electron sector.

```python
def assemble_hole_lift(
    hole_amplitudes: np.ndarray,
    hole_basis_states: np.ndarray,
    electron_basis_states: np.ndarray,
    n_orbitals: int,
) -> np.ndarray:
    """Assemble creation-lift columns in flattened determinant-orbital order.

    Parameters
    ----------
    hole_amplitudes : np.ndarray
        Hole-state amplitudes with shape
        (n_det, n_hole_states).
    hole_basis_states : np.ndarray
        Bitstrings for the hole sector.
    electron_basis_states : np.ndarray
        Bitstrings for the sector with one additional electron.
    n_orbitals : int
        Number of spin orbitals.

    Returns
    -------
    np.ndarray
        Lift matrix with columns ordered by
        I*n_orbitals+mu.
    """
    return result
```

### Step 6

form_eidos_pencil

Goal
----
Form the effective Hamiltonian and norm quadratic forms of the lifted variational state.

```python
def form_eidos_pencil(
    lift_columns: np.ndarray,
    sector_hamiltonian: np.ndarray,
) -> np.ndarray:
    """Form the two effective matrices for one orbital update.

    Parameters
    ----------
    lift_columns : np.ndarray
        Many-electron lift matrix W.
    sector_hamiltonian : np.ndarray
        Symmetric Hamiltonian acting on W's row space.

    Returns
    -------
    np.ndarray
        Array [W.T@H@W, W.T@W] with shape
        (2, n_columns, n_columns).
    """
    return result
```

### Step 7

solve_eidos_update

Goal
----
Solve the effective generalized eigenproblem after relative-cutoff whitening of its positive metric subspace.

```python
def solve_eidos_update(
    matrix_pencil: np.ndarray,
    relative_cutoff: float,
) -> np.ndarray:
    """Solve a singular-metric variational update.

    Parameters
    ----------
    matrix_pencil : np.ndarray
        Effective [Hamiltonian, metric] array with
        shape (2, n, n).
    relative_cutoff : float
        Positive relative metric-eigenvalue cutoff
        below one.

    Returns
    -------
    np.ndarray
        [energy, retained_rank, residual_norm,
        coefficient_vector...].
    """
    return result
```

### Step 8

select_orbital_update

Goal
----
Evaluate every determinant-one orbital frame through the seven preceding functions and return the frame with the lowest one-step variational energy.

```python
def select_orbital_update(
    one_body: np.ndarray,
    antisym_two_body: np.ndarray,
    determinant_orbitals: np.ndarray,
    mixing_candidates: np.ndarray,
    nuclear_energy: float,
    metric_cutoff: float,
) -> int:
    """Select a determinant-one frame for one exact orbital update.

    Parameters
    ----------
    one_body : np.ndarray
        Symmetric spin-orbital one-electron matrix.
    antisym_two_body : np.ndarray
        Antisymmetrized two-electron tensor v[p,q,r,s].
    determinant_orbitals : np.ndarray
        Orbital rows with shape (n_det, n_electrons, n_orbitals).
    mixing_candidates : np.ndarray
        Candidate SL(n) frames with shape
        (n_candidates, n_det, n_electrons, n_electrons).
    nuclear_energy : float
        Scalar additive nuclear energy.
    metric_cutoff : float
        Relative positive-metric eigenvalue cutoff.

    Returns
    -------
    int
        One-based index of the lowest-energy candidate.
    """
    return result
```
