# Chemistry-Quantum_Chemistry-38

## Problem

Environmental screening and incomplete determinant sampling can both bias intermolecular interaction energies. Using molecular constrained-RPA downfolding with its accompanying one-body and scalar energy corrections, together with a size-consistent construction of fragment and dimer determinant spaces from sampled configurations, determine the interaction energy \(E_{\mathrm{int}}=E_{AB}-E_A-E_B\) of the synthetic electronic system specified below.

Use eight orthonormal, fixed localized spatial orbitals indexed \(0,\ldots,7\), with energies in hartree, the following Coulomb factors in \(\mathrm{hartree}^{1/2}\), chemists’ integrals \(v_{pq,rs}=(pq|rs)=\sum_{\ell=1}^{2}L^{(\ell)}_{pq}L^{(\ell)}_{rs}\), and the following subsystem, reference-occupation, and active-space definitions:
\[
M=
\begin{pmatrix}
0.50&0.16&0.30&0.12\\
0.16&0.36&0.09&0.14\\
0.30&0.09&0.30&0.11\\
0.12&0.14&0.11&0.42
\end{pmatrix},
\qquad
N=
\begin{pmatrix}
0.20&-0.08&0.22&0.09\\
-0.08&0.28&0.07&-0.10\\
0.22&0.07&0.24&0.13\\
0.09&-0.10&0.13&0.32
\end{pmatrix},
\]
\[
R=\operatorname{diag}(0.12,0.08,0.05,0.04),
\qquad
S=0.03
\begin{pmatrix}1\\-1\\2\\1\end{pmatrix}
\begin{pmatrix}2&1&-1&1\end{pmatrix},
\]
\[
L^{(1)}=
\begin{pmatrix}M&R\\R^{T}&0.9M\end{pmatrix},
\qquad
L^{(2)}=
\begin{pmatrix}N&S\\S^{T}&1.1N\end{pmatrix},
\qquad
\epsilon=(-0.80,-0.55,-0.35,1.10,-0.75,-0.48,-0.29,1.25),
\]
\[
\begin{array}{c|c|c|c}
X&\text{orbital set }\mathcal O_X&
\text{doubly occupied reference orbitals }\mathcal I_X&
\text{active orbitals }\mathcal A_X\\ \hline
A&\{0,1,2,3\}&\{0\}&\{0,1,2\}\\
B&\{4,5,6,7\}&\{4\}&\{4,5,6\}\\
AB&\{0,1,2,3,4,5,6,7\}&\{0,4\}&\{0,1,2,4,5,6\}
\end{array}
\]
For each subsystem \(X\), restrict all four integral indices to \(\mathcal O_X\), take \(\mathcal O_X\setminus\mathcal A_X\) as its environment, set nuclear repulsion to zero, and define its one-electron Hamiltonian by
\[
h^{X}_{pq}
=\epsilon_p\delta_{pq}
-\sum_{i\in\mathcal I_X}
\left[2v_{pq,ii}-v_{pi,iq}\right],
\qquad p,q\in\mathcal O_X .
\]
The fixed dimer samples are the following occupied spatial-orbital sets \((I_\alpha,I_\beta)\), with fragment electron numbers \(N_A=N_B=2\), no additional sampling, and determinant creation operators ordered by ascending active \(\alpha\) orbitals followed by ascending active \(\beta\) orbitals:
\[
\begin{array}{c|c}
I_\alpha&I_\beta\\ \hline
\{0,4\}&\{0,4\}\\
\{0,4\}&\{1,4\}\\
\{0,4\}&\{0,5\}\\
\{0,4\}&\{1,5\}\\
\{2,4\}&\{2,4\}\\
\{0,6\}&\{0,6\}\\
\{0,4\}&\{0,1\}\\
\{0,4\}&\{4,5\}
\end{array}
\]

Use full static cRPA screening at \(\omega=0\) with vanishing broadening, excluding active-only particle–hole transitions from the screening polarization, and fix the direct-Coulomb RPA convention on spin-conserving spin-orbital transitions \(i\sigma\rightarrow a\sigma\) by
\[
D_{ia\sigma,jb\tau}
=(\epsilon_a-\epsilon_i)\delta_{ij}\delta_{ab}\delta_{\sigma\tau},
\qquad
K_{ia\sigma,jb\tau}=v_{ia,jb},
\qquad
\mathsf A=D+K,\quad \mathsf B=K,
\]
\[
E_c^{\mathrm{RPA}}
=\frac12\left(\sum_{\nu}\Omega_\nu-\operatorname{tr}\mathsf A\right),
\]
where both spin projections are included, \(\Omega_\nu\) are the positive RPA frequencies, and the supplied reference orbitals, occupations, and orbital energies remain fixed in every full-system and active-space RPA evaluation. Retrieve and apply the molecular downfolding prescription for the corrected one-electron terms and scalar environmental energy, and the size-consistent sampled-configuration prescription for spin completion and fragment/dimer configuration spaces, using all supplied samples. Define each total energy as the lowest eigenvalue of its resulting effective Hamiltonian including its scalar term, retaining the fragment spin-projection sectors induced by the completed samples and restricting the dimer to \(N_\alpha=N_\beta=2\), without additional determinant selection or replacement by full configuration interaction. Determine \(E_{\mathrm{int}}\) to six decimal places and justify the screening, energy bookkeeping, and configuration-space choices using the retrieved sources and decisive equations.

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

01_environmental_response.py

Goal
----
Compute the constrained environmental static density response.

```python
def environmental_response(v: "np.ndarray", eps: "np.ndarray", occupied: "np.ndarray", active: "np.ndarray") -> tuple["np.ndarray", "np.ndarray"]:
    """Compute the constrained environmental static density response.
    
    Parameters and conventions
    --------------------------
    v is a real finite chemists' tensor of shape (n,n,n,n), n>=1,
    with p/q, r/s and pair-exchange symmetry (atol=1e-12, rtol=0).
    eps is a real finite vector of length n. occupied is a vector of
    distinct integer indices in [0,n), each doubly occupied; empty is
    allowed. Every occupied-to-virtual gap must be positive. active is
    a nonempty vector of distinct integer indices in [0,n); its order
    defines the local active orbital order. Booleans are not indices.
    The reference orbitals, energies and occupations remain fixed.
    Complex dtype numerical inputs are rejected even if their imaginary
    parts are zero. Raise ValueError for violations, failed numerical
    solves, or nonfinite computed results.
    
    Return retained spatial transition labels (i,a) in occupied input
    order and increasing virtual label within each occupied index.
    The response is the retarded, zero-frequency, zero-broadening physical
    density response, including BOTH spin copies, expressed in this
    spatial transition basis; coupling entries carry no added spin factor.
    An empty constrained transition space returns shapes (0,2) and (0,0).
    Only the constrained RPA problem is solved here; it must be strictly
    stable. In the supplied convention this means that the symmetric
    charge-channel RPA squared-frequency matrix is positive definite.
    
    Returns
    -------
    tuple (pairs, response): int64 transition labels of shape (m,2) and the real spin-summed static response of shape (m,m).
    """
    return pairs, response
```

### Step 2

02_screened_interaction.py

Goal
----
Construct the full static-cRPA active interaction tensor.

```python
def screened_interaction(v: "np.ndarray", eps: "np.ndarray", occupied: "np.ndarray", active: "np.ndarray") -> "np.ndarray":
    """Construct the full static-cRPA active interaction tensor.
    
    Parameters and conventions
    --------------------------
    v is a real finite chemists' tensor of shape (n,n,n,n), n>=1,
    with p/q, r/s and pair-exchange symmetry (atol=1e-12, rtol=0).
    eps is a real finite vector of length n. occupied is a vector of
    distinct integer indices in [0,n), each doubly occupied; empty is
    allowed. Every occupied-to-virtual gap must be positive. active is
    a nonempty vector of distinct integer indices in [0,n); its order
    defines the local active orbital order. Booleans are not indices.
    The reference orbitals, energies and occupations remain fixed.
    Complex dtype numerical inputs are rejected even if their imaginary
    parts are zero. Raise ValueError for violations, failed numerical
    solves, or nonfinite computed results.
    
    Use environmental_response and its spatial, spin-summed convention.
    Return the static screened chemists' interaction on all active pair
    indices. The constrained response must be strictly stable. Empty
    constrained transition spaces and a fully active system are valid.
    
    Returns
    -------
    real array w of shape (a,a,a,a), in active input order.
    """
    return w
```

### Step 3

03_corrected_one_body.py

Goal
----
Construct the corrected one-electron part of the molecular effective Hamiltonian.

```python
def corrected_one_body(v: "np.ndarray", eps: "np.ndarray", occupied: "np.ndarray", active: "np.ndarray", w: "np.ndarray") -> "np.ndarray":
    """Construct the corrected one-electron part of the molecular effective Hamiltonian.
    
    Parameters and conventions
    --------------------------
    v is a real finite chemists' tensor of shape (n,n,n,n), n>=1,
    with p/q, r/s and pair-exchange symmetry (atol=1e-12, rtol=0).
    eps is a real finite vector of length n. occupied is a vector of
    distinct integer indices in [0,n), each doubly occupied; empty is
    allowed. Every occupied-to-virtual gap must be positive. active is
    a nonempty vector of distinct integer indices in [0,n); its order
    defines the local active orbital order. Booleans are not indices.
    The reference orbitals, energies and occupations remain fixed.
    Complex dtype numerical inputs are rejected even if their imaginary
    parts are zero. Raise ValueError for violations, failed numerical
    solves, or nonfinite computed results.
    
    w is the real finite screened chemists' tensor of shape (a,a,a,a),
    a=len(active), with the same three Coulomb symmetries and tolerance
    as v. It is already in active input order. Reject complex dtype w.
    Compute the one-body term of the retrieved molecular prescription.
    The bare reference h is defined by the problem statement. Occupied
    environment orbitals are supported. Do not recompute w or test that
    it equals a particular screening calculation. Nuclear repulsion is zero.
    
    Returns
    -------
    real array t_eff of shape (a,a), in active input order.
    """
    return t_eff
```

### Step 4

04_reference_offset.py

Goal
----
Evaluate the reference-energy contribution to the molecular scalar offset.

```python
def reference_offset(v: "np.ndarray", eps: "np.ndarray", occupied: "np.ndarray", active: "np.ndarray", w: "np.ndarray") -> float:
    """Evaluate the reference-energy contribution to the molecular scalar offset.
    
    Parameters and conventions
    --------------------------
    v is a real finite chemists' tensor of shape (n,n,n,n), n>=1,
    with p/q, r/s and pair-exchange symmetry (atol=1e-12, rtol=0).
    eps is a real finite vector of length n. occupied is a vector of
    distinct integer indices in [0,n), each doubly occupied; empty is
    allowed. Every occupied-to-virtual gap must be positive. active is
    a nonempty vector of distinct integer indices in [0,n); its order
    defines the local active orbital order. Booleans are not indices.
    The reference orbitals, energies and occupations remain fixed.
    Complex dtype numerical inputs are rejected even if their imaginary
    parts are zero. Raise ValueError for violations, failed numerical
    solves, or nonfinite computed results.
    
    w is the real finite screened chemists' tensor of shape (a,a,a,a),
    a=len(active), with the same three Coulomb symmetries and tolerance
    as v. It is already in active input order. Reject complex dtype w.
    Return the reference-energy part of the molecular scalar offset,
    excluding its RPA correlation part. Occupied environment orbitals
    are supported. The bare reference h is defined by the problem.
    Do not recompute w or test that it equals a particular screening
    calculation. Nuclear repulsion is zero.
    
    Returns
    -------
    native Python float: the reference-energy part of the scalar offset, in hartree.
    """
    return 0.0
```

### Step 5

05_correlation_offset.py

Goal
----
Evaluate the correlation-energy contribution to the molecular scalar offset.

```python
def correlation_offset(v: "np.ndarray", eps: "np.ndarray", occupied: "np.ndarray", active: "np.ndarray", w: "np.ndarray") -> float:
    """Evaluate the correlation-energy contribution to the molecular scalar offset.
    
    Parameters and conventions
    --------------------------
    v is a real finite chemists' tensor of shape (n,n,n,n), n>=1,
    with p/q, r/s and pair-exchange symmetry (atol=1e-12, rtol=0).
    eps is a real finite vector of length n. occupied is a vector of
    distinct integer indices in [0,n), each doubly occupied; empty is
    allowed. Every occupied-to-virtual gap must be positive. active is
    a nonempty vector of distinct integer indices in [0,n); its order
    defines the local active orbital order. Booleans are not indices.
    The reference orbitals, energies and occupations remain fixed.
    Complex dtype numerical inputs are rejected even if their imaginary
    parts are zero. Raise ValueError for violations, failed numerical
    solves, or nonfinite computed results.
    
    w is the real finite screened chemists' tensor of shape (a,a,a,a),
    a=len(active), with the same three Coulomb symmetries and tolerance
    as v. It is already in active input order. Reject complex dtype w.
    Return the correlation part of the molecular scalar offset under
    the supplied direct-Coulomb RPA convention. Its required full and
    active RPA problems must be strictly stable: the corresponding
    symmetric charge-channel squared-frequency matrices are positive
    definite. Empty transition spaces are valid. Do not recompute w or
    replace the supplied orbital energies by eigenvalues of t_eff.
    
    Returns
    -------
    native Python float: the correlation-energy part of the scalar offset, in hartree.
    """
    return 0.0
```

### Step 6

06_selected_spaces.py

Goal
----
Construct spin-completed, size-consistent fragment and dimer determinant spaces.

```python
def selected_spaces(n_a: int, n_b: int, n_e_a: int, samples: "np.ndarray") -> tuple["np.ndarray", "np.ndarray", "np.ndarray"]:
    """Construct the size-consistent selected configuration spaces.

    Parameters
    ----------
    n_a, n_b : int
        Positive counts of active spatial orbitals on A and B, sum<=10.
        Dimer spatial labels put A first, then B.
    n_e_a : int
        Neutral active electron count on A; the neutral B count is
        total sampled electron count minus n_e_a.
    samples : np.ndarray
        Raw alpha/beta masks in a nonempty (k,2) integer array, using
        bit p for spatial orbital p and common Nalpha,Nbeta.

    Apply the retrieved spin-completion and size-consistent
    fragment/dimer-space prescription. Retain the fragment spin sectors
    induced by the samples. The dimer retains the sampled total alpha
    and beta counts. Packed determinants use alpha_mask | (beta_mask
    << local_orbital_count), with alpha spin orbitals before beta.
    Sort each output by packed integer.

    Returns
    -------
    tuple (space_a, space_b, space_ab) of ascending unique np.int64 packed-determinant arrays using local orbital counts n_a, n_b and n_a+n_b.

    Raises
    ------
    ValueError
    If n_a or n_b is not a positive integer (booleans invalid),
    or n_a+n_b>10; if samples is not a nonempty integer (k,2) array
    of masks in [0,2**(n_a+n_b)) with common alpha and beta counts;
    if n_e_a is not an integer in [0,2*n_a], or the implied n_e_b is
    outside [0,2*n_b]; or if no spin-completed sample has n_e_a
    electrons on A. Duplicate sample rows are valid.
    """
    return space_a, space_b, space_ab
```

### Step 7

07_selected_energy.py

Goal
----
Evaluate a subsystem selected-space total energy including the molecular scalar offset.

```python
def selected_energy(v: "np.ndarray", eps: "np.ndarray", occupied: "np.ndarray", active: "np.ndarray", determinants: "np.ndarray") -> float:
    """Evaluate a complete downfolded subsystem total energy.

    Parameters
    ----------
    v, eps, occupied, active : np.ndarray
        Reference inputs: chemists' tensor, fixed orbital energies,
        doubly occupied reference indices and ordered active indices.
    determinants : np.ndarray
        Distinct packed determinants in the LOCAL active orbital order.
        Each has 2*len(intersection(occupied,active)) electrons.

    Use screened_interaction, corrected_one_body, reference_offset,
    and correlation_offset. Assemble the selected Hamiltonian in the
    supplied determinant order and return its lowest eigenvalue plus
    the two scalar contributions, each included once. Packed spin
    orbitals put all local alpha orbitals before all local beta orbitals.
    Do not enlarge the determinant space or change reference energies.

    Returns
    -------
    native Python float: the lowest selected-space eigenvalue plus the downfolded scalar offset.

    Raises
    ------
    ValueError
    If v or eps has complex dtype (even with zero imaginary parts),
    or cannot be converted to real finite numeric arrays;
    if eps is not a nonempty vector or v does not have shape (n,n,n,n),
    where n=len(eps); if v lacks p/q exchange, r/s exchange, or
    pair-exchange symmetry (absolute tolerance 1e-12, relative
    tolerance zero); if occupied is not a one-dimensional vector of
    distinct integer indices in [0,n) (an empty vector is allowed);
    if any occupied-to-virtual gap is nonpositive; or if a computed
    reference quantity is nonfinite.
    Also if active is not a nonempty one-dimensional vector of distinct
    integer indices in [0,n); if the full, constrained, or screened
    active-space RPA problem violates the RPA stability/finite
    result conditions stated below; or if a downfolded result is nonfinite.
    RPA stability means every eigenvalue of D^(1/2)(D+4K)D^(1/2) is positive;
    all RPA couplings must be finite and symmetric within atol=1e-12,
    rtol=0, and their eigendecompositions must succeed.
    For the selected-Hamiltonian assembly, let m=len(active), and let
    t_eff and W denote the downfolded one- and two-electron arrays.
    Raise if m is outside 1<=m<=8; if t_eff or W is not real and finite;
    if t_eff is not symmetric with shape (m,m); if W does not have
    shape (m,m,m,m) or lacks p/q, r/s, or pair-exchange symmetry
    (all symmetry checks use atol=1e-12, rtol=0); if determinants is
    not a nonempty vector of distinct integers in [0,2**(2*m)); if
    their total electron counts differ; or if the computed matrix
    contains a nonfinite entry.
    Different spin projections with the same total electron count
    are allowed, as is the vacuum determinant 0.
    Also if any determinant has an electron count other than twice
    the number of occupied indices included in active, if the final
    eigensolver fails, or if the final energy is nonfinite.
    """
    return 0.0
```

### Step 8

08_interaction_energy.py

Goal
----
Orchestrate the two-source screened, size-consistent selected-CI interaction energy.

```python
def interaction_energy(factors: "np.ndarray", eps: "np.ndarray", split: int, occupied: "np.ndarray", active: "np.ndarray", samples: "np.ndarray") -> float:
    """Compute the complete screened selected-CI interaction energy.

    Parameters
    ----------
    factors : np.ndarray
        Real symmetric Coulomb factors (r,n,n); v[p,q,r,s] is their
        Gram contraction sum_l factors[l,p,q]*factors[l,r,s].
    eps : np.ndarray
        Prescribed canonical energies (n,).
    split : int
        Full spatial labels [0,split) belong to A; [split,n) to B.
    occupied : np.ndarray
        Global doubly occupied reference labels, with no duplicates.
    active : np.ndarray
        Ordered active labels, A entries followed by B entries;
        internal fragment ordering need not be ascending.
    samples : np.ndarray
        Fixed raw (alpha_mask,beta_mask) rows whose bit positions refer
        to positions in active, NOT global orbital labels.

    Use selected_spaces at the active reference fragment electron
    counts. Restrict the BARE tensor and eps independently to A, B,
    and AB before each downfolding; each subsystem constructs its own
    reference h. Remap occupied/active labels into each subsystem's
    local full-orbital numbering. Solve each prescribed selected space
    with selected_energy, and return E_AB-E_A-E_B. Nuclear energies
    are zero. Do not extract fragment energies from an already
    screened dimer tensor, impose fragment Ms=0, or substitute full CI.

    Returns
    -------
    native Python float: E_AB-E_A-E_B, with no rounding.

    Raises
    ------
    ValueError
    If factors or eps has complex dtype (even with zero imaginary
    parts). If factors is not a real finite array (r,n,n) with 2<=n<=12
    and symmetric factors (atol=1e-12, rtol=0); r=0 is permitted.
    If eps is not a real finite vector of length n; if split is not
    an integer in [1,n) (booleans invalid); if occupied is not a vector
    of distinct integer indices in [0,n), or a reference occupied-to-
    virtual gap is nonpositive. If active is not a vector of 2 to 8
    distinct integer indices in [0,n), omits a fragment, or fails to
    put all A entries before all B entries. If samples is not a
    nonempty integer (k,2) array of in-range active masks; if any row
    does not have Nalpha=Nbeta equal to the number of active occupied
    reference orbitals; or if spin completion produces no neutral
    configuration at the active reference fragment electron counts.
    If any required full/constrained/active RPA problem is not strictly
    stable, an eigendecomposition fails, a required Coulomb or matrix
    symmetry fails at atol=1e-12, rtol=0, or any computed intermediate
    or energy is nonfinite. Strict RPA stability means positive
    eigenvalues of D^(1/2)(D+4K)D^(1/2) in the spatial convention.
    """
    return 0.0
```
