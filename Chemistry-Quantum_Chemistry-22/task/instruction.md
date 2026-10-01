# Chemistry-Quantum_Chemistry-22

## Problem

Thermal charge fluctuations in an embedded electronic fragment depend on both its orbital definition and the treatment of its environment. Using intrinsic atomic orbitals and the extended-valence and moment-expansion bath constructions of finite-temperature density matrix embedding theory, determine how the bath choice changes the interacting impurity charge variance for the supplied model.

Use six orthonormal spatial orbitals \(\{|e_i\rangle\}_{i=1}^{6}\), canonical fermion operators \(c_{i\sigma}\), dimensionless energy units with \(k_B=1\), and the fixed reference one-electron operator \(F\), minimal-basis coefficient matrix \(M\), and physical Hamiltonian
\[
F=
\begin{pmatrix}
-1.60&-0.45&0.12&0&0.18&-0.08\\
-0.45&-0.80&-0.35&0.16&0&0.11\\
0.12&-0.35&-0.15&-0.40&0.14&0\\
0&0.16&-0.40&0.35&-0.30&0.17\\
0.18&0&0.14&-0.30&0.90&-0.25\\
-0.08&0.11&0&0.17&-0.25&1.40
\end{pmatrix},
\qquad
M=
\begin{pmatrix}
1&0.12&0&0\\
0.08&1&0.10&0\\
0&0.06&1&0.09\\
0.04&0&0.08&1\\
0.30&-0.20&0.25&0.10\\
-0.15&0.28&0.05&0.32
\end{pmatrix},
\]
\[
|\widetilde\rho_\alpha\rangle=\sum_{i=1}^{6}|e_i\rangle M_{i\alpha},
\qquad
\widehat H=\sum_{\sigma\in\{\uparrow,\downarrow\}}\sum_{i,j=1}^{6}
F_{ij}c^\dagger_{i\sigma}c_{j\sigma}
+\sum_{i=1}^{6}U_i n_{i\uparrow}n_{i\downarrow},
\qquad
U=(2.4,1.7,2.1,1.3,1.9,1.5),
\quad n_{i\sigma}=c^\dagger_{i\sigma}c_{i\sigma}.
\]
The fixed closed-shell reference occupies the two lowest eigenvectors of \(F\); construct all four IAOs from this reference and the supplied minimal basis using the original polarized-orbital prescription with final symmetric orthogonalization, assign the first two IAOs to impurity \(A\), designate only the second as valence for bath generation, and retain both as active impurity orbitals.
Take \(\beta=2.3\) and obtain the spin-restricted thermal reference density from the fixed operator \(F\) with mean total electron number \(4\), keeping the IAO reference fixed and performing no Hartree–Fock or correlation-potential updates.

Compare the extended-valence bath retaining two bath orbitals with the valence-only moment-expansion bath through order two, using the complete orthogonal complement of impurity \(A\) as the environment and discarding linear dependencies below an absolute singular-value threshold of \(10^{-10}\).
For each construction, use the interacting-bath projection of \(\widehat H\), retaining all projected two-body terms and treating excluded spatial modes as empty without frozen-environment corrections.
Determine the two embedding chemical potentials simultaneously so that the exact grand-canonical mean electron numbers in the embedding and impurity spaces equal their respective thermal-reference values—twice the trace of the one-spin reference density restricted to each space—with all particle-number and spin sectors included and no low-temperature truncation.
Compute the single scalar
\[
\mathcal C=
\left[\langle\widehat N_A^2\rangle-\langle\widehat N_A\rangle^2\right]_{\mathrm{MEB}}
-
\left[\langle\widehat N_A^2\rangle-\langle\widehat N_A\rangle^2\right]_{\mathrm{EVB}},
\qquad
\widehat N_A=\sum_{\alpha=1}^{2}\sum_{\sigma}
a^\dagger_{\alpha\sigma}a_{\alpha\sigma},
\]
where \(a_{\alpha\sigma}\) annihilates an electron in impurity IAO \(\alpha\) and each expectation uses that construction’s independently number-matched interacting thermal state.
Report \(\mathcal C\) to six decimal places and identify the retrieved sources and decisive definitions, briefly justifying the impurity construction, distinct bath spaces, and interacting thermal weighting with compact equations or precise explanations; limit numerical results to the two charge variances and their difference.
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

01_intrinsic_orbitals.py

Goal
----
Construct the complete, symmetrically orthogonalized intrinsic atomic orbital basis from a fixed occupied reference and a nonorthogonal minimal basis.

```python
def intrinsic_orbitals(F: "np.ndarray", M: "np.ndarray", nocc: int) -> "np.ndarray":
    """Construct the original polarized IAOs in the orthonormal primary basis.

    Parameters
    ----------
    F : np.ndarray
        Real symmetric (n,n) fixed reference matrix, n >= 2.
    M : np.ndarray
        Real (n,m) minimal-basis coefficients, 1 <= m <= n.
    nocc : int
        Number of occupied spatial reference orbitals: 1 <= nocc < n,
        nocc <= m. Use the lowest eigenvectors of F. The reference is fixed,
        not a fractional-occupation projector.

    Returns
    -------
    Q : np.ndarray
        (n,m) original polarized IAOs, symmetrically orthogonalized together
        in the supplied column ordering. Use positive symmetric inverse
        square roots, not an arbitrary QR basis.

    Raises
    ------
    ValueError
        If data are not convertible to real finite float arrays, shapes or
        nocc violate the above bounds (bool is not an integer here), F differs
        from F.T by more than 1e-10 in any entry, or its occupied/unoccupied
        gap is <= 1e-10. Also if any eigenvalue of the minimal-basis Gram
        matrix, projected-occupied Gram matrix, or polarized-orbital Gram
        matrix is <= 1e-12. Accepted near-symmetric F is symmetrized.
    
    """
    return None  # placeholder
```

### Step 2

02_thermal_reference.py

Goal
----
Calculate a spin-restricted thermal reference density at a prescribed mean total electron number.

```python
def thermal_reference(F: "np.ndarray", beta: float, ne: float) -> "tuple[np.ndarray, float]":
    """Return the one-spin Fermi density and number-matching chemical potential.

    Parameters
    ----------
    F : np.ndarray
        Finite real symmetric (n,n) matrix, n >= 1; symmetry tolerance 1e-10.
    beta : float
        Finite inverse temperature, 0 < beta <= 100.
    ne : float
        Finite mean total electron number, 1e-6 <= ne <= 2*n-1e-6.

    Returns
    -------
    (D, mu) : tuple[np.ndarray, float]
        One-spin density D with 2*trace(D)=ne and native-float mu.
        Use the fixed F without a self-consistency update. Accepted
        near-symmetric F is symmetrized. Degenerate spectra are allowed.

    Raises
    ------
    ValueError
        If F is not convertible to a real finite square nonempty array,
        is asymmetric beyond 1e-10, or beta/ne are not real scalar numbers
        convertible to floats, are nonfinite, or violate the stated bounds.
    
    """
    return None  # placeholder
```

### Step 3

03_bath_projectors.py

Goal
----
Return gauge-independent EVB and order-two valence MEB projectors for a two-orbital impurity.

```python
def bath_projectors(D: "np.ndarray", A: "np.ndarray", cutoff: float = 1e-10) -> "tuple[np.ndarray, np.ndarray]":
    """Construct extended-valence and moment-expansion bath projectors.

    Parameters
    ----------
    D : np.ndarray
        Finite real symmetric (n,n) one-spin occupation matrix, n >= 2,
        eigenvalues in [-1e-10,1+1e-10]; do not renormalize its trace.
    A : np.ndarray
        Finite real (n,2) impurity coefficients, A.T@A=I within 1e-10
        entrywise. Both columns remain impurity; only the second is valence.
    cutoff : float
        Real finite scalar, 0 < cutoff < 1. Absolute singular-value cutoff.

    Returns
    -------
    (P_evb, P_meb) : tuple[np.ndarray, np.ndarray]
        Two (n,n) orthogonal environment projectors for the extended-valence
        and second-order moment-expansion bath prescriptions in Section
        II.2.1, equations (4)-(5), of
        https://arxiv.org/html/2601.01641v2.
        Use both impurity columns for the extended-valence bath and only
        the second impurity column as the moment-expansion seed.
        For each construction, concatenate its raw generating columns
        and retain left singular directions with singular values strictly
        greater than cutoff. Apply this cutoff to the concatenated raw
        matrix, without prior column normalization or sequential
        residual-vector thresholding. Retain at most two bath directions.
        Return the orthogonal projector onto the retained subspace.
        An empty retained span returns the zero projector. Symmetrize D.

    Raises
    ------
    ValueError
        If inputs are not convertible to finite real arrays/scalar, shapes
        violate the above, D is asymmetric beyond 1e-10 or violates its
        spectral bounds, A is not orthonormal within 1e-10, or cutoff is
        not a real scalar strictly between zero and one.
    """
    return None  # placeholder
```

### Step 4

04_embedded_operators.py

Goal
----
Project the interacting spatial-orbital Hamiltonian into an impurity-plus-bath space and construct full-Fock number operators.

```python
def embedded_operators(h: "np.ndarray", U: "np.ndarray", A: "np.ndarray", P: "np.ndarray") -> "tuple[np.ndarray, np.ndarray, np.ndarray]":
    """Build H, impurity number NA, and total number NT on the full Fock space.

    Parameters
    ----------
    h : np.ndarray
        Finite real symmetric (n,n) original one-body matrix, n >= 2.
    U : np.ndarray
        Finite real (n,) onsite opposite-spin interaction coefficients.
        Zero and negative coefficients are allowed.
    A : np.ndarray
        Finite real (n,2) orthonormal impurity columns, kept in this order.
    P : np.ndarray
        Finite real symmetric (n,n) bath orthogonal projector, orthogonal
        to A, rank r in {0,1,2}. All matrix tolerances below are 1e-9.

    Returns
    -------
    (H, NA, NT) : tuple[np.ndarray, np.ndarray, np.ndarray]
        Real matrices of size 2**(2*k), k=2+r. Define a bath basis by
        repeatedly taking a column of residual projector R, selecting the
        smallest index whose R[j,j] is within 1e-12 of its maximum,
        normalizing R[:,j] in Euclidean norm, then replacing R by
        sym(R-b b.T). Start R=sym(P), perform r iterations, E=[A,B].
        Basis states are ascending integers 0..2**(2*k)-1, with bits
        0..k-1 for up spin and k..2*k-1 for down spin. Apply canonical
        fermionic signs in that order. Project all original interactions
        U_i n_i_up n_i_down, not just embedding onsite terms. Excluded
        orbitals are empty; add no frozen-core, mean-field or constant term.

    Raises
    ------
    ValueError
        If arrays are not convertible to finite real floats; dimensions
        violate the above; h/P is asymmetric beyond 1e-9; A.T@A differs
        from I beyond 1e-9; P@P differs from P beyond 1e-9; P@A differs
        from zero beyond 1e-9; or the number of eigenvalues of sym(P)
        greater than 0.5 exceeds two. Symmetrize accepted h/P.
    
    """
    return None  # placeholder
```

### Step 5

05_thermal_response.py

Goal
----
Evaluate interacting thermal means, impurity charge variance and the analytic response to two chemical potentials.

```python
def thermal_response(H: "np.ndarray", NA: "np.ndarray", NT: "np.ndarray", beta: float, mu: "np.ndarray") -> "tuple[np.ndarray, float, np.ndarray]":
    """Evaluate the full thermal state and analytic two-field susceptibility.

    Parameters
    ----------
    H, NA, NT : np.ndarray
        Real finite symmetric (d,d) operators of equal nonzero size. They
        need not commute. Symmetry tolerance is 1e-9; symmetrize accepted
        inputs. Operators may be general observables for testing.
    beta : float
        Real finite scalar, 0 < beta <= 100.
    mu : np.ndarray
        Real finite shape (2,), ordered (mu_gc,mu_imp).

    Returns
    -------
    (means, variance, J) : tuple[np.ndarray, float, np.ndarray]
        For K=H-mu[0]*NT-mu[1]*NA and rho=exp(-beta*K)/Z,
        means=(Tr(rho NT),Tr(rho NA)), variance=Tr(rho NA@NA)-means[1]**2,
        and J[i,j]=d means[i]/d mu[j], at fixed H,NA,NT,beta.
        J must be analytic, including noncommuting eigenvector response
        and continuous limits at degeneracies; finite differences are
        excluded. Do not clip the returned variance or response.

    Raises
    ------
    ValueError
        If arrays are not convertible to finite real float arrays, operator
        shapes/symmetry or mu shape violate the above, or beta is not a
        real finite scalar in (0,100].
    
    """
    return None  # placeholder
```

### Step 6

06_match_numbers.py

Goal
----
Solve simultaneous grand-canonical total and impurity number constraints using analytic thermal response.

```python
def match_numbers(H: "np.ndarray", NA: "np.ndarray", NT: "np.ndarray", beta: float, targets: "np.ndarray") -> "tuple[np.ndarray, float, np.ndarray]":
    """Find a jointly number-matched interacting thermal state.

    Parameters
    ----------
    H, NA, NT, beta
        Same real, finite, dimensional, symmetry and beta contracts as
        thermal_response: nonempty equal square operators, symmetry
        tolerance 1e-9, scalar 0 < beta <= 100. The traceless parts of
        NT and NA must be linearly independent: the smaller eigenvalue
        of their 2x2 Frobenius Gram matrix must exceed 1e-12.
    targets : np.ndarray
        Finite real (2,), ordered (total,impurity). Each target must be
        strictly inside the corresponding symmetrized operator's spectral
        interval. Inputs must admit a finite simultaneous solution.

    Returns
    -------
    (mu, variance, means) : tuple[np.ndarray, float, np.ndarray]
        Ordered chemical potentials (gc,imp), impurity variance, and
        ordered means. Both residuals must be <= 1e-9 in absolute value.
        Use analytic thermal response in the fit, not finite differences. Do not freeze a potential after
        satisfying only one constraint. No numerical output clipping.

    Raises
    ------
    ValueError
        On any invalid thermal_response input, invalid targets, failure of
        the stated operator-independence/spectral conditions, or inability
        to obtain a finite joint fit meeting absolute residual 1e-9.
    
    """
    return None  # placeholder
```

### Step 7

07_compare_baths.py

Goal
----
Compare interacting impurity variances using separate EVB and MEB number-matched ensembles.

```python
def compare_baths(h: "np.ndarray", U: "np.ndarray", A: "np.ndarray", D: "np.ndarray", beta: float) -> tuple[float, float, float]:
    """Evaluate the two bath variances with independent chemical-potential fits.

    Parameters
    ----------
    h, U, A
        Same contracts as embedded_operators for these inputs: finite real
        h (n,n), U (n,), A (n,2), n >= 2, h symmetry and A orthonormality
        to 1e-9. A must also satisfy bath_projectors' stricter 1e-10 metric
        tolerance. No restriction on the signs of U.
    D : np.ndarray
        Same (n,n) one-spin occupation density contract as bath_projectors:
        real finite, symmetry 1e-10, spectral bounds [-1e-10,1+1e-10].
    beta : float
        Finite real scalar, 0 < beta <= 100.

    Returns
    -------
    (var_evb, var_meb, difference) : tuple[float, float, float]
        Native floats, with difference=var_meb-var_evb. Generate baths
        using cutoff=1e-10; both must have rank two. The impurity target
        is 2*trace(A.T D A); the target for bath projector P is
        2*trace((A A.T+P) D).
        Obtain each pair of chemical potentials by match_numbers.

    Raises
    ------
    ValueError
        On any invalid input under bath_projectors/embedded_operators/
        match_numbers contracts; if either bath rank is not two; or if
        either target is outside its open operator spectral interval or
        no finite simultaneous fit reaches absolute residual 1e-9.
    
    """
    return None  # placeholder
```

### Step 8

08_charge_variance_difference.py

Goal
----
Orchestrate the complete fixed-reference IAO plus finite-temperature interacting-bath variance comparison.

```python
def charge_variance_difference(F: "np.ndarray", M: "np.ndarray", U: "np.ndarray", beta: float = 2.3, ne: float = 4.0, nocc: int = 2) -> float:
    """Return the complete task's MEB-minus-EVB impurity variance difference.

    Parameters
    ----------
    F : np.ndarray
        Fixed real finite symmetric (n,n) reference and physical one-body
        matrix, n >= 4; symmetry tolerance 1e-10, symmetrized if accepted.
    M : np.ndarray
        Real finite (n,m) minimal-basis coefficients, 2 <= m <= n.
    U : np.ndarray
        Real finite shape (n,) onsite opposite-spin interactions.
    beta : float
        Real finite scalar, 0 < beta <= 100.
    ne : float
        Real finite scalar, 1e-6 <= ne <= 2*n-1e-6.
    nocc : int
        Fixed reference occupied spatial count, 1 <= nocc < n and
        nocc <= m, excluding bool; the occupied boundary gap must exceed
        1e-10. All three IAO Gram matrices must have eigenvalues > 1e-12.

    Returns
    -------
    difference : float
        Native unrounded float. Construct all IAOs before selecting the
        first two as active impurity; the second is the only MEB seed.
        Use thermal_reference and compare_baths with their exact contracts,
        requiring rank-two baths and simultaneous exact number matching.
        No F update, trace-one density, frozen-environment term, or thermal
        sector truncation.

    Raises
    ------
    ValueError
        On any nonreal/nonfinite/nonconvertible input or stated shape,
        parameter, reference-gap or Gram-spectrum violation; either bath
        not having rank two; a number target outside its open spectral
        interval; or inability to fit both thermal number constraints to
        absolute residual 1e-9. These include the inherited contracts of
        intrinsic_orbitals, thermal_reference, and compare_baths.
    
    """
    return None  # placeholder
```
