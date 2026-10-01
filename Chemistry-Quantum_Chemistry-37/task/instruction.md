# Chemistry-Quantum_Chemistry-37

## Problem

Near-degenerate electronic configurations can undermine perturbative predictions of excitation energies and their response to competing interactions. Locate the state-specific RSBW construction that selectively optimizes low-energy states and was demonstrated on lithium hydride and a four-hydrogen ring. Apply this construction to the published four-configuration spectroscopy model with two initially degenerate reference configurations, two perturbers, and couplings denoted \(K_\alpha,K_\beta,t,t'\), using its defining Hamiltonian and tabulated benchmark parameters to predict an excitation-gap response.

Retain the model’s published signed hopping convention and all tabulated parameters except for the controlled variations \(K_\alpha(a,b)/|t|=3.2+a\) and \(K_\beta(a,b)/|t|=K_\beta^{\mathrm{tab}}/|t|+b\), where \(a,b\) are dimensionless and the original configuration basis remains fixed. Target three states with selection thresholds \(\rho_{\min}=0.4\) and \(\rho'_{\min}=0.6\), using the symmetrized second-order effective Hamiltonian and the self-consistent second-order Brillouin–Wigner correction. For determinacy, include exactly degenerate unfrozen states in the candidate’s model space, interpret secondary enrichment as closure over unfrozen states, and use the published configuration order to break initial energy ties. For each corrected energy, select the root in the open interval between neighboring denominator poles that contains that state’s optimized zeroth-order energy, allowing an unbounded interval at a spectral endpoint.

Let \(\varepsilon_0(a,b)<\varepsilon_1(a,b)<\varepsilon_2(a,b)\) be the three resulting corrected energies in units of \(|t|\), and define \(\Delta(a,b)=\varepsilon_2(a,b)-\varepsilon_1(a,b)\). Compute the single scalar \(\left.\partial^2\Delta/\partial a\,\partial b\right|_{a=b=0}\) by analytic differentiation of the parameter-dependent construction, including its optimized reference states and implicit energy roots, on the locally unchanged selection branch; finite differences of separately solved models are excluded. Report the derivative to six decimal places and identify the sources of the model parameters and estimator; justify the local selection branch, reference updates, and root choices using compact equations or precise explanations that account for the optimized-basis response and implicit mixed differentiation. Limit numerical results to the baseline gap and mixed derivative; analytical equations are permitted and do not count as additional numerical results.

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

01_model_jet.py

Goal
----
Implement model_jet. The model fixes signed electronic couplings and reference energies. Analytic control derivatives start from a Hamiltonian jet whose mixed slot is zero even though the final energy response is not.

```python
def model_jet(a: float = 0.0, b: float = 0.0) -> np.ndarray:
    """Construct the four-configuration Hamiltonian and its control derivatives.

    Parameters
    ----------
    a, b : float
        Finite real scalar controls convertible to float.

        All energies are expressed in units of |t|. The ordered basis is
        (alpha, alpha_prime, beta, beta_prime), indexed by (0, 1, 2, 3).

        The real symmetric Hamiltonian is defined by

            H(a, b) = [
                [0,       3.2 + a, -1,      -1],
                [3.2 + a, 0,       -1.5,    -1.5],
                [-1,      -1.5,     3,       1.5 + b],
                [-1,      -1.5,     1.5 + b, 6]
            ].

        Only the explicitly indicated entries depend on the controls;
        all other entries remain fixed.

    Returns
    -------
    result : np.ndarray
        Floating-point array of shape (4, 4, 4), indexed as
        result[d, i, j]. The FIRST axis selects the derivative slot,
        and the last two axes index Hamiltonian rows and columns:

        result[0, :, :] = H(a, b)
        result[1, :, :] = partial H / partial a
        result[2, :, :] = partial H / partial b
        result[3, :, :] = partial^2 H / (partial a partial b)

        All derivatives are evaluated at the supplied controls.
        Entries are actual derivatives, not Taylor coefficients.

    Raises
    ------
    ValueError
        If either control is not a finite real scalar convertible to
        float, or if constructing the Hamiltonian produces any
        nonfinite entry.
    """
    return result
```

### Step 2

02_select_space.py

Goal
----
Implement select_space. State selection depends on coupling-to-energy-separation ratios. Secondary closure must include indirect couplings while excluding frozen states; exact degeneracies require selection without division.

```python
def select_space(H: np.ndarray, e: np.ndarray, k: int,
                         rho: float = 0.4, enrich: float = 0.6) -> np.ndarray:
    """Select a candidate model space with secondary enrichment closure.

    Parameters
    ----------
    H : np.ndarray
        Finite real symmetric (n,n) Hamiltonian, n>=1; symmetry atol=1e-12.
    e : np.ndarray
        Finite real reference energies of shape (n,). These need not equal
        diag(H). Indices below k are frozen and cannot enter the model space.
    k : int
        Candidate index, 0<=k<n; booleans are not accepted as integers.
    rho, enrich : float
        Finite real scalars satisfying 0<rho<enrich. Starting with k, include
        every unfrozen j whose |H[k,j]/(e[k]-e[j])| exceeds rho, including
        exact degeneracies regardless of coupling. Enrich to closure using
        the same pair ratio and threshold enrich. All tests are strict >;
        exact degeneracies always qualify. Never divide a zero denominator.

    Returns
    -------
    result : np.ndarray
        Sorted, distinct integer indices of the selected space, including k.

    Raises
    ------
    ValueError
        If arrays are not convertible to finite real arrays of the stated
        shapes, H is not symmetric within atol=1e-12 and rtol=0, k is invalid,
        or thresholds are not finite real scalars with 0<rho<enrich.
    """
    return result
```

### Step 3

03_effective_jet.py

Goal
----
Implement effective_jet. The effective Hamiltonian incorporates external-state perturbative contributions. Differentiation must include both coupling products and changing reference denominators before symmetrization.

```python
def effective_jet(H: np.ndarray, e: np.ndarray, P: np.ndarray) -> np.ndarray:
    """Construct the symmetrized second-order effective-Hamiltonian jet.

    Parameters
    ----------
    H : np.ndarray
        Finite real (4,n,n) jet, n>=1, each slot symmetric to atol=1e-12.
    e : np.ndarray
        Finite real reference-energy jet (4,n).
    P : np.ndarray
        Nonempty, strictly increasing integer index array in [0,n).
        Q is its complement. Slots are (value,a,b,ab), actual derivatives.
        For i,j in P the matrix is H_ij plus one half the sum over q in Q
        of H_iq H_qj [(e_i-e_q)^(-1)+(e_j-e_q)^(-1)]. Differentiate this
        expression through mixed order with P fixed. A full P gives H.

    Returns
    -------
    result : np.ndarray
        Shape (4,len(P),len(P)), the effective matrix and its derivatives.

    Raises
    ------
    ValueError
        If arrays cannot be converted to finite real arrays, shapes or P
        indices are invalid, any H slot fails symmetry at atol=1e-12,
        any baseline P-Q reference-energy separation is <=1e-10 in absolute
        value, or the calculated result is nonfinite.
    """
    return result
```

### Step 4

04_eigensystem_jet.py

Goal
----
Implement eigensystem_jet. Nondegenerate eigenvectors respond to perturbations as well as eigenvalues. Their mixed normalization term is necessary for a valid orthogonal basis jet.

```python
def eigensystem_jet(A: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Differentiate a simple symmetric eigensystem through mixed order.

    Parameters
    ----------
    A : np.ndarray
        Finite real symmetric jet (4,n,n), n>=1, in (value,a,b,ab) slots.
        Every slot must be symmetric to atol=1e-12, rtol=0. Baseline
        eigenvalues must have pairwise separations >1e-10. Eigenvectors
        are columns ordered by increasing eigenvalue. Fix each column's
        sign by making its largest-absolute baseline entry positive,
        choosing the lowest row index if tied. Derivatives use a smooth
        local continuation of this sign, not differentiation of argmax.

    Returns
    -------
    result : tuple[np.ndarray,np.ndarray]
        Eigenvalue jet (4,n) and normalized eigenvector jet (4,n,n).
        Include eigenvector response and mixed normalization terms.

    Raises
    ------
    ValueError
        If input is not convertible to a finite real jet of the stated
        shape and symmetry, baseline eigenvalues are separated by <=1e-10,
        the eigensolver fails, or the calculated result is nonfinite.
    """
    return result
```

### Step 5

05_update_partition.py

Goal
----
Implement update_partition. Effective eigenvalues define the new reference Hamiltonian. The physical Hamiltonian transforms in a parameter-dependent basis, and reference energies outside the selected space are retained.

```python
def update_partition(H: np.ndarray, e: np.ndarray, P: np.ndarray,
                             L: np.ndarray, V: np.ndarray, frozen: int = 0
                             ) -> tuple[np.ndarray, np.ndarray]:
    """Rotate a Hamiltonian jet and update its reference partition.

    Parameters
    ----------
    H, e : np.ndarray
        Finite real jets of shapes (4,n,n) and (4,n), n>=1. Each H slot
        is symmetric to atol=1e-12. Slots are (value,a,b,ab).
    P : np.ndarray
        Nonempty strictly increasing integer indices, all in [frozen,n).
    L, V : np.ndarray
        Finite real eigenvalue and eigenvector jets of shapes (4,p) and
        (4,p,p), p=len(P). V must be orthogonal through mixed order:
        the four jet slots of V.T@V equal (I,0,0,0) to atol=1e-9.
        Embed V on P and identity on its complement, transform H, replace
        e[:,P] with L, and retain every other reference energy unchanged.
    frozen : int
        Integer 0<=frozen<n (not bool). Preserve the prefix of this length;
        stably sort the remaining basis by updated baseline reference energy.

    Returns
    -------
    result : tuple[np.ndarray,np.ndarray]
        Transformed Hamiltonian jet (4,n,n) and reference jet (4,n), with
        one consistent permutation applied to every derivative slot.

    Raises
    ------
    ValueError
        If arrays are not convertible to finite real arrays of the stated
        shapes, H lacks the stated symmetry, P or frozen is invalid,
        V fails the stated derivative orthogonality test, or outputs
        contain nonfinite entries.
    """
    return result
```

### Step 6

06_bw_root_jet.py

Goal
----
Implement bw_root_jet. The second-order Brillouin-Wigner energy is an implicit root. Its response includes the changing poles, couplings and diagonal matrix element; vanishing baseline coupling can still have a nonzero mixed contribution.

```python
def bw_root_jet(H: np.ndarray, e: np.ndarray, k: int) -> np.ndarray:
    """Solve and analytically differentiate the selected second-order BW root.

    Parameters
    ----------
    H, e : np.ndarray
        Finite real jets (4,n,n) and (4,n), n>=1; each H slot symmetric
        to atol=1e-12. Slots contain actual derivatives (value,a,b,ab).
    k : int
        Integer index 0<=k<n, not bool. Solve
        F(z)=z-H_kk-sum_{j!=k} H_kj**2/(z-e_j)=0 in the open pole interval
        containing e_k. The interval may be unbounded; all e_j, j!=k,
        are treated as denominator poles even for zero coupling.
        Include derivatives of H, e and the implicit root. No finite
        differences. With n=1, return H[:,0,0].

    Returns
    -------
    result : np.ndarray
        Four-vector (root, root_a, root_b, root_ab).

    Raises
    ------
    ValueError
        If arrays are not convertible to finite real arrays of the stated
        shapes and symmetry, k is invalid, e_k lies within 1e-10 of another
        reference energy, the selected interval does not contain exactly
        one real root separated from both finite bounds by >1e-10,
        the baseline eigensolver fails, or the output is nonfinite.
    """
    return result
```

### Step 7

07_ssrsbw_jet.py

Goal
----
Implement ssrsbw_jet. Sequential state-specific optimization freezes previously accepted reference states but retains them as external perturbers. Each corrected energy is evaluated when its own reference is accepted.

```python
def ssrsbw_jet(H: np.ndarray, targets: int = 3,
                       rho: float = 0.4, enrich: float = 0.6,
                       max_updates: int = 50) -> np.ndarray:
    """Optimize states sequentially and return corrected energy derivatives.

    Parameters
    ----------
    H : np.ndarray
        Finite real symmetric Hamiltonian jet (4,n,n), n>=1, in actual
        (value,a,b,ab) derivatives. Symmetry atol=1e-12, rtol=0.
        Initially e is the diagonal jet of H. Stably order the initial
        basis by e[0], preserving input order for ties.
    targets : int
        Number of targeted states, 1<=targets<=n, not bool.
    rho, enrich : float
        Finite real scalars 0<rho<enrich, used by select_space.
    max_updates : int
        Positive integer, not bool; maximum transformations per target.
        For each target, use select_space, effective_jet, eigensystem_jet
        and update_partition until the selected space is a singleton.
        Compute bw_root_jet immediately, then freeze the optimized reference
        (not its corrected energy). Frozen states remain external perturbers.
        Finally sort the stored corrected energy jets by their baseline.
        Derivatives describe the selected fixed branch; do not differentiate
        the discrete selection decisions and do not use finite differences.

    Returns
    -------
    result : np.ndarray
        Shape (4,targets): sorted corrected energies and control derivatives.

    Raises
    ------
    ValueError
        If input shape, finiteness, reality, symmetry, integer controls or
        thresholds violate the conditions above; a required P-Q denominator
        has magnitude <=1e-10; an effective spectrum has separation <=1e-10;
        a rotation jet fails orthogonality at atol=1e-9; a BW reference lies
        within 1e-10 of a pole or lacks exactly one root >1e-10 from finite
        interval bounds; a numerical eigensolver fails; calculations become
        nonfinite; more than max_updates transformations are needed for a
        target; or final corrected energies have separation <=1e-10.
    """
    return result
```

### Step 8

08_gap_response.py

Goal
----
Implement gap_response. The requested observable is the mixed control derivative of the gap between the second and third sorted corrected energies, not an exact-diagonalization gap.

```python
def gap_response(a: float = 0.0, b: float = 0.0) -> float:
    """Return the mixed response of the specified corrected excitation gap.

    Parameters
    ----------
    a, b : float
        Finite real scalar controls with |a|<=0.001 and |b|<=0.001. This
        neighborhood retains the benchmark's selection branch. Build the
        published four-state model with model_jet and use ssrsbw_jet with
        targets=3, rho=0.4, enrich=0.6, max_updates=50.

    Returns
    -------
    result : float
        Native Python float d_a d_b (epsilon_2-epsilon_1), in |t| units.
        Return the unrounded value. Use analytic derivative propagation.

    Raises
    ------
    ValueError
        If controls are not finite real scalars convertible to float or
        exceed the stated neighborhood; a numerical eigensolver fails;
        computations become nonfinite; a required reference denominator,
        effective spectral gap, or corrected spectral gap has magnitude
        <=1e-10; a rotation jet fails orthogonality at atol=1e-9; a BW
        reference is within 1e-10 of a pole or its interval lacks exactly
        one root >1e-10 from finite bounds; or a target needs more than
        50 transformations.
    """
    return result
```
