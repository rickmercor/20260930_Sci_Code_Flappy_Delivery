# Physics-Condensed_Matter_Physics-43

## Background

Consider the translation-invariant periodic MPS
\[
\langle i_0\cdots i_{N-1}|\psi_N\rangle
=\operatorname{Tr}(A^{i_0}\cdots A^{i_{N-1}})
\]
with the matrices \(A^i\) given below; use the field-consensus conventions for lexicographic word order, row-major flattening, complex rank, canonical RREF, and basis comparison.

Let \(\mathcal A\) be the unital span of all products of the \(A^i\), let
\[
J=\{x\in\mathcal A:\operatorname{Tr}(xy)=0\ \forall y\in\mathcal A\},
\qquad
V=\sum_{x\in J}\operatorname{Im}(x),
\]
and obtain the injective normal factors from the restriction to \(V\) and the quotient action on \(\mathbb C^D/V\) (or \(A\) itself when \(J=0\)); for each factor \(F\), let \(L(F)\) be its injectivity length and \(r_\ell(F)\) the span dimension of its length-\(\ell\) products.

For \(k\in\{1,2\}\) and unit-modulus \(\omega\), define the factor-wise twisted local system by
\[
\sum_J O_{I,J}F^J-\mathbf1_{\{\omega=1\}}\epsilon F^I
=
F^{i_1}B^{(i_2,\ldots,i_k)}
-\omega B^{(i_1,\ldots,i_{k-1})}F^{i_k},
\]
with the single \(B^{()}\) used when \(k=1\), and denote its matrix by \(M_\omega(F,k)\). For \(\omega\ne1\), omit \(\epsilon\) from the unknown coordinates and interpret \(\pi_{O,\epsilon}\) as \(\pi_O\); include the \(\epsilon\) coordinate only when \(\omega=1\).

Let
\[
\tilde{\mathcal S}_\omega(A_j,k)=\pi_{O,\epsilon}\ker M_\omega(A_j,k),\qquad
\mathcal F_\omega(A_j,k)=\ker M_\omega(A_j,k)\cap\ker\pi_{O,\epsilon},
\]
\[
\tilde{\mathcal S}_\omega(A,k)=\bigcap_j\tilde{\mathcal S}_\omega(A_j,k),
\qquad
\mathcal S_\omega(A,k)=\pi_O\tilde{\mathcal S}_\omega(A,k).
\]

For admissible periodic sizes \(N\ge k\) satisfying \(\omega^N=1\), define
\[
\mathcal O_{N,\omega}(O)=\sum_{x=0}^{N-1}\omega^{-x}O_x,
\]
\[
\mathcal S_{N,\omega}=
\begin{cases}
\{O:\mathcal O_{N,1}(O)|\psi_N\rangle=e|\psi_N\rangle\},&\omega=1,\\
\{O:\mathcal O_{N,\omega}(O)|\psi_N\rangle=0\},&\omega\ne1,
\end{cases}
\]
\[
\mathcal Z_{N,\omega}=\{O:\mathcal O_{N,\omega}(O)=0\},
\qquad
s_N=\dim\mathcal S_{N,\omega},\quad
z_N=\dim\mathcal Z_{N,\omega},\quad
d_N=s_N-z_N.
\]

For \(k\ge2\), let
\[
\mathcal T_{k,\omega}
=\{I_d\otimes Q-\omega Q\otimes I_d\},
\]
with \(\mathcal T_{1,\omega}=\{0\}\); form \(Q_{\rm quot}\) by scanning the canonical rows of \(\mathcal S_\omega(A,k)\) and retaining exactly those that increase the rank over the rows of \(\mathcal T_{k,\omega}\), and define
\[
q=\dim Q_{\rm quot},\qquad
q_{\ne I}=q-\mathbf1_{\{\omega=1\}},
\]
\[
q_N=\dim\mathcal S_\omega(A,k)-\dim\bigl(\mathcal S_\omega(A,k)\cap\mathcal Z_{N,\omega}\bigr),
\qquad
e_N=(s_N-z_N)-q_N.
\]

Use
\[
d=3,\quad D=4,\quad k=2,\quad \omega=i,\quad N\in\{4,8\},
\]
\[
A^0=\begin{pmatrix}1&-1&1&0\\0&-1&0&0\\0&0&2&1\\0&1&0&1\end{pmatrix},\quad
A^1=\begin{pmatrix}-1&0&3&3\\0&0&2&2\\1&-1&1&0\\-1&1&1&2\end{pmatrix},\quad
A^2=\begin{pmatrix}1&-1&0&-2\\0&1&-1&-1\\0&-1&0&1\\0&0&0&-1\end{pmatrix},
\]
and
\[
X=\begin{pmatrix}1&1&0&0\\0&1&0&0\\0&0&1&1\\1&0&0&1\end{pmatrix}.
\]

Report: the normal-factor data and \(N_{\rm ref}=2\max_jL_j+2k-1\); the coordinate counts, shapes, ranks and nullities of the factor-wise \(M_\omega\); the dimensions of the projected and zero-fibre spaces and the canonical basis diagnostics for \(\mathcal S_\omega(A,k)\); \((s_N)\), \((z_N)\), \((d_N)\), and \(\langle\psi_N|\psi_N\rangle\); the dimension and pivots of \(\mathcal T_{k,\omega}\); \(q\), \(q_{\ne I}\), the size of \(Q_{\rm quot}\), the similarity and conjugation covariance checks, the intersection dimensions, \((q_N)\), and \((e_N)\).

Using recent primary literature, identify the complete local characterization of exact injective-MPS eigenstates and extract two source-specific statements not given above: the finite-size propagation statement and the ordinary untwisted auxiliary-\(B\) ambiguity; use them only to interpret the \(N=4\) versus \(N=8\) behavior and the \(\omega=i\) zero fibre, explicitly distinguishing the published untwisted theorem from the twisted computation here.

The single numeric target is
\[
d_8=s_8-z_8,
\]
and it must be reported as a finite decimal number.

## Problem

Consider the translation-invariant periodic MPS
\[
\langle i_0\cdots i_{N-1}|\psi_N\rangle
=\operatorname{Tr}(A^{i_0}\cdots A^{i_{N-1}})
\]
with the matrices \(A^i\) given below; use the field-consensus conventions for lexicographic word order, row-major flattening, complex rank, canonical RREF, and basis comparison.

Let \(\mathcal A\) be the unital span of all products of the \(A^i\), let
\[
J=\{x\in\mathcal A:\operatorname{Tr}(xy)=0\ \forall y\in\mathcal A\},
\qquad
V=\sum_{x\in J}\operatorname{Im}(x),
\]
and obtain the injective normal factors from the restriction to \(V\) and the quotient action on \(\mathbb C^D/V\) (or \(A\) itself when \(J=0\)); for each factor \(F\), let \(L(F)\) be its injectivity length and \(r_\ell(F)\) the span dimension of its length-\(\ell\) products.

For \(k\in\{1,2\}\) and unit-modulus \(\omega\), define the factor-wise twisted local system by
\[
\sum_J O_{I,J}F^J-\mathbf1_{\{\omega=1\}}\epsilon F^I
=
F^{i_1}B^{(i_2,\ldots,i_k)}
-\omega B^{(i_1,\ldots,i_{k-1})}F^{i_k},
\]
with the single \(B^{()}\) used when \(k=1\), and denote its matrix by \(M_\omega(F,k)\).

Let
\[
\tilde{\mathcal S}_\omega(A_j,k)=\pi_{O,\epsilon}\ker M_\omega(A_j,k),\qquad
\mathcal F_\omega(A_j,k)=\ker M_\omega(A_j,k)\cap\ker\pi_{O,\epsilon},
\]
\[
\tilde{\mathcal S}_\omega(A,k)=\bigcap_j\tilde{\mathcal S}_\omega(A_j,k),
\qquad
\mathcal S_\omega(A,k)=\pi_O\tilde{\mathcal S}_\omega(A,k).
\]

For admissible periodic sizes \(N\ge k\) satisfying \(\omega^N=1\), define
\[
\mathcal O_{N,\omega}(O)=\sum_{x=0}^{N-1}\omega^{-x}O_x,
\]
\[
\mathcal S_{N,\omega}=
\begin{cases}
\{O:\mathcal O_{N,1}(O)|\psi_N\rangle=e|\psi_N\rangle\},&\omega=1,\\
\{O:\mathcal O_{N,\omega}(O)|\psi_N\rangle=0\},&\omega\ne1,
\end{cases}
\]
\[
\mathcal Z_{N,\omega}=\{O:\mathcal O_{N,\omega}(O)=0\},
\qquad
s_N=\dim\mathcal S_{N,\omega},\quad
z_N=\dim\mathcal Z_{N,\omega},\quad
d_N=s_N-z_N.
\]

For \(k\ge2\), let
\[
\mathcal T_{k,\omega}
=\{I_d\otimes Q-\omega Q\otimes I_d\},
\]
with \(\mathcal T_{1,\omega}=\{0\}\); form \(Q_{\rm quot}\) by scanning the canonical rows of \(\mathcal S_\omega(A,k)\) and retaining exactly those that increase the rank over the rows of \(\mathcal T_{k,\omega}\), and define
\[
q=\dim Q_{\rm quot},\qquad
q_{\ne I}=q-\mathbf1_{\{\omega=1\}},
\]
\[
q_N=\dim\mathcal S_\omega(A,k)-\dim\bigl(\mathcal S_\omega(A,k)\cap\mathcal Z_{N,\omega}\bigr),
\qquad
e_N=(s_N-z_N)-q_N.
\]

Use
\[
d=3,\quad D=4,\quad k=2,\quad \omega=i,\quad N\in\{4,8\},
\]
\[
A^0=\begin{pmatrix}1&-1&1&0\\0&-1&0&0\\0&0&2&1\\0&1&0&1\end{pmatrix},\quad
A^1=\begin{pmatrix}-1&0&3&3\\0&0&2&2\\1&-1&1&0\\-1&1&1&2\end{pmatrix},\quad
A^2=\begin{pmatrix}1&-1&0&-2\\0&1&-1&-1\\0&-1&0&1\\0&0&0&-1\end{pmatrix},
\]
and
\[
X=\begin{pmatrix}1&1&0&0\\0&1&0&0\\0&0&1&1\\1&0&0&1\end{pmatrix}.
\]

Report: the normal-factor data and \(N_{\rm ref}=2\max_jL_j+2k-1\); the coordinate counts, shapes, ranks and nullities of the factor-wise \(M_\omega\); the dimensions of the projected and zero-fibre spaces and the canonical basis diagnostics for \(\mathcal S_\omega(A,k)\); \((s_N)\), \((z_N)\), \((d_N)\), and \(\langle\psi_N|\psi_N\rangle\); the dimension and pivots of \(\mathcal T_{k,\omega}\); \(q\), \(q_{\ne I}\), the size of \(Q_{\rm quot}\), the similarity and conjugation covariance checks, the intersection dimensions, \((q_N)\), and \((e_N)\).

Using recent primary literature, identify the complete local characterization of exact injective-MPS eigenstates and extract two source-specific statements not given above: the finite-size propagation statement and the ordinary untwisted auxiliary-\(B\) ambiguity; use them only to interpret the \(N=4\) versus \(N=8\) behavior and the \(\omega=i\) zero fibre, explicitly distinguishing the published untwisted theorem from the twisted computation here.

The single numeric target is
\[
d_8=s_8-z_8,
\]
and it must be reported as a finite decimal number.

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

Normal decomposition

Goal
----
Compute the basis-independent normal-decomposition invariants of an MPS tensor.

```python
def block_structure(A: np.ndarray, max_L: int = 6, tol: float = 1e-10) -> tuple:
    """Return basis-independent invariants of the normal decomposition.

    Parameters
    ----------
    A : np.ndarray
        Finite complex MPS tensor with shape (d, D, D).
    max_L : int
        Largest positive product length examined when certifying injectivity.
    tol : float
        Finite non-negative tolerance for deterministic complex row reduction.

    Returns
    -------
    result : tuple
        (num_factors, factor_dims, factor_L, algebra_dim, radical_dim), where
        factor_dims and factor_L are integer arrays in factor order.  The
        decomposition is defined by Alg(A), its trace-form radical J, and
        V=sum_{x in J} Im(x) as stated in the step background.

    Raises
    ------
    ValueError
        If an input is non-finite/invalid or the required factor(s) are not
        injective through max_L.
    """
    return result
```

### Step 2

Twisted local systems

Goal
----
Construct the ordered homogeneous local equation system for each injective

normal factor and return only its basis-independent dimensions.

```python
def twisted_local_equation_system(A: np.ndarray, k: int, omega: complex, max_L: int = 6, tol: float = 1e-10) -> tuple:
    """Return dimensions of the ordered local systems, one per normal factor.

    Use exactly the local equation and coordinate/equation ordering stated in
    the step background above.

    Parameters
    ----------
    A : np.ndarray
        Finite complex MPS tensor with shape (d, D, D).
    k : int
        Local operator range; must be 1 or 2.
    omega : complex
        Finite unit-modulus sector phase.  Epsilon is present iff omega=1
        within tol.
    max_L : int
        Largest positive product length used to certify the factors.
    tol : float
        Finite non-negative tolerance used in complex row reduction.

    Returns
    -------
    result : tuple
        (factor_L, N_ref, local_matrix_ranks, raw_nullities), with one rank and
        nullity per normal factor and N_ref=2*max(factor_L)+2*k-1.

    Raises
    ------
    ValueError
        If an input is non-finite/invalid or a factor is not injective through
        max_L.
    """
    return result
```

### Step 3

Intersected sector space

Goal
----
Construct the physical local operator space by projecting each normal-factor

kernel and intersecting the projected spaces.

```python
def twisted_operator_space(A: np.ndarray, k: int, omega: complex, max_L: int = 6, tol: float = 1e-10) -> tuple:
    """The projected local operator space of a momentum sector.

    Parameters
    ----------
    A : np.ndarray
        MPS tensor with shape (d, D, D), real or complex.
    k : int
        Local operator range; must be 1 or 2.
    omega : complex
        Unit-modulus sector phase.
    max_L : int
        Largest product length used to certify the factors.
    tol : float
        Non-negative tolerance used in nullspace and canonical-basis reduction.

    Returns
    -------
    result : tuple
        (raw_nullities, factor_projected_dims, operator_dim, zero_fibre_dims,
        operator_basis): per factor, the full nullity, the dimension of the
        (O, epsilon)-projection of the kernel and the dimension of its
        zero-projection fibre; then the dimension and canonical complex row
        basis of the operator projection of the intersection over factors.

    Raises
    ------
    ValueError
        If an input is invalid or a factor is not injective through max_L.
    """
    return result
```

### Step 4

Periodic momentum spaces

Goal
----
Compute the full-tensor finite periodic-chain momentum spaces and state norms.

```python
def periodic_momentum_spaces(A: np.ndarray, k: int, sizes: 'Sequence[int]', omega: complex, max_L: int = 6, tol: float = 1e-10) -> tuple:
    """Finite periodic-chain spaces for a momentum-weighted extensive operator.

    Parameters
    ----------
    A : np.ndarray
        MPS tensor with shape (d, D, D), real or complex.
    k : int
        Local operator range; must be 1 or 2.
    sizes : sequence of int
        Chain lengths to evaluate. Every size must satisfy N >= k and omega**N = 1
        within 10*tol.
    omega : complex
        Unit-modulus sector phase.
    max_L : int
        Largest product length used to certify the factors.
    tol : float
        Non-negative tolerance used throughout the linear algebra.

    Returns
    -------
    result : tuple
        (sizes, sector_dims, zero_sum_dims, distinct_dims, state_norms,
        local_dim, N_ref): the requested sizes, s_N, z_N, d_N, MPS norms, the
        local projected dimension of the tensor, and the reference size.

    Raises
    ------
    ValueError
        If an input or requested size is invalid, a size is not resonant with
        omega, or a factor is not injective through max_L.
    """
    return result
```

### Step 5

Twisted telescoping space

Goal
----
Construct the momentum-dependent telescoping density space with the exact

basis convention used by the benchmark.

```python
def twisted_telescopic_space(d: int, k: int, omega: complex, tol: float = 1e-10) -> tuple:
    """Return the canonical basis of T_{k,omega} defined above.

    Parameters
    ----------
    d : int
        Positive local physical dimension.
    k : int
        Local operator range; must be 1 or 2.
    omega : complex
        Unit-modulus sector phase.
    tol : float
        Non-negative tolerance used in canonical row reduction.

    Returns
    -------
    result : tuple
        (telescopic_dim, telescopic_basis): the dimension and canonical complex
        row basis of the twisted telescoping space.

    Raises
    ------
    ValueError
        If d, k, omega, or tol is invalid.
    """

    return result
```

### Step 6

Momentum quotients

Goal
----
Compute the twisted telescoping quotient, covariance checks, and finite-size

comparison using a fully specified representative-selection rule.

```python
def momentum_quotients(A: np.ndarray, k: int, sizes: 'Sequence[int]', omega: complex, X: "np.ndarray | None" = None, max_L: int = 6, tol: float = 1e-10) -> tuple:
    """Return the quotient and finite-size data using the rule above.

    Parameters
    ----------
    A : np.ndarray
        MPS tensor with shape (d, D, D), real or complex.
    k : int
        Local operator range; must be 1 or 2.
    sizes : sequence of int
        Resonant periodic chain lengths to evaluate.
    omega : complex
        Unit-modulus sector phase.
    X : np.ndarray or None
        Optional invertible D by D virtual similarity matrix; identity if None.
    max_L : int
        Largest product length used to certify the factors.
    tol : float
        Non-negative tolerance used throughout the linear algebra.

    Returns
    -------
    result : tuple
        (local_dim, telescopic_dim, quotient_dim, nonidentity_dim,
        similarity_union_dim, conjugation_union_dim, intersection_dims,
        finite_quotient_dims, excess_dims, quotient_basis).

    Raises
    ------
    ValueError
        If an input is invalid, X is singular, a size is non-resonant, or a
        factor is not injective through max_L.
    """
    return result
```

### Step 7

Momentum diagnostic

Goal
----
Assemble the complete momentum-resolved local/global calculation, verify that

all preceding sub-problems are mutually consistent, and return one scalar: the

physical momentum-sector dimension d_N=s_N-z_N at the last requested chain

size.

```python
def momentum_diagnostic(A: np.ndarray, k: int, sizes: 'Sequence[int]', omega: complex, X: "np.ndarray | None" = None, max_L: int = 6, tol: float = 1e-10) -> float:
    """Return d_N=s_N-z_N for the last requested resonant chain size.

    The function must execute the full Step-01--Step-06 pipeline and reject
    mutually inconsistent intermediate results; it is not a shortcut that
    computes only the final finite-size rank.

    Parameters
    ----------
    A : np.ndarray
        Finite complex MPS tensor with shape (d, D, D).
    k : int
        Local operator range; must be 1 or 2.
    sizes : sequence of int
        Nonempty sequence of resonant periodic chain lengths; the returned
        scalar corresponds to sizes[-1].
    omega : complex
        Finite unit-modulus sector phase.
    X : np.ndarray or None
        Optional finite invertible D by D virtual similarity matrix. If None,
        the identity (no transformation) is used.
    max_L : int
        Largest product length used to certify normal factors.
    tol : float
        Finite non-negative linear-algebra tolerance.

    Returns
    -------
    result : float
        The physical momentum-sector dimension d_N=s_N-z_N at sizes[-1].

    Raises
    ------
    ValueError
        If an input is invalid/non-finite, a size is non-resonant, or the
        preceding sub-problem results are mutually inconsistent.
    """
    return result
```
