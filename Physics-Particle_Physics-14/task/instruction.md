# Physics-Particle_Physics-14

## Background

Integration-by-parts identities reduce large families of loop integrals to smaller master-integral bases in perturbative particle physics. In the Baikov representation, polynomial syzygies can generate such identities without raising propagator powers. The cited work isolates a quotient of these syzygies by their critical part on the maximal cut and relates the resulting critical classes to surface terms in the large-dimensional-regulator limit.

The singular one-mass triangle provides a compact analytic example with a non-principal critical syzygy. The present benchmark uses that explicit source syzygy to set the critical normalization, reconstructs a deterministic lift inside a finite degree-bounded coefficient system, maps the lift to the source surface term, and evaluates a reproducible off-cut response path. The minimum-Euclidean-norm lift and the off-cut probe path are benchmark conventions; they are not claims about the paper's preferred representative.

## Problem

A compact one-loop triangle is used as a deterministic benchmark of the cited paper's critical-syzygy construction in the Baikov representation: use propagator coordinates \(z=(z_0,z_1,z_2)\), positive invariant \(s\), massless internal lines, and Cayley matrix \(C(s)=\begin{pmatrix}0&0&s\\0&0&0\\s&0&0\end{pmatrix}\), with the normalized polynomial \(\widehat B=4B/s\) represented in graded-lex basis \((1,z_0,z_1,z_2,z_0^2,z_0z_1,z_0z_2,z_1^2,z_1z_2,z_2^2)\). Recover from the source the Cayley-Menger formula for \(B\), the critical-syzygy quotient and the explicit non-principal one-mass-triangle syzygy; use the latter's critical component \(a_0=-2s^2\) as the normalization for the benchmark lift.

For the deterministic lift, use \(a_0\widehat B+\sum_{e=0}^2\bar a_e z_e\partial_e\widehat B+\sum_{e=0}^2\widetilde a_e z_e\widehat B=0\), with constant \(a_0\), each \(\bar a_e\) of total degree at most two, each \(\widetilde a_e\) of total degree at most one, and coefficient layout \([a_0;\bar a_0^{(2)};\bar a_1^{(2)};\bar a_2^{(2)};\widetilde a_0^{(1)};\widetilde a_1^{(1)};\widetilde a_2^{(1)}]\) using graded-lex order within each block; match all monomials through total degree four, then choose the unique minimum-Euclidean-norm coefficient vector satisfying the homogeneous equations and the source normalization. This minimum-norm choice is only a benchmark gauge for the paper's non-unique lift.

For unit propagator powers and \(\gamma=-2.5\), recover the paper's syzygy-to-surface-term map and construct \(S_s\); at probes \(u^{(1)}=(0.05,-0.03,0.02)\), \(u^{(2)}=(-0.04,0.06,0.01)\), and \(u^{(3)}=(0.02,0.01,-0.05)\), define \(r_\ell(s)=S_s(su^{(\ell)})/s^2\) for \(s=(2.0,3.5,5.0,7.5)\), and compute \(L=\sum_{j=2}^4\|r(s_j)-r(s_{j-1})\|_2\) without intermediate rounding. In the reasoning, give only the compact diagnostics needed to justify the result: the \(s=5\) Baikov polynomial, system shape/rank/nullity, lift norm and residual, the six constant terms of \(\bar a_e\) and \(\widetilde a_e\) at \(s=5\), and the coefficients of \(z_0\), \(z_1\), and \(z_0z_1\) in \(S_5\), the four response vectors, and the three path increments; explain separately the paper's cut check \(S(0)=a_0\) and the benchmark-only minimum-norm convention, then report \(L\) to exactly 12 digits after the decimal point.

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

01_triangle_normalized_baikov.py

Goal
----
Build the normalized one-mass-triangle Baikov polynomial.

```python
def triangle_normalized_baikov(s: float) -> "np.ndarray":
    '''Return normalized one-mass-triangle Baikov coefficients.

    Parameters
    ----------
    s : float
        Positive finite kinematic invariant.

    Returns
    -------
    np.ndarray
        Shape-(10,) coefficients of Bhat = 4 B / s in the fixed degree-two
        graded-lex basis (1,z0,z1,z2,z0^2,z0z1,z0z2,z1^2,z1z2,z2^2).

    Raises
    ------
    ValueError
        If s is non-finite or non-positive.
    '''
    return coeffs
```

### Step 2

02_triangle_source_nonprincipal_syzygy.py

Goal
----
Construct the paper's explicit non-principal one-mass-triangle syzygy.

```python
def triangle_source_nonprincipal_syzygy(s: float) -> "np.ndarray":
    '''Return the paper's explicit non-principal triangle syzygy.

    Parameters
    ----------
    s : float
        Positive finite kinematic invariant.

    Returns
    -------
    np.ndarray
        Shape-(43,) coefficient vector in the benchmark layout
        [a0; bar_a0^(2); bar_a1^(2); bar_a2^(2); tilde_a0^(1);
        tilde_a1^(1); tilde_a2^(1)].

    Raises
    ------
    ValueError
        If s is non-finite or non-positive.
    '''
    return coeffs
```

### Step 3

03_triangle_syzygy_matrix.py

Goal
----
Convert the degree-bounded triangle syzygy ansatz to coefficient matching.

```python
def triangle_syzygy_matrix(b: "np.ndarray") -> "np.ndarray":
    '''Assemble the degree-bounded syzygy coefficient-matching matrix.

    Parameters
    ----------
    b : np.ndarray
        Finite shape-(10,) coefficient vector of the normalized quadratic
        Baikov polynomial in the fixed degree-two graded-lex basis.

    Returns
    -------
    np.ndarray
        Shape-(35,43) coefficient matrix M for monomial matching through
        total degree four in the benchmark unknown layout.

    Raises
    ------
    ValueError
        If b does not have shape (10,) or contains non-finite values.
    '''
    return matrix
```

### Step 4

04_triangle_minimum_norm_lift.py

Goal
----
Lift the source critical class into the degree-bounded benchmark system.

```python
def triangle_minimum_norm_lift(s: float) -> "np.ndarray":
    '''Return the deterministic minimum-norm lift of the source critical class.

    Parameters
    ----------
    s : float
        Positive finite kinematic invariant.

    Returns
    -------
    np.ndarray
        Shape-(43,) minimum-Euclidean-norm coefficient vector satisfying the
        assembled syzygy system with the source critical normalization
        a0 = -2 s^2.

    Raises
    ------
    ValueError
        If s is non-finite or non-positive, or if the source triangle
        representative fails the assembled syzygy identity check.
    '''
    return coeffs
```

### Step 5

05_triangle_surface_polynomial.py

Goal
----
Map the reconstructed critical-syzygy lift to the source surface term.

```python
def triangle_surface_polynomial(s: float, gamma: float) -> "np.ndarray":
    '''Map the canonical syzygy lift to the unit-propagator surface polynomial.

    Parameters
    ----------
    s : float
        Positive finite kinematic invariant.
    gamma : float
        Finite nonzero Baikov exponent parameter in the source surface-term map.

    Returns
    -------
    np.ndarray
        Shape-(10,) coefficients of the quadratic surface polynomial S in the
        fixed degree-two graded-lex basis.

    Raises
    ------
    ValueError
        If gamma is non-finite or zero, or if an upstream s precondition fails.
    '''
    return coeffs
```

### Step 6

06_triangle_surface_path_table.py

Goal
----
Evaluate the source surface polynomial on an ordered kinematic probe path.

```python
def triangle_surface_path_table(s_values: "np.ndarray", gamma: float, probes: "np.ndarray") -> "np.ndarray":
    '''Evaluate normalized source-surface responses along an ordered s path.

    Parameters
    ----------
    s_values : np.ndarray
        Nonempty one-dimensional array of positive finite kinematic invariants.
    gamma : float
        Finite nonzero Baikov exponent parameter passed to the surface-term map.
    probes : np.ndarray
        Finite shape-(m,3) array of dimensionless off-cut probe coordinates.

    Returns
    -------
    np.ndarray
        Shape-(n,m) response table whose j,l entry is
        S_{s_j}(s_j * probes[l]) / s_j^2, preserving s_values order.

    Raises
    ------
    ValueError
        If s_values are invalid, probes do not have shape (m,3) or contain
        non-finite values, or an upstream gamma precondition fails.
    '''
    return table
```

### Step 7

07_cumulative_triangle_surface_path.py

Goal
----
Accumulate the Euclidean arc length of the ordered surface-response path.

```python
def cumulative_triangle_surface_path(s_values: "np.ndarray", gamma: float, probes: "np.ndarray") -> float:
    '''Return the cumulative Euclidean length of the ordered response path.

    Parameters
    ----------
    s_values : np.ndarray
        Nonempty one-dimensional array of positive finite kinematic invariants.
    gamma : float
        Finite nonzero Baikov exponent parameter.
    probes : np.ndarray
        Finite shape-(m,3) array of dimensionless off-cut probe coordinates.

    Returns
    -------
    float
        Sum_j ||r(s_j) - r(s_{j-1})||_2 as a native Python float, with 0.0
        returned for a one-point path.

    Raises
    ------
    ValueError
        If any upstream s_values, gamma, or probes precondition fails.
    '''
    return path_length
```
