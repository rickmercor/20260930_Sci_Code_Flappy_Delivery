# Mathematics-Numerical_Linear_Algebra-39

## Background

Density-matrix construction dominates the cost of linear-scaling electronic structure, where the step function of a large symmetric matrix must be formed without diagonalisation, using only additions, scalar multiplications and matrix-matrix products. Recursive polynomial expansions build it from low-degree component polynomials, and the fastest of them accelerate the early iterations by choosing, from bounds on the two eigenvalues that border the gap, component polynomials that amplify the gap as much as possible. Recursing on degree-eight component polynomials that equioscillate on the two intervals enlarges the class of polynomials a few multiplications can represent; each accelerated member has to be constructed by solving its defining nonlinear conditions, and the scheme keeps two kinds of bounds, safe outer ones that the construction relies on and inner ones that the choice of member relies on. On the bounds of a real Hartree-Fock example the conditioning phase of such a scheme can be audited exactly, iteration by iteration.

## Problem

Computing the matrix step function of a symmetric matrix, the spectral projector onto the eigenvalues above a gap, is a core primitive of numerical linear algebra and of linear-scaling electronic structure. Recursive polynomial expansions build it from low-degree component polynomials; a recent source recurses on component polynomials of degree eight, each chosen from a family of equioscillating polynomials constructed for the current eigenvalue bounds so that the homo-lumo gap is amplified as much as possible in every iteration. The source reports the saving in matrix multiplications this accelerated scheme achieves over the competing methods and what a superfluous iteration would cost on a Hartree-Fock density matrix of aspirin; it plots the condition number of the problem, the inverse gap between the inner bounds, iteration by iteration, but tabulates neither the accelerated polynomials nor the bounds they produce. Using the uploaded source as the authoritative reference, implement its accelerated scheme exactly as it specifies it, construct its equioscillating polynomials yourself, and audit the conditioning phase on the source's own aspirin bounds.

Everything the source fixes is to be taken from it and not from a plausible alternative: the parametrisation of a component polynomial by its seven stationary points, a scale and an offset; the equioscillatory conditions that define the family member with $L$ stationary points in the left interval and $7 - L$ in the right, with the ordering of the stationary points relative to the construction bounds; the modified conditions at a bound sitting at its limit, and the closed-form members obtained when both limits are imposed together with the symmetry that generates the members with few left stationary points from those with many; the evaluation scheme with three matrix multiplications and three matrices in memory, its real coefficients and its workspace rearrangement; the roles of the inner and outer bounds in the scheme, the threshold that deactivates the acceleration, the choice of the member applied in each iteration, the alternation once both inner bounds are close to their limits, the mapping of all four bounds through the applied polynomial; the termination tests with their orders and constants; the normalisation that maps a Fock matrix with known extremal eigenvalues to a matrix with spectrum in the unit interval; and the spectral data of its Hartree-Fock example.

Conventions fixed here. The energies of the example are the source's Table 1 values in the source's order $\lambda_{min}$, $\lambda^{out}_{homo}$, $\lambda^{in}_{homo}$, $\lambda^{in}_{lumo}$, $\lambda^{out}_{lumo}$, $\lambda_{max}$, namely $-20.6248$, $-0.3573$, $-0.3482$, $0.0355$, $0.0565$ and $51.7152$ (in the source's units), and after the source's normalisation, which maps the occupied end of the spectrum to the upper end of the unit interval, the four bounds are ordered $0 \le \lambda^{out}_{lumo} \le \lambda^{in}_{lumo} < \lambda^{in}_{homo} \le \lambda^{out}_{homo} \le 1$. The source's matrix is not available, so the audit runs on a synthetic $16 \times 16$ matrix consistent with the bounds: $\lambda_{lumo}$ sits at the midpoint of $[\lambda^{out}_{lumo}, \lambda^{in}_{lumo}]$ and $\lambda_{homo}$ at the midpoint of $[\lambda^{in}_{homo}, \lambda^{out}_{homo}]$; with $\mu = (\lambda^{in}_{lumo} + \lambda^{in}_{homo})/2$ the occupation count is $n_{occ} = \mathrm{round}(16(1 - \mu))$; the spectrum consists of $n_{occ}$ equidistant eigenvalues on $[\lambda_{homo}, 1]$ and $16 - n_{occ}$ on $[0, \lambda_{lumo}]$, both endpoints included, assembled in ascending order on the orthonormal frame $Q$ from the QR factorisation of the integer matrix $M_{ij} = (((i+1)(j+2)(i+j+3) + a(i+4)(j+5)) \bmod 47) - 23$, $i, j = 0, \dots, 15$, each column of $Q$ sign-fixed so that its first entry of modulus above $10^{-12}$ is positive, with $X_0 = Q\,\mathrm{diag}(\lambda)\,Q^T$ symmetrised. The scheme runs for exactly five iterations with the source's threshold, the member of an iteration is chosen among $L = 0, \dots, 7$ by the source's rule with the smallest index on a tie, the construction bounds of the family are either at their limits or at least the threshold away from them, and every equioscillating member must satisfy its defining conditions to $10^{-9}$ relative to the amplitude of its interval with its stationary points in the source's ordering; a solution of the conditions that violates the ordering is not the member. Arrays of bounds hold $\lambda^{in}_{lumo}$, $\lambda^{in}_{homo}$, $\lambda^{out}_{lumo}$, $\lambda^{out}_{homo}$ in this order, and mapped bounds are clipped to the unit interval, with the outer bounds kept enclosing the inner ones, against rounding. The condition number of the problem after an iteration is the inverse of the gap between the inner bounds. All outputs are float64, finite and deterministic.

Implement eleven functions. sp8_closed_form(l) returns the (9,) monomial coefficients of the closed-form member of index l. sp8_from_stationary_points(s) expands the source's parametrisation into (9,) monomial coefficients. equioscillation_residual(s, L, lam_lumo, lam_homo) evaluates the (9,) residual of the defining conditions, the modified ones at a limit, in the order fixed by the sub-problem docstring. sp8_equioscillatory(L, lam_lumo, lam_homo) returns the (9,) parameters of the ordered member. sastre_coefficients(b) returns the (10,) real evaluation coefficients, workspace_scalars(coeffs) the (4,) rearrangement scalars, and evaluate_p8(X, b) the polynomial of a symmetric matrix by the source's scheme. select_member(bounds, i, kappa) returns the (3,) applied index and the two construction bounds of iteration i. propagate_bounds(bounds, b) maps the four bounds through the applied polynomial. stop_check(tr_prev, tr_curr, l) returns the termination code. sp8_accelerated_audit(energies, seed, niter), the final step, must be assembled by calling the earlier functions: it returns the (niter, 6) array whose rows hold, after each iteration, the applied index, the four bounds and the idempotency trace of the matrix.

Evaluate sp8_accelerated_audit with the six aspirin energies above, seed 3 and five iterations.

In your reasoning report the conventions you recovered from the source and justify each from it: the parametrisation and the defining conditions of the family with the ordering of the stationary points; the modified conditions, the closed-form members and the symmetry with its mirrored bounds; the evaluation scheme with its closing equation; the two roles of the outer and inner bounds, the threshold, the selection rule, the alternation and the bound mapping; the termination tests with their orders and constants; the normalisation of the Fock matrix and the aspirin spectral data with the condition number it implies; and the cost the source reports for its accelerated scheme against the competing methods, across step locations and on the aspirin example. State how you solved the defining conditions, how you made sure the solution is the ordered one, and what the nine conditions reduce to on the stationary points alone.

Report numerically, as evidence that the chain was executed: the four normalised bounds and the condition number before the first iteration, with the occupation count and the idempotency trace of the synthetic matrix before the first iteration; for each of the first four iterations the applied index, the two construction bounds and the condition number after the iteration; the seven stationary points, the scale and the offset of the member applied in the first iteration and the values it takes at the two construction bounds; the inner and the outer bounds after the second iteration; the iteration after which both accelerations are deactivated and the iteration at which the alternation begins, with the member it starts with; the idempotency trace of the synthetic matrix after the first iteration; the condition number after the second iteration that would result from building the family from the inner bounds, from the closed-form members without acceleration, from the symmetry applied without mirroring the bounds (that is $1 - p_{(R,L;\lambda_{lumo},\lambda_{homo})}(1 - x)$), and from a plain complement in place of the symmetry (that is $1 - p_{(R,L;1-\lambda_{homo},1-\lambda_{lumo})}(x)$, the mirrored bounds kept but the argument not reflected); and the largest gap amplification the family can achieve in the narrow-gap limit at a gap centred at one half, with the member or members that achieve it. State also whether reversing the direction of the normalisation would change the condition numbers, and why. Give bounds, stationary points, levels and traces to at least six decimals and condition numbers to at least six significant figures. Short lists of numbers are allowed; do not paste the 16 by 16 matrices, their QR factors or per-iteration matrix entries. These are the scalars that determine the final number.

As the final answer, report the condition number of the aspirin problem after two iterations of the accelerated scheme, the inverse gap between the inner bounds, to six significant figures.

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
Output Format Requirements:
Emit <final_answer> immediately, then <reasoning>. Do not write a long derivation before the tags.
You must emit exactly one finite decimal inside <final_answer>...</final_answer>, even if the value is approximate or you are unsure.
Rules:
- The tags are required. Do not omit them or leave them empty.
- The tagged value must be a finite decimal (examples: 0.4847, 0.22, 1.05). Not NaN, not Inf, not a fraction string, not a vector, not prose.
- Put only that one number between the tags. No units, no words, no extra lines.
Keep <reasoning> short (a few hundred words). Show only the few scalars that determine the final number.
Do not paste the input matrices, full coefficient vectors, per-iteration paths, or per-fold candidate tables.
```

## Your task

Implement **all 11 functions** below into a single file at `/app/solution.py`. Requirements:

- Use the EXACT function name and signature shown for each step.
- Define every function at module scope (importable).
- Do NOT include tests, example usage, prints, or a `__main__` block.
- Available libraries: numpy, scipy, sympy, pandas, scikit-learn, networkx.

Start your file with these imports:

```python
import numpy as np
```

### Step 1

sp8_closed_form

Goal
----
Returns the monomial coefficients of the source's closed-form SP8 member of a given index, the tabulated members and their reflections.

```python
def sp8_closed_form(l: int) -> "np.ndarray":
    r"""Returns the monomial coefficients of the source's closed-form SP8 member of a given index, the tabulated members and their reflections.

    The source computes the matrix step function of a symmetric matrix $X_0$ with spectrum in $[0, \lambda_{lumo}]
    \cup [\lambda_{homo}, 1]$ by the recursive expansion $X_i = p_i(X_{i-1})$ with degree-eight component polynomials.
    Each component polynomial is a member $p_{(L, R; a, b)}$ of the SP8 family, $L + R = 7$, written in the source's
    parametrisation $p(x) = s_8 \sum_{k=0}^{7} (-1)^k e_k(s)\, x^{8-k}/(8-k) + s_9$ with $e_k$ the elementary
    symmetric polynomials of the stationary points $s_1, \dots, s_7$, so that $p'(x) = s_8 \prod_k (x - s_k)$. For
    construction bounds $0 \leq a < b \leq 1$ the member with $L$ stationary points in $[0, a]$ and $R$ in $[b, 1]$ is
    the unique solution of the equioscillatory conditions $p(s_k) = 0$ for odd $k \leq L$ and $p(s_k) = p(a)$ for even
    $k \leq L$; $p(s_{L+k}) = 1$ for odd $k \leq R$ and $p(s_{L+k}) = p(b)$ for even $k \leq R$; $p(0) = 0$ if $L$ is
    even and $p(0) = p(a)$ if $L$ is odd; $p(1) = 1$ if $R$ is even and $p(1) = p(b)$ if $R$ is odd; with the ordering
    $0 \leq s_L \leq \dots \leq s_1 \leq a < b \leq s_{L+1} \leq \dots \leq s_7 \leq 1$. At a bound sitting at its
    limit the conditions of that side are replaced by the modified ones: for $a = 0$, $s_k = 0$ for $k \leq L$ and
    $p(0) = 0$; for $b = 1$, $s_{L+k} = 1$ for $k \leq R$ and $p(1) = 1$; with both limits the member is the
    closed-form polynomial $p_{(L,R;0,1)}$ with $p' \propto x^L (x - 1)^R$. The expansion (the source's Algorithm 5.1)
    carries inner bounds $\lambda^{in}_{lumo} \leq \lambda^{in}_{homo}$ and outer bounds $\lambda^{out}_{lumo} \leq
    \lambda^{in}_{lumo}$, $\lambda^{in}_{homo} \leq \lambda^{out}_{homo}$ on the two eigenvalues, with the threshold
    $\kappa = 0.01$: the acceleration bounds are $a = \lambda^{out}_{lumo}$ if $\lambda^{out}_{lumo} > \kappa$ and $a
    = 0$ otherwise, and $b = \lambda^{out}_{homo}$ if $\lambda^{out}_{homo} < 1 - \kappa$ and $b = 1$ otherwise; the
    family is built for $(a, b)$; the member applied at iteration $i$ maximises $p(\lambda^{in}_{homo}) -
    p(\lambda^{in}_{lumo})$ over $L = 0, \dots, 7$ unless both inner bounds lie within $\kappa$ of their limits
    ($\lambda^{in}_{lumo} < \kappa$ and $1 - \kappa < \lambda^{in}_{homo}$), in which case $L = 3$ at odd and $L = 4$
    at even iterations; all four bounds are then mapped through the applied polynomial. The sign test $\mathrm{Tr}[X_i
    - X_i^2] \leq 0$ ends the expansion after any iteration; the parameter-free bound test is applied only once both
    acceleration bounds sit at their limits. The synthetic configurations are $16 \times 16$: from the source's Table
    1 energies (in the order $\lambda_{min}$, $\lambda^{out}_{homo}$, $\lambda^{in}_{homo}$, $\lambda^{in}_{lumo}$,
    $\lambda^{out}_{lumo}$, $\lambda_{max}$) the source's normalisation maps an energy $e$ to $(\lambda_{max} -
    e)/(\lambda_{max} - \lambda_{min})$; the spectrum places $\lambda_{lumo}$ at the midpoint of
    $[\lambda^{out}_{lumo}, \lambda^{in}_{lumo}]$ and $\lambda_{homo}$ at the midpoint of $[\lambda^{in}_{homo},
    \lambda^{out}_{homo}]$, with $n_{occ} = \mathrm{round}(16(1 - \mu))$, $\mu = (\lambda^{in}_{lumo} +
    \lambda^{in}_{homo})/2$, equidistant eigenvalues on $[\lambda_{homo}, 1]$ ($n_{occ}$ of them, endpoints included)
    and $16 - n_{occ}$ on $[0, \lambda_{lumo}]$, assembled on the orthonormal frame $Q$ of the QR factorisation of the
    integer matrix $M_{ij} = (((i+1)(j+2)(i+j+3) + a_{seed}(i+4)(j+5)) \bmod 47) - 23$, $i, j = 0..15$, each column of
    $Q$ sign-fixed so that its first entry of modulus above $10^{-12}$ is positive, and $X_0 =
    Q\,\mathrm{diag}(\lambda)\,Q^T$ symmetrised.

    This step returns the monomial coefficients of the closed-form member $p_{(l, 7-l; 0, 1)}$: the tabulated
    polynomial for $l \geq 4$ ($p_{(4,3)} = -35x^8 + 120x^7 - 140x^6 + 56x^5$, $p_{(5,2)} = 21x^8 - 48x^7 + 28x^6$,
    $p_{(6,1)} = -7x^8 + 8x^7$, $p_{(7,0)} = x^8$) and the reflected polynomial $1 - p_{(7-l, l; 0, 1)}(1 - x)$ for $l
    < 4$, expanded in monomials.

    Args:
        l: integer in 0..7, the number of stationary points of the member in the left interval.

    Returns:
        A numpy float64 array of shape $(9,)$: the monomial coefficients $b_0, \dots, b_8$ of $p(x) = \sum_k b_k x^k$.

    Raises:
        ValueError: if l is not an integer in 0..7.
    """
    return None
```

### Step 2

sp8_from_stationary_points

Goal
----
Expands the source's stationary-point parametrisation of a degree-eight polynomial into monomial coefficients.

```python
def sp8_from_stationary_points(s: "np.ndarray") -> "np.ndarray":
    r"""Expands the source's stationary-point parametrisation of a degree-eight polynomial into monomial coefficients.

    The source computes the matrix step function of a symmetric matrix $X_0$ with spectrum in $[0, \lambda_{lumo}]
    \cup [\lambda_{homo}, 1]$ by the recursive expansion $X_i = p_i(X_{i-1})$ with degree-eight component polynomials.
    Each component polynomial is a member $p_{(L, R; a, b)}$ of the SP8 family, $L + R = 7$, written in the source's
    parametrisation $p(x) = s_8 \sum_{k=0}^{7} (-1)^k e_k(s)\, x^{8-k}/(8-k) + s_9$ with $e_k$ the elementary
    symmetric polynomials of the stationary points $s_1, \dots, s_7$, so that $p'(x) = s_8 \prod_k (x - s_k)$. For
    construction bounds $0 \leq a < b \leq 1$ the member with $L$ stationary points in $[0, a]$ and $R$ in $[b, 1]$ is
    the unique solution of the equioscillatory conditions $p(s_k) = 0$ for odd $k \leq L$ and $p(s_k) = p(a)$ for even
    $k \leq L$; $p(s_{L+k}) = 1$ for odd $k \leq R$ and $p(s_{L+k}) = p(b)$ for even $k \leq R$; $p(0) = 0$ if $L$ is
    even and $p(0) = p(a)$ if $L$ is odd; $p(1) = 1$ if $R$ is even and $p(1) = p(b)$ if $R$ is odd; with the ordering
    $0 \leq s_L \leq \dots \leq s_1 \leq a < b \leq s_{L+1} \leq \dots \leq s_7 \leq 1$. At a bound sitting at its
    limit the conditions of that side are replaced by the modified ones: for $a = 0$, $s_k = 0$ for $k \leq L$ and
    $p(0) = 0$; for $b = 1$, $s_{L+k} = 1$ for $k \leq R$ and $p(1) = 1$; with both limits the member is the
    closed-form polynomial $p_{(L,R;0,1)}$ with $p' \propto x^L (x - 1)^R$. The expansion (the source's Algorithm 5.1)
    carries inner bounds $\lambda^{in}_{lumo} \leq \lambda^{in}_{homo}$ and outer bounds $\lambda^{out}_{lumo} \leq
    \lambda^{in}_{lumo}$, $\lambda^{in}_{homo} \leq \lambda^{out}_{homo}$ on the two eigenvalues, with the threshold
    $\kappa = 0.01$: the acceleration bounds are $a = \lambda^{out}_{lumo}$ if $\lambda^{out}_{lumo} > \kappa$ and $a
    = 0$ otherwise, and $b = \lambda^{out}_{homo}$ if $\lambda^{out}_{homo} < 1 - \kappa$ and $b = 1$ otherwise; the
    family is built for $(a, b)$; the member applied at iteration $i$ maximises $p(\lambda^{in}_{homo}) -
    p(\lambda^{in}_{lumo})$ over $L = 0, \dots, 7$ unless both inner bounds lie within $\kappa$ of their limits
    ($\lambda^{in}_{lumo} < \kappa$ and $1 - \kappa < \lambda^{in}_{homo}$), in which case $L = 3$ at odd and $L = 4$
    at even iterations; all four bounds are then mapped through the applied polynomial. The sign test $\mathrm{Tr}[X_i
    - X_i^2] \leq 0$ ends the expansion after any iteration; the parameter-free bound test is applied only once both
    acceleration bounds sit at their limits. The synthetic configurations are $16 \times 16$: from the source's Table
    1 energies (in the order $\lambda_{min}$, $\lambda^{out}_{homo}$, $\lambda^{in}_{homo}$, $\lambda^{in}_{lumo}$,
    $\lambda^{out}_{lumo}$, $\lambda_{max}$) the source's normalisation maps an energy $e$ to $(\lambda_{max} -
    e)/(\lambda_{max} - \lambda_{min})$; the spectrum places $\lambda_{lumo}$ at the midpoint of
    $[\lambda^{out}_{lumo}, \lambda^{in}_{lumo}]$ and $\lambda_{homo}$ at the midpoint of $[\lambda^{in}_{homo},
    \lambda^{out}_{homo}]$, with $n_{occ} = \mathrm{round}(16(1 - \mu))$, $\mu = (\lambda^{in}_{lumo} +
    \lambda^{in}_{homo})/2$, equidistant eigenvalues on $[\lambda_{homo}, 1]$ ($n_{occ}$ of them, endpoints included)
    and $16 - n_{occ}$ on $[0, \lambda_{lumo}]$, assembled on the orthonormal frame $Q$ of the QR factorisation of the
    integer matrix $M_{ij} = (((i+1)(j+2)(i+j+3) + a_{seed}(i+4)(j+5)) \bmod 47) - 23$, $i, j = 0..15$, each column of
    $Q$ sign-fixed so that its first entry of modulus above $10^{-12}$ is positive, and $X_0 =
    Q\,\mathrm{diag}(\lambda)\,Q^T$ symmetrised.

    This step expands the source's parametrisation into monomials: $p(x) = s_8 \sum_{k=0}^{7} (-1)^k e_k\,
    x^{8-k}/(8-k) + s_9$ with $e_0 = 1$ and $e_k$ the $k$-th elementary symmetric polynomial of $s_1, \dots, s_7$.

    Args:
        s: array of shape (9,), the parameters $s_1, \dots, s_7$ (stationary points), $s_8$ (scale) and $s_9$
        (offset).

    Returns:
        A numpy float64 array of shape $(9,)$: the monomial coefficients $b_0, \dots, b_8$.

    Raises:
        ValueError: if s does not hold nine finite values.
    """
    return None
```

### Step 3

equioscillation_residual

Goal
----
Evaluates the source's nine equioscillatory conditions, with the modified conditions at the limits, at a given parameter vector.

```python
def equioscillation_residual(s: "np.ndarray", L: int, lam_lumo: float, lam_homo: float) -> "np.ndarray":
    r"""Evaluates the source's nine equioscillatory conditions, with the modified conditions at the limits, at a given parameter vector.

    The source computes the matrix step function of a symmetric matrix $X_0$ with spectrum in $[0, \lambda_{lumo}]
    \cup [\lambda_{homo}, 1]$ by the recursive expansion $X_i = p_i(X_{i-1})$ with degree-eight component polynomials.
    Each component polynomial is a member $p_{(L, R; a, b)}$ of the SP8 family, $L + R = 7$, written in the source's
    parametrisation $p(x) = s_8 \sum_{k=0}^{7} (-1)^k e_k(s)\, x^{8-k}/(8-k) + s_9$ with $e_k$ the elementary
    symmetric polynomials of the stationary points $s_1, \dots, s_7$, so that $p'(x) = s_8 \prod_k (x - s_k)$. For
    construction bounds $0 \leq a < b \leq 1$ the member with $L$ stationary points in $[0, a]$ and $R$ in $[b, 1]$ is
    the unique solution of the equioscillatory conditions $p(s_k) = 0$ for odd $k \leq L$ and $p(s_k) = p(a)$ for even
    $k \leq L$; $p(s_{L+k}) = 1$ for odd $k \leq R$ and $p(s_{L+k}) = p(b)$ for even $k \leq R$; $p(0) = 0$ if $L$ is
    even and $p(0) = p(a)$ if $L$ is odd; $p(1) = 1$ if $R$ is even and $p(1) = p(b)$ if $R$ is odd; with the ordering
    $0 \leq s_L \leq \dots \leq s_1 \leq a < b \leq s_{L+1} \leq \dots \leq s_7 \leq 1$. At a bound sitting at its
    limit the conditions of that side are replaced by the modified ones: for $a = 0$, $s_k = 0$ for $k \leq L$ and
    $p(0) = 0$; for $b = 1$, $s_{L+k} = 1$ for $k \leq R$ and $p(1) = 1$; with both limits the member is the
    closed-form polynomial $p_{(L,R;0,1)}$ with $p' \propto x^L (x - 1)^R$. The expansion (the source's Algorithm 5.1)
    carries inner bounds $\lambda^{in}_{lumo} \leq \lambda^{in}_{homo}$ and outer bounds $\lambda^{out}_{lumo} \leq
    \lambda^{in}_{lumo}$, $\lambda^{in}_{homo} \leq \lambda^{out}_{homo}$ on the two eigenvalues, with the threshold
    $\kappa = 0.01$: the acceleration bounds are $a = \lambda^{out}_{lumo}$ if $\lambda^{out}_{lumo} > \kappa$ and $a
    = 0$ otherwise, and $b = \lambda^{out}_{homo}$ if $\lambda^{out}_{homo} < 1 - \kappa$ and $b = 1$ otherwise; the
    family is built for $(a, b)$; the member applied at iteration $i$ maximises $p(\lambda^{in}_{homo}) -
    p(\lambda^{in}_{lumo})$ over $L = 0, \dots, 7$ unless both inner bounds lie within $\kappa$ of their limits
    ($\lambda^{in}_{lumo} < \kappa$ and $1 - \kappa < \lambda^{in}_{homo}$), in which case $L = 3$ at odd and $L = 4$
    at even iterations; all four bounds are then mapped through the applied polynomial. The sign test $\mathrm{Tr}[X_i
    - X_i^2] \leq 0$ ends the expansion after any iteration; the parameter-free bound test is applied only once both
    acceleration bounds sit at their limits. The synthetic configurations are $16 \times 16$: from the source's Table
    1 energies (in the order $\lambda_{min}$, $\lambda^{out}_{homo}$, $\lambda^{in}_{homo}$, $\lambda^{in}_{lumo}$,
    $\lambda^{out}_{lumo}$, $\lambda_{max}$) the source's normalisation maps an energy $e$ to $(\lambda_{max} -
    e)/(\lambda_{max} - \lambda_{min})$; the spectrum places $\lambda_{lumo}$ at the midpoint of
    $[\lambda^{out}_{lumo}, \lambda^{in}_{lumo}]$ and $\lambda_{homo}$ at the midpoint of $[\lambda^{in}_{homo},
    \lambda^{out}_{homo}]$, with $n_{occ} = \mathrm{round}(16(1 - \mu))$, $\mu = (\lambda^{in}_{lumo} +
    \lambda^{in}_{homo})/2$, equidistant eigenvalues on $[\lambda_{homo}, 1]$ ($n_{occ}$ of them, endpoints included)
    and $16 - n_{occ}$ on $[0, \lambda_{lumo}]$, assembled on the orthonormal frame $Q$ of the QR factorisation of the
    integer matrix $M_{ij} = (((i+1)(j+2)(i+j+3) + a_{seed}(i+4)(j+5)) \bmod 47) - 23$, $i, j = 0..15$, each column of
    $Q$ sign-fixed so that its first entry of modulus above $10^{-12}$ is positive, and $X_0 =
    Q\,\mathrm{diag}(\lambda)\,Q^T$ symmetrised.

    This step evaluates the source's system of nine equations for the member $p_{(L, 7-L; \lambda_{lumo},
    \lambda_{homo})}$ at the parameters s, in the order: the $L$ left conditions for $k = 1, \dots, L$ (each written
    as $p(s_k)$ minus its level), the $R = 7 - L$ right conditions for $k = 1, \dots, R$ ($p(s_{L+k})$ minus its
    level), the condition at $x = 0$ ($p(0)$ minus its level) and the condition at $x = 1$ ($p(1)$ minus its level).
    At $\lambda_{lumo} = 0$ the left conditions are $s_k - 0$ for $k \leq L$ and $p(0) - 0$; at $\lambda_{homo} = 1$
    the right conditions are $s_{L+k} - 1$ for $k \leq R$ and $p(1) - 1$.

    Args:
        s: array of shape (9,), the parameters of the parametrisation.
        L: integer in 0..7, the number of stationary points in the left interval.
        lam_lumo: float in [0, 1), the construction bound of the left interval.
        lam_homo: float in (0, 1], the construction bound of the right interval, above lam_lumo.

    Returns:
        A numpy float64 array of shape $(9,)$: the residuals in that order.

    Raises:
        ValueError: if s does not hold nine finite values, if L is not an integer in 0..7, or if the bounds do not
        satisfy $0 \leq \lambda_{lumo} < \lambda_{homo} \leq 1$.
    """
    return None
```

### Step 4

sp8_equioscillatory

Goal
----
Solves the equioscillatory conditions for a member of the SP8 family at given construction bounds and returns its ordered parameters.

```python
def sp8_equioscillatory(L: int, lam_lumo: float, lam_homo: float) -> "np.ndarray":
    r"""Solves the equioscillatory conditions for a member of the SP8 family at given construction bounds and returns its ordered parameters.

    The source computes the matrix step function of a symmetric matrix $X_0$ with spectrum in $[0, \lambda_{lumo}]
    \cup [\lambda_{homo}, 1]$ by the recursive expansion $X_i = p_i(X_{i-1})$ with degree-eight component polynomials.
    Each component polynomial is a member $p_{(L, R; a, b)}$ of the SP8 family, $L + R = 7$, written in the source's
    parametrisation $p(x) = s_8 \sum_{k=0}^{7} (-1)^k e_k(s)\, x^{8-k}/(8-k) + s_9$ with $e_k$ the elementary
    symmetric polynomials of the stationary points $s_1, \dots, s_7$, so that $p'(x) = s_8 \prod_k (x - s_k)$. For
    construction bounds $0 \leq a < b \leq 1$ the member with $L$ stationary points in $[0, a]$ and $R$ in $[b, 1]$ is
    the unique solution of the equioscillatory conditions $p(s_k) = 0$ for odd $k \leq L$ and $p(s_k) = p(a)$ for even
    $k \leq L$; $p(s_{L+k}) = 1$ for odd $k \leq R$ and $p(s_{L+k}) = p(b)$ for even $k \leq R$; $p(0) = 0$ if $L$ is
    even and $p(0) = p(a)$ if $L$ is odd; $p(1) = 1$ if $R$ is even and $p(1) = p(b)$ if $R$ is odd; with the ordering
    $0 \leq s_L \leq \dots \leq s_1 \leq a < b \leq s_{L+1} \leq \dots \leq s_7 \leq 1$. At a bound sitting at its
    limit the conditions of that side are replaced by the modified ones: for $a = 0$, $s_k = 0$ for $k \leq L$ and
    $p(0) = 0$; for $b = 1$, $s_{L+k} = 1$ for $k \leq R$ and $p(1) = 1$; with both limits the member is the
    closed-form polynomial $p_{(L,R;0,1)}$ with $p' \propto x^L (x - 1)^R$. The expansion (the source's Algorithm 5.1)
    carries inner bounds $\lambda^{in}_{lumo} \leq \lambda^{in}_{homo}$ and outer bounds $\lambda^{out}_{lumo} \leq
    \lambda^{in}_{lumo}$, $\lambda^{in}_{homo} \leq \lambda^{out}_{homo}$ on the two eigenvalues, with the threshold
    $\kappa = 0.01$: the acceleration bounds are $a = \lambda^{out}_{lumo}$ if $\lambda^{out}_{lumo} > \kappa$ and $a
    = 0$ otherwise, and $b = \lambda^{out}_{homo}$ if $\lambda^{out}_{homo} < 1 - \kappa$ and $b = 1$ otherwise; the
    family is built for $(a, b)$; the member applied at iteration $i$ maximises $p(\lambda^{in}_{homo}) -
    p(\lambda^{in}_{lumo})$ over $L = 0, \dots, 7$ unless both inner bounds lie within $\kappa$ of their limits
    ($\lambda^{in}_{lumo} < \kappa$ and $1 - \kappa < \lambda^{in}_{homo}$), in which case $L = 3$ at odd and $L = 4$
    at even iterations; all four bounds are then mapped through the applied polynomial. The sign test $\mathrm{Tr}[X_i
    - X_i^2] \leq 0$ ends the expansion after any iteration; the parameter-free bound test is applied only once both
    acceleration bounds sit at their limits. The synthetic configurations are $16 \times 16$: from the source's Table
    1 energies (in the order $\lambda_{min}$, $\lambda^{out}_{homo}$, $\lambda^{in}_{homo}$, $\lambda^{in}_{lumo}$,
    $\lambda^{out}_{lumo}$, $\lambda_{max}$) the source's normalisation maps an energy $e$ to $(\lambda_{max} -
    e)/(\lambda_{max} - \lambda_{min})$; the spectrum places $\lambda_{lumo}$ at the midpoint of
    $[\lambda^{out}_{lumo}, \lambda^{in}_{lumo}]$ and $\lambda_{homo}$ at the midpoint of $[\lambda^{in}_{homo},
    \lambda^{out}_{homo}]$, with $n_{occ} = \mathrm{round}(16(1 - \mu))$, $\mu = (\lambda^{in}_{lumo} +
    \lambda^{in}_{homo})/2$, equidistant eigenvalues on $[\lambda_{homo}, 1]$ ($n_{occ}$ of them, endpoints included)
    and $16 - n_{occ}$ on $[0, \lambda_{lumo}]$, assembled on the orthonormal frame $Q$ of the QR factorisation of the
    integer matrix $M_{ij} = (((i+1)(j+2)(i+j+3) + a_{seed}(i+4)(j+5)) \bmod 47) - 23$, $i, j = 0..15$, each column of
    $Q$ sign-fixed so that its first entry of modulus above $10^{-12}$ is positive, and $X_0 =
    Q\,\mathrm{diag}(\lambda)\,Q^T$ symmetrised.

    This step solves the nine equioscillatory conditions for the member $p_{(L, 7-L; \lambda_{lumo},
    \lambda_{homo})}$, the modified conditions applying at a bound sitting at its limit, and returns the parameters of
    the unique solution that satisfies the ordering $0 \leq s_L \leq \dots \leq s_1 \leq \lambda_{lumo} <
    \lambda_{homo} \leq s_{L+1} \leq \dots \leq s_7 \leq 1$ (a solution violating the ordering is not the member). The
    construction bounds of the expansion are either at their limits or at least $\kappa = 0.01$ away from them; the
    solution must satisfy every condition to $10^{-9}$ relative to the amplitude $p(\lambda_{lumo}) - p(s_1)$ or
    $p(s_{L+1}) - p(\lambda_{homo})$ of its interval, or better.

    Args:
        L: integer in 0..7, the number of stationary points in the left interval.
        lam_lumo: float in [0, 1), the construction bound of the left interval.
        lam_homo: float in (0, 1], the construction bound of the right interval, above lam_lumo.

    Returns:
        A numpy float64 array of shape $(9,)$: $s_1, \dots, s_7, s_8, s_9$.

    Raises:
        ValueError: if L is not an integer in 0..7, if the bounds do not satisfy $0 \leq \lambda_{lumo} <
        \lambda_{homo} \leq 1$, or if no ordered solution is found.
    """
    return None
```

### Step 5

sastre_coefficients

Goal
----
Evaluates the real coefficients of the source's three-multiplication evaluation scheme for a degree-eight polynomial.

```python
def sastre_coefficients(b: "np.ndarray") -> "np.ndarray":
    r"""Evaluates the real coefficients of the source's three-multiplication evaluation scheme for a degree-eight polynomial.

    The source computes the matrix step function of a symmetric matrix $X_0$ with spectrum in $[0, \lambda_{lumo}]
    \cup [\lambda_{homo}, 1]$ by the recursive expansion $X_i = p_i(X_{i-1})$ with degree-eight component polynomials.
    Each component polynomial is a member $p_{(L, R; a, b)}$ of the SP8 family, $L + R = 7$, written in the source's
    parametrisation $p(x) = s_8 \sum_{k=0}^{7} (-1)^k e_k(s)\, x^{8-k}/(8-k) + s_9$ with $e_k$ the elementary
    symmetric polynomials of the stationary points $s_1, \dots, s_7$, so that $p'(x) = s_8 \prod_k (x - s_k)$. For
    construction bounds $0 \leq a < b \leq 1$ the member with $L$ stationary points in $[0, a]$ and $R$ in $[b, 1]$ is
    the unique solution of the equioscillatory conditions $p(s_k) = 0$ for odd $k \leq L$ and $p(s_k) = p(a)$ for even
    $k \leq L$; $p(s_{L+k}) = 1$ for odd $k \leq R$ and $p(s_{L+k}) = p(b)$ for even $k \leq R$; $p(0) = 0$ if $L$ is
    even and $p(0) = p(a)$ if $L$ is odd; $p(1) = 1$ if $R$ is even and $p(1) = p(b)$ if $R$ is odd; with the ordering
    $0 \leq s_L \leq \dots \leq s_1 \leq a < b \leq s_{L+1} \leq \dots \leq s_7 \leq 1$. At a bound sitting at its
    limit the conditions of that side are replaced by the modified ones: for $a = 0$, $s_k = 0$ for $k \leq L$ and
    $p(0) = 0$; for $b = 1$, $s_{L+k} = 1$ for $k \leq R$ and $p(1) = 1$; with both limits the member is the
    closed-form polynomial $p_{(L,R;0,1)}$ with $p' \propto x^L (x - 1)^R$. The expansion (the source's Algorithm 5.1)
    carries inner bounds $\lambda^{in}_{lumo} \leq \lambda^{in}_{homo}$ and outer bounds $\lambda^{out}_{lumo} \leq
    \lambda^{in}_{lumo}$, $\lambda^{in}_{homo} \leq \lambda^{out}_{homo}$ on the two eigenvalues, with the threshold
    $\kappa = 0.01$: the acceleration bounds are $a = \lambda^{out}_{lumo}$ if $\lambda^{out}_{lumo} > \kappa$ and $a
    = 0$ otherwise, and $b = \lambda^{out}_{homo}$ if $\lambda^{out}_{homo} < 1 - \kappa$ and $b = 1$ otherwise; the
    family is built for $(a, b)$; the member applied at iteration $i$ maximises $p(\lambda^{in}_{homo}) -
    p(\lambda^{in}_{lumo})$ over $L = 0, \dots, 7$ unless both inner bounds lie within $\kappa$ of their limits
    ($\lambda^{in}_{lumo} < \kappa$ and $1 - \kappa < \lambda^{in}_{homo}$), in which case $L = 3$ at odd and $L = 4$
    at even iterations; all four bounds are then mapped through the applied polynomial. The sign test $\mathrm{Tr}[X_i
    - X_i^2] \leq 0$ ends the expansion after any iteration; the parameter-free bound test is applied only once both
    acceleration bounds sit at their limits. The synthetic configurations are $16 \times 16$: from the source's Table
    1 energies (in the order $\lambda_{min}$, $\lambda^{out}_{homo}$, $\lambda^{in}_{homo}$, $\lambda^{in}_{lumo}$,
    $\lambda^{out}_{lumo}$, $\lambda_{max}$) the source's normalisation maps an energy $e$ to $(\lambda_{max} -
    e)/(\lambda_{max} - \lambda_{min})$; the spectrum places $\lambda_{lumo}$ at the midpoint of
    $[\lambda^{out}_{lumo}, \lambda^{in}_{lumo}]$ and $\lambda_{homo}$ at the midpoint of $[\lambda^{in}_{homo},
    \lambda^{out}_{homo}]$, with $n_{occ} = \mathrm{round}(16(1 - \mu))$, $\mu = (\lambda^{in}_{lumo} +
    \lambda^{in}_{homo})/2$, equidistant eigenvalues on $[\lambda_{homo}, 1]$ ($n_{occ}$ of them, endpoints included)
    and $16 - n_{occ}$ on $[0, \lambda_{lumo}]$, assembled on the orthonormal frame $Q$ of the QR factorisation of the
    integer matrix $M_{ij} = (((i+1)(j+2)(i+j+3) + a_{seed}(i+4)(j+5)) \bmod 47) - 23$, $i, j = 0..15$, each column of
    $Q$ sign-fixed so that its first entry of modulus above $10^{-12}$ is positive, and $X_0 =
    Q\,\mathrm{diag}(\lambda)\,Q^T$ symmetrised.

    This step evaluates the source's real evaluation coefficients of the three-multiplication scheme: $f_4 = b_8$,
    $c_1 = b_7/(2 f_4)$, $t_2 = b_6/f_4 - c_1^2$, $t_1 = b_5/f_4 - c_1 t_2$, $d_0 = (1 - t_2^2 + 4 b_4/f_4 - 4 c_1
    t_1)/4$, $e_2 = (t_2 + 1)/2$, $d_2 = (t_2 - 1)/2$, $e_1 = c_1 d_0 + t_1 e_2 - b_3/f_4$, $d_1 = t_1 - e_1$, $f_2 =
    b_2 - f_4 (d_0 e_2 + d_1 e_1)$, $f_1 = b_1 - f_4 d_0 e_1$, $f_0 = b_0$.

    Args:
        b: array of shape (9,), the monomial coefficients $b_0, \dots, b_8$ of a degree-eight polynomial with $b_8$
        nonzero.

    Returns:
        A numpy float64 array of shape $(10,)$: $c_1, d_0, d_1, d_2, e_1, e_2, f_0, f_1, f_2, f_4$ in this order.

    Raises:
        ValueError: if b does not have shape (9,) or its leading coefficient is zero.
    """
    return None
```

### Step 6

workspace_scalars

Goal
----
Evaluates the four workspace rearrangement scalars of the source's three-matrix evaluation algorithm.

```python
def workspace_scalars(coeffs: "np.ndarray") -> "np.ndarray":
    r"""Evaluates the four workspace rearrangement scalars of the source's three-matrix evaluation algorithm.

    The source computes the matrix step function of a symmetric matrix $X_0$ with spectrum in $[0, \lambda_{lumo}]
    \cup [\lambda_{homo}, 1]$ by the recursive expansion $X_i = p_i(X_{i-1})$ with degree-eight component polynomials.
    Each component polynomial is a member $p_{(L, R; a, b)}$ of the SP8 family, $L + R = 7$, written in the source's
    parametrisation $p(x) = s_8 \sum_{k=0}^{7} (-1)^k e_k(s)\, x^{8-k}/(8-k) + s_9$ with $e_k$ the elementary
    symmetric polynomials of the stationary points $s_1, \dots, s_7$, so that $p'(x) = s_8 \prod_k (x - s_k)$. For
    construction bounds $0 \leq a < b \leq 1$ the member with $L$ stationary points in $[0, a]$ and $R$ in $[b, 1]$ is
    the unique solution of the equioscillatory conditions $p(s_k) = 0$ for odd $k \leq L$ and $p(s_k) = p(a)$ for even
    $k \leq L$; $p(s_{L+k}) = 1$ for odd $k \leq R$ and $p(s_{L+k}) = p(b)$ for even $k \leq R$; $p(0) = 0$ if $L$ is
    even and $p(0) = p(a)$ if $L$ is odd; $p(1) = 1$ if $R$ is even and $p(1) = p(b)$ if $R$ is odd; with the ordering
    $0 \leq s_L \leq \dots \leq s_1 \leq a < b \leq s_{L+1} \leq \dots \leq s_7 \leq 1$. At a bound sitting at its
    limit the conditions of that side are replaced by the modified ones: for $a = 0$, $s_k = 0$ for $k \leq L$ and
    $p(0) = 0$; for $b = 1$, $s_{L+k} = 1$ for $k \leq R$ and $p(1) = 1$; with both limits the member is the
    closed-form polynomial $p_{(L,R;0,1)}$ with $p' \propto x^L (x - 1)^R$. The expansion (the source's Algorithm 5.1)
    carries inner bounds $\lambda^{in}_{lumo} \leq \lambda^{in}_{homo}$ and outer bounds $\lambda^{out}_{lumo} \leq
    \lambda^{in}_{lumo}$, $\lambda^{in}_{homo} \leq \lambda^{out}_{homo}$ on the two eigenvalues, with the threshold
    $\kappa = 0.01$: the acceleration bounds are $a = \lambda^{out}_{lumo}$ if $\lambda^{out}_{lumo} > \kappa$ and $a
    = 0$ otherwise, and $b = \lambda^{out}_{homo}$ if $\lambda^{out}_{homo} < 1 - \kappa$ and $b = 1$ otherwise; the
    family is built for $(a, b)$; the member applied at iteration $i$ maximises $p(\lambda^{in}_{homo}) -
    p(\lambda^{in}_{lumo})$ over $L = 0, \dots, 7$ unless both inner bounds lie within $\kappa$ of their limits
    ($\lambda^{in}_{lumo} < \kappa$ and $1 - \kappa < \lambda^{in}_{homo}$), in which case $L = 3$ at odd and $L = 4$
    at even iterations; all four bounds are then mapped through the applied polynomial. The sign test $\mathrm{Tr}[X_i
    - X_i^2] \leq 0$ ends the expansion after any iteration; the parameter-free bound test is applied only once both
    acceleration bounds sit at their limits. The synthetic configurations are $16 \times 16$: from the source's Table
    1 energies (in the order $\lambda_{min}$, $\lambda^{out}_{homo}$, $\lambda^{in}_{homo}$, $\lambda^{in}_{lumo}$,
    $\lambda^{out}_{lumo}$, $\lambda_{max}$) the source's normalisation maps an energy $e$ to $(\lambda_{max} -
    e)/(\lambda_{max} - \lambda_{min})$; the spectrum places $\lambda_{lumo}$ at the midpoint of
    $[\lambda^{out}_{lumo}, \lambda^{in}_{lumo}]$ and $\lambda_{homo}$ at the midpoint of $[\lambda^{in}_{homo},
    \lambda^{out}_{homo}]$, with $n_{occ} = \mathrm{round}(16(1 - \mu))$, $\mu = (\lambda^{in}_{lumo} +
    \lambda^{in}_{homo})/2$, equidistant eigenvalues on $[\lambda_{homo}, 1]$ ($n_{occ}$ of them, endpoints included)
    and $16 - n_{occ}$ on $[0, \lambda_{lumo}]$, assembled on the orthonormal frame $Q$ of the QR factorisation of the
    integer matrix $M_{ij} = (((i+1)(j+2)(i+j+3) + a_{seed}(i+4)(j+5)) \bmod 47) - 23$, $i, j = 0..15$, each column of
    $Q$ sign-fixed so that its first entry of modulus above $10^{-12}$ is positive, and $X_0 =
    Q\,\mathrm{diag}(\lambda)\,Q^T$ symmetrised.

    This step evaluates the four scalars of the source's workspace rearrangement between the second and the third
    multiplication: $r_1 = d_1 - (c_1/2)(d_2 - c_1^2/4)$, $r_2 = d_2 - c_1^2/4$, $r_3 = e_1 - d_1 - c_1/2$, $r_4 = f_1
    - f_2 (e_1 - d_1)$.

    Args:
        coeffs: array of shape (10,), the evaluation coefficients $c_1, d_0, d_1, d_2, e_1, e_2, f_0, f_1, f_2, f_4$.

    Returns:
        A numpy float64 array of shape $(4,)$: $r_1, r_2, r_3, r_4$.

    Raises:
        ValueError: if coeffs does not have shape (10,) or is not finite.
    """
    return None
```

### Step 7

evaluate_p8

Goal
----
Evaluates a degree-eight polynomial at a symmetric matrix with the source's three-multiplication, three-matrix algorithm.

```python
def evaluate_p8(X: "np.ndarray", b: "np.ndarray") -> "np.ndarray":
    r"""Evaluates a degree-eight polynomial at a symmetric matrix with the source's three-multiplication, three-matrix algorithm.

    The source computes the matrix step function of a symmetric matrix $X_0$ with spectrum in $[0, \lambda_{lumo}]
    \cup [\lambda_{homo}, 1]$ by the recursive expansion $X_i = p_i(X_{i-1})$ with degree-eight component polynomials.
    Each component polynomial is a member $p_{(L, R; a, b)}$ of the SP8 family, $L + R = 7$, written in the source's
    parametrisation $p(x) = s_8 \sum_{k=0}^{7} (-1)^k e_k(s)\, x^{8-k}/(8-k) + s_9$ with $e_k$ the elementary
    symmetric polynomials of the stationary points $s_1, \dots, s_7$, so that $p'(x) = s_8 \prod_k (x - s_k)$. For
    construction bounds $0 \leq a < b \leq 1$ the member with $L$ stationary points in $[0, a]$ and $R$ in $[b, 1]$ is
    the unique solution of the equioscillatory conditions $p(s_k) = 0$ for odd $k \leq L$ and $p(s_k) = p(a)$ for even
    $k \leq L$; $p(s_{L+k}) = 1$ for odd $k \leq R$ and $p(s_{L+k}) = p(b)$ for even $k \leq R$; $p(0) = 0$ if $L$ is
    even and $p(0) = p(a)$ if $L$ is odd; $p(1) = 1$ if $R$ is even and $p(1) = p(b)$ if $R$ is odd; with the ordering
    $0 \leq s_L \leq \dots \leq s_1 \leq a < b \leq s_{L+1} \leq \dots \leq s_7 \leq 1$. At a bound sitting at its
    limit the conditions of that side are replaced by the modified ones: for $a = 0$, $s_k = 0$ for $k \leq L$ and
    $p(0) = 0$; for $b = 1$, $s_{L+k} = 1$ for $k \leq R$ and $p(1) = 1$; with both limits the member is the
    closed-form polynomial $p_{(L,R;0,1)}$ with $p' \propto x^L (x - 1)^R$. The expansion (the source's Algorithm 5.1)
    carries inner bounds $\lambda^{in}_{lumo} \leq \lambda^{in}_{homo}$ and outer bounds $\lambda^{out}_{lumo} \leq
    \lambda^{in}_{lumo}$, $\lambda^{in}_{homo} \leq \lambda^{out}_{homo}$ on the two eigenvalues, with the threshold
    $\kappa = 0.01$: the acceleration bounds are $a = \lambda^{out}_{lumo}$ if $\lambda^{out}_{lumo} > \kappa$ and $a
    = 0$ otherwise, and $b = \lambda^{out}_{homo}$ if $\lambda^{out}_{homo} < 1 - \kappa$ and $b = 1$ otherwise; the
    family is built for $(a, b)$; the member applied at iteration $i$ maximises $p(\lambda^{in}_{homo}) -
    p(\lambda^{in}_{lumo})$ over $L = 0, \dots, 7$ unless both inner bounds lie within $\kappa$ of their limits
    ($\lambda^{in}_{lumo} < \kappa$ and $1 - \kappa < \lambda^{in}_{homo}$), in which case $L = 3$ at odd and $L = 4$
    at even iterations; all four bounds are then mapped through the applied polynomial. The sign test $\mathrm{Tr}[X_i
    - X_i^2] \leq 0$ ends the expansion after any iteration; the parameter-free bound test is applied only once both
    acceleration bounds sit at their limits. The synthetic configurations are $16 \times 16$: from the source's Table
    1 energies (in the order $\lambda_{min}$, $\lambda^{out}_{homo}$, $\lambda^{in}_{homo}$, $\lambda^{in}_{lumo}$,
    $\lambda^{out}_{lumo}$, $\lambda_{max}$) the source's normalisation maps an energy $e$ to $(\lambda_{max} -
    e)/(\lambda_{max} - \lambda_{min})$; the spectrum places $\lambda_{lumo}$ at the midpoint of
    $[\lambda^{out}_{lumo}, \lambda^{in}_{lumo}]$ and $\lambda_{homo}$ at the midpoint of $[\lambda^{in}_{homo},
    \lambda^{out}_{homo}]$, with $n_{occ} = \mathrm{round}(16(1 - \mu))$, $\mu = (\lambda^{in}_{lumo} +
    \lambda^{in}_{homo})/2$, equidistant eigenvalues on $[\lambda_{homo}, 1]$ ($n_{occ}$ of them, endpoints included)
    and $16 - n_{occ}$ on $[0, \lambda_{lumo}]$, assembled on the orthonormal frame $Q$ of the QR factorisation of the
    integer matrix $M_{ij} = (((i+1)(j+2)(i+j+3) + a_{seed}(i+4)(j+5)) \bmod 47) - 23$, $i, j = 0..15$, each column of
    $Q$ sign-fixed so that its first entry of modulus above $10^{-12}$ is positive, and $X_0 =
    Q\,\mathrm{diag}(\lambda)\,Q^T$ symmetrised.

    This step evaluates $p(X)$ with the source's three-multiplication, three-matrix algorithm: with the scheme
    coefficients and the rearrangement scalars, $M_1 = X$, $M_2 = M_1 M_1 + (c_1/2) M_1$, $M_3 = M_2 M_2 + r_1 M_1 +
    r_2 M_2$, $M_2 \leftarrow M_2 + r_3 M_1$, $M_1 \leftarrow r_4 M_1 + f_2 M_2 + f_0 I$, $M_2 \leftarrow M_2 + M_3$,
    $M_3 \leftarrow M_3 + d_0 I$, $p(X) = f_4 M_2 M_3 + M_1$, symmetrised as $(P + P^T)/2$. The result equals the
    polynomial $\sum_k b_k X^k$ to floating-point accuracy.

    Args:
        X: array of shape (n, n), a symmetric matrix.
        b: array of shape (9,), monomial coefficients with $b_8$ nonzero.

    Returns:
        A numpy float64 array of shape $(n, n)$: $p(X)$ symmetrised.

    Raises:
        ValueError: if X is not square, if b does not have shape (9,), or if its leading coefficient is zero.
    """
    return None
```

### Step 8

select_member

Goal
----
Performs the polynomial choice of one iteration of the source's accelerated scheme from the four bounds, the iteration index and the threshold.

```python
def select_member(bounds: "np.ndarray", i: int, kappa: float) -> "np.ndarray":
    r"""Performs the polynomial choice of one iteration of the source's accelerated scheme from the four bounds, the iteration index and the threshold.

    The source computes the matrix step function of a symmetric matrix $X_0$ with spectrum in $[0, \lambda_{lumo}]
    \cup [\lambda_{homo}, 1]$ by the recursive expansion $X_i = p_i(X_{i-1})$ with degree-eight component polynomials.
    Each component polynomial is a member $p_{(L, R; a, b)}$ of the SP8 family, $L + R = 7$, written in the source's
    parametrisation $p(x) = s_8 \sum_{k=0}^{7} (-1)^k e_k(s)\, x^{8-k}/(8-k) + s_9$ with $e_k$ the elementary
    symmetric polynomials of the stationary points $s_1, \dots, s_7$, so that $p'(x) = s_8 \prod_k (x - s_k)$. For
    construction bounds $0 \leq a < b \leq 1$ the member with $L$ stationary points in $[0, a]$ and $R$ in $[b, 1]$ is
    the unique solution of the equioscillatory conditions $p(s_k) = 0$ for odd $k \leq L$ and $p(s_k) = p(a)$ for even
    $k \leq L$; $p(s_{L+k}) = 1$ for odd $k \leq R$ and $p(s_{L+k}) = p(b)$ for even $k \leq R$; $p(0) = 0$ if $L$ is
    even and $p(0) = p(a)$ if $L$ is odd; $p(1) = 1$ if $R$ is even and $p(1) = p(b)$ if $R$ is odd; with the ordering
    $0 \leq s_L \leq \dots \leq s_1 \leq a < b \leq s_{L+1} \leq \dots \leq s_7 \leq 1$. At a bound sitting at its
    limit the conditions of that side are replaced by the modified ones: for $a = 0$, $s_k = 0$ for $k \leq L$ and
    $p(0) = 0$; for $b = 1$, $s_{L+k} = 1$ for $k \leq R$ and $p(1) = 1$; with both limits the member is the
    closed-form polynomial $p_{(L,R;0,1)}$ with $p' \propto x^L (x - 1)^R$. The expansion (the source's Algorithm 5.1)
    carries inner bounds $\lambda^{in}_{lumo} \leq \lambda^{in}_{homo}$ and outer bounds $\lambda^{out}_{lumo} \leq
    \lambda^{in}_{lumo}$, $\lambda^{in}_{homo} \leq \lambda^{out}_{homo}$ on the two eigenvalues, with the threshold
    $\kappa = 0.01$: the acceleration bounds are $a = \lambda^{out}_{lumo}$ if $\lambda^{out}_{lumo} > \kappa$ and $a
    = 0$ otherwise, and $b = \lambda^{out}_{homo}$ if $\lambda^{out}_{homo} < 1 - \kappa$ and $b = 1$ otherwise; the
    family is built for $(a, b)$; the member applied at iteration $i$ maximises $p(\lambda^{in}_{homo}) -
    p(\lambda^{in}_{lumo})$ over $L = 0, \dots, 7$ unless both inner bounds lie within $\kappa$ of their limits
    ($\lambda^{in}_{lumo} < \kappa$ and $1 - \kappa < \lambda^{in}_{homo}$), in which case $L = 3$ at odd and $L = 4$
    at even iterations; all four bounds are then mapped through the applied polynomial. The sign test $\mathrm{Tr}[X_i
    - X_i^2] \leq 0$ ends the expansion after any iteration; the parameter-free bound test is applied only once both
    acceleration bounds sit at their limits. The synthetic configurations are $16 \times 16$: from the source's Table
    1 energies (in the order $\lambda_{min}$, $\lambda^{out}_{homo}$, $\lambda^{in}_{homo}$, $\lambda^{in}_{lumo}$,
    $\lambda^{out}_{lumo}$, $\lambda_{max}$) the source's normalisation maps an energy $e$ to $(\lambda_{max} -
    e)/(\lambda_{max} - \lambda_{min})$; the spectrum places $\lambda_{lumo}$ at the midpoint of
    $[\lambda^{out}_{lumo}, \lambda^{in}_{lumo}]$ and $\lambda_{homo}$ at the midpoint of $[\lambda^{in}_{homo},
    \lambda^{out}_{homo}]$, with $n_{occ} = \mathrm{round}(16(1 - \mu))$, $\mu = (\lambda^{in}_{lumo} +
    \lambda^{in}_{homo})/2$, equidistant eigenvalues on $[\lambda_{homo}, 1]$ ($n_{occ}$ of them, endpoints included)
    and $16 - n_{occ}$ on $[0, \lambda_{lumo}]$, assembled on the orthonormal frame $Q$ of the QR factorisation of the
    integer matrix $M_{ij} = (((i+1)(j+2)(i+j+3) + a_{seed}(i+4)(j+5)) \bmod 47) - 23$, $i, j = 0..15$, each column of
    $Q$ sign-fixed so that its first entry of modulus above $10^{-12}$ is positive, and $X_0 =
    Q\,\mathrm{diag}(\lambda)\,Q^T$ symmetrised.

    This step performs the polynomial choice of one iteration of the source's scheme: it forms the acceleration bounds
    from the outer bounds and the threshold, builds the family for them, and returns the index the scheme applies (the
    alternation $L = 3$ at odd and $L = 4$ at even iterations when both inner bounds lie within the threshold of their
    limits, otherwise the index maximising the mapped inner gap, the smallest index on a tie) together with the two
    acceleration bounds.

    Args:
        bounds: array of shape (4,), the current bounds $\lambda^{in}_{lumo}, \lambda^{in}_{homo},
        \lambda^{out}_{lumo}, \lambda^{out}_{homo}$ in this order.
        i: integer of at least 1, the 1-based iteration index.
        kappa: float strictly between 0 and 1/2, the deactivation threshold.

    Returns:
        A numpy float64 array of shape $(3,)$: the selected index $L$, the acceleration bound $a$ and the acceleration
        bound $b$.

    Raises:
        ValueError: if i is not an integer of at least 1, if kappa is outside (0, 1/2), or if the bounds are not
        finite and ordered $0 \leq \lambda^{out}_{lumo} \leq \lambda^{in}_{lumo} < \lambda^{in}_{homo} \leq
        \lambda^{out}_{homo} \leq 1$.
    """
    return None
```

### Step 9

propagate_bounds

Goal
----
Maps the four eigenvalue bounds through the applied polynomial by scalar Horner evaluation.

```python
def propagate_bounds(bounds: "np.ndarray", b: "np.ndarray") -> "np.ndarray":
    r"""Maps the four eigenvalue bounds through the applied polynomial by scalar Horner evaluation.

    The source computes the matrix step function of a symmetric matrix $X_0$ with spectrum in $[0, \lambda_{lumo}]
    \cup [\lambda_{homo}, 1]$ by the recursive expansion $X_i = p_i(X_{i-1})$ with degree-eight component polynomials.
    Each component polynomial is a member $p_{(L, R; a, b)}$ of the SP8 family, $L + R = 7$, written in the source's
    parametrisation $p(x) = s_8 \sum_{k=0}^{7} (-1)^k e_k(s)\, x^{8-k}/(8-k) + s_9$ with $e_k$ the elementary
    symmetric polynomials of the stationary points $s_1, \dots, s_7$, so that $p'(x) = s_8 \prod_k (x - s_k)$. For
    construction bounds $0 \leq a < b \leq 1$ the member with $L$ stationary points in $[0, a]$ and $R$ in $[b, 1]$ is
    the unique solution of the equioscillatory conditions $p(s_k) = 0$ for odd $k \leq L$ and $p(s_k) = p(a)$ for even
    $k \leq L$; $p(s_{L+k}) = 1$ for odd $k \leq R$ and $p(s_{L+k}) = p(b)$ for even $k \leq R$; $p(0) = 0$ if $L$ is
    even and $p(0) = p(a)$ if $L$ is odd; $p(1) = 1$ if $R$ is even and $p(1) = p(b)$ if $R$ is odd; with the ordering
    $0 \leq s_L \leq \dots \leq s_1 \leq a < b \leq s_{L+1} \leq \dots \leq s_7 \leq 1$. At a bound sitting at its
    limit the conditions of that side are replaced by the modified ones: for $a = 0$, $s_k = 0$ for $k \leq L$ and
    $p(0) = 0$; for $b = 1$, $s_{L+k} = 1$ for $k \leq R$ and $p(1) = 1$; with both limits the member is the
    closed-form polynomial $p_{(L,R;0,1)}$ with $p' \propto x^L (x - 1)^R$. The expansion (the source's Algorithm 5.1)
    carries inner bounds $\lambda^{in}_{lumo} \leq \lambda^{in}_{homo}$ and outer bounds $\lambda^{out}_{lumo} \leq
    \lambda^{in}_{lumo}$, $\lambda^{in}_{homo} \leq \lambda^{out}_{homo}$ on the two eigenvalues, with the threshold
    $\kappa = 0.01$: the acceleration bounds are $a = \lambda^{out}_{lumo}$ if $\lambda^{out}_{lumo} > \kappa$ and $a
    = 0$ otherwise, and $b = \lambda^{out}_{homo}$ if $\lambda^{out}_{homo} < 1 - \kappa$ and $b = 1$ otherwise; the
    family is built for $(a, b)$; the member applied at iteration $i$ maximises $p(\lambda^{in}_{homo}) -
    p(\lambda^{in}_{lumo})$ over $L = 0, \dots, 7$ unless both inner bounds lie within $\kappa$ of their limits
    ($\lambda^{in}_{lumo} < \kappa$ and $1 - \kappa < \lambda^{in}_{homo}$), in which case $L = 3$ at odd and $L = 4$
    at even iterations; all four bounds are then mapped through the applied polynomial. The sign test $\mathrm{Tr}[X_i
    - X_i^2] \leq 0$ ends the expansion after any iteration; the parameter-free bound test is applied only once both
    acceleration bounds sit at their limits. The synthetic configurations are $16 \times 16$: from the source's Table
    1 energies (in the order $\lambda_{min}$, $\lambda^{out}_{homo}$, $\lambda^{in}_{homo}$, $\lambda^{in}_{lumo}$,
    $\lambda^{out}_{lumo}$, $\lambda_{max}$) the source's normalisation maps an energy $e$ to $(\lambda_{max} -
    e)/(\lambda_{max} - \lambda_{min})$; the spectrum places $\lambda_{lumo}$ at the midpoint of
    $[\lambda^{out}_{lumo}, \lambda^{in}_{lumo}]$ and $\lambda_{homo}$ at the midpoint of $[\lambda^{in}_{homo},
    \lambda^{out}_{homo}]$, with $n_{occ} = \mathrm{round}(16(1 - \mu))$, $\mu = (\lambda^{in}_{lumo} +
    \lambda^{in}_{homo})/2$, equidistant eigenvalues on $[\lambda_{homo}, 1]$ ($n_{occ}$ of them, endpoints included)
    and $16 - n_{occ}$ on $[0, \lambda_{lumo}]$, assembled on the orthonormal frame $Q$ of the QR factorisation of the
    integer matrix $M_{ij} = (((i+1)(j+2)(i+j+3) + a_{seed}(i+4)(j+5)) \bmod 47) - 23$, $i, j = 0..15$, each column of
    $Q$ sign-fixed so that its first entry of modulus above $10^{-12}$ is positive, and $X_0 =
    Q\,\mathrm{diag}(\lambda)\,Q^T$ symmetrised.

    This step maps the four bounds through the applied polynomial by scalar Horner evaluation, in the same order.

    Args:
        bounds: array of shape (4,), the bounds $\lambda^{in}_{lumo}, \lambda^{in}_{homo}, \lambda^{out}_{lumo},
        \lambda^{out}_{homo}$.
        b: array of shape (9,), the monomial coefficients of the applied polynomial.

    Returns:
        A numpy float64 array of shape $(4,)$: the mapped bounds.

    Raises:
        ValueError: if bounds does not hold four finite values or b does not have shape (9,).
    """
    return None
```

### Step 10

stop_check

Goal
----
Evaluates the source's sign test and parameter-free bound test for terminating the expansion after an iteration.

```python
def stop_check(tr_prev: float, tr_curr: float, l: int) -> float:
    r"""Evaluates the source's sign test and parameter-free bound test for terminating the expansion after an iteration.

    The source computes the matrix step function of a symmetric matrix $X_0$ with spectrum in $[0, \lambda_{lumo}]
    \cup [\lambda_{homo}, 1]$ by the recursive expansion $X_i = p_i(X_{i-1})$ with degree-eight component polynomials.
    Each component polynomial is a member $p_{(L, R; a, b)}$ of the SP8 family, $L + R = 7$, written in the source's
    parametrisation $p(x) = s_8 \sum_{k=0}^{7} (-1)^k e_k(s)\, x^{8-k}/(8-k) + s_9$ with $e_k$ the elementary
    symmetric polynomials of the stationary points $s_1, \dots, s_7$, so that $p'(x) = s_8 \prod_k (x - s_k)$. For
    construction bounds $0 \leq a < b \leq 1$ the member with $L$ stationary points in $[0, a]$ and $R$ in $[b, 1]$ is
    the unique solution of the equioscillatory conditions $p(s_k) = 0$ for odd $k \leq L$ and $p(s_k) = p(a)$ for even
    $k \leq L$; $p(s_{L+k}) = 1$ for odd $k \leq R$ and $p(s_{L+k}) = p(b)$ for even $k \leq R$; $p(0) = 0$ if $L$ is
    even and $p(0) = p(a)$ if $L$ is odd; $p(1) = 1$ if $R$ is even and $p(1) = p(b)$ if $R$ is odd; with the ordering
    $0 \leq s_L \leq \dots \leq s_1 \leq a < b \leq s_{L+1} \leq \dots \leq s_7 \leq 1$. At a bound sitting at its
    limit the conditions of that side are replaced by the modified ones: for $a = 0$, $s_k = 0$ for $k \leq L$ and
    $p(0) = 0$; for $b = 1$, $s_{L+k} = 1$ for $k \leq R$ and $p(1) = 1$; with both limits the member is the
    closed-form polynomial $p_{(L,R;0,1)}$ with $p' \propto x^L (x - 1)^R$. The expansion (the source's Algorithm 5.1)
    carries inner bounds $\lambda^{in}_{lumo} \leq \lambda^{in}_{homo}$ and outer bounds $\lambda^{out}_{lumo} \leq
    \lambda^{in}_{lumo}$, $\lambda^{in}_{homo} \leq \lambda^{out}_{homo}$ on the two eigenvalues, with the threshold
    $\kappa = 0.01$: the acceleration bounds are $a = \lambda^{out}_{lumo}$ if $\lambda^{out}_{lumo} > \kappa$ and $a
    = 0$ otherwise, and $b = \lambda^{out}_{homo}$ if $\lambda^{out}_{homo} < 1 - \kappa$ and $b = 1$ otherwise; the
    family is built for $(a, b)$; the member applied at iteration $i$ maximises $p(\lambda^{in}_{homo}) -
    p(\lambda^{in}_{lumo})$ over $L = 0, \dots, 7$ unless both inner bounds lie within $\kappa$ of their limits
    ($\lambda^{in}_{lumo} < \kappa$ and $1 - \kappa < \lambda^{in}_{homo}$), in which case $L = 3$ at odd and $L = 4$
    at even iterations; all four bounds are then mapped through the applied polynomial. The sign test $\mathrm{Tr}[X_i
    - X_i^2] \leq 0$ ends the expansion after any iteration; the parameter-free bound test is applied only once both
    acceleration bounds sit at their limits. The synthetic configurations are $16 \times 16$: from the source's Table
    1 energies (in the order $\lambda_{min}$, $\lambda^{out}_{homo}$, $\lambda^{in}_{homo}$, $\lambda^{in}_{lumo}$,
    $\lambda^{out}_{lumo}$, $\lambda_{max}$) the source's normalisation maps an energy $e$ to $(\lambda_{max} -
    e)/(\lambda_{max} - \lambda_{min})$; the spectrum places $\lambda_{lumo}$ at the midpoint of
    $[\lambda^{out}_{lumo}, \lambda^{in}_{lumo}]$ and $\lambda_{homo}$ at the midpoint of $[\lambda^{in}_{homo},
    \lambda^{out}_{homo}]$, with $n_{occ} = \mathrm{round}(16(1 - \mu))$, $\mu = (\lambda^{in}_{lumo} +
    \lambda^{in}_{homo})/2$, equidistant eigenvalues on $[\lambda_{homo}, 1]$ ($n_{occ}$ of them, endpoints included)
    and $16 - n_{occ}$ on $[0, \lambda_{lumo}]$, assembled on the orthonormal frame $Q$ of the QR factorisation of the
    integer matrix $M_{ij} = (((i+1)(j+2)(i+j+3) + a_{seed}(i+4)(j+5)) \bmod 47) - 23$, $i, j = 0..15$, each column of
    $Q$ sign-fixed so that its first entry of modulus above $10^{-12}$ is positive, and $X_0 =
    Q\,\mathrm{diag}(\lambda)\,Q^T$ symmetrised.

    This step evaluates the source's two termination tests after an iteration with member $l$: the sign test fires
    when $\mathrm{Tr}[X_i - X_i^2] \leq 0$, and otherwise the parameter-free bound test fires when $\mathrm{Tr}[X_i -
    X_i^2] > C_l\,(\mathrm{Tr}[X_{i-1} - X_{i-1}^2])^{q_l}$ with orders $q = (1, 2, 3, 4, 4, 3, 2, 1)$ and constants
    $C = (\infty, 28, 56, 82, 82, 56, 28, \infty)$ for $l = 0, \dots, 7$; the sign test takes precedence, and the
    members $l = 0$ and $l = 7$ never fire the bound test.

    Args:
        tr_prev: float, the idempotency trace $\mathrm{Tr}[X - X^2]$ before the iteration.
        tr_curr: float, the same trace after the iteration.
        l: integer in 0..7, the family member applied in the iteration.

    Returns:
        A float: 2.0 if the sign test fires, 1.0 if the bound test fires, 0.0 otherwise.

    Raises:
        ValueError: if l is not an integer in 0..7.
    """
    return None
```

### Step 11

sp8_accelerated_audit

Goal
----
Runs the source's accelerated SP8 expansion on the synthetic matrix consistent with given eigenvalue bounds and records the applied members, the mapped bounds and the idempotency traces.

```python
def sp8_accelerated_audit(energies: "np.ndarray", seed: int, niter: int) -> "np.ndarray":
    r"""Runs the source's accelerated SP8 expansion on the synthetic matrix consistent with given eigenvalue bounds and records the applied members, the mapped bounds and the idempotency traces.

    The source computes the matrix step function of a symmetric matrix $X_0$ with spectrum in $[0, \lambda_{lumo}]
    \cup [\lambda_{homo}, 1]$ by the recursive expansion $X_i = p_i(X_{i-1})$ with degree-eight component polynomials.
    Each component polynomial is a member $p_{(L, R; a, b)}$ of the SP8 family, $L + R = 7$, written in the source's
    parametrisation $p(x) = s_8 \sum_{k=0}^{7} (-1)^k e_k(s)\, x^{8-k}/(8-k) + s_9$ with $e_k$ the elementary
    symmetric polynomials of the stationary points $s_1, \dots, s_7$, so that $p'(x) = s_8 \prod_k (x - s_k)$. For
    construction bounds $0 \leq a < b \leq 1$ the member with $L$ stationary points in $[0, a]$ and $R$ in $[b, 1]$ is
    the unique solution of the equioscillatory conditions $p(s_k) = 0$ for odd $k \leq L$ and $p(s_k) = p(a)$ for even
    $k \leq L$; $p(s_{L+k}) = 1$ for odd $k \leq R$ and $p(s_{L+k}) = p(b)$ for even $k \leq R$; $p(0) = 0$ if $L$ is
    even and $p(0) = p(a)$ if $L$ is odd; $p(1) = 1$ if $R$ is even and $p(1) = p(b)$ if $R$ is odd; with the ordering
    $0 \leq s_L \leq \dots \leq s_1 \leq a < b \leq s_{L+1} \leq \dots \leq s_7 \leq 1$. At a bound sitting at its
    limit the conditions of that side are replaced by the modified ones: for $a = 0$, $s_k = 0$ for $k \leq L$ and
    $p(0) = 0$; for $b = 1$, $s_{L+k} = 1$ for $k \leq R$ and $p(1) = 1$; with both limits the member is the
    closed-form polynomial $p_{(L,R;0,1)}$ with $p' \propto x^L (x - 1)^R$. The expansion (the source's Algorithm 5.1)
    carries inner bounds $\lambda^{in}_{lumo} \leq \lambda^{in}_{homo}$ and outer bounds $\lambda^{out}_{lumo} \leq
    \lambda^{in}_{lumo}$, $\lambda^{in}_{homo} \leq \lambda^{out}_{homo}$ on the two eigenvalues, with the threshold
    $\kappa = 0.01$: the acceleration bounds are $a = \lambda^{out}_{lumo}$ if $\lambda^{out}_{lumo} > \kappa$ and $a
    = 0$ otherwise, and $b = \lambda^{out}_{homo}$ if $\lambda^{out}_{homo} < 1 - \kappa$ and $b = 1$ otherwise; the
    family is built for $(a, b)$; the member applied at iteration $i$ maximises $p(\lambda^{in}_{homo}) -
    p(\lambda^{in}_{lumo})$ over $L = 0, \dots, 7$ unless both inner bounds lie within $\kappa$ of their limits
    ($\lambda^{in}_{lumo} < \kappa$ and $1 - \kappa < \lambda^{in}_{homo}$), in which case $L = 3$ at odd and $L = 4$
    at even iterations; all four bounds are then mapped through the applied polynomial. The sign test $\mathrm{Tr}[X_i
    - X_i^2] \leq 0$ ends the expansion after any iteration; the parameter-free bound test is applied only once both
    acceleration bounds sit at their limits. The synthetic configurations are $16 \times 16$: from the source's Table
    1 energies (in the order $\lambda_{min}$, $\lambda^{out}_{homo}$, $\lambda^{in}_{homo}$, $\lambda^{in}_{lumo}$,
    $\lambda^{out}_{lumo}$, $\lambda_{max}$) the source's normalisation maps an energy $e$ to $(\lambda_{max} -
    e)/(\lambda_{max} - \lambda_{min})$; the spectrum places $\lambda_{lumo}$ at the midpoint of
    $[\lambda^{out}_{lumo}, \lambda^{in}_{lumo}]$ and $\lambda_{homo}$ at the midpoint of $[\lambda^{in}_{homo},
    \lambda^{out}_{homo}]$, with $n_{occ} = \mathrm{round}(16(1 - \mu))$, $\mu = (\lambda^{in}_{lumo} +
    \lambda^{in}_{homo})/2$, equidistant eigenvalues on $[\lambda_{homo}, 1]$ ($n_{occ}$ of them, endpoints included)
    and $16 - n_{occ}$ on $[0, \lambda_{lumo}]$, assembled on the orthonormal frame $Q$ of the QR factorisation of the
    integer matrix $M_{ij} = (((i+1)(j+2)(i+j+3) + a_{seed}(i+4)(j+5)) \bmod 47) - 23$, $i, j = 0..15$, each column of
    $Q$ sign-fixed so that its first entry of modulus above $10^{-12}$ is positive, and $X_0 =
    Q\,\mathrm{diag}(\lambda)\,Q^T$ symmetrised.

    The orchestrator. It must call the earlier functions rather than reimplementing them: it maps the energies to the
    four bounds on $[0, 1]$, builds the synthetic matrix consistent with them, and runs exactly niter iterations (the
    applied member taken from the closed forms when both accelerations are deactivated and from the equioscillatory
    solve otherwise, each solved member checked against its defining conditions) of the source's accelerated scheme
    with $\kappa = 0.01$, recording after every iteration the applied index, the four mapped bounds (clipped to the
    unit interval, the outer bounds kept enclosing the inner ones, against rounding) and the idempotency trace of the
    new matrix. The first two iterations are the conditioning phase of the audit: if a termination test that applies
    at that iteration (the sign test always, the bound test only once both accelerations are deactivated) fires after
    one of them, the configuration is rejected.

    Args:
        energies: array of shape (6,), the six energies in the order $\lambda_{min}$, $\lambda^{out}_{homo}$,
        $\lambda^{in}_{homo}$, $\lambda^{in}_{lumo}$, $\lambda^{out}_{lumo}$, $\lambda_{max}$.
        seed: non-negative integer, the frame seed $a_{seed}$ of the synthetic matrix.
        niter: integer of at least 2, the number of iterations to perform.

    Returns:
        A numpy float64 array of shape $(\text{niter}, 6)$: per row the applied index $L$, $\lambda^{in}_{lumo}$,
        $\lambda^{in}_{homo}$, $\lambda^{out}_{lumo}$, $\lambda^{out}_{homo}$ after the iteration, and
        $\mathrm{Tr}[X_i - X_i^2]$.

    Raises:
        ValueError: if the energies are not six finite values ordered $\lambda_{min} < \lambda^{out}_{homo} \leq
        \lambda^{in}_{homo} < \lambda^{in}_{lumo} \leq \lambda^{out}_{lumo} < \lambda_{max}$, if seed is not a
        non-negative integer, if niter is not an integer of at least 2, if a solved member fails its defining
        conditions, or if a termination test fires within the first two iterations.
    """
    return None
```
