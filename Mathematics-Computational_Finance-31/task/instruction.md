# Mathematics-Computational_Finance-31

## Background

The source article combines a nonlinear Black--Scholes model with compact three-point RBF-FD spatial differentiation and classical explicit RK4 time integration. Transaction costs make the effective volatility depend implicitly on the discrete Gamma, so an RK4 stage is not merely a linear combination of fixed rates: the central option value, Gamma, liquidity argument, inverse branch, and nonlinear rate must be recomputed at every stage.

The requested quantity is a fourth-order interaction sensitivity of the one-step nonlinear pricing operator, not a standard quoted market Greek. It probes how two coupled perturbation bundles interact twice each through stencil geometry, option values, market parameters, the implicit liquidity map, and RK4 stage feedback. A normalized $3\times3$ bivariate Taylor jet captures the required coefficient directly while avoiding the cancellation and step-size choices inherent in finite differencing.

The two neighboring values are deliberately frozen during the local stage evolution so that the diagnostic is a well-defined single semidiscrete row. This convention does not claim to reproduce the article's full coupled grid evolution.

## Problem

Using the supplied article, compute a fourth-order two-factor interaction sensitivity of one local nonlinear RBF-FD/RK4 update. The equations themselves are intentionally not reproduced here: recover the three-point stencil and its first- and second-derivative weights from Equations (20), (22)--(24), and (27)--(29); the applicable branch of the auxiliary liquidity relation from Equation (10); the transformed nonlinear pricing equation from Equation (11); and the classical four-stage RK construction and tableau in the discussion surrounding Equation (33).

Also recover two numerical conventions from the article's Computational Aspects and Example 1 material:

1. the rule relating the RBF shape parameter $c$ to the maximum neighboring-node spacing $h_{\max}$, and
2. the time step $\psi$ stated in the captions of the two Example 1 solution figures.

Apply the first convention at $h_{\max}=10$, calling the result $c_\star$, and use the figure-caption value as $\psi_\star$.

Let the twelve independent coordinates be ordered as

$$
p=(W_-,W_0,W_+,h,o,c,r,q,\tau,\alpha,\sigma_0,X),
$$

with baseline

$$
p_0=(8.4032,14.6457,22.2960,10,1,c_\star,0.1,0,0.75,0.01,0.2,100).
$$

The three option values are prescribed as an authored local state at $\tau_0=0.75$ for this sensitivity diagnostic; they are not asserted to be the article's numerical solution at that time.

Use the two directions

$$
u=(0.25,-0.5,0.75,0.1,-0.02,0.2,0.01,-0.015,0.03,0.0001,0.005,0.4)
$$

and

$$
v=(-0.4,0.2,0.1,-0.15,0.03,-0.25,-0.02,0.01,-0.04,-0.0002,0.008,-0.3),
$$

and define $p(\varepsilon,\eta)=p_0+\varepsilon u+\eta v$. All twelve coordinates follow these affine paths independently. In particular, the source-derived relation between $c$ and $h_{\max}$ is used only to construct $p_0$; do not re-impose it after perturbation.

Let $\Phi_{\psi_\star}(\varepsilon,\eta)$ denote one classical RK4 step of the central semidiscrete row obtained from the recovered equations. This is a frozen-neighbor local diagnostic, not a full-grid evolution. At each RK4 stage:

- use the correct RK4 stage time and the stage-updated central value $W_0$;
- retain $W_-$ and $W_+$ at their respective values in $p(\varepsilon,\eta)$, without stage-updating them;
- recompute all six RBF-FD weights, $W_X$, $W_{XX}$, the liquidity argument, the Equation (10) branch dictated by its sign, and the complete Equation (11) right-hand side.

Thus stages two and three each start from the original central value plus its own half-step increment, rather than being accumulated sequentially. Define the annualized one-step operator

$$
R(\varepsilon,\eta)
=\frac{\Phi_{\psi_\star}(\varepsilon,\eta)-W_0(\varepsilon,\eta)}{\psi_\star}.
$$

Carry a normalized bivariate Taylor jet

$$
J_{ij}(f)=\frac{1}{i!\,j!}
\left.\frac{\partial^{i+j}f}
{\partial\varepsilon^i\partial\eta^j}\right|_{(0,0)},
\qquad 0\le i,j\le2,
$$

with degrees truncated separately at two in each variable. Recompute the implicit liquidity branch coefficient-consistently at every stage. Use double precision without rounding intermediate values and make every scalar baseline branch solve satisfy an absolute implicit-equation residual below $10^{-12}$. Finite differences, complex-step estimates, or nine separate scalar RK4 evaluations are not accepted as the primary method.

Compute

$$
\left.
\frac{\partial^4 R}{\partial\varepsilon^2\partial\eta^2}
\right|_{(0,0)}.
$$

Remember that this derivative equals $2!2!\,J_{22}(R)=4J_{22}(R)$. In the reasoning, state both recovered source conventions and give enough stage or jet checkpoints to make the branch and RK4 construction auditable. Report one numeric value rounded to six decimal places.

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

rbf_fd_weights

Goal
----
Return the specified three-point RBF-FD weight triples.



The nodes are $X_-=X_0-h$, $X_0$, and $X_+=X_0+oh$.

The first-derivative weights are

$$a_-=-\frac{o(2c^2+h^2o)}{2c^2h(o+1)},\qquad

a_0=\frac{h(o-1)}{2c^2}+\frac{o-1}{ho},\qquad

a_+=\frac{h^2/c^2+2/o}{2h(o+1)}.$$

The second-derivative weights are

$$b_-=\frac{2/h^2-(o-3)o/c^2}{o+1},\qquad

b_0=\frac{(o^2-4o+1)/c^2-2/h^2}{o},\qquad

b_+=\frac{2c^2+h^2(3o-1)}{c^2h^2o(o+1)}.$$

Treat all geometry inputs independently. Do not impose a shape rule or

specialize to a uniform stencil. Use binary64 without intermediate rounding.

```python
def rbf_fd_weights(
    h: float, o: float, c: float
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    r"""Return the specified three-point RBF-FD weight triples.

    The nodes are $X_-=X_0-h$, $X_0$, and $X_+=X_0+oh$.
    The first-derivative weights are
    $$a_-=-\frac{o(2c^2+h^2o)}{2c^2h(o+1)},\qquad
    a_0=\frac{h(o-1)}{2c^2}+\frac{o-1}{ho},\qquad
    a_+=\frac{h^2/c^2+2/o}{2h(o+1)}.$$
    The second-derivative weights are
    $$b_-=\frac{2/h^2-(o-3)o/c^2}{o+1},\qquad
    b_0=\frac{(o^2-4o+1)/c^2-2/h^2}{o},\qquad
    b_+=\frac{2c^2+h^2(3o-1)}{c^2h^2o(o+1)}.$$
    Treat all geometry inputs independently. Do not impose a shape rule or
    specialize to a uniform stencil. Use binary64 without intermediate rounding.

    Parameters
    ----------
    h : float
        Finite left-node spacing $h>0$.
    o : float
        Finite right-to-left spacing ratio $o=(X_+-X_0)/(X_0-X_-)>0$.
    c : float
        Finite independent RBF shape parameter $c>0$.
        Inputs must keep the displayed arithmetic finite and representable.

    Returns
    -------
    tuple[tuple[float, float, float], tuple[float, float, float]]
        Exactly $((a_-,a_0,a_+),(b_-,b_0,b_+))$: first-derivative weights,
        then second-derivative weights, both in node order $(-,0,+)$.

    Raises
    ------
    No exception is required for inputs in the stated valid domain.
    Outside that domain, validation behavior is unspecified and ordinary
    Python arithmetic or type exceptions may propagate.
    """
    first_weights = (0.0, 0.0, 0.0)
    second_weights = (0.0, 0.0, 0.0)
    return (first_weights, second_weights)
```

### Step 2

spatial_derivatives

Goal
----
Apply two supplied derivative-weight triples to three nodal values.



In the common node order $(-,0,+)$, compute

$$w_x=a_-W_-+a_0W_0+a_+W_+,\qquad

w_{xx}=b_-W_-+b_0W_0+b_+W_+.$$

The weights already include all mesh-spacing factors. Apply them exactly

as supplied, without intermediate rounding or additional division.

```python
def spatial_derivatives(
    values: tuple[float, float, float],
    first_weights: tuple[float, float, float],
    second_weights: tuple[float, float, float],
) -> tuple[float, float]:
    r"""Apply two supplied derivative-weight triples to three nodal values.

    In the common node order $(-,0,+)$, compute
    $$w_x=a_-W_-+a_0W_0+a_+W_+,\qquad
    w_{xx}=b_-W_-+b_0W_0+b_+W_+.$$
    The weights already include all mesh-spacing factors. Apply them exactly
    as supplied, without intermediate rounding or additional division.

    Parameters
    ----------
    values : tuple[float, float, float]
        Exactly three finite nodal values $(W_-,W_0,W_+)$.
    first_weights : tuple[float, float, float]
        Exactly three finite first-derivative weights $(a_-,a_0,a_+)$.
    second_weights : tuple[float, float, float]
        Exactly three finite second-derivative weights $(b_-,b_0,b_+)$.
        Signed inputs and zero outputs are permitted. Inputs must keep
        products and sums finite in binary64 arithmetic.

    Returns
    -------
    tuple[float, float]
        The pair $(w_x,w_{xx})$: local Delta followed by local Gamma.

    Raises
    ------
    No exception is required for inputs in the stated valid domain.
    Outside that domain, validation behavior is unspecified and ordinary
    Python arithmetic or type exceptions may propagate.
    """
    wx = 0.0
    wxx = 0.0
    return (wx, wxx)
```

### Step 3

positive_z_from_x

Goal
----
Return the positive liquidity inverse and normalized sensitivities.



The unique positive inverse $Z=Z(x)$ is defined by

$$x=\left(\sqrt{Z}-\frac{\operatorname{arsinh}(\sqrt{Z})}

{\sqrt{1+Z}}\right)^2,\qquad x>0,\quad Z>0.$$

Also compute the analytic inverse responses

$$E=\frac{xZ'(x)}{Z(x)},\qquad C=\frac{x^2Z''(x)}{Z(x)}.$$

Derivatives refer to this mathematical inverse. Finite differences and

differentiation of numerical solver iterations are not accepted.

Cover the complete positive finite binary64 range, including subnormals,

using ordinary Python float arithmetic and stable evaluation where direct

subtraction cancels. No particular initial guess or solver is required.



With $y=\sqrt{Z}$ and

$g(y)=y-\operatorname{arsinh}(y)/\sqrt{1+y^2}$, acceptance requires

$$|g(y)-\sqrt{x}|\leq\mathrm{residual\_tol}\sqrt{x}.$$

A candidate must pass the residual check within the specified iteration

budget, including the candidate checked at the start of each iteration.

Do not silently relax the tolerance, exceed the budget, or mistake

cancellation or underflow for convergence. Audit the root with a stable

evaluation of the defining relation. All returned quantities must be finite.

```python
def positive_z_from_x(
    x: float,
    residual_tol: float = 1e-13,
    max_iter: int = 24,
) -> tuple[float, float, float]:
    r"""Return the positive liquidity inverse and normalized sensitivities.

    The unique positive inverse $Z=Z(x)$ is defined by
    $$x=\left(\sqrt{Z}-\frac{\operatorname{arsinh}(\sqrt{Z})}
    {\sqrt{1+Z}}\right)^2,\qquad x>0,\quad Z>0.$$
    Also compute the analytic inverse responses
    $$E=\frac{xZ'(x)}{Z(x)},\qquad C=\frac{x^2Z''(x)}{Z(x)}.$$
    Derivatives refer to this mathematical inverse. Finite differences and
    differentiation of numerical solver iterations are not accepted.
    Cover the complete positive finite binary64 range, including subnormals,
    using ordinary Python float arithmetic and stable evaluation where direct
    subtraction cancels. No particular initial guess or solver is required.

    With $y=\sqrt{Z}$ and
    $g(y)=y-\operatorname{arsinh}(y)/\sqrt{1+y^2}$, acceptance requires
    $$|g(y)-\sqrt{x}|\leq\mathrm{residual\_tol}\sqrt{x}.$$
    A candidate must pass the residual check within the specified iteration
    budget, including the candidate checked at the start of each iteration.
    Do not silently relax the tolerance, exceed the budget, or mistake
    cancellation or underflow for convergence. Audit the root with a stable
    evaluation of the defining relation. All returned quantities must be finite.

    Parameters
    ----------
    x : float
        Finite positive binary64 liquidity argument $x$, including subnormals.
    residual_tol : float, default 1e-13
        Finite relative tolerance $0<\mathrm{residual\_tol}<1$ for the
        unsquared transformed relation. It is not an absolute tolerance on $x$.
    max_iter : int, default 24
        Positive integer budget $\mathrm{max\_iter}\geq1$ for nonlinear
        iterations with residual checks. Unattainable tolerance may fail.

    Returns
    -------
    tuple[float, float, float]
        Exactly $(Z,E,C)$, where $Z>0$ and all entries are finite.
        The last two outputs are normalized responses, not $Z'$ and $Z''$.

    Raises
    ------
    ValueError
        If $x$ is nonfinite or nonpositive, the tolerance is nonfinite or
        outside the open interval $(0,1)$, or the budget is not a positive integer.
    ArithmeticError
        If no acceptable root is found within the budget or finite root and
        response values cannot be produced. Do not return an unconverged result.
    """
    z = 0.0
    elasticity = 0.0
    curvature = 0.0
    return z, elasticity, curvature
```

### Step 4

nonlinear_bs_rhs

Goal
----
Evaluate the specified nonlinear Black-Scholes local time derivative.



Use the transformed time-to-maturity convention

$$F=\frac12\sigma_0^2(1+z)X^2w_{xx}+(r-q)Xw_x-rw_0.$$

The liquidity correction and spatial derivatives are already supplied.

Evaluate this expression without intermediate rounding.

```python
def nonlinear_bs_rhs(
    wx: float,
    wxx: float,
    w0: float,
    X: float,
    r: float,
    q: float,
    sigma0: float,
    z: float,
) -> float:
    r"""Evaluate the specified nonlinear Black-Scholes local time derivative.

    Use the transformed time-to-maturity convention
    $$F=\frac12\sigma_0^2(1+z)X^2w_{xx}+(r-q)Xw_x-rw_0.$$
    The liquidity correction and spatial derivatives are already supplied.
    Evaluate this expression without intermediate rounding.

    Parameters
    ----------
    wx : float
        Finite local first derivative $w_x$; either sign is allowed.
    wxx : float
        Finite local second derivative $w_{xx}$; either sign is allowed.
    w0 : float
        Finite central option value $w_0$.
    X : float
        Finite underlying coordinate $X$, normally positive in the model.
    r : float
        Finite interest rate $r$.
    q : float
        Finite continuous dividend yield $q$.
    sigma0 : float
        Finite baseline volatility $\sigma_0$, squared in the expression.
    z : float
        Finite precomputed liquidity correction $z$; the financial positive
        branch has $z\geq0$, including the zero-liquidity limit.
        This algebraic evaluator does not enforce financial consistency.
        Inputs must keep all products and the sum finite in binary64 arithmetic.

    Returns
    -------
    float
        The unrounded local rate $F$, not an updated option value or a
        sensitivity coefficient. Its discount contribution is $-rw_0$.

    Raises
    ------
    No exception is required for inputs in the stated valid domain.
    Outside that domain, validation behavior is unspecified and ordinary
    Python arithmetic or type exceptions may propagate.
    """
    rhs = 0.0
    return rhs
```

### Step 5

spatial_taylor_jets

Goal
----
Return normalized Taylor jets for the local Delta and Gamma.



Every jet is a $3\times3$ matrix with normalization

$$A_{ij}=\frac{1}{i!j!}\partial_s^i\partial_t^j A(0,0),

\qquad 0\le i,j\le2.$$

Rows index the $s$ degree; columns index the $t$ degree. Arbitrary

incoming higher-order coefficients are part of the input. Multiplication

means convolution truncated separately in each variable:

$$(AB)_{ij}=\sum_{a=0}^{i}\sum_{b=0}^{j}A_{ab}B_{i-a,j-b}.$$

Division means multiplication by the reciprocal in this same algebra.

Do not interpret entries as point samples or unnormalized derivatives.



The nodes are $X_0-h$, $X_0$, and $X_0+oh$. Evaluate these weight

expressions on the full geometry jets:

$$a_-=-\frac{o(2c^2+h^2o)}{2c^2h(o+1)},\qquad

a_0=\frac{h(o-1)}{2c^2}+\frac{o-1}{ho},\qquad

a_+=\frac{h^2/c^2+2/o}{2h(o+1)},$$

$$b_-=\frac{2/h^2-(o-3)o/c^2}{o+1},\qquad

b_0=\frac{(o^2-4o+1)/c^2-2/h^2}{o},\qquad

b_+=\frac{2c^2+h^2(3o-1)}{c^2h^2o(o+1)}.$$

Then compute

$$\Delta=a_-W_-+a_0W_0+a_+W_+,\qquad

\Gamma=b_-W_-+b_0W_0+b_+W_+.$$

Treat geometry coordinates independently; do not impose $o=1$ or $c=3h$.

Use binary64 series arithmetic without rounding or finite differences.

```python
def spatial_taylor_jets(
    values_coeff: list[list[list[float]]],
    h_coeff: list[list[float]],
    o_coeff: list[list[float]],
    c_coeff: list[list[float]],
) -> tuple[list[list[float]], list[list[float]]]:
    r"""Return normalized Taylor jets for the local Delta and Gamma.

    Every jet is a $3\times3$ matrix with normalization
    $$A_{ij}=\frac{1}{i!j!}\partial_s^i\partial_t^j A(0,0),
    \qquad 0\le i,j\le2.$$
    Rows index the $s$ degree; columns index the $t$ degree. Arbitrary
    incoming higher-order coefficients are part of the input. Multiplication
    means convolution truncated separately in each variable:
    $$(AB)_{ij}=\sum_{a=0}^{i}\sum_{b=0}^{j}A_{ab}B_{i-a,j-b}.$$
    Division means multiplication by the reciprocal in this same algebra.
    Do not interpret entries as point samples or unnormalized derivatives.

    The nodes are $X_0-h$, $X_0$, and $X_0+oh$. Evaluate these weight
    expressions on the full geometry jets:
    $$a_-=-\frac{o(2c^2+h^2o)}{2c^2h(o+1)},\qquad
    a_0=\frac{h(o-1)}{2c^2}+\frac{o-1}{ho},\qquad
    a_+=\frac{h^2/c^2+2/o}{2h(o+1)},$$
    $$b_-=\frac{2/h^2-(o-3)o/c^2}{o+1},\qquad
    b_0=\frac{(o^2-4o+1)/c^2-2/h^2}{o},\qquad
    b_+=\frac{2c^2+h^2(3o-1)}{c^2h^2o(o+1)}.$$
    Then compute
    $$\Delta=a_-W_-+a_0W_0+a_+W_+,\qquad
    \Gamma=b_-W_-+b_0W_0+b_+W_+.$$
    Treat geometry coordinates independently; do not impose $o=1$ or $c=3h$.
    Use binary64 series arithmetic without rounding or finite differences.

    Parameters
    ----------
    values_coeff : list[list[list[float]]]
        Exactly three finite $3\times3$ jets in order $(W_-,W_0,W_+)$.
    h_coeff : list[list[float]]
        Finite $3\times3$ jet of the left spacing, with $h_{00}>0$.
    o_coeff : list[list[float]]
        Finite $3\times3$ jet of the right-to-left gap ratio, with $o_{00}>0$.
    c_coeff : list[list[float]]
        Finite $3\times3$ jet of the RBF shape parameter, with $c_{00}>0$.
        The valid numerical domain requires representable finite binary64
        intermediates and output coefficients.

    Returns
    -------
    tuple[list[list[float]], list[list[float]]]
        The normalized $3\times3$ jet of $\Delta$, followed by that of
        $\Gamma$, both as nested lists in the input row/column convention.

    Raises
    ------
    ValueError
        If there are not three value jets, any matrix is not $3\times3$,
        an input coefficient is nonfinite, or a required baseline is nonpositive.
    ArithmeticError
        Native arithmetic exceptions, including ZeroDivisionError and
        OverflowError, may propagate outside the representable numerical domain.
        Malformed objects outside the typed interface may raise TypeError.
    """
    delta_coeff = [[0.0] * 3 for _ in range(3)]
    gamma_coeff = [[0.0] * 3 for _ in range(3)]
    return delta_coeff, gamma_coeff
```

### Step 6

implicit_z_taylor_jet

Goal
----
Lift the positive liquidity inverse to a normalized bivariate jet.



Both input and output use $3\times3$ coefficient matrices with

$$A_{ij}=\frac{1}{i!j!}\partial_s^i\partial_t^jA(0,0),

\qquad0\le i,j\le2.$$

Rows index the $s$ degree and columns the $t$ degree. All nine incoming

entries are authoritative, including arbitrary higher-order coefficients.



The positive branch is specified by

$$x=\Phi(Z)=\left(\sqrt{Z}-\frac{\operatorname{arsinh}(\sqrt{Z})}

{\sqrt{1+Z}}\right)^2,\qquad Z>0.$$

Return the unique formal Taylor jet satisfying

$\Phi(Z(s,t))=x(s,t)$ through bidegree $(2,2)$ and the positive scalar

baseline $\Phi(Z_{00})=x_{00}$. The retained entry $(2,2)$ has total

degree four, so composition terms through total order four are required.

The scalar baseline may use positive_z_from_x; its first returned

component is $Z$, followed by dimensionless elasticity and curvature.

All remaining coefficients must satisfy the formal implicit relation.

Use a cancellation-safe analytic evaluation at small roots consistently

with the scalar solve. Point sampling, polynomial fitting, finite or

complex differences, and differentiating iteration traces are not accepted.

```python
def implicit_z_taylor_jet(
    x_coeff: list[list[float]],
    residual_tol: float = 1e-13,
    max_iter: int = 24,
) -> list[list[float]]:
    r"""Lift the positive liquidity inverse to a normalized bivariate jet.

    Both input and output use $3\times3$ coefficient matrices with
    $$A_{ij}=\frac{1}{i!j!}\partial_s^i\partial_t^jA(0,0),
    \qquad0\le i,j\le2.$$
    Rows index the $s$ degree and columns the $t$ degree. All nine incoming
    entries are authoritative, including arbitrary higher-order coefficients.

    The positive branch is specified by
    $$x=\Phi(Z)=\left(\sqrt{Z}-\frac{\operatorname{arsinh}(\sqrt{Z})}
    {\sqrt{1+Z}}\right)^2,\qquad Z>0.$$
    Return the unique formal Taylor jet satisfying
    $\Phi(Z(s,t))=x(s,t)$ through bidegree $(2,2)$ and the positive scalar
    baseline $\Phi(Z_{00})=x_{00}$. The retained entry $(2,2)$ has total
    degree four, so composition terms through total order four are required.
    The scalar baseline may use positive_z_from_x; its first returned
    component is $Z$, followed by dimensionless elasticity and curvature.
    All remaining coefficients must satisfy the formal implicit relation.
    Use a cancellation-safe analytic evaluation at small roots consistently
    with the scalar solve. Point sampling, polynomial fitting, finite or
    complex differences, and differentiating iteration traces are not accepted.

    Parameters
    ----------
    x_coeff : list[list[float]]
        Finite normalized $3\times3$ jet with $x_{00}>0$. The valid numerical
        domain requires finite representable fourth-order intermediates and
        coefficients. Full binary64 coverage is required of the scalar inverse,
        not of this higher-order lift.
    residual_tol : float, default 1e-13
        Finite scalar-root tolerance $0<\mathrm{residual\_tol}<1$.
        With $y=\sqrt{Z_{00}}$ and
        $g(y)=y-\operatorname{arsinh}(y)/\sqrt{1+y^2}$, require
        $|g(y)-\sqrt{x_{00}}|\le\mathrm{residual\_tol}\sqrt{x_{00}}$.
    max_iter : int, default 24
        Positive integer scalar-root iteration budget, with the same
        acceptance-within-budget semantics as positive_z_from_x.

    Returns
    -------
    list[list[float]]
        The finite normalized $3\times3$ jet of the positive $Z(s,t)$,
        preserving the input row and column convention.

    Raises
    ------
    ValueError
        If the input shape differs from $3\times3$, an input is nonfinite,
        $x_{00}\le0$, or the scalar tolerance or iteration budget is invalid.
    ArithmeticError
        If the scalar solve fails, its implicit slope is invalid, or a
        nonfinite inverse coefficient is produced. Native ZeroDivisionError
        or OverflowError may propagate when fourth-order arithmetic cannot
        be represented. Malformed objects outside the typed API may raise TypeError.
    """
    z_coeff = [[0.0] * 3 for _ in range(3)]
    return z_coeff
```

### Step 7

rk4_taylor_step

Goal
----
Evaluate a frozen-neighbor local RK4 update in bivariate Taylor algebra.



Each $3\times3$ matrix uses normalized coefficients

$$A_{ij}=\frac{1}{i!j!}\partial_s^i\partial_t^j A(0,0),

\qquad0\le i,j\le2.$$

Rows index the $s$ degree and columns the $t$ degree. Preserve arbitrary

higher-order incoming entries and truncate each degree separately.

Products and analytic compositions use this Taylor algebra, not pointwise

matrix arithmetic, finite differences, or scalar sampling.



For a stage central jet $Y$ and time jet $\theta$, obtain $(\Delta,\Gamma)$

by applying spatial_taylor_jets to $(W_-,Y,W_+)$ and geometry $(h,o,c)$.

This preceding function defines the six nonuniform RBF-FD weight jets.

Require $\Gamma_{00}>0$ at every stage and recompute

$$x=\exp(r\theta)\alpha^2X^2\Gamma,$$

$$x=\left(\sqrt{Z}-\frac{\operatorname{arsinh}(\sqrt{Z})}

{\sqrt{1+Z}}\right)^2,\qquad Z_{00}>0,$$

$$F(Y,\theta)=\frac12\sigma_0^2(1+Z)X^2\Gamma+(r-q)X\Delta-rY.$$

Use implicit_z_taylor_jet for the positive inverse, passing the root controls.

Only the central value and time change between stages. Keep the neighbor

jets and all other parameter jets fixed during stage evolution while

retaining all their perturbation coefficients. The stages and result are

$$k_1=F(W_0,\tau),$$

$$k_2=F(W_0+\psi k_1/2,\tau+\psi/2),$$

$$k_3=F(W_0+\psi k_2/2,\tau+\psi/2),$$

$$k_4=F(W_0+\psi k_3,\tau+\psi),$$

$$R=\frac{k_1+2k_2+2k_3+k_4}{6}.$$

Each stage starts from the original central value plus its own increment.

The output is the annualized increment, not the next option-value jet.

```python
def rk4_taylor_step(
    p_coeff: list[list[list[float]]],
    psi: float = 0.002,
    residual_tol: float = 1e-13,
    max_iter: int = 24,
) -> tuple[list[list[float]], tuple[tuple[float, ...], ...]]:
    r"""Evaluate a frozen-neighbor local RK4 update in bivariate Taylor algebra.

    Each $3\times3$ matrix uses normalized coefficients
    $$A_{ij}=\frac{1}{i!j!}\partial_s^i\partial_t^j A(0,0),
    \qquad0\le i,j\le2.$$
    Rows index the $s$ degree and columns the $t$ degree. Preserve arbitrary
    higher-order incoming entries and truncate each degree separately.
    Products and analytic compositions use this Taylor algebra, not pointwise
    matrix arithmetic, finite differences, or scalar sampling.

    For a stage central jet $Y$ and time jet $\theta$, obtain $(\Delta,\Gamma)$
    by applying spatial_taylor_jets to $(W_-,Y,W_+)$ and geometry $(h,o,c)$.
    This preceding function defines the six nonuniform RBF-FD weight jets.
    Require $\Gamma_{00}>0$ at every stage and recompute
    $$x=\exp(r\theta)\alpha^2X^2\Gamma,$$
    $$x=\left(\sqrt{Z}-\frac{\operatorname{arsinh}(\sqrt{Z})}
    {\sqrt{1+Z}}\right)^2,\qquad Z_{00}>0,$$
    $$F(Y,\theta)=\frac12\sigma_0^2(1+Z)X^2\Gamma+(r-q)X\Delta-rY.$$
    Use implicit_z_taylor_jet for the positive inverse, passing the root controls.
    Only the central value and time change between stages. Keep the neighbor
    jets and all other parameter jets fixed during stage evolution while
    retaining all their perturbation coefficients. The stages and result are
    $$k_1=F(W_0,\tau),$$
    $$k_2=F(W_0+\psi k_1/2,\tau+\psi/2),$$
    $$k_3=F(W_0+\psi k_2/2,\tau+\psi/2),$$
    $$k_4=F(W_0+\psi k_3,\tau+\psi),$$
    $$R=\frac{k_1+2k_2+2k_3+k_4}{6}.$$
    Each stage starts from the original central value plus its own increment.
    The output is the annualized increment, not the next option-value jet.

    Parameters
    ----------
    p_coeff : list[list[list[float]]]
        Exactly twelve finite $3\times3$ jets in the independent order
        $$(W_-,W_0,W_+,h,o,c,r,q,\tau,\alpha,\sigma_0,X).$$
        The baseline values of $h,o,c,\alpha,\sigma_0,X$ must be positive.
        Do not re-impose the baseline shape rule under perturbation.
    psi : float, default 0.002
        Finite positive scalar step $\psi$, held constant under perturbation.
        Require $0\le\tau_{00}$ and $\tau_{00}+\psi\le1$.
    residual_tol : float, default 1e-13
        Finite relative scalar-root tolerance in $(0,1)$, forwarded unchanged.
        It uses the unsquared transformed residual defined by
        implicit_z_taylor_jet and positive_z_from_x.
    max_iter : int, default 24
        Positive integer scalar-root iteration budget, forwarded unchanged.
        All fourth-order arithmetic must remain representable in binary64.

    Returns
    -------
    tuple[list[list[float]], tuple[tuple[float, ...], ...]]
        First: the finite normalized $3\times3$ jet of $R$ as nested lists.
        Second: exactly four baseline diagnostic tuples, in stage order.
        Each tuple is $(\tau_{\mathrm{stage}},W_{\mathrm{center}},
        \Delta,\Gamma,x,Z,F)$, all evaluated at $s=t=0$.

    Raises
    ------
    ValueError
        If there are not twelve correctly shaped finite jets, root controls
        or step are invalid, a required baseline is nonpositive, baseline
        stage times leave $[0,1]$, or any baseline stage has $\Gamma\le0$.
    ArithmeticError
        If an implicit solve fails or the annualized coefficients are
        nonfinite. Native arithmetic exceptions may propagate outside the
        representable domain. Malformed objects outside the typed API may
        raise TypeError.
    """
    annualized_coeff = [[0.0] * 3 for _ in range(3)]
    stage_rows = tuple()
    return annualized_coeff, stage_rows
```

### Step 8

solve

Goal
----
Return the fourth mixed sensitivity of the local annualized RK4 step.



Coordinate order is $(W_-,W_0,W_+,h,o,c,r,q,\tau,\alpha,\sigma_0,X)$.

Form independent affine paths $p(s,t)=p_0+s u+t v$, where $s$ and $t$

correspond to the public problem's $\varepsilon$ and $\eta$.

Seed each coordinate with normalized Taylor coefficients

$$J_{00}(p_k)=p_{0,k},\qquad J_{10}(p_k)=u_k,\qquad

J_{01}(p_k)=v_k,$$

with every other coefficient zero. Rows index the $s$ degree; columns

index the $t$ degree. Call rk4_taylor_step on the twelve jets and the

supplied scalar step. Its first output is $J(R)$ for the annualized

frozen-neighbor RK4 increment. Return

$$\left.\frac{\partial^4 R}{\partial s^2\partial t^2}\right|_{(0,0)}

=2!2!J_{22}(R)=4J_{22}(R).$$

Defaults apply independently to each omitted vector. Geometry coordinates

remain independent after perturbation. Do not fit sampled function values

or round the returned derivative to the public answer's display precision.

```python
def solve(
    base: tuple[float, ...] | None = None,
    direction_s: tuple[float, ...] | None = None,
    direction_t: tuple[float, ...] | None = None,
    psi: float = 0.002,
) -> float:
    r"""Return the fourth mixed sensitivity of the local annualized RK4 step.

    Coordinate order is $(W_-,W_0,W_+,h,o,c,r,q,\tau,\alpha,\sigma_0,X)$.
    Form independent affine paths $p(s,t)=p_0+s u+t v$, where $s$ and $t$
    correspond to the public problem's $\varepsilon$ and $\eta$.
    Seed each coordinate with normalized Taylor coefficients
    $$J_{00}(p_k)=p_{0,k},\qquad J_{10}(p_k)=u_k,\qquad
    J_{01}(p_k)=v_k,$$
    with every other coefficient zero. Rows index the $s$ degree; columns
    index the $t$ degree. Call rk4_taylor_step on the twelve jets and the
    supplied scalar step. Its first output is $J(R)$ for the annualized
    frozen-neighbor RK4 increment. Return
    $$\left.\frac{\partial^4 R}{\partial s^2\partial t^2}\right|_{(0,0)}
    =2!2!J_{22}(R)=4J_{22}(R).$$
    Defaults apply independently to each omitted vector. Geometry coordinates
    remain independent after perturbation. Do not fit sampled function values
    or round the returned derivative to the public answer's display precision.

    Parameters
    ----------
    base : tuple[float, ...] or None
        Twelve finite baseline coordinates. None selects
        $$(8.4032,14.6457,22.296,10,1,30,0.1,0,0.75,0.01,0.2,100).$$
        The baseline values of $h,o,c,\alpha,\sigma_0,X$ must be positive,
        and every baseline stage must have positive Gamma.
    direction_s : tuple[float, ...] or None
        Twelve finite components of $u$. None selects
        $$(0.25,-0.5,0.75,0.1,-0.02,0.2,0.01,-0.015,0.03,
        0.0001,0.005,0.4).$$
    direction_t : tuple[float, ...] or None
        Twelve finite components of $v$. None selects
        $$(-0.4,0.2,0.1,-0.15,0.03,-0.25,-0.02,0.01,-0.04,
        -0.0002,0.008,-0.3).$$
    psi : float, default 0.002
        Finite positive constant RK step $\psi$. Require
        $0\le\tau_0$ and $\tau_0+\psi\le1$.
        Root controls use rk4_taylor_step defaults. Inputs must keep
        fourth-order calculations and the final derivative finite in binary64.

    Returns
    -------
    float
        The unrounded mixed derivative $4J_{22}(R)$, not the raw normalized
        coefficient, next option value, or a formatted string.

    Raises
    ------
    ValueError
        If any supplied vector has length other than twelve or contains a
        nonfinite entry. Validation errors from rk4_taylor_step, including
        invalid step, baseline domain, stage times, or stage Gamma, propagate.
    ArithmeticError
        If a downstream implicit solve or Taylor calculation fails; native
        arithmetic exceptions may propagate outside the representable domain.
        Malformed objects outside the typed interface may raise TypeError.
    """
    fourth_interaction = 0.0
    return fourth_interaction
```
