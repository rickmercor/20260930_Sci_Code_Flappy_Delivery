# Mathematics-Computational_Finance-41

## Background

The source develops barycentric rational collocation and implicit-explicit BDF time stepping for jump-diffusion pricing. This task uses its American extension in Section 5: a linear intermediate solve and local Ikonen--Toivanen correction evolve a price and a constraint multiplier. Corrected histories, not unconstrained European prices, drive later nonlocal jump terms.

The spatial basis, analytic derivatives and Gaussian quadrature come from Sections 3--4. The exterior integral is the undiscounted American tail in Section 4. The grid is the Section 6 sinh construction. The task explicitly chooses the Section 5 IMEX-BDF1 startup, one correction per layer, and the finite left boundary $K-Ke^a$ from Section 2 instead of the limiting $K$ printed in Equation (5.2). These conventions also make the homogeneous obstacle-boundary transformation in Section 5.2 consistent.

The final Greek differentiates the same rational interpolant at fixed strike. The challenge is a coupled nonlocal obstacle evolution followed by analytic rational differentiation, not an unrelated extension, hidden interface assumption or excessive output precision.

This revision asks for a parameter curvature of the same finite method. The paper does not derive the jump-mean sensitivity recurrence: that is the task's authored extension. Its construction depends on the source's compensated generator, nonlocal rational quadrature, exterior integral, multistep intermediate systems and coupled IT correction. Parameter differentiation must include all of these, and the nonsmooth correction requires checking local branch stability. A continuous-model Greek or a frozen multiplier is a different quantity.

## Problem

Compute the jump-mean curvature of the Gamma of an American put using the supplied article's barycentric rational collocation, IMEX-BDF2 and Ikonen--Toivanen splitting method. The requested sensitivity differentiates the specified finite-grid numerical algorithm with all other inputs fixed. The article supplies the primal pricing method; the parameter-sensitivity calculation is an authored diagnostic of that method.

Use log-price $x=\log(S/K)$ and elapsed time from the payoff. The Merton parameters are
$$K=100,\quad T=0.25,\quad r=0.05,\quad\sigma=0.15,\quad\lambda=0.1,\quad\mu_J=-0.9,\quad s_J=0.45.$$
The log jump is normal with mean $\mu_J$ and standard deviation $s_J$. Recover the compensated differential operator and the American exterior-tail contribution from Sections 2 and 4. Use the American payoff outside the left endpoint, with zero right exterior contribution; there is no discount factor in this exterior payoff.

Fix $193$ spatial nodes
$$z_i=-1+\frac{2i}{192},\qquad x_i=1.5\frac{\sinh(0.4z_i)}{\sinh(0.4)},\qquad 0\leq i\leq192.$$
Use the degree-$3$ Floater--Hormann rational basis of Section 3 and its analytic first and second derivatives. Define differentiation matrices with evaluation points in rows and basis functions in columns. Form the interior jump matrix using one global $256$-point Gauss--Legendre rule mapped from $[-1,1]$ to $[-1.5,1.5]$, interpolating values at quadrature nodes with that same rational basis. Keep jump intensity separate from the interior matrix and exterior tail.

Initialize corrected prices $U_i^0=P_i=\max(K-Ke^{x_i},0)$ and constraint multipliers $\phi_i^0=0$. Take $121$ uniform time intervals of length $\Delta t=T/121$. Use the Section 5.1 IMEX-BDF1 startup with one IT correction, then the IMEX-BDF2 intermediate solve and IT correction in Equations (5.2)--(5.4). Apply exactly one correction per layer; carry the corrected price history and multiplier forward. The explicit interior jump term uses the previous price at startup and $2U^n-U^{n-1}$ thereafter. The time-independent American exterior tail is added once, and the multiplier in each intermediate equation is $\phi^n$, not an extrapolated multiplier. Do not iterate the splitting to convergence or introduce extra damping steps.

To fix the article's finite-boundary ambiguity, use the Section 2 values $U_0^n=K-Ke^{-1.5}$ and $U_{192}^n=0$ at every time layer, including each intermediate solve, rather than the limiting value $K$ printed in Equation (5.2). Retain boundary-column contributions in interior equations. Apply the obstacle correction only at interior nodes and set endpoint multipliers exactly to zero. Use correction scale $\Delta t$ at startup and $2\Delta t/3$ thereafter, with the initialization $\phi^0=0$ also in the startup condition. These conventions take precedence over inconsistent endpoint or startup notation in the article.

Let $u_h(x,T)$ interpolate the final corrected prices. At $S_\star=93.7$, calculate
$$\Gamma_h(S_\star)=\frac{\partial_x^2u_h(\log(S_\star/K),T)-\partial_xu_h(\log(S_\star/K),T)}{S_\star^2}.$$
The requested single scalar is
$$C_h=\left.\frac{\partial^2\Gamma_h(S_\star;\mu_J)}{\partial\mu_J^2}\right|_{\mu_J=-0.9}.$$
Hold the grid, quadrature rule, time intervals, strike, spot and all other model parameters fixed. Compute ordinary derivatives, not factorial-scaled Taylor coefficients. Derive and implement analytic first and second parameter sensitivities through the full finite algorithm, including the compensated differential operator, interior and exterior jump terms, implicit linear systems, extrapolation, and carried IT multiplier. Differentiate the same rational interpolant analytically to obtain the Gamma functional. Finite differences, complex-step perturbations and automatic-differentiation packages may be used for independent checking, but do not replace the required analytic sensitivity construction.

The payoff and endpoint values are independent of $\mu_J$, so their first and second parameter derivatives are zero. The target is the ordinary derivative of the complete finite algorithm. Justify its local interpretation by checking that the interior exercise/continuation decisions are stable under sufficiently small positive and negative changes of $\mu_J$; a silently frozen active set is not a substitute. At a degenerate exact correction tie, the reusable code interface prescribes the exercise branch, although such a convention alone does not establish existence of an ordinary second derivative. No exact interior ties occur in the default fixture. Explain the source-method construction and the differentiated calculation with enough numerical or executable evidence to substantiate the result. There is no required list of printed checkpoints or mesh-convergence study. Preserve double-precision intermediate values and round only $C_h$ to six decimal places.

## Output format

Put the scientific explanation in `<reasoning>...</reasoning>`, followed by `<final_answer>...</final_answer>` containing exactly one finite decimal with six digits after the decimal point, without units or a box.

## Output format

```
## Output format

Put the scientific explanation in `<reasoning>...</reasoning>`, followed by `<final_answer>...</final_answer>` containing exactly one finite decimal with six digits after the decimal point, without units or a box.
```

## Your task

Implement **all 10 functions** below into a single file at `/app/solution.py`. Requirements:

- Use the EXACT function name and signature shown for each step.
- Define every function at module scope (importable).
- Do NOT include tests, example usage, prints, or a `__main__` block.
- Available libraries: numpy, scipy, sympy, pandas, scikit-learn, networkx.

Start your file with these imports:

```python
import numpy as np
```

### Step 1

clustered_grid

Goal
----
Construct the prescribed sinh-clustered log-price grid, including the uniform-grid limit and exact endpoints. Follow the complete domain and return-shape contract in the function docstring.

```python
def clustered_grid(left: float, right: float, count: int, alpha: float = 0.4) -> "np.ndarray":
    r"""Return the prescribed sinh-clustered log-price grid.

    With $z_i=-1+2i/(M-1)$, midpoint $c=(a+b)/2$ and half-width
    $h=(b-a)/2$, return $x_i=c+h\sinh(\alpha z_i)/\sinh(\alpha)$.
    At $\alpha=0$, use the continuous limit $x_i=c+hz_i$. Set the
    first and last entries to $a$ and $b$ exactly. The count $M$ denotes
    nodes, not intervals; this task fixes the indexing convention in Section 6.

    Parameters
    ----------
    left : float
        Finite left endpoint $a$.
    right : float
        Finite right endpoint $b>a$.
    count : int
        Number of nodes $M\geq2$.
    alpha : float, default 0.4
        Finite clustering strength $\alpha\geq0$; zero gives uniform spacing.

    Returns
    -------
    numpy.ndarray, shape $(\mathrm{count},)$
        Increasing binary64 log-price coordinates including both endpoints.

    Raises
    ------
    ValueError
        If the numeric arguments violate the stated ordering, finiteness,
        integer-count or nonnegative-clustering conditions.

    Inputs must keep the displayed expressions finite and the resulting
    nodes distinct in binary64 arithmetic; behavior beyond this domain is
    otherwise unspecified.
    """
    return None
```

### Step 2

floater_hormann_weights

Goal
----
Construct normalized Floater--Hormann barycentric weights on the supplied increasing nodes. Preserve the specified degree convention, common-sign normalization and input validation.

```python
def floater_hormann_weights(nodes: "np.ndarray", degree: int) -> "np.ndarray":
    r"""Return normalized Floater--Hormann rational interpolation weights.

    For $M$ increasing nodes and degree $d$, define
    $$\widehat w_i=(-1)^i\sum_{k=\max(0,i-d)}^{\min(i,M-d-1)}
    \prod_{\substack{j=k\\j\ne i}}^{k+d}|x_i-x_j|^{-1},\qquad
    w_i=\widehat w_i/\max_j|\widehat w_j|.$$
    An empty product is one. The first weight is positive and subsequent
    signs alternate. This fixes the immaterial common sign in Section 3.

    Parameters
    ----------
    nodes : array_like, shape $(M,)$
        Finite strictly increasing real nodes, $M\geq2$.
    degree : int
        Local interpolation degree $0\leq d<M$.

    Returns
    -------
    numpy.ndarray, shape $(M,)$
        Weights in node order, with maximum absolute value one.

    Raises
    ------
    ValueError
        If the node vector or numeric degree violates its stated domain.

    Products and sums must remain finite and nonzero in binary64 arithmetic;
    behavior outside this representable domain is otherwise unspecified.
    """
    return None
```

### Step 3

rational_evaluation

Goal
----
Evaluate rational cardinal functions and their first two analytic derivatives at arbitrary query points, including exact and near-node limits. Preserve evaluation-row and interpolation-column ordering.

```python
def rational_evaluation(nodes: "np.ndarray", weights: "np.ndarray", points: "np.ndarray") -> tuple["np.ndarray", "np.ndarray", "np.ndarray"]:
    r"""Evaluate rational cardinal bases and their first two derivatives.

    The rational cardinal functions are
    $$\ell_j(p)=\frac{w_j/(p-x_j)}{\sum_k w_k/(p-x_k)}.$$
    Return $E_{ij}=\ell_j(p_i)$, $(D_1)_{ij}=\ell'_j(p_i)$ and
    $(D_2)_{ij}=\ell''_j(p_i)$, with derivatives taken with respect to $p$.
    At an exact node, use the analytic removable limits; the corresponding
    row of $E$ is a unit vector. Evaluate derivatives accurately near nodes.
    The second derivative belongs to this rational interpolant: do not use
    finite differences or square a nodal first-derivative matrix.

    Parameters
    ----------
    nodes : array_like, shape $(M,)$
        Finite strictly increasing real nodes, $M\geq2$.
    weights : array_like, shape $(M,)$
        Finite nonzero barycentric weights. Common nonzero rescaling has no
        effect on the result.
    points : float or array_like, shape $(Q,)$
        Finite evaluation coordinates; a scalar produces one output row.
        An empty vector produces three arrays with shape $(0,M)$.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]
        Exactly $(E,D_1,D_2)$, each binary64 with shape $(Q,M)$; rows are
        evaluation points and columns are interpolation nodes.

    Raises
    ------
    ValueError
        If the shapes, finiteness, node ordering or nonzero weights are invalid.
    ArithmeticError
        If the evaluated rational denominator is zero or nonfinite.

    Inputs must otherwise allow finite binary64 interpolation and derivative
    values. The analytic denominator is assumed nonzero away from nodes.
    """
    return None
```

### Step 4

merton_operators

Goal
----
Build the compensated Merton differential matrix and the global Gauss--Legendre interior jump matrix from the preceding interpolation routines. Keep the jump intensity and exterior tail separate as specified.

```python
def merton_operators(nodes: "np.ndarray", degree: int, quadrature_order: int,
                     r: float, sigma: float, intensity: float, jump_mean: float,
                     jump_std: float) -> tuple["np.ndarray", "np.ndarray"]:
    r"""Build the Merton differential and interior jump-integral matrices.

    Let $\lambda$ be the intensity, $\mu_J,s_J$ the log-jump mean and
    standard deviation, and $\zeta=\exp(\mu_J+s_J^2/2)-1$. With nodal
    rational derivative matrices from the previous steps, form
    $$D=\tfrac12\sigma^2D_2+
    (r-\tfrac12\sigma^2-\lambda\zeta)D_1-(r+\lambda)I.$$
    If $(y_q,\omega_q)$ is the $Q$-point Gauss--Legendre rule mapped to
    $[x_0,x_{M-1}]$, including the mapping factor in $\omega_q$, set
    $$J_{ij}=\sum_{q=1}^Q\omega_q\ell_j(y_q)
    \frac{\exp[-(y_q-x_i-\mu_J)^2/(2s_J^2)]}{\sqrt{2\pi}s_J}.$$
    Thus $J$ excludes intensity and exterior tails. Preserve all matrix rows;
    the time integrator imposes the boundary conditions. Call
    floater_hormann_weights and rational_evaluation to construct these objects.

    Parameters
    ----------
    nodes : array_like, shape $(M,)$
        Finite strictly increasing log-price nodes, $M\geq2$.
    degree : int
        Floater--Hormann degree $0\leq d<M$.
    quadrature_order : int
        Number $Q\geq1$ of Gauss--Legendre nodes over the entire interval.
    r : float
        Finite risk-free rate.
    sigma : float
        Finite strictly positive diffusion volatility.
    intensity : float
        Finite nonnegative Poisson jump intensity $\lambda$.
    jump_mean : float
        Finite mean $\mu_J$ of the normally distributed log jump.
    jump_std : float
        Finite strictly positive log-jump standard deviation $s_J$.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray]
        Exactly $(D,J)$, two dense binary64 arrays of shape $(M,M)$ with
        evaluation-node rows and value-node columns.

    Raises
    ------
    ValueError
        If numeric counts, model parameters or the node vector are invalid.
    ArithmeticError
        If rational_evaluation encounters an invalid rational denominator.

    Parameters and nodes must keep all intermediate formulas finite in
    binary64; behavior outside this representable domain is unspecified.
    """
    return None
```

### Step 5

american_boundary_tail

Goal
----
Compute the undiscounted American put boundary and analytic Gaussian exterior tail, using the complete domain, formula and return-shape contract in the raw docstring.

```python
def american_boundary_tail(nodes: 'np.ndarray', strike: 'float', jump_mean: 'float', jump_std: 'float') -> 'tuple':
    r"""Return the time-independent American Merton boundary and exterior tail.

    Write $a=x_0$ and $z_i=(a-x_i-\mu_J)/s_J$. The task adopts the
    finite-boundary convention in Section 2, not the limiting strike value
    printed in Equation (5.2): $b=(K-Ke^a,0)$. Section 4 gives
    $$R_i=K\Phi(z_i)-K\exp(x_i+\mu_J+s_J^2/2)\Phi(z_i-s_J).$$
    Here $\Phi$ is the standard normal CDF. Neither discounting nor jump
    intensity is included. The exterior payoff is used for all elapsed times.

    Parameters
    ----------
    nodes : array_like, shape $(M,)$
        Finite strictly increasing log-price nodes, $M\geq2$, with
        $x_0<0<x_{M-1}$.
    strike : float
        Finite strike $K>0$.
    jump_mean : float
        Finite normal log-jump mean $\mu_J$.
    jump_std : float
        Finite normal log-jump standard deviation $s_J>0$.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray]
        Boundary values of shape $(2,)$, left then right, and exterior
        integral of shape $(M,)$, in node order, without rounding.

    Raises
    ------
    ValueError
        For invalid shape, ordering, finiteness or stated sign conditions.

    All displayed expressions must remain finite in binary64; behavior
    outside that representable domain is unspecified.
    """
    return None
```

### Step 6

imex_american

Goal
----
Advance both corrected prices and constraint multipliers with one IMEX-BDF1 startup and one IT correction per layer, followed by IMEX-BDF2. Return both complete histories and respect the explicitly specified boundary convention.

```python
def imex_american(nodes: 'np.ndarray', D: 'np.ndarray', J: 'np.ndarray', strike: 'float', intensity: 'float', jump_mean: 'float', jump_std: 'float', maturity: 'float', steps: 'int') -> 'tuple':
    r"""Advance price and multiplier with exactly one IT correction per layer.

    Set $P_i=\max(K-Ke^{x_i},0)$, $U^0=P$, $\phi^0=0$ and
    $\Delta t=T/N$. Obtain $b,R$ from american_boundary_tail.
    The startup intermediate interior equations are
    $$(I-\Delta t D)\widetilde U^1=U^0+
      \Delta t\{\lambda(JU^0+R)+\phi^0\}.$$
    At later layers, for $n=1,\ldots,N-1$, use
    $$(3I-2\Delta t D)\widetilde U^{n+1}=4U^n-U^{n-1}
      +2\Delta t\{\lambda[J(2U^n-U^{n-1})+R]+\phi^n\}.$$
    In each linear system, impose the two endpoint values $b$ and retain
    their column contributions in the interior equations. Use the corrected
    histories in the extrapolation, not intermediate histories.
    For each interior component perform the Section 5 IT correction
    $$U^{n+1}=\max(P,\widetilde U^{n+1}-h\phi^n),\qquad
      \phi^{n+1}=\max(0,\phi^n+(P-\widetilde U^{n+1})/h),$$
    where $h=\Delta t$ at startup and $h=2\Delta t/3$ thereafter.
    Maxima are componentwise. Keep corrected endpoint prices equal to $b$
    and endpoint multipliers exactly zero. There are no inner iterations,
    multiplier extrapolation, damping steps or post-hoc clipping elsewhere.
    The multiplier formula is algebraically equivalent to Equation (5.4)
    and avoids cancellation on continuation nodes.

    Parameters
    ----------
    nodes : array_like, shape $(M,)$
        Finite increasing log-price nodes, $M\geq2$, $x_0<0<x_{M-1}$.
    D : array_like, shape $(M,M)$
        Finite local differential matrix, including compensated drift and loss.
    J : array_like, shape $(M,M)$
        Finite interior jump matrix without intensity or exterior tail.
    strike : float
        Finite positive strike $K$.
    intensity : float
        Finite jump intensity $\lambda\geq0$.
    jump_mean : float
        Finite normal log-jump mean $\mu_J$.
    jump_std : float
        Finite positive normal log-jump standard deviation $s_J$.
    maturity : float
        Finite positive final elapsed time $T$.
    steps : int
        Number of time intervals $N\geq1$. A single interval uses only startup.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray]
        Price history then multiplier history, each of shape $(N+1,M)$,
        with time rows and node columns. Row zero is exactly payoff and zero,
        respectively. Outputs are unrounded binary64 arrays.

    Raises
    ------
    ValueError
        If shapes, finiteness, ordering, signs or integer count are invalid.

    The boundary-modified systems must be nonsingular and all intermediate
    expressions finite in binary64. Behavior outside that domain is unspecified.
    """
    return None
```

### Step 7

put_greeks

Goal
----
Compute price, delta and gamma from analytic derivatives of the same rational interpolant at the supplied spot. Use the fixed-strike log-price chain rule and preserve full precision.

```python
def put_greeks(nodes: "np.ndarray", weights: "np.ndarray", values: "np.ndarray",
               spot: float, strike: float) -> tuple[float, float, float]:
    r"""Evaluate put price, delta and gamma from the same rational interpolant.

    At $x_*=\log(S/K)$, obtain $(E,D_1,D_2)$ by calling
    rational_evaluation with the supplied nodes and weights. With nodal
    values $U$, compute $u=EU$, $u_x=D_1U$, and $u_{xx}=D_2U$.
    Return
    $$(V,\Delta,\Gamma)=\left(u,\frac{u_x}{S},
       \frac{u_{xx}-u_x}{S^2}\right).$$
    The derivatives are with respect to the underlying price $S$, holding
    the strike fixed. Use analytic rational derivatives, including exact-node
    limits, without finite differencing, interpolation substitution or rounding.

    Parameters
    ----------
    nodes : array_like, shape $(M,)$
        Finite strictly increasing log-price nodes, $M\geq2$.
    weights : array_like, shape $(M,)$
        Finite nonzero rational interpolation weights matching the nodes.
    values : array_like, shape $(M,)$
        Finite nodal option values on this same grid.
    spot : float
        Finite strictly positive underlying price $S$ such that
        $\log(S/K)\in[x_0,x_{M-1}]$.
    strike : float
        Finite strictly positive strike $K$ defining the log-price coordinate.

    Returns
    -------
    tuple[float, float, float]
        Exactly the price, delta and gamma in that order, as unrounded scalars.

    Raises
    ------
    ValueError
        If the arrays or prices violate their stated shape, finiteness,
        ordering, sign or evaluation-interval conditions.
    ArithmeticError
        If rational_evaluation encounters an invalid denominator.

    Inputs must allow finite binary64 outputs; behavior outside the stated
    representable domain is otherwise unspecified.
    """
    return None
```

### Step 8

merton_mean_jets

Goal
----
Derive and assemble analytic first and second jump-mean derivatives of the compensated differential operator, the quadrature jump operator and the American exterior integral. Return ordinary derivatives in the documented tensor order.

```python
def merton_mean_jets(nodes: 'np.ndarray', degree: 'int', quadrature_order: 'int', r: 'float', sigma: 'float', intensity: 'float', jump_mean: 'float', jump_std: 'float', strike: 'float') -> 'tuple':
    r"""Return the first three jump-mean jets of the finite Merton operators.

    For $q=0,1,2$, the entries in slot $q$ are the ordinary derivatives
    $\partial_{\mu_J}^q D$, $\partial_{\mu_J}^q J$, and
    $\partial_{\mu_J}^q R$. These are derivatives, not Taylor coefficients.
    $D,J$ are exactly the objects defined by merton_operators and $R$ is
    the intensity-free American tail defined by american_boundary_tail.
    Hold nodes, degree, quadrature nodes/weights, strike and every parameter
    except jump_mean fixed. Differentiate the compensated drift as well as
    the normal density and the exterior integral. Compute analytic jets;
    finite differences or complex perturbations of a pricing routine are
    not the requested operator. Slot zero must call the preceding routines.

    Parameters
    ----------
    nodes : array_like, shape $(M,)$
        Finite increasing log-price nodes, $M\geq2$, straddling zero.
    degree : int
        Floater--Hormann degree $0\leq d<M$.
    quadrature_order : int
        Global Gauss--Legendre order $Q\geq1$.
    r, sigma, intensity, jump_mean, jump_std : float
        Same domains and meaning as merton_operators: finite rate and mean,
        positive volatility and jump standard deviation, nonnegative intensity.
    strike : float
        Finite positive strike, held fixed in the differentiation.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]
        $(D_{\rm jet},J_{\rm jet},R_{\rm jet})$, shapes $(3,M,M)$,
        $(3,M,M)$ and $(3,M)$. The leading axis is derivative order;
        matrix rows are evaluation nodes and columns are value nodes.

    Raises
    ------
    ValueError
        For violations of the stated numeric domains and shapes.
    ArithmeticError
        For a nonfinite or zero rational denominator.

    The constituent formulas must have finite binary64 values.
    """
    return None
```

### Step 9

differentiate_american

Goal
----
Derive and propagate first and second jump-mean derivatives through the complete discrete American IMEX-BDF1/BDF2 and IT evolution. Account for the derivative of the implicit matrix, jump extrapolation and carried constraint multiplier.

```python
def differentiate_american(nodes: 'np.ndarray', D_jets: 'np.ndarray', J_jets: 'np.ndarray', R_jets: 'np.ndarray', prices: 'np.ndarray', multipliers: 'np.ndarray', intensity: 'float', maturity: 'float') -> 'tuple':
    r"""Differentiate the complete IMEX/IT trajectory twice in jump mean.

    The supplied prices and multipliers are the base histories from
    imex_american, at the same nodes and model parameters used to form
    merton_mean_jets. Differentiate that finite algorithm, including its
    implicit systems, explicit jump extrapolation, carried multiplier,
    and the IT complementarity correction. The payoff, finite endpoint
    values, grid, quadrature and time intervals are independent of jump mean.
    Use analytic differentiation; do not estimate these derivatives from
    perturbed base solves, automatic differentiation packages, or complex steps.
    Ordinary derivatives and their product rules apply, not factorial-scaled
    Taylor coefficients. The base histories and operator jets are assumed
    mutually consistent; no reconstruction or reoptimization of them is needed.

    At each interior correction, a price strictly above its initial payoff
    identifies continuation. A price equal to payoff identifies exercise;
    for a degenerate exact tie with zero multiplier, use the exercise branch
    for derivative propagation. This makes the interface deterministic at
    ties. On a locally stable base branch pattern it equals differentiation
    of the full finite algorithm. The default task must be checked for this
    stability before interpreting the result as an ordinary derivative.
    Initial derivatives and all endpoint derivatives are exactly zero.

    Parameters
    ----------
    nodes : array_like, shape $(M,)$
        Finite increasing log-price nodes, $M\geq2$, straddling zero.
    D_jets, J_jets : array_like, shape $(3,M,M)$
        Finite matrices in derivative-order, evaluation-node, value-node order.
    R_jets : array_like, shape $(3,M)$
        Finite intensity-free exterior integral derivatives.
    prices, multipliers : array_like, shape $(N+1,M)$
        Consistent base histories, $N\geq1$. Row zero gives payoff and zero.
    intensity : float
        Finite nonnegative jump intensity, held fixed.
    maturity : float
        Finite positive horizon; $\Delta t=T/N$.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray]
        Price jets and multiplier jets, each shape $(3,N+1,M)$.
        Slot zero is a copy of the supplied history. Slots one and two
        are its first and second jump-mean derivatives, without rounding.

    Raises
    ------
    ValueError
        For invalid shape, ordering, finiteness or stated sign conditions.

    The boundary-modified linear systems must be nonsingular and all
    analytic derivatives must remain finite in binary64 arithmetic.
    """
    return None
```

### Step 10

solve

Goal
----
Compose the source-method primal solver and its analytic sensitivities to compute the second jump-mean derivative of the final rational Gamma.

```python
def solve(spot: 'float'=93.7, strike: 'float'=100.0, maturity: 'float'=0.25, r: 'float'=0.05, sigma: 'float'=0.15, intensity: 'float'=0.1, jump_mean: 'float'=-0.9, jump_std: 'float'=0.45, node_count: 'int'=193, steps: 'int'=121, quadrature_order: 'int'=256, degree: 'int'=3, half_width: 'float'=1.5, alpha: 'float'=0.4) -> 'float':
    r"""Return the second jump-mean derivative of finite-grid American Gamma.

    Compose clustered_grid, floater_hormann_weights, merton_mean_jets,
    imex_american, differentiate_american and put_greeks. The operator-jet
    routine in turn composes merton_operators, american_boundary_tail and
    rational_evaluation. Differentiate the finite algorithm with all inputs
    except jump_mean fixed. The output is $\partial_{\mu_J}^2\Gamma_h(S)$;
    it is neither Gamma itself nor the factorial-scaled quadratic coefficient.
    At degenerate ties use the derivative convention of differentiate_american.

    Parameters
    ----------
    spot, strike, maturity : float
        Finite positive underlying, strike, and elapsed horizon.
    r, sigma, intensity, jump_mean, jump_std : float
        Merton model parameters: finite rate and mean, positive sigma and
        jump_std, nonnegative intensity.
    node_count, steps, quadrature_order, degree : int
        Respectively $M\geq2$, $N\geq1$, $Q\geq1$, and $0\leq d<M$.
    half_width : float
        Finite positive $X$ defining the symmetric domain $[-X,X]$.
        The query must satisfy $|\log(S/K)|\leq X$.
    alpha : float
        Finite nonnegative sinh-clustering parameter; zero means uniform.

    Returns
    -------
    float
        Unrounded second jump-mean derivative of rational Gamma. The
        default parameters are those printed in the function signature.

    Raises
    ------
    ValueError
        For violations of any constituent input domain.
    ArithmeticError
        For an invalid rational denominator.

    All constituent expressions must be finite in binary64 and linear
    systems nonsingular. Ordinary-derivative interpretation requires a
    locally stable branch pattern, as stated in the task.
    """
    return None
```
