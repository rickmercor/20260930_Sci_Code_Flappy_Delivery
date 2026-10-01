# Mathematics-Computational_Finance-8

## Background

The Black-Scholes equation assumes that asset prices move without memory: the increments of the driving Brownian motion are independent, so the price of a claim depends on the present state alone. Empirical return series show slowly decaying autocorrelation of volatility and other signs of long-range dependence, and one way to build that into a pricing equation is to replace the first time derivative by a fractional one. With a Caputo derivative of order alpha in (0, 1] the pricing equation becomes non-local in time: the value at any moment is a weighted convolution over the whole past evolution of the solution, with a weakly singular kernel that gives recent states more weight than old ones, and alpha = 1 recovers the classical equation.

That memory changes what a good discretisation must do. A time-stepping formula for the Caputo operator has to carry the entire history at every step, so the cost grows with the number of levels, and any error made in the spatial operator is fed back into every later level rather than being damped step by step. High spatial accuracy on sparse stencils is therefore especially valuable, and the degenerate diffusion coefficient near zero asset value and the kink of the payoff at the strike make the asset direction hard to resolve with low-order formulas.

Local radial-basis-function finite differences (RBF-FD) build derivative weights on small stencils from interpolation by translates of a radial kernel, which gives sparse matrices, flexible node placement and potentially high order. Their weakness is conditioning: repeated differentiation of peaked multiquadric-type kernels produces large, oscillating factors in the shape parameter, and the local interpolation systems can be close to singular. Integrated-RBF constructions address this by building the interpolant from analytic antiderivatives of the kernel, which are smoother and flatter, and recovering the derivative operators by differentiating those primitives. This line of work applies that idea on seven-node stencils to the time-fractional Black-Scholes equation, derives closed-form weights for the first and second derivatives, and couples them to a standard fractional time-stepping scheme to price European options.

## Problem

Time-fractional Black-Scholes models replace the first time derivative of the pricing equation with a Caputo derivative, so that today's price depends on the whole history of the solution rather than on its last state, and any spatial error is carried forward and accumulated through that memory. A recent meshless discretisation addresses this with local radial-basis finite differences on seven-node stencils whose differentiation weights are generated not from a multiquadric-type kernel itself but from its analytic second antiderivative, which is meant to regularise the local interpolation.

In time to maturity tau = T - t the call value u(S, tau) solves the Caputo equation of order alpha in tau, taken from tau = 0, with right-hand side (1/2) sigma^2 S^2 u_SS + (r - q) S u_S - r u, initial value u(S, 0) = max(S - K, 0), and Dirichlet data u(0, tau) = 0 and u(S_max, tau) = S_max e^{-q tau} - K e^{-r tau} on the truncated domain [0, S_max].

Discretise the asset direction exactly as that scheme prescribes: seven-node stencils, a kernel shape parameter tied to the grid spacing, the scheme's closed-form weights for the first and second derivatives on every row that admits a centred stencil, and one-sided stencils on the first or last seven nodes for the rows next to each boundary, whose weights come from the exactness conditions of the same integrated kernel with no polynomial terms appended. Discretise the Caputo derivative with the standard L1 formula on a uniform grid in tau and march implicitly, imposing the two boundary values on the end rows at every level.

Use K = 100, r = 0.05, q = 0, sigma = 0.4, alpha = 0.8 and T = 1. The domain is [0, 300], that is S_max = 3K, carrying N = 61 equally spaced nodes including both end points, so that h = 5 and the strike is the node with index 20 counting from zero. The shape parameter is c = 4h, and the time grid has 400 uniform steps. Report the computed call value at S = K and tau = T.

In your reasoning, give numerically the value at the origin of the second antiderivative of the kernel that generates the weights, the centre weight of the interior second-derivative stencil multiplied by h^2, and the two leading L1 coefficients l_{0,n} and l_{1,n} at a level n >= 2, including the factor 1/Gamma(2 - alpha); say which grid rows are one-sided and how the memory history enters each implicit time level; and give the call value the same pipeline produces when every centred row instead takes its weights from solving the integrated-kernel exactness system.

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

01_integrated_kernels

Goal
----
Evaluate the generating kernel of the spatial scheme together with its first and second antiderivatives at a set of signed offsets.

```python
import numpy as np

def integrated_kernels(y: np.ndarray, c: float) -> np.ndarray:
    '''Evaluate the kernel and its first and second antiderivatives.

    Parameters
    ----------
    y : np.ndarray
        One-dimensional array of m signed offsets (m >= 1). A Python float is
        treated as a single offset.
    c : float
        Shape parameter of the kernel, c > 0.

    Returns
    -------
    values : np.ndarray
        Shape (m, 3) float array. Column 0 holds phi(y) = (y**2 + c**2)**(-5/2),
        column 1 its antiderivative phi1 with phi1(0) = 0, and column 2 the
        antiderivative phi2 of phi1 with phi2(0) = 1/(3 c**3).

    Raises
    ------
    ValueError
        If c is not a finite positive number, or if y is not a finite
        one-dimensional array with at least one entry.

    Notes
    -----
    The evaluation sandbox may execute this function in isolation: include
    every import your implementation needs (for example ``import numpy as np``)
    inside the function body.
    '''
    return np.zeros((np.atleast_1d(y).size, 3), dtype=float)  # placeholder
```

### Step 2

02_integrated_kernel_weights

Goal
----
Compute first- and second-derivative weights on an arbitrary stencil from the exactness conditions of the integrated kernel.

```python
import numpy as np

def integrated_kernel_weights(nodes: np.ndarray, centre: float, c: float) -> np.ndarray:
    '''Derivative weights from the integrated-kernel exactness system.

    Parameters
    ----------
    nodes : np.ndarray
        One-dimensional array of m >= 2 distinct, finite stencil nodes.
    centre : float
        Point at which the derivatives are approximated (finite).
    c : float
        Shape parameter of the kernel, c > 0.

    Returns
    -------
    weights : np.ndarray
        Shape (2, m) float array: row 0 first-derivative weights, row 1
        second-derivative weights, in the order of ``nodes``.

    Raises
    ------
    ValueError
        If nodes is not a finite one-dimensional array of at least two
        distinct values, if centre is not finite, or if c is not a finite
        positive number.

    Notes
    -----
    Include every import your implementation needs inside the function body.
    '''
    return np.zeros((2, np.asarray(nodes).size), dtype=float)  # placeholder
```

### Step 3

03_analytic_interior_weights

Goal
----
Evaluate the source's closed-form seven-node interior weights for the first and second derivatives on a uniform stencil.

```python
import numpy as np

def analytic_interior_weights(h: float, c: float) -> np.ndarray:
    '''Closed-form interior weights of the integrated-kernel scheme.

    Parameters
    ----------
    h : float
        Uniform node spacing, h > 0.
    c : float
        Shape parameter of the kernel, c > 0.

    Returns
    -------
    weights : np.ndarray
        Shape (2, 7) float array. Row 0 holds the first-derivative weights and
        row 1 the second-derivative weights for the offsets -3h, ..., 3h.

    Raises
    ------
    ValueError
        If h or c is not a finite positive number.

    Notes
    -----
    Include every import your implementation needs inside the function body.
    '''
    return np.zeros((2, 7), dtype=float)  # placeholder
```

### Step 4

04_assemble_differentiation_matrices

Goal
----
Assemble the global seven-band first- and second-derivative matrices of the integrated-kernel scheme on a uniform asset grid.

```python
import numpy as np

def assemble_differentiation_matrices(n_nodes: int, s_max: float, c_over_h: float) -> np.ndarray:
    '''Global differentiation matrices of the seven-node scheme.

    Parameters
    ----------
    n_nodes : int
        Number of grid nodes N, including both end points (N >= 7).
    s_max : float
        Right end of the asset grid [0, s_max], s_max > 0.
    c_over_h : float
        Ratio of the kernel shape parameter to the grid spacing, > 0.

    Returns
    -------
    matrices : np.ndarray
        Shape (2, N, N) float array: [0] first-derivative matrix, [1]
        second-derivative matrix. Rows 0 and N - 1 are zero.

    Raises
    ------
    ValueError
        If n_nodes is not an integer of at least 7, or if s_max or c_over_h is
        not a finite positive number.

    Notes
    -----
    Build the rows by calling ``analytic_interior_weights`` and
    ``integrated_kernel_weights`` rather than re-deriving them. Include every
    import your implementation needs inside the function body.
    '''
    return np.zeros((2, int(n_nodes), int(n_nodes)), dtype=float)  # placeholder
```

### Step 5

05_l1_caputo_coefficients

Goal
----
Compute the L1 coefficients that approximate the Caputo derivative of order alpha at one time level of a uniform grid.

```python
import numpy as np

def l1_caputo_coefficients(alpha: float, n_level: int) -> np.ndarray:
    '''L1 coefficients of the Caputo derivative at time level n.

    Parameters
    ----------
    alpha : float
        Order of the Caputo derivative, 0 < alpha <= 1.
    n_level : int
        Index n >= 1 of the time level at which the derivative is taken.

    Returns
    -------
    coefficients : np.ndarray
        Shape (n_level + 1,) float array; entry j multiplies u(tau_{n-j}).

    Raises
    ------
    ValueError
        If alpha is outside (0, 1] or n_level is not an integer >= 1.

    Notes
    -----
    Include every import your implementation needs inside the function body.
    '''
    return np.zeros(int(n_level) + 1, dtype=float)  # placeholder
```

### Step 6

06_fractional_bs_operator

Goal
----
Form the semi-discrete Black-Scholes spatial operator from the differentiation matrices, leaving the Dirichlet rows empty.

```python
import numpy as np

def fractional_bs_operator(s_grid: np.ndarray, diff_matrices: np.ndarray, sigma: float,
                           rate: float, dividend: float) -> np.ndarray:
    '''Semi-discrete spatial operator of the pricing equation.

    Parameters
    ----------
    s_grid : np.ndarray
        Strictly increasing asset grid of N >= 3 nodes.
    diff_matrices : np.ndarray
        Shape (2, N, N) array: [0] first-derivative and [1] second-derivative
        matrix on ``s_grid``.
    sigma : float
        Volatility, sigma >= 0.
    rate : float
        Risk-free rate.
    dividend : float
        Continuous dividend yield.

    Returns
    -------
    operator : np.ndarray
        Shape (N, N) float array with rows 0 and N - 1 set to zero.

    Raises
    ------
    ValueError
        If the grid is not a finite strictly increasing one-dimensional array
        of at least three nodes, if diff_matrices does not have shape
        (2, N, N) or is not finite, if sigma is negative, or if any
        coefficient is not finite.

    Notes
    -----
    Include every import your implementation needs inside the function body.
    '''
    return np.zeros((np.asarray(s_grid).size, np.asarray(s_grid).size), dtype=float)  # placeholder
```

### Step 7

07_l1_time_march

Goal
----
March the semi-discrete fractional system implicitly in time to maturity with the L1 scheme and return the terminal vector.

```python
import numpy as np

def l1_time_march(operator: np.ndarray, initial_values: np.ndarray, left_values: np.ndarray,
                  right_values: np.ndarray, alpha: float, maturity: float) -> np.ndarray:
    '''Implicit L1 time stepping of D^alpha U = L U with Dirichlet end values.

    Parameters
    ----------
    operator : np.ndarray
        Shape (N, N) spatial operator L (its first and last rows are ignored).
    initial_values : np.ndarray
        Shape (N,) solution at tau = 0.
    left_values : np.ndarray
        Shape (n + 1,) Dirichlet values of the first node at tau_0, ..., tau_n.
    right_values : np.ndarray
        Shape (n + 1,) Dirichlet values of the last node at tau_0, ..., tau_n.
    alpha : float
        Order of the Caputo derivative, 0 < alpha <= 1.
    maturity : float
        Final time, maturity > 0; the step is maturity / n.

    Returns
    -------
    terminal : np.ndarray
        Shape (N,) float array, the solution at tau = maturity.

    Raises
    ------
    ValueError
        If the shapes are inconsistent, n < 1, N < 3, any input is not
        finite, alpha is outside (0, 1], or maturity is not positive.

    Notes
    -----
    Obtain the history coefficients from ``l1_caputo_coefficients``. Include
    every import your implementation needs inside the function body.
    '''
    return np.zeros(np.asarray(initial_values).size, dtype=float)  # placeholder
```

### Step 8

08_price_at_strike

Goal
----
Chain the sub-problem functions 01-07 end-to-end and return the European call value at the strike under the time-fractional model. Orchestrator: yes - this is the final step; it integrates the earlier sub-problems (integrated_kernels, integrated_kernel_weights, analytic_interior_weights, assemble_differentiation_matrices, l1_caputo_coefficients, fractional_bs_operator, l1_time_march) rather than reimplementing them.

```python
import numpy as np

def price_at_strike(strike: float = 100.0, rate: float = 0.05, dividend: float = 0.0,
                    sigma: float = 0.4, alpha: float = 0.8, maturity: float = 1.0,
                    n_nodes: int = 61, n_steps: int = 400, smax_factor: float = 3.0,
                    c_over_h: float = 4.0) -> float:
    '''European call value at the strike under the time-fractional model.

    Parameters
    ----------
    strike : float
        Strike K > 0.
    rate : float
        Risk-free rate r.
    dividend : float
        Continuous dividend yield q.
    sigma : float
        Volatility, sigma > 0.
    alpha : float
        Order of the Caputo derivative in time to maturity, 0 < alpha <= 1.
    maturity : float
        Time to maturity T > 0.
    n_nodes : int
        Number of asset nodes N >= 7 on [0, smax_factor * K], both ends included.
    n_steps : int
        Number of uniform time steps n >= 1.
    smax_factor : float
        Right end of the asset domain in units of the strike, > 1.
    c_over_h : float
        Kernel shape parameter in units of the grid spacing, > 0.

    Returns
    -------
    value : float
        Call value at S = K and tau = T, as a native Python float.

    Raises
    ------
    ValueError
        If any argument is outside its stated domain or if the strike does not
        fall on a grid node.

    Notes
    -----
    This is the final, orchestrating step: call the public functions of
    sub-problems 01-07 and feed each returned value into the next, rather
    than reimplementing them. Include every import your implementation needs
    inside the function body.
    '''
    return 0.0  # placeholder
```
