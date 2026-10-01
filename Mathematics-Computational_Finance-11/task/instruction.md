# Mathematics-Computational_Finance-11

## Background

Continuous-time heterogeneous-agent models are the workhorse of modern quantitative macroeconomics. The Aiyagari-Bewley-Huggett tradition studies economies populated by a continuum of agents who are identical before the fact but become unequal after it, because each is hit by idiosyncratic income shocks and can only partially insure against them by saving. Achdou, Han, Lasry, Lions and Moll recast this class in the language of Mean Field Games, which turns the economy into a coupled pair of partial differential equations: a Hamilton-Jacobi-Bellman equation describing how a single agent optimises consumption given prices, and a Fokker-Planck-Kolmogorov equation describing how the cross-sectional distribution of wealth evolves under the resulting policy. Prices are not exogenous; they are pinned down by requiring that the aggregates implied by the distribution be consistent with the technology of the economy, which closes the system as a fixed point. This machinery is now standard equipment for quantifying the distributional consequences of fiscal and monetary policy.

What makes the mathematics distinctive is the borrowing constraint. Agents cannot run their wealth below an exogenous floor, and the constraint is imposed on the state rather than through a penalty, so the natural solution concept is a constrained viscosity solution. The associated Hamiltonian is finite only where the marginal value of wealth is non-negative, which places the problem outside the standard viscosity framework and requires a comparison principle to be established specifically for it. The constraint also leaves a visible economic fingerprint: agents with low income are pushed against the floor and accumulate there, so the stationary wealth measure is not a density but the sum of a density and a point mass at the constraint. That point mass is of direct economic interest, because constrained households are the ones whose consumption responds most sharply to transfers and to interest-rate changes, and it is also the part of the solution most sensitive to how the constraint is discretised.

A second strand of work replaces the traditional time-additive utility of these models with recursive preferences of the Epstein-Zin type. Time-additive utility forces a single parameter to govern two conceptually separate attitudes: aversion to risk across states, and willingness to substitute consumption across time. Epstein-Zin preferences separate them, which matters both for asset pricing and for how strongly precautionary saving responds to income risk. The separation carries a mathematical cost. The felicity aggregator of a recursive utility depends on the value function itself, not only on consumption, so the Hamiltonian inherits that dependence and the problem is no longer of the form that classical policy iteration was designed for. Whether the agent prefers early or late resolution of uncertainty turns out to change not just the economics but the structure of the numerical problem: in one regime the discretised equation obeys a comparison principle and can be attacked by a Newton-accelerated policy iteration, while in the other that principle is unavailable and convergence has to be obtained instead from monotonicity and invariance of an associated fixed-point map, in the spirit of order-theoretic fixed-point theorems used in the literature on dynamic programming over ordered spaces. Convergence proofs, barrier constructions and error estimates all have to be redone in this setting.

Solving the value function is only half of the system. The stationary distribution requires its own discretisation of the forward equation, and here more than one construction is available. One route discretises the forward equation directly as the adjoint of the upwind operator used for the value function. Another treats the economy as a discrete-time process observed at a finite step, in which an agent advances along the optimal saving policy and is redistributed onto the computational grid, and builds the stationary law as the invariant measure of that process. Which construction is used is a modelling decision rather than an implementation detail, because the two encode the borrowing constraint differently at the moment mass would otherwise be pushed past it. The same strand of work supplies the equilibrium closure, in which the interest rate is identified with the net marginal product of capital in a Cobb-Douglas technology, together with the structural identity that pins aggregate labour supply from the stationary income shares alone.

Taken together, these two strands leave a well-defined but non-trivial computational object: a stationary equilibrium of a heterogeneous-agent economy with recursive preferences, in which the value function, the wealth distribution and the interest rate must be solved simultaneously, the borrowing constraint is active and generates an atom, and the numerical treatment of both the aggregator's dependence on the value function and the forward step at finite resolution materially affects the answer.

## Problem

Continuous-time heterogeneous-agent models in the Aiyagari-Bewley-Huggett class couple a state-constrained Hamilton-Jacobi-Bellman equation for a single agent's value function with a forward equation for the stationary wealth distribution, and close the system with a market-clearing condition that determines the interest rate endogenously. Replacing time-additive utility by recursive Epstein-Zin preferences separates risk aversion $\gamma$ from the elasticity of intertemporal substitution $\psi$, which is the empirically relevant distinction, but it changes the structure of the optimisation problem in ways that the standard treatment of these models does not cover.

An agent holding wealth $x$ and receiving income $y_j$ chooses consumption $c \ge 0$ subject to $\dot{x} = rx + y_j - c$ and the borrowing constraint $x \ge \underline{x}$, with income alternating between $y_1$ and $y_2$ at Poisson rates $\lambda_1$ and $\lambda_2$.

Solve one concrete deterministic instance of this stationary equilibrium, using the following configuration:

- risk aversion $\gamma = 1.5$, elasticity of intertemporal substitution $\psi = 0.6$, discount rate $\rho = 0.045$
- incomes $y_1 = 0.4$ and $y_2 = 1.2$, with switching rates $\lambda_1 = 0.25$ and $\lambda_2 = 0.15$
- borrowing limit $\underline{x} = -0.25$
- value function and optimal policies obtained with a monotone upwind finite-difference scheme on the uniform grid covering $[-0.25,\, 24.75]$ with $1250$ intervals, the scheme being applied to the value function itself as it enters the recursive aggregator, and not to any monotone transform of it
- the value function and the policies are iterated to convergence, so that the reported value is insensitive to the stopping tolerance at the precision requested below
- wealth distribution taken to be the invariant law, normalised to unit total mass, of the discrete-time process on that grid which over one step of length $h = 1$ either advances the converged saving policy and interpolates linearly back onto the grid, or, with probability $\lambda_j h$, switches the income state and leaves wealth unchanged
- production is Cobb-Douglas with capital share $\alpha = 0.36$, depreciation $\delta = 0.08$ and total factor productivity $A = 0.5$; the equilibrium interest rate equals the net marginal product of capital, with aggregate capital the mean wealth and aggregate labour the mean income of the stationary distribution
- the equilibrium interest rate is located to at least $10^{-12}$

Report the total weight the stationary distribution places on the grid node at the borrowing limit, summed over the two income states, as a fraction of the total population and to at least eight significant figures.

Your reasoning should cover the felicity aggregator and the consumption rule it implies, the handling of the borrowing constraint in the discretisation together with the resulting value function and policies at the limit, the construction of the stationary distribution and its support, and the equilibrium closure.

## Output format

Scientific reasoning wrapped in <reasoning>...</reasoning> tags. Include enough intermediate calculations to justify the deterministic computation, without turning the response into a general pipeline summary. A single final numeric answer wrapped in <final_answer>...</final_answer> tags.

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

compute_model_constants

Goal
----
Derive the scalar constants that the recursive-utility formulation depends on, given the preference parameters and an interest rate.

Return a 1-D array of length 5, in this fixed order:

  index 0  the recursive-utility parameter combining risk aversion and the elasticity of intertemporal substitution, which is positive under the stated assumptions and whose position relative to one selects the timing regime

  index 1  the coefficient that multiplies the value function on the left-hand side of the stationary Hamilton-Jacobi-Bellman equation once the linear part of the felicity function has been absorbed into the aggregator

  index 2  the exponent carried by the value-function factor in the optimal consumption rule

  index 3  the exponent carried by the value-function factor in the modified aggregator

  index 4  the convenient constant, built from the discount rate, the elasticity of intertemporal substitution and the interest rate, that scales wealth in the supersolution barrier of the problem

Inputs are the risk aversion, the elasticity of intertemporal substitution, the discount rate and the interest rate, all real scalars. Raises ValueError if any input is not a finite real scalar, if risk aversion is not greater than one, if the elasticity of intertemporal substitution is not strictly between zero and one, or if the ordering discount rate greater than interest rate greater than zero fails.

```python
def compute_model_constants(gamma: float, psi: float, rho: float, r: float) -> np.ndarray:
    '''Return the five scalar constants of the recursive-utility formulation.

    Parameters
    ----------
    gamma : float
        Risk aversion, strictly greater than one.
    psi : float
        Elasticity of intertemporal substitution, strictly between zero and one.
    rho : float
        Subjective discount rate.
    r : float
        Interest rate, strictly between zero and the discount rate.

    Returns
    -------
    out : np.ndarray
        Shape (5,) float array holding, in order, the recursive-utility parameter, the
        coefficient multiplying the value function in the stationary equation, the exponent
        in the optimal consumption rule, the exponent in the modified aggregator, and the
        constant scaling wealth in the supersolution barrier.
    '''
    return out
```

### Step 2

compute_discrete_barriers

Goal
----
Build the two discrete grid functions that bracket any solution of the discretised Hamilton-Jacobi-Bellman equation.

The grid is uniform on the closed interval from the borrowing limit to the upper truncation point and has nint intervals, hence nint plus one nodes, with node zero at the borrowing limit.

Return a single 2-D array of shape (2 * (nint + 1), 2) obtained by stacking the two grid functions vertically. Rows 0 to nint hold the subsolution and rows nint + 1 to 2 * nint + 1 hold the supersolution. The two columns are the two income states, in the order low income then high income. Both grid functions take the same value in the two income states, so the two columns of each block are identical; the shape is kept two-dimensional so that the result lines up with the value function on the same grid.

The subsolution is built from the wealth-plus-lower-income flow. The supersolution is built from wealth measured in units that include the present value of the higher income stream, scaled by the barrier constant supplied as an argument. Both are ordinary constant-relative risk aversion expressions in those arguments.

Inputs are the risk aversion, the interest rate, the two income levels, the barrier constant, the borrowing limit, the upper truncation point and the number of intervals.

Raises ValueError if any real input is not finite, if nint is not a positive integer, if the upper truncation point does not exceed the borrowing limit, if risk aversion is not greater than one, if the interest rate or the barrier constant is not strictly positive, if the income levels do not satisfy higher income greater than lower income greater than zero, if the interest income plus lower income is not strictly positive at every node, if wealth plus the present value of the higher income stream is not strictly positive at every node, or if the resulting subsolution fails to lie at or below the resulting supersolution at every node.

```python
def compute_discrete_barriers(gamma: float, r: float, y1: float, y2: float, b: float,
                              xlow: float, xbar: float, nint: int) -> np.ndarray:
    '''Return the stacked discrete sub- and supersolution grid functions.

    Parameters
    ----------
    gamma : float
        Risk aversion, strictly greater than one.
    r : float
        Interest rate, strictly positive.
    y1 : float
        Lower income level, strictly positive.
    y2 : float
        Higher income level, strictly greater than the lower income level.
    b : float
        Barrier constant scaling wealth in the supersolution.
    xlow : float
        Borrowing limit, the first grid node.
    xbar : float
        Upper truncation point, the last grid node.
    nint : int
        Number of grid intervals, strictly positive.

    Returns
    -------
    out : np.ndarray
        Shape (2 * (nint + 1), 2) float array holding the subsolution in its first
        nint + 1 rows and the supersolution in its last nint + 1 rows, with the two
        columns being the low and high income states.
    '''
    return out
```

### Step 3

compute_candidate_consumptions

Goal
----
From a value function tabulated on the wealth grid, produce the two candidate consumption policies that a monotone upwind treatment of the recursive-utility Hamiltonian requires at every node.

The grid is uniform on the closed interval from the borrowing limit to the upper truncation point, with one node per row of the supplied value function and node zero at the borrowing limit. The value function has two columns, the two income states in the order low income then high income, and is strictly negative everywhere.

Return a single 2-D array with twice as many rows as the value function and the same two columns, obtained by stacking the two candidate policies vertically: the forward candidate in the first block of rows and the backward candidate in the second.

The returned pair must satisfy the following contract at every node, where the zero-saving consumption at a node is the interest income plus the income of that state:

  the saving rate implied by the forward candidate is greater than or equal to zero

  the saving rate implied by the backward candidate is less than or equal to zero

  at the first node the backward candidate equals the zero-saving consumption

  at the last node the forward candidate equals the zero-saving consumption

  wherever the relevant one-sided difference of the value function is not strictly positive, the forward candidate takes the zero-saving consumption and the backward candidate takes the supplied cap

The cap is a regularising upper bound on the backward candidate only. It must exceed the largest zero-saving consumption on the grid.

Inputs are the value function, the risk aversion, the elasticity of intertemporal substitution, the discount rate, the interest rate, the two income levels, the borrowing limit, the upper truncation point and the cap.

Raises ValueError if the value function is not a two-dimensional finite array with exactly two columns and at least two rows, if any of its entries is not strictly negative, if any real input is not finite, if risk aversion is not greater than one, if the elasticity of intertemporal substitution is not strictly between zero and one, if the discount rate or the interest rate is not strictly positive, if the income levels do not satisfy higher income greater than lower income greater than zero, if the upper truncation point does not exceed the borrowing limit, if the zero-saving consumption is not strictly positive at every node, or if the cap does not exceed the largest zero-saving consumption on the grid.

```python
def compute_candidate_consumptions(V: np.ndarray, gamma: float, psi: float, rho: float,
                                   r: float, y1: float, y2: float, xlow: float,
                                   xbar: float, cap: float) -> np.ndarray:
    '''Return the stacked forward and backward candidate consumption policies.

    Parameters
    ----------
    V : np.ndarray
        Shape (n, 2) strictly negative value function on the wealth grid, columns being
        the low and high income states.
    gamma : float
        Risk aversion, strictly greater than one.
    psi : float
        Elasticity of intertemporal substitution, strictly between zero and one.
    rho : float
        Subjective discount rate, strictly positive.
    r : float
        Interest rate, strictly positive.
    y1 : float
        Lower income level.
    y2 : float
        Higher income level.
    xlow : float
        Borrowing limit, the first grid node.
    xbar : float
        Upper truncation point, the last grid node.
    cap : float
        Regularising upper bound on the backward candidate.

    Returns
    -------
    out : np.ndarray
        Shape (2 * n, 2) float array holding the forward candidate in its first n rows and
        the backward candidate in its last n rows.
    '''
    return out
```

### Step 4

evaluate_fixed_policy

Goal
----
Solve for the value function that satisfies the discretised stationary equation exactly at a given, fixed pair of candidate consumption policies.

Holding the policies fixed removes the maximisation from the equation but does not make it linear: under recursive utility the felicity aggregator depends on the value function itself, so the resulting square system is nonlinear and must be solved iteratively. The value function is required to stay strictly negative throughout, because the aggregator involves a fractional power of a quantity built from it.

The grid is uniform on the closed interval from the borrowing limit to the upper truncation point, with node zero at the borrowing limit. The policies are supplied stacked, forward candidate in the first half of the rows and backward candidate in the second half, with two columns holding the two income states in the order low income then high income. The initial guess and the returned value function are single blocks of that same node count and column order.

The two income states are coupled by the Poisson switching terms, which are solved together with the drift terms rather than lagged.

Return the value function as a 2-D array with one row per node and two columns.

The iteration stops when the maximum absolute residual of the system falls below the supplied tolerance. Convergence must be reached within the supplied iteration budget.

Inputs are the stacked policies, the initial guess, the risk aversion, the elasticity of intertemporal substitution, the discount rate, the interest rate, the two income levels, the two switching rates, the borrowing limit, the upper truncation point, the residual tolerance and the iteration budget.

Raises ValueError if the stacked policies are not a two-dimensional finite array with two columns and an even number of rows of at least four, if the initial guess is not a two-dimensional finite array with two columns whose row count is half that of the policies, if any entry of the initial guess is not strictly negative, if any real input is not finite, if risk aversion is not greater than one, if the elasticity of intertemporal substitution is not strictly between zero and one, if the discount rate or the interest rate is not strictly positive, if either switching rate is not strictly positive, if the income levels do not satisfy higher income greater than lower income greater than zero, if the upper truncation point does not exceed the borrowing limit, if the zero-saving consumption is not strictly positive at every node, if the forward candidate implies a negative saving rate anywhere or the backward candidate implies a positive saving rate anywhere, if the tolerance or the iteration budget is not strictly positive, or if the residual tolerance is not reached within the iteration budget.

```python
def evaluate_fixed_policy(policies: np.ndarray, v_init: np.ndarray, gamma: float, psi: float,
                          rho: float, r: float, y1: float, y2: float, lam1: float,
                          lam2: float, xlow: float, xbar: float, tol: float,
                          max_iter: int) -> np.ndarray:
    '''Return the value function consistent with a fixed pair of consumption policies.

    Parameters
    ----------
    policies : np.ndarray
        Shape (2 * n, 2) stacked candidate consumptions, forward block then backward block.
    v_init : np.ndarray
        Shape (n, 2) strictly negative starting guess for the value function.
    gamma : float
        Risk aversion, strictly greater than one.
    psi : float
        Elasticity of intertemporal substitution, strictly between zero and one.
    rho : float
        Subjective discount rate, strictly positive.
    r : float
        Interest rate, strictly positive.
    y1 : float
        Lower income level.
    y2 : float
        Higher income level.
    lam1 : float
        Switching rate out of the low income state, strictly positive.
    lam2 : float
        Switching rate out of the high income state, strictly positive.
    xlow : float
        Borrowing limit, the first grid node.
    xbar : float
        Upper truncation point, the last grid node.
    tol : float
        Maximum absolute residual accepted as convergence.
    max_iter : int
        Maximum number of iterations allowed.

    Returns
    -------
    out : np.ndarray
        Shape (n, 2) float value function on the grid, strictly negative.
    '''
    return out
```

### Step 5

solve_value_function

Goal
----
Solve the discretised stationary Hamilton-Jacobi-Bellman equation by policy iteration and return both the converged value function and the resulting saving policy.

The grid is uniform on the closed interval from the borrowing limit to the upper truncation point, with node zero at the borrowing limit; its node count is taken from the supplied initial guess, which has one row per node and two columns holding the two income states in the order low income then high income.

The outer loop alternates two operations. It improves the pair of candidate consumption policies against the current value function, using the same forward and backward candidates and the same regularising cap as the candidate-policy construction, and it then re-solves the equation exactly at those fixed candidates. It stops when the largest absolute change in either candidate between successive outer iterations, maximised over nodes and summed over the two income states, falls below the supplied policy tolerance. The loop is initialised with both candidates equal to the zero-saving consumption, and the value function is carried from one outer iteration to the next as the starting point of the inner solve.

Return a single 2-D array with twice as many rows as the grid and two columns, obtained by stacking vertically the converged value function in the first block of rows and the saving policy in the second. The saving policy at a node is the sum of the saving rate implied by the forward candidate and the saving rate implied by the backward candidate at that node.

Inputs are the risk aversion, the elasticity of intertemporal substitution, the discount rate, the interest rate, the two income levels, the two switching rates, the borrowing limit, the upper truncation point, the initial guess for the value function, the regularising cap, the policy tolerance, the residual tolerance of the inner solve, the outer iteration budget and the inner iteration budget.

Raises ValueError if the initial guess is not a two-dimensional finite array with exactly two columns and at least two rows, if any of its entries is not strictly negative, if any real input is not finite, if risk aversion is not greater than one, if the elasticity of intertemporal substitution is not strictly between zero and one, if the discount rate or the interest rate is not strictly positive, if either switching rate is not strictly positive, if the income levels do not satisfy higher income greater than lower income greater than zero, if the upper truncation point does not exceed the borrowing limit, if the zero-saving consumption is not strictly positive at every node, if the cap does not exceed the largest zero-saving consumption on the grid, if either tolerance or either iteration budget is not strictly positive, if an inner solve fails to reach its residual tolerance within the inner budget, or if the outer loop fails to reach the policy tolerance within the outer budget.

```python
def solve_value_function(gamma: float, psi: float, rho: float, r: float, y1: float, y2: float,
                         lam1: float, lam2: float, xlow: float, xbar: float,
                         v_init: np.ndarray, cap: float, tol_policy: float,
                         tol_residual: float, max_outer: int, max_inner: int) -> np.ndarray:
    '''Return the converged value function stacked above the saving policy.

    Parameters
    ----------
    gamma : float
        Risk aversion, strictly greater than one.
    psi : float
        Elasticity of intertemporal substitution, strictly between zero and one.
    rho : float
        Subjective discount rate, strictly positive.
    r : float
        Interest rate, strictly positive.
    y1 : float
        Lower income level.
    y2 : float
        Higher income level.
    lam1 : float
        Switching rate out of the low income state, strictly positive.
    lam2 : float
        Switching rate out of the high income state, strictly positive.
    xlow : float
        Borrowing limit, the first grid node.
    xbar : float
        Upper truncation point, the last grid node.
    v_init : np.ndarray
        Shape (n, 2) strictly negative starting guess, which fixes the grid node count.
    cap : float
        Regularising upper bound on the backward candidate consumption.
    tol_policy : float
        Convergence tolerance on the change in the candidate policies.
    tol_residual : float
        Residual tolerance of each inner solve.
    max_outer : int
        Maximum number of outer policy iterations.
    max_inner : int
        Maximum number of iterations allowed in each inner solve.

    Returns
    -------
    out : np.ndarray
        Shape (2 * n, 2) float array holding the converged value function in its first n
        rows and the saving policy in its last n rows.
    '''
    return out
```

### Step 6

stationary_distribution

Goal
----
Compute the stationary wealth distribution generated by a converged saving policy, as the invariant law of a discrete-time process observed at a finite step.

The grid is uniform on the closed interval from the borrowing limit to the upper truncation point, with one node per row of the supplied saving policy and node zero at the borrowing limit. The two columns are the two income states, in the order low income then high income.

Over a single step of length h, an agent at node i in income state j does one of two things. With probability equal to one minus the switching rate of that state times h, wealth advances from the node position by h times the saving rate at that node and state, and the resulting position is distributed onto the two grid nodes bracketing it by linear interpolation, while the income state is unchanged; a landing position outside the grid is placed at the nearest endpoint. With the remaining probability the income state switches to the other one and wealth is left unchanged over that step.

Return the invariant law of that process as a 2-D array with one row per node and two columns, expressed as a density, so that the sum of all its entries multiplied by the grid spacing equals one. Under this convention the entry at node zero multiplied by the grid spacing is the weight the law places on the borrowing limit in that income state.

Inputs are the saving policy, the two switching rates, the step length, the borrowing limit and the upper truncation point.

Raises ValueError if the saving policy is not a two-dimensional finite array with exactly two columns and at least two rows, if any real input is not finite, if either switching rate is not strictly positive, if the step length is not strictly positive, if either switching rate multiplied by the step length is not strictly less than one, if the upper truncation point does not exceed the borrowing limit, or if the computed law has a negative entry.

```python
def stationary_distribution(s: np.ndarray, lam1: float, lam2: float, h: float,
                            xlow: float, xbar: float) -> np.ndarray:
    '''Return the invariant law of the one-step wealth process on the grid.

    Parameters
    ----------
    s : np.ndarray
        Shape (n, 2) saving policy on the wealth grid, columns being the low and high
        income states.
    lam1 : float
        Switching rate out of the low income state, strictly positive.
    lam2 : float
        Switching rate out of the high income state, strictly positive.
    h : float
        Step length of the discrete-time process, strictly positive.
    xlow : float
        Borrowing limit, the first grid node.
    xbar : float
        Upper truncation point, the last grid node.

    Returns
    -------
    out : np.ndarray
        Shape (n, 2) float array holding the invariant law as a density, normalised so that
        its sum multiplied by the grid spacing equals one.
    '''
    return out
```

### Step 7

compute_aggregates

Goal
----
Reduce a stationary wealth distribution to the four scalars the equilibrium condition needs.

The distribution is supplied on a uniform grid running from the borrowing limit to the upper truncation point, with one row per node, node zero at the borrowing limit, and two columns holding the two income states in the order low income then high income. It is a density, so the sum of all its entries multiplied by the grid spacing is one.

Return a 1-D array of length 4, in this fixed order:

  index 0  aggregate capital, the mean wealth under the distribution taken over all nodes and both income states

  index 1  aggregate labour supply, the mean income under the distribution

  index 2  the weight the distribution places on the borrowing limit in the low income state

  index 3  the weight the distribution places on the borrowing limit in the high income state

A weight at a node is the entry of the density at that node multiplied by the grid spacing.

Inputs are the distribution, the two income levels, the borrowing limit and the upper truncation point.

Raises ValueError if the distribution is not a two-dimensional finite array with exactly two columns and at least two rows, if any of its entries is negative, if any real input is not finite, if the income levels do not satisfy higher income greater than lower income greater than zero, if the upper truncation point does not exceed the borrowing limit, or if the total mass of the distribution differs from one by more than one part in a hundred million.

```python
def compute_aggregates(G: np.ndarray, y1: float, y2: float, xlow: float,
                       xbar: float) -> np.ndarray:
    '''Return aggregate capital, aggregate labour and the two weights at the borrowing limit.

    Parameters
    ----------
    G : np.ndarray
        Shape (n, 2) stationary distribution as a density on the wealth grid, columns being
        the low and high income states.
    y1 : float
        Lower income level.
    y2 : float
        Higher income level.
    xlow : float
        Borrowing limit, the first grid node.
    xbar : float
        Upper truncation point, the last grid node.

    Returns
    -------
    out : np.ndarray
        Shape (4,) float array holding aggregate capital, aggregate labour supply, the
        weight at the borrowing limit in the low income state, and the weight at the
        borrowing limit in the high income state, in that order.
    '''
    return out
```

### Step 8

solve_equilibrium

Goal
----
Solve the stationary equilibrium end to end and return the total weight the wealth distribution places on the borrowing limit.

This step orchestrates the whole pipeline. For a trial interest rate it derives the scalar constants of the recursive-utility formulation, builds the discrete barriers, solves the Hamilton-Jacobi-Bellman equation by policy iteration, forms the stationary distribution of the resulting saving policy at the given step length, and reduces that distribution to aggregate capital, aggregate labour supply and the two weights at the borrowing limit. The interest rate is then required to equal the net marginal product of capital of a Cobb-Douglas technology with the supplied capital share, depreciation rate and total factor productivity.

The equilibrium interest rate is located by bisection on the supplied bracket, halving until the bracket width falls below the supplied tolerance and taking the midpoint. The bracketing values must produce residuals of opposite sign, with the residual defined as the net marginal product of capital minus the trial interest rate.

The grid is uniform on the closed interval from the borrowing limit to the upper truncation point with the supplied number of intervals, and node zero is at the borrowing limit.

Three consistency checks are performed at the equilibrium and each raises on failure: the converged value function must lie between the two discrete barriers at every node and income state; re-deriving the candidate consumption policies from the converged value function must reproduce the saving policy returned by the policy iteration; and re-solving the equation at those candidate policies must reproduce the converged value function.

The policy iteration is run with a policy tolerance of 1e-7, an inner residual tolerance of 1e-12, an outer budget of 400 iterations and an inner budget of 100 iterations.

Return the sum of the two weights at the borrowing limit as a single float.

Inputs are the risk aversion, the elasticity of intertemporal substitution, the discount rate, the two income levels, the two switching rates, the borrowing limit, the upper truncation point, the number of grid intervals, the step length of the forward process, the capital share, the depreciation rate, the total factor productivity, the two bracketing interest rates, the bracket tolerance and the regularising cap.

Raises ValueError if any real input is not finite, if the number of grid intervals is not a positive integer, if the capital share is not strictly between zero and one, if the depreciation rate is negative, if the total factor productivity is not strictly positive, if the bracket is not ordered strictly increasing with both endpoints strictly positive and strictly below the discount rate, if the bracket tolerance is not strictly positive, if the residuals at the two bracketing interest rates do not have opposite signs, if aggregate capital is not strictly positive at any evaluated interest rate, if the recursive-utility parameter is below one so that the late-resolution policy iteration is not justified, or if any of the three consistency checks fails. Any error raised by the underlying stages is propagated.

```python
def solve_equilibrium(gamma: float, psi: float, rho: float, y1: float, y2: float,
                      lam1: float, lam2: float, xlow: float, xbar: float, nint: int,
                      h: float, alpha: float, delta: float, tfp: float, r_lo: float,
                      r_hi: float, tol_r: float, cap: float) -> float:
    '''Return the total stationary weight at the borrowing limit in equilibrium.

    Parameters
    ----------
    gamma : float
        Risk aversion, strictly greater than one.
    psi : float
        Elasticity of intertemporal substitution, strictly between zero and one.
    rho : float
        Subjective discount rate.
    y1 : float
        Lower income level.
    y2 : float
        Higher income level.
    lam1 : float
        Switching rate out of the low income state.
    lam2 : float
        Switching rate out of the high income state.
    xlow : float
        Borrowing limit, the first grid node.
    xbar : float
        Upper truncation point, the last grid node.
    nint : int
        Number of grid intervals.
    h : float
        Step length of the forward process.
    alpha : float
        Capital share of the Cobb-Douglas technology.
    delta : float
        Depreciation rate.
    tfp : float
        Total factor productivity.
    r_lo : float
        Lower bracketing interest rate.
    r_hi : float
        Upper bracketing interest rate.
    tol_r : float
        Bracket width below which the bisection stops.
    cap : float
        Regularising upper bound on the backward candidate consumption.

    Returns
    -------
    out : float
        Total weight the equilibrium stationary distribution places on the borrowing limit,
        summed over the two income states.
    '''
    return out
```
