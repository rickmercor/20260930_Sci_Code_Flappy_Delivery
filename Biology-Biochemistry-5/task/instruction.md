# Biology-Biochemistry-5

## Background

Intracellular biochemistry runs at low copy numbers, so the state of a reaction network is a vector of integers that changes by discrete jumps whenever a reaction fires. The standard model is a continuous-time Markov chain whose transition rates are the mass-action propensities of the reactions, and whose probability distribution obeys the chemical master equation. Because the copy number of a produced species is unbounded, the master equation is an infinite system of linear ordinary differential equations, and questions about the mean and the variance of a copy number, which are the quantities that quantify intrinsic noise, cannot be answered by solving it directly.

Moment equations are the usual shortcut. Applying the generator of the chain to a monomial of the copy number and taking expectations gives an ordinary differential equation for that moment. When every propensity is at most linear in the state, the equations for the moments up to a given order close and can be solved exactly. As soon as a bimolecular reaction is present, the equation for a moment of order k involves a moment of order k + 1, and the hierarchy never closes. Moment-closure schemes truncate it by an ansatz on the higher moments; they are widely used but provide no bound on the error they introduce, and they can produce unphysical results such as negative variances.

A different route works with the Kolmogorov backward equation, which propagates conditional expectations of a test function as a function of the initial state rather than propagating the distribution. Restricting the backward equation to a finite window of copy numbers turns it into a finite linear time-invariant system, with the conditional expectations at the states just outside the window appearing as an input. The key structural fact is monotonicity: the off-diagonal entries of the restricted operator are transition rates and hence nonnegative, so a larger input can only produce a larger output. This ordering is what allows bounds on the unknown boundary input to be converted into bounds on the moment of interest that hold at every time, and it means that the quality of the resulting bracket is controlled by how well the boundary input is bounded and by how large the window is.

Bounds on the boundary input follow from the observation that for mass-action propensities the generator maps a polynomial to a polynomial, and that the leading terms can be dominated by polynomials of lower degree on the nonnegative integers. Comparison arguments for ordinary differential equations then give closed, finite linear systems whose solutions bound the conditional moments started from a boundary state from above and from below. Combining such bounds for the first two moments yields certified brackets on the variance, which is the quantity of interest when characterizing the noise of gene expression, enzymatic turnover and other small-number cellular processes.

## Problem

Stochastic reaction networks are modelled as continuous-time Markov chains on molecular copy numbers, and the chemical master equation that governs their probability distribution is infinite-dimensional whenever the copy number is unbounded. For networks with bimolecular reactions the moment equations do not close, so transient moments are usually obtained by moment-closure approximations or stochastic simulation, neither of which comes with a guarantee. A recent line of work removes this gap by producing guaranteed, time-dependent upper and lower bounds on transient moments directly: it writes the Kolmogorov backward equation on a truncated window of states, treats the unknown conditional moment that would start from the first state outside the window as an external input to a monotone linear time-invariant system, and shows that substituting computable upper and lower bounds for that input yields outputs that bracket the true moment at every time. The input bounds themselves come from polynomial sandwiches on the generator acting on monomials, which close into a small linear system for the moments started at the boundary state. The inputs of the method are the network with its mass-action propensities and rate constants, the initial state, the truncation size and the evaluation time; the output is a certified bound on a transient moment.

Consider the dimerization network with a single species X and three reactions: production with propensity theta_1, degradation with propensity theta_2 x, and dimerization, which removes two copies of X, with propensity theta_3 x (x - 1), where x is the current copy number. Use theta_1 = 5, theta_2 = ln(2) / 20 and theta_3 = 0.02, start from X(0) = 0, and take the truncated window to be the copy numbers 0 through N - 1 with N = 20, so that the only state outside the window reachable in one jump from the window is the copy number N. The evaluation time is t = 20.

Set up the truncated backward equation for the test functions x and x^2 and identify how the window couples to the boundary state. Construct the upper polynomial bounds on the generator applied to x and to x^2 in the specific constructive form the approach uses for mass-action networks, assemble the closed linear system whose solution bounds the boundary conditional moments from above, adopt the lower input bound the approach prescribes, and integrate the upper and lower bounding systems exactly over [0, t]. From the resulting upper and lower bounds on E[X(t)] and on E[X(t)^2], form the guaranteed upper bound V_plus(t) on the variance Var[X(t)] = E[X(t)^2] - (E[X(t)])^2 that the approach delivers. In your reasoning, report the four moment bounds you obtain and state, in one sentence, what the source finds about how the gap between its upper and lower bounds behaves as the truncation size N is increased.

Report V_plus(20): the guaranteed upper bound on the variance of the copy number of X at time 20, as a single number.

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

truncated_generator

Goal
----
Assemble the generator of the dimerization reaction network of Table I restricted to the truncated state space, namely the copy numbers 0 through N-1. Each row holds the transition rates out of one state into the other truncated states, with the diagonal carrying the total outflow that the source's construction prescribes for a truncated state, so that the matrix is exactly the operator the backward equation runs on.

```python
def truncated_generator(n_states: int, theta: "np.ndarray") -> "np.ndarray":
    """Assemble the generator of the dimerization reaction network of Table I restricted to the truncated state space, namely the copy numbers 0 through N-1. Each row holds the transition rates out of one state into the other truncated states, with the diagonal carrying the total outflow that the source's construction prescribes for a truncated state, so that the matrix is exactly the operator the backward equation runs on.

    Parameters
    ----------
    n_states : int
        Number N of truncated states; the window is {0, ..., N-1}.
    theta : np.ndarray
        The three rate constants (theta_1, theta_2, theta_3) of Table I.

    Returns
    -------
    L : np.ndarray
        Truncated generator of shape (N, N).

    Raises
    ------
    ValueError
        If n_states is not an integer >= 2, or theta is not three finite positive rates.
    """
    return L
```

### Step 2

boundary_input_column

Goal
----
Return the input column through which the boundary conditional moment enters the truncated backward equation. Exactly one truncated state exchanges probability with the boundary in this network; the column carries that state's rate into the boundary and zero elsewhere.

```python
def boundary_input_column(n_states: int, theta: "np.ndarray") -> "np.ndarray":
    """Return the input column through which the boundary conditional moment enters the truncated backward equation. Exactly one truncated state exchanges probability with the boundary in this network; the column carries that state's rate into the boundary and zero elsewhere.

    Parameters
    ----------
    n_states : int
        Number N of truncated states.
    theta : np.ndarray
        The three rate constants (theta_1, theta_2, theta_3).

    Returns
    -------
    b : np.ndarray
        Boundary input column of shape (N,).

    Raises
    ------
    ValueError
        If n_states is not an integer >= 2, or theta is not three finite positive rates.
    """
    return b
```

### Step 3

upper_input_polynomial

Goal
----
Construct the polynomial upper bound of the source on the generator applied to the monomial x^mu for the dimerization network, returned as the coefficient vector [c_0, c_1, c_2] of c_0 + c_1 x + c_2 x^2. Use the source's constructive expression, which treats the reaction classes differently; do not return the exact generator action.

```python
def upper_input_polynomial(theta: "np.ndarray", mu: int) -> "np.ndarray":
    """Construct the polynomial upper bound of the source on the generator applied to the monomial x^mu for the dimerization network, returned as the coefficient vector [c_0, c_1, c_2] of c_0 + c_1 x + c_2 x^2. Use the source's constructive expression, which treats the reaction classes differently; do not return the exact generator action.

    Parameters
    ----------
    theta : np.ndarray
        The three rate constants (theta_1, theta_2, theta_3).
    mu : int
        Monomial order, 1 or 2.

    Returns
    -------
    coeffs : np.ndarray
        Coefficient vector [c_0, c_1, c_2].

    Raises
    ------
    ValueError
        If theta is not three finite positive rates, or mu is not 1 or 2.
    """
    return coeffs
```

### Step 4

input_bound_ode_matrix

Goal
----
Assemble the constant 3 x 3 matrix of the closed linear system that governs the upper input bounds for the first and second monomials, on the augmented state [u_1, u_2, 1], from the two polynomial coefficient vectors of the previous step. Apply the source's rule for how each cross coefficient couples the two bounds, and its choice of lower input bound for this network.

```python
def input_bound_ode_matrix(coeffs_mu1: "np.ndarray", coeffs_mu2: "np.ndarray") -> "np.ndarray":
    """Assemble the constant 3 x 3 matrix of the closed linear system that governs the upper input bounds for the first and second monomials, on the augmented state [u_1, u_2, 1], from the two polynomial coefficient vectors of the previous step. Apply the source's rule for how each cross coefficient couples the two bounds, and its choice of lower input bound for this network.

    Parameters
    ----------
    coeffs_mu1 : np.ndarray
        Coefficients [c_0, c_1, c_2] of the first-order upper polynomial.
    coeffs_mu2 : np.ndarray
        Coefficients [c_0, c_1, c_2] of the second-order upper polynomial.

    Returns
    -------
    M : np.ndarray
        Augmented-state matrix of shape (3, 3).

    Raises
    ------
    ValueError
        If either vector is not three finite entries, or the first-order polynomial has an x^2 term.
    """
    return M
```

### Step 5

bounding_output_at_time

Goal
----
Integrate the two bounding state-space models for the test function x^alpha and return their outputs [y_plus(t), y_minus(t)] for an initial copy number of zero. The upper system is driven by the input-bound system through the boundary column; use the source's initial conditions for the boundary inputs and its lower input choice.

```python
def bounding_output_at_time(generator: "np.ndarray", boundary_column: "np.ndarray",
                                     u_matrix: "np.ndarray", alpha: int, t: float) -> "np.ndarray":
    """Integrate the two bounding state-space models for the test function x^alpha and return their outputs [y_plus(t), y_minus(t)] for an initial copy number of zero. The upper system is driven by the input-bound system through the boundary column; use the source's initial conditions for the boundary inputs and its lower input choice.

    Parameters
    ----------
    generator : np.ndarray
        Truncated generator of shape (N, N).
    boundary_column : np.ndarray
        Boundary input column of shape (N,).
    u_matrix : np.ndarray
        Augmented-state matrix (3, 3) of the input-bound system.
    alpha : int
        Monomial order of the test function, 1 or 2.
    t : float
        Nonnegative time at which the bounds are evaluated.

    Returns
    -------
    y : np.ndarray
        Array [y_plus, y_minus].

    Raises
    ------
    ValueError
        If the generator is not square of size >= 2, the column or matrix shapes do not match, alpha is not 1 or 2, or t is negative.
    """
    return y
```

### Step 6

variance_bracket

Goal
----
Combine the four moment bounds into the guaranteed bracket [V_plus, V_minus] of the transient variance E[X^2] - (E[X])^2, pairing the second-moment and mean bounds the way the source does so that each side of the bracket is a valid bound.

```python
def variance_bracket(y1_plus: float, y1_minus: float, y2_plus: float, y2_minus: float) -> "np.ndarray":
    """Combine the four moment bounds into the guaranteed bracket [V_plus, V_minus] of the transient variance E[X^2] - (E[X])^2, pairing the second-moment and mean bounds the way the source does so that each side of the bracket is a valid bound.

    Parameters
    ----------
    y1_plus : float
        Upper bound on E[X(t)].
    y1_minus : float
        Lower bound on E[X(t)].
    y2_plus : float
        Upper bound on E[X(t)^2].
    y2_minus : float
        Lower bound on E[X(t)^2].

    Returns
    -------
    v : np.ndarray
        Array [V_plus, V_minus].

    Raises
    ------
    ValueError
        If any bound is not finite, or an upper bound lies below its lower bound.
    """
    return v
```

### Step 7

variance_upper_bound

Goal
----
Orchestrate the whole pipeline: build the truncated generator and boundary column, construct the two upper polynomials and the input-bound system, integrate the bounding models for the first and second moments, and return the guaranteed upper bound on the variance of the copy number at time t for an initial copy number of zero. Call the earlier step functions rather than reimplementing any of them.

```python
def variance_upper_bound(theta: "np.ndarray", n_states: int, t: float) -> float:
    """Orchestrate the whole pipeline: build the truncated generator and boundary column, construct the two upper polynomials and the input-bound system, integrate the bounding models for the first and second moments, and return the guaranteed upper bound on the variance of the copy number at time t for an initial copy number of zero. Call the earlier step functions rather than reimplementing any of them.

    Parameters
    ----------
    theta : np.ndarray
        The three rate constants (theta_1, theta_2, theta_3).
    n_states : int
        Number N of truncated states.
    t : float
        Nonnegative evaluation time.

    Returns
    -------
    v_plus : float
        Upper bound on V[X(t)].

    Raises
    ------
    ValueError
        If theta is not three finite positive rates, n_states is not an integer >= 2, or t is negative.
    """
    return v_plus
```
