# Biology-Ecology-41

## Background

Complex network dynamics can produce heterogeneous equilibrium responses across nodes, so measuring every component can be impractical. A useful reduction estimates a global network observable from a small subset of nodes while preserving behavior across a control-parameter sweep. The scientific setting here combines two nonlinear network dynamical systems on the same fixed topology: a bistable local response with network coupling and an epidemic model with recovery and network-mediated infection. The key challenge is that a node subset selected under one dynamical law is evaluated under a different law, so the calculation separates topology-dependent node selection from dynamics-dependent equilibrium activity. Approximation quality is assessed over an entire parameter sweep rather than at a single operating point, making heterogeneous node responses and transitions part of the observable being reduced.

## Problem

Complex networks can exhibit heterogeneous node responses even when all nodes obey the same nonlinear local law, making the full equilibrium state difficult to observe from a small number of measurements. This benchmark applies the paper's sentinel-node reduction method in its unweighted form to a deterministic 18-node undirected network and evaluates transfer of the selected sentinel set from a coupled double-well system to a susceptible-infectious-susceptible (SIS) system. Use the 32 undirected edges below with 0-based labels to construct the symmetric zero-diagonal adjacency matrix: (0,1), (0,2), (0,3), (0,4), (0,5), (0,6), (0,7), (0,17), (1,3), (1,5), (1,7), (1,8), (1,9), (1,15), (2,10), (2,12), (2,14), (3,4), (3,6), (3,8), (3,9), (3,11), (3,13), (3,14), (5,11), (5,12), (5,13), (5,15), (6,17), (8,10), (10,16), and (15,16). For the training system use the paper's coupled double-well dynamics over 31 evenly spaced control values from D=0 to D=1, with all node states initialized to 1; for the test system use the paper's SIS dynamics over 31 evenly spaced control values from lambda=0 to lambda=1, with all node states initialized to 0.01. In both systems use terminal time T=15 as the equilibrium proxy, and require numerical integration to be converged so that tightening the numerical tolerances changes every returned terminal state by less than 1e-8 in max norm. Use the paper's prescribed sentinel-set cardinality and combinatorial optimization method in the unweighted setting. For this benchmark, the stochastic search is made reproducible with NumPy default_rng and seed 123, and the reported sentinel set is the lowest-approximation-error set visited at any point in the search, not merely the final state of the walk. If multiple visited sets have exactly the same computed error, use the lexicographically smallest sorted node-label tuple as the deterministic tie-break. Evaluate that selected training set unchanged on the SIS sweep rather than selecting a new set for the test dynamics. For the approximation-error normalization, this benchmark follows the normalization used by the authors' released implementation: the squared discrepancies are divided by the sum of the full-network means over the sweep. The printed equation in the article contains an additional factor of L in that denominator; therefore the benchmark's convention is smaller than the printed-equation value by a factor of L. State which convention you used when reporting intermediate calculations. Report numerical intermediate quantities to at least 8 significant digits where practical. Define the benchmark's transfer-penalty diagnostic R as the SIS sweep approximation error for that unchanged set divided by its double-well training sweep approximation error.

## Output format

```
## Output format

Scientific reasoning wrapped in <reasoning>...</reasoning> tags. Include enough intermediate calculations to justify the deterministic computation, without turning the response into a general pipeline summary. A single final numeric answer wrapped in <final_answer>...</final_answer> tags.
```

## Your task

Implement **all 9 functions** below into a single file at `/app/solution.py`. Requirements:

- Use the EXACT function name and signature shown for each step.
- Define every function at module scope (importable).
- Do NOT include tests, example usage, prints, or a `__main__` block.
- Available libraries: numpy, scipy, sympy, pandas, scikit-learn, networkx.

Start your file with these imports:

```python
import numpy as np
```

### Step 1

network_summary

Goal
----
Compute basic structural invariants of a network adjacency matrix that are used to parameterize the sentinel calculation.

```python
def network_summary(A: "np.ndarray") -> "np.ndarray":
    '''Return structural invariants of a simple undirected network.

    Parameters
    ----------
    A : np.ndarray
        Square adjacency matrix for a simple undirected graph. Entries must be
        0 or 1, the diagonal must be zero, and the matrix must be symmetric.

    Returns
    -------
    summary : np.ndarray
        Array [N, M] as floating-point values, where N is the number of nodes
        and M is the number of undirected edges.

    Raises
    ------
    ValueError
        If A is not a finite square binary symmetric adjacency matrix with a
        zero diagonal or if the graph is disconnected.
    '''
    return summary
```

### Step 2

double_well_states

Goal
----
Compute the equilibrium-proxy node states of the coupled double-well network model across a control-parameter sweep.

```python
def double_well_states(A: "np.ndarray", D_grid: "np.ndarray", T: float) -> "np.ndarray":
    '''Return node states at the terminal integration time for the coupled double-well system.

    For each control value D, integrate the coupled double-well equations

    dx_i/dt = -(x_i - 1)(x_i - 3)(x_i - 5)
              + D Σ_j A_ij x_j,

    with x_i(0) = 1 for every node i. Return the node-state vector at
    terminal time T for every value in D_grid.

    Parameters
    ----------
    A : np.ndarray
        Square symmetric binary adjacency matrix of a connected simple graph.
    D_grid : np.ndarray
        Strictly increasing one-dimensional control-parameter grid in [0, 1].
    T : float
        Positive terminal time; the benchmark uses T = 15.

    Returns
    -------
    states : np.ndarray
        Array of shape (len(D_grid), N), with one terminal node-state vector
        per control value.

    Raises
    ------
    ValueError
        If the graph, grid, or terminal time violates the stated domain.
    '''
    return states
```

### Step 3

network_average

Goal
----
Compute the global equilibrium activity curve from a matrix of node states.

```python
def network_average(states: "np.ndarray") -> "np.ndarray":
    '''Return the unweighted network mean for each control-parameter value.

    Parameters
    ----------
    states : np.ndarray
        Finite two-dimensional array with shape (L, N), where rows correspond
        to control values and columns correspond to network nodes.

    Returns
    -------
    mean_activity : np.ndarray
        One mean node activity for each of the L rows.

    Raises
    ------
    ValueError
        If states is not a finite two-dimensional array with at least one row
        and one node.
    '''
    return mean_activity
```

### Step 4

sentinel_error

Goal
----
Evaluate the approximation error of a fixed sentinel node set over a complete state sweep.

```python
def sentinel_error(states: "np.ndarray", mean_activity: "np.ndarray", sentinel_indices: "np.ndarray") -> float:
    '''Return the dimensionless approximation error for the supplied sentinel set over the complete state sweep.

The benchmark follows the normalization used by the authors' released implementation: divide the summed squared discrepancies by the sum of full-network means. The article's printed equation includes an additional factor of L in the denominator; responses using that printed-equation form should be treated as the same convention up to the constant factor L.

    Parameters
    ----------
    states : np.ndarray
        Finite array of shape (L, N) containing node states over the sweep.
    mean_activity : np.ndarray
        Finite length-L full-network mean corresponding to the rows of states.
    sentinel_indices : np.ndarray
        One-dimensional integer node labels with no duplicates and all labels
        in [0, N-1].

    Returns
    -------
    error : float
        Dimensionless discrepancy measure between the sentinel estimate and the
        full-network activity curve.

    Raises
    ------
    ValueError
        If shapes, indices, finiteness, or the positive denominator requirement
        is violated.
    '''
    return error
```

### Step 5

select_sentinels

Goal
----
Select a small sentinel node set that minimizes approximation error over a training sweep.

```python
def select_sentinels(states: "np.ndarray", mean_activity: "np.ndarray", n: int, seed: int) -> "np.ndarray":
    '''Return a deterministic sentinel node set for a training sweep.

    Parameters
    ----------
    states : np.ndarray
        Finite array of shape (L, N) containing training node states.
    mean_activity : np.ndarray
        Length-L full-network activity curve corresponding to states.
    n : int
        Number of distinct sentinel nodes; 1 <= n < N.
    seed : int
        Non-negative random seed used to make the stochastic search reproducible.

    Returns
    -------
    sentinel_indices : np.ndarray
        Sorted integer node labels of length n corresponding to the lowest-error
        set visited during the search. Exact-error ties use lexicographic order.

    Raises
    ------
    ValueError
        If the state dimensions, subset size, seed, or positivity of the
        training normalization is invalid.
    '''
    return sentinel_indices
```

### Step 6

sis_states

Goal
----
Compute the equilibrium-proxy node states of the deterministic SIS network model across an infection-rate sweep.

```python
def sis_states(A: "np.ndarray", lambda_grid: "np.ndarray", T: float) -> "np.ndarray":
    '''Return terminal node infection probabilities for the deterministic SIS model.

    For each infection rate λ, integrate the deterministic SIS equations

    dx_i/dt = -x_i + λ (1 - x_i) Σ_j A_ij x_j,

    with x_i(0) = 0.01 for every node i. Return the node-state vector at
    terminal time T for every value in lambda_grid.

    Parameters
    ----------
    A : np.ndarray
        Square symmetric binary adjacency matrix of a connected simple graph.
    lambda_grid : np.ndarray
        Strictly increasing one-dimensional infection-rate grid in [0, 1].
    T : float
        Positive terminal time; the benchmark uses T = 15.

    Returns
    -------
    states : np.ndarray
        Array of shape (len(lambda_grid), N), one terminal node-state vector
        per infection-rate value.

    Raises
    ------
    ValueError
        If the graph, grid, or terminal time violates the stated domain.
    '''
    return states
```

### Step 7

transfer_error

Goal
----
Evaluate the previously selected sentinel set on a second network dynamics model.

```python
def transfer_error(test_states: "np.ndarray", sentinel_indices: "np.ndarray") -> float:
    '''Return the approximation error of a fixed sentinel set on test dynamics.

    Parameters
    ----------
    test_states : np.ndarray
        Finite array of shape (L, N) containing terminal node states for the
        test dynamics over its control-parameter sweep.
    sentinel_indices : np.ndarray
        One-dimensional integer node labels shared with the training network.

    Returns
    -------
    error : float
        Dimensionless approximation error between the sentinel estimate and the full
        network activity curve for the test dynamics.

    Raises
    ------
    ValueError
        If the state matrix or sentinel labels are invalid or the full-network
        normalization is non-positive.
    '''
    return error
```

### Step 8

transfer_penalty

Goal
----
Combine the training and test approximation errors into a dimensionless transfer-penalty ratio.

```python
def transfer_penalty(training_error: float, test_error: float) -> float:
    '''Return the ratio of test-dynamics approximation error to training error.

    Parameters
    ----------
    training_error : float
        Positive approximation error measured on the dynamics used for sentinel
        selection.
    test_error : float
        Non-negative approximation error measured on the second dynamics.

    Returns
    -------
    ratio : float
        Dimensionless ratio test_error / training_error.

    Raises
    ------
    ValueError
        If either input is non-finite, training_error is not positive, or
        test_error is negative.
    '''
    return ratio
```

### Step 9

sentinel_transfer_pipeline

Goal
----
Compute the complete sentinel-node transfer diagnostic for a network and two nonlinear dynamics sweeps.

```python
def sentinel_transfer_pipeline(A: "np.ndarray", D_grid: "np.ndarray", lambda_grid: "np.ndarray", seed: int = 123) -> float:
    '''Return the deterministic transfer-penalty ratio for the supplied network and sweeps.

    Parameters
    ----------
    A : np.ndarray
        Square symmetric binary adjacency matrix of a connected simple graph.
    D_grid : np.ndarray
        Strictly increasing double-well control grid in [0,1].
    lambda_grid : np.ndarray
        Strictly increasing SIS infection-rate grid in [0,1].
    seed : int, optional
        Non-negative random seed for the sentinel selection; the benchmark
        canonical value is 123.

    Returns
    -------
    ratio : float
        Test-dynamics approximation error divided by training-dynamics
        approximation error for the selected sentinel set.

    Raises
    ------
    ValueError
        If the graph, grids, or seed are outside the supported domain or if
        the training normalization produces a zero or negative denominator.
    '''
    return ratio
```
