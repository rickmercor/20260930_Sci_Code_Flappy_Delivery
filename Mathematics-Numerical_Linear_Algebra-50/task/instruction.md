# Mathematics-Numerical_Linear_Algebra-50

## Background

Robust control seeks a feedback law that performs acceptably for every system in a family, not just one known plant. When the family is small, one linear gain and one quadratic Lyapunov certificate can work for all of it. Larger uncertainty sets often need more than that, and one family of methods replaces the single design with a finite bank of candidate gains, each carrying its own quadratic certificate, together with a supervisor that decides which candidate is active at each step. Since the true system is never observed directly, that decision has to be made from the realized trajectory alone. This turns a static design into a dynamic one: the controller carries its own internal state tracking which candidate it currently trusts, and, when the bank satisfies the source's stronger margin condition, is guaranteed to settle on a valid candidate after finitely many switches.

## Problem

Consider a discrete-time linear system x(t+1) = A x(t) + B u(t) whose true matrices are unknown but known to lie in a compact, affinely parametrized uncertain family: A(theta) = A0 + theta_1 A1 + theta_2 A2 and B(theta) = B0 + theta_1 B1 + theta_2 B2, for theta = (theta_1, theta_2) in a bounded box. Recent work on robust stabilization of such families constructs a dynamic, switching state-feedback law from a finite set of candidate (gain, certificate) pairs (K_i, P_i), each pair certifying a decay rate alpha for some subset of the uncertain systems in the sense that the certificate's contraction margin is positive. A stronger, unit-normalized version of this margin, large enough to guarantee eventual settling on a single candidate, does not hold here, so this instance is a finite-horizon simulation of the switching mechanism rather than a case confirmed to settle. At each time step the controller checks whether the currently active certificate remains consistent with the observed state trajectory; if so it keeps the same gain, and if not it advances to the next candidate in a fixed cyclic order. The exact consistency test and the level it carries forward from step to step are part of the switching law itself and are not given here. When at least one candidate is valid for the true system, this scheme applies its consistency check without ever identifying which system it is controlling; a given candidate can be falsified and revisited more than once, and whether it eventually settles within any given horizon is not asserted here.
Take the decay rate alpha = 0.9, the nominal matrices

A0 =[[1.10, 0.20, 0.00], [0.00, 0.95, 0.30], [0.10, 0.00, 1.05]]

B0 =[[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]]

the direction matrices A1 (all zero except entry (1,2) = 1), A2 (all zero except entry (3,1) = 1), B1 (all zero except entry (3,1) = 1), and B2 (all zero except entry (2,2) = 1), the parameter vector theta = (-0.20, -0.20), and the initial plant state x(0) = (0.3, 0.8, -0.6). The candidate gains and certificates are

K_1 = [[-0.981340, -0.214471,  0.313087], [ 0.171413, -0.576659, -0.644444]]
K_2 = [[-0.894080,  0.235455, -0.527303], [-0.031926, -0.500010, -0.394591]]
K_3 = [[-0.998055, -0.677131,  0.093177], [-0.346396, -1.139076, -0.104864]]
K_4 = [[-0.599742,  0.020224, -0.514584], [-0.107856, -0.745123, -0.040444]]

P_1 = [[ 4.109287,  1.923951, -3.168718], [ 1.923951,  4.508113, -2.980617], [-3.168718, -2.980617,  7.039886]]
P_2 = [[ 2.535466, -0.450140,  1.084729], [-0.450140,  2.357141, -1.468640], [ 1.084729, -1.468640,  5.922588]]
P_3 = [[10.369840,  8.842806, -6.611408], [ 8.842806, 11.281408, -6.344938], [-6.611408, -6.344938,  6.939830]]
P_4 = [[ 2.580081,  1.236551, -1.007780], [ 1.236551,  3.831790, -2.254034], [-1.007780, -2.254034,  4.587889]]

with the candidates ordered 1 through 4 and cycled in that order. The controller's internal state starts at (0, 0), so the first step uses candidate 1. Simulate the resulting switched closed loop on the realized system A(theta), B(theta) for 20 steps, and report the Euclidean norm of the plant state after the 20th step, rounded to four decimal places. 

Output Format Requirements:
Emit <final_answer> immediately, then <reasoning>. Do not write a long derivation before the tags.
You must emit exactly one finite decimal inside <final_answer>...</final_answer>, even if the value is approximate or you are unsure.
Rules:
- The tags are required. Do not omit them or leave them empty.
- The tagged value must be a finite decimal (examples: 0.4847, 0.22, 1.05). Not NaN, not Inf, not a fraction string, not a vector, not prose.
- Put only that one number between the tags. No units, no words, no extra lines.
Keep <reasoning> focused and no longer than necessary to justify the final number, a page or less. Avoid pasting the input matrices, full coefficient vectors, or per-fold candidate tables; a small number of representative per-step checks is fine if they support the argument.

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

uncertain_system_matrices

Goal
----
Realize an affinely parametrized uncertain linear system at a parameter vector.

```python
import numpy as np

def uncertain_system_matrices(A0: np.ndarray, B0: np.ndarray, A_dirs: list,
                              B_dirs: list, theta: np.ndarray) -> np.ndarray:
    """Realize an affinely parametrized uncertain system at a parameter vector.

    Parameters
    ----------
    A0 : np.ndarray
        (n, n) nominal state matrix.
    B0 : np.ndarray
        (n, m) nominal input matrix.
    A_dirs : list
        Length-L list of (n, n) float arrays, one state direction per parameter.
    B_dirs : list
        Length-L list of (n, m) float arrays, one input direction per parameter.
    theta : np.ndarray
        (L,) float array of parameter values.

    Returns
    -------
    AB : np.ndarray
        (n, n + m) array [A, B] with the realized state matrix in the first n
        columns and the realized input matrix in the remaining m columns.

    Raises
    ------
    ValueError
        If A0 is not a square 2-D array; if B0 is not 2-D with the same number
        of rows as A0; if len(A_dirs) or len(B_dirs) differs from the length of
        theta; if any entry of A_dirs does not have the shape of A0 or any entry
        of B_dirs does not have the shape of B0; if theta is not one-dimensional;
        or if any input contains non-finite entries.
    """
    return AB
```

### Step 2

lyapunov_certificate_margin

Goal
----
Evaluate the margin by which a candidate certificate pair satisfies the closed-loop contraction inequality for one realized system.

```python
import numpy as np

def lyapunov_certificate_margin(A: np.ndarray, B: np.ndarray, K: np.ndarray,
                                P: np.ndarray, alpha: float) -> float:
    """Smallest eigenvalue of the closed-loop contraction difference.

    Parameters
    ----------
    A : np.ndarray
        (n, n) realized state matrix.
    B : np.ndarray
        (n, m) realized input matrix.
    K : np.ndarray
        (m, n) static state-feedback gain.
    P : np.ndarray
        (n, n) symmetric positive definite certificate matrix.
    alpha : float
        Decay rate in (0, 1].

    Returns
    -------
    margin : float
        Smallest eigenvalue of alpha**2 * P - (A + B K)^T P (A + B K), as a
        native Python float. Positive exactly when the certificate holds for
        this realization.

    Raises
    ------
    ValueError
        If A is not a square 2-D array; if B is not 2-D with the same number of
        rows as A; if K does not have shape (m, n) with m the number of columns
        of B; if P is not (n, n) or is not symmetric within atol 1e-10; if alpha
        is not a finite real number in (0, 1]; or if any input contains
        non-finite entries.
    """
    return margin
```

### Step 3

select_active_index

Goal
----
Select the certificate index the supervisor will use at the current step from the current state and the current controller state.

```python
import numpy as np

def select_active_index(x: np.ndarray, z: np.ndarray, P_list: list) -> int:
    """Choose the certificate index for the current step.

    Parameters
    ----------
    x : np.ndarray
        (n,) current state.
    z : np.ndarray
        (2,) controller state; z[0] is the promised level and z[1] is the
        previously active index stored as a float.
    P_list : list
        Length-q list of (n, n) symmetric positive definite certificate
        matrices, ordered so that entry j - 1 belongs to index j.

    Returns
    -------
    index : int
        One-based index in 1..q of the certificate to apply at this step.

    Raises
    ------
    ValueError
        If x is not one-dimensional; if z is not a one-dimensional array of
        length 2; if P_list is empty, or any entry does not have shape (n, n)
        with n the length of x, or any entry is not symmetric within atol 1e-10;
        or if any input contains non-finite entries.
    """
    return index
```

### Step 4

controller_state_update

Goal
----
Compute the controller state carried into the next step from the current state and the index selected at the current step.

```python
import numpy as np

def controller_state_update(x: np.ndarray, index: int, P_list: list,
                            alpha: float) -> np.ndarray:
    """Build the controller state carried into the next step.

    Parameters
    ----------
    x : np.ndarray
        (n,) current state.
    index : int
        One-based index in 1..q of the certificate selected at this step.
    P_list : list
        Length-q list of (n, n) symmetric positive definite certificate
        matrices, ordered so that entry j - 1 belongs to index j.
    alpha : float
        Decay rate in (0, 1].

    Returns
    -------
    z_next : np.ndarray
        (2,) float array; entry 0 is the level the next state's quadratic form
        must not exceed, entry 1 is index stored as a float.

    Raises
    ------
    ValueError
        If x is not one-dimensional; if index is not an integer (booleans
        excluded) in 1..q with q the length of P_list; if P_list is empty, or
        any entry does not have shape (n, n) with n the length of x, or any
        entry is not symmetric within atol 1e-10; if alpha is not a finite real
        number in (0, 1]; or if any input contains non-finite entries.
    """
    return z_next
```

### Step 5

closed_loop_step

Goal
----
Advance the plant state by one step under the static gain belonging to the selected certificate index.

```python
import numpy as np

def closed_loop_step(A: np.ndarray, B: np.ndarray, x: np.ndarray, index: int,
                     K_list: list) -> np.ndarray:
    """Advance the plant one step under the selected gain.

    Parameters
    ----------
    A : np.ndarray
        (n, n) realized state matrix.
    B : np.ndarray
        (n, m) realized input matrix.
    x : np.ndarray
        (n,) current plant state.
    index : int
        One-based index in 1..q of the gain selected at this step.
    K_list : list
        Length-q list of (m, n) gains, entry j - 1 belonging to index j.

    Returns
    -------
    x_next : np.ndarray
        (n,) plant state after one step.

    Raises
    ------
    ValueError
        If A is not a square 2-D array; if B is not 2-D with the same number of
        rows as A; if x is not one-dimensional of length n; if index is not an
        integer (booleans excluded) in 1..q with q the length of K_list; if
        K_list is empty or any entry does not have shape (m, n); or if any input
        contains non-finite entries.
    """
    return x_next
```

### Step 6

switched_trajectory

Goal
----
Run the switched closed loop for a fixed number of steps and return the plant states and the index active at each step.

```python
import numpy as np

def switched_trajectory(A: np.ndarray, B: np.ndarray, x0: np.ndarray,
                        K_list: list, P_list: list, alpha: float,
                        n_steps: int) -> np.ndarray:
    """Simulate the switched closed loop from the zero controller state.

    Parameters
    ----------
    A : np.ndarray
        (n, n) realized state matrix.
    B : np.ndarray
        (n, m) realized input matrix.
    x0 : np.ndarray
        (n,) initial plant state.
    K_list : list
        Length-q list of (m, n) gains, entry j - 1 belonging to index j.
    P_list : list
        Length-q list of (n, n) symmetric certificate matrices, entry j - 1
        belonging to index j.
    alpha : float
        Decay rate in (0, 1].
    n_steps : int
        Number of steps to simulate, at least 1.

    Returns
    -------
    trace : np.ndarray
        (n_steps, n + 1) float array. Row t holds the plant state after step
        t + 1 in its first n entries, and the one-based active index at step
        t + 1, stored as a float, in the last entry.

    Raises
    ------
    ValueError
        If A is not a square 2-D array; if B is not 2-D with the same number of
        rows as A; if x0 is not one-dimensional of length n; if K_list and
        P_list have different lengths or either is empty; if n_steps is not an
        integer (booleans excluded) or is less than 1; if alpha is not a finite
        real number in (0, 1]; if none of the (K_i, P_i) pairs in K_list,
        P_list has a positive certificate margin (via lyapunov_certificate_margin)
        for the given A, B, and alpha; or if any input contains non-finite entries.
    """
    return trace
```

### Step 7

terminal_state_norm

Goal
----
Run the whole pipeline on an affinely parametrized uncertain system and report the Euclidean norm of the plant state after a fixed horizon.

```python
import numpy as np

def terminal_state_norm(A0: np.ndarray, B0: np.ndarray, A_dirs: list,
                        B_dirs: list, theta: np.ndarray, x0: np.ndarray,
                        K_list: list, P_list: list, alpha: float,
                        n_steps: int) -> float:
    """Norm of the plant state after a fixed horizon of the switched loop.

    Parameters
    ----------
    A0 : np.ndarray
        (n, n) nominal state matrix.
    B0 : np.ndarray
        (n, m) nominal input matrix.
    A_dirs : list
        Length-L list of (n, n) state directions, one per parameter.
    B_dirs : list
        Length-L list of (n, m) input directions, one per parameter.
    theta : np.ndarray
        (L,) parameter vector naming the realization.
    x0 : np.ndarray
        (n,) initial plant state.
    K_list : list
        Length-q list of (m, n) gains, entry j - 1 belonging to index j.
    P_list : list
        Length-q list of (n, n) symmetric certificate matrices, entry j - 1
        belonging to index j.
    alpha : float
        Decay rate in (0, 1].
    n_steps : int
        Number of steps to simulate, at least 1.

    Returns
    -------
    norm : float
        Euclidean norm of the plant state after n_steps steps, as a native
        Python float.

    Raises
    ------
    ValueError
        If A0 is not a square 2-D array; if B0 is not 2-D with the same number
        of rows as A0; if len(A_dirs) or len(B_dirs) differs from the length of
        theta, or any direction does not match the shape of its nominal
        counterpart; if x0 is not one-dimensional of length n; if K_list and
        P_list have different lengths or either is empty; if n_steps is not an
        integer (booleans excluded) or is less than 1; if alpha is not a finite
        real number in (0, 1]; or if any input contains non-finite entries.
    """
    return norm
```
