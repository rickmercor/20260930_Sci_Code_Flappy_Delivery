# Mathematics-Computational_Finance-30

## Problem

A market maker quotes European call mids on a single normalised underlying across three
expiries and one common strike ladder. The quotes are free of arbitrage at the ladder
strikes, but the source's smoothed model class cannot reproduce them exactly. Your task is
to fit them with the smooth, strictly arbitrage-free non-parametric surface construction
given in the browsing source, and then report a diagnostic read off the fitted surface.

Configuration, all of which is given. The strike ladder is
K = [0.70, 0.80, 0.90, 0.95, 1.00, 1.05, 1.10, 1.20, 1.30] and the expiry nodes are
T = [0.20, 0.50, 1.00] with node dispersions V = [0.040, 0.075, 0.140], each stated in the
parameterisation the source fixes in its Section 2 preliminaries. For scenario index s, set
shift = 0.002 * ((s mod 5) - 2) and build the quoted dispersions as
Vq[j] = V[j] + 0.030 * (1 - K), then subtract (0.022 + shift) * exp(-((K - 1.00)/0.16)^2)
from row j = 1 and (0.030 + shift) * exp(-((K - 1.05)/0.14)^2) from row j = 2, with rows
indexed from zero so that these are the expiries T = 0.50 and T = 1.00. The quoted mid at
expiry j and ladder strike K is the source's own pricing function evaluated at unit spot,
that strike and Vq[j]. All quote weights are 1.

Fit one marginal per expiry to those mids using the fitting program the source states for
its homogeneous-strike case, at the source's recommended default setting of its smoothness
parameter, imposing every constraint that program lists. Then, at each of
the four probe points (T, K) = (0.75, 1.075), (0.40, 0.925), (0.90, 1.150) and
(0.30, 0.850), evaluate the model value on the fitted surface and the local variance
implied by it.

Report, for scenario s = 0, the sum of all eight of those numbers: the four model values
plus the four local variances.

Four choices the source does not pin down, so adopt these: for the between-node blend, use
the linear weight the source gives as its worked example, not the monotone spline it says
it prefers in practice; take the expiry sensitivity in the local-variance ratio as a
one-sided difference running forward from the probe expiry to the next expiry node; take
the strike curvature by a centred difference with a step equal to 2 percent of the probe
strike; and return 0.0 for a probe whose expiry step is empty or whose strike curvature is
not strictly positive.

In your reasoning, state the following six diagnostics: the shape of the marginal-to-price map, how many non-zero entries each fitted marginal carries, the total weighted absolute deviation from the quoted mids at the optimum, the sum of the four model values on their own, and the local variance at each of the first two probe points.

State the conventions you adopted and, for each one the source fixes, justify it from the
source.

Output Format Requirements:
Emit <reasoning> first, followed by <final_answer>. Do not write a long derivation before the tags.
You must emit exactly one finite decimal inside <final_answer>...</final_answer>, even if the value is approximate or you are unsure.
Rules:
- The tags are required. Do not omit them or leave them empty.
- The tagged value must be a finite decimal (examples: 0.4847, 0.22, 1.05). Not NaN, not Inf, not a fraction string, not a vector, not prose.
- Put only that one number between the tags. No units, no words, no extra lines.
-Use <reasoning> to explain the adopted conventions and report the requested diagnostics. 
-Do not paste the input matrices, full coefficient vectors, per-iteration paths, or per-fold candidate tables.

## Output format

```
Output Format Requirements:
Emit <reasoning> first, followed by <final_answer>. Do not write a long derivation before the tags.
You must emit exactly one finite decimal inside <final_answer>...</final_answer>, even if the value is approximate or you are unsure.
Rules:
- The tags are required. Do not omit them or leave them empty.
- The tagged value must be a finite decimal (examples: 0.4847, 0.22, 1.05). Not NaN, not Inf, not a fraction string, not a vector, not prose.
- Put only that one number between the tags. No units, no words, no extra lines.
-Use <reasoning> to explain the adopted conventions and report the requested diagnostics. 
-Do not paste the input matrices, full coefficient vectors, per-iteration paths, or per-fold candidate tables.
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

bs_call_price

Goal
----
Evaluate the European call pricing function that the source states in its Section 2 preliminaries, in exactly the parameterisation given there. The third argument is the dispersion parameter as that section defines it; do not re-scale it. Inputs broadcast against one another. The zero-dispersion and zero-strike limits must return the correct limiting value rather than a division error.

```python
def bs_call_price(spot: "float | np.ndarray", strike: "float | np.ndarray", total_var: "float | np.ndarray") -> "np.ndarray":
    """Evaluate the source's European call pricing function in exactly its own parameterisation.

    Parameters
    ----------
    spot : float or numpy.ndarray
        Spot level(s) of the driver; strictly positive.
    strike : float or numpy.ndarray
        Strike(s); non-negative.
    total_var : float or numpy.ndarray
        Dispersion parameter(s) in the source's Section 2 parameterisation; non-negative.

    Returns
    -------
    prices : numpy.ndarray
        float64 array of call values broadcast to the common shape of the three inputs.

    Raises
    ------
    ValueError
        If any dispersion is negative, any spot is not strictly positive, any strike is negative, or any input is not finite.
    """
    return prices
```

### Step 2

model_price_matrix

Goal
----
Build the matrix that sends a marginal density on the strike ladder to model prices at the ladder's quoted strikes for a single expiry, as the source assembles it around its Eq. (31). Each entry is the step 1 pricing function evaluated with the row's strike, the column's strike, and the expiry dispersion after the source's scaling by the third argument. The row set is the one Eq. (31) specifies, which is not every strike on the ladder.

```python
def model_price_matrix(strikes: "np.ndarray", atm_var: float, eta: float) -> "np.ndarray":
    """Build the single-expiry map from a marginal density on the strike ladder to model prices at the quoted strikes.

    Parameters
    ----------
    strikes : numpy.ndarray
        One-dimensional strike ladder, strictly increasing and strictly positive, at least three points.
    atm_var : float
        Node dispersion of the expiry, in the source's Section 2 parameterisation; non-negative.
    eta : float
        The source's scaling factor applied to the node dispersion (Section 3.1); non-negative.

    Returns
    -------
    price_map : numpy.ndarray
        float64 matrix with one column per ladder point and the row set Eq. (31) prescribes.

    Raises
    ------
    ValueError
        If the ladder is not one-dimensional, strictly increasing and strictly positive, if it has fewer than three points, or if atm_var or eta is negative or not finite.
    """
    return price_map
```

### Step 3

payoff_matrix

Goal
----
Build the square map the source uses in Section 4.1 to read a marginal density's own discrete call values off the strike ladder. Row index is the strike being valued, column index the ladder point carrying mass. Section 4.1 states which valuation this map applies, and it is not the one step 2 uses.

```python
def payoff_matrix(strikes: "np.ndarray") -> "np.ndarray":
    """Build the square map from a marginal density to that density's own discrete call values on the ladder.

    Parameters
    ----------
    strikes : numpy.ndarray
        One-dimensional strike ladder, strictly increasing and strictly positive.

    Returns
    -------
    ordering_map : numpy.ndarray
        float64 square matrix of side equal to the ladder length.

    Raises
    ------
    ValueError
        If the ladder is not one-dimensional, not strictly increasing, not strictly positive, or contains a non-finite value.
    """
    return ordering_map
```

### Step 4

marginal_constraints

Goal
----
Return the equality constraint block, as coefficient matrix and right-hand side, that the source requires of every per-expiry marginal in its program (SMP) in Section 4.1. Rows in the order the source lists them. This block is the same for every expiry, so build it once for the ladder.

```python
def marginal_constraints(strikes: "np.ndarray") -> "tuple[np.ndarray, np.ndarray]":
    """Return the equality block every per-expiry marginal must satisfy in the source's program.

    Parameters
    ----------
    strikes : numpy.ndarray
        One-dimensional strike ladder, strictly increasing and strictly positive.

    Returns
    -------
    A, b : tuple of numpy.ndarray
        A is the float64 coefficient matrix with one column per ladder point and one row per equality, in the source's order; b is the matching float64 right-hand-side vector.

    Raises
    ------
    ValueError
        If the ladder is not one-dimensional, not strictly increasing, not strictly positive, or contains a non-finite value.
    """
    return A, b
```

### Step 5

solve_marginals

Goal
----
Assemble and solve the fitting program (SMP) that Section 4.1 of the source writes down, returning one marginal per expiry. The three maps are handed in: the per-expiry price maps from step 2 stacked along the first axis, the ordering map from step 3, and the per-expiry equality block from step 4 as a pair. The source states which discrepancy measure the objective minimises and how the quote weights enter it; that choice is what keeps the program in the class Section 4.1 names. Impose the cross-expiry ordering the source lists under 'Constraints', stated there in terms of step 3's output rather than of the marginals themselves. Non-negativity applies to every marginal. If the program does not solve, return an all-zero array of the correct shape rather than raising.

```python
def solve_marginals(price_matrices: "np.ndarray", ordering_map: "np.ndarray", constraint_block: "tuple[np.ndarray, np.ndarray]", mid_quotes: "np.ndarray", weights: "np.ndarray") -> "np.ndarray":
    """Fit one marginal per expiry to the quoted mids under the source's program.

    Parameters
    ----------
    price_matrices : numpy.ndarray
        Array of shape (M, R, N): the step 2 map of each of the M expiries, stacked along the first axis.
    ordering_map : numpy.ndarray
        The step 3 square map of shape (N, N).
    constraint_block : tuple of numpy.ndarray
        The step 4 pair (A, b): A of shape (rows, N) and b of shape (rows,), applied to every expiry.
    mid_quotes : numpy.ndarray
        Quoted mids of shape (M, N), one row per expiry over the full ladder; only the rows the price map prices are fitted.
    weights : numpy.ndarray
        Non-negative quote weights of shape (M, N), aligned with mid_quotes.

    Returns
    -------
    marginals : numpy.ndarray
        float64 array of shape (M, N): the fitted marginal of each expiry on the ladder, or all zeros if the program does not solve.

    Raises
    ------
    ValueError
        If price_matrices is not three-dimensional, if R does not equal the row count Eq. (31) prescribes for N, if ordering_map is not (N, N), if mid_quotes or weights is not (M, N), or if the constraint block does not have N columns.
    """
    return marginals
```

### Step 6

smooth_surface_price

Goal
----
Value a single option at an arbitrary expiry and strike from the fitted marginals, following the source's Eq. (2). Between two expiry nodes the value is the source's weight-function blend of the two bracketing marginals; the source states in Eq. (2) which dispersion each of the two terms is priced at, and the answer is not 'each at its own node'. Use the linear weight function the source gives as its example directly below Eq. (2), the convention the task statement declares. Outside the node range, fall back on the nearest node's marginal priced at that node's own dispersion. Price through step 1.

```python
def smooth_surface_price(strikes: "np.ndarray", expiries: "np.ndarray", atm_vars: "np.ndarray", q: "np.ndarray", t_query: float, k_query: float, eta: float) -> float:
    """Value one option at an arbitrary expiry and strike off the fitted marginals, following the source's Eq. (2).

    Parameters
    ----------
    strikes : numpy.ndarray
        One-dimensional strike ladder, strictly increasing and strictly positive.
    expiries : numpy.ndarray
        One-dimensional, strictly increasing expiry nodes, at least one.
    atm_vars : numpy.ndarray
        Node dispersion of each expiry, same shape as expiries.
    q : numpy.ndarray
        Fitted marginals of shape (len(expiries), len(strikes)).
    t_query : float
        Query expiry; strictly positive and finite.
    k_query : float
        Query strike; strictly positive and finite.
    eta : float
        The source's scaling factor applied to every dispersion (Section 3.1).

    Returns
    -------
    value : float
        The model value at the queried expiry and strike, as a native Python float.

    Raises
    ------
    ValueError
        If the ladder is invalid as in step 2, if expiries is empty or not strictly increasing, if atm_vars or q has the wrong shape, or if t_query or k_query is not strictly positive and finite.
    """
    return value
```

### Step 7

discrete_local_variance

Goal
----
Read the local variance off the fitted smooth surface at the queried point, using the ratio the source recalls in its Section 3 discussion of continuous local volatility. Take the strike curvature by a centred difference with the step given as a fraction of the queried strike. Take the expiry sensitivity as a one-sided difference running forward from the queried expiry to the next expiry node, the convention the task statement declares; the backward alternative gives a materially different number. Value every point through step 6. A query lying exactly on an expiry node has an empty expiry step. Return 0.0 when the expiry step is empty or the strike curvature is not strictly positive, rather than raising or returning a non-finite value.

```python
def discrete_local_variance(strikes: "np.ndarray", expiries: "np.ndarray", atm_vars: "np.ndarray", q: "np.ndarray", t_query: float, k_query: float, eta: float, h_rel: float = 0.02) -> float:
    """Read the local variance off the fitted smooth surface at the queried point by finite differences.

    Parameters
    ----------
    strikes : numpy.ndarray
        One-dimensional strike ladder, strictly increasing and strictly positive.
    expiries : numpy.ndarray
        One-dimensional, strictly increasing expiry nodes.
    atm_vars : numpy.ndarray
        Node dispersion of each expiry, same shape as expiries.
    q : numpy.ndarray
        Fitted marginals of shape (len(expiries), len(strikes)).
    t_query : float
        Query expiry; strictly positive and finite.
    k_query : float
        Query strike; strictly positive and finite.
    eta : float
        The source's scaling factor applied to every dispersion (Section 3.1).
    h_rel : float, optional
        Strike step of the centred difference as a fraction of k_query; strictly positive. Default 0.02.

    Returns
    -------
    local_var : float
        The local variance at the queried point as a native Python float, or 0.0 when the expiry step is empty (the query lies on or beyond the last expiry node, or exactly on any node) or the strike curvature is not strictly positive.

    Raises
    ------
    ValueError
        If h_rel is not strictly positive and finite, or if step 6 rejects the surface inputs or the query point.
    """
    return local_var
```

### Step 8

sanos_surface_audit

Goal
----
Run the whole pipeline for the given scenario index and report the audit table. Build the scenario's quoted mids exactly as the problem statement specifies, build the three maps with steps 2, 3 and 4, hand them to step 5 to fit the marginals, then for each of the four probe points listed in the problem statement report the model value from step 6 and the local variance from step 7, in that column order and in the probe order given. Report only computed quantities; the probe coordinates are inputs. This step must call the earlier step functions; do not re-implement any of them here.

```python
def sanos_surface_audit(scenario: int) -> "np.ndarray":
    """Run the whole pipeline for one scenario index and report the audit table at the four probe points.

    Parameters
    ----------
    scenario : int
        Non-negative scenario index; only scenario mod 5 affects the quoted dispersions, as the problem statement specifies.

    Returns
    -------
    audit : numpy.ndarray
        float64 array of shape (4, 2): row r holds the model value and the local variance at probe r, in the problem statement's probe order.

    Raises
    ------
    ValueError
        If scenario is negative.
    """
    return audit
```
