# Mathematics-Numerical_Linear_Algebra-34

## Background

Sublevel sets and the volume problem.
For a polynomial $g$ on $\mathbb{R}^n$, the unit sublevel set $K=\{x : g(x)\le 1\}$ is a basic semialgebraic set, and computing its Lebesgue volume is the primitive underlying probability-of-event calculations, regions of attraction and reachable sets of polynomial control systems, and the integration of polynomials against the uniform measure on $K$. The set is typically non-convex, possibly disconnected, and given only implicitly through $g$.

Intrinsic hardness.
Exact volume computation is as hard as evaluating a permanent, even for a polytope given by its facets. In the membership-oracle model, no deterministic algorithm making polynomially many calls approximates the volume within a sub-exponential factor. Randomized schemes escape this but require convexity or log-concavity and return probabilistic guarantees; rejection sampling over a bounding box drops the convexity requirement, at a variance scaling with the inverse of the volume ratio $\operatorname{vol}K/\operatorname{vol}B$, typically exponentially small in $n$.

Moment-SOS hierarchies.
The deterministic alternative writes the volume as an infinite-dimensional linear program over measures dominated by Lebesgue measure on a bounding box, then truncates to moments of bounded degree. This produces a monotone sequence of certified upper bounds, with the dual constructing a polynomial majorant of the indicator $\mathbf{1}_K$. It is deterministic, comes with a priori guarantees, and handles non-convex sets.

Known obstacles.
The indicator is discontinuous, so its polynomial majorants exhibit a Gibbs phenomenon: bounds tighten only logarithmically in the relaxation order $r$, with persistent oscillation near the boundary. Separately, the order-$r$ relaxation manipulates semidefinite blocks of size $\binom{n+r}{n}$, which grows rapidly with the ambient dimension. A third obstacle is arithmetic rather than structural: relaxations of this family involve matrices whose conditioning degrades geometrically in the relaxation order, so the working precision needed to resolve a bound grows with the order, and fixed-precision computation ceases to be reliable well before the mathematics does.

## Problem

Let $B=[-1,1]^3$ and let $K=\{x\in\mathbb{R}^3 : g(x)\le 1\}$, where

$$g(x) = 2\left(x_1^{8}+x_2^{4}+x_3^{2}-x_1^{4}x_2^{2}\right).$$

The polynomial $g$ is strictly positive away from the origin, and $K\subseteq B$. Certify an upper bound on $\operatorname{vol}K$, the Lebesgue measure of $K$, under a strict information budget: the value you report may depend on $g$ only through the numbers $\int_B g(x)^k\,dx$ for $k=0,1,\dots,14$, together with the exponent pattern of $g$ itself. Report the smallest number that this information forces to be an upper bound on $\operatorname{vol}K$ — that is, the supremum of the volume over every configuration consistent with the permitted data. The bound must be rigorous, holding without any probabilistic, asymptotic, or heuristic argument; no sampling, quadrature, or direct integration against the indicator of $K$ may enter the reported value, and the true volume is not itself a valid answer. The budget constrains the value you report, not the checks you perform on it.

 Return a single floating-point number: the certified upper bound on $\operatorname{vol}K$, rounded to 10 decimal places.

## Output format

Scientific reasoning wrapped in <reasoning>...</reasoning> tags. Include enough intermediate calculations to justify the deterministic computation, without turning the response into a general pipeline summary. A single final numeric answer wrapped in <final_answer>...</final_answer> tags.

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

01_identify_scaling_structure

Goal
----
Recover the scaling structure of a polynomial from its exponent pattern alone.

There is a vector of positive integer weights $w$ and a positive integer $m$ for which

$$g(t^{w_1}x_1, \dots, t^{w_n}x_n) = t^{m} g(x), \qquad t > 0.$$

Determine the unique such pair in which the entries share no common factor, and return the weights followed by the degree. The coefficients of $g$ play no role: only the exponent pattern matters.

Raise a ValueError if the exponent array is empty or not two-dimensional, if the solution space is not one-dimensional, or if no strictly positive integer solution exists.

```python
import numpy as np


def identify_scaling_structure(exponents: np.ndarray) -> np.ndarray:
    """Recover the scaling weights and weighted degree of a polynomial.

    Parameters
    ----------
    exponents : np.ndarray
        Integer array of shape (s, n). Row i holds the exponent vector
        alpha_i of the i-th monomial of g. Coefficients are not supplied
        and are not needed.

    Returns
    -------
    np.ndarray
        Array of shape (n + 1,) and dtype float64, holding
        [w_1, ..., w_n, m]: the positive integer weights and weighted
        degree, in the representative whose entries share no common
        factor.

    Raises
    ------
    ValueError
        If the exponent array is empty or not two-dimensional, if the
        solution space is not one-dimensional, or if no strictly positive
        integer solution exists.
    """
    return np.zeros(exponents.shape[1] + 1, dtype=np.float64)
```

### Step 2

02_compute_certified_maximum

Goal
----
Compute a certified upper estimate of the maximum of the polynomial over the box.

The construction downstream requires a value $\bar g$ with $\bar g \ge \max_B g$, where $B = [-1,1]^n$. Determine such a value from the coefficients alone, using an estimate that depends on their magnitudes and not on their exponents. No optimization, sampling, or grid search is permitted.

The estimate need not be tight. Every bound produced downstream remains valid when $\bar g$ is replaced by any larger value, so a cheap certified over-estimate suffices and is preferred to an expensive exact maximization.

```python
import numpy as np


def compute_certified_maximum(coeffs: np.ndarray, exponents: np.ndarray) -> float:
    """Compute a certified upper estimate of the maximum of g over the unit box.

    Parameters
    ----------
    coeffs : np.ndarray
        Array of shape (s,) and dtype float64 holding the coefficients of
        the s monomials of g.
    exponents : np.ndarray
        Integer array of shape (s, n). Row i holds the exponent vector of
        the i-th monomial, matching coeffs[i].

    Returns
    -------
    float
        A value gbar satisfying gbar >= max over B of g, where
        B = [-1, 1]^n. The estimate is certified but need not be tight.

    Raises
    ------
    ValueError
        If the polynomial has no monomials, if the arrays are not
        one- and two-dimensional respectively, or if they disagree in
        length.
    """
    return 0.0
```

### Step 3

03_compute_box_moments

Goal
----
Compute the moments of a polynomial over the centred unit box.

For the uniform probability measure $\mu$ on $B = [-1,1]^n$, with $d\mu = 2^{-n}\mathbf{1}_B\,dx$, compute

$$y_k = \int_B g(x)^k\,d\mu(x) = 2^{-n}\int_B g(x)^k\,dx, \qquad k = 0, 1, \dots, K.$$

These are the only quantities through which $g$ enters the remainder of the pipeline: every later stage consumes $y_0, \dots, y_K$ and never touches the polynomial again. By construction $y_0 = 1$.

The moments must be correct to full double precision at every index in the budget, including indices where the value is large. Accumulated rounding is not acceptable, and the arithmetic strategy needed to avoid it is part of the problem.

```python
import numpy as np


def compute_box_moments(coeffs: np.ndarray, exponents: np.ndarray, max_order: int) -> np.ndarray:
    """Compute moments of a polynomial over the centred unit box.

    Parameters
    ----------
    coeffs : np.ndarray
        Array of shape (s,) and dtype float64 holding the coefficients of
        the s monomials of g. Values are assumed exactly representable.
    exponents : np.ndarray
        Integer array of shape (s, n). Row i holds the exponent vector of
        the i-th monomial, matching coeffs[i].
    max_order : int
        Largest power K to compute. Must be non-negative.

    Returns
    -------
    np.ndarray
        Array of shape (max_order + 1,) and dtype float64 holding
        [y_0, y_1, ..., y_K], where y_k is the integral of g^k against the
        uniform probability measure on [-1, 1]^n. Always y_0 = 1.

    Raises
    ------
    ValueError
        If the polynomial has no monomials, if the arrays are not
        one- and two-dimensional respectively, if they disagree in length,
        or if max_order is negative.
    """
    return np.zeros(max_order + 1, dtype=np.float64)
```

### Step 4

04_compute_reference_moments

Goal
----
Compute the constants relating the moments of $g$ over its unit sublevel set to the volume of that set.

Let $K = \{x : g(x) \le 1\}$ for a polynomial $g$ possessing the scaling structure identified earlier. Under that structure there exist constants $z_0, z_1, z_2, \dots$, depending only on the weights and the weighted degree, such that

$$\int_K g(x)^k\,dx = z_k \operatorname{vol}K \qquad \text{for every integer } k \ge 0.$$

Determine $z_0, \dots, z_K$ exactly.

Note what these constants are not. They do not depend on the coefficients of $g$, on the ambient dimension as such, or on the volume itself, which remains unknown at this stage. They are fixed rationals, known in advance of any computation involving the polynomial, and they are what makes the volume recoverable from moment data at all.

```python
import numpy as np


def compute_reference_moments(weights: np.ndarray, weighted_degree: int, max_order: int) -> np.ndarray:
    """Compute the constants relating moments of g over K to the volume of K.

    Parameters
    ----------
    weights : np.ndarray
        Integer array of shape (n,) holding the positive weights
        w = (w_1, ..., w_n) of the scaling structure.
    weighted_degree : int
        The weighted degree m, a positive integer, satisfying
        <w, alpha> = m for every monomial exponent alpha of g.
    max_order : int
        Largest index K to compute. Must be non-negative.

    Returns
    -------
    np.ndarray
        Array of shape (max_order + 1,) and dtype float64 holding
        [z_0, z_1, ..., z_K], where the integral of g^k over K equals
        z_k times the volume of K. Always z_0 = 1.

    Raises
    ------
    ValueError
        If any weight is not a positive integer, if the weighted degree
        is not a positive integer, or if max_order is negative.
    """
    return np.zeros(max_order + 1, dtype=np.float64)
```

### Step 5

05_compute_upper_bound

Goal
----
Compute the tightest upper bound on the volume that the available moment data forces.

Two sequences are supplied: the moments $y_0, \dots, y_K$ of the pushforward of the uniform measure on the box through $g$, and the constants $z_0, \dots, z_K$ relating moments of $g$ over $K$ to the volume of $K$.

Consider the family of sequences $(y_k - \alpha z_k)$ indexed by a scalar $\alpha \ge 0$. For small $\alpha$ such a sequence is the moment sequence of a non-negative measure; for large $\alpha$ it is not. Determine the largest $\alpha$ for which the truncated sequence remains admissible, using only the entries available within the supplied budget, and return the corresponding upper bound on $\operatorname{vol}K$.

The returned quantity bounds the volume of $K$ itself. The result must be correct to full double precision across the whole range of budgets the function may be called with, including budgets substantially larger than any single application requires.

```python
import numpy as np


def compute_upper_bound(box_moments: np.ndarray, reference_moments: np.ndarray, dimension: int) -> float:
    """Compute the tightest upper bound on the volume forced by the moment data.

    Parameters
    ----------
    box_moments : np.ndarray
        Array of shape (K + 1,) and dtype float64 holding the pushforward
        moments [y_0, ..., y_K].
    reference_moments : np.ndarray
        Array of shape (K + 1,) and dtype float64 holding the constants
        [z_0, ..., z_K]. Must match box_moments in length.
    dimension : int
        The ambient dimension n.

    Returns
    -------
    float
        An upper bound on the volume of K = {x : g(x) <= 1}, namely the
        largest admissible scalar rescaled to bound the volume itself.

    Raises
    ------
    ValueError
        If the moment sequences are empty, differ in length, or are not
        one-dimensional, or if the subtracted part fails to be positive
        definite.
    """
    return 0.0
```

### Step 6

06_compute_lower_bound

Goal
----
Compute the tightest lower bound on the volume that the available moment data forces.

The same two sequences are supplied as before, together with a certified value $\bar g \ge \max_B g$ and the ambient dimension.

Consider again the family $(y_k - \alpha z_k)$ indexed by $\alpha \ge 0$. Previously the requirement was only that the sequence remain the moments of a non-negative measure. Now impose the stronger requirement that the measure be supported within the interval $[1, \bar g]$, and determine the extremal $\alpha$ compatible with that requirement. Return the corresponding bound on $\operatorname{vol}K$.

Return the extremal value the data determines, exactly as the construction yields it and without adjustment of any kind. This is the sole return path: a budget too small to form the weighted construction at all admits no answer and must raise rather than return a substitute value.

The result must be correct to full double precision across the whole range of budgets the function may be called with. The execution environment provides NumPy and the Python standard library. No other package is available, and an implementation that depends on one will not run.

```python
import numpy as np


def compute_lower_bound(box_moments: np.ndarray, reference_moments: np.ndarray, gbar: float, dimension: int) -> float:
    """Compute the tightest lower bound on the volume forced by the moment data.

    Parameters
    ----------
    box_moments : np.ndarray
        Array of shape (K + 1,) and dtype float64 holding the pushforward
        moments [y_0, ..., y_K].
    reference_moments : np.ndarray
        Array of shape (K + 1,) and dtype float64 holding the constants
        [z_0, ..., z_K]. Must match box_moments in length.
    gbar : float
        A certified value satisfying gbar >= max over B of g. Must exceed 1.
    dimension : int
        The ambient dimension n.

    Returns
    -------
    float
        The bound on the volume of K = {x : g(x) <= 1} determined by the
        supplied data, returned exactly as the construction yields it.

    Raises
    ------
    ValueError
        If the moment sequences are empty, differ in length, or are not
        one-dimensional; if gbar does not exceed 1; if the budget is too
        small to form the weighted construction; or if the subtracted part
        fails to be positive definite.
    """
    return 0.0
```

### Step 7

07_compute_certified_volume_bound

Goal
----
Orchestrate the full pipeline and return the certified upper bound on the volume.

Given the polynomial $g$ as coefficients and exponent vectors, together with the moment budget $K$, carry out the complete computation by calling the public functions of the preceding steps in sequence: identify_scaling_structure to recover the weights and weighted degree from the exponent pattern, compute_certified_maximum to obtain a certified estimate of the range of $g$ over the box, compute_box_moments to obtain the moments of $g$ within the budget, compute_reference_moments to obtain the constants relating moments of $g$ over $K$ to the volume, and compute_upper_bound together with compute_lower_bound to obtain the two endpoints of the certified bracket.

Verify that the endpoints are consistent, in the sense that the lower does not exceed the upper, and return the upper endpoint. This is the tightest upper bound on $\operatorname{vol}K$ that the permitted moment data forces.

This step must call the public functions defined in the preceding steps rather than reimplementing their contents.

```python
import numpy as np


def compute_certified_volume_bound(coeffs: np.ndarray, exponents: np.ndarray, max_order: int) -> float:
    """Compute the certified upper bound on the volume of the unit sublevel set.

    Parameters
    ----------
    coeffs : np.ndarray
        Array of shape (s,) and dtype float64 holding the coefficients of
        the s monomials of g.
    exponents : np.ndarray
        Integer array of shape (s, n). Row i holds the exponent vector of
        the i-th monomial, matching coeffs[i]. The polynomial must possess
        an exact scaling structure, be strictly positive away from the
        origin, and have unit sublevel set contained in [-1, 1]^n.
    max_order : int
        The moment budget K. The bound may depend on g only through the
        integrals of g^k over the box for k = 0, ..., K.

    Returns
    -------
    float
        The tightest upper bound on the volume of K = {x : g(x) <= 1}
        that the permitted moment data forces.

    Raises
    ------
    ValueError
        If the exponent array is not two-dimensional, if any upstream
        stage rejects its input, or if the computed bracket is
        inconsistent.
    """
    return 0.0
```
