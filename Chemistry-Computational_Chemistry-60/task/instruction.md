# Chemistry-Computational_Chemistry-60

## Background

State-specific excited-state methods describe an electronic transition by the one-particle densities of its initial and final states. Analyses built on the difference between these two densities yield scalar descriptors that are compared across molecules and states to interpret the character of an excitation.

## Problem

The source paper analyzes an electronic transition from a pair of one-particle densities. This task is a deterministic numerical instance of the source protocol. Recover the protocol from the source paper and apply it to the AO-basis spin-block fixture printed below. Report the source protocol's scalar relaxation measure for this instance.

The evaluation instance is the fixture printed in this prompt; no new quantum-chemistry calculation is required.

The tagged value is this fixture's relaxation measure, not a whole-transition promotion or excitation count, a published test-set value or an energy.

Density-matrix fixture

The arrays below are one spin block of the initial and final AO densities and the AO overlap for an electron-conserving transition in a six-function basis. Every number below is supplied.

ε = 0.001

ε is the printed numerical tolerance for the occupation-change partition of this fixture. Use it wherever the source protocol needs a cutoff. Do not replace it by a figure cutoff from a plot.

AO overlap S:
[[1.00, 0.16, 0.05, 0.02, 0.01, 0.00],
 [0.16, 1.00, 0.11, 0.04, 0.03, 0.01],
 [0.05, 0.11, 1.00, 0.08, 0.05, 0.02],
 [0.02, 0.04, 0.08, 1.00, 0.14, 0.06],
 [0.01, 0.03, 0.05, 0.14, 1.00, 0.12],
 [0.00, 0.01, 0.02, 0.06, 0.12, 1.00]]

initial AO density P_i:
[[ 1.0274904082404372e+00, -1.6041968753456470e-01, -3.3042774771867789e-02, -4.7947505320175710e-03, -8.7335324545658290e-04,  1.6584156739246577e-03],
 [-1.6041968753456470e-01,  1.0383509345594149e+00, -1.0376856980275348e-01, -1.2739093513340627e-02, -1.0048844091019931e-02, -2.5421514002020901e-03],
 [-3.3042774771867789e-02, -1.0376856980275348e-01,  1.0189368328929171e+00, -3.5810348001059589e-02, -1.9236801914385145e-02, -6.3261350632161208e-03],
 [-4.7947505320175710e-03, -1.2739093513340627e-02, -3.5810348001059589e-02,  1.5931662799502915e-03,  8.9672832183644569e-04,  2.6503425728921108e-04],
 [-8.7335324545658290e-04, -1.0048844091019931e-02, -1.9236801914385145e-02,  8.9672832183644569e-04,  5.1509887859554915e-04,  1.5343693946516464e-04],
 [ 1.6584156739246577e-03, -2.5421514002020901e-03, -6.3261350632161208e-03,  2.6503425728921108e-04,  1.5343693946516464e-04,  5.0048601933326756e-05]]

final AO density P_f:
[[ 9.2227974900520182e-01, -7.6274042836591546e-02, -4.5930622110822598e-02,  2.9087928318826994e-01, -2.3384337216408001e-02, -6.4675836362762746e-03],
 [-7.6274042836591546e-02,  9.3110893295460517e-03, -4.9911860714836651e-02, -2.1599612430891132e-02, -8.2345528837583469e-03, -1.1615694782760022e-03],
 [-4.5930622110822598e-02, -4.9911860714836651e-02,  1.0139181393775361e+00, -4.5133552049371209e-02, -4.2019487645012850e-02,  4.5502858412161104e-02],
 [ 2.9087928318826994e-01, -2.1599612430891132e-02, -4.5133552049371209e-02,  9.7211352001987925e-02, -7.4306257192611150e-02,  4.3745954547255283e-04],
 [-2.3384337216408001e-02, -8.2345528837583469e-03, -4.2019487645012850e-02, -7.4306257192611150e-02,  1.0276668340285153e+00, -5.9547692454792238e-02],
 [-6.4675836362762746e-03, -1.1615694782760022e-03,  4.5502858412161104e-02,  4.3745954547255283e-04, -5.9547692454792238e-02,  5.3198963206647822e-03]]

The AO difference density is ΔP = P_f − P_i. Tr(S P_i) = Tr(S P_f) = 3. Every reconstruction of the difference density uses AO coefficients V that satisfy ΔP = V δ V^T. Report AO quantities in that convention.

In the reasoning, include the occupation-change magnitudes and the two unoccupied-block populations that determine the tagged number.

Return this relaxation measure as a single numerical scalar.

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

density_pair

Goal
----
Form the AO-basis difference density.

```python
def density_pair(
    P_i: np.ndarray,
    P_f: np.ndarray,
) -> np.ndarray:
    """
    Return the AO-basis difference density for the analysis.

    Parameters
    ----------
    P_i : np.ndarray
        Initial-state AO density, shape (n, n).
    P_f : np.ndarray
        Final-state AO density, shape (n, n).

    Returns
    -------
    np.ndarray
        Symmetric difference density, shape (n, n).

    Raises
    ------
    ValueError
        If the densities are not square, not aligned, or not finite.
    """
    return dP
```

### Step 2

ordered_pair

Goal
----
Build the ordered overlap-factor pair from the AO overlap S.

```python
def ordered_pair(
    S: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return the ordered overlap-factor pair (G, H).

    G @ G reconstructs S. G @ H is the identity.

    Parameters
    ----------
    S : np.ndarray
        AO overlap, shape (n, n).

    Returns
    -------
    G : np.ndarray
        First overlap factor, shape (n, n).
    H : np.ndarray
        Second overlap factor, shape (n, n).

    Raises
    ------
    ValueError
        If S is not square, not symmetric, not positive definite, or
        not finite.
    """
    return G, H
```

### Step 3

coefficient_pair

Goal
----
Construct the occupation-change spectrum.

```python
def coefficient_pair(
    dP: np.ndarray,
    G: np.ndarray,
    H: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return occupation-change numbers and AO coefficients.

    Form the G-image of dP. Its eigenvalues are delta. Map those
    eigenvectors with H to obtain V.

    Parameters
    ----------
    dP : np.ndarray
        AO difference density, shape (n, n).
    G : np.ndarray
        First overlap factor from step 2, shape (n, n).
    H : np.ndarray
        Second overlap factor from step 2, shape (n, n).

    Returns
    -------
    delta : np.ndarray
        Occupation-change numbers, shape (n,).
    V : np.ndarray
        AO coefficients, shape (n, n).

    Raises
    ------
    ValueError
        If dP and the overlap factors are misaligned or if numerical
        values are invalid.
    """
    return delta, V
```

### Step 4

image_matrix

Goal
----
Map the initial-state AO density with the supplied overlap factors.

```python
def image_matrix(
    P0: np.ndarray,
    G: np.ndarray,
    H: np.ndarray,
) -> np.ndarray:
    """
    Return the mapped initial-state AO density.

    Map P0 with G, not H.

    Parameters
    ----------
    P0 : np.ndarray
        Initial-state AO density, shape (n, n).
    G : np.ndarray
        First overlap factor from step 2, shape (n, n).
    H : np.ndarray
        Second overlap factor from step 2, shape (n, n).

    Returns
    -------
    np.ndarray
        Symmetric mapped density, shape (n, n).

    Raises
    ------
    ValueError
        If the matrices are misaligned, not square, or not finite.
    """
    return P_t
```

### Step 5

frame_matrix

Goal
----
Form the unoccupied projector of the mapped initial-state density.

```python
def frame_matrix(
    P_t: np.ndarray,
) -> np.ndarray:
    """
    Return the unoccupied projector of the mapped initial density.

    Q is I minus P_t.

    Parameters
    ----------
    P_t : np.ndarray
        Mapped initial-state density from step 4, shape (n, n).

    Returns
    -------
    np.ndarray
        Unoccupied projector, shape (n, n).

    Raises
    ------
    ValueError
        If P_t is not a nonempty finite square matrix.
    """
    return Q
```

### Step 6

group_values

Goal
----
Return the two unoccupied-block populations of this instance.

```python
def group_values(
    delta: np.ndarray,
    V: np.ndarray,
    Q: np.ndarray,
    G: np.ndarray,
    H: np.ndarray,
    eps: float,
) -> tuple[float, float]:
    """
    Return the two unoccupied-block populations.

    Form two reconstructing AO densities from (delta, V, eps) using
    the source first group and second group. Map each with G, not H.
    Score each mapped density against the projector Q.

    Parameters
    ----------
    delta : np.ndarray
        Occupation-change numbers, shape (n,).
    V : np.ndarray
        AO coefficients, shape (n, n).
    Q : np.ndarray
        Unoccupied projector from step 5, shape (n, n).
    G : np.ndarray
        First overlap factor from step 2, shape (n, n).
    H : np.ndarray
        Second overlap factor from step 2, shape (n, n).
    eps : float
        Printed fixture tolerance.

    Returns
    -------
    tuple[float, float]
        First-group block scalar, then the second-group block scalar.

    Raises
    ------
    ValueError
        If the arrays are misaligned, a value is non-finite, or eps
        is not in (0, 1).
    """
    return s1, s2
```

### Step 7

evaluate_instance

Goal
----
Evaluate the supplied AO densities and overlap.

```python
import numpy as np


def evaluate_instance(
    P_i: np.ndarray,
    P_f: np.ndarray,
    S: np.ndarray,
    eps: float,
) -> float:
    """
    Return the second-group unoccupied-block population.

    Parameters
    ----------
    P_i : np.ndarray
        Initial-state AO density, shape (n, n).
    P_f : np.ndarray
        Final-state AO density, shape (n, n).
    S : np.ndarray
        AO overlap, shape (n, n).
    eps : float
        Printed fixture tolerance, 0 < eps < 1.

    Returns
    -------
    float
        Second-group unoccupied-block population.

    Raises
    ------
    ValueError
        If any input array is invalid, eps is not in (0, 1), a pipeline
        stage returns a non-finite value, or the returned scalar is negative.
    """
    return value
```
