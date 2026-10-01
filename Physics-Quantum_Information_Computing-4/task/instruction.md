# Physics-Quantum_Information_Computing-4

## Background

Quantum key distribution grounds the secrecy of a shared key in the physical behaviour of the systems that carry it rather than in an assumption about an adversary's computational power. In the entanglement-based description of a protocol the two legitimate parties hold subsystems of a bipartite state whose purification may be held entirely by an adversary, and a prepare-and-measure implementation is brought into the same description by the source-replacement argument. What the parties can learn about that state is limited to the expectation values of the observables they actually measure during calibration and parameter estimation, so the state is pinned down only up to a family of linear conditions, and the security statement has to hold for every member of that family.

The secrecy of the extracted key is governed by the conditional entropy of the raw measurement record given the adversary's system. Privacy amplification through the leftover-hash lemma sacrifices exactly the amount of information the adversary could hold, and modern composable finite-size analyses assemble their statistical estimators from single-round entropy bounds of this kind. A bound that is too loose wastes key material, while a bound that is not rigorous voids the security claim outright, so the numerical treatment has to deliver a value that remains trustworthy in the presence of floating-point arithmetic.

Outside highly symmetric protocols, where the entropy can be written in closed form, this constrained entropy minimisation has to be carried out numerically over the space of density operators. The objective is nonlinear in the state while the recorded data enter linearly, so the problem falls outside standard linear and quadratic programming, and the practical obstacle is how quickly the number of optimisation variables grows with the Hilbert-space dimension. That growth matters for deployed systems, whose operating conditions drift and whose secrecy bound should ideally be recomputed from live data on the control hardware itself instead of read from a precomputed table.

## Problem

Turning parameter-estimation statistics into a rigorous bound on what an eavesdropper knows about the raw key decides whether a quantum key distribution device may claim a secret key at all, and the deciding quantity is the smallest conditional von Neumann entropy of the raw-key symbol given the adversary's system, minimised over every purification of every bipartite state compatible with the recorded statistics. Established numerical treatments cast this minimisation as a semidefinite program or as a conic program over the quantum relative entropy cone, whose working memory grows with the fourth power of the Hilbert-space dimension and which therefore cannot run on the control hardware of a deployed device; a recent alternative avoids both the semidefinite relaxation and any discretisation of the entropy, replacing the optimisation over the state space at each of its iterations by an unconstrained smooth convex optimisation over one real weight per recorded linear constraint, and delivering at each iteration both a feasible candidate lying above the minimum and a rigorous certificate lying below it, at a cost of one spectral decomposition of an operator of the original dimension per Newton step. It applies to a raw key produced by a general detector and not only by a projective measurement. Evaluate the certificate this scheme delivers for the instance specified below.

Alice and Bob each hold a qutrit, the joint basis is ordered so that the label of the pair $(a,b)$ is $3a+b$ with $a$ on Alice's side. Alice's raw key is the symbol reported by her detector, which returns her computational-basis outcome except that, with probability $\varepsilon = 0.04$, it returns a symbol drawn uniformly at random from $\{0,1,2\}$ instead, so the measurement that produces the key is the positive-operator-valued measure $E_a = (1-\varepsilon)\,|a\rangle\langle a| + (\varepsilon/3)\,\mathbb{1}$ on Alice's qutrit. Parameter estimation records two settings: the computational setting, in which both parties project onto the computational basis, and the Fourier setting, in which Alice projects onto $|x_j\rangle = 3^{-1/2}\sum_{l=0}^{2}\omega^{jl}|l\rangle$ and Bob projects onto $|y_k\rangle = 3^{-1/2}\sum_{l=0}^{2}\omega^{-kl}|l\rangle$ with $\omega=e^{2\pi i/3}$. The recorded outcome probabilities are

$$p^{Z} = \begin{pmatrix} 0.297 & 0.021 & 0.014 \\ 0.018 & 0.284 & 0.026 \\ 0.011 & 0.023 & 0.306 \end{pmatrix}, \qquad p^{X} = \begin{pmatrix} 0.281 & 0.029 & 0.019 \\ 0.024 & 0.292 & 0.031 \\ 0.017 & 0.026 & 0.281 \end{pmatrix},$$

with rows indexed by Alice's outcome and columns by Bob's outcome. The linear information available about the shared state consists of the sixteen equalities that fix the expectation of each joint projector of the two settings to the corresponding table entry, with the pair $(2,2)$ omitted in each setting because the remaining projectors of that setting together with the unit-trace condition already determine it.

Start from the maximally mixed state on the nine-dimensional joint system and run exactly five outer iterations, starting each inner optimisation over the weight vector from the all-zero vector, stopping it once every component of its gradient is below $10^{-12}$ in magnitude, and allowing it at most one hundred steps. Report, in nats and to at least seven decimal places, the certified lower bound delivered by the fifth outer iteration, evaluated at the state that entered that iteration together with the weight vector that iteration returned, and give the entropies, candidate values, certificates and certificate terms you quote to the same precision.

For orientation on the surrounding system: the error correction that consumes this raw key runs at an efficiency of $0.95$, the composable finite-size analysis that consumes this single-round bound uses a block length of $10^{12}$ rounds and a secrecy parameter of $10^{-10}$, the repeaterless bound at the operating loss of the link is $0.0112$ bit per channel use, and instances of this family have previously been treated with an eight-node Gauss-Radau relaxation of the relative entropy.

## Output format

```
## Output format

Scientific reasoning wrapped in <reasoning>...</reasoning> tags. Include enough intermediate calculations to justify the deterministic computation, without turning the response into a general pipeline summary. A single final numeric answer wrapped in <final_answer>...</final_answer> tags.
```

## Your task

Implement **all 12 functions** below into a single file at `/app/solution.py`. Requirements:

- Use the EXACT function name and signature shown for each step.
- Define every function at module scope (importable).
- Do NOT include tests, example usage, prints, or a `__main__` block.
- Available libraries: numpy, scipy, sympy, pandas, scikit-learn, networkx.

Start your file with these imports:

```python
import numpy as np
```

### Step 1

01_build_protocol_data

Goal
----
Assemble the constraint observables, the matching observed-moment vector and the detector

operators of a bipartite qudit key-distribution instance whose parameter estimation uses two

mutually unbiased bases.

```python
def build_protocol_data(d_local: int, p_z: "np.ndarray", p_x: "np.ndarray", epsilon: float) -> "tuple[np.ndarray, np.ndarray, np.ndarray]":
    '''Build the constraint observables, observed moments and detector operators.

    Parameters
    ----------
    d_local : int
        Local Hilbert-space dimension of Alice and of Bob, at least 2.
    p_z : np.ndarray
        Real array of shape (d_local, d_local) holding the computational-setting outcome
        probabilities, rows indexed by Alice, columns by Bob. Entries are nonnegative and
        sum to one within 1e-9.
    p_x : np.ndarray
        Real array of shape (d_local, d_local) holding the Fourier-setting outcome
        probabilities, with the same layout and validity requirements as p_z.
    epsilon : float
        Detector randomisation probability, strictly greater than 0 and at most 1.

    Returns
    -------
    operators : np.ndarray
        Complex array of shape (n, d, d) with n = 2 * (d_local ** 2 - 1) and
        d = d_local ** 2, holding the Hermitian constraint observables in the documented
        order.
    moments : np.ndarray
        Real array of shape (n, ) holding the observed value attached to each observable,
        in the same order.
    detector_ops : np.ndarray
        Complex array of shape (d_local, d, d) holding the detector operators, indexed by
        Alice's key symbol in increasing order.

    Raises
    ------
    ValueError
        If d_local is not an integer of at least 2, if p_z or p_x is not a real array of
        shape (d_local, d_local) with finite entries, if any entry is negative, if either
        table fails to sum to one within 1e-9, or if epsilon is not a real scalar in the
        half-open interval from 0 exclusive to 1 inclusive.
    '''
    return operators, moments, detector_ops
```

### Step 2

02_key_map_blocks

Goal
----
Map a bipartite operator through the detector operators of the protocol, returning the stack of

operators that describes the joint system once Alice's raw-key symbol has been recorded.

```python
def key_map_blocks(operator: "np.ndarray", detector_ops: "np.ndarray") -> "np.ndarray":
    '''Return the stack of key-conditioned operators produced by the detector operators.

    Parameters
    ----------
    operator : np.ndarray
        Square array of shape (d, d) acting on the ordered bipartite space. Real or complex
        input is accepted.
    detector_ops : np.ndarray
        Complex array of shape (m, d, d) with m at least 1, holding the detector operators
        indexed by Alice's key symbol.

    Returns
    -------
    blocks : np.ndarray
        Complex array of shape (m, d, d) whose entry a is the operator conditioned on key
        symbol a.

    Raises
    ------
    ValueError
        If operator is not a two-dimensional square array with at least one row, if its
        entries are not all finite, if detector_ops is not a three-dimensional array with at
        least one entry whose blocks are square of the same dimension as operator, or if the
        detector entries are not all finite.
    '''
    return blocks
```

### Step 3

03_entropy_production

Goal
----
Evaluate the entropy produced by the raw-key measurement process on a given bipartite state,

which is the objective whose constrained minimum bounds Eve's uncertainty about the raw key.

```python
def entropy_production(rho: "np.ndarray", detector_ops: "np.ndarray") -> float:
    '''Return the entropy production of the raw-key measurement process, in nats.

    Parameters
    ----------
    rho : np.ndarray
        Hermitian positive semidefinite array of shape (d, d) with unit trace, acting on the
        ordered bipartite space. Hermiticity is required within 1e-9 and the smallest
        eigenvalue within -1e-9.
    detector_ops : np.ndarray
        Complex array of shape (m, d, d) with m at least 1, holding the detector operators
        indexed by Alice's key symbol.

    Returns
    -------
    production : float
        Entropy produced by the process, in nats.

    Raises
    ------
    ValueError
        If rho is not a two-dimensional square array with at least one row, if its entries are
        not all finite, if it is not Hermitian within 1e-9, if its smallest eigenvalue is below
        -1e-9, if its trace differs from one by more than 1e-9, or if detector_ops is not a
        three-dimensional array of finite (d, d) blocks matching rho with at least one entry.
    '''
    return production
```

### Step 4

04_moving_reference_logarithm

Goal
----
Evaluate the Hermitian operator that the protocol's data-matching step uses as its reference

logarithm at a given bipartite state and detector.

```python
def moving_reference_logarithm(rho: "np.ndarray", detector_ops: "np.ndarray") -> "np.ndarray":
    '''Return the reference logarithm of the data-matching step at rho, in nats.

    Parameters
    ----------
    rho : np.ndarray
        Hermitian positive-definite array of shape (d, d) with unit trace, acting on the
        ordered bipartite space.
    detector_ops : np.ndarray
        Complex array of shape (m, d, d) with m at least 1, holding the detector operators
        indexed by Alice's key symbol. Every operator the detector conditions from rho must
        have smallest eigenvalue above 1e-12.

    Returns
    -------
    reference : np.ndarray
        Complex Hermitian array of shape (d, d) holding the reference logarithm.

    Raises
    ------
    ValueError
        If rho is not a Hermitian positive semidefinite square two-dimensional array of unit
        trace with finite entries, if detector_ops is not a three-dimensional array of finite
        (d, d) blocks matching rho with at least one entry, or if any operator the detector
        conditions from rho has smallest eigenvalue at or below 1e-12.
    '''
    return reference
```

### Step 5

05_objective_gradient

Goal
----
Evaluate the Hermitian operator that represents the derivative of the entropy production of the

raw-key measurement process at a given bipartite state and detector.

```python
def objective_gradient(rho: "np.ndarray", detector_ops: "np.ndarray") -> "np.ndarray":
    '''Return the Hermitian derivative of the entropy production at rho, in nats.

    Parameters
    ----------
    rho : np.ndarray
        Hermitian positive-definite array of shape (d, d) with unit trace, acting on the
        ordered bipartite space. Its smallest eigenvalue must exceed 1e-12.
    detector_ops : np.ndarray
        Complex array of shape (m, d, d) with m at least 1, holding the detector operators
        indexed by Alice's key symbol. Every operator the detector conditions from rho must
        have smallest eigenvalue above 1e-12.

    Returns
    -------
    gradient : np.ndarray
        Complex Hermitian array of shape (d, d) holding the derivative operator.

    Raises
    ------
    ValueError
        If rho is not a Hermitian positive semidefinite square two-dimensional array of unit
        trace with finite entries, if its smallest eigenvalue does not exceed 1e-12, if
        detector_ops is not a three-dimensional array of finite (d, d) blocks matching rho with
        at least one entry, or if any operator the detector conditions from rho has smallest
        eigenvalue at or below 1e-12.
    '''
    return gradient
```

### Step 6

06_gibbs_state

Goal
----
Build the exponential-family state generated by a Hermitian reference logarithm and a vector of

real weights attached to the constraint observables, and return it together with the logarithm

of its normalisation.

Each weight enters the exponent with a minus sign: with L the reference logarithm, M_i the

constraint observables and w_i the weights, the state is

exp(L - sum_i w_i M_i) / tr exp(L - sum_i w_i M_i), and the returned normalisation is

log tr exp(L - sum_i w_i M_i).

```python
def gibbs_state(log_reference: "np.ndarray", constraint_ops: "np.ndarray", weights: "np.ndarray") -> "tuple[np.ndarray, float]":
    '''Return the exponential-family state and the logarithm of its normalisation.

    Parameters
    ----------
    log_reference : np.ndarray
        Hermitian array of shape (d, d) holding the logarithm of the positive-definite
        reference operator. Hermiticity is required within 1e-9.
    constraint_ops : np.ndarray
        Complex array of shape (n, d, d) with n at least 0, holding Hermitian constraint
        observables. Hermiticity is required within 1e-9.
    weights : np.ndarray
        Real array of shape (n, ) with finite entries, holding one weight per observable in
        the order of constraint_ops.

    Returns
    -------
    state : np.ndarray
        Complex Hermitian positive-definite array of shape (d, d) with unit trace.
    log_partition : float
        Natural logarithm of the normalisation of the unnormalised exponential, in nats.

    Raises
    ------
    ValueError
        If log_reference is not a two-dimensional square array with at least one row and
        finite entries, if it is not Hermitian within 1e-9, if constraint_ops is not a
        three-dimensional array of finite (d, d) blocks matching log_reference, if any block
        is not Hermitian within 1e-9, or if weights is not a real finite array of shape (n, ).
    '''
    return state, log_partition
```

### Step 7

07_multiplier_objective

Goal
----
Evaluate the scalar objective that the protocol's data-matching step minimises over the real

weight vector attached to the constraint observables, together with its gradient.

The weights enter with the sign convention of Step 06.

```python
def multiplier_objective(log_reference: "np.ndarray", constraint_ops: "np.ndarray", moments: "np.ndarray", weights: "np.ndarray") -> "tuple[float, np.ndarray]":
    '''Return the data-matching objective and its gradient at the supplied weights.

    Parameters
    ----------
    log_reference : np.ndarray
        Hermitian array of shape (d, d) holding the logarithm of the positive-definite
        reference operator. Hermiticity is required within 1e-9.
    constraint_ops : np.ndarray
        Complex array of shape (n, d, d) with n at least 0, holding Hermitian constraint
        observables. Hermiticity is required within 1e-9.
    moments : np.ndarray
        Real array of shape (n, ) with finite entries, holding the observed value of each
        observable in the order of constraint_ops.
    weights : np.ndarray
        Real array of shape (n, ) with finite entries, holding one weight per observable in
        the order of constraint_ops.

    Returns
    -------
    value : float
        Objective value at the supplied weights, in nats.
    gradient : np.ndarray
        Real array of shape (n, ) holding the gradient of the objective with respect to the
        weights.

    Raises
    ------
    ValueError
        If log_reference is not a Hermitian square two-dimensional array with at least one row
        and finite entries, if constraint_ops is not a three-dimensional array of finite
        Hermitian (d, d) blocks matching log_reference, or if moments or weights is not a real
        finite array of shape (n, ).
    '''
    return value, gradient
```

### Step 8

08_bkm_susceptibility

Goal
----
Evaluate the curvature of the data-matching objective at a given real weight vector, returning

the matrix of its second derivatives with respect to the weights.

```python
def bkm_susceptibility(log_reference: "np.ndarray", constraint_ops: "np.ndarray", weights: "np.ndarray") -> "np.ndarray":
    '''Return the second-derivative matrix of the data-matching objective at the weights.

    Parameters
    ----------
    log_reference : np.ndarray
        Hermitian array of shape (d, d) holding the logarithm of the positive-definite
        reference operator. Hermiticity is required within 1e-9.
    constraint_ops : np.ndarray
        Complex array of shape (n, d, d) with n at least 0, holding Hermitian constraint
        observables. Hermiticity is required within 1e-9.
    weights : np.ndarray
        Real array of shape (n, ) with finite entries, holding one weight per observable in
        the order of constraint_ops.

    Returns
    -------
    curvature : np.ndarray
        Real symmetric array of shape (n, n) holding the second derivatives of the objective
        with respect to the weights.

    Raises
    ------
    ValueError
        If log_reference is not a Hermitian square two-dimensional array with at least one row
        and finite entries, if constraint_ops is not a three-dimensional array of finite
        Hermitian (d, d) blocks matching log_reference, or if weights is not a real finite
        array of shape (n, ).
    '''
    return curvature
```

### Step 9

09_solve_multipliers

Goal
----
Locate the real weight vector at which the data-matching objective attains its minimum, and

report how far the resulting exponential-family state still is from reproducing the observed

moments.

```python
def solve_multipliers(log_reference: "np.ndarray", constraint_ops: "np.ndarray", moments: "np.ndarray", initial_weights: "np.ndarray", tol: float, max_iter: int) -> "tuple[np.ndarray, float]":
    '''Return the minimizing weight vector and the residual reached at it.

    Parameters
    ----------
    log_reference : np.ndarray
        Hermitian array of shape (d, d) holding the logarithm of the positive-definite
        reference operator. Hermiticity is required within 1e-9.
    constraint_ops : np.ndarray
        Complex array of shape (n, d, d) with n at least 0, holding Hermitian constraint
        observables. Hermiticity is required within 1e-9.
    moments : np.ndarray
        Real array of shape (n, ) with finite entries, holding the observed value of each
        observable in the order of constraint_ops.
    initial_weights : np.ndarray
        Real array of shape (n, ) with finite entries, holding the starting weight vector.
    tol : float
        Positive stopping threshold on the largest magnitude among the gradient entries.
    max_iter : int
        Nonnegative maximum number of second-order steps.

    Returns
    -------
    weights : np.ndarray
        Real array of shape (n, ) holding the weight vector reached.
    residual : float
        Largest magnitude among the gradient entries at the returned weight vector.

    Raises
    ------
    ValueError
        If log_reference is not a Hermitian square two-dimensional array with at least one row
        and finite entries, if constraint_ops is not a three-dimensional array of finite
        Hermitian (d, d) blocks matching log_reference, if moments or initial_weights is not a
        real finite array of shape (n, ), if tol is not a positive finite real scalar, or if
        max_iter is not a nonnegative integer.
    '''
    return weights, residual
```

### Step 10

10_outer_iteration

Goal
----
Advance the pipeline by one outer iteration: from the current bipartite state, produce the next

candidate state and the real weight vector that generates it. The exponential family of Steps 06 to 09 is generated by the reference logarithm of Step 04 evaluated at the current state.

```python
def outer_iteration(rho: "np.ndarray", constraint_ops: "np.ndarray", moments: "np.ndarray", detector_ops: "np.ndarray", tol: float, max_iter: int) -> "tuple[np.ndarray, np.ndarray]":
    '''Return the next candidate state and the weight vector that generates it.

    Parameters
    ----------
    rho : np.ndarray
        Hermitian positive-definite array of shape (d, d) with unit trace, acting on the
        ordered bipartite space, with smallest eigenvalue above 1e-12.
    constraint_ops : np.ndarray
        Complex array of shape (n, d, d) with n at least 0, holding Hermitian constraint
        observables matching the dimension of rho.
    moments : np.ndarray
        Real array of shape (n, ) with finite entries, holding the observed value of each
        observable in the order of constraint_ops.
    detector_ops : np.ndarray
        Complex array of shape (m, d, d) with m at least 1, holding the detector operators
        indexed by Alice's key symbol. Every operator the detector conditions from rho must
        have smallest eigenvalue above 1e-12.
    tol : float
        Positive stopping threshold used by the inner search.
    max_iter : int
        Nonnegative step budget for the inner search.

    Returns
    -------
    next_state : np.ndarray
        Complex Hermitian positive-definite array of shape (d, d) with unit trace.
    weights : np.ndarray
        Real array of shape (n, ) holding the weight vector that generates next_state.

    Raises
    ------
    ValueError
        If rho fails the requirements above, if constraint_ops is not a three-dimensional array
        of finite Hermitian blocks matching rho, if moments is not a real finite array of shape
        (n, ), if detector_ops fails the requirements above, if tol is not a positive finite
        real scalar, or if max_iter is not a nonnegative integer.
    '''
    return next_state, weights
```

### Step 11

11_entropy_certificate

Goal
----
Evaluate the rigorous lower bound on the constrained minimum of the raw-key entropy production

that a given linearisation state together with a given real weight vector certifies.

The weights shift the derivative operator G of Step 05 with a plus sign, as G + sum_i w_i M_i,

with M_i the constraint observables and w_i the weights.

```python
def entropy_certificate(rho: "np.ndarray", constraint_ops: "np.ndarray", moments: "np.ndarray", detector_ops: "np.ndarray", weights: "np.ndarray") -> float:
    '''Return the certified lower bound on the constrained minimum, in nats.

    Parameters
    ----------
    rho : np.ndarray
        Hermitian positive-definite array of shape (d, d) with unit trace, acting on the
        ordered bipartite space, with smallest eigenvalue above 1e-12.
    constraint_ops : np.ndarray
        Complex array of shape (n, d, d) with n at least 0, holding Hermitian constraint
        observables matching the dimension of rho.
    moments : np.ndarray
        Real array of shape (n, ) with finite entries, holding the observed value of each
        observable in the order of constraint_ops.
    detector_ops : np.ndarray
        Complex array of shape (m, d, d) with m at least 1, holding the detector operators
        indexed by Alice's key symbol. Every operator the detector conditions from rho must
        have smallest eigenvalue above 1e-12.
    weights : np.ndarray
        Real array of shape (n, ) with finite entries, holding one weight per observable in the
        order of constraint_ops.

    Returns
    -------
    bound : float
        Certified lower bound on the constrained minimum of the entropy production, in nats.

    Raises
    ------
    ValueError
        If rho fails the requirements above, if constraint_ops is not a three-dimensional array
        of finite Hermitian blocks matching rho, if moments or weights is not a real finite
        array of shape (n, ), or if detector_ops fails the requirements above.
    '''
    return bound
```

### Step 12

12_certified_entropy_bracket

Goal
----
Run the complete pipeline for the two-setting bipartite instance and return the final candidate's entropy-production value and the final iteration's certified lower bound, both in nats.

```python
def certified_entropy_bracket(d_local: int, p_z: "np.ndarray", p_x: "np.ndarray", epsilon: float, n_outer: int, tol: float, max_iter: int) -> "tuple[float, float]":
    '''Return the candidate value and the certified lower bound of the instance, in nats.

    Parameters
    ----------
    d_local : int
        Local Hilbert-space dimension of Alice and of Bob, at least 2.
    p_z : np.ndarray
        Real array of shape (d_local, d_local) holding the computational-setting outcome
        probabilities, nonnegative and summing to one within 1e-9.
    p_x : np.ndarray
        Real array of shape (d_local, d_local) holding the Fourier-setting outcome
        probabilities, with the same layout and validity requirements as p_z.
    epsilon : float
        Detector randomisation probability, strictly greater than 0 and at most 1.
    n_outer : int
        Number of outer iterations, at least 1.
    tol : float
        Positive stopping threshold used by each inner search.
    max_iter : int
        Nonnegative step budget for each inner search.

    Returns
    -------
    candidate : float
        Entropy production of the state reached after the final outer iteration, in nats.
    bound : float
        Certified lower bound produced by the final outer iteration, in nats.

    Raises
    ------
    ValueError
        If d_local is not an integer of at least 2, if p_z or p_x fails the requirements above,
        if epsilon is not a real scalar in the half-open interval from 0 exclusive to 1
        inclusive, if n_outer is not an integer of at least 1, if tol is not a positive finite
        real scalar, or if max_iter is not a nonnegative integer.
    '''
    return candidate, bound
```
