# Physics-Quantum_Information_Computing-2

## Background

All quantities are dimensionless. Computational basis order is $|00\rangle$, $|01\rangle$, $|10\rangle$, $|11\rangle$, with Alice's resource qubit first. The initial resource is $|\Phi^+\rangle = (|00\rangle + |11\rangle)/\sqrt{2}$. A local amplitude-damping channel with parameter $p$ has Kraus matrices

$$
K_0(p)=\begin{pmatrix}1&0\\0&\sqrt{1-p}\end{pmatrix},\qquad
K_1(p)=\begin{pmatrix}0&\sqrt{p}\\0&0\end{pmatrix}.
$$

Alice measures her input qubit and her resource qubit in the ordered basis $(\Phi^+,\ \Phi^-,\ \Psi^+,\ \Psi^-)$, where $|\Psi^\pm\rangle = (|01\rangle \pm |10\rangle)/\sqrt{2}$, and Bob applies $(I,\ Z,\ X,\ XZ)$, respectively. For a pure Haar-distributed input $|\psi\rangle$, the corrected quantum branches have normalized states $\rho_j$ and Born probabilities $p_j$. In assessment $R$, the fidelity is $\langle\psi|\rho_j|\psi\rangle$ with $j$ recorded. In assessment $E$, the fidelity is $\langle\psi|\sum_j p_j \rho_j|\psi\rangle$; the classical message has already been used for Bob's correction when its record is erased. Both assessments retain the sampled hardware profile. Assessment $R$ uses the paper-grounded realized-fidelity framework; assessment $E$ and its quantum and classical erased-record distributions are constructed comparisons defined for this task.

The classical protocol measures the input in the computational basis and prepares the corresponding basis state at Bob. Assess it under the same $R$ or $E$ record convention as the quantum protocol. This protocol attains the optimal classical Haar mean fidelity.

The unknown reference damping parameters $(p_A, p_B)$ lie in $[0.01, 0.96]^2$. The following synthetic calibration values describe the reference resource with unit exposure on both links and the physical A link at Alice. They specify a unique ordered pair within absolute residual $10^{-9}$.

| Observable | Value |
|---|---:|
| $E[F]$ | 0.7413538548322107 |
| $E[F^2]$ | 0.5724789779364854 |
| $\Pr(F \le 0.62)$ | 0.22266734190342186 |
| $\Pr(F \le 0.81)$ | 0.5618389749676874 |
| $\Pr(F \le 0.91)$ | 0.9398264313912843 |

The hardware and robust design below are constructed extensions of the fidelity-distribution certification method. For profile $c$ and scenario $s$, the two physical-link damping parameters are

$$
d_A=1-(1-p_A)^{e_{A,c}\,r_{A,s}},\qquad
d_B=1-(1-p_B)^{e_{B,c}\,r_{B,s}}.
$$

Orientation 0 places $(d_A, d_B)$ at (Alice, Bob); orientation 1 places $(d_B, d_A)$ there. Scenario indices $0,1,2,3$ correspond, respectively, to exposure multipliers $(0.85, 1.20)$, $(1.00, 1.00)$, $(1.20, 0.85)$, $(1.15, 1.15)$. The scenario is a fixed but unknown condition for a run, and the same profile mixture is used in each scenario.

| Profile $c$ | Orientation | $e_A$ | $e_B$ | Cost $k_c$ |
|---:|---:|---:|---:|---:|
| 0 | 0 | 1.00 | 1.00 | 0.00 |
| 1 | 1 | 1.00 | 1.00 | 0.05 |
| 2 | 0 | 0.25 | 1.00 | 0.45 |
| 3 | 1 | 0.25 | 1.00 | 0.50 |
| 4 | 0 | 1.00 | 0.25 | 0.50 |
| 5 | 1 | 1.00 | 0.25 | 0.55 |
| 6 | 0 | 0.40 | 0.40 | 0.72 |
| 7 | 1 | 0.40 | 0.40 | 0.77 |
| 8 | 0 | 0.12 | 0.12 | 1.35 |

Let $W_{\alpha,\beta}(F) = \dfrac{F^{\alpha-1}(1-F)^{\beta-1}}{B(\alpha,\beta)}$. Importance indices $k = 0,1,2,3$ correspond to $(\alpha,\beta) = (3,1),\ (5,1),\ (8,1),\ (12,2.5)$. Let $C_{mk}$ be the classical expectation of $W_k$ in assessment $m$, and $E_{msc}$ the corresponding quantum expectation in scenario $s$ and profile $c$. The advantage of policy $x$ in assessment $m$ is the minimum, over all scenario–importance pairs $(s,k)$, of $\sum_c x_c\,\big(E_{msc}[W_k(F)] - C_{mk}\big)$. Policies are probability vectors over the nine profiles, subject to the cost and per-scenario high-fidelity constraints in the prompt. In an exact objective tie, choose the lexicographically smallest probability vector in increasing profile order. Binding scenario–importance pairs use zero-based indices and equality tolerance $10^{-7}$. The expectations refer to the continuous Haar ensemble; integrations must converge to absolute error $10^{-8}$ before optimization. Round only the reported results.

## Problem

Quantify the signed certification gap $\Delta = J_R - J_E$ for the teleportation hardware in the background, where $R$ retains the corrected measurement outcome and $E$ erases it. In each assessment, $J_m$ is the largest worst-case beta-weighted advantage over the specified classical protocol, optimized over a randomized hardware policy with expected cost at most $0.35$ and probability of fidelity strictly exceeding $0.88$ at least $0.25$ in every environmental scenario. Each assessment has its own policy, chosen before the Haar input, environmental scenario, and measurement outcome; the hardware-profile label remains available in both assessments. The five recorded-outcome calibration observations determine the ordered reference damping pair. Return $\Delta$ with absolute tolerance $0.000005$. In the reasoning, report that pair and, for each assessment, $J_m$, the nonzero profile probabilities, the binding scenario–importance pairs, minimum high-fidelity probability, and expected cost; these numerical diagnostics have absolute tolerance $0.00005$. Justify the two quantum fidelity distributions and the corresponding classical distributions.
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

damped_resource

Goal
----
Construct the amplitude-damped Bell resource.

```python
import numpy as np

def damped_resource(a: float, b: float) -> np.ndarray:
    """Construct the amplitude-damped Bell resource.

    a, b : float
        Alice and Bob damping probabilities in [0,1].
    Returns
    -------
    ndarray, shape (4,4), float
        Resource density matrix in the ordered basis 00,01,10,11.

    All quantities are dimensionless. Inputs are preserved.
    Raises
    ------
    ValueError
        For invalid input shapes or data outside the stated domain.
    """
    return result
```

### Step 2

conditional_fidelity_law

Goal
----
Encode a recorded or erased fidelity distribution.

```python
import numpy as np

def conditional_fidelity_law(rho: np.ndarray, mode: int = 0) -> np.ndarray:
    """Encode a recorded or erased fidelity distribution.

    rho : ndarray, shape (4,4)
        A density matrix in the damped Phi+ family, in basis 00,01,10,11.
    mode : int
        0 for recorded outcomes R; 1 for the erased channel E.
    Returns
    -------
    ndarray, shape (4,5), float
        Each row is [q0,q1,q2,p0,p1]. In mode 0 the rows are Bell outcomes
        Phi+,Phi-,Psi+,Psi-. Mode 1 encodes four identical artificial rows,
        each with p0=1/4 and p1=0, whose fidelity is the erased channel
        fidelity. These rows represent one normalized distribution.
        The outcome probability at Bloch
        coordinate z is p0+p1*z, and its conditional fidelity is
        (q0+q1*z+q2*z*z)/(p0+p1*z). The joint measure is
        (p0+p1*z)*dz/2. Zero-probability endpoints have zero measure.

    All quantities are dimensionless. Inputs are preserved.
    Raises
    ------
    ValueError
        For invalid input shapes or data outside the stated domain.
    """
    return result
```

### Step 3

fidelity_preimages

Goal
----
Find each branch preimage of the cumulative-fidelity event.

```python
import numpy as np

def fidelity_preimages(law: np.ndarray, threshold: float) -> np.ndarray:
    """Find each branch preimage of the cumulative-fidelity event.

    law : ndarray, shape (4,5)
        A physical conditional_fidelity_law output with columns q0,q1,q2,p0,p1.
    threshold : float
        Any finite fidelity threshold.
    Returns
    -------
    ndarray, shape (4,2,2), float
        For each Bell row, up to two maximal intervals [lo,hi] with lo<hi,
        ordered by lo, whose union is the event up to sets of zero Haar
        measure. An interval may end at a zero-probability endpoint, but
        isolated points, including an isolated zero-probability endpoint,
        are omitted, so a row whose event is a single point is [[0,0],[0,0]].
        Unused slots are [0,0]; the full event is [[-1,1],[0,0]]. Equality
        includes a constant branch.

    All quantities are dimensionless. Inputs are preserved.
    Raises
    ------
    ValueError
        For invalid input shapes or data outside the stated domain.
    """
    return result
```

### Step 4

fidelity_cdf

Goal
----
Evaluate the realized-fidelity cumulative distribution.

```python
import numpy as np

def fidelity_cdf(law: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    """Evaluate the realized-fidelity cumulative distribution.

    law : ndarray, shape (4,5)
        Physical conditional_fidelity_law output.
    thresholds : ndarray, shape (T,)
        Finite thresholds, including values outside [0,1]; T may be zero.
    Returns
    -------
    ndarray, shape (T,), float
        Pr(F<=thresholds[i]) in supplied order, using all four Bell outcomes.

    All quantities are dimensionless. Inputs are preserved.
    Raises
    ------
    ValueError
        For invalid input shapes or data outside the stated domain.
    """
    return result
```

### Step 5

fidelity_moments

Goal
----
Evaluate raw moments of the realized conditional fidelity.

```python
import numpy as np

def fidelity_moments(law: np.ndarray, orders: np.ndarray) -> np.ndarray:
    """Evaluate raw moments of the realized conditional fidelity.

    law : ndarray, shape (4,5)
        Physical conditional_fidelity_law output.
    orders : ndarray, shape (M,)
        Nonnegative integer exponents, in any order; M may be zero.
    Returns
    -------
    ndarray, shape (M,), float
        E[F**orders[i]], with zeroth moment equal to normalization.

    All quantities are dimensionless. Inputs are preserved.
    Raises
    ------
    ValueError
        For invalid input shapes or data outside the stated domain.
    """
    return result
```

### Step 6

importance_expectations

Goal
----
Evaluate beta-weighted expectations of the fidelity distribution.

```python
import numpy as np

def importance_expectations(law: np.ndarray, priors: np.ndarray) -> np.ndarray:
    """Evaluate beta-weighted expectations of the fidelity distribution.

    law : ndarray, shape (4,5)
        Physical conditional_fidelity_law output.
    priors : ndarray, shape (K,2)
        Finite [alpha,beta] rows with both shapes >=1; K may be zero.
    Returns
    -------
    ndarray, shape (K,), float
        E[F**(alpha-1)*(1-F)**(beta-1)/B(alpha,beta)] in row order.

    All quantities are dimensionless. Inputs are preserved.
    Raises
    ------
    ValueError
        For invalid input shapes or data outside the stated domain.
    """
    return result
```

### Step 7

classical_reference

Goal
----
Compute the classical reference for either record assessment.

```python
import numpy as np

def classical_reference(priors: np.ndarray, thresholds: np.ndarray, mode: int = 0) -> np.ndarray:
    """Compute the classical reference for either record assessment.

    priors : ndarray, shape (K,2)
        Finite beta shapes [alpha,beta], each >=1; K may be zero.
    thresholds : ndarray, shape (T,)
        Finite CDF thresholds, including values outside [0,1]; T may be zero.
    mode : int
        0 for the recorded classical outcome R; 1 for erased outcome E.
    Returns
    -------
    ndarray, shape (K+T,), float
        First K entries are classical E[W_k(F)]; last T entries are
        classical Pr(F<=thresholds[i]), each in supplied order.

    All quantities are dimensionless. Inputs are preserved.
    Raises
    ------
    ValueError
        For invalid input shapes or data outside the stated domain.
    """
    return result
```

### Step 8

infer_damping

Goal
----
Infer the ordered reference damping pair from distributional calibration.

```python
import numpy as np

def infer_damping(observations: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    """Infer the ordered reference damping pair from distributional calibration.

    observations : ndarray, shape (2+T,)
        [E[F], E[F**2], CDF(thresholds[0]), ..., CDF(thresholds[T-1])].
    thresholds : ndarray, shape (T,)
        Finite thresholds in supplied order. The data satisfy the stated
        unique-pair identifiability condition.
    Returns
    -------
    ndarray, shape (2,), float
        Inferred [p_A,p_B] in physical-link order.
        Inconsistent observations with best residual norm >1e-8 raise ValueError.

    All quantities are dimensionless. Inputs are preserved.
    Raises
    ------
    ValueError
        For invalid input shapes or data outside the stated domain.
    """
    return result
```

### Step 9

optimize_policy

Goal
----
Find the robust randomized hardware policy.

```python
import numpy as np

def optimize_policy(quality: np.ndarray, tails: np.ndarray, costs: np.ndarray, budget: float, tail_min: float) -> np.ndarray:
    """Find the robust randomized hardware policy.

    quality : ndarray, shape (R,C)
        Finite advantage rows, scenario-major and importance-minor when applicable.
    tails : ndarray, shape (S,C)
        High-fidelity probabilities in [0,1] for each scenario and profile.
    costs : ndarray, shape (C,)
        Nonnegative profile costs. R,S,C are positive.
    budget : float
        Finite nonnegative expected-cost ceiling.
    tail_min : float
        Minimum required tail probability in [0,1].
    Returns
    -------
    ndarray, shape (C+3,), float
        [optimal signed advantage, x[0], ..., x[C-1], minimum tail probability,
        expected cost]. An infeasible policy problem raises ValueError.

    All quantities are dimensionless. Inputs are preserved.
    Raises
    ------
    ValueError
        For invalid input shapes or data outside the stated domain.
    """
    return result
```

### Step 10

teleportation_benchmark

Goal
----
Compose the distribution-calibrated robust teleportation benchmark.

```python
import numpy as np

def teleportation_benchmark(observations: np.ndarray, thresholds: np.ndarray, profiles: np.ndarray, scenarios: np.ndarray, priors: np.ndarray, budget: float, target: float, tail_min: float) -> float:
    """Compose the distribution-calibrated robust teleportation benchmark.

    observations, thresholds : ndarray
        Calibration arrays of shapes (2+T,) and (T,), with the infer_damping contract.
    profiles : ndarray, shape (C,4)
        Rows [orientation,e_A,e_B,cost]; orientation in {0,1}, positive exposures,
        nonnegative cost; C>0. Orientation 1 exchanges the resulting dampings.
    scenarios : ndarray, shape (S,2)
        Positive physical exposure multipliers [r_A,r_B]; S>0.
    priors : ndarray, shape (K,2)
        Beta importance [alpha,beta] rows, shapes >=1; K>0.
    budget, target, tail_min : float
        Nonnegative cost ceiling, finite fidelity threshold, and reliability
        floor in [0,1]. The event is F>target. Use the exposure formula and
        common-policy objective in the global background.
    Returns
    -------
    float
        Signed difference J_R-J_E of separately optimized worst-case advantages,
        each subject to its own cost and per-scenario reliability constraints.
        Compose the full public pipeline, including infer_damping and
        optimize_policy. An infeasible design raises ValueError.

    All quantities are dimensionless. Inputs are preserved.
    Raises
    ------
    ValueError
        For invalid input shapes or data outside the stated domain.
    """
    return 0.0
```
