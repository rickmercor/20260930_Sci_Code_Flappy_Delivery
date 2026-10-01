# Physics-Condensed_Matter_Physics-34

## Background

This constructed finite Coulomb-gas problem applies orthogonal-symmetry random-matrix level statistics to quantum spectra. The inverse quartic confinement and observation protocol are synthetic applications of the source's general-weight polynomial method; the probability law is exactly discrete.

For zero-based labels \(i=0,\ldots,127\), the dimensionless nodes and masses are
\[
x_i=-2+4i/127+0.003\sin(1.7i),\qquad
w_i(\theta)=\exp[-x_i^2/4-\theta x_i^4/150+0.07x_i].
\]
Angles are in radians. A configuration \(X\) is a 32-element subset, with probability proportional to
\[
\prod_{i<j,\ i,j\in X}|x_j-x_i|\prod_{i\in X}w_i(\theta).
\]
The complete record is \(E=\{7\notin X,\;42\in X,\;96\notin X\}\); these integers are node labels. Other nodes are unobserved. The second-largest occupied coordinate is \(\lambda_{(2)}\), and \(q_a(c)=\Pr_\theta(\#\{i\in X:x_i>c\}=a\mid E)\) for \(a=0,1\).

The discrete skew product is bilinear:
\[
B_{ij}=\tfrac12 w_iw_j\operatorname{sign}(x_j-x_i),\quad
\langle f,g\rangle=f^TBg,\quad \operatorname{sign}(0)=0.
\]
Normalized degree-ordered sampled polynomials satisfy \(S^TBS=J_{16}\), where \(J_r=I_r\otimes\begin{pmatrix}0&1\\-1&0\end{pmatrix}\). Equivalent normalized polynomial gauges describe the same probabilities.

The direct Pfaffian convention is \(\Pr(A\subseteq X)=\operatorname{Pf}(K_A)\), with adjacent node components \((2i,2i+1)\) and blocks \(K(x,y)=\begin{pmatrix}I(x,y)&S(y,x)\\-S(x,y)&-D(x,y)\end{pmatrix}\). The masses define the discrete measure, including its normalization convention. On a queried upper domain, \(H\) denotes the restricted conditional correlation kernel and \(J-H\) its gap matrix.

Numerical allowances: \(2\times10^{-6}\) absolute for \(\theta\), \(\max(10^{-10},5\times10^{-4}|q_{\rm ref}|)\) for each requested probability, and \(5\times10^{-6}\) nats absolute for \(L\). Computation uses unrounded intermediates; the six checkpoints and \(L\) are the complete numerical reporting requirement.

## Problem

Infer the confinement of the 32-level orthogonal-symmetry ensemble specified below from an incomplete occupancy record, and predict a joint event for its second-largest level. Determine the unique \(\theta\in[0,4]\) satisfying \(\Pr_\theta(\lambda_{(2)}\le1.90\mid E)=0.0012\). Report \(L=-\ln\Pr_\theta(E\cap\{\lambda_{(2)}\le1.94\})\) to six decimal places as the tagged scalar. In the short reasoning block give the six decisive checkpoints \(\theta,\Pr(E),q_0(1.90),q_1(1.90),q_0(1.94),q_1(1.94)\), with probabilities to at least four significant figures. Justify the polynomial recurrence and its numerical stability at degree 31 using the source's general-weight method, and distinguish its point-conditioned second-level density from this finite cumulative event. Explain briefly how the count calculation remains valid for a singular gap matrix.

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

discrete_skew_metric

Goal
----
Eq. (10): finite discrete beta=1 skew measure in supplied node order.

```python
import numpy as np

def discrete_skew_metric(nodes, weights):
    """Construct the discrete beta=1 polynomial skew metric.

    Parameters
    ----------
    nodes : real array, (m,)
        Distinct dimensionless abscissae, m>=1, in any supplied order.
    weights : real array, (m,)
        Nonnegative dimensionless masses. Zero masses are permitted here.

    Returns
    -------
    float64 array, (m,m)
        B such that f.T@B@g is one half of the sum of
        f(x_i)g(x_j)sign(x_j-x_i)w_i*w_j over all ordered pairs.
        The diagonal sign is zero. Supplied node order is retained.

    Raises
    ------
    ValueError
        For nonfinite/nonreal inputs, wrong shapes, repeated nodes,
        an empty domain, or negative weights.
    """
    return np.zeros((len(nodes), len(nodes)))
```

### Step 2

skew_project

Goal
----
Algorithms 2–3 and Section 3.3: reorthogonalized symplectic projection with the declared ESR gauge; accumulated coefficients preserve the original-vector reconstruction.

```python
import numpy as np

def skew_project(basis, vector, metric, gauge=2):
    """Extend an ordered skew-orthogonal polynomial basis by one vector.

    Parameters
    ----------
    basis : real array, (m,k)
        Independent columns with complete consecutive pairs normalized to J.
        If k is odd, its last column is unpaired and skew-orthogonal to the
        earlier pairs. Under gauge 1 or 2 each even-indexed column has Euclidean
        norm one. Empty basis (m,0) is allowed.
    vector : real array, (m,)
        Candidate vector. The prescribed residual has a nonzero normalization.
    metric : real array, (m,m)
        Skew-symmetric bilinear metric B; no conjugation is used.
    gauge : int in {1,2,3}
        ESR1, ESR2, or ESR3m, respectively. For an unpaired candidate choose
        r11=||v||_2 (1,2) or r11=1 (3). To pair with the last column choose
        r12=last.T@v for ESR2 and zero for ESR1/ESR3m, then r22=last.T@B@v.

    Returns
    -------
    float64 array, (m+k+1,)
        Normalized new vector q, then k projection coefficients h, then scale d.
        The original vector equals basis@h+d*q. Apply two complete modified
        skew-projection passes, pairing consecutive columns. Each pass removes
        its own correction; h is their accumulated sum. Normalize only after
        both passes. The final partner uses the stated ESR gauge on each pass.
        All entries are dimensionless. Equivalent computations are accepted.

    Raises
    ------
    ValueError
        For invalid dimensions, nonfinite/nonreal values, gauge outside {1,2,3},
        a metric failing skew symmetry at atol=1e-12, or zero final scale.
        Basis normalization/independence are caller preconditions.
    """
    return np.zeros(len(vector) + np.shape(basis)[1] + 1)
```

### Step 3

symplectic_arnoldi

Goal
----
Algorithm 4 and Section 3.5: the general symplectic Arnoldi construction, rather than assuming a three-term Lanczos recurrence.

```python
import numpy as np

def symplectic_arnoldi(nodes, weights, count, gauge=2):
    """Construct sampled SOPs by the paper's symplectic Arnoldi map.

    Parameters
    ----------
    nodes, weights : real arrays, (m,)
        Distinct dimensionless nodes in supplied order, strictly positive masses.
    count : even int
        Number of polynomial columns, 2<=count<=m. Column j has degree j.
    gauge : int in {1,2,3}
        Same ESR choice as skew_project. The same gauge is used for all columns.

    Returns
    -------
    float64 array, (m,count)
        S[i,j]=p_j(nodes[i]), with S.T@B@S=J_(count/2). Start from the constant
        vector and use multiplication by nodes to extend the polynomial space.
        The two-pass skew_project convention fixes the otherwise free pair gauge.
        Nodes remain in supplied order; columns are ordered by polynomial degree.

    Raises
    ------
    ValueError
        For invalid metric inputs, nonpositive masses, invalid count/gauge,
        or a zero skew normalization during the construction.
    """
    return np.zeros((len(nodes), count))
```

### Step 4

orthogonal_ensemble_kernel

Goal
----
Section 4.2.1, applied to Eq. (10): assemble the direct Pfaffian finite orthogonal-ensemble kernel, including its sign term and gauge-invariant SOP pairing.

```python
import numpy as np

def orthogonal_ensemble_kernel(nodes, weights, basis):
    """Build the beta=1 Pfaffian correlation kernel from sampled SOPs.

    Parameters
    ----------
    nodes, weights : real arrays, (m,)
        Distinct dimensionless abscissae and nonnegative masses.
    basis : real array, (m,n)
        Even n>=2, sampled SOP columns normalized to the consecutive-pair J
        under discrete_skew_metric. Normalization is a caller precondition.

    Returns
    -------
    float64 array, (2*m,2*m)
        Direct Pfaffian kernel (not the quaternion-determinant kernel), with
        adjacent components (2*i,2*i+1) for node i, in the block orientation
        [[I(x,y),S(y,x)],[-S(x,y),-D(x,y)]] of Section 4.2.1.
        Use the discrete sign transform psi_j(x)=sum_y sign(x-y)w_y p_j(y)/2
        and the bilinear symplectic pairing of consecutive SOPs. I includes
        the bare -sign(x-y)/2 term. No quadrature factor is appended: weights
        already define the exact finite discrete ensemble. All entries are
        dimensionless; Pf(K_A) is the inclusion probability of node set A.

    Raises
    ------
    ValueError
        For invalid metric inputs or a nonfinite/nonreal basis of wrong shape
        or with a nonpositive/odd column count.
    """
    return np.zeros((2 * len(nodes), 2 * len(nodes)))
```

### Step 5

signed_pfaffian

Goal
----
Eq. (2) and the signed skew-factorization foundation: preserve the Pfaffian sign/phase. This supporting operation is established mathematics, not a new contribution claimed for this paper.

```python
import numpy as np

def signed_pfaffian(matrix):
    """Evaluate a signed, possibly complex Pfaffian.

    Parameters
    ----------
    matrix : real or complex array, (2*m,2*m)
        Finite skew-symmetric matrix A=-A.T, including singular and empty cases.

    Returns
    -------
    complex scalar
        Pf(A), with Pf([[0,a],[-a,0]])=a and Pf(empty)=1. This is a bilinear
        Pfaffian, not a Hermitian determinant. Preserve its sign/complex phase.

    Raises
    ------
    ValueError
        For a nonfinite matrix, nonsquare/odd shape, or failure of A=-A.T
        at absolute tolerance 1e-12 (rtol=0).
    """
    return 0j
```

### Step 6

condition_spectral_record

Goal
----
Proposition 2.5 and Algorithm 1: replay binary observations with the exact conditional kernel and multiply their successive probabilities. Original node labels survive previous eliminations.

```python
import numpy as np

def condition_spectral_record(kernel, record):
    """Apply sequential binary observations using conditional Pfaffian measures.

    Parameters
    ----------
    kernel : real array, (2*m,2*m)
        Valid finite Pfaffian kernel in adjacent node-component order.
    record : integer array, (r,2)
        Distinct ORIGINAL node labels and observed occupancy (0 or 1), in the
        requested elimination order. Unobserved nodes are unmeasured, not absent.
        Empty record is allowed. The joint observed event must have positive mass.

    Returns
    -------
    float64 array, (1+4*(m-r)**2,)
        First the joint probability of the record, then the conditional kernel
        on remaining original node labels in ascending order, flattened in C order.
        Use Proposition 2.5: for elimination block A subtract
        K_RA @ inv(K_A-(1-occupancy)*J_1) @ K_AR from K_RR.
        J_1=[[0,1],[-1,0]], hence inv(p*J_1)=-J_1/p. This equation fixes the
        sign even if a displayed pseudocode update uses a conflicting sign.
        Conditional probabilities within 1e-12 of [0,1] may be clipped for
        roundoff; exact allowed deterministic outcomes contribute probability 1.

    Raises
    ------
    ValueError
        For invalid dimensions, nonfinite/nonreal data, nonintegral/repeated or
        out-of-range labels, nonbinary observations, failure of skew symmetry
        at atol=1e-12, or a zero-probability requested observation. Also raise
        if any encountered conditional occupancy lies outside [-1e-12,1+1e-12].
    """
    return np.zeros(1 + 4 * (len(kernel) // 2 - len(record)) ** 2)
```

### Step 7

zero_one_level_probabilities

Goal
----
Conditional order-statistic construction around Eq. (15): obtain zero/one probabilities from the gap generating polynomial. Polynomial coefficients extend the nonsingular derivative expression to valid deterministic-count limits.

```python
import numpy as np

def zero_one_level_probabilities(kernel):
    """Return zero- and one-level probabilities on a restricted domain.

    Parameters
    ----------
    kernel : real or complex array, (2*m,2*m)
        Valid Pfaffian correlation kernel restricted to the queried nodes,
        with adjacent components. Complex symplectic gauge representations are
        allowed when their inclusion probabilities describe a real probability law.

    Returns
    -------
    float64 array, (2,)
        [P(number of nodes occupied=0), P(number of nodes occupied=1)]. These
        are the first two coefficients of Pf(J_m+(z-1)*K). This includes cases
        where J_m-K is singular and the empty domain gives [1,0]. Equivalent
        polynomial, coefficient, or nonsingular derivative calculations are valid.
        Outputs below zero by at most 2e-12 may be clipped to zero.

    Raises
    ------
    ValueError
        For the same finite even skew-matrix violations as signed_pfaffian,
        or coefficient imaginary part above 2e-10, or real coefficients outside
        [-2e-12,1+2e-12]. Validity of the entire point process is a precondition.
    """
    return np.zeros(2)
```

### Step 8

infer_confinement

Goal
----
Task-specific inverse use of the main paper's general-weight finite ensemble. Root finding is a supporting operation; its probability map must use the preceding scientific functions.

```python
import numpy as np

def infer_confinement(nodes, count, record, threshold, target, bracket, tilt=0.07):
    """Infer a quartic confinement parameter from a conditional tail probability.

    Parameters
    ----------
    nodes : real array, (m,)
        Distinct dimensionless nodes, supplied order fixes observation labels.
    count : even int, 2<=count<=m
        Fixed total number of levels before conditioning.
    record : integer array, (r,2)
        Original node labels and binary observations as in condition_spectral_record.
    threshold : finite float
        Queried unobserved nodes satisfy x>threshold (strict inequality).
    target : float in (0,1)
        Observed conditional probability of at most one level in that domain.
    bracket : real array, (2,)
        Finite increasing nonnegative parameter endpoints. The user-supplied
        equation is continuous with a unique root in this bracket, possibly
        at an endpoint. Endpoint probability residual <=1e-13 counts as a root.
    tilt : finite float
        Linear coefficient in log w=-x^2/4-theta*x^4/150+tilt*x.

    Returns
    -------
    float
        Dimensionless theta at that root, with parameter accuracy 1e-9 or better.
        Use the exact finite beta=1 ensemble and condition on the full record.
        Positive-mass observations and well-conditioned SOPs throughout the
        bracket are caller preconditions. This inverse problem is a task-specific
        application of the source method, not a fitted parameter from the paper.

    Raises
    ------
    ValueError
        For invalid scalar/bracket inputs, a root not bracketed, or any invalid
        contract propagated from symplectic_arnoldi/condition_spectral_record.
    """
    return 0.0
```

### Step 9

spectral_tail_surprisal

Goal
----
Final orchestrator for inferred conditional second-largest-level statistics, using every preceding public function through the composed call chain. The synthetic inference and observation protocol are explicitly task-specific.

```python
import numpy as np

def spectral_tail_surprisal(nodes, count, record, calibration_cut, target, prediction_cut, bracket, tilt=0.07):
    """Assemble the inferred finite-ensemble joint-event surprisal.

    Parameters
    ----------
    nodes : real array, (m,)
        Distinct dimensionless nodes in supplied label order.
    count : even int, 4<=count<=m
        Fixed number of levels. The final assembly extends a count-2 SOP prefix
        by the last pair, using discrete_skew_metric and skew_project.
    record : integer array, (r,2)
        Original labels and occupancies as in condition_spectral_record.
    target : float in (0,1)
        Conditional at-most-one calibration probability.
    bracket : real array, (2,)
        Increasing nonnegative confinement bounds containing the unique root.
    tilt : finite float
        Linear coefficient of the weight, as in infer_confinement.
    calibration_cut : finite float
        Strict upper-domain boundary for inference.
    prediction_cut : finite float
        Strict upper-domain boundary for prediction, larger than calibration_cut.
        All recorded-present nodes must be <= both cuts, so the resulting event
        states that the second-largest level is <=prediction_cut.

    Returns
    -------
    float
        -ln P(record AND at most one level above prediction_cut), at the inferred
        theta; natural logarithm, dimensionless nats. Compose the preceding
        public functions on the submission path, including through their calls
        to earlier steps. This final orchestrator must use every earlier function.
        No rounding is fed back into the calculation.

    Raises
    ------
    ValueError
        For nonfinite/misordered cuts, a recorded-present node above either cut,
        a nonpositive computed joint probability, or an earlier contract violation.
    """
    return 0.0
```
