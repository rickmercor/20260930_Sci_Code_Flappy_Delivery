# Physics-Particle_Physics-26

## Background

This synthetic inverse experiment uses the attached source’s operator construction and its unintegrated NLO dipole-chain kernels. The finite quadrature defines the calculation before continuum angular integration. Natural units apply, with \(N_c=n_f=3\), \(\beta_0=0\), \(\mu^2=1.3\,\mathrm{GeV}^2\), and \(\Lambda=0.2\) GeV. The run log records a beam energy of 140 GeV and a reference tune \((Q_0,a)=(1.0\,\mathrm{GeV},0.3)\).

Transverse coordinates \(x_i\) are in \(\mathrm{GeV}^{-1}\); weights \(w_i\) are in \(\mathrm{GeV}^{-2}\).

| i | x | y | w |
|---|---:|---:|---:|
| 0 | -1.10 | -0.35 | 0.13 |
| 1 | -0.55 | 0.80 | 0.11 |
| 2 | 0.15 | -0.90 | 0.16 |
| 3 | 0.65 | 0.40 | 0.12 |
| 4 | 1.25 | -0.20 | 0.14 |
| 5 | -0.25 | -0.10 | 0.09 |
| 6 | 0.05 | 1.25 | 0.15 |

The measure \(\int d^2x/(2\pi)\) is represented by \(\sum_iw_i\), with \(d_{ij}=|x_i-x_j|^2\) and \(S_{ii}=1\). A single emitted label differs from its dipole’s endpoints; a simultaneous two-emission kernel uses four distinct labels. In operator products this restriction applies separately to each emission. Chain expectation values factorize into adjacent dipoles, including repeated nonadjacent labels; disconnected chains factorize between components. The physical initial state is
\[
S_{ij}(0)=\exp\!\left[-\frac{Q_0^2d_{ij}}4\ln\!\left(\frac1{\Lambda\sqrt{d_{ij}}}+e\right)\right],\qquad i\ne j.
\]
The truncated matching maps and the solution of the truncated evolution equation are composed numerically, retaining the products this composition generates. Rapidity \(Y\) is the elapsed evolution interval, and target parameters are common to all measurements.

A probe \(H=(Y,b,\epsilon)\), with \(b\) in \(\mathrm{GeV}^{-1}\), has matched response
\[
M(H)=\sum_{i<j}h_{ij}\,[1-S^{\mathrm{matched}}_{ij}(Y)],\quad
h_{ij}=\frac{u_{ij}}{\sum_{p<q}u_{pq}},\quad
u_{ij}=e^{-d_{ij}/(2b^2)}\!\left[d_{ij}+\epsilon\big((x_i^x-x_j^x)^2-(x_i^y-x_j^y)^2\big)\right].
\]
Here \(S^{\mathrm{matched}}\) denotes the evolved chain state contracted with the source’s matched coefficient function. Responses and their independent Gaussian measurement standard deviations are dimensionless.

| Y | b | epsilon | measured M | standard deviation |
|---:|---:|---:|---:|---:|
| 0.35 | 0.65 | 0.30 | 0.427358983 | 0.00030 |
| 1.10 | 1.25 | -0.40 | 0.578560082 | 0.00025 |
| 0.70 | 0.90 | 0.55 | 0.514189898 | 0.00020 |
| 1.60 | 0.75 | -0.20 | 0.483489813 | 0.00030 |

The prediction probe is \((Y_*,b_*,\epsilon_*)=(2.0,0.95,0.45)\). The tabulated standard deviations are absolute errors; the information covariance uses derivatives of the complete matched forward model.

## Problem

Determine the one-standard-deviation uncertainty of the prediction probe for the finite-quadrature QCD target specified below. The forward model is the attached paper’s dipole-chain scheme transformation, with its evolution generator retained through second order in \(a=\bar\alpha_s\) and its target and coefficient-function matching retained through first order. The target parameters \(Q_0\) and \(a\) minimize the four measurements’ absolute-error-weighted \(\chi^2\) on \([0.55,1.4]\,\mathrm{GeV}\times[0.08,0.32]\); the minimum is interior and identifiable. Use the local Gauss–Newton information covariance of these two fitted parameters for the prediction uncertainty. In the short reasoning, report \(Q_0\), \(a\), \(\chi^2\), the central prediction, and the parameter correlation, together with the operator consistency underlying the calculation and the source’s remaining logarithmic limitation. The final scalar has absolute tolerance \(5\times10^{-8}\); each of the five diagnostics has tolerance \(\max(2\times10^{-4}|v_{\mathrm{ref}}|,2\times10^{-6})\).

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

emission_kernels

Goal
----
Construct the single-emission kernels, including the collinear rotation.

```python
def emission_kernels(points: "np.ndarray", weights: "np.ndarray", mu2: float, nf: int = 3, nc: int = 3) -> "np.ndarray":
    """points : ndarray, shape (n,2), distinct transverse coordinates in GeV^-1, n >= 3.
    weights : ndarray, shape (n,), nonnegative weights for d^2x/(2*pi), in GeV^-2.
    mu2 : positive float, collinear scale squared in GeV^2.
    nf : nonnegative integer flavor count.
    nc : integer color count >= 2.

    Returns
    -------
    ndarray, shape (3,n,n,n), dimensionless; entries [channel,i,j,k].
    Channels are weighted K0, weighted L, and weighted one-emission K1, in this order.
    Inputs must be finite. Invalid dimensions, scales, or prescribed domains raise ValueError. 
    """
    return result
```

### Step 2

pair_kernels

Goal
----
Construct the connected gluon and fermion pair-emission kernels.

```python
def pair_kernels(points: "np.ndarray", weights: "np.ndarray", nf: int = 3, nc: int = 3) -> "np.ndarray":
    """points : ndarray, shape (n,2), distinct transverse coordinates in GeV^-1, n >= 3.
    weights : ndarray, shape (n,), nonnegative weights for d^2x/(2*pi), in GeV^-2.
    nf : nonnegative integer flavor count.
    nc : integer color count >= 2.

    Returns
    -------
    ndarray, shape (2,n,n,n,n), dimensionless; entries [channel,i,j,k,l].
    Channel 0 is the weighted gluon kernel; channel 1 is the weighted fermion kernel.
    Inputs must be finite. Invalid dimensions, scales, or prescribed domains raise ValueError. 
    """
    return result
```

### Step 3

dipole_fields

Goal
----
Contract the leading and collinear operators against the dipole-chain state.

```python
def dipole_fields(s: "np.ndarray", one: "np.ndarray") -> "np.ndarray":
    """s : ndarray, shape (n,n), dimensionless dipole values, row = first endpoint. Algebraic cases may be nonsymmetric.
    one : ndarray, shape (3,n,n,n), weighted single-emission kernels in emission_kernels order.
    
    Returns
    -------
    ndarray, shape (2,n,n), dimensionless, ordered [K0 action, L action].
    Inputs must be finite. Invalid dimensions, scales, or prescribed domains raise ValueError. 
    """
    return result
```

### Step 4

nlo_action

Goal
----
Contract the unrotated fixed-coupling NLO operator including disconnected color chains.

```python
def nlo_action(s: "np.ndarray", one: "np.ndarray", two: "np.ndarray") -> "np.ndarray":
    """s : ndarray, shape (n,n), dimensionless dipole values, row = first endpoint. Algebraic cases may be nonsymmetric.
    one : ndarray, shape (3,n,n,n), weighted single-emission kernels in emission_kernels order.
    two : ndarray, shape (2,n,n,n,n), weighted pair_kernels output.

    Returns
    -------
    ndarray, shape (n,n), dimensionless unrotated K1 action; no factor of a^2.
    Inputs must be finite. Invalid dimensions, scales, or prescribed domains raise ValueError. 
    """
    return result
```

### Step 5

scheme_commutator

Goal
----
Evaluate the scheme-changing operator commutator on the nonlinear dipole state.

```python
def scheme_commutator(s: "np.ndarray", one: "np.ndarray") -> "np.ndarray":
    """s : ndarray, shape (n,n), dimensionless dipole values, row = first endpoint. Algebraic cases may be nonsymmetric.
    one : ndarray, shape (3,n,n,n), weighted single-emission kernels in emission_kernels order.
    
    Returns
    -------
    ndarray, shape (n,n), dimensionless projected commutator [K0,L].
    Inputs must be finite. Invalid dimensions, scales, or prescribed domains raise ValueError. 
    """
    return result
```

### Step 6

composite_matching

Goal
----
Match the composite target or the dual hard coefficient through first order.

```python
def composite_matching(s: "np.ndarray", one: "np.ndarray", alpha: float, direction: int) -> "np.ndarray":
    """s : ndarray, shape (n,n), dimensionless dipole values, row = first endpoint. Algebraic cases may be nonsymmetric.
    one : ndarray, shape (3,n,n,n), weighted single-emission kernels in emission_kernels order.
    alpha : nonnegative float, a = alpha_s*nc/pi, dimensionless.
    direction : integer, -1 for target rotation or +1 for dual probe matching.

    Returns
    -------
    ndarray, shape (n,n), dimensionless matched dipole values.
    Inputs must be finite. Invalid dimensions, scales, or prescribed domains raise ValueError. 
    """
    return result
```

### Step 7

evolve_rotated_dipole

Goal
----
Evolve the composite dipole with the finite-regulator rotated NLO generator.

```python
def evolve_rotated_dipole(s: "np.ndarray", one: "np.ndarray", two: "np.ndarray", alpha: float, rapidity: float) -> "np.ndarray":
    """s : ndarray, shape (n,n), dimensionless dipole values, row = first endpoint. Algebraic cases may be nonsymmetric.
    one : ndarray, shape (3,n,n,n), weighted single-emission kernels in emission_kernels order.
    two : ndarray, shape (2,n,n,n,n), weighted pair_kernels output.
    alpha : nonnegative float, dimensionless fixed a.
    rapidity : nonnegative float, dimensionless evolution interval.

    Returns
    -------
    ndarray, shape (n,n), evolved composite dipole, dimensionless.
    Inputs must be finite. Invalid dimensions, scales, or prescribed domains raise ValueError. 
    """
    return result
```

### Step 8

infer_rotated_cgc

Goal
----
Infer the finite-quadrature QCD target and propagate its correlated uncertainty.

```python
def infer_rotated_cgc(points: "np.ndarray", weights: "np.ndarray", measurements: "np.ndarray", prediction: "np.ndarray", mu2: float = 1.3, nf: int = 3, nc: int = 3) -> "np.ndarray":
    """points : ndarray, shape (n,2), distinct transverse coordinates in GeV^-1, n >= 3.
    weights : ndarray, shape (n,), nonnegative weights for d^2x/(2*pi), in GeV^-2.
    measurements : ndarray, shape (m,5), m >= 3; columns [Y,b,epsilon,observed M,sigma].
    prediction : ndarray, shape (3,), [Y,b,epsilon].
    mu2 : positive float, collinear scale squared in GeV^2.
    nf : nonnegative integer flavor count.
    nc : integer color count >= 2.
    Probe domains are Y >= 0, b > 0, abs(epsilon) < 1, sigma > 0.
    Lambda=0.2 GeV; bounds are Q0 in [0.55,1.4] GeV, a in [0.08,0.32].

    Returns
    -------
    ndarray, shape (6,), ordered [Q0, a, chi2, predicted M, correlation rho, uncertainty delta_M].
    Only Q0 has units (GeV); all other entries are dimensionless.
    The six entries follow the task definitions and precision rules.
    Inputs must be finite. Invalid dimensions, scales, or prescribed domains raise ValueError. 
    """
    return result
```
