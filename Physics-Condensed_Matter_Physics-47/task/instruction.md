# Physics-Condensed_Matter_Physics-47

## Background

Spatial entanglement in a many-body system contains information that a single global measure cannot resolve. For mixed states of coupled harmonic oscillators, logarithmic negativity quantifies quantum correlations across a partition, while a spatial contour assigns contributions to individual sites. A system maintained away from thermal equilibrium can carry energy through correlations between positions and momenta. Understanding the spatial redistribution of entanglement therefore requires a local description that retains these correlations and distinguishes changes in the amount of entanglement from changes in where it is concentrated. Gaussian covariance methods make this question accessible without representing the full many-body density operator.

## Problem

Consider an infinite harmonic chain with $H=\tfrac12\sum_{j\in\mathbb Z}[p_j^2+\mu^2q_j^2+(q_{j+1}-q_j)^2]$, $\mu=0.45$, and $\hbar$, oscillator mass, spring stiffness and lattice spacing equal to one.
Initially the two disconnected half-chains $j\leq-1$ and $j\geq0$ have inverse temperatures $\beta_L=\bar\beta-\delta$ and $\beta_R=\bar\beta+\delta$, with $\bar\beta=3.2$; join them and use the locally stationary state obtained by taking the thermodynamic limit before the long-time limit.
Retain the ordered sites $s=(-3,-2,-1,0,1,2)$, partitioned into $A_1=(-3,-2,-1)$ and $A_2=(0,1,2)$, and use the grouped quadratures $z=(q_{-3},q_{-2},q_{-1},q_0,q_1,q_2,p_{-3},p_{-2},p_{-1},p_0,p_1,p_2)$ with covariance $\Gamma_{ab}=\langle\{z_a,z_b\}\rangle/2$.

For this benchmark, define each real-space covariance block by the fixed sum $\Gamma^{ab}_{ij}=128^{-1}\sum_{k=0}^{127}e^{+iq_k(s_i-s_j)}\widehat\gamma_{ab}(q_k;\beta_L,\beta_R)$, where $\widehat\gamma$ is the stationary half-anticommutator covariance symbol of the joined chain, $q_k=-\pi+(k+\tfrac12)2\pi/128$, and $\omega(q)=\sqrt{\mu^2+4\sin^2(q/2)}$; this finite sum defines the requested observable and its grid remains fixed during differentiation.
Use the positive bosonic Gaussian logarithmic-negativity contour introduced in 2026 through the orthogonal symplectic mode participation of a Williamson frame after removing its symplectic stretching, with partial transpose on $A_2$, natural logarithms and vacuum symplectic eigenvalue $1/2$.
Writing its site contributions as $\mathsf E_i(\delta)$, define the normalized squared width about the joining bond as $\mathcal W(\delta)=\sum_{i=0}^{5}(s_i+\tfrac12)^2\mathsf E_i(\delta)/\sum_{i=0}^{5}\mathsf E_i(\delta)$.

Determine the dimensionless right derivative $d_+\mathcal W/d\delta$ at $\delta=0.9$, keeping all other specified inputs fixed, to absolute accuracy $5\times10^{-8}$.
Obtain the contour response analytically in a basis-independent positive-matrix formulation, including the right-directional rule for a possibly degenerate spectral threshold, using principal positive matrix functions and an absolute threshold tolerance of $10^{-10}$.
In the reasoning, characterize the stationary mode populations, the covariance symbol and the spatial contour law, and give the decisive scalar checks supporting the differentiated normalized width.

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

build_ness_symbols

Goal
----
Construct the three real Fourier-symbol channels and their analytic bias derivatives on the specified midpoint grid.

```python
def build_ness_symbols(mass: float, coupling: float, beta_mean: float,
                       bias: float, nq: int) -> "np.ndarray":
    """Evaluate the covariance symbols and their first derivative in bias.

    Parameters
    ----------
    mass : float
        Positive finite mass parameter, with oscillator masses and hbar set to one.
    coupling : float
        Positive finite nearest-neighbor spring coupling.
    beta_mean : float
        Finite mean inverse temperature, strictly greater than abs(bias).
    bias : float
        Finite reservoir inverse-temperature bias. Positive bias makes the left reservoir hotter. The derivative is with respect to this value.
    nq : int
        Even integer at least eight, specifying the midpoint momentum grid. Boolean values are invalid.

    Returns
    -------
    symbols : np.ndarray
        Real array of shape (2, nq, 3). Axis zero contains the value and bias derivative, respectively; the final axis is (qq, pp, mixed_imag). Use stable thermal factors at both small and large beta times omega.

    Raises
    ------
    ValueError
        If any input violates its scalar domain, nq is invalid, or the result cannot be represented by finite double-precision numbers.
    """
    return symbols
```

### Step 2

fourier_covariance

Goal
----
Fourier-transform both symbol slices and restrict the resulting covariance jet to the ordered site set.

```python
def fourier_covariance(symbols: "np.ndarray", positions: "np.ndarray") -> "np.ndarray":
    """Fourier transform a symbol jet onto an ordered selection of lattice sites.

    Parameters
    ----------
    symbols : np.ndarray
        Finite real array of shape (2, nq, 3), with even nq at least eight. Axes contain (value, bias derivative), midpoint momentum, and (qq, pp, mixed_imag), respectively. The qq and pp value channels must be strictly positive; derivative channels may have either sign.
    positions : np.ndarray
        Nonempty one-dimensional array of distinct finite integer-valued site coordinates, in the desired output order. Their span must be strictly less than nq. Boolean entries are invalid.

    Returns
    -------
    covariance_jet : np.ndarray
        Finite real array of shape (2, 2*m, 2*m), where m is the number of positions. The first slice is the symmetrized covariance and the second is its bias derivative, both in grouped (all q, all p) order. No input is modified.

    Raises
    ------
    ValueError
        If symbols or positions violate the stated shape, finiteness, positivity, integer-coordinate, uniqueness or span conditions, or if the resulting covariance is not representable as finite floats.
    """
    return covariance_jet
```

### Step 3

partial_transpose_jet

Goal
----
Apply momentum sign reversal on the selected local sites to both covariance slices.

```python
def partial_transpose_jet(covariance_jet: "np.ndarray",
                          transpose_sites: "np.ndarray") -> "np.ndarray":
    """Apply a subsystem partial transpose to the value and bias derivative.

    Parameters
    ----------
    covariance_jet : np.ndarray
        Finite real array of shape (2, 2*m, 2*m), with m at least one, in grouped (all q, all p) order. Each slice must be symmetric within absolute tolerance 1e-10 and zero relative tolerance. The symmetric part of the value slice must be strictly positive definite. No quantum uncertainty-condition test is required.
    transpose_sites : np.ndarray
        One-dimensional array of distinct finite integer-valued local site indices in [0, m). Boolean entries are invalid. An empty array and the set of all local sites are both valid.

    Returns
    -------
    transposed_jet : np.ndarray
        Real array of the same shape, with momentum signs reversed on the selected sites in both the covariance and its derivative. The output has independent storage and neither input is modified.

    Raises
    ------
    ValueError
        If the covariance shape, realness, finiteness, symmetry or positive definiteness is invalid, or if the site indices violate their domain.
    """
    return transposed_jet
```

### Step 4

symplectic_absolute_jet

Goal
----
Construct the covariance square root, the positive absolute matrix of its symplectic congruence, and both directional derivatives.

```python
def symplectic_absolute_jet(covariance_jet: "np.ndarray") -> "np.ndarray":
    """Return the square-root and symplectic-absolute value/tangent pairs.

    Parameters
    ----------
    covariance_jet : np.ndarray
        Real array of shape (2, 2*m, 2*m), m >= 1, containing a positive definite covariance G and a symmetric direction dG. Coordinates are grouped as all positions followed by all momenta. Symmetry is checked with absolute tolerance 1e-10 and zero relative tolerance.

    Returns
    -------
    absolute_jet : np.ndarray
        Real array of shape (4, 2*m, 2*m), ordered C, dC, B, dB, where C is the principal positive square root of G and B is the positive absolute value of C J C, with J = [[0,I],[-I,0]]. The directions are exact first variations along G + epsilon*dG at epsilon=0. All four matrices are symmetric. Repeated eigenvalues are valid. Caller inputs are not mutated.

    Raises
    ------
    ValueError
        If the input is not finite and real, has the wrong shape, contains nonsymmetric matrices, or G is not positive definite.
    """
    return absolute_jet
```

### Step 5

balance_contour_jet

Goal
----
Remove the Williamson polar metric and propagate its variation to obtain the symmetric matrix defining the spatial contour.

```python
def balance_contour_jet(covariance_jet: "np.ndarray", absolute_jet: "np.ndarray") -> "np.ndarray":
    """Construct the passive-polar contour matrix and its first variation.

    Parameters
    ----------
    covariance_jet : np.ndarray
        Real symmetric array (2, 2*m, 2*m), containing positive definite G and its direction dG in grouped position/momentum coordinates.
    absolute_jet : np.ndarray
        Real symmetric array (4, 2*m, 2*m), containing C, dC, B, dB from symplectic_absolute_jet for this same covariance and direction. C and B must be positive definite. The caller supplies a consistent pair; consistency with the defining identities is a precondition.

    Returns
    -------
    contour_jet : np.ndarray
        Real symmetric array (2, 2*m, 2*m), containing Phi and dPhi. Phi = K.T (D direct-sum D) K, where G = W.T (D direct-sum D) W is any Williamson decomposition and K is W's orthogonal polar factor. The output must be independent of the mode basis, including repeated symplectic eigenvalues. dPhi is the exact first variation. Caller inputs are not mutated.

    Raises
    ------
    ValueError
        If inputs are not finite real symmetric arrays with the stated matching shapes, or G, C or B is not positive definite. Symmetry uses absolute tolerance 1e-10 and zero relative tolerance.
    """
    return contour_jet
```

### Step 6

active_log_jet

Goal
----
Evaluate the active logarithm and its right directional derivative, treating a degenerate threshold subspace jointly.

```python
def active_log_jet(contour_jet: "np.ndarray") -> "np.ndarray":
    """Apply the negativity filter and its right directional derivative.

    Parameters
    ----------
    contour_jet : np.ndarray
        Real finite array of shape (2, 2*m, 2*m), with m >= 1. The slices are a symmetric positive definite Phi and a symmetric direction H. Symmetry uses absolute tolerance 1e-10. The filter is f(x) = max(0, -log(2*x)), with the natural logarithm. Eigenvalues within absolute tolerance 1e-10 of 0.5 are treated as exactly 0.5 in both slices; this is the numerical threshold convention. Outside that band use ordinary spectral divided differences. Within the full threshold eigenspace use the positive semidefinite part of -2 times the restriction of H. Inputs are not mutated.

    Returns
    -------
    log_jet : np.ndarray
        Real array of shape (2, 2*m, 2*m). Slice 0 is f(Phi); slice 1 is its right directional derivative under the threshold convention.

    Raises
    ------
    ValueError
        If the input is complex, nonfinite, has a wrong shape or nonsymmetric slices, or Phi is not positive definite.
    """
    return log_jet
```

### Step 7

negativity_width_response

Goal
----
Form the site contour and differentiate its normalized second spatial moment about a fixed center.

```python
def negativity_width_response(log_jet: "np.ndarray", positions: "np.ndarray", center: float) -> float:
    """Differentiate the second moment of the normalized site contour.

    Parameters
    ----------
    log_jet : np.ndarray
        Real finite symmetric array of shape (2, 2*m, 2*m), m >= 1, with symmetry absolute tolerance 1e-10. The two slices contain the active logarithm L and its right derivative dL, in all-q-then-all-p order. Site weights are e_i = (L_ii + L_(m+i,m+i))/2, and their derivatives use dL in the same way. The total weight must exceed 1e-14. A scientifically valid active-log jet is a precondition; positivity of each separate site weight is not checked.
    positions : np.ndarray
        Real finite one-dimensional array of m distinct integer-valued site positions, in the same order as the canonical coordinates. Boolean entries are invalid.
    center : float
        Finite real scalar specifying the fixed spatial origin; booleans are invalid. It does not vary with the differentiation parameter. Caller inputs are not mutated.

    Returns
    -------
    response : float
        Native Python float equal to the right derivative of sum((positions-center)**2 * e) / sum(e).

    Raises
    ------
    ValueError
        If inputs are complex or nonfinite, arrays have invalid shapes, log_jet is nonsymmetric, positions are not distinct integer-valued sites, center is not a finite real scalar, total weight <= 1e-14, or the spatial response cannot be represented as a finite float.
    """
    return response
```

### Step 8

compute_ness_negativity_response

Goal
----
Call build_ness_symbols, fourier_covariance, partial_transpose_jet, symplectic_absolute_jet, balance_contour_jet, active_log_jet, and negativity_width_response in that order; supply both transposed covariance and absolute jets to balance_contour_jet.

```python
def compute_ness_negativity_response(mass: float, coupling: float, beta_mean: float, bias: float, nq: int, positions: "np.ndarray", transpose_sites: "np.ndarray", center: float) -> float:
    """Compose the preceding seven public functions to evaluate the response.

    Parameters
    ----------
    mass : float
        Finite strictly positive oscillator frequency parameter.
    coupling : float
        Finite strictly positive nearest-neighbor spring constant.
    beta_mean : float
        Finite positive mean inverse temperature b.
    bias : float
        Finite inverse-temperature bias delta, satisfying abs(delta) < b. The left and right inverse temperatures are b-delta and b+delta. Differentiate with respect to delta holding every other input fixed.
    nq : int
        Even integer >= 8, defining q_l=-pi+2*pi*(l+0.5)/nq. The finite midpoint quadrature defines the observable exactly.
    positions : np.ndarray
        Real finite one-dimensional array of distinct integer-valued lattice sites, with at least one site and span strictly smaller than nq; boolean entries are invalid.
    transpose_sites : np.ndarray
        One-dimensional real array of distinct integer-valued local indices into positions, all between 0 and len(positions)-1. It may be empty. Boolean entries are invalid. Reverse only their momentum coordinates.
    center : float
        Fixed finite real scalar origin for the normalized second moment. Boolean scalar parameters are invalid. Use the preceding public functions in their listed order, keeping all-q-then-all-p ordering and the active-log threshold convention. Inputs are not mutated.

    Returns
    -------
    response : float
        Native Python float: the right derivative of the normalized spatial second moment of the negativity contour. The total negativity must exceed 1e-14. Compute analytic directional jets through the pipeline; finite differences are reserved for independent verification.

    Raises
    ------
    ValueError
        If any parameter violates its stated contract, a required positive definite matrix is invalid, a thermal symbol is nonfinite in floating point, or the total negativity is at most 1e-14. The numerical symmetry and threshold tolerances are inherited from preceding steps.
    """
    return response
```
