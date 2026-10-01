# Physics-Quantum_Information_Computing-17

## Background

Multilevel quantum systems can retain leakage population outside a computational subspace. Finite control pulses act in the presence of irreversible noise, so a physically reversed control experiment is not generally the inverse of an open-system channel.

Mid-circuit measurements make later controls depend on classical outcomes. Error mitigation in such circuits must respect that causal structure. Distinct noise mechanisms may drift independently, motivating local response measurements of a finite-order mitigation estimate.

## Problem

Noise mitigation for a dynamic quantum circuit must accommodate an irreversible measurement between coherent control blocks while responding to drift in distinct noise mechanisms. Apply the layerwise control-reversal mitigation construction for dynamic circuits published in March 2026, using its Taylor version and each complete three-pulse layer as a control unit; the inputs are the pulse schedule, two noise perturbations and measurement branches below, and the output is a mixed susceptibility of its residual bias fraction.

Two qutrits start in |00><00| and undergo six equal-duration layers, with a projective measurement of the first qutrit after the third layer and both outcomes retained without post-selection. Each layer contains the three listed pulses in temporal order A, B, C for equal thirds of its duration; layers 1 through 3 use the listed Hamiltonians, while layers 4 through 6 add H_F to every pulse only when the measurement yielded k=1. Let E_raw(x,y) be the exact noisy expectation of O and E_mit(x,y) its order-4 estimate from the specified layerwise protocol, while E_ideal is the expectation of the identical dynamic circuit with all noise rates zero; define R(x,y)=[E_mit(x,y)-E_ideal]/[E_raw(x,y)-E_ideal] and compute S=1000 times the mixed partial derivative d^4 R/(dx^2 dy^2) at x=y=0, not the coefficient multiplying x^2 y^2 in the power series.

Use hbar=1 and the basis |i,j> with i the first qutrit, with the same perturbed noise in every forward and pulse-inverse experiment and no readout error or noise during the instantaneous measurement; a listed jump C at rate g contributes g(C rho C^dagger - (C^dagger C rho + rho C^dagger C)/2) to the master equation. The derivatives refer to the analytic exact-expectation function with fixed controls and durations, no shot sampling and no finite-difference step prescribed, and the short reasoning should identify the inverse, cross-layer extrapolation, branch treatment and method used to differentiate the ratio, then report E_raw(0,0), E_ideal, E_mit(0,0) and the coefficient of x^2 y^2 in R as the determining scalars.
I = identity on one qutrit
X = |0><1| + |1><0|
Y = -i|0><1| + i|1><0|
Z = |0><0| - |1><1|
N = |1><1| + 2|2><2|
a = |0><1| + sqrt(2)|1><2|
ell = |2><1|
H_A = 1.3 X tensor X
H_B = 0.9 Z tensor I + 0.7 I tensor Y
H_C = 1.1 Y tensor Z + 0.4 N tensor I
H_F = 0.8 X tensor I
C_1, C_2 = a tensor I, I tensor a
g_1(x,y), g_2(x,y) = 0.06(1+x), 0.06(1+x)
C_3, C_4 = Z tensor I, I tensor Z
g_3(x,y), g_4(x,y) = 0.042(1+y), 0.042(1+y)
C_5, C_6 = ell tensor I, I tensor ell
g_5(x,y), g_6(x,y) = 0.03(1+y), 0.03(1+y)
x,y = independent dimensionless perturbations in a neighborhood of zero
P_0 = |0><0| tensor I
P_1 = (|1><1| + |2><2|) tensor I
measurement branch map = rho -> P_k rho P_k
O = |00><00|
total forward circuit duration = 1
number of layers = 6
Taylor mitigation order M = 4
Your final answer must be a single number: the value of S rounded to six digits after the decimal point.

For the numerical interfaces, the default configuration is noise_scale=0.06, feedforward=0.8, total_time=1.0, layers=6, order=4 and coefficient orders px=2, py=2. The coefficient orders specify the highest retained power of each independent perturbation.

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

01_qutrit_model

Goal
----
Construct a two-qutrit synthetic Markovian control model in the basis |i,j> with i first. On one qutrit define X=|0><1|+|1><0|, Y=-i|0><1|+i|1><0|, Z=diag(1,-1,0), N=diag(0,1,2), a=|0><1|+sqrt(2)|1><2| and ell=|2><1|. Return Hamiltonians [1.3 X tensor X, 0.9 Z tensor I+0.7 I tensor Y, 1.1 Y tensor Z+0.4 N tensor I], the feedforward Hamiltonian feedforward*X tensor I, jumps [a tensor I,I tensor a,Z tensor I,I tensor Z,ell tensor I,I tensor ell], rates noise_scale*[1,1,0.7,0.7,0.5,0.5], projectors [diag(1,0,0) tensor I,diag(0,1,1) tensor I], rho=|00><00| and observable=rho as a seven-tuple. Both arguments are finite nonboolean real scalars; the synthetic benchmark domain is noise_scale in [0,1] and feedforward in [-2,2]. Invalid inputs raise ValueError.

```python
def qutrit_model(noise_scale: float = 0.06, feedforward: float = 0.8) -> tuple:
    """Build the fixed operators with variable noise and feedback strengths.

    Parameters
    ----------
    noise_scale : float
        Finite nonboolean rate multiplier in [0,1].
    feedforward : float
        Finite nonboolean feedback amplitude in [-2,2].

    Returns
    -------
    tuple
        Hamiltonians, feedback, jumps, rates, projectors, density and observable.

    Raises
    ------
    ValueError
        If a scalar is not real, finite or in its stated domain.
    """
    return None


# EXPECTED RETURN LINE
# tuple of seven numeric arrays defining the two-qutrit model
```

### Step 2

02_lindblad_generators

Goal
----
Return separate coherent and common dissipative superoperators for the supplied Markovian pulse family.

```python
def lindblad_generators(hamiltonians: "np.ndarray", jumps: "np.ndarray", rates: "np.ndarray") -> tuple:
    """Build row-major coherent generators and their shared dissipator.

    Parameters
    ----------
    hamiltonians : array_like
        Hermitian stack (P,d,d) of finite bounded Hamiltonians.
    jumps : array_like
        Finite bounded jump stack (J,d,d), including an empty stack.
    rates : array_like
        Real nonnegative rates (J,) not exceeding 100.

    Returns
    -------
    tuple
        Coherent stack (P,d*d,d*d) and dissipator (d*d,d*d).

    Raises
    ------
    ValueError
        If dimensions fail (P>=1, J>=0, d>=2), arrays are not finite numeric
        with magnitudes<=100, rates are not real nonnegative or a Hamiltonian
        fails absolute Hermiticity tolerance 1e-10. An empty jump stack is
        (0,d,d). Nonfinite arithmetic also raises ValueError.
    """
    return None


# EXPECTED RETURN LINE
# tuple of complex arrays, coherent pulse generators and the common dissipator
```

### Step 3

03_pulse_channels

Goal
----
Return rectangular bivariate channel coefficients for a forward control layer and its physical inverse under the selected control-reversal construction.

```python
def pulse_channels(coherent: "np.ndarray", dissipative: "np.ndarray", direction_x: "np.ndarray", direction_y: "np.ndarray", durations: "np.ndarray", px: int = 2, py: int = 2) -> tuple:
    """Compose the bivariate response of forward and inverse control layers.

    Parameters
    ----------
    coherent : array_like
        Chronological coherent generators (L,n,n).
    dissipative : array_like
        Base noise generator (n,n).
    direction_x, direction_y : array_like
        Affine noise perturbation generators (n,n).
    durations : array_like
        Real chronological durations (L,) in [0,10].
    px, py : int
        Nonboolean integer coefficient orders in [0,2].

    Returns
    -------
    tuple
        Forward and physical inverse coefficient arrays (px+1,py+1,n,n).
        Entry [a,b] multiplies x^a*y^b, not its derivative. Both px and py
        default to 2. Required absolute and relative accuracy is 1e-9 in
        isolation; composition must meet the final residual tolerance.

    Raises
    ------
    ValueError
        If a declared shape fails, L<1, n is not d*d for an integer d>=2,
        orders are invalid, any array is nonnumeric/nonfinite or has
        magnitude>100, or a duration is not real in [0,10]. Also if a pulse
        duration times the sum of infinity norms of its coherent generator,
        base dissipator and two directions exceeds 100, or output is nonfinite.
    """
    return None


# EXPECTED RETURN LINE
# tuple of complex coefficient arrays for the forward and physical inverse layer
```

### Step 4

04_taylor_weights

Goal
----
Return the finite-order Taylor mitigation weights for the selected control-reversal construction.

```python
def taylor_weights(order: int) -> "np.ndarray":
    """Return the ordered odd-node Taylor extrapolation weights.

    Parameters
    ----------
    order : int
        Nonboolean integer type in [0,8]; floats are invalid.

    Returns
    -------
    numpy.ndarray
        Real vector of length order+1 in ascending amplification index.

    Raises
    ------
    ValueError
        If order is outside its type or range contract.
    """
    return None


# EXPECTED RETURN LINE
# numpy.ndarray, Taylor coefficients for settings zero through order
```

### Step 5

05_layer_amplification

Goal
----
Construct amplified-layer response coefficients from paired forward and physical-inverse layers for the selected finite-order control-reversal construction.

```python
def layer_amplification(forward: "np.ndarray", inverse: "np.ndarray", order: int) -> "np.ndarray":
    """Return amplified layer coefficients with one common setting axis.

    Parameters
    ----------
    forward, inverse : numpy.ndarray
        Paired layer series (L,P+1,Q+1,n,n), with row-major density action.
    order : int
        Nonboolean maximum amplification index in [0,8].

    Returns
    -------
    numpy.ndarray
        Complex amplified layers (order+1,L,P+1,Q+1,n,n), with ordinary
        power-series coefficients. P,Q lie in [0,2], L>=0 and n=d*d for
        an integer d>=2. Absolute and relative accuracy is 1e-9 in
        isolation; composition must meet the final residual tolerance.

    Raises
    ------
    ValueError
        If paired shapes or coefficient/matrix axes fail, order is not a
        nonboolean integer in [0,8], any input is nonnumeric/nonfinite or
        has magnitude>100, or an amplified coefficient is nonfinite.
    """
    return None


# EXPECTED RETURN LINE
# numpy.ndarray, amplified layer series with a common leading setting axis
```

### Step 6

06_dynamic_amplification

Goal
----
Compute exact observable-response coefficients for the dynamic circuit at every setting of the selected layerwise control-reversal construction.

```python
def dynamic_amplification(pre: "np.ndarray", pre_inverse: "np.ndarray", post: "np.ndarray", post_inverse: "np.ndarray", instrument: "np.ndarray", rho: "np.ndarray", observable: "np.ndarray", order: int) -> "np.ndarray":
    """Return unextrapolated dynamic-circuit expectation coefficients.

    Parameters
    ----------
    pre, pre_inverse : array_like
        Paired chronological layers (Lpre,P+1,Q+1,d*d,d*d).
    post, post_inverse : array_like
        Paired outcome-dependent layers (B,Lpost,P+1,Q+1,d*d,d*d).
    instrument : array_like
        Complete constant Kraus instrument (B,d,d), B>=1.
    rho : array_like
        Initial density matrix (d,d), d>=2, trace one within 1e-10,
        Hermitian within absolute tolerance 1e-10 and eigenvalues>=-1e-10.
    observable : array_like
        Hermitian observable (d,d), absolute Hermiticity tolerance 1e-10.
    order : int
        Nonboolean integer amplification order in [0,8].

    Returns
    -------
    numpy.ndarray
        Real array (order+1,P+1,Q+1), ordered by increasing amplification
        setting. Entry [j,a,b] multiplies x^a*y^b, not its derivative.
        P,Q are inferred from pre and lie in [0,2]. Empty temporal
        sequences preserve their stated shapes. Absolute and relative
        accuracy is 1e-9 in isolation. Composition must meet the final tolerance.

    Raises
    ------
    ValueError
        If shapes or domains fail, any input is nonnumeric/nonfinite or
        has magnitude>100, the instrument completeness error exceeds
        absolute elementwise tolerance 1e-10, or arithmetic is nonfinite.
        An expectation coefficient with imaginary magnitude>1e-9 is invalid.
    """
    return None

# EXPECTED RETURN LINE
# numpy.ndarray, unextrapolated bivariate expectations for each setting
```

### Step 7

07_bias_ratio

Goal
----
Return the ordinary bivariate power-series coefficients of the signed residual bias fraction defined in the global instance.

```python
def bias_ratio(expectations: "np.ndarray", weights: "np.ndarray", ideal: float) -> "np.ndarray":
    """Return power-series coefficients of the signed residual ratio.

    Parameters
    ----------
    expectations : array_like
        Real exact expectation coefficients (J,P+1,Q+1).
    weights : array_like
        Matching real extrapolation weights (J,).
    ideal : float
        Constant noise-free reference expectation.

    Returns
    -------
    numpy.ndarray
        Real ratio coefficient rectangle (P+1,Q+1), where [a,b] multiplies
        x^a*y^b. Require absolute and relative accuracy 1e-9 even under
        cancellation. Composition must meet the final residual tolerance.

    Raises
    ------
    ValueError
        If expectations is not (J,P+1,Q+1) with J in [1,9], P,Q in [0,2],
        weights is not (J,), or ideal is not a nonboolean real scalar.
        All inputs must be finite real with magnitudes<=100; weights sum
        to one within absolute tolerance 1e-10 and the unamplified constant
        expectation minus ideal must have magnitude>1e-10.
        Nonfinite output also raises ValueError.
    """
    return None


# EXPECTED RETURN LINE
# numpy.ndarray, bivariate power-series coefficient rectangle of the signed residual ratio
```

### Step 8

08_residual_bias

Goal
----
Return the scaled mixed noise susceptibility of the specified two-qutrit dynamic circuit by composing the seven upstream operations.

```python
def residual_bias(noise_scale: float = 0.06, total_time: float = 1.0, layers: int = 6, order: int = 4, feedforward: float = 0.8, px: int = 2, py: int = 2) -> float:
    """Return 1000 times the mixed derivative of the signed residual ratio.

    Parameters
    ----------
    noise_scale : float
        Finite nonboolean rate multiplier in [0.04,0.3], default 0.06.
    total_time : float
        Finite nonboolean forward duration in [0.6,2], default 1.0.
    layers : int
        Nonboolean even integer in [2,12], default 6.
    order : int
        Nonboolean integer Taylor mitigation order in [0,4], default 4.
    feedforward : float
        Finite nonboolean feedback amplitude in [-2,2], default 0.8.
    px, py : int
        Nonboolean derivative orders in [0,2], both default 2.

    Returns
    -------
    float
        Unrounded 1000*d^(px+py)R/(dx^px dy^py) at zero, including the
        ordinary-series factorial conversion. Required absolute and
        relative accuracy is 1e-8. Passing upstream isolated tolerances
        does not establish this end-to-end accuracy.

    Raises
    ------
    ValueError
        If a domain fails, a raw bias has magnitude<=1e-10 or an upstream
        validity/finite-arithmetic condition fails. Floats are invalid
        integer arguments.
    """
    return None

# EXPECTED RETURN LINE
# float, unrounded scaled mixed susceptibility of the signed residual ratio
```
