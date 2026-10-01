# Material_Science-Semiconductor_Materials-52

## Background

A constructed phase-resolved experiment on a dilute CsPbI3 nanocrystal ensemble is described in ps, meV and dimensionless field amplitudes, with hbar=0.6582119569 meV ps. Two independent phonon mode classes have energies (3.2,5.1) meV and coherent amplitude weights (0.63,0.37), in that order. The actual zero-phonon population-decay rate is gamma0=1/310 ps^-1. The optical carrier phase convention is exp(+i*phi) for excitation.

The calibration data obey the main paper's small-Huang-Rhys echo-amplitude law Psi0, including its written exponentials. Sigma(t)=sum_i w_i Psi0(t;Omega_i,S_i,gamma0,gamma_ph_i). The unweighted least-squares fit is over (S1,S2,gamma_ph1,gamma_ph2), with lower bounds (0.01,0.005,0.04,0.04) and upper bounds (0.18,0.12,0.35,0.35). Round the four converged fitted coordinates to seven decimal places before prediction. This instance has a unique best fit within numerical tolerance. gamma_ph denotes a population-decay rate, which sets the physical phonon lifetime.

The detector records signed parallel and cross heterodyne amplitudes Y=M(A_parallel,A_cross)^T. The independently characterized fine-structure splittings are (0.25,0.55,0.80) meV, with dispersion dephasing times (21,17,13) ps; these nuisance beats follow the main paper's isotropic-ensemble polarization model. The detector matrix, times, and measured rows are:

```python
M = [[1.0, 0.075], [-0.035, 0.91]]
times = [0.17, 0.39, 0.68, 0.94, 1.21, 1.58, 1.93, 2.37, 2.86, 3.41, 4.13, 5.07, 6.24, 7.81, 9.68, 12.03, 15.29, 19.37]
Y = [[2.375653279, 1.7582586482, 1.7035211663, 1.7225594908, 1.9080361027, 1.5395260639, 1.0925985855, 1.2827918215, 1.0345356415, 0.9666848089, 1.2239222733, 1.3924916511, 1.2512701821, 1.2274585368, 1.1048938415, 1.0960834178, 1.3421942552, 1.2251986651], [-0.0693478451, -0.0276750754, 0.0207849291, 0.0812233679, 0.1776811678, 0.2618057848, 0.2747511869, 0.4449630364, 0.4247748017, 0.3941245946, 0.4120523666, 0.3671592284, 0.3403742145, 0.4030103264, 0.3380416278, 0.3464142758, 0.2605912086, 0.3240218881]]
```

For prediction, use the source four-level, impulsive, population-transfer model before the final small-S amplitude expansion. Within each mode class, the basis is (|0>,|0'>,|X>,|X'>), and H0/hbar=diag(0,Omega,Delta,Delta+Omega). Optical population rates are fixed by the source overlap ratios relative to the given gamma0, retaining (1-S)^2. Each of the four optical transfers and two phonon-emission transfers is a separate Lindblad jump, as in the source's diagonal inflow. The pure-dephasing rate is zero.

The optical raising matrix R has its (X,X') by (0,0') block equal to [[1,sqrt(S)],[sqrt(S),1]], with all other entries zero, following the source's pulse-vertex convention. An impulse of area theta and phase phi has unitary exp[-i theta(exp(i phi)R+exp(-i phi)R†)]. Start in |0><0|. Pulse 1 is at time 0 and pulse 2 at time tau. G_i(tau,u;Delta) is the coefficient of theta1*theta2^2*exp[i(2phi2-phi1)] in the emitted polarization rho[X,0]+rho[X',0']+sqrt(S)(rho[X',0]+rho[X,0']), at time u after pulse 2; Taylor coefficients include their factorials. All powers of S generated inside this four-level response are retained. The definition is a constructed finite model specialization.

The prediction has tau=6.4 ps, an after-pulse-2 gate u in [6.18,6.92] ps, optical angular-frequency detunings Delta=(-1.4,-0.25,0.6,1.9) ps^-1, and coherent detuning weights q=(0.18,0.37,0.29,0.16). Define E(u)=sum_i,j w_i q_j G_i(tau,u;Delta_j). The reference E0 uses S1=S2=0 with all other inputs unchanged. The requested scalar is R_gate=integral_gate |E(u)|^2 du / integral_gate |E0(u)|^2 du, evaluated with 12-point Gauss-Legendre quadrature. The center-field ratio is C=E(6.55)/E0(6.55). Equivalent converged integrations receive credit within tolerance. Matrix vectorization in code interfaces is column-major. Fit and echo accuracy of 2e-6 absolute is sufficient for every reported real scalar.

## Problem

Determine the gated rephasing-power ratio R_gate for the exciton–polaron experiment specified in the scientific background. Infer the two Huang–Rhys factors and two phonon population-decay rates from the supplied polarization-mixed calibration amplitudes using the source's small-coupling calibration law. The prediction is the complex four-level response at the stated off-echo gate, with the source's optical pathways and population-transfer dynamics. Report R_gate as the final scalar, to six decimal places. In the reasoning give the four fitted coordinates in the stated order and the real and imaginary parts of the center-field ratio C, each to six decimal places with absolute tolerance 2e-6. Explain briefly the source-specific polarization cancellation, the origin of the double-frequency beat, and the relation between population decay and optical coherence in this calculation.

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

polaron_optical_rates

Goal
----
Compute the three optical population-decay rates.

```python
import numpy as np

def polaron_optical_rates(huang_rhys: float, gamma0: float) -> np.ndarray:
    """Compute the three optical population-decay rates.

    Parameters
    ----------
    huang_rhys : float in [0,1]
        Dimensionless S for the finite four-level model.
    gamma0 : nonnegative float
        Zero-phonon population-decay rate in ps^-1.

    Returns
    -------
    real ndarray, shape (3,), ordered [gamma0 (X→0), gamma_prime (each of X→0′ and X′→0), gamma1 (X′→0′)], in ps^-1.

    Raises
    ------
    ValueError
        If inputs fall outside the documented numerical domain.
    """
    return result
```

### Step 2

polaron_free_generator

Goal
----
Build the field-free Lindblad generator.

```python
import numpy as np

def polaron_free_generator(omega: float, detuning: float, rates: np.ndarray, gamma_ph: float) -> np.ndarray:
    """Build the field-free Lindblad generator.

    Parameters
    ----------
    omega : nonnegative float
        Phonon angular frequency in ps^-1.
    detuning : real float
        Zero-phonon optical angular-frequency detuning in ps^-1.
    rates : nonnegative real ndarray, shape (3,)
        [gamma0, gamma_prime, gamma1]: population rates of X→0, of each of X→0′ and X′→0, and of X′→0′, in ps^-1.
    gamma_ph : nonnegative float
        Phonon population-decay rate in ps^-1.

    Returns
    -------
    complex ndarray, shape (16,16), acting on vec_F(rho); entries are in ps^-1.

    Raises
    ------
    ValueError
        If inputs fall outside the documented numerical domain.
    """
    return result
```

### Step 3

polaron_pulse_coefficient

Goal
----
Obtain one pulse-area coefficient of the impulsive density map.

```python
import numpy as np

def polaron_pulse_coefficient(huang_rhys: float, phase: float, order: int) -> np.ndarray:
    """Obtain one pulse-area coefficient of the impulsive density map.

    Parameters
    ----------
    huang_rhys : float in [0,1]
        Dimensionless S.
    phase : real float
        Optical carrier phase in radians.
    order : integer in {0,1,2}
        Power of dimensionless pulse area in the Taylor series.

    Returns
    -------
    complex ndarray, shape (16,16), dimensionless pulse coefficient acting on vec_F(rho).

    Raises
    ------
    ValueError
        If inputs fall outside the documented numerical domain.
    """
    return result
```

### Step 4

single_mode_rephasing_echo

Goal
----
Compute the complex rephasing pathway coefficient for one phonon mode.

```python
import numpy as np

def single_mode_rephasing_echo(huang_rhys: float, omega: float, gamma0: float, gamma_ph: float, tau: float, u: float, detuning: float) -> complex:
    """Compute the complex rephasing pathway coefficient for one phonon mode.

    Parameters
    ----------
    huang_rhys : float in [0,1]
        Dimensionless S.
    omega : nonnegative float
        Phonon angular frequency in ps^-1.
    gamma0, gamma_ph : nonnegative floats
        Optical zero-phonon and phonon population-decay rates in ps^-1.
    tau, u : nonnegative floats
        Pulse separation and time after pulse 2, in ps.
    detuning : real float
        Zero-phonon optical angular-frequency detuning in ps^-1.

    Returns
    -------
    complex scalar, dimensionless coefficient of theta1*theta2^2*exp(i*(2*phase2-phase1)).

    Raises
    ------
    ValueError
        If inputs fall outside the documented numerical domain.
    """
    return 0j
```

### Step 5

weak_polaron_echo

Goal
----
Evaluate the source amplitude kernel used by calibration.

```python
import numpy as np

def weak_polaron_echo(times: np.ndarray, omega: float, huang_rhys: float, gamma0: float, gamma_ph: float) -> np.ndarray:
    """Evaluate the source amplitude kernel used by calibration.

    Parameters
    ----------
    times : nonnegative real ndarray, shape (T,)
        Pulse separations in ps.
    omega : nonnegative float
        Phonon angular frequency in ps^-1.
    huang_rhys : float in [0,0.2]
        Dimensionless S in the small-coupling calibration model.
    gamma0, gamma_ph : nonnegative floats
        Optical zero-phonon and phonon population-decay rates in ps^-1.

    Returns
    -------
    real ndarray, shape (T,), dimensionless Psi0 amplitude in input time order.

    Raises
    ------
    ValueError
        If inputs fall outside the documented numerical domain.
    """
    return result
```

### Step 6

recover_intrinsic_echo

Goal
----
Recover the polarization-invariant amplitude and fine-structure contrast.

```python
import numpy as np

def recover_intrinsic_echo(channels: np.ndarray, mixing: np.ndarray) -> np.ndarray:
    """Recover the polarization-invariant amplitude and fine-structure contrast.

    Parameters
    ----------
    channels : real ndarray, shape (2,T)
        Measured dimensionless heterodyne field amplitudes; row order parallel, cross.
    mixing : real ndarray, shape (2,2)
        Known nonsingular dimensionless detector response, measured=mixing@physical.

    Returns
    -------
    real ndarray, shape (2,T), rows [Sigma, polarization contrast rho], both dimensionless.

    Raises
    ------
    ValueError
        If inputs fall outside the documented numerical domain.
    """
    return result
```

### Step 7

fit_polaron_bath

Goal
----
Infer the two coupling strengths and phonon decay rates.

```python
import numpy as np

def fit_polaron_bath(times: np.ndarray, channels: np.ndarray, mixing: np.ndarray, omega: np.ndarray, weights: np.ndarray, gamma0: float) -> np.ndarray:
    """Infer the two coupling strengths and phonon decay rates.

    Parameters
    ----------
    times : nonnegative real ndarray, shape (T,)
        Calibration pulse separations in ps.
    channels : real ndarray, shape (2,T)
        Measured dimensionless parallel and cross amplitudes.
    mixing : nonsingular real ndarray, shape (2,2)
        Dimensionless detector response.
    omega : positive real ndarray, shape (2,)
        Distinct known phonon angular frequencies in ps^-1, in mode order.
    weights : positive real ndarray, shape (2,)
        Known mode amplitude weights, summing to one.
    gamma0 : positive float
        Known zero-phonon optical population-decay rate in ps^-1.

    Returns
    -------
    real ndarray, shape (4,), ordered [S1,S2,gamma_ph1,gamma_ph2]; first two dimensionless, last two ps^-1.

    Raises
    ------
    ValueError
        If inputs fall outside the documented numerical domain.
    """
    return result
```

### Step 8

predict_polaron_echo

Goal
----
Infer the bath and predict the gated rephasing power ratio.

```python
import numpy as np

def predict_polaron_echo(times: np.ndarray, channels: np.ndarray, mixing: np.ndarray, omega: np.ndarray, weights: np.ndarray, gamma0: float, tau: float, gate: np.ndarray, detunings: np.ndarray, detuning_weights: np.ndarray, diagnostic: int=0) -> float:
    """Infer the bath and predict the gated rephasing power ratio.

    Parameters
    ----------
    times : nonnegative real ndarray, shape (T,)
        Calibration pulse separations in ps.
    channels : real ndarray, shape (2,T)
        Measured dimensionless parallel and cross amplitudes.
    mixing : nonsingular real ndarray, shape (2,2)
        Dimensionless detector response.
    omega : positive real ndarray, shape (2,)
        Distinct known phonon angular frequencies in ps^-1, in mode order.
    weights : positive real ndarray, shape (2,)
        Known mode amplitude weights, summing to one.
    gamma0 : positive float
        Known zero-phonon optical population-decay rate in ps^-1.
    tau : nonnegative float
        Prediction pulse separation in ps.
    gate : real ndarray, shape (2,)
        Increasing nonnegative [u_min,u_max] in ps after pulse 2.
    detunings : real ndarray, shape (D,)
        Zero-phonon optical angular-frequency detunings in ps^-1.
    detuning_weights : nonnegative real ndarray, shape (D,)
        Coherent ensemble amplitude weights, summing to one.
    diagnostic : integer in {0,1,2,3,4,5,6}
        Selector [power ratio,S1,S2,gamma_ph1,gamma_ph2,Re center ratio,Im center ratio].
        Only entries 3 and 4 have units ps^-1; all others are dimensionless.

    Returns
    -------
    float scalar, the selected diagnostic with units stated above.

    Raises
    ------
    ValueError
        If inputs fall outside the documented numerical domain.
    """
    return 0.0
```
