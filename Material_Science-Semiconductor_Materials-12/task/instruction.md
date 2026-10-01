# Material_Science-Semiconductor_Materials-12

## Background

Single-electron pumps move a controlled number of electrons per cycle and are the basis of proposals for a quantum realisation of the ampere. Most of them work by modulating a quantum-dot level or its tunnel barriers with gate voltages. A different way to pump is to leave the level where it is and only switch which lead the dot is connected to: an electron picked up from one lead is released into the other, and when the second lead sits at the higher chemical potential the charge is moved against the bias. The energy for this comes from the work done each time a tunnel coupling is switched on or off, so such a device is a small engine that converts switching work into chemical energy.

In the textbook picture of slow switching, the dot relaxes to the stationary state with each lead before the next switch, the only memory of a step is the dot occupation, and the energy efficiency of the conversion is bounded by one half. Real leads are not featureless. When the tunnelling spectrum of a lead has structure on the energy scale of the dot level and the bias, the dot and the lead build up coherences that do not vanish when the coupling is switched off, and they decay on the time scale set by the width of the spectral feature. If the next switch comes before they have died out, they change both the charge transferred and the work done at the switch, and the pump performance oscillates with the step durations.

A spinless level coupled to noninteracting leads is exactly solvable at any coupling strength, because all its dynamics is carried by single-particle correlation functions. Recent exact calculations for leads with Lorentzian spectra showed that finite step durations can push the efficiency above the slow-switching bound. How high the efficiency of a given device can be pushed, where the optimum lies, and how the pumped current compares with metrological pumps are the questions that decide whether coherence is a useful resource for such engines.

## Problem

A single spinless quantum-dot level can pump electrons against a voltage bias without any gate modulation of the level itself. The dot is connected to the lead at the lower chemical potential, then switched over to the lead at the higher chemical potential, then back, and so on. The energy that drives the pump is the work done each time the tunnel couplings are switched. When the leads have structured spectra and the strokes are short, the dot keeps a coherence with a lead after it has been disconnected, and that coherence feeds back into the next switch. The task is to find how efficiently such a pump can turn switching work into chemical energy when only the duration of the second stroke is tuned.

Model the device by the Hamiltonian H(t) = eps d^+ d + sum over nu = 1, 2 and k of e_{nu k} c_{nu k}^+ c_{nu k} + sum over nu of g_nu(t) H_nu, with H_nu = sum over k of (t_{nu k} c_{nu k}^+ d + t_{nu k}^* d^+ c_{nu k}), and use units with hbar = e = 1 in which the bias is V = 1. The dot level is eps = 1.2. The leads are noninteracting and start in their zero-temperature ground states with chemical potentials mu_1 = 0 and mu_2 = V. Their tunnelling spectra are Lorentzian, J_nu(omega) = 2 pi sum over k of |t_{nu k}|^2 delta(omega - e_{nu k}) = Gamma_nu lambda_nu^2 / [(omega - omega_nu)^2 + lambda_nu^2], with omega_1 = 0, lambda_1 = 0.1, Gamma_1 = 2.5 for lead 1 and omega_2 = 1.2, lambda_2 = 0.25, Gamma_2 = 0.1 for lead 2. The pump repeats a cycle of two strokes with instantaneous switching: during stroke 1, of duration t1 = 1.128, g_1 = 1 and g_2 = 0; during stroke 2, of duration t2, g_1 = 0 and g_2 = 1. All quantities refer to the periodic state that the device reaches after infinitely many cycles, which does not depend on how it was prepared. N_pump is the mean number of electrons transferred into lead 2 per cycle. The work done at a switching instant is the jump of the expectation value of the coupling Hamiltonian g_1(t) H_1 + g_2(t) H_2 across that instant, W_a at the switch from stroke 1 to stroke 2 and W_b at the switch from stroke 2 back to stroke 1, and the energy efficiency is eta = N_pump V / (W_a + W_b).

Find the largest value eta* of eta over 1 <= t2 <= 12 and the duration t2* at which it occurs, with the leads treated exactly and every number converged well beyond the digits you quote.

Give eta* to four decimal places as the final answer. In your reasoning give t2* to three decimal places; N_pump, W_a and W_b at t2*; the dot occupation at the end of stroke 1 and at the end of stroke 2 at t2*; the position and height of the next highest local maximum of eta in the window; the pumped charge per cycle in the limit where both strokes are so long that the dot reaches its stationary state with each lead; and, if the bias corresponds to 100 micro-electronvolts, the pump frequency in GHz and the pumped current in pA at t2*. For comparison with metrology, also report the largest current and the demonstrated current accuracy of the GaAs quantum-dot pump driven with specially designed gate waveforms by the National Physical Laboratory group in 2012, and the operating frequency, accuracy and temperature of the single-trap silicon electron pump from NTT reported in 2014. From the 2026 study of tunnelling engines in a triple quantum dot powered by a charge detector on the detuned central dot, state which two operations the detector lets the engine perform at the same time, and the name the authors give to the effect in which the detector drives the device into a stationary dark state. Give every quantity and literature fact requested in this paragraph in your reasoning, each as a number or a short phrase.

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

01_reaction_coordinate_drift

Goal
----
Step 01: Drift matrix after the reaction-coordinate mapping.

```python
import numpy as np


def reaction_coordinate_drift(params: np.ndarray, g1: float, g2: float) -> np.ndarray:
    '''Drift matrix of (dot, reaction coordinate 1, reaction coordinate 2) for given switching factors.

    Parameters
    ----------
    params : np.ndarray
        Shape (7,): [eps, omega_1, lambda_1, Gamma_1, omega_2, lambda_2, Gamma_2]; widths and strengths positive.
    g1 : float
        Switching factor of lead 1, in [0, 1].
    g2 : float
        Switching factor of lead 2, in [0, 1].

    Returns
    -------
    result : np.ndarray
        Complex array of shape (3, 3), the matrix M of the Heisenberg equations for (d, r_1, r_2).

    Raises
    ------
    ValueError
        If params does not have shape (7,), has non-finite entries or a non-positive width or strength, or a
        switching factor lies outside [0, 1].
    '''
    return result
```

### Step 2

02_segment_response

Goal
----
Step 02: Response to a residual-lead mode over one step.

```python
import numpy as np
from scipy.linalg import expm


def segment_response(drift: np.ndarray, tau: float, omega: np.ndarray) -> np.ndarray:
    '''Accumulated response matrices R(omega) over a step of constant drift and duration tau.

    Parameters
    ----------
    drift : np.ndarray
        Complex array of shape (3, 3); every eigenvalue has negative real part.
    tau : float
        Step duration, finite and non-negative.
    omega : np.ndarray
        One-dimensional array of real mode energies.

    Returns
    -------
    result : np.ndarray
        Complex array of shape (len(omega), 3, 3) with R(omega) = int_0^tau exp[M (tau - s)] exp(-i omega s) ds.

    Raises
    ------
    ValueError
        If drift has the wrong shape or an eigenvalue with non-negative real part, tau is negative or not finite, or
        omega is not a finite one-dimensional array.
    '''
    return result
```

### Step 3

03_cycle_propagator

Goal
----
Step 03: One-cycle propagator of the coupling/decoupling pump.

```python
import numpy as np
from scipy.linalg import expm


def cycle_propagator(params: np.ndarray, t1: float, t2: float) -> np.ndarray:
    '''Homogeneous propagator of (d, r_1, r_2) over one step-1 plus step-2 cycle.

    Parameters
    ----------
    params : np.ndarray
        Shape (7,): [eps, omega_1, lambda_1, Gamma_1, omega_2, lambda_2, Gamma_2].
    t1 : float
        Duration of step 1 (dot coupled to lead 1), positive.
    t2 : float
        Duration of step 2 (dot coupled to lead 2), positive.

    Returns
    -------
    result : np.ndarray
        Complex array of shape (3, 3) mapping the operators at the start of step 1 to the end of step 2.

    Raises
    ------
    ValueError
        If params is invalid or a duration is not finite and positive.
    '''
    return result
```

### Step 4

04_limit_cycle_weights

Goal
----
Step 04: Periodic-state spectral weights of one residual lead.

```python
import numpy as np
from scipy.linalg import expm


def limit_cycle_weights(params: np.ndarray, t1: float, t2: float, omega: np.ndarray, lead: int) -> np.ndarray:
    '''Periodic-steady-state spectral weights Q_ij(omega) of one residual lead at the ends of both steps.

    Parameters
    ----------
    params : np.ndarray
        Shape (7,): [eps, omega_1, lambda_1, Gamma_1, omega_2, lambda_2, Gamma_2].
    t1 : float
        Duration of step 1, positive.
    t2 : float
        Duration of step 2, positive.
    omega : np.ndarray
        One-dimensional array of residual-lead mode energies.
    lead : int
        1 or 2, the residual lead whose modes are resolved.

    Returns
    -------
    result : np.ndarray
        Complex array of shape (len(omega), 2, 3, 3); [:, 0] at the end of step 2, [:, 1] at the end of step 1.

    Raises
    ------
    ValueError
        If params, the durations or omega are invalid, or lead is not 1 or 2.
    '''
    return result
```

### Step 5

05_limit_cycle_correlations

Goal
----
Step 05: Dot occupation and coherences in the periodic state.

```python
import numpy as np
from scipy.linalg import expm


def limit_cycle_correlations(params: np.ndarray, t1: float, t2: float, bias: float) -> np.ndarray:
    '''Periodic-steady-state dot occupation and coherences <r_nu^+ d> at the ends of step 2 and step 1.

    Parameters
    ----------
    params : np.ndarray
        Shape (7,): [eps, omega_1, lambda_1, Gamma_1, omega_2, lambda_2, Gamma_2].
    t1 : float
        Duration of step 1, positive.
    t2 : float
        Duration of step 2, positive.
    bias : float
        V = mu_2 - mu_1 with mu_1 = 0, positive.

    Returns
    -------
    result : np.ndarray
        Complex array of shape (2, 3) with rows [<d^+ d>, <r_1^+ d>, <r_2^+ d>]; [0] end of step 2, [1] end of step 1.

    Raises
    ------
    ValueError
        If params or the durations are invalid, or the bias is not finite and positive.
    '''
    return result
```

### Step 6

06_pumping_performance

Goal
----
Step 06: Pumped charge, switching work and energy efficiency.

```python
import numpy as np
from scipy.linalg import expm


def pumping_performance(params: np.ndarray, t1: float, t2: float, bias: float) -> np.ndarray:
    '''Pumped charge per cycle, the two switching works and the energy efficiency in the periodic steady state.

    Parameters
    ----------
    params : np.ndarray
        Shape (7,): [eps, omega_1, lambda_1, Gamma_1, omega_2, lambda_2, Gamma_2].
    t1 : float
        Duration of step 1, positive.
    t2 : float
        Duration of step 2, positive.
    bias : float
        V = mu_2 - mu_1 with mu_1 = 0, positive.

    Returns
    -------
    result : np.ndarray
        Float array [N_pump, W_a, W_b, eta].

    Raises
    ------
    ValueError
        If params, the durations or the bias are invalid.
    '''
    return result
```

### Step 7

07_maximal_pumping_efficiency

Goal
----
Step 07: Largest pumping efficiency over the step-2 duration (orchestrator).

```python
import numpy as np
from scipy.linalg import expm
from scipy.optimize import minimize_scalar


def maximal_pumping_efficiency(params: np.ndarray, t1: float, t2_min: float, t2_max: float, bias: float) -> float:
    '''Global maximum over t2 in [t2_min, t2_max] of the periodic-steady-state pumping efficiency.

    Parameters
    ----------
    params : np.ndarray
        Shape (7,): [eps, omega_1, lambda_1, Gamma_1, omega_2, lambda_2, Gamma_2].
    t1 : float
        Duration of step 1, positive.
    t2_min : float
        Lower end of the step-2 window, positive.
    t2_max : float
        Upper end of the step-2 window, larger than t2_min.
    bias : float
        V = mu_2 - mu_1 with mu_1 = 0, positive.

    Returns
    -------
    result : float
        The largest efficiency eta over the window.

    Raises
    ------
    ValueError
        If an input is invalid or t2_min >= t2_max.
    '''
    return result
```
