# Material_Science-Semiconductor_Materials-35

## Background

Resonant-tunnelling diodes remain one of the very few electronic devices that still deliver negative differential conductance at hundreds of gigahertz and beyond, which is why a diode placed across a small resonator, usually a slot or patch antenna, is a serious candidate for a compact, room-temperature terahertz source. The oscillators are simple to build, but they are hard to characterise once built: the diode sits inside the antenna, so its capacitance, the resonator's inductance and capacitance and the load presented by radiation and ohmic loss cannot be measured separately, and most of what is known about a finished oscillator is inferred from its output frequency, its output power and the dc current it draws.

The standard design picture treats the circuit as a van der Pol oscillator: the diode supplies a nonlinear negative conductance, the resonator supplies the inductance and capacitance, and the oscillation grows until the diode's large-signal gain just balances the load. In the classical treatment the diode characteristic is replaced by a cubic fitted through its peak and valley, and the oscillation frequency is taken to be the resonance of the inductance with the total capacitance. Real characteristics are strongly asymmetric about any practical operating point, the oscillation swings the diode across much of its negative-differential-conductance region, and both the nonlinear current and the diode's charge storage respond to that swing, so a real oscillator departs from the linear resonance in ways the simple picture does not capture.

Two long-standing puzzles follow from this. Oscillators are known to tune with bias, often by a few percent, and the tuning has traditionally been ascribed to the voltage dependence of the diode capacitance, although measured tuning curves show features that a capacitance change alone does not explain. And when device capacitances are fitted to measured oscillation frequencies, the fitted values have sometimes come out larger than independent estimates, suggesting that some other mechanism pulling the frequency down was being absorbed into the capacitance. Deciding how much of an observed frequency belongs to the resonator and how much to the nonlinear dynamics of the diode is therefore a practical question for anyone extracting circuit parameters from terahertz oscillator measurements.

## Problem

Resonant-tunnelling-diode (RTD) oscillators change frequency when their bias is changed, and the change is usually put down to the diode capacitance alone, which falls as the collector depletes, even though the first-order Poincare-Lindstedt treatment of the circuit as a generalized van der Pol oscillator shows that a realistic, strongly asymmetric diode characteristic also pulls the frequency below the resonator eigenfrequency by an amount that depends on bias. Your task is to separate the two effects in a set of measurements on one oscillator and recover the capacitance that the antenna and resonator place in parallel with the diode.

The diode has the quasi-static characteristic I(V) = 24 exp(-((V - 0.35)/0.18)^2) + 15 V^3, with V in volts and I in mA/um^2, and a small-signal capacitance C0(V) = Cb + Cq exp(-((V - 0.40)/0.12)^2) fF/um^2 whose background Cb = 7.0 fF/um^2 is known but whose quantum-well peak height Cq is not. The resonator's parallel capacitance C_r and inductance-area product L, both referred to unit diode area, and its load conductance G_l are unknown as well, and none of the four quantities changes with bias. At 0.44 V the oscillating diode draws a dc current density of 12.80 mA/um^2 and oscillates at 648.0 GHz, at 0.54 V it oscillates at 673.0 GHz, and when the bias is raised further the oscillation survives up to an edge, where the last frequency seen before it collapses is 674.6 GHz.

Treat each bias at first order in the small parameter eps = G0/(C omega0), where G0 is the maximum negative differential conductance of I(V), C = C_r + C_LS is the total capacitance at that bias, omega0 = 1/sqrt(L C), the ac voltage is scaled by the peak-to-valley voltage separation of I(V), harmonics n = 2 to 12 are retained, and the oscillation frequency is omega0 (1 + eps^2 kappa1) with kappa1 the first-order frequency coefficient. Take the dc current of the oscillating diode as I(V) plus the cycle average of the zero-order ac diode current, using the smallest amplitude that reproduces the measured value; take the stable, largest root wherever the zero-order amplitude condition, in which the fundamental component of the diode current balances the load, has several, and take the edge of the oscillation region to be the highest bias at which that condition still has a solution; and take C_LS as the capacitance of the linear capacitor that carries the same fundamental displacement current as C0(V) under the zero-order oscillation.

Report C_r in fF/um^2. In your reasoning give at least these scalars: the bias at the edge of the oscillation region and the zero-order amplitude at each of the three measurement points, in units of the peak-to-valley voltage separation, G_l in mS/um^2, kappa1 at each point, the fitted Cq and the C_LS at each point in fF/um^2, L in pH um^2, the fractional frequency shift eps^2 kappa1 at each point, and at each point the applicability criterion, the largest of |eps b1|, |eps b1^(n)| for n = 2 to 12 and |eps^2 kappa1|, where b1 and b1^(n) are the first-order quadrature and harmonic coefficients of the scaled voltage, which must not exceed 0.30. Also state the tuning range and the two centre frequencies found in the early report of voltage-controlled sub-terahertz oscillation in resonant-tunnelling diodes integrated with slot antennas, and the stray capacitance identified in the recent experimental comparison of oscillators built from RTDs with 1.6 nm and 1.0 nm barriers.

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

01_dc_current

Goal
----
Evaluate the quasi-static dc current density of the resonant-tunnelling diode for a parameterised characteristic.

```python
import numpy as np

def dc_current(V: np.ndarray, iv_params: np.ndarray) -> np.ndarray:
    '''Quasi-static dc current density of the diode.

    Parameters
    ----------
    V : numpy.ndarray
        Bias values in volts, any shape.
    iv_params : numpy.ndarray
        The characteristic parameters [A1, V1, w1, A2, V2, w2, B] in mA/um^2,
        V, V, mA/um^2, V, V and mA/(um^2 V^3).

    Returns
    -------
    I : numpy.ndarray
        Current density in mA/um^2, same shape as V.

    Raises
    ------
    ValueError
        If iv_params does not hold exactly seven values or either width w1, w2
        is not positive.
    '''
    return I
```

### Step 2

02_characteristic_scales

Goal
----
Locate the peak and valley of the characteristic and return the peak-to-valley separation and the maximum negative differential conductance.

```python
import numpy as np

def characteristic_scales(iv_params: np.ndarray) -> np.ndarray:
    '''Voltage and conductance scales of the dc characteristic.

    Parameters
    ----------
    iv_params : numpy.ndarray
        The characteristic parameters [A1, V1, w1, A2, V2, w2, B] in mA/um^2,
        V, V, mA/um^2, V, V and mA/(um^2 V^3).

    Returns
    -------
    scales : numpy.ndarray
        Array [Vp, Vv, dV, G0]: the peak and valley biases in volts, their
        separation in volts, and the maximum negative differential conductance
        inside that first region in mS/um^2 (a positive number).

    Raises
    ------
    ValueError
        If no current peak followed by a valley exists for 0 < V <= 2 V, or if
        iv_params is invalid as in step 01.
    '''
    return scales
```

### Step 3

03_harmonic_coefficients

Goal
----
Compute the cycle average and cosine Fourier coefficients of the ac diode current under a cosine voltage swing.

```python
import numpy as np

def harmonic_coefficients(Vdc: float, a0: float, dV: float, iv_params: np.ndarray, nmax: int) -> np.ndarray:
    '''Fourier coefficients of the ac diode current under a cosine swing.

    Parameters
    ----------
    Vdc : float
        Operating bias in volts.
    a0 : float
        Swing amplitude in units of dV (a0 = 0 means no swing).
    dV : float
        Peak-to-valley voltage separation in volts.
    iv_params : numpy.ndarray
        The characteristic parameters [A1, V1, w1, A2, V2, w2, B] in mA/um^2,
        V, V, mA/um^2, V, V and mA/(um^2 V^3).
    nmax : int
        Highest harmonic index returned.

    Returns
    -------
    c : numpy.ndarray
        Array [c_0, c_1, ..., c_nmax] in mA/um^2, where c_0 is the cycle
        average of i_D and c_n (n >= 1) its cosine Fourier coefficients.

    Raises
    ------
    ValueError
        If a0 < 0, dV <= 0 or nmax < 1, or if iv_params is invalid as in
        step 01.
    '''
    return c
```

### Step 4

04_oscillation_amplitude

Goal
----
Solve the zero-order gain balance for the stable oscillation amplitude at a given bias and load.

```python
import numpy as np

def oscillation_amplitude(Vdc: float, Gl: float, dV: float, iv_params: np.ndarray) -> float:
    '''Stable zero-order oscillation amplitude at a given bias and load.

    Parameters
    ----------
    Vdc : float
        Operating bias in volts.
    Gl : float
        Load conductance in mS/um^2.
    dV : float
        Peak-to-valley voltage separation in volts.
    iv_params : numpy.ndarray
        The characteristic parameters [A1, V1, w1, A2, V2, w2, B] in mA/um^2,
        V, V, mA/um^2, V, V and mA/(um^2 V^3).

    Returns
    -------
    a0 : float
        Largest amplitude in 0 < a0 <= 3, in units of dV, that satisfies the
        gain balance, or 0.0 if there is none.

    Raises
    ------
    ValueError
        If Gl < 0 or dV <= 0, or if iv_params is invalid as in step 01.
    '''
    return a0
```

### Step 5

05_oscillation_edge

Goal
----
Find the upper edge of the oscillation region, the turning point at which the gain-balance branch ends, and the amplitude there.

```python
import numpy as np

def oscillation_edge(Gl: float, dV: float, iv_params: np.ndarray) -> np.ndarray:
    '''Upper edge of the oscillation region for a given load.

    Parameters
    ----------
    Gl : float
        Load conductance in mS/um^2.
    dV : float
        Peak-to-valley voltage separation in volts.
    iv_params : numpy.ndarray
        The characteristic parameters [A1, V1, w1, A2, V2, w2, B] in mA/um^2,
        V, V, mA/um^2, V, V and mA/(um^2 V^3).

    Returns
    -------
    edge : numpy.ndarray
        Array [V_edge, a_edge]: the highest bias in volts at which the gain
        balance still has a solution, and the amplitude there in units of dV.

    Raises
    ------
    ValueError
        If Gl <= 0 or dV <= 0, if no bias in 0 < V <= 2 V supports an
        oscillation at this load, or if iv_params is invalid as in step 01.
    '''
    return edge
```

### Step 6

06_rectified_amplitude

Goal
----
Recover the zero-order oscillation amplitude from the dc current drawn by the oscillating diode.

```python
import numpy as np

def rectified_amplitude(Vdc: float, J_dc: float, dV: float, iv_params: np.ndarray) -> float:
    '''Zero-order amplitude implied by the dc current of the oscillating diode.

    Parameters
    ----------
    Vdc : float
        Operating bias in volts.
    J_dc : float
        Measured dc current density of the oscillating diode in mA/um^2.
    dV : float
        Peak-to-valley voltage separation in volts.
    iv_params : numpy.ndarray
        The characteristic parameters [A1, V1, w1, A2, V2, w2, B] in mA/um^2,
        V, V, mA/um^2, V, V and mA/(um^2 V^3).

    Returns
    -------
    a0 : float
        Smallest amplitude in 0 < a0 <= 3, in units of dV, for which
        I(Vdc) + c_0(a0) equals J_dc.

    Raises
    ------
    ValueError
        If no amplitude in 0 < a0 <= 3 reproduces J_dc, if dV <= 0, or if
        iv_params is invalid as in step 01.
    '''
    return a0
```

### Step 7

07_first_order_coefficients

Goal
----
Compute the first-order Poincare-Lindstedt quadrature, harmonic and frequency coefficients at one bias.

```python
import numpy as np

def first_order_coefficients(Vdc: float, a0: float, dV: float, G0: float, iv_params: np.ndarray, nmax: int) -> np.ndarray:
    '''First-order Poincare-Lindstedt coefficients at one bias.

    Parameters
    ----------
    Vdc : float
        Operating bias in volts.
    a0 : float
        Zero-order amplitude in units of dV.
    dV : float
        Peak-to-valley voltage separation in volts.
    G0 : float
        Maximum negative differential conductance in mS/um^2.
    iv_params : numpy.ndarray
        The characteristic parameters [A1, V1, w1, A2, V2, w2, B] in mA/um^2,
        V, V, mA/um^2, V, V and mA/(um^2 V^3).
    nmax : int
        Highest harmonic retained.

    Returns
    -------
    coeffs : numpy.ndarray
        Array [b1, b1^(2), ..., b1^(nmax), kappa1] of length nmax + 1, all
        dimensionless, in the normalisation defined above.

    Raises
    ------
    ValueError
        If a0 <= 0, dV <= 0, G0 <= 0 or nmax < 2, or if iv_params is invalid
        as in step 01.
    '''
    return coeffs
```

### Step 8

08_large_signal_capacitance

Goal
----
Compute the large-signal diode capacitance seen by the resonator under the zero-order swing.

```python
import numpy as np

def large_signal_capacitance(Vdc: float, a0: float, dV: float, cap_params: np.ndarray) -> float:
    '''Large-signal diode capacitance under the zero-order swing.

    Parameters
    ----------
    Vdc : float
        Operating bias in volts.
    a0 : float
        Zero-order amplitude in units of dV (a0 = 0 means no swing).
    dV : float
        Peak-to-valley voltage separation in volts.
    cap_params : numpy.ndarray
        Small-signal capacitance parameters [Cb, Cq, Vq, wq] in fF/um^2,
        fF/um^2, V and V.

    Returns
    -------
    C_LS : float
        Large-signal capacitance in fF/um^2.

    Raises
    ------
    ValueError
        If a0 < 0 or dV <= 0, or if cap_params does not hold exactly four
        values or its width wq is not positive.
    '''
    return C_LS
```

### Step 9

09_bias_frequency

Goal
----
Evaluate the first-order oscillation frequency, the small parameter and the applicability criterion for a given resonator.

```python
import numpy as np

def bias_frequency(Cr: float, ind: float, c_ls: float, coeffs: np.ndarray, G0: float) -> np.ndarray:
    '''First-order oscillation frequency, small parameter and applicability criterion.

    Parameters
    ----------
    Cr : float
        Resonator capacitance per unit diode area in fF/um^2.
    ind : float
        Inductance-area product of the resonator in pH um^2.
    c_ls : float
        Large-signal diode capacitance in fF/um^2.
    coeffs : numpy.ndarray
        First-order coefficients [b1, b1^(2), ..., b1^(nmax), kappa1] from
        step 07.
    G0 : float
        Maximum negative differential conductance in mS/um^2.

    Returns
    -------
    out : numpy.ndarray
        Array [f, eps, crit]: the oscillation frequency omega/(2 pi) in GHz,
        the small parameter eps and the applicability criterion.

    Raises
    ------
    ValueError
        If Cr < 0, c_ls <= 0, ind <= 0 or G0 <= 0, or if coeffs has fewer than
        three entries.
    '''
    return out
```

### Step 10

10_resonator_capacitance

Goal
----
Recover the resonator capacitance, its inductance and the quantum-well capacitance peak from a dc current and three bias-dependent oscillation frequencies (final orchestrator).

```python
import numpy as np

def resonator_capacitance(iv_params: np.ndarray, cap_fixed: np.ndarray, bias: np.ndarray, J_dc: float, freq: np.ndarray, nmax: int) -> float:
    '''Resonator capacitance recovered from bias-dependent oscillator data.

    Parameters
    ----------
    iv_params : numpy.ndarray
        The characteristic parameters [A1, V1, w1, A2, V2, w2, B] in mA/um^2,
        V, V, mA/um^2, V, V and mA/(um^2 V^3).
    cap_fixed : numpy.ndarray
        The known part of the small-signal capacitance model, [Cb, Vq, wq] in
        fF/um^2, V and V; the peak height Cq is unknown.
    bias : numpy.ndarray
        The two measured biases [V1, V2] in volts; the dc current is measured
        at V1. The third measurement is at the upper edge of the oscillation
        region, whose bias follows from the load.
    J_dc : float
        dc current density of the oscillating diode at V1 in mA/um^2.
    freq : numpy.ndarray
        Measured oscillation frequencies [f1, f2, f_edge] in GHz at V1, at V2
        and at the upper edge of the oscillation region.
    nmax : int
        Highest harmonic retained.

    Returns
    -------
    Cr : float
        Resonator capacitance per unit diode area in fF/um^2.

    Raises
    ------
    ValueError
        If bias does not hold two values, freq does not hold three, a frequency
        is not positive or nmax < 2; if the dc current implies a negative load
        conductance; if a bias supports no oscillation at that load; if no
        Cq >= 0, C_r >= 0 and L > 0 reproduce the three frequencies; if the
        applicability criterion exceeds 0.3 at any bias; or if an earlier step
        raises.
    '''
    return Cr
```
