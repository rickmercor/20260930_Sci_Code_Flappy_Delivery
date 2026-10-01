# Material_Science-Semiconductor_Materials-58

## Background

Excitonic insulators are narrow-gap semiconductors or semimetals in which electrons in the conduction band and holes in the valence band bind spontaneously into excitons in the ground state, so that the normal band insulator is replaced by a condensate of electron-hole pairs. Proposed in the 1960s, the state has attracted renewed interest with van der Waals materials: monolayers such as WTe2, atomically thin double layers of transition-metal dichalcogenides in which electrons and holes sit in separate layers, and bulk compounds such as Ta2NiSe5 and Ta2Pd3Te5 have all been put forward as candidates. Two regimes are distinguished. When the bare band gap is only slightly below the exciton binding energy the excitons are dilute, tightly bound and effectively bosonic, and the condensate is of Bose-Einstein type; when the bands overlap strongly the pairing resembles a Bardeen-Cooper-Schrieffer state with an excitation gap opened at a Fermi surface. In the dilute regime the condensate changes the band structure only mildly, by increasing the gap, and screening and coupling to the lattice are weak, which makes it the cleanest setting to test the theory.

Establishing the state experimentally is difficult because the spontaneous mixing of conduction and valence bands can be mimicked by lattice distortions, and global probes such as photoemission, transport or thermodynamic measurements average over inhomogeneous samples. Scanning tunnelling microscopy offers a local, energy-resolved alternative. For a bilayer geometry the tip couples only to the layer nearest to it; without interlayer correlations no current can flow out of that layer at biases inside the valence band, so any tunnelling signal at those energies is a direct fingerprint of condensed excitons, and its strength reflects how many there are. Theoretical work on this problem combines the mean-field (Hartree-Fock) description of the electron-hole condensate, which in the dilute limit can be organised as an expansion in the exciton density, with a golden-rule treatment of tunnelling into the resulting quasiparticle states.

Computationally the problem is a self-consistent two-band mean-field calculation in two dimensions with an unscreened, long-ranged Coulomb interaction. Its ingredients are the exciton bound-state problem, exchange self-energies built from the same Coulomb kernel, a self-consistency loop for the pairing amplitude and occupation, and the quasiparticle spectrum from which local tunnelling observables follow. Momentum-space formulations are natural for isotropic bands but require care, because the angular average of the two-dimensional Coulomb interaction has an integrable logarithmic singularity and the interlayer interaction has no closed angular form. This task builds that pipeline for a heterobilayer with unequal band masses and uses it in the inverse direction that an experiment would: from a measured satellite weight to the shape of the satellite feature at its onset.

## Problem

A scanning tunnelling microscope probes a semiconductor heterobilayer in which the conduction band lives in the top layer and the valence band in the bottom layer, a distance d apart, and the two host a dilute condensate of interlayer excitons. The tip couples only to the top layer, yet below the band gap it collects a satellite current that exists only because condensed excitons place conduction-band weight in the lower quasiparticle band. A recent theoretical study worked out the zero-temperature tunnelling spectroscopy of such condensates within Hartree-Fock mean-field theory in the low-density limit and showed that the integrated weight of this satellite feature measures the local exciton density. The task here is the inverse use of that framework for one measurement: from the integrated satellite current, find the conductance at the onset of the satellite feature.

Work in excitonic units (lengths in the exciton Bohr radius a_B*, energies in the exciton Rydberg Ry*, so that hbar^2/2m = 1 for the reduced mass m = m_c m_v/(m_c + m_v)) with the two-band effective-mass model at charge neutrality: isotropic bands epsilon_c(k) = (m/m_c) k^2 + E_G/2 and epsilon_v(k) = -(m/m_v) k^2 - E_G/2 with m_v = 2 m_c and a bare gap E_G that is to be found, an unscreened Coulomb interaction whose intralayer and interlayer Fourier components are 4 pi/q and 4 pi exp(-q d)/q with d = 0.25 a_B*, the dominant-term approximation (momentum-independent band overlaps and no interband scattering), and the charging energy of the parallel-plate capacitor formed by the two layers, whose inverse capacitance per unit area is 8 pi d in these units. The condensate is the self-consistent Hartree-Fock ground state of this model, with E_G tuned so that the condensate has exactly the exciton density fixed by the measurement.

The integrated satellite current, collected from the lower band edge down to biases far below it, is I_sat = -0.04 pi in units of G_0 Ry*/e, where G_0 = (2 pi e^2/hbar) g_s t^2 nu_0 nu_t is the conductance into a parabolic band with the reduced mass (density of states nu_0 = m/(2 pi hbar^2)), so that the total conductance deep inside a bare band of mass m_c is (m_c/m) G_0. Determine, within Fermi's golden rule at zero temperature, the differential conductance in units of G_0 that the tip records as the bias approaches the lower quasiparticle band edge from below; this onset conductance, to four decimal places, is the final answer.

The reasoning should give, to five significant figures, the exciton density implied by the measurement; the binding energy of the interlayer exciton and the inverse exciton compressibility d mu_ex/d n_ex of the dilute condensate for this bilayer together with their monolayer (d = 0) counterparts; the first-order estimate of the bare gap and the self-consistent bare gap that sustains the measured density; the excitation gap between the two quasiparticle band edges and the capacitive charging energy entering it; the conduction-band weight v_0^2 of the lower band at k = 0 together with its lowest-order estimate; the pairing self-energy and the Fock shift at k = 0 and their opposite effects on the excitation gap; the curvatures of the two quasiparticle bands at k = 0 and the bare curvatures they replace; the total conductance and the conductance into the top layer at the upper band edge, with the limits of the total conductance deep in the two bands and a check that both bands are monotonic; the value of the full satellite current recovered from the solution as a check of the measurement; why the density, unlike the bands, does not depend on the mass ratio; and how the numerics were converged, stating in each case the relations used to obtain the number.

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

01_exchange_integral

Goal
----
Step 01 - Exchange integral with the two-dimensional layered Coulomb kernel.

```python
import numpy as np
from math import comb
from typing import Callable
from scipy.special import ellipk, ellipe
from numpy.polynomial.legendre import leggauss


def exchange_integral(f: Callable, k_eval: np.ndarray, d: float) -> np.ndarray:
    '''Exchange integral (V_d f)(k) = int d^2k'/(2 pi)^2 V_|k-k'| f(|k'|) with V_q = 4 pi exp(-q d)/q.

    Parameters
    ----------
    f : Callable
        Isotropic, smooth, even function of the wavevector magnitude, called with a numpy array of
        non-negative magnitudes and returning finite values of the same shape. Decays at least as fast
        as |k'|^-3 at large |k'|.
    k_eval : np.ndarray
        One-dimensional array of finite non-negative magnitudes at which the integral is wanted (0 allowed).
    d : float
        Interlayer distance in units of a_B*, >= 0 (0 is the monolayer, kernel 4 pi/q).

    Returns
    -------
    result : np.ndarray
        Same shape as k_eval, (V_d f)(k_eval) in units of Ry* times the units of f, converged to a relative
        accuracy of 1e-9 at every point.

    Raises
    ------
    ValueError
        If d is negative or not finite, if k_eval is not a one-dimensional array of finite non-negative
        numbers, or if f returns non-finite values or an array of the wrong shape.
    '''
    return result
```

### Step 2

02_exciton_state

Goal
----
Step 02 - Binding energy and wave function of the interlayer exciton.

```python
import numpy as np
from scipy.linalg import eigh


def exciton_state(d: float, k_out: np.ndarray) -> np.ndarray:
    '''Binding energy and normalised 1s wave function of the interlayer exciton.

    Parameters
    ----------
    d : float
        Interlayer distance in units of a_B*, >= 0.
    k_out : np.ndarray
        One-dimensional array of finite non-negative wavevector magnitudes (0 allowed) at which phi is wanted.

    Returns
    -------
    result : np.ndarray
        Shape (len(k_out) + 1,): [E_b, phi(k_out[0]), ..., phi(k_out[-1])] with E_b in Ry* and phi normalised
        to int d^2k/(2 pi)^2 phi^2 = 1 with phi(0) > 0, each converged to a relative accuracy of 1e-9.

    Raises
    ------
    ValueError
        If d is negative or not finite, or if k_out is not a one-dimensional array of finite non-negative numbers.
    '''
    return result
```

### Step 3

03_inverse_compressibility

Goal
----
Step 03 - Inverse compressibility of the dilute exciton condensate.

```python
import numpy as np


def inverse_compressibility(d: float) -> float:
    '''Inverse exciton compressibility d mu_ex/d n_ex of the dilute condensate.

    Parameters
    ----------
    d : float
        Interlayer distance in units of a_B*, >= 0.

    Returns
    -------
    result : float
        d mu_ex/d n_ex in units of Ry* (a_B*)^2, converged to a relative accuracy of 1e-9.

    Raises
    ------
    ValueError
        If d is negative or not finite.
    '''
    return result
```

### Step 4

04_condensate_state

Goal
----
Step 04 - Self-consistent condensate: occupation, pairing amplitude and quasiparticle bands.

```python
import numpy as np


def condensate_state(EG: float, d: float, r: float, k_out: np.ndarray) -> np.ndarray:
    '''Self-consistent condensate: occupation, pairing amplitude and quasiparticle bands at given magnitudes.

    Parameters
    ----------
    EG : float
        Bare band gap E_G in Ry*, finite.
    d : float
        Interlayer distance in units of a_B*, >= 0.
    r : float
        Mass ratio m_v/m_c, > 0.
    k_out : np.ndarray
        One-dimensional array of finite non-negative wavevector magnitudes (0 allowed).

    Returns
    -------
    result : np.ndarray
        Shape (4, len(k_out)): rows [v_k^2, u_k v_k, E_{k,+}, E_{k,-}] at k_out, energies in Ry* measured from
        the middle of the bare gap. Trivial state (zeros and bare bands) when E_G >= E_b. Each value converged
        to 1e-9 relative accuracy (1e-9 absolute where it vanishes).

    Raises
    ------
    ValueError
        If E_G is not finite, d is negative or not finite, r is not a positive finite number, k_out is not a
        one-dimensional array of finite non-negative numbers, or the self-consistent iteration does not converge
        within 400 iterations.
    '''
    return result
```

### Step 5

05_exciton_density

Goal
----
Step 05 - Exciton density of the self-consistent condensate.

```python
import numpy as np


def exciton_density(EG: float, d: float) -> float:
    '''Exciton density n_ex of the self-consistent condensate.

    Parameters
    ----------
    EG : float
        Bare band gap E_G in Ry*, finite.
    d : float
        Interlayer distance in units of a_B*, >= 0.

    Returns
    -------
    result : float
        n_ex = int d^2k/(2 pi)^2 v_k^2 in units of (a_B*)^-2 (0.0 for E_G >= E_b), converged to a relative
        accuracy of 1e-9.

    Raises
    ------
    ValueError
        If E_G is not finite, d is negative or not finite, or the self-consistent iteration does not converge
        within 400 iterations.
    '''
    return result
```

### Step 6

06_gap_for_density

Goal
----
Step 06 - Bare gap that sustains a prescribed exciton density.

```python
import numpy as np


def gap_for_density(n_target: float, d: float) -> float:
    '''Bare gap E_G at which the self-consistent condensate has exciton density n_target.

    Parameters
    ----------
    n_target : float
        Target exciton density in units of (a_B*)^-2, >= 0.
    d : float
        Interlayer distance in units of a_B*, >= 0.

    Returns
    -------
    result : float
        E_G in Ry* (E_b when n_target is 0), such that the self-consistent density equals n_target to a relative
        accuracy of 1e-10.

    Raises
    ------
    ValueError
        If n_target is negative or not finite, or if d is negative or not finite.
    '''
    return result
```

### Step 7

07_averaged_conductance

Goal
----
Step 07 - Spatially averaged tunnelling conductance versus bias.

```python
import numpy as np
from scipy.optimize import brentq


def averaged_conductance(EG: float, d: float, r: float, biases: np.ndarray) -> np.ndarray:
    '''Spatially averaged zero-temperature tunnelling conductance dI/dV of the condensate.

    Parameters
    ----------
    EG : float
        Bare band gap E_G in Ry*, finite.
    d : float
        Interlayer distance in units of a_B*, >= 0.
    r : float
        Mass ratio m_v/m_c, > 0.
    biases : np.ndarray
        One-dimensional array of finite bias energies eV in Ry*, measured from the middle of the bare gap.

    Returns
    -------
    result : np.ndarray
        Shape (2, len(biases)) in units of G_0: row 0 the total conductance (both quasiparticle bands with unit
        weight), row 1 the conductance into the conduction-band layer only (upper band weighted by u_k^2, lower
        band by v_k^2). Zero inside the excitation gap; the inside limit within 1e-9 Ry* of an edge. Converged to
        1e-8 relative accuracy.

    Raises
    ------
    ValueError
        If E_G is not finite, d is negative or not finite, r is not positive and finite, biases is not a
        one-dimensional array of finite numbers, the self-consistent iteration does not converge, or either
        quasiparticle band is not monotonic on k >= 0.
    '''
    return result
```

### Step 8

08_satellite_current

Goal
----
Step 08 - Integrated current of the satellite feature.

```python
import numpy as np
from numpy.polynomial.legendre import leggauss


def satellite_current(EG: float, d: float, r: float, V: float) -> float:
    '''Current collected from the satellite feature between the lower band edge and the bias V.

    Parameters
    ----------
    EG : float
        Bare band gap E_G in Ry*, finite.
    d : float
        Interlayer distance in units of a_B*, >= 0.
    r : float
        Mass ratio m_v/m_c, > 0.
    V : float
        Bias energy eV in Ry* (measured from the middle of the bare gap), finite or -numpy.inf.

    Returns
    -------
    result : float
        I(V) = int_V^{E_{0,-}} (dI/dV') dV' of the conductance into the conduction-band layer, in units of
        G_0 Ry*/e, non-positive, 0.0 for V >= E_{0,-}, converged to 1e-8 relative accuracy.

    Raises
    ------
    ValueError
        If E_G is not finite, d is negative or not finite, r is not positive and finite, V is not a real number
        or -inf, the self-consistent iteration does not converge, or the lower band is not monotonic.
    '''
    return result
```

### Step 9

09_satellite_onset_conductance

Goal
----
Step 09 - Onset conductance of the satellite feature from the measured satellite weight (orchestrator).

```python
import numpy as np
from scipy.interpolate import CubicSpline


def satellite_onset_conductance(I_sat: float, d: float, r: float) -> float:
    '''Conductance at the onset of the satellite feature, from the measured integrated satellite current.

    Parameters
    ----------
    I_sat : float
        Integrated satellite current I(V -> -infinity) in units of G_0 Ry*/e, finite and negative.
    d : float
        Interlayer distance in units of a_B*, >= 0.
    r : float
        Mass ratio m_v/m_c, > 0.

    Returns
    -------
    result : float
        Limit of the conductance into the conduction-band layer as the bias approaches the lower band edge from
        below, in units of G_0, converged to 1e-7 relative accuracy.

    Raises
    ------
    ValueError
        If I_sat is not a finite negative number, d is negative or not finite, r is not positive and finite, the
        quasiparticle bands are not monotonic, or an internal consistency check of the pipeline fails.
    '''
    return result
```
