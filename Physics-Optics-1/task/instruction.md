# Physics-Optics-1

## Background

Photonic time crystals are the temporal counterpart of spatial photonic crystals. The permittivity of a spatially uniform medium is modulated in time, so the wavevector is conserved and momentum gaps open where waves are exponentially amplified. Driving the modulation quasiperiodically, with an irrational phase advance per temporal layer, produces a photonic time quasicrystal. It keeps momentum gaps without temporal periodicity, and its richer gap structure carries Chern labels over the Floquet and modulation phases.

Continuing the modulation phase into the complex plane turns the single-harmonic permittivity into a temporal analogue of the non-Hermitian Aubry–André–Harper model. The coefficient of the displacement-field wave equation is proportional to the reciprocal permittivity, so the layer transfer matrix differs from that of the standard AAH model. For analytic one-frequency $\mathrm{SL}(2,\mathbb{C})$ cocycles with irrational rotation, Avila's global theory says that the Lyapunov exponent is convex and piecewise affine in the imaginary phase displacement, and that its slope, the acceleration, is quantized to integers. In this optical setting that exponent is the bulk logarithmic amplification per layer. The integer then appears as a directly tunable, quantized growth response rather than a quantity reconstructed from localization or spectral winding.

Locating the transitions between integer phases requires either long direct propagation with the irrational rotation or an approximation scheme. Rational approximants of the rotation number give periodic auxiliary cells whose trace has an exactly known dependence on the continuation coordinate. This turns boundary prediction into a comparison of a few numbers, with no fitted growth rates. Because different input frequencies can sit in different integer phases under the same modulation, the integer difference controls how sensitively their relative output power responds to the continuation parameter. That gives a quantized handle on spectral selection in the propagated wave.

## Problem

When the permittivity of a spatially uniform dielectric is switched in time, momentum gaps open in which waves are amplified instead of reflected. Consider a quasiperiodically switched dielectric: equal-duration temporal layers whose relative permittivity follows a single-harmonic sinusoid with a golden-mean phase advance per layer, with the modulation phase continued into the complex plane by an imaginary displacement $h$, which makes the wave problem non-Hermitian. In the source paper's analysis, the bulk logarithmic amplification per layer is continuous and piecewise affine in $h$, with an integer slope inside each phase. The paper predicts the boundaries between these integer phases from an auxiliary periodic approximant of the rotation number, instead of by propagating through the irrational sequence. When two narrowband inputs belong to different integer phases, their relative output power changes exponentially with $h$, at a rate set by the integer difference and the number of layers. Your task is to use the paper's approximant-based boundary prediction to evaluate this relative spectral response for one configuration.

Here is the exact setup to use:

- Medium and modulation: the paper's analytic complex-phase model for a quasiperiodically switched dielectric, which is spatially uniform and non-magnetic, with an instantaneous, nondispersive complex relative permittivity. Background permittivity $\varepsilon_R=3.4$, modulation-amplitude parameter $\delta\varepsilon=1.0$, and golden-mean rotation $\alpha=(\sqrt{5}-1)/2$. Use the paper's own complex-continued waveform and its own two-component layer propagator, in the paper's main-text state basis.
- Two input bands, each treated at its centre frequency: $\omega_1\tau=15.54$ (band 1) and $\omega_2\tau=15.82$ (band 2), where $\omega_0\tau$ is the phase accumulated per layer in the unmodulated reference medium.
- Integer-phase boundaries: do not locate them by direct propagation with the irrational rotation or from fitted growth rates. Use the paper's own boundary predictor, built from the continued-fraction approximant of $\alpha$ whose denominator is $34$. Numerical protocol for this predictor: reference continuation height $h_0=0.80$, $64$ equally spaced real phase nodes over one period of the object the predictor is built from, and orders $0$ through $4$ only (these are the orders that are numerically resolved at this $h_0$; the others never dominate for $h\le 1.10$). Consult the paper's own equations for how the predictor is constructed and for its sign and normalization conventions. The predicted boundaries are not the same as the irrational-rotation bulk boundaries, and a different approximant gives a different prediction.
- For each band, the predicted integer response $n_j(h)$ over the continuation interval $0.60\le h\le 1.10$ is the order that dominates the predictor at that $h$. Do not assume that the integer difference $n_2-n_1$ keeps one sign over the interval.
- Propagation length $N=120$ layers. Use the paper's leading-order long-propagation relation between the integer difference of the two bands and the change of their output power ratio, taking the band-crossing structure inside the interval into account.

Report $\Delta_{\mathrm{dB}}$, the predicted change of $10\log_{10}(P_2/P_1)$ between $h=0.60$ and $h=1.10$ after $N=120$ layers, in dB. The answer is graded to an absolute tolerance of $10^{-4}$. In your reasoning, state the equations you use for the Fourier construction of the predictor, for the boundary crossings and for the power-ratio conversion, with enough supporting explanation to justify each of the following:

- confirm the interval lies inside the paper's nonsingular strip, stating the strip half-width;
- report the intercepts $b_0$ to $b_3$ of each band's predictor;
- report each band's predicted boundaries inside the interval;
- report the sub-intervals on which $n_2-n_1$ is nonzero, together with its value on each;
- report the integrals of $n_1(h)$ and $n_2(h)$ over the interval;
- compare the predicted band-2 boundary where $n_2$ goes from $1$ to $2$ with the value the paper reports for that band from direct bulk propagation.

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

analytic_strip_halfwidth

Goal
----
Compute the half-width $h_\star$ of the nonsingular analytic strip of the complex-continued quasiperiodic permittivity sequence of the time-switched dielectric, i.e. the largest imaginary phase displacement $|h|$ for which the continued single-harmonic permittivity stays nonzero for every real modulation phase.

```python
import numpy as np


def analytic_strip_halfwidth(eps_R: float, d_eps: float) -> float:
    r"""Half-width of the nonsingular strip of the continued permittivity.

    Args:
        eps_R (float): background (reference) relative permittivity $\varepsilon_R$, real and positive.
        d_eps (float): modulation-amplitude parameter $\delta\varepsilon$ at $h=0$, real and positive.

    Raises:
        ValueError: unless $\varepsilon_R>\delta\varepsilon>0$.

    Expected return:
        float: strictly positive; grows without bound as $\delta\varepsilon\to 0$ and tends to $0$
        as $\delta\varepsilon\to\varepsilon_R$ from below.
    """
    return 0.0
```

### Step 2

layer_transfer_matrix

Goal
----
Compute the $2\times 2$ propagator of one temporal layer of constant (possibly complex) relative permittivity $\varepsilon_m$ and dimensionless duration $\omega_0\tau$, acting on the paper's main-text two-component propagation state built from the displacement field and its scaled time derivative.

```python
import numpy as np


def layer_transfer_matrix(eps_m: complex, eps_R: float, w0tau: float) -> "np.ndarray":
    r"""One-layer propagator of the temporal layer sequence.

    Args:
        eps_m (complex): relative permittivity $\varepsilon_m$ of the layer (nonzero, may be complex).
        eps_R (float): real positive reference relative permittivity $\varepsilon_R$ that defines $\omega_0$.
        w0tau (float): phase $\omega_0\tau$ accumulated over one layer in the unmodulated reference
            medium, $\omega_0\tau>0$.

    Raises:
        ValueError: if $\varepsilon_m=0$, $\varepsilon_R\le 0$ or $\omega_0\tau\le 0$.

    Expected return:
        np.ndarray of shape (2, 2), complex, determinant $1$. For real $\varepsilon_m$ equal to
        $\varepsilon_R$ it is a pure phase rotation of the state; its two diagonal entries are equal.
    """
    return None
```

### Step 3

approximant_cell_trace

Goal
----
Compute the trace $F_q(\varphi,h)$ of the $q$-layer periodic-cell propagator of the paper's auxiliary rational approximant $p/q$ of the golden-mean rotation, for the complex-continued single-harmonic permittivity, at real initial phase $\varphi$ (in cycles) and continuation coordinate $h$.

```python
import numpy as np


def approximant_cell_trace(phi: float, h: float, p: int, q: int, eps_R: float,
                           d_eps: float, w0tau: float) -> complex:
    r"""Trace of the rational-approximant periodic-cell propagator.

    Args:
        phi (float): real initial modulation phase $\varphi$, in cycles.
        h (float): continuation coordinate $h$ (imaginary phase displacement).
        p (int): approximant numerator $p$, coprime with $q$.
        q (int): approximant denominator $q$ (cell length in layers), $q\ge 1$.
        eps_R (float): background relative permittivity $\varepsilon_R$.
        d_eps (float): modulation-amplitude parameter $\delta\varepsilon$ at $h=0$.
        w0tau (float): reference phase $\omega_0\tau$ accumulated per layer.

    Raises:
        ValueError: if $q<1$, if $p$ and $q$ are not coprime integers, or if $|h|$ is not
            strictly inside the nonsingular strip.

    Expected return:
        complex: invariant under $\varphi\to\varphi+1/q$; real when $h=0$ and the
        permittivities are real; its magnitude grows rapidly with $h$.
    """
    return None
```

### Step 4

trace_fourier_coefficients

Goal
----
Compute the complex Fourier amplitudes $C_n^{(q)}(h_0)$, $n=0,1,\dots,n_{\max}$, of the approximant cell trace at reference continuation height $h_0$, sampling the trace at $K$ equally spaced real phases over one of its periods.

```python
import numpy as np


def trace_fourier_coefficients(h0: float, p: int, q: int, eps_R: float, d_eps: float,
                               w0tau: float, n_max: int, K: int) -> "np.ndarray":
    r"""Fourier amplitudes of the approximant cell trace at height $h_0$.

    Args:
        h0 (float): reference continuation height $h_0$, strictly inside the strip.
        p (int): approximant numerator $p$ (coprime with $q$).
        q (int): approximant denominator $q$.
        eps_R (float): background relative permittivity $\varepsilon_R$.
        d_eps (float): modulation-amplitude parameter $\delta\varepsilon$ at $h=0$.
        w0tau (float): reference phase $\omega_0\tau$ accumulated per layer.
        n_max (int): highest retained non-negative Fourier order $n_{\max}\ge 0$.
        K (int): number $K$ of equally spaced real phase nodes over one period of the
            trace; must satisfy $K\ge 2n_{\max}+2$.

    Raises:
        ValueError: if $n_{\max}<0$ or $K<2n_{\max}+2$ (plus any error raised by the
            cell-trace evaluation).

    Expected return:
        np.ndarray of shape (n_max + 1,), complex. Index $n$ holds the order-$n$ amplitude.
    """
    return None
```

### Step 5

coefficient_intercepts

Goal
----
Convert the Fourier amplitudes of the approximant cell trace, obtained at reference height $h_0$, into the paper's real intercepts $b_n^{(q)}$ of the logarithmic coefficient magnitudes.

```python
import numpy as np


def coefficient_intercepts(coeffs: "np.ndarray", q: int, h0: float) -> "np.ndarray":
    r"""Intercepts of the logarithmic Fourier-coefficient magnitudes.

    Args:
        coeffs (np.ndarray): complex Fourier amplitudes $[C_0,C_1,\dots]$ of the cell
            trace, all sampled at the same reference height $h_0$; index = order $n$.
        q (int): approximant denominator $q$ used to build the coefficients.
        h0 (float): the reference continuation height $h_0$ at which they were sampled.

    Raises:
        ValueError: if any coefficient is exactly zero.

    Expected return:
        np.ndarray of float, same length as coeffs; for exact coefficients the result
        does not depend on $h_0$.
    """
    return None
```

### Step 6

integrated_dominant_order

Goal
----
Given the intercepts $b_n^{(q)}$ of one band, integrate the predicted integer bulk response $n(h)$ over the continuation interval $[h_a,h_b]$, where $n(h)$ is the Fourier order that dominates the limiting envelope of the logarithmic coefficient lines at height $h$.

```python
import numpy as np


def integrated_dominant_order(b: "np.ndarray", h_a: float, h_b: float) -> float:
    r"""Area under the predicted piecewise-constant integer response.

    Args:
        b (np.ndarray): real intercepts $[b_0,b_1,\dots,b_{n_{\max}}]$; index = Fourier order.
        h_a (float): lower end $h_a$ of the continuation interval.
        h_b (float): upper end $h_b$, with $h_b>h_a$.

    Raises:
        ValueError: if $h_b\le h_a$.

    Expected return:
        float: between $(\min n)(h_b-h_a)$ and $(\max n)(h_b-h_a)$ over the retained orders $n$; equal to
        $n\,(h_b-h_a)$ when one order $n$ dominates throughout the interval.
    """
    return 0.0
```

### Step 7

run_ptqc_spectral_contrast

Goal
----
Chain the earlier steps into the full prediction: check the continuation interval and the reference height against the nonsingular strip, build the approximant trace Fourier amplitudes and their intercepts for each of the two input bands, integrate each band's predicted integer response over $[h_a,h_b]$, and convert the integrated integer difference (band 2 minus band 1) into the change of the two-band output power ratio, in decibels, after $N$ layers. The reference implementation calls the earlier public functions by name rather than reproducing them.

```python
import numpy as np


def run_ptqc_spectral_contrast(w1tau: float, w2tau: float, eps_R: float, d_eps: float,
                               p: int, q: int, h0: float, h_a: float, h_b: float, N: int,
                               n_max: int, K: int) -> float:
    r"""Predicted change of the two-band output power ratio (dB).

    Args:
        w1tau (float): reference phase per layer $\omega_1\tau$ at the centre of band 1.
        w2tau (float): reference phase per layer $\omega_2\tau$ at the centre of band 2.
        eps_R (float): background relative permittivity $\varepsilon_R$.
        d_eps (float): modulation-amplitude parameter $\delta\varepsilon$ at $h=0$.
        p (int): approximant numerator $p$.
        q (int): approximant denominator $q$.
        h0 (float): reference continuation height $h_0$ for the Fourier amplitudes.
        h_a (float): lower end $h_a$ of the continuation interval, $h_a\ge 0$.
        h_b (float): upper end $h_b$, with $h_a<h_b<h_\star$ (the strip half-width).
        N (int): number of propagated layers, $N>0$.
        n_max (int): highest retained Fourier order $n_{\max}$.
        K (int): phase nodes $K$ per trace period.

    Raises:
        ValueError: if the interval or $h_0$ is not inside the nonsingular strip, if
            $N\le 0$, or if any earlier step rejects its inputs.

    Expected return:
        float: dB change; changes sign when the two bands are swapped and is
        proportional to $N$.
    """
    return 0.0
```
