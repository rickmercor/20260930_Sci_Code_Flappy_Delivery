# Material_Science-Molecular_Modeling-11

## Background

Overdamped Brownian motion in a tilted periodic (washboard) potential is a minimal non-equilibrium model for driven transport of adsorbates on corrugated surfaces, ions in channels with periodic free-energy profiles and molecular motors on filaments. Below the critical tilt the particle is trapped in local minima and escapes over low barriers. This produces a strongly enhanced long-time diffusivity near the critical tilt, a violation of the equilibrium Einstein relation between diffusivity and mobility, and transient non-Gaussian displacement statistics. Because the potential is periodic, the Fokker-Planck operator decomposes into Bloch wave-number sectors, and scattering functions and displacement cumulants follow from the spectrum of the resulting non-Hermitian operator.

## Problem

Analyze an overdamped Brownian particle in a tilted cosine potential using the source paper's Bloch and spectral construction. Determine the peak non-Gaussian parameter at an intermediate locked-regime tilt that the paper does not tabulate. Use the paper's generalized intermediate scattering function, stationary Fourier solution, perturbative cumulants, transport formulas, and biorthogonal eigenvectors.

Use $\beta U(x)=u\cos(Q_1x)-Q_1fx$, $Q_1=2\pi/L$, $u=U_1/(k_BT)$, and $f=FL/(2\pi k_BT)$. Positive $f$ drives motion toward $+x$; $f<u$ is the locked regime. Set $L=D=k_BT=1$. Time is measured in $L^2/D$, wave number in $1/L$, the Brillouin zone is $|q|<\pi$, and Fourier indices satisfy $|\mu|\le n_{max}$ with array position $\mu+n_{max}$. The initial position follows the stationary density of one period. Define $F_{\mu\nu}(q,t)=\langle e^{-i(q+Q_1\mu)x(t)}e^{i(q+Q_1\nu)x(0)}\rangle$, so $F_{00}(q,t)$ is the characteristic function of $\Delta x=x(t)-x(0)$. Its logarithm defines displacement cumulants $\kappa_j$ through powers of $-iq$. Use $v=\lim_{t\to\infty}\langle\Delta x\rangle/t$, $D_\infty=\lim_{t\to\infty}\kappa_2/(2t)$, $\mu_\infty=\partial v/\partial F$ at fixed $u$, $D(t)=(1/2)d\kappa_2/dt$, $\mathrm{Skew}(t)=\kappa_3/\kappa_2^{3/2}$, and $\alpha_2(t)=\kappa_4/(3\kappa_2^2)$.

Implement the eight named functions according to their individual signatures: wb_operator, wb_bands, wb_stationary, wb_generalized_isf, wb_transport, wb_cumulants, wb_shape, and wb_audit. Return separate real and imaginary parts for complex outputs. All outputs must be finite, deterministic float64 values. Validate inputs as stated in each signature and raise ValueError for invalid inputs. The orchestrator wb_audit must call the earlier functions and enforce its internal consistency checks. Use $t_{ref}=1/(4\pi^2u)$ and $q=\pi$ for reference observables. Peak convention: global maxima on $10^{-4}\le t\le10$, using an 80-point logarithmic grid and convergence refinement.

Evaluate wb_audit at $u=10$, $f/u=0.9$, and $n_{max}=20$. In your reasoning, reconstruct the connection between the paper's dynamical generator and the reported observables. Establish the spectral, stationary-state, and scattering-function conventions and their limiting cases; derive the reduced-to-full displacement statistics through order four; and check transport independently. Audit the paper's displayed formulas against normalization, symmetry, limiting cases, and direct numerical evaluation. Identify and quantify any discrepancy you find, explain the mobility comparison, and show why both reported peaks are global and numerically stable.

Report $v$, $D_\infty/D$, $k_BT\mu_\infty/D$, and their Einstein ratio; the complex stationary coefficient $\langle1|r_{00}\rangle$ and $F_{10}(\pi,t_{ref})$; $D(t_{ref})/D$, $\mathrm{Skew}(t_{ref})$, and $\alpha_2(t_{ref})$; the peak skewness and its time; and the peak $\alpha_2$ and its time. Also report both peaks at $f/u=0.8$ for $u=10$, and compare the peak times with harmonic relaxation and the peak non-Gaussianity at tilt ratios $0.8$, $0.9$, and $1.0$.

The final answer is the peak $\alpha_2$ of the displacement at $u=10$ and $f/u=0.9$, to five significant figures.

## Output format

```
Output Format Requirements:
Emit <final_answer> immediately, then <reasoning>. Do not write a long derivation before the tags.
You must emit exactly one finite decimal inside <final_answer>...</final_answer>, even if the value is approximate or you are unsure.
Rules:
- The tags are required. Do not omit them or leave them empty.
- The tagged value must be a finite decimal (examples: 0.4847, 0.22, 1.05). Not NaN, not Inf, not a fraction string, not a vector, not prose.
- Put only that one number between the tags. No units, no words, no extra lines.
Include enough intermediate calculations and source justifications to support the quantities the prompt asks you to report; do not pad the response with a general pipeline summary.
Do not paste the input matrices, full coefficient vectors, per-iteration paths, or per-fold candidate tables.
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

wb_operator

Goal
----
Builds the truncated Fourier-basis matrix of the Bloch-Fokker-Planck operator of an overdamped particle in a tilted cosine potential.

```python
import numpy as np


def wb_operator(u: float, f: float, q: float, n_max: int) -> np.ndarray:
    r"""u: non-negative float, the corrugation amplitude $U_1/k_BT$. f: float, the reduced tilt $FL/(2\pi k_BT)$; a
    positive f drives the particle towards $+x$. q: float, the Bloch wave number in units of $1/L$. n_max: non-negative
    integer, the Fourier truncation.

    Units are $L = 1$, $D = 1$, $k_BT = 1$ throughout. The particle diffuses in the tilted cosine potential
    $\beta U(x) = u\cos(2\pi x) - 2\pi f x$ under the Smoluchowski equation $\partial_t p = \partial_x[\partial_x p +
    p\,\partial_x(\beta U)]$. On Bloch functions $e^{iqx}w(x)$ with $w(x) = \sum_\nu \langle\nu|w\rangle e^{2\pi i\nu x}$ the
    operator acts as a matrix $\langle\mu|\mathcal L_q|\nu\rangle$ on the Fourier coefficients of $w$, tridiagonal
    in $\mu-\nu$, for $\mu, \nu = -n_{max}..n_{max}$.

    Returns a numpy float64 array of shape $(2, N, N)$ with $N = 2n_{max}+1$: the real part and the imaginary part of the
    truncated matrix, row and column index $\mu + n_{max}$.

    Raises:
        ValueError: on a non-finite input, a negative u, or a non-integer or negative n_max.
    """
    return None
```

### Step 2

wb_bands

Goal
----
Evaluates the lowest band of the Bloch-Fokker-Planck operator across the interior of the Brillouin zone.

```python
import numpy as np


def wb_bands(u: float, f: float, q_values: np.ndarray, n_max: int) -> np.ndarray:
    r"""u, f: as in the operator step. q_values: one-dimensional array of Bloch wave numbers with $|q| < \pi$ (the
    interior of the first Brillouin zone in units of $1/L$). n_max: positive integer truncation.

    For each q, the eigenvalues $\lambda$ of $-\mathcal L_q$ have non-negative real parts and the one with the smallest
    real part, $\lambda_0(q)$, is the lowest band, which contains $\lambda_0(0) = 0$ and is unique for $|q| < \pi$.

    Returns a numpy float64 array of shape $(2, n_q)$: the real parts and the imaginary parts of $\lambda_0(q)$ in units
    of $D/L^2$, in the order of q_values.

    Raises:
        ValueError: on a non-finite input, a negative u, an empty q_values, any $|q| \ge \pi$, or a non-integer or
            non-positive n_max.
    """
    return None
```

### Step 3

wb_stationary

Goal
----
Computes the Fourier coefficients of the stationary periodic density of the tilted cosine potential.

```python
import numpy as np


def wb_stationary(u: float, f: float, n_max: int) -> np.ndarray:
    r"""u, f: as in the operator step. n_max: non-negative integer truncation.

    The stationary periodic density $p_{st}(x)$ of the tilted cosine potential, normalised to unit integral over one
    period, has Fourier coefficients $\langle\mu|r_{00}\rangle = \int_0^1 e^{-2\pi i\mu x}p_{st}(x)\,dx$ (with $L = 1$),
    normalized so that $\langle 0|r_{00}\rangle = 1$.

    Returns a numpy float64 array of shape $(2, N)$, $N = 2n_{max}+1$: the real parts and the imaginary parts of
    $\langle\mu|r_{00}\rangle$ for $\mu = -n_{max}..n_{max}$, index $\mu + n_{max}$.

    Raises:
        ValueError: on a non-finite input, a negative u, or a non-integer or negative n_max.
    """
    return None
```

### Step 4

wb_generalized_isf

Goal
----
Evaluates the generalized intermediate scattering function of two density modes differing by a reciprocal-lattice vector from the propagated stationary vector.

```python
import numpy as np


def wb_generalized_isf(u: float, f: float, q: float, mu: int, nu: int, t: float, n_max: int) -> np.ndarray:
    r"""u, f, q: as in the operator step. mu, nu: integers with $|\mu|, |\nu| \le n_{max}$, the reciprocal-lattice
    indices. t: non-negative float, the time in units of $L^2/D$. n_max: positive integer truncation.

    The generalized intermediate scattering function is $F_{\mu\nu}(q,t) = \langle e^{-i(q+2\pi\mu)x(t)}\,e^{i(q+2\pi\nu)x(0)}
    \rangle$ with $x(0)$ drawn from the stationary density of one period. It is evaluated from the Bloch operator of the first step
    and the stationary Fourier coefficients of the third step, with no sampling (units $L = 1$), so that $F_{\mu\nu}(q,0) = \langle\mu-\nu|r_{00}\rangle$ and the diagonal $F_{\mu\mu}(q,0) = 1$.

    Returns a numpy float64 array of shape $(2,)$: the real part and the imaginary part of $F_{\mu\nu}(q,t)$.

    Raises:
        ValueError: on a non-finite input, a negative u, a negative t, non-integer mu or nu, $|\mu|$ or $|\nu|$ above
            n_max, or a non-integer or non-positive n_max.
    """
    return None
```

### Step 5

wb_transport

Goal
----
Evaluates the drift velocity, the long-time diffusivity and the mobility of the driven particle and their Einstein ratio.

```python
import numpy as np


def wb_transport(u: float, f: float, n_max: int) -> np.ndarray:
    r"""u, f: as in the operator step. n_max: positive integer truncation.

    The long-time transport coefficients of the displacement $\Delta x(t) = x(t) - x(0)$: the drift velocity
    $v = \lim_{t\to\infty}\langle\Delta x\rangle/t$, the long-time diffusivity $D_\infty = \lim_{t\to\infty}\mathrm{Var}[\Delta x]/2t$
    and the mobility $\mu_\infty = \partial v/\partial F$ at fixed corrugation, with $F = 2\pi f k_BT/L$ so that
    $k_BT\mu_\infty = (1/2\pi)\,\partial v/\partial f$. All three are exact spectral quantities of $\mathcal L_0$ and of
    $\mathcal L_1 = \partial\mathcal L_q/\partial q|_{q=0}$ and must be evaluated to $10^{-10}$ absolute.

    Returns a numpy float64 array of shape $(4,)$: $v$ in units of $D/L$, $D_\infty/D$, $k_BT\mu_\infty/D$ and the
    Einstein ratio $D_\infty/(k_BT\mu_\infty)$.

    Raises:
        ValueError: on a non-finite input, a negative u, or a non-integer or non-positive n_max.
    """
    return None
```

### Step 6

wb_cumulants

Goal
----
Evaluates the first four cumulants of the displacement of the driven particle at a given time from the wave-number dependence of the propagator.

```python
import numpy as np


def wb_cumulants(u: float, f: float, t: float, n_max: int) -> np.ndarray:
    r"""u, f: as in the operator step. t: positive float, the time in units of $L^2/D$. n_max: positive integer
    truncation.

    The cumulants $\kappa_j$ of the displacement $\Delta x(t) = x(t) - x(0)$ with $x(0)$ stationary are defined by the
    characteristic function $\langle e^{-iq\Delta x(t)}\rangle = F_{00}(q,t)$ through $\ln F_{00}(q,t) = \sum_{j\ge1}(-iq)^j\kappa_j/j!$. They must be evaluated exactly, to
    $10^{-10}$ absolute, from the dependence of the propagator on q at $q = 0$; finite differences in q are not accurate
    enough for the fourth cumulant.

    Returns a numpy float64 array of shape $(4,)$: $\kappa_1, \kappa_2, \kappa_3, \kappa_4$ in units of $L^j$.

    Raises:
        ValueError: on a non-finite input, a negative u, a non-positive t, or a non-integer or non-positive n_max.
    """
    return None
```

### Step 7

wb_shape

Goal
----
Evaluates the time-dependent diffusivity, the skewness and the non-Gaussian parameter of the displacement at a given time.

```python
import numpy as np


def wb_shape(u: float, f: float, t: float, n_max: int) -> np.ndarray:
    r"""u, f, t, n_max: as in the cumulants step.

    Returns a numpy float64 array of shape $(3,)$: the time-dependent diffusivity $D(t)/D$ with $D(t) = \tfrac12\,
    d\kappa_2/dt$ evaluated exactly (not by a finite difference in t); the skewness $\mathrm{Skew}(t) = \kappa_3/\kappa_2^{3/2}$;
    and the non-Gaussian parameter $\alpha_2(t) = \kappa_4/(3\kappa_2^2)$, both built from the cumulants of the
    displacement.

    Raises:
        ValueError: on any invalid argument of the cumulants step.
    """
    return None
```

### Step 8

wb_audit

Goal
----
Runs the whole chain: operator, lowest band, stationary density, generalized scattering function, transport coefficients, cumulants and shape parameters, with the band and the variance slope as checks, and locates the peak skewness and the peak non-Gaussian parameter.

```python
import numpy as np


def wb_audit(u: float, f_over_u: float, n_max: int) -> np.ndarray:
    r"""u: positive float, the corrugation amplitude. f_over_u: non-negative float, the tilt-to-amplitude ratio $f/u$.
    n_max: integer of at least 12, the Fourier truncation used throughout.

    Uses the reference time $t_{ref} = 1/(4\pi^2 u)$ (in units of $L^2/D$, the curvature time of an untilted well) and
    the zone-edge wave number $q = \pi$. Returns a numpy float64 array of shape $(13,)$: the drift velocity $v$; the
    long-time diffusivity $D_\infty/D$; the mobility $k_BT\mu_\infty/D$; the Einstein ratio $D_\infty/(k_BT\mu_\infty)$;
    the real and imaginary parts of the stationary coefficient $\langle 1|r_{00}\rangle$; the real and imaginary parts
    of $F_{10}(\pi, t_{ref})$; $D(t_{ref})/D$; $\mathrm{Skew}(t_{ref})$; $\alpha_2(t_{ref})$; the peak skewness
    $\max_t\mathrm{Skew}(t)$; and the peak non-Gaussian parameter $\max_t\alpha_2(t)$, both maxima taken over
    $10^{-4} \le t \le 10$ (located on a logarithmic grid of 80 points and refined to convergence). The chain is
    checked on the way: the $\mu = 0$ row of the operator must vanish (probability conservation); the drift velocity
    must agree with the slope of the imaginary part of the lowest band at $q = 0$ and the long-time diffusivity
    with half the curvature of its real part, both within $10^{-5}$ relative (central differences with step $10^{-3}$);
    the stationary coefficients must satisfy $\langle-\mu|r_{00}\rangle = \langle\mu|r_{00}\rangle^*$ and the density
    they build must be non-negative on a grid of 400 points; $F_{11}(\pi, 0)$ must be one within $10^{-10}$;
    $\kappa_1(t_{ref})$ must equal $v\,t_{ref}$ within $10^{-8}$ relative; and $D(t_{ref})$ must agree with a central
    difference of $\kappa_2/2$ (step $10^{-6}t_{ref}$) within $10^{-6}$ relative.

    Raises:
        ValueError: on a non-finite or non-positive u, a negative f_over_u, an n_max below 12, a failed check, or a
            peak that is not interior to the search window.
    """
    return None
```
