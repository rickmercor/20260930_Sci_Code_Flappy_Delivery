# Physics-Particle_Physics-17

## Background

Heavy quarkonia ($c\bar{c}$ and $b\bar{b}$ bound states) are a benchmark for understanding hadron formation in QCD. In non-relativistic QCD (NRQCD) factorization, the production cross section is a sum over intermediate $Q\bar{Q}$ configurations, each written as a perturbative short-distance coefficient times a non-perturbative long-distance matrix element (LDME). The LDMEs cannot be computed and must be extracted from global fits to measured $p_{\mathrm{T}}$-differential cross sections and polarizations, so precise cross sections over a wide $p_{\mathrm{T}}$ range directly constrain the theory. For S-wave states the leading colour-octet contributions are the ${}^1S_0^{(8)}$, ${}^3S_1^{(8)}$ and ${}^3P_J^{(8)}$ terms.

At hadron colliders the $\Upsilon(1S)$, $\Upsilon(2S)$ and $\Upsilon(3S)$ are reconstructed in the $\mu^+\mu^-$ channel, where the three states appear as closely spaced peaks between $9.4$ and $10.4$ GeV above a smooth continuum. Their yields are extracted by fitting the dimuon invariant-mass spectrum with resonance line shapes that combine a Gaussian core, set by the tracker momentum resolution, and a low-mass radiative tail from final-state photon emission. The yields are then corrected for luminosity, bin widths, detection efficiency and geometric acceptance.

The acceptance depends on the dimuon decay angular distribution, $W(\cos\theta)\propto1+\lambda_\theta\cos^2\theta$ in the helicity frame. Its polar anisotropy $\lambda_\theta$ ranges from $+1$ (fully transverse) to $-1$ (fully longitudinal). Experiments therefore quote cross sections for an assumed polarization, usually unpolarized, together with factors that let users re-interpret the results for other polarization hypotheses.

## Problem

Production cross sections of the $\Upsilon(1S)$, $\Upsilon(2S)$, and $\Upsilon(3S)$ bottomonium states are key inputs to NRQCD global fits of long-distance matrix elements, and they have recently been measured in pp collisions at $\sqrt{s}=13.6$ TeV with $37.4\ \mathrm{fb}^{-1}$ of 2022 data; that publication is referred to below as the paper. In each $(p_{\mathrm{T}}, |y|)$ bin the analysis extracts the three signal yields from a binned extended maximum-likelihood fit to the dimuon invariant-mass spectrum and converts them into $p_{\mathrm{T}}$-differential cross sections times dimuon branching fraction, per unit rapidity, under the assumption of unpolarized production. The input to your computation is a pseudo-data dimuon mass histogram for one kinematic bin, and the output is a polarization-rescaled $\Upsilon(3S)$ cross section.

The core of the problem is the paper's constrained three-resonance fit model, in which each $\Upsilon(nS)$ peak is a weighted sum of two Crystal Ball functions with their power-law tails on the low-mass side, with the tail parameters, width ratio, component weight, inter-peak mass offsets, and width scaling across the three states fixed exactly as in the paper, together with its second-order-polynomial background and the same set of free parameters. Here a Crystal Ball function of $t=(m-\mu)/\sigma$ means $\exp(-t^2/2)$ for $t>-\alpha$ and $A(B-t)^{-n}$ for $t\le-\alpha$, with $A=(n/\alpha)^n\exp(-\alpha^2/2)$ and $B=n/\alpha-\alpha$, so that the function and its first derivative are continuous at $t=-\alpha$. Every signal and background shape must be normalized to unit integral over the $8.5$–$11.5$ GeV fit window, and the expected content of each histogram bin is the integral of the model over that bin. Yields are converted to $\mathcal{B}\,d^2\sigma/(dp_{\mathrm{T}}\,dy)$ with the paper's master formula and rapidity-interval convention, and the unpolarized result is then rescaled to $\lambda_\theta^{\mathrm{HX}}=-0.5$ using the paper's Table 1 factors for this bin, which are rounded to two decimals and therefore not exactly mutually consistent: take their ratio $k(+1)/k(-1)$ as exact, require the correction to be exactly 1 for unpolarized production, and derive its $\lambda_\theta$ dependence from the dimuon decay angular distribution rather than by linear interpolation in $\lambda_\theta$.

Use the following configuration:

- Kinematic bin: $70<p_{\mathrm{T}}<100$ GeV, $|y|<0.6$; integrated luminosity $37.4\ \mathrm{fb}^{-1}$.
- Histogram: 75 bins of 40 MeV spanning $8.5$–$11.5$ GeV.
- Pseudo-data generator (this is not the fit model): the expected count in each bin is $\sum_{nS}N_{nS}\,G_{nS}+N_{\mathrm{bkg}}\,E$, where $G_{nS}$ is the integral over the bin of a Gaussian with mean $m_{nS}-0.012$ GeV and standard deviation $w_{nS}$, and $E$ is the integral over the bin of a density $\propto\exp(-0.4\,m/\mathrm{GeV})$, normalized on $8.5$–$11.5$ GeV, with $N_{1S}=2800$, $N_{2S}=1300$, $N_{3S}=900$, $N_{\mathrm{bkg}}=7000$, $w_{1S}=0.072$ GeV, $w_{2S}=0.077$ GeV, $w_{3S}=0.080$ GeV, and $m_{nS}$ the world-average $\Upsilon(nS)$ masses $m_{1S}=9.46040$ GeV, $m_{2S}=10.0234$ GeV and $m_{3S}=10.3551$ GeV (the same values enter the paper's inter-peak mass constraint).
- Observed counts: a single call `rng.poisson(expected)` on the 75-element expected-count array ordered by increasing mass, with `rng = np.random.default_rng(13600)`.
- Dimuon efficiency inputs: single-muon efficiencies $0.92$ and $0.90$, and a muon-pair (close-pair trigger and vertex) correction factor of $0.95$.
- Unpolarized acceptance: $0.38$.
- Target polarization: $\lambda_\theta^{\mathrm{HX}}=-0.5$.

Fit the pseudo-data with the paper's model, extract the $\Upsilon(3S)$ yield, compute its unpolarized cross section times branching fraction, and rescale it to the target polarization. Your final answer must be a single number: $\mathcal{B}\,d^2\sigma/(dp_{\mathrm{T}}\,dy)$ of the $\Upsilon(3S)$ for this bin under $\lambda_\theta^{\mathrm{HX}}=-0.5$, in pb/GeV.

Output Format Requirements:
Emit <final_answer> immediately, then <reasoning>. Do not write a long derivation before the tags.
You must emit exactly one finite decimal inside
<final_answer>...</final_answer>, even if the value is approximate or you are unsure.
Rules:
- The tags are required. Do not omit them or leave them empty.
- The tagged value must be a finite decimal (examples: 0.4847, 0.22, 1.05).
Not NaN, not Inf, not a fraction string, not a vector, not prose.
- Put only that one number between the tags. No units, no words, no extra lines.
Keep <reasoning> short (a few hundred words). Include enough intermediate calculations to justify the deterministic computation, without turning the response into a general pipeline summary.
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

crystal_ball_bin_fractions

Goal
----
Compute the fraction of a single Crystal Ball (CB) line shape that falls in each bin of a dimuon invariant-mass histogram, with the CB normalized to unit integral over the histogram window $[e_0, e_K]$, where $e_0<e_1<\dots<e_K$ are the bin edges (`edges[0]` to `edges[-1]`).



The CB function has a Gaussian core and a power-law tail on the low-mass side:



$$t=\dfrac{m-\mu}{\sigma},\qquad \mathrm{CB}(t)=\exp\Big(-\dfrac{t^2}{2}\Big)\ \ \mathrm{for}\ t>-\alpha,\qquad \mathrm{CB}(t)=A\,(B-t)^{-n}\ \ \mathrm{for}\ t\le-\alpha,$$



$$A=\Big(\dfrac{n}{\alpha}\Big)^{n}\exp\Big(-\dfrac{\alpha^2}{2}\Big),\qquad B=\dfrac{n}{\alpha}-\alpha.$$



The bin fractions are exact integrals over each bin divided by the exact integral over the window, and they must stay accurate to about $10^{-9}$ relative precision per bin in every regime the fit can visit: windows lying many $\sigma$ above the mean ($|t|$ up to about 30), windows entirely inside the power-law tail, and every $n>0$, including $n=1$ and values of $n$ arbitrarily close to (but not equal to) 1. The window integral is finite only because the window is finite.

```python
import numpy as np
from scipy.special import erf, erfc

def crystal_ball_bin_fractions(edges: np.ndarray, mu: float, sigma: float,
                               alpha: float, n: float) -> np.ndarray:
    r'''Window-normalized bin fractions of a low-side-tail Crystal Ball shape.

    Parameters
    ----------
    edges : np.ndarray
        1D array of $K+1$ strictly increasing, finite bin edges
        $e_0<\dots<e_K$ (GeV), with $K\geq1$.
    mu : float
        Peak position $\mu$ of the Gaussian core (GeV).
    sigma : float
        Width $\sigma>0$ of the Gaussian core (GeV).
    alpha : float
        Transition point $\alpha>0$ of the power-law tail, in units of
        $\sigma$; the tail is on the low-mass side, $t\leq-\alpha$ with
        $t=(m-\mu)/\sigma$.
    n : float
        Power-law exponent $n>0$ of the tail; $n=1$ and $n$ arbitrarily
        close to 1 must be supported.

    Returns
    -------
    fractions : np.ndarray
        Shape $(K,)$: the integral of the CB over each bin divided by its
        integral over $[e_0, e_K]$; the entries sum to 1. Each entry must be
        accurate to about $10^{-9}$ relative precision, also when the window
        lies far in the Gaussian upper tail ($|t|$ up to about 30).

    Raises
    ------
    ValueError
        If edges is not a 1D finite array of length $\geq2$ that is strictly
        increasing, if mu is not finite, if $\sigma\leq0$, $\alpha\leq0$ or
        $n\leq0$ (or any of them is not finite), or if the integral of the
        shape over the window underflows to zero in double precision.
    '''
    return fractions  # placeholder
```

### Step 2

upsilon_signal_expected_counts

Goal
----
Compute the expected per-bin signal counts of the three Upsilon(nS) resonances in one $(p_{\mathrm{T}}, |y|)$ bin, using the signal model of the CMS 13.6 TeV measurement.



Each resonance is described by the paper's per-state line shape, built from Crystal Ball functions (see $crystal_ball_bin_fractions$), and the three resonances are tied together by the paper's inter-state constraints, so that the only free signal parameters are the three yields and the mean $\mu_1$ and width $\sigma_1$ of the Upsilon(1S) line shape. The shape constants that the paper keeps fixed in the fit are stated in the background; none of them is an input. All shapes are normalized to unit integral on the histogram window $[e_0, e_K]$ (`edges[0]` to `edges[-1]`), and counts are bin integrals.

```python
import numpy as np

def upsilon_signal_expected_counts(edges: np.ndarray, yields: np.ndarray, mu1: float,
                                   sigma1: float, masses: np.ndarray) -> np.ndarray:
    r'''Expected per-bin counts of the three $\Upsilon(nS)$ peaks in the CMS fit model.

    Parameters
    ----------
    edges : np.ndarray
        1D array of $K+1$ strictly increasing bin edges (GeV).
    yields : np.ndarray
        Shape $(3,)$, yields $N_{1S}, N_{2S}, N_{3S}$ of $\Upsilon(1S)$,
        $\Upsilon(2S)$ and $\Upsilon(3S)$, each $\geq0$.
    mu1 : float
        Mean $\mu_1$ of the $\Upsilon(1S)$ line shape (GeV).
    sigma1 : float
        Core width $\sigma_1>0$ of the narrowest Crystal Ball component of
        the $\Upsilon(1S)$ line shape (GeV).
    masses : np.ndarray
        Shape $(3,)$, increasing positive world-average masses
        $M_{1S}, M_{2S}, M_{3S}$ (GeV).

    Returns
    -------
    counts : np.ndarray
        Shape $(3,K)$. Row $s$ is $N_s$ times the window-normalized bin
        fractions of the paper's line shape for state $s$, with the paper's
        fixed shape constants and inter-state constraints.

    Raises
    ------
    ValueError
        If yields or masses do not have shape $(3,)$ or contain non-finite
        values, if any yield is negative, if masses are not positive and
        strictly increasing, or under any condition for which
        crystal_ball_bin_fractions raises (invalid edges, non-finite mu1,
        $\sigma_1\leq0$).
    '''
    return counts  # placeholder
```

### Step 3

quadratic_background_bin_fractions

Goal
----
Compute the continuum background of the CMS Upsilon fit: a second-order polynomial in the dimuon mass, normalized to unit integral on the histogram window $[e_0, e_K]$, where $e_0$ and $e_K$ are the first and last bin edges (`edges[0]` and `edges[-1]`). With the reduced variable



$$x=\dfrac{m-m_c}{h},\qquad m_c=\dfrac{e_0+e_K}{2},\qquad h=\dfrac{e_K-e_0}{2},$$



so that $x$ runs from $-1$ to $+1$, the density is



$$p(x)\propto1+b_1x+b_2x^2.$$



The exact bin integrals follow from the primitive $P(x)=x+b_1x^2/2+b_2x^3/3$. Because the fit maximizes a likelihood, the choice of this parametrization of the quadratic family does not change the fitted yields.

```python
import numpy as np

def quadratic_background_bin_fractions(edges: np.ndarray, b1: float, b2: float) -> np.ndarray:
    r'''Window-normalized bin fractions of $p(x)\propto1+b_1x+b_2x^2$.

    Parameters
    ----------
    edges : np.ndarray
        1D array of $K+1$ strictly increasing, finite bin edges (GeV), with
        $K\geq1$. The reduced variable is $x=(m-m_c)/h$, with $m_c$ the
        window centre and $h$ the window half-width.
    b1 : float
        Linear coefficient $b_1$.
    b2 : float
        Quadratic coefficient $b_2$.

    Returns
    -------
    fractions : np.ndarray
        Shape $(K,)$, the exact integral of the polynomial over each bin
        divided by its integral over the whole window; the entries sum to 1.
        Only bin integrals are constrained: the polynomial itself may vanish
        or dip below zero inside a bin as long as that bin's integral stays
        positive.

    Raises
    ------
    ValueError
        If edges is not a finite, strictly increasing 1D array of length
        $\geq2$, if b1 or b2 is not finite, or if the integral of the
        polynomial over any bin is $\leq0$ (the background density must be
        positive in every bin).
    '''
    return fractions  # placeholder
```

### Step 4

generate_pseudo_data

Goal
----
Generate the deterministic pseudo-data dimuon mass histogram of the task. The generator is deliberately not the fit model: each Upsilon(nS) is a plain Gaussian (no radiative tail) and the continuum is a falling exponential. With bin edges $e_0<e_1<\dots<e_K$, the expected count in bin $i$ is



$$\nu_i=\sum_s N_s\Big[\Phi\Big(\dfrac{e_{i+1}-m_s}{w_s}\Big)-\Phi\Big(\dfrac{e_i-m_s}{w_s}\Big)\Big]+N_{\mathrm{bkg}}\,\dfrac{e^{-c\,e_i}-e^{-c\,e_{i+1}}}{e^{-c\,e_0}-e^{-c\,e_K}},$$



where $\Phi$ is the standard normal distribution function. The Gaussian integrals are not renormalized to the window; the exponential is normalized on the window. Every expected count must be computed to full relative precision, including bins many standard deviations away from a peak: an expectation that is tiny but non-zero and one that is exactly zero are not equivalent inputs to `rng.poisson` (a zero rate does not consume a random draw), so they lead to different pseudo-data. The observed counts are drawn with exactly one call `rng.poisson(nu)` on the whole array of $\nu_i$ (ordered by increasing mass), with `rng = np.random.default_rng(seed)`.

```python
import numpy as np
from scipy.special import erf, erfc

def generate_pseudo_data(edges: np.ndarray, yields: np.ndarray, means: np.ndarray,
                         widths: np.ndarray, n_bkg: float, exp_slope: float,
                         seed: int) -> np.ndarray:
    r'''Poisson pseudo-data from three Gaussian peaks plus an exponential continuum.

    Parameters
    ----------
    edges : np.ndarray
        1D array of $K+1$ strictly increasing, finite bin edges (GeV).
    yields : np.ndarray
        Shape $(3,)$, expected numbers of events $N_s\geq0$ of each Gaussian
        peak, applied to the untruncated Gaussian integral over each bin.
    means : np.ndarray
        Shape $(3,)$, Gaussian means $m_s$ (GeV).
    widths : np.ndarray
        Shape $(3,)$, Gaussian standard deviations $w_s>0$ (GeV).
    n_bkg : float
        Expected number $N_{\mathrm{bkg}}\geq0$ of background events in the
        window.
    exp_slope : float
        Slope $c>0$ of the background density $\propto e^{-c\,m}$, in
        $\mathrm{GeV^{-1}}$.
    seed : int
        Seed for np.random.default_rng.

    Returns
    -------
    counts : np.ndarray
        Shape $(K,)$, int64, drawn as rng.poisson(nu) in a single call on the
        array of expected counts $\nu_i$ ordered by increasing mass, where
        every $\nu_i$ is accurate to full relative precision (also in far
        Gaussian tails).

    Raises
    ------
    ValueError
        If edges is not a finite strictly increasing 1D array of length
        $\geq2$, if yields, means or widths do not have shape $(3,)$ or
        contain non-finite values, if any yield is negative or any width is
        $\leq0$, if $N_{\mathrm{bkg}}<0$, or if $c\leq0$.
    '''
    return counts  # placeholder
```

### Step 5

fit_upsilon_mass_spectrum

Goal
----
Extract the Upsilon(nS) yields from a dimuon invariant-mass histogram with the fit of the CMS 13.6 TeV analysis: the signal model of $upsilon_signal_expected_counts$ plus the continuum background of $quadratic_background_bin_fractions$, fitted with the statistical method used in the paper. The free parameters, in this order, are



$$[N_{1S},\ N_{2S},\ N_{3S},\ N_{\mathrm{bkg}},\ \mu_1,\ \sigma_1,\ b_1,\ b_2],$$



where $N_{\mathrm{bkg}}$ is the background yield in the window. Parameter points where the model is undefined (it raises `ValueError`) are excluded from the fit. The global optimum of the paper's fit statistic is unique for the task's data; any minimizer that reaches it is acceptable.

```python
import numpy as np
from scipy.optimize import minimize

def fit_upsilon_mass_spectrum(counts: np.ndarray, edges: np.ndarray, masses: np.ndarray,
                              start: np.ndarray) -> np.ndarray:
    r'''Fit the CMS three-$\Upsilon$ plus background model to a mass histogram.

    Parameters
    ----------
    counts : np.ndarray
        Shape $(K,)$, observed non-negative bin contents; non-integer values
        (for example Asimov data) are allowed.
    edges : np.ndarray
        Shape $(K+1,)$, strictly increasing bin edges (GeV).
    masses : np.ndarray
        Shape $(3,)$, world-average $\Upsilon(nS)$ masses (GeV).
    start : np.ndarray
        Shape $(8,)$, starting point
        $[N_{1S}, N_{2S}, N_{3S}, N_{\mathrm{bkg}}, \mu_1, \sigma_1, b_1, b_2]$
        at which the model is defined.

    Returns
    -------
    best_params : np.ndarray
        Shape $(8,)$, parameters at the optimum of the paper's fit statistic.

    Raises
    ------
    ValueError
        If counts is not 1D with len(edges) $-\,1$ finite non-negative
        entries, if start does not have shape $(8,)$ or is not finite, or if
        the model is undefined at the starting point (negative
        $N_{\mathrm{bkg}}$, invalid signal or background parameters, or a
        non-positive expected count in any bin).
    '''
    return best_params  # placeholder
```

### Step 6

cross_section_times_bf

Goal
----
Convert a fitted Upsilon signal yield in one $(p_{\mathrm{T}}, |y|)$ bin into the quantity reported by the CMS 13.6 TeV analysis: the $p_{\mathrm{T}}$-differential cross section times dimuon branching fraction, per unit rapidity, $\mathcal{B}\,d^2\sigma/(dy\,dp_{\mathrm{T}})$, in pb/GeV, as defined by eq. (1.1) of the paper. The luminosity units, the rapidity interval associated with an $|y|$ bin, and the way the single-muon efficiencies and the muon-pair correction factor enter the dimuon efficiency follow the paper and are stated in the background.

```python
import numpy as np

def cross_section_times_bf(n_signal: float, lumi_fb_inv: float, pt_min: float, pt_max: float,
                           abs_y_min: float, abs_y_max: float, eff_mu1: float, eff_mu2: float,
                           rho_pair: float, acceptance: float) -> float:
    r'''$\mathcal{B}\,d^2\sigma/(dy\,dp_{\mathrm{T}})$ in pb/GeV for one $(p_{\mathrm{T}}, |y|)$ bin.

    Parameters
    ----------
    n_signal : float
        Fitted signal yield $N\geq0$.
    lumi_fb_inv : float
        Integrated luminosity $\mathcal{L}>0$ in $\mathrm{fb^{-1}}$.
    pt_min, pt_max : float
        $p_{\mathrm{T}}$ bin edges (GeV), with $0\leq$ pt_min $<$ pt_max.
    abs_y_min, abs_y_max : float
        Edges of the $|y|$ (absolute rapidity) bin, with
        $0\leq$ abs_y_min $<$ abs_y_max.
    eff_mu1, eff_mu2 : float
        Single-muon efficiencies $\epsilon_{\mu1}$ and $\epsilon_{\mu2}$ of the
        two muons, each in $(0,1]$.
    rho_pair : float
        Muon-pair (close-pair trigger and vertex) correction factor
        $\rho_{\mathrm{pair}}$, in $(0,1]$.
    acceptance : float
        Dimuon acceptance $A$, in $(0,1]$.

    Returns
    -------
    xsec : float
        $\mathcal{B}\,d^2\sigma/(dy\,dp_{\mathrm{T}})$ as defined by the
        paper's eq. (1.1), native Python float in pb/GeV.

    Raises
    ------
    ValueError
        If any input is not finite, if $N<0$, $\mathcal{L}\leq0$,
        pt_min $<0$, pt_max $\leq$ pt_min, abs_y_min $<0$,
        abs_y_max $\leq$ abs_y_min, or if eff_mu1, eff_mu2, rho_pair or
        acceptance is outside $(0,1]$.
    '''
    return xsec  # placeholder
```

### Step 7

polarization_scale_factor.py

Goal
----
Rescale an unpolarized Upsilon cross section to an arbitrary polar anisotropy $\lambda_\theta$ in the helicity frame ($\lambda_\phi=\lambda_{\theta\phi}=0$), using the two extreme-scenario factors of the paper's Table 1: $k_+\approx A_0/A_{+1}$ (fully transverse, $\lambda_\theta=+1$) and $k_-\approx A_0/A_{-1}$ (fully longitudinal, $\lambda_\theta=-1$), where $A_\lambda$ is the dimuon acceptance of the bin when the parent is produced with decay distribution



$$W(\cos\theta)\propto1+\lambda_\theta\cos^2\theta.$$

```python
import numpy as np

def polarization_scale_factor(lambda_theta: float, k_plus: float, k_minus: float) -> float:
    r'''Cross-section scaling factor $\sigma(\lambda_\theta)/\sigma_{\mathrm{unpol}}$.

    Parameters
    ----------
    lambda_theta : float
        Polar anisotropy $\lambda_\theta$ in the helicity frame,
        $-1\leq\lambda_\theta\leq1$.
    k_plus : float
        Rounded Table 1 factor $k_+>0$ for $\lambda_\theta=+1$.
    k_minus : float
        Rounded Table 1 factor $k_->0$ for $\lambda_\theta=-1$.

    Returns
    -------
    k : float
        Native Python float $\sigma(\lambda_\theta)/\sigma_{\mathrm{unpol}}$,
        with $k(0)=1$ and $k(+1)/k(-1)=k_+/k_-$ exactly.

    Raises
    ------
    ValueError
        If any input is not finite, if $\lambda_\theta$ lies outside
        $[-1,1]$, if $k_+\leq0$ or $k_-\leq0$, or if $k_+/k_->2$ (no physical
        acceptance can produce such a ratio).
    '''
    return k  # placeholder
```

### Step 8

upsilon_polarized_cross_section

Goal
----
ORCHESTRATOR. End-to-end reproduction of the task:



1. Edges: 75 bins of 40 MeV on $[8.5, 11.5]$ GeV; masses are the world-average values given in the task, $[9.46040, 10.0234, 10.3551]$ GeV.

2. Pseudo-data: $generate_pseudo_data$ with yields $[2800, 1300, 900]$, means equal to the masses minus 0.012 GeV, widths $[0.072, 0.077, 0.080]$ GeV, $N_{\mathrm{bkg}}=7000$, exponential slope $c=0.4\ \mathrm{GeV^{-1}}$, and the given seed.

3. Fit: $fit_upsilon_mass_spectrum$ (signal from $upsilon_signal_expected_counts$, background from $quadratic_background_bin_fractions$) from the start point $[2500, 1200, 800, 7000, 9.45, 0.06, 0.0, 0.0]$.

4. Cross section of the requested state with $cross_section_times_bf$: $\mathcal{L}=37.4\ \mathrm{fb^{-1}}$, $70<p_{\mathrm{T}}<100$ GeV, $|y|<0.6$, single-muon efficiencies 0.92 and 0.90, $\rho_{\mathrm{pair}}=0.95$, $A=0.38$.

5. Rescale to $\lambda_\theta$ with $polarization_scale_factor$, using the paper's Table 1 factors for this $(p_{\mathrm{T}}, |y|)$ bin, $k_+=1.12$ ($\lambda_\theta=+1$) and $k_-=0.83$ ($\lambda_\theta=-1$), which are not inputs of this function.



The task answer uses $state_index = 2$, $\lambda_\theta=-0.5$ and $seed = 13600$.

```python
import numpy as np

def upsilon_polarized_cross_section(lambda_theta: float = -0.5, seed: int = 13600,
                                    state_index: int = 2) -> float:
    r'''Full pipeline: pseudo-data, constrained fit, cross section, polarization.

    Parameters
    ----------
    lambda_theta : float
        Target helicity-frame polar anisotropy $\lambda_\theta\in[-1,1]$.
    seed : int
        Seed for the pseudo-data Poisson draw.
    state_index : int
        0, 1 or 2 for $\Upsilon(1S)$, $\Upsilon(2S)$ or $\Upsilon(3S)$.

    Returns
    -------
    xsec : float
        $\mathcal{B}\,d^2\sigma/(dy\,dp_{\mathrm{T}})$ of the requested state in
        pb/GeV for $70<p_{\mathrm{T}}<100$ GeV and $|y|<0.6$, under the
        requested polarization, as a native Python float.

    Raises
    ------
    ValueError
        If state_index is not one of 0, 1, 2, or if $\lambda_\theta$ is not
        finite or lies outside $[-1,1]$.
    '''
    return xsec  # placeholder
```
