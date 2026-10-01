# Physics-Particle_Physics-10

## Background

The total cross section of Compton scattering is a textbook result at leading order (the Klein–Nishina formula, which falls as $\ln(s/m^2)/s$ at high energy), and its one-loop QED corrections have been known for decades. Precision photon-beam experiments now measure the process at the percent level and better at photon energies of several GeV, where the perturbative series shows an unusual feature: the corrections contain powers of the large logarithm $\ln(s/m^2)$ that grow with the order, so that the size and convergence of higher-order terms were an open question. Modern multi-loop technology makes the two-loop total cross section with full mass dependence accessible, and an analysis of the momentum regions responsible for the logarithms allows their all-order structure to be understood and summed. Because a bremsstrahlung beam populates the tagged window with an intensity falling roughly as $1/E_\gamma$, the quantity compared with data is the spectrum-weighted mean of the prediction over the window; in the variable $\ln E_\gamma$ this weight is uniform and the integrand is smooth, so a modest Gauss–Legendre rule converges to machine precision.

## Problem

Compton scattering, $e^-\gamma\to e^-\gamma$, is one of the cleanest processes in quantum electrodynamics and serves as a luminosity and polarization monitor in fixed-target photon experiments, yet its total cross section at high energy has long been theoretically uncomfortable: the order-$\alpha^3$ correction is large (about 8 % at $\sqrt s=1$ GeV and growing with energy) because of double-logarithmic terms $\alpha\ln^2(s/m^2)$ that fixed-order perturbation theory does not control. A recent paper closes this gap: it computes the total cross section at order $\alpha^4$ (NNLO) with full electron-mass dependence, presents it for the high-energy region $\tau=m^2/s\le0.01$ as an expansion in $\tau$ and $L=\ln(1/\tau)$ whose coefficients are numerical, identifies the diagrammatic origin of the leading logarithms, derives a closed form for the leading-logarithmic term at every order and sums the series to all orders in closed form, and compares the result with tagged-photon data taken with a bremsstrahlung beam. Your task is to reproduce the quantity such an experiment actually measures: the paper's most precise prediction, the NNLO cross section matched to the all-order leading-logarithmic resummation, averaged over a tagged-photon window with the bremsstrahlung spectral weight. Use the paper's high-energy expansions of the leading, next-to-leading and next-to-next-to-leading orders truncated exactly as the paper gives them, its closed-form leading-logarithmic terms, and its resummed expression; match additively, $\sigma_{\rm NNLO+LL}(E_\gamma)=\sigma_{\rm LO}+\sigma_{\rm NLO}+\sigma_{\rm NNLO}+\bigl[\sigma_{\rm LL}-\sum_{n=1}^{3}\sigma^{(n)}_{\rm LL}\bigr]$, where $\sigma^{(n)}_{\rm LL}$ is the leading-logarithmic term of order $\alpha^{n+1}$, so that no logarithm is counted twice. The paper's expansion coefficients, its leading-logarithmic formula and its resummed expression are not restated here — see the paper for their exact form. Use the following configuration:

- Fixed electron target: laboratory photon energy $E_\gamma$, $s=m^2+2mE_\gamma$, $\tau=m^2/s$
- Tagged window $E_{\min}=6.5$ GeV to $E_{\max}=11.1$ GeV with spectral weight $w(E_\gamma)\propto1/E_\gamma$, i.e. $\bar\sigma=\int_{E_{\min}}^{E_{\max}}\sigma_{\rm NNLO+LL}(E_\gamma)\,\frac{dE_\gamma}{E_\gamma}\Big/\int_{E_{\min}}^{E_{\max}}\frac{dE_\gamma}{E_\gamma}$, evaluated with a quadrature converged to better than $10^{-9}$ relative
- Electron mass $m=0.51099895$ MeV and fine-structure constant $\alpha=1/137.035999084$ (both fixed; on-shell scheme, no running)
- Final states with three charged particles excluded, as in the paper
- Natural units throughout, converted at the end with $1\,{\rm GeV}^{-2}=389.3793721\ \mu{\rm b}$

Let $\bar\sigma$ denote the spectrum-averaged matched cross section so obtained. Compute $\bar\sigma$ in microbarn.

## Output format

```
## Output format

Emit <final_answer> immediately, then <reasoning>. Do not write a long derivation before the tags.
You must emit exactly one finite decimal inside <final_answer>...</final_answer>, even if the value is approximate or you are unsure.
Rules:
- The tags are required. Do not omit them or leave them empty.
- The tagged value must be a finite decimal (examples: 0.4847, -0.22, 1.05). Not NaN, not Inf, not a fraction string, not a vector, not prose.
- Put only that one number between the tags. No units, no words, no extra lines.
Keep <reasoning> concise but complete: show how the spectral average is set up and evaluated, the averaged leading-, next-to-leading- and next-to-next-to-leading-order contributions, the averaged resummed remainder beyond NNLO, the evidence that the quadrature is converged, and the final average, since these are exactly what determines the final number.
Do not re-paste the configuration; reference it only as needed for the computation above.
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

step_01_compton_kinematics_lo

Goal
----
Convert a fixed-target photon energy into the Mandelstam variable s and mass ratio tau, and evaluate the exact leading-order (Klein-Nishina) total Compton cross section.

```python
def compton_kinematics_lo(E_gamma: float, m: float, alpha: float) -> tuple:
    '''Kinematics and Klein-Nishina total cross section for fixed-target Compton scattering.

    Parameters
    ----------
    E_gamma : float
        Laboratory photon energy in GeV, > 0.
    m : float
        Electron mass in GeV, > 0.
    alpha : float
        Fine-structure constant, > 0.

    Returns
    -------
    result : tuple of (float, float, float)
        (s, tau, sigma_LO): s = m^2 + 2 m E_gamma in GeV^2, tau = m^2/s, and
        the Klein-Nishina total cross section in GeV^-2, as native Python
        floats.

    Raises
    ------
    ValueError
        If E_gamma, m or alpha is not strictly positive.
    '''
    return s, tau, sigma_LO  # placeholder
```

### Step 2

step_02_lo_high_energy_series

Goal
----
Evaluate the paper's high-energy leading-order Compton total cross section through the tau^4 term.

```python
def lo_high_energy_series(tau: float, s: float, alpha: float) -> float:
    '''High-energy expansion of the leading-order total cross section.

    Parameters
    ----------
    tau : float
        m^2/s, with 0 < tau <= 0.01 (the domain of the expansion).
    s : float
        Mandelstam s in GeV^2, > 0.
    alpha : float
        Fine-structure constant, > 0.

    Returns
    -------
    sigma_LO : float
        The truncated high-energy expansion of the leading-order cross
        section in GeV^-2, as a native Python float.

    Raises
    ------
    ValueError
        If tau is outside (0, 0.01], or if s or alpha is not strictly positive.
    '''
    return sigma_LO  # placeholder
```

### Step 3

step_03_nlo_high_energy_series

Goal
----
Evaluate the paper's high-energy next-to-leading-order (order alpha^3) correction to the Compton total cross section, through the tau^3 term.

```python
def nlo_high_energy_series(tau: float, s: float, alpha: float) -> float:
    '''High-energy expansion of the order-alpha^3 correction.

    Parameters
    ----------
    tau : float
        m^2/s, with 0 < tau <= 0.01.
    s : float
        Mandelstam s in GeV^2, > 0.
    alpha : float
        Fine-structure constant, > 0.

    Returns
    -------
    sigma_NLO : float
        The order-alpha^3 correction to the total cross section from the
        paper's expansion (through tau^3), in GeV^-2, as a native Python
        float.

    Raises
    ------
    ValueError
        If tau is outside (0, 0.01], or if s or alpha is not strictly positive.
    '''
    return sigma_NLO  # placeholder
```

### Step 4

step_04_nnlo_high_energy_series

Goal
----
Evaluate the paper's high-energy next-to-next-to-leading-order (order alpha^4) correction to the Compton total cross section, through the tau^1 term.

```python
def nnlo_high_energy_series(tau: float, s: float, alpha: float) -> float:
    '''High-energy expansion of the order-alpha^4 correction.

    Parameters
    ----------
    tau : float
        m^2/s, with 0 < tau <= 0.01.
    s : float
        Mandelstam s in GeV^2, > 0.
    alpha : float
        Fine-structure constant, > 0.

    Returns
    -------
    sigma_NNLO : float
        The order-alpha^4 correction to the total cross section from the
        paper's expansion (through tau^1), in GeV^-2, as a native Python
        float.

    Raises
    ------
    ValueError
        If tau is outside (0, 0.01], or if s or alpha is not strictly positive.
    '''
    return sigma_NNLO  # placeholder
```

### Step 5

step_05_ll_ladder_term

Goal
----
Evaluate the paper's all-order leading-logarithmic contribution of the n-loop ladder to the Compton total cross section at order alpha^{n+1}, i.e. the N^{n-1}LO leading logarithm ln^{2n-1} tau with its closed-form coefficient.

```python
def ll_ladder_term(n: int, tau: float, s: float, alpha: float) -> float:
    '''Leading-logarithmic N^{n-1}LO ladder contribution to the total cross section.

    Parameters
    ----------
    n : int
        Loop order of the ladder, >= 1 (n = 1 is the leading-order
        logarithm, n = 2 the NLO double logarithm, ...).
    tau : float
        m^2/s, with 0 < tau < 1.
    s : float
        Mandelstam s in GeV^2, > 0.
    alpha : float
        Fine-structure constant, > 0.

    Returns
    -------
    sigma_LL_n : float
        The order-alpha^{n+1} leading-logarithmic term as defined by the
        paper, in GeV^-2, as a native Python float.

    Raises
    ------
    ValueError
        If n < 1, tau is outside (0, 1), or s or alpha is not strictly positive.
    '''
    return sigma_LL_n  # placeholder
```

### Step 6

step_06_ll_resummed

Goal
----
Evaluate the paper's all-order leading-logarithmic Compton total cross section and its remainder beyond a supplied fixed order.

```python
def ll_resummed(tau: float, s: float, alpha: float, n_max: int) -> tuple:
    '''Resummed leading logarithms and their remainder beyond n_max ladder orders.

    Parameters
    ----------
    tau : float
        m^2/s, with 0 < tau < 1.
    s : float
        Mandelstam s in GeV^2, > 0.
    alpha : float
        Fine-structure constant, > 0.
    n_max : int
        Number of leading ladder orders already contained in the fixed-order
        result (n_max = 3 for an NNLO calculation), >= 0.

    Returns
    -------
    result : tuple of (float, float)
        (sigma_LL, remainder): the paper's closed-form resummed
        leading-logarithmic cross section and sigma_LL minus the sum of the
        ladder terms n = 1..n_max (uses ll_ladder_term), both in GeV^-2 as
        native Python floats.

    Raises
    ------
    ValueError
        If tau is outside (0, 1), s or alpha is not strictly positive, or
        n_max < 0.
    '''
    return sigma_LL, remainder  # placeholder
```

### Step 7

step_07_compton_nnlo_ll

Goal
----
Compute the Compton total cross section at NNLO matched to the all-order leading-logarithmic resummation for a single fixed-target photon energy, and return it in microbarn.

```python
def compton_nnlo_ll(E_gamma: float, m: float, alpha: float) -> float:
    '''NNLO+LL Compton total cross section for a fixed-target photon energy.

    Parameters
    ----------
    E_gamma : float
        Laboratory photon energy in GeV, large enough that tau = m^2/s <= 0.01.
    m : float
        Electron mass in GeV, > 0.
    alpha : float
        Fine-structure constant, > 0.

    Returns
    -------
    sigma_ub : float
        sigma_{NNLO+LL} in microbarn, as a native Python float.

    Raises
    ------
    ValueError
        Propagated from the individual steps on invalid input, in particular
        if tau > 0.01 (outside the domain of the high-energy expansions).
    '''
    return sigma_ub  # placeholder
```

### Step 8

step_08_spectrum_averaged_cross_section

Goal
----
[ORCHESTRATOR] Compute the NNLO+LL Compton total cross section averaged over a supplied fixed-target photon-energy window with the specified 1/E_gamma spectral weight, using Gauss-Legendre quadrature in ln E_gamma.

```python
def spectrum_averaged_cross_section(E_min: float, E_max: float, m: float, alpha: float, n_nodes: int) -> float:
    '''1/E-weighted mean of the NNLO+LL Compton cross section over [E_min, E_max].

    Parameters
    ----------
    E_min : float
        Lower edge of the photon-energy window in GeV, > 0 and large enough
        that tau = m^2/s <= 0.01 (about 25.3 MeV for the electron mass).
    E_max : float
        Upper edge of the window in GeV, > E_min.
    m : float
        Electron mass in GeV, > 0.
    alpha : float
        Fine-structure constant, > 0.
    n_nodes : int
        Number of Gauss-Legendre nodes on [ln E_min, ln E_max], >= 1
        (n_nodes = 1 is the single mid-point exp((ln E_min + ln E_max)/2)).

    Returns
    -------
    sigma_bar : float
        The spectrum-averaged sigma_{NNLO+LL} in microbarn, as a native
        Python float.

    Raises
    ------
    ValueError
        If E_min <= 0, E_max <= E_min or n_nodes < 1; also propagated from the
        earlier steps if any node lies outside the high-energy domain
        (tau > 0.01) or m or alpha is not strictly positive.
    '''
    return sigma_bar  # placeholder
```
