# Material_Science-Molecular_Modeling-50

## Background

A polymer chain of freely jointed Kuhn segments in which one segment can stretch and dissociate is the simplest model of chain scission under load. A recent source builds it under displacement control: the chain of $n+1$ links is held at a prescribed end-to-end vector, the breakable link carries a Lennard-Jones energy, the other $n$ links stay rigid, and integrating the joint statistics of the link and of the two rigid fragments over the orientation of the link gives a free energy of the chain as a function of the bond length with two wells (intact and broken) and a transition state between them. From it the source derives the scission and healing barriers, an equilibrium probability for the chain to be intact at each extension, first-order kinetics of scission and healing with a factor counting the breakable links, and, for a chain pulled at a constant rate, the probability density of the final scission event (the one not followed by healing), whose mean extension and mean rupture force are rate dependent and bounded from below by their equilibrium values. The source evaluates all of this with the Kuhn-Grun free energy of a force-controlled chain in place of the exact end-to-end statistics of the rigid fragments, stating that the exact statistics leads to an integral that is computationally challenging and that the classical closed forms have been limited to short chains. Recover the construction from the source, validate a solver of your own against the checks the source reports, and take the step it does not: replace the Kuhn-Grun fragments by the exact freely-jointed-chain statistics, evaluated without loss of precision, and determine the rate-dependent mean rupture force of a chain of moderate length at a slow pulling rate. The load-bearing choices are the source's and are not derivable from the statement below: the form of the free energy under displacement control and the way the orientation of the breakable link is integrated out; the Kuhn-Grun free energy with its Pade inverse Langevin function; the definitions of the two barriers and of the critical extension; the rate equation with its multiplicity factor and its equal attempt and healing frequencies; the equilibrium probability and its half-probability condition; the constant-rate protocol and its normalised rate; the final-scission density with its no-healing survival factor; and the equilibrium bounds of the rupture statistics.

The model is the source's. The chain has $n+1$ identical links of Kuhn length $l$, freely jointed, one end fixed and the other held at the end-to-end vector $\vec y$. One link, the breakable one, has a variable length $x$ and the energy $V(x) = \epsilon_0[(l/x)^{12} - 2(l/x)^6]$; the other $n$ links are rigid. Lengths are measured in $l$ ($\bar x = x/l$, $\bar y = y/l$), energies in $k_BT$ ($\beta\epsilon_0 = \epsilon_0/k_BT$), forces in $k_BT/l$ ($\bar f = \beta f l$) and rates in the attempt frequency $\nu$. The end-to-end vector of a freely jointed chain of $n$ rigid unit links has the probability density $p_n(r)$ per unit volume, normalised to unit integral, the $n$-fold convolution of the uniform unit-sphere shell; $r p_n(r)$ is piecewise polynomial in $r$ on its support, $p_n(r)$ is that quantity divided by $r$ for $r>0$, and the alternating closed forms lose all their digits in double precision for long chains, so it must be evaluated in exact rational or multiple-precision arithmetic. The free energy of the chain at fixed $\vec y$ as a function of the bond length is $\beta A(\bar x;\bar y) = \beta V(\bar x) - \ln[\bar x^2\int_0^\pi p_n(\bar r_\phi)\sin\phi\,d\phi]$ with $\bar r_\phi^2 = \bar x^2 + \bar y^2 - 2\bar x\bar y\cos\phi$ and no further additive constant; the source's Kuhn-Grun variant replaces the fragment statistics by $\exp[-n\,h(L^{-1}(\bar r/n))]$ with $h(\bar f) = \bar f\coth\bar f + \ln(\bar f/\sinh\bar f)$ and $L^{-1}(\eta) = (3\eta - \eta^3)/(1 - \eta^2)$, and is kept beside the exact one. Below a critical extension $\bar y_c$ the landscape has an intact minimum $\bar x_1$ near 1, a transition-state maximum $\bar x_t$ and a broken minimum $\bar x_2$; the barriers are $E_s = A(\bar x_t) - A(\bar x_1)$ and $E_h = A(\bar x_t) - A(\bar x_2)$, the tension of the intact chain is the total derivative $\bar f(\bar y) = d\,\beta A(\bar x_1(\bar y);\bar y)/d\bar y$ taken at the exact intact minimum, and the critical extension is the fold at which $\bar x_1$ and $\bar x_t$ merge, with $\bar f_c$ the tension in the limit $\bar y \to \bar y_c^-$. The probability of an intact chain obeys $dP/dt = -(n+1)\nu e^{-\beta E_s}P + \nu e^{-\beta E_h}(1-P)$; its stationary value is $P_{eq}(\bar y)$, the equilibrium scission density is $\rho_{eq} = -dP_{eq}/d\bar y$, and $\bar y_{1/2}$ is the extension at which $P_{eq} = 1/2$. Under constant-rate pulling, $\bar y = \bar\gamma\nu t$ with $\bar\gamma = \gamma/(\nu l)$ and $P = 1$ at $\bar y = 0$, the final-scission density is $\rho_{s,last}(\bar y) = P(\bar y)k_s(\bar y)\bar\gamma^{-1}\exp[-\bar\gamma^{-1}\int_{\bar y}^{\infty}k_h\,d\bar y']$ with $k_s = (n+1)e^{-\beta E_s}$ and $k_h = e^{-\beta E_h}$ in units of $\nu$ and $k_h = 0$ beyond $\bar y_c$; it integrates to one, and the mean rupture extension and force are $\langle\bar y_s\rangle = \int\bar y\rho_{s,last}d\bar y$ and $\langle\bar f_s\rangle = \int\bar f(\bar y)\rho_{s,last}d\bar y$. Every quantity is exact and deterministic; nothing is sampled.

Implement eleven functions with these conventions: all lengths, energies, forces and rates are normalised as above, all outputs are float64, and unless a step says otherwise every value is expected to be accurate to $10^{-9}$ (the two constant-rate steps to $10^{-7}$).

`fjc_log_density(n, r)` returns $\ln p_n(r)$ for $0 < r < n$, exact for chains of up to a hundred links. `kuhn_grun_free_energy(n, beps, x, y)` returns the source's Kuhn-Grun free energy $\beta A_{KG}(\bar x;\bar y)$. `breakable_chain_free_energy(n, beps, x, y)` returns the exact free energy $\beta A(\bar x;\bar y)$, with the isotropic limit at $\bar y = 0$. `chain_landscape(n, beps, y)` returns, shape $(5,)$, $\bar x_1$, $\bar x_t$, $\bar x_2$, $\beta E_s$ and $\beta E_h$ of the exact landscape. `intact_branch_force(n, beps, y)` returns the tension $\bar f(\bar y)$. `chain_critical_point(n, beps)` returns, shape $(3,)$, $\bar y_c$, the merged bond length $\bar x_c$ and $\bar f_c$. `equilibrium_probability(n, beps, y)` returns $P_{eq}(\bar y)$, zero at and beyond $\bar y_c$. `equilibrium_rupture_statistics(n, beps)` returns, shape $(3,)$, $\bar y_{1/2}$, $\langle\bar y\rangle_{eq} = (\int\bar y\rho_{eq}d\bar y)/(\int\rho_{eq}d\bar y)$ and $\langle\bar f\rangle_{eq} = (\int\bar f\rho_{eq}d\bar y)/(\int\rho_{eq}d\bar y)$, conditional on a chain being intact at zero extension. `pulled_chain_survival(n, beps, gbar, y)` returns the intact probability $P(\bar y;\bar\gamma)$ of a chain pulled at the constant normalised rate $\bar\gamma$. `final_scission_statistics(n, beps, gbar)` returns, shape $(2,)$, $\langle\bar y_s\rangle$ and $\langle\bar f_s\rangle$. `chain_audit(n, beps, gbar)`, the orchestrator, must call the earlier functions rather than reimplementing them and returns thirteen values: $\beta E_{s0}$ and $\beta E_{h0}$, the barriers at zero extension; $\bar y_{1/2}$; $\langle\bar y\rangle_{eq}$; $\langle\bar f\rangle_{eq}$; $\bar y_c$; $\bar f_c$; $P(\bar y_{1/2};\bar\gamma)$; $\langle\bar y_s\rangle$; $\langle\bar f_s\rangle$; the rate enhancement $\langle\bar f_s\rangle/\langle\bar f\rangle_{eq}$; the half-probability extension $\bar y_{1/2}^{KG}$ obtained when the Kuhn-Grun landscape replaces the exact one throughout; and the Kuhn-Grun value of $\beta(E_s - E_h) - \ln(n+1)$ at the exact $\bar y_{1/2}$. The orchestrator also checks the chain against itself (the exact free energy at zero extension against the density, the equilibrium probability at $\bar y_{1/2}$ against one half, the normalisation of the final-scission density) and raises on a failed check.

All outputs are finite and deterministic: two runs on identical inputs must agree exactly. Validate inputs and raise `ValueError` on non-finite values, a number of links that is not an integer of at least two, a non-positive bond energy, a non-positive bond length, a negative extension, a pulling rate outside $(0, 10^{-2}]$, an end-to-end distance outside the open support $(0, n)$ of the density, a bond length and extension the rigid fragments cannot span (an infinite free energy), a landscape or a force requested at or beyond the critical extension, a link whose intact well does not exist even at zero extension, and, in the constant-rate and orchestrator steps, a final-scission density that does not integrate to one or a failed consistency check.

Evaluate the audit for $n = 30$, $\beta\epsilon_0 = 40$ and $\bar\gamma = 10^{-12}$ (ten nanometres per second for nanometre links at the source's attempt frequency).

In your reasoning report the conventions you used, and justify each from the source: the displacement-controlled free energy and the orientation integral of the breakable link; the Kuhn-Grun free energy, its inverse Langevin approximant and the range in which the source finds it adequate; the definitions of the two barriers, of the critical extension and of the critical force, and the source's statement about the critical force relative to the bond strength of the link; the source's evaluation of the intact-chain force at the equilibrium bond length and why the total derivative at the exact intact minimum is used here instead; the rate equation, its multiplicity factor and its frequencies; the equilibrium probability and its value where the two barriers are equal; the constant-rate protocol, the normalised pulling rate and the range of rates the source associates with experiments; the final-scission density, the reason it differs from the net scission density and its normalisation; and the equilibrium bounds of the rupture statistics. State why the exact fragment statistics has to be evaluated in exact arithmetic and what the source says about the classical closed forms, and describe how you evaluated it.

Report numerically, as evidence that the chain was executed: as checks against the source, the entropic reduction $\beta\Delta E_{s0}$ of the scission barrier and the healing barrier $\beta E_{h0}$ at zero extension for $n = 100$ and $\beta\epsilon_0 = 50$ with the Kuhn-Grun landscape, and the bond strength $\bar f_{max}$ of the Lennard-Jones link relative to $\beta\epsilon_0$. Then for the evaluated point, with the exact statistics: $\beta E_{s0}$ and $\beta E_{h0}$; $\bar y_{1/2}$ together with the Kuhn-Grun value $\bar y_{1/2}^{KG}$ and the Kuhn-Grun barrier imbalance $\beta(E_s - E_h) - \ln(n+1)$ at the exact $\bar y_{1/2}$; $\langle\bar y\rangle_{eq}$ and $\langle\bar f\rangle_{eq}$; $\bar y_c$, $\bar x_c$ and $\bar f_c$, and the largest tension reached on the intact branch and the extension at which it is reached; the survival $P(\bar y_{1/2};\bar\gamma)$ with what its distance from one half says about the freeze-out of healing; $\langle\bar y_s\rangle$ and $\langle\bar f_s\rangle$ and the rate enhancement $\langle\bar f_s\rangle/\langle\bar f\rangle_{eq}$. Assess how the mean rupture force changes under four counterfactuals: removing the multiplicity factor from the scission rate (including the shifted half-probability extension); omitting the no-healing survival factor from the final-scission density (report its raw mass and raw force moment, then normalize it before computing a mean); evaluating force at the equilibrium bond length; and replacing exact fragment statistics with Kuhn-Grun statistics. Also state whether the mean rupture force keeps decreasing without bound as the pulling rate tends to zero. These are the scalars that determine the final number.

As the final answer, report $\langle\bar f_s\rangle$, the mean rupture force of the final scission event in units of $k_BT/l$, for $n = 30$, $\beta\epsilon_0 = 40$ and $\bar\gamma = 10^{-12}$ with the exact fragment statistics, to five significant figures.

Output Format Requirements:
Emit <final_answer> immediately, then <reasoning>. Do not write a long derivation before the tags.
You must emit exactly one finite decimal inside <final_answer>...</final_answer>, even if the value is approximate or you are unsure.
Rules:
- The tags are required. Do not omit them or leave them empty.
- The tagged value must be a finite decimal (examples: 0.4847, 0.22, 1.05). Not NaN, not Inf, not a fraction string, not a vector, not prose.
- Put only that one number between the tags. No units, no words, no extra lines.
In <reasoning>, concisely report all requested source-based conventions, numerical diagnostics, and comparisons with enough digits to verify them. A full derivation is not required.
Do not paste the input matrices, full coefficient vectors, per-iteration paths, or per-fold candidate tables.

## Problem

A polymer chain of freely jointed Kuhn segments in which one segment can stretch and dissociate is the simplest model of chain scission under load. A recent source builds it under displacement control: the chain of $n+1$ links is held at a prescribed end-to-end vector, the breakable link carries a Lennard-Jones energy, the other $n$ links stay rigid, and integrating the joint statistics of the link and of the two rigid fragments over the orientation of the link gives a free energy of the chain as a function of the bond length with two wells (intact and broken) and a transition state between them. From it the source derives the scission and healing barriers, an equilibrium probability for the chain to be intact at each extension, first-order kinetics of scission and healing with a factor counting the breakable links, and, for a chain pulled at a constant rate, the probability density of the final scission event (the one not followed by healing), whose mean extension and mean rupture force are rate dependent and bounded from below by their equilibrium values. The source evaluates all of this with the Kuhn-Grun free energy of a force-controlled chain in place of the exact end-to-end statistics of the rigid fragments, stating that the exact statistics leads to an integral that is computationally challenging and that the classical closed forms have been limited to short chains. Recover the construction from the source, validate a solver of your own against the checks the source reports, and take the step it does not: replace the Kuhn-Grun fragments by the exact freely-jointed-chain statistics, evaluated without loss of precision, and determine the rate-dependent mean rupture force of a chain of moderate length at a slow pulling rate. The load-bearing choices are the source's and are not derivable from the statement below: the form of the free energy under displacement control and the way the orientation of the breakable link is integrated out; the Kuhn-Grun free energy with its Pade inverse Langevin function; the definitions of the two barriers and of the critical extension; the rate equation with its multiplicity factor and its equal attempt and healing frequencies; the equilibrium probability and its half-probability condition; the constant-rate protocol and its normalised rate; the final-scission density with its no-healing survival factor; and the equilibrium bounds of the rupture statistics.

The model is the source's. The chain has $n+1$ identical links of Kuhn length $l$, freely jointed, one end fixed and the other held at the end-to-end vector $\vec y$. One link, the breakable one, has a variable length $x$ and the energy $V(x) = \epsilon_0[(l/x)^{12} - 2(l/x)^6]$; the other $n$ links are rigid. Lengths are measured in $l$ ($\bar x = x/l$, $\bar y = y/l$), energies in $k_BT$ ($\beta\epsilon_0 = \epsilon_0/k_BT$), forces in $k_BT/l$ ($\bar f = \beta f l$) and rates in the attempt frequency $\nu$. The end-to-end vector of a freely jointed chain of $n$ rigid unit links has the probability density $p_n(r)$ per unit volume, normalised to unit integral, the $n$-fold convolution of the uniform unit-sphere shell; it is exact, piecewise polynomial in $r$, and its alternating closed forms lose all their digits in double precision for long chains, so it must be evaluated in exact rational or multiple-precision arithmetic. The free energy of the chain at fixed $\vec y$ as a function of the bond length is $\beta A(\bar x;\bar y) = \beta V(\bar x) - \ln[\bar x^2\int_0^\pi p_n(\bar r_\phi)\sin\phi\,d\phi]$ with $\bar r_\phi^2 = \bar x^2 + \bar y^2 - 2\bar x\bar y\cos\phi$ and no further additive constant; the source's Kuhn-Grun variant replaces the fragment statistics by $\exp[-n\,h(L^{-1}(\bar r/n))]$ with $h(\bar f) = \bar f\coth\bar f + \ln(\bar f/\sinh\bar f)$ and $L^{-1}(\eta) = (3\eta - \eta^3)/(1 - \eta^2)$, and is kept beside the exact one. Below a critical extension $\bar y_c$ the landscape has an intact minimum $\bar x_1$ near 1, a transition-state maximum $\bar x_t$ and a broken minimum $\bar x_2$; the barriers are $E_s = A(\bar x_t) - A(\bar x_1)$ and $E_h = A(\bar x_t) - A(\bar x_2)$, the tension of the intact chain is the total derivative $\bar f(\bar y) = d\,\beta A(\bar x_1(\bar y);\bar y)/d\bar y$ taken at the exact intact minimum, and the critical extension is the fold at which $\bar x_1$ and $\bar x_t$ merge, with $\bar f_c$ the tension in the limit $\bar y \to \bar y_c^-$. The probability of an intact chain obeys $dP/dt = -(n+1)\nu e^{-\beta E_s}P + \nu e^{-\beta E_h}(1-P)$; its stationary value is $P_{eq}(\bar y)$, the equilibrium scission density is $\rho_{eq} = -dP_{eq}/d\bar y$, and $\bar y_{1/2}$ is the extension at which $P_{eq} = 1/2$. Under constant-rate pulling, $\bar y = \bar\gamma\nu t$ with $\bar\gamma = \gamma/(\nu l)$ and $P = 1$ at $\bar y = 0$, the final-scission density is $\rho_{s,last}(\bar y) = P(\bar y)k_s(\bar y)\bar\gamma^{-1}\exp[-\bar\gamma^{-1}\int_{\bar y}^{\infty}k_h\,d\bar y']$ with $k_s = (n+1)e^{-\beta E_s}$ and $k_h = e^{-\beta E_h}$ in units of $\nu$ and $k_h = 0$ beyond $\bar y_c$; it integrates to one, and the mean rupture extension and force are $\langle\bar y_s\rangle = \int\bar y\rho_{s,last}d\bar y$ and $\langle\bar f_s\rangle = \int\bar f(\bar y)\rho_{s,last}d\bar y$. Every quantity is exact and deterministic; nothing is sampled.

Implement eleven functions with these conventions: all lengths, energies, forces and rates are normalised as above, all outputs are float64, and unless a step says otherwise every value is expected to be accurate to $10^{-9}$ (the two constant-rate steps to $10^{-7}$).

`fjc_log_density(n, r)` returns $\ln p_n(r)$ for $0 < r < n$, exact for chains of up to a hundred links. `kuhn_grun_free_energy(n, beps, x, y)` returns the source's Kuhn-Grun free energy $\beta A_{KG}(\bar x;\bar y)$. `breakable_chain_free_energy(n, beps, x, y)` returns the exact free energy $\beta A(\bar x;\bar y)$, with the isotropic limit at $\bar y = 0$. `chain_landscape(n, beps, y)` returns, shape $(5,)$, $\bar x_1$, $\bar x_t$, $\bar x_2$, $\beta E_s$ and $\beta E_h$ of the exact landscape. `intact_branch_force(n, beps, y)` returns the tension $\bar f(\bar y)$. `chain_critical_point(n, beps)` returns, shape $(3,)$, $\bar y_c$, the merged bond length $\bar x_c$ and $\bar f_c$. `equilibrium_probability(n, beps, y)` returns $P_{eq}(\bar y)$, zero at and beyond $\bar y_c$. `equilibrium_rupture_statistics(n, beps)` returns, shape $(3,)$, $\bar y_{1/2}$, $\langle\bar y\rangle_{eq} = (\int\bar y\rho_{eq}d\bar y)/(\int\rho_{eq}d\bar y)$ and $\langle\bar f\rangle_{eq} = (\int\bar f\rho_{eq}d\bar y)/(\int\rho_{eq}d\bar y)$, conditional on a chain being intact at zero extension. `pulled_chain_survival(n, beps, gbar, y)` returns the intact probability $P(\bar y;\bar\gamma)$ of a chain pulled at the constant normalised rate $\bar\gamma$. `final_scission_statistics(n, beps, gbar)` returns, shape $(2,)$, $\langle\bar y_s\rangle$ and $\langle\bar f_s\rangle$. `chain_audit(n, beps, gbar)`, the orchestrator, must call the earlier functions rather than reimplementing them and returns thirteen values: $\beta E_{s0}$ and $\beta E_{h0}$, the barriers at zero extension; $\bar y_{1/2}$; $\langle\bar y\rangle_{eq}$; $\langle\bar f\rangle_{eq}$; $\bar y_c$; $\bar f_c$; $P(\bar y_{1/2};\bar\gamma)$; $\langle\bar y_s\rangle$; $\langle\bar f_s\rangle$; the rate enhancement $\langle\bar f_s\rangle/\langle\bar f\rangle_{eq}$; the half-probability extension $\bar y_{1/2}^{KG}$ obtained when the Kuhn-Grun landscape replaces the exact one throughout; and the Kuhn-Grun value of $\beta(E_s - E_h) - \ln(n+1)$ at the exact $\bar y_{1/2}$. The orchestrator also checks the chain against itself (the exact free energy at zero extension against the density, the equilibrium probability at $\bar y_{1/2}$ against one half, the normalisation of the final-scission density) and raises on a failed check.

All outputs are finite and deterministic: two runs on identical inputs must agree exactly. Validate inputs and raise `ValueError` on non-finite values, a number of links that is not an integer of at least two, a non-positive bond energy, a non-positive bond length, a negative extension, a pulling rate outside $(0, 10^{-2}]$, an end-to-end distance outside the open support $(0, n)$ of the density, a bond length and extension the rigid fragments cannot span (an infinite free energy), a landscape or a force requested at or beyond the critical extension, a link whose intact well does not exist even at zero extension, and, in the constant-rate and orchestrator steps, a final-scission density that does not integrate to one or a failed consistency check.

Evaluate the audit for $n = 30$, $\beta\epsilon_0 = 40$ and $\bar\gamma = 10^{-12}$ (ten nanometres per second for nanometre links at the source's attempt frequency).

In your reasoning report the conventions you used, and justify each from the source: the displacement-controlled free energy and the orientation integral of the breakable link; the Kuhn-Grun free energy, its inverse Langevin approximant and the range in which the source finds it adequate; the definitions of the two barriers, of the critical extension and of the critical force, and the source's statement about the critical force relative to the bond strength of the link; the source's evaluation of the intact-chain force at the equilibrium bond length and why the total derivative at the exact intact minimum is used here instead; the rate equation, its multiplicity factor and its frequencies; the equilibrium probability and its value where the two barriers are equal; the constant-rate protocol, the normalised pulling rate and the range of rates the source associates with experiments; the final-scission density, the reason it differs from the net scission density and its normalisation; and the equilibrium bounds of the rupture statistics. State why the exact fragment statistics has to be evaluated in exact arithmetic and what the source says about the classical closed forms, and describe how you evaluated it.

Report numerically, as evidence that the chain was executed: as checks against the source, the entropic reduction $\beta\Delta E_{s0}$ of the scission barrier and the healing barrier $\beta E_{h0}$ at zero extension for $n = 100$ and $\beta\epsilon_0 = 50$ with the Kuhn-Grun landscape, and the bond strength $\bar f_{max}$ of the Lennard-Jones link relative to $\beta\epsilon_0$. Then for the evaluated point, with the exact statistics: $\beta E_{s0}$ and $\beta E_{h0}$; $\bar y_{1/2}$ together with the Kuhn-Grun value $\bar y_{1/2}^{KG}$ and the Kuhn-Grun barrier imbalance $\beta(E_s - E_h) - \ln(n+1)$ at the exact $\bar y_{1/2}$; $\langle\bar y\rangle_{eq}$ and $\langle\bar f\rangle_{eq}$; $\bar y_c$, $\bar x_c$ and $\bar f_c$, and the largest tension reached on the intact branch and the extension at which it is reached; the survival $P(\bar y_{1/2};\bar\gamma)$ with what its distance from one half says about the freeze-out of healing; $\langle\bar y_s\rangle$ and $\langle\bar f_s\rangle$ and the rate enhancement $\langle\bar f_s\rangle/\langle\bar f\rangle_{eq}$. Assess how the mean rupture force changes under four counterfactuals: removing the multiplicity factor from the scission rate (including the shifted half-probability extension); omitting the no-healing survival factor from the final-scission density (report its raw mass and raw force moment, then normalize it before computing a mean); evaluating force at the equilibrium bond length; and replacing exact fragment statistics with Kuhn-Grun statistics. Also state whether the mean rupture force keeps decreasing without bound as the pulling rate tends to zero. These are the scalars that determine the final number.

As the final answer, report $\langle\bar f_s\rangle$, the mean rupture force of the final scission event in units of $k_BT/l$, for $n = 30$, $\beta\epsilon_0 = 40$ and $\bar\gamma = 10^{-12}$ with the exact fragment statistics, to five significant figures.

Output Format Requirements:
Emit <final_answer> immediately, then <reasoning>. Do not write a long derivation before the tags.
You must emit exactly one finite decimal inside <final_answer>...</final_answer>, even if the value is approximate or you are unsure.
Rules:
- The tags are required. Do not omit them or leave them empty.
- The tagged value must be a finite decimal (examples: 0.4847, 0.22, 1.05). Not NaN, not Inf, not a fraction string, not a vector, not prose.
- Put only that one number between the tags. No units, no words, no extra lines.
In <reasoning>, concisely report all requested source-based conventions, numerical diagnostics, and comparisons with enough digits to verify them. A full derivation is not required.
Do not paste the input matrices, full coefficient vectors, per-iteration paths, or per-fold candidate tables.

## Output format

```
Output Format Requirements:
Emit <final_answer> immediately, then <reasoning>. Do not write a long derivation before the tags.
You must emit exactly one finite decimal inside <final_answer>...</final_answer>, even if the value is approximate or you are unsure.
Rules:
- The tags are required. Do not omit them or leave them empty.
- The tagged value must be a finite decimal (examples: 0.4847, 0.22, 1.05). Not NaN, not Inf, not a fraction string, not a vector, not prose.
- Put only that one number between the tags. No units, no words, no extra lines.
Keep <reasoning> short (a few hundred words). Show only the few scalars that determine the final number.
Do not paste the input matrices, full coefficient vectors, per-iteration paths, or per-fold candidate tables.
```

## Your task

Implement **all 11 functions** below into a single file at `/app/solution.py`. Requirements:

- Use the EXACT function name and signature shown for each step.
- Define every function at module scope (importable).
- Do NOT include tests, example usage, prints, or a `__main__` block.
- Available libraries: numpy, scipy, sympy, pandas, scikit-learn, networkx.

Start your file with these imports:

```python
import numpy as np
```

### Step 1

fjc_log_density

Goal
----
Returns the natural logarithm of the exact end-to-end probability density of a freely jointed chain of n rigid unit links at a given end-to-end distance, evaluated without loss of precision for chains of up to a hundred links.

```python
def fjc_log_density(n: int, r: float) -> float:
    r"""n: integer at least 2, the number of rigid links. r: float, the end-to-end distance in units of the link
    length, strictly between 0 and n.

    A freely jointed chain of $n$ rigid links of unit length has one end at the origin and the other end at the
    random vector $\vec r = \sum_{i=1}^n \vec l_i$, each link vector being uniformly distributed on the unit sphere and
    independent of the others. Its end-to-end probability density $p_n(r)$ (probability per unit volume of finding
    the free end at $\vec r$, depending only on $r = |\vec r|$ and normalised so that $\int p_n\,d^3r = 1$) is the
    $n$-fold convolution of the uniform spherical-shell distribution. The product $r p_n(r)$ is piecewise polynomial in $r$; $p_n(r)$ is this product divided by $r$ for $r>0$. Its
    alternating-sign closed form (Rayleigh, Treloar) loses all its digits in double precision for chains of a
    hundred links, so the evaluation must be carried out in exact rational or multiple-precision arithmetic (Python's
    built-in integers and fractions are sufficient and always available; third-party multiple-precision packages such
    as mpmath may be absent from the execution environment); the result is checked to a relative accuracy of $10^{-9}$
    and better, up to $n = 100$.

    Returns a Python float: $\ln p_n(r)$, the natural logarithm of the density.

    Raises:
        ValueError: if n is not an integer of at least 2, if r is not a finite number strictly between 0 and n
        (the density vanishes identically at and beyond the contour length).
    """
    return None
```

### Step 2

kuhn_grun_free_energy

Goal
----
Returns the source's Kuhn-Grun free energy of the breakable chain at a given bond length and end-to-end distance, with the Pade inverse Langevin function and the angular integral resolved to high accuracy.

```python
import numpy as np


def kuhn_grun_free_energy(n: int, beps: float, x: float, y: float) -> float:
    r"""n: integer at least 2, the number of rigid links. beps: positive float, the normalised bond energy
    $\beta\epsilon_0$. x: positive float, the normalised length $\bar x$ of the breakable link. y: non-negative float,
    the normalised end-to-end distance $\bar y$ of the chain.

    The chain has $n+1$ identical links of Kuhn length $l$, freely jointed, with one end fixed and the other end held
    at the end-to-end vector $\\vec y$ (displacement control). One link, the breakable one, has a variable length $x$
    and the Lennard-Jones energy $V(x) = \\epsilon_0[(l/x)^{12} - 2(l/x)^6]$; the other $n$ links are rigid and freely
    jointed and are described by the exact end-to-end statistics of a freely jointed chain (no Gaussian or
    Kuhn-Grun approximation unless a step says so). Lengths are measured in $l$ ($\\bar x = x/l$, $\\bar y = y/l$),
    energies in $k_BT$ ($\\beta\\epsilon_0 = \\epsilon_0/k_BT$) and rates in the attempt frequency $\\nu$. Integrating the
    joint statistics of the breakable link and of the two rigid fragments over the orientation of the link gives the
    free energy of the chain at fixed $\\vec y$ as a function of the bond length,
    $\\beta A(\\bar x;\\bar y) = \\beta V(\\bar x) - \\ln\\big[\\bar x^2 \\int_0^\\pi p_n(\\bar r_\\phi)\\,\\sin\\phi\\,d\\phi\\big]$ with
    $\\bar r_\\phi^2 = \\bar x^2 + \\bar y^2 - 2\\bar x\\bar y\\cos\\phi$, where $\\phi$ is the angle between the link vector and
    $\\vec y$ and $p_n(r)$ is the probability density (per unit volume, normalised to unit integral over all end-to-end
    vectors) of the end-to-end vector of a freely jointed chain of $n$ rigid unit links; no further additive constant
    is included. Below a critical extension $\\bar y_c$ this landscape has an intact minimum $\\bar x_1$ near 1, a
    transition-state maximum $\\bar x_t$ and a broken minimum $\\bar x_2 > \\bar x_t$; the scission and healing barriers
    are $E_s = A(\\bar x_t) - A(\\bar x_1)$ and $E_h = A(\\bar x_t) - A(\\bar x_2)$.

    This step evaluates the source's approximate landscape instead of the exact one: the rigid fragments are given
    the Kuhn-Grun free energy of a force-controlled freely jointed chain, so that
    $\beta A_{KG}(\bar x;\bar y) = \beta V(\bar x) - \ln[\bar x^2 I_n(\bar x,\bar y)]$ with
    $I_n = \int_{-1}^{1} \exp[-n\,h(L^{-1}(\bar r_c/n))]\,dc$, $\bar r_c^2 = \bar x^2 + \bar y^2 - 2\bar x\bar y c$,
    $h(\bar f) = \bar f\coth\bar f + \ln(\bar f/\sinh\bar f)$ and the inverse Langevin function taken as the Pade form
    $L^{-1}(\eta) = (3\eta - \eta^3)/(1 - \eta^2)$; the integrand is zero wherever $\bar r_c \ge n$. The integral must be
    resolved to a relative accuracy of $10^{-11}$ or better.

    Returns a Python float: $\beta A_{KG}(\bar x;\bar y)$.

    Raises:
        ValueError: on an invalid n, beps, x or y, or if $I_n$ vanishes because the fragments cannot span the gap
        ($|\bar y - \bar x| \ge n$).
    """
    return None
```

### Step 3

breakable_chain_free_energy

Goal
----
Returns the exact free energy of the breakable chain at a given bond length and end-to-end distance, the Lennard-Jones link combined with the exact orientation-averaged statistics of the two rigid fragments.

```python
import numpy as np


def breakable_chain_free_energy(n: int, beps: float, x: float, y: float) -> float:
    r"""n: integer at least 2, the number of rigid links. beps: positive float, $\beta\epsilon_0$. x: positive float,
    the normalised bond length $\bar x$. y: non-negative float, the normalised end-to-end distance $\bar y$.

    The chain has $n+1$ identical links of Kuhn length $l$, freely jointed, with one end fixed and the other end held
    at the end-to-end vector $\\vec y$ (displacement control). One link, the breakable one, has a variable length $x$
    and the Lennard-Jones energy $V(x) = \\epsilon_0[(l/x)^{12} - 2(l/x)^6]$; the other $n$ links are rigid and freely
    jointed and are described by the exact end-to-end statistics of a freely jointed chain (no Gaussian or
    Kuhn-Grun approximation unless a step says so). Lengths are measured in $l$ ($\\bar x = x/l$, $\\bar y = y/l$),
    energies in $k_BT$ ($\\beta\\epsilon_0 = \\epsilon_0/k_BT$) and rates in the attempt frequency $\\nu$. Integrating the
    joint statistics of the breakable link and of the two rigid fragments over the orientation of the link gives the
    free energy of the chain at fixed $\\vec y$ as a function of the bond length,
    $\\beta A(\\bar x;\\bar y) = \\beta V(\\bar x) - \\ln\\big[\\bar x^2 \\int_0^\\pi p_n(\\bar r_\\phi)\\,\\sin\\phi\\,d\\phi\\big]$ with
    $\\bar r_\\phi^2 = \\bar x^2 + \\bar y^2 - 2\\bar x\\bar y\\cos\\phi$, where $\\phi$ is the angle between the link vector and
    $\\vec y$ and $p_n(r)$ is the probability density (per unit volume, normalised to unit integral over all end-to-end
    vectors) of the end-to-end vector of a freely jointed chain of $n$ rigid unit links; no further additive constant
    is included. Below a critical extension $\\bar y_c$ this landscape has an intact minimum $\\bar x_1$ near 1, a
    transition-state maximum $\\bar x_t$ and a broken minimum $\\bar x_2 > \\bar x_t$; the scission and healing barriers
    are $E_s = A(\\bar x_t) - A(\\bar x_1)$ and $E_h = A(\\bar x_t) - A(\\bar x_2)$.

    This step evaluates the exact landscape $\beta A(\bar x;\bar y)$ defined above, with the exact density $p_n$ of the
    density step (normalised per unit volume). At $\bar y = 0$ the angular integral is isotropic and equals
    $2\,p_n(\bar x)$. The angular integral must be resolved to a relative accuracy of $10^{-11}$ or better (the product $r p_n(r)$ is piecewise polynomial in the radial distance $\bar r_\phi$, so the integral can be done in closed form once the cumulative
    moment of the density is available).

    Returns a Python float: $\beta A(\bar x;\bar y)$.

    Raises:
        ValueError: on an invalid n, beps, x or y, or if the free energy is infinite because the rigid fragments
        cannot span the gap ($|\bar y - \bar x| \ge n$).
    """
    return None
```

### Step 4

chain_landscape

Goal
----
Locates the intact minimum, the transition state and the broken minimum of the exact free-energy landscape at a given extension and returns them with the scission and healing barriers.

```python
import numpy as np


def chain_landscape(n: int, beps: float, y: float) -> np.ndarray:
    r"""n: integer at least 2, the number of rigid links. beps: positive float, $\beta\epsilon_0$. y: non-negative
    float, the normalised end-to-end distance $\bar y$.

    The chain has $n+1$ identical links of Kuhn length $l$, freely jointed, with one end fixed and the other end held
    at the end-to-end vector $\\vec y$ (displacement control). One link, the breakable one, has a variable length $x$
    and the Lennard-Jones energy $V(x) = \\epsilon_0[(l/x)^{12} - 2(l/x)^6]$; the other $n$ links are rigid and freely
    jointed and are described by the exact end-to-end statistics of a freely jointed chain (no Gaussian or
    Kuhn-Grun approximation unless a step says so). Lengths are measured in $l$ ($\\bar x = x/l$, $\\bar y = y/l$),
    energies in $k_BT$ ($\\beta\\epsilon_0 = \\epsilon_0/k_BT$) and rates in the attempt frequency $\\nu$. Integrating the
    joint statistics of the breakable link and of the two rigid fragments over the orientation of the link gives the
    free energy of the chain at fixed $\\vec y$ as a function of the bond length,
    $\\beta A(\\bar x;\\bar y) = \\beta V(\\bar x) - \\ln\\big[\\bar x^2 \\int_0^\\pi p_n(\\bar r_\\phi)\\,\\sin\\phi\\,d\\phi\\big]$ with
    $\\bar r_\\phi^2 = \\bar x^2 + \\bar y^2 - 2\\bar x\\bar y\\cos\\phi$, where $\\phi$ is the angle between the link vector and
    $\\vec y$ and $p_n(r)$ is the probability density (per unit volume, normalised to unit integral over all end-to-end
    vectors) of the end-to-end vector of a freely jointed chain of $n$ rigid unit links; no further additive constant
    is included. Below a critical extension $\\bar y_c$ this landscape has an intact minimum $\\bar x_1$ near 1, a
    transition-state maximum $\\bar x_t$ and a broken minimum $\\bar x_2 > \\bar x_t$; the scission and healing barriers
    are $E_s = A(\\bar x_t) - A(\\bar x_1)$ and $E_h = A(\\bar x_t) - A(\\bar x_2)$.

    This step locates the three critical points of the exact landscape at the given extension: the intact minimum
    $\bar x_1$ (the smallest local minimum, near the equilibrium bond length), the transition-state maximum
    $\bar x_t$ and the broken minimum $\bar x_2$ beyond it (the landscape rises without bound once the fragments
    reach their contour length), and returns them with the two barriers. Locations must be accurate to $10^{-10}$ and
    the barriers to $10^{-10}$.

    Returns a numpy float64 array of shape $(5,)$: $\bar x_1$, $\bar x_t$, $\bar x_2$, $\beta E_s$, $\beta E_h$.

    Raises:
        ValueError: on an invalid n, beps or y, or if the landscape has no intact well at this extension
        ($\bar y \ge \bar y_c$).
    """
    return None
```

### Step 5

intact_branch_force

Goal
----
Returns the tension of the intact chain at a given extension, the total derivative of the intact-branch free energy taken at the exact intact minimum.

```python
import numpy as np


def intact_branch_force(n: int, beps: float, y: float) -> float:
    r"""n: integer at least 2, the number of rigid links. beps: positive float, $\beta\epsilon_0$. y: non-negative
    float, the normalised end-to-end distance $\bar y$, below the critical extension.

    The chain has $n+1$ identical links of Kuhn length $l$, freely jointed, with one end fixed and the other end held
    at the end-to-end vector $\\vec y$ (displacement control). One link, the breakable one, has a variable length $x$
    and the Lennard-Jones energy $V(x) = \\epsilon_0[(l/x)^{12} - 2(l/x)^6]$; the other $n$ links are rigid and freely
    jointed and are described by the exact end-to-end statistics of a freely jointed chain (no Gaussian or
    Kuhn-Grun approximation unless a step says so). Lengths are measured in $l$ ($\\bar x = x/l$, $\\bar y = y/l$),
    energies in $k_BT$ ($\\beta\\epsilon_0 = \\epsilon_0/k_BT$) and rates in the attempt frequency $\\nu$. Integrating the
    joint statistics of the breakable link and of the two rigid fragments over the orientation of the link gives the
    free energy of the chain at fixed $\\vec y$ as a function of the bond length,
    $\\beta A(\\bar x;\\bar y) = \\beta V(\\bar x) - \\ln\\big[\\bar x^2 \\int_0^\\pi p_n(\\bar r_\\phi)\\,\\sin\\phi\\,d\\phi\\big]$ with
    $\\bar r_\\phi^2 = \\bar x^2 + \\bar y^2 - 2\\bar x\\bar y\\cos\\phi$, where $\\phi$ is the angle between the link vector and
    $\\vec y$ and $p_n(r)$ is the probability density (per unit volume, normalised to unit integral over all end-to-end
    vectors) of the end-to-end vector of a freely jointed chain of $n$ rigid unit links; no further additive constant
    is included. Below a critical extension $\\bar y_c$ this landscape has an intact minimum $\\bar x_1$ near 1, a
    transition-state maximum $\\bar x_t$ and a broken minimum $\\bar x_2 > \\bar x_t$; the scission and healing barriers
    are $E_s = A(\\bar x_t) - A(\\bar x_1)$ and $E_h = A(\\bar x_t) - A(\\bar x_2)$.

    The tension in an intact chain held at $\bar y$ is the total derivative of the intact-branch free energy,
    $\bar f(\bar y) = d\,\beta A(\bar x_1(\bar y);\bar y)/d\bar y$, in units of $k_BT/l$; by the extremum condition at
    $\bar x_1$ this is the partial derivative of $\beta A$ with respect to $\bar y$ taken at the exact intact minimum
    (not at the equilibrium bond length $\bar x = 1$). The value must be accurate to $10^{-9}$, which a finite
    difference of the free energy does not deliver; differentiate under the angular integral.

    Returns a Python float: $\bar f(\bar y)$.

    Raises:
        ValueError: on an invalid n, beps or y, or if no intact well exists at this extension.
    """
    return None
```

### Step 6

chain_critical_point

Goal
----
Returns the critical extension at which the intact minimum of the exact landscape merges with the transition state, together with the merged bond length and the critical force.

```python
import numpy as np


def chain_critical_point(n: int, beps: float) -> np.ndarray:
    r"""n: integer at least 2, the number of rigid links. beps: positive float, $\beta\epsilon_0$.

    The chain has $n+1$ identical links of Kuhn length $l$, freely jointed, with one end fixed and the other end held
    at the end-to-end vector $\\vec y$ (displacement control). One link, the breakable one, has a variable length $x$
    and the Lennard-Jones energy $V(x) = \\epsilon_0[(l/x)^{12} - 2(l/x)^6]$; the other $n$ links are rigid and freely
    jointed and are described by the exact end-to-end statistics of a freely jointed chain (no Gaussian or
    Kuhn-Grun approximation unless a step says so). Lengths are measured in $l$ ($\\bar x = x/l$, $\\bar y = y/l$),
    energies in $k_BT$ ($\\beta\\epsilon_0 = \\epsilon_0/k_BT$) and rates in the attempt frequency $\\nu$. Integrating the
    joint statistics of the breakable link and of the two rigid fragments over the orientation of the link gives the
    free energy of the chain at fixed $\\vec y$ as a function of the bond length,
    $\\beta A(\\bar x;\\bar y) = \\beta V(\\bar x) - \\ln\\big[\\bar x^2 \\int_0^\\pi p_n(\\bar r_\\phi)\\,\\sin\\phi\\,d\\phi\\big]$ with
    $\\bar r_\\phi^2 = \\bar x^2 + \\bar y^2 - 2\\bar x\\bar y\\cos\\phi$, where $\\phi$ is the angle between the link vector and
    $\\vec y$ and $p_n(r)$ is the probability density (per unit volume, normalised to unit integral over all end-to-end
    vectors) of the end-to-end vector of a freely jointed chain of $n$ rigid unit links; no further additive constant
    is included. Below a critical extension $\\bar y_c$ this landscape has an intact minimum $\\bar x_1$ near 1, a
    transition-state maximum $\\bar x_t$ and a broken minimum $\\bar x_2 > \\bar x_t$; the scission and healing barriers
    are $E_s = A(\\bar x_t) - A(\\bar x_1)$ and $E_h = A(\\bar x_t) - A(\\bar x_2)$.

    As the extension grows the intact minimum and the transition state approach each other and merge at the
    critical extension $\bar y_c$ (a fold: $\partial\beta A/\partial\bar x = \partial^2\beta A/\partial\bar x^2 = 0$ at the
    merged bond length $\bar x_c$), beyond which the chain has no intact state and breaks without a barrier. This
    step returns the critical extension, the merged bond length and the critical force $\bar f_c$, the intact-branch
    tension in the limit $\bar y \to \bar y_c^-$, all accurate to $10^{-9}$.

    Returns a numpy float64 array of shape $(3,)$: $\bar y_c$, $\bar x_c$, $\bar f_c$.

    Raises:
        ValueError: on an invalid n or beps, or if the link has no intact well even at zero extension.
    """
    return None
```

### Step 7

equilibrium_probability

Goal
----
Returns the equilibrium probability that a chain held at a given extension is intact, from the exact scission and healing barriers and the multiplicity of breakable links.

```python
import numpy as np


def equilibrium_probability(n: int, beps: float, y: float) -> float:
    r"""n: integer at least 2, the number of rigid links. beps: positive float, $\beta\epsilon_0$. y: non-negative
    float, the normalised end-to-end distance $\bar y$.

    The chain has $n+1$ identical links of Kuhn length $l$, freely jointed, with one end fixed and the other end held
    at the end-to-end vector $\\vec y$ (displacement control). One link, the breakable one, has a variable length $x$
    and the Lennard-Jones energy $V(x) = \\epsilon_0[(l/x)^{12} - 2(l/x)^6]$; the other $n$ links are rigid and freely
    jointed and are described by the exact end-to-end statistics of a freely jointed chain (no Gaussian or
    Kuhn-Grun approximation unless a step says so). Lengths are measured in $l$ ($\\bar x = x/l$, $\\bar y = y/l$),
    energies in $k_BT$ ($\\beta\\epsilon_0 = \\epsilon_0/k_BT$) and rates in the attempt frequency $\\nu$. Integrating the
    joint statistics of the breakable link and of the two rigid fragments over the orientation of the link gives the
    free energy of the chain at fixed $\\vec y$ as a function of the bond length,
    $\\beta A(\\bar x;\\bar y) = \\beta V(\\bar x) - \\ln\\big[\\bar x^2 \\int_0^\\pi p_n(\\bar r_\\phi)\\,\\sin\\phi\\,d\\phi\\big]$ with
    $\\bar r_\\phi^2 = \\bar x^2 + \\bar y^2 - 2\\bar x\\bar y\\cos\\phi$, where $\\phi$ is the angle between the link vector and
    $\\vec y$ and $p_n(r)$ is the probability density (per unit volume, normalised to unit integral over all end-to-end
    vectors) of the end-to-end vector of a freely jointed chain of $n$ rigid unit links; no further additive constant
    is included. Below a critical extension $\\bar y_c$ this landscape has an intact minimum $\\bar x_1$ near 1, a
    transition-state maximum $\\bar x_t$ and a broken minimum $\\bar x_2 > \\bar x_t$; the scission and healing barriers
    are $E_s = A(\\bar x_t) - A(\\bar x_1)$ and $E_h = A(\\bar x_t) - A(\\bar x_2)$.

    The probability $P$ that the chain is intact obeys the first-order kinetics
    $dP/dt = -(n+1)\\,\\nu\\,e^{-\\beta E_s(\\bar y)}\\,P + \\nu\\,e^{-\\beta E_h(\\bar y)}\\,(1-P)$: any one of the $n+1$ links may
    break, and the healing frequency equals the attempt frequency $\\nu$. Beyond $\\bar y_c$ no intact state exists
    ($E_s = 0$, $E_h = \\infty$).

    This step returns the equilibrium probability $P_{eq}(\bar y)$ of an intact chain held at $\bar y$, the stationary
    point of the kinetics, evaluated with the exact barriers; it is zero at and beyond the critical extension.

    Returns a Python float: $P_{eq}(\bar y)$.

    Raises:
        ValueError: on an invalid n, beps or y.
    """
    return None
```

### Step 8

equilibrium_rupture_statistics

Goal
----
Returns the extension at which the equilibrium intact probability is one half, and the equilibrium mean rupture extension and force, the slow-pulling bounds of the rupture statistics.

```python
import numpy as np


def equilibrium_rupture_statistics(n: int, beps: float) -> np.ndarray:
    r"""n: integer at least 2, the number of rigid links. beps: positive float, $\beta\epsilon_0$.

    The chain has $n+1$ identical links of Kuhn length $l$, freely jointed, with one end fixed and the other end held
    at the end-to-end vector $\\vec y$ (displacement control). One link, the breakable one, has a variable length $x$
    and the Lennard-Jones energy $V(x) = \\epsilon_0[(l/x)^{12} - 2(l/x)^6]$; the other $n$ links are rigid and freely
    jointed and are described by the exact end-to-end statistics of a freely jointed chain (no Gaussian or
    Kuhn-Grun approximation unless a step says so). Lengths are measured in $l$ ($\\bar x = x/l$, $\\bar y = y/l$),
    energies in $k_BT$ ($\\beta\\epsilon_0 = \\epsilon_0/k_BT$) and rates in the attempt frequency $\\nu$. Integrating the
    joint statistics of the breakable link and of the two rigid fragments over the orientation of the link gives the
    free energy of the chain at fixed $\\vec y$ as a function of the bond length,
    $\\beta A(\\bar x;\\bar y) = \\beta V(\\bar x) - \\ln\\big[\\bar x^2 \\int_0^\\pi p_n(\\bar r_\\phi)\\,\\sin\\phi\\,d\\phi\\big]$ with
    $\\bar r_\\phi^2 = \\bar x^2 + \\bar y^2 - 2\\bar x\\bar y\\cos\\phi$, where $\\phi$ is the angle between the link vector and
    $\\vec y$ and $p_n(r)$ is the probability density (per unit volume, normalised to unit integral over all end-to-end
    vectors) of the end-to-end vector of a freely jointed chain of $n$ rigid unit links; no further additive constant
    is included. Below a critical extension $\\bar y_c$ this landscape has an intact minimum $\\bar x_1$ near 1, a
    transition-state maximum $\\bar x_t$ and a broken minimum $\\bar x_2 > \\bar x_t$; the scission and healing barriers
    are $E_s = A(\\bar x_t) - A(\\bar x_1)$ and $E_h = A(\\bar x_t) - A(\\bar x_2)$.

    The probability $P$ that the chain is intact obeys the first-order kinetics
    $dP/dt = -(n+1)\\,\\nu\\,e^{-\\beta E_s(\\bar y)}\\,P + \\nu\\,e^{-\\beta E_h(\\bar y)}\\,(1-P)$: any one of the $n+1$ links may
    break, and the healing frequency equals the attempt frequency $\\nu$. Beyond $\\bar y_c$ no intact state exists
    ($E_s = 0$, $E_h = \\infty$).

    Pulled infinitely slowly, the chain follows its equilibrium probability, and the scission events are
    distributed with the density $\rho_{eq}(\bar y) = -dP_{eq}/d\bar y$ on $0 \le \bar y < \bar y_c$.
    Its mass is $M_{eq}=P_{eq}(0)-P_{eq}(\bar y_c^-)$, which can be less than one for a weak link.
    Conditioned on being intact at zero extension, the normalized rupture density is
    $\widehat\rho_{eq}=\rho_{eq}/M_{eq}$. This step returns the extension $\bar y_{1/2}$ at which
    $P_{eq}=1/2$, the conditional mean extension $\int_0^{\bar y_c}\bar y\,\widehat\rho_{eq}\,d\bar y$
    and conditional mean force $\int_0^{\bar y_c}\bar f(\bar y)\,\widehat\rho_{eq}\,d\bar y$,
    with $\bar f$ the intact-branch tension, all accurate to $10^{-9}$.

    Returns a numpy float64 array of shape $(3,)$: $\bar y_{1/2}$, $\langle\bar y\rangle_{eq}$, $\langle\bar f\rangle_{eq}$.

    Raises:
        ValueError: on an invalid n or beps, or if the link has no intact well at zero extension.
    """
    return None
```

### Step 9

pulled_chain_survival

Goal
----
Returns the probability that a chain pulled at a constant normalised rate is still intact at a given extension, from the stiff first-order kinetics of scission and healing with the exact barriers.

```python
import numpy as np


def pulled_chain_survival(n: int, beps: float, gbar: float, y: float) -> float:
    r"""n: integer at least 2, the number of rigid links. beps: positive float, $\beta\epsilon_0$. gbar: float in
    $(0, 10^{-2}]$, the normalised pulling rate $\bar\gamma = \gamma/(\nu l)$. y: non-negative float, the normalised
    end-to-end distance $\bar y$ at which the probability is wanted.

    The chain has $n+1$ identical links of Kuhn length $l$, freely jointed, with one end fixed and the other end held
    at the end-to-end vector $\\vec y$ (displacement control). One link, the breakable one, has a variable length $x$
    and the Lennard-Jones energy $V(x) = \\epsilon_0[(l/x)^{12} - 2(l/x)^6]$; the other $n$ links are rigid and freely
    jointed and are described by the exact end-to-end statistics of a freely jointed chain (no Gaussian or
    Kuhn-Grun approximation unless a step says so). Lengths are measured in $l$ ($\\bar x = x/l$, $\\bar y = y/l$),
    energies in $k_BT$ ($\\beta\\epsilon_0 = \\epsilon_0/k_BT$) and rates in the attempt frequency $\\nu$. Integrating the
    joint statistics of the breakable link and of the two rigid fragments over the orientation of the link gives the
    free energy of the chain at fixed $\\vec y$ as a function of the bond length,
    $\\beta A(\\bar x;\\bar y) = \\beta V(\\bar x) - \\ln\\big[\\bar x^2 \\int_0^\\pi p_n(\\bar r_\\phi)\\,\\sin\\phi\\,d\\phi\\big]$ with
    $\\bar r_\\phi^2 = \\bar x^2 + \\bar y^2 - 2\\bar x\\bar y\\cos\\phi$, where $\\phi$ is the angle between the link vector and
    $\\vec y$ and $p_n(r)$ is the probability density (per unit volume, normalised to unit integral over all end-to-end
    vectors) of the end-to-end vector of a freely jointed chain of $n$ rigid unit links; no further additive constant
    is included. Below a critical extension $\\bar y_c$ this landscape has an intact minimum $\\bar x_1$ near 1, a
    transition-state maximum $\\bar x_t$ and a broken minimum $\\bar x_2 > \\bar x_t$; the scission and healing barriers
    are $E_s = A(\\bar x_t) - A(\\bar x_1)$ and $E_h = A(\\bar x_t) - A(\\bar x_2)$.

    The probability $P$ that the chain is intact obeys the first-order kinetics
    $dP/dt = -(n+1)\\,\\nu\\,e^{-\\beta E_s(\\bar y)}\\,P + \\nu\\,e^{-\\beta E_h(\\bar y)}\\,(1-P)$: any one of the $n+1$ links may
    break, and the healing frequency equals the attempt frequency $\\nu$. Beyond $\\bar y_c$ no intact state exists
    ($E_s = 0$, $E_h = \\infty$).

    The chain is pulled at the constant rate $\gamma$, $\bar y(t) = \bar\gamma\,\nu t$, starting intact at $\bar y = 0$.
    This step returns the probability $P(\bar y;\bar\gamma)$ that it is still intact when the extension reaches
    $\bar y$. The rates exceed the pulling rate by many orders of magnitude at small extension and fall below it
    later, so the kinetics is stiff and must be integrated accordingly; the barriers are expensive, so an efficient
    implementation resolves them once on a grid fine enough for the requested accuracy. The value is checked to
    $10^{-7}$.

    Returns a Python float: $P(\bar y;\bar\gamma)$ (1 at zero extension, 0 at and beyond the critical extension).

    Raises:
        ValueError: on an invalid n, beps, gbar or y.
    """
    return None
```

### Step 10

final_scission_statistics

Goal
----
Returns the mean extension and the mean rupture force of the final scission event of a chain pulled at a constant normalised rate, from the intact probability, the scission rate and the probability of not healing afterwards.

```python
import numpy as np


def final_scission_statistics(n: int, beps: float, gbar: float) -> np.ndarray:
    r"""n: integer at least 2, the number of rigid links. beps: positive float, $\beta\epsilon_0$. gbar: float in
    $(0, 10^{-2}]$, the normalised pulling rate $\bar\gamma = \gamma/(\nu l)$.

    The chain has $n+1$ identical links of Kuhn length $l$, freely jointed, with one end fixed and the other end held
    at the end-to-end vector $\\vec y$ (displacement control). One link, the breakable one, has a variable length $x$
    and the Lennard-Jones energy $V(x) = \\epsilon_0[(l/x)^{12} - 2(l/x)^6]$; the other $n$ links are rigid and freely
    jointed and are described by the exact end-to-end statistics of a freely jointed chain (no Gaussian or
    Kuhn-Grun approximation unless a step says so). Lengths are measured in $l$ ($\\bar x = x/l$, $\\bar y = y/l$),
    energies in $k_BT$ ($\\beta\\epsilon_0 = \\epsilon_0/k_BT$) and rates in the attempt frequency $\\nu$. Integrating the
    joint statistics of the breakable link and of the two rigid fragments over the orientation of the link gives the
    free energy of the chain at fixed $\\vec y$ as a function of the bond length,
    $\\beta A(\\bar x;\\bar y) = \\beta V(\\bar x) - \\ln\\big[\\bar x^2 \\int_0^\\pi p_n(\\bar r_\\phi)\\,\\sin\\phi\\,d\\phi\\big]$ with
    $\\bar r_\\phi^2 = \\bar x^2 + \\bar y^2 - 2\\bar x\\bar y\\cos\\phi$, where $\\phi$ is the angle between the link vector and
    $\\vec y$ and $p_n(r)$ is the probability density (per unit volume, normalised to unit integral over all end-to-end
    vectors) of the end-to-end vector of a freely jointed chain of $n$ rigid unit links; no further additive constant
    is included. Below a critical extension $\\bar y_c$ this landscape has an intact minimum $\\bar x_1$ near 1, a
    transition-state maximum $\\bar x_t$ and a broken minimum $\\bar x_2 > \\bar x_t$; the scission and healing barriers
    are $E_s = A(\\bar x_t) - A(\\bar x_1)$ and $E_h = A(\\bar x_t) - A(\\bar x_2)$.

    The probability $P$ that the chain is intact obeys the first-order kinetics
    $dP/dt = -(n+1)\\,\\nu\\,e^{-\\beta E_s(\\bar y)}\\,P + \\nu\\,e^{-\\beta E_h(\\bar y)}\\,(1-P)$: any one of the $n+1$ links may
    break, and the healing frequency equals the attempt frequency $\\nu$. Beyond $\\bar y_c$ no intact state exists
    ($E_s = 0$, $E_h = \\infty$).

    The chain is pulled at the constant rate $\gamma$ from $\bar y = 0$ (intact) until it is broken for good. A chain
    that breaks may heal and break again; the event recorded in a rupture experiment is the final scission, the one
    not followed by healing. Its probability density in the extension is
    $\rho_{s,last}(\bar y) = P(\bar y)\,k_s(\bar y)/\bar\gamma \times \exp[-\bar\gamma^{-1}\int_{\bar y}^{\infty} k_h(\bar y')\,d\bar y']$
    with $P$ the intact probability of the survival step, $k_s = (n+1)e^{-\beta E_s}$ and $k_h = e^{-\beta E_h}$ in units
    of $\nu$, and $k_h = 0$ beyond the critical extension; it integrates to one. This step returns the mean extension
    $\langle\bar y_s\rangle = \int \bar y\,\rho_{s,last}\,d\bar y$ and the mean rupture force
    $\langle\bar f_s\rangle = \int \bar f(\bar y)\,\rho_{s,last}\,d\bar y$ with $\bar f$ the intact-branch tension at the
    moment of scission. Both values are checked to $10^{-7}$; the normalisation of the density is a useful check of
    the integration.

    Returns a numpy float64 array of shape $(2,)$: $\langle\bar y_s\rangle$, $\langle\bar f_s\rangle$.

    Raises:
        ValueError: on an invalid n, beps or gbar, or if the density does not integrate to one within $10^{-7}$.
    """
    return None
```

### Step 11

chain_audit

Goal
----
Runs the whole chain at one parameter set: zero-extension barriers, equilibrium rupture statistics, critical point, pulled-chain survival and final-scission statistics with the exact fragment statistics, the Kuhn-Grun comparison, and the internal consistency checks.

```python
import numpy as np


def chain_audit(n: int, beps: float, gbar: float) -> np.ndarray:
    r"""n: integer at least 2, the number of rigid links. beps: positive float, $\beta\epsilon_0$. gbar: float in
    $(0, 10^{-2}]$, the normalised pulling rate $\bar\gamma = \gamma/(\nu l)$.

    The chain has $n+1$ identical links of Kuhn length $l$, freely jointed, with one end fixed and the other end held
    at the end-to-end vector $\\vec y$ (displacement control). One link, the breakable one, has a variable length $x$
    and the Lennard-Jones energy $V(x) = \\epsilon_0[(l/x)^{12} - 2(l/x)^6]$; the other $n$ links are rigid and freely
    jointed and are described by the exact end-to-end statistics of a freely jointed chain (no Gaussian or
    Kuhn-Grun approximation unless a step says so). Lengths are measured in $l$ ($\\bar x = x/l$, $\\bar y = y/l$),
    energies in $k_BT$ ($\\beta\\epsilon_0 = \\epsilon_0/k_BT$) and rates in the attempt frequency $\\nu$. Integrating the
    joint statistics of the breakable link and of the two rigid fragments over the orientation of the link gives the
    free energy of the chain at fixed $\\vec y$ as a function of the bond length,
    $\\beta A(\\bar x;\\bar y) = \\beta V(\\bar x) - \\ln\\big[\\bar x^2 \\int_0^\\pi p_n(\\bar r_\\phi)\\,\\sin\\phi\\,d\\phi\\big]$ with
    $\\bar r_\\phi^2 = \\bar x^2 + \\bar y^2 - 2\\bar x\\bar y\\cos\\phi$, where $\\phi$ is the angle between the link vector and
    $\\vec y$ and $p_n(r)$ is the probability density (per unit volume, normalised to unit integral over all end-to-end
    vectors) of the end-to-end vector of a freely jointed chain of $n$ rigid unit links; no further additive constant
    is included. Below a critical extension $\\bar y_c$ this landscape has an intact minimum $\\bar x_1$ near 1, a
    transition-state maximum $\\bar x_t$ and a broken minimum $\\bar x_2 > \\bar x_t$; the scission and healing barriers
    are $E_s = A(\\bar x_t) - A(\\bar x_1)$ and $E_h = A(\\bar x_t) - A(\\bar x_2)$.

    The probability $P$ that the chain is intact obeys the first-order kinetics
    $dP/dt = -(n+1)\\,\\nu\\,e^{-\\beta E_s(\\bar y)}\\,P + \\nu\\,e^{-\\beta E_h(\\bar y)}\\,(1-P)$: any one of the $n+1$ links may
    break, and the healing frequency equals the attempt frequency $\\nu$. Beyond $\\bar y_c$ no intact state exists
    ($E_s = 0$, $E_h = \\infty$).

    The orchestrator of the chain. It evaluates, with the exact fragment statistics, the zero-extension barriers, the
    equilibrium rupture statistics, the critical point, the pulled-chain survival at the equilibrium half-probability
    extension and the final-scission statistics at the given rate, and compares the equilibrium half-probability
    extension with the one obtained when the source's Kuhn-Grun landscape replaces the exact one (same critical-point
    and barrier definitions, applied to the Kuhn-Grun free energy). It also checks the chain against itself: the
    exact free energy at zero extension must agree with the density step through $\beta A(\bar x;0) = \beta V(\bar x) -
    \ln[2\bar x^2 p_n(\bar x)]$, the equilibrium probability at $\bar y_{1/2}$ must be one half, and the final-scission
    density must integrate to one; a failed check raises.

    Returns a numpy float64 array of shape $(13,)$: $\beta E_{s0}$ and $\beta E_{h0}$ (barriers at zero extension);
    $\bar y_{1/2}$; $\langle\bar y\rangle_{eq}$; $\langle\bar f\rangle_{eq}$; $\bar y_c$; $\bar f_c$;
    $P(\bar y_{1/2};\bar\gamma)$; $\langle\bar y_s\rangle$; $\langle\bar f_s\rangle$; the rate enhancement
    $\langle\bar f_s\rangle/\langle\bar f\rangle_{eq}$; the Kuhn-Grun half-probability extension $\bar y_{1/2}^{KG}$; and
    the Kuhn-Grun value of $\beta(E_s - E_h) - \ln(n+1)$ at the exact $\bar y_{1/2}$ (zero would mean the two
    landscapes agree there).

    Raises:
        ValueError: on an invalid n, beps or gbar, on a link without an intact well, or on a failed consistency
        check.
    """
    return None
```
