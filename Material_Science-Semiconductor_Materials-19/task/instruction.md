# Material_Science-Semiconductor_Materials-19

## Background

Task built 2026-09-13 on a CC BY 4.0 preprint (no code or data deposited). Chain: the source's deterministic reduced electrothermal model of an ion strike integrated to high accuracy; its two burnout indicators with the window maximum of the temperature; the deterministic threshold within the observation window and its window-free counterpart defined by the unconstrained temperature peak; the feedback boundary of the source's phase diagram; the linear-noise covariance along the deterministic trajectory with the multiplicative carrier noise and the additive thermal noise; the Gaussian estimate of the burnout probability as the largest marginal exceedance of the boundary over the window; the Gaussian transition quantiles and width; the fixed-crossing-time Freidlin-Wentzell action from the Hamiltonian two-point boundary value problem; the rare-event exponent as the minimum over admissible crossing times fixed by the terminal Hamiltonian; and the orchestrator with its consistency checks. Validation: the source's printed deposited-work factors and thermal indicators of its recoverable and runaway reference strikes are reproduced, the Gaussian estimate equals one half at the deterministic threshold by construction, the terminal Hamiltonian is checked against the finite difference of the fixed-time action, the boundary value problem is checked for stability under changes of initial guess, mesh and tolerance, and the linear-noise probabilities and width are compared with Euler-Maruyama ensembles of the source's scheme. Target beyond the source: the sampled ensembles are replaced by deterministic theory, and the rare-event exponent of a deterministically recoverable strike, which no ensemble of the source's size can resolve, is evaluated exactly and compared with its linear-noise (Gaussian) counterpart. The fork table, the secondary numbers and their alternatives are recorded in the golden solution and in the writer response, not here.

## Problem

Single-event burnout of a silicon carbide power MOSFET begins when a heavy ion deposits a dense carrier track in the high-field part of the blocking region: the carriers are multiplied by impact ionisation, the current heats the region, and if avalanche feedback outruns carrier extraction and heat spreading the device fails irreversibly. A recent source reduces this to two collective variables of the sensitive region, an excess carrier population and a normalised temperature, driven by a pulsed ionisation source, coupled through a carrier-dependent field and a temperature-dependent avalanche factor, and relaxed by carrier loss and thermal conduction, and it formulates burnout as the first passage of the temperature to an absorbing boundary within a finite observation window. It adds multiplicative carrier noise and additive thermal noise, and it studies the resulting probabilistic transition band, the noise-induced burnout of deterministically recoverable strikes, the first-passage statistics and a feedback-relaxation phase diagram entirely by Monte Carlo ensembles of a few thousand Euler-Maruyama trajectories, with a binomial standard error near one percent. Using the uploaded source as the authoritative reference, recover its construction, validate a solver of your own against the deterministic checks it reports, and take the step it does not: replace the sampled ensembles by deterministic theory, the linear-noise (Gaussian) description of the transition band and the small-noise rare-event asymptotics of subthreshold burnout, and report how far the two descriptions disagree on the rare-event exponent of a recoverable strike.

Everything the source fixes is to be taken from it and not from a plausible alternative: the dimensionless reduced model with its pulse, field-redistribution, avalanche and energy-injection conventions; the definition of the thermal indicator as a maximum over the finite observation window; the Ito noise structure with carrier noise proportional to the population and the fixed link between the two noise amplitudes; the definition of the burnout probability and of the transition width; the coordinates of the phase diagram; and the baseline parameter set.

Conventions fixed here. In the dimensionless time $s$ the carrier population $n$ and the normalised temperature $\Theta$ obey $dn/ds = g(s;\ell) + [f(e(n), \Theta) - 1]\,n$ and $d\Theta/ds = \kappa\,e(n)\,n - r\,\Theta$ from $n(0) = \Theta(0) = 0$, with the ionisation pulse $g(s;\ell) = \ell\,(2\pi)^{-1/2}\sigma_s^{-1}\exp[-(s - s_0)^2/(2\sigma_s^2)]$, the field $e(n) = b\,[1 + \eta_n\,n/(n + n_s)]$ and the avalanche factor $f(e, \Theta) = A_f\exp[-B_f(1 + a_\Theta\Theta)/e]$. The deposited-work factor is $\Lambda_{dep} = \int_0^{s_c}\kappa\,e\,n\,ds$ and the thermal indicator is $\Lambda_{th} = \max_{0 \le s \le s_c}\Theta(s)$, the maximum taken over the whole window including its end; the deterministic threshold $\ell^*$ solves $\Lambda_{th} = 1$, and its window-free counterpart $\ell_{pk}$ makes the first temperature peak after the pulse, followed beyond the window, equal to one; on the boundary $\Lambda_{th} = 1$ the reference avalanche factor is $F^* = A_f^*\exp(-B_f/b)$, $A_f$ being varied at fixed $B_f$ and $a_\Theta$. The stochastic model is $dn = [g + (f - 1)n]\,ds + \sigma_n\,n\,dW_n$, $d\Theta = [\kappa e n - r\Theta]\,ds + \sigma_\Theta\,dW_\Theta$ in the Ito sense with independent Wiener processes and $\sigma_n = \sigma$, $\sigma_\Theta = 0.35\,\sigma$; burnout is $\Theta$ reaching one within the window. The linear-noise description is the covariance $\Sigma(s)$ of the fluctuations of $(n, \Theta)$ about the deterministic trajectory to leading order in $\sigma$, with the noise intensities evaluated on that trajectory, written as $\Sigma = \sigma^2\hat\Sigma$; it estimates the burnout probability by the largest Gaussian marginal exceedance over the window, $P_G = \max_{0 \le s \le s_c}\Phi\big((\bar\Theta - 1)/\sqrt{\Sigma_{\Theta\Theta}}\big)$ with $z_G = \max_s(\bar\Theta - 1)/\sqrt{\hat\Sigma_{\Theta\Theta}}$ the maximal standardised margin per unit $\sigma$, and the transition width is $\Delta\ell = \ell_{0.9} - \ell_{0.1}$ with $P_G(\ell_q) = q$. The rare-event exponent is the Freidlin-Wentzell quasi-potential of the boundary within the window: with the diffusion written as $\sigma^2 D$, $D = \mathrm{diag}(n^2, 0.35^2)$, the action of a path is $S[x] = \tfrac12\int(\dot x - A)^T D^{-1}(\dot x - A)\,ds$, $S(s_f)$ is its minimum over all paths from $(0, 0)$ at $s = 0$ that reach $\Theta = 1$ at the time $s_f$, and $S^* = \min_{s_0 < s_f \le s_c}S(s_f)$, so that the burnout probability of a recoverable strike behaves as $\exp(-S^*/\sigma^2)$ for weak noise; its Gaussian counterpart is $S_G = z_G^2/2$. Every quantity is exact and deterministic; nothing is sampled.

The audit of a strike returns thirteen float64 values in this order: $\Lambda_{dep}$; $\Lambda_{th}$; $\ell^*$; $\ell_{pk}$; $F^*$ at the relaxation strength of the parameter set; $P_G$; $\ell_{0.1}$; $\ell_{0.9}$; $\Delta\ell$; $S^*$; the optimal crossing time; $S_G$; and the ratio $S_G/S^*$. It is evaluated at the source's baseline parameters $s_0 = 2$, $\sigma_s = 0.18$, $b = 1.20$, $\eta_n = 0.35$, $n_s = 1$, $\kappa = 0.16$, $r = 0.18$, $A_f = 5.5$, $B_f = 2.2$, $a_\Theta = 0.12$, $s_c = 12$, for the strike $\ell = 0.70$ and the noise amplitude $\sigma = 0.08$.

In your reasoning report the conventions you used, and justify each from the source: the reduced electrothermal model and the physical meaning of $\ell$, $b$, $\kappa$, $r$ and the avalanche parameters; the finite observation window and the definition of the thermal indicator as its maximum, with the reason the deposited-work factor is not a burnout criterion; the Ito stochastic equations, the multiplicative form of the carrier noise and the fixed link between the two amplitudes; the definition of the burnout probability as a first passage within the window and of the transition width by its quantiles; the coordinates of the feedback-relaxation phase diagram and how the feedback coordinate is varied; and the source's numerical method with its ensemble size, time step and the standard error it quotes. State how the linear-noise covariance is propagated along the deterministic trajectory and why $\hat\Sigma$ does not depend on the noise amplitude; state the value the Gaussian estimate takes at the deterministic threshold and why it takes that value; and state what fixes the optimal crossing time of the rare-event path and how you decided whether it lies at the end of the window or inside it.

Report numerically, as evidence that the chain was executed: as checks against the source, the deposited-work factor and the thermal indicator of the recoverable strike $\ell = 0.70$ and of the runaway strike $\ell = 0.85$ at the baseline parameters, and the baseline reference avalanche factor $f(b, 0)$. Then for the evaluated point: $\ell^*$ and $\ell_{pk}$ with the time of the unconstrained peak; $F^*$ at $r = 0.18$; the three entries of $\hat\Sigma$ at the end of the window; $P_G$ at $\sigma = 0.08$ and at $\sigma = 0.025$; $\ell_{0.1}$, $\ell_{0.9}$ and $\Delta\ell$ at $\sigma = 0.08$ and $\Delta\ell$ at $\sigma = 0.16$; the fixed-time action $S(s_f)$ at $s_f = 10$ and $s_f = 11$; $S^*$ with its optimal crossing time; $S_G$; and the ratio $S_G/S^*$. State how $S^*$ would change if the crossing were allowed after the window closes, if the thermal noise amplitude were taken equal to the carrier one, and if the carrier noise were additive rather than proportional to the population, and how $\Delta\ell$ at $\sigma = 0.08$ would change if the window were ignored and if the temperature dependence of the avalanche factor were left out of the Jacobian. These are the scalars that determine the final number.

As the final answer, report the ratio $S_G/S^*$ of the Gaussian to the exact rare-event exponent of noise-induced burnout of the strike $\ell = 0.70$ at the baseline parameters, to five significant figures.

## Output format

```
Output Format Requirements:
Emit <final_answer> immediately, then <reasoning>. Do not write a long derivation before the tags.
You must emit exactly one finite decimal inside <final_answer>...</final_answer>, even if the value is approximate or you are unsure.
Rules:
- The tags are required. Do not omit them or leave them empty.
- The tagged value must be a finite decimal (examples: 0.4847, 0.22, 1.05). Not NaN, not Inf, not a fraction string, not a vector, not prose.
- Put only that one number between the tags. No units, no words, no extra lines.
Keep <reasoning> concise: state the conventions you adopted and report the values the problem asks for, as a compact table or a short list of labelled values, with the short explanations and source attributions it requests.
Do not paste the input matrices, full coefficient vectors or per-iteration output.
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

seb_response

Goal
----
Integrates the source's deterministic reduced electrothermal model of an ion strike and returns the carrier population, the normalised temperature and the accumulated electrothermal input at requested times.

```python
def seb_response(params: "np.ndarray", ell: float, s_eval: "np.ndarray") -> "np.ndarray":
    r"""Integrates the source's deterministic reduced electrothermal model of an ion strike and returns the carrier population, the normalised temperature and the accumulated electrothermal input at requested times.

    The source's reduced electrothermal model of single-event burnout in a SiC power MOSFET has two collective
    variables of the ion-strike sensitive region, the excess carrier population $n(s)$ and the normalised temperature
    $\Theta(s) = (T - T_0)/(T_c - T_0)$, in the dimensionless time $s = t/\tau_{loss}$. The ion strike injects
    carriers through the pulse $g(s;\ell) = \ell\,(2\pi)^{-1/2}\sigma_s^{-1}\exp[-(s - s_0)^2/(2\sigma_s^2)]$ of
    ionisation strength $\ell$; the coarse-grained field is $e(n) = b\,[1 + \eta_n\,n/(n + n_s)]$; the avalanche
    factor is $f(e, \Theta) = A_f\exp[-B_f(1 + a_\Theta\Theta)/e]$; the deterministic dynamics is $dn/ds = g(s;\ell) +
    [f(e(n), \Theta) - 1]\,n$ and $d\Theta/ds = \kappa\,e(n)\,n - r\,\Theta$ from $n(0) = \Theta(0) = 0$. Burnout is
    the first passage of $\Theta$ to the absorbing boundary $\Theta = 1$ within the observation window $0 \le s \le
    s_c$. The eleven model parameters are passed as one array in the fixed order $[s_0, \sigma_s, b, \eta_n, n_s,
    \kappa, r, A_f, B_f, a_\Theta, s_c]$. In the stochastic model the two variables obey the Ito equations $dn = [g +
    (f - 1)n]\,ds + \sigma_n\,n\,dW_n$ and $d\Theta = [\kappa e n - r\Theta]\,ds + \sigma_\Theta\,dW_\Theta$ with
    independent Wiener processes and the amplitudes linked by $\sigma_n = \sigma$, $\sigma_\Theta = 0.35\,\sigma$.

    This step integrates the deterministic (noise-free) response and returns, at the requested times, the carrier
    population, the normalised temperature and the accumulated electrothermal input $w(s) = \int_0^s
    \kappa\,e(n(u))\,n(u)\,du$ (the deposited work in units of the thermal margin). The deterministic trajectory is
    followed through the boundary $\Theta = 1$ without stopping. Values must be accurate to $10^{-10}$, which requires
    a high-order integrator with tight tolerances and steps that resolve the pulse.

    Args:
        params: array of eleven floats, the model parameters in the order stated below.
        ell: non-negative float, the ionisation strength.
        s_eval: array of evaluation times, each in $[0, s_c]$.

    Returns:
        A numpy float64 array of shape $(3, K)$ for $K$ evaluation times: row 0 is $n$, row 1 is $\Theta$, row 2 is
        $w$.

    Raises:
        ValueError: if params does not hold eleven finite numbers with positive pulse width, field, saturation scale,
        energy-injection scale, relaxation strength, avalanche scales and window, a non-negative pulse centre and
        thermal coefficient, a field-redistribution strength above $-1$ and a window that contains the pulse centre;
        if ell is negative or not finite; if any evaluation time lies outside $[0, s_c]$; or if the carrier population
        grows beyond $10^6$ (outside the reduced model's range).
    """
    return None
```

### Step 2

seb_indicators

Goal
----
Returns the source's two deterministic burnout indicators of a strike, the deposited-work factor and the window maximum of the normalised temperature, with the time of that maximum and the carrier peak.

```python
def seb_indicators(params: "np.ndarray", ell: float) -> "np.ndarray":
    r"""Returns the source's two deterministic burnout indicators of a strike, the deposited-work factor and the window maximum of the normalised temperature, with the time of that maximum and the carrier peak.

    The source's reduced electrothermal model of single-event burnout in a SiC power MOSFET has two collective
    variables of the ion-strike sensitive region, the excess carrier population $n(s)$ and the normalised temperature
    $\Theta(s) = (T - T_0)/(T_c - T_0)$, in the dimensionless time $s = t/\tau_{loss}$. The ion strike injects
    carriers through the pulse $g(s;\ell) = \ell\,(2\pi)^{-1/2}\sigma_s^{-1}\exp[-(s - s_0)^2/(2\sigma_s^2)]$ of
    ionisation strength $\ell$; the coarse-grained field is $e(n) = b\,[1 + \eta_n\,n/(n + n_s)]$; the avalanche
    factor is $f(e, \Theta) = A_f\exp[-B_f(1 + a_\Theta\Theta)/e]$; the deterministic dynamics is $dn/ds = g(s;\ell) +
    [f(e(n), \Theta) - 1]\,n$ and $d\Theta/ds = \kappa\,e(n)\,n - r\,\Theta$ from $n(0) = \Theta(0) = 0$. Burnout is
    the first passage of $\Theta$ to the absorbing boundary $\Theta = 1$ within the observation window $0 \le s \le
    s_c$. The eleven model parameters are passed as one array in the fixed order $[s_0, \sigma_s, b, \eta_n, n_s,
    \kappa, r, A_f, B_f, a_\Theta, s_c]$. In the stochastic model the two variables obey the Ito equations $dn = [g +
    (f - 1)n]\,ds + \sigma_n\,n\,dW_n$ and $d\Theta = [\kappa e n - r\Theta]\,ds + \sigma_\Theta\,dW_\Theta$ with
    independent Wiener processes and the amplitudes linked by $\sigma_n = \sigma$, $\sigma_\Theta = 0.35\,\sigma$.

    The source characterises the deterministic response by two indicators: the deposited-work factor $\Lambda_{dep} =
    w(s_c)$ and the thermal indicator $\Lambda_{th} = \max_{0 \le s \le s_c}\Theta(s)$, the maximum being taken over
    the whole observation window including its end point (the temperature may still be rising when the window closes).
    This step returns both, together with the time at which the window maximum of $\Theta$ is attained and the maximum
    of the carrier population over the window. The values of the maxima must be located to $10^{-9}$ (a grid maximum
    is not enough), whereas the time of a smooth maximum is conditioned only to about the square root of the accuracy
    of its value, so that time is expected to $10^{-6}$ and a case whose temperature maximum lies inside the window is
    compared at that tolerance. For a strike that never raises the temperature above the ambient value the time of the
    maximum is reported as 0.

    Args:
        params: array of eleven floats, the model parameters.
        ell: non-negative float, the ionisation strength.

    Returns:
        A numpy float64 array of shape $(4,)$: $\Lambda_{dep}$, $\Lambda_{th}$, the time of the window maximum of
        $\Theta$, and the window maximum of $n$.

    Raises:
        ValueError: on invalid params or ell as in the response step, or if the carrier population leaves the reduced
        model's range.
    """
    return None
```

### Step 3

seb_threshold

Goal
----
Returns the deterministic burnout threshold in ionisation strength within the observation window, its window-free counterpart defined by the unconstrained temperature peak, and the time of that peak.

```python
def seb_threshold(params: "np.ndarray") -> "np.ndarray":
    r"""Returns the deterministic burnout threshold in ionisation strength within the observation window, its window-free counterpart defined by the unconstrained temperature peak, and the time of that peak.

    The source's reduced electrothermal model of single-event burnout in a SiC power MOSFET has two collective
    variables of the ion-strike sensitive region, the excess carrier population $n(s)$ and the normalised temperature
    $\Theta(s) = (T - T_0)/(T_c - T_0)$, in the dimensionless time $s = t/\tau_{loss}$. The ion strike injects
    carriers through the pulse $g(s;\ell) = \ell\,(2\pi)^{-1/2}\sigma_s^{-1}\exp[-(s - s_0)^2/(2\sigma_s^2)]$ of
    ionisation strength $\ell$; the coarse-grained field is $e(n) = b\,[1 + \eta_n\,n/(n + n_s)]$; the avalanche
    factor is $f(e, \Theta) = A_f\exp[-B_f(1 + a_\Theta\Theta)/e]$; the deterministic dynamics is $dn/ds = g(s;\ell) +
    [f(e(n), \Theta) - 1]\,n$ and $d\Theta/ds = \kappa\,e(n)\,n - r\,\Theta$ from $n(0) = \Theta(0) = 0$. Burnout is
    the first passage of $\Theta$ to the absorbing boundary $\Theta = 1$ within the observation window $0 \le s \le
    s_c$. The eleven model parameters are passed as one array in the fixed order $[s_0, \sigma_s, b, \eta_n, n_s,
    \kappa, r, A_f, B_f, a_\Theta, s_c]$. In the stochastic model the two variables obey the Ito equations $dn = [g +
    (f - 1)n]\,ds + \sigma_n\,n\,dW_n$ and $d\Theta = [\kappa e n - r\Theta]\,ds + \sigma_\Theta\,dW_\Theta$ with
    independent Wiener processes and the amplitudes linked by $\sigma_n = \sigma$, $\sigma_\Theta = 0.35\,\sigma$.

    The deterministic burnout boundary of the source is the ionisation strength $\ell^*$ at which $\Lambda_{th}(\ell)
    = 1$ with the other parameters fixed. Because $\Lambda_{th}$ is a maximum over the finite window, this boundary
    depends on $s_c$; the window-free counterpart is the strength $\ell_{pk}$ at which the first maximum of $\Theta$
    after the pulse, followed beyond the window for as long as necessary, equals one. This step returns both
    thresholds, each accurate to $10^{-9}$, and the time of the unconstrained peak at $\ell_{pk}$; the time of a
    smooth maximum is conditioned only to about the square root of the accuracy of its value, so it is expected to
    $10^{-6}$ and the cases of this step are compared at that tolerance.

    Args:
        params: array of eleven floats, the model parameters.

    Returns:
        A numpy float64 array of shape $(3,)$: $\ell^*$, $\ell_{pk}$, and the time of the unconstrained peak at
        $\ell_{pk}$.

    Raises:
        ValueError: on invalid params, if no threshold exists below an ionisation strength of $10^4$, if the smallest
        ionisation strength already burns out, or if the temperature has not peaked within forty time units beyond the
        window.
    """
    return None
```

### Step 4

seb_feedback_boundary

Goal
----
Returns the reference avalanche factor on the source's deterministic feedback-relaxation boundary at a given relaxation strength and ionisation strength.

```python
def seb_feedback_boundary(params: "np.ndarray", ell: float, r: float) -> float:
    r"""Returns the reference avalanche factor on the source's deterministic feedback-relaxation boundary at a given relaxation strength and ionisation strength.

    The source's reduced electrothermal model of single-event burnout in a SiC power MOSFET has two collective
    variables of the ion-strike sensitive region, the excess carrier population $n(s)$ and the normalised temperature
    $\Theta(s) = (T - T_0)/(T_c - T_0)$, in the dimensionless time $s = t/\tau_{loss}$. The ion strike injects
    carriers through the pulse $g(s;\ell) = \ell\,(2\pi)^{-1/2}\sigma_s^{-1}\exp[-(s - s_0)^2/(2\sigma_s^2)]$ of
    ionisation strength $\ell$; the coarse-grained field is $e(n) = b\,[1 + \eta_n\,n/(n + n_s)]$; the avalanche
    factor is $f(e, \Theta) = A_f\exp[-B_f(1 + a_\Theta\Theta)/e]$; the deterministic dynamics is $dn/ds = g(s;\ell) +
    [f(e(n), \Theta) - 1]\,n$ and $d\Theta/ds = \kappa\,e(n)\,n - r\,\Theta$ from $n(0) = \Theta(0) = 0$. Burnout is
    the first passage of $\Theta$ to the absorbing boundary $\Theta = 1$ within the observation window $0 \le s \le
    s_c$. The eleven model parameters are passed as one array in the fixed order $[s_0, \sigma_s, b, \eta_n, n_s,
    \kappa, r, A_f, B_f, a_\Theta, s_c]$. In the stochastic model the two variables obey the Ito equations $dn = [g +
    (f - 1)n]\,ds + \sigma_n\,n\,dW_n$ and $d\Theta = [\kappa e n - r\Theta]\,ds + \sigma_\Theta\,dW_\Theta$ with
    independent Wiener processes and the amplitudes linked by $\sigma_n = \sigma$, $\sigma_\Theta = 0.35\,\sigma$.

    The source draws its feedback-relaxation phase diagram in the plane of the reference avalanche factor $F = f(b, 0)
    = A_f\exp(-B_f/b)$ and the relaxation strength $r$, varying $F$ through $A_f$ at fixed $B_f$ and $a_\Theta$. This
    step returns the value $F^*$ on the deterministic boundary $\Lambda_{th} = 1$ at the given $r$ and ionisation
    strength, accurate to $10^{-9}$.

    Args:
        params: array of eleven floats, the model parameters.
        ell: non-negative float, the ionisation strength.
        r: positive float, the thermal relaxation strength at which the boundary is evaluated (it replaces the value
        in params).

    Returns:
        A Python float: $F^*$.

    Raises:
        ValueError: on invalid params, ell or r, or if no boundary exists below an avalanche scale of $10^4$ or
        burnout persists as the avalanche feedback vanishes.
    """
    return None
```

### Step 5

seb_linear_noise

Goal
----
Propagates the linear-noise covariance of the carrier population and the normalised temperature along the deterministic trajectory, with the multiplicative carrier noise and the additive thermal noise of the source's stochastic model.

```python
def seb_linear_noise(params: "np.ndarray", ell: float, s_eval: "np.ndarray") -> "np.ndarray":
    r"""Propagates the linear-noise covariance of the carrier population and the normalised temperature along the deterministic trajectory, with the multiplicative carrier noise and the additive thermal noise of the source's stochastic model.

    The source's reduced electrothermal model of single-event burnout in a SiC power MOSFET has two collective
    variables of the ion-strike sensitive region, the excess carrier population $n(s)$ and the normalised temperature
    $\Theta(s) = (T - T_0)/(T_c - T_0)$, in the dimensionless time $s = t/\tau_{loss}$. The ion strike injects
    carriers through the pulse $g(s;\ell) = \ell\,(2\pi)^{-1/2}\sigma_s^{-1}\exp[-(s - s_0)^2/(2\sigma_s^2)]$ of
    ionisation strength $\ell$; the coarse-grained field is $e(n) = b\,[1 + \eta_n\,n/(n + n_s)]$; the avalanche
    factor is $f(e, \Theta) = A_f\exp[-B_f(1 + a_\Theta\Theta)/e]$; the deterministic dynamics is $dn/ds = g(s;\ell) +
    [f(e(n), \Theta) - 1]\,n$ and $d\Theta/ds = \kappa\,e(n)\,n - r\,\Theta$ from $n(0) = \Theta(0) = 0$. Burnout is
    the first passage of $\Theta$ to the absorbing boundary $\Theta = 1$ within the observation window $0 \le s \le
    s_c$. The eleven model parameters are passed as one array in the fixed order $[s_0, \sigma_s, b, \eta_n, n_s,
    \kappa, r, A_f, B_f, a_\Theta, s_c]$. In the stochastic model the two variables obey the Ito equations $dn = [g +
    (f - 1)n]\,ds + \sigma_n\,n\,dW_n$ and $d\Theta = [\kappa e n - r\Theta]\,ds + \sigma_\Theta\,dW_\Theta$ with
    independent Wiener processes and the amplitudes linked by $\sigma_n = \sigma$, $\sigma_\Theta = 0.35\,\sigma$.

    For weak noise the stochastic variables fluctuate around the deterministic trajectory, and to leading order the
    fluctuations are Gaussian with a covariance that obeys the linear-noise (Lyapunov) equation along that trajectory,
    $d\Sigma/ds = J\Sigma + \Sigma J^T + D$, with $J$ the Jacobian of the deterministic drift with respect to $(n,
    \Theta)$ (including the dependence of the field on $n$ and of the avalanche factor on both variables) and $D$ the
    Ito diffusion matrix of the stochastic model; $\Sigma(0) = 0$. This step returns the covariance per unit
    $\sigma^2$, i.e. $\hat\Sigma = \Sigma/\sigma^2$, which is independent of the noise amplitude, accurate to
    $10^{-9}$.

    Args:
        params: array of eleven floats, the model parameters.
        ell: non-negative float, the ionisation strength.
        s_eval: array of evaluation times in $[0, s_c]$.

    Returns:
        A numpy float64 array of shape $(3, K)$: $\hat\Sigma_{nn}$, $\hat\Sigma_{n\Theta}$ and
        $\hat\Sigma_{\Theta\Theta}$ at the evaluation times.

    Raises:
        ValueError: on invalid params, ell or evaluation times, or if the carrier population leaves the reduced
        model's range.
    """
    return None
```

### Step 6

seb_gaussian_burnout

Goal
----
Returns the linear-noise (Gaussian) estimate of the burnout probability of a strike within the observation window, the time of the most probable crossing and the maximal standardised margin to the boundary.

```python
def seb_gaussian_burnout(params: "np.ndarray", ell: float, sigma: float) -> "np.ndarray":
    r"""Returns the linear-noise (Gaussian) estimate of the burnout probability of a strike within the observation window, the time of the most probable crossing and the maximal standardised margin to the boundary.

    The source's reduced electrothermal model of single-event burnout in a SiC power MOSFET has two collective
    variables of the ion-strike sensitive region, the excess carrier population $n(s)$ and the normalised temperature
    $\Theta(s) = (T - T_0)/(T_c - T_0)$, in the dimensionless time $s = t/\tau_{loss}$. The ion strike injects
    carriers through the pulse $g(s;\ell) = \ell\,(2\pi)^{-1/2}\sigma_s^{-1}\exp[-(s - s_0)^2/(2\sigma_s^2)]$ of
    ionisation strength $\ell$; the coarse-grained field is $e(n) = b\,[1 + \eta_n\,n/(n + n_s)]$; the avalanche
    factor is $f(e, \Theta) = A_f\exp[-B_f(1 + a_\Theta\Theta)/e]$; the deterministic dynamics is $dn/ds = g(s;\ell) +
    [f(e(n), \Theta) - 1]\,n$ and $d\Theta/ds = \kappa\,e(n)\,n - r\,\Theta$ from $n(0) = \Theta(0) = 0$. Burnout is
    the first passage of $\Theta$ to the absorbing boundary $\Theta = 1$ within the observation window $0 \le s \le
    s_c$. The eleven model parameters are passed as one array in the fixed order $[s_0, \sigma_s, b, \eta_n, n_s,
    \kappa, r, A_f, B_f, a_\Theta, s_c]$. In the stochastic model the two variables obey the Ito equations $dn = [g +
    (f - 1)n]\,ds + \sigma_n\,n\,dW_n$ and $d\Theta = [\kappa e n - r\Theta]\,ds + \sigma_\Theta\,dW_\Theta$ with
    independent Wiener processes and the amplitudes linked by $\sigma_n = \sigma$, $\sigma_\Theta = 0.35\,\sigma$.

    In the linear-noise description the temperature at time $s$ is Gaussian with mean $\bar\Theta(s)$ (the
    deterministic value) and variance $\sigma^2\hat\Sigma_{\Theta\Theta}(s)$. The Gaussian estimate of the burnout
    probability within the window is the largest marginal probability of exceeding the boundary, $P_G = \max_{0 \le s
    \le s_c}\Phi\big((\bar\Theta(s) - 1)/(\sigma\sqrt{\hat\Sigma_{\Theta\Theta}(s)})\big)$ with $\Phi$ the standard
    normal distribution function; the maximum may sit at the end of the window. This step returns $P_G$, the time at
    which the maximum is attained and the maximal standardised margin $z_G = \max_s(\bar\Theta -
    1)/\sqrt{\hat\Sigma_{\Theta\Theta}}$ (independent of $\sigma$). $P_G$ and the margin must be accurate to
    $10^{-9}$; the time of a smooth maximum is conditioned only to about the square root of the accuracy of its value,
    so it is expected to $10^{-6}$ and a case whose maximising time lies inside the window is compared at that
    tolerance.

    Args:
        params: array of eleven floats, the model parameters.
        ell: non-negative float, the ionisation strength.
        sigma: positive float, the noise amplitude $\sigma$ (with $\sigma_n = \sigma$, $\sigma_\Theta = 0.35\sigma$).

    Returns:
        A numpy float64 array of shape $(3,)$: $P_G$, the maximising time, $z_G$.

    Raises:
        ValueError: on invalid params, ell or sigma, or if the carrier population leaves the reduced model's range.
    """
    return None
```

### Step 7

seb_gaussian_width

Goal
----
Returns the ionisation strengths at which the Gaussian burnout estimate equals 0.1 and 0.9 and the resulting transition width, the source's measure of stochastic threshold broadening.

```python
def seb_gaussian_width(params: "np.ndarray", sigma: float) -> "np.ndarray":
    r"""Returns the ionisation strengths at which the Gaussian burnout estimate equals 0.1 and 0.9 and the resulting transition width, the source's measure of stochastic threshold broadening.

    The source's reduced electrothermal model of single-event burnout in a SiC power MOSFET has two collective
    variables of the ion-strike sensitive region, the excess carrier population $n(s)$ and the normalised temperature
    $\Theta(s) = (T - T_0)/(T_c - T_0)$, in the dimensionless time $s = t/\tau_{loss}$. The ion strike injects
    carriers through the pulse $g(s;\ell) = \ell\,(2\pi)^{-1/2}\sigma_s^{-1}\exp[-(s - s_0)^2/(2\sigma_s^2)]$ of
    ionisation strength $\ell$; the coarse-grained field is $e(n) = b\,[1 + \eta_n\,n/(n + n_s)]$; the avalanche
    factor is $f(e, \Theta) = A_f\exp[-B_f(1 + a_\Theta\Theta)/e]$; the deterministic dynamics is $dn/ds = g(s;\ell) +
    [f(e(n), \Theta) - 1]\,n$ and $d\Theta/ds = \kappa\,e(n)\,n - r\,\Theta$ from $n(0) = \Theta(0) = 0$. Burnout is
    the first passage of $\Theta$ to the absorbing boundary $\Theta = 1$ within the observation window $0 \le s \le
    s_c$. The eleven model parameters are passed as one array in the fixed order $[s_0, \sigma_s, b, \eta_n, n_s,
    \kappa, r, A_f, B_f, a_\Theta, s_c]$. In the stochastic model the two variables obey the Ito equations $dn = [g +
    (f - 1)n]\,ds + \sigma_n\,n\,dW_n$ and $d\Theta = [\kappa e n - r\Theta]\,ds + \sigma_\Theta\,dW_\Theta$ with
    independent Wiener processes and the amplitudes linked by $\sigma_n = \sigma$, $\sigma_\Theta = 0.35\,\sigma$.

    The source measures stochastic threshold broadening by the transition width $\Delta\ell = \ell_{0.9} -
    \ell_{0.1}$, where $\ell_q$ is the ionisation strength at which the burnout probability equals $q$. This step
    returns the two quantiles and the width of the Gaussian estimate $P_G(\ell)$ of the burnout step, each accurate to
    $10^{-9}$.

    Args:
        params: array of eleven floats, the model parameters.
        sigma: positive float, the noise amplitude.

    Returns:
        A numpy float64 array of shape $(3,)$: $\ell_{0.1}$, $\ell_{0.9}$, $\Delta\ell$.

    Raises:
        ValueError: on invalid params or sigma, or if the Gaussian estimate does not reach 0.9 below an ionisation
        strength of $10^4$ or already exceeds 0.1 at the smallest strength considered.
    """
    return None
```

### Step 8

seb_instanton_action

Goal
----
Returns the minimum Freidlin-Wentzell action of a fluctuation path that carries the strike to the burnout boundary at a prescribed time, with the terminal Hamiltonian and the terminal carrier population of the optimal path.

```python
def seb_instanton_action(params: "np.ndarray", ell: float, s_f: float) -> "np.ndarray":
    r"""Returns the minimum Freidlin-Wentzell action of a fluctuation path that carries the strike to the burnout boundary at a prescribed time, with the terminal Hamiltonian and the terminal carrier population of the optimal path.

    The source's reduced electrothermal model of single-event burnout in a SiC power MOSFET has two collective
    variables of the ion-strike sensitive region, the excess carrier population $n(s)$ and the normalised temperature
    $\Theta(s) = (T - T_0)/(T_c - T_0)$, in the dimensionless time $s = t/\tau_{loss}$. The ion strike injects
    carriers through the pulse $g(s;\ell) = \ell\,(2\pi)^{-1/2}\sigma_s^{-1}\exp[-(s - s_0)^2/(2\sigma_s^2)]$ of
    ionisation strength $\ell$; the coarse-grained field is $e(n) = b\,[1 + \eta_n\,n/(n + n_s)]$; the avalanche
    factor is $f(e, \Theta) = A_f\exp[-B_f(1 + a_\Theta\Theta)/e]$; the deterministic dynamics is $dn/ds = g(s;\ell) +
    [f(e(n), \Theta) - 1]\,n$ and $d\Theta/ds = \kappa\,e(n)\,n - r\,\Theta$ from $n(0) = \Theta(0) = 0$. Burnout is
    the first passage of $\Theta$ to the absorbing boundary $\Theta = 1$ within the observation window $0 \le s \le
    s_c$. The eleven model parameters are passed as one array in the fixed order $[s_0, \sigma_s, b, \eta_n, n_s,
    \kappa, r, A_f, B_f, a_\Theta, s_c]$. In the stochastic model the two variables obey the Ito equations $dn = [g +
    (f - 1)n]\,ds + \sigma_n\,n\,dW_n$ and $d\Theta = [\kappa e n - r\Theta]\,ds + \sigma_\Theta\,dW_\Theta$ with
    independent Wiener processes and the amplitudes linked by $\sigma_n = \sigma$, $\sigma_\Theta = 0.35\,\sigma$.

    For weak noise the probability of a rare boundary crossing is governed by the Freidlin-Wentzell action of the most
    probable path. With the diffusion of the stochastic model written as $\sigma^2 D(n)$, $D = \mathrm{diag}(n^2,
    0.35^2)$, the action of a path $x(s) = (n, \Theta)$ is $S[x] = \tfrac12\int_0^{s_f}(\dot x - A)^T D^{-1}(\dot x -
    A)\,ds$ with $A$ the deterministic drift, so that the crossing probability behaves as $\exp(-S/\sigma^2)$. This
    step returns the minimum of $S$ over paths that start at $(0, 0)$ at $s = 0$ and reach $\Theta = 1$ exactly at $s
    = s_f$ with the carrier endpoint free, together with the value of the Hamiltonian $H = p\cdot A + \tfrac12 p^T D
    p$ of the optimal path at $s_f$ (which equals $-dS/ds_f$) and the carrier population of the optimal path at $s_f$.
    The minimiser is a solution of the Hamiltonian two-point boundary value problem; the action must be accurate to
    $10^{-9}$.

    Args:
        params: array of eleven floats, the model parameters.
        ell: non-negative float, the ionisation strength.
        s_f: float, the crossing time, in $(s_0, s_c]$.

    Returns:
        A numpy float64 array of shape $(3,)$: $S(s_f)$, $H(s_f)$, $n(s_f)$.

    Raises:
        ValueError: on invalid params or ell, if s_f is not in $(s_0, s_c]$, or if the boundary value problem does not
        converge.
    """
    return None
```

### Step 9

seb_subthreshold_rate

Goal
----
Returns the rare-event exponent of noise-induced subthreshold burnout within the observation window, the optimal crossing time and the Gaussian counterpart of the exponent.

```python
def seb_subthreshold_rate(params: "np.ndarray", ell: float) -> "np.ndarray":
    r"""Returns the rare-event exponent of noise-induced subthreshold burnout within the observation window, the optimal crossing time and the Gaussian counterpart of the exponent.

    The source's reduced electrothermal model of single-event burnout in a SiC power MOSFET has two collective
    variables of the ion-strike sensitive region, the excess carrier population $n(s)$ and the normalised temperature
    $\Theta(s) = (T - T_0)/(T_c - T_0)$, in the dimensionless time $s = t/\tau_{loss}$. The ion strike injects
    carriers through the pulse $g(s;\ell) = \ell\,(2\pi)^{-1/2}\sigma_s^{-1}\exp[-(s - s_0)^2/(2\sigma_s^2)]$ of
    ionisation strength $\ell$; the coarse-grained field is $e(n) = b\,[1 + \eta_n\,n/(n + n_s)]$; the avalanche
    factor is $f(e, \Theta) = A_f\exp[-B_f(1 + a_\Theta\Theta)/e]$; the deterministic dynamics is $dn/ds = g(s;\ell) +
    [f(e(n), \Theta) - 1]\,n$ and $d\Theta/ds = \kappa\,e(n)\,n - r\,\Theta$ from $n(0) = \Theta(0) = 0$. Burnout is
    the first passage of $\Theta$ to the absorbing boundary $\Theta = 1$ within the observation window $0 \le s \le
    s_c$. The eleven model parameters are passed as one array in the fixed order $[s_0, \sigma_s, b, \eta_n, n_s,
    \kappa, r, A_f, B_f, a_\Theta, s_c]$. In the stochastic model the two variables obey the Ito equations $dn = [g +
    (f - 1)n]\,ds + \sigma_n\,n\,dW_n$ and $d\Theta = [\kappa e n - r\Theta]\,ds + \sigma_\Theta\,dW_\Theta$ with
    independent Wiener processes and the amplitudes linked by $\sigma_n = \sigma$, $\sigma_\Theta = 0.35\,\sigma$.

    The rare-event exponent of noise-induced burnout within the window is $S^* = \min_{s_0 < s_f \le s_c} S(s_f)$, the
    smallest fixed-time action of the instanton step over admissible crossing times, so that the burnout probability
    of a deterministically recoverable strike behaves as $\exp(-S^*/\sigma^2)$ for weak noise. Because the terminal
    Hamiltonian is $-dS/ds_f$, the minimum sits at the end of the window when that Hamiltonian is non-negative there
    and otherwise at an interior crossing time where it vanishes (the free-time transversality condition). This step
    returns $S^*$, the optimal crossing time and the Gaussian (linear-noise) counterpart of the exponent, $S_{G} =
    z_G^2/2$ with $z_G$ the maximal standardised margin of the burnout step (negative below the deterministic
    threshold, where the deterministic temperature stays under the boundary throughout the window), each accurate to $10^{-9}$.

    Args:
        params: array of eleven floats, the model parameters.
        ell: non-negative float, the ionisation strength.

    Returns:
        A numpy float64 array of shape $(3,)$: $S^*$, the optimal crossing time, $S_G$.

    Raises:
        ValueError: on invalid params or ell, or if the instanton boundary value problem does not converge.
    """
    return None
```

### Step 10

seb_audit

Goal
----
Runs the whole chain at one strike: deterministic indicators and thresholds, the feedback boundary, the Gaussian burnout estimate and transition width, and the rare-event exponent with its Gaussian counterpart and their ratio, with the internal consistency checks.

```python
def seb_audit(params: "np.ndarray", ell: float, sigma: float) -> "np.ndarray":
    r"""Runs the whole chain at one strike: deterministic indicators and thresholds, the feedback boundary, the Gaussian burnout estimate and transition width, and the rare-event exponent with its Gaussian counterpart and their ratio, with the internal consistency checks.

    The source's reduced electrothermal model of single-event burnout in a SiC power MOSFET has two collective
    variables of the ion-strike sensitive region, the excess carrier population $n(s)$ and the normalised temperature
    $\Theta(s) = (T - T_0)/(T_c - T_0)$, in the dimensionless time $s = t/\tau_{loss}$. The ion strike injects
    carriers through the pulse $g(s;\ell) = \ell\,(2\pi)^{-1/2}\sigma_s^{-1}\exp[-(s - s_0)^2/(2\sigma_s^2)]$ of
    ionisation strength $\ell$; the coarse-grained field is $e(n) = b\,[1 + \eta_n\,n/(n + n_s)]$; the avalanche
    factor is $f(e, \Theta) = A_f\exp[-B_f(1 + a_\Theta\Theta)/e]$; the deterministic dynamics is $dn/ds = g(s;\ell) +
    [f(e(n), \Theta) - 1]\,n$ and $d\Theta/ds = \kappa\,e(n)\,n - r\,\Theta$ from $n(0) = \Theta(0) = 0$. Burnout is
    the first passage of $\Theta$ to the absorbing boundary $\Theta = 1$ within the observation window $0 \le s \le
    s_c$. The eleven model parameters are passed as one array in the fixed order $[s_0, \sigma_s, b, \eta_n, n_s,
    \kappa, r, A_f, B_f, a_\Theta, s_c]$. In the stochastic model the two variables obey the Ito equations $dn = [g +
    (f - 1)n]\,ds + \sigma_n\,n\,dW_n$ and $d\Theta = [\kappa e n - r\Theta]\,ds + \sigma_\Theta\,dW_\Theta$ with
    independent Wiener processes and the amplitudes linked by $\sigma_n = \sigma$, $\sigma_\Theta = 0.35\,\sigma$.

    The orchestrator of the chain. It evaluates the deterministic indicators of the strike, the two thresholds, the
    feedback boundary at the relaxation strength of params, the Gaussian burnout estimate of the strike at the given
    noise amplitude, the Gaussian transition quantiles and width at that amplitude, and the rare-event exponent of the
    strike with its optimal crossing time, its Gaussian counterpart and the ratio of the two. It checks the chain
    against itself: the Gaussian estimate at the deterministic threshold must equal one half, the deposited work of
    the indicator step must equal the accumulated input of the response step at the end of the window, the maximal
    standardised margin of the Gaussian step must be reproduced from the response and covariance steps at its
    maximising time, and the fixed-time action at the end of the window must not lie below the rare-event exponent; a
    failed check raises.

    Args:
        params: array of eleven floats, the model parameters.
        ell: non-negative float, the ionisation strength of the evaluated strike.
        sigma: positive float, the noise amplitude.

    Returns:
        A numpy float64 array of shape $(13,)$: $\Lambda_{dep}$; $\Lambda_{th}$; $\ell^*$; $\ell_{pk}$; $F^*$; $P_G$;
        $\ell_{0.1}$; $\ell_{0.9}$; $\Delta\ell$; $S^*$; the optimal crossing time; $S_G$; the ratio $S_G/S^*$ of the
        Gaussian to the exact rare-event exponent.

    Raises:
        ValueError: on invalid params, ell or sigma, or on a failed consistency check.
    """
    return None
```
