# Biology-Ecology-10

## Background

Dissolved-oxygen budgets in coastal and shelf waters are shaped by temperature-sensitive biological and chemical processes: primary production, phytoplankton mortality and respiration, organic-matter mineralization, nitrification, and denitrification. Process-based biogeochemical models increasingly give each process its own temperature sensitivity, because a single shared sensitivity can misattribute how warming redistributes oxygen sources and sinks. Quantifying how periodic, forced ecosystem states respond to each process-specific sensitivity shows which temperature responses control carbon, nitrogen, and oxygen cycling.
 
Many ecosystem models contain threshold rules, such as a nutrient preference that changes abruptly at a critical concentration. A threshold of this kind makes the model's right-hand side discontinuous. When the dynamics on both sides push the state toward the threshold, the physically meaningful solution follows the threshold itself, a sliding mode in the sense of Filippov, while a fixed-step integrator applied naively produces numerical chattering whose averages and parameter sensitivities are artifacts of the time step. A sliding mode is not permanent: when the forcing or a depleting resource makes one side stop pushing toward the threshold, the state leaves it, and the timing of that release is itself a sensitive ecological outcome. Handling such switches consistently is a prerequisite for credible sensitivity analysis of periodically forced, renewed systems such as mesocosms and flushed coastal basins.

## Problem

A process-based aquatic biogeochemical model assigns each biological and chemical rate its own temperature sensitivity and lets phytoplankton choose between ammonium and nitrate with a threshold rule that switches abruptly at a critical ammonium concentration. In a periodically renewed, nitrogen-drawn-down mesocosm, this switch can make the critical concentration an attracting surface that later releases the state as nitrate runs out, so a meaningful solution must follow the Filippov (sliding-mode) dynamics onto, along and off that surface rather than chatter across it. From the supplied seven-state specialization, forcing and process-specific Q10 values, compute the Frobenius norm of the matrix of periodic-cycle log-Q10 elasticities of ten cycle diagnostics.
 
Use the state order $\mathbf{y}=[O_2,\mathrm{Phy},POC_L,POC_S,DOC,NH_4,NO_3]$ and the Q10 order $\mathbf{q}=[Q_{10,\mu},Q_{10,m},Q_{10,r},Q_{10,\mathrm{Min}},Q_{10,\mathrm{Nit}},Q_{10,I}]$, applied through the cited OxyPOM source's process-specific temperature rule. Take one phytoplankton group with the source's light limitation (with its temperature-corrected reference intensity) and no light attenuation, nutrient limitation $DIN/(DIN+K_N)$ with $DIN=NH_4+NO_3$ and gross rate $\mu=f_I f_N\tau_\mu\mu^*$, the source's temperature-banded mortality $m$, growth-linked respiration $r$, ammonium-preference rule, total uptake $U_N=a_N(\mu-r)\mathrm{Phy}$, mineralization and degradation of the three organic pools, nitrification, and the source's partition of total mineralization $M_{\mathrm{tot}}$ between denitrification $D$ and the oxygen sink $M_{O_2}$, with the mineralization temperature factor in both redox weights. Hold lysis constant at $\lambda$, and write $L=(m+\lambda)\mathrm{Phy}$, $P=\mu\,\mathrm{Phy}$, $R=(r+c\,m)\mathrm{Phy}$ and $N_{O_2}=2N$ for nitrification $N$. The balances are $\dot O_2=k_{\mathrm{air}}(Sat-O_2)+P-R-N_{O_2}-M_{O_2}$, $\dot{\mathrm{Phy}}=(\mu-m-r-\lambda)\mathrm{Phy}$, $\dot{POC}_L=(1-f)L-d_{L\to S}-d_{L\to D}-M_L$, $\dot{POC}_S=d_{L\to S}-d_{S\to D}-M_S$, $\dot{DOC}=d_{L\to D}+d_{S\to D}-M_D$, $\dot{NH}_4=a_NfL+a_NM_{\mathrm{tot}}-U_{NH_4}-N$ and $\dot{NO}_3=N-D-U_{NO_3}$. With $\theta=2\pi t/\mathcal{P}$, the forcing is $T=T_0+A_T\sin\theta$, $I=I_0+A_I\sin(\theta-0.4)$, $k_{\mathrm{air}}=k_0+A_k\cos(\theta+0.2)$ and $Sat=Sat_0-\beta T$.
 
Integrate each period with classical RK4 on the grid $t_i=i\Delta t$ and follow the Filippov solution of the switched system: off the surface $NH_4=0.7$ the state follows the smooth one-sided branch of its side; on reaching the surface it slides, under the convex combination of the two one-sided fields that is tangent to the surface, when that surface is attracting, and otherwise crosses; while sliding it leaves the surface, on the side whose one-sided field stops pointing into it, as soon as the surface stops attracting. Every RK4 stage uses the field of the mode in which its step starts, and each mode change inside a grid interval is located as the first root of the ammonium level (arrival) or of the violated one-sided normal rate (exit) along a partial RK4 step from the start of the current sub-interval; a state that has just left the surface is not recaptured within that sub-interval. At each period's end apply $M(\mathbf{y},\mathbf{q})=\rho\,\mathbf{y}(\mathcal{P})+(1-\rho)\mathbf{y}_{\mathrm{renew}}$, iterate from `initial_cycle_start` and accept the first update whose infinity-norm change is at most `closure_tolerance`. On the periodic cycle, form $X=[\overline{O_2},\overline{POC_L+POC_S},\overline{\chi},\overline{NO_3},\overline{P},\overline{R},\overline{N_{O_2}},\overline{M_{O_2}},\varphi_s,t_{\mathrm{off}}]$. Here each overbar is an inclusive trapezoidal mean over the $N+1$ grid values of one period, $\chi$ is the ammonium share of nitrogen uptake (the equivalent share while sliding), $t_{\mathrm{off}}$ is the time sliding ends and $\varphi_s=(t_{\mathrm{off}}-t_{\mathrm{on}})/\mathcal P$ is the fraction of the period spent sliding after onset at $t_{\mathrm{on}}$. Return $\lVert E\rVert_F$ for $E_{jk}=X_j^{-1}\,dX_j/d\log Q_{10,k}$, the total derivative along the periodic branch including the response of the periodic start.
 
Use these exact inputs, with `parameters` $=[a_N,\mu^*,m^*,I^*,K_N,f,k_{\mathrm{Min},L},k_{\mathrm{Min},S},k_{\mathrm{Min},D},\kappa_{L\to S},\kappa_{L\to D},\kappa_{S\to D},r^*,c,\pi,k_{\mathrm{Nit}},K_{NH_4,\mathrm{Nit}},K_{O_2,\mathrm{Nit}},K_{NO_3},K_{O_2,\mathrm{den}},K'_{O_2,\mathrm{min}},\lambda]$ and `forcing_parameters` $=[T_0,A_T,\mathcal{P},I_0,A_I,k_0,A_k,Sat_0,\beta]$:
```python
initial_cycle_start = renewal_state = np.array([220., .2, 25., 40., 30., .4, 40.])
reference_q10 = np.array([1.4, 4.4, 1.1, 1.2, 1.12, 1.04])
parameters = np.array([.151, 1.2, .5, 30., .36, .5, .1, .01, .03, .5, .5, 1., .045, .7, .065, 28.6, 36., 31.3, 36., 94., 63., .05])
forcing_parameters = np.array([23., 2., 4., 90., 60., .2, .05, 340., 4.])
retention_fraction, time_step = .65, .04
closure_tolerance, max_cycles = 1.e-10, 500
```
 
In `<reasoning>`, state the source's light-limitation, mortality, ammonium-preference and denitrification-partition laws with the source equation numbers. Show why the critical-ammonium surface is attracting after onset, derive the equivalent ammonium share and the resulting sliding ammonium and nitrate balances, state the condition that ends sliding on the benchmark cycle and the side on which the state leaves, and give the total-derivative formula for the periodic elasticities. Report the periodic-start oxygen, phytoplankton, ammonium and nitrate, the onset and exit times and sliding fraction, the spectral radius of the renewal-map state Jacobian, the ten diagnostics, the six elasticity-column norms in Q10 order, the largest-magnitude signed elasticity with its diagnostic and Q10, and the final norm. Report state and diagnostic checkpoints to at least 10 significant figures and elasticity-derived values to at least 7.
 
Output Format Requirements:
Emit `<final_answer>` immediately, then `<reasoning>`. Do not write a long derivation before the tags.
You must emit exactly one finite decimal inside `<final_answer>...</final_answer>`, even if the value is approximate or you are unsure.
Rules:
- The tags are required. Do not omit them or leave them empty.
- The tagged value must be a finite decimal (examples: 0.4847, 0.22, 1.05). Not NaN, not Inf, not a fraction string, not a vector, not prose.
- Put only that one number between the tags. No units, no words, no extra lines.
Keep `<reasoning>` short (a few hundred words). Show only the laws, derivations and scalars requested above.
Do not paste the input arrays, full trajectories, full Jacobians, or the full elasticity matrix.

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

step_01_temperature_response_factors.py

Goal
----
Compute process-specific temperature multipliers from Q10 coefficients.

```python
def temperature_response_factors(
    temperature: float,
    q10_values: "np.ndarray",
    reference_temperature: float = 20.0,
) -> "np.ndarray":
    """Compute process-specific temperature multipliers from Q10 coefficients.

    Parameters:
        temperature: Finite water temperature in degrees Celsius.
        q10_values: Six finite positive process-specific Q10 values, except Step 1 accepts any nonempty length.
        reference_temperature: Finite reference temperature in degrees Celsius.

    Returns:
        A float64 multiplier vector in the input Q10 order.

    Raises:
        ValueError: If temperatures or Q10 values violate shape, finiteness, or positivity contracts.
    """
    return None
```

### Step 2

step_02_phytoplankton_process_rates.py

Goal
----
Compute eight ordered phytoplankton growth, mortality, respiration, loss, nutrient-uptake, and oxygen-budget values from the explicit formulas below.

```python
def phytoplankton_process_rates(
    temperature: float,
    light: float,
    phytoplankton: float,
    ammonium: float,
    nitrate: float,
    q10_values: "np.ndarray",
    parameters: "np.ndarray",
) -> "np.ndarray":
    """Compute the eight reduced phytoplankton process values.

    Required method:
        tau = temperature_response_factors(temperature, q10_values)
        I_star_T = parameters[3] * tau[5]
        f_light = 1 - exp(-light / I_star_T)
        DIN = ammonium + nitrate; DIN must be positive
        f_nutrient = DIN / (DIN + parameters[4])
        mu = f_light * f_nutrient * tau[0] * parameters[1]
        m = parameters[2] * tau[1] for T >= 20,
            parameters[2] for 5 < T < 20, and
            0.33 * parameters[2] for T <= 5
        r = parameters[14] * mu + (1 - parameters[14]) * parameters[12] * tau[2]
        L = (m + parameters[21]) * phytoplankton
        U_N = parameters[0] * (mu - r) * phytoplankton
        chi_NH4 = 1 if ammonium >= 0.7 else ammonium / DIN
        U_NH4 = chi_NH4 * U_N; U_NO3 = (1 - chi_NH4) * U_N
        P = mu * phytoplankton
        R = (r + parameters[13] * m) * phytoplankton

    Parameters:
        temperature: Finite water temperature in degrees Celsius.
        light: Nonnegative incident light.
        phytoplankton: Nonnegative phytoplankton carbon.
        ammonium: Nonnegative ammonium.
        nitrate: Nonnegative nitrate.
        q10_values: Six finite positive values in growth, mortality,
            respiration, mineralization, nitrification, optimal-light order.
        parameters: The 22-value finite positive parameter vector.

    Returns:
        float64 [mu, m, r, L, U_NH4, U_NO3, P, R].

    Raises:
        ValueError: If shapes, finiteness, or positivity contracts fail,
            or ammonium plus nitrate is zero.
    """
    return None
```

### Step 3

step_03_organic_matter_transformations.py

Goal
----
Compute seven ordered mineralization and transformation values for labile POC, semilabile POC, and DOC from the explicit rate laws below.

```python
def organic_matter_transformations(
    temperature: float,
    poc_labile: float,
    poc_semilabile: float,
    doc: float,
    q10_values: "np.ndarray",
    parameters: "np.ndarray",
) -> "np.ndarray":
    """Compute mineralization and transformation fluxes for three pools.

    Required method:
        tau_min = temperature_response_factors(temperature, q10_values)[3]
        M_L = parameters[6] * tau_min * poc_labile
        M_S = parameters[7] * tau_min * poc_semilabile
        M_D = parameters[8] * tau_min * doc
        d_L_to_S = parameters[9] * M_L
        d_L_to_D = parameters[10] * M_L
        d_S_to_D = parameters[11] * M_S
        M_tot = M_L + M_S + M_D

    Parameters:
        temperature: Finite water temperature in degrees Celsius.
        poc_labile: Nonnegative labile POC.
        poc_semilabile: Nonnegative semilabile POC.
        doc: Nonnegative DOC.
        q10_values: Six finite positive process-specific Q10 values.
        parameters: The 22-value finite positive parameter vector.

    Returns:
        float64 [M_L, M_S, M_D, d_L_to_S, d_L_to_D, d_S_to_D, M_tot].

    Raises:
        ValueError: If shapes, finiteness, or positivity contracts fail.
    """
    return None
```

### Step 4

step_04_nitrogen_oxygen_fluxes.py

Goal
----
Compute five ordered nitrification, denitrification, and oxygen-consumption values from the explicit redox rate laws below.

```python
def nitrogen_oxygen_fluxes(
    temperature: float,
    oxygen: float,
    ammonium: float,
    nitrate: float,
    total_mineralization: float,
    q10_values: "np.ndarray",
    parameters: "np.ndarray",
) -> "np.ndarray":
    """Compute nitrification, denitrification, and oxygen sinks.

    Required method:
        tau = temperature_response_factors(temperature, q10_values)
        N = parameters[15] * tau[4]
            * ammonium / (ammonium + parameters[16])
            * oxygen / (oxygen + parameters[17])
        phi_N = nitrate / (nitrate + parameters[18])
            * (1 - oxygen / (oxygen + parameters[19])) * tau[3]
        phi_O2 = oxygen / (oxygen + parameters[20]) * tau[3]
        f_den = phi_N / (phi_N + phi_O2)
        D = f_den * total_mineralization
        N_O2 = 2 * N
        M_O2 = (1 - f_den) * total_mineralization

    Parameters:
        temperature: Finite water temperature in degrees Celsius.
        oxygen: Nonnegative dissolved oxygen.
        ammonium: Nonnegative ammonium.
        nitrate: Nonnegative nitrate.
        total_mineralization: Nonnegative total carbon mineralization.
        q10_values: Six finite positive process-specific Q10 values.
        parameters: The 22-value finite positive parameter vector.

    Returns:
        float64 [N, f_den, D, N_O2, M_O2].

    Raises:
        ValueError: If shapes, finiteness, or positivity contracts fail.
    """
    return None
```

### Step 5

step_05_oxypom_reduced_rhs.py

Goal
----
Assemble the seven-state reduced OxyPOM derivative and four diagnostic fluxes from the explicit forcing functions and mass-balance equations below.

```python
def oxypom_reduced_rhs(
    time: float,
    state: "np.ndarray",
    q10_values: "np.ndarray",
    parameters: "np.ndarray",
    forcing_parameters: "np.ndarray",
) -> "tuple[np.ndarray, np.ndarray]":
    """Assemble the seven-state derivative and four diagnostic fluxes.

    State order:
        [O2, Phy, POC_L, POC_S, DOC, NH4, NO3]
    Forcing order:
        [T0, A_T, period, I0, A_I, k0, A_k, Sat0, beta]

    Required forcing:
        theta = 2*pi*time/period
        T = T0 + A_T*sin(theta)
        I = I0 + A_I*sin(theta - 0.4)
        k_air = k0 + A_k*cos(theta + 0.2)
        Sat = Sat0 - beta*T
        A = k_air*(Sat - O2)

    Required balances, using the ordered outputs of Steps 2-4:
        dO2 = A + P - R - N_O2 - M_O2
        dPhy = Phy*(mu - m - r - parameters[21])
        dPOC_L = (1 - parameters[5])*L - d_L_to_S - d_L_to_D - M_L
        dPOC_S = d_L_to_S - d_S_to_D - M_S
        dDOC = d_L_to_D + d_S_to_D - M_D
        dNH4 = parameters[0]*parameters[5]*L
            + parameters[0]*M_tot - U_NH4 - N
        dNO3 = N - D - U_NO3
        L from Step 2 is (m + parameters[21])*Phy.

    Parameters:
        time: Finite time in days.
        state: Seven finite nonnegative states in the declared order.
        q10_values: Six finite positive process-specific Q10 values.
        parameters: The 22-value finite positive parameter vector.
        forcing_parameters: Nine finite forcing values in the declared order.

    Returns:
        The seven-state derivative and [P, R, N_O2, M_O2].

    Raises:
        ValueError: If a contract fails, period is nonpositive, or forcing
            produces negative light, air exchange, or oxygen saturation.
    """
    return None
```

### Step 6

step_06_ammonium_sliding_field.py

Goal
----
Evaluate the one-sided ammonium rates and the Filippov sliding vector field on the critical-ammonium switching surface of the reduced OxyPOM model.

```python
def ammonium_sliding_field(
    time: float,
    state: "np.ndarray",
    q10_values: "np.ndarray",
    parameters: "np.ndarray",
    forcing_parameters: "np.ndarray",
) -> "np.ndarray":
    """Return the one-sided normal rates, sliding coefficient, share, and field.
 
    The ammonium entry of state is ignored and replaced by the critical
    concentration c = 0.7; every other entry is used as supplied.
 
    Parameters:
        time: Finite time in days, used for the Step-5 forcing.
        state: Seven finite nonnegative values [O2, Phy, POC_L, POC_S, DOC, NH4, NO3].
        q10_values: Six finite positive Q10 values in the declared process order.
        parameters: The 22-value finite positive parameter vector.
        forcing_parameters: Nine finite forcing values in the Step-5 order.
 
    Returns:
        float64 array of length 11, [f_minus_NH4, f_plus_NH4, alpha, chi_eq,
        f_s[0], ..., f_s[6]], where f_s is the (possibly extended) sliding
        field in the declared state order with an ammonium component of
        exactly zero. alpha lies in (0, 1) exactly when the surface is
        attracting.
 
    Raises:
        ValueError: If an input contract fails, or if f_minus_NH4 equals
            f_plus_NH4 so that no sliding combination exists.
    """
    return None
```

### Step 7

step_07_integrate_filippov_cycle_rk4.py

Goal
----
Integrate one forcing period of the Filippov solution of the reduced OxyPOM model, handling arrival at, sliding on, crossing of, and exit from the critical-ammonium surface, and apply the terminal renewal map.

```python
def integrate_filippov_cycle_rk4(
    cycle_start: "np.ndarray",
    q10_values: "np.ndarray",
    parameters: "np.ndarray",
    forcing_parameters: "np.ndarray",
    renewal_state: "np.ndarray",
    retention_fraction: float,
    time_step: float,
) -> "tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]":
    """Integrate one Filippov forcing cycle and apply the renewal map.
 
    Parameters:
        cycle_start: Seven finite nonnegative initial states in the declared order.
        q10_values: Six finite positive Q10 values in the declared process order.
        parameters: The 22-value finite positive parameter vector.
        forcing_parameters: Nine finite forcing values in the Step-5 order;
            entry 2 is the period P.
        renewal_state: Seven finite nonnegative renewal-water states.
        retention_fraction: Finite renewal retention fraction rho, 0 < rho < 1.
        time_step: Finite positive step; N = round(P / time_step) must satisfy
            N >= 1 and |N * time_step - P| <= 1e-12.
 
    Returns:
        times, states, fluxes, next_cycle_start, events, where times has shape
        (N + 1,); states holds the Filippov states at the grid times, shape
        (N + 1, 7); fluxes holds the Step-5 diagnostic fluxes [P, R, N_O2,
        M_O2] at every grid state, shape (N + 1, 4); next_cycle_start is
        rho * states[-1] + (1 - rho) * renewal_state, shape (7,); and events
        is a float64 array of shape (m, 2) listing, in time order, [time, code]
        for every mode change in the cycle (m may be 0). Codes: 0 = start of
        sliding (arrival, or a sliding start at t = 0); -1 = exit from sliding
        to below; 1 = exit from sliding to above; -2 = crossing downward
        through the surface; 2 = crossing upward through the surface. Event
        times must be located to an absolute accuracy of 1e-12 days or better.
 
    Raises:
        ValueError: If an input contract fails.
        RuntimeError: If a cycle starts on the surface with f_minus_NH4 <= 0
            <= f_plus_NH4 (repelling), if Step 6 cannot form a sliding field
            during a sliding step, if more than four events occur within one
            grid interval, or if an accepted grid state is nonfinite or negative.
    """
    return times, states, fluxes, next_cycle_start, events
```

### Step 8

step_08_filippov_periodic_q10_elasticities.py

Goal
----
Converge the renewed Filippov cycle to its periodic fixed point and return its ten cycle diagnostics, their total log-Q10 elasticities, and the stability radius of the renewal map.

```python
def filippov_periodic_q10_elasticities(
    initial_cycle_start: "np.ndarray",
    reference_q10: "np.ndarray",
    parameters: "np.ndarray",
    forcing_parameters: "np.ndarray",
    renewal_state: "np.ndarray",
    retention_fraction: float,
    time_step: float,
    closure_tolerance: float,
    max_cycles: int,
) -> "tuple[np.ndarray, np.ndarray, np.ndarray, float]":
    """Return the periodic start, diagnostics, elasticities, and map radius.
 
    The periodic start is obtained by iterating the Step-7 renewal map from
    initial_cycle_start and accepting the first update y_(n+1) with
    max |y_(n+1) - y_(n)| <= closure_tolerance. All derivatives are exact
    derivatives of the discrete Step-7 map; any method that reproduces them
    to a relative accuracy of 1e-7 or better is acceptable.
 
    Parameters:
        initial_cycle_start: Seven finite nonnegative initial states.
        reference_q10: Six finite positive Q10 values in the declared process order.
        parameters: The 22-value finite positive parameter vector.
        forcing_parameters: Nine finite forcing values in the Step-5 order.
        renewal_state: Seven finite nonnegative renewal-water states.
        retention_fraction: Finite renewal retention fraction, 0 < rho < 1.
        time_step: Positive RK4 step that divides the period under Step 7.
        closure_tolerance: Finite positive infinity-norm closure tolerance.
        max_cycles: Positive integer maximum number of renewal-map iterations.
 
    Returns:
        cycle_start, reference_diagnostics, elasticity_matrix, spectral_radius,
        where cycle_start is the accepted periodic start, shape (7,);
        reference_diagnostics is X in the declared order, shape (10,);
        elasticity_matrix is E with shape (10, 6), rows following X and columns
        following the Q10 order; and spectral_radius is the float spectral
        radius of the renewal-map state Jacobian at cycle_start.
 
    Raises:
        ValueError: If an input contract fails.
        RuntimeError: If the renewal map does not close within max_cycles, if
            the periodic cycle does not start sliding at a positive time or
            does not leave the surface afterwards,
            if Step 7 raises RuntimeError, if I - dM/dy is singular, or if a
            diagnostic is zero or nonfinite.
    """
    return cycle_start, reference_diagnostics, elasticity_matrix, spectral_radius
```

### Step 9

step_09_filippov_q10_sensitivity_norm.py

Goal
----
Run the complete Filippov periodic-sensitivity workflow and return the Frobenius norm of its 10-by-6 log-Q10 elasticity matrix.

```python
def filippov_q10_sensitivity_norm(
    initial_cycle_start: "np.ndarray",
    reference_q10: "np.ndarray",
    parameters: "np.ndarray",
    forcing_parameters: "np.ndarray",
    renewal_state: "np.ndarray",
    retention_fraction: float,
    time_step: float,
    closure_tolerance: float,
    max_cycles: int,
) -> float:
    """Return the Frobenius norm of the Step-8 elasticity matrix.
 
    Parameters:
        initial_cycle_start: Seven finite nonnegative initial states.
        reference_q10: Six finite positive Q10 values in the declared process order.
        parameters: The 22-value finite positive parameter vector.
        forcing_parameters: Nine finite forcing values in the Step-5 order.
        renewal_state: Seven finite nonnegative renewal-water states.
        retention_fraction: Finite renewal retention fraction, 0 < rho < 1.
        time_step: Positive RK4 step that divides the period under Step 7.
        closure_tolerance: Finite positive infinity-norm closure tolerance.
        max_cycles: Positive integer maximum number of renewal-map iterations.
 
    Returns:
        The Frobenius norm of the 10-by-6 elasticity matrix as a native float.
 
    Raises:
        ValueError: If an input contract of Step 8 fails.
        RuntimeError: If Step 8 raises RuntimeError.
    """
    return sensitivity_norm
```
