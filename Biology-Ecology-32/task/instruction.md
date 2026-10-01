# Biology-Ecology-32

## Background

Many animals change their ecological role as they grow, a pattern known as an ontogenetic niche shift. Small individuals of a species may be eaten by a competitor or prey species that they themselves eat once they are large. Examples include fish whose adults prey on another species whose adults eat their eggs and larvae, and insects whose larvae are attacked by the prey of the adults. Such mutual predation between life stages ("role reversal") links the fate of a predator population to how many of its young survive the juvenile period. Juvenile mortality in turn depends on the density of the species that the adults hunt.

Classical predator-prey models treat each population as a single number and cannot represent this stage-dependent interaction. Stage-structured and age-structured models follow the distribution of individuals over age or size. For age structure, the density of individuals of each age is transported along characteristics of a first-order partial differential equation, and births enter as a boundary condition at age zero equal to the total reproductive output of the population (the Kermack-McKendrick or McKendrick-von Foerster renewal formulation). When birth and death rates depend on the density of another species, the renewal equation becomes coupled to that species' dynamics. The resulting system can show sustained oscillations driven by the maturation delay as well as equilibrium coexistence.

Structured population models have a general steady-state and stability theory. A population is at equilibrium when its environment, here the density of the other species, makes the expected lifetime reproductive output of a newborn exactly one. The environment in turn must be consistent with the population it supports. Local stability of an equilibrium follows from linearising the coupled system and studying the growth rates of small perturbations, which for age-structured populations solve a characteristic equation rather than a finite matrix eigenvalue problem.

Alternative stable states are central to community ecology and management. When two stable states coexist for the same parameters, the history of the community decides which one is reached, and a disturbance can switch the system between them. Detecting this bistability requires more than direct simulation from one starting point. The full set of equilibria and the stability of each must be determined, together with the parameter values at which they appear, disappear or change stability.

## Problem

In many aquatic and terrestrial communities the roles of predator and prey switch with body size: adults of one species eat the other species, while the other species eats their juveniles. A recent age-structured model of such role reversal couples a logistic prey equation to a Kermack-McKendrick renewal equation for the predator age density. The predator birth and death rates depend on prey density, and smoothed indicator functions mark maturation. The authors analysed its long-term behaviour by simulating a first-order discretisation from a fixed initial condition and mapping attractor types over the maturation age $\tau^*$ and the rate $g$ at which prey consume juvenile predators. The model takes the prey size, the predator age density and 15 parameters as inputs and returns the dynamics of prey, juvenile predators and adult predators.

The prey size $x(t)$ and predator age density $u(t,\tau)$ on $0 \le \tau \le L$ obey $\frac{dx}{dt} = x\,(r - a x + s y_1 - b y_2)$ with $y_1 = \int_0^{\tau^*} u\,d\tau$ and $y_2 = \int_{\tau^*}^{L} u\,d\tau$, together with $u_t + u_\tau = -\mu(x,\tau)\,u$ and $u(t,0) = \int_0^L B(x,\tau)\,u(t,\tau)\,d\tau$. Individuals reaching age $L$ leave the population. The rates are $B(x,\tau) = k x\,\varphi_{\ge}(\tau) + \tilde{B}(\tau)\,(1 - e^{-\zeta x})$ and $\mu(x,\tau) = g x\,\varphi_{<}(\tau) + \mu_B(\tau) + \mu_M e^{-\rho x}$, where $\varphi_{\ge}(\tau) = 1/(1 + e^{-\nu(\tau - \tau^*)})$, $\varphi_{<}(\tau) = 1/(1 + e^{-\nu(\tau^* - \tau)})$, $\tilde{B}(\tau) = b_p\,(e^{-b_{ep}(\tau - \tau^*)} + 1)$ for $\tau \ge \tau^*$ and $0$ otherwise, and $\mu_B(\tau) = d_p\,e^{d_{ep}(\tau - L)}$.

A single simulation reports only one attractor, so it cannot show whether the long-term outcome depends on the initial populations. Answering that requires the steady states of this continuous model and their linear stability, both for the prey-only state $x = r/a$ without predators and for every steady state in which prey and predators coexist. Take $\tau^* = 2$, $L = 30$, $\nu = 100$, $r = 0.4$, $a = 0.1$, $k = 0.3$, $b = 0.8$, $s = 0.2$, $\zeta = 10$, $\mu_M = 1$, $\rho = 5$, $d_p = 0.4$, $b_p = 0.05$, $b_{ep} = 0.1$ and $d_{ep} = 0.1$. A steady state is linearly stable when every root of the characteristic equation of the model linearised about it has negative real part.

Consider all values of $g$ in $[0, 1]$. Find the set of $g$ for which the prey-only steady state is linearly stable and, at the same g, at least one coexistence steady state is also linearly stable. In your reasoning, state the conditions you used and the values of g at which each of these two stability conditions begins or ends. Your final answer must be a single number: the total length of that set of $g$ values.

Output Format Requirements:
Emit <final_answer> immediately, then <reasoning>. Do not write a long derivation before the tags.
You must emit exactly one finite decimal inside <final_answer>...</final_answer>, even if the value is approximate or you are unsure.
Rules:
- The tags are required. Do not omit them or leave them empty.
- The tagged value must be a finite decimal (examples: 0.4847, 0.22, 1.05). Not NaN, not Inf, not a fraction string, not a vector, not prose.
- Put only that one number between the tags. No units, no words, no extra lines.
Keep <reasoning> short (a few hundred words). State the steady-state and stability conditions you used and the few scalars that determine the final number.
Do not paste the input matrices, full coefficient vectors, per-iteration paths, or per-fold candidate tables.

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

01_net_reproductive_number

Goal
----
Compute the expected lifetime offspring of one newborn predator when the prey population is held fixed, in an age-structured predator-prey model with role reversal.

```python
def net_reproductive_number(x: float, tau_star: float, g: float, params: dict) -> float:
    '''Expected number of offspring produced over its lifetime by one newborn predator at a fixed prey size.

    Parameters
    ----------
    x : float
        Prey population size, held constant, x >= 0. Values above the prey carrying
        capacity r / a are admissible.
    tau_star : float
        Maturation age, 0 < tau_star < L.
    g : float
        Consumption rate of juvenile predators by prey, g >= 0.
    params : dict
        Model constants with keys r, a, k, b, s, zeta, mu_M, rho, d_p, b_p, b_ep, d_ep, L, nu.

    Returns
    -------
    R : float
        Expected lifetime offspring of a newborn predator in the model with prey size fixed at x,
        accurate to a relative error of 1e-10.

    Raises
    ------
    ValueError
        If x, tau_star or g is not a finite number, if x < 0 or g < 0, if params is not a dict
        containing finite numeric values for every required key, if L or nu in params is not
        finite and positive, or if tau_star does not satisfy 0 < tau_star < L.
    '''
    return R
```

### Step 2

02_coexistence_steady_states

Goal
----
Find every steady state of the age-structured role-reversal model in which prey and predators coexist.

```python
def coexistence_steady_states(tau_star: float, g: float, params: dict) -> "np.ndarray":
    '''All coexistence steady states of the model defined in net_reproductive_number.

    Parameters
    ----------
    tau_star : float
        Maturation age, 0 < tau_star < L.
    g : float
        Consumption rate of juvenile predators by prey, g >= 0.
    params : dict
        Model constants (keys as in net_reproductive_number).

    Returns
    -------
    states : np.ndarray
        Float array of shape (m, 4), one row [x*, u*(0), y1*, y2*] per coexistence steady state
        with x* > 0 and u*(0) > 0 (prey sizes above r / a included),
        sorted by increasing x*, where u*(0) is the steady newborn density and y1*, y2* are the
        steady juvenile and adult predator numbers; m may be 0. Entries are accurate to a relative
        error of 1e-8.

    Raises
    ------
    ValueError
        If tau_star or g is not a finite number, if g < 0, if params is not a dict containing
        finite numeric values for every required key, if L or nu in params is not finite and
        positive, or if tau_star does not satisfy 0 < tau_star < L.
    '''
    return states
```

### Step 3

03_rightmost_characteristic_root

Goal
----
Determine the leading growth rate of small perturbations around a coexistence steady state of the age-structured role-reversal model.

```python
def rightmost_characteristic_root(state: "np.ndarray", tau_star: float, g: float, params: dict) -> complex:
    '''Characteristic root with the largest real part at a coexistence steady state.

    Parameters
    ----------
    state : np.ndarray
        One row [x*, u*(0), y1*, y2*] as returned by coexistence_steady_states for the same
        tau_star, g and params.
    tau_star : float
        Maturation age, 0 < tau_star < L.
    g : float
        Consumption rate of juvenile predators by prey, g >= 0.
    params : dict
        Model constants (keys as in net_reproductive_number).

    Returns
    -------
    root : complex
        The root with the largest real part of the characteristic equation of the model defined
        in net_reproductive_number, linearised about the given steady state. If it belongs to a
        complex-conjugate pair, the member with positive imaginary part. Accurate to an absolute
        error of 1e-7.

    Raises
    ------
    ValueError
        If tau_star or g is not a finite number, if g < 0, if params is not a dict containing
        finite numeric values for every required key, if L or nu in params is not finite and
        positive, if tau_star does not satisfy 0 < tau_star < L, or if state is not a
        one-dimensional array of 4 finite numbers with the first two entries positive.
    '''
    return root
```

### Step 4

04_invasion_threshold

Goal
----
Find the juvenile-consumption rate above which predators can no longer invade a prey-only community.

```python
def invasion_threshold(tau_star: float, params: dict) -> float:
    '''Juvenile-consumption rate at which the prey-only steady state changes linear stability.

    Parameters
    ----------
    tau_star : float
        Maturation age, with 1 <= tau_star <= 2.
    params : dict
        Model constants (keys as in net_reproductive_number).

    Returns
    -------
    g_inv : float
        The value of g in [0, 1] such that the prey-only steady state x = r / a of the model
        defined in net_reproductive_number is linearly unstable for g < g_inv and linearly
        stable for g > g_inv (exactly one such value exists for these maturation ages).
        Accurate to an absolute error of 1e-8.

    Raises
    ------
    ValueError
        If tau_star is not a finite number satisfying 1 <= tau_star <= 2, if params is not a
        dict containing finite numeric values for every required key, or if L or nu in params
        is not finite and positive.
    '''
    return g_inv
```

### Step 5

05_fold_threshold

Goal
----
Find the juvenile-consumption rate beyond which prey and predators can no longer coexist in any steady state.

```python
def fold_threshold(tau_star: float, params: dict) -> float:
    """Smallest juvenile-consumption rate above which no coexistence steady state exists.

    Parameters
    ----------
    tau_star : float
        Maturation age, with 1 <= tau_star <= 2.
    params : dict
        Model constants, with the same required keys as
        ``net_reproductive_number``.

    Returns
    -------
    float
        The smallest ``g`` in [0, 1] such that the model has no coexistence
        steady state for any higher juvenile-consumption rate. Return 1.0 if
        coexistence still exists at g = 1. Accurate to an absolute error of
        1e-8.

    Raises
    ------
    ValueError
        If ``tau_star`` is not finite or is outside [1, 2], if ``params`` is
        not a dictionary containing every required finite numeric parameter,
        or if ``L`` or ``nu`` is not positive and finite.
    """
    return g_fold
```

### Step 6

06_stability_switch_threshold

Goal
----
Find the juvenile-consumption rate at which the low-prey coexistence steady state switches from oscillatory instability to stability.

```python
def stability_switch_threshold(tau_star: float, params: dict) -> float:
    '''Juvenile-consumption rate at which the lowest-prey coexistence steady state changes linear stability.

    Parameters
    ----------
    tau_star : float
        Maturation age, with 1 <= tau_star <= 2.
    params : dict
        Model constants (keys as in net_reproductive_number).

    Returns
    -------
    g_switch : float
        For these maturation ages, the coexistence steady state with the smallest prey size of
        the model defined in net_reproductive_number exists for every g in [0, g_fold), with g_fold
        as returned by fold_threshold, and changes linear stability at exactly one value of g in
        that interval: unstable below it and stable above it. Return that value, accurate to an
        absolute error of 1e-7.

    Raises
    ------
    ValueError
        If tau_star is not a finite number satisfying 1 <= tau_star <= 2, if params is not a
        dict containing finite numeric values for every required key, or if L or nu in params
        is not finite and positive.
    '''
    return g_switch
```

### Step 7

07_bistability_width

Goal
----
Measure the range of juvenile-consumption rates over which the role-reversal system has two alternative stable outcomes.

```python
def bistability_width(tau_star: float, params: dict) -> float:
    '''Length of the set of juvenile-consumption rates with two locally stable steady states.

    Parameters
    ----------
    tau_star : float
        Maturation age, with 1 <= tau_star <= 2.
    params : dict
        Model constants (keys as in net_reproductive_number).

    Returns
    -------
    width : float
        Total length of the set of g in [0, 1] for which, in the model defined in
        net_reproductive_number, the prey-only steady state is linearly stable and at least one
        coexistence steady state is linearly stable. Accurate to an absolute error of 1e-7.

    Raises
    ------
    ValueError
        If tau_star is not a finite number satisfying 1 <= tau_star <= 2, if params is not a
        dict containing finite numeric values for every required key, or if L or nu in params
        is not finite and positive.
    '''
    return width
```
