# Physics-Particle_Physics-39

## Background

Energetic partons propagating through a quark-gluon plasma undergo repeated interactions with the surrounding thermal medium. At high energies, medium-induced radiation can be treated mainly as a sequence of parton splittings, but this approximation becomes incomplete as the parton energy approaches the thermal scale. Finite-temperature quantum statistics and inverse merging processes then become important because they allow the shower to satisfy detailed balance and approach thermal equilibrium. A finite-temperature parton shower can represent this evolution through competing inelastic channels together with a Sudakov factor that gives the probability for a parton to propagate without an interaction.

## Problem

Consider a gluon with initial momentum $p_0=3.000\ \mathrm{GeV}$ propagating for $L=2.000\ \mathrm{GeV}^{-1}$ through a static gluonic medium. Work in natural units with $\alpha_s=0.300$, $C_A=3$, and $\hat q=0.0150\ \mathrm{GeV}^3$. The plasma temperature is not given directly and must first be inferred from finite-temperature splitting and merging rates.

Approximate every $z$ integral over $0.2\le z\le0.8$ using the following mapped five-point Gauss-Legendre rule:

| $j$ | $z_j$ | $w_j$ |
|---|---:|---:|
| 1 | 0.228146046218401 | 0.071078065516857 |
| 2 | 0.338459206968295 | 0.143588601149810 |
| 3 | 0.500000000000000 | 0.170666666666667 |
| 4 | 0.661540793031705 | 0.143588601149810 |
| 5 | 0.771853953781599 | 0.071078065516857 |

Let $\lambda_{\rm split}(T)$ and $\lambda_{\rm merge}(T)$ denote the quadrature-integrated finite-temperature splitting and merging hazards of the parent gluon. A calibration measurement gives

$$
B_{\rm obs}=0.05369259190766756\ \mathrm{GeV},
$$

where

$$
B(T)=\lambda_{\rm split}(T)-\lambda_{\rm merge}(T).
$$

Search only within $0.2\le T\le0.8\ \mathrm{GeV}$. In this interval, $B(T)=B_{\rm obs}$ has exactly two distinct temperature solutions. An independent no-interaction calibration over $L_{\rm cal}=1.000\ \mathrm{GeV}^{-1}$ gives

$$
S_{\rm obs}=0.9286581077917495,
$$

where the corresponding Sudakov survival is

$$
S(T)=\exp[-(\lambda_{\rm split}(T)+\lambda_{\rm merge}(T))L_{\rm cal}].
$$

Recover both temperature solutions and select the physical branch whose Sudakov survival matches $S_{\rm obs}$. Using that selected temperature, require the designated interaction to be the splitting contribution from the second quadrature node, $j=2$, with $z_\star=0.338459206968295$ and $w_\star=0.143588601149810$. The parent must undergo no earlier inelastic interaction, this designated split must occur at some $0<t<L$, and the two daughters with momenta $z_\star p_0$ and $(1-z_\star)p_0$ must undergo no further splitting or merging before $L$.

Determine the total probability of this exclusive shower history, integrated over the possible splitting time.

Output Format Requirements:

Emit `<final_answer>` immediately, then `<reasoning>`. Do not write a long derivation before the tags.

You must emit exactly one finite decimal inside `<final_answer>...</final_answer>`, even if the value is approximate or you are unsure.

Rules:
- The tags are required. Do not omit them or leave them empty.
- The tagged value must be a finite decimal.
- Put only that one number between the tags. No units, no words, no extra lines.
- Keep `<reasoning>` short.
- Show only the scalars and calculations needed to justify the final number.

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

01_compute_deep_lpm_rate.py

Goal
----
Evaluate the differential deep-LPM gluon splitting rate for a specified daughter momentum fraction and parent momentum.

```python
def compute_deep_lpm_rate(
    z: float,
    momentum: float,
    alpha_s: float,
    c_a: float,
    qhat: float,
) -> float:
    """Evaluate the differential deep-LPM gluon splitting rate.

    Parameters
    ----------
    z : float
        Daughter momentum fraction, with 0 < z < 1.
    momentum : float
        Parent gluon momentum in GeV.
    alpha_s : float
        Strong coupling constant.
    c_a : float
        Gluon color factor.
    qhat : float
        Transverse-momentum broadening coefficient in GeV^3.

    Returns
    -------
    rate : float
        Differential deep-LPM splitting rate.
    """
    return rate
```

### Step 2

02_compute_thermal_channel_rates.py

Goal
----
Evaluate the finite-temperature splitting and inverse-merging rates for a gluon at a specified momentum fraction.

```python
def compute_thermal_channel_rates(
    z: float,
    momentum: float,
    temperature: float,
    alpha_s: float,
    c_a: float,
    qhat: float,
) -> "np.ndarray":
    """Evaluate the finite-temperature splitting and merging rates.

    Parameters
    ----------
    z : float
        Daughter momentum fraction, with 0 < z < 1.
    momentum : float
        Gluon momentum in GeV.
    temperature : float
        Medium temperature in GeV.
    alpha_s : float
        Strong coupling constant.
    c_a : float
        Gluon color factor.
    qhat : float
        Transverse-momentum broadening coefficient in GeV^3.

    Returns
    -------
    rates : np.ndarray
        Array of shape (2,) containing [Gamma_1, Gamma_2] in GeV.
    """
    return rates
```

### Step 3

03_integrate_inelastic_hazard.py

Goal
----
Integrate the finite-temperature splitting and merging channels over the supplied momentum-fraction quadrature to obtain the total inelastic hazard.

```python
def integrate_inelastic_hazard(
    momentum: float,
    temperature: float,
    alpha_s: float,
    c_a: float,
    qhat: float,
    z_nodes: "np.ndarray",
    weights: "np.ndarray",
) -> "np.ndarray":
    """Integrate the finite-temperature inelastic shower rates.

    Parameters
    ----------
    momentum : float
        Gluon momentum in GeV.
    temperature : float
        Medium temperature in GeV.
    alpha_s : float
        Strong coupling constant.
    c_a : float
        Gluon color factor.
    qhat : float
        Transverse-momentum broadening coefficient in GeV^3.
    z_nodes : np.ndarray
        One-dimensional quadrature nodes.
    weights : np.ndarray
        Quadrature weights corresponding to z_nodes.

    Returns
    -------
    hazards : np.ndarray
        Array of shape (3,) containing the integrated splitting hazard,
        merging hazard, and their sum, in GeV.
    """
    return hazards
```

### Step 4

04_compute_designated_split_rate.py

Goal
----
Calculate the integrated rate of a specified discrete splitting subchannel in the quadrature representation of the shower.

```python
def compute_designated_split_rate(
    z_star: float,
    weight_star: float,
    momentum: float,
    temperature: float,
    alpha_s: float,
    c_a: float,
    qhat: float,
) -> float:
    """Evaluate the integrated rate of one designated splitting subchannel.

    Parameters
    ----------
    z_star : float
        Momentum fraction of the designated splitting node.
    weight_star : float
        Quadrature weight of the designated node.
    momentum : float
        Parent gluon momentum in GeV.
    temperature : float
        Medium temperature in GeV.
    alpha_s : float
        Strong coupling constant.
    c_a : float
        Gluon color factor.
    qhat : float
        Transverse-momentum broadening coefficient in GeV^3.

    Returns
    -------
    rate : float
        Integrated rate of the designated splitting subchannel in GeV.
    """
    return rate
```

### Step 5

05_compute_daughter_shower_state.py

Goal
----
Determine the two daughter momenta and their finite-temperature interaction hazards after the designated splitting.

```python
def compute_daughter_shower_state(
    z_star: float,
    parent_momentum: float,
    temperature: float,
    alpha_s: float,
    c_a: float,
    qhat: float,
    z_nodes: "np.ndarray",
    weights: "np.ndarray",
) -> "np.ndarray":
    """Evaluate daughter momenta and their total inelastic hazards.

    Parameters
    ----------
    z_star : float
        Momentum fraction carried by the first daughter.
    parent_momentum : float
        Parent gluon momentum in GeV.
    temperature : float
        Medium temperature in GeV.
    alpha_s : float
        Strong coupling constant.
    c_a : float
        Gluon color factor.
    qhat : float
        Transverse-momentum broadening coefficient in GeV^3.
    z_nodes : np.ndarray
        One-dimensional quadrature nodes.
    weights : np.ndarray
        Quadrature weights corresponding to z_nodes.

    Returns
    -------
    state : np.ndarray
        Array of shape (5,) containing [p1, p2, lambda1, lambda2,
        lambda1 + lambda2], with momenta and hazards in GeV.
    """
    return state
```

### Step 6

06_integrate_exclusive_history_probability.py

Goal
----
Recover both temperature branches consistent with a finite-temperature splitting-minus-merging calibration observable, then use an independent Sudakov survival measurement to select the physical branch.

```python
def infer_temperature_branches(
    momentum: float,
    alpha_s: float,
    c_a: float,
    qhat: float,
    z_nodes: "np.ndarray",
    weights: "np.ndarray",
    balance_observed: float,
    survival_observed: float,
    calibration_length: float,
    temperature_bounds: "np.ndarray",
) -> "np.ndarray":
    """Infer the two calibration temperature branches and select the physical one.

    Parameters
    ----------
    momentum : float
        Parent momentum in GeV.
    alpha_s : float
        Strong coupling.
    c_a : float
        Gluon color factor.
    qhat : float
        Transverse-momentum broadening coefficient in GeV^3.
    z_nodes : np.ndarray
        One-dimensional quadrature nodes.
    weights : np.ndarray
        One-dimensional quadrature weights corresponding to z_nodes.
    balance_observed : float
        Observed splitting-minus-merging hazard in GeV.
    survival_observed : float
        Observed no-interaction Sudakov survival probability.
    calibration_length : float
        Calibration propagation length in GeV^(-1).
    temperature_bounds : np.ndarray
        Two-element array [T_min, T_max] in GeV. For the supported
        inputs, the interval contains one interior maximum of the
        balance observable and exactly two distinct calibration roots.

    Returns
    -------
    result : np.ndarray
        Array of shape (3,) containing the lower temperature root,
        the upper temperature root, and the selected physical
        temperature, all in GeV.
    """
    return result
```

### Step 7

07_integrate_exclusive_history_probability.py

Goal
----
Integrate the exclusive probability for an ordered shower history containing an arbitrary number of designated emissions.

```python
def integrate_exclusive_history_probability(
    segment_hazards: "np.ndarray",
    split_rates: "np.ndarray",
    length: float,
) -> float:
    """Integrate a general ordered exclusive shower-history probability.

    Parameters
    ----------
    segment_hazards : np.ndarray
        One-dimensional array of shape (N+1,) containing the total
        inelastic hazards lambda_0, ..., lambda_N in GeV for the
        propagation segments before, between, and after emissions.
    split_rates : np.ndarray
        One-dimensional array of shape (N,) containing the designated
        emission rates r_1, ..., r_N in GeV.
    length : float
        Total propagation length in GeV^(-1).

    Returns
    -------
    probability : float
        Dimensionless exclusive-history probability.
    """
    return probability
```

### Step 8

08_solve_calibrated_thermal_shower_case.py

Goal
----
Infer the physical plasma temperature from the two-branch thermal calibration and use that temperature to evaluate the complete exclusive shower-history probability.

```python
def solve_calibrated_thermal_shower_case(
    p0: float,
    length: float,
    alpha_s: float,
    c_a: float,
    qhat: float,
    z_nodes: "np.ndarray",
    weights: "np.ndarray",
    split_index: int,
    balance_observed: float,
    survival_observed: float,
    calibration_length: float,
    temperature_bounds: "np.ndarray",
) -> float:
    """Solve the calibrated finite-temperature exclusive shower case.

    Parameters
    ----------
    p0 : float
        Initial parent momentum in GeV.
    length : float
        Shower propagation length in GeV^(-1).
    alpha_s : float
        Strong coupling.
    c_a : float
        Gluon color factor.
    qhat : float
        Transverse-momentum broadening coefficient in GeV^3.
    z_nodes : np.ndarray
        One-dimensional quadrature nodes.
    weights : np.ndarray
        One-dimensional quadrature weights.
    split_index : int
        Zero-based index of the designated quadrature splitting node.
    balance_observed : float
        Observed splitting-minus-merging hazard in GeV.
    survival_observed : float
        Independent calibration Sudakov survival probability.
    calibration_length : float
        Calibration propagation length in GeV^(-1).
    temperature_bounds : np.ndarray
        Two-element calibration temperature interval in GeV.

    Returns
    -------
    probability : float
        Dimensionless exclusive-history probability after selecting the
        physical calibration branch.
    """
    return probability
```
