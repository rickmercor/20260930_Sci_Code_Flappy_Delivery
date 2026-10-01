# Biology-Biochemistry-23

## Background

Double-stranded DNA held between two surfaces by opposite strands and loaded in shear is one of the basic force-bearing elements of single-molecule biophysics and DNA nanotechnology. Under a constant load the duplex eventually dissociates, and the distribution of rupture times, measured in optical tweezers or by atomic force microscopy, is summarised by an off-rate k(F). Bell's phenomenological relation, ln k(F) = ln k(0) + F d / k_B T, converts the force dependence of the off-rate into a single length d, the apparent transition-state distance, which is routinely reported and compared across constructs. Its molecular meaning, however, depends on the pathway by which the base pairs actually come apart.

For duplexes of ten or so base pairs the dominant pathway is fraying: single base pairs open and close at the two ends of the helix, and complete rupture occurs when the last pair breaks. The energetics of each step are those of nearest-neighbour thermodynamics, in which a base-pair stack contributes a stacking enthalpy and entropy, while forming the first pair and the ends of the helix carry separate initiation and terminal-penalty terms; these parameters are tabulated at standard salt and are corrected to the ionic conditions of an experiment. With rates for every elementary transition, the kinetics is a continuous-time Markov chain on the set of partially frayed states with one absorbing state, and its master equation is a finite linear system that can be integrated exactly.

Force enters through polymer mechanics. Opening a base pair releases two nucleotides of single-stranded DNA and shortens the intact duplex by one step; in the shear geometry only the released nucleotide on the pulled strand lies in the force path while its partner dangles unloaded, and under load each load-bearing part extends according to its own force-extension law, the flexible single-stranded segments as a worm-like chain and the short, stiff duplex as a rigid rod whose orientation fluctuates thermally. The resulting change in construct length, integrated over force, is the mechanical work that shifts the equilibrium constant of the step, and a transition-state position between the open and closed configurations decides how much of that work accelerates the forward rate and how much decelerates the reverse one. Because the duplexes involved are only a few nanometres long, the geometry of the double helix itself, not just its contour length, determines the distance over which the shear force acts.

The output of such a model is a time course of the probability of complete rupture, which is fitted by a single exponential in the same way as experimental survival curves, and the resulting off-rates at several forces are fitted by Bell's law. Comparing the apparent transition-state distance obtained this way with measured values tests whether the polymer-mechanical and thermodynamic ingredients of the model are the right ones, and the same framework predicts how the off-rate saturates at high force and how the distance depends on temperature.

## Problem

Short DNA duplexes are used as force-bearing links in single-molecule assays and in DNA nanodevices, and their lifetime under a shear force is the quantity that experiments measure through rupture-time distributions and summarise by an apparent transition-state distance, the length that multiplies force in Bell's law ln k(F) = ln k(0) + F d / k_B T. A recent kinetic approach describes the shear rupture of a short duplex as a master equation over single-base-pair events: base pairs open and close one at a time from either end of the duplex (no bulges, internal loops or sliding), every transition has an equilibrium constant set by nearest-neighbour thermodynamics plus the mechanical work of the change in construct length that the transition produces under force, the last base pair opens at a rate set by the non-stacking (initiation and terminal-penalty) free energy, and the force work is apportioned between the opening and closing rates by a specific placement of the transition state along the extension coordinate. The mechanical work follows from polymer models of the two parts of the construct, the intact duplex and the released single strands, with the duplex treated as a rigid rod whose length is its three-dimensional helical end-to-end distance between the two pulled termini. The inputs are the sequence thermodynamics, the polymer parameters, the temperature and the shear forces; the output is the force-dependent off-rate and, from it, the apparent transition-state distance.

Consider a single 9-base-pair polyA/polyT duplex sheared at T = 303.15 K by a constant force applied to the 5' termini of the two strands, i.e. at opposite ends of the duplex on the two different backbones. Use the nearest-neighbour stacking enthalpy dH_s = -7.6 kcal/mol per AA/TT step with the salt-corrected stacking entropy dS_s = -0.0920 kJ/(mol K), and the combined non-stacking (initiation plus two terminal AT penalties) enthalpy dH_non = +4.6 kcal/mol with salt-corrected entropy dS_non = +0.0502 kJ/(mol K); free energies are dG = dH - T dS, with 1 cal = 4.184 J, k_B = 1.380649e-23 J/K and N_A = 6.02214076e23 /mol. In the shear geometry only the released nucleotides of the pulled strand at each frayed end lie in the force path, so with n base pairs closed the load-bearing single-stranded DNA consists of the N - n released nucleotides of the pulled strands (the complementary released nucleotides dangle unloaded); model that load-bearing single-stranded DNA as one worm-like chain of N - n nucleotides through the Marko-Siggia interpolation formula with persistence length 0.77 nm and contour length 0.70 nm per nucleotide, and the duplex as a rigid rod with the rotational (Langevin) force-extension response, its rod length being the B-DNA helical end-to-end distance (2 nm diameter, 0.34 nm rise per base pair, 10.5 base pairs per turn) constructed as the approach prescribes. Rates are measured in units of the attempt rate k_a, which is common to all closing transitions at zero force.

Build the transition rates and the master-equation generator at each of the shear forces F = 3, 6 and 9 pN, propagate the probability from the fully closed duplex, and obtain the off-rate k(F) by fitting the probability of the fully ruptured state to 1 - exp(-k t) by unweighted nonlinear least squares on 1000 equally spaced times from 0 to the thermal dissociation time t_e = exp(-dG_duplex / RT) / k_a, where dG_duplex = dG_non + 8 dG_s is the hybridisation free energy of the full duplex at the working temperature. Then obtain the apparent transition-state distance as k_B T times the slope of the least-squares straight line through ln k(F) versus F at the three forces. In your reasoning, report the helical end-to-end length of the intact 9-base-pair duplex, the three off-rates in units of k_a, the transition-state distance you obtain when the helical rod length is replaced by the axial contour length between the two pulled termini, 0.34 (n - 1) nm for a duplex of n closed base pairs, and state in one sentence what the source finds about that replacement.

Report the apparent transition-state distance d in nanometres as a single number.

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

helical_end_to_end_distance

Goal
----
Return the end-to-end distance in nm of an n_bp base-pair B-DNA duplex measured between its two force-bearing termini under shear, i.e. the 5' terminus of one strand and the 5' terminus of the other strand, which lie at opposite ends of the duplex on the two different backbones. Use the source's explicit three-dimensional parametrisation of the two backbone helices with the standard B-DNA parameters 2 nm diameter, 0.34 nm rise per base pair and 10.5 base pairs per turn, and the source's convention for where on the two helices the two termini sit; do not return the contour length along the helix axis.

```python
def helical_end_to_end_distance(n_bp: int) -> float:
    """Return the end-to-end distance in nm of an n_bp base-pair B-DNA duplex measured between its two force-bearing termini under shear, i.e. the 5' terminus of one strand and the 5' terminus of the other strand, which lie at opposite ends of the duplex on the two different backbones. Use the source's explicit three-dimensional parametrisation of the two backbone helices with the standard B-DNA parameters 2 nm diameter, 0.34 nm rise per base pair and 10.5 base pairs per turn, and the source's convention for where on the two helices the two termini sit; do not return the contour length along the helix axis.

    Parameters
    ----------
    n_bp : int
        Number of closed base pairs in the duplex (>= 1).

    Returns
    -------
    distance : float
        End-to-end distance in nm.

    Raises
    ------
    ValueError
        If n_bp is not an integer >= 1.
    """
    return distance
```

### Step 2

rod_extension

Goal
----
Return the mean extension in nm along the force of a rigid rod of the given length (nm) under a constant force f (pN) at thermal energy k_B T (pN nm): the rotational (Langevin) average L [coth(f L / k_B T) - k_B T / (f L)]; a rod of zero length has zero extension.

```python
def rod_extension(length: float, force: float, k_bt: float) -> float:
    """Return the mean extension in nm along the force of a rigid rod of the given length (nm) under a constant force f (pN) at thermal energy k_B T (pN nm): the rotational (Langevin) average L [coth(f L / k_B T) - k_B T / (f L)]; a rod of zero length has zero extension.

    Parameters
    ----------
    length : float
        Rod length in nm (>= 0).
    force : float
        Force in pN (> 0).
    k_bt : float
        Thermal energy in pN nm (> 0).

    Returns
    -------
    extension : float
        Extension in nm.

    Raises
    ------
    ValueError
        If length is negative or not finite, or force or k_bt is not finite and positive.
    """
    return extension
```

### Step 3

ssdna_extension

Goal
----
Return the mean extension in nm of a single strand of n_nt nucleotides under force f (pN) at thermal energy k_B T (pN nm), modelled as a worm-like chain of contour length l_ss n_nt with persistence length lambda_ss through the Marko-Siggia interpolation formula f lambda_ss / k_B T = 1 / (4 (1 - u)^2) - 1/4 + u, u = <z>/L, solved for u on (0, 1); zero nucleotides give zero extension.

```python
def ssdna_extension(n_nt: int, force: float, k_bt: float, persistence_length: float,
                            interphosphate_distance: float) -> float:
    """Return the mean extension in nm of a single strand of n_nt nucleotides under force f (pN) at thermal energy k_B T (pN nm), modelled as a worm-like chain of contour length l_ss n_nt with persistence length lambda_ss through the Marko-Siggia interpolation formula f lambda_ss / k_B T = 1 / (4 (1 - u)^2) - 1/4 + u, u = <z>/L, solved for u on (0, 1); zero nucleotides give zero extension.

    Parameters
    ----------
    n_nt : int
        Number of nucleotides in the released strand (>= 0).
    force : float
        Force in pN (> 0).
    k_bt : float
        Thermal energy in pN nm (> 0).
    persistence_length : float
        lambda_ss in nm (> 0).
    interphosphate_distance : float
        l_ss in nm (> 0).

    Returns
    -------
    extension : float
        Extension in nm.

    Raises
    ------
    ValueError
        If n_nt is not a nonnegative integer or any other argument is not finite and positive.
    """
    return extension
```

### Step 4

mechanical_work

Goal
----
Return the mechanical work W_n(F) in pN nm associated with the formation of the n-th base pair (the n_closed - 1 -> n_closed transition) of an n_bp duplex under shear at temperature T (K): the integral from 0 to F of the change in construct length dx_n(f), where dx_n(f) is the extension of the n_closed-pair duplex rod (helical length, previous steps) minus that of the (n_closed - 1)-pair rod (a zero-pair duplex has zero length) plus the extension of the released strand with n_bp - n_closed nucleotides minus that with n_bp - n_closed + 1 nucleotides. Use k_B = 1.380649e-23 J/K.

```python
def mechanical_work(n_closed: int, force: float, n_bp: int, temperature: float, persistence_length: float,
                            interphosphate_distance: float) -> float:
    """Return the mechanical work W_n(F) in pN nm associated with the formation of the n-th base pair (the n_closed - 1 -> n_closed transition) of an n_bp duplex under shear at temperature T (K): the integral from 0 to F of the change in construct length dx_n(f), where dx_n(f) is the extension of the n_closed-pair duplex rod (helical length, previous steps) minus that of the (n_closed - 1)-pair rod (a zero-pair duplex has zero length) plus the extension of the released strand with n_bp - n_closed nucleotides minus that with n_bp - n_closed + 1 nucleotides. Use k_B = 1.380649e-23 J/K.

    Parameters
    ----------
    n_closed : int
        Number of closed pairs after the transition (2 <= n_closed <= n_bp).
    force : float
        Shear force F in pN (> 0).
    n_bp : int
        Total number of base pairs of the duplex (>= 2).
    temperature : float
        Temperature in K (> 0).
    persistence_length : float
        lambda_ss in nm.
    interphosphate_distance : float
        l_ss in nm.

    Returns
    -------
    work : float
        Work in pN nm.

    Raises
    ------
    ValueError
        If n_closed or n_bp are out of range, or force, temperature or the polymer parameters are not finite and positive.
    """
    return work
```

### Step 5

transition_rates

Goal
----
Return the single-base transition rates of the force-dependent nucleation-zipper model in units of the attempt rate k_a, as the array [k_ot, k_o(2), ..., k_o(n_bp), k_c(2), ..., k_c(n_bp)]: k_ot is the terminal opening rate of the last base pair, k_a exp(dG_non / RT), set by the non-stacking free energy dG_non = dH_non - T dS_non (initiation plus terminal penalties); k_o(n) is the opening rate of the n-th base pair (n -> n - 1 closed pairs) and k_c(n) the closing rate of the (n - 1) -> n transition. At zero force every closing rate equals k_a and every internal opening rate equals k_a exp(dG_s / RT) with the stacking free energy dG_s = dH_s - T dS_s. Under force the ratio k_c(n) / k_o(n) must equal the force-dependent equilibrium constant exp(-dG_s / RT) exp(W_n(F) / k_B T), with W_n the mechanical work of the previous step; how the factor exp(W_n / k_B T) is apportioned between k_c(n) and k_o(n) is fixed by the source's placement of the transition state along the extension coordinate, and the source's placement must be used. Enthalpies in kJ/mol, entropies in kJ/(mol K), R = k_B N_A with N_A = 6.02214076e23.

```python
def transition_rates(force: float, n_bp: int, temperature: float, dh_stack: float, ds_stack: float,
                             dh_non: float, ds_non: float, persistence_length: float,
                             interphosphate_distance: float) -> "np.ndarray":
    """Return the single-base transition rates of the force-dependent nucleation-zipper model in units of the attempt rate k_a, as the array [k_ot, k_o(2), ..., k_o(n_bp), k_c(2), ..., k_c(n_bp)]: k_ot is the terminal opening rate of the last base pair, k_a exp(dG_non / RT), set by the non-stacking free energy dG_non = dH_non - T dS_non (initiation plus terminal penalties); k_o(n) is the opening rate of the n-th base pair (n -> n - 1 closed pairs) and k_c(n) the closing rate of the (n - 1) -> n transition. At zero force every closing rate equals k_a and every internal opening rate equals k_a exp(dG_s / RT) with the stacking free energy dG_s = dH_s - T dS_s. Under force the ratio k_c(n) / k_o(n) must equal the force-dependent equilibrium constant exp(-dG_s / RT) exp(W_n(F) / k_B T), with W_n the mechanical work of the previous step; how the factor exp(W_n / k_B T) is apportioned between k_c(n) and k_o(n) is fixed by the source's placement of the transition state along the extension coordinate, and the source's placement must be used. Enthalpies in kJ/mol, entropies in kJ/(mol K), R = k_B N_A with N_A = 6.02214076e23.

    Parameters
    ----------
    force : float
        Shear force in pN (> 0).
    n_bp : int
        Number of base pairs (>= 2).
    temperature : float
        Temperature in K (> 0).
    dh_stack : float
        Stacking enthalpy per base-pair step, kJ/mol.
    ds_stack : float
        Salt-corrected stacking entropy per step, kJ/(mol K).
    dh_non : float
        Non-stacking enthalpy (initiation plus terminal penalties), kJ/mol.
    ds_non : float
        Salt-corrected non-stacking entropy, kJ/(mol K).
    persistence_length : float
        lambda_ss in nm.
    interphosphate_distance : float
        l_ss in nm.

    Returns
    -------
    rates : np.ndarray
        Array [k_ot, k_o(2), ..., k_o(n_bp), k_c(2), ..., k_c(n_bp)].

    Raises
    ------
    ValueError
        If force, temperature or the polymer parameters are not finite and positive, n_bp < 2, or a thermodynamic parameter is not finite.
    """
    return rates
```

### Step 6

master_equation_generator

Goal
----
Assemble the generator matrix T of the master equation dP/dt = T P for the nucleation-zipper rupture of an n_bp duplex from the rate array [k_ot, k_o(2), ..., k_o(n_bp), k_c(2), ..., k_c(n_bp)] of the previous step. States are the pairs (i, j) with i ruptured pairs counted from the left end and j from the right end, i + j <= n_bp - 1, ordered lexicographically by (i, j), followed by the fully ruptured single-strand state as the last index; n = n_bp - i - j closed pairs remain. From a state with n > 1 each end opens one pair at rate k_o(n); with n = 1 the last pair opens into the ruptured state at rate k_ot; an end with i > 0 (or j > 0) re-closes one pair at rate k_c(n + 1); the ruptured state is absorbing; bulges, internal loops and sliding are neglected. Column k of T holds the outflow of state k on the diagonal and the inflows to the other states off the diagonal.

```python
def master_equation_generator(n_bp: int, rates: "np.ndarray") -> "np.ndarray":
    """Assemble the generator matrix T of the master equation dP/dt = T P for the nucleation-zipper rupture of an n_bp duplex from the rate array [k_ot, k_o(2), ..., k_o(n_bp), k_c(2), ..., k_c(n_bp)] of the previous step. States are the pairs (i, j) with i ruptured pairs counted from the left end and j from the right end, i + j <= n_bp - 1, ordered lexicographically by (i, j), followed by the fully ruptured single-strand state as the last index; n = n_bp - i - j closed pairs remain. From a state with n > 1 each end opens one pair at rate k_o(n); with n = 1 the last pair opens into the ruptured state at rate k_ot; an end with i > 0 (or j > 0) re-closes one pair at rate k_c(n + 1); the ruptured state is absorbing; bulges, internal loops and sliding are neglected. Column k of T holds the outflow of state k on the diagonal and the inflows to the other states off the diagonal.

    Parameters
    ----------
    n_bp : int
        Number of base pairs (>= 2).
    rates : np.ndarray
        Array [k_ot, k_o(2), ..., k_o(n_bp), k_c(2), ..., k_c(n_bp)] of length 2 n_bp - 1.

    Returns
    -------
    generator : np.ndarray
        Generator matrix T of shape (N_states, N_states).

    Raises
    ------
    ValueError
        If n_bp < 2 or rates is not a finite nonnegative array of length 2 n_bp - 1.
    """
    return generator
```

### Step 7

rupture_off_rate

Goal
----
Return the rupture off-rate k (units of k_a) of a duplex whose master-equation generator T is given: propagate the probability vector from the fully closed duplex (index 0) as P(t) = exp(T t) P(0), sample the probability of the absorbing ruptured state (last index) at n_points equally spaced times from 0 to t_end inclusive, and fit the single-exponential law P_ss(t) = 1 - exp(-k t) by unweighted nonlinear least squares in k (initial guess 1 / t_end), iterating the fit to convergence with relative tolerances of 1e-14 on the parameter, the residual and the gradient (library defaults of about 1e-8 leave the last digits unconverged).

```python
def rupture_off_rate(generator: "np.ndarray", t_end: float, n_points: int) -> float:
    """Return the rupture off-rate k (units of k_a) of a duplex whose master-equation generator T is given: propagate the probability vector from the fully closed duplex (index 0) as P(t) = exp(T t) P(0), sample the probability of the absorbing ruptured state (last index) at n_points equally spaced times from 0 to t_end inclusive, and fit the single-exponential law P_ss(t) = 1 - exp(-k t) by unweighted nonlinear least squares in k (initial guess 1 / t_end), iterating the fit to convergence with relative tolerances of 1e-14 on the parameter, the residual and the gradient (library defaults of about 1e-8 leave the last digits unconverged).

    Parameters
    ----------
    generator : np.ndarray
        Square generator matrix T of the master equation.
    t_end : float
        End of the sampling window in units of 1 / k_a (> 0).
    n_points : int
        Number of equally spaced sample times including 0 and t_end (>= 3).

    Returns
    -------
    k_off : float
        Fitted off-rate in units of k_a.

    Raises
    ------
    ValueError
        If generator is not a finite square matrix of size >= 2, t_end is not positive, or n_points < 3.
    """
    return k_off
```

### Step 8

apparent_transition_state_distance

Goal
----
Orchestrate the pipeline: for each shear force build the transition rates and the master-equation generator, obtain the off-rate k(F) by the single-exponential fit over the thermal dissociation window t_end = exp(-(dG_non + (n_bp - 1) dG_s) / RT) in units of 1 / k_a with n_points samples, and return the apparent transition-state distance d = k_B T times the slope of the least-squares straight line of ln k(F) against F (Bell's relation), in nm. Call the earlier step functions rather than reimplementing them.

```python
def apparent_transition_state_distance(forces: "np.ndarray", n_bp: int, temperature: float, dh_stack: float,
                                               ds_stack: float, dh_non: float, ds_non: float, persistence_length: float,
                                               interphosphate_distance: float, n_points: int) -> float:
    """Orchestrate the pipeline: for each shear force build the transition rates and the master-equation generator, obtain the off-rate k(F) by the single-exponential fit over the thermal dissociation window t_end = exp(-(dG_non + (n_bp - 1) dG_s) / RT) in units of 1 / k_a with n_points samples, and return the apparent transition-state distance d = k_B T times the slope of the least-squares straight line of ln k(F) against F (Bell's relation), in nm. Call the earlier step functions rather than reimplementing them.

    Parameters
    ----------
    forces : np.ndarray
        Increasing 1-D array of shear forces in pN (>= 2 values).
    n_bp : int
        Number of base pairs (>= 2).
    temperature : float
        Temperature in K.
    dh_stack : float
        Stacking enthalpy, kJ/mol.
    ds_stack : float
        Salt-corrected stacking entropy, kJ/(mol K).
    dh_non : float
        Non-stacking enthalpy, kJ/mol.
    ds_non : float
        Salt-corrected non-stacking entropy, kJ/(mol K).
    persistence_length : float
        lambda_ss in nm.
    interphosphate_distance : float
        l_ss in nm.
    n_points : int
        Number of sample times of the exponential fit (>= 3).

    Returns
    -------
    d : float
        Apparent transition-state distance in nm.

    Raises
    ------
    ValueError
        If forces is not an increasing array of at least two positive values, n_bp < 2, temperature is not positive, or n_points < 3.
    """
    return d
```
