# Biology-Biochemistry-1

## Background

Proteins and DNA are biological compounds that are dynamic, crowded, and reactive. Their covalent bonds are mostly stable under ambient conditions, but tissue is constantly deformed, pulled, and stretched, and those forces funnel into individual chemical bonds, where they can tip a reaction from negligible to fast. Two competing fates of a stressed bond matter most: homolytic scission, which makes two radicals that then migrate and cause collateral damage, and heterolytic hydrolysis, which is a closed-shell reaction with water. Molecular dynamics cannot reach the timescale on which these reactions occur, so a recent study built an emulator that marries molecular dynamics with kinetic Monte Carlo: it samples a conformational ensemble and turns each snapshot geometry and force into a reaction rate through physics-based or experimentally derived models. Afterwards, it advances time with a rejection-free kinetic Monte Carlo step.
This allows the method to answer which reaction wins at each bond and how the balance shifts when a simple molecule becomes a dense network. The task supplied here reproduces the rate models and the kinetic-Monte-Carlo selection on one supplied tense-network system.

## Problem

Under a mechanical load, covalent bonds of a biomolecular network can either rupture homolytically, leaving radicals that migrate and cause further damage or be cleaved heterolytically by hydrolysis. The pathway that wins at each reactive site depends on how the local force couples to the bond dissociation energy. A recent study introduced a hybrid kinetic-Monte-Carlo / molecular-dynamics reaction emulator that predicts, from a conformational ensemble, the rate of each competing reaction at every reactive site, selects a reaction by rejection-free kinetic Monte Carlo, applies the chosen break to the network topology, relaxes the structure, and repeats. Its reaction-rate models reproduce experimental force-clamp data and reinterpret mechano-chemical observations in proteins and DNA.

Your task is to reproduce one deterministic scalar from that method. For a supplied tense-network system of six reactive bonds, compute the probability P that the first event is a hydrolysis and that each of the three events that follow is a homolysis, and report P to four decimal places. Every draw runs the study rejection-free kinetic Monte Carlo over the competing events available at that moment: the first draw over twelve channels (homolysis and hydrolysis at each of the six bonds), and each later draw over the channels of the bonds that remain. Once a bond has broken by any pathway, it is consumed: it no longer reacts, and its load redistributes onto the surviving bonds as described below.

After each break, regardless of pathway, the network relaxes and the surviving bonds forces change. In the supplied system that relaxation is a stated property of the network rather than being recomputed by simulation: the surviving bonds are stretched together, by one common additional extension, and each of them resists along its own bond potential, so the share of the load a bond picks up follows from how it responds where it currently sits. Before the break each survivor sits at the extension where its own restoring force balances the load it currently carries, which is the shifted well minimum of its own V_eff at that force. The common additional extension is whatever leaves the survivors carrying a fraction t = 0.045 of the broken bond force on top of what they carried before, and the forces that result are rounded to four decimal places. After a later break the relaxation acts on the forces the previous relaxation produced, not on the original forces. Surviving bonds keep their listed dissociation energies E_dis and force constants k, only the forces change.

A homolytic break also leaves a radical in the network while a hydrolytic cleavage is closed-shell and produces none. If there is another event afterwards, the radical attacks the surviving bond with the lowest rupture barrier once the network has relaxed, and weakens that bond before the next draw. If two survivors share the lowest barrier, the earlier one in the order of the site table below is attacked. Weakening multiplies that bond dissociation energy by (1 - r) with the network weakening fraction r = 0.060, and the result is rounded to four decimal places. 
Weakening acts on the bond E_dis alone: the force constant k and the forces are unaffected by it. If the same surviving bond receives two radicals over successive breaks, the weakening compounds E_dis is multiplied by (1 - r) again. A weakened E_dis is used in the homolysis rate law exactly as the prior E_dis would be.

The six reactive bonds of the system are listed below with their dissociation energies E_dis (kJ/mol) and effective force constants k (kJ/mol/nm^2). The simulations recorded each bond extension frame by frame in two or three independent replicas, and the per-frame extension readings of every replica are listed in time order. Instrument A recorded extensions in nm and instrument B recorded them in Angstrom. A frame whose force evaluation did not converge is recorded as exactly 0. A bond may be locally compressed in an individual frame: negative extensions are physical and enter the restoring force with their sign. Within a replica the production window is the part of the record after the last non-converged reading, the readings before it are equilibration and are not part of the average. The force of a frame is the restoring force of the Morse potential of that bond V(x) = E_dis (1 - exp(-beta x))^2 at the recorded extension x, with beta = sqrt(k / (2 E_dis)) and x in nm, so that F = 2 E_dis beta exp(-beta x) (1 - exp(-beta x)) in kJ mol^-1 nm^-1, a frame force is converted to pN with 1 kJ mol^-1 nm^-1 = 1.66054 pN. A replica force is the mean of its production-frame forces. A replica that deviates from the median replica force by more than one tenth of that median is contaminated and is discarded, at least one replica must survive. The local pulling force of the bond is the mean of its surviving replica forces, converted to nN and rounded to two decimal places (1 nN = 1000 pN).

* disulfide S-S : E_dis = 251.0, k = 15954.0, instrument A (nm)
    replica 1: 0.005954, 0, 0.025799, 0.02825
    replica 2: 0.005954, 0, 0.02952, 0.030823, 0.031486, 0.031486
    replica 3: 0.005954, 0, 0.012594, 0, 0.050917, 0.055129
* ester C-O : E_dis = 280.0, k = 16082.0, instrument B (Angstrom)
    replica 1: 0.05889, 0, 0.3388
    replica 2: 0.05889, 0, 0.3388, 0.36639
    replica 3: 0.05889, 0, 0.1241, 0, 0.34556, 0.37352, 0.38075
* glycosidic C-O : E_dis = 230.0, k = 27566.0, instrument A (nm)
    replica 1: 0.003409, 0, -0.003159, 0.017358, 0.018031
    replica 2: 0.003409, 0, 0.010065, 0.011192
* peroxide O-O : E_dis = 170.0, k = 73368.0, instrument B (Angstrom)
    replica 1: 0.01266, 0, 0.07328
    replica 2: 0.01266, 0, 0.07328, 0.07559, 0.07792
    replica 3: 0.01266, 0, 0.02608, 0, 0.07559, 0.08029
* peptide C-N : E_dis = 350.0, k = 22016.0, instrument A (nm)
    replica 1: 0.004252, 0, 0.028469
    replica 2: 0.004252, 0, 0.029388, 0.030323, 0.031275
* thioester C-S : E_dis = 260.0, k = 45653.0, instrument B (Angstrom)
    replica 1: 0.02036, 0, 0.1517, 0.16009
    replica 2: 0.02036, 0, 0.15797, 0.16653
    replica 3: 0.02036, 0, 0.04198, 0, 0.16437, 0.1687, 0.1731

The two competing rate laws are stated below as the study released implementation computes them: every constant they need is supplied here except the homolysis frequency prefactor.

* Homolysis (force-accelerated bond scission): k_hom = nu * exp(-dV / (R T)), where dV is the rupture barrier of the force-shifted bond potential V_eff(x) = E_dis (1 - exp(-beta x))^2 - F_c x with beta = sqrt(k / (2 E_dis)), dV = V_max - V_min, the elevation of the barrier top above the shifted well minimum of V_eff at the applied force, and dV = 0 once the force is large enough that the well minimum and the barrier top merge and vanish. Here E_dis is the bond current dissociation energy, including any radical weakening. The pulling force enters V_eff only after conversion, F_c = 602.2 kJ mol^-1 nm^-1 per nN. The frequency prefactor nu is the implementation calibrated default for this scission law and is part of what must be recovered from it. Evaluate the rate at T = 300 K with R = 8.31446261815324e-3 kJ mol^-1 K^-1.

* Hydrolysis (the study experiment-based force-clamp relation): ln(k_hyd / s^-1) is piecewise linear in the applied force in nN: 26.26 F - 19.77 for F <= 0.65 nN and -20.342988 + 0.070648 T_c + 1.605233 F for F > 0.75 nN, where T_c = 294.15 K is the temperature of the underlying single-molecule force-clamp calibration. In the transition band 0.65 < F <= 0.75 nN, both branch formulas must be evaluated at F and interpolated linearly between the two resulting ln k values, using the fraction (F - 0.65) / 0.10. The rate is k_hyd = exp(ln k_hyd) in s^-1.

* Selection: each draw is one step of the study rejection-free kinetic Monte Carlo over the events available at that draw, as described above.

In <reasoning>, state the homolysis prefactor you used, the six stage-one rupture barriers, the six stage-one rates of each pathway, the conditional probability that the second event is a homolysis after a hydrolysis at each of the six bonds, and the probability of the required pattern after the first, second and third events. These are the quantities that determine the final number.
State these quantities, the output requirements below give the tag rules

Output Format Requirements:
Emit <final_answer> immediately, then <reasoning>. Do not write a long derivation before the tags.
You must emit exactly one finite decimal inside <final_answer>...</final_answer>, even if the value is approximate or you are unsure.
Rules:
- The tags are required. Do not omit them or leave them empty.
- The tagged value must be a finite decimal (examples: 0.4847, 0.22, 1.05). Not NaN, not Inf, not a fraction string, not a vector, not prose.
- Put only that one number between the tags. No units, no words, no extra lines.
Keep <reasoning> clear and precise. Show only the few scalars that determine the final number.
Do not paste the input matrices, full coefficient vectors, per-iteration paths, or per-fold candidate tables.

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

01_site_table

Goal
----
Step 1 - the reactive-bond site table from the ensemble extension records.

```python
def site_table(records: "list[tuple]") -> "np.ndarray":
    """Validate the ensemble records and return the site table.

    Parameters
    ----------
    records : list of tuple
        Each entry is (label, E_dis, k, instrument, replicas): label str, dissociation
        energy E_dis in kJ/mol (positive), force constant k in kJ/mol/nm^2 (positive),
        instrument either "A" (extensions recorded in nm) or "B" (Angstrom), and
        `replicas`, a non-empty sequence of replica trajectories. Each trajectory is a
        non-empty 1-D sequence of extension readings, in the instrument length unit and
        in time order, finite and signed; a reading of exactly 0.0 marks a frame whose
        force evaluation did not converge.

    Returns
    -------
    np.ndarray
        Shape (n, 3), float array of [E_dis, k, F] rows, in input order: F is the bond's
        ensemble force in nN, the mean of its surviving replica production means,
        rounded to two decimals. Labels are validated but the numeric triple is returned.

    Raises
    ------
    ValueError
        If the record list is empty, if any entry is not a 5-tuple, if the label is not
        a string, if E_dis or k is not finite or not positive, if the instrument is
        neither "A" nor "B", if a replica is empty or not one-dimensional, if any
        reading is not finite, if a replica has no production reading after its last
        non-converged reading, or if every replica is discarded as contaminated.
    """
    return triples  # placeholder
```

### Step 2

02_stationary_points

Goal
----
Step 2 - the two stationary points of the force-shifted bond potential.

```python
def stationary_points(site_array: "np.ndarray") -> "np.ndarray":
    """The shifted well minimum and the barrier top of every loaded site.

    Parameters
    ----------
    site_array : array-like, shape (n, 3)
        Rows [E_dis, k, F]: dissociation energy in kJ/mol (positive, current value
        including any weakening), force constant in kJ/mol/nm^2 (positive), pulling force
        in nN (positive).

    Returns
    -------
    np.ndarray
        Shape (n, 2), float, in site-array order: column 0 is the extension in nm of the
        shifted well minimum of V_eff, column 1 the extension in nm of its barrier top.
        Where the force is at or above the largest the bond's own potential can sustain the
        two stationary points have merged, and both columns hold that one merged extension.

    Raises
    ------
    ValueError
        If the array is not (n, 3) or is empty, if any entry is not finite, or if any of
        E_dis, k or F is not positive.
    """
    return points  # placeholder
```

### Step 3

03_site_rates

Goal
----
Step 3 - the two competing rates of every reactive site.

```python
def site_rates(site_array: "np.ndarray") -> "tuple[np.ndarray, np.ndarray]":
    """Homolysis and hydrolysis rate of every site, in s^-1.

    Parameters
    ----------
    site_array : array-like, shape (n, 3)
        Rows [E_dis, k, F]: dissociation energy in kJ/mol (positive, current value
        including any weakening), force constant in kJ/mol/nm^2 (positive), pulling force
        in nN (positive).

    Returns
    -------
    tuple of np.ndarray
        (r_hom, r_hyd), each shape (n,) and dtype float, in site-array order.

    Raises
    ------
    ValueError
        If the array is not (n, 3) or is empty, if any entry is not finite, or if any of
        E_dis, k or F is not positive.
    """
    return r_hom, r_hyd  # placeholder
```

### Step 4

04_selection_probability

Goal
----
Step 4 - the rejection-free kMC first-event selection probability.

```python
def selection_probability(r_target: "np.ndarray", r_other: "np.ndarray") -> float:
    """Weighted first-event probability of the target pathway.

    Parameters
    ----------
    r_target : np.ndarray
        Shape (n,), rate of the pathway being asked about at each site, in s^-1
        (non-negative).
    r_other : np.ndarray
        Shape (n,), rate of the competing pathway at each site, in s^-1 (non-negative).

    Returns
    -------
    float
        P(target) = sum(r_target) / (sum(r_target) + sum(r_other)).

    Raises
    ------
    ValueError
        If the arrays are not equal-length one-dimensional non-negative finite arrays, or
        if the total rate is zero.
    """
    return p  # placeholder to fill
```

### Step 5

05_branch_probabilities

Goal
----
Step 5 - the first-event branch probabilities of the rejection-free kMC draw.

```python
def branch_probabilities(r_target: "np.ndarray", r_other: "np.ndarray") -> "np.ndarray":
    """First-event branch probability of the target pathway, per site.

    Parameters
    ----------
    r_target : np.ndarray
        Shape (n,), rate of the pathway being asked about at each site, in s^-1
        (non-negative).
    r_other : np.ndarray
        Shape (n,), rate of the competing pathway at each site, in s^-1 (non-negative).

    Returns
    -------
    np.ndarray
        Shape (n,); w_i = r_target_i / (sum(r_target) + sum(r_other)). The entries sum to
        the overall first-event target-pathway probability, not to one.

    Raises
    ------
    ValueError
        If the arrays are not equal-length one-dimensional non-negative finite arrays,
        or if the total rate is zero.
    """
    return w  # placeholder
```

### Step 6

06_relax_forces

Goal
----
Step 6 - the network relaxation that follows a break.

```python
def relax_forces(site_array: "np.ndarray",
                 broken_index: int,
                 transfer_fraction: float) -> "np.ndarray":
    """The surviving bonds' forces after the network relaxes, in nN.

    Parameters
    ----------
    site_array : array-like, shape (n, 3)
        Rows [E_dis, k, F] at the current stage, in original table order. F holds the forces
        current at this break, after any previous relaxation.
    broken_index : int
        Row index (0-based) of the bond that breaks at this event, in the PRE-break array.
    transfer_fraction : float
        The network's load-transfer fraction t, finite and non-negative: relaxation leaves
        the survivors carrying t times the broken bond's force in addition to their own.

    Returns
    -------
    np.ndarray
        Shape (n-1,), float: the survivors' forces in nN, in original table order, each
        rounded to four decimal places. An empty array when the broken bond was the only
        bond in the pool.

    Raises
    ------
    ValueError
        If the array is not (n, 3) or is empty, if any entry is not finite, if any of E_dis,
        k or F is not positive, if broken_index is outside [0, n), if transfer_fraction is
        not finite or is negative, or if the survivors cannot take up the transferred load
        with every one of them still below the largest force its own potential can sustain.
    """
    return forces  # placeholder
```

### Step 7

07_apply_break

Goal
----
Step 7 - applying one break: load redistribution and radical migration.

```python
def apply_break(site_array: "np.ndarray",
                broken_index: int,
                pathway: str,
                transfer_fraction: float,
                weakening_fraction: float) -> "np.ndarray":
    """The surviving pool after one break, relaxed and (if homolytic) weakened.

    Parameters
    ----------
    site_array : array-like, shape (n, 3)
        Rows [E_dis, k, F] at the current stage, in original table order. F holds the forces
        current at this break, after any previous relaxation.
    broken_index : int
        Row index (0-based) of the bond that breaks at this event, in the PRE-break array.
    pathway : str
        Either "hom" for a homolytic scission or "hyd" for a hydrolytic cleavage. Only a
        homolytic break leaves a radical.
    transfer_fraction : float
        The network's load-transfer fraction t, finite and non-negative, passed to the
        relaxation of step 6.
    weakening_fraction : float
        The network's radical weakening fraction r, in [0, 1). It is validated whichever
        pathway is given, and it changes nothing when the pathway is "hyd".

    Returns
    -------
    np.ndarray
        Shape (n-1, 3), the survivors in original table order, their forces relaxed and
        rounded to four decimals, and for a homolytic break the attacked row's E_dis
        multiplied by (1 - weakening_fraction) and rounded to four decimals. An empty
        (0, 3) array when the broken bond was the only bond in the pool.

    Raises
    ------
    ValueError
        If the array is not (n, 3) or is empty, if any entry is not finite, if any of E_dis,
        k or F is not positive, if pathway is neither "hom" nor "hyd", if broken_index is
        outside [0, n), if transfer_fraction is not finite or is negative, if
        weakening_fraction is not finite or leaves [0, 1), or if the relaxation of step 6
        has no solution for this pool.
    """
    return out
```

### Step 8

08_sequence_probability

Goal
----
Step 8 - the probability that the next events follow a prescribed pathway sequence.

```python
def sequence_probability(site_array: "np.ndarray",
                         pathways: "list[str]",
                         transfer_fraction: float,
                         weakening_fraction: float) -> float:
    """Probability that the next events follow the given pathway sequence, in order.

    Parameters
    ----------
    site_array : array-like, shape (n, 3)
        Rows [E_dis, k, F] of the pool the first event of the sequence is drawn from, in
        original table order.
    pathways : sequence of str
        The ordered pathways, each "hom" or "hyd"; length 1 to n.
    transfer_fraction : float
        The network's load-transfer fraction t, passed to every break.
    weakening_fraction : float
        The network's radical weakening fraction r, passed to every break.

    Returns
    -------
    float
        The probability in [0, 1] that the next len(pathways) events are exactly this
        sequence of pathways.

    Raises
    ------
    ValueError
        If the site array is invalid (see step 07), if pathways is empty, if it is longer
        than the number of bonds in the pool, if any element is neither "hom" nor "hyd", or
        if either rule fraction is invalid.
    """
    return p  # placeholder
```

### Step 9

09_four_event_mixed_cascade

Goal
----
Step 9 - orchestrator: the graded four-event mixed cascade.

```python
def four_event_mixed_cascade(records: "list[tuple]",
                             transfer_fraction: float,
                             weakening_fraction: float) -> float:
    """Run the complete record-to-cascade pipeline and return P(hyd, hom, hom, hom).

    Parameters
    ----------
    records : list of tuple
        Raw ensemble records in the Step 1 format. Each entry is
        (label, E_dis, k, instrument, replicas).
    transfer_fraction : float
        The network load-transfer fraction t, applied after every break.
    weakening_fraction : float
        The radical weakening fraction r, applied after every homolytic break.

    Returns
    -------
    float
        The four-event cascade probability in [0, 1].

    Raises
    ------
    ValueError
        If the raw records are invalid under Step 1; if fewer than four valid sites are
        supplied; if transfer_fraction is not finite or is negative; if
        weakening_fraction is not finite or lies outside [0, 1); or if a reached pool
        cannot satisfy the relaxation contract of Steps 6-8.
    """
    return p  # placeholder
```
