# Physics-Optics-30

## Background

Photonic quantum computing builds entangling operations from linear optics and measurement: in the Knill-Laflamme-Milburn picture, a two-qubit gate such as the CNOT is realized by Hong-Ou-Mandel interference of the control and target photons on partially polarizing beam splitters, followed by post-selection on a two-fold coincidence. The fidelity of such gates is therefore set largely by the quality of the input photons, which is characterized by two independently measured quantities: the HOM visibility V between the two inputs and the second-order coherence g2(0) of each stream. These are reported in the same way for heralded spontaneous parametric down-conversion (SPDC) and four-wave-mixing (FWM) sources, atomic-ensemble memories and solid-state quantum-dot emitters, but their multiphoton components come from different physical mechanisms: in heralded sources the extra photon shares the signal's temporal mode, while in quantum dots re-excitation places it in a different temporal mode.

This paper develops an analytic, event-by-event model of multiphoton errors in the PPBS CNOT gate. It models the extra photon's temporal mode through a continuous signal-noise overlap that interpolates between identical and distinguishable noise, keeps every emission event with up to three photons (including events where a photon is lost through an unmonitored gate port), and evaluates the post-selected two-qubit density matrix for all four Bell-state preparations. It then models the HOM visibility measurement with the same multiphoton events and shows how the visibility absorbs the noise-model dependence, which leads to a single fidelity law in terms of V and g2(0) that is tested against the authors' own gate and published SPDC, atomic-ensemble and quantum-dot gate data with no free parameters.

## Problem

Interference-based linear-optical entangling gates turn the imperfections of their single-photon sources directly into logical errors. Two figures of merit are reported for essentially every source: the Hong-Ou-Mandel (HOM) visibility V measured between the two gate inputs, and the second-order coherence g2(0) of each input stream. A recent analytic treatment of the partially-polarizing-beam-splitter (PPBS) CNOT gate shows that the translation of these two numbers into a gate fidelity is subtler than the usual single-photon picture: the extra photon of a multiphoton emission may occupy a temporal mode that only partly overlaps the signal photon, three-photon events can survive two-fold post-selection (including events in which one photon leaves through an unmonitored port), and the very same multiphoton events also depress the measured visibility. Your task is to evaluate that treatment's Bell-state fidelity for one specific pair of unbalanced sources, starting from measured quantities only.

Here is the exact setup to use:

- Gate: the standard coincidence-basis PPBS CNOT. The control photon comes from arm m (the "memory" arm) prepared in |D>, the target photon from arm s (the "source" arm) prepared in |H>, so the ideal output is the Bell state |Phi+> = (|HH> + |VV>)/sqrt(2) in the (control, target) computational basis. The PPBSs transmit horizontal polarization with intensity transmission 1 and vertical with 1/3, with the usual Hadamard wave plates before and after the gate on the target line. Success is post-selected on one detection in each of the two monitored outputs (control output c, target output t). Each monitored output is analysed in polarization with detectors that resolve polarization but not photon number and that integrate over the full arrival-time window. The two unmonitored PPBS ports are not detected.
- Sources: each arm's photon-number distribution is truncated at two photons (P(n >= 3) = 0) and is fixed exactly, with no first-order expansion, by that arm's second-order coherence and mean photon number per pulse at the gate input: g2_m = 0.018 and nbar_m = 0.42 for the control arm, g2_s = 0.041 and nbar_s = 0.77 for the target arm.
- Noise mode: whenever an arm emits two photons, one is the signal photon and the other (the noise photon) occupies a single temporal mode whose normalized overlap with that arm's own signal mode is M_sn = 0.25 (M_sn = 1 is identical noise, M_sn = 0 is fully distinguishable noise). The same M_sn applies to both arms. The two arms' signal photons have an intrinsic indistinguishability eta (squared overlap of their temporal wavepackets) that is not given directly.
- Measured visibility: the two-source HOM visibility at the gate input, measured with the same two sources on a single balanced beam splitter and defined against the fully delayed coincidence rate, is V = 0.912. Decode eta from V using the paper's own forward model of how each emission event (single photons, pairs, and three-photon events at partial noise overlap) contributes to that HOM measurement. Do not set eta = V, and do not use a first-order inversion.
- Fidelity bookkeeping: build the post-selected two-qubit density matrix by summing the coincidence contributions of every emission event with at most three photons in total (the one-plus-one event, the two pair-plus-vacuum events, and the two pair-plus-single events; drop the pair-plus-pair event), each weighted by its emission probability relative to the one-plus-one event. Normalize once, and report F = <Phi+|rho|Phi+>. Keep the three-photon events in which one photon exits an unmonitored port while the remaining two still produce a coincidence. Consult the paper's own event-by-event coincidence matrices rather than assuming that a three-photon event contributes like a single-photon event with a reduced overlap.

Report the Bell-state fidelity F. The answer is graded to an absolute tolerance of 1e-6. In your reasoning, report: the per-arm pair-to-single and vacuum-to-single emission ratios; the decoded intrinsic indistinguishability eta and the effective indistinguishability that governs the three-photon events at this M_sn; the trace and |Phi+> overlap of the paper's coincidence matrix for the (pair in the control arm, single photon in the target arm) event at this instance, and the corresponding three-photon coincidence rate at the HOM beam splitter; the paper's multiphoton gate-error coefficient per unit g2 at this (eta, M_sn), next to its platform-independent pair coefficient; eta decoded from the same V at M_sn = 0 and at M_sn = 1, together with the fidelity obtained in each case; and the paper's closed-form first-order estimate written in terms of V and the arm-averaged g2 alone, with the signed gap F - F_estimate.

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

arm_emission_ratios

Goal
----
Convert one arm's measured source statistics (second-order coherence and mean photon number per pulse at the gate input) into the two emission-probability ratios that weight every multiphoton event in both the gate and the HOM bookkeeping.

```python
import numpy as np


def arm_emission_ratios(g2: float, nbar: float) -> "np.ndarray":
    """Pair-to-single and vacuum-to-single emission ratios of one arm.

    Args:
        g2 (float): second-order coherence g2(0) of this arm, g2 >= 0.
        nbar (float): mean photon number per pulse delivered at the gate
            input, nbar > 0.

    Raises:
        ValueError: if g2 < 0, if nbar <= 0, or if (g2, nbar) do not define a
            valid two-photon-truncated distribution (non-positive single-photon
            probability or negative vacuum probability).

    Expected return:
        np.ndarray of shape (2,): [w, r]. w >= 0 and vanishes at g2 = 0;
        r >= 0 and decreases as the arm gets brighter.
    """
    return None
```

### Step 2

effective_indistinguishability

Goal
----
Compute the single effective indistinguishability that governs every three-photon emission event (a pair in one arm plus a single photon in the other) when the noise photon only partly overlaps its own arm's signal mode.

```python
import numpy as np


def effective_indistinguishability(eta: float, m_sn: float) -> float:
    """Effective indistinguishability of three-photon events.

    Args:
        eta (float): intrinsic indistinguishability of the two arms' signal
            photons, 0 <= eta <= 1.
        m_sn (float): normalized overlap between a noise photon and its own
            arm's signal mode, 0 <= m_sn <= 1 (1 = identical noise,
            0 = fully distinguishable noise).

    Raises:
        ValueError: if eta or m_sn lies outside [0, 1].

    Expected return:
        float: eta_eff, proportional to eta, equal to eta for identical noise
        and strictly smaller than eta (for eta > 0) for any m_sn < 1.
    """
    return 0.0
```

### Step 3

cnot_event_table

Goal
----
Tabulate, for each of the five emission events kept in the fidelity bookkeeping, the trace and the target-Bell-state overlap of that event's post-selected coincidence matrix at the output of the PPBS CNOT gate (control in |D>, target in |H>, target Bell state |Phi+>).

```python
import numpy as np


def cnot_event_table(eta: float, m_sn: float) -> "np.ndarray":
    """Trace and Bell overlap of each retained emission event's coincidence matrix.

    Args:
        eta (float): intrinsic indistinguishability, 0 <= eta <= 1.
        m_sn (float): signal-noise overlap, 0 <= m_sn <= 1.

    Raises:
        ValueError: if eta or m_sn lies outside [0, 1].

    Expected return:
        np.ndarray of shape (5, 2). Rows (1,1), (2,0), (0,2), (2,1), (1,2);
        column 0 is the trace, column 1 the overlap with |Phi+>. All entries
        are non-negative; the two pair-plus-vacuum rows have zero overlap; the
        two three-photon rows are equal to each other.
    """
    return None
```

### Step 4

hom_event_rates

Goal
----
Compute the two-fold coincidence rate at a single balanced HOM beam splitter, per unit event weight, for each class of emission event: one photon per arm, a pair in one arm with the other arm empty, and a pair in one arm with a single photon in the other.

```python
import numpy as np


def hom_event_rates(eta: float, m_sn: float) -> "np.ndarray":
    """Per-event coincidence rates at the balanced HOM beam splitter.

    Args:
        eta (float): intrinsic indistinguishability, 0 <= eta <= 1.
        m_sn (float): signal-noise overlap, 0 <= m_sn <= 1.

    Raises:
        ValueError: if eta or m_sn lies outside [0, 1].

    Expected return:
        np.ndarray of shape (3,): [Gamma_11, Gamma_pair, Gamma_3]. Gamma_11
        vanishes for perfectly indistinguishable photons; Gamma_pair does not
        depend on eta; Gamma_3 decreases linearly with eta.
    """
    return None
```

### Step 5

hom_visibility

Goal
----
Forward model of the measured two-source HOM visibility: combine each arm's emission ratios with the per-event HOM coincidence rates, and compare the total coincidence rate with the fully delayed rate.

```python
import numpy as np


def hom_visibility(eta: float, m_sn: float, g2_m: float, nbar_m: float,
                   g2_s: float, nbar_s: float) -> float:
    """Two-source HOM visibility including multiphoton events.

    Args:
        eta (float): intrinsic indistinguishability, 0 <= eta <= 1.
        m_sn (float): signal-noise overlap (same for both arms), 0 <= m_sn <= 1.
        g2_m (float): second-order coherence of the control (memory) arm.
        nbar_m (float): mean photon number per pulse of the control arm.
        g2_s (float): second-order coherence of the target (source) arm.
        nbar_s (float): mean photon number per pulse of the target arm.

    Raises:
        ValueError: propagated from invalid arm statistics or invalid
            eta / m_sn.

    Expected return:
        float: V, equal to eta for ideal single-photon sources and lower than
        eta once multiphoton events are present (for 0 < eta <= 1).
    """
    return 0.0
```

### Step 6

decode_intrinsic_eta

Goal
----
Invert the forward visibility model exactly: recover the intrinsic indistinguishability eta from a measured visibility V, the signal-noise overlap and both arms' source statistics.

```python
import numpy as np


def decode_intrinsic_eta(v: float, m_sn: float, g2_m: float, nbar_m: float,
                         g2_s: float, nbar_s: float) -> float:
    """Exact decoding of eta from the measured HOM visibility.

    Args:
        v (float): measured two-source HOM visibility, 0 < v <= 1.
        m_sn (float): signal-noise overlap (same for both arms), 0 <= m_sn <= 1.
        g2_m (float): second-order coherence of the control arm.
        nbar_m (float): mean photon number per pulse of the control arm.
        g2_s (float): second-order coherence of the target arm.
        nbar_s (float): mean photon number per pulse of the target arm.

    Raises:
        ValueError: if v lies outside (0, 1], if the decoded eta exceeds 1,
            or if the arm statistics / m_sn are invalid.

    Expected return:
        float: eta, at least as large as v whenever multiphoton events are
        present, and equal to v for ideal single-photon sources.
    """
    return 0.0
```

### Step 7

cnot_gate_fidelity

Goal
----
Assemble the |Phi+> Bell-state fidelity of the PPBS CNOT gate from the five retained emission events, given the intrinsic parameters and both arms' source statistics.

```python
import numpy as np


def cnot_gate_fidelity(eta: float, m_sn: float, g2_m: float, nbar_m: float,
                       g2_s: float, nbar_s: float) -> float:
    """Bell-state fidelity of the PPBS CNOT gate with multiphoton events.

    Args:
        eta (float): intrinsic indistinguishability, 0 <= eta <= 1.
        m_sn (float): signal-noise overlap (same for both arms), 0 <= m_sn <= 1.
        g2_m (float): second-order coherence of the control arm.
        nbar_m (float): mean photon number per pulse of the control arm.
        g2_s (float): second-order coherence of the target arm.
        nbar_s (float): mean photon number per pulse of the target arm.

    Raises:
        ValueError: propagated from invalid eta, m_sn or arm statistics.

    Expected return:
        float: F in (0, 1], equal to 1 only for perfectly indistinguishable
        ideal single photons and decreasing as g2 grows.
    """
    return 0.0
```

### Step 8

run_cnot_fidelity_pipeline

Goal
----
Chain the earlier steps end to end: decode eta from the measured visibility, confirm the decoded eta reproduces that visibility through the forward model, and evaluate the gate's Bell-state fidelity. The reference implementation calls the earlier public functions by name rather than reproducing their contents.

```python
import numpy as np


def run_cnot_fidelity_pipeline(v: float, m_sn: float, g2_m: float, nbar_m: float,
                               g2_s: float, nbar_s: float) -> float:
    """Measured (V, g2, nbar) -> decoded eta -> CNOT Bell-state fidelity.

    Args:
        v (float): measured two-source HOM visibility, 0 < v <= 1.
        m_sn (float): signal-noise overlap (same for both arms), 0 <= m_sn <= 1.
        g2_m (float): second-order coherence of the control arm.
        nbar_m (float): mean photon number per pulse of the control arm.
        g2_s (float): second-order coherence of the target arm.
        nbar_s (float): mean photon number per pulse of the target arm.

    Raises:
        ValueError: if any input is invalid, if the decoded eta exceeds 1, or
            if the decoded eta fails to reproduce v through the forward model.

    Expected return:
        float: the Bell-state fidelity F.
    """
    return 0.0
```
