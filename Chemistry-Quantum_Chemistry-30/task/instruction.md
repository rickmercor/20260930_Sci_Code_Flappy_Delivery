# Chemistry-Quantum_Chemistry-30

## Background

Restricted open-shell Kohn–Sham theory describes a singlet excitation with two determinants that share one set of spatial orbitals. That shared spatial set lets an energy-decomposition analysis written for a single determinant be applied to a ROKS state. The chemically useful split separates a primary open-shell promotion into a frozen intermediate from the subsequent spectator relaxation of the remaining doubly occupied pairs into the final ROKS state. Occupied–virtual orbitals for chemical valence attach an electron-promotion number to each spectator pair.

### Required analysis

Using the source paper's ROKS excitation EDA and the supplied fixture, determine the total spectator OVOCV promotion of the relaxation from the frozen intermediate to the final ROKS state.

### Final result

The requested answer is the total spectator OVOCV promotion for the supplied numerical instance.

## Problem

The source paper gives an energy decomposition for two-determinant ROKS excitations that isolates spectator relaxation via a frozen intermediate between the closed-shell reference and the final ROKS state, then ranks that channel with occupied–virtual orbitals for chemical valence. This task is a deterministic numerical instance of that protocol on the six-function AO fixture printed below. Recover the paper's frozen-intermediate construction and spectator promotion map; do not invent a substitute. Report the total spectator OVOCV promotion of the relaxation from that intermediate to the final ROKS state on this instance.

The evaluation instance is the fixture printed in this prompt. There is no second workspace file, and nothing else is injected at runtime. Copy the printed arrays into your own code if you need them.

Do not run ROKS, DFT, or Q-Chem. Do not replace the protocol by a single-determinant OVOCV analysis of the ground and excited densities, by a Löwdin DDNO relaxation number, or by a published formaldehyde, HCl, or DMABN table value. This fixture is a six-function AO instance, not the ωB97X-D formaldehyde calculation in Table 1. The tagged value is this fixture's total spectator OVOCV promotion: not a sentinel, not an excitation energy, and not a charge reported in me⁻.

The printed inputs are the overlap, the closed-shell occupied MOs, the final-state doubly occupied MOs, and the two final-state SOMOs. Do not restate the paper's projector, intermediate-orbital selection, or promotion map here.

### Orbital fixture

The arrays below are a closed-shell ground state and a ROKS excited state in a six-function AO basis. Every number below is supplied. The MO blocks already satisfy the overlap orthonormality printed with each block.

- AO overlap S:

```
[[1.00, 0.14, 0.05, 0.02, 0.01, 0.00],
 [0.14, 1.00, 0.12, 0.04, 0.02, 0.01],
 [0.05, 0.12, 1.00, 0.10, 0.04, 0.02],
 [0.02, 0.04, 0.10, 1.00, 0.13, 0.05],
 [0.01, 0.02, 0.04, 0.13, 1.00, 0.11],
 [0.00, 0.01, 0.02, 0.05, 0.11, 1.00]]
```

- closed-shell occupied MO coefficients C_GS (n_ao × n_occ):

```
[[ 1.0080406599907743e+00, -6.9107734218542841e-02, -1.8590005312309209e-02],
 [-6.9107734218542841e-02,  1.0129364516936252e+00, -5.7436889827125644e-02],
 [-1.8590005312309157e-02, -5.7436889827125685e-02,  1.0098591410196720e+00],
 [-6.4935187113773817e-03, -1.4397439364686361e-02, -4.7260175428624082e-02],
 [-3.0063239218411234e-03, -6.3781347257252600e-03, -1.4208074758959877e-02],
 [ 1.2504076860857564e-03, -3.2331890567884867e-03, -6.8473280638734531e-03]]
```

- final-state doubly occupied MO coefficients C_d_ES:

```
[[-4.6576117886276669e-01,  1.0639291920844979e-01],
 [ 9.0863740302918772e-01, -1.6814437758770151e-01],
 [ 1.5572227232294725e-01,  9.4660762156982658e-01],
 [ 1.3340540574226461e-02, -1.8170610498559844e-01],
 [-1.7641371855794502e-01,  2.2510340558592395e-01],
 [ 1.9650940743279659e-02, -2.3842184982255774e-01]]
```

- first final-state SOMO h_ES:

```
[8.0062071599233220e-01, 3.0231974389437583e-01, -5.6840749389178262e-02,
 4.3523291657322111e-01, -3.2531705238407906e-02, -8.7198934719083226e-03]
```

- second final-state SOMO l_ES:

```
[-2.6573150693216712e-03, -1.4102687216989006e-02, 1.5210872916881013e-01,
 -4.3751913834973843e-02, 2.3206081914053003e-01, 9.3467599363415466e-01]
```

C_GS^T S C_GS = I, C_d_ES^T S C_d_ES = I, and {h_ES, l_ES} are S-orthonormal and orthogonal to C_d_ES. The paper's frozen-intermediate construction and the spectator OVOCV promotion numbers are not restated here.

Return the total spectator OVOCV promotion as a single numerical scalar.

Output Format Requirements:

Emit <final_answer> immediately, then <reasoning>. Do not write a long derivation before the tags.

You must emit exactly one finite decimal inside <final_answer>...</final_answer>, even if the value is approximate or you are unsure.

Rules:

- The tags are required. Do not omit them or leave them empty.
- The tagged value must be a finite decimal (examples: 0.4847, 0.22, 1.05). Not NaN, not Inf, not a fraction string, not a vector, not prose.
- Put only that one number between the tags. No units, no words, no extra lines.
- Keep <reasoning> short (a few hundred words). Show only the few scalars that determine the final number: the paper's frozen-intermediate and OVOCV constants you applied, the ground-state occupied count, the selected projection-state occupations, the spectator pair promotions, and the tagged total.
- Do not paste the input MO coefficients, the full densities, or the overlap matrix.

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

validate_roks_orbitals

Goal
----
Validate the overlap and the ROKS orbital sets of the source paper.

```python
def validate_roks_orbitals(
    S: np.ndarray,
    C_GS: np.ndarray,
    C_d_ES: np.ndarray,
    h_ES: np.ndarray,
    l_ES: np.ndarray,
) -> int:
    """
    Return the number of occupied spatial orbitals in the ground state.

    Parameters
    ----------
    S : np.ndarray
        AO overlap, shape (n, n).
    C_GS : np.ndarray
        Closed-shell ground-state MO coefficients, shape (n, n_occ).
    C_d_ES : np.ndarray
        Final-state doubly MO coefficients, shape (n, n_d).
    h_ES : np.ndarray
        First final-state SOMO coefficients, shape (n,).
    l_ES : np.ndarray
        Second final-state SOMO coefficients, shape (n,).

    Returns
    -------
    int
        Number of occupied spatial orbitals in the ground state.

    Raises
    ------
    ValueError
        If the overlap is invalid, the orbital sets are misaligned, or
        the S-orthonormality conditions fail.
    """
    return n_occ
```

### Step 2

ground_state_density

Goal
----
Form the closed-shell ground-state spatial density.

```python
def ground_state_density(
    C_GS: np.ndarray,
) -> np.ndarray:
    """
    Return the closed-shell ground-state AO spatial density.

    Parameters
    ----------
    C_GS : np.ndarray
        Closed-shell occupied MO coefficients, shape (n, n_occ).

    Returns
    -------
    np.ndarray
        Symmetric spin-summed AO spatial density
        P_GS = 2 C_GS C_GS^T, shape (n, n).

    Raises
    ------
    ValueError
        If C_GS is not a finite, nonempty coefficient matrix.
    """
    return P_GS
```

### Step 3

frozen_projection_density

Goal
----
Form the projection-state density that precedes the frozen intermediate.

```python
def frozen_projection_density(
    P_GS: np.ndarray,
    h_ES: np.ndarray,
    l_ES: np.ndarray,
    S: np.ndarray,
) -> np.ndarray:
    """
    Return the projection-state AO density.

    Parameters
    ----------
    P_GS : np.ndarray
        Closed-shell ground-state AO spatial density, shape (n, n).
    h_ES : np.ndarray
        First final-state SOMO coefficients, shape (n,).
    l_ES : np.ndarray
        Second final-state SOMO coefficients, shape (n,).
    S : np.ndarray
        AO overlap, shape (n, n).

    Returns
    -------
    np.ndarray
        Symmetric projection-state AO density, shape (n, n).

    Raises
    ------
    ValueError
        If the arrays are misaligned or numerical values are invalid.
    """
    return P_PRJ
```

### Step 4

purify_intermediate_state

Goal
----
Build a valid intermediate frozen-state density.

```python
def purify_intermediate_state(
    P_PRJ: np.ndarray,
    h_ES: np.ndarray,
    l_ES: np.ndarray,
    S: np.ndarray,
    n_occ: int,
) -> np.ndarray:
    """
    Return the intermediate frozen-state AO spatial density.

    Parameters
    ----------
    P_PRJ : np.ndarray
        Projection-state AO density, shape (n, n).
    h_ES : np.ndarray
        First final-state SOMO coefficients, shape (n,).
    l_ES : np.ndarray
        Second final-state SOMO coefficients, shape (n,).
    S : np.ndarray
        AO overlap, shape (n, n).
    n_occ : int
        Number of occupied spatial orbitals in the ground state.

    Returns
    -------
    np.ndarray
        Symmetric intermediate AO spatial density, shape (n, n).

    Raises
    ------
    ValueError
        If the arrays are misaligned, n_occ is invalid, or values are
        non-finite.
    """
    return P_INT
```

### Step 5

excited_state_density

Goal
----
Form the final ROKS spatial density.

```python
def excited_state_density(
    C_d_ES: np.ndarray,
    h_ES: np.ndarray,
    l_ES: np.ndarray,
) -> np.ndarray:
    """
    Return the final-state AO spatial density.

    Parameters
    ----------
    C_d_ES : np.ndarray
        Final-state occupied MO coefficients, shape (n, n_d).
    h_ES : np.ndarray
        First final-state SOMO coefficients, shape (n,).
    l_ES : np.ndarray
        Second final-state SOMO coefficients, shape (n,).

    Returns
    -------
    np.ndarray
        Symmetric AO spatial density, shape (n, n).

    Raises
    ------
    ValueError
        If the coefficient arrays are misaligned or non-finite.
    """
    return P_ES
```

### Step 6

ovocv_relaxation_promotions

Goal
----
Compute OVOCV electron-promotion numbers for the spectator relaxation.

```python
def ovocv_relaxation_promotions(
    P_INT: np.ndarray,
    P_ES: np.ndarray,
    S: np.ndarray,
) -> np.ndarray:
    """
    Return OVOCV promotion numbers of the spectator relaxation.

    Parameters
    ----------
    P_INT : np.ndarray
        Intermediate frozen-state AO spatial density, shape (n, n).
    P_ES : np.ndarray
        Final-state AO spatial density, shape (n, n).
    S : np.ndarray
        AO overlap, shape (n, n).

    Returns
    -------
    np.ndarray
        Promotion numbers of the spectator OVOCV pairs, sorted in
        descending order.

    Raises
    ------
    ValueError
        If the densities are misaligned, are not valid ROKS spatial
        densities, or are non-finite.
    """
    return dQ
```

### Step 7

run_roks_excitation_eda

Goal
----
Orchestrate the complete ROKS excitation EDA on the printed fixture.

```python
def run_roks_excitation_eda(
    S: np.ndarray,
    C_GS: np.ndarray,
    C_d_ES: np.ndarray,
    h_ES: np.ndarray,
    l_ES: np.ndarray,
) -> float:
    """
    Execute the complete ROKS excitation EDA pipeline.

    Parameters
    ----------
    S : np.ndarray
        AO overlap, shape (n, n).
    C_GS : np.ndarray
        Closed-shell ground-state occupied MO coefficients, shape (n, n_occ).
    C_d_ES : np.ndarray
        Final-state doubly occupied MO coefficients, shape (n, n_d).
    h_ES : np.ndarray
        First final-state SOMO coefficients, shape (n,).
    l_ES : np.ndarray
        Second final-state SOMO coefficients, shape (n,).

    Returns
    -------
    float
        Total spectator OVOCV promotion of the supplied instance.

    Raises
    ------
    ValueError
        If any input array is invalid or a pipeline stage returns a
        non-finite promotion.
    """
    return Q_relax
```
