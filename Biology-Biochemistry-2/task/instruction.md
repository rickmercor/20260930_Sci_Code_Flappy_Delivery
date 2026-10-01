# Biology-Biochemistry-2

## Background

When a mammalian skeletal-muscle cell contracts, the calcium driving it is released at specialised junctions where the cell voltage-sensing membrane is facing the calcium store: rows of calcium-release channels sit directly opposite groups of voltage sensors. Only some of these channels actually lie against a sensor cluster, which means that the junction releases depends on how far the influence of those sensor-facing channels bring to the rest. 
Whether that neighbour-neighbour recruitment is strong enough to explain how the junction produces graded, inactivating calcium release under voltage control has been an open question in muscle physiology for a long time. 
Competing answers give measurably different voltage dependence for the flux recorded at the level of the whole cell. 

The laboratory behind the modelling treatment used here studies the junction as a statistical system: an ensemble of identical, mutually influencing stochastic channels whose free-energy changes derive from electrical work on the sensors. This connects the junction-level mechanism to cell-level flux records, subcellular calcium-spark events, and coupled gating in bilayers, across muscle tissues and species.

## Problem

The skeletal muscle contraction is driven by Ca2+ release through ryanodine receptor channels arranged in junctional arrays of two channel classes: one class is gated by adjacent voltage sensors, while the other class is not and is instead modulated through interactions with neighboring channels. A recent statistical-mechanical model treats such an array as a stochastic ensemble of interacting channels and reproduces the experimentally observed Ca2+ flux-voltage relationship of skeletal muscle with a single parameter set. 

Its inputs are a channel lattice, a kinetic parameter set, and a voltage-pulse protocol: its outputs are the ensemble-averaged Ca2+ flux time courses and the voltage dependence of their peak flux.

The task is to compute one concrete deterministic example of this formulation, using the recent literature to identify the quantitative treatment whose assumptions and representation match the supplied configuration. To do so, use the following configuration:

* lattice: 4 V-class + 4 C-class channels in a checkerboard double row, every contact is cross-class, interior channels contact 3 cross-class neighbors, and the two end channels of each row contact 2
* k_plus = 0.001 ms^-1 (opening rate), k_minus = 2.0 ms^-1 (closing rate)
* nu = 1.0, nu_V = 0.8 (distribution coefficients, their roles follow the matching published treatment)
* eps_OO = -5.7, eps_CC = -0.7 (kT)
* half-times = [3.5, 50, 20, 50] ms (inactivation and recovery half-times, in the order I1_ht, I2_ht, R1_ht, R2_ht)
* R_CV = 5 (flux of one open C-class channel in units of the flux through one open V channel)
* holding potential V_rest = -90 mV, pulse protocol samples the energy family eps_V = {0, -2, -4, -6, -7, -8, -9, -10, -12, -14} (kT), kT-to-mV conversion = 7.14 mV per kT

Compute the voltage-dependent gating rates and the coupling-modified transition rates at each pulse energy, the ensemble-averaged flux time course at each pulse energy, the peak flux of each ensemble-averaged time course, and the parameters of the Boltzmann fit to the peak flux-energy curve. The final answer must be a single number: the Boltzmann slope factor of the peak flux-voltage curve, expressed as a positive quantity in mV.

Numerical conventions:

* Use IEEE-754 binary64 arithmetic, do not round intermediate quantities.
* All channels are closed at t = 0 (resting configuration at the holding potential).
* The ensemble-averaged flux is the exact statistical expectation of the channel ensemble flux (the limit of the model average over its stochastic realizations), free of sampling error.
* Flux time courses are evaluated on a uniform 0.02 ms grid over [0, 100] ms.
* The peak flux is the maximum of the ensemble-averaged flux time course, no smoothing is applied.
* The Boltzmann fit is a three-parameter unweighted least-squares fit of the peak fluxes against the magnitudes |eps_V|, the slope factor is reported as a positive quantity and converted to mV with the supplied 7.14 mV per kT.

Output Format Requirements:
Emit <final_answer> immediately, then <reasoning>. Do not write a long derivation before the tags.
You must emit exactly one finite decimal inside <final_answer>...</final_answer>, even if the value is approximate or you are unsure.
Rules:
- The tags are required. Do not omit them or leave them empty.
- The tagged value must be a finite decimal (examples: 0.4847, 0.22, 1.05). Not NaN, not Inf, not a fraction string, not a vector, not prose.
- Put only that one number between the tags. No units, no words, no extra lines.
Keep <reasoning> short (a few hundred words). Show only the few scalars that determine the final number.
Do not paste the input matrices, full coefficient vectors, per-iteration paths, or per-fold candidate tables.

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

01_build_lattice_adjacency

Goal
----
Step 1 - the couplon lattice as a cross-class adjacency matrix.

```python
import numpy as np

def build_lattice_adjacency(n_per_class: int = 4) -> tuple:
    """Build the couplon's class mask and adjacency matrix for the checkerboard lattice.

    Parameters
    ----------
    n_per_class : int
        Number of channels of each class. The lattice has two rows of `n_per_class`
        channels, so the total channel count is 2 * n_per_class.

    Returns
    -------
    tuple (is_v, adjacency)
        is_v : np.ndarray
            Shape (2 * n_per_class,), float; entry 1.0 marks a channel of the class in
            contact with the voltage sensors, entry 0.0 the contact-free class.
            The checkerboard convention used elsewhere in the pipeline places the
            sensor-coupled class on lattice positions of even (row + column) parity.
        adjacency : np.ndarray
            Shape (2 * n_per_class, 2 * n_per_class), float; symmetric 0.0/1.0 matrix of
            the four-neighbourhood cross-class contacts.

    Raises
    ------
    ValueError
        If `n_per_class` is less than 2 or not an integer.
    """
    is_v = np.empty(2 * n_per_class, dtype=float)
    adjacency = np.empty((2 * n_per_class, 2 * n_per_class), dtype=float)
    return is_v, adjacency  # to complete!
```

### Step 2

02_contact_energy_changes

Goal
----
Step 2 - contact energy change on opening per channel.

```python
import numpy as np

def contact_energy_changes(adjacency: np.ndarray, open_mask: np.ndarray) -> np.ndarray:
    """Net contact energy change on opening for each channel of the couplon lattice.

    Parameters
    ----------
    adjacency : np.ndarray
        Shape (n, n), floating or boolean 0-1 matrix of the cross-class contacts, as built
        by `build_lattice_adjacency`. Must be square and symmetric within tolerance.
    open_mask : np.ndarray
        Shape (n,), 0.0/1.0 marks of which channels are currently open. Only open channels
        of either class contribute contact energy to a neighbour's opening barrier.

    Returns
    -------
    np.ndarray
        Shape (n,), native floats: the net contact energy change on opening, in kT, for
        each channel of the lattice.

    Raises
    ------
    ValueError
        If shapes are inconsistent, `adjacency` is not square/symmetric, `open_mask`
        contains non-binary marks, or values are not finite.
    """
    delta_energy_vector = np.empty(adjacency.shape[0], dtype=float)
    return delta_energy_vector  # placeholder to complete!
```

### Step 3

03_channel_transition_rates

Goal
----
Step 3 - per-channel transition rates of the two-class kinetic strategy.

```python
import numpy as np
 
def channel_transition_rates(state: int, is_v: bool, delta_energy: float,
                             eps_v: float, nu: float = 1.0, nu_v: float = 0.8) -> np.ndarray:
    """Rates of every transition leaving one channel, indexed by destination state.

    Parameters
    ----------
    state : int
        Current state of the channel, encoded 0 = closed, 1 = open, 2 = first inactivated
        state, 3 = second inactivated state.
    is_v : bool
        True if the channel belongs to the class in contact with the voltage sensors.
    delta_energy : float
        Net contact energy, in kT, associated with this channel's closed-to-open transition
        given the current states of its neighbours.
    eps_v : float
        Electrical energy of the applied pulse, in kT, delivered to the sensor-coupled class.
    nu : float, optional
        Distribution coefficient for the contact energy.
    nu_v : float, optional
        Distribution coefficient for the electrical energy.

    Returns
    -------
    numpy.ndarray
        Array of shape (4,) of native floats, entry d holding the rate in ms^-1 of the
        transition from `state` to state d, and 0.0 wherever no such transition exists,
        including entry `state` itself.

    Raises
    ------
    ValueError
        If `state` is not one of 0, 1, 2, 3; if a sensor-coupled channel is given an
        inactivated state; or if `delta_energy`, `eps_v`, `nu` or `nu_v` is not finite.
    """
    rates = np.empty(4, dtype=float)
    return rates  # placeholder to complete!
```

### Step 4

04_assemble_generator

Goal
----
Assemble the couplon's Markov generator from the per-channel transition law.

```python
import numpy as np

def assemble_generator(is_v, adjacency, eps_v, nu=1.0, nu_v=0.8):
    """Assemble the couplon generator in coordinate form at one pulse energy.

    Parameters
    ----------
    is_v : np.ndarray
        Shape (n,), 0.0/1.0 marks of the sensor-coupled class, from step 01.
    adjacency : np.ndarray
        Shape (n, n), float 0.0/1.0 cross-class contact matrix, from step 01.
    eps_v : float
        Electrical energy of the applied pulse, in kT, delivered to the sensor-coupled
        class.
    nu, nu_v : float
        Distribution coefficients of the treatment, applied to the contact and electrical
        energy changes respectively.

    Returns
    -------
    tuple (rows, cols, values)
        rows : np.ndarray of shape (nnz,), integer row indices (destination states)
        cols : np.ndarray of shape (nnz,), integer column indices (source states)
        values : np.ndarray of shape (nnz,), float transition rates in ms^-1
        The full generator is the 4^4 x 2^4 = 4096 (for the graded lattice) sparse matrix
        with entries values[k] at (rows[k], cols[k]); every column sums to zero.

    Raises
    ------
    ValueError
        If the inputs are inconsistent in shape, or if eps_v / nu / nu_v are not finite.
    """
    n_channels = np.asarray(is_v).size
    n_states = int(np.prod([2 if v else 4 for v in np.asarray(is_v)]))
    rows = np.empty(0, dtype=np.int64)
    cols = np.empty(0, dtype=np.int64)
    values = np.empty(0, dtype=float)
    return rows, cols, values  # placeholder to fill!
```

### Step 5

05_solve_master_equation

Goal
----
Solve the main equation and read out the ensemble-averaged flux time course

```python
import numpy as np
from scipy.integrate import solve_ivp
from scipy.sparse import csc_matrix

def solve_master_equation(rows: np.ndarray, cols: np.ndarray, values: np.ndarray,
                          flux_per_state: np.ndarray) -> np.ndarray:
    """Ensemble-averaged flux time course from a couplon generator in coordinate form.

    Parameters
    ----------
    rows, cols, values : np.ndarray
        Coordinate form of the generator Q (from step 04): destination state, source state,
        and transition rate in ms^-1. Values at rows == cols hold the diagonal entries.
    flux_per_state : np.ndarray
        Shape (n_states,). The flux observable: the contribution of each global state to
        the ensemble flux (open V channels count 1, open C channels count R_CV = 5, every
        other state component counts 0).

    Returns
    -------
    np.ndarray
        Shape (5001,): the exact ensemble-averaged flux on the uniform 0.02 ms grid over
        [0, 100] ms, starting from the all-closed configuration (global state index 0).

    Raises
    ------
    ValueError
        If the coordinate arrays are inconsistent, contain non-finite values, if the flux
        vector does not match the inferred state space, or if the inference of the state
        space size fails.
    """
    flux_time_course = np.empty(5001, dtype=float)
    return flux_time_course  # placeholder to fill
```

### Step 6

06_extract_flux_features

Goal
----
Peak and steady flux summarizers of one ensemble-averaged time course.

```python
import numpy as np

def extract_flux_features(flux_time_course: np.ndarray) -> np.ndarray:
    """Peak and steady flux of one ensemble-averaged trace.

    Parameters
    ----------
    flux_time_course : np.ndarray
        Shape (n,), the ensemble-averaged flux on the uniform 0.02 ms grid.

    Returns
    -------
    np.ndarray
        Shape (2,), native floats: [peak, steady], where peak is the trace maximum (no
        smoothing) and steady is the mean of the last 5 ms of the trace.

    Raises
    ------
    ValueError
        If the input is not a one-dimensional finite array with at least one full steady
        window of points.
    """
    features = np.empty(2, dtype=float)
    return features  # placeholder to complete
```

### Step 7

07_fit_boltzmann

Goal
----
The treatment Boltzmann fit of the peak flux-voltage curve.

```python
import numpy as np
from scipy.optimize import curve_fit

def fit_boltzmann(eps_family: np.ndarray, peak_fluxes: np.ndarray) -> np.ndarray:
    """Three-parameter Boltzmann fit of the peak flux-energy curve.

    Parameters
    ----------
    eps_family : np.ndarray
        Shape (m,), the pulse energies in kT (non-positive in the supplied family).
    peak_fluxes : np.ndarray
        Shape (m,), the ensemble-averaged peak flux at each pulse energy.

    Returns
    -------
    np.ndarray
        Shape (3,), native floats: [F_max, eps_bar, kappa], where eps_bar carries the
        treatment's sign convention (negative energy) and kappa is the positive slope
        factor, in kT.

    Raises
    ------
    ValueError
        If the inputs are inconsistent, shorter than three points, or non-finite.
    """
    params = np.empty(3, dtype=float)
    return params  # placeholder to complete!
```

### Step 8

08_orchestrator_kappa_mv

Goal
----
Orchestrator: the full graded pipeline, from lattice to the Boltzmann slope in mV.

```python
import numpy as np
 
def couplon_kappa_mv(n_per_class=4):
    """Boltzmann slope factor of the peak flux-voltage curve, in mV, for the graded couplon.

    Parameters
    ----------
    n_per_class : int
        Channels per class in the lattice (4 in the graded configuration). Kept as an
        argument so callers can build smaller exact configurations for smoke tests.

    Returns
    -------
    float
        The positive Boltzmann slope factor of the peak flux-voltage curve in mV,
        kappa(kT) * 7.14 mV/kT.

    Raises
    ------
    ValueError
        If n_per_class is less than 2 or not an integer.
    """
    kappa_mv = 0.0
    return kappa_mv  # placeholder to complete!
```
