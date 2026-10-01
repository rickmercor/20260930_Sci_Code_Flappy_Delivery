# Physics-Computational_Physics-7

## Background

Molten salt reactors (MSRs) are a reactor class in which the nuclear fuel is dissolved directly in a liquid salt coolant, rather than held in solid fuel rods. This gives MSRs some attractive features, such as continuous online reprocessing and inherent safety mechanisms tied to the fuel's own thermal expansion, but it also introduces a modeling complication that solid-fuel reactors do not have: the fuel itself is constantly moving.

In a conventional reactor, a fission product born at a given location stays at that location (aside from slow physical diffusion) for the rest of its lifetime, so its irradiation history and eventual decay can be tracked as a purely local, time-dependent process. In a circulating-fuel reactor, a nuclide produced by fission in the neutron-irradiated core region is swept along with the flowing salt into ex-core piping, heat exchangers, and other loop components where the neutron flux is negligible or zero, before potentially being carried back into the core again. Whether a given nuclide is inside or outside the irradiated region at any moment strongly affects how much further irradiation, transmutation, or decay it undergoes, which in turn affects the reactor's isotopic inventory and reactivity over time.

This coupling between fuel transport and nuclear transmutation is especially important for isotopes that strongly absorb neutrons, since their buildup and depletion can significantly perturb reactor operation. Predicting how such isotopes accumulate in different parts of a circulating-fuel loop, and how operational choices like flow rate affect that accumulation, is a standard problem in the operational analysis and safety modeling of circulating-fuel reactor concepts.

## Problem

Some forms of Molten Salt Reactors (MSRs) circulate liquid fuel through a loop, so isotopic evolution depends on in-core neutron irradiation as well as flow-related transport between regions. Recent literature models the primary loop of a circulating-fuel reactor as a set of coupled well-mixed depletion regions, with inter-region salt exchange written directly into the nuclide balance equations.

Consider a simplified model of the Molten Salt Reactor Experiment (MSRE) primary loop with 3 cells connected in series. Track only the four isotopes: U-235, I-135, Xe-135, and Cs-135.

### Initial Conditions

$$N_{\text{U-235}}^{t=0} = 8.5 \times 10^{-4} \text{ atoms/barn·cm}$$

$$N_{\text{I-135}}^{t=0} = N_{\text{Xe-135}}^{t=0} = N_{\text{Cs-135}}^{t=0} = 0$$

All three cells start with this same composition. No I-135, Xe-135 or Cs-135 is present anywhere at $t = 0$.

* **Core:** Volume $V_1 = 1.3 \times 10^6$ cm³, irradiated with flux $\bar{\phi} = 6.0 \times 10^{14}$ cm$^{-2}$s$^{-1}$.
* **External loop:** Volume $V_2 = 6.2 \times 10^5$ cm³, $\bar{\phi} = 0$.
* **Pump bowl:** Volume $V_3 = 8.16 \times 10^4$ cm³, $\bar{\phi} = 0$.

### Reactor Parameters

| Cell | Volume [cm³] | Neutron Flux $\bar{\phi}$ [cm$^{-2}$s$^{-1}$] |
|------|-------------|----------------------------------|
| Core | $1.3 \times 10^6$ | $6.0 \times 10^{14}$ |
| External loop | $6.2 \times 10^5$ | - |
| Pump bowl | $8.16 \times 10^4$ | - |

The salt circulates through the loop: core → loop → pump bowl → core.

Xe-135 is the only tracked species removed from the salt. It leaves the pump-bowl salt at $6.72 \times 10^{-4}$ s⁻¹ and collects in an off-gas holdup tank that receives no salt flow and no neutron flux; it keeps decaying there and nothing returns to the salt.

U-235 is continuously added to the pump bowl at a rate of $3.5 \times 10^{-10}$ atoms·barn⁻¹·cm⁻¹·s⁻¹. This rate is the rate of change of the U-235 number density in the pump bowl cell.

### Cross Sections

| Nuclide | $\sigma_\gamma$ (n,$\gamma$) [b] | $\sigma_f$ (n,fission) [b] | $\nu$ (total) |
|---------|----------------------------------|---------------------------|---------------|
| U-235   | $98.71$ | $585.1$ | $2.436$ |
| I-135   | $80.03$ | — | — |
| Xe-135  | $2.778 \times 10^{6}$ | — | — |
| Cs-135  | $8.302$ | — | — |

### Decay Data

| Nuclide | $\lambda$ [s$^{-1}$] | Decay mode | Daughter |
|---------|-----------|---------------------|----------|
| U-235   | $3.12 \times 10^{-17}$ | $\alpha$ (stable on depletion timescale) | — |
| I-135   | $2.93 \times 10^{-5}$ | $\beta^-$ (BR = 1.0) | Xe-135 |
| Xe-135  | $2.11 \times 10^{-5}$ | $\beta^-$ (BR = 1.0) | Cs-135 |
| Cs-135  | $9.55 \times 10^{-15}$ | $\beta^-$ (effectively stable) | — |

### Fission Yields:

| Product | Yield $y_i$ |
|---------|------------|
| I-135   | $0.0628$ |
| Xe-135  | $0.0016$ |
| Cs-135  | $0$ |

Recent literature on depletion in circulating-fuel reactors solves the coupled system of well-mixed cells, including a constant external source, with a dedicated time-integration method. Using that literature's method, with ten equal steps over the 100 h interval, hold all physical coefficients constant, keep the external feed continuous, and optimize over $0 \le Q \le 7.5\times10^5$ cm³/s.

Explain the coupled model, the time-integration method and its treatment of the external source, and the physical origin of the optimum, including the uniform-composition-cell assumption, whether the source's justification for that assumption holds for Xe-135 in the core at the optimum, and the chemical basis for isotope removal. At the optimal flow rate, report the total Cs-135 inventory of the model at $t = 100$ h in atoms and where it is located.

Report the flow rate which maximizes the Cs-135 number density in the core cell at $t = 100$ h, in units of cm³/s, to 4 significant figures.

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

build_transmutation_matrix

Goal
----
Implement `build_transmutation_matrix` which constructs the neutron reaction matrix for a single cell.

```python
def build_transmutation_matrix(flux: float,
                               sigma_gamma: "np.ndarray", sigma_f: "np.ndarray",
                               fission_yields: "np.ndarray") -> "np.ndarray":
    '''Constructs the neutron transmutation matrix for a single cell.

    Parameters
    ----------
    flux : float
        Cell-averaged scalar neutron flux [cm⁻² s⁻¹].
    sigma_gamma : np.ndarray
        Shape (m,) radiative capture cross-sections [barn].
        Capture products leave the tracked nuclide set.
    sigma_f : np.ndarray
        Shape (m,) fission cross-sections [barn] (0 if not fissile).
    fission_yields : np.ndarray
        Shape (m, m) independent fission yields. fission_yields[j, i] = yield of i from j.

    Returns
    -------
    T : np.ndarray
        Shape (m, m) reaction-rate matrix acting on the isotope number-density
        vector. Rows identify produced nuclides and columns identify parents.

    Raises
    ------
    ValueError
        If flux is negative.
    '''
    return T
```

### Step 2

build_cell_matrix

Goal
----
Implement `build_cell_matrix` which adds radioactive decay and engineered removal
to the transmutation matrix to form the complete single-cell depletion matrix.

```python
def build_cell_matrix(T: "np.ndarray",
                      decay_constants: "np.ndarray", branching_ratios: "np.ndarray",
                      removal_rates: "np.ndarray") -> "np.ndarray":
    '''Adds decay and removal terms to the transmutation matrix.

    Parameters
    ----------
    T : np.ndarray
        Shape (m, m) transmutation matrix from step 1.
    decay_constants : np.ndarray
        Shape (m,) decay constants [s⁻¹].
    branching_ratios : np.ndarray
        Shape (m, m) decay branching ratios. branching_ratios[j, i] = BR for j → i.
    removal_rates : np.ndarray
        Shape (m,) engineered removal rates [s⁻¹] (0 if not removed).

    Returns
    -------
    A : np.ndarray
        Shape (m, m) complete depletion matrix.

    Raises
    ------
    ValueError
        If any decay constant is negative.
    '''
    return A
```

### Step 3

build_flow_coupling

Goal
----
Implement `build_flow_coupling` which constructs the flow coupling matrices for the multi-cell system.

```python
def build_flow_coupling(n_cells: int, m: int, volumes: "np.ndarray",
                        Q: float, flow_fractions: "np.ndarray"
                        ) -> tuple[list, list]:
    '''Constructs flow coupling matrices for the multi-cell system.

    Each cell has uniform composition, and flow transports all tracked isotopes
    with the bulk salt. The matrices act on isotope number densities.

    Parameters
    ----------
    n_cells : int
        Number of cells.
    m : int
        Number of tracked isotopes.
    volumes : np.ndarray
        Shape (n_cells,) cell volumes [cm³].
    Q : float
        Volumetric flow rate [cm³/s].
    flow_fractions : np.ndarray
        Shape (n_cells, n_cells) flow fractions. flow_fractions[k, l] = fraction from k to l.
        Row sums should equal 1.0 for cells with outflow.

    Returns
    -------
    outflow_diags : list
        List of n_cells (m, m) negative diagonal matrices for outflow from each cell.
    inflow_blocks : list of lists
        n_cells × n_cells list. inflow_blocks[k][l] is an (m, m) matrix for flow k → l,
        placed at block row l, block column k in the system matrix.

    Raises
    ------
    ValueError
        If n_cells < 1, Q < 0, or any volume is not positive.
    '''
    return outflow_diags, inflow_blocks
```

### Step 4

assemble_system_matrix

Goal
----
Implement `assemble_system_matrix` which builds the full coupled system matrix R.

```python
def assemble_system_matrix(cell_matrices: list, outflow_diags: list,
                           inflow_blocks: list) -> "np.ndarray":
    '''Assembles the full coupled system matrix R.

    Parameters
    ----------
    cell_matrices : list
        List of n (m, m) per-cell depletion matrices from build_cell_matrix.
    outflow_diags : list
        List of n (m, m) outflow diagonal matrices from build_flow_coupling.
    inflow_blocks : list of lists
        n × n list from build_flow_coupling. inflow_blocks[k][l] is an (m, m)
        flow matrix from cell k to cell l, placed at block row l, block column k.

    Returns
    -------
    R : np.ndarray
        Shape (m*n, m*n) system matrix.

    Raises
    ------
    ValueError
        If cell_matrices is empty.
    '''
    return R
```

### Step 5

solve_timestep

Goal
----
Implement `solve_timestep` which advances the isotopic state vector by one time step with a continuously acting external source.

```python
def solve_timestep(R: "np.ndarray", dt: float, N: "np.ndarray",
                   S: "np.ndarray") -> "np.ndarray":
    '''Evolves the linear system dN/dt = R N + S over the supplied time step.

    Parameters
    ----------
    R : np.ndarray
        Shape (n, n) system matrix, constant during the step.
        Singular and nearly singular matrices are valid inputs.
    dt : float
        Time step [s].
    N : np.ndarray
        Shape (n,) current state vector of isotopic number densities.
    S : np.ndarray
        Shape (n,) external source vector [atoms·barn⁻¹·cm⁻¹·s⁻¹],
        constant throughout the step. May be all zeros.

    Returns
    -------
    N_new : np.ndarray
        Shape (n,) state vector at time t + dt, including the continuously
        acting source.

    Raises
    ------
    ValueError
        If dt is not positive.
    '''
    return N_new
```

### Step 6

run_msre_depletion

Goal
----
Implement `run_msre_depletion` which sets up and runs the full MSRE depletion calculation.

```python
def run_msre_depletion(n_cells: int, m: int,
                       fluxes: "np.ndarray", volumes: "np.ndarray",
                       Q: float, flow_fractions: "np.ndarray",
                       sigma_gamma: "np.ndarray", sigma_f: "np.ndarray",
                       fission_yields: "np.ndarray",
                       decay_constants: "np.ndarray", branching_ratios: "np.ndarray",
                       removal_rates: "np.ndarray",
                       addition_rates: "np.ndarray",
                       N_0: "np.ndarray",
                       total_time: float, n_steps: int) -> float:
    '''Sets up and runs the MSRE depletion calculation.

    The supplied physical parameters remain constant over the simulation.

    Parameters
    ----------
    n_cells : int
        Number of cells.
    m : int
        Number of tracked isotopes.
    fluxes : np.ndarray
        Shape (n_cells,) cell-averaged fluxes [cm⁻² s⁻¹].
    volumes : np.ndarray
        Shape (n_cells,) cell volumes [cm³].
    Q : float
        Volumetric flow rate [cm³/s].
    flow_fractions : np.ndarray
        Shape (n_cells, n_cells) flow fractions.
    sigma_gamma : np.ndarray
        Shape (m,) capture cross-sections [barn].
    sigma_f : np.ndarray
        Shape (m,) fission cross-sections [barn].
    fission_yields : np.ndarray
        Shape (m, m) independent fission yields. fission_yields[j, i] = yield of i from j.
    decay_constants : np.ndarray
        Shape (m,) decay constants [s⁻¹].
    branching_ratios : np.ndarray
        Shape (m, m) decay branching ratios. branching_ratios[j, i] = BR for j → i.
    removal_rates : np.ndarray
        Shape (n_cells, m) per-cell removal rates [s⁻¹].
    addition_rates : np.ndarray
        Shape (n_cells, m) per-cell addition rates [atoms·barn⁻¹·cm⁻¹·s⁻¹].
    N_0 : np.ndarray
        Initial number densities [atoms·barn⁻¹·cm⁻¹].
        Shape (m,) for uniform initial composition (tiled across cells),
        or shape (n_cells*m,) for per-cell initial state.
    total_time : float
        Non-negative total simulation time [s]. At zero time, return the initial
        Cs-135 number density in the core cell without advancing the state.
    n_steps : int
        Number of equal time steps.

    Returns
    -------
    Cs135_core : float
        Cs-135 density in the core cell at the end of the simulation
        [atoms·barn⁻¹·cm⁻¹]. The core is cell 0; Cs-135 is isotope index 3.

    Raises
    ------
    ValueError
        If n_steps < 1 or total_time < 0.
    '''
    return Cs135_core
```

### Step 7

find_optimal_flow

Goal
----
Implement `find_optimal_flow` which finds the volumetric flow rate that maximises the core Cs-135 number density.

```python
def find_optimal_flow(n_cells: int, m: int,
                      fluxes: "np.ndarray", volumes: "np.ndarray",
                      flow_fractions: "np.ndarray",
                      sigma_gamma: "np.ndarray", sigma_f: "np.ndarray",
                      fission_yields: "np.ndarray",
                      decay_constants: "np.ndarray", branching_ratios: "np.ndarray",
                      removal_rates: "np.ndarray",
                      addition_rates: "np.ndarray",
                      N_0: "np.ndarray",
                      total_time: float, n_steps: int,
                      Q_min: float, Q_max: float) -> float:
    '''Finds the flow rate that maximises Cs-135 in the core.

    Parameters
    ----------
    n_cells : int
        Number of cells.
    m : int
        Number of tracked isotopes.
    fluxes : np.ndarray
        Shape (n_cells,) cell-averaged fluxes [cm⁻² s⁻¹].
    volumes : np.ndarray
        Shape (n_cells,) cell volumes [cm³].
    flow_fractions : np.ndarray
        Shape (n_cells, n_cells) flow fractions.
    sigma_gamma : np.ndarray
        Shape (m,) capture cross-sections [barn].
    sigma_f : np.ndarray
        Shape (m,) fission cross-sections [barn].
    fission_yields : np.ndarray
        Shape (m, m) independent fission yields. fission_yields[j, i] = yield of i from j.
    decay_constants : np.ndarray
        Shape (m,) decay constants [s⁻¹].
    branching_ratios : np.ndarray
        Shape (m, m) decay branching ratios. branching_ratios[j, i] = BR for j → i.
    removal_rates : np.ndarray
        Shape (n_cells, m) per-cell removal rates [s⁻¹].
    addition_rates : np.ndarray
        Shape (n_cells, m) per-cell addition rates [atoms·barn⁻¹·cm⁻¹·s⁻¹].
    N_0 : np.ndarray
        Initial number densities [atoms·barn⁻¹·cm⁻¹]. Shape (m,) for uniform
        composition across all cells, or shape (n_cells*m,) for per-cell state.
    total_time : float
        Non-negative total simulation time [s], as in run_msre_depletion.
    n_steps : int
        Number of equal time steps.
    Q_min : float
        Inclusive lower bound for Q [cm³/s].
    Q_max : float
        Inclusive upper bound for Q [cm³/s].

    Returns
    -------
    Q_opt : float
        Four-significant-figure representation of the flow rate that maximises
        the core Cs-135 density at total_time over [Q_min, Q_max].
        The bounds apply before rounding; the returned value may lie outside them.

    Raises
    ------
    ValueError
        If Q_min exceeds Q_max.
    '''
    return Q_opt
```
