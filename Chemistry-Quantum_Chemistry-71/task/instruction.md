# Chemistry-Quantum_Chemistry-71

## Background

Electrocyclic reactions close a conjugated polyene into a ring, or open the ring again, through a single concerted motion in which the terminal groups rotate. The Woodward-Hoffmann rules predict which sense of rotation is thermally allowed: for six pi electrons the disrotatory motion, which keeps a mirror plane, and for four the conrotatory motion, which keeps a twofold axis. Along the forbidden motion a bonding orbital of the reactant correlates with an antibonding orbital of the product, so the frontier orbitals must cross, and the equivalent Dewar-Zimmerman view describes the forbidden transition state as an antiaromatic, Möbius-type ring.

Multiconfigurational calculations show what happens at such a crossing: the ground state stops being a single closed-shell configuration and becomes an open-shell singlet diradicaloid, with the highest occupied and lowest unoccupied natural orbitals approaching single occupation. The same phenomenon appears in broken-symmetry unrestricted Hartree-Fock theory as an instability of the restricted solution, and the diradical character can be quantified either from correlated natural occupations or from spin-projected unrestricted solutions. Not every forbidden pathway carries a large barrier, however; strong polarization or extended conjugation can stabilize the transition state or give it zwitterionic rather than diradical character.

A recent group-theoretic treatment formalizes forbiddenness as an obstruction to reaching a symmetry-breaking target state by symmetry-preserving evolution, and maps the frontier bond onto a two-level system whose rotation angle measures the loss of bonding character. Minimal correlated pi-electron models with an on-site repulsion reproduce the orbital crossings of these reactions qualitatively and allow the relation between correlated and mean-field measures of diradical character to be studied on paths where the symmetry constraint is enforced exactly.

## Problem

Woodward-Hoffmann forbidden electrocyclizations can be described as a symmetry-induced obstruction to bonding disruption: along a path that keeps the symmetry element, the frontier bond has to pass from closed-shell to open-shell singlet character. A recent study quantifies this loss of bonding character with a two-level frontier-bond representation whose rotation angle Theta is obtained from the highest occupied natural-orbital occupation, with associated ionic and covalent weights defined by the same source construction. For one diarylethene that study reports CAS(10,10) rises of 28.3 degrees on the forbidden conrotatory path and 6.1 degrees on the allowed disrotatory path. The same source connects a Yamaguchi spin-projected unrestricted-Hartree-Fock diradical measure to an analogous mixing angle. The question here is what that inexpensive spin-projected estimate gives at the forbidden transition state of a correlated pi model matched to both reported rises.
The model is the ring closure of Z-1,3,5-hexatriene to 1,3-cyclohexadiene with six pi sites 1 to 6 along the chain and six pi electrons, described by a Hubbard Hamiltonian with one-electron matrix h (zero diagonal) and on-site repulsion U between electrons of opposite spin. Bonds 1-2, 3-4 and 5-6 have resonance integral -2.8 eV and bonds 2-3 and 4-5 share a single-bond resonance integral beta_s, and all interior p orbitals point along +z. Sites 1 and 6 lie on the x axis, site 1 at negative x, at distance r = 3.0 - 1.46 s angstrom for reaction progress s from 0 (open chain) to 1 (ring). With phi = 90 s degrees the p orbital on site 1 points along a = (sin phi, 0, cos phi), tilting towards site 6, and the p orbital on site 6 is the image of a under the conserved symmetry element: the twofold rotation about the y axis for the conrotatory path and the reflection through the plane x = 0 for the disrotatory path. The couplings 1-2 and 5-6 are -2.8 eV times the dot product of the two p directions, and the coupling 1-6 is tau_pi (a . b) + (tau_sigma - tau_pi)(a . u)(b . u), with b the p direction on site 6, u = (1, 0, 0), tau_pi = -2.0 exp[-1.4 (r - 1.54)] eV and tau_sigma = 3.2 exp[-1.2 (r - 1.54)] eV.
For each pair (U, beta_s), take the lowest singlet state of the six electrons, exactly, at every s; its transition state on a path is the point of highest energy for 0 <= s <= 1, and define the rise of the source-paper frontier mixing angle as the transition-state value minus the value at s = 0, both obtained from the highest occupied natural orbital of that singlet using the source construction. Fit U and beta_s so that the rise is 28.3 degrees on the conrotatory path and 6.1 degrees on the disrotatory path at the same time; exactly one such pair exists with U between 3 and 8 eV and beta_s between -2.4 and -1.2 eV. At that pair and at the conrotatory transition state, take the real unrestricted Hartree-Fock determinant of lowest energy with three spin-up and three spin-down electrons, with energy sum_ij h_ij (P_up + P_down)_ij + U sum_i (P_up)_ii (P_down)_ii; define n_H as the third largest eigenvalue of P_up + P_down and use the source-defined Yamaguchi spin-projection mapping to obtain the diradical index and corresponding mixing angle. Give that spin-projected mixing angle in degrees as the final answer, to five significant figures.
In your reasoning also give the fitted U and beta_s in eV; the progress s and the rise of the singlet energy from s = 0 at the conrotatory transition state; the correlated Theta at s = 0 and at that transition state, and W_ionic there; the occupation n_H of the unrestricted solution and y there; the progress s of the disrotatory transition state of the fitted model and the spin-projected angle at it; and how much higher the conrotatory energy rise is than the disrotatory one. For context, also give the angle Theta implied by the CASSCF(12,12) HOMO occupation that Zhou, Kukier, Gordiy, Hoffmann, Seeman and Houk (2024) report for the transition state of the forbidden disrotatory ring opening of cyclobutene, and the amount by which they find the forbidden conrotatory 1,3-cyclohexadiene and hexatriene path to lie above the allowed disrotatory one.
Output Format
Scientific reasoning wrapped in <reasoning>...</reasoning> tags. Include enough intermediate calculations to justify the deterministic computation, without turning the response into a general pipeline summary. A single final numeric answer wrapped in <final_answer>...</final_answer> tags.

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

01_pi_hamiltonian

Goal
----
Step 01: Pi Hamiltonian along a conrotatory or disrotatory ring closure. One-electron pi Hamiltonian of a linear polyene closing to a ring along a symmetry-constrained electrocyclic path.

```python
def pi_hamiltonian(n_sites: int, s: float, mode: str, params: dict) -> "np.ndarray":
    '''Tight-binding pi Hamiltonian of an n-site polyene at progress s along a conrotatory or disrotatory ring closure.

    Parameters
    ----------
    n_sites : int
        Number of carbon sites in the open chain, even integer >= 4.
    s : float
        Reaction progress, finite, 0 <= s <= 1 (0 open chain, 1 ring closed).
    mode : str
        'con' for conrotatory or 'dis' for disrotatory closure.
    params : dict
        'beta_double', 'beta_single' : resonance integrals in eV of the formal double and single bonds;
        'r_open', 'r_closed' : distance in angstrom between the terminal sites at s = 0 and s = 1;
        'tau_pi', 'zeta_pi' : prefactor in eV and decay constant in 1/angstrom of the pi-type terminal coupling;
        'tau_sigma', 'zeta_sigma' : prefactor in eV and decay constant in 1/angstrom of the sigma-type terminal coupling.

    Returns
    -------
    h : np.ndarray
        Real symmetric (n_sites, n_sites) matrix in eV with zero diagonal. Sites 0, ..., n_sites - 1 follow the chain; the
        bond between sites i and i + 1 has resonance integral beta_double for even i and beta_single for odd i. Interior p
        orbitals point along +z. The terminal sites lie on the x axis, site 0 at negative x and site n_sites - 1 at positive
        x, at distance r = r_open - (r_open - r_closed) s. With phi = (pi/2) s, the p orbital of site 0 has direction
        a = (sin phi, 0, cos phi), tilting towards site n_sites - 1; the p orbital b of site n_sites - 1 is the image of a
        under the conserved symmetry element, the twofold rotation about the y axis for 'con' and the reflection through
        the plane x = 0 for 'dis', taken up to an overall sign. The coupling of each terminal site with its chain neighbour
        is the bond's resonance integral times the dot product of the two p directions. The coupling between sites 0 and
        n_sites - 1 is tau_pi(r) (a . b) + (tau_sigma(r) - tau_pi(r)) (a . u)(b . u), with u = (1, 0, 0) and
        tau_pi(r) = tau_pi exp(-zeta_pi (r - r_closed)), tau_sigma(r) = tau_sigma exp(-zeta_sigma (r - r_closed)).
        Results are compared only through quantities that do not depend on the sign convention of the basis orbitals.

    Raises
    ------
    ValueError
        If n_sites is not an even integer >= 4, s is not finite or outside [0, 1], or mode is not 'con' or 'dis'.
    '''
    return h
```

### Step 2

02_singlet_ground_state

Goal
----
Step 02: Lowest singlet and natural occupations of a half-filled Hubbard model. Lowest singlet state of a Hubbard-type pi-electron model at half filling, with its natural orbital occupations.

```python
def singlet_ground_state(h: "np.ndarray", U: float) -> "np.ndarray":
    '''Energy and natural orbital occupations of the lowest singlet of a half-filled Hubbard model.

    Parameters
    ----------
    h : np.ndarray
        Real symmetric (n, n) one-electron matrix in eV, n even with 2 <= n <= 10, finite.
    U : float
        On-site repulsion in eV, finite and >= 0.

    Returns
    -------
    res : np.ndarray
        Real array of length n + 1: res[0] is the energy in eV of the lowest-energy eigenstate with total spin S = 0 of
        H = sum_{i,j,sigma} h_ij c+_{i sigma} c_{j sigma} + U sum_i n_{i up} n_{i down} with n electrons (n/2 of each
        spin), and res[1:] are the eigenvalues of its spin-summed one-particle density matrix
        gamma_ij = sum_sigma <c+_{i sigma} c_{j sigma}> in descending order. Inputs are such that this singlet is
        nondegenerate.

    Raises
    ------
    ValueError
        If h is not a finite symmetric square matrix of even size between 2 and 10, U is not finite or U < 0, or no
        singlet is found among the six lowest states with n/2 electrons of each spin.
    '''
    return res
```

### Step 3

03_frontier_mixing_descriptors

Goal
----
Step 03: Frontier orbital mixing angle and ionic and covalent weights. Orbital mixing angle of the frontier bond and the ionic and covalent weights it implies.

```python
def frontier_mixing_descriptors(h: "np.ndarray", U: float) -> "np.ndarray":
    """Frontier orbital mixing angle and ionic/covalent weights of the lowest singlet of a half-filled Hubbard model.

    Parameters
    ----------
    h : np.ndarray
        Real symmetric (n, n) one-electron matrix in eV, as in singlet_ground_state.
    U : float
        On-site repulsion in eV, finite and >= 0.

    Returns
    -------
    d : np.ndarray
        Real array (Theta, W_ionic, W_cov), in that order. Theta is in degrees and is obtained from the highest occupied
        natural-orbital occupation of the lowest singlet using the source paper's frontier-bond construction; W_ionic
        and W_cov are the corresponding source-defined ionic and covalent weights.

    Raises
    ------
    ValueError
        If h or U violates the conditions of singlet_ground_state.
    """
    return d
```

### Step 4

04_path_transition_state

Goal
----
Step 04: Transition state and rise of the mixing angle along a constrained path. Highest point of the correlated pi energy along a symmetry-constrained electrocyclic path and the change of the frontier mixing angle.

```python
def path_transition_state(n_sites: int, mode: str, U: float, params: dict) -> "np.ndarray":
    '''Location, energy rise and frontier mixing angles of the energy maximum along a constrained ring-closure path.

    Parameters
    ----------
    n_sites : int
        Number of pi sites, even integer >= 4 and <= 10.
    mode : str
        'con' or 'dis', as in pi_hamiltonian.
    U : float
        On-site repulsion in eV, finite and > 0.
    params : dict
        Model parameters as in pi_hamiltonian.

    Returns
    -------
    ts : np.ndarray
        Real array (s_ts, dE, Theta_0, Theta_ts, dTheta). With E(s) the lowest singlet energy of
        pi_hamiltonian(n_sites, s, mode, params) and U (singlet_ground_state), s_ts is the progress in (0, 1) at which E
        attains its maximum over [0, 1], dE = E(s_ts) - E(0) in eV, Theta_0 and Theta_ts are the frontier mixing angles in
        degrees (frontier_mixing_descriptors) at s = 0 and s = s_ts, and dTheta = Theta_ts - Theta_0. Inputs are such that
        the maximum over [0, 1] is attained at a single interior point, where E is smooth, and any other local maximum
        lies at least 0.05 eV lower. s_ts is returned within 1e-8 and the angles within 1e-6 degrees.

    Raises
    ------
    ValueError
        If n_sites or mode violates the conditions of pi_hamiltonian, n_sites > 10, U is not finite or U <= 0, or the
        maximum of E over [0, 1] lies at s = 0 or s = 1.
    '''
    return ts
```

### Step 5

05_yamaguchi_mixing_angle

Goal
----
Step 05: Yamaguchi spin-projected mixing angle from unrestricted Hartree-Fock. Spin-projected unrestricted Hartree-Fock estimate of diradical character and the mixing angle it implies.

```python
def yamaguchi_mixing_angle(h: "np.ndarray", U: float) -> "np.ndarray":
    """Lowest unrestricted Hartree-Fock solution of a half-filled Hubbard model and its source-defined spin-projected mixing descriptors.

    Parameters
    ----------
    h : np.ndarray
        Real symmetric (n, n) one-electron matrix in eV, as in singlet_ground_state.
    U : float
        On-site repulsion in eV, finite and >= 0.

    Returns
    -------
    y : np.ndarray
        Real array (E_uhf, n_H, y_diradical, Theta_puhf), in that order. E_uhf is in eV for the lowest real determinant
        with n/2 spin-up and n/2 spin-down electrons under the energy functional specified in the problem statement;
        n_H is the (n/2)-th largest eigenvalue of P_up + P_down for that determinant; y_diradical and Theta_puhf
        (degrees) are obtained from n_H using the source-defined Yamaguchi spin-projection mapping. Inputs are such that
        all determinants of lowest energy share the same n_H.

    Raises
    ------
    ValueError
        If h or U violates the conditions of singlet_ground_state.
    """
    return y
```

### Step 6

06_calibrate_model

Goal
----
Step 06: Two-parameter calibration on the rises of the mixing angle. Two-parameter calibration of a correlated pi model against the mixing-angle changes of a forbidden and an allowed pathway.

```python
def calibrate_model(n_sites: int, target_con: float, target_dis: float, params: dict, U_bounds: tuple,
                    beta_bounds: tuple) -> "np.ndarray":
    '''On-site repulsion and single-bond resonance integral that reproduce the mixing-angle changes of both pathways.

    Parameters
    ----------
    n_sites : int
        Number of pi sites, as in path_transition_state.
    target_con : float
        Target rise of the frontier mixing angle in degrees on the conrotatory path, finite.
    target_dis : float
        Target rise of the frontier mixing angle in degrees on the disrotatory path, finite.
    params : dict
        Model parameters as in pi_hamiltonian; the value stored under 'beta_single' is ignored and replaced by the
        trial value.
    U_bounds : tuple
        (U_low, U_high) in eV, finite with 0 < U_low < U_high.
    beta_bounds : tuple
        (beta_low, beta_high) in eV for the single-bond resonance integral, finite with beta_low < beta_high < 0.

    Returns
    -------
    fit : np.ndarray
        Real array (U_star, beta_single_star). At these values dTheta of path_transition_state(n_sites, 'con', U_star,
        params with beta_single = beta_single_star) equals target_con and dTheta of the same call with 'dis' equals
        target_dis. Inputs are such that exactly one such pair exists in the open rectangle U_bounds x beta_bounds, the
        conditions of path_transition_state hold there, and the pair is returned within 1e-6 of its exact value.

    Raises
    ------
    ValueError
        If target_con or target_dis is not finite, the bounds are not finite, U_low <= 0, U_high <= U_low,
        beta_high <= beta_low, beta_high >= 0, or the search leaves the rectangle without converging.
    '''
    return fit
```

### Step 7

07_calibrated_puhf_angle

Goal
----
Step 07: Spin-projected angle at the calibrated transition state (orchestrator). Mean-field diradical angle at the forbidden transition state of a pi model calibrated on two reported angle changes.

```python
def calibrated_puhf_angle(n_sites: int, mode: str, target_con: float, target_dis: float, params: dict,
                          U_bounds: tuple, beta_bounds: tuple) -> float:
    '''Spin-projected mean-field mixing angle at the transition state of a model fitted to two angle changes.

    Parameters
    ----------
    n_sites : int
        Number of pi sites, as in path_transition_state.
    mode : str
        'con' or 'dis', the path whose transition state is used.
    target_con : float
        Target rise of the frontier mixing angle in degrees on the conrotatory path, as in calibrate_model.
    target_dis : float
        Target rise of the frontier mixing angle in degrees on the disrotatory path, as in calibrate_model.
    params : dict
        Model parameters as in pi_hamiltonian; 'beta_single' is replaced by the fitted value.
    U_bounds : tuple
        (U_low, U_high) in eV, as in calibrate_model.
    beta_bounds : tuple
        (beta_low, beta_high) in eV, as in calibrate_model.

    Returns
    -------
    theta_puhf : float
        With (U_star, beta_star) = calibrate_model(n_sites, target_con, target_dis, params, U_bounds, beta_bounds),
        p the parameters with beta_single = beta_star, and s_ts the first entry of
        path_transition_state(n_sites, mode, U_star, p), the value Theta_puhf in degrees of
        yamaguchi_mixing_angle(pi_hamiltonian(n_sites, s_ts, mode, p), U_star), returned within 1e-4 degrees.

    Raises
    ------
    ValueError
        If mode is not 'con' or 'dis', or any input violates the conditions of calibrate_model.
    '''
    return theta_puhf
```
