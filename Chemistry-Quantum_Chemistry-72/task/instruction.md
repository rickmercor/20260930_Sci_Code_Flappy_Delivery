# Chemistry-Quantum_Chemistry-72

## Background

Weakly bound complexes such as a rare-gas atom attached to a hydrogen halide have energy levels that are measured to high precision by microwave and infrared spectroscopy, and those levels are among the most sensitive probes of the anisotropic intermolecular potential. Turning measured levels into a potential, or a potential into predicted levels, needs a bound-state method that treats the wide-amplitude motion along the intermolecular distance accurately and that can be run many times inside a fit.

Coupled-channel methods expand the wavefunction in basis functions for every coordinate except the intermolecular distance and solve the resulting set of coupled second-order radial equations on a grid. For an atom and a rigid linear rotor each rotor state defines a channel, the rotor energies and centrifugal terms sit on the diagonal of the coupling matrix, and the anisotropic terms of the potential couple the channels. Channels whose rotor energy plus the local diagonal potential lies above the total energy are locally closed and their solutions grow or decay exponentially, so the equations are propagated in a form that stays numerically stable, most often as a log-derivative matrix. Solutions that satisfy the boundary condition at short range and at long range can be joined smoothly only at the energies of bound states, which reduces bound-state location to a one-dimensional search for zeros of functions of energy.

That search has two practical difficulties. A multichannel spectrum can contain many states in a small energy range, including near-degenerate pairs belonging to different channels, and the functions whose zeros mark the states have poles as well as zeros, so a root finder needs both a way of counting the states in a window and a way of deciding which function to follow for each one. Oscillation theory for matrix Sturm-Liouville problems supplies the counting: the number of singular points of the wavefunction matrix across the whole range equals the number of states below the trial energy. The same questions arise when a state is sought as a function of an external field or a potential parameter at fixed energy, as in fitting a potential to one observed level.

## Problem

Coupled-channel bound-state calculations propagate log-derivative matrices outwards from short range and inwards from long range to a matching distance, and locate a state where one eigenvalue of the matching matrix passes through zero. A recent treatment identifies the eigenvalue that belongs to a given state at every energy where it exists, using the split of the multichannel node count into its propagation and matching parts, and uses the same machinery to locate states as a function of a parameter other than energy. Here that parameter is an overall scaling factor of the interaction in a model complex of an atom with a linear molecule: the factor is refitted so that one level sits at an observed energy, and the output is the energy the refitted model predicts for the level just below it.

Treat the complex at total angular momentum zero as an atom and a rigid linear rotor with reduced mass 19.2 amu and rotational constant 10.4 cm^-1, keeping the rotor states j = 0 to 6. The interaction is lam times the sum over L = 0, 1, 2 of V_L(R) P_L(cos theta), where R is the distance from the centre of mass of the molecule to the atom, theta is the angle between the molecular axis and that distance vector, and V_L(R) = 180 cm^-1 [a_L (3.8 angstrom / R)^12 - 2 b_L (3.8 angstrom / R)^6] with (a_0, a_1, a_2) = (1, 0.25, 0.35) and (b_0, b_1, b_2) = (1, 0.15, 0.30). Use hbar^2 / (2 amu angstrom^2) = 16.8576292 cm^-1, measure energies from the energy of the separated atom and j = 0 rotor, and require every radial channel function to vanish at R = 2.5 angstrom and at R = 15 angstrom.

Label the bound states of this walled problem by the value of the multichannel node count immediately above each state, so the lowest state is state 1 and states of all channels are counted together. Find the smallest lam in [1, 1.4] at which state 17 lies at -30 cm^-1 and give, as the final answer, the energy of state 16 at that lam in cm^-1 to five significant figures, converged to better than 1e-4 cm^-1. Wherever a propagation is needed, start the outward log-derivative matrix at 2.5 angstrom from 1e30 times the unit matrix and the inward one at 15 angstrom from -1e30 times the unit matrix, match them at 4.0 angstrom as the outward matrix minus the inward matrix, and take the node count as the number of singular points of the wavefunction matrix met by the two propagations plus the number of negative eigenvalues of the matching matrix.

In your reasoning first state the number of coupled channels of this problem and the distance and depth of the minimum of the isotropic part of the interaction at lam = 1. Then give, at lam = 1 and -30 cm^-1, the number of singular points met by the two propagations, the number of negative matching eigenvalues, and the position (counting the eigenvalues in ascending order from 1) and value in inverse angstrom of the matching eigenvalue that belongs to state 16; the energies of states 15, 16 and 17 at lam = 1; the fitted lam and the energies of states 15 and 18 at it; and the lam at which state 16 instead lies at -30 cm^-1 together with the energy of state 17 there. Also say what the node count at lam = 1 and -30 cm^-1 becomes if the two large starting matrices are given the opposite signs. 

Output Format Requirements:
Scientific reasoning wrapped in <reasoning>...</reasoning> tags. Include enough intermediate calculations to justify the deterministic computation, without turning the response into a general pipeline summary. A single final numeric answer wrapped in <final_answer>...</final_answer> tags.

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

01_channel_matrix

Goal
----
Step 01: Coupled-channel matrix of an atom plus rigid rotor for given J and parity. Coupled-channel interaction matrix of an atom and a rigid linear rotor for a given total angular momentum and parity.

```python
def channel_matrix(R: float, params: dict) -> "np.ndarray":
    '''Coupled-channel matrix W(R) of an atom plus rigid linear rotor complex, in inverse square angstrom.

    Parameters
    ----------
    R : float
        Intermolecular distance in angstrom, finite and > 0.
    params : dict
        Model parameters with keys
        'mu'     : reduced mass in amu (> 0);
        'B'      : rotational constant in cm^-1 (>= 0);
        'eps'    : well-depth scale in cm^-1 (> 0);
        'Rm'     : distance scale in angstrom (> 0);
        'a'      : sequence of three floats (a_0, a_1, a_2);
        'b'      : sequence of three floats (b_0, b_1, b_2);
        'jmax'   : largest rotor quantum number, integer >= 0;
        'J'      : total angular momentum, integer >= 0;
        'parity' : total parity, +1 or -1;
        'scale'  : dimensionless factor lambda multiplying the whole interaction potential (> 0).

    Returns
    -------
    W : np.ndarray
        Real symmetric N x N array. The channels are all space-fixed coupled states |(j l) J M> built from rotor
        functions of the molecular axis with j = 0, ..., jmax and partial waves of the intermolecular vector with any l
        allowed by the triangle rule for (j, l, J), restricted to total parity (-1)^(j + l) equal to 'parity'; N is
        their number. With the interaction V(R, theta) = scale * sum_{L=0,1,2} V_L(R) P_L(cos theta), where theta is
        the angle between the molecular axis and the intermolecular vector and
        V_L(R) = eps [ a_L (Rm / R)^12 - 2 b_L (Rm / R)^6 ] in cm^-1, and with C = 16.8576292 cm^-1 = hbar^2 / (2 amu
        angstrom^2), W = (mu / C) [ H_rot + C l(l + 1) / (mu R^2) + V ] represented in that basis, where H_rot has
        eigenvalue B j(j + 1). The radial equations are d^2 psi / dR^2 = [W(R) - (mu / C) E I] psi for energy E in
        cm^-1. The order of the channels and the sign of each basis function are free: results are compared only
        through quantities that do not depend on them, such as the eigenvalues of W.

    Raises
    ------
    ValueError
        If R is not finite or <= 0, a required parameter is missing or out of range, or no channel exists for the
        requested J, parity and jmax.
    '''
    return W
```

### Step 2

02_propagate_log_derivative

Goal
----
Step 02: Johnson log-derivative propagation with node count. Log-derivative propagation of the coupled-channel equations with a multichannel node count.

```python
def propagate_log_derivative(params: dict, E: float, R_start: float, R_end: float, n_steps: int,
                             Y_start: float) -> tuple:
    '''Carry the log-derivative matrix of the atom plus rigid rotor coupled equations from R_start to R_end.

    Parameters
    ----------
    params : dict
        Model parameters as in channel_matrix (keys mu, B, eps, Rm, a, b, jmax, J, parity, scale).
    E : float
        Total energy in cm^-1, finite.
    R_start : float
        Starting distance in angstrom, finite and > 0.
    R_end : float
        Final distance in angstrom, finite, > 0 and different from R_start; it may be smaller than R_start.
    n_steps : int
        Number of Simpson panels, integer >= 1. The grid is R_k = R_start + k h, k = 0, ..., 2 n_steps, with
        h = (R_end - R_start) / (2 n_steps), so h is negative for inward propagation.
    Y_start : float
        Finite value y0; the log-derivative matrix at R_start is y0 times the unit matrix.

    Returns
    -------
    result : tuple
        (Y_end, nodes) from Johnson log-derivative propagation on the specified Simpson grid, including its midpoint
        correction. Y_end is the symmetric log-derivative matrix at R_end (np.ndarray of shape (N, N) in the channel
        basis of channel_matrix, inverse angstrom); nodes is the multichannel node count accumulated on this segment,
        returned as a Python int. Y_end is compared only through its eigenvalues.

    Raises
    ------
    ValueError
        If E, R_start, R_end or Y_start is not finite, a distance is <= 0, R_start equals R_end, or n_steps is not a
        positive integer.
    '''
    return result
```

### Step 3

03_matching_spectrum

Goal
----
Step 03: Matching-matrix spectrum and node-count split. Matching matrix of a bound-state calculation and the two parts of the multichannel node count.

```python
def matching_spectrum(params: dict, E: float, grid: dict) -> "np.ndarray":
    '''Node-count contributions and ascending eigenvalues of the log-derivative matching matrix at energy E.

    Parameters
    ----------
    params : dict
        Model parameters as in channel_matrix.
    E : float
        Trial energy in cm^-1, finite.
    grid : dict
        Keys 'R_min', 'R_match', 'R_max' (angstrom, 0 < R_min < R_match < R_max) and 'h' (target step, angstrom, > 0).
        The wavefunction vanishes at R_min and at R_max. The outward segment uses
        n_outward = max(1, round((R_match - R_min) / (2 h))) Simpson panels from R_min to R_match, starting from
        Y = 1e30 I, and the inward segment uses n_inward = max(1, round((R_max - R_match) / (2 h))) panels from R_max to
        R_match, starting from Y = -1e30 I, both with propagate_log_derivative (round is Python's round, halves to even).

    Returns
    -------
    result : np.ndarray
        Float array of length N + 2: [n_sum, n_match, e_1, ..., e_N] with N the number of channels of channel_matrix. n_sum is the node count of
        the outward segment plus that of the inward segment. Y_match is the symmetric part of Y_outward - Y_inward, where
        Y_outward and Y_inward are the log-derivative matrices at R_match from the outward and the inward segment; e_1 <= ... <= e_N are
        its eigenvalues (inverse angstrom) and n_match is the number of them below zero.

    Raises
    ------
    ValueError
        If E is not finite, the grid keys are missing, the distances are not ordered as 0 < R_min < R_match < R_max,
        or h is not finite and > 0.
    '''
    return result
```

### Step 4

04_tracked_eigenvalue

Goal
----
Step 04: Eigenvalue that belongs to a given bound state. The eigenvalue of the matching matrix that belongs to a given bound state, and the energies at which it can be followed.

```python
def tracked_eigenvalue(params: dict, m: int, E: float, grid: dict) -> "np.ndarray":
    '''Index and value of the matching-matrix eigenvalue that passes through zero at the energy of state m.

    Parameters
    ----------
    params : dict
        Model parameters as in channel_matrix.
    m : int
        Bound-state label, the node count immediately above the state; integer >= 1.
    E : float
        Trial energy in cm^-1, finite.
    grid : dict
        Propagation grid as in matching_spectrum.

    Returns
    -------
    result : np.ndarray
        Float array [i, e] identifying the matching-matrix eigenvalue that passes through zero at the energy of state
        m. The position i is counted from 1 in the ascending matching spectrum at E, and e is its value in inverse
        angstrom. At energies where that state's matching eigenvalue is unavailable, return [0.0, 0.0].

    Raises
    ------
    ValueError
        If m is not an integer >= 1 or E is not finite.
    '''
    return result
```

### Step 5

05_bound_state_energies

Goal
----
Step 05: All bound states in an energy window. All coupled-channel bound states in an energy window, converged on individual eigenvalues of the matching matrix.

```python
def bound_state_energies(params: dict, E_bottom: float, E_top: float, R_min: float, R_max: float,
                         tol: float) -> "np.ndarray":
    '''Converged energies of every bound state of the walled coupled-channel problem between E_bottom and E_top.

    Parameters
    ----------
    params : dict
        Model parameters as in channel_matrix.
    E_bottom : float
        Lower end of the energy window in cm^-1, finite, not within 1e-6 cm^-1 of a bound state.
    E_top : float
        Upper end of the energy window in cm^-1, finite, > E_bottom and not within 1e-6 cm^-1 of a bound state.
    R_min : float
        Inner wall in angstrom, finite, > 0 and < 4.0; every radial channel function vanishes there.
    R_max : float
        Outer wall in angstrom, finite and > 4.0; every radial channel function vanishes there.
    tol : float
        Required absolute accuracy of each energy in cm^-1, finite and >= 1e-9.

    Returns
    -------
    energies : np.ndarray
        One-dimensional float array in ascending order holding every eigenvalue E of the exact coupled radial
        equations d^2 psi / dR^2 = [W(R) - (mu / C) E I] psi (W from channel_matrix) with psi(R_min) = psi(R_max) = 0
        that lies in (E_bottom, E_top), each within tol of its exact value. Degenerate eigenvalues are repeated. The
        exact multichannel node count n(E), the number of eigenvalues below E, labels the states: the returned
        energies are those of states n(E_bottom) + 1, ..., n(E_top). Finite-grid results from matching_spectrum carry a
        step-size error that must be removed to reach tol. Wherever a propagation is needed the outward and inward
        matrices are matched at 4.0 angstrom, the matching distance this problem fixes, so the walls must enclose it.

    Raises
    ------
    ValueError
        If E_bottom, E_top, R_min, R_max or tol is not finite, E_top <= E_bottom, R_min <= 0, R_max <= R_min,
        tol < 1e-9, or 4.0 angstrom does not lie strictly between R_min and R_max.
    '''
    return energies
```

### Step 6

06_scale_for_level

Goal
----
Step 06: Potential scaling factor for a level at fixed energy. Potential scaling factor that places a chosen coupled-channel bound state at a fixed energy.

```python
def scale_for_level(params: dict, m: int, E: float, lam_low: float, lam_high: float, R_min: float, R_max: float,
                    tol: float) -> float:
    '''Converged interaction scaling factor at which bound state m of the walled coupled-channel problem lies at energy E.

    Parameters
    ----------
    params : dict
        Model parameters as in channel_matrix; the value stored under 'scale' is ignored and replaced by the trial factor.
    m : int
        Bound-state label, the exact node count immediately above the state; integer >= 1.
    E : float
        Energy in cm^-1 at which state m is required, finite.
    lam_low : float
        Lower end of the search interval for the scaling factor, finite and > 0.
    lam_high : float
        Upper end of the search interval, finite and > lam_low.
    R_min : float
        Inner wall in angstrom, finite, > 0 and < 4.0; every radial channel function vanishes there.
    R_max : float
        Outer wall in angstrom, finite and > 4.0; every radial channel function vanishes there.
    tol : float
        Required absolute accuracy of the scaling factor, finite and >= 1e-10.

    Returns
    -------
    lam_star : float
        With n(lam) the exact multichannel node count at energy E of the coupled radial equations for params with
        'scale' set to lam and psi(R_min) = psi(R_max) = 0 (the number of eigenvalues below E), and n(lam)
        non-decreasing on [lam_low, lam_high], lam_star is the infimum of lam in that interval with n(lam) >= m,
        returned within tol of its exact value. The interval ends are not within 1e-6 of a crossing of E by any state.
        Propagations are matched at 4.0 angstrom, the matching distance this problem fixes, so the walls must enclose it.

    Raises
    ------
    ValueError
        If m is not an integer >= 1, E, lam_low, lam_high, R_min, R_max or tol is not finite, lam_low <= 0,
        lam_high <= lam_low, R_min <= 0, R_max <= R_min, tol < 1e-10, 4.0 angstrom does not lie strictly between R_min
        and R_max, or the interval does not bracket the state, that is unless n(lam_low) < m <= n(lam_high).
    '''
    return lam_star
```

### Step 7

07_predicted_partner_energy

Goal
----
Step 07: Predicted partner level after refitting the scale (orchestrator). Energy of a partner bound state predicted after the potential has been scaled to reproduce one observed state.

```python
def predicted_partner_energy(params: dict, m_fit: int, E_obs: float, m_pred: int, lam_low: float, lam_high: float,
                             R_min: float, R_max: float) -> float:
    '''Converged energy of state m_pred after scaling the interaction so that state m_fit lies at E_obs.

    Parameters
    ----------
    params : dict
        Model parameters as in channel_matrix; 'scale' is ignored.
    m_fit : int
        Label of the fitted state, integer >= 1.
    E_obs : float
        Observed energy of the fitted state in cm^-1, finite.
    m_pred : int
        Label of the predicted state, integer >= 1 and different from m_fit.
    lam_low : float
        Lower end of the search interval for the scaling factor, finite and > 0.
    lam_high : float
        Upper end of the search interval, finite and > lam_low.
    R_min : float
        Inner wall in angstrom, finite, > 0 and < 4.0; every radial channel function vanishes there.
    R_max : float
        Outer wall in angstrom, finite and > 4.0; every radial channel function vanishes there.

    Returns
    -------
    E_pred : float
        With lam_star the exact scaling factor of scale_for_level(params, m_fit, E_obs, lam_low, lam_high, R_min, R_max,
        tol), the exact energy in cm^-1 of state m_pred (labels as in bound_state_energies) of the walled problem for
        params with 'scale' set to lam_star, returned within 5e-9 cm^-1. Propagations are matched at 4.0 angstrom,
        the matching distance this problem fixes, so the walls must enclose it.

    Raises
    ------
    ValueError
        If m_fit or m_pred is not an integer >= 1, m_pred equals m_fit, E_obs, lam_low, lam_high, R_min or R_max is not
        finite, lam_low <= 0, lam_high <= lam_low, R_min <= 0, R_max <= R_min, 4.0 angstrom does not lie strictly
        between R_min and R_max, or the interval does not bracket state m_fit at E_obs.
    '''
    return E_pred
```
