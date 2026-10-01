# Transient fold angle of a gel beam heated by a laser spot

## Background

Gels whose equilibrium swelling depends on temperature can be made to change shape with light: where light is absorbed, the gel warms and swells or deswells, and if the warming is uneven through the thickness of a slender strip, the strip bends. A small laser spot heats only a short stretch of a long strip, so the bending is concentrated there, while the rest of the strip stays at ambient temperature and its free arms simply rotate. Predicting the fold therefore comes down to two linked pieces: the transient conduction of heat through the cross-section near the spot, driven by the light absorbed there and limited by cooling at the surfaces, and the link between the depth-wise imbalance of that temperature rise and the change in slope it produces across the heated stretch.

## Problem

A slender beam of square cross-section, made of a photo-thermo-responsive hydrogel, has both of its ends left free. A laser is aimed perpendicular to the beam's upper face and switched on at a fixed point along the beam's length; the laser spot is well inside the beam's ends, and the light it delivers is absorbed as it travels down through the material, generating heat. The beam's coefficient of thermal expansion is negative (the gel deswells and contracts, rather than expanding, when it warms), so the resulting temperature field drives the beam to bend into a V shape centred on the illuminated point, with the two straight arms of the V rotating away from each other as heat accumulates. Because the cross-section is thin, heat takes a finite amount of time to diffuse across it and build up the temperature distribution responsible for the bending, so the sharpness of the V at any given moment depends on how long the laser has been on.

The beam has length $L$ and square cross-section of side $h$, with thermal conductivity $k$, specific heat capacity $c_\theta$, density $\rho_0$, and longitudinal coefficient of thermal expansion $\gamma_1$ (negative, for this deswelling gel). The laser delivers a Gaussian spot of radius $w_{\rm dim}$ and total power $P$: the intensity incident on the upper face is $I = [P/(\pi w_{\rm dim}^2)]\exp(-r^2/w_{\rm dim}^2)$, where $r$ is the distance from the centre of the spot, and a fraction $\eta_{\rm th}$ of the absorbed optical power is converted into heat; the light is attenuated through the depth of the material according to the Beer-Lambert law with attenuation coefficient $\beta_{\rm dim}$. All exterior surfaces of the beam, including the illuminated face, exchange heat with the surroundings by Newton convection with heat transfer coefficient $H$. The material and laser parameters for this batch of beams are:

| Quantity | Symbol | Value |
|---|---|---|
| Length | $L$ | $0.025~\mathrm{m}$ |
| Cross-section side | $h$ | $0.0025~\mathrm{m}$ |
| Thermal conductivity | $k$ | $0.55~\mathrm{W\,m^{-1}K^{-1}}$ |
| Specific heat capacity | $c_\theta$ | $4.0\times10^3~\mathrm{J\,kg^{-1}K^{-1}}$ |
| Density | $\rho_0$ | $1000~\mathrm{kg\,m^{-3}}$ |
| Longitudinal thermal expansion coefficient | $\gamma_1$ | $-0.015~\mathrm{K^{-1}}$ |
| Optical attenuation coefficient | $\beta_{\rm dim}$ | $220~\mathrm{m^{-1}}$ |
| Fraction of absorbed light converted to heat | $\eta_{\rm th}$ | $0.8$ |
| Heat transfer coefficient | $H$ | $150~\mathrm{W\,m^{-2}K^{-1}}$ |
| Laser spot radius | $w_{\rm dim}$ | $0.005~\mathrm{m}$ |
| Laser power over spot area, $P/(\pi w_{\rm dim}^2)$ | | $2.5\times10^4~\mathrm{W\,m^{-2}}$ |

Report, as a single number, the fold angle of the V (in radians) $3.0~\mathrm{s}$ after the laser is switched on. The fold angle is the angle through which one arm of the V is rotated relative to the other, zero for the undeformed beam. The mechanical response is quasi-static, and the rotations of the arms are small, of the order of the aspect ratio $h/L$; take each arm's rotation equal to its slope, its transverse deflection per unit length along the beam (the small-rotation approximation of linear beam kinematics), so that the fold angle is the magnitude of the jump in slope across the illuminated point. Alongside it, report the Biot number, the dimensionless optical attenuation coefficient, and the dimensionless laser radius that characterise the transverse (cross-sectional) heat problem, the dimensionless group that sets the overall strength of the thermomechanical coupling in the reduced (asymptotic) beam model, and the thermal moment of the temperature field at the requested instant, $M^*(t)=\frac{12}{h^4}\int\!\!\int\!\!\int x_2\,(T-T_0)\,\mathrm{d}A\,\mathrm{d}x_1$ in kelvin, where $T-T_0$ is the temperature rise above ambient, $x_2$ is the depth coordinate measured from the beam's mid-plane and increasing away from the illuminated face, and the integral runs over the cross-section and along the whole beam. Also state, as internal checks: what the thermally driven bending would reduce to in the hypothetical case that the absorbed light were instead distributed uniformly through the depth of the beam rather than following the stated attenuation law; and the fold angle in the long-time (fully equilibrated) limit, together with how the fold angle at $3.0~\mathrm{s}$ compares with it.

As a consistency check on the conversion from the asymptotic model to the physical angle, also report the joint rescaling of transverse deflection and thermal first moment used in the model's final scalar beam formulation. Express the rescaled deflection in terms of dimensional transverse deflection $v$ and cross-section side $h$, and the rescaled thermal moment in terms of $M^*(t)$ and the model's reference temperature rise $\Delta T$. Use the normalization of the scalar transverse equation and its associated nonlinear axial-force balance, including for this free-ended case.

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

compute_dimensionless_groups

Goal
----
Derive the four dimensionless groups and the dimensionless elapsed time that control the transverse heat problem and the thermomechanical coupling of a laser-heated photoresponsive beam.

```python
import numpy as np


def compute_dimensionless_groups(
    length: float,
    thickness: float,
    thermal_conductivity: float,
    specific_heat: float,
    density: float,
    thermal_expansion: float,
    attenuation_coefficient: float,
    heat_conversion_fraction: float,
    heat_transfer_coefficient: float,
    laser_radius: float,
    incident_intensity_prefactor: float,
    elapsed_time: float,
) -> np.ndarray:
    """Dimensionless groups and elapsed time for the beam heating problem.

    Parameters
    ----------
    length : float
        Beam length L (m), > 0.
    thickness : float
        Beam (square cross-section) side length h (m), > 0.
    thermal_conductivity : float
        Isotropic thermal conductivity k (W/m/K), > 0.
    specific_heat : float
        Specific heat capacity c_theta (J/kg/K), > 0.
    density : float
        Initial density rho0 (kg/m^3), > 0.
    thermal_expansion : float
        Longitudinal coefficient of thermal expansion gamma1 (1/K), finite
        and nonzero.
    attenuation_coefficient : float
        Dimensional optical attenuation coefficient beta_dim (1/m), > 0.
    heat_conversion_fraction : float
        Fraction eta_th of absorbed optical power converted to heat,
        0 < eta_th <= 1.
    heat_transfer_coefficient : float
        Convective heat transfer coefficient H (W/m^2/K), > 0.
    laser_radius : float
        Laser beam radius w_dim (m), > 0.
    incident_intensity_prefactor : float
        The ratio P/(pi*w_dim^2), with units W/m^2, > 0.
    elapsed_time : float
        Physical elapsed time since the laser was switched on (s), >= 0.

    Returns
    -------
    groups : np.ndarray
        Array of shape (5,): [Bi, beta, w, Gamma1, t], the Biot number, the
        dimensionless attenuation coefficient, the dimensionless laser
        radius, the thermomechanical coupling constant, and the
        dimensionless elapsed time, in that order.

    Raises
    ------
    ValueError
        If length, thickness, thermal_conductivity, specific_heat, density,
        attenuation_coefficient, heat_transfer_coefficient, laser_radius or
        incident_intensity_prefactor is not a finite number > 0; if
        thermal_expansion is not finite and nonzero; if
        heat_conversion_fraction is not in (0, 1]; or if elapsed_time is
        not a finite number >= 0.
    """
    return groups  # placeholder
```

### Step 2

solve_transverse_eigenbasis

Goal
----
Solve the transverse (cross-sectional) Robin eigenvalue problem for a beam cooled by Newton convection on both faces, for both eigenfunction parities at once, and return every eigenpair merged into a single array ordered by increasing eigenvalue.

```python
import numpy as np


def solve_transverse_eigenbasis(bi: float, n_modes: int) -> np.ndarray:
    """Merged even+odd transverse Robin-BC eigenbasis.

    Parameters
    ----------
    bi : float
        Biot number of the transverse faces, a finite number > 0.
    n_modes : int
        Number of eigenvalues to find IN EACH family, an integer >= 1
        (so 2*n_modes rows are returned in total).

    Returns
    -------
    modes : np.ndarray
        Array of shape (2*n_modes, 3). Column 0 holds the eigenvalues nu
        (strictly positive), column 1 the normalization constants C
        (strictly positive), and column 2 the parity tag (0.0 for the even
        family, 1.0 for the odd family). Rows are sorted by strictly
        increasing eigenvalue across both families combined.

    Raises
    ------
    ValueError
        If bi is not a finite number > 0, or if n_modes is not an integer
        >= 1.
    """
    return modes  # placeholder
```

### Step 3

project_heat_source_onto_modes

Goal
----
Project the laser's dimensionless intensity profile onto every transverse eigenmode returned by the eigenbasis step, using whichever physical direction is relevant to that mode's own parity.

```python
import numpy as np


def project_heat_source_onto_modes(modes: np.ndarray, beta: float, w: float) -> np.ndarray:
    """Per-mode projection of the laser's dimensionless intensity profile, branching internally by parity.

    Parameters
    ----------
    modes : np.ndarray
        Shape (N, 3), N >= 2, as returned by solve_transverse_eigenbasis (the two parity counts need not be equal):
        column 0 eigenvalue (> 0), column 1 normalization constant (> 0),
        column 2 parity tag (0.0 or 1.0 only).
    beta : float
        Dimensionless optical (depth-wise) attenuation coefficient, a
        finite number > 0.
    w : float
        Dimensionless laser radius, a finite number > 0.

    Returns
    -------
    proj : np.ndarray
        Array of shape (N,): the projection value for each row of modes,
        in the same order as modes.

    Raises
    ------
    ValueError
        If modes is not two-dimensional with exactly 3 columns and at
        least 2 rows; if any eigenvalue or normalization constant in modes
        is not finite and > 0; if any parity tag is not exactly 0.0 or
        1.0; or if beta or w is not a finite number > 0.
    """
    return proj  # placeholder
```

### Step 4

compute_cross_sectional_moment_weights

Goal
----
Compute the cross-sectional moment-arm weight for every transverse eigenmode, using whichever physical averaging is relevant to that mode's own parity.

```python
import numpy as np


def compute_cross_sectional_moment_weights(modes: np.ndarray) -> np.ndarray:
    """Per-mode cross-sectional moment-arm weight, branching internally by parity.

    Parameters
    ----------
    modes : np.ndarray
        Shape (N, 3), N >= 2, as returned by solve_transverse_eigenbasis (the two parity counts need not be equal):
        column 0 eigenvalue (> 0), column 1 normalization constant (> 0),
        column 2 parity tag (0.0 or 1.0 only).

    Returns
    -------
    weights : np.ndarray
        Array of shape (N,): the moment-arm weight for each row of
        modes, in the same order as modes.

    Raises
    ------
    ValueError
        If modes is not two-dimensional with exactly 3 columns and at
        least 2 rows; if any eigenvalue or normalization constant in modes
        is not finite and > 0; or if any parity tag is not exactly 0.0 or
        1.0.
    """
    return weights  # placeholder
```

### Step 5

compute_transient_modal_temperatures

Goal
----
Assemble the transient modal temperature amplitude for every combination of a depth (odd-parity) mode and a lateral (even-parity) mode, at a given elapsed time.

```python
import numpy as np


def compute_transient_modal_temperatures(modes: np.ndarray, proj: np.ndarray, w: float, t: float) -> np.ndarray:
    """Transient modal temperature amplitudes Theta(t) for every odd/even mode pair.

    Parameters
    ----------
    modes : np.ndarray
        Shape (N, 3), N >= 2, as returned by solve_transverse_eigenbasis, with the two parity counts not necessarily equal:
        column 0 eigenvalue (> 0), column 1 normalization constant (> 0),
        column 2 parity tag (0.0 or 1.0 only), with at least one row of
        each parity present.
    proj : np.ndarray
        Shape (N,), as returned by project_heat_source_onto_modes,
        row-aligned with modes, finite.
    w : float
        Dimensionless laser radius, a finite number > 0.
    t : float
        Dimensionless elapsed time since the laser was switched on, a
        finite number >= 0.

    Returns
    -------
    theta : np.ndarray
        Array of shape (n_odd, n_even): the transient modal temperature
        amplitude for each (odd, even) mode-pair combination, with rows
        ordered as the odd-parity rows appear in modes and columns ordered
        as the even-parity rows appear in modes.

    Raises
    ------
    ValueError
        If modes is not two-dimensional with exactly 3 columns and at
        least one row of each parity; if proj does not have one entry per
        row of modes, or contains a non-finite value; if w is not a finite
        number > 0; or if t is not a finite number >= 0.
    """
    return theta  # placeholder
```

### Step 6

assemble_thermal_moment

Goal
----
Assemble the total transient thermal moment M2(t), the first moment of the dimensionless inner temperature field about the beam's mid-plane, from the per-mode moment-arm weights and the transient modal temperature amplitudes.

```python
import numpy as np


def assemble_thermal_moment(modes: np.ndarray, weights: np.ndarray, theta: np.ndarray) -> float:
    """Assemble the total transient thermal moment M2(t) (first moment of the inner temperature field).

    Parameters
    ----------
    modes : np.ndarray
        Shape (N, 3), N >= 2, as returned by solve_transverse_eigenbasis, with
        at least one row of each parity (the two parity counts need not be
        equal): column 2 is the parity tag (0.0 or 1.0 only), used here only
        to identify which entries of weights belong to which parity.
    weights : np.ndarray
        Shape (N,), as returned by compute_cross_sectional_moment_weights,
        row-aligned with modes, finite.
    theta : np.ndarray
        Shape (n_odd, n_even), as returned by
        compute_transient_modal_temperatures, with rows ordered as the
        odd-parity rows appear in modes and columns ordered as the
        even-parity rows appear in modes, finite.

    Returns
    -------
    m : float
        The total transient thermal moment M2(t) = Integral X2 * theta
        dX2 dX3 dX1 over the cross-section and the heated region, in the
        pipeline's dimensionless scalings, with no additional rescaling.

    Raises
    ------
    ValueError
        If modes is not two-dimensional with exactly 3 columns and at
        least one row of each parity; if weights does not have exactly
        one entry per row of modes; if theta is not two-dimensional with
        shape (n_odd, n_even) matching the number of odd-parity and
        even-parity rows in modes; or if any input contains a non-finite
        value.
    """
    return m  # placeholder
```

### Step 7

compute_fold_angle

Goal
----
Orchestrator: compute the light-induced fold angle of a free-ended photoresponsive thermoelastic beam from its dimensional material, geometric and laser parameters.

```python
import numpy as np


def compute_fold_angle(
    length: float = 0.025,
    thickness: float = 0.0025,
    thermal_conductivity: float = 0.55,
    specific_heat: float = 4.0e3,
    density: float = 1000.0,
    thermal_expansion: float = -0.015,
    attenuation_coefficient: float = 220.0,
    heat_conversion_fraction: float = 0.8,
    heat_transfer_coefficient: float = 150.0,
    laser_radius: float = 0.005,
    incident_intensity_prefactor: float = 2.5e4,
    elapsed_time: float = 3.0,
    n_modes: int = 4,
) -> float:
    """Light-induced fold angle of a free-ended photoresponsive beam.

    Parameters
    ----------
    length : float
        Beam length L (m), > 0.
    thickness : float
        Beam (square cross-section) side length h (m), > 0.
    thermal_conductivity : float
        Isotropic thermal conductivity k (W/m/K), > 0.
    specific_heat : float
        Specific heat capacity c_theta (J/kg/K), > 0.
    density : float
        Initial density rho0 (kg/m^3), > 0.
    thermal_expansion : float
        Longitudinal coefficient of thermal expansion gamma1 (1/K), finite
        and nonzero (may be negative, as for a deswelling hydrogel).
    attenuation_coefficient : float
        Dimensional optical attenuation coefficient beta_dim (1/m), > 0.
    heat_conversion_fraction : float
        Fraction eta_th of absorbed optical power converted to heat,
        0 < eta_th <= 1.
    heat_transfer_coefficient : float
        Convective heat transfer coefficient H (W/m^2/K), > 0.
    laser_radius : float
        Laser beam radius w_dim (m), > 0.
    incident_intensity_prefactor : float
        The ratio P/(pi*w_dim^2), where P is the laser power (W), giving
        units of W/m^2, > 0.
    elapsed_time : float
        Physical elapsed time since the laser was switched on (s), >= 0.
    n_modes : int
        Number of odd and even transverse modes to retain (each), an
        integer >= 1.

    Returns
    -------
    phi : float
        The fold angle in radians, >= 0: the relative rotation of the two
        arms, taken as the magnitude of the jump in the physical slope
        dv/dx1 across the illuminated point (small-rotation approximation).

    Raises
    ------
    ValueError
        If any underlying step raises ValueError on its own inputs, or if
        length, thickness, thermal_conductivity, specific_heat, density,
        attenuation_coefficient, heat_transfer_coefficient, laser_radius or
        incident_intensity_prefactor is not a finite number > 0; if
        thermal_expansion is not finite and nonzero; if
        heat_conversion_fraction is not in (0, 1]; if elapsed_time is not a
        finite number >= 0; or if n_modes is not an integer >= 1.
    """
    return phi  # placeholder
```
