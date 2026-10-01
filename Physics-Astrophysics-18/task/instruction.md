# Physics-Astrophysics-18

## Background

Very-long-baseline interferometry at millimetre wavelengths now resolves emission on scales
comparable to the event horizons of the nearest supermassive black holes. That capability moved
the interpretive bottleneck: the limiting factor is no longer angular resolution but the theory
needed to connect a feature in an image to a physical process in the plasma around the hole.

Making that connection requires a model of the accreting material — where it radiates, how it
absorbs, and how it moves — together with a spacetime through which the light propagates.
General-relativistic magnetohydrodynamic simulations supply the plasma physics from first
principles, but they are expensive and awkward to steer toward a particular transient event, which
makes systematic exploration of a parameter space impractical. A parallel line of work therefore
builds accretion environments analytically, from geometry alone: emission and absorption profiles
written in closed form, together with a prescribed velocity field for the flow. Such models trade
physical self-consistency for the ability to sweep parameters rapidly and to isolate the image
signature of one structure at a time — a thickened disk, an embedded shock front, a localized
flare — which is what makes them useful for asking which image features diagnose which process.

Because these constructions are formulated for a general stationary axisymmetric metric, they are
not tied to any one background. Exact rotating solutions of general relativity carrying parameters
beyond mass and spin — geometric deformations, external fields, or modified asymptotics — offer
alternative spacetimes in which the same accretion model can be ray-traced, and comparing the
resulting images is one route to asking how sharply horizon-scale observations constrain the
geometry itself. Photon propagation in such spacetimes is generally less tractable than in Kerr:
the hidden symmetries that render the geodesic equations separable are special to particular
metrics rather than generic, and where they are absent the trajectories must be obtained by direct
numerical integration, with the radiative transfer carried along the same rays.

## Problem

Interpreting horizon-scale images of accreting black holes requires emission models flexible enough to represent transient high-energy events — flares, outward-propagating shock fronts, stream–stream collisions — without the cost of general-relativistic magnetohydrodynamic simulations, whose expense makes systematic parameter exploration impractical. Purely geometric analytic environments fill that gap, and because they are formulated for a general stationary axisymmetric metric they can be carried over to spacetimes other than Kerr. 

Consider a geometrically thick disk whose radial attenuation profile carries a smooth plateau representing a region of locally sustained radiative efficiency, a ring-like Gaussian bump representing an outward-propagating shock, and one compact localized emission region representing a flare. The same three shapes reappear in the absorption coefficient under independent weights. The disk is placed around an exact Ricci-flat rotating black hole of four-dimensional general relativity that carries, besides its mass and rotation, a third integration constant deforming the geometry and rendering it non-asymptotically flat. The accreting matter falls inward while rotating, and its three-velocity is the one measured in the local zero-angular-momentum frame. Rays are traced backward from the local sky of an observer at finite radius, and the observed specific intensity is accumulated along each ray up to the point where the ray is terminated. Compute the observed specific intensity at each of the screen points listed below and report their sum. In your reasoning, identify the covariant photon-momentum components conserved along each null geodesic by stationarity and axisymmetry.

Use the following configuration (geometrized units, `G = c = 1`, physical mass `M = 1`):

- physical mass `M = 1`, dimensionless spin `chi = J/M^2 = 0.94`, deformation parameter `BM = 0.01`
- observer position: radius `r_o = 50`, polar angle `80 degrees`, azimuth `phi_o = 0`; the spot azimuth `phi_s` is measured in the same azimuthal coordinate
- inner boundary `r_in` set to the outer horizon radius
- emission weights: `j_0 = 1.0`, `j_1 = 1.0`, `j_2 = 1.0`, `j_3 = 1.0`
- disk: `p_1 = -1.5`, `p_2 = -0.5`, `sigma_dtheta = 0.1`, `beta = 0.1`
- plateau: `r_p = 1.5`, `w_p = 3.0`, `j_p = 1.0`
- bump: `r_b = 6.0`, `sigma_br = 0.5`, `sigma_btheta = 0.15`
- spot: `r_s = 8.0`, `theta_s = pi/2`, `phi_s = 3*pi/2`, `sigma_sr = 2.0`, `sigma_stheta = pi/36`, `sigma_sphi = pi/18`; the azimuthal term enters the Gaussian exponent directly, with `sigma_sphi` unrescaled
- absorption weights: `alpha_0 = 0.02`, `alpha_1 = 1.0`, `alpha_2 = 0.5`, `alpha_3 = 0.5`, with the same shape parameters as the emission components
- flow: `V_max = 0.9`, `psi = 0.9`, `p_3 = 0.5`, `lambda = 10.0`, zero polar three-velocity
- screen points: `beta = 0`, `alpha` in `{-10, -8, -6, -4, -2, 0, 2, 4, 6, 8, 10}`, in units of `M`
- integration: terminate a ray when `r < 1.0001 r_+`, keeping the intensity accumulated up to that point, or when the ray, travelling outward, returns to `r = r_o`; relative tolerance `1e-9`, absolute tolerance `1e-11`

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

01_ml_parameters_and_horizon

Goal
----
Convert the physical mass, dimensionless spin and deformation parameter into the metric's

integration constants and the constants of the thermodynamically normalised Killing frame,

and return the outer horizon radius alongside them.

```python
import numpy as np


def ml_parameters_and_horizon(mass: float, chi: float, b: float) -> np.ndarray:
    '''Derived constants and outer horizon radius of the spindle-deformed black hole.

    Parameters
    ----------
    mass : float
        Physical mass M of the black hole.
    chi : float
        Dimensionless spin J / M^2, with |chi| <= 1.
    b : float
        Spindle deformation parameter B, in units of inverse mass.

    Returns
    -------
    result : np.ndarray
        Array of shape (11,) of native floats, in the order
        [m, a, I1, I2, P0, eps1, eps2, gamma, lam1, lam2, r_plus].

    Raises
    ------
    ValueError
        If mass is not positive, |chi| exceeds 1, b is negative, or the
        parameter set admits no horizon.
    '''
    return result
```

### Step 2

02_ml_metric_components

Goal
----
Evaluate the covariant metric components of the spindle-deformed black hole at a point, in

the frame reached by applying both mandatory coordinate transformations.

```python
import numpy as np


def ml_metric_components(r: float, x: float, params: np.ndarray, b: float) -> np.ndarray:
    '''Covariant metric components of the spindle-deformed black hole.

    Parameters
    ----------
    r : float
        Boyer-Lindquist-like radial coordinate.
    x : float
        Polar coordinate x = cos(theta), with |x| < 1.
    params : np.ndarray
        Array of shape (11,) as returned by ml_parameters_and_horizon.
    b : float
        Spindle deformation parameter B.

    Returns
    -------
    result : np.ndarray
        Array of shape (5,), the components [g_tt, g_rr, g_xx, g_phph, g_tph]
        in the doubly-transformed frame.

    Raises
    ------
    ValueError
        If r is not positive or |x| is not less than 1.
    '''
    return result
```

### Step 3

03_inverse_metric_and_hamiltonian

Goal
----
Invert the metric's t-phi block to obtain the contravariant components, and evaluate the

photon Hamiltonian for a given covariant momentum.

```python
import numpy as np


def inverse_metric_and_hamiltonian(r: float, x: float, params: np.ndarray,
                                   b: float, p_cov: np.ndarray) -> np.ndarray:
    '''Contravariant metric components and the photon Hamiltonian.

    Parameters
    ----------
    r : float
        Radial coordinate.
    x : float
        Polar coordinate x = cos(theta).
    params : np.ndarray
        Array of shape (11,) as returned by ml_parameters_and_horizon.
    b : float
        Spindle deformation parameter B.
    p_cov : np.ndarray
        Covariant momentum of shape (4,), ordered [p_t, p_r, p_x, p_phi].

    Returns
    -------
    result : np.ndarray
        Array of shape (6,): [g^tt, g^rr, g^xx, g^phph, g^tph, H].

    Raises
    ------
    ValueError
        If r is not positive, |x| is not less than 1, or p_cov does not have
        shape (4,).
    '''
    return result
```

### Step 4

04_zamo_flow_four_velocity

Goal
----
Build the four-velocity of the accreting matter from the prescribed three-velocity

components measured in the local zero-angular-momentum frame.

```python
import numpy as np


def zamo_flow_four_velocity(r: float, x: float, params: np.ndarray, b: float,
                            r_in: float, v_max: float, p3: float,
                            psi: float, lam: float) -> np.ndarray:
    '''Four-velocity of the accreting matter in the local black-hole frame.

    Parameters
    ----------
    r : float
        Radial coordinate.
    x : float
        Polar coordinate x = cos(theta).
    params : np.ndarray
        Array of shape (11,) as returned by ml_parameters_and_horizon.
    b : float
        Spindle deformation parameter B.
    r_in : float
        Inner boundary of the accretion model.
    v_max : float
        Maximum radial three-velocity of the flow.
    p3 : float
        Exponent controlling the radial acceleration of the flow.
    psi : float
        Overall rotational speed of the flow.
    lam : float
        Suppression scale for the azimuthal motion in the inner region.

    Returns
    -------
    result : np.ndarray
        Array of shape (4,), the contravariant four-velocity [u^t, u^r, u^x, u^phi].

    Raises
    ------
    ValueError
        If r is not positive, |x| is not less than 1, or the prescribed
        three-velocity is not subluminal.
    '''
    return result
```

### Step 5

05_emission_and_absorption

Goal
----
Evaluate the total emission and absorption coefficients of the multi-component accretion

environment at a point, from the disk, plateau, bump and localized-spot laws under their

independent weights.

```python
import numpy as np


def emission_and_absorption(r: float, x: float, phi: float, r_in: float,
                            emis_params: dict, absorb_params: dict) -> np.ndarray:
    '''Total emission and absorption coefficients of the accretion environment.

    Parameters
    ----------
    r : float
        Radial coordinate.
    x : float
        Polar coordinate x = cos(theta).
    phi : float
        Azimuthal coordinate, accumulated along the ray and not wrapped.
    r_in : float
        Inner boundary of the accretion model.
    emis_params : dict
        Emission parameters; see the step background for the key list.
    absorb_params : dict
        Absorption parameters with keys alpha0, alpha1, alpha2, alpha3.

    Returns
    -------
    result : np.ndarray
        Array of shape (2,): [j_nu, alpha_nu], as native floats.

    Raises
    ------
    ValueError
        If |x| exceeds 1, r_in is not positive, any plateau, disk, bump or
        spot width is not positive, a required key is missing from
        emis_params or absorb_params, or the effective radius is not positive.
    '''
    return result
```

### Step 6

06_screen_to_initial_conditions

Goal
----
Convert a point on the observer's screen into the position and covariant momentum of the

backward-traced ray that reaches it.

```python
import numpy as np


def screen_to_initial_conditions(alpha: float, beta: float, params: np.ndarray,
                                 b: float, r_o: float, inclination: float) -> np.ndarray:
    '''Initial conditions of a backward-traced ray from a point on the observer screen.

    Parameters
    ----------
    alpha : float
        Horizontal screen coordinate, in units of mass.
    beta : float
        Vertical screen coordinate, in units of mass.
    params : np.ndarray
        Array of shape (11,) as returned by ml_parameters_and_horizon.
    b : float
        Spindle deformation parameter B.
    r_o : float
        Observer radius.
    inclination : float
        Observer polar angle in radians.

    Returns
    -------
    result : np.ndarray
        Array of shape (8,): [t, r, x, phi, p_t, p_r, p_x, p_phi].

    Raises
    ------
    ValueError
        If r_o is not positive, or inclination is not strictly between 0 and
        pi (the observer frame is singular on the axis).
    '''
    return result
```

### Step 7

07_transfer_rhs

Goal
----
Evaluate the affine-parameter derivative of the combined geodesic and radiative-transfer

state at a point along a ray.

```python
import numpy as np


def transfer_rhs(tau: float, state: np.ndarray, params: np.ndarray, b: float,
                 r_in: float, emis_params: dict, absorb_params: dict,
                 flow_params: dict) -> np.ndarray:
    '''Right-hand side of the coupled geodesic and radiative-transfer system.

    Parameters
    ----------
    tau : float
        Affine parameter; the system is autonomous, so this is unused.
    state : np.ndarray
        Array of shape (10,):
        [t, r, x, phi, p_t, p_r, p_x, p_phi, optical_depth, intensity],
        where optical_depth is the optical depth accumulated along the ray
        from the observer to the current point and intensity is the observed
        specific intensity contributed by the ray between the observer and
        the current point.
    params : np.ndarray
        Array of shape (11,) as returned by ml_parameters_and_horizon.
    b : float
        Spindle deformation parameter B.
    r_in : float
        Inner boundary of the accretion model.
    emis_params : dict
        Emission parameters; see step 05.
    absorb_params : dict
        Absorption parameters with keys alpha0, alpha1, alpha2, alpha3.
    flow_params : dict
        Flow parameters with keys v_max, p3, psi, lam.

    Returns
    -------
    result : np.ndarray
        Array of shape (10,), the affine-parameter derivative of the state.

    Raises
    ------
    ValueError
        If state does not have shape (10,), its radial coordinate is not
        positive, its polar coordinate has |x| >= 1, r_in is not positive, or
        flow_params lacks a required key.
    '''
    return result
```

### Step 8

08_line_summed_intensity

Goal
----
Trace every listed screen point through the full pipeline, apply the horizon-capture and

escape rules, and return the summed observed specific intensity.

```python
from typing import Sequence
import numpy as np
from scipy.integrate import solve_ivp


def line_summed_intensity(alphas: Sequence[float], beta: float, mass: float, chi: float, b: float,
                          r_o: float, inclination: float, emis_params: dict,
                          absorb_params: dict, flow_params: dict,
                          rtol: float, atol: float) -> float:
    '''Summed observed specific intensity over a row of screen points.

    Parameters
    ----------
    alphas : Sequence[float]
        Horizontal screen coordinates of the pixels, in units of mass.
    beta : float
        Vertical screen coordinate shared by all the pixels.
    mass : float
        Physical mass M of the black hole.
    chi : float
        Dimensionless spin J / M^2.
    b : float
        Spindle deformation parameter B.
    r_o : float
        Observer radius.
    inclination : float
        Observer polar angle in radians.
    emis_params : dict
        Emission parameters; see step 05.
    absorb_params : dict
        Absorption parameters with keys alpha0, alpha1, alpha2, alpha3.
    flow_params : dict
        Flow parameters with keys v_max, p3, psi, lam.
    rtol : float
        Relative tolerance of the integrator.
    atol : float
        Absolute tolerance of the integrator.

    Returns
    -------
    result : float
        The sum of the observed specific intensity over the pixels, as a
        native Python float.

    Raises
    ------
    ValueError
        If alphas is empty, or either tolerance is not positive.
    '''
    return result
```
