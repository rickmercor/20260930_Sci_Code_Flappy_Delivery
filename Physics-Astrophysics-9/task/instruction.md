# Physics-Astrophysics-9

## Background

Solar-system bodies too faint to detect in any single exposure can still be found by summing many exposures along the path such a body would follow, so that its signal accumulates while the noise averages down. When the exposures span days rather than hours, both the body and the observer move on curved paths and the apparent motion is strongly nonlinear, so the combination has to be organised around trial orbits rather than straight-line motions on the detector.

The task below concerns how completely one trial orbit of such a search gathers the signal of a faint body, and what that means for the faintest body the search can flag.

## Problem

I am searching archival Hubble ACS exposures of an ecliptic field for trans-Neptunian objects too faint to detect in any single exposure. For each initial condition on my search grid, every pixel of every exposure is registered under that initial condition onto a canvas of barycentric ecliptic directions at a reference epoch t_ref, and the registered exposures are stacked as an inverse-variance-weighted sum of PSF-matched-filtered images; a canvas maximum that reaches the candidate threshold is flagged.

One grid initial condition asserts, at t_ref, barycentric distance r = 32.006 au, radial velocity v_r = -0.956 km/s, speed perpendicular to the radius vector v_Ω = 5.6565 km/s, orbital inclination i = 9.194° to the J2000 ecliptic, and κ = +1 (ecliptic latitude increasing). The table lists ten exposures of a faint object I have since confirmed: mid-exposure time t - t_ref (TDB days), the telescope's barycentric ICRF position (au), the object's astrometric right ascension and declination (light-time retarded, without aberration or light deflection), and the exposure's PSF-fit flux uncertainty σ_k (e⁻/s). Take two-body motion about the Sun fixed at the barycentre with GM = k² (k = 0.01720209895 au^(3/2)/day), c = 173.1446326846693 au/day, 1 au = 149597870.7 km, an ecliptic obtained by rotating the ICRF frame about its x axis through the obliquity 23.4392911°, and a canvas PSF that is a circular Gaussian of standard deviation 0.042 arcsec.

Now suppose an object moving exactly like mine had the constant flux of the fainter of the two objects newly discovered by the published full-15-day-baseline shift-and-stack search of the 2003 HST ACS Kuiper-belt field, and adopt the significance threshold at which that search flagged candidate peaks in its grid stacks. By how many magnitudes could such an object be fainter still while the expected (noise-free) significance at the maximum of this initial condition's stack still reaches that threshold? Report the answer in magnitudes to ten significant figures.

| k | t - t_ref (d) | x (au) | y (au) | z (au) | RA (deg) | Dec (deg) | σ_k (e⁻/s) |
|---|---|---|---|---|---|---|---|
| 1 | -7.3518 | -0.532560188 | 0.759515749 | 0.329298322 | 218.43041933 | -12.04702376 | 0.029 |
| 2 | -5.9046 | -0.553041423 | 0.747239391 | 0.323958665 | 218.44465668 | -12.04793270 | 0.033 |
| 3 | -4.6189 | -0.571049125 | 0.735867417 | 0.319053923 | 218.45644418 | -12.04851262 | 0.028 |
| 4 | -3.0772 | -0.592159359 | 0.721842746 | 0.312948320 | 218.46949490 | -12.04876521 | 0.034 |
| 5 | -1.4653 | -0.613876126 | 0.706713902 | 0.306388170 | 218.48209218 | -12.04872461 | 0.030 |
| 6 | 0.5207 | -0.639985759 | 0.687238631 | 0.297942069 | 218.49570749 | -12.04804181 | 0.029 |
| 7 | 1.9631 | -0.658489175 | 0.672538484 | 0.291566922 | 218.50429262 | -12.04711698 | 0.033 |
| 8 | 3.7094 | -0.680427115 | 0.654311486 | 0.283678731 | 218.51355724 | -12.04566106 | 0.031 |
| 9 | 5.2418 | -0.699178130 | 0.637757483 | 0.276514388 | 218.52032553 | -12.04393920 | 0.036 |
| 10 | 7.1852 | -0.722214969 | 0.616087644 | 0.267111801 | 218.52710154 | -12.04113391 | 0.030 |

In <reasoning>, the scalars I need you to state are: the candidate threshold you adopted and the separate post-refinement retention threshold reported by that search, identifying the stage to which each applies; the reported central fluxes of both new discoveries, identifying which is fainter and adopting its central flux; the light-travel time for exposure 1; the barycentric ecliptic longitude and latitude at t_ref, in degrees to six decimal places, at which exposure 1 registers, the same for exposure 10, and the same for the stack maximum; the fraction of the perfectly registered significance that remains at that maximum; the largest misregistration among the ten exposures; and how the ten misregistrations are arranged, as a pattern with exposure time and its rate. Those are the derived quantities that determine the final number, so stating them is what the output requirements below call for; what those requirements exclude is restating the supplied table.

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

01_solve_universal_anomaly

Goal
----
Solve the universal Kepler equation for the anomaly a two-body orbit of any conic type reaches a given time after its epoch.

```python
def solve_universal_anomaly(r0: float, vr0: float, alpha: float, mu: float, dt: float) -> float:
    """Return the universal anomaly reached a time ``dt`` after an epoch.

    At the epoch the body is at distance ``r0`` from the attracting centre
    with radial velocity ``vr0``; ``alpha = 2 / r0 - v0**2 / mu`` is the
    reciprocal semi-major axis (positive for ellipses, zero for parabolas,
    negative for hyperbolas) and ``mu`` is the gravitational parameter, all
    in one consistent unit system. The universal anomaly ``chi`` is the
    variable for which the Lagrange coefficients at ``epoch + dt`` are
    ``f = 1 - chi**2 * C(alpha * chi**2) / r0`` and
    ``g = dt - chi**3 * S(alpha * chi**2) / sqrt(mu)``, where ``C`` and ``S``
    are the Stumpff functions. It vanishes at ``dt = 0`` and increases
    monotonically with ``dt``. Return it with relative accuracy ``1e-13``.

    Parameters
    ----------
    r0 : float
        Positive distance at the epoch.
    vr0 : float
        Radial velocity at the epoch.
    alpha : float
        Reciprocal semi-major axis; at most ``2 / r0 - vr0**2 / mu`` so that
        the speed perpendicular to the radius vector is real.
    mu : float
        Positive gravitational parameter.
    dt : float
        Time from the epoch; negative values run backwards.

    Returns
    -------
    float
        The universal anomaly ``chi``.

    Raises
    ------
    ValueError
        If any argument is not a finite real number (booleans are
        rejected), if ``r0`` or ``mu`` is not positive, or if ``alpha``
        exceeds ``2 / r0 - vr0**2 / mu``.
    """
    return 0.0
```

### Step 2

02_propagate_two_body_state

Goal
----
Advance a position and velocity along its two-body orbit by a given time with universal-variable Lagrange coefficients.

```python
def propagate_two_body_state(
    state: "np.ndarray",
    mu: float,
    dt: float,
    anomaly_fn: "Callable[..., float]",
) -> "np.ndarray":
    """Return the two-body state reached a time ``dt`` after an epoch.

    ``state = (x, y, z, vx, vy, vz)`` is the position and velocity relative
    to an attracting centre of gravitational parameter ``mu`` at the epoch,
    in one consistent unit system. Advance it along its Keplerian orbit with
    the universal-variable Lagrange coefficients ``f`` and ``g`` and their
    time derivatives. The universal anomaly is
    ``anomaly_fn(r0, vr0, alpha, mu, dt)``, which follows the contract of
    ``solve_universal_anomaly`` (distance ``r0``, radial velocity ``vr0`` and
    reciprocal semi-major axis ``alpha = 2 / r0 - v0**2 / mu`` at the epoch).
    Return the new state with relative accuracy ``1e-12``.

    Parameters
    ----------
    state : "np.ndarray"
        Six finite numbers; the position must be nonzero.
    mu : float
        Positive gravitational parameter.
    dt : float
        Time from the epoch; negative values run backwards.
    anomaly_fn : callable
        Universal-anomaly solver with the signature of
        ``solve_universal_anomaly``.

    Returns
    -------
    np.ndarray
        Float array of shape ``(6,)``: position then velocity after ``dt``.

    Raises
    ------
    ValueError
        If ``state`` is not six finite numbers or has a zero position,
        ``mu`` is not a finite positive number, ``dt`` is not a finite real
        number, ``anomaly_fn`` is not callable, or ``anomaly_fn`` returns
        anything other than a finite real number.
    """
    return new_state
```

### Step 3

03_evaluate_radial_track

Goal
----
Evaluate the barycentric distance and its rate of change at an arbitrary epoch for every two-body orbit that shares an asserted distance, radial velocity and tangential speed at the reference epoch.

```python
def evaluate_radial_track(
    r_ref: float,
    vr_ref: float,
    vt_ref: float,
    mu: float,
    dt: float,
    anomaly_fn: "Callable[..., float]",
) -> "np.ndarray":
    """Return the distance and distance rate a time ``dt`` after the reference epoch.

    Every two-body orbit that, at the reference epoch, lies at distance
    ``r_ref`` from the attracting centre (gravitational parameter ``mu``) with
    radial velocity ``vr_ref`` and speed ``vt_ref`` perpendicular to its
    radius vector shares one distance history ``r(t)``, whatever the
    orientation of the orbit. Return ``r`` and ``dr/dt`` at
    ``reference epoch + dt``, with relative accuracy ``1e-12``. The universal
    anomaly is ``anomaly_fn(r0, vr0, alpha, mu, dt)``, which follows the
    contract of ``solve_universal_anomaly`` (``alpha = 2 / r0 - v0**2 / mu``).

    Parameters
    ----------
    r_ref : float
        Positive distance at the reference epoch.
    vr_ref : float
        Radial velocity at the reference epoch.
    vt_ref : float
        Nonnegative speed perpendicular to the radius vector at the
        reference epoch.
    mu : float
        Positive gravitational parameter.
    dt : float
        Time from the reference epoch; negative values run backwards.
    anomaly_fn : callable
        Universal-anomaly solver with the signature of
        ``solve_universal_anomaly``.

    Returns
    -------
    np.ndarray
        Float array ``[r, rdot]``.

    Raises
    ------
    ValueError
        If any number is not finite and real (booleans are rejected),
        ``r_ref`` or ``mu`` is not positive, ``vt_ref`` is negative,
        ``anomaly_fn`` is not callable, ``anomaly_fn`` returns anything
        other than a finite real number, or the distance history reaches the
        attracting centre at ``dt``.
    """
    return track
```

### Step 4

04_solve_light_time_range

Goal
----
Find the range along a pixel's line of sight at which the source meets an asserted barycentric distance history at the moment the observed light left it.

```python
def solve_light_time_range(
    direction: "np.ndarray",
    observer: "np.ndarray",
    t_obs: float,
    radial_fn: "Callable[[float], np.ndarray]",
    light_speed: float,
) -> "np.ndarray":
    """Return the range and emission epoch at which a line of sight meets a distance history.

    Light received at epoch ``t_obs`` by an observer at barycentric position
    ``observer``, arriving from the unit vector ``direction``, left a source
    at barycentric position ``observer + rho * direction`` at the emission
    epoch ``t_emit = t_obs - rho / light_speed``. ``radial_fn(t)`` returns
    ``[r, rdot]``, an asserted barycentric distance of the source and its
    rate of change at epoch ``t`` on the same time axis as ``t_obs``. Return
    ``[rho, t_emit]`` for the range ``rho > 0`` at which
    ``|observer + rho * direction|`` equals ``radial_fn(t_emit)[0]``.
    Select the smallest positive range if a synthetic distance history has
    more than one mathematical intersection. Supplied histories have a
    single outward crossing before any more distant intersection, and the
    observer lies inside the asserted distance at the observation epoch.
    ``rho`` must have relative accuracy ``1e-13``.

    Parameters
    ----------
    direction : "np.ndarray"
        Unit vector of shape ``(3,)`` (norm within ``1e-9`` of one).
    observer : "np.ndarray"
        Barycentric observer position of shape ``(3,)``.
    t_obs : float
        Observation epoch.
    radial_fn : callable
        Function of one epoch returning two finite numbers ``[r, rdot]``
        with ``r > 0``.
    light_speed : float
        Positive speed of light in the units of the other arguments.

    Returns
    -------
    np.ndarray
        Float array ``[rho, t_emit]``.

    Raises
    ------
    ValueError
        If ``direction`` or ``observer`` is not three finite numbers,
        ``direction`` is not a unit vector, ``t_obs`` is not a finite real
        number, ``light_speed`` is not a finite positive number,
        ``radial_fn`` is not callable or returns anything other than two
        finite numbers with a positive distance, or
        ``|observer| >= radial_fn(t_obs)[0]``.
    """
    return range_and_epoch
```

### Step 5

05_construct_emission_state

Goal
----
Build the barycentric position and velocity, at the moment its light left it, of the source placed along a pixel's line of sight on the orbit that carries a prescribed radial velocity, angular-momentum magnitude, inclination and sense of latitude motion.

```python
def construct_emission_state(
    direction: "np.ndarray",
    observer: "np.ndarray",
    rho: float,
    radial_rate: float,
    angular_momentum: float,
    inclination: float,
    kappa: int,
) -> "np.ndarray":
    """Return the emission-epoch state of the source a range ``rho`` along a line of sight.

    ``direction`` (a unit vector) and ``observer`` are barycentric ecliptic
    vectors whose ``z`` axis points to the north ecliptic pole, and the
    source lies at ``position = observer + rho * direction``. Return that
    position together with the velocity there of the Keplerian orbit whose
    radial velocity at ``position`` is ``radial_rate``, whose specific
    angular momentum ``r x v`` has magnitude ``angular_momentum`` and makes
    the angle ``inclination`` (radians) with the ``+z`` axis, and along which
    the ecliptic latitude ``asin(z / |r|)`` is increasing if ``kappa = +1``
    and decreasing if ``kappa = -1`` (``kappa`` makes no difference where
    the orbit is at its extreme latitude). Lengths and times follow the
    units of the inputs.

    Parameters
    ----------
    direction : "np.ndarray"
        Unit vector of shape ``(3,)`` (norm within ``1e-9`` of one).
    observer : "np.ndarray"
        Barycentric observer position of shape ``(3,)``.
    rho : float
        Positive range along the line of sight.
    radial_rate : float
        Velocity component along the barycentric radius vector.
    angular_momentum : float
        Positive magnitude of the specific angular momentum.
    inclination : float
        Angle between the angular momentum and ``+z``, in ``[0, pi]``.
    kappa : int
        ``+1`` or ``-1``.

    Returns
    -------
    np.ndarray
        Float array of shape ``(6,)``: position then velocity.

    Raises
    ------
    ValueError
        If ``direction`` or ``observer`` is not three finite numbers,
        ``direction`` is not a unit vector, ``rho`` is not a finite positive
        number, the position lies on the ``z`` axis, ``radial_rate`` is not
        finite, ``angular_momentum`` is not a finite positive number,
        ``inclination`` is not a finite number in ``[0, pi]``, ``kappa`` is
        not the integer ``+1`` or ``-1`` (booleans are rejected), or no orbit
        of this inclination passes through the position because the
        magnitude of its latitude exceeds ``min(inclination, pi -
        inclination)`` beyond rounding.
    """
    return emission_state
```

### Step 6

06_register_detection

Goal
----
Map one pixel of one exposure onto the barycentric ecliptic canvas at the reference epoch under an asserted initial condition.

```python
def register_detection(
    right_ascension: float,
    declination: float,
    t_obs: float,
    observer: "np.ndarray",
    initial_condition: tuple,
    obliquity: float,
    light_speed: float,
    radial_fn: "Callable[[float], np.ndarray]",
    light_time_fn: "Callable[..., np.ndarray]",
    state_fn: "Callable[..., np.ndarray]",
    propagate_fn: "Callable[[np.ndarray, float], np.ndarray]",
) -> "np.ndarray":
    """Return the canvas direction at which one pixel registers under an initial condition.

    The canvas holds barycentric ecliptic directions at the reference epoch
    ``t_ref``, and every epoch is measured from ``t_ref``. The pixel is seen
    at epoch ``t_obs`` at ICRF right ascension and declination (degrees;
    astrometric, that is light-time retarded and free of aberration) by an
    observer at barycentric ICRF position ``observer``. The ecliptic frame
    shares the ICRF ``x`` axis and has its ``z`` axis along the north
    ecliptic pole, whose ICRF direction is ``(0, -sin(eps), cos(eps))`` with
    ``eps = obliquity`` degrees. ``initial_condition = (r_ref, vr_ref,
    vt_ref, inclination, kappa)`` asserts, at ``t_ref``, the barycentric
    distance, radial velocity and speed perpendicular to the radius vector,
    the inclination of the orbit to the ecliptic in degrees, and the sign of
    the rate of ecliptic latitude; lengths and times share the units of
    ``observer``, ``light_speed`` and ``t_obs``.

    Register the pixel as the source on the unique orbit that has the
    distance history ``radial_fn``, the angular momentum and ``kappa`` of the
    initial condition, and light reaching the observer from this pixel's
    direction, and return that orbit's barycentric ecliptic longitude in
    ``[0, 360)`` and latitude at ``t_ref``, in degrees. Every vector passed
    to the callables is in the ecliptic frame:

    - ``radial_fn(t) -> [r, rdot]`` is the distance history shared by all
      orbits of the initial condition;
    - ``light_time_fn(direction, observer, t_obs, radial_fn, light_speed)
      -> [rho, t_emit]`` follows ``solve_light_time_range``;
    - ``state_fn(direction, observer, rho, radial_rate, angular_momentum,
      inclination, kappa) -> state`` follows ``construct_emission_state``
      (inclination in radians);
    - ``propagate_fn(state, dt) -> state`` follows
      ``propagate_two_body_state`` with its gravitational parameter fixed.

    Parameters
    ----------
    right_ascension, declination : float
        Pixel direction in degrees; the declination lies in ``[-90, 90]``.
    t_obs : float
        Observation epoch relative to ``t_ref``.
    observer : "np.ndarray"
        Barycentric ICRF observer position, shape ``(3,)``.
    initial_condition : tuple
        ``(r_ref > 0, vr_ref, vt_ref > 0, inclination in [0, 180],
        kappa = +1 or -1)``.
    obliquity : float
        Angle between the ICRF equator and the ecliptic, in degrees.
    light_speed : float
        Positive speed of light.
    radial_fn, light_time_fn, state_fn, propagate_fn : callable
        The callables described above.

    Returns
    -------
    np.ndarray
        Float array ``[longitude, latitude]`` in degrees.

    Raises
    ------
    ValueError
        If any number is not finite and real, the declination is outside
        ``[-90, 90]``, ``observer`` is not three finite numbers,
        ``initial_condition`` violates its stated ranges (booleans are
        rejected for ``kappa``), ``light_speed`` is not positive, a callable
        is missing, a callable returns an array of the wrong shape or with
        non-finite entries, or a callable raises ``ValueError``.
    """
    return canvas_direction
```

### Step 7

07_maximize_stacked_significance

Goal
----
Evaluate the expected significance of an inverse-variance-weighted PSF-matched-filter stack of registered exposures at its maximum, where it lies, and how far the worst-registered exposure sits from it.

```python
def maximize_stacked_significance(
    longitudes: "np.ndarray",
    latitudes: "np.ndarray",
    flux_errors: "np.ndarray",
    flux: float,
    psf_sigma: float,
) -> "np.ndarray":
    """Return the expected significance at the maximum of a matched-filter stack.

    Exposure ``k`` of a point source of constant flux ``flux`` registers on
    the canvas at the direction (``longitudes[k]``, ``latitudes[k]``), in
    degrees, and has PSF-fit flux uncertainty ``flux_errors[k]`` in the same
    flux unit. The canvas PSF is a circular Gaussian of standard deviation
    ``psf_sigma`` arcseconds. The stack is the inverse-variance-weighted
    PSF-matched filter: every exposure is cross-correlated with the PSF, the
    filtered exposures and their filtered variances are summed over
    exposures, and the significance at a canvas direction is the summed
    filtered signal divided by the square root of the summed filtered
    variance. Return the expected (noise-free) significance at its global
    maximum over canvas directions, the direction of that maximum, and the
    largest great-circle distance between the maximum and a registered
    exposure. Locate the maximum to ``1e-9`` arcsec.

    Parameters
    ----------
    longitudes, latitudes : "np.ndarray"
        Registered directions in degrees, nonempty 1-D arrays of equal
        length; latitudes lie in ``[-90, 90]``.
    flux_errors : "np.ndarray"
        Positive PSF-fit flux uncertainties, one per exposure.
    flux : float
        Positive source flux.
    psf_sigma : float
        Positive PSF standard deviation in arcseconds.

    Returns
    -------
    np.ndarray
        Float array ``[significance, longitude, latitude, largest_offset]``:
        the significance at the maximum, the maximum's longitude in
        ``[0, 360)`` and latitude in degrees, and the largest distance in
        arcseconds from the maximum to a registered exposure.

    Raises
    ------
    ValueError
        If an array is not a nonempty 1-D array of finite numbers, the
        lengths differ, a latitude lies outside ``[-90, 90]``, a flux error
        is not positive, or ``flux`` or ``psf_sigma`` is not a finite
        positive number.
    """
    return peak
```

### Step 8

08_estimate_detection_headroom

Goal
----
Register every exposure of a faint object under one trial initial condition, stack them, and return how many magnitudes fainter the object could be while its stack maximum still reaches a candidate threshold.

```python
def estimate_detection_headroom(
    times: "np.ndarray",
    observers: "np.ndarray",
    right_ascensions: "np.ndarray",
    declinations: "np.ndarray",
    flux_errors: "np.ndarray",
    initial_condition: tuple,
    flux: float,
    threshold: float,
    psf_sigma: float = 0.042,
    gm: float = 0.01720209895 ** 2,
    light_speed: float = 173.1446326846693,
    obliquity: float = 23.4392911,
    au_km: float = 149597870.7,
) -> float:
    """Return the magnitude headroom with which one initial condition's stack flags an object.

    Exposure ``k`` was taken at ``times[k]`` (TDB days from the canvas
    epoch ``t_ref``) from the barycentric ICRF position ``observers[k]``
    (au); the object's astrometric position in it is
    (``right_ascensions[k]``, ``declinations[k]``) in degrees, and
    ``flux_errors[k]`` is its PSF-fit flux uncertainty in the unit of
    ``flux``. ``initial_condition = (r_ref, vr_ref, vt_ref, inclination,
    kappa)`` asserts, at ``t_ref``, the barycentric distance (au), radial
    velocity and speed perpendicular to the radius vector (km/s, converted
    with ``au_km`` kilometres per au and 86400 s per day), the orbital
    inclination to the ecliptic (degrees) and the sign of the ecliptic
    latitude rate. Motion is two-body about a centre of gravitational
    parameter ``gm`` (au^3/day^2) fixed at the barycentre, light travels at
    ``light_speed`` (au/day), and the ecliptic is the ICRF frame tilted by
    ``obliquity`` degrees about its ``x`` axis.

    Register every exposure's object position on the barycentric ecliptic
    canvas at ``t_ref`` under the initial condition, stack the exposures
    with the inverse-variance-weighted PSF-matched filter of a circular
    Gaussian canvas PSF of standard deviation ``psf_sigma`` arcseconds, and
    return ``2.5 * log10(nu_max / threshold)``, where ``nu_max`` is the
    expected significance at the stack maximum for an object of constant
    flux ``flux``: the number of magnitudes by which the object could be
    fainter and still reach ``threshold`` (negative if it already falls
    short).

    Parameters
    ----------
    times : "np.ndarray"
        Exposure epochs, shape ``(N,)``.
    observers : "np.ndarray"
        Observer positions, shape ``(N, 3)``.
    right_ascensions, declinations : "np.ndarray"
        Object positions in degrees, shape ``(N,)`` each.
    flux_errors : "np.ndarray"
        Positive PSF-fit flux uncertainties, shape ``(N,)``.
    initial_condition : tuple
        ``(r_ref > 0, vr_ref, vt_ref > 0, inclination in [0, 180],
        kappa = +1 or -1)``.
    flux, threshold : float
        Positive object flux and positive candidate significance threshold.
    psf_sigma, gm, light_speed, au_km : float
        Positive PSF width (arcsec), gravitational parameter, speed of light
        and kilometres per au.
    obliquity : float
        Finite obliquity in degrees.

    Returns
    -------
    float
        The headroom ``2.5 * log10(nu_max / threshold)`` in magnitudes.

    Raises
    ------
    ValueError
        If the arrays are not finite with the stated shapes and a common
        ``N >= 1``, any scalar is outside its stated domain,
        ``initial_condition`` violates its stated ranges, or any stage
        rejects its input (for example when an exposure cannot be reached by
        an orbit of the asserted inclination).
    """
    return 0.0
```
