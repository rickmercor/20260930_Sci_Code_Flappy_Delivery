# Physics-Optics-4

## Background

A surface plasmon polariton is a TM-polarised electromagnetic mode bound to a
metal-dielectric interface, propagating in the plane with an in-plane wave number set by the
permittivities on either side and decaying evanescently away from the surface. Confined in one direction only, a pulsed SPP of finite transverse extent behaves like a
two-dimensional light sheet: it spreads diffractively along the surface and, because the SPP
dispersion is curved, dispersively in time. Both limit how far a short plasmonic pulse stays
compact, which is the practical obstacle to surface-enhanced spectroscopy and nonlinear
interaction at high peak field.

The baseline comes from the space-time wave packet literature: restricting the
spatiotemporal spectrum from a two-dimensional patch on the light-cone to a one-dimensional
curve makes the time-averaged intensity propagation-invariant, and the shape of that curve
fixes the axial group velocity independently of the medium. On a metal surface the relevant cone is the SPP light-cone,
k_x^2 + k_z^2 = k_SPP(omega)^2, where k_x is the transverse and k_z the axial in-plane wave
number. Intersecting that cone with a tilted spectral plane gives the ideal space-time SPP,
whose group velocity follows from the tilt angle alone. That construction is standard and
pre-dates this task.

Two further ingredients are ordinary multilayer electromagnetism. A real surface is not one
interface: coating, finite-thickness metal film and substrate each shift the mode, and the
shift follows exactly from the stack's TM mode condition rather than from perturbing the
two-interface result. And when the in-plane wave number falls below the substrate light
line the mode becomes leaky - radiating into the substrate as well as losing energy
ohmically - so the substrate transverse wave number is imaginary and the physical solution
is the outgoing branch.

The quantity this task turns on is the distinction between the phase index Re(k_SPP)c/omega
and the group index c d(Re k_SPP)/d(omega). They differ by several percent on a coated
metal, and it is the group index, not the phase index, that sets how a wave packet's
envelope moves. Anisotropy matters for the same reason: a spin-coated polymer film is generally uniaxial
with its optic axis along the normal, and for a TM mode the transverse wave number there
involves the ratio of ordinary to extraordinary permittivity.

What is deliberately not supplied here is the specific spectral construction the task asks
about, or the relation between its control angle and the resulting group velocity. That is
post-cutoff work and is what the browsing sources carry.

## Problem

A surface plasmon polariton (SPP) launched as a short pulse spreads in two ways at once: its
transverse profile diffracts and its axial profile disperses. Both can be suppressed by
sculpting the spatiotemporal spectrum of the wave packet so that its support collapses from a
two-dimensional patch on the SPP light-cone onto a one-dimensional curve, producing a
space-time SPP (ST-SPP) that travels rectilinearly without diffraction. The curve is realized
by imposing a V-shaped projection of the spectral support onto the (k_x, omega/c) plane, whose
opening angle alpha is the single control parameter of the synthesis: it tunes the axial group
velocity of the packet away from that of an ordinary plane-wave pulsed SPP, one sign of alpha
giving a subluminal packet and the other a superluminal one. The V's apex position within the
band, and which sign of alpha belongs to which regime, are part of that construction and are
not given here.

For the four-layer stack specified below, find the superluminal opening angle at which
the ST-SPP becomes exactly luminal - that is, the value of |alpha| for which the axial group
velocity of the ST-SPP equals the vacuum speed of light exactly. Report |alpha| in degrees to
six significant figures.

The surface is a four-layer stack: vacuum above, a non-dispersive UNIAXIAL dielectric
coating of thickness d_c whose optic axis lies along the surface normal, a metal film of thickness d_m, and a semi-infinite substrate below. The
surface wave is the TM guided mode of that stack. Because its in-plane wave number lies
BELOW the substrate light line, the mode is leaky: it radiates into the substrate as well as
losing power ohmically, so its wave number k_SPP(omega) is complex for two distinct reasons.
That wave number sets the light-cone k_x^2 + k_z^2 = Re k_SPP(omega)^2 on which the ST-SPP
spectrum lives; the axial wave number is then
k_z(omega) = +sqrt(Re k_SPP(omega)^2 - k_x(omega)^2). The packet is built from a spectral
band specified in vacuum wavelength: centred on lambda_o with full extent d_lambda, so it
occupies lambda_o - d_lambda/2 <= lambda <= lambda_o + d_lambda/2 and nothing outside it.

Parameters. The metal permittivity is a Drude term plus one Lorentz oscillator,

    eps_m(omega) = eps_inf - omega_p^2/(omega^2 + i*gamma_D*omega)
                   + f1*omega_1^2/(omega_1^2 - omega^2 - i*gamma_1*omega),

with eps_inf = 3.70, omega_p = 1.3600e16 rad/s, gamma_D = 3.0000e13 rad/s, f1 = 0.2000,
omega_1 = 6.0000e15 rad/s and gamma_1 = 1.0000e15 rad/s. The coating is uniaxial with ordinary index
n_o = 1.520 in the plane of the surface and extraordinary index n_e = 1.495 along the
normal, and is d_c = 12.00 nm thick; the metal film is d_m = 55.00 nm thick; the
substrate has relative permittivity eps_sub = 3.67^2; the superstrate is vacuum, eps = 1.
The band is centred at lambda_o = 780.0 nm with full extent d_lambda = 96.0 nm. Take
c = 2.99792458e8 m/s. Angular frequency and vacuum wavelength are related by
omega = 2*pi*c/lambda.

Conventions, so that the result is determined rather than inferred:

- The time convention is exp(-i*omega*t), so Im(eps_m) > 0. In the vacuum superstrate the mode
  is bound: take the root of kappa^2 = beta^2 - eps*(omega/c)^2 with positive real part. The
  mode condition is even in the coating's and the film's transverse wave numbers, so those
  branches are immaterial.
- In the substrate it is neither. There kappa is imaginary and the mode radiates away, so take
  the outgoing root, Im(kappa_sub) <= 0. The principal square root gives the incoming branch,
  whose mode gains amplitude as it propagates - impossible in a passive stack, and visible as
  a negative propagation length.
- In an isotropic layer the transverse wave number satisfies kappa^2 = beta^2 - eps*(omega/c)^2.
  In the uniaxial coating it does not: for the TM mode there,
  kappa^2 = (eps_o/eps_e)*beta^2 - eps_o*(omega/c)^2, and the ordinary permittivity is the
  one that enters the layer's admittance. Setting eps_o = eps_e recovers the isotropic form.
- Of the roots of the stack's mode condition, take the one continuously connected to the SPP
  of the uncoated semi-infinite metal, beta = (omega/c)*sqrt(eps_s*eps_m/(eps_s+eps_m)).
- All spectral geometry - light-cone, k_x, k_z, every group velocity - uses the real part of
  the mode wave number. The imaginary part enters only the propagation length 1/(2*Im k_SPP)
  and the split of that loss between absorption and radiation. Define
  that split as an EXCESS-ATTENUATION fraction: solve the same stack a second time with the
  metal made semi-infinite, which removes the substrate and leaves only ohmic absorption,
  and report 1 - Im(k_SPP, semi-infinite metal)/Im(k_SPP). Do not use a Poynting-flux or
  Joule-integral partition; those give a different number.
- The band-centre angular frequency is omega_o = 2*pi*c/lambda_o; everything below described
  as "at the band centre" is evaluated there.
- The plane-wave SPP group index is n_SPP = c*d(Re k_SPP)/d(omega) evaluated at omega_o. The
  axial group velocity of the ST-SPP is d(omega)/d(k_z) evaluated at omega_o along the packet's
  own spectral curve.
- Take that derivative exactly. Do not replace it by a series expansion in the fractional
  bandwidth: an expansion to leading order in d_lambda/lambda_o shifts the answer in its sixth
  significant figure.
- The free-space field that the coupler converts into the ST-SPP carries the same transverse
  wave number as the surface wave, so a spectral component at omega leaves the synthesis
  system at an angle phi(omega) to the axis with sin(phi) = c*|k_x(omega)|/omega. The required
  numerical aperture is the largest value of sin(phi) over the band.

What to report. Give the final |alpha| in degrees, and state these intermediate scalars, each
to at least six significant figures: Re(eps_m) and Im(eps_m) at omega_o; the mode index Re(k_SPP)*c/omega of
the four-layer stack at omega_o; the propagation length at omega_o in micrometres; the
excess-attenuation radiative fraction defined above; the group index
n_SPP; and, at the angle you report, the transverse wave number k_x at omega_o in rad/um, the
axial wave number k_z at omega_o in rad/um, and the required numerical aperture. Do not
reproduce the parameter table or show iteration paths.

Then answer each of the following in one sentence, naming the source for each. From the experimental work that
introduced this construction, report: the groove density of the grating and the focal length
of the cylindrical lens in its pulse synthesizer, and how its spatial light modulator gives
each wavelength its own propagation angle; the width, depth and length of the nano-slits it
milled to launch its ST-SPPs, and the length of the second set of slits it used for
conventional SPP wave packets; and the group velocities, with uncertainties, that it measured
for its striped, subluminal and superluminal ST-SPPs, together with the conventional-SPP
group velocity it compared them with. Name the earlier free-space work that introduced the
V-shaped spectral projection, and say what bending the projection away from a straight line
reintroduces. Say what property of the map between omega and |k_x| the diffraction-free
condition demands, and what that property forces about where the V's apex sits; and say
which sign convention on alpha produces a group velocity below that of the plane-wave SPP.

Separately, compare apertures on the stack specified here: give the numerical aperture your
V-shaped construction requires, and the numerical aperture a luminal IDEAL ST-SPP - a
straight spectral plane omega - omega_o = (k_z - k_o')*c, with k_o' = Re k_SPP(omega_o) - would require over the same band,
and state which frequencies of that band the ideal plane can be built at all. Do not assume
the ideal construction needs the larger aperture.

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
Output Format Requirements:
Emit <final_answer> immediately, then <reasoning>. Do not write a long derivation before the tags.
You must emit exactly one finite decimal inside <final_answer>...</final_answer>, even if the value is approximate or you are unsure.
Rules:
- The tags are required. Do not omit them or leave them empty.
- The tagged value must be a finite decimal (examples: 0.4847, 0.22, 1.05). Not NaN, not Inf, not a fraction string, not a vector, not prose.
- Put only that one number between the tags. No units, no words, no extra lines.
Keep <reasoning> short (a few hundred words). Show only the few scalars that determine the final number.
Do not paste the input matrices, full coefficient vectors, per-iteration paths, or per-fold candidate tables.
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

film_mode_index

Goal
----
Real mode index of the TM guided mode of the four-layer stack.

```python
def film_mode_index(nu_thz: float, d_m_nm: float) -> float:
    """Real mode index of the TM guided mode of the four-layer stack.

    Args:
        nu_thz: ordinary frequency in THz.
        d_m_nm: metal-film thickness in nm.

    Returns:
        float: mode index Re(beta)*c/w, dimensionless.

    Raises:
        ValueError: if nu_thz is not finite and positive; or if d_m_nm is not finite and positive.
    """
    return None
```

### Step 2

propagation_length

Goal
----
Propagation length of the leaky surface mode, in micrometres.

```python
def propagation_length(nu_thz: float, d_m_nm: float) -> float:
    """Propagation length of the leaky surface mode, in micrometres.

    Args:
        nu_thz: ordinary frequency in THz.
        d_m_nm: metal-film thickness in nm.

    Returns:
        float: propagation length in micrometres.

    Raises:
        ValueError: if nu_thz is not finite and positive; or if d_m_nm is not finite and positive.
    """
    return None
```

### Step 3

spp_group_index

Goal
----
Group index c*d(Re beta)/d(omega) of the plane-wave pulsed surface mode.

```python
def spp_group_index(nu_thz: float, d_m_nm: float) -> float:
    """Group index c*d(Re beta)/d(omega) of the plane-wave pulsed surface mode.

    Args:
        nu_thz: ordinary frequency in THz.
        d_m_nm: metal-film thickness in nm.

    Returns:
        float: group index, dimensionless.

    Raises:
        ValueError: if nu_thz is not finite and positive; or if d_m_nm is not finite and positive.
    """
    return None
```

### Step 4

transverse_wavenumber

Goal
----
Transverse wave number k_x of the V-shaped spatiotemporal spectrum, in rad/um.

```python
def transverse_wavenumber(nu_thz: float, alpha_deg: float) -> float:
    """Transverse wave number k_x of the V-shaped spatiotemporal spectrum, in
    rad/um.

    Args:
        nu_thz: ordinary frequency in THz.
        alpha_deg: opening angle in degrees; negative is the subluminal branch, positive the superluminal one.

    Returns:
        float: transverse wave number in rad/um.

    Raises:
        ValueError: if nu_thz is not finite and positive; or if alpha_deg is not finite, or |alpha_deg| is not strictly between 0 and 90.
    """
    return None
```

### Step 5

axial_wavenumber

Goal
----
Axial wave number k_z of the ST-SPP spectral curve, in rad/um.

```python
def axial_wavenumber(nu_thz: float, alpha_deg: float, d_m_nm: float) -> float:
    """Axial wave number k_z of the ST-SPP spectral curve, in rad/um.

    Args:
        nu_thz: ordinary frequency in THz.
        alpha_deg: opening angle in degrees.
        d_m_nm: metal-film thickness in nm.

    Returns:
        float: axial wave number in rad/um.

    Raises:
        ValueError: if nu_thz is not finite and positive; or if alpha_deg is not finite, or |alpha_deg| is not strictly between 0 and 90; or if d_m_nm is not finite and positive; or if |alpha_deg| is at or below the minimum opening angle at nu_thz, where |k_x(nu_thz)| >= Re(k_SPP(nu_thz)) and k_z is not real.
    """
    return None
```

### Step 6

st_group_index

Goal
----
Axial group index of the ST-SPP at the band-centre frequency.

```python
def st_group_index(alpha_deg: float, k_spp_o_rad_um: float, n_spp: float) -> float:
    """Axial group index of the ST-SPP at the band-centre frequency.

    Args:
        alpha_deg: opening angle in degrees; negative subluminal, positive superluminal.
        k_spp_o_rad_um: Re(k_SPP) at the band centre in rad/um, from step 01.
        n_spp: plane-wave group index at the band centre, from step 03.

    Returns:
        float: axial group index c/v, dimensionless.

    Raises:
        ValueError: if alpha_deg is not finite, or |alpha_deg| is not strictly between 0 and 90; or if k_spp_o_rad_um is not finite and positive; or if n_spp is not finite and positive; or if |alpha_deg| is at or below the minimum opening angle, where |k_x(omega_o)| >= k_spp_o_rad_um and the square root is not real.
    """
    return None
```

### Step 7

required_numerical_aperture

Goal
----
Free-space numerical aperture the synthesis system must supply.

```python
def required_numerical_aperture(alpha_deg: float) -> float:
    """Free-space numerical aperture the synthesis system must supply.

    Args:
        alpha_deg: opening angle in degrees.

    Returns:
        float: numerical aperture, dimensionless.

    Raises:
        ValueError: if alpha_deg is not finite, or |alpha_deg| is not strictly between 0 and 90.
    """
    return None
```

### Step 8

luminal_opening_angle

Goal
----
Superluminal opening angle at which the ST-SPP group velocity equals c
exactly.

```python
def luminal_opening_angle(d_m_nm: float) -> float:
    """Superluminal opening angle at which the ST-SPP group velocity equals c
    exactly.

    Args:
        d_m_nm: metal-film thickness in nm.

    Returns:
        float: opening angle in degrees.

    Raises:
        ValueError: if d_m_nm is not finite and positive; or if the propagation length is not positive (incoming substrate branch); or if no superluminal luminal angle exists; or if the result fails the light-cone or numerical-aperture check.
    """
    return None
```
