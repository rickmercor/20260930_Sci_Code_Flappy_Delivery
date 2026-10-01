# Physics-Optics-49

## Background

Hyperbolic materials, whose principal permittivities have opposite signs, support strongly confined electromagnetic modes with large wavevectors. Natural phonon-polaritonic crystals such as hexagonal boron nitride (hBN) and alpha-MoO3 are hyperbolic inside their Reststrahlen bands, the spectral intervals between transverse- and longitudinal-optical phonon frequencies where one permittivity component turns negative. This makes them low-loss mid-infrared platforms for sub-diffraction imaging, enhanced emission and waveguiding. Three-dimensional topological insulators (TIs) such as Bi2Se3 have an insulating bulk and a high mid-infrared permittivity. Their topological character appears in macroscopic electrodynamics as an axion term, which changes the electromagnetic boundary conditions wherever the axion angle jumps. At an interface this couples TE and TM fields with a strength set by the fine-structure constant.

This paper develops analytic criteria for surface electromagnetic waves bound to hyperbolic-material / topological-insulator interfaces. It separates the roles of dielectric contrast, bulk axion coupling and optical-axis orientation in setting where the surface wave exists, how deeply it penetrates each medium, and where spectral band gaps open. For lossless natural hyperbolic crystals it gives closed-form band edges and surface-wave bandwidths. It also shows that a thin TI film on a low-permittivity substrate can substantially widen the allowed window compared with a bulk TI. For dissipative artificial hyperbolic media it gives physical selection rules for complex dispersion roots. The results are design guidelines for tuning confinement, dispersion and propagation windows of mid-infrared surface waves in these heterostructures.

## Problem

At the interface between a hyperbolic material and a three-dimensional topological insulator (TI), the TI's bulk magnetoelectric (axion) response enters Maxwell's equations only through the boundary conditions, where it mixes TE and TM polarizations with a dimensionless coupling alpha = alpha_fs * Delta_theta / pi. A recent analytic theory of the lossless surface electromagnetic wave bound to such an interface gives closed-form criteria for where in the hyperbolic material's Reststrahlen band this surface wave can exist, how it is confined, and how its allowed spectral window changes when the bulk TI is replaced by a thin TI film on a low-permittivity substrate. Your task is to apply that theory to one specific thin-film structure and report the total transverse thickness of the surface wave at a prescribed operating frequency.

Here is the exact setup to use:

- Geometry: a planar interface at x = 0, with the hyperbolic material occupying x < 0 and the TI side occupying x > 0. The surface wave propagates along z with real propagation constant beta and is uniform along y. The hyperbolic material's optical axis is tangential to the interface and perpendicular to the propagation direction (the paper's l = t_y configuration). Everything is lossless (all permittivities real) and the magnetic permeabilities are 1. Use Gaussian-unit axion electrodynamics with an axion-angle jump Delta_theta = pi across the interface and alpha_fs = 7.2973525693e-3.
- Hyperbolic material: hexagonal boron nitride (hBN), uniaxial, with each principal permittivity given by a single lossless Lorentz phonon term eps(omega) = eps_inf * (omega_LO^2 - omega^2) / (omega_TO^2 - omega^2), where omega is the spectroscopic wavenumber in cm^-1. For the ordinary (in-plane) component use eps_inf,o = 4.87, omega_TO,o = 1360 cm^-1, omega_LO,o = 1614 cm^-1. For the extraordinary component use eps_inf,e = 2.95, omega_TO,e = 760 cm^-1, omega_LO,e = 825 cm^-1. Work inside the ordinary Reststrahlen band, where eps_o < 0 < eps_e (the type-I hyperbolic regime).
- TI side: this is not a semi-infinite TI. It is a Bi2Se3 film of thickness d = 12 nm (eps_TI = 41) on a semi-infinite SiO2 substrate (eps_sub = 3.9). Following the paper, represent this film-plus-substrate by a single effective TI-side permittivity eps_2,eff seen by the evanescent surface-wave field. The paper uses its own specific thin-film estimate with its own fixed length scale. Consult the paper for this construction rather than substituting a generic mixing rule or the bulk TI permittivity. Keep the finite axion coupling alpha when you use eps_2,eff.
- Allowed window and operating frequency: the lossless localized surface wave exists only in part of the ordinary Reststrahlen band. The lower edge of this allowed window is omega_TO,o. The upper edge is the paper's finite-coupling band edge omega_. The axion coupling shifts it slightly away from the purely dielectric criterion, so consult the paper's own band-edge condition for how alpha enters rather than assuming the non-topological one. Take the operating wavenumber at the midpoint of the allowed window, omega_op = omega_TO,o + 0.5 * (omega_ - omega_TO,o).
- At omega_op, solve the paper's lossless dispersion relation for this configuration, including the axion coupling, for the real effective index n = beta/k0 on the localized branch. Here k0 = 2piomega_op, with omega_op in cm^-1. Then evaluate the TI-side penetration depth delta_TI and the two hyperbolic-medium penetration depths, delta_1 (TE-like) and delta_2 (TM-like). Each is the inverse of the corresponding real transverse decay constant for this optical-axis orientation.

Report delta_total = delta_TI + delta_1, the total surface-wave thickness in nanometres at omega_op. The answer is graded to an absolute tolerance of 1e-4 nm. In your reasoning, report the following:

- eps_2,eff, omega_*, the surface-wave bandwidth fraction f_SW (the paper's own definition), and omega_op
- eps_o and eps_e at omega_op, and n
- delta_TI, delta_1 and delta_2, together with their ordering and the reason for it
- the paper's own tabulated benchmark values of eps_2,eff and f_SW for a 5 nm Bi2Se3 film on SiO2 next to hBN, used as a check of your thin-film construction

Output Format Requirements: Emit <final_answer> immediately, then <reasoning>. Do not write a long derivation before the tags. You must emit exactly one finite decimal inside <final_answer>...</final_answer>, even if the value is approximate or you are unsure. Rules:

- The tags are required. Do not omit them or leave them empty.
- The tagged value must be a finite decimal (examples: 0.4847, 2.29, 1.05). Not NaN, not Inf, not a fraction string, not a vector, not prose.
- Put only that one number between the tags. No units, no words, no extra lines. Keep <reasoning> short (a few hundred words). Show only the few scalars that determine the final number. Do not paste full derivations of the paper's dispersion relation or band-edge analysis.

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

hbn_uniaxial_permittivities

Goal
----
Evaluate the two principal relative permittivities of hexagonal boron nitride (hBN), the ordinary (in-plane) component eps_o and the extraordinary (optical-axis) component eps_e, at a given mid-infrared wavenumber, each modelled by a single lossless Lorentz phonon resonance.

```python
import numpy as np


def hbn_uniaxial_permittivities(omega_cm: float) -> "np.ndarray":
    """Ordinary and extraordinary relative permittivities of lossless hBN.

    Args:
        omega_cm (float): wavenumber in cm^-1, strictly positive and not equal
            to either transverse-optical resonance (1360 or 760 cm^-1).

    Raises:
        ValueError: if omega_cm <= 0 or omega_cm coincides with a TO resonance.

    Expected return:
        np.ndarray of shape (2,): [eps_o, eps_e]. Inside the upper Reststrahlen
        band (1360 < omega_cm < 1614) eps_o is negative and eps_e is positive.
    """
    return None
```

### Step 2

thin_film_effective_permittivity

Goal
----
Compute the effective topological-insulator-side permittivity eps_2,eff seen by the evanescent surface-wave field when the topological insulator is not a semi-infinite bulk but a thin film of thickness d on a dielectric substrate.

```python
import numpy as np


def thin_film_effective_permittivity(d_nm: float, eps_ti: float, eps_sub: float) -> float:
    """Effective TI-side permittivity for a thin TI film on a substrate.

    Args:
        d_nm (float): TI film thickness in nm; must be strictly positive and
            strictly smaller than the paper's own effective sampling length.
        eps_ti (float): relative permittivity of the TI material, > 0.
        eps_sub (float): relative permittivity of the substrate, > 0.

    Raises:
        ValueError: if d_nm <= 0, if d_nm is not smaller than the paper's
            sampling length, or if eps_ti <= 0 or eps_sub <= 0.

    Expected return:
        float: eps_2,eff, lying strictly between eps_sub and eps_ti and moving
        monotonically toward eps_ti as the film thickens.
    """
    return 0.0
```

### Step 3

finite_coupling_band_edge

Goal
----
Compute the upper band-edge wavenumber omega_* of the lossless surface-wave window, the frequency inside the ordinary Reststrahlen band at which the localized HM/TI surface wave ceases to exist, including the finite topological (axion) interface coupling alpha.

```python
import numpy as np


def finite_coupling_band_edge(eps_inf_o: float, w_to: float, w_lo: float, eps2: float, alpha: float) -> float:
    """Upper edge of the lossless HM/TI surface-wave window.

    Args:
        eps_inf_o (float): high-frequency ordinary permittivity, > 0.
        w_to (float): ordinary TO phonon wavenumber (cm^-1), > 0.
        w_lo (float): ordinary LO phonon wavenumber (cm^-1), > w_to.
        eps2 (float): TI-side permittivity (bulk or effective), > 0.
        alpha (float): dimensionless axion interface coupling, >= 0.

    Raises:
        ValueError: if w_lo <= w_to, w_to <= 0, eps_inf_o <= 0, eps2 <= 0,
            or alpha < 0.

    Expected return:
        float: omega_* in cm^-1, strictly inside (w_to, w_lo); it decreases
        toward w_to as eps2 grows and moves slightly downward as alpha grows.
    """
    return 0.0
```

### Step 4

surface_wave_bandwidth_fraction

Goal
----
Compute the surface-wave bandwidth fraction f_SW, the fraction of the ordinary Reststrahlen band that is occupied by the allowed lossless surface-wave window, using the band edge from the previous step.

```python
import numpy as np


def surface_wave_bandwidth_fraction(eps_inf_o: float, w_to: float, w_lo: float, eps2: float, alpha: float) -> float:
    """Fraction of the Reststrahlen band occupied by the surface-wave window.

    Args:
        eps_inf_o (float): high-frequency ordinary permittivity, > 0.
        w_to (float): ordinary TO phonon wavenumber (cm^-1).
        w_lo (float): ordinary LO phonon wavenumber (cm^-1), > w_to.
        eps2 (float): TI-side permittivity (bulk or effective), > 0.
        alpha (float): dimensionless axion interface coupling, >= 0.

    Raises:
        ValueError: under the same conditions as the band-edge step.

    Expected return:
        float: f_SW in (0, 1); larger for smaller eps2.
    """
    return 0.0
```

### Step 5

axion_dispersion_mismatch

Goal
----
Evaluate the mismatch of the lossless HM/TI surface-wave dispersion relation for the tangential optical-axis configuration (optical axis in the interface plane, perpendicular to the propagation direction) at a trial real effective index n = beta/k0, for hBN on the hyperbolic side.

```python
import numpy as np


def axion_dispersion_mismatch(n: float, omega_cm: float, eps2: float, alpha: float) -> float:
    """Mismatch of the tangential-axis HM/TI dispersion relation at trial n.

    Args:
        n (float): trial real effective index beta/k0; n^2 must exceed both
            eps2 and eps_e(omega) so that every decay constant is real.
        omega_cm (float): wavenumber in cm^-1 inside the ordinary Reststrahlen
            band of hBN (eps_o < 0).
        eps2 (float): TI-side permittivity (bulk or effective), > 0.
        alpha (float): dimensionless axion interface coupling, >= 0.

    Raises:
        ValueError: if eps_o(omega) >= 0, if n^2 <= max(eps2, eps_e), or if
            alpha < 0.

    Expected return:
        float: dimensionless mismatch; negative just above the lower bound of
        n^2, positive at large n inside the allowed window, zero on the
        surface-wave branch.
    """
    return 0.0
```

### Step 6

surface_wave_effective_index

Goal
----
Find the real effective index n = beta/k0 of the localized lossless HM/TI surface wave (tangential optical axis, hBN hyperbolic side) at a given wavenumber, by locating the zero of the dispersion mismatch from the previous step on the localized interval.

```python
import numpy as np


def surface_wave_effective_index(omega_cm: float, eps2: float, alpha: float) -> float:
    """Effective index of the localized HM/TI surface wave.

    Args:
        omega_cm (float): wavenumber in cm^-1 inside the ordinary Reststrahlen
            band of hBN.
        eps2 (float): TI-side permittivity (bulk or effective), > 0.
        alpha (float): dimensionless axion interface coupling, >= 0.

    Raises:
        ValueError: if eps_o(omega) >= 0, or if no localized solution exists at
            this wavenumber (e.g. omega_cm lies in the surface-wave band gap).

    Expected return:
        float: n, the unique root on the localized interval; n grows without
        bound as omega_cm approaches the band edge from below.
    """
    return 0.0
```

### Step 7

surface_wave_penetration_depths

Goal
----
Convert a converged effective index into the three lossless transverse penetration depths of the surface wave (tangential optical axis, hBN hyperbolic side): into the TI side (delta_TI), and the TE-like (delta_1) and TM-like (delta_2) depths into the hyperbolic medium, in nanometres.

```python
import numpy as np


def surface_wave_penetration_depths(n: float, omega_cm: float, eps2: float) -> "np.ndarray":
    """Lossless transverse penetration depths of the HM/TI surface wave.

    Args:
        n (float): effective index beta/k0 with n^2 > max(eps2, eps_e).
        omega_cm (float): wavenumber in cm^-1 inside the ordinary Reststrahlen
            band of hBN.
        eps2 (float): TI-side permittivity (bulk or effective), > 0.

    Raises:
        ValueError: if eps_o(omega) >= 0 or n^2 <= max(eps2, eps_e).

    Expected return:
        np.ndarray of shape (3,): [delta_TI, delta_1, delta_2] in nm, all
        positive; delta_2 is always the smallest of the three.
    """
    return None
```

### Step 8

run_hm_ti_surface_wave_pipeline

Goal
----
Chain the earlier steps for an hBN / thin-film Bi2Se3-on-substrate interface (tangential optical axis, axion coupling alpha = alpha_fs for a jump Delta_theta = pi): build the effective TI-side permittivity, locate the finite-coupling band edge, choose the operating wavenumber at a given fractional position inside the allowed window, solve for the effective index, and return the total surface-wave thickness. The reference implementation calls the earlier public functions by name.

```python
import numpy as np


def run_hm_ti_surface_wave_pipeline(d_nm: float, eps_sub: float, eps_ti: float, window_fraction: float) -> float:
    """Total thickness of the hBN / thin-film-TI surface wave.

    Args:
        d_nm (float): TI film thickness in nm (see the thin-film step).
        eps_sub (float): substrate permittivity, > 0.
        eps_ti (float): TI permittivity, > 0.
        window_fraction (float): fractional position x of the operating
            wavenumber inside the allowed window, 0 < x < 1.

    Raises:
        ValueError: if window_fraction is not strictly between 0 and 1, or if
            any earlier step rejects its inputs.

    Expected return:
        float: delta_total in nm (positive).
    """
    return 0.0
```
