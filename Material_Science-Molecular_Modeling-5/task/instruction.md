# Material_Science-Molecular_Modeling-5

## Background

Two-dimensional Coulomb liquids are the classical model for charges confined to a plane: ions
adsorbed at an interface, vortices in thin superconducting films, or the charges of the
two-dimensional Coulomb gas that underlies the Berezinskii-Kosterlitz-Thouless transition. The
pair potential between unit charges is logarithmic, -Gamma ln(r/a), so the coupling Gamma is
the only energy scale, and giving the ions a hard core of diameter a introduces the second
control parameter, the reduced density rho a^2. Monte Carlo simulations of this model exist
over a broad range of couplings and densities, and they show that the plain Debye-Hueckel
theory, which ignores both the hard core and the non-linear short-range correlations, fails
already at moderate coupling and at any density that is not dilute.

Field-theoretic treatments of electrolytes recover the Debye-Hueckel result as the Gaussian
level of a functional integral and improve on it in different ways. A splitting technique
separates the Coulomb potential into a long-range part, which is weakly coupled and can be
treated at the Gaussian level with screening, and a short-range part, which is strongly
coupled and is treated exactly through a Mayer function together with the hard core. The
splitting is done in reciprocal space with a filter polynomial and a splitting length; because
the grand potential does not depend on how the potential is split, its stationarity with
respect to the splitting length turns into a variational condition once the correlators are
approximated. The scheme is self-consistent because the screened long-range kernel is
determined by the ion density and the filter at the same time as the splitting length.

The first correction beyond this mixed Gaussian-Mayer level is the chain of two Mayer
functions through an intermediate ion, summed over the species of that ion with its density,
from which the ring already contained in the screened kernel is subtracted. It is a
two-dimensional convolution of functions that jump at the contact distance, so the geometry of
two overlapping exclusion disks enters directly. From the total correlation functions follow the
partial and charge structure factors, the excess energy through the logarithmic potential, and
the specific heat through the coupling derivative of the energy; the last two are the
quantities on which such theories are usually judged against simulation, and the sign of the
excess energy at strong coupling is one of their qualitative predictions.

## Problem

A two-dimensional liquid of charged hard disks of diameter a, with equal numbers of monovalent cations and anions at total reduced density rho a^2 and the logarithmic Coulomb potential -Gamma ln(r/a) between unit charges, is treated with the self-consistent Debye-Hueckel (SCDH) scheme of the source paper, defined below with all lengths in units of a, wavenumbers in units of 1/a, u = r/a, bars denoting two-dimensional Fourier transforms, and f * g the two-dimensional convolution (the integral over the plane of f(r') g(|r - r'|)).

```
filter:           S(q) = 1 + (sigma q)^2 + (sigma q)^4 + (sigma q)^6 + (sigma q)^8,  sigma = splitting length
screening:        kappa0^2 = 2 pi Gamma rho a^2
long-range part:  vlbar(q) = 2 pi Gamma / (q^2 S(q));  screened kernel Glbar(q) = 2 pi Gamma / (q^2 S(q) + kappa0^2),  Gl(u) its inverse transform
short-range part: vs(u) = inverse transform of 2 pi Gamma (S - 1) / (q^2 S) = -Gamma ln u - vl(u)
Mayer functions:  w = vs + Gl;  h+-(u) = exp(+w) - 1 and h++(u) = exp(-w) - 1 for u >= 1;  h+- = h++ = -1 for u < 1
                  (transform of the hard-core disk: -2 pi J1(q)/q)
variational:      integral over q from 0 to infinity of q [hbar++(q) - hbar+-(q) + 2 Glbar(q)] d vlbar/d sigma = 0  fixes sigma
corrections:      T_ij = sum over the intermediate species n of n_n [h_in * h_nj - q_i q_j q_n^2 Gl * Gl],  n_+ = n_- = rho a^2/2, q = +1, -1
                  so T++ = (rho a^2/2)[h++ * h++ + h+- * h+- - 2 Gl * Gl],  T+- = rho a^2 [h++ * h+- + Gl * Gl]
                  (the Mayer functions keep their hard-core value -1 inside the convolutions)
total correlations: H_ij = h_ij + (1 + h_ij) T_ij
structure factors:  S_ij(q) = delta_ij/2 + (rho a^2/4) Hbar_ij(q);  SZZ(q) = 1 + (rho a^2/2)[Hbar++(q) - Hbar+-(q)]
excess energy:    E = beta u_ex/rho = (pi Gamma rho a^2/2) integral from u = 1 to infinity of u ln(u) [H+-(u) - H++(u)] du
                  Debye-Hueckel limit (sigma = 0, corrections off): (Gamma/2) K0(kappa0)
specific heat:    CV = E - Gamma dE/dGamma at fixed density, sigma re-solved at every coupling;  DH limit (Gamma/4) kappa0 K1(kappa0)
```

The convolution form of T_ij with the species densities is the operative definition of the corrections in this task: the source also prints reciprocal-space versions of these integrals, and where their printed prefactor differs from the convolution defined here, the convolution governs.

Work at Gamma = 2.0 and rho a^2 = 0.10, a state the source does not tabulate, and converge every value to at least six significant figures: solve the variational condition for sigma*; evaluate the three energies (corrections off, first order, Debye-Hueckel), the first-order specific heat, S+-(q) at q = 2, SZZ(q) at q = 0.5, the total pair potential w at contact, the two Mayer functions at contact, T++ at contact, Gl at u = 0, and the variational residual in its reciprocal-space definition at the trial length sigma = 0.7. Then, at the same density, find the coupling Gamma0 in 8 <= Gamma <= 9.2 at which the first-order excess energy changes sign, re-solving sigma at every coupling, and give sigma* there. State how the conditionally convergent integral defining vs, the convolutions of functions that jump at contact, and the coupling derivative in CV were handled, and, as a control, verify that with sigma = 0 and the corrections off the chain reproduces the Debye-Hueckel energy at Gamma = 0.02 and rho a^2 = 0.15 to second order in the coupling, reporting the relative deviation. The final answer is Gamma0, the sign-change coupling of the first-order energy at rho a^2 = 0.10 (not the sign change of the uncorrected energy, and not the value obtained with the source's printed reciprocal-space prefactor).

Your reasoning should also report, as evidence that the chain was executed: sigma* at (2.0, 0.10); the three energies and the specific heat; S+-(2) and SZZ(0.5); w(1), h+-(1) and h++(1); T++(1); Gl(0); the residual at sigma = 0.7 and its sign on either side of sigma*; sigma* at Gamma0; the Debye-Hueckel control and its relative deviation; the lens area of two overlapping unit disks used in the convolutions and the distance at which the corrections have a kink; and, from the source, the coupling near which it reports the sign change of the energy at rho a^2 = 0.15, the factor by which its printed reciprocal-space prefactor for the correction integrals differs from the convolution definition used here, the reason it gives for the eighth-order filter, and the simulation data sets against which it validated the theory.

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

Implement **all 10 functions** below into a single file at `/app/solution.py`. Requirements:

- Use the EXACT function name and signature shown for each step.
- Define every function at module scope (importable).
- Do NOT include tests, example usage, prints, or a `__main__` block.
- Available libraries: numpy, scipy, sympy, pandas, scikit-learn, networkx.

Start your file with these imports:

```python
import numpy as np
```

### Step 1

filter_kernels

Goal
----
Return, for a one-dimensional array of positive wavenumbers q (lengths are measured in the disk diameter, so q is dimensionless), the four reciprocal-space building blocks of the splitting scheme at splitting length sigma, coupling Gamma and total reduced ion density rho: the filter polynomial S(q) = 1 + (sigma q)^2 + (sigma q)^4 + (sigma q)^6 + (sigma q)^8; the long-range potential 2 pi Gamma/(q^2 S(q)); the screened long-range kernel 2 pi Gamma/(q^2 S(q) + kappa0^2), where kappa0^2 = 2 pi Gamma rho is the Debye-Hueckel parameter of the full ion density; and the derivative of the long-range potential with respect to sigma at fixed q. Reject a coupling or density that is not positive and finite, a negative splitting length, and any wavenumber that is not positive and finite.

```python
def filter_kernels(q: "np.ndarray", sigma: float, Gamma: float, rho: float) -> "np.ndarray":
    """Return, for a one-dimensional array of positive wavenumbers q (lengths are measured in the
    disk diameter, so q is dimensionless), the four reciprocal-space building blocks of the
    splitting scheme at splitting length sigma, coupling Gamma and total reduced ion density
    rho: the filter polynomial S(q) = 1 + (sigma q)^2 + (sigma q)^4 + (sigma q)^6 + (sigma q)^8;
    the long-range potential 2 pi Gamma/(q^2 S(q)); the screened long-range kernel 2 pi
    Gamma/(q^2 S(q) + kappa0^2), where kappa0^2 = 2 pi Gamma rho is the Debye-Hueckel parameter
    of the full ion density; and the derivative of the long-range potential with respect to
    sigma at fixed q. Reject a coupling or density that is not positive and finite, a negative
    splitting length, and any wavenumber that is not positive and finite.

    Args:
        q: array-like of shape (n,) of positive finite wavenumbers in units of the inverse disk diameter.
        sigma: non-negative finite float, the splitting length in units of the disk diameter (sigma = 0 is the
            Debye-Hueckel limit of the scheme).
        Gamma: positive finite float, the Coulomb coupling.
        rho: positive finite float, the total reduced ion density rho a^2 (both species together).

    Returns:
        A numpy float64 array of shape (4, n) whose rows hold, in order, the filter S(q), the long-range
        potential 2 pi Gamma/(q^2 S(q)), the screened kernel 2 pi Gamma/(q^2 S(q) + kappa0^2) with
        kappa0^2 = 2 pi Gamma rho, and the derivative of the long-range potential with respect to sigma at
        fixed q, all evaluated at the n wavenumbers.

    Raises:
        ValueError: if q is not a one-dimensional array of positive finite values; if sigma is negative or
            not finite; or if Gamma or rho is not a positive finite number.
    """
    return None
```

### Step 2

long_range_kernel

Goal
----
Return the real-space screened long-range kernel Gl(u) at the non-negative distances u: the inverse two-dimensional Fourier transform of the screened kernel of the previous step, that is Gamma times the integral over q from zero to infinity of q J0(q u) divided by q^2 S(q) + kappa0^2. The result must be accurate to a relative error of 1e-10 at every distance, including u = 0 where the kernel is finite, and must reduce to the Debye-Hueckel kernel Gamma K0(kappa0 u) when sigma = 0. The integrand is a rational function of q^2 whose denominator is a polynomial of degree five in q^2 with positive coefficients, so no oscillatory quadrature is needed if that structure is used. Reject invalid parameters and negative or non-finite distances.

```python
def long_range_kernel(u: "np.ndarray", sigma: float, Gamma: float, rho: float) -> "np.ndarray":
    """Return the real-space screened long-range kernel Gl(u) at the non-negative distances u: the
    inverse two-dimensional Fourier transform of the screened kernel of the previous step, that
    is Gamma times the integral over q from zero to infinity of q J0(q u) divided by q^2 S(q) +
    kappa0^2. The result must be accurate to a relative error of 1e-10 at every distance,
    including u = 0 where the kernel is finite, and must reduce to the Debye-Hueckel kernel
    Gamma K0(kappa0 u) when sigma = 0. The integrand is a rational function of q^2 whose
    denominator is a polynomial of degree five in q^2 with positive coefficients, so no
    oscillatory quadrature is needed if that structure is used. Reject invalid parameters and
    negative or non-finite distances.

    Args:
        u: array-like of shape (n,) of non-negative finite distances in units of the disk diameter (u = 0 is
            allowed; the kernel is finite there for sigma > 0).
        sigma: non-negative finite float, the splitting length in units of the disk diameter (sigma = 0 is the
            Debye-Hueckel limit of the scheme).
        Gamma: positive finite float, the Coulomb coupling.
        rho: positive finite float, the total reduced ion density rho a^2 (both species together).

    Returns:
        A numpy float64 array of shape (n,) holding the screened long-range kernel Gl(u) = Gamma times the
        integral over q from 0 to infinity of q J0(q u)/(q^2 S(q) + kappa0^2), at the n distances, accurate to a
        relative error of 1e-10; at sigma = 0 it equals Gamma K0(kappa0 u).

    Raises:
        ValueError: if u is not a one-dimensional array of non-negative finite values; if sigma is negative
            or not finite; or if Gamma or rho is not a positive finite number.
    """
    return None
```

### Step 3

short_range_potential

Goal
----
Return the short-range part of the Coulomb potential at the positive distances u: the bare potential -Gamma ln u minus the unscreened long-range potential of step 1, which is Gamma times the integral over q from zero to infinity of (S(q) - 1)/(q S(q)) times J0(q u). This integral converges only conditionally at large q, so a truncated numerical quadrature is not acceptable: the result must be accurate to a relative error of 1e-10 for 0.1 <= u <= 10, must vanish identically when sigma = 0, must carry the logarithmic singularity of the bare potential as u tends to zero, and must decay to zero at distances large compared with sigma. Reject invalid parameters and distances that are not positive and finite.

```python
def short_range_potential(u: "np.ndarray", sigma: float, Gamma: float) -> "np.ndarray":
    """Return the short-range part of the Coulomb potential at the positive distances u: the bare
    potential -Gamma ln u minus the unscreened long-range potential of step 1, which is Gamma
    times the integral over q from zero to infinity of (S(q) - 1)/(q S(q)) times J0(q u). This
    integral converges only conditionally at large q, so a truncated numerical quadrature is not
    acceptable: the result must be accurate to a relative error of 1e-10 for 0.1 <= u <= 10,
    must vanish identically when sigma = 0, must carry the logarithmic singularity of the bare
    potential as u tends to zero, and must decay to zero at distances large compared with sigma.
    Reject invalid parameters and distances that are not positive and finite.

    Args:
        u: array-like of shape (n,) of positive finite distances in units of the disk diameter.
        sigma: non-negative finite float, the splitting length in units of the disk diameter (sigma = 0 makes
            the short-range part vanish identically).
        Gamma: positive finite float, the Coulomb coupling.

    Returns:
        A numpy float64 array of shape (n,) holding the short-range potential vs(u) = Gamma times the integral
        over q from 0 to infinity of (S(q) - 1)/(q S(q)) J0(q u), equal to -Gamma ln u minus the unscreened
        long-range potential, at the n distances, accurate to a relative error of 1e-10 for 0.1 <= u <= 10.

    Raises:
        ValueError: if u is not a one-dimensional array of positive finite values; if sigma is negative or not
            finite; or if Gamma is not a positive finite number.
    """
    return None
```

### Step 4

mayer_functions

Goal
----
Return the Mayer functions of the short-range scheme at the non-negative distances u for a pair of opposite unit charges and for a pair of like unit charges: both equal minus one inside the hard core u < 1, and outside it each is the Boltzmann factor of the total pair potential vs(u) + Gl(u) of the two previous steps minus one, with the sign convention that opposite charges attract, so their Mayer function is positive at contact when the potential is positive. Reject invalid parameters and negative or non-finite distances.

```python
def mayer_functions(u: "np.ndarray", sigma: float, Gamma: float, rho: float) -> "np.ndarray":
    """Return the Mayer functions of the short-range scheme at the non-negative distances u for a
    pair of opposite unit charges and for a pair of like unit charges: both equal minus one
    inside the hard core u < 1, and outside it each is the Boltzmann factor of the total pair
    potential vs(u) + Gl(u) of the two previous steps minus one, with the sign convention that
    opposite charges attract, so their Mayer function is positive at contact when the potential
    is positive. Reject invalid parameters and negative or non-finite distances.

    Args:
        u: array-like of shape (n,) of non-negative finite distances in units of the disk diameter.
        sigma: non-negative finite float, the splitting length in units of the disk diameter (sigma = 0 is the
            Debye-Hueckel limit of the scheme).
        Gamma: positive finite float, the Coulomb coupling.
        rho: positive finite float, the total reduced ion density rho a^2 (both species together).

    Returns:
        A numpy float64 array of shape (2, n): the first row is the opposite-charge Mayer function
        h+-(u) = exp(+w(u)) - 1 and the second row the like-charge Mayer function h++(u) = exp(-w(u)) - 1,
        with w = vs + Gl the total pair potential of the two previous steps, for u >= 1; both rows equal -1
        inside the hard core u < 1.

    Raises:
        ValueError: if u is not a one-dimensional array of non-negative finite values; if sigma is negative
            or not finite; or if Gamma or rho is not a positive finite number.
    """
    return None
```

### Step 5

variational_residual

Goal
----
Return the residual of the variational identity that fixes the splitting length: the integral over q from zero to infinity of q times [hbar++(q) - hbar+-(q) + 2 Glbar(q)] times the sigma-derivative of the long-range potential, where hbar++ and hbar+- are the two-dimensional Fourier transforms of the like-charge and opposite-charge Mayer functions of the previous step, including the hard-core disk whose transform is -2 pi J1(q)/q, and Glbar is the screened kernel of step 1. The residual must be accurate to 1e-8 in absolute terms for a positive sigma; it may be evaluated in reciprocal space or, by Parseval's theorem, as an equivalent radial integral in real space, but its value and sign must be those of the reciprocal-space definition. Reject invalid parameters and a non-positive splitting length.

```python
def variational_residual(sigma: float, Gamma: float, rho: float) -> float:
    """Return the residual of the variational identity that fixes the splitting length: the
    integral over q from zero to infinity of q times [hbar++(q) - hbar+-(q) + 2 Glbar(q)] times
    the sigma-derivative of the long-range potential, where hbar++ and hbar+- are the two-
    dimensional Fourier transforms of the like-charge and opposite-charge Mayer functions of the
    previous step, including the hard-core disk whose transform is -2 pi J1(q)/q, and Glbar is
    the screened kernel of step 1. The residual must be accurate to 1e-8 in absolute terms for a
    positive sigma; it may be evaluated in reciprocal space or, by Parseval's theorem, as an
    equivalent radial integral in real space, but its value and sign must be those of the
    reciprocal-space definition. Reject invalid parameters and a non-positive splitting length.

    Args:
        sigma: positive finite float, the trial splitting length in units of the disk diameter.
        Gamma: positive finite float, the Coulomb coupling.
        rho: positive finite float, the total reduced ion density rho a^2.

    Returns:
        A Python float, the residual of the variational identity in its reciprocal-space definition: the
        integral over q from 0 to infinity of q [hbar++(q) - hbar+-(q) + 2 Glbar(q)] d vlbar/d sigma, with the
        two-dimensional transforms of the Mayer functions including the hard-core disk (transform
        -2 pi J1(q)/q); accurate to 1e-8 in absolute terms. It is negative for a splitting length below the
        variational value and positive above it.

    Raises:
        ValueError: if sigma is not a positive finite number (sigma = 0 is rejected here); or if Gamma or rho
            is not a positive finite number.
    """
    return None
```

### Step 6

splitting_length

Goal
----
Return the splitting length sigma at which the variational residual of the previous step vanishes, for the given coupling and density. Bracket the sign change of the residual by scanning splitting lengths between 0.05 and 5 disk diameters, then converge the root to an absolute tolerance of 1e-12 with a bracketing root finder; the residual has a single sign change on that range for the states considered here. Reject invalid parameters and raise an error if no sign change is found.

```python
def splitting_length(Gamma: float, rho: float) -> float:
    """Return the splitting length sigma at which the variational residual of the previous step
    vanishes, for the given coupling and density. Bracket the sign change of the residual by
    scanning splitting lengths between 0.05 and 5 disk diameters, then converge the root to an
    absolute tolerance of 1e-12 with a bracketing root finder; the residual has a single sign
    change on that range for the states considered here. Reject invalid parameters and raise an
    error if no sign change is found.

    Args:
        Gamma: positive finite float, the Coulomb coupling.
        rho: positive finite float, the total reduced ion density rho a^2.

    Returns:
        A Python float, the variational splitting length sigma* in units of the disk diameter: the root of the
        variational residual of the previous step, bracketed on a scan of [0.05, 5] and converged to an absolute
        tolerance of 1e-12.

    Raises:
        ValueError: if Gamma or rho is not a positive finite number.
        RuntimeError: if the residual has no sign change on [0.05, 5].
    """
    return None
```

### Step 7

pair_convolutions

Goal
----
Return the first-order correlation corrections T++(r) and T+-(r) at the distances r >= 1 for a symmetric 1:1 electrolyte in which each species has number density rho/2. For species i and j, T_ij is the sum over the third species n of n_n times the two-dimensional convolution of the Mayer functions h_in and h_nj, minus q_i q_j q_n^2 n_n times the convolution of the screened kernel Gl with itself, where the convolution of two radial functions f and g is the integral over the plane of f(r') g(|r - r'|) and the Mayer functions include the hard-core disk on which they equal minus one. The result must be accurate to 1e-9 in absolute terms; the convolutions of functions that jump at the contact distance have to be evaluated with the geometry of the overlapping disks made explicit, not by a truncated Fourier quadrature. Reject invalid parameters and distances below one or non-finite.

```python
def pair_convolutions(r: "np.ndarray", sigma: float, Gamma: float, rho: float) -> "np.ndarray":
    """Return the first-order correlation corrections T++(r) and T+-(r) at the distances r >= 1 for
    a symmetric 1:1 electrolyte in which each species has number density rho/2. For species i
    and j, T_ij is the sum over the third species n of n_n times the two-dimensional convolution
    of the Mayer functions h_in and h_nj, minus q_i q_j q_n^2 n_n times the convolution of the
    screened kernel Gl with itself, where the convolution of two radial functions f and g is the
    integral over the plane of f(r') g(|r - r'|) and the Mayer functions include the hard-core
    disk on which they equal minus one. The result must be accurate to 1e-9 in absolute terms;
    the convolutions of functions that jump at the contact distance have to be evaluated with
    the geometry of the overlapping disks made explicit, not by a truncated Fourier quadrature.
    Reject invalid parameters and distances below one or non-finite.

    Args:
        r: array-like of shape (n,) of finite distances r >= 1 in units of the disk diameter.
        sigma: non-negative finite float, the splitting length in units of the disk diameter (sigma = 0 is the
            Debye-Hueckel limit of the scheme).
        Gamma: positive finite float, the Coulomb coupling.
        rho: positive finite float, the total reduced ion density rho a^2 (both species together).

    Returns:
        A numpy float64 array of shape (2, n): the first row is T++(r) = (rho/2)[h++ * h++ + h+- * h+- -
        2 Gl * Gl] and the second row T+-(r) = rho [h++ * h+- + Gl * Gl], with * the two-dimensional
        convolution and the Mayer functions equal to -1 inside the hard core, at the n distances, accurate to
        1e-9 in absolute terms.

    Raises:
        ValueError: if r is not a one-dimensional array of finite values with every entry >= 1; if sigma is
            negative or not finite; or if Gamma or rho is not a positive finite number.
    """
    return None
```

### Step 8

structure_factors

Goal
----
Return the partial structure factors S++(q) and S+-(q) and the charge-charge structure factor SZZ(q) at wavenumbers q in (0, 20]. The total correlation functions are H_ij = h_ij + (1 + h_ij) T_ij with the Mayer functions of step 4 and the corrections of step 7, and they equal minus one inside the hard core. With mole fractions one half, S_ij(q) is delta_ij/2 plus rho/4 times the two-dimensional Fourier transform of H_ij, including the hard-core disk, and SZZ is the charge-weighted combination of the partial structure factors normalised so that it tends to one for an ideal gas of point charges. The transforms must be accurate to 1e-8 in absolute terms, which requires the correlation functions on a radial grid extending far enough for the screened tails to have decayed. Reject invalid parameters and wavenumbers outside (0, 20].

```python
def structure_factors(q: "np.ndarray", sigma: float, Gamma: float, rho: float) -> "np.ndarray":
    """Return the partial structure factors S++(q) and S+-(q) and the charge-charge structure
    factor SZZ(q) at wavenumbers q in (0, 20]. The total correlation functions are H_ij = h_ij +
    (1 + h_ij) T_ij with the Mayer functions of step 4 and the corrections of step 7, and they
    equal minus one inside the hard core. With mole fractions one half, S_ij(q) is delta_ij/2
    plus rho/4 times the two-dimensional Fourier transform of H_ij, including the hard-core
    disk, and SZZ is the charge-weighted combination of the partial structure factors normalised
    so that it tends to one for an ideal gas of point charges. The transforms must be accurate
    to 1e-8 in absolute terms, which requires the correlation functions on a radial grid
    extending far enough for the screened tails to have decayed. Reject invalid parameters and
    wavenumbers outside (0, 20].

    Args:
        q: array-like of shape (n,) of finite wavenumbers in (0, 20], in units of the inverse disk diameter.
        sigma: non-negative finite float, the splitting length in units of the disk diameter (sigma = 0 is the
            Debye-Hueckel limit of the scheme).
        Gamma: positive finite float, the Coulomb coupling.
        rho: positive finite float, the total reduced ion density rho a^2 (both species together).

    Returns:
        A numpy float64 array of shape (3, n) whose rows are S++(q), S+-(q) and SZZ(q) at the n wavenumbers:
        S_ij = delta_ij/2 + (rho/4) Hbar_ij(q) with H_ij = h_ij + (1 + h_ij) T_ij (equal to -1 inside the hard
        core) and SZZ = 1 + (rho/2)[Hbar++ - Hbar+-]; accurate to 1e-8 in absolute terms.

    Raises:
        ValueError: if q is not a one-dimensional array of finite values inside (0, 20]; if sigma is negative
            or not finite; or if Gamma or rho is not a positive finite number.
    """
    return None
```

### Step 9

excess_energy

Goal
----
Return three values of the reduced excess energy per ion, beta u_ex divided by rho: the energy of the scheme with the correlation corrections switched off, in which the total correlation functions are replaced by the Mayer functions; the energy at first order, with the total correlation functions H_ij of the previous step; and the Debye-Hueckel closed form (Gamma/2) K0(kappa0). The excess energy is half the sum over ordered pairs of species of the product of densities, the Coulomb potential and the pair distribution function integrated over the plane; with the hard core, electroneutrality and the 1:1 symmetry it reduces to (pi Gamma rho/2) times the integral from contact to infinity of u ln u times the difference between the opposite-charge and the like-charge total correlation functions. The first two values must be accurate to 1e-7 in absolute terms. Reject invalid parameters.

```python
def excess_energy(Gamma: float, rho: float, sigma: float) -> "np.ndarray":
    """Return three values of the reduced excess energy per ion, beta u_ex divided by rho: the
    energy of the scheme with the correlation corrections switched off, in which the total
    correlation functions are replaced by the Mayer functions; the energy at first order, with
    the total correlation functions H_ij of the previous step; and the Debye-Hueckel closed form
    (Gamma/2) K0(kappa0). The excess energy is half the sum over ordered pairs of species of the
    product of densities, the Coulomb potential and the pair distribution function integrated
    over the plane; with the hard core, electroneutrality and the 1:1 symmetry it reduces to (pi
    Gamma rho/2) times the integral from contact to infinity of u ln u times the difference
    between the opposite-charge and the like-charge total correlation functions. The first two
    values must be accurate to 1e-7 in absolute terms. Reject invalid parameters.

    Args:
        Gamma: positive finite float, the Coulomb coupling.
        rho: positive finite float, the total reduced ion density rho a^2.
        sigma: non-negative finite float, the splitting length in units of the disk diameter (sigma = 0 is the
            Debye-Hueckel limit, in which the uncorrected energy reproduces the closed form to second order in
            the coupling).

    Returns:
        A numpy float64 array of shape (3,) holding, in order, the reduced excess energy per ion beta u_ex/rho
        with the correlation corrections switched off (H replaced by h), the first-order energy with
        H_ij = h_ij + (1 + h_ij) T_ij, and the Debye-Hueckel closed form (Gamma/2) K0(kappa0); the first two
        accurate to 1e-7 in absolute terms.

    Raises:
        ValueError: if Gamma or rho is not a positive finite number; or if sigma is negative or not finite.
    """
    return None
```

### Step 10

coulomb_liquid_audit

Goal
----
Assemble the audit of the self-consistent scheme at coupling Gamma and density rho. Row 0: the variational splitting length, then the three excess-energy values of step 9 at that splitting length. Row 1: the reduced specific heat of the first-order scheme, which is the excess energy minus Gamma times its derivative with respect to Gamma at fixed density, with the splitting length re-solved at every coupling and the derivative converged to 1e-8 by Richardson extrapolation of central differences; the Debye-Hueckel specific heat (Gamma/4) kappa0 K1(kappa0); the opposite-charge structure factor at q = 2; and the charge structure factor at q = 0.5. Row 2: the coupling gamma0 in the bracket [gamma_lo, gamma_hi] at which the first-order excess energy changes sign at this density, converged to 1e-10 with a bracketing root finder over the full chain; the variational splitting length at gamma0; the total pair potential vs + Gl at contact for the input state; and T++ at contact for the input state. Row 3: the screened kernel Gl at u = 0; the variational residual at the trial splitting length 0.7; and the opposite-charge and like-charge Mayer functions at contact, all for the input state. Every entry must be produced by the corresponding earlier step. Reject invalid parameters, a bracket that is not ordered and positive, and a bracket on which the energy does not change sign.

```python
def coulomb_liquid_audit(Gamma: float, rho: float, gamma_lo: float, gamma_hi: float) -> "np.ndarray":
    """Assemble the audit of the self-consistent scheme at coupling Gamma and density rho. Row 0:
    the variational splitting length, then the three excess-energy values of step 9 at that
    splitting length. Row 1: the reduced specific heat of the first-order scheme, which is the
    excess energy minus Gamma times its derivative with respect to Gamma at fixed density, with
    the splitting length re-solved at every coupling and the derivative converged to 1e-8 by
    Richardson extrapolation of central differences; the Debye-Hueckel specific heat (Gamma/4)
    kappa0 K1(kappa0); the opposite-charge structure factor at q = 2; and the charge structure
    factor at q = 0.5. Row 2: the coupling gamma0 in the bracket [gamma_lo, gamma_hi] at which
    the first-order excess energy changes sign at this density, converged to 1e-10 with a
    bracketing root finder over the full chain; the variational splitting length at gamma0; the
    total pair potential vs + Gl at contact for the input state; and T++ at contact for the
    input state. Row 3: the screened kernel Gl at u = 0; the variational residual at the trial
    splitting length 0.7; and the opposite-charge and like-charge Mayer functions at contact,
    all for the input state. Every entry must be produced by the corresponding earlier step.
    Reject invalid parameters, a bracket that is not ordered and positive, and a bracket on
    which the energy does not change sign.

    Args:
        Gamma: positive finite float, the Coulomb coupling of the input state.
        rho: positive finite float, the total reduced ion density rho a^2 of the input state.
        gamma_lo: positive finite float, the lower end of the coupling bracket for the sign change of the
            first-order excess energy at density rho.
        gamma_hi: finite float greater than gamma_lo, the upper end of that bracket.

    Returns:
        A numpy float64 array of shape (4, 4). Row 0: the variational splitting length, the uncorrected
        excess energy, the first-order excess energy and the Debye-Hueckel energy at the input state. Row 1:
        the first-order reduced specific heat CV = E - Gamma dE/dGamma at fixed density (splitting length
        re-solved at every coupling, derivative converged to 1e-8), the Debye-Hueckel specific heat
        (Gamma/4) kappa0 K1(kappa0), S+-(q = 2) and SZZ(q = 0.5). Row 2: the sign-change coupling gamma0 of
        the first-order energy in [gamma_lo, gamma_hi] (converged to 1e-10), the variational splitting length
        at gamma0, the total pair potential vs + Gl at contact and T++ at contact for the input state. Row 3:
        Gl at u = 0, the variational residual at the trial splitting length 0.7, and the opposite-charge and
        like-charge Mayer functions at contact, all for the input state.

    Raises:
        ValueError: if Gamma or rho is not a positive finite number; or if the bracket does not satisfy
            0 < gamma_lo < gamma_hi with both ends finite.
        RuntimeError: if the first-order energy does not change sign on the bracket.
    """
    return None
```
