# Chemistry-Quantum_Chemistry-45

## Background

Light-harvesting antennae, molecular aggregates and organic photovoltaic films all move excitation energy through networks of chromophores whose mutual couplings span orders of magnitude. Where a group of pigments is strongly coupled, the excitation is delocalised over it and moves coherently; between such groups the coupling is weak and the transfer is incoherent, so the whole assembly is usefully coarse-grained into segments connected by rate constants. The standard way to obtain those rate constants treats the coupling between segments as a perturbation and keeps the lowest non-vanishing order, which is quadratic in that coupling. In the time domain this yields a compact object: an emission-side propagator for the donor, an absorption-side propagator for the acceptor, and a single time integral over their overlap, weighted by the inter-segment coupling matrix.

That construction rests on a separation of timescales. The donor and acceptor manifolds must lose mutual coherence quickly compared with the time it takes population to move between them, and each segment must equilibrate internally before any transfer occurs. When both hold, one rate constant per direction is enough. The trouble is that real antennae sit uncomfortably close to the boundary: inter-segment couplings of a few tens of wavenumbers, environments whose correlation functions carry components from tens of femtoseconds to picoseconds, and energy gaps of the same order as the disorder. Benchmarks against numerically exact propagation of the full electron-nuclear problem show the perturbative rate drifting high as the coupling is increased, and the discrepancy appears well before the transfer becomes visibly coherent, so it cannot be diagnosed by inspecting the population dynamics alone.

The natural repair is to keep the next term in the same perturbation series. Extracting a rate from it is considerably subtler than at the lowest order. At second order the acceptor population simply grows linearly in time once the initial coherence has relaxed, and reading off a rate is a matter of differentiating it. At the next order the population acquires more than one kind of time dependence, and not all of what appears there is new physics: some of it describes sequences of events that the coupled rate equations, once solved, already produce on their own. Deciding which part of the higher-order term may legitimately be added to the lowest-order rate constant, and which part must be discarded because it is already counted elsewhere, is the central difficulty of the construction and has no counterpart at second order.

The mathematical machinery is shared with nonlinear optical spectroscopy. A quartic term carries four interaction vertices and therefore several distinct time orderings, each drawn as its own double-sided Feynman diagram, and each contributes an integral over three time intervals rather than one. Two of those intervals are coherence intervals during which the two manifolds are propagated in opposite directions and dephase relative to each other; the interval between them is a waiting time during which both sides of the diagram sit in the same manifold. This is exactly the structure of a two-dimensional electronic spectrum, with the inter-segment coupling matrix playing the role that the transition dipole plays there.

What decides whether the higher-order term matters is the memory of the system, meaning everything that makes the two coherence intervals fail to be statistically independent: the correlation times of the bath, and, in multi-chromophore segments, relaxation among the excitons. Long memory keeps the two intervals correlated for longer and enlarges the correction, while in the opposite limit of a bath with no memory at all the correction disappears entirely. This matters in practice for environments whose correlation functions are multi-exponential, as those extracted from molecular dynamics usually are, because the components at opposite ends of that range do not play the same role in the theory and cannot be handled by the same machinery.

## Problem

Excitation energy transfer between two weakly coupled chromophores is almost always modelled with a rate that is second order in the electronic coupling between them, but that description degrades once the coupling grows or the environment holds its memory for as long as the donor-acceptor coherence survives, and in that intermediate regime the second-order rate is systematically too fast; the next non-vanishing term in the perturbation series is fourth order in the coupling, is negative, and restores agreement with numerically exact propagation over a far wider range of couplings. Your task is to compute, for the donor-acceptor pair specified below, the transfer rate that includes this fourth-order correction, in ps^-1.

Treat both chromophores as two-level sites in a single-excitation manifold with hbar = 1 and every energy expressed as an angular frequency, converting from cm^-1 with 2*pi*c and c = 2.99792458e-5 cm/fs. Each site energy fluctuates independently with an autocorrelation function that is a sum of exponential components, so the transfer depends on the pair only through the difference of the two site energies and enters the theory as a single second-order-cumulant lineshape function; a component whose correlation time reaches or exceeds a stated threshold is not memory at all on the timescale of the transfer, and must be excluded from that lineshape and carried instead as frozen Gaussian broadening of the donor-acceptor gap. The fourth-order object is a response function of three time intervals, two intervals during which the donor and acceptor manifolds carry mutual coherence separated by a waiting interval during which they do not.

Build the lineshape from the memory components alone and obtain the second-order rate from the time integral of the second-order response. Then construct the fourth-order response, reduce it to a rate density that depends only on the waiting time, extract from that density the part that constitutes a genuine fourth-order transfer event, and add the resulting correction to the second-order rate at each frozen value of the gap. Average that corrected rate over the static gap distribution by Gauss-Hermite quadrature against a standard normal weight, using the probabilists' nodes and weights of numpy.polynomial.hermite_e.hermegauss divided by sqrt(2*pi).

Use these values:
- both chromophores carry the same three bath components, with RMS site-energy fluctuation amplitudes 90, 110 and 70 cm^-1 and correlation times 40, 180 and 3000 fs
- a component counts as static when its correlation time is at least 1000 fs
- mean acceptor-minus-donor energy gap 120 cm^-1, electronic coupling between the chromophores 26 cm^-1
- every coherence-interval integral on a uniform grid from 0 to 200 fs in steps of 2 fs, the waiting-time integral on a uniform grid from 0 to 2400 fs in steps of 4 fs, both endpoints included, all quadratures by the trapezoidal rule
- 9 Gauss-Hermite nodes for the static average

Report the disorder-averaged fourth-order-corrected transfer rate in ps^-1, and in your reasoning also give:
- the RMS static width of the gap in cm^-1, and the dephasing lineshape evaluated at 100 fs
- the expression you used for the second-order rate, and its value at the mean gap in ps^-1
- the equilibration rate between the two chromophores at the mean gap in ps^-1
- the coefficient of the term quadratic in time in the quartic contribution to the acceptor population, at the mean gap, in ps^-2
- the numerical prefactor multiplying the fourth power of the coupling in the fourth-order rate density, with the reason for its sign and its magnitude
- the oscillatory factor carried by each of the two contributions to that density
- the rate density at the mean gap at waiting times of 0, 400 and 2400 fs in ps^-2, the constant it settles on in ps^-2, how closely the value at 2400 fs agrees with that constant, and any relation you find between that constant and the second-order rate
- the fourth-order correction in ps^-1 both at the mean gap and at a gap one static standard deviation above it
- the disorder-averaged second-order rate and the disorder-averaged fourth-order correction in ps^-1, and the ratio of those two averages

Using external scientific literature beyond the supplied paper, also:
- explain the limitation of representing strong static disorder as slow dynamic disorder, and report the sampling strategy recommended for recovering the corresponding distribution of transfer rates
- state the published thermal correction used to enforce detailed balance between two segments without changing their equilibration rate, and apply it to the high-temperature, equal-population, one-site-per-segment model used here
- report the short- and long-distance power-law regimes published for excitation transfer between B850 rings, explain the physical origin assigned to the short-distance regime, and state whether the present two-single-site model tests that mechanism
- report the three published nonzero relaxation eigenmode rates for the FMO complex and place the final corrected rate calculated here within that set

Cite the external source used for each of these four literature-based points.

Output Format Requirements:
Emit <final_answer> immediately, then <reasoning>. Do not write a long derivation before the tags.
You must emit exactly one finite decimal inside <final_answer>...</final_answer>, even if the value is approximate or you are unsure.
Rules:
- The tags are required. Do not omit them or leave them empty.
- The tagged value must be a finite decimal (examples: 0.4847, 0.22, 1.05). Not NaN, not Inf, not a fraction string, not a vector, not prose.
- Put only that one number between the tags. No units, no words, no extra lines.
Keep <reasoning> concise while reporting all the requested numerical and literature-based quantities.
Do not paste the input matrices, full coefficient vectors, per-iteration paths, or per-fold candidate tables.

## Output format

```
## Output format

Scientific reasoning wrapped in <reasoning>...</reasoning> tags. Include enough intermediate calculations to justify the deterministic computation, without turning the response into a general pipeline summary. A single final numeric answer wrapped in <final_answer>...</final_answer> tags.
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

dephasing_lineshape

Goal
----
Build the lineshape function g(t) that controls how fast the donor-acceptor

coherence dephases, keeping only the bath components that carry genuine memory.



The two chromophores each sit in their own overdamped Brownian-oscillator bath, so every

site energy fluctuates with an autocorrelation function that is a sum of exponentials,

<dE(t) dE(0)> = sum_j sigma_j^2 exp(-t / tau_j). What controls the transfer is the

difference of the two site energies, so when the two baths are uncorrelated their

variances add component by component and the second-order cumulant gives one lineshape

function per component.



A component whose correlation time is comparable to, or longer than, the population

transfer itself is not memory at all: on the timescale of the transfer that component is

frozen, and folding it into g(t) makes the higher-order waiting-time integral fail to

converge. Such components must be excluded here and carried separately as static

inhomogeneous broadening of the donor-acceptor gap. The threshold is supplied as

tau_static_fs, and a component is treated as static when tau_j >= tau_static_fs.



Energies arrive as angular wavenumbers in cm^-1 and must be converted to rad/fs with

2 * pi * c, c = 2.99792458e-5 cm/fs, before g(t) is assembled; g(t) itself is dimensionless.



Formulas

--------

Lambda_j = (2 pi c sigma_D,j)^2 + (2 pi c sigma_A,j)^2          [rad^2 fs^-2]

g(t) = sum_{j : tau_j < tau_static} Lambda_j tau_j^2 (exp(-t/tau_j) + t/tau_j - 1)



Returns

-------

np.ndarray of shape (len(t_fs),): the dimensionless lineshape g(t)

```python
import numpy as np


def dephasing_lineshape(t_fs, sigma_donor, sigma_acceptor, tau_fs,
                        tau_static_fs: float) -> np.ndarray:
    '''Second-order cumulant lineshape for the donor-acceptor energy gap.

    Parameters
    ----------
    t_fs : array_like
        Times in fs at which g is wanted; every entry must be finite and >= 0.
    sigma_donor, sigma_acceptor : array_like
        RMS site-energy fluctuation amplitude of each bath component, in cm^-1,
        for the donor and the acceptor. Same length as tau_fs; entries must be
        finite and >= 0.
    tau_fs : array_like
        Bath correlation time of each component, in fs; entries must be finite and > 0.
    tau_static_fs : float
        Components with tau_j >= tau_static_fs are treated as static disorder and
        contribute nothing to g(t). Must be finite and > 0.

    Returns
    -------
    np.ndarray
        Shape (len(t_fs),) array of dimensionless g(t) values, float dtype.

    Raises
    ------
    ValueError
        If the three component lists differ in length, if any component list is
        empty, if any entry is non-finite, if any sigma is negative, if any tau is
        non-positive, if tau_static_fs is non-positive, or if any requested time is
        negative or non-finite.
    '''
    return g_values


#
```

### Step 2

static_gap_width

Goal
----
Collect the bath components that were excluded from the lineshape and turn them

into a single inhomogeneous width for the donor-acceptor energy gap.



A component whose correlation time reaches or exceeds the threshold does not dephase the

donor-acceptor coherence; it simply offsets the gap by an amount that is frozen for the whole

transfer event. Because the donor and acceptor baths are uncorrelated, the variances of those

frozen offsets add, and because the transfer depends only on the difference of the two site

energies, the donor and acceptor contributions add on the same footing.



The result is the standard deviation of a Gaussian distribution of the gap. Every quantity

here stays in cm^-1: no conversion to angular frequency is applied at this step, because the

width is later added directly to the mean gap, which is also quoted in cm^-1.



Formulas

--------

sigma_static = sqrt( sum_{j : tau_j >= tau_static} ( sigma_D,j^2 + sigma_A,j^2 ) )



Returns

-------

float, the RMS static spread of the donor-acceptor gap in cm^-1 (zero when no component

is static)

```python
import numpy as np


def static_gap_width(sigma_donor, sigma_acceptor, tau_fs, tau_static_fs: float) -> float:
    '''RMS inhomogeneous width of the donor-acceptor gap from the near-static bath modes.

    Parameters
    ----------
    sigma_donor, sigma_acceptor : array_like
        RMS site-energy fluctuation amplitude of each bath component, in cm^-1.
        Same length as tau_fs; entries must be finite and >= 0.
    tau_fs : array_like
        Bath correlation time of each component, in fs; entries must be finite and > 0.
    tau_static_fs : float
        Components with tau_j >= tau_static_fs count as static. Must be finite and > 0.

    Returns
    -------
    float
        Standard deviation of the gap distribution in cm^-1; exactly 0.0 when no
        component reaches the threshold.

    Raises
    ------
    ValueError
        If the three component lists differ in length, if any list is empty, if any
        entry is non-finite, if any sigma is negative, if any tau is non-positive, or
        if tau_static_fs is non-positive.
    '''
    return width_cm


#
```

### Step 3

second_order_response

Goal
----
Assemble the complex response function whose time integral gives the standard

lowest-order incoherent transfer rate between the two chromophores.



Once the donor-acceptor coherence has been reduced to a single dephasing lineshape, the

response of the pair is the product of two factors: a phase that winds at the mean gap

between the two site energies, and the decay envelope exp(-g(t)) built in step 1. The

response is normalised so that R(0) = 1 exactly, and it carries no factor of the coupling:

the coupling enters later, where the rate is formed.



The sign convention matters and is fixed here: the gap is defined as acceptor minus donor,

and the phase advances as exp(-i omega t) with omega = 2 pi c times that gap. Reversing the

sign of the gap conjugates the response, which is what makes the backward rate equal to the

forward rate at infinite temperature.



Formulas

--------

omega = 2 pi c * gap_cm,  c = 2.99792458e-5 cm/fs

R(t) = exp(-g(t)) * exp(-i omega t)



Returns

-------

np.ndarray of shape (len(t_fs),), complex dtype

```python
import numpy as np


def second_order_response(t_fs, sigma_donor, sigma_acceptor, tau_fs,
                          tau_static_fs: float, gap_cm: float) -> np.ndarray:
    '''Complex second-order transfer response of the donor-acceptor pair.

    Parameters
    ----------
    t_fs : array_like
        Times in fs; every entry must be finite and >= 0.
    sigma_donor, sigma_acceptor : array_like
        RMS site-energy fluctuation amplitudes per bath component, in cm^-1.
    tau_fs : array_like
        Bath correlation times per component, in fs.
    tau_static_fs : float
        Components with tau_j >= tau_static_fs are static and excluded from g(t).
    gap_cm : float
        Mean acceptor-minus-donor excitation energy gap, in cm^-1. May be any finite
        real number, including negative.

    Returns
    -------
    np.ndarray
        Shape (len(t_fs),) complex array with R(0) = 1.

    Raises
    ------
    ValueError
        On any of the bath or time faults listed for the lineshape step, or if
        gap_cm is not finite.
    '''
    return response


#
```

### Step 4

second_order_rate

Goal
----
Turn the response function into the lowest-order incoherent transfer rate for one

frozen value of the donor-acceptor gap.



The rate is twice the real part of the time integral of the response, scaled by the square

of the inter-site electronic coupling. Working in units where hbar = 1 and every energy is

an angular frequency, the coupling is converted from cm^-1 with the same 2 pi c factor used

for the gap, so the bracket has units of inverse femtoseconds; the result is reported in

inverse picoseconds.



The time integral runs over a uniform grid from 0 to t_max_fs with spacing dt_fs, evaluated

with the trapezoidal rule. The grid is fixed by n = round(t_max_fs / dt_fs) + 1 points placed

by numpy.linspace between 0 and t_max_fs inclusive, so that the endpoint is hit exactly.

t_max_fs must be long enough that the response has decayed; that is a property of the bath,

not something this routine checks.



At infinite temperature the backward rate is obtained by flipping the sign of the gap, and

because that conjugates the response it leaves the real part unchanged. The equilibration

rate between the pair is therefore exactly twice this number.



Formulas

--------

J_ang = 2 pi c * j_cm;  S = trapz(R(t), t) over the grid

k2 = 2 * J_ang^2 * Re(S) * 1e3          [ps^-1]



Returns

-------

float, the second-order transfer rate in ps^-1

```python
import numpy as np


def second_order_rate(sigma_donor, sigma_acceptor, tau_fs, tau_static_fs: float,
                      gap_cm: float, j_cm: float, t_max_fs: float, dt_fs: float) -> float:
    '''Lowest-order incoherent transfer rate at one frozen gap.

    Parameters
    ----------
    sigma_donor, sigma_acceptor : array_like
        RMS site-energy fluctuation amplitudes per bath component, in cm^-1.
    tau_fs : array_like
        Bath correlation times per component, in fs.
    tau_static_fs : float
        Components with tau_j >= tau_static_fs are static and excluded from g(t).
    gap_cm : float
        Mean acceptor-minus-donor gap, in cm^-1.
    j_cm : float
        Electronic coupling between the two chromophores, in cm^-1. Any finite real
        value is allowed; only its square enters.
    t_max_fs : float
        Upper limit of the coherence-time integral, in fs; must be finite and > 0.
    dt_fs : float
        Grid spacing of that integral, in fs; must be finite, > 0 and <= t_max_fs.

    Returns
    -------
    float
        Transfer rate in ps^-1.

    Raises
    ------
    ValueError
        On any bath fault listed for the lineshape step, if gap_cm or j_cm is not
        finite, or if t_max_fs or dt_fs is non-positive or dt_fs exceeds t_max_fs.
    '''
    return rate_ps


#
```

### Step 5

nonrephasing_kernel

Goal
----
Build the non-rephasing member of the fourth-order transfer response, the one whose

two coherence intervals accumulate phase in the same direction.



In this diagram the phase acquired during the first coherence interval is not undone during

the second, so the two intervals appear as a sum in the oscillatory factor and any static

spread in the gap is reinforced rather than cancelled.



The three time arguments are the two coherence intervals t1 and t3, during which the donor

and acceptor manifolds are propagated in opposite directions and the pair loses its mutual

coherence, and the waiting interval tw between them, during which both sides of the diagram

sit in the same manifold. The three arguments broadcast against one another, so passing a

column, a row and a depth vector returns the full three-dimensional block in one call.



Six evaluations of the lineshape enter, one at each of the three intervals on its own and one

at each of the three consecutive sums t1+tw, tw+t3 and t1+tw+t3, each carrying a coefficient

of either plus or minus one. Those six signs are not given here and must be worked out. They

are fixed uniquely by two limits the kernel has to satisfy, and both are worth checking once

the kernel is written.



Formulas

--------

omega = 2 pi c * gap_cm,  c = 2.99792458e-5 cm/fs

Phi = exp(-i omega (t1 + t3)) * exp( s1 g(t1) + s3 g(t3) + sw g(tw)

                                     + s1w g(t1+tw) + sw3 g(tw+t3) + s13 g(t1+tw+t3) )

with each s in {+1, -1}, determined by requiring, with R the second-order response of step 3,

    Phi -> R(t1) R(t3)   as tw -> infinity, for every t1 and t3

    Phi  = R(t1 + t3)    at tw = 0,         for every t1 and t3



Returns

-------

np.ndarray, complex, of the broadcast shape of t1_fs, tw_fs and t3_fs

```python
import numpy as np


def nonrephasing_kernel(t1_fs, tw_fs, t3_fs, sigma_donor, sigma_acceptor, tau_fs,
                        tau_static_fs: float, gap_cm: float) -> np.ndarray:
    '''non-rephasing kernel of the fourth-order transfer response.

    Parameters
    ----------
    t1_fs, tw_fs, t3_fs : array_like
        First coherence interval, waiting interval and second coherence interval,
        in fs. They must broadcast together; every entry must be finite and >= 0.
    sigma_donor, sigma_acceptor : array_like
        RMS site-energy fluctuation amplitudes per bath component, in cm^-1.
    tau_fs : array_like
        Bath correlation times per component, in fs.
    tau_static_fs : float
        Components with tau_j >= tau_static_fs are static and excluded from g(t).
    gap_cm : float
        Mean acceptor-minus-donor gap, in cm^-1.

    Returns
    -------
    np.ndarray
        Complex array of the broadcast shape.

    Raises
    ------
    ValueError
        On any bath fault listed for the lineshape step, if gap_cm is not finite,
        if any time is negative or non-finite, or if the three time arguments do
        not broadcast together.
    '''
    return kernel


#
```

### Step 6

rephasing_kernel

Goal
----
Build the rephasing member of the fourth-order transfer response, the one whose two

coherence intervals accumulate phase in opposite directions.



In this diagram the phase acquired during the first coherence interval is undone during the

second, so the two intervals appear as a difference in the oscillatory factor. Every sign in

the waiting-time-dependent group of lineshape terms is also reversed relative to its

non-rephasing partner.



The three time arguments are the two coherence intervals t1 and t3, during which the donor

and acceptor manifolds are propagated in opposite directions and the pair loses its mutual

coherence, and the waiting interval tw between them, during which both sides of the diagram

sit in the same manifold. The three arguments broadcast against one another, so passing a

column, a row and a depth vector returns the full three-dimensional block in one call.



The same six lineshape evaluations enter as in the non-rephasing kernel, at the three

intervals on their own and at the three consecutive sums. The two terms that depend on t1 or

t3 alone keep the signs they carry there; every term that involves the waiting interval

carries the opposite sign. The signs are not given here and must be worked out, and the

long-waiting-time limit below fixes them uniquely.



Formulas

--------

omega = 2 pi c * gap_cm,  c = 2.99792458e-5 cm/fs

Phi = exp(-i omega (t3 - t1)) * exp( s1 g(t1) + s3 g(t3) + sw g(tw)

                                     + s1w g(t1+tw) + sw3 g(tw+t3) + s13 g(t1+tw+t3) )

with each s in {+1, -1}, determined by requiring, with R the second-order response of step 3,

    Phi -> conj(R(t1)) R(t3)   as tw -> infinity, for every t1 and t3

Unlike its non-rephasing partner this kernel is not a function of a single combined time at

tw = 0, so that limit gives no further constraint here.



Returns

-------

np.ndarray, complex, of the broadcast shape of t1_fs, tw_fs and t3_fs

```python
import numpy as np


def rephasing_kernel(t1_fs, tw_fs, t3_fs, sigma_donor, sigma_acceptor, tau_fs,
                     tau_static_fs: float, gap_cm: float) -> np.ndarray:
    '''rephasing kernel of the fourth-order transfer response.

    Parameters
    ----------
    t1_fs, tw_fs, t3_fs : array_like
        First coherence interval, waiting interval and second coherence interval,
        in fs. They must broadcast together; every entry must be finite and >= 0.
    sigma_donor, sigma_acceptor : array_like
        RMS site-energy fluctuation amplitudes per bath component, in cm^-1.
    tau_fs : array_like
        Bath correlation times per component, in fs.
    tau_static_fs : float
        Components with tau_j >= tau_static_fs are static and excluded from g(t).
    gap_cm : float
        Mean acceptor-minus-donor gap, in cm^-1.

    Returns
    -------
    np.ndarray
        Complex array of the broadcast shape.

    Raises
    ------
    ValueError
        On any bath fault listed for the lineshape step, if gap_cm is not finite,
        if any time is negative or non-finite, or if the three time arguments do
        not broadcast together.
    '''
    return kernel


#
```

### Step 7

waiting_time_profile

Goal
----
Integrate the two kernels over both coherence intervals to obtain the fourth-order

rate density as a function of the waiting time alone.



At each waiting time the two coherence intervals are integrated out over the same uniform

grid used for the second-order rate, the two kernels are summed, and the result is scaled into

a rate density. The scale factor is a signed integer A times the fourth power of the coupling.

A is not given here and must be worked out. Its sign follows from the fact that this order of

perturbation theory removes population the lowest order has already counted. Its magnitude

follows from how many distinct time orderings of the four interaction vertices survive in

total, given that the two kernels summed inside F cover only those orderings in which the

waiting interval sits on the acceptor, and that for a pair of single chromophores the

remaining orderings contribute equally to those two.



The two coherence integrals both use the trapezoidal rule on the grid of

n = round(t_max_fs / dt_fs) + 1 points from 0 to t_max_fs, and the waiting times are the

n_w = round(tw_max_fs / dtw_fs) + 1 points from 0 to tw_max_fs. Only the real part survives.

Units are chosen so that the profile is a rate per unit time in ps^-2: multiply the doubly

integrated kernel, which carries fs^2, by 1e6.



This profile does not decay to zero. It approaches a non-zero constant, and what that

constant is, and why it has to be removed before the waiting-time integral is taken, is the

subject of the next two steps.



Formulas

--------

J_ang = 2 pi c * j_cm

F(tw) = trapz_t1 trapz_t3 [ Phi_NR(t1,tw,t3) + Phi_R(t1,tw,t3) ]

profile(tw) = A * J_ang^4 * Re F(tw) * 1e6           [ps^-2],  A a signed integer to determine



Returns

-------

np.ndarray of shape (round(tw_max_fs/dtw_fs) + 1,), float dtype, in ps^-2

```python
import numpy as np


def waiting_time_profile(sigma_donor, sigma_acceptor, tau_fs, tau_static_fs: float,
                         gap_cm: float, j_cm: float, t_max_fs: float, dt_fs: float,
                         tw_max_fs: float, dtw_fs: float) -> np.ndarray:
    '''Fourth-order rate density as a function of the waiting time.

    Parameters
    ----------
    sigma_donor, sigma_acceptor : array_like
        RMS site-energy fluctuation amplitudes per bath component, in cm^-1.
    tau_fs : array_like
        Bath correlation times per component, in fs.
    tau_static_fs : float
        Components with tau_j >= tau_static_fs are static and excluded from g(t).
    gap_cm : float
        Mean acceptor-minus-donor gap, in cm^-1.
    j_cm : float
        Electronic coupling between the two chromophores, in cm^-1.
    t_max_fs, dt_fs : float
        Upper limit and spacing of both coherence-interval integrals, in fs.
    tw_max_fs, dtw_fs : float
        Upper limit and spacing of the waiting-time grid, in fs.

    Returns
    -------
    np.ndarray
        Real array over the waiting-time grid, in ps^-2.

    Raises
    ------
    ValueError
        On any bath fault listed for the lineshape step, if gap_cm or j_cm is not
        finite, or if either grid limit is non-positive, either spacing is
        non-positive, or a spacing exceeds its own limit.
    '''
    return profile_ps2


#
```

### Step 8

plateau_constant

Goal
----
Evaluate the constant that the fourth-order rate profile approaches once the two

coherence intervals stop knowing about each other.



Push the waiting time to infinity and every lineshape term that couples the two coherence

intervals drops out, so the doubly integrated kernel separates into a product of

single-interval integrals and the constant follows in closed form, without touching the

numerical waiting-time grid at all. Work out what each kernel leaves behind in that limit,

sum the two, and apply the same scale factor and unit conversion the profile uses. Evaluating

the profile at a large waiting time instead is not an acceptable substitute: the point of this

step is the closed form.



Physically this plateau is not a fourth-order transfer event at all. It is a forward hop

followed by an independent backward hop, two lowest-order processes that happen to occur in

sequence, and the kinetic model already accounts for it. It grows without bound as the

waiting window is extended, so it has to be removed before the waiting-time integral is taken.



The single-interval integral uses exactly the same grid and trapezoidal rule as the

second-order rate, so that the plateau computed here and the profile computed in the

previous step agree in the long-waiting-time limit to within the discretisation error.



Formulas

--------

S  = trapz(R(t), t) over the coherence grid          [fs, complex]

J_ang = 2 pi c * j_cm

C = A * J_ang^4 * Re( ... ) * 1e6                    [ps^-2]

with A the same signed integer as in the previous step and the bracket built from S alone.



Returns

-------

float, the plateau of the fourth-order rate profile in ps^-2 (negative for any non-zero

coupling)

```python
import numpy as np


def plateau_constant(sigma_donor, sigma_acceptor, tau_fs, tau_static_fs: float,
                     gap_cm: float, j_cm: float, t_max_fs: float, dt_fs: float) -> float:
    '''Long-waiting-time limit of the fourth-order rate profile.

    Parameters
    ----------
    sigma_donor, sigma_acceptor : array_like
        RMS site-energy fluctuation amplitudes per bath component, in cm^-1.
    tau_fs : array_like
        Bath correlation times per component, in fs.
    tau_static_fs : float
        Components with tau_j >= tau_static_fs are static and excluded from g(t).
    gap_cm : float
        Mean acceptor-minus-donor gap, in cm^-1.
    j_cm : float
        Electronic coupling between the two chromophores, in cm^-1.
    t_max_fs, dt_fs : float
        Upper limit and spacing of the coherence-time integral, in fs.

    Returns
    -------
    float
        The plateau in ps^-2.

    Raises
    ------
    ValueError
        On any bath fault listed for the lineshape step, if gap_cm or j_cm is not
        finite, or if t_max_fs or dt_fs is non-positive or dt_fs exceeds t_max_fs.
    '''
    return plateau_ps2


#
```

### Step 9

fourth_order_rate

Goal
----
Subtract the plateau from the rate profile and integrate what is left over the

waiting time, giving the correction that is added to the lowest-order rate.



Only the part of the profile that decays carries genuine fourth-order transfer. The constant

part is a pair of independent lowest-order hops and belongs to the quadratic-in-time growth

of the acceptor population, not to a rate, so integrating it would give a number that simply

grows with the waiting window rather than converging. Removing the plateau first leaves an

integrand that decays to zero on the timescale of the bath memory, and its integral is finite.



The waiting-time integral uses the trapezoidal rule on the same grid as the profile, but the

waiting times must be expressed in picoseconds so that a profile in ps^-2 integrates to a rate

in ps^-1. The window must be long enough that the integrand has reached zero; when it has not,

the result drifts with tw_max_fs, which is the practical signal that the window is too short.



The correction is negative whenever the coupling is non-zero, so including it always lowers

the predicted transfer rate. Its magnitude grows as the fourth power of the coupling and

increases with the bath memory, so it matters most when the segments are close together and

the environment is slow.



Formulas

--------

k4 = trapz( profile(tw) - C, tw/1000 )               [ps^-1]



Returns

-------

float, the fourth-order rate correction in ps^-1

```python
import numpy as np


def fourth_order_rate(sigma_donor, sigma_acceptor, tau_fs, tau_static_fs: float,
                      gap_cm: float, j_cm: float, t_max_fs: float, dt_fs: float,
                      tw_max_fs: float, dtw_fs: float) -> float:
    '''Fourth-order correction to the transfer rate at one frozen gap.

    Parameters
    ----------
    sigma_donor, sigma_acceptor : array_like
        RMS site-energy fluctuation amplitudes per bath component, in cm^-1.
    tau_fs : array_like
        Bath correlation times per component, in fs.
    tau_static_fs : float
        Components with tau_j >= tau_static_fs are static and excluded from g(t).
    gap_cm : float
        Mean acceptor-minus-donor gap, in cm^-1.
    j_cm : float
        Electronic coupling between the two chromophores, in cm^-1.
    t_max_fs, dt_fs : float
        Upper limit and spacing of both coherence-interval integrals, in fs.
    tw_max_fs, dtw_fs : float
        Upper limit and spacing of the waiting-time grid, in fs.

    Returns
    -------
    float
        The fourth-order rate correction in ps^-1; negative for any non-zero coupling.

    Raises
    ------
    ValueError
        On any bath fault listed for the lineshape step, if gap_cm or j_cm is not
        finite, or if either grid limit is non-positive, either spacing is
        non-positive, or a spacing exceeds its own limit.
    '''
    return k4_ps


#
```

### Step 10

corrected_transfer_rate

Goal
----
Run the whole pipeline and report the transfer rate that includes the fourth-order

correction, averaged over the frozen disorder in the donor-acceptor gap.



Split the bath once: the components with short correlation times dephase the pair and go into

the lineshape, the near-static ones do not dephase anything and instead spread the gap. Then,

for each gap drawn from that spread, form the lowest-order rate and its fourth-order

correction and add them, and average the sum over the gap distribution.



The average is a Gauss-Hermite quadrature against a standard normal weight. Use the

probabilists' nodes and weights from numpy.polynomial.hermite_e.hermegauss, which integrate

against exp(-x^2/2) and whose weights sum to sqrt(2 pi); dividing them by sqrt(2 pi) makes

them a probability average. Node i samples the gap at gap_cm + sigma_static * x_i.



Both rates are linear in the population, so averaging them separately and adding is the same

as averaging their sum; what must not be done is to average the gap first and evaluate one

rate at the mean, because both rates are strongly curved in the gap.



At every node the constant that the fourth-order rate density settles on is also evaluated and

checked against minus twice the square of the second-order rate at that same gap. The two are

equal identically, because both are built from the same integrated second-order response, so

the check costs nothing and catches a wrong diagram multiplicity or a wrong overall prefactor

in either branch. A relative disagreement above 1e-6 is treated as a fault and rejected.



Formulas

--------

sigma_static from the near-static components; x_i, w_i from hermegauss(n_nodes), w_i /= sqrt(2 pi)

C(gap) = -2 k2(gap)^2 must hold at every node to a relative 1e-6

k = sum_i w_i [ k2(gap_cm + sigma_static x_i) + k4(gap_cm + sigma_static x_i) ]



Returns

-------

float, the disorder-averaged fourth-order-corrected transfer rate in ps^-1

```python
import numpy as np


def corrected_transfer_rate(sigma_donor, sigma_acceptor, tau_fs, tau_static_fs: float,
                            gap_cm: float, j_cm: float, t_max_fs: float, dt_fs: float,
                            tw_max_fs: float, dtw_fs: float, n_nodes: int) -> float:
    '''Disorder-averaged transfer rate including the fourth-order correction.

    Parameters
    ----------
    sigma_donor, sigma_acceptor : array_like
        RMS site-energy fluctuation amplitudes per bath component, in cm^-1.
    tau_fs : array_like
        Bath correlation times per component, in fs.
    tau_static_fs : float
        Components with tau_j >= tau_static_fs are static: they are excluded from the
        lineshape and instead broaden the gap.
    gap_cm : float
        Mean acceptor-minus-donor gap, in cm^-1.
    j_cm : float
        Electronic coupling between the two chromophores, in cm^-1.
    t_max_fs, dt_fs : float
        Upper limit and spacing of both coherence-interval integrals, in fs.
    tw_max_fs, dtw_fs : float
        Upper limit and spacing of the waiting-time grid, in fs.
    n_nodes : int
        Number of Gauss-Hermite nodes for the static-disorder average; must be >= 1.

    Returns
    -------
    float
        The corrected transfer rate in ps^-1.

    Raises
    ------
    ValueError
        On any bath fault listed for the lineshape step, if gap_cm or j_cm is not
        finite, if either grid limit is non-positive, if either spacing is
        non-positive or exceeds its own limit, if n_nodes is not a positive integer,
        or if at any node the plateau of the fourth-order rate density disagrees with
        minus twice the square of the second-order rate by more than a relative 1e-6.
    '''
    return rate_ps


#
```
