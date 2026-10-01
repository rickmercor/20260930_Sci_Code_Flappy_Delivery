# Material_Science-Semiconductor_Materials-5

## Background

Optical excitations in semiconductor quantum dots interact with lattice vibrations. Frequency-selective probes reveal how that environment changes correlations among emitted photons. Their noninvasive limit extracts optical statistics as the probe occupations vanish. Connected correlations isolate statistical dependence beyond contributions involving fewer detection channels.

## Problem

Determine the normalized connected third-order, zero-delay photon coincidence of a coherently driven semiconductor quantum dot in a thermal acoustic-phonon bath, resolved by three finite-bandwidth probes in their noninvasive limit. Use the joint emitter–probe, nonsecular Born–Markov description with its complete complex phonon response. The dot and each probe are two-level systems with ground–excited ordering, rotating-wave exchange coupling, and a classical dot drive; the probes have independent radiative losses and couple optically only to the dot.

In units of ps and angular frequency, the dot's polaron-shifted transition frequency \(\omega_0'\) lies \(\delta=\omega_0'-\omega_L=0.013\,\mathrm{ps}^{-1}\) above the laser frequency \(\omega_L\), the dot's Rabi frequency is \(\Omega=0.22\,\mathrm{ps}^{-1}\), and its population-decay rate is \(\gamma=0.012\,\mathrm{ps}^{-1}\). The probe detunings from the laser, \(\omega_m-\omega_L\), are \((-0.45,-0.64,-0.84)\,\mathrm{ps}^{-1}\), population-decay rates are \((0.025,0.033,0.029)\,\mathrm{ps}^{-1}\), and real exchange couplings are \(\epsilon_m=\lambda r_m\), where \((r_1,r_2,r_3)=(1,1.2,0.9)\) and \(\lambda\to0^+\) has units \(\mathrm{ps}^{-1}\). The phonons couple to the dot's excited-state projector and have \(J(\nu)=\alpha\nu^3e^{-(\nu/\nu_c)^2}\) for \(\nu\ge0\), with \(\alpha=0.027\,\mathrm{ps}^2\), \(\nu_c=2.2\,\mathrm{ps}^{-1}\), \(T=4\,\mathrm K\), and \(k_B/\hbar=0.1309203391\,\mathrm{ps}^{-1}\mathrm K^{-1}\).

Use the shifted-transition convention in which the thermal rate operator is constructed with the stated dot detuning, and the interaction counterterm compensates the static polaron shift while retaining the frequency-dependent dispersive contribution. For this constructed benchmark, “connected” means the joint third cumulant of the three distinct probe occupation numbers divided by the product of their stationary means, with all moments taken in the same joint stationary family before taking \(\lambda\to0^+\). The optical catalog gives a nominal wavelength of 950 nm and independent, background-free photon-retention efficiencies \((0.61,0.48,0.72)\).

Justify the bath reduction and shifted-resonance convention, and report the three normalized pair coincidences and the normalized triple coincidence in the reasoning. Give the final dimensionless connected coincidence to absolute accuracy \(2\times10^{-5}\).

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

joint_operators

Goal
----
Construct the coupled emitter–probe Hamiltonian family.

```python
def joint_operators(delta: float, omega: float, detunings: "np.ndarray", directions: "np.ndarray") -> "np.ndarray":
    """Construct the joint Hamiltonian family and embedded operators.

    Parameters
    ----------
    delta : float
        Detuning omega0'-omega_L of the dot's polaron-shifted transition
        above the laser, in ps^-1.
    omega : float
        Real Rabi frequency in ps^-1; the drive amplitude is omega/2.
    detunings : np.ndarray
        Real shape (N,), 0<=N<=3, probe detunings omega_m-omega_L in ps^-1.
    directions : np.ndarray
        Real dimensionless shape (N,). Probe m has exchange coupling
        lambda*directions[m], where lambda is real and has units ps^-1.

    Returns
    -------
    result : np.ndarray
        Complex shape (N+4,D,D), D=2**(N+1), in order H0, V, A, dot lowering,
        and the N probe lowering operators. H(lambda)=H0+lambda*V is the
        joint rotating-frame Hamiltonian; H0 has units ps^-1 and V is
        dimensionless. A is the dot excited-state projector. Tensor order
        is dot, probe 1, ..., probe N, each in ground, excited order.
    """
    return result
```

### Step 2

bath_response

Goal
----
Evaluate the thermal phonon response on its analytic frequency strip.

```python
def bath_response(frequencies: "np.ndarray", alpha: float, cutoff: float, temperature: float) -> "np.ndarray":
    """Evaluate the causal thermal phonon response and its analytic continuation.

    Parameters
    ----------
    frequencies : np.ndarray
        Complex array of any shape, angular frequencies w in ps^-1,
        with abs(real(w))<=6*cutoff and abs(imag(w))<=pi*theta,
        where theta=(k_B/hbar)*temperature.
    alpha : float
        Nonnegative spectral-density strength in ps^2.
    cutoff : float
        Cutoff in ps^-1, 0.8<=cutoff<=3.
    temperature : float
        Temperature in K, 1<=temperature<=10, using
        k_B/hbar=0.1309203391 ps^-1 K^-1.

    Returns
    -------
    result : np.ndarray
        Complex array with frequencies.shape, in ps^-1. On the real axis,
        F(w) is the causal Abel limit of integral_0^infinity exp(i*w*t)C(t)dt,
        where C is the equilibrium harmonic-bath correlation for
        J(v)=alpha*v^3*exp(-(v/cutoff)^2), v>=0. For complex w, return the
        analytic continuation of this real-axis response across the stated
        strip. Both dispersive and dissipative contributions are included.
        Required error is at most 3e-9*max(1 ps^-1, abs(F(w))).
    """
    return result
```

### Step 3

phonon_rate_series

Goal
----
Expand the paper’s joint-system phonon rate operator in the probe coupling.

```python
def phonon_rate_series(hamiltonian0: "np.ndarray", exchange: "np.ndarray", coupling_operator: "np.ndarray", alpha: float, cutoff: float, temperature: float, order: int) -> "np.ndarray":
    """Compute Taylor coefficients of the complete joint phonon rate operator.

    Parameters
    ----------
    hamiltonian0 : np.ndarray
        Hermitian shape (D,D), in ps^-1, with spectral span <=4*cutoff.
    exchange : np.ndarray
        Dimensionless Hermitian shape (D,D), defining
        H(lambda)=hamiltonian0+lambda*exchange for real lambda in ps^-1.
    coupling_operator : np.ndarray
        Dimensionless Hermitian shape (D,D), the phonon coupling A.
    alpha : float
        Nonnegative acoustic strength in ps^2.
    cutoff : float
        Cutoff in ps^-1, 0.8<=cutoff<=3.
    temperature : float
        Temperature in K, 1<=temperature<=10, with the bath_response units.
    order : int
        Highest Taylor power, 0<=order<=6. Degenerate eigenvalues are allowed.

    Returns
    -------
    result : np.ndarray
        Complex shape (order+1,D,D), the ordinary power-series coefficients
        Z_k in Z(lambda)=sum_k lambda**k Z_k about zero. Here Z(lambda) is
        integral_0^infinity C(t)A(-t;lambda)dt, and A(-t;lambda) evolves
        backward under the complete H(lambda). Coefficient k has units
        ps**(k-1); it includes the derivative factorial denominator.
        Use the same causal bath response as bath_response.
    """
    return result
```

### Step 4

phonon_superoperator

Goal
----
Construct the nonsecular phonon generator series.

```python
def phonon_superoperator(coupling_operator: "np.ndarray", rate_series: "np.ndarray") -> "np.ndarray":
    """Represent the full phonon generator coefficient by coefficient.

    Parameters
    ----------
    coupling_operator : np.ndarray
        Dimensionless Hermitian shape (D,D), the phonon coupling A.
    rate_series : np.ndarray
        Complex shape (K+1,D,D), 0<=K<=6, ordinary Taylor coefficients
        of the joint rate operator for a real coupling scale lambda.

    Returns
    -------
    result : np.ndarray
        Complex shape (K+1,D*D,D*D), ordinary Taylor coefficients of the
        complete nonsecular Born–Markov phonon generator determined by
        the supplied coupling and causal rate operators. Coefficient k
        has units ps**(k-1). The matrix representation stacks density-matrix
        columns, so vec_F(K_k[rho])=result[k]@vec_F(rho).
    """
    return result
```

### Step 5

total_liouvillian

Goal
----
Assemble the shifted-resonance joint generator series.

```python
def total_liouvillian(operators: "np.ndarray", phonon_series: "np.ndarray", gamma: float, widths: "np.ndarray", alpha: float, cutoff: float) -> "np.ndarray":
    """Assemble the joint Liouvillian Taylor series in the shifted convention.

    Parameters
    ----------
    operators : np.ndarray
        Complex shape (N+4,D,D), the joint_operators stack H0,V,A,lowering
        operators, with the stated polaron-shifted dot detuning.
    phonon_series : np.ndarray
        Complex shape (K+1,D*D,D*D), 0<=K<=6, from phonon_superoperator
        for this Hamiltonian family and its complete complex bath response.
    gamma : float
        Positive dot population-decay rate in ps^-1.
    widths : np.ndarray
        Positive shape (N,), probe population-decay rates in ps^-1.
    alpha : float
        Nonnegative strength of the same acoustic bath, in ps^2.
    cutoff : float
        Cutoff of the same bath in ps^-1, 0.8<=cutoff<=3.

    Returns
    -------
    result : np.ndarray
        Complex shape (K+1,D*D,D*D), ordinary coefficients of the complete
        column-vectorized Liouvillian L(lambda). Include coherent dynamics,
        the supplied phonon series, independent radiative losses, and the
        interaction counterterm compensating the static polaron shift.
        A population-decay rate k multiplies c rho c^dagger-{c^dagger c,rho}/2.
        Coefficient j has units ps**(j-1).
    """
    return result
```

### Step 6

stationary_density

Goal
----
Recover the normalized stationary family coefficient by coefficient.

```python
def stationary_density(generators: "np.ndarray") -> "np.ndarray":
    """Solve for the stationary density-matrix Taylor coefficients.

    Parameters
    ----------
    generators : np.ndarray
        Complex shape (K+1,D*D,D*D), 0<=K<=6, ordinary Taylor coefficients
        of a trace- and Hermiticity-preserving generator for real lambda.
        Matrices use column vectorization. L(0) has a unique trace-one
        stationary state and a nonsingular trace-constrained system.

    Returns
    -------
    result : np.ndarray
        Complex shape (K+1,D,D), ordinary coefficients rho_k of the common
        stationary family satisfying L(lambda)vec_F(rho(lambda))=0 and
        Tr(rho(lambda))=1 through order K. Coefficient k is Hermitian and
        has units ps**k; the derivative factorial denominator is included.
    """
    return result
```

### Step 7

subset_spectra

Goal
----
Extract the noninvasive physical spectra for every probe subset.

```python
def subset_spectra(density_series: "np.ndarray", lowering_operators: "np.ndarray", widths: "np.ndarray", directions: "np.ndarray") -> "np.ndarray":
    """Extract noninvasive photon spectra from a stationary density series.

    Parameters
    ----------
    density_series : np.ndarray
        Complex shape (K+1,D,D), ordinary coefficients of the common
        normalized stationary family, with K>=2*N. For each subset of m
        distinct probes its occupation moment starts at order lambda**(2*m).
    lowering_operators : np.ndarray
        Complex shape (N,D,D), 1<=N<=3, distinct probe lowering operators
        embedded in the same basis as density_series.
    widths : np.ndarray
        Positive shape (N,), population-decay rates in ps^-1.
    directions : np.ndarray
        Nonzero real dimensionless shape (N,), with epsilon_m=lambda*r_m.

    Returns
    -------
    result : np.ndarray
        Real shape (2**N-1,), the noninvasive physical photon spectrum for
        each nonempty subset, as obtained from the sensor occupation formula
        in the limit lambda->0. Position mask-1 uses bit0 for probe1.
        Normalization uses a unit-amplitude monochromatic field: at angular
        detuning x, a weak probe has occupation (lambda*r)**2/(x**2+(width/2)**2),
        and its single-photon spectrum integrates to one over x. This
        independent channel calibration also sets each joint spectrum.
        An m-photon spectrum has units ps**m. All moments belong to the
        same stationary family.
    """
    return result
```

### Step 8

connected_coincidence

Goal
----
Compute the noninvasive connected coincidence from the joint stationary family.

```python
def connected_coincidence(parameters: dict) -> float:
    """Compute the noninvasive connected third photon coincidence (orchestrator).

    Parameters
    ----------
    parameters : dict
        Keys delta, omega, detunings, gamma, and widths give the dot
        detuning omega0'-omega_L of its polaron-shifted transition above the
        laser, Rabi frequency, three probe detunings omega_m-omega_L, dot
        population decay, and three probe population decays in ps^-1.
        Decay rates are positive.
        directions is a real length-three sequence of nonzero dimensionless
        exchange ratios, epsilon_m=lambda*directions[m], lambda in ps^-1.
        alpha>=0 is in ps^2, 0.8<=cutoff<=3 is in ps^-1, and temperature
        is in K with 1<=temperature<=10. Use k_B/hbar=0.1309203391 ps^-1 K^-1.
        delta uses the compensated shifted-transition convention. The
        uncoupled joint Hamiltonian has spectral span <=4*cutoff. The
        stationary family is unique near zero, with positive leading
        single-probe occupations.

    Returns
    -------
    result : float
        Native dimensionless scalar: the lambda->0+ limit of the joint
        third cumulant of the three probe occupations divided by their
        stationary means. Required absolute accuracy is 2e-5. Compose
        joint_operators, bath_response through phonon_rate_series,
        phonon_rate_series, phonon_superoperator, total_liouvillian,
        stationary_density, and subset_spectra.
    """
    return result
```
