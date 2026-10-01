# Chemistry-Computational_Chemistry-89

## Background

Many chemical reactions at low temperature proceed by quantum tunnelling through the barrier rather than by passage over it. Hydrogen-atom and proton transfers are the classic cases: well below the temperature at which classical barrier crossing freezes out, measured rate constants exceed transition-state estimates by orders of magnitude, and the apparent activation energy flattens. Rate theories for this regime must therefore treat the nuclear motion quantum mechanically while remaining affordable enough to use with potential energy surfaces obtained from electronic-structure calculations.

Tunnelling is not confined to the reaction coordinate. As bonds break and form, the vibrations orthogonal to the reaction path change frequency, soften and become anharmonic, and the path that dominates tunnelling need not follow the minimum-energy path. Treating those transverse vibrations as harmonic oscillators of fixed frequency, as conventional rate theories do, is therefore a real approximation rather than a formality, and its error grows as the anharmonicity varies along the path.

Because of this, low-dimensional model surfaces with known quantum-mechanical rate constants are the standard proving ground for tunnelling rate theories. Models in which a vibration orthogonal to the reaction coordinate changes character along the path are especially informative, since they separate effects that come from the transverse motion from those that come from the barrier itself.

## Problem

Semiclassical instanton theory predicts thermal rate constants for reactions dominated by quantum tunnelling from a single optimal tunnelling path.

A particle of mass 1836.15267 mₑ, the same for both coordinates, moves in a plane with mass-scaled coordinates x and y (atomic units throughout, ħ = k_B = 1). Along x it crosses a standard asymmetric Eckart barrier whose symmetric hyperbolic-secant-squared component has amplitude 0.01135 E_h and full width at half maximum 1.2956192 a₀, and whose product asymptote lies 0.00485 E_h below the reactant asymptote, the two components sharing one range; the reactant asymptote is the zero of energy, reactants lie at x → −∞ and products at x → +∞. At every x the motion along y is a Morse oscillator with harmonic frequency 0.00415 E_h whose minimum lies at y = 0 and contributes no energy there; its dimensionless anharmonicity constant, the ratio of the usual spectroscopic constants ωₑxₑ and ωₑ, equals 0.0138 in the asymptotic channels and rises to 0.0735 at x = 0, following a Gaussian in x that is centred at the origin and has standard deviation 0.565 a₀. The surface is fixed: none of its parameters scales with ħ. The inverse temperature is 2975 E_h⁻¹.

Compute the thermal rate constant for reaction from the reactant channel to the product channel, defined as a flux per unit length of reactant density along x with the reactant vibration thermally populated. Expand that rate constant asymptotically in ħ at fixed βħ, keep its complete first-order relative correction, and resum that correction exponentially instead of adding it as a partial sum.

Report the rate constant in cm molecule⁻¹ s⁻¹, taking one atomic unit of velocity to be 2.18769126364 × 10⁸ cm s⁻¹, as a single number with four significant figures, converged to that precision.

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

Surface derivative tensors

Goal
----
Evaluate derivative tensors, of orders zero to four, of the two-dimensional reaction surface of the problem statement at a set of points.

```python
def potential_derivative_tensor(points: "np.ndarray", order: int, params: dict) -> "np.ndarray":
    '''Derivative tensor of the reaction surface at each point.

    Parameters
    ----------
    points : np.ndarray
        Array of shape (n, 2) holding the coordinates (x, y) in bohr of
        n >= 1 points. A single point may be given with shape (2,); it is
        treated as n = 1.
    order : int
        Derivative order k, an integer in 0..4.
    params : dict
        Surface parameters in atomic units, with keys "V0" (amplitude of the
        symmetric hyperbolic-secant-squared component of the Eckart barrier,
        hartree), "a" (the range parameter of that barrier, the length that
        scales x in it, bohr), "V_inf" (product asymptote relative to the
        reactant asymptote, hartree), "m" (mass shared by both coordinates,
        electron masses), "omega_e" (harmonic frequency of the Morse
        vibration, hartree), "chi_inf" and "chi_0" (its dimensionless
        anharmonicity constant in the asymptotic channels and at x = 0), and
        "sigma_e" (standard deviation of the Gaussian in x along which that
        constant varies, bohr).

    Returns
    -------
    tensor : np.ndarray
        Array of shape (n,) + (2,) * k. Entry [p, i_1, ..., i_k] is the
        k-th partial derivative of the surface with respect to
        q_{i_1}, ..., q_{i_k} at point p, where q_0 = x and q_1 = y
        (hartree / bohr^k). For k = 0 the shape is (n,) and the entries are
        energies measured from the reactant asymptote. The tensor is
        symmetric in its derivative indices.

    Raises
    ------
    ValueError
        If order is not an integer in 0..4 or points cannot be read as an
        (n, 2) array with n >= 1.
    '''
    return tensor
```

### Step 2

Discretized tunnelling orbit

Goal
----
Locate the discretized periodic tunnelling orbit of the reaction surface at a given inverse temperature, represented by N beads in imaginary time.

```python
def ring_polymer_instanton(beta: float, n_beads: int, params: dict) -> "np.ndarray":
    '''Converged N-bead tunnelling orbit of the reaction surface.

    Parameters
    ----------
    beta : float
        Inverse temperature in 1/hartree (hbar = k_B = 1). The imaginary-time
        step of the chain is beta / n_beads.
    n_beads : int
        Number of beads N, an even integer >= 4. The chain is closed: bead
        N - 1 is a neighbour of bead 0.
    params : dict
        Surface parameters, with the keys used by the surface derivative
        tensors (atomic units).

    Returns
    -------
    beads : np.ndarray
        Array of shape (N, 2) with the (x, y) coordinates in bohr of the
        beads, in folded order: bead N-1-i equals bead i, and beads
        0..N/2-1 run with strictly decreasing x from the product-side turning
        region to the reactant-side turning region. The configuration is
        converged to numerical precision, the final Newton step being below
        1e-10 times the barrier range parameter.

    Raises
    ------
    ValueError
        If n_beads is not an even integer >= 4, beta is not positive, or the
        search does not reach a delocalized, x-ordered stationary orbit, for
        example because the temperature is at or above the crossover
        temperature and the beads collapse onto the barrier top.
    '''
    return beads
```

### Step 3

Leading-order instanton rate

Goal
----
Compute the leading-order semiclassical instanton rate constant of the reaction surface from a converged discretized tunnelling orbit.

```python
def leading_order_instanton_rate(beads: "np.ndarray", beta: float, params: dict) -> float:
    '''Leading-order instanton rate constant for a given tunnelling orbit.

    Parameters
    ----------
    beads : np.ndarray
        Array of shape (N, 2), N >= 4, with the (x, y) bead coordinates in
        bohr of a converged orbit at this beta. The chain is closed: bead
        N - 1 is a neighbour of bead 0.
    beta : float
        Inverse temperature in 1/hartree (hbar = k_B = 1).
    params : dict
        Surface parameters, with the keys used by the surface derivative
        tensors (atomic units).

    Returns
    -------
    k0 : float
        The leading-order rate constant in atomic units, that is bohr per
        atomic unit of time, being a flux per unit length of reactant density
        along x with the reactant vibration thermally populated.

    Raises
    ------
    ValueError
        If beads is not an (N, 2) array with N >= 4.
    '''
    return k0
```

### Step 4

Transverse fluctuation propagator

Goal
----
Build the imaginary-time fluctuation propagator of one coordinate about a discretized orbit, given the curvature of the potential in that coordinate at each bead.

```python
def ring_polymer_propagator(curvatures: "np.ndarray", beta: float, m: float) -> "np.ndarray":
    '''Inverse action Hessian of one fluctuation coordinate on a closed chain.

    Parameters
    ----------
    curvatures : np.ndarray
        One-dimensional array of length N >= 3 with the second derivative of
        the potential with respect to the fluctuation coordinate, in
        hartree / bohr^2, at each of the N beads, in chain order.
    beta : float
        Inverse temperature in 1/hartree (hbar = 1); the imaginary-time step
        is beta / N.
    m : float
        Mass of the coordinate in electron masses.

    Returns
    -------
    G : np.ndarray
        Symmetric (N, N) array, the inverse of the action Hessian, in
        bohr^2 per unit of action. Entry [i, j] refers to beads i and j of
        the same chain order as `curvatures`.

    Raises
    ------
    ValueError
        If curvatures is not one-dimensional with N >= 3, beta or m is not
        positive, or the action Hessian is not positive definite.
    '''
    return G
```

### Step 5

Anharmonic fluctuation correction

Goal
----
Compute the first-order, order-hbar relative correction to the Gaussian fluctuation integral of one coordinate that arises from its anharmonicity at the beads.

```python
def anharmonic_fluctuation_correction(propagator: "np.ndarray", cubic: "np.ndarray", quartic: "np.ndarray", beta: float) -> float:
    '''First-order relative correction from bead-local cubic and quartic anharmonicity.

    Parameters
    ----------
    propagator : np.ndarray
        Symmetric positive-definite (N, N) array, the fluctuation propagator
        of the coordinate on the closed chain, in bohr^2 per unit of action.
    cubic : np.ndarray
        Length-N array of third derivatives of the potential with respect to
        the fluctuation coordinate at each bead, in hartree / bohr^3, in the
        same chain order as the propagator.
    quartic : np.ndarray
        Length-N array of the corresponding fourth derivatives, in
        hartree / bohr^4.
    beta : float
        Inverse temperature in 1/hartree (hbar = 1); the imaginary-time step
        of the chain is beta / N.

    Returns
    -------
    correction : float
        The first-order relative correction, dimensionless, to be added to 1
        in the corrected fluctuation integral. It is the sum of the
        contributions of the two derivative orders.

    Raises
    ------
    ValueError
        If propagator is not square, cubic and quartic are not
        one-dimensional arrays of its size, or beta is not positive.
    '''
    return correction
```

### Step 6

Reactant partition-function correction

Goal
----
Compute the first-order, order-hbar relative correction to the reactant partition function of the reaction surface.

```python
def reactant_partition_correction(beta: float, n_beads: int, params: dict) -> float:
    '''First-order relative correction to the reactant partition function.

    Parameters
    ----------
    beta : float
        Inverse temperature in 1/hartree (hbar = k_B = 1).
    n_beads : int
        Number of beads N >= 3 of the chain used for the reactant channel,
        matching the chain used for the tunnelling orbit.
    params : dict
        Surface parameters, with the keys used by the surface derivative
        tensors (atomic units).

    Returns
    -------
    correction : float
        The first-order relative correction to the reactant partition
        function, dimensionless, defined with the same sign convention as
        the correction about the tunnelling orbit, that is as the term to be
        added to 1.

    Raises
    ------
    ValueError
        If n_beads is not an integer >= 3, or beta, the vibrational
        frequency or the reactant-channel anharmonicity constant is not
        positive.
    '''
    return correction
```

### Step 7

Reaction-coordinate rate correction

Goal
----
Compute the complete first-order, order-hbar relative correction to the thermal rate constant of motion on a one-dimensional Eckart barrier, at fixed thermal time.

```python
def eckart_tunneling_correction(beta: float, V0: float, V_inf: float, a: float, m: float) -> float:
    '''First-order relative rate correction for a one-dimensional Eckart barrier.

    Parameters
    ----------
    beta : float
        Inverse temperature in 1/hartree (hbar = k_B = 1). It must exceed the
        crossover value of this barrier.
    V0 : float
        Amplitude of the symmetric hyperbolic-secant-squared component of the
        barrier, in hartree; V0 > 0.
    V_inf : float
        Product asymptote relative to the reactant asymptote, in hartree,
        with -4 V0 < V_inf < 4 V0 so that the barrier has an interior
        maximum.
    a : float
        Range parameter of the barrier, the length that scales the
        coordinate in it, in bohr; a > 0.
    m : float
        Mass of the coordinate in electron masses; m > 0.

    Returns
    -------
    correction : float
        The complete first-order relative correction for this coordinate,
        dimensionless, defined as the term to be added to 1 in the corrected
        rate constant.

    Raises
    ------
    ValueError
        If V0, a or m is not positive, V_inf is outside (-4 V0, 4 V0), or
        beta is at or below the crossover value, where no tunnelling orbit
        exists.
    '''
    return correction
```

### Step 8

Corrected resummed rate constant

Goal
----
Assemble the first-order corrected, exponentially resummed instanton rate constant of the reaction surface at a given inverse temperature.

```python
def corrected_instanton_rate(beta: float, n_beads: int, params: dict) -> float:
    '''First-order corrected, exponentially resummed instanton rate constant.

    Parameters
    ----------
    beta : float
        Inverse temperature in 1/hartree (hbar = k_B = 1), below the
        crossover temperature of the barrier.
    n_beads : int
        Number of beads N, an even integer, used for the tunnelling orbit and
        for every chain-based quantity derived from it. Larger N approaches
        the continuum limit.
    params : dict
        Surface parameters, with the keys used by the surface derivative
        tensors (atomic units).

    Returns
    -------
    k : float
        The first-order corrected, exponentially resummed rate constant in
        cm molecule^-1 s^-1, that is a flux per unit length of reactant
        density along x with the reactant vibration thermally populated.

    Raises
    ------
    ValueError
        If any component rejects its input: an invalid bead number, a
        temperature at or above the crossover temperature, or invalid
        surface parameters.
    '''
    return k
```
