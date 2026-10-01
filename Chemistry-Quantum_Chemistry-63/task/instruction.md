# Chemistry-Quantum_Chemistry-63

## Background

## Where the Born-Oppenheimer picture assigns a phase

Separating electronic from nuclear motion by the mass ratio produces, for each
electronic state, a potential energy surface and a connection. Transporting a
non-degenerate electronic eigenstate around a closed nuclear loop returns it to
itself multiplied by a phase, the line integral of that connection around the
loop. When the loop encircles a point at which two surfaces touch, the phase is
not an accident of the path: it counts the touching, and in a two-state model it
takes the value pi for any loop that encircles the intersection once. This is
the sign change that Longuet-Higgins and co-workers found in the dynamical
Jahn-Teller problem and that Herzberg and Longuet-Higgins turned into a general
criterion for the presence of an intersection.

## The E x e Jahn-Teller model

The smallest system that shows the effect is a pair of degenerate electronic
states coupled linearly to a pair of degenerate vibrational modes. Writing the
two mode displacements in polar form, the electronic Hamiltonian splits into a
harmonic term proportional to the identity, which shifts both surfaces equally
and therefore cannot affect any electronic state, and a traceless term of
magnitude proportional to the radius whose direction rotates once as the polar
angle advances through 2 pi. The two surfaces form the familiar cone touching at
the origin. Because the traceless term rotates through only half a turn in the
space of electronic states while the nuclear angle turns through a full one, the
real eigenvector of the lower surface comes back to minus itself, and it is that
antiperiodicity which carries the phase.

A constant term added to the traceless part removes the touching and turns the
cone into a pair of separated sheets. The phase then stops being quantised: it
falls continuously to zero as the loop shrinks and rises towards the quantised
value only for loops much larger than the length scale on which the constant
term and the linear term balance. It is also, on such a regularised model, safe
to demand a single valued electronic gauge, because the eigenvector no longer
returns to minus itself; some such demand is essential, since a numerical
eigensolver hands back each eigenvector with an arbitrary phase and any residual
phase freedom makes a derivative meaningless. Fixing one component of the
coefficient vector to be real and non-negative is the cheapest way to do it, and
it is legitimate exactly when that component has no zero on the loop.

## Factorising the molecular wavefunction

The Born-Oppenheimer product is an approximation, but a product ansatz need not
be. Abedi, Maitra and Gross showed that the full molecular wavefunction can be
written exactly as a nuclear factor multiplied by an electronic factor that
depends parametrically on the nuclear coordinates, with the two factors obeying
coupled equations. The nuclear equation looks like an ordinary Schroedinger
equation for a charged particle, with a vector potential given by the connection
of the electronic factor and a scalar potential that reduces to the
Born-Oppenheimer surface plus its diagonal correction when the electronic factor
is a Born-Oppenheimer eigenstate. The electronic equation carries, in addition
to the Born-Oppenheimer Hamiltonian, an operator that encodes the correlation
between the two subsystems and that involves the nuclear factor only through the
gradient of its logarithm, the quantity usually called the nuclear momentum
function. Solving the electronic equation is as expensive as the correlated
problem it came from, which is why approximating it is the practical question.

## What the geometric phase becomes once the nuclei move

Min, Abedi, Kim and Gross asked whether the quantised molecular Berry phase
survives when the nuclei are not clamped, and Requist, Tandetzky and Gross
showed within the exact factorisation that the phase built from the exact
electronic factor is an ordinary geometric phase: it depends on the loop, and
its quantisation in the clamped limit is a property of that limit alone.
Requist, Proetto and Gross later analysed the Berry curvature of the regularised
E x e model asymptotically and established how the phase behaves as the loop
shrinks. What none of that literature supplies is a constructive scheme for
obtaining the correction from Born-Oppenheimer quantities alone.

## The nuclear factor is a solution, not an input

The second of the two coupled equations is the one the nuclear factor obeys, and its shape is
what makes the exact factorization useful: it is an ordinary Schroedinger equation for a
particle in a vector and a scalar potential, both of them functionals of the electronic
factor. The vector potential is the Berry connection, so the nuclear problem is a gauge
problem, and the gauge freedom that was a nuisance for the electronic states becomes a
practical tool here, because a well chosen gauge can make an otherwise two-dimensional
problem separable. The scalar potential is not simply the Born-Oppenheimer surface. Which
further terms belong in it, and with what weight in the mass ratio, is fixed by the
factorization itself rather than chosen, and getting that wrong changes the nuclear state and
therefore everything built on it.

A nuclear state obtained this way is a very different object from a convenient closed-form
ansatz. It knows where the Born-Oppenheimer surface has its minimum, how the angular momentum
pushes it outward, and how much the corrections to the scalar potential move it, and its
logarithmic derivative at a chosen radius, which is what the electronic equation actually
needs, is sensitive to all three.

## Numerical remarks

One fact makes this kind of calculation cheap: a loop integral of a smooth periodic
integrand is a case in which the equally spaced trapezoidal rule converges geometrically
rather than algebraically, so a few hundred angles already exhaust double precision and the
answer stops depending on the grid. A radial eigenvalue problem discretised with the
three-point second difference is tridiagonal, and a tridiagonal solver returns its lowest
eigenpair in a time that grows linearly with the number of grid points, so a grid of several
thousand points is not expensive.

Coordinates chosen for their symmetry repay care. A frame built from the radial and angular
unit vectors of a plane is orthonormal but rotates from point to point, and quantities that
look purely kinematic in a Cartesian frame do not keep the same form in it.

## Problem

Freezing the nuclei assigns the lower member of a degenerate electronic pair a geometric phase that is quantised and completely independent of the path taken around a conical intersection, but that quantisation is an artefact of the Born-Oppenheimer separation itself, and treating the electronic and nuclear factors as genuinely coupled restores an ordinary geometric phase whose size reports directly on the state the nuclei are actually in. The exact electronic factor is as hard to obtain as the fully correlated problem it came from, which is what makes the size of that correction a question worth asking rather than a quantity one reads off; what is wanted here is the leading nonadiabatic correction to the phase for a given vibronic model, nuclear state and closed nuclear contour.

Two degenerate vibrational modes span the nuclear plane R = Q(cos theta, sin theta), and in the basis of the two degenerate electronic states the Born-Oppenheimer Hamiltonian is H(Q, theta) = (K/2) Q^2 * 1 + g Q (sigma_3 cos theta - sigma_1 sin theta) + Delta sigma_2, with K = 1.00, g = 0.60 and Delta = 0.12 in atomic units, the last term lifting the residual degeneracy that the first two leave at the origin. Beyond the Born-Oppenheimer approximation the electronic factor obeys an equation of motion carrying the electron-nuclear correlation operator of the exact factorization, and the electron-to-nuclear mass ratio is mu = 0.01; mu and the nuclear mass below are independent inputs and neither is derived from the other.

The nuclear factor is not prescribed. Take it to be the ground state of the corresponding exact-factorization nuclear equation built on the lower adiabatic state, with nuclear mass M0 = 100.0 and integer angular momentum quantum number m = 3, written in the gauge in which the vector potential is purely angular and depends on the radius alone, and solved on the equally spaced radial grid of 4001 points running from Q = 0.02 to Q = 2.02 with the radial function held at zero at both ends. Fix the phase of every adiabatic electronic state by requiring the second component of its coefficient vector to be real and non-negative. Working to first order in mu on the anticlockwise circle of radius Q = 0.35, which is a point of that grid, report the correction to the geometric phase of the lower adiabatic state as a percentage of its uncorrected value on the same circle. Inside the reasoning tags state the uncorrected geometric phase on that circle, the adiabatic energy gap there, the vector potential you used at the loop, whatever enters the scalar potential of your nuclear equation beyond the Born-Oppenheimer surface and its value at the loop, the ground-state eigenvalue of that nuclear equation and where the state it gives is localised, the nuclear momentum function at the loop, the matrix element of the correlation operator and the correction to the connection both evaluated at the polar angle 0.7 radians, the radius at which the two terms of the vibronic Hamiltonian balance and the unrounded correction to the phase itself, together with the construction you used, the sources you relied on for it, and the limiting and scaling checks you ran.

Output Format Requirements:
Emit <final_answer> immediately, then <reasoning>. Do not write a long derivation before the tags.
You must emit exactly one finite decimal inside <final_answer>...</final_answer>, even if the value is approximate or you are unsure.
Rules:
- The tags are required. Do not omit them or leave them empty.
- The tagged value must be a finite decimal (examples: 0.4847, 0.22, 1.05). Not NaN, not Inf, not a fraction string, not a vector, not prose.
- Put only that one number between the tags. No units, no words, no extra lines.
Use <reasoning> for the chain that produces the number: the scalars named above, the construction, the sources and the checks. Give what a reader needs to reproduce the result and no more.
Do not paste the input matrices, full coefficient vectors, per-iteration paths, or per-fold candidate tables.

## Output format

```
## Output format

Scientific reasoning wrapped in <reasoning>...</reasoning> tags. Include enough intermediate calculations to justify the deterministic computation, without turning the response into a general pipeline summary. A single final numeric answer wrapped in <final_answer>...</final_answer> tags.
```

## Your task

Implement **all 12 functions** below into a single file at `/app/solution.py`. Requirements:

- Use the EXACT function name and signature shown for each step.
- Define every function at module scope (importable).
- Do NOT include tests, example usage, prints, or a `__main__` block.
- Available libraries: numpy, scipy, sympy, pandas, scikit-learn, networkx.

Start your file with these imports:

```python
import numpy as np
```

### Step 1

Born-Oppenheimer Hamiltonian of the vibronic model

Goal
----
Return the real and the imaginary part of the two-state Born-Oppenheimer Hamiltonian of the gapped two-mode vibronic model at one nuclear configuration.

```python
def bo_hamiltonian(Q: float, theta: float, g: float, K: float,
                   Delta: float) -> "np.ndarray":
    '''Electronic Hamiltonian of the model at one nuclear configuration.

    Parameters
    ----------
    Q : float
        Radial nuclear coordinate, strictly positive.
    theta : float
        Polar angle of the nuclear coordinate, in radians.
    g : float
        Linear vibronic coupling strength, strictly positive.
    K : float
        Harmonic force constant of the two normal modes.
    Delta : float
        Strength of the constant term that opens the gap, strictly positive.

    Returns
    -------
    stacked : np.ndarray
        Real array of shape (2, 2, 2). stacked[0] holds the real part and
        stacked[1] the imaginary part of the 2 x 2 electronic Hamiltonian.

    Raises
    ------
    ValueError
        If any argument is not a finite scalar, or if Q, g or Delta is not
        strictly positive.
    '''
    return stacked
```

### Step 2

Phase-fixed adiabatic electronic states

Goal
----
Diagonalise that Hamiltonian and return the two adiabatic states with their phases fixed by requiring the second component of each to be real and non-negative, lower state first.

```python
def adiabatic_states(Q: float, theta: float, g: float, K: float,
                     Delta: float) -> "np.ndarray":
    '''Phase-fixed adiabatic electronic states at one nuclear configuration.

    Parameters
    ----------
    Q : float
        Radial nuclear coordinate, strictly positive.
    theta : float
        Polar angle of the nuclear coordinate, in radians.
    g : float
        Linear vibronic coupling strength, strictly positive.
    K : float
        Harmonic force constant of the two normal modes.
    Delta : float
        Strength of the constant term that opens the gap, strictly positive.

    Returns
    -------
    stacked : np.ndarray
        Real array of shape (2, 2, 2). stacked[0] holds the real part and
        stacked[1] the imaginary part of a 2 x 2 matrix whose column 0 is the
        lower adiabatic state and whose column 1 is the upper adiabatic state.
        The phase of each column is fixed by requiring its second component to
        be real and non-negative.

    Raises
    ------
    ValueError
        If any argument is not a finite scalar, or if Q, g or Delta is not
        strictly positive.
    '''
    return stacked
```

### Step 3

Berry connections of the two adiabatic states

Goal
----
Return the Berry connections of the lower and the upper adiabatic state, in the orthonormal polar frame of the nuclear plane.

```python
def berry_connections(Q: float, theta: float, g: float, K: float,
                      Delta: float) -> "np.ndarray":
    '''Berry connections of the lower and the upper adiabatic state.

    Parameters
    ----------
    Q : float
        Radial nuclear coordinate, strictly positive.
    theta : float
        Polar angle of the nuclear coordinate, in radians.
    g : float
        Linear vibronic coupling strength, strictly positive.
    K : float
        Harmonic force constant of the two normal modes.
    Delta : float
        Strength of the constant term that opens the gap, strictly positive.

    Returns
    -------
    connections : np.ndarray
        Real array of shape (2, 2). Row 0 is the connection of the lower state
        and row 1 the connection of the upper state; within a row column 0 is
        the radial component and column 1 the angular component, both in the
        orthonormal polar frame.

    Raises
    ------
    ValueError
        If any argument is not a finite scalar, or if Q, g or Delta is not
        strictly positive.
    '''
    return connections
```

### Step 4

Geometric phase of the lower state on the loop

Goal
----
Integrate the Berry connection of the lower state anticlockwise around the circle of radius Q on an equally spaced grid and return the geometric phase.

```python
def bo_geometric_phase(Q: float, g: float, K: float, Delta: float,
                       n_theta: int) -> float:
    '''Geometric phase of the lower adiabatic state on the loop of radius Q.

    Parameters
    ----------
    Q : float
        Radius of the circular nuclear contour, strictly positive.
    g : float
        Linear vibronic coupling strength, strictly positive.
    K : float
        Harmonic force constant of the two normal modes.
    Delta : float
        Strength of the constant term that opens the gap, strictly positive.
    n_theta : int
        Number of equally spaced quadrature points on the loop, at least 8.
        The grid is theta_k = 2 pi k / n_theta for k = 0 .. n_theta - 1.

    Returns
    -------
    phase : float
        The geometric phase, in radians.

    Raises
    ------
    ValueError
        If any of Q, g, K or Delta is not a finite scalar, if Q, g or Delta is
        not strictly positive, or if n_theta is not an integer of at least 8.
    '''
    return phase
```

### Step 5

Derivative coupling between the adiabatic states

Goal
----
Return the matrix element of the nuclear gradient taken between the upper adiabatic state on the left and the lower one on the right, in the orthonormal polar frame.

```python
def interstate_coupling(Q: float, theta: float, g: float, K: float,
                        Delta: float) -> "np.ndarray":
    '''Nuclear derivative coupling from the lower to the upper adiabatic state.

    Parameters
    ----------
    Q : float
        Radial nuclear coordinate, strictly positive.
    theta : float
        Polar angle of the nuclear coordinate, in radians.
    g : float
        Linear vibronic coupling strength, strictly positive.
    K : float
        Harmonic force constant of the two normal modes.
    Delta : float
        Strength of the constant term that opens the gap, strictly positive.

    Returns
    -------
    coupling : np.ndarray
        Real array of shape (2, 2). Row 0 holds the real part and row 1 the
        imaginary part; within a row column 0 is the radial component and
        column 1 the angular component, in the orthonormal polar frame. The
        quantity represented is the matrix element of the nuclear gradient with
        the upper state on the left and the lower state on the right.

    Raises
    ------
    ValueError
        If any argument is not a finite scalar, or if Q, g or Delta is not
        strictly positive.
    '''
    return coupling
```

### Step 6

Divergence of the derivative-coupling field

Goal
----
Return the divergence, in the nuclear plane, of the derivative-coupling field of the previous step.

```python
def coupling_divergence(Q: float, theta: float, g: float, K: float,
                        Delta: float) -> "np.ndarray":
    '''Divergence of the derivative-coupling field at one nuclear point.

    Parameters
    ----------
    Q : float
        Radial nuclear coordinate, strictly positive.
    theta : float
        Polar angle of the nuclear coordinate, in radians.
    g : float
        Linear vibronic coupling strength, strictly positive.
    K : float
        Harmonic force constant of the two normal modes.
    Delta : float
        Strength of the constant term that opens the gap, strictly positive.

    Returns
    -------
    divergence : np.ndarray
        Real array of shape (2,) holding the real part and then the imaginary
        part of the divergence of the derivative-coupling field, the field
        being the one returned by the previous step.

    Raises
    ------
    ValueError
        If any argument is not a finite scalar, or if Q, g or Delta is not
        strictly positive.
    '''
    return divergence
```

### Step 7

Diagonal correction to the potential energy surface

Goal
----
Return the term the exact factorization adds to the Born-Oppenheimer surface in the scalar potential of the nuclear equation, at one nuclear configuration.

```python
def diagonal_correction(Q: float, theta: float, g: float, K: float,
                        Delta: float, M0: float) -> float:
    '''Diagonal correction to the Born-Oppenheimer surface at one configuration.

    Parameters
    ----------
    Q : float
        Radial nuclear coordinate, strictly positive.
    theta : float
        Polar angle of the nuclear coordinate, in radians.
    g : float
        Linear vibronic coupling strength, strictly positive.
    K : float
        Harmonic force constant of the two normal modes.
    Delta : float
        Strength of the constant term that opens the gap, strictly positive.
    M0 : float
        Nuclear mass carried by the two normal modes, strictly positive. The
        mass-ratio parameter that weights this term in the nuclear equation is
        not applied here.

    Returns
    -------
    correction : float
        The diagonal correction at that nuclear configuration.

    Raises
    ------
    ValueError
        If any of Q, theta, g, K, Delta is not a finite scalar, if Q, g or
        Delta is not strictly positive, or if M0 is not a finite positive
        scalar.
    '''
    return correction
```

### Step 8

Ground state of the nuclear factor

Goal
----
Solve the radial nuclear equation of the exact factorization on the given grid and return that grid together with the ground-state radial function on it.

```python
def nuclear_radial_state(g: float, K: float, Delta: float, m: int, M0: float,
                         mu: float, Q_min: float, Q_max: float,
                         n_grid: int) -> "np.ndarray":
    '''Ground state of the radial nuclear equation on the given grid.

    Parameters
    ----------
    g : float
        Linear vibronic coupling strength, strictly positive.
    K : float
        Harmonic force constant of the two normal modes.
    Delta : float
        Strength of the constant term that opens the gap, strictly positive.
    m : int
        Integer angular momentum quantum number of the nuclear factor.
    M0 : float
        Nuclear mass carried by the two normal modes, strictly positive.
    mu : float
        Electron-to-nuclear mass ratio, strictly positive.
    Q_min, Q_max : float
        Ends of the radial grid, with 0 < Q_min < Q_max. The radial function is
        held at zero outside them.
    n_grid : int
        Number of equally spaced grid points, at least 16, so that the spacing
        is (Q_max - Q_min) / (n_grid - 1).

    Returns
    -------
    state : np.ndarray
        Real array of shape (2, n_grid). Row 0 is the radial grid and row 1 is
        the radial part of the nuclear factor on it, real, scaled so that its
        largest magnitude is one and so that it is positive there.

    Raises
    ------
    ValueError
        If g, K or Delta is not a finite scalar, if g or Delta is not strictly
        positive, if m is not an integer scalar, if M0 or mu is not a finite
        positive scalar, if the grid ends do not satisfy 0 < Q_min < Q_max, or
        if n_grid is not an integer of at least 16.
    '''
    return state
```

### Step 9

Nuclear momentum function at the loop

Goal
----
Return the nuclear momentum function of the exact-factorization framework at the loop radius, built from the solved nuclear ground state.

```python
def nuclear_momentum(Q: float, g: float, K: float, Delta: float, m: int,
                     M0: float, mu: float, Q_min: float, Q_max: float,
                     n_grid: int) -> "np.ndarray":
    '''Nuclear momentum function at the loop radius.

    Parameters
    ----------
    Q : float
        Loop radius. It must coincide with one of the interior points of the
        radial grid defined by Q_min, Q_max and n_grid.
    g : float
        Linear vibronic coupling strength, strictly positive.
    K : float
        Harmonic force constant of the two normal modes.
    Delta : float
        Strength of the constant term that opens the gap, strictly positive.
    m : int
        Integer angular momentum quantum number of the nuclear factor.
    M0 : float
        Nuclear mass carried by the two normal modes, strictly positive.
    mu : float
        Electron-to-nuclear mass ratio, strictly positive.
    Q_min, Q_max : float
        Ends of the radial grid, with 0 < Q_min < Q_max.
    n_grid : int
        Number of equally spaced grid points, at least 16.

    Returns
    -------
    momentum : "np.ndarray"
        Real array of shape (2, 2). Row 0 holds the real part and row 1 the
        imaginary part; within a row column 0 is the radial component and
        column 1 the angular component, in the orthonormal polar frame.

    Raises
    ------
    ValueError
        If any scalar argument is not finite, if g, Delta, M0 or mu is not
        strictly positive, if m is not an integer scalar, if the grid ends do
        not satisfy 0 < Q_min < Q_max, if n_grid is not an integer of at least
        16, or if Q is not an interior point of that grid.
    '''
    return momentum
```

### Step 10

Matrix element of the electron-nuclear correlation operator

Goal
----
Return the matrix element of the electron-nuclear correlation operator between the upper adiabatic state on the left and the lower one on the right, without the mass-ratio prefactor.

```python
def correlation_coupling(Q: float, theta: float, g: float, K: float,
                         Delta: float, M0: float,
                         momentum: "np.ndarray") -> "np.ndarray":
    '''Off-diagonal element of the electron-nuclear correlation operator.

    Parameters
    ----------
    Q : float
        Radial nuclear coordinate, strictly positive.
    theta : float
        Polar angle of the nuclear coordinate, in radians.
    g : float
        Linear vibronic coupling strength, strictly positive.
    K : float
        Harmonic force constant of the two normal modes.
    Delta : float
        Strength of the constant term that opens the gap, strictly positive.
    M0 : float
        Nuclear mass carried by the two normal modes, strictly positive. The
        mass-ratio prefactor of the perturbation is not applied here.
    momentum : np.ndarray
        The nuclear momentum function at this radius, in the shape returned by
        the previous step: real part in row 0, imaginary part in row 1, radial
        component in column 0 and angular component in column 1.

    Returns
    -------
    element : np.ndarray
        Real array of shape (2,) holding the real part and then the imaginary
        part of the matrix element, with the upper adiabatic state on the left
        and the lower one on the right.

    Raises
    ------
    ValueError
        If any of Q, theta, g, K, Delta is not a finite scalar, if Q, g or
        Delta is not strictly positive, if M0 is not a finite positive scalar,
        or if momentum is not a finite real array of shape (2, 2).
    '''
    return element
```

### Step 11

Leading correction to the Berry connection

Goal
----
Return the first-order correction, in the mass ratio, to the Berry connection of the lower adiabatic state.

```python
def connection_correction(Q: float, theta: float, g: float, K: float,
                          Delta: float, M0: float, mu: float,
                          momentum: "np.ndarray") -> "np.ndarray":
    '''Leading correction to the Berry connection of the lower state.

    Parameters
    ----------
    Q : float
        Radial nuclear coordinate, strictly positive.
    theta : float
        Polar angle of the nuclear coordinate, in radians.
    g : float
        Linear vibronic coupling strength, strictly positive.
    K : float
        Harmonic force constant of the two normal modes.
    Delta : float
        Strength of the constant term that opens the gap, strictly positive.
    M0 : float
        Nuclear mass carried by the two normal modes, strictly positive.
    mu : float
        Electron-to-nuclear mass ratio that scales the perturbation, strictly
        positive.
    momentum : np.ndarray
        The nuclear momentum function at this radius, in the shape returned by
        the nuclear momentum step.

    Returns
    -------
    correction : np.ndarray
        Real array of shape (2,) holding the radial and then the angular
        component of the correction to the Berry connection, in the orthonormal
        polar frame.

    Raises
    ------
    ValueError
        If any of Q, theta, g, K, Delta is not a finite scalar, if Q, g or
        Delta is not strictly positive, if M0 or mu is not a finite positive
        scalar, or if momentum is not a finite real array of shape (2, 2).
    '''
    return correction
```

### Step 12

Relative size of the nonadiabatic correction

Goal
----
Run the whole pipeline and return the correction to the geometric phase as a percentage of the uncorrected phase on the same loop.

```python
def nonadiabatic_phase_ratio(Q: float = 0.35, g: float = 0.60,
                             K: float = 1.00, Delta: float = 0.12,
                             m: int = 3, M0: float = 100.0,
                             mu: float = 0.01,
                             Q_min: float = 0.02,
                             Q_max: float = 2.02,
                             n_grid: int = 4001,
                             n_theta: int = 2048) -> float:
    '''Nonadiabatic correction to the geometric phase, as a percentage.

    Parameters
    ----------
    Q : float
        Radius of the circular nuclear contour. It must coincide with one of
        the interior points of the radial grid.
    g : float
        Linear vibronic coupling strength, strictly positive.
    K : float
        Harmonic force constant of the two normal modes.
    Delta : float
        Strength of the constant term that opens the gap, strictly positive.
    m : int
        Integer angular momentum quantum number of the nuclear factor.
    M0 : float
        Nuclear mass carried by the two normal modes, strictly positive.
    mu : float
        Electron-to-nuclear mass ratio, strictly positive.
    Q_min, Q_max : float
        Ends of the radial grid, with 0 < Q_min < Q_max.
    n_grid : int
        Number of equally spaced radial grid points, at least 16.
    n_theta : int
        Number of equally spaced quadrature points on the loop, at least 8.
        The grid is theta_k = 2 pi k / n_theta for k = 0 .. n_theta - 1.

    Returns
    -------
    ratio : float
        One hundred times the ratio of the correction to the geometric phase
        to the uncorrected geometric phase, both on the same loop.

    Raises
    ------
    ValueError
        If any scalar argument is not finite, if Q, g, Delta, M0 or mu is not
        strictly positive, if m is not an integer scalar, if the grid ends do
        not satisfy 0 < Q_min < Q_max, if n_grid is not an integer of at least
        16, if n_theta is not an integer of at least 8, or if Q is not an
        interior point of the radial grid.
    '''
    return ratio
```
