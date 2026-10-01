# Physics-Condensed_Matter_Physics-35

## Background

Heat, mass and charge all spread by the same universal law: a flux proportional to minus the gradient of a potential, and a stored quantity proportional to that potential through a capacity. The law is reciprocal. Swap the hot and cold ends of a bar and the heat current reverses exactly; there is no diffusive analogue of the diode, the isolator or the circulator that microwave and photonic engineering take for granted. That absence matters practically, because routing heat away from one component without letting it return, partitioning a thermal load among several sinks, or steering an ionic flux around a ring are all problems that reciprocal materials cannot solve however cleverly they are shaped. The metamaterial community has therefore spent a decade looking for ways to break diffusive reciprocity, and the honest answers have been few: genuine nonlinearity, or a medium that physically moves.

Physical motion is the textbook route. A rotating shell or a circulating fluid superposes advection on diffusion, the Péclet number measures the bias, and the transport becomes directional. It is also the route that defeats device engineering, because a circulator needs ports fixed in the laboratory frame while the medium beneath them turns. Spatiotemporal modulation replaces the motion with a wave. If the material parameters themselves are programmed as travelling waves — resistances and capacitances switched in a rolling pattern in an electrical network, or a heat capacity swung by a moving magnetic field in a magnetocaloric solid — then time-reversal symmetry is broken by the modulation rather than by mass transport, and the boundaries stay where they were. What such a medium does is not obvious, because modulating the conductivity alone changes nothing about reciprocity; the bias appears only through the coupling between a modulated conductivity and a modulated capacity, so both coefficients must travel together and the term in which the potential multiplies the time derivative of the capacity is exactly the term that carries the effect.

Analysing this is harder than the usual metamaterial problem. The governing equation has coefficients that depend on space and time, so no effective-medium average of the static kind applies, and the two standard escapes in the literature — assuming a small modulation depth, or dropping the awkward capacity-derivative term — remove precisely the physics of interest whenever the modulation is strong. What survives is a Bloch analysis in the co-moving frame: the coefficients become functions of a single travelling coordinate, the steady states with a stationary laboratory-frame profile are picked out by locking the Bloch frequency to the Bloch wavenumber through the modulation speed, and the truncated Fourier problem yields a finite set of complex spatial decay constants together with the periodic envelope belonging to each. Fixing the potential at two points of the loop then selects a combination of those states, and the time-averaged flux it carries turns out to be produced entirely by the single non-decaying member of the set.

The pay-off of that exact treatment is a description in effective terms that a device engineer can use. The modulated medium behaves, as far as its terminals can tell, like an ordinary advection–diffusion medium with an effective conductivity and an effective advective coefficient, whose ratio is a single intrinsic inverse length that measures the nonreciprocity of the material independently of how it is terminated. That intrinsic parameter is not the whole story, though: what a port actually experiences is a rectification ratio, built from the two flux densities the segment passes when its terminal potentials are exchanged, and that quantity depends on the boundary values as well as on the medium. The two together — one intrinsic, one boundary-sensitive — are what decide whether a given modulation makes a usable diode or circulator, and how many cells a branch must contain before the convenient homogenised description of it can be trusted at all.

## Problem

Diffusion in ordinary matter is reciprocal, so exchanging the two terminal potentials of a conductor merely reverses the flux through it. Driving both transport coefficients of a diffusive medium — the conductivity $\sigma$ and the capacity $c$ — as co-propagating travelling waves breaks that symmetry and endows the medium with an effective drift, so a segment of it rectifies and a closed loop of it circulates flux while no material ever moves. The one-dimensional balance is $\partial_t(c\Phi) = \partial_x(\sigma\,\partial_x\Phi)$ with $\sigma(x,t) = \sigma(x - v_0 t)$ and $c(x,t) = c(x - v_0 t)$, both of spatial period $d$; because the two coefficients are modulated together the problem does not collapse to a diffusion equation with an effective diffusivity, and the $\Phi\,\partial_t c$ contribution that small-amplitude treatments discard must be retained in full. Under fixed terminal potentials the relevant states are the ones whose spatial profile is stationary in the laboratory frame, and the medium enters the resulting algebraic problem only through the Fourier coefficients of $\sigma$ and $c$.

Consider a branch of length $L = 1$ m occupying $x \in [0, 1]$ and containing $s = 5$ modulation cells, so $d = L/s$ and $\beta = 2\pi/d$. With $\chi = x - v_0 t$ the profiles are

$$\sigma(\chi) = \sigma_0\left[1 + 0.7\cos(\beta\chi) + 0.2\cos(2\beta\chi + \tfrac{\pi}{3})\right], \qquad c(\chi) = c_0\left[1 + 0.5\cos(\beta\chi + \tfrac{2\pi}{5})\right],$$

with $\sigma_0 = 1$ S m$^{-1}$, $c_0 = 100$ F m$^{-3}$ and $v_0 = 0.06$ m s$^{-1}$. The end at $x = 0$ is held at $\Phi_H = 30$ V and the end at $x = 1$ m at $\Phi_L = 10$ V, and every Fourier expansion is truncated at order 16. Define the rectification ratio of the branch as the sum of the two time-averaged flux densities obtained by exchanging the two terminal potentials, divided by the larger of their two magnitudes. Your final answer must be a single number: that rectification ratio divided by the value the same definition returns when the branch is described instead by the effective advection–diffusion medium the identical modulation produces in the limit of infinitely many cells within the same branch. Alongside that number report, in brief, the selection rule that makes a laboratory-frame profile admissible, how the admissible decay constants are obtained and which one carries the time-averaged flux, what the two exchanged fluxes come to, the normalisation fixing each periodic envelope, the way a fixed terminal potential constrains the expansion, the conditioning the boundary inversion needs at this truncation order, the intrinsic and limiting parameters your two descriptions produce and how they compare, whether an independent route to the rectification agrees, how far the terminal layers reach against the branch length, and how far the truncation order has converged; short assertions carrying their numbers are what is wanted, not derivations.

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

01_compute_modulation_fourier_coefficients

Goal
----
Reduce the two travelling-wave material profiles to the complex Fourier coefficients through which they enter the modulated diffusion problem.

```python
def compute_modulation_fourier_coefficients(sigma_mean: float,
                                            sigma_amplitudes: "np.ndarray",
                                            sigma_phases: "np.ndarray",
                                            capacity_mean: float,
                                            capacity_amplitudes: "np.ndarray",
                                            capacity_phases: "np.ndarray",
                                            order: int) -> "np.ndarray":
    """Reduce the two modulation profiles to their complex Fourier coefficients.

    Parameters
    ----------
    sigma_mean : float
        Mean conductivity of the profile, strictly positive.
    sigma_amplitudes : np.ndarray
        Relative amplitudes of harmonics 1, 2, ... of the conductivity.
    sigma_phases : np.ndarray
        Phase offsets in radians of the same conductivity harmonics.
    capacity_mean : float
        Mean capacity of the profile, strictly positive.
    capacity_amplitudes : np.ndarray
        Relative amplitudes of harmonics 1, 2, ... of the capacity.
    capacity_phases : np.ndarray
        Phase offsets in radians of the same capacity harmonics.
    order : int
        Fourier truncation order, at least 1 and at least the highest
        harmonic present in either profile.

    Returns
    -------
    modes : np.ndarray
        Complex array of shape (2, 2 * order + 1). Row 0 holds the
        conductivity coefficients and row 1 the capacity coefficients, both
        ordered by harmonic index from -order to +order.

    Raises
    ------
    ValueError
        If order is not an integer >= 1, if either profile mean is not a
        finite number > 0, if an amplitude array and its phase array have
        different lengths, if any amplitude or phase is not finite, if order
        is smaller than the highest harmonic present in either profile, or if
        the amplitudes drive either profile to zero or below anywhere in the
        cell.
    """
    return modes  # placeholder
```

### Step 2

02_assemble_secular_matrix

Goal
----
Assemble the harmonic coupling matrix whose determinant vanishes on the admissible spatial decay constants of the modulated medium.

```python
def assemble_secular_matrix(alpha: complex,
                            sigma_modes: "np.ndarray",
                            capacity_modes: "np.ndarray",
                            beta: float,
                            modulation_speed: float) -> "np.ndarray":
    """Assemble the harmonic coupling matrix at one trial decay constant.

    Parameters
    ----------
    alpha : complex
        Trial spatial decay constant in inverse metres.
    sigma_modes : np.ndarray
        Complex conductivity coefficients of shape (2 * order + 1), indexed
        from harmonic -order to +order. The profile carries no harmonics
        beyond that range, so any coefficient whose harmonic index falls
        outside it is zero.
    capacity_modes : np.ndarray
        Complex capacity coefficients with the same shape and ordering, under
        the same convention.
    beta : float
        Modulation wavenumber in inverse metres, nonzero.
    modulation_speed : float
        Speed of the travelling modulation in metres per second.

    Returns
    -------
    matrix : np.ndarray
        Complex array of shape (2 * order + 1, 2 * order + 1) whose row index
        runs over the projected harmonic and whose column index runs over the
        envelope harmonic, both from -order to +order.

    Raises
    ------
    ValueError
        If sigma_modes is not a one-dimensional array of odd length >= 3, if
        capacity_modes does not have the same shape as sigma_modes, if any
        coefficient is not finite, if beta is not a finite nonzero number, if
        modulation_speed is not a finite number, or if alpha is not finite.
    """
    return matrix  # placeholder
```

### Step 3

03_solve_secular_roots

Goal
----
Extract every admissible spatial decay constant of the modulated medium from the vanishing of the harmonic coupling determinant.

```python
def solve_secular_roots(sigma_modes: "np.ndarray",
                        capacity_modes: "np.ndarray",
                        beta: float,
                        modulation_speed: float) -> "np.ndarray":
    """Solve the secular equation for every spatial decay constant.

    Parameters
    ----------
    sigma_modes : np.ndarray
        Complex conductivity coefficients of shape (2 * order + 1), indexed
        from harmonic -order to +order.
    capacity_modes : np.ndarray
        Complex capacity coefficients with the same shape and ordering.
    beta : float
        Modulation wavenumber in inverse metres, nonzero.
    modulation_speed : float
        Speed of the travelling modulation in metres per second.

    Returns
    -------
    alphas : np.ndarray
        Complex array of shape (4 * order + 2) holding every root of the
        secular equation, sorted by ascending real part rounded to six
        decimals, with ties broken by ascending imaginary part. Ordering
        inside a numerically degenerate group is immaterial.

    Raises
    ------
    ValueError
        If sigma_modes is not a one-dimensional array of odd length >= 3, if
        capacity_modes does not have the same shape as sigma_modes, if any
        coefficient is not finite, if beta is not a finite nonzero number, if
        modulation_speed is not a finite number, or if the Toeplitz matrix of
        the conductivity coefficients is singular.
    """
    return alphas  # placeholder
```

### Step 4

04_compute_bloch_fourier_components

Goal
----
Recover the periodic envelope belonging to one spatial decay constant as the null vector of the harmonic coupling matrix under a fixed normalisation.

```python
def compute_bloch_fourier_components(alpha: complex,
                                     sigma_modes: "np.ndarray",
                                     capacity_modes: "np.ndarray",
                                     beta: float,
                                     modulation_speed: float) -> "np.ndarray":
    """Recover the normalised periodic envelope of one Bloch state.

    Parameters
    ----------
    alpha : complex
        Spatial decay constant of the state, a root of the secular equation.
    sigma_modes : np.ndarray
        Complex conductivity coefficients of shape (2 * order + 1), indexed
        from harmonic -order to +order.
    capacity_modes : np.ndarray
        Complex capacity coefficients with the same shape and ordering.
    beta : float
        Modulation wavenumber in inverse metres, nonzero.
    modulation_speed : float
        Speed of the travelling modulation in metres per second.

    Returns
    -------
    envelope : np.ndarray
        Complex array of shape (2 * order + 1) holding the Fourier
        coefficients of the envelope from harmonic -order to +order, with the
        zeroth coefficient equal to one.

    Raises
    ------
    ValueError
        If sigma_modes is not a one-dimensional array of odd length >= 3, if
        capacity_modes does not have the same shape as sigma_modes, if any
        coefficient is not finite, if beta is not a finite nonzero number, if
        modulation_speed is not a finite number, if alpha is not finite, or if
        the reduced coupling matrix is singular.
    """
    return envelope  # placeholder
```

### Step 5

05_compute_boundary_coefficients

Goal
----
Impose the two fixed terminal potentials on the superposition of Bloch states and extract the two boundary coefficients belonging to the non-decaying state.

```python
def compute_boundary_coefficients(alphas: "np.ndarray",
                                  envelopes: "np.ndarray",
                                  position_high: float,
                                  position_low: float) -> "np.ndarray":
    """Extract the boundary coefficients of the non-decaying Bloch state.

    Parameters
    ----------
    alphas : np.ndarray
        Complex spatial decay constants of shape (4 * order + 2).
    envelopes : np.ndarray
        Complex array of shape (2 * order + 1, 4 * order + 2) whose column k
        holds the envelope coefficients of the state with decay constant
        alphas[k], indexed by harmonic from -order to +order.
    position_high : float
        Position in metres at which the higher potential is imposed.
    position_low : float
        Position in metres at which the lower potential is imposed.

    Returns
    -------
    coefficients : np.ndarray
        Complex array of shape (2). The amplitude of the state whose decay
        constant is smallest in magnitude equals coefficients[0] times the
        potential imposed at position_high plus coefficients[1] times the
        potential imposed at position_low.

    Raises
    ------
    ValueError
        If alphas is not one-dimensional, if envelopes is not two-dimensional
        with an odd number of rows >= 3, if alphas and envelopes do not supply
        4 * order + 2 states, if any entry of either is not finite, if either
        terminal position is not a finite number, if the two terminals sit at
        the same position, if some envelope is identically zero, or if the
        stacked boundary matrix is singular.
    """
    return coefficients  # placeholder
```

### Step 6

06_compute_time_averaged_flux

Goal
----
Evaluate the time-averaged flux density the branch passes under one assignment of the two terminal potentials.

```python
def compute_time_averaged_flux(sigma_modes: "np.ndarray",
                               star_envelope: "np.ndarray",
                               beta: float,
                               boundary_coefficients: "np.ndarray",
                               phi_high: float,
                               phi_low: float) -> float:
    """Evaluate the time-averaged flux density carried by the branch.

    Parameters
    ----------
    sigma_modes : np.ndarray
        Complex conductivity coefficients of shape (2 * order + 1), indexed
        from harmonic -order to +order.
    star_envelope : np.ndarray
        Complex envelope coefficients of the non-decaying state, same shape
        and ordering.
    beta : float
        Modulation wavenumber in inverse metres, nonzero.
    boundary_coefficients : np.ndarray
        Complex array of shape (2) as returned by the boundary step, the first
        entry belonging to the terminal held at phi_high.
    phi_high : float
        Potential imposed at the terminal belonging to the first coefficient.
    phi_low : float
        Potential imposed at the terminal belonging to the second coefficient.

    Returns
    -------
    flux : float
        Time-averaged flux density in the direction of increasing position, as
        a native Python float.

    Raises
    ------
    ValueError
        If sigma_modes is not a one-dimensional array of odd length >= 3, if
        star_envelope does not have the same shape as sigma_modes, if
        boundary_coefficients does not hold exactly two entries, if any array
        entry is not finite, if beta is not a finite nonzero number, or if
        either terminal potential is not a finite number.
    """
    return flux  # placeholder
```

### Step 7

07_compute_effective_parameters

Goal
----
Map the modulated branch onto an equivalent advection-diffusion medium and extract its effective conductivity, advective coefficient and their ratio.

```python
def compute_effective_parameters(sigma_modes: "np.ndarray",
                                 star_envelope: "np.ndarray",
                                 beta: float,
                                 boundary_coefficients: "np.ndarray",
                                 length: float) -> "np.ndarray":
    """Extract the effective parameters of the equivalent medium.

    Parameters
    ----------
    sigma_modes : np.ndarray
        Complex conductivity coefficients of shape (2 * order + 1), indexed
        from harmonic -order to +order.
    star_envelope : np.ndarray
        Complex envelope coefficients of the non-decaying state, same shape
        and ordering.
    beta : float
        Modulation wavenumber in inverse metres, nonzero.
    boundary_coefficients : np.ndarray
        Complex array of shape (2) as returned by the boundary step, the first
        entry belonging to the higher-potential terminal.
    length : float
        Separation of the two terminals in metres, strictly positive.

    Returns
    -------
    parameters : np.ndarray
        Real array of shape (3) holding the effective conductivity, the
        effective advective coefficient and the advection-diffusion ratio, in
        that order.

    Raises
    ------
    ValueError
        If sigma_modes is not a one-dimensional array of odd length >= 3, if
        star_envelope does not have the same shape as sigma_modes, if
        boundary_coefficients does not hold exactly two entries, if any array
        entry is not finite, if beta is not a finite nonzero number, if length
        is not a finite number > 0, if the second boundary coefficient is
        zero, if the two boundary coefficients do not give a positive real
        exponential factor to numerical precision, or if they cancel so that
        the matching leaves the effective conductivity undetermined.
    """
    return parameters  # placeholder
```

### Step 8

08_compute_rectification_ratio

Goal
----
Convert the intrinsic drift-to-diffusion ratio of a branch into the boundary-sensitive figure of merit that its terminals actually experience.

```python
def compute_rectification_ratio(advection_ratio: float,
                                length: float,
                                phi_high: float,
                                phi_low: float) -> float:
    """Evaluate the rectification ratio of a branch under fixed potentials.

    Parameters
    ----------
    advection_ratio : float
        Ratio of the effective advective coefficient to the effective
        conductivity, in inverse metres.
    length : float
        Separation of the two terminals in metres, strictly positive.
    phi_high : float
        Potential imposed at the first terminal.
    phi_low : float
        Potential imposed at the second terminal.

    Returns
    -------
    rectification : float
        Sum of the two time-averaged flux densities obtained by exchanging the
        terminal potentials, divided by the larger of their magnitudes, as a
        native Python float.

    Raises
    ------
    ValueError
        If advection_ratio, phi_high or phi_low is not a finite number, or if
        length is not a finite number > 0.
    """
    return rectification  # placeholder
```

### Step 9

09_compute_homogenized_limit

Goal
----
Evaluate the effective parameters the same modulation produces in the limit of infinitely many cells inside a branch of fixed length.

```python
def compute_homogenized_limit(sigma_modes: "np.ndarray",
                              capacity_modes: "np.ndarray",
                              modulation_speed: float) -> "np.ndarray":
    """Evaluate the effective parameters in the many-cell limit.

    Parameters
    ----------
    sigma_modes : np.ndarray
        Complex conductivity coefficients of shape (2 * order + 1), indexed
        from harmonic -order to +order. The profile carries no harmonics
        beyond that range, so any coefficient whose harmonic index falls
        outside it is zero.
    capacity_modes : np.ndarray
        Complex capacity coefficients with the same shape and ordering, under
        the same convention.
    modulation_speed : float
        Speed of the travelling modulation in metres per second.

    Returns
    -------
    parameters : np.ndarray
        Real array of shape (3) holding the limiting effective conductivity,
        the limiting effective advective coefficient and their ratio, in that
        order.

    Raises
    ------
    ValueError
        If sigma_modes is not a one-dimensional array of odd length >= 3, if
        capacity_modes does not have the same shape as sigma_modes, if any
        coefficient is not finite, if modulation_speed is not a finite number,
        if the index-weighted conductivity coupling restricted to the nonzero
        harmonics is singular, or if the limiting effective conductivity does
        not come out positive.
    """
    return parameters  # placeholder
```

### Step 10

10_run_diffusive_circulator_pipeline

Goal
----
Final orchestrator. Chain every step to compare the rectification a modulated branch actually delivers with the rectification its many-cell homogenised description predicts.

```python
def run_diffusive_circulator_pipeline(n_cells: int = 5,
                                      modulation_speed: float = 0.06,
                                      phi_high: float = 30.0,
                                      phi_low: float = 10.0,
                                      length: float = 1.0,
                                      sigma_mean: float = 1.0,
                                      capacity_mean: float = 100.0,
                                      sigma_amplitudes: tuple = (0.7, 0.2),
                                      sigma_phases: tuple = (0.0, 1.0471975511965976),
                                      capacity_amplitudes: tuple = (0.5,),
                                      capacity_phases: tuple = (1.2566370614359172,),
                                      order: int = 16) -> float:
    """Compare the rectification of a modulated branch with its many-cell limit.

    Parameters
    ----------
    n_cells : int
        Number of modulation cells inside the branch, at least 1.
    modulation_speed : float
        Speed of the travelling modulation in metres per second, nonzero.
    phi_high : float
        Potential imposed at the terminal at position zero, in volts.
    phi_low : float
        Potential imposed at the terminal at the far end, in volts.
    length : float
        Length of the branch in metres, strictly positive.
    sigma_mean : float
        Mean conductivity of the profile, strictly positive.
    capacity_mean : float
        Mean capacity of the profile, strictly positive.
    sigma_amplitudes : tuple
        Relative amplitudes of harmonics 1, 2, ... of the conductivity.
    sigma_phases : tuple
        Phase offsets in radians of the same conductivity harmonics.
    capacity_amplitudes : tuple
        Relative amplitudes of harmonics 1, 2, ... of the capacity.
    capacity_phases : tuple
        Phase offsets in radians of the same capacity harmonics.
    order : int
        Fourier truncation order used throughout.

    Returns
    -------
    ratio : float
        Rectification ratio of the branch divided by the rectification ratio of
        its many-cell homogenised description, as a native Python float.

    Raises
    ------
    ValueError
        If n_cells is not an integer >= 1, if length is not a finite number
        > 0, if modulation_speed is not a finite nonzero number, if the profile
        means, amplitudes, phases or truncation order do not define finite
        strictly positive modulation profiles, if a derived secular or boundary
        system is singular or inconsistent, if the branch passes no flux, if
        the flux and effective-medium routes disagree, or if the homogenised
        rectification vanishes so that the requested ratio is undefined.
    """
    return ratio  # placeholder
```
