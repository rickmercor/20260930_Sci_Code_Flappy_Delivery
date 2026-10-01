# Chemistry-Quantum_Chemistry-20

## Background

Proton transfer along a hydrogen bond is commonly described by a one-dimensional double-well cross-section of the potential-energy surface, and because the proton is light much of the thermal transfer proceeds by tunnelling through the barrier rather than by passage over it. Rate theories of the flux–transmission type express the thermal rate as a Boltzmann average, over the stationary states of the double well, of the probability flux moving towards the product well multiplied by the fraction of that flux transmitted through the barrier. Their predictions depend on how the stationary states are represented, which makes double-well potentials with exactly solvable Schrödinger equations attractive for this purpose.

## Problem

I am modelling proton transfer across a symmetric hydrogen bond from my own relaxed quantum-chemistry scan of the bridging proton along the donor–acceptor axis. The scan gives a symmetric double well whose two minima lie 0.55 Å on either side of the bond midpoint, with a potential-energy barrier of 5.00 kcal/mol at the midpoint, for a donor–acceptor distance of about 2.78 Å. I want the proton, with the bare proton mass, described by the exactly solvable trigonometric double-well potential confined between the two heavy atoms and fitted exactly to that barrier height and minimum position. The confining distance should move as little as possible from 2.78 Å so that the potential keeps its exact spheroidal-function solution. On that model, take the thermal rate constant from the flux–transmission rate theory reformulated on the exact eigenfunctions, with each eigenfunction taken positive at the minimum on the positive-coordinate (product) side. Use CODATA 2018 constants and 1 kcal = 4.184 kJ. What is the proton-transfer rate constant at 400 K, in units of 10¹² s⁻¹? In your reasoning, report the fitted model and the intermediate scalars the number rests on, with the relations you used, and tell me which states carry the transfer at this temperature and in what proportion. Stating those is what the output requirements below call for.
I am modelling proton transfer across a symmetric hydrogen bond from my own relaxed quantum-chemistry scan of the bridging proton along the donor–acceptor axis. The scan gives a symmetric double well whose two minima lie 0.55 Å on either side of the bond midpoint, with a potential-energy barrier of 5.00 kcal/mol at the midpoint, for a donor–acceptor distance of about 2.78 Å.

Use the exactly solvable trigonometric double-well model on the reduced coordinate x = πX/(2L) with energies in units of ħ²π²/(8ML²). The potential is
U(x) = (m² − 1/4) tan²x − p² sin²x
on −π/2 < x < π/2, with integer order m ≥ 1 and real parameter p > 0. The well minima satisfy cos x_min = [(m² − 1/4)/p²]^{1/4}. Fit p and m to the given physical barrier height and minimum positions through the paper’s barrier/width parameterization, using the bare proton mass M. Choose the confining half-width L so that m is an integer and the corresponding donor–acceptor distance 2L moves as little as possible from 2.78 Å (break ties by the usual closest-L rule in the source). Keep the exact spheroidal eigenfunctions of this well.

On that fitted model, compute the thermal proton-transfer rate constant at 400 K from the paper’s flux–transmission rate theory on those exact eigenfunctions. Take each eigenfunction real, normalized on x, and positive at the minimum on the positive-coordinate (product) side. Use CODATA 2018 constants and 1 kcal = 4.184 kJ. Report the rate in units of 10¹² s⁻¹.

In your reasoning, report the fitted model (L, m, p and related reduced barrier/width scalars), the intermediate scalars the number rests on, the source relations you used, and which states carry the transfer at this temperature and in what proportion.

## Output format

```
## Output format

Scientific reasoning wrapped in <reasoning>...</reasoning> tags. Include enough intermediate calculations to justify the deterministic computation, without turning the response into a general pipeline summary. A single final numeric answer wrapped in <final_answer>...</final_answer> tags.
```

## Your task

Implement **all 7 functions** below into a single file at `/app/solution.py`. Requirements:

- Use the EXACT function name and signature shown for each step.
- Define every function at module scope (importable).
- Do NOT include tests, example usage, prints, or a `__main__` block.
- Available libraries: numpy, scipy, sympy, pandas, scikit-learn, networkx.

Start your file with these imports:

```python
import numpy as np
```

### Step 1

01_fit_trigonometric_well

Goal
----
Fit the exactly solvable trigonometric double well to a symmetric proton-transfer scan and return the confining half-width with the two potential parameters.

```python
def fit_trigonometric_well(
    barrier_kcal_per_mol: float,
    minimum_offset_angstrom: float,
    donor_acceptor_distance_angstrom: float,
) -> "np.ndarray":
    """Return the confining half-width and the parameters of the fitted trigonometric double well.

    The proton, of mass ``M``, moves on ``-L <= X <= L`` between the heavy
    atoms. With ``x = pi X / (2 L)`` and energies measured in units of
    ``hbar**2 pi**2 / (8 M L**2)``, the Schrodinger equation becomes
    ``psi'' + (eps - U(x)) psi = 0`` on ``-pi/2 < x < pi/2`` with the
    potential ``U(x) = (m**2 - 1/4) tan(x)**2 - p**2 sin(x)**2``, where the
    order ``m`` is a positive integer and ``p > 0``. The fitted potential
    reproduces exactly the barrier height ``U(0) - U(x_min)`` and the
    position ``X_min`` of the minimum with ``X_min > 0``. For a trial
    half-width these two conditions give ``m`` as a real number that grows
    monotonically with ``L``; the returned ``L`` is the half-width closest to
    half the donor-acceptor distance at which ``m`` is an integer of at least
    one (the smaller half-width on an exact tie), and ``p`` is the value
    that goes with it. Use ``hbar = 1.054571817e-34`` J s,
    ``M = 1.67262192369e-27`` kg, the Avogadro constant
    ``6.02214076e23`` per mol and ``1 kcal = 4184`` J.

    Parameters
    ----------
    barrier_kcal_per_mol : float
        Barrier height of the scan at the midpoint, in kcal/mol.
    minimum_offset_angstrom : float
        Distance of each minimum from the midpoint, in angstrom.
    donor_acceptor_distance_angstrom : float
        Approximate heavy-atom distance, in angstrom.

    Returns
    -------
    np.ndarray
        Float array ``[L, m, p]`` with ``L`` in angstrom, relative accuracy
        of ``1e-12`` or better in ``L`` and ``p``.

    Raises
    ------
    ValueError
        If an argument is not a finite positive real number (booleans are
        rejected), or if the minimum offset is not smaller than half the
        donor-acceptor distance.
    """
    return fit
```

### Step 2

02_compute_well_energy_levels

Goal
----
Compute the lowest energy levels of the trigonometric double well exactly, in the reduced energy units of the model.

```python
def compute_well_energy_levels(m: int, p: float, n_levels: int) -> "np.ndarray":
    """Return the lowest energy levels of the trigonometric double well.

    The levels ``eps_0 < eps_1 < ...`` are the eigenvalues of
    ``psi'' + (eps - U(x)) psi = 0`` on ``-pi/2 < x < pi/2`` with
    ``U(x) = (m**2 - 1/4) tan(x)**2 - p**2 sin(x)**2`` and ``psi`` vanishing
    at both ends, in the reduced units in which the barrier top is
    ``U(0) = 0``. Each returned level must be exact up to rounding: relative
    error below ``1e-10`` (absolute error below ``1e-10`` when
    ``|eps| < 1``).

    Parameters
    ----------
    m : int
        Order of the potential, a positive integer.
    p : float
        Positive potential parameter.
    n_levels : int
        Number of levels to return, a positive integer.

    Returns
    -------
    np.ndarray
        Float array of shape ``(n_levels,)`` holding ``eps_0, ...,
        eps_{n_levels-1}`` in ascending order.

    Raises
    ------
    ValueError
        If ``m`` or ``n_levels`` is not a positive integer (booleans are
        rejected) or if ``p`` is not a finite positive real number.
    """
    return levels
```

### Step 3

03_evaluate_well_eigenfunction

Goal
----
Evaluate a normalized eigenfunction of the trigonometric double well and its coordinate derivative at given points, exactly up to rounding.

```python
def evaluate_well_eigenfunction(m: int, p: float, q: int, x: "np.ndarray") -> "np.ndarray":
    """Return the q-th eigenfunction of the trigonometric double well and its derivative.

    ``psi_q`` is the eigenfunction of ``psi'' + (eps - U(x)) psi = 0`` on
    ``-pi/2 < x < pi/2``, with ``U(x) = (m**2 - 1/4) tan(x)**2 -
    p**2 sin(x)**2`` and ``psi`` vanishing at both ends, that belongs to the
    level ``eps_q`` of ``compute_well_energy_levels`` (``q = 0`` is the
    ground state). It is normalized so that the integral of ``psi_q**2`` over
    the interval is one, and its sign is chosen so that ``psi_q(x_min) > 0``
    at the minimum ``x_min > 0`` of ``U``. Returned values must be exact up
    to rounding: relative error below ``1e-10`` (absolute error below
    ``1e-10`` for magnitudes below one).

    Parameters
    ----------
    m : int
        Order of the potential, a positive integer.
    p : float
        Potential parameter with ``p**2 > m**2 - 1/4``, so that ``U`` is a
        double well.
    q : int
        Level index, a nonnegative integer.
    x : np.ndarray
        Nonempty one-dimensional array of points with ``|x| < pi/2``.

    Returns
    -------
    np.ndarray
        Float array of shape ``(2, x.size)``: row 0 holds ``psi_q(x)`` and row
        1 holds ``d psi_q / dx``.

    Raises
    ------
    ValueError
        If ``m`` is not a positive integer or ``q`` a nonnegative integer
        (booleans are rejected), if ``p`` is not finite with
        ``p**2 > m**2 - 1/4``, or if ``x`` is empty, not one-dimensional,
        not finite or has a point with ``|x| >= pi/2``.
    """
    return values
```

### Step 4

04_compute_right_moving_flux

Goal
----
Compute the regularizing constant and the right-moving probability flux at the product-well minimum for a level below the barrier top of the trigonometric double well.

```python
def compute_right_moving_flux(m: int, p: float, q: int) -> "np.ndarray":
    """Return the flux regularizer mu_q**2 and the right-moving flux J_q of a level below the barrier top.

    Let ``eps_q < 0`` be the level of ``compute_well_energy_levels`` (the
    barrier top is ``U(0) = 0``), ``psi_q`` its eigenfunction from
    ``evaluate_well_eigenfunction`` and ``x_min > 0`` the minimum of
    ``U(x) = (m**2 - 1/4) tan(x)**2 - p**2 sin(x)**2``. Write
    ``psi_q = F cos(chi) = psi_r + psi_l`` with
    ``psi_r = (F / 2) exp(i chi)`` and ``psi_l = (F / 2) exp(-i chi)``,
    where the phase is fixed by::

        tan(chi(x)) = -(psi_q'(x) / psi_q(x)) * alpha(x) / (psi_q(x)**2 + mu**2)
        alpha(x) = exp(sqrt(2) |psi_q'(x_min)| (|x| - x_min) / s),
        s = sqrt(psi_q(x_min)**2 + mu**2)

    and ``mu**2 > 0`` solves::

        (eps_q - U(x_min)) (2 s**3 - 3 sqrt(2) psi_q(x_min) s**2)
            + 2 psi_q'(x_min)**2 (s - 2 sqrt(2) psi_q(x_min)) = 0,

    taking the largest real root ``s > |psi_q(x_min)|``. Primes denote
    ``d/dx``. The flux is the probability current of the right-moving part,
    ``J_q = Im(conj(psi_r) d psi_r / dx)``, evaluated at ``x = x_min`` in the
    reduced units of the model.

    Parameters
    ----------
    m : int
        Order of the potential, a positive integer.
    p : float
        Potential parameter with ``p**2 > m**2 - 1/4``.
    q : int
        Index of a level with ``eps_q < 0``.

    Returns
    -------
    np.ndarray
        Float array ``[mu_q**2, J_q]``.

    Raises
    ------
    ValueError
        If ``m``, ``p`` or ``q`` is invalid as in ``evaluate_well_eigenfunction``,
        if ``eps_q >= 0``, or if the equation for ``s`` has no real root
        with ``s > |psi_q(x_min)|``.
    """
    return flux
```

### Step 5

05_compute_transmission_probability

Goal
----
Compute the quantum transmission probability through the barrier for a level below the barrier top from the scattering states matched at its inner turning points.

```python
def compute_transmission_probability(m: int, p: float, q: int, mu2: float) -> float:
    """Return the transmission probability |T_q|**2 of a level below the barrier top.

    ``psi_q``, ``F``, ``chi``, ``alpha``, ``psi_r``, ``psi_l`` and ``x_min``
    are defined as in ``compute_right_moving_flux``, with the constant
    ``mu**2`` set to ``mu2`` (``compute_right_moving_flux`` supplies the
    regularizer of the level, but any positive value is accepted). The phase
    is odd, ``chi(-x) = -chi(x)``. Let ``a_q`` be the inner turning point,
    ``0 < a_q < x_min`` with ``U(a_q) = eps_q``. Matching the scattering
    states in value and slope at ``x = -a_q`` and ``x = a_q``, with
    reflection amplitude ``R_q = 1 - T_q``, gives::

        T_q = psi_q(a_q) / (psi_r(a_q) - psi_l(-a_q))          for odd q
        T_q = psi_q'(a_q) / (psi_r'(a_q) - psi_l'(-a_q))       for even q

    where primes denote ``d/dx``. Return ``|T_q|**2``.

    Parameters
    ----------
    m : int
        Order of the potential, a positive integer.
    p : float
        Potential parameter with ``p**2 > m**2 - 1/4``.
    q : int
        Index of a level with ``eps_q < 0``.
    mu2 : float
        Positive finite value of ``mu**2``.

    Returns
    -------
    float
        The transmission probability ``|T_q|**2``.

    Raises
    ------
    ValueError
        If ``m``, ``p`` or ``q`` is invalid as in ``evaluate_well_eigenfunction``,
        if ``eps_q >= 0``, or if ``mu2`` is not a finite positive real number
        (booleans are rejected).
    """
    return transmission
```

### Step 6

06_compute_reduced_rate_constant

Goal
----
Compute the Boltzmann-averaged reduced proton-transfer rate constant of the trigonometric double well at a given reduced inverse temperature.

```python
def compute_reduced_rate_constant(m: int, p: float, beta: float) -> float:
    """Return the reduced rate constant k(beta) of the trigonometric double well.

    With the levels ``eps_q`` of ``compute_well_energy_levels``::

        k(beta) = sum_q exp(-beta eps_q) g_q / sum_q exp(-beta eps_q)

    where both sums run over all levels ``q = 0, 1, 2, ...``, ``g_q`` is the
    product of ``J_q`` from ``compute_right_moving_flux`` and ``|T_q|**2``
    from ``compute_transmission_probability`` (evaluated with that level's
    ``mu_q**2``) for every level with ``eps_q < 0``, and ``g_q = 1`` for every
    level with ``eps_q >= 0``. The sums must be converged: including further
    levels may change ``k`` by less than ``1e-12`` relative.

    Parameters
    ----------
    m : int
        Order of the potential, a positive integer.
    p : float
        Potential parameter with ``p**2 > m**2 - 1/4``.
    beta : float
        Reduced inverse temperature, the energy unit of the model divided by
        ``k_B T``; finite and positive.

    Returns
    -------
    float
        The reduced rate constant ``k(beta)``.

    Raises
    ------
    ValueError
        If ``beta`` is not a finite positive real number (booleans are
        rejected), if ``m`` or ``p`` is invalid as in
        ``evaluate_well_eigenfunction``, or if the sums do not converge within
        1024 levels.
    """
    return rate
```

### Step 7

07_compute_proton_transfer_rate

Goal
----
Compose every earlier step to obtain the thermal proton-transfer rate constant, in units of 1e12 per second, of a symmetric hydrogen bond described by a fitted trigonometric double well. Orchestrator: it fits the double well to the scan (fit_trigonometric_well), converts the temperature to the reduced inverse temperature of the fitted model and evaluates the reduced rate constant (compute_reduced_rate_constant, which uses compute_well_energy_levels, compute_right_moving_flux and compute_transmission_probability, the last two built on evaluate_well_eigenfunction), then converts that rate to per-second units, consuming each output rather than reimplementing any step.

```python
def compute_proton_transfer_rate(
    barrier_kcal_per_mol: float = 5.0,
    minimum_offset_angstrom: float = 0.55,
    donor_acceptor_distance_angstrom: float = 2.78,
    temperature_kelvin: float = 400.0,
) -> float:
    """Return the thermal proton-transfer rate constant in units of 1e12 per second.

    Fit the trigonometric double well to the scan with
    ``fit_trigonometric_well`` to obtain ``[L, m, p]``, form the reduced
    inverse temperature ``beta = hbar**2 pi**2 / (8 M L**2 k_B T)``, evaluate
    ``k(beta)`` with ``compute_reduced_rate_constant`` and return
    ``Gamma = k(beta) hbar / (M L**2)`` divided by ``1e12`` per second. Here
    ``M`` is the proton mass, ``L`` is converted from angstrom to metres, and
    the constants are ``hbar = 1.054571817e-34`` J s,
    ``M = 1.67262192369e-27`` kg and ``k_B = 1.380649e-23`` J/K. The defaults
    reproduce the problem statement.

    Parameters
    ----------
    barrier_kcal_per_mol : float
        Barrier height of the scan at the midpoint, in kcal/mol.
    minimum_offset_angstrom : float
        Distance of each minimum from the midpoint, in angstrom.
    donor_acceptor_distance_angstrom : float
        Approximate heavy-atom distance, in angstrom.
    temperature_kelvin : float
        Temperature in kelvin.

    Returns
    -------
    float
        Rate constant ``Gamma`` in units of ``1e12`` per second.

    Raises
    ------
    ValueError
        If ``temperature_kelvin`` is not a finite positive real number
        (booleans are rejected), or if the scan inputs are invalid as in
        ``fit_trigonometric_well``.
    """
    return 0.0
```
