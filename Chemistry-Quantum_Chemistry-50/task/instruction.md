# Chemistry-Quantum_Chemistry-50

## Background

Hydrogen transfer at low temperature proceeds largely by tunnelling, and semiclassical instanton theory describes it through an optimal imaginary-time tunnelling path that is usually located on a discretized ring polymer. The standard leading-order theory treats every fluctuation about that path harmonically, so it misses anharmonicity, particularly in vibrations perpendicular to the path whose shape changes as the reaction proceeds. Because the path and its derivatives must eventually come from expensive electronic-structure calculations, corrections that reuse the same path and need only local information along it are especially valuable.

## Problem

I need a low-temperature hydrogen-transfer rate from ring-polymer instanton theory carried one order further in ℏ, in the expansion at fixed thermal time βℏ, so that anharmonicity along and across the tunnelling path is captured from local derivatives on that path alone. Work in atomic units with ℏ = 1 and mass m = 1836 for both coordinates, and use the surface V(x,y) = V₀ sech²(x/a) + D(x)[1 − exp(−b(x)y)]², where V₀ = 0.0100 E_h, the barrier-top imaginary frequency has magnitude ω_b = 0.0052 E_h, a = [2V₀/(mω_b²)]¹ᐟ², D(x) = ω_e/[4χ_e(x)], b(x) = [2mω_eχ_e(x)]¹ᐟ², and χ_e(x) = χ_∞ + (χ₀ − χ_∞) exp[−x²/(2σ²)] with σ = 0.35 a₀.

Use the Morse level convention E_n = ω_e(n + 1/2) − ω_eχ_e(n + 1/2)², so the supplied 0→1 and 0→2 lines determine ω_e and the dimensionless χ_e rather than being required to satisfy the harmonic relation E₀₂ = 2E₀₁. The transition energies (E₀₁,E₀₂) are (0.002928, 0.005784) E_h in the reactant asymptote and (0.00246, 0.00438) E_h at the barrier top; infer the common ω_e and the two values χ_∞ and χ₀ from these pairs.

Evaluate the method on one 48-bead ring polymer with 24 beads on each imaginary-time half, using the primitive action S = Σ_i {m|X_{i+1} − X_i|²/(2d_i) + d_i[V(X_i) + V(X_{i+1})]/2}, where d_i = 2τ/48 on the first half and 2(β − τ)/48 on the second. Use the dividing surface x = 0 to fix the x coordinate of the bead at each junction, take time-split derivatives at fixed bead numbers, and evaluate the reactant partition function with the same 48-bead discretization.

Use the first-order steepest-descent flux–flux estimator k = (2Z_r)⁻¹∫c_ff(t)dt, with c_ff(−iτ) = C(τ)∫dX Φ(X)exp[−S(X;τ)], C(τ) = (4m²)⁻¹(m/2π)⁴⁸d_a⁻²⁴d_b⁻²⁴, and Z_r = (m/2π)⁴⁸(β/48)⁻⁴⁸∫dX exp(−S_r), including the method's four-end-bead flux factor Φ, both mirror-image stationary paths, and |Φ|. Combine the dimensionless correction as ℏΓ₁ = ℏΓ₁⁽ˣ⁾ + ℏΓ₁⁽ᵗ⁾ − ℏΓ₁⁽ʳ⁾ and use the cumulant form k₁c = k₀ exp(ℏΓ₁); for the determinant part of the temporal prefactor use Ω″/Ω = −(2D₂ − 3D₁²)/4, with D₁ = F′/F and D₂ = F″/F (the corresponding 1/8 coefficient belongs to the Taylor expansion, not to the second derivative).

At β = 3600 E_h⁻¹, compute the thermal rate constant per unit length in units of 10⁻¹² atomic units. In your reasoning, give the inferred Morse parameters, the leading-order rate, and the parts of the first-order correction contributed by the fixed-split path integral, the time integration, and the reactant partition function for x = 0, then state what they imply about the reliability of the leading-order rate and how much the stretch's anharmonicity matters. Also state how changing the dividing surface affects the fixed-split contribution, the time-integration contribution, and their sum.

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

01_assemble_action_derivatives

Goal
----
Evaluate a partial derivative with respect to the imaginary-time split of the discretized flux-side ring-polymer action on the coupled Eckart-Morse surface, together with its gradient, Hessian and bead-resolved third- and fourth-derivative tensors.

```python
def assemble_action_derivatives(beads: "np.ndarray", surface: "np.ndarray", beta: float, tau: float,
                                order: int) -> "np.ndarray":
    """Return the order-th partial tau derivative of the discretized action and its coordinate derivatives.

    Atomic units with hbar = 1 are used throughout. ``surface`` holds
    ``(V0, a, m, omega, chi_inf, chi_0, sigma)`` and defines, for the two
    coordinates (x, y), both of mass ``m``, the potential

        V(x, y) = V0 sech^2(x / a) + D(x) [1 - exp(-b(x) y)]^2,
        D(x) = omega / (4 chi(x)),   b(x) = sqrt(2 m omega chi(x)),
        chi(x) = chi_inf + (chi_0 - chi_inf) exp(-x^2 / (2 sigma^2)),

    a symmetric Eckart barrier along x and a Morse stretch along y with
    x-independent harmonic frequency ``omega`` and anharmonicity constant
    ``chi(x)``. The ring polymer has ``N`` beads ``X_0, ..., X_{N-1}``
    (``N`` even, ``X_N = X_0``); segment ``i`` joins beads ``i`` and
    ``i + 1`` and has time step ``d_i = tau / (N/2)`` for ``i < N/2`` and
    ``d_i = (beta - tau) / (N/2)`` otherwise. The action is

        S(X; tau) = sum_i [ m |X_{i+1} - X_i|^2 / (2 d_i)
                            + d_i (V(X_i) + V(X_{i+1})) / 2 ].

    The derivative with respect to ``tau`` is partial: bead positions and the
    bead numbers of the two halves are held fixed. Coordinates are flattened
    bead-major, index ``2 i + c`` with ``c = 0`` for x and ``c = 1`` for y.

    Parameters
    ----------
    beads : np.ndarray
        Bead positions, shape ``(N, 2)``, with ``N`` even and ``N >= 4``.
        Every bead must lie on the line ``y = 0``.
    surface : np.ndarray
        The seven positive surface parameters described above.
    beta : float
        Total imaginary time (inverse temperature).
    tau : float
        Duration of the first half, ``0 < tau < beta``.
    order : int
        Order of the partial tau derivative, from 0 (the action itself) to 4.

    Returns
    -------
    np.ndarray
        One-dimensional array of length ``1 + 2N + 4N^2 + 8N + 16N``, the
        concatenation of: the order-th partial tau derivative ``S_n`` of the
        action; the gradient of ``S_n`` (length ``2N``); the Hessian of
        ``S_n`` (``2N x 2N``, row-major); and, for every bead ``i``, the third
        (``2 x 2 x 2``) and then, after all third-derivative tensors, the
        fourth (``2 x 2 x 2 x 2``) derivative tensor of ``S_n`` with respect
        to the coordinates of bead ``i`` alone, in bead order.

    Raises
    ------
    ValueError
        If ``beads`` is not a finite ``(N, 2)`` array with even ``N >= 4``,
        if any bead has ``y != 0``, if ``surface`` does not hold seven finite
        positive numbers, if ``tau`` does not satisfy ``0 < tau < beta``, or
        if ``order`` is not an integer from 0 to 4.
    """
    return derivatives
```

### Step 2

02_locate_flux_instanton

Goal
----
Locate the delocalized stationary ring-polymer path of the flux-side action with one bead of each half fixed on the dividing surface x = 0, and differentiate that path twice with respect to the imaginary-time split.

```python
def locate_flux_instanton(surface: "np.ndarray", beta: float, tau: float, n_beads: int) -> "np.ndarray":
    """Return the stationary flux-side ring-polymer path and its first two tau derivatives.

    The surface, the ``N``-bead action ``S(X; tau)`` and the coordinate
    layout are those of ``assemble_action_derivatives``. The x coordinates of
    beads ``0`` and ``N/2`` (the junctions of the two halves) are fixed on the
    dividing surface ``x = 0``; every bead lies on ``y = 0``; all other
    coordinates are free. The stationary path ``X(tau)`` is the minimum of
    ``S(X; tau)`` over the free coordinates in which beads ``1, ..., N/2 - 1``
    have ``x > 0`` and beads ``N/2 + 1, ..., N - 1`` have ``x < 0``. It must
    be converged to full double precision. Its derivatives with respect to
    ``tau`` are total derivatives along the family of stationary paths, with
    the bead numbers of the two halves fixed and the two pinned coordinates
    remaining at zero.

    Parameters
    ----------
    surface : np.ndarray
        The seven surface parameters ``(V0, a, m, omega, chi_inf, chi_0, sigma)``.
    beta : float
        Total imaginary time; it must exceed ``2 pi / omega_b`` with
        ``omega_b = sqrt(2 V0 / (m a^2))``.
    tau : float
        Duration of the first half, ``0 < tau < beta``.
    n_beads : int
        Number of beads ``N``, even and at least 4.

    Returns
    -------
    np.ndarray
        Array of shape ``(3, N, 2)``: the stationary bead positions, their
        first derivatives with respect to ``tau`` and their second
        derivatives with respect to ``tau``.

    Raises
    ------
    ValueError
        If ``n_beads`` is not an even integer of at least 4, if ``beta`` does
        not exceed ``2 pi / omega_b``, if the inputs are invalid in the sense
        of ``assemble_action_derivatives``, or if no stationary path with the
        stated sign pattern is found.
    """
    return path
```

### Step 3

03_compute_spatial_correction

Goal
----
Compute the first-order hbar correction coefficient of the discretized flux-flux path integral at a fixed imaginary-time split.

```python
def compute_spatial_correction(surface: "np.ndarray", beta: float, tau: float, n_beads: int) -> float:
    """Return the fixed-split first-order correction coefficient of the flux-flux path integral.

    Atomic units with hbar = 1 are used throughout.

    Parameters
    ----------
    surface : np.ndarray
        The seven surface parameters ``(V0, a, m, omega, chi_inf, chi_0, sigma)``.
    beta : float
        Total imaginary time.
    tau : float
        Duration of the first half, ``0 < tau < beta``.
    n_beads : int
        Number of beads ``N``, even and at least 4.

    Returns
    -------
    float
        The first-order correction coefficient at the specified imaginary-time split.

    Raises
    ------
    ValueError
        If the inputs are invalid.
    """
    return correction
```

### Step 4

04_compute_action_time_derivatives

Goal
----
Evaluate the stationary action as a function of the imaginary-time split together with its first four total derivatives with respect to that split.

```python
def compute_action_time_derivatives(surface: "np.ndarray", beta: float, tau: float,
                                    n_beads: int) -> "np.ndarray":
    """Return the stationary action and its first four total derivatives with respect to tau.

    Let ``X(tau)`` be the stationary path of ``locate_flux_instanton`` and
    ``W(tau) = S(X(tau); tau)`` the action of ``assemble_action_derivatives``
    evaluated on it, as a function of the split ``tau`` with the bead numbers
    of the two halves fixed and the two pinned coordinates at zero.

    Parameters
    ----------
    surface : np.ndarray
        The seven surface parameters ``(V0, a, m, omega, chi_inf, chi_0, sigma)``.
    beta : float
        Total imaginary time, as in ``locate_flux_instanton``.
    tau : float
        Duration of the first half, ``0 < tau < beta``.
    n_beads : int
        Number of beads ``N``, even and at least 4.

    Returns
    -------
    np.ndarray
        Array ``[W, dW/dtau, d2W/dtau2, d3W/dtau3, d4W/dtau4]`` of shape ``(5,)``.

    Raises
    ------
    ValueError
        If the inputs are invalid in the sense of ``locate_flux_instanton``.
    """
    return derivatives

# EXPECTED RETURN
#
```

### Step 5

05_compute_prefactor_time_derivatives

Goal
----
Evaluate the first and second total derivatives of the logarithm of the flux-flux prefactor with respect to the imaginary-time split, following the stationary path.

```python
def compute_prefactor_time_derivatives(surface: "np.ndarray", beta: float, tau: float,
                                       n_beads: int) -> "np.ndarray":
    """Return the first and second total tau derivatives of the log prefactor.

    Derivatives are total with respect to ``tau`` along the stationary-path
    family, with the bead numbers of the two halves held fixed.

    Parameters
    ----------
    surface : np.ndarray
        The seven surface parameters ``(V0, a, m, omega, chi_inf, chi_0, sigma)``.
    beta : float
        Total imaginary time.
    tau : float
        Duration of the first half, ``0 < tau < beta``.
    n_beads : int
        Number of beads ``N``, even and at least 4.

    Returns
    -------
    np.ndarray
        Array ``[d ln|A| / dtau, d^2 ln|A| / dtau^2]`` of shape ``(2,)``.

    Raises
    ------
    ValueError
        If the inputs are invalid.
    """
    return derivatives
```

### Step 6

06_compute_reactant_correction

Goal
----
Compute the first-order hbar correction coefficient of the discretized reactant partition function.

```python
def compute_reactant_correction(surface: "np.ndarray", beta: float, n_beads: int) -> float:
    """Return the first-order correction coefficient of the discretized reactant partition function.

    Atomic units with hbar = 1 are used throughout.

    Parameters
    ----------
    surface : np.ndarray
        The seven surface parameters ``(V0, a, m, omega, chi_inf, chi_0, sigma)``.
    beta : float
        Total imaginary time, positive and finite.
    n_beads : int
        Number of beads ``N``, even and at least 4.

    Returns
    -------
    float
        The first-order reactant correction coefficient ``G_r``.

    Raises
    ------
    ValueError
        If ``n_beads`` is not an even integer of at least 4, if ``beta`` is not
        positive and finite, or if ``surface`` does not hold seven finite
        positive numbers.
    """
    return correction
```

### Step 7

07_compute_corrected_rate

Goal
----
Build the coupled Eckart-Morse surface from spectroscopic data and return the cumulant-resummed first-order corrected ring-polymer instanton rate constant per unit length.

```python
def compute_corrected_rate(barrier: "np.ndarray", stretch_lines: "np.ndarray", width: float, mass: float,
                           beta: float, n_beads: int) -> float:
    """Return the cumulant-resummed first-order instanton rate constant in units of 1e-12 a.u.

    Atomic units with hbar = 1 are used throughout.

    Parameters
    ----------
    barrier : np.ndarray
        ``(V0, omega_b)``, the positive Eckart barrier height and magnitude of
        its barrier-top imaginary frequency.
    stretch_lines : np.ndarray
        Shape ``(2, 2)``. Each row contains the Morse stretch's ``(E01, E02)``
        transition energies, first in the reactant asymptote and then at the
        barrier top. All entries must be positive, both rows must imply the
        same harmonic frequency, and both anharmonicity constants must be
        positive.
    width : float
        Positive standard deviation of the Gaussian interpolation of the
        stretch anharmonicity.
    mass : float
        Positive mass of both coordinates.
    beta : float
        Inverse temperature, larger than ``2 pi / omega_b``.
    n_beads : int
        Number of beads ``N``, even and at least 4.

    Returns
    -------
    float
        The cumulant-resummed first-order rate constant per unit length,
        multiplied by ``1e12``.

    Raises
    ------
    ValueError
        If ``barrier``, ``stretch_lines``, ``width`` or ``mass`` do not have
        the stated shapes and positive finite values, if either site gives a
        non-positive anharmonicity constant, if the two sites' harmonic
        frequencies differ by more than 1e-9 relative, or if ``beta`` and
        ``n_beads`` do not satisfy the stated conditions.
    """
    return rate
```
