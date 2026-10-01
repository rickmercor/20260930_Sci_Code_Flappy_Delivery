# Chemistry-Computational_Chemistry-8

## Background

Damage and fracture in elastomers, gels and other polymer networks begin at the molecular scale, when covalent backbone bonds in load-bearing strands break under tension. Covalent bond rupture is thermally activated, so an applied force does not snap a bond directly but lowers its free-energy barrier and accelerates rupture by many orders of magnitude. Predicting how quickly a stretched strand breaks, and where along it the break is likely to occur, is therefore a basic input to network-scale models of polymer damage, and it depends on how the chain's conformational fluctuations distribute the load among its bonds.

## Problem

I measure how fast single polymer strands break under tension and where their first backbone scission occurs, and I want to use those observations to predict a loading geometry I have not measured. My strand has 11 identical atoms of mass 1.99e-26 kg joined by 10 backbone bonds, each a Morse spring with beta D_e = 279, a l_e = 2.15 and l_e = 1.525e-10 m, where beta = 1/(k_B T) and T = 293.15 K. The angle phi between successive bond vectors carries the penalty k_phi (phi - phi_e)^2 / 2, but both k_phi and phi_e are unknown; restrict beta k_phi pi^2 to 1000-8000 and phi_e to 50-100 degrees, with free torsions and no nonbonded interactions. With the free strand pulled by equal and opposite forces at f l_e/D_e = 0.94, its mean time to first scission is 18.921926 ns, and 0.99662638 of first scissions occur in either of its two terminal bonds. Infer the two bending parameters jointly from those observables, then predict the mean time to first scission, in picoseconds, to at least four significant figures, when atom 0 is held fixed, and the last atom is pulled at f l_e/D_e = 1.04. Model the first scission with equilibrium transition-state theory; neglect friction and recrossing, and converge any discretization you use. In the reasoning, also state the inferred beta k_phi pi^2 and phi_e, the barrier height and scission fraction of the tethered terminal bond in the requested configuration, and what the result implies about where the strand breaks.

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

01_compute_bond_kinetic_prefactors

Goal
----
Return, for every backbone bond of a chain that is either tethered at one end or free, the frequency prefactor of its transition-state rupture rate when bond lengths are measured in units of the equilibrium bond length.

```python
# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def compute_bond_kinetic_prefactors(
    n_bonds: int,
    atom_mass_kg: float,
    bond_length_m: float,
    temperature_k: float,
    tethered: bool,
) -> "np.ndarray":
    """Return the kinetic frequency prefactor of every bond of a chain.

    The chain has atoms 0 to ``n_bonds`` of identical mass ``atom_mass_kg``,
    and bond ``i`` (1-based) joins atoms ``i - 1`` and ``i``. If ``tethered``
    is True, atom 0 is held fixed and atoms 1 to ``n_bonds`` carry
    Maxwell-Boltzmann momenta at ``temperature_k``; otherwise every atom
    carries them. For each bond, return the equilibrium average of the
    positive part of the rate of change of its length, divided by
    ``bond_length_m`` so that the value is a frequency in s^-1. Use
    k_B = 1.380649e-23 J/K.

    Parameters
    ----------
    n_bonds : int
        Number of backbone bonds, at least 1.
    atom_mass_kg : float
        Mass of every atom in kg.
    bond_length_m : float
        Equilibrium bond length in m, the unit of the bond-length coordinate.
    temperature_k : float
        Temperature in K.
    tethered : bool
        Whether atom 0 is held fixed.

    Returns
    -------
    np.ndarray
        Float array of shape ``(n_bonds,)`` in s^-1; index 0 is the bond
        containing atom 0.

    Raises
    ------
    ValueError
        If ``n_bonds`` is not an integer of at least 1 (booleans are
        rejected), if any of ``atom_mass_kg``, ``bond_length_m`` and
        ``temperature_k`` is not a finite positive real number, or if
        ``tethered`` is not a bool.
    """
    return prefactors
```

### Step 2

02_build_bending_kernel

Goal
----
Tabulate the logarithm of the orientational coupling at every link of a possibly heterogeneous chain, on a Gauss-Legendre grid of polar angles measured from the force direction, after integrating out relative azimuth.

```python
# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def build_bending_kernel(
    n_theta: int,
    n_omega: int,
    stiffness_reduced: "np.ndarray",
    equilibrium_angle_deg: "np.ndarray",
) -> "np.ndarray":
    """Return log azimuth-integrated bending kernels for successive links.

    Polar angles are the ``n_theta`` Gauss-Legendre nodes of ``[0, pi]``
    (``numpy.polynomial.legendre.leggauss`` mapped affinely, in its increasing
    order). The one-dimensional inputs contain one stiffness and equilibrium
    angle for each link between successive bonds. Entry ``[q, a, b]`` is the
    natural logarithm of the coupling at link ``q`` between the successor at
    polar angle ``theta_a`` and its predecessor at ``theta_b``. The coupling is
    the integral over relative azimuth ``omega`` in ``[0, 2 pi)`` of
    ``exp(-beta v_ben(phi))``, with ``beta v_ben = 0.5 * beta_k_phi[q] *
    (phi - phi_e[q])**2``, ``beta_k_phi = stiffness_reduced / pi**2``, and
    angles converted to radians. Use the ``n_omega``-point periodic trapezoidal
    rule at nodes ``2 pi k / n_omega`` and accumulate it with log-sum-exp so
    every returned entry stays finite even when the ordinary kernel underflows.

    Parameters
    ----------
    n_theta : int
        Number of polar-angle nodes, at least 2.
    n_omega : int
        Number of azimuthal nodes, at least 4.
    stiffness_reduced : np.ndarray
        Non-empty one-dimensional array of linkwise reduced bending stiffnesses
        ``beta k_phi pi**2``, all finite and nonnegative.
    equilibrium_angle_deg : np.ndarray
        One-dimensional array of the same shape containing linkwise equilibrium
        angles in [0, 180] degrees.

    Returns
    -------
    np.ndarray
        Finite float array of shape ``(n_links, n_theta, n_theta)`` containing
        natural logarithms of the azimuth-integrated kernels.

    Raises
    ------
    ValueError
        If ``n_theta`` is not an integer of at least 2 or ``n_omega`` not an
        integer of at least 4 (booleans are rejected), if
        either link-parameter input is not a non-empty one-dimensional array,
        their shapes differ, a stiffness is negative or non-finite, or an
        equilibrium angle is non-finite or lies outside [0, 180].
    """
    return kernel
```

### Step 3

03_compute_intact_bond_weights

Goal
----
For every bond of a Morse chain under constant tension, integrate its Boltzmann weight over the intact range of reduced bond lengths at each polar angle of a Gauss-Legendre grid, and return the natural logarithm.

```python
# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def compute_intact_bond_weights(
    thresholds: "np.ndarray",
    force_reduced: float,
    beta_de: float,
    a_le: float,
    n_theta: int,
    n_length: int,
) -> "np.ndarray":
    """Return the log intact weight of every bond on the polar-angle grid.

    Lengths are reduced, ``x = l / l_e``. Bond ``j`` has Morse energy
    ``beta v(x) = beta_de * (1 - exp(-a_le * (x - 1)))**2`` and feels the
    reduced force ``force_reduced = f l_e / D_e``, so the force couples to the
    projection ``x cos(theta)`` of the bond on the force axis. Entry
    ``[j, k]`` is the natural log of the integral, over ``0.5 <= x <=
    thresholds[j]``, of the Boltzmann factor of the stretching energy and the
    force coupling times the polar part ``x**2 sin(theta_k)`` of the bond-vector
    volume element (no azimuthal factor). ``theta_k`` are the ``n_theta``
    Gauss-Legendre nodes of ``[0, pi]`` in ``leggauss`` order, and the length
    integral is the ``n_length``-node Gauss-Legendre rule on
    ``[0.5, thresholds[j]]``. Evaluate the quadrature in the log domain so
    the returned logarithms remain finite when the force bias or Morse depth
    makes the unnormalized Boltzmann factors overflow or underflow.

    Parameters
    ----------
    thresholds : np.ndarray
        One-dimensional reduced rupture thresholds, each finite and above 0.5.
    force_reduced : float
        Reduced force, finite and nonnegative.
    beta_de : float
        Reduced dissociation energy, finite and positive.
    a_le : float
        Reduced inverse Morse range, finite and positive.
    n_theta : int
        Number of polar-angle nodes, at least 2.
    n_length : int
        Number of bond-length nodes, at least 2.

    Returns
    -------
    np.ndarray
        Float array of shape ``(len(thresholds), n_theta)``.

    Raises
    ------
    ValueError
        If ``thresholds`` is not a non-empty one-dimensional array of finite
        values above 0.5, if ``force_reduced`` is negative or not finite, if
        ``beta_de`` or ``a_le`` is not finite and positive, or if ``n_theta``
        or ``n_length`` is not an integer of at least 2 (booleans rejected).
    """
    return log_weights
```

### Step 4

04_compute_angular_weights

Goal
----
Propagate normalized forward and backward orientational messages through the link-specific log kernels of a chain and return, for every bond, the angular weight imposed by the rest of the intact chain.

```python
# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def compute_angular_weights(log_intact: "np.ndarray", log_kernels: "np.ndarray") -> "np.ndarray":
    """Return the angular weight imposed on each bond by the rest of the chain.

    Angles are the ``n_theta`` Gauss-Legendre nodes of ``[0, pi]`` in
    ``leggauss`` order, and every angular integral uses that rule.
    ``log_intact[j]`` is the log intact weight of bond ``j`` on the grid, and
    ``log_kernels[q, a, b]`` is the log coupling at link ``q`` between bond
    ``q + 1`` at node ``a`` and predecessor bond ``q`` at node ``b``. Thus a
    chain of ``n_bonds`` uses exactly ``n_bonds - 1`` link kernels. For bond
    ``i`` (0-based), the forward shape is the intact-basin
    weight of bonds ``0..i-1`` conditioned on the polar angle of bond ``i``, and
    the backward shape is that of bonds ``i+1..n-1``; each shape is normalized
    to unit integral over the angle, and a side with no bonds has the uniform
    shape ``1 / pi``. The returned weight of bond ``i`` is the product of its
    forward and backward shapes. Perform the two recurrences with log-sum-exp,
    including the Gauss-Legendre integration weights inside each reduction,
    and normalize every message in the log domain. Adding an arbitrary finite
    constant to any row of ``log_intact`` or to any full link kernel must not
    change the result, even when ordinary exponentiation would overflow or
    underflow.

    Parameters
    ----------
    log_intact : np.ndarray
        Finite float array of shape ``(n_bonds, n_theta)``.
    log_kernels : np.ndarray
        Finite float array of shape ``(n_bonds - 1, n_theta, n_theta)``. For a
        one-bond chain this has shape ``(0, n_theta, n_theta)``.

    Returns
    -------
    np.ndarray
        Float array of shape ``(n_bonds, n_theta)``.

    Raises
    ------
    ValueError
        If ``log_intact`` is not a finite two-dimensional array with at least
        one row and two columns, or if ``log_kernels`` is not a finite array
        with the required link and angle dimensions.
    """
    return weights
```

### Step 5

05_evaluate_bond_pmf

Goal
----
Evaluate the potential of mean force and its first two reduced-length derivatives for one bond of a tensioned Morse chain, given the angular weight imposed by the rest of the intact chain.

```python
# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def evaluate_bond_pmf(
    x_values: "np.ndarray",
    angular_weight: "np.ndarray",
    force_reduced: float,
    beta_de: float,
    a_le: float,
) -> "np.ndarray":
    """Return the reduced bond-length PMF and its first two derivatives.

    The bond has reduced length ``x = l / l_e``, Morse energy
    ``beta v(x) = beta_de * (1 - exp(-a_le * (x - 1)))**2`` and feels the
    reduced force ``force_reduced = f l_e / D_e`` along the polar axis.
    ``angular_weight`` is the weight that the rest of the intact chain places
    on the bond's polar angle, tabulated at the Gauss-Legendre nodes of
    ``[0, pi]`` (``leggauss`` order, one node per entry); it does not include
    the bond's own ``sin(theta)`` measure, and every angular integral uses
    that rule. Column 0 is ``beta W(x) - beta W(x_values[0])`` in units of
    k_B T, where ``W`` is the potential of mean force of the bond length at
    fixed ``x`` in three dimensions. Columns 1 and 2 are the analytic first
    and second derivatives of ``beta W`` with respect to ``x`` at every
    requested value. The constant subtraction affects only column 0.

    Parameters
    ----------
    x_values : np.ndarray
        One-dimensional, non-empty, finite, positive reduced lengths.
    angular_weight : np.ndarray
        One-dimensional finite nonnegative weights, at least two entries and
        a positive sum.
    force_reduced : float
        Reduced force, finite and nonnegative.
    beta_de : float
        Reduced dissociation energy, finite and positive.
    a_le : float
        Reduced inverse Morse range, finite and positive.

    Returns
    -------
    np.ndarray
        Finite float array of shape ``(len(x_values), 3)``. Column 0 is the
        relative PMF and starts at zero; columns 1 and 2 are its slope and
        curvature with respect to reduced length.

    Raises
    ------
    ValueError
        If ``x_values`` is not a non-empty one-dimensional array of finite
        positive values, if ``angular_weight`` is not a one-dimensional finite
        nonnegative array with at least two entries and a positive sum, if
        ``force_reduced`` is negative or not finite, or if ``beta_de`` or
        ``a_le`` is not finite and positive.
    """
    return pmf
```

### Step 6

06_locate_pmf_stationary_points

Goal
----
For every bond of a tensioned chain, locate the bonded minimum and the barrier top of its bond-length potential of mean force from the angular weight imposed by the rest of the chain.

```python
# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def locate_pmf_stationary_points(
    angular_weights: "np.ndarray",
    force_reduced: float,
    beta_de: float,
    a_le: float,
) -> "np.ndarray":
    """Return the bonded minimum and barrier top of each bond's PMF.

    Row ``i`` of ``angular_weights`` is the angular weight of bond ``i`` on the
    Gauss-Legendre nodes of ``[0, pi]`` (``leggauss`` order), and the PMF plus
    its first two derivatives are those returned by ``evaluate_bond_pmf`` with
    the same force and Morse parameters. Tabulate the derivative column at
    ``x = 0.9 + 0.002 k`` for
    ``k = 0, ..., 2050``. The bonded minimum lies in the first interval where
    the derivative goes from negative to nonnegative, and the barrier top in
    the first later interval where it goes from positive to nonpositive. Refine
    each root of the derivative inside its interval to an absolute accuracy of
    ``1e-12`` in ``x``. Verify from the analytic curvature column that the
    first root has positive curvature and the second has negative curvature.

    Parameters
    ----------
    angular_weights : np.ndarray
        Finite nonnegative array ``(n_bonds, n_theta)``, ``n_theta >= 2``, positive row sums.
    force_reduced : float
        Reduced force ``f l_e / D_e``, finite and nonnegative.
    beta_de : float
        Reduced dissociation energy, finite and positive.
    a_le : float
        Reduced inverse Morse range, finite and positive.

    Returns
    -------
    np.ndarray
        Float array of shape ``(n_bonds, 2)`` holding the reduced lengths of
        the bonded minimum and the barrier top of each bond.

    Raises
    ------
    ValueError
        If ``angular_weights`` is not a 2D finite nonnegative array with at least one
        row, two columns and positive row sums, if ``force_reduced`` is negative or not
        finite, if ``beta_de`` or ``a_le`` is not finite and positive, or if some bond
        has no bonded minimum followed by a barrier top on the tabulated range.
    """
    return points
```

### Step 7

07_compute_bond_scission_rates

Goal
----
Convert each bond's tabulated potential of mean force into its bond-resolved transition-state rupture rate, with the dividing surface at the bond's rupture threshold and the intact basin below it.

```python
# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def compute_bond_scission_rates(
    pmf_table: "np.ndarray",
    barrier_tops: "np.ndarray",
    prefactors: "np.ndarray",
) -> "np.ndarray":
    """Return the transition-state rupture rate of every bond in s^-1.

    Row ``i`` of ``pmf_table`` holds ``beta W_i`` in units of k_B T, up to an
    arbitrary additive constant per row: column 0 at the reduced rupture
    threshold ``barrier_tops[i]``, and columns ``1..n_length`` at the
    ``n_length`` Gauss-Legendre nodes of ``[0.5, barrier_tops[i]]`` in
    ``leggauss`` order. The intact basin of bond ``i`` is ``0.5 <= x <=
    barrier_tops[i]`` and its population integral uses that same
    Gauss-Legendre rule. ``prefactors[i]`` is the kinetic frequency prefactor
    of bond ``i`` for the reduced length coordinate, in s^-1.

    Parameters
    ----------
    pmf_table : np.ndarray
        Finite float array of shape ``(n_bonds, n_length + 1)``, ``n_length >= 2``.
    barrier_tops : np.ndarray
        Finite float array of shape ``(n_bonds,)`` with every entry above 0.5.
    prefactors : np.ndarray
        Finite positive float array of shape ``(n_bonds,)``.

    Returns
    -------
    np.ndarray
        Float array of shape ``(n_bonds,)`` in s^-1.

    Raises
    ------
    ValueError
        If ``pmf_table`` is not a finite two-dimensional array with at least
        one row and three columns, if ``barrier_tops`` is not a finite
        one-dimensional array of matching length with entries above 0.5, or if
        ``prefactors`` is not a finite positive one-dimensional array of
        matching length.
    """
    return rates
```

### Step 8

08_predict_tethered_scission_lifetime

Goal
----
Infer both the bending stiffness and the equilibrium bond angle of a Morse chain from its free-strand lifetime and terminal-scission fraction, then predict its tethered lifetime at another force. Orchestrator: yes; In each trial pair and the prediction, use all seven preceding reference steps to build prefactors, propagate the angular statistics, converge the bond-specific dividing surfaces, evaluate the PMFs and sum the bond-resolved rates.

```python
# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def predict_tethered_scission_lifetime(
    measured_lifetime_ns: float = 18.921926,
    measured_terminal_share: float = 0.99662638,
    measured_force_reduced: float = 0.94,
    predicted_force_reduced: float = 1.04,
    n_bonds: int = 10,
    beta_de: float = 279.0,
    a_le: float = 2.15,
    atom_mass_kg: float = 1.99e-26,
    bond_length_m: float = 1.525e-10,
    temperature_k: float = 293.15,
    n_theta: int = 160,
    n_omega: int = 256,
    n_length: int = 600,
    stiffness_low: float = 1000.0,
    stiffness_high: float = 8000.0,
    angle_low_deg: float = 50.0,
    angle_high_deg: float = 100.0,
) -> float:
    """Return the predicted tethered mean time to first scission, in ps.

    The calibration data belong to a free strand at ``measured_force_reduced``.
    ``measured_terminal_share`` is the probability that the first broken bond is
    either terminal bond. Infer ``s = beta k_phi pi**2`` and the equilibrium
    bond-vector angle simultaneously. For each pair, converge the bond-specific
    rupture thresholds from 2.0 until no threshold changes by ``1e-11`` (at most
    60 sweeps), build the bond-resolved rates, and match the log lifetime and the
    log odds of the combined terminal share. Use bounded nonlinear least squares
    in ``(log(s), angle_deg)`` over the supplied bounds, starting at both interval
    midpoints, with ``xtol = ftol = gtol = 1e-10`` and at most 100 evaluations.
    Reject a fit whose largest residual exceeds ``2e-6``. With the inferred pair,
    repeat the same calculation with atom 0 fixed at
    ``predicted_force_reduced`` and return the lifetime in picoseconds.

    Every PMF uses ``n_length`` Gauss-Legendre nodes on ``[0.5, x_b]`` plus its
    converged barrier top. The kinetic prefactors, bending kernel, intact weights,
    angular messages, stationary points, PMFs and rates must come from the seven
    preceding functions. The defaults reproduce the problem statement.

    Raises
    ------
    ValueError
        If a measured observable or fit bound is invalid, fewer than three bonds
        are requested, the observations cannot be matched inside the bounds, a
        consuming step rejects an argument, or threshold iteration fails.
    """
    return 0.0
```
