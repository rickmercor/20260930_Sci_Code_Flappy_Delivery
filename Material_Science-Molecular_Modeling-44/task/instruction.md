# Material_Science-Molecular_Modeling-44

## Background

Covalent polymer chains carry mechanical load through their backbone, and when the load is large enough a backbone bond eventually dissociates. Because bond rupture is a thermally activated barrier crossing, its kinetics are described by transition-state theory: the rate is the equilibrium one-way flux through a dividing surface that separates the bonded state from the dissociated state, divided by the equilibrium population of the bonded state. For a single bond described by an anharmonic (Morse-type) potential, an applied tensile force tilts the potential, lowers the barrier between the bonded well and the dissociation plateau, and moves the barrier top toward the well until the two merge at a critical force beyond which rupture is barrierless. Phenomenological force-dependent rate laws (Bell-type and their extensions) are low-order expansions of this picture in the force.

In a polymer chain the situation is richer. The tension is transmitted along a backbone whose bond vectors have three-dimensional orientations, so that the force acting along a bond depends on its polar angle with respect to the pulling direction, and bond stretching competes with the orientational entropy of the bond-vector measure. In freely jointed chains the bond orientations are statistically independent and every bond sees the same effective free-energy landscape. In chains with a finite bending stiffness, neighbouring bond orientations are correlated through the bond angle, the preferred zigzag geometry constrains which absolute orientations a bond can adopt, and these constraints are truncated at the chain ends. The effective landscape for stretching one bond while the rest of the chain remains intact is a potential of mean force along that bond's length, and it becomes position dependent along the chain. Chain scission is then a first-passage problem with several competing rupture channels, one per bond, whose rates must be resolved individually and summed.

Transfer-matrix techniques are the natural tool for the equilibrium statistics of such chains: nearest-neighbour orientational correlations are propagated along the chain through an angular kernel, and constrained configurational integrals reduce to one-dimensional angular quadratures. Combined with variational transition-state theory, in which the dividing surface of each rupture channel is chosen where the flux through it is stationary, this yields deterministic, sampling-free rates that span many orders of magnitude and are exponentially sensitive to the activation barriers, so careful log-domain numerics are required.

The force a chain segment carries at a given end-to-end distance is the other half of the problem. Classical entropic chain models (freely jointed, freely rotating, worm-like) describe the low-force regime, but a highly stretched carbon backbone stores energy in bond stretching and bond-angle opening, so realistic single-chain models let the bond length and bond angle deform with the applied stretch and interpolate the entropic response of the freely rotating chain over the entire force range. Minimizing the sum of the energetic and entropic contributions at fixed extension yields effective bond parameters and the chain force, which can then be handed to a rupture-kinetics calculation.

## Problem

Mechanochemical scission of covalent backbone bonds is the elementary event of damage and fracture in loaded polymer networks. Two ingredients are needed to predict it for a chain segment stretched inside a network: the force the segment transmits at its imposed extension, and the rate at which that force breaks one of its bonds. Recent single-chain models supply both: a deformable freely rotating chain with stretch-dependent bond length and bond angle, built on an explicit force–extension relation for the freely rotating chain that is valid over the entire force range, gives the chain force at a prescribed end-to-end distance; and a bond-resolved, multichannel first-rupture formulation within variational transition-state theory, in which the constrained potential of mean force of every bond follows exactly from the chain's statistical mechanics with the orientational correlations induced by the bending stiffness, gives the scission rate at that force. Given the chain model, the imposed extension and the temperature, the combined method returns the mean time the segment survives before losing its first backbone bond.

Consider a chain of N + 1 identical atoms of mass m = 1.99 × 10⁻²⁶ kg at positions r_0, …, r_N, connected by N = 12 backbone bonds with bond vectors l_i = r_i − r_{i−1}, bond lengths l_i and polar angles θ_i measured from the direction of the chain force. Every bond stores a Morse stretching energy v_str(l) = D_e [1 − e^{−a(l − l_e)}]² with D_e = 1.13 × 10⁻¹⁸ J, a = 1.409 Å⁻¹ and l_e = 1.525 Å; every interior atom i = 1, …, N − 1 stores a harmonic bending energy v_ben(ϑ) = k_φ (ϑ − ϑ_e)²/2 in its valence angle ϑ_i, the angle between the two bonds that meet at atom i, with k_φ = 5.0 × 10⁻¹⁹ J rad⁻² and ϑ_e = 109.5°; torsional and non-bonded interactions are absent. The chain is in equilibrium at T = 293.15 K with k_B = 1.380649 × 10⁻²³ J K⁻¹, and it is held at an end-to-end distance r = 1.22 R_0, where R_0 = N l_e sin(ϑ_e/2) is the end-to-end length of the undeformed zigzag.

First obtain the chain force. Represent the stretched chain by a uniform, non-fluctuating bond length l and bond angle, chosen to minimize the Helmholtz free energy Ψ = N v_str(l) + (N − 1) v_ben(ϑ) + Ψ_ent at the prescribed end-to-end distance, where Ψ_ent is the entropic free energy of a freely rotating chain with that bond length and bond angle, obtained by integrating the explicit force–extension relation of the freely rotating chain mentioned above (expressed through the inverse Langevin function, evaluated exactly, of the extension relative to the contour length, the contour length and the Kuhn length of the freely rotating chain), using the closed-form entropic free energy that accompanies that relation; the chain force is f = dΨ/dr at the optimum. Then treat scission at that constant force, applied to atom N with atom 0 held fixed at the origin (a rigid anchor carrying no momentum) and all mobile atoms carrying classical Maxwell–Boltzmann momenta, as a multichannel first-rupture problem in bond-length space: bond i is ruptured once l_i ≥ l_i^‡, the chain is intact while every bond lies below its threshold, and the chain-scission rate k_chain is the equilibrium one-way flux from the intact basin through the dividing surface, normalized by the equilibrium population of the intact basin; this flux decomposes into bond-resolved rupture channels. Place every threshold variationally at the barrier top of the potential of mean force of that bond along its own length, computed with all other bonds constrained below their thresholds, and determine the coupled thresholds self-consistently. Evaluate each bond-resolved rate from the full flux-over-population expression of transition-state theory (the well is integrated, with no harmonic approximation), using the exact configurational statistics of the model with the nearest-neighbour orientational correlations resolved deterministically by quadrature, and sum the channels.

Report the mean time to the first scission, τ_chain = 1/k_chain, in hours as the final answer. State also the optimal bond length and bond angle of the stretched chain and the resulting chain force in nN, the kinetic prefactors of the anchored bond and of a non-anchored bond, the collinear reference rate, i.e. the rupture rate of one bond (any bond other than the anchored one) in a chain whose bonds are all held parallel to the force, the rate of that collinear chain of 12 bonds, the self-consistent rupture thresholds of the terminal and of the interior bonds in units of l_e, the activation barriers of a terminal bond, of a first nonterminal bond and of a central bond of the three-dimensional chain in units of D_e, the rupture rates of its two terminal bonds, the collinear critical force above which the transition-state construction ceases to apply, and, in one sentence, which bond of the chain ruptures fastest and why.

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

Implement **all 9 functions** below into a single file at `/app/solution.py`. Requirements:

- Use the EXACT function name and signature shown for each step.
- Define every function at module scope (importable).
- Do NOT include tests, example usage, prints, or a `__main__` block.
- Available libraries: numpy, scipy, sympy, pandas, scikit-learn, networkx.

Start your file with these imports:

```python
import numpy as np
```

### Step 1

dfrc_chain_force

Goal
----
Determine the chain force transmitted by a polymer chain held at a prescribed end-to-end distance with the deformable freely rotating chain (dFRC) model of the elasticity source (its Eqs. 7-9): the chain is represented by a uniform, non-fluctuating bond length l and bond-vector angle phi (the angle between successive bond vectors) chosen to minimize the Helmholtz free energy Psi(l, phi; r) = N v_str(l) + (N - 1) v_ben(phi) + Psi_ent(r; l, phi) at the prescribed end-to-end distance r = stretch_ratio * R_0, where R_0 = N l_e cos(phi_e / 2) is the end-to-end length of the undeformed zigzag, v_str(l) = D_e [1 - exp(-a (l - l_e))]^2 is the Morse stretching energy and v_ben(phi) = k_phi (phi - phi_e)^2 / 2. Psi_ent is the entropic free energy of a freely rotating chain with bond length l and bond angle phi, whose contour length is R_max = N l cos(phi / 2) and whose Kuhn length is l_k = 2 l cos(phi / 2) / (1 - cos phi); with the relative extension r* = r / R_max (0 < r* < 1) and eta = L^-1(r*) the exact inverse of the Langevin function L(x) = coth x - 1/x, the source's explicit force-extension relation (its Eq. 8) is f = (k_B T / l_k) { eta + (1/2) r*^2 / (1 - r*)^2 [1 - r*^(l_k / l - 1)] }, and its closed-form free energy (its Eq. 9) is Psi_ent = k_B T (R_max / l_k) [ r* eta + ln(eta / sinh eta) + (1 + r* (1 - r*)) / (2 (1 - r*)) + ln(1 - r*) - (1/2) B(r*; 2 + l_k / l, -1) ], where B(x; a, b) = int_0^x t^(a - 1) (1 - t)^(b - 1) dt is the incomplete beta function (here b = -1, integrand t^(a - 1) (1 - t)^-2, a = 2 + l_k / l). Eq. 9 is the integral of Eq. 8 over r plus an additive constant that depends on l and phi (through R_max / l_k and l_k / l); that constant must be kept, because it moves the optimum: integrating Eq. 8 from r = 0 instead does not give the source's free energy. Evaluate the inverse Langevin function exactly (no Pade or other approximant) and the incomplete beta function to a relative accuracy of 1e-10 or better for r* close to 1. The chain force is f = dPsi_ent / dr at the optimum, i.e. Eq. 8 evaluated at (l*, phi*). Return the optimal bond length in units of l_e, the optimal bond-vector angle in radians and the reduced chain force f l_e / D_e. When beta_kphi = 0 (freely jointed bonds) the angle is undetermined: use the freely jointed reduction l_k = l, R_max = N l, for which the correction term of Eq. 8 vanishes and Eq. 9 reduces to Psi_ent = N k_B T [r* eta + ln(eta / sinh eta)] up to an l-independent constant; optimize l alone and return 0.0 for the angle. The optimum is unique in the physical domain (r* < 1); converge it to 1e-12 in both variables (bond length in l_e, angle in radians). Finite-difference gradients cannot reach that accuracy next to the r* -> 1 singularity of Psi_ent; use the stationarity conditions with analytic derivatives: Psi_ent depends on l only through r*, so dPsi_ent/dl = -(r / l) f with f from Eq. 8, and dPsi_ent/dphi involves dB(r*; a, -1)/da = int_0^r* t^(a - 1) ln t (1 - t)^-2 dt. Reduced units: lengths in units of the equilibrium bond length l_e, energies in units of the dissociation energy D_e, so beta_de = D_e/(k_B T), a_le = a l_e and f_red = f l_e / D_e; angles in radians.

```python
def dfrc_chain_force(stretch_ratio: float, n_bonds: int, beta_de: float, a_le: float, beta_kphi: float,
                             phi_e: float) -> tuple:
    """Determine the chain force transmitted by a polymer chain held at a prescribed end-to-end distance with the deformable freely rotating chain (dFRC) model of the elasticity source (its Eqs. 7-9): the chain is represented by a uniform, non-fluctuating bond length l and bond-vector angle phi (the angle between successive bond vectors) chosen to minimize the Helmholtz free energy Psi(l, phi; r) = N v_str(l) + (N - 1) v_ben(phi) + Psi_ent(r; l, phi) at the prescribed end-to-end distance r = stretch_ratio * R_0, where R_0 = N l_e cos(phi_e / 2) is the end-to-end length of the undeformed zigzag, v_str(l) = D_e [1 - exp(-a (l - l_e))]^2 is the Morse stretching energy and v_ben(phi) = k_phi (phi - phi_e)^2 / 2. Psi_ent is the entropic free energy of a freely rotating chain with bond length l and bond angle phi, whose contour length is R_max = N l cos(phi / 2) and whose Kuhn length is l_k = 2 l cos(phi / 2) / (1 - cos phi); with the relative extension r* = r / R_max (0 < r* < 1) and eta = L^-1(r*) the exact inverse of the Langevin function L(x) = coth x - 1/x, the source's explicit force-extension relation (its Eq. 8) is f = (k_B T / l_k) { eta + (1/2) r*^2 / (1 - r*)^2 [1 - r*^(l_k / l - 1)] }, and its closed-form free energy (its Eq. 9) is Psi_ent = k_B T (R_max / l_k) [ r* eta + ln(eta / sinh eta) + (1 + r* (1 - r*)) / (2 (1 - r*)) + ln(1 - r*) - (1/2) B(r*; 2 + l_k / l, -1) ], where B(x; a, b) = int_0^x t^(a - 1) (1 - t)^(b - 1) dt is the incomplete beta function (here b = -1, integrand t^(a - 1) (1 - t)^-2, a = 2 + l_k / l). Eq. 9 is the integral of Eq. 8 over r plus an additive constant that depends on l and phi (through R_max / l_k and l_k / l); that constant must be kept, because it moves the optimum: integrating Eq. 8 from r = 0 instead does not give the source's free energy. Evaluate the inverse Langevin function exactly (no Pade or other approximant) and the incomplete beta function to a relative accuracy of 1e-10 or better for r* close to 1. The chain force is f = dPsi_ent / dr at the optimum, i.e. Eq. 8 evaluated at (l*, phi*). Return the optimal bond length in units of l_e, the optimal bond-vector angle in radians and the reduced chain force f l_e / D_e. When beta_kphi = 0 (freely jointed bonds) the angle is undetermined: use the freely jointed reduction l_k = l, R_max = N l, for which the correction term of Eq. 8 vanishes and Eq. 9 reduces to Psi_ent = N k_B T [r* eta + ln(eta / sinh eta)] up to an l-independent constant; optimize l alone and return 0.0 for the angle. The optimum is unique in the physical domain (r* < 1); converge it to 1e-12 in both variables (bond length in l_e, angle in radians). Finite-difference gradients cannot reach that accuracy next to the r* -> 1 singularity of Psi_ent; use the stationarity conditions with analytic derivatives: Psi_ent depends on l only through r*, so dPsi_ent/dl = -(r / l) f with f from Eq. 8, and dPsi_ent/dphi involves dB(r*; a, -1)/da = int_0^r* t^(a - 1) ln t (1 - t)^-2 dt. Reduced units: lengths in units of the equilibrium bond length l_e, energies in units of the dissociation energy D_e, so beta_de = D_e/(k_B T), a_le = a l_e and f_red = f l_e / D_e; angles in radians.

    Parameters
    ----------
    stretch_ratio : float
        Prescribed end-to-end distance divided by R_0 = N l_e cos(phi_e / 2) (positive).
    n_bonds : int
        Number of bonds N (>= 2).
    beta_de : float
        Reduced dissociation energy D_e / (k_B T).
    a_le : float
        Reduced Morse range parameter a l_e.
    beta_kphi : float
        Bending stiffness in units of k_B T per rad^2 (non-negative).
    phi_e : float
        Equilibrium angle between successive bond vectors in radians, strictly inside (0, pi).

    Returns
    -------
    result : tuple
        (l_star, phi_star, f_red) as native floats.

    Raises
    ------
    ValueError
        If stretch_ratio, beta_de or a_le is not finite positive, n_bonds is not an integer >= 2, beta_kphi is negative or not finite, phi_e is outside (0, pi), or no physical optimum exists for the supplied stretch ratio.
    """
    return result
```

### Step 2

collinear_reference_rate

Goal
----
Compute the collinear (one-dimensional) reference of the chain-scission problem: every bond is aligned with the force, so the bond-length potential of mean force is the Morse stretching energy tilted by the force (source, Sec. S3.2). Return the bonded minimum and the barrier top of that tilted potential (closed form), the activation barrier between them, and the bond-resolved transition-state-theory rate of one such bond evaluated from the full flux-over-population expression of the source (its Eq. S96) with the supplied kinetic prefactor, integrating the Boltzmann factor of the potential over the bonded interval [0, l_bar] with n_l trapezoid points. Reduced units: lengths in units of the equilibrium bond length l_e, energies in units of the dissociation energy D_e, so beta_de = D_e/(k_B T), a_le = a l_e and f_red = f l_e / D_e; angles in radians.

```python
def collinear_reference_rate(f_red: float, beta_de: float, a_le: float, prefactor: float,
                                     n_l: int = 20001) -> tuple:
    """Compute the collinear (one-dimensional) reference of the chain-scission problem: every bond is aligned with the force, so the bond-length potential of mean force is the Morse stretching energy tilted by the force (source, Sec. S3.2). Return the bonded minimum and the barrier top of that tilted potential (closed form), the activation barrier between them, and the bond-resolved transition-state-theory rate of one such bond evaluated from the full flux-over-population expression of the source (its Eq. S96) with the supplied kinetic prefactor, integrating the Boltzmann factor of the potential over the bonded interval [0, l_bar] with n_l trapezoid points. Reduced units: lengths in units of the equilibrium bond length l_e, energies in units of the dissociation energy D_e, so beta_de = D_e/(k_B T), a_le = a l_e and f_red = f l_e / D_e; angles in radians.

    Parameters
    ----------
    f_red : float
        Reduced force f l_e / D_e, positive and below a_le / 2.
    beta_de : float
        Reduced dissociation energy D_e / (k_B T).
    a_le : float
        Reduced Morse range parameter a l_e.
    prefactor : float
        Kinetic prefactor of the bond in 1/s (one-sided mean bond-length velocity divided by l_e).
    n_l : int
        Number of trapezoid points on [0, l_bar] (default 20001).

    Returns
    -------
    result : tuple
        (l_min, l_bar, barrier, rate) as native floats.

    Raises
    ------
    ValueError
        If f_red, beta_de, a_le or prefactor is not finite positive, n_l is not an integer >= 3, or f_red is at or above the collinear critical force a_le / 2.
    """
    return result
```

### Step 3

bending_kernel

Goal
----
Build the angular coupling kernel of two successive bonds on a polar-angle grid: for each pair of polar angles (theta, theta_prime) measured from the force axis, the Boltzmann factor of the harmonic bending energy in the bond angle between the two bond vectors, integrated over their relative azimuthal angle on [0, 2 pi) (source, Sec. S2.1; the same kernel underlies the transfer-matrix elasticity of the deformable-bond chain). Use the midpoint rule with n_omega equally spaced azimuths. The bond angle is the angle between the bond vectors, and the bending energy is k_phi (phi - phi_e)^2 / 2 in units of k_B T, i.e. beta_kphi = k_phi / (k_B T).

```python
def bending_kernel(theta: "np.ndarray", beta_kphi: float, phi_e: float, n_omega: int = 512) -> "np.ndarray":
    """Build the angular coupling kernel of two successive bonds on a polar-angle grid: for each pair of polar angles (theta, theta_prime) measured from the force axis, the Boltzmann factor of the harmonic bending energy in the bond angle between the two bond vectors, integrated over their relative azimuthal angle on [0, 2 pi) (source, Sec. S2.1; the same kernel underlies the transfer-matrix elasticity of the deformable-bond chain). Use the midpoint rule with n_omega equally spaced azimuths. The bond angle is the angle between the bond vectors, and the bending energy is k_phi (phi - phi_e)^2 / 2 in units of k_B T, i.e. beta_kphi = k_phi / (k_B T).

    Parameters
    ----------
    theta : np.ndarray
        Polar-angle grid in [0, pi] (radians), shape (n_theta,).
    beta_kphi : float
        Bending stiffness in units of k_B T per rad^2 (non-negative).
    phi_e : float
        Equilibrium bond angle between successive bond vectors in radians, in [0, pi].
    n_omega : int
        Number of midpoint azimuthal nodes (default 512).

    Returns
    -------
    kernel : np.ndarray
        Symmetric kernel of shape (n_theta, n_theta).

    Raises
    ------
    ValueError
        If theta is not a finite one-dimensional array inside [0, pi], beta_kphi is negative or not finite, phi_e is outside [0, pi], or n_omega is not a positive integer.
    """
    return kernel
```

### Step 4

log_intact_weight

Goal
----
Compute, on the polar-angle grid, the natural logarithm of the local intact weight of one bond whose rupture threshold is l_thr (source, Sec. S2.1, the local intact weight): I(theta) = sin(theta) * integral from 0 to l_thr of exp(-beta_de * [v_str(l) - f_red * l * cos(theta)]) * l^2 dl, with v_str the Morse stretching energy, i.e. the bond's Boltzmann factor integrated over its allowed length interval with the full three-dimensional bond-vector measure l^2 sin(theta) dl dtheta (the azimuth already integrated out). The sin(theta) factor of the measure is included in this weight, so that the polar-angle integrals of the later steps use plain dtheta quadrature weights. Use the trapezoid rule with n_l equally spaced points on [0, l_thr] and evaluate in the log domain so that the result is finite for beta_de of several hundred. Reduced units: lengths in units of the equilibrium bond length l_e, energies in units of the dissociation energy D_e, so beta_de = D_e/(k_B T), a_le = a l_e and f_red = f l_e / D_e; angles in radians.

```python
def log_intact_weight(theta: "np.ndarray", l_thr: float, f_red: float, beta_de: float, a_le: float,
                              n_l: int = 4001) -> "np.ndarray":
    """Compute, on the polar-angle grid, the natural logarithm of the local intact weight of one bond whose rupture threshold is l_thr (source, Sec. S2.1, the local intact weight): I(theta) = sin(theta) * integral from 0 to l_thr of exp(-beta_de * [v_str(l) - f_red * l * cos(theta)]) * l^2 dl, with v_str the Morse stretching energy, i.e. the bond's Boltzmann factor integrated over its allowed length interval with the full three-dimensional bond-vector measure l^2 sin(theta) dl dtheta (the azimuth already integrated out). The sin(theta) factor of the measure is included in this weight, so that the polar-angle integrals of the later steps use plain dtheta quadrature weights. Use the trapezoid rule with n_l equally spaced points on [0, l_thr] and evaluate in the log domain so that the result is finite for beta_de of several hundred. Reduced units: lengths in units of the equilibrium bond length l_e, energies in units of the dissociation energy D_e, so beta_de = D_e/(k_B T), a_le = a l_e and f_red = f l_e / D_e; angles in radians.

    Parameters
    ----------
    theta : np.ndarray
        Polar-angle grid strictly inside (0, pi), shape (n_theta,).
    l_thr : float
        Rupture threshold of the bond in units of l_e (positive).
    f_red : float
        Reduced force f l_e / D_e (positive).
    beta_de : float
        Reduced dissociation energy D_e / (k_B T).
    a_le : float
        Reduced Morse range parameter a l_e.
    n_l : int
        Number of trapezoid points on [0, l_thr] (default 4001).

    Returns
    -------
    log_i : np.ndarray
        Log intact weight on the theta grid.

    Raises
    ------
    ValueError
        If theta is not a finite one-dimensional array strictly inside (0, pi), l_thr, f_red, beta_de or a_le is not finite positive, or n_l is not an integer >= 3.
    """
    return log_i
```

### Step 5

constrained_pmf

Goal
----
Evaluate, at the supplied lengths, the constrained potential of mean force of one bond of the chain and its length derivative. For bond `bond_index` (0-based; bond 0 is attached to the anchored atom), W_i(l) = -(1/beta) ln G_i(l), where G_i(l) is the configurational weight of the chain with bond i held at length l and every other bond j integrated over its allowed interval [0, l_j_thr] with the full three-dimensional bond-vector measure and the bending coupling between successive bonds (source, Secs. S1.4 and S2). Return a (2, n_l) array: row 0 is W_i(l) - W_i(l_e), i.e. the profile with its additive constant fixed by its value at l = l_e (= 1 in reduced units), row 1 is dW_i/dl. The polar-angle grid with its quadrature weights, the bending kernel of the kernel step and the log intact weights of all bonds (one row per bond, thresholds already applied) are supplied; the nearest-neighbour orientational correlations of the whole chain must be resolved exactly on this grid, without sampling. At l = 0 the profile is +inf and the derivative -inf. Reduced units: lengths in units of the equilibrium bond length l_e, energies in units of the dissociation energy D_e, so beta_de = D_e/(k_B T), a_le = a l_e and f_red = f l_e / D_e; angles in radians.

```python
def constrained_pmf(theta: "np.ndarray", w_theta: "np.ndarray", kernel: "np.ndarray", log_i: "np.ndarray",
                            bond_index: int, l_values: "np.ndarray", f_red: float, beta_de: float,
                            a_le: float) -> "np.ndarray":
    """Evaluate, at the supplied lengths, the constrained potential of mean force of one bond of the chain and its length derivative. For bond `bond_index` (0-based; bond 0 is attached to the anchored atom), W_i(l) = -(1/beta) ln G_i(l), where G_i(l) is the configurational weight of the chain with bond i held at length l and every other bond j integrated over its allowed interval [0, l_j_thr] with the full three-dimensional bond-vector measure and the bending coupling between successive bonds (source, Secs. S1.4 and S2). Return a (2, n_l) array: row 0 is W_i(l) - W_i(l_e), i.e. the profile with its additive constant fixed by its value at l = l_e (= 1 in reduced units), row 1 is dW_i/dl. The polar-angle grid with its quadrature weights, the bending kernel of the kernel step and the log intact weights of all bonds (one row per bond, thresholds already applied) are supplied; the nearest-neighbour orientational correlations of the whole chain must be resolved exactly on this grid, without sampling. At l = 0 the profile is +inf and the derivative -inf. Reduced units: lengths in units of the equilibrium bond length l_e, energies in units of the dissociation energy D_e, so beta_de = D_e/(k_B T), a_le = a l_e and f_red = f l_e / D_e; angles in radians.

    Parameters
    ----------
    theta : np.ndarray
        Polar-angle grid strictly inside (0, pi), increasing, shape (n_theta,).
    w_theta : np.ndarray
        Positive quadrature weights of the theta grid, shape (n_theta,).
    kernel : np.ndarray
        Bending kernel of shape (n_theta, n_theta) from the kernel step.
    log_i : np.ndarray
        Log intact weights of all bonds at their thresholds, shape (n_bonds, n_theta).
    bond_index : int
        Index of the selected bond, 0 .. n_bonds - 1 (0 = bond attached to the anchor).
    l_values : np.ndarray
        Lengths at which to evaluate the profile, in units of l_e, non-negative, shape (n_l,).
    f_red : float
        Reduced force f l_e / D_e (positive).
    beta_de : float
        Reduced dissociation energy D_e / (k_B T).
    a_le : float
        Reduced Morse range parameter a l_e.

    Returns
    -------
    profile : np.ndarray
        Array of shape (2, n_l): relative profile and its derivative.

    Raises
    ------
    ValueError
        If the grid is invalid, the kernel is not a finite non-negative (n_theta, n_theta) array, log_i is not an (n_bonds, n_theta) array without NaN or +inf, bond_index is outside [0, n_bonds), l_values is not a one-dimensional finite non-negative array, f_red, beta_de or a_le is not finite positive, or the orientational weight underflows to zero.
    """
    return profile
```

### Step 6

pmf_stationary_points

Goal
----
Locate the bonded minimum and the barrier top of the constrained potential of mean force of bond `bond_index` (the profile of the previous step) and return them together with the activation barrier W_i(l_bar) - W_i(l_min) in D_e. Bracket sign changes of the derivative on n_scan equally spaced lengths in [l_lo, l_hi], take the first minimum and the first maximum beyond it, and refine each root to 1e-13 in l. Call the previous step's function for the profile and its derivative. Reduced units: lengths in units of the equilibrium bond length l_e, energies in units of the dissociation energy D_e, so beta_de = D_e/(k_B T), a_le = a l_e and f_red = f l_e / D_e; angles in radians.

```python
def pmf_stationary_points(theta: "np.ndarray", w_theta: "np.ndarray", kernel: "np.ndarray", log_i: "np.ndarray",
                                  bond_index: int, f_red: float, beta_de: float, a_le: float,
                                  l_lo: float = 0.8, l_hi: float = 6.0, n_scan: int = 2601) -> tuple:
    """Locate the bonded minimum and the barrier top of the constrained potential of mean force of bond `bond_index` (the profile of the previous step) and return them together with the activation barrier W_i(l_bar) - W_i(l_min) in D_e. Bracket sign changes of the derivative on n_scan equally spaced lengths in [l_lo, l_hi], take the first minimum and the first maximum beyond it, and refine each root to 1e-13 in l. Call the previous step's function for the profile and its derivative. Reduced units: lengths in units of the equilibrium bond length l_e, energies in units of the dissociation energy D_e, so beta_de = D_e/(k_B T), a_le = a l_e and f_red = f l_e / D_e; angles in radians.

    Parameters
    ----------
    theta : np.ndarray
        Polar-angle grid strictly inside (0, pi), increasing, shape (n_theta,).
    w_theta : np.ndarray
        Positive quadrature weights of the theta grid, shape (n_theta,).
    kernel : np.ndarray
        Bending kernel of shape (n_theta, n_theta).
    log_i : np.ndarray
        Log intact weights of all bonds at their thresholds, shape (n_bonds, n_theta).
    bond_index : int
        Index of the selected bond, 0 .. n_bonds - 1.
    f_red : float
        Reduced force f l_e / D_e (positive).
    beta_de : float
        Reduced dissociation energy D_e / (k_B T).
    a_le : float
        Reduced Morse range parameter a l_e.
    l_lo : float
        Lower end of the scan interval in l_e (default 0.8).
    l_hi : float
        Upper end of the scan interval in l_e (default 6.0).
    n_scan : int
        Number of equally spaced scan lengths (default 2601).

    Returns
    -------
    result : tuple
        (l_min, l_bar, barrier) as native floats.

    Raises
    ------
    ValueError
        If the profile step raises for the supplied data, l_lo or l_hi is not finite positive with l_hi > l_lo, n_scan is not an integer >= 3, or the profile has no bonded minimum or no barrier top on the scan interval (force at or above the critical force).
    """
    return result
```

### Step 7

self_consistent_thresholds

Goal
----
Determine the rupture thresholds of all bonds of an n_bonds chain self-consistently: starting from a common initial threshold l_init, form the log intact weights of all bonds at the current thresholds, locate the barrier top of every bond's constrained potential of mean force, set each threshold to its own barrier top, and repeat until the largest threshold change is below tol (source, Sec. S2.2, the self-consistent scheme; at most max_iter sweeps). Call the earlier step functions for the intact weights and the stationary points. Reduced units: lengths in units of the equilibrium bond length l_e, energies in units of the dissociation energy D_e, so beta_de = D_e/(k_B T), a_le = a l_e and f_red = f l_e / D_e; angles in radians.

```python
def self_consistent_thresholds(theta: "np.ndarray", w_theta: "np.ndarray", kernel: "np.ndarray", f_red: float,
                                       beta_de: float, a_le: float, n_bonds: int, l_init: float,
                                       tol: float = 1e-10, max_iter: int = 50, n_l: int = 4001) -> "np.ndarray":
    """Determine the rupture thresholds of all bonds of an n_bonds chain self-consistently: starting from a common initial threshold l_init, form the log intact weights of all bonds at the current thresholds, locate the barrier top of every bond's constrained potential of mean force, set each threshold to its own barrier top, and repeat until the largest threshold change is below tol (source, Sec. S2.2, the self-consistent scheme; at most max_iter sweeps). Call the earlier step functions for the intact weights and the stationary points. Reduced units: lengths in units of the equilibrium bond length l_e, energies in units of the dissociation energy D_e, so beta_de = D_e/(k_B T), a_le = a l_e and f_red = f l_e / D_e; angles in radians.

    Parameters
    ----------
    theta : np.ndarray
        Polar-angle grid strictly inside (0, pi), increasing, shape (n_theta,).
    w_theta : np.ndarray
        Positive quadrature weights of the theta grid, shape (n_theta,).
    kernel : np.ndarray
        Bending kernel of shape (n_theta, n_theta).
    f_red : float
        Reduced force f l_e / D_e (positive).
    beta_de : float
        Reduced dissociation energy D_e / (k_B T).
    a_le : float
        Reduced Morse range parameter a l_e.
    n_bonds : int
        Number of bonds (positive).
    l_init : float
        Common initial threshold in units of l_e (positive).
    tol : float
        Convergence tolerance on the largest threshold change (default 1e-10).
    max_iter : int
        Maximum number of sweeps (default 50).
    n_l : int
        Trapezoid points of the intact-weight integrals (default 4001).

    Returns
    -------
    thresholds : np.ndarray
        Converged thresholds of shape (n_bonds,).

    Raises
    ------
    ValueError
        If the grid is invalid, n_bonds or max_iter is not a positive integer, l_init or tol is not finite positive, any earlier step raises for the supplied data, or the iteration does not converge within max_iter sweeps.
    """
    return thresholds
```

### Step 8

bond_tst_rate

Goal
----
Evaluate the bond-resolved transition-state-theory rate of one bond in 1/s from its tabulated constrained potential of mean force on a length grid that runs from 0 to the bond's barrier top (the last grid point) and from the bond's kinetic prefactor: the source's full flux-over-population expression (its Eq. S50), the prefactor times the Boltzmann factor of the profile at the barrier top divided by the integral of the Boltzmann factor of the profile over the bonded interval [0, l_bar], using the trapezoid rule on the supplied grid and working in the log domain. The additive constant of the profile is irrelevant; the value at l = 0 may be +inf (zero weight). Reduced units: lengths in units of the equilibrium bond length l_e, energies in units of the dissociation energy D_e, so beta_de = D_e/(k_B T), a_le = a l_e and f_red = f l_e / D_e; angles in radians.

```python
def bond_tst_rate(l_values: "np.ndarray", pmf_values: "np.ndarray", beta_de: float, prefactor: float) -> float:
    """Evaluate the bond-resolved transition-state-theory rate of one bond in 1/s from its tabulated constrained potential of mean force on a length grid that runs from 0 to the bond's barrier top (the last grid point) and from the bond's kinetic prefactor: the source's full flux-over-population expression (its Eq. S50), the prefactor times the Boltzmann factor of the profile at the barrier top divided by the integral of the Boltzmann factor of the profile over the bonded interval [0, l_bar], using the trapezoid rule on the supplied grid and working in the log domain. The additive constant of the profile is irrelevant; the value at l = 0 may be +inf (zero weight). Reduced units: lengths in units of the equilibrium bond length l_e, energies in units of the dissociation energy D_e, so beta_de = D_e/(k_B T), a_le = a l_e and f_red = f l_e / D_e; angles in radians.

    Parameters
    ----------
    l_values : np.ndarray
        Strictly increasing lengths in units of l_e from 0 to the barrier top, shape (n_l,), n_l >= 3.
    pmf_values : np.ndarray
        Constrained potential of mean force at l_values in D_e (any additive constant; +inf allowed at l = 0).
    beta_de : float
        Reduced dissociation energy D_e / (k_B T).
    prefactor : float
        Kinetic prefactor of the bond in 1/s.

    Returns
    -------
    rate : float
        Bond-resolved TST rate in 1/s.

    Raises
    ------
    ValueError
        If l_values is not a strictly increasing one-dimensional array starting at 0 with at least 3 points, pmf_values does not match it or contains NaN, -inf or a non-finite value beyond the first point, or beta_de or prefactor is not finite positive.
    """
    return rate
```

### Step 9

mean_scission_time

Goal
----
Orchestrate the whole pipeline in SI units and return the mean time to the first scission, 1 / k_chain in hours, of an n_bonds chain of atoms of mass `mass`, anchored at atom 0 and held at the end-to-end distance stretch_ratio * R_0 (R_0 = N l_e times the sine of half the equilibrium valence angle, the length of the undeformed zigzag). Every bond stores the Morse stretching energy of the earlier steps, and every interior atom stores a harmonic bending energy k_phi (theta_v - theta_v_e)^2 / 2 in its valence angle theta_v, the angle between the two bonds meeting at that atom, with equilibrium valence angle valence_angle_deg. Convert to reduced units (beta_de = d_e / (k_b T), a_le = a l_e, beta_kphi = k_phi / (k_b T)) and to the angle convention of the earlier steps, obtain the constant chain force from the dFRC step (its optimum converged to 1e-12 in both variables, as that step requires), build a Gauss-Legendre grid of n_theta nodes on [0, pi], take the collinear reference barrier top as the initial threshold, build the bending kernel with n_omega azimuths, converge the self-consistent thresholds, form the intact weights at the converged thresholds, locate every bond's stationary points, tabulate every bond's profile on n_well equally spaced lengths from 0 to its barrier top, evaluate every bond-resolved rate with its kinetic prefactor (source, Sec. S1.3): the one-sided thermal average of the velocity of the bond-length coordinate of that bond, (2 pi m_l / (k_b T))^(-1/2) in m/s with m_l the effective mass of that coordinate, divided by l_e to match the reduced length unit; sum the bond-resolved rates and return the reciprocal of the chain-scission rate in hours. Call the earlier step functions rather than reimplementing them.

```python
def mean_scission_time(n_bonds: int, stretch_ratio: float, d_e: float, a: float, l_e: float, k_phi: float,
                               valence_angle_deg: float, temperature: float, mass: float,
                               k_b: float = 1.380649e-23, n_theta: int = 400, n_omega: int = 512,
                               n_well: int = 20001) -> float:
    """Orchestrate the whole pipeline in SI units and return the mean time to the first scission, 1 / k_chain in hours, of an n_bonds chain of atoms of mass `mass`, anchored at atom 0 and held at the end-to-end distance stretch_ratio * R_0 (R_0 = N l_e times the sine of half the equilibrium valence angle, the length of the undeformed zigzag). Every bond stores the Morse stretching energy of the earlier steps, and every interior atom stores a harmonic bending energy k_phi (theta_v - theta_v_e)^2 / 2 in its valence angle theta_v, the angle between the two bonds meeting at that atom, with equilibrium valence angle valence_angle_deg. Convert to reduced units (beta_de = d_e / (k_b T), a_le = a l_e, beta_kphi = k_phi / (k_b T)) and to the angle convention of the earlier steps, obtain the constant chain force from the dFRC step (its optimum converged to 1e-12 in both variables, as that step requires), build a Gauss-Legendre grid of n_theta nodes on [0, pi], take the collinear reference barrier top as the initial threshold, build the bending kernel with n_omega azimuths, converge the self-consistent thresholds, form the intact weights at the converged thresholds, locate every bond's stationary points, tabulate every bond's profile on n_well equally spaced lengths from 0 to its barrier top, evaluate every bond-resolved rate with its kinetic prefactor (source, Sec. S1.3): the one-sided thermal average of the velocity of the bond-length coordinate of that bond, (2 pi m_l / (k_b T))^(-1/2) in m/s with m_l the effective mass of that coordinate, divided by l_e to match the reduced length unit; sum the bond-resolved rates and return the reciprocal of the chain-scission rate in hours. Call the earlier step functions rather than reimplementing them.

    Parameters
    ----------
    n_bonds : int
        Number of bonds N (>= 2); atoms 0..N with atom 0 anchored.
    stretch_ratio : float
        Prescribed end-to-end distance divided by the undeformed zigzag length R_0 (positive).
    d_e : float
        Morse dissociation energy in J.
    a : float
        Morse range parameter in 1/m.
    l_e : float
        Equilibrium bond length in m.
    k_phi : float
        Bending stiffness in J/rad^2 (non-negative).
    valence_angle_deg : float
        Equilibrium valence angle at each interior atom in degrees, in (0, 180).
    temperature : float
        Temperature in K.
    mass : float
        Atomic mass in kg (all atoms).
    k_b : float
        Boltzmann constant in J/K (default 1.380649e-23).
    n_theta : int
        Gauss-Legendre nodes on [0, pi] (default 400).
    n_omega : int
        Midpoint azimuthal nodes of the kernel (default 512).
    n_well : int
        Equally spaced lengths from 0 to the barrier top for the well integral (default 20001).

    Returns
    -------
    tau_chain : float
        Mean time to first scission in hours.

    Raises
    ------
    ValueError
        If n_bonds, n_theta or n_well is not a valid integer, stretch_ratio, d_e, a, l_e, temperature, mass or k_b is not finite positive, k_phi is negative, valence_angle_deg is outside (0, 180), or any earlier step raises (for example a stretch whose chain force lies at or above the critical force).
    """
    return tau_chain
```
