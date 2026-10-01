# Physics-Condensed_Matter_Physics-21

## Background

Matter compressed far beyond the density of an atomic nucleus is one of the least directly accessible states in physics. It cannot be made in bulk in a laboratory, it cannot be probed by scattering an external beam through it, and the interactions that govern it are strong enough that the usual expansion in a small coupling fails over exactly the range of densities where the interesting behaviour occurs.

What makes the problem tractable at all is that the two ends are under control. At densities a little above nuclear saturation, systematic expansions of the nuclear interaction built on the symmetries of the underlying theory give a controlled description with quantifiable uncertainty. At asymptotically high densities the coupling weakens and a perturbative treatment becomes reliable again. The difficulty lies in the intervening region, spanning more than an order of magnitude in density, where neither description applies and where any change of degrees of freedom would occur.

The relation between pressure, density and energy density in this regime is not arbitrary. Thermodynamic identities tie the variables together, mechanical stability forbids certain behaviours outright, and the requirement that sound propagate slower than light imposes a bound on how rapidly the pressure may rise. These conditions are strong enough that fixing the two ends restricts the interpolation between them far more than intuition suggests, and they couple distant densities: behaviour at one density constrains what remains possible at another.

This has made the field a natural setting for methods that place probability measures over whole function spaces rather than over the parameters of an assumed form. The aim is to express what is genuinely known without smuggling in the shape of an answer, so that the data are allowed to speak. Constructing such a measure is delicate, because the physical requirements are global constraints on the function rather than local conditions that can be checked pointwise, and a construction that violates them produces samples that are not merely improbable but unphysical.

## Problem

Cold, charge-neutral, beta-equilibrated bulk matter at densities well above nuclear saturation is constrained from two directions: a low-density description whose validity ends a little above saturation, and a high-density description valid only once asymptotic freedom sets in. Between them the equation of state is not free, because any admissible interpolation must be mechanically stable, must not propagate sound faster than light, and must reproduce the pressure difference between the two endpoints. Those three requirements alone confine every intermediate thermodynamic state to a bounded region of the space spanned by the chemical potential $$\mu$$, the number density $$n$$ and the pressure $$p$$, and they induce long-range correlations: stiffness at intermediate densities must be paid for by softening later.

A bridge between the endpoints is built here by recursive subdivision rather than by assuming a functional form. A single intermediate state is placed inside the allowed region; the interval then splits in two, and the same rule is applied to each half using its own endpoints, so that structure develops on every scale resolved. Placing a state requires a probability measure on the allowed region, and a uniform measure with the Euclidean metric in $$(\mu,n,p)$$ is adopted. Because a slice of the region at fixed chemical potential is a triangle, the state is placed by drawing the chemical potential from its induced marginal, then the density from the distribution conditional on it, then the pressure from the distribution conditional on both.

A bridge built this way carries structure down to the finest scale resolved, so a correlation length is imposed on it by evolving the chemical potential as a function of density under a conservative diffusion in an unphysical flow time, holding the chemical potential fixed at both ends. The diffusion coefficient is chosen so that the correlation length imprinted on the squared sound speed is a fixed fraction of the density at every density. Inside the interval this smoothing conserves the energy-density difference exactly, so thermodynamic consistency is preserved there, but at the two ends energy density flows through the edges, and the accumulated flow measures how far the smoothed bridge has drifted from exact consistency.

Construct the bridge specified below and determine the net energy density that crosses the two boundaries over the whole smoothing, that is the resulting change in the energy-density difference between the endpoints. Use the following finite-step benchmark; the result is a property of this prescribed discretisation rather than a continuum limit.

| Quantity | Benchmark convention |
| --- | --- |
| Endpoints | $$\beta_L=(1.00\ \mathrm{GeV},\ 0.32\ \mathrm{fm^{-3}},\ 0.018\ \mathrm{GeV\,fm^{-3}})$$ and $$\beta_H=(2.60\ \mathrm{GeV},\ 4.80\ \mathrm{fm^{-3}},\ 3.500\ \mathrm{GeV\,fm^{-3}})$$ |
| Allowed region | At each $$\mu$$ the density lies in $$[n_{\min},n_{\max}]$$ and, for each admissible $$(\mu,n)$$, the pressure lies in $$[p_{\min},p_{\max}]$$, with $$\Delta p=p_H-p_L$$ and crossover $$\mu_c=\sqrt{\mu_L\mu_H(\mu_Hn_H-\mu_Ln_L-2\Delta p)/(\mu_Ln_H-\mu_Hn_L)}$$: $$n_{\min}=n_L\mu/\mu_L$$ for $$\mu\le\mu_c$$ and $$[\mu^3n_H-\mu\mu_H(\mu_Hn_H-2\Delta p)]/[(\mu^2-\mu_L^2)\mu_H]$$ above; $$n_{\max}=[\mu^3n_L-\mu_L\mu(\mu_Ln_L+2\Delta p)]/[(\mu^2-\mu_H^2)\mu_L]$$ for $$\mu<\mu_c$$ and $$n_H\mu/\mu_H$$ above; $$p_{\min}=p_L+\frac{\mu^2-\mu_L^2}{2\mu}n_{\min}$$; and, with $$n_c=n_{\max}(\mu_L)\mu/\mu_L$$, $$p_{\max}=p_L+\frac{\mu^2-\mu_L^2}{2\mu}n$$ for $$n\le n_c$$ and $$p_H-\frac{\mu_H^2-\mu^2}{2\mu}n$$ above |
| Placement | Prescribed quantiles $$0.5$$ in each of the chemical potential, the density and the pressure, applied in that order at every insertion and using the marginals induced by the uniform measure |
| Refinement | Six levels; each interval is split at its own prescribed quantile point and both halves are refined to the remaining depth |
| Initial profile | The refined states interpolated linearly in $$\mu$$ against $$n$$ onto $$201$$ uniformly spaced densities spanning $$[n_L,n_H]$$ inclusively |
| Smoothing | Conservative explicit diffusion of $$\mu(n)$$ in flow time from $$0$$ to $$1$$ in $$4000$$ uniform steps, with the inter-point flux equal to the diffusion coefficient averaged over the two neighbouring points times their chemical-potential difference divided by the grid spacing, the end values held fixed, and a correlation length equal to $$0.20$$ times the density |
| Sound-speed diagnostics | Evaluate the density derivative of the smoothed chemical potential by second-order central differences on the uniform grid; take the smallest and largest squared sound speed over interior grid points only |
| Excluded | No stellar-structure calculation of any kind enters this task |

Include in the reasoning the crossover chemical potential, the chemical potential at the median of its induced marginal, the full first inserted state, the diffusion coefficient at $$n=1\ \mathrm{fm^{-3}}$$ together with the value of the quantity that must not be positive for the smoothing to preserve causality, the smallest and largest squared sound speed of the smoothed bridge, the number of states in the refined bridge counting both endpoints, and the accumulated boundary energy flux obtained when the refinement is omitted entirely and the straight chord between the endpoints is smoothed instead.

Report the accumulated boundary energy flux in $$\mathrm{GeV\,fm^{-3}}$$ to an absolute accuracy of $$10^{-9}$$, and give each diagnostic listed above to at least six significant figures where it is non-zero.

Output Format Requirements:
Emit <final_answer> immediately, then <reasoning>. Do not write a long derivation before the tags.
You must emit exactly one finite decimal inside <final_answer>...</final_answer>, even if the value is approximate or you are unsure.
Rules:
- The tags are required. Do not omit them or leave them empty.
- The tagged value must be a finite decimal (examples: 0.4847, 0.22, 1.05). Not NaN, not Inf, not a fraction string, not a vector, not prose.
- Put only that one number between the tags. No units, no words, no extra lines. Keep <reasoning> short (a few hundred words). Show only the few scalars that determine the final number.
- Do not paste the input matrices, full coefficient vectors, per-iteration paths, or per-fold candidate tables.

## Output format

```
## Output format

Scientific reasoning wrapped in <reasoning>...</reasoning> tags. Include enough intermediate calculations to justify the deterministic computation, without turning the response into a general pipeline summary. A single final numeric answer wrapped in <final_answer>...</final_answer> tags.
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

allowed_density_bounds

Goal
----
Evaluate the smallest and largest number density a stable, causal and thermodynamically consistent equation of state may take at a given chemical potential.

```python
def allowed_density_bounds(chemical_potential: float, beta_low: "np.ndarray",
                           beta_high: "np.ndarray") -> "np.ndarray":
    '''Compute the density envelopes of the allowed region at one chemical potential.

    Parameters
    ----------
    chemical_potential : float
        Finite chemical potential in GeV, with beta_low[0] <= chemical_potential
        <= beta_high[0].
    beta_low, beta_high : np.ndarray
        Positive finite shape (3,) endpoints (mu, n, p) in GeV, fm^-3 and GeV fm^-3, with
        beta_low[0] < beta_high[0], beta_low[1] < beta_high[1] and
        beta_low[2] < beta_high[2]. They must enclose a nonempty allowed region:
        muL*nH - muH*nL > 0 and the finite real crossover mu_c defined below
        must lie strictly between muL and muH.

    Returns
    -------
    bounds : np.ndarray
        Finite shape (2,) in fm^-3, the minimum and maximum allowed number
        density in that order. Writing dp for the pressure difference and
        mu_c = sqrt(muL*muH*(muH*nH - muL*nL - 2*dp)/(muL*nH - muH*nL)) for the
        crossover chemical potential, the minimum is nL*mu/muL when
        mu <= mu_c and (mu**3*nH - mu*muH*(muH*nH - 2*dp))/((mu**2 - muL**2)*muH)
        otherwise, while the maximum is
        (mu**3*nL - muL*mu*(muL*nL + 2*dp))/((mu**2 - muH**2)*muL) when mu < mu_c
        and nH*mu/muH otherwise.

    Raises
    ------
    ValueError
        If the inputs are not finite real values of the stated shapes, if the
        endpoint positivity or ordering fails, if the endpoints do not enclose
        the nonempty allowed region specified above, if the chemical potential
        lies outside the endpoint interval, or if the evaluated bounds are not
        finite.
    '''
    return bounds
```

### Step 2

allowed_pressure_bounds

Goal
----
Evaluate the smallest and largest pressure a stable, causal and thermodynamically consistent equation of state may take at a given chemical potential and number density.

```python
def allowed_pressure_bounds(chemical_potential: float, density: float,
                            beta_low: "np.ndarray", beta_high: "np.ndarray") -> "np.ndarray":
    '''Compute the pressure edges of the allowed slice at one chemical potential and density.

    Parameters
    ----------
    chemical_potential : float
        Finite chemical potential in GeV inside the endpoint interval.
    density : float
        Finite number density in fm^-3, between the density bounds returned by
        the preceding step at this chemical potential.
    beta_low, beta_high : np.ndarray
        Finite shape (3,) endpoints (mu, n, p) in GeV, fm^-3 and GeV fm^-3.

    Returns
    -------
    bounds : np.ndarray
        Finite shape (2,) in GeV fm^-3, the minimum and maximum allowed pressure
        in that order. The minimum is pL + (mu**2 - muL**2)/(2*mu) times the
        minimum allowed density at this chemical potential. Writing
        n_c = nmax(muL)*mu/muL for the crossover density, the maximum is
        pL + (mu**2 - muL**2)/(2*mu)*density when density <= n_c and
        pH - (muH**2 - mu**2)/(2*mu)*density otherwise.

    Raises
    ------
    ValueError
        If the inputs are not finite real values of the stated shapes, if a
        preceding step rejects the endpoints or chemical potential, if the
        density lies outside its allowed bounds, or if the evaluated pressure
        bounds are not finite.
    '''
    return bounds
```

### Step 3

chemical_potential_quantile

Goal
----
Invert the marginal chemical-potential distribution of a uniform measure on the allowed region, returning the chemical potential at a requested quantile.

```python
def chemical_potential_quantile(quantile: float, beta_low: "np.ndarray",
                                beta_high: "np.ndarray") -> float:
    '''Return the chemical potential at a requested quantile of the induced marginal.

    Parameters
    ----------
    quantile : float
        Finite quantile in [0, 1].
    beta_low, beta_high : np.ndarray
        Finite shape (3,) endpoints (mu, n, p) in GeV, fm^-3 and GeV fm^-3.

    Returns
    -------
    value : float
        Chemical potential in GeV as a native Python float, the point at which
        the cumulative distribution of the marginal density reaches the
        requested quantile. The marginal density is proportional to
        mu*(mu**2 - muL**2)/(muH**2 - mu**2) below the crossover chemical
        potential and to mu*(muH**2 - mu**2)/(mu**2 - muL**2) above it, each
        branch scaled so the density is continuous at the crossover. A quantile
        of 0 returns muL and a quantile of 1 returns muH. The value has an
        absolute accuracy of 1e-10 GeV.

    Raises
    ------
    ValueError
        If the quantile is not a finite real scalar in [0, 1], if a preceding
        step rejects the endpoints, or if the inversion does not converge to a
        finite value.
    '''
    return value
```

### Step 4

quantile_point_in_volume

Goal
----
Select a single thermodynamic state inside the allowed region at prescribed quantiles of the uniform measure, drawing the three variables in sequence.

```python
def quantile_point_in_volume(quantiles: "np.ndarray", beta_low: "np.ndarray",
                             beta_high: "np.ndarray") -> "np.ndarray":
    '''Place one state in the allowed region at prescribed quantiles.

    Parameters
    ----------
    quantiles : np.ndarray
        Finite shape (3,) with every entry in [0, 1], the quantiles used for the
        chemical potential, the density and the pressure in that order.
    beta_low, beta_high : np.ndarray
        Finite shape (3,) endpoints (mu, n, p) in GeV, fm^-3 and GeV fm^-3.

    Returns
    -------
    point : np.ndarray
        Finite shape (3,), the state (mu, n, p) in GeV, fm^-3 and GeV fm^-3.
        The chemical potential is taken at its quantile of the induced marginal.
        The density is taken at its quantile of the conditional distribution
        whose density is proportional to the width of the pressure interval at
        that density. The pressure is taken at its quantile of the uniform
        distribution between the pressure edges, so it equals
        pmin + quantiles[2]*(pmax - pmin). Each conditional inversion has an
        absolute accuracy of 1e-10 in its own units. At an endpoint chemical
        potential, where the slice has zero area, use the midpoint of the
        density bounds and the common pressure edge.

    Raises
    ------
    ValueError
        If the quantiles are not a finite real shape (3,) array inside [0, 1],
        if a preceding step rejects its inputs, or if the selected state is not
        finite.
    '''
    return point
```

### Step 5

self_similar_refine

Goal
----
Build a self-similar bridge between two endpoints by recursively inserting states into the allowed region.

```python
def self_similar_refine(beta_low: "np.ndarray", beta_high: "np.ndarray", depth: int,
                        quantiles: "np.ndarray") -> "np.ndarray":
    '''Construct the ordered sequence of states forming a self-similar bridge.

    Parameters
    ----------
    beta_low, beta_high : np.ndarray
        Finite shape (3,) endpoints (mu, n, p) in GeV, fm^-3 and GeV fm^-3.
    depth : int
        Integer number of refinement levels, at least 0 and at most 12. A depth
        of zero inserts no intermediate state.
    quantiles : np.ndarray
        Finite shape (3,) with every entry in [0, 1], applied unchanged at every
        insertion.

    Returns
    -------
    states : np.ndarray
        Finite shape (K, 3) with K = 2**depth + 1, the states (mu, n, p) ordered
        by increasing density, beginning with beta_low and ending with
        beta_high. At each level the interval is split by inserting the state at
        the prescribed quantiles of its own allowed region, and the two
        resulting subintervals are refined in the same way to the remaining
        depth, the lower one first.

    Raises
    ------
    ValueError
        If depth is not an integer in [0, 12], if a preceding step rejects its
        inputs, or if the constructed sequence is not finite or not strictly
        increasing in density.
    '''
    return states
```

### Step 6

diffusion_profile

Goal
----
Evaluate the density-dependent diffusion coefficient that imposes a fixed fractional correlation length, together with the margin by which it satisfies the causality-preserving condition.

```python
def diffusion_profile(density_grid: "np.ndarray", correlation_fraction: float) -> "np.ndarray":
    '''Evaluate the diffusion coefficient and its causality margin on a density grid.

    Parameters
    ----------
    density_grid : np.ndarray
        Finite strictly increasing shape (N,) densities in fm^-3, N >= 3, all
        strictly positive.
    correlation_fraction : float
        Finite positive fractional correlation length, the ratio of the
        correlation length to the density.

    Returns
    -------
    profile : np.ndarray
        Finite shape (3, N). Row 0 is the diffusion coefficient
        density_grid**2 * correlation_fraction**2 / 4, which makes the
        correlation length after unit flow time equal to
        correlation_fraction times the density. Row 1 is its first derivative
        with respect to density. Row 2 is the causality margin, the second
        derivative less the first derivative divided by the density, which must
        not be positive anywhere for the smoothing to preserve causality.

    Raises
    ------
    ValueError
        If the grid is not a finite strictly increasing array of at least three
        positive densities, if the correlation fraction is not a finite
        positive real scalar, or if the evaluated profile is not finite.
    '''
    return profile
```

### Step 7

diffuse_chemical_potential

Goal
----
Smooth a bridge by evolving the chemical potential under a conservative diffusion in flow time, and report the energy density transported across the domain boundaries.

```python
def diffuse_chemical_potential(density_grid: "np.ndarray", chemical_potential: "np.ndarray",
                               correlation_fraction: float, n_tau_steps: int) -> "np.ndarray":
    '''Diffuse the chemical potential and accumulate the boundary energy flux.

    Parameters
    ----------
    density_grid : np.ndarray
        Finite strictly increasing shape (N,) densities in fm^-3, N >= 3, all
        strictly positive and uniformly spaced.
    chemical_potential : np.ndarray
        Finite shape (N,) chemical potentials in GeV on that grid.
    correlation_fraction : float
        Finite positive fractional correlation length.
    n_tau_steps : int
        Integer number of uniform flow-time steps, at least 1, covering the
        flow time from zero to one.

    Returns
    -------
    result : np.ndarray
        Finite shape (N + 1,). Element 0 is the accumulated boundary energy
        flux in GeV fm^-3, and elements 1 to N are the chemical potential in GeV
        after unit flow time.

        The evolution uses the conservative explicit update in which the flux
        between neighbouring grid points is the diffusion coefficient averaged
        over the two points, times their chemical-potential difference, divided
        by the grid spacing. Interior values advance by the flow-time step times
        the difference of the neighbouring fluxes divided by the grid spacing,
        and the first and last values are held fixed. The accumulated boundary
        flux adds, at every step, the flow-time step times the flux at the last
        interval less the flux at the first interval.

    Raises
    ------
    ValueError
        If the arrays are not finite real values of matching shape, if the grid
        is not uniformly spaced, if a preceding step rejects its inputs, if
        n_tau_steps is not an integer of at least 1, or if the evolved result is
        not finite.
    '''
    return result
```

### Step 8

sound_speed_squared

Goal
----
Evaluate the squared speed of sound of a zero-temperature equation of state given as chemical potential against number density.

```python
def sound_speed_squared(density_grid: "np.ndarray",
                        chemical_potential: "np.ndarray") -> "np.ndarray":
    '''Evaluate the squared speed of sound on a density grid.

    Parameters
    ----------
    density_grid : np.ndarray
        Finite strictly increasing shape (N,) densities in fm^-3, N >= 3, all
        strictly positive.
    chemical_potential : np.ndarray
        Finite strictly positive shape (N,) chemical potentials in GeV on that
        grid.

    Returns
    -------
    speeds : np.ndarray
        Finite dimensionless shape (N,), the squared speed of sound, evaluated
        as the density divided by the chemical potential, times the derivative
        of the chemical potential with respect to density. The derivative uses
        second-order central differences in the interior and second-order
        one-sided differences at the two ends.

    Raises
    ------
    ValueError
        If the grid is not a finite strictly increasing array of at least three
        positive densities, if the chemical potential is not a finite positive
        array of matching shape, or if the evaluated speeds are not finite.
    '''
    return speeds
```

### Step 9

bridge_boundary_flux

Goal
----
Assemble the full construction for one dense-matter bridge and report the energy density transported across its boundaries during smoothing.

```python
def bridge_boundary_flux(beta_low: "np.ndarray", beta_high: "np.ndarray", depth: int,
                         quantiles: "np.ndarray", correlation_fraction: float,
                         n_grid_points: int, n_tau_steps: int) -> "np.ndarray":
    '''Run the bridge construction and report its boundary flux and peak sound speed.

    Parameters
    ----------
    beta_low, beta_high : np.ndarray
        Finite shape (3,) endpoints (mu, n, p) in GeV, fm^-3 and GeV fm^-3.
    depth : int
        Integer refinement depth in [0, 12].
    quantiles : np.ndarray
        Finite shape (3,) with every entry in [0, 1].
    correlation_fraction : float
        Finite positive fractional correlation length.
    n_grid_points : int
        Integer number of uniformly spaced density grid points, at least 3,
        spanning the two endpoint densities inclusively.
    n_tau_steps : int
        Integer number of uniform flow-time steps, at least 1.

    Returns
    -------
    summary : np.ndarray
        Finite shape (2,). Element 0 is the accumulated boundary energy flux in
        GeV fm^-3, and element 1 is the largest squared speed of sound of the
        smoothed bridge, taken over the interior grid points only.

        The refined states are interpolated linearly in chemical potential
        against density onto the uniform grid to form the initial condition,
        which is then diffused for unit flow time.

    Raises
    ------
    ValueError
        If n_grid_points is not an integer of at least 3, or if a preceding
        step rejects its inputs, or if the summary is not finite.
    '''
    return summary
```
