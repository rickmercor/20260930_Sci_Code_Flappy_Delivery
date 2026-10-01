# Certified coupled Hall-MHD source step

## Background

The benchmark uses permeability `mu=1`, mass-norm constant `c_m=0.92`, artificial-resistivity constants `c_low=0.25` and `c_res=1`, Cartesian component order `[x,y,z]`, and the paper's `exp(i(kx-omega t))` whistler convention.

Candidate rows `[h,tau]`, in fixed order, are `[[0.15,0.002],[0.12,0.0012],[0.09,0.0007],[0.075,0.0004],[0.06,0.00022],[0.052,0.00016],[0.045,0.00012]]`. The three scenarios have `rho=[1,0.65,0.35]`, background field `[0.4,0.55,0.7]`, ion skin depth `[0.5,0.8,1]`, physical resistivity `[0.004,0.006,0.008]`, and wavenumber `[3,4,5]`. Four ordered nodes use lumped masses `[0.22,0.28,0.24,0.26]` and local scales `[0.9,1,1.1,1.2]`. Arrays below use `[scenario,node,component]` order for vector fields and `[scenario,node]` order for internal energy.

The ion velocity is `[[[0.10,0,0],[0.05,0.02,0],[-0.03,0.04,0],[0.08,-0.02,0]],[[0.15,-0.05,0],[0.08,0.06,0.02],[-0.10,0.03,0],[0.12,0.01,-0.03]],[[0.20,-0.10,0.02],[0.12,0.08,-0.04],[-0.15,0.05,0.06],[0.18,-0.02,-0.05]]]`. The baseline weak current loads are defined by the supplied curl operator below, using `current_load_s,n=mass_n*(C@H_s.ravel())_n`.


The magnetic nodes are `[[[0.42,0.02,0],[0.38,-0.01,0.01],[0.45,0,-0.02],[0.35,-0.01,0.01]],[[0.58,0.02,0.01],[0.52,-0.02,0.02],[0.61,0.01,-0.03],[0.49,-0.01,0]],[[0.74,0.03,0.01],[0.66,-0.03,0.03],[0.77,0.01,-0.04],[0.63,-0.01,0]]]`. The residual load is `[[[0.098782890826296,0,0],[0,0.188585518850202,0],[0,0,0.269407884071717],[-0.103187576771625,0.103187576771625,0]],[[0.187334370236751,0,0],[0,0.357638343179252,0],[0,0,0.510911918827502],[-0.195687527962311,0.195687527962311,0]],[[0.375200051172705,0,0],[0,0.668538272998638,0],[0,0,0.859549208141105],[-0.376252855234349,0.376252855234349,0]]]`. The internal energy is `[[0.60,0.62,0.58,0.61],[0.50,0.52,0.48,0.51],[0.40,0.42,0.38,0.41]]`.

The source discretization is a four-point, mass-quadrature realization of the paper's source variational system, with zero boundary flux; it is not a full mesh-convergence reproduction. In node-major `[x,y,z]` order define `M=diag(repeat(mass,3))`, `D_x=3*[[-1,1,0,0],[-1,1,0,0],[0,0,-1,1],[0,0,-1,1]]` and `D_y=3*[[-1,0,1,0],[0,-1,0,1],[-1,0,1,0],[0,-1,0,1]]`. The discrete curl is specified by `(CH)_x=D_y H_z`, `(CH)_y=-D_x H_z` and `(CH)_z=D_x H_y-D_y H_x`. Thus the weak magnetic operator uses `C.T M`; both the magnetic inner product and nodal quadrature use M. The fixed source operator describes the supplied patch, while candidate h controls the artificial-resistivity length and the declared work estimate.

The coupled uncertainty lattice has modulus 37, generators `[1,7,11,16,20,26,31]`, and relative half-widths `[0.04,0.03,0.05,0.06,0.02,0.05,0.08]` for density, background field, ion skin depth, physical resistivity, wavenumber, current load, and residual load. Lattice member `q` uses coordinate `2*((q*a_j) mod 37)/36-1`, and each member is crossed with the three scenarios, giving 111 stress states per candidate.

For each perturbed state, the local nodal length is `h*node_scale_n`; rescale every magnetic node by `h0_q/h0_s` and set the domain measure to the sum of the lumped masses. Use the arithmetic mean over nodes of the final blended-resistivity column as the modal resistivity in the whistler calculation. The minimum blended resistivity supplies the coercivity certificates; endpoint bounds are defined below.

For lattice member q use `C_q=((1+0.05*xi_5)/(1+0.03*xi_1))*C`, so perturbed magnetic samples and weak current loads describe the same field. The semidiscrete source equations are `rho_q*dv/dt=mu*(J cross H)` and `mu*M*dH/dt=-C_q.T*M*[mu*(H cross v)+r*J+(mu*d_i,q/rho_q)*(J cross H)]`, with `J=C_q H`; nodal products are flattened in node-major order. Here r is frozen at the old state's blended nodal resistivity. Advance this coupled system by one implicit-midpoint step of duration tau, taking the root continuous from the old fields at tau=0; the prescribed instances lie in its unique-root regime. The infinity norms of the momentum residual divided by rho_q and of the magnetic residual divided componentwise by `mu*repeat(mass,3)` must each be at most `1e-11*(1+max(abs(v_old),abs(H_old)))`.

Initial momentum is `rho_q*v_old` and initial mechanical-energy density is `internal+rho_q*|v_old|^2/2`. Apply the paper's nodal mechanical-energy update using the solved momentum and midpoint current, with nodal Joule power density `Q=r*|J_mid|^2` and integrated heat `Delta_J=tau*sum_n(mass_n*Q_n)`. The heat fraction is `Delta_J/sum_n(mass_n*internal_n)`. Magnetic energy means `mu*sum_n(mass_n*|H_new,n|^2)/2`, evaluated from the solved field; the total-energy defect is `sum_n(mass_n*(E_new,n-E_old,n))+mu*sum_n(mass_n*(|H_new,n|^2-|H_old,n|^2))/2`. For the coercivity bounds, use the larger old/new maximum current norm and the larger old/new maximum electron speed, with the same frozen minimum resistivity. The critical heat state is indexed by q in 0,...,36 and scenario in 1,2,3; exact ties retain this order.

Wave diagnostics compare a unit-amplitude right-polarized mode, using the first-order resistive dispersion approximation, with its Crank--Nicolson propagation after the smallest whole number of steps of size `tau` spanning one real-frequency period: `N=ceil(2*pi/(Re(omega)*tau))`, with both modes evaluated at `t=N*tau`. The amplitude diagnostic is the absolute relative modulus error, the phase diagnostic is the absolute principal phase of the numerical-to-exact ratio in radians, and log damping is the absolute difference of their natural log moduli. Each candidate's robust score is the maximum of its four worst-state diagnostics divided by their respective limits. Coercivity triples are normalized by `[c_m^2*rho_q, mu, tau*r_min/2]`; candidate margins are the minima over all three coefficients and all stress states.

The four score limits, in order, are amplitude error `1e-5`, phase error `8e-5`, log-damping error `1e-5`, and Joule heat fraction `2.5e-7`. The B.6 normalized margin must be at least `0.10`, the updated minimum internal energy must be positive, and the B.7 certificate remains a reported diagnostic. Work is `ceil(1/h)^2*(sum_states N + 3*111)`; feasibility is applied before minimum-work selection, and exact work ties retain candidate order.

## Problem

Determine the least-work certified design for the nondimensional 2.5D resistive Hall-MHD source system below, using its coupled implicit-midpoint velocity/magnetic-field update, residual-based resistivity, whistler accuracy and endpoint Newton-coercivity certificates. Evaluate the seven candidates over the prescribed coupled uncertainty lattice; the designated scalar is the selected row's maximum normalized wave-or-heating diagnostic, rounded once to six decimal places. In `<reasoning>`, report the selected `[h,tau]`, its maximum heat fraction, controlling diagnostic, largest normalized wave error, B.6 and B.7 minima, minimum updated internal energy, exact work and the feasible-candidate count. Identify the lattice member and scenario attaining its maximum heat fraction, and report the magnetic energy computed from that state's solved field and an upper bound for the largest absolute total-energy-balance defect over the selected row's states. Also identify the highest-work rejected row and give its robust score, B.6 minimum and failed limits. Report non-integer diagnostics other than the roundoff-defect bound with at least ten significant digits, and give concise source-grounded explanations of the coupled source-energy structure, the artificial-resistivity construction and the distinction between the two coercivity estimates; the matrices, lattice and decision thresholds are a new synthetic evaluation configuration.

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

lumped_current_projection

Goal
----
Recover the ordered nodal current proxy from a lumped L2 load.

```python
import numpy as np

def lumped_current_projection(lumped_mass, current_load):
    """Project a vector load with the lumped mass.

    Parameters
    ----------
    lumped_mass : array_like, shape (n,)
        Positive nodal lumped masses.
    current_load : array_like, shape (n,3)
        Cartesian weak curl loads in node order.
    Returns
    -------
    ndarray, shape (n,3)
        Projected Cartesian nodal current in the same order.
    """
    return None
```

### Step 2

projected_electron_velocity

Goal
----
Evaluate the projected electron velocity used by the induction subsystem.

```python
import numpy as np

def projected_electron_velocity(density, velocity, projected_current, ion_skin_depth):
    """Form the nodal electron velocity and speed.

    Parameters
    ----------
    density : float
        Positive scenario density.
    velocity : array_like, shape (n,3)
        Ion velocity in Cartesian component order.
    projected_current : array_like, shape (n,3)
        Projected nodal current in matching order.
    ion_skin_depth : float
        Nonnegative Hall parameter.
    Returns
    -------
    ndarray, shape (n,4)
        Columns [ve_x,ve_y,ve_z,|ve|] in node order.
    """
    return None
```

### Step 3

rescaled_induction_residual

Goal
----
Construct the paper's nodal induction-residual indicator.

```python
import numpy as np

def rescaled_induction_residual(lumped_mass, domain_measure,
                                magnetic_field, residual_load):
    """Project and normalize an induction residual load.

    Parameters
    ----------
    lumped_mass : array_like, shape (n,)
        Positive nodal masses.
    domain_measure : float
        Positive measure used by the magnetic mean.
    magnetic_field : array_like, shape (n,3)
        Cartesian nodal magnetic samples.
    residual_load : array_like, shape (n,3)
        Cartesian weak residual loads.
    Returns
    -------
    ndarray, shape (n,)
        Nonnegative rescaled indicator in node order.
    """
    return None
```

### Step 4

blended_artificial_resistivity

Goal
----
Evaluate the paper's low-order, residual, and effective nodal resistivities.

```python
import numpy as np

def blended_artificial_resistivity(mesh_h, physical_resistivity,
                                   electron_speed, rescaled_residual,
                                   node_scale, c_low=0.25, c_res=1.0):
    """Construct low-order, residual, and effective resistivities.

    Parameters
    ----------
    mesh_h : float
        Positive mesh size.
    physical_resistivity : float
        Nonnegative physical floor.
    electron_speed : array_like, shape (n,)
        Ordered nonnegative nodal speeds.
    rescaled_residual : array_like, shape (n,)
        Ordered nonnegative residual indicators.
    node_scale : array_like, shape (n,)
        Positive local length scales.
    c_low : float
        Nonnegative low-order constant.
    c_res : float
        Nonnegative residual constant.
    Returns
    -------
    ndarray, shape (n,3)
        Columns [r_low,r_res,r_effective] in node order.
    """
    return None
```

### Step 5

resistive_whistler_mode

Goal
----
Evaluate the right-polarized resistive whistler mode and one-period attenuation.

```python
import numpy as np

def resistive_whistler_mode(density, background_field, ion_skin_depth,
                            resistivity, wavenumber):
    """Evaluate one right-polarized resistive whistler mode.

    Parameters
    ----------
    density : float
        Positive modal density.
    background_field : float
        Positive background magnetic field.
    ion_skin_depth : float
        Nonnegative Hall coefficient.
    resistivity : float
        Nonnegative resistive coefficient.
    wavenumber : float
        Positive modal wavenumber.
    Returns
    -------
    ndarray, shape (4,)
        [Re(omega),Im(omega),period,one_period_amplitude].
    """
    return None
```

### Step 6

crank_nicolson_whistler_certificate

Goal
----
Certify the modal Crank--Nicolson source update against the continuous whistler mode.

```python
import numpy as np

def crank_nicolson_whistler_certificate(omega_real, omega_imag,
                                        time_step, periods=1.0):
    """Compare a centered modal update with continuous propagation.

    Parameters
    ----------
    omega_real : float
        Positive real modal frequency.
    omega_imag : float
        Nonpositive imaginary modal frequency.
    time_step : float
        Positive update size.
    periods : float
        Positive comparison horizon in periods.
    Returns
    -------
    ndarray, shape (6,)
        [amplitude_error,phase_error,log_error,steps,time,absolute_mode].
    """
    return None
```

### Step 7

newton_coercivity_certificates

Goal
----
Evaluate both normalized Newton-Jacobian coercivity certificates from the paper.

```python
import numpy as np

def newton_coercivity_certificates(time_step, permeability, mass_constant,
                                    density_min, current_bound,
                                    electron_speed_bound, resistivity_min):
    """Evaluate the two normalized coercivity coefficient triples.

    Parameters
    ----------
    time_step : float
        Positive time step.
    permeability : float
        Positive magnetic permeability.
    mass_constant : float
        Positive mass-norm constant.
    density_min : float
        Positive density lower bound.
    current_bound : float
        Nonnegative projected-current bound.
    electron_speed_bound : float
        Nonnegative electron-speed bound.
    resistivity_min : float
        Positive effective-resistivity lower bound.
    Returns
    -------
    ndarray, shape (6,)
        Normalized [A6,B6,C6,A7,B7,C7] without clipping.
    """
    return None
```

### Step 8

source_energy_update

Goal
----
Update nodal mechanical energy and predict the magnetic norm implied by the energy balance.

```python
import numpy as np

def source_energy_update(lumped_mass, density, momentum_old, momentum_new,
                         total_energy_old, joule_power, time_step,
                         magnetic_norm_old_sq, permeability=1.0):
    """Advance nodal source energy and form an independent magnetic-balance prediction.

    Parameters
    ----------
    lumped_mass : array_like, shape (n,)
        Positive nodal masses.
    density : array_like, shape (n,)
        Positive nodal densities.
    momentum_old : array_like, shape (n,3)
        Cartesian momentum before the source solve.
    momentum_new : array_like, shape (n,3)
        Cartesian momentum after the source solve.
    total_energy_old : array_like, shape (n,)
        Old nodal total energy.
    joule_power : array_like, shape (n,)
        Nonnegative nodal Joule power.
    time_step : float
        Positive source-step size.
    magnetic_norm_old_sq : float
        Nonnegative old magnetic norm squared.
    permeability : float
        Positive magnetic permeability.
    Returns
    -------
    ndarray, shape (n+3,)
        Nodal E_new followed by [magnetic_norm_new_sq,min_internal,Joule_heat].
    """
    return np.zeros(len(lumped_mass)+3, dtype=float)
```

### Step 9

paper_candidate_table

Goal
----
Aggregate paper-native wave, coercivity, and energy certificates over coupled states.

```python
import numpy as np

def paper_candidate_table(candidates, wave_cube, coercivity_cube, energy_cube,
                          amplitude_limit=1e-5, phase_limit=8e-5,
                          log_damping_limit=1e-5,
                          heat_fraction_limit=1e-4,
                          b6_margin_min=0.10):
    """Aggregate stress-state certificates into candidate rows.

    Parameters
    ----------
    candidates : array_like, shape (C,2)
        Ordered [h,tau] rows.
    wave_cube : array_like, shape (C,S,6)
        Modal certificates over stress states.
    coercivity_cube : array_like, shape (C,S,6)
        Normalized B.6/B.7 coefficient triples.
    energy_cube : array_like, shape (C,S,2)
        [minimum_internal,heat_fraction] certificates.
    amplitude_limit : float
        Positive amplitude-error scale.
    phase_limit : float
        Positive phase-error scale.
    log_damping_limit : float
        Positive log-damping-error scale.
    heat_fraction_limit : float
        Positive Joule-heat-fraction scale.
    b6_margin_min : float
        Positive admissibility threshold.
    Returns
    -------
    ndarray, shape (C,12)
        [h,tau,wave3,b6min,b7min,heat,internal,score,work,feasible].
    """
    return None
```

### Step 10

hall_mhd_paper_design

Goal
----
Orchestrate the coupled implicit-midpoint source solve and select the least-work robustly certified design.

```python
import numpy as np

def hall_mhd_paper_design(candidates, density, background_field,
                          ion_skin_depth, physical_resistivity, wavenumber,
                          velocity, current_load, lumped_mass, magnetic_nodes,
                          residual_load, node_scale, internal_energy,
                          permeability=1.0, mass_constant=0.92,
                          c_low=0.25, c_res=1.0,
                          amplitude_limit=1e-5, phase_limit=8e-5,
                          log_damping_limit=1e-5,
                          heat_fraction_limit=2.5e-7,
                          b6_margin_min=0.10, *, source_curl):
    """Select a certified resistive Hall-MHD design.

    Parameters
    ----------
    candidates : array_like, shape (C,2)
        Ordered [h,tau] design rows.
    density : array_like, shape (S,)
        Positive scenario densities.
    background_field : array_like, shape (S,)
        Positive scenario background fields.
    ion_skin_depth : array_like, shape (S,)
        Nonnegative scenario Hall coefficients.
    physical_resistivity : array_like, shape (S,)
        Nonnegative scenario physical resistivities.
    wavenumber : array_like, shape (S,)
        Positive scenario wavenumbers.
    velocity : array_like, shape (S,n,3)
        Ordered Cartesian ion velocities.
    current_load : array_like, shape (S,n,3)
        Ordered Cartesian weak curl loads.
    lumped_mass : array_like, shape (n,)
        Positive nodal masses.
    magnetic_nodes : array_like, shape (S,n,3)
        Ordered Cartesian magnetic samples.
    residual_load : array_like, shape (S,n,3)
        Ordered Cartesian weak residual loads.
    node_scale : array_like, shape (n,)
        Positive local length scales.
    internal_energy : array_like, shape (S,n)
        Positive nodal internal energy.
    permeability : float
        Positive magnetic permeability.
    mass_constant : float
        Positive mass-norm constant.
    c_low : float
        Nonnegative low-order resistivity constant.
    c_res : float
        Nonnegative residual-resistivity constant.
    amplitude_limit : float
        Positive amplitude-error limit.
    phase_limit : float
        Positive phase-error limit.
    log_damping_limit : float
        Positive log-damping-error limit.
    heat_fraction_limit : float
        Positive Joule-heat-fraction limit.
    b6_margin_min : float
        Positive B.6 admissibility threshold.
    source_curl : array_like, shape (3*n,3*n)
        Required finite real node-major discrete curl for the midpoint solve.
        Baseline weak current loads equal M times C times magnetic_nodes.
    Raises
    ------
    ValueError
        Invalid shapes, non-real curl, incompatible current loads, nonpositive
        initial internal energy, unresolved source solve or no feasible row.
    Returns
    -------
    float
        Selected row's robust score rounded once to six decimals.
    """
    return 0.0
```
