# Physics-Optics-48

## Background

Scientific overview

The source lattice uses a mirror-symmetric pair of nonlinear active plasma lenses to preserve achromatic transport while cancelling the leading positional geometric aberration. The benchmark extends that paper-specific construction to an uncertainty-robust energy-scaling design and checks the asymptotic model against the exact two-lens map and a correlated lens-misalignment action.

Instance data and author-defined decision contract

Reference state (all quantities are signed where shown):

E_r = 50 GeV
L_r = sqrt(5) m
l_r = 2 sqrt(5) m
beta_r = 0.015 sqrt(5) m
B_r = 1.0 T
tau_r = -61.42 m^-1
R56_r = -1.0e-4 m

The exponent p denotes the magnitude of the main-dipole field's energy-scaling exponent. Scan 101 uniformly spaced p values from 0.25 through 0.75, inclusive. For each p, scan 401 logarithmically spaced energy nodes from 50 through 5000 GeV, inclusive and in ascending order. Generalize the source paper's staging-similarity prescription continuously in p while preserving its achromatic and mirror-symmetry invariants.

The robust design set is the full Cartesian product

sigma_delta in {0.018, 0.032},
eps_nx in {8e-6, 12e-6} m rad,
eps_ny in {0.10e-6, 0.18e-6} m rad,
A_ISR in {0.025, 0.040}.

A_ISR is the dimensionless horizontal incoherent-synchrotron-radiation growth amplitude at E_r. For each scenario define

R_x = sqrt(1 + g_ch^2 + g_x^2 + g_ISR^2),
R_y = sqrt(1 + g_ch^2 + g_y^2).

Thus unity is inside the square root and independent increments are combined in quadrature. At a candidate's last feasible node, define its normalized load as

max(R_x_worst/R_x_max, R_y_worst/R_y_max, |tau_x|/tau_max, B_min/B),

where the two emittance entries are the componentwise worst ratios over the scenario set.

At every energy node, form the componentwise worst horizontal and vertical outgoing-to-incoming normalized-emittance ratios over all 16 rows. A node is feasible when both ratios are at most 1.25, the nonlinear-lens taper magnitude is at most 520 m^-1, and the main-dipole field is at least 0.040 T. Feasibility is inclusive. A candidate's score is its longest feasible prefix from the first energy node. Rank candidates by the highest last-feasible energy, then the smaller maximum normalized utilization of the four limits at that node, then the smaller p. At the first node after the selected prefix, identify the active limit by first-maximum priority in the order horizontal emittance, vertical emittance, taper magnitude, minimum field.

For the exact-map validation use physicists' Gauss-Hermite order 7, the largest horizontal and vertical normalized emittances in the robust design set, equal rms lens offsets of 0.20 times the matched horizontal beam size, and correlation rho=-0.35. Use the angle-dominated Appendix-D reduction: sample the independent entrance slopes, set the first-lens positions by the preceding drift, and propagate the exact repeated nonlinear kicks. When reconstructing exact emittance, restore the independent matched positional variance eps beta before taking each centered covariance determinant. Define q_g=max(g_x,exact/g_x,asym, g_y,exact/g_y,asym). Define n_x,n_y=log2[s(tau_x)/s(tau_x/2)], where s is the centered rms final positional residual in the corresponding plane. First-order cancellation in this construction is positional; angular terms survive. The equal-rms offset tolerance is the common lens-offset rms, expressed in matched sigma_x units, for which the mean induced horizontal action equals eps_nx.

Additional source-era telemetry available for physical consistency checks:

10 GeV operating example; 50 pC charge; 3 micrometre rms bunch length; 0.050 m plasma-lens length; 0.850 m chicane-dipole length; 0.250 m central-sextupole length; 0.025 m inter-element gaps; 978.6 T/m central plasma-lens gradient; 0.022 T and -0.261 T chicane fields; 9921 T/m^2 sextupole field; 2.74 degree net bend; 359.15 degree total phase advance.

## Problem

Determine the largest robust prefix-feasible beam energy for a mirror-symmetric achromatic nonlinear-plasma-lens staging lattice, jointly selecting the main-dipole field-scaling exponent from the supplied scan. Recover the required similarity, local-achromaticity, chromatic-aberration, nonlinear-lens, radiation, and alignment relations from the source paper. Apply the supplied uncertainty design, feasibility limits, node ordering, and deterministic tie-break to the complete scan. Validate the selected design against the source's exact nonlinear two-lens map and the correlated two-lens offset action obtained from the paper's drift-kick equations. Report the selected energy in GeV, rounded once to 0.1 GeV, as the tagged scalar. In the short reasoning report the selected exponent to 0.001; the selected-node worst-corner growth quartet g_ch, g_x, g_y, and g_ISR; both worst-case emittance ratios; the larger exact-to-asymptotic geometric-growth ratio; both measured cancellation orders; the correlated-offset action fraction and its equal-rms tolerance to 5 decimal places; the taper magnitude to 0.1 m^-1; the dipole field to 5 decimal places in tesla; and the named next-node active limit.

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

scale_staging_lattice

Goal
----
Scale the paper's mirror-symmetric nonlinear-plasma-lens lattice to one energy and dipole exponent.

```python
import numpy as np

def scale_staging_lattice(energy_gev, exponent, reference):
    """Return scaled lattice quantities at one energy.

    Args:
        energy_gev: Positive finite beam energy in GeV.
        exponent: Finite dipole-field exponent p.
        reference: Length-7 numerical array [E_r,L_r,l_r,beta_r,B_r,tau_r,R56_r]
            in [GeV,m,m,m,T,m^-1,m].

    Returns:
        numpy.ndarray: Length-7 float array [L,l,beta,B,tau_x,D_x,R56] in
        [m,m,m,T,m^-1,m,m].

    Raises:
        ValueError: If energy is nonpositive/nonfinite, exponent is nonfinite,
            reference is not a finite length-7 physical array (E_r,L_r,l_r,
            beta_r,B_r positive and tau_r nonzero), or the scaled result is
            outside the finite supported numerical domain.
    """
    return None
```

### Step 2

second_order_chromatic_growth

Goal
----
Evaluate the paper's achromatic-lattice second-order chromatic emittance increment.

```python
import math

def second_order_chromatic_growth(sigma_delta, length_m, half_gap_m, beta_m):
    """Return the dimensionless second-order chromatic emittance increment.

    Args:
        sigma_delta: Nonnegative dimensionless rms relative energy spread.
        length_m: Positive lattice length L in metres.
        half_gap_m: Positive half-cell gap l in metres.
        beta_m: Positive beta function in metres.

    Returns:
        float: One finite nonnegative dimensionless scalar.

    Raises:
        ValueError: If any input is nonfinite, sigma_delta is negative, a
            length is nonpositive, or the result is not a finite float.
    """
    return None
```

### Step 3

nonlinear_geometric_growth

Goal
----
Evaluate the residual O(tau_x^2) normalized-emittance increments after the paper's -I cancellation.

```python
import numpy as np

def nonlinear_geometric_growth(energy_gev, length_m, half_gap_m, beta_m,
                               tau_per_m, eps_nx_m_rad, eps_ny_m_rad):
    """Return residual geometric increments [g_x,g_y].

    Args:
        energy_gev: Positive beam energy in GeV.
        length_m: Positive lattice length L in metres.
        half_gap_m: Positive half-cell gap l in metres.
        beta_m: Positive beta function in metres.
        tau_per_m: Finite nonlinear-lens taper tau_x in inverse metres.
        eps_nx_m_rad: Positive horizontal normalized emittance in m rad.
        eps_ny_m_rad: Positive vertical normalized emittance in m rad.

    Returns:
        numpy.ndarray: Length-2 nonnegative float array [g_x,g_y], dimensionless.

    Raises:
        ValueError: If an input is nonfinite; energy, a length, or an emittance
            is nonpositive; or either result is outside the finite supported
            numerical domain.
    """
    return None
```

### Step 4

incoherent_radiation_growth

Goal
----
Propagate the calibrated bending-plane ISR increment under a general dipole-field exponent.

```python
import math

def incoherent_radiation_growth(energy_gev, exponent, reference_energy_gev,
                                reference_growth):
    """Return the dimensionless horizontal ISR emittance increment.

    Args:
        energy_gev: Positive beam energy E in GeV.
        exponent: Finite dipole scaling exponent p.
        reference_energy_gev: Positive reference energy E_r in GeV.
        reference_growth: Nonnegative dimensionless ISR increment at E_r.

    Returns:
        float: One finite nonnegative dimensionless scalar.

    Raises:
        ValueError: If an input is nonfinite, either energy is nonpositive,
            reference_growth is negative, or a nonzero result cannot be
            represented as a finite float.
    """
    return None
```

### Step 5

combine_emittance_increments

Goal
----
Combine independent chromatic, geometric and ISR increments into outgoing-emittance ratios.

```python
import numpy as np

def combine_emittance_increments(chromatic_growth, geometric_growth, isr_growth):
    """Return outgoing-to-incoming emittance ratios [R_x,R_y].

    Args:
        chromatic_growth: Nonnegative dimensionless chromatic increment.
        geometric_growth: Length-2 nonnegative array [g_x,g_y].
        isr_growth: Nonnegative dimensionless horizontal ISR increment.

    Returns:
        numpy.ndarray: Length-2 float array [R_x,R_y], dimensionless.

    Raises:
        ValueError: If any increment is nonfinite or negative, or if
            geometric_growth is not a finite length-2 array, or either
            quadrature result cannot be represented as a finite float.
    """
    return None
```

### Step 6

robust_energy_profile

Goal
----
Evaluate the componentwise worst uncertainty corner at every energy for one exponent.

```python
import numpy as np

def robust_energy_profile(exponent, energies_gev, scenarios, reference):
    """Return worst-case ratios and resources for one exponent.

    Args:
        exponent: Finite p.
        energies_gev: Length-N increasing positive numerical array in GeV.
        scenarios: Shape-(S,4) array with columns [sigma_delta,eps_nx,eps_ny,A]
            in [1,m rad,m rad,1], with S>=1.
        reference: Seven-element reference array used by scale_staging_lattice.

    Returns:
        numpy.ndarray: Shape-(N,4) float array with columns
        [max_R_x,max_R_y,abs_tau_x,B] in [1,1,m^-1,T].

    Raises:
        ValueError: If arrays have the wrong shape, contain nonfinite or
            nonphysical values, energies are not positive and strictly
            increasing, exponent is nonfinite, or a downstream result is
            outside its documented finite supported domain.
    """
    return None
```

### Step 7

select_minimax_exponent

Goal
----
Apply prefix feasibility and the deterministic minimax tie-break across exponent candidates.

```python
import numpy as np

def select_minimax_exponent(exponents, energies_gev, scenarios, reference, limits):
    """Return the selected exponent and prefix-feasibility certificate.

    Args:
        exponents: Nonempty finite one-dimensional candidate-p array.
        energies_gev: Positive strictly increasing energy grid in GeV.
        scenarios: Shape-(S,4) array [sigma_delta,eps_nx,eps_ny,A].
        reference: Seven-entry reference-lattice array.
        limits: Positive four-entry array [R_x_max,R_y_max,abs_tau_max,B_min].

    Returns:
        numpy.ndarray: Length-5 float array [p,E_last,last_index,load,next_code],
        where E_last is in GeV, last_index and next_code are exact integer-valued
        floats, load is dimensionless, and next_code is in {0,1,2,3,4}.

    Raises:
        ValueError: If any array is malformed or nonfinite, physical grid or
            limit conditions fail, a downstream result is outside its finite
            supported domain, or no exponent is feasible at the first node.
    """
    return None
```

### Step 8

transport_nonlinear_lens_pair

Goal
----
Propagate one or many angular rays through the paper's exact two-lens nonlinear thin-map.

```python
import numpy as np

def transport_nonlinear_lens_pair(initial_slopes, length_m, half_gap_m, tau_per_m):
    """Return the exact final phase-space coordinates for angle-dominated rays.

    Args:
        initial_slopes: Finite float array with final dimension 2, [x'_0,y'_0].
        length_m: Positive drift length L in metres.
        half_gap_m: Positive half-cell gap l in metres.
        tau_per_m: Finite nonlinear taper tau_x in inverse metres.

    Returns:
        numpy.ndarray: Same leading shape as initial_slopes and final dimension 4,
        ordered [x_3,x'_3,y_3,y'_3] in [m,rad,m,rad].

    Raises:
        ValueError: If the ray array is empty, malformed, or nonfinite; a
            drift length is nonpositive; tau_x is nonfinite; or the mapped
            result is outside the finite supported numerical domain.
    """
    return None
```

### Step 9

nonlinear_lens_validity_certificate

Goal
----
Construct an exact Appendix-D/E validity certificate from Gaussian moments and correlated lens offsets.

```python
import numpy as np

def nonlinear_lens_validity_certificate(energy_gev, length_m, half_gap_m, beta_m,
                                        tau_per_m, eps_nx_m_rad, eps_ny_m_rad,
                                        offset_fraction, offset_correlation,
                                        quadrature_order):
    """Return exact-map, asymptotic, cancellation and alignment diagnostics.

    Args:
        energy_gev: Positive energy in GeV.
        length_m, half_gap_m, beta_m: Positive lengths L,l,beta in metres.
        tau_per_m: Finite taper tau_x in inverse metres satisfying
            1e-12 <= abs(tau_x)*max(L,l,beta) <= 1e6.
        eps_nx_m_rad, eps_ny_m_rad: Positive normalized emittances in m rad.
        offset_fraction: Positive rms lens offset in units of matched sigma_x.
        offset_correlation: Correlation rho of the two equal-rms lens offsets.
        quadrature_order: Odd integer from 5 through 15 inclusive.

    Returns:
        numpy.ndarray: Length-8 float array
        [g_x_exact,g_y_exact,g_x_asym,g_y_asym,n_x,n_y,<J_x>/eps_nx,
         sigma_Delta_max/sigma_x].

    Raises:
        ValueError: If an input is nonfinite; a physical scale, emittance, or
            offset_fraction is nonpositive; rho is outside [-1,1];
            quadrature_order is not a finite odd integer from 5 through 15;
            the taper fails the documented resolved-probe range; or any map,
            moment, action, or returned diagnostic is not finite and resolved.
    """
    return None
```

### Step 10

run_robust_scaling_benchmark

Goal
----
Run the whole robust plasma-lens scaling benchmark and return its rounded energy ceiling.

```python
import numpy as np

def run_robust_scaling_benchmark(exponents=None, energies_gev=None, scenarios=None,
                                 reference=None, limits=None):
    """Return the robust energy ceiling for the supplied or canonical benchmark.

    Args:
        exponents: Optional finite nonempty one-dimensional candidate-p array;
            canonical default is 0.25+0.005*k, k=0,...,100.
        energies_gev: Optional positive strictly increasing energy grid in GeV;
            canonical default is 50*exp(j*ln(100)/400), j=0,...,400.
        scenarios: Optional finite shape-(S,4) array [sigma_delta,eps_nx,
            eps_ny,A_ISR]; canonical default is the 16-row Cartesian product
            [0.018,0.032] x [8e-6,12e-6] x [0.10e-6,0.18e-6] x
            [0.025,0.040], with the last coordinate varying fastest.
        reference: Optional seven-entry [E_r,L_r,l_r,beta_r,B_r,tau_r,R56_r];
            canonical default is [50,sqrt(5),2sqrt(5),0.015sqrt(5),1,-61.42,-1e-4].
        limits: Optional positive [R_x_max,R_y_max,abs_tau_max,B_min];
            canonical default is [1.25,1.25,520,0.040].
            Either omit all five arguments or supply all five.

    Returns:
        float: Selected energy in GeV, rounded once to 0.1 GeV.

    Raises:
        ValueError: If only some optional arguments are supplied; an explicit
            array is malformed/nonfinite or violates its documented physical
            domain; no exponent is feasible at the first energy; or a
            downstream calculation leaves its supported finite domain.
        RuntimeError: If the independently recomputed selected-node profile,
            exact-map probe, or fixed validity certificate is internally
            inconsistent with the public subproblem results.
    """
    return None
```
