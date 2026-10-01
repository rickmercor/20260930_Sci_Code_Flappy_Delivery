# Physics-Astrophysics-40

## Background

# Scientific background

Pre-impact uncertainty and material variability affect both the number of meteorites that survive and their joint ground-position and mass distribution. Atmospheric entry, repeated fragmentation, and dark flight therefore feed a single event-level prediction rather than independent calculations.

A finite survivor inventory and a continuous reconstruction of its position-log-mass footprint have different roles: the inventory controls depletion, while the normalized footprint supports measurement convolution and integration over a finite search target.

A complete search catalog may mix objects from the witnessed fall with accumulated meteorites from older events. Their origin labels remain latent, so campaign-A evidence and campaign-B prediction must share one finite physical inventory rather than classify finds before inference. The second campaign revisits the same surviving objects with unchanged true marks.

The local atmosphere, force law, numerical thresholds, finite-inventory surrogate, recovery responses, reporting laws, and nuisance priors are task-defined assumptions. The source paper supplies the physical fragmentation method; the continuous-density fitting contract and statistical coupling are task-defined assumptions.

## Problem

## Setup

An uncertain stony impactor fragments in a rotating, windy atmosphere, and a complete first search catalog contains an unknown mixture of its survivors and accumulated meteorites from older falls, constraining the chance of a new witnessed-fall recovery in a second campaign.

Coordinates are local east-north-up meters, velocities `m/s`, masses `kg`, strengths `Pa`, and logarithms natural; the eight simulations represent alternative physical states of one event, with one population `K`, one scenario `r`, and one efficiency `p` shared by every witnessed-fall object and both campaigns.

## Inputs

One `Generator(PCG64(761903))` supplies eight realizations, each consuming `g=standard_normal(6)` for `mu+numpy.linalg.cholesky(C)@g` with the lower factor, then independent uniforms in the following range order, followed by child angles in FIFO queue order:

```text
mu=(0,0,100000,12400,180,-5250)
C=((420^2,.35*420*260,0,0,0,0),(.35*420*260,260^2,0,0,0,0),
   (0,0,95^2,0,0,-.25*95*18),(0,0,0,26^2,.2*26*11,0),
   (0,0,0,.2*26*11,11^2,0),(0,0,-.25*95*18,0,0,18^2))
ranges=(density [.92,1.08], S [.42e6,.72e6], dust [.18,.34], largest [.25,.38],
        beta [.48,.66], wind [.88,1.12], drag [.90,1.10], sigma_abl [7.2e-8,8.8e-8])
mass0=185000*density
theta=((S-.42e6)/.30e6,(dust-.18)/.16,(beta-.48)/.18,(wind-.88)/.24)
```

| Population and recovery inputs | Value |
|:---|:---:|
| Proposal for `theta` | Uniform on `(0,1)^4` |
| Prior for `K=0,1,2`, before either calibration | `(.30,.45,.25)` |
| Independent Beta pairs, theta order, `K=0` | `((2,5),(5,2),(2,2),(2,5))` |
| Independent Beta pairs, `K=1` | `((2,2),(2,2),(2,2),(2,2))` |
| Independent Beta pairs, `K=2` | `((5,2),(2,5),(5,2),(5,2))` |
| Complete first-campaign catalog `(east,north,mass)` | Exactly three finds, in reported order: `(196900,4550,.095)`, `(199650,5250,.022)`, `(198150,5050,.058)`; no other detections; no measured fall-association labels |
| New-fall standardized measurement covariance | `((.0875^2,.25*.0875*.125,0),(.25*.0875*.125,.125^2,0),(0,0,.18^2))` |
| Prediction ellipse center and semiaxes | `(198500,4850)`, `(2600,1300)` |
| First-axis angle, counterclockwise from east | `22 degrees` |
| Prediction mass interval | `[.025,.18]` |
| Independent efficiency calibration `(trials,successes)` | `(13,2)`, `(17,5)`, `(11,3)`, `(19,6)`; response one for every analog object, sharing the new fall's unknown visibility `p` and population `K` |
| Visibility law conditional on `K=0,1,2` | `p~Beta(alpha_K,beta_K)` with pairs `((2.3,5.7),(4.1,10.3),(1.2,3.8))`; independent of `r` given `K`; common across calibration and both campaigns |
| Accumulated-meteorite rate prior | `lambda~Gamma(shape=2.4,rate=1.6)`, density `1.6^2.4*lambda^(2.4-1)*exp(-1.6*lambda)/Gamma(2.4)`, independent of `K,r,p` |
| Accumulated-only independent control | Four detections at exposure `3`, count law `Poisson(3*lambda)` |
| Accumulated detections in campaign A | `Poisson(lambda)` count, independently marked with standardized reported density `g=N(mu_g,V_g)`; independent of new-fall detections given the latent variables |
| Accumulated reported marks | `mu_g=(.1,-.15,.05)`, `V_g=((.7,.08,.12),(.08,.5,-.06),(.12,-.06,.9))`; already includes selection and reporting error, no additional `p`, response, or convolution |
| Response function | `s_j(m)=expit(ln(m/m_j)/w_j)`, `expit(t)=1/(1+exp(-t))` |
| First campaign | Entire ground plane and all positive masses; `(m_A,w_A)=(.06,.75)` |
| Second campaign | The target ellipse and mass interval only; `(m_B,w_B)=(.035,.9)` |
| Detection and measurement | Bernoulli probability `p*s_j(m)` for each eligible true object; independent trials conditional on its true mark and `p`; measurement errors only after detection |

## Physical model

Each sphere obeys the prescribed local model below, with terminal-event priority downward ground, downward `.001 kg`, then upward ram pressure `rho*|v-w|^2/2=S`, breakups only at depths zero and one, and retained impacts at least `.01 kg`.

| Local physical definition | Value |
|:---|:---:|
| Height and air density | `h=max(z,0)`, `rho=1.225*exp(-h/7200)` |
| Temperature and sound speed | `T=216.65+71*exp(-((h-9000)/8500)^2)`, `c=sqrt(1.4*287*T)` |
| Wind `(east,north,up)` | `wind_scale*(8+17*exp(-((h-12000)/7000)^2),-3+24*exp(-((h-10000)/6500)^2)-8*exp(-((h-31000)/9000)^2),0)` |
| Drag coefficient | Linear interpolation at `min(abs(v-w)/c,4)` through Mach `(0,.5,.8,1,1.2,2,3,4)` and `Cd=(.47,.48,.55,.92,1.08,.91,.72,.64)`, multiplied by `drag` |
| Sphere and drag acceleration | Density `3050 kg/m^3`, `r=(3*m/(4*pi*3050))^(1/3)`, `A=pi*r^2`, `-rho*Cd*A*abs(v-w)*(v-w)/(2*m)` |
| Gravity and Coriolis acceleration | `(0,0,-9.80665*(6371000/(6371000+h))^2)` and `(2*7.2921159e-5*vy,-2*7.2921159e-5*vx,0)` |
| Ablation | `dm/dt=-sigma_abl*rho*A*abs(v-w)^3/2` for `abs(v-w)>=3000`, otherwise zero |
| Vector speed notation | `abs(v-w)` denotes Euclidean norm |

At breakup, resolved children follow the source-specific mass-partition, transverse-dispersion, and momentum-conservation model, with child proposals attempted only for generated ranks `k=2,...,7`, the task-defined `.2 kg` proposal and residual floor, `.16` strength exponent capped at `15e6 Pa`, generated-rank child-angle draws, depth increment, and FIFO ordering. For each parent velocity, set `e=v/|v|`, use axis `(0,0,1)` when `|e_z|<.9` and `(0,1,0)` otherwise, define `b1=normalize(cross(e,axis))` and `b2=cross(e,b1)`, and map child angle `phi` to `cos(phi)*b1+sin(phi)*b2`.

Each scenario's continuous footprint is a four-component full-covariance Gaussian mixture in `z=((east-198000)/4000,(north-5000)/2000,ln(mass)-ln(.05))`, fitted with equal weights over that scenario's retained fragments, while its prior probability conditional on `K` is proportional to the product of the four normalized Beta densities at its `theta`, normalized over the eight scenarios.

For the recovery model, scenario `r` has exactly its simulated survivor count `N_r` with iid latent true marks from its continuous footprint, only genuinely associated first-campaign detections deplete this inventory, and the observed catalog is an independent uniform random permutation of the merged new-fall and accumulated detections conditional on their total count. Campaign B searches the same remaining physical objects with the same true marks; the supplied finite-inventory surrogate, population-linked visibility, Gaussian accumulated reporting density, and Gamma rate uncertainty are task-defined extensions of the cited physical model, with both calibration channels held out of the field catalog.

## Task

Compute **the dimensionless posterior probability `P_search` of at least one new witnessed-fall recovery in campaign B** conditional on the complete first catalog and both calibrations. In the concise reasoning, report the eight scenario-ordered survivor counts, the three joint contributions `C_K=Pr(K AND at least one new witnessed-fall recovery in B | all evidence)` for `K=0,1,2`, and their sum `P_search`; summarize the mass-partition and transverse-velocity laws used in the calculation.

## Numerical conventions

Use float64 and the following deterministic conventions without clipping or rounding probabilities or resetting the random stream.

| Numerical convention | Value |
|:---|:---:|
| Trajectories | SciPy `solve_ivp(method="RK45")`, `[0,5000]`, `rtol=2e-10`, `atol=(1e-5,1e-5,1e-5,1e-6,1e-6,1e-6,1e-12)`, `max_step=.35` |
| Events | Directions `(-1,-1,+1)`, first recorded event in declared priority, impact altitude set to zero |
| GMM initialization, separately per scenario | Stable mergesort by `.72*z0-.28*z1+.35*z2`, `array_split` into four blocks, block means, uniform component weights, common global population covariance |
| Covariance regularization | Add `diag(.0064,.0100,.0049)` initially and after each covariance update |
| EM | Exactly 120 weighted full-covariance Gaussian-mixture updates; use population rather than sample covariance and preserve component order |
| Density support | Continuous normalized Gaussians on all standardized space, no sample truncation or edge renormalization |
| Search quadrature | Independent 56-node radial and 88-node angular Gauss-Legendre rules on `[0,1]` and `[0,2*pi]`, ellipse map `center+R(22 degrees)@(2600*r*cos(phi),1300*r*sin(phi))`; 48-node Gauss-Legendre on `[ln(.025/.05),ln(.18/.05)]` |
| Continuous response integrals | Integrate each normalized joint scenario density and stated response over the mapped position/log-mass domain; ellipse-map Jacobian `2600*1300*r/(4000*2000)` |
| Full-line logistic expectations | 64-node Gauss-Hermite with weight `exp(-x^2)` and the usual normal mean/standard-deviation transformation, also for latent mass conditional on a noisy mark |
| Efficiency integration | Use 64-node Gauss-Jacobi after `p=(x+1)/2` for normalized Beta-weighted visibility integrals; the calibration binomials and field evidence share the population-specific `p` |
| Catalog probability measure | Use the joint complete count and randomly ordered marks generated by the binomial new-fall and Poisson accumulated processes, with uniform interleaving conditional on both counts; do not condition away either count |
| Rate integration | Analytically marginalize the Gamma posterior under the control count likelihood, using the stated rate convention |
| Measurement covariance validation | Symmetry tolerance `1e-12`, then symmetrize; eigenvalues below `-1e-12` invalid, other negative eigenvalues set to zero before use |
| Reporting tolerances, absolute | `P_search` is `2e-5`; each `C_K` is `1e-5`; survivor counts and ordering are exact |
| Code-comparison tolerances, absolute | Step 06 east/north coordinates `5e-3 m`, masses `5e-6 kg`, and run labels, row ordering, survivor counts, nuisance values, and breakup counts exact; Step 10 and Integration-test probabilities `1e-7` |

Output Format Requirements:
Emit <final_answer> immediately, then <reasoning>. Do not write a long
derivation before the tags.
You must emit exactly one finite decimal inside
<final_answer>...</final_answer>, even if the value is approximate or you
are unsure.
Rules:
- The tags are required. Do not omit them or leave them empty.
- The tagged value must be a finite decimal (examples: 0.4847, 0.22, 1.05).
Not NaN, not Inf, not a fraction string, not a vector, not prose.
- Put only that one number between the tags. No units, no words, no extra
lines.
Keep <reasoning> short (a few hundred words). Show only the few scalars that
determine the final number.
Do not paste the input matrices, full coefficient vectors, per-iteration
paths, or per-fold candidate tables.

## Output format

```
Output Format Requirements:
Emit <final_answer> immediately, then <reasoning>. Do not write a long
derivation before the tags.
You must emit exactly one finite decimal inside
<final_answer>...</final_answer>, even if the value is approximate or you
are unsure.
Rules:
- The tags are required. Do not omit them or leave them empty.
- The tagged value must be a finite decimal (examples: 0.4847, 0.22, 1.05).
Not NaN, not Inf, not a fraction string, not a vector, not prose.
- Put only that one number between the tags. No units, no words, no extra
lines.
Keep <reasoning> short (a few hundred words). Show only the few scalars that
determine the final number.
Do not paste the input matrices, full coefficient vectors, per-iteration
paths, or per-fold candidate tables.
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

sample entry

Goal
----
Step 1: draw one entry state, physical realization, and nuisance point.

```python
"""Step 1: draw one entry state, physical realization, and nuisance point."""
import numpy as np

def sample_entry(rng=None):
    """Draw one entry realization in the stated random-stream order.

    Parameters
    ----------
    rng : numpy.random.Generator or None
        Advancing stream with standard_normal and uniform methods, or a new
        PCG64(761903) stream when None. Draws and ranges are in the statement.

    Returns
    -------
    ndarray, shape (18,)
        (x,y,z,vx,vy,vz,mass,strength,dust,largest,beta,wind_scale,
        cd_scale,sigma_abl,u_strength,u_dust,u_beta,u_wind).

    Raises
    ------
    ValueError
        If rng lacks either required random-number method.
    Notes
    -----
    Return the stated deterministic result to rtol=1e-10 and atol=1e-12.
    The supplied numerical controls define the discrete target.
    """
    return None
```

### Step 2

atmospheric rhs

Goal
----
Step 2: evaluate local atmosphere, loading, drag, gravity, and ablation.

```python
"""Step 2: evaluate local atmosphere, loading, drag, gravity, and ablation."""
import math
import numpy as np

def atmospheric_rhs(state, wind_scale=1.0, cd_scale=1.0, sigma_abl=8e-08):
    """Evaluate the prescribed spherical-fragment dynamics.

    Parameters
    ----------
    state : array_like, shape (7,)
        Finite (east,north,up,vx,vy,vz,mass), with positive mass.
    wind_scale, cd_scale, sigma_abl : float
        Finite wind multiplier >=0, drag multiplier >0, and ablation
        coefficient >=0 in kg/J. Atmosphere and force definitions are in
        the statement and this step's scientific background.

    Returns
    -------
    ndarray, shape (16,)
        Seven state derivatives, then (rho,sound,wind_x,wind_y,wind_z,
        relative_speed,ram_pressure,Cd,area).

    Raises
    ------
    ValueError
        For wrong state shape, nonfinite input, nonpositive mass or drag
        multiplier, or negative wind multiplier or ablation coefficient.
    Notes
    -----
    Return the stated deterministic result to rtol=1e-10 and atol=1e-12.
    The supplied numerical controls define the discrete target.
    """
    return None
```

### Step 3

propagate fragment

Goal
----
Step 3: propagate one fragment to impact, depletion, or breakup.

```python
"""Step 3: propagate one fragment to impact, depletion, or breakup."""
import numpy as np
from scipy.integrate import solve_ivp

def propagate_fragment(fragment, wind_scale, cd_scale, sigma_abl, rtol=2e-10, max_step=0.35, max_depth=2):
    """Propagate a fragment to the first terminal event.

    Parameters
    ----------
    fragment : array_like, shape (9,)
        Finite (state7,strength,depth), positive mass and strength, and
        nonnegative integer depth.
    wind_scale, cd_scale, sigma_abl : float
        Finite scales with domains >=0, >0, >=0, respectively, as in stage 2.
    rtol, max_step : float
        Finite positive RK45 relative tolerance and maximum time step in s.
    max_depth : int
        Finite nonnegative integer, not bool, below which breakup is enabled.

    Returns
    -------
    ndarray, shape (8,)
        Event code (0 ground, 1 depletion, 2 breakup), then terminal state7.

    Raises
    ------
    ValueError
        For invalid shape, nonfinite data or controls, nonpositive mass,
        strength, rtol, max_step or drag scale, negative wind or ablation,
        or a depth outside its integer domain.
    RuntimeError
        If integration fails or no terminal event occurs by 5000 s.
    Notes
    -----
    Finite terminal-state components are compared with absolute tolerance
    5e-3; the event code is exact. The supplied numerical controls define
    the discrete target.
    """
    return None
```

### Step 4

partition fragment cloud

Goal
----
Step 4: partition a disrupted parent and assign transverse kicks.

```python
"""Step 4: partition a disrupted parent and assign transverse kicks."""
import math
import numpy as np

def partition_fragment_cloud(fragment, dust_fraction, largest_fraction, beta, wind_scale, rng=None, alpha=0.16, strength_cap=15000000.0, minimum_mass=0.2, kick_power=2.0, momentum_correct=True):
    """Partition one disrupted parent and assign transverse velocities.

    Parameters
    ----------
    fragment : array_like, shape (9,)
        Finite (state7,strength,depth), positive mass and strength,
        nonzero velocity, and nonnegative integer depth.
    dust_fraction, largest_fraction, beta : float
        Finite fractions in [0,1), (0,1), and (0,1), respectively.
    wind_scale : float
        Finite nonnegative wind multiplier.
    rng : numpy.random.Generator or None
        Advancing uniform stream, default PCG64(741).
    alpha, strength_cap, minimum_mass, kick_power : float
        Finite nonnegative strength exponent, positive cap in Pa, positive
        proposal/residual floor in kg, and nonnegative radius-ratio exponent
        inside the square root of the kick magnitude.
        The first largest child is retained even below minimum_mass.
    momentum_correct : bool
        Whether to remove the resolved-mass-weighted mean kick.

    Returns
    -------
    ndarray, shape (n_children,9)
        Generated-rank-order (state7,child_strength,parent_depth+1) rows.

    Raises
    ------
    ValueError
        For a shape or domain violation above, any nonfinite numeric input,
        zero parent velocity, nonboolean momentum_correct, or rng without
        a uniform method.
    Notes
    -----
    Return the stated deterministic result to rtol=1e-10 and atol=1e-12.
    The supplied numerical controls define the discrete target.
    """
    return None
```

### Step 5

simulate realization

Goal
----
Step 5: propagate one queue-based cascading-fragment realization.

```python
"""Step 5: propagate one queue-based cascading-fragment realization."""
import numpy as np

def simulate_realization(rng=None, kick_power=2.0, momentum_correct=True, cascade=True, rtol=2e-10, max_step=0.35, max_depth=2):
    """Simulate one FIFO fragmentation realization.

    Parameters
    ----------
    rng : numpy.random.Generator or None
        Advancing stream with standard_normal and uniform, default PCG64(761903).
    kick_power : float
        Finite nonnegative radius-ratio exponent inside the kick square root.
    momentum_correct, cascade : bool
        Kick centering and repeated-breakup switches. With cascade=False,
        the first breakup remains enabled when max_depth>0.
    rtol, max_step : float
        Finite positive RK45 controls.
    max_depth : int
        Finite nonnegative breakup depth limit, not bool.

    Returns
    -------
    ndarray, shape (n_impacts,8)
        (east,north,mass,uS,uD,uBeta,uWind,breakups) in retained FIFO order.

    Raises
    ------
    ValueError
        For nonfinite or out-of-domain controls, nonboolean switches, or
        rng lacking either random-number method.
    RuntimeError
        If propagation fails or the realization has no retained impacts.
    Notes
    -----
    Comparison fixtures round east/north to 0.1 m and mass to 1e-6 kg;
    array shape, FIFO row order, nuisance values, and discrete breakup count
    remain exact. The supplied numerical controls define the discrete target.
    """
    return None
```

### Step 6

build impact cloud

Goal
----
Step 6: assemble the proposal cloud and retain realization metadata.

```python
"""Step 6: assemble the proposal cloud and retain realization metadata."""
import numpy as np

def build_impact_cloud(seed=761903, runs=8, kick_power=2.0, momentum_correct=True, cascade=True, rtol=2e-10, max_step=0.35, max_depth=2):
    """Assemble independent entry realizations from one advancing stream.

    Parameters
    ----------
    seed, runs : int
        Finite integers, not bool. seed>=0 and 1<=runs<=1000.
    kick_power : float
        Finite nonnegative radius-ratio exponent inside the kick square root.
    momentum_correct, cascade : bool
        Kick centering and repeated-breakup switches.
    rtol, max_step : float
        Finite positive RK45 controls.
    max_depth : int
        Finite nonnegative breakup depth limit, not bool.

    Returns
    -------
    ndarray, shape (n_impacts,9)
        (east,north,mass,run,uS,uD,uBeta,uWind,breakups), zero-based run labels.

    Raises
    ------
    ValueError
        For any nonfinite or out-of-domain numeric control or nonboolean switch.
    RuntimeError
        If a realization fails or has no retained impacts.
    Notes
    -----
    Differential comparisons use absolute tolerances of 5e-3 m for east and
    north and 5e-6 kg for mass. Row count, row order, run labels, nuisance
    values, and breakup counts are exact.
    """
    return None
```

### Step 7

compute population weights

Goal
----
Step 7: importance-reweight the proposal cloud for three populations.

```python
"""Step 7: importance-reweight the proposal cloud for three populations."""
import math
import numpy as np
from scipy.special import betaln, logsumexp

def compute_population_weights(impact_cloud, balance_realizations=False):
    """Compute three normalized per-impact population-weight rows.

    Parameters
    ----------
    impact_cloud : array_like, shape (n,p), n>=1, p>=8
        Finite rows with nonnegative integer run labels in column 3 and
        nuisance coordinates strictly in (0,1) in columns 4:8.
    balance_realizations : bool
        A Python or NumPy boolean. If True, divide each run's row weights
        by its descendant count.

    Returns
    -------
    ndarray, shape (3,n)
        Positive normalized weights for K=0,1,2 in input impact order.

    Raises
    ------
    ValueError
        For malformed or nonfinite cloud, invalid run labels, or nuisance
        coordinates outside the open unit interval, or a nonboolean switch.
    RuntimeError
        If normalized weights underflow or become nonfinite.
    Notes
    -----
    Return the stated deterministic result to rtol=1e-10 and atol=1e-12.
    The supplied numerical controls define the discrete target.
    """
    return None
```

### Step 8

fit ground gmm

Goal
----
Step 8: fit a weighted three-dimensional mass-location Gaussian mixture.

```python
"""Step 8: fit a weighted three-dimensional mass-location Gaussian mixture."""
import math
import numpy as np
from scipy.special import logsumexp

def fit_ground_gmm(impact_cloud, sample_weights, components=4, iterations=120, regularization=(0.0064, 0.01, 0.0049)):
    """Fit the deterministic weighted full-covariance Gaussian mixture.

    Parameters
    ----------
    impact_cloud : array_like, shape (n,p), n>=1, p>=3
        Finite rows starting with east, north, and positive mass.
    sample_weights : array_like, shape (n,)
        Finite nonnegative weights with positive total and positive weight
        in every deterministic initialization block.
    components, iterations : int
        Finite integers, not bool, with 1<=components<=n and iterations>=1.
    regularization : array_like, shape (3,)
        Finite positive diagonal covariance additions in standardized units.

    Returns
    -------
    ndarray, shape (13*components,)
        All weights, all means (component-major), then all covariances
        (component-major, row-major), preserving initialization order.

    Raises
    ------
    ValueError
        For any shape, finiteness, mass, weight, initialization-block, or
        numeric-control domain violation above.
    RuntimeError
        If a component becomes empty or a covariance is not positive definite.
    Notes
    -----
    Return the stated deterministic result to rtol=1e-10 and atol=1e-12.
    The supplied numerical controls define the discrete target.
    """
    return None
```

### Step 9

infer campaign recovery

Goal
----
Step 9: infer second-campaign recovery from a mixed first catalog.

```python
"""Step 9: infer second-campaign recovery from a mixed first catalog."""
import math
import numpy as np
from scipy.special import expit, gammaln, logsumexp, roots_jacobi, betaln

def infer_campaign_recovery(population_models, inventory, scenario_priors, observations=((196900.0, 4550.0, 0.095), (199650.0, 5250.0, 0.022), (198150.0, 5050.0, 0.058)), observation_covariance=((0.0875 ** 2, 0.25 * 0.0875 * 0.125, 0.0), (0.25 * 0.0875 * 0.125, 0.125 ** 2, 0.0), (0.0, 0.0, 0.18 ** 2)), calibration=((13, 2), (17, 5), (11, 3), (19, 6)), beta_prior=((2.3, 5.7), (4.1, 10.3), (1.2, 3.8)), population_prior=(0.3, 0.45, 0.25), survey_response=(0.06, 0.75, 0.035, 0.9), ellipse_center=(198500.0, 4850.0), ellipse_axes=(2600.0, 1300.0), ellipse_angle_degrees=22.0, mass_bounds=(0.025, 0.18), hermite_nodes=64, radial_nodes=56, angular_nodes=88, mass_nodes=48, efficiency_nodes=64, background_model=(0.1, -0.15, 0.05, 0.7, 0.08, 0.12, 0.08, 0.5, -0.06, 0.12, -0.06, 0.9), background_prior=(2.4, 1.6), background_control=(4, 3.0), background_scale=1.0):
    """Infer new-fall recovery from a complete catalog with unknown fall associations.

Parameters
----------
population_models : array_like, shape (R,13*C)
    The column count must be a positive multiple of 13; set
    C=population_models.shape[1]//13. Each row stores weights in columns
    [0:C], row-major standardized means in [C:4*C] reshaped to (C,3),
    and row-major 3x3 covariances in [4*C:13*C] reshaped to (C,3,3).
    R,C>=1; weights are positive and sum to one within absolute 1e-12,
    and covariances are symmetric positive-definite with symmetry
    tolerance 1e-12.
inventory : array_like, shape (R,)
    Finite nonboolean integer N_r>=0. Counts smaller than the catalog are
    valid because other detections can be accumulated meteorites.
scenario_priors : array_like, shape (H,R)
    H>=1 positive probabilities, each row sums to one within 1e-12.
observations : array_like, shape (n,3)
    Complete merged campaign-A catalog, 0<=n<=12, meters east/north and
    positive kilograms. Empty (0,3) valid. A value-independent random
    permutation orders the union of new-fall and accumulated detections.
observation_covariance : array_like, shape (3,3)
    New-fall Gaussian reporting covariance in standardized coordinates.
    Symmetry tolerance 1e-12; symmetrize, reject eigenvalues below -1e-12,
    set other negative eigenvalues to zero. Zero covariance is valid.
calibration : array_like, shape (J,2)
    J>=1 independent binomial (trials,successes), nonboolean integers,
    trials>=1, 0<=successes<=trials. Response exactly one. Calibration
    analogs share the unknown new-fall visibility p, not the background rate.
beta_prior : array_like, shape (H,2) or (2,)
    Positive finite Beta shapes for p conditional on population K; a single
    pair broadcasts over H. Population prior precedes calibration, so its
    normalized marginal likelihood updates the population weights too.
population_prior : array_like, shape (H,)
    Positive pre-calibration probabilities summing to one within 1e-12.
survey_response : array_like, shape (4,)
    Positive finite (m_A,w_A,m_B,w_B), in kg and natural-log-mass width.
    New-fall Bernoulli probability p*expit(log(m/m_j)/w_j); A covers all
    space and positive mass, B the target. Trials independent conditional
    on true mark and p. Only actual new-fall A detections deplete N_r.
ellipse_center, ellipse_axes : array_like, shape (2,)
    Finite center in meters and positive semiaxes in meters.
ellipse_angle_degrees : float
    Finite counterclockwise angle from east to the first semiaxis.
mass_bounds : array_like, shape (2,)
    Positive strictly increasing closed true target mass interval in kg.
hermite_nodes, radial_nodes, angular_nodes, mass_nodes, efficiency_nodes : int
    Nonboolean finite integer orders >=2. Hermite exp(-x*x); Legendre
    radius/angle/log-mass; Jacobi efficiency weight includes k associated
    detections for each latent subset, not the full catalog size n.
background_model : array_like, shape (12,)
    Accumulated-meteorite reported-mark density in the SAME standardized
    coordinates: mean followed by row-major symmetric positive-definite
    3x3 covariance. Symmetry tolerance 1e-12. Already includes reporting
    error and detection selection; do not convolve or select it again.
background_prior : array_like, shape (2,)
    Positive Gamma shape/rate (u0,v0) for accumulated detection intensity
    lambda per unit exposure; independent of K, r, p and new-fall marks.
background_control : array_like, shape (2,)
    Finite (count,exposure), nonboolean integer count>=0, exposure>0,
    neither entry boolean. Count is Poisson(exposure*lambda), in an
    independent accumulated-only control survey with the same rate.
background_scale : float
    Finite nonnegative campaign-A exposure relative to the control unit.
    Background A count is Poisson(background_scale*lambda). Zero disables
    background, in which case at least one N_r must be >=n.

Returns
-------
ndarray, shape (1+2*H+n,)
    [P_search, posterior_K, association_i, population_recovery_K].
    Association is probability catalog item i belongs to the witnessed
    fall. P_search concerns at least one NEW witnessed-fall recovery in B,
    never an accumulated meteorite. Population recovery is the joint
    probability of population K AND at least one new recovery; these H
    entries sum to P_search. All conditioning includes both
    calibration data sets and the complete merged count and marks.

Raises
------
ValueError
    For nonfinite inputs or stated shape/domain violations.
RuntimeError
    If quadrature or floating arithmetic yields nonfinite evidence or
    invalid posterior probabilities.

Notes
-----
The H populations label physical and visibility distributions; r and p are
shared by all witnessed-fall objects. Conditional on r, its N_r latent
marks are iid from the continuous GMM. Campaign B searches the same true
marks remaining after campaign A. No absolute log evidence is returned.
The runtime uses NumPy 2.x, which has no ``np.math`` namespace; use the
imported standard-library ``math`` module or the supplied SciPy special functions.
Return deterministic rtol=1e-10, atol=1e-12 agreement; these code-test
tolerances differ from reasoning tolerances.
"""
    return None
```

### Step 10

solve strewn probability

Goal
----
Step 10: compose the complete posterior predictive recovery calculation.

```python
"""Step 10: compose the complete posterior predictive recovery calculation."""
import numpy as np

def solve_strewn_probability(seed=761903, runs=8, kick_power=2.0, momentum_correct=True, cascade=True, rtol=2e-10, max_step=0.35, max_depth=2, components=4, iterations=120, regularization=(0.0064, 0.01, 0.0049), balance_realizations=True, survey_response=(0.06, 0.75, 0.035, 0.9), calibration=((13, 2), (17, 5), (11, 3), (19, 6)), hermite_nodes=64, radial_nodes=56, angular_nodes=88, mass_nodes=48, efficiency_nodes=64, beta_prior=((2.3, 5.7), (4.1, 10.3), (1.2, 3.8)), background_model=(0.1, -0.15, 0.05, 0.7, 0.08, 0.12, 0.08, 0.5, -0.06, 0.12, -0.06, 0.9), background_prior=(2.4, 1.6), background_control=(4, 3.0), background_scale=1.0):
    """Compose the complete finite-inventory posterior predictive probability.

    Parameters
    ----------
    seed, runs : int
        Nonboolean finite integers, seed>=0 and 1<=runs<=1000.
    kick_power : float
        Finite nonnegative radius-ratio exponent inside the kick square root.
    momentum_correct, cascade : bool
        Resolved momentum centering and repeated-breakup switches.
    rtol, max_step : float
        Positive finite RK45 controls.
    max_depth : int
        Nonboolean finite nonnegative breakup depth limit.
    components, iterations : int
        Nonboolean integers, components>=1 and no greater than the number
        of retained impacts in any scenario; iterations>=1.
    regularization : array_like, shape (3,)
        Positive finite standardized covariance diagonal additions.
    balance_realizations : bool
        True supplies equal base proposal probability per event scenario;
        False supplies a survivor-count-weighted base proposal. Stage 7's
        normalized row weights are summed within each scenario to form its
        population-conditional prior. Neither choice reweights marks within
        an individual scenario, whose GMM uses uniform fragment weights.
    survey_response : array_like, shape (4,)
        Positive finite first/second half-response masses and log widths,
        with exactly the meaning specified in infer_campaign_recovery.
    calibration : array_like, shape (J,2)
        Integer binomial trial/success batches as in infer_campaign_recovery.
    hermite_nodes, radial_nodes, angular_nodes, mass_nodes, efficiency_nodes : int
        Nonboolean finite quadrature orders >=2 with the stage 9 meanings.

    beta_prior, background_model, background_prior, background_control, background_scale
        Population-linked visibility and accumulated-meteorite nuisance
        parameters with exactly the domains and meanings of stage 9.

    Returns
    -------
    float
        Probability of at least one new second-campaign recovery, using the
        fixed catalog, priors, measurement law, and target in the statement.

    Raises
    ------
    ValueError
        For domain violations specified in the consuming stages, including
        an inventory smaller than the component count or impossible zero-background catalog.
    RuntimeError
        For predecessor propagation, covariance, or probability failures.

    Notes
    -----
    The default composes the physical scenarios and both specified campaigns.
    Differential probability comparisons use absolute tolerance 1e-7.
    """
    return None
```
