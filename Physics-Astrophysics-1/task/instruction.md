# Physics-Astrophysics-1

## Background

Gravitational-wave astronomy uses populations of compact-binary mergers to study how stellar remnants form and evolve across cosmic history. Binary black holes can emerge through stellar binary evolution or be assembled through gravitational interactions in dense environments. Their merger rates therefore contain information about stellar birth, the production of black holes, and the processes that bring pairs into sufficiently tight orbits. Understanding how those rates change with cosmic epoch helps distinguish competing accounts of the origin of merging binaries.

Progenitor formation and merger generally occur at different cosmic times. The intervening delay includes stellar evolution, possible dynamical interactions, and orbital decay driven by gravitational radiation. Different binary properties and formation environments produce different distributions of these delays. Consequently, the mergers occurring at one epoch receive contributions from systems formed at many earlier epochs. A broad delay distribution smooths temporal structure in the underlying formation history, making the relation between stellar birth and observed mergers indirect.

Observed event counts must also be translated into an intrinsic population rate. Detector sensitivity varies with source properties and distance, so population inference accounts for selection effects and measurement uncertainty. Cosmology relates redshift to cosmic time and volume, while the distinction between source time and observer time matters when defining an event rate. These ingredients allow rate histories to be compared consistently, although their reconstruction becomes less constrained where observations provide limited coverage.

Recovering a formation history from a merger history is an inverse problem conditional on the assumed delay model. Smoothing can erase information, and inversion can amplify uncertainty, motivating regularization and physical consistency checks. An event rate must be nonnegative, but a negative reconstructed value must be interpreted alongside the chosen inverse prescription, observational uncertainty, and assumptions outside the measured range. Such consistency tests connect population observations to models of binary evolution while making the limitations of the inference explicit.

## Problem

Delay-time models for binary black holes can be constrained by requiring the formation history inferred from their merger rate to remain nonnegative. Use the regularized deconvolution approach developed for this test in a recent GWTC-4.0 analysis to constrain the shallow component of a two-population delay distribution.

Take three deterministic rate scenarios from the median and lower and upper 90% credible endpoints of the September 2025 GWTC-4.0 population analysis's Power Law Redshift estimate at $$z_p=0.2$$. Use the following finite-grid benchmark, with rates defined per comoving volume and source-frame year.

| Quantity | Benchmark convention |
| --- | --- |
| Cosmology | Flat $$\Lambda$$CDM, $$H_0=67.7\ \mathrm{km\,s^{-1}\,Mpc^{-1}}$$, $$\Omega_m=0.31$$, no radiation; inverse-Hubble-unit conversion $$977.79222168\ \mathrm{Gyr}$$ |
| Sampling | $$4096$$ uniform cosmic times over $$[0,t_0]$$, both endpoints included; $$t_0$$ is the present cosmic age, the age at $$z=0$$; the cap at redshift $$60$$ applies only to the sampled redshifts |
| Untapered merger history | $$R(z)=R_p[(1+z)/1.2]^{3.2}$$ for each retrieved pivot rate $$R_p$$ |
| Boundary conditioning | Multiplicative taper to zero over $$0\le z\le0.1$$; blend to the same absolute $$300\ \mathrm{Gpc^{-3}\,yr^{-1}}$$ over $$1.5\le z\le1.55$$, constant above $$1.55$$; both use $$w(x)=[1-\cos(\pi x)]/2$$ across a unit interval |
| Delay components | Continuous unit-integral power laws, steep $$p_s\propto\tau^{-1.73}$$ and shallow $$p_l\propto\tau^{-0.99}$$, on $$[0.01\ \mathrm{Gyr},t_0]$$, zero outside |
| Discrete inversion | Kernel weights $$p\Delta t$$; transform length $$2N$$; first $$N$$ inverse samples retained, where $$N=4096$$ |
| Physicality window | Every sampled point with $$0.1\le z\le1.5$$ |

For a fraction $$f$$ of systems in the shallow component, infer the formation history using the mixture delay law and the formation-history analysis's relative-peak-power Wiener prescription. Find the largest common fraction $$f_*\in[0,1]$$ for which all three histories are nonnegative in the window, verifying that the compatible fractions connect to zero.

Include in the reasoning the retrieved rate and credible offsets, both component densities at $$1\ \mathrm{Gyr}$$, a diagnostic window minimum at $$f=0.5$$ obtained by replacing the pivot rate with $$250\ \mathrm{Gpc^{-3}\,yr^{-1}}$$ and leaving every other benchmark convention unchanged, the lower-rate scenario's minimum formation rate at $$f=0.5$$, the three individual critical fractions, and the median- and upper-rate scenarios' minimum formation rates at the common bound.

Report the common fraction to an absolute accuracy of $$10^{-6}$$, and give each diagnostic listed above to at least six significant figures.

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

compute_cosmic_age

Goal
----
Cosmic age at a given redshift

```python
import numpy as np

def compute_cosmic_age(z: float) -> float:
    '''Age of the universe at redshift z for flat LCDM.

    Parameters
    ----------
    z : float
        Redshift, must be a finite scalar with z >= 0.

    Returns
    -------
    t : float
        Cosmic age t(z) in Gyr for H0 = 67.7 km/s/Mpc, Omega_m = 0.31,
        flat geometry, radiation neglected.

    Raises
    ------
    ValueError
        If z is not a finite scalar, or z < 0.
    '''
    return None  # placeholder
```

### Step 2

build_time_redshift_grid

Goal
----
Uniform cosmic time grid and its redshift mapping

```python
import numpy as np

def build_time_redshift_grid(n_samples: int) -> np.ndarray:
    '''Uniform cosmic time grid on [0, t0] with its redshift mapping.

    Parameters
    ----------
    n_samples : int
        Number of uniformly spaced cosmic time samples, must be an
        integer >= 16.

    Returns
    -------
    grid : np.ndarray
        Array of shape (2, n_samples): row 0 holds the cosmic time samples
        t in Gyr, uniformly spaced from 0 to t0 inclusive; row 1 holds the
        corresponding redshifts z(t), capped at z = 60.

    Raises
    ------
    ValueError
        If n_samples is not an integer (booleans and floats are rejected),
        or n_samples < 16.
    '''
    return None  # placeholder
```

### Step 3

affine_merger_family

Goal
----
Affine uncertainty in a conditioned merger rate history

```python
import numpy as np

def affine_merger_family(z_grid: np.ndarray, pivot_rate: float, r_asym: float,
                         amplitudes: np.ndarray, kappa: float = 3.2) -> np.ndarray:
    r'''Construct the central history and three affine uncertainty modes.

    Parameters
    ----------
    z_grid : np.ndarray
        Nonempty finite real one-dimensional redshift grid, all entries >= 0.
    pivot_rate : float
        Finite positive central rate at z=0.2, in Gpc^-3 yr^-1.
    r_asym : float
        Finite nonnegative common high-redshift boundary, in Gpc^-3 yr^-1.
    amplitudes : np.ndarray
        Finite nonnegative real shape (3,), with sum strictly below one. The three dimensionless amplitudes scale the normalization, lower-redshift feature and higher-redshift feature, in that order. These bounds ensure a nonnegative merger history for every coefficient in [-1,1]^3.
    kappa : float
        Finite redshift exponent, default 3.2.

    Returns
    -------
    family : np.ndarray
        Finite shape (4,N), in Gpc^-3 yr^-1. Row 0 is the central history; rows 1–3 multiply the three latent coefficients. These are affine coefficients, not derivative or Taylor coefficients.

    Raises
    ------
    ValueError
        If the finite real grid, scalar, shape or amplitude-domain contracts fail, or if the evaluated family is nonfinite.
    '''
    return None
```

### Step 4

delay_time_distribution

Goal
----
Continuous delay-density normalization for population mixtures

```python
import numpy as np

def delay_time_distribution(tau_grid: np.ndarray, alpha: float,
                            tau_min: float, tau_max: float) -> np.ndarray:
    r'''Sample a continuously normalized power-law delay density.

    Parameters
    ----------
    tau_grid : np.ndarray
        Nonempty one-dimensional finite real delay times >= 0, in Gyr.
    alpha : float
        Finite real exponent, including -1 and its neighborhood.
    tau_min, tau_max : float
        Finite real support limits in Gyr satisfying 0 < tau_min < tau_max.

    Returns
    -------
    density : np.ndarray
        Finite shape (N,), in Gyr^-1. Inclusive support, zero outside. Continuous normalization is preserved without rescaling the sampled sum.

    Raises
    ------
    ValueError
        If the input domains fail or the density cannot be evaluated finitely.
    '''
    return None
```

### Step 5

wiener_formation_jet

Goal
----
Formation histories and their response to the delay-mixture fraction

```python
import numpy as np

def wiener_formation_jet(merger_family: np.ndarray, steep_density: np.ndarray,
                         shallow_density: np.ndarray, fraction: float,
                         dt: float) -> np.ndarray:
    r'''Evaluate the regularized formation family and its local fraction response.

    Parameters
    ----------
    merger_family : np.ndarray
        Finite real shape (4,N), N >= 2. Row 0 is the central merger history;
        rows 1–3 multiply the three latent rate coefficients. Units are
        Gpc^-3 yr^-1, and signed entries are allowed.
    steep_density, shallow_density : np.ndarray
        Finite real shape (N,), nonnegative with positive sum, in Gyr^-1.
        Preserve the supplied sampled amplitudes; both component densities
        have already been continuously normalized.
    fraction : float
        Finite real mixture fraction in [0,1].
    dt : float
        Finite positive sample spacing in Gyr.

    Returns
    -------
    jet : np.ndarray
        Finite shape (3,4,N). Leading rows are value, first derivative and
        second derivative with respect to fraction, in that order. These
        are ordinary derivatives, without factorial rescaling. All four
        affine rows use the same fraction-dependent inverse and penalty.

    Raises
    ------
    ValueError
        If the finite real input, array-shape, nonnegative density, fraction
        or spacing contracts fail, or the spectral result is nonfinite.
    '''
    return None
```

### Step 6

physicality_vertex_jet

Goal
----
Motion of the coefficient region allowed by formation physicality

```python
import numpy as np

def physicality_vertex_jet(formation_family_jet: np.ndarray,
                           window_mask: np.ndarray, lower: np.ndarray,
                           upper: np.ndarray) -> np.ndarray:
    r'''Track the vertices of a smooth physicality-region branch.

    Parameters
    ----------
    formation_family_jet : np.ndarray
        Finite real shape (3,4,N), N >= 1. Axis 0 gives true derivative orders zero, one and two with respect to the same dimensionless perturbation. Within each order, row 0 is the constant formation coefficient and rows 1–3 multiply the latent coordinates; all coefficients have rate units.
    window_mask : np.ndarray
        Boolean shape (N,), selecting at least one inequality that must remain nonnegative.
    lower, upper : np.ndarray
        Fixed finite real box bounds of shape (3,), with lower < upper coordinatewise.

    Returns
    -------
    vertex_jet : np.ndarray
        Finite shape (3,V,3), containing distinct extreme vertices with true derivative orders on axis 0. Vertex order is unrestricted, but the same vertex must occupy each column of all three derivative orders. A locally persistent empty or zero-volume intersection returns shape (3,0,3).

    Raises
    ------
    ValueError
        If finite real shapes, the nonempty boolean window or ordered box contracts fail, or numerical geometry or linear systems cannot produce finite vertex jets. Supported moving regions have simple vertices and locally fixed face incidence; topology changes and non-simple moving vertices are outside this input domain. Fully static input jets may describe non-simple vertices. These local geometric preconditions need not be numerically certified.
    '''
    return None
```

### Step 7

gaussian_polytope_moment_jet

Goal
----
Gaussian probability and moment response of a moving compatible region

```python
import numpy as np

def gaussian_polytope_moment_jet(vertex_jet: np.ndarray, mean: np.ndarray,
                                covariance: np.ndarray,
                                tol: float = 1e-8) -> np.ndarray:
    r'''Integrate the response of Gaussian raw moments to region motion.

    Parameters
    ----------
    vertex_jet : np.ndarray
        Finite real shape (3,V,3). Axis 0 contains positions and true first and second derivatives, with matched vertex identities across orders. The base region is a convex hull; vertex order is unrestricted. Jets describe a locally fixed face-incidence branch whose facets remain planar through second order. A locally persistent empty or zero-volume region returns zeros.
    mean : np.ndarray
        Fixed finite real Gaussian mean of shape (3,), in the original latent coordinates.
    covariance : np.ndarray
        Fixed finite real covariance of shape (3,3), symmetric within absolute tolerance 1e-12 and positive definite, with spectral condition number <= 1e6. Base vertices must lie within Mahalanobis distance 12 of the mean; these bounds define the resolved numerical domain.
    tol : float
        Finite accuracy target in [1e-10,1e-5], default 1e-8. Each returned entry should have absolute error <= tol times one plus its magnitude.

    Returns
    -------
    moment_jet : np.ndarray
        Finite shape (3,10), with true derivative orders on axis 0. Columns are integrals of [1,u1,u2,u3,u1^2,u1*u2,u1*u3,u2^2,u2*u3,u3^2] against the fixed Gaussian density. No factorial division or probability normalization is applied. A locally persistent zero-volume region gives thirty zeros.

    Raises
    ------
    ValueError
        If finite real shapes, the resolved covariance domain or tolerance contracts fail, or numerical geometry or integration cannot produce finite responses at the requested accuracy. The fixed-incidence and facet-planarity conditions are input preconditions and need not be numerically certified.
    '''
    return None
```

### Step 8

condition_gaussian_mixture_jet

Goal
----
Response of the physicality-conditioned coefficient posterior

```python
import numpy as np

def condition_gaussian_mixture_jet(accepted_moment_jets: np.ndarray,
                                   prior_moments: np.ndarray,
                                   weights: np.ndarray) -> np.ndarray:
    r'''Reweight a Gaussian mixture and propagate its local fraction response.

    Parameters
    ----------
    accepted_moment_jets : np.ndarray
        Finite real shape (G,3,10), G >= 1. The middle axis contains value,
        first derivative and second ordinary fraction derivative. The last
        axis orders the raw integrals of [1,u1,u2,u3,u1^2,u1*u2,u1*u3,
        u2^2,u2*u3,u3^2]. Component masses at order zero are nonnegative.
    prior_moments : np.ndarray
        Finite real shape (G,10), in the same moment order, for each
        untruncated Gaussian integrated over the common fixed box.
        These moments are constant with respect to fraction. Inputs are
        consistent with a smooth family whose accepted region is contained
        in the fixed box at each fraction; containment need not be
        numerically certified. A zero accepted mass is locally
        zero, with vanishing response.
    weights : np.ndarray
        Finite real nonnegative shape (G,), with positive total. These
        are weights before global box truncation. Their sum need not be one.

    Returns
    -------
    summary_jet : np.ndarray
        Finite shape (3,10): rows give value and first/second ordinary
        fraction derivatives. Columns contain the physicality probability
        conditional on the box, the three conditional means, and centered
        covariance entries [11,12,13,22,23,33]. Locally zero accepted mass
        returns all zeros; the mean/covariance entries are then placeholders.

    Raises
    ------
    ValueError
        If finite real shapes, nonnegative order-zero masses or weight
        contracts fail, the weighted prior mass is nonpositive, or the
        conditional response is nonfinite.
    '''
    return None
```

### Step 9

reconvolution_error

Goal
----
Reconvolution validity of a regularized formation history

```python
import numpy as np

def reconvolution_error(r_form: np.ndarray, p_tau: np.ndarray,
                        r_merge: np.ndarray, dt: float,
                        window_mask: np.ndarray) -> float:
    r'''Measure how well a reconstruction reproduces its own merger history.

    Parameters
    ----------
    r_form : np.ndarray
        Finite real one-dimensional reconstructed formation history, N >= 2 samples, in Gpc^-3 yr^-1. Signed entries are allowed, because a regularized inverse may undershoot.
    p_tau : np.ndarray
        Finite real one-dimensional delay density on the same uniform grid, shape (N,), in Gyr^-1. The supplied sampled amplitudes are used as given.
    r_merge : np.ndarray
        Finite real one-dimensional merger history on the same grid, shape (N,), in Gpc^-3 yr^-1, nonzero on every selected sample.
    dt : float
        Finite positive sample spacing in Gyr, supplying the kernel quadrature weight.
    window_mask : np.ndarray
        Boolean shape (N,) comparison window selecting at least one sample.

    Returns
    -------
    err : float
        Finite native Python float, the maximum over the selected samples of the absolute reconvolution residual divided by the absolute merger history. Zero-padding to length 2N and retention of the first N samples make the comparison acyclic on the supplied grid.

    Raises
    ------
    ValueError
        If the finite real input, array-shape, boolean-mask, spacing or nonzero-denominator contracts fail, if the mask selects no sample, or if the diagnostic is nonfinite.
    '''
    return None
```

### Step 10

run_formation_response_pipeline

Goal
----
Response of the inferred delay-mixture bound to physicality confidence

```python
import numpy as np

def run_formation_response_pipeline(pivot_rates: np.ndarray, r_asym: float,
                                       n_samples: int, amplitudes: np.ndarray,
                                       weights: np.ndarray, means: np.ndarray,
                                       covariances: np.ndarray,
                                       required_probability: float = 0.5,
                                       alpha_steep: float = -1.73,
                                       alpha_shallow: float = -0.99,
                                       kappa: float = 3.2,
                                       max_reconvolution_error: float = 0.1) -> np.ndarray:
    r'''Find a physicality bound and its response to the required probability.

    Parameters
    ----------
    pivot_rates : np.ndarray
        Nonempty finite real positive one-dimensional array of central rates at z=0.2, in Gpc^-3 yr^-1. All scenarios share the same latent coefficient draw; the event is simultaneous physicality of every scenario.
    r_asym : float
        Finite nonnegative absolute high-redshift boundary, in Gpc^-3 yr^-1, shared by all scenarios.
    n_samples : int
        Integer >= 16 for the preceding inclusive cosmic-time grid; the inclusive 0.1 <= z <= 1.5 window must contain a sample.
    amplitudes : np.ndarray
        Finite nonnegative shape (3,), sum < 1, used by the affine merger-family step. Zero amplitudes recover the deterministic common physicality bound.
    weights : np.ndarray
        Finite nonnegative shape (G,), G >= 1, with positive total. Gaussian-mixture weights before global truncation on the common box [-1,1]^3.
    means : np.ndarray
        Finite real shape (G,3), each coordinate in [-1,1]. These synthetic coefficient-posterior parameters are supplied inputs, not catalog results.
    covariances : np.ndarray
        Finite real shape (G,3,3), each symmetric within absolute tolerance 1e-12 with eigenvalues in [0.1,2]. This resolved posterior domain keeps the box probability and conditional moments numerically well conditioned.
    required_probability : float
        Finite probability threshold in [0.05,0.95], default 0.5. The joint acceptance probability as a function of shallow fraction is nonincreasing on [0,1], and at zero fraction strictly exceeds this threshold. These are input preconditions, not requested numerical certifications. For nonzero amplitudes, either fraction one exceeds the threshold in a neighborhood of q, or there is a unique interior root with strictly negative first probability derivative and locally fixed full-dimensional polytope incidences, with simple moving vertices. This smoothness precondition applies at the critical fraction; the region may change topology elsewhere. Zero amplitudes are also supported: the deterministic cutoff is independent of q in (0,1).
    alpha_steep, alpha_shallow : float
        Finite real delay exponents with alpha_steep < alpha_shallow, continuously normalized separately on [0.01 Gyr,t0].
    kappa : float
        Finite real central redshift exponent, default 3.2.
    max_reconvolution_error : float
        Finite positive bound on the reconvolution validity diagnostic, default 0.1. The regularized inverse is certified at both mixture endpoints of every scenario before any inference, so a sampling too coarse to resolve the delay kernel is rejected rather than reported.

    Returns
    -------
    diagnostics : np.ndarray
        Finite shape (3,11). Row 0 contains the critical shallow fraction, its joint physicality probability conditional on the box, three conditional means, and centered covariance entries [11,12,13,22,23,33]. Rows 1 and 2 contain the first and second ordinary derivatives of all these quantities with respect to required_probability q, holding every other argument fixed. Derivatives refer to the mathematical critical-fraction curve, not to the discrete iterations of a numerical solver. Select fraction one when it remains feasible locally in q. Its response rows, and those in the zero-amplitude deterministic case, are zero because these outputs are then locally independent of q. The fraction has absolute accuracy 1e-8; each other reported quantity has error at most 2e-5 times one plus its magnitude. At a deterministic cutoff use its feasible-side limit so the conditional moments are defined. Any sufficiently accurate deterministic numerical method is acceptable.

    Raises
    ------
    ValueError
        If the finite real input, shape, physical parameter or resolved posterior contracts fail; if the grid window is empty; if any scenario fails the endpoint reconvolution certification; if the all-steep endpoint does not exceed the required probability; if the result is nonfinite; or if a preceding function rejects its inputs. The monotonicity and local-incidence preconditions need not be certified.
    '''
    return None
```
