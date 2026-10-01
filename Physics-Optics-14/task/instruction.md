# Physics-Optics-14

## Background

All lengths are in micrometres and all phases are in radians. Source cells (a,b) with a,b=0,...,15 have centers (0.25(b-7.5),0.25(a-7.5)), area A_s=0.0625 and flattened index n=16a+b. Observation labels (u,v) range over {-3,...,3}^2 in row-major order (v outer), with centers (0.5u,0.5v) and area A_p=0.25. The bright set B consists of points satisfying |u|<=2, |v|<=2 and (u<0 or v<0), together with (u,v)=(2,2); the target T is its 0/1 indicator. The incident field amplitude is A_n=exp(-((x_n-0.11)^2+(y_n+0.07)^2)/2.2^2), and each candidate starts at phi_n=0.35 sin(0.71n)+0.17 cos(0.33n). The wavelength is 0.633, the design distance is 4.0, and certification planes are (3.95,4.0,4.05).

The specified cell-center optical model is E_m=sum_n H_mn A_n exp(i phi_n), with
\[
H_{mn}=\frac{A_s z(1-ikR_{mn})e^{ikR_{mn}}}{2\pi R_{mn}^3},\qquad k=2\pi/\lambda,\quad R_{mn}^2=|\boldsymbol r_m-\boldsymbol r_n|^2+z^2.
\]
Intensities are |E_m|^2 in incident-peak-intensity units, and P_in=A_s sum_n A_n^2. The image metrics are mu=mean_{m in B} I_m, RMSE=sqrt(mean_m (I_m/mu-T_m)^2), SD=sqrt(mean_m (I_m-mu)^2), and eta=A_p sum_{m in B} I_m/P_in. Both means in RMSE and SD range over all 49 observations; mu uses B. A zero norm uses its zero subgradient.

Candidate rows (w_RMSE,w_SD,w_Eff), in order, are (1,0.12,1), (1,0.35,1), (1,1,0.1), (1,0.7,0.65), (1,0.3,0.2), (1,0,1). At one-based update i, s_i=(1+tanh((i-175)/40))/2 and L_i=s_i w_RMSE RMSE+(1-s_i)w_SD SD+(1-s_i)w_Eff(1-eta). Its gradient is evaluated at the current phase before update i. Adam uses learning rate 0.005, beta1=0.9, beta2=0.999, epsilon=1e-8 outside the square root, bias correction at i, fresh zero accumulators for each candidate, and simultaneous unwrapped phase updates.

Manufactured phases use the nearest of 128 evenly spaced values on [0,2pi); a half-level tie selects the increasing phase, modulo 128. With q=(phi mod 2pi)/(2pi/128), values within four binary64 ulps of floor(q)+1/2 use the half-level tie convention. The independent rotation error at cell n is Normal(0,sigma_n^2), with sigma_n=0.044+0.014(1+sin(0.27n)). The cross-polarized geometric phase is twice the nanobrick rotation angle. For certification, the real and imaginary field at each observation follow the bivariate Gaussian having the exact local first and second moments of that independently perturbed phasor sum. Singular covariance laws use their limiting probability distribution. Reliability is a per-location marginal requirement at each plane; exposure endpoints denote the infimum and supremum of the admissible interval. The normalized exposure t has threshold tI=1, cap t<=40, and minimum ensemble-mean diffraction efficiency 0.22 at every plane. The constructed geometry, errors and exposure certificate are benchmark extensions; the Gaussian closure is the specified surrogate model. Probability inversion must converge to absolute CDF error <=1e-7 for nonsingular laws.

Fabrication log: sapphire support, nominal nanobrick height 0.300, inspection microscope 63x. The supplied coordinates refer directly to the mask and resist planes.

## Problem

Certify the largest logarithmic exposure latitude of the six phase-only holographic designs specified below using the cited study’s phase-probability-shaping objective and its local bivariate-Gaussian field closure.
Each design receives 600 independent-pixel Adam updates at the design plane, followed by the prescribed 128-level fabrication and independent nanobrick rotation errors.
At every certification plane, each bright observation must exceed unit exposure dose with probability at least 0.95, while each dark observation must stay below it with probability at least 0.95; the dose cap and minimum diffraction efficiency apply to the entire certificate.
For eligible designs with a nonempty interval, let R=ln(t_+/t_-), select the largest R with the lowest zero-based candidate index at an exact tie, and return R; the numerical family uses -1.0 when no design is eligible.
In the reasoning, report the five-value certificate (candidate index, t_-, t_+, minimum plane efficiency, optimality margin R_best-R_runner-up over eligible designs), the intensity derivative that accounts for normalization, and the local field covariance underlying the exposure probabilities, with a brief source-grounded interpretation of phase locality and longitudinal robustness.
Use full precision internally and report the final scalar to five decimal places; certificate quantities are dimensionless and accepted within 2e-5 relative or 2e-6 absolute error, while the final scalar is accepted within 2e-5 absolute error.
The final public implementation must compose all preceding public functions, directly or transitively, using their returned results.

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

01_rs_operator.py

Goal
----
Rayleigh–Sommerfeld cell operator

```python
def rs_operator(
    source_xy: "np.ndarray",
    target_xy: "np.ndarray",
    wavelength: float,
    z: float,
    source_area: float,
) -> "np.ndarray":
    """Rayleigh–Sommerfeld cell operator.

    Use the complex cell-center Rayleigh–Sommerfeld kernel H_mn=A_s
    z(1-ikR_mn) exp(ikR_mn)/(2 pi R_mn^3), k=2 pi/lambda. Coordinates are
    transverse micrometres; z>0. Source cells have equal area A_s. This
    fixes the outgoing-wave sign and retains the near-field term.

    Parameters
    source_xy : float ndarray (N,2): source-cell centers [x,y],
    micrometres.
    target_xy : float ndarray (M,2): observation centers [x,y],
    micrometres, in target order.
    wavelength : positive float: wavelength in micrometres.
    z : positive float: propagation distance in micrometres.
    source_area : positive float: one source-cell area in square
    micrometres.

    Returns
    complex ndarray (M,N). Element [m,n] maps unit source-cell field to
    observation field; dimensionless.

    Raises
    ValueError for invalid dimensions, nonfinite inputs or parameters
    outside the stated domain.
    """
    return
```

### Step 2

02_image_metrics.py

Goal
----
Normalized image-quality differentials

```python
def image_metrics(
    intensity: "np.ndarray",
    target: "np.ndarray",
    bright: "np.ndarray",
    pixel_area: float,
    incident_power: float,
) -> "np.ndarray":
    """Normalized image-quality differentials.

    For intensity I, bright set B, and target T, mu=mean(I[B]);
    R=sqrt(mean((I/mu-T)^2)); S=sqrt(mean((I-mu)^2)); eta=A_p
    sum(I[B])/P_in. Both means inside R and S range over every observation
    sample. Differentiate with respect to each raw intensity, including the
    dependence of mu. At R=0 or S=0 use the zero subgradient for that norm.
    These are the constructed mask’s explicit observation domains for the
    source Methods objective.

    Parameters
    intensity : float ndarray (M,): nonnegative raw intensities; positive
    bright-region mean.
    target : float ndarray (M,): desired intensity profile relative to the
    bright mean.
    bright : bool ndarray (M,): True at bright target locations; nonempty.
    pixel_area : positive float: equal observation-cell quadrature area,
    square micrometres.
    incident_power : positive float: source-area-weighted sum of incident
    intensity.

    Returns
    float ndarray (3,M+1). Rows [RMSE,SD,eta]; column 0 is the metric and
    columns 1...M its derivatives with respect to raw intensity in
    observation order. RMSE and eta are dimensionless; SD has intensity
    units.

    Raises
    ValueError for invalid dimensions, nonfinite inputs or parameters
    outside the stated domain.
    """
    return
```

### Step 3

03_aps_gradient.py

Goal
----
Scheduled phase-probability objective

```python
def aps_gradient(
    phase: "np.ndarray",
    operator: "np.ndarray",
    amplitude: "np.ndarray",
    target: "np.ndarray",
    bright: "np.ndarray",
    pixel_area: float,
    incident_power: float,
    iteration: int,
    weights: "np.ndarray",
    transition: float,
    width: float,
) -> "np.ndarray":
    """Scheduled phase-probability objective.

    At one-based iteration i, s=(1+tanh((i-transition)/width))/2. The
    objective is s*w[0]*R+(1-s)*w[1]*S+(1-s)*w[2]*(1-eta), evaluated on
    E=H@(amplitude*exp(1j*phase)). Return its phase derivative, retaining
    the intensity normalization’s dependence on phase. The same iteration
    index controls the objective and the Adam update.

    Parameters
    phase : float ndarray (N,): cell phases in radians, in source order.
    operator : complex ndarray (M,N): source-to-observation field operator.
    amplitude : float ndarray (N,): nonnegative incident field amplitudes,
    in source order.
    target : float ndarray (M,): desired intensity profile relative to the
    bright mean.
    bright : bool ndarray (M,): True at bright target locations; nonempty.
    pixel_area : positive float: equal observation-cell quadrature area,
    square micrometres.
    incident_power : positive float: source-area-weighted sum of incident
    intensity.
    iteration : int >=1: one-based objective and update index.
    weights : float ndarray (3,): nonnegative maxima [w_RMSE,w_SD,w_Eff].
    transition : finite float: midpoint of the tanh objective schedule, in
    iterations.
    width : positive float: schedule transition width, in iterations.

    Returns
    float ndarray (N+1,). Entry 0 is the scheduled objective; entries 1...N
    are its phase derivatives in source order.

    Raises
    ValueError for invalid dimensions, nonfinite inputs or parameters
    outside the stated domain.
    """
    return
```

### Step 4

04_optimize_mask.py

Goal
----
Independent-pixel APS trajectory

```python
def optimize_mask(
    initial_phase: "np.ndarray",
    operator: "np.ndarray",
    amplitude: "np.ndarray",
    target: "np.ndarray",
    bright: "np.ndarray",
    pixel_area: float,
    incident_power: float,
    weights: "np.ndarray",
    iterations: int,
    transition: float,
    width: float,
    learning_rate: float,
) -> "np.ndarray":
    """Independent-pixel APS trajectory.

    Each candidate starts from the supplied phase with fresh zero first and
    second moment accumulators. Apply the scheduled APS phase derivative
    for i=1,...,iterations. Adam has beta1=0.9, beta2=0.999 and
    epsilon=1e-8 outside the square root; use bias-corrected moments and
    learning_rate. All pixel phases update simultaneously, in their
    unwrapped real representation. This step composes aps_gradient.

    Parameters
    initial_phase : float ndarray (N,): starting unwrapped cell phases in
    radians.
    operator : complex ndarray (M,N): source-to-observation field operator.
    amplitude : float ndarray (N,): nonnegative incident field amplitudes,
    in source order.
    target : float ndarray (M,): desired intensity profile relative to the
    bright mean.
    bright : bool ndarray (M,): True at bright target locations; nonempty.
    pixel_area : positive float: equal observation-cell quadrature area,
    square micrometres.
    incident_power : positive float: source-area-weighted sum of incident
    intensity.
    weights : float ndarray (3,): nonnegative maxima [w_RMSE,w_SD,w_Eff].
    iterations : int >=0: number of complete simultaneous phase updates.
    transition : finite float: midpoint of the tanh objective schedule, in
    iterations.
    width : positive float: schedule transition width, in iterations.
    learning_rate : positive float: Adam step size.

    Returns
    float ndarray (N,). Final unwrapped phases in radians, in source order.

    Raises
    ValueError for invalid dimensions, nonfinite inputs or parameters
    outside the stated domain.
    """
    return
```

### Step 5

05_fabrication_moments.py

Goal
----
Fabricated geometric-phase moments

```python
def fabrication_moments(
    phase: "np.ndarray", levels: int, rotation_sigma: "np.ndarray"
) -> "np.ndarray":
    """Fabricated geometric-phase moments.

    Quantize phase to levels uniformly spaced values on [0,2pi), using
    nearest level and choosing the increasing phase at an exact half-level
    tie, modulo levels. The nanobrick rotation error is independent
    Normal(0,rotation_sigma[n]^2) at each cell; its geometric-phase error
    is twice that angle. Return the first two complex phasor moments at
    each cell. With q=(phase mod 2pi)/(2pi/levels), values within four
    binary64 ulps of floor(q)+1/2 use the half-level tie convention. The
    rotation-error model is a constructed fabrication perturbation; the
    source supplies the geometric phase and phase-statistics framework.

    Parameters
    phase : float ndarray (N,): cell phases in radians, in source order.
    levels : int >=2: number of uniformly spaced manufactured phase levels.
    rotation_sigma : float ndarray (N,): nonnegative standard deviation of
    nanobrick rotation error in radians.

    Returns
    complex ndarray (2,N). Rows E[Z] and E[Z^2], dimensionless, in source
    order.

    Raises
    ValueError for invalid dimensions, nonfinite inputs or parameters
    outside the stated domain.
    """
    return
```

### Step 6

06_field_statistics.py

Goal
----
Local noncircular field statistics

```python
def field_statistics(
    operator: "np.ndarray", amplitude: "np.ndarray", moments: "np.ndarray"
) -> "np.ndarray":
    """Local noncircular field statistics.

    The fabricated field at observation m is the sum of
    H_mn*amplitude[n]*Z_n over independent unit-modulus phasors.
    moments[k-1,n]=E[Z_n**k] for k=1,2. Calculate each observation’s own
    mean and real/imaginary covariance before applying the source’s
    bivariate-Gaussian closure. The supplied moments may vary between
    cells. Cross-observation correlations are immaterial to the marginal
    reliability criterion.

    Parameters
    operator : complex ndarray (M,N): source-to-observation field operator.
    amplitude : float ndarray (N,): nonnegative incident field amplitudes,
    in source order.
    moments : complex ndarray (2,N): rows E[Z] and E[Z^2], in source order;
    admissible unit-phasor moments.

    Returns
    float ndarray (M,5). Columns [mean U,mean V,var U,cov(U,V),var V],
    observation order. Means have field-amplitude units and covariance
    entries have intensity units.

    Raises
    ValueError for invalid dimensions, nonfinite inputs or parameters
    outside the stated domain.
    """
    return
```

### Step 7

07_intensity_quantiles.py

Goal
----
Gaussian-closure intensity quantiles

```python
def intensity_quantiles(
    statistics: "np.ndarray", probabilities: "np.ndarray"
) -> "np.ndarray":
    """Gaussian-closure intensity quantiles.

    For each row, (U,V) has the stated bivariate Gaussian law, including
    degenerate positive-semidefinite laws, and intensity is U^2+V^2. Return
    the generalized inverse CDF inf{x:Pr(I<=x)>=p} for every supplied
    probability. This is numerical inversion of the intensity law
    underlying the source Eq. (1), with absolute probability error at most
    1e-7 for nonsingular inputs. A zero-covariance law has its
    deterministic intensity as every quantile.

    Parameters
    statistics : float ndarray (M,5): rows [mean U,mean V,var
    U,cov(U,V),var V]; finite, positive-semidefinite covariances.
    probabilities : float ndarray (Q,): probabilities strictly between 0
    and 1, output order preserved.

    Returns
    float ndarray (M,Q). Intensity quantiles, with row order from
    statistics and column order from probabilities; same intensity units as
    the squared field.

    Raises
    ValueError for invalid dimensions, nonfinite inputs or parameters
    outside the stated domain.
    """
    return
```

### Step 8

08_exposure_window.py

Goal
----
Longitudinal exposure certificate

```python
def exposure_window(
    statistics: "np.ndarray",
    quantiles: "np.ndarray",
    bright: "np.ndarray",
    pixel_area: float,
    incident_power: float,
    dose_cap: float,
) -> "np.ndarray":
    """Longitudinal exposure certificate.

    A normalized exposure t prints a location when t*I>=1. Each bright
    location must print with probability at least 1-p; each dark location
    must remain below threshold with probability at least 1-p, separately
    at every declared longitudinal plane. quantiles[...,0] and [...,1]
    contain the p and 1-p intensity quantiles. Intersect these marginal
    exposure intervals with t<=dose_cap. Efficiency at a plane uses the
    ensemble mean bright-region intensity divided by incident power. Return
    the intersection endpoints and the minimum plane efficiency. This
    marginal certificate uses per-location probabilities. Exposure
    endpoints denote the infimum and supremum; deterministic dark
    thresholds can make the upper endpoint open.

    Parameters
    statistics : float ndarray (Z,M,5): per-plane rows [mean U,mean V,var
    U,cov(U,V),var V].
    quantiles : float ndarray (Z,M,2): nonnegative intensity quantiles at
    [p,1-p], with p<1/2.
    bright : bool ndarray (M,): True at bright target locations; nonempty.
    At least one dark observation is also required.
    pixel_area : positive float: equal observation-cell quadrature area,
    square micrometres.
    incident_power : positive float: source-area-weighted sum of incident
    intensity.
    dose_cap : positive float: maximum dimensionless normalized exposure.

    Returns
    float ndarray (3,). [lower exposure,upper exposure,minimum plane
    efficiency], all dimensionless. lower is +inf when a bright p-quantile
    is zero; an empty interval may have upper<=lower.

    Raises
    ValueError for invalid dimensions, nonfinite inputs or parameters
    outside the stated domain.
    """
    return
```

### Step 9

09_certify_hologram.py

Goal
----
Select and certify the hologram

```python
def certify_hologram(
    source_xy: "np.ndarray",
    target_xy: "np.ndarray",
    amplitude: "np.ndarray",
    initial_phase: "np.ndarray",
    target: "np.ndarray",
    bright: "np.ndarray",
    wavelength: float,
    design_z: float,
    planes: "np.ndarray",
    source_area: float,
    pixel_area: float,
    weights: "np.ndarray",
    rotation_sigma: "np.ndarray",
    levels: int,
    iterations: int,
    eta_min: float,
    dose_cap: float,
    tail_probability: float = 0.05,
    transition: float = 175.0,
    width: float = 40.0,
    learning_rate: float = 0.005,
) -> float:
    """Select and certify the hologram.

    Compose every preceding public step, directly or transitively, with all
    intermediate results used. Design each candidate at design_z; certify
    its quantized, perturbed mask at planes. A candidate is eligible when
    minimum plane efficiency >= eta_min and its exposure interval has
    positive width. Maximize log(upper/lower), choosing the earliest
    candidate at an exact tie. Return -1.0 if none is eligible. The single
    scalar is the selected logarithmic exposure latitude. This
    design-selection and reliability certificate is a task-defined
    extension of the source method.

    Parameters
    source_xy : float ndarray (N,2): source-cell centers [x,y],
    micrometres.
    target_xy : float ndarray (M,2): observation centers [x,y],
    micrometres, in target order.
    amplitude : float ndarray (N,): nonnegative incident field amplitudes,
    in source order.
    initial_phase : float ndarray (N,): starting unwrapped cell phases in
    radians.
    target : float ndarray (M,): desired intensity profile relative to the
    bright mean.
    bright : bool ndarray (M,): True at bright target locations; nonempty.
    At least one dark observation is also required.
    wavelength : positive float: wavelength in micrometres.
    design_z : positive float: design-plane distance in micrometres.
    planes : float ndarray (Z,): positive certification distances in
    micrometres, ordered as supplied.
    source_area : positive float: one source-cell area in square
    micrometres.
    pixel_area : positive float: equal observation-cell quadrature area,
    square micrometres.
    weights : float ndarray (C,3): candidate rows [w_RMSE,w_SD,w_Eff] in
    selection order.
    rotation_sigma : float ndarray (N,): nonnegative standard deviation of
    nanobrick rotation error in radians.
    levels : int >=2: number of uniformly spaced manufactured phase levels.
    iterations : int >=0: number of complete simultaneous phase updates.
    eta_min : nonnegative float: minimum allowed worst-plane diffraction
    efficiency.
    dose_cap : positive float: maximum dimensionless normalized exposure.
    tail_probability : float in (0,0.5): allowed marginal error probability
    (default 0.05).
    transition : finite float: midpoint of the tanh objective schedule, in
    iterations.
    width : positive float: schedule transition width, in iterations.
    learning_rate : positive float: Adam step size.

    Returns
    float. Selected dimensionless log(upper/lower), or -1.0 when no
    candidate is eligible.

    Raises
    ValueError for invalid dimensions, nonfinite inputs or parameters
    outside the stated domain.
    """
    return
```
