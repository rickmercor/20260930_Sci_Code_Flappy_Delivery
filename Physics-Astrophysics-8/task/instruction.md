# Physics-Astrophysics-8

## Background

Scintillator measurements blur emitted $\gamma$-ray spectra through energy loss, finite detection efficiency and finite energy resolution. Recovering an emitted spectrum from Poisson counts is consequently an ill-conditioned inverse problem, with statistical background adding further uncertainty. Empirical-Bayes models use information from the observed spectrum to set a regularising prior. Examining that prior in an experimentally resolvable space helps assess its centre and uncertainty before posterior inference.

## Problem

Unfolding the $\gamma$-ray spectrum recorded at a fixed excitation energy is an ill-conditioned Poisson inverse problem, because the detector response redistributes and broadens the emitted intensity until distinct emitted spectra become nearly indistinguishable once folded. An empirical-Bayes treatment regularises it through a prior on the emitted spectrum that is centred on a data-driven reference, itself a Richardson-Lucy iterate of the signal run, and that carries a bin-wise width. The prior is specified in emitted space, and its consequences are judged in the space in which the framework reports its spectra, where draws are summarised by a mean curve and a simultaneous band. The reduced instance below freezes one such prior construction for a single excitation-energy bin on an active grid of $J = 8$ $\gamma$-energy bins.

The redistribution operator $\mathbf{D}$ carries in entry $(i, j)$ the probability that a photon emitted in bin $j$ deposits its energy in bin $i$. A photon that traverses the crystal without depositing energy is never registered, so each column of $\mathbf{D}$ sums to the detection efficiency of that bin rather than to one. The resolution operator $\mathbf{G}_\gamma$ carries in entry $(i, j)$ the probability that an energy deposited in bin $j$ is registered in bin $i$, and it preserves counts, so every column of $\mathbf{G}_\gamma$ sums to one.

```python
import numpy as np

D = np.array([
    [0.9600, 0.0920, 0.0634, 0.0528, 0.0466, 0.0421, 0.0386, 0.0356],
    [0.0000, 0.8280, 0.0686, 0.0506, 0.0412, 0.0352, 0.0309, 0.0275],
    [0.0000, 0.0000, 0.7480, 0.0646, 0.0511, 0.0427, 0.0368, 0.0323],
    [0.0000, 0.0000, 0.0000, 0.6720, 0.0611, 0.0502, 0.0427, 0.0370],
    [0.0000, 0.0000, 0.0000, 0.0000, 0.6000, 0.0577, 0.0486, 0.0418],
    [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.5321, 0.0545, 0.0466],
    [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.4679, 0.0513],
    [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.4079],
])
G_gamma = np.array([
    [0.9220, 0.0724, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
    [0.0780, 0.8552, 0.0724, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
    [0.0000, 0.0724, 0.8552, 0.0724, 0.0000, 0.0000, 0.0000, 0.0000],
    [0.0000, 0.0000, 0.0724, 0.8552, 0.0724, 0.0000, 0.0000, 0.0000],
    [0.0000, 0.0000, 0.0000, 0.0724, 0.8552, 0.0724, 0.0000, 0.0000],
    [0.0000, 0.0000, 0.0000, 0.0000, 0.0724, 0.8552, 0.0724, 0.0000],
    [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0724, 0.8552, 0.0780],
    [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0724, 0.9220],
])
```

The signal run recorded the counts $\mathbf{n}$, and a background run of equal exposure recorded $\mathbf{n}_{\mathrm{off}}$ in the absence of signal. The emitted spectrum from which the signal run was generated is $\boldsymbol{\mu}_{\mathrm{true}}$.

```python
n = np.array([381, 1541, 460, 157, 293, 39, 0, 0])
n_off = np.array([48, 154, 51, 22, 51, 15, 1, 0])
mu_true = np.array([60, 1800, 400, 120, 480, 6, 0.3, 0.1])
```

Wherever the construction resamples the signal run, use the $K = 4$ Poisson resamples below in place of that resampling, one resample per row.

```python
N_res = np.array([
    [406, 1578, 429, 153, 292, 35, 0, 0],
    [403, 1568, 417, 147, 302, 38, 0, 0],
    [364, 1606, 456, 166, 298, 22, 0, 0],
    [358, 1573, 453, 172, 290, 35, 0, 0],
])
```

The prior is drawn $N = 16$ times. Wherever a draw calls for a standard-normal variate, take it from $\mathbf{Z}$, and wherever it calls for a variate of a distribution that is not normal, evaluate the inverse cumulative distribution function of that distribution at the corresponding entry of $\mathbf{U}$, one row per draw and one column per bin.

```python
Z = np.array([
    [-1.3172, 1.2482, -0.0462, 0.5644, 0.4015, 0.8277, -0.5209, -1.1960],
    [0.2368, -0.2066, 0.6623, -1.0854, -0.1586, 0.0253, 0.3285, 0.9852],
    [1.7075, 0.8812, 0.7809, -0.2051, -0.9937, 1.5846, -0.9145, -0.7679],
    [-0.9799, -1.0106, -1.2303, 1.0254, 0.1357, -1.3587, -2.2404, -0.2426],
    [-0.6562, 1.3738, -0.1800, 1.7620, 2.2083, 0.1282, 0.5415, -0.2978],
    [-1.1052, -0.5546, 0.4613, 1.5446, -0.3304, -0.1789, -0.3454, -0.3855],
    [0.4446, -0.5375, 0.1057, 0.6215, -0.7146, -0.8187, 0.4618, 0.2888],
    [-0.7490, -1.3887, -2.0672, -1.0401, -0.8608, 0.0542, -0.0914, 0.7233],
    [0.1356, -1.7114, -0.2807, 1.2544, -0.0028, -0.2288, 0.0483, 1.6938],
    [0.6963, 0.3910, 1.9035, 0.3795, -0.9446, -0.6751, -0.3260, -0.1255],
    [-1.3818, 0.5132, 0.3204, -0.6988, -0.6736, 0.3571, 0.2233, -0.3653],
    [1.6635, -0.4442, 0.4165, 1.8852, -0.6552, -1.2765, -0.5137, 0.7405],
    [0.0872, 0.7130, 1.0826, 0.4673, -2.1087, -1.9236, 0.2541, -1.0312],
    [-1.1075, 0.8420, -0.3870, 0.1768, -0.0485, -1.0076, -0.4778, 1.6000],
    [0.4420, 0.5495, 0.0880, -0.1985, 1.5141, 1.5562, -0.3511, -1.1659],
    [0.3934, 0.5039, 0.7601, -0.9601, 1.1018, -1.0292, -0.4393, -0.4425],
])
U = np.array([
    [0.0888, 0.9355, 0.0318, 0.8306, 0.0715, 0.0664, 0.8101, 0.9222],
    [0.2122, 0.5367, 0.5288, 0.9245, 0.9291, 0.5789, 0.3794, 0.9377],
    [0.1072, 0.6361, 0.0406, 0.5973, 0.2047, 0.2923, 0.4446, 0.3373],
    [0.7535, 0.0263, 0.1654, 0.4910, 0.9082, 0.1059, 0.2709, 0.8614],
    [0.4247, 0.7255, 0.0231, 0.1075, 0.4890, 0.0209, 0.5067, 0.9778],
    [0.3624, 0.5765, 0.7857, 0.5459, 0.3383, 0.1387, 0.2239, 0.4842],
    [0.8331, 0.4785, 0.7814, 0.3562, 0.6291, 0.6273, 0.4827, 0.7815],
    [0.2814, 0.8762, 0.3056, 0.7721, 0.8010, 0.3934, 0.2598, 0.5451],
    [0.1599, 0.9630, 0.5310, 0.0950, 0.0614, 0.6113, 0.1181, 0.2301],
    [0.9018, 0.1109, 0.0210, 0.1611, 0.8519, 0.3587, 0.8923, 0.1646],
    [0.4891, 0.5280, 0.1943, 0.8251, 0.0464, 0.3037, 0.9450, 0.5611],
    [0.7510, 0.7096, 0.5391, 0.4929, 0.1551, 0.2387, 0.2519, 0.5734],
    [0.9566, 0.7836, 0.6671, 0.4128, 0.9785, 0.1016, 0.6567, 0.0204],
    [0.7842, 0.5067, 0.8415, 0.6694, 0.8716, 0.1006, 0.0680, 0.7919],
    [0.2552, 0.0236, 0.8433, 0.1673, 0.7827, 0.7424, 0.9036, 0.2916],
    [0.7761, 0.2803, 0.7159, 0.5054, 0.0913, 0.2354, 0.1605, 0.2408],
])
```

The iteration that produces the reference starts from a flat emitted spectrum holding the total count of the run it is applied to less the total background reference, spread evenly over the $J$ bins, and runs for at most $500$ updates, the flat start being iteration $0$ and each resampled run starting from its own flat spectrum in the same way. The reference is selected with a window of $10$ updates, a change-to-noise threshold of $2$ and a run of $10$ consecutive qualifying iterations, the selected iteration being the first of that run, or the cap if no such run exists. The iterate is monitored in the space in which the framework reports its spectra, and the noise level at an iteration is the square root of the summed sample variances, with divisor $K - 1$, of the components of the monitored iterate across the $K$ resampled runs, divided by the Euclidean norm of the mean of that iterate over the resampled runs. The remaining baseline hyperparameters are a Gamma shape of $1$ for the background prior, a reference floor of $0.1$ counts, width bounds of $1$ and $3$, a count scale of $100$ counts, a Gamma shape of $1$ for the emitted intensity, an envelope mass of $0.75$, and a guard constant of $10^{-12}$ added to every denominator and used as a floor on every band half-width.

1. Following the framework's frozen conventions throughout, build the empirical prior of this excitation-energy bin, from the background reference and the selected Richardson-Lucy iterate to the centre and the width of every bin.
2. Generate the $N$ prior draws, map them to the space in which the framework reports its spectra, and summarise them by their mean curve over all $N$ draws and the simultaneous band of the stated mass.
3. Report as the single final scalar the mean over the $J$ bins of the absolute deviation of the mean curve from the truth in that space, each deviation scaled by the half-width of the band in that bin.

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

01_compose_detector_response

Goal
----
Compose the detector response of the scintillator array from its redistribution and resolution operators.

```python
def compose_detector_response(
    redistribution: np.ndarray,
    resolution: np.ndarray,
) -> np.ndarray:
    r"""Compose the redistribution and resolution operators into the detector response.

    Parameters
    ----------
    redistribution : np.ndarray
        Energy-redistribution operator, shape (J, J), non-negative and finite,
        entry (i, j) being the probability that a photon emitted in bin j
        deposits its energy in bin i. Every column sums to the detection
        efficiency of that bin, a value above zero and at most one to within
        1e-6.
    resolution : np.ndarray
        Resolution-broadening operator, shape (J, J), non-negative and
        finite, entry (i, j) being the probability that an energy deposited
        in bin j is registered in bin i. Every column sums to one to within
        1e-6.

    Returns
    -------
    response : np.ndarray
        Shape (J, J). The composite response applied to an emitted spectrum
        gives the expected detected signal, and its column sums are the
        detection efficiencies. The product is returned as it is and is not
        renormalised.

    Raises
    ------
    ValueError
        If either operator is not a square two-dimensional array, if the two
        differ in shape, if the grid is empty, if an entry is negative or not
        finite, if a column of the redistribution sums to zero or exceeds one
        by more than 1e-6, or if a column of the resolution does not sum to
        one to within 1e-6.
    """
    return
```

### Step 2

02_build_background_reference

Goal
----
Build the fixed background expectation that the reference iteration folds into its forward model.

```python
def build_background_reference(
    off_counts: np.ndarray,
    prior_shape: float,
) -> np.ndarray:
    r"""Compute the per-bin background reference from the background run.

    Parameters
    ----------
    off_counts : np.ndarray
        Counts of the background run, shape (J,), non-negative integer
        values, at least one of them positive, recorded with the same
        exposure as the signal run.
    prior_shape : float
        Shape parameter of the Gamma prior on the latent background
        expectation, strictly positive. The rate of that prior is not an
        input. It is fixed by the background run itself.

    Returns
    -------
    background_reference : np.ndarray
        Shape (J,). The fixed background expectation of every bin used inside
        the reference iteration, strictly positive in every bin.

    Raises
    ------
    ValueError
        If ``off_counts`` is not one-dimensional or is empty, if an entry is
        negative, not finite or not an integer value, if every entry is zero,
        or if ``prior_shape`` is not a finite positive number.
    """
    return
```

### Step 3

03_run_richardson_lucy_iterates

Goal
----
Run the background-aware Richardson-Lucy iteration on the signal run and keep every iterate.

```python
def run_richardson_lucy_iterates(
    on_counts: np.ndarray,
    response: np.ndarray,
    background_reference: np.ndarray,
    n_iterations: int,
    guard: float,
) -> np.ndarray:
    r"""Iterate the background-aware update from the flat start and keep every iterate.

    Parameters
    ----------
    on_counts : np.ndarray
        Counts of the signal run, shape (J,), non-negative integer values.
    response : np.ndarray
        Composite detector response, shape (J, J), non-negative and finite,
        entry (i, j) mapping emitted bin j to registered bin i, every column
        summing to the detection efficiency of that bin, above zero and at
        most one to within 1e-6. The column sums are the sensitivities that
        divide the update.
    background_reference : np.ndarray
        Fixed background expectation of every bin, shape (J,), non-negative
        and finite, in counts.
    n_iterations : int
        Number of updates to apply, zero or more. Zero returns the flat
        start alone.
    guard : float
        Non-negative constant added to every predicted count before the
        observed counts are divided by it.

    Returns
    -------
    iterates : np.ndarray
        Shape (n_iterations + 1, J). Row t is the emitted spectrum after t
        updates, row 0 the flat start whose bins all equal the net count of
        the signal run divided by the number of bins.

    Raises
    ------
    ValueError
        If the arrays are not of the shapes above, are empty or disagree on
        the number of bins, if an entry is negative or not finite, if a count
        is not an integer value, if a column of the response sums to zero or
        exceeds one by more than 1e-6, if ``n_iterations`` is negative or not
        an integer, if ``guard`` is negative or not finite, or if the net count
        of the signal run, its total less the total background reference, is
        not positive.
    """
    return
```

### Step 4

04_select_reference_iteration

Goal
----
Select the Richardson-Lucy iterate that serves as the reference of the empirical prior.

```python
def select_reference_iteration(
    on_counts: np.ndarray,
    resampled_counts: np.ndarray,
    response: np.ndarray,
    resolution: np.ndarray,
    background_reference: np.ndarray,
    window: int,
    ratio_threshold: float,
    run_length: int,
    max_iterations: int,
    guard: float,
) -> int:
    r"""Apply the change-to-noise stopping rule and return the selected iteration.

    Parameters
    ----------
    on_counts : np.ndarray
        Counts of the signal run, shape (J,), non-negative integer values.
    resampled_counts : np.ndarray
        Poisson resamples of the signal run, shape (K, J) with K at least 2,
        one resample per row, non-negative integer values. Their spread at
        each iteration is a sample variance with divisor K minus one.
    response : np.ndarray
        Composite detector response, shape (J, J), every column summing to
        the detection efficiency of that bin, above zero and at most one to
        within 1e-6.
    resolution : np.ndarray
        Resolution-broadening operator, shape (J, J), non-negative and
        finite, mapping an emitted spectrum to its resolution-limited form.
    background_reference : np.ndarray
        Fixed background expectation of every bin, shape (J,), non-negative
        and finite.
    window : int
        Number of updates between the two iterates whose difference defines
        the deterministic change, at least 1.
    ratio_threshold : float
        Largest admissible value of the change divided by the noise level,
        finite and positive.
    run_length : int
        Number of consecutive iterations, starting at the candidate, over
        which the ratio must stay at or below the threshold, at least 1.
    max_iterations : int
        Number of updates run, at least 1. It is also the fallback when no
        qualifying run of iterations exists below it.
    guard : float
        Non-negative constant added to every denominator, and to every
        predicted count inside the iteration.

    Returns
    -------
    selected : int
        The selected iteration index between ``window`` and
        ``max_iterations``. It is ``max_iterations`` when no candidate
        qualifies, including when ``window + run_length - 1`` exceeds
        ``max_iterations`` so that no run of iterations fits.

    Raises
    ------
    ValueError
        If the arrays are not of the shapes above, are empty or disagree on
        the number of bins, if there are fewer than two resamples, if a count
        is negative, not finite or not an integer value, if an entry of the
        response or of the resolution is negative or not finite, if a column
        of the response sums to zero or exceeds one by more than 1e-6, if
        ``window``,
        ``run_length`` or ``max_iterations`` is not a positive integer, if
        ``ratio_threshold`` is not finite and positive, if ``guard`` is
        negative or not finite, or if the signal run or any resample does
        not hold more counts than the background reference.
    """
    return
```

### Step 5

05_build_prior_centre_and_width

Goal
----
Turn the selected Richardson-Lucy iterate into the centre and the bin-wise width of the empirical prior.

```python
def build_prior_centre_and_width(
    reference_iterate: np.ndarray,
    resolution: np.ndarray,
    floor: float,
    sigma_min: float,
    sigma_max: float,
    count_scale: float,
) -> np.ndarray:
    r"""Floor the selected iterate and derive the adaptive prior width of every bin.

    Parameters
    ----------
    reference_iterate : np.ndarray
        The selected Richardson-Lucy iterate in emitted space, shape (J,),
        non-negative and finite, in counts.
    resolution : np.ndarray
        Resolution-broadening operator, shape (J, J), non-negative and
        finite, every column summing to one to within 1e-6.
    floor : float
        Lowest admissible centre of any bin, in counts, finite and strictly
        positive.
    sigma_min : float
        Lower bound of the log-scale width, finite and strictly positive.
    sigma_max : float
        Upper bound of the log-scale width, finite and at least
        ``sigma_min``.
    count_scale : float
        Resolution-limited count level at which the shape information is
        half activated, finite and strictly positive.

    Returns
    -------
    centre_and_width : np.ndarray
        Shape (2, J). Row 0 is the floored centre, every entry at least
        ``floor``. Row 1 is the width of every bin, lying between
        ``sigma_min`` and ``sigma_max``.

    Raises
    ------
    ValueError
        If the arrays are not of the shapes above, are empty or disagree on
        the number of bins, if an entry is negative or not finite, if a
        column of the resolution does not sum to one to within 1e-6, or if
        ``floor``, ``sigma_min``, ``sigma_max`` or ``count_scale`` violates
        the conditions above.
    """
    return
```

### Step 6

06_draw_prior_spectra

Goal
----
Generate resolution-limited draws of the empirical prior from supplied standard-normal and uniform variates.

```python
def draw_prior_spectra(
    centre: np.ndarray,
    width: np.ndarray,
    gamma_shape: float,
    normal_draws: np.ndarray,
    uniform_draws: np.ndarray,
    resolution: np.ndarray,
) -> np.ndarray:
    r"""Map supplied variates through the two-layer prior and the resolution operator.

    Parameters
    ----------
    centre : np.ndarray
        Prior centre of every bin in emitted space, shape (J,), finite and
        strictly positive.
    width : np.ndarray
        Log-scale width of every bin, shape (J,), finite and strictly
        positive.
    gamma_shape : float
        Shape parameter of the conditional Gamma distribution, finite and
        strictly positive. The rate is the shape divided by the latent mean
        of the bin.
    normal_draws : np.ndarray
        Standard-normal variates, shape (N, J), finite, feeding the Gaussian
        layer of the corresponding bin and draw.
    uniform_draws : np.ndarray
        Uniform variates, shape (N, J), strictly between 0 and 1, evaluated
        through the inverse cumulative distribution function of the
        conditional Gamma distribution of the corresponding bin and draw.
    resolution : np.ndarray
        Resolution-broadening operator, shape (J, J), non-negative and
        finite, every column summing to one to within 1e-6.

    Returns
    -------
    draws : np.ndarray
        Shape (N, J). Row s is the s-th prior draw of the emitted spectrum
        mapped through the resolution operator.

    Raises
    ------
    ValueError
        If the arrays are not of the shapes above or disagree on the number
        of bins or draws, if there are no draws or no bins, if a centre or
        width is not finite and strictly positive, if ``gamma_shape`` is not
        finite and strictly positive, if a normal variate is not finite, if a
        uniform variate lies outside the open unit interval, or if the
        resolution is negative, not finite or not column-normalised to within
        1e-6.
    """
    return
```

### Step 7

07_build_global_rank_envelope

Goal
----
Summarise a set of spectral draws by their mean curve and a simultaneous global rank-envelope band.

```python
def build_global_rank_envelope(
    draws: np.ndarray,
    mass: float,
) -> np.ndarray:
    r"""Build the mean curve and the simultaneous rank-envelope band of a set of draws.

    Parameters
    ----------
    draws : np.ndarray
        Spectral draws, shape (N, J) with N at least 2, finite, one draw per
        row and one gamma-energy bin per column.
    mass : float
        Envelope mass level, strictly greater than 0 and at most 1. The
        number of retained curves is the ceiling of ``mass`` times N.

    Returns
    -------
    envelope : np.ndarray
        Shape (3, J). Row 0 is the mean over all N draws, row 1 the
        bin-wise minimum over the retained curves and row 2 the bin-wise
        maximum over the retained curves.

    Raises
    ------
    ValueError
        If ``draws`` is not a two-dimensional array with at least two rows
        and one column, if an entry is not finite, or if ``mass`` is not a
        finite number in the half-open interval (0, 1].
    """
    return
```

### Step 8

08_orchestrate_prior_diagnostic

Goal
----
Run the whole empirical-prior construction for one excitation-energy bin and reduce it to the scalars that measure how the prior sits against the truth.

```python
def orchestrate_prior_diagnostic(
    redistribution: np.ndarray,
    resolution: np.ndarray,
    on_counts: np.ndarray,
    off_counts: np.ndarray,
    resampled_counts: np.ndarray,
    emitted_truth: np.ndarray,
    normal_draws: np.ndarray,
    uniform_draws: np.ndarray,
    background_prior_shape: float,
    window: int,
    ratio_threshold: float,
    run_length: int,
    max_iterations: int,
    floor: float,
    sigma_min: float,
    sigma_max: float,
    count_scale: float,
    gamma_shape: float,
    mass: float,
    guard: float,
) -> np.ndarray:
    r"""Execute the complete prior construction and return its three scalars.

    Parameters
    ----------
    redistribution : np.ndarray
        Energy-redistribution operator, shape (J, J), every column summing to
        the detection efficiency of that bin.
    resolution : np.ndarray
        Resolution-broadening operator, shape (J, J), column-normalised.
    on_counts : np.ndarray
        Counts of the signal run, shape (J,), non-negative integer values.
    off_counts : np.ndarray
        Counts of the background run, shape (J,), non-negative integer
        values, at least one of them positive.
    resampled_counts : np.ndarray
        Poisson resamples of the signal run, shape (K, J) with K at least 2.
    emitted_truth : np.ndarray
        Emitted spectrum the signal run was generated from, shape (J,),
        non-negative and finite.
    normal_draws : np.ndarray
        Standard-normal variates of the Gaussian prior layer, shape (N, J).
    uniform_draws : np.ndarray
        Uniform variates of the conditional Gamma layer, shape (N, J),
        strictly inside the unit interval.
    background_prior_shape : float
        Shape of the Gamma prior on the background expectation, positive.
    window : int
        Window of the deterministic change, at least 1.
    ratio_threshold : float
        Largest admissible change-to-noise ratio, positive.
    run_length : int
        Number of consecutive qualifying iterations required, at least 1.
    max_iterations : int
        Number of updates run and the fallback selection, at least 1.
    floor : float
        Lowest admissible prior centre in counts, positive.
    sigma_min, sigma_max : float
        Bounds of the log-scale prior width, ``sigma_max`` at least
        ``sigma_min``.
    count_scale : float
        Resolution-limited count level of half activation, positive.
    gamma_shape : float
        Shape of the conditional Gamma distribution, positive.
    mass : float
        Envelope mass level in (0, 1].
    guard : float
        Non-negative constant guarding every denominator and the band
        half-width.

    Returns
    -------
    results : np.ndarray
        Shape (3,). The mean over bins of the absolute scaled deviation of
        the mean curve from the resolution-limited truth, the selected
        reference iteration, and the largest absolute scaled deviation over
        bins. Entry 0 is the final scalar.

    Raises
    ------
    ValueError
        If any upstream contract fails, if ``emitted_truth`` is not of shape
        (J,) or holds a negative or non-finite entry, or if a reported
        scalar is not finite.
    """
    return
```
