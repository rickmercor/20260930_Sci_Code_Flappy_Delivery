# Material_Science-Semiconductor_Materials-13

## Background

# Scientific background

Electrically active point defects govern how long a photogenerated carrier survives in a semiconductor, and therefore how efficient a solar cell, a detector or a power device can be. A defect that introduces a level deep in the band gap acts as a recombination centre: it captures an electron and then a hole, and the rate at which it destroys carriers scales with the product of its concentration and its capture cross section. Two defects can therefore matter equally to device performance while differing by more than an order of magnitude in density, which is why quantifying each species separately, rather than the total defect load, is what actually guides material optimisation. In III--V material grown by metalorganic vapour-phase epitaxy the question is pressing, because pushing the growth rate up to make the process economic is known to multiply some native defect populations, and knowing which of them to suppress requires resolving them individually.

Deep-level transient spectroscopy has been the standard tool for this since the 1970s. A junction is held in reverse bias, a short filling pulse collapses the space charge region so that deep levels capture majority carriers, and the bias is restored; the levels then emit their carriers thermally, and the resulting change in the space charge produces a capacitance transient whose decay constant is the emission rate and whose amplitude is proportional to the density of the level. Because the emission rate depends exponentially on the trap depth, sweeping temperature while correlating the transient against a fixed rate window turns the measurement into a spectrum with one peak per level, and the peak position and shape yield the trap depth, the capture cross section and the concentration.

That classical evaluation rests on the transient being a single exponential. When two levels lie within a few tens of millielectronvolts of one another the correlated signal is a superposition of two peaks that broadens and merges rather than resolving, and the extracted parameters correspond to no real defect. Two families of remedy exist. The first transforms the transient into a distribution over emission rates by numerically inverting a Laplace-like integral equation and requires regularisation because the inversion of a sum of exponentials is severely ill-posed. The second fits a multi-exponential model directly to the time-domain record with rates and amplitudes free together, which avoids regularisation but is a non-convex problem that is sensitive to noise and requires the number of components to be fixed in advance.

Recent methodological work in defect spectroscopy combines those remedies sequentially: a regularised inverse identifies emission rates, an Arrhenius description is fitted across temperature, and a time-domain least-squares fit then determines the amplitudes at fixed rates. This reduces the free amplitude problem to a linear fit and confines the analysis to a temperature window in which every level's emission lies inside the acquisition time span. The task asks the solver to recover from the literature why the Arrhenius-smoothed rates and time-domain amplitudes are essential rather than interchangeable implementation choices. The physical conversion from capacitance deflection to trap density also contains a level-dependent partial-probing correction fixed by trap depth and junction-bias geometry; when those junction inputs are not supplied, the deterministic output is an explicitly uncorrected or apparent concentration obtained by setting that factor to unity, not a claim that it cancels exactly between levels.

## Problem

Deep-level transient spectroscopy is the standard probe of electrically active defects in semiconductors, but its conventional rate-window evaluation presumes a single exponential relaxation and therefore merges the signatures of two levels whose emission rates are close, returning a broadened peak from which no individual concentration can be read; separating them matters because two levels present at very different densities can contribute comparably to Shockley--Read--Hall recombination, so material optimisation aimed only at the stronger signature can leave the sample limited by the weaker level. Given a temperature series of capacitance transients from a junction hosting two majority-carrier deep levels, the analysis returns the apparent trap concentration of each under the convention stated below, and the quantity of interest here is their ratio.

The transient is the linear superposition of one decaying exponential per level on the quiescent capacitance, each exponential decaying at that level's thermal emission rate and carrying an amplitude proportional to the density of states it emptied; recovering the two rates from a single record is an ill-posed inverse problem, while recovering the two amplitudes at known rates is not, and the two halves of the analysis are best not attempted at once.

Reproduce and then analyse the following acquisition: an $n$-type junction with shallow doping $N_D = 2 \times 10^{16}\,\mathrm{cm}^{-3}$ and quiescent capacitance $C(V_R) = 204.5\,\mathrm{pF}$ hosts two levels emitting into a band of effective mass $0.063\,m_e$, the emission prefactor being built from the root-mean-square thermal velocity $\sqrt{3k_BT/m^*}$ rather than the Maxwell mean speed, with trap depths $0.711$ and $0.658\,\mathrm{eV}$, infinite-temperature capture cross sections $1.8 \times 10^{-15}$ and $9.1 \times 10^{-15}\,\mathrm{cm}^2$, and capacitance deflections $-0.75$ and $-0.05\,\mathrm{pF}$; transients are recorded at $300$ to $480\,\mathrm{K}$ in $3\,\mathrm{K}$ steps, sampled at $100\,\mathrm{kHz}$ for $10^5$ points at times $k/f_s$ for $k = 1 \ldots 10^5$, with additive Gaussian noise of standard deviation $\sigma(T)=0.0032\,\mathrm{pF} \times \sqrt{T/350\,\mathrm{K}}$ taken as one block of $10^5$ standard normal draws per transient from `numpy.random.default_rng(20260722)` with the temperatures in increasing order. Bin each transient into $100$ geometrically spaced time intervals spanning the first to the last sample, discarding empty intervals and representing each surviving interval by the mean time and the mean capacitance of the samples it holds; retain the raw count $n$ of every surviving bin to define its relative row multiplier $\sqrt{n}$.

Generate and bin all transients, but apply the inversion, peak extraction, Arrhenius regression, fixed-rate amplitude fit, and final averaging only to temperatures in the active window $370$ to $450\,\mathrm{K}$; on the stated grid these are the $27$ temperatures $372, 375, \ldots, 450\,\mathrm{K}$; for each binned transient in that active window, obtain the non-negative emission-rate spectral density, referred to that transient's own last binned value rather than to the quiescent capacitance, on a grid of $150$ geometrically spaced rates from $0.1$ to $10^{5}\,\mathrm{s}^{-1}$ by minimising $\|\operatorname{diag}(\sqrt{n})(Kf-y)\|_2^2+10^{-4}\|f\|_2^2$; the per-transient noise scale $\sigma(T)$ is used only to generate the data and is deliberately not divided out of these inversion rows or used to rescale the fixed penalty. Take the two features of largest peak height to be the two levels, define a feature's rate as the spectral-weight-weighted geometric mean of the grid rates it spans, and assign each separating minimum grid point to the feature on its right so that adjacent cluster slices neither overlap nor leave a gap. Determine the individual capacitance deflections in the time domain rather than from the spectral density, at emission rates held fixed by the Arrhenius description of each level, using the same exact row multiplier $\sqrt n$ and refitting the quiescent capacitance alongside the deflections. For the concentration conversion set the level-dependent partial-probing factors to $\gamma_1=\gamma_2=1$, so the target is explicitly the uncorrected or apparent concentration ratio $N_{T,k}^{\mathrm{app}}=2N_D|\Delta C_k|/C(V_R)$ rather than a bias-geometry-corrected physical ratio. Report the ratio of the larger recovered apparent trap concentration to the smaller one, and let your reasoning state the trap depth, capture cross section and capacitance deflection recovered for each level, identify which level is recovered less faithfully and why, explain the paper-specific reasons for recomputing fixed rates from the Arrhenius description and for rejecting spectral-weight amplitudes, give the apparent ratio that a reading of the spectral weights alone would have produced instead, and compare the levels' apparent $\sigma_\infty N_T^{\mathrm{app}}$ contributions to Shockley--Read--Hall recombination.


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

Implement **all 11 functions** below into a single file at `/app/solution.py`. Requirements:

- Use the EXACT function name and signature shown for each step.
- Define every function at module scope (importable).
- Do NOT include tests, example usage, prints, or a `__main__` block.
- Available libraries: numpy, scipy, sympy, pandas, scikit-learn, networkx.

Start your file with these imports:

```python
import numpy as np
```

### Step 1

01_compute_emission_rate

Goal
----
Return the thermal emission rate of a deep level with a given activation energy and high-temperature capture cross section, at each of a set of sample temperatures.

```python
import numpy as np

def compute_emission_rate(temperature: np.ndarray, activation_energy: float,
                          sigma_inf: float, mass_ratio: float = 0.063) -> np.ndarray:
    """Return the thermal emission rate of a deep level at each temperature.

    Parameters
    ----------
    temperature : np.ndarray
        Sample temperatures in K, all strictly positive.
    activation_energy : float
        Trap depth below the receiving band edge in eV (activation_energy > 0).
    sigma_inf : float
        Capture cross section extrapolated to infinite temperature in cm^2
        (sigma_inf > 0).
    mass_ratio : float
        Carrier effective mass of the receiving band in units of the free
        electron mass (mass_ratio > 0).

    Returns
    -------
    rate : np.ndarray
        Emission rate in 1/s, one entry per input temperature, equal to
        sigma_inf * v_th * N_C * exp(-activation_energy / (k_B T)) with the
        root-mean-square thermal velocity v_th = sqrt(3 k_B T / m*) in cm/s and
        the effective density of states N_C = 2 (2 pi m* k_B T / h^2)^(3/2) in
        1/cm^3.

    Raises
    ------
    ValueError
        If any argument is outside the stated domain.

    Notes
    -----
    The CODATA 2018 constants are used throughout, as k_B = 8.617333262e-5 eV/K
    and 1.380649e-23 J/K, h = 6.62607015e-34 J s and m_e = 9.1093837015e-31 kg.
    """
    return rate  # placeholder
```

### Step 2

02_generate_capacitance_transients

Goal
----
Return the noisy capacitance transient recorded at each sample temperature for a junction containing several deep levels, given the levels' emission parameters and capacitance deflections.

```python
import numpy as np

def generate_capacitance_transients(temperature: np.ndarray, activation_energy: np.ndarray,
                                    sigma_inf: np.ndarray, deflection: np.ndarray,
                                    base_capacitance: float, sampling_rate: float,
                                    n_samples: int, noise_ref: float,
                                    temperature_ref: float, seed: int,
                                    mass_ratio: float = 0.063) -> np.ndarray:
    """Return the noisy capacitance transients of a multi-level junction.

    Parameters
    ----------
    temperature : np.ndarray
        Sample temperatures in K, in the order they are acquired.
    activation_energy : np.ndarray
        Trap depth of each deep level in eV.
    sigma_inf : np.ndarray
        Infinite-temperature capture cross section of each level in cm^2.
    deflection : np.ndarray
        Capacitance deflection of each level in pF, negative for majority
        carrier capture.
    base_capacitance : float
        Quiescent capacitance at the reverse bias in pF.
    sampling_rate : float
        Sampling rate in Hz (sampling_rate > 0).
    n_samples : int
        Number of samples per transient (n_samples >= 1).
    noise_ref : float
        Standard deviation in pF of the additive noise at the reference
        temperature (noise_ref >= 0).
    temperature_ref : float
        Reference temperature in K at which noise_ref applies.
    seed : int
        Seed of the NumPy default random generator; the noise of each
        temperature is drawn in the order the temperatures are given.
    mass_ratio : float
        Carrier effective mass of the receiving band in units of the free
        electron mass.

    Returns
    -------
    transients : np.ndarray
        Array of shape (n_temperatures, n_samples) holding the capacitance in
        pF sampled at times sampling_rate**-1, 2/sampling_rate, ... .

    Raises
    ------
    ValueError
        If any argument is outside the stated domain.
    """
    return transients  # placeholder
```

### Step 3

03_bin_transient_logarithmic

Goal
----
Return the logarithmically binned form of one capacitance transient, giving for each retained bin its mean sampling time, its mean capacitance and the number of raw samples it averages.

```python
import numpy as np

def bin_transient_logarithmic(times: np.ndarray, capacitance: np.ndarray,
                              n_bins: int) -> np.ndarray:
    """Return one capacitance transient re-expressed on a logarithmic time axis.

    Parameters
    ----------
    times : np.ndarray
        Strictly increasing, strictly positive sampling times in s.
    capacitance : np.ndarray
        Capacitance in pF sampled at those times, same length as times.
    n_bins : int
        Number of logarithmically spaced intervals spanning the first to the
        last sampling time (n_bins >= 1).

    Returns
    -------
    binned : np.ndarray
        Array of shape (n_kept, 3) whose columns are the mean sampling time of
        the bin in s, the mean capacitance of the bin in pF, and the number of
        raw samples averaged in that bin. Empty bins are omitted, so n_kept may
        be smaller than n_bins.

    Raises
    ------
    ValueError
        If any argument is outside the stated domain.
    """
    return binned  # placeholder
```

### Step 4

04_build_inversion_system

Goal
----
Return the weighted linear system whose solution is the emission-rate spectral density of one binned capacitance transient or a batch of temperature-indexed transients, as the weighted exponential kernel augmented by the weighted right-hand side.

```python
import numpy as np

def build_inversion_system(binned: np.ndarray, emission_grid: np.ndarray) -> np.ndarray:
    """Return weighted exponential kernels augmented by their right-hand sides.

    Parameters
    ----------
    binned : np.ndarray
        One record of shape (n_kept, 3), or a temperature-indexed batch of
        shape (n_records, n_kept, 3), with n_kept >= 1. The final axis holds
        bin mean time in s, bin mean capacitance in pF and raw sample count.
        Every sample count must be strictly positive.
    emission_grid : np.ndarray
        Emission rates in 1/s at which the spectral density is discretised, as
        a non-empty one-dimensional array whose entries are all finite and
        strictly positive.

    Returns
    -------
    system : np.ndarray
        For one record, shape (n_kept, n_rates + 1); for a batch, shape
        (n_records, n_kept, n_rates + 1). The first n_rates entries of the
        final axis are the weighted exponential kernel and the last entry is
        the weighted, baseline-removed and sign-corrected transient. Batch
        records are processed independently and retain their input order.

    Notes
    -----
    Each row is multiplied by exactly the square root of its bin sample count.
    The common per-transient noise scale is not divided out, because doing so
    would rescale the residual relative to the fixed penalty in the next step.

    Raises
    ------
    ValueError
        If any argument is outside the stated domain.
    """
    return system  # placeholder
```

### Step 5

05_solve_regularized_spectrum

Goal
----
Return the non-negative emission-rate spectral density obtained from one weighted inversion system or a temperature-indexed batch by Tikhonov-regularized least squares.

```python
import numpy as np

def solve_regularized_spectrum(system: np.ndarray, regularization: float) -> np.ndarray:
    """Return the non-negative spectral density of the weighted inversion system.

    Parameters
    ----------
    system : np.ndarray
        One array of shape (n_kept, n_rates + 1), or a temperature-indexed
        batch of shape (n_records, n_kept, n_rates + 1). The final axis holds
        the weighted exponential kernel followed by the weighted right-hand
        side.
    regularization : float
        Tikhonov regularization parameter (regularization >= 0), in the same
        units as the right-hand side divided by the spectral density.

    Returns
    -------
    density : np.ndarray
        For one system, shape (n_rates,); for a batch, shape
        (n_records, n_rates). Each row is the independently recovered
        non-negative density for the corresponding input record.

    Raises
    ------
    ValueError
        If any argument is outside the stated domain.
    """
    return density  # placeholder
```

### Step 6

06_extract_spectral_peaks

Goal
----
Return the emission rate and the enclosed spectral weight of each of the strongest features of an emission-rate spectral density, partitioning the rate grid at the minima that separate those features.

```python
import numpy as np

def extract_spectral_peaks(density: np.ndarray, emission_grid: np.ndarray,
                           n_peaks: int) -> np.ndarray:
    """Return the rate and the enclosed weight of the strongest spectral features.

    Parameters
    ----------
    density : np.ndarray
        Non-negative spectral density sampled on the emission-rate grid.
    emission_grid : np.ndarray
        Strictly increasing, strictly positive emission rates in 1/s, same
        length as density.
    n_peaks : int
        Number of features to return (n_peaks >= 1).

    Returns
    -------
    features : np.ndarray
        Array of shape (n_found, 2) sorted by increasing emission rate, whose
        columns are the emission rate of the feature in 1/s and the spectral
        weight enclosed by it in pF. n_found is the smaller of n_peaks and the
        number of interior local maxima of the density.

    Notes
    -----
    A grid point that is the separating minimum between adjacent retained
    maxima belongs to the feature on its right. Every grid point is therefore
    included in exactly one cluster slice.

    Raises
    ------
    ValueError
        If any argument is outside the stated domain.
    """
    return features  # placeholder
```

### Step 7

07_fit_arrhenius_parameters

Goal
----
Return the trap depth and the infinite-temperature capture cross section of a level from its emission rates measured across a set of temperatures.

```python
import numpy as np

def fit_arrhenius_parameters(temperature: np.ndarray, emission_rate: np.ndarray,
                             mass_ratio: float = 0.063) -> np.ndarray:
    """Return the trap depth and infinite-temperature cross section of a level.

    Parameters
    ----------
    temperature : np.ndarray
        Temperatures in K at which the emission rate was determined, all
        strictly positive and at least two distinct values.
    emission_rate : np.ndarray
        Emission rate in 1/s at each temperature, all strictly positive, same
        length as temperature.
    mass_ratio : float
        Carrier effective mass of the receiving band in units of the free
        electron mass (mass_ratio > 0).

    Returns
    -------
    parameters : np.ndarray
        Array of shape (2,) holding the trap depth in eV and the
        infinite-temperature capture cross section in cm^2.

    Raises
    ------
    ValueError
        If any argument is outside the stated domain.
    """
    return parameters  # placeholder
```

### Step 8

08_fit_constrained_amplitudes

Goal
----
Return the individual capacitance deflections and quiescent capacitance of one binned transient or a temperature-indexed batch, obtained by independent weighted linear least-squares fits with the emission rates held fixed.

```python
import numpy as np

def fit_constrained_amplitudes(binned: np.ndarray, rates: np.ndarray) -> np.ndarray:
    """Return the level deflections and the quiescent capacitance at fixed rates.

    Parameters
    ----------
    binned : np.ndarray
        One record of shape (n_kept, 3), or a temperature-indexed batch of
        shape (n_records, n_kept, 3). The final axis holds bin mean time in s,
        bin mean capacitance in pF and raw sample count.
    rates : np.ndarray
        For one record, shape (n_rates,); for a batch, shape
        (n_records, n_rates). These are the emission rates in 1/s at which the
        exponential components are held, all strictly positive. Each record
        must have at least n_rates + 1 binned points.

    Returns
    -------
    components : np.ndarray
        For one record, shape (n_rates + 1,); for a batch, shape
        (n_records, n_rates + 1). The final axis holds the magnitude of each
        level deflection in pF, in rate order, followed by that record's fitted
        quiescent capacitance in pF.

    Raises
    ------
    ValueError
        If any argument is outside the stated domain, including when there are
        fewer binned points than fitted coefficients (rates.size + 1).
    """
    return components  # placeholder
```

### Step 9

09_compute_defect_concentration

Goal
----
Return the apparent trap concentration of each level from its capacitance deflection, the shallow doping density of the probed layer and the quiescent capacitance at the reverse bias, under an explicitly supplied multiplicative partial-probing correction.

```python
import numpy as np

def compute_defect_concentration(deflection: np.ndarray, doping_density: float,
                                 base_capacitance: float,
                                 correction: float = 1.0) -> np.ndarray:
    """Return the apparent trap concentration from each capacitance deflection.

    Parameters
    ----------
    deflection : np.ndarray
        Capacitance deflection of each level in pF; the sign is ignored.
    doping_density : float
        Shallow donor or acceptor density of the probed layer in 1/cm^3
        (doping_density > 0).
    base_capacitance : float
        Quiescent capacitance at the reverse bias in pF
        (base_capacitance > 0), in the same units as deflection.
    correction : float
        Multiplicative dimensionless partial-probing correction (correction >
        0). The returned concentration is
        2*doping_density*abs(deflection)*correction/base_capacitance. The
        benchmark passes exactly 1.0, so its reported concentrations are
        uncorrected.

    Returns
    -------
    concentration : np.ndarray
        Apparent trap concentration of each level in 1/cm^3 under the supplied
        common multiplicative correction.

    Raises
    ------
    ValueError
        If any argument is outside the stated domain.
    """
    return concentration  # placeholder
```

### Step 10

10_average_active_window

Goal
----
Return the mean and the relative scatter of each level's capacitance deflection over the temperatures of the active window, from the deflections recovered at every measured temperature.

```python
import numpy as np

def average_active_window(temperature: np.ndarray, deflection: np.ndarray,
                          window_low: float, window_high: float) -> np.ndarray:
    """Return the mean and relative scatter of each level's deflection in a window.

    Parameters
    ----------
    temperature : np.ndarray
        Temperatures in K at which the deflections were recovered, of length
        n_temperatures.
    deflection : np.ndarray
        Array of shape (n_temperatures, n_levels) holding the magnitude of the
        capacitance deflection of each level at each temperature, in pF. An
        identically zero level is valid and must not raise an exception.
    window_low : float
        Lower bound of the active window in K, inclusive.
    window_high : float
        Upper bound of the active window in K, inclusive
        (window_high >= window_low).

    Returns
    -------
    summary : np.ndarray
        Array of shape (n_levels, 2) whose columns are the mean capacitance
        deflection of the level over the window in pF and the standard
        deviation of that deflection divided by its mean. If a level's window
        mean is exactly zero, its relative scatter is np.inf.

    Raises
    ------
    ValueError
        If any argument is outside the stated domain.
    """
    return summary  # placeholder
```

### Step 11

11_run_dlts_concentration_pipeline

Goal
----
Chain the sub-problem functions 01-10 end to end on a synthetic multi-level capacitance data set and return the ratio of the largest recovered apparent trap concentration to the smallest, with every partial-probing correction set to one.

```python
import numpy as np

def run_dlts_concentration_pipeline(temperature: np.ndarray = None,
                                    activation_energy: tuple = (0.711, 0.658),
                                    sigma_inf: tuple = (1.8e-15, 9.1e-15),
                                    deflection: tuple = (-0.75, -0.05),
                                    base_capacitance: float = 204.5,
                                    doping_density: float = 2.0e16,
                                    sampling_rate: float = 1.0e5,
                                    n_samples: int = 100000,
                                    noise_ref: float = 0.0032,
                                    temperature_ref: float = 350.0,
                                    seed: int = 20260722,
                                    n_bins: int = 100,
                                    rate_min: float = 0.1,
                                    rate_max: float = 1.0e5,
                                    n_rates: int = 150,
                                    regularization: float = 0.01,
                                    window_low: float = 370.0,
                                    window_high: float = 450.0,
                                    mass_ratio: float = 0.063) -> float:
    """Return the largest-to-smallest recovered apparent concentration ratio.

    Parameters
    ----------
    temperature : np.ndarray
        Measured temperatures in K, in acquisition order. None selects the
        benchmark grid, 300 to 480 K inclusive in steps of 3 K.
    activation_energy : tuple
        Trap depth of each level in eV used to synthesise the data. Must
        contain at least two values.
    sigma_inf : tuple
        Infinite-temperature capture cross section of each level in cm^2 used
        to synthesise the data. Must have the same length as activation_energy.
    deflection : tuple
        Capacitance deflection of each level in pF used to synthesise the data.
        Must have the same length as activation_energy.
    base_capacitance : float
        Quiescent capacitance at the reverse bias in pF.
    doping_density : float
        Shallow doping density of the probed layer in 1/cm^3.
    sampling_rate : float
        Sampling rate of the acquisition in Hz.
    n_samples : int
        Number of samples per transient.
    noise_ref : float
        Noise standard deviation in pF at the reference temperature.
    temperature_ref : float
        Reference temperature in K for the noise scaling.
    seed : int
        Seed of the NumPy default random generator.
    n_bins : int
        Number of logarithmically spaced time intervals used to bin each
        transient.
    rate_min, rate_max : float
        Bounds in 1/s of the geometric emission-rate grid.
    n_rates : int
        Number of points on the emission-rate grid.
    regularization : float
        Tikhonov regularization parameter of the inversion.
    window_low, window_high : float
        Bounds in K of the active temperature window.
    mass_ratio : float
        Carrier effective mass of the receiving band in units of the free
        electron mass.

    Returns
    -------
    ratio : float
        Largest recovered apparent trap concentration divided by the smallest
        across all supplied levels, with every partial-probing correction set
        to one. For the two-level benchmark this is the stronger-to-weaker
        ratio requested in the problem statement.

    Raises
    ------
    ValueError
        If fewer than two levels are supplied, the level-parameter arrays have
        different lengths, or any other argument is outside the stated domain.
    """
    return ratio  # placeholder
```
