# Biology-Biochemistry-41

## Background

In amphibian skeletal muscle, intracellular calcium release often appears as brief, localized events (calcium sparks) that sit between the scale of a single channel and a whole-cell transient. Whether one spark is the opening of one release channel, of an entire cluster at once, or of several channels recruited one after another has been debated for decades, and the answer determines how the event amplitude should be read: as a fixed quantum, as an all-or-none cluster response, or as the sum of a variable number of recruited elements. The distinction is only visible when the imaging is fast enough to resolve the event rising phase, because slower scans average any internal structure away. 

The laboratory behind the analysis procedure used here studies these events in frog skeletal muscle with high-speed line scans. Its spatial Gaussian fit turns each event into two time series (with a normalized amplitude and the width of the fluorescence profile) and those two series support a two linked analyses: 
- a step analysis that resolves how many quantal release steps compose the event, 
- a signal-mass analysis that measures the mass of the release signal, whose time derivative is proportional to the net release flux through the open channels. 

The task supplies both fitted series for one synthetic event and asks for the signal-mass increment carried by one release step.

## Problem

When the skeletal muscle contracts, calcium is released from an intracellular store. This single local release event is named a calcium spark and is a brief rise in cytosolic calcium. In one class of vertebrate muscle, high-speed line-scan imaging shows the rising phase of a spark as a staircase of discrete, nearly equal increments: the event amplitude sums over however many small channel groups opened. A recent study of these events specified an analysis procedure that reconstructs the number of release steps composing a single event from its digitized fluorescence record, and quantified the mass of the release signal that produces it.

Your task is to reproduce one deterministic instance of that analysis. For one event, compute the signal-mass increment corresponding to a single release step, in F/F0 x um^3, and report it to four decimal places. Using the recent literature, identify the study whose acquisition settings and analysis conventions match the supplied configuration and apply its procedure.

The studs analysis consumes two Gaussian-fitted series per event, both supplied below. The amplitude series is normalized fluorescence (F/F0, resting level 1.0000); the FWHM series is the full width at half maximum of the spatial fluorescence profile, in um. Both are digitized at the study's acquisition setting, dt = 0.0154 ms, and both are given as integer counts (amplitude in 1e-4 F/F0, FWHM in 1e-3 um) in time order. Each series follows the study own convention for samples whose spatial fit did not converge. Every convention that turns these two series into the requested quantity - the portion of the record each quantity is computed over, how the step levels are located, any correction applied to the step count, and the signal-mass definition - follows the matching published study.

The supplied amplitude series (integer counts of 1e-4 F/F0, in time order):

[
  10169, 10229, 10255, 10358, 10426, 10517, 10709, 10595, 10642, 10572, 10687, 10654, 10692, 10572, 10619, 10731,
  10701, 10657, 10716, 10592, 10719, 10640, 10571, 10554, 10553, 10713, 10589, 10799, 10608, 10670, 10664, 10690,
  10596, 10619, 10602, 10637, 10580, 10560, 10647, 10722, 10679, 10578, 10621, 10643, 10653, 10637, 10623, 10733,
  10649, 10634, 10541, 10594, 10565, 10629, 10543, 10575, 10489, 10588, 10529, 10680, 10698, 10759, 10719, 10881,
  10973, 11124, 11181, 11272, 11199, 11185, 11292, 11179, 11164, 11215, 11229, 11179, 11251, 11445, 11311, 11193,
  11378, 11267, 11259, 11258, 11257, 11247, 11227, 11179, 11189, 11186, 11339, 11225, 11238, 11258, 11198, 11209,
  11189, 11190, 11209, 11259, 11329, 11299, 11198, 11270, 11178, 11221, 11364, 11233, 11271, 11253, 11245, 11212,
  11237, 11344, 11217, 11203, 11380, 11120, 11130, 11238, 11177, 11269, 11383, 11289, 11447, 11623, 11825, 11768,
  11867, 11795, 11844, 11886, 11897, 11882, 11846, 11794, 11909, 11845, 11820, 11853, 11898, 11780, 11842, 11896,
  11828, 11726, 11821, 11896, 11891, 11885, 11900, 11806, 11826, 11860, 11883, 11780, 11884, 11880, 11873, 11871,
  11914, 11755, 11890, 11777, 11911, 11920, 11766, 11812, 11817, 11886, 11927, 11832, 11825, 11809, 11804, 11854,
  11880, 11777, 11826, 11882, 11860, 11888, 11858, 12010, 12132, 12437, 12633, 12717, 12836, 13098, 13097, 13131,
  13047, 13156, 13124, 13234, 13067, 13103, 13162, 13039, 13146, 13180, 13128, 13092, 13173, 13003, 12973, 13036,
  13067, 13091, 13064, 13200, 13149, 13036, 13045, 13110, 13147, 13141, 13138, 13053, 13061, 13129, 13120, 13166,
  13002, 13082, 13131, 13071, 13084, 13060, 13063, 13135, 13144, 13020, 13090, 13148, 13109, 13097, 13089, 13099,
  13132, 13134, 13028, 13189, 13116, 13370, 13331, 13343, 13560, 13546, 13665, 13788, 13735, 13684, 13774, 13753,
  13682, 13688, 13749, 13649, 13599, 13796, 13717, 13700, 13681, 13782, 13685, 13662, 13851, 13853, 13709, 13660,
  13690, 13792, 13759, 13831, 13661, 13775, 13702, 13672, 13642, 13737, 13645, 13757, 13774, 13653, 0, 0,
  0, 0, 0, 0, 0, 0, 13758, 13691, 13796, 13728, 13748, 13664, 13802, 13694, 13806, 13721,
  13620, 13735, 13755, 13679, 13776, 13722, 13695, 13746, 13647, 13689, 13735, 13654, 13853, 13710, 13758, 13692,
  13654, 13737, 13765, 13687, 13698, 13722, 13743, 13766, 13672, 13757, 13692, 13755, 13812, 13832, 13664, 13772,
  13672, 13713, 13699, 13733, 13677, 13659, 13645, 13695, 13708, 13785, 13815, 13778, 13737, 13830, 13684, 13635,
  13675, 13736, 13658, 13647, 13687, 13722, 13800, 13873, 13670, 13744, 13592, 13692, 13589, 13576, 13585, 13594,
  13724, 13541, 13538, 13534, 13493, 13559, 13470, 13528, 13512, 13508, 13521, 13519, 13391, 13493, 13397, 13457,
  13422, 13419, 13384, 13268, 13411, 13224, 13240, 13358, 13190, 13315, 13217, 13330, 13161, 13184, 13134, 13000,
  13234, 13256, 13154, 13195, 13181, 13147, 13060, 13095, 13056, 13012, 12983, 13051, 13051, 13055, 13116, 13029,
  12933, 12951, 12791, 12907, 12903, 12953, 12968, 12957, 12822
]

The supplied FWHM series (integer counts of 1e-3 um, same samples, same order):

[
  1494, 1486, 1557, 1450, 1507, 1418, 1537, 1559, 1537, 1532, 1481, 1541, 1470, 1539, 1533, 1546,
  1574, 1542, 1514, 1626, 1481, 1542, 1597, 1539, 1562, 1439, 1600, 1546, 1563, 1588, 1587, 1599,
  1571, 1590, 1628, 1563, 1563, 1605, 1518, 1603, 1568, 1507, 1648, 1539, 1516, 1609, 1540, 1712,
  1522, 1597, 1669, 1592, 1612, 1667, 1594, 1553, 1583, 1706, 1580, 1557, 1605, 1620, 1660, 1664,
  1647, 1658, 1713, 1682, 1634, 1640, 1585, 1696, 1611, 1655, 1629, 1684, 1676, 1640, 1648, 1696,
  1644, 1742, 1662, 1586, 1708, 1700, 1702, 1742, 1713, 1668, 1674, 1663, 1693, 1685, 1696, 1766,
  1648, 1696, 1678, 1775, 1800, 1687, 1685, 1756, 1742, 1688, 1717, 1815, 1731, 1714, 1733, 1820,
  1741, 1812, 1751, 1750, 1848, 1800, 1728, 1813, 1740, 1793, 1856, 1764, 1823, 1755, 1801, 1760,
  1809, 1805, 1786, 1838, 1835, 1764, 1812, 1754, 1746, 1774, 1775, 1813, 1783, 1767, 1842, 1798,
  1763, 1775, 1782, 1868, 1820, 1911, 1836, 1872, 1894, 1823, 1938, 1863, 1856, 1855, 1904, 1965,
  1821, 1817, 1912, 1842, 1862, 1893, 1898, 1903, 1915, 1828, 1870, 1855, 1859, 1861, 1962, 1811,
  1893, 1899, 1898, 1907, 1892, 1886, 1857, 1877, 1855, 1879, 1910, 1948, 1868, 1865, 1863, 1919,
  1900, 1941, 1934, 1934, 1958, 1998, 1976, 1980, 1996, 1953, 1897, 1903, 1966, 2019, 1965, 2025,
  1994, 1971, 1951, 1998, 1922, 2004, 1967, 1974, 1993, 1962, 2001, 1993, 1994, 2047, 1963, 2011,
  1995, 2063, 2015, 2068, 2019, 1956, 2001, 1968, 2034, 1993, 2041, 2058, 2008, 2044, 2038, 2033,
  2044, 2052, 2034, 1993, 2129, 2016, 1991, 2100, 2089, 2007, 2117, 2041, 2033, 2082, 2016, 2012,
  2072, 2072, 2097, 2015, 1998, 2044, 2028, 1989, 2062, 2114, 2096, 2105, 2095, 2064, 2115, 2104,
  2083, 2067, 2096, 2086, 2152, 2122, 2141, 2059, 2054, 2106, 2093, 2168, 2146, 2084, 0, 0,
  0, 0, 0, 0, 0, 0, 2182, 2203, 2159, 2163, 2217, 2245, 2203, 2154, 2260, 2157,
  2275, 2178, 2221, 2196, 2203, 2225, 2245, 2177, 2142, 2250, 2189, 2167, 2168, 2138, 2205, 2205,
  2256, 2167, 2239, 2196, 2227, 2151, 2133, 2241, 2315, 2232, 2180, 2267, 2241, 2267, 2255, 2243,
  2306, 2212, 2255, 2224, 2259, 2216, 2226, 2185, 2297, 2275, 2260, 2217, 2233, 2293, 2248, 2307,
  2329, 2313, 2269, 2239, 2315, 2240, 2313, 2299, 2290, 2229, 2288, 2267, 2343, 2272, 2382, 2266,
  2297, 2270, 2289, 2273, 2344, 2258, 2305, 2280, 2271, 2247, 2303, 2321, 2348, 2270, 2319, 2292,
  2255, 2342, 2270, 2343, 2399, 2290, 2325, 2309, 2256, 2332, 2282, 2337, 2341, 2295, 2264, 2407,
  2269, 2320, 2302, 2257, 2293, 2278, 2299, 2331, 2248, 2295, 2309, 2323, 2296, 2302, 2248, 2284,
  2308, 2304, 2270, 2397, 2285, 2321, 2310, 2267, 2303
]

Numerical conventions:
 
* Divide the amplitude counts by 10000 and the FWHM counts by 1000 to obtain F/F0 and um.
* Compute the requested per-step scalar by evaluating the study's signal-mass expression for one event-specific quantal amplitude step, using the mean FWHM over the study-defined signal-mass interval.
* The result is a single positive decimal number, reported to four decimal places.
* The adopted analysis conventions - the portion of the record each quantity is computed over, the method used to locate the step levels, any correction applied to the step count, and the signal-mass definition - all follow the matching published study.
* In your reasoning, report the adopted bin width and its justification, the start of the signal-mass interval and how it was fixed, the interval length in samples and milliseconds, the mean FWHM over the interval, the signal mass at the interval start and its maximum over the interval, the lowest resolved step level, this event's quantal step size, the number of release steps composing the event, and the signal-mass increment per step.



Output Format Requirements:
Emit <final_answer> immediately, then <reasoning>. Do not write a long derivation before the tags.
You must emit exactly one finite decimal inside <final_answer>...</final_answer>, even if the value is approximate or you are unsure.
Rules:
- The tags are required. Do not omit them or leave them empty.
- The tagged value must be a finite decimal (examples: 0.4847, 0.22, 1.05). Not NaN, not Inf, not a fraction string, not a vector, not prose.
- Put only that one number between the tags. No units, no words, no extra lines.
Keep <reasoning> short (a few hundred words). Show only the few scalars that determine the final number.
Do not paste the input matrices, full coefficient vectors, per-iteration paths, or per-fold candidate tables.
Do not reproduce the supplied series.

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

01_decode_serie

Goal
----
Step 1 - decode the two digitized series.

```python
def decode_series(
    amp_counts: "numpy.typing.ArrayLike",
    fwhm_counts: "numpy.typing.ArrayLike",
    dt_ms: float = 0.0154,
    amp_unit: float = 1.0e-4,
    fwhm_unit: float = 1.0e-3,
) -> "tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray]":
    """Decode the amplitude and FWHM count series to normalized units.

    Parameters
    ----------
    amp_counts : sequence of int
        Amplitude in integer multiples of `amp_unit` (F/F0, so 10000 is the resting
        level 1.0000), in time order. A zero marks a sample whose spatial fit did
        not converge.
    fwhm_counts : sequence of int
        Full width at half maximum in integer multiples of `fwhm_unit` (um), same
        samples and same failure-marker convention.
    dt_ms : float, optional
        Sampling interval in milliseconds.
    amp_unit : float, optional
        Size of one amplitude count in F/F0.
    fwhm_unit : float, optional
        Size of one FWHM count in um.

    Returns
    -------
    tuple of numpy.ndarray
        ``(t, amp, fwhm, valid)``: time in milliseconds starting at 0.0 for the first
        sample; amplitude in F/F0; FWHM in um; and a boolean array, True at samples
        where both counts are positive, i.e. where the spatial fit converged and the
        sample is a measurement.

    Raises
    ------
    ValueError
        If the two count series do not share a non-empty 1-D shape, if any count is
        non-finite, non-integer or negative, or if `dt_ms`, `amp_unit` or `fwhm_unit`
        is not positive and finite.
    """
    return result
```

### Step 2

02_signal_mass_interval

Goal
----
Step 2 - the signal-mass interval.

```python
def signal_mass_interval(
    amp: "numpy.ndarray",
    fwhm: "numpy.ndarray",
    valid: "numpy.ndarray",
    baseline: float = 1.0,
    peak_fraction: float = 0.20,
    tail_ms: float = 1.0,
    dt_ms: float = 0.0154,
) -> "tuple[int, int]":
    """Resolve the start and end indices of the signal-mass interval.

    Parameters
    ----------
    amp : numpy.ndarray
        Amplitude in F/F0 (see step 01).
    fwhm : numpy.ndarray
        FWHM in um, same shape as `amp`.
    valid : numpy.ndarray of bool
        Converged-sample mask, same shape as `amp` (see step 01).
    baseline : float, optional
        Normalized resting level the amplitude threshold is measured above.
    peak_fraction : float, optional
        Fraction of the peak amplitude above baseline that criterion (a) requires.
    tail_ms : float, optional
        Milliseconds the interval extends past the peak sample.
    dt_ms : float, optional
        Sampling interval in milliseconds.

    Returns
    -------
    tuple of int
        ``(ts1, ts2)``: the first and last sample indices of the interval, inclusive.
        ts1 is the earliest sample satisfying criterion (a) that has no later failure
        marker before the peak, and ts2 is the sample `tail_ms` after the peak.

    Raises
    ------
    ValueError
        If the three arrays do not share a non-empty 1-D shape, if any amplitude or
        FWHM entry is non-finite, if the parameters are out of range, if the record
        has no converged sample, if the peak is not a converged sample or does not
        rise above the baseline, or if the record does not extend `tail_ms` past the
        peak.
    """
    return result
```

### Step 3

03_signal_mass_curve

Goal
----
Step 3 - the signal-mass curve over the interval.

```python
def signal_mass_curve(
    amp: "numpy.ndarray",
    fwhm: "numpy.ndarray",
    valid: "numpy.ndarray",
    ts1: int,
    ts2: int,
    baseline: float = 1.0,
    k: float = 1.206,
) -> "tuple[numpy.ndarray, bool]":
    """Evaluate the signal-mass curve over an inclusive interval.

    Parameters
    ----------
    amp : numpy.ndarray
        Amplitude in F/F0 (see step 01).
    fwhm : numpy.ndarray
        FWHM in um, same shape as `amp`.
    valid : numpy.ndarray of bool
        Converged-sample mask, same shape as `amp`.
    ts1, ts2 : int
        First and last sample indices of the interval, inclusive (see step 02).
    baseline : float, optional
        Normalized resting level. The signal-mass amplitude is the spark's height
        above this level, so it is subtracted from the supplied series.
    k : float, optional
        The study's signal-mass coefficient, in F/F0^-1 um^-3.

    Returns
    -------
    tuple
        ``(sm, continuous)``: `sm` is a float array of length ``ts2 - ts1 + 1``
        holding SM in F/F0 x um^3 at each sample of the interval, with NaN at
        samples where either series is unavailable; `continuous` is True exactly
        when every sample of the interval is a converged measurement.

    Raises
    ------
    ValueError
        If the arrays do not share a non-empty 1-D shape, if any entry is
        non-finite, if the interval indices are outside the record or reversed, or
        if `baseline` or `k` is not finite (`k` must also be positive).
    """
    return result
```

### Step 4

04_interval_statistics

Goal
----
Step 4 - the interval statistics behind the per-step increment.

```python
def interval_statistics(
    amp: "numpy.ndarray",
    fwhm: "numpy.ndarray",
    valid: "numpy.ndarray",
    ts1: int,
    ts2: int,
    baseline: float = 1.0,
    k: float = 1.206,
) -> "tuple[int, float, float, float]":
    """Collect the signal-mass interval's summary statistics.

    Parameters
    ----------
    amp : numpy.ndarray
        Amplitude in F/F0 (see step 01).
    fwhm : numpy.ndarray
        FWHM in um, same shape as `amp`.
    valid : numpy.ndarray of bool
        Converged-sample mask, same shape as `amp`.
    ts1, ts2 : int
        First and last sample indices of the interval, inclusive (see step 02).
    baseline : float, optional
        Normalized resting level, subtracted from the amplitude before the signal
        mass is formed (see step 03).
    k : float, optional
        The study's signal-mass coefficient.

    Returns
    -------
    tuple
        ``(n_samples, mean_fwhm, sm_at_start, sm_max)``: the interval's length in
        samples (int); the mean FWHM over its converged samples, in um (float); the
        signal mass at its first sample, in F/F0 x um^3 (float); and the largest
        signal mass over the interval (float).

    Raises
    ------
    ValueError
        If the inputs fail the validation of steps 01-03, if the interval holds no
        converged sample, if its first sample is not converged, or if its indices
        lie outside the record or are reversed.
    """
    return result
```

### Step 5

05_reference_bin_width

Goal
----
Step 5 - the histogram scale: Scott's rule at the study's background noise.

```python
def reference_bin_width(
    sigma_background: float,
    n_active: int,
    tested: "numpy.typing.ArrayLike" = (0.015, 0.020, 0.025),
) -> "tuple[float, float]":
    """Evaluate Scott's normal-reference rule and select the tested bin it lands on.

    Parameters
    ----------
    sigma_background : float
        Standard deviation of the acquisition's raw spark-free background, in F/F0.
    n_active : int
        Number of samples in the analyzed segment.
    tested : sequence of float, optional
        The candidate bin widths the study tested, in F/F0.

    Returns
    -------
    tuple of float
        ``(scott_raw, adopted)``: Scott's normal-reference bin width
        ``3.49 * sigma_background * n_active ** (-1/3)``, and the member of `tested`
        closest to it. Distance is absolute; an exact tie selects the smaller member.

    Raises
    ------
    ValueError
        If `sigma_background` is not positive and finite, `n_active` is not a positive
        integer, or `tested` is empty or holds a value that is not positive and finite.
    """
    return result
```

### Step 6

06_level_histogram

Goal
----
Step 6 - the level histogram of the analyzed amplitudes.

```python
def level_histogram(
    amp_act: "numpy.ndarray",
    bin_width: float = 0.015,
    anchor: float = 1.0,
) -> "tuple[numpy.ndarray, numpy.ndarray]":
    """Bin the activation-phase amplitudes at the adopted width.

    Parameters
    ----------
    amp_act : numpy.ndarray
        Analyzed amplitudes in F/F0: the converged samples of the activation phase,
        through the peak (see steps 01-02).
    bin_width : float, optional
        Histogram bin width in F/F0 (the study's adopted value).
    anchor : float, optional
        Level the bin edges are anchored to (the resting baseline), so that bin
        alignment is independent of the record's noise.

    Returns
    -------
    tuple of numpy.ndarray
        ``(counts, centers)``: integer bin counts and the bin centers in F/F0.

    Raises
    ------
    ValueError
        If the input is empty or non-finite, or `bin_width` is not positive.
    """
    return result
```

### Step 7

07_resolve_levels

Goal
----
Step 7 - resolve the step levels: histogram modes refined by a Gaussian fit.

```python
def resolve_step_levels(
    counts: "numpy.ndarray",
    centers: "numpy.ndarray",
    bin_width: float = 0.015,
    prominence: float = 5.0,
) -> "numpy.ndarray":
    """Resolve the fluorescence step levels from a level histogram.

    Parameters
    ----------
    counts : numpy.ndarray
        Integer bin counts of the level histogram (see step 06).
    centers : numpy.ndarray
        Bin centers in F/F0, same length as `counts`.
    bin_width : float, optional
        Histogram bin width (the study's adopted value).
    prominence : float, optional
        Minimum prominence (in counts) a histogram mode must have to count as a
        level.

    Returns
    -------
    numpy.ndarray
        The sorted step-level positions in F/F0 (Gaussian-refined when the fit
        converges; otherwise the raw mode centers).

    Raises
    ------
    ValueError
        If the histogram inputs are misaligned, empty, or non-finite, or if
        `bin_width` or `prominence` is out of range.
    """
    return result
```

### Step 8

08_merge_correction

Goal
----
Step 8 - the double-quantum (merge) correction of the step count.

```python
def apply_merge_correction(
    levels: "numpy.ndarray",
    factor: float = 1.5,
) -> "tuple[int, float]":
    """Apply the double-quantum merge correction to the resolved step levels.

    Parameters
    ----------
    levels : numpy.ndarray
        Sorted step-level positions in F/F0 (see step 07).
    factor : float, optional
        A consecutive-level gap strictly greater than `factor` times the median
        gap is read as two steps.

    Returns
    -------
    tuple
        ``(step_count, step_size)``: the merge-corrected number of release steps
        (int) and the mean corrected inter-level gap in F/F0 (float; 0.0 when a
        single level leaves no gap).

    Raises
    ------
    ValueError
        If `levels` is empty or non-finite, or `factor` is out of range.
    """
    return result
```

### Step 9

09_signal_mass_increment.py

Goal
----
Step 9 -  The signal-mass increment per release step.

```python
def signal_mass_increment(
    amp_counts: "numpy.typing.ArrayLike",
    fwhm_counts: "numpy.typing.ArrayLike",
    dt_ms: float = 0.0154,
    sigma_background: float = 0.0346,
) -> float:
    """Per-release-step signal-mass increment of one event.

    Parameters
    ----------
    amp_counts : sequence of int
        Amplitude in integer multiples of 1e-4 F/F0, in time order (see step 01).
    fwhm_counts : sequence of int
        FWHM in integer multiples of 1e-3 um, same samples (see step 01).
    dt_ms : float, optional
        Sampling interval in milliseconds.
    sigma_background : float, optional
        Standard deviation of the acquisition's raw spark-free background, in F/F0,
        from which the histogram bin width is fixed (see step 05).

    Returns
    -------
    float
        The signal-mass increment per release step, in F/F0 x um^3.

    Raises
    ------
    ValueError
        If the record fails the input validation of the pipeline steps.
    """
    return result
```
