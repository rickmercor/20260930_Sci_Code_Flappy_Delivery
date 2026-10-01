# Biology-Ecology-49

## Background

Environmental DNA, the genetic material that animals shed into the water around them, has become a standard way to tell whether a fish is present in a river, and it is increasingly used to follow seasonal migrations without catching or handling anything. Its weakness is that a single water sample is a noisy witness: concentrations at one place swing by orders of magnitude from day to day, so a negative result is weak evidence of absence. Monitoring programmes therefore need to know how likely a visit is to miss a species that is there, and when in a migration season a visit is worth making at all.

## Problem

Environmental DNA surveys of a river fish can come back negative while the species is present, because the concentration a water sample assays fluctuates far more than its seasonal mean. Consider a reach 20 km long, with x measured in km upstream from a sampling station at x = 0: water flows towards x = 0 at 20 km per day, no eDNA enters at the top of the reach, and the reach holds none on day 0. Migrants pass x = 0 during a season of 80 days and swim upstream at a ground speed of 2 km per day, so a migrant that passes x = 0 on day t is at x on day t + x/2, and it sheds eDNA wherever it is. The concentration Y_t(x), in copies per ml, satisfies

    dY = ( 20 dY/dx + 0.012 Z_{t - x/2} - 1.5 Y ) dt + 1.2 sqrt(Y) W(dt, dx),

with W space-time white noise on the reach. Treat the reach as continuous in space and do not replace the noise by a spatial grid. Here Z_t is the unit-time migrant count in fish per day: the nonnegative diffusion dZ = ( a(t) - 30 Z/(80 - t) ) dt + sqrt( D(t) Z ) dB with Z = 0 at t = 0 and at t = 80, whose mean is 1200 (th/th_pk)^6 ((1 - th)/(1 - th_pk))^8 with th = t/80 and th_pk = 6/14, and whose coefficient of variation is 1.8 at every time strictly inside the season; a(t) and D(t) are the continuous pair that gives the process that mean and that coefficient of variation.

One replicate sample at the station assays the water that passes it during a single collection interval, which at the flow speed is the lowest 1 km of the reach at the sampling instant; it recovers a Poisson number of amplifiable copies whose mean is 1.30 ml times the mean of Y over that kilometre, and replicates are conditionally independent given the field. An upstream site samples the kilometre from 2 km to 3 km in the same way. A visit fails when no replicate at any of its sites records a copy.

The monitoring standard is that three replicates at the station alone must fail with probability at most 0.05, and the visit is made on the first whole day from 1 to 79 that meets it. The station keeps its three replicates on that visit, and at the upstream site the number of replicates is the smallest whole number whose expected copy yield on that day, from the expected concentration of the water sampled there, is at least 6 copies.

In the reasoning, report for day 20 the expected concentration of the water a station sample assays and the failure probability of three replicates there; the oldest migration age in days that can leave eDNA in any station sample; the first and the last whole day meeting the standard, and the station failure probability on the first of them; and, on that day, the expected concentration of the water the upstream sample assays and the number of replicates the rule sets there. State briefly how you fixed the coefficients of the count process, how eDNA already in the reach is carried into a sample, how you averaged over the random run, and what the two sites share. Your final answer is the probability that the planned two-site visit records no copies at all.

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

01_migration_rate_coefficients.py

Goal
----
Drift source and squared diffusion coefficient of the unit-time migrant count.

```python
def migration_rate_coefficients(times: "np.ndarray", run: "np.ndarray") -> "np.ndarray":
    '''Drift source and squared diffusion coefficient of the unit-time migrant count.

    The unit-time count Z_t, in fish per day, is the nonnegative diffusion

        dZ = (a(t) - pull * Z / (season - t)) dt + sqrt(D(t) * Z) dB,

    on 0 < t < season, with Z = 0 at t = 0 and at t = season. Its mean follows the
    seasonal profile

        mean(t) = peak * (th/th_pk)**shape_up * ((1-th)/(1-th_pk))**shape_down,
        th = t/season,  th_pk = shape_up/(shape_up + shape_down),

    and its coefficient of variation equals cv at every time strictly inside the
    season. Exactly one pair of continuous functions (a, D) gives the process that
    mean and that coefficient of variation; return that pair.

    Parameters
    ----------
    times : np.ndarray
        Shape (n,), times in days at which the coefficients are wanted, each a
        finite value with 0 < t < season.
    run : np.ndarray
        Shape (6,), the run description in the order
        [season length in days, peak of the mean count in fish per day, shape_up,
        shape_down, coefficient of variation, pull]. The season length, the peak,
        the coefficient of variation and the pull are positive; shape_up and
        shape_down are greater than 1.

    Returns
    -------
    coefficients : np.ndarray
        Shape (2, n) of floats. Row 0 holds a(t) in fish per day squared, row 1
        holds D(t) in fish per day squared, in the order of times.

    Raises
    ------
    ValueError
        If times is not a one-dimensional array of finite values strictly inside
        the season, or if run is not a shape (6,) array of finite values with
        positive season length, peak, coefficient of variation and pull and with
        shape_up and shape_down greater than 1.
    '''
    return coefficients  # placeholder
```

### Step 2

02_mean_sampled_concentration.py

Goal
----
Expected eDNA concentration of the water a sample assays, in copies per ml.

```python
def mean_sampled_concentration(day: float, site_start: float, window_length: float,
                               reach: "np.ndarray", run: "np.ndarray",
                               shedding_rate: float) -> float:
    '''Expected eDNA concentration of the water a sample assays, in copies per ml.

    The reach is the interval from 0 to its length in km, with x measured upstream
    from the downstream end. River water moves towards x = 0 at the flow speed;
    no eDNA enters at x = reach length and the reach holds none on day 0. Migrants
    pass x = 0 during the season and move upstream at the ground speed, so a
    migrant that passes x = 0 on day t is at x on day t + x/(ground speed). eDNA
    is shed at a rate of shedding_rate times the unit-time migrant count present
    at that place and time, and decays at the decay rate. A sample taken on the
    given day assays the water that occupies the km window from site_start to
    site_start + window_length at that instant; its concentration is the mean of
    the field over that window.

    The expected count profile is the one of step 1; the noise on the field is
    centred and does not enter this expectation.

    Parameters
    ----------
    day : float
        Survey day, a finite value with 0 < day <= season length.
    site_start : float
        Downstream edge of the sampled window in km, finite, at least 0.
    window_length : float
        Length of the sampled window in km, finite and positive, with
        site_start + window_length not exceeding the reach length.
    reach : np.ndarray
        Shape (5,), in the order [reach length in km, flow speed in km per day,
        upstream ground speed of migrants in km per day, eDNA decay rate per day,
        noise intensity]. The first four are positive; the noise intensity is not
        used here and is only required to be finite and not negative.
    run : np.ndarray
        Shape (6,), the run description of step 1.
    shedding_rate : float
        eDNA shed per ml per day per unit of the unit-time migrant count, finite
        and positive.

    Returns
    -------
    concentration : float
        Expected concentration of the sampled water in copies per ml, converged in
        the quadrature to better than 1e-10 relative, as a native Python float.

    Raises
    ------
    ValueError
        If day is not finite and inside the season, if site_start or
        window_length place the window outside the reach, if reach is not a shape
        (5,) array of finite values with positive length, speeds and decay rate
        and a nonnegative noise intensity, if run fails the step 1 contract, or if
        shedding_rate is not finite and positive.
    '''
    return concentration  # placeholder
```

### Step 3

03_sample_memory_horizon.py

Goal
----
Oldest migration age, in days, that can leave eDNA in the sampled water.

```python
def sample_memory_horizon(site_start: float, reach: "np.ndarray") -> float:
    '''Oldest migration age, in days, that can leave eDNA in the sampled water.

    Geometry and conventions are those of step 2: x runs upstream from 0 to the
    reach length, water moves towards x = 0 at the flow speed, migrants pass
    x = 0 and move upstream at the ground speed, and eDNA enters the reach only
    where a migrant is. Return the largest age s such that migrants passing x = 0
    exactly s days before the survey can still leave eDNA in the water occupying
    the sampled window at the survey instant; migration older than that cannot.
    Take the run to have been under way long enough that its own start does not
    limit the answer; the answer does not depend on the length of the window.

    Parameters
    ----------
    site_start : float
        Downstream edge of the sampled window in km, finite, at least 0 and less
        than the reach length.
    reach : np.ndarray
        Shape (5,), the reach description of step 2.

    Returns
    -------
    horizon : float
        The age in days, as a native Python float.

    Raises
    ------
    ValueError
        If site_start is not finite, is negative or is not less than the reach
        length, or if reach fails the step 2 contract.
    '''
    return horizon  # placeholder
```

### Step 4

04_washout_exponent.py

Goal
----
Laplace exponent per unit concentration carried by the water that will be sampled.

```python
def washout_exponent(backward_times: "np.ndarray", assayed_volume: float,
                     window_length: float, reach: "np.ndarray") -> "np.ndarray":
    '''Laplace exponent per unit concentration carried by the water that will be sampled.

    Geometry and conventions are those of step 2. The eDNA field obeys

        dY = (flow * dY/dx + shedding - decay * Y) dt + noise * sqrt(Y) W(dt, dx),

    with W space-time white noise on the reach, so that the field over any
    stretch of water is driven by noise independent of that over any other. A
    replicate sample recovers a Poisson number of amplifiable copies whose mean is
    assayed_volume / window_length times the integral of the field over the
    sampled window, and replicates are conditionally independent given the field.

    Fix a backward time s and condition on the whole field s days before the
    survey, with all shedding switched off from then on. The log probability that
    the survey records no copies is then an integral over the reach of a
    coefficient times the field. That coefficient vanishes on water that will have
    left the window by the survey instant and equals one and the same number on
    the water that will be inside it. Return that number at each backward time.
    At s = 0 it is minus assayed_volume divided by window_length.

    Parameters
    ----------
    backward_times : np.ndarray
        Shape (n,), ages in days before the survey, finite and not negative.
    assayed_volume : float
        Effective assayed volume of the whole survey at this window, in ml,
        finite and positive.
    window_length : float
        Length of the sampled window in km, finite and positive.
    reach : np.ndarray
        Shape (5,), the reach description of step 2, whose fifth entry is the
        noise intensity in copies**0.5 km**0.5 ml**-0.5 day**-0.5.

    Returns
    -------
    exponent : np.ndarray
        Shape (n,) of floats, in ml per km, in the order of backward_times. Every
        entry is negative.

    Raises
    ------
    ValueError
        If backward_times is not a non-empty one-dimensional array of finite
        nonnegative values, if assayed_volume or window_length is not finite and
        positive, or if reach fails the step 2 contract.
    '''
    return exponent  # placeholder
```

### Step 5

05_sample_weight_kernel.py

Goal
----
Weight each past day of the run carries in the survey's log failure probability.

```python
def sample_weight_kernel(backward_times: "np.ndarray", site_starts: "np.ndarray",
                         assayed_volumes: "np.ndarray", window_length: float,
                         reach: "np.ndarray", shedding_rate: float) -> "np.ndarray":
    '''Weight each past day of the run carries in the survey's log failure probability.

    Geometry, field and assay are those of steps 2 and 4. The survey takes one
    window of length window_length at each site in site_starts, with the effective
    assayed volume of that site's replicates given by the matching entry of
    assayed_volumes; the windows do not overlap. The survey fails when it records
    no copies anywhere.

    Conditional on the whole path of the unit-time migrant count, the log
    probability of failure is the integral over ages s from 0 to the survey day of
    kernel(s) times the count s days before the survey. Return kernel at each
    entry of backward_times, in the order given.

    Parameters
    ----------
    backward_times : np.ndarray
        Shape (n,), ages in days before the survey, finite and not negative.
    site_starts : np.ndarray
        Shape (k,), downstream edges of the windows in km, finite, not negative,
        and each with site_start + window_length inside the reach.
    assayed_volumes : np.ndarray
        Shape (k,), effective assayed volume in ml at each site, finite and
        positive.
    window_length : float
        Length of every sampled window in km, finite and positive.
    reach : np.ndarray
        Shape (5,), the reach description of step 2.
    shedding_rate : float
        eDNA shed per ml per day per unit of the unit-time migrant count, finite
        and positive.

    Returns
    -------
    kernel : np.ndarray
        Shape (n,) of floats, in copies per ml per day per unit count, in the
        order of backward_times. Every entry is negative or zero.

    Raises
    ------
    ValueError
        If backward_times is not a non-empty one-dimensional array of finite
        nonnegative values, if site_starts and assayed_volumes are not
        one-dimensional arrays of the same non-zero length, if any window falls
        outside the reach, if any assayed volume is not finite and positive, if
        window_length or shedding_rate is not finite and positive, or if reach
        fails the step 2 contract.
    '''
    return kernel  # placeholder
```

### Step 6

06_nondetection_probability.py

Goal
----
Probability that the survey records no copies at any of its sites.

```python
def nondetection_probability(day: float, site_starts: "np.ndarray",
                             assayed_volumes: "np.ndarray", window_length: float,
                             reach: "np.ndarray", run: "np.ndarray",
                             shedding_rate: float) -> float:
    '''Probability that the survey records no copies at any of its sites.

    The reach, the field, the assay and the survey layout are those of steps 2, 4
    and 5, and the unit-time migrant count is the process of step 1, with no eDNA
    in the reach on day 0 and no count outside the season. Return the
    unconditional probability of failure for a survey run on the given day.

    Parameters
    ----------
    day : float
        Survey day, a finite value with 0 < day <= season length.
    site_starts : np.ndarray
        Shape (k,), downstream edges of the windows in km, as in step 5.
    assayed_volumes : np.ndarray
        Shape (k,), effective assayed volume in ml at each site, as in step 5.
    window_length : float
        Length of every sampled window in km, finite and positive.
    reach : np.ndarray
        Shape (5,), the reach description of step 2.
    run : np.ndarray
        Shape (6,), the run description of step 1.
    shedding_rate : float
        eDNA shed per ml per day per unit of the unit-time migrant count, finite
        and positive.

    Returns
    -------
    probability : float
        The probability, between 0 and 1, converged in the time step to better
        than 1e-8 relative, as a native Python float.

    Raises
    ------
    ValueError
        If day is not finite and inside the season, if the step 5 contract on
        site_starts, assayed_volumes, window_length, reach or shedding_rate is
        broken, or if run fails the step 1 contract.
    '''
    return probability  # placeholder
```

### Step 7

07_survey_window_days.py

Goal
----
First and last whole day on which a one-site survey meets the failure standard.

```python
def survey_window_days(site_start: float, assayed_volume: float, window_length: float,
                       reach: "np.ndarray", run: "np.ndarray", shedding_rate: float,
                       failure_target: float, first_day: int, last_day: int) -> "np.ndarray":
    '''First and last whole day on which a one-site survey meets the failure standard.

    The reach, the field, the assay and the count are those of steps 1 to 6. A
    survey is acceptable on a whole day when its probability of recording no
    copies at the single window starting at site_start is at most
    failure_target. Search the whole days from first_day to last_day inclusive
    and return the first and the last acceptable one.

    Parameters
    ----------
    site_start : float
        Downstream edge of the sampled window in km, as in step 5.
    assayed_volume : float
        Effective assayed volume of the survey's replicates at that window, in ml,
        finite and positive.
    window_length : float
        Length of the sampled window in km, finite and positive.
    reach : np.ndarray
        Shape (5,), the reach description of step 2.
    run : np.ndarray
        Shape (6,), the run description of step 1.
    shedding_rate : float
        eDNA shed per ml per day per unit of the unit-time migrant count, finite
        and positive.
    failure_target : float
        The largest acceptable probability of recording no copies, finite, greater
        than 0 and less than 1.
    first_day : int
        First whole day of the search, an integer of at least 1.
    last_day : int
        Last whole day of the search, an integer of at least first_day and no
        greater than the season length.

    Returns
    -------
    window : np.ndarray
        Shape (2,) of floats, the first and the last acceptable whole day.

    Raises
    ------
    ValueError
        If failure_target is not finite, greater than 0 and less than 1, if
        first_day or last_day is not an integer with 1 <= first_day <= last_day
        and last_day no greater than the season length, if no day in the range is
        acceptable, or if any contract of steps 1, 2, 5 or 6 is broken.
    '''
    return window  # placeholder
```

### Step 8

08_planned_replicates.py

Goal
----
Replicates planned at a site, from the expected copy yield of one replicate.

```python
def planned_replicates(day: float, site_start: float, window_length: float,
                       reach: "np.ndarray", run: "np.ndarray", shedding_rate: float,
                       capture_volume: float, copy_target: float) -> int:
    '''Replicates planned at a site, from the expected copy yield of one replicate.

    One replicate assays a volume of capture_volume ml of the water in the window
    starting at site_start, so its expected copy yield is capture_volume times the
    expected concentration of that water, as in step 2. Return the smallest whole
    number of replicates whose expected yield, the number of replicates times the
    expected yield of one, is at least copy_target.

    Parameters
    ----------
    day : float
        Day of the visit, a finite value with 0 < day <= season length.
    site_start : float
        Downstream edge of the sampled window in km, as in step 2.
    window_length : float
        Length of the sampled window in km, finite and positive.
    reach : np.ndarray
        Shape (5,), the reach description of step 2.
    run : np.ndarray
        Shape (6,), the run description of step 1.
    shedding_rate : float
        eDNA shed per ml per day per unit of the unit-time migrant count, finite
        and positive.
    capture_volume : float
        Effective assayed volume of one replicate in ml, finite and positive.
    copy_target : float
        Expected copy yield the site must reach, finite and positive.

    Returns
    -------
    replicates : int
        The number of replicates, as a native Python int of at least 1.

    Raises
    ------
    ValueError
        If capture_volume or copy_target is not finite and positive, if the
        expected concentration of the sampled water is zero, or if any contract of
        steps 1 or 2 is broken.
    '''
    return replicates  # placeholder
```

### Step 9

09_survey_failure_probability.py

Goal
----
Probability that the planned two-site visit records no copies anywhere.

```python
def survey_failure_probability(reach: "np.ndarray", run: "np.ndarray", shedding_rate: float,
                               window_length: float, station_start: float,
                               station_replicates: int, upstream_start: float,
                               capture_volume: float, failure_target: float,
                               copy_target: float, first_day: int, last_day: int) -> float:
    '''Probability that the planned two-site visit records no copies anywhere.

    The plan is built in three moves, all on the model of steps 1 to 8. The visit
    is made on the first whole day between first_day and last_day on which a
    survey of station_replicates replicates at the station window alone meets the
    failure standard failure_target. At the upstream window the number of
    replicates is then set by the planning rule of step 8 for that day, with the
    same capture_volume and the given copy_target. Return the probability that the
    visit, both windows together, records no copies at all.

    Parameters
    ----------
    reach : np.ndarray
        Shape (5,), the reach description of step 2.
    run : np.ndarray
        Shape (6,), the run description of step 1.
    shedding_rate : float
        eDNA shed per ml per day per unit of the unit-time migrant count, finite
        and positive.
    window_length : float
        Length of every sampled window in km, finite and positive.
    station_start : float
        Downstream edge of the station window in km, as in step 5.
    station_replicates : int
        Number of replicates taken at the station, an integer of at least 1.
    upstream_start : float
        Downstream edge of the upstream window in km, as in step 5; the two
        windows do not overlap.
    capture_volume : float
        Effective assayed volume of one replicate in ml, finite and positive.
    failure_target : float
        The largest acceptable probability of recording no copies at the station
        window alone, finite, greater than 0 and less than 1.
    copy_target : float
        Expected copy yield the upstream window must reach, finite and positive.
    first_day : int
        First whole day of the search, an integer of at least 1.
    last_day : int
        Last whole day of the search, an integer of at least first_day and no
        greater than the season length.

    Returns
    -------
    probability : float
        The probability that the visit records no copies at either window,
        between 0 and 1, as a native Python float.

    Raises
    ------
    ValueError
        If station_replicates is not an integer of at least 1, or if any contract
        of steps 1 to 8 is broken.
    '''
    return probability  # placeholder
```
