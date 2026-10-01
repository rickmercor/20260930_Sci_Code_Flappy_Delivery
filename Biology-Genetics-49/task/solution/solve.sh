#!/bin/bash
# Oracle: writes the gold solution so the verifier scores reward 1.0.
cat > /app/solution.py <<'SCICODE_GOLD_EOF'
import numpy as np
try:
    import scipy, scipy.linalg, scipy.sparse, scipy.optimize, scipy.spatial, scipy.integrate, scipy.stats, scipy.special, scipy.signal
except Exception:
    pass
import math

import numpy as np
from scipy import special


def _response_names() -> tuple:
    """The five size responses, in the order used throughout the pipeline."""
    return ("arithmetic", "geometric", "harmonic", "root_mean_square", "s_shaped")


def _check_size(name: str, value: float, minimum: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise ValueError(f"{name} must be a finite number from {minimum:g} to 1e7")
    v = float(value)
    if not np.isfinite(v) or v < minimum or v > 1e7:
        raise ValueError(f"{name} must be a finite number from {minimum:g} to 1e7")
    return v


def _check_response(response: str) -> str:
    if not isinstance(response, str) or response not in _response_names():
        raise ValueError("response must be one of " + ", ".join(_response_names()))
    return response


def _size_table(p: "np.ndarray", na: float, nm: float, kind: str) -> "np.ndarray":
    """Unchecked census size and its integral from 0, for a float array p in [0, 1]."""
    if kind == "arithmetic":
        size = (1.0 - p) * na + p * nm
        integral = p * (na + 0.5 * (nm - na) * p)
    elif kind == "geometric":
        rate = np.log(nm / na)
        size = na * np.exp(p * rate)
        integral = na * p * special.exprel(p * rate)             # Na (e^{pL} - 1)/L without cancellation
    elif kind == "harmonic":
        x = (na / nm - 1.0) * p
        size = na / (1.0 + x)
        ratio = np.ones_like(x)
        large = np.abs(x) > 1e-8
        ratio[large] = np.log1p(x[large]) / x[large]
        ratio[~large] = 1.0 - x[~large] / 2.0 + x[~large] ** 2 / 3.0
        integral = na * p * ratio                                # log(1 + x)/(1/Nm - 1/Na)
    elif kind == "root_mean_square":
        a = (1.0 - p) * na * na + p * nm * nm
        b = na * na
        size = np.sqrt(a)
        integral = (2.0 / 3.0) * p * (a + np.sqrt(a * b) + b) / (np.sqrt(a) + np.sqrt(b))
    else:                                                        # the source's S-shaped response
        share = p * p / (p * p + (1.0 - p) ** 2)
        size = na + (nm - na) * share
        integral = na * p + (nm - na) * (0.5 * p + 0.25 * np.log1p(2.0 * p * (p - 1.0)))
    return np.column_stack([size, integral])


def size_response(frequency: "np.ndarray", ancestral_size: float, mutant_size: float, response: str) -> "np.ndarray":
    try:
        p = np.asarray(frequency, dtype=float)
    except (TypeError, ValueError):
        raise ValueError("frequency must be a one-dimensional array of at least one finite number in [0, 1]")
    if p.ndim != 1 or p.size < 1 or not np.all(np.isfinite(p)) or np.any(p < 0.0) or np.any(p > 1.0):
        raise ValueError("frequency must be a one-dimensional array of at least one finite number in [0, 1]")
    na = _check_size("ancestral_size", ancestral_size, 1.0)
    nm = _check_size("mutant_size", mutant_size, 1.0)
    kind = _check_response(response)
    return _size_table(p, na, nm, kind)

import math
import numpy as np
from scipy import integrate


def _check_diffusion_inputs(initial_frequency: float, selection: float, ancestral_size: float, mutant_size: float,
                            response: str, offspring_variance: float) -> tuple:
    for name, value in (("initial_frequency", initial_frequency), ("selection", selection),
                        ("offspring_variance", offspring_variance)):
        if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)) \
                or not np.isfinite(float(value)):
            raise ValueError(f"{name} must be a finite number")
    p0, s, var = float(initial_frequency), float(selection), float(offspring_variance)
    if not 0.0 < p0 <= 0.1:
        raise ValueError("initial_frequency must satisfy 0 < initial_frequency <= 0.1")
    if not 0.0 <= s <= 0.05:
        raise ValueError("selection must be a finite number from 0 to 0.05")
    if not 0.1 <= var <= 10.0:
        raise ValueError("offspring_variance must be a finite number from 0.1 to 10")
    na = _check_size("ancestral_size", ancestral_size, 2.0)
    nm = _check_size("mutant_size", mutant_size, 2.0)
    if max(na, nm) > 1e6:
        raise ValueError("ancestral_size and mutant_size must be finite numbers from 2 to 1e6")
    if max(na, nm) > 100.0 * min(na, nm):
        raise ValueError("the larger census size must not exceed 100 times the smaller")
    kind = _check_response(response)
    if s * max(na, nm) / var > 200.0:
        raise ValueError("selection * max(ancestral_size, mutant_size) / offspring_variance must not exceed 200")
    return p0, s, na, nm, kind, var


def _break_points(lower: float, upper: float, rates: tuple) -> list:
    """Interior points at a few multiples of each decay length 1/rate, to guide adaptive quadrature."""
    points = set()
    for rate in rates:
        if rate > 0.0:
            for multiple in (0.5, 2.0, 8.0, 32.0, 128.0):
                x = lower + multiple / rate
                if lower < x < upper:
                    points.add(x)
    return sorted(points) or None


def fixation_probability(initial_frequency: float, selection: float, ancestral_size: float, mutant_size: float,
                                 response: str, offspring_variance: float) -> float:
    p0, s, na, nm, kind, var = _check_diffusion_inputs(initial_frequency, selection, ancestral_size, mutant_size,
                                                       response, offspring_variance)
    if s == 0.0:
        return p0
    c = 2.0 * s / var

    def _weight(y):
        return math.exp(-c * size_response(np.array([y]), na, nm, kind)[0, 1])

    options = dict(epsabs=0.0, epsrel=1e-12, limit=500)
    head = integrate.quad(_weight, 0.0, p0, **options)[0]
    tail = integrate.quad(_weight, p0, 1.0, points=_break_points(p0, 1.0, (c * min(na, nm), c * max(na, nm))),
                          **options)[0]
    return float(head / (head + tail))

import math
import numpy as np
from scipy import special


def _check_count(name: str, value: int, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer, float, np.floating)):
        raise ValueError(f"{name} must be an integer >= {minimum}")
    v = float(value)
    if not np.isfinite(v) or v != math.floor(v) or v < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(v)


def introduction_log_likelihood(fixations: int, introductions: int, probability: float) -> float:
    n = _check_count("introductions", introductions, 1)
    k = _check_count("fixations", fixations, 0)
    if k > n:
        raise ValueError("fixations must be an integer from 0 to introductions")
    if isinstance(probability, bool) or not isinstance(probability, (int, float, np.integer, np.floating)):
        raise ValueError("probability must be a finite number with 0 < probability < 1")
    q = float(probability)
    if not np.isfinite(q) or not 0.0 < q < 1.0:
        raise ValueError("probability must be a finite number with 0 < probability < 1")
    log_choose = special.gammaln(n + 1.0) - special.gammaln(k + 1.0) - special.gammaln(n - k + 1.0)
    return float(log_choose + k * math.log(q) + (n - k) * math.log1p(-q))

import numpy as np


def _check_strains(fixations: "np.ndarray", mutant_sizes: "np.ndarray", selections: "np.ndarray") -> tuple:
    arrays = []
    for name, value in (("fixations", fixations), ("mutant_sizes", mutant_sizes), ("selections", selections)):
        try:
            a = np.asarray(value, dtype=float)
        except (TypeError, ValueError):
            raise ValueError("fixations, mutant_sizes and selections must be one-dimensional numeric arrays of equal length >= 1")
        if a.ndim != 1 or a.size < 1 or not np.all(np.isfinite(a)):
            raise ValueError("fixations, mutant_sizes and selections must be one-dimensional numeric arrays of equal length >= 1")
        arrays.append(a)
    if not arrays[0].size == arrays[1].size == arrays[2].size:
        raise ValueError("fixations, mutant_sizes and selections must be one-dimensional numeric arrays of equal length >= 1")
    return tuple(arrays)


def response_log_likelihoods(fixations: "np.ndarray", introductions: int, mutant_sizes: "np.ndarray",
                                     selections: "np.ndarray", ancestral_size: float, offspring_variance: float) -> "np.ndarray":
    counts, sizes, advantages = _check_strains(fixations, mutant_sizes, selections)
    n = _check_count("introductions", introductions, 1)
    na = _check_size("ancestral_size", ancestral_size, 10.0)
    if na > 1e6:
        raise ValueError("ancestral_size must be a finite number from 10 to 1e6")
    log_likelihoods = np.zeros(len(_response_names()))
    for j, kind in enumerate(_response_names()):
        total = 0.0
        for count, size, advantage in zip(counts, sizes, advantages):
            u = fixation_probability(1.0 / na, float(advantage), na, float(size), kind, offspring_variance)
            total += introduction_log_likelihood(count, n, u)
        log_likelihoods[j] = total
    return log_likelihoods

import math
import numpy as np
from scipy import integrate


def conditional_fixation_time(initial_frequency: float, selection: float, ancestral_size: float,
                                      mutant_size: float, response: str, offspring_variance: float) -> float:
    p0, s, na, nm, kind, var = _check_diffusion_inputs(initial_frequency, selection, ancestral_size, mutant_size,
                                                       response, offspring_variance)
    c = 2.0 * s / var
    rates = (c * min(na, nm), c * max(na, nm))
    inner = dict(epsabs=0.0, epsrel=1e-12, limit=500)
    outer = dict(epsabs=0.0, epsrel=1e-11, limit=500)

    def _row(x):
        return size_response(np.array([x]), na, nm, kind)[0]

    if s == 0.0:
        # neutral: u(x) = x, the tail ratio is 1 - x and the head ratio is x
        drift = 2.0 / var
        first = drift * (_row(1.0)[1] - _row(p0)[1])
        second = integrate.quad(lambda x: drift * _row(x)[0] * x / (1.0 - x), 0.0, p0, **outer)[0]
        return float(first + (1.0 - p0) / p0 * second)

    def _weight(y):
        return math.exp(-c * _row(y)[1])

    head = integrate.quad(_weight, 0.0, p0, **inner)[0]
    total = head + integrate.quad(_weight, p0, 1.0, points=_break_points(p0, 1.0, rates), **inner)[0]

    def _u(x):
        if x <= p0:
            return integrate.quad(_weight, 0.0, x, **inner)[0] / total
        return (head + integrate.quad(_weight, p0, x, points=_break_points(p0, x, rates), **inner)[0]) / total

    def _tail_ratio(x):
        # integral from x to 1 of the weight, divided by the weight at x: no cancellation, no underflow
        fx = _row(x)[1]
        return integrate.quad(lambda y: math.exp(-c * (_row(y)[1] - fx)), x, 1.0,
                              points=_break_points(x, 1.0, rates), **inner)[0]

    def _sojourn_after(x):
        return 2.0 * _row(x)[0] * _u(x) * _tail_ratio(x) / (var * x * (1.0 - x))

    f_start = _row(p0)[1]

    def _sojourn_before(x):
        # the term below the start frequency, with its factor (1 - u0) = weight(p0) * tail_ratio(p0) / total taken inside
        head_x = integrate.quad(_weight, 0.0, x, **inner)[0]
        return 2.0 * _row(x)[0] * head_x * head_x * math.exp(-c * (f_start - _row(x)[1])) / (var * x * (1.0 - x))

    first = integrate.quad(_sojourn_after, p0, 1.0, points=_break_points(p0, 1.0, rates), **outer)[0]
    second = integrate.quad(_sojourn_before, 0.0, p0, **outer)[0]
    u0 = head / total
    return float(first + _tail_ratio(p0) * second / (u0 * total * total))

import numpy as np


def strain_sweep_table(mutant_sizes: "np.ndarray", selections: "np.ndarray", ancestral_size: float,
                               offspring_variance: float, response: str) -> "np.ndarray":
    sizes = None
    try:
        sizes = np.asarray(mutant_sizes, dtype=float)
        advantages = np.asarray(selections, dtype=float)
    except (TypeError, ValueError):
        sizes = None
    if sizes is None or sizes.ndim != 1 or sizes.size < 1 or advantages.ndim != 1 or advantages.size != sizes.size \
            or not np.all(np.isfinite(sizes)) or not np.all(np.isfinite(advantages)):
        raise ValueError("mutant_sizes and selections must be one-dimensional numeric arrays of equal length >= 1")
    na = _check_size("ancestral_size", ancestral_size, 10.0)
    if na > 1e6:
        raise ValueError("ancestral_size must be a finite number from 10 to 1e6")
    table = np.zeros((sizes.size, 2))
    for i, (size, advantage) in enumerate(zip(sizes, advantages)):
        table[i, 0] = fixation_probability(1.0 / na, float(advantage), na, float(size), response, offspring_variance)
        table[i, 1] = conditional_fixation_time(1.0 / na, float(advantage), na, float(size), response,
                                                        offspring_variance)
    return table

import numpy as np


def first_sweep_duration(fixations: "np.ndarray", introductions: int, mutant_sizes: "np.ndarray",
                                 selections: "np.ndarray", ancestral_size: float, offspring_variance: float) -> float:
    log_likelihoods = response_log_likelihoods(fixations, introductions, mutant_sizes, selections,
                                                       ancestral_size, offspring_variance)
    decided = _response_names()[int(np.argmax(log_likelihoods))]       # the first maximum wins a tie
    table = strain_sweep_table(mutant_sizes, selections, ancestral_size, offspring_variance, decided)
    return float(np.dot(table[:, 0], table[:, 1]) / np.sum(table[:, 0]))
SCICODE_GOLD_EOF
