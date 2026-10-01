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

def build_expression_panel(n_subjects: int, n_cell_types: int, zero_fraction: float,
                                   factor_correlation: float, seed: int) -> np.ndarray:
    # Local imports keep the oracle self-contained when the harness
    # executes it in isolation.
    import numpy as np
    from scipy.special import ndtri

    def _check_int(name, value, low, high):
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise ValueError(f"{name} must be an integer")
        if not (low <= int(value) <= high):
            raise ValueError(f"{name} must lie between {low} and {high}")
        return int(value)

    n_rows = _check_int("n_subjects", n_subjects, 2, 10 ** 6)
    n_cols = _check_int("n_cell_types", n_cell_types, 1, 14)
    seed_value = _check_int("seed", seed, -(2 ** 63), 2 ** 63 - 1)
    if not (isinstance(zero_fraction, (int, float)) and np.isfinite(zero_fraction)
            and 0.0 < float(zero_fraction) < 1.0):
        raise ValueError("zero_fraction must be a finite number strictly between 0 and 1")
    if not (isinstance(factor_correlation, (int, float)) and np.isfinite(factor_correlation)
            and 0.0 <= float(factor_correlation) < 1.0):
        raise ValueError("factor_correlation must be a finite number in the interval [0, 1)")

    rho = float(factor_correlation)
    rng = np.random.default_rng(seed_value)
    factor = rng.standard_normal(n_rows)
    idiosyncratic = rng.standard_normal((n_rows, n_cols))
    latent = np.sqrt(rho) * factor[:, None] + np.sqrt(1.0 - rho) * idiosyncratic

    # Tobit censoring at the quantile that leaves the requested share of zeros.
    censoring_point = float(ndtri(float(zero_fraction)))
    raw = np.maximum(latent - censoring_point, 0.0)

    spread = raw.std(axis=0)
    if np.any(spread <= 0.0):
        raise ValueError("a cell type is entirely censored; the panel cannot be standardised")
    panel = (raw - raw.mean(axis=0)) / spread
    return np.asarray(panel, dtype=float)

import numpy as np

def compute_subset_weights(panel: np.ndarray, maf: float) -> np.ndarray:
    # Local imports keep the oracle self-contained when the harness
    # executes it in isolation.
    import numpy as np

    matrix = np.asarray(panel, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] < 2 or matrix.shape[1] < 1:
        raise ValueError("panel must be a 2D array with at least 2 rows and 1 column")
    if matrix.shape[1] > 14:
        raise ValueError("panel must have at most 14 columns")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("panel must be finite")
    if not (isinstance(maf, (int, float)) and np.isfinite(maf)
            and 0.0 < float(maf) <= 0.5):
        raise ValueError("maf must be a finite number in the half-open interval (0, 0.5]")

    n_subjects, n_cell_types = matrix.shape
    frequency = float(maf)
    genotype_sd = np.sqrt(2.0 * frequency * (1.0 - frequency))

    # Conditional on the panel the score vector has this covariance.
    cross = matrix.T @ matrix / n_subjects

    weights = np.empty((2 ** n_cell_types - 1, n_subjects), dtype=float)
    for mask in range(1, 2 ** n_cell_types):
        members = np.array([c for c in range(n_cell_types) if (mask >> c) & 1], dtype=int)
        ones = np.ones(members.size, dtype=float)
        block = cross[np.ix_(members, members)]
        try:
            direction = np.linalg.solve(block, ones)
        except np.linalg.LinAlgError as exc:
            raise ValueError("a cell-type submatrix of the panel is singular") from exc
        scale = float(ones @ direction)
        if not (np.isfinite(scale) and scale > 0.0):
            raise ValueError("a cell-type submatrix of the panel is not positive definite")
        weights[mask - 1] = (matrix[:, members] @ direction) / (
            np.sqrt(n_subjects) * genotype_sd * np.sqrt(scale))
    return weights

import numpy as np

def compute_subset_correlation(weights: np.ndarray, maf: float) -> np.ndarray:
    # Local imports keep the oracle self-contained when the harness
    # executes it in isolation.
    import numpy as np

    table = np.asarray(weights, dtype=float)
    if table.ndim != 2 or table.shape[0] < 1 or table.shape[1] < 1:
        raise ValueError("weights must be a non-empty 2D array")
    if not np.all(np.isfinite(table)):
        raise ValueError("weights must be finite")
    if not (isinstance(maf, (int, float)) and np.isfinite(maf)
            and 0.0 < float(maf) <= 0.5):
        raise ValueError("maf must be a finite number in the half-open interval (0, 0.5]")

    frequency = float(maf)
    genotype_variance = 2.0 * frequency * (1.0 - frequency)
    covariance = genotype_variance * (table @ table.T)
    covariance = 0.5 * (covariance + covariance.T)

    spread = np.sqrt(np.diag(covariance))
    if np.any(spread <= 0.0):
        raise ValueError("a subset statistic has zero null variance")
    correlation = covariance / np.outer(spread, spread)
    np.fill_diagonal(correlation, 1.0)
    return np.clip(correlation, -1.0, 1.0)

import numpy as np

def build_neighbour_table(n_cell_types: int) -> np.ndarray:
    # Local imports keep the oracle self-contained when the harness
    # executes it in isolation.
    import numpy as np

    if isinstance(n_cell_types, bool) or not isinstance(n_cell_types, (int, np.integer)):
        raise ValueError("n_cell_types must be an integer")
    count = int(n_cell_types)
    if not 1 <= count <= 14:
        raise ValueError("n_cell_types must lie between 1 and 14")

    table = np.full((2 ** count - 1, count), -1, dtype=int)
    for mask in range(1, 2 ** count):
        for cell_type in range(count):
            toggled = mask ^ (1 << cell_type)
            # The empty set carries no statistic and is never visited.
            table[mask - 1, cell_type] = toggled - 1 if toggled >= 1 else -1
    return table

import numpy as np

def compute_dlm_pvalue(correlation: np.ndarray, neighbours: np.ndarray, threshold: float,
                               n_nodes: int = 400, tail_width: float = 8.0) -> float:
    # Local imports keep the oracle self-contained when the harness
    # executes it in isolation.
    import numpy as np
    from scipy.special import ndtr

    matrix = np.asarray(correlation, dtype=float)
    table = np.asarray(neighbours)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or matrix.shape[0] < 1:
        raise ValueError("correlation must be a square 2D array of order at least 1")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("correlation must be finite")
    if table.ndim != 2 or table.shape[0] != matrix.shape[0] or table.shape[1] < 1:
        raise ValueError("neighbours must be 2D with one row per subset")
    if not np.issubdtype(table.dtype, np.integer):
        raise ValueError("neighbours must hold integers")
    if int(table.max(initial=-1)) >= matrix.shape[0]:
        raise ValueError("neighbours refers to a subset outside the correlation matrix")
    if not (isinstance(threshold, (int, float)) and np.isfinite(threshold)
            and float(threshold) > 0.0):
        raise ValueError("threshold must be a finite number > 0")
    if isinstance(n_nodes, bool) or not isinstance(n_nodes, (int, np.integer)) or int(n_nodes) < 2:
        raise ValueError("n_nodes must be an integer >= 2")
    if not (isinstance(tail_width, (int, float)) and np.isfinite(tail_width)
            and float(tail_width) > 0.0):
        raise ValueError("tail_width must be a finite number > 0")

    level_floor = float(threshold)
    half_width = 0.5 * float(tail_width)
    nodes, quad_weights = np.polynomial.legendre.leggauss(int(n_nodes))
    levels = half_width * (nodes + 1.0) + level_floor
    density = np.exp(-0.5 * levels * levels) / np.sqrt(2.0 * np.pi)

    total = 0.0
    for site in range(matrix.shape[0]):
        log_product = np.zeros_like(levels)
        for column in range(table.shape[1]):
            partner = int(table[site, column])
            if partner < 0:
                continue
            rho = float(np.clip(matrix[site, partner], -1.0 + 1e-12, 1.0 - 1e-12))
            conditional_sd = np.sqrt(1.0 - rho * rho)
            inside = (ndtr((1.0 - rho) * levels / conditional_sd)
                      - ndtr(-(1.0 + rho) * levels / conditional_sd))
            log_product += np.log(np.clip(inside, 1e-300, None))
        total += half_width * float(np.sum(quad_weights * 2.0 * np.exp(log_product) * density))
    return float(total)

import numpy as np

def compute_subset_cgf(weight_row: np.ndarray, maf: float, tilt: float) -> tuple:
    # Local imports keep the oracle self-contained when the harness
    # executes it in isolation.
    import numpy as np

    coefficients = np.asarray(weight_row, dtype=float)
    if coefficients.ndim != 1 or coefficients.size < 1:
        raise ValueError("weight_row must be a non-empty 1D array")
    if not np.all(np.isfinite(coefficients)):
        raise ValueError("weight_row must be finite")
    if not (isinstance(maf, (int, float)) and np.isfinite(maf)
            and 0.0 < float(maf) <= 0.5):
        raise ValueError("maf must be a finite number in the half-open interval (0, 0.5]")
    if not (isinstance(tilt, (int, float)) and np.isfinite(tilt)):
        raise ValueError("tilt must be a finite number")

    frequency = float(maf)
    centred = np.array([0.0, 1.0, 2.0]) - 2.0 * frequency
    log_prior = np.log(np.array([(1.0 - frequency) ** 2,
                                 2.0 * frequency * (1.0 - frequency),
                                 frequency ** 2]))

    # Per-subject log weights of the tilted three-point law, shifted by their
    # maximum so that a large tilt cannot overflow.
    exponents = float(tilt) * np.outer(coefficients, centred) + log_prior[None, :]
    shift = np.max(exponents, axis=1)
    scaled = np.exp(exponents - shift[:, None])
    normaliser = scaled.sum(axis=1)
    cgf = float(np.sum(shift + np.log(normaliser)))

    tilted = scaled / normaliser[:, None]
    tilted_mean = tilted @ centred
    tilted_variance = tilted @ (centred * centred) - tilted_mean * tilted_mean
    first = float(np.sum(coefficients * tilted_mean))
    second = float(np.sum(coefficients * coefficients * tilted_variance))
    return (cgf, first, second)

import numpy as np

def solve_tilting_parameters(weights: np.ndarray, maf: float, threshold: float) -> np.ndarray:
    # Local imports keep the oracle self-contained when the harness
    # executes it in isolation.
    import numpy as np

    table = np.asarray(weights, dtype=float)
    if table.ndim != 2 or table.shape[0] < 1 or table.shape[1] < 1:
        raise ValueError("weights must be a non-empty 2D array")
    if not np.all(np.isfinite(table)):
        raise ValueError("weights must be finite")
    if not (isinstance(maf, (int, float)) and np.isfinite(maf)
            and 0.0 < float(maf) <= 0.5):
        raise ValueError("maf must be a finite number in the half-open interval (0, 0.5]")
    if not (isinstance(threshold, (int, float)) and np.isfinite(threshold)
            and float(threshold) > 0.0):
        raise ValueError("threshold must be a finite number > 0")

    frequency = float(maf)
    centred = np.array([0.0, 1.0, 2.0]) - 2.0 * frequency
    log_prior = np.log(np.array([(1.0 - frequency) ** 2,
                                 2.0 * frequency * (1.0 - frequency),
                                 frequency ** 2]))

    def _derivatives(row, tilt):
        exponents = tilt * np.outer(row, centred) + log_prior[None, :]
        scaled = np.exp(exponents - np.max(exponents, axis=1)[:, None])
        tilted = scaled / scaled.sum(axis=1)[:, None]
        mean = tilted @ centred
        variance = tilted @ (centred * centred) - mean * mean
        return float(np.sum(row * mean)), float(np.sum(row * row * variance))

    def _solve(row, target):
        low, high = -1.0, 1.0
        while _derivatives(row, low)[0] > target:
            if low < -1.0e8:
                raise ValueError("threshold lies outside the range the subset statistic can attain")
            low *= 2.0
        while _derivatives(row, high)[0] < target:
            if high > 1.0e8:
                raise ValueError("threshold lies outside the range the subset statistic can attain")
            high *= 2.0
        tilt = 0.5 * (low + high)
        for _ in range(400):
            first, second = _derivatives(row, tilt)
            residual = first - target
            if residual > 0.0:
                high = tilt
            else:
                low = tilt
            if abs(residual) <= 1e-12 or (high - low) <= 1e-14 * max(1.0, abs(tilt)):
                break
            step = tilt - residual / second if second > 0.0 else 0.5 * (low + high)
            tilt = step if low < step < high else 0.5 * (low + high)
        return tilt

    level = float(threshold)
    tilts = np.empty((table.shape[0], 2), dtype=float)
    for index in range(table.shape[0]):
        tilts[index, 0] = _solve(table[index], +level)
        tilts[index, 1] = _solve(table[index], -level)
    return tilts

import numpy as np

def sample_tilted_genotypes(weight_row: np.ndarray, maf: float, tilt: float,
                                    uniforms: np.ndarray) -> np.ndarray:
    # Local imports keep the oracle self-contained when the harness
    # executes it in isolation.
    import numpy as np

    coefficients = np.asarray(weight_row, dtype=float)
    draws = np.asarray(uniforms, dtype=float)
    if coefficients.ndim != 1 or coefficients.size < 1:
        raise ValueError("weight_row must be a non-empty 1D array")
    if not np.all(np.isfinite(coefficients)):
        raise ValueError("weight_row must be finite")
    if draws.ndim != 1 or draws.size != coefficients.size:
        raise ValueError("uniforms must be 1D with one entry per subject")
    if not np.all((draws >= 0.0) & (draws < 1.0)):
        raise ValueError("uniforms must lie in the half-open interval [0, 1)")
    if not (isinstance(maf, (int, float)) and np.isfinite(maf)
            and 0.0 < float(maf) <= 0.5):
        raise ValueError("maf must be a finite number in the half-open interval (0, 0.5]")
    if not (isinstance(tilt, (int, float)) and np.isfinite(tilt)):
        raise ValueError("tilt must be a finite number")

    frequency = float(maf)
    centred = np.array([0.0, 1.0, 2.0]) - 2.0 * frequency
    log_prior = np.log(np.array([(1.0 - frequency) ** 2,
                                 2.0 * frequency * (1.0 - frequency),
                                 frequency ** 2]))

    exponents = float(tilt) * np.outer(coefficients, centred) + log_prior[None, :]
    exponents -= np.max(exponents, axis=1, keepdims=True)
    tilted = np.exp(exponents)
    tilted /= tilted.sum(axis=1, keepdims=True)

    # Inverse cumulative rule: the smallest genotype whose cumulative tilted
    # probability strictly exceeds the uniform variate.
    cumulative = np.cumsum(tilted, axis=1)
    genotypes = (draws[:, None] > cumulative[:, :2]).sum(axis=1)
    return genotypes.astype(float)

import numpy as np

def compute_is_weight(weights: np.ndarray, tilts: np.ndarray, cgf_values: np.ndarray,
                              genotypes: np.ndarray, maf: float, threshold: float) -> float:
    # Local imports keep the oracle self-contained when the harness
    # executes it in isolation.
    import numpy as np

    table = np.asarray(weights, dtype=float)
    tilt_table = np.asarray(tilts, dtype=float)
    cgf_table = np.asarray(cgf_values, dtype=float)
    draw = np.asarray(genotypes, dtype=float)
    if table.ndim != 2 or table.shape[0] < 1 or table.shape[1] < 1:
        raise ValueError("weights must be a non-empty 2D array")
    if tilt_table.shape != (table.shape[0], 2):
        raise ValueError("tilts must have shape (n_subsets, 2)")
    if cgf_table.shape != (table.shape[0], 2):
        raise ValueError("cgf_values must have shape (n_subsets, 2)")
    if draw.ndim != 1 or draw.size != table.shape[1]:
        raise ValueError("genotypes must be 1D with one entry per subject")
    if not np.all(np.isin(draw, (0.0, 1.0, 2.0))):
        raise ValueError("genotypes must take the values 0, 1 or 2")
    if not (np.all(np.isfinite(table)) and np.all(np.isfinite(tilt_table))
            and np.all(np.isfinite(cgf_table))):
        raise ValueError("weights, tilts and cgf_values must be finite")
    if not (isinstance(maf, (int, float)) and np.isfinite(maf)
            and 0.0 < float(maf) <= 0.5):
        raise ValueError("maf must be a finite number in the half-open interval (0, 0.5]")
    if not (isinstance(threshold, (int, float)) and np.isfinite(threshold)
            and float(threshold) > 0.0):
        raise ValueError("threshold must be a finite number > 0")

    statistics = table @ (draw - 2.0 * float(maf))
    if np.max(np.abs(statistics)) <= float(threshold):
        return 0.0

    exponents = np.concatenate([tilt_table[:, 0] * statistics - cgf_table[:, 0],
                                tilt_table[:, 1] * statistics - cgf_table[:, 1]])
    shift = float(np.max(exponents))
    log_mixture = shift + float(np.log(np.sum(np.exp(exponents - shift))))
    return float(2.0 * table.shape[0] * np.exp(-log_mixture))

import numpy as np 

def run_importance_sampling(weights: np.ndarray, tilts: np.ndarray, cgf_values: np.ndarray,
                                    maf: float, threshold: float, n_sims: int, seed: int) -> tuple:
    # Local imports keep the oracle self-contained when the harness
    # executes it in isolation.
    import glob
    import importlib.util
    import os
    import sys

    import numpy as np

    table = np.asarray(weights, dtype=float)
    tilt_table = np.asarray(tilts, dtype=float)
    cgf_table = np.asarray(cgf_values, dtype=float)
    if table.ndim != 2 or table.shape[0] < 1 or table.shape[1] < 1:
        raise ValueError("weights must be a non-empty 2D array")
    if tilt_table.shape != (table.shape[0], 2) or cgf_table.shape != (table.shape[0], 2):
        raise ValueError("tilts and cgf_values must have shape (n_subsets, 2)")
    if not (np.all(np.isfinite(table)) and np.all(np.isfinite(tilt_table))
            and np.all(np.isfinite(cgf_table))):
        raise ValueError("weights, tilts and cgf_values must be finite")
    if not (isinstance(maf, (int, float)) and np.isfinite(maf)
            and 0.0 < float(maf) <= 0.5):
        raise ValueError("maf must be a finite number in the half-open interval (0, 0.5]")
    if not (isinstance(threshold, (int, float)) and np.isfinite(threshold)
            and float(threshold) > 0.0):
        raise ValueError("threshold must be a finite number > 0")
    for name, value, floor in (("n_sims", n_sims, 2), ("seed", seed, -(2 ** 63))):
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or int(value) < floor:
            raise ValueError(f"{name} must be an integer >= {floor}")

    # -- Resolve the oracle functions of sub-problems 08-09, which this step
    #    composes rather than reimplements. Preference order: (1) already
    #    present in the executing namespace (shared-namespace harness),
    #    (2) loaded from a sibling sub-problem file matched by name pattern
    #    (standalone execution; file prefixes may vary), (3) the public
    #    function of the same step if the harness injected it.
    def _resolve_step(oracle_name, pattern):
        namespace = globals()
        candidate = namespace.get(oracle_name)
        if callable(candidate):
            return candidate
        search_dirs = []
        if "__file__" in namespace:
            search_dirs.append(os.path.dirname(os.path.abspath(namespace["__file__"])))
        cwd = os.getcwd()
        search_dirs += [cwd, os.path.join(cwd, "sub_problems")]
        if sys.argv and sys.argv[0]:
            search_dirs.append(os.path.dirname(os.path.abspath(sys.argv[0])))
        seen = set()
        search_dirs = [d for d in search_dirs if not (d in seen or seen.add(d))]
        for directory in search_dirs:
            for path in sorted(glob.glob(os.path.join(directory, pattern))):
                spec = importlib.util.spec_from_file_location(
                    os.path.basename(path)[:-3], path)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                if hasattr(module, oracle_name):
                    return getattr(module, oracle_name)
        public = namespace.get(oracle_name.replace("", "", 1))
        if callable(public):
            return public
        raise RuntimeError(f"cannot resolve required step function {oracle_name}")

    tilted_draw = _resolve_step(
        "sample_tilted_genotypes", "*sample_tilted_genotypes*.py")
    replicate_weight = _resolve_step(
        "compute_is_weight", "*compute_is_weight*.py")

    n_subsets, n_subjects = table.shape
    rng = np.random.default_rng(int(seed))
    contributions = np.empty(int(n_sims), dtype=float)
    for replicate in range(int(n_sims)):
        driver = int(rng.integers(n_subsets))
        branch = 0 if rng.random() >= 0.5 else 1
        draws = rng.random(n_subjects)
        # -- Sub-problem 08: one genotype vector under the driving subset's tilted law.
        genotypes = tilted_draw(table[driver], maf, float(tilt_table[driver, branch]), draws)
        # -- Sub-problem 09: the likelihood-ratio contribution of that vector.
        contributions[replicate] = replicate_weight(
            table, tilt_table, cgf_table, genotypes, maf, threshold)

    pvalue = float(contributions.mean())
    standard_error = float(np.sqrt(contributions.var(ddof=1) / int(n_sims)))
    return (pvalue, standard_error)

import numpy as np 

def run_asset_discrepancy_pipeline(n_subjects: int = 120, n_cell_types: int = 7,
                                           zero_fraction: float = 0.6,
                                           factor_correlation: float = 0.4,
                                           panel_seed: int = 20260824, maf: float = 0.02,
                                           threshold: float = 7.0, n_sims: int = 50000,
                                           sampling_seed: int = 2026) -> float:
    # Local imports keep the oracle self-contained when the harness
    # executes it in isolation.
    import glob
    import importlib.util
    import os
    import sys

    import numpy as np

    # -- Resolve the oracle functions of sub-problems 01-10. Preference order:
    #    (1) already present in the executing namespace (shared-namespace
    #    harness), (2) loaded from a sibling sub-problem file matched by name
    #    pattern (standalone execution; file prefixes may vary), (3) the
    #    public function of the same step if the harness injected it.
    def _resolve_step(oracle_name, pattern):
        namespace = globals()
        candidate = namespace.get(oracle_name)
        if callable(candidate):
            return candidate
        search_dirs = []
        if "__file__" in namespace:
            search_dirs.append(os.path.dirname(os.path.abspath(namespace["__file__"])))
        cwd = os.getcwd()
        search_dirs += [cwd, os.path.join(cwd, "sub_problems")]
        if sys.argv and sys.argv[0]:
            search_dirs.append(os.path.dirname(os.path.abspath(sys.argv[0])))
        seen = set()
        search_dirs = [d for d in search_dirs if not (d in seen or seen.add(d))]
        for directory in search_dirs:
            for path in sorted(glob.glob(os.path.join(directory, pattern))):
                spec = importlib.util.spec_from_file_location(
                    os.path.basename(path)[:-3], path)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                if hasattr(module, oracle_name):
                    return getattr(module, oracle_name)
        public = namespace.get(oracle_name.replace("", "", 1))
        if callable(public):
            return public
        raise RuntimeError(f"cannot resolve required step function {oracle_name}")

    build_panel = _resolve_step(
        "build_expression_panel", "*build_expression_panel*.py")
    subset_weights = _resolve_step(
        "compute_subset_weights", "*compute_subset_weights*.py")
    subset_correlation = _resolve_step(
        "compute_subset_correlation", "*compute_subset_correlation*.py")
    neighbour_table = _resolve_step(
        "build_neighbour_table", "*build_neighbour_table*.py")
    dlm_pvalue = _resolve_step(
        "compute_dlm_pvalue", "*compute_dlm_pvalue*.py")
    subset_cgf = _resolve_step(
        "compute_subset_cgf", "*compute_subset_cgf*.py")
    tilting_parameters = _resolve_step(
        "solve_tilting_parameters", "*solve_tilting_parameters*.py")
    importance_sampling = _resolve_step(
        "run_importance_sampling", "*run_importance_sampling*.py")

    # -- Sub-problem 01: the panel every later step conditions on.
    panel = build_panel(n_subjects, n_cell_types, zero_fraction, factor_correlation, panel_seed)

    # -- Sub-problems 02-05: the analytic significance under normality.
    weights = subset_weights(panel, maf)
    correlation = subset_correlation(weights, maf)
    neighbours = neighbour_table(n_cell_types)
    analytic = float(dlm_pvalue(correlation, neighbours, threshold))

    # -- Sub-problems 06-10: the genotype-conditional estimate.
    tilts = tilting_parameters(weights, maf, threshold)
    cgf_values = np.empty_like(np.asarray(tilts, dtype=float))
    for index in range(cgf_values.shape[0]):
        cgf_values[index, 0] = subset_cgf(weights[index], maf, float(tilts[index, 0]))[0]
        cgf_values[index, 1] = subset_cgf(weights[index], maf, float(tilts[index, 1]))[0]
    estimate = importance_sampling(weights, tilts, cgf_values, maf, threshold,
                                   n_sims, sampling_seed)
    conditional = float(estimate[0])

    if not (analytic > 0.0 and conditional > 0.0):
        raise ValueError("both significances must be strictly positive to be compared")
    return float(np.log10(analytic / conditional))
SCICODE_GOLD_EOF
