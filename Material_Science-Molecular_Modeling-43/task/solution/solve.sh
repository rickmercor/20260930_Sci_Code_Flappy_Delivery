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

def framework_msd_screen(
    positions: "np.ndarray", non_li: "np.ndarray", max_msd: float, start: int = 0
) -> tuple["np.ndarray", bool]:
    """Check non-Li framework rigidity against a disclosed fixture limit.

    The paper specifies the non-Li MSD check, but not a numeric cutoff.
    ``max_msd`` is therefore an instance control, not a sourced threshold.
    Positions must be unwrapped; shape is (time, atoms, Cartesian 3).
    """
    x = np.asarray(positions, dtype=np.float64)
    mask = np.asarray(non_li)
    if x.ndim != 3 or x.shape[0] < 2 or x.shape[2] != 3:
        raise ValueError("positions must have shape (time>=2, atoms, 3)")
    if mask.shape != (x.shape[1],) or mask.dtype != bool or not mask.any():
        raise ValueError("non_li must select at least one framework atom")
    if not np.isfinite(x).all() or not np.isfinite(max_msd) or max_msd < 0:
        raise ValueError("positions and cutoff must be finite")
    if not isinstance(start, (int, np.integer)) or not 0 <= start < x.shape[0]:
        raise ValueError("invalid start")
    displacement = x[:, mask] - x[0, mask]
    trace = np.mean(np.sum(displacement * displacement, axis=-1), axis=1)
    return trace, bool(np.max(trace[start:]) <= max_msd)

import numpy as np

def feature_coverage(
    mace: "np.ndarray", aimd: "np.ndarray", bins: int, eigen_cutoff: float = 1.0
) -> tuple[float, "np.ndarray"]:
    """Apply paper Eqs. 4–5 to jointly standardized, PCA-reduced embeddings.

    The input is *precomputed* fixed-length pretrained-feature embeddings.
    It does not simulate the embedding network or DIRECT cluster selection.
    Each coordinate uses shared bins over the combined projected range;
    the denominator counts AIMD-occupied bins.
    """
    a, b = np.asarray(mace, float), np.asarray(aimd, float)
    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[1]:
        raise ValueError("two feature matrices with equal width are required")
    if min(a.shape[0], b.shape[0]) < 2 or a.shape[1] < 1:
        raise ValueError("each panel needs at least two rows")
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("nonfinite feature")
    if not isinstance(bins, (int, np.integer)) or bins < 2:
        raise ValueError("bins must be an integer at least two")
    if not np.isfinite(eigen_cutoff) or eigen_cutoff <= 0:
        raise ValueError("invalid eigenvalue cutoff")
    joined = np.concatenate((a, b), axis=0)
    mu = joined.mean(axis=0)
    sd = joined.std(axis=0)
    sd = np.where(sd > 0, sd, 1.0)
    standardized = (joined - mu) / sd
    _, singular, vt = np.linalg.svd(standardized, full_matrices=False)
    eigen = singular**2 / (len(joined) - 1)
    keep = eigen > eigen_cutoff
    if not np.any(keep):
        raise ValueError("no retained pretrained-feature component")
    pa = (a - mu) / sd @ vt[keep].T
    pb = (b - mu) / sd @ vt[keep].T
    ratios = []
    for d in range(pa.shape[1]):
        lo = min(pa[:, d].min(), pb[:, d].min())
        hi = max(pa[:, d].max(), pb[:, d].max())
        if lo == hi:
            lo, hi = lo - 0.5, hi + 0.5
        ha, edges = np.histogram(pa[:, d], bins=bins, range=(lo, hi))
        hb, _ = np.histogram(pb[:, d], bins=edges)
        ratios.append(np.count_nonzero((ha > 0) & (hb > 0)) / np.count_nonzero(hb))
    values = np.asarray(ratios, dtype=np.float64)
    return float(values.mean()), values

import numpy as np

def relative_rmse(reference: "np.ndarray", prediction: "np.ndarray") -> float:
    """Compute source Eq. 1 in percent on an explicitly chosen test panel."""
    y, p = np.asarray(reference, float), np.asarray(prediction, float)
    if y.shape != p.shape or y.ndim != 1 or y.size < 2:
        raise ValueError("one-dimensional matched test panels are required")
    if not np.isfinite(y).all() or not np.isfinite(p).all():
        raise ValueError("nonfinite label")
    reference_square = np.dot(y, y)
    if reference_square <= 0:
        raise ValueError("reference RMS must be positive")
    return float(100 * np.sqrt(np.dot(y - p, y - p) / reference_square))

import numpy as np

def radial_distribution(
    positions: "np.ndarray", box_length: float, edges: "np.ndarray"
) -> "np.ndarray":
    """Compute paper Eq. 2 using a stated cubic periodic-box discretization.

    Positions have shape (frames, atoms, 3). Pair distances use minimum-image
    convention. Each shell's mean pair count is multiplied by
    2*volume/(shell_volume*N*(N-1)), as in the source equation.
    """
    x, r = np.asarray(positions, float), np.asarray(edges, float)
    if x.ndim != 3 or x.shape[0] < 1 or x.shape[1] < 2 or x.shape[2] != 3:
        raise ValueError("positions must have shape (frames, atoms>=2, 3)")
    if r.ndim != 1 or r.size < 3 or r[0] < 0 or not np.all(np.diff(r) > 0):
        raise ValueError("radial edges must be increasing and nonnegative")
    if not np.isfinite(x).all() or not np.isfinite(r).all():
        raise ValueError("nonfinite geometry")
    if not np.isfinite(box_length) or box_length <= 0 or r[-1] > box_length / 2:
        raise ValueError("radial range exceeds the minimum-image radius")
    n = x.shape[1]
    upper = np.triu_indices(n, 1)
    hist = np.zeros(r.size - 1, dtype=np.float64)
    for frame in x:
        delta = frame[upper[0]] - frame[upper[1]]
        delta -= box_length * np.round(delta / box_length)
        hist += np.histogram(np.linalg.norm(delta, axis=1), bins=r)[0]
    hist /= x.shape[0]
    shell_volume = 4 * np.pi / 3 * (r[1:] ** 3 - r[:-1] ** 3)
    return hist * (2 * box_length**3 / (shell_volume * n * (n - 1)))

import numpy as np

def rdf_stability(
    aimd_rdf: "np.ndarray",
    model_rdf_by_frame: "np.ndarray",
    edges: "np.ndarray",
    frame_dt_ps: float,
    window_ps: float = 1.0,
    threshold: float = 0.2,
) -> tuple["np.ndarray", float | None]:
    """Discretize the source Eq. 3 short-window RDF deviation test.

    The 1 ps window and 0.2 threshold are sourced. The supplied radial range,
    rectangular-bin quadrature, complete trailing windows, and first-crossing
    convention are disclosed instance choices.
    """
    ref = np.asarray(aimd_rdf, float)
    model = np.asarray(model_rdf_by_frame, float)
    r = np.asarray(edges, float)
    if model.ndim != 2 or ref.shape != (model.shape[1],) or r.shape != (ref.size + 1,):
        raise ValueError("RDF panels and radial edges have mismatched shape")
    if not np.isfinite(ref).all() or not np.isfinite(model).all() or not np.isfinite(r).all():
        raise ValueError("nonfinite RDF")
    if not np.all(np.diff(r) > 0) or frame_dt_ps <= 0 or window_ps <= 0 or threshold < 0:
        raise ValueError("invalid grid or stability controls")
    width = window_ps / frame_dt_ps
    nearest = round(width)
    if nearest < 1 or not np.isclose(width, nearest, atol=1e-10, rtol=0):
        raise ValueError("window must contain a whole number of frames")
    if model.shape[0] < nearest:
        raise ValueError("trajectory is shorter than the stability window")
    deviations = np.empty(model.shape[0] - nearest + 1, dtype=np.float64)
    for i in range(nearest - 1, model.shape[0]):
        short_mean = model[i - nearest + 1 : i + 1].mean(axis=0)
        deviations[i - nearest + 1] = np.dot(np.abs(ref - short_mean), np.diff(r))
    failures = np.flatnonzero(deviations > threshold)
    first_failure = None if failures.size == 0 else float((failures[0] + nearest) * frame_dt_ps)
    return deviations, first_failure

import numpy as np

def diatomic_turning_point(distance: "np.ndarray", energy: "np.ndarray") -> float:
    """Find the first repulsive-to-attractive Li–Li force sign change.

    Segment forces are minus secant slopes. Interpolate linearly between
    adjacent segment midpoints at the first positive-to-negative force
    crossing. A smooth curve without a crossing receives the paper's 0.1 Å
    plotting marker; this is a plotting convention, not a physical minimum.
    """
    r, e = np.asarray(distance, float), np.asarray(energy, float)
    if r.ndim != 1 or e.shape != r.shape or r.size < 3:
        raise ValueError("matched distance and energy vectors need >=3 points")
    if not np.isfinite(r).all() or not np.isfinite(e).all() or not np.all(np.diff(r) > 0):
        raise ValueError("distance must increase; all values must be finite")
    mid = (r[:-1] + r[1:]) / 2
    force = -np.diff(e) / np.diff(r)
    for i in range(len(force) - 1):
        if force[i] > 0 and force[i + 1] <= 0:
            weight = force[i] / (force[i] - force[i + 1])
            return float(mid[i] + weight * (mid[i + 1] - mid[i]))
    return 0.1

import numpy as np

def _student_design(features: "np.ndarray") -> "np.ndarray":
    x = np.asarray(features, dtype=np.float64)
    return np.column_stack((np.ones(len(x)), x, x[:, 0] * x[:, 1], x[:, 0] ** 2))


def _fixture(seed: int = 7261, pseudo_size: int = 2000) -> list[dict]:
    """Generate disclosed surrogate inputs for LGPS, LATP, and LYC order.

    All dimensions, random draws, and arrays in this routine are benchmark
    controls. They are not claimed to be trajectories or MACE weights from the
    source publication.
    """
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if isinstance(pseudo_size, bool) or not isinstance(pseudo_size, (int, np.integer)):
        raise ValueError("pseudo_size must be an integer")
    if pseudo_size < 100:
        raise ValueError("pseudo_size must be at least 100")
    rng = np.random.default_rng(seed)
    systems = []
    for system in range(3):
        initial = rng.uniform(1.0, 11.0, size=(200, 3))
        direction = rng.normal(size=(200, 3))
        direction /= np.linalg.norm(direction, axis=1)[:, None]
        time = np.arange(36, dtype=np.float64)
        displacement = time[:, None, None] * direction[None, :, :] * (0.0015 + 0.0001 * system)
        displacement[:, :8] *= 12
        trajectory = initial[None, :, :] + displacement
        aimd_positions = trajectory + rng.normal(scale=0.030, size=trajectory.shape)
        model_positions = aimd_positions + rng.normal(
            scale=(0.002 if system != 1 else 0.036), size=trajectory.shape
        )

        common = rng.normal(size=(160, 9))
        mace_embeddings = common + rng.normal(scale=0.22 + 0.02 * system, size=common.shape)
        aimd_embeddings = common + rng.normal(scale=0.14, size=common.shape)

        r = np.linspace(0.2, 1.5, 22)
        energy = 3.0 / r**2
        if system == 1:
            energy = energy - 3.0 * np.exp(-((r - 0.91) / 0.12) ** 2)

        pseudo_features = rng.normal(size=(pseudo_size, 6))
        test_features = rng.normal(size=(1000, 6))
        reference_coeff = np.array([0.55, -0.31, 0.20, 0.11, -0.19, 0.24]) + 0.02 * system
        teacher_coeff = reference_coeff + np.array([0.025, -0.018, 0.013, 0.0, 0.01, -0.007])
        dft_test = (
            test_features @ reference_coeff
            + 0.055 * np.sin(test_features[:, 0] * test_features[:, 1])
        )
        systems.append(
            {
                "name": ("LGPS", "LATP", "LYC")[system],
                "trajectory": trajectory,
                "non_li": np.arange(200) >= 8,
                "aimd_positions": aimd_positions,
                "model_positions": model_positions,
                "mace_embeddings": mace_embeddings,
                "aimd_embeddings": aimd_embeddings,
                "distance": r,
                "diatomic_energy": energy,
                "pseudo_features": pseudo_features,
                "teacher_coeff": teacher_coeff,
                "test_features": test_features,
                "dft_test": dft_test,
            }
        )
    return systems


def _teacher_label_pseudodata(
    pseudo_features: "np.ndarray", teacher_coeff: "np.ndarray"
) -> tuple["np.ndarray", "np.ndarray"]:
    """Label the supplied MD-feature panel by the supplied adapted teacher.

    No DFT labels are obtained in this operation. The fixed analytic teacher
    stands in for a neural adapted MACE model and must be named as such in
    every solver-facing description.
    """
    x, c = np.asarray(pseudo_features, float), np.asarray(teacher_coeff, float)
    if x.ndim != 2 or x.shape[0] < 100 or x.shape[1] != 6 or c.shape != (6,):
        raise ValueError("expected >=100 pseudo rows, six features, six coefficients")
    if not np.isfinite(x).all() or not np.isfinite(c).all():
        raise ValueError("nonfinite pseudo data")
    labels = x @ c + 0.055 * np.sin(x[:, 0] * x[:, 1])
    return x.copy(), labels


def _audit_details(seed: int = 7261, pseudo_size: int = 2000) -> tuple[float, dict]:
    """Run the seven source-informed stages and return the mean RDF threshold-excess burden and per-system evidence."""
    systems = _fixture(seed, pseudo_size)
    per_system = []
    for data in systems:
        msd, framework_ok = framework_msd_screen(
            data["trajectory"], data["non_li"], max_msd=0.02, start=10
        )
        coverage, ratios = feature_coverage(
            data["mace_embeddings"], data["aimd_embeddings"], bins=24
        )
        edges = np.linspace(0.0, 6.0, 25)
        aimd_rdf = radial_distribution(data["aimd_positions"], 12.0, edges)
        model_rdf = np.stack(
            [radial_distribution(frame[None], 12.0, edges) for frame in data["model_positions"]]
        )
        deviations, failure_time = rdf_stability(
            aimd_rdf, model_rdf, edges, frame_dt_ps=0.1
        )
        excess_burden = float(np.mean(np.maximum(deviations - 0.2, 0.0)))
        turning = diatomic_turning_point(data["distance"], data["diatomic_energy"])
        pseudo_x, pseudo_y = _teacher_label_pseudodata(
            data["pseudo_features"], data["teacher_coeff"]
        )
        train_design = _student_design(pseudo_x)
        ridge = 0.1
        gram = train_design.T @ train_design + ridge * np.eye(train_design.shape[1])
        coefficients = np.linalg.solve(gram, train_design.T @ pseudo_y)
        predicted = _student_design(data["test_features"]) @ coefficients
        error_percent = relative_rmse(data["dft_test"], predicted)
        per_system.append(
            {
                "system": data["name"],
                "framework_ok": framework_ok,
                "framework_terminal_msd": float(msd[-1]),
                "coverage": coverage,
                "retained_coordinates": int(len(ratios)),
                "rdf_max_deviation": float(deviations.max()),
                "rdf_excess_burden": excess_burden,
                "rdf_first_failure_ps": failure_time,
                "diatomic_turning_angstrom": turning,
                "pseudo_rows": int(len(pseudo_x)),
                "heldout_rrmse_percent": error_percent,
            }
        )
    result = float(np.mean([row["rdf_excess_burden"] for row in per_system]))
    return result, {"per_system": per_system, "mean_rdf_excess_burden": result}


def compute_source_method_audit(seed: int = 7261, pseudo_size: int = 2000) -> tuple[float, "np.ndarray"]:
    score, details = _audit_details(seed, pseudo_size)
    rows = details["per_system"]
    record = np.array([[float(row["framework_ok"]), row["framework_terminal_msd"], row["coverage"], row["rdf_max_deviation"], row["rdf_excess_burden"], -1.0 if row["rdf_first_failure_ps"] is None else row["rdf_first_failure_ps"], row["diatomic_turning_angstrom"], row["pseudo_rows"], row["heldout_rrmse_percent"]] for row in rows], dtype=np.float64)
    return score, record
SCICODE_GOLD_EOF
