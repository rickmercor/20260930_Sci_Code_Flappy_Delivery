"""
Apply the source-defined diagnostics to three deterministic synthetic systems and return their per-system audit record with J equal to the arithmetic mean of each system's complete-window RDF threshold-excess burden. For each system, average max(D − 0.2, 0) over its 27 complete one-picosecond windows, where D is the paper-defined integrated RDF discrepancy. The synthetic fixture and compact ridge student are disclosed stand-ins, not actual MACE, ELoRA, NEP or DFT results. The public orchestrator calls all six preceding public functions. Keep both maximum RDF deviations and held-out transition-energy RRMSEs in the native three-row record as separate diagnostics.

The synthetic fixture and compact ridge student are disclosed stand-ins, not actual MACE, ELoRA, NEP or DFT results. The stage calls all six preceding public functions. For each system, compute the complete-window RDF threshold-excess burden as the arithmetic mean of max(D − 0.2, 0) across its complete one-picosecond RDF windows. Return the source-method diagnostics in a native three-row record together with J, the arithmetic mean of the three systemwise threshold-excess burdens. Keep held-out transition-energy RRMSE as a separate record diagnostic; it is not the scalar returned as J. The protected reference calls only protected predecessors.

Returns
-------
tuple[float, np.ndarray], J (the mean RDF threshold-excess burden) and the (3, 9) float64 per-system audit record
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def compute_source_method_audit(seed: int = 7261, pseudo_size: int = 2000) -> tuple[float, "np.ndarray"]:
    """Return mean threshold-excess RDF burden and its per-system audit.

    Default seed is 7261 and default teacher-labelled pseudo-set size is
    2000. Seed must be a nonnegative integer; pseudo_size is an integer at
    least 100. Generate the fixed synthetic instance specified in the task's
    instance-data appendix. Call all six preceding public steps in system
    order LGPS, LATP, LYC. The teacher labels the pseudo-rows without a DFT
    call. A disclosed ridge student stand-in is fit only to those labels; it
    is never refit on the independent DFT-labelled test rows. Return the
    arithmetic mean of the three systemwise mean excesses max(D−0.2,0) across all complete one-picosecond RDF windows and a (3,9)
    float64 record. Rows are LGPS, LATP, LYC; columns are framework-pass
    flag (1/0), terminal framework MSD in square angstrom, coverage ratio,
    maximum RDF deviation, mean threshold-excess burden, first RDF failure time in ps (-1 if absent),
    diatomic turning position in angstrom, pseudo-row count, and held-out
    RRMSE percent, in that order.
    Raise ValueError on invalid seed or pseudo_size.
    """
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

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
        msd, framework_ok = _oracle_framework_msd_screen(
            data["trajectory"], data["non_li"], max_msd=0.02, start=10
        )
        coverage, ratios = _oracle_feature_coverage(
            data["mace_embeddings"], data["aimd_embeddings"], bins=24
        )
        edges = np.linspace(0.0, 6.0, 25)
        aimd_rdf = _oracle_radial_distribution(data["aimd_positions"], 12.0, edges)
        model_rdf = np.stack(
            [_oracle_radial_distribution(frame[None], 12.0, edges) for frame in data["model_positions"]]
        )
        deviations, failure_time = _oracle_rdf_stability(
            aimd_rdf, model_rdf, edges, frame_dt_ps=0.1
        )
        excess_burden = float(np.mean(np.maximum(deviations - 0.2, 0.0)))
        turning = _oracle_diatomic_turning_point(data["distance"], data["diatomic_energy"])
        pseudo_x, pseudo_y = _teacher_label_pseudodata(
            data["pseudo_features"], data["teacher_coeff"]
        )
        train_design = _student_design(pseudo_x)
        ridge = 0.1
        gram = train_design.T @ train_design + ridge * np.eye(train_design.shape[1])
        coefficients = np.linalg.solve(gram, train_design.T @ pseudo_y)
        predicted = _student_design(data["test_features"]) @ coefficients
        error_percent = _oracle_relative_rmse(data["dft_test"], predicted)
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


def _oracle_compute_source_method_audit(seed: int = 7261, pseudo_size: int = 2000) -> tuple[float, "np.ndarray"]:
    score, details = _audit_details(seed, pseudo_size)
    rows = details["per_system"]
    record = np.array([[float(row["framework_ok"]), row["framework_terminal_msd"], row["coverage"], row["rdf_max_deviation"], row["rdf_excess_burden"], -1.0 if row["rdf_first_failure_ps"] is None else row["rdf_first_failure_ps"], row["diatomic_turning_angstrom"], row["pseudo_rows"], row["heldout_rrmse_percent"]] for row in rows], dtype=np.float64)
    return score, record

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    setup = "import numpy as np\ndef _audit_pack(result):\n    score, record = result\n    return np.r_[score, float(record.shape == (3,9)), record.ravel()]\n"
    return [
        {"setup": setup, "call": "_audit_pack(compute_source_method_audit(7261,2000))", "gold_call": "_audit_pack(_oracle_compute_source_method_audit(7261,2000))", "tol": 1e-8},
        {"setup": setup, "call": "_audit_pack(compute_source_method_audit(7261,100))", "gold_call": "_audit_pack(_oracle_compute_source_method_audit(7261,100))", "tol": 1e-8},
        {"setup": setup, "call": "_audit_pack(compute_source_method_audit(7261,1200))", "gold_call": "_audit_pack(_oracle_compute_source_method_audit(7261,1200))", "tol": 1e-8},
        {"setup": setup, "call": "_audit_pack(compute_source_method_audit(801,2000))", "gold_call": "_audit_pack(_oracle_compute_source_method_audit(801,2000))", "tol": 1e-8},
        {"setup": setup, "call": "_audit_pack(compute_source_method_audit(0,2000))", "gold_call": "_audit_pack(_oracle_compute_source_method_audit(0,2000))", "tol": 1e-8},
        {"setup": 'import numpy as np\ndef _raises(fn):\n    try:\n        fn()\n    except ValueError:\n        return 1\n    return 0\n', "call": '_raises(lambda: compute_source_method_audit(7261,99))', "gold_call": '_raises(lambda: _oracle_compute_source_method_audit(7261,99))', "tol": 0},
    ]
