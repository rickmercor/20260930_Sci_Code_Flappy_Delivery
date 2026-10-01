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


def validate_rism_data(
    charged_h,
    charged_c,
    zero_h,
    zero_c,
    conformer_labels,
):
    arrays = {
        "charged_h": charged_h,
        "charged_c": charged_c,
        "zero_h": zero_h,
        "zero_c": zero_c,
    }

    for name, arr in arrays.items():
        if not isinstance(arr, np.ndarray):
            raise ValueError(f"{name} must be a NumPy array")

        if arr.ndim != 5:
            raise ValueError(
                f"{name} must have shape "
                "(n_conformers, n_sites, nx, ny, nz)"
            )

        if any(dim <= 0 for dim in arr.shape):
            raise ValueError(f"{name} contains an empty dimension")

        if not np.all(np.isfinite(arr)):
            raise ValueError(f"{name} contains non-finite values")

    if charged_h.shape != charged_c.shape:
        raise ValueError(
            "charged_h and charged_c must have identical shapes"
        )

    if zero_h.shape != zero_c.shape:
        raise ValueError(
            "zero_h and zero_c must have identical shapes"
        )

    if charged_h.shape != zero_h.shape:
        raise ValueError(
            "charged and zero-charge fields must have identical shapes"
        )

    labels = np.asarray(conformer_labels)

    if labels.ndim != 1:
        raise ValueError("conformer_labels must be one-dimensional")

    if len(labels) != charged_h.shape[0]:
        raise ValueError(
            "number of conformer labels must match number of conformers"
        )

    if len(labels) == 0:
        raise ValueError("at least one conformer is required")

import numpy as np


def compute_solvation_terms(
    h,
    c,
    grid_spacing,
    rho,
    temperature=298.15,
    k_b=0.00198720425864,
):
    h = np.asarray(h, dtype=float)
    c = np.asarray(c, dtype=float)
    rho = np.asarray(rho, dtype=float)

    if h.shape != c.shape:
        raise ValueError("h and c must have identical shapes")

    if h.ndim != 5:
        raise ValueError("h and c must be five-dimensional")

    if not np.all(np.isfinite(h)):
        raise ValueError("h contains non-finite values")

    if not np.all(np.isfinite(c)):
        raise ValueError("c contains non-finite values")

    if not np.isfinite(grid_spacing) or grid_spacing <= 0:
        raise ValueError("grid_spacing must be positive and finite")

    if rho.ndim == 0:
        rho = np.full(h.shape[1], float(rho), dtype=float)
    elif rho.ndim != 1 or rho.shape[0] != h.shape[1]:
        raise ValueError(
            "rho must be a scalar or a one-dimensional array "
            "with one density per solvent site"
        )

    if not np.all(np.isfinite(rho)) or np.any(rho <= 0):
        raise ValueError("rho must be positive and finite")

    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be positive and finite")

    if not np.isfinite(k_b) or k_b <= 0:
        raise ValueError("k_b must be positive and finite")

    kBT = k_b * temperature
    dV = grid_spacing ** 3
    rho_b = rho.reshape((1, -1, 1, 1, 1))
    axes = (1, 2, 3, 4)

    hnc = 0.5 * h**2 - c - 0.5 * c * h
    kh = 0.5 * h**2 * (h < 0) - c - 0.5 * c * h
    gf = -c - 0.5 * c * h

    sc_hnc = kBT * np.sum(rho_b * hnc, axis=axes) * dV
    sc_kh = kBT * np.sum(rho_b * kh, axis=axes) * dV
    gf_value = kBT * np.sum(rho_b * gf, axis=axes) * dV

    return np.column_stack([sc_hnc, sc_kh, gf_value])

import numpy as np


def compute_pmv(
    c,
    grid_spacing,
    rho,
    chi_kt,
):
    c = np.asarray(c, dtype=float)
    rho = np.asarray(rho, dtype=float)

    if c.ndim != 5:
        raise ValueError("c must be five-dimensional")

    if not np.all(np.isfinite(c)):
        raise ValueError("c contains non-finite values")

    if not np.isfinite(grid_spacing) or grid_spacing <= 0:
        raise ValueError("grid_spacing must be positive and finite")

    if not np.isfinite(chi_kt):
        raise ValueError("chi_kt must be finite")

    if rho.ndim == 0:
        rho = np.full(c.shape[1], float(rho), dtype=float)
    elif rho.ndim != 1 or rho.shape[0] != c.shape[1]:
        raise ValueError(
            "rho must be a scalar or a one-dimensional array "
            "with one density per solvent site"
        )

    if not np.all(np.isfinite(rho)) or np.any(rho <= 0):
        raise ValueError("rho must be positive and finite")

    dV = grid_spacing ** 3
    rho_b = rho.reshape((1, -1, 1, 1, 1))
    integral = np.sum(rho_b * c, axis=(1, 2, 3, 4)) * dV
    return chi_kt * (1.0 - integral)

import numpy as np


def build_descriptors(
    charged_terms,
    zero_terms,
    pmv_charged,
    pmv_zero,
):
    charged_terms = np.asarray(charged_terms, dtype=float)
    zero_terms = np.asarray(zero_terms, dtype=float)
    v = np.asarray(pmv_charged, dtype=float).reshape(-1)
    v0 = np.asarray(pmv_zero, dtype=float).reshape(-1)

    if charged_terms.ndim != 2 or charged_terms.shape[1] != 3:
        raise ValueError("charged_terms must have shape (n_conformers, 3)")
    if zero_terms.ndim != 2 or zero_terms.shape[1] != 3:
        raise ValueError("zero_terms must have shape (n_conformers, 3)")
    if charged_terms.shape != zero_terms.shape:
        raise ValueError("charged and zero-charge terms must have the same shape")
    n = charged_terms.shape[0]
    if v.shape != (n,) or v0.shape != (n,):
        raise ValueError("PMV arrays must have one entry per conformer")
    for name, arr in (
        ("charged_terms", charged_terms),
        ("zero_terms", zero_terms),
        ("pmv_charged", v),
        ("pmv_zero", v0),
    ):
        if not np.all(np.isfinite(arr)):
            raise ValueError(f"{name} contains non-finite values")

    sc_hnc, sc_kh, gf = charged_terms.T
    sc_hnc0, sc_kh0, gf0 = zero_terms.T

    full = np.column_stack([sc_hnc, sc_kh, gf, v, sc_hnc0, sc_kh0, gf0, v0])
    sub = np.column_stack([sc_kh, gf, sc_kh0, gf0, v0])
    hnc = np.column_stack([sc_hnc, sc_hnc0, v0])
    return full, sub, hnc

import numpy as np


def apply_milc(descriptors, coefficients):
    descriptors = np.asarray(descriptors, dtype=float)
    coefficients = np.asarray(coefficients, dtype=float).reshape(-1)

    if descriptors.ndim != 2:
        raise ValueError("descriptors must be two-dimensional")
    if coefficients.ndim != 1:
        raise ValueError("coefficients must be one-dimensional")
    if descriptors.shape[1] != coefficients.shape[0]:
        raise ValueError(
            "descriptor columns must match the coefficient vector length"
        )
    if descriptors.shape[0] == 0:
        raise ValueError("at least one conformer is required")
    if not np.all(np.isfinite(descriptors)):
        raise ValueError("descriptors contain non-finite values")
    if not np.all(np.isfinite(coefficients)):
        raise ValueError("coefficients contain non-finite values")

    return descriptors @ coefficients

import numpy as np


def average_conformers(predictions):
    predictions = np.asarray(predictions, dtype=float).reshape(-1)
    if predictions.size == 0:
        raise ValueError("at least one conformer is required")
    if not np.all(np.isfinite(predictions)):
        raise ValueError("predictions contain non-finite values")
    return float(np.mean(predictions))

import numpy as np


def run_milc_pipeline(
    charged_h,
    charged_c,
    zero_h,
    zero_c,
    conformer_labels,
    grid_spacing,
    rho,
    chi_kt,
    coefficients,
    temperature=298.15,
    k_b=0.00198720425864,
    descriptor_set="sub",
) -> float:
    """Deterministic reference implementation for the complete MILC pipeline."""

    # Step 01: Validate inputs
    validate_rism_data(
        charged_h,
        charged_c,
        zero_h,
        zero_c,
        conformer_labels,
    )

    # Step 02: Solvation functionals
    # Charged and zero-charge states are independent evaluations.
    charged_terms = compute_solvation_terms(
        charged_h,
        charged_c,
        grid_spacing,
        rho,
        temperature,
        k_b,
    )
    zero_terms = compute_solvation_terms(
        zero_h,
        zero_c,
        grid_spacing,
        rho,
        temperature,
        k_b,
    )

    if charged_terms.shape != zero_terms.shape:
        raise ValueError(
            "charged and zero-charge solvation terms must have "
            "the same shape"
        )
    if charged_terms.ndim != 2 or charged_terms.shape[1] != 3:
        raise ValueError(
            "solvation terms must have shape (n_conformers, 3)"
        )
    if charged_terms.shape[0] != np.asarray(charged_h).shape[0]:
        raise ValueError(
            "solvation terms must contain one row per conformer"
        )
    if not np.all(np.isfinite(charged_terms)):
        raise ValueError("charged solvation terms contain non-finite values")
    if not np.all(np.isfinite(zero_terms)):
        raise ValueError(
            "zero-charge solvation terms contain non-finite values"
        )

    # Step 03: Partial molar volume from the direct correlation
    pmv_charged = compute_pmv(
        charged_c,
        grid_spacing,
        rho,
        chi_kt,
    )
    pmv_zero = compute_pmv(
        zero_c,
        grid_spacing,
        rho,
        chi_kt,
    )

    n_conformers = charged_terms.shape[0]
    pmv_charged = np.asarray(pmv_charged, dtype=float).reshape(-1)
    pmv_zero = np.asarray(pmv_zero, dtype=float).reshape(-1)
    if pmv_charged.shape != (n_conformers,) or pmv_zero.shape != (
        n_conformers,
    ):
        raise ValueError("PMV arrays must have one entry per conformer")
    if not np.all(np.isfinite(pmv_charged)):
        raise ValueError("charged PMV contains non-finite values")
    if not np.all(np.isfinite(pmv_zero)):
        raise ValueError("zero-charge PMV contains non-finite values")

    # Step 04: Descriptor matrices
    full, sub, hnc = build_descriptors(
        charged_terms,
        zero_terms,
        pmv_charged,
        pmv_zero,
    )

    if full.ndim != 2 or full.shape != (n_conformers, 8):
        raise ValueError("Full descriptors must have shape (n_conformers, 8)")
    if sub.ndim != 2 or sub.shape != (n_conformers, 5):
        raise ValueError("Sub descriptors must have shape (n_conformers, 5)")
    if hnc.ndim != 2 or hnc.shape != (n_conformers, 3):
        raise ValueError("HNC descriptors must have shape (n_conformers, 3)")
    if not np.all(np.isfinite(full)):
        raise ValueError("Full descriptors contain non-finite values")
    if not np.all(np.isfinite(sub)):
        raise ValueError("Sub descriptors contain non-finite values")
    if not np.all(np.isfinite(hnc)):
        raise ValueError("HNC descriptors contain non-finite values")

    key = str(descriptor_set).lower()
    if key == "sub":
        descriptors = sub
    elif key == "hnc":
        descriptors = hnc
    elif key == "full":
        descriptors = full
    else:
        raise ValueError("descriptor_set must be 'sub', 'hnc', or 'full'")

    # Step 05: Zero-intercept MILC inner product
    predictions = apply_milc(descriptors, coefficients)

    predictions = np.asarray(predictions, dtype=float).reshape(-1)
    if predictions.shape != (n_conformers,):
        raise ValueError(
            "MILC predictions must contain one value per conformer"
        )
    if not np.all(np.isfinite(predictions)):
        raise ValueError("MILC predictions contain non-finite values")

    # Step 06: Arithmetic conformer average
    delta_g_milc = average_conformers(predictions)
    return float(delta_g_milc)
SCICODE_GOLD_EOF
