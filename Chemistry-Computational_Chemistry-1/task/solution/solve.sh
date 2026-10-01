#!/bin/bash
# Oracle: writes the gold solution so the verifier scores reward 1.0.
cat > /app/solution.py <<'SCICODE_GOLD_EOF'
import numpy as np
try:
    import scipy, scipy.linalg, scipy.sparse, scipy.optimize, scipy.spatial, scipy.integrate, scipy.stats, scipy.special, scipy.signal
except Exception:
    pass
import math

def evaluate_reference_surface(
    X,
    scale=0.05,
    kappa=2.0,
    xc=(-0.5, 0.75),
):
    """Reference implementation for evaluate_reference_surface."""
    import numpy as np

    amplitude = np.array([-200.0, -100.0, -170.0, 15.0])
    a_coeff = np.array([-1.0, -1.0, -6.5, 0.7])
    b_coeff = np.array([0.0, 0.0, 11.0, 0.6])
    c_coeff = np.array([-10.0, -10.0, -6.5, 0.7])
    x_centre = np.array([1.0, 0.0, -0.5, -1.0])
    y_centre = np.array([0.0, 0.5, 1.5, 1.0])
    terms = np.arange(4)

    points = np.asarray(X, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError(
            "X must be a two-dimensional array of shape (n_points, 2)"
        )
    if not np.all(np.isfinite(points)):
        raise ValueError("X must contain only finite values")
    if not (np.isfinite(scale) and float(scale) > 0.0):
        raise ValueError("scale must be a finite number > 0")
    if not (np.isfinite(kappa) and float(kappa) > 0.0):
        raise ValueError("kappa must be a finite number > 0")

    centre = np.asarray(xc, dtype=float)
    if centre.shape != (2,) or not np.all(np.isfinite(centre)):
        raise ValueError("xc must hold two finite components")

    dx = points[:, [0]] - x_centre[None, terms]
    dy = points[:, [1]] - y_centre[None, terms]
    gauss = amplitude[None, terms] * np.exp(
        a_coeff[terms] * dx**2 + b_coeff[terms] * dx * dy + c_coeff[terms] * dy**2
    )
    offset = points - centre[None, :]

    value = float(scale) * gauss.sum(axis=1) + 0.5 * float(kappa) * np.sum(
        offset**2, axis=1
    )
    grad_x = float(scale) * (
        gauss * (2.0 * a_coeff[terms] * dx + b_coeff[terms] * dy)
    ).sum(axis=1) + float(kappa) * offset[:, 0]
    grad_y = float(scale) * (
        gauss * (b_coeff[terms] * dx + 2.0 * c_coeff[terms] * dy)
    ).sum(axis=1) + float(kappa) * offset[:, 1]

    return np.stack([value, grad_x, grad_y], axis=1)

def evaluate_reduced_surface(
    X,
    scale=0.05,
    kappa=2.0,
    xc=(-0.5, 0.75),
):
    """Reference implementation for evaluate_reduced_surface."""
    import numpy as np

    amplitude = np.array([-200.0, -100.0, -170.0, 15.0])
    a_coeff = np.array([-1.0, -1.0, -6.5, 0.7])
    b_coeff = np.array([0.0, 0.0, 11.0, 0.6])
    c_coeff = np.array([-10.0, -10.0, -6.5, 0.7])
    x_centre = np.array([1.0, 0.0, -0.5, -1.0])
    y_centre = np.array([0.0, 0.5, 1.5, 1.0])
    terms = np.array([0, 2])

    points = np.asarray(X, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError(
            "X must be a two-dimensional array of shape (n_points, 2)"
        )
    if not np.all(np.isfinite(points)):
        raise ValueError("X must contain only finite values")
    if not (np.isfinite(scale) and float(scale) > 0.0):
        raise ValueError("scale must be a finite number > 0")
    if not (np.isfinite(kappa) and float(kappa) > 0.0):
        raise ValueError("kappa must be a finite number > 0")

    centre = np.asarray(xc, dtype=float)
    if centre.shape != (2,) or not np.all(np.isfinite(centre)):
        raise ValueError("xc must hold two finite components")

    dx = points[:, [0]] - x_centre[None, terms]
    dy = points[:, [1]] - y_centre[None, terms]
    gauss = amplitude[None, terms] * np.exp(
        a_coeff[terms] * dx**2 + b_coeff[terms] * dx * dy + c_coeff[terms] * dy**2
    )
    offset = points - centre[None, :]

    value = float(scale) * gauss.sum(axis=1) + 0.5 * float(kappa) * np.sum(
        offset**2, axis=1
    )
    grad_x = float(scale) * (
        gauss * (2.0 * a_coeff[terms] * dx + b_coeff[terms] * dy)
    ).sum(axis=1) + float(kappa) * offset[:, 0]
    grad_y = float(scale) * (
        gauss * (b_coeff[terms] * dx + 2.0 * c_coeff[terms] * dy)
    ).sum(axis=1) + float(kappa) * offset[:, 1]

    return np.stack([value, grad_x, grad_y], axis=1)

def compute_propagated_means(
    p_prev,
    force,
    dt,
    gamma,
):
    """Reference implementation for compute_propagated_means."""
    import numpy as np

    p = np.asarray(p_prev, float); F = np.asarray(force, float)
    if p.shape != F.shape: raise ValueError("p_prev and force must match in shape")
    if p.ndim != 2 or p.shape[1] != 2: raise ValueError("inputs must be (n_points, 2)")
    if dt <= 0 or gamma <= 0: raise ValueError("dt and gamma must be > 0")
    a = np.exp(-gamma*dt)
    return a*p + (1.0 + a)*(dt/2.0)*F

def compute_separation_fractions(
    mean_target,
    mean_draft,
    dt,
    gamma,
    mass,
    kbt,
):
    """Reference implementation for compute_separation_fractions."""
    import numpy as np
    from math import erf

    mt = np.asarray(mean_target, float); md = np.asarray(mean_draft, float)
    if mt.shape != md.shape: raise ValueError("mean arrays must match in shape")
    if mt.ndim != 2 or mt.shape[1] != 2: raise ValueError("inputs must be (n_points, 2)")
    if min(dt, gamma, mass, kbt) <= 0: raise ValueError("dt, gamma, mass and kbt must be > 0")
    sig2 = mass*kbt*(1.0 - np.exp(-2.0*gamma*dt))
    dn = np.sqrt(np.sum((md - mt)**2, axis=1)/sig2)
    return np.array([erf(v/np.sqrt(8.0)) for v in dn])

def derive_coordinate_geometry(
    X,
    c,
    k,
):
    """Reference implementation for derive_coordinate_geometry."""
    import numpy as np

    P = np.asarray(X, float)
    if P.ndim != 2 or P.shape[1] != 2: raise ValueError("X must be (n_points, 2)")
    xi = P[:, 0] + c*np.sin(k*P[:, 1])
    g  = c*k*np.cos(k*P[:, 1]); s = 1.0 + g**2
    gp = -c*k*k*np.sin(k*P[:, 1])
    return np.stack([xi, 1.0/s, g/s, gp*(1.0 - g**2)/s**2], axis=1)

def assemble_gradient_terms(
    grad_U,
    cv_geom,
):
    """Reference implementation for assemble_gradient_terms."""
    import numpy as np

    G = np.asarray(grad_U, float); C = np.asarray(cv_geom, float)
    if G.ndim != 2 or G.shape[1] != 2: raise ValueError("grad_U must be (n_points, 2)")
    if C.shape[0] != G.shape[0] or C.shape[1] != 4: raise ValueError("cv_geom must be (n_points, 4)")
    return C[:, 1]*G[:, 0] + C[:, 2]*G[:, 1] - C[:, 3]

def aggregate_binned_averages(
    cv,
    values,
    logw,
    edges,
):
    """Reference implementation for aggregate_binned_averages."""
    import numpy as np

    y = np.asarray(cv, float); v = np.asarray(values, float); a = np.asarray(logw, float)
    e = np.asarray(edges, float)
    if not (y.shape == v.shape == a.shape): raise ValueError("cv, values and logw must match in shape")
    if e.ndim != 1 or e.size < 2: raise ValueError("edges must be 1-D with >= 2 entries")
    idx = np.digitize(y, e) - 1
    out = np.full(e.size - 1, np.nan)
    for b in range(e.size - 1):
        sel = idx == b
        if not np.any(sel): continue
        aa = a[sel]; w = np.exp(aa - aa.max())
        out[b] = float(np.sum(w*v[sel])/np.sum(w))
    return out

def resolve_profile_scalar(
    x_lo,
    x_hi,
    y_lo,
    y_hi,
    n_grid,
    n_bins,
    cv_lo,
    cv_hi,
    c,
    k,
    dt,
    gamma,
    mass,
    kbt,
    scale=0.05,
    kappa=2.0,
    xc=(-0.5, 0.75),
):
    """Reference implementation for resolve_profile_scalar."""
    # Local imports keep the oracle self-contained when the harness executes it
    # in isolation.
    import glob
    import importlib.util
    import os
    import sys

    import numpy as np

    # -- Resolve the oracle functions of sub-problems 01-07. Preference order:
    #    (1) already present in the executing namespace (shared-namespace
    #    harness), (2) loaded from a sibling sub-problem file matched by name.
    #    There is deliberately no fallback to the public names: in a shared
    #    namespace those are the candidate's implementations, and falling back
    #    to them would let the gold side of the comparison execute candidate
    #    code. If no oracle can be found the orchestrator fails loudly.
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
        raise RuntimeError("cannot resolve required step function " + oracle_name)

    evaluate_reference_surface = _resolve_step(
        "evaluate_reference_surface", "*evaluate_reference_surface*.py")
    evaluate_reduced_surface = _resolve_step(
        "evaluate_reduced_surface", "*evaluate_reduced_surface*.py")
    compute_propagated_means = _resolve_step(
        "compute_propagated_means", "*compute_propagated_means*.py")
    compute_separation_fractions = _resolve_step(
        "compute_separation_fractions", "*compute_separation_fractions*.py")
    derive_coordinate_geometry = _resolve_step(
        "derive_coordinate_geometry", "*derive_coordinate_geometry*.py")
    assemble_gradient_terms = _resolve_step(
        "assemble_gradient_terms", "*assemble_gradient_terms*.py")
    aggregate_binned_averages = _resolve_step(
        "aggregate_binned_averages", "*aggregate_binned_averages*.py")

    n = int(n_grid)
    if n < 2 or int(n_bins) < 2: raise ValueError("n_grid and n_bins must be >= 2")
    gx = x_lo + (np.arange(n) + 0.5)*(x_hi - x_lo)/n
    gy = y_lo + (np.arange(n) + 0.5)*(y_hi - y_lo)/n
    XX, YY = np.meshgrid(gx, gy, indexing="ij")
    P = np.stack([XX.ravel(), YY.ravel()], axis=1)
    tg = evaluate_reference_surface(P, scale, kappa, xc)
    dr = evaluate_reduced_surface(P, scale, kappa, xc)
    lw = -(tg[:, 0] - tg[:, 0].min())
    p0 = np.zeros_like(P)
    mt = compute_propagated_means(p0, -tg[:, 1:3], dt, gamma)   # force = -gradient
    md = compute_propagated_means(p0, -dr[:, 1:3], dt, gamma)
    beta = compute_separation_fractions(mt, md, dt, gamma, mass, kbt)
    geom = derive_coordinate_geometry(P, c, k)
    D    = assemble_gradient_terms(tg[:, 1:3], geom)
    edges = np.linspace(cv_lo, cv_hi, int(n_bins) + 1)
    mf = aggregate_binned_averages(geom[:, 0], D,    lw, edges)
    mb = aggregate_binned_averages(geom[:, 0], beta, lw, edges)
    if np.any(np.isnan(mf)) or np.any(np.isnan(mb)): raise ValueError("an empty CV bin left a conditional average undefined")
    centers = 0.5*(edges[:-1] + edges[1:]); h = centers[1] - centers[0]
    F = np.concatenate([[0.0], np.cumsum(0.5*h*(mf[:-1] + mf[1:]))])
    return float(F[int(np.argmax(mb))] - F[int(np.argmin(mb))])
SCICODE_GOLD_EOF
