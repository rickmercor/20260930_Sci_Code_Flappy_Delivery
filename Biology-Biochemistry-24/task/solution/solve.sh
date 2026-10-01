#!/bin/bash
# Oracle: writes the gold solution so the verifier scores reward 1.0.
cat > /app/solution.py <<'SCICODE_GOLD_EOF'
import numpy as np
try:
    import scipy, scipy.linalg, scipy.sparse, scipy.optimize, scipy.spatial, scipy.integrate, scipy.stats, scipy.special, scipy.signal
except Exception:
    pass
import math

def compute_cell_energy(area: float, perimeter: float, kappa: float, chi: float) -> float:
    import numpy as np
    def _num(name, value):
        try:
            value = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} must be a real number, got {value!r}") from exc
        if not np.isfinite(value):
            raise ValueError(f"{name} must be finite, got {value!r}")
        return value
    area = _num("area", area)
    perimeter = _num("perimeter", perimeter)
    kappa = _num("kappa", kappa)
    chi = _num("chi", chi)
    if area < 0.0:
        raise ValueError(f"area must be non-negative, got {area!r}")
    if perimeter < 0.0:
        raise ValueError(f"perimeter must be non-negative, got {perimeter!r}")
    return float(0.5 * ((area - 1.0)**2 + kappa * (perimeter - chi)**2))

def compute_vertex_forces(vertices: np.ndarray, kappa: float, chi: float) -> np.ndarray:
    import numpy as np

    def _num(name, value):
        try:
            value = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} must be a real number, got {value!r}") from exc
        if not np.isfinite(value):
            raise ValueError(f"{name} must be finite, got {value!r}")
        return value

    def _poly(name, value, min_vertices=3):
        value = np.asarray(value, dtype=float)
        if value.ndim != 2 or value.shape[1] != 2:
            raise ValueError(f"{name} must have shape (n, 2), got {value.shape}")
        if value.shape[0] < min_vertices:
            raise ValueError(f"{name} needs at least {min_vertices} rows, got {value.shape[0]}")
        if not np.all(np.isfinite(value)):
            raise ValueError(f"{name} must contain only finite values")
        return value

    vertices = _poly("vertices", vertices)
    kappa = _num("kappa", kappa)
    chi = _num("chi", chi)

    V = vertices.shape[0]
    x = vertices[:, 0]
    y = vertices[:, 1]
    
    area = 0.5 * np.sum(x * np.roll(y, -1) - y * np.roll(x, -1))
    
    dx = np.roll(x, -1) - x
    dy = np.roll(y, -1) - y
    edge_lengths = np.sqrt(dx**2 + dy**2)
    perimeter = np.sum(edge_lengths)
    L = edge_lengths + 1e-12
    
    dE_dA = area - 1.0
    dE_dP = kappa * (perimeter - chi)
    
    forces = np.zeros_like(vertices)

    for i in range(V):
        prev_i = (i - 1) % V
        next_i = (i + 1) % V

        dA_dxi = 0.5 * (y[next_i] - y[prev_i])
        dA_dyi = 0.5 * (x[prev_i] - x[next_i])
        
        dP_dxi = (
            (x[i] - x[prev_i]) / L[prev_i]
            + (x[i] - x[next_i]) / L[i]
        )
        dP_dyi = (
            (y[i] - y[prev_i]) / L[prev_i]
            + (y[i] - y[next_i]) / L[i]
        )
        
        fx = -(dE_dA * dA_dxi + dE_dP * dP_dxi)
        fy = -(dE_dA * dA_dyi + dE_dP * dP_dyi)
        forces[i] = [fx, fy]
        
    return forces

def update_vertex_positions(
    vertices: np.ndarray,
    forces: np.ndarray,
    gamma: float,
    dt: float
) -> np.ndarray:
    import numpy as np

    def _num(name, value):
        try:
            value = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} must be a real number, got {value!r}") from exc
        if not np.isfinite(value):
            raise ValueError(f"{name} must be finite, got {value!r}")
        return value

    def _poly(name, value, min_vertices=3):
        value = np.asarray(value, dtype=float)
        if value.ndim != 2 or value.shape[1] != 2:
            raise ValueError(f"{name} must have shape (n, 2), got {value.shape}")
        if value.shape[0] < min_vertices:
            raise ValueError(f"{name} needs at least {min_vertices} rows, got {value.shape[0]}")
        if not np.all(np.isfinite(value)):
            raise ValueError(f"{name} must contain only finite values")
        return value

    vertices = _poly("vertices", vertices)
    forces = _poly("forces", forces)

    if forces.shape != vertices.shape:
        raise ValueError(
            f"forces must match vertices in shape, got {forces.shape} and "
            f"{vertices.shape}"
        )

    gamma = _num("gamma", gamma)
    dt = _num("dt", dt)

    if gamma == 0.0:
        raise ValueError("gamma must be non-zero; the overdamped update divides by it")

    displacement = (forces / gamma) * dt
    return vertices + displacement

def check_t1_transitions(vertices: np.ndarray, d_T: float) -> float:
    import numpy as np
    def _num(name, value):
        try:
            value = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} must be a real number, got {value!r}") from exc
        if not np.isfinite(value):
            raise ValueError(f"{name} must be finite, got {value!r}")
        return value
    def _poly(name, value, min_vertices=3):
        value = np.asarray(value, dtype=float)
        if value.ndim != 2 or value.shape[1] != 2:
            raise ValueError(f"{name} must have shape (n, 2), got {value.shape}")
        if value.shape[0] < min_vertices:
            raise ValueError(f"{name} needs at least {min_vertices} rows, got {value.shape[0]}")
        if not np.all(np.isfinite(value)):
            raise ValueError(f"{name} must contain only finite values")
        return value
    vertices = _poly("vertices", vertices)
    d_T = _num("d_T", d_T)
    x = vertices[:, 0]
    y = vertices[:, 1]
    dx = np.roll(x, -1) - x
    dy = np.roll(y, -1) - y
    L = np.sqrt(dx**2 + dy**2)
    num_transitions = np.sum(L < d_T)
    return float(num_transitions)

def compute_virial_stress(vertices: np.ndarray, forces: np.ndarray) -> float:
    import numpy as np
    def _poly(name, value, min_vertices=3):
        value = np.asarray(value, dtype=float)
        if value.ndim != 2 or value.shape[1] != 2:
            raise ValueError(f"{name} must have shape (n, 2), got {value.shape}")
        if value.shape[0] < min_vertices:
            raise ValueError(f"{name} needs at least {min_vertices} rows, got {value.shape[0]}")
        if not np.all(np.isfinite(value)):
            raise ValueError(f"{name} must contain only finite values")
        return value
    vertices = _poly("vertices", vertices)
    forces = _poly("forces", forces)
    if forces.shape != vertices.shape:
        raise ValueError(f"forces must match vertices in shape, got {forces.shape} and "
                         f"{vertices.shape}")
    x = vertices[:, 0]
    y = vertices[:, 1]
    area = 0.5 * np.abs(np.sum(x * np.roll(y, -1) - y * np.roll(x, -1)))
    if area == 0:
        return 0.0
    
    sigma_xx = np.sum(vertices[:, 0] * forces[:, 0]) / area
    sigma_yy = np.sum(vertices[:, 1] * forces[:, 1]) / area
    
    return float(sigma_xx + sigma_yy)

def compute_maxwell_stress(lambdas: np.ndarray, stresses: np.ndarray, lambda_U: float, lambda_N: float) -> float:
        import numpy as np
        def _num(name, value):
            try:
                value = float(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{name} must be a real number, got {value!r}") from exc
            if not np.isfinite(value):
                raise ValueError(f"{name} must be finite, got {value!r}")
            return value
        lambdas = np.asarray(lambdas, dtype=float)
        stresses = np.asarray(stresses, dtype=float)
        if lambdas.ndim != 1 or stresses.ndim != 1:
            raise ValueError("lambdas and stresses must be one-dimensional")
        if lambdas.shape != stresses.shape:
            raise ValueError(f"lambdas and stresses must be the same length, got {lambdas.shape} and "
                             f"{stresses.shape}")
        if not (np.all(np.isfinite(lambdas)) and np.all(np.isfinite(stresses))):
            raise ValueError("lambdas and stresses must contain only finite values")
        lambda_U = _num("lambda_U", lambda_U)
        lambda_N = _num("lambda_N", lambda_N)
        mask = (lambdas >= lambda_U) & (lambdas <= lambda_N)
        if not np.any(mask):
            return 0.0
        
        l_sub = lambdas[mask]
        s_sub = stresses[mask]
        
        if len(l_sub) < 2 or lambda_N == lambda_U:
            return 0.0
            
        integral = np.trapezoid(s_sub, l_sub)
        s_star = integral / (lambda_N - lambda_U)
        return float(s_star)

def orchestrator_pipeline(
    kappa: float,
    chi: float,
    gamma: float,
    dt: float,
    d_T: float
) -> float:
    import numpy as np

    def _num(name, value):
        try:
            value = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} must be a real number, got {value!r}") from exc
        if not np.isfinite(value):
            raise ValueError(f"{name} must be finite, got {value!r}")
        return value

    kappa = _num("kappa", kappa)
    chi = _num("chi", chi)
    gamma = _num("gamma", gamma)
    dt = _num("dt", dt)
    d_T = _num("d_T", d_T)

    if gamma == 0.0:
        raise ValueError("gamma must be non-zero; the overdamped update divides by it")
    
    # Initialize a regular hexagon
    angles = np.linspace(0, 2 * np.pi, 6, endpoint=False)
    vertices = np.column_stack((np.cos(angles), np.sin(angles)))
    
    # Step 1: Compute initial energy
    x = vertices[:, 0]
    y = vertices[:, 1]
    area = 0.5 * np.abs(
        np.sum(x * np.roll(y, -1) - y * np.roll(x, -1))
    )

    dx = np.roll(x, -1) - x
    dy = np.roll(y, -1) - y
    perimeter = np.sum(np.sqrt(dx**2 + dy**2))
    
    energy = compute_cell_energy(
        area,
        perimeter,
        kappa,
        chi
    )
    
    # Step 2: Compute forces and derive the required scalar summary
    forces = compute_vertex_forces(
        vertices,
        kappa,
        chi
    )
    force_mag = float(
        np.sum(np.linalg.norm(forces, axis=1))
    )
    
    # Step 3: Update positions and derive the required scalar summary
    new_vertices = update_vertex_positions(
        vertices,
        forces,
        gamma,
        dt
    )
    max_disp = float(
        np.max(np.linalg.norm(new_vertices - vertices, axis=1))
    )
    
    # Step 4: Check T1 transitions
    num_t1 = check_t1_transitions(
        new_vertices,
        d_T
    )
    
    # Step 5: Compute Virial stress
    stress_trace = compute_virial_stress(
        new_vertices,
        forces
    )
    
    # Step 6: Compute Maxwell stress
    lambdas = np.linspace(1.0, 2.0, 11)
    stresses = stress_trace * lambdas

    s_star = compute_maxwell_stress(
        lambdas,
        stresses,
        1.2,
        1.8
    )
    
    return float(
        energy
        + force_mag
        + max_disp
        + num_t1
        + stress_trace
        + s_star
    )
SCICODE_GOLD_EOF
