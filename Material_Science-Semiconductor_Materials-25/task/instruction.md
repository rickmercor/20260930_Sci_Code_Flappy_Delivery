# Material_Science-Semiconductor_Materials-25

## Background

In a doped semiconductor junction the doping levels set almost everything about how the device behaves: the potential barrier between the two sides, the width of the region swept clear of mobile carriers, the peak electric field that limits breakdown, and the current that flows under forward bias. The forward current is governed by minority carriers injected across the junction, whose equilibrium population falls as the doping rises and depends on the intrinsic carrier density of the material, itself set by the band gap and the temperature.Classical device theory predicts these quantities in closed form by assuming that the transition region has sharp edges and contains no mobile charge, and that injection is weak enough to leave the majority populations unchanged. Those assumptions are convenient rather than exact: mobile carriers spill into the transition region over a screening length, the edges are smooth, and the quasi-Fermi levels are not perfectly flat across the junction. Comparing full transport simulation with the closed-form theory shows where the simple picture holds and where it breaks down.Simulating the transport accurately is itself demanding. Carrier densities vary over many orders of magnitude across a junction and the potential changes steeply, so a discretisation that does not respect the exponential character of the solution produces oscillations or negative densities. Schemes that do respect it have long been standard on meshes with a particular geometric property, and extending them to the distorted, general meshes that real device geometries require is an active area of numerical analysis.

## Problem

Semiconductor device simulation predicts how a doped silicon junction behaves by solving the stationary drift-diffusion equations, which couple the Poisson equation for the electrostatic potential to continuity equations for electrons and holes. Near a PN junction the potential changes steeply, so carrier transport is drift-dominated and a plain central discretisation is unstable. The standard exponentially fitted finite volume remedy relies on Voronoi control volumes and therefore on a Delaunay triangulation, which real device meshes do not always provide. Recent work removes that restriction by placing unknowns on both the primal cells and the vertices of a general triangulation and reconstructing fluxes on diamond cells, with harmonic averaging of the exponential of the potential along both the primal and the dual edge direction, so that each direction carries a Scharfetter-Gummel-type factor and the scheme stays locally conservative on non-Delaunay meshes. Your task is to apply that scheme to a silicon PN junction and quantify how far the simulated device behaviour departs from classical textbook theory as the doping is varied.Device and material. The domain is the square [0, 1 um] x [0, 1 um] with a symmetric abrupt junction at y = 0.5 um. The net doping, evaluated at the position of each unknown, is N = +N0 for y < 0.5 um, N = -N0 for y > 0.5 um and N = 0 at y = 0.5 um, for N0 = 1e16, 3e16 and 1e17 cm^-3. The contacts at y = 0 (0 V) and y = 1 um (applied voltage Va) are Ohmic: the majority density is (|N| + sqrt(N^2 + 4 n_ie^2))/2, the minority density is n_ie^2 divided by the majority density, and psi = V_applied + V_T ln(n/n_ie). The sides x = 0 and x = 1 um are insulating, with zero normal field and zero normal electron and hole current. Recombination is neglected, R_n = R_p = 0. Silicon at 300 K: q = 1.602192e-19 C, V_T = 0.025852 V, eps = 1.035941e-12 C/(V cm), n_ie = 1.087386e10 cm^-3, mu_n = 1417.0 and mu_p = 470.5 cm^2/(V s), with D = V_T mu.Mesh and solution. Use M = 64. Place vertices at (x_i, y_j) = (i h, j h) with h = 1 um / M and i, j = 0..M, then shift every vertex with 0 < i < M in odd rows j by +0.4 h in x, and split each cell (i,j)-(i+1,j)-(i+1,j+1)-(i,j+1) into two triangles along the diagonal from (i,j) to (i+1,j+1); about one interior edge in six then violates the Delaunay condition. Unknowns psi, n and p live on the primal cells (triangle barycentres, with boundary edges acting as degenerate primal cells at their midpoints) and on the vertices. Solve the coupled nonlinear system by Newton's method from the charge-neutral equilibrium state, ramping Va from 0 V, resetting any negative density to 1e-20 cm^-3, and stopping when the largest Newton update is below 1e-10 in psi/V_T and below 1e-6 relative in n and p.

Quantities and conventions. All row averages below are over the vertices of a horizontal vertex row, which all share the same y. At Va = 0, the depletion width is W_sim = y_p - y_n, where y_n is where the row-averaged n first falls to N0/2 on the n-side and y_p is where the row-averaged p first reaches N0/2 on the p-side, each by linear interpolation between the bracketing rows; the peak field is E_sim = max_j |psi_j+1 - psi_j| / h + q N0 h / (2 eps) in V/cm, the second term moving the finite-difference field from midway between rows onto the junction. At Va = 0.5 V, J_sim is the current density through the top contact, obtained from the primal-mesh electron and hole fluxes there and positive for conventional current entering the device. The textbook comparisons use V_bi = V_T ln(N0^2 / n_ie^2), the depletion-approximation values W_DA = sqrt(4 eps V_bi / (q N0)) and E_DA = q N0 W_DA / (2 eps), and the short-diode law J_SD = q (n_ie^2 / N0) (exp(Va/V_T) - 1) (D_n + D_p) / W_prime, in which the neutral width on each side is W_prime = 0.5 um - sqrt(4 eps (V_bi - Va) / (q N0)) / 2, that is half the device minus half the depletion width evaluated at the applied bias. Signed percentage differences are 100 (X_sim / X_textbook - 1).Report the supporting table with one row per N0 = 1e16, 3e16, 1e17 cm^-3 and the columns in the order W_sim (um), E_sim (V/cm), J_sim (A/cm^2), W_DA (um), E_DA (V/cm), J_SD (A/cm^2), then the signed percentage differences for W, E and J; explain how each quantity scales with N0 and why the simulation departs from the textbook formulas. Your final answer must be a single number: the simulated forward current density J_sim at Va = 0.5 V for N0 = 1e16 cm^-3, in A/cm^2.

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

01_bernoulli

Goal
----
Evaluate the Bernoulli function B(t) = t/(e^t - 1) element-wise, accurately for every real t.

```python
def bernoulli(t: "np.ndarray") -> "np.ndarray":
    """Bernoulli function B(t) = t / (exp(t) - 1), applied element-wise.

    Parameters
    ----------
    t : "np.ndarray"
        Real array of any shape (a potential drop divided by the thermal voltage).

    Returns
    -------
    B : "np.ndarray"
        Same shape as t, float. B(0) = 1. Must be accurate near t = 0 and must
        not overflow or return nan for |t| up to at least 800.

    Notes
    -----
    Include every import your implementation needs inside the function body.
    """
    return B  # placeholder
```

### Step 2

02_contact_densities

Goal
----
Compute the charge-neutral thermal-equilibrium potential and carrier densities of silicon for a given net doping, used at Ohmic contacts and as the Newton initial guess.

```python
def contact_densities(N: "np.ndarray", V_applied: float = 0.0) -> "np.ndarray":
    """Charge-neutral equilibrium psi, n, p of silicon (300 K) for net doping N.

    Parameters
    ----------
    N : "np.ndarray"
        Net doping N_D - N_A in cm^-3, any shape (positive = n-type).
    V_applied : float
        Applied contact voltage in volts.

    Returns
    -------
    state : "np.ndarray"
        Shape (3,) + N.shape: row 0 is psi in volts, row 1 is n and row 2 is p
        in cm^-3. Use V_T = 0.025852 V and n_ie = 1.087386e10 cm^-3, compute the
        majority density as (|N| + sqrt(N^2 + 4 n_ie^2)) / 2 and the minority
        density as n_ie^2 / majority, and set psi = V_applied + V_T ln(n / n_ie).
        For N = 0 both densities equal n_ie.

    Notes
    -----
    Include every import your implementation needs inside the function body.
    """
    return state  # placeholder
```

### Step 3

03_build_mesh

Goal
----
Build the distorted, non-Delaunay triangular mesh of the unit square (lengths in micrometres) on which the device is solved.

```python
def build_mesh(M: int) -> tuple:
    """Shifted-row triangulation of the unit square [0, 1] x [0, 1] (micrometres).

    Parameters
    ----------
    M : int
        Number of cells per side; must be even and >= 2.

    Returns
    -------
    points : "np.ndarray"
        Shape ((M+1)**2, 2), float. Vertex (i, j), i, j = 0..M, has index
        j*(M+1) + i and coordinates (i*h, j*h) with h = 1/M, except that vertices
        with j odd and 0 < i < M are shifted by +0.4*h in x.
    triangles : "np.ndarray"
        Shape (2*M*M, 3), int. For j = 0..M-1 (outer loop) and i = 0..M-1 (inner
        loop), with a, b, c, d the indices of (i,j), (i+1,j), (i+1,j+1), (i,j+1),
        append (a, b, c) then (a, c, d).

    Notes
    -----
    Raise ValueError if M is odd or below 2.
    Include every import your implementation needs inside the function body.
    """
    return points, triangles  # placeholder
```

### Step 4

04_ddfv_geometry

Goal
----
Build the discrete duality finite volume (DDFV) primal, dual and diamond meshes from a triangulation, with the geometric coefficients of every diamond.

```python
def ddfv_geometry(points: "np.ndarray", triangles: "np.ndarray") -> dict:
    """DDFV mesh data for a 2D triangulation.

    Parameters
    ----------
    points : "np.ndarray"
        Shape (V, 2), vertex coordinates.
    triangles : "np.ndarray"
        Shape (T, 3), int vertex indices.

    Returns
    -------
    geom : dict with these numpy arrays
        "tri_area"     (T,)   triangle areas.
        "tri_centroid" (T, 2) triangle barycentres.
        "bnd_edges"    (Eb, 2) int, boundary edges (edges belonging to one
                       triangle) as (vmin, vmax), sorted lexicographically.
        "bnd_mid"      (Eb, 2) midpoints of those edges.
        "dual_area"    (V,)   dual-cell area of each vertex.
        "diamonds"     (ND, 4) int, one row [K, L, Ks, Ls] per edge, rows in
                       lexicographic order of (vmin, vmax) over all edges.
                       Ks = vmin, Ls = vmax. K is the lowest-index triangle
                       containing the edge. L is the other triangle, or, for a
                       boundary edge, T + m where m is its row in "bnd_edges".
                       Primal node indices run over triangles 0..T-1, then
                       boundary edges T..T+Eb-1.
        "alpha", "beta", "gamma", "diamond_area"  (ND,) each, with
                       alpha = |s|^2/(2|D|), beta = |s||s*| (n_KL . n_KsLs)/(2|D|),
                       gamma = |s*|^2/(2|D|), where s = x_Ls - x_Ks, s* = x_L - x_K,
                       n_KL is the unit normal to s with n_KL . s* > 0, n_KsLs is
                       the unit normal to s* with n_KsLs . s > 0, and
                       |D| = |s* x s| / 2 (the 2D cross product of the diagonals).

    Notes
    -----
    Include every import your implementation needs inside the function body.
    """
    return geom  # placeholder
```

### Step 5

05_ddfv_ha_fluxes

Goal
----
Compute the DDFV-HA electrostatic, electron and hole fluxes across the primal and dual edge of every diamond.

```python
def ddfv_ha_fluxes(geom: dict, u: "np.ndarray", n: "np.ndarray", p: "np.ndarray",
                   lam2: float, Dn: float, Dp: float) -> "np.ndarray":
    """DDFV-HA fluxes on every diamond.

    Parameters
    ----------
    geom : dict
        Output of ddfv_geometry.
    u, n, p : "np.ndarray"
        Node values of shape (T + Eb + V,): primal nodes (triangles, then
        boundary edges) followed by vertices. u is the potential in thermal
        voltages; n and p are scaled carrier densities.
    lam2 : float
        Scaled permittivity multiplying the electrostatic fluxes.
    Dn, Dp : float
        Electron and hole diffusion coefficients.

    Returns
    -------
    fluxes : "np.ndarray"
        Shape (6, ND), rows: electrostatic primal, electrostatic dual, electron
        primal, electron dual, hole primal, hole dual. Primal fluxes point from
        K to L, dual fluxes from Ks to Ls. With a = u_K - u_L, b = u_Ks - u_Ls,
        B the Bernoulli function and (alpha, beta, gamma) from geom:
          s_K = B(a) n_K - B(-a) n_L,  s_Ks = B(b) n_Ks - B(-b) n_Ls,
          t_K = B(-a) p_K - B(a) p_L,  t_Ks = B(-b) p_Ks - B(b) p_Ls,
          rows = [lam2 (alpha a + beta b), lam2 (beta a + gamma b),
                  Dn (alpha s_K + beta s_Ks), Dn (beta s_K + gamma s_Ks),
                  Dp (alpha t_K + beta t_Ks), Dp (beta t_K + gamma t_Ks)].

    Notes
    -----
    Include every import your implementation needs inside the function body.
    """
    return fluxes  # placeholder
```

### Step 6

06_assemble_system

Goal
----
Assemble the residual and analytic sparse Jacobian of the coupled DDFV-HA Poisson and carrier-continuity equations, with Dirichlet nodes held fixed.

```python
def assemble_system(geom: dict, u: "np.ndarray", n: "np.ndarray", p: "np.ndarray",
                    Nd: "np.ndarray", dirichlet: "np.ndarray",
                    lam2: float, Dn: float, Dp: float) -> tuple:
    """Residual and Jacobian of the scaled DDFV-HA drift-diffusion system.

    Parameters
    ----------
    geom : dict
        Output of ddfv_geometry.
    u, n, p, Nd : "np.ndarray"
        Node values, shape (Nn,) with Nn = T + Eb + V (triangles, boundary
        edges, vertices): potential in thermal voltages, scaled carrier densities
        and scaled net doping.
    dirichlet : "np.ndarray"
        Bool, shape (Nn,), True for nodes whose u, n, p are held fixed.
    lam2, Dn, Dp : float
        As in ddfv_ha_fluxes.

    Returns
    -------
    R : "np.ndarray"
        Shape (3*Nn,), unknowns interleaved as index 3*node + (0: u, 1: n, 2: p).
        For each node: R_u = sum(signed electrostatic fluxes) - area*(p - n + Nd),
        R_n = sum(signed electron fluxes), R_p = sum(signed hole fluxes), where a
        diamond's primal fluxes count +1 at K and -1 at L, its dual fluxes +1 at
        Ks and -1 at Ls, and area is tri_area for triangles, 0 for boundary edges
        and dual_area for vertices. Entries of Dirichlet nodes are 0.
    J : scipy.sparse matrix
        Shape (3*Nn, 3*Nn), the exact Jacobian dR/d(u, n, p), with the rows of
        Dirichlet nodes replaced by identity rows.

    Notes
    -----
    Include every import your implementation needs inside the function body.
    """
    return R, J  # placeholder
```

### Step 7

07_solve_drift_diffusion

Goal
----
Solve the stationary drift-diffusion problem for the symmetric silicon PN junction on the distorted mesh with Newton's method and voltage ramping.

```python
def solve_drift_diffusion(N0: float, Va: float, M: int) -> "np.ndarray":
    """Stationary DDFV-HA solution of the symmetric abrupt silicon PN junction.

    Parameters
    ----------
    N0 : float
        Doping magnitude in cm^-3: N = +N0 for y < 0.5, -N0 for y > 0.5, 0 at
        y = 0.5 (y in micrometres, evaluated at each node's position:
        barycentre, boundary-edge midpoint or vertex).
    Va : float
        Voltage on the top contact (y = 1) in volts; the bottom contact (y = 0)
        is at 0 V. Va >= 0.
    M : int
        Mesh parameter for build_mesh.

    Returns
    -------
    state : "np.ndarray"
        Shape (3, Nn): psi (V), n (cm^-3), p (cm^-3) on all nodes in the node
        order of ddfv_geometry (triangles, boundary edges, vertices).

    Notes
    -----
    Silicon at 300 K: q = 1.602192e-19 C, eps = 1.035941e-12 C/(V cm),
    V_T = 0.025852 V, mu_n = 1417.0 and mu_p = 470.5 cm^2/(V s), D = V_T mu.
    Scaling: u = psi / V_T, n and p divided by 1e16 cm^-3, lengths in
    micrometres, lam2 = eps V_T / (q 1e16 (1e-4)^2).
    Dirichlet nodes: boundary edges with both ends on y = 0 or both on y = 1,
    and vertices with y = 0 or y = 1, set by contact_densities (plus Va on top).
    Newton: start from contact_densities of the local doping at 0 V; after each
    update clip scaled n, p below at 1e-36 and reset the Dirichlet nodes to
    their exact contact values; stop when, over the non-Dirichlet nodes,
    max|du| < 1e-10 and the largest relative update of n and p is < 1e-6
    (at most 60 iterations).
    Ramp: solve at 0 V, then step the voltage starting at 0.05 V, doubling the
    step after each success and halving it after a failure, until Va.
    Include every import your implementation needs inside the function body.
    """
    return state  # placeholder
```

### Step 8

08_terminal_current

Goal
----
Compute the terminal current density through the top Ohmic contact from the primal DDFV-HA electron and hole fluxes.

```python
def terminal_current(psi: "np.ndarray", n: "np.ndarray", p: "np.ndarray", M: int) -> float:
    """Current density through the top contact (y = 1) in A/cm^2.

    Parameters
    ----------
    psi, n, p : "np.ndarray"
        Node values in V, cm^-3, cm^-3, shape (Nn,), in the node order of
        ddfv_geometry(*build_mesh(M)), as returned by solve_drift_diffusion.
    M : int
        Mesh parameter.

    Returns
    -------
    J : float
        q * 1e16 * (sum F - sum G) / 1e-4, where F and G are the primal electron
        and hole fluxes of ddfv_ha_fluxes (u = psi/V_T, n and p divided by 1e16,
        lam2 as in solve_drift_diffusion, Dn = V_T*1417.0, Dp = V_T*470.5) over
        the boundary diamonds whose edge lies on y = 1. Positive for
        conventional current entering the device through the top contact.

    Notes
    -----
    Include every import your implementation needs inside the function body.
    """
    return J  # placeholder
```

### Step 9

09_junction_electrostatics

Goal
----
Extract the depletion width and the peak electric field of the junction from the equilibrium vertex solution.

```python
def junction_electrostatics(psi_v: "np.ndarray", n_v: "np.ndarray", p_v: "np.ndarray",
                            N0: float, M: int) -> "np.ndarray":
    """Depletion width and peak field from the vertex solution.

    Parameters
    ----------
    psi_v, n_v, p_v : "np.ndarray"
        Vertex values (V, cm^-3, cm^-3), shape ((M+1)**2,), vertex (i, j) at
        index j*(M+1) + i as in build_mesh.
    N0 : float
        Doping magnitude in cm^-3.
    M : int
        Mesh parameter; row spacing h = 1/M micrometres.

    Returns
    -------
    result : "np.ndarray"
        Shape (2,): [W in micrometres, E_max in V/cm]. With row averages
        psi_j, n_j, p_j and y_j = j h: y_n is where n_j first drops below N0/2
        scanning up from j = 0, and y_p is where p_j first exceeds N0/2 scanning
        up from j = M/2, each by linear interpolation between the bracketing rows;
        W = y_p - y_n. E_max = max_j |psi_{j+1} - psi_j| / h_cm + q N0 h_cm/(2 eps)
        with h_cm = 1e-4/M, q = 1.602192e-19 C, eps = 1.035941e-12 C/(V cm).

    Notes
    -----
    Include every import your implementation needs inside the function body.
    """
    return result  # placeholder
```

### Step 10

10_textbook_predictions

Goal
----
Compute the depletion-approximation and short-diode-law predictions for the symmetric silicon junction from its material parameters.

```python
def textbook_predictions(N0: float, Va: float) -> "np.ndarray":
    """Depletion-approximation and short-diode predictions for the symmetric junction.

    Parameters
    ----------
    N0 : float
        Doping magnitude on each side, cm^-3.
    Va : float
        Forward bias in volts for the current (0 <= Va < V_bi).

    Returns
    -------
    result : "np.ndarray"
        Shape (4,): [V_bi (V), W_DA (micrometres, at 0 V), E_DA (V/cm, at 0 V),
        J_SD (A/cm^2, at Va)], with V_bi = V_T ln(N0^2/n_ie^2),
        W_DA = sqrt(4 eps V_bi/(q N0)), E_DA = q N0 W_DA/(2 eps),
        J_SD = q (n_ie^2/N0) (exp(Va/V_T) - 1) (D_n + D_p) / W', and
        W' = 0.5e-4 cm - sqrt(4 eps (V_bi - Va)/(q N0))/2. Constants:
        q = 1.602192e-19 C, eps = 1.035941e-12 C/(V cm), V_T = 0.025852 V,
        n_ie = 1.087386e10 cm^-3, D_n = V_T*1417.0, D_p = V_T*470.5 cm^2/s.

    Notes
    -----
    Include every import your implementation needs inside the function body.
    """
    return result  # placeholder
```

### Step 11

11_simulated_forward_current

Goal
----
Final orchestrator: run the doping study of the silicon junction and return the simulated forward current density at the lowest doping level.

```python
def simulated_forward_current(M: int = 64, Va: float = 0.5) -> float:
    """Simulated forward current density of the junction at N0 = 1e16 cm^-3.

    Builds, for each N0 in (1e16, 3e16, 1e17) cm^-3, the row
    [W_sim (micrometres), E_sim (V/cm), J_sim (A/cm^2), W_DA (micrometres),
    E_DA (V/cm), J_SD (A/cm^2), 100*(W_sim/W_DA - 1), 100*(E_sim/E_DA - 1),
    100*(J_sim/J_SD - 1)], where W_sim and E_sim come from the 0 V solution,
    J_sim from the solution at Va, and the rest from the textbook predictions.

    Parameters
    ----------
    M : int
        Mesh parameter of the distorted triangulation (64 for the reported result).
    Va : float
        Forward bias in volts applied to the top contact.

    Returns
    -------
    current : float
        J_sim for N0 = 1e16 cm^-3 in A/cm^2, positive for conventional current
        entering the device through the top contact.
    """
    return current  # placeholder
```
