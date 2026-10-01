# Physics-Particle_Physics-21

## Background

Natural units are used. With \(X=mx\), \(\Phi=\sqrt{\lambda}\phi/m\) and \(K=\kappa/(m\sqrt{\lambda})\), the continuum modified action is \(\lambda^{-1}\int d^4X[\tfrac12(\partial\Phi)^2+\tfrac12\Phi^2-\Phi^4/24+K\Phi^3]\). The benchmark below is a constructed finite radial realization of the paper's constraint geometry; its Gaussian weights are defined in this finite space. The analytic massless continuum family \(\Phi_\rho(r)=4\sqrt3\rho/(\rho^2+r^2)\) has action \(16\pi^2/\lambda\), independent of \(\rho\).

For \(j=0,\ldots,N\), set \(r_j=R(j/N)^p\), \(\Phi_N=0\), \(a_0=0\), and \(a_{j+1}=(r_j+r_{j+1})/2\). The \(N\) variable fields have weights \(w_j=(\pi^2/2)(a_{j+1}^4-a_j^4)\) and edge coefficients \(c_j=2\pi^2 a_{j+1}^3/(r_{j+1}-r_j)\). Define
\[
s(\Phi)=\frac12\sum_{j=0}^{N-1}c_j(\Phi_{j+1}-\Phi_j)^2+\sum_{j=0}^{N-1}w_j\left(\frac{\Phi_j^2}{2}-\frac{\Phi_j^4}{24}\right),\qquad
\xi(\Phi)=\sum_{j=0}^{N-1}w_j\Phi_j^3.
\]
The kinetic expression fixes the regular origin boundary and the outer Dirichlet boundary. The physical action is \(s/\lambda\), and the ambient Euclidean field coordinates are \(y_j=\sqrt{w_j}\Phi_j\). The family consists of the nonzero, positive, decreasing stationary profiles of \(s+K\xi\) connected continuously across the stated \(K\) interval. Its single interior maximum of \(\xi(K)\), lying in \(K\in[-0.32,-0.21]\), defines \((K_c,\xi_c)\). The grid defines the observable exactly.

For a profile at fixed \(\xi\), let \(C\) be the intrinsic Hessian of \(s\) restricted to the nonlinear hypersurface \(\xi(y)=\text{constant}\), in an orthonormal tangent frame. Let \(D=\det C\), and label a stationary profile by the number \(b\) of negative eigenvalues of \(C\). For every constraint in the integration interval there are two profiles in the declared family, with labels 0 and 1. For the requested response diagnostic, define \(H=\nabla_y^2(s+K\xi)\), \(z=\nabla_y\xi\), and \(\nu=z^TH^{-1}z/(z^Tz)\); these \(H\) matrices are nonsingular. All eigenvalue counts and determinants concern radial fluctuations in this \(N\)-dimensional model.

The intrinsic constrained Gaussian weight is
\[
G_b(\xi)=\exp[-s_b(\xi)/\lambda]\,|\det(C_b(\xi)/\lambda)|^{-1/2},\qquad
Z_b=\int_{0.84\xi_c}^{0.97\xi_c}G_b(\xi)\,d\xi.
\]
\(\langle q\rangle_1\) is the mean of \(q=\xi/\xi_c\) with this branch-1 weight and the same integration measure. If \((t_i,u_i)\) are the 24 Gauss–Legendre nodes and weights on \([-1,1]\), use \(q_i=0.905+0.065t_i\) and weights \(0.065\xi_c u_i\). Intermediate quantities retain full floating-point precision. In the code interfaces, noncanonical geometry and integration parameters use these same definitions; valid branch inputs have the stated fold and both positive profiles, and nonsingular Hessians at quadrature nodes.

## Problem

An effective scalar sector with a metastable origin is represented by the finite radial action and cubic constraint in the background. Using the attached paper's nonperturbative constrained-instanton formulation, compute the dimensionless Gaussian suppression \(B=-\log(Z_1/Z_0)\) at \(\lambda=0.7\), where the two constrained stationary branches are labelled by their intrinsic negative-mode counts \(b=0,1\). Use \(N=160\), \(R=16\), \(p=2\), and integrate over \(\xi/\xi_c\in[0.84,0.97]\) with the specified 24-point Gauss–Legendre rule; \(\xi_c\) is the maximum constraint attained by the positive decreasing stationary family with \(K\in[-0.65,-0.15]\).

In the reasoning, give the eight-number certificate \((K_c,\xi_c,K_0(q_m),K_1(q_m),\Delta s(q_m),\Delta\log|D|(q_m),\langle q\rangle_1,\nu(-0.22))\), where \(q_m=0.905\), each difference is branch 1 minus branch 0, and \(D\) uses the Hessian of \(s\) on the constraint surface. Briefly justify the fluctuation curvature, the determinant relation supported by the paper, the branch labels, the radial field metric, and the integration measure. State the scope of the paper's determinant derivation and the status of its extension to function spaces.

Also state the paper's conjugate relation \(K=-ds/d\xi\) on this family, use the fixed sign of \(K\) to say why the two branches meet at a cusp rather than a smooth peak at \((K_c,\xi_c)\), give the susceptibility \(d\xi/dK\) that locates \(K_c\), and identify the small-\(|K|\) and large-\(|K|\) branches, with the continuum action the small-\(|K|\) branch approaches as \(K\to0\) and the number of constrained negative modes on each.

Report \(B\) to five decimal places and the certificate to six significant figures; absolute error \(5\times10^{-5}\) in \(B\) and \(5\times10^{-5}\max(1,|x|)\) in each certificate entry are accepted.

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

radial_geometry

Goal
----
Build the radial field metric and the fixed-action stiffness data.

```python
def radial_geometry(n: int, radius: float, power: float) -> "np.ndarray":
    """Build the radial field metric and the fixed-action stiffness data.

    n : int, n >= 2
        Number of variable radial fields, excluding the fixed outer endpoint.
    radius : float > 0
        Dimensionless outer radius.
    power : float >= 1
        Power in r_j=radius*(j/n)**power.
    Returns
    -------
    ndarray, shape (n,4)
        Columns [r_j,w_j,c_j,L_jj] in outward order. L_jj=c_j+c_(j-1), with c_(-1)=0. The definitions of a,w,c are those in the background.

    Raises
    ------
    ValueError
        If numerical input domains or array shapes are invalid.
    """
    return result
```

### Step 2

action_jet

Goal
----
Evaluate the constrained action and its derivatives in the physical radial field metric.

```python
def action_jet(phi: "np.ndarray", kappa: float, geometry: "np.ndarray") -> "np.ndarray":
    """Evaluate the constrained action and its derivatives in the physical radial field metric.

    phi : ndarray, shape (N,)
        Dimensionless variable fields; the endpoint phi_N is fixed to zero.
    kappa : float
        Dimensionless cubic Lagrange multiplier K, of either sign.
    geometry : ndarray, shape (N,4)
        Columns are radius r, positive cell weight w, outward edge c, and stiffness diagonal Ljj; rows run from origin outward. All entries are dimensionless.
    Returns
    -------
    ndarray, shape (3+2*N,)
        Entries [s,xi,s+K*xi], followed by N components of grad_y(s+K*xi), then N diagonal entries of H=Hess_y(s+K*xi), with y=sqrt(w)*phi. All are dimensionless. H_(j,j+1)=-c_j/sqrt(w_j*w_(j+1)).

    Raises
    ------
    ValueError
        If numerical input domains or array shapes are invalid.
    """
    return result
```

### Step 3

stationary_profile

Goal
----
Recover the nonzero positive decreasing stationary family of the modified action.

```python
def stationary_profile(kappa: float, geometry: "np.ndarray") -> "np.ndarray":
    """Recover the nonzero positive decreasing stationary family of the modified action.

    kappa : float in [-0.65,-0.15]
        Dimensionless cubic multiplier K.
    geometry : ndarray, shape (N,4)
        Columns are radius r, positive cell weight w, outward edge c, and stiffness diagonal Ljj; rows run from origin outward. All entries are dimensionless.
    Returns
    -------
    ndarray, shape (N,)
        The positive decreasing profile stationary for s+K*xi, in outward order, on the connected family described in the background. Geometry is restricted to cases where this profile exists. The zero solution is a distinct stationary point. A residual max(abs(grad_y)) below 1e-6 is sufficient.

    Raises
    ------
    ValueError
        If numerical input domains or array shapes are invalid.
    """
    return result
```

### Step 4

normal_response

Goal
----
Compute the auxiliary normal response and modified-Hessian spectral data.

```python
def normal_response(jet: "np.ndarray", phi: "np.ndarray", geometry: "np.ndarray") -> "np.ndarray":
    """Compute the auxiliary normal response and modified-Hessian spectral data.

    jet : ndarray, shape (3+2*N,)
        Action-jet layout; only its last N entries, the diagonal of H, are used.
    phi : ndarray, shape (N,)
        Dimensionless fields; at least one entry is nonzero.
    geometry : ndarray, shape (N,4)
        Columns are radius r, positive cell weight w, outward edge c, and stiffness diagonal Ljj; rows run from origin outward. All entries are dimensionless.
    Returns
    -------
    ndarray, length 5
        [negative_count(H), nu, dxi_dK, log(abs(det(H))), dot(z,z)], where z=grad_y(xi), H*psi=z, nu=dot(z,psi)/dot(z,z), and dxi_dK is the stationary-family susceptibility. H is nonsingular with eigenvalues separated from zero by at least 1e-8. Synthetic tridiagonal fluctuation jets in the tests obey the same algebraic contract.

    Raises
    ------
    ValueError
        If numerical input domains or array shapes are invalid.
    """
    return result
```

### Step 5

constraint_fold

Goal
----
Locate the constraint maximum and its original action on the nonzero family.

```python
def constraint_fold(geometry: "np.ndarray") -> "np.ndarray":
    """Locate the constraint maximum and its original action on the nonzero family.

    geometry : ndarray, shape (N,4)
        Columns are radius r, positive cell weight w, outward edge c, and stiffness diagonal Ljj; rows run from origin outward. All entries are dimensionless.
    Returns
    -------
    ndarray, length 3
        [K_c,xi_c,s_c] at the unique interior maximum of xi(K), with K_c in [-0.32,-0.21]. All values are dimensionless. This is the connected nonzero family returned by stationary_profile.

    Raises
    ------
    ValueError
        If numerical input domains or array shapes are invalid.
    """
    return result
```

### Step 6

matched_branches

Goal
----
Invert the constraint on both branches and label the profiles by constrained inertia.

```python
def matched_branches(fraction: float, fold: "np.ndarray", geometry: "np.ndarray") -> "np.ndarray":
    """Invert the constraint on both branches and label the profiles by constrained inertia.

    fraction : float, 0 < fraction < 1
        Desired xi/xi_c; both roots must lie in [-0.65,-0.15].
    fold : ndarray, shape (3,)
        [K_c,xi_c,s_c] from constraint_fold.
    geometry : ndarray, shape (N,4)
        Columns are radius r, positive cell weight w, outward edge c, and stiffness diagonal Ljj; rows run from origin outward. All entries are dimensionless.
    Returns
    -------
    ndarray, shape (2,N+1)
        Row b contains [K_b,phi_b[0],...,phi_b[N-1]], for constrained negative-mode count b=0 then b=1. Both profiles have xi=fraction*xi_c. All entries are dimensionless.

    Raises
    ------
    ValueError
        If numerical input domains or array shapes are invalid.
    """
    return result
```

### Step 7

branch_logweights

Goal
----
Evaluate the intrinsic radial Gaussian weight of each stationary branch.

```python
def branch_logweights(branches: "np.ndarray", coupling: float, geometry: "np.ndarray") -> "np.ndarray":
    """Evaluate the intrinsic radial Gaussian weight of each stationary branch.

    branches : ndarray, shape (2,N+1)
        Two rows [K,phi[0],...,phi[N-1]], each stationary for its multiplier.
    coupling : float > 0
        Dimensionless lambda in physical action s/lambda.
    geometry : ndarray, shape (N,4)
        Columns are radius r, positive cell weight w, outward edge c, and stiffness diagonal Ljj; rows run from origin outward. All entries are dimensionless.
    Returns
    -------
    ndarray, shape (2,4)
        Columns [s,nu,log(abs(D)),log(G)] for each input row, preserving input order. D is the intrinsic determinant of s on the cubic-constraint surface in the y metric. G=exp(-s/lambda)/sqrt(abs(det(C/lambda))). Both D values are nonzero.

    Raises
    ------
    ValueError
        If numerical input domains or array shapes are invalid.
    """
    return result
```

### Step 8

integrated_suppression

Goal
----
Aggregate the two constrained Gaussian measures and a branch-one mean.

```python
def integrated_suppression(fractions: "np.ndarray", weights: "np.ndarray", logweights: "np.ndarray", xi_critical: float) -> "np.ndarray":
    """Aggregate the two constrained Gaussian measures and a branch-one mean.

    fractions : ndarray, shape (M,)
        Dimensionless quadrature coordinates q=xi/xi_c.
    weights : ndarray, shape (M,)
        Positive quadrature weights for dq, in the same order.
    logweights : ndarray, shape (M,2)
        Logarithms of G, with columns constrained branches b=0 and b=1.
    xi_critical : float > 0
        Dimensionless xi_c multiplying dq to give dxi.
    Returns
    -------
    ndarray, length 4
        [B,log(Z_0),log(Z_1),mean_q_1], with B=log(Z_0)-log(Z_1). The mean uses the normalized branch-1 Gaussian measure. Inputs are finite; logweights may be too negative for direct exponentiation.

    Raises
    ------
    ValueError
        If numerical input domains or array shapes are invalid.
    """
    return result
```

### Step 9

radial_competition

Goal
----
Compose the full constrained-instanton branch competition and compact certificate.

```python
def radial_competition(coupling: float = 0.7, lower: float = 0.84, upper: float = 0.97, n: int = 160, radius: float = 16.0, power: float = 2.0, order: int = 24) -> "np.ndarray":
    """Compose the full constrained-instanton branch competition and compact certificate.

    coupling : float > 0, default 0.7
        Dimensionless lambda.
    lower, upper : floats, default 0.84, 0.97
        Integration limits in q=xi/xi_c, with 0<lower<upper<1 and both branches present.
    n : int, default 160
        Number of variable radial fields.
    radius, power : floats, default 16 and 2
        The geometry parameters defined by radial_geometry.
    order : int >= 2, default 24
        Gauss-Legendre quadrature order.
    Returns
    -------
    ndarray, length 9
        [B,K_c,xi_c,K_0(q_m),K_1(q_m),delta_s(q_m),delta_logabsD(q_m),mean_q_1,nu_anchor], with q_m=(lower+upper)/2, deltas branch 1 minus branch 0, and nu_anchor at K=-0.22. All entries are dimensionless. The public implementation must call and combine every preceding public function, including the anchor action and normal response. Geometry is restricted to the family domain in the background.

    Raises
    ------
    ValueError
        If numerical input domains or array shapes are invalid.
    """
    return result
```
