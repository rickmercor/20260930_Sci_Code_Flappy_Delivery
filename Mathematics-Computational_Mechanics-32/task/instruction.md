# Mathematics-Computational_Mechanics-32

## Background

Nonlocal wave equations replace the second spatial derivative of the classical wave
equation by an integral operator: the acceleration at a point is driven by the weighted
differences between the displacement there and the displacement at every point within a
horizon of radius delta, the weight being a kernel that decays with the distance as an
inverse power. The scalar peridynamic model of wave propagation is of this form. When the
kernel is scaled so that its second moment over the horizon is one, the operator converges
to minus the second derivative as the horizon shrinks, and the nonlocal problem converges to
the local wave equation; for a finite horizon the two differ, most visibly in the
dispersion relation, where the nonlocal frequency of a plane wave falls below its local
value the more the wavelength approaches the horizon.

Discontinuous Galerkin methods approximate the solution by polynomials of a fixed degree on
each cell with no continuity across cell interfaces, and are attractive for nonlocal
problems because the exact solutions may themselves be discontinuous. Their construction
for a nonlocal operator is not a copy of the local one, however: there is no derivative
to integrate by parts, hence no numerical flux to choose. The method studied here
reformulates the equation with an auxiliary field that depends on the interaction distance
as well as on position, discretizes that field in the same broken polynomial space, and
recovers a local discontinuous Galerkin method with a definite flux pairing in the
vanishing-horizon limit. The spatial discretization is coupled with a Crank-Nicolson time
integration that is implicit, unconditionally stable and conserves a discrete energy
exactly.

The task solves a periodic problem with a non-integrable kernel, a two-mode initial
profile, piecewise-linear polynomials on a coarse mesh, and a moderate time step, so that
the nonlocal dispersion, the spatial discretization and the time integration all leave
their mark on the reported L2 error. The exact solution is available in closed form once
the eigenvalues of the periodic nonlocal operator on the two plane waves present are
computed, so no manufactured source term is needed. Standard numpy linear algebra
suffices; dense matrices of size 16 are involved.

## Problem

The one-dimensional nonlocal wave equation u_tt + L_delta u = 0 replaces the second
derivative by an integral operator over a horizon of radius delta,
L_delta u(x) = -2 int_{x-delta}^{x+delta} (u(y) - u(x)) gamma_delta(x - y) dy, with the
radial kernel gamma_delta(s) = (3 - alpha) / (2 delta^(3 - alpha)) |s|^(-alpha), whose
second moment over the horizon is one so that the local wave equation is recovered as the
horizon vanishes. A recent discontinuous Galerkin method for this equation introduces an
auxiliary variable indexed by the interaction distance, discretizes the reformulated system
in a broken polynomial space, and integrates in time with a Crank-Nicolson scheme that
conserves a discrete energy.

Adopt that method exactly as the source prescribes it: how the auxiliary variable is defined
from the solution (which difference quotient, at which distance), how it is represented for
every interaction distance, how the two operators of the reformulated system are built and
how the coupling between cells arises from them, and how the second time derivative and the
operator's argument are discretized in the fully discrete scheme. Evaluate every integral
over a cell of a shifted piecewise polynomial exactly, and resolve the integrals over the
interaction distance to at least eight significant digits, taking the singular kernel at
s = 0 and the breakpoints of the integrand into account.

Work on the periodic unit interval (0, 1) with alpha = 3/2 and delta = 1/4, initial data
u(x, 0) = sin(2 pi x) + cos(4 pi x)/2 and u_t(x, 0) = 0, piecewise-linear polynomials on
N = 8 uniform cells, the L2 projection of the initial data as the discrete initial state,
time step h_t = 0.01 up to T = 2, and, since the scheme is a two-step recursion, start it
with the fictitious value at t = -h_t equal to the value at t = +h_t (the discrete
counterpart of the zero initial velocity), so that the scheme at n = 0 determines the first
step. Derive the exact solution of this nonlocal problem for the given initial data (the
periodic nonlocal operator is diagonal on plane waves) and compute the L2 error of the
discrete solution at T = 2 the way the source reports errors: on every cell a 4-point
Gauss-Lobatto rule applied to the squared pointwise difference between the exact solution
and the discrete solution, summed over cells, square root taken.

Report that L2 error at T = 2. In the reasoning, also give the two exact nonlocal angular
frequencies of the modes present, the two smallest distinct non-zero eigenvalues and the
largest eigenvalue of the negative of the discrete spatial operator (the matrix that, applied to the
coefficient vector of the discrete solution, gives its second time derivative), the value of
the conserved fully discrete energy of the source's scheme, and the same L2 error evaluated
at t = 1; state the conventions you adopted and justify each from the source, and state what
the source proves about the fully discrete energy and which local DG method its scheme
becomes when the horizon vanishes.

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
Output Format Requirements:
Emit <final_answer> immediately, then <reasoning>. Do not write a long derivation before the tags.
You must emit exactly one finite decimal inside <final_answer>...</final_answer>, even if the value is approximate or you are unsure.
Rules:
- The tags are required. Do not omit them or leave them empty.
- The tagged value must be a finite decimal (examples: 0.4847, 0.22, 1.05). Not NaN, not Inf, not a fraction string, not a vector, not prose.
- Put only that one number between the tags. No units, no words, no extra lines.
Keep <reasoning> short (a few hundred words). Show only the few scalars that determine the final number.
Do not paste the input matrices, full coefficient vectors, per-iteration paths, or per-fold candidate tables.
```

## Your task

Implement **all 8 functions** below into a single file at `/app/solution.py`. Requirements:

- Use the EXACT function name and signature shown for each step.
- Define every function at module scope (importable).
- Do NOT include tests, example usage, prints, or a `__main__` block.
- Available libraries: numpy, scipy, sympy, pandas, scikit-learn, networkx.

Start your file with these imports:

```python
import numpy as np
```

### Step 1

nonlocal_symbol

Goal
----
Return the eigenvalue of the nonlocal operator L_delta of the source, Eq (2.1), acting on the plane wave exp(i xi x), for the radial kernel family of Eq (5.1) with exponent alpha and horizon delta. The kernel is non-integrable for alpha >= 1, so evaluate the defining integral with a quadrature that resolves the algebraic behaviour of the integrand at s = 0 to at least ten significant digits for every 0 < alpha < 3. Raise ValueError if delta is not positive or alpha is not in (0, 3).

```python
import numpy as np


def nonlocal_symbol(xi, alpha, delta):
    """float, the eigenvalue of L_delta on exp(i xi x)."""
    return 0.0
```

### Step 2

l2_projection

Goal
----
Discretisation conventions used throughout: the domain is (0, 1) with periodic continuation, N uniform cells I_j = ((j-1)h, jh), h = 1/N, j = 1..N, local coordinate xi = 2 (x - x_j)/h in [-1, 1] with x_j the cell centre, modal basis P_0, ..., P_k (Legendre polynomials, not normalised) on every cell, and coefficient vectors stored cell-major: entry j*(k+1) + m is the coefficient of P_m on cell j+1 (0-based j). Return the coefficient vector of the L2 projection of the vectorised function fun onto the piecewise-polynomial space of degree k, Eq (3.1) of the source, i.e. on every cell the polynomial whose inner products with P_0..P_k match those of fun; integrate accurately enough that smooth functions are projected to round-off. Raise ValueError if N is not a positive integer, k is not a non-negative integer, or fun does not return an array of the same shape as its input.

```python
import numpy as np


def l2_projection(fun, N, k):
    """ndarray of float64 with shape (N*(k+1),): the cell-major modal coefficients."""
    return np.zeros(N * (k + 1), dtype=np.float64)
```

### Step 3

shift_projection_matrix

Goal
----
Discretisation conventions used throughout: the domain is (0, 1) with periodic continuation, N uniform cells I_j = ((j-1)h, jh), h = 1/N, j = 1..N, local coordinate xi = 2 (x - x_j)/h in [-1, 1] with x_j the cell centre, modal basis P_0, ..., P_k (Legendre polynomials, not normalised) on every cell, and coefficient vectors stored cell-major: entry j*(k+1) + m is the coefficient of P_m on cell j+1 (0-based j). Return the dense matrix that maps the coefficient vector of a function u_h in the degree-k space to the coefficient vector of the L2 projection onto the same space of the shifted function x -> u_h(x + s), s > 0, with u_h continued periodically, i.e. the building block of the operators of Eq (2.5) of the source. The shifted function is itself piecewise polynomial with its own breakpoints, and the projection integrals must be exact, not approximated by a fixed quadrature on the cell. Raise ValueError if N is not a positive integer, k is not a non-negative integer, or s is not positive.

```python
import numpy as np


def shift_projection_matrix(N, k, s):
    """ndarray of float64 with shape (N*(k+1), N*(k+1))."""
    return np.zeros((N * (k + 1), N * (k + 1)), dtype=np.float64)
```

### Step 4

nonlocal_dg_operator

Goal
----
Discretisation conventions used throughout: the domain is (0, 1) with periodic continuation, N uniform cells I_j = ((j-1)h, jh), h = 1/N, j = 1..N, local coordinate xi = 2 (x - x_j)/h in [-1, 1] with x_j the cell centre, modal basis P_0, ..., P_k (Legendre polynomials, not normalised) on every cell, and coefficient vectors stored cell-major: entry j*(k+1) + m is the coefficient of P_m on cell j+1 (0-based j). Return the dense matrix B of the semi-discrete DG scheme of the source, Eq (2.6), for the reformulated system (2.4) with the operators of Eq (2.5) and the kernel family of Eq (5.1) with exponent alpha and horizon delta, written as d^2 u/dt^2 = B u for the coefficient vector u of the degree-k solution (the mass matrix already inverted). The auxiliary variable of (2.4) must be represented in the same degree-k space for every interaction distance s; the shifted-projection matrices it needs are supplied by the callable shift_matrix(N, k, s) (the previous step), which must be used for every distance. The integral over s in (0, delta] must be converged to at least eight significant digits, accounting for the singular kernel and the breakpoints of the integrand at multiples of h. Raise ValueError if N is not a positive integer, k is not a non-negative integer, delta is not positive, alpha is not in (0, 3), or shift_matrix is not callable.

```python
import numpy as np


def nonlocal_dg_operator(N, k, alpha, delta, shift_matrix):
    """ndarray of float64 with shape (N*(k+1), N*(k+1)): the matrix B with d^2 u/dt^2 = B u."""
    return np.zeros((N * (k + 1), N * (k + 1)), dtype=np.float64)
```

### Step 5

discrete_energy

Goal
----
Discretisation conventions used throughout: the domain is (0, 1) with periodic continuation, N uniform cells I_j = ((j-1)h, jh), h = 1/N, j = 1..N, local coordinate xi = 2 (x - x_j)/h in [-1, 1] with x_j the cell centre, modal basis P_0, ..., P_k (Legendre polynomials, not normalised) on every cell, and coefficient vectors stored cell-major: entry j*(k+1) + m is the coefficient of P_m on cell j+1 (0-based j). Given two consecutive coefficient vectors u_new = u^{n+1} and u_old = u^n of the fully discrete scheme with time step ht, and the operator matrix B of the previous step (d^2 u/dt^2 = B u), return the fully discrete energy of Theorem 4.1 of the source: the squared L2 norm of the difference quotient (u^{n+1} - u^n)/ht plus the nonlocal energy of the two states, each expressed through B and the mass matrix. Raise ValueError if ht is not positive, if N or k is invalid, or if the vectors and B do not match N*(k+1).

```python
import numpy as np


def discrete_energy(u_new, u_old, ht, N, k, B):
    """float, the fully discrete energy of the pair (u^{n+1}, u^n)."""
    return 0.0
```

### Step 6

crank_nicolson_march

Goal
----
Discretisation conventions used throughout: the domain is (0, 1) with periodic continuation, N uniform cells I_j = ((j-1)h, jh), h = 1/N, j = 1..N, local coordinate xi = 2 (x - x_j)/h in [-1, 1] with x_j the cell centre, modal basis P_0, ..., P_k (Legendre polynomials, not normalised) on every cell, and coefficient vectors stored cell-major: entry j*(k+1) + m is the coefficient of P_m on cell j+1 (0-based j). Advance the semi-discrete system d^2 u/dt^2 = B u from the coefficient vector u0 at t = 0 with zero initial velocity over nsteps uniform steps of size ht using the two-level Crank-Nicolson scheme of Eq (4.1) of the source (second-order central difference in time, the operator applied to the source's time average of the auxiliary states). Start the two-step recursion as the task statement prescribes for zero initial velocity: the fictitious value at t = -ht equals the value at t = +ht, and the scheme at n = 0 then determines u^1. Return a (2, N*(k+1)) array whose first row is u at t = nsteps*ht and whose second row is u at t = (nsteps-1)*ht. Raise ValueError if ht is not positive, nsteps is not a positive integer, N or k is invalid, or the sizes do not match.

```python
import numpy as np


def crank_nicolson_march(u0, ht, nsteps, N, k, B):
    """ndarray of float64 with shape (2, N*(k+1)): rows u^{nsteps} and u^{nsteps-1}."""
    return np.zeros((2, N * (k + 1)), dtype=np.float64)
```

### Step 7

gauss_lobatto_l2_error

Goal
----
Discretisation conventions used throughout: the domain is (0, 1) with periodic continuation, N uniform cells I_j = ((j-1)h, jh), h = 1/N, j = 1..N, local coordinate xi = 2 (x - x_j)/h in [-1, 1] with x_j the cell centre, modal basis P_0, ..., P_k (Legendre polynomials, not normalised) on every cell, and coefficient vectors stored cell-major: entry j*(k+1) + m is the coefficient of P_m on cell j+1 (0-based j). Return the L2 error between the vectorised reference function fun and the degree-k DG function with coefficient vector coef, evaluated exactly as the source does in Section 5: on every cell a (k+3)-point Gauss-Lobatto rule (end points included) applied to the squared pointwise difference, summed over cells, square root taken. Raise ValueError if N or k is invalid, coef does not have N*(k+1) entries, or fun does not return an array of the input's shape.

```python
import numpy as np


def gauss_lobatto_l2_error(coef, fun, N, k):
    """float, the Gauss-Lobatto L2 error."""
    return 0.0
```

### Step 8

nonlocal_wave_dg_l2_error

Goal
----
Discretisation conventions used throughout: the domain is (0, 1) with periodic continuation, N uniform cells I_j = ((j-1)h, jh), h = 1/N, j = 1..N, local coordinate xi = 2 (x - x_j)/h in [-1, 1] with x_j the cell centre, modal basis P_0, ..., P_k (Legendre polynomials, not normalised) on every cell, and coefficient vectors stored cell-major: entry j*(k+1) + m is the coefficient of P_m on cell j+1 (0-based j). Orchestrator. Solve the nonlocal wave equation of Eq (2.1) of the source on the periodic unit interval with the kernel family of Eq (5.1) (exponent alpha, horizon delta), initial data u(x,0) = sin(2 pi x) + cos(4 pi x)/2 and zero initial velocity, with the DG scheme of Eq (2.6) of degree k on N cells and the Crank-Nicolson scheme of Eq (4.1) with step ht up to time T. Project the initial data (step 2), assemble the operator (step 4, which uses step 3), march in time (step 6), verify with step 5 that the discrete energy after the first step and after the last step agree to a relative 1e-9 (raise ValueError otherwise), build the exact solution of the nonlocal problem from the eigenvalues of step 1 for the two modes present, and return the Gauss-Lobatto L2 error of step 7 at time T. Call the earlier step functions rather than reimplementing them. Raise ValueError if T is not a positive integer multiple of ht or any argument is invalid.

```python
import numpy as np


def nonlocal_wave_dg_l2_error(N, k, alpha, delta, ht, T):
    """float, the Gauss-Lobatto L2 error e_u(T) of the DG-CN solution."""
    return 0.0
```
