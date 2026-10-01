# Mathematics-Computational_Mechanics-45

## Background

Rate-dependent soft solids are modelled at finite strain by splitting the deformation gradient into elastic and viscous parts and carrying the viscous part as an internal variable. Choosing the scalar functions that govern the recoverable and the dissipative response is the hard part, because committing to a fixed analytical form trades expressiveness against robustness.

Work in this area therefore treats those functions as objects to be identified from mechanical test data while keeping the thermodynamic structure of generalized standard materials intact, so that admissibility does not have to be learned. Spline representations are attractive here because the identified functions remain readable as constitutive curves.

The identification returns constitutive functions together with the invariant ranges over which they are supported, and such methods are judged by how reproducibly they converge from different starting points and by how closely the predicted stress tracks loading and unloading data.

## Problem

Soft solids such as filled elastomers are strongly rate dependent at large strain, and calibrating a finite viscoelastic model for them means identifying scalar constitutive functions rather than fitting a handful of parameters. Data-adaptive formulations represent those functions as splines inside the generalized standard material structure, where a free energy and a dual dissipation potential fix the entire model and thermodynamic admissibility holds by construction. The awkward part is that the interpolation domain of each spline is itself unknown: the invariant range a non-equilibrium Maxwell branch visits is decided by its own internal evolution, not by the applied deformation, so the domains have to be identified along with the functions.

Each scalar constitutive function is built from its curvature, with the second derivative expanded in a degree-one B-spline basis on the current interpolation domain, the expansion coefficients and the left-endpoint slope each the image of an unconstrained free variable under one fixed smooth map onto the positive reals, and the function itself recovered by integrating twice from a vanishing value at the left endpoint. Under incompressible uniaxial loading the branch state is carried by the elastic left Cauchy-Green tensor: its isochoric invariants, taken in the polyconvex form used for spline-based energies, feed the branch free energy, while the deviatoric non-equilibrium Kirchhoff stress feeds the dual dissipation potential through a single stress invariant formed from the first two invariants of that stress. The branch is integrated implicitly by an exponential mapping algorithm that leaves the viscous deformation unimodular, and the local system it produces is solved by Newton iteration at every increment.

Calibration decouples the interpolation domains from the constitutive parameters by a block-alternating scheme: holding the identified response fixed, each outer iteration adapts the upper endpoint of every interpolation domain from the invariants the forward solve actually samples and then transfers the spline representation onto the moved basis so that the response is preserved. Solve one deterministic instance of this domain adaptation using the following configuration:

- deformation: incompressible uniaxial, F = diag(lam, lam^-0.5, lam^-0.5)
- stretch history: stretch_rate = 0.05 per second, stretch_max = 2.0, dt = 1.0 s, 40 increments in total, so the end-of-increment stretches run 1.05, 1.10, ..., 2.00 and then 1.95, 1.90, ..., 1.00
- initial internal state: at the start of every forward solve the viscous deformation is the identity
- branch structure: one non-equilibrium branch, free energy split as psi1(I1_e) + psi2(I2_e), dual dissipation potential phi(J_tau)
- spline resolution: three curvature coefficients per constitutive function, placed on a uniform grid of the interpolation domain, so theta = (theta_0, theta_1, theta_2, theta_3) with theta_0 the left-endpoint slope variable
- psi1: x_1 = 3, initial x_end = 3.02, theta = (-1.0, -2.0, -2.5, -3.0)
- psi2: x_1 = 0, initial x_end = 0.05, theta = (-2.0, -3.0, -3.5, -4.0)
- phi: x_1 = 0, initial x_end = 0.10, theta = (-4.0, -5.0, -5.5, -6.0)
- invariant samples: the value taken by each of I1_e, I2_e and J_tau at the converged state of every one of the 40 increments
- outer loop: n_outer = 3, smoothing parameter alpha_smooth = 5.0, relaxation eta = 0.5
- local Newton solve: residual infinity norm below 1e-12, at most 50 iterations

Your final answer must be a single number: the adapted upper endpoint x_end of the dissipation-potential interpolation domain after the three outer iterations.

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

01_curvature_spline_values

Goal
----
Evaluate one curvature-based spline constitutive function and its first two

derivatives at a set of arguments, given the interpolation domain and the

unconstrained free variables that parameterise it.

```python
import numpy as np


def curvature_spline_values(x_1: float, x_end: float, theta, x):
    """Evaluate a curvature-based spline and its first two derivatives.

    Parameters
    ----------
    x_1 : float
        Left endpoint of the interpolation domain.
    x_end : float
        Right endpoint of the interpolation domain, strictly greater than x_1.
    theta : array-like of shape (n_c + 1,)
        Free variables, theta[0] mapping to the left-endpoint slope and
        theta[1:] mapping to the n_c curvature coefficients.
    x : array-like of shape (m,)
        Arguments at which the function is evaluated.

    Returns
    -------
    spline_values : tuple of three np.ndarray of shape (m,)
        The function values normalised so that f(x_1) = 0, the first
        derivatives and the second derivatives, in that order.

    Raises
    ------
    ValueError
        If x_end is not strictly greater than x_1, if theta holds fewer than
        three entries, or if theta or x is not one dimensional.
    """
    return spline_values
```

### Step 2

02_uniaxial_kinematics

Goal
----
Return the trial elastic state of one increment of incompressible homogeneous

uniaxial loading, together with the squared principal stretches that map between

the elastic strain measure and the internal variable.

```python
import numpy as np


def uniaxial_kinematics(stretch: float, cv_inverse):
    """Return the trial elastic state of one uniaxial increment.

    Parameters
    ----------
    stretch : float
        End-of-increment stretch in the loading direction.
    cv_inverse : array-like of shape (3,)
        Principal values of the inverse viscous right Cauchy-Green tensor
        carried over from the previous increment.

    Returns
    -------
    trial_state : tuple of two np.ndarray of shape (3,)
        The principal values of the trial elastic left Cauchy-Green tensor and
        the squared principal stretches of the deformation gradient.

    Raises
    ------
    ValueError
        If stretch is not strictly positive or if cv_inverse does not hold
        exactly three strictly positive entries.
    """
    return trial_state
```

### Step 3

03_isochoric_invariants

Goal
----
Return the two isochoric invariants of a symmetric positive definite strain

measure from its three principal values, in the polyconvex form used by

spline-based constitutive functions.

```python
import numpy as np


def isochoric_invariants(principal_values):
    """Return the two isochoric invariants of a strain measure.

    Parameters
    ----------
    principal_values : array-like of shape (3,)
        Principal values of a symmetric positive definite strain measure.

    Returns
    -------
    invariants : tuple of two floats
        The first isochoric invariant and the transformed second isochoric
        invariant, the latter zero in the undeformed state.

    Raises
    ------
    ValueError
        If principal_values does not hold exactly three entries or if any entry
        is not strictly positive.
    """
    return invariants
```

### Step 4

04_branch_kirchhoff_stress

Goal
----
Return the deviatoric non-equilibrium Kirchhoff stress of one Maxwell branch in

its principal basis, together with the scalar stress invariant that the dual

dissipation potential depends on.

```python
import numpy as np


def branch_kirchhoff_stress(beta_e, psi1_spline, psi2_spline):
    """Return the branch stress and its deviatoric invariant.

    Parameters
    ----------
    beta_e : array-like of shape (3,)
        Principal values of the elastic left Cauchy-Green tensor.
    psi1_spline : tuple
        (x_1, x_end, theta) for the first free energy contribution.
    psi2_spline : tuple
        (x_1, x_end, theta) for the second free energy contribution.

    Returns
    -------
    branch_stress : tuple
        The principal components of the deviatoric non-equilibrium Kirchhoff
        stress and the deviatoric stress invariant driving the dual dissipation
        potential, in that order.

    Raises
    ------
    ValueError
        If beta_e does not hold exactly three strictly positive entries or if a
        spline descriptor is not a triple.
    """
    return branch_stress
```

### Step 5

05_viscous_flow_rate

Goal
----
Return the principal components of the viscous deformation-type rate of one

Maxwell branch from its deviatoric non-equilibrium Kirchhoff stress and the dual

dissipation potential.

```python
import numpy as np


def viscous_flow_rate(tau_neq, J_tau: float, phi_spline):
    """Return the principal viscous deformation-type rate of one branch.

    Parameters
    ----------
    tau_neq : array-like of shape (3,)
        Principal components of the deviatoric non-equilibrium Kirchhoff stress.
    J_tau : float
        Deviatoric stress invariant at which the dual potential is evaluated.
    phi_spline : tuple
        (x_1, x_end, theta) for the dual dissipation potential.

    Returns
    -------
    d_e : np.ndarray of shape (3,)
        Principal components of the viscous deformation-type rate.

    Raises
    ------
    ValueError
        If tau_neq does not hold exactly three entries, if J_tau is negative, or
        if phi_spline is not a triple.
    """
    return d_e
```

### Step 6

06_local_branch_update

Goal
----
Advance one Maxwell branch over a single time increment with the implicit

exponential mapping algorithm and return the converged principal values of the

elastic left Cauchy-Green tensor.

```python
import numpy as np


def local_branch_update(beta_trial, dt: float, psi1_spline, psi2_spline,
                        phi_spline, tol: float = 1e-12, max_iter: int = 50):
    """Advance one Maxwell branch over a single increment.

    Parameters
    ----------
    beta_trial : array-like of shape (3,)
        Principal values of the trial elastic left Cauchy-Green tensor.
    dt : float
        Time increment.
    psi1_spline, psi2_spline : tuple
        (x_1, x_end, theta) for the two free energy contributions.
    phi_spline : tuple
        (x_1, x_end, theta) for the dual dissipation potential.
    tol : float
        Tolerance on the residual infinity norm.
    max_iter : int
        Maximum number of Newton iterations.

    Returns
    -------
    beta_e : np.ndarray of shape (3,)
        Converged principal values of the elastic left Cauchy-Green tensor.

    Raises
    ------
    ValueError
        If beta_trial does not hold exactly three strictly positive entries, if
        dt is not strictly positive, or if max_iter is not a positive integer.
    """
    return beta_e
```

### Step 7

07_adapt_domain_endpoint

Goal
----
Return the updated upper endpoint of one spline interpolation domain from the

invariants the forward solve sampled, using a smooth upper-tail statistic and a

relaxed update.

```python
import numpy as np


def adapt_domain_endpoint(x_end: float, samples, alpha_smooth: float,
                          eta: float) -> float:
    """Return the relaxed update of one interpolation domain endpoint.

    Parameters
    ----------
    x_end : float
        Current upper endpoint of the interpolation domain.
    samples : array-like of shape (n,)
        Invariant values sampled by the current forward solve.
    alpha_smooth : float
        Smoothing parameter of the upper-tail statistic, strictly positive.
    eta : float
        Relaxation factor of the endpoint update, in (0, 1].

    Returns
    -------
    x_end_new : float
        Updated upper endpoint of the interpolation domain.

    Raises
    ------
    ValueError
        If samples is empty, if alpha_smooth is not strictly positive, or if eta
        lies outside (0, 1].
    """
    return x_end_new
```

### Step 8

08_project_spline_parameters

Goal
----
Transfer one curvature-based spline onto a moved interpolation domain and return

the free variables of the transferred representation.

```python
import numpy as np


def project_spline_parameters(x_1: float, x_end_old: float, theta_old,
                              x_end_new: float, samples):
    """Transfer a curvature-based spline onto a moved interpolation domain.

    Parameters
    ----------
    x_1 : float
        Left endpoint of the interpolation domain.
    x_end_old : float
        Upper endpoint the current representation is defined on.
    theta_old : array-like of shape (n_c + 1,)
        Free variables of the current representation.
    x_end_new : float
        Updated upper endpoint.
    samples : array-like of shape (n,)
        Invariant values the transfer is fitted on.

    Returns
    -------
    theta_new : np.ndarray of shape (n_c + 1,)
        Free variables of the transferred representation.

    Raises
    ------
    ValueError
        If either upper endpoint fails to exceed x_1, if theta_old holds fewer
        than three entries, or if samples is empty.
    """
    return theta_new
```

### Step 9

09_run_full_pipeline

Goal
----
Run the block-alternating calibration loop for the non-equilibrium branch and

return the adapted upper endpoint of the dissipation-potential interpolation

domain.

```python
import numpy as np


def run_full_pipeline(n_outer: int = 3, alpha_smooth: float = 5.0,
                      eta: float = 0.5, dt: float = 1.0,
                      stretch_rate: float = 0.05,
                      stretch_max: float = 2.0) -> float:
    '''Run the block-alternating calibration and return the adapted endpoint.

    Parameters
    ----------
    n_outer : int
        Number of outer iterations of the block-alternating scheme.
    alpha_smooth : float
        Smoothing parameter of the upper-tail statistic.
    eta : float
        Relaxation factor of the endpoint update.
    dt : float
        Time increment of the stretch history.
    stretch_rate : float
        Rate of the prescribed stretch.
    stretch_max : float
        Peak stretch of the loading and unloading history.

    Returns
    -------
    x_end_phi : float
        Adapted upper endpoint of the dissipation-potential interpolation
        domain after n_outer outer iterations.

    Raises
    ------
    ValueError
        If n_outer is not a non-negative integer, if stretch_max is not greater
        than one, or if the branch violates thermodynamic admissibility by
        storing a negative free energy or accumulating a negative reduced
        dissipation.
    '''
    return x_end_phi
```
