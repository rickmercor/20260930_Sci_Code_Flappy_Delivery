# Physics-Optics-17

## Background

Layered dielectric particles support optical resonances whose frequencies and losses respond to both geometry and material composition. Differentiable scattering calculations make these dependencies available to numerical design. A finite optimization trajectory tests whether complex electromagnetic sensitivities remain consistent through the physical objective and its parameterization.

## Problem

Quantify the improvement achieved by a fixed differentiable spectral-design run for a three-layer, concentric, isotropic, nonmagnetic nanosphere in vacuum, using the 2026 differentiable multilayer Mie-scattering approach that evaluates the layered-sphere coefficients through stable logarithmic-derivative recurrences and propagates design derivatives through the complete scattering calculation; retrieve its within-shell quotient and coupled electric/magnetic composite-impedance prescriptions. Use time dependence \(e^{-i\omega t}\), outgoing spherical Hankel functions of the first kind, and wavelength-independent relative refractive indices \(m_l=n_l+i\kappa_l\); all lengths below are in nm.

Let the nine physical design parameters be \(p=(r_1,d_2,d_3,n_1,n_2,n_3,\kappa_1,\kappa_2,\kappa_3)\), where the interface radii are \((r_1,r_1+d_2,r_1+d_2+d_3)\), and map unconstrained real variables to them by \(p_i=a_i+(b_i-a_i)/(1+e^{-u_i})\), with
\[
a=(35,12,15,2.8,1.35,2.1,0,0.015,0.005),\qquad
b=(85,48,65,4.2,2.15,3.3,0.07,0.16,0.08).
\]
At \(\lambda_j=430+17j\), \(j=0,\ldots,24\), compute the scattering and absorption efficiencies \(Q_s,Q_a\), defined as their cross sections divided by \(\pi R^2\), where \(R=r_1+d_2+d_3\); use multipoles \(n=1,\ldots,12\), and use order 80 with zero terminal regular logarithmic derivative as the fixed downward-recursion seed. These fixed truncations define the experiment; differentiate with respect to all nine real \(u_i\), including material, interface and area-normalization dependence.

The objective is
\[
L(u)=\frac1{25}\sum_{j=0}^{24}\left[(Q_s(\lambda_j;u)-v_j)^2+
\tfrac14Q_a(\lambda_j;u)^2\right],\qquad
v_j=0.15+3\exp\left[-\tfrac12((\lambda_j-620)/70)^2\right].
\]
Starting from \(u^{(0)}=(0.15,-0.4,0.35,-0.25,0.3,-0.1,-0.8,0.2,-0.5)\), perform exactly 40 full-batch Adam updates with learning rate 0.06, moment factors 0.9 and 0.999, both moments initially zero, and their usual bias corrections at iterations \(t=1,\ldots,40\); the denominator is the square root of the bias-corrected second moment plus \(10^{-8}\), with gradients evaluated at the old state and without weight decay, projection, early stopping or subsequent polishing.

Report \(\log_{10}(L(u^{(0)})/L(u^{(40)}))\) to absolute error at most \(10^{-7}\). In brief reasoning, give the initial and final losses, the three final interface radii, the three final complex indices, and the Euclidean norm of the final gradient with respect to \(u\), each to absolute error \(10^{-5}\); establish the radial logarithmic recurrence, the source's within-shell quotient and electric/magnetic composite-impedance prescriptions, the exterior coefficient extraction, and the real-parameter chain rule through the loss and physical normalization.

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

logarithmic_wave_jets

Goal
----
Stable logarithmic Riccati-Bessel data and parameter tangents.

```python
def logarithmic_wave_jets(
    z: "np.ndarray",
    dz: "np.ndarray",
    nmax: int,
    depth: int,
) -> "np.ndarray":
    """Return D1=psi'/psi and D3=xi'/xi with all first tangents.

    Use psi_n(z)=z*j_n(z), xi_n(z)=z*h_n^(1)(z). Seed D1_depth=0,
    then D1_(n-1)=n/z-1/(D1_n+n/z) for n=depth,...,1. Set D3_0=i,
    P0=(1-exp(2*i*z))/2, and for n=1,...,nmax set
    Pn=P_(n-1)*(n/z-D1_(n-1))*(n/z-D3_(n-1)), D3_n=D1_n+i/Pn.
    Propagate tangents through the finite recurrences, including the zero
    seed. The source uses this product recurrence to avoid separately large
    regular and outgoing radial factors. Evaluate 1-exp(2*i*z) accurately.

    Parameters
    ----------
    z : ndarray, shape (B,), complex128
        Arguments with 0.2 <= Re(z) <= 10 and 0 <= Im(z) <= 2, away
        from zeros of the regular or outgoing radial functions.
    dz : ndarray, shape (B, K), complex128
        Derivatives of z with respect to the K real parameters.
    nmax : int
        Largest returned order, between 1 and 16 inclusive.
    depth : int
        Downward seed order; nmax < depth <= 100 and depth >= 40.

    Returns
    -------
    ndarray, shape (nmax+1, 2, B, K+1), complex128
        Jets in order, kind (D1 then D3), batch, jet-slot order.

    Raises
    ------
    ValueError
        If any argument is zero, or nmax/depth violate their ranges.
    """
    return None
```

### Step 2

shell_ratio_jets

Goal
----
Stable propagation of the within-shell radial quotient.

```python
def shell_ratio_jets(
    z1: "np.ndarray",
    dz1: "np.ndarray",
    z2: "np.ndarray",
    dz2: "np.ndarray",
    inner: "np.ndarray",
    outer: "np.ndarray",
) -> "np.ndarray":
    """Return jets of Q_n=(psi_n(z1)/xi_n(z1))/(psi_n(z2)/xi_n(z2)).

    Here z1=m*k*r_inner and z2=m*k*r_outer for the same shell material.
    Use Q0=exp(2*i*(z2-z1))*expm1(2*i*z1)/expm1(2*i*z2). Then
    Qn=Q_(n-1)*(z1/z2)**2 *
    [(z2*D1_n(z2)+n)*(n-z2*D3_(n-1)(z2))] /
    [(z1*D1_n(z1)+n)*(n-z1*D3_(n-1)(z1))].
    Differentiate both the arguments and supplied logarithmic data.

    Parameters
    ----------
    z1, z2 : ndarray, shape (B,), complex128
        Inner/outer arguments in the preceding step's nonsingular domain,
        with a common complex phase and 0 < abs(z1) <= abs(z2).
    dz1, dz2 : ndarray, shape (B, K), complex128
        Real-parameter tangents of the arguments; their variations may differ.
    inner, outer : ndarray, shape (nmax+1, 2, B, K+1), complex128
        D1/D3 jets at z1/z2 from the preceding step, with 1 <= nmax <= 16.

    Returns
    -------
    ndarray, shape (nmax+1, B, K+1), complex128
        Quotient jets, including order zero. Equal bounding states give Q=1
        and zero tangents when their argument tangents also agree.
    """
    return None
```

### Step 3

composite_impedance_jets

Goal
----
Electric and magnetic composite impedances across a shell.

```python
def composite_impedance_jets(
    previous: "np.ndarray",
    m1: "np.ndarray",
    dm1: "np.ndarray",
    m2: "np.ndarray",
    dm2: "np.ndarray",
    inner: "np.ndarray",
    outer: "np.ndarray",
    ratio: "np.ndarray",
) -> "np.ndarray":
    """Return the source's electric and magnetic composite impedance jets.

    For electric polarization define G1=m2*H_previous-m1*D1_inner and
    G2=m2*H_previous-m1*D3_inner. For magnetic polarization interchange
    m1 and m2 in these two definitions. In either polarization,
    H_new=(G2*D1_outer-Q*G1*D3_outer)/(G2-Q*G1).
    In the core both H polarizations equal D1. Propagate every tangent;
    all products and quotients here act on jets, not independently on slots.

    Parameters
    ----------
    previous : ndarray, shape (nmax+1, 2, B, K+1), complex128
        Previous composite jets; polarization order is electric, magnetic.
    m1, m2 : ndarray, shape (B,), complex128
        Adjacent material indices, 1 <= Re(m) <= 4.5, 0 <= Im(m) <= 0.2.
    dm1, dm2 : ndarray, shape (B, K), complex128
        Their real-parameter derivatives.
    inner, outer : ndarray, shape (nmax+1, 2, B, K+1), complex128
        D1/D3 data for m2 at the inner and outer shell radii.
    ratio : ndarray, shape (nmax+1, B, K+1), complex128
        The shell Q jets. Inputs have 1 <= nmax <= 16 and finite,
        nonzero composite denominators.

    Returns
    -------
    ndarray, shape (nmax+1, 2, B, K+1), complex128
        Updated electric/magnetic composite jets.
    """
    return None
```

### Step 4

host_radial_jets

Goal
----
Host Riccati-Bessel functions with geometric tangents.

```python
def host_radial_jets(
    x: "np.ndarray",
    dx: "np.ndarray",
    logarithms: "np.ndarray",
) -> "np.ndarray":
    """Return the regular and outgoing host radial jets.

    Initialize psi_0=sin(x), xi_0=-i*exp(i*x). For n=1,...,nmax,
    psi_n=psi_(n-1)*(n/x-D1_(n-1)) and
    xi_n=xi_(n-1)*(n/x-D3_(n-1)). Propagate all tangents, including the
    host size parameter x. The host is vacuum with outgoing h_n^(1).

    Parameters
    ----------
    x : ndarray, shape (B,), float64
        Host size parameters, 0.2 <= x <= 3, away from radial zeros.
    dx : ndarray, shape (B, K), float64
        Derivatives of x with respect to real design parameters.
    logarithms : ndarray, shape (nmax+1, 2, B, K+1), complex128
        Corresponding D1/D3 jets with 1 <= nmax <= 16.

    Returns
    -------
    ndarray, shape (nmax+1, 2, B, K+1), complex128
        psi/xi jets, in that kind order.
    """
    return None
```

### Step 5

layered_coefficient_jets

Goal
----
Layered-sphere electric and magnetic scattering coefficients.

```python
def layered_coefficient_jets(
    x: "np.ndarray",
    dx: "np.ndarray",
    m: "np.ndarray",
    dm: "np.ndarray",
    composite: "np.ndarray",
    radial: "np.ndarray",
) -> "np.ndarray":
    """Return electric a_n and magnetic b_n jets at the outer boundary.

    For n=1,...,nmax, use h=H_a/m for a_n and h=m*H_b for b_n;
    each coefficient is [(h+n/x)*psi_n-psi_(n-1)] /
    [(h+n/x)*xi_n-xi_(n-1)]. Propagate material and geometric tangents.
    The convention is outgoing h_n^(1) with e^(-i*omega*t).

    Parameters
    ----------
    x : ndarray, shape (B,), float64
        Outer host size parameters, 0.2 <= x <= 3.
    dx : ndarray, shape (B, K), float64
        Derivatives of the outer size parameters.
    m : ndarray, shape (B,), complex128
        Outer material index, 1 <= Re(m) <= 4.5, 0 <= Im(m) <= 0.2.
    dm : ndarray, shape (B, K), complex128
        Derivatives of the outer index.
    composite : ndarray, shape (nmax+1, 2, B, K+1), complex128
        Electric/magnetic composite jets at the outer boundary.
    radial : ndarray, shape (nmax+1, 2, B, K+1), complex128
        Host psi/xi jets. Inputs have 1 <= nmax <= 16 and nonzero
        scattering denominators.

    Returns
    -------
    ndarray, shape (nmax, 2, B, K+1), complex128
        Coefficient jets in multipole (starting at 1), polarization,
        batch and jet-slot order.
    """
    return None
```

### Step 6

mie_efficiency_jets

Goal
----
Dimensionless efficiencies and their full shape derivatives.

```python
def mie_efficiency_jets(
    x: "np.ndarray",
    dx: "np.ndarray",
    coeff: "np.ndarray",
) -> "np.ndarray":
    """Return extinction, scattering and absorption efficiency jets.

    For w_n=2*n+1 and f=2/x**2, set Qext=f*sum(w_n*Re(a_n+b_n)),
    Qsca=f*sum(w_n*(abs(a_n)**2+abs(b_n)**2)), Qabs=Qext-Qsca.
    Include df=-4*dx/x**3 and d(abs(a)**2)=2*Re(conj(a)*da).
    Efficiencies are cross sections divided by pi*r_outer**2; they are real.

    Parameters
    ----------
    x : ndarray, shape (B,), float64
        Outer size parameters in [0.2, 3].
    dx : ndarray, shape (B, K), float64
        Their derivatives with respect to the real design parameters.
    coeff : ndarray, shape (nmax, 2, B, K+1), complex128
        Electric/magnetic coefficient jets, 1 <= nmax <= 16.

    Returns
    -------
    ndarray, shape (3, B, K+1), float64
        Efficiency jets in extinction, scattering, absorption order.
    """
    return None
```

### Step 7

spectral_loss_gradient

Goal
----
Differentiable spectral objective for the three-layer design.

```python
def spectral_loss_gradient(
    u: "np.ndarray",
) -> "np.ndarray":
    """Return the spectral objective and all nine design derivatives.

    Use three nonmagnetic concentric layers in vacuum with e^(-i omega t)
    time convention. Set p=lo+(hi-lo)/(1+exp(-u)) elementwise, with
    lo=(35,12,15,2.8,1.35,2.1,0,0.015,0.005) and
    hi=(85,48,65,4.2,2.15,3.3,0.07,0.16,0.08).
    The first three entries are core radius and two shell thicknesses in nm;
    radii are their cumulative sums. Entries 3:6 are real refractive indices
    and entries 6:9 are positive extinction coefficients, so m=p[3:6]+i*p[6:9].
    All materials are nondispersive. For lambda_j=430+17*j nm, j=0,...,24, set
    x_l=2*pi*r_l/lambda_j. Use multipoles n=1,...,12 and D1_80=0 as the fixed
    downward seed, differentiating every recurrence with respect to all nine
    u entries. Initialize both composite impedances from the core's D1;
    propagate across each shell using its material at both bounding radii;
    apply the outer coefficient formulas and host radial functions. Define
    Qext, Qsca and Qabs by geometric-area normalization, 2/x_outer**2.
    The target is v_j=0.15+3*exp(-0.5*((lambda_j-620)/70)**2). The loss is
    L=mean((Qsca-v)**2+0.25*Qabs**2). Include the derivatives of the area
    normalization and of the sigmoid, cumulative radii and complex materials.
    This is a fixed finite numerical experiment; no convergence stopping rule
    or adaptive multipole selection changes it.

    Parameters
    ----------
    u : ndarray, shape (9,), float64
        Finite real unconstrained variables with abs(u_i) <= 6.

    Returns
    -------
    ndarray, shape (10,), float64
        The loss L followed by its nine derivatives dL/du_i in input order.

    Raises
    ------
    ValueError
        If u has the wrong shape, nonfinite entries, or abs(u_i) > 6.
    """
    return None
```

### Step 8

adam_design_step

Goal
----
Bias-corrected Adam step for the physical design variables.

```python
def adam_design_step(
    state: "np.ndarray",
    grad: "np.ndarray",
    step: int,
    learning_rate: float,
) -> "np.ndarray":
    """Return one Adam update of variables and their two moment vectors.

    Set m_new=0.9*m_old+0.1*g and v_new=0.999*v_old+0.001*g**2.
    With iteration t=step, update u_new=u_old-learning_rate*
    [m_new/(1-0.9**t)]/[sqrt(v_new/(1-0.999**t))+1e-8].
    The small constant is outside the square root. There is no weight decay,
    projection, convergence stopping or moment resetting.

    Parameters
    ----------
    state : ndarray, shape (3, 9), float64
        Rows are u, first moment m and nonnegative second moment v.
    grad : ndarray, shape (9,), float64
        Gradient with respect to u at the old state. All data are finite.
    step : int
        One-based iteration number in [1, 60].
    learning_rate : float
        Positive learning rate at most 0.06.

    Returns
    -------
    ndarray, shape (3, 9), float64
        Updated u, m and v rows.

    Raises
    ------
    ValueError
        If step is not an integer in [1,60] or the learning rate is outside
        (0,0.06].
    """
    return None
```

### Step 9

run_layered_design

Goal
----
Final orchestrator: the fixed differentiable spectral design experiment.

```python
def run_layered_design(
    steps: int,
    learning_rate: float,
    initial: "np.ndarray",
) -> "np.ndarray":
    """Run the whole differentiable design using the earlier step functions.

    Use the preceding spectral loss and its full u-gradient at each old
    state, with zero first and second moments initially. Apply exactly
    steps Adam updates, numbered 1 through steps. Evaluate the loss and
    gradient again at the final updated u. Use the preceding sigmoid bounds
    to report physical parameters p. The benchmark is steps=40,
    learning_rate=0.06 and initial=(0.15,-0.4,0.35,-0.25,0.3,-0.1,
    -0.8,0.2,-0.5). No stopping test, randomization or extra polishing is used.
    Every earlier function contributes through the spectral-loss chain and
    Adam step. The state arrays must be carried between all updates.

    Parameters
    ----------
    steps : int
        Number of updates, from 0 to 60 inclusive.
    learning_rate : float
        Positive learning rate at most 0.06.
    initial : ndarray, shape (9,), float64
        Initial u; entries in [-1,1], with generated states required to stay
        in the spectral-loss domain [-6,6].

    Returns
    -------
    ndarray, shape (22,), float64
        log10(L_initial/L_final), L_initial, L_final, Euclidean norm of the
        final u-gradient, nine final physical parameters (core radius,
        two thicknesses, three real indices, three extinction coefficients),
        then nine final u entries. Zero steps gives a log reduction of zero.

    Raises
    ------
    ValueError
        If steps is not an integer in [0,60], the learning rate is outside
        (0,0.06], or initial has the wrong shape, nonfinite entries or an
        entry outside [-1,1].
    """
    return None
```
