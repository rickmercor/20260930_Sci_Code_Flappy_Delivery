# Physics-Optics-22

## Background

Heat deposited in an optically pumped crystal changes its refractive index and can focus a propagating beam. Longitudinal absorption makes this focusing nonuniform, while a structured beam carries power in radial rings that a Gaussian width alone cannot describe. A model that divides the crystal into uniform optical slices is convenient, but the inferred heat load can differ systematically from that obtained with a continuously varying lens. The size of this discrepancy matters when a finite detector aperture is used to set an operating point.

## Problem

A cylindrically symmetric structured beam traverses an end-heated crystal whose radially parabolic thermal lens weakens along the propagation axis, and the deposited heat is adjusted to hold a fixed fraction of optical power inside an exit aperture. Work in a scalar paraxial model with radial heat conduction, constant on-axis index, an unbounded parabolic transverse index profile, and no optical loss, end-face refraction, stress contribution, or aperture before the exit plane. The crystal has length $L=0.010\,\mathrm m$, pumped radius $a=2.00\times10^{-4}\,\mathrm m$, conductivity $\kappa=14.0\,\mathrm{W\,m^{-1}\,K^{-1}}$, index $n_0=1.82$, thermo-optic coefficient $\beta=7.30\times10^{-6}\,\mathrm{K^{-1}}$, and absorption $\alpha_0=180\,\mathrm{m^{-1}}$; total deposited heat $Q$ normalizes a uniform-within-$a$ source proportional to $e^{-\alpha z}$. The entrance field is $U(r,0)=J_0(k\sin\theta\,r)e^{-r^2/w_0^2}$, with vacuum wavelength $\lambda=1.064\times10^{-6}\,\mathrm m$, internal wave number $k=2\pi n_0/\lambda$, envelope radius $w_0=7.00\times10^{-5}\,\mathrm m$, and internal cone angle $\theta=0.00300\,\mathrm{rad}$. Use the continuous position-slope ray transfer and axisymmetric diffraction transform to determine the field, and compare it with the transfer formed by $N$ equal constant-curvature slices, each evaluated at its axial midpoint and multiplied in propagation order with the last slice on the left.

For the exit radius $R=2.50\times10^{-5}\,\mathrm m$, let $F(Q,\alpha)$ be the power inside the aperture divided by the total entrance power, retaining the radial rings; let $F_N(Q,\alpha)$ use the sliced transfer with the same entrance field and normalization. At $\alpha_0$, define $Q_*$ and $Q_{*,N}$ as the simple roots of $F(Q_*,\alpha_0)=F_N(Q_{*,N},\alpha_0)=0.700000$ on the branch that contains $Q_*\in[250,300]\,\mathrm W$ and $Q_{*,N}$ for sufficiently large $N$. For the accompanying code functions, `order` defaults to 64 in `compute_ray_response` and 96 in `compute_aperture_response`, `solve_capture_heat`, `compute_midpoint_transfer_error`, and `compute_midpoint_heat_bias`; a supplied order overrides the default. Determine the single dimensionless leading relative heat-bias coefficient
$$
C=\lim_{N\to\infty}N^2\frac{Q_{*,N}-Q_*}{Q_*}
$$
to absolute error at most $2\times10^{-6}$, and in the reasoning justify the thermal curvature, both transfer descriptions, field and power normalization, root branch, and limiting response, reporting the entrance power $P_0$, $Q_*$, $g$ in $r''+g e^{-\alpha z}r=0$, the partial heat slope $F_Q$, and the leading sliced-fraction error $E_F=\lim_{N\to\infty}N^2(F_N-F)$ at $Q_*$.

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

01_compute_thermal_curvature

Goal
----
Compute the entrance ray curvature and its two thermal partial derivatives.

```python
def compute_thermal_curvature(
    heat: float,
    alpha: float,
    length: float,
    pump_radius: float,
    conductivity: float,
    index: float,
    thermooptic: float,
) -> "np.ndarray":
    r"""Return the curvature and its heat-load and absorption derivatives.

    Parameters
    ----------
    heat : float
        Total deposited heat $Q\geq0$, in $\mathrm{W}$.
    alpha : float
        Absorption coefficient $\alpha\geq0$, in $\mathrm{m^{-1}}$.
    length : float
        Crystal length $L>0$, in $\mathrm{m}$.
    pump_radius : float
        Pump radius $a>0$, in $\mathrm{m}$.
    conductivity : float
        Thermal conductivity $\kappa>0$, in $\mathrm{W\,m^{-1}\,K^{-1}}$.
    index : float
        On-axis index $n_0>0$.
    thermooptic : float
        Thermo-optic coefficient $\beta>0$, in $\mathrm{K^{-1}}$.

    Returns
    -------
    curvature : np.ndarray
        Shape $(3,)$ in the order $(g,\partial_Qg,\partial_\alpha g)$,
        with units $\mathrm{m^{-2}}$, $\mathrm{m^{-2}\,W^{-1}}$, and
        $\mathrm{m^{-1}}$, respectively.

    Raises
    ------
    ValueError
        If an input is nonfinite, a nonnegative input is negative, or a
        strictly positive input is nonpositive.

    Notes
    -----
    The zero-absorption limit holds total deposited heat fixed.
    """
    return
```

### Step 2

02_compute_ray_matrix

Goal
----
Compute continuous paraxial ray transport through exponentially decreasing focusing.

```python
def compute_ray_matrix(g: float, alpha: float, distance: float) -> "np.ndarray":
    r"""Return the transfer matrix for the position-slope state.

    Parameters
    ----------
    g : float
        Entrance curvature $g\geq0$, in $\mathrm{m^{-2}}$.
    alpha : float
        Absorption coefficient $\alpha\geq0$, in $\mathrm{m^{-1}}$.
    distance : float
        Propagation distance $z\geq0$, in $\mathrm{m}$.

    Returns
    -------
    matrix : np.ndarray
        Real shape $(2,2)$ array $M$ mapping entrance $(r,r')$ to exit
        $(r,r')$. Its diagonal entries are dimensionless, $B$ has units
        $\mathrm{m}$, and $C$ has units $\mathrm{m^{-1}}$.

    Raises
    ------
    ValueError
        If an input is negative or nonfinite, or finite transport cannot
        be computed at the supplied numerical scale.
    """
    return
```

### Step 3

03_compute_ray_response

Goal
----
Compute a directional derivative of the continuous optical transfer matrix.

```python
def compute_ray_response(
    g: float,
    alpha: float,
    length: float,
    g_direction: float,
    alpha_direction: float,
    order: int = 64,
) -> "np.ndarray":
    r"""Return a transfer derivative along the supplied parameter direction.

    Parameters
    ----------
    g : float
        Finite curvature $g\geq0$, in $\mathrm{m^{-2}}$.
    alpha : float
        Finite absorption $\alpha\geq0$, in $\mathrm{m^{-1}}$.
    length : float
        Finite propagation length $L\geq0$, in $\mathrm{m}$.
    g_direction : float
        Finite $\dot g=dg/d\varepsilon$.
    alpha_direction : float
        Finite $\dot\alpha=d\alpha/d\varepsilon$.
    order : int, optional
        Quadrature order, an integer at least $16$; default $64$.

    Returns
    -------
    response : np.ndarray
        Shape $(2,2)$ real array $\dot M(L)$. Units are those of each
        transfer entry divided by the units of $\varepsilon$.

    Raises
    ------
    ValueError
        If a ray input violates its nonnegative finite contract, either
        direction is nonfinite, or the quadrature order is invalid.

    Notes
    -----
    Boundary derivatives are continuous one-sided extensions where needed.
    """
    return
```

### Step 4

04_compute_input_power

Goal
----
Compute the total radial power of the unit-amplitude entrance field.

```python
def compute_input_power(wave_number: float, waist: float, cone_angle: float) -> float:
    r"""Return the entrance power in the unit-amplitude field convention.

    Parameters
    ----------
    wave_number : float
        Positive finite internal $k$, in $\mathrm{m^{-1}}$.
    waist : float
        Positive finite amplitude-envelope radius $w$, in $\mathrm{m}$.
    cone_angle : float
        Finite internal angle $0\leq\theta<\pi/2$, in radians.

    Returns
    -------
    power : float
        Positive total $2\pi\int_0^\infty |U(r,0)|^2r\,dr$, in
        $\mathrm{m^2}$ under unit field amplitude.

    Raises
    ------
    ValueError
        If the wave number or waist is nonpositive or nonfinite, or the
        angle is nonfinite or outside its stated interval.
    """
    return
```

### Step 5

05_compute_field_response

Goal
----
Compute the propagated radial field and its directional derivative.

```python
def compute_field_response(
    radii: "np.ndarray",
    matrix: "np.ndarray",
    response: "np.ndarray",
    wave_number: float,
    waist: float,
    cone_angle: float,
) -> "np.ndarray":
    r"""Return the radial field and its transfer-direction response.

    Parameters
    ----------
    radii : np.ndarray
        Nonempty finite nonnegative real vector of radii, in $\mathrm{m}$.
    matrix : np.ndarray
        Real finite shape $(2,2)$ position-slope transfer with
        $|\det M-1|\leq10^{-8}$.
    response : np.ndarray
        Real finite shape $(2,2)$ derivative $\dot M$ satisfying
        $|D\dot A+A\dot D-C\dot B-B\dot C|\leq10^{-7}$.
    wave_number : float
        Positive finite internal $k$, in $\mathrm{m^{-1}}$.
    waist : float
        Positive finite input amplitude-envelope radius $w$, in $\mathrm{m}$.
    cone_angle : float
        Finite internal angle $0\leq\theta<\pi/2$, in radians.

    Returns
    -------
    field : np.ndarray
        Complex shape $(2,N)$ array: row zero is $U(r)$ and row one is
        $\dot U(r)$ at the input radii in their original order.

    Raises
    ------
    ValueError
        If radii or matrix shapes, reality, finiteness, determinant, or
        tangent constraints fail; if a beam parameter is invalid; or if
        the evaluated field or response is nonfinite.

    Notes
    -----
    Inputs are not modified. The stated phase convention also applies at $B=0$.
    """
    return
```

### Step 6

06_compute_aperture_response

Goal
----
Integrate the captured optical power fraction and its directional derivative.

```python
def compute_aperture_response(
    matrix: "np.ndarray",
    response: "np.ndarray",
    wave_number: float,
    waist: float,
    cone_angle: float,
    aperture: float,
    order: int = 96,
) -> "np.ndarray":
    r"""Return the aperture fraction and its transfer-direction derivative.

    Parameters
    ----------
    matrix : np.ndarray
        Finite real determinant-one transfer of shape $(2,2)$, with
        determinant tolerance $10^{-8}$.
    response : np.ndarray
        Finite real shape $(2,2)$ tangent derivative, with absolute
        first-order determinant tolerance $10^{-7}$.
    wave_number : float
        Positive finite internal wave number, in $\mathrm{m^{-1}}$.
    waist : float
        Positive finite entrance envelope radius, in $\mathrm{m}$.
    cone_angle : float
        Finite internal angle in $[0,\pi/2)$, in radians.
    aperture : float
        Finite aperture radius $R\geq0$, in $\mathrm{m}$.
    order : int, optional
        Integer radial quadrature order at least $16$, default $96$.

    Returns
    -------
    capture : np.ndarray
        Shape $(2,)$ real array $(F,\dot F)$, where $F$ is dimensionless
        and $\dot F$ is per unit of the directional parameter.

    Raises
    ------
    ValueError
        If aperture or order is invalid, or a matrix, direction, or beam
        input violates the field-response contracts, including nonfinite
        evaluated fields.

    Notes
    -----
    Inputs are not modified. A zero aperture returns two zeros.
    """
    return
```

### Step 7

07_solve_capture_heat

Goal
----
Find the heat load on a specified branch of the aperture-capture equation.

```python
def solve_capture_heat(
    thermal: "np.ndarray",
    beam: "np.ndarray",
    alpha: float,
    aperture: float,
    target: float,
    heat_interval: "np.ndarray",
    order: int = 96,
) -> float:
    r"""Return the heat load meeting the aperture target on the specified branch.

    Parameters
    ----------
    thermal : np.ndarray
        Finite positive shape $(5,)$ vector $(L,a,\kappa,n_0,\beta)$ in
        $\mathrm{m}$, $\mathrm{m}$, $\mathrm{W\,m^{-1}\,K^{-1}}$,
        dimensionless units, and $\mathrm{K^{-1}}$, respectively.
    beam : np.ndarray
        Finite real shape $(3,)$ vector $(\lambda,w_0,\theta)$: positive
        vacuum wavelength and envelope radius in $\mathrm{m}$, and
        internal cone angle in $[0,\pi/2)$ radians.
    alpha : float
        Finite absorption $\alpha\geq0$, in $\mathrm{m^{-1}}$.
    aperture : float
        Positive finite exit aperture radius, in $\mathrm{m}$.
    target : float
        Finite desired captured fraction strictly between zero and one.
    heat_interval : np.ndarray
        Finite real shape $(2,)$ vector of strictly increasing positive
        heat loads, in $\mathrm{W}$, enclosing one simple root of the
        capture equation; an endpoint root is allowed.
    order : int, optional
        Integer quadrature order at least $16$, default $96$.

    Returns
    -------
    heat : float
        Heat load $Q_*$ in $\mathrm{W}$ with capture residual at most
        $10^{-9}$ for a resolved quadrature.

    Raises
    ------
    ValueError
        If input arrays have invalid shape, reality, finiteness, or range;
        alpha, aperture, target, or order is invalid; the interval does not
        bracket a root; or a called thermal, transfer, or field computation
        rejects its data.
    RuntimeError
        If the bracketed scalar solve fails to converge.

    Notes
    -----
    Uniqueness inside the supplied interval is a caller precondition.
    Inputs are not modified. There is no random state.
    """
    return
```

### Step 8

08_compute_midpoint_transfer_error

Goal
----
Find the leading transfer error of a midpoint-sliced thermal lens.

```python
def compute_midpoint_transfer_error(
    g: float, alpha: float, length: float, order: int = 96
) -> "np.ndarray":
    r"""Return the leading midpoint-stack transfer error.

    Parameters
    ----------
    g : float
        Finite entrance ray curvature $g\geq0$, in $\mathrm{m^{-2}}$.
    alpha : float
        Finite absorption coefficient $\alpha\geq0$, in $\mathrm{m^{-1}}$.
    length : float
        Finite propagation length $L\geq0$, in $\mathrm{m}$.
    order : int, optional
        Integer quadrature order at least $16$, default $96$.

    Returns
    -------
    error : np.ndarray
        Finite real shape $(2,2)$ coefficient $E_M$, with position-slope
        transfer entry units $(1,\mathrm m;\mathrm{m^{-1}},1)$.

    Raises
    ------
    ValueError
        If an input is nonfinite or negative, order is invalid, or the
        evaluated matrix coefficient is nonfinite.

    Notes
    -----
    No input is modified and no random state is used.
    """
    return
```

### Step 9

09_compute_midpoint_heat_bias

Goal
----
Determine the leading operating-heat bias caused by midpoint slicing.

```python
def compute_midpoint_heat_bias(
    thermal: "np.ndarray",
    beam: "np.ndarray",
    alpha: float,
    aperture: float,
    target: float,
    heat_interval: "np.ndarray",
    order: int = 96,
) -> float:
    r"""Return the dimensionless leading sliced-root heat bias.

    Parameters
    ----------
    thermal : np.ndarray
        Finite positive shape $(5,)$ vector $(L,a,\kappa,n_0,\beta)$,
        in $\mathrm m$, $\mathrm m$, $\mathrm{W\,m^{-1}\,K^{-1}}$,
        dimensionless units, and $\mathrm{K^{-1}}$.
    beam : np.ndarray
        Finite real shape $(3,)$ vector $(\lambda,w_0,\theta)$, with
        positive wavelength and waist in $\mathrm m$ and internal angle
        in $[0,\pi/2)$ radians.
    alpha : float
        Finite absorption $\alpha\geq0$, in $\mathrm{m^{-1}}$.
    aperture : float
        Positive finite exit aperture radius in $\mathrm m$.
    target : float
        Desired finite captured fraction strictly between zero and one.
    heat_interval : np.ndarray
        Finite shape $(2,)$ positive increasing heat bracket in $\mathrm W$
        containing one simple continuous root.
    order : int, optional
        Integer quadrature order at least $16$, default $96$.

    Returns
    -------
    bias : float
        Dimensionless leading coefficient $C$ of the relative heat bias.

    Raises
    ------
    ValueError
        If an input has invalid shape, finiteness, reality, or range;
        the interval fails to bracket a root; or the heat response vanishes.
    RuntimeError
        If the bracketed operating-heat solve does not converge.

    Notes
    -----
    Inputs are not modified. The interval selects the desired root branch.
    """
    return
```
