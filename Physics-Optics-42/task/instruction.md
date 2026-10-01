# Physics-Optics-42

## Background

Electromagnetic chirality distinguishes waves of opposite handedness. In an isotropic chiral material, combinations of the electric and magnetic fields can be organized into left- and right-circular eigenfields, which propagate with different effective material factors even when the permittivity and permeability themselves are constant.

For a structure that is periodic in one lateral direction and bounded vertically, radiation leaving the computational layer can be represented by diffraction orders. Transparent boundary operators encode whether each order is propagating or evanescent without explicitly meshing the exterior half-spaces.

When a smooth material profile depends on a deformation parameter, regular perturbation theory can replace a sequence of unrelated scattering solves by recursively forced problems sharing one reference operator. Expanding about a nonzero parameter value allows the reference medium itself to be spatially varying, while analyticity in the deformation and spatial variables supports high-order spectral discretization.

Taylor series may cease to converge before reaching a scientifically interesting real deformation even though the underlying field remains regular there. Rational continuation can extend the usable parameter range, while a volume-averaged field intensity provides a robust measure of near-field enhancement throughout the layer.

## Problem

Periodic chiral layers can respond very differently to the two circular components of an incident electromagnetic field, and a useful scalar measure of their internal response is the volume-averaged electric-field intensity. Implement a NumPy/SciPy function that returns the second deformation derivative of the rationally continued volume-averaged electric-field intensity inside a two-dimensional periodic chiral slab.

Use the Drude–Born–Fedorov constitutive model with length-valued chirality \(\chi\), the \(e^{-i\omega t}\) convention, and permittivity and permeability equal to their normalized vacuum values throughout, so only chirality differs between the layer and exterior. The fields are independent of \(y\), and one slab cell is \(0\leq x<d\), \(-h\leq z\leq h\). Set \(d=0.8\), \(h=0.95\), \(t=0.5\), \(w=8\), \(\lambda=0.7\), \(\theta=\pi/6\), background chirality \(\bar\chi=0.006\), envelope amplitude \(\chi_a=0.04\), and lateral modulation scale \(a_{\rm lat}=1\). The chirality is \(\chi=\bar\chi-\delta X\), where \(X=\chi_a\Phi(z)(1+0.14z/h)G(x)\), \(\Phi(z)=[\tanh(w(z+t))-\tanh(w(z-t))]/2\), and \(G(x)=1+a_{\rm lat}[0.23\cos(2\pi x/d)-0.19\sin(4\pi x/d)+0.13\cos(6\pi x/d)+0.07\sin(8\pi x/d)]\).

Expand about \(\delta_c=0.25\), evaluate at \(\delta_\star=-1\), and retain orders zero through eighteen. Illuminate from \(z>h\) with the TE electric field \(\mathbf E^{\rm inc}(x,z)=(0,1,0)\exp[i k_0(x\sin\theta-z\cos\theta)]\), where \(k_0=2\pi/\lambda\), and decompose its boundary data into the two circular channels. Treat both exterior half-spaces as achiral vacuum while allowing the interior chirality trace to remain nonzero, and enforce tangential-field transmission before applying the outgoing Fourier Dirichlet-to-Neumann maps. Resolve the arbitrary-center recursive scalar problems with lateral nodes \(x_j=jd/18\), \(j=0,\ldots,17\), the retained Fourier basis \(\exp[i(\alpha+2\pi p/d)x]\) with \(\alpha=k_0\sin\theta\) and \(p=-9,\ldots,8\), and the 23 Chebyshev-Lobatto nodes \(z_r=h\cos(\pi r/22)\), \(r=0,\ldots,22\); use nodal products without dealiasing. Reconstruct the Cartesian electric-field coefficient series and form the Taylor coefficients of its volume-averaged squared magnitude, using the periodic trapezoidal rule in \(x\) and Clenshaw–Curtis quadrature in \(z\), with the volume integral divided by \(2hd\). Construct the normalized \([9/9]\) Pade continuation of that scalar series.

Return its second derivative at \(\delta_\star-\delta_c\) as one unrounded native Python `float` computed with binary64 real and complex arithmetic.

## Output format

```
## Output format

Scientific reasoning wrapped in <reasoning>...</reasoning> tags. Include enough intermediate calculations to justify the deterministic computation, without turning the response into a general pipeline summary. A single final numeric answer wrapped in <final_answer>...</final_answer> tags.
```

## Your task

Implement **all 10 functions** below into a single file at `/app/solution.py`. Requirements:

- Use the EXACT function name and signature shown for each step.
- Define every function at module scope (importable).
- Do NOT include tests, example usage, prints, or a `__main__` block.
- Available libraries: numpy, scipy, sympy, pandas, scikit-learn, networkx.

Start your file with these imports:

```python
import numpy as np
```

### Step 1

01_spectral_operators_and_weights

Goal
----
Construct tensor-product spectral derivatives and normalized volume weights.

```python
def spectral_operators_and_weights(nx: int, nz: int, d: float, h: float, alpha: float) -> "np.ndarray":
    '''Construct full-grid quasiperiodic derivative and quadrature operators.

    Parameters
    ----------
    nx : int
        Even number of equispaced lateral nodes, at least 4.
    nz : int
        Chebyshev polynomial degree, at least 2; there are nz + 1 nodes.
    d : float
        Positive lateral period in the common length unit.
    h : float
        Positive slab half-height in the common length unit.
    alpha : float
        Real Bloch wavenumber in inverse-length units.

    Returns
    -------
    operators : np.ndarray
        Complex128 array of shape (3, N, N), N = nx * (nz + 1).
        operators[0] is the x derivative, operators[1] the z derivative,
        and operators[2] a real diagonal matrix of normalized tensor-product
        quadrature weights.  Fields are flattened in C order with index
        j * (nz + 1) + r, x_j = j*d/nx, and
        z_r = h*cos(pi*r/nz), so r = 0 is the top boundary.  The diagonal
        weights sum to one and approximate (2*h*d)^(-1) times the volume
        integral.

    Raises
    ------
    ValueError
        If nx is odd or less than 4, nz is less than 2, d or h is not
        positive and finite, or alpha is not finite.
    '''
    return operators
```

### Step 2

02_radiation_and_bohren_data

Goal
----
Construct radiation branches and circular-channel incident boundary data.

```python
def radiation_and_bohren_data(nx: int, d: float, h: float, wavelength: float, theta: float) -> "np.ndarray":
    '''Return ordered diffraction data and unit-TE circular-channel forcing.

    Parameters
    ----------
    nx : int
        Even Fourier-node count, at least 4.
    d : float
        Positive lateral period.
    h : float
        Positive slab half-height.
    wavelength : float
        Positive vacuum wavelength.
    theta : float
        Finite incidence angle in radians, with abs(theta) < pi/2.

    Returns
    -------
    data : np.ndarray
        Complex128 array of shape (nx, 5), with rows ordered by
        p = -nx/2, ..., nx/2-1.  The columns are alpha_p, gamma_p,
        -i*gamma_p, the LCP top-forcing coefficient, and the RCP
        top-forcing coefficient.  Propagating gamma_p uses the nonnegative
        real root and evanescent gamma_p uses +i times the positive root.
        Only p = 0 has nonzero forcing for the incident field normalized by
        E_y = 1 and H_y = 0.

    Raises
    ------
    ValueError
        If dimensions or positive parameters are invalid, theta is outside
        the stated range, or any retained order is at a Wood anomaly, defined
        by abs(k0^2 - alpha_p^2) <= 64*eps*k0^2.
    '''
    return data
```

### Step 3

03_centered_channel_profiles

Goal
----
Build the envelope and arbitrary-center circular-channel coefficients.

```python
def centered_channel_profiles(sign: int, nx: int, nz: int, d: float, h: float, t: float, sharpness: float, wavelength: float, chi_bar: float, chi_amplitude: float, lateral_scale: float, delta_center: float) -> "np.ndarray":
    '''Construct the smooth envelope and one circular channel's coefficients.

    Parameters
    ----------
    sign : int
        +1 for the left-circular channel or -1 for the right-circular channel.
    nx, nz : int
        Even lateral node count and Chebyshev degree, using the same nodes as
        spectral_operators_and_weights.
    d, h : float
        Positive period and half-height.
    t : float
        Positive bump half-thickness with t < h.
    sharpness : float
        Positive inverse-length tanh sharpness.
    wavelength : float
        Positive vacuum wavelength.
    chi_bar : float
        Finite background chirality in length units.
    chi_amplitude : float
        Finite envelope amplitude in length units.
    lateral_scale : float
        Finite multiplier of the four fixed lateral harmonics.
    delta_center : float
        Finite global deformation value about which the local series is formed.

    Returns
    -------
    profiles : np.ndarray
        Float64 array of shape (3, N), flattened in C order.  Row zero is the
        envelope X, row one the centered material factor rho_c, and row two
        the local linear coefficient rho_1.  The envelope uses the exact four
        harmonics and vertical asymmetry stated in the problem.

    Raises
    ------
    ValueError
        If sign is not +1 or -1, grid or positive geometric inputs are
        invalid, t is not below h, a finite scalar is required but absent, or
        the centered profile violates max(abs(k0*(chi_bar-delta_center*X))) < 1.
    '''
    return profiles
```

### Step 4

04_assemble_transmission_operator

Goal
----
Assemble the transmission-matched arbitrary-center scattering operator.

```python
def assemble_transmission_operator(profiles: "np.ndarray", operators: "np.ndarray", radiation: "np.ndarray", nx: int, nz: int) -> "np.ndarray":
    '''Assemble one circular channel's transmission-matched collocation matrix.

    Parameters
    ----------
    profiles : np.ndarray
        Array of shape (3, N) from centered_channel_profiles.
    operators : np.ndarray
        Array of shape (3, N, N) from spectral_operators_and_weights.
    radiation : np.ndarray
        Array of shape (nx, 5) from radiation_and_bohren_data.
    nx, nz : int
        Grid dimensions with N = nx * (nz + 1).

    Returns
    -------
    matrix : np.ndarray
        Complex128 array of shape (N, N).  Interior rows discretize the
        centered weighted-divergence Helmholtz operator.  For each lateral
        node, the r = 0 and r = nz rows enforce tangential-field continuity
        to achiral vacuum and the supplied outgoing Fourier multiplier.  The
        normal derivative is therefore multiplied by rho_c at that boundary
        node; multiplication precedes application of the nonlocal DtN map.

    Raises
    ------
    ValueError
        If nx or nz is invalid or the supplied array shapes are inconsistent.
    '''
    return matrix
```

### Step 5

05_assemble_transmission_source

Goal
----
Assemble one transmission-matched perturbation source term.

```python
def assemble_transmission_source(previous: "np.ndarray", previous_two: "np.ndarray", profiles: "np.ndarray", operators: "np.ndarray", nx: int, nz: int) -> "np.ndarray":
    '''Form one arbitrary-center interior and interface recurrence source.

    Parameters
    ----------
    previous : np.ndarray
        Complex nodal field v_(m-1) of shape (N,).
    previous_two : np.ndarray
        Complex nodal field v_(m-2) of shape (N,); use exact zeros when the
        order does not yet exist.
    profiles : np.ndarray
        Shape (3, N) channel profiles containing rho_c and rho_1.
    operators : np.ndarray
        Shape (3, N, N) spectral operators; rows zero and one are D_x and D_z.
    nx, nz : int
        Grid dimensions with N = nx * (nz + 1).

    Returns
    -------
    source : np.ndarray
        Complex128 array of shape (N,) containing the centered recurrence
        source.  Nodal multiplication is used without dealiasing.  At r = 0
        the entry is rho_1 times the top normal derivative of v_(m-1); at
        r = nz it is the negative of that product.  These signs correspond
        to outward transmission conditions into achiral vacuum.

    Raises
    ------
    ValueError
        If grid dimensions or input shapes are inconsistent.
    '''
    return source
```

### Step 6

06_solve_transmission_channel_series

Goal
----
Generate a transmission-matched deformation series for one channel.

```python
def solve_transmission_channel_series(sign: int, matrix: "np.ndarray", profiles: "np.ndarray", operators: "np.ndarray", radiation: "np.ndarray", max_order: int) -> "np.ndarray":
    '''Solve all transmission-matched orders for one circular channel.

    Parameters
    ----------
    sign : int
        +1 for LCP or -1 for RCP.
    matrix : np.ndarray
        Complex transmission-matched boundary-value matrix of shape (N, N).
    profiles : np.ndarray
        Channel profiles of shape (3, N).
    operators : np.ndarray
        Spectral operators of shape (3, N, N).
    radiation : np.ndarray
        Radiation and incident data of shape (nx, 5).
    max_order : int
        Nonnegative highest deformation order.

    Returns
    -------
    series : np.ndarray
        Complex128 array of shape (max_order + 1, N). Row m contains the
        coefficient of the local deformation variable to power m in the
        selected scalar circular channel.

    Raises
    ------
    ValueError
        If sign, max_order, array shapes, or inferred grid dimensions are
        invalid.
    '''
    return series
```

### Step 7

07_reconstruct_electric_series

Goal
----
Reconstruct Cartesian electric-field coefficients from both scalar channels.

```python
def reconstruct_electric_series(left_series: "np.ndarray", right_series: "np.ndarray", left_profiles: "np.ndarray", right_profiles: "np.ndarray", operators: "np.ndarray", k0: float) -> "np.ndarray":
    '''Reconstruct the physical electric-field coefficient series.

    Parameters
    ----------
    left_series, right_series : np.ndarray
        Complex scalar series of identical shape (M + 1, N).
    left_profiles, right_profiles : np.ndarray
        Corresponding channel profiles, each of shape (3, N).
    operators : np.ndarray
        Spectral operators of shape (3, N, N); rows zero and one are D_x and D_z.
    k0 : float
        Positive finite vacuum wavenumber.

    Returns
    -------
    electric : np.ndarray
        Complex128 array of shape (M + 1, N, 3), with the final axis ordered
        as x, y, z.  It uses the order-expanded transverse Beltrami
        reconstruction and the normalized inverse circular transformation
        associated with the stated channel and time-harmonic conventions.

    Raises
    ------
    ValueError
        If k0 is invalid or any input shape is inconsistent.
    '''
    return electric
```

### Step 8

08_electric_intensity_coefficients

Goal
----
Form the real Taylor series of normalized volume-averaged electric intensity.

```python
def electric_intensity_coefficients(electric: "np.ndarray", operators: "np.ndarray") -> "np.ndarray":
    '''Compute the scalar intensity-series coefficients from electric fields.

    Parameters
    ----------
    electric : np.ndarray
        Complex array of shape (M + 1, N, 3) containing Cartesian electric
        field coefficients.
    operators : np.ndarray
        Array of shape (3, N, N) whose third matrix is the normalized diagonal
        quadrature matrix.

    Returns
    -------
    coefficients : np.ndarray
        Float64 array of shape (M + 1,) containing orders zero through M of
        the normalized quadrature volume average of the squared magnitude of
        the electric-field deformation series.

    Raises
    ------
    ValueError
        If input shapes are inconsistent or the quadrature diagonal is not
        finite and real to absolute tolerance 1e-13.
    '''
    return coefficients
```

### Step 9

09_pade_second_derivative

Goal
----
Continue a real Taylor series and evaluate its rational curvature.

```python
def pade_second_derivative(coefficients: "np.ndarray", numerator_degree: int, denominator_degree: int, evaluation: float) -> float:
    '''Return the second derivative of a normalized Pade approximant.

    Parameters
    ----------
    coefficients : np.ndarray
        One-dimensional finite real Taylor coefficient array.  At least
        numerator_degree + denominator_degree + 1 coefficients are required;
        later entries, if present, are ignored.
    numerator_degree, denominator_degree : int
        Nonnegative numerator and denominator degrees.
    evaluation : float
        Finite real point at which to evaluate the second derivative.

    Returns
    -------
    curvature : float
        Native Python float containing the analytic second derivative of the
        normalized rational approximant, with denominator constant term one.
        The denominator coefficients are obtained from the direct square
        coefficient-matching system; no least-squares fallback is used.

    Raises
    ------
    ValueError
        If degrees, coefficient shape, coefficient finiteness, or evaluation
        are invalid; if the denominator system is rank deficient; or if the
        evaluated denominator is zero within 64 machine epsilons of its
        absolute-term scale.
    '''
    return curvature
```

### Step 10

10_chiral_field_intensity_curvature

Goal
----
Compose the complete transmission-matched chiral continuation pipeline.

```python
def chiral_field_intensity_curvature(d: float, h: float, t: float, sharpness: float, wavelength: float, theta: float, chi_bar: float, chi_amplitude: float, lateral_scale: float, delta_center: float, delta_target: float, nx: int, nz: int, max_order: int, pade_numerator: int, pade_denominator: int) -> float:
    '''Run the complete arbitrary-center chiral-field intensity calculation.

    Parameters
    ----------
    d, h : float
        Positive lateral period and slab half-height.
    t : float
        Positive envelope half-thickness below h.
    sharpness : float
        Positive tanh-profile sharpness.
    wavelength : float
        Positive vacuum wavelength.
    theta : float
        Incidence angle in radians with abs(theta) < pi/2.
    chi_bar, chi_amplitude : float
        Finite background chirality and envelope amplitude.
    lateral_scale : float
        Finite multiplier of the four fixed lateral harmonics.
    delta_center, delta_target : float
        Finite global expansion center and requested physical deformation.
    nx : int
        Even Fourier-node count, at least 4.
    nz : int
        Chebyshev degree, at least 2.
    max_order : int
        Nonnegative maximum deformation order.
    pade_numerator, pade_denominator : int
        Nonnegative Pade degrees satisfying their sum no greater than
        max_order.

    Returns
    -------
    curvature : float
        Native Python float containing the analytic second derivative, at
        delta_target - delta_center, of the normalized Pade approximant built
        from the volume-averaged electric-intensity coefficient series.  The
        complete two-channel recursive pipeline is used.  No random state is
        used, and no intermediate rounding occurs.

    Raises
    ------
    ValueError
        Under the invalid conditions documented by the component functions,
        if the Pade degrees exceed max_order, or if the physical target
        profile violates max(abs(k0*chi)) < 1.
    '''
    return curvature
```
