# Material_Science-Semiconductor_Materials-16

## Problem

The optical absorption of a crystal and the quantum geometry of its filled electronic states are two views of one object: the interband f-sum rule holds that the optical spectral weight, integrated against the reciprocal of the photon energy, is fixed by the Brillouin-zone average of the quantum metric of the occupied bands. Testing that identity on a real material means building a band model accurate enough to carry both sides of it at once, because the two are assembled from the same interband velocity matrix elements but weighted by different powers of the transition energy.

Consider monolayer ZrS2 in the 1T structure, described by the eleven-band orthogonal Slater-Koster tight-binding Hamiltonian that spans the five transition-metal d orbitals together with the three p orbitals on each of the two chalcogen planes, using the optimized Slater-Koster parameters and the lattice geometry reported for that compound. Take the orbital basis, in order, as the three p orbitals of the top chalcogen plane, then the metal d orbitals ordered as the axially symmetric one, the two in-plane ones, then the two out-of-plane ones, then the three p orbitals of the bottom chalcogen plane; and take the seventeen parameters, in order, as the three d on-site levels, the two p on-site levels, the three nearest-neighbour d-d integrals, the two nearest-neighbour p-p integrals, the two p-d integrals, the three next-nearest-neighbour d-d integrals and the two next-nearest-neighbour p-p integrals. The real-space primitive vectors of the triangular lattice are a(1, 0) and a(1/2, sqrt(3)/2), with a the in-plane lattice constant.

Sample the Brillouin zone on a uniform 30 by 30 grid anchored at the zone centre and spanned by the corresponding primitive reciprocal lattice vectors, take the six lowest bands to be occupied and the remaining five empty, and build the interband transitions along the first Cartesian axis from the velocity operator of this Hamiltonian. Broaden each transition into a normalised Lorentzian of half width 0.01 eV to obtain the real part of the diagonal interband optical conductivity, expressed per unit cell area in units of the conductance quantum, and integrate that conductivity divided by the photon energy using the trapezoidal rule over 13001 points spread uniformly across the closed window from 1.0 eV to 14.0 eV. Report that integrated optical spectral weight in units of square Angstrom.

State the conventions you adopted at each point where the framework leaves a choice open, justifying each from the source literature, and in your reasoning also report six further quantities for the same monolayer on the same grid: pi times the zone-averaged quantum metric of the occupied bands in square Angstrom, the signed percentage by which the integrated spectral weight exceeds it, the band gap in eV, the lowest vertical transition energy at the zone centre in eV, and the effective masses of the lowest empty band at the zone edge midpoint along the directions back towards the zone centre and on towards the zone corner, each in units of the free electron mass and each obtained by fitting a parabola to seven samples spaced 0.01 inverse Angstrom apart and centred on that extremum.

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
Do not paste the input matrices, full parameter vectors, per-k-point tables, or the sampled conductivity spectrum.
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

hopping_vectors

Goal
----
Return the fifteen bond vectors of a 1T-MX2 monolayer, stacked in one array in this order: the six metal-chalcogen bonds, then three in-plane nearest-neighbour vectors, then three in-plane next-nearest-neighbour vectors, then three vectors reaching from one chalcogen plane to the other. Within each of those four families, list the vectors in order of increasing azimuthal angle about the metal site, measured from the positive x axis. Only half of each in-plane family is listed, since a later step supplies the opposite members. The source fixes a convention here that the natural reading does not; follow the source.

```python
def hopping_vectors(a: float, theta: float) -> "np.ndarray":
    """Return the fifteen bond vectors of a 1T-MX2 monolayer, stacked in one array in this order: the six metal-chalcogen bonds, then three in-plane nearest-neighbour vectors, then three in-plane next-nearest-neighbour vectors, then three vectors reaching from one chalcogen plane to the other. Within each of those four families, list the vectors in order of increasing azimuthal angle about the metal site, measured from the positive x axis. Only half of each in-plane family is listed, since a later step supplies the opposite members. The source fixes a convention here that the natural reading does not; follow the source.

    Returns
    -------
    ndarray of shape (15, 3), the bond vectors in Angstrom, in the order described.

    Raises
    ------
    ValueError: if the lattice constant is not positive, or if the bond angle does not lie strictly between 0 and pi/2.
    """
    return result  # placeholder
```

### Step 2

bloch_hamiltonian

Goal
----
Return the eleven-band Bloch Hamiltonian at one wavevector, built from the bond vectors of the first step and the parameter vector, when deriv is -1; when deriv is 0, 1 or 2 return instead its derivative with respect to that Cartesian component of the wavevector. Place every block so the result is Hermitian, and put the crystal-field-split on-site energies on the diagonal. The source fixes a convention here that the natural reading does not; follow the source.

```python
def bloch_hamiltonian(k: "np.ndarray", vectors: "np.ndarray", params: "np.ndarray", deriv: int) -> "np.ndarray":
    """Return the eleven-band Bloch Hamiltonian at one wavevector, built from the bond vectors of the first step and the parameter vector, when deriv is -1; when deriv is 0, 1 or 2 return instead its derivative with respect to that Cartesian component of the wavevector. Place every block so the result is Hermitian, and put the crystal-field-split on-site energies on the diagonal. The source fixes a convention here that the natural reading does not; follow the source.

    Returns
    -------
    ndarray of shape (11, 11), complex: in eV when deriv is -1, in eV Angstrom otherwise.

    Raises
    ------
    ValueError: if k is not a vector of length three, if vectors does not have shape (15, 3), if params does not hold seventeen entries, or if deriv is not one of -1, 0, 1, 2.
    """
    return result  # placeholder
```

### Step 3

brillouin_zone

Goal
----
Return the two primitive reciprocal lattice vectors of the triangular lattice followed by the zone centre, the edge midpoint and the zone corner, stacked in one array in that order, for the real-space primitive vectors given in the problem statement. The source fixes a convention here that the natural reading does not; follow the source.

```python
def brillouin_zone(a: float) -> "np.ndarray":
    """Return the two primitive reciprocal lattice vectors of the triangular lattice followed by the zone centre, the edge midpoint and the zone corner, stacked in one array in that order, for the real-space primitive vectors given in the problem statement. The source fixes a convention here that the natural reading does not; follow the source.

    Returns
    -------
    ndarray of shape (5, 3), in inverse Angstrom: the two reciprocal lattice vectors, then the three high-symmetry points.

    Raises
    ------
    ValueError: if the lattice constant is not positive.
    """
    return result  # placeholder
```

### Step 4

indirect_gap

Goal
----
Return the band gap of the monolayer, sampling the zone on a uniform grid of n_grid by n_grid points anchored at the zone centre and spanned by the two reciprocal lattice vectors, with the given number of occupied bands. The source fixes a convention here that the natural reading does not; follow the source.

```python
def indirect_gap(vectors: "np.ndarray", reciprocal: "np.ndarray", params: "np.ndarray", n_grid: int, n_occupied: int) -> float:
    """Return the band gap of the monolayer, sampling the zone on a uniform grid of n_grid by n_grid points anchored at the zone centre and spanned by the two reciprocal lattice vectors, with the given number of occupied bands. The source fixes a convention here that the natural reading does not; follow the source.

    Returns
    -------
    float, the band gap in eV.

    Raises
    ------
    ValueError: if reciprocal does not have shape (5, 3), if n_grid is not positive, if n_occupied does not lie strictly between 0 and 11, or if any forwarded argument is invalid.
    """
    return 0.0  # placeholder
```

### Step 5

effective_mass

Goal
----
Return the effective mass of one band at one band extremum along one direction in momentum space, in units of the free electron mass, by fitting a parabola to that band on n_points samples spaced dk apart and centred on the extremum. The source fixes a convention here that the natural reading does not; follow the source.

```python
def effective_mass(vectors: "np.ndarray", params: "np.ndarray", band: int, k_extremum: "np.ndarray", direction: "np.ndarray", dk: float, n_points: int) -> float:
    """Return the effective mass of one band at one band extremum along one direction in momentum space, in units of the free electron mass, by fitting a parabola to that band on n_points samples spaced dk apart and centred on the extremum. The source fixes a convention here that the natural reading does not; follow the source.

    Returns
    -------
    float, the effective mass in units of the free electron mass.

    Raises
    ------
    ValueError: if direction or k_extremum is not a vector of length three, if direction is zero, if dk is not positive, if n_points is not an odd integer of at least three, or if the band is flat along the chosen direction.
    """
    return 0.0  # placeholder
```

### Step 6

interband_transitions

Goal
----
Return one row for every interband transition available on the same uniform zone grid: its energy, and the squared magnitude of the velocity matrix element between the two states along the given axis. Rows run over grid points in the order the grid is generated, and within a grid point over the pairs of bands an optical transition can connect. The source fixes a convention here that the natural reading does not; follow the source.

```python
def interband_transitions(vectors: "np.ndarray", reciprocal: "np.ndarray", params: "np.ndarray", n_grid: int, n_occupied: int, axis: int) -> "np.ndarray":
    """Return one row for every interband transition available on the same uniform zone grid: its energy, and the squared magnitude of the velocity matrix element between the two states along the given axis. Rows run over grid points in the order the grid is generated, and within a grid point over the pairs of bands an optical transition can connect. The source fixes a convention here that the natural reading does not; follow the source.

    Returns
    -------
    ndarray of shape (N, 2): the transition energy in eV in the first column and the squared velocity matrix element in eV^2 Angstrom^2 in the second.

    Raises
    ------
    ValueError: if reciprocal does not have shape (5, 3), if n_grid is not positive, if n_occupied does not lie strictly between 0 and 11, if axis is not 0, 1 or 2, or if any forwarded argument is invalid.
    """
    return result  # placeholder
```

### Step 7

quantum_metric

Goal
----
Return the zone-averaged quantum metric of the occupied states along the axis the transition list was built for, from that list and the linear size of the grid it was built on. The source fixes a convention here that the natural reading does not; follow the source.

```python
def quantum_metric(transitions: "np.ndarray", n_grid: int) -> float:
    """Return the zone-averaged quantum metric of the occupied states along the axis the transition list was built for, from that list and the linear size of the grid it was built on. The source fixes a convention here that the natural reading does not; follow the source.

    Returns
    -------
    float, the zone-averaged quantum metric in Angstrom^2.

    Raises
    ------
    ValueError: if transitions is not a non-empty array of shape (N, 2), if n_grid is not positive, or if any transition energy is not positive.
    """
    return 0.0  # placeholder
```

### Step 8

optical_conductivity

Goal
----
Return the real part of the diagonal interband optical conductivity at each photon energy in omega, from the transition list, the linear grid size, and a broadening eta that replaces each sharp transition by a normalised Lorentzian of that half width. Return it per unit cell area, in units of the conductance quantum. The source fixes a convention here that the natural reading does not; follow the source.

```python
def optical_conductivity(omega: "np.ndarray", transitions: "np.ndarray", n_grid: int, eta: float) -> "np.ndarray":
    """Return the real part of the diagonal interband optical conductivity at each photon energy in omega, from the transition list, the linear grid size, and a broadening eta that replaces each sharp transition by a normalised Lorentzian of that half width. Return it per unit cell area, in units of the conductance quantum. The source fixes a convention here that the natural reading does not; follow the source.

    Returns
    -------
    ndarray of shape (len(omega),), the real part of the conductivity in units of the conductance quantum per unit cell area.

    Raises
    ------
    ValueError: if transitions is not a non-empty array of shape (N, 2), if n_grid is not positive, if the broadening is not positive, or if any transition energy is not positive.
    """
    return result  # placeholder
```

### Step 9

spectral_weight

Goal
----
Return the integral of the real part of the optical conductivity divided by photon energy, taken by the trapezoidal rule over the photon energies supplied and the conductivity sampled at them. The source fixes a convention here that the natural reading does not; follow the source.

```python
def spectral_weight(omega: "np.ndarray", conductivity: "np.ndarray") -> float:
    """Return the integral of the real part of the optical conductivity divided by photon energy, taken by the trapezoidal rule over the photon energies supplied and the conductivity sampled at them. The source fixes a convention here that the natural reading does not; follow the source.

    Returns
    -------
    float, the integrated spectral weight in Angstrom^2.

    Raises
    ------
    ValueError: if omega and conductivity are not matching one-dimensional arrays of at least two points, or if any photon energy is not positive.
    """
    return 0.0  # placeholder
```

### Step 10

tb_optics_report

Goal
----
Run the whole analysis on one monolayer and report it. Build the geometry and the zone once and pass them down. Take the band gap on the declared grid; the lowest vertical transition at the zone centre; the effective masses of the highest occupied band at the zone centre towards the edge midpoint and towards the corner, and of the lowest empty band at the edge midpoint towards the zone centre and on towards the corner, all with the declared fit spacing and sample count; the single mass describing that anisotropic lower valley as a whole; then build the transition list along the first Cartesian axis and from it both pi times the zone-averaged quantum metric and, on a uniform grid of n_omega photon energies spanning the declared window, the conductivity and its integrated spectral weight; and finally report how far apart those last two are, as a percentage of the first. The source fixes a convention here that the natural reading does not; follow the source.

```python
def tb_optics_report(a: float, theta: float, params: "np.ndarray", n_grid: int, n_occupied: int, eta: float, omega_min: float, omega_max: float, n_omega: int, dk: float, n_points: int) -> "np.ndarray":
    """Run the whole analysis on one monolayer and report it. Build the geometry and the zone once and pass them down. Take the band gap on the declared grid; the lowest vertical transition at the zone centre; the effective masses of the highest occupied band at the zone centre towards the edge midpoint and towards the corner, and of the lowest empty band at the edge midpoint towards the zone centre and on towards the corner, all with the declared fit spacing and sample count; the single mass describing that anisotropic lower valley as a whole; then build the transition list along the first Cartesian axis and from it both pi times the zone-averaged quantum metric and, on a uniform grid of n_omega photon energies spanning the declared window, the conductivity and its integrated spectral weight; and finally report how far apart those last two are, as a percentage of the first. The source fixes a convention here that the natural reading does not; follow the source.

    Returns
    -------
    ndarray of shape (10,): the band gap in eV; the lowest vertical transition energy at the zone centre in eV; the two masses of the highest occupied band at the zone centre, towards the edge midpoint then towards the corner; the two masses of the lowest empty band at the edge midpoint, towards the zone centre then towards the corner; the single mass describing that valley; pi times the zone-averaged quantum metric in Angstrom^2; the integrated optical spectral weight in Angstrom^2; and the signed percentage by which the second exceeds the first.

    Raises
    ------
    ValueError: if the frequency window does not satisfy 0 < omega_min < omega_max, if n_omega is less than two, if the quantum metric vanishes, or if any forwarded value is invalid.
    """
    return result  # placeholder
```
