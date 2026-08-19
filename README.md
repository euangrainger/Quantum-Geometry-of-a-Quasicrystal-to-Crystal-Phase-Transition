# Quantum-Geometry-of-a-Quasicrystal-to-Crystal-Phase-Transition

## Repository overview and computation flow

The main computational modes / simulation space can be found in the `hat_transition` folder. The sequence for producing the physical observables of the Hat -> Tetrille transition follows. Additionally, there are / will be additional readme files for breaking down each area further. 

### 1) Generate the lattice data object and bond labels

This is a two-stage split: `_tiling.py` produces the *geometry* (where every hat sits), and `lattice/generator.py` turns the geometry into a *graph* data object ( of sites, bonds, labels). 

- `_tiling.py`

Encodes the hat outline and the four metatiles (H, T, P, F) together with their substitution rules. Iterating the substitution to a chosen inflation level produces the placement of every hat in the patch.

- `lattice/generator.py`

Turns the placed hats into the simulation graph. Each hat is subdivided into its eight kites (`KITE_BONDS` lists each kite's four corners in terms of the reference points of the decorated hat outline). Walking every kite of every placed hat, coincident corners are merged into unique sites and kite edges become bonds.

  * the hat boundary carries 14 vertices 
  * every face of the resulting graph is a quadrilateral (a kite)

The edges are classified as they are collected. Each edges lying on a hat polygon are labelled `dual`, the remaining kite edges `tetrille`. Finally, the boundary completion `_complete_hexagons` fills the hexagons left incomplete at the patch boundary, embedding the patch in the vertex set of the periodic parent lattice (the Tetrille).

(A second orthogonal edge classification for the longer transition `inner`/`outer` from `lattice/bonds.py` s applied later in the build. This is after the Kwant system is finalised, because it needs the final graph's vertex degrees)

### 2) Verify the labelling and build the geometric masks

- `lattice/bipartite.py`

Ensures the construction obeys the bipartite structure and flags any possible upstream misconfiguration. A BFS two-colouring assigns sublattice labels A/B, returning any same-colour edges as conflicts (the non-bipartite flag). The passed anchor vertices (the hexagonal centres pin the global colour choice, so the *signed* sublattice imbalance is comparable across patches and seeds. The imbalance gives the minimum zero-mode count protected by chiral
symmetry.

- `lattice/bulk.py`

Builds the bulk mask so that physical measures, such as IPR and the Quantum Metric, can be taken without interference from the open boundaries. The concave hull (alpha-shape, alpha = 0.9) of the site cloud is eroded inwards by a buffer distance (1.8 lattice units). Sites inside the eroded polygon are bulk. Returns a boolean mask over the site indices.

### 3) Generate the NN TB model on the system data structure

We then define a Kwant system on the data structure and read the bond labels in order to modulate the couplings according to the imposed geometry.

- `simulation/hopping.py`

The script for hopping parameterisation,`HoppingParams` (the parameter set) and `hopping_value(params, ab, io)` the pure scalar rule mapping a bond's classification to its
coupling. On the Hat -> Tetrille leg this reduces to `dual` bonds at `-t` and `tetrille` bonds at `-lambda t`.

- `simulation/system.py`

Wraps Kwant. `build_system` constructs and finalises the graph (removes the dangling sites). The finalised site ordering becomes the ordering of the downstream array. `_hopping` is the Kwant callback adapter. It reassembles `HoppingParams` from Kwant's flattened parameters, looks up the bond's labels, and delegates to `hopping_value`. `hamiltonian_reduced` restricts to the *active subspace* so that sites that are dynamically disconnected at the current parameters (at lambda = 0 every tetrille-only site) are dropped, since each would otherwise contribute a spurious exact zero modes.

### 4) Compute the physical observables

- `analysis/quantum_metric_realspace.py`

The local real-space quantum-metric marker (TODO: write this up before push)

- `scripts/qm_sweep.py`

The overall sweep driver, it builds the system at each lambda on the transition grid, dense-diagonalises the active subspace, and stores the energetic spectrum (eigenvecs can be turned off however), per-state IPR moments, sublattice/site-class weights, and the quantum metric (half-filled, zero-mode-subspace, and gap-filling projectors). This feeds the static representation that can be done to demonstrate the results without having to re-pass the hat_transition folder. 

