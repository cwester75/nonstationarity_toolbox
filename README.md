# Nonlinear Dynamics Toolbox

A research-oriented Python toolbox for nonlinear dynamics, chaos analysis,
and symbolic dynamics. Implements the complete Chapter 1 pipeline covering
one-dimensional maps, two-dimensional maps, and fractal diagnostics.

## Features

**1-D Maps & Chaos**
- Logistic map, tent map, circle map, Bernoulli map
- Lyapunov exponent, fixed point classification, stability analysis
- Bifurcation diagrams, Feigenbaum delta estimation
- Symbolic dynamics, Shannon entropy, topological entropy
- Invariant density estimation, autocorrelation
- Transient chaos, escape times, chaotic repellers

**2-D Maps & Attractors**
- Henon map, Baker map (standard and generalized)
- Lyapunov spectrum via QR decomposition, Kaplan-Yorke dimension
- Fixed point finder and classifier (stable/unstable node, saddle, spiral)
- Basin of attraction computation
- Hyperchaos detection

**Fractal & Dimension Analysis**
- Correlation integral (Grassberger-Procaccia)
- Box-counting (capacity) dimension
- Newton fractals in the complex plane

**Advanced Topics**
- Melnikov analysis for homoclinic chaos
- Ruelle-Takens-Newhouse scenario detection
- Periodic orbit continuation, topological degree
- Ergodicity and mixing tests (Baker map)
- Chaotic communication (modulation/demodulation)
- Image dynamics (Arnold cat map, Baker map on images)

**Visualization**
- Cobweb diagrams, phase portraits, bifurcation plots
- Lyapunov exponent curves, Newton fractals
- Invariant density, correlation dimension scaling
- Basin of attraction heatmaps, escape time diagrams
- Arnold tongues, devil's staircase
- Melnikov function, Feigenbaum convergence

## Install

```
pip install -r requirements.txt
```

Or install as a package:

```
pip install -e .
```

## Run Tests

```
pytest
```

## Examples

```
python examples/logistic_bifurcation_demo.py
python examples/lyapunov_demo.py
python examples/symbolic_dynamics_demo.py
python examples/henon_demo.py
```

## Experiments

```
python experiments/logistic_period_doubling.py
python experiments/feigenbaum_estimation.py
python experiments/henon_attractor_demo.py
python experiments/newton_fractal_demo.py
```

## Package Structure

```
nonlinear/
    core/           Maps, trajectories, orbit tools
    maps2d/         Henon map, Baker map, 2-D iteration
    diagnostics/    Lyapunov, fixed points, stability, basins
    statistics/     Density, autocorrelation, correlation integral, dimension
    spectral/       DFT, FFT, power spectrum
    bifurcation/    Bifurcation diagrams, Feigenbaum, chaos detection
    symbolic/       Partitions, symbolic sequences, entropy
    phase/          Circle map, rotation number, Arnold tongues
    repellers/      Escape maps, transient chaos
    optimization/   Newton's method (1-D, complex, N-D)
    chaos/          Ruelle-Takens, Melnikov analysis
    topology/       Periodic orbits, fixed point theorems
    ergodic/        Ergodicity and mixing tests
    applications/   Image dynamics
    information/    Chaotic communication
    visualization/  All plotting functions
```
