[![CI](https://github.com/60k41p/MultiNEAT/actions/workflows/ci.yml/badge.svg)](https://github.com/60k41p/MultiNEAT/actions/workflows/ci.yml)

# MultiNEAT

MultiNEAT is a portable C++17 library for neuroevolution — training neural networks with a genetic algorithm. It is based on NEAT, which evolves both topology and weights through complexification from minimal genomes, historical markings for crossover alignment, and speciation with fitness sharing.

* Kenneth O. Stanley and Risto Miikkulainen, "Evolving Neural Networks through Augmenting Topologies," *Evolutionary Computation* 10(2), 2002. PDF: <https://nn.cs.utexas.edu/downloads/papers/stanley.ec02.pdf>

## Features

* **Evolution modes:** canonical NEAT (generational `Epoch()`), steady-state rtNEAT (`Tick()`), phased/simplifying search + delta coding, novelty search with behavior archive. (Note: `DetectCompetetiveCoevolutionStagnation`/`KillWorst*` parameters exist in `Parameters` but are never read by the evolution loop, so they currently have no effect.)
* **Genome types:** `PERCEPTRON` and `LAYERED` seeds, FS-NEAT (feature-selective, starts sparsely connected per `FS_NEAT_links`), recurrent links + leaky-integrator neurons, HyperNEAT/CPPN via `Substrate` (custom connectivity, weight-only query). ES-HyperNEAT code is present but disabled (`#if 0`) and experimental.
* **Variation:** add-neuron / add-link / remove-link / remove-simple-neuron mutations, weight perturb/replace, per-neuron activation-type/A/B/time-constant/bias mutation, multipoint/average crossover, inter-species crossover, elitism (champ copy; `EliteFraction` knob itself is currently hardcoded to 1 in `Species::Reproduce`), clone/archive control, custom topology constraints.
* **Neurons:** 14 activation functions (signed/unsigned sigmoid, tanh, tanh-cubic, signed/unsigned step, signed/unsigned Gauss, abs, signed/unsigned sine, linear, ReLU, softplus); multiple `Activate*()` paths including fast and leaky.
* **Speciation & selection:** tunable compatibility coefficients (disjoint/excess/weight/activation/time-constant/bias/function/traits), dynamic threshold, young-boost/old-penalty, tournament / roulette-wheel / truncation on the top `SurvivalRate` fraction. (Note: the `SelectionMode` enum lists rank/Boltzmann/stochastic modes, but the reproducer only implements the three above.)
* **Traits:** evolvable per-neuron/link/genome traits (`int`/`float`/`str`/`intset`/`floatset`) with dependency gating and speciation weighting.
* **Persistence:** save/load for `Genome`, `Population`, `Parameters`, `NeuralNetwork`.

## Requirements

* A C++17 compiler
* CMake 3.16 or later

## Building

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

This produces both a static library (`libMultiNEAT.a`) and a shared library (`libMultiNEAT.dylib`/`.so`/`.dll`), and installs them together with the headers:

```bash
cmake --install build
```

When building and linking manually, compile with `-std=c++17`.

### Usage

Add MultiNEAT as a CMake subdirectory and link against `MultiNEAT` (static) or `MultiNEAT_shared` (shared), then include the library headers:

```cpp
#include "Genome.h"
#include "Population.h"
#include "NeuralNetwork.h"
#include "Parameters.h"
```

### Codebase Lineage

This is a fork of <https://github.com/peter-ch/MultiNEAT>.

The primary divergence from the upstream is that the library is now `std`-only pure C++17; with Python bindings and the Boost dependency removed.
Furthermore, traits now use `std::variant`, RNG uses `std::mt19937`, and cycle detection uses Kahn's algorithm.

The conversion was benchmarked via an A/B harness against the [upstream](https://github.com/peter-ch/MultiNEAT/commit/7e3d9e326aa4d7c314fe263df18518c8e0e5aaac). Indicative 5-run medians on macOS/AppleClang — real workloads on par or faster; only the artificial pure-draw RNG microbenchmark regressed due to the libc++ `std::mt19937` cost:

| Benchmark | Boost | Std-only |
| --- | --- | --- |
| Trait hot-path | 69 ms | 63 ms |
| Cycle detection | 43 ms | 20 ms |
| End-to-end evolution | 9 ms | 8 ms |
| RNG throughput (pure draws) | 348 ms | 957 ms |
