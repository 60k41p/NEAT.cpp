[![CI](https://github.com/60k41p/NEAT.cpp/actions/workflows/ci.yml/badge.svg)](https://github.com/60k41p/NEAT.cpp/actions/workflows/ci.yml)

# NEAT.cpp

NEAT.cpp is a portable C++17 library for neuroevolution — training neural networks with a genetic algorithm. It is based on NEAT, which evolves both topology and weights through complexification from minimal genomes, historical markings for crossover alignment, and speciation with fitness sharing.

* Kenneth O. Stanley and Risto Miikkulainen, "Evolving Neural Networks through Augmenting Topologies," *Evolutionary Computation* 10(2), 2002. PDF: <https://nn.cs.utexas.edu/downloads/papers/stanley.ec02.pdf>

## Features

* **Evolution modes:** canonical NEAT (generational `Epoch()`), steady-state rtNEAT (`Tick()`), phased/simplifying search + delta coding, novelty search with behavior archive. (Note: `DetectCompetetiveCoevolutionStagnation`/`KillWorst*` parameters exist in `Parameters` but are never read by the evolution loop, so they currently have no effect.)
* **Genome types:** `PERCEPTRON` and `LAYERED` seeds, FS-NEAT (feature-selective, starts sparsely connected per `FS_NEAT_links`), recurrent links + leaky-integrator neurons, HyperNEAT/CPPN via `Substrate` (custom connectivity, weight-only query), ES-HyperNEAT (quadtree/octree subdivision with variance/band/LEO pruning).
* **Variation:** add-neuron / add-link (cycle-guarded, uniform over admissible pairs) / remove-link / remove-simple-neuron mutations, weight perturb/replace with uniform/Gaussian/Cauchy/polynomial distributions, per-neuron activation-type/A/B/time-constant/bias/spiking-parameter mutation, multipoint/average/single-point/blend/SBX crossover, inter-species crossover, `EliteFraction` elitism, clone/archive control, custom topology constraints (`Parameters::SetCustomConstraintsFunction`), adaptive multi-operator mutation budgets.
* **Neurons:** 18 activation functions (signed/unsigned sigmoid, tanh, tanh-cubic, signed/unsigned step, signed/unsigned Gauss, abs, signed/unsigned sine, linear, ReLU, softplus, LIF / adaptive-LIF / Izhikevich spiking, McCulloch-Pitts); multiple `Activate*()` paths including fast and leaky, plus event-driven `StepSpiking`/`SimulateSpiking` with current/binary/Poisson inputs and spike/rate/filtered/membrane outputs, per-connection STDP, and e-prop online learning (`EPropLearner`).
* **Speciation & selection:** tunable compatibility coefficients (disjoint/excess/weight/activation/time-constant/bias/function/spiking-link/spiking-neuron/traits), dynamic threshold (legacy step or proportional control), young-boost/old-penalty, parent selection (`LEGACY_SELECTION` preserving truncation/roulette/tournament switches, plus `TRUNCATION`, `ROULETTE`, `RANK_LINEAR`, `RANK_EXP`, `TOURNAMENT`, `STOCHASTIC`, `BOLTZMANN`), offspring allocation (`LARGEST_REMAINDER`/`STOCHASTIC_REMAINDER` with `MinSpeciesSize`/`SpeciesElitism` floors), and population-wide fitness scaling (`SHIFTED`/`LINEAR_RANK`/`SIGMA`/`BOLTZMANN`).
* **Traits:** evolvable per-neuron/link/genome traits (`int`/`float`/`str`/`intset`/`floatset`) with dependency gating and speciation weighting.
* **Persistence:** save/load for `Genome`, `Population`, `Parameters`, `NeuralNetwork`, plus versioned string `Serialize`/`Deserialize` checkpoints (genome format 4, species format 2, population format 2) with `Validate()` diagnostics on every level.

## Requirements

* A C++17 compiler
* CMake 3.16 or later

## Building

All build trees live under a single git-ignored `build/` directory, one subdirectory per configuration:

```bash
cmake -S . -B build/release -DCMAKE_BUILD_TYPE=Release
cmake --build build/release
```

This produces both a static library (`libNEAT.a`) and a shared library (`libNEAT.dylib`/`.so`/`.dll`), and installs them together with the headers:

```bash
cmake --install build/release
```

When building and linking manually, compile with `-std=c++17`.

### Usage

Add NEAT.cpp as a CMake subdirectory and link against `NEAT.cpp` (static) or `NEAT.cpp_shared` (shared), then include the library headers:

```cpp
#include "Genome.h"
#include "Population.h"
#include "NeuralNetwork.h"
#include "Parameters.h"
```

### Codebase Lineage

This is a fork of <https://github.com/peter-ch/MultiNEAT>.

* The library is now `std`-only pure C++17; with Python bindings and the Boost dependency removed
* Traits now use `std::variant`, RNG uses `std::mt19937`, and cycle detection uses Kahn's algorithm.
* The MultiNEAT v2 feature set has been backported (see `PLAN.md`): spiking neuron models with STDP and e-prop learning, ES-HyperNEAT, the full selection/crossover/mutation/scaling/representative/allocation/threshold control surface, and versioned serialization — with v2 defaults adopted, so evolution trajectories differ from the v1 baseline by design.
* Robust unit testing and benchmark harnesses were added, and many performance optimisations were applied.

### Performance

The conversion was benchmarked via an A/B harness against the [upstream](https://github.com/peter-ch/MultiNEAT/commit/7e3d9e326aa4d7c314fe263df18518c8e0e5aaac). Indicative 5-run medians on macOS/AppleClang — real workloads on par or faster; only the artificial pure-draw RNG microbenchmark regressed due to the libc++ `std::mt19937` cost:

| Benchmark | Boost | Std-only |
| --- | --- | --- |
| Trait hot-path | 69 ms | 63 ms |
| Cycle detection | 43 ms | 20 ms |
| End-to-end evolution | 9 ms | 8 ms |
| RNG throughput (pure draws) | 348 ms | 957 ms |

Hot genome paths were reworked algorithmically rather than tuned: repeated linear scans for neuron-id lookups are replaced by a one-time id→index table, and loop detection became a single topological sweep. On a ~1k-link genome phenotype construction is about 9x faster and pairwise compatibility distance about 4.5x faster, loop detection is now linear in the graph instead of quadratic, and small-genome builds improved by roughly a sixth. All of it is behaviour-preserving — seeded runs replay identical trajectories and solve reference tasks at the same generations, so saved genomes and running experiments are unaffected. The harness and reproducible numbers live in [benchmarks/RESULTS.md](benchmarks/RESULTS.md).
