# About MultiNEAT

MultiNEAT is a portable software library for performing neuroevolution, a form of machine learning that
trains neural networks with a genetic algorithm. It is based on NEAT, an advanced method for evolving
neural networks through complexification. The neural networks in NEAT begin evolution with very simple
genomes which grow over successive generations. The individuals in the evolving population are grouped
by similarity into species, and each of them can compete only with the individuals in the same species.

The combined effect of speciation, starting from the simplest initial structure and the correct
matching of the genomes through marking genes with historical markings yields an algorithm which
is proven to be very effective in many domains and benchmarks against other methods.

NEAT was developed around 2002 by Kenneth Stanley in the University of Texas at Austin.

### License

GNU Lesser General Public License v3.0

### Documentation

[http://multineat.com/docs.html](http://multineat.com/docs.html)

### Requirements

* CMake 3.5 or later
* A C++17 compiler
* **No external dependencies** (standard library only — Boost is no longer used)

### Boost removal

All Boost usage was removed: traits now use `std::variant`/`std::get`, the RNG uses
`std::mt19937` + `std::` distributions, and cycle detection uses an O(V+E) Kahn rewrite instead of
`boost::topological_sort`. Public APIs are unchanged, and the produced binaries contain no Boost
symbols (verified with `nm`/`otool`).

Before/after verification (5-run medians, macOS/AppleClang, BEFORE = Boost build):

| Benchmark | BEFORE (Boost) | AFTER (std-only) |
| --- | --- | --- |
| Trait hot-path | 69 ms | 63 ms |
| Cycle detection | 43 ms | 20 ms |
| End-to-end evolution | 9 ms | 8 ms |
| RNG throughput | 348 ms | 957 ms |

Functional equivalence between the builds was verified with an A/B harness (identical pass/fail
verdicts for all RNG/trait/cycle/evolution checks). The RNG microbenchmark regression is a
documented libc++ cost (`std::mt19937` and `std::` distributions are slower than Boost's
implementations); it only affects the artificial pure-draw microbenchmark, not real workloads.

#### To build

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

This produces both a static library (`libMultiNEAT.a`) and a shared library (`libMultiNEAT.dylib`/`.so`),
and installs them together with the headers:

```bash
cmake --install build
```

#### To use in your project

Add MultiNEAT as a CMake subdirectory and link against `MultiNEAT` (static) or `MultiNEAT_shared` (shared),
then include the library headers:

```cpp
#include "Genome.h"
#include "Population.h"
#include "NeuralNetwork.h"
#include "Parameters.h"
```

When building and linking manually, compile with `-std=c++17`; no additional libraries are required.
