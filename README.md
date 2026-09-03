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
* A C++11 compiler
* Boost (headers for `any`, `variant`, `shared_ptr`, `graph`, `random`, `accumulators`; libraries for `date_time` and `serialization`)

There is **no Python dependency**. This is a pure C++ library.

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

> **ABI note:** the build defines `USE_BOOST_RANDOM`, which changes the layout of `RNG` and classes that
> embed it (e.g. `Population`). When consuming via CMake this define is propagated automatically, but if
> you build and link manually you **must** compile your sources with `-DUSE_BOOST_RANDOM` (and link against
> the same Boost libraries) to keep the class layouts consistent with the library.


