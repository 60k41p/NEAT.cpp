# AGENTS.md

NEAT.cpp: C++17 neuroevolution library.

## General Rules

- Never guess or assume anything. When in doubt, check against the source code and the references.

## Code Rules

- Write idiomatic, DRY, high performance code.
- Use modern C++ language features, up to and including the C++17 standard.
- Use descriptive names for identifiers over short ones. Use a trailing underscore for private members (`foo_`), camelCase for functions, and PascalCase for types.
- Use explicit type names instead of `auto` unless the type names are extremely unwieldy.
- Conform to [Google C++ style guidelines](references/Google C++ Style Guide.md) with `.clang-format` settings applied on top.
- Header files must also serve as documentation; with code comments describing the purpose of the class, code unit or functionality that the header represents as a whole, with all properties and methods described individually, and citations of source material (title & section for papers and books, relative file path for reference source code). In implementation files, add comments only when the code is not self-explanatory, unintuitive or complex. Do not wrap comment lines manually before 200 characters.
- Every file in `src/` opens with the license banner, then a `/* File: ... Description: ... */` block, then (in headers) `#pragma once` — headers use no `#ifndef` include guards. Files derived from upstream MultiNEAT keep `Copyright (C) 2012 Peter Chervenski`, add `Modifications Copyright (C) 2026 Gökalp Özcan` and the "This file has been modified from its original version" notice (LGPL requires a prominent statement of modification); files original to this fork (`AssertMacros.h`, `Traits.h`, `Traits.cpp`) carry only the 2026 copyright and must not claim an upstream original.
- Source files use LF line endings.

## Validation & Test Coverage

Depending on what was modified and the change surface of the modifications, the respective test suite(s) must be run.

Add, remove, fix or enhance tests to ensure comprehensive coverage is maintained, that the tests are in line with the latest code state, and that edge cases are also tested. Never reduce or simplify a valid test to make it pass. The subdirectory structure should follow that of the production code. No standalone tests! The tests must always exercise the exact same code as production.

## Tooling

### Build & Test

The `build/` directory contains the build artifacts, one subdirectory per configuration.

```bash
cmake -S . -B build/debug -DCMAKE_BUILD_TYPE=Debug -DNEATCPP_ENABLE_TESTS=ON
cmake --build build/debug
ctest --test-dir build/debug --output-on-failure        # all tests
ctest --test-dir build/debug -R TestGenome              # single test
ctest --test-dir build/debug -L Evolution               # by label: Unit, IO, Evolution, Compliance
```

Standard variants (CI uses the same names, see `.github/workflows/ci.yml`):

| Directory | Configure flags |
| --------- | --------------- |
| `build/debug` | `-DCMAKE_BUILD_TYPE=Debug -DNEATCPP_ENABLE_TESTS=ON` |
| `build/release` | `-DCMAKE_BUILD_TYPE=Release -DNEATCPP_ENABLE_TESTS=ON` |
| `build/dbgassert` | Debug + `-DCMAKE_CXX_FLAGS=-DDEBUG` (turns on `ASSERT()`) |
| `build/san` | Debug + `-DCMAKE_CXX_FLAGS="-fsanitize=address,undefined -fno-omit-frame-pointer"` and matching linker flags |
| `build/coverage` | Debug + `-DCMAKE_CXX_FLAGS="--coverage -O0 -g"` and matching linker flags |
| `build/bench` | `-DCMAKE_BUILD_TYPE=Release -DNEATCPP_ENABLE_TESTS=OFF -DNEATCPP_ENABLE_BENCHMARKS=ON` |

Timing baselines and optimization notes for the `build/bench` harness live in `benchmarks/RESULTS.md`.

Sources are auto-discovered (`file(GLOB ... CONFIGURE_DEPENDS)`): a new file under `src/`, `tests/` or `benchmarks/` is picked up by the next build without editing any `CMakeLists.txt` or reconfiguring by hand.

CMake's `create_test_sourcelist` builds one driver (`NEATcppTests`); every `tests/TestX.cpp` is discovered automatically and must define `int TestX(int argc, char* argv[])`. A test categorises itself with directive comments at the top of the file — `// CTest-Labels: Evolution;Fast` and `// CTest-Timeout: 600` (defaults: `Unit;Fast` and 30s; unknown labels, malformed directives or a non-positive timeout fail configure; they are read at configure time, so editing one needs a reconfigure to take effect) — so labels are `Unit`, `IO`, `Evolution`, `Compliance`, `Fast`. Tests self-check with a local `CHECK` macro and must print "Test passed" — CTest `PASS_REGULAR_EXPRESSION` requires it; a test that exits 0 without printing it FAILs. Fixture data path: `NEATCPP_TEST_DATA_DIR` compile definition (see `tests/data/`). For a quick guide on CTest, see <https://cmake.org/cmake/help/book/mastering-cmake/chapter/Testing%20With%20CMake%20and%20CTest.html>.

Use Debug (or run the sanitizer config) when touching library code: `ASSERT()` in `src/AssertMacros.h` compiles to nothing without `DEBUG` defined, so Release silently skips invariant checks. CI also runs ASan+UBSan and GCC coverage with `--fail-under-line 50`.

### Formatting

CI checks with `clang-format==22.1.8` (pin this version; other versions reformat differently):

```bash
clang-format -i src/*.h src/*.cpp tests/*.cpp benchmarks/*.cpp
```

## Codebase gotchas

- Dead/disabled code: an old MutateGenome path is `#if 0`'d in `src/Species.cpp` — don't "fix" it; it is intentionally inactive. (ES-HyperNEAT was formerly `#if 0`'d as well; the MultiNEAT v2 port replaced it with a live implementation.)
- Most `Parameters` knobs are consumed by the evolution loop (selection/crossover/weight-mutation/representative/offspring/threshold/scaling modes, `EliteFraction`, `StagnationPenalty`, adaptive mutation). Still-unimplemented knobs: competitive-coevolution `DetectCompetetiveCoevolutionStagnation`/`KillWorst*` (validated but never read) and the 2D-loop `Depth`/`Qtree_Z` (used only by the 3D octree path). README documents each; verify a knob is actually consumed before assuming it works.
- Single flat namespace `NEAT`, headers in `src/` (public API: Genome, Population, NeuralNetwork, Parameters, Substrate, Traits).
- Serialization (`.NEAT` files) is hand-rolled text parsing in Save/Load methods — format changes break `tests/data/minimal.NEAT` fixtures and backward compat. Versioned string checkpoints (`GenomeFormat 4`, `SpeciesFormat 2`, `PopulationFormat 2` via `Serialize`/`Deserialize`) are the forward-compatible path.
