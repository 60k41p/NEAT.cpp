# AGENTS.md

NEAT.cpp: C++17-only neuroevolution library (NEAT variant). Fork of peter-ch/MultiNEAT with Boost and Python bindings removed — std-only (`std::variant` traits, `std::mt19937` RNG, Kahn's cycle detection). Do not reintroduce Boost/Python dependencies.

## Build & test

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug -DNEATCPP_ENABLE_TESTS=ON
cmake --build build
ctest --test-dir build --output-on-failure        # all tests
ctest --test-dir build -R TestGenome              # single test
ctest --test-dir build -L Evolution               # by label: Unit, IO, Evolution, Compliance
```

Use Debug (or run the sanitizer config) when touching library code: `ASSERT()` in `src/AssertMacros.h` compiles to nothing without `DEBUG` defined, so Release silently skips invariant checks. CI also runs ASan+UBSan and GCC coverage with `--fail-under-line 50` (new code must stay covered).

## Formatting

CI checks with `clang-format==22.1.8` (pin this version; other versions reformat differently):

```bash
clang-format -i src/*.h src/*.cpp tests/*.cxx
```

Non-default style (see `.clang-format`): 4-space indent, 160 columns, pointers right-aligned (`int *p`), namespace contents indented, `BinPackParameters: false`.

## Tests

No gtest/catch2. CMake's `create_test_sourcelist` builds one driver (`NEATcppTests`); each `tests/TestX.cxx` defines `int TestX(int argc, char* argv[])` — a new test file must be added to `NEATCPP_TEST_SOURCES` in `tests/CMakeLists.txt` and follow this exact signature. Tests self-check with a local `CHECK` macro and must print "Test passed" — CTest `PASS_REGULAR_EXPRESSION` requires it; a test that exits 0 without printing it FAILs. Fixture data path: `NEATCPP_TEST_DATA_DIR` compile definition (see `tests/data/`).

## Codebase gotchas

- Dead/disabled code: ES-HyperNEAT is `#if 0`'d in `src/Genome.cpp`; an old MutateGenome path is `#if 0`'d in `src/Species.cpp` — don't "fix" these; they are intentionally inactive.
- Some `Parameters` fields are parsed/saved but never read by the evolution loop (e.g. `EliteFraction`, competitive coevolution knobs, unimplemented `SelectionMode` values). README documents each; verify a knob is actually consumed before assuming it works.
- Single flat namespace `NEAT`, headers in `src/` (public API: Genome, Population, NeuralNetwork, Parameters, Substrate, Traits).
- Serialization (`.NEAT` files) is hand-rolled text parsing in Save/Load methods — format changes break `tests/data/minimal.NEAT` fixtures and backward compat.
