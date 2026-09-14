# Benchmarks

Reproducible timing baselines for NEAT.cpp, used to validate that
optimizations are real and to catch regressions.

## How to run

```bash
cmake -S . -B build-bench -DCMAKE_BUILD_TYPE=Release -DNEATCPP_ENABLE_TESTS=OFF -DNEATCPP_ENABLE_BENCHMARKS=ON
cmake --build build-bench -j
./build-bench/benchmarks/NEATcppBench
```

The harness (`benchmarks/Benchmarks.cxx`) uses fixed seeds and fixed iteration
counts, so output is comparable across runs and changes. Values below are
ns/op unless noted; machine: Apple clang 17.0.0, Intel i5-8259U @ 2.30 GHz
(8 cores), macOS 15.7.3. Baseline commit: `aa31413` (uplift-2).

## Baseline (2026-09-15)

| Benchmark                    | ops   | ns/op     | notes                                          |
| ---------------------------- | ----- | --------- | ---------------------------------------------- |
| BuildPhenotype small         | 200k  | 728       | 10-in/3-out perceptron (13 links)              |
| BuildPhenotype large         | 2k    | 162,952   | complexified genome (~213 neurons, ~1k links)  |
| Activate large net           | 20k   | 4,145     | one Flush+Input+Activate over ~1k connections  |
| CompatibilityDistance        | 20k   | 47,841    | two large genomes (~1k links each)             |
| Copy+mutate genome           | 50k   | 81,500    | genome copy + Mutate_LinkWeights + AddLink     |
| Epoch pop100                 | 100   | 873,553   | full generation on pop=100 XOR-style topology  |
| XOR solve seed1              | 1     | 139.7 ms  | reference recipe; solved at generation 31      |
| Genome save+load             | 100   | 3,924,968 | large genome round-trip through text file      |
| Population save+load         | 20    | 12,727,178| pop=100 full save/load round-trip              |

Run-to-run spread observed on this machine: <10% on all benchmarks except
Population save+load (~15%, disk noise).

## Observations / optimization candidates

- `Copy+mutate genome` (~82 us) dominates reproduction: every mating and every
  species copy round-trips full `std::vector`s of genes (and `NeuralNetwork`'s
  per-neuron sensitivity matrix is part of `Neuron`, copied with genomes? —
  verify). Any per-epoch cost scales with this.
- `CompatibilityDistance` (~48 us) is O(links) per pair but called for every
  individual per epoch with `DynamicCompatibility`; fine for pop=100, quadratic
  pressure for pop=1000.
- Text serialization (`Genome save+load`, ~3.9 ms/op) uses `ostringstream` +
  `%3.8f`-style formatting per field.
- `BuildPhenotype` large (~163 us) rebuilds all neurons/connections from scratch
  per call even when topology is unchanged (weight-only updates could be
  incremental — API change, defer).

## Change log

| Date       | Commit | Change                                              | Deltas |
| ---------- | ------ | --------------------------------------------------- | ------ |
| 2026-09-15 | (base) | Baseline recorded after rename + test/bugfix commits | —     |
