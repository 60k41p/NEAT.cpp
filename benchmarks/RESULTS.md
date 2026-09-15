# Benchmarks

Reproducible timing baselines for NEAT.cpp, used to validate that optimizations are real and to catch regressions.

## How to run

```bash
cmake -S . -B build/bench -DCMAKE_BUILD_TYPE=Release -DNEATCPP_ENABLE_TESTS=OFF -DNEATCPP_ENABLE_BENCHMARKS=ON
cmake --build build/bench -j
./build/bench/benchmarks/NEATcppBench
```

The harness (`benchmarks/Benchmarks.cpp`) uses fixed seeds and fixed iteration counts, so output is comparable across runs and changes. Values below are ns/op unless noted; machine: Apple clang 17.0.0, Intel i5 @ 2.30 GHz (8 cores), macOS 15.7.3. Baseline commit: `aa31413` (uplift-2).

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

Run-to-run spread observed on this machine: <10% on all benchmarks except Population save+load (~15%, disk noise).

## Observations / optimisation candidates

- `Copy+mutate genome` (~82 us) dominates reproduction: every mating and every species copy round-trips full `std::vector`s of genes (and `NeuralNetwork`'s per-neuron sensitivity matrix is part of `Neuron`, copied with genomes? — verify). Any per-epoch cost scales with this.
- `CompatibilityDistance` (~48 us) is O(links) per pair but called for every individual per epoch with `DynamicCompatibility`; fine for pop=100, quadratic pressure for pop=1000.
- Text serialization (`Genome save+load`, ~3.9 ms/op) uses `ostringstream` + `%3.8f`-style formatting per field.
- `BuildPhenotype` large (~163 us) rebuilds all neurons/connections from scratch per call even when topology is unchanged (weight-only updates could be incremental — API change, defer).

## MultiNEAT v2 port (2026-09-16, `feat/mn2-port`)

Release config (`build/bench`), same machine. Workloads are seed-identical to
prior runs, but generation dynamics changed (v2 defaults + operators), so the
large-genome fixtures differ slightly in size from earlier tables.

| Benchmark                    | ops   | ns/op       | vs uplift-2   | notes |
| ---------------------------- | ----- | ----------- | ------------- | ----- |
| BuildPhenotype small         | 200k  | 1,021       | +69%          | spiking field copies + dangling-endpoint throw |
| BuildPhenotype large         | 2k    | 40,694      | +124%         | same; fixture also differs |
| Activate large net           | 20k   | 14,341      | +238%         | topology validation + `IsSpiking` + dispatched activation per call |
| CompatibilityDistance        | 20k   | 86,543      | +706%         | sort-safety check + spiking diffs; fixture differs |
| Copy+mutate genome           | 50k   | 1,207,137   | **+1486%**    | see note below |
| Epoch pop100                 | 100   | 1,433,705   | +67%          | quotas, transforms, sorting, exact-size accounting |
| XOR solve seed1              | 1     | 83.5 ms     | faster; gen 21 (was 31) | dynamics shifted, still solves |
| Genome save+load             | 100   | 15,960,547  | +307%         | +`NeuronSpiking`/`LinkSpiking` lines per gene |
| Population save+load         | 20    | 55,441,539  | +336%         | same reason |

### Copy+mutate note (deliberate trade-off)

`Mutate_AddLink` now guarantees the new forward link can never close a
directed cycle (DFS/bitset reachability guard) and samples uniformly over all
admissible pairs. Cost per call on a dense ~200-neuron/~1k-link genome:

- bounded random tries (`LinkTries`, fast path, wins when free pairs are common),
- exhaustive bitset-DP + reservoir-sampling fallback (no candidate storage/sort)
  only when sampling fails.

The 50k-op stress test hammers the dense worst case back-to-back (fallback on
most calls); real evolution calls `AddLink` on small genomes where the fast
path decides in microseconds — `Epoch pop100` (+67%) is the representative
metric. A segfault found by this benchmark (stale merge iterator in
`CompatibilityDistance`) was fixed and is covered by ASan in CI configs.

## Change log

| Date | Commit | Change | Deltas |
| ---------- | ------ | --------------------------------------------------- | ------ |
| 2026-09-15 | (base) | Baseline recorded after rename + test/bugfix commits | — |
| 2026-09-15 | optimisation | Algorithmic fixes: BuildPhenotype ID-index table, CompatibilityDistance neuron-lookup hoist, HasLoops O(V+E) Kahn, IsDeadEndNeuron type table. XOR solve generations unchanged (30/16/35/41/30) — bit-identical trajectories. | see table below |

## Post-optimisation (2026-09-15)

| Benchmark                    | ops   | ns/op     | vs baseline |
| ---------------------------- | ----- | --------- | ----------- |
| BuildPhenotype small         | 200k  | 605       | -17%        |
| BuildPhenotype large         | 2k    | 18,202    | **-89%**    |
| Activate large net           | 20k   | 4,239     | ~same       |
| CompatibilityDistance        | 20k   | 10,728    | **-78%**    |
| Copy+mutate genome           | 50k   | 76,132    | -7%         |
| Epoch pop100                 | 100   | 857,885   | ~same (noise band 85–113) |
| XOR solve seed1              | 1     | 136.4 ms  | ~same; still gen 31 |
| Genome save+load             | 100   | 3,715,181 | ~same       |
| Population save+load         | 20    | 12,802,737| ~same       |

Rejected: replacing `Mutate_AddLink`'s per-try `HasLink` linear scan with an `unordered_set` of existing (from,to) pairs — building the set per call cost ~2× more than the cache-friendly scans it replaced (Copy+mutate 82→187 µs).
