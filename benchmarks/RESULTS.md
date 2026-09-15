# Benchmarks

Reproducible timing baselines for NEAT.cpp, used to validate that optimizations are real and to catch regressions.

## How to run

```bash
cmake -S . -B build/bench -DCMAKE_BUILD_TYPE=Release -DNEATCPP_ENABLE_TESTS=OFF -DNEATCPP_ENABLE_BENCHMARKS=ON
cmake --build build/bench -j
./build/bench/benchmarks/NEATcppBench
```

The harness (`benchmarks/Benchmarks.cpp`) uses fixed seeds and fixed iteration counts, so output is comparable across runs and changes. Timing is calling-thread CPU time (user + system, via `CLOCK_THREAD_CPUTIME_ID` / Mach `thread_info` / `GetThreadTimes` depending on platform): time the thread is descheduled or blocked on I/O is excluded. Values below are ns/op unless noted; machine: Homebrew clang 22.1.7, Intel i5-8259U @ 2.30 GHz, macOS 15.7.3. Baseline commit: `9cdeb9d` (uplift-3) plus the thread-CPU harness change.

## Baseline (2026-09-15, thread CPU time)

Medians of 3 runs:

| Benchmark                    | ops   | ns/op     | notes                                          |
| ---------------------------- | ----- | --------- | ---------------------------------------------- |
| BuildPhenotype small         | 200k  | 890       | 10-in/3-out perceptron (13 links)              |
| BuildPhenotype large         | 2k    | 24,870    | complexified genome (~213 neurons, ~1k links)  |
| Activate large net           | 20k   | 6,058     | one Flush+Input+Activate over ~1k connections  |
| CompatibilityDistance        | 20k   | 14,903    | two large genomes (~1k links each)             |
| Copy+mutate genome           | 50k   | 103,638   | genome copy + Mutate_LinkWeights + AddLink     |
| Epoch pop100                 | 100   | 1,174,720 | full generation on pop=100 XOR-style topology  |
| XOR solve seed1              | 1     | 172.7 ms  | reference recipe; solved at generation 31      |
| Genome save+load             | 100   | 4,381,060 | large genome round-trip through text file (CPU only, excludes I/O wait) |
| Population save+load         | 20    | 15,362,050| pop=100 full save/load round-trip (CPU only, excludes I/O wait) |

Run-to-run spread observed on this machine: mostly <10%, up to ~15% on Activate and the save/load benchmarks.

## Float A/B (Real = float, 2026-09-15)

Same-machine, same-harness comparison of `HEAD` (`double`) vs the `Real = float` conversion. Each cell is the median of 3 runs, taken back-to-back on this machine (AppleClang, macOS). Deltas within ±~10% are inside the observed run-to-run noise band.

| Benchmark                    | double (HEAD) | float (Real) | delta      |
| ---------------------------- | ------------- | ------------ | ---------- |
| BuildPhenotype small         | 791           | 838          | +6% ~noise |
| BuildPhenotype large         | 22,598        | 21,239       | -6% ~noise |
| Activate large net           | 5,451         | 5,233        | -4% ~noise |
| CompatibilityDistance        | 13,701        | 14,418       | +5% ~noise |
| Copy+mutate genome           | 103,628       | 86,442       | **-17%**   |
| Epoch pop100                 | 1,247,280     | 1,705,910    | +37%¹      |
| XOR solve seed1              | 178.1 ms (gen 31) | 110.4 ms (gen 22) | trajectory² |
| Genome save+load             | 4,493,510     | 4,495,070    | ~same      |
| Population save+load         | 17,167,300    | 16,070,950   | -6% ~noise |

¹ Epoch is not apples-to-apples: `float` RNG draws (`uniform_real_distribution<float>`) form a different
deterministic stream than `double`, so the same seed evolves a different population. A workload probe over the
same 100-epoch recipe (seed 21) shows float carrying ~1.7x the link-work (8046 vs 4750 summed links, 12–18 vs
8–14 species). Per unit of link-work, float is ~19% faster (0.212 vs 0.263 ms per summed link). The per-op
microbenchmarks above (same size class, fixed genomes) show parity outside Copy+mutate.

² XOR generations are trajectory-dependent by the same RNG-stream argument; both solve far inside the typical
budget, and the TestEvolution suite solves all 5 reference seeds in float (Debug, Release, ASan+UBSan, ASSERT).

Memory footprint (`sizeof`, same compiler):

| Type         | double | float | delta |
| ------------ | ------ | ----- | ----- |
| Connection   | 48     | 28    | -42%  |
| Neuron       | 176    | 112   | -36%  |
| LinkGene     | 56     | 48    | -14%  |
| NeuronGene   | 88     | 64    | -27%  |
| Genome shell | 144    | 128   | -11%  |
| Parameters   | 944    | 568   | -40%  |

Conclusion: float buys clearly smaller memory (~15–40% per object; halved `vector<Real>` payloads) and a real
Copy+mutate win, with activation/compatibility/phenotype parity. It does not make NEAT compute-bound paths
dramatically faster — they are dominated by `exp/tanh/sin` and pointer-chasing, not scalar width. Seeded
trajectories intentionally differ from the double era (no bit-identical replay); stability is statistically
unchanged (5/5 XOR seeds solve, determinism per seed preserved).

## Observations / optimisation candidates

- `Copy+mutate genome` (~104 us) dominates reproduction: every mating and every species copy round-trips full `std::vector`s of genes (and `NeuralNetwork`'s per-neuron sensitivity matrix is part of `Neuron`, copied with genomes? — verify). Any per-epoch cost scales with this.
- `CompatibilityDistance` (~15 us) is O(links) per pair but called for every individual per epoch with `DynamicCompatibility`; fine for pop=100, quadratic pressure for pop=1000.
- Text serialization (`Genome save+load`, ~4.4 ms/op CPU) uses `ostringstream` + `%3.8f`-style formatting per field; wall latency is higher since blocked I/O is excluded from these numbers.
- `BuildPhenotype` large (~25 us) rebuilds all neurons/connections from scratch per call even when topology is unchanged (weight-only updates could be incremental — API change, defer).

## Change log

| Date | Commit | Change | Deltas |
| ---------- | ------ | --------------------------------------------------- | ------ |
| 2026-09-15 | (float) | Global `Real = float` conversion (new `src/Types.h` scalar alias; flip it to `double` for an exact-precision A/B) | Float A/B section above; seeded trajectories differ by design (RNG stream), stability statistically unchanged |
| 2026-09-15 | (thread-cpu) | Harness switched from wall time (`steady_clock`) to calling-thread CPU time; new baseline recorded (medians of 3 runs, XOR still gen 31) | new baseline above; old wall-time tables moved to Historical section |
| 2026-09-15 | (base) | Baseline recorded after rename + test/bugfix commits | see Historical section |
| 2026-09-15 | optimisation | Algorithmic fixes: BuildPhenotype ID-index table, CompatibilityDistance neuron-lookup hoist, HasLoops O(V+E) Kahn, IsDeadEndNeuron type table. XOR solve generations unchanged (30/16/35/41/30) — bit-identical trajectories. | see Historical section |

## Historical wall-time tables (superseded)

Recorded with `steady_clock` wall time at `aa31413` (uplift-2), Apple clang 17.0.0, same machine class. Kept for reference; not comparable with the thread-CPU baseline above.

Pre-optimisation:

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

Post-optimisation (wall):

| Benchmark                    | ops   | ns/op     | vs pre-opt |
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
