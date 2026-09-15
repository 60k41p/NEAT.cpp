# MultiNEAT v2 Backport Plan — NEAT.cpp (Full Port, excl. Bindings)

Source: `MN2-diff.txt` (25,521 lines, 29 files, `src/` subtree v1 → v2).
Direction: v1 (`-`) → v2 (`+`). Line refs below are `MN2-diff.txt:<line>`.

Agreed scope:
- **Full v2 port** (all genuine improvements + new features + retunes).
- **Accept v2 behavior** (dynamics/defaults/fixtures may shift; tests updated, not weakened).
- **No bindings** (`src/Bindings.cpp`, ~1,554 lines pybind11, excluded; pure C++17 stays).

Non-goals: `Bindings.cpp`, `MultiNEATAssert.h` compat shim, `Assert.h` Python/`abort` model,
`Main.cpp` as `src/Main.cpp` (demo only — reuse pattern in tests).

General constraints (per `AGENTS.md`):
- C++17, Google style + `.clang-format` (`clang-format==22.1.8`), `foo_` / `camelCase` / `PascalCase`,
  explicit types over `auto`, LF endings, `#pragma once` (no `#ifndef` guards in this fork).
- License headers: modified upstream files keep `Copyright (C) 2012 Peter Chervenski` +
  `Modifications Copyright (C) 2026 Gökalp Özcan` + prominent modification notice; fork-original
  files (`AssertMacros.h`, `Traits.h`, `Traits.cpp`) keep 2026-only header.
- Headers are documentation (purpose + per-member docs + citations); `.cpp` comments only when
  non-obvious; no manual wrap before 200 chars.
- Sources auto-discovered (`file(GLOB CONFIGURE_DEPENDS)`): new `SpikingLearning.*`,
  `Serialization.h`, `FileIO.h` need no `CMakeLists.txt` edit.
- Tests: `tests/TestX.cpp` with `int TestX(int, char**)` + `// CTest-Labels:` /
  `// CTest-Timeout:` directives, local `CHECK`, print `Test passed`, `PASS_REGULAR_EXPRESSION`;
  data via `NEATCPP_TEST_DATA_DIR` (`tests/data/`); mirror `src/` structure; never simplify a valid
  test to make it pass.
- Validation matrix: `build/debug`, `build/dbgassert` (`-DDEBUG` for `ASSERT()`),
  `build/san` (ASan+UBSan), `build/coverage` (`--fail-under-line 50`), `build/bench`;
  `ctest -L Unit/IO/Evolution/Compliance`; CI names in `.github/workflows/ci.yml`.
- Gotchas preserved unless v2 explicitly replaces them: ES-HyperNEAT `#if 0` is lifted here
  (v2 fully implements it); old `MutateGenome #if 0` in `Species.cpp` stays dead.

Execution order: 0 → 1 → 2 → 3 → 5 (params before loop) → 6 → 4 (spiking/ES-HyperNEAT last,
biggest surface) → 7. Each phase builds + tests before next.

---

## 0. Prep / baseline (no behavior change)

1. Branch `feat/mn2-port`.
2. Record baselines:
   ```bash
   cmake -S . -B build/debug -DCMAKE_BUILD_TYPE=Debug -DNEATCPP_ENABLE_TESTS=ON
   cmake --build build/debug
   ctest --test-dir build/debug --output-on-failure
   ```
   Plus `build/bench` numbers for `benchmarks/RESULTS.md` and copy of `tests/data/minimal.NEAT`.
3. Migration policy (breaking, accepted): `.NEAT` moves to `GenomeFormat 1-4` with new sections
   (`GenomeState/Traits`, `x/y`, spiking blocks). Old files need converter + fixture migration;
   document in `README.md` (Parameters section + serialization compat note).
4. Add tracking todos per phase below; keep exactly one `in_progress`.

---

## 1. Infra — `Assert.h:1-120`, `Random.*:20502-20842`, `Utils.*:25257-25521`, `FileIO.h:1680-1715`, `Serialization.h:20842-21174`, `Main.cpp:10850-11368`

### 1.1 `ASSERT` / `VERIFY` (`src/AssertMacros.h:106-109`)
- Bug: release `VERIFY(expr)` expands to nothing — side effects silently dropped.
- Backport v2 `75-103` semantics only: `NDEBUG → VERIFY = (void)expr`, debug → `assert(expr)`;
  adapted to this fork as `DEBUG`-gated `VERIFY` that always evaluates `expr` even in release.
- Do NOT import: `MULTINEAT_LEGACY_ASSERT_H` guard, `<assert.h>` shadowing shim, `abort()` model.
  Keep throw-`runtime_error` `ASSERT` so catch sites behave identically debug/release.

### 1.2 `Random.{h,cpp}`
- `RandInt` throw on `x > y` (`20551-57`); current `randInt` has no validation (UB in
  `uniform_int` when `min > max`).
- `Roulette(const&)` (`20640-735`): empty/finite/negative throws, max-scaled anti-overflow sum,
  all-zero → uniform, `lastNonZero` fallback. Current delegates to `discrete_distribution`
  with different empty/NaN/negative/all-zero semantics — port checks deliberately.
- New `RandNormal` / `RandCauchy` + `finite` / `scale > 0` checks, Cauchy non-finite resample
  (`20594-631`) — needed by `GAUSSIAN/CAUCHY/POLYNOMIAL` weight mutation.
- New engine `Serialize` / `Deserialize` (`20739-61`) — needed for checkpointing (Phase 6).
- Already done (verify, no action): `mt19937`, `RandFloat [0,1)`.
- Do NOT silently change: `RandFloatSigned = 2U-1` vs current `U1-U2` triangular, ctor auto-seed
  `RNG(){TimeSeed();}` — adopt explicitly as v2-behavior with reproducibility test, not as
  drive-by.

### 1.3 `Utils.{h,cpp}`
- `scale(vector)`: add empty-guard + `reserve` (`25266-73`); `[tr_min,max]` honor already fixed.
- `getMaxMin`: add empty → `0,0` guard (`25307-14`); `lowest()/max()` fix already present.
- `rounded → lround` (`25416`); current truncates, wrong for negatives (`-1.6 → -1` vs `-2`).
- Scalar `scale` degenerate guards: `tr_min == tr_max → snap`, `|max-min| < eps → midpoint`
  (`25437-500`); current divides by zero → `inf/NaN`.
- Skip: `Scale` default args, `RoundUnderOffset` ternary, `Clamp` whitespace, `itos/ftos`
  comment strip (current `floatToString` with `max_digits10` is superset).

### 1.4 New `FileIO.h` + `Serialization.h`
- `NEAT::detail::OpenFile` (`1680-1715`): null guard + `fopen_s` on `_MSC_VER` else `fopen`.
- `Serialization.h` (`20842-174`): `RequireStream`, `Write/ReadTraitValue(s)/Traits/
  TraitParameters`, `ReadNumericVector`, `UseRoundTripPrecision = max_digits10`,
  `quoted` + `variant::index` markers. Hand-rolled `Save/Load` stays, but uses these helpers.

### 1.5 `Main.cpp` — do not add `src/Main.cpp`; reuse XOR/smoke/CLI
(`--spiking/--mcculloch-pitts/--smoke/--generations/--population/--seed`, `Epoch()` loop,
JSON summary) as test patterns only.

Tests: `TestRandom` (bounds throw, roulette edge cases, distribution smoke),
`TestUtils` (empty scale/getMaxMin, negative rounding, degenerate scale),
`TestAssertMacros` (release `VERIFY` evaluates).

---

## 2. Traits + Genes — `Traits.h:24933-25251`, `Genes.h:1715-2871`

### 2.1 `Traits.h`
- Default-init `intsetelement::value = 0`, `floatsetelement::value = 0.0` (`24960,24990`);
  current leaves POD uninit on default-construct.
- Add `Trait::operator== / !=` (`25232-50`); no current equivalent.
- Skip: `#ifndef → #pragma once`, `Real` vs `double` (intentional abstraction).

### 2.2 `Gene::InitTraits` (`1832-1907` vs `Genes.h:113-150`)
- `if → else-if` chain + `else throw invalid_argument(unknown trait type)`; current silently
  ignores unknown `type`.
- `min > max` check for `int`/`float`; add `<string>/<stdexcept>/<variant>` includes explicitly.

### 2.3 `Gene::MateTraits` (`1926-2018` vs `Genes.h:154-199`)
- `find` guard + `continue` if key missing; current `traits_[key]` inserts default entry
  (map bloat + bogus distance).
- Overflow-safe int average `(long long m1+m2)/2`; current `(m1+m2)/2` overflows.

### 2.4 `Gene::MutateTraits` (`2032-2300` vs `Genes.h:203-315`) — critical fix
- `find` guard + range-`for`, `!empty()` + `.at()`; current `operator[]` inserts.
- `min > max` validation.
- Guaranteed-progress: replace-exclusion `RandInt(min,max-1)+bump`, 32-attempt modify + `Clamp`
  + `nextafter`/step fallback, alternatives-filter for `str/intset/floatset`. Current
  `while(cur == val)` loops forever on `min == max`, singleton set, `mutPower == 0`, or clamp.

### 2.5 `Gene::GetTraitDistances` (`2308-2417` vs `Genes.h:318-376`)
- Add `const` overload + non-`const` forwarder; current only non-`const`, blocks `const Gene`.
- `find` guards for `mine` and `mine_dep/other_dep` (no `other.at` throw on asymmetric maps).

### 2.6 `LinkGene` (`2452-2613`) / `NeuronGene` (`2686-2866`)
- Add spiking fields + defaults (`m_SynapticDelay/TimeConstant/STDP*`,
  `m_SpikeThreshold/Reset/Resting/Refractory/Resistance/Adaptation*/Rate/IzhikevichA-D/
  MCPInhibitoryVeto` + `InitSpikingDefaults()`); zero-init default ctor (current `NeuronGene(){}` leaves POD uninit).
- Adopt v2 `operator==` full-field + `operator= =default` as breaking change (alters lookup/
  mating identity; `<,>,!=` must be reconciled); note `INPUT/BIAS` field-skip semantics change
  needs test coverage.

Tests: `TestGenes` (unknown trait throw, asymmetric mate/distance, singleton/zero-power
mutation termination, overflow average, const distance).

---

## 3. Genome core — `Genome.h:10068-10232`, `Genome.cpp:2871-10068`

### 3.1 Construction / seed (`3441-3834`)
- Validate `usable_inputs`, `FS_NEAT_links <= in*out`, `NumHidden/Layers >= 0`, unknown `SeedType`,
  forbid `FS+LAYERED`.
- Fix FS-NEAT: dedup `used` set, `DontUseBias`-aware, correct `linkcount` (old per-output loop
  double-counted bias links, linear `made_already` scan could duplicate/overrun,
  `FS_NEAT_links == 1` throw workaround).

### 3.2 Accessors (`3142-3181`, `4243-4400`)
- `RemoveLinkGene(innovID)` via `find_if`; `RemoveNeuronGene` via `remove_if`+erase.
  Fixes old `RemoveLinkGene(idx)` clearing all if `idx == 0` + `O(n²)` erase loop hitting it.
- `GetLastNeuron/InnovID` returns `max` (verify off-by-one at `Genome.cpp:554,567`);
  `GetNeuron/LinkByID` → `out_of_range` with message (was `ASSERT + throw exception()`).
- `SetDepth` range check; `SetNeuronXY` via `.at()` (`4127,4152`).

### 3.3 `IsDeadEnd/Cleanup/HasDeadEnds` (`4158,8608,8655`)
- Cache `bias_id`, skip `IsLoopedRecurrent`, `while(i)+continue` erase (fixes indexed-`for`
  skipping next neuron); old `IsDeadEnd` checked `FromType != BIAS` for both directions.

### 3.4 `HasLoops/BuildPhenotype/DerivePhenotypicChanges` (`4336,4404,5623`)
- Kahn indegree DAG, no `boost`, `true` on dangling endpoint.
- `BuildPhenotype`: `unordered_map` + reserve (was `GetNeuronIndex` per link `O(n·m)`),
  dangling throw (was silent `-1`), copy spiking fields, `hebb_rate` via `std::get`.
- `Derive`: size-check early return (was silent miscopy).

### 3.5 `BuildHyperNEAT` (`4565-4835`) — partial
- Empty-substrate / `ValidateSpatialSubstrate` throw, `max_weight/time_const` finite checks,
  `CalculateDepth`-scaled `cppn_activation_steps` (was fixed `8`), custom-connectivity
  `size == 4` + bounds validation. Keep `SetInputOutputDimensions` alias; do not remove typo
  `Dimentions` (compat).

### 3.6 `BuildESHyperNEAT` (`4835-5621`) — full new feature
- Port `TreeNode`, `sample_connections`, variance/band/LEO pruning, reachability prune,
  `FinalizeSpatialConnections`; lift `#if 0` guard.

### 3.7 `CompatibilityDistance/IsCompatible` (`5636-6608`) — partial
- Unsorted-safe (sort copies if needed), `maxsize < 1 → 1`, `M < 1 → 1`, normalizer guard,
  NaN/Inf → 0, sorted-merge / `unordered_map` neuron compare; `E` only at end, interleaved → `D`
  (old lumped all tails as `E`).
- Add `SpikingLink/NeuronDiff`, genome-trait weighting; drop `this == &g || ID ==` shortcut.

### 3.8 `Mutate_LinkWeights` (`6620-7393`)
- `UNIFORM/GAUSSIAN/CAUCHY/POLYNOMIAL` (`WeightMutationDistribution/Sigma/Cauchy/PolyEta`),
  `tailstart = 0.9N`, `WeightReplacementRate` logic, `Clamp`, `did_mutate` only on change.
  Old ignored distribution, dead `ontail`, severe = uniform resample.

### 3.9 `Mutate_AddNeuron/AddLink` (`8009-8406`) — high value
- `AddNeuron`: `eligible_links` filter (skip `BIAS` source, `SplitRecurrent/LoopedRecurrent`),
  `NeuronTries` guard, single pick (was 256 random tries, bias-split spin); spiking delay
  halving only if spiking enabled.
- `AddLink`: enumerate `candidates`, DFS `feedforward_adjacency + reachable_from_target` DAG
  guarantee, bias isolation, `endpoint_key` dedup, sorted for reproducibility. Old had no cycle
  check and allowed `OUTPUT`-source.

### 3.10 `RemoveLink/RemoveSimpleNeuron` (`8406-8470`)
- `RandInt(0,N-1)` uniform (was biased `RandFloat*(N-1)+Clamp`); missing → `false`
  (was `ASSERT` crash); spiking delay-sum / tau-avg.

### 3.11 `Mate/MateWithMode` (`8721-9466`) — partial
- Sort copies, I/O count mismatch throw, I/O `ID/Type` check, `endpoint_key` dedup
  (was `HasLink O(n²)`), preserve `initial_num_*`, unified `add_child_neuron`.
- Add `BLEND/Alpha`, `SBX/Eta`, `SINGLE_POINT`; `Mate → MateWithMode(MULTIPOINT/AVERAGE)` wrapper.

### 3.12 `Depth/CalculateDepth` (`9481-9637`)
- Iterative Kahn skipping recurrent, non-recurrent-cycle throw, `max(1U,maximum)`.
  Verify vs current `Genome.cpp:2777`.

### 3.13 Persistence (`9640-, 4052-, 3264-, 9907-`) — breaking, accepted
- `GenomeFormat 1-4`, `GenomeState/Traits`, `x/y`, `NeuronSpiking/LinkSpiking`,
  `Read/WriteTraits + RoundTripPrecision`, stream ctor, `Validate()` (unique IDs, I/O order,
  finite, spiking ranges), `Serialize/Deserialize`, `IsIdenticalTo` update.
- Migrate `tests/data/minimal.NEAT`; add round-trip + old-format-reject/migrate tests.
- `GetRandomActivation` prob validation; `FailsConstraints: CustomConstraints →
  FailsCustomConstraints` rename adopted.

Tests: `TestGenome` (seed validation, remove/last-ID, cleanup, loops/phenotype/derive,
compat edge cases, add neuron/link invariants, remove-uniformity, mate modes, depth,
validate/serialize round-trip).

---

## 4. Phenotype + spiking + substrate

### 4.1 `NeuralNetwork.{h,cpp}` (`14474-14923`, `11382-14474`)
Fixes (cherry-pick first):
- `tanh` drops `aShift` (`11424`; `src/NeuralNetwork.cpp:61` still `tanh(x*slope+shift)`-adjacent bug).
- `softplus` stable `max + log1p(exp(-abs))` (`11743`; `:104` is `log(1+exp(x))` overflow);
  same for `tanh_cubic+shift` (`11556`).
- RTRL: snapshot `m_source_activation` (`12142`), retain `m_last_input` (`11898`), full
  `activation_derivative()` (`11782-864`), index `connection_indices/incoming` once
  (`13411-435`), `Validate + auto-Init` (`13379-403`). Current `:541-577` is old `O(N⁴)`
  with live `m_activation` source and only 2 derivatives. Add `InitSparseRTRLMatrix` (`12057`).
- Geometry: `sqrt(dx²+dy²+dz²)` over `x/y/z` (`13255-261`) + true sum (`13263-276`); old squared
  dist + `GetTotalConnectionLength → size()` (`14654-706`). Keep current names, add `Length`
  alias like v2 (`14888`).
- `Flush` clears `signal/source_activation/synaptic_current/traces/pending/spike/refractory/
  adaptation/time/history` (`12265-300`); at minimum `signal_ = 0` (`:465` gap).
- `Adapt` magnitude form `max(0,mag+delta)` + `Clamp` + range check (`13333-364`);
  old `-(w+delta)` mishandles negatives (`:518-524`).
- `InputExact` size check (`12337`), leaky `dt finite / timeconst > 0` (`12216-239`),
  `ValidateNetworkTopology` (`11441`); hot `ActivateFast` stays unchecked.

Features (deferred to end of phase, opt-in cost):
- Spiking runtime (`12734-230,14720-788`): `SPIKING_LIF/ADAPTIVE/IZHIKEVICH/MCCULLOCH_PITTS`,
  `SpikingInputMode CURRENT/BINARY/POISSON`, `OutputMode SPIKE/RATE/FILTERED/MEMBRANE`
  (`14511-573`), `StepSpiking/SimulateSpiking/Output*/IsSpiking/SeedSpiking` (`14828-864`),
  thresholds/refractory/adaptation/Izhikevich state, 100k spike history, delays/currents.
- STDP per-connection (`14676-684,13173`): traces, `tau±`, `EnableSTDP` (`12641`).
- Delay/geometry: `m_synaptic_delay/time_constant/m_length` (`14671-688`),
  `UpdateConnectionGeometry` (`13278`), `Substrate:max_connection_length/use_spatial_distance/
  conduction_velocity` (`24735,24911`) + pruning hooks (`3085-135,5483`).
- Helpers: `ActivateSteps/ActivateBatch` (`12387,12418`), string `Serialize/Deserialize`
  (`14918`), multi-target `RTRL_update_error[_sparse]` (`14803-815`), `MCCULLOCH_PITTS` rate
  neuron (`11957`).

### 4.2 New `SpikingLearning.{h,cpp}` (`23238-24721`, ~1.5k lines)
- `EPropConfig/State/Step/SequenceResult` (`24569-623`), `AdamW/SGD`,
  `RANDOM/SYMMETRIC/UNIFORM` feedback, `FAST_SIGMOID/TRIANGULAR/ARCTAN` surrogates,
  `MSE/HUBER`, eligibility traces, `Serialize`. No current counterpart — add as new files
  (auto-discovered) with `TestSpikingLearning`.

### 4.3 `Substrate.{h,cpp}` (`24721-24933`)
- `SetCustomConnectivity` `size == 4` / type / index validation (`24796-847`; `:87` blind copy).
- `GetMaxDims const` + `IsThreeDimensional` (`24854-890`); init-lists; 3D octree fields.

### 4.4 `PhenotypeBehavior.h` (`17686-706`)
- Unnamed params (`Acquire/Distance_To`) — warning silence only.

Perf guard: fixes cost ~0; spiking adds `~100B/neuron`, per-step `exp`, unbounded history —
keep off-path (`IsSpiking` branch) and update `benchmarks/RESULTS.md` + `save/load 4.4ms` bench.

---

## 5. Parameters — `Parameters.h:17314-17686`, `Parameters.cpp:14923-17314`

### 5.1 Structure
- `MULTINEAT_PARAMETER_FIELDS(X)` + `Write/ReadParameters`, `Serialize/Deserialize`,
  `Validate(error*)`, `ConfigureSpiking(enable_stdp)` / `ConfigureMcCullochPitts(veto,stdp)`,
  `Set/Get/FailsCustomConstraintsFunction` (`std::function` + legacy pointer).
- `Save`: round-trip precision + X-macro loop, null/close checks; `Load`: `Reset()` first,
  `unordered_map<string,reader>`, `Elitism` alias → `EliteFraction`, strict `true/false`,
  skip unknown keys, throw on bad value.
- `Validate()`: `PopulationSize/Min/MaxSpecies`, tries, `MaxLinks/Neurons`,
  `Initial ≤ MaxDepth ≤ 9`, dims `> 0`, enum ranges (7 enums), `MinSpeciesSize/SpeciesElitism/
  TargetSpecies` vs `PopulationSize`, probs `∈ [0,1]`, crossover-sum `≤ 1`, ordered finite
  ranges, spiking positivity, `OperatorsPerOffspring ∈ [1,256]` + `*MaxFactor ≤ 1024`,
  `RankPressure ∈ [1,2]`, activation-sum `> 0`.

### 5.2 Defaults retune (adopt wholesale — accepted break)
`Young 5→15`, `Old 30/0.50→80/0.75`, `PreferFitter 0.25→0.50`, `Tournament true→false`,
`+TruncationSelection=true`, `EliteFraction 1e-6→1e-4`, `BiasProb 0→0.01`,
`Recur/Loop 0.25/0.25→0.20/0.50`, `Weights 0.90/0.25/1.0→0.80/0.20/0.80`,
`MaxPower 1.0/1.0→1.5/3.0`, `Min/MaxA 1.0→4.9`, `WeightDiff 0.5→0.1`,
`MinCompat/Modifier 0.0/0.1→0.1/0.2`, +`Depth/Qtree_Z`.

### 5.3 Enums + knobs (all consumed — port all)
- `SelectionMode: LEGACY(-1),TRUNCATION,ROULETTE,RANK_LINEAR,RANK_EXP,TOURNAMENT,STOCHASTIC,BOLTZMANN`
- `CrossoverMode: MULTIPOINT,AVERAGE,SINGLE_POINT,BLEND,SIMULATED_BINARY`
- `WeightMutationMode: UNIFORM,GAUSSIAN,CAUCHY,POLYNOMIAL`
- `SpeciesRepresentativeMode: FIRST,RANDOM,BEST,MEDOID`
- `OffspringAllocationMode: LARGEST/STOCHASTIC_REMAINDER`
- `CompatibilityThresholdMode: LEGACY/PROPORTIONAL`
- `FitnessScalingMode: SHIFTED,LINEAR_RANK,SIGMA,BOLTZMANN`
- Consumed knobs: `TruncationSelection`, `ParentSelectionMode+RankPressure/Exponent/
  BoltzmannTemp`, `Single/Blend/SBXRate+Alpha+Eta`, `WeightMutationDistribution/Sigma/Cauchy/
  PolyEta`, `RepresentativeSelection+Candidates`, `MinSpeciesSize/SpeciesElitism`,
  `StagnationPenalty`, `EliteFraction` honored, `ThresholdControl+Target/Gain/MaxCompat`,
  `RequireEvaluated/RejectNonFinite` (default `false`, opt-in), `OperatorsPerOffspring+
  Adaptive*`, `FitnessScaling*`, spiking/MCP/STDP ranges + 4 act probs (needs Phase 4 stack).
- Still dead (keep `README` warning): `Depth/Qtree_Z` for 2D loop, `DetectCompetitive/KillWorst*`.
- `ConfigureSpiking`: zero rates, `LIF/ALIF/Izhi = 0.65/0.25/0.10`, `MutActType = 0.05`,
  `MutSpikeN/L = 0.25/0.15`, `Recurrent ≥ 0.2`, loops, `DiffCoeff = 0.1`.
  `ConfigureMcCullochPitts`: calls above, `MCP = 1.0`, veto init/mut.

Tests: `TestParameters` (defaults snapshot, load alias/unknown-skip/strict-bool/bad-throw,
`Validate()` table, `Configure*` smoke, serialize round-trip).

---

## 6. Evolution loop — `Population.*:20218-502,17706-20218`, `Species.*:23105-238,21174-23105`, `Innovation.*:10782-850,10232-850`

### 6.1 Speciation / representative
- `ChooseRepresentativeIndex{FIRST,BEST,RANDOM,MEDOID}` + `RepresentativeSelectionCandidates`
  (`17737-803,18994-9003`); default `FIRST` reproduces old behavior; `MEDOID O(n²)` + RNG use noted.
- `Speciation == false` single-species bypass (`18388-397`).
- `ReassignSpecies` rewrite + `remove_if` empty-species (`19336-417,19646-662`); fixes erase-skip.

### 6.2 Offspring allocation (`18752-9009`)
- Fractional `requirements → quotas → floors → floor+remainder`, `MinSpeciesSize/SpeciesElitism`
  floors, `LARGEST_REMAINDER` (`stable_sort`), excess trim preserving floors,
  `throw` if `total != PopulationSize`. Replaces bonus clones of `Species[0]` leader.
- `STOCHASTIC_REMAINDER` via `Roulette(remainders)` (`18910-928`); default `LARGEST`.

### 6.3 Fitness scaling / sharing (`17805-844,18422-477,21864-998`)
- `TransformFitnessValues{SHIFTED,LINEAR_RANK,SIGMA,BOLTZMANN}` population-wide,
  `AdjustFitness(offset/vector)` overloads, `max_adjusted` norm to `1.0`, `eps = 1e-12`.
  Old: per-species `+1e-7`, `ASSERT(fitness >= 0)`, `adjusted/size`. Adopt even though default-on
  shifts dynamics (accepted).
- Separable fixes inside: `StagnationDelta`-gated `gensNoImprovement` reset + `bestGenome`
  update (`21943-956`), `StagnationPenalty` param (`21975`), clamp overflow, non-finite → `1e-12`,
  `CalculateAverageFitness` scaled sum + count-only-evaluated (`22319-378`).

### 6.4 Selection / crossover (`21210-759,21255-273,22174-181,22524-530,21675-694`)
- `ParentSelectionMode{...}` + `NormalizeSelectionWeights`, `AdjFitness` basis (was `Fitness`),
  none-evaluated throw (was random fallback); pure tournament picks best (`sort → [0]`).
- `SelectCrossoverMode{...}` replaces `RandFloat < Multipoint ? Mate(false) : Mate(true)`.
  Note extra `RandFloat` draw alters stream — gate carefully, update determinism tests.
- Interspecies mate + `ChooseParentSpecies`: shift negatives, uniform fallback, leader
  `AdjFitness` (`22094-154,22430-493,19284-333`); fixes negative-fitness `Roulette` hang.

### 6.5 Compat-threshold adaptation (`18684-734,19520-570`)
- `PROPORTIONAL exp(gain*err)` to `TargetSpecies` + `clamp(Min,Max)` + `interval > 0` guard.

### 6.6 Elitism (`22024-065`)
- Honor `EliteFraction` (`Rounded(frac*size)`, capped), copy distinct `Individuals[elite_count]`
  (was `GetLeader()` N×, always 1). Accepted behavior change; default `0.0001` is near-disabled.

### 6.7 Innovation indexing (`10300-340,10563-726,10814-840`)
- `EndpointKey → unordered_map` + `EnsureIndex/RebuildIndex/AppendToIndex` (perf fix);
  public `m_Innovations` edits require manual `RebuildIndex`.
- `ctor/Init(last) → last+1` off-by-one fix, `ValidateInnovationState`, strict I/O + `Serialize`,
  `.at()`, overflow guards; needs `minimal.NEAT` counter-migration test.

### 6.8 Determinism / RNG (`21197,21325-352,17772-779,20037-061`)
- Drop `global_rng`, ID-hash color, Fisher-Yates medoid sampling on pop `RNG`,
  `RNG.Serialize` in checkpoint.

### 6.9 Epoch-loop hardening
- `lowest()` not `min()`, finite/`IsEvaluated` guards, live `NumGenomes()`,
  `bestFitnessEver = lowest()`, evaluated-first `GetBestGenome/GetLeader` + empty throw,
  `SameGenomeIDCheck → .cpp` + ID-overflow throws, `out_of_range` accessors, `RemoveWorst`
  offset shift, `Tick` skip-unevaluated, pre-clone `SortGenes`,
  `GenomesAreClones(IsIdenticalTo || distance)`, `MutateGenome` 512-bound + `MaxNeurons/Links`
  caps, `Validate/Serialize/Deserialize`. `RequireEvaluated/RejectNonFinite` strict epoch is
  opt-in (`false` default). Leave `#if 0` old `MutateGenome` path alone.

Tests: `TestPopulation/TestSpecies/TestInnovation` (representative modes, allocation exact-sum,
scaling modes, selection/crossover modes, proportional threshold, elite-fraction, index rebuild,
determinism seed, epoch guards). Labeled `Evolution`.

---

## 7. Validation, docs, rollout

1. Per phase + final:
   ```bash
   cmake -S . -B build/debug -DCMAKE_BUILD_TYPE=Debug -DNEATCPP_ENABLE_TESTS=ON
   cmake --build build/debug
   ctest --test-dir build/debug --output-on-failure
   ctest --test-dir build/debug -R TestGenome
   ctest --test-dir build/debug -L Evolution
   ```
   Plus `build/dbgassert`, `build/san`, `build/coverage`, `build/bench` per `AGENTS.md` table.
2. Coverage: add/extend tests for every new branch; keep `--fail-under-line 50` green.
3. `clang-format -i src/*.h src/*.cpp tests/*.cpp benchmarks/*.cpp` (v22.1.8).
4. Update `README.md` (new knobs table, dead-knob warnings retained for competitive coevolution),
   `benchmarks/RESULTS.md` (spiking/EProp/RTRL/serialization numbers), `tests/data/minimal.NEAT`
   + compat tests.
5. Final `git status / diff / log --oneline -10` review; commit only on request.

## Risks / watchlist
- Serialization break + counter off-by-one: gate with migration test on `minimal.NEAT`.
- Dynamics shift: retunes + allocation + scaling + elitism change trajectories by design —
  update `Evolution`/`Compliance` expectations, do not weaken tests.
- RNG stream: new draws (crossover select, medoid, stochastic remainder) break bit-repro;
  pin seeds per test.
- Perf: `MEDOID O(n²)`, spike history 100k, per-step `exp`, `UpdateConnectionGeometry` —
  keep behind flags; verify `Activate 6µs/1k-conn` baseline does not regress when features off.
- API: `Dimentions` typo kept as alias; `operator==` identity change is the subtlest break —
  audit all `==`/lookup call sites.
