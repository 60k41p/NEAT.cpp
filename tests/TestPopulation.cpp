// Seeded evolution smoke/regression tests for NEAT::Population.
// Small populations + fixed RNG seeds keep these fast and deterministic.
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

#include "Genome.h"
#include "Parameters.h"
#include "Population.h"
#include "Random.h"

namespace {

    int g_failures = 0;

#define CHECK(cond)                                                                         \
    do {                                                                                    \
        if (!(cond)) {                                                                      \
            std::cerr << "FAILED " << __FILE__ << ":" << __LINE__ << ": " << #cond << "\n"; \
            ++g_failures;                                                                   \
        }                                                                                   \
    } while (0)

    NEAT::Parameters SmallParams() {
        NEAT::Parameters p;
        p.Reset();
        p.PopulationSize = 20;
        p.AllowClones = true;
        p.MutateAddNeuronProb = 0.1;
        p.MutateAddLinkProb = 0.2;
        p.MutateRemLinkProb = 0.02;
        p.MutateWeightsProb = 0.8;
        p.SurvivalRate = 0.3;
        return p;
    }

    NEAT::Genome MakeSeed() {
        NEAT::Parameters p;
        p.Reset();
        NEAT::GenomeInitStruct init;
        init.NumInputs = 3;
        init.NumOutputs = 1;
        init.SeedType = NEAT::PERCEPTRON;
        return NEAT::Genome(p, init);
    }

    // Trivial evaluator: reward genomes with more links (exercises fitness flow without any domain code). Deterministic given the population state.
    void EvaluateByLinkCount(NEAT::Population &pop) {
        for (unsigned i = 0; i < pop.NumGenomes(); ++i) {
            NEAT::Genome &g = pop.AccessGenomeByIndex(static_cast<int>(i));
            g.SetFitness(1.0 + static_cast<double>(g.NumLinks()));
            g.SetEvaluated();
        }
    }

    // SameGenomeIDCheck throws on duplicates; convert that into a CHECK failure
    // so a regression reports cleanly instead of terminating the driver.
    void ExpectUniqueIDs(NEAT::Population &pop, int line) {
        try {
            pop.SameGenomeIDCheck();
        } catch (const std::exception &e) {
            std::cerr << "FAILED TestPopulation.cxx:" << line << ": unique genome IDs (" << e.what() << ")\n";
            ++g_failures;
        }
    }

#define EXPECT_UNIQUE_IDS(pop) ExpectUniqueIDs(pop, __LINE__)

}  // namespace

int TestPopulation(int argc, char *argv[]) {
    (void)argc;
    (void)argv;
    using namespace NEAT;

    // Construction invariants: size, generation 0, unique IDs.
    {
        Parameters params = SmallParams();
        Genome seed = MakeSeed();
        Population pop(seed, params, true, 1.0, 42);
        CHECK(pop.NumGenomes() == params.PopulationSize);
        CHECK(pop.GetGeneration() == 0);
        EXPECT_UNIQUE_IDS(pop);
        CHECK(pop.GetNextGenomeID() >= params.PopulationSize);
    }

    // Epoch() advances generations, preserves size and ID uniqueness.
    {
        Parameters params = SmallParams();
        Genome seed = MakeSeed();
        Population pop(seed, params, true, 1.0, 42);
        for (int gen = 0; gen < 5; ++gen) {
            EvaluateByLinkCount(pop);
            pop.Epoch();
            CHECK(pop.GetGeneration() == static_cast<unsigned>(gen + 1));
            CHECK(pop.NumGenomes() == params.PopulationSize);
            EXPECT_UNIQUE_IDS(pop);
        }
        // Epoch() leaves newborns unevaluated (fitness 0), so evaluate once
        // more before asserting on best fitness.
        EvaluateByLinkCount(pop);
        pop.Sort();
        CHECK(pop.GetBestGenome().GetFitness() > 0.0);
    }

    // Fixed-seed determinism: two identical runs reach identical best fitness.
    {
        auto run = [] {
            Parameters params = SmallParams();
            Genome seed = MakeSeed();
            Population pop(seed, params, true, 1.0, 1234);
            for (int gen = 0; gen < 4; ++gen) {
                EvaluateByLinkCount(pop);
                pop.Epoch();
            }
            EvaluateByLinkCount(pop);
            pop.Sort();
            return pop.GetBestGenome().GetFitness();
        };
        const double a = run();
        const double b = run();
        CHECK(a == b);
    }

    // Tick() (steady-state) replaces one individual per call after evaluation.
    {
        Parameters params = SmallParams();
        Genome seed = MakeSeed();
        Population pop(seed, params, true, 1.0, 99);
        EvaluateByLinkCount(pop);
        const unsigned before = pop.NumGenomes();
        Genome deleted;
        Genome *baby = pop.Tick(deleted);
        CHECK(baby != nullptr);
        CHECK(pop.NumGenomes() == before);
        EXPECT_UNIQUE_IDS(pop);
        // Tick on an unevaluated population must throw, not hang.
        Population fresh(seed, params, true, 1.0, 100);
        bool threw = false;
        try {
            Genome d2;
            (void)fresh.Tick(d2);
        } catch (const std::runtime_error &) {
            threw = true;
        }
        CHECK(threw);
    }

    // Accessors: by index round-trips, out-of-range throws.
    {
        Parameters params = SmallParams();
        Genome seed = MakeSeed();
        Population pop(seed, params, true, 1.0, 5);
        Genome &g0 = pop.AccessGenomeByIndex(0);
        CHECK(pop.AccessGenomeByID(g0.GetID()).GetID() == g0.GetID());
        bool threw = false;
        try {
            (void)pop.AccessGenomeByIndex(static_cast<int>(pop.NumGenomes()) + 10);
        } catch (const std::runtime_error &) {
            threw = true;
        }
        CHECK(threw);
        threw = false;
        try {
            (void)pop.AccessGenomeByID(-999999);
        } catch (const std::runtime_error &) {
            threw = true;
        }
        CHECK(threw);
    }

    // Save/Load round-trip preserves population size and parameters.
    {
        Parameters params = SmallParams();
        Genome seed = MakeSeed();
        Population pop(seed, params, true, 1.0, 2024);
        EvaluateByLinkCount(pop);
        const auto tmp = std::filesystem::temp_directory_path() / "multineat_test_pop.txt";
        pop.Save(tmp.string().c_str());

        Population loaded(tmp.string());
        CHECK(loaded.NumGenomes() == pop.NumGenomes());
        CHECK(loaded.m_Parameters.PopulationSize == params.PopulationSize);
        EXPECT_UNIQUE_IDS(loaded);

        std::error_code ec;
        std::filesystem::remove(tmp, ec);
    }

    // Sort orders species best-first; ChooseParentSpecies stays in range;
    // best-ever and generation counters advance through Epoch().
    {
        Parameters params = SmallParams();
        Population pop(MakeSeed(), params, true, 1.0, 5);
        EvaluateByLinkCount(pop);
        pop.Epoch();
        EvaluateByLinkCount(pop);
        pop.Epoch();
        CHECK(pop.GetGeneration() == 2);
        CHECK(pop.GetBestFitnessEver() > 0.0);
        pop.Sort();
        for (size_t i = 1; i < pop.m_Species.size(); ++i) {
            CHECK(pop.m_Species[i - 1].GetBestFitness() >= pop.m_Species[i].GetBestFitness());
        }
        CHECK(pop.ChooseParentSpecies() < pop.m_Species.size());
        EXPECT_UNIQUE_IDS(pop);
    }

    // Constant fitness makes the stagnation counter climb monotonically.
    {
        Parameters params = SmallParams();
        Population pop(MakeSeed(), params, true, 1.0, 8);
        for (unsigned k = 0; k < 5; ++k) {
            for (unsigned i = 0; i < pop.NumGenomes(); ++i) {
                pop.AccessGenomeByIndex(static_cast<int>(i)).SetFitness(1.0);
                pop.AccessGenomeByIndex(static_cast<int>(i)).SetEvaluated();
            }
            const unsigned before = pop.GetStagnation();
            pop.Epoch();
            CHECK(pop.GetStagnation() >= before);
        }
        CHECK(pop.GetStagnation() >= 3);
    }

    // RemoveWorstIndividual kills exactly the worst evaluated genome; ClearEmptySpecies and ReassignSpecies keep the population consistent.
    {
        Parameters params = SmallParams();
        Population pop(MakeSeed(), params, true, 1.0, 9);
        EvaluateByLinkCount(pop);
        double minfit = std::numeric_limits<double>::max();
        for (unsigned i = 0; i < pop.NumGenomes(); ++i) {
            minfit = std::min(minfit, pop.AccessGenomeByIndex(static_cast<int>(i)).GetFitness());
        }
        const unsigned size0 = pop.NumGenomes();
        Genome removed = pop.RemoveWorstIndividual();
        CHECK(removed.GetFitness() <= minfit + 1e-12);
        CHECK(pop.NumGenomes() == size0 - 1);
        pop.ClearEmptySpecies();
        pop.ReassignSpecies(0);
        EXPECT_UNIQUE_IDS(pop);
    }

    // Tick() replaces one evaluated individual per call and preserves size.
    {
        Parameters params = SmallParams();
        Population pop(MakeSeed(), params, true, 1.0, 11);
        for (int k = 0; k < 10; ++k) {
            EvaluateByLinkCount(pop);
            Genome deleted;
            Genome *baby = pop.Tick(deleted);
            CHECK(baby != nullptr);
            CHECK(pop.NumGenomes() == params.PopulationSize);
        }
        EXPECT_UNIQUE_IDS(pop);
    }

    // Novelty search plumbing: InitPhenotypeBehaviorData wires every genome to a
    // behavior slot and zeroes fitness; one tick runs the full pipeline without
    // corrupting the population. (With the base PhenotypeBehavior, distance is 0
    // so nothing archives, and the base Successful() contract is 'true'.)
    {
        Parameters params = SmallParams();
        params.NoveltySearch_P_min = 0.0;
        Population pop(MakeSeed(), params, true, 1.0, 13);

        std::vector<PhenotypeBehavior> behaviors;
        std::vector<PhenotypeBehavior> archive;
        pop.InitPhenotypeBehaviorData(&behaviors, &archive);
        CHECK(behaviors.size() == pop.NumGenomes());
        CHECK(archive.empty());
        for (unsigned i = 0; i < pop.NumGenomes(); ++i) {
            CHECK(pop.AccessGenomeByIndex(static_cast<int>(i)).m_PhenotypeBehavior != nullptr);
            CHECK(pop.AccessGenomeByIndex(static_cast<int>(i)).GetFitness() == 0.0);
            pop.AccessGenomeByIndex(static_cast<int>(i)).SetEvaluated();  // Tick() needs evaluated individuals
        }

        Genome out;
        const bool solved = pop.NoveltySearchTick(out);
        CHECK(solved);  // base Successful() returns true by contract
        CHECK(pop.NumGenomes() == params.PopulationSize);
        EXPECT_UNIQUE_IDS(pop);
    }

    if (g_failures != 0) {
        std::cerr << "Test failed: TestPopulation with " << g_failures << " failure(s)\n";
        return 1;
    }
    std::cout << "Test passed: TestPopulation\n";
    return 0;
}
