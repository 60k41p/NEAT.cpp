// Seeded evolution smoke/regression tests for NEAT::Population.
// Small populations + fixed RNG seeds keep these fast and deterministic.
#include <cmath>
#include <filesystem>
#include <iostream>
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

    // Trivial evaluator: reward genomes with more links (exercises fitness flow
    // without any domain code). Deterministic given the population state.
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

    if (g_failures != 0) {
        std::cerr << "Test failed: TestPopulation with " << g_failures << " failure(s)\n";
        return 1;
    }
    std::cout << "Test passed: TestPopulation\n";
    return 0;
}
