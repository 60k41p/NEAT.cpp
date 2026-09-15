// Seeded evolution smoke/regression tests for NEAT::Population.
// Small populations + fixed RNG seeds keep these fast and deterministic.
//
// CTest-Labels: Evolution;Fast
// CTest-Timeout: 240
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

    NEAT::Parameters smallParams() {
        NEAT::Parameters p;
        p.reset();
        p.populationSize = 20;
        p.allowClones = true;
        p.mutateAddNeuronProb = 0.1;
        p.mutateAddLinkProb = 0.2;
        p.mutateRemLinkProb = 0.02;
        p.mutateWeightsProb = 0.8;
        p.survivalRate = 0.3;
        return p;
    }

    NEAT::Genome makeSeed() {
        NEAT::Parameters p;
        p.reset();
        NEAT::GenomeInitStruct init;
        init.numInputs = 3;
        init.numOutputs = 1;
        init.seedType = NEAT::PERCEPTRON;
        return NEAT::Genome(p, init);
    }

    // Trivial evaluator: reward genomes with more links (exercises fitness flow without any domain code). Deterministic given the population state.
    void evaluateByLinkCount(NEAT::Population &pop) {
        for (unsigned i = 0; i < pop.numGenomes(); ++i) {
            NEAT::Genome &g = pop.accessGenomeByIndex(static_cast<int>(i));
            g.setFitness(1.0 + static_cast<double>(g.numLinks()));
            g.setEvaluated();
        }
    }

    // SameGenomeIDCheck throws on duplicates; convert that into a CHECK failure
    // so a regression reports cleanly instead of terminating the driver.
    void expectUniqueIDs(NEAT::Population &pop, int line) {
        try {
            pop.sameGenomeIDCheck();
        } catch (const std::exception &e) {
            std::cerr << "FAILED TestPopulation.cpp:" << line << ": unique genome IDs (" << e.what() << ")\n";
            ++g_failures;
        }
    }

#define EXPECT_UNIQUE_IDS(pop) expectUniqueIDs(pop, __LINE__)

}  // namespace

int TestPopulation(int argc, char *argv[]) {
    (void)argc;
    (void)argv;
    using namespace NEAT;

    // Construction invariants: size, generation 0, unique IDs.
    {
        Parameters params = smallParams();
        Genome seed = makeSeed();
        Population pop(seed, params, true, 1.0, 42);
        CHECK(pop.numGenomes() == params.populationSize);
        CHECK(pop.getGeneration() == 0);
        EXPECT_UNIQUE_IDS(pop);
        CHECK(pop.getNextGenomeID() >= params.populationSize);
    }

    // Epoch() advances generations, preserves size and ID uniqueness.
    {
        Parameters params = smallParams();
        Genome seed = makeSeed();
        Population pop(seed, params, true, 1.0, 42);
        for (int gen = 0; gen < 5; ++gen) {
            evaluateByLinkCount(pop);
            pop.epoch();
            CHECK(pop.getGeneration() == static_cast<unsigned>(gen + 1));
            CHECK(pop.numGenomes() == params.populationSize);
            EXPECT_UNIQUE_IDS(pop);
        }
        // Epoch() leaves newborns unevaluated (fitness 0), so evaluate once
        // more before asserting on best fitness.
        evaluateByLinkCount(pop);
        pop.sort();
        CHECK(pop.getBestGenome().getFitness() > 0.0);
    }

    // Fixed-seed determinism: two identical runs reach identical best fitness.
    {
        auto run = [] {
            Parameters params = smallParams();
            Genome seed = makeSeed();
            Population pop(seed, params, true, 1.0, 1234);
            for (int gen = 0; gen < 4; ++gen) {
                evaluateByLinkCount(pop);
                pop.epoch();
            }
            evaluateByLinkCount(pop);
            pop.sort();
            return pop.getBestGenome().getFitness();
        };
        const double a = run();
        const double b = run();
        CHECK(a == b);
    }

    // Tick() (steady-state) replaces one individual per call after evaluation.
    {
        Parameters params = smallParams();
        Genome seed = makeSeed();
        Population pop(seed, params, true, 1.0, 99);
        evaluateByLinkCount(pop);
        const unsigned before = pop.numGenomes();
        Genome deleted;
        Genome *baby = pop.tick(deleted);
        CHECK(baby != nullptr);
        CHECK(pop.numGenomes() == before);
        EXPECT_UNIQUE_IDS(pop);
        // Tick on an unevaluated population must throw, not hang.
        Population fresh(seed, params, true, 1.0, 100);
        bool threw = false;
        try {
            Genome d2;
            (void)fresh.tick(d2);
        } catch (const std::runtime_error &) {
            threw = true;
        }
        CHECK(threw);
    }

    // Accessors: by index round-trips, out-of-range throws.
    {
        Parameters params = smallParams();
        Genome seed = makeSeed();
        Population pop(seed, params, true, 1.0, 5);
        Genome &g0 = pop.accessGenomeByIndex(0);
        CHECK(pop.accessGenomeByID(g0.getID()).getID() == g0.getID());
        bool threw = false;
        try {
            (void)pop.accessGenomeByIndex(static_cast<int>(pop.numGenomes()) + 10);
        } catch (const std::runtime_error &) {
            threw = true;
        }
        CHECK(threw);
        threw = false;
        try {
            (void)pop.accessGenomeByID(-999999);
        } catch (const std::runtime_error &) {
            threw = true;
        }
        CHECK(threw);
    }

    // Save/Load round-trip preserves population size and parameters.
    {
        Parameters params = smallParams();
        Genome seed = makeSeed();
        Population pop(seed, params, true, 1.0, 2024);
        evaluateByLinkCount(pop);
        const std::filesystem::path tmp = std::filesystem::temp_directory_path() / "multineat_test_pop.txt";
        pop.save(tmp.string().c_str());

        Population loaded(tmp.string());
        CHECK(loaded.numGenomes() == pop.numGenomes());
        CHECK(loaded.parameters_.populationSize == params.populationSize);
        EXPECT_UNIQUE_IDS(loaded);

        std::error_code ec;
        std::filesystem::remove(tmp, ec);
    }

    // Sort orders species best-first; ChooseParentSpecies stays in range;
    // best-ever and generation counters advance through Epoch().
    {
        Parameters params = smallParams();
        Population pop(makeSeed(), params, true, 1.0, 5);
        evaluateByLinkCount(pop);
        pop.epoch();
        evaluateByLinkCount(pop);
        pop.epoch();
        CHECK(pop.getGeneration() == 2);
        CHECK(pop.getBestFitnessEver() > 0.0);
        pop.sort();
        for (size_t i = 1; i < pop.species_.size(); ++i) {
            CHECK(pop.species_[i - 1].getBestFitness() >= pop.species_[i].getBestFitness());
        }
        CHECK(pop.chooseParentSpecies() < pop.species_.size());
        EXPECT_UNIQUE_IDS(pop);
    }

    // Constant fitness makes the stagnation counter climb monotonically.
    {
        Parameters params = smallParams();
        Population pop(makeSeed(), params, true, 1.0, 8);
        for (unsigned k = 0; k < 5; ++k) {
            for (unsigned i = 0; i < pop.numGenomes(); ++i) {
                pop.accessGenomeByIndex(static_cast<int>(i)).setFitness(1.0);
                pop.accessGenomeByIndex(static_cast<int>(i)).setEvaluated();
            }
            const unsigned before = pop.getStagnation();
            pop.epoch();
            CHECK(pop.getStagnation() >= before);
        }
        CHECK(pop.getStagnation() >= 3);
    }

    // RemoveWorstIndividual kills exactly the worst evaluated genome; ClearEmptySpecies and ReassignSpecies keep the population consistent.
    {
        Parameters params = smallParams();
        Population pop(makeSeed(), params, true, 1.0, 9);
        evaluateByLinkCount(pop);
        double minfit = std::numeric_limits<double>::max();
        for (unsigned i = 0; i < pop.numGenomes(); ++i) {
            minfit = std::min(minfit, pop.accessGenomeByIndex(static_cast<int>(i)).getFitness());
        }
        const unsigned size0 = pop.numGenomes();
        Genome removed = pop.removeWorstIndividual();
        CHECK(removed.getFitness() <= minfit + 1e-12);
        CHECK(pop.numGenomes() == size0 - 1);
        pop.clearEmptySpecies();
        pop.reassignSpecies(0);
        EXPECT_UNIQUE_IDS(pop);
    }

    // Tick() replaces one evaluated individual per call and preserves size.
    {
        Parameters params = smallParams();
        Population pop(makeSeed(), params, true, 1.0, 11);
        for (int k = 0; k < 10; ++k) {
            evaluateByLinkCount(pop);
            Genome deleted;
            Genome *baby = pop.tick(deleted);
            CHECK(baby != nullptr);
            CHECK(pop.numGenomes() == params.populationSize);
        }
        EXPECT_UNIQUE_IDS(pop);
    }

    // Novelty search plumbing: InitPhenotypeBehaviorData wires every genome to a
    // behavior slot and zeroes fitness; one tick runs the full pipeline without
    // corrupting the population. (With the base PhenotypeBehavior, distance is 0
    // so nothing archives, and the base Successful() contract is 'true'.)
    {
        Parameters params = smallParams();
        params.noveltySearchPMin = 0.0;
        Population pop(makeSeed(), params, true, 1.0, 13);

        std::vector<PhenotypeBehavior> behaviors;
        std::vector<PhenotypeBehavior> archive;
        pop.initPhenotypeBehaviorData(&behaviors, &archive);
        CHECK(behaviors.size() == pop.numGenomes());
        CHECK(archive.empty());
        for (unsigned i = 0; i < pop.numGenomes(); ++i) {
            CHECK(pop.accessGenomeByIndex(static_cast<int>(i)).phenotypeBehavior_ != nullptr);
            CHECK(pop.accessGenomeByIndex(static_cast<int>(i)).getFitness() == 0.0);
            pop.accessGenomeByIndex(static_cast<int>(i)).setEvaluated();  // Tick() needs evaluated individuals
        }

        Genome out;
        const bool solved = pop.noveltySearchTick(out);
        CHECK(solved);  // base Successful() returns true by contract
        CHECK(pop.numGenomes() == params.populationSize);
        EXPECT_UNIQUE_IDS(pop);
    }

    if (g_failures != 0) {
        std::cerr << "Test failed: TestPopulation with " << g_failures << " failure(s)\n";
        return 1;
    }
    std::cout << "Test passed: TestPopulation\n";
    return 0;
}
