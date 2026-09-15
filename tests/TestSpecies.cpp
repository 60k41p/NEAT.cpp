// Tests for NEAT::Species: selection, sorting, fitness sharing.
//
// CTest-Labels: Evolution;Fast
// CTest-Timeout: 120
#include <iostream>
#include <stdexcept>

#include "Genome.h"
#include "Parameters.h"
#include "Random.h"
#include "Species.h"

namespace {

    int g_failures = 0;

#define CHECK(cond)                                                                         \
    do {                                                                                    \
        if (!(cond)) {                                                                      \
            std::cerr << "FAILED " << __FILE__ << ":" << __LINE__ << ": " << #cond << "\n"; \
            ++g_failures;                                                                   \
        }                                                                                   \
    } while (0)

    NEAT::Parameters defaultParams() {
        NEAT::Parameters p;
        p.reset();
        return p;
    }

    NEAT::Genome makeSeed() {
        NEAT::Parameters p = defaultParams();
        NEAT::GenomeInitStruct init;
        init.numInputs = 3;
        init.numOutputs = 1;
        init.seedType = NEAT::PERCEPTRON;
        return NEAT::Genome(p, init);
    }

    NEAT::Genome makeScoredSeed(int id, double fitness) {
        NEAT::Genome g = makeSeed();
        g.setID(id);
        g.setFitness(fitness);
        g.setEvaluated();
        return g;
    }

}  // namespace

int TestSpecies(int argc, char *argv[]) {
    (void)argc;
    (void)argv;
    using namespace NEAT;

    // Construction seeds one individual; AddIndividual grows it.
    {
        Parameters p = defaultParams();
        Genome seed = makeScoredSeed(1, 1.0);
        Species s(seed, p, 7);
        CHECK(s.id() == 7);
        CHECK(s.numIndividuals() == 1);
        Genome extra = makeScoredSeed(2, 2.0);
        s.addIndividual(extra);
        CHECK(s.numIndividuals() == 2);
    }

    // SortIndividuals orders best-first; average fitness is the mean.
    {
        Parameters p = defaultParams();
        Genome seed = makeScoredSeed(1, 1.0);
        Species s(seed, p, 1);
        Genome g2 = makeScoredSeed(2, 5.0);
        Genome g3 = makeScoredSeed(3, 3.0);
        s.addIndividual(g2);
        s.addIndividual(g3);
        s.sortIndividuals();
        CHECK(s.individuals_[0].getFitness() >= s.individuals_[1].getFitness());
        CHECK(s.individuals_[1].getFitness() >= s.individuals_[2].getFitness());
        s.calculateAverageFitness();
        const double expected = (1.0 + 5.0 + 3.0) / 3.0;
        CHECK(std::fabs(s.averageFitness_ - expected) < 1e-9);
        CHECK(s.getLeader().getFitness() == 5.0);
    }

    // AdjustFitness (fitness sharing + age modifiers) preserves membership
    // and keeps adjusted fitness non-negative.
    {
        Parameters p = defaultParams();
        Genome seed = makeScoredSeed(1, 2.0);
        Species s(seed, p, 1);
        Genome g2 = makeScoredSeed(2, 4.0);
        s.addIndividual(g2);
        s.adjustFitness(p);
        CHECK(s.numIndividuals() == 2);
        for (const Genome &ind : s.individuals_) {
            CHECK(ind.getAdjFitness() >= 0.0);
        }
        s.countOffspring();
        CHECK(s.getOffspringRqd() >= 0.0);
    }

    // GetIndividual returns an evaluated member; empty/unevaluated throws.
    {
        Parameters p = defaultParams();
        p.tournamentSelection = true;
        p.tournamentSize = 2;
        RNG rng;
        rng.seed(17);
        Genome seed = makeScoredSeed(1, 1.0);
        Species s(seed, p, 1);
        Genome g2 = makeScoredSeed(2, 9.0);
        Genome g3 = makeScoredSeed(3, 5.0);
        s.addIndividual(g2);
        s.addIndividual(g3);
        s.sortIndividuals();
        Genome &picked = s.getIndividual(p, rng);
        CHECK(picked.isEvaluated());
        Genome &rnd = s.getRandomIndividual(rng);
        CHECK(rnd.getID() == 1 || rnd.getID() == 2 || rnd.getID() == 3);
    }
    {
        Parameters p = defaultParams();
        RNG rng;
        rng.seed(1);
        Genome seed = makeSeed();  // not evaluated
        Species s(seed, p, 1);
        bool threw = false;
        try {
            (void)s.getIndividual(p, rng);
        } catch (const std::runtime_error &) {
            threw = true;
        }
        CHECK(threw);

        Species empty;
        threw = false;
        try {
            (void)empty.getIndividual(p, rng);
        } catch (const std::runtime_error &) {
            threw = true;
        }
        CHECK(threw);
    }

    // RemoveIndividual shrinks; Clear empties.
    {
        Parameters p = defaultParams();
        Genome seed = makeScoredSeed(1, 1.0);
        Species s(seed, p, 1);
        Genome g2 = makeScoredSeed(2, 2.0);
        s.addIndividual(g2);
        CHECK(s.numIndividuals() == 2);
        s.removeIndividual(0);
        CHECK(s.numIndividuals() == 1);
        s.clear();
        CHECK(s.numIndividuals() == 0);
    }

    // GetRepresentative returns the first individual; empty species throws.
    {
        Parameters p = defaultParams();
        Genome seed = makeScoredSeed(1, 1.0);
        Species s(seed, p, 1);
        Genome rep2 = makeScoredSeed(2, 9.0);
        s.addIndividual(rep2);
        s.sortIndividuals();
        CHECK(s.getRepresentative().getID() == s.individuals_[0].getID());
        Species empty;
        bool threw = false;
        try {
            (void)empty.getRepresentative();
        } catch (const std::runtime_error &) {
            threw = true;
        }
        CHECK(threw);
    }

    // AdjustFitness divides by species size; long-stagnant non-best species get killed off.
    {
        Parameters p = defaultParams();
        Genome seed = makeScoredSeed(1, 1.0);
        Species s(seed, p, 1);
        Genome second = makeScoredSeed(2, 3.0);
        s.addIndividual(second);
        s.adjustFitness(p);
        // Species age 0 < YoungAgeTreshold, so fitness gets the young-age boost
        // and is then divided by species size.
        CHECK(std::fabs(s.individuals_[0].getAdjFitness() - 0.5 * p.youngAgeFitnessBoost) < 1e-9);
        CHECK(std::fabs(s.individuals_[1].getAdjFitness() - 1.5 * p.youngAgeFitnessBoost) < 1e-9);

        // Stagnation beyond the threshold crushes the adjusted fitness —
        // but never for the species flagged best (the fresh constructor sets that).
        s.setBestSpecies(false);
        s.gensNoImprovement_ = p.speciesMaxStagnation + 1;
        s.adjustFitness(p);
        CHECK(s.individuals_[1].getAdjFitness() < 1e-6);
    }

    if (g_failures != 0) {
        std::cerr << "Test failed: TestSpecies with " << g_failures << " failure(s)\n";
        return 1;
    }
    std::cout << "Test passed: TestSpecies\n";
    return 0;
}
