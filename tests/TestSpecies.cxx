// Tests for NEAT::Species: selection, sorting, fitness sharing.
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

    NEAT::Parameters DefaultParams() {
        NEAT::Parameters p;
        p.Reset();
        return p;
    }

    NEAT::Genome MakeSeed() {
        NEAT::Parameters p = DefaultParams();
        NEAT::GenomeInitStruct init;
        init.NumInputs = 3;
        init.NumOutputs = 1;
        init.SeedType = NEAT::PERCEPTRON;
        return NEAT::Genome(p, init);
    }

    NEAT::Genome MakeScoredSeed(int id, double fitness) {
        NEAT::Genome g = MakeSeed();
        g.SetID(id);
        g.SetFitness(fitness);
        g.SetEvaluated();
        return g;
    }

}  // namespace

int TestSpecies(int argc, char *argv[]) {
    (void)argc;
    (void)argv;
    using namespace NEAT;

    // Construction seeds one individual; AddIndividual grows it.
    {
        Parameters p = DefaultParams();
        Genome seed = MakeScoredSeed(1, 1.0);
        Species s(seed, p, 7);
        CHECK(s.ID() == 7);
        CHECK(s.NumIndividuals() == 1);
        Genome extra = MakeScoredSeed(2, 2.0);
        s.AddIndividual(extra);
        CHECK(s.NumIndividuals() == 2);
    }

    // SortIndividuals orders best-first; average fitness is the mean.
    {
        Parameters p = DefaultParams();
        Genome seed = MakeScoredSeed(1, 1.0);
        Species s(seed, p, 1);
        Genome g2 = MakeScoredSeed(2, 5.0);
        Genome g3 = MakeScoredSeed(3, 3.0);
        s.AddIndividual(g2);
        s.AddIndividual(g3);
        s.SortIndividuals();
        CHECK(s.m_Individuals[0].GetFitness() >= s.m_Individuals[1].GetFitness());
        CHECK(s.m_Individuals[1].GetFitness() >= s.m_Individuals[2].GetFitness());
        s.CalculateAverageFitness();
        const double expected = (1.0 + 5.0 + 3.0) / 3.0;
        CHECK(std::fabs(s.m_AverageFitness - expected) < 1e-9);
        CHECK(s.GetLeader().GetFitness() == 5.0);
    }

    // AdjustFitness (fitness sharing + age modifiers) preserves membership
    // and keeps adjusted fitness non-negative.
    {
        Parameters p = DefaultParams();
        Genome seed = MakeScoredSeed(1, 2.0);
        Species s(seed, p, 1);
        Genome g2 = MakeScoredSeed(2, 4.0);
        s.AddIndividual(g2);
        s.AdjustFitness(p);
        CHECK(s.NumIndividuals() == 2);
        for (const auto &ind : s.m_Individuals) {
            CHECK(ind.GetAdjFitness() >= 0.0);
        }
        s.CountOffspring();
        CHECK(s.GetOffspringRqd() >= 0.0);
    }

    // GetIndividual returns an evaluated member; empty/unevaluated throws.
    {
        Parameters p = DefaultParams();
        p.TournamentSelection = true;
        p.TournamentSize = 2;
        RNG rng;
        rng.Seed(17);
        Genome seed = MakeScoredSeed(1, 1.0);
        Species s(seed, p, 1);
        Genome g2 = MakeScoredSeed(2, 9.0);
        Genome g3 = MakeScoredSeed(3, 5.0);
        s.AddIndividual(g2);
        s.AddIndividual(g3);
        s.SortIndividuals();
        Genome &picked = s.GetIndividual(p, rng);
        CHECK(picked.IsEvaluated());
        Genome &rnd = s.GetRandomIndividual(rng);
        CHECK(rnd.GetID() == 1 || rnd.GetID() == 2 || rnd.GetID() == 3);
    }
    {
        Parameters p = DefaultParams();
        RNG rng;
        rng.Seed(1);
        Genome seed = MakeSeed();  // not evaluated
        Species s(seed, p, 1);
        bool threw = false;
        try {
            (void)s.GetIndividual(p, rng);
        } catch (const std::runtime_error &) {
            threw = true;
        }
        CHECK(threw);

        Species empty;
        threw = false;
        try {
            (void)empty.GetIndividual(p, rng);
        } catch (const std::runtime_error &) {
            threw = true;
        }
        CHECK(threw);
    }

    // RemoveIndividual shrinks; Clear empties.
    {
        Parameters p = DefaultParams();
        Genome seed = MakeScoredSeed(1, 1.0);
        Species s(seed, p, 1);
        Genome g2 = MakeScoredSeed(2, 2.0);
        s.AddIndividual(g2);
        CHECK(s.NumIndividuals() == 2);
        s.RemoveIndividual(0);
        CHECK(s.NumIndividuals() == 1);
        s.Clear();
        CHECK(s.NumIndividuals() == 0);
    }

    // GetRepresentative returns the first individual; empty species throws.
    {
        Parameters p = DefaultParams();
        Genome seed = MakeScoredSeed(1, 1.0);
        Species s(seed, p, 1);
        Genome rep2 = MakeScoredSeed(2, 9.0);
        s.AddIndividual(rep2);
        s.SortIndividuals();
        CHECK(s.GetRepresentative().GetID() == s.m_Individuals[0].GetID());
        Species empty;
        bool threw = false;
        try {
            (void)empty.GetRepresentative();
        } catch (const std::runtime_error &) {
            threw = true;
        }
        CHECK(threw);
    }

    // AdjustFitness divides by species size; long-stagnant non-best species get killed off.
    {
        Parameters p = DefaultParams();
        Genome seed = MakeScoredSeed(1, 1.0);
        Species s(seed, p, 1);
        Genome second = MakeScoredSeed(2, 3.0);
        s.AddIndividual(second);
        s.AdjustFitness(p);
        // Species age 0 < YoungAgeTreshold, so fitness gets the young-age boost
        // and is then divided by species size.
        CHECK(std::fabs(s.m_Individuals[0].GetAdjFitness() - 0.5 * p.YoungAgeFitnessBoost) < 1e-9);
        CHECK(std::fabs(s.m_Individuals[1].GetAdjFitness() - 1.5 * p.YoungAgeFitnessBoost) < 1e-9);

        // Stagnation beyond the threshold crushes the adjusted fitness —
        // but never for the species flagged best (the fresh constructor sets that).
        s.SetBestSpecies(false);
        s.m_GensNoImprovement = p.SpeciesMaxStagnation + 1;
        s.AdjustFitness(p);
        CHECK(s.m_Individuals[1].GetAdjFitness() < 1e-6);
    }

    if (g_failures != 0) {
        std::cerr << "Test failed: TestSpecies with " << g_failures << " failure(s)\n";
        return 1;
    }
    std::cout << "Test passed: TestSpecies\n";
    return 0;
}
