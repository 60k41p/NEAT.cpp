// Tests for NEAT::Species: selection, sorting, fitness sharing.
//
// CTest-Labels: Evolution;Fast
// CTest-Timeout: 120
#include <iostream>
#include <stdexcept>
#include <string>

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

    NEAT::Genome MakeScoredSeed(int id, Real fitness) {
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
        const Real expected = (1.0 + 5.0 + 3.0) / 3.0;
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
        CHECK(std::fabs(s.m_Individuals[0].GetAdjFitness() - 0.5 * p.YoungAgeFitnessBoost) < 1e-6);
        CHECK(std::fabs(s.m_Individuals[1].GetAdjFitness() - 1.5 * p.YoungAgeFitnessBoost) < 1e-6);

        // Stagnation beyond the threshold crushes the adjusted fitness —
        // but never for the species flagged best (the fresh constructor sets that).
        s.SetBestSpecies(false);
        s.m_GensNoImprovement = p.SpeciesMaxStagnation + 1;
        s.AdjustFitness(p);
        CHECK(s.m_Individuals[1].GetAdjFitness() < 1e-6);
    }

    // Explicit parent-selection modes all return evaluated members.
    {
        const SelectionMode modes[] = {TRUNCATION, ROULETTE, RANK_LINEAR, RANK_EXP, TOURNAMENT, STOCHASTIC, BOLTZMANN};
        for (SelectionMode mode : modes) {
            Parameters p = DefaultParams();
            p.ParentSelectionMode = mode;
            p.TruncationSelection = false;
            RNG rng;
            rng.Seed(100 + static_cast<int>(mode));
            Genome g1 = MakeScoredSeed(1, 1.0);
            Genome g2 = MakeScoredSeed(2, 9.0);
            Genome g3 = MakeScoredSeed(3, 5.0);
            Genome g4 = MakeScoredSeed(4, 0.5);
            Species s(g1, p, 1);
            s.AddIndividual(g2);
            s.AddIndividual(g3);
            s.AddIndividual(g4);
            s.SortIndividuals();
            s.AdjustFitness(p);
            Genome &picked = s.GetIndividual(p, rng);
            CHECK(picked.IsEvaluated());
            int id = picked.GetID();
            CHECK(id >= 1 && id <= 4);
        }
    }

    // Truncation restricts the parent pool to the survival fraction.
    {
        Parameters p = DefaultParams();
        p.ParentSelectionMode = TRUNCATION;
        p.SurvivalRate = 0.34;  // 1 of 3 survives truncation
        RNG rng;
        rng.Seed(7);
        Genome g1 = MakeScoredSeed(1, 1.0);
        Genome g2 = MakeScoredSeed(2, 9.0);
        Genome g3 = MakeScoredSeed(3, 5.0);
        Species s(g1, p, 1);
        s.AddIndividual(g2);
        s.AddIndividual(g3);
        s.SortIndividuals();
        s.AdjustFitness(p);
        for (int i = 0; i < 10; ++i) {
            CHECK(s.GetIndividual(p, rng).GetID() == 2);  // only the leader survives
        }
    }

    // StagnationDelta gates the no-improvement reset and best-genome tracking.
    {
        Parameters p = DefaultParams();
        p.StagnationDelta = 1.0;
        Genome seed = MakeSeed();  // unevaluated: best starts at lowest()
        Species s(seed, p, 1);
        Genome g1 = MakeScoredSeed(1, 5.0);
        s.AddIndividual(g1);
        s.m_GensNoImprovement = 10;
        s.AdjustFitness(p);  // fitness 5.0 > lowest(): best updates
        CHECK(s.m_GensNoImprovement == 0);
        CHECK(s.m_BestGenome.GetID() == 1);
        s.m_GensNoImprovement = 10;
        Genome slightly_better = MakeScoredSeed(2, 5.5);  // +0.5 < delta
        s.AddIndividual(slightly_better);
        s.AdjustFitness(p);
        CHECK(s.m_GensNoImprovement == 10);              // counter not reset
        CHECK(s.m_BestGenome.GetID() == 2);              // best still tracks the max
        Genome clearly_better = MakeScoredSeed(3, 7.0);  // +2.0 >= delta
        s.AddIndividual(clearly_better);
        s.AdjustFitness(p);
        CHECK(s.m_GensNoImprovement == 0);
        CHECK(s.m_BestGenome.GetID() == 3);
    }

    // Serialize/Deserialize round-trips members and state.
    {
        Parameters p = DefaultParams();
        Genome g1 = MakeScoredSeed(1, 2.0);
        Genome g2 = MakeScoredSeed(2, 4.0);
        Species s(g1, p, 3);
        s.AddIndividual(g2);
        s.AdjustFitness(p);
        const std::string data = s.Serialize();
        const Species r = Species::Deserialize(data);
        CHECK(r.ID() == 3);
        CHECK(r.NumIndividuals() == 2);
        CHECK(r.m_Individuals[0].IsIdenticalTo(s.m_Individuals[0]));
    }

    if (g_failures != 0) {
        std::cerr << "Test failed: TestSpecies with " << g_failures << " failure(s)\n";
        return 1;
    }
    std::cout << "Test passed: TestSpecies\n";
    return 0;
}
