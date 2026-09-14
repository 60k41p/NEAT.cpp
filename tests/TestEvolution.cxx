// End-to-end neuroevolution test: evolve a network that solves XOR via the
// public Population/Epoch API, the way a real user would.
//
// Parameters follow the reference recipe from upstream MultiNEAT's
// examples/TestNEAT_xor.py (which reliably solves XOR): squared-error fitness,
// fixed steep unsigned sigmoids, pure mutation (no crossover), rare neuron
// addition, dynamic compatibility. All five seeds typically solve within ~50
// generations; the budget below gives a large cross-platform margin.
#include <algorithm>
#include <cmath>
#include <iostream>
#include <vector>

#include "Genome.h"
#include "NeuralNetwork.h"
#include "Parameters.h"
#include "Population.h"

namespace {

    using namespace NEAT;

    int g_failures = 0;

#define CHECK(cond)                                                                         \
    do {                                                                                    \
        if (!(cond)) {                                                                      \
            std::cerr << "FAILED " << __FILE__ << ":" << __LINE__ << ": " << #cond << "\n"; \
            ++g_failures;                                                                   \
        }                                                                                   \
    } while (0)

    // Two inputs + trailing bias slot (GenomeInitStruct::NumInputs counts the bias).
    const double kXorIn[4][2] = {{0.0, 0.0}, {0.0, 1.0}, {1.0, 0.0}, {1.0, 1.0}};
    const double kXorTarget[4] = {0.0, 1.0, 1.0, 0.0};

    // Absolute error of a genome on XOR summed over the four patterns.
    double XorError(Genome &g) {
        NeuralNetwork net;
        g.BuildPhenotype(net);
        double err = 0.0;
        for (int p = 0; p < 4; ++p) {
            net.Flush();
            std::vector<double> in = {kXorIn[p][0], kXorIn[p][1], 1.0};
            net.Input(in);
            net.Activate();
            net.Activate();  // let any recurrent activity settle, as the reference does
            err += std::fabs(net.Output()[0] - kXorTarget[p]);
        }
        return err;
    }

    double XorFitness(Genome &g) {
        const double rem = 4.0 - XorError(g);
        return rem * rem;  // squared margin, as in the reference example
    }

    void EvaluateXOR(Population &pop) {
        for (unsigned i = 0; i < pop.NumGenomes(); ++i) {
            Genome &g = pop.AccessGenomeByIndex(static_cast<int>(i));
            g.SetFitness(XorFitness(g));
            g.SetEvaluated();
        }
    }

    Parameters XorParams() {
        Parameters p;
        p.PopulationSize = 100;
        p.DynamicCompatibility = true;
        p.NormalizeGenomeSize = true;
        p.WeightDiffCoeff = 0.1;
        p.CompatTreshold = 2.0;
        p.YoungAgeTreshold = 15;
        p.SpeciesMaxStagnation = 15;
        p.OldAgeTreshold = 35;
        p.MinSpecies = 2;
        p.MaxSpecies = 10;
        p.RouletteWheelSelection = false;
        p.RecurrentProb = 0.0;
        p.OverallMutationRate = 1.0;
        p.ArchiveEnforcement = false;
        p.MutateWeightsProb = 0.05;
        p.WeightMutationMaxPower = 0.5;
        p.WeightReplacementMaxPower = 8.0;
        p.MutateWeightsSevereProb = 0.0;
        p.WeightMutationRate = 0.25;
        p.WeightReplacementRate = 0.9;
        p.MaxWeight = 8.0;
        p.MutateAddNeuronProb = 0.001;
        p.MutateAddLinkProb = 0.3;
        p.MutateRemLinkProb = 0.0;
        p.MinActivationA = 4.9;
        p.MaxActivationA = 4.9;
        p.ActivationFunction_SignedSigmoid_Prob = 0.0;
        p.ActivationFunction_UnsignedSigmoid_Prob = 1.0;
        p.ActivationFunction_Tanh_Prob = 0.0;
        p.ActivationFunction_SignedStep_Prob = 0.0;
        p.CrossoverRate = 0.0;
        p.MultipointCrossoverRate = 0.0;
        p.SurvivalRate = 0.2;
        p.MutateNeuronTraitsProb = 0.0;
        p.MutateLinkTraitsProb = 0.0;
        p.AllowLoops = true;
        p.AllowClones = true;
        return p;
    }

    Genome MakeXorSeed() {
        Parameters q;
        q.Reset();
        GenomeInitStruct init;
        init.NumInputs = 3;  // 2 problem inputs + bias
        init.NumOutputs = 1;
        init.SeedType = PERCEPTRON;
        return Genome(q, init);
    }

    // Runs XOR evolution with one seed. Returns the generation at which the
    // best fitness exceeded 15.0 (summed error < ~0.13), or -1 if the budget ran out.
    int EvolveXor(unsigned long rng_seed, unsigned max_generations, double *best_fitness_out = nullptr) {
        Population pop(MakeXorSeed(), XorParams(), true, 1.0, static_cast<int>(rng_seed));
        double best_seen = -1.0;
        for (unsigned gen = 0; gen < max_generations; ++gen) {
            EvaluateXOR(pop);
            double best = 0.0;
            for (unsigned i = 0; i < pop.NumGenomes(); ++i) {
                best = std::max(best, pop.AccessGenomeByIndex(static_cast<int>(i)).GetFitness());
            }
            best_seen = std::max(best_seen, best);
            if (best > 15.0) {
                if (best_fitness_out) {
                    *best_fitness_out = best;
                }
                return static_cast<int>(gen);
            }
            pop.Epoch();
            // The current leader can disappear when its species receives no
            // offspring. The population's historical record must not regress.
            CHECK(pop.GetBestFitnessEver() >= best_seen - 1e-9);
        }
        if (best_fitness_out) {
            *best_fitness_out = best_seen;
        }
        return -1;
    }

}  // namespace

int TestEvolution(int argc, char *argv[]) {
    (void)argc;
    (void)argv;
    using namespace NEAT;

    // Five seeded runs; the reference recipe solves XOR reliably (typically
    // generation < 50), so demand all five within the 300-generation budget.
    int solved = 0;
    for (unsigned long seed = 1; seed <= 5; ++seed) {
        const int gen = EvolveXor(seed, 300);
        if (gen >= 0) {
            ++solved;
            std::cout << "XOR solved with seed " << seed << " at generation " << gen << "\n";
        }
    }
    CHECK(solved == 5);

    // The winner's phenotype must actually implement XOR.
    if (solved >= 1) {
        Population pop(MakeXorSeed(), XorParams(), true, 1.0, 1);
        int gens = 0;
        while (gens < 300) {
            EvaluateXOR(pop);
            double best = 0.0;
            for (unsigned i = 0; i < pop.NumGenomes(); ++i) {
                best = std::max(best, pop.AccessGenomeByIndex(static_cast<int>(i)).GetFitness());
            }
            if (best > 15.0) {
                break;
            }
            pop.Epoch();
            ++gens;
        }
        EvaluateXOR(pop);
        Genome best = pop.GetBestGenome();
        CHECK(XorError(best) < 0.13);
        // m_BestFitnessEver is only updated inside Epoch(); since we stop right
        // after evaluation, just require it tracked the earlier generations.
        CHECK(pop.GetBestFitnessEver() > 0.0);
        pop.SameGenomeIDCheck();
    }

    // Determinism: identical seeds replay identical best fitness trajectories.
    {
        auto first_gens_best = [](unsigned long seed, unsigned gens) {
            Population pop(MakeXorSeed(), XorParams(), true, 1.0, static_cast<int>(seed));
            double best = 0.0;
            for (unsigned g = 0; g < gens; ++g) {
                EvaluateXOR(pop);
                for (unsigned i = 0; i < pop.NumGenomes(); ++i) {
                    best = std::max(best, pop.AccessGenomeByIndex(static_cast<int>(i)).GetFitness());
                }
                pop.Epoch();
            }
            return best;
        };
        CHECK(first_gens_best(77, 40) == first_gens_best(77, 40));
    }

    if (g_failures != 0) {
        std::cerr << "Test failed: TestEvolution with " << g_failures << " failure(s)\n";
        return 1;
    }
    std::cout << "Test passed: TestEvolution\n";
    return 0;
}
