// End-to-end neuroevolution test: evolve a network that solves XOR via the
// public Population/Epoch API, the way a real user would.
//
// Parameters follow the reference recipe from upstream MultiNEAT's
// examples/TestNEAT_xor.py (which reliably solves XOR): squared-error fitness,
// fixed steep unsigned sigmoids, pure mutation (no crossover), rare neuron
// addition, dynamic compatibility. All five seeds typically solve within ~50
// generations; the budget below gives a large cross-platform margin.
//
// CTest-Labels: Evolution
// CTest-Timeout: 600
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
    double xorError(Genome &g) {
        NeuralNetwork net;
        g.buildPhenotype(net);
        double err = 0.0;
        for (int p = 0; p < 4; ++p) {
            net.flush();
            std::vector<double> in = {kXorIn[p][0], kXorIn[p][1], 1.0};
            net.input(in);
            net.activate();
            net.activate();  // let any recurrent activity settle, as the reference does
            err += std::fabs(net.output()[0] - kXorTarget[p]);
        }
        return err;
    }

    double xorFitness(Genome &g) {
        const double rem = 4.0 - xorError(g);
        return rem * rem;  // squared margin, as in the reference example
    }

    void evaluateXOR(Population &pop) {
        for (unsigned i = 0; i < pop.numGenomes(); ++i) {
            Genome &g = pop.accessGenomeByIndex(static_cast<int>(i));
            g.setFitness(xorFitness(g));
            g.setEvaluated();
        }
    }

    Parameters xorParams() {
        Parameters p;
        p.populationSize = 100;
        p.dynamicCompatibility = true;
        p.normalizeGenomeSize = true;
        p.weightDiffCoeff = 0.1;
        p.compatTreshold = 2.0;
        p.youngAgeTreshold = 15;
        p.speciesMaxStagnation = 15;
        p.oldAgeTreshold = 35;
        p.minSpecies = 2;
        p.maxSpecies = 10;
        p.rouletteWheelSelection = false;
        p.recurrentProb = 0.0;
        p.overallMutationRate = 1.0;
        p.archiveEnforcement = false;
        p.mutateWeightsProb = 0.05;
        p.weightMutationMaxPower = 0.5;
        p.weightReplacementMaxPower = 8.0;
        p.mutateWeightsSevereProb = 0.0;
        p.weightMutationRate = 0.25;
        p.weightReplacementRate = 0.9;
        p.maxWeight = 8.0;
        p.mutateAddNeuronProb = 0.001;
        p.mutateAddLinkProb = 0.3;
        p.mutateRemLinkProb = 0.0;
        p.minActivationA = 4.9;
        p.maxActivationA = 4.9;
        p.activationFunctionSignedSigmoidProb = 0.0;
        p.activationFunctionUnsignedSigmoidProb = 1.0;
        p.activationFunctionTanhProb = 0.0;
        p.activationFunctionSignedStepProb = 0.0;
        p.crossoverRate = 0.0;
        p.multipointCrossoverRate = 0.0;
        p.survivalRate = 0.2;
        p.mutateNeuronTraitsProb = 0.0;
        p.mutateLinkTraitsProb = 0.0;
        p.allowLoops = true;
        p.allowClones = true;
        return p;
    }

    Genome makeXorSeed() {
        Parameters q;
        q.reset();
        GenomeInitStruct init;
        init.numInputs = 3;  // 2 problem inputs + bias
        init.numOutputs = 1;
        init.seedType = PERCEPTRON;
        return Genome(q, init);
    }

    // Runs XOR evolution with one seed. Returns the generation at which the
    // best fitness exceeded 15.0 (summed error < ~0.13), or -1 if the budget ran out.
    int evolveXor(unsigned long rngSeed, unsigned maxGenerations, double *bestFitnessOut = nullptr) {
        Population pop(makeXorSeed(), xorParams(), true, 1.0, static_cast<int>(rngSeed));
        double bestSeen = -1.0;
        for (unsigned gen = 0; gen < maxGenerations; ++gen) {
            evaluateXOR(pop);
            double best = 0.0;
            for (unsigned i = 0; i < pop.numGenomes(); ++i) {
                best = std::max(best, pop.accessGenomeByIndex(static_cast<int>(i)).getFitness());
            }
            bestSeen = std::max(bestSeen, best);
            if (best > 15.0) {
                if (bestFitnessOut) {
                    *bestFitnessOut = best;
                }
                return static_cast<int>(gen);
            }
            pop.epoch();
            // The current leader can disappear when its species receives no
            // offspring. The population's historical record must not regress.
            CHECK(pop.getBestFitnessEver() >= bestSeen - 1e-9);
        }
        if (bestFitnessOut) {
            *bestFitnessOut = bestSeen;
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
        const int gen = evolveXor(seed, 300);
        if (gen >= 0) {
            ++solved;
            std::cout << "XOR solved with seed " << seed << " at generation " << gen << "\n";
        }
    }
    CHECK(solved == 5);

    // The winner's phenotype must actually implement XOR.
    if (solved >= 1) {
        Population pop(makeXorSeed(), xorParams(), true, 1.0, 1);
        int gens = 0;
        while (gens < 300) {
            evaluateXOR(pop);
            double best = 0.0;
            for (unsigned i = 0; i < pop.numGenomes(); ++i) {
                best = std::max(best, pop.accessGenomeByIndex(static_cast<int>(i)).getFitness());
            }
            if (best > 15.0) {
                break;
            }
            pop.epoch();
            ++gens;
        }
        evaluateXOR(pop);
        Genome best = pop.getBestGenome();
        CHECK(xorError(best) < 0.13);
        // m_BestFitnessEver is only updated inside Epoch(); since we stop right
        // after evaluation, just require it tracked the earlier generations.
        CHECK(pop.getBestFitnessEver() > 0.0);
        pop.sameGenomeIDCheck();
    }

    // Determinism: identical seeds replay identical best fitness trajectories.
    {
        auto firstGensBest = [](unsigned long seed, unsigned gens) {
            Population pop(makeXorSeed(), xorParams(), true, 1.0, static_cast<int>(seed));
            double best = 0.0;
            for (unsigned g = 0; g < gens; ++g) {
                evaluateXOR(pop);
                for (unsigned i = 0; i < pop.numGenomes(); ++i) {
                    best = std::max(best, pop.accessGenomeByIndex(static_cast<int>(i)).getFitness());
                }
                pop.epoch();
            }
            return best;
        };
        CHECK(firstGensBest(77, 40) == firstGensBest(77, 40));
    }

    if (g_failures != 0) {
        std::cerr << "Test failed: TestEvolution with " << g_failures << " failure(s)\n";
        return 1;
    }
    std::cout << "Test passed: TestEvolution\n";
    return 0;
}
