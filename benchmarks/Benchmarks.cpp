// Micro/macro benchmarks for NEAT.cpp. Standalone executable; prints one line
// per benchmark: <name> <elapsed_ms> (<ops> ops, <ns_per_op> ns/op).
//
// Build:  cmake -S . -B build/bench -DCMAKE_BUILD_TYPE=Release -DNEATCPP_ENABLE_BENCHMARKS=ON
// Run:    ./build/bench/benchmarks/NEATcppBench
//
// Seeded and fixed-iteration so runs are comparable across changes;
// compare output against RESULTS.md baselines.
#include <chrono>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "Genome.h"
#include "Innovation.h"
#include "NeuralNetwork.h"
#include "Parameters.h"
#include "Population.h"
#include "Random.h"

using namespace NEAT;
using Clock = std::chrono::steady_clock;

namespace {

    struct Result {
        const char *name;
        double totalMs;
        unsigned long ops;
    };
    std::vector<Result> g_results;

    void report(const char *name, double totalMs, unsigned long ops) {
        const double nsPerOp = (totalMs * 1e6) / static_cast<double>(ops);
        std::printf("%-28s %10.2f ms total  %8lu ops  %10.1f ns/op\n", name, totalMs, ops, nsPerOp);
        g_results.push_back({name, totalMs, ops});
    }

    Parameters defaultParams() {
        Parameters p;
        p.reset();
        return p;
    }

    Genome seedGenome(int numInputs, int numOutputs) {
        Parameters p = defaultParams();
        GenomeInitStruct init;
        init.numInputs = numInputs;
        init.numOutputs = numOutputs;
        init.seedType = PERCEPTRON;
        return Genome(p, init);
    }

    // Grow a large genome: num_inputs inputs + num_outputs outputs + many hidden
    // neurons/links, representative of a complexified NEAT topology.
    Genome largeGenome(int numInputs, int numOutputs, int targetHidden, unsigned seed) {
        Parameters p = defaultParams();
        RNG rng;
        rng.seed(seed);
        InnovationDatabase innovs;
        innovs.init(1, 1000);
        Genome g = seedGenome(numInputs, numOutputs);
        while (g.numNeurons() - numInputs - numOutputs < targetHidden) {
            if (!g.mutateAddNeuron(innovs, p, rng)) {
                break;
            }
        }
        for (int i = 0; i < targetHidden * 4; ++i) {
            g.mutateAddLink(innovs, p, rng);
        }
        return g;
    }

    void benchPhenotypeBuildSmall() {
        Genome g = seedGenome(10, 3);
        const unsigned ops = 200000;
        NeuralNetwork net;
        const Clock::time_point t0 = Clock::now();
        for (unsigned i = 0; i < ops; ++i) {
            g.buildPhenotype(net);
        }
        const Clock::time_point t1 = Clock::now();
        report("BuildPhenotype small", std::chrono::duration<double, std::milli>(t1 - t0).count(), ops);
    }

    void benchPhenotypeBuildLarge() {
        Genome g = largeGenome(10, 3, 200, 11);
        const unsigned ops = 2000;
        NeuralNetwork net;
        const Clock::time_point t0 = Clock::now();
        for (unsigned i = 0; i < ops; ++i) {
            g.buildPhenotype(net);
        }
        const Clock::time_point t1 = Clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        report("BuildPhenotype large", ms, ops);
    }

    void benchActivate() {
        Genome g = largeGenome(10, 3, 200, 12);
        NeuralNetwork net;
        g.buildPhenotype(net);
        std::vector<double> in(net.numInputs_, 0.5);
        const unsigned ops = 20000;
        const Clock::time_point t0 = Clock::now();
        for (unsigned i = 0; i < ops; ++i) {
            net.flush();
            net.input(in);
            net.activate();
        }
        const Clock::time_point t1 = Clock::now();
        report("Activate large net", std::chrono::duration<double, std::milli>(t1 - t0).count(), ops);
    }

    void benchCompatibilityDistance() {
        Genome a = largeGenome(10, 3, 200, 13);
        Genome b = largeGenome(10, 3, 200, 14);
        Parameters p = defaultParams();
        const unsigned ops = 20000;
        volatile double sink = 0.0;
        const Clock::time_point t0 = Clock::now();
        for (unsigned i = 0; i < ops; ++i) {
            sink += a.compatibilityDistance(b, p);
        }
        const Clock::time_point t1 = Clock::now();
        report("CompatibilityDistance", std::chrono::duration<double, std::milli>(t1 - t0).count(), ops);
    }

    void benchMutateGenome() {
        Parameters p = defaultParams();
        RNG rng;
        rng.seed(15);
        InnovationDatabase innovs;
        innovs.init(1, 1000);
        Genome g = largeGenome(10, 3, 200, 16);
        const unsigned ops = 50000;
        const Clock::time_point t0 = Clock::now();
        for (unsigned i = 0; i < ops; ++i) {
            Genome copy(g);
            copy.mutateLinkWeights(p, rng);
            copy.mutateAddLink(innovs, p, rng);
        }
        const Clock::time_point t1 = Clock::now();
        report("Copy+mutate genome", std::chrono::duration<double, std::milli>(t1 - t0).count(), ops);
    }

    void benchEpoch() {
        // One full Epoch of a small XOR-style population (the dominant per-generation cost).
        Parameters p;
        p.populationSize = 100;
        p.dynamicCompatibility = true;
        p.compatTreshold = 2.0;
        p.crossoverRate = 0.0;
        p.survivalRate = 0.2;
        Genome g = seedGenome(3, 1);
        Population pop(g, p, true, 1.0, 21);
        const unsigned ops = 100;
        const Clock::time_point t0 = Clock::now();
        for (unsigned i = 0; i < ops; ++i) {
            for (unsigned j = 0; j < pop.numGenomes(); ++j) {
                Genome &gg = pop.accessGenomeByIndex(static_cast<int>(j));
                gg.setFitness(1.0 + static_cast<double>(gg.numLinks()));
                gg.setEvaluated();
            }
            pop.epoch();
        }
        const Clock::time_point t1 = Clock::now();
        report("Epoch pop100", std::chrono::duration<double, std::milli>(t1 - t0).count(), ops);
    }

    void benchXorSolve() {
        // The reference XOR recipe from upstream MultiNEAT: generation count and
        // wall time until best fitness > 15.0 (seeded, deterministic).
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

        Parameters q;
        q.reset();
        GenomeInitStruct init;
        init.numInputs = 3;
        init.numOutputs = 1;
        init.seedType = PERCEPTRON;
        Population pop(Genome(q, init), p, true, 1.0, 1);

        const double xi[4][2] = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
        const double xt[4] = {0, 1, 1, 0};

        const Clock::time_point t0 = Clock::now();
        unsigned gen = 0;
        for (; gen < 300; ++gen) {
            for (unsigned i = 0; i < pop.numGenomes(); ++i) {
                Genome &gg = pop.accessGenomeByIndex(static_cast<int>(i));
                NeuralNetwork net;
                gg.buildPhenotype(net);
                double err = 0.0;
                for (int pat = 0; pat < 4; ++pat) {
                    net.flush();
                    std::vector<double> in = {xi[pat][0], xi[pat][1], 1.0};
                    net.input(in);
                    net.activate();
                    net.activate();
                    err += std::fabs(net.output()[0] - xt[pat]);
                }
                const double rem = 4.0 - err;
                gg.setFitness(rem * rem);
                gg.setEvaluated();
            }
            if (pop.getBestFitnessEver() > 15.0) {
                break;
            }
            pop.epoch();
        }
        const Clock::time_point t1 = Clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        std::printf("(xor solved at generation %u)\n", gen);
        report("XOR solve seed1", ms, 1);
    }

    void benchGenomeSaveLoad() {
        Genome g = largeGenome(10, 3, 200, 17);
        const std::string path = std::string("/tmp/neatcpp_bench_genome.txt");
        const unsigned ops = 100;
        const Clock::time_point t0 = Clock::now();
        for (unsigned i = 0; i < ops; ++i) {
            g.save(path.c_str());
            Genome g2(path.c_str());
        }
        const Clock::time_point t1 = Clock::now();
        report("Genome save+load", std::chrono::duration<double, std::milli>(t1 - t0).count(), ops);
        std::remove(path.c_str());
    }

    void benchPopulationSaveLoad() {
        Parameters p;
        p.populationSize = 100;
        Genome g = seedGenome(10, 3);
        Population pop(g, p, true, 1.0, 22);
        const std::string path = std::string("/tmp/neatcpp_bench_pop.txt");
        const unsigned ops = 20;
        const Clock::time_point t0 = Clock::now();
        for (unsigned i = 0; i < ops; ++i) {
            pop.save(path.c_str());
            Population pop2(path);
        }
        const Clock::time_point t1 = Clock::now();
        report("Population save+load", std::chrono::duration<double, std::milli>(t1 - t0).count(), ops);
        std::remove(path.c_str());
    }

}  // namespace

int main() {
    std::printf("NEAT.cpp benchmarks\n");
    benchPhenotypeBuildSmall();
    benchPhenotypeBuildLarge();
    benchActivate();
    benchCompatibilityDistance();
    benchMutateGenome();
    benchEpoch();
    benchXorSolve();
    benchGenomeSaveLoad();
    benchPopulationSaveLoad();
    return 0;
}
