// Tests for NEAT::Genome: structure, compatibility, phenotype, mutations.
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>

#include "Genome.h"
#include "Innovation.h"
#include "NeuralNetwork.h"
#include "Parameters.h"
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

    bool Near(double a, double b, double eps = 1e-9) { return std::fabs(a - b) <= eps; }

    NEAT::Parameters DefaultParams() {
        NEAT::Parameters p;
        p.Reset();
        return p;
    }

    NEAT::Genome MakeSeed(int num_inputs = 3, int num_outputs = 1) {
        NEAT::Parameters p = DefaultParams();
        NEAT::GenomeInitStruct init;
        init.NumInputs = num_inputs;
        init.NumOutputs = num_outputs;
        init.SeedType = NEAT::PERCEPTRON;
        return NEAT::Genome(p, init);
    }

}  // namespace

int TestGenome(int argc, char *argv[]) {
    (void)argc;
    (void)argv;
    using namespace NEAT;

    // Seed structure: fully connected perceptron, deterministic counts.
    {
        Genome g = MakeSeed(3, 2);
        CHECK(g.NumInputs() == 3);
        CHECK(g.NumOutputs() == 2);
        CHECK(g.NumNeurons() == 5);  // 3 in/bias + 2 out
        CHECK(g.NumLinks() == 6);    // 3*2
        // GetLast* return the next free ID (max + 1), not the max itself.
        CHECK(g.GetLastNeuronID() == 6);
        CHECK(g.GetLastInnovationID() == 7);
        CHECK(!g.HasDeadEnds());
        CHECK(!g.HasLoops());
        Parameters p = DefaultParams();
        CHECK(!g.FailsConstraints(p));
        g.CalculateDepth();
        CHECK(g.GetDepth() >= 1);
    }

    // Copy/assign preserve structure; == compares ID only.
    {
        Genome a = MakeSeed();
        Genome b = a;
        CHECK(b.NumNeurons() == a.NumNeurons());
        CHECK(b.NumLinks() == a.NumLinks());
        CHECK(b == a);
        b.SetID(a.GetID() + 100);
        CHECK(!(b == a));
        Genome c;
        c = a;
        CHECK(c == a);
    }

    // Compatibility: clone distance is 0 and compatible; fitness accessors work.
    {
        Parameters p = DefaultParams();
        Genome a = MakeSeed();
        Genome b = a;
        CHECK(Near(a.CompatibilityDistance(b, p), 0.0));
        CHECK(a.IsCompatibleWith(b, p));
        a.SetFitness(2.5);
        CHECK(Near(a.GetFitness(), 2.5));
        a.SetAdjFitness(1.25);
        CHECK(Near(a.GetAdjFitness(), 1.25));
        CHECK(!a.IsEvaluated());
        a.SetEvaluated();
        CHECK(a.IsEvaluated());
        a.ResetEvaluated();
        CHECK(!a.IsEvaluated());
    }

    // BuildPhenotype mirrors genome size and runs.
    {
        Genome g = MakeSeed(3, 1);
        NeuralNetwork net;
        g.BuildPhenotype(net);
        CHECK(net.NumInputs() == 3 && net.NumOutputs() == 1);
        CHECK(net.m_neurons.size() == g.NumNeurons());
        CHECK(net.m_connections.size() == g.NumLinks());
        std::vector<double> in{0.5, -0.5, 1.0};
        net.Flush();
        net.Input(in);
        net.Activate();
        CHECK(net.Output().size() == 1);
        CHECK(!std::isnan(net.Output()[0]));
    }

    // SortGenes orders links by innovation ID; Cleanup on a healthy seed is a no-op.
    {
        Genome g = MakeSeed();
        g.SortGenes();
        for (unsigned i = 1; i < g.NumLinks(); ++i) {
            CHECK(g.GetLinkByIndex(i - 1).InnovationID() <= g.GetLinkByIndex(i).InnovationID());
        }
        const unsigned nn = g.NumNeurons(), nl = g.NumLinks();
        (void)g.Cleanup();
        CHECK(g.NumNeurons() <= nn && g.NumLinks() <= nl);
        // Lookup helpers agree.
        CHECK(g.GetNeuronIndex(g.GetNeuronByIndex(0).ID()) == 0);
        CHECK(g.GetLinkIndex(g.GetLinkByIndex(0).InnovationID()) == 0);
        CHECK(g.GetLastNeuronID() > 0 && g.GetLastInnovationID() > 0);
    }

    // Seeded structural mutations: AddNeuron / AddLink succeed given retries.
    {
        Parameters p = DefaultParams();
        RNG rng;
        rng.Seed(123);
        InnovationDatabase innovs;
        innovs.Init(1, 1000);
        Genome g = MakeSeed();
        const unsigned nn0 = g.NumNeurons(), nl0 = g.NumLinks();
        bool added_neuron = false;
        for (int i = 0; i < 50 && !added_neuron; ++i) {
            added_neuron = g.Mutate_AddNeuron(innovs, p, rng);
        }
        CHECK(added_neuron);
        CHECK(g.NumNeurons() == nn0 + 1);
        CHECK(g.NumLinks() > nl0);

        bool added_link = false;
        for (int i = 0; i < 50 && !added_link; ++i) {
            added_link = g.Mutate_AddLink(innovs, p, rng);
        }
        CHECK(added_link);

        // Weight perturbation keeps weights finite.
        CHECK(g.Mutate_LinkWeights(p, rng) || true);  // may no-op; just must not crash
        for (unsigned i = 0; i < g.NumLinks(); ++i) {
            CHECK(std::isfinite(g.GetLinkByIndex(i).GetWeight()));
        }
    }

    // Mate of two clones with seeded RNG yields a valid baby.
    {
        Parameters p = DefaultParams();
        RNG rng;
        rng.Seed(7);
        Genome mom = MakeSeed();
        Genome dad = MakeSeed();
        mom.SetFitness(2.0);
        dad.SetFitness(1.0);
        Genome baby = mom.Mate(dad, false, false, rng, p);
        CHECK(baby.NumNeurons() > 0 && baby.NumLinks() > 0);
        CHECK(baby.NumInputs() == mom.NumInputs() && baby.NumOutputs() == mom.NumOutputs());
    }

    // Save/Load round-trip preserves topology, weights, and ID.
    {
        Genome g = MakeSeed(3, 1);
        g.SetID(4242);
        g.Randomize_LinkWeights(DefaultParams(), [] {
            RNG r;
            r.Seed(3);
            return r;
        }());
        const auto tmp = std::filesystem::temp_directory_path() / "multineat_test_genome.txt";
        g.Save(tmp.string().c_str());

        Genome loaded(tmp.string().c_str());
        CHECK(loaded.GetID() == 4242);
        CHECK(loaded.NumNeurons() == g.NumNeurons());
        CHECK(loaded.NumLinks() == g.NumLinks());
        for (unsigned i = 0; i < g.NumLinks(); ++i) {
            CHECK(loaded.GetLinkByIndex(i).InnovationID() == g.GetLinkByIndex(i).InnovationID());
            // Genome::Save uses %3.8f, so allow float-printing tolerance.
            CHECK(Near(loaded.GetLinkByIndex(i).GetWeight(), g.GetLinkByIndex(i).GetWeight(), 1e-6));
        }
        std::error_code ec;
        std::filesystem::remove(tmp, ec);
    }

    // Missing genome file throws (no hang, unlike garbage without markers).
    {
        bool threw = false;
        try {
            Genome bad("/nonexistent/path/that/should/not/exist.genome");
            (void)bad;
        } catch (const std::runtime_error &) {
            threw = true;
        } catch (...) {
            threw = true;
        }
        CHECK(threw);
    }

    if (g_failures != 0) {
        std::cerr << "Test failed: TestGenome with " << g_failures << " failure(s)\n";
        return 1;
    }
    std::cout << "Test passed: TestGenome\n";
    return 0;
}
