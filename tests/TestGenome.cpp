// Tests for NEAT::Genome: structure, compatibility, phenotype, mutations.
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>

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
        // NOTE: named RNG required here; binding a temporary to the
        // non-const RNG& parameter is an MSVC extension GCC rejects.
        RNG rng;
        rng.Seed(3);
        g.Randomize_LinkWeights(DefaultParams(), rng);
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

    // Missing genome file throws (no hang).
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

    // Regression: a file without markers must throw instead of spinning on EOF forever.
    {
        const auto tmp = std::filesystem::temp_directory_path() / "neatcpp_test_garbage_genome.txt";
        {
            std::ofstream out(tmp);
            out << "this file has no markers at all\n";
        }
        bool threw = false;
        try {
            Genome g(tmp.string().c_str());
            (void)g;
        } catch (...) {
            threw = true;
        }
        CHECK(threw);
        std::error_code ec;
        std::filesystem::remove(tmp, ec);
    }

    // Regression: truncated genome (GenomeStart but no GenomeEnd) must throw.
    {
        Genome g = MakeSeed();
        const auto src = std::filesystem::temp_directory_path() / "neatcpp_test_good_genome.txt";
        const auto trunc = std::filesystem::temp_directory_path() / "neatcpp_test_trunc_genome.txt";
        g.Save(src.string().c_str());
        std::ifstream in(src);
        std::string body((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
        body.erase(body.find("GenomeEnd"));
        {
            std::ofstream out(trunc);
            out << body;
        }
        bool threw = false;
        try {
            Genome g2(trunc.string().c_str());
            (void)g2;
        } catch (...) {
            threw = true;
        }
        CHECK(threw);
        std::error_code ec;
        std::filesystem::remove(src, ec);
        std::filesystem::remove(trunc, ec);
    }

    // Regression: Mutate_RemoveLink removes exactly one link and keeps the
    // genome consistent. The underlying RemoveLinkGene used to erase by
    // position: a requested innovation ID of 0 wiped every link and larger IDs
    // could erase unrelated entries.
    {
        RNG rng;
        rng.Seed(11);
        Genome g = MakeSeed(3, 2);
        const unsigned n0 = g.NumLinks();
        std::vector<int> before;
        for (unsigned i = 0; i < g.NumLinks(); ++i) {
            before.push_back(g.GetLinkByIndex(static_cast<int>(i)).InnovationID());
        }
        CHECK(g.Mutate_RemoveLink(rng));
        CHECK(g.NumLinks() == n0 - 1);
        // Remaining links are the original ones minus exactly one, order preserved.
        size_t j = 0;
        int missing = -1;
        for (int id : before) {
            if (j < g.NumLinks() && g.GetLinkByIndex(static_cast<int>(j)).InnovationID() == id) {
                ++j;
            } else {
                missing = id;
            }
        }
        CHECK(missing != -1);
        CHECK(j == static_cast<size_t>(g.NumLinks()));
        CHECK(!g.HasDeadEnds());
    }

    // Regression: Mutate_RemoveSimpleNeuron drops the hidden neuron and its
    // links without erasing wrong positions or leaving dangling links.
    {
        Parameters p = DefaultParams();
        RNG rng;
        rng.Seed(7);
        InnovationDatabase innovs;
        innovs.Init(1, 1000);
        Genome g = MakeSeed();
        for (int i = 0; i < 50 && g.NumNeurons() == 4; ++i) {
            g.Mutate_AddNeuron(innovs, p, rng);
        }
        CHECK(g.NumNeurons() > 4);
        bool removed = false;
        for (int i = 0; i < 50 && !removed; ++i) {
            removed = g.Mutate_RemoveSimpleNeuron(innovs, p, rng);
        }
        CHECK(removed);
        CHECK(g.NumNeurons() == 4);
        // Every link still references existing neurons.
        std::vector<int> neuron_ids;
        for (unsigned i = 0; i < g.NumNeurons(); ++i) {
            neuron_ids.push_back(g.GetNeuronByIndex(static_cast<int>(i)).ID());
        }
        for (unsigned i = 0; i < g.NumLinks(); ++i) {
            const LinkGene &l = g.GetLinkByIndex(static_cast<int>(i));
            CHECK(std::find(neuron_ids.begin(), neuron_ids.end(), l.FromNeuronID()) != neuron_ids.end());
            CHECK(std::find(neuron_ids.begin(), neuron_ids.end(), l.ToNeuronID()) != neuron_ids.end());
        }
    }

    // Regression: with MultipointCrossoverRate = PreferFitterParentRate = 1,
    // matching genes must come from the *fitter* parent (was inverted: picked
    // the weaker one).
    {
        Parameters p = DefaultParams();
        p.MultipointCrossoverRate = 1.0;
        p.PreferFitterParentRate = 1.0;
        RNG rng;
        rng.Seed(31);
        Genome mom = MakeSeed(3, 2);
        Genome dad = MakeSeed(3, 2);
        for (int i = 0; i < 10; ++i) {
            mom.Mutate_LinkWeights(p, rng);
            dad.Mutate_LinkWeights(p, rng);
        }
        bool any_differ = false;
        for (unsigned i = 0; i < mom.NumLinks(); ++i) {
            if (mom.GetLinkByIndex(static_cast<int>(i)).GetWeight() != dad.GetLinkByIndex(static_cast<int>(i)).GetWeight()) {
                any_differ = true;
            }
        }
        CHECK(any_differ);
        mom.SetFitness(10.0);
        dad.SetFitness(1.0);
        Genome baby = mom.Mate(dad, false, false, rng, p);
        CHECK(baby.NumLinks() == mom.NumLinks());
        for (unsigned i = 0; i < baby.NumLinks(); ++i) {
            const double wm = mom.GetLinkByIndex(static_cast<int>(i)).GetWeight();
            const double wd = dad.GetLinkByIndex(static_cast<int>(i)).GetWeight();
            if (wm != wd) {
                CHECK(Near(baby.GetLinkByIndex(static_cast<int>(i)).GetWeight(), wm));
            }
        }
    }

    // Compatibility distance: zero to self, grows with disjoint genes, and
    // IsCompatibleWith agrees with the threshold.
    {
        Parameters p = DefaultParams();
        RNG rng;
        rng.Seed(99);
        Genome a = MakeSeed(3, 2);
        Genome b = MakeSeed(3, 2);
        CHECK(Near(a.CompatibilityDistance(a, p), 0.0));
        CHECK(Near(a.CompatibilityDistance(b, p), 0.0));  // identical topology
        CHECK(a.IsCompatibleWith(b, p));

        // Remove links from b only: each removal makes b missing a gene that a
        // has, i.e. adds disjoint genes to the pair, so the distance strictly grows.
        double prev = 0.0;
        int removed = 0;
        for (int i = 0; i < 3 && b.NumLinks() > 0; ++i) {
            if (b.Mutate_RemoveLink(rng)) {
                ++removed;
                const double d = a.CompatibilityDistance(b, p);
                CHECK(d > prev);
                prev = d;
            }
        }
        CHECK(removed == 3);
        CHECK(a.IsCompatibleWith(b, p) == (prev <= p.CompatTreshold));
    }

    // DerivePhenotypicChanges copies network weights back into the genome.
    {
        Parameters p = DefaultParams();
        Genome g = MakeSeed(3, 2);
        NeuralNetwork net;
        g.BuildPhenotype(net);
        CHECK(net.m_connections.size() == static_cast<size_t>(g.NumLinks()));
        for (unsigned i = 0; i < net.m_connections.size(); ++i) {
            net.m_connections[i].m_weight = 0.25 * static_cast<double>(i) - 1.0;
        }
        g.DerivePhenotypicChanges(net);
        for (unsigned i = 0; i < net.m_connections.size(); ++i) {
            CHECK(Near(g.GetLinkByIndex(static_cast<int>(i)).GetWeight(), net.m_connections[i].m_weight));
        }
    }

    // Phenotype outputs are deterministic and match a freshly built network.
    {
        Parameters p = DefaultParams();
        Genome g = MakeSeed(3, 1);
        NeuralNetwork net1, net2;
        g.BuildPhenotype(net1);
        g.BuildPhenotype(net2);
        std::vector<double> in{0.3, 0.6, 0.9};
        net1.Input(in);
        net1.Activate();
        net2.Input(in);
        net2.Activate();
        CHECK(Near(net1.Output()[0], net2.Output()[0]));
    }

    if (g_failures != 0) {
        std::cerr << "Test failed: TestGenome with " << g_failures << " failure(s)\n";
        return 1;
    }
    std::cout << "Test passed: TestGenome\n";
    return 0;
}
