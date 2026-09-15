// Tests for NEAT::Genome: structure, compatibility, phenotype, mutations.
//
// CTest-Labels: Evolution;Fast
// CTest-Timeout: 120
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

    bool near(double a, double b, double eps = 1e-9) { return std::fabs(a - b) <= eps; }

    NEAT::Parameters defaultParams() {
        NEAT::Parameters p;
        p.reset();
        return p;
    }

    NEAT::Genome makeSeed(int numInputs = 3, int numOutputs = 1) {
        NEAT::Parameters p = defaultParams();
        NEAT::GenomeInitStruct init;
        init.numInputs = numInputs;
        init.numOutputs = numOutputs;
        init.seedType = NEAT::PERCEPTRON;
        return NEAT::Genome(p, init);
    }

}  // namespace

int TestGenome(int argc, char *argv[]) {
    (void)argc;
    (void)argv;
    using namespace NEAT;

    // Seed structure: fully connected perceptron, deterministic counts.
    {
        Genome g = makeSeed(3, 2);
        CHECK(g.numInputs() == 3);
        CHECK(g.numOutputs() == 2);
        CHECK(g.numNeurons() == 5);  // 3 in/bias + 2 out
        CHECK(g.numLinks() == 6);    // 3*2
        // GetLast* return the next free ID (max + 1), not the max itself.
        CHECK(g.getLastNeuronID() == 6);
        CHECK(g.getLastInnovationID() == 7);
        CHECK(!g.hasDeadEnds());
        CHECK(!g.hasLoops());
        Parameters p = defaultParams();
        CHECK(!g.failsConstraints(p));
        g.calculateDepth();
        CHECK(g.getDepth() >= 1);
    }

    // Copy/assign preserve structure; == compares ID only.
    {
        Genome a = makeSeed();
        Genome b = a;
        CHECK(b.numNeurons() == a.numNeurons());
        CHECK(b.numLinks() == a.numLinks());
        CHECK(b == a);
        b.setID(a.getID() + 100);
        CHECK(!(b == a));
        Genome c;
        c = a;
        CHECK(c == a);
    }

    // Compatibility: clone distance is 0 and compatible; fitness accessors work.
    {
        Parameters p = defaultParams();
        Genome a = makeSeed();
        Genome b = a;
        CHECK(near(a.compatibilityDistance(b, p), 0.0));
        CHECK(a.isCompatibleWith(b, p));
        a.setFitness(2.5);
        CHECK(near(a.getFitness(), 2.5));
        a.setAdjFitness(1.25);
        CHECK(near(a.getAdjFitness(), 1.25));
        CHECK(!a.isEvaluated());
        a.setEvaluated();
        CHECK(a.isEvaluated());
        a.resetEvaluated();
        CHECK(!a.isEvaluated());
    }

    // BuildPhenotype mirrors genome size and runs.
    {
        Genome g = makeSeed(3, 1);
        NeuralNetwork net;
        g.buildPhenotype(net);
        CHECK(net.numInputs() == 3 && net.numOutputs() == 1);
        CHECK(net.neurons_.size() == g.numNeurons());
        CHECK(net.connections_.size() == g.numLinks());
        std::vector<double> in{0.5, -0.5, 1.0};
        net.flush();
        net.input(in);
        net.activate();
        CHECK(net.output().size() == 1);
        CHECK(!std::isnan(net.output()[0]));
    }

    // SortGenes orders links by innovation ID; Cleanup on a healthy seed is a no-op.
    {
        Genome g = makeSeed();
        g.sortGenes();
        for (unsigned i = 1; i < g.numLinks(); ++i) {
            CHECK(g.getLinkByIndex(i - 1).innovationID() <= g.getLinkByIndex(i).innovationID());
        }
        const unsigned nn = g.numNeurons(), nl = g.numLinks();
        (void)g.cleanup();
        CHECK(g.numNeurons() <= nn && g.numLinks() <= nl);
        // Lookup helpers agree.
        CHECK(g.getNeuronIndex(g.getNeuronByIndex(0).id()) == 0);
        CHECK(g.getLinkIndex(g.getLinkByIndex(0).innovationID()) == 0);
        CHECK(g.getLastNeuronID() > 0 && g.getLastInnovationID() > 0);
    }

    // Seeded structural mutations: AddNeuron / AddLink succeed given retries.
    {
        Parameters p = defaultParams();
        RNG rng;
        rng.seed(123);
        InnovationDatabase innovs;
        innovs.init(1, 1000);
        Genome g = makeSeed();
        const unsigned nn0 = g.numNeurons(), nl0 = g.numLinks();
        bool addedNeuron = false;
        for (int i = 0; i < 50 && !addedNeuron; ++i) {
            addedNeuron = g.mutateAddNeuron(innovs, p, rng);
        }
        CHECK(addedNeuron);
        CHECK(g.numNeurons() == nn0 + 1);
        CHECK(g.numLinks() > nl0);

        bool addedLink = false;
        for (int i = 0; i < 50 && !addedLink; ++i) {
            addedLink = g.mutateAddLink(innovs, p, rng);
        }
        CHECK(addedLink);

        // Weight perturbation keeps weights finite.
        CHECK(g.mutateLinkWeights(p, rng) || true);  // may no-op; just must not crash
        for (unsigned i = 0; i < g.numLinks(); ++i) {
            CHECK(std::isfinite(g.getLinkByIndex(i).getWeight()));
        }
    }

    // Mate of two clones with seeded RNG yields a valid baby.
    {
        Parameters p = defaultParams();
        RNG rng;
        rng.seed(7);
        Genome mom = makeSeed();
        Genome dad = makeSeed();
        mom.setFitness(2.0);
        dad.setFitness(1.0);
        Genome baby = mom.mate(dad, false, false, rng, p);
        CHECK(baby.numNeurons() > 0 && baby.numLinks() > 0);
        CHECK(baby.numInputs() == mom.numInputs() && baby.numOutputs() == mom.numOutputs());
    }

    // Save/Load round-trip preserves topology, weights, and ID.
    {
        Genome g = makeSeed(3, 1);
        g.setID(4242);
        // NOTE: named RNG required here; binding a temporary to the
        // non-const RNG& parameter is an MSVC extension GCC rejects.
        RNG rng;
        rng.seed(3);
        g.randomizeLinkWeights(defaultParams(), rng);
        const std::filesystem::path tmp = std::filesystem::temp_directory_path() / "multineat_test_genome.txt";
        g.save(tmp.string().c_str());

        Genome loaded(tmp.string().c_str());
        CHECK(loaded.getID() == 4242);
        CHECK(loaded.numNeurons() == g.numNeurons());
        CHECK(loaded.numLinks() == g.numLinks());
        for (unsigned i = 0; i < g.numLinks(); ++i) {
            CHECK(loaded.getLinkByIndex(i).innovationID() == g.getLinkByIndex(i).innovationID());
            // Genome::Save uses %3.8f, so allow float-printing tolerance.
            CHECK(near(loaded.getLinkByIndex(i).getWeight(), g.getLinkByIndex(i).getWeight(), 1e-6));
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
        const std::filesystem::path tmp = std::filesystem::temp_directory_path() / "neatcpp_test_garbage_genome.txt";
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
        Genome g = makeSeed();
        const std::filesystem::path src = std::filesystem::temp_directory_path() / "neatcpp_test_good_genome.txt";
        const std::filesystem::path trunc = std::filesystem::temp_directory_path() / "neatcpp_test_trunc_genome.txt";
        g.save(src.string().c_str());
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
        rng.seed(11);
        Genome g = makeSeed(3, 2);
        const unsigned n0 = g.numLinks();
        std::vector<int> before;
        for (unsigned i = 0; i < g.numLinks(); ++i) {
            before.push_back(g.getLinkByIndex(static_cast<int>(i)).innovationID());
        }
        CHECK(g.mutateRemoveLink(rng));
        CHECK(g.numLinks() == n0 - 1);
        // Remaining links are the original ones minus exactly one, order preserved.
        size_t j = 0;
        int missing = -1;
        for (int id : before) {
            if (j < g.numLinks() && g.getLinkByIndex(static_cast<int>(j)).innovationID() == id) {
                ++j;
            } else {
                missing = id;
            }
        }
        CHECK(missing != -1);
        CHECK(j == static_cast<size_t>(g.numLinks()));
        CHECK(!g.hasDeadEnds());
    }

    // Regression: Mutate_RemoveSimpleNeuron drops the hidden neuron and its
    // links without erasing wrong positions or leaving dangling links.
    {
        Parameters p = defaultParams();
        RNG rng;
        rng.seed(7);
        InnovationDatabase innovs;
        innovs.init(1, 1000);
        Genome g = makeSeed();
        for (int i = 0; i < 50 && g.numNeurons() == 4; ++i) {
            g.mutateAddNeuron(innovs, p, rng);
        }
        CHECK(g.numNeurons() > 4);
        bool removed = false;
        for (int i = 0; i < 50 && !removed; ++i) {
            removed = g.mutateRemoveSimpleNeuron(innovs, p, rng);
        }
        CHECK(removed);
        CHECK(g.numNeurons() == 4);
        // Every link still references existing neurons.
        std::vector<int> neuronIds;
        for (unsigned i = 0; i < g.numNeurons(); ++i) {
            neuronIds.push_back(g.getNeuronByIndex(static_cast<int>(i)).id());
        }
        for (unsigned i = 0; i < g.numLinks(); ++i) {
            const LinkGene &l = g.getLinkByIndex(static_cast<int>(i));
            CHECK(std::find(neuronIds.begin(), neuronIds.end(), l.fromNeuronID()) != neuronIds.end());
            CHECK(std::find(neuronIds.begin(), neuronIds.end(), l.toNeuronID()) != neuronIds.end());
        }
    }

    // Regression: with MultipointCrossoverRate = PreferFitterParentRate = 1,
    // matching genes must come from the *fitter* parent (was inverted: picked
    // the weaker one).
    {
        Parameters p = defaultParams();
        p.multipointCrossoverRate = 1.0;
        p.preferFitterParentRate = 1.0;
        RNG rng;
        rng.seed(31);
        Genome mom = makeSeed(3, 2);
        Genome dad = makeSeed(3, 2);
        for (int i = 0; i < 10; ++i) {
            mom.mutateLinkWeights(p, rng);
            dad.mutateLinkWeights(p, rng);
        }
        bool anyDiffer = false;
        for (unsigned i = 0; i < mom.numLinks(); ++i) {
            if (mom.getLinkByIndex(static_cast<int>(i)).getWeight() != dad.getLinkByIndex(static_cast<int>(i)).getWeight()) {
                anyDiffer = true;
            }
        }
        CHECK(anyDiffer);
        mom.setFitness(10.0);
        dad.setFitness(1.0);
        Genome baby = mom.mate(dad, false, false, rng, p);
        CHECK(baby.numLinks() == mom.numLinks());
        for (unsigned i = 0; i < baby.numLinks(); ++i) {
            const double wm = mom.getLinkByIndex(static_cast<int>(i)).getWeight();
            const double wd = dad.getLinkByIndex(static_cast<int>(i)).getWeight();
            if (wm != wd) {
                CHECK(near(baby.getLinkByIndex(static_cast<int>(i)).getWeight(), wm));
            }
        }
    }

    // Compatibility distance: zero to self, grows with disjoint genes, and
    // IsCompatibleWith agrees with the threshold.
    {
        Parameters p = defaultParams();
        RNG rng;
        rng.seed(99);
        Genome a = makeSeed(3, 2);
        Genome b = makeSeed(3, 2);
        CHECK(near(a.compatibilityDistance(a, p), 0.0));
        CHECK(near(a.compatibilityDistance(b, p), 0.0));  // identical topology
        CHECK(a.isCompatibleWith(b, p));

        // Remove links from b only: each removal makes b missing a gene that a
        // has, i.e. adds disjoint genes to the pair, so the distance strictly grows.
        double prev = 0.0;
        int removed = 0;
        for (int i = 0; i < 3 && b.numLinks() > 0; ++i) {
            if (b.mutateRemoveLink(rng)) {
                ++removed;
                const double d = a.compatibilityDistance(b, p);
                CHECK(d > prev);
                prev = d;
            }
        }
        CHECK(removed == 3);
        CHECK(a.isCompatibleWith(b, p) == (prev <= p.compatTreshold));
    }

    // DerivePhenotypicChanges copies network weights back into the genome.
    {
        Parameters p = defaultParams();
        Genome g = makeSeed(3, 2);
        NeuralNetwork net;
        g.buildPhenotype(net);
        CHECK(net.connections_.size() == static_cast<size_t>(g.numLinks()));
        for (unsigned i = 0; i < net.connections_.size(); ++i) {
            net.connections_[i].weight_ = 0.25 * static_cast<double>(i) - 1.0;
        }
        g.derivePhenotypicChanges(net);
        for (unsigned i = 0; i < net.connections_.size(); ++i) {
            CHECK(near(g.getLinkByIndex(static_cast<int>(i)).getWeight(), net.connections_[i].weight_));
        }
    }

    // Phenotype outputs are deterministic and match a freshly built network.
    {
        Parameters p = defaultParams();
        Genome g = makeSeed(3, 1);
        NeuralNetwork net1, net2;
        g.buildPhenotype(net1);
        g.buildPhenotype(net2);
        std::vector<double> in{0.3, 0.6, 0.9};
        net1.input(in);
        net1.activate();
        net2.input(in);
        net2.activate();
        CHECK(near(net1.output()[0], net2.output()[0]));
    }

    if (g_failures != 0) {
        std::cerr << "Test failed: TestGenome with " << g_failures << " failure(s)\n";
        return 1;
    }
    std::cout << "Test passed: TestGenome\n";
    return 0;
}
