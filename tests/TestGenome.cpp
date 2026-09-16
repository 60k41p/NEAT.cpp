// Tests for NEAT::Genome: structure, compatibility, phenotype, mutations.
//
// CTest-Labels: Evolution;Fast
// CTest-Timeout: 120
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <vector>

#include "Genome.h"
#include "Innovation.h"
#include "NeuralNetwork.h"
#include "Parameters.h"
#include "Random.h"
#include "Substrate.h"
#include "Traits.h"
using NEAT::Real;

namespace {

    int g_failures = 0;

#define CHECK(cond)                                                                         \
    do {                                                                                    \
        if (!(cond)) {                                                                      \
            std::cerr << "FAILED " << __FILE__ << ":" << __LINE__ << ": " << #cond << "\n"; \
            ++g_failures;                                                                   \
        }                                                                                   \
    } while (0)

    bool Near(Real a, Real b, Real eps = 1e-9) { return std::fabs(a - b) <= eps; }

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
        // GetLast* return the max used ID (v2 semantics); the next free ID is max + 1.
        CHECK(g.GetLastNeuronID() == 5);
        CHECK(g.GetLastInnovationID() == 6);
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
        std::vector<Real> in{0.5, -0.5, 1.0};
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
            const Real wm = mom.GetLinkByIndex(static_cast<int>(i)).GetWeight();
            const Real wd = dad.GetLinkByIndex(static_cast<int>(i)).GetWeight();
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
        Real prev = 0.0;
        int removed = 0;
        for (int i = 0; i < 3 && b.NumLinks() > 0; ++i) {
            if (b.Mutate_RemoveLink(rng)) {
                ++removed;
                const Real d = a.CompatibilityDistance(b, p);
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
            net.m_connections[i].m_weight = 0.25 * static_cast<Real>(i) - 1.0;
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
        std::vector<Real> in{0.3, 0.6, 0.9};
        net1.Input(in);
        net1.Activate();
        net2.Input(in);
        net2.Activate();
        CHECK(Near(net1.Output()[0], net2.Output()[0]));
    }

    // Serialize/Deserialize round-trips state, traits and spiking parameters.
    {
        Parameters p = DefaultParams();
        RNG rng;
        rng.Seed(11);
        Genome g = MakeSeed(3, 2);
        g.SetID(777);
        g.SetFitness(3.5);
        g.Randomize_SpikingParameters(p, rng);
        const std::string data = g.Serialize();
        CHECK(data.find("GenomeFormat 5") != std::string::npos);
        Genome h = Genome::Deserialize(data);
        CHECK(h.GetID() == 777);
        CHECK(Near(h.GetFitness(), 3.5));
        CHECK(g.IsIdenticalTo(h));
        CHECK(h.Validate());
        // Legacy file Save/Load preserves topology and spiking state
        // (weights keep %3.8f file precision; exact state needs Serialize).
        const auto tmp = std::filesystem::temp_directory_path() / "multineat_test_genome_spiking.txt";
        g.Save(tmp.string().c_str());
        Genome file_loaded(tmp.string().c_str());
        CHECK(file_loaded.GetID() == g.GetID());
        CHECK(file_loaded.NumNeurons() == g.NumNeurons());
        CHECK(file_loaded.NumLinks() == g.NumLinks());
        for (unsigned i = 0; i < g.NumLinks(); ++i) {
            CHECK(Near(file_loaded.GetLinkByIndex(static_cast<int>(i)).GetWeight(), g.GetLinkByIndex(static_cast<int>(i)).GetWeight(), 1e-6));
            CHECK(Near(file_loaded.GetLinkByIndex(static_cast<int>(i)).m_SynapticDelay, g.GetLinkByIndex(static_cast<int>(i)).m_SynapticDelay, 1e-12));
        }
        for (unsigned i = 0; i < g.NumNeurons(); ++i) {
            CHECK(Near(file_loaded.GetNeuronByIndex(static_cast<int>(i)).m_SpikeThreshold, g.GetNeuronByIndex(static_cast<int>(i)).m_SpikeThreshold, 1e-12));
        }
        std::error_code ec;
        std::filesystem::remove(tmp, ec);
    }

    // Validate rejects structural violations with a message.
    {
        Genome g = MakeSeed(2, 1);
        std::string error;
        CHECK(g.Validate(&error));
        CHECK(error.empty());
        // Duplicate neuron IDs fail validation.
        Genome bad = g;
        bad.m_NeuronGenes.push_back(bad.m_NeuronGenes.front());
        CHECK(!bad.Validate(&error));
        CHECK(!error.empty());
    }

    // MateWithMode: all crossover modes produce valid babies.
    {
        Parameters p = DefaultParams();
        RNG rng;
        rng.Seed(21);
        InnovationDatabase innovs;
        Genome mom = MakeSeed(3, 1);
        Genome dad = MakeSeed(3, 1);
        innovs.Init(mom);
        for (int i = 0; i < 10; ++i) {
            (void)mom.Mutate_AddNeuron(innovs, p, rng);
            (void)dad.Mutate_AddLink(innovs, p, rng);
        }
        mom.SetFitness(5.0);
        dad.SetFitness(4.0);
        const CrossoverMode modes[] = {MULTIPOINT, AVERAGE, SINGLE_POINT, BLEND, SIMULATED_BINARY};
        for (CrossoverMode mode : modes) {
            Genome baby = mom.MateWithMode(dad, mode, false, rng, p);
            CHECK(baby.NumInputs() == mom.NumInputs());
            CHECK(baby.NumOutputs() == mom.NumOutputs());
            CHECK(baby.Validate());
        }
        // Mismatched I/O counts throw.
        Genome other = MakeSeed(4, 1);
        bool threw = false;
        try {
            (void)mom.MateWithMode(other, MULTIPOINT, false, rng, p);
        } catch (const std::invalid_argument &) {
            threw = true;
        }
        CHECK(threw);
    }

    // Spiking mutators stay in range and report honestly.
    {
        Parameters p = DefaultParams();
        p.ConfigureSpiking(false);
        RNG rng;
        rng.Seed(31);
        Genome g = MakeSeed(2, 1);
        for (unsigned i = 0; i < g.NumNeurons(); ++i) {
            if (g.m_NeuronGenes[i].Type() == OUTPUT) g.m_NeuronGenes[i].m_ActFunction = SPIKING_LIF;
        }
        g.Randomize_SpikingParameters(p, rng);
        (void)g.Mutate_NeuronSpikingParameters(p, rng);
        (void)g.Mutate_LinkSpikingParameters(p, rng);
        CHECK(g.Validate());
        for (unsigned i = 0; i < g.NumLinks(); ++i) {
            const LinkGene &l = g.GetLinkByIndex(static_cast<int>(i));
            CHECK(l.m_SynapticDelay >= p.MinSynapticDelay && l.m_SynapticDelay <= p.MaxSynapticDelay);
        }
    }

    // HyperNEAT and ES-HyperNEAT build phenotypes from substrates.
    {
        Parameters p = DefaultParams();
        // CPPN needs 2*2+1 inputs (2D coords x2 + bias) and 2 outputs.
        GenomeInitStruct cppn_init;
        cppn_init.NumInputs = 5;
        cppn_init.NumOutputs = 2;
        cppn_init.SeedType = PERCEPTRON;
        Genome cppn(p, cppn_init);
        CHECK(cppn.NumInputs() == 5 && cppn.NumOutputs() == 2);

        std::vector<std::vector<Real>> inputs{{0.0, 0.0}, {1.0, 0.0}};
        std::vector<std::vector<Real>> hidden;
        std::vector<std::vector<Real>> outputs{{0.5, 1.0}};
        Substrate subst(inputs, hidden, outputs);
        NeuralNetwork net;
        cppn.BuildHyperNEATPhenotype(net, subst);
        CHECK(net.NumInputs() == 2 && net.NumOutputs() == 1);

        // Empty substrate is rejected.
        Substrate empty;
        bool threw = false;
        try {
            cppn.BuildHyperNEATPhenotype(net, empty);
        } catch (const std::invalid_argument &) {
            threw = true;
        }
        CHECK(threw);

        // ES-HyperNEAT runs on the same substrate and validates its inputs.
        NeuralNetwork es_net;
        cppn.BuildESHyperNEATPhenotype(es_net, subst, p);
        CHECK(es_net.NumInputs() == 2 && es_net.NumOutputs() == 1);
        threw = false;
        try {
            cppn.BuildESHyperNEATPhenotype(es_net, empty, p);
        } catch (const std::invalid_argument &) {
            threw = true;
        }
        CHECK(threw);
    }

    // Parametric mutators report whether any value actually changed.
    {
        Parameters p = DefaultParams();
        RNG rng;
        rng.Seed(41);
        Genome g = MakeSeed(2, 1);
        // Default A/B/time-constant/bias powers and ranges are degenerate, so nothing can change.
        CHECK(!g.Mutate_NeuronActivations_A(p, rng));
        CHECK(!g.Mutate_NeuronActivations_B(p, rng));
        CHECK(!g.Mutate_NeuronTimeConstants(p, rng));
        CHECK(!g.Mutate_NeuronBiases(p, rng));

        p.ActivationAMutationMaxPower = 1.0;
        p.ActivationBMutationMaxPower = 1.0;
        p.TimeConstantMutationMaxPower = 1.0;
        p.BiasMutationMaxPower = 1.0;
        p.MinActivationA = 0.0;
        p.MaxActivationA = 10.0;
        p.MinActivationB = -5.0;
        p.MaxActivationB = 5.0;
        p.MinNeuronTimeConstant = 0.0;
        p.MaxNeuronTimeConstant = 2.0;
        p.MinNeuronBias = -2.0;
        p.MaxNeuronBias = 2.0;
        CHECK(g.Mutate_NeuronActivations_A(p, rng));
        CHECK(g.Mutate_NeuronActivations_B(p, rng));
        CHECK(g.Mutate_NeuronTimeConstants(p, rng));
        CHECK(g.Mutate_NeuronBiases(p, rng));
        CHECK(g.Validate());
    }

    // IsIdenticalTo is sensitive to the MCP inhibitory veto flag.
    {
        Genome g = MakeSeed(2, 1);
        Genome h = g;
        CHECK(g.IsIdenticalTo(h));
        for (auto &n : h.m_NeuronGenes) {
            if (n.Type() == OUTPUT) n.m_MCPInhibitoryVeto = !n.m_MCPInhibitoryVeto;
        }
        CHECK(!g.IsIdenticalTo(h));
    }

    // LinkGene ==/!= form a consistent pair over topology and weights.
    {
        LinkGene a(1, 2, 7, 0.5, false);
        LinkGene b(1, 2, 9, 0.5, false);
        CHECK(a == b);     // historical innovation IDs are ignored...
        CHECK(!(a != b));  // ...and != agrees with ==.
        LinkGene c(1, 2, 7, 0.6, false);
        CHECK(!(a == c));
        CHECK(a != c);
    }

    // Seed genomes initialize traits on inputs and bias as well.
    {
        Parameters p = DefaultParams();
        TraitParameters tp;
        tp.type = "float";
        tp.m_MutationProb = 1.0;
        tp.m_ImportanceCoeff = 1.0;
        FloatTraitParameters d;
        d.min = 0.0;
        d.max = 1.0;
        d.mut_power = 0.5;
        d.mut_replace_prob = 0.5;
        tp.m_Details = d;
        p.NeuronTraits["seed_trait"] = tp;
        GenomeInitStruct init;
        init.NumInputs = 3;
        init.NumOutputs = 1;
        init.SeedType = PERCEPTRON;
        Genome g(p, init);
        CHECK(!g.m_NeuronGenes.empty());
        for (const auto &n : g.m_NeuronGenes) {
            CHECK(n.m_Traits.count("seed_trait") == 1);
        }
    }

    // Plain HyperNEAT finalizes spatial connections (lengths, delays, pruning).
    {
        Parameters p = DefaultParams();
        GenomeInitStruct cppn_init;
        cppn_init.NumInputs = 5;
        cppn_init.NumOutputs = 1;
        cppn_init.SeedType = PERCEPTRON;
        Genome cppn(p, cppn_init);

        std::vector<std::vector<Real>> inputs{{0.0, 0.0}};
        std::vector<std::vector<Real>> hidden;
        std::vector<std::vector<Real>> outputs{{0.5, 1.0}};
        Substrate subst(inputs, hidden, outputs);
        subst.m_query_weights_only = true;
        // The coordinate-list ctor disables direct input->output links; opt back in.
        subst.m_allow_input_output_links = true;

        NeuralNetwork plain;
        cppn.BuildHyperNEATPhenotype(plain, subst);
        CHECK(plain.m_connections.size() == 1);
        if (plain.m_connections.size() == 1) {
            CHECK(plain.m_connections[0].m_length > 0.0);
        }

        // A zero max length prunes every non-degenerate axon.
        Substrate pruned = subst;
        pruned.m_max_connection_length = 0.0;
        NeuralNetwork pruned_net;
        cppn.BuildHyperNEATPhenotype(pruned_net, pruned);
        CHECK(pruned_net.m_connections.empty());

        // Spatial delays mirror axon length over the conduction velocity.
        Substrate delayed = subst;
        delayed.m_use_spatial_distance_for_delays = true;
        delayed.m_conduction_velocity = 1.0;
        NeuralNetwork delayed_net;
        cppn.BuildHyperNEATPhenotype(delayed_net, delayed);
        CHECK(delayed_net.m_connections.size() == 1);
        if (delayed_net.m_connections.size() == 1) {
            CHECK(delayed_net.m_connections[0].m_synaptic_delay > 0.0);
        }
    }

    // Layered seeds stack fully connected hidden layers.
    {
        Parameters p = DefaultParams();
        GenomeInitStruct init;
        init.NumInputs = 3;
        init.NumOutputs = 2;
        init.SeedType = LAYERED;
        init.NumHidden = 2;
        init.NumLayers = 2;
        Genome g(p, init);
        CHECK(g.NumNeurons() == 9);  // 3 in/bias + 2 out + 2x2 hidden
        CHECK(g.NumLinks() == 14);   // 3*2 + 2*2 + 2*2
        CHECK(g.Validate());
        NeuralNetwork net;
        g.BuildPhenotype(net);
        CHECK(net.m_connections.size() == g.NumLinks());
    }

    // Link enable bit: toggle, add-neuron disablement, phenotype masking.
    {
        Parameters p = DefaultParams();
        RNG rng;
        rng.Seed(51);
        InnovationDatabase innovs;

        // Empty genomes have nothing to toggle.
        Genome empty;
        CHECK(!empty.Mutate_ToggleEnable(rng));

        Genome g = MakeSeed(2, 1);
        innovs.Init(g);
        CHECK(g.NumLinks() == 2);
        for (unsigned i = 0; i < g.NumLinks(); ++i) CHECK(g.GetLinkByIndex(static_cast<int>(i)).IsEnabled());

        // Toggling flips exactly one link's bit.
        CHECK(g.Mutate_ToggleEnable(rng));
        unsigned disabled = 0;
        for (unsigned i = 0; i < g.NumLinks(); ++i) {
            if (!g.GetLinkByIndex(static_cast<int>(i)).IsEnabled()) ++disabled;
        }
        CHECK(disabled == 1);

        // Add-neuron keeps (disables) the split link instead of deleting it.
        Genome h = MakeSeed(2, 1);
        CHECK(h.Mutate_AddNeuron(innovs, p, rng));
        CHECK(h.NumLinks() == 4);  // 2 kept (one disabled) + 2 new
        unsigned h_disabled = 0;
        for (unsigned i = 0; i < h.NumLinks(); ++i) {
            if (!h.GetLinkByIndex(static_cast<int>(i)).IsEnabled()) ++h_disabled;
        }
        CHECK(h_disabled == 1);
        CHECK(h.Validate());

        // The phenotype expresses only enabled links.
        NeuralNetwork net;
        h.BuildPhenotype(net);
        CHECK(net.m_connections.size() == 3);

        // Phenotypic changes map back by endpoints, skipping disabled links.
        net.m_connections[0].m_weight = 42.0;
        h.DerivePhenotypicChanges(net);
        unsigned at_42 = 0;
        for (unsigned i = 0; i < h.NumLinks(); ++i) {
            if (Near(h.GetLinkByIndex(static_cast<int>(i)).GetWeight(), 42.0)) ++at_42;
        }
        CHECK(at_42 == 1);

        // A hidden neuron kept alive only by disabled links is a dead end.
        Genome d = MakeSeed(2, 1);
        CHECK(d.Mutate_AddNeuron(innovs, p, rng));
        CHECK(!d.HasDeadEnds());
        int hidden_id = -1;
        for (unsigned i = 0; i < d.NumNeurons(); ++i) {
            if (d.GetNeuronByIndex(static_cast<int>(i)).Type() == HIDDEN) hidden_id = d.GetNeuronByIndex(static_cast<int>(i)).ID();
        }
        CHECK(hidden_id > 0);
        for (unsigned i = 0; i < d.NumLinks(); ++i) {
            LinkGene &l = d.m_LinkGenes[i];
            if (l.FromNeuronID() == hidden_id || l.ToNeuronID() == hidden_id) l.SetEnabled(false);
        }
        CHECK(d.HasDeadEnds());
        for (unsigned i = 0; i < d.NumLinks(); ++i) d.m_LinkGenes[i].SetEnabled(true);
        CHECK(!d.HasDeadEnds());

        // A cycle made only of disabled links is not a loop.
        Genome c = MakeSeed(2, 1);
        const int output_id = c.NumInputs() + 1;
        c.m_LinkGenes.emplace_back(output_id, 1, c.GetLastInnovationID() + 1, 0.5, false);
        CHECK(c.HasLoops());
        c.m_LinkGenes.back().SetEnabled(false);
        CHECK(!c.HasLoops());
    }

    // Mate propagates the enable bit (paper Section 4 rule).
    {
        Parameters p = DefaultParams();
        p.PreferFitterParentRate = 1.0;
        RNG rng;
        rng.Seed(52);
        Genome mom = MakeSeed(3, 1);
        Genome dad = MakeSeed(3, 1);
        mom.SetID(1);
        dad.SetID(2);
        mom.SetFitness(2.0);
        dad.SetFitness(1.0);
        mom.m_LinkGenes[0].SetEnabled(false);

        p.DisabledGeneInheritRate = 1.0;
        Genome baby = mom.MateWithMode(dad, MULTIPOINT, false, rng, p);
        CHECK(baby.NumLinks() == 3);
        CHECK(!baby.GetLinkByInnovID(1).IsEnabled());

        p.DisabledGeneInheritRate = 0.0;
        Genome baby2 = mom.MateWithMode(dad, MULTIPOINT, false, rng, p);
        CHECK(baby2.NumLinks() == 3);
        CHECK(baby2.GetLinkByInnovID(1).IsEnabled());

        // Both parents enabled: the child is always enabled.
        mom.m_LinkGenes[0].SetEnabled(true);
        p.DisabledGeneInheritRate = 1.0;
        Genome baby3 = mom.MateWithMode(dad, MULTIPOINT, false, rng, p);
        CHECK(baby3.GetLinkByInnovID(1).IsEnabled());
    }

    // CompatibilityDistance never normalizes small genomes (paper Eq. 1).
    {
        auto coeffs = [](Parameters &p) {
            p.NormalizeGenomeSize = true;
            p.ExcessCoeff = 1.0;
            p.DisjointCoeff = 1.0;
            p.WeightDiffCoeff = 0.0;
            p.ActivationADiffCoeff = 0.0;
            p.ActivationBDiffCoeff = 0.0;
            p.TimeConstantDiffCoeff = 0.0;
            p.BiasDiffCoeff = 0.0;
            p.ActivationFunctionDiffCoeff = 0.0;
            p.SpikingNeuronDiffCoeff = 0.0;
            p.SpikingLinkDiffCoeff = 0.0;
        };
        // Small genomes: one excess gene counts whole (N = 1).
        Parameters p = DefaultParams();
        coeffs(p);
        GenomeInitStruct fs;
        fs.NumInputs = 3;
        fs.NumOutputs = 2;
        fs.SeedType = PERCEPTRON;
        fs.FS_NEAT = true;
        fs.FS_NEAT_links = 2;
        Genome small_a(DefaultParams(), fs);
        Genome small_b = small_a;
        // Add a link on a free endpoint pair with a fresh innovation number.
        bool added = false;
        for (int from = 1; from <= 3 && !added; ++from) {
            for (int to = 4; to <= 5 && !added; ++to) {
                bool used = false;
                for (unsigned i = 0; i < small_a.NumLinks(); ++i) {
                    const LinkGene &l = small_a.GetLinkByIndex(static_cast<int>(i));
                    if (l.FromNeuronID() == from && l.ToNeuronID() == to) used = true;
                }
                if (!used) {
                    small_b.m_LinkGenes.emplace_back(from, to, 1000000, 0.25, false);
                    added = true;
                }
            }
        }
        CHECK(added);
        CHECK(small_a.NumLinks() < 20 && small_b.NumLinks() < 20);
        CHECK(Near(small_a.CompatibilityDistance(small_b, p), 1.0));

        // Large genomes: the same single excess gene is divided by N >= 20.
        GenomeInitStruct big;
        big.NumInputs = 6;
        big.NumOutputs = 5;
        big.SeedType = PERCEPTRON;
        big.FS_NEAT = true;
        big.FS_NEAT_links = 20;
        Genome big_a(DefaultParams(), big);
        CHECK(big_a.NumLinks() >= 20);
        Genome big_b = big_a;
        added = false;
        for (int from = 1; from <= 6 && !added; ++from) {
            for (int to = 7; to <= 11 && !added; ++to) {
                bool used = false;
                for (unsigned i = 0; i < big_a.NumLinks(); ++i) {
                    const LinkGene &l = big_a.GetLinkByIndex(static_cast<int>(i));
                    if (l.FromNeuronID() == from && l.ToNeuronID() == to) used = true;
                }
                if (!used) {
                    big_b.m_LinkGenes.emplace_back(from, to, 1000000, 0.25, false);
                    added = true;
                }
            }
        }
        CHECK(added);
        const Real big_dist = big_a.CompatibilityDistance(big_b, p);
        CHECK(big_dist > 0.0 && big_dist < 0.1);
    }

    // Format 5 persists the enable bit; format 4 still loads (bit defaults on).
    {
        Genome g = MakeSeed(2, 1);
        g.SetID(4242);
        g.m_LinkGenes[0].SetEnabled(false);
        const std::string data = g.Serialize();
        CHECK(data.find("GenomeFormat 5") != std::string::npos);
        Genome h = Genome::Deserialize(data);
        CHECK(g.IsIdenticalTo(h));
        CHECK(!h.GetLinkByIndex(0).IsEnabled());
        CHECK(h.GetLinkByIndex(1).IsEnabled());

        // Legacy file Save/Load preserves the bit.
        const auto tmp = std::filesystem::temp_directory_path() / "multineat_test_genome_enabled.txt";
        g.Save(tmp.string().c_str());
        Genome file_loaded(tmp.string().c_str());
        CHECK(!file_loaded.GetLinkByIndex(0).IsEnabled());
        CHECK(file_loaded.GetLinkByIndex(1).IsEnabled());
        std::error_code ec;
        std::filesystem::remove(tmp, ec);

        // A format-4 payload (no trailing bit) loads with every link enabled.
        std::ostringstream v4;
        std::istringstream lines(data);
        std::string line;
        while (std::getline(lines, line)) {
            if (line.rfind("GenomeFormat 5", 0) == 0) {
                v4 << "GenomeFormat 4\n";
            } else if (line.rfind("Link ", 0) == 0) {
                v4 << line.substr(0, line.find_last_of(' ')) << "\n";
            } else {
                v4 << line << "\n";
            }
        }
        Genome legacy = Genome::Deserialize(v4.str());
        CHECK(legacy.NumLinks() == g.NumLinks());
        for (unsigned i = 0; i < legacy.NumLinks(); ++i) CHECK(legacy.GetLinkByIndex(static_cast<int>(i)).IsEnabled());
    }

    // LEO seeding and LEO-gated HyperNEAT phenotypes.
    {
        // LeoSeed reserves output index 1 as an UNSIGNED_STEP LEO neuron.
        Parameters p = DefaultParams();
        p.Leo = true;
        p.LeoSeed = true;
        GenomeInitStruct cppn_init;
        cppn_init.NumInputs = 5;
        cppn_init.NumOutputs = 2;
        cppn_init.SeedType = PERCEPTRON;
        Genome cppn(p, cppn_init);
        CHECK(cppn.NumOutputs() == 2);
        CHECK(cppn.m_NeuronGenes[5].m_ActFunction == UNSIGNED_SIGMOID);
        CHECK(cppn.m_NeuronGenes[6].m_ActFunction == UNSIGNED_STEP);

        // LeoSeed without Leo, or with fewer than two outputs, is rejected.
        Parameters bad = DefaultParams();
        bad.LeoSeed = true;
        bool threw = false;
        try {
            Genome rejected(bad, cppn_init);
        } catch (const std::invalid_argument &) {
            threw = true;
        }
        CHECK(threw);
        Parameters few = DefaultParams();
        few.Leo = true;
        few.LeoSeed = true;
        GenomeInitStruct one_out;
        one_out.NumInputs = 5;
        one_out.NumOutputs = 1;
        one_out.SeedType = PERCEPTRON;
        threw = false;
        try {
            Genome rejected(few, one_out);
        } catch (const std::invalid_argument &) {
            threw = true;
        }
        CHECK(threw);

        // GeometrySeed biases the source-x input onto the weight output.
        Parameters geo = DefaultParams();
        geo.GeometrySeed = true;
        Genome gseed(geo, cppn_init);
        bool found_x_link = false;
        for (unsigned i = 0; i < gseed.NumLinks(); ++i) {
            const LinkGene &l = gseed.GetLinkByIndex(static_cast<int>(i));
            if (l.FromNeuronID() == 1 && l.ToNeuronID() == 6) {
                CHECK(Near(l.GetWeight(), 1.0));
                found_x_link = true;
            } else {
                CHECK(Near(l.GetWeight(), 0.0));
            }
        }
        CHECK(found_x_link);

        // LEO-gated build: output 0 is the weight, output 1 the LEO signal.
        Parameters leo = DefaultParams();
        leo.Leo = true;
        leo.LeoThreshold = 0.1;
        Genome gated(DefaultParams(), cppn_init);
        for (auto &l : gated.m_LinkGenes) l.SetWeight(0.0);
        // Bias (input 5) drives the weight output high and LEO low.
        for (auto &l : gated.m_LinkGenes) {
            if (l.FromNeuronID() == 5 && l.ToNeuronID() == 6) l.SetWeight(8.0);
            if (l.FromNeuronID() == 5 && l.ToNeuronID() == 7) l.SetWeight(-8.0);
        }
        std::vector<std::vector<Real>> inputs{{0.0, 0.0}};
        std::vector<std::vector<Real>> hidden;
        std::vector<std::vector<Real>> outputs{{0.5, 1.0}};
        Substrate subst(inputs, hidden, outputs);
        subst.m_query_weights_only = true;
        subst.m_allow_input_output_links = true;
        NeuralNetwork suppressed;
        gated.BuildHyperNEATPhenotype(suppressed, subst, leo);
        CHECK(suppressed.m_connections.empty());
        // Raising LEO above the threshold expresses the connection.
        for (auto &l : gated.m_LinkGenes) {
            if (l.FromNeuronID() == 5 && l.ToNeuronID() == 7) l.SetWeight(8.0);
        }
        NeuralNetwork expressed;
        gated.BuildHyperNEATPhenotype(expressed, subst, leo);
        CHECK(expressed.m_connections.size() == 1);
        if (expressed.m_connections.size() == 1) CHECK(expressed.m_connections[0].m_weight > 0.0);
        // The legacy two-argument build keeps the historical gate-on-positive.
        NeuralNetwork legacy_build;
        gated.BuildHyperNEATPhenotype(legacy_build, subst);
        CHECK(legacy_build.m_connections.size() == 1);

        // LEO needs room for the signal: one output is not enough.
        Genome narrow(DefaultParams(), one_out);
        threw = false;
        try {
            narrow.BuildHyperNEATPhenotype(expressed, subst, leo);
        } catch (const std::invalid_argument &) {
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
