// Tests for traits/genes (src/Traits.h, src/Genes.h).
#include <cmath>
#include <iostream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>

#include "Genes.h"
#include "Random.h"
#include "Serialization.h"
#include "Traits.h"

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

    NEAT::TraitParameters MakeIntTrait(int mn, int mx, double mut_prob = 1.0) {
        NEAT::TraitParameters tp;
        tp.type = "int";
        tp.m_MutationProb = mut_prob;
        tp.m_ImportanceCoeff = 1.0;
        NEAT::IntTraitParameters d;
        d.min = mn;
        d.max = mx;
        d.mut_power = 2;
        d.mut_replace_prob = 0.5;
        tp.m_Details = d;
        return tp;
    }

    NEAT::TraitParameters MakeFloatTrait(double mn, double mx, double mut_prob = 1.0) {
        NEAT::TraitParameters tp;
        tp.type = "float";
        tp.m_MutationProb = mut_prob;
        tp.m_ImportanceCoeff = 1.0;
        NEAT::FloatTraitParameters d;
        d.min = mn;
        d.max = mx;
        d.mut_power = 0.5;
        d.mut_replace_prob = 0.5;
        tp.m_Details = d;
        return tp;
    }

}  // namespace

int TestTraitsGenes(int argc, char *argv[]) {
    (void)argc;
    (void)argv;
    using namespace NEAT;

    // --- Gene::InitTraits for each supported type -------------------------
    {
        RNG rng;
        rng.Seed(42);
        std::map<std::string, TraitParameters> tp;
        tp["speed"] = MakeIntTrait(0, 10);
        tp["rate"] = MakeFloatTrait(-1.0, 1.0);

        TraitParameters str_tp;
        str_tp.type = "str";
        str_tp.m_MutationProb = 1.0;
        StringTraitParameters sdet;
        sdet.set = {"red", "green", "blue"};
        sdet.probs = {0.2, 0.5, 0.3};
        str_tp.m_Details = sdet;
        tp["color"] = str_tp;

        TraitParameters iset_tp;
        iset_tp.type = "intset";
        iset_tp.m_MutationProb = 1.0;
        IntSetTraitParameters idet;
        idet.set = {intsetelement{1}, intsetelement{2}, intsetelement{3}};
        idet.probs = {0.3, 0.3, 0.4};
        iset_tp.m_Details = idet;
        tp["mode"] = iset_tp;

        TraitParameters fset_tp;
        fset_tp.type = "floatset";
        fset_tp.m_MutationProb = 1.0;
        FloatSetTraitParameters fdet;
        fdet.set = {floatsetelement{0.5}, floatsetelement{1.5}};
        fdet.probs = {0.5, 0.5};
        fset_tp.m_Details = fdet;
        tp["gain"] = fset_tp;

        Gene g;
        g.InitTraits(tp, rng);
        CHECK(g.m_Traits.count("speed") == 1);
        CHECK(g.m_Traits.count("rate") == 1);
        CHECK(g.m_Traits.count("color") == 1);
        CHECK(g.m_Traits.count("mode") == 1);
        CHECK(g.m_Traits.count("gain") == 1);

        const int speed = std::get<int>(g.m_Traits["speed"].value);
        CHECK(speed >= 0 && speed <= 10);
        const double rate = std::get<double>(g.m_Traits["rate"].value);
        CHECK(rate >= -1.0 && rate <= 1.0);
        const std::string color = std::get<std::string>(g.m_Traits["color"].value);
        CHECK(color == "red" || color == "green" || color == "blue");
    }

    // --- Empty sets throw ---------------------------------------------------
    {
        RNG rng;
        rng.Seed(1);
        for (const char *kind : {"str", "intset", "floatset"}) {
            std::map<std::string, TraitParameters> tp;
            TraitParameters p;
            p.type = kind;
            if (std::string(kind) == "str") {
                p.m_Details = StringTraitParameters();
            } else if (std::string(kind) == "intset") {
                p.m_Details = IntSetTraitParameters();
            } else {
                p.m_Details = FloatSetTraitParameters();
            }
            tp["x"] = p;
            Gene g;
            bool threw = false;
            try {
                g.InitTraits(tp, rng);
            } catch (const std::runtime_error &) {
                threw = true;
            }
            CHECK(threw);
        }
    }

    // --- MateTraits: happy path + type mismatch ------------------------------
    {
        RNG rng;
        rng.Seed(5);
        Gene a, b;
        Trait ta, tb;
        ta.value = 4;
        tb.value = 8;
        a.m_Traits["k"] = ta;
        b.m_Traits["k"] = tb;
        a.MateTraits(b.m_Traits, rng);
        const int v = std::get<int>(a.m_Traits["k"].value);
        // Either parent (4/8) or the average (6).
        CHECK(v == 4 || v == 6 || v == 8);
    }
    {
        // Keys missing on one side are skipped (no map insertion / no throw).
        RNG rng;
        rng.Seed(5);
        Gene a, b;
        Trait ta, tb;
        ta.value = 4;
        tb.value = 8;
        a.m_Traits["k"] = ta;
        b.m_Traits["other"] = tb;
        a.MateTraits(b.m_Traits, rng);
        CHECK(a.m_Traits.count("other") == 0);
        CHECK(std::get<int>(a.m_Traits["k"].value) == 4);
    }
    {
        RNG rng;
        rng.Seed(5);
        Gene a, b;
        Trait ta, tb;
        ta.value = 4;
        tb.value = std::string("oops");
        a.m_Traits["k"] = ta;
        b.m_Traits["k"] = tb;
        bool threw = false;
        try {
            a.MateTraits(b.m_Traits, rng);
        } catch (const std::runtime_error &) {
            threw = true;
        }
        CHECK(threw);
    }

    // --- GetTraitDistances ----------------------------------------------------
    {
        Gene a, b;
        Trait t1, t2, s1, s2;
        t1.value = 3;
        t2.value = 10;
        s1.value = std::string("x");
        s2.value = std::string("y");
        a.m_Traits["i"] = t1;
        b.m_Traits["i"] = t2;
        a.m_Traits["s"] = s1;
        b.m_Traits["s"] = s2;
        const auto dist = a.GetTraitDistances(b.m_Traits);
        CHECK(dist.count("i") == 1 && Near(dist.at("i"), 7.0));
        CHECK(dist.count("s") == 1 && Near(dist.at("s"), 1.0));

        // Identical strings => distance 0.
        b.m_Traits["s"] = s1;
        const auto dist2 = a.GetTraitDistances(b.m_Traits);
        CHECK(Near(dist2.at("s"), 0.0));

        // Mismatched variant types throw.
        b.m_Traits["i"].value = std::string("nope");
        bool threw = false;
        try {
            (void)a.GetTraitDistances(b.m_Traits);
        } catch (const std::runtime_error &) {
            threw = true;
        }
        CHECK(threw);
    }
    {
        // Dependency gating: distance is skipped unless both sides have the gate trait set to one of dep_values.
        Gene a, b;
        Trait gate_a, gate_b, v_a, v_b;
        gate_a.value = std::string("on");
        gate_b.value = std::string("off");
        v_a.value = 1;
        v_b.value = 9;
        v_a.dep_key = "gate";
        v_b.dep_key = "gate";
        v_a.dep_values.emplace_back(std::string("on"));
        v_b.dep_values.emplace_back(std::string("on"));
        a.m_Traits["gate"] = gate_a;
        b.m_Traits["gate"] = gate_b;
        a.m_Traits["v"] = v_a;
        b.m_Traits["v"] = v_b;
        const auto dist = a.GetTraitDistances(b.m_Traits);
        CHECK(dist.count("v") == 0);

        b.m_Traits["gate"] = gate_a;  // both "on" now
        const auto dist2 = a.GetTraitDistances(b.m_Traits);
        CHECK(dist2.count("v") == 1 && Near(dist2.at("v"), 8.0));
    }
    {
        // Asymmetric maps: missing keys are skipped, and const genes work.
        Gene a, b;
        Trait t1, t2;
        t1.value = 3;
        t2.value = 10;
        a.m_Traits["i"] = t1;
        b.m_Traits["i"] = t2;
        b.m_Traits["ghost"] = t2;
        const Gene &ca = a;
        const auto dist = ca.GetTraitDistances(b.m_Traits);
        CHECK(dist.count("i") == 1 && Near(dist.at("i"), 7.0));
        CHECK(dist.count("ghost") == 0);
    }

    // --- MutateTraits stays in range ------------------------------------------
    {
        RNG rng;
        rng.Seed(9);
        std::map<std::string, TraitParameters> tp;
        tp["speed"] = MakeIntTrait(0, 10);
        Gene g;
        g.InitTraits(tp, rng);
        for (int i = 0; i < 50; ++i) {
            (void)g.MutateTraits(tp, rng);
            const int v = std::get<int>(g.m_Traits["speed"].value);
            CHECK(v >= 0 && v <= 10);
        }
    }

    // --- Trait validation rejects bad schemas -----------------------------------
    {
        RNG rng;
        rng.Seed(4);
        Gene g;
        // Unknown type throws instead of silently creating an empty trait.
        std::map<std::string, TraitParameters> bad;
        TraitParameters p;
        p.type = "bogus";
        bad["x"] = p;
        bool threw = false;
        try {
            g.InitTraits(bad, rng);
        } catch (const std::invalid_argument &) {
            threw = true;
        }
        CHECK(threw);
        // min > max throws.
        std::map<std::string, TraitParameters> inverted;
        inverted["speed"] = MakeIntTrait(10, 0);
        threw = false;
        try {
            g.InitTraits(inverted, rng);
        } catch (const std::invalid_argument &) {
            threw = true;
        }
        CHECK(threw);
    }

    // --- Serialization round-trips traits and schemas ---------------------------
    // (mirrors Genome's usage: marker token consumed first, then the reader
    // takes the count and entries).
    {
        std::map<std::string, TraitParameters> schemas;
        schemas["speed"] = MakeIntTrait(0, 10);
        schemas["rate"] = MakeFloatTrait(-1.0, 1.0);
        std::ostringstream schema_out;
        NEAT::Serialization::WriteTraitParameters(schema_out, "Schemas", schemas);
        std::istringstream schema_in(schema_out.str());
        std::string schema_marker;
        schema_in >> schema_marker;
        CHECK(schema_marker == "Schemas");
        const auto read_back = NEAT::Serialization::ReadTraitParameters(schema_in);
        CHECK(read_back.size() == 2);
        CHECK(read_back.at("speed").type == "int");
        CHECK(read_back.at("rate").type == "float");

        Gene g;
        RNG rng;
        rng.Seed(4);
        g.InitTraits(schemas, rng);
        std::ostringstream trait_out;
        NEAT::Serialization::WriteTraits(trait_out, "Traits", g.m_Traits);
        std::istringstream trait_in(trait_out.str());
        std::string trait_marker;
        trait_in >> trait_marker;
        CHECK(trait_marker == "Traits");
        const auto traits_back = NEAT::Serialization::ReadTraits(trait_in);
        CHECK(traits_back.size() == g.m_Traits.size());
        CHECK(traits_back.at("speed") == g.m_Traits.at("speed"));
    }

    // --- LinkGene / NeuronGene basics ------------------------------------------
    {
        LinkGene l(1, 2, 7, 0.5, false);
        CHECK(l.FromNeuronID() == 1);
        CHECK(l.ToNeuronID() == 2);
        CHECK(l.InnovationID() == 7);
        CHECK(Near(l.GetWeight(), 0.5));
        CHECK(!l.IsRecurrent());
        CHECK(!l.IsLoopedRecurrent());
        // Spiking defaults are sane and inert for rate networks.
        CHECK(!l.m_STDPEnabled);
        CHECK(Near(l.m_SynapticDelay, 0.0));
        LinkGene loop(4, 4, 8, 1.0, true);
        CHECK(loop.IsLoopedRecurrent());
        // operator== compares the full field set (v2 semantics): identical
        // genes compare equal, genes differing in endpoints/weight do not,
        // while ordering (<) still uses the innovation ID.
        CHECK((LinkGene(1, 2, 7, 0.5) == LinkGene(1, 2, 7, 0.5)));
        CHECK(!((LinkGene(1, 2, 7, 0.0) == LinkGene(9, 9, 7, 5.0))));
        CHECK((LinkGene(1, 2, 7, 0.0) < LinkGene(1, 2, 8, 0.0)));

        NeuronGene n(HIDDEN, 42, 0.5);
        n.Init(1.0, 0.0, 1.0, 0.1, TANH);
        CHECK(n.ID() == 42);
        CHECK(n.Type() == HIDDEN);
        CHECK(n.m_ActFunction == TANH);
        // operator== compares the full field set (v2 semantics).
        CHECK((NeuronGene(HIDDEN, 42, 0.5) == NeuronGene(HIDDEN, 42, 0.5)));
        CHECK(!((NeuronGene(HIDDEN, 42, 0.0) == NeuronGene(HIDDEN, 42, 1.0))));
        CHECK(!((NeuronGene(HIDDEN, 42, 0.0) == NeuronGene(OUTPUT, 42, 0.0))));
        // Spiking defaults + default construction is zero-initialized.
        NeuronGene d;
        CHECK(d.m_ID == 0 && d.m_Type == NONE);
        CHECK(Near(d.m_SpikeThreshold, 1.0));
        CHECK(Near(d.m_IzhikevichC, -65.0));
        CHECK(d.m_MCPInhibitoryVeto);
        CHECK(IsSpikingActivation(SPIKING_LIF));
        CHECK(IsSpikingActivation(MCCULLOCH_PITTS));
        CHECK(!IsSpikingActivation(TANH));
    }

    // --- MutateTraits terminates on degenerate schemas -------------------------
    {
        RNG rng;
        rng.Seed(3);
        // Singleton set: nothing to change to, must return quickly, not hang.
        std::map<std::string, TraitParameters> tp;
        TraitParameters p;
        p.type = "str";
        p.m_MutationProb = 1.0;
        StringTraitParameters sdet;
        sdet.set = {"only"};
        sdet.probs = {1.0};
        p.m_Details = sdet;
        tp["s"] = p;
        Gene g;
        g.InitTraits(tp, rng);
        (void)g.MutateTraits(tp, rng);
        CHECK(std::get<std::string>(g.m_Traits["s"].value) == "only");
    }

    if (g_failures != 0) {
        std::cerr << "Test failed: TestTraitsGenes with " << g_failures << " failure(s)\n";
        return 1;
    }
    std::cout << "Test passed: TestTraitsGenes\n";
    return 0;
}
