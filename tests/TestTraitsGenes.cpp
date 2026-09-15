// Tests for traits/genes (src/Traits.h, src/Genes.h).
#include <cmath>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>

#include "Genes.h"
#include "Random.h"
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

    bool near(double a, double b, double eps = 1e-9) { return std::fabs(a - b) <= eps; }

    NEAT::TraitParameters makeIntTrait(int mn, int mx, double mutProb = 1.0) {
        NEAT::TraitParameters tp;
        tp.type = "int";
        tp.mutationProb_ = mutProb;
        tp.importanceCoeff_ = 1.0;
        NEAT::IntTraitParameters d;
        d.min = mn;
        d.max = mx;
        d.mutPower = 2;
        d.mutReplaceProb = 0.5;
        tp.details_ = d;
        return tp;
    }

    NEAT::TraitParameters makeFloatTrait(double mn, double mx, double mutProb = 1.0) {
        NEAT::TraitParameters tp;
        tp.type = "float";
        tp.mutationProb_ = mutProb;
        tp.importanceCoeff_ = 1.0;
        NEAT::FloatTraitParameters d;
        d.min = mn;
        d.max = mx;
        d.mutPower = 0.5;
        d.mutReplaceProb = 0.5;
        tp.details_ = d;
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
        rng.seed(42);
        std::map<std::string, TraitParameters> tp;
        tp["speed"] = makeIntTrait(0, 10);
        tp["rate"] = makeFloatTrait(-1.0, 1.0);

        TraitParameters strTp;
        strTp.type = "str";
        strTp.mutationProb_ = 1.0;
        StringTraitParameters sdet;
        sdet.set = {"red", "green", "blue"};
        sdet.probs = {0.2, 0.5, 0.3};
        strTp.details_ = sdet;
        tp["color"] = strTp;

        TraitParameters isetTp;
        isetTp.type = "intset";
        isetTp.mutationProb_ = 1.0;
        IntSetTraitParameters idet;
        idet.set = {IntSetElement{1}, IntSetElement{2}, IntSetElement{3}};
        idet.probs = {0.3, 0.3, 0.4};
        isetTp.details_ = idet;
        tp["mode"] = isetTp;

        TraitParameters fsetTp;
        fsetTp.type = "floatset";
        fsetTp.mutationProb_ = 1.0;
        FloatSetTraitParameters fdet;
        fdet.set = {FloatSetElement{0.5}, FloatSetElement{1.5}};
        fdet.probs = {0.5, 0.5};
        fsetTp.details_ = fdet;
        tp["gain"] = fsetTp;

        Gene g;
        g.initTraits(tp, rng);
        CHECK(g.traits_.count("speed") == 1);
        CHECK(g.traits_.count("rate") == 1);
        CHECK(g.traits_.count("color") == 1);
        CHECK(g.traits_.count("mode") == 1);
        CHECK(g.traits_.count("gain") == 1);

        const int speed = std::get<int>(g.traits_["speed"].value);
        CHECK(speed >= 0 && speed <= 10);
        const double rate = std::get<double>(g.traits_["rate"].value);
        CHECK(rate >= -1.0 && rate <= 1.0);
        const std::string color = std::get<std::string>(g.traits_["color"].value);
        CHECK(color == "red" || color == "green" || color == "blue");
    }

    // --- Empty sets throw ---------------------------------------------------
    {
        RNG rng;
        rng.seed(1);
        for (const char *kind : {"str", "intset", "floatset"}) {
            std::map<std::string, TraitParameters> tp;
            TraitParameters p;
            p.type = kind;
            if (std::string(kind) == "str") {
                p.details_ = StringTraitParameters();
            } else if (std::string(kind) == "intset") {
                p.details_ = IntSetTraitParameters();
            } else {
                p.details_ = FloatSetTraitParameters();
            }
            tp["x"] = p;
            Gene g;
            bool threw = false;
            try {
                g.initTraits(tp, rng);
            } catch (const std::runtime_error &) {
                threw = true;
            }
            CHECK(threw);
        }
    }

    // --- MateTraits: happy path + type mismatch ------------------------------
    {
        RNG rng;
        rng.seed(5);
        Gene a, b;
        Trait ta, tb;
        ta.value = 4;
        tb.value = 8;
        a.traits_["k"] = ta;
        b.traits_["k"] = tb;
        a.mateTraits(b.traits_, rng);
        const int v = std::get<int>(a.traits_["k"].value);
        // Either parent (4/8) or the average (6).
        CHECK(v == 4 || v == 6 || v == 8);
    }
    {
        RNG rng;
        rng.seed(5);
        Gene a, b;
        Trait ta, tb;
        ta.value = 4;
        tb.value = std::string("oops");
        a.traits_["k"] = ta;
        b.traits_["k"] = tb;
        bool threw = false;
        try {
            a.mateTraits(b.traits_, rng);
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
        a.traits_["i"] = t1;
        b.traits_["i"] = t2;
        a.traits_["s"] = s1;
        b.traits_["s"] = s2;
        const std::map<std::string, double> dist = a.getTraitDistances(b.traits_);
        CHECK(dist.count("i") == 1 && near(dist.at("i"), 7.0));
        CHECK(dist.count("s") == 1 && near(dist.at("s"), 1.0));

        // Identical strings => distance 0.
        b.traits_["s"] = s1;
        const std::map<std::string, double> dist2 = a.getTraitDistances(b.traits_);
        CHECK(near(dist2.at("s"), 0.0));

        // Mismatched variant types throw.
        b.traits_["i"].value = std::string("nope");
        bool threw = false;
        try {
            (void)a.getTraitDistances(b.traits_);
        } catch (const std::runtime_error &) {
            threw = true;
        }
        CHECK(threw);
    }
    {
        // Dependency gating: distance is skipped unless both sides have the gate trait set to one of dep_values.
        Gene a, b;
        Trait gateA, gateB, vA, vB;
        gateA.value = std::string("on");
        gateB.value = std::string("off");
        vA.value = 1;
        vB.value = 9;
        vA.depKey = "gate";
        vB.depKey = "gate";
        vA.depValues.emplace_back(std::string("on"));
        vB.depValues.emplace_back(std::string("on"));
        a.traits_["gate"] = gateA;
        b.traits_["gate"] = gateB;
        a.traits_["v"] = vA;
        b.traits_["v"] = vB;
        const std::map<std::string, double> dist = a.getTraitDistances(b.traits_);
        CHECK(dist.count("v") == 0);

        b.traits_["gate"] = gateA;  // both "on" now
        const std::map<std::string, double> dist2 = a.getTraitDistances(b.traits_);
        CHECK(dist2.count("v") == 1 && near(dist2.at("v"), 8.0));
    }

    // --- MutateTraits stays in range ------------------------------------------
    {
        RNG rng;
        rng.seed(9);
        std::map<std::string, TraitParameters> tp;
        tp["speed"] = makeIntTrait(0, 10);
        Gene g;
        g.initTraits(tp, rng);
        for (int i = 0; i < 50; ++i) {
            (void)g.mutateTraits(tp, rng);
            const int v = std::get<int>(g.traits_["speed"].value);
            CHECK(v >= 0 && v <= 10);
        }
    }

    // --- LinkGene / NeuronGene basics ------------------------------------------
    {
        LinkGene l(1, 2, 7, 0.5, false);
        CHECK(l.fromNeuronID() == 1);
        CHECK(l.toNeuronID() == 2);
        CHECK(l.innovationID() == 7);
        CHECK(near(l.getWeight(), 0.5));
        CHECK(!l.isRecurrent());
        CHECK(!l.isLoopedRecurrent());
        LinkGene loop(4, 4, 8, 1.0, true);
        CHECK(loop.isLoopedRecurrent());
        CHECK((LinkGene(1, 2, 7, 0.0) == LinkGene(9, 9, 7, 5.0)));

        NeuronGene n(HIDDEN, 42, 0.5);
        n.init(1.0, 0.0, 1.0, 0.1, TANH);
        CHECK(n.id() == 42);
        CHECK(n.type() == HIDDEN);
        CHECK(n.actFunction_ == TANH);
        // operator== compares ID and Type.
        CHECK((NeuronGene(HIDDEN, 42, 0.0) == NeuronGene(HIDDEN, 42, 1.0)));
        CHECK(!((NeuronGene(HIDDEN, 42, 0.0) == NeuronGene(OUTPUT, 42, 0.0))));
    }

    if (g_failures != 0) {
        std::cerr << "Test failed: TestTraitsGenes with " << g_failures << " failure(s)\n";
        return 1;
    }
    std::cout << "Test passed: TestTraitsGenes\n";
    return 0;
}
