// Tests for NEAT::InnovationDatabase (src/Innovation.h/.cpp).
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iostream>

#include "Genome.h"
#include "Innovation.h"
#include "Parameters.h"

namespace {

    int g_failures = 0;

#define CHECK(cond)                                                                         \
    do {                                                                                    \
        if (!(cond)) {                                                                      \
            std::cerr << "FAILED " << __FILE__ << ":" << __LINE__ << ": " << #cond << "\n"; \
            ++g_failures;                                                                   \
        }                                                                                   \
    } while (0)

    NEAT::Genome makeSeedGenome() {
        NEAT::Parameters params;
        params.reset();
        NEAT::GenomeInitStruct init;
        init.numInputs = 3;
        init.numOutputs = 1;
        init.seedType = NEAT::PERCEPTRON;
        return NEAT::Genome(params, init);
    }

}  // namespace

int TestInnovation(int argc, char *argv[]) {
    (void)argc;
    (void)argv;
    using namespace NEAT;

    // Empty database: lookups miss.
    {
        InnovationDatabase db;
        CHECK(db.innovations_.empty());
        CHECK(db.checkInnovation(1, 2, NEW_LINK) == -1);
        CHECK(db.checkLastInnovation(1, 2, NEW_LINK) == -1);
        CHECK(db.checkAllInnovations(1, 2, NEW_LINK).empty());
        CHECK(db.findNeuronID(1, 2) == -1);
    }

    // Link innovations: add/find, last-wins semantics.
    {
        InnovationDatabase db;
        db.init(1, 1);
        const int id1 = db.addLinkInnovation(1, 2);
        CHECK(id1 == 1);
        const int id2 = db.addLinkInnovation(2, 3);
        CHECK(id2 == 2);
        CHECK(db.checkInnovation(1, 2, NEW_LINK) == id1);
        CHECK(db.checkLastInnovation(1, 2, NEW_LINK) == id1);
        CHECK(db.checkInnovation(1, 2, NEW_NEURON) == -1);  // type matters
        CHECK(db.getInnovationByIndex(0).id() == id1);
        CHECK(db.getInnovationByIndex(1).innovType() == NEW_LINK);
    }

    // Neuron innovations: IDs advance, Find* resolves the split link.
    {
        InnovationDatabase db;
        db.init(10, 20);
        const int nid = db.addNeuronInnovation(1, 2, HIDDEN);
        CHECK(nid == 20);
        CHECK(db.findNeuronID(1, 2) == nid);
        CHECK(db.findLastNeuronID(1, 2) == nid);
        CHECK(db.checkInnovation(1, 2, NEW_NEURON) == 10);
        const int nid2 = db.addNeuronInnovation(1, 2, HIDDEN);
        CHECK(nid2 == 21);
        // First match vs last match differ once duplicated.
        CHECK(db.checkInnovation(1, 2, NEW_NEURON) == 10);
        CHECK(db.checkLastInnovation(1, 2, NEW_NEURON) == 11);
        CHECK(db.findLastNeuronID(1, 2) == nid2);
        CHECK(db.checkAllInnovations(1, 2, NEW_NEURON).size() == 2);
    }

    // Flush + Init(genome) rebuilds link entries from the seed genome.
    {
        InnovationDatabase db;
        db.init(1, 1);
        db.addLinkInnovation(1, 2);
        CHECK(!db.innovations_.empty());
        db.flush();
        CHECK(db.innovations_.empty());

        const Genome seed = makeSeedGenome();
        db.init(seed);
        CHECK(db.innovations_.size() == seed.numLinks());
        for (const Innovation &innov : db.innovations_) {
            CHECK(innov.innovType() == NEW_LINK);
        }
    }

    // Save/Init(ifstream) round-trip preserves entries and counters.
    {
        const std::filesystem::path tmp = std::filesystem::temp_directory_path() / "multineat_test_innov.db";
        {
            InnovationDatabase db;
            db.init(100, 200);
            db.addLinkInnovation(1, 2);
            db.addNeuronInnovation(2, 3, HIDDEN);
            FILE *f = std::fopen(tmp.string().c_str(), "w");
            CHECK(f != nullptr);
            db.save(f);
            std::fclose(f);
        }
        {
            InnovationDatabase db2;
            std::ifstream in(tmp.string());
            CHECK(in.is_open());
            db2.init(in);
            CHECK(db2.innovations_.size() == 2);
            CHECK(db2.checkInnovation(1, 2, NEW_LINK) == 100);
            CHECK(db2.findNeuronID(2, 3) != -1);
            // Counters advanced past the added entries.
            CHECK(db2.addLinkInnovation(7, 8) == 102);
        }
        std::error_code ec;
        std::filesystem::remove(tmp, ec);
    }

    // Regression: garbage (no InnovationDatabaseStart marker) must throw
    // instead of spinning on EOF forever.
    {
        const std::filesystem::path tmp = std::filesystem::temp_directory_path() / "neatcpp_test_garbage_innov.db";
        {
            std::ofstream out(tmp);
            out << "no markers here\n";
        }
        InnovationDatabase db;
        std::ifstream in(tmp.string());
        bool threw = false;
        try {
            db.init(in);
        } catch (...) {
            threw = true;
        }
        CHECK(threw);
        std::error_code ec;
        std::filesystem::remove(tmp, ec);
    }

    if (g_failures != 0) {
        std::cerr << "Test failed: TestInnovation with " << g_failures << " failure(s)\n";
        return 1;
    }
    std::cout << "Test passed: TestInnovation\n";
    return 0;
}
