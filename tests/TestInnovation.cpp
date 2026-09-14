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

    NEAT::Genome MakeSeedGenome() {
        NEAT::Parameters params;
        params.Reset();
        NEAT::GenomeInitStruct init;
        init.NumInputs = 3;
        init.NumOutputs = 1;
        init.SeedType = NEAT::PERCEPTRON;
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
        CHECK(db.m_Innovations.empty());
        CHECK(db.CheckInnovation(1, 2, NEW_LINK) == -1);
        CHECK(db.CheckLastInnovation(1, 2, NEW_LINK) == -1);
        CHECK(db.CheckAllInnovations(1, 2, NEW_LINK).empty());
        CHECK(db.FindNeuronID(1, 2) == -1);
    }

    // Link innovations: add/find, last-wins semantics.
    {
        InnovationDatabase db;
        db.Init(1, 1);
        const int id1 = db.AddLinkInnovation(1, 2);
        CHECK(id1 == 1);
        const int id2 = db.AddLinkInnovation(2, 3);
        CHECK(id2 == 2);
        CHECK(db.CheckInnovation(1, 2, NEW_LINK) == id1);
        CHECK(db.CheckLastInnovation(1, 2, NEW_LINK) == id1);
        CHECK(db.CheckInnovation(1, 2, NEW_NEURON) == -1);  // type matters
        CHECK(db.GetInnovationByIdx(0).ID() == id1);
        CHECK(db.GetInnovationByIdx(1).InnovType() == NEW_LINK);
    }

    // Neuron innovations: IDs advance, Find* resolves the split link.
    {
        InnovationDatabase db;
        db.Init(10, 20);
        const int nid = db.AddNeuronInnovation(1, 2, HIDDEN);
        CHECK(nid == 20);
        CHECK(db.FindNeuronID(1, 2) == nid);
        CHECK(db.FindLastNeuronID(1, 2) == nid);
        CHECK(db.CheckInnovation(1, 2, NEW_NEURON) == 10);
        const int nid2 = db.AddNeuronInnovation(1, 2, HIDDEN);
        CHECK(nid2 == 21);
        // First match vs last match differ once duplicated.
        CHECK(db.CheckInnovation(1, 2, NEW_NEURON) == 10);
        CHECK(db.CheckLastInnovation(1, 2, NEW_NEURON) == 11);
        CHECK(db.FindLastNeuronID(1, 2) == nid2);
        CHECK(db.CheckAllInnovations(1, 2, NEW_NEURON).size() == 2);
    }

    // Flush + Init(genome) rebuilds link entries from the seed genome.
    {
        InnovationDatabase db;
        db.Init(1, 1);
        db.AddLinkInnovation(1, 2);
        CHECK(!db.m_Innovations.empty());
        db.Flush();
        CHECK(db.m_Innovations.empty());

        const Genome seed = MakeSeedGenome();
        db.Init(seed);
        CHECK(db.m_Innovations.size() == seed.NumLinks());
        for (const auto &innov : db.m_Innovations) {
            CHECK(innov.InnovType() == NEW_LINK);
        }
    }

    // Save/Init(ifstream) round-trip preserves entries and counters.
    {
        const auto tmp = std::filesystem::temp_directory_path() / "multineat_test_innov.db";
        {
            InnovationDatabase db;
            db.Init(100, 200);
            db.AddLinkInnovation(1, 2);
            db.AddNeuronInnovation(2, 3, HIDDEN);
            FILE *f = std::fopen(tmp.string().c_str(), "w");
            CHECK(f != nullptr);
            db.Save(f);
            std::fclose(f);
        }
        {
            InnovationDatabase db2;
            std::ifstream in(tmp.string());
            CHECK(in.is_open());
            db2.Init(in);
            CHECK(db2.m_Innovations.size() == 2);
            CHECK(db2.CheckInnovation(1, 2, NEW_LINK) == 100);
            CHECK(db2.FindNeuronID(2, 3) != -1);
            // Counters advanced past the added entries.
            CHECK(db2.AddLinkInnovation(7, 8) == 102);
        }
        std::error_code ec;
        std::filesystem::remove(tmp, ec);
    }

    // Regression: garbage (no InnovationDatabaseStart marker) must throw
    // instead of spinning on EOF forever.
    {
        const auto tmp = std::filesystem::temp_directory_path() / "neatcpp_test_garbage_innov.db";
        {
            std::ofstream out(tmp);
            out << "no markers here\n";
        }
        InnovationDatabase db;
        std::ifstream in(tmp.string());
        bool threw = false;
        try {
            db.Init(in);
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
