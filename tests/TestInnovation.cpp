// Tests for NEAT::InnovationDatabase (src/Innovation.h/.cpp).
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>

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
    // Init takes the last-used IDs; the next free ID is one past that.
    {
        InnovationDatabase db;
        db.Init(1, 1);
        const int id1 = db.AddLinkInnovation(1, 2);
        CHECK(id1 == 2);
        const int id2 = db.AddLinkInnovation(2, 3);
        CHECK(id2 == 3);
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
        CHECK(nid == 21);
        CHECK(db.FindNeuronID(1, 2) == nid);
        CHECK(db.FindLastNeuronID(1, 2) == nid);
        CHECK(db.CheckInnovation(1, 2, NEW_NEURON) == 11);
        const int nid2 = db.AddNeuronInnovation(1, 2, HIDDEN);
        CHECK(nid2 == 22);
        // First match vs last match differ once duplicated.
        CHECK(db.CheckInnovation(1, 2, NEW_NEURON) == 11);
        CHECK(db.CheckLastInnovation(1, 2, NEW_NEURON) == 12);
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
            CHECK(db2.CheckInnovation(1, 2, NEW_LINK) == 101);
            CHECK(db2.FindNeuronID(2, 3) != -1);
            // Counters advanced past the added entries.
            CHECK(db2.AddLinkInnovation(7, 8) == 103);
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

    // Index heals after external edits; Serialize round-trips; out-of-range access throws.
    {
        InnovationDatabase db;
        db.Init(1, 1);
        db.AddLinkInnovation(1, 2);
        db.AddNeuronInnovation(1, 2, HIDDEN);
        CHECK(db.CheckInnovation(1, 2, NEW_LINK) != -1);
        const std::string data = db.Serialize();
        const InnovationDatabase db2 = InnovationDatabase::Deserialize(data);
        CHECK(db2.m_Innovations.size() == db.m_Innovations.size());
        CHECK(db2.CheckInnovation(1, 2, NEW_LINK) == db.CheckInnovation(1, 2, NEW_LINK));
        CHECK(db2.ValidateInnovationState());
        // External edit bypassing Add*: lookups still work via lazy rebuild.
        db.m_Innovations.emplace_back(Innovation(999, NEW_LINK, 5, 6, NONE, -1));
        CHECK(db.CheckInnovation(5, 6, NEW_LINK) == 999);
        CHECK(db.CheckAllInnovations(5, 6, NEW_LINK).size() == 1);
        CHECK(!db.ValidateInnovationState());  // synthetic ID exceeds counters
        bool threw = false;
        try {
            (void)db2.GetInnovationByIdx(1000000);
        } catch (const std::out_of_range &) {
            threw = true;
        }
        CHECK(threw);
        // Counters overflow-guard at INT_MAX.
        InnovationDatabase full;
        full.Init(std::numeric_limits<int>::max() - 2, 1);
        (void)full.AddLinkInnovation(1, 2);
        threw = false;
        try {
            (void)full.AddLinkInnovation(3, 4);
        } catch (const std::overflow_error &) {
            threw = true;
        }
        CHECK(threw);
    }

    if (g_failures != 0) {
        std::cerr << "Test failed: TestInnovation with " << g_failures << " failure(s)\n";
        return 1;
    }
    std::cout << "Test passed: TestInnovation\n";
    return 0;
}
