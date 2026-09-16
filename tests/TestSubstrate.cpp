// Tests for NEAT::Substrate (src/Substrate.h/.cpp).
#include <iostream>
#include <vector>

#include "Genes.h"
#include "Substrate.h"
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

}  // namespace

int TestSubstrate(int argc, char *argv[]) {
    (void)argc;
    (void)argv;
    using namespace NEAT;

    // Default construction: documented flag state.
    {
        Substrate s;
        CHECK(!s.m_leaky);
        CHECK(!s.m_with_distance);
        CHECK(!s.m_query_weights_only);
        CHECK(s.m_allow_input_hidden_links);
        CHECK(s.m_allow_input_output_links);
        CHECK(!s.m_allow_hidden_hidden_links);
        CHECK(s.m_allow_hidden_output_links);
        CHECK(s.GetMaxDims() == 0);
        // 0 dims * 2 + bias.
        CHECK(s.GetMinCPPNInputs() == 1);
        // link on/off + weight.
        CHECK(s.GetMinCPPNOutputs() == 2);
    }

    // Dimensionality math over mixed coordinate sizes.
    {
        std::vector<std::vector<Real>> ins{{0.0, 0.0}, {1.0, 1.0}};
        std::vector<std::vector<Real>> hid{{0.0, 0.0}};
        std::vector<std::vector<Real>> outs{{0.0, 0.0}};
        Substrate s(ins, hid, outs);
        CHECK(s.GetMaxDims() == 2);
        CHECK(s.GetMinCPPNInputs() == 2 * 2 + 1);
        CHECK(s.GetMinCPPNOutputs() == 2);

        // 3-D inputs raise the max.
        s.m_input_coords.push_back({0.0, 0.0, 0.0});
        CHECK(s.GetMaxDims() == 3);
        CHECK(s.GetMinCPPNInputs() == 3 * 2 + 1);
    }

    // Flag-driven CPPN output dimensionality.
    {
        Substrate s;
        s.m_query_weights_only = true;
        CHECK(s.GetMinCPPNOutputs() == 1);
        s.m_leaky = true;
        CHECK(s.GetMinCPPNOutputs() == 3);  // 1 + time_const + bias
        s.m_query_weights_only = false;
        CHECK(s.GetMinCPPNOutputs() == 4);  // 2 + time_const + bias
        s.m_with_distance = true;
        CHECK(s.GetMinCPPNInputs() == 0 * 2 + 1 + 1);
    }

    // Custom connectivity set/clear round-trip (neurons must be set first).
    {
        Substrate s;
        CHECK(s.m_custom_connectivity.empty());
        s.m_input_coords = {{0.0, 0.0}, {1.0, 0.0}};
        s.m_hidden_coords = {{0.5, 0.5}};
        s.m_output_coords = {{0.5, 1.0}};
        std::vector<std::vector<int>> conns{
            {static_cast<int>(INPUT), 0, static_cast<int>(OUTPUT), 0},
            {static_cast<int>(INPUT), 1, static_cast<int>(HIDDEN), 0},
        };
        s.SetCustomConnectivity(conns);
        CHECK(s.m_custom_connectivity.size() == 2);
        CHECK(s.m_custom_connectivity[0][0] == static_cast<int>(INPUT));
        CHECK(s.m_custom_connectivity[0][2] == static_cast<int>(OUTPUT));
        CHECK(s.m_custom_connectivity[1][3] == 0);
        s.ClearCustomConnectivity();
        CHECK(s.m_custom_connectivity.empty());
    }

    // Malformed custom connectivity is rejected.
    {
        Substrate s;
        s.m_input_coords = {{0.0}};
        s.m_output_coords = {{1.0}};
        bool threw = false;
        try {
            std::vector<std::vector<int>> bad{{static_cast<int>(INPUT), 0, static_cast<int>(OUTPUT)}};
            s.SetCustomConnectivity(bad);
        } catch (const std::invalid_argument &) {
            threw = true;
        }
        CHECK(threw);
        threw = false;
        try {
            std::vector<std::vector<int>> bad{{static_cast<int>(INPUT), 7, static_cast<int>(OUTPUT), 0}};
            s.SetCustomConnectivity(bad);
        } catch (const std::out_of_range &) {
            threw = true;
        }
        CHECK(threw);
        CHECK(s.m_custom_connectivity.empty());
    }

    // PrintInfo must not crash (smoke).
    {
        Substrate s;
        s.PrintInfo();
    }

    if (g_failures != 0) {
        std::cerr << "Test failed: TestSubstrate with " << g_failures << " failure(s)\n";
        return 1;
    }
    std::cout << "Test passed: TestSubstrate\n";
    return 0;
}
