// Tests for NEAT::Substrate (src/Substrate.h/.cpp).
#include <iostream>
#include <vector>

#include "Genes.h"
#include "Substrate.h"

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
        CHECK(!s.leaky_);
        CHECK(!s.withDistance_);
        CHECK(!s.queryWeightsOnly_);
        CHECK(s.allowInputHiddenLinks_);
        CHECK(s.allowInputOutputLinks_);
        CHECK(!s.allowHiddenHiddenLinks_);
        CHECK(s.allowHiddenOutputLinks_);
        CHECK(s.getMaxDims() == 0);
        // 0 dims * 2 + bias.
        CHECK(s.getMinCPPNInputs() == 1);
        // link on/off + weight.
        CHECK(s.getMinCPPNOutputs() == 2);
    }

    // Dimensionality math over mixed coordinate sizes.
    {
        std::vector<std::vector<double>> ins{{0.0, 0.0}, {1.0, 1.0}};
        std::vector<std::vector<double>> hid{{0.0, 0.0}};
        std::vector<std::vector<double>> outs{{0.0, 0.0}};
        Substrate s(ins, hid, outs);
        CHECK(s.getMaxDims() == 2);
        CHECK(s.getMinCPPNInputs() == 2 * 2 + 1);
        CHECK(s.getMinCPPNOutputs() == 2);

        // 3-D inputs raise the max.
        s.inputCoords_.push_back({0.0, 0.0, 0.0});
        CHECK(s.getMaxDims() == 3);
        CHECK(s.getMinCPPNInputs() == 3 * 2 + 1);
    }

    // Flag-driven CPPN output dimensionality.
    {
        Substrate s;
        s.queryWeightsOnly_ = true;
        CHECK(s.getMinCPPNOutputs() == 1);
        s.leaky_ = true;
        CHECK(s.getMinCPPNOutputs() == 3);  // 1 + time_const + bias
        s.queryWeightsOnly_ = false;
        CHECK(s.getMinCPPNOutputs() == 4);  // 2 + time_const + bias
        s.withDistance_ = true;
        CHECK(s.getMinCPPNInputs() == 0 * 2 + 1 + 1);
    }

    // Custom connectivity set/clear round-trip.
    {
        Substrate s;
        CHECK(s.customConnectivity_.empty());
        std::vector<std::vector<int>> conns{
            {static_cast<int>(INPUT), 0, static_cast<int>(OUTPUT), 0},
            {static_cast<int>(INPUT), 1, static_cast<int>(HIDDEN), 0},
        };
        s.setCustomConnectivity(conns);
        CHECK(s.customConnectivity_.size() == 2);
        CHECK(s.customConnectivity_[0][0] == static_cast<int>(INPUT));
        CHECK(s.customConnectivity_[0][2] == static_cast<int>(OUTPUT));
        CHECK(s.customConnectivity_[1][3] == 0);
        s.clearCustomConnectivity();
        CHECK(s.customConnectivity_.empty());
    }

    // PrintInfo must not crash (smoke).
    {
        Substrate s;
        s.printInfo();
    }

    if (g_failures != 0) {
        std::cerr << "Test failed: TestSubstrate with " << g_failures << " failure(s)\n";
        return 1;
    }
    std::cout << "Test passed: TestSubstrate\n";
    return 0;
}
