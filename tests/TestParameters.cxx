// Tests for NEAT::Parameters save/load/reset (src/Parameters.h/.cpp).
//
// NOTE: examples/DefaultConfig.NEAT is intentionally NOT loaded here. It is a
// legacy file without NEAT_ParametersStart/End markers, and Parameters::Load
// spins forever searching for the start marker on such files. The round-trip
// below uses freshly Saved files plus tests/data/minimal.NEAT instead.
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>

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

    bool Near(double a, double b, double eps = 1e-9) { return std::fabs(a - b) <= eps; }

    std::string ReadWholeFile(const std::filesystem::path &p) {
        std::ifstream in(p, std::ios::binary);
        return std::string((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    }

}  // namespace

int TestParameters(int argc, char *argv[]) {
    (void)argc;
    (void)argv;
    using namespace NEAT;

    // Reset() establishes documented defaults.
    {
        Parameters p;
        p.Reset();
        CHECK(p.PopulationSize == 300);
        CHECK(p.Speciation == true);
        CHECK(p.DynamicCompatibility == true);
        CHECK(p.MinSpecies == 5);
        CHECK(p.MaxSpecies == 10);
        CHECK(p.AllowClones == true);
        CHECK(p.SurvivalRate > 0.0 && p.SurvivalRate <= 1.0);
        CHECK(p.CrossoverRate >= 0.0 && p.CrossoverRate <= 1.0);
        CHECK(p.CompatTreshold > 0.0);
        CHECK(p.MutateAddNeuronProb >= 0.0 && p.MutateAddNeuronProb <= 1.0);
        CHECK(p.MutateAddLinkProb >= 0.0 && p.MutateAddLinkProb <= 1.0);
        CHECK(p.CustomConstraints == nullptr);
    }

    // Save emits the framing markers (compliance-relevant contract).
    {
        Parameters p;
        p.Reset();
        const auto tmp = std::filesystem::temp_directory_path() / "multineat_test_params.neat";
        p.Save(tmp.string().c_str());
        const std::string body = ReadWholeFile(tmp);
        CHECK(body.find("NEAT_ParametersStart") != std::string::npos);
        CHECK(body.find("NEAT_ParametersEnd") != std::string::npos);
        CHECK(body.find("PopulationSize") != std::string::npos);
        std::error_code ec;
        std::filesystem::remove(tmp, ec);
    }

    // Save/Load round-trip preserves edited values.
    {
        Parameters p;
        p.Reset();
        p.PopulationSize = 42;
        p.Speciation = false;
        p.CompatTreshold = 3.5;
        p.MutateAddNeuronProb = 0.123;
        p.MutateWeightsProb = 0.9;
        p.SurvivalRate = 0.33;
        p.TournamentSize = 7;

        const auto tmp = std::filesystem::temp_directory_path() / "multineat_test_params_rt.neat";
        p.Save(tmp.string().c_str());

        Parameters q;
        q.Reset();
        // ifstream overload
        {
            std::ifstream in(tmp.string());
            CHECK(in.is_open());
            CHECK(q.Load(in) == 0);
        }
        CHECK(q.PopulationSize == 42);
        CHECK(q.Speciation == false);
        CHECK(Near(q.CompatTreshold, 3.5));
        CHECK(Near(q.MutateAddNeuronProb, 0.123));
        CHECK(Near(q.MutateWeightsProb, 0.9));
        CHECK(Near(q.SurvivalRate, 0.33));
        CHECK(q.TournamentSize == 7);

        // const char* overload on the same file.
        Parameters r;
        r.Reset();
        CHECK(r.Load(tmp.string().c_str()) == 0);
        CHECK(r.PopulationSize == 42);

        std::error_code ec;
        std::filesystem::remove(tmp, ec);
    }

    // Loading the curated minimal fixture works and overrides Reset values.
    {
        const std::filesystem::path fixture = std::filesystem::path(MULTINEAT_TEST_DATA_DIR) / "minimal.NEAT";
        CHECK(std::filesystem::exists(fixture));
        Parameters p;
        p.Reset();
        CHECK(p.Load(fixture.string().c_str()) == 0);
        CHECK(p.PopulationSize == 20);
        CHECK(p.Speciation == true);
        CHECK(Near(p.CompatTreshold, 4.25));
    }

    // Missing file is a silent no-op returning 0 (documented behavior).
    {
        Parameters p;
        p.Reset();
        p.PopulationSize = 77;
        CHECK(p.Load("/nonexistent/path/that/should/not/exist.NEAT") == 0);
        CHECK(p.PopulationSize == 77);  // untouched
    }

    if (g_failures != 0) {
        std::cerr << "Test failed: TestParameters with " << g_failures << " failure(s)\n";
        return 1;
    }
    std::cout << "Test passed: TestParameters\n";
    return 0;
}
