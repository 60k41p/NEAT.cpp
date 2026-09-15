// Tests for NEAT::Parameters save/load/reset (src/Parameters.h/.cpp).
//
// NOTE: examples/DefaultConfig.NEAT is intentionally NOT loaded here. It is a
// legacy file without NEAT_ParametersStart/End markers, and Parameters::Load
// spins forever searching for the start marker on such files. The round-trip
// below uses freshly Saved files plus tests/data/minimal.NEAT instead.
//
// CTest-Labels: Unit;IO;Fast
// CTest-Timeout: 60
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>

#include "Parameters.h"

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

    bool near(Real a, Real b, Real eps = 1e-9) { return std::fabs(a - b) <= eps; }

    std::string readWholeFile(const std::filesystem::path &p) {
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
        p.reset();
        CHECK(p.populationSize == 300);
        CHECK(p.speciation == true);
        CHECK(p.dynamicCompatibility == true);
        CHECK(p.minSpecies == 5);
        CHECK(p.maxSpecies == 10);
        CHECK(p.allowClones == true);
        CHECK(p.survivalRate > 0.0 && p.survivalRate <= 1.0);
        CHECK(p.crossoverRate >= 0.0 && p.crossoverRate <= 1.0);
        CHECK(p.compatTreshold > 0.0);
        CHECK(p.mutateAddNeuronProb >= 0.0 && p.mutateAddNeuronProb <= 1.0);
        CHECK(p.mutateAddLinkProb >= 0.0 && p.mutateAddLinkProb <= 1.0);
        CHECK(p.customConstraints == nullptr);
    }

    // Save emits the framing markers (compliance-relevant contract).
    {
        Parameters p;
        p.reset();
        const std::filesystem::path tmp = std::filesystem::temp_directory_path() / "multineat_test_params.neat";
        p.save(tmp.string().c_str());
        const std::string body = readWholeFile(tmp);
        CHECK(body.find("NEAT_ParametersStart") != std::string::npos);
        CHECK(body.find("NEAT_ParametersEnd") != std::string::npos);
        CHECK(body.find("PopulationSize") != std::string::npos);
        std::error_code ec;
        std::filesystem::remove(tmp, ec);
    }

    // Save/Load round-trip preserves edited values.
    {
        Parameters p;
        p.reset();
        p.populationSize = 42;
        p.speciation = false;
        p.compatTreshold = 3.5;
        p.mutateAddNeuronProb = 0.123;
        p.mutateWeightsProb = 0.9;
        p.survivalRate = 0.33;
        p.tournamentSize = 7;

        const std::filesystem::path tmp = std::filesystem::temp_directory_path() / "multineat_test_params_rt.neat";
        p.save(tmp.string().c_str());

        Parameters q;
        q.reset();
        // ifstream overload
        {
            std::ifstream in(tmp.string());
            CHECK(in.is_open());
            CHECK(q.load(in) == 0);
        }
        CHECK(q.populationSize == 42);
        CHECK(q.speciation == false);
        CHECK(near(q.compatTreshold, 3.5));
        CHECK(near(q.mutateAddNeuronProb, 0.123));
        CHECK(near(q.mutateWeightsProb, 0.9));
        CHECK(near(q.survivalRate, 0.33));
        CHECK(q.tournamentSize == 7);

        // const char* overload on the same file.
        Parameters r;
        r.reset();
        CHECK(r.load(tmp.string().c_str()) == 0);
        CHECK(r.populationSize == 42);

        std::error_code ec;
        std::filesystem::remove(tmp, ec);
    }

    // Loading the curated minimal fixture works and overrides Reset values.
    {
        const std::filesystem::path fixture = std::filesystem::path(NEATCPP_TEST_DATA_DIR) / "minimal.NEAT";
        CHECK(std::filesystem::exists(fixture));
        Parameters p;
        p.reset();
        CHECK(p.load(fixture.string().c_str()) == 0);
        CHECK(p.populationSize == 20);
        CHECK(p.speciation == true);
        CHECK(near(p.compatTreshold, 4.25));
    }

    // Missing file is a silent no-op returning 0 (documented behavior).
    {
        Parameters p;
        p.reset();
        p.populationSize = 77;
        CHECK(p.load("/nonexistent/path/that/should/not/exist.NEAT") == 0);
        CHECK(p.populationSize == 77);  // untouched
    }

    // Regression: an existing file without the NEAT_ParametersStart marker must
    // return non-zero quickly (it used to loop forever on EOF).
    {
        const std::filesystem::path tmp = std::filesystem::temp_directory_path() / "neatcpp_test_garbage_params.NEAT";
        {
            std::ofstream out(tmp);
            out << "no markers here at all\n";
        }
        Parameters p;
        p.reset();
        CHECK(p.load(tmp.string().c_str()) != 0);
        std::error_code ec;
        std::filesystem::remove(tmp, ec);
    }

    // Regression: a truncated parameters file (start marker, no end marker)
    // must return non-zero instead of spinning on EOF.
    {
        Parameters q;
        q.reset();
        const std::filesystem::path src = std::filesystem::temp_directory_path() / "neatcpp_test_good_params.NEAT";
        const std::filesystem::path trunc = std::filesystem::temp_directory_path() / "neatcpp_test_trunc_params.NEAT";
        q.save(src.string().c_str());
        std::ifstream in(src);
        std::string body((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
        body.erase(body.find("NEAT_ParametersEnd"));
        {
            std::ofstream out(trunc);
            out << body;
        }
        Parameters p;
        p.reset();
        CHECK(p.load(trunc.string().c_str()) != 0);
        std::error_code ec;
        std::filesystem::remove(src, ec);
        std::filesystem::remove(trunc, ec);
    }

    if (g_failures != 0) {
        std::cerr << "Test failed: TestParameters with " << g_failures << " failure(s)\n";
        return 1;
    }
    std::cout << "Test passed: TestParameters\n";
    return 0;
}
