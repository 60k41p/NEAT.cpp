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

#include "Genome.h"
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
        const std::filesystem::path fixture = std::filesystem::path(NEATCPP_TEST_DATA_DIR) / "minimal.NEAT";
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

    // Regression: an existing file without the NEAT_ParametersStart marker must
    // return non-zero quickly (it used to loop forever on EOF).
    {
        const auto tmp = std::filesystem::temp_directory_path() / "neatcpp_test_garbage_params.NEAT";
        {
            std::ofstream out(tmp);
            out << "no markers here at all\n";
        }
        Parameters p;
        p.Reset();
        CHECK(p.Load(tmp.string().c_str()) != 0);
        std::error_code ec;
        std::filesystem::remove(tmp, ec);
    }

    // Regression: a truncated parameters file (start marker, no end marker)
    // must return non-zero instead of spinning on EOF.
    {
        Parameters q;
        q.Reset();
        const auto src = std::filesystem::temp_directory_path() / "neatcpp_test_good_params.NEAT";
        const auto trunc = std::filesystem::temp_directory_path() / "neatcpp_test_trunc_params.NEAT";
        q.Save(src.string().c_str());
        std::ifstream in(src);
        std::string body((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
        body.erase(body.find("NEAT_ParametersEnd"));
        {
            std::ofstream out(trunc);
            out << body;
        }
        Parameters p;
        p.Reset();
        CHECK(p.Load(trunc.string().c_str()) != 0);
        std::error_code ec;
        std::filesystem::remove(src, ec);
        std::filesystem::remove(trunc, ec);
    }

    // v2 defaults: retuned evolution dynamics and new algorithm controls.
    {
        Parameters p;
        p.Reset();
        CHECK(p.YoungAgeTreshold == 15);
        CHECK(p.OldAgeTreshold == 80);
        CHECK(Near(p.OldAgePenalty, 0.75));
        CHECK(Near(p.PreferFitterParentRate, 0.5));
        CHECK(p.TruncationSelection == true);
        CHECK(p.TournamentSelection == false);
        CHECK(Near(p.EliteFraction, 0.0001));
        CHECK(Near(p.MutateAddLinkFromBiasProb, 0.01));
        CHECK(Near(p.RecurrentProb, 0.2));
        CHECK(Near(p.RecurrentLoopProb, 0.5));
        CHECK(Near(p.MutateWeightsProb, 0.8));
        CHECK(Near(p.WeightMutationMaxPower, 1.5));
        CHECK(Near(p.WeightReplacementMaxPower, 3.0));
        CHECK(Near(p.MinActivationA, 4.9) && Near(p.MaxActivationA, 4.9));
        CHECK(Near(p.WeightDiffCoeff, 0.1));
        CHECK(Near(p.MinCompatTreshold, 0.1));
        CHECK(Near(p.CompatTresholdModifier, 0.2));
        CHECK(p.NeuronTries == 64);
        CHECK(p.ParentSelectionMode == LEGACY_SELECTION);
        CHECK(p.WeightMutationDistribution == UNIFORM_MUTATION);
        CHECK(p.SpeciesRepresentativeSelection == FIRST_REPRESENTATIVE);
        CHECK(p.OffspringAllocation == LARGEST_REMAINDER);
        CHECK(p.CompatibilityThresholdControl == LEGACY_COMPATIBILITY_THRESHOLD);
        CHECK(p.FitnessScaling == SHIFTED_FITNESS_SCALING);
        CHECK(Near(p.MutationOperatorsPerOffspring, 1.0));
        CHECK(p.RequireEvaluatedGenomes == false);
        CHECK(p.RejectNonFiniteFitness == false);
        std::string error;
        CHECK(p.Validate(&error));
        CHECK(error.empty());
    }

    // Validate rejects broken configurations with a message.
    {
        Parameters p;
        p.Reset();
        p.PopulationSize = 0;
        std::string error;
        CHECK(!p.Validate(&error));
        CHECK(!error.empty());
        p.Reset();
        p.RankSelectionPressure = 5.0;
        CHECK(!p.Validate());
        p.Reset();
        p.MultipointCrossoverRate = 0.8;
        p.SinglePointCrossoverRate = 0.5;
        CHECK(!p.Validate());
    }

    // ConfigureSpiking / ConfigureMcCullochPitts presets.
    {
        Parameters p;
        p.Reset();
        CHECK(Near(p.ActivationFunction_SpikingLIF_Prob, 0.0));
        p.ConfigureSpiking(false);
        CHECK(Near(p.ActivationFunction_SpikingLIF_Prob, 0.65));
        CHECK(Near(p.MutateNeuronSpikingParametersProb, 0.25));
        CHECK(p.Validate());
        p.ConfigureMcCullochPitts(true, false);
        CHECK(Near(p.ActivationFunction_McCullochPitts_Prob, 1.0));
        CHECK(Near(p.ActivationFunction_SpikingLIF_Prob, 0.0));
        CHECK(p.Validate());
    }

    // Serialize/Deserialize round-trips the full field set.
    {
        Parameters p;
        p.Reset();
        p.PopulationSize = 64;
        p.ParentSelectionMode = TOURNAMENT;
        p.WeightMutationDistribution = GAUSSIAN_MUTATION;
        p.MutateNeuronSpikingParametersProb = 0.3;
        const std::string data = p.Serialize();
        const Parameters q = Parameters::Deserialize(data);
        CHECK(q.PopulationSize == 64);
        CHECK(q.ParentSelectionMode == TOURNAMENT);
        CHECK(q.WeightMutationDistribution == GAUSSIAN_MUTATION);
        CHECK(Near(q.MutateNeuronSpikingParametersProb, 0.3));
        CHECK(q.Validate());
    }

    // Custom-constraints callables (legacy pointer and std::function).
    {
        Parameters p;
        p.Reset();
        GenomeInitStruct init;
        init.NumInputs = 2;
        init.NumOutputs = 1;
        init.SeedType = PERCEPTRON;
        Genome g(p, init);
        CHECK(!p.FailsCustomConstraints(g));
        p.SetCustomConstraintsFunction([](Genome &) { return true; });
        CHECK(p.FailsCustomConstraints(g));
        CHECK(p.CustomConstraints == nullptr);
    }

    if (g_failures != 0) {
        std::cerr << "Test failed: TestParameters with " << g_failures << " failure(s)\n";
        return 1;
    }
    std::cout << "Test passed: TestParameters\n";
    return 0;
}
