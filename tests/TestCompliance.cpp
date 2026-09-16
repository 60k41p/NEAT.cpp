// Compliance tests: project-wide invariants that are not tied to one class.
// (Mastering CMake: "tests do not necessarily have to involve running some
// part of the software" — these check saved-file contracts and Reset sanity.)
//
// CTest-Labels: Compliance;Fast
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>

#include "Genome.h"
#include "NeuralNetwork.h"
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

    bool ProbOk(Real v) { return v >= 0.0 && v <= 1.0; }

}  // namespace

int TestCompliance(int argc, char *argv[]) {
    (void)argc;
    (void)argv;
    using namespace NEAT;

    // Reset() probability-like knobs stay in [0,1]; sizes/thresholds sane.
    {
        Parameters p;
        p.Reset();
        CHECK(ProbOk(p.MutateAddNeuronProb));
        CHECK(ProbOk(p.MutateAddLinkProb));
        CHECK(ProbOk(p.MutateRemLinkProb));
        CHECK(ProbOk(p.MutateRemSimpleNeuronProb));
        CHECK(ProbOk(p.MutateWeightsProb));
        CHECK(ProbOk(p.CrossoverRate));
        CHECK(ProbOk(p.OverallMutationRate));
        CHECK(ProbOk(p.InterspeciesCrossoverRate));
        CHECK(ProbOk(p.MultipointCrossoverRate));
        CHECK(ProbOk(p.SurvivalRate));
        CHECK(p.PopulationSize > 0);
        // CompatTreshold is clamped into [MinCompatTreshold, MaxCompatTreshold].
        CHECK(p.CompatTreshold > 0.0 && p.MinCompatTreshold >= 0.0);
        CHECK(p.MaxCompatTreshold >= p.MinCompatTreshold);
        CHECK(p.MinNeuronBias <= p.MaxNeuronBias);
        CHECK(p.MinWeight <= p.MaxWeight);
        CHECK(p.TournamentSize > 0);
        // Reset() defaults validate cleanly.
        std::string validation_error;
        CHECK(p.Validate(&validation_error));
        CHECK(validation_error.empty());
        // New algorithm controls stay in their documented domains.
        CHECK(p.RankSelectionPressure >= 1.0 && p.RankSelectionPressure <= 2.0);
        CHECK(p.MutationOperatorsPerOffspring >= 1.0);
        CHECK(p.SinglePointCrossoverRate + p.BlendCrossoverRate + p.SimulatedBinaryCrossoverRate + p.MultipointCrossoverRate <= 1.0 + 1e-12);
    }

    // Saved Parameters files always carry the framing markers.
    {
        Parameters p;
        p.Reset();
        const auto tmp = std::filesystem::temp_directory_path() / "multineat_compliance_params.neat";
        p.Save(tmp.string().c_str());
        std::ifstream in(tmp.string());
        CHECK(in.is_open());
        std::string body((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
        CHECK(body.find("NEAT_ParametersStart") != std::string::npos);
        CHECK(body.find("NEAT_ParametersEnd") != std::string::npos);
        std::error_code ec;
        std::filesystem::remove(tmp, ec);
    }

    // Saved Genome files always carry GenomeStart/GenomeEnd + Neuron/Link rows.
    {
        Parameters p;
        p.Reset();
        GenomeInitStruct init;
        init.NumInputs = 3;
        init.NumOutputs = 1;
        Genome g(p, init);
        const auto tmp = std::filesystem::temp_directory_path() / "multineat_compliance_genome.txt";
        g.Save(tmp.string().c_str());
        std::ifstream in(tmp.string());
        CHECK(in.is_open());
        std::string body((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
        CHECK(body.find("GenomeStart") != std::string::npos);
        CHECK(body.find("GenomeEnd") != std::string::npos);
        CHECK(body.find("Neuron") != std::string::npos);
        CHECK(body.find("Link") != std::string::npos);
        std::error_code ec;
        std::filesystem::remove(tmp, ec);
    }

    // Saved NeuralNetwork files always carry NNstart/NNend.
    {
        NeuralNetwork net;
        net.Clear();
        const auto tmp = std::filesystem::temp_directory_path() / "multineat_compliance_nn.txt";
        net.Save(tmp.string().c_str());
        std::ifstream in(tmp.string());
        CHECK(in.is_open());
        std::string body((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
        CHECK(body.find("NNstart") != std::string::npos);
        CHECK(body.find("NNend") != std::string::npos);
        std::error_code ec;
        std::filesystem::remove(tmp, ec);
    }

    // Fresh seed genomes satisfy the default constraints (no dead ends that would make Population construction spin retrying).
    {
        Parameters p;
        p.Reset();
        GenomeInitStruct init;
        init.NumInputs = 3;
        init.NumOutputs = 1;
        Genome g(p, init);
        CHECK(!g.FailsConstraints(p));
        CHECK(g.NumLinks() > 0);
        CHECK(g.Validate());
    }

    // String serialization carries version markers and round-trips.
    {
        Parameters p;
        p.Reset();
        GenomeInitStruct init;
        init.NumInputs = 2;
        init.NumOutputs = 1;
        Genome g(p, init);
        const std::string data = g.Serialize();
        CHECK(data.find("GenomeFormat 4") != std::string::npos);
        CHECK(data.find("GenomeState") != std::string::npos);
        CHECK(Genome::Deserialize(data).IsIdenticalTo(g));
    }

    if (g_failures != 0) {
        std::cerr << "Test failed: TestCompliance with " << g_failures << " failure(s)\n";
        return 1;
    }
    std::cout << "Test passed: TestCompliance\n";
    return 0;
}
