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

    bool probOk(Real v) { return v >= 0.0 && v <= 1.0; }

}  // namespace

int TestCompliance(int argc, char *argv[]) {
    (void)argc;
    (void)argv;
    using namespace NEAT;

    // Reset() probability-like knobs stay in [0,1]; sizes/thresholds sane.
    {
        Parameters p;
        p.reset();
        CHECK(probOk(p.mutateAddNeuronProb));
        CHECK(probOk(p.mutateAddLinkProb));
        CHECK(probOk(p.mutateRemLinkProb));
        CHECK(probOk(p.mutateRemSimpleNeuronProb));
        CHECK(probOk(p.mutateWeightsProb));
        CHECK(probOk(p.crossoverRate));
        CHECK(probOk(p.overallMutationRate));
        CHECK(probOk(p.interspeciesCrossoverRate));
        CHECK(probOk(p.multipointCrossoverRate));
        CHECK(probOk(p.survivalRate));
        CHECK(p.populationSize > 0);
        // MinCompatTreshold defaults to 0.0 ( CompatTreshold is clamped up to
        // it in Population), so only non-negativity is required here.
        CHECK(p.compatTreshold > 0.0 && p.minCompatTreshold >= 0.0);
        CHECK(p.minNeuronBias <= p.maxNeuronBias);
        CHECK(p.minWeight <= p.maxWeight);
        CHECK(p.tournamentSize > 0);
    }

    // Saved Parameters files always carry the framing markers.
    {
        Parameters p;
        p.reset();
        const std::filesystem::path tmp = std::filesystem::temp_directory_path() / "multineat_compliance_params.neat";
        p.save(tmp.string().c_str());
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
        p.reset();
        GenomeInitStruct init;
        init.numInputs = 3;
        init.numOutputs = 1;
        Genome g(p, init);
        const std::filesystem::path tmp = std::filesystem::temp_directory_path() / "multineat_compliance_genome.txt";
        g.save(tmp.string().c_str());
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
        net.clear();
        const std::filesystem::path tmp = std::filesystem::temp_directory_path() / "multineat_compliance_nn.txt";
        net.save(tmp.string().c_str());
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
        p.reset();
        GenomeInitStruct init;
        init.numInputs = 3;
        init.numOutputs = 1;
        Genome g(p, init);
        CHECK(!g.failsConstraints(p));
        CHECK(g.numLinks() > 0);
    }

    if (g_failures != 0) {
        std::cerr << "Test failed: TestCompliance with " << g_failures << " failure(s)\n";
        return 1;
    }
    std::cout << "Test passed: TestCompliance\n";
    return 0;
}
