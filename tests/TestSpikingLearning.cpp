// Tests for NEAT::EPropLearner and spiking-network simulation
// (src/SpikingLearning.h/.cpp, src/NeuralNetwork spiking paths).
//
// CTest-Labels: Unit;Fast
// CTest-Timeout: 120
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "Genome.h"
#include "NeuralNetwork.h"
#include "Parameters.h"
#include "SpikingLearning.h"

namespace {

    int g_failures = 0;

#define CHECK(cond)                                                                         \
    do {                                                                                    \
        if (!(cond)) {                                                                      \
            std::cerr << "FAILED " << __FILE__ << ":" << __LINE__ << ": " << #cond << "\n"; \
            ++g_failures;                                                                   \
        }                                                                                   \
    } while (0)

    bool Near(Real a, Real b, Real eps = 1e-9) { return std::fabs(a - b) <= eps; }

    // A minimal spiking network: 1 input + bias + 1 LIF output, fully connected.
    NEAT::NeuralNetwork MakeSpikingNet() {
        using namespace NEAT;
        Parameters p;
        p.Reset();
        p.ConfigureSpiking(false);
        GenomeInitStruct init;
        init.NumInputs = 2;  // 1 problem input + bias
        init.NumOutputs = 1;
        init.SeedType = PERCEPTRON;
        Genome g(p, init);
        for (unsigned i = 0; i < g.NumNeurons(); ++i) {
            if (g.m_NeuronGenes[i].Type() == HIDDEN || g.m_NeuronGenes[i].Type() == OUTPUT) {
                g.m_NeuronGenes[i].m_ActFunction = SPIKING_LIF;
            }
        }
        RNG rng;
        rng.Seed(1);
        g.Randomize_SpikingParameters(p, rng);
        NeuralNetwork net;
        g.BuildPhenotype(net);
        return net;
    }

}  // namespace

int TestSpikingLearning(int argc, char *argv[]) {
    (void)argc;
    (void)argv;
    using namespace NEAT;

    // IsSpiking / activation-type helpers.
    {
        CHECK(IsSpikingActivation(SPIKING_LIF));
        CHECK(IsSpikingActivation(SPIKING_ADAPTIVE_LIF));
        CHECK(IsSpikingActivation(SPIKING_IZHIKEVICH));
        CHECK(IsSpikingActivation(MCCULLOCH_PITTS));
        CHECK(IsSpikingActivation(SPIKING_MCCULLOCH_PITTS));
        CHECK(!IsSpikingActivation(TANH));
        CHECK(!IsSpikingActivation(UNSIGNED_SIGMOID));
        NeuralNetwork net = MakeSpikingNet();
        CHECK(net.IsSpiking());
        NeuralNetwork rate;
        CHECK(!rate.IsSpiking());
    }

    // StepSpiking validates inputs and advances the clock/outputs.
    {
        NeuralNetwork net = MakeSpikingNet();
        net.SetSpikingInputMode(CURRENT_INPUT);
        net.SetSpikingOutputMode(SPIKE_OUTPUT);
        CHECK(Near(net.SpikingTime(), 0.0));
        const std::vector<Real> out = net.StepSpiking({0.5, 0.5});
        CHECK(out.size() == 1);
        CHECK(net.SpikingTime() > 0.0);
        bool threw = false;
        try {
            (void)net.StepSpiking({0.5});
        } catch (const std::invalid_argument &) {
            threw = true;
        }
        CHECK(threw);
        threw = false;
        try {
            (void)net.StepSpiking({0.5, 0.5}, 0.0);
        } catch (const std::invalid_argument &) {
            threw = true;
        }
        CHECK(threw);
    }

    // Output modes decode consistently; SimulateSpiking batches steps.
    {
        NeuralNetwork net = MakeSpikingNet();
        net.SetSpikingInputMode(CURRENT_INPUT);
        const auto seq = net.SimulateSpiking({{1.0, 1.0}, {0.0, 0.0}, {1.0, 1.0}});
        CHECK(seq.size() == 3);
        net.SetSpikingOutputMode(FIRING_RATE_OUTPUT);
        CHECK(net.OutputDecoded().size() == 1);
        net.SetSpikingOutputMode(FILTERED_SPIKE_OUTPUT);
        CHECK(net.OutputDecoded().size() == 1);
        net.SetSpikingOutputMode(MEMBRANE_POTENTIAL_OUTPUT);
        CHECK(net.OutputDecoded().size() == 1);
        CHECK(net.OutputSpikes().size() == 1);
        CHECK(net.OutputRates().size() == 1);
        CHECK(net.OutputFilteredSpikes().size() == 1);
        CHECK(net.OutputMembranePotentials().size() == 1);
        // Spike history records input spikes (binary-spike mode emits events).
        net.SetSpikingInputMode(BINARY_SPIKE_INPUT);
        (void)net.StepSpiking({1.0, 0.0});
        CHECK(!net.GetSpikeHistory().empty());
        net.ClearSpikeHistory();
        CHECK(net.GetSpikeHistory().empty());
        net.SetSpikingInputMode(CURRENT_INPUT);
    }

    // Time step / modes / seed validation.
    {
        NeuralNetwork net = MakeSpikingNet();
        net.SetSpikingTimeStep(0.002);
        CHECK(Near(net.SpikingTimeStep(), 0.002));
        bool threw = false;
        try {
            net.SetSpikingTimeStep(-1.0);
        } catch (const std::invalid_argument &) {
            threw = true;
        }
        CHECK(threw);
        net.SeedSpiking(12345);
        net.EnableSpikeRecording(false);
        CHECK(net.GetSpikeHistory().empty());
        net.EnableSpikeRecording(true);
        // STDP can be toggled network-wide.
        net.EnableSTDP(true);
        net.EnableSTDP(false);
    }

    // McCulloch-Pitts rate evaluation with inhibitory veto.
    {
        NeuralNetwork net;
        Neuron in_n, out_n;
        in_n.m_activation_function_type = LINEAR;
        in_n.m_type = INPUT;
        out_n.m_activation_function_type = MCCULLOCH_PITTS;
        out_n.m_type = OUTPUT;
        out_n.m_spike_threshold = 0.5;
        out_n.m_mcp_inhibitory_veto = true;
        net.AddNeuron(in_n);
        net.AddNeuron(out_n);
        Connection conn;
        conn.m_source_neuron_idx = 0;
        conn.m_target_neuron_idx = 1;
        conn.m_weight = 1.0;
        net.AddConnection(conn);
        net.SetInputOutputDimentions(1, 1);
        std::vector<Real> in{1.0};
        net.Input(in);
        net.Activate();
        CHECK(Near(net.Output()[0], 1.0));
    }

    // EProp rejects rate networks, trains spiking ones.
    {
        NeuralNetwork rate;
        EPropLearner learner{EPropConfig()};
        bool threw = false;
        try {
            learner.Initialize(rate);
        } catch (const std::invalid_argument &) {
            threw = true;
        }
        CHECK(threw);

        NeuralNetwork net = MakeSpikingNet();
        CHECK(!learner.IsInitialized());
        learner.Initialize(net);
        CHECK(learner.IsInitialized());
        const EPropStepResult r = learner.TrainStep(net, {0.5, 0.5}, {1.0});
        CHECK(r.outputs.size() == 1);
        CHECK(std::isfinite(r.loss));
        CHECK(r.gradient_norm >= 0.0);
        const EPropSequenceResult seq = learner.TrainSequence(net, {{1.0, 0.0}, {0.0, 1.0}}, {{1.0}, {0.0}});
        CHECK(seq.outputs.size() == 2);
        CHECK(seq.losses.size() == 2);
        CHECK(std::isfinite(seq.mean_loss));
        CHECK(seq.optimizer_updates > 0);
        // Optimizer state round-trips through Serialize.
        const std::string state = learner.Serialize();
        const EPropLearner restored = EPropLearner::Deserialize(state);
        CHECK(restored.IsInitialized());
        CHECK(restored.OptimizerStep() == learner.OptimizerStep());
    }

    if (g_failures != 0) {
        std::cerr << "Test failed: TestSpikingLearning with " << g_failures << " failure(s)\n";
        return 1;
    }
    std::cout << "Test passed: TestSpikingLearning\n";
    return 0;
}
