// Golden tests for NEAT::NeuralNetwork activation + save/load.
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <vector>

#include "NeuralNetwork.h"

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

    // Builds a 2-input, 1-output LINEAR network:
    //   out = 2.0*in0 - 1.0*in1 + 0.5
    NEAT::NeuralNetwork MakeLinearNet() {
        using namespace NEAT;
        NeuralNetwork net;
        net.Clear();
        for (int i = 0; i < 3; ++i) {
            Neuron n;
            n.m_a = 1.0;
            n.m_b = 0.0;
            n.m_timeconst = 1.0;
            n.m_bias = 0.0;
            n.m_membrane_potential = 0.0;
            n.m_activation = 0.0;
            n.m_activesum = 0.0;
            n.m_activation_function_type = LINEAR;
            n.m_type = (i < 2) ? INPUT : OUTPUT;
            net.AddNeuron(n);
        }
        // Output neuron shift: af_linear(x, b) = x + b.
        net.m_neurons[2].m_b = 0.5;
        net.SetInputOutputDimentions(2, 1);
        Connection c0;
        c0.m_source_neuron_idx = 0;
        c0.m_target_neuron_idx = 2;
        c0.m_weight = 2.0;
        c0.m_recur_flag = false;
        Connection c1 = c0;
        c1.m_source_neuron_idx = 1;
        c1.m_weight = -1.0;
        net.AddConnection(c0);
        net.AddConnection(c1);
        return net;
    }

}  // namespace

int TestNeuralNetwork(int argc, char *argv[]) {
    (void)argc;
    (void)argv;
    using namespace NEAT;

    // LINEAR golden: Flush + Input + Activate + Output.
    {
        NeuralNetwork net = MakeLinearNet();
        std::vector<double> in{3.0, 1.0};
        net.Flush();
        net.Input(in);
        net.Activate();
        const std::vector<double> out = net.Output();
        CHECK(out.size() == 1);
        CHECK(Near(out[0], 2.0 * 3.0 - 1.0 * 1.0 + 0.5));  // 5.5
    }

    // Flush zeroes activations; Output() windows the right neurons.
    {
        NeuralNetwork net = MakeLinearNet();
        std::vector<double> in{1.0, 1.0};
        net.Input(in);
        net.Activate();
        CHECK(!Near(net.Output()[0], 0.0));
        net.Flush();
        CHECK(Near(net.m_neurons[2].m_activation, 0.0));
        CHECK(net.NumInputs() == 2 && net.NumOutputs() == 1);
    }

    // UNSIGNED_SIGMOID(0) == 0.5 golden.
    {
        NeuralNetwork net;
        net.Clear();
        Neuron ni, no;
        ni.m_type = INPUT;
        no.m_type = OUTPUT;
        no.m_a = 1.0;
        no.m_b = 0.0;
        no.m_activation_function_type = UNSIGNED_SIGMOID;
        net.AddNeuron(ni);
        net.AddNeuron(no);
        net.SetInputOutputDimentions(1, 1);
        Connection c;
        c.m_source_neuron_idx = 0;
        c.m_target_neuron_idx = 1;
        c.m_weight = 0.0;
        c.m_recur_flag = false;
        net.AddConnection(c);
        std::vector<double> in{0.0};
        net.Flush();
        net.Input(in);
        net.Activate();
        CHECK(Near(net.Output()[0], 0.5, 1e-9));
    }

    // RELU clamps negatives, passes positives.
    {
        NeuralNetwork net;
        net.Clear();
        Neuron ni, no;
        ni.m_type = INPUT;
        no.m_type = OUTPUT;
        no.m_activation_function_type = RELU;
        net.AddNeuron(ni);
        net.AddNeuron(no);
        net.SetInputOutputDimentions(1, 1);
        Connection c;
        c.m_source_neuron_idx = 0;
        c.m_target_neuron_idx = 1;
        c.m_weight = 1.0;
        c.m_recur_flag = false;
        net.AddConnection(c);

        std::vector<double> neg{-2.0};
        net.Flush();
        net.Input(neg);
        net.Activate();
        CHECK(Near(net.Output()[0], 0.0));

        std::vector<double> pos{2.5};
        net.Flush();
        net.Input(pos);
        net.Activate();
        CHECK(Near(net.Output()[0], 2.5));
    }

    // ActivateFast runs (unsigned-sigmoid fast path) and stays in (0,1).
    // ActivateUseInternalBias adds m_bias before the sigmoid.
    {
        NeuralNetwork net = MakeLinearNet();
        for (auto &n : net.m_neurons) {
            n.m_activation_function_type = UNSIGNED_SIGMOID;
            n.m_a = 1.0;
            n.m_b = 0.0;
        }
        std::vector<double> in{0.5, -0.25};
        net.Flush();
        net.Input(in);
        net.ActivateFast();
        const double y = net.Output()[0];
        CHECK(y > 0.0 && y < 1.0);

        NeuralNetwork net2 = MakeLinearNet();
        for (auto &n : net2.m_neurons) {
            n.m_activation_function_type = UNSIGNED_SIGMOID;
            n.m_a = 1.0;
            n.m_b = 0.0;
            n.m_bias = 0.0;
        }
        // Zero all weights so only the internal bias drives the output.
        for (auto &c : net2.m_connections) {
            c.m_weight = 0.0;
        }
        net2.m_neurons[2].m_bias = 1.0;
        std::vector<double> zero{0.0, 0.0};
        net2.Flush();
        net2.Input(zero);
        net2.ActivateUseInternalBias();
        CHECK(Near(net2.Output()[0], 1.0 / (1.0 + std::exp(-1.0)), 1e-9));
    }

    // ActivateLeaky with timeconst == dtime reduces to the plain sum.
    {
        NeuralNetwork net = MakeLinearNet();
        std::vector<double> in{3.0, 1.0};
        net.Flush();
        net.Input(in);
        net.ActivateLeaky(1.0);
        CHECK(Near(net.Output()[0], 5.5));
    }

    // Save/Load round-trip preserves topology, weights, and behavior.
    {
        NeuralNetwork net = MakeLinearNet();
        const auto tmp = std::filesystem::temp_directory_path() / "multineat_test_nn.txt";
        net.Save(tmp.string().c_str());

        NeuralNetwork loaded;
        {
            std::ifstream in(tmp.string());
            CHECK(in.is_open());
            CHECK(loaded.Load(in));
        }
        CHECK(loaded.NumInputs() == 2 && loaded.NumOutputs() == 1);
        CHECK(loaded.m_neurons.size() == net.m_neurons.size());
        CHECK(loaded.m_connections.size() == net.m_connections.size());
        CHECK(Near(loaded.m_connections[0].m_weight, 2.0));
        CHECK(Near(loaded.m_connections[1].m_weight, -1.0));

        std::vector<double> in{3.0, 1.0};
        loaded.Flush();
        loaded.Input(in);
        loaded.Activate();
        CHECK(Near(loaded.Output()[0], 5.5));

        std::error_code ec;
        std::filesystem::remove(tmp, ec);
    }

    // Loading garbage returns false instead of crashing.
    {
        const auto tmp = std::filesystem::temp_directory_path() / "multineat_test_nn_garbage.txt";
        {
            std::ofstream out(tmp.string());
            out << "this is not a neural network file\n";
        }
        NeuralNetwork net;
        std::ifstream in(tmp.string());
        CHECK(!net.Load(in));
        std::error_code ec;
        std::filesystem::remove(tmp, ec);
    }

    if (g_failures != 0) {
        std::cerr << "Test failed: TestNeuralNetwork with " << g_failures << " failure(s)\n";
        return 1;
    }
    std::cout << "Test passed: TestNeuralNetwork\n";
    return 0;
}
