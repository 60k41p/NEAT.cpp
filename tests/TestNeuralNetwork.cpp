// Golden tests for NEAT::NeuralNetwork activation + save/load.
//
// CTest-Labels: Unit;IO;Fast
// CTest-Timeout: 60
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <vector>

#include "NeuralNetwork.h"
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

    bool Near(Real a, Real b, Real eps = 1e-9) { return std::fabs(a - b) <= eps; }

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
        std::vector<Real> in{3.0, 1.0};
        net.Flush();
        net.Input(in);
        net.Activate();
        const std::vector<Real> out = net.Output();
        CHECK(out.size() == 1);
        CHECK(Near(out[0], 2.0 * 3.0 - 1.0 * 1.0 + 0.5));  // 5.5
    }

    // Flush zeroes activations; Output() windows the right neurons.
    {
        NeuralNetwork net = MakeLinearNet();
        std::vector<Real> in{1.0, 1.0};
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
        std::vector<Real> in{0.0};
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

        std::vector<Real> neg{-2.0};
        net.Flush();
        net.Input(neg);
        net.Activate();
        CHECK(Near(net.Output()[0], 0.0));

        std::vector<Real> pos{2.5};
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
        std::vector<Real> in{0.5, -0.25};
        net.Flush();
        net.Input(in);
        net.ActivateFast();
        const Real y = net.Output()[0];
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
        std::vector<Real> zero{0.0, 0.0};
        net2.Flush();
        net2.Input(zero);
        net2.ActivateUseInternalBias();
        CHECK(Near(net2.Output()[0], 1.0 / (1.0 + std::exp(-1.0)), 1e-9));
    }

    // ActivateLeaky with timeconst == dtime reduces to the plain sum.
    {
        NeuralNetwork net = MakeLinearNet();
        std::vector<Real> in{3.0, 1.0};
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

        std::vector<Real> in{3.0, 1.0};
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

    // Golden table for all 14 activation functions through Activate(),
    // using a hand-built 1-in / 1-out network.
    {
        struct Case {
            NEAT::ActivationFunction fn;
            double x, a, b, expected;
        };
        const Real x = 0.7;
        const std::vector<Case> cases = {
            {UNSIGNED_SIGMOID, x, 1, 0, 1.0 / (1.0 + exp(-x))},
            {SIGNED_SIGMOID, x, 1, 0, 2.0 * (1.0 / (1.0 + exp(-x)) - 0.5)},
            {TANH, x, 1, 0, tanh(x)},
            {TANH_CUBIC, x, 1, 0, tanh(x * x * x)},
            {SIGNED_STEP, x, 1, 0, 1.0},
            {UNSIGNED_STEP, x, 1, 0, 1.0},
            {SIGNED_GAUSS, x, 1, 0, 2.0 * (exp(-x * x) - 0.5)},
            {UNSIGNED_GAUSS, x, 1, 0, exp(-x * x)},
            {ABS, -x, 1, 0, x},
            {SIGNED_SINE, x, 1, 0, sin(x)},
            {UNSIGNED_SINE, x, 1, 0, (sin(x) + 1.0) / 2.0},
            {LINEAR, -x, 1, 0, -x},
            {RELU, -x, 1, 0, 0.0},
            {SOFTPLUS, x, 1, 0, log(1.0 + exp(x))},
            // slope/shift parameters are honored (TANH deliberately ignores b, as upstream)
            {UNSIGNED_SIGMOID, x, 2.0, 0.5, 1.0 / (1.0 + exp(-(2.0 * x + 0.5)))},
            {SIGNED_SINE, x, 2.0, 0.5, sin(2.0 * x + 0.5)},
            {UNSIGNED_GAUSS, x, 2.0, 0.5, exp(-2.0 * x * x + 0.5)},
            {LINEAR, x, 1, 0.5, x + 0.5},
            {SIGNED_STEP, 0.3, 1, 0.5, -1.0},
            {UNSIGNED_STEP, 0.3, 1, 0.5, 0.0},
        };
        for (const Case &c : cases) {
            NeuralNetwork net;
            Neuron in_n, out_n;
            in_n.m_activesum = in_n.m_activation = 0;
            in_n.m_a = 1;
            in_n.m_b = 0;
            in_n.m_timeconst = 1;
            in_n.m_bias = 0;
            in_n.m_membrane_potential = 0;
            in_n.m_activation_function_type = LINEAR;
            in_n.m_type = INPUT;
            out_n = in_n;
            out_n.m_type = OUTPUT;
            out_n.m_activation_function_type = c.fn;
            out_n.m_a = c.a;
            out_n.m_b = c.b;
            Connection conn;
            conn.m_source_neuron_idx = 0;
            conn.m_target_neuron_idx = 1;
            conn.m_weight = 1.0;
            conn.m_signal = 0;
            conn.m_recur_flag = false;
            conn.m_hebb_rate = 0;
            conn.m_hebb_pre_rate = 0;
            net.AddNeuron(in_n);
            net.AddNeuron(out_n);
            net.AddConnection(conn);
            net.SetInputOutputDimentions(1, 1);
            std::vector<Real> in{static_cast<Real>(c.x)};
            net.Input(in);
            net.Activate();
            CHECK(net.Output().size() == 1);
            CHECK(Near(net.Output()[0], c.expected, 1e-5));
        }
    }

    // ActivateFast applies each neuron's own activation function through the
    // unchecked hot path (no topology validation; use Activate() to validate).
    {
        NeuralNetwork net;
        Neuron in_n, out_n;
        in_n.m_activesum = in_n.m_activation = 0;
        in_n.m_a = 1;
        in_n.m_b = in_n.m_timeconst = in_n.m_bias = in_n.m_membrane_potential = 0;
        in_n.m_activation_function_type = LINEAR;
        in_n.m_type = INPUT;
        out_n = in_n;
        out_n.m_type = OUTPUT;
        out_n.m_activation_function_type = LINEAR;  // linear passes the sum through
        Connection conn;
        conn.m_source_neuron_idx = 0;
        conn.m_target_neuron_idx = 1;
        conn.m_weight = 0.0;
        conn.m_signal = 0;
        conn.m_recur_flag = false;
        conn.m_hebb_rate = conn.m_hebb_pre_rate = 0;
        net.AddNeuron(in_n);
        net.AddNeuron(out_n);
        net.AddConnection(conn);
        net.SetInputOutputDimentions(1, 1);
        std::vector<Real> in{5.0};
        net.Input(in);
        net.ActivateFast();
        CHECK(Near(net.Output()[0], 0.0));  // linear passes 5.0 * 0.0 through
        // An unsigned-sigmoid neuron still squashes through the fast path.
        out_n.m_activation_function_type = UNSIGNED_SIGMOID;
        NeuralNetwork net2;
        net2.AddNeuron(in_n);
        net2.AddNeuron(out_n);
        net2.AddConnection(conn);
        net2.SetInputOutputDimentions(1, 1);
        net2.Input(in);
        net2.ActivateFast();
        CHECK(Near(net2.Output()[0], 0.5));  // sigmoid(0)
    }

    // ActivateUseInternalBias adds m_bias to the activation sum.
    {
        NeuralNetwork net;
        Neuron in_n, out_n;
        in_n.m_activesum = in_n.m_activation = 0;
        in_n.m_a = 1;
        in_n.m_b = in_n.m_timeconst = in_n.m_bias = in_n.m_membrane_potential = 0;
        in_n.m_activation_function_type = LINEAR;
        in_n.m_type = INPUT;
        out_n = in_n;
        out_n.m_type = OUTPUT;
        out_n.m_activation_function_type = LINEAR;
        out_n.m_bias = 0.5;
        Connection conn;
        conn.m_source_neuron_idx = 0;
        conn.m_target_neuron_idx = 1;
        conn.m_weight = 2.0;
        conn.m_signal = 0;
        conn.m_recur_flag = false;
        conn.m_hebb_rate = conn.m_hebb_pre_rate = 0;
        net.AddNeuron(in_n);
        net.AddNeuron(out_n);
        net.AddConnection(conn);
        net.SetInputOutputDimentions(1, 1);
        std::vector<Real> in{3.0};
        net.Input(in);
        net.Activate();
        CHECK(Near(net.Output()[0], 6.0));  // bias ignored by Activate
        net.Flush();
        net.Input(in);
        net.ActivateUseInternalBias();
        CHECK(Near(net.Output()[0], 6.5));  // 2*3 + 0.5
    }

    // ActivateLeaky integrates the membrane potential over steps.
    {
        NeuralNetwork net;
        Neuron in_n, out_n;
        in_n.m_activesum = in_n.m_activation = 0;
        in_n.m_a = 1;
        in_n.m_b = in_n.m_timeconst = in_n.m_bias = in_n.m_membrane_potential = 0;
        in_n.m_activation_function_type = LINEAR;
        in_n.m_type = INPUT;
        out_n = in_n;
        out_n.m_type = OUTPUT;
        out_n.m_activation_function_type = LINEAR;
        out_n.m_timeconst = 1.0;
        Connection conn;
        conn.m_source_neuron_idx = 0;
        conn.m_target_neuron_idx = 1;
        conn.m_weight = 1.0;
        conn.m_signal = 0;
        conn.m_recur_flag = false;
        conn.m_hebb_rate = conn.m_hebb_pre_rate = 0;
        net.AddNeuron(in_n);
        net.AddNeuron(out_n);
        net.AddConnection(conn);
        net.SetInputOutputDimentions(1, 1);
        std::vector<Real> in{4.0};
        net.Input(in);
        net.ActivateLeaky(0.5);  // mp = 0.5 * 4 = 2
        CHECK(Near(net.Output()[0], 2.0));
        net.ActivateLeaky(0.5);  // mp = 0.5 * 2 + 0.5 * 4 = 3
        CHECK(Near(net.Output()[0], 3.0));
    }

    // Flush zeroes activations, active sums and membrane potentials.
    {
        NeuralNetwork net(false);  // XOR topology, random weights
        net.Flush();
        std::vector<Real> in{1.0, 1.0, 1.0};
        net.Input(in);
        net.Activate();
        CHECK(!Near(net.Output()[0], 0.0));
        net.Flush();
        for (const auto &n : net.m_neurons) {
            CHECK(n.m_activation == 0.0);
            CHECK(n.m_activesum == 0.0);
            CHECK(n.m_membrane_potential == 0.0);
        }
    }

    // RTRL pipeline: gradients -> error -> weights, all finite with a real update.
    {
        NeuralNetwork net(false);
        for (auto &n : net.m_neurons) {
            n.m_activation_function_type = UNSIGNED_SIGMOID;  // RTRL knows sigmoids only
        }
        net.InitRTRLMatrix();
        net.Flush();
        CHECK(net.m_neurons[0].m_sensitivity_matrix.size() == net.m_neurons.size());
        std::vector<Real> in{1.0, 1.0, 1.0};
        net.Input(in);
        net.Activate();
        const Real w_before = net.m_connections[0].m_weight;
        net.RTRL_update_gradients();
        net.RTRL_update_error(1.0);
        net.RTRL_update_weights();
        CHECK(std::isfinite(net.m_connections[0].m_weight));
        CHECK(!Near(net.m_connections[0].m_weight, w_before, 1e-12));
        net.FlushCube();
    }

    // Hebbian Adapt moves weights and clamps them to [-MaxWeight, MaxWeight].
    {
        Parameters p;
        p.Reset();
        NeuralNetwork net;
        Neuron in_n, out_n;
        in_n.m_activesum = in_n.m_activation = 0;
        in_n.m_a = 1;
        in_n.m_b = in_n.m_timeconst = in_n.m_bias = in_n.m_membrane_potential = 0;
        in_n.m_activation_function_type = LINEAR;
        in_n.m_type = INPUT;
        out_n = in_n;
        out_n.m_type = OUTPUT;
        out_n.m_activation_function_type = LINEAR;
        Connection conn;
        conn.m_source_neuron_idx = 0;
        conn.m_target_neuron_idx = 1;
        conn.m_weight = 0.5;
        conn.m_signal = 0;
        conn.m_recur_flag = false;
        conn.m_hebb_rate = 0.1;
        conn.m_hebb_pre_rate = 0.1;
        net.AddNeuron(in_n);
        net.AddNeuron(out_n);
        net.AddConnection(conn);
        net.SetInputOutputDimentions(1, 1);
        std::vector<Real> in{1.0};
        net.Input(in);
        net.Activate();  // out = 0.5
        net.Adapt(p);
        // delta = 0.1*(0.5-0.5)*1*0.5 + 0.1*0.5*1*(0.5-1) = -0.025
        CHECK(Near(net.m_connections[0].m_weight, 0.475, 1e-9));

        // Oversized weight gets clamped to MaxWeight.
        net.m_connections[0].m_weight = 999.0;
        net.Flush();
        net.Input(in);
        net.Activate();
        net.Adapt(p);
        CHECK(net.m_connections[0].m_weight <= p.MaxWeight);
    }

    // InputExact requires exactly NumInputs values.
    {
        NeuralNetwork net;
        Neuron in_n, out_n;
        in_n.m_type = INPUT;
        out_n.m_type = OUTPUT;
        out_n.m_activation_function_type = LINEAR;
        net.AddNeuron(in_n);
        net.AddNeuron(out_n);
        Connection conn;
        conn.m_source_neuron_idx = 0;
        conn.m_target_neuron_idx = 1;
        conn.m_weight = 1.0;
        net.AddConnection(conn);
        net.SetInputOutputDimentions(1, 1);
        net.InputExact({2.0});
        net.Activate();
        CHECK(Near(net.Output()[0], 2.0));
        bool threw = false;
        try {
            net.InputExact({1.0, 2.0});
        } catch (const std::invalid_argument &) {
            threw = true;
        }
        CHECK(threw);
    }

    // Connection geometry: Euclidean length, true total, delay conversion.
    {
        NeuralNetwork net;
        Neuron a, b;
        a.m_x = 0.0;
        a.m_y = 0.0;
        a.m_z = 0.0;
        b.m_x = 3.0;
        b.m_y = 4.0;
        b.m_z = 0.0;
        CHECK(Near(net.GetConnectionLenght(a, b), 5.0));
        CHECK(Near(net.GetConnectionLength(a, b), 5.0));
        a.m_type = INPUT;
        b.m_type = OUTPUT;
        net.AddNeuron(a);
        net.AddNeuron(b);
        Connection conn;
        conn.m_source_neuron_idx = 0;
        conn.m_target_neuron_idx = 1;
        conn.m_weight = 1.0;
        net.AddConnection(conn);
        net.SetInputOutputDimentions(1, 1);
        CHECK(Near(net.GetTotalConnectionLength(), 5.0));
        net.UpdateConnectionGeometry(true, 5.0);
        CHECK(Near(net.m_connections[0].m_length, 5.0));
        CHECK(Near(net.m_connections[0].m_synaptic_delay, 1.0));
    }

    // ActivateSteps repeats activation; ActivateBatch evaluates samples independently.
    {
        NeuralNetwork net;
        Neuron in_n, out_n;
        in_n.m_type = INPUT;
        out_n.m_type = OUTPUT;
        out_n.m_activation_function_type = LINEAR;
        net.AddNeuron(in_n);
        net.AddNeuron(out_n);
        Connection conn;
        conn.m_source_neuron_idx = 0;
        conn.m_target_neuron_idx = 1;
        conn.m_weight = 2.0;
        net.AddConnection(conn);
        net.SetInputOutputDimentions(1, 1);
        net.Flush();
        std::vector<Real> in{1.5};
        net.Input(in);
        net.ActivateSteps(3);
        CHECK(Near(net.Output()[0], 3.0));
        const auto batch = net.ActivateBatch({{1.0}, {2.0}});
        CHECK(batch.size() == 2);
        CHECK(Near(batch[0][0], 2.0));
        CHECK(Near(batch[1][0], 4.0));
    }

    // Sparse RTRL initializes indexed sensitivities; Flush clears spiking state.
    {
        NeuralNetwork net;
        Neuron in_n, out_n;
        in_n.m_type = INPUT;
        out_n.m_type = OUTPUT;
        out_n.m_activation_function_type = UNSIGNED_SIGMOID;
        net.AddNeuron(in_n);
        net.AddNeuron(out_n);
        Connection conn;
        conn.m_source_neuron_idx = 0;
        conn.m_target_neuron_idx = 1;
        conn.m_weight = 0.5;
        net.AddConnection(conn);
        net.SetInputOutputDimentions(1, 1);
        net.InitSparseRTRLMatrix();
        CHECK(net.SparseRTRLStateSize() == 2);  // neurons x connections
        net.RTRL_update_gradients_sparse();
        net.RTRL_update_error_sparse(0.5);
        net.RTRL_update_weights();
        net.Flush();
        CHECK(Near(net.m_connections[0].m_signal, 0.0));
        CHECK(Near(net.m_connections[0].m_source_activation, 0.0));
    }

    // Serialize/Deserialize round-trips topology, weights and spiking state.
    {
        NeuralNetwork net;
        Neuron in_n, out_n;
        in_n.m_type = INPUT;
        out_n.m_type = OUTPUT;
        out_n.m_activation_function_type = TANH;
        out_n.m_spike_threshold = 1.5;
        net.AddNeuron(in_n);
        net.AddNeuron(out_n);
        Connection conn;
        conn.m_source_neuron_idx = 0;
        conn.m_target_neuron_idx = 1;
        conn.m_weight = -0.75;
        conn.m_synaptic_delay = 0.004;
        net.AddConnection(conn);
        net.SetInputOutputDimentions(1, 1);
        const std::string data = net.Serialize();
        const NeuralNetwork loaded = NeuralNetwork::Deserialize(data);
        CHECK(loaded.NumInputs() == 1 && loaded.NumOutputs() == 1);
        CHECK(loaded.m_neurons.size() == 2 && loaded.m_connections.size() == 1);
        CHECK(Near(loaded.m_connections[0].m_weight, -0.75));
        CHECK(Near(loaded.m_connections[0].m_synaptic_delay, 0.004));
        CHECK(Near(loaded.m_neurons[1].m_spike_threshold, 1.5));
        CHECK(loaded.m_neurons[1].m_activation_function_type == TANH);
    }

    if (g_failures != 0) {
        std::cerr << "Test failed: TestNeuralNetwork with " << g_failures << " failure(s)\n";
        return 1;
    }
    std::cout << "Test passed: TestNeuralNetwork\n";
    return 0;
}
