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

    bool near(double a, double b, double eps = 1e-9) { return std::fabs(a - b) <= eps; }

    // Builds a 2-input, 1-output LINEAR network:
    //   out = 2.0*in0 - 1.0*in1 + 0.5
    NEAT::NeuralNetwork makeLinearNet() {
        using namespace NEAT;
        NeuralNetwork net;
        net.clear();
        for (int i = 0; i < 3; ++i) {
            Neuron n;
            n.a_ = 1.0;
            n.b_ = 0.0;
            n.timeconst_ = 1.0;
            n.bias_ = 0.0;
            n.membranePotential_ = 0.0;
            n.activation_ = 0.0;
            n.activesum_ = 0.0;
            n.activationFunctionType_ = LINEAR;
            n.type_ = (i < 2) ? INPUT : OUTPUT;
            net.addNeuron(n);
        }
        // Output neuron shift: af_linear(x, b) = x + b.
        net.neurons_[2].b_ = 0.5;
        net.setInputOutputDimensions(2, 1);
        Connection c0;
        c0.sourceNeuronIndex_ = 0;
        c0.targetNeuronIndex_ = 2;
        c0.weight_ = 2.0;
        c0.recurFlag_ = false;
        Connection c1 = c0;
        c1.sourceNeuronIndex_ = 1;
        c1.weight_ = -1.0;
        net.addConnection(c0);
        net.addConnection(c1);
        return net;
    }

}  // namespace

int TestNeuralNetwork(int argc, char *argv[]) {
    (void)argc;
    (void)argv;
    using namespace NEAT;

    // LINEAR golden: Flush + Input + Activate + Output.
    {
        NeuralNetwork net = makeLinearNet();
        std::vector<double> in{3.0, 1.0};
        net.flush();
        net.input(in);
        net.activate();
        const std::vector<double> out = net.output();
        CHECK(out.size() == 1);
        CHECK(near(out[0], 2.0 * 3.0 - 1.0 * 1.0 + 0.5));  // 5.5
    }

    // Flush zeroes activations; Output() windows the right neurons.
    {
        NeuralNetwork net = makeLinearNet();
        std::vector<double> in{1.0, 1.0};
        net.input(in);
        net.activate();
        CHECK(!near(net.output()[0], 0.0));
        net.flush();
        CHECK(near(net.neurons_[2].activation_, 0.0));
        CHECK(net.numInputs() == 2 && net.numOutputs() == 1);
    }

    // UNSIGNED_SIGMOID(0) == 0.5 golden.
    {
        NeuralNetwork net;
        net.clear();
        Neuron ni, no;
        ni.type_ = INPUT;
        no.type_ = OUTPUT;
        no.a_ = 1.0;
        no.b_ = 0.0;
        no.activationFunctionType_ = UNSIGNED_SIGMOID;
        net.addNeuron(ni);
        net.addNeuron(no);
        net.setInputOutputDimensions(1, 1);
        Connection c;
        c.sourceNeuronIndex_ = 0;
        c.targetNeuronIndex_ = 1;
        c.weight_ = 0.0;
        c.recurFlag_ = false;
        net.addConnection(c);
        std::vector<double> in{0.0};
        net.flush();
        net.input(in);
        net.activate();
        CHECK(near(net.output()[0], 0.5, 1e-9));
    }

    // RELU clamps negatives, passes positives.
    {
        NeuralNetwork net;
        net.clear();
        Neuron ni, no;
        ni.type_ = INPUT;
        no.type_ = OUTPUT;
        no.activationFunctionType_ = RELU;
        net.addNeuron(ni);
        net.addNeuron(no);
        net.setInputOutputDimensions(1, 1);
        Connection c;
        c.sourceNeuronIndex_ = 0;
        c.targetNeuronIndex_ = 1;
        c.weight_ = 1.0;
        c.recurFlag_ = false;
        net.addConnection(c);

        std::vector<double> neg{-2.0};
        net.flush();
        net.input(neg);
        net.activate();
        CHECK(near(net.output()[0], 0.0));

        std::vector<double> pos{2.5};
        net.flush();
        net.input(pos);
        net.activate();
        CHECK(near(net.output()[0], 2.5));
    }

    // ActivateFast runs (unsigned-sigmoid fast path) and stays in (0,1).
    // ActivateUseInternalBias adds m_bias before the sigmoid.
    {
        NeuralNetwork net = makeLinearNet();
        for (Neuron &n : net.neurons_) {
            n.activationFunctionType_ = UNSIGNED_SIGMOID;
            n.a_ = 1.0;
            n.b_ = 0.0;
        }
        std::vector<double> in{0.5, -0.25};
        net.flush();
        net.input(in);
        net.activateFast();
        const double y = net.output()[0];
        CHECK(y > 0.0 && y < 1.0);

        NeuralNetwork net2 = makeLinearNet();
        for (Neuron &n : net2.neurons_) {
            n.activationFunctionType_ = UNSIGNED_SIGMOID;
            n.a_ = 1.0;
            n.b_ = 0.0;
            n.bias_ = 0.0;
        }
        // Zero all weights so only the internal bias drives the output.
        for (Connection &c : net2.connections_) {
            c.weight_ = 0.0;
        }
        net2.neurons_[2].bias_ = 1.0;
        std::vector<double> zero{0.0, 0.0};
        net2.flush();
        net2.input(zero);
        net2.activateUseInternalBias();
        CHECK(near(net2.output()[0], 1.0 / (1.0 + std::exp(-1.0)), 1e-9));
    }

    // ActivateLeaky with timeconst == dtime reduces to the plain sum.
    {
        NeuralNetwork net = makeLinearNet();
        std::vector<double> in{3.0, 1.0};
        net.flush();
        net.input(in);
        net.activateLeaky(1.0);
        CHECK(near(net.output()[0], 5.5));
    }

    // Save/Load round-trip preserves topology, weights, and behavior.
    {
        NeuralNetwork net = makeLinearNet();
        const std::filesystem::path tmp = std::filesystem::temp_directory_path() / "multineat_test_nn.txt";
        net.save(tmp.string().c_str());

        NeuralNetwork loaded;
        {
            std::ifstream in(tmp.string());
            CHECK(in.is_open());
            CHECK(loaded.load(in));
        }
        CHECK(loaded.numInputs() == 2 && loaded.numOutputs() == 1);
        CHECK(loaded.neurons_.size() == net.neurons_.size());
        CHECK(loaded.connections_.size() == net.connections_.size());
        CHECK(near(loaded.connections_[0].weight_, 2.0));
        CHECK(near(loaded.connections_[1].weight_, -1.0));

        std::vector<double> in{3.0, 1.0};
        loaded.flush();
        loaded.input(in);
        loaded.activate();
        CHECK(near(loaded.output()[0], 5.5));

        std::error_code ec;
        std::filesystem::remove(tmp, ec);
    }

    // Loading garbage returns false instead of crashing.
    {
        const std::filesystem::path tmp = std::filesystem::temp_directory_path() / "multineat_test_nn_garbage.txt";
        {
            std::ofstream out(tmp.string());
            out << "this is not a neural network file\n";
        }
        NeuralNetwork net;
        std::ifstream in(tmp.string());
        CHECK(!net.load(in));
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
        const double x = 0.7;
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
            Neuron inN, outN;
            inN.activesum_ = inN.activation_ = 0;
            inN.a_ = 1;
            inN.b_ = 0;
            inN.timeconst_ = 1;
            inN.bias_ = 0;
            inN.membranePotential_ = 0;
            inN.activationFunctionType_ = LINEAR;
            inN.type_ = INPUT;
            outN = inN;
            outN.type_ = OUTPUT;
            outN.activationFunctionType_ = c.fn;
            outN.a_ = c.a;
            outN.b_ = c.b;
            Connection conn;
            conn.sourceNeuronIndex_ = 0;
            conn.targetNeuronIndex_ = 1;
            conn.weight_ = 1.0;
            conn.signal_ = 0;
            conn.recurFlag_ = false;
            conn.hebbRate_ = 0;
            conn.hebbPreRate_ = 0;
            net.addNeuron(inN);
            net.addNeuron(outN);
            net.addConnection(conn);
            net.setInputOutputDimensions(1, 1);
            std::vector<double> in{c.x};
            net.input(in);
            net.activate();
            CHECK(net.output().size() == 1);
            CHECK(near(net.output()[0], c.expected, 1e-9));
        }
    }

    // ActivateFast assumes unsigned sigmoid regardless of the neuron's type.
    {
        NeuralNetwork net;
        Neuron inN, outN;
        inN.activesum_ = inN.activation_ = 0;
        inN.a_ = 1;
        inN.b_ = inN.timeconst_ = inN.bias_ = inN.membranePotential_ = 0;
        inN.activationFunctionType_ = LINEAR;
        inN.type_ = INPUT;
        outN = inN;
        outN.type_ = OUTPUT;
        outN.activationFunctionType_ = LINEAR;  // ActivateFast must still apply unsigned sigmoid
        Connection conn;
        conn.sourceNeuronIndex_ = 0;
        conn.targetNeuronIndex_ = 1;
        conn.weight_ = 0.0;
        conn.signal_ = 0;
        conn.recurFlag_ = false;
        conn.hebbRate_ = conn.hebbPreRate_ = 0;
        net.addNeuron(inN);
        net.addNeuron(outN);
        net.addConnection(conn);
        net.setInputOutputDimensions(1, 1);
        std::vector<double> in{5.0};
        net.input(in);
        net.activateFast();
        CHECK(near(net.output()[0], 0.5));  // sigmoid(0) — linear would give 0.0
    }

    // ActivateUseInternalBias adds m_bias to the activation sum.
    {
        NeuralNetwork net;
        Neuron inN, outN;
        inN.activesum_ = inN.activation_ = 0;
        inN.a_ = 1;
        inN.b_ = inN.timeconst_ = inN.bias_ = inN.membranePotential_ = 0;
        inN.activationFunctionType_ = LINEAR;
        inN.type_ = INPUT;
        outN = inN;
        outN.type_ = OUTPUT;
        outN.activationFunctionType_ = LINEAR;
        outN.bias_ = 0.5;
        Connection conn;
        conn.sourceNeuronIndex_ = 0;
        conn.targetNeuronIndex_ = 1;
        conn.weight_ = 2.0;
        conn.signal_ = 0;
        conn.recurFlag_ = false;
        conn.hebbRate_ = conn.hebbPreRate_ = 0;
        net.addNeuron(inN);
        net.addNeuron(outN);
        net.addConnection(conn);
        net.setInputOutputDimensions(1, 1);
        std::vector<double> in{3.0};
        net.input(in);
        net.activate();
        CHECK(near(net.output()[0], 6.0));  // bias ignored by Activate
        net.flush();
        net.input(in);
        net.activateUseInternalBias();
        CHECK(near(net.output()[0], 6.5));  // 2*3 + 0.5
    }

    // ActivateLeaky integrates the membrane potential over steps.
    {
        NeuralNetwork net;
        Neuron inN, outN;
        inN.activesum_ = inN.activation_ = 0;
        inN.a_ = 1;
        inN.b_ = inN.timeconst_ = inN.bias_ = inN.membranePotential_ = 0;
        inN.activationFunctionType_ = LINEAR;
        inN.type_ = INPUT;
        outN = inN;
        outN.type_ = OUTPUT;
        outN.activationFunctionType_ = LINEAR;
        outN.timeconst_ = 1.0;
        Connection conn;
        conn.sourceNeuronIndex_ = 0;
        conn.targetNeuronIndex_ = 1;
        conn.weight_ = 1.0;
        conn.signal_ = 0;
        conn.recurFlag_ = false;
        conn.hebbRate_ = conn.hebbPreRate_ = 0;
        net.addNeuron(inN);
        net.addNeuron(outN);
        net.addConnection(conn);
        net.setInputOutputDimensions(1, 1);
        std::vector<double> in{4.0};
        net.input(in);
        net.activateLeaky(0.5);  // mp = 0.5 * 4 = 2
        CHECK(near(net.output()[0], 2.0));
        net.activateLeaky(0.5);  // mp = 0.5 * 2 + 0.5 * 4 = 3
        CHECK(near(net.output()[0], 3.0));
    }

    // Flush zeroes activations, active sums and membrane potentials.
    {
        NeuralNetwork net(false);  // XOR topology, random weights
        net.flush();
        std::vector<double> in{1.0, 1.0, 1.0};
        net.input(in);
        net.activate();
        CHECK(!near(net.output()[0], 0.0));
        net.flush();
        for (const Neuron &n : net.neurons_) {
            CHECK(n.activation_ == 0.0);
            CHECK(n.activesum_ == 0.0);
            CHECK(n.membranePotential_ == 0.0);
        }
    }

    // RTRL pipeline: gradients -> error -> weights, all finite with a real update.
    {
        NeuralNetwork net(false);
        for (Neuron &n : net.neurons_) {
            n.activationFunctionType_ = UNSIGNED_SIGMOID;  // RTRL knows sigmoids only
        }
        net.initRTRLMatrix();
        net.flush();
        CHECK(net.neurons_[0].sensitivityMatrix_.size() == net.neurons_.size());
        std::vector<double> in{1.0, 1.0, 1.0};
        net.input(in);
        net.activate();
        const double wBefore = net.connections_[0].weight_;
        net.rtrlUpdateGradients();
        net.rtrlUpdateError(1.0);
        net.rtrlUpdateWeights();
        CHECK(std::isfinite(net.connections_[0].weight_));
        CHECK(!near(net.connections_[0].weight_, wBefore, 1e-12));
        net.flushCube();
    }

    // Hebbian Adapt moves weights and clamps them to [-MaxWeight, MaxWeight].
    {
        Parameters p;
        p.reset();
        NeuralNetwork net;
        Neuron inN, outN;
        inN.activesum_ = inN.activation_ = 0;
        inN.a_ = 1;
        inN.b_ = inN.timeconst_ = inN.bias_ = inN.membranePotential_ = 0;
        inN.activationFunctionType_ = LINEAR;
        inN.type_ = INPUT;
        outN = inN;
        outN.type_ = OUTPUT;
        outN.activationFunctionType_ = LINEAR;
        Connection conn;
        conn.sourceNeuronIndex_ = 0;
        conn.targetNeuronIndex_ = 1;
        conn.weight_ = 0.5;
        conn.signal_ = 0;
        conn.recurFlag_ = false;
        conn.hebbRate_ = 0.1;
        conn.hebbPreRate_ = 0.1;
        net.addNeuron(inN);
        net.addNeuron(outN);
        net.addConnection(conn);
        net.setInputOutputDimensions(1, 1);
        std::vector<double> in{1.0};
        net.input(in);
        net.activate();  // out = 0.5
        net.adapt(p);
        // delta = 0.1*(0.5-0.5)*1*0.5 + 0.1*0.5*1*(0.5-1) = -0.025
        CHECK(near(net.connections_[0].weight_, 0.475, 1e-9));

        // Oversized weight gets clamped to MaxWeight.
        net.connections_[0].weight_ = 999.0;
        net.flush();
        net.input(in);
        net.activate();
        net.adapt(p);
        CHECK(net.connections_[0].weight_ <= p.maxWeight);
    }

    if (g_failures != 0) {
        std::cerr << "Test failed: TestNeuralNetwork with " << g_failures << " failure(s)\n";
        return 1;
    }
    std::cout << "Test passed: TestNeuralNetwork\n";
    return 0;
}
