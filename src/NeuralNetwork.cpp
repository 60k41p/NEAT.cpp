/*
 * NEAT.cpp: Portable, Zero-dependency C++17 NeuroEvolution Library
 *
 * Copyright (C) 2012 Peter Chervenski
 * Modifications Copyright (C) 2026 Gökalp Özcan
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 *
 * This file has been modified from its original version by Gökalp Özcan in 2026.
 *
 * Contact info:
 * Peter Chervenski <spookey@abv.bg>
 * Shane Ryan <shane.mcdonald.ryan@gmail.com>
 * Gökalp Özcan <gokalp@mail.com>
 */

/*
 * File:        NeuralNetwork.cpp
 * Description: Implementation of the phenotype activation functions.
 */

#include "NeuralNetwork.h"

#include <cfloat>
#include <cmath>
#include <fstream>
#include <iostream>
#include <random>
#include <sstream>
#include <string>

#include "AssertMacros.h"
#include "Utils.h"

#define LEARNING_RATE 0.0001

namespace NEAT {

    /////////////////////////////////////
    // The set of activation functions //
    /////////////////////////////////////

    inline Real activationSigmoidUnsigned(Real x, Real slope, Real shift) { return 1.0 / (1.0 + std::exp(-slope * x - shift)); }

    inline Real activationSigmoidSigned(Real x, Real slope, Real shift) {
        Real y = activationSigmoidUnsigned(x, slope, shift);
        return (y - 0.5) * 2.0;
    }

    inline Real activationTanh(Real x, Real slope, Real shift) { return std::tanh(x * slope); }

    inline Real activationTanhCubic(Real x, Real slope, Real shift) { return std::tanh(x * x * x * slope); }

    inline Real activationStepSigned(Real x, Real shift) {
        Real y;
        if (x > shift) {
            y = 1.0;
        } else {
            y = -1.0;
        }

        return y;
    }

    inline Real activationStepUnsigned(Real x, Real shift) {
        if (x > (0.5 + shift)) {
            return 1.0;
        } else {
            return 0.0;
        }
    }

    inline Real activationGaussSigned(Real x, Real slope, Real shift) {
        Real y = std::exp(-slope * x * x + shift);  // TODO: Need separate a, b per activation function
        return (y - 0.5) * 2.0;
    }

    inline Real activationGaussUnsigned(Real x, Real slope, Real shift) { return std::exp(-slope * x * x + shift); }

    inline Real activationAbs(Real x, Real shift) { return ((x + shift) < 0.0) ? -(x + shift) : (x + shift); }

    inline Real activationSineSigned(Real x, Real freq, Real shift) { return std::sin(x * freq + shift); }

    inline Real activationSineUnsigned(Real x, Real freq, Real shift) {
        Real y = std::sin((x * freq + shift));
        return (y + 1.0) / 2.0;
    }

    inline Real activationLinear(Real x, Real shift) { return (x + shift); }

    inline Real activationRelu(Real x) { return (x > 0) ? x : 0; }

    inline Real activationSoftplus(Real x) { return std::log(1 + std::exp(x)); }

    Real unsignedSigmoidDerivative(Real x) { return x * (1 - x); }

    Real tanhDerivative(Real x) { return 1 - x * x; }

    ///////////////////////////////////////
    // Neural network class implementation
    ///////////////////////////////////////
    NeuralNetwork::NeuralNetwork(bool minimal) {
        if (!minimal) {
            // build an XOR network

            // The input neurons are 3 // indexes 0 1 2
            Neuron i1{}, i2{}, i3{};

            // The output neuron       // index 3
            Neuron o1{};

            // The hidden neuron       // index 4
            Neuron h1{};

            neurons_.emplace_back(i1);
            neurons_.emplace_back(i2);
            neurons_.emplace_back(i3);
            neurons_.emplace_back(o1);
            neurons_.emplace_back(h1);

            // The connections
            Connection c{};

            c.sourceNeuronIndex_ = 0;
            c.targetNeuronIndex_ = 3;
            c.weight_ = 0;
            connections_.emplace_back(c);

            c.sourceNeuronIndex_ = 1;
            c.targetNeuronIndex_ = 3;
            c.weight_ = 0;
            connections_.emplace_back(c);

            c.sourceNeuronIndex_ = 2;
            c.targetNeuronIndex_ = 3;
            c.weight_ = 0;
            connections_.emplace_back(c);

            c.sourceNeuronIndex_ = 0;
            c.targetNeuronIndex_ = 4;
            c.weight_ = 0;
            connections_.emplace_back(c);

            c.sourceNeuronIndex_ = 1;
            c.targetNeuronIndex_ = 4;
            c.weight_ = 0;
            connections_.emplace_back(c);

            c.sourceNeuronIndex_ = 2;
            c.targetNeuronIndex_ = 4;
            c.weight_ = 0;
            connections_.emplace_back(c);

            c.sourceNeuronIndex_ = 4;
            c.targetNeuronIndex_ = 3;
            c.weight_ = 0;
            connections_.emplace_back(c);

            numInputs_ = 3;
            numOutputs_ = 1;

            // Initialize the network's weights (make them random)
            std::mt19937 weightEngine(std::random_device{}());
            std::uniform_real_distribution<Real> weightDist(-0.5, 0.5);
            for (unsigned int i = 0; i < connections_.size(); i++) {
                connections_[i].weight_ = weightDist(weightEngine);
            }

            // clean up other neuron data as well
            for (unsigned int i = 0; i < neurons_.size(); i++) {
                neurons_[i].a_ = 1;
                neurons_[i].b_ = 0;
                neurons_[i].timeconst_ = neurons_[i].bias_ = neurons_[i].membranePotential_ = 0;
            }

            initRTRLMatrix();
        } else {
            // an empty network
            numInputs_ = numOutputs_ = 0;
            totalError_ = 0;
            // clean up other neuron data as well
            for (unsigned int i = 0; i < neurons_.size(); i++) {
                neurons_[i].a_ = 1;
                neurons_[i].b_ = 0;
                neurons_[i].timeconst_ = neurons_[i].bias_ = neurons_[i].membranePotential_ = 0;
            }
            clear();
        }
    }

    NeuralNetwork::NeuralNetwork() {
        // an empty network
        numInputs_ = numOutputs_ = 0;
        totalError_ = 0;
        // clean up other neuron data as well
        for (unsigned int i = 0; i < neurons_.size(); i++) {
            neurons_[i].a_ = 1;
            neurons_[i].b_ = 0;
            neurons_[i].timeconst_ = neurons_[i].bias_ = neurons_[i].membranePotential_ = 0;
        }
        clear();
    }

    void NeuralNetwork::initRTRLMatrix() {
        // Allocate memory for the neurons sensitivity matrices.
        for (unsigned int i = 0; i < neurons_.size(); i++) {
            neurons_[i].sensitivityMatrix_.resize(neurons_.size());  // first dimention
            for (unsigned int j = 0; j < neurons_.size(); j++) {
                neurons_[i].sensitivityMatrix_[j].resize(neurons_.size());  // second dimention
            }
        }

        // now clear it
        flushCube();
        // clear out the other RTRL stuff as well
        totalError_ = 0;
        totalWeightChange_.resize(connections_.size());
        for (unsigned int i = 0; i < connections_.size(); i++) {
            totalWeightChange_[i] = 0;
        }
    }

    void NeuralNetwork::activateFast() {
        // Loop connections. Calculate each connection's output signal.
        for (unsigned int i = 0; i < connections_.size(); i++) {
            connections_[i].signal_ = neurons_[connections_[i].sourceNeuronIndex_].activation_ * connections_[i].weight_;
        }
        // Loop the connections again. This time add the signals to the target neurons. This will largely require out of order memory writes. This is the one
        // loop where this will happen.
        for (unsigned int i = 0; i < connections_.size(); i++) {
            neurons_[connections_[i].targetNeuronIndex_].activesum_ += connections_[i].signal_;
        }
        // Now loop nodes_activesums, pass the signals through the activation function and store the result back to nodes_activations also skip inputs since
        // they do not get an activation
        for (unsigned int i = numInputs_; i < neurons_.size(); i++) {
            Real x = neurons_[i].activesum_;
            neurons_[i].activesum_ = 0;
            // Apply the activation function
            Real y = 0.0;
            y = activationSigmoidUnsigned(x, neurons_[i].a_, neurons_[i].b_);
            neurons_[i].activation_ = y;
        }
    }

    void NeuralNetwork::activate() {
        // Loop connections. Calculate each connection's output signal.
        for (unsigned int i = 0; i < connections_.size(); i++) {
            connections_[i].signal_ = neurons_[connections_[i].sourceNeuronIndex_].activation_ * connections_[i].weight_;
        }
        // Loop the connections again. This time add the signals to the target neurons. This will largely require out of order memory writes. This is the one
        // loop where this will happen.
        for (unsigned int i = 0; i < connections_.size(); i++) {
            neurons_[connections_[i].targetNeuronIndex_].activesum_ += connections_[i].signal_;
        }
        // Now loop nodes_activesums, pass the signals through the activation function and store the result back to nodes_activations also skip inputs since
        // they do not get an activation
        for (unsigned int i = numInputs_; i < neurons_.size(); i++) {
            Real x = neurons_[i].activesum_;
            neurons_[i].activesum_ = 0;
            // Apply the activation function
            Real y = 0.0;
            switch (neurons_[i].activationFunctionType_) {
                case SIGNED_SIGMOID:
                    y = activationSigmoidSigned(x, neurons_[i].a_, neurons_[i].b_);
                    break;
                case UNSIGNED_SIGMOID:
                    y = activationSigmoidUnsigned(x, neurons_[i].a_, neurons_[i].b_);
                    break;
                case TANH:
                    y = activationTanh(x, neurons_[i].a_, neurons_[i].b_);
                    break;
                case TANH_CUBIC:
                    y = activationTanhCubic(x, neurons_[i].a_, neurons_[i].b_);
                    break;
                case SIGNED_STEP:
                    y = activationStepSigned(x, neurons_[i].b_);
                    break;
                case UNSIGNED_STEP:
                    y = activationStepUnsigned(x, neurons_[i].b_);
                    break;
                case SIGNED_GAUSS:
                    y = activationGaussSigned(x, neurons_[i].a_, neurons_[i].b_);
                    break;
                case UNSIGNED_GAUSS:
                    y = activationGaussUnsigned(x, neurons_[i].a_, neurons_[i].b_);
                    break;
                case ABS:
                    y = activationAbs(x, neurons_[i].b_);
                    break;
                case SIGNED_SINE:
                    y = activationSineSigned(x, neurons_[i].a_, neurons_[i].b_);
                    break;
                case UNSIGNED_SINE:
                    y = activationSineUnsigned(x, neurons_[i].a_, neurons_[i].b_);
                    break;
                case LINEAR:
                    y = activationLinear(x, neurons_[i].b_);
                    break;
                case RELU:
                    y = activationRelu(x);
                    break;
                case SOFTPLUS:
                    y = activationSoftplus(x);
                    break;
                default:
                    y = activationSigmoidUnsigned(x, neurons_[i].a_, neurons_[i].b_);
                    break;
            }
            neurons_[i].activation_ = y;
        }
    }

    void NeuralNetwork::activateUseInternalBias() {
        // Loop connections. Calculate each connection's output signal.
        for (unsigned int i = 0; i < connections_.size(); i++) {
            connections_[i].signal_ = neurons_[connections_[i].sourceNeuronIndex_].activation_ * connections_[i].weight_;
        }
        // Loop the connections again. This time add the signals to the target neurons. This will largely require out of order memory writes. This is the one
        // loop where this will happen.
        for (unsigned int i = 0; i < connections_.size(); i++) {
            neurons_[connections_[i].targetNeuronIndex_].activesum_ += connections_[i].signal_;
        }
        // Now loop nodes_activesums, pass the signals through the activation function and store the result back to nodes_activations also skip inputs since
        // they do not get an activation
        for (unsigned int i = numInputs_; i < neurons_.size(); i++) {
            Real x = neurons_[i].activesum_ + neurons_[i].bias_;
            neurons_[i].activesum_ = 0;
            // Apply the activation function
            Real y = 0.0;
            switch (neurons_[i].activationFunctionType_) {
                case SIGNED_SIGMOID:
                    y = activationSigmoidSigned(x, neurons_[i].a_, neurons_[i].b_);
                    break;
                case UNSIGNED_SIGMOID:
                    y = activationSigmoidUnsigned(x, neurons_[i].a_, neurons_[i].b_);
                    break;
                case TANH:
                    y = activationTanh(x, neurons_[i].a_, neurons_[i].b_);
                    break;
                case TANH_CUBIC:
                    y = activationTanhCubic(x, neurons_[i].a_, neurons_[i].b_);
                    break;
                case SIGNED_STEP:
                    y = activationStepSigned(x, neurons_[i].b_);
                    break;
                case UNSIGNED_STEP:
                    y = activationStepUnsigned(x, neurons_[i].b_);
                    break;
                case SIGNED_GAUSS:
                    y = activationGaussSigned(x, neurons_[i].a_, neurons_[i].b_);
                    break;
                case UNSIGNED_GAUSS:
                    y = activationGaussUnsigned(x, neurons_[i].a_, neurons_[i].b_);
                    break;
                case ABS:
                    y = activationAbs(x, neurons_[i].b_);
                    break;
                case SIGNED_SINE:
                    y = activationSineSigned(x, neurons_[i].a_, neurons_[i].b_);
                    break;
                case UNSIGNED_SINE:
                    y = activationSineUnsigned(x, neurons_[i].a_, neurons_[i].b_);
                    break;
                case LINEAR:
                    y = activationLinear(x, neurons_[i].b_);
                    break;
                case RELU:
                    y = activationRelu(x);
                    break;
                case SOFTPLUS:
                    y = activationSoftplus(x);
                    break;
                default:
                    y = activationSigmoidUnsigned(x, neurons_[i].a_, neurons_[i].b_);
                    break;
            }
            neurons_[i].activation_ = y;
        }
    }

    void NeuralNetwork::activateLeaky(Real dtime) {
        // Loop connections. Calculate each connection's output signal.
        for (unsigned int i = 0; i < connections_.size(); i++) {
            connections_[i].signal_ = neurons_[connections_[i].sourceNeuronIndex_].activation_ * connections_[i].weight_;
        }
        // Loop the connections again. This time add the signals to the target neurons. This will largely require out of order memory writes. This is the one
        // loop where this will happen.
        for (unsigned int i = 0; i < connections_.size(); i++) {
            neurons_[connections_[i].targetNeuronIndex_].activesum_ += connections_[i].signal_;
        }
        // Now we have the leaky integrator step for the neurons
        for (unsigned int i = numInputs_; i < neurons_.size(); i++) {
            Real timeFactor = dtime / neurons_[i].timeconst_;
            neurons_[i].membranePotential_ = (1.0 - timeFactor) * neurons_[i].membranePotential_ + timeFactor * neurons_[i].activesum_;
        }
        // Now loop nodes_activesums, pass the signals through the activation function and store the result back to nodes_activations also skip inputs since
        // they do not get an activation
        for (unsigned int i = numInputs_; i < neurons_.size(); i++) {
            Real x = neurons_[i].membranePotential_ + neurons_[i].bias_;
            neurons_[i].activesum_ = 0;
            // Apply the activation function
            Real y = 0.0;
            switch (neurons_[i].activationFunctionType_) {
                case SIGNED_SIGMOID:
                    y = activationSigmoidSigned(x, neurons_[i].a_, neurons_[i].b_);
                    break;
                case UNSIGNED_SIGMOID:
                    y = activationSigmoidUnsigned(x, neurons_[i].a_, neurons_[i].b_);
                    break;
                case TANH:
                    y = activationTanh(x, neurons_[i].a_, neurons_[i].b_);
                    break;
                case TANH_CUBIC:
                    y = activationTanhCubic(x, neurons_[i].a_, neurons_[i].b_);
                    break;
                case SIGNED_STEP:
                    y = activationStepSigned(x, neurons_[i].b_);
                    break;
                case UNSIGNED_STEP:
                    y = activationStepUnsigned(x, neurons_[i].b_);
                    break;
                case SIGNED_GAUSS:
                    y = activationGaussSigned(x, neurons_[i].a_, neurons_[i].b_);
                    break;
                case UNSIGNED_GAUSS:
                    y = activationGaussUnsigned(x, neurons_[i].a_, neurons_[i].b_);
                    break;
                case ABS:
                    y = activationAbs(x, neurons_[i].b_);
                    break;
                case SIGNED_SINE:
                    y = activationSineSigned(x, neurons_[i].a_, neurons_[i].b_);
                    break;
                case UNSIGNED_SINE:
                    y = activationSineUnsigned(x, neurons_[i].a_, neurons_[i].b_);
                    break;
                case LINEAR:
                    y = activationLinear(x, neurons_[i].b_);
                    break;
                case RELU:
                    y = activationRelu(x);
                    break;
                case SOFTPLUS:
                    y = activationSoftplus(x);
                    break;
                default:
                    y = activationSigmoidUnsigned(x, neurons_[i].a_, neurons_[i].b_);
                    break;
            }
            neurons_[i].activation_ = y;
        }
    }

    void NeuralNetwork::flush() {
        for (unsigned int i = 0; i < neurons_.size(); i++) {
            neurons_[i].activation_ = 0;
            neurons_[i].activesum_ = 0;
            neurons_[i].membranePotential_ = 0;
        }
    }

    void NeuralNetwork::flushCube() {
        // clear the cube
        for (unsigned int i = 0; i < neurons_.size(); i++)
            for (unsigned int j = 0; j < neurons_.size(); j++)
                for (unsigned int k = 0; k < neurons_.size(); k++) neurons_[k].sensitivityMatrix_[i][j] = 0;
    }
    void NeuralNetwork::input(std::vector<Real> &inputs) {
        unsigned mx = inputs.size();
        if (mx > numInputs_) {
            mx = numInputs_;
        }

        for (unsigned int i = 0; i < mx; i++) {
            neurons_[i].activation_ = inputs[i];
        }
    }

    std::vector<Real> NeuralNetwork::output() {
        std::vector<Real> output;
        for (int i = 0; i < numOutputs_; i++) {
            output.emplace_back(neurons_[i + numInputs_].activation_);
        }
        return output;
    }

    void NeuralNetwork::adapt(Parameters &parameters) {
        // find max absolute magnitude of the weight
        Real maxWeight = -999999999;
        for (unsigned int i = 0; i < connections_.size(); i++) {
            if (std::fabs(connections_[i].weight_) > maxWeight) {
                maxWeight = std::fabs(connections_[i].weight_);
            }
        }

        for (unsigned int i = 0; i < connections_.size(); i++) {
            /////////////////////////////////////
            // modify weight of that connection
            ////
            Real incomingNeuronActivation = neurons_[connections_[i].sourceNeuronIndex_].activation_;
            Real outgoingNeuronActivation = neurons_[connections_[i].targetNeuronIndex_].activation_;
            if (connections_[i].weight_ > 0)  // positive weight
            {
                Real delta = (connections_[i].hebbRate_ * (maxWeight - connections_[i].weight_) * incomingNeuronActivation * outgoingNeuronActivation) +
                             connections_[i].hebbPreRate_ * maxWeight * incomingNeuronActivation * (outgoingNeuronActivation - 1.0);
                connections_[i].weight_ = (connections_[i].weight_ + delta);
            } else if (connections_[i].weight_ < 0)  // negative weight
            {
                // In the inhibatory case, we strengthen the synapse when output is low and input is high
                Real delta =
                    connections_[i].hebbPreRate_ * (maxWeight - connections_[i].weight_) * incomingNeuronActivation * (1.0 - outgoingNeuronActivation) -
                    connections_[i].hebbRate_ * maxWeight * incomingNeuronActivation * outgoingNeuronActivation;
                connections_[i].weight_ = -(connections_[i].weight_ + delta);
            }

            clamp(connections_[i].weight_, -parameters.maxWeight, parameters.maxWeight);
        }
    }

    int NeuralNetwork::connectionExists(int to, int from) {
        for (unsigned int i = 0; i < connections_.size(); i++) {
            if ((connections_[i].sourceNeuronIndex_ == from) && (connections_[i].targetNeuronIndex_ == to)) {
                return i;
            }
        }

        return -1;
    }

    void NeuralNetwork::rtrlUpdateGradients() {
        // for every neuron
        for (unsigned int k = numInputs_; k < neurons_.size(); k++) {
            // for all possible connections
            for (unsigned int i = numInputs_; i < neurons_.size(); i++)
                // to
                for (unsigned int j = 0; j < neurons_.size(); j++)  // from
                {
                    int index = connectionExists(i, j);
                    if (index != -1) {
                        // Real t_derivative = unsigned_sigmoid_derivative( m_neurons[k].m_activation );
                        Real derivative = 0;
                        if (neurons_[k].activationFunctionType_ == NEAT::UNSIGNED_SIGMOID) {
                            derivative = unsignedSigmoidDerivative(neurons_[k].activation_);
                        } else if (neurons_[k].activationFunctionType_ == NEAT::TANH) {
                            derivative = tanhDerivative(neurons_[k].activation_);
                        }

                        Real sum = 0;
                        // calculate the other sum
                        for (unsigned int l = 0; l < neurons_.size(); l++) {
                            int lIndex = connectionExists(k, l);
                            if (lIndex != -1) {
                                sum += connections_[lIndex].weight_ * neurons_[l].sensitivityMatrix_[i][j];
                            }
                        }

                        if (i == k) {
                            sum += neurons_[j].activation_;
                        }
                        neurons_[k].sensitivityMatrix_[i][j] = derivative * sum;
                    } else {
                        neurons_[k].sensitivityMatrix_[i][j] = 0;
                    }
                }
        }
    }

    // please pay attention. notice here only one output is assumed
    void NeuralNetwork::rtrlUpdateError(Real target) {
        // add to total error
        totalError_ = (target - output()[0]);
        // adjust each weight
        for (unsigned int i = 0; i < neurons_.size(); i++)  // to
        {
            for (unsigned int j = 0; j < neurons_.size(); j++)  // from
            {
                int index = connectionExists(i, j);
                if (index != -1) {
                    // we know the first output's index is m_num_inputs
                    Real delta = totalError_ * neurons_[numInputs_].sensitivityMatrix_[i][j];
                    totalWeightChange_[index] += delta * LEARNING_RATE;
                }
            }
        }
    }

    void NeuralNetwork::rtrlUpdateWeights() {
        for (unsigned int i = 0; i < connections_.size(); i++) {
            connections_[i].weight_ += totalWeightChange_[i];
            totalWeightChange_[i] = 0;  // clear this out
        }
        totalError_ = 0;
    }

    void NeuralNetwork::save(const char *filename) {
        FILE *fil = fopen(filename, "w");
        save(fil);
        fclose(fil);
    }

    void NeuralNetwork::save(FILE *file) {
        fprintf(file, "NNstart\n");
        // save num inputs/outputs and stuff
        fprintf(file, "%d %d\n", numInputs_, numOutputs_);
        // save neurons
        for (unsigned int i = 0; i < neurons_.size(); i++) {
            // TYPE .. A .. B .. time_const .. bias .. activation_function_type .. split_y
            fprintf(file, "neuron %d %3.18f %3.18f %3.18f %3.18f %d %3.18f\n", static_cast<int>(neurons_[i].type_), neurons_[i].a_, neurons_[i].b_,
                    neurons_[i].timeconst_, neurons_[i].bias_, static_cast<int>(neurons_[i].activationFunctionType_), neurons_[i].splitY_);
        }
        // save connections
        for (unsigned int i = 0; i < connections_.size(); i++) {
            // from .. to .. weight.. isrecur
            fprintf(file, "connection %d %d %3.18f %d %3.18f %3.18f\n", connections_[i].sourceNeuronIndex_, connections_[i].targetNeuronIndex_,
                    connections_[i].weight_, static_cast<int>(connections_[i].recurFlag_), connections_[i].hebbRate_, connections_[i].hebbPreRate_);
        }
        // end
        fprintf(file, "NNend\n\n");
    }

    bool NeuralNetwork::load(std::ifstream &dataFile) {
        std::string str;
        bool noStart = true, noEnd = true;

        if (!dataFile) {
            std::ostringstream tStream;
            tStream << "NN file error!" << '\n';
            //    throw NS::Exception(tStream.str());
        }

        // search for NNstart
        do {
            dataFile >> str;
            if (str == "NNstart") noStart = false;

        } while ((str != "NNstart") && (!dataFile.eof()));

        if (noStart) return false;

        clear();

        // read in the input/output dimentions
        dataFile >> numInputs_;
        dataFile >> numOutputs_;

        // read in all data
        do {
            dataFile >> str;

            // a neuron?
            if (str == "neuron") {
                Neuron n;

                // for type and aftype
                int type, aftype;

                dataFile >> type;
                dataFile >> n.a_;
                dataFile >> n.b_;
                dataFile >> n.timeconst_;
                dataFile >> n.bias_;
                dataFile >> aftype;
                dataFile >> n.splitY_;

                n.type_ = static_cast<NEAT::NeuronType>(type);
                n.activationFunctionType_ = static_cast<NEAT::ActivationFunction>(aftype);

                neurons_.emplace_back(n);
            }

            // a connection?
            if (str == "connection") {
                Connection c;

                int isrecur;

                dataFile >> c.sourceNeuronIndex_;
                dataFile >> c.targetNeuronIndex_;
                dataFile >> c.weight_;
                dataFile >> isrecur;

                dataFile >> c.hebbRate_;
                dataFile >> c.hebbPreRate_;

                c.recurFlag_ = static_cast<bool>(isrecur);

                connections_.emplace_back(c);
            }

            if (str == "NNend") noEnd = false;
        } while ((str != "NNend") && (!dataFile.eof()));

        if (noEnd) {
            std::ostringstream tStream;
            tStream << "NNend not found in file!" << '\n';
            //    throw NS::Exception(tStream.str());
        }

        return true;
    }
    bool NeuralNetwork::load(const char *filename) {
        std::ifstream dataFile(filename);
        return load(dataFile);
    }

};  // namespace NEAT
