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
 * File:        NeuralNetwork.h
 * Description: Phenotype representation and activation: Connection/Neuron structs plus the NeuralNetwork
 *              runtime built from a Genome (see Genome::buildPhenotype()). Supports plain feed-forward
 *              activation (activate/activateFast), internal-bias and leaky-integrator modes, Hebbian
 *              lifetime adaptation (adapt()) and RTRL gradient machinery for recurrent learning.
 *
 * References: Stanley & Miikkulainen (2002), Section 2 (phenotype decoding); Williams & Zipser, "A Learning
 *             Algorithm for Continually Running Fully Recurrent Neural Networks" (1989) for the RTRL terms;
 *             Hebbian updates follow the trait-gated "hebb_rate"/"hebb_pre_rate" link traits.
 *             Intra-repo users: src/Genome.h, src/Genome.cpp, src/Substrate.h, tests/TestNeuralNetwork.cpp.
 */

#pragma once

#include <vector>

#include "Genes.h"
#include "Types.h"

namespace NEAT {

    // One weighted edge of the phenotype: source/target neuron indexes plus the live signal cache.
    class Connection {
       public:
        // Index of the source neuron in NeuralNetwork::neurons_.
        int sourceNeuronIndex_;
        // Index of the target neuron in NeuralNetwork::neurons_.
        int targetNeuronIndex_;
        // Connection weight (copied from the genome at build time).
        Real weight_;
        // Cached weight * source activation, refreshed by activate*().
        Real signal_;

        // Recurrence flag (display/diagnostics only; activation order does not depend on it).
        bool recurFlag_;

        // Hebbian lifetime-learning rates (see adapt()); ignored when the link traits are absent.
        Real hebbRate_;
        Real hebbPreRate_;

        // Compares by topology (source/target indexes) so tests can match structure ignoring weights.
        bool operator==(Connection const &other) const {
            if ((sourceNeuronIndex_ == other.sourceNeuronIndex_) && (targetNeuronIndex_ == other.targetNeuronIndex_)) /*&&
                                                                      (m_weight == other.m_weight) &&
                                                                      (m_recur_flag == other.m_recur_flag))*/
                return true;
            else
                return false;
        }
    };

    // One node of the phenotype: live activation state plus the copied-over genome parameters.
    class Neuron {
       public:
        // Summed weighted input for the current step.
        Real activesum_;
        // activesum_ passed through the activation function.
        Real activation_;

        // Activation-function parameters (slope/shift/time-constant/bias slots; meaning depends on activationFunctionType_).
        Real a_, b_, timeconst_, bias_;
        // Leaky-integrator membrane potential (activateLeaky() only).
        Real membranePotential_;
        // Which activation function activate() applies to this neuron.
        ActivationFunction activationFunctionType_;

        // Display coordinates and substrate position (HyperNEAT queries); splitY_ is network depth.
        Real x_, y_, z_;
        Real sx_, sy_, sz_;
        std::vector<Real> substrateCoords_;
        Real splitY_;
        NeuronType type_;

        // Per-neuron sensitivity cube for RTRL learning (see initRTRLMatrix()).
        std::vector<std::vector<Real> > sensitivityMatrix_;

        // Compares by role/depth/activation type so tests can match structure ignoring live state.
        bool operator==(Neuron const &other) const {
            if ((type_ == other.type_) && (splitY_ == other.splitY_) && (activationFunctionType_ == other.activationFunctionType_))
                return true;
            else
                return false;
        }
    };

    // The executable phenotype. Build it from a genome (Genome::buildPhenotype()), feed inputs via input(),
    // run one of the activate*() modes, then read output(). Not thread-safe: activation mutates caches.
    class NeuralNetwork {
        /////////////////////
        // RTRL bookkeeping (see initRTRLMatrix(); empty unless RTRL learning runs)
        Real totalError_;

        // Accumulated per-connection weight change, always sized like connections_.
        std::vector<Real> totalWeightChange_;
        /////////////////////

        // Returns the connection index for the (to, from) pair, or -1 when absent.
        int connectionExists(int to, int from);

       public:
        unsigned int numInputs_, numOutputs_;
        // All edges; indexes must stay consistent with Neuron positions in neurons_.
        std::vector<Connection> connections_;
        std::vector<Neuron> neurons_;

        NeuralNetwork(bool minimal);  // if given false, the constructor will create a standard XOR network topology.
        NeuralNetwork();

        // Allocates the per-neuron sensitivity cube; call after the topology is final.
        void initRTRLMatrix();
        // assumes that neuron and connection data are already initialized

        // Single synchronous step assuming unsigned sigmoids everywhere (fastest; skips the type switch).
        void activateFast();
        // Single synchronous step honoring each neuron's activation function.
        void activate();
        // Like activate() but adds the neuron bias term during summation.
        void activateUseInternalBias();
        // Leaky-integrator step with the given time delta.
        void activateLeaky(Real step);

        // RTRL gradient accumulation / error injection / weight update triplet.
        void rtrlUpdateGradients();
        void rtrlUpdateError(Real target);
        // Performs the backprop step.
        void rtrlUpdateWeights();

        // Hebbian lifetime adaptation gated by the link "hebb_rate" traits.
        void adapt(Parameters &parameters);

        // Zeroes all activations (keeps topology and weights).
        void flush();
        // Zeroes the RTRL sensitivity cube.
        void flushCube();

        // Loads the input layer (size must equal numInputs()).
        void input(std::vector<Real> &inputs);

        // Reads the output layer after activation.
        std::vector<Real> output();

        // Appends one neuron/connection (no dedup; callers keep indexes consistent).
        void addNeuron(const Neuron &n) { neurons_.push_back(n); }
        void addConnection(const Connection &c) { connections_.push_back(c); }
        // Copies out a single connection/neuron by position.
        Connection getConnectionByIndex(unsigned int index) const { return connections_[index]; }
        Neuron getNeuronByIndex(unsigned int index) const { return neurons_[index]; }
        // Records the input/output counts (must match the leading/trailing neuron roles).
        void setInputOutputDimensions(const unsigned int i, const unsigned int o) {
            numInputs_ = i;
            numOutputs_ = o;
        }
        unsigned int numInputs() const { return numInputs_; }
        unsigned int numOutputs() const { return numOutputs_; }

        // Resets to an empty network with zero I/O.
        void clear() {
            neurons_.clear();
            connections_.clear();
            totalWeightChange_.clear();
            setInputOutputDimensions(0, 0);
        }

        // Squared Euclidean distance between two neurons' substrate coordinates (HyperNEAT diagnostics).
        Real getConnectionLength(const Neuron &source, const Neuron &target) {
            Real dist = 0.0;
            for (unsigned int i = 0; i < source.substrateCoords_.size(); i++) {
                dist += (target.substrateCoords_[i] - source.substrateCoords_[i]) * (target.substrateCoords_[i] - source.substrateCoords_[i]);
            }
            return dist;
        }

        // Number of connections (legacy name spoke of length; it has always been a count).
        Real getConnectionCount() { return static_cast<Real>(connections_.size()); }

        // one-shot save/load
        void save(const char *filename);
        bool load(const char *filename);

        // save/load from already opened files for reading/writing
        void save(FILE *file);
        bool load(std::ifstream &dataFile);
    };

};  // namespace NEAT
