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
 * Description: Definition for the phenotype data structures.
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <string>
#include <utility>
#include <vector>

#include "Genes.h"
#include "Types.h"

namespace NEAT {

    // How external values enter a spiking network and how outputs are read back.
    enum SpikingInputMode { CURRENT_INPUT = 0, BINARY_SPIKE_INPUT, POISSON_RATE_INPUT };

    enum SpikingOutputMode { SPIKE_OUTPUT = 0, FIRING_RATE_OUTPUT, FILTERED_SPIKE_OUTPUT, MEMBRANE_POTENTIAL_OUTPUT };

    // A recorded spike (or external input event) on the spiking time axis.
    struct SpikeEvent {
        Real time = 0.0;
        int neuron_index = 0;
        Real amplitude = 1.0;
        bool input = false;
    };

    // A synaptically delayed event waiting for delivery.
    struct PendingSynapticEvent {
        Real delivery_time = 0.0;
        Real amplitude = 0.0;
        Real source_amplitude = 0.0;
    };

    class Connection {
       public:
        int m_source_neuron_idx;  // index of source neuron
        int m_target_neuron_idx;  // index of target neuron
        Real m_weight;            // weight of the connection
        Real m_signal;            // weight * input signal

        bool m_recur_flag;  // recurrence flag for displaying purposes
        // can be ignored

        // Hebbian learning parameters Ignored in case there is no lifetime learning
        Real m_hebb_rate;
        Real m_hebb_pre_rate;

        // Source activation snapshot required for exact online RTRL gradients
        // after recurrent neuron state advances.
        Real m_source_activation;
        // Current-based exponential synapse state.
        Real m_synaptic_delay;
        Real m_synaptic_time_constant;
        Real m_synaptic_current;
        Real m_presynaptic_signal;
        // Pair-based STDP state.
        bool m_stdp_enabled;
        Real m_stdp_plus;
        Real m_stdp_minus;
        Real m_stdp_tau_plus;
        Real m_stdp_tau_minus;
        Real m_stdp_pre_trace;
        Real m_stdp_post_trace;
        Real m_stdp_min_weight;
        Real m_stdp_max_weight;
        // Physical axon length in substrate coordinate units, populated by
        // HyperNEAT builders (or UpdateConnectionGeometry); separate from the
        // mutable synaptic delay.
        Real m_length;
        std::vector<PendingSynapticEvent> m_pending_events;

        Connection()
            : m_source_neuron_idx(0),
              m_target_neuron_idx(0),
              m_weight(0.0),
              m_signal(0.0),
              m_recur_flag(false),
              m_hebb_rate(0.0),
              m_hebb_pre_rate(0.0),
              m_source_activation(0.0),
              m_synaptic_delay(0.0),
              m_synaptic_time_constant(0.005),
              m_synaptic_current(0.0),
              m_presynaptic_signal(0.0),
              m_stdp_enabled(false),
              m_stdp_plus(0.01),
              m_stdp_minus(0.012),
              m_stdp_tau_plus(0.02),
              m_stdp_tau_minus(0.02),
              m_stdp_pre_trace(0.0),
              m_stdp_post_trace(0.0),
              m_stdp_min_weight(-8.0),
              m_stdp_max_weight(8.0),
              m_length(0.0) {}

        // comparison operator (nessesary for boost::python)
        bool operator==(Connection const &other) const {
            if ((m_source_neuron_idx == other.m_source_neuron_idx) && (m_target_neuron_idx == other.m_target_neuron_idx))
                return true;
            else
                return false;
        }
    };

    class Neuron {
       public:
        Real m_activesum;   // the synaptic input
        Real m_activation;  // the synaptic input passed through the activation function

        Real m_a, m_b, m_timeconst, m_bias;  // misc parameters
        Real m_membrane_potential;           // used in leaky integrator mode
        ActivationFunction m_activation_function_type;

        // displaying and stuff
        Real m_x, m_y, m_z;
        Real m_sx, m_sy, m_sz;
        std::vector<Real> m_substrate_coords;
        Real m_split_y;
        NeuronType m_type;

        // the sensitivity matrix of this neuron (for RTRL learning)
        std::vector<std::vector<Real> > m_sensitivity_matrix;

        // Pre-activation retained for exact derivatives of non-monotonic
        // activation functions during online learning.
        Real m_last_input;
        // Spiking state (LIF / adaptive LIF / Izhikevich / McCulloch-Pitts).
        Real m_spike_threshold;
        Real m_reset_potential;
        Real m_resting_potential;
        Real m_refractory_period;
        Real m_refractory_remaining;
        Real m_membrane_resistance;
        Real m_adaptation_time_constant;
        Real m_adaptation_increment;
        Real m_adaptation;
        Real m_izhikevich_a;
        Real m_izhikevich_b;
        Real m_izhikevich_c;
        Real m_izhikevich_d;
        Real m_izhikevich_recovery;
        bool m_spike;
        std::uint64_t m_spike_count;
        Real m_last_spike_time;
        Real m_rate_trace;
        Real m_rate_time_constant;
        bool m_mcp_inhibitory_veto;
        // Transient per-tick state for the canonical absolute inhibitory rule.
        bool m_inhibitory_input;

        Neuron()
            : m_activesum(0.0),
              m_activation(0.0),
              m_a(1.0),
              m_b(0.0),
              m_timeconst(1.0),
              m_bias(0.0),
              m_membrane_potential(0.0),
              m_activation_function_type(UNSIGNED_SIGMOID),
              m_x(0.0),
              m_y(0.0),
              m_z(0.0),
              m_sx(0.0),
              m_sy(0.0),
              m_sz(0.0),
              m_split_y(0.0),
              m_type(NONE),
              m_last_input(0.0),
              m_spike_threshold(1.0),
              m_reset_potential(0.0),
              m_resting_potential(0.0),
              m_refractory_period(0.002),
              m_refractory_remaining(0.0),
              m_membrane_resistance(1.0),
              m_adaptation_time_constant(0.1),
              m_adaptation_increment(0.1),
              m_adaptation(0.0),
              m_izhikevich_a(0.02),
              m_izhikevich_b(0.2),
              m_izhikevich_c(-65.0),
              m_izhikevich_d(8.0),
              m_izhikevich_recovery(-13.0),
              m_spike(false),
              m_spike_count(0),
              m_last_spike_time(-1.0),
              m_rate_trace(0.0),
              m_rate_time_constant(0.05),
              m_mcp_inhibitory_veto(true),
              m_inhibitory_input(false) {}

        // comparison operator (nessesary for boost::python)
        bool operator==(Neuron const &other) const {
            if ((m_type == other.m_type) && (m_split_y == other.m_split_y) && (m_activation_function_type == other.m_activation_function_type))
                return true;
            else
                return false;
        }
    };

    class NeuralNetwork {
        /////////////////////
        // RTRL variables
        Real m_total_error = 0.0;

        // Always the size of m_connections
        std::vector<Real> m_total_weight_change;
        // Sparse RTRL sensitivity rows, indexed like m_total_weight_change.
        std::vector<std::vector<Real> > m_sparse_rtrl_sensitivities;
        // Spiking simulation clock and configuration.
        Real m_spiking_time = 0.0;
        Real m_spiking_time_step = 0.001;
        SpikingInputMode m_spiking_input_mode = CURRENT_INPUT;
        SpikingOutputMode m_spiking_output_mode = SPIKE_OUTPUT;
        bool m_record_spikes = true;
        std::size_t m_max_recorded_spikes = 100000;
        std::uint64_t m_spiking_rng_state = UINT64_C(0x9e3779b97f4a7c15);
        std::vector<SpikeEvent> m_spike_history;
        /////////////////////

       public:
        unsigned int m_num_inputs = 0, m_num_outputs = 0;
        std::vector<Connection> m_connections;  // array size - number of connections
        std::vector<Neuron> m_neurons;

        NeuralNetwork(bool a_Minimal);  // if given false, the constructor will create a standard XOR network topology.
        NeuralNetwork();

        void InitRTRLMatrix();  // initializes the sensitivity cube for RTRL learning.
        // assumes that neuron and connection data are already initialized
        void InitSparseRTRLMatrix();  // indexed sensitivities for sparse topologies.

        void ActivateFast();             // fast path; dispatches per-neuron activations like Activate()
        void Activate();                 // any activation functions are supported
        void ActivateUseInternalBias();  // like Activate() but uses m_bias as well
        void ActivateLeaky(Real step);   // activates in leaky integrator mode

        void RTRL_update_gradients();
        void RTRL_update_gradients_sparse();
        void RTRL_update_error(Real a_target);
        void RTRL_update_error(const std::vector<Real> &targets, Real learning_rate = 0.0001);
        void RTRL_update_error_sparse(Real a_target, Real learning_rate = 0.0001);
        void RTRL_update_error_sparse(const std::vector<Real> &targets, Real learning_rate = 0.0001);
        void RTRL_update_weights();  // performs the backprop step

        // Hebbian learning
        void Adapt(Parameters &a_Parameters);

        // returns the index if that connection exists or -1 otherwise
        int ConnectionExists(int a_to, int a_from);

        void Flush();      // clears all activations
        void FlushCube();  // clears the sensitivity cube

        void Input(std::vector<Real> &a_Inputs);
        // Like Input() but requires exactly NumInputs() values.
        void InputExact(const std::vector<Real> &a_Inputs);

        std::vector<Real> Output();

        // Repeated activation and batched evaluation helpers.
        void ActivateSteps(unsigned int steps, bool fast = true);
        std::vector<std::vector<Real> > ActivateBatch(const std::vector<std::vector<Real> > &inputs, unsigned int steps = 1, bool use_internal_bias = false);

        // Spiking simulation entry points.
        std::vector<Real> StepSpiking(const std::vector<Real> &inputs, Real time_step = -1.0);
        std::vector<std::vector<Real> > SimulateSpiking(const std::vector<std::vector<Real> > &inputs, Real time_step = -1.0, bool reset = false);
        std::vector<Real> OutputSpikes() const;
        std::vector<Real> OutputRates() const;
        std::vector<Real> OutputFilteredSpikes() const;
        std::vector<Real> OutputMembranePotentials() const;
        std::vector<Real> OutputDecoded() const;
        bool IsSpiking() const;
        Real SpikingTime() const { return m_spiking_time; }
        Real SpikingTimeStep() const { return m_spiking_time_step; }
        void SetSpikingTimeStep(Real time_step);
        void SetSpikingInputMode(SpikingInputMode mode);
        SpikingInputMode GetSpikingInputMode() const { return m_spiking_input_mode; }
        void SetSpikingOutputMode(SpikingOutputMode mode);
        SpikingOutputMode GetSpikingOutputMode() const { return m_spiking_output_mode; }
        void SeedSpiking(std::uint64_t seed);
        void EnableSpikeRecording(bool enabled, std::size_t max_events = 100000);
        const std::vector<SpikeEvent> &GetSpikeHistory() const { return m_spike_history; }
        void ClearSpikeHistory() { m_spike_history.clear(); }
        void EnableSTDP(bool enabled);
        std::size_t SparseRTRLStateSize() const;

        // accessor methods
        void AddNeuron(const Neuron &a_n) { m_neurons.push_back(a_n); }
        void AddConnection(const Connection &a_c) { m_connections.push_back(a_c); }
        Connection GetConnectionByIndex(unsigned int a_idx) const { return m_connections.at(a_idx); }
        Neuron GetNeuronByIndex(unsigned int a_idx) const { return m_neurons.at(a_idx); }
        void SetInputOutputDimentions(const unsigned int a_i, const unsigned int a_o) {
            m_num_inputs = a_i;
            m_num_outputs = a_o;
        }
        void SetInputOutputDimensions(const unsigned int inputs, const unsigned int outputs) { SetInputOutputDimentions(inputs, outputs); }
        unsigned int NumInputs() const { return m_num_inputs; }
        unsigned int NumOutputs() const { return m_num_outputs; }

        // clears the network and makes it a minimal one
        void Clear() {
            m_neurons.clear();
            m_connections.clear();
            m_total_weight_change.clear();
            m_sparse_rtrl_sensitivities.clear();
            m_spike_history.clear();
            m_spiking_time = 0.0;
            m_total_error = 0.0;
            SetInputOutputDimentions(0, 0);
        }

        // Euclidean distance between two neurons in x/y/z space.
        Real GetConnectionLength(const Neuron &source, const Neuron &target);

        Real GetTotalConnectionLength();

        // Recomputes stored physical axon lengths from neuron x/y/z. When
        // requested, length / conduction_velocity becomes each axon's delay.
        void UpdateConnectionGeometry(bool update_delays = false, Real conduction_velocity = 1.0);

        // one-shot save/load
        void Save(const char *a_filename);
        bool Load(const char *a_filename);

        // save/load from already opened files for reading/writing
        void Save(FILE *a_file);
        bool Load(std::ifstream &a_DataFile);

        std::string Serialize() const;
        static NeuralNetwork Deserialize(const std::string &data);
    };

};  // namespace NEAT
