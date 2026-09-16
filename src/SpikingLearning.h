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

/* File:        SpikingLearning.h
 * Description: Eligibility-propagation (e-prop) online learning for spiking networks: configuration, per-connection eligibility traces, feedback alignment and
 * AdamW/SGD updates.
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "NeuralNetwork.h"
#include "Types.h"

namespace NEAT {
    enum EPropOptimizer { EPROP_ADAMW = 0, EPROP_SGD };

    enum EPropFeedbackMode { EPROP_RANDOM_FEEDBACK = 0, EPROP_SYMMETRIC_FEEDBACK, EPROP_UNIFORM_FEEDBACK };

    enum EPropSurrogate { EPROP_FAST_SIGMOID = 0, EPROP_TRIANGULAR, EPROP_ARCTAN };

    enum EPropLoss { EPROP_MEAN_SQUARED_ERROR = 0, EPROP_HUBER_LOSS };

    struct EPropConfig {
        Real learning_rate = 0.001;
        EPropOptimizer optimizer = EPROP_ADAMW;
        EPropFeedbackMode feedback_mode = EPROP_RANDOM_FEEDBACK;
        EPropSurrogate surrogate = EPROP_FAST_SIGMOID;
        EPropLoss loss = EPROP_MEAN_SQUARED_ERROR;
        Real surrogate_scale = 10.0;
        Real surrogate_dampening = 0.3;
        Real gradient_clip_norm = 1.0;
        Real weight_decay = 0.0;
        Real adam_beta1 = 0.9;
        Real adam_beta2 = 0.999;
        Real adam_epsilon = 1.0e-8;
        Real huber_delta = 1.0;
        Real min_weight = -8.0;
        Real max_weight = 8.0;
        std::size_t update_interval = 1;
        std::uint64_t random_seed = UINT64_C(0x6a09e667f3bcc909);
        bool train_input_connections = true;
        bool train_hidden_connections = true;
        bool train_output_connections = true;
        bool train_recurrent_connections = true;
        bool allow_stdp = false;
    };

    struct EPropConnectionState {
        Real synaptic_trace = 0.0;
        Real voltage_eligibility = 0.0;
        Real adaptation_eligibility = 0.0;
        Real readout_eligibility = 0.0;
        Real gradient = 0.0;
        Real first_moment = 0.0;
        Real second_moment = 0.0;
    };

    struct EPropStepResult {
        std::vector<Real> outputs;
        Real loss = 0.0;
        Real gradient_norm = 0.0;
        std::size_t updated_connections = 0;
        bool update_applied = false;
    };

    struct EPropSequenceResult {
        std::vector<std::vector<Real>> outputs;
        std::vector<Real> losses;
        Real mean_loss = 0.0;
        Real final_gradient_norm = 0.0;
        std::size_t optimizer_updates = 0;
        std::size_t updated_connections = 0;
    };

    class EPropLearner {
        std::vector<EPropConnectionState> m_connection_state;
        std::vector<Real> m_feedback;
        std::vector<int> m_sources;
        std::vector<int> m_targets;
        std::vector<int> m_neuron_types;
        std::vector<int> m_activation_types;
        std::size_t m_neuron_count = 0;
        std::size_t m_input_count = 0;
        std::size_t m_output_count = 0;
        std::size_t m_accumulated_steps = 0;
        std::uint64_t m_optimizer_step = 0;
        Real m_last_gradient_norm = 0.0;
        std::size_t m_last_updated_connections = 0;

        void ValidateConfig() const;
        void ValidateTopology(const NeuralNetwork &network) const;
        bool IsTrainable(const NeuralNetwork &network, std::size_t connection_index) const;
        Real SurrogateDerivative(const Neuron &neuron) const;
        std::vector<Real> BroadcastOutputSignals(const std::vector<Real> &output_signals) const;
        void AccumulateDirectSignals(NeuralNetwork &network, const std::vector<Real> &neuron_signals, Real time_step);

       public:
        EPropConfig m_config;

        EPropLearner() = default;
        explicit EPropLearner(const EPropConfig &config) : m_config(config) {}

        void Initialize(const NeuralNetwork &network);
        void RefreshFeedback(const NeuralNetwork &network);
        bool IsInitialized() const { return m_neuron_count > 0 || !m_connection_state.empty(); }
        void ResetEligibility();
        void ResetOptimizer();
        void ZeroGradients();

        EPropStepResult TrainStep(NeuralNetwork &network, const std::vector<Real> &inputs, const std::vector<Real> &targets, Real time_step = -1.0);
        EPropStepResult TrainStepWithSignals(NeuralNetwork &network,
                                             const std::vector<Real> &inputs,
                                             const std::vector<Real> &learning_signals,
                                             Real time_step = -1.0);
        EPropSequenceResult TrainSequence(NeuralNetwork &network,
                                          const std::vector<std::vector<Real>> &inputs,
                                          const std::vector<std::vector<Real>> &targets,
                                          Real time_step = -1.0,
                                          bool reset_network = true,
                                          bool apply_final_update = true);
        void AccumulateLearningSignals(NeuralNetwork &network, const std::vector<Real> &learning_signals, Real time_step = -1.0);
        EPropStepResult ApplyGradients(NeuralNetwork &network);

        const std::vector<EPropConnectionState> &ConnectionStates() const { return m_connection_state; }
        const std::vector<Real> &FeedbackMatrix() const { return m_feedback; }
        std::uint64_t OptimizerStep() const { return m_optimizer_step; }
        std::size_t AccumulatedSteps() const { return m_accumulated_steps; }

        std::string Serialize() const;
        static EPropLearner Deserialize(const std::string &data);
    };
}  // namespace NEAT
