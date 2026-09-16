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
 * File:        Substrate.h
 * Description: Definition for the Substrate class (HyperNEAT hypercube specification).
 */

#pragma once

#include <vector>

#include "NeuralNetwork.h"
#include "Types.h"

namespace NEAT {

    //-----------------------------------------------------------------------
    // The substrate describes the phenotype space that is used by HyperNEAT
    // It basically contains 3 lists of coordinates - for the nodes.
    class Substrate {
       public:
        std::vector<std::vector<Real> > m_input_coords;
        std::vector<std::vector<Real> > m_hidden_coords;
        std::vector<std::vector<Real> > m_output_coords;

        // the substrate is made from leaky integrator neurons?
        bool m_leaky;

        // the additional distance input is used?
        // NOTE: don't use it, not working yet
        bool m_with_distance;

        // these flags control the connectivity of the substrate
        bool m_allow_input_hidden_links;
        bool m_allow_input_output_links;
        bool m_allow_hidden_hidden_links;
        bool m_allow_hidden_output_links;
        bool m_allow_output_hidden_links;
        bool m_allow_output_output_links;
        bool m_allow_looped_hidden_links;
        bool m_allow_looped_output_links;

        // custom connectivity if this is not empty, the phenotype builder will use this to query all connections it's a list of [src_code, src_idx, dst_code,
        // dst_idx] where code is NeuronType (int, the enum) and idx is the index in the m_input_coords, m_hidden_coords and m_output_coords respectively
        std::vector<std::vector<int> > m_custom_connectivity;
        bool m_custom_conn_obeys_flags;  // if this is true, the flags restricting the topology above will still apply

        // this enforces custom or full connectivity if it is true, connections are always made and the weights will be queried only
        bool m_query_weights_only;

        // the activation functions of hidden/output neurons
        ActivationFunction m_hidden_nodes_activation;
        ActivationFunction m_output_nodes_activation;

        // additional parameters
        Real m_max_weight_and_bias;
        Real m_min_time_const;
        Real m_max_time_const;

        // Physical wiring budget: connections longer than this (in substrate
        // coordinate units) are pruned by ES-HyperNEAT finalization.
        // Negative disables pruning.
        Real m_max_connection_length;
        // When true, axonal delays are set from length / m_conduction_velocity.
        bool m_use_spatial_distance_for_delays;
        Real m_conduction_velocity;

        Substrate();
        Substrate(std::vector<std::vector<Real> > &a_inputs, std::vector<std::vector<Real> > &a_hidden, std::vector<std::vector<Real> > &a_outputs);

        // Sets a custom connectivity scheme The neurons must be set before calling this
        void SetCustomConnectivity(std::vector<std::vector<int> > &a_conns);

        // Clears it
        void ClearCustomConnectivity();

        int GetMaxDims() const;

        // True for three-dimensional substrates (drives octree vs quadtree).
        bool IsThreeDimensional() const { return GetMaxDims() >= 3; }

        // Return the minimum input dimensionality of the CPPN
        int GetMinCPPNInputs();
        // Return the minimum output dimensionality of the CPPN
        int GetMinCPPNOutputs();

        // Prints some info about itself
        void PrintInfo();
    };

}  // namespace NEAT
