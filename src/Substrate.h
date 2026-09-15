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
 * Description: HyperNEAT substrate specification: the geometric layout of input/hidden/output-node
 *              coordinates plus the connectivity policy (layer-pair flags or an explicit custom list) used
 *              by Genome::buildHyperNEATPhenotype() to query the CPPN. Also carries the CPPN I/O
 *              dimensionality helpers and the leaky/time-constant ranges applied to substrate neurons.
 *
 * References: Stanley, D'Ambrosio & Gauci, "A Hypercube-Based Encoding for Evolving Large-Scale Neural
 *             Networks" (2009), Sections 2-4 (substrate, CPPN query, geometry); ES-HyperNEAT extensions use
 *             the quadtree knobs in src/Parameters.h. Intra-repo users: src/Genome.h, src/Genome.cpp,
 *             tests/TestSubstrate.cpp.
 */

#pragma once

#include <vector>

#include "NeuralNetwork.h"
#include "Types.h"

namespace NEAT {

    //-----------------------------------------------------------------------
    // The substrate describes the phenotype space queried by HyperNEAT: three coordinate lists whose
    // pairwise combinations become CPPN queries, filtered by the connectivity flags below.
    class Substrate {
       public:
        // Node coordinates per layer; each entry is one point in substrate space.
        std::vector<std::vector<Real> > inputCoords_;
        std::vector<std::vector<Real> > hiddenCoords_;
        std::vector<std::vector<Real> > outputCoords_;

        // Build substrate neurons as leaky integrators (uses minTimeConst_/maxTimeConst_ below).
        bool leaky_;

        // Append the Euclidean source-target distance to the CPPN query. NOTE: experimental, not working yet.
        bool withDistance_;

        // Layer-pair connectivity flags: each allows links from the first named layer to the second.
        bool allowInputHiddenLinks_;
        bool allowInputOutputLinks_;
        bool allowHiddenHiddenLinks_;
        bool allowHiddenOutputLinks_;
        bool allowOutputHiddenLinks_;
        bool allowOutputOutputLinks_;
        bool allowLoopedHiddenLinks_;
        bool allowLoopedOutputLinks_;

        // Explicit connectivity overriding the flags: rows of [sourceCode, sourceIndex, targetCode, targetIndex]
        // where code is a NeuronType value and the index addresses the matching coordinate list.
        std::vector<std::vector<int> > customConnectivity_;
        // When true, the layer-pair flags above still filter the custom list.
        bool customConnObeysFlags_;

        // When true, every allowed connection is created and the CPPN only supplies weights (no LEO gating).
        bool queryWeightsOnly_;

        // Activation functions assigned to created hidden/output neurons.
        ActivationFunction hiddenNodesActivation_;
        ActivationFunction outputNodesActivation_;

        // Weight/bias magnitude cap and leaky-integrator time-constant range for created neurons.
        Real maxWeightAndBias_;
        Real minTimeConst_;
        Real maxTimeConst_;

        Substrate();
        Substrate(std::vector<std::vector<Real> > &inputs, std::vector<std::vector<Real> > &hidden, std::vector<std::vector<Real> > &outputs);

        // Replaces the connectivity scheme; the coordinate lists must already be populated.
        void setCustomConnectivity(std::vector<std::vector<int> > &conns);

        // Clears the custom scheme, restoring flag-driven connectivity.
        void clearCustomConnectivity();

        // Maximum coordinate dimensionality across all three layers.
        int getMaxDims();

        // Minimum CPPN input dimensionality (source + target coordinates, plus bias/distance when enabled).
        int getMinCPPNInputs();
        // Minimum CPPN output dimensionality (weight, and LEO/link-expression outputs when enabled).
        int getMinCPPNOutputs();

        // Prints the layer sizes and active connectivity policy to stdout (diagnostics only).
        void printInfo();
    };

}  // namespace NEAT
