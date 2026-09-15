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
 * File:        Substrate.cpp
 * Description: Implementation of the Substrate class.
 */

#include "Substrate.h"

#include <vector>

#include "NeuralNetwork.h"
#include "Utils.h"

namespace NEAT {

    Substrate::Substrate() {
        leaky_ = false;
        withDistance_ = false;
        customConnObeysFlags_ = true;
        queryWeightsOnly_ = false;
        allowInputHiddenLinks_ = true;
        allowInputOutputLinks_ = true;
        allowHiddenHiddenLinks_ = false;
        allowHiddenOutputLinks_ = true;
        allowOutputHiddenLinks_ = false;
        allowOutputOutputLinks_ = false;
        allowLoopedHiddenLinks_ = false;
        allowLoopedOutputLinks_ = false;
        hiddenNodesActivation_ = UNSIGNED_SIGMOID;
        outputNodesActivation_ = UNSIGNED_SIGMOID;
        maxWeightAndBias_ = 5.0;
        minTimeConst_ = 0.1;
        maxTimeConst_ = 1.0;
    };

    Substrate::Substrate(std::vector<std::vector<Real> > &inputs, std::vector<std::vector<Real> > &hidden, std::vector<std::vector<Real> > &outputs) {
        leaky_ = false;
        withDistance_ = false;
        queryWeightsOnly_ = false;
        hiddenNodesActivation_ = NEAT::UNSIGNED_SIGMOID;
        outputNodesActivation_ = NEAT::UNSIGNED_SIGMOID;
        allowInputHiddenLinks_ = true;
        allowInputOutputLinks_ = false;
        allowHiddenHiddenLinks_ = false;
        allowHiddenOutputLinks_ = true;
        allowOutputHiddenLinks_ = false;
        allowOutputOutputLinks_ = false;
        allowLoopedHiddenLinks_ = false;
        allowLoopedOutputLinks_ = false;

        maxWeightAndBias_ = 5.0;
        minTimeConst_ = 0.1;
        maxTimeConst_ = 1.0;
        customConnObeysFlags_ = false;

        inputCoords_ = inputs;
        hiddenCoords_ = hidden;
        outputCoords_ = outputs;
    }

    void Substrate::setCustomConnectivity(std::vector<std::vector<int> > &conns) {
        for (unsigned int i = 0; i < conns.size(); i++) {
            NeuronType srcType = (NeuronType)conns[i][0];
            int srcIndex = conns[i][1];
            NeuronType dstType = (NeuronType)conns[i][2];
            int dstIndex = conns[i][3];

            std::vector<int> c;
            c.emplace_back(srcType);
            c.emplace_back(srcIndex);
            c.emplace_back(dstType);
            c.emplace_back(dstIndex);

            customConnectivity_.emplace_back(c);
        }
    }

    void Substrate::clearCustomConnectivity() { customConnectivity_.clear(); }

    int Substrate::getMinCPPNInputs() {
        // determine the dimensionality across the entire substrate
        int cppnInputs = getMaxDims() * 2;  // twice, because we query 2 points at a time

        // the distance input
        if (withDistance_) {
            cppnInputs += 1;
        }

        return cppnInputs + 1;  // always count the bias
    }

    int Substrate::getMinCPPNOutputs() {
        int outs = 0;
        if (queryWeightsOnly_) {
            outs = 1;
        } else {
            outs = 2;  // (link on/off, weight)
        }
        if (leaky_) {
            return outs + 2;  // + time_const and bias
        } else {
            return outs;
        }
    }

    int Substrate::getMaxDims() {
        unsigned int maxDims = 0;
        for (unsigned int i = 0; i < inputCoords_.size(); i++) {
            if (maxDims < inputCoords_[i].size()) {
                maxDims = inputCoords_[i].size();
            }
        }
        for (unsigned int i = 0; i < hiddenCoords_.size(); i++) {
            if (maxDims < hiddenCoords_[i].size()) {
                maxDims = hiddenCoords_[i].size();
            }
        }
        for (unsigned int i = 0; i < outputCoords_.size(); i++) {
            if (maxDims < outputCoords_[i].size()) {
                maxDims = outputCoords_[i].size();
            }
        }
        return maxDims;
    }

    void Substrate::printInfo() {
        std::cerr << "Inputs: " << inputCoords_.size() << "\n";
        std::cerr << "Hidden: " << hiddenCoords_.size() << "\n";
        std::cerr << "Outputs: " << outputCoords_.size() << "\n\n";
        std::cerr << "Dimensions: " << getMinCPPNInputs() << "\n";
    }
    // namespace NEAT

}  // namespace NEAT
