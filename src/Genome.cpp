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
 * File:        Genome.cpp
 * Description: Implementation of the Genome class.
 */

#include "Genome.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <queue>
#include <utility>

#include "AssertMacros.h"
#include "Parameters.h"
#include "Random.h"
#include "Utils.h"

namespace NEAT {

    // forward
    ActivationFunction getRandomActivation(const Parameters &parameters, RNG &rng);

    // Square helper for distance/length computations below.
    inline Real square(Real x) { return x * x; }

    // Create an empty genome
    Genome::Genome() {
        id_ = 0;
        fitness_ = 0;
        depth_ = 0;
        linkGenes_.clear();
        neuronGenes_.clear();
        numInputs_ = 0;
        numOutputs_ = 0;
        adjustedFitness_ = 0;
        offspringAmount_ = 0;
        evaluated_ = false;
        phenotypeBehavior_ = nullptr;
        initialNumNeurons_ = 0;
        initialNumLinks_ = 0;
    }

    // Copy constructor
    Genome::Genome(const Genome &g) {
        id_ = g.id_;
        depth_ = g.depth_;
        neuronGenes_ = g.neuronGenes_;
        linkGenes_ = g.linkGenes_;
        genomeGene_ = g.genomeGene_;
        fitness_ = g.fitness_;
        numInputs_ = g.numInputs_;
        numOutputs_ = g.numOutputs_;
        adjustedFitness_ = g.adjustedFitness_;
        offspringAmount_ = g.offspringAmount_;
        evaluated_ = g.evaluated_;
        phenotypeBehavior_ = g.phenotypeBehavior_;
        initialNumNeurons_ = g.initialNumNeurons_;
        initialNumLinks_ = g.initialNumLinks_;
    }

    // assignment operator
    Genome &Genome::operator=(const Genome &g) {
        // self assignment guard
        if (this != &g) {
            id_ = g.id_;
            depth_ = g.depth_;
            neuronGenes_ = g.neuronGenes_;
            linkGenes_ = g.linkGenes_;
            genomeGene_ = g.genomeGene_;
            fitness_ = g.fitness_;
            adjustedFitness_ = g.adjustedFitness_;
            numInputs_ = g.numInputs_;
            numOutputs_ = g.numOutputs_;
            offspringAmount_ = g.offspringAmount_;
            evaluated_ = g.evaluated_;
            phenotypeBehavior_ = g.phenotypeBehavior_;
            initialNumNeurons_ = g.initialNumNeurons_;
            initialNumLinks_ = g.initialNumLinks_;
        }

        return *this;
    }

    // New constructor that creates a fully-connected CTRNN
    /*
    Genome::Genome(int a_ID,
                   int a_NumInputs,
                   int a_NumHidden, // ignored for seed type == 0, specifies number of hidden units if seed type == 1
                   int a_NumOutputs, ActivationFunction a_OutputActType,
                   ActivationFunction a_HiddenActType,
                   const Parameters &a_Parameters)
    {
        ASSERT((a_NumInputs > 1) && (a_NumOutputs > 0));
        RNG t_RNG;
        t_RNG.TimeSeed();

        m_ID = a_ID;
        int t_innovnum = 1, t_nnum = 1;

        if (a_Parameters.DontUseBiasNeuron == false)
        {

            // Create the input neurons.
            // Warning! The last one is a bias!
            // The order of the neurons is very important. It is the following: INPUTS, BIAS, OUTPUTS, HIDDEN ... (no limit)
            for (unsigned int i = 0; i < (a_NumInputs - 1); i++)
            {
                NeuronGene n = NeuronGene(INPUT, t_nnum, 0.0);
                // Initialize the traits
                n.InitTraits(a_Parameters.NeuronTraits, t_RNG);
                m_NeuronGenes.emplace_back(n);
                t_nnum++;
            }
            // add the bias
            NeuronGene n = NeuronGene(BIAS, t_nnum, 0.0);
            // Initialize the traits
            n.InitTraits(a_Parameters.NeuronTraits, t_RNG);

            m_NeuronGenes.emplace_back(n);
            t_nnum++;
        }
        else
        {
            // Create the input neurons without marking the last node as bias. The order of the neurons is very important. It is the following: INPUTS, OUTPUTS,
            // HIDDEN ... (no limit)
            for (unsigned int i = 0; i < a_NumInputs; i++)
            {
                NeuronGene n = NeuronGene(INPUT, t_nnum, 0.0);
                // Initialize the traits
                n.InitTraits(a_Parameters.NeuronTraits, t_RNG);

                m_NeuronGenes.emplace_back(n);
                t_nnum++;
            }
        }

        // now the outputs
        for (unsigned int i = 0; i < (a_NumOutputs); i++)
        {
            NeuronGene t_ngene(OUTPUT, t_nnum, 1.0);
            // Initialize the neuron gene's properties
            t_ngene.Init((a_Parameters.MinActivationA + a_Parameters.MaxActivationA) / 2.0f,
                         (a_Parameters.MinActivationB + a_Parameters.MaxActivationB) / 2.0f,
                         (a_Parameters.MinNeuronTimeConstant + a_Parameters.MaxNeuronTimeConstant) / 2.0f,
                         (a_Parameters.MinNeuronBias + a_Parameters.MaxNeuronBias) / 2.0f,
                         a_OutputActType);
            // Initialize the traits
            t_ngene.InitTraits(a_Parameters.NeuronTraits, t_RNG);

            m_NeuronGenes.emplace_back(t_ngene);
            t_nnum++;
        }

        for (unsigned int i = 0; i < a_NumHidden; i++)
        {
            NeuronGene t_ngene(HIDDEN, t_nnum, 1.0);
            // Initialize the neuron gene's properties
            t_ngene.Init((a_Parameters.MinActivationA + a_Parameters.MaxActivationA) / 2.0f,
                         (a_Parameters.MinActivationB + a_Parameters.MaxActivationB) / 2.0f,
                         (a_Parameters.MinNeuronTimeConstant + a_Parameters.MaxNeuronTimeConstant) / 2.0f,
                         (a_Parameters.MinNeuronBias + a_Parameters.MaxNeuronBias) / 2.0f,
                         a_HiddenActType);
            // Initialize the traits
            t_ngene.InitTraits(a_Parameters.NeuronTraits, t_RNG);
            t_ngene.m_SplitY = 0.5;

            m_NeuronGenes.emplace_back(t_ngene);
            t_nnum++;
        }

        // Fully connect every neuron to every other. Only inputs don't receive output.
        for (unsigned int i = a_NumInputs; i < (a_NumInputs+a_NumOutputs+a_NumHidden); i++)
        {
            for (unsigned int j = 0; j < (a_NumInputs+a_NumOutputs+a_NumHidden); j++)
            {
                // add the link created with zero weights. needs future random initialization. !!!!!!!!
                LinkGene l = LinkGene(j + 1, i + 1, t_innovnum, 0.0, false);
                l.InitTraits(a_Parameters.LinkTraits, t_RNG);
                m_LinkGenes.emplace_back(l);
                t_innovnum++;
            }
        }

        // Also initialize the Genome's traits
        m_GenomeGene.InitTraits(a_Parameters.GenomeTraits, t_RNG);

        m_Evaluated = false;
        m_NumInputs = a_NumInputs;
        m_NumOutputs = a_NumOutputs;
        m_Fitness = 0.0;
        m_AdjustedFitness = 0.0;
        m_OffspringAmount = 0.0;
        m_Depth = 0;
        m_PhenotypeBehavior = NULL;

        m_initial_num_neurons = NumNeurons();
        m_initial_num_links = NumLinks();
    }*/

    Genome::Genome(const Parameters &parameters, const GenomeInitStruct &in) {
        ASSERT((in.numInputs > 1) && (in.numOutputs > 0));
        RNG rng;
        rng.timeSeed();

        id_ = 0;
        int innovnum = 1, nnum = 1;
        GenomeSeedType seedType = in.seedType;

        // override seed_type if 0 hidden units are specified
        if ((seedType == LAYERED) && (in.numHidden == 0)) {
            seedType = PERCEPTRON;
        }

        if (parameters.dontUseBiasNeuron == false) {
            // Create the input neurons.
            // Warning! The last one is a bias!
            // The order of the neurons is very important. It is the following: INPUTS, BIAS, OUTPUTS, HIDDEN ... (no limit)
            for (unsigned int i = 0; i < (in.numInputs - 1); i++) {
                NeuronGene n = NeuronGene(INPUT, nnum, 0.0);
                // Initialize the traits
                neuronGenes_.emplace_back(n);
                nnum++;
            }
            // add the bias
            NeuronGene n = NeuronGene(BIAS, nnum, 0.0);
            // Initialize the traits

            neuronGenes_.emplace_back(n);
            nnum++;
        } else {
            // Create the input neurons without marking the last node as bias. The order of the neurons is very important. It is the following: INPUTS, OUTPUTS,
            // HIDDEN ... (no limit)
            for (unsigned int i = 0; i < in.numInputs; i++) {
                NeuronGene n = NeuronGene(INPUT, nnum, 0.0);
                // Initialize the traits

                neuronGenes_.emplace_back(n);
                nnum++;
            }
        }

        // now the outputs
        for (unsigned int i = 0; i < (in.numOutputs); i++) {
            NeuronGene ngene(OUTPUT, nnum, 1.0);
            // Initialize the neuron gene's properties
            ngene.init((parameters.minActivationA + parameters.maxActivationA) / 2.0f, (parameters.minActivationB + parameters.maxActivationB) / 2.0f,
                       (parameters.minNeuronTimeConstant + parameters.maxNeuronTimeConstant) / 2.0f,
                       (parameters.minNeuronBias + parameters.maxNeuronBias) / 2.0f, in.outputActType);
            // Initialize the traits
            ngene.initTraits(parameters.neuronTraits, rng);

            neuronGenes_.emplace_back(ngene);
            nnum++;
        }

        // Now add LEO
        /*if (a_Parameters.Leo)
        {
            NeuronGene t_ngene(OUTPUT, t_nnum, 1.0);
            // Initialize the neuron gene's properties
            t_ngene.Init((a_Parameters.MinActivationA + a_Parameters.MaxActivationA) / 2.0f,
                         (a_Parameters.MinActivationB + a_Parameters.MaxActivationB) / 2.0f,
                         (a_Parameters.MinNeuronTimeConstant + a_Parameters.MaxNeuronTimeConstant) / 2.0f,
                         (a_Parameters.MinNeuronBias + a_Parameters.MaxNeuronBias) / 2.0f,
                         UNSIGNED_STEP);
            // Initialize the traits
            t_ngene.InitTraits(a_Parameters.NeuronTraits, t_RNG);

            m_NeuronGenes.emplace_back(t_ngene);
            t_nnum++;
            in.NumOutputs++;
        }*/

        // add and connect hidden neurons if seed type is != 0
        if ((in.seedType == LAYERED) && (in.numHidden > 0)) {
            Real ltInc = 1.0 / (in.numLayers + 1);
            Real initlt = ltInc;
            for (unsigned int n = 0; n < in.numLayers; n++) {
                for (unsigned int i = 0; i < in.numHidden; i++) {
                    NeuronGene ngene(HIDDEN, nnum, 1.0);
                    // Initialize the neuron gene's properties
                    ngene.init((parameters.minActivationA + parameters.maxActivationA) / 2.0f, (parameters.minActivationB + parameters.maxActivationB) / 2.0f,
                               (parameters.minNeuronTimeConstant + parameters.maxNeuronTimeConstant) / 2.0f,
                               (parameters.minNeuronBias + parameters.maxNeuronBias) / 2.0f, in.hiddenActType);
                    // Initialize the traits
                    ngene.initTraits(parameters.neuronTraits, rng);
                    ngene.splitY_ = initlt;

                    neuronGenes_.emplace_back(ngene);
                    nnum++;
                }

                initlt += ltInc;
            }

            if (!in.fsNeat) {
                int lastDestId = in.numInputs + in.numOutputs + 1;
                int lastSrcId = 1;
                int prevLayerSize = in.numInputs;

                for (unsigned int n = 0; n < in.numLayers; n++) {
                    // The links from each previous layer to this hidden node
                    for (unsigned int i = 0; i < in.numHidden; i++) {
                        for (unsigned int j = 0; j < prevLayerSize; j++) {
                            // add the link created with zero weights. needs future random initialization. !!!!!!!! init traits (TODO: maybe init empty traits?)
                            LinkGene l = LinkGene(j + lastSrcId, i + lastDestId, innovnum, 0.0, false);
                            l.initTraits(parameters.linkTraits, rng);
                            linkGenes_.emplace_back(l);
                            innovnum++;
                        }
                    }

                    lastDestId += in.numHidden;
                    if (n == 0) {
                        // for the first hidden layer, jump over the outputs too
                        lastSrcId += prevLayerSize + in.numOutputs;
                    } else {
                        lastSrcId += prevLayerSize;
                    }
                    prevLayerSize = in.numHidden;
                }

                lastDestId = in.numInputs + 1;

                // The links from each previous layer to this output node
                for (unsigned int i = 0; i < in.numOutputs; i++) {
                    for (unsigned int j = 0; j < prevLayerSize; j++) {
                        // add the link created with zero weights. needs future random initialization. !!!!!!!! init traits (TODO: maybe init empty traits?)
                        LinkGene l = LinkGene(j + lastSrcId, i + lastDestId, innovnum, 0.0, false);
                        l.initTraits(parameters.linkTraits, rng);
                        linkGenes_.emplace_back(l);
                        innovnum++;
                    }
                }

                /*if (a_Parameters.DontUseBiasNeuron == false)
                {
                    // Connect the bias as well
                    for (unsigned int i = 0; i < a_NumOutputs; i++)
                    {
                        // add the link created with zero weights. needs future random initialization. !!!!!!!!
                        LinkGene l = LinkGene(a_NumInputs, i + last_dest_id, t_innovnum, 0.0, false);
                        l.InitTraits(a_Parameters.LinkTraits, t_RNG);
                        m_LinkGenes.emplace_back(l);
                        t_innovnum++;
                    }
                }*/
            }
        } else  // The links connecting every input to every output - perceptron structure
        {
            if ((!in.fsNeat) && (seedType == PERCEPTRON)) {
                for (unsigned int i = 0; i < (in.numOutputs); i++) {
                    for (unsigned int j = 0; j < in.numInputs; j++) {
                        // add the link created with zero weights. needs future random initialization. !!!!!!!!
                        LinkGene l = LinkGene(j + 1, i + in.numInputs + 1, innovnum, 0.0, false);
                        l.initTraits(parameters.linkTraits, rng);
                        linkGenes_.emplace_back(l);
                        innovnum++;
                    }
                }
            } else {
                // Start very minimally - connect a random input to each output
                // Also connect the bias to every output

                std::vector<std::pair<int, int>> madeAlready;
                bool there = false;
                int linksmade = 0;

                // do this a few times for more initial links created
                // TODO: make sure the innovations don't repeat for the same input/output pairs
                while (linksmade < in.fsNeatLinks) {
                    for (unsigned int i = 0; i < in.numOutputs; i++) {
                        int inpId = rng.randInt(1, in.numInputs - 1);
                        int biasId = in.numInputs;
                        int outpId = in.numInputs + 1 + i;

                        // check if there already
                        there = false;
                        for (std::vector<std::pair<int, int>>::iterator it = madeAlready.begin(); it != madeAlready.end(); it++) {
                            if ((it->first == inpId) && (it->second == outpId)) {
                                there = true;
                                break;
                            }
                        }

                        if (!there) {
                            // created with zero weights. needs future random initialization. !!!!!!!!
                            LinkGene l = LinkGene(inpId, outpId, innovnum, 0.0, false);
                            l.initTraits(parameters.linkTraits, rng);
                            linkGenes_.emplace_back(l);
                            innovnum++;

                            if (parameters.dontUseBiasNeuron == false) {
                                LinkGene bl = LinkGene(biasId, outpId, innovnum, 0.0, false);
                                bl.initTraits(parameters.linkTraits, rng);
                                linkGenes_.emplace_back(bl);
                                innovnum++;
                            }

                            linksmade++;
                            madeAlready.emplace_back(inpId, outpId);
                        }
                    }
                }
            }
        }

        if (in.fsNeat && (in.fsNeatLinks == 1)) {
            throw std::runtime_error("Known bug - don't use FS-NEAT with just 1 link and 1/1/1 genome");
        }

        // Also initialize the Genome's traits
        genomeGene_.initTraits(parameters.genomeTraits, rng);

        evaluated_ = false;
        numInputs_ = in.numInputs;
        numOutputs_ = in.numOutputs;
        fitness_ = 0.0;
        adjustedFitness_ = 0.0;
        offspringAmount_ = 0.0;
        depth_ = 0;
        phenotypeBehavior_ = nullptr;

        initialNumNeurons_ = numNeurons();
        initialNumLinks_ = numLinks();
    }

    void Genome::setDepth(unsigned int d) { depth_ = d; }

    unsigned int Genome::getDepth() const { return depth_; }

    void Genome::setID(int id) { id_ = id; }

    int Genome::getID() const { return id_; }

    void Genome::setAdjFitness(Real af) { adjustedFitness_ = af; }

    void Genome::setFitness(Real f) { fitness_ = f; }

    Real Genome::getAdjFitness() const { return adjustedFitness_; }

    Real Genome::getFitness() const { return fitness_; }

    void Genome::setNeuronY(unsigned int index, int y) {
        ASSERT(index < neuronGenes_.size());
        neuronGenes_[index].y = y;
    }

    void Genome::setNeuronX(unsigned int index, int x) {
        ASSERT(index < neuronGenes_.size());
        neuronGenes_[index].x = x;
    }

    void Genome::setNeuronXY(unsigned int index, int x, int y) {
        ASSERT(index < neuronGenes_.size());
        neuronGenes_[index].x = x;
        neuronGenes_[index].y = y;
    }

    LinkGene Genome::getLinkByIndex(int index) const {
        ASSERT(index < linkGenes_.size());
        return linkGenes_[index];
    }

    LinkGene Genome::getLinkByInnovID(int id) const {
        ASSERT(hasLinkByInnovID(id));
        for (unsigned int i = 0; i < linkGenes_.size(); i++)
            if (linkGenes_[i].innovationID() == id) return linkGenes_[i];

        // should never reach this code
        throw std::runtime_error("operation failed");
    }

    NeuronGene Genome::getNeuronByIndex(int index) const {
        ASSERT(index < neuronGenes_.size());
        return neuronGenes_[index];
    }

    NeuronGene Genome::getNeuronByID(int id) const {
        ASSERT(hasNeuronID(id));
        int index = getNeuronIndex(id);
        ASSERT(index != -1);
        return neuronGenes_[index];
    }

    Real Genome::getOffspringAmount() const { return offspringAmount_; }

    void Genome::setOffspringAmount(Real oa) { offspringAmount_ = oa; }

    bool Genome::isEvaluated() const { return evaluated_; }

    void Genome::setEvaluated() { evaluated_ = true; }

    void Genome::resetEvaluated() { evaluated_ = false; }

    // A little helper function to find the index of a neuron, given its ID
    // returns -1 if not found
    int Genome::getNeuronIndex(int id) const {
        ASSERT(id > 0);

        for (unsigned int i = 0; i < numNeurons(); i++) {
            if (neuronGenes_[i].id() == id) {
                return i;
            }
        }

        return -1;
    }

    // A little helper function to find the index of a link, given its innovation ID
    // returns -1 if not found
    int Genome::getLinkIndex(int innovID) const {
        ASSERT(innovID > 0);
        ASSERT(numLinks() > 0);

        for (unsigned int i = 0; i < numLinks(); i++) {
            if (linkGenes_[i].innovationID() == innovID) {
                return i;
            }
        }

        return -1;
    }

    // returns the max neuron ID
    int Genome::getLastNeuronID() const {
        ASSERT(numNeurons() > 0);

        int maxid = 0;

        for (unsigned int i = 0; i < numNeurons(); i++) {
            if (neuronGenes_[i].id() > maxid) maxid = neuronGenes_[i].id();
        }

        return maxid + 1;
    }

    // returns the max innovation Id
    int Genome::getLastInnovationID() const {
        ASSERT(numLinks() > 0);

        int maxid = 0;

        for (unsigned int i = 0; i < numLinks(); i++) {
            if (linkGenes_[i].innovationID() > maxid) maxid = linkGenes_[i].innovationID();
        }

        return maxid + 1;
    }

    // Returns true if the specified neuron ID is present in the genome
    bool Genome::hasNeuronID(int id) const {
        ASSERT(id > 0);
        ASSERT(numNeurons() > 0);

        for (unsigned int i = 0; i < numNeurons(); i++) {
            if (neuronGenes_[i].id() == id) {
                return true;
            }
        }

        return false;
    }

    // Returns true if the specified link is present in the genome
    bool Genome::hasLink(int n1id, int n2id) const {
        ASSERT((n1id > 0) && (n2id > 0));

        for (unsigned int i = 0; i < numLinks(); i++) {
            if ((linkGenes_[i].fromNeuronID() == n1id) && (linkGenes_[i].toNeuronID() == n2id)) {
                return true;
            }
        }

        return false;
    }

    bool Genome::failsConstraints(const Parameters &parameters) {
        if (hasDeadEnds() || (numLinks() == 0)) {
            return true;  // no reason to continue
        }

        if ((hasLoops() && (parameters.allowLoops == false))) {
            return true;
        }

        if (parameters.customConstraints != nullptr) {
            if (parameters.customConstraints(*this)) {
                return true;
            }
        }

        return false;
    }

    bool Genome::hasLoops() {
        NeuralNetwork net;
        buildPhenotype(net);
        // Detect directed cycles using Kahn's algorithm (indegree-based
        // topological sort). If every node cannot be processed, a cycle exists.
        const int n = static_cast<int>(net.neurons_.size());
        const int e = static_cast<int>(net.connections_.size());
        std::vector<int> indegree(n, 0);
        // Flat out-adjacency (head/next chains) so each edge is visited once
        // overall instead of once per popped node (was O(nodes x edges)); two
        // vector allocations regardless of graph size.
        std::vector<int> head(n, -1);
        std::vector<int> next(e, -1);

        for (int i = 0; i < e; i++) {
            int src = net.connections_[i].sourceNeuronIndex_;
            int tgt = net.connections_[i].targetNeuronIndex_;
            if ((tgt >= 0) && (tgt < n)) {
                indegree[tgt]++;
                if ((src >= 0) && (src < n)) {
                    next[i] = head[src];
                    head[src] = i;
                }
            }
        }

        std::vector<int> stack;
        for (int i = 0; i < n; i++) {
            if (indegree[i] == 0) {
                stack.push_back(i);
            }
        }

        int visited = 0;
        while (!stack.empty()) {
            int node = stack.back();
            stack.pop_back();
            visited++;
            for (int k = head[node]; k != -1; k = next[k]) {
                int tgt = net.connections_[k].targetNeuronIndex_;
                indegree[tgt]--;
                if (indegree[tgt] == 0) {
                    stack.push_back(tgt);
                }
            }
        }

        // A self-loop on a single node (src == tgt) is also a cycle
        bool selfLoop = false;
        for (int i = 0; i < net.connections_.size(); i++) {
            if (net.connections_[i].sourceNeuronIndex_ == net.connections_[i].targetNeuronIndex_) {
                selfLoop = true;
                break;
            }
        }

        return (visited < n) || selfLoop;
    }

    // Returns true if the specified link is present in the genome
    bool Genome::hasLinkByInnovID(int id) const {
        ASSERT(id > 0);

        for (unsigned int i = 0; i < numLinks(); i++) {
            if (linkGenes_[i].innovationID() == id) {
                return true;
            }
        }

        return false;
    }

    // This builds a fastnetwork structure out from the genome
    void Genome::buildPhenotype(NeuralNetwork &net) {
        // first clear out the network
        net.clear();
        net.setInputOutputDimensions(numInputs_, numOutputs_);

        // Build an ID->index table once so the connection loop below resolves
        // endpoints in O(1); the previous GetNeuronIndex() rescans made this
        // O(links x neurons).
        int maxId = 0;
        for (unsigned int i = 0; i < numNeurons(); i++) {
            if (neuronGenes_[i].id() > maxId) {
                maxId = neuronGenes_[i].id();
            }
        }
        std::vector<int> idToIndex(static_cast<size_t>(maxId) + 1, -1);

        // Fill the net with the neurons
        for (unsigned int i = 0; i < numNeurons(); i++) {
            Neuron n;

            idToIndex[static_cast<size_t>(neuronGenes_[i].id())] = static_cast<int>(i);

            n.a_ = neuronGenes_[i].a_;
            n.b_ = neuronGenes_[i].b_;
            n.timeconst_ = neuronGenes_[i].timeConstant_;
            n.bias_ = neuronGenes_[i].bias_;
            n.activationFunctionType_ = neuronGenes_[i].actFunction_;
            n.splitY_ = neuronGenes_[i].splitY();
            n.type_ = neuronGenes_[i].type();

            net.addNeuron(n);
        }

        // Fill the net with the connections
        for (unsigned int i = 0; i < numLinks(); i++) {
            Connection c;

            const int from = linkGenes_[i].fromNeuronID();
            const int to = linkGenes_[i].toNeuronID();
            c.sourceNeuronIndex_ = (from >= 0 && from <= maxId) ? idToIndex[static_cast<size_t>(from)] : getNeuronIndex(from);
            c.targetNeuronIndex_ = (to >= 0 && to <= maxId) ? idToIndex[static_cast<size_t>(to)] : getNeuronIndex(to);
            c.weight_ = linkGenes_[i].getWeight();
            c.recurFlag_ = linkGenes_[i].isRecurrent();

            //////////////////////
            // default values
            c.hebbRate_ = 0.3;
            c.hebbPreRate_ = 0.1;

            // if a float trait "hebb_rate" exists
            if (linkGenes_[i].traits_.count("hebb_rate") == 1) {
                try {
                    c.hebbRate_ = std::get<Real>(linkGenes_[i].traits_["hebb_rate"].value);
                } catch (std::exception e) {
                    // do nothing
                }
            }
            // if a float trait "hebb_pre_rate" exists
            if (linkGenes_[i].traits_.count("hebb_pre_rate") == 1) {
                try {
                    c.hebbPreRate_ = std::get<Real>(linkGenes_[i].traits_["hebb_pre_rate"].value);
                } catch (std::exception e) {
                    // do nothing
                }
            }

            //////////////////////

            net.addConnection(c);
        }

        net.flush();

        // Note however that the RTRL variables are not initialized.
        // The user must manually call the InitRTRLMatrix() method to do it.
        // This is because of storage issues. RTRL need not to be used every time.
    }

    // Builds a HyperNEAT phenotype based on the substrate The CPPN input dimensionality must match the largest number of dimensions in the substrate The output
    // dimensionality is determined according to flags set in the substrate

    // The procedure uses the [0] CPPN output for creating nodes, and if the substrate is leaky, [1] and [2] for time constants and biases Also assumes the CPPN
    // uses signed activation outputs
    void Genome::buildHyperNEATPhenotype(NeuralNetwork &net, Substrate &subst) {
        // We need a substrate with at least one input and output
        ASSERT(subst.inputCoords_.size() > 0);
        ASSERT(subst.outputCoords_.size() > 0);

        int maxDims = subst.getMaxDims();

        // Make sure the CPPN dimensionality is right
        ASSERT(subst.getMinCPPNInputs() > 0);
        ASSERT(numInputs() >= subst.getMinCPPNInputs());
        ASSERT(numOutputs() >= subst.getMinCPPNOutputs());
        if (subst.leaky_) {
            ASSERT(numOutputs() >= subst.getMinCPPNOutputs());
        }

        // Now we create the substrate (net)
        net.setInputOutputDimensions(static_cast<unsigned short>(subst.inputCoords_.size()), static_cast<unsigned short>(subst.outputCoords_.size()));

        // Inputs
        for (unsigned int i = 0; i < subst.inputCoords_.size(); i++) {
            Neuron n;

            n.a_ = 1;
            n.b_ = 0;
            n.substrateCoords_ = subst.inputCoords_[i];
            ASSERT(n.substrateCoords_.size() > 0);  // prevent 0D points
            n.activationFunctionType_ = NEAT::LINEAR;
            n.type_ = NEAT::INPUT;

            net.addNeuron(n);
        }

        // Output
        for (unsigned int i = 0; i < subst.outputCoords_.size(); i++) {
            Neuron n;

            n.a_ = 1;
            n.b_ = 0;
            n.substrateCoords_ = subst.outputCoords_[i];
            ASSERT(n.substrateCoords_.size() > 0);  // prevent 0D points
            n.activationFunctionType_ = subst.outputNodesActivation_;
            n.type_ = NEAT::OUTPUT;

            net.addNeuron(n);
        }

        // Hidden
        for (unsigned int i = 0; i < subst.hiddenCoords_.size(); i++) {
            Neuron n;

            n.a_ = 1;
            n.b_ = 0;
            n.substrateCoords_ = subst.hiddenCoords_[i];
            ASSERT(n.substrateCoords_.size() > 0);  // prevent 0D points
            n.activationFunctionType_ = subst.hiddenNodesActivation_;
            n.type_ = NEAT::HIDDEN;

            net.addNeuron(n);
        }

        // Begin querying the CPPN Create the neural network that will represent the CPPN
        NeuralNetwork tempPhenotype(true);
        buildPhenotype(tempPhenotype);
        tempPhenotype.flush();

        // To ensure network relaxation
        int dp = 8;
        if (!hasLoops()) {
            calculateDepth();
            dp = getDepth();
        }

        // now loop over every potential connection in the substrate and take its weight

        // For leaky substrates, first loop over the neurons and set their properties
        if (subst.leaky_) {
            for (unsigned int i = net.numInputs(); i < net.neurons_.size(); i++) {
                // neuron specific stuff
                tempPhenotype.flush();

                // Inputs for the generation of time consts and biases across the nodes in the substrate We input only the position of the first node and ignore
                // the other one
                std::vector<Real> inputs;
                inputs.resize(numInputs());

                for (unsigned int n = 0; n < net.neurons_[i].substrateCoords_.size(); n++) {
                    inputs[n] = net.neurons_[i].substrateCoords_[n];
                }

                if (subst.withDistance_) {
                    // compute the Eucledian distance between the point and the origin
                    Real sum = 0;
                    for (int n = 0; n < maxDims; n++) {
                        sum += square(inputs[n]);
                    }
                    sum = std::sqrt(sum);
                    inputs[numInputs() - 2] = sum;
                }
                inputs[numInputs() - 1] = 1.0;  // the CPPN's bias

                tempPhenotype.input(inputs);

                // activate as many times as deep
                for (int d = 0; d < dp; d++) {
                    tempPhenotype.activate();
                }

                Real tc = tempPhenotype.output()[numOutputs() - 2];
                Real bias = tempPhenotype.output()[numOutputs() - 1];

                clamp(tc, -1, 1);
                clamp(bias, -1, 1);

                // rescale the values
                scale(tc, -1, 1, subst.minTimeConst_, subst.maxTimeConst_);
                scale(bias, -1, 1, -subst.maxWeightAndBias_, subst.maxWeightAndBias_);

                net.neurons_[i].timeconst_ = tc;
                net.neurons_[i].bias_ = bias;
            }
        }

        // list of src_idx, dst_idx pairs of all connections to query
        std::vector<std::vector<int>> toQuery;

        // There isn't custom connectiviy scheme?
        if (subst.customConnectivity_.empty()) {
            // only incoming connections, so loop only the hidden and output neurons
            for (int i = net.numInputs(); i < net.neurons_.size(); i++) {
                // loop all neurons
                for (int j = 0; j < net.neurons_.size(); j++) {
                    // this is connection "j" to "i"

                    // conditions for canceling the CPPN query
                    if (((!subst.allowInputHiddenLinks_) && ((net.neurons_[j].type_ == INPUT) && (net.neurons_[i].type_ == HIDDEN)))

                        || ((!subst.allowInputOutputLinks_) && ((net.neurons_[j].type_ == INPUT) && (net.neurons_[i].type_ == OUTPUT)))

                        || ((!subst.allowHiddenHiddenLinks_) && ((net.neurons_[j].type_ == HIDDEN) && (net.neurons_[i].type_ == HIDDEN) && (i != j)))

                        || ((!subst.allowHiddenOutputLinks_) && ((net.neurons_[j].type_ == HIDDEN) && (net.neurons_[i].type_ == OUTPUT)))

                        || ((!subst.allowOutputHiddenLinks_) && ((net.neurons_[j].type_ == OUTPUT) && (net.neurons_[i].type_ == HIDDEN)))

                        || ((!subst.allowOutputOutputLinks_) && ((net.neurons_[j].type_ == OUTPUT) && (net.neurons_[i].type_ == OUTPUT) && (i != j)))

                        || ((!subst.allowLoopedHiddenLinks_) && ((net.neurons_[j].type_ == HIDDEN) && (net.neurons_[i].type_ == HIDDEN) && (i == j)))

                        || ((!subst.allowLoopedOutputLinks_) && ((net.neurons_[j].type_ == OUTPUT) && (net.neurons_[i].type_ == OUTPUT) && (i == j)))

                    ) {
                        continue;
                    }

                    // Save potential link to query
                    std::vector<int> link;
                    link.emplace_back(j);
                    link.emplace_back(i);
                    toQuery.emplace_back(link);
                }
            }
        } else {
            // use the custom connectivity
            for (unsigned int idx = 0; idx < subst.customConnectivity_.size(); idx++) {
                NeuronType srcType = (NeuronType)subst.customConnectivity_[idx][0];
                int srcIndex = subst.customConnectivity_[idx][1];
                NeuronType dstType = (NeuronType)subst.customConnectivity_[idx][2];
                int dstIndex = subst.customConnectivity_[idx][3];

                // determine the indices in the NN
                int j = 0;  // src
                int i = 0;  // dst

                if ((srcType == INPUT) || (srcType == BIAS)) {
                    j = srcIndex;
                } else if (srcType == HIDDEN) {
                    j = subst.inputCoords_.size() + subst.outputCoords_.size() + srcIndex;
                } else if (srcType == OUTPUT) {
                    j = subst.inputCoords_.size() + srcIndex;
                }

                if ((dstType == INPUT) || (dstType == BIAS)) {
                    i = dstIndex;
                } else if (dstType == HIDDEN) {
                    i = subst.inputCoords_.size() + subst.outputCoords_.size() + dstIndex;
                } else if (dstType == OUTPUT) {
                    i = subst.inputCoords_.size() + dstIndex;
                }

                // conditions for canceling the CPPN query
                if (subst.customConnObeysFlags_ &&
                    (((!subst.allowInputHiddenLinks_) && ((net.neurons_[j].type_ == INPUT) && (net.neurons_[i].type_ == HIDDEN)))

                     || ((!subst.allowInputOutputLinks_) && ((net.neurons_[j].type_ == INPUT) && (net.neurons_[i].type_ == OUTPUT)))

                     || ((!subst.allowHiddenHiddenLinks_) && ((net.neurons_[j].type_ == HIDDEN) && (net.neurons_[i].type_ == HIDDEN) && (i != j)))

                     || ((!subst.allowHiddenOutputLinks_) && ((net.neurons_[j].type_ == HIDDEN) && (net.neurons_[i].type_ == OUTPUT)))

                     || ((!subst.allowOutputHiddenLinks_) && ((net.neurons_[j].type_ == OUTPUT) && (net.neurons_[i].type_ == HIDDEN)))

                     || ((!subst.allowOutputOutputLinks_) && ((net.neurons_[j].type_ == OUTPUT) && (net.neurons_[i].type_ == OUTPUT) && (i != j)))

                     || ((!subst.allowLoopedHiddenLinks_) && ((net.neurons_[j].type_ == HIDDEN) && (net.neurons_[i].type_ == HIDDEN) && (i == j)))

                     || ((!subst.allowLoopedOutputLinks_) && ((net.neurons_[j].type_ == OUTPUT) && (net.neurons_[i].type_ == OUTPUT) && (i == j))))) {
                    continue;
                }

                // Save potential link to query
                std::vector<int> link;
                link.emplace_back(j);
                link.emplace_back(i);
                toQuery.emplace_back(link);
            }
        }

        // Query and create all links
        for (unsigned int conn = 0; conn < toQuery.size(); conn++) {
            int j = toQuery[conn][0];
            int i = toQuery[conn][1];

            // Take the weight of this connection by querying the CPPN
            // as many times as deep (recurrent or looped CPPNs may be very slow!!!*)
            std::vector<Real> inputs;
            inputs.resize(numInputs());

            int fromDims = net.neurons_[j].substrateCoords_.size();
            int toDims = net.neurons_[i].substrateCoords_.size();

            // input the node positions to the CPPN from
            for (int n = 0; n < fromDims; n++) {
                inputs[n] = net.neurons_[j].substrateCoords_[n];
            }
            // to
            for (int n = 0; n < toDims; n++) {
                inputs[maxDims + n] = net.neurons_[i].substrateCoords_[n];
            }

            // the input is like
            // x000|xx00|1 - 1D -> 2D connection
            // xx00|xx00|1 - 2D -> 2D connection
            // xx00|xxx0|1 - 2D -> 3D connection
            // if max_dims is 4 and no distance input

            if (subst.withDistance_) {
                // compute the Eucledian distance between the two points differing dimensionality doesn't matter as the extra dimensions are 0s
                Real sum = 0;
                for (int n = 0; n < maxDims; n++) {
                    sum += square(inputs[n] - inputs[maxDims + n]);
                }
                sum = std::sqrt(sum);

                inputs[numInputs() - 2] = sum;
            }

            inputs[numInputs() - 1] = 1.0;

            // flush between each query
            tempPhenotype.flush();
            tempPhenotype.input(inputs);

            // activate as many times as deep
            for (int d = 0; d < dp; d++) {
                tempPhenotype.activate();
            }

            // the output is a weight
            Real link = 0;
            Real weight = 0;

            if (subst.queryWeightsOnly_) {
                weight = tempPhenotype.output()[0];
            } else {
                link = tempPhenotype.output()[0];
                weight = tempPhenotype.output()[1];
            }

            if (((link > 0) && (!subst.queryWeightsOnly_)) || (subst.queryWeightsOnly_)) {
                // now this weight will be scaled
                weight *= subst.maxWeightAndBias_;

                // build the connection
                Connection c;

                c.sourceNeuronIndex_ = j;
                c.targetNeuronIndex_ = i;
                c.weight_ = weight;
                c.recurFlag_ = false;

                net.addConnection(c);
            }
        }
    }

    // Projects the weight changes of a phenotype back to the genome.
    // WARNING! Using this too often in conjuction with RTRL can confuse evolution.
    void Genome::derivePhenotypicChanges(NeuralNetwork &net) {
        // the a_Net and the genome must have identical topology. if the topology differs, no changes will be made to the genome

        // Since we don't have a comparison operator yet, we are going to assume
        // identical topolgy
        // TODO: create that comparison operator for NeuralNetworks

        // Iterate through the links and replace weights
        for (unsigned int i = 0; i < numLinks(); i++) {
            linkGenes_[i].setWeight(net.getConnectionByIndex(i).weight_);
        }

        // TODO: if neuron parameters were changed, derive them
        // * in future expansions
    }

    // std::map<std::pair<int,int>, Real> distance_cache;

    // Returns the absolute distance between this genome and a_G
    Real Genome::compatibilityDistance(Genome &g, Parameters &parameters) {
        // first check if in cache, if so, return that

        // New - if there is a behavior in the genomes, return their distance

        // iterators for moving through the genomes' genes
        std::vector<LinkGene>::iterator g1;
        std::vector<LinkGene>::iterator g2;

        // this variable is the total distance between the genomes if it passes beyond the compatibility treshold, the function returns false
        Real totalDistance = 0.0;

        Real totalWeightDifference = 0.0;
        Real totalTimeconstantDifference = 0.0;
        Real totalBiasDifference = 0.0;
        Real totalADifference = 0.0;
        Real totalBDifference = 0.0;
        Real totalNumActivationDifference = 0.0;
        std::map<std::string, Real> totalNeuronTraitDifference;
        std::map<std::string, Real> totalLinkTraitDifference;
        std::map<std::string, Real> genomeLinkTraitDifference;

        // count of matching genes
        Real numExcess = 0;
        Real numDisjoint = 0;
        Real numMatchingLinks = 0;
        Real numMatchingNeurons = 0;

        // calculate genome trait difference here
        genomeLinkTraitDifference = genomeGene_.getTraitDistances(g.genomeGene_.traits_);

        // used for percentage of excess/disjoint genes calculation
        int maxGenomeSize = static_cast<int>(numLinks() < g.numLinks()) ? (g.numLinks()) : (numLinks());
        int maxNeurons = static_cast<int>(numNeurons() < g.numNeurons()) ? (g.numNeurons()) : (numNeurons());

        g1 = linkGenes_.begin();
        g2 = g.linkGenes_.begin();

        // Step through the genes until both genomes end
        while (!((g1 == linkGenes_.end()) && ((g2 == g.linkGenes_.end())))) {
            // end of first genome?
            if (g1 == linkGenes_.end()) {
                // add to the total distance
                numExcess++;
                g2++;
            } else if (g2 == g.linkGenes_.end())
            // end of second genome?
            {
                // add to the total distance
                numExcess++;
                g1++;
            } else {
                // extract the innovation numbers
                int g1innov = g1->innovationID();
                int g2innov = g2->innovationID();

                // matching genes?
                if (g1innov == g2innov) {
                    numMatchingLinks++;

                    if (parameters.weightDiffCoeff > 0.0) {
                        Real wdiff = (g1->getWeight() - g2->getWeight());
                        if (wdiff < 0) wdiff = -wdiff;  // make sure it is positive
                        totalWeightDifference += wdiff;
                    }

                    // calculate link trait difference here
                    std::map<std::string, Real> linkTraitDifference = g1->getTraitDistances(g2->traits_);
                    // add to the totals
                    for (std::map<std::string, Real>::iterator it = linkTraitDifference.begin(); it != linkTraitDifference.end(); it++) {
                        if (totalLinkTraitDifference.count(it->first) == 0) {
                            totalLinkTraitDifference[it->first] = it->second;
                        } else {
                            totalLinkTraitDifference[it->first] += it->second;
                        }
                    }

                    g1++;
                    g2++;
                } else if (g1innov < g2innov)  // disjoint
                {
                    numDisjoint++;
                    g1++;
                } else if (g1innov > g2innov)  // disjoint
                {
                    numDisjoint++;
                    g2++;
                }
            }
        }

        // find matching neuron IDs
        // One ID->index table for the other genome (instead of HasNeuronID +
        // repeated GetNeuronByID linear rescans per matched neuron, which made
        // this loop quadratic).
        int otherMaxId = 0;
        for (unsigned int i = 0; i < g.numNeurons(); i++) {
            if (g.neuronGenes_[i].id() > otherMaxId) {
                otherMaxId = g.neuronGenes_[i].id();
            }
        }
        std::vector<int> otherIndex(static_cast<size_t>(otherMaxId) + 1, -1);
        for (unsigned int i = 0; i < g.numNeurons(); i++) {
            otherIndex[static_cast<size_t>(g.neuronGenes_[i].id())] = static_cast<int>(i);
        }

        for (unsigned int i = numInputs(); i < numNeurons(); i++) {
            // no inputs considered for comparison
            if ((neuronGenes_[i].type() != INPUT) && (neuronGenes_[i].type() != BIAS)) {
                const int id = neuronGenes_[i].id();
                const int oi = (id >= 0 && id <= otherMaxId) ? otherIndex[static_cast<size_t>(id)] : -1;
                // a match
                if (oi != -1) {
                    const NeuronGene &otherGene = g.neuronGenes_[static_cast<size_t>(oi)];
                    numMatchingNeurons++;

                    if (parameters.activationADiffCoeff > 0.0) {
                        Real aDifference = neuronGenes_[i].a_ - otherGene.a_;
                        if (aDifference < 0.0f) aDifference = -aDifference;
                        totalADifference += aDifference;
                    }

                    if (parameters.activationBDiffCoeff > 0.0) {
                        Real bDifference = neuronGenes_[i].b_ - otherGene.b_;
                        if (bDifference < 0.0f) bDifference = -bDifference;
                        totalBDifference += bDifference;
                    }

                    if (parameters.timeConstantDiffCoeff > 0.0) {
                        Real timeConstantDifference = neuronGenes_[i].timeConstant_ - otherGene.timeConstant_;
                        if (timeConstantDifference < 0.0f) timeConstantDifference = -timeConstantDifference;
                        totalTimeconstantDifference += timeConstantDifference;
                    }

                    if (parameters.biasDiffCoeff > 0.0) {
                        Real biasDifference = neuronGenes_[i].bias_ - otherGene.bias_;
                        if (biasDifference < 0.0f) biasDifference = -biasDifference;
                        totalBiasDifference += biasDifference;
                    }

                    // Activation function type difference is found
                    if (parameters.activationFunctionDiffCoeff > 0.0) {
                        if (neuronGenes_[i].actFunction_ != otherGene.actFunction_) {
                            totalNumActivationDifference++;
                        }
                    }

                    // calculate and add node trait difference here
                    std::map<std::string, Real> neuronTraitDifference = neuronGenes_[i].getTraitDistances(otherGene.traits_);
                    // add to the totals
                    for (std::map<std::string, Real>::iterator it = neuronTraitDifference.begin(); it != neuronTraitDifference.end(); it++) {
                        if (totalNeuronTraitDifference.count(it->first) == 0) {
                            totalNeuronTraitDifference[it->first] = it->second;
                        } else {
                            totalNeuronTraitDifference[it->first] += it->second;
                        }
                    }
                }
            }
        }

        // choose between normalizing for genome size or not
        Real normalizer = 1.0;
        if (parameters.normalizeGenomeSize) {
            normalizer = static_cast<Real>(maxGenomeSize);
        }

        // if there are no matching links or neurons, make it 1.0 to avoid divide error
        if (numMatchingLinks <= 0) numMatchingLinks = 1;
        if (numMatchingNeurons <= 0) numMatchingNeurons = 1;
        if (normalizer <= 0.0) normalizer = 1.0;
        Real tnrm = 1.0 / normalizer;
        Real tnml = 1.0 / numMatchingLinks;
        Real tnmn = 1.0 / numMatchingNeurons;

        totalDistance = (parameters.excessCoeff * (numExcess * tnrm)) + (parameters.disjointCoeff * (numDisjoint * tnrm)) +
                        (parameters.weightDiffCoeff * (totalWeightDifference * tnml)) + (parameters.activationADiffCoeff * (totalADifference * tnmn)) +
                        (parameters.activationBDiffCoeff * (totalBDifference * tnmn)) +
                        (parameters.timeConstantDiffCoeff * (totalTimeconstantDifference * tnmn)) + (parameters.biasDiffCoeff * (totalBiasDifference * tnmn)) +
                        (parameters.activationFunctionDiffCoeff * (totalNumActivationDifference * tnmn));

        // add trait differences according to each one's coeff

        for (std::map<std::string, Real>::iterator it = totalLinkTraitDifference.begin(); it != totalLinkTraitDifference.end(); it++) {
            Real n = (parameters.linkTraits[it->first].importanceCoeff_ * it->second) * tnml;
            if (std::isnan(n) || std::isinf(n)) n = 0.0;
            totalDistance += n;
        }
        for (std::map<std::string, Real>::iterator it = totalNeuronTraitDifference.begin(); it != totalNeuronTraitDifference.end(); it++) {
            Real n = (parameters.neuronTraits[it->first].importanceCoeff_ * it->second) * tnmn;
            if (std::isnan(n) || std::isinf(n)) n = 0.0;
            totalDistance += n;
        }
        for (std::map<std::string, Real>::iterator it = genomeLinkTraitDifference.begin(); it != genomeLinkTraitDifference.end(); it++) {
            Real n = (parameters.genomeTraits[it->first].importanceCoeff_ * it->second);
            if (std::isnan(n) || std::isinf(n)) n = 0.0;
            totalDistance += n;
        }

        // store in cache

        return totalDistance;
    }

    // Returns true if this genome and a_G are compatible (belong in the same species)
    bool Genome::isCompatibleWith(Genome &g, Parameters &parameters) {
        // full compatibility cases
        if (this == &g) return true;

        if (getID() == g.getID()) return true;

        /*if ((NumLinks() == 0) && (a_G.NumLinks() == 0))
            return true;*/

        Real totalDistance = compatibilityDistance(g, parameters);

        if (totalDistance <= parameters.compatTreshold)
            return true;  // compatible
        else
            return false;  // incompatible
    }

    // Returns a random activation function from the canonical set based ot probabilities
    ActivationFunction getRandomActivation(const Parameters &parameters, RNG &rng) {
        std::vector<Real> probs;

        probs.emplace_back(parameters.activationFunctionSignedSigmoidProb);
        probs.emplace_back(parameters.activationFunctionUnsignedSigmoidProb);
        probs.emplace_back(parameters.activationFunctionTanhProb);
        probs.emplace_back(parameters.activationFunctionTanhCubicProb);
        probs.emplace_back(parameters.activationFunctionSignedStepProb);
        probs.emplace_back(parameters.activationFunctionUnsignedStepProb);
        probs.emplace_back(parameters.activationFunctionSignedGaussProb);
        probs.emplace_back(parameters.activationFunctionUnsignedGaussProb);
        probs.emplace_back(parameters.activationFunctionAbsProb);
        probs.emplace_back(parameters.activationFunctionSignedSineProb);
        probs.emplace_back(parameters.activationFunctionUnsignedSineProb);
        probs.emplace_back(parameters.activationFunctionLinearProb);
        probs.emplace_back(parameters.activationFunctionReluProb);
        probs.emplace_back(parameters.activationFunctionSoftplusProb);

        return (NEAT::ActivationFunction)rng.roulette(probs);
    }

    // Adds a new neuron to the genome returns true if succesful
    bool Genome::mutateAddNeuron(InnovationDatabase &innovs, const Parameters &parameters, RNG &rng) {
        // No links to split - go away..
        if (numLinks() == 0) return false;

        // Also we need at least one neuron with 2 incoming links before we split any
        /*bool good=false;
        for (int i=NumInputs(); i<m_NeuronGenes.size(); i++)
        {
            if (LinksOutputtingTo(m_NeuronGenes[i].ID()) > 1)
            {
                good = true;
                break;
            }
        }
        if (!good)
            return false;*/

        // First find a link that to be split
        ////////////////////

        // Select a random link for now
        bool linkFound = false;
        int linkNum = 0;
        int in = 0, out = 0;
        LinkGene chosenlink(0, 0, -1, 0, false);  // to save it for later

        // number of tries to find a good link or give up
        int tries = 256;
        while (!linkFound) {
            if (numLinks() == 1) {
                linkNum = 0;
            }
            /*else if (NumLinks() == 2)
            {
                t_link_num = Rounded(a_RNG.RandFloat());
            }*/
            else {
                // if (NumLinks() > 8)
                {
                    linkNum = rng.randInt(0, numLinks() - 1);  // random selection
                }
                /*else
            {
                // this selects older links for splitting
                Real t_r = abs(RandGaussSigned()/3.0);
                Clamp(t_r, 0, 1);
                t_link_num =  static_cast<int>(t_r * (NumLinks()-1));
            }*/
            }

            in = linkGenes_[linkNum].fromNeuronID();
            out = linkGenes_[linkNum].toNeuronID();

            ASSERT((in > 0) && (out > 0));

            linkFound = true;

            // In case there is only one link, coming from a bias - just quit

            // unless the parameter is set
            if (parameters.dontUseBiasNeuron == false) {
                if ((neuronGenes_[getNeuronIndex(in)].type() == BIAS) && (numLinks() == 1)) {
                    return false;
                }

                // Do not allow splitting a link coming from a bias
                if (neuronGenes_[getNeuronIndex(in)].type() == BIAS) {
                    linkFound = false;
                }
            }

            // Do not allow splitting of recurrent links
            if (!parameters.splitRecurrent) {
                if (linkGenes_[linkNum].isRecurrent()) {
                    if ((!parameters.splitLoopedRecurrent) && (in == out)) {
                        linkFound = false;
                    }
                }
            }

            tries--;
            if (tries <= 0) {
                return false;
            }
        }
        // Now the link has been selected

        // the weight of the link that is being split
        Real origWeight = linkGenes_[linkNum].getWeight();
        chosenlink = linkGenes_[linkNum];  // save the whole link

        // remove the link from the genome
        // find it first and then erase it
        // TODO: add option to keep the link, but disabled
        std::vector<LinkGene>::iterator iter;
        for (iter = linkGenes_.begin(); iter != linkGenes_.end(); iter++) {
            if (iter->innovationID() == linkGenes_[linkNum].innovationID()) {
                // found it! now erase..
                linkGenes_.erase(iter);
                break;
            }
        }

        // Check if an innovation of this type already occured somewhere in the population
        int innovid = innovs.checkInnovation(in, out, NEW_NEURON);

        // the new neuron and links ids
        int nid = 0;
        int l1id = 0;
        int l2id = 0;

        // This is a novel innovation?
        if (innovid == -1) {
            // Add the new neuron innovation
            nid = innovs.addNeuronInnovation(in, out, HIDDEN);
            // add the first link innovation
            l1id = innovs.addLinkInnovation(in, nid);
            // add the second innovation
            l2id = innovs.addLinkInnovation(nid, out);

            // Adjust the SplitY
            Real sy = neuronGenes_[getNeuronIndex(in)].splitY() + neuronGenes_[getNeuronIndex(out)].splitY();
            sy /= 2.0;

            // Create the neuron gene
            NeuronGene ngene(HIDDEN, nid, sy);

            Real a = rng.randFloat();
            Real b = rng.randFloat();
            Real tc = rng.randFloat();
            Real bs = rng.randFloat();
            scale(a, 0, 1, parameters.minActivationA, parameters.maxActivationA);
            scale(b, 0, 1, parameters.minActivationB, parameters.maxActivationB);
            scale(tc, 0, 1, parameters.minNeuronTimeConstant, parameters.maxNeuronTimeConstant);
            scale(bs, 0, 1, parameters.minNeuronBias, parameters.maxNeuronBias);

            clamp(a, parameters.minActivationA, parameters.maxActivationA);
            clamp(b, parameters.minActivationB, parameters.maxActivationB);
            clamp(tc, parameters.minNeuronTimeConstant, parameters.maxNeuronTimeConstant);
            clamp(bs, parameters.minNeuronBias, parameters.maxNeuronBias);

            // Initialize the neuron gene's properties
            ngene.init(a, b, tc, bs, getRandomActivation(parameters, rng));

            // Initialize the traits
            // if (a_RNG.RandFloat() < 0.5)
            //{
            ngene.initTraits(parameters.neuronTraits, rng);
            //}
            // else
            //{   // mate instead of randomizing
            //    t_ngene.m_Traits = m_NeuronGenes[GetNeuronIndex(t_in)].m_Traits;
            //    t_ngene.MateTraits(m_NeuronGenes[GetNeuronIndex(t_out)].m_Traits, a_RNG);
            //}

            // Add the NeuronGene
            neuronGenes_.emplace_back(ngene);

            // Now the links

            // Make sure the recurrent flag is kept
            bool recurrentflag = chosenlink.isRecurrent();

            // First link
            LinkGene l1 = LinkGene(in, nid, l1id, 1.0, recurrentflag);
            // make sure this weight is in the allowed interval
            clamp(l1.weight_, parameters.minWeight, parameters.maxWeight);
            // Init the link's traits
            l1.initTraits(parameters.linkTraits, rng);
            linkGenes_.emplace_back(l1);

            // Second link
            LinkGene l2 = LinkGene(nid, out, l2id, origWeight, recurrentflag);
            // Init the link's traits
            l2.initTraits(parameters.linkTraits, rng);
            linkGenes_.emplace_back(l2);
        } else {
            // This innovation already happened, so inherit it.

            // get the neuron ID
            nid = innovs.findNeuronID(in, out);
            ASSERT(nid != -1);

            // if such an innovation happened, these must exist
            l1id = innovs.checkInnovation(in, nid, NEW_LINK);
            l2id = innovs.checkInnovation(nid, out, NEW_LINK);

            ASSERT((l1id > 0) && (l2id > 0));

            // Perhaps this innovation occured more than once. Find the first such innovation that had occured, but the genome not having the same id.. If
            // didn't find such, then add new innovation.
            std::vector<int> indexs = innovs.checkAllInnovations(in, out, NEW_NEURON);
            bool found = false;
            for (unsigned int i = 0; i < indexs.size(); i++) {
                if (!hasNeuronID(innovs.getInnovationByIndex(indexs[i]).neuronID())) {
                    // found such innovation & this genome doesn't have that neuron ID
                    // So we are going to inherit the innovation
                    nid = innovs.getInnovationByIndex(indexs[i]).neuronID();

                    // these must exist
                    l1id = innovs.checkInnovation(in, nid, NEW_LINK);
                    l2id = innovs.checkInnovation(nid, out, NEW_LINK);

                    ASSERT((l1id > 0) && (l2id > 0));

                    found = true;
                    break;
                }
            }

            // Such an innovation was not found or the genome has all neuron IDs So we are going to add new innovation
            if (!found) {
                // Add 3 new innovations and replace the variables with them

                // Add the new neuron innovation
                nid = innovs.addNeuronInnovation(in, out, HIDDEN);
                // add the first link innovation
                l1id = innovs.addLinkInnovation(in, nid);
                // add the second innovation
                l2id = innovs.addLinkInnovation(nid, out);
            }

            // Add the neuron and the links
            Real sy = neuronGenes_[getNeuronIndex(in)].splitY() + neuronGenes_[getNeuronIndex(out)].splitY();
            sy /= 2.0;

            // Create the neuron gene
            NeuronGene ngene(HIDDEN, nid, sy);

            Real a = rng.randFloat();
            Real b = rng.randFloat();
            Real tc = rng.randFloat();
            Real bs = rng.randFloat();
            scale(a, 0, 1, parameters.minActivationA, parameters.maxActivationA);
            scale(b, 0, 1, parameters.minActivationB, parameters.maxActivationB);
            scale(tc, 0, 1, parameters.minNeuronTimeConstant, parameters.maxNeuronTimeConstant);
            scale(bs, 0, 1, parameters.minNeuronBias, parameters.maxNeuronBias);

            clamp(a, parameters.minActivationA, parameters.maxActivationA);
            clamp(b, parameters.minActivationB, parameters.maxActivationB);
            clamp(tc, parameters.minNeuronTimeConstant, parameters.maxNeuronTimeConstant);
            clamp(bs, parameters.minNeuronBias, parameters.maxNeuronBias);

            // Initialize the neuron gene's properties
            ngene.init(a, b, tc, bs, getRandomActivation(parameters, rng));

            // Initialize the traits
            // if (a_RNG.RandFloat() < 0.5)
            //{
            ngene.initTraits(parameters.neuronTraits, rng);
            //}// mate instead of randomizing
            // else
            //{
            //    t_ngene.m_Traits = m_NeuronGenes[GetNeuronIndex(t_in)].m_Traits;
            //    t_ngene.MateTraits(m_NeuronGenes[GetNeuronIndex(t_out)].m_Traits, a_RNG);
            //}

            // Make sure the recurrent flag is kept
            bool recurrentflag = chosenlink.isRecurrent();

            // Add the NeuronGene
            neuronGenes_.emplace_back(ngene);
            // First link
            LinkGene l1 = LinkGene(in, nid, l1id, 1.0, recurrentflag);
            // make sure this weight is in the allowed interval
            clamp(l1.weight_, parameters.minWeight, parameters.maxWeight);
            // initialize the link's traits
            l1.initTraits(parameters.linkTraits, rng);
            linkGenes_.emplace_back(l1);
            // Second link
            LinkGene l2 = LinkGene(nid, out, l2id, origWeight, recurrentflag);
            // initialize the link's traits
            l2.initTraits(parameters.linkTraits, rng);
            linkGenes_.emplace_back(l2);
        }

        return true;
    }

    // Adds a new link to the genome returns true if succesful
    bool Genome::mutateAddLink(InnovationDatabase &innovs, const Parameters &parameters, RNG &rng) {
        // this variable tells where is the first noninput node
        int firstNoninput = 0;

        // The pair of neurons that has to be connected (1 - in, 2 - out)
        // It may be the same neuron - this means that the connection is a looped recurrent one.
        // These are indexes in the NeuronGenes array!
        int n1index = 0, n2index = 0;

        // Should we make this connection recurrent?
        bool makeRecurrent = false;

        // If so, should it be a looped one?
        bool loopedRecurrent = false;

        // Should it come from the bias neuron?
        bool makeBias = false;

        // Counter of tries to find a candidate pair of neuron/s to connect.
        unsigned int numTries = 0;

        // Decide whether the connection will be recurrent or not..
        if (rng.randFloat() < parameters.recurrentProb) {
            makeRecurrent = true;

            if (rng.randFloat() < parameters.recurrentLoopProb) {
                loopedRecurrent = true;
            }
        }
        // if not recurrent, there is a probability that this link will be from the bias
        // if such link doesn't already exist.
        // in case such link exists, search for a standard feed-forward connection place
        else {
            if (rng.randFloat() < parameters.mutateAddLinkFromBiasProb) {
                makeBias = true;
            }
        }

        // Try to find a good pair of neurons
        bool found = false;

        // Find the first noninput node
        for (unsigned int i = 0; i < numNeurons(); i++) {
            if ((neuronGenes_[i].type() == INPUT) || (neuronGenes_[i].type() == BIAS)) {
                firstNoninput++;
            } else {
                break;
            }
        }

        // A forward link is characterized with the fact that the From neuron has less or equal SplitY value

        // find a good pair of nodes for a forward link
        if (!makeRecurrent) {
            // first see if this should come from the bias or not
            bool foundBias = true;
            n1index = static_cast<int>(numInputs() - 1);  // the bias is always the last input
            // try to find a neuron that is not connected to the bias already
            numTries = 0;
            do {
                n2index = rng.randInt(firstNoninput, static_cast<int>(numNeurons() - 1));
                numTries++;

                if (numTries >= parameters.linkTries) {
                    // couldn't find anything
                    foundBias = false;
                    break;
                }
            } while ((hasLink(neuronGenes_[n1index].id(), neuronGenes_[n2index].id())));  // already present?

            // so if we found that link, we can skip the rest of the things
            if (foundBias && makeBias) {
                found = true;
            }
            // otherwise continue trying to find a normal forward link
            else {
                numTries = 0;
                // try to find a standard forward connection
                do {
                    n1index = rng.randInt(0, static_cast<int>(numNeurons() - 1));
                    n2index = rng.randInt(firstNoninput, static_cast<int>(numNeurons() - 1));
                    numTries++;

                    if (numTries >= parameters.linkTries) {
                        // couldn't find anything say goodbye
                        return false;
                    }
                } while (
                    //(m_NeuronGenes[t_n1idx].SplitY() > m_NeuronGenes[t_n2idx].SplitY()) // backward?
                    //||
                    (hasLink(neuronGenes_[n1index].id(), neuronGenes_[n2index].id()))  // already present?
                    || (neuronGenes_[n1index].type() == OUTPUT)                        // consider connections out of outputs recurrent
                    || (n1index == n2index)                                            // make sure they differ
                );

                // it found a good pair of neurons
                found = true;
            }
        }
        // find a good pair of nodes for a recurrent link (non-looped)
        else if (makeRecurrent && !loopedRecurrent) {
            numTries = 0;
            do {
                n1index = rng.randInt(firstNoninput, static_cast<int>(numNeurons() - 1));
                n2index = rng.randInt(firstNoninput, static_cast<int>(numNeurons() - 1));
                numTries++;

                if (numTries >= parameters.linkTries) {
                    // couldn't find anything say goodbye
                    return false;
                }
            }
            // NOTE: this considers output-output connections as forward. Should be fixed.
            while (
                //(m_NeuronGenes[t_n1idx].SplitY() <= m_NeuronGenes[t_n2idx].SplitY()) // forward?
                //||
                (hasLink(neuronGenes_[n1index].id(), neuronGenes_[n2index].id()))  // already present?
                || (n1index == n2index)                                            // they should differ
            );

            // it found a good pair of neurons
            found = true;
        }
        // find a good neuron to make a looped recurrent link
        else if (makeRecurrent && loopedRecurrent) {
            numTries = 0;
            do {
                n1index = n2index = rng.randInt(firstNoninput, static_cast<int>(numNeurons() - 1));
                numTries++;

                if (numTries >= parameters.linkTries) {
                    // couldn't find anything say goodbye
                    return false;
                }
            } while ((hasLink(neuronGenes_[n1index].id(), neuronGenes_[n2index].id()))  // already present?
                                                                                        //||
                     //(m_NeuronGenes[t_n1idx].Type() == OUTPUT) // do not allow looped recurrent on the outputs (experimental)
            );

            // it found a good pair of neurons
            found = true;
        }

        // To make sure it is all right
        if (!found) {
            return false;
        }

        // This link MUST NOT be a part of the genome by any reason
        ASSERT((!hasLink(neuronGenes_[n1index].id(), neuronGenes_[n2index].id())));  // already present?

        // extract the neuron IDs from the indexes
        int n1id = neuronGenes_[n1index].id();
        int n2id = neuronGenes_[n2index].id();

        // So we have a good pair of neurons to connect. See the innovation database if this is novel innovation.
        int innovid = innovs.checkInnovation(n1id, n2id, NEW_LINK);

        // Choose the weight for this link
        Real weight = rng.randFloat();
        scale(weight, 0, 1, parameters.minWeight, parameters.maxWeight);

        // A novel innovation?
        if (innovid == -1) {
            // Make new innovation
            innovid = innovs.addLinkInnovation(n1id, n2id);
        }

        // Create and add the link
        LinkGene l = LinkGene(n1id, n2id, innovid, weight, makeRecurrent);
        // init the link's traits
        l.initTraits(parameters.linkTraits, rng);
        linkGenes_.emplace_back(l);

        // All done.
        return true;
    }

    ///////////
    // Helper functions for the pruning procedure

    // Removes the link with the given innovation ID (the semantics the header declares
    // and that Mutate_RemoveLink/Cleanup rely on). The previous "simple index" version
    // erased by position — wiping the whole link list whenever a_idx was 0, and
    // erasing out-of-range positions when callers (correctly) passed innovation IDs.
    void Genome::removeLinkGene(int innovid) {
        for (std::vector<LinkGene>::iterator curlink = linkGenes_.begin(); curlink != linkGenes_.end(); ++curlink) {
            if (curlink->innovationID() == innovid) {
                linkGenes_.erase(curlink);
                return;
            }
        }
    }

    // Deletes the neuron gene and every link touching it.
    void Genome::removeNeuronGene(int id) {
        // the list of links connected to this neuron
        std::vector<int> linkRemovalQueue;

        bool removed = false;

        do {
            removed = false;
            // Remove all links connected to this neuron ID
            for (int i = 0; i < numLinks(); i++) {
                if ((linkGenes_[i].fromNeuronID() == id) || (linkGenes_[i].toNeuronID() == id)) {
                    // found one, remove it (by innovation ID, the sole RemoveLinkGene contract)
                    removeLinkGene(linkGenes_[i].innovationID());
                    removed = true;
                    break;
                }
            }
        } while (removed);

        // Now remove them
        /*for (unsigned int i = 0; i < t_link_removal_queue.size(); i++)
        {
            RemoveLinkGene(t_link_removal_queue[i]);
        }*/

        // Now is safe to remove the neuron find it first
        std::vector<NeuronGene>::iterator curneuron = neuronGenes_.begin();

        while (curneuron != neuronGenes_.end()) {
            if (curneuron->id() == id) {
                // found it, erase and quit
                neuronGenes_.erase(curneuron);
                break;
            }

            curneuron++;
        }
    }

    // Returns true is the specified neuron ID is a dead end or isolated
    bool Genome::isDeadEndNeuron(int id) const {
        bool noIncoming = true;
        bool noOutgoing = true;

        // ID->type table so the link scan below does not rescan the neuron list
        // for every connection (was O(links x neurons) per query).
        int maxId = 0;
        for (unsigned int i = 0; i < numNeurons(); i++) {
            if (neuronGenes_[i].id() > maxId) {
                maxId = neuronGenes_[i].id();
            }
        }
        std::vector<int> idToType(static_cast<size_t>(maxId) + 1, -1);
        for (unsigned int i = 0; i < numNeurons(); i++) {
            idToType[static_cast<size_t>(neuronGenes_[i].id())] = static_cast<int>(neuronGenes_[i].type());
        }

        // search the links and prove both are wrong
        for (unsigned int i = 0; i < numLinks(); i++) {
            const int from = linkGenes_[i].fromNeuronID();
            const int fromType = (from >= 0 && from <= maxId) ? idToType[static_cast<size_t>(from)] : static_cast<int>(getNeuronByID(from).type());
            // there is a link going to this neuron, so there are incoming don't count the link if it is recurrent or coming from a bias
            if ((linkGenes_[i].toNeuronID() == id) && (!linkGenes_[i].isLoopedRecurrent()) && (fromType != static_cast<int>(BIAS))) {
                noIncoming = false;
            }

            // there is a link going from this neuron, so there are outgoing don't count the link if it is recurrent or coming from a bias
            if ((linkGenes_[i].fromNeuronID() == id) && (!linkGenes_[i].isLoopedRecurrent()) && (fromType != static_cast<int>(BIAS))) {
                noOutgoing = false;
            }
        }

        // if just one of these is true, this neuron is a dead end
        if (noIncoming || noOutgoing) {
            return true;
        } else {
            return false;
        }
    }

    // Search the genome for isolated structure and clean it up Returns true is something was removed
    bool Genome::cleanup() {
        bool removed = false;

        // remove any dead-end hidden neurons
        for (unsigned int i = 0; i < numNeurons(); i++) {
            if (neuronGenes_[i].type() == HIDDEN) {
                if (isDeadEndNeuron(neuronGenes_[i].id())) {
                    removeNeuronGene(neuronGenes_[i].id());
                    removed = true;
                }
            }
        }

        // a special case are isolated outputs - these are outputs having
        // one and only one looped recurrent connection
        // we simply remove these connections and leave the outputs naked.
        for (unsigned int i = 0; i < numNeurons(); i++) {
            if (neuronGenes_[i].type() == OUTPUT) {
                // Only outputs with 1 input and 1 output connection are considered.
                if ((linksInputtingFrom(neuronGenes_[i].id()) == 1) && (linksOutputtingTo(neuronGenes_[i].id()) == 1)) {
                    // that must be a lonely looped recurrent,
                    // because we know that the outputs are the dead end of the network
                    // find this link
                    for (unsigned int j = 0; j < numLinks(); j++) {
                        if (linkGenes_[j].toNeuronID() == neuronGenes_[i].id()) {
                            // Remove it.
                            removeLinkGene(linkGenes_[j].innovationID());
                            removed = true;
                        }
                    }
                }
            }
        }

        return removed;
    }

    // Returns true if has any dead end
    bool Genome::hasDeadEnds() const {
        // any dead-end hidden neurons?
        for (unsigned int i = 0; i < numNeurons(); i++) {
            if (neuronGenes_[i].type() == HIDDEN) {
                if (isDeadEndNeuron(neuronGenes_[i].id())) {
                    return true;
                }
            }
        }

        // a special case are isolated outputs - these are outputs having
        // one and only one looped recurrent connection or no connections at all
        for (unsigned int i = 0; i < numNeurons(); i++) {
            if (neuronGenes_[i].type() == OUTPUT) {
                // Only outputs with 1 input and 1 output connection are considered.
                if ((linksInputtingFrom(neuronGenes_[i].id()) == 1) && (linksOutputtingTo(neuronGenes_[i].id()) == 1)) {
                    // that must be a lonely looped recurrent,
                    // because we know that the outputs are the dead end of the network
                    return true;
                }

                // There may be cases for totally isolated outputs Consider this if only one output is present
                if (numOutputs() == 1)
                    if ((linksInputtingFrom(neuronGenes_[i].id()) == 0) && (linksOutputtingTo(neuronGenes_[i].id()) == 0)) {
                        return true;
                    }
            }
        }

        return false;
    }

    // Remove a link from the genome
    // A cleanup procedure is invoked so any dead-ends or stranded neurons are also deleted
    // returns true if succesful
    bool Genome::mutateRemoveLink(RNG &rng) {
        // at least 2 links must be present in the genome
        if (numLinks() < 2) return false;

        // find a random link to remove with tendency to remove older connections
        Real randnum = rng.randFloat();  // RandGaussSigned()/4;
        clamp(randnum, 0, 1);

        int linkIndex = static_cast<int>(randnum * static_cast<Real>(numLinks() - 1));  // RandInt(0, static_cast<int>(NumLinks()-1));

        // remove it
        removeLinkGene(linkGenes_[linkIndex].innovationID());

        // Now cleanup
        // Cleanup();

        return true;
    }

    // Returns the count of links inputting from the specified neuron ID
    int Genome::linksInputtingFrom(int id) const {
        int counter = 0;
        for (unsigned int i = 0; i < numLinks(); i++) {
            if (linkGenes_[i].fromNeuronID() == id) counter++;
        }

        return counter;
    }

    // Returns the count of links outputting to the specified neuron ID
    int Genome::linksOutputtingTo(int id) const {
        int counter = 0;
        for (unsigned int i = 0; i < numLinks(); i++) {
            if (linkGenes_[i].toNeuronID() == id) counter++;
        }

        return counter;
    }

    // Replaces a hidden neuron having only one input and only one output with a direct link between them.
    bool Genome::mutateRemoveSimpleNeuron(InnovationDatabase &innovs, const Parameters &parameters, RNG &rng) {
        // At least one hidden node must be present
        if (numNeurons() == (numInputs() + numOutputs())) return false;

        // Build a list of candidate neurons for deletion
        // Indexes!
        std::vector<int> neuronsToDelete;
        for (int i = 0; i < numNeurons(); i++) {
            if ((linksInputtingFrom(neuronGenes_[i].id()) == 1) && (linksOutputtingTo(neuronGenes_[i].id()) == 1) && (neuronGenes_[i].type() == HIDDEN)) {
                neuronsToDelete.emplace_back(i);
            }
        }

        // If the list is empty, say goodbye
        if (neuronsToDelete.empty()) return false;

        // Now choose a random one to delete
        int choice;
        if (neuronsToDelete.size() == 2)
            choice = rounded(rng.randFloat());
        else
            choice = rng.randInt(0, static_cast<int>(neuronsToDelete.size() - 1));

        // the links in & out
        int l1index = -1, l2index = -1;

        // find the link outputting to the neuron
        for (unsigned int i = 0; i < numLinks(); i++) {
            if (linkGenes_[i].toNeuronID() == neuronGenes_[neuronsToDelete[choice]].id()) {
                l1index = i;
                break;
            }
        }
        // find the link inputting from the neuron
        for (unsigned int i = 0; i < numLinks(); i++) {
            if (linkGenes_[i].fromNeuronID() == neuronGenes_[neuronsToDelete[choice]].id()) {
                l2index = i;
                break;
            }
        }

        ASSERT((l1index >= 0) && (l2index >= 0));

        // OK now see if a link connecting the original 2 nodes is present. If it is, we will just delete the neuron and quit.
        if (hasLink(linkGenes_[l1index].fromNeuronID(), linkGenes_[l2index].toNeuronID())) {
            removeNeuronGene(neuronGenes_[neuronsToDelete[choice]].id());
            return true;
        }
        // Else the link is not present and we will replace the neuron and 2 links with one link
        else {
            // Remember the first link's weight
            Real weight = linkGenes_[l1index].getWeight();

            // See the innovation database for an innovation number
            int innovid = innovs.checkInnovation(linkGenes_[l1index].fromNeuronID(), linkGenes_[l2index].toNeuronID(), NEW_LINK);

            // a novel innovation?
            if (innovid == -1) {
                // Save the IDs for a while
                int from = linkGenes_[l1index].fromNeuronID();
                int to = linkGenes_[l2index].toNeuronID();

                // Remove the neuron and its links now
                removeNeuronGene(neuronGenes_[neuronsToDelete[choice]].id());

                // Add the innovation and the link gene
                int newinnov = innovs.addLinkInnovation(from, to);
                LinkGene lg = LinkGene(from, to, newinnov, weight, false);
                lg.initTraits(parameters.linkTraits, rng);

                linkGenes_.emplace_back(lg);

                // bye
                return true;
            }
            // not a novel innovation
            else {
                // Save the IDs for a while
                int from = linkGenes_[l1index].fromNeuronID();
                int to = linkGenes_[l2index].toNeuronID();

                // Remove the neuron and its links now
                removeNeuronGene(neuronGenes_[neuronsToDelete[choice]].id());

                // Add the link
                LinkGene lg = LinkGene(from, to, innovid, weight, false);
                lg.initTraits(parameters.linkTraits, rng);
                linkGenes_.emplace_back(lg);

                // TODO: Maybe inherit the traits from one of the links

                // bye
                return true;
            }
        }

        return false;
    }

    // Perturbs the weights
    bool Genome::mutateLinkWeights(const Parameters &parameters, RNG &rng) {
        // The end part of the genome
        int genometail = 0;
        if (numLinks() > initialNumLinks_) {
            genometail = static_cast<int>(static_cast<Real>(numLinks()) * 0.8);
        }
        if (genometail < initialNumLinks_) {
            genometail = initialNumLinks_;
        }

        bool didMutate = false;

        // This tells us if this mutation will shake things up
        bool severeMutation;

        if (rng.randFloat() < parameters.mutateWeightsSevereProb) {
            severeMutation = true;
        } else {
            severeMutation = false;
        }

        // For all links..
        for (unsigned int i = 0; i < linkGenes_.size(); i++) {
            if ((!severeMutation) && (rng.randFloat() < parameters.weightMutationRate)) {
                bool ontail = false;  //(i >= t_genometail);
                Real linkGenesWeight = linkGenes_[i].getWeight();

                if (ontail || (rng.randFloat() < parameters.weightReplacementRate)) {
                    linkGenesWeight = rng.randFloatSigned() * parameters.weightReplacementMaxPower;

                    // t_LinkGenesWeight = a_RNG.RandFloat();
                    // Scale(t_LinkGenesWeight, 0.0, 1.0, a_Parameters.MinWeight, a_Parameters.MaxWeight);
                } else {
                    linkGenesWeight += rng.randFloatSigned() * parameters.weightMutationMaxPower;
                }

                clamp(linkGenesWeight, parameters.minWeight, parameters.maxWeight);
                linkGenes_[i].setWeight(linkGenesWeight);

                didMutate = true;
            } else if (severeMutation) {
                if (rng.randFloat() < parameters.weightMutationRate) {
                    Real linkGenesWeight = rng.randFloat();
                    scale(linkGenesWeight, 0.0, 1.0, parameters.minWeight, parameters.maxWeight);
                    linkGenes_[i].setWeight(linkGenesWeight);

                    didMutate = true;
                }
            }
        }

        return didMutate;
    }

    // Set all link weights to random values between [-R .. R]
    void Genome::randomizeLinkWeights(const Parameters &parameters, RNG &rng) {
        // For all links..
        for (unsigned int i = 0; i < numLinks(); i++) {
            Real nf = 0;
            nf = rng.randFloat();
            scale(nf, 0.0, 1.0, parameters.minWeight, parameters.maxWeight);
            linkGenes_[i].setWeight(nf);
        }
    }

    // Randomize traits
    void Genome::randomizeTraits(const Parameters &parameters, RNG &rng) {
        for (NeuronGene &neuronGene : neuronGenes_) {
            neuronGene.initTraits(parameters.neuronTraits, rng);
        }
        for (LinkGene &linkGene : linkGenes_) {
            linkGene.initTraits(parameters.linkTraits, rng);
        }

        genomeGene_.initTraits(parameters.genomeTraits, rng);
    }

    // Perturbs the A parameters of the neuron activation functions
    bool Genome::mutateNeuronActivationsA(const Parameters &parameters, RNG &rng) {
        // for all neurons..
        for (unsigned int i = 0; i < numNeurons(); i++) {
            // skip inputs and bias
            if ((neuronGenes_[i].type() != INPUT) && (neuronGenes_[i].type() != BIAS)) {
                Real randnum = rng.randFloatSigned() * parameters.activationAMutationMaxPower;

                neuronGenes_[i].a_ += randnum;

                clamp(neuronGenes_[i].a_, parameters.minActivationA, parameters.maxActivationA);
            }
        }

        return true;
    }

    // Perturbs the B parameters of the neuron activation functions
    bool Genome::mutateNeuronActivationsB(const Parameters &parameters, RNG &rng) {
        // for all neurons..
        for (unsigned int i = 0; i < numNeurons(); i++) {
            // skip inputs and bias
            if ((neuronGenes_[i].type() != INPUT) && (neuronGenes_[i].type() != BIAS)) {
                Real randnum = rng.randFloatSigned() * parameters.activationBMutationMaxPower;

                neuronGenes_[i].b_ += randnum;

                clamp(neuronGenes_[i].b_, parameters.minActivationB, parameters.maxActivationB);
            }
        }

        return true;
    }

    // Changes the activation function type for a random neuron
    bool Genome::mutateNeuronActivationType(const Parameters &parameters, RNG &rng) {
        // the first non-input neuron
        int firstIndex = numInputs();
        int choice = rng.randInt(firstIndex, neuronGenes_.size() - 1);

        int cur = neuronGenes_[choice].actFunction_;

        neuronGenes_[choice].actFunction_ = getRandomActivation(parameters, rng);
        if (neuronGenes_[choice].actFunction_ == cur)  // same as before?
        {
            return false;
        } else {
            return true;
        }
    }

    // Perturbs the neuron time constants
    bool Genome::mutateNeuronTimeConstants(const Parameters &parameters, RNG &rng) {
        // for all neurons..
        for (unsigned int i = 0; i < numNeurons(); i++) {
            // skip inputs and bias
            if ((neuronGenes_[i].type() != INPUT) && (neuronGenes_[i].type() != BIAS)) {
                Real randnum = rng.randFloatSigned() * parameters.timeConstantMutationMaxPower;

                neuronGenes_[i].timeConstant_ += randnum;

                clamp(neuronGenes_[i].timeConstant_, parameters.minNeuronTimeConstant, parameters.maxNeuronTimeConstant);
            }
        }

        return true;
    }

    // Perturbs the neuron biases
    bool Genome::mutateNeuronBiases(const Parameters &parameters, RNG &rng) {
        // for all neurons..
        for (unsigned int i = 0; i < numNeurons(); i++) {
            // skip inputs and bias
            if ((neuronGenes_[i].type() != INPUT) && (neuronGenes_[i].type() != BIAS)) {
                Real randnum = rng.randFloatSigned() * parameters.biasMutationMaxPower;

                neuronGenes_[i].bias_ += randnum;

                clamp(neuronGenes_[i].bias_, parameters.minNeuronBias, parameters.maxNeuronBias);
            }
        }

        return true;
    }

    bool Genome::mutateNeuronTraits(const Parameters &parameters, RNG &rng) {
        bool didMutate = false;
        for (std::vector<NeuronGene>::iterator it = neuronGenes_.begin(); it != neuronGenes_.end(); it++) {
            // don't mutate inputs and bias
            if ((it->type() != INPUT) && (it->type() != BIAS)) {
                didMutate |= it->mutateTraits(parameters.neuronTraits, rng);
            }
        }
        return didMutate;
    }

    bool Genome::mutateLinkTraits(const Parameters &parameters, RNG &rng) {
        bool didMutate = false;
        for (std::vector<LinkGene>::iterator it = linkGenes_.begin(); it != linkGenes_.end(); it++) {
            didMutate |= it->mutateTraits(parameters.linkTraits, rng);
        }
        return didMutate;
    }

    bool Genome::mutateGenomeTraits(const Parameters &parameters, RNG &rng) { return genomeGene_.mutateTraits(parameters.genomeTraits, rng); }

    // Mate this genome with dad and return the baby
    // This is multipoint mating - genes inherited randomly
    // Disjoint and excess genes are inherited from the fittest parent
    // If fitness is equal, the smaller genome is assumed to be the better one
    Genome Genome::mate(Genome &dad, bool mateAverage, bool interSpecies, RNG &rng, Parameters &parameters) {
        // Cannot mate with itself
        if (getID() == dad.getID()) return *this;

        // helps make the code clearer
        enum parentType {
            MOM,
            DAD,
        };

        // This is the fittest genome.
        parentType better;

        // This empty genome will hold the baby
        Genome baby;

        // create iterators so we can step through each parents genes and set them to the first gene of each parent
        std::vector<LinkGene>::iterator curMom = linkGenes_.begin();
        std::vector<LinkGene>::iterator curDad = dad.linkGenes_.begin();

        // this will hold a copy of the gene we wish to add at each step
        LinkGene selectedgene(0, 0, -1, 0, false);

        // Mate the GenomeGene first Determine if it will pick either gene or mate it
        if (rng.randFloat() < parameters.multipointCrossoverRate) {
            // pick
            Gene n;

            if (rng.randFloat() < parameters.preferFitterParentRate) {
                n = (getFitness() > dad.getFitness()) ? genomeGene_ : dad.genomeGene_;
            } else {
                n = (rng.randFloat() < 0.5) ? genomeGene_ : dad.genomeGene_;
            }
            baby.genomeGene_ = n;
        } else {
            // mate
            Gene n = genomeGene_;
            n.mateTraits(dad.genomeGene_.traits_, rng);
            baby.genomeGene_ = n;
        }

        // Make sure all inputs/outputs are present in the baby
        // Essential to FS-NEAT

        if (!parameters.dontUseBiasNeuron) {
            // the inputs
            unsigned int i = 0;
            for (i = 0; i < numInputs_ - 1; i++) {
                // Determine if it will pick either gene or mate it
                /*if (a_RNG.RandFloat() < a_Parameters.MultipointCrossoverRate)
                {
                    // pick
                    NeuronGene n;
                    // most of the time pick from the fitter parent
                    if (a_RNG.RandFloat() < a_Parameters.PreferFitterParentRate)
                    {
                        n = (GetFitness() > a_Dad.GetFitness())? m_NeuronGenes[i] : a_Dad.m_NeuronGenes[i];
                    }
                    else
                    {
                        // pick randomly
                        n = (a_RNG.RandFloat() < 0.5)? m_NeuronGenes[i] : a_Dad.m_NeuronGenes[i];
                    }

                    t_baby.m_NeuronGenes.emplace_back(n);
                }
                else
                {*/
                // mate
                // n.MateTraits(a_Dad.m_NeuronGenes[i].m_Traits, a_RNG);
                baby.neuronGenes_.emplace_back(neuronGenes_[i]);
                //}
            }
            /*if (a_RNG.RandFloat() < a_Parameters.MultipointCrossoverRate)
            {
                // the bias
                NeuronGene nb;
                if (a_RNG.RandFloat() < a_Parameters.PreferFitterParentRate)
                {
                    nb = (GetFitness() > a_Dad.GetFitness())? m_NeuronGenes[i] : a_Dad.m_NeuronGenes[i];
                }
                else
                {
                    nb = (a_RNG.RandFloat() < 0.5) ? m_NeuronGenes[i] : a_Dad.m_NeuronGenes[i];
                }
                t_baby.m_NeuronGenes.emplace_back(nb);
            }
            else
            {*/
            // mate
            // nb.MateTraits(a_Dad.m_NeuronGenes[i].m_Traits, a_RNG);
            baby.neuronGenes_.emplace_back(neuronGenes_[i]);
            //}
        } else {
            // the inputs
            for (unsigned int i = 0; i < numInputs_; i++) {
                /*if (a_RNG.RandFloat() < a_Parameters.MultipointCrossoverRate)
                {
                    NeuronGene n;
                    if (a_RNG.RandFloat() < a_Parameters.PreferFitterParentRate)
                    {
                        n = (GetFitness() > a_Dad.GetFitness())? m_NeuronGenes[i] : a_Dad.m_NeuronGenes[i];
                    }
                    else
                    {
                        n = (a_RNG.RandFloat() < 0.5) ? m_NeuronGenes[i] : a_Dad.m_NeuronGenes[i];
                    }
                    t_baby.m_NeuronGenes.emplace_back(n);
                }
                else
                {*/
                // n.MateTraits(a_Dad.m_NeuronGenes[i].m_Traits, a_RNG);
                baby.neuronGenes_.emplace_back(neuronGenes_[i]);
                //}
            }
        }

        // the outputs
        for (unsigned int i = 0; i < numOutputs_; i++) {
            NeuronGene tempneuron(OUTPUT, 0, 1);

            if (rng.randFloat() < parameters.multipointCrossoverRate) {
                if (rng.randFloat() < parameters.preferFitterParentRate) {
                    if (getFitness() > dad.getFitness()) {
                        // from mother
                        tempneuron = getNeuronByIndex(i + numInputs_);
                    } else {
                        // from father
                        tempneuron = dad.getNeuronByIndex(i + numInputs_);
                    }
                } else {
                    // random pick
                    if (rng.randFloat() < 0.5) {
                        // from mother
                        tempneuron = getNeuronByIndex(i + numInputs_);
                    } else {
                        // from father
                        tempneuron = dad.getNeuronByIndex(i + numInputs_);
                    }
                }
            } else {
                // mating from mother
                tempneuron = getNeuronByIndex(i + numInputs_);
                tempneuron.mateTraits(dad.getNeuronByIndex(i + numInputs_).traits_, rng);
            }

            baby.neuronGenes_.emplace_back(tempneuron);
        }

        // if they are of equal fitness use the shorter (because we want to keep the networks as small as possible)
        if (getFitness() == dad.getFitness()) {
            // if they are of equal fitness and length just choose one at random
            if (numLinks() == dad.numLinks()) {
                if (rng.randFloat() < 0.5) {
                    better = MOM;
                } else {
                    better = DAD;
                }
            } else {
                if (numLinks() < dad.numLinks()) {
                    better = MOM;
                } else {
                    better = DAD;
                }
            }
        } else {
            if (getFitness() > dad.getFitness()) {
                better = MOM;
            } else {
                better = DAD;
            }
        }

        //////////////////////////////////////////////////////////
        // The better genome has been chosen. Now we mate them.
        //////////////////////////////////////////////////////////

        // for cleaning up
        LinkGene emptygene(0, 0, -1, 0, false);
        bool skip = false;
        int innovMom, innovDad;

        // step through each parents link genes until we reach the end of both
        while (!((curMom == linkGenes_.end()) && (curDad == dad.linkGenes_.end()))) {
            selectedgene = emptygene;
            skip = false;
            innovMom = innovDad = 0;

            // the end of mum's genes have been reached EXCESS
            if (curMom == linkGenes_.end()) {
                // select dads gene
                selectedgene = *curDad;
                // move onto dad's next gene
                curDad++;

                // if mom is fittest, abort adding
                if (better == MOM) {
                    skip = true;
                }
            }

            // the end of dads's genes have been reached EXCESS
            else if (curDad == dad.linkGenes_.end()) {
                // add mums gene
                selectedgene = *curMom;
                // move onto mum's next gene
                curMom++;

                // if dad is fittest, abort adding
                if (better == DAD) {
                    skip = true;
                }
            } else {
                // extract the innovation numbers
                innovMom = curMom->innovationID();
                innovDad = curDad->innovationID();

                // if both innovations match
                if (innovMom == innovDad) {
                    // get a gene from either parent or average
                    if (rng.randFloat() < parameters.multipointCrossoverRate) {
                        if (rng.randFloat() < parameters.preferFitterParentRate) {
                            // Prefer the *fitter* parent's gene (was inverted: picked the mom when she was worse)
                            if (getFitness() > dad.getFitness()) {
                                selectedgene = *curMom;
                            } else {
                                selectedgene = *curDad;
                            }
                        } else {
                            if (rng.randFloat() < 0.5) {
                                selectedgene = *curMom;
                            } else {
                                selectedgene = *curDad;
                            }
                        }
                    } else {
                        selectedgene = *curMom;
                        const Real weight = (curDad->getWeight() + curMom->getWeight()) / 2.0;
                        selectedgene.setWeight(weight);
                        // Mate traits here
                        selectedgene.mateTraits(curDad->traits_, rng);
                    }

                    // move onto next gene of each parent
                    curMom++;
                    curDad++;
                } else  // DISJOINT
                    if (innovMom < innovDad) {
                        selectedgene = *curMom;
                        curMom++;

                        if (better == DAD) {
                            skip = true;
                        }
                    } else  // DISJOINT
                        if (innovDad < innovMom) {
                            selectedgene = *curDad;
                            curDad++;

                            if (better == MOM) {
                                skip = true;
                            }
                        }
            }

            // for interspecies mating, allow all genes through
            if (interSpecies) {
                skip = false;
            }

            // If the selected gene's innovation number is negative,
            // this means that no gene is selected (should be skipped)
            // also check the baby if it already has this link (maybe unnecessary)
            if ((selectedgene.innovationID() > 0) && (!baby.hasLink(selectedgene.fromNeuronID(), selectedgene.toNeuronID()))) {
                if (!skip) {
                    baby.linkGenes_.emplace_back(selectedgene);

                    // Check if we already have the nodes referred to in t_selectedgene. If not, they need to be added.

                    // NeuronGene t_ngene1(NONE, 0, 0);
                    // NeuronGene t_ngene2(NONE, 0, 0);

                    // mom has a neuron ID not present in the baby? From
                    if ((!baby.hasNeuronID(selectedgene.fromNeuronID())) && (hasNeuronID(selectedgene.fromNeuronID()))) {
                        // See if dad has the same neuron.
                        if (dad.hasNeuronID(selectedgene.fromNeuronID())) {
                            // if so, then choose randomly which neuron the baby shoud inherit
                            if (rng.randFloat() < parameters.multipointCrossoverRate) {
                                if (rng.randFloat() < parameters.preferFitterParentRate) {
                                    if (getFitness() > dad.getFitness()) {
                                        // add mom's neuron to the baby
                                        baby.neuronGenes_.emplace_back(neuronGenes_[getNeuronIndex(selectedgene.fromNeuronID())]);
                                    } else {
                                        // add dad's neuron to the baby
                                        baby.neuronGenes_.emplace_back(dad.neuronGenes_[dad.getNeuronIndex(selectedgene.fromNeuronID())]);
                                    }
                                } else {
                                    if (rng.randFloat() < 0.5) {
                                        // add mom's neuron to the baby
                                        baby.neuronGenes_.emplace_back(neuronGenes_[getNeuronIndex(selectedgene.fromNeuronID())]);
                                    } else {
                                        // add dad's neuron to the baby
                                        baby.neuronGenes_.emplace_back(dad.neuronGenes_[dad.getNeuronIndex(selectedgene.fromNeuronID())]);
                                    }
                                }
                            } else {
                                // mate the neurons
                                NeuronGene firstNeuron = neuronGenes_[getNeuronIndex(selectedgene.fromNeuronID())];
                                NeuronGene secondNeuron = dad.neuronGenes_[dad.getNeuronIndex(selectedgene.fromNeuronID())];
                                firstNeuron.mateTraits(secondNeuron.traits_, rng);
                                baby.neuronGenes_.emplace_back(firstNeuron);
                            }
                        } else {
                            // add mom's neuron to the baby
                            baby.neuronGenes_.emplace_back(neuronGenes_[getNeuronIndex(selectedgene.fromNeuronID())]);
                        }
                    }

                    // To
                    if ((!baby.hasNeuronID(selectedgene.toNeuronID())) && (hasNeuronID(selectedgene.toNeuronID()))) {
                        // See if dad has the same neuron.
                        if (dad.hasNeuronID(selectedgene.toNeuronID())) {
                            if (rng.randFloat() < parameters.multipointCrossoverRate) {
                                if (rng.randFloat() < parameters.preferFitterParentRate) {
                                    if (getFitness() > dad.getFitness()) {
                                        // add mom's neuron to the baby
                                        baby.neuronGenes_.emplace_back(neuronGenes_[getNeuronIndex(selectedgene.toNeuronID())]);
                                    } else {
                                        // add dad's neuron to the baby
                                        baby.neuronGenes_.emplace_back(dad.neuronGenes_[dad.getNeuronIndex(selectedgene.toNeuronID())]);
                                    }
                                } else {
                                    // if so, then choose randomly which neuron the baby shoud inherit
                                    if (rng.randFloat() < 0.5) {
                                        // add mom's neuron to the baby
                                        baby.neuronGenes_.emplace_back(neuronGenes_[getNeuronIndex(selectedgene.toNeuronID())]);
                                    } else {
                                        // add dad's neuron to the baby
                                        baby.neuronGenes_.emplace_back(dad.neuronGenes_[dad.getNeuronIndex(selectedgene.toNeuronID())]);
                                    }
                                }
                            } else {
                                // mate the neurons
                                NeuronGene firstNeuron = neuronGenes_[getNeuronIndex(selectedgene.toNeuronID())];
                                NeuronGene secondNeuron = dad.neuronGenes_[dad.getNeuronIndex(selectedgene.toNeuronID())];
                                firstNeuron.mateTraits(secondNeuron.traits_, rng);
                                baby.neuronGenes_.emplace_back(firstNeuron);
                            }
                        } else {
                            // add mom's neuron to the baby
                            baby.neuronGenes_.emplace_back(neuronGenes_[getNeuronIndex(selectedgene.toNeuronID())]);
                        }
                    }

                    // dad has a neuron ID not present in the baby? From
                    if ((!baby.hasNeuronID(selectedgene.fromNeuronID())) && (dad.hasNeuronID(selectedgene.fromNeuronID()))) {
                        // See if mom has the same neuron
                        if (hasNeuronID(selectedgene.fromNeuronID())) {
                            if (rng.randFloat() < parameters.multipointCrossoverRate) {
                                if (rng.randFloat() < parameters.preferFitterParentRate) {
                                    if (getFitness() < dad.getFitness()) {
                                        // add dad's neuron to the baby
                                        baby.neuronGenes_.emplace_back(dad.neuronGenes_[dad.getNeuronIndex(selectedgene.fromNeuronID())]);
                                    } else {
                                        // add mom's neuron to the baby
                                        baby.neuronGenes_.emplace_back(neuronGenes_[getNeuronIndex(selectedgene.fromNeuronID())]);
                                    }
                                } else {
                                    // if so, then choose randomly which neuron the baby shoud inherit
                                    if (rng.randFloat() < 0.5) {
                                        // add dad's neuron to the baby
                                        baby.neuronGenes_.emplace_back(dad.neuronGenes_[dad.getNeuronIndex(selectedgene.fromNeuronID())]);
                                    } else {
                                        // add mom's neuron to the baby
                                        baby.neuronGenes_.emplace_back(neuronGenes_[getNeuronIndex(selectedgene.fromNeuronID())]);
                                    }
                                }
                            } else {
                                // mate the neurons
                                NeuronGene firstNeuron = dad.neuronGenes_[dad.getNeuronIndex(selectedgene.fromNeuronID())];
                                NeuronGene secondNeuron = neuronGenes_[getNeuronIndex(selectedgene.fromNeuronID())];
                                firstNeuron.mateTraits(secondNeuron.traits_, rng);
                                baby.neuronGenes_.emplace_back(firstNeuron);
                            }
                        } else {
                            // add dad's neuron to the baby
                            baby.neuronGenes_.emplace_back(dad.neuronGenes_[dad.getNeuronIndex(selectedgene.fromNeuronID())]);
                        }
                    }

                    // To
                    if ((!baby.hasNeuronID(selectedgene.toNeuronID())) && (dad.hasNeuronID(selectedgene.toNeuronID()))) {
                        // See if mom has the same neuron
                        if (hasNeuronID(selectedgene.toNeuronID())) {
                            if (rng.randFloat() < parameters.multipointCrossoverRate) {
                                if (rng.randFloat() < parameters.preferFitterParentRate) {
                                    if (getFitness() < dad.getFitness()) {
                                        // add dad's neuron to the baby
                                        baby.neuronGenes_.emplace_back(dad.neuronGenes_[dad.getNeuronIndex(selectedgene.toNeuronID())]);
                                    } else {
                                        // add mom's neuron to the baby
                                        baby.neuronGenes_.emplace_back(neuronGenes_[getNeuronIndex(selectedgene.toNeuronID())]);
                                    }
                                } else {
                                    // if so, then choose randomly which neuron the baby shoud inherit
                                    if (rng.randFloat() < 0.5) {
                                        // add dad's neuron to the baby
                                        baby.neuronGenes_.emplace_back(dad.neuronGenes_[dad.getNeuronIndex(selectedgene.toNeuronID())]);
                                    } else {
                                        // add mom's neuron to the baby
                                        baby.neuronGenes_.emplace_back(neuronGenes_[getNeuronIndex(selectedgene.toNeuronID())]);
                                    }
                                }
                            } else {
                                // mate neurons
                                NeuronGene firstNeuron = dad.neuronGenes_[dad.getNeuronIndex(selectedgene.toNeuronID())];
                                NeuronGene secondNeuron = neuronGenes_[getNeuronIndex(selectedgene.toNeuronID())];
                                firstNeuron.mateTraits(secondNeuron.traits_, rng);
                                baby.neuronGenes_.emplace_back(firstNeuron);
                            }
                        } else {
                            // add dad's neuron to the baby
                            baby.neuronGenes_.emplace_back(dad.neuronGenes_[dad.getNeuronIndex(selectedgene.toNeuronID())]);
                        }
                    }
                }
            }
        }  // end while

        baby.numInputs_ = numInputs_;
        baby.numOutputs_ = numOutputs_;

        // Sort the baby's genes
        baby.sortGenes();

        return baby;
    }

    // Sorts the genes of the genome The neurons by IDs and the links by innovation numbers.
    bool neuronCompare(NeuronGene &ls, NeuronGene &rs) { return ls.id() < rs.id(); }

    bool linkCompare(LinkGene &ls, LinkGene &rs) { return ls.innovationID() < rs.innovationID(); }

    void Genome::sortGenes() {
        std::sort(neuronGenes_.begin(), neuronGenes_.end(), neuronCompare);
        std::sort(linkGenes_.begin(), linkGenes_.end(), linkCompare);
    }

    unsigned int Genome::neuronDepth(int neuronID, unsigned int depth) {
        unsigned int currentDepth;
        unsigned int maxDepth = depth;

        if (depth > 16384) {
            // oops! a possible loop in the network!
            return 16384;
        }

        // Base case
        if ((getNeuronByID(neuronID).type() == INPUT) || (getNeuronByID(neuronID).type() == BIAS)) {
            return depth;
        }

        // Find all links outputting to this neuron ID
        std::vector<int> inputtingLinksIndex;
        for (unsigned int i = 0; i < numLinks(); i++) {
            if (linkGenes_[i].toNeuronID() == neuronID) inputtingLinksIndex.emplace_back(i);
        }

        // For all incoming links..
        for (unsigned int i = 0; i < inputtingLinksIndex.size(); i++) {
            LinkGene link = getLinkByIndex(inputtingLinksIndex[i]);

            // RECURSION
            currentDepth = neuronDepth(link.fromNeuronID(), depth + 1);
            if (currentDepth > maxDepth) maxDepth = currentDepth;
        }

        return maxDepth;
    }

    void Genome::calculateDepth() {
        unsigned int maxDepth = 0;
        unsigned int curDepth = 0;

        // The quick case - if no hidden neurons,
        // the depth is 1
        if (numNeurons() == (numInputs_ + numOutputs_)) {
            depth_ = 1;
            return;
        }

        // make a list of all output IDs
        std::vector<int> outputIds;
        for (unsigned int i = 0; i < numNeurons(); i++) {
            if (neuronGenes_[i].type() == OUTPUT) {
                outputIds.emplace_back(neuronGenes_[i].id());
            }
        }

        // For each output
        for (unsigned int i = 0; i < outputIds.size(); i++) {
            curDepth = neuronDepth(outputIds[i], 0);

            if (curDepth > maxDepth) maxDepth = curDepth;
        }

        depth_ = maxDepth;
    }

    //////////////////////////////////////////////////////////////////////////////////
    // Saving/Loading methods
    //////////////////////////////////////////////////////////////////////////////////

    // Builds this genome from a file
    Genome::Genome(const char *fileName) {
        std::ifstream dataFile(fileName);
        *this = Genome(dataFile);
        dataFile.close();
    }

    // Builds the genome from an *opened* file
    Genome::Genome(std::ifstream &dataFile) {
        std::string str;

        if (!dataFile) {
            std::ostringstream tStream;
            tStream << "Genome file error!" << '\n';
            throw std::runtime_error("Genome file error!");
        }

        // search for GenomeStart (guard against EOF: a stream extraction failure
        // leaves t_Str unchanged, so without the eof check this loop never ends)
        do {
            dataFile >> str;
            if (dataFile.eof()) {
                throw std::runtime_error("Genome file error: GenomeStart not found!");
            }
        } while (str != "GenomeStart");

        // read the genome ID
        unsigned int gid;
        dataFile >> gid;
        id_ = gid;

        // read the genome until GenomeEnd is encountered
        do {
            dataFile >> str;
            if (dataFile.eof()) {
                throw std::runtime_error("Genome file error: GenomeEnd not found!");
            }

            if (str == "Neuron") {
                int id, type, activationfunc;
                Real splity, a, b, timeconst, bias;

                dataFile >> id;
                dataFile >> type;
                dataFile >> splity;

                dataFile >> activationfunc;
                dataFile >> a;
                dataFile >> b;
                dataFile >> timeconst;
                dataFile >> bias;

                // TODO read neuron traits

                NeuronGene neuron(static_cast<NeuronType>(type), id, splity);
                neuron.init(a, b, timeconst, bias, static_cast<ActivationFunction>(activationfunc));

                neuronGenes_.emplace_back(neuron);
            }

            if (str == "Link") {
                int from, to, innov, isrecur;
                Real weight;

                dataFile >> from;
                dataFile >> to;
                dataFile >> innov;
                dataFile >> isrecur;
                dataFile >> weight;

                // TODO read link traits

                linkGenes_.emplace_back(LinkGene(from, to, innov, weight, static_cast<bool>(isrecur)));
            }
        } while (str != "GenomeEnd");

        // Init additional stuff
        // count inputs/outputs
        numInputs_ = 0;
        numOutputs_ = 0;
        for (unsigned int i = 0; i < numNeurons(); i++) {
            if ((neuronGenes_[i].type() == INPUT) || (neuronGenes_[i].type() == BIAS)) {
                numInputs_++;
            }

            if (neuronGenes_[i].type() == OUTPUT) {
                numOutputs_++;
            }
        }

        fitness_ = 0.0;
        adjustedFitness_ = 0.0;
        offspringAmount_ = 0.0;
        depth_ = 0;
        phenotypeBehavior_ = nullptr;
        evaluated_ = false;
    }

    // Saves this genome to a file
    void Genome::save(const char *fileName) {
        FILE *file;
        file = fopen(fileName, "w");
        save(file);
        fclose(file);
    }

    // Saves this genome to an already opened file for writing
    void Genome::save(FILE *file) {
        fprintf(file, "GenomeStart %d\n", getID());

        // loop over the neurons and save each one
        for (unsigned int i = 0; i < numNeurons(); i++) {
            // Save neuron
            fprintf(file, "Neuron %d %d %3.8f %d %3.8f %3.8f %3.8f %3.8f\n", neuronGenes_[i].id(), static_cast<int>(neuronGenes_[i].type()),
                    neuronGenes_[i].splitY(), static_cast<int>(neuronGenes_[i].actFunction_), neuronGenes_[i].a_, neuronGenes_[i].b_,
                    neuronGenes_[i].timeConstant_, neuronGenes_[i].bias_);
            // TODO write neuron traits
        }

        // loop over the connections and save each one
        for (unsigned int i = 0; i < numLinks(); i++) {
            fprintf(file, "Link %d %d %d %d %3.8f\n", linkGenes_[i].fromNeuronID(), linkGenes_[i].toNeuronID(), linkGenes_[i].innovationID(),
                    static_cast<int>(linkGenes_[i].isRecurrent()), linkGenes_[i].getWeight());
            // TODO write link traits
        }

        fprintf(file, "GenomeEnd\n\n");
    }

    void Genome::printTraits(std::map<std::string, Trait> &traits) {
        for (std::map<std::string, Trait>::iterator t = traits.begin(); t != traits.end(); t++) {
            bool doit = false;
            std::string s = t->second.depKey;
            // std::string sv = bs::get<std::string>(t->second.dep_values);
            if (s != "") {
                // there is such trait..
                if (traits.count(s) != 0) {
                    /*int a; Real b; std::string c;
                    if ((*it).m_Traits[s].value.type() == typeid(int))
                        a = bs::get<int>((*it).m_Traits[s].value);
                    if ((*it).m_Traits[s].value.type() == typeid(Real))
                        b = bs::get<Real>((*it).m_Traits[s].value);
                    if ((*it).m_Traits[s].value.type() == typeid(std::string))
                        c = bs::get<std::string>((*it).m_Traits[s].value);

                    int a1; Real b1; std::string c1;
                    if ((t->second.dep_values).type() == typeid(int))
                        a1 = bs::get<int>((t->second.dep_values));
                    if ((t->second.dep_values).type() == typeid(Real))
                        b1 = bs::get<Real>((t->second.dep_values));
                    if ((t->second.dep_values).type() == typeid(std::string))
                        c1 = bs::get<std::string>((t->second.dep_values));*/

                    // and it has the right value?
                    for (int ix = 0; ix < t->second.depValues.size(); ix++) {
                        if (traits[s].value == (t->second.depValues[ix])) {
                            doit = true;
                            break;
                        }
                    }
                }
            } else {
                doit = true;
            }

            if (doit) {
                std::cout << t->first << " - ";
                if (std::holds_alternative<int>(t->second.value)) {
                    std::cout << std::get<int>(t->second.value);
                }
                if (std::holds_alternative<Real>(t->second.value)) {
                    std::cout << std::get<Real>(t->second.value);
                }
                if (std::holds_alternative<std::string>(t->second.value)) {
                    std::cout << "\"" << std::get<std::string>(t->second.value) << "\"";
                }
                if (std::holds_alternative<IntSetElement>(t->second.value)) {
                    std::cout << (std::get<IntSetElement>(t->second.value)).value;
                }
                if (std::holds_alternative<FloatSetElement>(t->second.value)) {
                    std::cout << (std::get<FloatSetElement>(t->second.value)).value;
                }

                std::cout << ", ";
            }
        }
    }

    void Genome::printAllTraits() {
        std::cout << "====================================================================\n";
        std::cout << "Genome:\n"
                  << "==================================\n";
        printTraits(genomeGene_.traits_);

        std::cout << "\n";

        std::cout << "====================================================================\n";
        std::cout << "Neurons:\n"
                  << "==================================\n";
        for (std::vector<NeuronGene>::iterator it = neuronGenes_.begin(); it != neuronGenes_.end(); it++) {
            std::cout << "ID: " << it->id() << " : ";
            printTraits((*it).traits_);

            std::cout << "\n";
        }
        std::cout << "==================================\n";

        std::cout << "Links:\n"
                  << "==================================\n";
        for (std::vector<LinkGene>::iterator it = linkGenes_.begin(); it != linkGenes_.end(); it++) {
            std::cout << "ID: " << it->innovationID() << " : ";
            printTraits((*it).traits_);
            std::cout << "\n";
        }
        std::cout << "==================================\n";
        std::cout << "====================================================================\n";
    }

    ////////////////////////////////////////////
    // Evovable Substrate Hyper NEAT.
    // For more info on the algorithm check: http://eplex.cs.ucf.edu/ESHyperNEAT/
    ////////////////////////////////////////////

#if 0
    
    //divide and init for n dimensions
    
    void Genome::buildESHyperNEATPhenotypeND(NeuralNetwork &net, Substrate &subst, Parameters &params)
    {
        ASSERT(subst.inputCoords_.size() > 0);
        ASSERT(subst.outputCoords_.size() > 0);

        unsigned int inputCount = subst.inputCoords_.size();
        unsigned int outputCount = subst.outputCoords_.size();
        unsigned int hiddenIndex = inputCount + outputCount;
        unsigned int sourceIndex = 0;
        unsigned int targetIndex = 0;
        unsigned int hiddenCounter = 0;
        unsigned int maxNodes = std::pow(4, params.maxDepth);
        unsigned int coordLen = subst.inputCoords_.at(0).size();
        std::vector<TempConnection> TempConnections;
        TempConnections.reserve(maxNodes + 1);

        std::vector<Real> point;
        
        point.reserve(coordLen);
        
        boost::shared_ptr<NTree> root;

        boost::unordered_map<std::vector<Real>, int> hiddenNodes;
        hiddenNodes.reserve(maxNodes);

        boost::unordered_map<std::vector<Real>, int> temp;
        temp.reserve(maxNodes);

        boost::unordered_map<std::vector<Real>, int> unexploredNodes;
        unexploredNodes.reserve(maxNodes);

        net.neurons_.reserve(maxNodes);
        net.connections_.reserve((maxNodes * (maxNodes - 1)) / 2);
        net.setInputOutputDimensions(static_cast<unsigned short>(inputCount),
                                     static_cast<unsigned short>(outputCount));


        NeuralNetwork tempPhenotype(true);
        buildPhenotype(tempPhenotype);

        // Find Inputs to Hidden connections.
        for (unsigned int i = 0; i < inputCount; i++)
        {
            // Get the nTree
            std::vector <Real> rootCoord;
            rootCoord.reserve(coordLen);
            for(unsigned int cLen = 0; cLen < coordLen; cLen++)
            {
                rootCoord.push(0.0);
            }
            root = boost::shared_ptr<NTree>(
                    new NTree(params.nTreeCoord, params.width, params.height, 1));
            divideInitializeND(subst.inputCoords_[i], root, tempPhenotype, params, true, 0.0);
            TempConnections.clear();
            pruneExpressND(subst.inputCoords_[i], root, tempPhenotype, params, TempConnections, true);

            for (unsigned int j = 0; j < TempConnections.size(); j++)
            {
                if (std::std::abs(TempConnections[j].weight * subst.maxWeightAndBias_) <
                    0.2/*subst.m_link_threshold*/) // TODO: fix this
                    continue;

                // Find the hidden node in the hidden nodes. If it is not there add it.
                if (hiddenNodes.find(TempConnections[j].target) == hiddenNodes.end())
                {
                    targetIndex = hiddenCounter++;
                    hiddenNodes.emplace(TempConnections[j].target, targetIndex));
                }
                    // Add connection
                else
                {
                    targetIndex = hiddenNodes.find(TempConnections[j].target)->second;
                }

                Connection tc;
                tc.sourceNeuronIndex_ = i;
                tc.targetNeuronIndex_ = targetIndex + hiddenIndex;
                tc.weight_ = TempConnections[j].weight * subst.maxWeightAndBias_;
                tc.recurFlag_ = false;

                net.connections_.push_back(tc);

            }
        }
        // Hidden to hidden. Basically the same procedure as above repeated IterationLevel times (see the params)
        unexploredNodes = hiddenNodes;
        for (unsigned int i = 0; i < params.iterationLevel; i++)
        {
            boost::unordered_map<std::vector<Real>, int>::iterator itrHid;
            for (itrHid = unexploredNodes.begin(); itrHid != unexploredNodes.end(); itrHid++)
            {
                root = boost::shared_ptr<NTree>(
                        new NTree(params.nTreeCoord, params.width, params.height, 1));
                divideInitializeND(itrHid->first, root, tempPhenotype, params, true, 0.0);
                TempConnections.clear();
                pruneExpress(itrHid->first, root, tempPhenotype, params, TempConnections, true);
                //root.reset();

                for (unsigned int k = 0; k < TempConnections.size(); k++)
                {
                    if (std::std::abs(TempConnections[k].weight * subst.maxWeightAndBias_) <
                        0.2/*subst.m_link_threshold*/) // TODO: fix this
                        continue;

                    if (hiddenNodes.find(TempConnections[k].target) == hiddenNodes.end())
                    {
                        targetIndex = hiddenCounter++;
                        hiddenNodes.emplace(TempConnections[k].target, targetIndex));
                    }
                    else if(!params.feedForward) // TODO: This can be skipped if building a feed forwad network.
                    {
                        targetIndex = hiddenNodes.find(TempConnections[k].target)->second;
                    }

                    Connection tc;
                    tc.sourceNeuronIndex_ = itrHid->second + hiddenIndex;  // NO!!!
                    tc.targetNeuronIndex_ = targetIndex + hiddenIndex;
                    tc.weight_ = TempConnections[k].weight * subst.maxWeightAndBias_;
                    tc.recurFlag_ = false;

                    net.connections_.push_back(tc);

                }
            }
            // Now get the newly discovered hidden nodes
            boost::unordered_map<std::vector<Real>, int>::iterator itr1;
            for (itr1 = hiddenNodes.begin(); itr1 != hiddenNodes.end(); itr1++)
            {
                if (unexploredNodes.find(itr1->first) == unexploredNodes.end())
                {
                    temp.emplace(itr1->first, itr1->second));
                }
            }
            unexploredNodes = temp;
        }

        // Finally Output to Hidden. Note that unlike before, here we connect the outputs to existing hidden nodes and no new nodes are added.
        for (unsigned int i = 0; i < outputCount; i++)
        {
            root = boost::shared_ptr<NTree>(
                    new NTree(params.nTreeCoord, params.width, params.height, 1));
            divideInitialize(subst.outputCoords_[i], root, tempPhenotype, params, false, 0.0);
            TempConnections.clear();
            pruneExpress(subst.outputCoords_[i], root, tempPhenotype, params, TempConnections, false);

            for (unsigned int j = 0; j < TempConnections.size(); j++)
            {
                // Make sure the link weight is above the expected threshold.
                if (std::std::abs(TempConnections[j].weight * subst.maxWeightAndBias_) <
                    0.2 /*subst.m_link_threshold*/) // TODO: fix this
                    continue;

                if (hiddenNodes.find(TempConnections[j].source) != hiddenNodes.end())
                {
                    sourceIndex = hiddenNodes.find(TempConnections[j].source)->second;

                    Connection tc;
                    tc.sourceNeuronIndex_ = sourceIndex + hiddenIndex;
                    tc.targetNeuronIndex_ = i + inputCount;

                    tc.weight_ = TempConnections[j].weight * subst.maxWeightAndBias_;
                    tc.recurFlag_ = false;

                    net.connections_.push_back(tc);
                }
            }
        }
        // Add the neurons.Input first, followed by bias, output and hidden. In this order.

        for (unsigned int i = 0; i < inputCount - 1; i++)
        {
            Neuron n;
            n.a_ = 1;
            n.b_ = 0;
            n.substrateCoords_ = subst.inputCoords_[i];
            n.activationFunctionType_ = NEAT::LINEAR;
            n.type_ = NEAT::INPUT;
            net.neurons_.push_back(n);
        }
        // Bias n.
        Neuron n;
        n.a_ = 1;
        n.b_ = 0;
        n.substrateCoords_ = subst.inputCoords_[inputCount - 1];
        n.activationFunctionType_ = NEAT::LINEAR;
        n.type_ = NEAT::BIAS;
        net.neurons_.push_back(n);

        for (unsigned int i = 0; i < outputCount; i++)
        {
            Neuron n;
            n.a_ = 1;
            n.b_ = 0;
            n.substrateCoords_ = subst.outputCoords_[i];
            n.activationFunctionType_ = subst.outputNodesActivation_;
            n.type_ = NEAT::OUTPUT;
            net.neurons_.push_back(n);
        }

        boost::unordered_map<std::vector<Real>, int>::iterator itr;
        for (itr = hiddenNodes.begin(); itr != hiddenNodes.end(); itr++)
        {
            Neuron n;
            n.a_ = 1;
            n.b_ = 0;
            n.substrateCoords_ = itr->first;

            ASSERT(n.substrateCoords_.size() > 0); // prevent 0D points
            n.activationFunctionType_ = subst.hiddenNodesActivation_;
            n.type_ = NEAT::HIDDEN;
            net.neurons_.push_back(n);
        }

        // Clean the generated network from dangling connections and we're good to go.
        cleanNet(net.connections_, inputCount, outputCount, hiddenNodes.size());
    }
    
    void Genome::buildESHyperNEATPhenotype(NeuralNetwork &net, Substrate &subst, Parameters &params)
    {
        ASSERT(subst.inputCoords_.size() > 0);
        ASSERT(subst.outputCoords_.size() > 0);

        unsigned int inputCount = subst.inputCoords_.size();
        unsigned int outputCount = subst.outputCoords_.size();
        unsigned int hiddenIndex = inputCount + outputCount;
        unsigned int sourceIndex = 0;
        unsigned int targetIndex = 0;
        unsigned int hiddenCounter = 0;
        unsigned int maxNodes = std::pow(4, params.maxDepth);

        std::vector<TempConnection> TempConnections;
        TempConnections.reserve(maxNodes + 1);

        std::vector<Real> point;
        point.reserve(3);

        boost::shared_ptr<QuadPoint> root;

        boost::unordered_map<std::vector<Real>, int> hiddenNodes;
        hiddenNodes.reserve(maxNodes);

        boost::unordered_map<std::vector<Real>, int> temp;
        temp.reserve(maxNodes);

        boost::unordered_map<std::vector<Real>, int> unexploredNodes;
        unexploredNodes.reserve(maxNodes);

        net.neurons_.reserve(maxNodes);
        net.connections_.reserve((maxNodes * (maxNodes - 1)) / 2);
        net.setInputOutputDimensions(static_cast<unsigned short>(inputCount),
                                     static_cast<unsigned short>(outputCount));


        NeuralNetwork tempPhenotype(true);
        buildPhenotype(tempPhenotype);

        // Find Inputs to Hidden connections.
        for (unsigned int i = 0; i < inputCount; i++)
        {
            // Get the Quadtree and express the connections in it for this input
            root = boost::shared_ptr<QuadPoint>(
                    new QuadPoint(params.qtreeX, params.qtreeY, params.width, params.height, 1));
            divideInitialize(subst.inputCoords_[i], root, tempPhenotype, params, true, 0.0);
            TempConnections.clear();
            pruneExpress(subst.inputCoords_[i], root, tempPhenotype, params, TempConnections, true);

            for (unsigned int j = 0; j < TempConnections.size(); j++)
            {
                if (std::std::abs(TempConnections[j].weight * subst.maxWeightAndBias_) <
                    0.2/*subst.m_link_threshold*/) // TODO: fix this
                    continue;

                // Find the hidden node in the hidden nodes. If it is not there add it.
                if (hiddenNodes.find(TempConnections[j].target) == hiddenNodes.end())
                {
                    targetIndex = hiddenCounter++;
                    hiddenNodes.emplace(TempConnections[j].target, targetIndex));
                }
                    // Add connection
                else
                {
                    targetIndex = hiddenNodes.find(TempConnections[j].target)->second;
                }

                Connection tc;
                tc.sourceNeuronIndex_ = i;
                tc.targetNeuronIndex_ = targetIndex + hiddenIndex;
                tc.weight_ = TempConnections[j].weight * subst.maxWeightAndBias_;
                tc.recurFlag_ = false;

                net.connections_.emplace_back(tc);

            }
        }
        // Hidden to hidden. Basically the same procedure as above repeated IterationLevel times (see the params)
        unexploredNodes = hiddenNodes;
        for (unsigned int i = 0; i < params.iterationLevel; i++)
        {
            boost::unordered_map<std::vector<Real>, int>::iterator itrHid;
            for (itrHid = unexploredNodes.begin(); itrHid != unexploredNodes.end(); itrHid++)
            {
                root = boost::shared_ptr<QuadPoint>(
                        new QuadPoint(params.qtreeX, params.qtreeY, params.width, params.height, 1));
                divideInitialize(itrHid->first, root, tempPhenotype, params, true, 0.0);
                TempConnections.clear();
                pruneExpress(itrHid->first, root, tempPhenotype, params, TempConnections, true);
                //root.reset();

                for (unsigned int k = 0; k < TempConnections.size(); k++)
                {
                    if (std::std::abs(TempConnections[k].weight * subst.maxWeightAndBias_) <
                        0.2/*subst.m_link_threshold*/) // TODO: fix this
                        continue;

                    if (hiddenNodes.find(TempConnections[k].target) == hiddenNodes.end())
                    {
                        targetIndex = hiddenCounter++;
                        hiddenNodes.emplace(TempConnections[k].target, targetIndex));
                    }
                    else // TODO: This can be skipped if building a feed forwad network.
                    {
                        targetIndex = hiddenNodes.find(TempConnections[k].target)->second;
                    }

                    Connection tc;
                    tc.sourceNeuronIndex_ = itrHid->second + hiddenIndex;  // NO!!!
                    tc.targetNeuronIndex_ = targetIndex + hiddenIndex;
                    tc.weight_ = TempConnections[k].weight * subst.maxWeightAndBias_;
                    tc.recurFlag_ = false;

                    net.connections_.emplace_back(tc);

                }
            }
            // Now get the newly discovered hidden nodes
            boost::unordered_map<std::vector<Real>, int>::iterator itr1;
            for (itr1 = hiddenNodes.begin(); itr1 != hiddenNodes.end(); itr1++)
            {
                if (unexploredNodes.find(itr1->first) == unexploredNodes.end())
                {
                    temp.emplace(itr1->first, itr1->second));
                }
            }
            unexploredNodes = temp;
        }

        // Finally Output to Hidden. Note that unlike before, here we connect the outputs to existing hidden nodes and no new nodes are added.
        for (unsigned int i = 0; i < outputCount; i++)
        {
            root = boost::shared_ptr<QuadPoint>(
                    new QuadPoint(params.qtreeX, params.qtreeY, params.width, params.height, 1));
            divideInitialize(subst.outputCoords_[i], root, tempPhenotype, params, false, 0.0);
            TempConnections.clear();
            pruneExpress(subst.outputCoords_[i], root, tempPhenotype, params, TempConnections, false);

            for (unsigned int j = 0; j < TempConnections.size(); j++)
            {
                // Make sure the link weight is above the expected threshold.
                if (std::std::abs(TempConnections[j].weight * subst.maxWeightAndBias_) <
                    0.2 /*subst.m_link_threshold*/) // TODO: fix this
                    continue;

                if (hiddenNodes.find(TempConnections[j].source) != hiddenNodes.end())
                {
                    sourceIndex = hiddenNodes.find(TempConnections[j].source)->second;

                    Connection tc;
                    tc.sourceNeuronIndex_ = sourceIndex + hiddenIndex;
                    tc.targetNeuronIndex_ = i + inputCount;

                    tc.weight_ = TempConnections[j].weight * subst.maxWeightAndBias_;
                    tc.recurFlag_ = false;

                    net.connections_.emplace_back(tc);
                }
            }
        }
        // Add the neurons.Input first, followed by bias, output and hidden. In this order.

        for (unsigned int i = 0; i < inputCount - 1; i++)
        {
            Neuron n;
            n.a_ = 1;
            n.b_ = 0;
            n.substrateCoords_ = subst.inputCoords_[i];
            n.activationFunctionType_ = NEAT::LINEAR;
            n.type_ = NEAT::INPUT;
            net.neurons_.emplace_back(n);
        }
        // Bias n.
        Neuron n;
        n.a_ = 1;
        n.b_ = 0;
        n.substrateCoords_ = subst.inputCoords_[inputCount - 1];
        n.activationFunctionType_ = NEAT::LINEAR;
        n.type_ = NEAT::BIAS;
        net.neurons_.emplace_back(n);

        for (unsigned int i = 0; i < outputCount; i++)
        {
            Neuron n;
            n.a_ = 1;
            n.b_ = 0;
            n.substrateCoords_ = subst.outputCoords_[i];
            n.activationFunctionType_ = subst.outputNodesActivation_;
            n.type_ = NEAT::OUTPUT;
            net.neurons_.emplace_back(n);
        }

        boost::unordered_map<std::vector<Real>, int>::iterator itr;
        for (itr = hiddenNodes.begin(); itr != hiddenNodes.end(); itr++)
        {
            Neuron n;
            n.a_ = 1;
            n.b_ = 0;
            n.substrateCoords_ = itr->first;

            ASSERT(n.substrateCoords_.size() > 0); // prevent 0D points
            n.activationFunctionType_ = subst.hiddenNodesActivation_;
            n.type_ = NEAT::HIDDEN;
            net.neurons_.emplace_back(n);
        }

        // Clean the generated network from dangling connections and we're good to go.
        cleanNet(net.connections_, inputCount, outputCount, hiddenNodes.size());
    }
    // uses n dimensional sub division tree to determine placement of hidden nodes in the substrate
    void Genome::divideInitializeND(const std::vector<Real> &node,
                                  boost::shared_ptr<NTree> &root,
                                  NeuralNetwork &cppn,
                                  Parameters &params,
                                  const bool &outgoing)
    {
        int cppDepth = 8;
        
        // some of the division, the permutation of center points in particular has been included with the tree struct and will simply be called here
        std::vector<Real> inputs;
        
        boost::shared_ptr<NTree> p;
        std::queue<boost::shared_ptr<NTree> > q;
        q.push(p);
        while(!q.empty())
        {
            p = q.front();
            p.setChildren();
            for (unsigned int i = 0; i < p->children.size(); i++)
            {
                inputs.clear();
                inputs.reserve(cppn.numInputs());
                if(outgoing)
                {
                    inputs = node;
                    for(unsigned int ci = 0; ci < node.size(); i++)
                    {
                        inputs.push_back(p->children[i]->coord[ci]);
                    }
                }
                else
                {
                    inputs = p->children[i]->coord;
                    for(unsigned int ci = 0; ci < node.size(); i++)
                    {
                        inputs.push_back(node[ci]);
                    }
                }
                inputs[inputs.size() - 1] = (params.cppnBias);
                                cppn.flush();
                cppn.input(inputs);

                for (int d = 0; d < cppnDepth; d++)
                {
                    cppn.activate();
                }
                p->children[i]->weight = cppn.output()[0];
                if (params.leo)
                {
                    p->children[i]->leo = cppn.output()[cppn.output().size() - 1];
                }
                cppn.flush();

            }

            if ((p->level < params.initialDepth) ||
                ((p->level < params.maxDepth) && variance(p) > params.divisionThreshold))
            {
                for(unsigned int addIndex = 0; addIndex < p->children.size(); addIndex)
                {
                    q.push(p->children[addIndex]);
                }
            }
            q.pop();
        }
        return;
        
    }
    // Used to determine the placement of hidden neurons in the Evolvable Substrate.
    void Genome::divideInitialize(const std::vector<Real> &node,
                                  boost::shared_ptr<QuadPoint> &root,
                                  NeuralNetwork &cppn,
                                  Parameters &params,
                                  const bool &outgoing,
                                  const Real &zCoord)
    {   // Have to check if this actually does something useful here
        //CalculateDepth();
        int cppnDepth = 8;//GetDepth();

        std::vector<Real> inputs;

        // Standard Tree stuff. Create children, check their output with the CPPN and if they have higher variance add them to their parent. Repeat with the
        // children until maxDepth has been reached or if the variance isn't high enough.
        boost::shared_ptr<QuadPoint> p;

        std::queue<boost::shared_ptr<QuadPoint> > q;
        q.push(root);
        while (!q.empty())
        {
            p = q.front();
            // Add children
            p->children.emplace_back(boost::shared_ptr<QuadPoint>(
                    new QuadPoint(p->x - p->width / 2, p->y - p->height / 2, p->width / 2, p->height / 2,
                                  p->level + 1)));
            p->children.emplace_back(boost::shared_ptr<QuadPoint>(
                    new QuadPoint(p->x - p->width / 2, p->y + p->height / 2, p->width / 2, p->height / 2,
                                  p->level + 1)));
            p->children.emplace_back(boost::shared_ptr<QuadPoint>(
                    new QuadPoint(p->x + p->width / 2, p->y + p->height / 2, p->width / 2, p->height / 2,
                                  p->level + 1)));
            p->children.emplace_back(boost::shared_ptr<QuadPoint>(
                    new QuadPoint(p->x + p->width / 2, p->y - p->height / 2, p->width / 2, p->height / 2,
                                  p->level + 1)));

            for (unsigned int i = 0; i < p->children.size(); i++)
            {
                inputs.clear();
                inputs.reserve(cppn.numInputs());

                if (outgoing)
                {
                    // node goes here
                    inputs = node;

                    inputs.emplace_back(p->children[i]->x);
                    inputs.emplace_back(p->children[i]->y);
                    inputs.emplace_back(p->children[i]->z);
                }

                else
                {
                    // QuadPoint goes first
                    inputs.emplace_back(p->children[i]->x);
                    inputs.emplace_back(p->children[i]->y);
                    inputs.emplace_back(p->children[i]->z);

                    inputs.emplace_back(node[0]);
                    inputs.emplace_back(node[1]);
                    inputs.emplace_back(node[2]);
                }

                // Bias
                inputs[inputs.size() - 1] = (params.cppnBias);

                cppn.flush();
                cppn.input(inputs);

                for (int d = 0; d < cppnDepth; d++)
                {
                    cppn.activate();
                }
                p->children[i]->weight = cppn.output()[0];
                if (params.leo)
                {
                    p->children[i]->leo = cppn.output()[cppn.output().size() - 1];
                }
                cppn.flush();

            }

            if ((p->level < params.initialDepth) ||
                ((p->level < params.maxDepth) && variance(p) > params.divisionThreshold))
            {
                for (unsigned int i = 0; i < 4; i++)
                {
                    q.push(p->children[i]);
                }
            }
            q.pop();

        }

        return;
    }

    void Genome::pruneExpressND(const std::vector<Real> &node,
                              boost::shared_ptr<NTree> &root,
                              NeuralNetwork &cppn,
                              Parameters &params,
                              std::vector<Genome::TempConnection> &connections,
                              const bool &outgoing)
    {
        if (root->children[0] == nullptr)
        {
            return;
        }

        else
        {
            for (unsigned int i = 0; i < root->children.size(); i++)
            {
                if(variance(root->children[i]) > params.varianceThreshold)
                {
                    pruneExpressND(node, root->children[i], cppn, params, connections, outgoing);
                }
                
                else if(!params.leo || (params.leo && root->children[i]->leo > params.leoThreshold))
                {
                    int cppDepth = 8; //seems to be hard coded across the codebase, seems like plenty of depth to me!
                    std::vector<Real> childArray;
                    for(unsigned int cIx = 0; cIx < root->children[i]->coord.size(); cIx++)
                    {
                        std::vector<Real> fullIn;
                        std::vector<Real> fullIn2;
                        std::vector<Real> inputs2;
                        std::vector<Real> inputs;
                        int rootIndex = 0;
                        int sign = -1;
                        Real dimenSplit1 = root->children[i]->coord[cIx] - root->width;
                        Real dimenSplit2 = root->children[i]->coord[cIx] + root->width;
                        for(unsigned int c2Ix = 0; c2Ix < node.size(); c2Ix++)
                        {
                            if(c2Ix == cIx)
                            {
                                inputs.append(root->children[i].coord.at(c2Ix));
                                inputs2.append(root->children[i]->coord.at(c2Ix));
                            } else {
                                inputs.append(dimenSplit2);
                                inputs2.append(dimenSplit1);
                            }
                        }
                        if(outgoing)
                        {
                            fullIn = node;
                            fullIn2 = fullIn;
                            fulllIn.insert(fullIn.end(), inputs.begin(), inputs.end());
                            fullIn2.insert(fullIn2.end(), inputs2.begin(), inputs2.end());
                        }
                        else
                        {
                            fullIn2 = inputs2;
                            fullIn = inputs;
                            fullIn2.insert(fullIn2.end(), node.begin(), node.end());
                            fullIn.insert(fullIn.end(), node.begin(), node.end());
                        }
                        fullIn.push_back(params.cppnBias);
                        fullIn2.push_back(params.cppnBias);
                        cppn.inputs(fullIn);
                        childArray.append(cppn.activate()[0]);
                        for (int d = 0; d < cppnDepth; d++)
                        {
                            cppn.activate();
                        }
                        childArray.append(std::abs(root->child[i]->weight - output()[0]));
                        cppn.flush();
                        cppn.inputs(fullIn2);
                        childArray.append(cppn.activate()[0]);
                        for (int d = 0; d < cppnDepth; d++)
                        {
                            cppn.activate();
                        }
                        childArray.append(std::abs(root->child[i]->weight - output()[0]));
                    }
                    Real biggestSmallest = std::min(childArray[0], childArray[1]);
                    unsigned int pairIndex = 2;
                    while(pairIndex < childArray.size()/2)
                    {
                        unsigned int newMin = std::min(childArray[pairIndex], childArray[pairIndex + 1]);
                        if(newMin > biggestSmallest)
                        {
                            biggestSmallest = newMin;
                        }
                        pairIndex += 2;
                    }
                    if(biggestSmallest > params.bandThreshold)
                    {
                        if(outgoing)
                        {
                            TempConnection tc(node, root->children[i]->coord, root->children[i]->weight, node.size());
                        }
                        else
                        {
                            TempConnection tc(root->children[i]->coord, node, root->children[i]->weight, node.size());
                        }
                        connections.push_back(tc);
                    }
                }
            }
        }
    // We take the tree generated above and see which connections can be expressed on the basis of Variance threshold,
    // Band threshold and LEO.
    void Genome::pruneExpress(const std::vector<Real> &node,
                              boost::shared_ptr<QuadPoint> &root,
                              NeuralNetwork &cppn,
                              Parameters &params,
                              std::vector<Genome::TempConnection> &connections,
                              const bool &outgoing)
    {
        if (root->children[0] == nullptr)
        {
            return;
        }

        else
        {
            for (unsigned int i = 0; i < 4; i++)
            {
                if (variance(root->children[i]) > params.varianceThreshold)
                {
                    pruneExpress(node, root->children[i], cppn, params, connections, outgoing);
                }

                    // Band Pruning phase. If LEO is turned off this should always happen. If it is not it should only happen if the LEO output is greater than
                    // a specified threshold
                else if (!params.leo || (params.leo && root->children[i]->leo > params.leoThreshold))
                {
                    //CalculateDepth();
                    int cppnDepth = 8;//GetDepth();

                    Real dLeft, dRight, dTop, dBottom;
                    std::vector<Real> inputs;

                    int rootIndex = 0;

                    if (outgoing)
                    {
                        inputs = node;
                        inputs.emplace_back(root->children[i]->x);
                        inputs.emplace_back(root->children[i]->y);
                        inputs.emplace_back(root->children[i]->z);

                        rootIndex = node.size();
                    }

                    else
                    {
                        inputs.emplace_back(root->children[i]->x);
                        inputs.emplace_back(root->children[i]->y);
                        inputs.emplace_back(root->children[i]->z);
                        inputs.emplace_back(node[0]);
                        inputs.emplace_back(node[1]);
                        inputs.emplace_back(node[2]);
                    }

                    // Left
                    inputs.emplace_back(params.cppnBias);
                    inputs[rootIndex] -= root->width;

                    cppn.input(inputs);

                    for (int d = 0; d < cppnDepth; d++)
                    {
                        cppn.activate();
                    }

                    dLeft = std::abs(root->children[i]->weight - cppn.output()[0]);
                    cppn.flush();

                    // Right
                    inputs[rootIndex] += 2 * (root->width);
                    cppn.input(inputs);

                    for (int d = 0; d < cppnDepth; d++)
                    {
                        cppn.activate();
                    }

                    dRight = std::abs(root->children[i]->weight - cppn.output()[0]);
                    cppn.flush();

                    // Top
                    inputs[rootIndex] -= root->width;
                    inputs[rootIndex + 1] -= root->width;
                    cppn.input(inputs);

                    for (int d = 0; d < cppnDepth; d++)
                    {
                        cppn.activate();
                    }

                    dTop = std::abs(root->children[i]->weight - cppn.output()[0]);
                    cppn.flush();
                    // Bottom
                    inputs[rootIndex + 1] += 2 * root->width;
                    cppn.input(inputs);

                    for (int d = 0; d < cppnDepth; d++)
                    {
                        cppn.activate();
                    }

                    dBottom = std::abs(root->children[i]->weight - cppn.output()[0]);
                    cppn.flush();

                    if (std::max(std::min(dTop, dBottom), std::min(dLeft, dRight)) > params.bandThreshold)
                    {
                        Genome::TempConnection tc;
                        //Yeah its ugly
                        if (outgoing)
                        {
                            tc.source = node;

                            tc.target.emplace_back(root->children[i]->x);
                            tc.target.emplace_back(root->children[i]->y);
                            tc.target.emplace_back(root->children[i]->z);
                        }
                        else
                        {
                            tc.source.emplace_back(root->children[i]->x);
                            tc.source.emplace_back(root->children[i]->y);
                            tc.source.emplace_back(root->children[i]->z);

                            tc.target = node;
                        }
                        // Normalize
                        // TODO: Put in Parameters
                        tc.weight = root->children[i]->weight;
                        connections.emplace_back(tc);
                    }
                }
            }
        }
        return;
    }

    Real Genome::varianceND(boost::shared_ptr<NTree> &point){
        if(point->children.empty()){
            return 0.0;
        }
        
        boost::accumulators::accumulatorSet<Real, boost::accumulators::stats<boost::accumulators::tag::variance> > acc;
        for (unsigned int i = 0; i < point->children.size(); i++){
            acc(point->children[i]->weight);)
        }
        return boost::accumulators::variance(acc);
    }
    // Calculates the variance of a given Quadpoint. Maybe an alternative solution would be to add this in the Quadpoint const.
    Real Genome::variance(boost::shared_ptr<QuadPoint> &point)
    {
        if (point->children.empty())
        {
            return 0.0;
        }

        boost::accumulators::accumulatorSet<Real, boost::accumulators::stats<boost::accumulators::tag::variance> > acc;
        for (unsigned int i = 0; i < 4; i++)
        {
            acc(point->children[i]->weight);
        }

        return boost::accumulators::variance(acc);
    }

    // Helper method for Variance
    void Genome::collectValues(std::vector<Real> &vals, boost::shared_ptr<QuadPoint> &point)
    {
        //In theory we shouldn't get here at all.
        if (point == nullptr)
        {
            return;
        }

        if (point->children.size() > 0)
        {
            for (unsigned int i = 0; i < 4; i++)
            {
                collectValues(vals, point->children[i]);
            }
        }

        else
        {   // Here, Apparently it treats the point a if it is not initialized
            vals.emplace_back(point->weight);
        }
    }


    // Removes all the dangling connections. This still leaves the nodes though,
    void Genome::cleanNet(std::vector<Connection> &connections, unsigned int inputCount,
                           unsigned int outputCount, unsigned int hiddenCount)
    {
        bool looseConnections = true;
        int nodeCount = inputCount + outputCount + hiddenCount;
        std::vector<Connection> temp;
        temp.reserve(connections.size());
        while (looseConnections)
        {
            std::vector<bool> hasOutgoing(nodeCount, false);
            std::vector<bool> hasIncoming(nodeCount, false);
            // Make sure inputs and outputs are covered.
            for (unsigned int i = 0; i < outputCount + inputCount; i++)
            {
                hasOutgoing[i] = true;
                hasIncoming[i] = true;
            }

            // Move on to the nodes.
            for (unsigned int i = 0; i < connections.size(); i++)
            {
                if (connections[i].sourceNeuronIndex_ != connections[i].targetNeuronIndex_)
                {
                    hasOutgoing[connections[i].sourceNeuronIndex_] = true;
                    hasIncoming[connections[i].targetNeuronIndex_] = true;
                }

            }

            looseConnections = false;

            std::vector<Connection>::iterator itr;
            for (itr = connections.begin(); itr < connections.end();)
            {
                if (!hasOutgoing[itr->targetNeuronIndex_] || !hasIncoming[itr->sourceNeuronIndex_])
                {
                    itr = connections.erase(itr);
                    if (!looseConnections)
                    {
                        looseConnections = true;
                    }

                }
                else
                {
                    itr++;
                }
            }
        }
    }
#endif

}  // namespace NEAT
