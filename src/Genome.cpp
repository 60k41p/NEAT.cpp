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

#include <math.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <functional>
#include <limits>
#include <map>
#include <memory>
#include <queue>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "AssertMacros.h"
#include "FileIO.h"
#include "Parameters.h"
#include "Random.h"
#include "Serialization.h"
#include "Utils.h"

namespace NEAT {

    // forward
    ActivationFunction GetRandomActivation(const Parameters &a_Parameters, RNG &a_RNG);

    // squared x
    inline Real sqr(Real x) { return x * x; }

    // Uniform draw from [minimum .. maximum] (degenerate range returns the bound).
    inline Real RandomRange(RNG &rng, Real minimum, Real maximum) {
        if (minimum == maximum) return minimum;
        return minimum + rng.RandFloat() * (maximum - minimum);
    }

    // Initializes a new neuron gene's spiking parameters from the evolvable
    // ranges (midpoint when no RNG is given, e.g. deterministic builders).
    inline void InitializeNeuronSpiking(NeuronGene &neuron, const Parameters &parameters, RNG *rng = nullptr) {
        const auto value = [rng](Real minimum, Real maximum) {
            return rng == nullptr ? minimum + (maximum - minimum) * 0.5 : RandomRange(*rng, minimum, maximum);
        };
        if (IsSpikingActivation(neuron.m_ActFunction)) {
            neuron.m_TimeConstant = value(parameters.MinSpikingTimeConstant, parameters.MaxSpikingTimeConstant);
        }
        neuron.m_SpikeThreshold = neuron.m_ActFunction == SPIKING_IZHIKEVICH ? value(parameters.MinIzhikevichThreshold, parameters.MaxIzhikevichThreshold)
                                                                             : value(parameters.MinSpikeThreshold, parameters.MaxSpikeThreshold);
        neuron.m_ResetPotential = value(parameters.MinResetPotential, parameters.MaxResetPotential);
        neuron.m_RestingPotential = value(parameters.MinRestingPotential, parameters.MaxRestingPotential);
        neuron.m_RefractoryPeriod = value(parameters.MinRefractoryPeriod, parameters.MaxRefractoryPeriod);
        neuron.m_MembraneResistance = value(parameters.MinMembraneResistance, parameters.MaxMembraneResistance);
        neuron.m_AdaptationTimeConstant = value(parameters.MinAdaptationTimeConstant, parameters.MaxAdaptationTimeConstant);
        neuron.m_AdaptationIncrement = value(parameters.MinAdaptationIncrement, parameters.MaxAdaptationIncrement);
        neuron.m_RateTimeConstant = value(parameters.MinSpikeRateTimeConstant, parameters.MaxSpikeRateTimeConstant);
        neuron.m_IzhikevichA = value(parameters.MinIzhikevichA, parameters.MaxIzhikevichA);
        neuron.m_IzhikevichB = value(parameters.MinIzhikevichB, parameters.MaxIzhikevichB);
        neuron.m_IzhikevichC = value(parameters.MinIzhikevichC, parameters.MaxIzhikevichC);
        neuron.m_IzhikevichD = value(parameters.MinIzhikevichD, parameters.MaxIzhikevichD);
        if (neuron.m_ActFunction == MCCULLOCH_PITTS) {
            neuron.m_MCPInhibitoryVeto =
                rng == nullptr ? parameters.InitialMCPInhibitoryVetoProb >= 0.5 : rng->RandFloat() < parameters.InitialMCPInhibitoryVetoProb;
        }
    }

    // Initializes a new link gene's synapse/STDP parameters from the evolvable ranges.
    inline void InitializeLinkSpiking(LinkGene &link, const Parameters &parameters, RNG *rng = nullptr) {
        const auto value = [rng](Real minimum, Real maximum) {
            return rng == nullptr ? minimum + (maximum - minimum) * 0.5 : RandomRange(*rng, minimum, maximum);
        };
        link.m_SynapticDelay = value(parameters.MinSynapticDelay, parameters.MaxSynapticDelay);
        link.m_SynapticTimeConstant = value(parameters.MinSynapticTimeConstant, parameters.MaxSynapticTimeConstant);
        link.m_STDPEnabled = rng == nullptr ? parameters.InitialSTDPEnabledProb >= 1.0 : rng->RandFloat() < parameters.InitialSTDPEnabledProb;
        link.m_STDPPlus = value(parameters.MinSTDPPlus, parameters.MaxSTDPPlus);
        link.m_STDPMinus = value(parameters.MinSTDPMinus, parameters.MaxSTDPMinus);
        link.m_STDPTauPlus = value(parameters.MinSTDPTau, parameters.MaxSTDPTau);
        link.m_STDPTauMinus = value(parameters.MinSTDPTau, parameters.MaxSTDPTau);
        link.m_STDPMinWeight = parameters.MinWeight;
        link.m_STDPMaxWeight = parameters.MaxWeight;
    }

    // Copies a substrate coordinate into a phenotype neuron, preserving the
    // historical x/y/z and substrate-coordinate aliases.
    inline void SetSpatialCoordinates(Neuron &neuron, const std::vector<Real> &coordinate) {
        neuron.m_substrate_coords = coordinate;
        neuron.m_x = coordinate.size() > 0 ? coordinate[0] : 0.0;
        neuron.m_y = coordinate.size() > 1 ? coordinate[1] : 0.0;
        neuron.m_z = coordinate.size() > 2 ? coordinate[2] : 0.0;
        neuron.m_sx = neuron.m_x;
        neuron.m_sy = neuron.m_y;
        neuron.m_sz = neuron.m_z;
    }

    // Validates that every substrate coordinate group is non-empty and finite.
    inline void ValidateSpatialSubstrate(const Substrate &substrate, const char *algorithm) {
        const auto validate_group = [algorithm](const std::vector<std::vector<Real>> &coordinates, const char *group) {
            for (const auto &coordinate : coordinates) {
                if (coordinate.empty() || !std::all_of(coordinate.begin(), coordinate.end(), [](Real value) { return std::isfinite(value); })) {
                    throw std::invalid_argument(std::string(algorithm) + " " + group + " coordinates must be non-empty and finite");
                }
            }
        };
        validate_group(substrate.m_input_coords, "input");
        validate_group(substrate.m_output_coords, "output");
        validate_group(substrate.m_hidden_coords, "hidden");
    }

    // Create an empty genome
    Genome::Genome() {
        m_ID = 0;
        m_Fitness = 0;
        m_Depth = 0;
        m_LinkGenes.clear();
        m_NeuronGenes.clear();
        m_NumInputs = 0;
        m_NumOutputs = 0;
        m_AdjustedFitness = 0;
        m_OffspringAmount = 0;
        m_Evaluated = false;
        m_PhenotypeBehavior = NULL;
        m_initial_num_neurons = 0;
        m_initial_num_links = 0;
    }

    // Copy constructor
    Genome::Genome(const Genome &a_G) {
        m_ID = a_G.m_ID;
        m_Depth = a_G.m_Depth;
        m_NeuronGenes = a_G.m_NeuronGenes;
        m_LinkGenes = a_G.m_LinkGenes;
        m_GenomeGene = a_G.m_GenomeGene;
        m_Fitness = a_G.m_Fitness;
        m_NumInputs = a_G.m_NumInputs;
        m_NumOutputs = a_G.m_NumOutputs;
        m_AdjustedFitness = a_G.m_AdjustedFitness;
        m_OffspringAmount = a_G.m_OffspringAmount;
        m_Evaluated = a_G.m_Evaluated;
        m_PhenotypeBehavior = a_G.m_PhenotypeBehavior;
        m_initial_num_neurons = a_G.m_initial_num_neurons;
        m_initial_num_links = a_G.m_initial_num_links;
    }

    // assignment operator
    Genome &Genome::operator=(const Genome &a_G) {
        // self assignment guard
        if (this != &a_G) {
            m_ID = a_G.m_ID;
            m_Depth = a_G.m_Depth;
            m_NeuronGenes = a_G.m_NeuronGenes;
            m_LinkGenes = a_G.m_LinkGenes;
            m_GenomeGene = a_G.m_GenomeGene;
            m_Fitness = a_G.m_Fitness;
            m_AdjustedFitness = a_G.m_AdjustedFitness;
            m_NumInputs = a_G.m_NumInputs;
            m_NumOutputs = a_G.m_NumOutputs;
            m_OffspringAmount = a_G.m_OffspringAmount;
            m_Evaluated = a_G.m_Evaluated;
            m_PhenotypeBehavior = a_G.m_PhenotypeBehavior;
            m_initial_num_neurons = a_G.m_initial_num_neurons;
            m_initial_num_links = a_G.m_initial_num_links;
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

    Genome::Genome(const Parameters &a_Parameters, const GenomeInitStruct &in) {
        const int usable_inputs = a_Parameters.DontUseBiasNeuron ? in.NumInputs : in.NumInputs - 1;
        if (usable_inputs < 1 || in.NumOutputs < 1 || in.NumHidden < 0 || in.NumLayers < 0)
            throw std::invalid_argument("Genome: input, output, hidden, and layer counts are invalid.");
        if (in.FS_NEAT && (in.FS_NEAT_links < 1 || static_cast<long long>(in.FS_NEAT_links) > static_cast<long long>(usable_inputs) * in.NumOutputs))
            throw std::invalid_argument("Genome: FS_NEAT_links exceeds the number of unique input-to-output links.");
        RNG t_RNG;
        t_RNG.TimeSeed();

        m_ID = 0;
        int t_innovnum = 1, t_nnum = 1;
        GenomeSeedType seed_type = in.SeedType;

        if (seed_type != PERCEPTRON && seed_type != LAYERED) throw std::invalid_argument("Genome: unknown seed type");

        // override seed_type if 0 hidden units are specified
        if ((seed_type == LAYERED) && (in.NumHidden == 0)) {
            seed_type = PERCEPTRON;
        }

        if (in.FS_NEAT && seed_type == LAYERED) throw std::invalid_argument("Genome: FS-NEAT initialization does not support layered seeds");

        if (a_Parameters.DontUseBiasNeuron == false) {
            // Create the input neurons.
            // Warning! The last one is a bias!
            // The order of the neurons is very important. It is the following: INPUTS, BIAS, OUTPUTS, HIDDEN ... (no limit)
            for (unsigned int i = 0; i < (in.NumInputs - 1); i++) {
                NeuronGene n = NeuronGene(INPUT, t_nnum, 0.0);
                // Initialize the traits
                // n.InitTraits(a_Parameters.NeuronTraits, t_RNG); // no need to init traits for inputs
                m_NeuronGenes.emplace_back(n);
                t_nnum++;
            }
            // add the bias
            NeuronGene n = NeuronGene(BIAS, t_nnum, 0.0);
            // Initialize the traits
            // n.InitTraits(a_Parameters.NeuronTraits, t_RNG); // no need to init traits for inputs

            m_NeuronGenes.emplace_back(n);
            t_nnum++;
        } else {
            // Create the input neurons without marking the last node as bias. The order of the neurons is very important. It is the following: INPUTS, OUTPUTS,
            // HIDDEN ... (no limit)
            for (unsigned int i = 0; i < in.NumInputs; i++) {
                NeuronGene n = NeuronGene(INPUT, t_nnum, 0.0);
                // Initialize the traits
                // n.InitTraits(a_Parameters.NeuronTraits, t_RNG); // no need to init traits for inputs

                m_NeuronGenes.emplace_back(n);
                t_nnum++;
            }
        }

        // now the outputs
        for (unsigned int i = 0; i < (in.NumOutputs); i++) {
            NeuronGene t_ngene(OUTPUT, t_nnum, 1.0);
            // Initialize the neuron gene's properties
            t_ngene.Init((a_Parameters.MinActivationA + a_Parameters.MaxActivationA) / 2.0f, (a_Parameters.MinActivationB + a_Parameters.MaxActivationB) / 2.0f,
                         (a_Parameters.MinNeuronTimeConstant + a_Parameters.MaxNeuronTimeConstant) / 2.0f,
                         (a_Parameters.MinNeuronBias + a_Parameters.MaxNeuronBias) / 2.0f, in.OutputActType);
            InitializeNeuronSpiking(t_ngene, a_Parameters);
            // Initialize the traits
            t_ngene.InitTraits(a_Parameters.NeuronTraits, t_RNG);

            m_NeuronGenes.emplace_back(t_ngene);
            t_nnum++;
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
        if ((in.SeedType == LAYERED) && (in.NumHidden > 0)) {
            Real lt_inc = 1.0 / (in.NumLayers + 1);
            Real initlt = lt_inc;
            for (unsigned int n = 0; n < in.NumLayers; n++) {
                for (unsigned int i = 0; i < in.NumHidden; i++) {
                    NeuronGene t_ngene(HIDDEN, t_nnum, 1.0);
                    // Initialize the neuron gene's properties
                    t_ngene.Init((a_Parameters.MinActivationA + a_Parameters.MaxActivationA) / 2.0f,
                                 (a_Parameters.MinActivationB + a_Parameters.MaxActivationB) / 2.0f,
                                 (a_Parameters.MinNeuronTimeConstant + a_Parameters.MaxNeuronTimeConstant) / 2.0f,
                                 (a_Parameters.MinNeuronBias + a_Parameters.MaxNeuronBias) / 2.0f, in.HiddenActType);
                    InitializeNeuronSpiking(t_ngene, a_Parameters);
                    // Initialize the traits
                    t_ngene.InitTraits(a_Parameters.NeuronTraits, t_RNG);
                    t_ngene.m_SplitY = initlt;

                    m_NeuronGenes.emplace_back(t_ngene);
                    t_nnum++;
                }

                initlt += lt_inc;
            }

            if (!in.FS_NEAT) {
                int last_dest_id = in.NumInputs + in.NumOutputs + 1;
                int last_src_id = 1;
                int prev_layer_size = in.NumInputs;

                for (unsigned int n = 0; n < in.NumLayers; n++) {
                    // The links from each previous layer to this hidden node
                    for (unsigned int i = 0; i < in.NumHidden; i++) {
                        for (unsigned int j = 0; j < prev_layer_size; j++) {
                            // add the link created with zero weights. needs future random initialization. !!!!!!!! init traits (TODO: maybe init empty traits?)
                            LinkGene l = LinkGene(j + last_src_id, i + last_dest_id, t_innovnum, 0.0, false);
                            InitializeLinkSpiking(l, a_Parameters);
                            l.InitTraits(a_Parameters.LinkTraits, t_RNG);
                            m_LinkGenes.emplace_back(l);
                            t_innovnum++;
                        }
                    }

                    last_dest_id += in.NumHidden;
                    if (n == 0) {
                        // for the first hidden layer, jump over the outputs too
                        last_src_id += prev_layer_size + in.NumOutputs;
                    } else {
                        last_src_id += prev_layer_size;
                    }
                    prev_layer_size = in.NumHidden;
                }

                last_dest_id = in.NumInputs + 1;

                // The links from each previous layer to this output node
                for (unsigned int i = 0; i < in.NumOutputs; i++) {
                    for (unsigned int j = 0; j < prev_layer_size; j++) {
                        // add the link created with zero weights. needs future random initialization. !!!!!!!! init traits (TODO: maybe init empty traits?)
                        LinkGene l = LinkGene(j + last_src_id, i + last_dest_id, t_innovnum, 0.0, false);
                        InitializeLinkSpiking(l, a_Parameters);
                        l.InitTraits(a_Parameters.LinkTraits, t_RNG);
                        m_LinkGenes.emplace_back(l);
                        t_innovnum++;
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
            if ((!in.FS_NEAT) && (seed_type == PERCEPTRON)) {
                for (unsigned int i = 0; i < (in.NumOutputs); i++) {
                    for (unsigned int j = 0; j < in.NumInputs; j++) {
                        // add the link created with zero weights. needs future random initialization. !!!!!!!!
                        LinkGene l = LinkGene(j + 1, i + in.NumInputs + 1, t_innovnum, 0.0, false);
                        InitializeLinkSpiking(l, a_Parameters);
                        l.InitTraits(a_Parameters.LinkTraits, t_RNG);
                        m_LinkGenes.emplace_back(l);
                        t_innovnum++;
                    }
                }
            } else {
                // Start very minimally - connect a random input to each output
                // Also connect the bias to every output

                std::vector<std::pair<int, int>> made_already;
                bool there = false;
                int linksmade = 0;

                // do this a few times for more initial links created
                // TODO: make sure the innovations don't repeat for the same input/output pairs
                while (linksmade < in.FS_NEAT_links) {
                    for (unsigned int i = 0; i < in.NumOutputs; i++) {
                        int t_inp_id = t_RNG.RandInt(1, usable_inputs);
                        int t_bias_id = in.NumInputs;
                        int t_outp_id = in.NumInputs + 1 + i;

                        // check if there already
                        there = false;
                        for (auto it = made_already.begin(); it != made_already.end(); it++) {
                            if ((it->first == t_inp_id) && (it->second == t_outp_id)) {
                                there = true;
                                break;
                            }
                        }

                        if (!there) {
                            // created with zero weights. needs future random initialization. !!!!!!!!
                            LinkGene l = LinkGene(t_inp_id, t_outp_id, t_innovnum, 0.0, false);
                            InitializeLinkSpiking(l, a_Parameters);
                            l.InitTraits(a_Parameters.LinkTraits, t_RNG);
                            m_LinkGenes.emplace_back(l);
                            t_innovnum++;

                            if (a_Parameters.DontUseBiasNeuron == false) {
                                LinkGene bl = LinkGene(t_bias_id, t_outp_id, t_innovnum, 0.0, false);
                                InitializeLinkSpiking(bl, a_Parameters);
                                bl.InitTraits(a_Parameters.LinkTraits, t_RNG);
                                m_LinkGenes.emplace_back(bl);
                                t_innovnum++;
                            }

                            linksmade++;
                            made_already.push_back(std::make_pair(t_inp_id, t_outp_id));
                        }
                    }
                }
            }
        }

        // Also initialize the Genome's traits
        m_GenomeGene.InitTraits(a_Parameters.GenomeTraits, t_RNG);

        m_Evaluated = false;
        m_NumInputs = in.NumInputs;
        m_NumOutputs = in.NumOutputs;
        m_Fitness = 0.0;
        m_AdjustedFitness = 0.0;
        m_OffspringAmount = 0.0;
        m_Depth = 0;
        m_PhenotypeBehavior = NULL;

        m_initial_num_neurons = NumNeurons();
        m_initial_num_links = NumLinks();
    }

    void Genome::SetDepth(unsigned int a_d) {
        if (a_d > static_cast<unsigned int>(std::numeric_limits<int>::max())) throw std::out_of_range("Genome depth exceeds the supported range");
        m_Depth = static_cast<int>(a_d);
    }

    unsigned int Genome::GetDepth() const { return m_Depth; }

    void Genome::SetID(int a_id) { m_ID = a_id; }

    int Genome::GetID() const { return m_ID; }

    void Genome::SetAdjFitness(Real a_af) { m_AdjustedFitness = a_af; }

    void Genome::SetFitness(Real a_f) { m_Fitness = a_f; }

    Real Genome::GetAdjFitness() const { return m_AdjustedFitness; }

    Real Genome::GetFitness() const { return m_Fitness; }

    void Genome::SetNeuronY(unsigned int a_idx, int a_y) { m_NeuronGenes.at(a_idx).y = a_y; }

    void Genome::SetNeuronX(unsigned int a_idx, int a_x) { m_NeuronGenes.at(a_idx).x = a_x; }

    void Genome::SetNeuronXY(unsigned int a_idx, int a_x, int a_y) {
        m_NeuronGenes.at(a_idx).x = a_x;
        m_NeuronGenes.at(a_idx).y = a_y;
    }

    LinkGene Genome::GetLinkByIndex(int a_idx) const {
        ASSERT(a_idx < m_LinkGenes.size());
        return m_LinkGenes[a_idx];
    }

    LinkGene Genome::GetLinkByInnovID(int a_ID) const {
        for (unsigned int i = 0; i < m_LinkGenes.size(); i++)
            if (m_LinkGenes[i].InnovationID() == a_ID) return m_LinkGenes[i];

        // should never reach this code
        throw std::out_of_range("Genome::GetLinkByInnovID: unknown innovation ID");
    }

    NeuronGene Genome::GetNeuronByIndex(int a_idx) const {
        ASSERT(a_idx < m_NeuronGenes.size());
        return m_NeuronGenes[a_idx];
    }

    NeuronGene Genome::GetNeuronByID(int a_ID) const {
        int t_idx = GetNeuronIndex(a_ID);
        if (t_idx < 0) throw std::out_of_range("Genome::GetNeuronByID: unknown neuron ID");
        return m_NeuronGenes[t_idx];
    }

    Real Genome::GetOffspringAmount() const { return m_OffspringAmount; }

    void Genome::SetOffspringAmount(Real a_oa) { m_OffspringAmount = a_oa; }

    bool Genome::IsEvaluated() const { return m_Evaluated; }

    void Genome::SetEvaluated() { m_Evaluated = true; }

    void Genome::ResetEvaluated() { m_Evaluated = false; }

    // A little helper function to find the index of a neuron, given its ID
    // returns -1 if not found
    int Genome::GetNeuronIndex(int a_ID) const {
        ASSERT(a_ID > 0);

        for (unsigned int i = 0; i < NumNeurons(); i++) {
            if (m_NeuronGenes[i].ID() == a_ID) {
                return i;
            }
        }

        return -1;
    }

    // A little helper function to find the index of a link, given its innovation ID
    // returns -1 if not found
    int Genome::GetLinkIndex(int a_InnovID) const {
        ASSERT(a_InnovID > 0);
        ASSERT(NumLinks() > 0);

        for (unsigned int i = 0; i < NumLinks(); i++) {
            if (m_LinkGenes[i].InnovationID() == a_InnovID) {
                return i;
            }
        }

        return -1;
    }

    // returns the max neuron ID (callers add one for the next free ID)
    int Genome::GetLastNeuronID() const {
        int t_maxid = 0;

        for (unsigned int i = 0; i < NumNeurons(); i++) {
            if (m_NeuronGenes[i].ID() > t_maxid) t_maxid = m_NeuronGenes[i].ID();
        }

        return t_maxid;
    }

    // returns the max innovation Id (callers add one for the next free ID)
    int Genome::GetLastInnovationID() const {
        int t_maxid = 0;

        for (unsigned int i = 0; i < NumLinks(); i++) {
            if (m_LinkGenes[i].InnovationID() > t_maxid) t_maxid = m_LinkGenes[i].InnovationID();
        }

        return t_maxid;
    }

    // Returns true if the specified neuron ID is present in the genome
    bool Genome::HasNeuronID(int a_ID) const {
        ASSERT(a_ID > 0);
        ASSERT(NumNeurons() > 0);

        for (unsigned int i = 0; i < NumNeurons(); i++) {
            if (m_NeuronGenes[i].ID() == a_ID) {
                return true;
            }
        }

        return false;
    }

    // Returns true if the specified link is present in the genome
    bool Genome::HasLink(int a_n1id, int a_n2id) const {
        ASSERT((a_n1id > 0) && (a_n2id > 0));

        for (unsigned int i = 0; i < NumLinks(); i++) {
            if ((m_LinkGenes[i].FromNeuronID() == a_n1id) && (m_LinkGenes[i].ToNeuronID() == a_n2id)) {
                return true;
            }
        }

        return false;
    }

    bool Genome::HasLoops() {
        NeuralNetwork net;
        BuildPhenotype(net);

        // Detect directed cycles using Kahn's algorithm (indegree-based
        // topological sort). If every node cannot be processed, a cycle exists.
        const int n = static_cast<int>(net.m_neurons.size());
        const int e = static_cast<int>(net.m_connections.size());
        std::vector<int> indegree(n, 0);
        // Flat out-adjacency (head/next chains) so each edge is visited once
        // overall instead of once per popped node (was O(nodes x edges)); two
        // vector allocations regardless of graph size.
        std::vector<int> t_head(n, -1);
        std::vector<int> t_next(e, -1);

        for (int i = 0; i < e; i++) {
            int src = net.m_connections[i].m_source_neuron_idx;
            int tgt = net.m_connections[i].m_target_neuron_idx;
            if ((tgt >= 0) && (tgt < n)) {
                indegree[tgt]++;
                if ((src >= 0) && (src < n)) {
                    t_next[i] = t_head[src];
                    t_head[src] = i;
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
            for (int k = t_head[node]; k != -1; k = t_next[k]) {
                int tgt = net.m_connections[k].m_target_neuron_idx;
                indegree[tgt]--;
                if (indegree[tgt] == 0) {
                    stack.push_back(tgt);
                }
            }
        }

        // A self-loop on a single node (src == tgt) is also a cycle
        bool self_loop = false;
        for (int i = 0; i < net.m_connections.size(); i++) {
            if (net.m_connections[i].m_source_neuron_idx == net.m_connections[i].m_target_neuron_idx) {
                self_loop = true;
                break;
            }
        }

        return (visited < n) || self_loop;
    }

    // Returns true if the specified link is present in the genome
    bool Genome::HasLinkByInnovID(int id) const {
        ASSERT(id > 0);

        for (unsigned int i = 0; i < NumLinks(); i++) {
            if (m_LinkGenes[i].InnovationID() == id) {
                return true;
            }
        }

        return false;
    }

    // This builds a fastnetwork structure out from the genome
    void Genome::BuildPhenotype(NeuralNetwork &a_Net) {
        // first clear out the network
        a_Net.Clear();
        a_Net.SetInputOutputDimentions(m_NumInputs, m_NumOutputs);

        // Build an ID->index table once so the connection loop below resolves
        // endpoints in O(1); the previous GetNeuronIndex() rescans made this
        // O(links x neurons).
        int t_max_id = 0;
        for (unsigned int i = 0; i < NumNeurons(); i++) {
            if (m_NeuronGenes[i].ID() > t_max_id) {
                t_max_id = m_NeuronGenes[i].ID();
            }
        }
        std::vector<int> t_id_to_index(static_cast<size_t>(t_max_id) + 1, -1);

        // Fill the net with the neurons
        for (unsigned int i = 0; i < NumNeurons(); i++) {
            Neuron t_n;

            t_id_to_index[static_cast<size_t>(m_NeuronGenes[i].ID())] = static_cast<int>(i);

            t_n.m_a = m_NeuronGenes[i].m_A;
            t_n.m_b = m_NeuronGenes[i].m_B;
            t_n.m_timeconst = m_NeuronGenes[i].m_TimeConstant;
            t_n.m_bias = m_NeuronGenes[i].m_Bias;
            t_n.m_activation_function_type = m_NeuronGenes[i].m_ActFunction;
            t_n.m_spike_threshold = m_NeuronGenes[i].m_SpikeThreshold;
            t_n.m_reset_potential = m_NeuronGenes[i].m_ResetPotential;
            t_n.m_resting_potential = m_NeuronGenes[i].m_RestingPotential;
            t_n.m_refractory_period = m_NeuronGenes[i].m_RefractoryPeriod;
            t_n.m_membrane_resistance = m_NeuronGenes[i].m_MembraneResistance;
            t_n.m_adaptation_time_constant = m_NeuronGenes[i].m_AdaptationTimeConstant;
            t_n.m_adaptation_increment = m_NeuronGenes[i].m_AdaptationIncrement;
            t_n.m_rate_time_constant = m_NeuronGenes[i].m_RateTimeConstant;
            t_n.m_izhikevich_a = m_NeuronGenes[i].m_IzhikevichA;
            t_n.m_izhikevich_b = m_NeuronGenes[i].m_IzhikevichB;
            t_n.m_izhikevich_c = m_NeuronGenes[i].m_IzhikevichC;
            t_n.m_izhikevich_d = m_NeuronGenes[i].m_IzhikevichD;
            t_n.m_mcp_inhibitory_veto = m_NeuronGenes[i].m_MCPInhibitoryVeto;
            t_n.m_split_y = m_NeuronGenes[i].SplitY();
            t_n.m_type = m_NeuronGenes[i].Type();
            t_n.m_x = static_cast<Real>(m_NeuronGenes[i].x);
            t_n.m_y = static_cast<Real>(m_NeuronGenes[i].y);

            a_Net.AddNeuron(t_n);
        }

        // Fill the net with the connections
        for (unsigned int i = 0; i < NumLinks(); i++) {
            Connection t_c;

            const int t_from = m_LinkGenes[i].FromNeuronID();
            const int t_to = m_LinkGenes[i].ToNeuronID();
            const int t_from_idx = (t_from >= 0 && t_from <= t_max_id) ? t_id_to_index[static_cast<size_t>(t_from)] : -1;
            const int t_to_idx = (t_to >= 0 && t_to <= t_max_id) ? t_id_to_index[static_cast<size_t>(t_to)] : -1;
            if (t_from_idx < 0 || t_to_idx < 0) {
                throw std::runtime_error("Genome contains a link whose endpoint neuron does not exist");
            }
            t_c.m_source_neuron_idx = t_from_idx;
            t_c.m_target_neuron_idx = t_to_idx;
            t_c.m_weight = m_LinkGenes[i].GetWeight();
            t_c.m_recur_flag = m_LinkGenes[i].IsRecurrent();
            t_c.m_synaptic_delay = m_LinkGenes[i].m_SynapticDelay;
            t_c.m_synaptic_time_constant = m_LinkGenes[i].m_SynapticTimeConstant;
            t_c.m_stdp_enabled = m_LinkGenes[i].m_STDPEnabled;
            t_c.m_stdp_plus = m_LinkGenes[i].m_STDPPlus;
            t_c.m_stdp_minus = m_LinkGenes[i].m_STDPMinus;
            t_c.m_stdp_tau_plus = m_LinkGenes[i].m_STDPTauPlus;
            t_c.m_stdp_tau_minus = m_LinkGenes[i].m_STDPTauMinus;
            t_c.m_stdp_min_weight = m_LinkGenes[i].m_STDPMinWeight;
            t_c.m_stdp_max_weight = m_LinkGenes[i].m_STDPMaxWeight;

            //////////////////////
            // default values
            t_c.m_hebb_rate = 0.3;
            t_c.m_hebb_pre_rate = 0.1;

            // if a float trait "hebb_rate" exists
            if (m_LinkGenes[i].m_Traits.count("hebb_rate") == 1) {
                try {
                    t_c.m_hebb_rate = std::get<Real>(m_LinkGenes[i].m_Traits.at("hebb_rate").value);
                } catch (...) {
                    // do nothing
                }
            }
            // if a float trait "hebb_pre_rate" exists
            if (m_LinkGenes[i].m_Traits.count("hebb_pre_rate") == 1) {
                try {
                    t_c.m_hebb_pre_rate = std::get<Real>(m_LinkGenes[i].m_Traits.at("hebb_pre_rate").value);
                } catch (...) {
                    // do nothing
                }
            }

            //////////////////////

            a_Net.AddConnection(t_c);
        }

        a_Net.Flush();

        // Note however that the RTRL variables are not initialized.
        // The user must manually call the InitRTRLMatrix() method to do it.
        // This is because of storage issues. RTRL need not to be used every time.
    }

    // Builds a HyperNEAT phenotype based on the substrate The CPPN input dimensionality must match the largest number of dimensions in the substrate The output
    // dimensionality is determined according to flags set in the substrate

    // The procedure uses the [0] CPPN output for creating nodes, and if the substrate is leaky, [1] and [2] for time constants and biases Also assumes the CPPN
    // uses signed activation outputs
    void Genome::BuildHyperNEATPhenotype(NeuralNetwork &net, Substrate &subst) {
        // We need a substrate with at least one input and output
        if (subst.m_input_coords.empty() || subst.m_output_coords.empty())
            throw std::invalid_argument("A HyperNEAT substrate requires input and output coordinates");
        ValidateSpatialSubstrate(subst, "HyperNEAT");

        int max_dims = subst.GetMaxDims();

        // Make sure the CPPN dimensionality is right
        if (static_cast<int>(m_NumInputs) < subst.GetMinCPPNInputs() || static_cast<int>(m_NumOutputs) < subst.GetMinCPPNOutputs())
            throw std::invalid_argument("The CPPN does not provide enough inputs or outputs for the substrate");
        if (!std::isfinite(subst.m_max_weight_and_bias) || subst.m_max_weight_and_bias < 0.0 || !std::isfinite(subst.m_min_time_const) ||
            !std::isfinite(subst.m_max_time_const) || subst.m_min_time_const > subst.m_max_time_const)
            throw std::invalid_argument("The substrate weight, bias, or time-constant range is invalid");
        if (subst.m_leaky) {
            ASSERT(NumOutputs() >= subst.GetMinCPPNOutputs());
        }

        // Now we create the substrate (net)
        net.Clear();
        net.SetInputOutputDimentions(static_cast<unsigned short>(subst.m_input_coords.size()), static_cast<unsigned short>(subst.m_output_coords.size()));

        // Inputs
        for (unsigned int i = 0; i < subst.m_input_coords.size(); i++) {
            Neuron t_n;

            t_n.m_a = 1;
            t_n.m_b = 0;
            SetSpatialCoordinates(t_n, subst.m_input_coords[i]);
            t_n.m_activation_function_type = NEAT::LINEAR;
            t_n.m_type = NEAT::INPUT;

            net.AddNeuron(t_n);
        }

        // Output
        for (unsigned int i = 0; i < subst.m_output_coords.size(); i++) {
            Neuron t_n;

            t_n.m_a = 1;
            t_n.m_b = 0;
            SetSpatialCoordinates(t_n, subst.m_output_coords[i]);
            t_n.m_activation_function_type = subst.m_output_nodes_activation;
            t_n.m_type = NEAT::OUTPUT;

            net.AddNeuron(t_n);
        }

        // Hidden
        for (unsigned int i = 0; i < subst.m_hidden_coords.size(); i++) {
            Neuron t_n;

            t_n.m_a = 1;
            t_n.m_b = 0;
            SetSpatialCoordinates(t_n, subst.m_hidden_coords[i]);
            t_n.m_activation_function_type = subst.m_hidden_nodes_activation;
            t_n.m_type = NEAT::HIDDEN;

            net.AddNeuron(t_n);
        }

        // Begin querying the CPPN Create the neural network that will represent the CPPN
        NeuralNetwork t_temp_phenotype(true);
        BuildPhenotype(t_temp_phenotype);
        t_temp_phenotype.Flush();

        // To ensure network relaxation
        int dp = 8;
        if (!HasLoops()) {
            CalculateDepth();
            dp = GetDepth();
        }

        // now loop over every potential connection in the substrate and take its weight

        // For leaky substrates, first loop over the neurons and set their properties
        if (subst.m_leaky) {
            for (unsigned int i = net.NumInputs(); i < net.m_neurons.size(); i++) {
                // neuron specific stuff
                t_temp_phenotype.Flush();

                // Inputs for the generation of time consts and biases across the nodes in the substrate We input only the position of the first node and ignore
                // the other one
                std::vector<Real> t_inputs;
                t_inputs.resize(NumInputs());

                for (unsigned int n = 0; n < net.m_neurons[i].m_substrate_coords.size(); n++) {
                    t_inputs[n] = net.m_neurons[i].m_substrate_coords[n];
                }

                if (subst.m_with_distance) {
                    // compute the Eucledian distance between the point and the origin
                    Real sum = 0;
                    for (int n = 0; n < max_dims; n++) {
                        sum += sqr(t_inputs[n]);
                    }
                    sum = std::sqrt(sum);
                    t_inputs[NumInputs() - 2] = sum;
                }
                t_inputs[NumInputs() - 1] = 1.0;  // the CPPN's bias

                t_temp_phenotype.Input(t_inputs);

                // activate as many times as deep
                for (int d = 0; d < dp; d++) {
                    t_temp_phenotype.Activate();
                }

                Real t_tc = t_temp_phenotype.Output()[NumOutputs() - 2];
                Real t_bias = t_temp_phenotype.Output()[NumOutputs() - 1];

                Clamp(t_tc, -1, 1);
                Clamp(t_bias, -1, 1);

                // rescale the values
                Scale(t_tc, -1, 1, subst.m_min_time_const, subst.m_max_time_const);
                Scale(t_bias, -1, 1, -subst.m_max_weight_and_bias, subst.m_max_weight_and_bias);

                net.m_neurons[i].m_timeconst = t_tc;
                net.m_neurons[i].m_bias = t_bias;
            }
        }

        // list of src_idx, dst_idx pairs of all connections to query
        std::vector<std::vector<int>> t_to_query;

        // There isn't custom connectiviy scheme?
        if (subst.m_custom_connectivity.size() == 0) {
            // only incoming connections, so loop only the hidden and output neurons
            for (int i = net.NumInputs(); i < net.m_neurons.size(); i++) {
                // loop all neurons
                for (int j = 0; j < net.m_neurons.size(); j++) {
                    // this is connection "j" to "i"

                    // conditions for canceling the CPPN query
                    if (((!subst.m_allow_input_hidden_links) && ((net.m_neurons[j].m_type == INPUT) && (net.m_neurons[i].m_type == HIDDEN)))

                        || ((!subst.m_allow_input_output_links) && ((net.m_neurons[j].m_type == INPUT) && (net.m_neurons[i].m_type == OUTPUT)))

                        || ((!subst.m_allow_hidden_hidden_links) && ((net.m_neurons[j].m_type == HIDDEN) && (net.m_neurons[i].m_type == HIDDEN) && (i != j)))

                        || ((!subst.m_allow_hidden_output_links) && ((net.m_neurons[j].m_type == HIDDEN) && (net.m_neurons[i].m_type == OUTPUT)))

                        || ((!subst.m_allow_output_hidden_links) && ((net.m_neurons[j].m_type == OUTPUT) && (net.m_neurons[i].m_type == HIDDEN)))

                        || ((!subst.m_allow_output_output_links) && ((net.m_neurons[j].m_type == OUTPUT) && (net.m_neurons[i].m_type == OUTPUT) && (i != j)))

                        || ((!subst.m_allow_looped_hidden_links) && ((net.m_neurons[j].m_type == HIDDEN) && (net.m_neurons[i].m_type == HIDDEN) && (i == j)))

                        || ((!subst.m_allow_looped_output_links) && ((net.m_neurons[j].m_type == OUTPUT) && (net.m_neurons[i].m_type == OUTPUT) && (i == j)))

                    ) {
                        continue;
                    }

                    // Save potential link to query
                    std::vector<int> t_link;
                    t_link.emplace_back(j);
                    t_link.emplace_back(i);
                    t_to_query.emplace_back(t_link);
                }
            }
        } else {
            // use the custom connectivity
            for (unsigned int idx = 0; idx < subst.m_custom_connectivity.size(); idx++) {
                if (subst.m_custom_connectivity[idx].size() != 4) throw std::invalid_argument("Malformed custom substrate connection");
                NeuronType src_type = (NeuronType)subst.m_custom_connectivity[idx][0];
                int src_idx = subst.m_custom_connectivity[idx][1];
                NeuronType dst_type = (NeuronType)subst.m_custom_connectivity[idx][2];
                int dst_idx = subst.m_custom_connectivity[idx][3];
                if (src_idx < 0 || dst_idx < 0) throw std::invalid_argument("Custom substrate connection index out of range");

                // determine the indices in the NN
                int j = 0;  // src
                int i = 0;  // dst

                if ((src_type == INPUT) || (src_type == BIAS)) {
                    j = src_idx;
                } else if (src_type == HIDDEN) {
                    j = subst.m_input_coords.size() + subst.m_output_coords.size() + src_idx;
                } else if (src_type == OUTPUT) {
                    j = subst.m_input_coords.size() + src_idx;
                }

                if ((dst_type == INPUT) || (dst_type == BIAS)) {
                    i = dst_idx;
                } else if (dst_type == HIDDEN) {
                    i = subst.m_input_coords.size() + subst.m_output_coords.size() + dst_idx;
                } else if (dst_type == OUTPUT) {
                    i = subst.m_input_coords.size() + dst_idx;
                }

                // conditions for canceling the CPPN query
                if (subst.m_custom_conn_obeys_flags &&
                    (((!subst.m_allow_input_hidden_links) && ((net.m_neurons[j].m_type == INPUT) && (net.m_neurons[i].m_type == HIDDEN)))

                     || ((!subst.m_allow_input_output_links) && ((net.m_neurons[j].m_type == INPUT) && (net.m_neurons[i].m_type == OUTPUT)))

                     || ((!subst.m_allow_hidden_hidden_links) && ((net.m_neurons[j].m_type == HIDDEN) && (net.m_neurons[i].m_type == HIDDEN) && (i != j)))

                     || ((!subst.m_allow_hidden_output_links) && ((net.m_neurons[j].m_type == HIDDEN) && (net.m_neurons[i].m_type == OUTPUT)))

                     || ((!subst.m_allow_output_hidden_links) && ((net.m_neurons[j].m_type == OUTPUT) && (net.m_neurons[i].m_type == HIDDEN)))

                     || ((!subst.m_allow_output_output_links) && ((net.m_neurons[j].m_type == OUTPUT) && (net.m_neurons[i].m_type == OUTPUT) && (i != j)))

                     || ((!subst.m_allow_looped_hidden_links) && ((net.m_neurons[j].m_type == HIDDEN) && (net.m_neurons[i].m_type == HIDDEN) && (i == j)))

                     || ((!subst.m_allow_looped_output_links) && ((net.m_neurons[j].m_type == OUTPUT) && (net.m_neurons[i].m_type == OUTPUT) && (i == j))))) {
                    continue;
                }

                // Save potential link to query
                std::vector<int> t_link;
                t_link.emplace_back(j);
                t_link.emplace_back(i);
                t_to_query.emplace_back(t_link);
            }
        }

        // Query and create all links
        for (unsigned int conn = 0; conn < t_to_query.size(); conn++) {
            int j = t_to_query[conn][0];
            int i = t_to_query[conn][1];

            // Take the weight of this connection by querying the CPPN
            // as many times as deep (recurrent or looped CPPNs may be very slow!!!*)
            std::vector<Real> t_inputs;
            t_inputs.resize(NumInputs());

            int from_dims = net.m_neurons[j].m_substrate_coords.size();
            int to_dims = net.m_neurons[i].m_substrate_coords.size();

            // input the node positions to the CPPN from
            for (int n = 0; n < from_dims; n++) {
                t_inputs[n] = net.m_neurons[j].m_substrate_coords[n];
            }
            // to
            for (int n = 0; n < to_dims; n++) {
                t_inputs[max_dims + n] = net.m_neurons[i].m_substrate_coords[n];
            }

            // the input is like
            // x000|xx00|1 - 1D -> 2D connection
            // xx00|xx00|1 - 2D -> 2D connection
            // xx00|xxx0|1 - 2D -> 3D connection
            // if max_dims is 4 and no distance input

            if (subst.m_with_distance) {
                // compute the Eucledian distance between the two points differing dimensionality doesn't matter as the extra dimensions are 0s
                Real sum = 0;
                for (int n = 0; n < max_dims; n++) {
                    sum += sqr(t_inputs[n] - t_inputs[max_dims + n]);
                }
                sum = std::sqrt(sum);

                t_inputs[NumInputs() - 2] = sum;
            }

            t_inputs[NumInputs() - 1] = 1.0;

            // flush between each query
            t_temp_phenotype.Flush();
            t_temp_phenotype.Input(t_inputs);

            // activate as many times as deep
            for (int d = 0; d < dp; d++) {
                t_temp_phenotype.Activate();
            }

            // the output is a weight
            Real t_link = 0;
            Real t_weight = 0;

            if (subst.m_query_weights_only) {
                t_weight = t_temp_phenotype.Output()[0];
            } else {
                t_link = t_temp_phenotype.Output()[0];
                t_weight = t_temp_phenotype.Output()[1];
            }

            if (((t_link > 0) && (!subst.m_query_weights_only)) || (subst.m_query_weights_only)) {
                // now this weight will be scaled
                t_weight *= subst.m_max_weight_and_bias;

                // build the connection
                Connection t_c;

                t_c.m_source_neuron_idx = j;
                t_c.m_target_neuron_idx = i;
                t_c.m_weight = t_weight;
                t_c.m_recur_flag = false;

                net.AddConnection(t_c);
            }
        }
    }

    // Projects the weight changes of a phenotype back to the genome.
    // WARNING! Using this too often in conjuction with RTRL can confuse evolution.
    void Genome::DerivePhenotypicChanges(NeuralNetwork &a_Net) {
        // the a_Net and the genome must have identical topology. if the topology differs, no changes will be made to the genome
        if (a_Net.m_connections.size() != m_LinkGenes.size()) return;

        // Since we don't have a comparison operator yet, we are going to assume
        // identical topolgy
        // TODO: create that comparison operator for NeuralNetworks

        // Iterate through the links and replace weights
        for (unsigned int i = 0; i < NumLinks(); i++) {
            m_LinkGenes[i].SetWeight(a_Net.GetConnectionByIndex(i).m_weight);
        }

        // TODO: if neuron parameters were changed, derive them
        // * in future expansions
    }

    // std::map<std::pair<int,int>, Real> distance_cache;

    // Returns the absolute distance between this genome and a_G
    Real Genome::CompatibilityDistance(Genome &a_G, Parameters &a_Parameters) {
        // first check if in cache, if so, return that
        /*auto q1 = std::make_pair(this->GetID(), a_G.GetID());
        auto q2 = std::make_pair(a_G.GetID(), this->GetID());
        if (distance_cache.count(q1) > 0)
        {
            return distance_cache[q1];
        }
        else if (distance_cache.count(q2) > 0)
        {
            return distance_cache[q2];
        }*/

        // New - if there is a behavior in the genomes, return their distance

        // iterators for moving through the genomes' genes
        std::vector<LinkGene>::const_iterator t_g1;
        std::vector<LinkGene>::const_iterator t_g2;

        // this variable is the total distance between the genomes if it passes beyond the compatibility treshold, the function returns false
        Real t_total_distance = 0.0;

        Real t_total_weight_difference = 0.0;
        Real t_total_timeconstant_difference = 0.0;
        Real t_total_bias_difference = 0.0;
        Real t_total_A_difference = 0.0;
        Real t_total_B_difference = 0.0;
        Real t_total_num_activation_difference = 0.0;
        Real t_total_spiking_neuron_difference = 0.0;
        Real t_total_spiking_link_difference = 0.0;
        std::map<std::string, Real> t_total_neuron_trait_difference;
        std::map<std::string, Real> t_total_link_trait_difference;
        std::map<std::string, Real> t_genome_link_trait_difference;

        // count of matching genes
        Real t_num_excess = 0;
        Real t_num_disjoint = 0;
        Real t_num_matching_links = 0;
        Real t_num_matching_neurons = 0;

        // calculate genome trait difference here
        t_genome_link_trait_difference = m_GenomeGene.GetTraitDistances(a_G.m_GenomeGene.m_Traits);

        // used for percentage of excess/disjoint genes calculation
        int t_max_genome_size = static_cast<int>(NumLinks() < a_G.NumLinks()) ? (a_G.NumLinks()) : (NumLinks());
        int t_max_neurons = static_cast<int>(NumNeurons() < a_G.NumNeurons()) ? (a_G.NumNeurons()) : (NumNeurons());

        t_g1 = m_LinkGenes.begin();
        t_g2 = a_G.m_LinkGenes.begin();

        auto by_innovation = [](const LinkGene &lhs, const LinkGene &rhs) { return lhs.InnovationID() < rhs.InnovationID(); };
        // Sort copies when a genome is unsorted so the merge below classifies
        // disjoint/excess correctly instead of misreading order as divergence.
        std::vector<LinkGene> t_sorted_1, t_sorted_2;
        const std::vector<LinkGene> *t_links_1 = &m_LinkGenes;
        const std::vector<LinkGene> *t_links_2 = &a_G.m_LinkGenes;
        if (!std::is_sorted(m_LinkGenes.begin(), m_LinkGenes.end(), by_innovation)) {
            t_sorted_1 = m_LinkGenes;
            std::sort(t_sorted_1.begin(), t_sorted_1.end(), by_innovation);
            t_links_1 = &t_sorted_1;
        }
        if (!std::is_sorted(a_G.m_LinkGenes.begin(), a_G.m_LinkGenes.end(), by_innovation)) {
            t_sorted_2 = a_G.m_LinkGenes;
            std::sort(t_sorted_2.begin(), t_sorted_2.end(), by_innovation);
            t_links_2 = &t_sorted_2;
        }
        t_g1 = t_links_1->begin();
        t_g2 = t_links_2->begin();
        // Step through the genes until both genomes end
        while (!((t_g1 == t_links_1->end()) && ((t_g2 == t_links_2->end())))) {
            // end of first genome?
            if (t_g1 == t_links_1->end()) {
                // add to the total distance
                t_num_excess++;
                t_g2++;
            } else if (t_g2 == t_links_2->end())
            // end of second genome?
            {
                // add to the total distance
                t_num_excess++;
                t_g1++;
            } else {
                // extract the innovation numbers
                int t_g1innov = t_g1->InnovationID();
                int t_g2innov = t_g2->InnovationID();

                // matching genes?
                if (t_g1innov == t_g2innov) {
                    t_num_matching_links++;

                    if (a_Parameters.WeightDiffCoeff > 0.0) {
                        Real t_wdiff = (t_g1->GetWeight() - t_g2->GetWeight());
                        if (t_wdiff < 0) t_wdiff = -t_wdiff;  // make sure it is positive
                        t_total_weight_difference += t_wdiff;
                    }

                    if (a_Parameters.SpikingLinkDiffCoeff > 0.0) {
                        const LinkGene &t_first = *t_g1;
                        const LinkGene &t_second = *t_g2;
                        auto normalized = [](Real lhs, Real rhs, Real minimum, Real maximum) {
                            const Real span = maximum - minimum;
                            return span > 0.0 ? std::abs(lhs - rhs) / span : (lhs == rhs ? 0.0 : 1.0);
                        };
                        Real difference = 0.0;
                        difference +=
                            normalized(t_first.m_SynapticDelay, t_second.m_SynapticDelay, a_Parameters.MinSynapticDelay, a_Parameters.MaxSynapticDelay);
                        difference += normalized(t_first.m_SynapticTimeConstant, t_second.m_SynapticTimeConstant, a_Parameters.MinSynapticTimeConstant,
                                                 a_Parameters.MaxSynapticTimeConstant);
                        difference += (t_first.m_STDPEnabled == t_second.m_STDPEnabled) ? 0.0 : 1.0;
                        difference += normalized(t_first.m_STDPPlus, t_second.m_STDPPlus, a_Parameters.MinSTDPPlus, a_Parameters.MaxSTDPPlus);
                        difference += normalized(t_first.m_STDPMinus, t_second.m_STDPMinus, a_Parameters.MinSTDPMinus, a_Parameters.MaxSTDPMinus);
                        difference += normalized(t_first.m_STDPTauPlus, t_second.m_STDPTauPlus, a_Parameters.MinSTDPTau, a_Parameters.MaxSTDPTau);
                        difference += normalized(t_first.m_STDPTauMinus, t_second.m_STDPTauMinus, a_Parameters.MinSTDPTau, a_Parameters.MaxSTDPTau);
                        t_total_spiking_link_difference += difference / 7.0;
                    }

                    // calculate link trait difference here
                    std::map<std::string, Real> link_trait_difference = t_g1->GetTraitDistances(t_g2->m_Traits);
                    // add to the totals
                    for (auto it = link_trait_difference.begin(); it != link_trait_difference.end(); it++) {
                        if (t_total_link_trait_difference.count(it->first) == 0) {
                            t_total_link_trait_difference[it->first] = it->second;
                        } else {
                            t_total_link_trait_difference[it->first] += it->second;
                        }
                    }

                    t_g1++;
                    t_g2++;
                } else if (t_g1innov < t_g2innov)  // disjoint
                {
                    t_num_disjoint++;
                    t_g1++;
                } else if (t_g1innov > t_g2innov)  // disjoint
                {
                    t_num_disjoint++;
                    t_g2++;
                }
            }
        }

        // find matching neuron IDs
        // One ID->index table for the other genome (instead of HasNeuronID +
        // repeated GetNeuronByID linear rescans per matched neuron, which made
        // this loop quadratic).
        int t_other_max_id = 0;
        for (unsigned int i = 0; i < a_G.NumNeurons(); i++) {
            if (a_G.m_NeuronGenes[i].ID() > t_other_max_id) {
                t_other_max_id = a_G.m_NeuronGenes[i].ID();
            }
        }
        std::vector<int> t_other_index(static_cast<size_t>(t_other_max_id) + 1, -1);
        for (unsigned int i = 0; i < a_G.NumNeurons(); i++) {
            t_other_index[static_cast<size_t>(a_G.m_NeuronGenes[i].ID())] = static_cast<int>(i);
        }

        for (unsigned int i = NumInputs(); i < NumNeurons(); i++) {
            // no inputs considered for comparison
            if ((m_NeuronGenes[i].Type() != INPUT) && (m_NeuronGenes[i].Type() != BIAS)) {
                const int t_id = m_NeuronGenes[i].ID();
                const int t_oi = (t_id >= 0 && t_id <= t_other_max_id) ? t_other_index[static_cast<size_t>(t_id)] : -1;
                // a match
                if (t_oi != -1) {
                    const NeuronGene &t_other_gene = a_G.m_NeuronGenes[static_cast<size_t>(t_oi)];
                    t_num_matching_neurons++;

                    if (a_Parameters.ActivationADiffCoeff > 0.0) {
                        Real t_A_difference = m_NeuronGenes[i].m_A - t_other_gene.m_A;
                        if (t_A_difference < 0.0f) t_A_difference = -t_A_difference;
                        t_total_A_difference += t_A_difference;
                    }

                    if (a_Parameters.ActivationBDiffCoeff > 0.0) {
                        Real t_B_difference = m_NeuronGenes[i].m_B - t_other_gene.m_B;
                        if (t_B_difference < 0.0f) t_B_difference = -t_B_difference;
                        t_total_B_difference += t_B_difference;
                    }

                    if (a_Parameters.TimeConstantDiffCoeff > 0.0) {
                        Real t_time_constant_difference = m_NeuronGenes[i].m_TimeConstant - t_other_gene.m_TimeConstant;
                        if (t_time_constant_difference < 0.0f) t_time_constant_difference = -t_time_constant_difference;
                        t_total_timeconstant_difference += t_time_constant_difference;
                    }

                    if (a_Parameters.BiasDiffCoeff > 0.0) {
                        Real t_bias_difference = m_NeuronGenes[i].m_Bias - t_other_gene.m_Bias;
                        if (t_bias_difference < 0.0f) t_bias_difference = -t_bias_difference;
                        t_total_bias_difference += t_bias_difference;
                    }

                    // Activation function type difference is found
                    if (a_Parameters.ActivationFunctionDiffCoeff > 0.0) {
                        if (m_NeuronGenes[i].m_ActFunction != t_other_gene.m_ActFunction) {
                            t_total_num_activation_difference++;
                        }
                    }

                    if (a_Parameters.SpikingNeuronDiffCoeff > 0.0 &&
                        (IsSpikingActivation(m_NeuronGenes[i].m_ActFunction) || IsSpikingActivation(t_other_gene.m_ActFunction))) {
                        const NeuronGene &t_mine = m_NeuronGenes[i];
                        auto normalized = [](Real lhs, Real rhs, Real minimum, Real maximum) {
                            const Real span = maximum - minimum;
                            return span > 0.0 ? std::abs(lhs - rhs) / span : (lhs == rhs ? 0.0 : 1.0);
                        };
                        Real difference = 0.0;
                        difference +=
                            normalized(t_mine.m_SpikeThreshold, t_other_gene.m_SpikeThreshold, a_Parameters.MinSpikeThreshold, a_Parameters.MaxSpikeThreshold);
                        difference +=
                            normalized(t_mine.m_ResetPotential, t_other_gene.m_ResetPotential, a_Parameters.MinResetPotential, a_Parameters.MaxResetPotential);
                        difference += normalized(t_mine.m_RestingPotential, t_other_gene.m_RestingPotential, a_Parameters.MinRestingPotential,
                                                 a_Parameters.MaxRestingPotential);
                        difference += normalized(t_mine.m_RefractoryPeriod, t_other_gene.m_RefractoryPeriod, a_Parameters.MinRefractoryPeriod,
                                                 a_Parameters.MaxRefractoryPeriod);
                        difference += normalized(t_mine.m_MembraneResistance, t_other_gene.m_MembraneResistance, a_Parameters.MinMembraneResistance,
                                                 a_Parameters.MaxMembraneResistance);
                        difference += (t_mine.m_MCPInhibitoryVeto == t_other_gene.m_MCPInhibitoryVeto) ? 0.0 : 1.0;
                        t_total_spiking_neuron_difference += difference / 6.0;
                    }

                    // calculate and add node trait difference here
                    std::map<std::string, Real> neuron_trait_difference = m_NeuronGenes[i].GetTraitDistances(t_other_gene.m_Traits);
                    // add to the totals
                    for (auto it = neuron_trait_difference.begin(); it != neuron_trait_difference.end(); it++) {
                        if (t_total_neuron_trait_difference.count(it->first) == 0) {
                            t_total_neuron_trait_difference[it->first] = it->second;
                        } else {
                            t_total_neuron_trait_difference[it->first] += it->second;
                        }
                    }
                }
            }
        }

        // choose between normalizing for genome size or not
        Real t_normalizer = 1.0;
        if (a_Parameters.NormalizeGenomeSize) {
            t_normalizer = static_cast<Real>(t_max_genome_size);
        }

        // if there are no matching links or neurons, make it 1.0 to avoid divide error
        if (t_num_matching_links <= 0) t_num_matching_links = 1;
        if (t_num_matching_neurons <= 0) t_num_matching_neurons = 1;
        if (t_normalizer <= 0.0) t_normalizer = 1.0;
        Real tnrm = 1.0 / t_normalizer;
        Real tnml = 1.0 / t_num_matching_links;
        Real tnmn = 1.0 / t_num_matching_neurons;

        t_total_distance =
            (a_Parameters.ExcessCoeff * (t_num_excess * tnrm)) + (a_Parameters.DisjointCoeff * (t_num_disjoint * tnrm)) +
            (a_Parameters.WeightDiffCoeff * (t_total_weight_difference * tnml)) + (a_Parameters.ActivationADiffCoeff * (t_total_A_difference * tnmn)) +
            (a_Parameters.ActivationBDiffCoeff * (t_total_B_difference * tnmn)) +
            (a_Parameters.TimeConstantDiffCoeff * (t_total_timeconstant_difference * tnmn)) + (a_Parameters.BiasDiffCoeff * (t_total_bias_difference * tnmn)) +
            (a_Parameters.ActivationFunctionDiffCoeff * (t_total_num_activation_difference * tnmn)) +
            (a_Parameters.SpikingNeuronDiffCoeff * (t_total_spiking_neuron_difference * tnmn)) +
            (a_Parameters.SpikingLinkDiffCoeff * (t_total_spiking_link_difference * tnml));

        // add trait differences according to each one's coeff (find-guarded:
        // genomes may carry traits absent from the schema)

        for (auto it = t_total_link_trait_difference.begin(); it != t_total_link_trait_difference.end(); it++) {
            const auto schema = a_Parameters.LinkTraits.find(it->first);
            if (schema == a_Parameters.LinkTraits.end()) continue;
            Real n = (schema->second.m_ImportanceCoeff * it->second) * tnml;
            if (std::isnan(n) || std::isinf(n)) n = 0.0;
            t_total_distance += n;
        }
        for (auto it = t_total_neuron_trait_difference.begin(); it != t_total_neuron_trait_difference.end(); it++) {
            const auto schema = a_Parameters.NeuronTraits.find(it->first);
            if (schema == a_Parameters.NeuronTraits.end()) continue;
            Real n = (schema->second.m_ImportanceCoeff * it->second) * tnmn;
            if (std::isnan(n) || std::isinf(n)) n = 0.0;
            t_total_distance += n;
        }
        for (auto it = t_genome_link_trait_difference.begin(); it != t_genome_link_trait_difference.end(); it++) {
            const auto schema = a_Parameters.GenomeTraits.find(it->first);
            if (schema == a_Parameters.GenomeTraits.end()) continue;
            Real n = (schema->second.m_ImportanceCoeff * it->second);
            if (std::isnan(n) || std::isinf(n)) n = 0.0;
            t_total_distance += n;
        }

        // store in cache
        // distance_cache[std::make_pair(this->GetID(), a_G.GetID())] = t_total_distance;

        return t_total_distance;
    }

    // Returns true if this genome and a_G are compatible (belong in the same species)
    bool Genome::IsCompatibleWith(Genome &a_G, Parameters &a_Parameters) {
        // full compatibility cases
        if (this == &a_G) return true;

        Real t_total_distance = CompatibilityDistance(a_G, a_Parameters);

        /*if ((NumLinks() == 0) && (a_G.NumLinks() == 0))
            return true;*/

        if (t_total_distance <= a_Parameters.CompatTreshold)
            return true;  // compatible
        else
            return false;  // incompatible
    }

    // Returns a random activation function from the canonical set based ot probabilities
    ActivationFunction GetRandomActivation(const Parameters &a_Parameters, RNG &a_RNG) {
        std::vector<Real> t_probs;

        t_probs.emplace_back(a_Parameters.ActivationFunction_SignedSigmoid_Prob);
        t_probs.emplace_back(a_Parameters.ActivationFunction_UnsignedSigmoid_Prob);
        t_probs.emplace_back(a_Parameters.ActivationFunction_Tanh_Prob);
        t_probs.emplace_back(a_Parameters.ActivationFunction_TanhCubic_Prob);
        t_probs.emplace_back(a_Parameters.ActivationFunction_SignedStep_Prob);
        t_probs.emplace_back(a_Parameters.ActivationFunction_UnsignedStep_Prob);
        t_probs.emplace_back(a_Parameters.ActivationFunction_SignedGauss_Prob);
        t_probs.emplace_back(a_Parameters.ActivationFunction_UnsignedGauss_Prob);
        t_probs.emplace_back(a_Parameters.ActivationFunction_Abs_Prob);
        t_probs.emplace_back(a_Parameters.ActivationFunction_SignedSine_Prob);
        t_probs.emplace_back(a_Parameters.ActivationFunction_UnsignedSine_Prob);
        t_probs.emplace_back(a_Parameters.ActivationFunction_Linear_Prob);
        t_probs.emplace_back(a_Parameters.ActivationFunction_Relu_Prob);
        t_probs.emplace_back(a_Parameters.ActivationFunction_Softplus_Prob);
        t_probs.emplace_back(a_Parameters.ActivationFunction_SpikingLIF_Prob);
        t_probs.emplace_back(a_Parameters.ActivationFunction_SpikingAdaptiveLIF_Prob);
        t_probs.emplace_back(a_Parameters.ActivationFunction_SpikingIzhikevich_Prob);
        t_probs.emplace_back(a_Parameters.ActivationFunction_McCullochPitts_Prob);

        Real total = 0.0;
        for (const Real probability : t_probs) {
            if (!std::isfinite(probability) || probability < 0.0)
                throw std::invalid_argument("Activation-function probabilities must be finite and non-negative");
            total += probability;
        }
        if (!std::isfinite(total) || total <= 0.0) throw std::invalid_argument("At least one activation function must have positive probability");

        return (NEAT::ActivationFunction)a_RNG.Roulette(t_probs);
    }

    // Adds a new neuron to the genome returns true if succesful
    bool Genome::Mutate_AddNeuron(InnovationDatabase &a_Innovs, const Parameters &a_Parameters, RNG &a_RNG) {
        // No links to split - go away..
        if (NumLinks() == 0 || a_Parameters.NeuronTries <= 0) return false;

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

        // Collect splittable links up front (skipping bias sources and
        // disallowed recurrent links) and pick one uniformly, instead of
        // rejection-sampling up to 256 times.
        std::vector<std::size_t> t_eligible_links;
        t_eligible_links.reserve(m_LinkGenes.size());
        for (std::size_t t_index = 0; t_index < m_LinkGenes.size(); ++t_index) {
            const LinkGene &t_link = m_LinkGenes[t_index];
            const int t_source_index = GetNeuronIndex(t_link.FromNeuronID());
            if (t_source_index < 0) continue;
            if (!a_Parameters.DontUseBiasNeuron && m_NeuronGenes[static_cast<std::size_t>(t_source_index)].Type() == BIAS) {
                continue;
            }
            if (t_link.IsRecurrent()) {
                if (t_link.IsLoopedRecurrent()) {
                    if (!a_Parameters.SplitLoopedRecurrent) continue;
                } else if (!a_Parameters.SplitRecurrent) {
                    continue;
                }
            }
            t_eligible_links.push_back(t_index);
        }
        if (t_eligible_links.empty()) return false;

        const std::size_t t_link_num = t_eligible_links[static_cast<std::size_t>(a_RNG.RandInt(0, static_cast<int>(t_eligible_links.size()) - 1))];
        int t_in = 0, t_out = 0;
        LinkGene t_chosenlink(0, 0, -1, 0, false);  // to save it for later

        {
            t_in = m_LinkGenes[t_link_num].FromNeuronID();
            t_out = m_LinkGenes[t_link_num].ToNeuronID();

            ASSERT((t_in > 0) && (t_out > 0));

            // In case there is only one link, coming from a bias - just quit

            // unless the parameter is set
            if (a_Parameters.DontUseBiasNeuron == false) {
                if ((m_NeuronGenes[GetNeuronIndex(t_in)].Type() == BIAS) && (NumLinks() == 1)) {
                    return false;
                }
            }
        }
        // Now the link has been selected

        // the weight of the link that is being split
        Real t_orig_weight = m_LinkGenes[t_link_num].GetWeight();
        t_chosenlink = m_LinkGenes[t_link_num];  // save the whole link

        // remove the link from the genome
        // TODO: add option to keep the link, but disabled
        RemoveLinkGene(m_LinkGenes[t_link_num].InnovationID());

        // Check if an innovation of this type already occured somewhere in the population
        int t_innovid = a_Innovs.CheckInnovation(t_in, t_out, NEW_NEURON);

        // the new neuron and links ids
        int t_nid = 0;
        int t_l1id = 0;
        int t_l2id = 0;

        // This is a novel innovation?
        if (t_innovid == -1) {
            // Add the new neuron innovation
            t_nid = a_Innovs.AddNeuronInnovation(t_in, t_out, HIDDEN);
            // add the first link innovation
            t_l1id = a_Innovs.AddLinkInnovation(t_in, t_nid);
            // add the second innovation
            t_l2id = a_Innovs.AddLinkInnovation(t_nid, t_out);

            // Adjust the SplitY
            Real t_sy = m_NeuronGenes[GetNeuronIndex(t_in)].SplitY() + m_NeuronGenes[GetNeuronIndex(t_out)].SplitY();
            t_sy /= 2.0;

            // Create the neuron gene
            NeuronGene t_ngene(HIDDEN, t_nid, t_sy);

            Real t_A = a_RNG.RandFloat();
            Real t_B = a_RNG.RandFloat();
            Real t_TC = a_RNG.RandFloat();
            Real t_Bs = a_RNG.RandFloat();
            Scale(t_A, 0, 1, a_Parameters.MinActivationA, a_Parameters.MaxActivationA);
            Scale(t_B, 0, 1, a_Parameters.MinActivationB, a_Parameters.MaxActivationB);
            Scale(t_TC, 0, 1, a_Parameters.MinNeuronTimeConstant, a_Parameters.MaxNeuronTimeConstant);
            Scale(t_Bs, 0, 1, a_Parameters.MinNeuronBias, a_Parameters.MaxNeuronBias);

            Clamp(t_A, a_Parameters.MinActivationA, a_Parameters.MaxActivationA);
            Clamp(t_B, a_Parameters.MinActivationB, a_Parameters.MaxActivationB);
            Clamp(t_TC, a_Parameters.MinNeuronTimeConstant, a_Parameters.MaxNeuronTimeConstant);
            Clamp(t_Bs, a_Parameters.MinNeuronBias, a_Parameters.MaxNeuronBias);

            // Initialize the neuron gene's properties
            t_ngene.Init(t_A, t_B, t_TC, t_Bs, GetRandomActivation(a_Parameters, a_RNG));
            InitializeNeuronSpiking(t_ngene, a_Parameters, &a_RNG);

            // Initialize the traits
            // if (a_RNG.RandFloat() < 0.5)
            //{
            t_ngene.InitTraits(a_Parameters.NeuronTraits, a_RNG);
            //}
            // else
            //{   // mate instead of randomizing
            //    t_ngene.m_Traits = m_NeuronGenes[GetNeuronIndex(t_in)].m_Traits;
            //    t_ngene.MateTraits(m_NeuronGenes[GetNeuronIndex(t_out)].m_Traits, a_RNG);
            //}

            // Add the NeuronGene
            m_NeuronGenes.emplace_back(t_ngene);

            // Now the links

            // Make sure the recurrent flag is kept
            bool t_recurrentflag = t_chosenlink.IsRecurrent();

            // First link
            LinkGene l1 = LinkGene(t_in, t_nid, t_l1id, 1.0, t_recurrentflag);
            // make sure this weight is in the allowed interval
            Clamp(l1.m_Weight, a_Parameters.MinWeight, a_Parameters.MaxWeight);
            // Split synapses inherit the parent delay (halved) and STDP state.
            l1.m_SynapticDelay = t_chosenlink.m_SynapticDelay * 0.5;
            l1.m_SynapticTimeConstant = t_chosenlink.m_SynapticTimeConstant;
            l1.m_STDPEnabled = t_chosenlink.m_STDPEnabled;
            l1.m_STDPPlus = t_chosenlink.m_STDPPlus;
            l1.m_STDPMinus = t_chosenlink.m_STDPMinus;
            l1.m_STDPTauPlus = t_chosenlink.m_STDPTauPlus;
            l1.m_STDPTauMinus = t_chosenlink.m_STDPTauMinus;
            l1.m_STDPMinWeight = t_chosenlink.m_STDPMinWeight;
            l1.m_STDPMaxWeight = t_chosenlink.m_STDPMaxWeight;
            // Init the link's traits
            l1.InitTraits(a_Parameters.LinkTraits, a_RNG);
            m_LinkGenes.emplace_back(l1);

            // Second link
            LinkGene l2 = LinkGene(t_nid, t_out, t_l2id, t_orig_weight, t_recurrentflag);
            l2.m_SynapticDelay = t_chosenlink.m_SynapticDelay * 0.5;
            l2.m_SynapticTimeConstant = t_chosenlink.m_SynapticTimeConstant;
            l2.m_STDPEnabled = t_chosenlink.m_STDPEnabled;
            l2.m_STDPPlus = t_chosenlink.m_STDPPlus;
            l2.m_STDPMinus = t_chosenlink.m_STDPMinus;
            l2.m_STDPTauPlus = t_chosenlink.m_STDPTauPlus;
            l2.m_STDPTauMinus = t_chosenlink.m_STDPTauMinus;
            l2.m_STDPMinWeight = t_chosenlink.m_STDPMinWeight;
            l2.m_STDPMaxWeight = t_chosenlink.m_STDPMaxWeight;
            // Init the link's traits
            l2.InitTraits(a_Parameters.LinkTraits, a_RNG);
            m_LinkGenes.emplace_back(l2);
        } else {
            // This innovation already happened, so inherit it.

            // get the neuron ID
            t_nid = a_Innovs.FindNeuronID(t_in, t_out);
            ASSERT(t_nid != -1);

            // if such an innovation happened, these must exist
            t_l1id = a_Innovs.CheckInnovation(t_in, t_nid, NEW_LINK);
            t_l2id = a_Innovs.CheckInnovation(t_nid, t_out, NEW_LINK);

            ASSERT((t_l1id > 0) && (t_l2id > 0));

            // Perhaps this innovation occured more than once. Find the first such innovation that had occured, but the genome not having the same id.. If
            // didn't find such, then add new innovation.
            std::vector<int> t_idxs = a_Innovs.CheckAllInnovations(t_in, t_out, NEW_NEURON);
            bool t_found = false;
            for (unsigned int i = 0; i < t_idxs.size(); i++) {
                if (!HasNeuronID(a_Innovs.GetInnovationByIdx(t_idxs[i]).NeuronID())) {
                    // found such innovation & this genome doesn't have that neuron ID
                    // So we are going to inherit the innovation
                    t_nid = a_Innovs.GetInnovationByIdx(t_idxs[i]).NeuronID();

                    // these must exist
                    t_l1id = a_Innovs.CheckInnovation(t_in, t_nid, NEW_LINK);
                    t_l2id = a_Innovs.CheckInnovation(t_nid, t_out, NEW_LINK);

                    ASSERT((t_l1id > 0) && (t_l2id > 0));

                    t_found = true;
                    break;
                }
            }

            // Such an innovation was not found or the genome has all neuron IDs So we are going to add new innovation
            if (!t_found) {
                // Add 3 new innovations and replace the variables with them

                // Add the new neuron innovation
                t_nid = a_Innovs.AddNeuronInnovation(t_in, t_out, HIDDEN);
                // add the first link innovation
                t_l1id = a_Innovs.AddLinkInnovation(t_in, t_nid);
                // add the second innovation
                t_l2id = a_Innovs.AddLinkInnovation(t_nid, t_out);
            }

            // Add the neuron and the links
            Real t_sy = m_NeuronGenes[GetNeuronIndex(t_in)].SplitY() + m_NeuronGenes[GetNeuronIndex(t_out)].SplitY();
            t_sy /= 2.0;

            // Create the neuron gene
            NeuronGene t_ngene(HIDDEN, t_nid, t_sy);

            Real t_A = a_RNG.RandFloat();
            Real t_B = a_RNG.RandFloat();
            Real t_TC = a_RNG.RandFloat();
            Real t_Bs = a_RNG.RandFloat();
            Scale(t_A, 0, 1, a_Parameters.MinActivationA, a_Parameters.MaxActivationA);
            Scale(t_B, 0, 1, a_Parameters.MinActivationB, a_Parameters.MaxActivationB);
            Scale(t_TC, 0, 1, a_Parameters.MinNeuronTimeConstant, a_Parameters.MaxNeuronTimeConstant);
            Scale(t_Bs, 0, 1, a_Parameters.MinNeuronBias, a_Parameters.MaxNeuronBias);

            Clamp(t_A, a_Parameters.MinActivationA, a_Parameters.MaxActivationA);
            Clamp(t_B, a_Parameters.MinActivationB, a_Parameters.MaxActivationB);
            Clamp(t_TC, a_Parameters.MinNeuronTimeConstant, a_Parameters.MaxNeuronTimeConstant);
            Clamp(t_Bs, a_Parameters.MinNeuronBias, a_Parameters.MaxNeuronBias);

            // Initialize the neuron gene's properties
            t_ngene.Init(t_A, t_B, t_TC, t_Bs, GetRandomActivation(a_Parameters, a_RNG));
            InitializeNeuronSpiking(t_ngene, a_Parameters, &a_RNG);

            // Initialize the traits
            // if (a_RNG.RandFloat() < 0.5)
            //{
            t_ngene.InitTraits(a_Parameters.NeuronTraits, a_RNG);
            //}// mate instead of randomizing
            // else
            //{
            //    t_ngene.m_Traits = m_NeuronGenes[GetNeuronIndex(t_in)].m_Traits;
            //    t_ngene.MateTraits(m_NeuronGenes[GetNeuronIndex(t_out)].m_Traits, a_RNG);
            //}

            // Make sure the recurrent flag is kept
            bool t_recurrentflag = t_chosenlink.IsRecurrent();

            // Add the NeuronGene
            m_NeuronGenes.emplace_back(t_ngene);
            LinkGene l1 = LinkGene(t_in, t_nid, t_l1id, 1.0, t_recurrentflag);
            // make sure this weight is in the allowed interval
            Clamp(l1.m_Weight, a_Parameters.MinWeight, a_Parameters.MaxWeight);
            // Split synapses inherit the parent delay (halved) and STDP state.
            l1.m_SynapticDelay = t_chosenlink.m_SynapticDelay * 0.5;
            l1.m_SynapticTimeConstant = t_chosenlink.m_SynapticTimeConstant;
            l1.m_STDPEnabled = t_chosenlink.m_STDPEnabled;
            l1.m_STDPPlus = t_chosenlink.m_STDPPlus;
            l1.m_STDPMinus = t_chosenlink.m_STDPMinus;
            l1.m_STDPTauPlus = t_chosenlink.m_STDPTauPlus;
            l1.m_STDPTauMinus = t_chosenlink.m_STDPTauMinus;
            l1.m_STDPMinWeight = t_chosenlink.m_STDPMinWeight;
            l1.m_STDPMaxWeight = t_chosenlink.m_STDPMaxWeight;
            // initialize the link's traits
            l1.InitTraits(a_Parameters.LinkTraits, a_RNG);
            m_LinkGenes.emplace_back(l1);

            // Second link
            LinkGene l2 = LinkGene(t_nid, t_out, t_l2id, t_orig_weight, t_recurrentflag);
            l2.m_SynapticDelay = t_chosenlink.m_SynapticDelay * 0.5;
            l2.m_SynapticTimeConstant = t_chosenlink.m_SynapticTimeConstant;
            l2.m_STDPEnabled = t_chosenlink.m_STDPEnabled;
            l2.m_STDPPlus = t_chosenlink.m_STDPPlus;
            l2.m_STDPMinus = t_chosenlink.m_STDPMinus;
            l2.m_STDPTauPlus = t_chosenlink.m_STDPTauPlus;
            l2.m_STDPTauMinus = t_chosenlink.m_STDPTauMinus;
            l2.m_STDPMinWeight = t_chosenlink.m_STDPMinWeight;
            l2.m_STDPMaxWeight = t_chosenlink.m_STDPMaxWeight;
            // initialize the link's traits
            l2.InitTraits(a_Parameters.LinkTraits, a_RNG);
            m_LinkGenes.emplace_back(l2);
        }

        return true;
    }

    // Adds a new link to the genome returns true if succesful
    bool Genome::Mutate_AddLink(InnovationDatabase &a_Innovs, const Parameters &a_Parameters, RNG &a_RNG) {
        if (m_NeuronGenes.empty() || a_Parameters.LinkTries == 0) return false;

        // The pair of neurons that has to be connected (1 - in, 2 - out)
        // It may be the same neuron - this means that the connection is a looped recurrent one.
        // These are indexes in the NeuronGenes array!
        int t_n1idx = 0, t_n2idx = 0;

        // Should we make this connection recurrent?
        bool t_MakeRecurrent = false;

        // If so, should it be a looped one?
        bool t_LoopedRecurrent = false;

        // Should it come from the bias neuron?
        bool t_MakeBias = false;

        // Decide whether the connection will be recurrent or not..
        if (a_RNG.RandFloat() < a_Parameters.RecurrentProb) {
            t_MakeRecurrent = true;

            if (a_RNG.RandFloat() < a_Parameters.RecurrentLoopProb) {
                t_LoopedRecurrent = true;
            }
        }
        // if not recurrent, there is a probability that this link will be from the bias
        // if such link doesn't already exist.
        // in case such link exists, search for a standard feed-forward connection place
        else {
            if (!a_Parameters.DontUseBiasNeuron && a_RNG.RandFloat() < a_Parameters.MutateAddLinkFromBiasProb) {
                t_MakeBias = true;
            }
        }

        // Find a good pair of neurons. Forward-reachability for all neurons is
        // computed once (reverse-topological bitset DP); bounded random
        // sampling then finds a pair fast in the common case, with an
        // exhaustive two-pass uniform fallback guaranteeing completeness.
        // Forward links can never close a directed cycle through the
        // feedforward links.
        const auto t_is_non_input = [](NeuronType type) { return type != INPUT && type != BIAS; };

        std::unordered_map<int, std::size_t> t_id_to_index;
        t_id_to_index.reserve(m_NeuronGenes.size() * 2);
        for (std::size_t i = 0; i < m_NeuronGenes.size(); ++i) t_id_to_index.emplace(m_NeuronGenes[i].ID(), i);
        const std::size_t t_vertex_count = m_NeuronGenes.size();
        std::vector<std::vector<std::size_t>> t_successors(t_vertex_count);
        std::vector<std::size_t> t_indegree(t_vertex_count, 0);
        for (const LinkGene &t_link : m_LinkGenes) {
            if (t_link.IsRecurrent()) continue;
            const auto t_source = t_id_to_index.find(t_link.FromNeuronID());
            const auto t_target = t_id_to_index.find(t_link.ToNeuronID());
            if (t_source == t_id_to_index.end() || t_target == t_id_to_index.end()) continue;
            t_successors[t_source->second].push_back(t_target->second);
            ++t_indegree[t_target->second];
        }
        // Kahn order (feedforward links are acyclic by construction, but
        // tolerate legacy cycles by appending leftovers in index order).
        std::vector<std::size_t> t_topo;
        t_topo.reserve(t_vertex_count);
        std::vector<std::size_t> t_order_stack;
        for (std::size_t i = 0; i < t_vertex_count; ++i) {
            if (t_indegree[i] == 0) t_order_stack.push_back(i);
        }
        while (!t_order_stack.empty()) {
            const std::size_t current = t_order_stack.back();
            t_order_stack.pop_back();
            t_topo.push_back(current);
            for (std::size_t next : t_successors[current]) {
                if (--t_indegree[next] == 0) t_order_stack.push_back(next);
            }
        }
        for (std::size_t i = 0; i < t_vertex_count; ++i) {
            bool seen = false;
            for (std::size_t t : t_topo) {
                if (t == i) {
                    seen = true;
                    break;
                }
            }
            if (!seen) t_topo.push_back(i);
        }
        const std::size_t t_words = (t_vertex_count + 63) / 64;
        std::vector<std::uint64_t> t_reach(t_vertex_count * t_words, 0);
        for (std::size_t ti = t_vertex_count; ti > 0; --ti) {
            const std::size_t u = t_topo[ti - 1];
            for (std::size_t v : t_successors[u]) {
                t_reach[u * t_words + v / 64] |= 1ULL << (v % 64);
                const std::uint64_t *row_v = &t_reach[v * t_words];
                std::uint64_t *row_u = &t_reach[u * t_words];
                for (std::size_t w = 0; w < t_words; ++w) row_u[w] |= row_v[w];
            }
        }

        const auto t_endpoint_key = [](int source, int target) {
            return (static_cast<std::uint64_t>(static_cast<std::uint32_t>(source)) << 32U) | static_cast<std::uint32_t>(target);
        };
        std::vector<std::uint64_t> t_existing_links;
        t_existing_links.reserve(m_LinkGenes.size());
        for (const LinkGene &t_link : m_LinkGenes) {
            t_existing_links.push_back(t_endpoint_key(t_link.FromNeuronID(), t_link.ToNeuronID()));
        }
        std::sort(t_existing_links.begin(), t_existing_links.end());
        const auto t_link_exists = [&](std::size_t source, std::size_t target) {
            return std::binary_search(t_existing_links.begin(), t_existing_links.end(), t_endpoint_key(m_NeuronGenes[source].ID(), m_NeuronGenes[target].ID()));
        };

        const auto t_pair_valid = [&](std::size_t source, std::size_t target) {
            const NeuronType source_type = m_NeuronGenes[source].Type();
            const NeuronType target_type = m_NeuronGenes[target].Type();
            if (!t_is_non_input(target_type)) return false;
            if (t_MakeBias) return source_type == BIAS && !t_link_exists(source, target);
            if (!t_MakeRecurrent) {
                if (source == target || source_type == OUTPUT) return false;
                const bool closes_cycle = ((t_reach[target * t_words + source / 64] >> (source % 64)) & 1ULL) != 0;
                return !closes_cycle && !t_link_exists(source, target);
            }
            if (!t_is_non_input(source_type)) return false;
            if (t_LoopedRecurrent) return source == target && !t_link_exists(source, target);
            return source != target && !t_link_exists(source, target);
        };

        bool t_found = false;
        {
            // Collect non-input indexes once for the sampling draws.
            std::vector<std::size_t> t_non_inputs;
            std::vector<std::size_t> t_biases;
            t_non_inputs.reserve(t_vertex_count);
            for (std::size_t i = 0; i < t_vertex_count; ++i) {
                if (t_is_non_input(m_NeuronGenes[i].Type())) t_non_inputs.push_back(i);
                if (m_NeuronGenes[i].Type() == BIAS) t_biases.push_back(i);
            }
            const int t_count = static_cast<int>(t_vertex_count);
            const int t_non_input_count = static_cast<int>(t_non_inputs.size());
            const int t_bias_count = static_cast<int>(t_biases.size());
            for (unsigned int attempt = 0; attempt < a_Parameters.LinkTries && !t_found; ++attempt) {
                std::size_t source = 0, target = 0;
                if (t_LoopedRecurrent) {
                    if (t_non_inputs.empty()) break;
                    source = target = t_non_inputs[static_cast<std::size_t>(a_RNG.RandInt(0, t_non_input_count - 1))];
                } else if (t_MakeRecurrent) {
                    if (t_non_inputs.empty()) break;
                    source = t_non_inputs[static_cast<std::size_t>(a_RNG.RandInt(0, t_non_input_count - 1))];
                    target = t_non_inputs[static_cast<std::size_t>(a_RNG.RandInt(0, t_non_input_count - 1))];
                } else if (t_MakeBias) {
                    if (t_biases.empty() || t_non_inputs.empty()) break;
                    source = t_biases[static_cast<std::size_t>(a_RNG.RandInt(0, t_bias_count - 1))];
                    target = t_non_inputs[static_cast<std::size_t>(a_RNG.RandInt(0, t_non_input_count - 1))];
                } else {
                    if (t_non_inputs.empty()) break;
                    source = static_cast<std::size_t>(a_RNG.RandInt(0, t_count - 1));
                    target = t_non_inputs[static_cast<std::size_t>(a_RNG.RandInt(0, t_non_input_count - 1))];
                }
                if (t_pair_valid(source, target)) {
                    t_n1idx = static_cast<int>(source);
                    t_n2idx = static_cast<int>(target);
                    t_found = true;
                }
            }
        }

        if (!t_found) {
            // Reservoir sampling: single source-major pass, uniform over the
            // valid pairs with no candidate storage.
            std::size_t t_seen = 0;
            bool t_picked = false;
            for (std::size_t source = 0; source < t_vertex_count; ++source) {
                for (std::size_t target = 0; target < t_vertex_count; ++target) {
                    if (!t_pair_valid(source, target)) continue;
                    ++t_seen;
                    if (t_seen > static_cast<std::size_t>(std::numeric_limits<int>::max())) return false;
                    if (static_cast<std::size_t>(a_RNG.RandInt(0, static_cast<int>(t_seen) - 1)) == 0) {
                        t_n1idx = static_cast<int>(source);
                        t_n2idx = static_cast<int>(target);
                        t_picked = true;
                    }
                }
            }
            if (!t_picked) return false;
        }

        // This link MUST NOT be a part of the genome by any reason
        ASSERT((!HasLink(m_NeuronGenes[t_n1idx].ID(), m_NeuronGenes[t_n2idx].ID())));  // already present?

        // extract the neuron IDs from the indexes
        int t_n1id = m_NeuronGenes[t_n1idx].ID();
        int t_n2id = m_NeuronGenes[t_n2idx].ID();

        // So we have a good pair of neurons to connect. See the innovation database if this is novel innovation.
        int t_innovid = a_Innovs.CheckInnovation(t_n1id, t_n2id, NEW_LINK);

        // Choose the weight for this link
        Real t_weight = a_RNG.RandFloat();
        Scale(t_weight, 0, 1, a_Parameters.MinWeight, a_Parameters.MaxWeight);

        // A novel innovation?
        if (t_innovid == -1) {
            // Make new innovation
            t_innovid = a_Innovs.AddLinkInnovation(t_n1id, t_n2id);
        }

        // Create and add the link
        LinkGene l = LinkGene(t_n1id, t_n2id, t_innovid, t_weight, t_MakeRecurrent);
        InitializeLinkSpiking(l, a_Parameters, &a_RNG);
        // init the link's traits
        l.InitTraits(a_Parameters.LinkTraits, a_RNG);
        m_LinkGenes.emplace_back(l);

        // All done.
        return true;
    }

    ///////////
    // Helper functions for the pruning procedure

    // Removes the link with the specified innovation ID
    /*void Genome::RemoveLinkGene(int a_InnovID)
    {
        // for iterating through the genes
        std::vector<LinkGene>::iterator t_curlink = m_LinkGenes.begin();

        while (t_curlink != m_LinkGenes.end())
        {
            if (t_curlink->InnovationID() == a_InnovID)
            {
                // found it - erase & quit
                t_curlink = m_LinkGenes.erase(t_curlink);
                break;
            }

            t_curlink++;
        }
    }*/

    // Removes the link with the given innovation ID (the semantics the header declares
    // and that Mutate_RemoveLink/Cleanup rely on). The previous "simple index" version
    // erased by position — wiping the whole link list whenever a_idx was 0, and
    // erasing out-of-range positions when callers (correctly) passed innovation IDs.
    void Genome::RemoveLinkGene(int a_innovid) {
        for (auto t_curlink = m_LinkGenes.begin(); t_curlink != m_LinkGenes.end(); ++t_curlink) {
            if (t_curlink->InnovationID() == a_innovid) {
                m_LinkGenes.erase(t_curlink);
                return;
            }
        }
    }

    // Remove node Links connected to this node are also removed
    void Genome::RemoveNeuronGene(int a_ID) {
        // the list of links connected to this neuron
        std::vector<int> t_link_removal_queue;

        bool removed = false;

        do {
            removed = false;
            // Remove all links connected to this neuron ID
            for (int i = 0; i < NumLinks(); i++) {
                if ((m_LinkGenes[i].FromNeuronID() == a_ID) || (m_LinkGenes[i].ToNeuronID() == a_ID)) {
                    // found one, remove it (by innovation ID, the sole RemoveLinkGene contract)
                    RemoveLinkGene(m_LinkGenes[i].InnovationID());
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
        std::vector<NeuronGene>::iterator t_curneuron = m_NeuronGenes.begin();

        while (t_curneuron != m_NeuronGenes.end()) {
            if (t_curneuron->ID() == a_ID) {
                // found it, erase and quit
                m_NeuronGenes.erase(t_curneuron);
                break;
            }

            t_curneuron++;
        }
    }

    // Returns true is the specified neuron ID is a dead end or isolated
    bool Genome::IsDeadEndNeuron(int a_ID) const {
        bool t_no_incoming = true;
        bool t_no_outgoing = true;
        int bias_id = -1;
        for (const NeuronGene &neuron : m_NeuronGenes) {
            if (neuron.Type() == BIAS) {
                bias_id = neuron.ID();
                break;
            }
        }

        for (size_t i = 0, end = m_LinkGenes.size(); i < end; ++i) {
            const LinkGene &l = m_LinkGenes[i];
            // there is a link going to this neuron, so there are incoming; don't count the link if it is looped recurrent or coming from a bias
            if ((l.ToNeuronID() == a_ID) && (!l.IsLoopedRecurrent()) && l.FromNeuronID() != bias_id) {
                t_no_incoming = false;
            }

            // there is a link going from this neuron, so there are outgoing; don't count the link if it is looped recurrent
            if ((l.FromNeuronID() == a_ID) && (!l.IsLoopedRecurrent())) {
                t_no_outgoing = false;
            }
            if (!t_no_incoming && !t_no_outgoing) return false;
        }

        // if just one of these is true, this neuron is a dead end
        return (t_no_incoming || t_no_outgoing);
    }

    // Search the genome for isolated structure and clean it up Returns true is something was removed
    bool Genome::Cleanup() {
        bool t_removed = false;

        // remove any dead-end hidden neurons (index-free loop: erasing shifts
        // later elements, so only advance when nothing was removed)
        for (std::size_t i = 0; i < m_NeuronGenes.size();) {
            if (m_NeuronGenes[i].Type() == HIDDEN && IsDeadEndNeuron(m_NeuronGenes[i].ID())) {
                RemoveNeuronGene(m_NeuronGenes[i].ID());
                t_removed = true;
                continue;
            }
            ++i;
        }

        // a special case are isolated outputs - these are outputs having
        // one and only one looped recurrent connection
        // we simply remove these connections and leave the outputs naked.
        for (std::size_t i = 0; i < m_NeuronGenes.size(); ++i) {
            if (m_NeuronGenes[i].Type() == OUTPUT) {
                // Only outputs with 1 input and 1 output connection are considered.
                if ((LinksInputtingFrom(m_NeuronGenes[i].ID()) == 1) && (LinksOutputtingTo(m_NeuronGenes[i].ID()) == 1)) {
                    // that must be a lonely looped recurrent,
                    // because we know that the outputs are the dead end of the network
                    // find this link
                    for (std::size_t j = 0; j < m_LinkGenes.size(); ++j) {
                        if (m_LinkGenes[j].ToNeuronID() == m_NeuronGenes[i].ID()) {
                            // Remove it.
                            RemoveLinkGene(m_LinkGenes[j].InnovationID());
                            t_removed = true;
                            break;
                        }
                    }
                }
                if (NumOutputs() == 1)
                    if ((LinksInputtingFrom(m_NeuronGenes[i].ID()) == 0) && (LinksOutputtingTo(m_NeuronGenes[i].ID()) == 0)) return true;
            }
        }

        return t_removed;
    }

    // Returns true if has any dead end
    bool Genome::HasDeadEnds() const {
        // any dead-end hidden neurons?
        for (unsigned int i = 0; i < NumNeurons(); i++) {
            if (m_NeuronGenes[i].Type() == HIDDEN) {
                if (IsDeadEndNeuron(m_NeuronGenes[i].ID())) {
                    return true;
                }
            }
        }

        // a special case are isolated outputs - these are outputs having
        // one and only one looped recurrent connection or no connections at all
        for (unsigned int i = 0; i < NumNeurons(); i++) {
            if (m_NeuronGenes[i].Type() == OUTPUT) {
                // Only outputs with 1 input and 1 output connection are considered.
                if ((LinksInputtingFrom(m_NeuronGenes[i].ID()) == 1) && (LinksOutputtingTo(m_NeuronGenes[i].ID()) == 1)) {
                    // that must be a lonely looped recurrent,
                    // because we know that the outputs are the dead end of the network
                    return true;
                }

                // There may be cases for totally isolated outputs Consider this if only one output is present
                if (NumOutputs() == 1)
                    if ((LinksInputtingFrom(m_NeuronGenes[i].ID()) == 0) && (LinksOutputtingTo(m_NeuronGenes[i].ID()) == 0)) {
                        return true;
                    }
            }
        }

        return false;
    }

    // Remove a link from the genome
    // A cleanup procedure is invoked so any dead-ends or stranded neurons are also deleted
    // returns true if succesful
    bool Genome::Mutate_RemoveLink(RNG &a_RNG) {
        // at least 2 links must be present in the genome
        if (NumLinks() < 2) return false;

        // find a uniformly random link to remove
        int t_link_index = a_RNG.RandInt(0, static_cast<int>(NumLinks()) - 1);

        // remove it
        RemoveLinkGene(m_LinkGenes[t_link_index].InnovationID());

        // Now cleanup
        // Cleanup();

        return true;
    }

    // Returns the count of links inputting from the specified neuron ID
    int Genome::LinksInputtingFrom(int a_ID) const {
        int t_counter = 0;
        for (unsigned int i = 0; i < NumLinks(); i++) {
            if (m_LinkGenes[i].FromNeuronID() == a_ID) t_counter++;
        }

        return t_counter;
    }

    // Returns the count of links outputting to the specified neuron ID
    int Genome::LinksOutputtingTo(int a_ID) const {
        int t_counter = 0;
        for (unsigned int i = 0; i < NumLinks(); i++) {
            if (m_LinkGenes[i].ToNeuronID() == a_ID) t_counter++;
        }

        return t_counter;
    }

    // Replaces a hidden neuron having only one input and only one output with a direct link between them.
    bool Genome::Mutate_RemoveSimpleNeuron(InnovationDatabase &a_Innovs, const Parameters &a_Parameters, RNG &a_RNG) {
        // At least one hidden node must be present
        if (NumNeurons() == (NumInputs() + NumOutputs())) return false;

        // Build a list of candidate neurons for deletion
        // Indexes!
        std::vector<int> t_neurons_to_delete;
        for (int i = 0; i < NumNeurons(); i++) {
            if ((LinksInputtingFrom(m_NeuronGenes[i].ID()) == 1) && (LinksOutputtingTo(m_NeuronGenes[i].ID()) == 1) && (m_NeuronGenes[i].Type() == HIDDEN)) {
                t_neurons_to_delete.emplace_back(i);
            }
        }

        // If the list is empty, say goodbye
        if (t_neurons_to_delete.size() == 0) return false;

        // Now choose a random one to delete (uniform over candidates)
        int t_choice = a_RNG.RandInt(0, static_cast<int>(t_neurons_to_delete.size() - 1));

        // the links in & out
        int t_l1idx = -1, t_l2idx = -1;

        // find the link outputting to the neuron
        for (unsigned int i = 0; i < NumLinks(); i++) {
            if (m_LinkGenes[i].ToNeuronID() == m_NeuronGenes[t_neurons_to_delete[t_choice]].ID()) {
                t_l1idx = i;
                break;
            }
        }
        // find the link inputting from the neuron
        for (unsigned int i = 0; i < NumLinks(); i++) {
            if (m_LinkGenes[i].FromNeuronID() == m_NeuronGenes[t_neurons_to_delete[t_choice]].ID()) {
                t_l2idx = i;
                break;
            }
        }

        ASSERT((t_l1idx >= 0) && (t_l2idx >= 0));
        if (t_l1idx < 0 || t_l2idx < 0) return false;

        const LinkGene t_incoming = m_LinkGenes[t_l1idx];
        const LinkGene t_outgoing = m_LinkGenes[t_l2idx];
        // Combines the two removed synapses' spiking state into a replacement link.
        const auto t_apply_combined_spiking = [&](LinkGene &link) {
            link.m_SynapticDelay = t_incoming.m_SynapticDelay + t_outgoing.m_SynapticDelay;
            Clamp(link.m_SynapticDelay, a_Parameters.MinSynapticDelay, a_Parameters.MaxSynapticDelay);
            link.m_SynapticTimeConstant = (t_incoming.m_SynapticTimeConstant + t_outgoing.m_SynapticTimeConstant) * 0.5;
            link.m_STDPEnabled = t_incoming.m_STDPEnabled || t_outgoing.m_STDPEnabled;
            link.m_STDPPlus = (t_incoming.m_STDPPlus + t_outgoing.m_STDPPlus) * 0.5;
            link.m_STDPMinus = (t_incoming.m_STDPMinus + t_outgoing.m_STDPMinus) * 0.5;
            link.m_STDPTauPlus = (t_incoming.m_STDPTauPlus + t_outgoing.m_STDPTauPlus) * 0.5;
            link.m_STDPTauMinus = (t_incoming.m_STDPTauMinus + t_outgoing.m_STDPTauMinus) * 0.5;
            link.m_STDPMinWeight = a_Parameters.MinWeight;
            link.m_STDPMaxWeight = a_Parameters.MaxWeight;
        };

        // OK now see if a link connecting the original 2 nodes is present. If it is, we will just delete the neuron and quit.
        if (HasLink(m_LinkGenes[t_l1idx].FromNeuronID(), m_LinkGenes[t_l2idx].ToNeuronID())) {
            RemoveNeuronGene(m_NeuronGenes[t_neurons_to_delete[t_choice]].ID());
            return true;
        }
        // Else the link is not present and we will replace the neuron and 2 links with one link
        else {
            // Remember the first link's weight
            Real t_weight = m_LinkGenes[t_l1idx].GetWeight();

            // See the innovation database for an innovation number
            int t_innovid = a_Innovs.CheckInnovation(m_LinkGenes[t_l1idx].FromNeuronID(), m_LinkGenes[t_l2idx].ToNeuronID(), NEW_LINK);

            // a novel innovation?
            if (t_innovid == -1) {
                // Save the IDs for a while
                int from = m_LinkGenes[t_l1idx].FromNeuronID();
                int to = m_LinkGenes[t_l2idx].ToNeuronID();

                // Remove the neuron and its links now
                RemoveNeuronGene(m_NeuronGenes[t_neurons_to_delete[t_choice]].ID());

                // Add the innovation and the link gene
                int t_newinnov = a_Innovs.AddLinkInnovation(from, to);
                LinkGene lg = LinkGene(from, to, t_newinnov, t_weight, false);
                t_apply_combined_spiking(lg);
                lg.InitTraits(a_Parameters.LinkTraits, a_RNG);

                m_LinkGenes.emplace_back(lg);

                // bye
                return true;
            }
            // not a novel innovation
            else {
                // Save the IDs for a while
                int from = m_LinkGenes[t_l1idx].FromNeuronID();
                int to = m_LinkGenes[t_l2idx].ToNeuronID();

                // Remove the neuron and its links now
                RemoveNeuronGene(m_NeuronGenes[t_neurons_to_delete[t_choice]].ID());

                // Add the link
                LinkGene lg = LinkGene(from, to, t_innovid, t_weight, false);
                t_apply_combined_spiking(lg);
                lg.InitTraits(a_Parameters.LinkTraits, a_RNG);
                m_LinkGenes.emplace_back(lg);

                // TODO: Maybe inherit the traits from one of the links

                // bye
                return true;
            }
        }

        return false;
    }

    // Perturbs the weights
    bool Genome::Mutate_LinkWeights(const Parameters &a_Parameters, RNG &a_RNG) {
        bool did_mutate = false;
        bool severe = (a_RNG.RandFloat() < a_Parameters.MutateWeightsSevereProb);
        // The end part of the genome (newer genes get replaced more often)
        int tailstart = 0;
        if (NumLinks() > static_cast<unsigned int>(std::max(0, m_initial_num_links))) tailstart = static_cast<int>(NumLinks() * 0.9);
        if (tailstart <= m_initial_num_links) tailstart = m_initial_num_links;
        for (size_t i = 0, end = m_LinkGenes.size(); i < end; ++i) {
            if (!severe && (a_RNG.RandFloat() < a_Parameters.WeightMutationRate)) {
                const Real original = m_LinkGenes[i].GetWeight();
                Real w = original;
                bool in_tail = (static_cast<int>(i) >= tailstart);
                if (in_tail || a_RNG.RandFloat() < a_Parameters.WeightReplacementRate)
                    w = a_RNG.RandFloatSigned() * a_Parameters.WeightReplacementMaxPower;
                else {
                    switch (a_Parameters.WeightMutationDistribution) {
                        case UNIFORM_MUTATION:
                            w += a_RNG.RandFloatSigned() * a_Parameters.WeightMutationMaxPower;
                            break;

                        case GAUSSIAN_MUTATION:
                            if (a_Parameters.WeightMutationMaxPower > 0.0) {
                                w += a_RNG.RandNormal(0.0, a_Parameters.WeightMutationSigma * a_Parameters.WeightMutationMaxPower);
                            }
                            break;

                        case CAUCHY_MUTATION:
                            if (a_Parameters.WeightMutationMaxPower > 0.0) {
                                w += a_RNG.RandCauchy(0.0, a_Parameters.WeightMutationCauchyScale * a_Parameters.WeightMutationMaxPower);
                            }
                            break;

                        case POLYNOMIAL_MUTATION: {
                            const Real range = a_Parameters.MaxWeight - a_Parameters.MinWeight;
                            if (range <= 0.0 || a_Parameters.WeightMutationMaxPower <= 0.0) break;
                            Clamp(w, a_Parameters.MinWeight, a_Parameters.MaxWeight);
                            const Real delta_lower = (w - a_Parameters.MinWeight) / range;
                            const Real delta_upper = (a_Parameters.MaxWeight - w) / range;
                            const Real draw = a_RNG.RandFloat();
                            const Real exponent = 1.0 / (a_Parameters.WeightMutationPolynomialEta + 1.0);
                            Real delta = 0.0;
                            if (draw <= 0.5) {
                                const Real value =
                                    2.0 * draw + (1.0 - 2.0 * draw) * std::pow(1.0 - delta_lower, a_Parameters.WeightMutationPolynomialEta + 1.0);
                                delta = std::pow(value, exponent) - 1.0;
                            } else {
                                const Real value =
                                    2.0 * (1.0 - draw) + 2.0 * (draw - 0.5) * std::pow(1.0 - delta_upper, a_Parameters.WeightMutationPolynomialEta + 1.0);
                                delta = 1.0 - std::pow(value, exponent);
                            }
                            w += delta * std::min(range, a_Parameters.WeightMutationMaxPower);
                            break;
                        }

                        default:
                            throw std::invalid_argument("Unsupported weight mutation distribution");
                    }
                }
                Clamp(w, a_Parameters.MinWeight, a_Parameters.MaxWeight);
                if (w != original) {
                    m_LinkGenes[i].SetWeight(w);
                    did_mutate = true;
                }
            } else if (severe) {
                if (a_RNG.RandFloat() < a_Parameters.WeightMutationRate) {
                    const Real original = m_LinkGenes[i].GetWeight();
                    Real w = a_RNG.RandFloat();
                    Scale(w, 0.0, 1.0, a_Parameters.MinWeight, a_Parameters.MaxWeight);
                    if (w != original) {
                        m_LinkGenes[i].SetWeight(w);
                        did_mutate = true;
                    }
                }
            }
        }
        return did_mutate;
    }

    // Set all link weights to random values between [-R .. R]
    void Genome::Randomize_LinkWeights(const Parameters &a_Parameters, RNG &a_RNG) {
        // For all links..
        for (unsigned int i = 0; i < NumLinks(); i++) {
            Real nf = 0;
            nf = a_RNG.RandFloat();
            Scale(nf, 0.0, 1.0, a_Parameters.MinWeight, a_Parameters.MaxWeight);
            m_LinkGenes[i].SetWeight(nf);
        }
    }

    // Randomize traits
    void Genome::Randomize_Traits(const Parameters &a_Parameters, RNG &a_RNG) {
        for (auto &m_NeuronGene : m_NeuronGenes) {
            m_NeuronGene.InitTraits(a_Parameters.NeuronTraits, a_RNG);
        }
        for (auto &m_LinkGene : m_LinkGenes) {
            m_LinkGene.InitTraits(a_Parameters.LinkTraits, a_RNG);
        }

        m_GenomeGene.InitTraits(a_Parameters.GenomeTraits, a_RNG);
    }

    // Perturbs the A parameters of the neuron activation functions
    bool Genome::Mutate_NeuronActivations_A(const Parameters &a_Parameters, RNG &a_RNG) {
        // for all neurons..
        for (unsigned int i = 0; i < NumNeurons(); i++) {
            // skip inputs and bias
            if ((m_NeuronGenes[i].Type() != INPUT) && (m_NeuronGenes[i].Type() != BIAS)) {
                Real t_randnum = a_RNG.RandFloatSigned() * a_Parameters.ActivationAMutationMaxPower;

                m_NeuronGenes[i].m_A += t_randnum;

                Clamp(m_NeuronGenes[i].m_A, a_Parameters.MinActivationA, a_Parameters.MaxActivationA);
            }
        }

        return true;
    }

    // Perturbs the B parameters of the neuron activation functions
    bool Genome::Mutate_NeuronActivations_B(const Parameters &a_Parameters, RNG &a_RNG) {
        // for all neurons..
        for (unsigned int i = 0; i < NumNeurons(); i++) {
            // skip inputs and bias
            if ((m_NeuronGenes[i].Type() != INPUT) && (m_NeuronGenes[i].Type() != BIAS)) {
                Real t_randnum = a_RNG.RandFloatSigned() * a_Parameters.ActivationBMutationMaxPower;

                m_NeuronGenes[i].m_B += t_randnum;

                Clamp(m_NeuronGenes[i].m_B, a_Parameters.MinActivationB, a_Parameters.MaxActivationB);
            }
        }

        return true;
    }

    // Changes the activation function type for a random neuron
    bool Genome::Mutate_NeuronActivation_Type(const Parameters &a_Parameters, RNG &a_RNG) {
        if (m_NeuronGenes.size() <= static_cast<std::size_t>(m_NumInputs)) return false;
        // the first non-input neuron
        int t_first_idx = NumInputs();
        int t_choice = a_RNG.RandInt(t_first_idx, static_cast<int>(m_NeuronGenes.size()) - 1);

        int cur = m_NeuronGenes[t_choice].m_ActFunction;

        m_NeuronGenes[t_choice].m_ActFunction = GetRandomActivation(a_Parameters, a_RNG);
        if (m_NeuronGenes[t_choice].m_ActFunction == cur)  // same as before?
        {
            return false;
        } else {
            // A new spiking mode needs valid spiking state from the evolvable ranges.
            InitializeNeuronSpiking(m_NeuronGenes[t_choice], a_Parameters, &a_RNG);
            return true;
        }
    }

    // Perturbs the neuron time constants
    bool Genome::Mutate_NeuronTimeConstants(const Parameters &a_Parameters, RNG &a_RNG) {
        // for all neurons..
        for (unsigned int i = 0; i < NumNeurons(); i++) {
            // skip inputs and bias
            if ((m_NeuronGenes[i].Type() != INPUT) && (m_NeuronGenes[i].Type() != BIAS)) {
                Real t_randnum = a_RNG.RandFloatSigned() * a_Parameters.TimeConstantMutationMaxPower;

                m_NeuronGenes[i].m_TimeConstant += t_randnum;

                if (IsSpikingActivation(m_NeuronGenes[i].m_ActFunction))
                    Clamp(m_NeuronGenes[i].m_TimeConstant, a_Parameters.MinSpikingTimeConstant, a_Parameters.MaxSpikingTimeConstant);
                else
                    Clamp(m_NeuronGenes[i].m_TimeConstant, a_Parameters.MinNeuronTimeConstant, a_Parameters.MaxNeuronTimeConstant);
            }
        }

        return true;
    }

    // Perturbs the neuron biases
    bool Genome::Mutate_NeuronBiases(const Parameters &a_Parameters, RNG &a_RNG) {
        // for all neurons..
        for (unsigned int i = 0; i < NumNeurons(); i++) {
            // skip inputs and bias
            if ((m_NeuronGenes[i].Type() != INPUT) && (m_NeuronGenes[i].Type() != BIAS)) {
                Real t_randnum = a_RNG.RandFloatSigned() * a_Parameters.BiasMutationMaxPower;

                m_NeuronGenes[i].m_Bias += t_randnum;

                Clamp(m_NeuronGenes[i].m_Bias, a_Parameters.MinNeuronBias, a_Parameters.MaxNeuronBias);
            }
        }

        return true;
    }

    bool Genome::Mutate_NeuronTraits(const Parameters &a_Parameters, RNG &a_RNG) {
        bool did_mutate = false;
        for (auto it = m_NeuronGenes.begin(); it != m_NeuronGenes.end(); it++) {
            // don't mutate inputs and bias
            if ((it->Type() != INPUT) && (it->Type() != BIAS)) {
                did_mutate |= it->MutateTraits(a_Parameters.NeuronTraits, a_RNG);
            }
        }
        return did_mutate;
    }

    bool Genome::Mutate_LinkTraits(const Parameters &a_Parameters, RNG &a_RNG) {
        bool did_mutate = false;
        for (auto it = m_LinkGenes.begin(); it != m_LinkGenes.end(); it++) {
            did_mutate |= it->MutateTraits(a_Parameters.LinkTraits, a_RNG);
        }
        return did_mutate;
    }

    bool Genome::Mutate_GenomeTraits(const Parameters &a_Parameters, RNG &a_RNG) { return m_GenomeGene.MutateTraits(a_Parameters.GenomeTraits, a_RNG); }

    void Genome::Randomize_SpikingParameters(const Parameters &a_Parameters, RNG &a_RNG) {
        for (auto &neuron : m_NeuronGenes) {
            if (neuron.Type() != INPUT && neuron.Type() != BIAS) InitializeNeuronSpiking(neuron, a_Parameters, &a_RNG);
        }
        for (auto &link : m_LinkGenes) InitializeLinkSpiking(link, a_Parameters, &a_RNG);
    }

    bool Genome::Mutate_NeuronSpikingParameters(const Parameters &a_Parameters, RNG &a_RNG) {
        bool mutated = false;
        const auto perturb = [&](Real &value, Real minimum, Real maximum) {
            if (a_RNG.RandFloat() >= a_Parameters.SpikingParameterMutationRate) return;
            const Real original = value;
            const Real span = maximum - minimum;
            value += a_RNG.RandFloatSigned() * span * a_Parameters.SpikingParameterMutationPower;
            Clamp(value, minimum, maximum);
            if (value == original && minimum < maximum) value = RandomRange(a_RNG, minimum, maximum);
            mutated = mutated || value != original;
        };
        for (auto &neuron : m_NeuronGenes) {
            if (neuron.Type() == INPUT || neuron.Type() == BIAS) continue;
            if (IsSpikingActivation(neuron.m_ActFunction))
                perturb(neuron.m_TimeConstant, a_Parameters.MinSpikingTimeConstant, a_Parameters.MaxSpikingTimeConstant);
            if (neuron.m_ActFunction == SPIKING_IZHIKEVICH)
                perturb(neuron.m_SpikeThreshold, a_Parameters.MinIzhikevichThreshold, a_Parameters.MaxIzhikevichThreshold);
            else
                perturb(neuron.m_SpikeThreshold, a_Parameters.MinSpikeThreshold, a_Parameters.MaxSpikeThreshold);
            perturb(neuron.m_ResetPotential, a_Parameters.MinResetPotential, a_Parameters.MaxResetPotential);
            perturb(neuron.m_RestingPotential, a_Parameters.MinRestingPotential, a_Parameters.MaxRestingPotential);
            perturb(neuron.m_RefractoryPeriod, a_Parameters.MinRefractoryPeriod, a_Parameters.MaxRefractoryPeriod);
            perturb(neuron.m_MembraneResistance, a_Parameters.MinMembraneResistance, a_Parameters.MaxMembraneResistance);
            perturb(neuron.m_AdaptationTimeConstant, a_Parameters.MinAdaptationTimeConstant, a_Parameters.MaxAdaptationTimeConstant);
            perturb(neuron.m_AdaptationIncrement, a_Parameters.MinAdaptationIncrement, a_Parameters.MaxAdaptationIncrement);
            perturb(neuron.m_RateTimeConstant, a_Parameters.MinSpikeRateTimeConstant, a_Parameters.MaxSpikeRateTimeConstant);
            perturb(neuron.m_IzhikevichA, a_Parameters.MinIzhikevichA, a_Parameters.MaxIzhikevichA);
            perturb(neuron.m_IzhikevichB, a_Parameters.MinIzhikevichB, a_Parameters.MaxIzhikevichB);
            perturb(neuron.m_IzhikevichC, a_Parameters.MinIzhikevichC, a_Parameters.MaxIzhikevichC);
            perturb(neuron.m_IzhikevichD, a_Parameters.MinIzhikevichD, a_Parameters.MaxIzhikevichD);
            if (neuron.m_ActFunction == MCCULLOCH_PITTS && a_RNG.RandFloat() < a_Parameters.MutateMCPInhibitoryVetoProb) {
                neuron.m_MCPInhibitoryVeto = !neuron.m_MCPInhibitoryVeto;
                mutated = true;
            }
        }
        return mutated;
    }

    bool Genome::Mutate_LinkSpikingParameters(const Parameters &a_Parameters, RNG &a_RNG) {
        bool mutated = false;
        const auto perturb = [&](Real &value, Real minimum, Real maximum) {
            if (a_RNG.RandFloat() >= a_Parameters.SpikingParameterMutationRate) return;
            const Real original = value;
            value += a_RNG.RandFloatSigned() * (maximum - minimum) * a_Parameters.SpikingParameterMutationPower;
            Clamp(value, minimum, maximum);
            if (value == original && minimum < maximum) value = RandomRange(a_RNG, minimum, maximum);
            mutated = mutated || value != original;
        };
        for (auto &link : m_LinkGenes) {
            perturb(link.m_SynapticDelay, a_Parameters.MinSynapticDelay, a_Parameters.MaxSynapticDelay);
            perturb(link.m_SynapticTimeConstant, a_Parameters.MinSynapticTimeConstant, a_Parameters.MaxSynapticTimeConstant);
            perturb(link.m_STDPPlus, a_Parameters.MinSTDPPlus, a_Parameters.MaxSTDPPlus);
            perturb(link.m_STDPMinus, a_Parameters.MinSTDPMinus, a_Parameters.MaxSTDPMinus);
            perturb(link.m_STDPTauPlus, a_Parameters.MinSTDPTau, a_Parameters.MaxSTDPTau);
            perturb(link.m_STDPTauMinus, a_Parameters.MinSTDPTau, a_Parameters.MaxSTDPTau);
            if (a_RNG.RandFloat() < a_Parameters.SpikingParameterMutationRate) {
                link.m_STDPEnabled = !link.m_STDPEnabled;
                mutated = true;
            }
            link.m_STDPMinWeight = a_Parameters.MinWeight;
            link.m_STDPMaxWeight = a_Parameters.MaxWeight;
        }
        return mutated;
    }

    // Mate this genome with dad and return the baby
    // This is multipoint mating - genes inherited randomly
    // Disjoint and excess genes are inherited from the fittest parent
    // If fitness is equal, the smaller genome is assumed to be the better one
    Genome Genome::Mate(Genome &a_Dad, bool a_MateAverage, bool a_InterSpecies, RNG &a_RNG, Parameters &a_Parameters) {
        return MateWithMode(a_Dad, a_MateAverage ? AVERAGE : MULTIPOINT, a_InterSpecies, a_RNG, a_Parameters);
    }

    Genome Genome::MateWithMode(Genome &a_Dad, CrossoverMode a_Mode, bool a_InterSpecies, RNG &a_RNG, Parameters &a_Parameters) {
        if (a_Mode < MULTIPOINT || a_Mode > SIMULATED_BINARY) throw std::invalid_argument("Unsupported crossover mode");
        if (m_NumInputs != a_Dad.m_NumInputs || m_NumOutputs != a_Dad.m_NumOutputs)
            throw std::invalid_argument("Cannot mate genomes with different input/output dimensions");
        // Cannot mate with itself
        if (GetID() == a_Dad.GetID()) return *this;

        // helps make the code clearer
        enum t_parent_type {
            MOM,
            DAD,
        };

        // This is the fittest genome.
        t_parent_type t_better;

        // This empty genome will hold the baby
        Genome t_baby;

        // if they are of equal fitness use the shorter (because we want to keep the networks as small as possible)
        if (GetFitness() == a_Dad.GetFitness()) {
            // if they are of equal fitness and length just choose one at random
            if (NumLinks() == a_Dad.NumLinks()) {
                t_better = (a_RNG.RandFloat() < 0.5) ? MOM : DAD;
            } else {
                t_better = (NumLinks() < a_Dad.NumLinks()) ? MOM : DAD;
            }
        } else {
            t_better = (GetFitness() > a_Dad.GetFitness()) ? MOM : DAD;
        }

        // Sort copies so the merge below classifies disjoint/excess correctly
        // even if a parent's genes are unsorted.
        std::vector<LinkGene> mom_links = m_LinkGenes;
        std::vector<LinkGene> dad_links = a_Dad.m_LinkGenes;
        std::sort(mom_links.begin(), mom_links.end());
        std::sort(dad_links.begin(), dad_links.end());
        auto t_curMom = mom_links.begin();
        auto t_curDad = dad_links.begin();

        const bool average_traits = a_Mode == AVERAGE || a_Mode == BLEND || a_Mode == SIMULATED_BINARY;
        const bool prefer_fitter_neurons = a_Mode == MULTIPOINT || a_Mode == SINGLE_POINT;

        // Mate the GenomeGene first.
        if (!average_traits) {
            Gene n;
            if (a_RNG.RandFloat() < a_Parameters.PreferFitterParentRate)
                n = (t_better == MOM) ? m_GenomeGene : a_Dad.m_GenomeGene;
            else
                n = (a_RNG.RandFloat() < 0.5) ? m_GenomeGene : a_Dad.m_GenomeGene;
            t_baby.m_GenomeGene = n;
        } else {
            Gene n = m_GenomeGene;
            n.MateTraits(a_Dad.m_GenomeGene.m_Traits, a_RNG);
            t_baby.m_GenomeGene = n;
        }

        // I/O neurons come first by index in both parents; they must agree.
        t_baby.m_NeuronGenes.reserve(static_cast<std::size_t>(m_NumInputs + m_NumOutputs));
        for (int index = 0; index < m_NumInputs + m_NumOutputs; ++index) {
            const NeuronGene mom = GetNeuronByIndex(index);
            const NeuronGene dad = a_Dad.GetNeuronByIndex(index);
            if (mom.ID() != dad.ID() || mom.Type() != dad.Type()) throw std::invalid_argument("Cannot mate genomes with incompatible input/output neurons");
            NeuronGene child = mom;
            if (average_traits) {
                child.MateTraits(dad.m_Traits, a_RNG);
            } else if (a_RNG.RandFloat() < a_Parameters.PreferFitterParentRate) {
                child = (t_better == MOM) ? mom : dad;
            } else {
                child = (a_RNG.RandFloat() < 0.5) ? mom : dad;
            }
            t_baby.m_NeuronGenes.push_back(child);
        }

        // SINGLE_POINT setup: count matching links, then draw the cut point and side.
        std::size_t matching_link_count = 0;
        if (a_Mode == SINGLE_POINT) {
            auto mom = mom_links.begin();
            auto dad = dad_links.begin();
            while (mom != mom_links.end() && dad != dad_links.end()) {
                if (mom->InnovationID() == dad->InnovationID()) {
                    ++matching_link_count;
                    ++mom;
                    ++dad;
                } else if (mom->InnovationID() < dad->InnovationID()) {
                    ++mom;
                } else {
                    ++dad;
                }
            }
        }
        const std::size_t single_point =
            (a_Mode != SINGLE_POINT || matching_link_count == 0) ? 0 : static_cast<std::size_t>(a_RNG.RandInt(0, static_cast<int>(matching_link_count)));
        const bool mom_before_single_point = (a_Mode != SINGLE_POINT || a_RNG.RandFloat() < 0.5);
        std::size_t matching_link_index = 0;

        // Endpoint lookup tables and baby dedup sets.
        std::unordered_map<int, const NeuronGene *> mom_neurons;
        std::unordered_map<int, const NeuronGene *> dad_neurons;
        mom_neurons.reserve(m_NeuronGenes.size());
        dad_neurons.reserve(a_Dad.m_NeuronGenes.size());
        for (const auto &neuron : m_NeuronGenes) mom_neurons.emplace(neuron.ID(), &neuron);
        for (const auto &neuron : a_Dad.m_NeuronGenes) dad_neurons.emplace(neuron.ID(), &neuron);

        std::unordered_set<int> child_neuron_ids;
        child_neuron_ids.reserve(m_NeuronGenes.size() + a_Dad.m_NeuronGenes.size());
        for (const auto &neuron : t_baby.m_NeuronGenes) child_neuron_ids.insert(neuron.ID());
        std::unordered_set<std::uint64_t> child_endpoints;
        child_endpoints.reserve(m_LinkGenes.size() + a_Dad.m_LinkGenes.size());
        const auto endpoint_key = [](int source, int target) {
            return (static_cast<std::uint64_t>(static_cast<std::uint32_t>(source)) << 32U) | static_cast<std::uint32_t>(target);
        };
        const auto add_child_neuron = [&](int neuron_id) {
            if (child_neuron_ids.count(neuron_id) != 0) return;
            const auto mom = mom_neurons.find(neuron_id);
            const auto dad = dad_neurons.find(neuron_id);
            const NeuronGene *selected = nullptr;
            if (mom != mom_neurons.end() && dad != dad_neurons.end()) {
                if (prefer_fitter_neurons) {
                    selected = (t_better == MOM) ? mom->second : dad->second;
                } else {
                    selected = (a_RNG.RandFloat() < 0.5) ? mom->second : dad->second;
                }
            } else if (mom != mom_neurons.end()) {
                selected = mom->second;
            } else if (dad != dad_neurons.end()) {
                selected = dad->second;
            }
            if (selected == nullptr) throw std::logic_error("Crossover selected a link with a missing endpoint");
            t_baby.m_NeuronGenes.push_back(*selected);
            child_neuron_ids.insert(neuron_id);
        };

        // this will hold a copy of the gene we wish to add at each step
        LinkGene t_emptygene(0, 0, -1, 0, false);

        // step through each parents link genes until we reach the end of both
        while (!((t_curMom == mom_links.end()) && (t_curDad == dad_links.end()))) {
            LinkGene t_selectedgene = t_emptygene;
            bool t_skip = false;

            // the end of mum's genes have been reached EXCESS
            if (t_curMom == mom_links.end()) {
                // select dads gene
                t_selectedgene = *t_curDad;
                // move onto dad's next gene
                t_curDad++;

                // if mom is fittest, abort adding
                if (t_better == MOM) {
                    t_skip = true;
                }
            }

            // the end of dads's genes have been reached EXCESS
            else if (t_curDad == dad_links.end()) {
                // add mums gene
                t_selectedgene = *t_curMom;
                // move onto mum's next gene
                t_curMom++;

                // if dad is fittest, abort adding
                if (t_better == DAD) {
                    t_skip = true;
                }
            } else {
                // extract the innovation numbers
                const int t_innov_mom = t_curMom->InnovationID();
                const int t_innov_dad = t_curDad->InnovationID();

                // if both innovations match
                if (t_innov_mom == t_innov_dad) {
                    switch (a_Mode) {
                        case MULTIPOINT:
                            if (a_RNG.RandFloat() < a_Parameters.PreferFitterParentRate)
                                t_selectedgene = (t_better == MOM) ? *t_curMom : *t_curDad;
                            else
                                t_selectedgene = (a_RNG.RandFloat() < 0.5) ? *t_curMom : *t_curDad;
                            break;
                        case AVERAGE:
                            t_selectedgene = *t_curMom;
                            t_selectedgene.SetWeight((t_curDad->GetWeight() + t_curMom->GetWeight()) / 2.0);
                            t_selectedgene.MateTraits(t_curDad->m_Traits, a_RNG);
                            break;
                        case SINGLE_POINT: {
                            const bool take_mom = (matching_link_index < single_point) == mom_before_single_point;
                            t_selectedgene = take_mom ? *t_curMom : *t_curDad;
                            break;
                        }
                        case BLEND: {
                            t_selectedgene = *t_curMom;
                            const Real mom_weight = t_curMom->GetWeight();
                            const Real dad_weight = t_curDad->GetWeight();
                            const Real minimum = std::min(mom_weight, dad_weight);
                            const Real maximum = std::max(mom_weight, dad_weight);
                            const Real span = maximum - minimum;
                            Real weight =
                                minimum - a_Parameters.CrossoverBlendAlpha * span + a_RNG.RandFloat() * (span * (1.0 + 2.0 * a_Parameters.CrossoverBlendAlpha));
                            Clamp(weight, a_Parameters.MinWeight, a_Parameters.MaxWeight);
                            t_selectedgene.SetWeight(weight);
                            t_selectedgene.MateTraits(t_curDad->m_Traits, a_RNG);
                            break;
                        }
                        case SIMULATED_BINARY: {
                            t_selectedgene = *t_curMom;
                            const Real draw = a_RNG.RandFloat();
                            const Real exponent = 1.0 / (a_Parameters.CrossoverSBXEta + 1.0);
                            const Real beta = draw <= 0.5 ? std::pow(2.0 * draw, exponent) : std::pow(1.0 / (2.0 * (1.0 - draw)), exponent);
                            const Real first = t_curMom->GetWeight();
                            const Real second = t_curDad->GetWeight();
                            Real weight = a_RNG.RandFloat() < 0.5 ? 0.5 * ((1.0 + beta) * first + (1.0 - beta) * second)
                                                                  : 0.5 * ((1.0 - beta) * first + (1.0 + beta) * second);
                            Clamp(weight, a_Parameters.MinWeight, a_Parameters.MaxWeight);
                            t_selectedgene.SetWeight(weight);
                            t_selectedgene.MateTraits(t_curDad->m_Traits, a_RNG);
                            break;
                        }
                    }

                    ++matching_link_index;
                    // move onto next gene of each parent
                    t_curMom++;
                    t_curDad++;
                } else  // DISJOINT
                    if (t_innov_mom < t_innov_dad) {
                        t_selectedgene = *t_curMom;
                        t_curMom++;

                        if (t_better == DAD) {
                            t_skip = true;
                        }
                    } else  // DISJOINT
                        if (t_innov_dad < t_innov_mom) {
                            t_selectedgene = *t_curDad;
                            t_curDad++;

                            if (t_better == MOM) {
                                t_skip = true;
                            }
                        }
            }

            // for interspecies mating, allow all genes through
            if (a_InterSpecies) {
                t_skip = false;
            }

            // If the selected gene's innovation number is negative,
            // this means that no gene is selected (should be skipped).
            // The endpoint set also guards against duplicate links.
            if ((t_selectedgene.InnovationID() > 0) && (!t_skip)) {
                const std::uint64_t key = endpoint_key(t_selectedgene.FromNeuronID(), t_selectedgene.ToNeuronID());
                if (child_endpoints.insert(key).second) {
                    t_baby.m_LinkGenes.push_back(t_selectedgene);
                    // Check if we already have the nodes referred to in t_selectedgene. If not, they need to be added.
                    add_child_neuron(t_selectedgene.FromNeuronID());
                    add_child_neuron(t_selectedgene.ToNeuronID());
                }
            }
        }  // end while

        t_baby.m_NumInputs = m_NumInputs;
        t_baby.m_NumOutputs = m_NumOutputs;
        t_baby.m_initial_num_neurons = m_initial_num_neurons;
        t_baby.m_initial_num_links = m_initial_num_links;

        // Sort the baby's genes
        t_baby.SortGenes();

        return t_baby;
    }

    // Sorts the genes of the genome The neurons by IDs and the links by innovation numbers.
    bool neuron_compare(NeuronGene &a_ls, NeuronGene &a_rs) { return a_ls.ID() < a_rs.ID(); }

    bool link_compare(LinkGene &a_ls, LinkGene &a_rs) { return a_ls.InnovationID() < a_rs.InnovationID(); }

    void Genome::SortGenes() {
        std::sort(m_NeuronGenes.begin(), m_NeuronGenes.end(), neuron_compare);
        std::sort(m_LinkGenes.begin(), m_LinkGenes.end(), link_compare);
    }

    unsigned int Genome::NeuronDepth(int a_NeuronID, unsigned int a_Depth) {
        unsigned int t_current_depth;
        unsigned int t_max_depth = a_Depth;

        if (a_Depth > 16384) {
            // oops! a possible loop in the network!
            // DBG(" ERROR! Trying to get the depth of a looped network!");
            return 16384;
        }

        // Base case
        if ((GetNeuronByID(a_NeuronID).Type() == INPUT) || (GetNeuronByID(a_NeuronID).Type() == BIAS)) {
            return a_Depth;
        }

        // Find all links outputting to this neuron ID
        std::vector<int> t_inputting_links_idx;
        for (unsigned int i = 0; i < NumLinks(); i++) {
            if (m_LinkGenes[i].ToNeuronID() == a_NeuronID) t_inputting_links_idx.emplace_back(i);
        }

        // For all incoming links..
        for (unsigned int i = 0; i < t_inputting_links_idx.size(); i++) {
            LinkGene t_link = GetLinkByIndex(t_inputting_links_idx[i]);

            // RECURSION
            t_current_depth = NeuronDepth(t_link.FromNeuronID(), a_Depth + 1);
            if (t_current_depth > t_max_depth) t_max_depth = t_current_depth;
        }

        return t_max_depth;
    }

    void Genome::CalculateDepth() {
        if (m_NeuronGenes.empty()) {
            m_Depth = 0;
            return;
        }
        // Iterative Kahn topological pass over the non-recurrent links:
        // no recursion depth limit, recurrent links skipped, and a cycle made
        // of non-recurrent links throws instead of silently capping at 16384.
        std::map<int, std::size_t> neuron_indices;
        for (std::size_t i = 0; i < m_NeuronGenes.size(); ++i) neuron_indices[m_NeuronGenes[i].ID()] = i;
        std::vector<std::vector<std::size_t>> outgoing(m_NeuronGenes.size());
        std::vector<std::size_t> indegree(m_NeuronGenes.size(), 0);
        for (const auto &link : m_LinkGenes) {
            if (link.IsRecurrent()) continue;
            const auto source = neuron_indices.find(link.FromNeuronID());
            const auto target = neuron_indices.find(link.ToNeuronID());
            if (source == neuron_indices.end() || target == neuron_indices.end())
                throw std::runtime_error("Genome contains a link whose endpoint neuron does not exist");
            outgoing[source->second].push_back(target->second);
            ++indegree[target->second];
        }
        std::vector<std::size_t> queue;
        queue.reserve(m_NeuronGenes.size());
        for (std::size_t i = 0; i < indegree.size(); ++i) {
            if (indegree[i] == 0) queue.push_back(i);
        }
        std::vector<unsigned int> depth(m_NeuronGenes.size(), 0);
        std::size_t head = 0;
        while (head < queue.size()) {
            const std::size_t source = queue[head++];
            for (std::size_t target : outgoing[source]) {
                depth[target] = std::max(depth[target], depth[source] + 1);
                if (--indegree[target] == 0) queue.push_back(target);
            }
        }
        if (queue.size() != m_NeuronGenes.size()) {
            throw std::runtime_error("Genome contains a cycle made of non-recurrent links");
        }
        unsigned int maximum = 0;
        for (std::size_t i = 0; i < m_NeuronGenes.size(); ++i) {
            if (m_NeuronGenes[i].Type() == OUTPUT) maximum = std::max(maximum, depth[i]);
        }
        m_Depth = static_cast<int>(std::max(1U, maximum));
    }

    //////////////////////////////////////////////////////////////////////////////////
    // Saving/Loading methods
    //////////////////////////////////////////////////////////////////////////////////

    // Builds this genome from a file
    Genome::Genome(const char *a_FileName) {
        std::ifstream t_DataFile(a_FileName);
        *this = Genome(t_DataFile);
        t_DataFile.close();
    }

    // Builds the genome from an *opened* file
    Genome::Genome(std::ifstream &a_DataFile) {
        std::string t_Str;

        if (!a_DataFile) {
            ostringstream tStream;
            tStream << "Genome file error!" << std::endl;
            throw std::runtime_error("Genome file error!");
        }

        // search for GenomeStart (guard against EOF: a stream extraction failure
        // leaves t_Str unchanged, so without the eof check this loop never ends)
        do {
            a_DataFile >> t_Str;
            if (a_DataFile.eof()) {
                throw std::runtime_error("Genome file error: GenomeStart not found!");
            }
        } while (t_Str != "GenomeStart");

        // read the genome ID
        unsigned int t_gid;
        a_DataFile >> t_gid;
        m_ID = t_gid;

        // read the genome until GenomeEnd is encountered
        do {
            a_DataFile >> t_Str;
            if (a_DataFile.eof()) {
                throw std::runtime_error("Genome file error: GenomeEnd not found!");
            }

            if (t_Str == "Neuron") {
                int t_id, t_type, t_activationfunc;
                Real t_splity, t_a, t_b, t_timeconst, t_bias;

                a_DataFile >> t_id;
                a_DataFile >> t_type;
                a_DataFile >> t_splity;

                a_DataFile >> t_activationfunc;
                a_DataFile >> t_a;
                a_DataFile >> t_b;
                a_DataFile >> t_timeconst;
                a_DataFile >> t_bias;

                // TODO read neuron traits

                NeuronGene t_neuron(static_cast<NeuronType>(t_type), t_id, t_splity);
                t_neuron.Init(t_a, t_b, t_timeconst, t_bias, static_cast<ActivationFunction>(t_activationfunc));

                m_NeuronGenes.emplace_back(t_neuron);
            }
            if (t_Str == "NeuronSpiking") {
                if (m_NeuronGenes.empty()) throw std::runtime_error("Genome file error: NeuronSpiking appears before a neuron.");
                NeuronGene &t_neuron = m_NeuronGenes.back();
                a_DataFile >> t_neuron.m_SpikeThreshold >> t_neuron.m_ResetPotential >> t_neuron.m_RestingPotential >> t_neuron.m_RefractoryPeriod >>
                    t_neuron.m_MembraneResistance >> t_neuron.m_AdaptationTimeConstant >> t_neuron.m_AdaptationIncrement >> t_neuron.m_RateTimeConstant >>
                    t_neuron.m_IzhikevichA >> t_neuron.m_IzhikevichB >> t_neuron.m_IzhikevichC >> t_neuron.m_IzhikevichD;
            }
            if (t_Str == "Link") {
                int t_from, t_to, t_innov, t_isrecur;
                Real t_weight;

                a_DataFile >> t_from;
                a_DataFile >> t_to;
                a_DataFile >> t_innov;
                a_DataFile >> t_isrecur;
                a_DataFile >> t_weight;

                // TODO read link traits

                m_LinkGenes.emplace_back(LinkGene(t_from, t_to, t_innov, t_weight, static_cast<bool>(t_isrecur)));
            }

            if (t_Str == "LinkSpiking") {
                if (m_LinkGenes.empty()) throw std::runtime_error("Genome file error: LinkSpiking appears before a link.");
                LinkGene &t_link = m_LinkGenes.back();
                int t_stdp_enabled = 0;
                a_DataFile >> t_link.m_SynapticDelay >> t_link.m_SynapticTimeConstant >> t_stdp_enabled >> t_link.m_STDPPlus >> t_link.m_STDPMinus >>
                    t_link.m_STDPTauPlus >> t_link.m_STDPTauMinus >> t_link.m_STDPMinWeight >> t_link.m_STDPMaxWeight;
                t_link.m_STDPEnabled = (t_stdp_enabled != 0);
            }
        } while (t_Str != "GenomeEnd");

        // Init additional stuff
        // count inputs/outputs
        m_NumInputs = 0;
        m_NumOutputs = 0;
        for (unsigned int i = 0; i < NumNeurons(); i++) {
            if ((m_NeuronGenes[i].Type() == INPUT) || (m_NeuronGenes[i].Type() == BIAS)) {
                m_NumInputs++;
            }

            if (m_NeuronGenes[i].Type() == OUTPUT) {
                m_NumOutputs++;
            }
        }

        m_Fitness = 0.0;
        m_AdjustedFitness = 0.0;
        m_OffspringAmount = 0.0;
        m_Depth = 0;
        m_PhenotypeBehavior = NULL;
        m_Evaluated = false;
    }

    // Saves this genome to a file
    void Genome::Save(const char *a_FileName) {
        if (a_FileName == nullptr) throw std::invalid_argument("Genome filename is null");
        FILE *t_file = detail::OpenFile(a_FileName, "w");
        if (t_file == nullptr) throw std::runtime_error("Cannot open genome file for writing");
        Save(t_file);
        fclose(t_file);
    }

    // Saves this genome to an already opened file for writing
    void Genome::Save(FILE *a_file) {
        fprintf(a_file, "GenomeStart %d\n", GetID());

        // loop over the neurons and save each one
        for (unsigned int i = 0; i < NumNeurons(); i++) {
            const NeuronGene &ng = m_NeuronGenes[i];
            // Save neuron
            fprintf(a_file, "Neuron %d %d %3.8f %d %3.8f %3.8f %3.8f %3.8f\n", ng.ID(), static_cast<int>(ng.Type()), ng.SplitY(),
                    static_cast<int>(ng.m_ActFunction), ng.m_A, ng.m_B, ng.m_TimeConstant, ng.m_Bias);
            // TODO write neuron traits
            fprintf(a_file,
                    "NeuronSpiking %3.18f %3.18f %3.18f %3.18f %3.18f "
                    "%3.18f %3.18f %3.18f %3.18f %3.18f %3.18f %3.18f\n",
                    ng.m_SpikeThreshold, ng.m_ResetPotential, ng.m_RestingPotential, ng.m_RefractoryPeriod, ng.m_MembraneResistance,
                    ng.m_AdaptationTimeConstant, ng.m_AdaptationIncrement, ng.m_RateTimeConstant, ng.m_IzhikevichA, ng.m_IzhikevichB, ng.m_IzhikevichC,
                    ng.m_IzhikevichD);
        }

        // loop over the connections and save each one
        for (unsigned int i = 0; i < NumLinks(); i++) {
            const LinkGene &lg = m_LinkGenes[i];
            fprintf(a_file, "Link %d %d %d %d %3.8f\n", lg.FromNeuronID(), lg.ToNeuronID(), lg.InnovationID(), static_cast<int>(lg.IsRecurrent()),
                    lg.GetWeight());
            // TODO write link traits
            fprintf(a_file,
                    "LinkSpiking %3.18f %3.18f %d %3.18f %3.18f "
                    "%3.18f %3.18f %3.18f %3.18f\n",
                    lg.m_SynapticDelay, lg.m_SynapticTimeConstant, static_cast<int>(lg.m_STDPEnabled), lg.m_STDPPlus, lg.m_STDPMinus, lg.m_STDPTauPlus,
                    lg.m_STDPTauMinus, lg.m_STDPMinWeight, lg.m_STDPMaxWeight);
        }

        fprintf(a_file, "GenomeEnd\n\n");
    }

    // Builds this genome from any input stream (format versions 1-4 and legacy).
    Genome::Genome(std::istream &data) : Genome() {
        if (!data) throw std::runtime_error("Invalid input stream provided to Genome constructor.");

        std::string token;
        while (data >> token && token != "GenomeStart") {
        }
        if (token != "GenomeStart") throw std::runtime_error("Genome: missing GenomeStart marker.");

        data >> m_ID;
        if (!data) throw std::runtime_error("Genome: missing genome ID.");

        int format_version = 1;
        bool has_state = false;
        bool found_end = false;
        int last_neuron = -1;
        int last_link = -1;
        while (data >> token) {
            if (token == "GenomeEnd") {
                found_end = true;
                break;
            }
            if (token == "GenomeFormat") {
                data >> format_version;
                if (format_version < 1 || format_version > 4) throw std::runtime_error("Genome: unsupported serialization format.");
            } else if (token == "GenomeState") {
                int evaluated = 0;
                data >> m_Fitness >> m_AdjustedFitness >> m_OffspringAmount >> m_Depth >> m_NumInputs >> m_NumOutputs >> evaluated >> m_initial_num_neurons >>
                    m_initial_num_links;
                m_Evaluated = evaluated != 0;
                has_state = true;
            } else if (token == "GenomeTraits") {
                m_GenomeGene.m_Traits = Serialization::ReadTraits(data);
            } else if (token == "Neuron") {
                int id, type, activation;
                Real split_y, a, b, time_constant, bias;
                data >> id >> type >> split_y >> activation >> a >> b >> time_constant >> bias;
                NeuronGene neuron(static_cast<NeuronType>(type), id, split_y);
                neuron.m_ActFunction = static_cast<ActivationFunction>(activation);
                neuron.m_A = a;
                neuron.m_B = b;
                neuron.m_TimeConstant = time_constant;
                neuron.m_Bias = bias;
                if (format_version >= 2) data >> neuron.x >> neuron.y;
                if (format_version >= 3) {
                    data >> neuron.m_SpikeThreshold >> neuron.m_ResetPotential >> neuron.m_RestingPotential >> neuron.m_RefractoryPeriod >>
                        neuron.m_MembraneResistance >> neuron.m_AdaptationTimeConstant >> neuron.m_AdaptationIncrement >> neuron.m_RateTimeConstant >>
                        neuron.m_IzhikevichA >> neuron.m_IzhikevichB >> neuron.m_IzhikevichC >> neuron.m_IzhikevichD;
                }
                if (format_version >= 4) {
                    int inhibitory_veto = 1;
                    data >> inhibitory_veto;
                    neuron.m_MCPInhibitoryVeto = inhibitory_veto != 0;
                }
                m_NeuronGenes.push_back(neuron);
                last_neuron = static_cast<int>(m_NeuronGenes.size()) - 1;
            } else if (token == "NeuronTraits") {
                if (last_neuron < 0) throw std::runtime_error("Genome: NeuronTraits appears before a neuron.");
                m_NeuronGenes[static_cast<std::size_t>(last_neuron)].m_Traits = Serialization::ReadTraits(data);
            } else if (token == "NeuronSpiking") {
                if (last_neuron < 0) throw std::runtime_error("Genome: NeuronSpiking appears before a neuron.");
                NeuronGene &neuron = m_NeuronGenes[static_cast<std::size_t>(last_neuron)];
                data >> neuron.m_SpikeThreshold >> neuron.m_ResetPotential >> neuron.m_RestingPotential >> neuron.m_RefractoryPeriod >>
                    neuron.m_MembraneResistance >> neuron.m_AdaptationTimeConstant >> neuron.m_AdaptationIncrement >> neuron.m_RateTimeConstant >>
                    neuron.m_IzhikevichA >> neuron.m_IzhikevichB >> neuron.m_IzhikevichC >> neuron.m_IzhikevichD;
            } else if (token == "Link") {
                int from, to, innovation, recurrent;
                Real weight;
                data >> from >> to >> innovation >> recurrent >> weight;
                m_LinkGenes.emplace_back(from, to, innovation, weight, recurrent != 0);
                last_link = static_cast<int>(m_LinkGenes.size()) - 1;
                if (format_version >= 3) {
                    LinkGene &link = m_LinkGenes[static_cast<std::size_t>(last_link)];
                    int stdp_enabled = 0;
                    data >> link.m_SynapticDelay >> link.m_SynapticTimeConstant >> stdp_enabled >> link.m_STDPPlus >> link.m_STDPMinus >> link.m_STDPTauPlus >>
                        link.m_STDPTauMinus >> link.m_STDPMinWeight >> link.m_STDPMaxWeight;
                    link.m_STDPEnabled = stdp_enabled != 0;
                }
            } else if (token == "LinkTraits") {
                if (last_link < 0) throw std::runtime_error("Genome: LinkTraits appears before a link.");
                m_LinkGenes[static_cast<std::size_t>(last_link)].m_Traits = Serialization::ReadTraits(data);
            } else if (token == "LinkSpiking") {
                if (last_link < 0) throw std::runtime_error("Genome: LinkSpiking appears before a link.");
                LinkGene &link = m_LinkGenes[static_cast<std::size_t>(last_link)];
                int stdp_enabled = 0;
                data >> link.m_SynapticDelay >> link.m_SynapticTimeConstant >> stdp_enabled >> link.m_STDPPlus >> link.m_STDPMinus >> link.m_STDPTauPlus >>
                    link.m_STDPTauMinus >> link.m_STDPMinWeight >> link.m_STDPMaxWeight;
                link.m_STDPEnabled = stdp_enabled != 0;
            } else {
                std::string ignored;
                std::getline(data, ignored);
            }
            if (!data) throw std::runtime_error("Genome: malformed serialized data.");
        }
        if (!found_end) throw std::runtime_error("Genome: missing GenomeEnd marker.");
        if (!has_state) {
            m_NumInputs = 0;
            m_NumOutputs = 0;
            for (const auto &neuron : m_NeuronGenes) {
                if (neuron.Type() == INPUT || neuron.Type() == BIAS)
                    ++m_NumInputs;
                else if (neuron.Type() == OUTPUT)
                    ++m_NumOutputs;
            }
            m_initial_num_neurons = static_cast<int>(m_NeuronGenes.size());
            m_initial_num_links = static_cast<int>(m_LinkGenes.size());
        }
        m_PhenotypeBehavior = NULL;
        std::string validation_error;
        if (!Validate(&validation_error)) throw std::runtime_error("Genome: invalid serialized data: " + validation_error);
    }

    std::string Genome::Serialize() const {
        std::string validation_error;
        if (!Validate(&validation_error)) throw std::runtime_error("Genome::Serialize: " + validation_error);
        std::ostringstream output;
        Serialization::UseRoundTripPrecision(output);
        output << "GenomeStart " << GetID() << "\n";
        output << "GenomeFormat 4\n";
        output << "GenomeState " << m_Fitness << ' ' << m_AdjustedFitness << ' ' << m_OffspringAmount << ' ' << m_Depth << ' ' << m_NumInputs << ' '
               << m_NumOutputs << ' ' << static_cast<int>(m_Evaluated) << ' ' << m_initial_num_neurons << ' ' << m_initial_num_links << '\n';
        Serialization::WriteTraits(output, "GenomeTraits", m_GenomeGene.m_Traits);
        for (const auto &neuron : m_NeuronGenes) {
            output << "Neuron " << neuron.m_ID << ' ' << static_cast<int>(neuron.m_Type) << ' ' << neuron.m_SplitY << ' '
                   << static_cast<int>(neuron.m_ActFunction) << ' ' << neuron.m_A << ' ' << neuron.m_B << ' ' << neuron.m_TimeConstant << ' ' << neuron.m_Bias
                   << ' ' << neuron.x << ' ' << neuron.y << ' ' << neuron.m_SpikeThreshold << ' ' << neuron.m_ResetPotential << ' ' << neuron.m_RestingPotential
                   << ' ' << neuron.m_RefractoryPeriod << ' ' << neuron.m_MembraneResistance << ' ' << neuron.m_AdaptationTimeConstant << ' '
                   << neuron.m_AdaptationIncrement << ' ' << neuron.m_RateTimeConstant << ' ' << neuron.m_IzhikevichA << ' ' << neuron.m_IzhikevichB << ' '
                   << neuron.m_IzhikevichC << ' ' << neuron.m_IzhikevichD << ' ' << static_cast<int>(neuron.m_MCPInhibitoryVeto) << '\n';
            Serialization::WriteTraits(output, "NeuronTraits", neuron.m_Traits);
        }
        for (const auto &link : m_LinkGenes) {
            output << "Link " << link.m_FromNeuronID << ' ' << link.m_ToNeuronID << ' ' << link.m_InnovationID << ' ' << static_cast<int>(link.m_IsRecurrent)
                   << ' ' << link.m_Weight << ' ' << link.m_SynapticDelay << ' ' << link.m_SynapticTimeConstant << ' ' << static_cast<int>(link.m_STDPEnabled)
                   << ' ' << link.m_STDPPlus << ' ' << link.m_STDPMinus << ' ' << link.m_STDPTauPlus << ' ' << link.m_STDPTauMinus << ' '
                   << link.m_STDPMinWeight << ' ' << link.m_STDPMaxWeight << '\n';
            Serialization::WriteTraits(output, "LinkTraits", link.m_Traits);
        }
        output << "GenomeEnd\n";
        return output.str();
    }

    Genome Genome::Deserialize(const std::string &data) {
        std::istringstream input(data);
        return Genome(input);
    }

    bool Genome::Validate(std::string *error) const {
        const auto fail = [error](const std::string &message) {
            if (error != nullptr) *error = message;
            return false;
        };

        if (m_NumInputs < 0 || m_NumOutputs < 0) return fail("Genome input/output counts cannot be negative");
        if (m_Depth < 0) return fail("Genome depth cannot be negative");
        if (!std::isfinite(m_Fitness) || !std::isfinite(m_AdjustedFitness) || !std::isfinite(m_OffspringAmount))
            return fail("Genome fitness and offspring state must be finite");
        if (static_cast<std::size_t>(m_NumInputs) + static_cast<std::size_t>(m_NumOutputs) > m_NeuronGenes.size())
            return fail("Genome input/output counts exceed its neuron count");
        if (m_initial_num_neurons < 0 || m_initial_num_links < 0) return fail("Genome initial complexity cannot be negative");

        std::map<int, bool> neuron_ids;
        int actual_inputs = 0;
        int actual_outputs = 0;
        for (std::size_t i = 0; i < m_NeuronGenes.size(); ++i) {
            const auto &neuron = m_NeuronGenes[i];
            if (neuron.ID() <= 0 || !neuron_ids.emplace(neuron.ID(), true).second) return fail("Genome neuron IDs must be positive and unique");
            if (neuron.Type() < INPUT || neuron.Type() > OUTPUT) return fail("Genome contains an invalid neuron type");
            if (neuron.m_ActFunction < SIGNED_SIGMOID || neuron.m_ActFunction > MCCULLOCH_PITTS) return fail("Genome contains an invalid activation function");
            if (!std::isfinite(neuron.m_SplitY) || !std::isfinite(neuron.m_A) || !std::isfinite(neuron.m_B) || !std::isfinite(neuron.m_TimeConstant) ||
                !std::isfinite(neuron.m_Bias) || !std::isfinite(neuron.m_SpikeThreshold) || !std::isfinite(neuron.m_ResetPotential) ||
                !std::isfinite(neuron.m_RestingPotential) || !std::isfinite(neuron.m_RefractoryPeriod) || !std::isfinite(neuron.m_MembraneResistance) ||
                !std::isfinite(neuron.m_AdaptationTimeConstant) || !std::isfinite(neuron.m_AdaptationIncrement) || !std::isfinite(neuron.m_RateTimeConstant) ||
                !std::isfinite(neuron.m_IzhikevichA) || !std::isfinite(neuron.m_IzhikevichB) || !std::isfinite(neuron.m_IzhikevichC) ||
                !std::isfinite(neuron.m_IzhikevichD))
                return fail("Genome neuron parameters must be finite");
            if (IsSpikingActivation(neuron.m_ActFunction) && neuron.m_TimeConstant <= 0.0)
                return fail("Spiking genome neurons require positive time constants");
            if (neuron.m_RefractoryPeriod < 0.0 || neuron.m_MembraneResistance <= 0.0 || neuron.m_AdaptationTimeConstant <= 0.0 ||
                neuron.m_RateTimeConstant <= 0.0)
                return fail(
                    "Spiking neuron resistance and adaptation time constants must be positive and refractory periods "
                    "non-negative");
            if (neuron.Type() == INPUT || neuron.Type() == BIAS)
                ++actual_inputs;
            else if (neuron.Type() == OUTPUT)
                ++actual_outputs;
            if (i < static_cast<std::size_t>(m_NumInputs)) {
                if (neuron.Type() != INPUT && neuron.Type() != BIAS) return fail("Genome input neurons are not stored first");
            } else if (i < static_cast<std::size_t>(m_NumInputs + m_NumOutputs) && neuron.Type() != OUTPUT) {
                return fail("Genome output neurons do not follow its inputs");
            }
        }
        if (actual_inputs != m_NumInputs || actual_outputs != m_NumOutputs) return fail("Genome input/output counts do not match its neuron types");
        std::map<int, bool> innovation_ids;
        std::set<std::pair<int, int>> link_endpoints;
        for (const auto &link : m_LinkGenes) {
            if (link.InnovationID() <= 0 || !innovation_ids.emplace(link.InnovationID(), true).second)
                return fail("Genome innovation IDs must be positive and unique");
            if (!link_endpoints.emplace(link.FromNeuronID(), link.ToNeuronID()).second) return fail("Genome link endpoints must be unique");
            if (neuron_ids.count(link.FromNeuronID()) == 0 || neuron_ids.count(link.ToNeuronID()) == 0) return fail("Genome link endpoint does not exist");
            if (!std::isfinite(link.GetWeight())) return fail("Genome link weights must be finite");
            if (!std::isfinite(link.m_SynapticDelay) || !std::isfinite(link.m_SynapticTimeConstant) || !std::isfinite(link.m_STDPPlus) ||
                !std::isfinite(link.m_STDPMinus) || !std::isfinite(link.m_STDPTauPlus) || !std::isfinite(link.m_STDPTauMinus) ||
                !std::isfinite(link.m_STDPMinWeight) || !std::isfinite(link.m_STDPMaxWeight))
                return fail("Genome synapse parameters must be finite");
            if (link.m_SynapticDelay < 0.0 || link.m_SynapticTimeConstant <= 0.0 || link.m_STDPTauPlus <= 0.0 || link.m_STDPTauMinus <= 0.0)
                return fail("Genome synapse delays must be non-negative and time constants positive");
        }
        return true;
    }

    bool Genome::IsIdenticalTo(const Genome &other) const {
        if (m_NumInputs != other.m_NumInputs || m_NumOutputs != other.m_NumOutputs || m_NeuronGenes.size() != other.m_NeuronGenes.size() ||
            m_LinkGenes.size() != other.m_LinkGenes.size()) {
            return false;
        }
        if (m_GenomeGene.m_Traits != other.m_GenomeGene.m_Traits) return false;
        for (std::size_t i = 0; i < m_NeuronGenes.size(); ++i) {
            if (!(m_NeuronGenes[i] == other.m_NeuronGenes[i])) return false;
            if (m_NeuronGenes[i].m_Traits != other.m_NeuronGenes[i].m_Traits) return false;
        }
        for (std::size_t i = 0; i < m_LinkGenes.size(); ++i) {
            if (!(m_LinkGenes[i] == other.m_LinkGenes[i])) return false;
            if (m_LinkGenes[i].m_Traits != other.m_LinkGenes[i].m_Traits) return false;
        }
        return true;
    }

    void Genome::PrintTraits(std::map<std::string, Trait> &traits) {
        for (auto t = traits.begin(); t != traits.end(); t++) {
            bool doit = false;
            std::string s = t->second.dep_key;
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
                    for (int ix = 0; ix < t->second.dep_values.size(); ix++) {
                        if (traits[s].value == (t->second.dep_values[ix])) {
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
                if (std::holds_alternative<intsetelement>(t->second.value)) {
                    std::cout << (std::get<intsetelement>(t->second.value)).value;
                }
                if (std::holds_alternative<floatsetelement>(t->second.value)) {
                    std::cout << (std::get<floatsetelement>(t->second.value)).value;
                }

                std::cout << ", ";
            }
        }
    }

    void Genome::PrintAllTraits() {
        std::cout << "====================================================================\n";
        std::cout << "Genome:\n"
                  << "==================================\n";
        PrintTraits(m_GenomeGene.m_Traits);

        std::cout << "\n";

        std::cout << "====================================================================\n";
        std::cout << "Neurons:\n"
                  << "==================================\n";
        for (auto it = m_NeuronGenes.begin(); it != m_NeuronGenes.end(); it++) {
            std::cout << "ID: " << it->ID() << " : ";
            PrintTraits((*it).m_Traits);

            std::cout << "\n";
        }
        std::cout << "==================================\n";

        std::cout << "Links:\n"
                  << "==================================\n";
        for (auto it = m_LinkGenes.begin(); it != m_LinkGenes.end(); it++) {
            std::cout << "ID: " << it->InnovationID() << " : ";
            PrintTraits((*it).m_Traits);
            std::cout << "\n";
        }
        std::cout << "==================================\n";
        std::cout << "====================================================================\n";
    }

    ////////////////////////////////////////////
    // Evovable Substrate Hyper NEAT.
    // For more info on the algorithm check: http://eplex.cs.ucf.edu/ESHyperNEAT/
    ////////////////////////////////////////////

    // Prunes over-long axons and optionally converts lengths into delays.
    inline void FinalizeSpatialConnections(NeuralNetwork &network, const Substrate &substrate) {
        network.UpdateConnectionGeometry(substrate.m_use_spatial_distance_for_delays, substrate.m_conduction_velocity);
        if (substrate.m_max_connection_length >= 0.0) {
            network.m_connections.erase(
                std::remove_if(network.m_connections.begin(), network.m_connections.end(),
                               [&substrate](const Connection &connection) { return connection.m_length > substrate.m_max_connection_length; }),
                network.m_connections.end());
        }
    }

    void Genome::BuildESHyperNEATPhenotype(NeuralNetwork &net, Substrate &subst, Parameters &params) {
        if (subst.m_input_coords.empty() || subst.m_output_coords.empty()) {
            throw std::invalid_argument("An ES-HyperNEAT substrate requires input and output coordinates");
        }
        ValidateSpatialSubstrate(subst, "ES-HyperNEAT");
        if (params.InitialDepth > params.MaxDepth) {
            throw std::invalid_argument("ES-HyperNEAT InitialDepth cannot exceed MaxDepth");
        }
        const int substrate_dimensions = subst.GetMaxDims();
        if (substrate_dimensions > 3) {
            throw std::invalid_argument("ES-HyperNEAT supports two- or three-dimensional substrates");
        }
        const bool three_dimensional = substrate_dimensions >= 3;
        // Depth nine permits 349,525 quadtree nodes; an octree reaches a similar
        // size at depth six. Reject larger trees before an accidental parameter
        // value can exhaust memory or make construction effectively unbounded.
        const unsigned int max_safe_depth = three_dimensional ? 6U : 9U;
        if (params.MaxDepth > max_safe_depth) {
            throw std::invalid_argument(std::string("ES-HyperNEAT MaxDepth exceeds the supported safe ") +
                                        (three_dimensional ? "3D limit of 6" : "2D limit of 9"));
        }
        const std::pair<const char *, Real> finite_values[] = {{"DivisionThreshold", params.DivisionThreshold},
                                                               {"VarianceThreshold", params.VarianceThreshold},
                                                               {"BandThreshold", params.BandThreshold},
                                                               {"CPPN_Bias", params.CPPN_Bias},
                                                               {"Width", params.Width},
                                                               {"Height", params.Height},
                                                               {"Depth", params.Depth},
                                                               {"Qtree_X", params.Qtree_X},
                                                               {"Qtree_Y", params.Qtree_Y},
                                                               {"Qtree_Z", params.Qtree_Z},
                                                               {"LeoThreshold", params.LeoThreshold},
                                                               {"maximum substrate weight", subst.m_max_weight_and_bias}};
        for (const auto &value : finite_values) {
            if (!std::isfinite(value.second)) {
                throw std::invalid_argument(std::string("ES-HyperNEAT ") + value.first + " must be finite");
            }
        }
        if (params.DivisionThreshold < 0.0 || params.VarianceThreshold < 0.0 || params.BandThreshold < 0.0 || params.Width <= 0.0 || params.Height <= 0.0 ||
            (three_dimensional && params.Depth <= 0.0) || subst.m_max_weight_and_bias < 0.0) {
            throw std::invalid_argument(
                "ES-HyperNEAT thresholds and weight range must be non-negative, "
                "and spatial tree dimensions must be positive");
        }

        const int dimensions = three_dimensional ? 3 : 2;
        const int required_inputs = dimensions * 2 + (subst.m_with_distance ? 1 : 0) + 1;
        const int required_outputs = params.Leo ? 2 : 1;
        if (m_NumInputs < required_inputs || m_NumOutputs < required_outputs) {
            throw std::invalid_argument(
                "The CPPN does not provide enough inputs or outputs for "
                "ES-HyperNEAT");
        }
        const auto validate_coordinates = [dimensions](const std::vector<std::vector<Real>> &coordinates) {
            for (const auto &coordinate : coordinates) {
                if (coordinate.empty() || coordinate.size() > static_cast<std::size_t>(dimensions) ||
                    !std::all_of(coordinate.begin(), coordinate.end(), [](Real value) { return std::isfinite(value); })) {
                    return false;
                }
            }
            return true;
        };
        if (!validate_coordinates(subst.m_input_coords) || !validate_coordinates(subst.m_output_coords)) {
            throw std::invalid_argument(
                "ES-HyperNEAT substrate coordinates must be finite and "
                "dimensionally consistent");
        }
        if (subst.m_input_coords.size() > static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
            subst.m_output_coords.size() > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
            throw std::length_error("ES-HyperNEAT substrate exceeds the supported index range");
        }

        struct TreeNode {
            Real x;
            Real y;
            Real z;
            Real width;
            Real height;
            Real depth;
            Real weight = 0.0;
            Real leo = 0.0;
            unsigned int level;
            std::vector<std::unique_ptr<TreeNode>> children;

            bool Divided() const { return !children.empty(); }
        };
        struct CandidateConnection {
            std::vector<Real> source;
            std::vector<Real> target;
            Real weight;
        };

        const std::size_t children_per_node = three_dimensional ? 8U : 4U;
        std::size_t tree_node_limit = 1;
        std::size_t nodes_at_level = 1;
        const unsigned int subdivision_levels = std::max(1U, params.MaxDepth);
        for (unsigned int level = 0; level < subdivision_levels; ++level) {
            nodes_at_level *= children_per_node;
            tree_node_limit += nodes_at_level;
        }

        NeuralNetwork cppn(true);
        BuildPhenotype(cppn);
        int cppn_activation_steps = 8;
        if (!HasLoops()) {
            CalculateDepth();
            cppn_activation_steps = std::max(1, static_cast<int>(GetDepth()));
        }

        const auto normalized_coordinate = [dimensions](const std::vector<Real> &coordinate) {
            std::vector<Real> result(static_cast<std::size_t>(dimensions), 0.0);
            std::copy(coordinate.begin(), coordinate.end(), result.begin());
            return result;
        };
        const auto query_cppn = [&](const std::vector<Real> &source, const std::vector<Real> &target) {
            std::vector<Real> inputs(static_cast<std::size_t>(m_NumInputs), 0.0);
            for (int dimension = 0; dimension < dimensions; ++dimension) {
                if (static_cast<std::size_t>(dimension) < source.size())
                    inputs[static_cast<std::size_t>(dimension)] = source[static_cast<std::size_t>(dimension)];
                if (static_cast<std::size_t>(dimension) < target.size())
                    inputs[static_cast<std::size_t>(dimensions + dimension)] = target[static_cast<std::size_t>(dimension)];
            }
            if (subst.m_with_distance) {
                Real squared_distance = 0.0;
                for (int dimension = 0; dimension < dimensions; ++dimension) {
                    const Real difference = inputs[static_cast<std::size_t>(dimension)] - inputs[static_cast<std::size_t>(dimensions + dimension)];
                    squared_distance += difference * difference;
                }
                inputs[inputs.size() - 2] = std::sqrt(squared_distance);
            }
            inputs.back() = params.CPPN_Bias;

            cppn.Flush();
            cppn.Input(inputs);
            for (int step = 0; step < cppn_activation_steps; ++step) cppn.Activate();
            const std::vector<Real> outputs = cppn.Output();
            if (outputs.size() < static_cast<std::size_t>(required_outputs) || !std::isfinite(outputs.front()) ||
                (params.Leo && !std::isfinite(outputs.back()))) {
                throw std::runtime_error("ES-HyperNEAT CPPN produced an invalid output");
            }
            return std::pair<Real, Real>(outputs.front(), params.Leo ? outputs.back() : 0.0);
        };
        const auto variance = [](const TreeNode &node) {
            if (!node.Divided()) return static_cast<Real>(0);
            Real mean = 0.0;
            for (const auto &child : node.children) mean += child->weight;
            mean /= static_cast<Real>(node.children.size());
            Real result = 0.0;
            for (const auto &child : node.children) {
                const Real difference = child->weight - mean;
                result += difference * difference;
            }
            return result / static_cast<Real>(node.children.size());
        };
        const auto generated_coordinate = [dimensions](Real x, Real y, Real z) {
            std::vector<Real> coordinate(static_cast<std::size_t>(dimensions), 0.0);
            coordinate[0] = x;
            coordinate[1] = y;
            if (dimensions == 3) coordinate[2] = z;
            return coordinate;
        };

        const auto sample_connections = [&](const std::vector<Real> &fixed, bool outgoing) {
            auto root = std::make_unique<TreeNode>(TreeNode{params.Qtree_X,
                                                            params.Qtree_Y,
                                                            three_dimensional ? params.Qtree_Z : 0.0f,
                                                            params.Width,
                                                            params.Height,
                                                            three_dimensional ? params.Depth : 0.0f,
                                                            0.0,
                                                            0.0,
                                                            1,
                                                            {}});
            std::queue<TreeNode *> pending;
            pending.push(root.get());
            std::size_t tree_nodes = 1;
            while (!pending.empty()) {
                TreeNode *parent = pending.front();
                pending.pop();
                const Real child_width = parent->width / 2.0;
                const Real child_height = parent->height / 2.0;
                const Real child_depth = three_dimensional ? parent->depth / 2.0 : 0.0;
                const Real xs[] = {parent->x - child_width, parent->x - child_width, parent->x + child_width, parent->x + child_width};
                const Real ys[] = {parent->y - child_height, parent->y + child_height, parent->y + child_height, parent->y - child_height};
                const std::size_t depth_layers = three_dimensional ? 2U : 1U;
                parent->children.reserve(children_per_node);
                for (std::size_t depth_layer = 0; depth_layer < depth_layers; ++depth_layer) {
                    const Real z = three_dimensional ? parent->z + (depth_layer == 0 ? -child_depth : child_depth) : 0.0;
                    for (std::size_t index = 0; index < 4; ++index) {
                        auto child = std::make_unique<TreeNode>(
                            TreeNode{xs[index], ys[index], z, child_width, child_height, child_depth, 0.0, 0.0, parent->level + 1, {}});
                        const std::vector<Real> coordinate = generated_coordinate(child->x, child->y, child->z);
                        const auto outputs = outgoing ? query_cppn(fixed, coordinate) : query_cppn(coordinate, fixed);
                        child->weight = outputs.first;
                        child->leo = outputs.second;
                        parent->children.push_back(std::move(child));
                    }
                }
                tree_nodes += children_per_node;
                if (tree_nodes > tree_node_limit) {
                    throw std::runtime_error("ES-HyperNEAT quadtree exceeded its calculated limit");
                }
                if (parent->level < params.InitialDepth || (parent->level < params.MaxDepth && variance(*parent) > params.DivisionThreshold)) {
                    for (auto &child : parent->children) pending.push(child.get());
                }
            }

            std::vector<CandidateConnection> connections;
            std::function<void(const TreeNode &)> prune_and_express;
            prune_and_express = [&](const TreeNode &parent) {
                for (const auto &child_pointer : parent.children) {
                    const TreeNode &child = *child_pointer;
                    if (child.Divided() && variance(child) > params.VarianceThreshold) {
                        prune_and_express(child);
                        continue;
                    }
                    if (params.Leo && child.leo <= params.LeoThreshold) continue;

                    const std::vector<Real> center = generated_coordinate(child.x, child.y, child.z);
                    std::vector<Real> left = center;
                    std::vector<Real> right = center;
                    std::vector<Real> top = center;
                    std::vector<Real> bottom = center;
                    std::vector<Real> front = center;
                    std::vector<Real> back = center;
                    left[0] -= parent.width;
                    right[0] += parent.width;
                    top[1] -= parent.height;
                    bottom[1] += parent.height;
                    if (three_dimensional) {
                        front[2] -= parent.depth;
                        back[2] += parent.depth;
                    }
                    const auto boundary_difference = [&](const std::vector<Real> &coordinate) {
                        const Real boundary_weight = outgoing ? query_cppn(fixed, coordinate).first : query_cppn(coordinate, fixed).first;
                        return std::abs(child.weight - boundary_weight);
                    };
                    const Real horizontal = std::min(boundary_difference(left), boundary_difference(right));
                    const Real vertical = std::min(boundary_difference(top), boundary_difference(bottom));
                    const Real spatial = three_dimensional ? std::min(boundary_difference(front), boundary_difference(back)) : 0.0;
                    if (std::max({horizontal, vertical, spatial}) <= params.BandThreshold) continue;

                    connections.push_back(outgoing ? CandidateConnection{fixed, center, child.weight} : CandidateConnection{center, fixed, child.weight});
                }
            };
            prune_and_express(*root);
            return connections;
        };

        const std::size_t input_count = subst.m_input_coords.size();
        const std::size_t output_count = subst.m_output_coords.size();
        const std::size_t hidden_offset = input_count + output_count;
        std::map<std::vector<Real>, std::size_t> hidden_indices;
        std::vector<std::vector<Real>> hidden_coordinates;
        const auto add_hidden = [&](const std::vector<Real> &coordinate) -> std::pair<std::size_t, bool> {
            const auto existing = hidden_indices.find(coordinate);
            if (existing != hidden_indices.end()) return {existing->second, false};
            if (hidden_coordinates.size() >= tree_node_limit) {
                throw std::length_error("ES-HyperNEAT generated too many hidden nodes");
            }
            const std::size_t index = hidden_coordinates.size();
            hidden_coordinates.push_back(coordinate);
            hidden_indices.emplace(coordinate, index);
            return {index, true};
        };

        std::vector<Connection> generated_connections;
        std::set<std::pair<std::size_t, std::size_t>> connection_endpoints;
        const auto add_connection = [&](std::size_t source, std::size_t target, Real raw_weight) {
            if (source == target || !std::isfinite(raw_weight) || !connection_endpoints.emplace(source, target).second) {
                return;
            }
            if (source > static_cast<std::size_t>(std::numeric_limits<int>::max()) || target > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
                throw std::length_error("ES-HyperNEAT connection index exceeds the supported range");
            }
            Connection connection;
            connection.m_source_neuron_idx = static_cast<int>(source);
            connection.m_target_neuron_idx = static_cast<int>(target);
            connection.m_weight = raw_weight * subst.m_max_weight_and_bias;
            generated_connections.push_back(connection);
        };

        for (std::size_t input = 0; input < input_count; ++input) {
            const std::vector<Real> coordinate = normalized_coordinate(subst.m_input_coords[input]);
            for (const auto &candidate : sample_connections(coordinate, true)) {
                const auto hidden = add_hidden(candidate.target);
                add_connection(input, hidden_offset + hidden.first, candidate.weight);
            }
        }

        std::vector<std::vector<Real>> frontier = hidden_coordinates;
        for (unsigned int iteration = 0; iteration < params.IterationLevel && !frontier.empty(); ++iteration) {
            std::vector<std::vector<Real>> next_frontier;
            for (const auto &source_coordinate : frontier) {
                const auto source = hidden_indices.find(source_coordinate);
                if (source == hidden_indices.end()) throw std::logic_error("ES-HyperNEAT lost a generated hidden node");
                for (const auto &candidate : sample_connections(source_coordinate, true)) {
                    const auto target = add_hidden(candidate.target);
                    add_connection(hidden_offset + source->second, hidden_offset + target.first, candidate.weight);
                    if (target.second) next_frontier.push_back(candidate.target);
                }
            }
            frontier = std::move(next_frontier);
        }

        for (std::size_t output = 0; output < output_count; ++output) {
            const std::vector<Real> coordinate = normalized_coordinate(subst.m_output_coords[output]);
            for (const auto &candidate : sample_connections(coordinate, false)) {
                const auto source = hidden_indices.find(candidate.source);
                if (source != hidden_indices.end()) {
                    add_connection(hidden_offset + source->second, input_count + output, candidate.weight);
                }
            }
        }

        std::vector<Neuron> generated_neurons;
        generated_neurons.reserve(hidden_offset + hidden_coordinates.size());
        for (std::size_t index = 0; index < input_count; ++index) {
            Neuron neuron;
            neuron.m_a = 1.0;
            neuron.m_b = 0.0;
            SetSpatialCoordinates(neuron, subst.m_input_coords[index]);
            neuron.m_activation_function_type = LINEAR;
            neuron.m_type = index + 1 == input_count ? BIAS : INPUT;
            generated_neurons.push_back(neuron);
        }
        for (const auto &coordinate : subst.m_output_coords) {
            Neuron neuron;
            neuron.m_a = 1.0;
            neuron.m_b = 0.0;
            SetSpatialCoordinates(neuron, coordinate);
            neuron.m_activation_function_type = subst.m_output_nodes_activation;
            neuron.m_type = OUTPUT;
            generated_neurons.push_back(neuron);
        }
        for (const auto &coordinate : hidden_coordinates) {
            Neuron neuron;
            neuron.m_a = 1.0;
            neuron.m_b = 0.0;
            SetSpatialCoordinates(neuron, coordinate);
            neuron.m_activation_function_type = subst.m_hidden_nodes_activation;
            neuron.m_type = HIDDEN;
            generated_neurons.push_back(neuron);
        }

        // Apply physical constraints before reachability pruning so an axon
        // removed for being too long cannot leave a retained hidden island.
        for (auto &connection : generated_connections) {
            const Neuron &source = generated_neurons[static_cast<std::size_t>(connection.m_source_neuron_idx)];
            const Neuron &target = generated_neurons[static_cast<std::size_t>(connection.m_target_neuron_idx)];
            const Real dx = target.m_x - source.m_x;
            const Real dy = target.m_y - source.m_y;
            const Real dz = target.m_z - source.m_z;
            connection.m_length = std::sqrt(dx * dx + dy * dy + dz * dz);
            if (subst.m_use_spatial_distance_for_delays) {
                connection.m_synaptic_delay = connection.m_length / subst.m_conduction_velocity;
            }
        }
        if (subst.m_max_connection_length >= 0.0) {
            generated_connections.erase(std::remove_if(generated_connections.begin(), generated_connections.end(),
                                                       [&subst](const Connection &connection) { return connection.m_length > subst.m_max_connection_length; }),
                                        generated_connections.end());
        }

        // Retain only hidden nodes that are both reachable from an input and can
        // reach an output. This removes disconnected islands and entire cycles,
        // not merely nodes with an immediate missing predecessor/successor.
        const std::size_t neuron_count = generated_neurons.size();
        std::vector<bool> forward_reachable(neuron_count, false);
        std::vector<bool> backward_reachable(neuron_count, false);
        std::fill(forward_reachable.begin(), forward_reachable.begin() + static_cast<std::ptrdiff_t>(input_count), true);
        std::fill(backward_reachable.begin() + static_cast<std::ptrdiff_t>(input_count),
                  backward_reachable.begin() + static_cast<std::ptrdiff_t>(hidden_offset), true);
        bool changed = true;
        while (changed) {
            changed = false;
            for (const auto &connection : generated_connections) {
                const std::size_t source = static_cast<std::size_t>(connection.m_source_neuron_idx);
                const std::size_t target = static_cast<std::size_t>(connection.m_target_neuron_idx);
                if (forward_reachable[source] && !forward_reachable[target]) {
                    forward_reachable[target] = true;
                    changed = true;
                }
            }
        }
        changed = true;
        while (changed) {
            changed = false;
            for (const auto &connection : generated_connections) {
                const std::size_t source = static_cast<std::size_t>(connection.m_source_neuron_idx);
                const std::size_t target = static_cast<std::size_t>(connection.m_target_neuron_idx);
                if (backward_reachable[target] && !backward_reachable[source]) {
                    backward_reachable[source] = true;
                    changed = true;
                }
            }
        }

        std::vector<int> remap(neuron_count, -1);
        std::vector<Neuron> pruned_neurons;
        pruned_neurons.reserve(neuron_count);
        for (std::size_t index = 0; index < neuron_count; ++index) {
            const bool fixed_node = index < hidden_offset;
            if (fixed_node || (forward_reachable[index] && backward_reachable[index])) {
                remap[index] = static_cast<int>(pruned_neurons.size());
                pruned_neurons.push_back(generated_neurons[index]);
            }
        }
        std::vector<Connection> pruned_connections;
        pruned_connections.reserve(generated_connections.size());
        for (auto connection : generated_connections) {
            const int source = remap[static_cast<std::size_t>(connection.m_source_neuron_idx)];
            const int target = remap[static_cast<std::size_t>(connection.m_target_neuron_idx)];
            if (source >= 0 && target >= 0) {
                connection.m_source_neuron_idx = source;
                connection.m_target_neuron_idx = target;
                pruned_connections.push_back(connection);
            }
        }

        net.Clear();
        net.SetInputOutputDimensions(static_cast<unsigned int>(input_count), static_cast<unsigned int>(output_count));
        net.m_neurons = std::move(pruned_neurons);
        net.m_connections = std::move(pruned_connections);
        FinalizeSpatialConnections(net, subst);
        net.Flush();
    }

}  // namespace NEAT
