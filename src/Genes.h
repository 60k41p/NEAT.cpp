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
 * File:        Genes.h
 * Description: Definitions for the Neuron and Link gene classes.
 */

#pragma once

#include <cmath>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

#include "Parameters.h"
#include "Random.h"
#include "Traits.h"
#include "Types.h"
#include "Utils.h"

namespace NEAT {

    //////////////////////////////////////////////
    // Enumeration for all available neuron types
    //////////////////////////////////////////////
    enum NeuronType { NONE = 0, INPUT, BIAS, HIDDEN, OUTPUT };

    //////////////////////////////////////////////////////////
    // Enumeration for all possible activation function types
    //////////////////////////////////////////////////////////
    enum ActivationFunction {
        SIGNED_SIGMOID = 0,  // Sigmoid function   (default) (blurred cutting plane)
        UNSIGNED_SIGMOID,
        TANH,
        TANH_CUBIC,
        SIGNED_STEP,  // Treshold (0 or 1)  (cutting plane)
        UNSIGNED_STEP,
        SIGNED_GAUSS,  // Gaussian           (symettry)
        UNSIGNED_GAUSS,
        ABS,          // Absolute value |x| (another symettry)
        SIGNED_SINE,  // Sine wave          (smooth repetition)
        UNSIGNED_SINE,
        LINEAR,  // Linear f(x)=x      (combining coordinate frames only)

        RELU,  // Rectifiers
        SOFTPLUS,
        // Stateful activation modes advanced by NeuralNetwork::StepSpiking.
        // Appended to preserve the numeric values of every historical mode.
        SPIKING_LIF,
        SPIKING_ADAPTIVE_LIF,
        SPIKING_IZHIKEVICH,
        // The discrete-time McCulloch-Pitts model. The SPIKING_ prefix is
        // retained as an alias so its stateful/event-driven nature is clear
        // beside the other spiking modes without making the historical name
        // awkward to use.
        MCCULLOCH_PITTS,
        SPIKING_MCCULLOCH_PITTS = MCCULLOCH_PITTS
    };

    // Returns true for the stateful spiking/McCulloch-Pitts activation modes.
    inline bool IsSpikingActivation(ActivationFunction function) {
        return function == SPIKING_LIF || function == SPIKING_ADAPTIVE_LIF || function == SPIKING_IZHIKEVICH || function == MCCULLOCH_PITTS;
    }

    //////////////////////////////////
    // Base Gene class
    //////////////////////////////////
    class Gene {
       public:
        // Arbitrary traits
        std::map<std::string, Trait> m_Traits;

        Gene &operator=(const Gene &) = default;

        // Randomize based on parameters
        void InitTraits(const std::map<std::string, TraitParameters> &tp, RNG &a_RNG) {
            for (auto it = tp.begin(); it != tp.end(); ++it) {
                // Check what kind of type is this and create such trait
                TraitType t;

                if (it->second.type == "int") {
                    IntTraitParameters itp = std::get<IntTraitParameters>(it->second.m_Details);
                    if (itp.min > itp.max) {
                        throw std::invalid_argument("Integer trait minimum exceeds maximum");
                    }
                    t = a_RNG.RandInt(itp.min, itp.max);
                } else if (it->second.type == "float") {
                    FloatTraitParameters itp = std::get<FloatTraitParameters>(it->second.m_Details);
                    if (itp.min > itp.max) {
                        throw std::invalid_argument("Floating-point trait minimum exceeds maximum");
                    }
                    Real x = a_RNG.RandFloat();
                    Scale(x, 0, 1, itp.min, itp.max);
                    t = x;
                } else if (it->second.type == "str" || it->second.type == "string") {
                    StringTraitParameters itp = std::get<StringTraitParameters>(it->second.m_Details);
                    std::vector<Real> probs = itp.probs;
                    if (itp.set.empty()) {
                        throw std::runtime_error("Empty set of string traits");
                    }
                    probs.resize(itp.set.size());  // in case it didn't match length

                    int idx = a_RNG.Roulette(probs);
                    t = itp.set[idx];
                } else if (it->second.type == "intset") {
                    IntSetTraitParameters itp = std::get<IntSetTraitParameters>(it->second.m_Details);
                    std::vector<Real> probs = itp.probs;
                    if (itp.set.empty()) {
                        throw std::runtime_error("Empty set of int traits");
                    }
                    probs.resize(itp.set.size());

                    int idx = a_RNG.Roulette(probs);
                    t = itp.set[idx];
                } else if (it->second.type == "floatset") {
                    FloatSetTraitParameters itp = std::get<FloatSetTraitParameters>(it->second.m_Details);
                    std::vector<Real> probs = itp.probs;
                    if (itp.set.empty()) {
                        throw std::runtime_error("Empty set of float traits");
                    }
                    probs.resize(itp.set.size());

                    int idx = a_RNG.Roulette(probs);
                    t = itp.set[idx];
                } else {
                    throw std::invalid_argument("Unknown trait type: " + it->second.type);
                }
                Trait tr;
                tr.value = t;
                tr.dep_key = it->second.dep_key;
                tr.dep_values = it->second.dep_values;
                // todo check for invalid dep_values types here
                m_Traits[it->first] = tr;
            }
        }

        // Traits are merged with this other parent
        void MateTraits(const std::map<std::string, Trait> &t, RNG &a_RNG) {
            for (auto it = t.begin(); it != t.end(); ++it) {
                // Both parents must share the key; otherwise there is nothing to mate.
                const auto mineIt = m_Traits.find(it->first);
                if (mineIt == m_Traits.end()) {
                    continue;
                }
                TraitType mine = mineIt->second.value;
                TraitType yours = it->second.value;

                if (mine.index() != yours.index()) {
                    throw std::runtime_error("Types of traits don't match in mating");
                }

                {
                    if (a_RNG.RandFloat() < 0.5)  // pick either one
                    {
                        m_Traits[it->first].value = (a_RNG.RandFloat() < 0.5) ? mine : yours;
                    } else {
                        // try to average if numeric
                        if (std::holds_alternative<int>(mine)) {
                            int m1 = std::get<int>(mine);
                            int m2 = std::get<int>(yours);
                            m_Traits[it->first].value = static_cast<int>((static_cast<long long>(m1) + static_cast<long long>(m2)) / 2LL);
                        } else if (std::holds_alternative<Real>(mine)) {
                            Real m1 = std::get<Real>(mine);
                            Real m2 = std::get<Real>(yours);
                            m_Traits[it->first].value = (m1 + m2) / 2;
                        } else if (std::holds_alternative<std::string>(mine)) {
                            // strings are always either-or
                            m_Traits[it->first].value = (a_RNG.RandFloat() < 0.5) ? mine : yours;
                        } else if (std::holds_alternative<intsetelement>(mine)) {
                            // int sets are always either-or
                            m_Traits[it->first].value = (a_RNG.RandFloat() < 0.5) ? mine : yours;
                        } else if (std::holds_alternative<floatsetelement>(mine)) {
                            // float sets are always either-or
                            m_Traits[it->first].value = (a_RNG.RandFloat() < 0.5) ? mine : yours;
                        }
                    }
                }
            }
        }

        // Traits are mutated according to parameters
        bool MutateTraits(const std::map<std::string, TraitParameters> &tp, RNG &a_RNG) {
            bool did_mutate = false;
            for (auto it = tp.begin(); it != tp.end(); ++it) {
                auto traitIt = m_Traits.find(it->first);
                if (traitIt == m_Traits.end()) {
                    continue;
                }

                // only mutate the trait if it's enabled
                bool doit = false;
                if (!it->second.dep_key.empty()) {
                    // there is such trait..
                    if (m_Traits.count(it->second.dep_key) != 0) {
                        // and it matches any of the right values?
                        for (const auto &dv : it->second.dep_values) {
                            if (m_Traits.at(it->second.dep_key).value == dv) {
                                doit = true;
                                break;
                            }
                        }
                    }
                } else {
                    doit = true;
                }

                if (doit) {
                    // Mutate?
                    if (a_RNG.RandFloat() < it->second.m_MutationProb) {
                        const std::string &ty = it->second.type;
                        if (ty == "int") {
                            IntTraitParameters itp = std::get<IntTraitParameters>(it->second.m_Details);
                            if (itp.min > itp.max) {
                                throw std::invalid_argument("Integer trait minimum exceeds maximum");
                            }
                            int val = std::get<int>(traitIt->second.value);
                            int original = val;
                            // determine type of mutation - modify or replace, according to parameters
                            if (a_RNG.RandFloat() < itp.mut_replace_prob) {
                                // replace, guaranteeing a different value when possible
                                if (itp.min == itp.max) {
                                    continue;
                                }
                                if (original >= itp.min && original <= itp.max) {
                                    val = a_RNG.RandInt(itp.min, itp.max - 1);
                                    if (val >= original) {
                                        ++val;
                                    }
                                } else {
                                    val = a_RNG.RandInt(itp.min, itp.max);
                                }
                                traitIt->second.value = val;
                                did_mutate = true;
                            } else {
                                // modify
                                if (itp.mut_power <= 0 || itp.min == itp.max) {
                                    continue;
                                }
                                for (int attempt = 0; attempt < 32 && val == original; ++attempt) {
                                    val = original + a_RNG.RandInt(-itp.mut_power, itp.mut_power);
                                    Clamp(val, itp.min, itp.max);
                                }
                                if (val == original) {
                                    val = (original > itp.min) ? original - 1 : original + 1;
                                    Clamp(val, itp.min, itp.max);
                                }
                                traitIt->second.value = val;
                                did_mutate = true;
                            }
                        } else if (ty == "float") {
                            FloatTraitParameters itp = std::get<FloatTraitParameters>(it->second.m_Details);
                            if (itp.min > itp.max) {
                                throw std::invalid_argument("Floating-point trait minimum exceeds maximum");
                            }
                            Real val = std::get<Real>(traitIt->second.value);
                            Real original = val;
                            // determine type of mutation - modify or replace, according to parameters
                            if (a_RNG.RandFloat() < itp.mut_replace_prob) {
                                // replace
                                if (itp.min == itp.max) {
                                    continue;
                                }
                                val = a_RNG.RandFloat();
                                Scale(val, 0.0, 1.0, itp.min, itp.max);
                                if (val == original) {
                                    val = std::nextafter(original, original < itp.max ? itp.max : itp.min);
                                }
                                traitIt->second.value = val;
                                did_mutate = true;
                            } else {
                                // modify
                                if (itp.mut_power <= 0.0 || itp.min == itp.max) {
                                    continue;
                                }
                                for (int attempt = 0; attempt < 32 && val == original; ++attempt) {
                                    val = original + a_RNG.RandFloatSigned() * itp.mut_power;
                                    Clamp(val, itp.min, itp.max);
                                }
                                if (val == original) {
                                    val = std::nextafter(original, original < itp.max ? itp.max : itp.min);
                                }
                                traitIt->second.value = val;
                                did_mutate = true;
                            }

                        } else if (ty == "str" || ty == "string") {
                            StringTraitParameters itp = std::get<StringTraitParameters>(it->second.m_Details);
                            const std::string original = std::get<std::string>(traitIt->second.value);
                            std::vector<std::string> alternatives;
                            std::vector<Real> probs;
                            for (std::size_t i = 0; i < itp.set.size(); ++i) {
                                if (itp.set[i] != original) {
                                    alternatives.push_back(itp.set[i]);
                                    probs.push_back(i < itp.probs.size() ? itp.probs[i] : 0.0);
                                }
                            }
                            if (alternatives.empty()) {
                                continue;
                            }
                            // now choose the new idx from the set
                            traitIt->second.value = alternatives[static_cast<std::size_t>(a_RNG.Roulette(probs))];
                            did_mutate = true;
                        } else if (ty == "intset") {
                            IntSetTraitParameters itp = std::get<IntSetTraitParameters>(it->second.m_Details);
                            const intsetelement original = std::get<intsetelement>(traitIt->second.value);
                            std::vector<intsetelement> alternatives;
                            std::vector<Real> probs;
                            for (std::size_t i = 0; i < itp.set.size(); ++i) {
                                if (itp.set[i].value != original.value) {
                                    alternatives.push_back(itp.set[i]);
                                    probs.push_back(i < itp.probs.size() ? itp.probs[i] : 0.0);
                                }
                            }
                            if (alternatives.empty()) {
                                continue;
                            }
                            // now choose the new idx from the set
                            traitIt->second.value = alternatives[static_cast<std::size_t>(a_RNG.Roulette(probs))];
                            did_mutate = true;
                        } else if (ty == "floatset") {
                            FloatSetTraitParameters itp = std::get<FloatSetTraitParameters>(it->second.m_Details);
                            const floatsetelement original = std::get<floatsetelement>(traitIt->second.value);
                            std::vector<floatsetelement> alternatives;
                            std::vector<Real> probs;
                            for (std::size_t i = 0; i < itp.set.size(); ++i) {
                                if (itp.set[i].value != original.value) {
                                    alternatives.push_back(itp.set[i]);
                                    probs.push_back(i < itp.probs.size() ? itp.probs[i] : 0.0);
                                }
                            }
                            if (alternatives.empty()) {
                                continue;
                            }
                            // now choose the new idx from the set
                            traitIt->second.value = alternatives[static_cast<std::size_t>(a_RNG.Roulette(probs))];
                            did_mutate = true;
                        }
                    }
                }
            }

            return did_mutate;
        }
        // Compute and return distances between each matching pair of traits.
        // The non-const overload forwards to the const one so distance queries
        // also work on const genes.
        std::map<std::string, Real> GetTraitDistances(const std::map<std::string, Trait> &other) {
            return static_cast<const Gene &>(*this).GetTraitDistances(other);
        }

        std::map<std::string, Real> GetTraitDistances(const std::map<std::string, Trait> &other) const {
            std::map<std::string, Real> dist;
            for (auto it = other.begin(); it != other.end(); ++it) {
                const auto mineIt = m_Traits.find(it->first);
                if (mineIt == m_Traits.end()) {
                    continue;
                }
                TraitType mine = mineIt->second.value;
                TraitType yours = it->second.value;

                if (mine.index() != yours.index()) {
                    throw std::runtime_error("Types of traits don't match in distance measure");
                }

                // only do it if the trait if it's enabled
                // todo: not sure about the distance, think more about it
                bool doit = false;
                if (!it->second.dep_key.empty()) {
                    // there is such trait..
                    const auto mineDep = m_Traits.find(it->second.dep_key);
                    const auto otherDep = other.find(it->second.dep_key);
                    if (mineDep != m_Traits.end() && otherDep != other.end()) {
                        // and it has the right value? also the other genome has to have the trait turned on
                        for (const auto &dv : it->second.dep_values) {
                            if ((mineDep->second.value == dv) && (otherDep->second.value == dv)) {
                                doit = true;
                                break;
                            }
                        }
                    }
                } else {
                    doit = true;
                }

                if (doit) {
                    if (std::holds_alternative<int>(mine)) {
                        // distance between ints - calculate directly
                        dist[it->first] = std::abs(std::get<int>(mine) - std::get<int>(yours));
                    } else if (std::holds_alternative<Real>(mine)) {
                        // distance between floats - calculate directly
                        dist[it->first] = std::abs(std::get<Real>(mine) - std::get<Real>(yours));
                    } else if (std::holds_alternative<std::string>(mine)) {
                        // distance between strings - matching is 0, non-matching is 1
                        dist[it->first] = (std::get<std::string>(mine) == std::get<std::string>(yours)) ? 0.0 : 1.0;
                    } else if (std::holds_alternative<intsetelement>(mine)) {
                        // distance between ints - calculate directly
                        dist[it->first] = std::abs(std::get<intsetelement>(mine).value - std::get<intsetelement>(yours).value);
                    } else if (std::holds_alternative<floatsetelement>(mine)) {
                        // distance between floats - calculate directly
                        dist[it->first] = std::abs(std::get<floatsetelement>(mine).value - std::get<floatsetelement>(yours).value);
                    }
                }
            }

            return dist;
        }
    };

    //////////////////////////////////
    // This class defines a link gene
    //
    // A link gene carries an enable bit (Stanley & Miikkulainen 2002,
    // Section 3.2, Figure 3): disabled genes stay in the genome as
    // historical markers but are skipped when building the phenotype.
    // Add-neuron mutation disables (rather than deletes) the split link,
    // and crossover re-enables a matching gene with probability
    // (1 - DisabledGeneInheritRate) when either parent has it disabled.
    //////////////////////////////////
    class LinkGene : public Gene {
        /////////////////////
        // Members
        /////////////////////

       public:
        // These variables are initialized once and cannot be changed anymore

        // The IDs of the neurons that this link connects
        int m_FromNeuronID, m_ToNeuronID;

        // The link's innovation ID
        int m_InnovationID;

        // This variable is modified during evolution The weight of the connection
        Real m_Weight;

        // Whether the link is expressed in the phenotype. Disabled links
        // remain in the genome so crossover can reactivate them later.
        bool m_Enabled;

        // Is it recurrent?
        bool m_IsRecurrent;

        // Spiking synapse parameters. They are inert during the historical
        // rate-network activation paths.
        Real m_SynapticDelay;
        Real m_SynapticTimeConstant;
        bool m_STDPEnabled;
        Real m_STDPPlus;
        Real m_STDPMinus;
        Real m_STDPTauPlus;
        Real m_STDPTauMinus;
        Real m_STDPMinWeight;
        Real m_STDPMaxWeight;

       public:
        Real GetWeight() const { return m_Weight; }

        void SetWeight(const Real a_Weight) { m_Weight = a_Weight; }

        ////////////////
        // Constructors
        ////////////////
        LinkGene() {
            m_FromNeuronID = 0;
            m_ToNeuronID = 0;
            m_InnovationID = 0;
            m_Weight = 0.0;
            m_Enabled = true;
            m_IsRecurrent = false;
            m_SynapticDelay = 0.0;
            m_SynapticTimeConstant = 0.005;
            m_STDPEnabled = false;
            m_STDPPlus = 0.01;
            m_STDPMinus = 0.012;
            m_STDPTauPlus = 0.02;
            m_STDPTauMinus = 0.02;
            m_STDPMinWeight = -8.0;
            m_STDPMaxWeight = 8.0;
        }

        LinkGene(int a_InID, int a_OutID, int a_InnovID, Real a_Wgt, bool a_Recurrent = false) {
            m_FromNeuronID = a_InID;
            m_ToNeuronID = a_OutID;
            m_InnovationID = a_InnovID;

            m_Weight = a_Wgt;
            m_Enabled = true;
            m_IsRecurrent = a_Recurrent;
            m_SynapticDelay = 0.0;
            m_SynapticTimeConstant = 0.005;
            m_STDPEnabled = false;
            m_STDPPlus = 0.01;
            m_STDPMinus = 0.012;
            m_STDPTauPlus = 0.02;
            m_STDPTauMinus = 0.02;
            m_STDPMinWeight = -8.0;
            m_STDPMaxWeight = 8.0;
        }

        // assigment operator
        LinkGene &operator=(const LinkGene &) = default;

        //////////////
        // Methods
        //////////////

        // Access to static (const) variables
        int FromNeuronID() const { return m_FromNeuronID; }

        int ToNeuronID() const { return m_ToNeuronID; }

        int InnovationID() const { return m_InnovationID; }

        bool IsRecurrent() const { return m_IsRecurrent; }

        bool IsEnabled() const { return m_Enabled; }

        void SetEnabled(bool a_Enabled) { m_Enabled = a_Enabled; }

        bool IsLoopedRecurrent() const { return m_FromNeuronID == m_ToNeuronID; }

        // overload '<', '>', '!=' and '==' used for sorting and comparison.
        // Ordering uses the innovation ID; equality compares topology, weights
        // and the enable bit, ignoring the historical innovation ID
        // (reference topology semantics).
        // '!=' is the negation of '==' so the pair stays consistent.
        friend bool operator<(const LinkGene &a_lhs, const LinkGene &a_rhs) { return (a_lhs.m_InnovationID < a_rhs.m_InnovationID); }

        friend bool operator>(const LinkGene &a_lhs, const LinkGene &a_rhs) { return (a_lhs.m_InnovationID > a_rhs.m_InnovationID); }

        friend bool operator==(const LinkGene &a_lhs, const LinkGene &a_rhs) {
            return (a_lhs.m_FromNeuronID == a_rhs.m_FromNeuronID && a_lhs.m_ToNeuronID == a_rhs.m_ToNeuronID && a_lhs.m_Weight == a_rhs.m_Weight &&
                    a_lhs.m_Enabled == a_rhs.m_Enabled && a_lhs.m_IsRecurrent == a_rhs.m_IsRecurrent && a_lhs.m_SynapticDelay == a_rhs.m_SynapticDelay &&
                    a_lhs.m_SynapticTimeConstant == a_rhs.m_SynapticTimeConstant && a_lhs.m_STDPEnabled == a_rhs.m_STDPEnabled &&
                    a_lhs.m_STDPPlus == a_rhs.m_STDPPlus && a_lhs.m_STDPMinus == a_rhs.m_STDPMinus && a_lhs.m_STDPTauPlus == a_rhs.m_STDPTauPlus &&
                    a_lhs.m_STDPTauMinus == a_rhs.m_STDPTauMinus && a_lhs.m_STDPMinWeight == a_rhs.m_STDPMinWeight &&
                    a_lhs.m_STDPMaxWeight == a_rhs.m_STDPMaxWeight);
        }

        friend bool operator!=(const LinkGene &a_lhs, const LinkGene &a_rhs) { return !(a_lhs == a_rhs); }
    };

    ////////////////////////////////////
    // This class defines a neuron gene
    ////////////////////////////////////
    class NeuronGene : public Gene {
        /////////////////////
        // Members
        /////////////////////

       public:
        // These variables are initialized once and cannot be changed anymore

        // Its unique identification number
        int m_ID;

        // Its type and role in the network
        NeuronType m_Type;

       public:
        // These variables are modified during evolution Safe to access directly

        // useful for displaying the genome
        int x, y;
        // Position (depth) within the network
        Real m_SplitY;

        /////////////////////////////////////////////////////////
        // Any additional properties of the neuron
        // should be added here. This may include
        // time constant & bias for leaky integrators,
        // activation function type,
        // activation function slope (or maybe other properties),
        // etc...
        /////////////////////////////////////////////////////////

        // Additional parameters associated with the
        // neuron's activation function.
        // The current activation function may not use
        // any of them anyway.
        // A is usually used to alter the function's slope with a scalar
        // B is usually used to force a bias to the neuron
        // -------------------
        // Sigmoid : using A, B (slope, shift)
        // Step    : using B    (shift)
        // Gauss   : using A, B (slope, shift))
        // Abs     : using B    (shift)
        // Sine    : using A    (frequency, phase)
        // Square  : using A, B (high phase lenght, low phase length)
        // Linear  : using B    (shift)
        Real m_A, m_B;

        // Time constant value used when the neuron is activating in leaky integrator mode
        Real m_TimeConstant;

        // Bias value used when the neuron is activating in leaky integrator mode
        Real m_Bias;

        // The type of activation function the neuron has
        ActivationFunction m_ActFunction;

        // Parameters shared by the spiking activation modes. LIF uses the
        // threshold/reset/resting/refractory/resistance values. Adaptive LIF
        // additionally uses the adaptation values. Izhikevich uses its
        // canonical a/b/c/d parameterization.
        Real m_SpikeThreshold;
        Real m_ResetPotential;
        Real m_RestingPotential;
        Real m_RefractoryPeriod;
        Real m_MembraneResistance;
        Real m_AdaptationTimeConstant;
        Real m_AdaptationIncrement;
        Real m_RateTimeConstant;
        Real m_IzhikevichA;
        Real m_IzhikevichB;
        Real m_IzhikevichC;
        Real m_IzhikevichD;
        // In the original McCulloch-Pitts calculus any active inhibitory
        // afferent vetoes firing, irrespective of excitatory drive.
        bool m_MCPInhibitoryVeto;

        ////////////////
        // Constructors
        ////////////////
        NeuronGene() {
            m_ID = 0;
            m_Type = NONE;
            x = 0;
            y = 0;
            m_SplitY = 0.0;
            m_A = 0.0;
            m_B = 0.0;
            m_TimeConstant = 0.0;
            m_Bias = 0.0;
            m_ActFunction = UNSIGNED_SIGMOID;
            InitSpikingDefaults();
        }

        friend bool operator==(const NeuronGene &a_lhs, const NeuronGene &a_rhs) {
            return (a_lhs.m_ID == a_rhs.m_ID && a_lhs.m_Type == a_rhs.m_Type && a_lhs.x == a_rhs.x && a_lhs.y == a_rhs.y && a_lhs.m_SplitY == a_rhs.m_SplitY &&
                    a_lhs.m_A == a_rhs.m_A && a_lhs.m_B == a_rhs.m_B && a_lhs.m_TimeConstant == a_rhs.m_TimeConstant && a_lhs.m_Bias == a_rhs.m_Bias &&
                    a_lhs.m_ActFunction == a_rhs.m_ActFunction && a_lhs.m_SpikeThreshold == a_rhs.m_SpikeThreshold &&
                    a_lhs.m_ResetPotential == a_rhs.m_ResetPotential && a_lhs.m_RestingPotential == a_rhs.m_RestingPotential &&
                    a_lhs.m_RefractoryPeriod == a_rhs.m_RefractoryPeriod && a_lhs.m_MembraneResistance == a_rhs.m_MembraneResistance &&
                    a_lhs.m_AdaptationTimeConstant == a_rhs.m_AdaptationTimeConstant && a_lhs.m_AdaptationIncrement == a_rhs.m_AdaptationIncrement &&
                    a_lhs.m_RateTimeConstant == a_rhs.m_RateTimeConstant && a_lhs.m_IzhikevichA == a_rhs.m_IzhikevichA &&
                    a_lhs.m_IzhikevichB == a_rhs.m_IzhikevichB && a_lhs.m_IzhikevichC == a_rhs.m_IzhikevichC && a_lhs.m_IzhikevichD == a_rhs.m_IzhikevichD &&
                    a_lhs.m_MCPInhibitoryVeto == a_rhs.m_MCPInhibitoryVeto);
        }

        NeuronGene(NeuronType a_type, int a_id, Real a_splity) {
            m_ID = a_id;
            m_Type = a_type;
            m_SplitY = a_splity;

            // Initialize the node specific parameters
            m_A = 0.0;
            m_B = 0.0;
            m_TimeConstant = 0.0;
            m_Bias = 0.0;
            m_ActFunction = UNSIGNED_SIGMOID;

            x = 0;
            y = 0;
            InitSpikingDefaults();
        }

        // assigment operator
        NeuronGene &operator=(const NeuronGene &) = default;

        //////////////
        // Methods
        //////////////

        // Accessing static (const) variables
        int ID() const { return m_ID; }

        NeuronType Type() const { return m_Type; }

        Real SplitY() const { return m_SplitY; }

        // Initializing
        void Init(Real a_A, Real a_B, Real a_TimeConstant, Real a_Bias, ActivationFunction a_ActFunc) {
            m_A = a_A;
            m_B = a_B;
            m_TimeConstant = a_TimeConstant;
            m_Bias = a_Bias;
            m_ActFunction = a_ActFunc;
        }

        // Initializes the spiking parameters to biophysically sane defaults.
        void InitSpikingDefaults() {
            m_SpikeThreshold = 1.0;
            m_ResetPotential = 0.0;
            m_RestingPotential = 0.0;
            m_RefractoryPeriod = 0.002;
            m_MembraneResistance = 1.0;
            m_AdaptationTimeConstant = 0.1;
            m_AdaptationIncrement = 0.1;
            m_RateTimeConstant = 0.05;
            m_IzhikevichA = 0.02;
            m_IzhikevichB = 0.2;
            m_IzhikevichC = -65.0;
            m_IzhikevichD = 8.0;
            m_MCPInhibitoryVeto = true;
        }
    };

}  // namespace NEAT
