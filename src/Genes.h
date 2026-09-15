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
 * Description: Genotype building blocks: the Gene base (universal-trait storage plus init/mate/mutate/distance
 *              helpers) and its two specializations, LinkGene (a weighted connection endpoint pair) and
 *              NeuronGene (a node with activation parameters). Genomes in src/Genome.h are vectors of these;
 *              the phenotype in src/NeuralNetwork.h is built from them.
 *
 * References: Stanley & Miikkulainen, "Evolving Neural Networks through Augmenting Topologies" (2002),
 *             Sections 2-3 (gene representation, innovation numbers); see
 *             references/Evolving Neural Networks through Augmenting Topologies.pdf.md. Trait mechanics:
 *             src/Traits.h. Intra-repo users: src/Genome.h, src/Genome.cpp, src/Innovation.h.
 */

#pragma once

#include <iostream>
#include <map>
#include <vector>

#include "Parameters.h"
#include "Random.h"
#include "Traits.h"
#include "Utils.h"

namespace NEAT {

    //////////////////////////////////////////////
    // Neuron roles in the network. NONE marks an unset slot; BIAS is the constant-on helper neuron.
    //////////////////////////////////////////////
    enum NeuronType { NONE = 0, INPUT, BIAS, HIDDEN, OUTPUT };

    //////////////////////////////////////////////////////////
    // Activation function applied by a neuron at phenotype activation time.
    // See the matching evaluators in src/NeuralNetwork.cpp.
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
        SOFTPLUS
    };

    //////////////////////////////////
    // Base Gene class: universal-trait storage plus the shared init/mate/mutate/distance operators
    // used by neuron, link and genome-level genes.
    //////////////////////////////////
    class Gene {
       public:
        // Per-trait runtime values keyed by trait name (see src/Traits.h).
        std::map<std::string, Trait> traits_;

        Gene &operator=(const Gene &g) {
            if (this != &g) {
                traits_ = g.traits_;
            }

            return *this;
        }

       private:
        // Draws an index into a discrete trait set honoring the configured selection probabilities.
        // Throws std::runtime_error when the set is empty.
        static int pickSetIndex(const std::vector<double> &probs, size_t setSize, const char *kind, RNG &rng) {
            if (setSize == 0) {
                throw std::runtime_error(std::string("Empty set of ") + kind + " traits");
            }
            std::vector<double> adjustedProbs = probs;
            adjustedProbs.resize(setSize);
            return rng.roulette(adjustedProbs);
        }

       public:
        // Randomizes every declared trait from its parameter definition (uniform ints/floats, roulette for sets).
        void initTraits(const std::map<std::string, TraitParameters> &tp, RNG &rng) {
            for (std::map<std::string, TraitParameters>::const_iterator it = tp.begin(); it != tp.end(); it++) {
                // Check what kind of type is this and create such trait
                TraitType t;

                if (it->second.type == "int") {
                    IntTraitParameters itp = std::get<IntTraitParameters>(it->second.details_);
                    t = rng.randInt(itp.min, itp.max);
                }
                if (it->second.type == "float") {
                    FloatTraitParameters itp = std::get<FloatTraitParameters>(it->second.details_);
                    double x = rng.randFloat();
                    scale(x, 0, 1, itp.min, itp.max);
                    t = x;
                }
                if (it->second.type == "str") {
                    StringTraitParameters itp = std::get<StringTraitParameters>(it->second.details_);
                    int index = pickSetIndex(itp.probs, itp.set.size(), "string", rng);
                    t = itp.set[index];
                }
                if (it->second.type == "intset") {
                    IntSetTraitParameters itp = std::get<IntSetTraitParameters>(it->second.details_);
                    int index = pickSetIndex(itp.probs, itp.set.size(), "int", rng);
                    t = itp.set[index];
                }
                if (it->second.type == "floatset") {
                    FloatSetTraitParameters itp = std::get<FloatSetTraitParameters>(it->second.details_);
                    int index = pickSetIndex(itp.probs, itp.set.size(), "float", rng);
                    t = itp.set[index];
                }
                Trait tr;
                tr.value = t;
                tr.depKey = it->second.depKey;
                tr.depValues = it->second.depValues;
                // todo check for invalid dep_values types here
                traits_[it->first] = tr;
            }
        }

        // Merges the other parent's traits into this gene: each trait is picked from either parent or
        // averaged for numeric types. Throws std::runtime_error when variant types mismatch.
        void mateTraits(const std::map<std::string, Trait> &t, RNG &rng) {
            for (std::map<std::string, Trait>::const_iterator it = t.begin(); it != t.end(); it++) {
                TraitType mine = traits_[it->first].value;
                TraitType yours = it->second.value;

                if (mine.index() != yours.index()) {
                    // std::cout << "t1:" << mine << " t2:" << yours << "\n";
                    throw std::runtime_error("Types of traits doesn't match");
                }

                {
                    if (rng.randFloat() < 0.5)  // pick either one
                    {
                        traits_[it->first].value = (rng.randFloat() < 0.5) ? mine : yours;
                    } else {
                        // try to average
                        if (std::holds_alternative<int>(mine)) {
                            int m1 = std::get<int>(mine);
                            int m2 = std::get<int>(yours);
                            traits_[it->first].value = (m1 + m2) / 2;
                        }

                        if (std::holds_alternative<double>(mine)) {
                            double m1 = std::get<double>(mine);
                            double m2 = std::get<double>(yours);
                            traits_[it->first].value = (m1 + m2) / 2.0;
                        }

                        if (std::holds_alternative<std::string>(mine)) {
                            // strings are always either-or
                            traits_[it->first].value = (rng.randFloat() < 0.5) ? mine : yours;
                        }

                        if (std::holds_alternative<IntSetElement>(mine)) {
                            // int sets are always either-or
                            traits_[it->first].value = (rng.randFloat() < 0.5) ? mine : yours;
                        }

                        if (std::holds_alternative<FloatSetElement>(mine)) {
                            // float sets are always either-or
                            traits_[it->first].value = (rng.randFloat() < 0.5) ? mine : yours;
                        }
                    }
                }
            }
        }

        // Mutates traits honoring dependency gating (depKey/depValues) and per-trait mutation probabilities.
        // Returns true when at least one trait changed.
        bool mutateTraits(const std::map<std::string, TraitParameters> &tp, RNG &rng) {
            bool didMutate = false;
            for (std::map<std::string, TraitParameters>::const_iterator it = tp.begin(); it != tp.end(); it++) {
                // only mutate the trait if it's enabled
                bool doit = false;
                if (it->second.depKey != "") {
                    // there is such trait..
                    if (traits_.count(it->second.depKey) != 0) {
                        // and it matches any of the right values?
                        for (int ix = 0; ix < it->second.depValues.size(); ix++) {
                            if (traits_[it->second.depKey].value == it->second.depValues[ix]) {
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
                    if (rng.randFloat() < it->second.mutationProb_) {
                        if (it->second.type == "int") {
                            IntTraitParameters itp = std::get<IntTraitParameters>(it->second.details_);

                            // determine type of mutation - modify or replace, according to parameters
                            if (rng.randFloat() < itp.mutReplaceProb) {
                                // replace
                                int val = std::get<int>(traits_[it->first].value);
                                int cur = val;
                                while (cur == val) {
                                    val = rng.randInt(itp.min, itp.max);
                                }
                                traits_[it->first].value = val;
                                didMutate = true;
                            } else {
                                // modify
                                int val = std::get<int>(traits_[it->first].value);
                                int cur = val;
                                while (cur == val) {
                                    val += rng.randInt(-itp.mutPower, itp.mutPower);
                                    clamp(val, itp.min, itp.max);
                                }
                                traits_[it->first].value = val;
                                didMutate = true;
                            }
                        } else if (it->second.type == "float") {
                            FloatTraitParameters itp = std::get<FloatTraitParameters>(it->second.details_);

                            // determine type of mutation - modify or replace, according to parameters
                            if (rng.randFloat() < itp.mutReplaceProb) {
                                // replace
                                double val = std::get<double>(traits_[it->first].value);
                                double cur = val;
                                while (cur == val) {
                                    val = rng.randFloat();
                                    scale(val, 0.0, 1.0, itp.min, itp.max);
                                }
                                traits_[it->first].value = val;
                                didMutate = true;
                            } else {
                                // modify
                                double val = std::get<double>(traits_[it->first].value);
                                double cur = val;
                                while (cur == val) {
                                    val += rng.randFloatSigned() * itp.mutPower;
                                    clamp(val, itp.min, itp.max);
                                }
                                traits_[it->first].value = val;
                                didMutate = true;
                            }

                        } else if (it->second.type == "str") {
                            StringTraitParameters itp = std::get<StringTraitParameters>(it->second.details_);
                            std::string cur = std::get<std::string>(traits_[it->first].value);
                            int index = pickSetIndex(itp.probs, itp.set.size(), "string", rng);

                            while (cur == itp.set[index]) {
                                index = pickSetIndex(itp.probs, itp.set.size(), "string", rng);
                            }
                            // now choose the new index from the set
                            traits_[it->first].value = itp.set[index];
                            didMutate = true;
                        } else if (it->second.type == "intset") {
                            IntSetTraitParameters itp = std::get<IntSetTraitParameters>(it->second.details_);
                            IntSetElement cur = std::get<IntSetElement>(traits_[it->first].value);
                            int index = pickSetIndex(itp.probs, itp.set.size(), "int", rng);

                            while (cur.value == itp.set[index].value) {
                                index = pickSetIndex(itp.probs, itp.set.size(), "int", rng);
                            }
                            // now choose the new index from the set
                            traits_[it->first].value = itp.set[index];
                            didMutate = true;
                        } else if (it->second.type == "floatset") {
                            FloatSetTraitParameters itp = std::get<FloatSetTraitParameters>(it->second.details_);
                            FloatSetElement cur = std::get<FloatSetElement>(traits_[it->first].value);
                            int index = pickSetIndex(itp.probs, itp.set.size(), "float", rng);

                            while (cur.value == itp.set[index].value) {
                                index = pickSetIndex(itp.probs, itp.set.size(), "float", rng);
                            }
                            // now choose the new index from the set
                            traits_[it->first].value = itp.set[index];
                            didMutate = true;
                        }
                    }
                }
            }

            return didMutate;
        }
        // Per-trait absolute distances to another gene's traits (0/1 for strings, magnitude for the rest),
        // gated the same way as mutation. Throws std::runtime_error when variant types mismatch.
        std::map<std::string, double> getTraitDistances(const std::map<std::string, Trait> &other) {
            std::map<std::string, double> dist;
            for (std::map<std::string, Trait>::const_iterator it = other.begin(); it != other.end(); it++) {
                TraitType mine = traits_[it->first].value;
                TraitType yours = it->second.value;

                if (mine.index() != yours.index()) {
                    throw std::runtime_error("Types of traits don't match");
                }

                // only do it if the trait if it's enabled
                // todo: not sure about the distance, think more about it
                bool doit = false;
                if (it->second.depKey != "") {
                    // there is such trait..
                    if (traits_.count(it->second.depKey) != 0) {
                        // and it has the right value? also the other genome has to have the trait turned on
                        for (int ix = 0; ix < it->second.depValues.size(); ix++) {
                            if ((traits_[it->second.depKey].value == it->second.depValues[ix]) &&
                                (other.at(it->second.depKey).value == it->second.depValues[ix])) {
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
                    }
                    if (std::holds_alternative<double>(mine)) {
                        // distance between floats - calculate directly
                        dist[it->first] = std::abs(std::get<double>(mine) - std::get<double>(yours));
                    }
                    if (std::holds_alternative<std::string>(mine)) {
                        // distance between strings - matching is 0, non-matching is 1
                        if (std::get<std::string>(mine) == std::get<std::string>(yours)) {
                            dist[it->first] = 0.0;
                        } else {
                            dist[it->first] = 1.0;
                        }
                    }
                    if (std::holds_alternative<IntSetElement>(mine)) {
                        // distance between ints - calculate directly
                        dist[it->first] = std::abs((std::get<IntSetElement>(mine)).value - (std::get<IntSetElement>(yours)).value);
                    }
                    if (std::holds_alternative<FloatSetElement>(mine)) {
                        // distance between floats - calculate directly
                        dist[it->first] = std::abs((std::get<FloatSetElement>(mine)).value - (std::get<FloatSetElement>(yours)).value);
                    }
                }
            }

            return dist;
        }
    };

    //////////////////////////////////
    // A connection gene: the innovation-numbered, weighted edge between two neuron IDs. Sorted and
    // compared by innovation ID so genomes stay aligned during mating.
    //////////////////////////////////
    class LinkGene : public Gene {
        /////////////////////
        // Members (fixed at creation except the weight/traits, which evolve)
        /////////////////////

       public:
        // The IDs of the neurons that this link connects (source -> target).
        int fromNeuronID_, toNeuronID_;

        // The link's innovation ID (alignment key; see src/Innovation.h).
        int innovationID_;

        // The weight of the connection (mutated during evolution).
        double weight_;

        // Whether the link is recurrent (target is at an equal or earlier depth).
        bool isRecurrent_;

       public:
        double getWeight() const { return weight_; }

        void setWeight(const double weight) { weight_ = weight; }

        ////////////////
        // Constructors
        ////////////////
        LinkGene() {
            fromNeuronID_ = 0;
            toNeuronID_ = 0;
            innovationID_ = 0;
            weight_ = 0;
            isRecurrent_ = false;
        }

        LinkGene(int inID, int outID, int innovID, double wgt, bool recurrent = false) {
            fromNeuronID_ = inID;
            toNeuronID_ = outID;
            innovationID_ = innovID;

            weight_ = wgt;
            isRecurrent_ = recurrent;
        }

        // assigment operator
        LinkGene &operator=(const LinkGene &g) {
            if (this != &g) {
                fromNeuronID_ = g.fromNeuronID_;
                toNeuronID_ = g.toNeuronID_;
                weight_ = g.weight_;
                isRecurrent_ = g.isRecurrent_;
                innovationID_ = g.innovationID_;
                traits_ = g.traits_;
            }

            return *this;
        }

        //////////////
        // Methods
        //////////////

        // Access to static (const) variables
        int fromNeuronID() const { return fromNeuronID_; }

        int toNeuronID() const { return toNeuronID_; }

        int innovationID() const { return innovationID_; }

        bool isRecurrent() const { return isRecurrent_; }

        bool isLoopedRecurrent() const { return fromNeuronID_ == toNeuronID_; }

        // overload '<', '>', '!=' and '==' used for sorting and comparison (we use the innovation ID as the criteria)
        friend bool operator<(const LinkGene &lhs, const LinkGene &rhs) { return (lhs.innovationID_ < rhs.innovationID_); }

        friend bool operator>(const LinkGene &lhs, const LinkGene &rhs) { return (lhs.innovationID_ > rhs.innovationID_); }

        friend bool operator!=(const LinkGene &lhs, const LinkGene &rhs) { return (lhs.innovationID_ != rhs.innovationID_); }

        friend bool operator==(const LinkGene &lhs, const LinkGene &rhs) { return (lhs.innovationID_ == rhs.innovationID_); }
    };

    ////////////////////////////////////
    // A neuron gene: one node of the genotype with its activation parameters. Input/bias neurons only
    // use the identity fields; hidden/output neurons additionally evolve A/B/time-constant/bias traits.
    ////////////////////////////////////
    class NeuronGene : public Gene {
        /////////////////////
        // Members (identity is fixed at creation; activation fields evolve)
        /////////////////////

       public:
        // Unique identification number of the neuron.
        int id_;

        // Role of the neuron in the network.
        NeuronType type_;

       public:
        // Display coordinates, safe to access directly.
        int x, y;
        // Depth of the neuron within the network (0 = input side).
        double splitY_;

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
        double a_, b_;

        // Time constant value used when the neuron is activating in leaky integrator mode
        double timeConstant_;

        // Bias value used when the neuron is activating in leaky integrator mode
        double bias_;

        // The type of activation function the neuron has
        ActivationFunction actFunction_;

        ////////////////
        // Constructors
        ////////////////
        NeuronGene() {}

        /*friend bool operator!=(const NeuronGene &a_lhs, const NeuronGene &a_rhs)
        {
            return (a_lhs.m_ID != a_rhs.m_ID);
        }*/

        friend bool operator==(const NeuronGene &lhs, const NeuronGene &rhs) {
            return (lhs.id_ == rhs.id_) && (lhs.type_ == rhs.type_)
                //(a_lhs.m_SplitY == a_rhs.m_SplitY) &&
                //(a_lhs.m_A == a_rhs.m_A) &&
                //(a_lhs.m_B == a_rhs.m_B) &&
                //(a_lhs.m_TimeConstant == a_rhs.m_TimeConstant) &&
                //(a_lhs.m_Bias == a_rhs.m_Bias) &&
                //(a_lhs.m_ActFunction == a_rhs.m_ActFunction)
                ;
        }

        NeuronGene(NeuronType type, int id, double splity) {
            id_ = id;
            type_ = type;
            splitY_ = splity;

            // Initialize the node specific parameters
            a_ = 0.0f;
            b_ = 0.0f;
            timeConstant_ = 0.0f;
            bias_ = 0.0f;
            actFunction_ = UNSIGNED_SIGMOID;

            x = 0;
            y = 0;
        }

        // assigment operator
        NeuronGene &operator=(const NeuronGene &g) {
            if (this != &g) {
                id_ = g.id_;
                type_ = g.type_;
                splitY_ = g.splitY_;

                // maybe inputs don't need that
                if ((type_ != NeuronType::INPUT) && (type_ != NeuronType::BIAS)) {
                    x = g.x;
                    y = g.y;
                    a_ = g.a_;
                    b_ = g.b_;
                    timeConstant_ = g.timeConstant_;
                    bias_ = g.bias_;
                    actFunction_ = g.actFunction_;
                    traits_ = g.traits_;
                }
            }

            return *this;
        }

        //////////////
        // Methods
        //////////////

        // Accessing static (const) variables
        int id() const { return id_; }

        NeuronType type() const { return type_; }

        double splitY() const { return splitY_; }

        // Initializing
        void init(double a, double b, double timeConstant, double bias, ActivationFunction actFunc) {
            a_ = a;
            b_ = b;
            timeConstant_ = timeConstant;
            bias_ = bias;
            actFunction_ = actFunc;
        }
    };

}  // namespace NEAT
