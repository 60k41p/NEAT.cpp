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
 * File:        Innovation.h
 * Description: Historical-marking database for NEAT speciation and mating alignment. Each structural
 *              mutation (new link / new neuron) registers an Innovation; genomes created by splitting the
 *              same link share its neuron ID, so identical structures stay comparable forever and
 *              compatibility distances remain meaningful across generations.
 *
 * References: Stanley & Miikkulainen, "Evolving Neural Networks through Augmenting Topologies" (2002),
 *             Section 3 (historical markings, innovation numbers); see
 *             references/Evolving Neural Networks through Augmenting Topologies.pdf.md. Intra-repo users:
 *             src/Genome.h, src/Genome.cpp, src/Population.h, tests/TestInnovation.cpp.
 */

#pragma once

#include <cmath>
#include <fstream>
#include <vector>

#include "Genes.h"
#include "Genome.h"

namespace NEAT {

    ////////////////////////////////////////////////
    // Structural mutation kinds tracked by the database.
    ////////////////////////////////////////////////
    enum InnovationType { NEW_NEURON, NEW_LINK };

    //////////////////////////////////////////////
    // One historical marking: what structural change happened, between which neurons, and the IDs
    // assigned to it. All fields are fixed at creation and read-only afterwards.
    //////////////////////////////////////////////
    class Innovation {
        /////////////////////
        // Members (immutable after construction)
        /////////////////////

       private:
        // Database-wide innovation number.
        int id_;

        // Kind of structural change.
        InnovationType innovType_;

        // For NEW_LINK: the connected neuron IDs. For NEW_NEURON: the split link's endpoints.
        int fromNeuronID_, toNeuronID_;
        // For NEW_NEURON: the ID assigned to the created neuron.
        int neuronID_;
        // For NEW_NEURON: the role assigned to the created neuron.
        NeuronType neuronType_;

       public:
        ////////////////////////////
        // Constructors
        ////////////////////////////
        Innovation(int id, InnovationType innovType, int from, int to, NeuronType nType, int nid) {
            id_ = id;
            innovType_ = innovType;
            fromNeuronID_ = from;
            toNeuronID_ = to;
            neuronType_ = nType;
            neuronID_ = nid;
        }

        Innovation() {
            id_ = 0;
            fromNeuronID_ = 0;
            toNeuronID_ = 0;
            neuronID_ = 0;
        }

        ////////////////////////////
        // Destructor
        ////////////////////////////

        ////////////////////////////
        // Methods
        ////////////////////////////

        // Accessors (all read-only; the database assigns the values at creation).
        int id() const { return id_; }
        InnovationType innovType() const { return innovType_; }
        int fromNeuronID() const { return fromNeuronID_; }
        int toNeuronID() const { return toNeuronID_; }
        int neuronID() const { return neuronID_; }
        NeuronType getNeuronType() const { return neuronType_; }
    };

    // forward
    class Genome;

    ////////////////////////////////////////////////////////
    // The population-wide registry of structural innovations plus the next-ID counters. Query it before
    // creating structure (checkInnovation/findNeuronID) and register through it (addLinkInnovation/
    // addNeuronInnovation) so parallel lineages share numbering.
    ////////////////////////////////////////////////////////
    class InnovationDatabase {
       private:
        /////////////////////
        // Next free IDs (bumped by the add*() calls)
        /////////////////////

        int nextNeuronID_;
        int nextInnovationNum_;

       public:
        ////////////////////////////
        // Constructors
        ////////////////////////////
        // All registered innovations, in registration order.
        std::vector<Innovation> innovations_;
        // Creates an empty database.
        InnovationDatabase();

        // Creates an empty database resuming after the given innovation/neuron IDs (load path).
        InnovationDatabase(int lastInnovationNum, int lastNeuronID);

        ////////////////////////////
        // Methods
        ////////////////////////////

        // Resets to empty, resuming after the given IDs.
        void init(int lastInnovationNum, int lastNeuronID);

        // Rebuilds the registry from one genome's genes.
        void init(const Genome &genome);

        // Loads from an open stream positioned at an InnovationDatabaseStart marker.
        void init(std::ifstream &file);

        // Innovation number for the (in, out, type) change, or -1 when unseen. For NEW_LINK, in/out are
        // the connected neurons; for NEW_NEURON they are the split link's endpoints.
        int checkInnovation(int in, int out, InnovationType type) const;
        // Same query restricted to the most recent matching entry.
        int checkLastInnovation(int in, int out, InnovationType type) const;

        // Positions of every matching innovation in innovations_.
        std::vector<int> checkAllInnovations(int in, int out, InnovationType type) const;

        // Neuron ID created by splitting (in, out), or -1 when no such split is registered.
        int findNeuronID(int in, int out) const;
        // Same query restricted to the most recent matching split.
        int findLastNeuronID(int in, int out) const;

        // Registers a link innovation, returning its new innovation number.
        int addLinkInnovation(int in, int out);

        // Registers a neuron innovation for the split of (in, out), returning the new neuron ID.
        int addNeuronInnovation(int in, int out, NeuronType type);

        // Drops all entries (counters are preserved).
        void flush();

        // Copies out the innovation at the given position.
        Innovation getInnovationByIndex(int index) const { return innovations_[static_cast<size_t>(index)]; };

        // Appends the database to an open file (used by Population::save()).
        void save(FILE *file);
    };

}  // namespace NEAT
