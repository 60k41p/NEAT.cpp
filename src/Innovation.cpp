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
 * File:        Innovation.cpp
 * Description: Implementation for the Innovation and InnovationDatabase classes.
 */

#include "Innovation.h"

#include <fstream>
#include <string>

#include "Genes.h"
#include "Genome.h"
#include "assert.h"

namespace NEAT {

    // Creates an empty database
    InnovationDatabase::InnovationDatabase() {
        nextInnovationNum_ = 1;  // innovations start at 1
        nextNeuronID_ = 1;       // neuron IDs start at 1
        innovations_.clear();
    }

    // Creates an empty database but this time sets the next innov number and neuron ID
    InnovationDatabase::InnovationDatabase(int lastInnovationNum, int lastNeuronID) {
        ASSERT((lastInnovationNum > 0) && (lastNeuronID > 0));

        nextInnovationNum_ = lastInnovationNum;
        nextNeuronID_ = lastNeuronID;
        innovations_.clear();
    }

    // Initializes an empty database
    void InnovationDatabase::init(int lastInnovationNum, int lastNeuronID) {
        flush();

        nextNeuronID_ = lastNeuronID;
        nextInnovationNum_ = lastInnovationNum;
    }

    // Initializes a database from a given genome
    void InnovationDatabase::init(const Genome &genome) {
        innovations_.clear();
        for (unsigned int i = 0; i < genome.numLinks(); i++) {
            Innovation innov(genome.getLinkByIndex(i).innovationID(), NEW_LINK, genome.getLinkByIndex(i).fromNeuronID(), genome.getLinkByIndex(i).toNeuronID(),
                             NONE, -1);
            innovations_.emplace_back(innov);
        }

        nextNeuronID_ = genome.getLastNeuronID();
        nextInnovationNum_ = genome.getLastInnovationID();
    }

    void InnovationDatabase::init(std::ifstream &dataFile) {
        innovations_.clear();
        nextInnovationNum_ = 0;
        nextNeuronID_ = 0;

        std::string str;

        // search for InnovationDatabaseStart (with EOF guard: failed extraction
        // leaves t_str unchanged, so the loop would otherwise never terminate)
        do {
            dataFile >> str;
            if (dataFile.eof()) {
                throw std::runtime_error("Innovation database file error: InnovationDatabaseStart not found!");
            }
        } while (str != "InnovationDatabaseStart");

        // Read the last innov numbers
        dataFile >> str;
        dataFile >> nextInnovationNum_;
        dataFile >> str;
        dataFile >> nextNeuronID_;

        // Read the database until InnovationDatabaseEnd is encountered
        do {
            dataFile >> str;
            if (dataFile.eof()) {
                throw std::runtime_error("Innovation database file error: InnovationDatabaseEnd not found!");
            }

            if (str == "Innovation") {
                // Read in the innovation
                int id, from, to, innovtype, neurontype, nid;

                dataFile >> id;
                dataFile >> innovtype;
                dataFile >> from;
                dataFile >> to;
                dataFile >> neurontype;
                dataFile >> nid;

                innovations_.emplace_back(Innovation(id, static_cast<InnovationType>(innovtype), from, to, static_cast<NeuronType>(neurontype), nid));
            }

        } while (str != "InnovationDatabaseEnd");
    }

    // The file is assumed to be opened
    void InnovationDatabase::save(FILE *file) {
        fprintf(file, "InnovationDatabaseStart\n");
        fprintf(file, "NextInnovNum: %d\n", nextInnovationNum_);
        fprintf(file, "NextNeuronID: %d\n", nextNeuronID_);

        // Now save all innovations
        for (unsigned int i = 0; i < innovations_.size(); i++) {
            fprintf(file, "Innovation %d %d %d %d %d %d\n", innovations_[i].id(), static_cast<int>(innovations_[i].innovType()), innovations_[i].fromNeuronID(),
                    innovations_[i].toNeuronID(), static_cast<int>(innovations_[i].getNeuronType()), innovations_[i].neuronID());
        }
        fprintf(file, "InnovationDatabaseEnd\n\n");
    }

    // Checks the database if the innovation has already occured
    // Returns the innovation id if true or -1 if false
    // If it is a NEW_LINK innovation, in & out specify the neuron IDs being connected
    // If it is a NEW_NEURON innovation, in & out specify the connection that was split
    int InnovationDatabase::checkInnovation(int in, int out, InnovationType type) const {
        ASSERT((in > 0) && (out > 0));
        ASSERT((type == NEW_NEURON) || (type == NEW_LINK));

        // search the list for a match
        for (unsigned int i = 0; i < innovations_.size(); i++) {
            if ((innovations_[i].fromNeuronID() == in) && (innovations_[i].toNeuronID() == out) && (innovations_[i].innovType() == type)) {
                // match found?
                return innovations_[i].id();
            }
        }

        // not found
        return -1;
    }

    int InnovationDatabase::checkLastInnovation(int in, int out, InnovationType type) const {
        ASSERT((in > 0) && (out > 0));
        ASSERT((type == NEW_NEURON) || (type == NEW_LINK));
        int id = -1;

        // search the list for a match
        for (unsigned int i = 0; i < innovations_.size(); i++) {
            if ((innovations_[i].fromNeuronID() == in) && (innovations_[i].toNeuronID() == out) && (innovations_[i].innovType() == type)) {
                // match found?
                id = innovations_[i].id();
            }
        }

        return id;
    }

    // returns a list of indexes in the database of identical innovations
    std::vector<int> InnovationDatabase::checkAllInnovations(int in, int out, InnovationType type) const {
        ASSERT((in > 0) && (out > 0));
        ASSERT((type == NEW_NEURON) || (type == NEW_LINK));

        std::vector<int> indexs;
        indexs.clear();

        // search the list for a match
        for (unsigned int i = 0; i < innovations_.size(); i++) {
            if ((innovations_[i].fromNeuronID() == in) && (innovations_[i].toNeuronID() == out) && (innovations_[i].innovType() == type)) {
                // match found?
                indexs.emplace_back(i);
            }
        }

        return indexs;
    }

    // Returns the neuron ID given the in and out neurons
    // If not found, returns -1
    int InnovationDatabase::findNeuronID(int in, int out) const {
        ASSERT((in > 0) && (out > 0));

        // search the list for a match
        for (unsigned int i = 0; i < innovations_.size(); i++) {
            if ((innovations_[i].fromNeuronID() == in) && (innovations_[i].toNeuronID() == out) && (innovations_[i].innovType() == NEW_NEURON)) {
                // match found?
                return innovations_[i].neuronID();
            }
        }

        // Not found
        return -1;
    }

    int InnovationDatabase::findLastNeuronID(int in, int out) const {
        ASSERT((in > 0) && (out > 0));
        int id = -1;

        // search the list for a match
        for (unsigned int i = 0; i < innovations_.size(); i++) {
            if ((innovations_[i].fromNeuronID() == in) && (innovations_[i].toNeuronID() == out) && (innovations_[i].innovType() == NEW_NEURON)) {
                // match found?
                id = innovations_[i].neuronID();
            }
        }

        return id;
    }

    // Adds a new link innovation and returns its ID Increments the m_NextInnovationNum internally
    int InnovationDatabase::addLinkInnovation(int in, int out) {
        ASSERT((in > 0) && (out > 0));

        innovations_.emplace_back(Innovation(nextInnovationNum_, NEW_LINK, in, out, NONE, -1));
        nextInnovationNum_++;

        return (nextInnovationNum_ - 1);
    }

    // Adds a new neuron innovation and returns the new neuron ID in and out specify the connection that was split type specifies the type of neuron Increments
    // the m_NextNeuronID and m_NextInnovationNum internally
    int InnovationDatabase::addNeuronInnovation(int in, int out, NeuronType nType) {
        ASSERT((in > 0) && (out > 0));
        ASSERT(!((nType == INPUT) || (nType == BIAS) || (nType == OUTPUT)));

        innovations_.emplace_back(Innovation(nextInnovationNum_, NEW_NEURON, in, out, nType, nextNeuronID_));
        nextInnovationNum_++;
        nextNeuronID_++;

        return (nextNeuronID_ - 1);
    }

    // Clears all innovations in the database
    void InnovationDatabase::flush() { innovations_.clear(); }

}  // namespace NEAT
