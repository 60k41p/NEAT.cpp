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
#include <limits>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>

#include "Genes.h"
#include "Genome.h"
#include "Serialization.h"
#include "assert.h"

namespace NEAT {

    std::uint64_t InnovationDatabase::EndpointKey(int a_in, int a_out) {
        return (static_cast<std::uint64_t>(static_cast<std::uint32_t>(a_in)) << 32U) | static_cast<std::uint32_t>(a_out);
    }

    void InnovationDatabase::AppendToIndex(std::size_t a_index) const {
        const Innovation &innovation = m_Innovations.at(a_index);
        auto &index = (innovation.InnovType() == NEW_LINK) ? m_LinkInnovationIndex : m_NeuronInnovationIndex;
        index[EndpointKey(innovation.FromNeuronID(), innovation.ToNeuronID())].push_back(static_cast<int>(a_index));
    }

    void InnovationDatabase::RebuildIndex() const {
        m_LinkInnovationIndex.clear();
        m_NeuronInnovationIndex.clear();
        m_LinkInnovationIndex.reserve(m_Innovations.size());
        m_NeuronInnovationIndex.reserve(m_Innovations.size());
        for (std::size_t index = 0; index < m_Innovations.size(); ++index) AppendToIndex(index);
        m_IndexedInnovationCount = m_Innovations.size();
        m_IndexedInnovationData = m_Innovations.data();
    }

    void InnovationDatabase::EnsureIndex() const {
        if (m_IndexedInnovationCount != m_Innovations.size() || m_IndexedInnovationData != m_Innovations.data()) {
            RebuildIndex();
        }
    }

    // Creates an empty database
    InnovationDatabase::InnovationDatabase() {
        m_NextInnovationNum = 1;  // innovations start at 1
        m_NextNeuronID = 1;       // neuron IDs start at 1
        m_Innovations.clear();
    }

    // Creates an empty database but this time sets the next innov number and neuron ID
    InnovationDatabase::InnovationDatabase(int a_LastInnovationNum, int a_LastNeuronID) {
        if (a_LastInnovationNum < 0 || a_LastNeuronID < 0 || a_LastInnovationNum == std::numeric_limits<int>::max() ||
            a_LastNeuronID == std::numeric_limits<int>::max()) {
            throw std::invalid_argument("Innovation counters cannot be negative");
        }

        m_NextInnovationNum = a_LastInnovationNum + 1;
        m_NextNeuronID = a_LastNeuronID + 1;
        m_Innovations.clear();
    }

    // Initializes an empty database
    void InnovationDatabase::Init(int a_LastInnovationNum, int a_LastNeuronID) {
        if (a_LastInnovationNum < 0 || a_LastNeuronID < 0 || a_LastInnovationNum == std::numeric_limits<int>::max() ||
            a_LastNeuronID == std::numeric_limits<int>::max()) {
            throw std::invalid_argument("Innovation counters cannot be negative");
        }
        Flush();

        m_NextNeuronID = a_LastNeuronID + 1;
        m_NextInnovationNum = a_LastInnovationNum + 1;
    }

    // Initializes a database from a given genome
    void InnovationDatabase::Init(const Genome &a_Genome) {
        Flush();
        for (unsigned int i = 0; i < a_Genome.NumLinks(); i++) {
            Innovation t_innov(a_Genome.GetLinkByIndex(i).InnovationID(), NEW_LINK, a_Genome.GetLinkByIndex(i).FromNeuronID(),
                               a_Genome.GetLinkByIndex(i).ToNeuronID(), NONE, -1);
            m_Innovations.emplace_back(t_innov);
        }
        RebuildIndex();

        // GetLast* return the max used ID; the next free ID is one past that.
        const int last_neuron = a_Genome.GetLastNeuronID();
        const int last_innovation = a_Genome.GetLastInnovationID();
        if (last_neuron == std::numeric_limits<int>::max() || last_innovation == std::numeric_limits<int>::max()) {
            throw std::overflow_error("Genome has exhausted the innovation ID space");
        }
        m_NextNeuronID = last_neuron + 1;
        m_NextInnovationNum = last_innovation + 1;
    }

    void InnovationDatabase::Init(std::ifstream &a_DataFile) {
        Flush();
        m_NextInnovationNum = 0;
        m_NextNeuronID = 0;

        std::string t_str;

        // search for InnovationDatabaseStart
        while (a_DataFile >> t_str && t_str != "InnovationDatabaseStart") {
        }
        if (t_str != "InnovationDatabaseStart") throw std::runtime_error("InnovationDatabase::Init: missing start marker.");

        // Read the last innov numbers
        a_DataFile >> t_str;
        a_DataFile >> m_NextInnovationNum;
        a_DataFile >> t_str;
        a_DataFile >> m_NextNeuronID;
        if (!a_DataFile || m_NextInnovationNum < 1 || m_NextNeuronID < 1) {
            throw std::runtime_error("InnovationDatabase::Init: invalid counters.");
        }

        // Read the database until InnovationDatabaseEnd is encountered
        while (a_DataFile >> t_str) {
            if (t_str == "Innovation") {
                // Read in the innovation
                int t_id, t_from, t_to, t_innovtype, t_neurontype, t_nid;

                a_DataFile >> t_id;
                a_DataFile >> t_innovtype;
                a_DataFile >> t_from;
                a_DataFile >> t_to;
                a_DataFile >> t_neurontype;
                a_DataFile >> t_nid;
                if (!a_DataFile || (t_innovtype != NEW_NEURON && t_innovtype != NEW_LINK) || t_id < 1 || t_from < 1 || t_to < 1) {
                    throw std::runtime_error("InnovationDatabase::Init: invalid innovation.");
                }

                m_Innovations.emplace_back(
                    Innovation(t_id, static_cast<InnovationType>(t_innovtype), t_from, t_to, static_cast<NeuronType>(t_neurontype), t_nid));
            } else if (t_str == "InnovationDatabaseEnd") {
                std::string error;
                if (!ValidateInnovationState(&error)) throw std::runtime_error("InnovationDatabase::Init: " + error);
                RebuildIndex();
                return;
            }
        }
        throw std::runtime_error("InnovationDatabase::Init: missing end marker.");
    }

    // The file is assumed to be opened
    void InnovationDatabase::Save(FILE *a_file) {
        if (a_file == nullptr) throw std::invalid_argument("InnovationDatabase::Save: file is null.");
        fprintf(a_file, "InnovationDatabaseStart\n");
        fprintf(a_file, "NextInnovNum: %d\n", m_NextInnovationNum);
        fprintf(a_file, "NextNeuronID: %d\n", m_NextNeuronID);

        // Now save all innovations
        for (unsigned int i = 0; i < m_Innovations.size(); i++) {
            fprintf(a_file, "Innovation %d %d %d %d %d %d\n", m_Innovations[i].ID(), static_cast<int>(m_Innovations[i].InnovType()),
                    m_Innovations[i].FromNeuronID(), m_Innovations[i].ToNeuronID(), static_cast<int>(m_Innovations[i].GetNeuronType()),
                    m_Innovations[i].NeuronID());
        }
        fprintf(a_file, "InnovationDatabaseEnd\n\n");
    }

    // Checks the database if the innovation has already occured
    // Returns the innovation id if true or -1 if false
    // If it is a NEW_LINK innovation, in & out specify the neuron IDs being connected
    // If it is a NEW_NEURON innovation, in & out specify the connection that was split
    int InnovationDatabase::CheckInnovation(int a_In, int a_Out, InnovationType a_Type) const {
        if (a_In <= 0 || a_Out <= 0 || (a_Type != NEW_NEURON && a_Type != NEW_LINK)) throw std::invalid_argument("Invalid innovation query");

        // first match wins
        EnsureIndex();
        const auto &index = (a_Type == NEW_LINK) ? m_LinkInnovationIndex : m_NeuronInnovationIndex;
        const auto match = index.find(EndpointKey(a_In, a_Out));
        if (match == index.end() || match->second.empty()) return -1;
        return m_Innovations[static_cast<std::size_t>(match->second.front())].ID();
    }

    int InnovationDatabase::CheckLastInnovation(int a_In, int a_Out, InnovationType a_Type) const {
        if (a_In <= 0 || a_Out <= 0 || (a_Type != NEW_NEURON && a_Type != NEW_LINK)) throw std::invalid_argument("Invalid innovation query");

        // last match wins
        EnsureIndex();
        const auto &index = (a_Type == NEW_LINK) ? m_LinkInnovationIndex : m_NeuronInnovationIndex;
        const auto match = index.find(EndpointKey(a_In, a_Out));
        if (match == index.end() || match->second.empty()) return -1;
        return m_Innovations[static_cast<std::size_t>(match->second.back())].ID();
    }

    // returns a list of indexes in the database of identical innovations
    std::vector<int> InnovationDatabase::CheckAllInnovations(int a_In, int a_Out, InnovationType a_Type) const {
        if (a_In <= 0 || a_Out <= 0 || (a_Type != NEW_NEURON && a_Type != NEW_LINK)) throw std::invalid_argument("Invalid innovation query");

        EnsureIndex();
        const auto &index = (a_Type == NEW_LINK) ? m_LinkInnovationIndex : m_NeuronInnovationIndex;
        const auto match = index.find(EndpointKey(a_In, a_Out));
        if (match == index.end()) return std::vector<int>();
        return match->second;
    }

    // Returns the neuron ID given the in and out neurons
    // If not found, returns -1
    int InnovationDatabase::FindNeuronID(int a_In, int a_Out) const {
        if (a_In <= 0 || a_Out <= 0) throw std::invalid_argument("Invalid neuron innovation query");

        EnsureIndex();
        const auto match = m_NeuronInnovationIndex.find(EndpointKey(a_In, a_Out));
        if (match == m_NeuronInnovationIndex.end() || match->second.empty()) return -1;
        return m_Innovations[static_cast<std::size_t>(match->second.front())].NeuronID();
    }

    int InnovationDatabase::FindLastNeuronID(int a_In, int a_Out) const {
        if (a_In <= 0 || a_Out <= 0) throw std::invalid_argument("Invalid neuron innovation query");

        EnsureIndex();
        const auto match = m_NeuronInnovationIndex.find(EndpointKey(a_In, a_Out));
        if (match == m_NeuronInnovationIndex.end() || match->second.empty()) return -1;
        return m_Innovations[static_cast<std::size_t>(match->second.back())].NeuronID();
    }

    // Adds a new link innovation and returns its ID Increments the m_NextInnovationNum internally
    int InnovationDatabase::AddLinkInnovation(int a_In, int a_Out) {
        if (a_In <= 0 || a_Out <= 0) throw std::invalid_argument("Innovation endpoints must be positive");
        if (m_NextInnovationNum == std::numeric_limits<int>::max()) throw std::overflow_error("Innovation database has exhausted the innovation ID space");

        EnsureIndex();
        m_Innovations.emplace_back(Innovation(m_NextInnovationNum, NEW_LINK, a_In, a_Out, NONE, -1));
        AppendToIndex(m_Innovations.size() - 1);
        m_IndexedInnovationCount = m_Innovations.size();
        m_IndexedInnovationData = m_Innovations.data();
        m_NextInnovationNum++;

        return (m_NextInnovationNum - 1);
    }

    // Adds a new neuron innovation and returns the new neuron ID in and out specify the connection that was split type specifies the type of neuron Increments
    // the m_NextNeuronID and m_NextInnovationNum internally
    int InnovationDatabase::AddNeuronInnovation(int a_In, int a_Out, NeuronType a_NType) {
        if (a_In <= 0 || a_Out <= 0 || a_NType != HIDDEN) throw std::invalid_argument("Neuron innovations require positive endpoints and a hidden neuron");
        if (m_NextInnovationNum == std::numeric_limits<int>::max() || m_NextNeuronID == std::numeric_limits<int>::max())
            throw std::overflow_error("Innovation database has exhausted the innovation ID space");

        EnsureIndex();
        m_Innovations.emplace_back(Innovation(m_NextInnovationNum, NEW_NEURON, a_In, a_Out, a_NType, m_NextNeuronID));
        AppendToIndex(m_Innovations.size() - 1);
        m_IndexedInnovationCount = m_Innovations.size();
        m_IndexedInnovationData = m_Innovations.data();
        m_NextInnovationNum++;
        m_NextNeuronID++;

        return (m_NextNeuronID - 1);
    }

    // Clears all innovations in the database
    void InnovationDatabase::Flush() {
        m_Innovations.clear();
        m_LinkInnovationIndex.clear();
        m_NeuronInnovationIndex.clear();
        m_IndexedInnovationCount = 0;
        m_IndexedInnovationData = m_Innovations.data();
    }

    bool InnovationDatabase::ValidateInnovationState(std::string *error) const {
        std::set<int> innovation_ids;
        int maximum_innovation = 0;
        int maximum_neuron = 0;
        for (const Innovation &innovation : m_Innovations) {
            if (innovation.ID() <= 0 || !innovation_ids.insert(innovation.ID()).second) {
                if (error != nullptr) *error = "Innovation database IDs must be positive and unique";
                return false;
            }
            if (innovation.FromNeuronID() <= 0 || innovation.ToNeuronID() <= 0) {
                if (error != nullptr) *error = "Innovation database endpoints must be positive";
                return false;
            }
            if (innovation.InnovType() == NEW_NEURON) {
                if (innovation.NeuronID() <= 0 || innovation.GetNeuronType() != HIDDEN) {
                    if (error != nullptr) *error = "Neuron innovation data is invalid";
                    return false;
                }
                maximum_neuron = std::max(maximum_neuron, innovation.NeuronID());
            }
            maximum_innovation = std::max(maximum_innovation, innovation.ID());
        }
        if (m_NextInnovationNum <= 0 || m_NextNeuronID <= 0) {
            if (error != nullptr) *error = "Innovation counters must be positive";
            return false;
        }
        if (m_NextInnovationNum <= maximum_innovation || m_NextNeuronID <= maximum_neuron) {
            if (error != nullptr) *error = "Innovation counters would reuse an existing ID";
            return false;
        }
        return true;
    }

    std::string InnovationDatabase::Serialize() const {
        std::ostringstream output;
        Serialization::UseRoundTripPrecision(output);
        output << "InnovationDatabaseStart\n";
        output << "NextInnovNum: " << m_NextInnovationNum << "\n";
        output << "NextNeuronID: " << m_NextNeuronID << "\n";
        for (const auto &innovation : m_Innovations) {
            output << "Innovation " << innovation.ID() << ' ' << static_cast<int>(innovation.InnovType()) << ' ' << innovation.FromNeuronID() << ' '
                   << innovation.ToNeuronID() << ' ' << static_cast<int>(innovation.GetNeuronType()) << ' ' << innovation.NeuronID() << '\n';
        }
        output << "InnovationDatabaseEnd\n";
        return output.str();
    }

    InnovationDatabase InnovationDatabase::Deserialize(const std::string &data) {
        InnovationDatabase database;
        std::istringstream input(data);
        std::string token;
        while (input >> token && token != "InnovationDatabaseStart") {
        }
        if (token != "InnovationDatabaseStart") throw std::runtime_error("InnovationDatabase: missing start marker.");
        input >> token;
        if (token != "NextInnovNum:") throw std::runtime_error("InnovationDatabase: missing innovation counter.");
        input >> database.m_NextInnovationNum;
        input >> token;
        if (token != "NextNeuronID:") throw std::runtime_error("InnovationDatabase: missing neuron counter.");
        input >> database.m_NextNeuronID;
        while (input >> token && token != "InnovationDatabaseEnd") {
            if (token != "Innovation") throw std::runtime_error("InnovationDatabase: missing Innovation marker.");
            int id, type, from, to, neuron_type, nid;
            input >> id >> type >> from >> to >> neuron_type >> nid;
            if (!input) throw std::runtime_error("InnovationDatabase: malformed entry.");
            database.m_Innovations.emplace_back(Innovation(id, static_cast<InnovationType>(type), from, to, static_cast<NeuronType>(neuron_type), nid));
        }
        if (token != "InnovationDatabaseEnd") throw std::runtime_error("InnovationDatabase: missing end marker.");
        database.RebuildIndex();
        std::string error;
        if (!database.ValidateInnovationState(&error)) throw std::runtime_error("InnovationDatabase: " + error);
        return database;
    }

}  // namespace NEAT
