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
 * File:        Population.h
 * Description: Definition for the Population class.
 */

#pragma once

#include <float.h>

#include <string>
#include <vector>

#include "Genes.h"
#include "Genome.h"
#include "Innovation.h"
#include "Parameters.h"
#include "PhenotypeBehavior.h"
#include "Random.h"
#include "Species.h"
#include "Types.h"

namespace NEAT {

    //////////////////////////////////////////////
    // The Population class
    //////////////////////////////////////////////

    enum SearchMode { COMPLEXIFYING, SIMPLIFYING, BLENDED };

    class Species;

    class Population {
        /////////////////////
        // Members
        /////////////////////

       private:
        // The innovation database
        InnovationDatabase m_InnovationDatabase;

        // next genome ID
        unsigned int m_NextGenomeID = 0;

        // next species ID
        unsigned int m_NextSpeciesID = 0;

        ////////////////////////////
        // Phased searching members

        // The current mode of search
        SearchMode m_SearchMode = BLENDED;

        // The current Mean Population Complexity
        Real m_CurrentMPC = 0.0;

        // The MPC from the previous generation (for comparison)
        Real m_OldMPC = 0.0;

        // The base MPC (for switching between complexifying/simplifying phase)
        Real m_BaseMPC = 0.0;

        // Separates the population into species based on compatibility distance
        void Speciate();

        // Adjusts each species's fitness
        void AdjustFitness();

        // Calculates how many offspring each genome should have
        void CountOffspring();

        // Empties all species
        void ResetSpecies();

        // Updates the species
        void UpdateSpecies();

        // Calculates the current mean population complexity
        void CalculateMPC();

        // best fitness ever achieved
        Real m_BestFitnessEver = std::numeric_limits<Real>::lowest();

        // Keep a local copy of the best ever genome found in the run
        Genome m_BestGenome;
        Genome m_BestGenomeEver;

        // Number of generations since the best fitness changed
        unsigned int m_GensSinceBestFitnessLastChanged = 0;

        // Number of evaluations since the best fitness changed
        unsigned int m_EvalsSinceBestFitnessLastChanged = 0;

        // How many generations passed until the last change of MPC
        unsigned int m_GensSinceMPCLastChanged = 0;

        // The initial list of genomes
        std::vector<Genome> m_Genomes;

       public:
        // The archive
        std::vector<Genome> m_GenomeArchive;

        // Random number generator
        RNG m_RNG;

        // Evolution parameters
        Parameters m_Parameters;

        // Current generation
        unsigned int m_Generation = 0;

        // The list of species
        std::vector<Species> m_Species;

        int m_ID = 0;

        ////////////////////////////
        // Constructors
        ////////////////////////////

        // Initializes a population from a seed genome G. Then it initializes all weights
        // To small numbers between -R and R.
        // The population size is determined by GlobalParameters.PopulationSize
        Population(const Genome &a_G, const Parameters &a_Parameters, bool a_RandomizeWeights, Real a_RandomRange, int a_RNG_seed);

        // Loads a population from a file.
        Population(const std::string a_FileName);

        Population() {};

        ////////////////////////////
        // Destructor
        ////////////////////////////

        // TODO: move all header code into the source file,
        // make as much private members as possible

        ////////////////////////////
        // Methods
        ////////////////////////////

        // Access
        SearchMode GetSearchMode() const { return m_SearchMode; }
        Real GetCurrentMPC() const { return m_CurrentMPC; }
        Real GetBaseMPC() const { return m_BaseMPC; }

        unsigned int NumGenomes() const {
            unsigned int num = 0;
            for (unsigned int i = 0; i < m_Species.size(); i++) {
                num += m_Species[i].m_Individuals.size();
            }
            return num;
        }

        unsigned int GetGeneration() const { return m_Generation; }
        Real GetBestFitnessEver() const { return m_BestFitnessEver; }
        Genome GetBestGenome() const {
            if (m_Species.empty()) throw std::runtime_error("Population::GetBestGenome: population is empty.");

            Real best = std::numeric_limits<Real>::lowest();
            int idx_species = 0;
            int idx_genome = 0;
            bool found = false;
            for (unsigned int i = 0; i < m_Species.size(); i++) {
                for (unsigned int j = 0; j < m_Species[i].m_Individuals.size(); j++) {
                    const Genome &genome = m_Species[i].m_Individuals[j];
                    if (!genome.IsEvaluated() || !std::isfinite(genome.GetFitness())) {
                        continue;
                    }
                    if (!found || genome.GetFitness() > best) {
                        best = genome.GetFitness();
                        idx_species = i;
                        idx_genome = j;
                        found = true;
                    }
                }
            }

            if (!found) {
                for (const auto &species : m_Species) {
                    if (!species.m_Individuals.empty()) return species.m_Individuals.front();
                }
                throw std::runtime_error("Population::GetBestGenome: population has no genomes.");
            }
            return m_Species[idx_species].m_Individuals[idx_genome];
        }

        unsigned int GetStagnation() const { return m_GensSinceBestFitnessLastChanged; }
        unsigned int GetMPCStagnation() const { return m_GensSinceMPCLastChanged; }

        unsigned int GetNextGenomeID() const { return m_NextGenomeID; }
        unsigned int GetNextSpeciesID() const { return m_NextSpeciesID; }
        void IncrementNextGenomeID() {
            if (m_NextGenomeID == static_cast<unsigned int>(std::numeric_limits<int>::max())) throw std::overflow_error("Genome ID space is exhausted");
            ++m_NextGenomeID;
        }
        void IncrementNextSpeciesID() {
            if (m_NextSpeciesID == static_cast<unsigned int>(std::numeric_limits<int>::max())) throw std::overflow_error("Species ID space is exhausted");
            ++m_NextSpeciesID;
        }

        // Make sure no same genome IDs exist in the population
        void SameGenomeIDCheck() {
            // count how much each ID found has occured
            std::map<int, int> ids;
            for (unsigned int i = 0; i < m_Species.size(); i++) {
                for (unsigned int j = 0; j < m_Species[i].m_Individuals.size(); j++) {
                    ids[m_Species[i].m_Individuals[j].GetID()] = 0;
                }
            }
            for (unsigned int i = 0; i < m_Species.size(); i++) {
                for (unsigned int j = 0; j < m_Species[i].m_Individuals.size(); j++) {
                    ids[m_Species[i].m_Individuals[j].GetID()] += 1;
                }
            }

            for (auto it = ids.begin(); it != ids.end(); it++) {
                if (it->second > 1) {
                    throw std::runtime_error("Genome ID " + std::to_string(it->first) + " appears " + std::to_string(it->second) +
                                             " times in the population\n");
                }
            }
        }

        Genome &AccessGenomeByIndex(int const a_idx);
        Genome &AccessGenomeByID(int const a_id);

        InnovationDatabase &AccessInnovationDatabase() { return m_InnovationDatabase; }

        // Sorts each species's genomes by fitness
        void Sort();

        // Performs one generation and reproduces the genomes
        void Epoch();

        // Saves the whole population to a file
        void Save(const char *a_FileName);

        // Population checkpointing (text format with parameters, innovation
        // database, RNG state and all genomes).
        void SaveState(const char *a_FileName) const;
        std::string Serialize() const;
        static Population Deserialize(const std::string &data);

        // Checks population invariants (size match, valid parameters, valid genomes).
        bool Validate(std::string *error = nullptr) const;

        //////////////////////
        // NEW STUFF
        std::vector<Species> m_TempSpecies;  // useful in reproduction

        //////////////////////
        // Real-Time methods

        // Estimates the estimated average fitness for all species
        // void EstimateAllAverages();

        // Reproduce the population champ only
        // Genome ReproduceChamp();

        // Choose the parent species that will reproduce
        // This is a real-time version of fitness sharing
        // Returns the species index
        unsigned int ChooseParentSpecies();

        // Removes worst member of the whole population that has been around for a minimum amount of time returns the genome that was just deleted (may be
        // useful)
        Genome RemoveWorstIndividual();

        void ClearEmptySpecies();

        // The main reaitime tick. Analog to Epoch(). Replaces the worst evaluated individual with a new one.
        // Returns a pointer to the new baby.
        // and copies the genome that was deleted to a_geleted_genome
        Genome *Tick(Genome &a_deleted_genome);

        // Takes an individual and puts it in its apropriate species Useful in realtime when the compatibility treshold changes
        void ReassignSpecies(int a_genome_idx);

        unsigned int m_NumEvaluations = 0;

        ///////////////////////////////
        // Novelty search

        // A pointer to the archive of PhenotypeBehaviors Necessary to contain derived custom classes.
        // Null unless InitPhenotypeBehaviorData() was called. The raw-pointer
        // overload is the supported C++ API (no Bindings.cpp by design); the
        // reference shared_ptr/GetBehaviorArchive convenience is intentionally omitted.
        std::vector<PhenotypeBehavior> *m_BehaviorArchive = nullptr;

        // Call this function to allocate memory for your custom behaviors. This initializes everything.
        void InitPhenotypeBehaviorData(std::vector<PhenotypeBehavior> *a_population, std::vector<PhenotypeBehavior> *a_archive);

        // This is the main method performing novelty search. Performs one reproduction and assigns novelty scores based on the current population and the
        // archive. If a successful behavior was encountered, returns true and the genome a_SuccessfulGenome is overwritten with the genome generating the
        // successful behavior
        bool NoveltySearchTick(Genome &a_SuccessfulGenome);

        Real ComputeSparseness(Genome &genome);

        // counters for archive stagnation
        unsigned int m_GensSinceLastArchiving = 0;
        unsigned int m_QuickAddCounter = 0;
    };

}  // namespace NEAT
