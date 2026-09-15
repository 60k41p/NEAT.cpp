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

#include <cfloat>
#include <vector>

#include "Genes.h"
#include "Genome.h"
#include "Innovation.h"
#include "Parameters.h"
#include "PhenotypeBehavior.h"
#include "Random.h"
#include "Species.h"

namespace NEAT {

    //////////////////////////////////////////////
    // Phased-search mode: complexifying grows structure, simplifying prunes it, blended mixes both.
    // See Parameters::phasedSearching and Population::epoch().
    //////////////////////////////////////////////
    enum SearchMode { COMPLEXIFYING, SIMPLIFYING, BLENDED };

    class Species;

    // The evolving population: species list, innovation registry, best-genome tracking and the
    // generational (epoch()) plus real-time (tick()) and novelty-search (noveltySearchTick()) drivers.
    // Construct from a seed genome, evaluate fitness externally each generation, then call epoch().
    class Population {
        /////////////////////
        // Members
        /////////////////////

       private:
        // Registry of structural innovations shared by all members (see src/Innovation.h).
        InnovationDatabase innovationDatabase_;

        // Next genome/species identifiers to assign.
        unsigned int nextGenomeID_;

        // Next species identifier.
        unsigned int nextSpeciesID_;

        ////////////////////////////
        // Phased searching members

        // Current complexify/simplify phase.
        SearchMode searchMode_;

        // Current mean population complexity (average genome size metric).
        double currentMPC_;

        // Previous generation's MPC, for phase-change detection.
        double oldMPC_;

        // MPC baseline that the simplifying phase returns toward.
        double baseMPC_;

        // Groups all members into species by compatibility distance.
        void speciate();

        // Shares fitness within each species (see Species::adjustFitness()).
        void adjustFitness();

        // Assigns offspring quotas to genomes and species.
        void countOffspring();

        // Clears every species' member list (keeps the species shells).
        void resetSpecies();

        // Refreshes best/worst markers, ages and stagnation counters.
        void updateSpecies();

        // Recomputes currentMPC_ from member genome sizes.
        void calculateMPC();

        // Best fitness ever observed in this run.
        double bestFitnessEver_;

        // Best genome of the current generation and of the whole run.
        Genome bestGenome_;
        Genome bestGenomeEver_;

        // Generations since the run-best fitness improved (drives stagnation handling).
        unsigned int gensSinceBestFitnessLastChanged_;

        // Evaluations since the run-best fitness improved (real-time evolution).
        unsigned int evalsSinceBestFitnessLastChanged_;

        // Generations since the MPC last changed (drives phase switching).
        unsigned int gensSinceMPCLastChanged_;

        // Seed genomes used at construction (kept for re-seeding checks).
        std::vector<Genome> genomes_;

       public:
        // Archive of past champions enforced by Parameters::archiveEnforcement.
        std::vector<Genome> genomeArchive_;

        // Population-owned RNG stream.
        RNG rng_;

        // Active evolution knobs (see src/Parameters.h).
        Parameters parameters_;

        // Current generation counter.
        unsigned int generation_;

        // The live species list.
        std::vector<Species> species_;

        // Population identifier (file/load bookkeeping).
        int id_;

        ////////////////////////////
        // Constructors
        ////////////////////////////

        // Clones the seed genome into a full population; randomizes link weights into [-randomRange .. randomRange]
        // when randomizeWeights is set. The size comes from parameters.populationSize.
        Population(const Genome &g, const Parameters &parameters, bool randomizeWeights, double randomRange, int rngSeed);

        // Loads a population from a saved file (see save()).
        Population(const std::string fileName);

        Population() {};

        ////////////////////////////
        // Methods
        ////////////////////////////

        // Current phased-search state (see SearchMode).
        SearchMode getSearchMode() const { return searchMode_; }
        double getCurrentMPC() const { return currentMPC_; }
        double getBaseMPC() const { return baseMPC_; }

        // Total member count across all species.
        unsigned int numGenomes() const {
            unsigned int num = 0;
            for (unsigned int i = 0; i < species_.size(); i++) {
                num += static_cast<unsigned int>(species_[i].individuals_.size());
            }
            return num;
        }

        unsigned int getGeneration() const { return generation_; }
        double getBestFitnessEver() const { return bestFitnessEver_; }
        // Copies out the fittest genome across all species (defined in Population.cpp).
        Genome getBestGenome() const;

        // Generations/evaluations since the run-best fitness improved.
        unsigned int getStagnation() const { return gensSinceBestFitnessLastChanged_; }
        unsigned int getMPCStagnation() const { return gensSinceMPCLastChanged_; }

        unsigned int getNextGenomeID() const { return nextGenomeID_; }
        unsigned int getNextSpeciesID() const { return nextSpeciesID_; }
        void incrementNextGenomeID() { nextGenomeID_++; }
        void incrementNextSpeciesID() { nextSpeciesID_++; }

        // Throws std::runtime_error when any genome ID occurs more than once (defined in Population.cpp).
        void sameGenomeIDCheck();

        // Mutable access to a member by position/ID (throws std::runtime_error when absent).
        Genome &accessGenomeByIndex(int const index);
        Genome &accessGenomeByID(int const id);

        InnovationDatabase &accessInnovationDatabase() { return innovationDatabase_; }

        // Sorts every species' members fittest-first.
        void sort();

        // Runs speciation, fitness sharing, offspring allocation and reproduction for one generation.
        void epoch();

        // Persists the whole population including species and innovation state.
        void save(const char *fileName);

        // Staging area for offspring during reproduction.
        std::vector<Species> tempSpecies_;

        //////////////////////
        // Real-Time methods

        // Samples the parent species for steady-state reproduction (fitness-proportionate).
        unsigned int chooseParentSpecies();

        // Removes and returns the worst long-lived member (used by tick()).
        Genome removeWorstIndividual();

        // Drops species left without members.
        void clearEmptySpecies();

        // Steady-state step: replaces the worst evaluated member with one offspring. Returns a pointer
        // into the population storage (do not retain across further ticks) and copies the replaced
        // genome into deletedGenome.
        Genome *tick(Genome &deletedGenome);

        // Moves one member to its compatible species (used when the compatibility threshold shifts).
        void reassignSpecies(int genomeIndex);

        // Lifetime evaluation counter.
        unsigned int numEvaluations_;

        ///////////////////////////////
        // Novelty search (see Lehman & Stanley 2011; descriptors derive from PhenotypeBehavior)

        // External behavior archive (owned by the caller; see initPhenotypeBehaviorData()).
        std::vector<PhenotypeBehavior> *behaviorArchive_;

        // Wires caller-owned behavior storage for the population and the archive.
        void initPhenotypeBehaviorData(std::vector<PhenotypeBehavior> *population, std::vector<PhenotypeBehavior> *archive);

        // Performs one novelty reproduction and sparseness assignment. Returns true when a successful
        // behavior was found, overwriting successfulGenome with the genome that produced it.
        bool noveltySearchTick(Genome &successfulGenome);

        // Mean behavioral distance of the genome to its K nearest neighbors (population + archive).
        double computeSparseness(Genome &genome);

        // Generations since the last archive addition / consecutive quick additions (Pmin adaptation).
        unsigned int gensSinceLastArchiving_;
        unsigned int quickAddCounter_;
    };

}  // namespace NEAT
