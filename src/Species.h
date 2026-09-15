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
 * File:        Species.h
 * Description: A compatibility species: a leader genome plus its member list with age, stagnation and
 *              offspring bookkeeping. Species own fitness sharing (adjustFitness()), parent selection
 *              (getIndividual()) and per-species reproduction (reproduce()/reproduceOne()), driven by the
 *              knobs in src/Parameters.h. Populations in src/Population.h hold the species list.
 *
 * References: Stanley & Miikkulainen (2002), Sections 3-4 (speciation, fitness sharing, stagnation);
 *             SelectionMode enumerators mirror the historical MultiNEAT options, of which only the default
 *             truncation path is exercised by the evolution loop (see the AGENTS.md gotchas). Intra-repo
 *             users: src/Population.h, src/Population.cpp, src/Genome.h, tests/TestSpecies.cpp.
 */

#pragma once

#include <vector>

#include "Genes.h"
#include "Genome.h"
#include "Innovation.h"

namespace NEAT {

    // forward
    class Population;

    //////////////////////////////////////////////
    // The Species class
    //////////////////////////////////////////////

    // Parent-selection policy for getIndividual(). Only the default truncation path is consumed by the
    // evolution loop; the remaining values are parsed/preserved for compatibility (see AGENTS.md gotchas).
    enum SelectionMode { TRUNCATION, ROULETTE, RANK_LINEAR, RANK_EXP, TOURNAMENT, STOCHASTIC, BOLTZMANN };

    // A compatibility group with its members and age/offspring accounting.
    class Species {
        /////////////////////
        // Members
        /////////////////////

       private:
        // Species identifier.
        int id_;

        // Population-level best/worst markers, set by Population each generation.
        bool bestSpecies_;
        // Whether this is the worst species of the population.
        bool worstSpecies_;

        // Age in generations / evaluations since the species was created.
        unsigned int ageGenerations_;
        // Age in evaluations.
        unsigned int ageEvaluations_;

        // Offspring quota assigned by countOffspring() for the next generation.
        double offspringRqd_;

       public:
        // Best fitness observed in this species (updated on addIndividual()).
        double bestFitness_;

        // Copy of the best genome seen (used by co-evolutionary setups).
        Genome bestGenome_;

        // Generations since the species improved (drives stagnation kills).
        unsigned int gensNoImprovement_;
        // Evaluations since the species improved (real-time evolution).
        unsigned int evalsNoImprovement_;

        // Display color, safe to access directly.
        int r_, g_, b_;

        // Mean fitness of the current members (see calculateAverageFitness()).
        double averageFitness_;

        ////////////////////////////
        // Constructors
        ////////////////////////////

        Species() {
            id_ = 0;
            bestSpecies_ = false;
            worstSpecies_ = false;
            offspringRqd_ = 0;
            ageGenerations_ = 0;
            ageEvaluations_ = 0;
            bestFitness_ = 0;
            gensNoImprovement_ = 0;
            evalsNoImprovement_ = 0;
            r_ = g_ = b_ = 0;
        };

        // Seeds a species from its founding genome with the given species ID.
        Species(const Genome &seed, const Parameters &parameters, int id);

        // Assignment operator.
        Species &operator=(const Species &g);

        // Compares by species ID (sufficient as an identity key).
        bool operator==(Species const &other) const { return id_ == other.id_; }

        ////////////////////////////
        // Destructor
        ////////////////////////////

        ////////////////////////////
        // Methods
        ////////////////////////////

        // Cached best fitness (may lag behind members; see getActualBestFitness() for a recompute).
        double getBestFitness() const { return bestFitness_; }
        // Recomputes the best fitness over evaluated members.
        double getActualBestFitness() const {
            double f = std::numeric_limits<double>::min();
            for (int i = 0; i < individuals_.size(); i++) {
                if (individuals_[i].isEvaluated()) {
                    if (individuals_[i].getFitness() > f) {
                        f = individuals_[i].getFitness();
                    }
                }
            }
            return f;
        }
        // Marks this species as the population best/worst for the generation.
        void setBestSpecies(bool t) { bestSpecies_ = t; }
        void setWorstSpecies(bool t) { worstSpecies_ = t; }
        // Ages the species by one generation and resets improvement counters.
        void increaseAgeGens() { ageGenerations_++; }
        void resetAgeGens() {
            ageGenerations_ = 0;
            gensNoImprovement_ = 0;
        }
        void increaseGensNoImprovement() { gensNoImprovement_++; }
        // Ages the species by one evaluation and resets improvement counters.
        void increaseAgeEvals() { ageEvaluations_++; }
        void resetAgeEvals() {
            ageEvaluations_ = 0;
            evalsNoImprovement_ = 0;
        }
        void increaseEvalsNoImprovement() { evalsNoImprovement_++; }
        // Offspring quota for the next generation (see countOffspring()).
        void setOffspringRqd(double ofs) { offspringRqd_ = ofs; }
        double getOffspringRqd() const { return offspringRqd_; }
        unsigned int numIndividuals() { return static_cast<unsigned int>(individuals_.size()); }
        void clearIndividuals() { individuals_.clear(); }
        int id() { return id_; }
        int gensNoImprovement() { return static_cast<int>(gensNoImprovement_); }
        int evalsNoImprovement() { return static_cast<int>(evalsNoImprovement_); }
        int ageGens() { return static_cast<int>(ageGenerations_); }
        int ageEvals() { return static_cast<int>(ageEvaluations_); }
        // Copies out the member at the given position.
        Genome getIndividualByIndex(int index) const { return (individuals_[index]); }
        bool isBestSpecies() const { return bestSpecies_; }
        bool isWorstSpecies() const { return worstSpecies_; }
        // Counts evaluated members.
        int numEvaluated() {
            int x = 0;
            for (int i = 0; i < individuals_.size(); i++) {
                if (individuals_[i].isEvaluated()) x++;
            }
            return x;
        }

        // Returns the member with the best fitness (the species representative for compatibility).
        Genome &getLeader();

        // Returns the representative member used for compatibility comparisons.
        Genome &getRepresentative();

        // Adds a member, refreshing best fitness/genome and age bookkeeping.
        void addIndividual(Genome &newIndividual);

        // Returns a parent sampled from the fittest survivalRate fraction (see Parameters).
        Genome &getIndividual(Parameters &parameters, RNG &rng);

        // Returns a uniformly sampled member.
        Genome &getRandomIndividual(RNG &rng);

        // Computes this species' offspring quota from shared fitness.
        void countOffspring();

        // Shares fitness within the species: boosts young species, penalizes old ones and heavily
        // penalizes species stagnating past the dropoff age (see Parameters::speciesMaxStagnation).
        void adjustFitness(Parameters &parameters);

        // Sorts members fittest-first.
        void sortIndividuals();

        // The member genomes owned by this species.
        std::vector<Genome> individuals_;

        // Produces the next generation for this species into the population's staging area.
        void reproduce(Population &pop, Parameters &parameters, RNG &rng);

        // Mutates one offspring genome (structural + weight + trait mutations per Parameters rates).
        // babyIsClone marks an asexual copy that still undergoes mutation.
        void mutateGenome(bool babyIsClone, Population &pop, Genome &baby, Parameters &parameters, RNG &rng);

        // Removes all members.
        void clear() { individuals_.clear(); }

        ////////////////////////////////////////
        // Real-time methods

        // Refreshes averageFitness_ from evaluated members.
        void calculateAverageFitness();

        // Produces a single offspring genome (real-time evolution tick).
        Genome reproduceOne(Population &pop, Parameters &parameters, RNG &rng);

        // Removes the member at the given position.
        void removeIndividual(unsigned int index);
    };

}  // namespace NEAT
