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
 * File:        Population.cpp
 * Description: Implementation of the Population class.
 */

#include "Population.h"

#include <stdio.h>

#include <algorithm>
#include <fstream>
#include <sstream>

#include "AssertMacros.h"
#include "Genome.h"
#include "Parameters.h"
#include "PhenotypeBehavior.h"
#include "Random.h"
#include "Species.h"
#include "Utils.h"

namespace NEAT {

    // The constructor
    Population::Population(const Genome &seed, const Parameters &parameters, bool randomizeWeights, double randomizationRange, int rngSeed) {
        rng_.seed(rngSeed);
        bestFitnessEver_ = 0.0;
        parameters_ = parameters;

        generation_ = 0;
        numEvaluations_ = 0;
        nextGenomeID_ = parameters_.populationSize;
        nextSpeciesID_ = 1;
        gensSinceBestFitnessLastChanged_ = 0;
        gensSinceMPCLastChanged_ = 0;

        // Spawn the population
        for (unsigned int i = 0; i < parameters_.populationSize; i++) {
            Genome clone = seed;
            clone.setID(i);
            genomes_.emplace_back(clone);
        }

        // Now now initialize each genome's weights
        for (unsigned int i = 0; i < genomes_.size(); i++) {
            if (randomizeWeights) {
                bool isInvalid = true;
                while (isInvalid) {
                    genomes_[i].randomizeLinkWeights(parameters, rng_);
                    // randomize the traits as well
                    genomes_[i].randomizeTraits(parameters, rng_);
                    // and mutate nodes one initial time
                    genomes_[i].mutateNeuronActivationsA(parameters, rng_);
                    genomes_[i].mutateNeuronActivationsB(parameters, rng_);
                    genomes_[i].mutateNeuronActivationType(parameters, rng_);
                    genomes_[i].mutateNeuronTimeConstants(parameters, rng_);
                    genomes_[i].mutateNeuronBiases(parameters, rng_);

                    // check in the population if there is a clone of that genome
                    isInvalid = false;
                    if (!parameters_.allowClones) {
                        for (unsigned int j = 0; j < genomes_.size(); j++) {
                            if (i != j)  // don't compare the same genome
                            {
                                if (genomes_[i].compatibilityDistance(genomes_[j], parameters_) < parameters_.minDeltaCompatEqualGenomes)  // equal genomes?
                                {
                                    isInvalid = true;
                                    break;
                                }
                            }
                        }
                    }

                    // Also don't let any genome to fail the constraints
                    if (!isInvalid)  // doesn't make sense to do the test if already failed
                    {
                        if (genomes_[i].failsConstraints(parameters)) {
                            isInvalid = true;
                        }
                    }
                }
            }

            // m_Genomes[i].CalculateDepth();
        }
        // Speciate
        speciate();

        // set these phased search variables now since used in MutateGenome
        if (parameters_.phasedSearching) {
            searchMode_ = COMPLEXIFYING;
        } else {
            searchMode_ = BLENDED;
        }

        // Initialize the innovation database
        innovationDatabase_.init(seed);

        bestGenome_ = species_[0].individuals_[0];  // GetLeader();

        id_ = 0;

        // Sort();

        // Set up the rest of the phased search variables
        calculateMPC();
        baseMPC_ = currentMPC_;
        oldMPC_ = baseMPC_;

        // Reset IDs to be sure
        int cid = 0;
        for (int i = 0; i < species_.size(); i++) {
            for (int j = 0; j < species_[i].individuals_.size(); j++) {
                species_[i].individuals_[j].setID(cid);
                cid++;
            }
        }

        innovationDatabase_.innovations_.reserve(50000);
    }

    Population::Population(const std::string sFileName) {
        const char *fileName = sFileName.c_str();
        bestFitnessEver_ = 0.0;

        generation_ = 0;
        numEvaluations_ = 0;
        nextSpeciesID_ = 1;
        id_ = 0;
        gensSinceBestFitnessLastChanged_ = 0;
        gensSinceMPCLastChanged_ = 0;

        std::ifstream dataFile(fileName);
        if (!dataFile.is_open()) throw std::runtime_error("operation failed");
        std::string str;

        // Load the parameters
        parameters_.load(dataFile);

        // Load the innovation database
        innovationDatabase_.init(dataFile);

        // Load all genomes
        for (unsigned int i = 0; i < parameters_.populationSize; i++) {
            Genome genome(dataFile);
            genomes_.emplace_back(genome);
        }
        dataFile.close();

        nextGenomeID_ = 0;
        for (unsigned int i = 0; i < genomes_.size(); i++) {
            if (genomes_[i].getID() > nextGenomeID_) {
                nextGenomeID_ = genomes_[i].getID();
            }
        }
        nextGenomeID_++;

        // Initialize
        speciate();
        bestGenome_ = species_[0].getLeader();

        // Sort();

        // Set up the phased search variables
        calculateMPC();
        baseMPC_ = currentMPC_;
        oldMPC_ = baseMPC_;
        if (parameters_.phasedSearching) {
            searchMode_ = COMPLEXIFYING;
        } else {
            searchMode_ = BLENDED;
        }
    }

    // Save a whole population to a file
    void Population::save(const char *fileName) {
        FILE *file = fopen(fileName, "w");

        // Save the parameters
        parameters_.save(file);

        // Save the innovation database
        innovationDatabase_.save(file);

        // Save each genome
        for (unsigned i = 0; i < species_.size(); i++) {
            for (unsigned j = 0; j < species_[i].individuals_.size(); j++) {
                species_[i].individuals_[j].save(file);
            }
        }

        // bye
        fclose(file);
    }

    // Calculates the current mean population complexity
    void Population::calculateMPC() {
        currentMPC_ = 0;

        for (unsigned int i = 0; i < genomes_.size(); i++) {
            currentMPC_ += accessGenomeByIndex(i).numLinks();
        }

        currentMPC_ /= genomes_.size();
    }

    // Separates the population into species also adjusts the compatibility treshold if this feature is enabled
    void Population::speciate() {
        // iterate through the genome list and speciate at least 1 genome must be present
        ASSERT(genomes_.size() > 0);

        // first clear out the species
        species_.clear();

        bool added = false;

        // NOTE: we are comparing the new generation's genomes to the representatives from species creation time!
        //
        for (unsigned int i = 0; i < genomes_.size(); i++) {
            added = false;

            // iterate through each species and check if compatible. If compatible, then add to the species. if not compatible, create a new species.
            for (unsigned int j = 0; j < species_.size(); j++) {
                if (species_[j].numIndividuals() > 0) {
                    if (genomes_[i].isCompatibleWith(species_[j].getRepresentative(), parameters_)) {
                        // Compatible, add to species
                        species_[j].addIndividual(genomes_[i]);
                        added = true;

                        break;
                    }
                }
            }

            if (!added) {
                // didn't find compatible species, create new species
                species_.push_back(Species(genomes_[i], parameters_, nextSpeciesID_));
                nextSpeciesID_++;
            }
        }

        // Remove all empty species (cleanup routine for every case..)
        clearEmptySpecies();
    }

    // Adjust the fitness of all species
    void Population::adjustFitness() {
        ASSERT(genomes_.size() > 0);
        ASSERT(species_.size() > 0);

        for (unsigned int i = 0; i < species_.size(); i++) {
            species_[i].adjustFitness(parameters_);  // m_Species[i].m_Parameters);
        }
    }

    // Calculates how many offspring each genome should have
    void Population::countOffspring() {
        ASSERT(genomes_.size() > 0);
        ASSERT(genomes_.size() == parameters_.populationSize);

        double totalAdjustedFitness = 0.0;
        double averageAdjustedFitness = 0.0;
        Genome t;

        // get the total adjusted fitness for all individuals
        for (unsigned int i = 0; i < species_.size(); i++) {
            for (unsigned int j = 0; j < species_[i].individuals_.size(); j++) {
                totalAdjustedFitness += species_[i].individuals_[j].getAdjFitness();

                // std::cout << m_Species[i].m_Individuals[j].GetFitness() << " " << m_Species[i].m_Individuals[j].GetAdjFitness() << "\n";
            }
        }

        // must be above 0
        ASSERT(totalAdjustedFitness > 0.0);

        averageAdjustedFitness = totalAdjustedFitness / static_cast<double>(parameters_.populationSize);
        if (averageAdjustedFitness == 0.0) {
            averageAdjustedFitness = 1.0;
        }

        // std::cout << t_average_adjusted_fitness << "\n";

        // Calculate how much offspring each individual should have
        for (unsigned int i = 0; i < species_.size(); i++) {
            for (unsigned int j = 0; j < species_[i].individuals_.size(); j++) {
                species_[i].individuals_[j].setOffspringAmount(species_[i].individuals_[j].getAdjFitness() / averageAdjustedFitness);
            }
        }

        // Now count how many offpring each species should have
        for (unsigned int i = 0; i < species_.size(); i++) {
            species_[i].countOffspring();
        }
    }

    // This little tool function helps ordering the genomes by fitness
    bool speciesGreater(Species &ls, Species &rs) { return ((ls.getBestFitness()) > (rs.getBestFitness())); }
    void Population::sort() {
        ASSERT(species_.size() > 0);

        // Step through each species and sort its members by fitness
        for (unsigned int i = 0; i < species_.size(); i++) {
            ASSERT(species_[i].numIndividuals() > 0);
            species_[i].sortIndividuals();
        }

        // Now sort the species by fitness (best first)
        std::sort(species_.begin(), species_.end(), speciesGreater);

        // for(int i=0;i<m_Species.size();i++)
        // std::cout << m_Species[i].GetBestFitness() << "\n";
        // std::cout << "\n\n";
    }

    // Updates the species
    void Population::updateSpecies() {
        // search for the current best species ID if not at generation #0
        int oldbestid = -1, newbestid = -1;
        int oldbestindex = -1;
        if (generation_ > 0) {
            for (unsigned int i = 0; i < species_.size(); i++) {
                if (species_[i].isBestSpecies()) {
                    oldbestid = species_[i].id();
                    oldbestindex = i;
                }
            }
            ASSERT(oldbestid != -1);
            ASSERT(oldbestindex != -1);
        }

        for (unsigned int i = 0; i < species_.size(); i++) {
            species_[i].setBestSpecies(false);
        }

        bool marked = false;  // new best species marked?

        for (unsigned int i = 0; i < species_.size(); i++) {
            // Reset the species and update its age
            species_[i].increaseAgeGens();
            species_[i].increaseGensNoImprovement();
            species_[i].setOffspringRqd(0);

            // Mark the best species so it is guaranteed to survive
            // Only one species will be marked - in case several species
            // have equally best fitness
            if ((species_[i].getBestFitness() >= bestFitnessEver_) && (!marked)) {
                species_[i].setBestSpecies(true);
                marked = true;
                newbestid = species_[i].id();
            }
        }

        // This prevents the previous best species from sudden death If the best species happened to be another one, reset the old species age so it still will
        // have a chance of survival and improvement if it grows old and stagnates again, it is no longer the best one so it will die off anyway.
        if ((oldbestid != newbestid) && (oldbestid != -1)) {
            species_[oldbestindex].resetAgeGens();
        }
    }

    // the epoch method - the heart of the GA
    void Population::epoch() {
        // So, all genomes are evaluated..
        for (unsigned int i = 0; i < species_.size(); i++) {
            for (unsigned int j = 0; j < species_[i].individuals_.size(); j++) {
                species_[i].individuals_[j].setEvaluated();
            }
        }

        // Sort each species's members by fitness and the species by fitness
        // Sort();

        // Update species stagnation info & stuff
        updateSpecies();

        ///////////////////
        // Preparation
        ///////////////////

        // Adjust the species's fitness
        adjustFitness();

        // Count the offspring of each individual and species
        countOffspring();

        // Incrementing the global stagnation counter, we can check later for global stagnation
        gensSinceBestFitnessLastChanged_++;
        // Find and save the best genome and fitness
        for (unsigned int i = 0; i < species_.size(); i++) {
            // Update best genome info
            species_[i].bestGenome_ = species_[i].getLeader();

            for (unsigned int j = 0; j < species_[i].individuals_.size(); j++) {
                // Make sure all are evaluated as we don't run in realtime
                species_[i].individuals_[j].setEvaluated();

                const double fitness = species_[i].individuals_[j].getFitness();
                if (bestFitnessEver_ < fitness) {
                    // Reset the stagnation counter only if the fitness jump is greater or equal to the delta.
                    if (fabs(fitness - bestFitnessEver_) >= parameters_.stagnationDelta) {
                        gensSinceBestFitnessLastChanged_ = 0;
                    }

                    bestFitnessEver_ = fitness;
                    bestGenomeEver_ = species_[i].individuals_[j];
                }
            }
        }

        // Find and save the current best genome
        double bestf = std::numeric_limits<double>::min();
        for (unsigned int i = 0; i < species_.size(); i++) {
            for (unsigned int j = 0; j < species_[i].individuals_.size(); j++) {
                if (species_[i].individuals_[j].getFitness() > bestf) {
                    bestf = species_[i].individuals_[j].getFitness();
                    bestGenome_ = species_[i].individuals_[j];
                }
            }
        }

        // adjust the compatibility threshold
        if (parameters_.dynamicCompatibility == true) {
            if ((generation_ % parameters_.compatTreshChangeIntervalGenerations) == 0) {
                if (species_.size() > parameters_.maxSpecies) {
                    parameters_.compatTreshold += parameters_.compatTresholdModifier;
                } else if (species_.size() < parameters_.minSpecies) {
                    parameters_.compatTreshold -= parameters_.compatTresholdModifier;
                }
            }

            if (parameters_.compatTreshold < parameters_.minCompatTreshold) parameters_.compatTreshold = parameters_.minCompatTreshold;
        }

        // A special case for global stagnation.
        // Delta coding - if there is a global stagnation
        // for dropoff age + 10 generations, focus the search on the top 2 species,
        // in case there are more than 2, of course
        if (parameters_.deltaCoding) {
            if (gensSinceBestFitnessLastChanged_ > (parameters_.speciesMaxStagnation + 10)) {
                // make the top 2 reproduce by 50% individuals
                // and the rest - no offspring
                if (species_.size() > 2) {
                    // The first two will reproduce
                    species_[0].setOffspringRqd(parameters_.populationSize / 2);
                    species_[1].setOffspringRqd(parameters_.populationSize / 2);

                    // The rest will not
                    for (unsigned int i = 2; i < species_.size(); i++) {
                        species_[i].setOffspringRqd(0);
                    }

                    // Now reset the stagnation counter and species age
                    species_[0].resetAgeGens();
                    species_[1].resetAgeGens();
                    gensSinceBestFitnessLastChanged_ = 0;
                }
            }
        }

        //////////////////////////////////
        // Phased searching core logic
        //////////////////////////////////
        // Update the current MPC
        calculateMPC();
        if (parameters_.phasedSearching) {
            // Keep track of complexity when in simplifying phase
            if (searchMode_ == SIMPLIFYING) {
                // The MPC has lowered?
                if (currentMPC_ < oldMPC_) {
                    // reset that
                    gensSinceMPCLastChanged_ = 0;
                    oldMPC_ = currentMPC_;
                } else {
                    gensSinceMPCLastChanged_++;
                }
            }

            // At complexifying phase?
            if (searchMode_ == COMPLEXIFYING) {
                // Need to begin simplification?
                if (currentMPC_ > (baseMPC_ + parameters_.simplifyingPhaseMPCTreshold)) {
                    // Do this only if the whole population is stagnating
                    if (gensSinceBestFitnessLastChanged_ > parameters_.simplifyingPhaseStagnationTreshold) {
                        // Change the current search mode
                        searchMode_ = SIMPLIFYING;

                        // Reset variables for simplifying mode
                        gensSinceMPCLastChanged_ = 0;
                        oldMPC_ = std::numeric_limits<double>::max();  // Really big one

                        // reset the age of species
                        for (unsigned int i = 0; i < species_.size(); i++) {
                            species_[i].resetAgeGens();
                        }
                    }
                }
            } else if (searchMode_ == SIMPLIFYING)
            // At simplifying phase?
            {
                // The MPC reached its floor level?
                if (gensSinceMPCLastChanged_ > parameters_.complexityFloorGenerations) {
                    // Re-enter complexifying phase
                    searchMode_ = COMPLEXIFYING;

                    // Set the base MPC with the current MPC
                    baseMPC_ = currentMPC_;

                    // reset the age of species
                    for (unsigned int i = 0; i < species_.size(); i++) {
                        species_[i].resetAgeGens();
                    }
                }
            }
        }

        /////////////////////////////
        // Reproduction
        /////////////////////////////

        // Perform reproduction for each species
        tempSpecies_.clear();
        tempSpecies_ = species_;
        for (unsigned int i = 0; i < tempSpecies_.size(); i++) {
            tempSpecies_[i].clear();
            tempSpecies_[i].addIndividual(species_[i].individuals_[0]);
        }

        for (unsigned int i = 0; i < species_.size(); i++) {
            species_[i].reproduce(*this, parameters_, rng_);
        }
        for (unsigned int i = 0; i < tempSpecies_.size(); i++) {
            tempSpecies_[i].removeIndividual(0);
        }
        species_ = tempSpecies_;

        // Remove all empty species (cleanup routine for every case..)
        for (unsigned int i = 0; i < species_.size(); i++) {
            if (species_[i].individuals_.empty()) {
                species_.erase(species_.begin() + i);
                i--;
            }
        }
        // If the total amount of genomes reproduced is less than the population size,
        // due to some floating point rounding error,
        // we will add some bonus clones of the first species's leader to it

        unsigned int totalGenomes = 0;
        for (unsigned int i = 0; i < species_.size(); i++) totalGenomes += static_cast<unsigned int>(species_[i].individuals_.size());

        // Rounding of the per-species offspring quotas can also overshoot the
        // population size (not just undershoot). Trim the surplus newborns —
        // the freshly created babies at the end of the last species — so the
        // population-size invariant (SameGenomeIDCheck, AccessGenomeByIndex,
        // CountOffspring) always holds.
        while (totalGenomes > parameters_.populationSize) {
            int last = static_cast<int>(species_.size()) - 1;
            while (last >= 0 && species_[last].individuals_.empty()) {
                last--;
            }
            if (last < 0) {
                break;  // cannot happen (total > 0), but never spin
            }
            species_[last].removeIndividual(static_cast<unsigned int>(species_[last].individuals_.size() - 1));
            if (species_[last].individuals_.empty()) {
                species_.erase(species_.begin() + last);
            }
            totalGenomes--;
        }

        if (totalGenomes < parameters_.populationSize) {
            int nts = parameters_.populationSize - totalGenomes;

            while (nts--) {
                ASSERT(species_.size() > 0);
                Genome tg = species_[0].individuals_[0];
                // Bonus clones must get fresh IDs, otherwise SameGenomeIDCheck()
                // (and AccessGenomeByID()) observe duplicate IDs in the population.
                tg.setID(nextGenomeID_);
                nextGenomeID_++;
                species_[0].addIndividual(tg);
            }
        }

        // Increase generation number
        generation_++;

        // At this point we may also empty our innovation database This is the place where we control whether we want to keep innovation numbers forever or not.
        if (!parameters_.innovationsForever) {
            innovationDatabase_.flush();
        }
    }

    Genome Population::getBestGenome() const {
        double best = std::numeric_limits<double>::min();
        unsigned int indexSpecies = 0;
        unsigned int indexGenome = 0;
        for (unsigned int i = 0; i < species_.size(); i++) {
            for (unsigned int j = 0; j < species_[i].individuals_.size(); j++) {
                if (species_[i].individuals_[j].getFitness() > best) {
                    best = species_[i].individuals_[j].getFitness();
                    indexSpecies = i;
                    indexGenome = j;
                }
            }
        }

        return species_[indexSpecies].individuals_[indexGenome];
    }

    void Population::sameGenomeIDCheck() {
        // Count occurrences of each genome ID.
        std::map<int, int> ids;
        for (unsigned int i = 0; i < species_.size(); i++) {
            for (unsigned int j = 0; j < species_[i].individuals_.size(); j++) {
                ids[species_[i].individuals_[j].getID()] += 1;
            }
        }

        for (std::map<int, int>::iterator it = ids.begin(); it != ids.end(); it++) {
            if (it->second > 1) {
                std::ostringstream message;
                message << "Genome ID " << it->first << " appears " << it->second << " times in the population\n";
                throw std::runtime_error(message.str());
            }
        }
    }

    Genome &Population::accessGenomeByIndex(int const index) {
        // The genomes live in the species; m_Genomes is only the initial seed list
        // and goes stale after the first Epoch, so bounds-check against the
        // actual number of individuals.
        ASSERT(index >= 0);
        ASSERT(index < static_cast<int>(numGenomes()));
        int counter = 0;

        for (unsigned int i = 0; i < species_.size(); i++) {
            for (unsigned int j = 0; j < species_[i].individuals_.size(); j++) {
                if (counter == index)  // reached the index?
                {
                    return species_[i].individuals_[j];
                }

                counter++;
            }
        }

        std::ostringstream message;
        message << "No such index in population - " << index << "\n";

        // not found?!
        throw std::runtime_error(message.str());
    }

    Genome &Population::accessGenomeByID(int const id) {
        for (unsigned int i = 0; i < species_.size(); i++) {
            for (unsigned int j = 0; j < species_[i].individuals_.size(); j++) {
                if (species_[i].individuals_[j].getID() == id)  // reached the ID?
                {
                    return species_[i].individuals_[j];
                }
            }
        }

        std::ostringstream message;
        message << "No such ID in population - " << id << "\n";

        // not found?!
        throw std::runtime_error(message.str());
    }

    /////////////////////////////////
    // Realtime code

    // Decides which species should have offspring. Returns the index of the species
    unsigned int Population::chooseParentSpecies() {
        ASSERT(species_.size() > 0);

        unsigned int curspecies = 0;
        // do
        std::vector<double> probs;
        for (int i = 0; i < species_.size(); i++) {
            if ((species_[i].numEvaluated() == 0) || (species_[i].numIndividuals() == 0)) {
                probs.push_back(0.0);
            } else {
                probs.push_back(species_[i].averageFitness_);
            }
        }
        curspecies = rng_.roulette(probs);

        return curspecies;
    }

    // Takes a genome and assigns it to a different species (where it belongs)
    void Population::reassignSpecies(int genomeIndex) {
        // first remember where is this genome exactly
        int speciesIndex = 0, genomeRelIndex = 0;
        int counter = 0;

        // to keep the genome
        Genome genome;

        // search for it
        speciesIndex = 0;
        for (int i = 0; i < species_.size(); i++) {
            genomeRelIndex = 0;
            if ((counter + species_[i].individuals_.size()) > genomeIndex) {
                // it's here
                genomeRelIndex = genomeIndex - counter;
                break;
            } else {
                counter += species_[i].individuals_.size();
                speciesIndex++;
            }
        }

        // save the individual
        genome = species_[speciesIndex].individuals_[genomeRelIndex];

        // Remove it from its species
        species_[speciesIndex].removeIndividual(genomeRelIndex);

        // Find a new species for this genome
        bool found = false;
        std::vector<Species>::iterator curSpecies = species_.begin();

        // No species yet?
        if (curSpecies == species_.end()) {
            // create the first species and place the baby there
            species_.emplace_back(Species(genome, parameters_, getNextSpeciesID()));
            incrementNextSpeciesID();
        } else {
            // try to find a compatible species
            Genome toCompare = curSpecies->getRepresentative();

            found = false;
            while ((curSpecies != species_.end()) && (!found)) {
                if (genome.isCompatibleWith(toCompare, parameters_)) {
                    // found a compatible species
                    curSpecies->addIndividual(genome);
                    found = true;  // the search is over
                } else {
                    // keep searching for a matching non-empty species

                    while (1) {
                        curSpecies++;
                        if (curSpecies == species_.end()) {
                            break;
                        }
                        if (curSpecies->numIndividuals() > 0) {
                            toCompare = curSpecies->getRepresentative();
                            break;
                        }
                    }
                }
            }

            // if couldn't find a match, make a new species
            if (!found) {
                species_.emplace_back(Species(genome, parameters_, getNextSpeciesID()));
                incrementNextSpeciesID();
            }
        }
    }

    // Main realtime loop. We assume that the whole population was evaluated once before calling this.
    // Returns a pointer to the baby in the population. It will be the only individual that was not evaluated.
    // Set the m_Evaluated flag of the baby to true after evaluation!
    Genome *Population::tick(Genome &deletedGenome) {
        // Make sure at least one individual is evaluated
        int ne = 0;
        for (int i = 0; i < species_.size(); i++) {
            ne += species_[i].numEvaluated();
        }
        if (ne == 0) {
            throw std::runtime_error("Called Tick() on population with no evaluated individuals.\n");
        }

#ifdef VDEBUG
        std::cout << "tracking stuff\n";
#endif

        numEvaluations_++;

        // Find and save the best genome and fitness
        evalsSinceBestFitnessLastChanged_++;
        for (int i = 0; i < species_.size(); i++) {
            // m_Species[i].IncreaseEvalsNoImprovement();

            for (int j = 0; j < species_[i].individuals_.size(); j++) {
                // if (m_Species[i].m_Individuals[j].GetFitness() <= 0.0)
                //{
                //     m_Species[i].m_Individuals[j].SetFitness(0.00001);
                // }

                double fitness = species_[i].individuals_[j].getFitness();
                if (std::isnan(fitness) || std::isinf(fitness)) {
                    fitness = 0;
                }

                if (fitness > bestFitnessEver_) {
                    // Reset the stagnation counter only if the fitness jump is greater or equal to the delta.
                    if (fabs(fitness - bestFitnessEver_) >= parameters_.stagnationDelta) {
                        evalsSinceBestFitnessLastChanged_ = 0;
                    }

                    bestFitnessEver_ = fitness;
                    bestGenomeEver_ = species_[i].individuals_[j];
                }
            }
        }

        double f = std::numeric_limits<double>::min();
        for (int i = 0; i < species_.size(); i++) {
            for (int j = 0; j < species_[i].individuals_.size(); j++) {
                if (species_[i].individuals_[j].getFitness() > f) {
                    f = species_[i].individuals_[j].getFitness();
                    bestGenome_ = species_[i].individuals_[j];
                }

                if (species_[i].individuals_[j].getFitness() > species_[i].getBestFitness()) {
                    species_[i].bestFitness_ = species_[i].individuals_[j].getFitness();
                    species_[i].evalsNoImprovement_ = 0;
                }
            }
        }

        // adjust the compatibility treshold
        bool changed = false;
        if (parameters_.dynamicCompatibility == true) {
            double oldcompat = parameters_.compatTreshold;
            if ((numEvaluations_ % parameters_.compatTreshChangeIntervalEvaluations) == 0) {
                if (species_.size() > parameters_.maxSpecies) {
                    parameters_.compatTreshold += parameters_.compatTresholdModifier;
                } else if (species_.size() < parameters_.minSpecies) {
                    parameters_.compatTreshold -= parameters_.compatTresholdModifier;
                }

                if (parameters_.compatTreshold < parameters_.minCompatTreshold) parameters_.compatTreshold = parameters_.minCompatTreshold;

                if (parameters_.compatTreshold != oldcompat) {
                    changed = true;
                }
            }
        }

        // If the compatibility treshold was changed, reassign all individuals by species
        if (changed) {
            genomes_.clear();
            for (unsigned int i = 0; i < species_.size(); i++) {
                for (unsigned int j = 0; j < species_[i].individuals_.size(); j++) {
                    genomes_.push_back(species_[i].individuals_[j]);
                }
            }

            speciate();
        }

#ifdef VDEBUG
        sameGenomeIDCheck();
#endif

#ifdef VDEBUG
        std::cout << "remove worst\n";
#endif
        // Remove the worst individual
        deletedGenome = removeWorstIndividual();

#ifdef VDEBUG
        std::cout << "calc avg fitness\n";
#endif
        // Recalculate all averages for each species
        // If the average species fitness of a species is 0,
        // then there are no evaluated individuals in it.
        for (unsigned int i = 0; i < species_.size(); i++) {
            species_[i].calculateAverageFitness();
        }

#ifdef VDEBUG
        std::cout << "choose parents\n";
#endif
        // Now spawn the new offspring
        unsigned int parentSpeciesIndex = chooseParentSpecies();

        Genome baby = species_[parentSpeciesIndex].reproduceOne(*this, parameters_,  // m_Species[t_parent_species_index].m_Parameters,
                                                                rng_);
        ASSERT(baby.numInputs() > 0);
        ASSERT(baby.numOutputs() > 0);
        Genome *toReturn = nullptr;

#ifdef VDEBUG
        std::cout << "placing baby in species\n";
#endif

        // Add the baby to its proper species
        bool found = false;
        std::vector<Species>::iterator curSpecies = species_.begin();

        // No species yet?
        if (curSpecies == species_.end()) {
            // create the first species and place the baby there
            species_.push_back(Species(baby, parameters_, getNextSpeciesID()));  // clone the pop's parameters when creating species
            // the last one
            toReturn = &(species_[species_.size() - 1].individuals_[species_[species_.size() - 1].individuals_.size() - 1]);
            incrementNextSpeciesID();

#ifdef VDEBUG
            std::cout << "made new species\n";
#endif
        } else {
            // try to find a compatible species
            Genome toCompare = curSpecies->getRepresentative();

            found = false;
            while ((curSpecies != species_.end()) && (!found)) {
                if (baby.isCompatibleWith(toCompare, parameters_)) {
                    // found a compatible species
                    curSpecies->addIndividual(baby);
                    toReturn = &(curSpecies->individuals_[curSpecies->individuals_.size() - 1]);
                    found = true;  // the search is over

                    // increase the evals counter for the new species
                    curSpecies->increaseEvalsNoImprovement();

#ifdef VDEBUG
                    std::cout << "found compatible species\n";
#endif
                } else {
                    // keep searching for a matching species
                    while (1) {
                        curSpecies++;
                        if (curSpecies == species_.end()) {
                            break;
                        }
                        if (curSpecies->numIndividuals() > 0) {
                            toCompare = curSpecies->getRepresentative();
                            break;
                        }
                    };
                }
            }

            // if couldn't find a match, make a new species
            if (!found) {
                species_.push_back(Species(baby, parameters_, getNextSpeciesID()));  // clone the pop's parameters when creating species
                // the last one
                toReturn = &(species_[species_.size() - 1].individuals_[species_[species_.size() - 1].individuals_.size() - 1]);
                incrementNextSpeciesID();

#ifdef VDEBUG
                std::cout << "made new species\n";
#endif
            }
        }

#ifdef VDEBUG
        std::cout << "\n";
#endif

        ASSERT(toReturn != nullptr);

        return toReturn;
    }

    void Population::clearEmptySpecies() {
        std::vector<Species>::iterator cs = species_.begin();
        while (cs != species_.end()) {
            if (cs->numIndividuals() == 0) {
                // remove the dead species
                cs = species_.erase(cs);

                if (cs != species_.begin())  // in case the first species are dead
                    cs--;
            }

            cs++;
        }
    }

    Genome Population::removeWorstIndividual() {
        unsigned int worstIndex = 0;         // within the species
        unsigned int worstSpeciesIndex = 0;  // within the population
        double worstFitness = std::numeric_limits<double>::max();
        int numev = 0;

        Genome genome;

        bool found = false;

        // Find and kill the individual with the worst *adjusted* fitness
        for (unsigned int i = 0; i < species_.size(); i++) {
            if (species_[i].individuals_.size() > 0) {
                double adjinv = 1.0 / static_cast<double>(species_[i].individuals_.size());
                for (unsigned int j = 0; j < species_[i].individuals_.size(); j++) {
                    // only evaluated individuals can be removed
                    if (species_[i].individuals_[j].isEvaluated()) {
                        numev++;
                        double adjustedFitness = species_[i].individuals_[j].getFitness() * adjinv;
                        if (std::isnan(adjustedFitness) || std::isinf(adjustedFitness)) {
                            adjustedFitness = 0;
                        }

                        if (adjustedFitness < worstFitness) {
                            worstFitness = adjustedFitness;
                            worstIndex = j;
                            worstSpeciesIndex = i;
                            found = true;
                        }
                    }
                }
            }
        }

        if (found) {
            genome = species_[worstSpeciesIndex].individuals_[worstIndex];

            // make sure this isn't the only evaluated individual
            if (numev <= 1) {
                return genome;
            }

            // The individual is now removed
            species_[worstSpeciesIndex].removeIndividual(worstIndex);

            // If the species becomes empty, remove the species as well
            if (species_[worstSpeciesIndex].individuals_.empty()) {
                species_.erase(species_.begin() + worstSpeciesIndex);
            }
        } else {
            // set ID of -1 to indicate nothing was removed
            genome.setID(-1);
#ifdef VDEBUG
            std::cout << "RemoveWorst did not remove anything.\n";
#endif
        }

        return genome;
    }

    //////////////////////////////////////////
    // Novelty Search Code
    //////////////////////////////////////////

    // Call this function to allocate memory for your custom
    // behaviors. This initializes everything.
    // Warning! All derived classes MUST NOT have any member variables! Change the algorithms only!
    void Population::initPhenotypeBehaviorData(std::vector<PhenotypeBehavior> *population, std::vector<PhenotypeBehavior> *archive) {
        // Now make each genome point to its behavior
        population->resize(numGenomes());
        behaviorArchive_ = archive;
        behaviorArchive_->clear();

        ASSERT(population->size() == numGenomes());
        int counter = 0;
        for (unsigned int i = 0; i < species_.size(); i++) {
            for (unsigned int j = 0; j < species_[i].individuals_.size(); j++, counter++) {
                species_[i].individuals_[j].phenotypeBehavior_ = &((*population)[counter]);
                species_[i].individuals_[j].setFitness(0);
            }
        }
    }

    double Population::computeSparseness(Genome &genome) {
        // this will hold the distances from our new behavior
        std::vector<double> distancesList;
        distancesList.clear();

        // first add all distances from the population
        for (unsigned int i = 0; i < species_.size(); i++) {
            for (unsigned int j = 0; j < species_[i].individuals_.size(); j++) {
                double distance = genome.phenotypeBehavior_->distanceTo(species_[i].individuals_[j].phenotypeBehavior_);
                distancesList.emplace_back(distance);
            }
        }

        // then add all distances from the archive
        for (unsigned int i = 0; i < behaviorArchive_->size(); i++) {
            distancesList.emplace_back(genome.phenotypeBehavior_->distanceTo(&((*behaviorArchive_)[i])));
        }

        // sort the list, smaller first
        std::sort(distancesList.begin(), distancesList.end());

        // now compute the sparseness
        double sparseness = 0;
        for (unsigned int i = 1; i < (parameters_.noveltySearchK + 1); i++) {
            sparseness += distancesList[i];
        }
        sparseness /= parameters_.noveltySearchK;

        return sparseness;
    }

    // This is the main method performing novelty search. Performs one reproduction and assigns novelty scores based on the current population and the archive.
    // If a successful behavior was encountered, returns true and the genome a_SuccessfulGenome is overwritten with the genome generating the successful
    // behavior
    bool Population::noveltySearchTick(Genome &successfulGenome) {
        // Recompute the sparseness/fitness for all individuals in the population
        // This will introduce the constant pressure to do something new
        if ((numEvaluations_ % parameters_.noveltySearchRecomputeSparsenessEach) == 0) {
            for (unsigned int i = 0; i < species_.size(); i++) {
                for (unsigned int j = 0; j < species_[i].individuals_.size(); j++) {
                    species_[i].individuals_[j].setFitness(computeSparseness(species_[i].individuals_[j]));
                }
            }
        }

        // OK now get the new baby
        Genome tempGenome;
        Genome *newBaby = tick(tempGenome);

        // replace the new individual's behavior to point to the dead one's
        newBaby->phenotypeBehavior_ = tempGenome.phenotypeBehavior_;

        // Now it is time to acquire the new behavior from the baby
        bool success = newBaby->phenotypeBehavior_->acquire(newBaby);

        // if found a successful one, just copy it and return true
        if (success) {
            successfulGenome = *newBaby;
            return true;
        }

        // We have the new behavior, now let's calculate the sparseness of the point in behavior space
        double sparseness = computeSparseness(*newBaby);

        // OK now we have the sparseness for this behavior if the sparseness is above Pmin, add this behavior to the archive
        gensSinceLastArchiving_++;
        if (sparseness > parameters_.noveltySearchPMin) {
            behaviorArchive_->emplace_back(*(newBaby->phenotypeBehavior_));
            gensSinceLastArchiving_ = 0;
            quickAddCounter_++;
        } else {
            // no addition to the archive
            quickAddCounter_ = 0;
        }

        // dynamic Pmin
        if (parameters_.noveltySearchDynamicPMin) {
            // too many generations without adding to the archive?
            if (gensSinceLastArchiving_ > parameters_.noveltySearchNoArchivingStagnationThreshold) {
                parameters_.noveltySearchPMin *= parameters_.noveltySearchPMinLoweringMultiplier;
                if (parameters_.noveltySearchPMin < parameters_.noveltySearchPMinMin) {
                    parameters_.noveltySearchPMin = parameters_.noveltySearchPMinMin;
                }
            }

            // too much additions to the archive (one after another)?
            if (quickAddCounter_ > parameters_.noveltySearchQuickArchivingMinEvaluations) {
                parameters_.noveltySearchPMin *= parameters_.noveltySearchPMinRaisingMultiplier;
            }
        }

        // Now we assign a fitness score based on the sparseness
        // This is still now clear how, but for now fitness = sparseness
        newBaby->setFitness(sparseness);

        successfulGenome = *newBaby;

        // OK now last thing, check if this behavior is the one we're looking for.
        return newBaby->phenotypeBehavior_->successful();
    }

}  // namespace NEAT
