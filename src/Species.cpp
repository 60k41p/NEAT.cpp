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
 * File:        Species.cpp
 * Description: Implementation of the Species class.
 */

#include "Species.h"

#include <algorithm>
#include <sstream>
#include <string>

#include "Genome.h"
#include "Parameters.h"
#include "Population.h"
#include "Random.h"
#include "Utils.h"

// #define COMPAT_EQUALITY_DELTA 0.0000001

namespace NEAT {
    RNG globalRng;

    // Sorts the members of this species by fitness
    /*bool fitness_greater(Genome *ls, Genome *rs)
    {
        return ((ls->GetFitness()) > (rs->GetFitness()));
    }*/

    bool genomeGreater(Genome &ls, Genome &rs) { return (ls.getFitness() > rs.getFitness()); }

    bool indexFitnessPairGreater(std::pair<int, double> &ls, std::pair<int, double> &rs) { return (ls.second > rs.second); }

    // initializes a species with a representative genome and an ID number
    Species::Species(const Genome &genome, const Parameters &parameters, int id) {
        id_ = id;

        // copy the initializing genome locally.
        // it is now the representative of the species.
        // m_Representative = a_Genome;
        bestGenome_ = genome;

        // add the first and only one individual
        individuals_.emplace_back(genome);

        ageGenerations_ = 0;
        gensNoImprovement_ = 0;
        evalsNoImprovement_ = 0;
        offspringRqd_ = 0;
        bestFitness_ = genome.getFitness();
        bestSpecies_ = true;
        worstSpecies_ = false;
        averageFitness_ = 0;
        // m_Parameters = a_Parameters;

        // Choose a random color
        // RNG rng;
        // rng.TimeSeed();
        r_ = static_cast<int>(globalRng.randFloat() * 255);
        g_ = static_cast<int>(globalRng.randFloat() * 255) + 100;
        if (g_ > 255) g_ = 255;
        b_ = static_cast<int>(globalRng.randFloat() * 255);
    }

    Species &Species::operator=(const Species &s) {
        // self assignment guard
        if (this != &s) {
            id_ = s.id_;
            // m_Representative = a_S.m_Representative;
            bestGenome_ = s.bestGenome_;
            bestSpecies_ = s.bestSpecies_;
            worstSpecies_ = s.worstSpecies_;
            bestFitness_ = s.bestFitness_;
            gensNoImprovement_ = s.gensNoImprovement_;
            evalsNoImprovement_ = s.evalsNoImprovement_;
            averageFitness_ = s.averageFitness_;
            ageGenerations_ = s.ageGenerations_;
            offspringRqd_ = s.offspringRqd_;
            r_ = s.r_;
            g_ = s.g_;
            b_ = s.b_;
            individuals_ = s.individuals_;
        }

        return *this;
    }

    // adds a new member to the species and updates variables
    void Species::addIndividual(Genome &genome) { individuals_.emplace_back(genome); }

    // Individual selection routine
    Genome &Species::getIndividual(Parameters &parameters, RNG &rng)  // const
    {
        if (individuals_.empty()) {
            std::ostringstream message;
            message << "Attempted GetIndividual() but no individuals in species ID " << id_;
            throw std::runtime_error(message.str());
        }

        // Make a pool of only evaluated individuals!
        std::vector<std::pair<int, double> > evaluated;
        for (unsigned int i = 0; i < individuals_.size(); i++) {
            if (individuals_[i].isEvaluated()) {
                evaluated.emplace_back(i, individuals_[i].getFitness());
            }
        }

        // None are evaluated - fall back to random individual
        if (evaluated.empty()) {
            std::ostringstream message;
            message << "Attempted GetIndividual() but no evaluated individuals in species ID " << id_;
            throw std::runtime_error(message.str());
        }
        if (evaluated.size() == 1) {
            return (individuals_[evaluated[0].first]);
        } else if (evaluated.size() == 2) {
            return (individuals_[evaluated[rounded(rng.randFloat())].first]);
        }

        // Warning!!!! The individuals must be sorted by best fitness for this to work
        int chosenOne = 0;

        if (parameters.tournamentSelection) {
            std::vector<std::pair<int, double> > picked;
            // choose N individuals at random
            for (int i = 0; i < parameters.tournamentSize; i++) {
                int c = rng.randInt(0, evaluated.size() - 1);
                picked.push_back(evaluated[c]);
            }

            std::sort(picked.begin(), picked.end(), indexFitnessPairGreater);
            std::vector<double> probs;
            for (int i = 0; i < picked.size(); i++) {
                probs.push_back(picked.size() - i);  // t_picked[i].second);
            }
            chosenOne = picked[rng.roulette(probs)].first;
        } else {
            // sort them here just to make sure
            std::sort(evaluated.begin(), evaluated.end(), indexFitnessPairGreater);

            // Here might be introduced better selection scheme, but this works OK for now
            if (!parameters.rouletteWheelSelection) {
                int numParents = static_cast<int>(parameters.survivalRate * static_cast<double>(individuals_.size()));

                if (numParents >= evaluated.size()) {
                    numParents = evaluated.size() - 1;
                }
                if (numParents < 1) {
                    numParents = 1;
                }

                chosenOne = evaluated[rng.randInt(0, numParents)].first;
            } else {
                // roulette wheel selection
                int numParents = evaluated.size();
                std::vector<double> probs;
                for (unsigned int i = 0; i < numParents; i++) {
                    probs.push_back(evaluated[i].second);
                }
                chosenOne = evaluated[rng.roulette(probs)].first;
            }
        }

        return (individuals_[chosenOne]);
    }

    // returns a completely random individual
    Genome &Species::getRandomIndividual(RNG &rng)  // const
    {
        if (individuals_.empty())  // no members yet, return representative
        {
            std::ostringstream message;
            message << "Attempted GetRandomIndividual() but no individuals in species ID " << id_;
            throw std::runtime_error(message.str());
        } else if (individuals_.size() == 1) {
            return individuals_[0];
        } else {
            int randChoice = 0;
            randChoice = rng.randInt(0, static_cast<int>(individuals_.size() - 1));
            return (individuals_[randChoice]);
        }
    }

    // returns the leader (the member having the best fitness)
    Genome &Species::getLeader()  // const
    {
        // Don't store the leader any more Perform a search over the members and return the most fit member

        // if empty, return representative
        if (individuals_.empty()) {
            std::ostringstream message;
            message << "Attempted GetLeader() but no individuals in species ID " << id_;
            throw std::runtime_error(message.str());
        }

        double maxFitness = std::numeric_limits<double>::min();
        int leaderIndex = 0;
        for (unsigned int i = 0; i < individuals_.size(); i++) {
            double f = individuals_[i].getFitness();
            if (maxFitness < f) {
                maxFitness = f;
                leaderIndex = i;
            }
        }

        // ASSERT(t_leader_idx != -1);
        return (individuals_[leaderIndex]);
    }

    Genome &Species::getRepresentative()  // const
    {
        if (individuals_.size() > 0) {
            return individuals_[0];
        } else {
            std::ostringstream message;
            message << "Attempted GetRepresentative() but no individuals in species ID " << id_;
            throw std::runtime_error(message.str());
        }
    }

    // calculates how many offspring this species should spawn
    void Species::countOffspring() {
        offspringRqd_ = 0;

        for (unsigned int i = 0; i < individuals_.size(); i++) {
            offspringRqd_ += individuals_[i].getOffspringAmount();
        }
    }

    // this method performs fitness sharing it also boosts the fitness of the young and penalizes old species
    void Species::adjustFitness(Parameters &parameters) {
        ASSERT(individuals_.size() > 0);

        // iterate through the members
        for (unsigned int i = 0; i < individuals_.size(); i++) {
            double fitness = individuals_[i].getFitness();

            // the fitness must be positive
            ASSERT(fitness >= 0.0);

            // this prevents the fitness to be below zero
            if (fitness <= 0.0) fitness = 0.0000000001;

            // this prevents nan or infinity to be fitness
            if (std::isnan(fitness)) fitness = 0.0000000001;
            if (std::isinf(fitness)) fitness = 0.0000000001;

            // update the best fitness and stagnation counter
            if (fitness > bestFitness_) {
                bestFitness_ = fitness;
                gensNoImprovement_ = 0;
            }

            // boost the fitness up to some young age
            if (ageGenerations_ < parameters.youngAgeTreshold) {
                fitness *= parameters.youngAgeFitnessBoost;
            }

            // penalty for old species
            if (ageGenerations_ > parameters.oldAgeTreshold) {
                fitness *= parameters.oldAgePenalty;
            }

            // extreme penalty if this species is stagnating for too long time one exception if this is the best species found so far
            if (gensNoImprovement_ > parameters.speciesMaxStagnation) {
                // the best species is always allowed to live
                if (!bestSpecies_) {
                    // when the fitness is lowered that much, the species will likely have 0 offspring and therefore will not survive
                    fitness *= 0.0000001;
                }
            }

            unsigned int ms = individuals_.size();
            ASSERT(ms > 0);
            if (ms == 0) {
                ms = 1;
            }

            // Compute the adjusted fitness for this member
            individuals_[i].setAdjFitness(fitness / static_cast<double>(ms));
        }
    }

    void Species::sortIndividuals() { std::sort(individuals_.begin(), individuals_.end(), genomeGreater); }

    // Removes an individual from the species by its index within the species
    void Species::removeIndividual(unsigned int index) {
        ASSERT(index < individuals_.size());
        individuals_.erase(individuals_.begin() + index);
    }

    // Reproduce mates & mutates the individuals of the species
    // It may access the global species list in the population
    // because some babies may turn out to belong in another species
    // that have to be created.
    // Also calls Birth() for every new baby
    void Species::reproduce(Population &pop, Parameters &parameters, RNG &rng) {
        Genome baby;  // temp genome for reproduction

        unsigned int offspringCount = rounded(getOffspringRqd());
        unsigned int eliteOffspring = 1;  // Rounded(a_Parameters.EliteFraction * m_Individuals.size());
        if (eliteOffspring < 1)           // can't be 0
        {
            eliteOffspring = 1;
        }
        // ensure we have a champ
        unsigned int eliteCount = 0;
        // no offspring?! yikes.. dead species!
        if (offspringCount == 0) {
            // maybe do something else?
            return;
        }

        //////////////////////////
        // Reproduction

        // Spawn t_offspring_count babies
        // bool t_champ_chosen = false;
        bool babyExistsInPop = false;
        while (offspringCount--) {
            // clear baby just in case
            baby = Genome();

            // Select the elite first..

            if (eliteCount < eliteOffspring) {
                // t_baby = m_Individuals[elite_count];
                baby = getLeader();  // m_Individuals[elite_count];
                eliteCount++;
            } else {
                unsigned int constraintTrials = parameters.constraintTrials;  // to prevent infinite loops

                // std::cout << "offspring count:" << t_offspring_count << "\n";
                // std::cout << "making baby\n";

                do  // - while the baby already exists somewhere in the new population or turned invalid in some way
                {
                    // this tells us if the baby is a result of mating
                    bool mated = false;

                    // There must be individuals there..
                    ASSERT(numIndividuals() > 0);

                    // std::cout << "trying to mate..";

                    // for a species of size 1 we can only mutate
                    // NOTE: but does it make sense since we know this is the champ?
                    if (numIndividuals() == 1) {
                        baby = getIndividual(parameters, rng);
                        mated = false;
                    }
                    // else we can mate
                    else {
                        // choose whether to mate at all Do not allow crossover when in simplifying phase
                        if ((rng.randFloat() < parameters.crossoverRate) && (pop.getSearchMode() != SIMPLIFYING)) {
                            // get the father
                            Genome mom;
                            Genome dad;
                            bool interspecies = false;

                            // There is a probability that the father may come from another species
                            if ((rng.randFloat() < parameters.interspeciesCrossoverRate) && (pop.species_.size() > 1)) {
                                /// Find different species via roulette over average fitness as probability
                                std::vector<double> probs;
                                double allp = 0;
                                for (int i = 0; i < pop.species_.size(); i++) {
                                    if (pop.species_[i].id_ == id_) {
                                        probs.push_back(0.0);
                                    } else {
                                        probs.push_back(pop.species_[i].averageFitness_);
                                    }
                                    allp += probs[probs.size() - 1];
                                }
                                if (allp > 0) {
                                    int diffspec = rng.roulette(probs);
                                    mom = getIndividual(parameters, rng);
                                    dad = pop.species_[diffspec].getIndividual(parameters, rng);
                                    interspecies = true;
                                } else {
                                    continue;
                                }
                            } else {
                                // Mate within species
                                mom = getIndividual(parameters, rng);
                                dad = getIndividual(parameters, rng);

                                // The other parent should be a different one number of tries to find different parent
                                int tries = 32;
                                while (((mom.getID() == dad.getID())) && (tries--)) {
                                    mom = getIndividual(parameters, rng);
                                    dad = getIndividual(parameters, rng);
                                }

                                interspecies = false;
                            }

                            // OK we have both mom and dad so mate them Choose randomly one of two types of crossover
                            if (rng.randFloat() < parameters.multipointCrossoverRate) {
                                baby = mom.mate(dad, false, interspecies, rng, parameters);
                            } else {
                                baby = mom.mate(dad, true, interspecies, rng, parameters);
                            }

                            mated = true;
                        }
                        // don't mate - reproduce one individual asexually
                        else {
                            baby = getIndividual(parameters, rng);
                            mated = false;
                        }
                    }

                    // std::cout << "mated:" << t_mated << "\n";

                    // std::cout << "trying to mutate..";

                    // Mutate the baby
                    bool dummy = false;
                    if ((!mated) || (rng.randFloat() < parameters.overallMutationRate)) {
                        mutateGenome(dummy, pop, baby, parameters, rng);
                    }

                    // std::cout << "mutated." << "\n";

                    // Check if this baby is already present somewhere in the offspring we don't want that
                    babyExistsInPop = false;
                    // Unless of course, we want clones to exist
                    if (!parameters.allowClones) {
                        for (unsigned int i = 0; i < pop.tempSpecies_.size(); i++) {
                            for (unsigned int j = 0; j < pop.tempSpecies_[i].individuals_.size(); j++) {
                                if ((baby.compatibilityDistance(pop.tempSpecies_[i].individuals_[j],
                                                                parameters) < parameters.minDeltaCompatEqualGenomes)  // identical genome?
                                ) {
                                    babyExistsInPop = true;
                                    break;
                                }
                            }
                        }
                    }

                    // In case we want to enforce always new individuals
                    if (parameters.archiveEnforcement) {
                        for (unsigned int i = 0; i < pop.genomeArchive_.size(); i++) {
                            if ((baby.compatibilityDistance(pop.genomeArchive_[i],
                                                            parameters) < parameters.minDeltaCompatEqualGenomes)  // identical genome?
                            ) {
                                babyExistsInPop = true;
                                break;
                            }
                        }
                    }

                    // std::cout << "baby exists in pop:" << t_baby_exists_in_pop << "\n";
                } while ((babyExistsInPop || (baby.failsConstraints(parameters))) && (constraintTrials--));  // end do

                // std::cout << "done after " << a_Parameters.ConstraintTrials - t_constraint_trials << "\n";
                // std::cout << "fails constraints:" << t_baby.FailsConstraints(a_Parameters) << "\n\n";
            }

            // We have a new offspring now give the offspring a new ID
            baby.setID(pop.getNextGenomeID());
            pop.incrementNextGenomeID();

            // sort the baby's genes
            baby.sortGenes();

            // clear the baby's fitness
            baby.setFitness(0);
            baby.setAdjFitness(0);
            baby.setOffspringAmount(0);

            baby.resetEvaluated();

            // Archive the baby if needed
            if (parameters.archiveEnforcement) {
                pop.genomeArchive_.emplace_back(baby);
            }

            //////////////////////////////////
            // put the baby to its species  //
            //////////////////////////////////

            // before Reproduce() is invoked, it is assumed that a
            // clone of the population exists with the name of m_TempSpecies
            // we will store results there.
            // after all reproduction completes, the original species will be replaced back

            bool found = false;
            std::vector<Species>::iterator curSpecies = pop.tempSpecies_.begin();

            // No species yet?
            if (curSpecies == pop.tempSpecies_.end()) {
                // create the first species and place the baby there
                pop.tempSpecies_.emplace_back(Species(baby, parameters, pop.getNextSpeciesID()));
                pop.incrementNextSpeciesID();
            } else {
                // try to find a compatible species
                Genome toCompare = curSpecies->getRepresentative();  // was GetRepresentative()

                found = false;
                while ((curSpecies != pop.tempSpecies_.end()) && (!found)) {
                    if (baby.isCompatibleWith(toCompare, parameters)) {
                        // found a compatible species
                        curSpecies->addIndividual(baby);
                        found = true;  // the search is over
                    } else {
                        // keep searching for a matching species
                        /*t_cur_species++;
                        if (t_cur_species != a_Pop.m_TempSpecies.end())
                        {
                            t_to_compare = t_cur_species->GetRepresentative(); // was GetRepresentative()
                        }*/

                        while (1) {
                            curSpecies++;
                            if (curSpecies == pop.tempSpecies_.end()) {
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
                    pop.tempSpecies_.emplace_back(Species(baby, parameters, pop.getNextSpeciesID()));
                    pop.incrementNextSpeciesID();
                }
            }
        }
    }

    ////////////
    // Real-time code
    void Species::calculateAverageFitness() {
        double totalFitness = 0;
        int numIndividuals = 0;

        // consider individuals that were evaluated only!
        for (unsigned int i = 0; i < individuals_.size(); i++) {
            if (individuals_[i].isEvaluated()) {
                double tf = individuals_[i].getFitness();
                if (std::isinf(tf) || std::isnan(tf))  // nan/inf guard
                {
                    tf = 0.0;
                }
                totalFitness += tf;
            }
            numIndividuals++;
        }

        if (numIndividuals > 0) {
            averageFitness_ = totalFitness / static_cast<double>(numIndividuals);
        } else {
            averageFitness_ = 0;
        }
    }

    Genome Species::reproduceOne(Population &pop, Parameters &parameters, RNG &rng) {
        //////////////////////////
        // Reproduction
        bool babyExistsInPop = false;
        bool babyIsClone = false;
        int constraintTrials = parameters.constraintTrials;

        // Spawn only one baby
        Genome baby;  // = GetRandomIndividual(a_RNG); // for storing the result

        do  // - while the baby turned invalid in some way
        {
            baby = Genome();  // clear baby

            // this tells us if the baby is a result of mating
            bool mated = false;

            // There must be individuals there..
            ASSERT(numIndividuals() > 0);

            // for a species of size 1 we can only mutate
            // NOTE: but does it make sense since we know this is the champ?
            if (numIndividuals() == 1) {
                baby = getIndividual(parameters, rng);
                mated = false;
            }
            // else we can mate
            else {
                // choose whether to mate at all Do not allow crossover when in simplifying phase
                if ((rng.randFloat() < parameters.crossoverRate) && (pop.getSearchMode() != SIMPLIFYING)) {
                    // get the mother and father
                    Genome mom;
                    Genome dad;
                    bool interspecies = false;

                    // There is a probability that the father may come from another species
                    if ((rng.randFloat() < parameters.interspeciesCrossoverRate) && (pop.species_.size() > 1)) {
                        // Find different species via roulette over average fitness as probability
                        std::vector<double> probs;
                        double allp = 0;
                        for (int i = 0; i < pop.species_.size(); i++) {
                            if ((pop.species_[i].id_ == id_) || (pop.species_[i].numEvaluated() == 0)) {
                                probs.push_back(0.0);
                            } else {
                                probs.push_back(pop.species_[i].averageFitness_);
                            }
                            allp += probs[probs.size() - 1];
                        }
                        if (allp > 0) {
                            int diffspec = rng.roulette(probs);
                            mom = getIndividual(parameters, rng);
                            dad = pop.species_[diffspec].getIndividual(parameters, rng);
                            interspecies = true;
                        } else {
                            continue;
                        }
                    } else {
                        // Mate within species
                        mom = getIndividual(parameters, rng);
                        dad = getIndividual(parameters, rng);

                        // The other parent should be a different one number of tries to find different parent we can mate the same mom and dad and still get
                        // different baby
                        int tries = 32;
                        while (((mom.getID() == dad.getID())) && (tries--)) {
                            mom = getIndividual(parameters, rng);
                            dad = getIndividual(parameters, rng);
                        }
                        interspecies = false;
                    }

                    // OK we have both mom and dad so mate them Choose randomly one of two types of crossover
                    if (rng.randFloat() < parameters.multipointCrossoverRate) {
                        baby = mom.mate(dad, false, interspecies, rng, parameters);
                    } else {
                        baby = mom.mate(dad, true, interspecies, rng, parameters);
                    }

#ifdef VDEBUG
                    std::cout << "mated baby\n";
#endif
                    mated = true;
                }
                // don't mate - reproduce one individual asexually
                else {
                    baby = getIndividual(parameters, rng);
                    mated = false;
                }
            }

            // Mutate the baby
            babyIsClone = false;
            bool dummy = false;
            if ((!mated) || (rng.randFloat() < parameters.overallMutationRate)) {
                mutateGenome(dummy, pop, baby, parameters, rng);
#ifdef VDEBUG
                std::cout << "mutated baby\n";
#endif
            }

            // Check if this baby is already present somewhere in the offspring we don't want that
            babyExistsInPop = false;
            // Unless of course, we want clones to exist
            if (!parameters.allowClones) {
                for (unsigned int i = 0; i < pop.species_.size(); i++) {
                    for (unsigned int j = 0; j < pop.species_[i].individuals_.size(); j++) {
                        if ((baby.compatibilityDistance(pop.species_[i].individuals_[j],
                                                        parameters) < parameters.minDeltaCompatEqualGenomes)  // identical genome?
                        ) {
                            babyExistsInPop = true;
                            break;
                        }
                    }
                }
            }

            // In case we want to enforce always new individuals
            if (parameters.archiveEnforcement && (!babyExistsInPop)) {
                for (unsigned int i = 0; i < pop.genomeArchive_.size(); i++) {
                    if ((baby.compatibilityDistance(pop.genomeArchive_[i],
                                                    parameters) < parameters.minDeltaCompatEqualGenomes)  // identical genome?
                    ) {
                        babyExistsInPop = true;
                        break;
                    }
                }
            }
        } while ((babyExistsInPop || baby.failsConstraints(parameters)) && (constraintTrials--));  // end do

        // We have a new offspring now give the offspring a new ID
        baby.setID(pop.getNextGenomeID());
        pop.incrementNextGenomeID();

        // sort the baby's genes
        baby.sortGenes();

        // clear the baby's fitness
        baby.setFitness(0);
        baby.setAdjFitness(0);
        baby.setOffspringAmount(0);

        baby.resetEvaluated();

        // In case of archiving, add the new baby to the archive
        if (parameters.archiveEnforcement) {
            pop.genomeArchive_.emplace_back(baby);
        }

#ifdef VDEBUG
        std::cout << "baby success\n";
#endif

        return baby;
    }

    // Mutates a genome
    void Species::mutateGenome(bool babyIsClone, Population &pop, Genome &baby, Parameters &parameters, RNG &rng) {
#if 0
        if ((rng.randFloat() < parameters.mutateAddNeuronProb) && ((pop.getSearchMode() == COMPLEXIFYING) || (pop.getSearchMode() == BLENDED)))
        {
            if (parameters.maxNeurons > 0)
            {
                if ((baby.numNeurons() - (baby.numInputs() + baby.numOutputs())) < parameters.maxNeurons)
                {
                    baby.mutateAddNeuron(pop.accessInnovationDatabase(), parameters, rng);
                }
            }
            else
            {
                baby.mutateAddNeuron(pop.accessInnovationDatabase(), parameters, rng);
            }
        }
        else if ((rng.randFloat() < parameters.mutateAddLinkProb) && ((pop.getSearchMode() == COMPLEXIFYING) || (pop.getSearchMode() == BLENDED)))
        {
            if (parameters.maxLinks > 0)
            {
                if (baby.numLinks() < parameters.maxLinks)
                {
                    baby.mutateAddLink(pop.accessInnovationDatabase(), parameters, rng);
                }
            }
            else
            {
                baby.mutateAddLink(pop.accessInnovationDatabase(), parameters, rng);
            }
        }
        else if ((rng.randFloat() < parameters.mutateRemSimpleNeuronProb) && ((pop.getSearchMode() == SIMPLIFYING) || (pop.getSearchMode() == BLENDED)))
        {
            baby.mutateRemoveSimpleNeuron(pop.accessInnovationDatabase(), parameters, rng);
        }
        else if ((rng.randFloat() < parameters.mutateRemLinkProb) && ((pop.getSearchMode() == SIMPLIFYING) || (pop.getSearchMode() == BLENDED)))
        {
            // Keep doing this mutation until it is sure that the baby will not end up having dead ends or no links
            Genome savedBaby = baby;
            bool noLinks = false, hasDeadEnds = false;

            int tries = 128;
            do
            {
                tries--;
                if (tries <= 0)
                {
                    savedBaby = baby;
                    break; // give up
                }
    
                savedBaby = baby;
                savedBaby.mutateRemoveLink(rng);
    
                noLinks = hasDeadEnds = false;
    
                if (savedBaby.numLinks() == 0)
                    noLinks = true;
    
                hasDeadEnds = savedBaby.hasDeadEnds();
    
            }
            while (noLinks || hasDeadEnds);

            baby = savedBaby;
        }
        else
        {
            if (rng.randFloat() < parameters.mutateNeuronActivationTypeProb)
            {
                baby.mutateNeuronActivationType(parameters, rng);
            }
    
            if (rng.randFloat() < parameters.mutateWeightsProb)
            {
                baby.mutateLinkWeights(parameters, rng);
            }
    
            if (rng.randFloat() < parameters.mutateActivationAProb)
            {
                baby.mutateNeuronActivationsA(parameters, rng);
            }
    
            if (rng.randFloat() < parameters.mutateActivationBProb)
            {
                baby.mutateNeuronActivationsB(parameters, rng);
            }
    
            if (rng.randFloat() < parameters.mutateNeuronTimeConstantsProb)
            {
                baby.mutateNeuronTimeConstants(parameters, rng);
            }
    
            if (rng.randFloat() < parameters.mutateNeuronBiasesProb)
            {
                baby.mutateNeuronBiases(parameters, rng);
            }
    
            if (rng.randFloat() < parameters.mutateNeuronTraitsProb)
            {
                baby.mutateNeuronTraits(parameters, rng);
            }
    
            if (rng.randFloat() < parameters.mutateLinkTraitsProb)
            {
                baby.mutateLinkTraits(parameters, rng);
            }
    
            if (rng.randFloat() < parameters.mutateGenomeTraitsProb)
            {
                baby.mutateGenomeTraits(parameters, rng);
            }
        }

#else
        // We will perform roulette wheel selection to choose the type of mutation and will mutate the baby This method guarantees that the baby will be mutated
        // at least with one mutation
        enum MutationTypes {
            ADD_NODE = 0,
            ADD_LINK,
            REMOVE_NODE,
            REMOVE_LINK,
            CHANGE_ACTIVATION_FUNCTION,
            MUTATE_WEIGHTS,
            MUTATE_ACTIVATION_A,
            MUTATE_ACTIVATION_B,
            MUTATE_TIMECONSTS,
            MUTATE_BIASES,
            MUTATE_NEURON_TRAITS,
            MUTATE_LINK_TRAITS,
            MUTATE_GENOME_TRAITS
        };
        std::vector<int> muts;
        std::vector<double> mutProbs;

        // ADD_NODE;
        mutProbs.emplace_back(parameters.mutateAddNeuronProb);

        // ADD_LINK;
        mutProbs.emplace_back(parameters.mutateAddLinkProb);

        // REMOVE_NODE;
        mutProbs.emplace_back(parameters.mutateRemSimpleNeuronProb);

        // REMOVE_LINK;
        mutProbs.emplace_back(parameters.mutateRemLinkProb);

        // CHANGE_ACTIVATION_FUNCTION;
        mutProbs.emplace_back(parameters.mutateNeuronActivationTypeProb);

        // MUTATE_WEIGHTS;
        mutProbs.emplace_back(parameters.mutateWeightsProb);

        // MUTATE_ACTIVATION_A;
        mutProbs.emplace_back(parameters.mutateActivationAProb);

        // MUTATE_ACTIVATION_B;
        mutProbs.emplace_back(parameters.mutateActivationBProb);

        // MUTATE_TIMECONSTS;
        mutProbs.emplace_back(parameters.mutateNeuronTimeConstantsProb);

        // MUTATE_BIASES;
        mutProbs.emplace_back(parameters.mutateNeuronBiasesProb);

        // MUTATE_NEURON_TRAITS;
        mutProbs.emplace_back(parameters.mutateNeuronTraitsProb);

        // MUTATE_LINK_TRAITS;
        mutProbs.emplace_back(parameters.mutateLinkTraitsProb);

        // MUTATE_GENOME_TRAITS;
        mutProbs.emplace_back(parameters.mutateGenomeTraitsProb);

        // Special consideration for phased searching - do not allow certain mutations depending on the search mode
        // also don't use additive mutations if we just want to get rid of the clones
        if ((pop.getSearchMode() == SIMPLIFYING) || babyIsClone) {
            mutProbs[ADD_NODE] = 0;  // add node
            mutProbs[ADD_LINK] = 0;  // add link
        }
        if ((pop.getSearchMode() == COMPLEXIFYING) || babyIsClone) {
            mutProbs[REMOVE_NODE] = 0;  // rem node
            mutProbs[REMOVE_LINK] = 0;  // rem link
        }

        bool mutationSuccess = false;

        // repeat until successful
        while (mutationSuccess == false) {
            int ChosenMutation = rng.roulette(mutProbs);

            // Now mutate based on the choice
            switch (ChosenMutation) {
                case ADD_NODE:
                    mutationSuccess = baby.mutateAddNeuron(pop.accessInnovationDatabase(), parameters, rng);
                    break;

                case ADD_LINK:
                    mutationSuccess = baby.mutateAddLink(pop.accessInnovationDatabase(), parameters, rng);
                    break;

                case REMOVE_NODE:
                    mutationSuccess = baby.mutateRemoveSimpleNeuron(pop.accessInnovationDatabase(), parameters, rng);
                    break;

                case REMOVE_LINK: {
                    // Keep doing this mutation until it is sure that the baby will not end up having dead ends or no links
                    Genome savedBaby = baby;
                    bool noLinks = false, hasDeadEnds = false;

                    int tries = 128;
                    do {
                        tries--;
                        if (tries <= 0) {
                            savedBaby = baby;
                            break;  // give up
                        }

                        savedBaby = baby;
                        mutationSuccess = savedBaby.mutateRemoveLink(rng);

                        noLinks = hasDeadEnds = false;

                        if (savedBaby.numLinks() == 0) noLinks = true;

                        hasDeadEnds = savedBaby.hasDeadEnds();

                    } while (noLinks || hasDeadEnds);

                    baby = savedBaby;
                } break;

                case CHANGE_ACTIVATION_FUNCTION:
                    mutationSuccess = baby.mutateNeuronActivationType(parameters, rng);
                    break;

                case MUTATE_WEIGHTS:
                    mutationSuccess = baby.mutateLinkWeights(parameters, rng);
                    break;

                case MUTATE_ACTIVATION_A:
                    mutationSuccess = baby.mutateNeuronActivationsA(parameters, rng);
                    break;

                case MUTATE_ACTIVATION_B:
                    mutationSuccess = baby.mutateNeuronActivationsB(parameters, rng);
                    break;

                case MUTATE_TIMECONSTS:
                    mutationSuccess = baby.mutateNeuronTimeConstants(parameters, rng);
                    break;

                case MUTATE_BIASES:
                    mutationSuccess = baby.mutateNeuronBiases(parameters, rng);
                    break;

                case MUTATE_NEURON_TRAITS:
                    mutationSuccess = baby.mutateNeuronTraits(parameters, rng);
                    break;

                case MUTATE_LINK_TRAITS:
                    mutationSuccess = baby.mutateLinkTraits(parameters, rng);
                    break;

                case MUTATE_GENOME_TRAITS:
                    mutationSuccess = baby.mutateGenomeTraits(parameters, rng);
                    break;

                default:
                    mutationSuccess = false;
                    break;
            }
        }
#endif
    }

}  // namespace NEAT
