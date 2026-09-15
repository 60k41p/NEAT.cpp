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
 * File:        Parameters.h
 * Description: Central evolution configuration: population/speciation control, GA rates, phased-search and
 *              novelty-search knobs, per-operator mutation probabilities and ranges, compatibility weights,
 *              ES-HyperNEAT quadtree settings and the universal-trait maps. Construct with defaults,
 *              tune fields directly, then persist with save()/load(). Some parsed fields are not consumed
 *              by the loop (competitive-coevolution kills, eliteFraction, non-default SelectionMode paths);
 *              they are preserved for file compatibility — verify a knob is read in src/Population.cpp,
 *              src/Species.cpp or src/Genome.cpp before relying on it.
 *
 * References: Stanley & Miikkulainen (2002) for speciation/compatibility coefficients; phased search and
 *             delta coding follow the MultiNEAT lineage; novelty knobs follow Lehman & Stanley (2011);
 *             ES-HyperNEAT quadtree knobs follow Risi & Stanley. On-disk format: tests/data/minimal.NEAT
 *             fixture plus tests/TestParameters.cpp round-trips. Intra-repo users: src/Population.h,
 *             src/Species.h, src/Genome.h.
 */

#pragma once

#include <map>

#include "Traits.h"
#include "Types.h"

namespace NEAT {

    // forward
    class Genome;

    //////////////////////////////////////////////
    // Every knob of the evolutionary run. All fields are plain data with working built-in defaults
    // (see reset()); the text format is "Name value" lines (see tests/data/minimal.NEAT).
    //////////////////////////////////////////////
    class Parameters {
       public:
        /////////////////////
        // Members
        /////////////////////

        ////////////////////
        // Basic parameters
        ////////////////////

        // Size of population
        unsigned int populationSize;

        // Controls the use of speciation. When off, the population will consist of only one species.
        bool speciation;

        // If true, this enables dynamic compatibility thresholding It will keep the number of species between MinSpecies and MaxSpecies
        bool dynamicCompatibility;

        // Minimum number of species
        unsigned int minSpecies;

        // Maximum number of species
        unsigned int maxSpecies;

        // Don't wipe the innovation database each generation?
        bool innovationsForever;

        // Allow clones or nearly identical genomes to exist simultaneously in the population.
        // This is useful for non-deterministic environments,
        // as the same individual will get more than one chance to prove himself, also
        // there will be more chances the same individual to mutate in different ways.
        // The drawback is greatly increased time for reproduction. If you want to
        // search quickly, yet less efficient, leave this to true.
        bool allowClones;

        // Keep an archive of genomes and don't allow any new genome to exist in the acrhive or the population
        bool archiveEnforcement;

        // Normalize genome size when calculating compatibility
        bool normalizeGenomeSize;

        // Pointer to a function that specifies custom topology constraints Should return true if the genome FAILS to meet the constraints
        bool (*customConstraints)(Genome &g);

        ////////////////////////////////
        // GA Parameters
        ////////////////////////////////

        // AgeGens treshold, meaning if a species is below it, it is considered young
        unsigned int youngAgeTreshold;

        // Fitness boost multiplier for young species (1.0 means no boost)
        // Make sure it is >= 1.0 to avoid confusion
        Real youngAgeFitnessBoost;

        // Number of generations without improvement (stagnation) allowed for a species
        unsigned int speciesMaxStagnation;

        // Minimum jump in fitness necessary to be considered as improvement. Setting this value to 0.0 makes the system to behave like regular NEAT.
        Real stagnationDelta;

        // AgeGens threshold, meaning if a species if above it, it is considered old
        unsigned int oldAgeTreshold;

        // Multiplier that penalizes old species.
        // Make sure it is < 1.0 to avoid confusion.
        Real oldAgePenalty;

        // Detect competetive coevolution stagnation
        // This kills the worst species of age >N (each X generations)
        bool detectCompetetiveCoevolutionStagnation;

        // Each X generation..
        int killWorstSpeciesEach;

        // Of age above..
        int killWorstAge;

        // Percent of best individuals that are allowed to reproduce. 1.0 = 100%
        Real survivalRate;

        // Probability for a baby to result from sexual reproduction (crossover/mating). 1.0 = 100%
        Real crossoverRate;

        // If a baby results from sexual reproduction, this probability determines if mutation will be performed after crossover. 1.0 = 100% (always mutate
        // after crossover)
        Real overallMutationRate;

        // Probability for a baby to result from inter-species mating.
        Real interspeciesCrossoverRate;

        // Probability for a baby gene to result from Multipoint Crossover when mating. 1.0 = 100% The default if the Average mating.
        Real multipointCrossoverRate;

        // Probability that when doing multipoint crossover,
        // the gene of the fitter parent will be prefered, instead of choosing one at random
        Real preferFitterParentRate;

        // Performing roulette wheel selection or not?
        bool rouletteWheelSelection;

        // If true, will do tournament selection
        bool tournamentSelection;

        // For tournament selection
        unsigned int tournamentSize;

        // Fraction of individuals to be copied unchanged
        Real eliteFraction;

        ///////////////////////////////////
        // Phased Search parameters   //
        ///////////////////////////////////

        // Using phased search or not
        bool phasedSearching;

        // Using delta coding or not
        bool deltaCoding;

        // What is the MPC + base MPC needed to begin simplifying phase
        unsigned int simplifyingPhaseMPCTreshold;

        // How many generations of global stagnation should have passed to enter simplifying phase
        unsigned int simplifyingPhaseStagnationTreshold;

        // How many generations of MPC stagnation are needed to turn back on complexifying
        unsigned int complexityFloorGenerations;

        /////////////////////////////////////
        // Novelty Search parameters       //
        /////////////////////////////////////

        // the K constant
        unsigned int noveltySearchK;

        // Sparseness treshold. Add to the archive if above
        Real noveltySearchPMin;

        // Dynamic Pmin?
        bool noveltySearchDynamicPMin;

        // How many evaluations should pass without adding to the archive in order to lower Pmin
        unsigned int noveltySearchNoArchivingStagnationThreshold;

        // How should it be multiplied (make it less than 1.0)
        Real noveltySearchPMinLoweringMultiplier;

        // Not lower than this value
        Real noveltySearchPMinMin;

        // How many one-after-another additions to the archive should
        // pass in order to raise Pmin
        unsigned int noveltySearchQuickArchivingMinEvaluations;

        // How should it be multiplied (make it more than 1.0)
        Real noveltySearchPMinRaisingMultiplier;

        // Per how many evaluations to recompute the sparseness
        unsigned int noveltySearchRecomputeSparsenessEach;

        ///////////////////////////////////
        // Mutation parameters
        ///////////////////////////////////

        // Probability for a baby to be mutated with the Add-Neuron mutation.
        Real mutateAddNeuronProb;

        // Allow splitting of any recurrent links
        bool splitRecurrent;

        // Allow splitting of looped recurrent links
        bool splitLoopedRecurrent;

        // Maximum number of tries to find a link to split
        int neuronTries;

        // Probability for a baby to be mutated with the Add-Link mutation
        Real mutateAddLinkProb;

        // Probability for a new incoming link to be from the bias neuron;
        Real mutateAddLinkFromBiasProb;

        // Probability for a baby to be mutated with the Remove-Link mutation
        Real mutateRemLinkProb;

        // Probability for a baby that a simple neuron will be replaced with a link
        Real mutateRemSimpleNeuronProb;

        // Maximum number of tries to find 2 neurons to add/remove a link
        unsigned int linkTries;

        // Maximum number of links in the genome (originals not counted). -1 is unlimited
        int maxLinks;

        // Maximum number of neurons in the genome (originals not counted). -1 is unlimited
        int maxNeurons;

        // Probability that a link mutation will be made recurrent
        Real recurrentProb;

        // Probability that a recurrent link mutation will be looped
        Real recurrentLoopProb;

        // Probability for a baby's weights to be mutated
        Real mutateWeightsProb;

        // Probability for a severe (shaking) weight mutation
        Real mutateWeightsSevereProb;

        // Probability for a particular gene to be mutated. 1.0 = 100%
        Real weightMutationRate;

        // Probability for a particular gene to be mutated via replacement of the weight. 1.0 = 100%
        Real weightReplacementRate;

        // Maximum perturbation for a weight mutation
        Real weightMutationMaxPower;

        // Maximum magnitude of a replaced weight
        Real weightReplacementMaxPower;

        // Maximum weight
        Real maxWeight;

        // Minimum weight
        Real minWeight;

        // Probability for a baby's A activation function parameters to be perturbed
        Real mutateActivationAProb;

        // Probability for a baby's B activation function parameters to be perturbed
        Real mutateActivationBProb;

        // Maximum magnitude for the A parameter perturbation
        Real activationAMutationMaxPower;

        // Maximum magnitude for the B parameter perturbation
        Real activationBMutationMaxPower;

        // Maximum magnitude for time costants perturbation
        Real timeConstantMutationMaxPower;

        // Maximum magnitude for biases perturbation
        Real biasMutationMaxPower;

        // Activation parameter A min/max
        Real minActivationA;
        Real maxActivationA;

        // Activation parameter B min/max
        Real minActivationB;
        Real maxActivationB;

        // Probability for a baby that an activation function type will be changed for a single neuron considered a structural mutation because of the large
        // impact on fitness
        Real mutateNeuronActivationTypeProb;

        // Probabilities for a particular activation function appearance
        Real activationFunctionSignedSigmoidProb;
        Real activationFunctionUnsignedSigmoidProb;
        Real activationFunctionTanhProb;
        Real activationFunctionTanhCubicProb;
        Real activationFunctionSignedStepProb;
        Real activationFunctionUnsignedStepProb;
        Real activationFunctionSignedGaussProb;
        Real activationFunctionUnsignedGaussProb;
        Real activationFunctionAbsProb;
        Real activationFunctionSignedSineProb;
        Real activationFunctionUnsignedSineProb;
        Real activationFunctionLinearProb;
        Real activationFunctionReluProb;
        Real activationFunctionSoftplusProb;

        // Probability for a baby's neuron time constant values to be mutated
        Real mutateNeuronTimeConstantsProb;

        // Probability for a baby's neuron bias values to be mutated
        Real mutateNeuronBiasesProb;

        // Time constant range
        Real minNeuronTimeConstant;
        Real maxNeuronTimeConstant;

        // Bias range
        Real minNeuronBias;
        Real maxNeuronBias;

        /////////////////////////////////////
        // Speciation parameters
        /////////////////////////////////////

        // Percent of disjoint genes importance
        Real disjointCoeff;

        // Percent of excess genes importance
        Real excessCoeff;

        // Node-specific activation parameter A difference importance
        Real activationADiffCoeff;

        // Node-specific activation parameter B difference importance
        Real activationBDiffCoeff;

        // Average weight difference importance
        Real weightDiffCoeff;

        // Average time constant difference importance
        Real timeConstantDiffCoeff;

        // Average bias difference importance
        Real biasDiffCoeff;

        // Activation function type difference importance
        Real activationFunctionDiffCoeff;

        // Compatibility treshold
        Real compatTreshold;

        // Minumal value of the compatibility treshold
        Real minCompatTreshold;

        // Modifier per generation for keeping the species stable
        Real compatTresholdModifier;

        // Per how many generations to change the treshold
        unsigned int compatTreshChangeIntervalGenerations;

        // Per how many evaluations to change the treshold
        unsigned int compatTreshChangeIntervalEvaluations;

        // What is the minimal difference needed for not to be a clone
        Real minDeltaCompatEqualGenomes;

        // How many times to test a genome for constraint failure or being a clone (when AllowClones=False)
        int constraintTrials;

        /////////////////////////////
        // Genome properties params
        /////////////////////////////

        // When true, don't have a special bias neuron and treat all inputs equal
        bool dontUseBiasNeuron;
        bool allowLoops;

        /////////////////////////////
        // ES HyperNEAT params
        /////////////////////////////

        Real divisionThreshold;

        Real varianceThreshold;

        // Used for Band prunning.
        Real bandThreshold;

        // Max and Min Depths of the quadtree
        unsigned int initialDepth;

        unsigned int maxDepth;

        // How many hidden layers before connecting nodes to output. At 0 there is one hidden layer. At 1, there are two and so on.
        unsigned int iterationLevel;

        // The Bias value for the CPPN queries.
        Real cppnBias;

        // Quadtree Dimensions
        // The range of the tree. Typically set to 2,
        Real width;
        Real height;

        // The (x, y) coordinates of the tree
        Real qtreeX;

        Real qtreeY;

        // Use Link Expression output
        bool leo;

        // Threshold above which a connection is expressed
        Real leoThreshold;

        // Use geometric seeding. Currently only along the X axis. 1
        bool leoSeed;
        bool geometrySeed;

        /////////////////////////////////////
        // Universal traits
        /////////////////////////////////////
        std::map<std::string, TraitParameters> neuronTraits;
        std::map<std::string, TraitParameters> linkTraits;
        std::map<std::string, TraitParameters> genomeTraits;
        Real mutateNeuronTraitsProb;
        Real mutateLinkTraitsProb;
        Real mutateGenomeTraitsProb;

        /////////////////////////////////////
        // Constructors
        /////////////////////////////////////

        // Loads built-in defaults (see reset()).
        Parameters();

        ////////////////////////////////////
        // Methods
        ////////////////////////////////////

        // Loads "Name value" lines from a file path / open stream. Returns 0 on success.
        int load(const char *filename);
        // Loads from an already-opened stream.
        int load(std::ifstream &dataFile);

        void save(const char *filename);
        // Appends the parameters to an already-opened file for writing.
        void save(FILE *fstream);

        // Restores built-in defaults.
        void reset();
    };

}  // namespace NEAT
