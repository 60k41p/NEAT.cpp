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
 * File:        Parameters.cpp
 * Description: Contains the implementation of the Parameters class and the global parameters object
 */

#include "Parameters.h"

#include <fstream>
#include <iostream>
#include <string>

namespace NEAT {

    // Load defaults
    void Parameters::reset() {
        ////////////////////
        // Basic parameters
        ////////////////////

        // Size of population
        populationSize = 300;

        // Speciation on/off
        speciation = true;

        // If true, this enables dynamic compatibility thresholding It will keep the number of species between MinSpecies and MaxSpecies
        dynamicCompatibility = true;

        // Minimum number of species
        minSpecies = 5;

        // Maximum number of species
        maxSpecies = 10;

        // Don't wipe the innovation database each generation?
        innovationsForever = true;

        // Allow clones or nearly identical genomes to exist simultaneously in the population.
        // This is useful for non-deterministic environments,
        // as the same individual will get more than one chance to prove himself, also
        // there will be more chances the same individual to mutate in different ways.
        // The drawback is greatly increased time for reproduction. If you want to
        // search quickly, yet less efficient, leave this to true.
        allowClones = true;

        // Keep an archive of genomes and don't allow any new genome to exist in the archive or the population
        archiveEnforcement = false;

        // When true, don't have a special bias neuron and treat all inputs equal
        dontUseBiasNeuron = false;

        // When false, this prevents any recurrent pathways in the genomes from forming
        allowLoops = true;

        // Normalize genome size when calculating compatibility
        normalizeGenomeSize = false;

        // Pointer to a function that specifies custom topology/trait constraints
        // Should return true if the genome FAILS to meet the constraints
        customConstraints = nullptr;

        ////////////////////////////////
        // GA Parameters
        ////////////////////////////////

        // AgeGens treshold, meaning if a species is below it, it is considered young
        youngAgeTreshold = 5;

        // Fitness boost multiplier for young species (1.0 means no boost)
        // Make sure it is >= 1.0 to avoid confusion
        youngAgeFitnessBoost = 1.1;

        // Number of generations or evaluations without improvement (stagnation) allowed for a species
        speciesMaxStagnation = 25000;

        // Minimum jump in fitness necessary to be considered as improvement. Setting this value to 0.0 makes the system to behave like regular NEAT.
        stagnationDelta = 0.0;

        // AgeGens threshold, meaning if a species is above it, it is considered old
        oldAgeTreshold = 30;

        // Multiplier that penalizes old species.
        // Make sure it is <= 1.0 to avoid confusion.
        oldAgePenalty = 0.5;

        // Detect competetive coevolution stagnation
        // This kills the worst species of age >N (each X generations)
        detectCompetetiveCoevolutionStagnation = false;
        // Each X generation..
        killWorstSpeciesEach = 15;
        // Of age above..
        killWorstAge = 10;

        // Percent of best individuals that are allowed to reproduce. 1.0 = 100%
        survivalRate = 0.2;

        // Probability for a baby to result from sexual reproduction (crossover/mating). 1.0 = 100%
        // If asexual reprodiction is chosen, the baby will be mutated 100%
        crossoverRate = 0.7;

        // If a baby results from sexual reproduction, this probability determines if mutation will be performed after crossover. 1.0 = 100% (always mutate
        // after crossover)
        overallMutationRate = 0.75;

        // Probability for a baby to result from inter-species mating.
        interspeciesCrossoverRate = 0.0001;

        // Probability for a baby to result from Multipoint Crossover when mating. 1.0 = 100% The default is the Average mating.
        multipointCrossoverRate = 0.75;

        // Probability that when doing multipoint crossover,
        // the gene of the fitter parent will be prefered, instead of choosing one at random
        preferFitterParentRate = 0.25;

        // Performing roulette wheel selection or not?
        rouletteWheelSelection = false;

        // If true, will do tournament selection
        tournamentSelection = true;

        // For tournament selection
        tournamentSize = 5;

        // Fraction of individuals to be copied unchanged
        eliteFraction = 0.000001;

        // How many times to test a genome for constraint failure or being a clone (when AllowClones=False)
        constraintTrials = 2000000;

        ///////////////////////////////////
        // Phased Search parameters   //
        ///////////////////////////////////

        // Using phased search or not
        phasedSearching = false;

        // Using delta coding or not
        deltaCoding = false;

        // What is the MPC + base MPC needed to begin simplifying phase
        simplifyingPhaseMPCTreshold = 20;

        // How many generations of global stagnation should have passed to enter simplifying phase
        simplifyingPhaseStagnationTreshold = 30;

        // How many generations of MPC stagnation are needed to turn back on complexifying
        complexityFloorGenerations = 40;

        /////////////////////////////////////
        // Novelty Search parameters       //
        /////////////////////////////////////

        // the K constant
        noveltySearchK = 15;

        // Sparseness treshold. Add to the archive if above
        noveltySearchPMin = 0.5;

        // Dynamic Pmin?
        noveltySearchDynamicPMin = true;

        // How many evaluations should pass without adding to the archive in order to lower Pmin
        noveltySearchNoArchivingStagnationThreshold = 150;

        // How should it be multiplied (make it less than 1.0)
        noveltySearchPMinLoweringMultiplier = 0.9;

        // Not lower than this value
        noveltySearchPMinMin = 0.05;

        // How many one-after-another additions to the archive should
        // pass in order to raise Pmin
        noveltySearchQuickArchivingMinEvaluations = 8;

        // How should it be multiplied (make it more than 1.0)
        noveltySearchPMinRaisingMultiplier = 1.1;

        // Per how many evaluations to recompute the sparseness of the population
        noveltySearchRecomputeSparsenessEach = 25;

        ///////////////////////////////////
        // Structural Mutation parameters
        ///////////////////////////////////

        // Probability for a baby to be mutated with the Add-Neuron mutation.
        mutateAddNeuronProb = 0.01;

        // Allow splitting of any recurrent links
        splitRecurrent = false;

        // Allow splitting of looped recurrent links
        splitLoopedRecurrent = false;

        // Probability for a baby to be mutated with the Add-Link mutation
        mutateAddLinkProb = 0.03;

        // Probability for a new incoming link to be from the bias neuron;
        // This enforces it. A value of 0.0 doesn't mean there will not be such links
        mutateAddLinkFromBiasProb = 0.0;

        // Probability for a baby to be mutated with the Remove-Link mutation
        mutateRemLinkProb = 0.0;

        // Probability for a baby that a simple neuron will be replaced with a link
        mutateRemSimpleNeuronProb = 0.0;

        // Maximum number of tries to find 2 neurons to add/remove a link
        linkTries = 64;

        // Maximum number of links in the genome (originals not counted). -1 is unlimited
        maxLinks = -1;

        // Maximum number of neurons in the genome (originals not counted). -1 is unlimited
        maxNeurons = -1;

        // Probability that a link mutation will be made recurrent
        recurrentProb = 0.25;

        // Probability that a recurrent link mutation will be looped
        recurrentLoopProb = 0.25;

        ///////////////////////////////////
        // Parameter Mutation parameters
        ///////////////////////////////////

        // Probability for a baby's weights to be mutated
        mutateWeightsProb = 0.90;

        // Probability for a severe (shaking) weight mutation
        mutateWeightsSevereProb = 0.25;

        // Probability for a particular gene's weight to be mutated. 1.0 = 100%
        weightMutationRate = 1.0;

        // Maximum perturbation for a weight mutation
        weightMutationMaxPower = 1.0;

        // Probability for a particular gene to be mutated via replacement of the weight. 1.0 = 100%
        weightReplacementRate = 0.2;

        // Maximum magnitude of a replaced weight
        weightReplacementMaxPower = 1.0;

        // Maximum weight
        maxWeight = 8.0;

        // Minimum weight
        minWeight = -8.0;

        // Probability for a baby's A activation function parameters to be perturbed
        mutateActivationAProb = 0.0;

        // Probability for a baby's B activation function parameters to be perturbed
        mutateActivationBProb = 0.0;

        // Maximum magnitude for the A parameter perturbation
        activationAMutationMaxPower = 0.0;

        // Maximum magnitude for the B parameter perturbation
        activationBMutationMaxPower = 0.0;

        // Activation parameter A min/max
        minActivationA = 1.0;
        maxActivationA = 1.0;

        // Activation parameter B min/max
        minActivationB = 0.0;
        maxActivationB = 0.0;

        // Maximum magnitude for time costants perturbation
        timeConstantMutationMaxPower = 0.0;

        // Maximum magnitude for biases perturbation
        biasMutationMaxPower = weightMutationMaxPower;

        // Probability for a baby's neuron time constant values to be mutated
        mutateNeuronTimeConstantsProb = 0.0;

        // Probability for a baby's neuron bias values to be mutated
        mutateNeuronBiasesProb = 0.0;

        // Time constant range
        minNeuronTimeConstant = 0.0;
        maxNeuronTimeConstant = 0.0;

        // Bias range
        minNeuronBias = 0.0;
        maxNeuronBias = 0.0;

        // Probability for a baby that an activation function type will be changed for a single neuron considered a structural mutation because of the large
        // impact on fitness
        mutateNeuronActivationTypeProb = 0.0;

        // Probabilities for a particular activation function appearance
        activationFunctionSignedSigmoidProb = 0.0;
        activationFunctionUnsignedSigmoidProb = 1.0;
        activationFunctionTanhProb = 0.0;
        activationFunctionTanhCubicProb = 0.0;
        activationFunctionSignedStepProb = 0.0;
        activationFunctionUnsignedStepProb = 0.0;
        activationFunctionSignedGaussProb = 0.0;
        activationFunctionUnsignedGaussProb = 0.0;
        activationFunctionAbsProb = 0.0;
        activationFunctionSignedSineProb = 0.0;
        activationFunctionUnsignedSineProb = 0.0;
        activationFunctionLinearProb = 0.0;
        activationFunctionReluProb = 0.0;
        activationFunctionSoftplusProb = 0.0;

        // Trait mutation probabilities
        mutateNeuronTraitsProb = 0.0;
        mutateLinkTraitsProb = 0.0;
        mutateGenomeTraitsProb = 0.0;

        /////////////////////////////
        // Genome properties params
        /////////////////////////////

        /////////////////////////////////////
        // Speciation parameters
        /////////////////////////////////////

        // Percent of disjoint genes importance
        disjointCoeff = 1.0;

        // Percent of excess genes importance
        excessCoeff = 1.0;

        // Average weight difference importance
        weightDiffCoeff = 0.5;

        // Node-specific activation parameter A difference importance
        activationADiffCoeff = 0.0;

        // Node-specific activation parameter B difference importance
        activationBDiffCoeff = 0.0;

        // Average time constant difference importance
        timeConstantDiffCoeff = 0.0;

        // Average bias difference importance
        biasDiffCoeff = 0.0;

        // Activation function type difference importance
        activationFunctionDiffCoeff = 0.0;

        // Compatibility treshold
        compatTreshold = 3.0;

        // Minumal value of the compatibility treshold
        minCompatTreshold = 0.0;

        // Modifier per generation for keeping the species stable
        compatTresholdModifier = 0.1;

        // Per how many generations to change the treshold (used in generational mode)
        compatTreshChangeIntervalGenerations = 1;

        // Per how many evaluations to change the treshold (used in steady state mode)
        compatTreshChangeIntervalEvaluations = 1;

        // Minimal distance for two individuals to be considered different (as in clones or not)
        minDeltaCompatEqualGenomes = 0.0000001;

        //////////////////////////////
        // ES-HyperNEAT parameters

        divisionThreshold = 0.03;

        varianceThreshold = 0.03;

        // Used for Band prunning.
        bandThreshold = 0.3;

        // Max and Min Depths of the quadtree
        initialDepth = 3;
        maxDepth = 3;

        // How many hidden layers before connecting nodes to output. At 0 there is one hidden layer. At 1, there are two and so on.
        iterationLevel = 1;

        // The Bias value for the CPPN queries.
        cppnBias = 1.0;

        // Quadtree Dimensions
        // The range of the tree. Typically set to 2,
        width = 2.0;

        height = 2.0;

        // The (x, y) coordinates of the tree
        qtreeX = 0.0;

        qtreeY = 0.0;

        // Use Link Expression output
        leo = false;

        // Threshold above which a connection is expressed
        leoThreshold = 0.1;

        // Use geometric seeding. Currently only along the X axis. 1
        leoSeed = false;

        geometrySeed = false;
    }

    Parameters::Parameters() { reset(); }

    int Parameters::load(std::ifstream &dataFile) {
        std::string s, tf;
        // EOF guard: extraction failure leaves s unchanged, so without this a
        // file missing the marker would spin forever.
        do {
            dataFile >> s;
            if (dataFile.eof()) {
                return 1;
            }
        } while (s != "NEAT_ParametersStart");

        while (s != "NEAT_ParametersEnd") {
            dataFile >> s;
            if (dataFile.eof()) {
                return 1;  // missing NEAT_ParametersEnd marker — would otherwise spin forever
            }

            if (s == "PopulationSize") dataFile >> populationSize;

            if (s == "Speciation") {
                dataFile >> tf;
                if (tf == "true" || tf == "1" || tf == "1.0")
                    speciation = true;
                else
                    speciation = false;
            }

            if (s == "DynamicCompatibility") {
                dataFile >> tf;
                if (tf == "true" || tf == "1" || tf == "1.0")
                    dynamicCompatibility = true;
                else
                    dynamicCompatibility = false;
            }

            if (s == "MinSpecies") dataFile >> minSpecies;

            if (s == "MaxSpecies") dataFile >> maxSpecies;

            if (s == "InnovationsForever") {
                dataFile >> tf;
                if (tf == "true" || tf == "1" || tf == "1.0")
                    innovationsForever = true;
                else
                    innovationsForever = false;
            }

            if (s == "AllowClones") {
                dataFile >> tf;
                if (tf == "true" || tf == "1" || tf == "1.0")
                    allowClones = true;
                else
                    allowClones = false;
            }

            if (s == "NormalizeGenomeSize") {
                dataFile >> tf;
                if (tf == "true" || tf == "1" || tf == "1.0")
                    normalizeGenomeSize = true;
                else
                    normalizeGenomeSize = false;
            }

            if (s == "ConstraintTrials") dataFile >> constraintTrials;

            if (s == "YoungAgeTreshold") dataFile >> youngAgeTreshold;

            if (s == "YoungAgeFitnessBoost") dataFile >> youngAgeFitnessBoost;

            if (s == "SpeciesMaxStagnation") dataFile >> speciesMaxStagnation;

            if (s == "StagnationDelta") dataFile >> stagnationDelta;

            if (s == "OldAgeTreshold") dataFile >> oldAgeTreshold;

            if (s == "OldAgePenalty") dataFile >> oldAgePenalty;

            if (s == "DetectCompetetiveCoevolutionStagnation") {
                dataFile >> tf;
                if (tf == "true" || tf == "1" || tf == "1.0")
                    detectCompetetiveCoevolutionStagnation = true;
                else
                    detectCompetetiveCoevolutionStagnation = false;
            }

            if (s == "KillWorstSpeciesEach") dataFile >> killWorstSpeciesEach;

            if (s == "KillWorstAge") dataFile >> killWorstAge;

            if (s == "SurvivalRate") dataFile >> survivalRate;

            if (s == "CrossoverRate") dataFile >> crossoverRate;

            if (s == "OverallMutationRate") dataFile >> overallMutationRate;

            if (s == "InterspeciesCrossoverRate") dataFile >> interspeciesCrossoverRate;

            if (s == "MultipointCrossoverRate") dataFile >> multipointCrossoverRate;

            if (s == "PreferFitterParentRate") dataFile >> preferFitterParentRate;

            if (s == "RouletteWheelSelection") {
                dataFile >> tf;
                if (tf == "true" || tf == "1" || tf == "1.0")
                    rouletteWheelSelection = true;
                else
                    rouletteWheelSelection = false;
            }

            if (s == "TournamentSelection") {
                dataFile >> tf;
                if (tf == "true" || tf == "1" || tf == "1.0")
                    tournamentSelection = true;
                else
                    tournamentSelection = false;
            }

            if (s == "PhasedSearching") {
                dataFile >> tf;
                if (tf == "true" || tf == "1" || tf == "1.0")
                    phasedSearching = true;
                else
                    phasedSearching = false;
            }

            if (s == "DeltaCoding") {
                dataFile >> tf;
                if (tf == "true" || tf == "1" || tf == "1.0")
                    deltaCoding = true;
                else
                    deltaCoding = false;
            }

            if (s == "SimplifyingPhaseMPCTreshold") dataFile >> simplifyingPhaseMPCTreshold;

            if (s == "SimplifyingPhaseStagnationTreshold") dataFile >> simplifyingPhaseStagnationTreshold;

            if (s == "ComplexityFloorGenerations") dataFile >> complexityFloorGenerations;

            if (s == "NoveltySearch_K") dataFile >> noveltySearchK;

            if (s == "NoveltySearch_P_min") dataFile >> noveltySearchPMin;

            if (s == "NoveltySearch_Dynamic_Pmin") {
                dataFile >> tf;
                if (tf == "true" || tf == "1" || tf == "1.0")
                    noveltySearchDynamicPMin = true;
                else
                    noveltySearchDynamicPMin = false;
            }

            if (s == "NoveltySearch_No_Archiving_Stagnation_Treshold") dataFile >> noveltySearchNoArchivingStagnationThreshold;

            if (s == "NoveltySearch_Pmin_lowering_multiplier") dataFile >> noveltySearchPMinLoweringMultiplier;

            if (s == "NoveltySearch_Pmin_min") dataFile >> noveltySearchPMinMin;

            if (s == "NoveltySearch_Quick_Archiving_Min_Evaluations") dataFile >> noveltySearchQuickArchivingMinEvaluations;

            if (s == "NoveltySearch_Pmin_raising_multiplier") dataFile >> noveltySearchPMinRaisingMultiplier;

            if (s == "NoveltySearch_Recompute_Sparseness_Each") dataFile >> noveltySearchRecomputeSparsenessEach;

            if (s == "MutateAddNeuronProb") dataFile >> mutateAddNeuronProb;

            if (s == "SplitRecurrent") {
                dataFile >> tf;
                if (tf == "true" || tf == "1" || tf == "1.0")
                    splitRecurrent = true;
                else
                    splitRecurrent = false;
            }

            if (s == "SplitLoopedRecurrent") {
                dataFile >> tf;
                if (tf == "true" || tf == "1" || tf == "1.0")
                    splitLoopedRecurrent = true;
                else
                    splitLoopedRecurrent = false;
            }

            if (s == "MutateAddLinkProb") dataFile >> mutateAddLinkProb;

            if (s == "MutateAddLinkFromBiasProb") dataFile >> mutateAddLinkFromBiasProb;

            if (s == "MutateRemLinkProb") dataFile >> mutateRemLinkProb;

            if (s == "MutateRemSimpleNeuronProb") dataFile >> mutateRemSimpleNeuronProb;

            if (s == "LinkTries") dataFile >> linkTries;

            if (s == "MaxLinks") dataFile >> maxLinks;
            if (s == "MaxNeurons") dataFile >> maxNeurons;

            if (s == "RecurrentProb") dataFile >> recurrentProb;

            if (s == "RecurrentLoopProb") dataFile >> recurrentLoopProb;

            if (s == "MutateWeightsProb") dataFile >> mutateWeightsProb;

            if (s == "MutateWeightsSevereProb") dataFile >> mutateWeightsSevereProb;

            if (s == "WeightMutationRate") dataFile >> weightMutationRate;

            if (s == "WeightMutationMaxPower") dataFile >> weightMutationMaxPower;

            if (s == "WeightReplacementRate") dataFile >> weightReplacementRate;

            if (s == "WeightReplacementMaxPower") dataFile >> weightReplacementMaxPower;

            if (s == "MaxWeight") dataFile >> maxWeight;

            if (s == "MinWeight") dataFile >> minWeight;

            if (s == "MutateActivationAProb") dataFile >> mutateActivationAProb;

            if (s == "MutateActivationBProb") dataFile >> mutateActivationBProb;

            if (s == "ActivationAMutationMaxPower") dataFile >> activationAMutationMaxPower;

            if (s == "ActivationBMutationMaxPower") dataFile >> activationBMutationMaxPower;

            if (s == "MinActivationA") dataFile >> minActivationA;

            if (s == "MaxActivationA") dataFile >> maxActivationA;

            if (s == "MinActivationB") dataFile >> minActivationB;

            if (s == "MaxActivationB") dataFile >> maxActivationB;

            if (s == "TimeConstantMutationMaxPower") dataFile >> timeConstantMutationMaxPower;

            if (s == "BiasMutationMaxPower") dataFile >> biasMutationMaxPower;

            if (s == "MutateNeuronTimeConstantsProb") dataFile >> mutateNeuronTimeConstantsProb;

            if (s == "MutateNeuronBiasesProb") dataFile >> mutateNeuronBiasesProb;

            if (s == "MinNeuronTimeConstant") dataFile >> minNeuronTimeConstant;

            if (s == "MaxNeuronTimeConstant") dataFile >> maxNeuronTimeConstant;

            if (s == "MinNeuronBias") dataFile >> minNeuronBias;

            if (s == "MaxNeuronBias") dataFile >> maxNeuronBias;

            if (s == "MutateNeuronActivationTypeProb") dataFile >> mutateNeuronActivationTypeProb;

            if (s == "ActivationFunction_SignedSigmoid_Prob") dataFile >> activationFunctionSignedSigmoidProb;
            if (s == "ActivationFunction_UnsignedSigmoid_Prob") dataFile >> activationFunctionUnsignedSigmoidProb;
            if (s == "ActivationFunction_Tanh_Prob") dataFile >> activationFunctionTanhProb;
            if (s == "ActivationFunction_TanhCubic_Prob") dataFile >> activationFunctionTanhCubicProb;
            if (s == "ActivationFunction_SignedStep_Prob") dataFile >> activationFunctionSignedStepProb;
            if (s == "ActivationFunction_UnsignedStep_Prob") dataFile >> activationFunctionUnsignedStepProb;
            if (s == "ActivationFunction_SignedGauss_Prob") dataFile >> activationFunctionSignedGaussProb;
            if (s == "ActivationFunction_UnsignedGauss_Prob") dataFile >> activationFunctionUnsignedGaussProb;
            if (s == "ActivationFunction_Abs_Prob") dataFile >> activationFunctionAbsProb;
            if (s == "ActivationFunction_SignedSine_Prob") dataFile >> activationFunctionSignedSineProb;
            if (s == "ActivationFunction_UnsignedSine_Prob") dataFile >> activationFunctionUnsignedSineProb;
            if (s == "ActivationFunction_Linear_Prob") dataFile >> activationFunctionLinearProb;
            if (s == "ActivationFunction_Relu_Prob") dataFile >> activationFunctionReluProb;
            if (s == "ActivationFunction_Softplus_Prob") dataFile >> activationFunctionSoftplusProb;

            if (s == "DontUseBiasNeuron") {
                dataFile >> tf;
                if (tf == "true" || tf == "1" || tf == "1.0")
                    dontUseBiasNeuron = true;
                else
                    dontUseBiasNeuron = false;
            }

            if (s == "AllowLoops") {
                dataFile >> tf;
                if (tf == "true" || tf == "1" || tf == "1.0")
                    allowLoops = true;
                else
                    allowLoops = false;
            }

            if (s == "ArchiveEnforcement") {
                dataFile >> tf;
                if (tf == "true" || tf == "1" || tf == "1.0")
                    archiveEnforcement = true;
                else
                    archiveEnforcement = false;
            }

            if (s == "DisjointCoeff") dataFile >> disjointCoeff;

            if (s == "ExcessCoeff") dataFile >> excessCoeff;

            if (s == "WeightDiffCoeff") dataFile >> weightDiffCoeff;

            if (s == "ActivationADiffCoeff") dataFile >> activationADiffCoeff;

            if (s == "ActivationBDiffCoeff") dataFile >> activationBDiffCoeff;

            if (s == "TimeConstantDiffCoeff") dataFile >> timeConstantDiffCoeff;

            if (s == "BiasDiffCoeff") dataFile >> biasDiffCoeff;

            if (s == "ActivationFunctionDiffCoeff") dataFile >> activationFunctionDiffCoeff;

            if (s == "CompatTreshold") dataFile >> compatTreshold;

            if (s == "MinCompatTreshold") dataFile >> minCompatTreshold;

            if (s == "CompatTresholdModifier") dataFile >> compatTresholdModifier;

            if (s == "CompatTreshChangeInterval_Generations") dataFile >> compatTreshChangeIntervalGenerations;

            if (s == "CompatTreshChangeInterval_Evaluations") dataFile >> compatTreshChangeIntervalEvaluations;

            if (s == "MinDeltaCompatEqualGenomes") dataFile >> minDeltaCompatEqualGenomes;

            if (s == "DivisionThreshold") dataFile >> divisionThreshold;

            if (s == "VarianceThreshold") dataFile >> varianceThreshold;

            if (s == "BandThreshold") dataFile >> bandThreshold;

            if (s == "InitialDepth") dataFile >> initialDepth;

            if (s == "MaxDepth") dataFile >> maxDepth;

            if (s == "IterationLevel") dataFile >> iterationLevel;

            if (s == "TournamentSize") dataFile >> tournamentSize;

            if (s == "CPPN_Bias") dataFile >> cppnBias;

            if (s == "Width") dataFile >> width;

            if (s == "Height") dataFile >> height;

            if (s == "Qtree_X") dataFile >> qtreeX;

            if (s == "Qtree_Y") dataFile >> qtreeY;

            if (s == "Leo") {
                dataFile >> tf;
                if (tf == "true" || tf == "1" || tf == "1.0")
                    leo = true;
                else
                    leo = false;
            }
            if (s == "GeometrySeed") {
                dataFile >> tf;
                if (tf == "true" || tf == "1" || tf == "1.0")
                    geometrySeed = true;
                else
                    geometrySeed = false;
            }

            if (s == "LeoThreshold") dataFile >> leoThreshold;

            if (s == "LeoSeed") {
                dataFile >> tf;
                if (tf == "true" || tf == "1" || tf == "1.0")
                    leoSeed = true;
                else
                    leoSeed = false;
            }
            if (s == "Elitism") {
                dataFile >> eliteFraction;
            }
        }

        return 0;
    }

    int Parameters::load(const char *fileName) {
        std::ifstream data(fileName);
        if (!data.is_open()) return 0;

        int result = load(data);
        data.close();
        return result;
    }

    void Parameters::save(const char *filename) {
        FILE *f = fopen(filename, "w");
        save(f);
        fclose(f);
    }

    void Parameters::save(FILE *fstream) {
        fprintf(fstream, "NEAT_ParametersStart\n");

        fprintf(fstream, "PopulationSize %d\n", populationSize);
        fprintf(fstream, "Speciation %s\n", speciation == true ? "true" : "false");
        fprintf(fstream, "DynamicCompatibility %s\n", dynamicCompatibility == true ? "true" : "false");
        fprintf(fstream, "MinSpecies %d\n", minSpecies);
        fprintf(fstream, "MaxSpecies %d\n", maxSpecies);
        fprintf(fstream, "InnovationsForever %s\n", innovationsForever == true ? "true" : "false");
        fprintf(fstream, "AllowClones %s\n", allowClones == true ? "true" : "false");
        fprintf(fstream, "NormalizeGenomeSize %s\n", normalizeGenomeSize == true ? "true" : "false");
        fprintf(fstream, "ConstraintTrials %d\n", constraintTrials);
        fprintf(fstream, "YoungAgeTreshold %d\n", youngAgeTreshold);
        fprintf(fstream, "YoungAgeFitnessBoost %3.20f\n", youngAgeFitnessBoost);
        fprintf(fstream, "SpeciesMaxStagnation %d\n", speciesMaxStagnation);
        fprintf(fstream, "StagnationDelta %3.20f\n", stagnationDelta);
        fprintf(fstream, "OldAgeTreshold %d\n", oldAgeTreshold);
        fprintf(fstream, "OldAgePenalty %3.20f\n", oldAgePenalty);
        fprintf(fstream, "DetectCompetetiveCoevolutionStagnation %s\n", detectCompetetiveCoevolutionStagnation == true ? "true" : "false");
        fprintf(fstream, "KillWorstSpeciesEach %d\n", killWorstSpeciesEach);
        fprintf(fstream, "KillWorstAge %d\n", killWorstAge);
        fprintf(fstream, "SurvivalRate %3.20f\n", survivalRate);
        fprintf(fstream, "CrossoverRate %3.20f\n", crossoverRate);
        fprintf(fstream, "OverallMutationRate %3.20f\n", overallMutationRate);
        fprintf(fstream, "InterspeciesCrossoverRate %3.20f\n", interspeciesCrossoverRate);
        fprintf(fstream, "MultipointCrossoverRate %3.20f\n", multipointCrossoverRate);
        fprintf(fstream, "PreferFitterParentRate %3.20f\n", preferFitterParentRate);
        fprintf(fstream, "RouletteWheelSelection %s\n", rouletteWheelSelection == true ? "true" : "false");
        fprintf(fstream, "PhasedSearching %s\n", phasedSearching == true ? "true" : "false");
        fprintf(fstream, "DeltaCoding %s\n", deltaCoding == true ? "true" : "false");
        fprintf(fstream, "SimplifyingPhaseMPCTreshold %d\n", simplifyingPhaseMPCTreshold);
        fprintf(fstream, "SimplifyingPhaseStagnationTreshold %d\n", simplifyingPhaseStagnationTreshold);
        fprintf(fstream, "ComplexityFloorGenerations %d\n", complexityFloorGenerations);
        fprintf(fstream, "NoveltySearch_K %d\n", noveltySearchK);
        fprintf(fstream, "NoveltySearch_P_min %3.20f\n", noveltySearchPMin);
        fprintf(fstream, "NoveltySearch_Dynamic_Pmin %s\n", noveltySearchDynamicPMin == true ? "true" : "false");
        fprintf(fstream, "NoveltySearch_No_Archiving_Stagnation_Treshold %d\n", noveltySearchNoArchivingStagnationThreshold);
        fprintf(fstream, "NoveltySearch_Pmin_lowering_multiplier %3.20f\n", noveltySearchPMinLoweringMultiplier);
        fprintf(fstream, "NoveltySearch_Pmin_min %3.20f\n", noveltySearchPMinMin);
        fprintf(fstream, "NoveltySearch_Quick_Archiving_Min_Evaluations %d\n", noveltySearchQuickArchivingMinEvaluations);
        fprintf(fstream, "NoveltySearch_Pmin_raising_multiplier %3.20f\n", noveltySearchPMinRaisingMultiplier);
        fprintf(fstream, "NoveltySearch_Recompute_Sparseness_Each %d\n", noveltySearchRecomputeSparsenessEach);
        fprintf(fstream, "MutateAddNeuronProb %3.20f\n", mutateAddNeuronProb);
        fprintf(fstream, "SplitRecurrent %s\n", splitRecurrent == true ? "true" : "false");
        fprintf(fstream, "SplitLoopedRecurrent %s\n", splitLoopedRecurrent == true ? "true" : "false");
        fprintf(fstream, "NeuronTries %d\n", neuronTries);
        fprintf(fstream, "MutateAddLinkProb %3.20f\n", mutateAddLinkProb);
        fprintf(fstream, "MutateAddLinkFromBiasProb %3.20f\n", mutateAddLinkFromBiasProb);
        fprintf(fstream, "MutateRemLinkProb %3.20f\n", mutateRemLinkProb);
        fprintf(fstream, "MutateRemSimpleNeuronProb %3.20f\n", mutateRemSimpleNeuronProb);
        fprintf(fstream, "LinkTries %d\n", linkTries);
        fprintf(fstream, "MaxLinks %d\n", maxLinks);
        fprintf(fstream, "MaxNeurons %d\n", maxNeurons);
        fprintf(fstream, "RecurrentProb %3.20f\n", recurrentProb);
        fprintf(fstream, "RecurrentLoopProb %3.20f\n", recurrentLoopProb);
        fprintf(fstream, "MutateWeightsProb %3.20f\n", mutateWeightsProb);
        fprintf(fstream, "MutateWeightsSevereProb %3.20f\n", mutateWeightsSevereProb);
        fprintf(fstream, "WeightMutationRate %3.20f\n", weightMutationRate);
        fprintf(fstream, "WeightMutationMaxPower %3.20f\n", weightMutationMaxPower);
        fprintf(fstream, "WeightReplacementRate %3.20f\n", weightReplacementRate);
        fprintf(fstream, "WeightReplacementMaxPower %3.20f\n", weightReplacementMaxPower);
        fprintf(fstream, "MaxWeight %3.20f\n", maxWeight);
        fprintf(fstream, "MinWeight %3.20f\n", minWeight);
        fprintf(fstream, "MutateActivationAProb %3.20f\n", mutateActivationAProb);
        fprintf(fstream, "MutateActivationBProb %3.20f\n", mutateActivationBProb);
        fprintf(fstream, "ActivationAMutationMaxPower %3.20f\n", activationAMutationMaxPower);
        fprintf(fstream, "ActivationBMutationMaxPower %3.20f\n", activationBMutationMaxPower);
        fprintf(fstream, "TimeConstantMutationMaxPower %3.20f\n", timeConstantMutationMaxPower);
        fprintf(fstream, "BiasMutationMaxPower %3.20f\n", biasMutationMaxPower);
        fprintf(fstream, "MinActivationA %3.20f\n", minActivationA);
        fprintf(fstream, "MaxActivationA %3.20f\n", maxActivationA);
        fprintf(fstream, "MinActivationB %3.20f\n", minActivationB);
        fprintf(fstream, "MaxActivationB %3.20f\n", maxActivationB);
        fprintf(fstream, "MutateNeuronActivationTypeProb %3.20f\n", mutateNeuronActivationTypeProb);
        fprintf(fstream, "ActivationFunction_SignedSigmoid_Prob %3.20f\n", activationFunctionSignedSigmoidProb);
        fprintf(fstream, "ActivationFunction_UnsignedSigmoid_Prob %3.20f\n", activationFunctionUnsignedSigmoidProb);
        fprintf(fstream, "ActivationFunction_Tanh_Prob %3.20f\n", activationFunctionTanhProb);
        fprintf(fstream, "ActivationFunction_TanhCubic_Prob %3.20f\n", activationFunctionTanhCubicProb);
        fprintf(fstream, "ActivationFunction_SignedStep_Prob %3.20f\n", activationFunctionSignedStepProb);
        fprintf(fstream, "ActivationFunction_UnsignedStep_Prob %3.20f\n", activationFunctionUnsignedStepProb);
        fprintf(fstream, "ActivationFunction_SignedGauss_Prob %3.20f\n", activationFunctionSignedGaussProb);
        fprintf(fstream, "ActivationFunction_UnsignedGauss_Prob %3.20f\n", activationFunctionUnsignedGaussProb);
        fprintf(fstream, "ActivationFunction_Abs_Prob %3.20f\n", activationFunctionAbsProb);
        fprintf(fstream, "ActivationFunction_SignedSine_Prob %3.20f\n", activationFunctionSignedSineProb);
        fprintf(fstream, "ActivationFunction_UnsignedSine_Prob %3.20f\n", activationFunctionUnsignedSineProb);
        fprintf(fstream, "ActivationFunction_Linear_Prob %3.20f\n", activationFunctionLinearProb);
        fprintf(fstream, "ActivationFunction_Relu_Prob %3.20f\n", activationFunctionReluProb);
        fprintf(fstream, "ActivationFunction_Softplus_Prob %3.20f\n", activationFunctionSoftplusProb);
        fprintf(fstream, "MutateNeuronTimeConstantsProb %3.20f\n", mutateNeuronTimeConstantsProb);
        fprintf(fstream, "MutateNeuronBiasesProb %3.20f\n", mutateNeuronBiasesProb);
        fprintf(fstream, "MinNeuronTimeConstant %3.20f\n", minNeuronTimeConstant);
        fprintf(fstream, "MaxNeuronTimeConstant %3.20f\n", maxNeuronTimeConstant);
        fprintf(fstream, "MinNeuronBias %3.20f\n", minNeuronBias);
        fprintf(fstream, "MaxNeuronBias %3.20f\n", maxNeuronBias);
        fprintf(fstream, "DontUseBiasNeuron %s\n", dontUseBiasNeuron == true ? "true" : "false");
        fprintf(fstream, "ArchiveEnforcement %s\n", archiveEnforcement == true ? "true" : "false");
        fprintf(fstream, "AllowLoops %s\n", allowLoops == true ? "true" : "false");
        fprintf(fstream, "DisjointCoeff %3.20f\n", disjointCoeff);
        fprintf(fstream, "ExcessCoeff %3.20f\n", excessCoeff);
        fprintf(fstream, "ActivationADiffCoeff %3.20f\n", activationADiffCoeff);
        fprintf(fstream, "ActivationBDiffCoeff %3.20f\n", activationBDiffCoeff);
        fprintf(fstream, "WeightDiffCoeff %3.20f\n", weightDiffCoeff);
        fprintf(fstream, "TimeConstantDiffCoeff %3.20f\n", timeConstantDiffCoeff);
        fprintf(fstream, "BiasDiffCoeff %3.20f\n", biasDiffCoeff);
        fprintf(fstream, "ActivationFunctionDiffCoeff %3.20f\n", activationFunctionDiffCoeff);
        fprintf(fstream, "CompatTreshold %3.20f\n", compatTreshold);
        fprintf(fstream, "MinCompatTreshold %3.20f\n", minCompatTreshold);
        fprintf(fstream, "CompatTresholdModifier %3.20f\n", compatTresholdModifier);
        fprintf(fstream, "CompatTreshChangeInterval_Generations %d\n", compatTreshChangeIntervalGenerations);
        fprintf(fstream, "CompatTreshChangeInterval_Evaluations %d\n", compatTreshChangeIntervalEvaluations);
        fprintf(fstream, "MinDeltaCompatEqualGenomes %3.20f\n", minDeltaCompatEqualGenomes);

        fprintf(fstream, "DivisionThreshold %3.20f\n", divisionThreshold);
        fprintf(fstream, "VarianceThreshold %3.20f\n", varianceThreshold);
        fprintf(fstream, "BandThreshold %3.20f\n", bandThreshold);
        fprintf(fstream, "InitialDepth %d\n", initialDepth);
        fprintf(fstream, "MaxDepth %d\n", maxDepth);
        fprintf(fstream, "IterationLevel %d\n", iterationLevel);
        fprintf(fstream, "TournamentSelection %s\n", tournamentSelection == true ? "true" : "false");
        fprintf(fstream, "TournamentSize %d\n", tournamentSize);
        fprintf(fstream, "CPPN_Bias %3.20f\n", cppnBias);
        fprintf(fstream, "Width %3.20f\n", width);
        fprintf(fstream, "Height %3.20f\n", height);
        fprintf(fstream, "Qtree_X %3.20f\n", qtreeX);
        fprintf(fstream, "Qtree_Y %3.20f\n", qtreeY);
        fprintf(fstream, "Leo %s\n", leo == true ? "true" : "false");
        fprintf(fstream, "LeoThreshold %3.20f\n", leoThreshold);
        fprintf(fstream, "LeoSeed %s\n", leoSeed == true ? "true" : "false");
        fprintf(fstream, "GeometrySeed %s\n", geometrySeed == true ? "true" : "false");
        fprintf(fstream, "Elitism %3.20f\n", eliteFraction);

        fprintf(fstream, "NEAT_ParametersEnd\n");
    }

}  // namespace NEAT
