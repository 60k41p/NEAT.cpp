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

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>

#include "FileIO.h"
#include "Serialization.h"

namespace NEAT {

    // Load defaults
    void Parameters::Reset() {
        ////////////////////
        // Basic parameters
        ////////////////////

        // Size of population
        PopulationSize = 300;

        // Speciation on/off
        Speciation = true;

        // If true, this enables dynamic compatibility thresholding It will keep the number of species between MinSpecies and MaxSpecies
        DynamicCompatibility = true;

        // Minimum number of species
        MinSpecies = 5;

        // Maximum number of species
        MaxSpecies = 10;

        // Don't wipe the innovation database each generation?
        InnovationsForever = true;

        // Allow clones or nearly identical genomes to exist simultaneously in the population.
        // This is useful for non-deterministic environments,
        // as the same individual will get more than one chance to prove himself, also
        // there will be more chances the same individual to mutate in different ways.
        // The drawback is greatly increased time for reproduction. If you want to
        // search quickly, yet less efficient, leave this to true.
        AllowClones = true;

        // Keep an archive of genomes and don't allow any new genome to exist in the archive or the population
        ArchiveEnforcement = false;

        // When true, don't have a special bias neuron and treat all inputs equal
        DontUseBiasNeuron = false;

        // When false, this prevents any recurrent pathways in the genomes from forming
        AllowLoops = true;

        // Normalize genome size when calculating compatibility
        NormalizeGenomeSize = false;

        // Pointer to a function that specifies custom topology/trait constraints
        // Should return true if the genome FAILS to meet the constraints
        CustomConstraints = NULL;
        m_CustomConstraintsFunction = {};

        ////////////////////////////////
        // GA Parameters
        ////////////////////////////////

        // AgeGens treshold, meaning if a species is below it, it is considered young
        YoungAgeTreshold = 15;

        // Fitness boost multiplier for young species (1.0 means no boost)
        // Make sure it is >= 1.0 to avoid confusion
        YoungAgeFitnessBoost = 1.1;

        // Number of generations or evaluations without improvement (stagnation) allowed for a species
        SpeciesMaxStagnation = 25000;

        // Minimum jump in fitness necessary to be considered as improvement. Setting this value to 0.0 makes the system to behave like regular NEAT.
        StagnationDelta = 0.0;

        // AgeGens threshold, meaning if a species is above it, it is considered old
        OldAgeTreshold = 80;

        // Multiplier that penalizes old species.
        // Make sure it is <= 1.0 to avoid confusion.
        OldAgePenalty = 0.75;

        // Detect competetive coevolution stagnation
        // This kills the worst species of age >N (each X generations)
        DetectCompetetiveCoevolutionStagnation = false;
        // Each X generation..
        KillWorstSpeciesEach = 15;
        // Of age above..
        KillWorstAge = 10;

        // Percent of best individuals that are allowed to reproduce. 1.0 = 100%
        SurvivalRate = 0.2;

        // Probability for a baby to result from sexual reproduction (crossover/mating). 1.0 = 100%
        // If asexual reprodiction is chosen, the baby will be mutated 100%
        CrossoverRate = 0.7;

        // If a baby results from sexual reproduction, this probability determines if mutation will be performed after crossover. 1.0 = 100% (always mutate
        // after crossover)
        OverallMutationRate = 0.75;

        // Probability for a baby to result from inter-species mating.
        InterspeciesCrossoverRate = 0.0001;

        // Probability for a baby to result from Multipoint Crossover when mating. 1.0 = 100% The default is the Average mating.
        MultipointCrossoverRate = 0.75;

        // Probability that when doing multipoint crossover,
        // the gene of the fitter parent will be prefered, instead of choosing one at random
        PreferFitterParentRate = 0.5;

        // Performing truncation selection or not? (goes first)
        TruncationSelection = true;

        // Performing roulette wheel selection or not?
        RouletteWheelSelection = false;

        // If true, will do tournament selection
        TournamentSelection = false;

        // For tournament selection
        TournamentSize = 5;

        // Fraction of individuals to be copied unchanged
        EliteFraction = 0.0001;

        // How many times to test a genome for constraint failure or being a clone (when AllowClones=False)
        ConstraintTrials = 2000000;

        ///////////////////////////////////
        // Phased Search parameters   //
        ///////////////////////////////////

        // Using phased search or not
        PhasedSearching = false;

        // Using delta coding or not
        DeltaCoding = false;

        // What is the MPC + base MPC needed to begin simplifying phase
        SimplifyingPhaseMPCTreshold = 20;

        // How many generations of global stagnation should have passed to enter simplifying phase
        SimplifyingPhaseStagnationTreshold = 30;

        // How many generations of MPC stagnation are needed to turn back on complexifying
        ComplexityFloorGenerations = 40;

        /////////////////////////////////////
        // Novelty Search parameters       //
        /////////////////////////////////////

        // the K constant
        NoveltySearch_K = 15;

        // Sparseness treshold. Add to the archive if above
        NoveltySearch_P_min = 0.5;

        // Dynamic Pmin?
        NoveltySearch_Dynamic_Pmin = true;

        // How many evaluations should pass without adding to the archive in order to lower Pmin
        NoveltySearch_No_Archiving_Stagnation_Treshold = 150;

        // How should it be multiplied (make it less than 1.0)
        NoveltySearch_Pmin_lowering_multiplier = 0.9;

        // Not lower than this value
        NoveltySearch_Pmin_min = 0.05;

        // How many one-after-another additions to the archive should
        // pass in order to raise Pmin
        NoveltySearch_Quick_Archiving_Min_Evaluations = 8;

        // How should it be multiplied (make it more than 1.0)
        NoveltySearch_Pmin_raising_multiplier = 1.1;

        // Per how many evaluations to recompute the sparseness of the population
        NoveltySearch_Recompute_Sparseness_Each = 25;

        ///////////////////////////////////
        // Structural Mutation parameters
        ///////////////////////////////////

        // Probability for a baby to be mutated with the Add-Neuron mutation.
        MutateAddNeuronProb = 0.01;

        // Allow splitting of any recurrent links
        SplitRecurrent = false;

        // Allow splitting of looped recurrent links
        SplitLoopedRecurrent = false;

        // Maximum number of tries to find a link to split
        NeuronTries = 64;

        // Probability for a baby to be mutated with the Add-Link mutation
        MutateAddLinkProb = 0.03;

        // Probability for a new incoming link to be from the bias neuron;
        // This enforces it. A value of 0.0 doesn't mean there will not be such links
        MutateAddLinkFromBiasProb = 0.01;

        // Probability for a baby to be mutated with the Remove-Link mutation
        MutateRemLinkProb = 0.0;

        // Probability for a baby that a simple neuron will be replaced with a link
        MutateRemSimpleNeuronProb = 0.0;

        // Maximum number of tries to find 2 neurons to add/remove a link
        LinkTries = 64;

        // Maximum number of links in the genome (originals not counted). -1 is unlimited
        MaxLinks = -1;

        // Maximum number of neurons in the genome (originals not counted). -1 is unlimited
        MaxNeurons = -1;

        // Probability that a link mutation will be made recurrent
        RecurrentProb = 0.2;

        // Probability that a recurrent link mutation will be looped
        RecurrentLoopProb = 0.5;

        ///////////////////////////////////
        // Parameter Mutation parameters
        ///////////////////////////////////

        // Probability for a baby's weights to be mutated
        MutateWeightsProb = 0.80;

        // Probability for a severe (shaking) weight mutation
        MutateWeightsSevereProb = 0.2;

        // Probability for a particular gene's weight to be mutated. 1.0 = 100%
        WeightMutationRate = 0.8;

        // Maximum perturbation for a weight mutation
        WeightMutationMaxPower = 1.5;

        // Probability for a particular gene to be mutated via replacement of the weight. 1.0 = 100%
        WeightReplacementRate = 0.2;

        // Maximum magnitude of a replaced weight
        WeightReplacementMaxPower = 3.0;

        // Maximum weight
        MaxWeight = 8.0;

        // Minimum weight
        MinWeight = -8.0;

        // Probability for a baby's A activation function parameters to be perturbed
        MutateActivationAProb = 0.0;

        // Probability for a baby's B activation function parameters to be perturbed
        MutateActivationBProb = 0.0;

        // Maximum magnitude for the A parameter perturbation
        ActivationAMutationMaxPower = 0.0;

        // Maximum magnitude for the B parameter perturbation
        ActivationBMutationMaxPower = 0.0;

        // Activation parameter A min/max
        MinActivationA = 4.9;
        MaxActivationA = 4.9;

        // Activation parameter B min/max
        MinActivationB = 0.0;
        MaxActivationB = 0.0;

        // Maximum magnitude for time costants perturbation
        TimeConstantMutationMaxPower = 0.0;

        // Maximum magnitude for biases perturbation
        BiasMutationMaxPower = WeightMutationMaxPower;

        // Probability for a baby's neuron time constant values to be mutated
        MutateNeuronTimeConstantsProb = 0.0;

        // Probability for a baby's neuron bias values to be mutated
        MutateNeuronBiasesProb = 0.0;

        // Time constant range
        MinNeuronTimeConstant = 0.0;
        MaxNeuronTimeConstant = 0.0;

        // Bias range
        MinNeuronBias = 0.0;
        MaxNeuronBias = 0.0;

        // Probability for a baby that an activation function type will be changed for a single neuron considered a structural mutation because of the large
        // impact on fitness
        MutateNeuronActivationTypeProb = 0.0;

        // Probabilities for a particular activation function appearance
        ActivationFunction_SignedSigmoid_Prob = 0.0;
        ActivationFunction_UnsignedSigmoid_Prob = 1.0;
        ActivationFunction_Tanh_Prob = 0.0;
        ActivationFunction_TanhCubic_Prob = 0.0;
        ActivationFunction_SignedStep_Prob = 0.0;
        ActivationFunction_UnsignedStep_Prob = 0.0;
        ActivationFunction_SignedGauss_Prob = 0.0;
        ActivationFunction_UnsignedGauss_Prob = 0.0;
        ActivationFunction_Abs_Prob = 0.0;
        ActivationFunction_SignedSine_Prob = 0.0;
        ActivationFunction_UnsignedSine_Prob = 0.0;
        ActivationFunction_Linear_Prob = 0.0;
        ActivationFunction_Relu_Prob = 0.0;
        ActivationFunction_Softplus_Prob = 0.0;

        // Trait mutation probabilities
        MutateNeuronTraitsProb = 0.0;
        MutateLinkTraitsProb = 0.0;
        MutateGenomeTraitsProb = 0.0;

        /////////////////////////////
        // Genome properties params
        /////////////////////////////

        /////////////////////////////////////
        // Speciation parameters
        /////////////////////////////////////

        // Percent of disjoint genes importance
        DisjointCoeff = 1.0;

        // Percent of excess genes importance
        ExcessCoeff = 1.0;

        // Average weight difference importance
        WeightDiffCoeff = 0.1;

        // Node-specific activation parameter A difference importance
        ActivationADiffCoeff = 0.0;

        // Node-specific activation parameter B difference importance
        ActivationBDiffCoeff = 0.0;

        // Average time constant difference importance
        TimeConstantDiffCoeff = 0.0;

        // Average bias difference importance
        BiasDiffCoeff = 0.0;

        // Activation function type difference importance
        ActivationFunctionDiffCoeff = 0.0;

        // Compatibility treshold
        CompatTreshold = 3.0;

        // Minumal value of the compatibility treshold
        MinCompatTreshold = 0.1;

        // Modifier per generation for keeping the species stable
        CompatTresholdModifier = 0.2;

        // Per how many generations to change the treshold (used in generational mode)
        CompatTreshChangeInterval_Generations = 1;

        // Per how many evaluations to change the treshold (used in steady state mode)
        CompatTreshChangeInterval_Evaluations = 1;

        // Minimal distance for two individuals to be considered different (as in clones or not)
        MinDeltaCompatEqualGenomes = 0.0000001;

        //////////////////////////////
        // ES-HyperNEAT parameters

        DivisionThreshold = 0.03;

        VarianceThreshold = 0.03;

        // Used for Band prunning.
        BandThreshold = 0.3;

        // Max and Min Depths of the quadtree
        InitialDepth = 3;
        MaxDepth = 3;

        // How many hidden layers before connecting nodes to output. At 0 there is one hidden layer. At 1, there are two and so on.
        IterationLevel = 1;

        // The Bias value for the CPPN queries.
        CPPN_Bias = 1.0;

        // Quadtree Dimensions
        // The range of the tree. Typically set to 2,
        Width = 2.0;

        Height = 2.0;

        // The (x, y) coordinates of the tree
        Qtree_X = 0.0;

        Qtree_Y = 0.0;

        // Use Link Expression output
        Leo = false;

        // Threshold above which a connection is expressed
        LeoThreshold = 0.1;

        // Use geometric seeding. Currently only along the X axis. 1
        LeoSeed = false;

        GeometrySeed = false;

        Depth = 2.0;
        Qtree_Z = 0.0;

        ActivationFunction_SpikingLIF_Prob = 0.0;
        ActivationFunction_SpikingAdaptiveLIF_Prob = 0.0;
        ActivationFunction_SpikingIzhikevich_Prob = 0.0;
        ActivationFunction_McCullochPitts_Prob = 0.0;

        MutateNeuronSpikingParametersProb = 0.0;
        MutateLinkSpikingParametersProb = 0.0;
        SpikingParameterMutationRate = 0.2;
        SpikingParameterMutationPower = 0.1;
        InitialMCPInhibitoryVetoProb = 1.0;
        MutateMCPInhibitoryVetoProb = 0.05;
        MinSpikingTimeConstant = 0.005;
        MaxSpikingTimeConstant = 0.05;
        MinSpikeThreshold = 0.5;
        MaxSpikeThreshold = 2.0;
        MinResetPotential = -0.5;
        MaxResetPotential = 0.5;
        MinRestingPotential = -0.5;
        MaxRestingPotential = 0.5;
        MinRefractoryPeriod = 0.0;
        MaxRefractoryPeriod = 0.01;
        MinMembraneResistance = 0.1;
        MaxMembraneResistance = 2.0;
        MinAdaptationTimeConstant = 0.02;
        MaxAdaptationTimeConstant = 1.0;
        MinAdaptationIncrement = 0.0;
        MaxAdaptationIncrement = 0.5;
        MinSpikeRateTimeConstant = 0.01;
        MaxSpikeRateTimeConstant = 0.2;
        MinIzhikevichA = 0.01;
        MaxIzhikevichA = 0.1;
        MinIzhikevichThreshold = 25.0;
        MaxIzhikevichThreshold = 35.0;
        MinIzhikevichB = 0.1;
        MaxIzhikevichB = 0.3;
        MinIzhikevichC = -80.0;
        MaxIzhikevichC = -50.0;
        MinIzhikevichD = 0.0;
        MaxIzhikevichD = 10.0;
        MinSynapticDelay = 0.0;
        MaxSynapticDelay = 0.02;
        MinSynapticTimeConstant = 0.001;
        MaxSynapticTimeConstant = 0.05;
        InitialSTDPEnabledProb = 0.0;
        MinSTDPPlus = 0.0;
        MaxSTDPPlus = 0.05;
        MinSTDPMinus = 0.0;
        MaxSTDPMinus = 0.05;
        MinSTDPTau = 0.005;
        MaxSTDPTau = 0.1;

        SpikingNeuronDiffCoeff = 0.0;
        SpikingLinkDiffCoeff = 0.0;

        ParentSelectionMode = LEGACY_SELECTION;
        RankSelectionPressure = 1.7;
        RankSelectionExponent = 4.0;
        BoltzmannTemperature = 1.0;

        SinglePointCrossoverRate = 0.0;
        BlendCrossoverRate = 0.0;
        SimulatedBinaryCrossoverRate = 0.0;
        CrossoverBlendAlpha = 0.5;
        CrossoverSBXEta = 10.0;

        WeightMutationDistribution = UNIFORM_MUTATION;
        WeightMutationSigma = 1.0;
        WeightMutationCauchyScale = 1.0;
        WeightMutationPolynomialEta = 20.0;

        SpeciesRepresentativeSelection = FIRST_REPRESENTATIVE;
        RepresentativeSelectionCandidates = 0;
        OffspringAllocation = LARGEST_REMAINDER;
        MinSpeciesSize = 0;
        SpeciesElitism = 0;
        StagnationPenalty = 0.0000001;

        CompatibilityThresholdControl = LEGACY_COMPATIBILITY_THRESHOLD;
        TargetSpecies = 0;
        CompatibilityThresholdGain = 0.25;
        MaxCompatTreshold = 1.0e9;

        RequireEvaluatedGenomes = false;
        RejectNonFiniteFitness = false;

        MutationOperatorsPerOffspring = 1.0;
        AdaptiveMutationStart = 0;
        AdaptiveMutationRate = 0.0;
        AdaptiveMutationMaxFactor = 1.0;
        FitnessScaling = SHIFTED_FITNESS_SCALING;
        FitnessRankPressure = 1.5;
        FitnessSigmaScale = 2.0;
        FitnessBoltzmannTemperature = 1.0;
    }

    Parameters::Parameters() { Reset(); }

    int Parameters::Load(std::istream &a_DataFile) {
        // Parse into a fresh-defaulted copy so unknown/missing keys resolve to
        // v2 defaults; only commit on success (v2 ReadParameters semantics).
        Parameters loaded;
        std::string s, tf;
        // EOF guard: extraction failure leaves s unchanged, so without this a
        // file missing the marker would spin forever.
        do {
            a_DataFile >> s;
            if (a_DataFile.eof()) {
                return 1;
            }
        } while (s != "NEAT_ParametersStart");

        while (s != "NEAT_ParametersEnd") {
            a_DataFile >> s;
            if (a_DataFile.eof()) {
                return 1;  // missing NEAT_ParametersEnd marker — would otherwise spin forever
            }

            if (s == "PopulationSize") a_DataFile >> loaded.PopulationSize;

            if (s == "Speciation") {
                a_DataFile >> tf;
                if (tf == "true" || tf == "1" || tf == "1.0")
                    loaded.Speciation = true;
                else
                    loaded.Speciation = false;
            }

            if (s == "DynamicCompatibility") {
                a_DataFile >> tf;
                if (tf == "true" || tf == "1" || tf == "1.0")
                    loaded.DynamicCompatibility = true;
                else
                    loaded.DynamicCompatibility = false;
            }

            if (s == "MinSpecies") a_DataFile >> loaded.MinSpecies;

            if (s == "MaxSpecies") a_DataFile >> loaded.MaxSpecies;

            if (s == "InnovationsForever") {
                a_DataFile >> tf;
                if (tf == "true" || tf == "1" || tf == "1.0")
                    loaded.InnovationsForever = true;
                else
                    loaded.InnovationsForever = false;
            }

            if (s == "AllowClones") {
                a_DataFile >> tf;
                if (tf == "true" || tf == "1" || tf == "1.0")
                    loaded.AllowClones = true;
                else
                    loaded.AllowClones = false;
            }

            if (s == "NormalizeGenomeSize") {
                a_DataFile >> tf;
                if (tf == "true" || tf == "1" || tf == "1.0")
                    loaded.NormalizeGenomeSize = true;
                else
                    loaded.NormalizeGenomeSize = false;
            }

            if (s == "ConstraintTrials") a_DataFile >> loaded.ConstraintTrials;

            if (s == "YoungAgeTreshold") a_DataFile >> loaded.YoungAgeTreshold;

            if (s == "YoungAgeFitnessBoost") a_DataFile >> loaded.YoungAgeFitnessBoost;

            if (s == "SpeciesMaxStagnation") a_DataFile >> loaded.SpeciesMaxStagnation;

            if (s == "StagnationDelta") a_DataFile >> loaded.StagnationDelta;

            if (s == "OldAgeTreshold") a_DataFile >> loaded.OldAgeTreshold;

            if (s == "OldAgePenalty") a_DataFile >> loaded.OldAgePenalty;

            if (s == "DetectCompetetiveCoevolutionStagnation") {
                a_DataFile >> tf;
                if (tf == "true" || tf == "1" || tf == "1.0")
                    loaded.DetectCompetetiveCoevolutionStagnation = true;
                else
                    loaded.DetectCompetetiveCoevolutionStagnation = false;
            }

            if (s == "KillWorstSpeciesEach") a_DataFile >> loaded.KillWorstSpeciesEach;

            if (s == "KillWorstAge") a_DataFile >> loaded.KillWorstAge;

            if (s == "SurvivalRate") a_DataFile >> loaded.SurvivalRate;

            if (s == "CrossoverRate") a_DataFile >> loaded.CrossoverRate;

            if (s == "OverallMutationRate") a_DataFile >> loaded.OverallMutationRate;

            if (s == "InterspeciesCrossoverRate") a_DataFile >> loaded.InterspeciesCrossoverRate;

            if (s == "MultipointCrossoverRate") a_DataFile >> loaded.MultipointCrossoverRate;

            if (s == "PreferFitterParentRate") a_DataFile >> loaded.PreferFitterParentRate;

            if (s == "RouletteWheelSelection") {
                a_DataFile >> tf;
                if (tf == "true" || tf == "1" || tf == "1.0")
                    loaded.RouletteWheelSelection = true;
                else
                    loaded.RouletteWheelSelection = false;
            }

            if (s == "TournamentSelection") {
                a_DataFile >> tf;
                if (tf == "true" || tf == "1" || tf == "1.0")
                    loaded.TournamentSelection = true;
                else
                    loaded.TournamentSelection = false;
            }

            if (s == "PhasedSearching") {
                a_DataFile >> tf;
                if (tf == "true" || tf == "1" || tf == "1.0")
                    loaded.PhasedSearching = true;
                else
                    loaded.PhasedSearching = false;
            }

            if (s == "DeltaCoding") {
                a_DataFile >> tf;
                if (tf == "true" || tf == "1" || tf == "1.0")
                    loaded.DeltaCoding = true;
                else
                    loaded.DeltaCoding = false;
            }

            if (s == "SimplifyingPhaseMPCTreshold") a_DataFile >> loaded.SimplifyingPhaseMPCTreshold;

            if (s == "SimplifyingPhaseStagnationTreshold") a_DataFile >> loaded.SimplifyingPhaseStagnationTreshold;

            if (s == "ComplexityFloorGenerations") a_DataFile >> loaded.ComplexityFloorGenerations;

            if (s == "NoveltySearch_K") a_DataFile >> loaded.NoveltySearch_K;

            if (s == "NoveltySearch_P_min") a_DataFile >> loaded.NoveltySearch_P_min;

            if (s == "NoveltySearch_Dynamic_Pmin") {
                a_DataFile >> tf;
                if (tf == "true" || tf == "1" || tf == "1.0")
                    loaded.NoveltySearch_Dynamic_Pmin = true;
                else
                    loaded.NoveltySearch_Dynamic_Pmin = false;
            }

            if (s == "NoveltySearch_No_Archiving_Stagnation_Treshold") a_DataFile >> loaded.NoveltySearch_No_Archiving_Stagnation_Treshold;

            if (s == "NoveltySearch_Pmin_lowering_multiplier") a_DataFile >> loaded.NoveltySearch_Pmin_lowering_multiplier;

            if (s == "NoveltySearch_Pmin_min") a_DataFile >> loaded.NoveltySearch_Pmin_min;

            if (s == "NoveltySearch_Quick_Archiving_Min_Evaluations") a_DataFile >> loaded.NoveltySearch_Quick_Archiving_Min_Evaluations;

            if (s == "NoveltySearch_Pmin_raising_multiplier") a_DataFile >> loaded.NoveltySearch_Pmin_raising_multiplier;

            if (s == "NoveltySearch_Recompute_Sparseness_Each") a_DataFile >> loaded.NoveltySearch_Recompute_Sparseness_Each;

            if (s == "MutateAddNeuronProb") a_DataFile >> loaded.MutateAddNeuronProb;

            if (s == "SplitRecurrent") {
                a_DataFile >> tf;
                if (tf == "true" || tf == "1" || tf == "1.0")
                    loaded.SplitRecurrent = true;
                else
                    loaded.SplitRecurrent = false;
            }

            if (s == "SplitLoopedRecurrent") {
                a_DataFile >> tf;
                if (tf == "true" || tf == "1" || tf == "1.0")
                    loaded.SplitLoopedRecurrent = true;
                else
                    loaded.SplitLoopedRecurrent = false;
            }

            if (s == "MutateAddLinkProb") a_DataFile >> loaded.MutateAddLinkProb;

            if (s == "MutateAddLinkFromBiasProb") a_DataFile >> loaded.MutateAddLinkFromBiasProb;

            if (s == "MutateRemLinkProb") a_DataFile >> loaded.MutateRemLinkProb;

            if (s == "MutateRemSimpleNeuronProb") a_DataFile >> loaded.MutateRemSimpleNeuronProb;

            if (s == "LinkTries") a_DataFile >> loaded.LinkTries;

            if (s == "MaxLinks") a_DataFile >> loaded.MaxLinks;
            if (s == "MaxNeurons") a_DataFile >> loaded.MaxNeurons;

            if (s == "RecurrentProb") a_DataFile >> loaded.RecurrentProb;

            if (s == "RecurrentLoopProb") a_DataFile >> loaded.RecurrentLoopProb;

            if (s == "MutateWeightsProb") a_DataFile >> loaded.MutateWeightsProb;

            if (s == "MutateWeightsSevereProb") a_DataFile >> loaded.MutateWeightsSevereProb;

            if (s == "WeightMutationRate") a_DataFile >> loaded.WeightMutationRate;

            if (s == "WeightMutationMaxPower") a_DataFile >> loaded.WeightMutationMaxPower;

            if (s == "WeightReplacementRate") a_DataFile >> loaded.WeightReplacementRate;

            if (s == "WeightReplacementMaxPower") a_DataFile >> loaded.WeightReplacementMaxPower;

            if (s == "MaxWeight") a_DataFile >> loaded.MaxWeight;

            if (s == "MinWeight") a_DataFile >> loaded.MinWeight;

            if (s == "MutateActivationAProb") a_DataFile >> loaded.MutateActivationAProb;

            if (s == "MutateActivationBProb") a_DataFile >> loaded.MutateActivationBProb;

            if (s == "ActivationAMutationMaxPower") a_DataFile >> loaded.ActivationAMutationMaxPower;

            if (s == "ActivationBMutationMaxPower") a_DataFile >> loaded.ActivationBMutationMaxPower;

            if (s == "MinActivationA") a_DataFile >> loaded.MinActivationA;

            if (s == "MaxActivationA") a_DataFile >> loaded.MaxActivationA;

            if (s == "MinActivationB") a_DataFile >> loaded.MinActivationB;

            if (s == "MaxActivationB") a_DataFile >> loaded.MaxActivationB;

            if (s == "TimeConstantMutationMaxPower") a_DataFile >> loaded.TimeConstantMutationMaxPower;

            if (s == "BiasMutationMaxPower") a_DataFile >> loaded.BiasMutationMaxPower;

            if (s == "MutateNeuronTimeConstantsProb") a_DataFile >> loaded.MutateNeuronTimeConstantsProb;

            if (s == "MutateNeuronBiasesProb") a_DataFile >> loaded.MutateNeuronBiasesProb;

            if (s == "MinNeuronTimeConstant") a_DataFile >> loaded.MinNeuronTimeConstant;

            if (s == "MaxNeuronTimeConstant") a_DataFile >> loaded.MaxNeuronTimeConstant;

            if (s == "MinNeuronBias") a_DataFile >> loaded.MinNeuronBias;

            if (s == "MaxNeuronBias") a_DataFile >> loaded.MaxNeuronBias;

            if (s == "MutateNeuronActivationTypeProb") a_DataFile >> loaded.MutateNeuronActivationTypeProb;

            if (s == "ActivationFunction_SignedSigmoid_Prob") a_DataFile >> loaded.ActivationFunction_SignedSigmoid_Prob;
            if (s == "ActivationFunction_UnsignedSigmoid_Prob") a_DataFile >> loaded.ActivationFunction_UnsignedSigmoid_Prob;
            if (s == "ActivationFunction_Tanh_Prob") a_DataFile >> loaded.ActivationFunction_Tanh_Prob;
            if (s == "ActivationFunction_TanhCubic_Prob") a_DataFile >> loaded.ActivationFunction_TanhCubic_Prob;
            if (s == "ActivationFunction_SignedStep_Prob") a_DataFile >> loaded.ActivationFunction_SignedStep_Prob;
            if (s == "ActivationFunction_UnsignedStep_Prob") a_DataFile >> loaded.ActivationFunction_UnsignedStep_Prob;
            if (s == "ActivationFunction_SignedGauss_Prob") a_DataFile >> loaded.ActivationFunction_SignedGauss_Prob;
            if (s == "ActivationFunction_UnsignedGauss_Prob") a_DataFile >> loaded.ActivationFunction_UnsignedGauss_Prob;
            if (s == "ActivationFunction_Abs_Prob") a_DataFile >> loaded.ActivationFunction_Abs_Prob;
            if (s == "ActivationFunction_SignedSine_Prob") a_DataFile >> loaded.ActivationFunction_SignedSine_Prob;
            if (s == "ActivationFunction_UnsignedSine_Prob") a_DataFile >> loaded.ActivationFunction_UnsignedSine_Prob;
            if (s == "ActivationFunction_Linear_Prob") a_DataFile >> loaded.ActivationFunction_Linear_Prob;
            if (s == "ActivationFunction_Relu_Prob") a_DataFile >> loaded.ActivationFunction_Relu_Prob;
            if (s == "ActivationFunction_Softplus_Prob") a_DataFile >> loaded.ActivationFunction_Softplus_Prob;

            if (s == "DontUseBiasNeuron") {
                a_DataFile >> tf;
                if (tf == "true" || tf == "1" || tf == "1.0")
                    loaded.DontUseBiasNeuron = true;
                else
                    loaded.DontUseBiasNeuron = false;
            }

            if (s == "AllowLoops") {
                a_DataFile >> tf;
                if (tf == "true" || tf == "1" || tf == "1.0")
                    loaded.AllowLoops = true;
                else
                    loaded.AllowLoops = false;
            }

            if (s == "ArchiveEnforcement") {
                a_DataFile >> tf;
                if (tf == "true" || tf == "1" || tf == "1.0")
                    loaded.ArchiveEnforcement = true;
                else
                    loaded.ArchiveEnforcement = false;
            }

            if (s == "DisjointCoeff") a_DataFile >> loaded.DisjointCoeff;

            if (s == "ExcessCoeff") a_DataFile >> loaded.ExcessCoeff;

            if (s == "WeightDiffCoeff") a_DataFile >> loaded.WeightDiffCoeff;

            if (s == "ActivationADiffCoeff") a_DataFile >> loaded.ActivationADiffCoeff;

            if (s == "ActivationBDiffCoeff") a_DataFile >> loaded.ActivationBDiffCoeff;

            if (s == "TimeConstantDiffCoeff") a_DataFile >> loaded.TimeConstantDiffCoeff;

            if (s == "BiasDiffCoeff") a_DataFile >> loaded.BiasDiffCoeff;

            if (s == "ActivationFunctionDiffCoeff") a_DataFile >> loaded.ActivationFunctionDiffCoeff;

            if (s == "CompatTreshold") a_DataFile >> loaded.CompatTreshold;

            if (s == "MinCompatTreshold") a_DataFile >> loaded.MinCompatTreshold;

            if (s == "CompatTresholdModifier") a_DataFile >> loaded.CompatTresholdModifier;

            if (s == "CompatTreshChangeInterval_Generations") a_DataFile >> loaded.CompatTreshChangeInterval_Generations;

            if (s == "CompatTreshChangeInterval_Evaluations") a_DataFile >> loaded.CompatTreshChangeInterval_Evaluations;

            if (s == "MinDeltaCompatEqualGenomes") a_DataFile >> loaded.MinDeltaCompatEqualGenomes;

            if (s == "DivisionThreshold") a_DataFile >> loaded.DivisionThreshold;

            if (s == "VarianceThreshold") a_DataFile >> loaded.VarianceThreshold;

            if (s == "BandThreshold") a_DataFile >> loaded.BandThreshold;

            if (s == "InitialDepth") a_DataFile >> loaded.InitialDepth;

            if (s == "MaxDepth") a_DataFile >> loaded.MaxDepth;

            if (s == "IterationLevel") a_DataFile >> loaded.IterationLevel;

            if (s == "TournamentSize") a_DataFile >> loaded.TournamentSize;

            if (s == "CPPN_Bias") a_DataFile >> loaded.CPPN_Bias;

            if (s == "Width") a_DataFile >> loaded.Width;

            if (s == "Height") a_DataFile >> loaded.Height;

            if (s == "Qtree_X") a_DataFile >> loaded.Qtree_X;

            if (s == "Qtree_Y") a_DataFile >> loaded.Qtree_Y;

            if (s == "Leo") {
                a_DataFile >> tf;
                if (tf == "true" || tf == "1" || tf == "1.0")
                    loaded.Leo = true;
                else
                    loaded.Leo = false;
            }
            if (s == "GeometrySeed") {
                a_DataFile >> tf;
                if (tf == "true" || tf == "1" || tf == "1.0")
                    loaded.GeometrySeed = true;
                else
                    loaded.GeometrySeed = false;
            }

            if (s == "LeoThreshold") a_DataFile >> loaded.LeoThreshold;

            if (s == "LeoSeed") {
                a_DataFile >> tf;
                if (tf == "true" || tf == "1" || tf == "1.0")
                    loaded.LeoSeed = true;
                else
                    loaded.LeoSeed = false;
            }
            if (s == "Elitism") {
                a_DataFile >> loaded.EliteFraction;
            }
            if (s == "EliteFraction") {
                a_DataFile >> loaded.EliteFraction;
            }
            if (s == "TruncationSelection") {
                a_DataFile >> tf;
                if (tf == "true" || tf == "1" || tf == "1.0")
                    loaded.TruncationSelection = true;
                else
                    loaded.TruncationSelection = false;
            }
            if (s == "Depth") a_DataFile >> loaded.Depth;
            if (s == "Qtree_Z") a_DataFile >> loaded.Qtree_Z;
            if (s == "ActivationFunction_SpikingLIF_Prob") a_DataFile >> loaded.ActivationFunction_SpikingLIF_Prob;
            if (s == "ActivationFunction_SpikingAdaptiveLIF_Prob") a_DataFile >> loaded.ActivationFunction_SpikingAdaptiveLIF_Prob;
            if (s == "ActivationFunction_SpikingIzhikevich_Prob") a_DataFile >> loaded.ActivationFunction_SpikingIzhikevich_Prob;
            if (s == "ActivationFunction_McCullochPitts_Prob") a_DataFile >> loaded.ActivationFunction_McCullochPitts_Prob;
            if (s == "MutateNeuronSpikingParametersProb") a_DataFile >> loaded.MutateNeuronSpikingParametersProb;
            if (s == "MutateLinkSpikingParametersProb") a_DataFile >> loaded.MutateLinkSpikingParametersProb;
            if (s == "SpikingParameterMutationRate") a_DataFile >> loaded.SpikingParameterMutationRate;
            if (s == "SpikingParameterMutationPower") a_DataFile >> loaded.SpikingParameterMutationPower;
            if (s == "InitialMCPInhibitoryVetoProb") a_DataFile >> loaded.InitialMCPInhibitoryVetoProb;
            if (s == "MutateMCPInhibitoryVetoProb") a_DataFile >> loaded.MutateMCPInhibitoryVetoProb;
            if (s == "MinSpikingTimeConstant") a_DataFile >> loaded.MinSpikingTimeConstant;
            if (s == "MaxSpikingTimeConstant") a_DataFile >> loaded.MaxSpikingTimeConstant;
            if (s == "MinSpikeThreshold") a_DataFile >> loaded.MinSpikeThreshold;
            if (s == "MaxSpikeThreshold") a_DataFile >> loaded.MaxSpikeThreshold;
            if (s == "MinResetPotential") a_DataFile >> loaded.MinResetPotential;
            if (s == "MaxResetPotential") a_DataFile >> loaded.MaxResetPotential;
            if (s == "MinRestingPotential") a_DataFile >> loaded.MinRestingPotential;
            if (s == "MaxRestingPotential") a_DataFile >> loaded.MaxRestingPotential;
            if (s == "MinRefractoryPeriod") a_DataFile >> loaded.MinRefractoryPeriod;
            if (s == "MaxRefractoryPeriod") a_DataFile >> loaded.MaxRefractoryPeriod;
            if (s == "MinMembraneResistance") a_DataFile >> loaded.MinMembraneResistance;
            if (s == "MaxMembraneResistance") a_DataFile >> loaded.MaxMembraneResistance;
            if (s == "MinAdaptationTimeConstant") a_DataFile >> loaded.MinAdaptationTimeConstant;
            if (s == "MaxAdaptationTimeConstant") a_DataFile >> loaded.MaxAdaptationTimeConstant;
            if (s == "MinAdaptationIncrement") a_DataFile >> loaded.MinAdaptationIncrement;
            if (s == "MaxAdaptationIncrement") a_DataFile >> loaded.MaxAdaptationIncrement;
            if (s == "MinSpikeRateTimeConstant") a_DataFile >> loaded.MinSpikeRateTimeConstant;
            if (s == "MaxSpikeRateTimeConstant") a_DataFile >> loaded.MaxSpikeRateTimeConstant;
            if (s == "MinIzhikevichA") a_DataFile >> loaded.MinIzhikevichA;
            if (s == "MaxIzhikevichA") a_DataFile >> loaded.MaxIzhikevichA;
            if (s == "MinIzhikevichThreshold") a_DataFile >> loaded.MinIzhikevichThreshold;
            if (s == "MaxIzhikevichThreshold") a_DataFile >> loaded.MaxIzhikevichThreshold;
            if (s == "MinIzhikevichB") a_DataFile >> loaded.MinIzhikevichB;
            if (s == "MaxIzhikevichB") a_DataFile >> loaded.MaxIzhikevichB;
            if (s == "MinIzhikevichC") a_DataFile >> loaded.MinIzhikevichC;
            if (s == "MaxIzhikevichC") a_DataFile >> loaded.MaxIzhikevichC;
            if (s == "MinIzhikevichD") a_DataFile >> loaded.MinIzhikevichD;
            if (s == "MaxIzhikevichD") a_DataFile >> loaded.MaxIzhikevichD;
            if (s == "MinSynapticDelay") a_DataFile >> loaded.MinSynapticDelay;
            if (s == "MaxSynapticDelay") a_DataFile >> loaded.MaxSynapticDelay;
            if (s == "MinSynapticTimeConstant") a_DataFile >> loaded.MinSynapticTimeConstant;
            if (s == "MaxSynapticTimeConstant") a_DataFile >> loaded.MaxSynapticTimeConstant;
            if (s == "InitialSTDPEnabledProb") a_DataFile >> loaded.InitialSTDPEnabledProb;
            if (s == "MinSTDPPlus") a_DataFile >> loaded.MinSTDPPlus;
            if (s == "MaxSTDPPlus") a_DataFile >> loaded.MaxSTDPPlus;
            if (s == "MinSTDPMinus") a_DataFile >> loaded.MinSTDPMinus;
            if (s == "MaxSTDPMinus") a_DataFile >> loaded.MaxSTDPMinus;
            if (s == "MinSTDPTau") a_DataFile >> loaded.MinSTDPTau;
            if (s == "MaxSTDPTau") a_DataFile >> loaded.MaxSTDPTau;
            if (s == "SpikingNeuronDiffCoeff") a_DataFile >> loaded.SpikingNeuronDiffCoeff;
            if (s == "SpikingLinkDiffCoeff") a_DataFile >> loaded.SpikingLinkDiffCoeff;
            if (s == "ParentSelectionMode") {
                int mode = LEGACY_SELECTION;
                a_DataFile >> mode;
                loaded.ParentSelectionMode = static_cast<SelectionMode>(mode);
            }
            if (s == "RankSelectionPressure") a_DataFile >> loaded.RankSelectionPressure;
            if (s == "RankSelectionExponent") a_DataFile >> loaded.RankSelectionExponent;
            if (s == "BoltzmannTemperature") a_DataFile >> loaded.BoltzmannTemperature;
            if (s == "SinglePointCrossoverRate") a_DataFile >> loaded.SinglePointCrossoverRate;
            if (s == "BlendCrossoverRate") a_DataFile >> loaded.BlendCrossoverRate;
            if (s == "SimulatedBinaryCrossoverRate") a_DataFile >> loaded.SimulatedBinaryCrossoverRate;
            if (s == "CrossoverBlendAlpha") a_DataFile >> loaded.CrossoverBlendAlpha;
            if (s == "CrossoverSBXEta") a_DataFile >> loaded.CrossoverSBXEta;
            if (s == "WeightMutationDistribution") {
                int mode = UNIFORM_MUTATION;
                a_DataFile >> mode;
                loaded.WeightMutationDistribution = static_cast<WeightMutationMode>(mode);
            }
            if (s == "WeightMutationSigma") a_DataFile >> loaded.WeightMutationSigma;
            if (s == "WeightMutationCauchyScale") a_DataFile >> loaded.WeightMutationCauchyScale;
            if (s == "WeightMutationPolynomialEta") a_DataFile >> loaded.WeightMutationPolynomialEta;
            if (s == "SpeciesRepresentativeSelection") {
                int mode = FIRST_REPRESENTATIVE;
                a_DataFile >> mode;
                loaded.SpeciesRepresentativeSelection = static_cast<SpeciesRepresentativeMode>(mode);
            }
            if (s == "RepresentativeSelectionCandidates") a_DataFile >> loaded.RepresentativeSelectionCandidates;
            if (s == "OffspringAllocation") {
                int mode = LARGEST_REMAINDER;
                a_DataFile >> mode;
                loaded.OffspringAllocation = static_cast<OffspringAllocationMode>(mode);
            }
            if (s == "MinSpeciesSize") a_DataFile >> loaded.MinSpeciesSize;
            if (s == "SpeciesElitism") a_DataFile >> loaded.SpeciesElitism;
            if (s == "StagnationPenalty") a_DataFile >> loaded.StagnationPenalty;
            if (s == "CompatibilityThresholdControl") {
                int mode = LEGACY_COMPATIBILITY_THRESHOLD;
                a_DataFile >> mode;
                loaded.CompatibilityThresholdControl = static_cast<CompatibilityThresholdMode>(mode);
            }
            if (s == "TargetSpecies") a_DataFile >> loaded.TargetSpecies;
            if (s == "CompatibilityThresholdGain") a_DataFile >> loaded.CompatibilityThresholdGain;
            if (s == "MaxCompatTreshold") a_DataFile >> loaded.MaxCompatTreshold;
            if (s == "RequireEvaluatedGenomes") {
                a_DataFile >> tf;
                if (tf == "true" || tf == "1" || tf == "1.0")
                    loaded.RequireEvaluatedGenomes = true;
                else
                    loaded.RequireEvaluatedGenomes = false;
            }
            if (s == "RejectNonFiniteFitness") {
                a_DataFile >> tf;
                if (tf == "true" || tf == "1" || tf == "1.0")
                    loaded.RejectNonFiniteFitness = true;
                else
                    loaded.RejectNonFiniteFitness = false;
            }
            if (s == "MutationOperatorsPerOffspring") a_DataFile >> loaded.MutationOperatorsPerOffspring;
            if (s == "AdaptiveMutationStart") a_DataFile >> loaded.AdaptiveMutationStart;
            if (s == "AdaptiveMutationRate") a_DataFile >> loaded.AdaptiveMutationRate;
            if (s == "AdaptiveMutationMaxFactor") a_DataFile >> loaded.AdaptiveMutationMaxFactor;
            if (s == "FitnessScaling") {
                int mode = SHIFTED_FITNESS_SCALING;
                a_DataFile >> mode;
                loaded.FitnessScaling = static_cast<FitnessScalingMode>(mode);
            }
            if (s == "FitnessRankPressure") a_DataFile >> loaded.FitnessRankPressure;
            if (s == "FitnessSigmaScale") a_DataFile >> loaded.FitnessSigmaScale;
            if (s == "FitnessBoltzmannTemperature") a_DataFile >> loaded.FitnessBoltzmannTemperature;
            // Unknown keys are skipped for forward compatibility.
        }

        *this = loaded;
        return 0;
    }

    int Parameters::Load(const char *a_FileName) {
        std::ifstream data(a_FileName);
        if (!data.is_open()) return 0;

        int result = Load(data);
        data.close();
        return result;
    }

    void Parameters::Save(const char *filename) {
        if (filename == nullptr) throw std::invalid_argument("Parameters::Save: filename is null.");
        FILE *f = detail::OpenFile(filename, "w");
        if (f == nullptr) throw std::runtime_error("Parameters::Save: cannot open output file.");
        try {
            Save(f);
        } catch (...) {
            fclose(f);
            throw;
        }
        if (fclose(f) != 0) throw std::runtime_error("Parameters::Save: failed to close output file.");
    }

    // Writes every field in Load-compatible "Key value" form.
    static void WriteParameters(std::ostream &output, const Parameters &p) {
        Serialization::UseRoundTripPrecision(output);
        output << "NEAT_ParametersStart\n";
        output << "PopulationSize " << (p.PopulationSize) << '\n';
        output << "Speciation " << (p.Speciation ? "true" : "false") << '\n';
        output << "DynamicCompatibility " << (p.DynamicCompatibility ? "true" : "false") << '\n';
        output << "MinSpecies " << (p.MinSpecies) << '\n';
        output << "MaxSpecies " << (p.MaxSpecies) << '\n';
        output << "InnovationsForever " << (p.InnovationsForever ? "true" : "false") << '\n';
        output << "AllowClones " << (p.AllowClones ? "true" : "false") << '\n';
        output << "NormalizeGenomeSize " << (p.NormalizeGenomeSize ? "true" : "false") << '\n';
        output << "ConstraintTrials " << (p.ConstraintTrials) << '\n';
        output << "YoungAgeTreshold " << (p.YoungAgeTreshold) << '\n';
        output << "YoungAgeFitnessBoost " << (p.YoungAgeFitnessBoost) << '\n';
        output << "SpeciesMaxStagnation " << (p.SpeciesMaxStagnation) << '\n';
        output << "StagnationDelta " << (p.StagnationDelta) << '\n';
        output << "OldAgeTreshold " << (p.OldAgeTreshold) << '\n';
        output << "OldAgePenalty " << (p.OldAgePenalty) << '\n';
        output << "DetectCompetetiveCoevolutionStagnation " << (p.DetectCompetetiveCoevolutionStagnation ? "true" : "false") << '\n';
        output << "KillWorstSpeciesEach " << (p.KillWorstSpeciesEach) << '\n';
        output << "KillWorstAge " << (p.KillWorstAge) << '\n';
        output << "SurvivalRate " << (p.SurvivalRate) << '\n';
        output << "CrossoverRate " << (p.CrossoverRate) << '\n';
        output << "OverallMutationRate " << (p.OverallMutationRate) << '\n';
        output << "InterspeciesCrossoverRate " << (p.InterspeciesCrossoverRate) << '\n';
        output << "MultipointCrossoverRate " << (p.MultipointCrossoverRate) << '\n';
        output << "PreferFitterParentRate " << (p.PreferFitterParentRate) << '\n';
        output << "RouletteWheelSelection " << (p.RouletteWheelSelection ? "true" : "false") << '\n';
        output << "PhasedSearching " << (p.PhasedSearching ? "true" : "false") << '\n';
        output << "DeltaCoding " << (p.DeltaCoding ? "true" : "false") << '\n';
        output << "SimplifyingPhaseMPCTreshold " << (p.SimplifyingPhaseMPCTreshold) << '\n';
        output << "SimplifyingPhaseStagnationTreshold " << (p.SimplifyingPhaseStagnationTreshold) << '\n';
        output << "ComplexityFloorGenerations " << (p.ComplexityFloorGenerations) << '\n';
        output << "NoveltySearch_K " << (p.NoveltySearch_K) << '\n';
        output << "NoveltySearch_P_min " << (p.NoveltySearch_P_min) << '\n';
        output << "NoveltySearch_Dynamic_Pmin " << (p.NoveltySearch_Dynamic_Pmin ? "true" : "false") << '\n';
        output << "NoveltySearch_No_Archiving_Stagnation_Treshold " << (p.NoveltySearch_No_Archiving_Stagnation_Treshold) << '\n';
        output << "NoveltySearch_Pmin_lowering_multiplier " << (p.NoveltySearch_Pmin_lowering_multiplier) << '\n';
        output << "NoveltySearch_Pmin_min " << (p.NoveltySearch_Pmin_min) << '\n';
        output << "NoveltySearch_Quick_Archiving_Min_Evaluations " << (p.NoveltySearch_Quick_Archiving_Min_Evaluations) << '\n';
        output << "NoveltySearch_Pmin_raising_multiplier " << (p.NoveltySearch_Pmin_raising_multiplier) << '\n';
        output << "NoveltySearch_Recompute_Sparseness_Each " << (p.NoveltySearch_Recompute_Sparseness_Each) << '\n';
        output << "MutateAddNeuronProb " << (p.MutateAddNeuronProb) << '\n';
        output << "SplitRecurrent " << (p.SplitRecurrent ? "true" : "false") << '\n';
        output << "SplitLoopedRecurrent " << (p.SplitLoopedRecurrent ? "true" : "false") << '\n';
        output << "NeuronTries " << (p.NeuronTries) << '\n';
        output << "MutateAddLinkProb " << (p.MutateAddLinkProb) << '\n';
        output << "MutateAddLinkFromBiasProb " << (p.MutateAddLinkFromBiasProb) << '\n';
        output << "MutateRemLinkProb " << (p.MutateRemLinkProb) << '\n';
        output << "MutateRemSimpleNeuronProb " << (p.MutateRemSimpleNeuronProb) << '\n';
        output << "LinkTries " << (p.LinkTries) << '\n';
        output << "MaxLinks " << (p.MaxLinks) << '\n';
        output << "MaxNeurons " << (p.MaxNeurons) << '\n';
        output << "RecurrentProb " << (p.RecurrentProb) << '\n';
        output << "RecurrentLoopProb " << (p.RecurrentLoopProb) << '\n';
        output << "MutateWeightsProb " << (p.MutateWeightsProb) << '\n';
        output << "MutateWeightsSevereProb " << (p.MutateWeightsSevereProb) << '\n';
        output << "WeightMutationRate " << (p.WeightMutationRate) << '\n';
        output << "WeightMutationMaxPower " << (p.WeightMutationMaxPower) << '\n';
        output << "WeightReplacementRate " << (p.WeightReplacementRate) << '\n';
        output << "WeightReplacementMaxPower " << (p.WeightReplacementMaxPower) << '\n';
        output << "MaxWeight " << (p.MaxWeight) << '\n';
        output << "MinWeight " << (p.MinWeight) << '\n';
        output << "MutateActivationAProb " << (p.MutateActivationAProb) << '\n';
        output << "MutateActivationBProb " << (p.MutateActivationBProb) << '\n';
        output << "ActivationAMutationMaxPower " << (p.ActivationAMutationMaxPower) << '\n';
        output << "ActivationBMutationMaxPower " << (p.ActivationBMutationMaxPower) << '\n';
        output << "TimeConstantMutationMaxPower " << (p.TimeConstantMutationMaxPower) << '\n';
        output << "BiasMutationMaxPower " << (p.BiasMutationMaxPower) << '\n';
        output << "MinActivationA " << (p.MinActivationA) << '\n';
        output << "MaxActivationA " << (p.MaxActivationA) << '\n';
        output << "MinActivationB " << (p.MinActivationB) << '\n';
        output << "MaxActivationB " << (p.MaxActivationB) << '\n';
        output << "MutateNeuronActivationTypeProb " << (p.MutateNeuronActivationTypeProb) << '\n';
        output << "ActivationFunction_SignedSigmoid_Prob " << (p.ActivationFunction_SignedSigmoid_Prob) << '\n';
        output << "ActivationFunction_UnsignedSigmoid_Prob " << (p.ActivationFunction_UnsignedSigmoid_Prob) << '\n';
        output << "ActivationFunction_Tanh_Prob " << (p.ActivationFunction_Tanh_Prob) << '\n';
        output << "ActivationFunction_TanhCubic_Prob " << (p.ActivationFunction_TanhCubic_Prob) << '\n';
        output << "ActivationFunction_SignedStep_Prob " << (p.ActivationFunction_SignedStep_Prob) << '\n';
        output << "ActivationFunction_UnsignedStep_Prob " << (p.ActivationFunction_UnsignedStep_Prob) << '\n';
        output << "ActivationFunction_SignedGauss_Prob " << (p.ActivationFunction_SignedGauss_Prob) << '\n';
        output << "ActivationFunction_UnsignedGauss_Prob " << (p.ActivationFunction_UnsignedGauss_Prob) << '\n';
        output << "ActivationFunction_Abs_Prob " << (p.ActivationFunction_Abs_Prob) << '\n';
        output << "ActivationFunction_SignedSine_Prob " << (p.ActivationFunction_SignedSine_Prob) << '\n';
        output << "ActivationFunction_UnsignedSine_Prob " << (p.ActivationFunction_UnsignedSine_Prob) << '\n';
        output << "ActivationFunction_Linear_Prob " << (p.ActivationFunction_Linear_Prob) << '\n';
        output << "ActivationFunction_Relu_Prob " << (p.ActivationFunction_Relu_Prob) << '\n';
        output << "ActivationFunction_Softplus_Prob " << (p.ActivationFunction_Softplus_Prob) << '\n';
        output << "ActivationFunction_SpikingLIF_Prob " << (p.ActivationFunction_SpikingLIF_Prob) << '\n';
        output << "ActivationFunction_SpikingAdaptiveLIF_Prob " << (p.ActivationFunction_SpikingAdaptiveLIF_Prob) << '\n';
        output << "ActivationFunction_SpikingIzhikevich_Prob " << (p.ActivationFunction_SpikingIzhikevich_Prob) << '\n';
        output << "ActivationFunction_McCullochPitts_Prob " << (p.ActivationFunction_McCullochPitts_Prob) << '\n';
        output << "MutateNeuronTimeConstantsProb " << (p.MutateNeuronTimeConstantsProb) << '\n';
        output << "MutateNeuronBiasesProb " << (p.MutateNeuronBiasesProb) << '\n';
        output << "MinNeuronTimeConstant " << (p.MinNeuronTimeConstant) << '\n';
        output << "MaxNeuronTimeConstant " << (p.MaxNeuronTimeConstant) << '\n';
        output << "MinNeuronBias " << (p.MinNeuronBias) << '\n';
        output << "MaxNeuronBias " << (p.MaxNeuronBias) << '\n';
        output << "DontUseBiasNeuron " << (p.DontUseBiasNeuron ? "true" : "false") << '\n';
        output << "ArchiveEnforcement " << (p.ArchiveEnforcement ? "true" : "false") << '\n';
        output << "AllowLoops " << (p.AllowLoops ? "true" : "false") << '\n';
        output << "DisjointCoeff " << (p.DisjointCoeff) << '\n';
        output << "ExcessCoeff " << (p.ExcessCoeff) << '\n';
        output << "ActivationADiffCoeff " << (p.ActivationADiffCoeff) << '\n';
        output << "ActivationBDiffCoeff " << (p.ActivationBDiffCoeff) << '\n';
        output << "WeightDiffCoeff " << (p.WeightDiffCoeff) << '\n';
        output << "TimeConstantDiffCoeff " << (p.TimeConstantDiffCoeff) << '\n';
        output << "BiasDiffCoeff " << (p.BiasDiffCoeff) << '\n';
        output << "ActivationFunctionDiffCoeff " << (p.ActivationFunctionDiffCoeff) << '\n';
        output << "SpikingNeuronDiffCoeff " << (p.SpikingNeuronDiffCoeff) << '\n';
        output << "SpikingLinkDiffCoeff " << (p.SpikingLinkDiffCoeff) << '\n';
        output << "CompatTreshold " << (p.CompatTreshold) << '\n';
        output << "MinCompatTreshold " << (p.MinCompatTreshold) << '\n';
        output << "CompatTresholdModifier " << (p.CompatTresholdModifier) << '\n';
        output << "CompatTreshChangeInterval_Generations " << (p.CompatTreshChangeInterval_Generations) << '\n';
        output << "CompatTreshChangeInterval_Evaluations " << (p.CompatTreshChangeInterval_Evaluations) << '\n';
        output << "MinDeltaCompatEqualGenomes " << (p.MinDeltaCompatEqualGenomes) << '\n';
        output << "DivisionThreshold " << (p.DivisionThreshold) << '\n';
        output << "VarianceThreshold " << (p.VarianceThreshold) << '\n';
        output << "BandThreshold " << (p.BandThreshold) << '\n';
        output << "InitialDepth " << (p.InitialDepth) << '\n';
        output << "MaxDepth " << (p.MaxDepth) << '\n';
        output << "IterationLevel " << (p.IterationLevel) << '\n';
        output << "TournamentSelection " << (p.TournamentSelection ? "true" : "false") << '\n';
        output << "TruncationSelection " << (p.TruncationSelection ? "true" : "false") << '\n';
        output << "TournamentSize " << (p.TournamentSize) << '\n';
        output << "CPPN_Bias " << (p.CPPN_Bias) << '\n';
        output << "Width " << (p.Width) << '\n';
        output << "Height " << (p.Height) << '\n';
        output << "Depth " << (p.Depth) << '\n';
        output << "Qtree_X " << (p.Qtree_X) << '\n';
        output << "Qtree_Y " << (p.Qtree_Y) << '\n';
        output << "Qtree_Z " << (p.Qtree_Z) << '\n';
        output << "Leo " << (p.Leo ? "true" : "false") << '\n';
        output << "LeoThreshold " << (p.LeoThreshold) << '\n';
        output << "LeoSeed " << (p.LeoSeed ? "true" : "false") << '\n';
        output << "GeometrySeed " << (p.GeometrySeed ? "true" : "false") << '\n';
        output << "Elitism " << (p.EliteFraction) << '\n';
        output << "EliteFraction " << (p.EliteFraction) << '\n';
        output << "MutateNeuronSpikingParametersProb " << (p.MutateNeuronSpikingParametersProb) << '\n';
        output << "MutateLinkSpikingParametersProb " << (p.MutateLinkSpikingParametersProb) << '\n';
        output << "SpikingParameterMutationRate " << (p.SpikingParameterMutationRate) << '\n';
        output << "SpikingParameterMutationPower " << (p.SpikingParameterMutationPower) << '\n';
        output << "InitialMCPInhibitoryVetoProb " << (p.InitialMCPInhibitoryVetoProb) << '\n';
        output << "MutateMCPInhibitoryVetoProb " << (p.MutateMCPInhibitoryVetoProb) << '\n';
        output << "MinSpikingTimeConstant " << (p.MinSpikingTimeConstant) << '\n';
        output << "MaxSpikingTimeConstant " << (p.MaxSpikingTimeConstant) << '\n';
        output << "MinSpikeThreshold " << (p.MinSpikeThreshold) << '\n';
        output << "MaxSpikeThreshold " << (p.MaxSpikeThreshold) << '\n';
        output << "MinResetPotential " << (p.MinResetPotential) << '\n';
        output << "MaxResetPotential " << (p.MaxResetPotential) << '\n';
        output << "MinRestingPotential " << (p.MinRestingPotential) << '\n';
        output << "MaxRestingPotential " << (p.MaxRestingPotential) << '\n';
        output << "MinRefractoryPeriod " << (p.MinRefractoryPeriod) << '\n';
        output << "MaxRefractoryPeriod " << (p.MaxRefractoryPeriod) << '\n';
        output << "MinMembraneResistance " << (p.MinMembraneResistance) << '\n';
        output << "MaxMembraneResistance " << (p.MaxMembraneResistance) << '\n';
        output << "MinAdaptationTimeConstant " << (p.MinAdaptationTimeConstant) << '\n';
        output << "MaxAdaptationTimeConstant " << (p.MaxAdaptationTimeConstant) << '\n';
        output << "MinAdaptationIncrement " << (p.MinAdaptationIncrement) << '\n';
        output << "MaxAdaptationIncrement " << (p.MaxAdaptationIncrement) << '\n';
        output << "MinSpikeRateTimeConstant " << (p.MinSpikeRateTimeConstant) << '\n';
        output << "MaxSpikeRateTimeConstant " << (p.MaxSpikeRateTimeConstant) << '\n';
        output << "MinIzhikevichA " << (p.MinIzhikevichA) << '\n';
        output << "MaxIzhikevichA " << (p.MaxIzhikevichA) << '\n';
        output << "MinIzhikevichThreshold " << (p.MinIzhikevichThreshold) << '\n';
        output << "MaxIzhikevichThreshold " << (p.MaxIzhikevichThreshold) << '\n';
        output << "MinIzhikevichB " << (p.MinIzhikevichB) << '\n';
        output << "MaxIzhikevichB " << (p.MaxIzhikevichB) << '\n';
        output << "MinIzhikevichC " << (p.MinIzhikevichC) << '\n';
        output << "MaxIzhikevichC " << (p.MaxIzhikevichC) << '\n';
        output << "MinIzhikevichD " << (p.MinIzhikevichD) << '\n';
        output << "MaxIzhikevichD " << (p.MaxIzhikevichD) << '\n';
        output << "MinSynapticDelay " << (p.MinSynapticDelay) << '\n';
        output << "MaxSynapticDelay " << (p.MaxSynapticDelay) << '\n';
        output << "MinSynapticTimeConstant " << (p.MinSynapticTimeConstant) << '\n';
        output << "MaxSynapticTimeConstant " << (p.MaxSynapticTimeConstant) << '\n';
        output << "InitialSTDPEnabledProb " << (p.InitialSTDPEnabledProb) << '\n';
        output << "MinSTDPPlus " << (p.MinSTDPPlus) << '\n';
        output << "MaxSTDPPlus " << (p.MaxSTDPPlus) << '\n';
        output << "MinSTDPMinus " << (p.MinSTDPMinus) << '\n';
        output << "MaxSTDPMinus " << (p.MaxSTDPMinus) << '\n';
        output << "MinSTDPTau " << (p.MinSTDPTau) << '\n';
        output << "MaxSTDPTau " << (p.MaxSTDPTau) << '\n';
        output << "ParentSelectionMode " << (static_cast<int>(p.ParentSelectionMode)) << '\n';
        output << "RankSelectionPressure " << (p.RankSelectionPressure) << '\n';
        output << "RankSelectionExponent " << (p.RankSelectionExponent) << '\n';
        output << "BoltzmannTemperature " << (p.BoltzmannTemperature) << '\n';
        output << "SinglePointCrossoverRate " << (p.SinglePointCrossoverRate) << '\n';
        output << "BlendCrossoverRate " << (p.BlendCrossoverRate) << '\n';
        output << "SimulatedBinaryCrossoverRate " << (p.SimulatedBinaryCrossoverRate) << '\n';
        output << "CrossoverBlendAlpha " << (p.CrossoverBlendAlpha) << '\n';
        output << "CrossoverSBXEta " << (p.CrossoverSBXEta) << '\n';
        output << "WeightMutationDistribution " << (static_cast<int>(p.WeightMutationDistribution)) << '\n';
        output << "WeightMutationSigma " << (p.WeightMutationSigma) << '\n';
        output << "WeightMutationCauchyScale " << (p.WeightMutationCauchyScale) << '\n';
        output << "WeightMutationPolynomialEta " << (p.WeightMutationPolynomialEta) << '\n';
        output << "SpeciesRepresentativeSelection " << (static_cast<int>(p.SpeciesRepresentativeSelection)) << '\n';
        output << "RepresentativeSelectionCandidates " << (p.RepresentativeSelectionCandidates) << '\n';
        output << "OffspringAllocation " << (static_cast<int>(p.OffspringAllocation)) << '\n';
        output << "MinSpeciesSize " << (p.MinSpeciesSize) << '\n';
        output << "SpeciesElitism " << (p.SpeciesElitism) << '\n';
        output << "StagnationPenalty " << (p.StagnationPenalty) << '\n';
        output << "CompatibilityThresholdControl " << (static_cast<int>(p.CompatibilityThresholdControl)) << '\n';
        output << "TargetSpecies " << (p.TargetSpecies) << '\n';
        output << "CompatibilityThresholdGain " << (p.CompatibilityThresholdGain) << '\n';
        output << "MaxCompatTreshold " << (p.MaxCompatTreshold) << '\n';
        output << "RequireEvaluatedGenomes " << (p.RequireEvaluatedGenomes ? "true" : "false") << '\n';
        output << "RejectNonFiniteFitness " << (p.RejectNonFiniteFitness ? "true" : "false") << '\n';
        output << "MutationOperatorsPerOffspring " << (p.MutationOperatorsPerOffspring) << '\n';
        output << "AdaptiveMutationStart " << (p.AdaptiveMutationStart) << '\n';
        output << "AdaptiveMutationRate " << (p.AdaptiveMutationRate) << '\n';
        output << "AdaptiveMutationMaxFactor " << (p.AdaptiveMutationMaxFactor) << '\n';
        output << "FitnessScaling " << (static_cast<int>(p.FitnessScaling)) << '\n';
        output << "FitnessRankPressure " << (p.FitnessRankPressure) << '\n';
        output << "FitnessSigmaScale " << (p.FitnessSigmaScale) << '\n';
        output << "FitnessBoltzmannTemperature " << (p.FitnessBoltzmannTemperature) << '\n';
        output << "NEAT_ParametersEnd\n";
    }

    void Parameters::Save(FILE *a_fstream) {
        if (a_fstream == nullptr) throw std::invalid_argument("Parameters::Save: file is null.");
        std::ostringstream output;
        WriteParameters(output, *this);
        const std::string data = output.str();
        if (std::fwrite(data.data(), 1, data.size(), a_fstream) != data.size()) throw std::runtime_error("Parameters::Save: failed to write output.");
    }

    void Parameters::ConfigureSpiking(bool enable_stdp) {
        ActivationFunction_SignedSigmoid_Prob = 0.0;
        ActivationFunction_UnsignedSigmoid_Prob = 0.0;
        ActivationFunction_Tanh_Prob = 0.0;
        ActivationFunction_TanhCubic_Prob = 0.0;
        ActivationFunction_SignedStep_Prob = 0.0;
        ActivationFunction_UnsignedStep_Prob = 0.0;
        ActivationFunction_SignedGauss_Prob = 0.0;
        ActivationFunction_UnsignedGauss_Prob = 0.0;
        ActivationFunction_Abs_Prob = 0.0;
        ActivationFunction_SignedSine_Prob = 0.0;
        ActivationFunction_UnsignedSine_Prob = 0.0;
        ActivationFunction_Linear_Prob = 0.0;
        ActivationFunction_Relu_Prob = 0.0;
        ActivationFunction_Softplus_Prob = 0.0;
        ActivationFunction_SpikingLIF_Prob = 0.65;
        ActivationFunction_SpikingAdaptiveLIF_Prob = 0.25;
        ActivationFunction_SpikingIzhikevich_Prob = 0.10;
        ActivationFunction_McCullochPitts_Prob = 0.0;

        MutateNeuronActivationTypeProb = 0.05;
        MutateNeuronSpikingParametersProb = 0.25;
        MutateLinkSpikingParametersProb = 0.15;
        RecurrentProb = std::max(RecurrentProb, static_cast<Real>(0.2));
        AllowLoops = true;
        InitialSTDPEnabledProb = enable_stdp ? 0.1 : 0.0;
        SpikingNeuronDiffCoeff = 0.1;
        SpikingLinkDiffCoeff = 0.1;
    }

    void Parameters::ConfigureMcCullochPitts(bool inhibitory_veto, bool enable_stdp) {
        ConfigureSpiking(enable_stdp);
        ActivationFunction_SpikingLIF_Prob = 0.0;
        ActivationFunction_SpikingAdaptiveLIF_Prob = 0.0;
        ActivationFunction_SpikingIzhikevich_Prob = 0.0;
        ActivationFunction_McCullochPitts_Prob = 1.0;
        InitialMCPInhibitoryVetoProb = inhibitory_veto ? 1.0 : 0.0;
        MutateMCPInhibitoryVetoProb = 0.05;
    }

    std::string Parameters::Serialize() const {
        std::ostringstream output;
        WriteParameters(output, *this);
        return output.str();
    }

    Parameters Parameters::Deserialize(const std::string &data) {
        Parameters parameters;
        std::istringstream input(data);
        if (parameters.Load(input) != 0) throw std::runtime_error("Parameters::Deserialize: missing or incomplete parameter block.");
        return parameters;
    }

    void Parameters::SetCustomConstraintsFunction(std::function<bool(Genome &)> callback) {
        m_CustomConstraintsFunction = std::move(callback);
        if (m_CustomConstraintsFunction) CustomConstraints = NULL;
    }

    std::function<bool(Genome &)> Parameters::GetCustomConstraintsFunction() const {
        if (m_CustomConstraintsFunction) return m_CustomConstraintsFunction;
        if (CustomConstraints != NULL) return CustomConstraints;
        return {};
    }

    bool Parameters::FailsCustomConstraints(Genome &genome) const {
        if (m_CustomConstraintsFunction) return m_CustomConstraintsFunction(genome);
        return CustomConstraints != NULL && CustomConstraints(genome);
    }

    bool Parameters::Validate(std::string *error) const {
        const auto fail = [error](const std::string &message) {
            if (error != nullptr) *error = message;
            return false;
        };
        const auto finite_range = [&fail](const char *name, Real minimum, Real maximum) {
            if (!std::isfinite(minimum) || !std::isfinite(maximum) || minimum > maximum) {
                return fail(std::string(name) + " must have a finite, ordered minimum and maximum");
            }
            return true;
        };
        const auto probability = [&fail](const char *name, Real value) {
            if (!std::isfinite(value) || value < 0.0 || value > 1.0) {
                return fail(std::string(name) + " must be between 0 and 1");
            }
            return true;
        };

        if (PopulationSize == 0) return fail("PopulationSize must be greater than zero");
        if (PopulationSize >= static_cast<unsigned int>(std::numeric_limits<int>::max())) return fail("PopulationSize exceeds the supported ID range");
        if (MinSpecies == 0 || MinSpecies > MaxSpecies) return fail("MinSpecies and MaxSpecies must define a non-empty range");
        if (ConstraintTrials <= 0) return fail("ConstraintTrials must be greater than zero");
        if (NeuronTries <= 0) return fail("NeuronTries must be greater than zero");
        if (LinkTries == 0) return fail("LinkTries must be greater than zero");
        if (MaxLinks < -1 || MaxNeurons < -1) return fail("MaxLinks and MaxNeurons must be -1 or non-negative");
        if (InitialDepth > MaxDepth) return fail("InitialDepth cannot exceed MaxDepth");
        if (MaxDepth > 9) return fail("MaxDepth exceeds the supported safe limit of 9");
        if (Width <= 0.0 || Height <= 0.0 || Depth <= 0.0) return fail("Width, Height, and Depth must be positive");
        if ((TournamentSelection || ParentSelectionMode == TOURNAMENT) && TournamentSize == 0)
            return fail("TournamentSize must be greater than zero when tournament selection is enabled");
        if (ParentSelectionMode < LEGACY_SELECTION || ParentSelectionMode > BOLTZMANN) return fail("ParentSelectionMode is not a supported selection mode");
        if (WeightMutationDistribution < UNIFORM_MUTATION || WeightMutationDistribution > POLYNOMIAL_MUTATION)
            return fail("WeightMutationDistribution is not a supported mutation mode");
        if (SpeciesRepresentativeSelection < FIRST_REPRESENTATIVE || SpeciesRepresentativeSelection > MEDOID_REPRESENTATIVE)
            return fail("SpeciesRepresentativeSelection is not a supported mode");
        if (OffspringAllocation < LARGEST_REMAINDER || OffspringAllocation > STOCHASTIC_REMAINDER) return fail("OffspringAllocation is not a supported mode");
        if (CompatibilityThresholdControl < LEGACY_COMPATIBILITY_THRESHOLD || CompatibilityThresholdControl > PROPORTIONAL_COMPATIBILITY_THRESHOLD)
            return fail("CompatibilityThresholdControl is not a supported mode");
        if (FitnessScaling < SHIFTED_FITNESS_SCALING || FitnessScaling > BOLTZMANN_FITNESS_SCALING) return fail("FitnessScaling is not a supported mode");
        if (MinSpeciesSize > PopulationSize) return fail("MinSpeciesSize cannot exceed PopulationSize");
        if (SpeciesElitism > PopulationSize) return fail("SpeciesElitism cannot exceed PopulationSize");
        if (MinSpeciesSize > 0 && SpeciesElitism > 0 &&
            static_cast<std::uint64_t>(MinSpeciesSize) * static_cast<std::uint64_t>(SpeciesElitism) > PopulationSize)
            return fail("MinSpeciesSize times SpeciesElitism cannot exceed PopulationSize");
        if (TargetSpecies > PopulationSize) return fail("TargetSpecies cannot exceed PopulationSize");
        if (DetectCompetetiveCoevolutionStagnation && (KillWorstSpeciesEach <= 0 || KillWorstAge < 0))
            return fail("competitive coevolution stagnation detection requires a positive interval and non-negative age");

        const std::pair<const char *, Real> probabilities[] = {{"SurvivalRate", SurvivalRate},
                                                               {"CrossoverRate", CrossoverRate},
                                                               {"OverallMutationRate", OverallMutationRate},
                                                               {"InterspeciesCrossoverRate", InterspeciesCrossoverRate},
                                                               {"MultipointCrossoverRate", MultipointCrossoverRate},
                                                               {"SinglePointCrossoverRate", SinglePointCrossoverRate},
                                                               {"BlendCrossoverRate", BlendCrossoverRate},
                                                               {"SimulatedBinaryCrossoverRate", SimulatedBinaryCrossoverRate},
                                                               {"PreferFitterParentRate", PreferFitterParentRate},
                                                               {"EliteFraction", EliteFraction},
                                                               {"MutateAddNeuronProb", MutateAddNeuronProb},
                                                               {"MutateAddLinkProb", MutateAddLinkProb},
                                                               {"MutateAddLinkFromBiasProb", MutateAddLinkFromBiasProb},
                                                               {"MutateRemLinkProb", MutateRemLinkProb},
                                                               {"MutateRemSimpleNeuronProb", MutateRemSimpleNeuronProb},
                                                               {"RecurrentProb", RecurrentProb},
                                                               {"RecurrentLoopProb", RecurrentLoopProb},
                                                               {"MutateWeightsProb", MutateWeightsProb},
                                                               {"MutateWeightsSevereProb", MutateWeightsSevereProb},
                                                               {"WeightMutationRate", WeightMutationRate},
                                                               {"WeightReplacementRate", WeightReplacementRate},
                                                               {"MutateActivationAProb", MutateActivationAProb},
                                                               {"MutateActivationBProb", MutateActivationBProb},
                                                               {"MutateNeuronActivationTypeProb", MutateNeuronActivationTypeProb},
                                                               {"MutateNeuronTimeConstantsProb", MutateNeuronTimeConstantsProb},
                                                               {"MutateNeuronBiasesProb", MutateNeuronBiasesProb},
                                                               {"MutateNeuronTraitsProb", MutateNeuronTraitsProb},
                                                               {"MutateLinkTraitsProb", MutateLinkTraitsProb},
                                                               {"MutateGenomeTraitsProb", MutateGenomeTraitsProb},
                                                               {"ActivationFunction_SignedSigmoid_Prob", ActivationFunction_SignedSigmoid_Prob},
                                                               {"ActivationFunction_UnsignedSigmoid_Prob", ActivationFunction_UnsignedSigmoid_Prob},
                                                               {"ActivationFunction_Tanh_Prob", ActivationFunction_Tanh_Prob},
                                                               {"ActivationFunction_TanhCubic_Prob", ActivationFunction_TanhCubic_Prob},
                                                               {"ActivationFunction_SignedStep_Prob", ActivationFunction_SignedStep_Prob},
                                                               {"ActivationFunction_UnsignedStep_Prob", ActivationFunction_UnsignedStep_Prob},
                                                               {"ActivationFunction_SignedGauss_Prob", ActivationFunction_SignedGauss_Prob},
                                                               {"ActivationFunction_UnsignedGauss_Prob", ActivationFunction_UnsignedGauss_Prob},
                                                               {"ActivationFunction_Abs_Prob", ActivationFunction_Abs_Prob},
                                                               {"ActivationFunction_SignedSine_Prob", ActivationFunction_SignedSine_Prob},
                                                               {"ActivationFunction_UnsignedSine_Prob", ActivationFunction_UnsignedSine_Prob},
                                                               {"ActivationFunction_Linear_Prob", ActivationFunction_Linear_Prob},
                                                               {"ActivationFunction_Relu_Prob", ActivationFunction_Relu_Prob},
                                                               {"ActivationFunction_Softplus_Prob", ActivationFunction_Softplus_Prob},
                                                               {"ActivationFunction_SpikingLIF_Prob", ActivationFunction_SpikingLIF_Prob},
                                                               {"ActivationFunction_SpikingAdaptiveLIF_Prob", ActivationFunction_SpikingAdaptiveLIF_Prob},
                                                               {"ActivationFunction_SpikingIzhikevich_Prob", ActivationFunction_SpikingIzhikevich_Prob},
                                                               {"ActivationFunction_McCullochPitts_Prob", ActivationFunction_McCullochPitts_Prob},
                                                               {"MutateNeuronSpikingParametersProb", MutateNeuronSpikingParametersProb},
                                                               {"MutateLinkSpikingParametersProb", MutateLinkSpikingParametersProb},
                                                               {"SpikingParameterMutationRate", SpikingParameterMutationRate},
                                                               {"InitialMCPInhibitoryVetoProb", InitialMCPInhibitoryVetoProb},
                                                               {"MutateMCPInhibitoryVetoProb", MutateMCPInhibitoryVetoProb},
                                                               {"InitialSTDPEnabledProb", InitialSTDPEnabledProb}};
        for (const auto &item : probabilities) {
            if (!probability(item.first, item.second)) return false;
        }
        const Real crossover_mode_total = MultipointCrossoverRate + SinglePointCrossoverRate + BlendCrossoverRate + SimulatedBinaryCrossoverRate;
        if (!std::isfinite(crossover_mode_total) || crossover_mode_total > 1.0 + 1.0e-12) return fail("crossover method probabilities must sum to at most 1");

        if (!finite_range("weight range", MinWeight, MaxWeight) || !finite_range("activation A range", MinActivationA, MaxActivationA) ||
            !finite_range("activation B range", MinActivationB, MaxActivationB) ||
            !finite_range("neuron time-constant range", MinNeuronTimeConstant, MaxNeuronTimeConstant) ||
            !finite_range("neuron bias range", MinNeuronBias, MaxNeuronBias)) {
            return false;
        }
        const std::pair<const char *, std::pair<Real, Real>> spiking_ranges[] = {
            {"spiking time-constant range", {MinSpikingTimeConstant, MaxSpikingTimeConstant}},
            {"spike-threshold range", {MinSpikeThreshold, MaxSpikeThreshold}},
            {"reset-potential range", {MinResetPotential, MaxResetPotential}},
            {"resting-potential range", {MinRestingPotential, MaxRestingPotential}},
            {"refractory-period range", {MinRefractoryPeriod, MaxRefractoryPeriod}},
            {"membrane-resistance range", {MinMembraneResistance, MaxMembraneResistance}},
            {"adaptation time-constant range", {MinAdaptationTimeConstant, MaxAdaptationTimeConstant}},
            {"adaptation-increment range", {MinAdaptationIncrement, MaxAdaptationIncrement}},
            {"spike-rate time-constant range", {MinSpikeRateTimeConstant, MaxSpikeRateTimeConstant}},
            {"Izhikevich a range", {MinIzhikevichA, MaxIzhikevichA}},
            {"Izhikevich threshold range", {MinIzhikevichThreshold, MaxIzhikevichThreshold}},
            {"Izhikevich b range", {MinIzhikevichB, MaxIzhikevichB}},
            {"Izhikevich c range", {MinIzhikevichC, MaxIzhikevichC}},
            {"Izhikevich d range", {MinIzhikevichD, MaxIzhikevichD}},
            {"synaptic-delay range", {MinSynapticDelay, MaxSynapticDelay}},
            {"synaptic time-constant range", {MinSynapticTimeConstant, MaxSynapticTimeConstant}},
            {"STDP potentiation range", {MinSTDPPlus, MaxSTDPPlus}},
            {"STDP depression range", {MinSTDPMinus, MaxSTDPMinus}},
            {"STDP trace time-constant range", {MinSTDPTau, MaxSTDPTau}}};
        for (const auto &range : spiking_ranges) {
            if (!finite_range(range.first, range.second.first, range.second.second)) {
                return false;
            }
        }
        if (MinSpikingTimeConstant <= 0.0 || MinSynapticTimeConstant <= 0.0 || MinAdaptationTimeConstant <= 0.0 || MinSpikeRateTimeConstant <= 0.0 ||
            MinSTDPTau <= 0.0 || MinRefractoryPeriod < 0.0 || MinMembraneResistance <= 0.0 || MinSynapticDelay < 0.0 || MinSTDPPlus < 0.0 ||
            MinSTDPMinus < 0.0) {
            return fail(
                "spiking time constants and resistance must be positive; delays, refractory periods, and STDP amplitudes cannot be "
                "negative");
        }

        const std::pair<const char *, Real> non_negative[] = {{"YoungAgeFitnessBoost", YoungAgeFitnessBoost},
                                                              {"StagnationDelta", StagnationDelta},
                                                              {"OldAgePenalty", OldAgePenalty},
                                                              {"WeightMutationMaxPower", WeightMutationMaxPower},
                                                              {"WeightReplacementMaxPower", WeightReplacementMaxPower},
                                                              {"ActivationAMutationMaxPower", ActivationAMutationMaxPower},
                                                              {"ActivationBMutationMaxPower", ActivationBMutationMaxPower},
                                                              {"TimeConstantMutationMaxPower", TimeConstantMutationMaxPower},
                                                              {"BiasMutationMaxPower", BiasMutationMaxPower},
                                                              {"CrossoverBlendAlpha", CrossoverBlendAlpha},
                                                              {"CrossoverSBXEta", CrossoverSBXEta},
                                                              {"WeightMutationPolynomialEta", WeightMutationPolynomialEta},
                                                              {"DisjointCoeff", DisjointCoeff},
                                                              {"ExcessCoeff", ExcessCoeff},
                                                              {"ActivationADiffCoeff", ActivationADiffCoeff},
                                                              {"ActivationBDiffCoeff", ActivationBDiffCoeff},
                                                              {"WeightDiffCoeff", WeightDiffCoeff},
                                                              {"TimeConstantDiffCoeff", TimeConstantDiffCoeff},
                                                              {"BiasDiffCoeff", BiasDiffCoeff},
                                                              {"ActivationFunctionDiffCoeff", ActivationFunctionDiffCoeff},
                                                              {"SpikingNeuronDiffCoeff", SpikingNeuronDiffCoeff},
                                                              {"SpikingLinkDiffCoeff", SpikingLinkDiffCoeff},
                                                              {"CompatTreshold", CompatTreshold},
                                                              {"MinCompatTreshold", MinCompatTreshold},
                                                              {"CompatTresholdModifier", CompatTresholdModifier},
                                                              {"MinDeltaCompatEqualGenomes", MinDeltaCompatEqualGenomes},
                                                              {"NoveltySearch_P_min", NoveltySearch_P_min},
                                                              {"NoveltySearch_Pmin_min", NoveltySearch_Pmin_min},
                                                              {"DivisionThreshold", DivisionThreshold},
                                                              {"VarianceThreshold", VarianceThreshold},
                                                              {"BandThreshold", BandThreshold},
                                                              {"SpikingParameterMutationPower", SpikingParameterMutationPower},
                                                              {"StagnationPenalty", StagnationPenalty},
                                                              {"CompatibilityThresholdGain", CompatibilityThresholdGain},
                                                              {"AdaptiveMutationRate", AdaptiveMutationRate}};
        for (const auto &item : non_negative) {
            if (!std::isfinite(item.second) || item.second < 0.0) return fail(std::string(item.first) + " must be finite and non-negative");
        }
        if (!std::isfinite(MaxCompatTreshold) || MaxCompatTreshold < MinCompatTreshold)
            return fail("MaxCompatTreshold must be finite and at least MinCompatTreshold");
        if (CompatTreshold > MaxCompatTreshold) return fail("CompatTreshold cannot exceed MaxCompatTreshold");
        if (!std::isfinite(MutationOperatorsPerOffspring) || MutationOperatorsPerOffspring < 1.0 || MutationOperatorsPerOffspring > 256.0)
            return fail("MutationOperatorsPerOffspring must be finite and in [1, 256]");
        if (!std::isfinite(AdaptiveMutationMaxFactor) || AdaptiveMutationMaxFactor < 1.0 || AdaptiveMutationMaxFactor > 256.0 ||
            MutationOperatorsPerOffspring * AdaptiveMutationMaxFactor > 1024.0)
            return fail("Adaptive mutation settings exceed the supported operator budget");
        if (!std::isfinite(RankSelectionPressure) || RankSelectionPressure < 1.0 || RankSelectionPressure > 2.0)
            return fail("RankSelectionPressure must be between 1 and 2");
        if (!std::isfinite(FitnessRankPressure) || FitnessRankPressure < 1.0 || FitnessRankPressure > 2.0)
            return fail("FitnessRankPressure must be between 1 and 2");
        const std::pair<const char *, Real> positive_values[] = {
            {"RankSelectionExponent", RankSelectionExponent}, {"BoltzmannTemperature", BoltzmannTemperature},
            {"WeightMutationSigma", WeightMutationSigma},     {"WeightMutationCauchyScale", WeightMutationCauchyScale},
            {"FitnessSigmaScale", FitnessSigmaScale},         {"FitnessBoltzmannTemperature", FitnessBoltzmannTemperature}};
        for (const auto &item : positive_values) {
            if (!std::isfinite(item.second) || item.second <= 0.0) return fail(std::string(item.first) + " must be finite and positive");
        }
        if (!std::isfinite(NoveltySearch_Pmin_lowering_multiplier) || NoveltySearch_Pmin_lowering_multiplier <= 0.0 ||
            !std::isfinite(NoveltySearch_Pmin_raising_multiplier) || NoveltySearch_Pmin_raising_multiplier <= 0.0)
            return fail("novelty threshold multipliers must be finite and positive");
        const std::pair<const char *, Real> finite_values[] = {
            {"CPPN_Bias", CPPN_Bias}, {"Width", Width},     {"Height", Height},   {"Depth", Depth},
            {"Qtree_X", Qtree_X},     {"Qtree_Y", Qtree_Y}, {"Qtree_Z", Qtree_Z}, {"LeoThreshold", LeoThreshold}};
        for (const auto &item : finite_values) {
            if (!std::isfinite(item.second)) return fail(std::string(item.first) + " must be finite");
        }

        const Real activation_total =
            ActivationFunction_SignedSigmoid_Prob + ActivationFunction_UnsignedSigmoid_Prob + ActivationFunction_Tanh_Prob + ActivationFunction_TanhCubic_Prob +
            ActivationFunction_SignedStep_Prob + ActivationFunction_UnsignedStep_Prob + ActivationFunction_SignedGauss_Prob +
            ActivationFunction_UnsignedGauss_Prob + ActivationFunction_Abs_Prob + ActivationFunction_SignedSine_Prob + ActivationFunction_UnsignedSine_Prob +
            ActivationFunction_Linear_Prob + ActivationFunction_Relu_Prob + ActivationFunction_Softplus_Prob + ActivationFunction_SpikingLIF_Prob +
            ActivationFunction_SpikingAdaptiveLIF_Prob + ActivationFunction_SpikingIzhikevich_Prob + ActivationFunction_McCullochPitts_Prob;
        if ((MutateAddNeuronProb > 0.0 || MutateNeuronActivationTypeProb > 0.0) && activation_total <= 0.0)
            return fail("at least one activation function must have positive probability");

        const auto validate_set_probabilities = [&fail](const std::string &prefix, std::size_t set_size, const std::vector<Real> &probs) {
            if (!probs.empty() && probs.size() != set_size) return fail(prefix + "probability count must match the set size");
            for (const Real value : probs) {
                if (!std::isfinite(value) || value < 0.0) return fail(prefix + "set probabilities must be finite and non-negative");
            }
            return true;
        };
        const auto validate_traits = [&](const char *category, const std::map<std::string, TraitParameters> &schemas) {
            for (const auto &entry : schemas) {
                const TraitParameters &schema = entry.second;
                const std::string prefix = std::string(category) + " trait '" + entry.first + "': ";
                if (!std::isfinite(schema.m_ImportanceCoeff) || schema.m_ImportanceCoeff < 0.0)
                    return fail(prefix + "importance must be finite and non-negative");
                if (!probability((prefix + "mutation probability").c_str(), schema.m_MutationProb)) return false;
                if (schema.type == "int") {
                    if (!std::holds_alternative<IntTraitParameters>(schema.m_Details)) return fail(prefix + "detail type does not match");
                    const auto &detail = std::get<IntTraitParameters>(schema.m_Details);
                    if (detail.min > detail.max || detail.mut_power < 0) return fail(prefix + "integer range is invalid");
                    if (!probability((prefix + "replacement probability").c_str(), detail.mut_replace_prob)) return false;
                } else if (schema.type == "float") {
                    if (!std::holds_alternative<FloatTraitParameters>(schema.m_Details)) return fail(prefix + "detail type does not match");
                    const auto &detail = std::get<FloatTraitParameters>(schema.m_Details);
                    if (!std::isfinite(detail.min) || !std::isfinite(detail.max) || detail.min > detail.max || !std::isfinite(detail.mut_power) ||
                        detail.mut_power < 0.0)
                        return fail(prefix + "floating-point range is invalid");
                    if (!probability((prefix + "replacement probability").c_str(), detail.mut_replace_prob)) return false;
                } else if (schema.type == "str") {
                    if (!std::holds_alternative<StringTraitParameters>(schema.m_Details)) return fail(prefix + "detail type does not match");
                    const auto &detail = std::get<StringTraitParameters>(schema.m_Details);
                    if (detail.set.empty()) return fail(prefix + "set cannot be empty");
                    if (!validate_set_probabilities(prefix, detail.set.size(), detail.probs)) return false;
                } else if (schema.type == "intset") {
                    if (!std::holds_alternative<IntSetTraitParameters>(schema.m_Details)) return fail(prefix + "detail type does not match");
                    const auto &detail = std::get<IntSetTraitParameters>(schema.m_Details);
                    if (detail.set.empty()) return fail(prefix + "set cannot be empty");
                    if (!validate_set_probabilities(prefix, detail.set.size(), detail.probs)) return false;
                } else if (schema.type == "floatset") {
                    if (!std::holds_alternative<FloatSetTraitParameters>(schema.m_Details)) return fail(prefix + "detail type does not match");
                    const auto &detail = std::get<FloatSetTraitParameters>(schema.m_Details);
                    if (detail.set.empty()) return fail(prefix + "set cannot be empty");
                    if (!validate_set_probabilities(prefix, detail.set.size(), detail.probs)) return false;
                    for (const auto &value : detail.set) {
                        if (!std::isfinite(value.value)) return fail(prefix + "set values must be finite");
                    }
                } else {
                    return fail(prefix + "unsupported type");
                }
                if (!schema.dep_key.empty()) {
                    if (schemas.find(schema.dep_key) == schemas.end()) return fail(prefix + "dependency key does not exist");
                    if (schema.dep_values.empty()) return fail(prefix + "dependency values cannot be empty");
                }
            }
            return true;
        };
        return validate_traits("neuron", NeuronTraits) && validate_traits("link", LinkTraits) && validate_traits("genome", GenomeTraits);
    }

}  // namespace NEAT
