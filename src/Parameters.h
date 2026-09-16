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
 * Description: Definition for the parameters class.
 */

#pragma once

#include <cstdio>
#include <fstream>
#include <functional>
#include <map>
#include <string>
// #include "Genes.h"
#include "Traits.h"
#include "Types.h"
// #include "Species.h"

namespace NEAT {

    // forward
    class Genome;

    // Parent-selection algorithms. LEGACY_SELECTION preserves the historical
    // TruncationSelection/RouletteWheelSelection/TournamentSelection switches.
    // The remaining values make the previously advertised selection modes
    // explicit and mutually exclusive.
    enum SelectionMode { LEGACY_SELECTION = -1, TRUNCATION = 0, ROULETTE, RANK_LINEAR, RANK_EXP, TOURNAMENT, STOCHASTIC, BOLTZMANN };

    // Link-gene recombination used for matching innovations. The historical
    // Genome::Mate boolean maps to MULTIPOINT or AVERAGE.
    enum CrossoverMode { MULTIPOINT = 0, AVERAGE, SINGLE_POINT, BLEND, SIMULATED_BINARY };

    // Distribution used when perturbing (rather than replacing) link weights.
    enum WeightMutationMode { UNIFORM_MUTATION = 0, GAUSSIAN_MUTATION, CAUCHY_MUTATION, POLYNOMIAL_MUTATION };

    // Strategy used to carry a species representative into the next generation.
    // FIRST_REPRESENTATIVE preserves the historical sorted leader behavior.
    enum SpeciesRepresentativeMode { FIRST_REPRESENTATIVE = 0, RANDOM_REPRESENTATIVE, BEST_REPRESENTATIVE, MEDOID_REPRESENTATIVE };

    // Strategy used to turn fractional species quotas into integer offspring.
    enum OffspringAllocationMode { LARGEST_REMAINDER = 0, STOCHASTIC_REMAINDER };

    // Dynamic compatibility-threshold controller.
    enum CompatibilityThresholdMode { LEGACY_COMPATIBILITY_THRESHOLD = 0, PROPORTIONAL_COMPATIBILITY_THRESHOLD };

    // Transformation applied to raw objective values before age adjustment and
    // explicit fitness sharing. SHIFTED_FITNESS_SCALING preserves the historical
    // behavior while computing it in an overflow-safe normalized domain.
    enum FitnessScalingMode { SHIFTED_FITNESS_SCALING = 0, LINEAR_RANK_FITNESS_SCALING, SIGMA_FITNESS_SCALING, BOLTZMANN_FITNESS_SCALING };

    //////////////////////////////////////////////
    // The NEAT Parameters class
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
        unsigned int PopulationSize;

        // Controls the use of speciation. When off, the population will consist of only one species.
        bool Speciation;

        // If true, this enables dynamic compatibility thresholding It will keep the number of species between MinSpecies and MaxSpecies
        bool DynamicCompatibility;

        // Minimum number of species
        unsigned int MinSpecies;

        // Maximum number of species
        unsigned int MaxSpecies;

        // Don't wipe the innovation database each generation?
        bool InnovationsForever;

        // Allow clones or nearly identical genomes to exist simultaneously in the population.
        // This is useful for non-deterministic environments,
        // as the same individual will get more than one chance to prove himself, also
        // there will be more chances the same individual to mutate in different ways.
        // The drawback is greatly increased time for reproduction. If you want to
        // search quickly, yet less efficient, leave this to true.
        bool AllowClones;

        // Keep an archive of genomes and don't allow any new genome to exist in the acrhive or the population
        bool ArchiveEnforcement;

        // Normalize genome size when calculating compatibility
        bool NormalizeGenomeSize;

        // Pointer to a function that specifies custom topology constraints Should return true if the genome FAILS to meet the constraints
        bool (*CustomConstraints)(Genome &g);

        ////////////////////////////////
        // GA Parameters
        ////////////////////////////////

        // AgeGens treshold, meaning if a species is below it, it is considered young
        unsigned int YoungAgeTreshold;

        // Fitness boost multiplier for young species (1.0 means no boost)
        // Make sure it is >= 1.0 to avoid confusion
        Real YoungAgeFitnessBoost;

        // Number of generations without improvement (stagnation) allowed for a species
        unsigned int SpeciesMaxStagnation;

        // Minimum jump in fitness necessary to be considered as improvement. Setting this value to 0.0 makes the system to behave like regular NEAT.
        Real StagnationDelta;

        // AgeGens threshold, meaning if a species if above it, it is considered old
        unsigned int OldAgeTreshold;

        // Multiplier that penalizes old species.
        // Make sure it is < 1.0 to avoid confusion.
        Real OldAgePenalty;

        // Detect competetive coevolution stagnation
        // This kills the worst species of age >N (each X generations)
        bool DetectCompetetiveCoevolutionStagnation;

        // Each X generation..
        int KillWorstSpeciesEach;

        // Of age above..
        int KillWorstAge;

        // Percent of best individuals that are allowed to reproduce. 1.0 = 100%
        Real SurvivalRate;

        // Probability for a baby to result from sexual reproduction (crossover/mating). 1.0 = 100%
        Real CrossoverRate;

        // If a baby results from sexual reproduction, this probability determines if mutation will be performed after crossover. 1.0 = 100% (always mutate
        // after crossover)
        Real OverallMutationRate;

        // Probability for a baby to result from inter-species mating.
        Real InterspeciesCrossoverRate;

        // Probability for a baby gene to result from Multipoint Crossover when mating. 1.0 = 100% The default if the Average mating.
        Real MultipointCrossoverRate;

        // Probability that when doing multipoint crossover,
        // the gene of the fitter parent will be prefered, instead of choosing one at random
        Real PreferFitterParentRate;

        // Performing truncation selection or not? (goes first)
        bool TruncationSelection;

        // Performing roulette wheel selection or not?
        bool RouletteWheelSelection;

        // If true, will do tournament selection
        bool TournamentSelection;

        // For tournament selection
        unsigned int TournamentSize;

        // Fraction of individuals to be copied unchanged. Elitism is retained as
        // a source-compatible spelling used by older MultiNEAT clients.
        union {
            Real EliteFraction;
            Real Elitism;
        };

        ///////////////////////////////////
        // Phased Search parameters   //
        ///////////////////////////////////

        // Using phased search or not
        bool PhasedSearching;

        // Using delta coding or not
        bool DeltaCoding;

        // What is the MPC + base MPC needed to begin simplifying phase
        unsigned int SimplifyingPhaseMPCTreshold;

        // How many generations of global stagnation should have passed to enter simplifying phase
        unsigned int SimplifyingPhaseStagnationTreshold;

        // How many generations of MPC stagnation are needed to turn back on complexifying
        unsigned int ComplexityFloorGenerations;

        /////////////////////////////////////
        // Novelty Search parameters       //
        /////////////////////////////////////

        // the K constant
        unsigned int NoveltySearch_K;

        // Sparseness treshold. Add to the archive if above
        Real NoveltySearch_P_min;

        // Dynamic Pmin?
        bool NoveltySearch_Dynamic_Pmin;

        // How many evaluations should pass without adding to the archive in order to lower Pmin
        unsigned int NoveltySearch_No_Archiving_Stagnation_Treshold;

        // How should it be multiplied (make it less than 1.0)
        Real NoveltySearch_Pmin_lowering_multiplier;

        // Not lower than this value
        Real NoveltySearch_Pmin_min;

        // How many one-after-another additions to the archive should
        // pass in order to raise Pmin
        unsigned int NoveltySearch_Quick_Archiving_Min_Evaluations;

        // How should it be multiplied (make it more than 1.0)
        Real NoveltySearch_Pmin_raising_multiplier;

        // Per how many evaluations to recompute the sparseness
        unsigned int NoveltySearch_Recompute_Sparseness_Each;

        ///////////////////////////////////
        // Mutation parameters
        ///////////////////////////////////

        // Probability for a baby to be mutated with the Add-Neuron mutation.
        Real MutateAddNeuronProb;

        // Allow splitting of any recurrent links
        bool SplitRecurrent;

        // Allow splitting of looped recurrent links
        bool SplitLoopedRecurrent;

        // Maximum number of tries to find a link to split
        int NeuronTries;

        // Probability for a baby to be mutated with the Add-Link mutation
        Real MutateAddLinkProb;

        // Probability for a new incoming link to be from the bias neuron;
        Real MutateAddLinkFromBiasProb;

        // Probability for a baby to be mutated with the Remove-Link mutation
        Real MutateRemLinkProb;

        // Probability for a baby that a simple neuron will be replaced with a link
        Real MutateRemSimpleNeuronProb;

        // Maximum number of tries to find 2 neurons to add/remove a link
        unsigned int LinkTries;

        // Maximum number of links in the genome (originals not counted). -1 is unlimited
        int MaxLinks;

        // Maximum number of neurons in the genome (originals not counted). -1 is unlimited
        int MaxNeurons;

        // Probability that a link mutation will be made recurrent
        Real RecurrentProb;

        // Probability that a recurrent link mutation will be looped
        Real RecurrentLoopProb;

        // Probability for a baby's weights to be mutated
        Real MutateWeightsProb;

        // Probability for a severe (shaking) weight mutation
        Real MutateWeightsSevereProb;

        // Probability for a particular gene to be mutated. 1.0 = 100%
        Real WeightMutationRate;

        // Probability for a particular gene to be mutated via replacement of the weight. 1.0 = 100%
        Real WeightReplacementRate;

        // Maximum perturbation for a weight mutation
        Real WeightMutationMaxPower;

        // Maximum magnitude of a replaced weight
        Real WeightReplacementMaxPower;

        // Maximum weight
        Real MaxWeight;

        // Minimum weight
        Real MinWeight;

        // Probability for a baby's A activation function parameters to be perturbed
        Real MutateActivationAProb;

        // Probability for a baby's B activation function parameters to be perturbed
        Real MutateActivationBProb;

        // Maximum magnitude for the A parameter perturbation
        Real ActivationAMutationMaxPower;

        // Maximum magnitude for the B parameter perturbation
        Real ActivationBMutationMaxPower;

        // Maximum magnitude for time costants perturbation
        Real TimeConstantMutationMaxPower;

        // Maximum magnitude for biases perturbation
        Real BiasMutationMaxPower;

        // Activation parameter A min/max
        Real MinActivationA;
        Real MaxActivationA;

        // Activation parameter B min/max
        Real MinActivationB;
        Real MaxActivationB;

        // Probability for a baby that an activation function type will be changed for a single neuron considered a structural mutation because of the large
        // impact on fitness
        Real MutateNeuronActivationTypeProb;

        // Probabilities for a particular activation function appearance
        Real ActivationFunction_SignedSigmoid_Prob;
        Real ActivationFunction_UnsignedSigmoid_Prob;
        Real ActivationFunction_Tanh_Prob;
        Real ActivationFunction_TanhCubic_Prob;
        Real ActivationFunction_SignedStep_Prob;
        Real ActivationFunction_UnsignedStep_Prob;
        Real ActivationFunction_SignedGauss_Prob;
        Real ActivationFunction_UnsignedGauss_Prob;
        Real ActivationFunction_Abs_Prob;
        Real ActivationFunction_SignedSine_Prob;
        Real ActivationFunction_UnsignedSine_Prob;
        Real ActivationFunction_Linear_Prob;
        Real ActivationFunction_Relu_Prob;
        Real ActivationFunction_Softplus_Prob;
        Real ActivationFunction_SpikingLIF_Prob;
        Real ActivationFunction_SpikingAdaptiveLIF_Prob;
        Real ActivationFunction_SpikingIzhikevich_Prob;
        Real ActivationFunction_McCullochPitts_Prob;

        // Probability for a baby's neuron time constant values to be mutated
        Real MutateNeuronTimeConstantsProb;

        // Probability for a baby's neuron bias values to be mutated
        Real MutateNeuronBiasesProb;

        // Time constant range
        Real MinNeuronTimeConstant;
        Real MaxNeuronTimeConstant;

        // Bias range
        Real MinNeuronBias;
        Real MaxNeuronBias;

        /////////////////////////////////////
        // Spiking-neural-network parameters
        /////////////////////////////////////

        // Probabilities that the corresponding built-in parameter mutation is
        // selected during reproduction. Zero preserves historical evolution.
        Real MutateNeuronSpikingParametersProb;
        Real MutateLinkSpikingParametersProb;

        // Per-field mutation rate and the maximum fraction of a field's allowed
        // range used by one perturbation.
        Real SpikingParameterMutationRate;
        Real SpikingParameterMutationPower;

        // The canonical model gives any active inhibitory afferent an absolute
        // veto. These probabilities make that rule heritable while allowing
        // weighted-threshold variants when desired.
        Real InitialMCPInhibitoryVetoProb;
        Real MutateMCPInhibitoryVetoProb;

        // Evolvable LIF and adaptive-LIF ranges.
        Real MinSpikingTimeConstant;
        Real MaxSpikingTimeConstant;
        Real MinSpikeThreshold;
        Real MaxSpikeThreshold;
        Real MinResetPotential;
        Real MaxResetPotential;
        Real MinRestingPotential;
        Real MaxRestingPotential;
        Real MinRefractoryPeriod;
        Real MaxRefractoryPeriod;
        Real MinMembraneResistance;
        Real MaxMembraneResistance;
        Real MinAdaptationTimeConstant;
        Real MaxAdaptationTimeConstant;
        Real MinAdaptationIncrement;
        Real MaxAdaptationIncrement;
        Real MinSpikeRateTimeConstant;
        Real MaxSpikeRateTimeConstant;

        // Evolvable Izhikevich a/b/c/d ranges.
        Real MinIzhikevichA;
        Real MaxIzhikevichA;
        Real MinIzhikevichThreshold;
        Real MaxIzhikevichThreshold;
        Real MinIzhikevichB;
        Real MaxIzhikevichB;
        Real MinIzhikevichC;
        Real MaxIzhikevichC;
        Real MinIzhikevichD;
        Real MaxIzhikevichD;

        // Evolvable current-based exponential synapse and STDP ranges.
        Real MinSynapticDelay;
        Real MaxSynapticDelay;
        Real MinSynapticTimeConstant;
        Real MaxSynapticTimeConstant;
        Real InitialSTDPEnabledProb;
        Real MinSTDPPlus;
        Real MaxSTDPPlus;
        Real MinSTDPMinus;
        Real MaxSTDPMinus;
        Real MinSTDPTau;
        Real MaxSTDPTau;

        /////////////////////////////////////
        // Speciation parameters
        /////////////////////////////////////

        // Percent of disjoint genes importance
        Real DisjointCoeff;

        // Percent of excess genes importance
        Real ExcessCoeff;

        // Node-specific activation parameter A difference importance
        Real ActivationADiffCoeff;

        // Node-specific activation parameter B difference importance
        Real ActivationBDiffCoeff;

        // Average weight difference importance
        Real WeightDiffCoeff;

        // Average time constant difference importance
        Real TimeConstantDiffCoeff;

        // Average bias difference importance
        Real BiasDiffCoeff;

        // Activation function type difference importance
        Real ActivationFunctionDiffCoeff;

        // Distance contributed by matching spiking neuron and synapse
        // parameters. Defaults are zero for compatibility.
        Real SpikingNeuronDiffCoeff;
        Real SpikingLinkDiffCoeff;

        // Compatibility treshold
        Real CompatTreshold;

        // Minumal value of the compatibility treshold
        Real MinCompatTreshold;

        // Modifier per generation for keeping the species stable
        Real CompatTresholdModifier;

        // Per how many generations to change the treshold
        unsigned int CompatTreshChangeInterval_Generations;

        // Per how many evaluations to change the treshold
        unsigned int CompatTreshChangeInterval_Evaluations;

        // What is the minimal difference needed for not to be a clone
        Real MinDeltaCompatEqualGenomes;

        // How many times to test a genome for constraint failure or being a clone (when AllowClones=False)
        int ConstraintTrials;

        /////////////////////////////
        // Genome properties params
        /////////////////////////////

        // When true, don't have a special bias neuron and treat all inputs equal
        bool DontUseBiasNeuron;
        bool AllowLoops;

        /////////////////////////////
        // ES HyperNEAT params
        /////////////////////////////

        Real DivisionThreshold;

        Real VarianceThreshold;

        // Used for Band prunning.
        Real BandThreshold;

        // Max and Min Depths of the quadtree
        unsigned int InitialDepth;

        unsigned int MaxDepth;

        // How many hidden layers before connecting nodes to output. At 0 there is one hidden layer. At 1, there are two and so on.
        unsigned int IterationLevel;

        // The Bias value for the CPPN queries.
        Real CPPN_Bias;

        // Quadtree / octree dimensions
        // The range of the tree. Typically set to 2,
        Real Width;
        Real Height;
        Real Depth;

        // The (x, y, z) coordinates of the tree
        Real Qtree_X;

        Real Qtree_Y;
        Real Qtree_Z;

        // Use Link Expression output
        bool Leo;

        // Threshold above which a connection is expressed
        Real LeoThreshold;

        // Use geometric seeding. Currently only along the X axis. 1
        bool LeoSeed;
        bool GeometrySeed;

        /////////////////////////////////////
        // Universal traits
        /////////////////////////////////////
        std::map<std::string, TraitParameters> NeuronTraits;
        std::map<std::string, TraitParameters> LinkTraits;
        std::map<std::string, TraitParameters> GenomeTraits;
        Real MutateNeuronTraitsProb;
        Real MutateLinkTraitsProb;
        Real MutateGenomeTraitsProb;

        /////////////////////////////////////
        // Advanced algorithm controls
        /////////////////////////////////////

        // LEGACY_SELECTION keeps all historical selection switches functional.
        SelectionMode ParentSelectionMode;

        // Baker linear-ranking pressure in [1, 2]. A value of 1 is uniform and
        // 2 gives the strongest valid linear ranking pressure.
        Real RankSelectionPressure;

        // Positive exponential decay applied to normalized rank.
        Real RankSelectionExponent;

        // Positive softmax temperature for Boltzmann selection.
        Real BoltzmannTemperature;

        // Additional crossover probabilities. MultipointCrossoverRate remains
        // unchanged; any probability left over selects average crossover.
        Real SinglePointCrossoverRate;
        Real BlendCrossoverRate;
        Real SimulatedBinaryCrossoverRate;

        // BLX-alpha expansion and SBX distribution index.
        Real CrossoverBlendAlpha;
        Real CrossoverSBXEta;

        // The default UNIFORM_MUTATION exactly preserves historical mutation.
        WeightMutationMode WeightMutationDistribution;

        // Gaussian standard-deviation multiplier, Cauchy scale multiplier, and
        // bounded polynomial-mutation distribution index.
        Real WeightMutationSigma;
        Real WeightMutationCauchyScale;
        Real WeightMutationPolynomialEta;

        // Species representatives can remain leader-based for exact historical
        // behavior, be sampled, or use a compatibility-distance medoid. A zero
        // candidate limit makes MEDOID_REPRESENTATIVE examine every individual.
        SpeciesRepresentativeMode SpeciesRepresentativeSelection;
        unsigned int RepresentativeSelectionCandidates;

        // Exact offspring allocation controls. MinSpeciesSize protects niches
        // that receive a non-zero quota; SpeciesElitism also guarantees a quota
        // to the best N species. Defaults preserve historical apportionment.
        OffspringAllocationMode OffspringAllocation;
        unsigned int MinSpeciesSize;
        unsigned int SpeciesElitism;

        // Multiplier applied to a stagnant non-champion species. This replaces
        // the previously hard-coded value while keeping that value as default.
        Real StagnationPenalty;

        // Proportional control is smoother than the historical one-step
        // threshold update. TargetSpecies == 0 uses the midpoint of the existing
        // MinSpecies/MaxSpecies interval.
        CompatibilityThresholdMode CompatibilityThresholdControl;
        unsigned int TargetSpecies;
        Real CompatibilityThresholdGain;
        Real MaxCompatTreshold;

        // Optional strict evaluation guards. They are disabled by default so
        // established workflows that rely on Epoch() marking genomes evaluated
        // continue to work.
        bool RequireEvaluatedGenomes;
        bool RejectNonFiniteFitness;

        // Expected number of mutation operators applied to a mutated offspring.
        // Stagnation adaptation multiplies this budget after the configured
        // generation and is disabled when AdaptiveMutationRate is zero.
        Real MutationOperatorsPerOffspring;
        unsigned int AdaptiveMutationStart;
        Real AdaptiveMutationRate;
        Real AdaptiveMutationMaxFactor;

        // Population-wide objective transforms used for offspring allocation.
        // These are independent from the within-species parent selector.
        FitnessScalingMode FitnessScaling;
        Real FitnessRankPressure;
        Real FitnessSigmaScale;
        Real FitnessBoltzmannTemperature;

        /////////////////////////////////////
        // Constructors
        /////////////////////////////////////

        // Load defaults
        Parameters();

        ////////////////////////////////////
        // Methods
        ////////////////////////////////////

        // Load the parameters from a file returns 0 on success
        int Load(const char *filename);
        // Load the parameters from an already opened stream for reading
        int Load(std::istream &a_DataFile);

        void Save(const char *filename);
        // Saves the parameters to an already opened file for writing
        void Save(FILE *a_fstream);

        // resets the parameters to built-in defaults
        void Reset();

        // Opt-in preset for evolving mixed spiking topologies. Existing users
        // retain rate-network defaults until this is called or fields are set
        // explicitly.
        void ConfigureSpiking(bool enable_stdp = false);

        // Opt-in preset for pure McCulloch-Pitts evolution. Neuron thresholds,
        // refractory periods, axonal delays, and the inhibitory-veto rule remain
        // evolvable through the normal spiking mutation operators.
        void ConfigureMcCullochPitts(bool inhibitory_veto = true, bool enable_stdp = false);

        // Complete, round-trippable persistence. Function callbacks are omitted.
        std::string Serialize() const;
        static Parameters Deserialize(const std::string &data);

        // Checks ranges and cross-field invariants without changing values.
        bool Validate(std::string *error = nullptr) const;

        // The legacy function-pointer field cannot represent capturing
        // callables. These helpers preserve it for existing C++ users while
        // allowing each Parameters instance to own an independent callable.
        void SetCustomConstraintsFunction(std::function<bool(Genome &)> callback);
        std::function<bool(Genome &)> GetCustomConstraintsFunction() const;
        bool FailsCustomConstraints(Genome &genome) const;

       private:
        std::function<bool(Genome &)> m_CustomConstraintsFunction;
    };

}  // namespace NEAT
