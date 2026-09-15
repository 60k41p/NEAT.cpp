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
 * File:        Genome.h
 * Description: The NEAT genotype: innovation-numbered neuron/link gene lists with fitness bookkeeping,
 *              plus the full operator set — phenotype construction (buildPhenotype/buildHyperNEATPhenotype),
 *              structural and parametric mutation (mutate*), multipoint mating (mate), compatibility
 *              measurement (compatibilityDistance/isCompatibleWith) and persistence (save/load).
 *              Genomes are evaluated externally (setFitness), speciated by Population and activated as
 *              NeuralNetwork phenotypes.
 *
 * References: Stanley & Miikkulainen, "Evolving Neural Networks through Augmenting Topologies" (2002),
 *             Sections 2-5 (genotype, historical markings, speciation, complexification); see
 *             references/Evolving Neural Networks through Augmenting Topologies.pdf.md. HyperNEAT decoding:
 *             Stanley et al. (2009) with the substrate in src/Substrate.h. Intra-repo users: src/Population.h,
 *             src/Species.h, src/Innovation.h, tests/TestGenome.cpp, tests/TestEvolution.cpp.
 */

#pragma once

#include <queue>
#include <vector>

#include "AssertMacros.h"
#include "Genes.h"
#include "Innovation.h"
#include "NeuralNetwork.h"
#include "PhenotypeBehavior.h"
#include "Random.h"
#include "Substrate.h"
#include "Types.h"

namespace NEAT {

    //////////////////////////////////////////////
    // The Genome class
    //////////////////////////////////////////////

    // forward
    class Innovation;

    class InnovationDatabase;

    class PhenotypeBehavior;

    // Draws an activation function honoring the per-function probabilities in Parameters.
    extern ActivationFunction getRandomActivation(Parameters &parameters, RNG &rng);

    // Seed topologies for a fresh genome: perceptron-like minimal net, or a layered net with numHidden units.
    enum GenomeSeedType { PERCEPTRON = 0, LAYERED = 1 };

    // Seed descriptor consumed by the Genome(parameters, init) constructor. Note: numInputs counts the bias
    // input, so an XOR task with 2 problem inputs uses 3 (see tests/TestEvolution.cpp).
    class GenomeInitStruct {
       public:
        // Input count including the bias neuron (unless Parameters::dontUseBiasNeuron).
        int numInputs;
        // Hidden-unit count, used only when seedType == LAYERED.
        int numHidden;
        // Output count.
        int numOutputs;
        // Feature-select start (minimal connectivity evolved upward); keep false for dense seeds.
        bool fsNeat;
        // Activation functions for created output/hidden neurons.
        ActivationFunction outputActType;
        ActivationFunction hiddenActType;
        // Which seed topology to build.
        GenomeSeedType seedType;
        // Layer count for layered seeds.
        int numLayers;
        // Initial link count for FS-NEAT seeds.
        int fsNeatLinks;

        GenomeInitStruct() {
            numInputs = 1;
            numHidden = 0;
            numOutputs = 1;
            fsNeat = 0;
            fsNeatLinks = 1;
            hiddenActType = UNSIGNED_SIGMOID;
            outputActType = UNSIGNED_SIGMOID;
            seedType = GenomeSeedType::PERCEPTRON;
            numLayers = 0;
        }
    };

    // The evolvable genotype. Gene lists are public for direct inspection; identity/fitness/depth go
    // through accessors. Copyable and assignable (deep copy including traits and behavior pointer state).
    class Genome {
        /////////////////////
        // Members
        /////////////////////
       private:
        // Unique genome identifier (see Population::nextGenomeID_).
        int id_;

        // Input/output counts including the bias neuron when enabled.
        int numInputs_;
        int numOutputs_;

        // Raw fitness assigned externally via setFitness().
        Real fitness_;

        // Fitness after species sharing (see Species::adjustFitness()).
        Real adjustedFitness_;

        // Longest input-to-output path (see calculateDepth()).
        int depth_;

        // Offspring quota for the next generation (see Population::countOffspring()).
        Real offspringAmount_;

        ////////////////////
        // Structural queries (private helpers)

        // Whether a neuron ID exists in neuronGenes_.
        bool hasNeuronID(int id) const;

        // Whether a directed link n1id -> n2id exists in linkGenes_.
        bool hasLink(int n1id, int n2id) const;

        // Whether a link with the given innovation ID exists.
        bool hasLinkByInnovID(int id) const;

        // Deletes the link gene (plus orphaned structure on cleanup()).
        void removeLinkGene(int innovid);

        // Deletes the neuron gene and all links touching it.
        void removeNeuronGene(int id);

        // Number of links fed by / feeding into the given neuron ID.
        int linksInputtingFrom(int id) const;

        // Number of links feeding into the given neuron ID.
        int linksOutputtingTo(int id) const;

        // Longest backward path from the neuron to an input (cycle-unsafe; see hasLoops() guard).
        unsigned int neuronDepth(int neuronID, unsigned int depth);

        // Whether the neuron is isolated or feeds no output (cleanup target).
        bool isDeadEndNeuron(int id) const;

       public:
        // Neuron/link gene lists (kept sorted by ID/innovation number; see sortGenes()).
        std::vector<NeuronGene> neuronGenes_;
        std::vector<LinkGene> linkGenes_;

        // Genome-level traits (evolved via mutateGenomeTraits(); see src/Traits.h).
        Gene genomeGene_;

        // Whether fitness is current (steady-state evolution skips unevaluated members).
        bool evaluated_;

        // Complexity at seeding time (baseline for phased-search MPC comparisons).
        int initialNumNeurons_;
        int initialNumLinks_;

        // Behavior descriptor for novelty search (owned by the caller; may be nullptr otherwise).
        PhenotypeBehavior *phenotypeBehavior_;

        ////////////////////////////
        // Constructors
        ////////////////////////////

        // Builds an empty genome.
        Genome();

        // Deep-copy constructor.
        Genome(const Genome &g);

        // Deep-copy assignment (self-assignment safe).
        Genome &operator=(const Genome &g);

        // Compares by genome ID (identity key; ignores topology and fitness).
        bool operator==(Genome const &other) const { return id_ == other.id_; }

        // Loads a genome from a saved file path.
        Genome(const char *filename);

        // Loads a genome from an open input stream positioned at a GenomeStart marker.
        Genome(std::ifstream &dataFile);

        // Builds a minimal seed genome from the init descriptor (perceptron or layered).
        Genome(const Parameters &parameters, const GenomeInitStruct &initStruct);

        ////////////////////////////
        // Methods
        ////////////////////////////

        ////////////////////
        // Gene accessors (copies; use getNeuronIndex()/getLinkIndex() plus neuronGenes_/linkGenes_ to mutate in place)

        // Copies out the neuron/link gene with the given ID/innovation number (throws when absent).
        NeuronGene getNeuronByID(int id) const;

        NeuronGene getNeuronByIndex(int index) const;

        LinkGene getLinkByInnovID(int id) const;

        LinkGene getLinkByIndex(int index) const;

        // Position of the neuron ID in neuronGenes_ (throws when absent).
        int getNeuronIndex(int id) const;

        // Position of the innovation ID in linkGenes_ (throws when absent).
        int getLinkIndex(int innovid) const;

        unsigned int numNeurons() const { return static_cast<unsigned int>(neuronGenes_.size()); }

        unsigned int numLinks() const { return static_cast<unsigned int>(linkGenes_.size()); }

        unsigned int numInputs() const { return static_cast<unsigned int>(numInputs_); }

        unsigned int numOutputs() const { return static_cast<unsigned int>(numOutputs_); }

        // Display-coordinate setters by gene position.
        void setNeuronXY(unsigned int index, int x, int y);

        void setNeuronX(unsigned int index, int x);

        void setNeuronY(unsigned int index, int y);

        // Raw / shared fitness accessors (fitness is set externally after evaluation).
        Real getFitness() const;

        Real getAdjFitness() const;

        void setFitness(Real f);

        void setAdjFitness(Real af);

        int getID() const;

        void setID(int id);

        // Network depth (see calculateDepth()).
        unsigned int getDepth() const;

        void setDepth(unsigned int d);

        // Whether any neuron is isolated or feeds no output.
        bool hasDeadEnds() const;

        // Whether any directed cycle exists.
        bool hasLoops();

        // Whether the genome violates structural constraints: dead ends, empty nets, disallowed loops
        // or the caller-supplied Parameters::customConstraints predicate. Defined in Genome.cpp.
        bool failsConstraints(const Parameters &parameters);

        // Offspring quota accessors (see Population::countOffspring()).
        Real getOffspringAmount() const;

        void setOffspringAmount(Real oa);

        // Decodes the genotype into a runnable phenotype (direct encoding).
        void buildPhenotype(NeuralNetwork &net);

        // Copies phenotype weight changes back into the genome's link genes.
        void derivePhenotypicChanges(NeuralNetwork &net);

        // Decodes via the CPPN over the given substrate (indirect HyperNEAT encoding).
        void buildHyperNEATPhenotype(NeuralNetwork &net, Substrate &subst);

        // Persists the genome in the text format understood by the Genome(path) constructor.
        void save(const char *filename);

        // Appends the genome to an open file (used by Population::save()).
        void save(FILE *fstream);

        // Dumps one trait map / all gene traits to stdout (diagnostics).
        void printTraits(std::map<std::string, Trait> &traits);
        void printAllTraits();

        // Maximum neuron ID / innovation number in use (next-ID computation).
        int getLastNeuronID() const;

        // Maximum innovation number in use.
        int getLastInnovationID() const;

        // Orders neuron genes by ID and link genes by innovation number.
        void sortGenes();

        // Orders genomes fittest-first for sorting.
        friend bool operator<(const Genome &lhs, const Genome &rhs) { return (lhs.fitness_ > rhs.fitness_); }

        // Whether the compatibility distance to g is within Parameters::compatTreshold.
        bool isCompatibleWith(Genome &g, Parameters &parameters);

        // Weighted compatibility distance to g (disjoint/excess/weight/activation/trait terms).
        Real compatibilityDistance(Genome &g, Parameters &parameters);

        // Recomputes depth_ as the longest input-to-output path.
        void calculateDepth();

        ////////////
        // Mutation (each returns true when it changed the genome; rates come from Parameters)
        ////////////

        // Splits an existing link with a new neuron, registering both innovations.
        bool mutateAddNeuron(InnovationDatabase &innovs, const Parameters &parameters, RNG &rng);

        // Adds a feed-forward (or, with Parameters::recurrentProb, recurrent) link between unconnected neurons.
        bool mutateAddLink(InnovationDatabase &innovs, const Parameters &parameters, RNG &rng);

        // Deletes a random link, then removes newly orphaned neurons/links.
        bool mutateRemoveLink(RNG &rng);

        // Bypasses a 1-in/1-out hidden neuron with a direct link, then deletes the neuron.
        bool mutateRemoveSimpleNeuron(InnovationDatabase &innovs, const Parameters &parameters, RNG &rng);

        // Perturbs link weights (severe shake vs. per-gene perturbation per Parameters).
        bool mutateLinkWeights(const Parameters &parameters, RNG &rng);

        // Resets every link weight uniformly into [-R .. R] (see Parameters::weightReplacementMaxPower).
        void randomizeLinkWeights(const Parameters &parameters, RNG &rng);

        // Re-rolls every gene trait from its parameter definition.
        void randomizeTraits(const Parameters &parameters, RNG &rng);

        // Perturbs the activation slope (A) / shift (B) of random neurons.
        bool mutateNeuronActivationsA(const Parameters &parameters, RNG &rng);

        // Perturbs the activation shift (B) of random neurons.
        bool mutateNeuronActivationsB(const Parameters &parameters, RNG &rng);

        // Reassigns one random neuron's activation function (structural-scale change).
        bool mutateNeuronActivationType(const Parameters &parameters, RNG &rng);

        // Perturbs leaky-integrator time constants / biases of random neurons.
        bool mutateNeuronTimeConstants(const Parameters &parameters, RNG &rng);

        // Perturbs biases of random neurons.
        bool mutateNeuronBiases(const Parameters &parameters, RNG &rng);

        // Perturbs neuron / link / genome-level universal traits.
        bool mutateNeuronTraits(const Parameters &parameters, RNG &rng);

        // Perturbs link universal traits.
        bool mutateLinkTraits(const Parameters &parameters, RNG &rng);

        // Perturbs genome-level universal traits.
        bool mutateGenomeTraits(const Parameters &parameters, RNG &rng);

        ///////////
        // Mating
        ///////////

        // Crosses this genome with dad, returning the offspring. Multipoint genes are picked randomly
        // (or averaged when averagemating); disjoint/excess genes come from the fitter parent, or the
        // smaller genome on tied fitness. Set interspecies to allow cross-species mating.
        Genome mate(Genome &dad, bool averagemating, bool interspecies, RNG &rng, Parameters &parameters);

        //////////
        // Utility
        //////////

        // Removes isolated/stranded structure. Returns true when anything was deleted.
        bool cleanup();

        // Evaluation-flag accessors for steady-state evolution.
        bool isEvaluated() const;

        void setEvaluated();

        void resetEvaluated();

#if 0  // Intentionally inactive ES-HyperNEAT prototype (kept for reference; do not enable without porting off boost and fixing the quadtree code).

        /////////////////////////////////////////////
        // Evolvable Substrate HyperNEAT
        ////////////////////////////////////////////


        // A connection between two points. Stores weight and the coordinates of the points
        struct TempConnection
        {
            std::vector<Real> source;
            std::vector<Real> target;
            Real weight;

            TempConnection()
            {
                source.reserve(3);
                target.reserve(3);
                weight = 0;
            }

            TempConnection(std::vector<Real> source, std::vector<Real> target,
                           Real weight)
            {
                source = source;
                target = target;
                weight = weight;
                source.reserve(3);
                target.reserve(3);
            }

            TempConnection(std::vector<Real> source, std::vector<Real> target, Real weight, unsigned int coordSize)
            {
                source = source;
                target = target;
                weight = weight;
            }

            ~TempConnection()
            {};

            bool operator==(const TempConnection &rhs) const
            {
                return (source == rhs.source && target == rhs.target);
            }

            bool operator!=(const TempConnection &rhs) const
            {
                return (source != rhs.source && target != rhs.target);
            }
        };

        // A quadpoint in the HyperCube.
        struct QuadPoint
        {
            Real x;
            Real y;
            Real z;
            Real width;
            Real weight;
            Real height;
            Real variance;
            int level;
            // Do I use this?
            Real leo;


            std::vector<boost::shared_ptr<QuadPoint> > children;

            QuadPoint()
            {
                x = y = z = width = height = weight = variance = leo = 0;
                level = 0;
                children.reserve(4);
            }

            QuadPoint(Real x, Real y, Real width, Real height, int level)
            {
                x = x;
                y = y;
                z = 0.0;
                width = width;
                height = height;
                level = level;
                weight = 0.0;
                leo = 0.0;
                variance = 0.0;
                children.reserve(4);
                children.clear();
            }

            // Mind the Z
            QuadPoint(Real x, Real y, Real z, Real width, Real height,
                      int level)
            {
                x = x;
                y = y;
                z = z;
                width = width;
                height = height;
                level = level;
                weight = 0.0;
                variance = 0.0;
                leo = 0.0;
                children.reserve(4);
                children.clear();
            }

            ~QuadPoint()
            {
            };
        };


        struct NTree
        {
            std::vector<Real> coord;
            Real weight;
            Real varience;
            int lvl;
            Real width;
            Real leo = 0.0;
            std::vector<boost::shared_ptr<NTree> > children;

            NTree(std::vector<Real> coordIn, Real wdth, Real level)
            {
                width = wdth;
                lvl = level;
                coord = coordIn;
            };

        public:

            void setChildren()
            {
                for(unsigned int ix = 0; ix < 2**coord.size(); ix++){
                    std::string sumPermute = toBinary(ix, coord.size());
                    std::vector<Real> childCoords;
                    int childParamLen = sumPermute.length();
                    childCoords.reserve(childParamLen);
                    for(unsigned int signIx = 0; signIx < childParamLen; signIx++)
                    {
                        if(sumPermute[signIx] == "0")
                        {
                            childCoords.push_back(coord[signIx] + width/2.0);
                        }
                        else
                        {
                            childCoords.push_back(coord[signIx] - width/2.0);
                        }
                        children.push_back(new NTree(childCoords, width/2.0, sslvl+1));
                    }
                }
            }

            string toBinary(unsigned int n, int minLen)
            {
                std::string r;
                while(n!=0)
                {
                    r=(n%2==0 ?"0":"1")+r; n/=2;

                }
                if(r.length() < minLen)
                {
                    int diff = minLen - r.length();
                    for(unsigned int x = 0; x < diff; x++)
                    {
                        r = '0' +r;
                    }
                }
                return r;
            }
        };
        void buildESHyperNEATPhenotypeND(NeuralNetwork &net, Substrate &subst, Parameters &params);
        void buildESHyperNEATPhenotype(NeuralNetwork &net, Substrate &subst, Parameters &params);

        void divideInitialize(const std::vector<Real> &node,
                              boost::shared_ptr<QuadPoint> &root,
                              NeuralNetwork &cppn, Parameters &params,
                              const bool &outgoing, const Real &zCoord);

        void pruneExpress(const std::vector<Real> &node,
                          boost::shared_ptr<QuadPoint> &root, NeuralNetwork &cppn,
                          Parameters &params, std::vector<Genome::TempConnection> &connections,
                          const bool &outgoing);
        void divideInitializeND(const std::vector<Real> &node,
                              boost::shared_ptr<NTree> &root,
                              NeuralNetwork &cppn, Parameters &params,
                              const bool &outgoing, const Real &zCoord);

        void pruneExpressND(const std::vector<Real> &node,
                          boost::shared_ptr<NTree> &root, NeuralNetwork &cppn,
                          Parameters &params, std::vector<Genome::TempConnection> &connections,
                          const bool &outgoing);


        void collectValues(std::vector<Real> &vals, boost::shared_ptr<QuadPoint> &point);

        Real variance(boost::shared_ptr<QuadPoint> &point);

        void cleanNet(std::vector<Connection> &connections, unsigned int inputCount,
                       unsigned int outputCount, unsigned int hiddenCount);
#endif
    };

}  // namespace NEAT
