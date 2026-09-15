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
#include <cmath>
#include <cstddef>
#include <fstream>
#include <iomanip>
#include <limits>
#include <numeric>
#include <set>
#include <sstream>
#include <stdexcept>

#include "AssertMacros.h"
#include "Genome.h"
#include "Parameters.h"
#include "PhenotypeBehavior.h"
#include "Random.h"
#include "Species.h"
#include "Utils.h"

namespace NEAT {

    // Transforms raw objective values into non-negative allocation weights.
    // SHIFTED (default) preserves the historical shift in an overflow-safe
    // normalized domain; the rank/sigma/Boltzmann modes are opt-in.
    inline std::vector<double> TransformFitnessValues(const std::vector<double> &raw_fitness, const Parameters &parameters) {
        if (raw_fitness.empty()) return {};

        constexpr long double epsilon = 1.0e-12L;
        std::vector<double> transformed(raw_fitness.size(), 0.0);
        std::vector<std::size_t> finite_indices;
        finite_indices.reserve(raw_fitness.size());
        for (std::size_t i = 0; i < raw_fitness.size(); ++i) {
            if (std::isfinite(raw_fitness[i])) finite_indices.push_back(i);
        }
        if (finite_indices.empty()) {
            std::fill(transformed.begin(), transformed.end(), 1.0);
            return transformed;
        }

        switch (parameters.FitnessScaling) {
            case SHIFTED_FITNESS_SCALING: {
                long double minimum = static_cast<long double>(raw_fitness[finite_indices.front()]);
                long double maximum = minimum;
                for (const std::size_t index : finite_indices) {
                    minimum = std::min(minimum, static_cast<long double>(raw_fitness[index]));
                    maximum = std::max(maximum, static_cast<long double>(raw_fitness[index]));
                }
                if (minimum <= 0.0L) {
                    constexpr long double legacy_offset = 1.0e-7L;
                    long double scale = std::max(std::abs(minimum), std::abs(maximum));
                    if (scale <= 0.0L) scale = 1.0L;
                    if (scale <= legacy_offset) {
                        const long double normalizer = maximum - minimum + legacy_offset;
                        for (const std::size_t index : finite_indices) {
                            transformed[index] = static_cast<double>((static_cast<long double>(raw_fitness[index]) - minimum + legacy_offset) / normalizer);
                        }
                    } else {
                        const long double scaled_minimum = minimum / scale;
                        const long double scaled_offset = legacy_offset / scale;
                        const long double scaled_range = maximum / scale - scaled_minimum;
                        const long double normalizer = scaled_range + scaled_offset;
                        for (const std::size_t index : finite_indices) {
                            transformed[index] =
                                static_cast<double>((static_cast<long double>(raw_fitness[index]) / scale - scaled_minimum + scaled_offset) / normalizer);
                        }
                    }
                } else {
                    for (const std::size_t index : finite_indices) {
                        transformed[index] = static_cast<double>(static_cast<long double>(raw_fitness[index]) / maximum);
                    }
                }
                break;
            }

            case LINEAR_RANK_FITNESS_SCALING: {
                std::stable_sort(finite_indices.begin(), finite_indices.end(),
                                 [&raw_fitness](std::size_t lhs, std::size_t rhs) { return raw_fitness[lhs] > raw_fitness[rhs]; });
                const long double count = static_cast<long double>(finite_indices.size());
                std::size_t first = 0;
                while (first < finite_indices.size()) {
                    std::size_t last = first + 1;
                    while (last < finite_indices.size() && raw_fitness[finite_indices[last]] == raw_fitness[finite_indices[first]]) {
                        ++last;
                    }
                    const long double average_rank = (static_cast<long double>(first) + static_cast<long double>(last - 1)) / 2.0L;
                    long double weight = 1.0L;
                    if (finite_indices.size() > 1) {
                        const long double pressure = parameters.FitnessRankPressure;
                        weight = (2.0L - pressure) / count + 2.0L * (count - average_rank - 1.0L) * (pressure - 1.0L) / (count * (count - 1.0L));
                    }
                    for (std::size_t rank = first; rank < last; ++rank) {
                        transformed[finite_indices[rank]] = static_cast<double>(std::max(epsilon, weight));
                    }
                    first = last;
                }
                break;
            }

            case SIGMA_FITNESS_SCALING: {
                long double scale = 0.0L;
                for (const std::size_t index : finite_indices) {
                    scale = std::max(scale, std::abs(static_cast<long double>(raw_fitness[index])));
                }
                if (scale <= 0.0L) scale = 1.0L;
                long double mean = 0.0L;
                long double sum_squared_deviation = 0.0L;
                std::size_t count = 0;
                for (const std::size_t index : finite_indices) {
                    ++count;
                    const long double value = static_cast<long double>(raw_fitness[index]) / scale;
                    const long double delta = value - mean;
                    mean += delta / static_cast<long double>(count);
                    sum_squared_deviation += delta * (value - mean);
                }
                const long double deviation = count > 1 ? std::sqrt(sum_squared_deviation / static_cast<long double>(count)) : 0.0L;
                for (const std::size_t index : finite_indices) {
                    const long double weight = deviation > 0.0L ? 1.0L + (static_cast<long double>(raw_fitness[index]) / scale - mean) /
                                                                             (static_cast<long double>(parameters.FitnessSigmaScale) * deviation)
                                                                : 1.0L;
                    transformed[index] = static_cast<double>(std::max(epsilon, weight));
                }
                break;
            }

            case BOLTZMANN_FITNESS_SCALING: {
                long double maximum = static_cast<long double>(raw_fitness[finite_indices.front()]);
                for (const std::size_t index : finite_indices) {
                    maximum = std::max(maximum, static_cast<long double>(raw_fitness[index]));
                }
                for (const std::size_t index : finite_indices) {
                    const long double exponent =
                        (static_cast<long double>(raw_fitness[index]) - maximum) / static_cast<long double>(parameters.FitnessBoltzmannTemperature);
                    transformed[index] = static_cast<double>(std::max(epsilon, std::exp(exponent)));
                }
                break;
            }
        }

        double maximum = 0.0;
        for (const std::size_t index : finite_indices) {
            const double value = transformed[index];
            if (std::isfinite(value)) maximum = std::max(maximum, value);
        }
        if (maximum <= 0.0) {
            for (const std::size_t index : finite_indices) transformed[index] = 1.0;
        }
        return transformed;
    }

    // Picks a representative individual index for the next generation's
    // speciation (FIRST preserves the sorted-leader behavior; MEDOID is O(n^2)).
    inline std::size_t ChooseRepresentativeIndex(Species &species, Parameters &parameters, RNG &rng) {
        const std::size_t count = species.m_Individuals.size();
        if (count == 0) throw std::runtime_error("Cannot choose a representative from an empty species");
        switch (parameters.SpeciesRepresentativeSelection) {
            case FIRST_REPRESENTATIVE:
            case BEST_REPRESENTATIVE:
                // Epoch sorts every species best-first before reaching this point.
                return 0;

            case RANDOM_REPRESENTATIVE:
                return static_cast<std::size_t>(rng.RandInt(0, static_cast<int>(count) - 1));

            case MEDOID_REPRESENTATIVE:
                break;
        }

        std::vector<std::size_t> candidates(count);
        std::iota(candidates.begin(), candidates.end(), std::size_t{0});
        const std::size_t limit =
            parameters.RepresentativeSelectionCandidates == 0 ? count : std::min(count, static_cast<std::size_t>(parameters.RepresentativeSelectionCandidates));
        // Partial Fisher-Yates sampling bounds expensive medoid searches without
        // biasing toward the sorted leaders.
        for (std::size_t i = 0; i < limit; ++i) {
            const std::size_t selected = static_cast<std::size_t>(rng.RandInt(static_cast<int>(i), static_cast<int>(count) - 1));
            std::swap(candidates[i], candidates[selected]);
        }
        candidates.resize(limit);

        std::size_t best_index = candidates.front();
        double best_distance = std::numeric_limits<double>::max();
        for (const std::size_t candidate : candidates) {
            double distance = 0.0;
            for (std::size_t other = 0; other < count; ++other) {
                if (candidate == other) continue;
                distance += species.m_Individuals[candidate].CompatibilityDistance(species.m_Individuals[other], parameters);
            }
            if (distance < best_distance) {
                best_distance = distance;
                best_index = candidate;
            }
        }
        return best_index;
    }

    // The constructor
    Population::Population(const Genome &a_Seed, const Parameters &a_Parameters, bool a_RandomizeWeights, double a_RandomizationRange, int a_RNG_seed) {
        std::string parameter_error;
        if (!a_Parameters.Validate(&parameter_error)) throw std::invalid_argument("Invalid evolution parameters: " + parameter_error);
        if (a_RandomizationRange < 0.0 || !std::isfinite(a_RandomizationRange))
            throw std::invalid_argument("Randomization range must be finite and non-negative");

        m_RNG.Seed(a_RNG_seed);
        m_BestFitnessEver = std::numeric_limits<double>::lowest();
        m_Parameters = a_Parameters;

        m_Generation = 0;
        m_NumEvaluations = 0;
        m_NextGenomeID = m_Parameters.PopulationSize;
        m_NextSpeciesID = 1;
        m_GensSinceBestFitnessLastChanged = 0;
        m_GensSinceMPCLastChanged = 0;

        // Spawn the population
        for (unsigned int i = 0; i < m_Parameters.PopulationSize; i++) {
            Genome t_clone = a_Seed;
            t_clone.SetID(i);
            m_Genomes.emplace_back(t_clone);
        }

        // Now now initialize each genome's weights
        for (unsigned int i = 0; i < m_Genomes.size(); i++) {
            if (a_RandomizeWeights) {
                bool is_invalid = true;
                const int max_attempts = std::max(1, a_Parameters.ConstraintTrials);
                for (int attempt = 0; attempt < max_attempts && is_invalid; ++attempt) {
                    Parameters initialization_parameters = a_Parameters;
                    initialization_parameters.MinWeight = std::max(a_Parameters.MinWeight, -a_RandomizationRange);
                    initialization_parameters.MaxWeight = std::min(a_Parameters.MaxWeight, a_RandomizationRange);
                    if (initialization_parameters.MinWeight > initialization_parameters.MaxWeight) {
                        throw std::invalid_argument("Randomization range does not overlap the configured weight range");
                    }
                    m_Genomes[i].Randomize_LinkWeights(initialization_parameters, m_RNG);
                    // randomize the traits as well
                    m_Genomes[i].Randomize_Traits(a_Parameters, m_RNG);
                    m_Genomes[i].Randomize_SpikingParameters(a_Parameters, m_RNG);
                    // and mutate nodes one initial time
                    m_Genomes[i].Mutate_NeuronActivations_A(a_Parameters, m_RNG);
                    m_Genomes[i].Mutate_NeuronActivations_B(a_Parameters, m_RNG);
                    m_Genomes[i].Mutate_NeuronActivation_Type(a_Parameters, m_RNG);
                    m_Genomes[i].Mutate_NeuronTimeConstants(a_Parameters, m_RNG);
                    m_Genomes[i].Mutate_NeuronBiases(a_Parameters, m_RNG);

                    // check in the population if there is a clone of that genome
                    is_invalid = false;
                    if (!m_Parameters.AllowClones) {
                        for (unsigned int j = 0; j < m_Genomes.size(); j++) {
                            if (i != j)  // don't compare the same genome
                            {
                                if (m_Genomes[i].IsIdenticalTo(m_Genomes[j]) ||
                                    (m_Parameters.MinDeltaCompatEqualGenomes > 0.0 && m_Genomes[i].CompatibilityDistance(m_Genomes[j], m_Parameters) <
                                                                                          m_Parameters.MinDeltaCompatEqualGenomes)  // equal genomes?
                                ) {
                                    is_invalid = true;
                                    break;
                                }
                            }
                        }
                    }

                    // Also don't let any genome to fail the constraints
                    if (!is_invalid)  // doesn't make sense to do the test if already failed
                    {
                        if (m_Genomes[i].FailsConstraints(a_Parameters)) {
                            is_invalid = true;
                        }
                    }
                }
                if (is_invalid) {
                    throw std::runtime_error("Unable to initialize a valid, non-cloning genome within ConstraintTrials");
                }
            }

            // m_Genomes[i].CalculateDepth();
        }
        // Speciate
        Speciate();

        // set these phased search variables now since used in MutateGenome
        if (m_Parameters.PhasedSearching) {
            m_SearchMode = COMPLEXIFYING;
        } else {
            m_SearchMode = BLENDED;
        }

        // Initialize the innovation database
        m_InnovationDatabase.Init(a_Seed);

        m_BestGenome = m_Species[0].m_Individuals[0];  // GetLeader();

        m_ID = 0;

        // Sort();

        // Set up the rest of the phased search variables
        CalculateMPC();
        m_BaseMPC = m_CurrentMPC;
        m_OldMPC = m_BaseMPC;

        // Reset IDs to be sure
        int cid = 0;
        for (int i = 0; i < m_Species.size(); i++) {
            for (int j = 0; j < m_Species[i].m_Individuals.size(); j++) {
                m_Species[i].m_Individuals[j].SetID(cid);
                cid++;
            }
        }

        m_InnovationDatabase.m_Innovations.reserve(50000);
    }

    Population::Population(const std::string a_sFileName) {
        auto a_FileName = a_sFileName.c_str();
        m_BestFitnessEver = std::numeric_limits<double>::lowest();

        m_Generation = 0;
        m_NumEvaluations = 0;
        m_NextSpeciesID = 1;
        m_ID = 0;
        m_GensSinceBestFitnessLastChanged = 0;
        m_GensSinceMPCLastChanged = 0;

        std::ifstream t_DataFile(a_FileName);
        if (!t_DataFile.is_open()) throw std::exception();
        std::string t_str;

        // Load the parameters
        m_Parameters.Load(t_DataFile);

        // Load the innovation database
        m_InnovationDatabase.Init(t_DataFile);

        // Load all genomes
        for (unsigned int i = 0; i < m_Parameters.PopulationSize; i++) {
            Genome t_genome(t_DataFile);
            m_Genomes.emplace_back(t_genome);
        }
        t_DataFile.close();

        m_NextGenomeID = 0;
        for (unsigned int i = 0; i < m_Genomes.size(); i++) {
            if (m_Genomes[i].GetID() > m_NextGenomeID) {
                m_NextGenomeID = m_Genomes[i].GetID();
            }
        }
        m_NextGenomeID++;

        // Initialize
        Speciate();
        m_BestGenome = m_Species[0].GetLeader();

        // Sort();

        // Set up the phased search variables
        CalculateMPC();
        m_BaseMPC = m_CurrentMPC;
        m_OldMPC = m_BaseMPC;
        if (m_Parameters.PhasedSearching) {
            m_SearchMode = COMPLEXIFYING;
        } else {
            m_SearchMode = BLENDED;
        }
    }

    // Save a whole population to a file
    void Population::Save(const char *a_FileName) {
        FILE *t_file = fopen(a_FileName, "w");

        // Save the parameters
        m_Parameters.Save(t_file);

        // Save the innovation database
        m_InnovationDatabase.Save(t_file);

        // Save each genome
        for (unsigned i = 0; i < m_Species.size(); i++) {
            for (unsigned j = 0; j < m_Species[i].m_Individuals.size(); j++) {
                m_Species[i].m_Individuals[j].Save(t_file);
            }
        }

        // bye
        fclose(t_file);
    }

    // Calculates the current mean population complexity
    void Population::CalculateMPC() {
        m_CurrentMPC = 0;

        for (unsigned int i = 0; i < m_Genomes.size(); i++) {
            m_CurrentMPC += AccessGenomeByIndex(i).NumLinks();
        }

        m_CurrentMPC /= m_Genomes.size();
    }

    // Separates the population into species also adjusts the compatibility treshold if this feature is enabled
    void Population::Speciate() {
        // iterate through the genome list and speciate at least 1 genome must be present
        ASSERT(m_Genomes.size() > 0);

        // first clear out the species
        m_Species.clear();

        // With speciation disabled the population is a single species.
        if (!m_Parameters.Speciation) {
            if (!m_Genomes.empty()) {
                m_Species.emplace_back(m_Genomes.front(), m_Parameters, m_NextSpeciesID++);
                for (std::size_t i = 1; i < m_Genomes.size(); ++i) {
                    m_Species.front().AddIndividual(m_Genomes[i]);
                }
            }
            return;
        }

        bool t_added = false;

        // NOTE: we are comparing the new generation's genomes to the representatives from species creation time!
        //
        for (unsigned int i = 0; i < m_Genomes.size(); i++) {
            t_added = false;

            // iterate through each species and check if compatible. If compatible, then add to the species. if not compatible, create a new species.
            for (unsigned int j = 0; j < m_Species.size(); j++) {
                if (m_Species[j].NumIndividuals() > 0) {
                    if (m_Genomes[i].IsCompatibleWith(m_Species[j].GetRepresentative(), m_Parameters)) {
                        // Compatible, add to species
                        m_Species[j].AddIndividual(m_Genomes[i]);
                        t_added = true;

                        break;
                    }
                }
            }

            if (!t_added) {
                // didn't find compatible species, create new species
                m_Species.push_back(Species(m_Genomes[i], m_Parameters, m_NextSpeciesID));
                m_NextSpeciesID++;
            }
        }

        // Remove all empty species (cleanup routine for every case..)
        ClearEmptySpecies();
    }

    // Adjust the fitness of all species using population-wide transformed
    // values, then normalize adjusted fitness to a maximum of 1.0 so extreme
    // age multipliers cannot overflow offspring allocation.
    void Population::AdjustFitness() {
        ASSERT(m_Genomes.size() > 0);
        ASSERT(m_Species.size() > 0);

        std::vector<double> raw_fitness;
        raw_fitness.reserve(NumGenomes());
        for (const auto &species : m_Species) {
            for (const auto &genome : species.m_Individuals) raw_fitness.push_back(genome.GetFitness());
        }
        const std::vector<double> transformed = TransformFitnessValues(raw_fitness, m_Parameters);

        std::size_t offset = 0;
        for (auto &species : m_Species) {
            const std::size_t species_size = species.m_Individuals.size();
            species.AdjustFitness(m_Parameters, std::vector<double>(transformed.begin() + static_cast<std::ptrdiff_t>(offset),
                                                                    transformed.begin() + static_cast<std::ptrdiff_t>(offset + species_size)));
            offset += species_size;
        }

        double maximum_adjusted = 0.0;
        for (const auto &species : m_Species) {
            for (const auto &genome : species.m_Individuals) {
                if (std::isfinite(genome.GetAdjFitness())) {
                    maximum_adjusted = std::max(maximum_adjusted, genome.GetAdjFitness());
                }
            }
        }
        if (maximum_adjusted > 0.0 && std::isfinite(maximum_adjusted)) {
            for (auto &species : m_Species) {
                for (auto &genome : species.m_Individuals) {
                    genome.SetAdjFitness(genome.GetAdjFitness() / maximum_adjusted);
                }
            }
        }
    }

    // Calculates how many offspring each genome should have
    void Population::CountOffspring() {
        ASSERT(m_Genomes.size() > 0);
        ASSERT(m_Genomes.size() == m_Parameters.PopulationSize);

        const unsigned int population_size = NumGenomes();
        if (population_size == 0) {
            throw std::runtime_error("Cannot count offspring for an empty population");
        }
        if (population_size != m_Parameters.PopulationSize) {
            throw std::runtime_error("Population size does not match Parameters::PopulationSize");
        }

        double t_total_adjusted_fitness = 0.0;
        double t_average_adjusted_fitness = 0.0;
        Genome t_t;

        // get the total adjusted fitness for all individuals
        for (unsigned int i = 0; i < m_Species.size(); i++) {
            for (unsigned int j = 0; j < m_Species[i].m_Individuals.size(); j++) {
                t_total_adjusted_fitness += m_Species[i].m_Individuals[j].GetAdjFitness();

                // std::cout << m_Species[i].m_Individuals[j].GetFitness() << " " << m_Species[i].m_Individuals[j].GetAdjFitness() << "\n";
            }
        }

        // must be above 0
        ASSERT(t_total_adjusted_fitness > 0.0);

        t_average_adjusted_fitness = t_total_adjusted_fitness / static_cast<double>(population_size);
        if (!std::isfinite(t_average_adjusted_fitness) || t_average_adjusted_fitness <= 0.0) {
            t_average_adjusted_fitness = 1.0;
        }

        // std::cout << t_average_adjusted_fitness << "\n";

        // Calculate how much offspring each individual should have
        for (unsigned int i = 0; i < m_Species.size(); i++) {
            for (unsigned int j = 0; j < m_Species[i].m_Individuals.size(); j++) {
                m_Species[i].m_Individuals[j].SetOffspringAmount(m_Species[i].m_Individuals[j].GetAdjFitness() / t_average_adjusted_fitness);
            }
        }

        // Now count how many offpring each species should have
        for (unsigned int i = 0; i < m_Species.size(); i++) {
            m_Species[i].CountOffspring();
        }
    }

    // This little tool function helps ordering the genomes by fitness
    bool species_greater(Species &ls, Species &rs) { return ((ls.GetBestFitness()) > (rs.GetBestFitness())); }
    void Population::Sort() {
        ASSERT(m_Species.size() > 0);

        // Step through each species and sort its members by fitness
        for (unsigned int i = 0; i < m_Species.size(); i++) {
            ASSERT(m_Species[i].NumIndividuals() > 0);
            m_Species[i].SortIndividuals();
        }

        // Now sort the species by fitness (best first)
        std::sort(m_Species.begin(), m_Species.end(), species_greater);

        // for(int i=0;i<m_Species.size();i++)
        // std::cout << m_Species[i].GetBestFitness() << "\n";
        // std::cout << "\n\n";
    }

    // Updates the species
    void Population::UpdateSpecies() {
        // search for the current best species ID if not at generation #0
        // (the previous best species may have gone extinct under exact
        // offspring quotas; in that case there is simply no age to preserve)
        int t_oldbestid = -1, t_newbestid = -1;
        int t_oldbestidx = -1;
        if (m_Generation > 0) {
            for (unsigned int i = 0; i < m_Species.size(); i++) {
                if (m_Species[i].IsBestSpecies()) {
                    t_oldbestid = m_Species[i].ID();
                    t_oldbestidx = i;
                }
            }
        }

        for (unsigned int i = 0; i < m_Species.size(); i++) {
            m_Species[i].SetBestSpecies(false);
        }

        bool t_marked = false;  // new best species marked?

        for (unsigned int i = 0; i < m_Species.size(); i++) {
            // Reset the species and update its age
            m_Species[i].IncreaseAgeGens();
            m_Species[i].IncreaseGensNoImprovement();
            m_Species[i].SetOffspringRqd(0);

            // Mark the best species so it is guaranteed to survive
            // Only one species will be marked - in case several species
            // have equally best fitness
            if ((m_Species[i].GetBestFitness() >= m_BestFitnessEver) && (!t_marked)) {
                m_Species[i].SetBestSpecies(true);
                t_marked = true;
                t_newbestid = m_Species[i].ID();
            }
        }

        // This prevents the previous best species from sudden death If the best species happened to be another one, reset the old species age so it still will
        // have a chance of survival and improvement if it grows old and stagnates again, it is no longer the best one so it will die off anyway.
        if ((t_oldbestid != t_newbestid) && (t_oldbestid != -1)) {
            m_Species[t_oldbestidx].ResetAgeGens();
        }
    }

    // the epoch method - the heart of the GA
    void Population::Epoch() {
        std::string parameter_error;
        if (!m_Parameters.Validate(&parameter_error)) throw std::invalid_argument("Invalid evolution parameters: " + parameter_error);
        if (m_Species.empty() || NumGenomes() == 0) {
            throw std::runtime_error("Cannot run Epoch on an empty population");
        }
        // Historical Epoch() treated every member as evaluated. Strict modes are
        // additive and let experiments fail early instead of silently selecting
        // missing or non-finite measurements.
        for (const auto &species : m_Species) {
            for (const auto &genome : species.m_Individuals) {
                if (m_Parameters.RequireEvaluatedGenomes && !genome.IsEvaluated()) {
                    throw std::runtime_error("Epoch requires every genome to be evaluated");
                }
                if (m_Parameters.RejectNonFiniteFitness && !std::isfinite(genome.GetFitness())) {
                    throw std::runtime_error("Epoch requires every fitness value to be finite");
                }
            }
        }
        // So, all genomes are evaluated..
        for (unsigned int i = 0; i < m_Species.size(); i++) {
            for (unsigned int j = 0; j < m_Species[i].m_Individuals.size(); j++) {
                m_Species[i].m_Individuals[j].SetEvaluated();
            }
        }

        // Sort each species's members by fitness and the species by fitness
        // (best-first order is required by truncation selection and elitism)
        Sort();

        // Update species stagnation info & stuff
        UpdateSpecies();

        ///////////////////
        // Preparation
        ///////////////////

        // Adjust the species's fitness
        AdjustFitness();

        // Count the offspring of each individual and species
        CountOffspring();

        // Incrementing the global stagnation counter, we can check later for global stagnation
        m_GensSinceBestFitnessLastChanged++;
        // Find and save the best genome and fitness
        for (unsigned int i = 0; i < m_Species.size(); i++) {
            // Update best genome info
            m_Species[i].m_BestGenome = m_Species[i].GetLeader();

            for (unsigned int j = 0; j < m_Species[i].m_Individuals.size(); j++) {
                // Make sure all are evaluated as we don't run in realtime
                m_Species[i].m_Individuals[j].SetEvaluated();

                const double t_Fitness = m_Species[i].m_Individuals[j].GetFitness();
                if (!std::isfinite(t_Fitness)) continue;
                if (m_BestFitnessEver < t_Fitness) {
                    // Reset the stagnation counter only if the fitness jump is greater or equal to the delta.
                    if (fabs(t_Fitness - m_BestFitnessEver) >= m_Parameters.StagnationDelta) {
                        m_GensSinceBestFitnessLastChanged = 0;
                    }

                    m_BestFitnessEver = t_Fitness;
                    m_BestGenomeEver = m_Species[i].m_Individuals[j];
                }
            }
        }

        // Find and save the current best genome
        double t_bestf = std::numeric_limits<double>::lowest();
        for (unsigned int i = 0; i < m_Species.size(); i++) {
            for (unsigned int j = 0; j < m_Species[i].m_Individuals.size(); j++) {
                const double fitness = m_Species[i].m_Individuals[j].GetFitness();
                if (std::isfinite(fitness) && fitness > t_bestf) {
                    t_bestf = fitness;
                    m_BestGenome = m_Species[i].m_Individuals[j];
                }
            }
        }

        // adjust the compatibility threshold
        if (m_Parameters.DynamicCompatibility == true) {
            if (m_Parameters.CompatTreshChangeInterval_Generations > 0 && (m_Generation % m_Parameters.CompatTreshChangeInterval_Generations) == 0) {
                if (m_Parameters.CompatibilityThresholdControl == PROPORTIONAL_COMPATIBILITY_THRESHOLD) {
                    const unsigned int target = m_Parameters.TargetSpecies > 0
                                                    ? m_Parameters.TargetSpecies
                                                    : m_Parameters.MinSpecies + (m_Parameters.MaxSpecies - m_Parameters.MinSpecies) / 2U;
                    const double normalized_error = (static_cast<double>(m_Species.size()) - static_cast<double>(target)) / static_cast<double>(target);
                    m_Parameters.CompatTreshold *= std::exp(m_Parameters.CompatibilityThresholdGain * normalized_error);
                } else {
                    if (m_Species.size() > m_Parameters.MaxSpecies) {
                        m_Parameters.CompatTreshold += m_Parameters.CompatTresholdModifier;
                    } else if (m_Species.size() < m_Parameters.MinSpecies) {
                        m_Parameters.CompatTreshold -= m_Parameters.CompatTresholdModifier;
                    }
                }
            }

            m_Parameters.CompatTreshold = std::clamp(m_Parameters.CompatTreshold, m_Parameters.MinCompatTreshold, m_Parameters.MaxCompatTreshold);
        }

        // A special case for global stagnation.
        // Delta coding - if there is a global stagnation
        // for dropoff age + 10 generations, focus the search on the top 2 species,
        // in case there are more than 2, of course
        if (m_Parameters.DeltaCoding) {
            if (m_GensSinceBestFitnessLastChanged > (m_Parameters.SpeciesMaxStagnation + 10)) {
                // make the top 2 reproduce by 50% individuals
                // and the rest - no offspring
                if (m_Species.size() > 2) {
                    // The first two will reproduce
                    m_Species[0].SetOffspringRqd(m_Parameters.PopulationSize / 2);
                    m_Species[1].SetOffspringRqd(m_Parameters.PopulationSize / 2);

                    // The rest will not
                    for (unsigned int i = 2; i < m_Species.size(); i++) {
                        m_Species[i].SetOffspringRqd(0);
                    }

                    // Now reset the stagnation counter and species age
                    m_Species[0].ResetAgeGens();
                    m_Species[1].ResetAgeGens();
                    m_GensSinceBestFitnessLastChanged = 0;
                }
            }
        }

        //////////////////////////////////
        // Phased searching core logic
        //////////////////////////////////
        // Update the current MPC
        CalculateMPC();
        if (m_Parameters.PhasedSearching) {
            // Keep track of complexity when in simplifying phase
            if (m_SearchMode == SIMPLIFYING) {
                // The MPC has lowered?
                if (m_CurrentMPC < m_OldMPC) {
                    // reset that
                    m_GensSinceMPCLastChanged = 0;
                    m_OldMPC = m_CurrentMPC;
                } else {
                    m_GensSinceMPCLastChanged++;
                }
            }

            // At complexifying phase?
            if (m_SearchMode == COMPLEXIFYING) {
                // Need to begin simplification?
                if (m_CurrentMPC > (m_BaseMPC + m_Parameters.SimplifyingPhaseMPCTreshold)) {
                    // Do this only if the whole population is stagnating
                    if (m_GensSinceBestFitnessLastChanged > m_Parameters.SimplifyingPhaseStagnationTreshold) {
                        // Change the current search mode
                        m_SearchMode = SIMPLIFYING;

                        // Reset variables for simplifying mode
                        m_GensSinceMPCLastChanged = 0;
                        m_OldMPC = std::numeric_limits<double>::max();  // Really big one

                        // reset the age of species
                        for (unsigned int i = 0; i < m_Species.size(); i++) {
                            m_Species[i].ResetAgeGens();
                        }
                    }
                }
            } else if (m_SearchMode == SIMPLIFYING)
            // At simplifying phase?
            {
                // The MPC reached its floor level?
                if (m_GensSinceMPCLastChanged > m_Parameters.ComplexityFloorGenerations) {
                    // Re-enter complexifying phase
                    m_SearchMode = COMPLEXIFYING;

                    // Set the base MPC with the current MPC
                    m_BaseMPC = m_CurrentMPC;

                    // reset the age of species
                    for (unsigned int i = 0; i < m_Species.size(); i++) {
                        m_Species[i].ResetAgeGens();
                    }
                }
            }
        }

        /////////////////////////////
        // Reproduction
        /////////////////////////////

        // Convert per-species fractional requirements into exact integer quotas.
        // Optional floors protect viable niches before the remaining capacity is
        // apportioned in proportion to adjusted fitness.
        {
            std::vector<double> quotas(m_Species.size(), 0.0);
            std::vector<double> requirements(m_Species.size(), 0.0);
            for (std::size_t i = 0; i < m_Species.size(); ++i) {
                const double requirement = m_Species[i].GetOffspringRqd();
                if (!std::isfinite(requirement) || requirement < 0.0) {
                    throw std::runtime_error("Species offspring requirements must be finite and non-negative");
                }
                requirements[i] = m_Species[i].IsWorstSpecies() ? 0.0 : requirement;
            }

            if (m_Parameters.MinSpeciesSize == 0 && m_Parameters.SpeciesElitism == 0) {
                quotas = requirements;
            } else {
                std::vector<unsigned int> floors(m_Species.size(), 0);
                std::vector<bool> included(m_Species.size(), false);
                unsigned int reserved = 0;
                const std::size_t protected_count = std::min(m_Species.size(), static_cast<std::size_t>(m_Parameters.SpeciesElitism));

                // Protected species are considered first because m_Species is sorted
                // best-first. Validation guarantees at least a one-member reserve can
                // fit; a larger requested floor may intentionally reduce the number
                // of retained niches.
                for (std::size_t i = 0; i < protected_count; ++i) {
                    const unsigned int desired = std::max(1U, m_Parameters.MinSpeciesSize);
                    if (reserved + desired > m_Parameters.PopulationSize) break;
                    floors[i] = desired;
                    included[i] = true;
                    reserved += desired;
                }
                for (std::size_t i = 0; i < m_Species.size(); ++i) {
                    if (included[i] || requirements[i] <= 0.0) continue;
                    if (m_Parameters.MinSpeciesSize > 0) {
                        if (reserved + m_Parameters.MinSpeciesSize > m_Parameters.PopulationSize) {
                            continue;
                        }
                        floors[i] = m_Parameters.MinSpeciesSize;
                        reserved += m_Parameters.MinSpeciesSize;
                    }
                    included[i] = true;
                }
                if (std::none_of(included.begin(), included.end(), [](bool value) { return value; })) included.front() = true;
                const unsigned int remaining = m_Parameters.PopulationSize - reserved;
                double total_weight = 0.0;
                for (std::size_t i = 0; i < requirements.size(); ++i) {
                    if (included[i]) total_weight += requirements[i];
                }
                if (total_weight <= 0.0) {
                    total_weight = static_cast<double>(std::count(included.begin(), included.end(), true));
                    for (std::size_t i = 0; i < quotas.size(); ++i) {
                        if (included[i]) quotas[i] = static_cast<double>(floors[i]) + static_cast<double>(remaining) / total_weight;
                    }
                } else {
                    for (std::size_t i = 0; i < quotas.size(); ++i) {
                        if (included[i]) {
                            quotas[i] = static_cast<double>(floors[i]) + static_cast<double>(remaining) * requirements[i] / total_weight;
                        }
                    }
                }
            }

            std::vector<unsigned int> offspring_counts(m_Species.size(), 0);
            std::vector<double> offspring_remainders(m_Species.size(), 0.0);
            unsigned int assigned_offspring = 0;
            for (std::size_t i = 0; i < quotas.size(); ++i) {
                const double integral = std::floor(quotas[i]);
                if (integral > static_cast<double>(std::numeric_limits<unsigned int>::max())) {
                    throw std::overflow_error("Species offspring requirement is too large");
                }
                offspring_counts[i] = static_cast<unsigned int>(integral);
                offspring_remainders[i] = quotas[i] - integral;
                assigned_offspring += offspring_counts[i];
            }

            if (assigned_offspring < m_Parameters.PopulationSize) {
                unsigned int remaining = m_Parameters.PopulationSize - assigned_offspring;
                if (m_Parameters.OffspringAllocation == STOCHASTIC_REMAINDER) {
                    while (remaining > 0) {
                        double total = std::accumulate(offspring_remainders.begin(), offspring_remainders.end(), 0.0);
                        if (total <= 0.0) break;
                        const std::size_t selected = static_cast<std::size_t>(m_RNG.Roulette(offspring_remainders));
                        ++offspring_counts[selected];
                        offspring_remainders[selected] = 0.0;
                        --remaining;
                    }
                }
                if (remaining > 0) {
                    std::vector<std::size_t> order;
                    order.reserve(m_Species.size());
                    for (std::size_t i = 0; i < m_Species.size(); ++i) {
                        if (quotas[i] > 0.0) order.push_back(i);
                    }
                    if (order.empty()) throw std::runtime_error("No species is eligible to receive offspring");
                    std::stable_sort(order.begin(), order.end(), [&offspring_remainders](std::size_t lhs, std::size_t rhs) {
                        return offspring_remainders[lhs] > offspring_remainders[rhs];
                    });
                    for (unsigned int i = 0; i < remaining; ++i) {
                        ++offspring_counts[order[static_cast<std::size_t>(i) % order.size()]];
                    }
                }
            } else if (assigned_offspring > m_Parameters.PopulationSize) {
                unsigned int excess = assigned_offspring - m_Parameters.PopulationSize;
                for (std::size_t i = offspring_counts.size(); i > 0 && excess > 0; --i) {
                    const std::size_t index = i - 1;
                    const unsigned int protected_floor = (m_Parameters.MinSpeciesSize > 0 && offspring_counts[index] > 0) ? m_Parameters.MinSpeciesSize : 0U;
                    const unsigned int available = offspring_counts[index] > protected_floor ? offspring_counts[index] - protected_floor : 0U;
                    const unsigned int reduction = std::min(excess, available);
                    offspring_counts[index] -= reduction;
                    excess -= reduction;
                }
                if (excess != 0) throw std::runtime_error("Unable to reconcile species offspring counts");
            }
            for (std::size_t i = 0; i < m_Species.size(); ++i) {
                m_Species[i].SetOffspringRqd(static_cast<double>(offspring_counts[i]));
            }
        }

        // Perform reproduction for each species
        m_TempSpecies.clear();
        m_TempSpecies = m_Species;
        const std::size_t existing_species_count = m_TempSpecies.size();
        for (unsigned int i = 0; i < m_TempSpecies.size(); i++) {
            const std::size_t representative = ChooseRepresentativeIndex(m_Species[i], m_Parameters, m_RNG);
            Genome representative_genome = m_Species[i].m_Individuals[representative];
            m_TempSpecies[i].Clear();
            m_TempSpecies[i].AddIndividual(representative_genome);
        }

        for (unsigned int i = 0; i < m_Species.size(); i++) {
            m_Species[i].Reproduce(*this, m_Parameters, m_RNG);
        }
        // Only the original species contain the representative placeholder.
        // Species created during reproduction contain real offspring at index 0.
        for (std::size_t i = 0; i < existing_species_count; ++i) {
            m_TempSpecies[i].RemoveIndividual(0);
        }
        m_Species = m_TempSpecies;

        // Remove all empty species (cleanup routine for every case..)
        ClearEmptySpecies();
        // Now reassign the representatives for each species
        /*for(unsigned int i=0; i<m_Species.size(); i++)
        {
            m_Species[i].SetRepresentative( m_Species[i].m_Individuals[0] );
        }*/

        unsigned int t_total_genomes = 0;
        for (unsigned int i = 0; i < m_Species.size(); i++) t_total_genomes += static_cast<unsigned int>(m_Species[i].m_Individuals.size());

        if (t_total_genomes != m_Parameters.PopulationSize) {
            throw std::runtime_error("Reproduction did not preserve the configured population size");
        }

        // Increase generation number
        m_Generation++;

        // At this point we may also empty our innovation database This is the place where we control whether we want to keep innovation numbers forever or not.
        if (!m_Parameters.InnovationsForever) {
            m_InnovationDatabase.Flush();
        }
    }

    Genome g_dummy;  // empty genome
    Genome &Population::AccessGenomeByIndex(int const a_idx) {
        // The genomes live in the species; m_Genomes is only the initial seed list
        // and goes stale after the first Epoch, so bounds-check against the
        // actual number of individuals.
        if (a_idx < 0) {
            throw std::out_of_range("Population genome index cannot be negative");
        }
        int t_counter = 0;

        for (unsigned int i = 0; i < m_Species.size(); i++) {
            for (unsigned int j = 0; j < m_Species[i].m_Individuals.size(); j++) {
                if (t_counter == a_idx)  // reached the index?
                {
                    return m_Species[i].m_Individuals[j];
                }

                t_counter++;
            }
        }

        throw std::out_of_range("Population genome index is out of range");
    }

    Genome &Population::AccessGenomeByID(int const a_id) {
        for (unsigned int i = 0; i < m_Species.size(); i++) {
            for (unsigned int j = 0; j < m_Species[i].m_Individuals.size(); j++) {
                if (m_Species[i].m_Individuals[j].GetID() == a_id)  // reached the ID?
                {
                    return m_Species[i].m_Individuals[j];
                }
            }
        }

        throw std::out_of_range("No genome with ID " + std::to_string(a_id) + " exists in the population");
    }

    /////////////////////////////////
    // Realtime code

    // Decides which species should have offspring. Returns the index of the species
    // Decides which species should have offspring. Returns the index of the species
    unsigned int Population::ChooseParentSpecies() {
        if (m_Species.empty()) throw std::runtime_error("Cannot choose a parent from an empty population");

        std::vector<std::size_t> eligible;
        std::vector<double> probs;
        double minimum = 0.0;
        for (std::size_t i = 0; i < m_Species.size(); ++i) {
            const auto &species = m_Species[i];
            if (species.NumEvaluated() == 0 || species.NumIndividuals() == 0) continue;
            const double fitness = std::isfinite(species.m_AverageFitness) ? species.m_AverageFitness : 0.0;
            minimum = eligible.empty() ? fitness : std::min(minimum, fitness);
            eligible.push_back(i);
            probs.push_back(fitness);
        }
        if (eligible.empty()) {
            throw std::runtime_error("No evaluated species is available for reproduction");
        }
        if (minimum < 0.0) {
            for (double &probability : probs) probability -= minimum;
        }
        if (std::none_of(probs.begin(), probs.end(), [](double probability) { return probability > 0.0; })) {
            std::fill(probs.begin(), probs.end(), 1.0);
        }

        return static_cast<unsigned int>(eligible[static_cast<std::size_t>(m_RNG.Roulette(probs))]);
    }

    void Population::ReassignSpecies(int a_genome_idx) {
        if (a_genome_idx < 0 || static_cast<unsigned int>(a_genome_idx) >= NumGenomes()) {
            throw std::out_of_range("Population genome index is out of range");
        }

        int counter = 0;
        std::size_t source_species = 0;
        std::size_t source_genome = 0;
        for (; source_species < m_Species.size(); ++source_species) {
            const int species_size = static_cast<int>(m_Species[source_species].m_Individuals.size());
            if (a_genome_idx < counter + species_size) {
                source_genome = static_cast<std::size_t>(a_genome_idx - counter);
                break;
            }
            counter += species_size;
        }
        Genome genome = m_Species[source_species].m_Individuals[source_genome];
        m_Species[source_species].RemoveIndividual(static_cast<unsigned int>(source_genome));

        bool found = false;
        for (auto &species : m_Species) {
            if (species.NumIndividuals() > 0 && genome.IsCompatibleWith(species.GetRepresentative(), m_Parameters)) {
                species.AddIndividual(genome);
                found = true;
                break;
            }
        }

        if (!found) {
            m_Species.emplace_back(genome, m_Parameters, GetNextSpeciesID());
            IncrementNextSpeciesID();
        }
        ClearEmptySpecies();
    }

    Genome *Population::Tick(Genome &a_deleted_genome) {
        // Make sure at least one individual is evaluated
        int ne = 0;
        for (int i = 0; i < m_Species.size(); i++) {
            ne += m_Species[i].NumEvaluated();
        }
        if (ne == 0) {
            throw std::runtime_error("Called Tick() on population with no evaluated individuals.\n");
        }

#ifdef VDEBUG
        std::cout << "tracking stuff\n";
#endif

        m_NumEvaluations++;

        // Find and save the best genome and fitness
        m_EvalsSinceBestFitnessLastChanged++;
        for (int i = 0; i < m_Species.size(); i++) {
            // m_Species[i].IncreaseEvalsNoImprovement();

            for (int j = 0; j < m_Species[i].m_Individuals.size(); j++) {
                // if (m_Species[i].m_Individuals[j].GetFitness() <= 0.0)
                //{
                //     m_Species[i].m_Individuals[j].SetFitness(0.00001);
                //}

                if (!m_Species[i].m_Individuals[j].IsEvaluated()) continue;
                double t_fitness = m_Species[i].m_Individuals[j].GetFitness();
                if (std::isnan(t_fitness) || std::isinf(t_fitness)) {
                    t_fitness = 0;
                }

                if (t_fitness > m_BestFitnessEver) {
                    // Reset the stagnation counter only if the fitness jump is greater or equal to the delta.
                    if (fabs(t_fitness - m_BestFitnessEver) >= m_Parameters.StagnationDelta) {
                        m_EvalsSinceBestFitnessLastChanged = 0;
                    }

                    m_BestFitnessEver = t_fitness;
                    m_BestGenomeEver = m_Species[i].m_Individuals[j];
                }
            }
        }

        double t_f = std::numeric_limits<double>::min();
        for (int i = 0; i < m_Species.size(); i++) {
            for (int j = 0; j < m_Species[i].m_Individuals.size(); j++) {
                if (m_Species[i].m_Individuals[j].GetFitness() > t_f) {
                    t_f = m_Species[i].m_Individuals[j].GetFitness();
                    m_BestGenome = m_Species[i].m_Individuals[j];
                }

                if (m_Species[i].m_Individuals[j].GetFitness() > m_Species[i].GetBestFitness()) {
                    m_Species[i].m_BestFitness = m_Species[i].m_Individuals[j].GetFitness();
                    m_Species[i].m_EvalsNoImprovement = 0;
                }
            }
        }

        // adjust the compatibility treshold
        bool t_changed = false;
        if (m_Parameters.DynamicCompatibility == true) {
            double t_oldcompat = m_Parameters.CompatTreshold;
            if ((m_NumEvaluations % m_Parameters.CompatTreshChangeInterval_Evaluations) == 0) {
                if (m_Species.size() > m_Parameters.MaxSpecies) {
                    m_Parameters.CompatTreshold += m_Parameters.CompatTresholdModifier;
                } else if (m_Species.size() < m_Parameters.MinSpecies) {
                    m_Parameters.CompatTreshold -= m_Parameters.CompatTresholdModifier;
                }

                if (m_Parameters.CompatTreshold < m_Parameters.MinCompatTreshold) m_Parameters.CompatTreshold = m_Parameters.MinCompatTreshold;

                if (m_Parameters.CompatTreshold != t_oldcompat) {
                    t_changed = true;
                }
            }
        }

        // Sort individuals within species by fitness
        // Sort();

        // If the compatibility treshold was changed, reassign all individuals by species
        if (t_changed) {
            /*int numgs=0;
            for(int i=0; i<m_Species.size(); i++)
            {
                numgs += m_Species[i].m_Individuals.size();
            }

        #ifdef VDEBUG
            std::cout << "reassigning species. numgs=" << numgs << "\n";
        #endif

            for(int i=0; i<numgs; i++)
            {
                ReassignSpecies(i);
            }

            // After reassigning, some empty species may be left, so delete them
            ClearEmptySpecies();*/

            m_Genomes.clear();
            for (unsigned int i = 0; i < m_Species.size(); i++) {
                for (unsigned int j = 0; j < m_Species[i].m_Individuals.size(); j++) {
                    m_Genomes.push_back(m_Species[i].m_Individuals[j]);
                }
            }

            Speciate();
        }

        // Faster reassign
        /*if (t_changed)
        {
            std::cout << "reassigning species\n";

            // Perform reproduction for each species
            m_TempSpecies.clear();
            m_TempSpecies = m_Species;
            for(int i=0; i<m_TempSpecies.size(); i++)
            {
                m_TempSpecies[i].Clear();
            }

            std::vector<Genome*> allgenomes;
            for(int i=0; i<m_Species.size();i++)
            {
                for(int j=0; j<m_Species[i].m_Individuals.size(); j++)
                {
                    allgenomes.push_back(&m_Species[i].m_Individuals[j]);
                }
            }

            for(int i=0; i<allgenomes.size(); i++)
            {
                // Add the baby to its proper species
                bool t_found = false;
                auto t_cur_species = m_TempSpecies.begin();
                Genome& baby = *(allgenomes[i]);

                // No species yet?
                if (t_cur_species == m_TempSpecies.end())
                {
                    // create the first species and place the baby there
                    m_TempSpecies.push_back( Species(baby, m_Parameters, GetNextSpeciesID()) ); // clone the pop's parameters when creating species
                    IncrementNextSpeciesID();
                }
                else
                {
                    // try to find a compatible species
                    Genome& t_to_compare = t_cur_species->GetRepresentative(); // was GetRepresentative()

                    t_found = false;
                    while((t_cur_species != m_TempSpecies.end()) && (!t_found))
                    {
                        if (baby.IsCompatibleWith( t_to_compare, m_Parameters ))
                        {
                            // found a compatible species
                            t_cur_species->AddIndividual(baby);
                            t_found = true; // the search is over
                        }
                        else
                        {
                            // keep searching for a matching species
                            t_cur_species++;
                            if (t_cur_species != m_TempSpecies.end())
                            {
                                t_to_compare = t_cur_species->GetRepresentative(); // was GetRepresentative()
                            }
                        }
                    }

                    // if couldn't find a match, make a new species
                    if (!t_found)
                    {
                        m_TempSpecies.push_back( Species(baby, m_Parameters, GetNextSpeciesID()) ); // clone the pop's parameters when creating species
                        IncrementNextSpeciesID();
                    }
                }
            }

            m_Species = m_TempSpecies;

            // After reassigning, some empty species may be left, so delete them
            ClearEmptySpecies();
        }*/

#ifdef VDEBUG
        SameGenomeIDCheck();
#endif

#ifdef VDEBUG
        std::cout << "remove worst\n";
#endif
        // Remove the worst individual
        a_deleted_genome = RemoveWorstIndividual();

#ifdef VDEBUG
        std::cout << "calc avg fitness\n";
#endif
        // Recalculate all averages for each species
        // If the average species fitness of a species is 0,
        // then there are no evaluated individuals in it.
        for (unsigned int i = 0; i < m_Species.size(); i++) {
            m_Species[i].CalculateAverageFitness();
        }

#ifdef VDEBUG
        std::cout << "choose parents\n";
#endif
        // Now spawn the new offspring
        unsigned int t_parent_species_index = ChooseParentSpecies();

        Genome t_baby = m_Species[t_parent_species_index].ReproduceOne(*this, m_Parameters,  // m_Species[t_parent_species_index].m_Parameters,
                                                                       m_RNG);
        ASSERT(t_baby.NumInputs() > 0);
        ASSERT(t_baby.NumOutputs() > 0);
        Genome *t_to_return = NULL;

#ifdef VDEBUG
        std::cout << "placing baby in species\n";
#endif

        // Add the baby to its proper species
        bool t_found = false;
        auto t_cur_species = m_Species.begin();

        // No species yet?
        if (t_cur_species == m_Species.end()) {
            // create the first species and place the baby there
            m_Species.push_back(Species(t_baby, m_Parameters, GetNextSpeciesID()));  // clone the pop's parameters when creating species
            // the last one
            t_to_return = &(m_Species[m_Species.size() - 1].m_Individuals[m_Species[m_Species.size() - 1].m_Individuals.size() - 1]);
            IncrementNextSpeciesID();

#ifdef VDEBUG
            std::cout << "made new species\n";
#endif
        } else {
            // try to find a compatible species
            Genome t_to_compare = t_cur_species->GetRepresentative();

            t_found = false;
            while ((t_cur_species != m_Species.end()) && (!t_found)) {
                if (t_baby.IsCompatibleWith(t_to_compare, m_Parameters)) {
                    // found a compatible species
                    t_cur_species->AddIndividual(t_baby);
                    t_to_return = &(t_cur_species->m_Individuals[t_cur_species->m_Individuals.size() - 1]);
                    t_found = true;  // the search is over

                    // increase the evals counter for the new species
                    t_cur_species->IncreaseEvalsNoImprovement();

#ifdef VDEBUG
                    std::cout << "found compatible species\n";
#endif
                } else {
                    // keep searching for a matching species
                    /*t_cur_species++;
                    while((t_cur_species->NumIndividuals() == 0) && (t_cur_species != m_Species.end()))
                        t_cur_species++;

                    if (t_cur_species != m_Species.end())
                    {
                        t_to_compare = t_cur_species->GetRepresentative(); // was GetRepresentative()
                    }*/

                    while (1) {
                        t_cur_species++;
                        if (t_cur_species == m_Species.end()) {
                            break;
                        }
                        if (t_cur_species->NumIndividuals() > 0) {
                            t_to_compare = t_cur_species->GetRepresentative();
                            break;
                        }
                        /*else
                        {
                            t_cur_species++;
                        }*/
                    };
                }
            }

            // if couldn't find a match, make a new species
            if (!t_found) {
                m_Species.push_back(Species(t_baby, m_Parameters, GetNextSpeciesID()));  // clone the pop's parameters when creating species
                // the last one
                t_to_return = &(m_Species[m_Species.size() - 1].m_Individuals[m_Species[m_Species.size() - 1].m_Individuals.size() - 1]);
                IncrementNextSpeciesID();

#ifdef VDEBUG
                std::cout << "made new species\n";
#endif
            }
        }

#ifdef VDEBUG
        std::cout << "\n";
#endif

        ASSERT(t_to_return != NULL);

        return t_to_return;
    }

    void Population::ClearEmptySpecies() {
        m_Species.erase(std::remove_if(m_Species.begin(), m_Species.end(), [](const Species &species) { return species.NumIndividuals() == 0; }),
                        m_Species.end());
    }

    Genome Population::RemoveWorstIndividual() {
        unsigned int t_worst_idx = 0;          // within the species
        unsigned int t_worst_species_idx = 0;  // within the population
        double t_worst_fitness = std::numeric_limits<double>::max();
        int numev = 0;

        Genome t_genome;

        bool found = false;

        // Shift fitness into the non-negative domain (as in fitness sharing)
        // so negative-fitness individuals compare correctly.
        double minimum_fitness = 0.0;
        bool have_finite_fitness = false;
        for (unsigned int i = 0; i < m_Species.size(); i++) {
            for (unsigned int j = 0; j < m_Species[i].m_Individuals.size(); j++) {
                if (m_Species[i].m_Individuals[j].IsEvaluated() && std::isfinite(m_Species[i].m_Individuals[j].GetFitness())) {
                    const double fitness = m_Species[i].m_Individuals[j].GetFitness();
                    minimum_fitness = have_finite_fitness ? std::min(minimum_fitness, fitness) : fitness;
                    have_finite_fitness = true;
                }
            }
        }
        const double fitness_offset = have_finite_fitness && minimum_fitness <= 0.0 ? -minimum_fitness + 1.0e-7 : 0.0;

        // Find and kill the individual with the worst fitness-shared score.
        for (unsigned int i = 0; i < m_Species.size(); i++) {
            if (m_Species[i].m_Individuals.size() > 0) {
                double adjinv = 1.0 / static_cast<double>(m_Species[i].m_Individuals.size());
                for (unsigned int j = 0; j < m_Species[i].m_Individuals.size(); j++) {
                    // only evaluated individuals can be removed
                    if (m_Species[i].m_Individuals[j].IsEvaluated()) {
                        numev++;
                        const double fitness = std::isfinite(m_Species[i].m_Individuals[j].GetFitness()) ? m_Species[i].m_Individuals[j].GetFitness() : 0.0;
                        const double t_adjusted_fitness = (fitness + fitness_offset) * adjinv;

                        if (t_adjusted_fitness < t_worst_fitness) {
                            t_worst_fitness = t_adjusted_fitness;
                            t_worst_idx = j;
                            t_worst_species_idx = i;
                            found = true;
                        }
                    }
                }
            }
        }

        if (found) {
            t_genome = m_Species[t_worst_species_idx].m_Individuals[t_worst_idx];

            // make sure this isn't the only evaluated individual
            if (numev <= 1) {
                return t_genome;
            }

            // The individual is now removed
            m_Species[t_worst_species_idx].RemoveIndividual(t_worst_idx);

            // If the species becomes empty, remove the species as well
            if (m_Species[t_worst_species_idx].m_Individuals.size() == 0) {
                m_Species.erase(m_Species.begin() + t_worst_species_idx);
            }
        } else {
            // set ID of -1 to indicate nothing was removed
            t_genome.SetID(-1);
#ifdef VDEBUG
            std::cout << "RemoveWorst did not remove anything.\n";
#endif
        }

        return t_genome;
    }

    //////////////////////////////////////////
    // Novelty Search Code
    //////////////////////////////////////////

    // Call this function to allocate memory for your custom
    // behaviors. This initializes everything.
    // Warning! All derived classes MUST NOT have any member variables! Change the algorithms only!
    void Population::InitPhenotypeBehaviorData(std::vector<PhenotypeBehavior> *a_population, std::vector<PhenotypeBehavior> *a_archive) {
        // Now make each genome point to its behavior
        a_population->resize(NumGenomes());
        m_BehaviorArchive = a_archive;
        m_BehaviorArchive->clear();

        ASSERT(a_population->size() == NumGenomes());
        int counter = 0;
        for (unsigned int i = 0; i < m_Species.size(); i++) {
            for (unsigned int j = 0; j < m_Species[i].m_Individuals.size(); j++, counter++) {
                m_Species[i].m_Individuals[j].m_PhenotypeBehavior = &((*a_population)[counter]);
                m_Species[i].m_Individuals[j].SetFitness(0);
            }
        }
    }

    double Population::ComputeSparseness(Genome &genome) {
        // this will hold the distances from our new behavior
        std::vector<double> t_distances_list;
        t_distances_list.clear();

        // first add all distances from the population
        for (unsigned int i = 0; i < m_Species.size(); i++) {
            for (unsigned int j = 0; j < m_Species[i].m_Individuals.size(); j++) {
                double distance = genome.m_PhenotypeBehavior->Distance_To(m_Species[i].m_Individuals[j].m_PhenotypeBehavior);
                t_distances_list.emplace_back(distance);
            }
        }

        // then add all distances from the archive
        for (unsigned int i = 0; i < m_BehaviorArchive->size(); i++) {
            t_distances_list.emplace_back(genome.m_PhenotypeBehavior->Distance_To(&((*m_BehaviorArchive)[i])));
        }

        // sort the list, smaller first
        std::sort(t_distances_list.begin(), t_distances_list.end());

        // now compute the sparseness
        double t_sparseness = 0;
        for (unsigned int i = 1; i < (m_Parameters.NoveltySearch_K + 1); i++) {
            t_sparseness += t_distances_list[i];
        }
        t_sparseness /= m_Parameters.NoveltySearch_K;

        return t_sparseness;
    }

    // This is the main method performing novelty search. Performs one reproduction and assigns novelty scores based on the current population and the archive.
    // If a successful behavior was encountered, returns true and the genome a_SuccessfulGenome is overwritten with the genome generating the successful
    // behavior
    bool Population::NoveltySearchTick(Genome &a_SuccessfulGenome) {
        // Recompute the sparseness/fitness for all individuals in the population
        // This will introduce the constant pressure to do something new
        if ((m_NumEvaluations % m_Parameters.NoveltySearch_Recompute_Sparseness_Each) == 0) {
            for (unsigned int i = 0; i < m_Species.size(); i++) {
                for (unsigned int j = 0; j < m_Species[i].m_Individuals.size(); j++) {
                    m_Species[i].m_Individuals[j].SetFitness(ComputeSparseness(m_Species[i].m_Individuals[j]));
                }
            }
        }

        // OK now get the new baby
        Genome t_temp_genome;
        Genome *t_new_baby = Tick(t_temp_genome);

        // replace the new individual's behavior to point to the dead one's
        t_new_baby->m_PhenotypeBehavior = t_temp_genome.m_PhenotypeBehavior;

        // Now it is time to acquire the new behavior from the baby
        bool t_success = t_new_baby->m_PhenotypeBehavior->Acquire(t_new_baby);

        // if found a successful one, just copy it and return true
        if (t_success) {
            a_SuccessfulGenome = *t_new_baby;
            return true;
        }

        // We have the new behavior, now let's calculate the sparseness of the point in behavior space
        double t_sparseness = ComputeSparseness(*t_new_baby);

        // OK now we have the sparseness for this behavior if the sparseness is above Pmin, add this behavior to the archive
        m_GensSinceLastArchiving++;
        if (t_sparseness > m_Parameters.NoveltySearch_P_min) {
            // check to see if this behavior is already present in the archive if it is already present, abort addition
            bool present = false;

            // you can actually skip this code if the behavior comparison gets too slow maybe they don't repeat?
            /*for(unsigned int i=0; i<(*m_BehaviorArchive).size(); i++)
            {
                if ( (*(t_new_baby->m_PhenotypeBehavior)).m_Data == (*m_BehaviorArchive)[i].m_Data )
                {
                    present = true;
                    break;
                }
            }*/

            if (!present) {
                m_BehaviorArchive->emplace_back(*(t_new_baby->m_PhenotypeBehavior));
                m_GensSinceLastArchiving = 0;
                m_QuickAddCounter++;
            }
        } else {
            // no addition to the archive
            m_QuickAddCounter = 0;
        }

        // dynamic Pmin
        if (m_Parameters.NoveltySearch_Dynamic_Pmin) {
            // too many generations without adding to the archive?
            if (m_GensSinceLastArchiving > m_Parameters.NoveltySearch_No_Archiving_Stagnation_Treshold) {
                m_Parameters.NoveltySearch_P_min *= m_Parameters.NoveltySearch_Pmin_lowering_multiplier;
                if (m_Parameters.NoveltySearch_P_min < m_Parameters.NoveltySearch_Pmin_min) {
                    m_Parameters.NoveltySearch_P_min = m_Parameters.NoveltySearch_Pmin_min;
                }
            }

            // too much additions to the archive (one after another)?
            if (m_QuickAddCounter > m_Parameters.NoveltySearch_Quick_Archiving_Min_Evaluations) {
                m_Parameters.NoveltySearch_P_min *= m_Parameters.NoveltySearch_Pmin_raising_multiplier;
            }
        }

        // Now we assign a fitness score based on the sparseness
        // This is still now clear how, but for now fitness = sparseness
        t_new_baby->SetFitness(t_sparseness);

        a_SuccessfulGenome = *t_new_baby;

        // OK now last thing, check if this behavior is the one we're looking for.
        return t_new_baby->m_PhenotypeBehavior->Successful();
    }

    bool Population::Validate(std::string *error) const {
        const auto fail = [error](const std::string &message) {
            if (error != nullptr) *error = message;
            return false;
        };

        std::string parameter_error;
        if (!m_Parameters.Validate(&parameter_error)) return fail("Population parameters are invalid: " + parameter_error);
        if (m_Species.empty()) {
            return NumGenomes() == 0 ? true : fail("Population has genomes but no species");
        }
        if (NumGenomes() != m_Parameters.PopulationSize) {
            return fail("Population genome count does not match Parameters::PopulationSize");
        }
        if (!std::isfinite(m_BestFitnessEver) || !std::isfinite(m_CurrentMPC) || !std::isfinite(m_OldMPC) || !std::isfinite(m_BaseMPC)) {
            return fail("Population fitness and complexity state must be finite");
        }

        std::set<int> genome_ids;
        std::set<int> species_ids;
        int maximum_genome_id = -1;
        int maximum_species_id = 0;
        for (const auto &species : m_Species) {
            if (species.NumIndividuals() == 0) return fail("Population contains an empty species");
            if (species.ID() <= 0 || !species_ids.insert(species.ID()).second) return fail("Population species IDs must be positive and unique");
            maximum_species_id = std::max(maximum_species_id, species.ID());

            for (const auto &genome : species.m_Individuals) {
                std::string genome_error;
                if (!genome.Validate(&genome_error)) return fail("Population contains an invalid genome: " + genome_error);
                if (genome.GetID() < 0 || !genome_ids.insert(genome.GetID()).second) return fail("Population genome IDs must be non-negative and unique");
                maximum_genome_id = std::max(maximum_genome_id, genome.GetID());
            }
        }
        if (m_NextGenomeID <= static_cast<unsigned int>(maximum_genome_id)) return fail("Population next genome ID would reuse an existing ID");
        if (m_NextSpeciesID <= static_cast<unsigned int>(maximum_species_id)) return fail("Population next species ID would reuse an existing ID");
        return true;
    }

    namespace {
        std::string ReadDelimitedBlock(std::istream &input, const std::string &start, const std::string &end) {
            std::string token;
            input >> token;
            if (token != start) throw std::runtime_error("Population::Deserialize: missing " + start + " marker.");

            std::ostringstream block;
            block << start;
            std::string line;
            std::getline(input, line);
            block << line << '\n';
            while (std::getline(input, line)) {
                block << line << '\n';
                if (line == end) return block.str();
            }
            throw std::runtime_error("Population::Deserialize: missing " + end + " marker.");
        }
    }  // namespace

    std::string Population::Serialize() const {
        std::string validation_error;
        if (!Validate(&validation_error)) {
            throw std::runtime_error("Population::Serialize: " + validation_error);
        }
        std::ostringstream output;
        output << std::setprecision(std::numeric_limits<double>::max_digits10);
        output << "PopulationStart\n";
        output << "PopulationFormat 2\n";
        output << "PopulationState " << m_Generation << ' ' << m_NumEvaluations << ' ' << m_NextGenomeID << ' ' << m_NextSpeciesID << ' ' << m_BestFitnessEver
               << ' ' << m_ID << ' ' << m_GensSinceBestFitnessLastChanged << ' ' << m_EvalsSinceBestFitnessLastChanged << ' ' << m_GensSinceMPCLastChanged
               << ' ' << static_cast<int>(m_SearchMode) << ' ' << m_CurrentMPC << ' ' << m_OldMPC << ' ' << m_BaseMPC << ' ' << m_GensSinceLastArchiving << ' '
               << m_QuickAddCounter << '\n';
        output << "RNG " << std::quoted(m_RNG.Serialize()) << '\n';
        output << "Parameters\n" << m_Parameters.Serialize();
        output << "InnovationDatabase\n" << m_InnovationDatabase.Serialize();
        output << "BestGenome\n" << m_BestGenome.Serialize();
        output << "BestGenomeEver\n" << m_BestGenomeEver.Serialize();
        output << "GenomeArchive " << m_GenomeArchive.size() << '\n';
        for (const auto &genome : m_GenomeArchive) output << genome.Serialize();
        output << "Species " << m_Species.size() << '\n';
        for (const auto &species : m_Species) output << species.Serialize();
        output << "PopulationEnd\n";
        return output.str();
    }

    Population Population::Deserialize(const std::string &data) {
        std::istringstream input(data);
        std::string token;
        input >> token;
        if (token != "PopulationStart") throw std::runtime_error("Population::Deserialize: missing PopulationStart marker.");

        Population population;
        input >> token;
        if (token != "PopulationFormat") throw std::runtime_error("Population::Deserialize: expected PopulationFormat.");
        int version = 0;
        input >> version;
        if (version != 2) throw std::runtime_error("Population::Deserialize: unsupported format.");
        input >> token;
        if (token != "PopulationState") throw std::runtime_error("Population::Deserialize: missing PopulationState marker.");

        int search_mode = 0;
        input >> population.m_Generation >> population.m_NumEvaluations >> population.m_NextGenomeID >> population.m_NextSpeciesID >>
            population.m_BestFitnessEver >> population.m_ID >> population.m_GensSinceBestFitnessLastChanged >> population.m_EvalsSinceBestFitnessLastChanged >>
            population.m_GensSinceMPCLastChanged >> search_mode >> population.m_CurrentMPC >> population.m_OldMPC >> population.m_BaseMPC >>
            population.m_GensSinceLastArchiving >> population.m_QuickAddCounter;
        if (search_mode < COMPLEXIFYING || search_mode > BLENDED) throw std::runtime_error("Population::Deserialize: invalid search mode.");
        population.m_SearchMode = static_cast<SearchMode>(search_mode);

        std::string rng_state;
        input >> token >> std::quoted(rng_state);
        if (token != "RNG") throw std::runtime_error("Population::Deserialize: missing RNG marker.");
        population.m_RNG.Deserialize(rng_state);

        input >> token;
        if (token != "Parameters") throw std::runtime_error("Population::Deserialize: missing Parameters marker.");
        {
            std::ostringstream params_text;
            params_text << token << '\n';
            std::string line;
            while (std::getline(input, line)) {
                params_text << line << '\n';
                if (line == "NEAT_ParametersEnd") break;
            }
            std::istringstream params_input(params_text.str());
            if (population.m_Parameters.Load(params_input) != 0) throw std::runtime_error("Population::Deserialize: malformed parameters.");
        }

        input >> token;
        if (token != "InnovationDatabase") throw std::runtime_error("Population::Deserialize: missing InnovationDatabase marker.");
        population.m_InnovationDatabase = InnovationDatabase::Deserialize(ReadDelimitedBlock(input, "InnovationDatabaseStart", "InnovationDatabaseEnd"));

        input >> token;
        if (token != "BestGenome") throw std::runtime_error("Population::Deserialize: missing BestGenome marker.");
        population.m_BestGenome = Genome(input);
        input >> token;
        if (token != "BestGenomeEver") throw std::runtime_error("Population::Deserialize: missing BestGenomeEver marker.");
        population.m_BestGenomeEver = Genome(input);

        std::size_t archive_count = 0;
        input >> token >> archive_count;
        if (token != "GenomeArchive") throw std::runtime_error("Population::Deserialize: missing GenomeArchive marker.");
        population.m_GenomeArchive.clear();
        population.m_GenomeArchive.reserve(archive_count);
        for (std::size_t i = 0; i < archive_count; ++i) population.m_GenomeArchive.emplace_back(input);

        std::size_t species_count = 0;
        input >> token >> species_count;
        if (token != "Species") throw std::runtime_error("Population::Deserialize: missing Species marker.");
        population.m_Species.clear();
        population.m_Species.reserve(species_count);
        for (std::size_t i = 0; i < species_count; ++i) {
            population.m_Species.push_back(Species::Deserialize(ReadDelimitedBlock(input, "SpeciesStart", "SpeciesEnd")));
        }

        input >> token;
        if (token != "PopulationEnd") throw std::runtime_error("Population::Deserialize: missing PopulationEnd marker.");

        population.m_Genomes.clear();
        for (const auto &species : population.m_Species) {
            population.m_Genomes.insert(population.m_Genomes.end(), species.m_Individuals.begin(), species.m_Individuals.end());
        }
        population.m_TempSpecies.clear();
        population.m_BehaviorArchive = nullptr;
        return population;
    }

    void Population::SaveState(const char *a_FileName) const {
        if (a_FileName == nullptr) {
            throw std::invalid_argument("Population checkpoint filename is null");
        }

        std::ofstream output(a_FileName, std::ios::binary | std::ios::trunc);
        if (!output.is_open()) {
            throw std::runtime_error("Cannot open population checkpoint for writing");
        }
        output << Serialize();
        output.close();
        if (!output) {
            throw std::runtime_error("Failed to write population checkpoint");
        }
    }

}  // namespace NEAT
