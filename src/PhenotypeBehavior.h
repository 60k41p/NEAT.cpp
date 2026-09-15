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
 * File:        PhenotypeBehavior.h
 * Description: Base class for novelty-search behavior characterizations: maps an evaluated genome to a
 *              behavior descriptor (data_) with a distance metric (distanceTo()) and an optional success
 *              predicate (successful()). Derive your domain's descriptor from this class and hand the
 *              populations to Population::initPhenotypeBehaviorData()/noveltySearchTick().
 *
 * References: Lehman & Stanley, "Abandoning Objectives: Evolution through the Search for Novelty Alone"
 *             (2011), Sections 2-4 (behavior space, sparseness, archive); novelty knobs live in
 *             src/Parameters.h and the archive in src/Population.h. Intra-repo users: src/Genome.h
 *             (phenotypeBehavior_), src/Population.cpp.
 */

#pragma once

#include <vector>

#include "AssertMacros.h"
#include "Types.h"

namespace NEAT {

    class Genome;

    // Always derive your domain's behavior characterization from this class and override acquire(),
    // distanceTo() and (for goal-directed runs) successful().
    class PhenotypeBehavior {
       public:
        virtual ~PhenotypeBehavior() {};

        // Behavior descriptor: a 2D matrix of Reals of arbitrary size. Sufficient for most domains
        // (e.g. endpoint coordinates, trajectory samples).
        std::vector<std::vector<Real> > data_;

        // Evaluates the genome, fills data_ and returns true when a successful (goal) behavior was observed.
        virtual bool acquire(Genome *genome) { return false; }

        // Behavioral distance to another descriptor (drives the sparseness score).
        virtual Real distanceTo(PhenotypeBehavior *other) { return 0; }

        // Whether this behavior counts as solving the task. Always true by default, so open-ended
        // runs need not override it.
        virtual bool successful() { return true; }

        // Compares descriptors element-wise (used by tests and archive checks).
        bool operator==(PhenotypeBehavior const &other) const { return data_ == other.data_; }
    };

};  // namespace NEAT
