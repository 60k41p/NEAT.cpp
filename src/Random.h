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
 * File:        Random.h
 * Description: Deterministic RNG wrapper (std::mt19937) used by every stochastic operator: initialization,
 *              mutation, mating, roulette selection and trait draws. Seed it explicitly for reproducible runs.
 *
 * References: C++ <random> distributions (uniform_int/uniform_real/normal/discrete_distribution).
 *             Intra-repo users: src/Genome.h, src/Population.h, src/Species.h, src/Genes.h.
 */

#pragma once

#include <limits>
#include <random>
#include <vector>

namespace NEAT {

    // Deterministic 32-bit Mersenne Twister engine. Copyable; copy the whole RNG to fork a stream.
    class RNG {
        // Engine state (seeded via seed()/timeSeed()).
        std::mt19937 gen_;

       public:
        // Seeds the random number generator with this value
        void seed(long seed);

        // Seeds the random number generator with time
        void timeSeed();

        // Returns randomly either 1 or -1
        int randPosNeg();

        // Returns a random integer between X and Y
        int randInt(int x, int y);

        // Returns a random number from a uniform distribution in the range of [0 .. 1]
        double randFloat();

        // Returns a random number from a uniform distribution in the range of [-1 .. 1]
        double randFloatSigned();

        // Returns a random number from a gaussian (normal) distribution in the range of [-1 .. 1]
        double randGaussSigned();

        // Returns an index sampled proportionally to the given weights (throws on empty input).
        int roulette(const std::vector<double> &probs);
    };

}  // namespace NEAT
