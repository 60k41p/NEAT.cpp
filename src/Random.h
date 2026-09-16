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
 * Description: Declarations for a class dealing with random numbers.
 */

#pragma once

#include <limits>
#include <random>
#include <string>
#include <vector>

#include "Types.h"

namespace NEAT {

    class RNG {
        std::mt19937 gen;

       public:
        // Default-constructs with a time-based seed so a bare RNG is usable
        // without an explicit Seed() call. Seed() explicitly for reproducible runs.
        RNG() { TimeSeed(); }

        // Seeds the random number generator with this value
        void Seed(long seed);

        // Seeds the random number generator with time
        void TimeSeed();

        // Returns randomly either 1 or -1
        int RandPosNeg();

        // Returns a random integer between X and Y (throws on X > Y)
        int RandInt(int x, int y);

        // Returns a random number from a uniform distribution in the range of [0 .. 1]
        Real RandFloat();

        // Returns a random number from a uniform distribution in the range of [-1 .. 1]
        Real RandFloatSigned();

        // Returns a random number from a gaussian (normal) distribution in the range of [-1 .. 1]
        Real RandGaussSigned();

        // Returns a random number from a normal distribution (throws on non-finite
        // mean or non-positive/non-finite standard deviation)
        Real RandNormal(Real mean = 0.0, Real standardDeviation = 1.0);

        // Returns a random number from a Cauchy distribution (throws on non-finite
        // location or non-positive/non-finite scale; resamples non-finite draws)
        Real RandCauchy(Real location = 0.0, Real scale = 1.0);

        // Returns an index given a vector of probabilities (throws on empty,
        // non-finite or negative input; all-zero falls back to uniform choice)
        int Roulette(const std::vector<Real> &a_probs);

        // Serializes/deserializes the engine state for deterministic checkpoints
        std::string Serialize() const;
        void Deserialize(const std::string &data);
    };

}  // namespace NEAT
