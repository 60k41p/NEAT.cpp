///////////////////////////////////////////////////////////////////////////////////////////
//    NEAT.cpp - C++ NeuroEvolution of Augmenting Topologies Library
//
//    Copyright (C) 2012 Peter Chervenski
//
//    This program is free software: you can redistribute it and/or modify
//    it under the terms of the GNU Lesser General Public License as published by
//    the Free Software Foundation, either version 3 of the License, or
//    (at your option) any later version.
//
//    This program is distributed in the hope that it will be useful,
//    but WITHOUT ANY WARRANTY; without even the implied warranty of
//    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//    GNU General Public License for more details.
//
//    You should have received a copy of the GNU Lesser General Public License
//    along with this program.  If not, see < http://www.gnu.org/licenses/ >.
//
//    Contact info:
//
//    Peter Chervenski < spookey@abv.bg >
//    Shane Ryan < shane.mcdonald.ryan@gmail.com >
///////////////////////////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////////////
// File:        Random.cpp
// Description: Definition for a class dealing with random numbers.
///////////////////////////////////////////////////////////////////////////////

#include "Random.h"

#include <math.h>

#include <chrono>

#include "Utils.h"

namespace NEAT {

    // Seeds the random number generator with this value
    void RNG::Seed(long a_Seed) { gen.seed(a_Seed); }

    void RNG::TimeSeed() {
        auto now = std::chrono::system_clock::now().time_since_epoch();
        long ms = (long)std::chrono::duration_cast<std::chrono::milliseconds>(now).count();
        Seed(ms);
    }

    // Returns randomly either 1 or -1
    int RNG::RandPosNeg() {
        std::uniform_int_distribution<int> dist(0, 1);
        int choice = dist(gen);
        if (choice == 0)
            return -1;
        else
            return 1;
    }

    // Returns a random integer between X and Y
    int RNG::RandInt(int aX, int aY) {
        std::uniform_int_distribution<int> dist(aX, aY);
        return dist(gen);
    }

    // Returns a random number from a uniform distribution in the range of [0 .. 1]
    double RNG::RandFloat() {
        std::uniform_real_distribution<double> dist(0.0, 1.0);
        return dist(gen);
    }

    // Returns a random number from a uniform distribution in the range of [-1 .. 1]
    double RNG::RandFloatSigned() { return (RandFloat() - RandFloat()); }

    // Returns a random number from a gaussian (normal) distribution in the range of [-1 .. 1]
    double RNG::RandGaussSigned() {
        std::normal_distribution<double> dist;
        double pick = dist(gen);
        Clamp(pick, -1, 1);
        return pick;
    }

    int RNG::Roulette(std::vector<double> &a_probs) {
        std::discrete_distribution<int> d_dist(a_probs.begin(), a_probs.end());
        return d_dist(gen);
    }

}  // namespace NEAT
   // namespace NEAT
