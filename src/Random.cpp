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
 * File:        Random.cpp
 * Description: Definition for a class dealing with random numbers.
 */

#include "Random.h"

#include <chrono>
#include <cmath>

#include "Utils.h"

namespace NEAT {

    // Seeds the random number generator with this value
    void RNG::seed(long seed) { gen_.seed(seed); }

    void RNG::timeSeed() {
        const std::chrono::system_clock::duration now = std::chrono::system_clock::now().time_since_epoch();
        long ms = static_cast<long>(std::chrono::duration_cast<std::chrono::milliseconds>(now).count());
        seed(ms);
    }

    // Returns randomly either 1 or -1
    int RNG::randPosNeg() {
        std::uniform_int_distribution<int> dist(0, 1);
        int choice = dist(gen_);
        if (choice == 0)
            return -1;
        else
            return 1;
    }

    // Returns a random integer between X and Y
    int RNG::randInt(int min, int max) {
        std::uniform_int_distribution<int> dist(min, max);
        return dist(gen_);
    }

    // Returns a random number from a uniform distribution in the range of [0 .. 1]
    double RNG::randFloat() {
        std::uniform_real_distribution<double> dist(0.0, 1.0);
        return dist(gen_);
    }

    // Returns a random number from a uniform distribution in the range of [-1 .. 1]
    double RNG::randFloatSigned() { return (randFloat() - randFloat()); }

    // Returns a random number from a gaussian (normal) distribution in the range of [-1 .. 1]
    double RNG::randGaussSigned() {
        std::normal_distribution<double> dist;
        double pick = dist(gen_);
        clamp(pick, -1, 1);
        return pick;
    }

    int RNG::roulette(const std::vector<double> &probs) {
        std::discrete_distribution<int> dDist(probs.begin(), probs.end());
        return dDist(gen_);
    }

}  // namespace NEAT
