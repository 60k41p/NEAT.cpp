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

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <sstream>
#include <stdexcept>

namespace NEAT {

    // Seeds the random number generator with this value
    void RNG::Seed(long a_Seed) { gen.seed(static_cast<std::mt19937::result_type>(a_Seed)); }

    void RNG::TimeSeed() {
        auto now = std::chrono::system_clock::now().time_since_epoch();
        long ms = (long)std::chrono::duration_cast<std::chrono::milliseconds>(now).count();
        Seed(ms);
    }

    // Returns randomly either 1 or -1
    int RNG::RandPosNeg() { return (RandInt(0, 1) == 0) ? -1 : 1; }

    // Returns a random integer between X and Y
    int RNG::RandInt(int aX, int aY) {
        if (aX > aY) throw std::invalid_argument("RNG::RandInt: invalid range (x > y).");
        std::uniform_int_distribution<int> dist(aX, aY);
        return dist(gen);
    }

    // Returns a random number from a uniform distribution in the range of [0 .. 1]
    double RNG::RandFloat() {
        std::uniform_real_distribution<double> dist(0.0, 1.0);
        return dist(gen);
    }

    // Returns a random number from a uniform distribution in the range of [-1 .. 1]
    double RNG::RandFloatSigned() { return 2.0 * RandFloat() - 1.0; }

    // Returns a random number from a gaussian (normal) distribution in the range of [-1 .. 1]
    double RNG::RandGaussSigned() {
        std::normal_distribution<double> dist(0.0, 1.0);
        double val = dist(gen);
        if (val > 1.0) val = 1.0;
        if (val < -1.0) val = -1.0;
        return val;
    }

    double RNG::RandNormal(double mean, double standardDeviation) {
        if (!std::isfinite(mean) || !std::isfinite(standardDeviation) || standardDeviation <= 0.0)
            throw std::invalid_argument("RNG::RandNormal requires a finite mean and positive standard deviation.");
        std::normal_distribution<double> distribution(mean, standardDeviation);
        return distribution(gen);
    }

    double RNG::RandCauchy(double location, double scale) {
        if (!std::isfinite(location) || !std::isfinite(scale) || scale <= 0.0)
            throw std::invalid_argument("RNG::RandCauchy requires a finite location and positive scale.");
        std::cauchy_distribution<double> distribution(location, scale);
        double result = distribution(gen);
        // The mathematical distribution is unbounded. Resample the extremely
        // rare non-finite floating-point result so callers always receive a usable value.
        while (!std::isfinite(result)) result = distribution(gen);
        return result;
    }

    int RNG::Roulette(const std::vector<double> &a_probs) {
        if (a_probs.empty()) throw std::invalid_argument("RNG::Roulette: probability vector is empty.");

        double maximum = 0.0;
        for (double p : a_probs) {
            if (!std::isfinite(p)) throw std::invalid_argument("RNG::Roulette: probabilities must be finite.");
            if (p < 0.0) throw std::invalid_argument("RNG::Roulette: probabilities cannot be negative.");
            maximum = std::max(maximum, p);
        }

        if (maximum <= 0.0) {
            int maxIndex = static_cast<int>(a_probs.size()) - 1;
            std::uniform_int_distribution<int> dist(0, maxIndex);
            return dist(gen);
        }

        // Scaling every weight by the maximum leaves the categorical distribution
        // unchanged and guarantees the running total cannot overflow.
        long double totalWide = 0.0L;
        for (double probability : a_probs) totalWide += probability / maximum;
        if (!std::isfinite(totalWide) || totalWide <= 0.0L || totalWide > static_cast<long double>(std::numeric_limits<double>::max()))
            throw std::overflow_error("RNG::Roulette: normalized probability total overflowed.");
        const double total = static_cast<double>(totalWide);
        std::uniform_real_distribution<double> dist(0.0, total);
        double r = dist(gen);
        double run = 0.0;
        size_t lastNonZero = 0;
        for (size_t idx = 0; idx < a_probs.size(); idx++) {
            const double w = a_probs[idx] / maximum;
            if (w > 0.0) {
                lastNonZero = idx;
                if (r < run + w) return static_cast<int>(idx);
                run += w;
            }
        }
        return static_cast<int>(lastNonZero);
    }

    std::string RNG::Serialize() const {
        std::ostringstream output;
        output << gen;
        return output.str();
    }

    void RNG::Deserialize(const std::string &data) {
        std::istringstream input(data);
        input >> gen;
        if (!input) throw std::invalid_argument("RNG::Deserialize: invalid generator state.");
    }

}  // namespace NEAT
