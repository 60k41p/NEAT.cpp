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
 * File:        Utils.h
 * Description: Small numeric and string helpers shared by the genome, phenotype and
 *              speciation code (range queries, clamping, scaling, rounding and
 *              number-to-string conversion).
 *
 * References: Stanley & Miikkulainen, "Evolving Neural Networks through Augmenting
 *             Topologies" (2002), Sections 3-4 (compatibility and mutation magnitudes
 *             that these helpers support); see references/Evolving Neural Networks
 *             through Augmenting Topologies.pdf.md. Intra-repo users: src/Genes.h
 *             (trait initialization/mutation), src/Genome.cpp (weight handling).
 */

#pragma once

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

#include "AssertMacros.h"
#include "Random.h"

namespace NEAT {

    // Finds the minimum and maximum of a value list.
    // Uses lowest() (most negative) rather than min() (smallest positive) so an
    // all-negative input still yields a correct maximum.
    inline void getMaxMin(const std::vector<double> &vals, double &min, double &max) {
        max = std::numeric_limits<double>::lowest();
        min = std::numeric_limits<double>::max();
        for (std::vector<double>::const_iterator it = vals.begin(); it != vals.end(); ++it) {
            const double currentVal = (*it);
            if (currentVal > max) max = currentVal;

            if (currentVal < min) min = currentVal;
        }
    }

    // Converts an integer to a string.
    inline std::string intToString(const int arg) {
        std::ostringstream buffer;

        // send the int to the ostringstream
        buffer << arg;

        // capture the string
        return buffer.str();
    }

    // Converts a double to a string.
    inline std::string floatToString(const double arg) {
        std::ostringstream buffer;

        // send the double to the ostringstream
        buffer << arg;

        // capture the string
        return buffer.str();
    }

    // Clamps the value between the given bounds (inclusive). Bounds convert to the value type.
    template <typename ValueType, typename BoundType>
    inline void clamp(ValueType &value, const BoundType min, const BoundType max) {
        ASSERT(static_cast<ValueType>(min) <= static_cast<ValueType>(max));

        if (value < min) {
            value = min;
            return;
        }

        if (value > max) {
            value = max;
            return;
        }
    }

    // Rounds a double to the nearest integer (halves round up).
    inline int rounded(const double val) {
        const int integral = static_cast<int>(val);
        const double mantissa = val - integral;

        if (mantissa < 0.5) {
            return integral;
        }

        else {
            return integral + 1;
        }
    }

    // Rounds a double up or down depending on whether its mantissa is below the offset.
    inline int roundUnderOffset(const double val, const double offset) {
        const int integral = static_cast<int>(val);
        const double mantissa = val - integral;

        if (mantissa < offset) {
            return integral;
        } else {
            return integral + 1;
        }
    }

    // Scales the value "value", which lies in range [min .. max], into its relative
    // value in the range [targetMin .. targetMax]. Example: value=2 in [0 .. 4]
    // scaled to [-12 .. 12] gives 0.
    template <typename ValueType>
    inline void scale(ValueType &value, const double min, const double max, const double targetMin, const double targetMax) {
        const double sourceRange = max - min;
        const double targetRange = targetMax - targetMin;
        const double relativePosition = (value - min) / sourceRange;
        value = static_cast<ValueType>(targetMin + targetRange * relativePosition);
    }

    // Scales every entry of the vector from its current [min .. max] range into [targetMin .. targetMax].
    // Defined in Utils.cpp.
    void scale(std::vector<double> &values, const double targetMin, const double targetMax);

}  // namespace NEAT
