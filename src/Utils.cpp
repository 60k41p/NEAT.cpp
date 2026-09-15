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
 * File:        Utils.cpp
 * Description: Utility methods
 */

#include "Utils.h"

namespace NEAT {

    // Scales every entry of the vector from its current [min .. max] range into [targetMin .. targetMax].
    void scale(std::vector<double> &values, const double targetMin, const double targetMax) {
        double max = std::numeric_limits<double>::lowest();
        double min = std::numeric_limits<double>::max();
        getMaxMin(values, min, max);
        std::vector<double> valuesScaled;
        for (std::vector<double>::const_iterator it = values.begin(); it != values.end(); ++it) {
            double valueToBeScaled = (*it);
            scale(valueToBeScaled, min, max, targetMin, targetMax);
            valuesScaled.push_back(valueToBeScaled);
        }

        values = valuesScaled;
    }

}  // namespace NEAT
