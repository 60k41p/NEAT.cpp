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
 * Description: some handy little functions
 */

#pragma once

#include <math.h>
#include <stdlib.h>

#include <algorithm>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

#include "AssertMacros.h"
#include "Random.h"
#include "Types.h"

namespace NEAT {

    inline void GetMaxMin(const std::vector<Real> &a_Vals, Real &a_Min, Real &a_Max) {
        if (a_Vals.empty()) {
            a_Min = 0;
            a_Max = 0;
            return;
        }
        auto result = std::minmax_element(a_Vals.begin(), a_Vals.end());
        a_Min = *result.first;
        a_Max = *result.second;
    }

    // converts an integer to a string
    inline std::string itos(const int a_Arg) {
        std::ostringstream t_Buffer;

        // send the int to the ostringstream
        t_Buffer << a_Arg;

        // capture the string
        return t_Buffer.str();
    }

    // converts a Real to a string with enough digits to parse back to the same value
    inline std::string ftos(const Real a_Arg) {
        std::ostringstream t_Buffer;

        // send the Real to the ostringstream
        t_Buffer << std::setprecision(std::numeric_limits<Real>::max_digits10) << a_Arg;

        // capture the string
        return t_Buffer.str();
    }

    // clamps the first argument between the second two
    inline void Clamp(Real &a_Arg, const Real a_Min, const Real a_Max) {
        ASSERT(a_Min <= a_Max);

        if (a_Arg < a_Min) {
            a_Arg = a_Min;
            return;
        }

        if (a_Arg > a_Max) {
            a_Arg = a_Max;
            return;
        }
    }

    // float overload retained for reference parity (Real == double today, so a
    // float lvalue cannot bind to Real& without it)
    inline void Clamp(float &a_Arg, const float a_Min, const float a_Max) {
        ASSERT(a_Min <= a_Max);

        if (a_Arg < a_Min) {
            a_Arg = a_Min;
            return;
        }

        if (a_Arg > a_Max) {
            a_Arg = a_Max;
            return;
        }
    }

    // clamps the first argument between the second two
    inline void Clamp(int &a_Arg, const int a_Min, const int a_Max) {
        ASSERT(a_Min <= a_Max);

        if (a_Arg < a_Min) {
            a_Arg = a_Min;
            return;
        }

        if (a_Arg > a_Max) {
            a_Arg = a_Max;
            return;
        }
    }

    // rounds a Real to the nearest integer (lround: halves away from zero, correct for negatives)
    inline int Rounded(const Real a_Val) { return static_cast<int>(std::lround(a_Val)); }

    // rounds a Real up or down depending on whether its mantissa is higher or lower than offset
    inline int RoundUnderOffset(const Real a_Val, const Real a_Offset) {
        // ASSERT(a_Offset < 1 && a_Offset > -1); ???!? Should this be a test for the offset
        const int t_Integral = static_cast<int>(a_Val);
        const Real t_Mantissa = a_Val - t_Integral;

        return (t_Mantissa < a_Offset) ? t_Integral : t_Integral + 1;
    }

    // Scales the value "a", that is in range [a_min .. a_max] into its relative value in the range [tr_min .. tr_max] Example: A=2, in the range [0 .. 4] .. we
    // want to scale it to the range [-12 .. 12] .. we get 0..
    inline void Scale(Real &a, const Real a_min, const Real a_max, const Real a_tr_min, const Real a_tr_max) {
        //        ASSERT((a >= a_min) && (a <= a_max));
        //        ASSERT(a_min <= a_max);
        //        ASSERT(a_tr_min <= a_tr_max);

        if (a_tr_min == a_tr_max) {
            a = a_tr_min;
            return;
        }
        if (std::fabs(a_max - a_min) < std::numeric_limits<Real>::epsilon()) {
            a = (a_tr_min + a_tr_max) / 2.0;
            return;
        }
        const Real t_a_r = a_max - a_min;
        const Real t_r = a_tr_max - a_tr_min;
        const Real rel_a = (a - a_min) / t_a_r;
        a = a_tr_min + t_r * rel_a;
    }

    inline Real Abs(Real x) { return (x < 0) ? -x : x; }

    // Scales every entry of the vector from its current [min .. max] range into [a_tr_min .. a_tr_max].
    // Defined in Utils.cpp.
    void Scale(std::vector<Real> &a_Values, const Real a_tr_min = 0.0, const Real a_tr_max = 1.0);

}  // namespace NEAT
