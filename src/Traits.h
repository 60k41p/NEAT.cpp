/*
 * NEAT.cpp: Portable, Zero-dependency C++17 NeuroEvolution Library
 *
 * Copyright (C) 2026 Gökalp Özcan
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
 * Contact info:
 * Gökalp Özcan <gokalp@mail.com>
 */

/*
 * File:        Traits.h
 * Description: Definitions for the trait parameter types (int, float, string and set variants) built on std::variant.
 */

#pragma once

#include <cmath>
#include <string>
#include <variant>
#include <vector>

#include "Types.h"

namespace NEAT {
    class intsetelement {
       public:
        int value = 0;

        // Comparison operator
        bool operator==(const intsetelement &rhs) const { return rhs.value == value; }

        // Assignment operator
        intsetelement &operator=(const intsetelement &) = default;
    };
    class floatsetelement {
       public:
        Real value = 0.0;

        // Comparison operator
        bool operator==(const floatsetelement &rhs) const { return rhs.value == value; }
        floatsetelement &operator=(const floatsetelement &) = default;
    };

    typedef std::variant<int, Real, std::string, intsetelement, floatsetelement> TraitType;

    class IntTraitParameters {
       public:
        int min, max;
        int mut_power;          // magnitude of max change up/down
        Real mut_replace_prob;  // probability to replace when mutating

        IntTraitParameters() {
            min = 0;
            max = 0;
            mut_power = 0;
            mut_replace_prob = 0;
        }
        IntTraitParameters &operator=(const IntTraitParameters &) = default;
    };
    class FloatTraitParameters {
       public:
        Real min, max;
        Real mut_power;         // magnitude of max change up/down
        Real mut_replace_prob;  // probability to replace when mutating

        FloatTraitParameters() {
            min = 0;
            max = 0;
            mut_power = 0;
            mut_replace_prob = 0;
        }
        FloatTraitParameters &operator=(const FloatTraitParameters &) = default;
    };
    class StringTraitParameters {
       public:
        std::vector<std::string> set;  // the set of possible strings
        std::vector<Real> probs;       // their respective probabilities for appearance
        StringTraitParameters &operator=(const StringTraitParameters &) = default;
    };
    class IntSetTraitParameters {
       public:
        std::vector<intsetelement> set;  // the set of possible ints
        std::vector<Real> probs;         // their respective probabilities for appearance
        IntSetTraitParameters &operator=(const IntSetTraitParameters &) = default;
    };
    class FloatSetTraitParameters {
       public:
        std::vector<floatsetelement> set;  // the set of possible floats
        std::vector<Real> probs;           // their respective probabilities for appearance
        FloatSetTraitParameters &operator=(const FloatSetTraitParameters &) = default;
    };

    class TraitParameters {
       public:
        Real m_ImportanceCoeff;
        Real m_MutationProb;

        std::string type;  // can be "int", "float", "str", "intset", "floatset"
        std::variant<IntTraitParameters, FloatTraitParameters, StringTraitParameters, IntSetTraitParameters, FloatSetTraitParameters> m_Details;

        std::string dep_key;                // counts only if this other trait exists..
        std::vector<TraitType> dep_values;  // and has one of these values

        // keep dep_key empty and no conditional logic will apply

        TraitParameters() {
            m_ImportanceCoeff = 0;
            m_MutationProb = 0;
            type = "int";
            m_Details = IntTraitParameters();
            dep_key = "";
            dep_values.emplace_back(std::string(""));
        }
        TraitParameters &operator=(const TraitParameters &) = default;
    };

    class Trait {
       public:
        TraitType value;

        Trait() {
            value = 0;
            dep_values.emplace_back(0);
            dep_key = "";
        }

        std::string dep_key;                // counts only if this other trait exists..
        std::vector<TraitType> dep_values;  // and has this value
        Trait &operator=(const Trait &) = default;

        bool operator==(const Trait &rhs) const { return value == rhs.value && dep_key == rhs.dep_key && dep_values == rhs.dep_values; }

        bool operator!=(const Trait &rhs) const { return !(*this == rhs); }
    };

}  // namespace NEAT
