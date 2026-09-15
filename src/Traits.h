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
 * Description: Universal-trait system: typed, evolvable per-gene parameters (this file) consumed by
 *              the gene classes in src/Genes.h. A TraitParameters entry declares one named trait and its
 *              mutation behavior; each Gene then carries a concrete Trait value for it. Traits participate
 *              in mating (averaged or picked), mutation (replace vs. perturb) and speciation (weighted by
 *              importanceCoeff_ into the compatibility distance in src/Genome.cpp).
 *
 * References: Stanley & Miikkulainen, "Evolving Neural Networks through Augmenting Topologies" (2002),
 *             Section 4 (compatibility distance extended here with trait terms); see
 *             references/Evolving Neural Networks through Augmenting Topologies.pdf.md. Intra-repo users:
 *             src/Genes.h (Gene::initTraits/mateTraits/mutateTraits/getTraitDistances),
 *             src/Parameters.h (neuronTraits/linkTraits/genomeTraits maps), tests/TestTraitsGenes.cpp.
 */

#pragma once

#include <cmath>
#include <string>
#include <variant>
#include <vector>

#include "Types.h"

namespace NEAT {
    // A single selectable integer of a discrete trait set (e.g. a "loid" selector). Compared by value.
    class IntSetElement {
       public:
        // The selected integer.
        int value;

        // Comparison operator (compares by value).
        bool operator==(const IntSetElement &rhs) const { return rhs.value == value; }

        // Assignment operator.
        IntSetElement &operator=(const IntSetElement &g) {
            if (this != &g) {
                value = g.value;
            }

            return *this;
        }
    };
    // A single selectable float of a discrete trait set. Compared by value.
    class FloatSetElement {
       public:
        // The selected float.
        Real value;

        // Comparison operator (compares by value).
        bool operator==(const FloatSetElement &rhs) const { return rhs.value == value; }

        // Assignment operator.
        FloatSetElement &operator=(const FloatSetElement &g) {
            if (this != &g) {
                value = g.value;
            }

            return *this;
        }
    };

    // The runtime value of any trait: plain int/Real/string or one discrete-set element.
    using TraitType = std::variant<int, Real, std::string, IntSetElement, FloatSetElement>;

    // Mutation parameters for an integer trait: uniform init/mutation inside [min .. max].
    class IntTraitParameters {
       public:
        // Inclusive value range.
        int min, max;
        // Maximum perturbation up/down applied by a "modify" (non-replace) mutation.
        int mutPower;
        // Probability that a mutation replaces the value instead of perturbing it.
        Real mutReplaceProb;

        // Builds a zeroed parameter set.
        IntTraitParameters() {
            min = 0;
            max = 0;
            mutPower = 0;
            mutReplaceProb = 0;
        }

        // Assignment operator.
        IntTraitParameters &operator=(const IntTraitParameters &g) {
            if (this != &g) {
                min = g.min;
                max = g.max;
                mutPower = g.mutPower;
                mutReplaceProb = g.mutReplaceProb;
            }

            return *this;
        }
    };
    // Mutation parameters for a floating-point trait: uniform init/mutation inside [min .. max].
    class FloatTraitParameters {
       public:
        // Inclusive value range.
        Real min, max;
        // Maximum perturbation up/down applied by a "modify" (non-replace) mutation.
        Real mutPower;
        // Probability that a mutation replaces the value instead of perturbing it.
        Real mutReplaceProb;

        // Builds a zeroed parameter set.
        FloatTraitParameters() {
            min = 0;
            max = 0;
            mutPower = 0;
            mutReplaceProb = 0;
        }

        // Assignment operator.
        FloatTraitParameters &operator=(const FloatTraitParameters &g) {
            if (this != &g) {
                min = g.min;
                max = g.max;
                mutPower = g.mutPower;
                mutReplaceProb = g.mutReplaceProb;
            }

            return *this;
        }
    };
    // Selection-set parameters for a string trait: values are drawn with the given probabilities.
    class StringTraitParameters {
       public:
        // The admissible strings.
        std::vector<std::string> set;
        // Per-entry selection probabilities (resized to the set; see Gene::pickSetIndex in src/Genes.h).
        std::vector<Real> probs;
        // Assignment operator.
        StringTraitParameters &operator=(const StringTraitParameters &g) {
            if (this != &g) {
                set = g.set;
                probs = g.probs;
            }

            return *this;
        }
    };
    // Selection-set parameters for an integer trait.
    class IntSetTraitParameters {
       public:
        // The admissible integers.
        std::vector<IntSetElement> set;
        // Per-entry selection probabilities (resized to the set; see Gene::pickSetIndex in src/Genes.h).
        std::vector<Real> probs;

        // Assignment operator.
        IntSetTraitParameters &operator=(const IntSetTraitParameters &g) {
            if (this != &g) {
                set = g.set;
                probs = g.probs;
            }

            return *this;
        }
    };
    // Selection-set parameters for a float trait.
    class FloatSetTraitParameters {
       public:
        // The admissible floats.
        std::vector<FloatSetElement> set;
        // Per-entry selection probabilities (resized to the set; see Gene::pickSetIndex in src/Genes.h).
        std::vector<Real> probs;

        // Assignment operator.
        FloatSetTraitParameters &operator=(const FloatSetTraitParameters &g) {
            if (this != &g) {
                set = g.set;
                probs = g.probs;
            }

            return *this;
        }
    };

    // Declares one named evolvable trait: its type tag, type-specific details, mutation rate,
    // compatibility weight and optional dependency gating. The type tag selects the active
    // details_ alternative: "int", "float", "str" (alias "string"), "intset" or "floatset".
    class TraitParameters {
       public:
        // Weight of this trait's distance in the genome compatibility calculation.
        Real importanceCoeff_;
        // Per-reproduction probability that this trait is mutated.
        Real mutationProb_;

        // Type tag: "int", "float", "str", "intset" or "floatset" (legacy files may say "string"/"pyobject"; only the five above are honored).
        std::string type;
        // Type-specific parameters selected by type.
        std::variant<IntTraitParameters, FloatTraitParameters, StringTraitParameters, IntSetTraitParameters, FloatSetTraitParameters> details_;

        // Optional gating: the trait only counts (distance/mutation) when the named other trait exists...
        std::string depKey;
        // ...and holds one of these values. Keep depKey empty and no conditional logic will apply.
        std::vector<TraitType> depValues;

        // Builds a default integer-trait declaration with no gating.
        TraitParameters() {
            importanceCoeff_ = 0;
            mutationProb_ = 0;
            type = "int";
            details_ = IntTraitParameters();
            depKey = "";
            depValues.emplace_back(std::string(""));
        }

        // Assignment operator.
        TraitParameters &operator=(const TraitParameters &g) {
            if (this != &g) {
                importanceCoeff_ = g.importanceCoeff_;
                mutationProb_ = g.mutationProb_;
                type = g.type;
                details_ = g.details_;
                depKey = g.depKey;
                depValues = g.depValues;
            }

            return *this;
        }
    };

    // A concrete per-gene trait value: the runtime value plus the gating copy inherited at init/mate time.
    class Trait {
       public:
        // The current value.
        TraitType value;

        // Builds a zero integer trait with no gating.
        Trait() {
            value = 0;
            depValues.emplace_back(0);
            depKey = "";
        }

        // Gating key copied from the declaration (see TraitParameters::depKey).
        std::string depKey;
        // Gating values copied from the declaration (see TraitParameters::depValues).
        std::vector<TraitType> depValues;

        // Assignment operator.
        Trait &operator=(const Trait &g) {
            if (this != &g) {
                value = g.value;
                depValues = g.depValues;
                depKey = g.depKey;
            }

            return *this;
        }
    };

}  // namespace NEAT
