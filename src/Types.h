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
 * File:        Types.h
 * Description: Central scalar type alias for the whole library. Real is the floating-point type used for
 *              weights, activations, fitness values, compatibility distances, parameters, RNG draws, trait
 *              values and substrate coordinates. It is float (halved memory/bandwidth vs. double; single
 *              precision is ample for neuroevolution magnitudes) and can be flipped back to double in this
 *              one place for an exact-precision A/B comparison.
 *
 * References: IEEE 754 single-precision semantics; intra-repo users: every header under src/ that declares
 *             a numeric field or signature (Genes, Genome, NeuralNetwork, Parameters, Population, Species,
 *             Substrate, Traits, Random, Utils, PhenotypeBehavior).
 */

#pragma once

namespace NEAT {

    // Scalar type for all NEAT numerics (see file description above).
    using Real = float;

}  // namespace NEAT
