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

/* File: FileIO.h Description: Portable C file opening helper (fopen_s on MSVC, fopen elsewhere) with null-argument guard. */

#pragma once

#include <cstdio>

namespace NEAT {
    namespace detail {

        inline std::FILE *OpenFile(const char *filename, const char *mode) {
            if (filename == nullptr || mode == nullptr) {
                return nullptr;
            }

#ifdef _MSC_VER
            std::FILE *file = nullptr;
            return fopen_s(&file, filename, mode) == 0 ? file : nullptr;
#else
            return std::fopen(filename, mode);
#endif
        }

    }  // namespace detail
}  // namespace NEAT
