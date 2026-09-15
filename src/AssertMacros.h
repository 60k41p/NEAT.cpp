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
 * File:        AssertMacros.h
 * Description: ASSERT()/VERIFY() invariant macros, compiled in with DEBUG and optimised out otherwise.
 */

#pragma once

#include <assert.h>

#include <exception>
#include <iostream>
#include <stdexcept>
#include <string>

// kill any existing declarations
#ifdef ASSERT
#undef ASSERT
#endif

#ifdef VERIFY
#undef VERIFY
#endif

#ifdef DEBUG

#if 1

//--------------
//  debug macros
//--------------
#define BREAK_CPU()  //__asm { int 3 }

// Throws std::runtime_error (not bare std::exception) so that code paths which
// contractually throw runtime_error keep the same catchable type in debug builds.
#define ASSERT(expr)                                                                                                                  \
    {                                                                                                                                 \
        if (!(expr)) {                                                                                                                \
            std::cout << "\n*** ASSERT! ***\n" << __FILE__ ", line " << __LINE__ << ": " << #expr << " is false\n\n";                 \
            throw std::runtime_error(std::string("Assertion failed: ") + #expr + " at " + __FILE__ + ":" + std::to_string(__LINE__)); \
        }                                                                                                                             \
    }

#define VERIFY(expr)                                                                                                        \
    {                                                                                                                       \
        if (!(expr)) {                                                                                                      \
            std::cout << "\n*** VERIFY FAILED ***\n" << __FILE__ ", line " << __LINE__ << ": " << #expr << " is false\n\n"; \
            BREAK_CPU();                                                                                                    \
        }                                                                                                                   \
    }
#else

#define ASSERT(expr)                           \
    {                                          \
        if (!(expr)) {                         \
            std::cout << "\n*** ASSERT ***\n"; \
            assert(expr);                      \
        }                                      \
    }

#define VERIFY(expr)                                  \
    {                                                 \
        if (!(expr)) {                                \
            std::cout << "\n*** VERIFY FAILED ***\n"; \
            assert(expr);                             \
        }                                             \
    }

#endif

#else  // _DEBUG

//--------------
//  release macros
//--------------

// ASSERT gets optimised out completely
#define ASSERT(expr)

// verify has expression evaluated, but no further action taken
#define VERIFY(expr)  // if( expr ) {}

#endif
