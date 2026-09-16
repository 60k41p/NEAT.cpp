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

void Scale(vector<Real> &a_Values, const Real a_tr_min, const Real a_tr_max) {
    if (a_Values.empty()) return;

    Real t_max = 0.0, t_min = 0.0;
    GetMaxMin(a_Values, t_min, t_max);
    vector<Real> t_ValuesScaled;
    t_ValuesScaled.reserve(a_Values.size());
    for (vector<Real>::const_iterator t_It = a_Values.begin(); t_It != a_Values.end(); ++t_It) {
        Real t_ValueToBeScaled = (*t_It);
        Scale(t_ValueToBeScaled, t_min, t_max, a_tr_min, a_tr_max);
        t_ValuesScaled.push_back(t_ValueToBeScaled);
    }

    a_Values = t_ValuesScaled;
}
