/*
 *  This file is a part of TiledArray.
 *  Copyright (C) 2026  Virginia Tech
 *
 *  This program is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program.  If not, see <http://www.gnu.org/licenses/>.
 *
 *  enzyme_rules.cpp
 *  Enzyme custom-derivative registration for TiledArray's AD primitives.
 */

// This translation unit is the single place where TiledArray registers its B1
// functional primitives (contract, add, scale, permute, mult, conj,
// elementwise, reductions) as Enzyme custom derivative rules
// (augmented-forward + reverse), so that host C++ differentiated by the Enzyme
// plugin treats each TA primitive as atomic and never descends into the
// MADNESS/BLAS/MPI layers Enzyme cannot trace.
//
// It is compiled with the Enzyme Clang pass plugin
// (-fpass-plugin=ClangEnzyme-<ver>.so); see cmake/modules/FindOrFetchEnzyme.cmake
// and enzyme_dependency_plan.md for that wiring.
//
// STATUS: scaffold only. The derivative-rule math and the
// __enzyme_register_gradient_* / __enzyme_register_fwddiff_* registrations are
// delivered by the autodiff plan (tiledarray_autodiff_plan.md B5). This file
// exists so the build, ABI, and toolchain plumbing can land and be tested
// independently of that math.

#include <TiledArray/config.h>

#ifdef TILEDARRAY_HAS_ENZYME

namespace TiledArray::ad {

// Anchor symbol: keeps this TU non-empty (and its object in the link) before
// any registration globals are added. Extended/replaced when the B5 rules land.
const char* enzyme_rules_scaffold() noexcept {
  return "TiledArray Enzyme AD rules (scaffold)";
}

}  // namespace TiledArray::ad

#endif  // TILEDARRAY_HAS_ENZYME
