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

// The B5 custom-rule *bodies* live in enzyme_rules.h (a header, not this TU):
// Enzyme reads its type metadata from the rule function bodies in whichever
// module it transforms, so a declaration-only shim makes the plugin abort in
// type analysis. Each TU that differentiates host C++ over the TA primitives
// therefore compiles the rule bodies itself by including that header.
//
// This TU exists so the build/ABI/toolchain wiring (autodiff plan B5 / the
// enzyme dependency plan) has a library source to compile under the Enzyme pass
// plugin (-fpass-plugin=ClangEnzyme-<ver>.so; see FindOrFetchEnzyme.cmake). It
// pulls the rule header in as a compile-under-plugin smoke test — the static
// rules it instantiates here are internal and unused (no __enzyme_autodiff call
// in this TU), but their compiling under the plugin guards the library build.

#include <TiledArray/config.h>

#ifdef TILEDARRAY_HAS_ENZYME

#include <TiledArray/ad/enzyme_rules.h>

namespace TiledArray::ad {

/// Anchor symbol keeping this TU's object in the link; the differentiable rules
/// themselves are header-provided (enzyme_rules.h).
const char* enzyme_rules_anchor() noexcept {
  return "TiledArray Enzyme AD rules";
}

}  // namespace TiledArray::ad

#endif  // TILEDARRAY_HAS_ENZYME
