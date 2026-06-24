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
 *  energy.cpp
 *  Phase-2 end-to-end check: differentiate a host C++ "energy = chain of
 *  contractions" routine with Enzyme over TiledArray's custom rules, and
 *  cross-check the gradient against the native reverse-mode tape (tape.h) and a
 *  directional finite difference.
 */

// This is the Phase-2 acceptance demo (tiledarray_autodiff_plan.md, "Native vs
// Enzyme"): the SAME scalar energy is differentiated two independent ways — by
// the Enzyme plugin through the registered B1 custom rules (enzyme.h /
// enzyme_rules.cpp), and by the native define-by-run tape — and the two
// gradients must agree (they share the Part-A VJP math), with finite
// differences pinning the math itself.
//
// It is a standalone main (not a Boost module) so it can be the single TU
// compiled with `-fpass-plugin=ClangEnzyme-<ver>.so`; the rest of TA links in
// plugin-free. Built on demand (`ninja ad_enzyme_energy`), gated on
// TILEDARRAY_HAS_ENZYME.

#include <cmath>
#include <cstdio>

#include "tiledarray.h"

// enzyme_rules.h brings in the shim + custom-rule bodies and the registration
// globals (Enzyme needs them in this — the differentiated — module); enzyme.h
// declares the __enzyme_autodiff driver intrinsic and activity markers.
#include "TiledArray/ad/enzyme_rules.h"
#include "TiledArray/ad/enzyme.h"
#include "TiledArray/ad/ops.h"
#include "TiledArray/ad/tape.h"

using namespace TiledArray;
using RArray = TA::TArrayD;

// Enzyme activity-marker globals (matched by name by the plugin) and the
// autodiff intrinsic come from enzyme.h as `extern`; define the markers here.
int enzyme_dup;
int enzyme_const;
int enzyme_out;

namespace {

// Index annotations shared by the host routine and the references.
constexpr const char* kA = "i,k";
constexpr const char* kB = "k,j";
constexpr const char* kC = "i,j";

}  // namespace

// Instantiate the contraction rule for this TU's pattern: the annotation triple
// is baked into the symbol `ta_ad_contract_ik_kj_ij` (P0.3), so the shim is a
// fixed-arity function of array pointers only — no phantom annotation shadows.
// `add`/`sqnorm` are spec-free and pre-registered in enzyme_rules.h (tag `d`).
// Must be at global scope (not an anonymous namespace) so the registration
// global keeps the unmangled name the Enzyme pass scans for; the baked
// annotations match kA/kB/kC above.
TA_AD_CONTRACT_RULE(RArray, ik_kj_ij, "i,k", "k,j", "i,j");

namespace {

// Deterministic dense fill (counter-based, so np=1 runs are reproducible).
RArray make_dense(World& world, const TiledRange& tr, double seed) {
  RArray a(world, tr);
  double n = seed;
  for (auto tidx : a.trange().tiles_range()) {
    auto range = a.trange().make_tile_range(tidx);
    RArray::value_type tile(range);
    for (std::size_t i = 0; i < tile.size(); ++i) {
      n = std::fmod(n * 1103515245.0 + 12345.0, 2147483648.0);
      tile[i] = (n / 2147483648.0) * 2.0 - 1.0;  // in [-1, 1)
    }
    a.set(tidx, tile);
  }
  world.gop.fence();
  return a;
}

// E(A) = || A("i,k")·B("k,j") + H("i,j") ||²  via the plain functional ops;
// used for the finite-difference reference (no Enzyme, no tape).
double energy_val(const RArray& A, const RArray& B, const RArray& H) {
  RArray c1 = ad::contract(A, B, kA, kB, kC);
  RArray c2 = ad::add(c1, H);
  return ad::squared_norm(c2);
}

// dE/dA via the native reverse-mode tape (tape.h). B, H are constants.
RArray grad_tape(const RArray& A, const RArray& B, const RArray& H) {
  ad::Tape<RArray> tape;
  auto vA = ad::make_leaf(tape, A, /*requires_grad=*/true);
  auto vB = ad::make_leaf(tape, B, /*requires_grad=*/false);
  auto vH = ad::make_leaf(tape, H, /*requires_grad=*/false);
  auto c1 = ad::contract(vA, vB, kA, kB, kC);
  auto c2 = ad::add(c1, vH);
  auto s = ad::squared_norm(c2);
  tape.backward(s.id, 1.0);
  return tape.adjoint(vA.id).value();
}

double rel_diff(const RArray& x, const RArray& y) {
  const double dn = ad::norm2(ad::subt(x, y));
  const double xn = ad::norm2(x);
  return dn / (xn > 0 ? xn : 1.0);
}

}  // namespace

// The host routine Enzyme differentiates: a chain of the registered B1 custom
// rules (enzyme.h shims), so Enzyme treats each as atomic and never descends
// into MADNESS/BLAS/MPI. `extern "C"` for a stable symbol to hand __enzyme_autodiff.
//
// Crucially, the intermediates `c1`, `c2`, `e` are *caller-allocated* out-params,
// not locals: Enzyme cannot differentiate the C++ construction/destruction of a
// `DistArray` (it would try to reverse `reset_pimpl` and abort), so the
// differentiated body must contain no `DistArray` lifetime — only calls to the
// atomic custom rules. The caller owns every primal and every shadow.
extern "C" void ta_energy(const RArray* A, const RArray* B, const RArray* H,
                          RArray* c1, RArray* c2, double* e) {
  ta_ad_contract_ik_kj_ij(A, B, c1);
  ta_ad_add_d(c1, H, c2);
  ta_ad_sqnorm_d(c2, e);
}

int main(int argc, char** argv) {
  World& world = TA::initialize(argc, argv);
  int status = 0;

  {
    const TiledRange trA{{0, 2, 4}, {0, 2, 3}};  // (i, k): 4×3
    const TiledRange trB{{0, 2, 3}, {0, 2}};     // (k, j): 3×2
    const TiledRange trC{{0, 2, 4}, {0, 2}};     // (i, j): 4×2

    RArray A = make_dense(world, trA, 1.0);
    RArray B = make_dense(world, trB, 7.0);
    RArray H = make_dense(world, trC, 13.0);

    double E = 0.0;
    {
      RArray c1, c2;
      ta_energy(&A, &B, &H, &c1, &c2, &E);
      world.gop.fence();
    }

    // (1) Enzyme gradient w.r.t. A. Every primal and shadow is caller-owned.
    // Shadows are default-constructed (uninitialized) = the symbolic-zero seed;
    // the reverse rules move the first cotangent contribution in. All operands
    // are marked active (enzyme_dup) so each custom-rule call site has the
    // all-shadow activity its registered aug/reverse signature expects; we only
    // read back dA. `de = 1` seeds the scalar output's cotangent.
    RArray c1, c2;                    // primal intermediates (filled by forward)
    RArray dA, dB, dH, dc1, dc2;      // shadows (symbolic zero)
    double e = 0.0, de = 1.0;
    __enzyme_autodiff((void*)ta_energy, enzyme_dup, &A, &dA, enzyme_dup, &B, &dB,
                      enzyme_dup, &H, &dH, enzyme_dup, &c1, &dc1, enzyme_dup, &c2,
                      &dc2, enzyme_dup, &e, &de);
    world.gop.fence();

    // (2) native-tape gradient.
    RArray gA = grad_tape(A, B, H);

    // (3) directional finite difference along a random V: compare to
    // <grad, V>. Validates the derivative math itself (not just rule parity).
    RArray V = make_dense(world, trA, 99.0);
    const double eps = 1.0e-6;
    const double Ep = energy_val(ad::add(A, ad::scale(V, eps)), B, H);
    const double Em = energy_val(ad::add(A, ad::scale(V, -eps)), B, H);
    const double fd = (Ep - Em) / (2 * eps);
    const double an_enzyme = ad::dot(dA, V);
    const double an_tape = ad::dot(gA, V);

    const double parity = rel_diff(dA, gA);              // Enzyme vs tape
    const double fd_err_e = std::abs(an_enzyme - fd);    // Enzyme vs FD
    const double fd_err_t = std::abs(an_tape - fd);      // tape   vs FD

    if (world.rank() == 0) {
      std::printf("E(A,B,H)              = %.10f\n", E);
      std::printf("||dA_enzyme - dA_tape|| / ||dA_tape|| = %.3e\n", parity);
      std::printf("<dA_enzyme,V> = %.10f\n", an_enzyme);
      std::printf("<dA_tape,  V> = %.10f\n", an_tape);
      std::printf("finite-diff   = %.10f\n", fd);
      std::printf("|<dA_enzyme,V> - fd|  = %.3e\n", fd_err_e);
      std::printf("|<dA_tape,  V> - fd|  = %.3e\n", fd_err_t);
    }

    const bool ok = (parity < 1.0e-9) && (fd_err_e < 1.0e-4) &&
                    (fd_err_t < 1.0e-4);
    if (!ok) {
      status = 1;
      if (world.rank() == 0) std::printf("RESULT: FAIL\n");
    } else if (world.rank() == 0) {
      std::printf("RESULT: PASS\n");
    }
  }

  TA::finalize();
  return status;
}
