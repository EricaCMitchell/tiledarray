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
 *  eig.cpp
 *  Phase-4 Enzyme cross-check for the heig custom rule (autodiff plan B5):
 *  differentiate the energy E(A) = Σ_i w_i λ_i + ⟨B, U W Uᵀ⟩ — written as a
 *  chain of registered shims — with the Enzyme plugin, and cross-check the
 *  gradient against the native tape (tape.h) and finite differences. Also
 *  exercises the degenerate eigenvalue-only (Hellmann–Feynman) path, which
 *  must not trip the degeneracy policy (F is built lazily), and the forward
 *  (JVP) registration via __enzyme_fwddiff.
 */

#include <cmath>
#include <cstdio>

#include "tiledarray.h"

#include "TiledArray/ad/enzyme.h"
#include "TiledArray/ad/enzyme_rules.h"
#include "TiledArray/ad/heig.h"
#include "TiledArray/ad/ops.h"
#include "TiledArray/ad/tape.h"

using namespace TiledArray;
using RArray = TA::TArrayD;

// Enzyme activity markers + intrinsics (declared extern in enzyme.h).
TA_AD_DEFINE_ENZYME_MARKERS;

// --- baked spec-bearing rules this TU uses (P0.3) --------------------------
// heig with the default (error) degeneracy policy; the two contraction
// patterns of the eigenvector functional. ta_ad_dot_d comes from the
// spec-free set stamped in enzyme_rules.h.
TA_AD_HEIG_RULE(RArray, d, TiledArray::ad::HeigDiffPolicy{});
TA_AD_CONTRACT_RULE(RArray, ik_kl_il, "i,k", "k,l", "i,l");
TA_AD_CONTRACT_RULE(RArray, il_jl_ij, "i,l", "j,l", "i,j");

// E(A) = Σ_i w_i λ_i + ⟨B, U W Uᵀ⟩, all intermediates and shadows
// caller-allocated (no local DistArray in the differentiated body); the
// scalar locals differentiate natively.
extern "C" void ta_eigE(const RArray* A, const RArray* w, const RArray* B,
                        const RArray* Wd, RArray* evals, RArray* evecs,
                        RArray* t1, RArray* t2, double* e) {
  ta_ad_heig_d(A, evals, evecs);
  double s1 = 0, s2 = 0;
  ta_ad_dot_d(w, evals, &s1);              // s1 = Σ w_i λ_i
  ta_ad_contract_ik_kl_il(evecs, Wd, t1);  // t1 = U·W
  ta_ad_contract_il_jl_ij(t1, evecs, t2);  // t2 = U W Uᵀ
  ta_ad_dot_d(B, t2, &s2);                 // s2 = ⟨B, U W Uᵀ⟩
  *e = s1 + s2;
}

// Eigenvalue-only energy Σ_i λ_i (w1 = ones): the Hellmann–Feynman path,
// well-defined even at exact degeneracy.
extern "C" void ta_eigTr(const RArray* A, const RArray* w1, RArray* evals,
                         RArray* evecs, double* e) {
  ta_ad_heig_d(A, evals, evecs);
  ta_ad_dot_d(w1, evals, e);
}

namespace {

bool g_fail = false;
void check(bool ok, const char* what, double err, double tol) {
  if (TA::get_default_world().rank() == 0)
    std::printf("  %-40s err=%.3e  tol=%.1e  %s\n", what, err, tol,
                ok ? "ok" : "FAIL");
  if (!ok) g_fail = true;
}

// Deterministic dense fill (counter-based; reproducible at any np).
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

RArray make_symmetric(World& world, const TiledRange& tr, double seed) {
  RArray x = make_dense(world, tr, seed);
  return ad::add(x, ad::permute(x, "i,j", "j,i"));
}

double rel_diff(const RArray& x, const RArray& y) {
  const double dn = ad::norm2(ad::subt(x, y));
  const double xn = ad::norm2(x);
  return dn / (xn > 0 ? xn : 1.0);
}

// The same energy on plain arrays (for finite differences).
double eigE_val(const RArray& A, const RArray& w, const RArray& B,
                const RArray& Wd) {
  auto r = ad::heig(A);
  const double s1 = ad::dot(w, r.evals);
  RArray t1 = ad::contract(r.evecs, Wd, "i,k", "k,l", "i,l");
  RArray t2 = ad::contract(t1, r.evecs, "i,l", "j,l", "i,j");
  return s1 + ad::dot(B, t2);
}

// The same energy on the native reverse-mode tape (the dual witness).
RArray eigE_grad_tape(const RArray& A, const RArray& w, const RArray& B,
                      const RArray& Wd) {
  ad::Tape<RArray> tape;
  auto vA = ad::make_leaf(tape, A, true);
  auto vw = ad::make_leaf(tape, w, false);
  auto vB = ad::make_leaf(tape, B, false);
  auto vW = ad::make_leaf(tape, Wd, false);
  auto r = ad::heig(vA);
  auto s1 = ad::dot(vw, r.evals);
  auto t1 = ad::contract(r.evecs, vW, "i,k", "k,l", "i,l");
  auto t2 = ad::contract(t1, r.evecs, "i,l", "j,l", "i,j");
  auto s2 = ad::dot(vB, t2);
  tape.accumulate_scalar(s1.id, 1.0);  // seed both energy terms
  tape.backward(s2.id, 1.0);
  return tape.adjoint(vA.id).value();
}

}  // namespace

int main(int argc, char** argv) {
  World& world = TA::initialize(argc, argv);
  {
    const TiledRange trSq{{0, 2, 4}, {0, 2, 4}};  // 4x4, 2x2 tiles
    const TiledRange trV{{0, 2, 4}};              // matching rank-1 tiling

    RArray A = make_symmetric(world, trSq, 3.0);
    RArray B = make_symmetric(world, trSq, 21.0);
    RArray w = make_dense(world, trV, 9.0);
    RArray V = make_symmetric(world, trSq, 55.0);  // FD / tangent direction
    RArray Wd(world, trSq);
    Wd.init_elements([](const auto& idx) {
      return idx[0] == idx[1] ? 1.0 + 0.5 * idx[0] : 0.0;
    });
    world.gop.fence();

    // --- reverse: dE/dA via Enzyme vs native tape vs finite differences ----
    if (world.rank() == 0) std::printf("[1] reverse through heig\n");
    RArray evals, evecs, t1, t2, dA, dw, dB, dWd, devals, devecs, dt1, dt2;
    double e = 0, de = 1;
    __enzyme_autodiff((void*)ta_eigE, TA_AD_DUP(A, dA), TA_AD_DUP(w, dw),
                      TA_AD_DUP(B, dB), TA_AD_DUP(Wd, dWd),
                      TA_AD_DUP(evals, devals), TA_AD_DUP(evecs, devecs),
                      TA_AD_DUP(t1, dt1), TA_AD_DUP(t2, dt2), TA_AD_DUP(e, de));
    world.gop.fence();

    RArray gA = eigE_grad_tape(A, w, B, Wd);
    check(rel_diff(dA, gA) < 1e-12, "reverse: enzyme vs tape",
          rel_diff(dA, gA), 1e-12);

    const double eps = 1e-5;
    const double fd = (eigE_val(ad::add(A, ad::scale(V, eps)), w, B, Wd) -
                       eigE_val(ad::subt(A, ad::scale(V, eps)), w, B, Wd)) /
                      (2 * eps);
    check(std::abs(ad::dot(dA, V) - fd) < 1e-4, "reverse: <grad,V> vs fd",
          std::abs(ad::dot(dA, V) - fd), 1e-4);

    // --- forward: dE along V via Enzyme fwddiff -----------------------------
    if (world.rank() == 0) std::printf("[2] forward through heig\n");
    RArray fevals, fevecs, ft1, ft2, tevals, tevecs, tt1, tt2, tw, tB, tWd;
    RArray tA = ad::scale(V, 1.0);  // input tangent seed = V
    double fe = 0, dfe = 0;
    __enzyme_fwddiff((void*)ta_eigE, TA_AD_DUP(A, tA), TA_AD_DUP(w, tw),
                     TA_AD_DUP(B, tB), TA_AD_DUP(Wd, tWd),
                     TA_AD_DUP(fevals, tevals), TA_AD_DUP(fevecs, tevecs),
                     TA_AD_DUP(ft1, tt1), TA_AD_DUP(ft2, tt2),
                     TA_AD_DUP(fe, dfe));
    world.gop.fence();

    check(std::abs(dfe - ad::dot(gA, V)) < 1e-9, "forward: jvp vs <grad,V>",
          std::abs(dfe - ad::dot(gA, V)), 1e-9);
    check(std::abs(dfe - fd) < 1e-4, "forward: jvp vs finite-diff",
          std::abs(dfe - fd), 1e-4);

    // --- degenerate eigenvalue-only (Hellmann–Feynman): gradient == I, no
    // throw under the default error policy (F is built lazily) --------------
    if (world.rank() == 0) std::printf("[3] degenerate Hellmann-Feynman\n");
    RArray D(world, trSq);
    const double dvals[4] = {1.0, 1.0, 2.0, 3.0};
    D.init_elements([&dvals](const auto& idx) {
      return idx[0] == idx[1] ? dvals[idx[0]] : 0.0;
    });
    RArray ones(world, trV);
    ones.init_elements([](const auto&) { return 1.0; });
    world.gop.fence();

    RArray gevals, gevecs, dD, dones, dgevals, dgevecs;
    double tre = 0, dtre = 1;
    __enzyme_autodiff((void*)ta_eigTr, TA_AD_DUP(D, dD), TA_AD_DUP(ones, dones),
                      TA_AD_DUP(gevals, dgevals), TA_AD_DUP(gevecs, dgevecs),
                      TA_AD_DUP(tre, dtre));
    world.gop.fence();

    check(std::abs(tre - 7.0) < 1e-10, "degenerate: tr A primal",
          std::abs(tre - 7.0), 1e-10);
    RArray I = ad::detail::identity_like(D);
    check(rel_diff(dD, I) < 1e-10, "degenerate: gradient == identity",
          rel_diff(dD, I), 1e-10);
  }
  if (world.rank() == 0)
    std::printf("RESULT: %s\n", g_fail ? "FAIL" : "PASS");
  TA::finalize();
  return g_fail ? 1 : 0;
}
