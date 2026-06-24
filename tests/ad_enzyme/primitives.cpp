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
 *  primitives.cpp
 *  Phase-2 coverage check for the expanded Enzyme rule set (autodiff plan B5,
 *  enzyme_integration_suggestions.md P1.1/P1.2/P1.3): differentiate host C++ over
 *  the additional primitives — reverse AND forward mode, real, complex, and
 *  sparse — and cross-check each gradient against the native tape (tape.h) and a
 *  finite difference, the same dual-witness discipline as energy.cpp.
 */

#include <cmath>
#include <complex>
#include <cstdio>

#include "tiledarray.h"

#include "TiledArray/ad/enzyme_rules.h"
#include "TiledArray/ad/enzyme.h"
#include "TiledArray/ad/ops.h"
#include "TiledArray/ad/tape.h"

using namespace TiledArray;
using RArray = TA::TArrayD;
using ZArray = TA::TArrayZ;
using SArray = TA::TSpArrayD;
using cd = std::complex<double>;

// Enzyme activity markers + intrinsics (declared extern in enzyme.h).
int enzyme_dup;
int enzyme_const;
int enzyme_out;
int enzyme_dupnoneed;  // P2.2

// --- baked spec-bearing rules this TU uses (P0.3) --------------------------
// scale by 1.5 (real); contract i,k * k,j -> i,j (real, for the dupnoneed /
// inactive sections; sparse variant for the sparse energy); transpose and
// elementwise exp (with f' = exp) for the section-D coverage.
TA_AD_SCALE_RULE(RArray, 1p5, 1.5);
TA_AD_CONTRACT_RULE(RArray, ik_kj_ij, "i,k", "k,j", "i,j");
TA_AD_CONTRACT_RULE(SArray, sp_ik_kj_ij, "i,k", "k,j", "i,j");
TA_AD_PERMUTE_RULE(RArray, transpose, "i,j", "j,i");
TA_AD_ELEMENTWISE_RULE(RArray, exp, [](double v) { return std::exp(v); },
                       [](double v) { return std::exp(v); });

// A non-differentiable host "probe" used by the P2.6 section: marked
// `enzyme_inactive` so Enzyme treats an incidental runtime call as constant and
// skips it instead of trying (and failing) to reverse it. The counter proves it
// actually ran at differentiation time without contributing a derivative.
long g_probe_calls = 0;
TA_AD_INACTIVE void ta_ad_probe(const RArray* x) {
  (void)x;
  ++g_probe_calls;
}

namespace {

bool g_fail = false;
void check(bool ok, const char* what, double err, double tol) {
  if (TA::get_default_world().rank() == 0)
    std::printf("  %-34s err=%.3e  tol=%.1e  %s\n", what, err, tol,
                ok ? "ok" : "FAIL");
  if (!ok) g_fail = true;
}

// Deterministic dense fill (counter-based; reproducible at np=1).
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

// A block-sparse array with a fixed nonzero pattern, random nonzero values.
template <typename Pred>
SArray make_sparse(World& world, const TiledRange& tr, double seed,
                   Pred nonzero) {
  Tensor<float> shape_tensor(tr.tiles_range(), 0.0f);
  for (auto it = tr.tiles_range().begin(); it != tr.tiles_range().end(); ++it)
    if (nonzero(*it)) shape_tensor[*it] = 1.0f;
  SArray a(world, tr, SparseShape<float>(shape_tensor, tr));
  double n = seed;
  for (auto idx : *a.pmap()) {
    if (a.is_zero(idx)) continue;
    auto range = a.trange().make_tile_range(idx);
    Tensor<double> tile(range);
    for (std::size_t i = 0; i < tile.size(); ++i) {
      n = std::fmod(n * 1103515245.0 + 12345.0, 2147483648.0);
      tile[i] = (n / 2147483648.0) * 2.0 - 1.0;
    }
    a.set(idx, tile);
  }
  world.gop.fence();
  return a;
}

double rel_diff(const RArray& x, const RArray& y) {
  const double dn = ad::norm2(ad::subt(x, y));
  const double xn = ad::norm2(x);
  return dn / (xn > 0 ? xn : 1.0);
}

}  // namespace

// ===========================================================================
// Section A — real reverse + forward over mult / subt / scale / norm2.
//   L(X) = || 1.5 * (X∘X - X) ||
// Exercises: mult (saved residual), subt (linear), scale (baked α), norm2.
// ===========================================================================
extern "C" void ta_realA(const RArray* X, RArray* y, RArray* z, RArray* w,
                         double* e) {
  ta_ad_mult_d(X, X, y);    // y = X∘X
  ta_ad_subt_d(y, X, z);    // z = y - X
  ta_ad_scale_1p5(z, w);    // w = 1.5 z
  ta_ad_norm2_d(w, e);      // e = ‖w‖
}

namespace {

double realA_val(const RArray& X) {
  RArray y = ad::mult(X, X);
  RArray z = ad::subt(y, X);
  RArray w = ad::scale(z, 1.5);
  return ad::norm2(w);
}

RArray realA_grad_tape(const RArray& X) {
  ad::Tape<RArray> tape;
  auto vX = ad::make_leaf(tape, X, true);
  auto y = ad::mult(vX, vX);
  auto z = ad::subt(y, vX);
  auto w = ad::scale(z, 1.5);
  auto e = ad::norm2(w);
  tape.backward(e.id, 1.0);
  return tape.adjoint(vX.id).value();
}

void section_A(World& world) {
  if (world.rank() == 0) std::printf("[A] real reverse+forward (mult/subt/scale/norm2)\n");
  const TiledRange tr{{0, 2, 4}, {0, 2, 3}};
  RArray X = make_dense(world, tr, 3.0);
  RArray V = make_dense(world, tr, 51.0);

  // reverse: dL/dX via Enzyme.
  RArray y, z, w, dX, dy, dz, dw;
  double e = 0, de = 1;
  __enzyme_autodiff((void*)ta_realA, enzyme_dup, &X, &dX, enzyme_dup, &y, &dy,
                    enzyme_dup, &z, &dz, enzyme_dup, &w, &dw, enzyme_dup, &e,
                    &de);
  world.gop.fence();

  RArray gX = realA_grad_tape(X);
  check(rel_diff(dX, gX) < 1e-9, "reverse: enzyme vs tape", rel_diff(dX, gX),
        1e-9);

  // forward: directional derivative dL along V via Enzyme fwddiff.
  RArray fy, fz, fw, ty, tz, tw;
  RArray tX = ad::scale(V, 1.0);  // input tangent seed = V (a real copy)
  double fe = 0, dfe = 0;
  __enzyme_fwddiff((void*)ta_realA, enzyme_dup, &X, &tX, enzyme_dup, &fy, &ty,
                   enzyme_dup, &fz, &tz, enzyme_dup, &fw, &tw, enzyme_dup, &fe,
                   &dfe);
  world.gop.fence();

  const double an = ad::dot(gX, V);  // <grad, V>
  const double eps = 1e-6;
  const double fd =
      (realA_val(ad::add(X, ad::scale(V, eps))) -
       realA_val(ad::add(X, ad::scale(V, -eps)))) /
      (2 * eps);
  check(std::abs(dfe - an) < 1e-7, "forward: jvp vs <grad,V>",
        std::abs(dfe - an), 1e-7);
  check(std::abs(dfe - fd) < 1e-4, "forward: jvp vs finite-diff",
        std::abs(dfe - fd), 1e-4);
}

}  // namespace

// ===========================================================================
// Section B — complex ½z² convention litmus (P1.3).
//   f(z) = ½ Σ z²  (dot is bilinear/unconjugated).  Under the project's
//   reverse-mode convention grad returns the CONJUGATE of the holomorphic
//   derivative, so df/dz at z = 1+i must be conj(z) = 1 - i. A missing or
//   doubled conjugate would not crash — only this value test catches it, and it
//   must run on the ENZYME path, not just the tape.
// ===========================================================================
extern "C" void ta_litmus(const ZArray* z, cd* out) {
  cd d;
  ta_ad_dot_z(z, z, &d);  // d = Σ z²
  *out = 0.5 * d;         // ½ Σ z²  (Enzyme differentiates the scalar 0.5* natively)
}

namespace {

void section_B(World& world) {
  if (world.rank() == 0) std::printf("[B] complex 1/2 z^2 convention litmus\n");
  const TiledRange tr{{0, 1}};  // single 1-element tile
  ZArray z(world, tr);
  {
    auto range = z.trange().make_tile_range(0);
    ZArray::value_type tile(range);
    tile[0] = cd{1.0, 1.0};  // z = 1 + i
    z.set(0, tile);
    world.gop.fence();
  }

  ZArray dz;
  cd out{0, 0}, dout{1.0, 0.0};  // seed s̄ = 1
  __enzyme_autodiff((void*)ta_litmus, enzyme_dup, &z, &dz, enzyme_dup, &out,
                    &dout);
  world.gop.fence();

  cd g = dz.find(0).get()[0];
  const cd expected{1.0, -1.0};  // conj(1 + i)
  if (world.rank() == 0)
    std::printf("  grad = (%.6f, %.6f)  expected (1, -1)\n", g.real(),
                g.imag());
  check(std::abs(g - expected) < 1e-9, "complex litmus: grad == conj(z)",
        std::abs(g - expected), 1e-9);

  // parity against the native tape (shares the same VJP math).
  ad::Tape<ZArray> tape;
  auto vz = ad::make_leaf(tape, z, true);
  auto d = ad::dot(vz, vz);
  // ½·dot via a scalar scale on the cotangent seed: backward with s̄ = 0.5.
  tape.backward(d.id, cd{0.5, 0.0});
  cd gt = tape.adjoint(vz.id).value().find(0).get()[0];
  check(std::abs(g - gt) < 1e-12, "complex litmus: enzyme vs tape",
        std::abs(g - gt), 1e-12);
}

}  // namespace

// ===========================================================================
// Section C — sparse reverse (P1.3): block-sparse energy
//   E = || A·B + H ||², gradient w.r.t A, Enzyme vs tape (structure + value).
// ===========================================================================
extern "C" void ta_spE(const SArray* A, const SArray* B, const SArray* H,
                       SArray* c1, SArray* c2, double* e) {
  ta_ad_contract_sp_ik_kj_ij(A, B, c1);
  ta_ad_add_sd(c1, H, c2);
  ta_ad_sqnorm_sd(c2, e);
}

namespace {

SArray spE_grad_tape(const SArray& A, const SArray& B, const SArray& H) {
  ad::Tape<SArray> tape;
  auto vA = ad::make_leaf(tape, A, true);
  auto vB = ad::make_leaf(tape, B, false);
  auto vH = ad::make_leaf(tape, H, false);
  auto c1 = ad::contract(vA, vB, "i,k", "k,j", "i,j");
  auto c2 = ad::add(c1, vH);
  auto s = ad::squared_norm(c2);
  tape.backward(s.id, 1.0);
  return tape.adjoint(vA.id).value();
}

double sp_rel_diff(const SArray& x, const SArray& y) {
  SArray d = ad::subt(x, y);
  const double dn = std::sqrt(ad::squared_norm(d));
  const double xn = std::sqrt(ad::squared_norm(x));
  return dn / (xn > 0 ? xn : 1.0);
}

void section_C(World& world) {
  if (world.rank() == 0) std::printf("[C] sparse reverse (contract/add/sqnorm)\n");
  const float saved = SparseShape<float>::threshold();
  SparseShape<float>::threshold(std::numeric_limits<float>::min());
  {
    // 3×3 tile grid; drop one tile from each operand so the pattern is nontrivial.
    TiledRange tr{{0, 2, 4, 6}, {0, 2, 4, 6}};
    auto dropA = [](const auto& i) { return !(i[0] == 0 && i[1] == 2); };
    auto dropB = [](const auto& i) { return !(i[0] == 1 && i[1] == 1); };
    auto all = [](const auto&) { return true; };
    SArray A = make_sparse(world, tr, 5.0, dropA);
    SArray B = make_sparse(world, tr, 17.0, dropB);
    SArray H = make_sparse(world, tr, 29.0, all);

    SArray c1, c2, dA, dB, dH, dc1, dc2;
    double e = 0, de = 1;
    __enzyme_autodiff((void*)ta_spE, enzyme_dup, &A, &dA, enzyme_dup, &B, &dB,
                      enzyme_dup, &H, &dH, enzyme_dup, &c1, &dc1, enzyme_dup,
                      &c2, &dc2, enzyme_dup, &e, &de);
    world.gop.fence();

    SArray gA = spE_grad_tape(A, B, H);
    check(sp_rel_diff(dA, gA) < 1e-9, "sparse reverse: enzyme vs tape",
          sp_rel_diff(dA, gA), 1e-9);
  }
  SparseShape<float>::threshold(saved);
}

}  // namespace

// ===========================================================================
// Section D — reverse parity (vs tape + FD) for the structurally-distinct
// remaining primitives: elementwise (nonlinear, f'-residual) + sum, and
// permute (inverse-permutation VJP) + inner_product (sesquilinear).
// ===========================================================================
extern "C" void ta_expsum(const RArray* X, RArray* y, double* e) {
  ta_ad_ew_exp(X, y);    // y = exp(X) elementwise
  ta_ad_sum_d(y, e);     // e = Σ exp(X)
}
extern "C" void ta_permip(const RArray* X, RArray* y, double* e) {
  ta_ad_permute_transpose(X, y);  // y = Xᵀ
  ta_ad_inner_d(y, X, e);         // e = Σ conj(Xᵀ)·X
}

namespace {

double expsum_val(const RArray& X) {
  return ad::sum(ad::elementwise(
      X, [](double v) { return std::exp(v); },
      [](double v) { return std::exp(v); }));
}
RArray expsum_grad_tape(const RArray& X) {
  ad::Tape<RArray> tape;
  auto vX = ad::make_leaf(tape, X, true);
  auto y = ad::elementwise(
      vX, [](double v) { return std::exp(v); },
      [](double v) { return std::exp(v); });
  auto e = ad::sum(y);
  tape.backward(e.id, 1.0);
  return tape.adjoint(vX.id).value();
}
double permip_val(const RArray& X) {
  return ad::inner_product(ad::permute(X, "i,j", "j,i"), X);
}
RArray permip_grad_tape(const RArray& X) {
  ad::Tape<RArray> tape;
  auto vX = ad::make_leaf(tape, X, true);
  auto y = ad::permute(vX, "i,j", "j,i");
  auto e = ad::inner_product(y, vX);
  tape.backward(e.id, 1.0);
  return tape.adjoint(vX.id).value();
}

void section_D(World& world) {
  if (world.rank() == 0)
    std::printf("[D] reverse parity for elementwise/sum/permute/inner_product\n");
  const TiledRange tr{{0, 2, 4}, {0, 2, 4}};  // square so transpose round-trips
  RArray X = make_dense(world, tr, 7.0);
  RArray V = make_dense(world, tr, 71.0);
  const double eps = 1e-6;

  {  // exp + sum
    RArray y, dX, dy;
    double e = 0, de = 1;
    __enzyme_autodiff((void*)ta_expsum, enzyme_dup, &X, &dX, enzyme_dup, &y, &dy,
                      enzyme_dup, &e, &de);
    world.gop.fence();
    RArray gX = expsum_grad_tape(X);
    check(rel_diff(dX, gX) < 1e-9, "expsum reverse: enzyme vs tape",
          rel_diff(dX, gX), 1e-9);
    const double fd = (expsum_val(ad::add(X, ad::scale(V, eps))) -
                       expsum_val(ad::add(X, ad::scale(V, -eps)))) /
                      (2 * eps);
    check(std::abs(ad::dot(dX, V) - fd) < 1e-4, "expsum reverse: <grad,V> vs fd",
          std::abs(ad::dot(dX, V) - fd), 1e-4);
  }
  {  // permute + inner_product
    RArray y, dX, dy;
    double e = 0, de = 1;
    __enzyme_autodiff((void*)ta_permip, enzyme_dup, &X, &dX, enzyme_dup, &y, &dy,
                      enzyme_dup, &e, &de);
    world.gop.fence();
    RArray gX = permip_grad_tape(X);
    check(rel_diff(dX, gX) < 1e-9, "permip reverse: enzyme vs tape",
          rel_diff(dX, gX), 1e-9);
    const double fd = (permip_val(ad::add(X, ad::scale(V, eps))) -
                       permip_val(ad::add(X, ad::scale(V, -eps)))) /
                      (2 * eps);
    check(std::abs(ad::dot(dX, V) - fd) < 1e-4, "permip reverse: <grad,V> vs fd",
          std::abs(ad::dot(dX, V) - fd), 1e-4);
  }
}

}  // namespace

// ===========================================================================
// Section E — checkpoint/recompute seam (P2.1). The residual a reverse rule
// saves is a `Checkpoint`: it may hold the operand materialized (today's
// default) OR a recompute recipe. Prove the recompute branch reproduces the
// materialized one bit-for-bit, so the seam is safe to switch on later.
// ===========================================================================
namespace {

void section_E(World& world) {
  if (world.rank() == 0) std::printf("[E] checkpoint/recompute seam (P2.1)\n");
  namespace ed = TiledArray::ad::edetail;
  const TiledRange tr{{0, 2, 4}, {0, 2, 3}};
  RArray base = make_dense(world, tr, 123.0);

  // Materialized: holds the operand directly (the zero-overhead default).
  ed::Checkpoint<RArray> held(base);
  // Recompute: stores a recipe (here: rebuild the operand from `base`) instead
  // of pinning it; get() materializes and memoizes on first read.
  ed::Checkpoint<RArray> recomputed(
      std::function<RArray()>([&] { return ad::scale(base, 1.0); }));

  const double err = rel_diff(held.get(), recomputed.get());
  check(err < 1e-15, "checkpoint: recompute == materialized", err, 1e-15);
  // Second read must hit the memoized copy (still equal, no recompute side
  // effects).
  const double err2 = rel_diff(held.get(), recomputed.get());
  check(err2 < 1e-15, "checkpoint: memoized second read", err2, 1e-15);
}

}  // namespace

// ===========================================================================
// Sections F & G share the energy chain E(A) = ||A·B + H||² and its native-tape
// gradient w.r.t. A. F exercises enzyme_dupnoneed (P2.2); G exercises an
// enzyme_inactive runtime probe (P2.6). Both demand the gradient be UNCHANGED.
// ===========================================================================
namespace {

RArray energyF_grad_tape(const RArray& A, const RArray& B, const RArray& H) {
  ad::Tape<RArray> tape;
  auto vA = ad::make_leaf(tape, A, true);
  auto vB = ad::make_leaf(tape, B, false);
  auto vH = ad::make_leaf(tape, H, false);
  auto c1 = ad::contract(vA, vB, "i,k", "k,j", "i,j");
  auto c2 = ad::add(c1, vH);
  auto s = ad::squared_norm(c2);
  tape.backward(s.id, 1.0);
  return tape.adjoint(vA.id).value();
}

}  // namespace

// E = ||A·B + H||². c1 = A·B feeds the *linear* add (which saves no residual),
// so c1's primal is never read in the reverse pass — the P2.2 dead-intermediate.
extern "C" void ta_energyF(const RArray* A, const RArray* B, const RArray* H,
                           RArray* c1, RArray* c2, double* e) {
  ta_ad_contract_ik_kj_ij(A, B, c1);
  ta_ad_add_d(c1, H, c2);
  ta_ad_sqnorm_d(c2, e);
}

// Same chain, plus an incidental non-differentiable runtime call on c1.
extern "C" void ta_energyG(const RArray* A, const RArray* B, const RArray* H,
                           RArray* c1, RArray* c2, double* e) {
  ta_ad_contract_ik_kj_ij(A, B, c1);
  ta_ad_probe(c1);  // enzyme_inactive: differentiated, but skipped by Enzyme
  ta_ad_add_d(c1, H, c2);
  ta_ad_sqnorm_d(c2, e);
}

namespace {

// Build the operands and the trusted gradient for the shared energy chain.
struct EnergyFixture {
  RArray A, B, H, gA;
};
EnergyFixture make_energy(World& world) {
  const TiledRange trA{{0, 2, 4}, {0, 2, 3}};  // (i,k)
  const TiledRange trB{{0, 2, 3}, {0, 2}};     // (k,j)
  const TiledRange trC{{0, 2, 4}, {0, 2}};     // (i,j)
  RArray A = make_dense(world, trA, 2.0);
  RArray B = make_dense(world, trB, 8.0);
  RArray H = make_dense(world, trC, 14.0);
  RArray gA = energyF_grad_tape(A, B, H);
  return {A, B, H, gA};
}

// Section F — P2.2: mark the dead intermediate c1 `dupnoneed`; gradient unchanged.
void section_F(World& world) {
  if (world.rank() == 0)
    std::printf("[F] enzyme_dupnoneed for a dead intermediate (P2.2)\n");
  auto fx = make_energy(world);
  RArray c1, c2, dA, dB, dH, dc1, dc2;
  double e = 0, de = 1;
  __enzyme_autodiff((void*)ta_energyF, TA_AD_DUP(fx.A, dA), TA_AD_DUP(fx.B, dB),
                    TA_AD_DUP(fx.H, dH), TA_AD_DUPNONEED(c1, dc1),
                    TA_AD_DUP(c2, dc2), TA_AD_DUP(e, de));
  world.gop.fence();
  check(rel_diff(dA, fx.gA) < 1e-9, "dupnoneed: gradient unchanged vs tape",
        rel_diff(dA, fx.gA), 1e-9);
}

// Section G — P2.6: an enzyme_inactive runtime call inside the differentiated
// body is skipped (no derivative) but still executes; gradient unchanged.
void section_G(World& world) {
  if (world.rank() == 0)
    std::printf("[G] enzyme_inactive runtime probe (P2.6)\n");
  auto fx = make_energy(world);
  const long before = g_probe_calls;
  RArray c1, c2, dA, dB, dH, dc1, dc2;
  double e = 0, de = 1;
  __enzyme_autodiff((void*)ta_energyG, TA_AD_DUP(fx.A, dA), TA_AD_DUP(fx.B, dB),
                    TA_AD_DUP(fx.H, dH), TA_AD_DUP(c1, dc1), TA_AD_DUP(c2, dc2),
                    TA_AD_DUP(e, de));
  world.gop.fence();
  check(rel_diff(dA, fx.gA) < 1e-9, "inactive: gradient unchanged vs tape",
        rel_diff(dA, fx.gA), 1e-9);
  // The probe ran during the differentiated forward pass (Enzyme reverse mode
  // still executes the augmented forward), proving it was skipped, not elided.
  check(g_probe_calls > before, "inactive: probe executed (not differentiated)",
        0.0, 0.0);
}

}  // namespace

int main(int argc, char** argv) {
  World& world = TA::initialize(argc, argv);
  section_A(world);
  section_B(world);
  section_C(world);
  section_D(world);
  section_E(world);
  section_F(world);
  section_G(world);
  if (world.rank() == 0)
    std::printf("RESULT: %s\n", g_fail ? "FAIL" : "PASS");
  TA::finalize();
  return g_fail ? 1 : 0;
}
