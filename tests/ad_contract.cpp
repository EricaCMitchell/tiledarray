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
 *  ad_contract.cpp
 *  Phase-0 de-risk for the autodiff plan: verify the `contract` VJP.
 */

#include "tiledarray.h"
#include "unit_test_config.h"

#include "TiledArray/ad/ops.h"
#include "TiledArray/ad/shadow.h"
#include "TiledArray/ad/tape.h"

using namespace TiledArray;

namespace {

using ADArray = TA::TArray<double>;  // DistArray<Tensor<double>, DensePolicy>

// Fill every local tile with random values in [-1, 1]. Each rank fills the
// tiles it owns; the per-tile values therefore differ between np=1 and np=2,
// but every gradient identity checked below is rank-count-invariant by
// construction (the live SPMD check the plan calls for).
ADArray rand_array(const TiledRange& tr) {
  ADArray a(*GlobalFixture::world, tr);
  for (auto idx : *a.pmap()) {
    if (a.is_zero(idx)) continue;
    auto range = a.trange().make_tile_range(idx);
    Tensor<double> tile(range);
    for (std::size_t i = 0; i < tile.size(); ++i)
      tile[i] = (int(GlobalFixture::world->rand() % 2001) - 1000) / 1000.0;
    a.set(idx, tile);
  }
  GlobalFixture::world->gop.fence();
  return a;
}

// phi(A, B) = <Cbar, factor * A * B> -- the scalar whose gradient w.r.t. A, B
// the contract VJP (seeded with Cbar) must reproduce. dot is TA's unconjugated
// (bilinear) reduction, which is the right pairing for these real arrays.
double phi(const ADArray& Cbar, const ADArray& A, const ADArray& B, double factor,
           const std::string& aA, const std::string& aB,
           const std::string& aC) {
  ADArray C = ad::contract(A, B, aA, aB, aC, factor);
  return Cbar(aC).dot(C(aC)).get();
}

struct ADFixture {
  // i:{0,2,5}  k:{0,3,4}  j:{0,2}  -- multi-tile in every mode; k tiling shared
  // between A and B so the contraction is well-formed.
  TiledRange trA{{0, 2, 5}, {0, 3, 4}};  // (i, k)
  TiledRange trB{{0, 3, 4}, {0, 2}};     // (k, j)
  TiledRange trC{{0, 2, 5}, {0, 2}};     // (i, j)
  std::string aA = "i,k", aB = "k,j", aC = "i,j";
  double factor = 1.5;  // exercise the fused-gemm `factor` bookkeeping
};

}  // namespace

BOOST_FIXTURE_TEST_SUITE(ad_contract_suite, ADFixture)

// Ā and B̄ from the tape match the directional derivatives of phi computed by
// central finite differences. phi is linear in each operand (contract is
// bilinear, the other operand fixed), so the central difference is exact up to
// roundoff -- a wrong transpose, sign, or `factor` in the VJP fails this hard.
BOOST_AUTO_TEST_CASE(contract_vjp_vs_finite_difference) {
  ADArray A = rand_array(trA), B = rand_array(trB), Cbar = rand_array(trC);
  ADArray dA = rand_array(trA), dB = rand_array(trB);

  // reverse-mode gradient
  ad::Tape<ADArray> tape;
  auto va = ad::make_leaf(tape, A);
  auto vb = ad::make_leaf(tape, B);
  auto vc = ad::contract(va, vb, aA, aB, aC, factor);
  tape.backward(vc.id, Cbar);
  BOOST_REQUIRE(!tape.adjoint(va.id).is_zero());
  BOOST_REQUIRE(!tape.adjoint(vb.id).is_zero());
  const ADArray& Abar = tape.adjoint(va.id).value();
  const ADArray& Bbar = tape.adjoint(vb.id).value();

  // analytic directional derivatives <Ā, dA>, <B̄, dB>
  const double gA = Abar(aA).dot(dA(aA)).get();
  const double gB = Bbar(aB).dot(dB(aB)).get();

  // central finite differences of phi
  const double eps = 0.1;  // phi is linear in the perturbed operand => exact
  const double fdA = (phi(Cbar, ad::add(A, ad::scale(dA, eps)), B, factor, aA,
                          aB, aC) -
                      phi(Cbar, ad::add(A, ad::scale(dA, -eps)), B, factor, aA,
                          aB, aC)) /
                     (2.0 * eps);
  const double fdB = (phi(Cbar, A, ad::add(B, ad::scale(dB, eps)), factor, aA,
                          aB, aC) -
                      phi(Cbar, A, ad::add(B, ad::scale(dB, -eps)), factor, aA,
                          aB, aC)) /
                     (2.0 * eps);

  BOOST_CHECK_CLOSE(gA, fdA, 1e-6);  // 1e-6 % == 1e-8 relative
  BOOST_CHECK_CLOSE(gB, fdB, 1e-6);
}

// The forward/reverse adjoint-consistency identity (Part A, "the cheapest,
// strongest test"): Re<Cbar, JVP(dA, dB)> == Re<Ā, dA> + Re<B̄, dB>.
// JVP(dA, dB) = factor * (dA * B + A * dB). All terms are linear, so this is
// exact, and it exercises Ā and B̄ together against the JVP.
BOOST_AUTO_TEST_CASE(contract_jvp_vjp_adjoint_consistency) {
  ADArray A = rand_array(trA), B = rand_array(trB), Cbar = rand_array(trC);
  ADArray dA = rand_array(trA), dB = rand_array(trB);

  ad::Tape<ADArray> tape;
  auto va = ad::make_leaf(tape, A);
  auto vb = ad::make_leaf(tape, B);
  auto vc = ad::contract(va, vb, aA, aB, aC, factor);
  tape.backward(vc.id, Cbar);
  const ADArray& Abar = tape.adjoint(va.id).value();
  const ADArray& Bbar = tape.adjoint(vb.id).value();

  ADArray jvp = ad::add(ad::contract(dA, B, aA, aB, aC, factor),
                      ad::contract(A, dB, aA, aB, aC, factor));
  const double lhs = Cbar(aC).dot(jvp(aC)).get();
  const double rhs =
      Abar(aA).dot(dA(aA)).get() + Bbar(aB).dot(dB(aB)).get();

  BOOST_CHECK_CLOSE(lhs, rhs, 1e-9);
}

// Symbolic zero: a leaf that never feeds a recorded op has a zero gradient,
// represented as the un-materialized (null) cotangent -- "unconnected input =>
// zero gradient", with no array allocated for it.
BOOST_AUTO_TEST_CASE(unconnected_input_is_symbolic_zero) {
  ADArray A = rand_array(trA), B = rand_array(trB), Cbar = rand_array(trC);
  ADArray D = rand_array(trA);  // recorded as a leaf but never used

  ad::Tape<ADArray> tape;
  auto va = ad::make_leaf(tape, A);
  auto vb = ad::make_leaf(tape, B);
  auto vd = ad::make_leaf(tape, D);
  auto vc = ad::contract(va, vb, aA, aB, aC, factor);
  tape.backward(vc.id, Cbar);

  BOOST_CHECK(tape.adjoint(vd.id).is_zero());
  BOOST_CHECK(!tape.adjoint(va.id).is_zero());
}

BOOST_AUTO_TEST_SUITE_END()
