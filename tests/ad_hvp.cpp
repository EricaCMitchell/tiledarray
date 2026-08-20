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
 *  ad_hvp.cpp
 *  Composability check: a Hessian-vector product from forward-over-reverse
 *  mode. A forward-mode Dual goes through the reverse tape.
 */

#include "tiledarray.h"
#include "unit_test_config.h"

#include "TiledArray/ad/dual.h"
#include "TiledArray/ad/ops.h"
#include "TiledArray/ad/tape.h"

using namespace TiledArray;

namespace {

using RArray = TA::TArray<double>;
using DualR = ad::Dual<RArray>;

RArray rand_array(const TiledRange& tr) {
  RArray a(*GlobalFixture::world, tr);
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

double max_abs_diff(const RArray& x, const RArray& y) {
  RArray d;
  d("i,j") = x("i,j") - y("i,j");
  double m = 0.0;
  for (auto idx : *d.pmap()) {
    if (d.is_zero(idx)) continue;
    auto tile = d.find(idx).get();
    for (std::size_t i = 0; i < tile.size(); ++i)
      m = std::max(m, std::abs(tile[i]));
  }
  return m;
}

// E(X) = ||X·X||^2. Plain reverse mode gives the gradient g(X) = ∂E/∂X, with
// the squared-norm cotangent C̄ = 2C as an explicit seed.
RArray grad_E(const RArray& X) {
  ad::Tape<RArray> tape;
  auto vx = ad::make_leaf(tape, X);
  auto vc = ad::contract(vx, vx, "i,k", "k,j", "i,j");
  RArray seed;
  seed("i,j") = 2.0 * vc.value("i,j");  // C̄ = d||C||^2/dC = 2C
  tape.backward(vc.id, seed);
  return tape.adjoint(vx.id).value();
}

}  // namespace

BOOST_AUTO_TEST_SUITE(ad_hvp_suite)

// Forward-over-reverse: the gradient computation runs on Dual<RArray> leaves
// seeded with the direction v. The reverse pass uses only the AD primitives, so
// it dispatches to the Dual overloads. The gradient then carries g as its
// primal and the Hessian-vector product Hv as its tangent.
BOOST_AUTO_TEST_CASE(hessian_vector_product_matches_fd_of_gradient) {
  TiledRange trSq{{0, 2, 4}, {0, 2, 4}};  // 4x4, square for X·X
  RArray A = rand_array(trSq), v = rand_array(trSq);

  ad::Tape<DualR> tape;
  auto va = ad::make_leaf(tape, ad::make_dual(A, v));
  auto vc = ad::contract(va, va, "i,k", "k,j", "i,j");
  DualR seed = ad::scale(vc.value, 2.0);  // C̄ = 2C as a dual (tangent = 2·dC)
  tape.backward(vc.id, seed);

  const DualR& g = tape.adjoint(va.id).value();
  BOOST_REQUIRE(g.has_tangent());
  const RArray& g_primal = g.primal;
  const RArray& Hv = *g.tangent;

  // primal of the dual gradient is exactly the plain reverse-mode gradient
  BOOST_CHECK_SMALL(max_abs_diff(g_primal, grad_E(A)), 1e-11);

  // Hv vs central finite differences of the gradient along v
  const double eps = 1e-4;
  RArray Ap, Am;
  Ap("i,j") = A("i,j") + eps * v("i,j");
  Am("i,j") = A("i,j") - eps * v("i,j");
  RArray gp = grad_E(Ap), gm = grad_E(Am);
  RArray fd;
  fd("i,j") = (gp("i,j") - gm("i,j")) * (1.0 / (2 * eps));

  BOOST_CHECK_SMALL(max_abs_diff(Hv, fd), 1e-6);
}

BOOST_AUTO_TEST_SUITE_END()
