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
 *  ad_forward.cpp
 *  Phase-1: forward-mode (Dual / JVP) correctness over the primitive set.
 */

#include "tiledarray.h"
#include "unit_test_config.h"

#include "TiledArray/ad/dual.h"
#include "TiledArray/ad/ops.h"

using namespace TiledArray;

namespace {

using RArray = TA::TArray<double>;

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

template <typename Array>
double max_abs_diff(const Array& x, const Array& y) {
  Array d;
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

struct FwdFixture {
  TiledRange trA{{0, 2, 5}, {0, 3, 4}};  // (i, k)
  TiledRange trB{{0, 3, 4}, {0, 2}};     // (k, j)
  TiledRange tr2{{0, 2, 5}, {0, 2, 3}};  // (i, j) 5x3
  TiledRange trSq{{0, 2, 4}, {0, 2, 4}};
  std::string aA = "i,k", aB = "k,j", aC = "i,j";
};

}  // namespace

BOOST_FIXTURE_TEST_SUITE(ad_forward_suite, FwdFixture)

// contract JVP: dC = factor*(dA*B + A*dB)
BOOST_AUTO_TEST_CASE(contract_jvp) {
  RArray A = rand_array(trA), B = rand_array(trB);
  RArray dA = rand_array(trA), dB = rand_array(trB);
  const double factor = 1.5;

  auto da = ad::make_dual(A, dA);
  auto db = ad::make_dual(B, dB);
  auto dc = ad::contract(da, db, aA, aB, aC, factor);

  RArray ref = ad::add(ad::contract(dA, B, aA, aB, aC, factor),
                       ad::contract(A, dB, aA, aB, aC, factor));
  BOOST_REQUIRE(dc.has_tangent());
  BOOST_CHECK_SMALL(max_abs_diff(*dc.tangent, ref), 1e-12);
  // primal is the plain contraction
  BOOST_CHECK_SMALL(
      max_abs_diff(dc.primal, ad::contract(A, B, aA, aB, aC, factor)), 1e-13);
}

// linear ops: add/subt/scale/permute, tangent matches the op on tangents.
BOOST_AUTO_TEST_CASE(linear_jvps) {
  RArray A = rand_array(tr2), B = rand_array(tr2);
  RArray dA = rand_array(tr2), dB = rand_array(tr2);
  auto da = ad::make_dual(A, dA), db = ad::make_dual(B, dB);

  auto s = ad::add(da, db);
  BOOST_CHECK_SMALL(max_abs_diff(*s.tangent, ad::add(dA, dB)), 1e-13);

  auto d = ad::subt(da, db);
  BOOST_CHECK_SMALL(max_abs_diff(*d.tangent, ad::subt(dA, dB)), 1e-13);

  auto sc = ad::scale(da, 2.5);
  BOOST_CHECK_SMALL(max_abs_diff(*sc.tangent, ad::scale(dA, 2.5)), 1e-13);

  auto p = ad::permute(da, "i,j", "j,i");
  RArray refp;
  refp("j,i") = dA("i,j");
  BOOST_CHECK_SMALL(max_abs_diff(*p.tangent, refp), 1e-13);
}

// Hadamard mult JVP: dA∘B + A∘dB
BOOST_AUTO_TEST_CASE(mult_jvp) {
  RArray A = rand_array(tr2), B = rand_array(tr2);
  RArray dA = rand_array(tr2), dB = rand_array(tr2);
  auto da = ad::make_dual(A, dA), db = ad::make_dual(B, dB);
  auto dc = ad::mult(da, db);
  RArray ref = ad::add(ad::mult(dA, B), ad::mult(A, dB));
  BOOST_CHECK_SMALL(max_abs_diff(*dc.tangent, ref), 1e-12);
}

// elementwise f(x)=x^2: tangent = 2A∘dA, verified by central FD.
BOOST_AUTO_TEST_CASE(elementwise_jvp) {
  RArray A = rand_array(tr2), dA = rand_array(tr2);
  auto f = [](double x) { return x * x; };
  auto df = [](double x) { return 2 * x; };
  auto da = ad::make_dual(A, dA);
  auto dc = ad::elementwise(da, f, df);

  const double eps = 1e-4;
  RArray fp = ad::elementwise(ad::add(A, ad::scale(dA, eps)), f, df);
  RArray fm = ad::elementwise(ad::add(A, ad::scale(dA, -eps)), f, df);
  RArray fd;
  fd("i,j") = (fp("i,j") - fm("i,j")) * (1.0 / (2 * eps));
  BOOST_CHECK_SMALL(max_abs_diff(*dc.tangent, fd), 1e-7);
}

// scalar reductions: tangent vs central finite differences along dA, dB.
BOOST_AUTO_TEST_CASE(reduction_jvps) {
  RArray A = rand_array(trSq), dA = rand_array(trSq);
  RArray B = rand_array(trSq), dB = rand_array(trSq);
  auto da = ad::make_dual(A, dA), db = ad::make_dual(B, dB);
  const double eps = 1e-4;

  auto fd_unary = [&](auto op) {
    return (op(ad::add(A, ad::scale(dA, eps))) -
            op(ad::add(A, ad::scale(dA, -eps)))) /
           (2 * eps);
  };

  BOOST_CHECK_CLOSE(ad::trace(da).tangent,
                    fd_unary([](const RArray& x) { return ad::trace(x); }),
                    1e-5);
  BOOST_CHECK_CLOSE(ad::sum(da).tangent,
                    fd_unary([](const RArray& x) { return ad::sum(x); }), 1e-5);
  BOOST_CHECK_CLOSE(
      ad::squared_norm(da).tangent,
      fd_unary([](const RArray& x) { return ad::squared_norm(x); }), 1e-4);
  BOOST_CHECK_CLOSE(ad::norm2(da).tangent,
                    fd_unary([](const RArray& x) { return ad::norm2(x); }),
                    1e-4);

  // dot(A,B): tangent = dot(dA,B) + dot(A,dB)
  auto dotd = ad::dot(da, db);
  double dot_fd =
      (ad::dot(ad::add(A, ad::scale(dA, eps)), ad::add(B, ad::scale(dB, eps))) -
       ad::dot(ad::add(A, ad::scale(dA, -eps)),
               ad::add(B, ad::scale(dB, -eps)))) /
      (2 * eps);
  BOOST_CHECK_CLOSE(dotd.tangent, dot_fd, 1e-4);
}

// Symbolic-zero tangent: a constant Dual carries no tangent, and an op over
// only-constant operands produces no tangent (unconnected => zero).
BOOST_AUTO_TEST_CASE(constant_has_symbolic_zero_tangent) {
  RArray A = rand_array(trA), B = rand_array(trB);
  auto ca = ad::make_constant(A), cb = ad::make_constant(B);
  auto dc = ad::contract(ca, cb, aA, aB, aC);
  BOOST_CHECK(!dc.has_tangent());

  // one active operand => tangent only from that operand
  RArray dA = rand_array(trA);
  auto da = ad::make_dual(A, dA);
  auto dc2 = ad::contract(da, cb, aA, aB, aC);
  BOOST_REQUIRE(dc2.has_tangent());
  BOOST_CHECK_SMALL(max_abs_diff(*dc2.tangent, ad::contract(dA, B, aA, aB, aC)),
                    1e-12);
}

BOOST_AUTO_TEST_SUITE_END()
