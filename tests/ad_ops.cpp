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
 *  ad_ops.cpp
 *  Forward-value correctness of the functional primitive set.
 */

#include "tiledarray.h"
#include "unit_test_config.h"

#include "TiledArray/ad/ops.h"

using namespace TiledArray;

namespace {

using RArray = TA::TArray<double>;
using CArray = TA::TArray<std::complex<double>>;

template <typename Array>
Array rand_array(const TiledRange& tr) {
  using T = typename Array::element_type;
  Array a(*GlobalFixture::world, tr);
  for (auto idx : *a.pmap()) {
    if (a.is_zero(idx)) continue;
    auto range = a.trange().make_tile_range(idx);
    typename Array::value_type tile(range);
    for (std::size_t i = 0; i < tile.size(); ++i) {
      double re = (int(GlobalFixture::world->rand() % 2001) - 1000) / 1000.0;
      if constexpr (std::is_same_v<T, std::complex<double>>) {
        double im = (int(GlobalFixture::world->rand() % 2001) - 1000) / 1000.0;
        tile[i] = T(re, im);
      } else {
        tile[i] = re;
      }
    }
    a.set(idx, tile);
  }
  GlobalFixture::world->gop.fence();
  return a;
}

// max-abs difference between two same-shaped arrays
template <typename Array>
double max_abs_diff(const Array& x, const Array& y) {
  using T = typename Array::element_type;
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

struct OpsFixture {
  TiledRange tr2{{0, 2, 5}, {0, 2, 3}};       // (i, j) 5x3
  TiledRange tr2T{{0, 2, 3}, {0, 2, 5}};      // (j, i) 3x5
  TiledRange trSq{{0, 2, 4}, {0, 2, 4}};      // square for trace
};

}  // namespace

BOOST_FIXTURE_TEST_SUITE(ad_ops_suite, OpsFixture)

BOOST_AUTO_TEST_CASE(subt_value) {
  RArray a = rand_array<RArray>(tr2), b = rand_array<RArray>(tr2);
  RArray c = ad::subt(a, b);
  RArray ref;
  ref("i,j") = a("i,j") - b("i,j");
  BOOST_CHECK_SMALL(max_abs_diff(c, ref), 1e-14);
}

BOOST_AUTO_TEST_CASE(permute_value) {
  RArray a = rand_array<RArray>(tr2);
  RArray c = ad::permute(a, "i,j", "j,i");
  BOOST_CHECK(c.trange() == tr2T);
  RArray ref;
  ref("j,i") = a("i,j");
  BOOST_CHECK_SMALL(max_abs_diff(c, ref), 1e-14);
}

BOOST_AUTO_TEST_CASE(hadamard_mult_value) {
  RArray a = rand_array<RArray>(tr2), b = rand_array<RArray>(tr2);
  RArray c = ad::mult(a, b);
  RArray ref;
  ref("i,j") = a("i,j") * b("i,j");
  BOOST_CHECK_SMALL(max_abs_diff(c, ref), 1e-14);
}

BOOST_AUTO_TEST_CASE(conj_value) {
  CArray a = rand_array<CArray>(tr2);
  CArray c = ad::conj(a);
  CArray ref;
  ref("i,j") = a("i,j").conj();
  BOOST_CHECK_SMALL(max_abs_diff(c, ref), 1e-14);
}

BOOST_AUTO_TEST_CASE(elementwise_value) {
  RArray a = rand_array<RArray>(tr2);
  // f(x) = x^2, f'(x) = 2x
  RArray c = ad::elementwise(
      a, [](double x) { return x * x; }, [](double x) { return 2 * x; });
  RArray ref;
  ref("i,j") = a("i,j") * a("i,j");
  BOOST_CHECK_SMALL(max_abs_diff(c, ref), 1e-14);
}

BOOST_AUTO_TEST_CASE(trace_value) {
  RArray a = rand_array<RArray>(trSq);
  double t = ad::trace(a);
  double ref = a("i,j").trace().get();
  BOOST_CHECK_CLOSE(t, ref, 1e-12);
}

BOOST_AUTO_TEST_CASE(sum_value) {
  RArray a = rand_array<RArray>(tr2);
  double s = ad::sum(a);
  double ref = a("i,j").sum().get();
  BOOST_CHECK_CLOSE(s, ref, 1e-12);
}

BOOST_AUTO_TEST_CASE(squared_norm_value) {
  RArray a = rand_array<RArray>(tr2);
  double s = ad::squared_norm(a);
  double ref = a("i,j").squared_norm().get();
  BOOST_CHECK_CLOSE(s, ref, 1e-12);
}

BOOST_AUTO_TEST_CASE(norm2_value) {
  RArray a = rand_array<RArray>(tr2);
  double s = ad::norm2(a);
  double ref = std::sqrt(a("i,j").squared_norm().get());
  BOOST_CHECK_CLOSE(s, ref, 1e-12);
}

BOOST_AUTO_TEST_CASE(dot_is_bilinear_unconjugated) {
  CArray a = rand_array<CArray>(tr2), b = rand_array<CArray>(tr2);
  std::complex<double> s = ad::dot(a, b);
  std::complex<double> ref = a("i,j").dot(b("i,j")).get();
  BOOST_CHECK_SMALL(std::abs(s - ref), 1e-12);
}

BOOST_AUTO_TEST_CASE(inner_product_is_sesquilinear) {
  CArray a = rand_array<CArray>(tr2), b = rand_array<CArray>(tr2);
  std::complex<double> s = ad::inner_product(a, b);
  std::complex<double> ref = a("i,j").inner_product(b("i,j")).get();
  BOOST_CHECK_SMALL(std::abs(s - ref), 1e-12);
}

BOOST_AUTO_TEST_SUITE_END()
