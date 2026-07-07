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
 *  ad_eig.cpp
 *  Phase-4: differentiable symmetric/Hermitian eigendecomposition (ad::heig) —
 *  functional primitive, JVP/VJP kernels, forward (Dual) and reverse (Var)
 *  rules, degeneracy policy, and forward-over-reverse composition.
 */

#include "tiledarray.h"
#include "unit_test_config.h"

#include "TiledArray/ad/heig.h"
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

// random Hermitian matrix: X + Xᴴ (guarantees the heig input precondition)
template <typename Array>
Array rand_hermitian(const TiledRange& tr) {
  Array X = rand_array<Array>(tr);
  Array A;
  A("i,j") = X("i,j") + X("j,i").conj();
  return A;
}

// max |x - y| over all elements (local tiles; every rank checks its subset)
template <typename Array>
double max_abs_diff(const Array& x, const Array& y) {
  Array d;
  const std::string annot =
      TiledArray::ad::detail::canonical_annotation(x.trange().rank());
  d(annot) = x(annot) - y(annot);
  double m = 0.0;
  for (auto idx : *d.pmap()) {
    if (d.is_zero(idx)) continue;
    auto tile = d.find(idx).get();
    for (std::size_t i = 0; i < tile.size(); ++i)
      m = std::max(m, std::abs(tile[i]));
  }
  return m;
}

// Frobenius norm of x - y (a collective observation, same on every rank)
template <typename Array>
double fro_diff(const Array& x, const Array& y) {
  Array d;
  const std::string annot =
      TiledArray::ad::detail::canonical_annotation(x.trange().rank());
  d(annot) = x(annot) - y(annot);
  return std::sqrt(std::abs(d(annot).squared_norm().get()));
}

struct EigFixture {
  TiledRange trSq{{0, 2, 4}, {0, 2, 4}};  // 4x4, 2x2 tiles
  TiledRange trV{{0, 2, 4}};              // matching rank-1 tiling
};

}  // namespace

BOOST_FIXTURE_TEST_SUITE(ad_eig_suite, EigFixture)

BOOST_AUTO_TEST_CASE(heig_primal) {
  RArray A = rand_hermitian<RArray>(trSq);
  auto r = ad::heig(A);

  // evals array matches the linalg::heig eigenvalue vector elementwise
  auto [evals_vec, U_ref] = TiledArray::math::linalg::heig(A);
  for (auto idx : *r.evals.pmap()) {
    if (r.evals.is_zero(idx)) continue;
    auto tile = r.evals.find(idx).get();
    for (auto i : tile.range())
      BOOST_CHECK_SMALL(tile(i) - evals_vec[i[0]], 1e-10);
  }

  // reconstruction: ‖U diag(λ) Uᵀ − A‖ small
  RArray Lam(A.world(), trSq);
  Lam.init_elements([v = evals_vec](const auto& idx) {
    return idx[0] == idx[1] ? v[idx[0]] : 0.0;
  });
  RArray R;
  R("i,j") = r.evecs("i,k") * Lam("k,l") * r.evecs("j,l");
  BOOST_CHECK_SMALL(fro_diff(R, A), 1e-10);

  // orthonormality: ‖UᵀU − I‖ small
  RArray G;
  G("i,j") = r.evecs("k,i") * r.evecs("k,j");
  RArray I(A.world(), trSq);
  I.init_elements(
      [](const auto& idx) { return idx[0] == idx[1] ? 1.0 : 0.0; });
  BOOST_CHECK_SMALL(fro_diff(G, I), 1e-10);

  BOOST_CHECK(r.min_gap > 0);
}

BOOST_AUTO_TEST_CASE(degeneracy_scan) {
  // diag(1,1,2,3) → exactly degenerate → min_gap == 0
  RArray D(*GlobalFixture::world, trSq);
  const double dvals[4] = {1.0, 1.0, 2.0, 3.0};
  D.init_elements([&dvals](const auto& idx) {
    return idx[0] == idx[1] ? dvals[idx[0]] : 0.0;
  });
  auto r = ad::heig(D);
  BOOST_CHECK_SMALL(r.min_gap, 1e-12);

  // the policy itself: error throws at zero gap, broaden does not
  BOOST_CHECK_THROW(
      ad::detail::enforce_degeneracy_policy(ad::HeigDiffPolicy{}, 0.0),
      TiledArray::Exception);
  ad::HeigDiffPolicy broaden{ad::HeigDiffPolicy::Degeneracy::broaden};
  BOOST_CHECK_NO_THROW(ad::detail::enforce_degeneracy_policy(broaden, 0.0));
  // a healthy gap passes under the default (error) policy
  BOOST_CHECK_NO_THROW(
      ad::detail::enforce_degeneracy_policy(ad::HeigDiffPolicy{}, 0.5));
}

BOOST_AUTO_TEST_CASE(structure_helpers) {
  RArray v = rand_array<RArray>(trV);
  RArray M = rand_array<RArray>(trSq);

  // ones_like: rank-1, all elements 1
  RArray ones = ad::detail::ones_like(v);
  for (auto idx : *ones.pmap()) {
    if (ones.is_zero(idx)) continue;
    auto tile = ones.find(idx).get();
    for (std::size_t i = 0; i < tile.size(); ++i)
      BOOST_CHECK_SMALL(tile[i] - 1.0, 1e-15);
  }

  // identity_like: rank-2, δ_ij
  RArray I = ad::detail::identity_like(M);
  for (auto idx : *I.pmap()) {
    if (I.is_zero(idx)) continue;
    auto tile = I.find(idx).get();
    for (auto e : tile.range())
      BOOST_CHECK_SMALL(tile(e) - (e[0] == e[1] ? 1.0 : 0.0), 1e-15);
  }

  // diag embed/extract round-trip in the primitive set: v == extract(embed(v))
  RArray embedded = ad::mult(ad::contract(v, ones, "i", "j", "i,j"), I);
  RArray back = ad::contract(ad::mult(embedded, I), ones, "i,j", "j", "i");
  BOOST_CHECK_SMALL(max_abs_diff(back, v), 1e-13);
}

BOOST_AUTO_TEST_CASE(f_matrix_values) {
  // evals = [1, 2, 4] as a single-tile rank-1 array
  const TiledRange trE{{0, 3}};
  const double vals[3] = {1.0, 2.0, 4.0};
  RArray evals(*GlobalFixture::world, trE);
  evals.init_elements([&vals](const auto& idx) { return vals[idx[0]]; });

  auto check_f = [&](const RArray& F, double tol) {
    for (auto idx : *F.pmap()) {
      if (F.is_zero(idx)) continue;
      auto tile = F.find(idx).get();
      for (auto e : tile.range()) {
        const double expected =
            e[0] == e[1] ? 0.0 : 1.0 / (vals[e[1]] - vals[e[0]]);
        BOOST_CHECK_SMALL(tile(e) - expected, tol);
      }
    }
  };

  // error policy: exact reciprocal gaps, antisymmetric, zero diagonal
  check_f(ad::detail::make_f_matrix(evals, ad::HeigDiffPolicy{}), 1e-14);

  // broaden (default ε = 1e-12) on a well-gapped spectrum: same within 1e-9
  ad::HeigDiffPolicy broaden{ad::HeigDiffPolicy::Degeneracy::broaden};
  check_f(ad::detail::make_f_matrix(evals, broaden), 1e-9);
}

BOOST_AUTO_TEST_CASE(jvp_vjp_kernel_adjoint_consistency) {
  // the defining adjoint identity, exact linear algebra (no FD):
  //   Re⟨λ̄, dλ⟩ + Re⟨Ū, dU⟩ == Re⟨Ā, dA⟩
  // for Hermitian dA (the input is constrained Hermitian, so Ā is
  // Hermitian-projected and only pairs correctly against Hermitian dA)
  auto run = [&](auto tag) {
    using Array = decltype(tag);
    Array A = rand_hermitian<Array>(trSq);
    auto r = ad::heig(A);
    Array dA = rand_hermitian<Array>(trSq);
    Array lbar = rand_array<Array>(trV);
    Array ubar = rand_array<Array>(trSq);

    auto [dlam, dU] = ad::detail::heig_jvp(r.evals, r.evecs, dA,
                                           ad::HeigDiffPolicy{}, r.min_gap);
    Array abar = ad::detail::heig_vjp(r.evals, r.evecs, &lbar, &ubar,
                                      ad::HeigDiffPolicy{}, r.min_gap);

    const double lhs = std::real(ad::inner_product(lbar, dlam)) +
                       std::real(ad::inner_product(ubar, dU));
    const double rhs = std::real(ad::inner_product(abar, dA));
    BOOST_CHECK_SMALL(lhs - rhs, 1e-9);
  };
  run(RArray{});
  run(CArray{});
}

BOOST_AUTO_TEST_SUITE_END()
