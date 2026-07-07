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

// f(A) = Σ_i w_i λ_i as a plain-array function (for FD reference values)
double weighted_evals(const RArray& A, const RArray& w) {
  auto r = ad::heig(A);
  return ad::dot(w, r.evals);
}

// g(A) = ⟨B, U W Uᵀ⟩ with W = diag(w) — gauge-invariant in U (each column
// enters as w_i u_i u_iᵀ, immune to sign flips)
double evec_functional(const RArray& A, const RArray& B, const RArray& Wd) {
  auto r = ad::heig(A);
  RArray T1 = ad::contract(r.evecs, Wd, "i,k", "k,l", "i,l");
  RArray T2 = ad::contract(T1, r.evecs, "i,l", "j,l", "i,j");  // U W Uᵀ
  return ad::dot(B, T2);
}

BOOST_AUTO_TEST_CASE(forward_evals_fd) {
  RArray A = rand_hermitian<RArray>(trSq);
  RArray dA = rand_hermitian<RArray>(trSq);
  RArray w = rand_array<RArray>(trV);

  auto r = ad::heig(ad::make_dual(A, dA));
  BOOST_REQUIRE(r.evals.has_tangent());
  const double jvp = ad::dot(ad::make_constant(w), r.evals).tangent;

  const double h = 1e-5;
  const double fd = (weighted_evals(ad::add(A, ad::scale(dA, h)), w) -
                     weighted_evals(ad::subt(A, ad::scale(dA, h)), w)) /
                    (2 * h);
  BOOST_CHECK_CLOSE(jvp, fd, 1e-3);  // percent → rel 1e-5
}

BOOST_AUTO_TEST_CASE(forward_evecs_fd) {
  RArray A = rand_hermitian<RArray>(trSq);
  RArray dA = rand_hermitian<RArray>(trSq);
  RArray B = rand_hermitian<RArray>(trSq);
  // W = diag(w) with distinct weights
  RArray Wd(*GlobalFixture::world, trSq);
  Wd.init_elements([](const auto& idx) {
    return idx[0] == idx[1] ? 1.0 + 0.5 * idx[0] : 0.0;
  });

  auto r = ad::heig(ad::make_dual(A, dA));
  BOOST_REQUIRE(r.evecs.has_tangent());
  auto Bc = ad::make_constant(B);
  auto Wc = ad::make_constant(Wd);
  auto T1 = ad::contract(r.evecs, Wc, "i,k", "k,l", "i,l");
  auto T2 = ad::contract(T1, r.evecs, "i,l", "j,l", "i,j");
  const double jvp = ad::dot(Bc, T2).tangent;

  const double h = 1e-5;
  const double fd = (evec_functional(ad::add(A, ad::scale(dA, h)), B, Wd) -
                     evec_functional(ad::subt(A, ad::scale(dA, h)), B, Wd)) /
                    (2 * h);
  BOOST_CHECK_CLOSE(jvp, fd, 1e-3);
}

BOOST_AUTO_TEST_CASE(forward_degenerate_policy) {
  // exactly degenerate spectrum {1, 1, 2, 3}; forward mode is eager, so an
  // active tangent must trip the error policy at the call
  RArray D(*GlobalFixture::world, trSq);
  const double dvals[4] = {1.0, 1.0, 2.0, 3.0};
  D.init_elements([&dvals](const auto& idx) {
    return idx[0] == idx[1] ? dvals[idx[0]] : 0.0;
  });
  RArray dA = rand_hermitian<RArray>(trSq);

  BOOST_CHECK_THROW(ad::heig(ad::make_dual(D, dA)), TiledArray::Exception);

  ad::HeigDiffPolicy broaden{ad::HeigDiffPolicy::Degeneracy::broaden};
  auto r = ad::heig(ad::make_dual(D, dA), broaden);
  BOOST_REQUIRE(r.evecs.has_tangent());
  BOOST_CHECK(std::isfinite(ad::norm2(*r.evecs.tangent)));
  BOOST_CHECK(std::isfinite(ad::norm2(*r.evals.tangent)));
}

BOOST_AUTO_TEST_CASE(forward_constant_input) {
  RArray A = rand_hermitian<RArray>(trSq);
  auto r = ad::heig(ad::make_constant(A));
  BOOST_CHECK(!r.evals.has_tangent());
  BOOST_CHECK(!r.evecs.has_tangent());
  BOOST_CHECK(r.min_gap > 0);
}

BOOST_AUTO_TEST_CASE(reverse_fd_real) {
  // tape gradient of g(A) = ⟨B, U W Uᵀ⟩ + Σ_i w'_i λ_i vs central FD along
  // random Hermitian directions
  RArray A = rand_hermitian<RArray>(trSq);
  RArray B = rand_hermitian<RArray>(trSq);
  RArray wp = rand_array<RArray>(trV);
  RArray Wd(*GlobalFixture::world, trSq);
  Wd.init_elements([](const auto& idx) {
    return idx[0] == idx[1] ? 1.0 + 0.5 * idx[0] : 0.0;
  });

  ad::Tape<RArray> tape;
  auto va = ad::make_leaf(tape, A);
  auto r = ad::heig(va);
  auto vB = ad::make_leaf(tape, B, false);
  auto vW = ad::make_leaf(tape, Wd, false);
  auto vwp = ad::make_leaf(tape, wp, false);
  auto t1 = ad::contract(r.evecs, vW, "i,k", "k,l", "i,l");
  auto t2 = ad::contract(t1, r.evecs, "i,l", "j,l", "i,j");
  auto s1 = ad::dot(vB, t2);
  auto s2 = ad::dot(vwp, r.evals);
  BOOST_CHECK_CLOSE(s1.value + s2.value,
                    evec_functional(A, B, Wd) + weighted_evals(A, wp), 1e-9);
  tape.accumulate_scalar(s2.id, 1.0);  // seed the second output term
  tape.backward(s1.id, 1.0);
  const RArray& Abar = tape.adjoint(va.id).value();

  const double h = 1e-5;
  for (int k = 0; k < 3; ++k) {
    RArray dA = rand_hermitian<RArray>(trSq);
    RArray Ap = ad::add(A, ad::scale(dA, h)), Am = ad::subt(A, ad::scale(dA, h));
    const double fd = (evec_functional(Ap, B, Wd) + weighted_evals(Ap, wp) -
                       evec_functional(Am, B, Wd) - weighted_evals(Am, wp)) /
                      (2 * h);
    BOOST_CHECK_CLOSE(ad::dot(Abar, dA), fd, 1e-2);  // percent → rel 1e-4
  }
}

BOOST_AUTO_TEST_CASE(reverse_adjoint_consistency_complex) {
  // adjoint consistency through the Var layer against the Dual JVP —
  // catches missing conjugates in the recorded VJP
  CArray A = rand_hermitian<CArray>(trSq);
  CArray dA = rand_hermitian<CArray>(trSq);
  CArray lbar = rand_array<CArray>(trV);
  CArray ubar = rand_array<CArray>(trSq);

  ad::Tape<CArray> tape;
  auto va = ad::make_leaf(tape, A);
  auto r = ad::heig(va);
  tape.accumulate_array(r.evecs.id, ubar);  // seed Ū, then λ̄ via backward
  tape.backward(r.evals.id, lbar);
  const CArray& Abar = tape.adjoint(va.id).value();

  auto rd = ad::heig(ad::make_dual(A, dA));
  const double lhs = std::real(ad::inner_product(lbar, *rd.evals.tangent)) +
                     std::real(ad::inner_product(ubar, *rd.evecs.tangent));
  const double rhs = std::real(ad::inner_product(Abar, dA));
  BOOST_CHECK_SMALL(lhs - rhs, 1e-9);
}

BOOST_AUTO_TEST_CASE(reverse_hellmann_feynman_degenerate) {
  // f = Σ_i λ_i = tr A on an EXACTLY degenerate A: the gradient is the
  // identity (analytic) and the DEFAULT error policy must NOT throw — only
  // the eigenvalue cotangent is nonzero, so F is never built (lazy path)
  RArray D(*GlobalFixture::world, trSq);
  const double dvals[4] = {1.0, 1.0, 2.0, 3.0};
  D.init_elements([&dvals](const auto& idx) {
    return idx[0] == idx[1] ? dvals[idx[0]] : 0.0;
  });
  RArray w1(*GlobalFixture::world, trV);
  w1.init_elements([](const auto&) { return 1.0; });

  ad::Tape<RArray> tape;
  auto va = ad::make_leaf(tape, D);
  auto r = ad::heig(va);
  auto vw = ad::make_leaf(tape, w1, false);
  auto s = ad::dot(vw, r.evals);
  BOOST_CHECK_CLOSE(s.value, 7.0, 1e-8);  // tr D
  BOOST_CHECK_NO_THROW(tape.backward(s.id, 1.0));
  BOOST_CHECK_SMALL(
      fro_diff(tape.adjoint(va.id).value(), ad::detail::identity_like(D)),
      1e-10);
}

BOOST_AUTO_TEST_CASE(reverse_degenerate_error_and_broaden) {
  // exactly degenerate A with a nontrivial eigenbasis: A = Q diag(1,1,2,3) Qᵀ
  RArray Arand = rand_hermitian<RArray>(trSq);
  auto q = ad::heig(Arand);
  RArray Lam(*GlobalFixture::world, trSq);
  const double dvals[4] = {1.0, 1.0, 2.0, 3.0};
  Lam.init_elements([&dvals](const auto& idx) {
    return idx[0] == idx[1] ? dvals[idx[0]] : 0.0;
  });
  RArray Adeg;
  Adeg("i,j") = q.evecs("i,k") * Lam("k,l") * q.evecs("j,l");

  RArray B = rand_hermitian<RArray>(trSq);
  // weights EQUAL on the degenerate pair (ascending → columns 0, 1), so the
  // functional is invariant under rotations inside the degenerate subspace
  RArray Wd(*GlobalFixture::world, trSq);
  const double wvals[4] = {2.0, 2.0, -1.0, 3.0};
  Wd.init_elements([&wvals](const auto& idx) {
    return idx[0] == idx[1] ? wvals[idx[0]] : 0.0;
  });

  auto grad = [&](const ad::HeigDiffPolicy& pol) {
    ad::Tape<RArray> tape;
    auto va = ad::make_leaf(tape, Adeg);
    auto r = ad::heig(va, pol);
    auto vB = ad::make_leaf(tape, B, false);
    auto vW = ad::make_leaf(tape, Wd, false);
    auto t1 = ad::contract(r.evecs, vW, "i,k", "k,l", "i,l");
    auto t2 = ad::contract(t1, r.evecs, "i,l", "j,l", "i,j");
    auto s = ad::dot(vB, t2);
    tape.backward(s.id, 1.0);  // error policy throws here (lazy F)
    return RArray(tape.adjoint(va.id).value());
  };

  // evec cotangent flows → the error policy throws at backward()
  BOOST_CHECK_THROW(grad(ad::HeigDiffPolicy{}), TiledArray::Exception);

  // broaden: finite gradient matching FD (the functional is smooth at the
  // degeneracy thanks to the equal weights)
  ad::HeigDiffPolicy br{ad::HeigDiffPolicy::Degeneracy::broaden};
  RArray Abar = grad(br);
  const double h = 1e-4;
  for (int k = 0; k < 3; ++k) {
    RArray dA = rand_hermitian<RArray>(trSq);
    const double fd =
        (evec_functional(ad::add(Adeg, ad::scale(dA, h)), B, Wd) -
         evec_functional(ad::subt(Adeg, ad::scale(dA, h)), B, Wd)) /
        (2 * h);
    BOOST_CHECK_CLOSE(ad::dot(Abar, dA), fd, 0.1);  // percent → rel 1e-3
  }
}

BOOST_AUTO_TEST_CASE(reverse_activity) {
  RArray A = rand_hermitian<RArray>(trSq);
  RArray w = rand_array<RArray>(trV);

  // inactive leaf → both outputs inactive, nothing recorded
  {
    ad::Tape<RArray> tape;
    auto va = ad::make_leaf(tape, A, false);
    auto r = ad::heig(va);
    BOOST_CHECK(!r.evals.active);
    BOOST_CHECK(!r.evecs.active);
  }

  // unconnected evecs output → its adjoint stays the symbolic zero
  {
    ad::Tape<RArray> tape;
    auto va = ad::make_leaf(tape, A);
    auto r = ad::heig(va);
    auto vw = ad::make_leaf(tape, w, false);
    auto s = ad::dot(vw, r.evals);
    tape.backward(s.id, 1.0);
    BOOST_CHECK(tape.adjoint(r.evecs.id).is_zero());
    BOOST_CHECK(!tape.adjoint(va.id).is_zero());
  }
}

BOOST_AUTO_TEST_CASE(hvp_through_heig) {
  // forward-over-reverse through heig: f(A) = Σ_i w_i λ_i (λ̄ = w seeds the
  // eigenvalue output), HVP H·V via Tape<Dual> vs central FD of the
  // first-order tape gradient. Exercises dF and the second-order eig terms —
  // the payoff of writing the kernels in B1 primitives (no new rule code).
  using DualR = ad::Dual<RArray>;
  RArray A = rand_hermitian<RArray>(trSq);
  RArray V = rand_hermitian<RArray>(trSq);
  RArray w = rand_array<RArray>(trV);

  auto grad_f = [&](const RArray& X) {
    ad::Tape<RArray> tape;
    auto vx = ad::make_leaf(tape, X);
    auto r = ad::heig(vx);
    tape.backward(r.evals.id, w);
    return RArray(tape.adjoint(vx.id).value());
  };

  ad::Tape<DualR> tape;
  auto va = ad::make_leaf(tape, ad::make_dual(A, V));
  auto r = ad::heig(va);
  tape.backward(r.evals.id, ad::make_constant(w));  // λ̄ = w (constant seed)
  const DualR& g = tape.adjoint(va.id).value();
  BOOST_REQUIRE(g.has_tangent());

  // primal of the dual gradient == the plain reverse-mode gradient
  BOOST_CHECK_SMALL(fro_diff(g.primal, grad_f(A)), 1e-11);

  // tangent == H·V vs FD of the gradient along V
  const double h = 1e-4;
  RArray gp = grad_f(ad::add(A, ad::scale(V, h)));
  RArray gm = grad_f(ad::subt(A, ad::scale(V, h)));
  RArray fd = ad::scale(ad::subt(gp, gm), 1.0 / (2 * h));
  BOOST_CHECK_SMALL(fro_diff(*g.tangent, fd),
                    1e-3 * std::sqrt(ad::squared_norm(fd)));
}

BOOST_AUTO_TEST_SUITE_END()
