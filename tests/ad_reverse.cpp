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
 *  ad_reverse.cpp
 *  Phase-1: reverse-mode (Tape / VJP) correctness over the primitive set,
 *  activity tracking, the frozen-operand guard, and complex conjugate handling.
 */

#include "tiledarray.h"
#include "unit_test_config.h"

#include "TiledArray/ad/ops.h"
#include "TiledArray/ad/tape.h"

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

struct RevFixture {
  TiledRange trA{{0, 2, 5}, {0, 3, 4}};  // (i, k)
  TiledRange trB{{0, 3, 4}, {0, 2}};     // (k, j)
  TiledRange tr2{{0, 2, 5}, {0, 2, 3}};  // (i, j) 5x3
  TiledRange trSq{{0, 2, 4}, {0, 2, 4}};
  std::string aA = "i,k", aB = "k,j", aC = "i,j";
};

}  // namespace

BOOST_FIXTURE_TEST_SUITE(ad_reverse_suite, RevFixture)

// add: gradient of <Cbar, A+B> is Cbar w.r.t. both operands.
BOOST_AUTO_TEST_CASE(add_vjp) {
  RArray A = rand_array<RArray>(tr2), B = rand_array<RArray>(tr2);
  RArray Cbar = rand_array<RArray>(tr2);
  ad::Tape<RArray> tape;
  auto va = ad::make_leaf(tape, A), vb = ad::make_leaf(tape, B);
  auto vc = ad::add(va, vb);
  tape.backward(vc.id, Cbar);
  // d<Cbar, A+B>/dA = Cbar
  BOOST_CHECK_CLOSE(tape.adjoint(va.id).value()("i,j").dot(A("i,j")).get(),
                    Cbar("i,j").dot(A("i,j")).get(), 1e-9);
  // adjoint equals Cbar exactly
  RArray d;
  d("i,j") = tape.adjoint(vb.id).value()("i,j") - Cbar("i,j");
  BOOST_CHECK_SMALL(std::sqrt(d("i,j").squared_norm().get()), 1e-13);
}

// subt: B's adjoint is -Cbar.
BOOST_AUTO_TEST_CASE(subt_vjp) {
  RArray A = rand_array<RArray>(tr2), B = rand_array<RArray>(tr2);
  RArray Cbar = rand_array<RArray>(tr2);
  ad::Tape<RArray> tape;
  auto va = ad::make_leaf(tape, A), vb = ad::make_leaf(tape, B);
  auto vc = ad::subt(va, vb);
  tape.backward(vc.id, Cbar);
  RArray da, db;
  da("i,j") = tape.adjoint(va.id).value()("i,j") - Cbar("i,j");
  db("i,j") = tape.adjoint(vb.id).value()("i,j") + Cbar("i,j");
  BOOST_CHECK_SMALL(std::sqrt(da("i,j").squared_norm().get()), 1e-13);
  BOOST_CHECK_SMALL(std::sqrt(db("i,j").squared_norm().get()), 1e-13);
}

// The adjoint-consistency identity Re<Cbar, JVP(dX)> == Re<VJP(Cbar), dX>
// applied to the whole chain  C = scale(permute(mult(A,B)))  exercises several
// rules at once with exact (linear-in-direction) arithmetic.
BOOST_AUTO_TEST_CASE(mult_permute_scale_adjoint_consistency) {
  RArray A = rand_array<RArray>(tr2), B = rand_array<RArray>(tr2);
  RArray dA = rand_array<RArray>(tr2), dB = rand_array<RArray>(tr2);
  // result of permute("i,j"->"j,i") has transposed range
  TiledRange tr2T{{0, 2, 3}, {0, 2, 5}};
  RArray Cbar = rand_array<RArray>(tr2T);
  const double alpha = 0.75;

  ad::Tape<RArray> tape;
  auto va = ad::make_leaf(tape, A), vb = ad::make_leaf(tape, B);
  auto vh = ad::mult(va, vb);
  auto vp = ad::permute(vh, "i,j", "j,i");
  auto vc = ad::scale(vp, alpha);
  tape.backward(vc.id, Cbar);
  const RArray& Abar = tape.adjoint(va.id).value();
  const RArray& Bbar = tape.adjoint(vb.id).value();

  // forward JVP of the same chain
  RArray h, p, jvp;
  h("i,j") = dA("i,j") * B("i,j") + A("i,j") * dB("i,j");
  jvp("j,i") = alpha * h("i,j");
  const double lhs = Cbar("j,i").dot(jvp("j,i")).get();
  const double rhs =
      Abar("i,j").dot(dA("i,j")).get() + Bbar("i,j").dot(dB("i,j")).get();
  BOOST_CHECK_CLOSE(lhs, rhs, 1e-9);
}

// scalar reductions as output: gradient vs central finite differences.
BOOST_AUTO_TEST_CASE(reduction_vjp_vs_finite_difference) {
  RArray A = rand_array<RArray>(trSq), dA = rand_array<RArray>(trSq);
  const double eps = 1e-4;

  auto grad_dir = [&](auto record) {
    ad::Tape<RArray> tape;
    auto va = ad::make_leaf(tape, A);
    auto s = record(tape, va);
    tape.backward(s.id);  // seed s̄ = 1
    const RArray& g = tape.adjoint(va.id).value();
    return g("i,j").dot(dA("i,j")).get();
  };
  auto fd = [&](auto op) {
    RArray Ap, Am;
    Ap("i,j") = A("i,j") + eps * dA("i,j");
    Am("i,j") = A("i,j") - eps * dA("i,j");
    return (op(Ap) - op(Am)) / (2 * eps);
  };

  BOOST_CHECK_CLOSE(grad_dir([](ad::Tape<RArray>& t, ad::Var<RArray>& v) {
                      return ad::sum(v);
                    }),
                    fd([](const RArray& x) { return ad::sum(x); }), 1e-5);
  BOOST_CHECK_CLOSE(grad_dir([](ad::Tape<RArray>& t, ad::Var<RArray>& v) {
                      return ad::squared_norm(v);
                    }),
                    fd([](const RArray& x) { return ad::squared_norm(x); }),
                    1e-4);
  BOOST_CHECK_CLOSE(grad_dir([](ad::Tape<RArray>& t, ad::Var<RArray>& v) {
                      return ad::norm2(v);
                    }),
                    fd([](const RArray& x) { return ad::norm2(x); }), 1e-4);
  BOOST_CHECK_CLOSE(grad_dir([](ad::Tape<RArray>& t, ad::Var<RArray>& v) {
                      return ad::trace(v);
                    }),
                    fd([](const RArray& x) { return ad::trace(x); }), 1e-5);
}

// elementwise f(x)=x^3 reduced by sum: d/dA sum(A^3) = 3 A^2, dotted with dA.
BOOST_AUTO_TEST_CASE(elementwise_vjp) {
  RArray A = rand_array<RArray>(tr2), dA = rand_array<RArray>(tr2);
  auto f = [](double x) { return x * x * x; };
  auto df = [](double x) { return 3 * x * x; };

  ad::Tape<RArray> tape;
  auto va = ad::make_leaf(tape, A);
  auto vy = ad::elementwise(va, f, df);
  auto vs = ad::sum(vy);
  tape.backward(vs.id);
  const RArray& g = tape.adjoint(va.id).value();

  RArray ref;  // 3 A^2
  ref("i,j") = 3.0 * A("i,j") * A("i,j");
  BOOST_CHECK_CLOSE(g("i,j").dot(dA("i,j")).get(),
                    ref("i,j").dot(dA("i,j")).get(), 1e-9);
}

// Complex: dot is bilinear/unconjugated, inner_product is sesquilinear.
// Verify both VJPs against the analytic conjugate structure of Part A.
BOOST_AUTO_TEST_CASE(complex_dot_and_inner_product_vjp) {
  CArray A = rand_array<CArray>(tr2), B = rand_array<CArray>(tr2);
  const std::complex<double> sbar(0.3, -0.7);

  {  // dot: Ā += s̄·conj(B)
    ad::Tape<CArray> tape;
    auto va = ad::make_leaf(tape, A), vb = ad::make_leaf(tape, B);
    auto vs = ad::dot(va, vb);
    tape.backward(vs.id, sbar);
    CArray refA;
    refA("i,j") = sbar * B("i,j").conj();
    CArray d;
    d("i,j") = tape.adjoint(va.id).value()("i,j") - refA("i,j");
    BOOST_CHECK_SMALL(std::sqrt(std::abs(d("i,j").squared_norm().get())),
                      1e-12);
  }
  {  // inner_product: Ā += conj(s̄)·B ; B̄ += s̄·A
    ad::Tape<CArray> tape;
    auto va = ad::make_leaf(tape, A), vb = ad::make_leaf(tape, B);
    auto vs = ad::inner_product(va, vb);
    tape.backward(vs.id, sbar);
    CArray refA, refB;
    refA("i,j") = std::conj(sbar) * B("i,j");
    refB("i,j") = sbar * A("i,j");
    CArray da, db;
    da("i,j") = tape.adjoint(va.id).value()("i,j") - refA("i,j");
    db("i,j") = tape.adjoint(vb.id).value()("i,j") - refB("i,j");
    BOOST_CHECK_SMALL(std::sqrt(std::abs(da("i,j").squared_norm().get())),
                      1e-12);
    BOOST_CHECK_SMALL(std::sqrt(std::abs(db("i,j").squared_norm().get())),
                      1e-12);
  }
}

// Complex finite-difference check for the non-holomorphic reductions
// (squared_norm, norm2) — background §6.2 flags these as the dangerous ones,
// where a missing conjugate in the VJP fails *silently* (stays finite, no NaN).
// Both are real-valued, so under the Part A "plus" pairing the gradient Ā
// satisfies the real directional-derivative identity
//   d/dε f(A + ε·dA)|₀ = Re⟨Ā, dA⟩,   ⟨x,y⟩ = Σ conj(x)·y,
// for a complex perturbation dA. The analytic Re(inner_product(Ā, dA)) must
// therefore match a central finite difference of f along dA.
BOOST_AUTO_TEST_CASE(complex_reduction_vjp_vs_finite_difference) {
  CArray A = rand_array<CArray>(trSq), dA = rand_array<CArray>(trSq);
  const double eps = 1e-4;

  auto grad_dir = [&](auto record) {
    ad::Tape<CArray> tape;
    auto va = ad::make_leaf(tape, A);
    auto s = record(tape, va);
    tape.backward(s.id);  // seed s̄ = 1 (real)
    const CArray& g = tape.adjoint(va.id).value();
    return std::real(ad::inner_product(g, dA));  // Re⟨Ā, dA⟩
  };
  auto fd = [&](auto op) {
    CArray Ap, Am;
    Ap("i,j") = A("i,j") + eps * dA("i,j");
    Am("i,j") = A("i,j") - eps * dA("i,j");
    return (op(Ap) - op(Am)) / (2 * eps);
  };

  BOOST_CHECK_CLOSE(grad_dir([](ad::Tape<CArray>&, ad::Var<CArray>& v) {
                      return ad::squared_norm(v);
                    }),
                    fd([](const CArray& x) { return ad::squared_norm(x); }),
                    1e-4);
  BOOST_CHECK_CLOSE(grad_dir([](ad::Tape<CArray>&, ad::Var<CArray>& v) {
                      return ad::norm2(v);
                    }),
                    fd([](const CArray& x) { return ad::norm2(x); }), 1e-4);
}

// Convention litmus (Krämer §6.3, called out as *mandatory* in the plan's
// Verification section): the gradient of f(z)=½z² at z=1+i must be 1−i under
// the "plus"/conjugating convention fixed in Part A. A result of 1+i would mean
// the opposite "minus"/JAX convention had crept in. The mismatch stays finite
// (no NaN) and is otherwise silent, so this assertion is the decisive guard.
//
// Mechanism: y = Σ ½z² seeded with s̄ = 1 broadcasts C̄ = 1 onto the
// elementwise op, whose VJP Ā = conj(f'(A))∘C̄ = conj(z) yields exactly the
// convention's gradient.
BOOST_AUTO_TEST_CASE(complex_half_z_squared_convention_litmus) {
  CArray z(*GlobalFixture::world, TiledRange{{0, 1}, {0, 1}});
  for (auto idx : *z.pmap()) {
    auto range = z.trange().make_tile_range(idx);
    CArray::value_type tile(range, std::complex<double>(1.0, 1.0));
    z.set(idx, tile);
  }
  z.world().gop.fence();

  auto f = [](std::complex<double> v) { return 0.5 * v * v; };
  auto df = [](std::complex<double> v) { return v; };

  ad::Tape<CArray> tape;
  auto vz = ad::make_leaf(tape, z);
  auto vy = ad::sum(ad::elementwise(vz, f, df));  // y = Σ ½z²
  tape.backward(vy.id);                            // seed s̄ = 1

  const std::complex<double> g = tape.adjoint(vz.id).value().find(0).get()[0];
  BOOST_CHECK_CLOSE(g.real(), 1.0, 1e-12);
  BOOST_CHECK_CLOSE(g.imag(), -1.0, 1e-12);
}

// Fan-out: A is consumed by two ops, so reverse must SUM the two cotangents.
// y = sum( (A∘P) + (A∘Q) )  =>  dy/dA = P + Q.
BOOST_AUTO_TEST_CASE(fan_out_sums_cotangents) {
  RArray A = rand_array<RArray>(tr2);
  RArray P = rand_array<RArray>(tr2), Q = rand_array<RArray>(tr2);
  ad::Tape<RArray> tape;
  auto va = ad::make_leaf(tape, A);
  auto vp = ad::make_leaf(tape, P, /*requires_grad=*/false);
  auto vq = ad::make_leaf(tape, Q, /*requires_grad=*/false);
  auto vsum = ad::add(ad::mult(va, vp), ad::mult(va, vq));
  auto vy = ad::sum(vsum);
  tape.backward(vy.id);
  const RArray& g = tape.adjoint(va.id).value();
  RArray ref;
  ref("i,j") = P("i,j") + Q("i,j");
  RArray d;
  d("i,j") = g("i,j") - ref("i,j");
  BOOST_CHECK_SMALL(std::sqrt(d("i,j").squared_norm().get()), 1e-12);
}

// Activity tracking: a constant (requires_grad=false) operand gets no shadow.
BOOST_AUTO_TEST_CASE(inactive_operand_has_no_gradient) {
  RArray A = rand_array<RArray>(trA), B = rand_array<RArray>(trB);
  RArray Cbar = rand_array<RArray>(TiledRange{{0, 2, 5}, {0, 2}});
  ad::Tape<RArray> tape;
  auto va = ad::make_leaf(tape, A, /*requires_grad=*/true);
  auto vb = ad::make_leaf(tape, B, /*requires_grad=*/false);  // constant
  auto vc = ad::contract(va, vb, aA, aB, aC);
  BOOST_CHECK(vc.active);
  tape.backward(vc.id, Cbar);
  BOOST_CHECK(!tape.adjoint(va.id).is_zero());
  BOOST_CHECK(!vb.active);
}

// Frozen-operand guard: mutating a recorded operand before backward() trips
// the guard rather than silently returning a wrong gradient.
BOOST_AUTO_TEST_CASE(frozen_operand_guard_trips_on_mutation) {
  RArray A = rand_array<RArray>(trA), B = rand_array<RArray>(trB);
  RArray Cbar = rand_array<RArray>(TiledRange{{0, 2, 5}, {0, 2}});
  ad::Tape<RArray> tape;
  auto va = ad::make_leaf(tape, A), vb = ad::make_leaf(tape, B);
  auto vc = ad::contract(va, vb, aA, aB, aC);  // saves A, B as residuals

  // mutate A in place (visible through the shared impl the tape captured)
  for (auto idx : *A.pmap()) {
    if (A.is_zero(idx)) continue;
    auto tile = A.find(idx).get();
    auto scaled = tile.scale(2.0);
    A.set(idx, scaled);
    break;
  }
  A.world().gop.fence();

  BOOST_CHECK_THROW(tape.backward(vc.id, Cbar), TiledArray::Exception);
}

BOOST_AUTO_TEST_SUITE_END()
