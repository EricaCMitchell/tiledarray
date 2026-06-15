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
 *  produce.cpp
 *  Producer for the JAX dual-verification harness (ad_jax_dual_verification_plan.md).
 *
 *  Runs the native C++ AD layer (`src/TiledArray/ad/`) on deterministic inputs
 *  and dumps inputs, parameters, and results to `golden.json`. A companion
 *  `verify.py` recomputes the references in JAX and asserts agreement. This file
 *  is the *subject under test*; JAX is the independent oracle. Built on demand
 *  (`ninja ad_jax_produce`), never with `all` -- so it is immune to the
 *  unrelated `arena_tensor_kernels` break that blocks the monolithic `ta_test`.
 */

#include <complex>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <ostream>
#include <random>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "tiledarray.h"

#include "TiledArray/ad/dual.h"
#include "TiledArray/ad/ops.h"
#include "TiledArray/ad/tape.h"

using namespace TiledArray;

namespace {

using RArray = TA::TArray<double>;                 // dense real
using CArray = TA::TArray<std::complex<double>>;   // dense complex
using SpArray = TA::TSpArray<double>;              // sparse real

// ---------------------------------------------------------------------------
// Minimal JSON DOM + writer (no external dependency).
//
// Numbers are printed at std::setprecision(17) so float64/complex128 round-trip
// to Python exactly. Strings here are only annotations / fn tags / names, none
// of which contain quotes or backslashes, so no escaping is needed.
// ---------------------------------------------------------------------------
struct JsonValue;
using JsonObj = std::vector<std::pair<std::string, JsonValue>>;
using JsonArr = std::vector<JsonValue>;

struct JsonValue {
  std::variant<std::nullptr_t, double, std::string, JsonArr, JsonObj> v;
  JsonValue() : v(nullptr) {}
  JsonValue(double d) : v(d) {}
  JsonValue(int i) : v(static_cast<double>(i)) {}
  JsonValue(long i) : v(static_cast<double>(i)) {}
  JsonValue(std::size_t i) : v(static_cast<double>(i)) {}
  JsonValue(const char* s) : v(std::string(s)) {}
  JsonValue(std::string s) : v(std::move(s)) {}
  JsonValue(JsonArr a) : v(std::move(a)) {}
  JsonValue(JsonObj o) : v(std::move(o)) {}

  void dump(std::ostream& os) const {
    if (std::holds_alternative<std::nullptr_t>(v)) {
      os << "null";
    } else if (std::holds_alternative<double>(v)) {
      os << std::setprecision(17) << std::get<double>(v);
    } else if (std::holds_alternative<std::string>(v)) {
      os << '"' << std::get<std::string>(v) << '"';
    } else if (std::holds_alternative<JsonArr>(v)) {
      const auto& a = std::get<JsonArr>(v);
      os << '[';
      for (std::size_t i = 0; i < a.size(); ++i) {
        if (i) os << ',';
        a[i].dump(os);
      }
      os << ']';
    } else {
      const auto& o = std::get<JsonObj>(v);
      os << '{';
      for (std::size_t i = 0; i < o.size(); ++i) {
        if (i) os << ',';
        os << '"' << o[i].first << "\":";
        o[i].second.dump(os);
      }
      os << '}';
    }
  }
};

// ---------------------------------------------------------------------------
// Deterministic input generation.
//
// A fixed-seed mt19937_64 + uniform_real_distribution(-1, 1) drives all fills,
// in a fixed (tiles_range, then within-tile) traversal order, so a re-run
// reproduces byte-identical inputs. RNG is *not* shared with Python: the
// producer emits the actual values it used (see plan section 2).
// ---------------------------------------------------------------------------
std::mt19937_64 g_rng(0xA5A5C0FFEEULL);
std::uniform_real_distribution<double> g_dist(-1.0, 1.0);

template <typename T>
T draw() {
  if constexpr (std::is_same_v<T, std::complex<double>>) {
    const double re = g_dist(g_rng);
    const double im = g_dist(g_rng);
    return T(re, im);
  } else {
    return g_dist(g_rng);
  }
}

// Dense array filled in canonical tile order.
template <typename Array>
Array make_dense(World& world, const TiledRange& tr) {
  using T = typename Array::element_type;
  Array a(world, tr);
  for (auto tidx : a.trange().tiles_range()) {
    auto range = a.trange().make_tile_range(tidx);
    typename Array::value_type tile(range);
    for (std::size_t i = 0; i < tile.size(); ++i) tile[i] = draw<T>();
    a.set(tidx, tile);
  }
  world.gop.fence();
  return a;
}

// Block-sparse array: nonzero-tile pattern given by `nonzero` predicate.
template <typename Pred>
SpArray make_sparse(World& world, const TiledRange& tr, Pred nonzero) {
  Tensor<float> shape_tensor(tr.tiles_range(), 0.0f);
  for (auto it = tr.tiles_range().begin(); it != tr.tiles_range().end(); ++it)
    if (nonzero(*it)) shape_tensor[*it] = 1.0f;
  SpArray a(world, tr, TiledArray::SparseShape<float>(shape_tensor, tr));
  for (auto tidx : a.trange().tiles_range()) {
    if (a.is_zero(tidx)) continue;
    auto range = a.trange().make_tile_range(tidx);
    Tensor<double> tile(range);
    for (std::size_t i = 0; i < tile.size(); ++i) tile[i] = draw<double>();
    a.set(tidx, tile);
  }
  world.gop.fence();
  return a;
}

// ---------------------------------------------------------------------------
// Flatten a DistArray to row-major (plan section 4.1, `to_rowmajor`).
//
// Iterates elements_range() in row-major order, reading each element from its
// owning tile (all local at np=1). Sparse zero tiles contribute zeros. Complex
// data is interleaved [re, im, re, im, ...] with a "complex" flag.
// ---------------------------------------------------------------------------
template <typename Array>
JsonValue tensor_json(const Array& a) {
  using T = typename Array::element_type;
  constexpr bool is_complex = std::is_same_v<T, std::complex<double>>;
  const auto& er = a.trange().elements_range();

  JsonArr shape;
  for (auto e : er.extent()) shape.push_back(JsonValue(static_cast<long>(e)));

  JsonArr data;
  for (auto idx : er) {  // row-major
    const auto tidx = a.trange().element_to_tile(idx);
    if (a.is_zero(tidx)) {
      data.push_back(JsonValue(0.0));
      if constexpr (is_complex) data.push_back(JsonValue(0.0));
      continue;
    }
    auto tile = a.find(tidx).get();
    const T val = tile[tile.range().ordinal(idx)];
    if constexpr (is_complex) {
      data.push_back(JsonValue(val.real()));
      data.push_back(JsonValue(val.imag()));
    } else {
      data.push_back(JsonValue(static_cast<double>(val)));
    }
  }

  JsonObj o;
  o.emplace_back("shape", JsonValue(std::move(shape)));
  o.emplace_back("complex", JsonValue(is_complex ? 1 : 0));
  o.emplace_back("data", JsonValue(std::move(data)));
  return JsonValue(std::move(o));
}

JsonValue scalar_json(double s) { return JsonValue(s); }

JsonValue scalar_json(std::complex<double> s) {
  JsonObj o;
  o.emplace_back("re", JsonValue(s.real()));
  o.emplace_back("im", JsonValue(s.imag()));
  return JsonValue(std::move(o));
}

// ---------------------------------------------------------------------------
// Scenario accumulation.
// ---------------------------------------------------------------------------
JsonArr g_scenarios;

struct Scenario {
  std::string name;
  JsonObj inputs, params, outputs;
  explicit Scenario(std::string n) : name(std::move(n)) {}

  template <typename Array>
  Scenario& in(const std::string& k, const Array& a) {
    inputs.emplace_back(k, tensor_json(a));
    return *this;
  }
  Scenario& param(const std::string& k, JsonValue v) {
    params.emplace_back(k, std::move(v));
    return *this;
  }
  template <typename S>
  Scenario& param_scalar(const std::string& k, S s) {
    params.emplace_back(k, scalar_json(s));
    return *this;
  }
  template <typename Array>
  Scenario& out(const std::string& k, const Array& a) {
    outputs.emplace_back(k, tensor_json(a));
    return *this;
  }
  template <typename S>
  Scenario& out_scalar(const std::string& k, S s) {
    outputs.emplace_back(k, scalar_json(s));
    return *this;
  }
  void commit() {
    JsonObj o;
    o.emplace_back("name", JsonValue(name));
    o.emplace_back("inputs", JsonValue(std::move(inputs)));
    o.emplace_back("params", JsonValue(std::move(params)));
    o.emplace_back("outputs", JsonValue(std::move(outputs)));
    g_scenarios.push_back(JsonValue(std::move(o)));
  }
};

// Elementwise functions used by `elementwise` scenarios, tagged for the verifier.
auto sq = [](double x) { return x * x; };
auto dsq = [](double x) { return 2.0 * x; };
auto cube = [](double x) { return x * x * x; };
auto dcube = [](double x) { return 3.0 * x * x; };

// ===========================================================================
// Scenario builders, grouped to mirror the six ad_*.cpp test files.
// ===========================================================================

void forward_values(World& world) {
  const TiledRange tr2{{0, 2, 5}, {0, 2, 3}};  // (i,j) 5x3
  const TiledRange trSq{{0, 2, 4}, {0, 2, 4}};  // square

  // subt
  {
    RArray a = make_dense<RArray>(world, tr2), b = make_dense<RArray>(world, tr2);
    Scenario("subt").in("A", a).in("B", b).out("C", ad::subt(a, b)).commit();
  }
  // permute (transpose)
  {
    RArray a = make_dense<RArray>(world, tr2);
    Scenario("permute")
        .in("A", a)
        .param("in_annot", "i,j")
        .param("out_annot", "j,i")
        .out("C", ad::permute(a, "i,j", "j,i"))
        .commit();
  }
  // hadamard mult
  {
    RArray a = make_dense<RArray>(world, tr2), b = make_dense<RArray>(world, tr2);
    Scenario("mult").in("A", a).in("B", b).out("C", ad::mult(a, b)).commit();
  }
  // conj (complex)
  {
    CArray a = make_dense<CArray>(world, tr2);
    Scenario("conj").in("A", a).out("C", ad::conj(a)).commit();
  }
  // elementwise x^2
  {
    RArray a = make_dense<RArray>(world, tr2);
    Scenario("elementwise")
        .in("A", a)
        .param("fn", "x^2")
        .out("C", ad::elementwise(a, sq, dsq))
        .commit();
  }
  // trace
  {
    RArray a = make_dense<RArray>(world, trSq);
    Scenario("trace").in("A", a).out_scalar("s", ad::trace(a)).commit();
  }
  // sum
  {
    RArray a = make_dense<RArray>(world, tr2);
    Scenario("sum").in("A", a).out_scalar("s", ad::sum(a)).commit();
  }
  // squared_norm
  {
    RArray a = make_dense<RArray>(world, tr2);
    Scenario("squared_norm")
        .in("A", a)
        .out_scalar("s", ad::squared_norm(a))
        .commit();
  }
  // norm2
  {
    RArray a = make_dense<RArray>(world, tr2);
    Scenario("norm2").in("A", a).out_scalar("s", ad::norm2(a)).commit();
  }
  // dot (complex, bilinear/unconjugated)
  {
    CArray a = make_dense<CArray>(world, tr2), b = make_dense<CArray>(world, tr2);
    Scenario("dot").in("A", a).in("B", b).out_scalar("s", ad::dot(a, b)).commit();
  }
  // inner_product (complex, sesquilinear)
  {
    CArray a = make_dense<CArray>(world, tr2), b = make_dense<CArray>(world, tr2);
    Scenario("inner_product")
        .in("A", a)
        .in("B", b)
        .out_scalar("s", ad::inner_product(a, b))
        .commit();
  }
}

void forward_jvp(World& world) {
  const TiledRange trA{{0, 2, 5}, {0, 3, 4}};  // (i,k)
  const TiledRange trB{{0, 3, 4}, {0, 2}};     // (k,j)
  const TiledRange tr2{{0, 2, 5}, {0, 2, 3}};  // (i,j)
  const TiledRange trSq{{0, 2, 4}, {0, 2, 4}};
  const std::string aA = "i,k", aB = "k,j", aC = "i,j";

  // contract JVP
  {
    RArray A = make_dense<RArray>(world, trA), B = make_dense<RArray>(world, trB);
    RArray dA = make_dense<RArray>(world, trA), dB = make_dense<RArray>(world, trB);
    const double factor = 1.5;
    auto dc = ad::contract(ad::make_dual(A, dA), ad::make_dual(B, dB), aA, aB, aC,
                           factor);
    Scenario("contract_jvp")
        .in("A", A).in("B", B).in("dA", dA).in("dB", dB)
        .param("a_annot", aA).param("b_annot", aB).param("c_annot", aC)
        .param("factor", factor)
        .out("primal", dc.primal).out("tangent", *dc.tangent)
        .commit();
  }
  // add / subt JVP
  {
    RArray A = make_dense<RArray>(world, tr2), B = make_dense<RArray>(world, tr2);
    RArray dA = make_dense<RArray>(world, tr2), dB = make_dense<RArray>(world, tr2);
    auto da = ad::make_dual(A, dA), db = ad::make_dual(B, dB);
    auto s = ad::add(da, db);
    Scenario("add_jvp").in("A", A).in("B", B).in("dA", dA).in("dB", dB)
        .out("primal", s.primal).out("tangent", *s.tangent).commit();
    auto d = ad::subt(da, db);
    Scenario("subt_jvp").in("A", A).in("B", B).in("dA", dA).in("dB", dB)
        .out("primal", d.primal).out("tangent", *d.tangent).commit();
  }
  // scale JVP
  {
    RArray A = make_dense<RArray>(world, tr2), dA = make_dense<RArray>(world, tr2);
    const double alpha = 2.5;
    auto sc = ad::scale(ad::make_dual(A, dA), alpha);
    Scenario("scale_jvp").in("A", A).in("dA", dA).param("alpha", alpha)
        .out("primal", sc.primal).out("tangent", *sc.tangent).commit();
  }
  // permute JVP
  {
    RArray A = make_dense<RArray>(world, tr2), dA = make_dense<RArray>(world, tr2);
    auto p = ad::permute(ad::make_dual(A, dA), "i,j", "j,i");
    Scenario("permute_jvp").in("A", A).in("dA", dA)
        .param("in_annot", "i,j").param("out_annot", "j,i")
        .out("primal", p.primal).out("tangent", *p.tangent).commit();
  }
  // hadamard mult JVP
  {
    RArray A = make_dense<RArray>(world, tr2), B = make_dense<RArray>(world, tr2);
    RArray dA = make_dense<RArray>(world, tr2), dB = make_dense<RArray>(world, tr2);
    auto dc = ad::mult(ad::make_dual(A, dA), ad::make_dual(B, dB));
    Scenario("mult_jvp").in("A", A).in("B", B).in("dA", dA).in("dB", dB)
        .out("primal", dc.primal).out("tangent", *dc.tangent).commit();
  }
  // elementwise x^2 JVP
  {
    RArray A = make_dense<RArray>(world, tr2), dA = make_dense<RArray>(world, tr2);
    auto dc = ad::elementwise(ad::make_dual(A, dA), sq, dsq);
    Scenario("elementwise_jvp").in("A", A).in("dA", dA).param("fn", "x^2")
        .out("primal", dc.primal).out("tangent", *dc.tangent).commit();
  }
  // reduction JVPs (scalar primal + scalar tangent)
  {
    RArray A = make_dense<RArray>(world, trSq), dA = make_dense<RArray>(world, trSq);
    RArray B = make_dense<RArray>(world, trSq), dB = make_dense<RArray>(world, trSq);
    auto da = ad::make_dual(A, dA), db = ad::make_dual(B, dB);
    auto emit = [&](const std::string& name, auto ds) {
      Scenario(name).in("A", A).in("dA", dA)
          .out_scalar("primal", ds.primal).out_scalar("tangent", ds.tangent)
          .commit();
    };
    emit("trace_jvp", ad::trace(da));
    emit("sum_jvp", ad::sum(da));
    emit("squared_norm_jvp", ad::squared_norm(da));
    emit("norm2_jvp", ad::norm2(da));
    auto dotd = ad::dot(da, db);
    Scenario("dot_jvp").in("A", A).in("B", B).in("dA", dA).in("dB", dB)
        .out_scalar("primal", dotd.primal).out_scalar("tangent", dotd.tangent)
        .commit();
  }
}

void reverse_vjp(World& world) {
  const TiledRange trA{{0, 2, 5}, {0, 3, 4}};   // (i,k)
  const TiledRange trB{{0, 3, 4}, {0, 2}};      // (k,j)
  const TiledRange trC{{0, 2, 5}, {0, 2}};      // (i,j)
  const TiledRange tr2{{0, 2, 5}, {0, 2, 3}};   // (i,j) 5x3
  const TiledRange tr2T{{0, 2, 3}, {0, 2, 5}};  // transposed
  const TiledRange trSq{{0, 2, 4}, {0, 2, 4}};
  const std::string aA = "i,k", aB = "k,j", aC = "i,j";

  // contract VJP (the step-1 end-to-end scenario)
  {
    RArray A = make_dense<RArray>(world, trA), B = make_dense<RArray>(world, trB);
    RArray Cbar = make_dense<RArray>(world, trC);
    const double factor = 1.5;
    ad::Tape<RArray> tape;
    auto va = ad::make_leaf(tape, A), vb = ad::make_leaf(tape, B);
    auto vc = ad::contract(va, vb, aA, aB, aC, factor);
    tape.backward(vc.id, Cbar);
    Scenario("contract_vjp")
        .in("A", A).in("B", B).in("Cbar", Cbar)
        .param("a_annot", aA).param("b_annot", aB).param("c_annot", aC)
        .param("factor", factor)
        .out("Abar", tape.adjoint(va.id).value())
        .out("Bbar", tape.adjoint(vb.id).value())
        .commit();
  }
  // add VJP
  {
    RArray A = make_dense<RArray>(world, tr2), B = make_dense<RArray>(world, tr2);
    RArray Cbar = make_dense<RArray>(world, tr2);
    ad::Tape<RArray> tape;
    auto va = ad::make_leaf(tape, A), vb = ad::make_leaf(tape, B);
    auto vc = ad::add(va, vb);
    tape.backward(vc.id, Cbar);
    Scenario("add_vjp").in("A", A).in("B", B).in("Cbar", Cbar)
        .out("Abar", tape.adjoint(va.id).value())
        .out("Bbar", tape.adjoint(vb.id).value()).commit();
  }
  // subt VJP
  {
    RArray A = make_dense<RArray>(world, tr2), B = make_dense<RArray>(world, tr2);
    RArray Cbar = make_dense<RArray>(world, tr2);
    ad::Tape<RArray> tape;
    auto va = ad::make_leaf(tape, A), vb = ad::make_leaf(tape, B);
    auto vc = ad::subt(va, vb);
    tape.backward(vc.id, Cbar);
    Scenario("subt_vjp").in("A", A).in("B", B).in("Cbar", Cbar)
        .out("Abar", tape.adjoint(va.id).value())
        .out("Bbar", tape.adjoint(vb.id).value()).commit();
  }
  // mult -> permute -> scale chain VJP
  {
    RArray A = make_dense<RArray>(world, tr2), B = make_dense<RArray>(world, tr2);
    RArray Cbar = make_dense<RArray>(world, tr2T);
    const double alpha = 0.75;
    ad::Tape<RArray> tape;
    auto va = ad::make_leaf(tape, A), vb = ad::make_leaf(tape, B);
    auto vh = ad::mult(va, vb);
    auto vp = ad::permute(vh, "i,j", "j,i");
    auto vc = ad::scale(vp, alpha);
    tape.backward(vc.id, Cbar);
    Scenario("mult_permute_scale_vjp")
        .in("A", A).in("B", B).in("Cbar", Cbar).param("alpha", alpha)
        .out("Abar", tape.adjoint(va.id).value())
        .out("Bbar", tape.adjoint(vb.id).value()).commit();
  }
  // reduction VJPs (seed s̄ = 1), output gradient array
  {
    RArray A = make_dense<RArray>(world, trSq);
    auto emit = [&](const std::string& name, auto record) {
      ad::Tape<RArray> tape;
      auto va = ad::make_leaf(tape, A);
      auto s = record(va);
      tape.backward(s.id);
      Scenario(name).in("A", A).out("grad", tape.adjoint(va.id).value()).commit();
    };
    emit("sum_vjp", [](ad::Var<RArray>& v) { return ad::sum(v); });
    emit("squared_norm_vjp", [](ad::Var<RArray>& v) { return ad::squared_norm(v); });
    emit("norm2_vjp", [](ad::Var<RArray>& v) { return ad::norm2(v); });
    emit("trace_vjp", [](ad::Var<RArray>& v) { return ad::trace(v); });
  }
  // elementwise x^3 reduced by sum: grad = 3 A^2
  {
    RArray A = make_dense<RArray>(world, tr2);
    ad::Tape<RArray> tape;
    auto va = ad::make_leaf(tape, A);
    auto vy = ad::elementwise(va, cube, dcube);
    auto vs = ad::sum(vy);
    tape.backward(vs.id);
    Scenario("elementwise_vjp").in("A", A).param("fn", "x^3")
        .out("grad", tape.adjoint(va.id).value()).commit();
  }
  // fan-out: y = sum(A∘P + A∘Q), grad = P + Q
  {
    RArray A = make_dense<RArray>(world, tr2);
    RArray P = make_dense<RArray>(world, tr2), Q = make_dense<RArray>(world, tr2);
    ad::Tape<RArray> tape;
    auto va = ad::make_leaf(tape, A);
    auto vp = ad::make_leaf(tape, P, /*requires_grad=*/false);
    auto vq = ad::make_leaf(tape, Q, /*requires_grad=*/false);
    auto vsum = ad::add(ad::mult(va, vp), ad::mult(va, vq));
    auto vy = ad::sum(vsum);
    tape.backward(vy.id);
    Scenario("fan_out_vjp").in("A", A).in("P", P).in("Q", Q)
        .out("grad", tape.adjoint(va.id).value()).commit();
  }
}

void reverse_vjp_complex(World& world) {
  const TiledRange tr2{{0, 2, 5}, {0, 2, 3}};
  const std::complex<double> sbar(0.3, -0.7);

  CArray A = make_dense<CArray>(world, tr2), B = make_dense<CArray>(world, tr2);

  // dot VJP (bilinear): Ā = s̄·conj(B), B̄ = s̄·conj(A)
  {
    ad::Tape<CArray> tape;
    auto va = ad::make_leaf(tape, A), vb = ad::make_leaf(tape, B);
    auto vs = ad::dot(va, vb);
    tape.backward(vs.id, sbar);
    Scenario("complex_dot_vjp").in("A", A).in("B", B).param_scalar("sbar", sbar)
        .out("Abar", tape.adjoint(va.id).value())
        .out("Bbar", tape.adjoint(vb.id).value()).commit();
  }
  // inner_product VJP (sesquilinear): Ā = conj(s̄)·B, B̄ = s̄·A
  {
    ad::Tape<CArray> tape;
    auto va = ad::make_leaf(tape, A), vb = ad::make_leaf(tape, B);
    auto vs = ad::inner_product(va, vb);
    tape.backward(vs.id, sbar);
    Scenario("complex_inner_product_vjp")
        .in("A", A).in("B", B).param_scalar("sbar", sbar)
        .out("Abar", tape.adjoint(va.id).value())
        .out("Bbar", tape.adjoint(vb.id).value()).commit();
  }
}

// E(X) = ||X·X||^2; reverse-mode gradient g = dE/dX (seed C̄ = 2C).
RArray grad_E(World& world, const RArray& X) {
  ad::Tape<RArray> tape;
  auto vx = ad::make_leaf(tape, X);
  auto vc = ad::contract(vx, vx, "i,k", "k,j", "i,j");
  RArray seed;
  seed("i,j") = 2.0 * vc.value("i,j");
  tape.backward(vc.id, seed);
  return tape.adjoint(vx.id).value();
}

void hvp(World& world) {
  const TiledRange trSq{{0, 2, 4}, {0, 2, 4}};  // 4x4 square
  RArray X = make_dense<RArray>(world, trSq), v = make_dense<RArray>(world, trSq);

  // forward-over-reverse: gradient computed on Dual leaves seeded with v.
  using DualR = ad::Dual<RArray>;
  ad::Tape<DualR> tape;
  auto vx = ad::make_leaf(tape, ad::make_dual(X, v));
  auto vc = ad::contract(vx, vx, "i,k", "k,j", "i,j");
  DualR seed = ad::scale(vc.value, 2.0);
  tape.backward(vc.id, seed);
  const DualR& g = tape.adjoint(vx.id).value();

  Scenario("hvp").in("X", X).in("v", v)
      .out("g", g.primal).out("Hv", *g.tangent).commit();
}

void sparse_contract_vjp(World& world) {
  const float saved = TiledArray::SparseShape<float>::threshold();
  TiledArray::SparseShape<float>::threshold(std::numeric_limits<float>::min());

  const TiledRange trA{{0, 2, 4, 6}, {0, 2, 4, 6}};
  const TiledRange trB{{0, 2, 4, 6}, {0, 2, 4, 6}};
  const TiledRange trC{{0, 2, 4, 6}, {0, 2, 4, 6}};
  const std::string aA = "i,k", aB = "k,j", aC = "i,j";

  auto patA = [](const auto& idx) { return !(idx[0] == 0 && idx[1] == 1); };
  auto patB = [](const auto& idx) { return !(idx[0] == 2 && idx[1] == 0); };
  auto patC = [](const auto&) { return true; };

  SpArray A = make_sparse(world, trA, patA);
  SpArray B = make_sparse(world, trB, patB);
  SpArray Cbar = make_sparse(world, trC, patC);

  ad::Tape<SpArray> tape;
  auto va = ad::make_leaf(tape, A), vb = ad::make_leaf(tape, B);
  auto vc = ad::contract(va, vb, aA, aB, aC);
  tape.backward(vc.id, Cbar);

  // densified dump: zero blocks -> zeros, so the dense JAX recompute matches.
  Scenario("sparse_contract_vjp")
      .in("A", A).in("B", B).in("Cbar", Cbar)
      .param("a_annot", aA).param("b_annot", aB).param("c_annot", aC)
      .out("Abar", tape.adjoint(va.id).value())
      .out("Bbar", tape.adjoint(vb.id).value())
      .commit();

  TiledArray::SparseShape<float>::threshold(saved);
}

}  // namespace

int main(int argc, char** argv) {
  World& world = TA::initialize(argc, argv);

  forward_values(world);
  forward_jvp(world);
  reverse_vjp(world);
  reverse_vjp_complex(world);
  hvp(world);
  sparse_contract_vjp(world);

  // The output path is argv[1] if given, else golden.json in the cwd.
  const std::string out_path = (argc > 1) ? argv[1] : "golden.json";
  if (world.rank() == 0) {
    JsonObj root;
    root.emplace_back("scenarios", JsonValue(std::move(g_scenarios)));
    std::ofstream os(out_path);
    JsonValue(std::move(root)).dump(os);
    os << '\n';
  }
  world.gop.fence();

  TA::finalize();
  return 0;
}
