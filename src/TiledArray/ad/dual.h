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
 *  dual.h
 *  Forward-mode (JVP) dual numbers for TiledArray's AD layer.
 */

#ifndef TILEDARRAY_AD_DUAL_H__INCLUDED
#define TILEDARRAY_AD_DUAL_H__INCLUDED

#include <complex>
#include <optional>
#include <string>
#include <utility>

#include <TiledArray/ad/ops.h>

/// \file dual.h
///
/// Forward-mode AD. A `Dual<Array>` carries `{primal, tangent}`, and every
/// functional primitive produces the paired primal and tangent. This mode is
/// cheap, because it needs no tape. Use it when the number of input
/// perturbations is small.
///
/// The tangent is a symbolic zero until something flows into it. A null
/// `std::optional` holds that state, as the reverse-mode shadow does. A
/// constant operand (`make_constant`) carries no tangent, an op over only
/// constant operands produces no tangent, and an untouched gradient direction
/// allocates no array. This prevents the `SparsePolicy` `fill(0)` problem and
/// gives a zero tangent for an unconnected input at no cost.
///
/// Reductions return a `DualScalar<S>`, which holds a primal scalar and a
/// tangent scalar. The forward ops use the same primitives as the rest of the
/// layer. Thus a `Dual` whose `Array` is a reverse-mode `Var` composes into
/// forward-over-reverse, which is the route to Hessian-vector products.

namespace TiledArray::ad {

/// A scalar dual number: primal value plus its directional derivative.
template <typename S>
struct DualScalar {
  S primal{};
  S tangent{};
};

/// An array dual number: a primal `Array` and an optional tangent `Array`.
///
/// `tangent == std::nullopt` is the symbolic zero: no tangent flows here.
///
/// The `numeric_type` and `scalar_type` aliases come from the underlying
/// `Array`. They let a reverse-mode `Tape<Dual<Array>>` treat a dual as its
/// element type, which is the mechanism behind forward-over-reverse
/// (Hessian-vector products). The VJP closures of the tape use only the
/// primitives, thus they dispatch to the dual overloads below and carry the
/// forward tangent through the backward pass.
template <typename Array>
struct Dual {
  using numeric_type = typename Array::numeric_type;
  using scalar_type = typename Array::scalar_type;

  Array primal;
  std::optional<Array> tangent;

  bool has_tangent() const noexcept { return tangent.has_value(); }
};

/// Build an active dual `{primal, tangent}`.
template <typename Array>
Dual<Array> make_dual(Array primal, Array tangent) {
  return Dual<Array>{std::move(primal), std::move(tangent)};
}

/// Build a constant (inactive) dual: primal only, symbolic-zero tangent.
template <typename Array>
Dual<Array> make_constant(Array primal) {
  return Dual<Array>{std::move(primal), std::nullopt};
}

namespace detail {

/// Sum two optional tangents, propagating the symbolic zero.
template <typename Array>
std::optional<Array> opt_add(std::optional<Array> x, std::optional<Array> y) {
  if (!x) return y;
  if (!y) return x;
  return std::optional<Array>(ad::add(*x, *y));
}

/// Real part for real or complex scalars (`std::real` covers both).
template <typename S>
auto real_of(const S& s) {
  return std::real(s);
}

}  // namespace detail

/// JVP of `contract`: `dC = factor*(dA·B + A·dB)`.
template <typename Array>
Dual<Array> contract(const Dual<Array>& a, const Dual<Array>& b,
                     const std::string& a_annot, const std::string& b_annot,
                     const std::string& c_annot,
                     typename Array::numeric_type factor = 1) {
  Array primal =
      ad::contract(a.primal, b.primal, a_annot, b_annot, c_annot, factor);
  std::optional<Array> t;
  if (a.has_tangent())
    t = ad::contract(*a.tangent, b.primal, a_annot, b_annot, c_annot, factor);
  if (b.has_tangent())
    t = detail::opt_add<Array>(
        std::move(t), ad::contract(a.primal, *b.tangent, a_annot, b_annot,
                                   c_annot, factor));
  return Dual<Array>{std::move(primal), std::move(t)};
}

/// JVP of `add`: `dA + dB`.
template <typename Array>
Dual<Array> add(const Dual<Array>& a, const Dual<Array>& b) {
  Array primal = ad::add(a.primal, b.primal);
  return Dual<Array>{std::move(primal),
                     detail::opt_add<Array>(a.tangent, b.tangent)};
}

/// JVP of `subt`: `dA - dB`.
template <typename Array>
Dual<Array> subt(const Dual<Array>& a, const Dual<Array>& b) {
  Array primal = ad::subt(a.primal, b.primal);
  std::optional<Array> t = a.tangent;
  if (b.has_tangent())
    t = detail::opt_add<Array>(
        std::move(t), ad::scale(*b.tangent, typename Array::numeric_type{-1}));
  return Dual<Array>{std::move(primal), std::move(t)};
}

/// JVP of `scale` by a *constant* `alpha`: `alpha·dA`.
template <typename Array>
Dual<Array> scale(const Dual<Array>& a, typename Array::numeric_type alpha) {
  Array primal = ad::scale(a.primal, alpha);
  std::optional<Array> t;
  if (a.has_tangent()) t = ad::scale(*a.tangent, alpha);
  return Dual<Array>{std::move(primal), std::move(t)};
}

/// JVP of `permute`: `P(dA)`.
template <typename Array>
Dual<Array> permute(const Dual<Array>& a, const std::string& in_annot,
                    const std::string& out_annot) {
  Array primal = ad::permute(a.primal, in_annot, out_annot);
  std::optional<Array> t;
  if (a.has_tangent()) t = ad::permute(*a.tangent, in_annot, out_annot);
  return Dual<Array>{std::move(primal), std::move(t)};
}

/// JVP of Hadamard `mult`: `dA∘B + A∘dB`.
template <typename Array>
Dual<Array> mult(const Dual<Array>& a, const Dual<Array>& b) {
  Array primal = ad::mult(a.primal, b.primal);
  std::optional<Array> t;
  if (a.has_tangent()) t = ad::mult(*a.tangent, b.primal);
  if (b.has_tangent())
    t = detail::opt_add<Array>(std::move(t), ad::mult(a.primal, *b.tangent));
  return Dual<Array>{std::move(primal), std::move(t)};
}

/// JVP of `conj` (antilinear): `conj(dA)`.
template <typename Array>
Dual<Array> conj(const Dual<Array>& a) {
  Array primal = ad::conj(a.primal);
  std::optional<Array> t;
  if (a.has_tangent()) t = ad::conj(*a.tangent);
  return Dual<Array>{std::move(primal), std::move(t)};
}

/// JVP of unary `elementwise` f: `f'(A) ∘ dA`. `f` must be holomorphic or
/// real.
template <typename Array, typename F, typename DF>
Dual<Array> elementwise(const Dual<Array>& a, F f, DF df) {
  Array primal = ad::elementwise(a.primal, f, df);
  std::optional<Array> t;
  if (a.has_tangent()) {
    Array fp = ad::elementwise(a.primal, df, df);  // f'(A) elementwise
    t = ad::mult(fp, *a.tangent);
  }
  return Dual<Array>{std::move(primal), std::move(t)};
}

/// JVP of `trace`: `trace(dA)`.
template <typename Array>
DualScalar<typename Array::numeric_type> trace(const Dual<Array>& a) {
  using S = typename Array::numeric_type;
  return DualScalar<S>{ad::trace(a.primal),
                       a.has_tangent() ? ad::trace(*a.tangent) : S{}};
}

/// JVP of `sum`: `sum(dA)`.
template <typename Array>
DualScalar<typename Array::numeric_type> sum(const Dual<Array>& a) {
  using S = typename Array::numeric_type;
  return DualScalar<S>{ad::sum(a.primal),
                       a.has_tangent() ? ad::sum(*a.tangent) : S{}};
}

/// JVP of `squared_norm`: `2·Re⟨A, dA⟩` (real-valued).
template <typename Array>
DualScalar<typename Array::scalar_type> squared_norm(const Dual<Array>& a) {
  using S = typename Array::scalar_type;
  const S primal = ad::squared_norm(a.primal);
  S tangent{};
  if (a.has_tangent())
    tangent = S{2} * detail::real_of(ad::inner_product(a.primal, *a.tangent));
  return DualScalar<S>{primal, tangent};
}

/// JVP of `norm2`: `Re⟨A, dA⟩ / ‖A‖`. It is undefined at `A = 0`.
template <typename Array>
DualScalar<typename Array::scalar_type> norm2(const Dual<Array>& a) {
  using S = typename Array::scalar_type;
  const S n = ad::norm2(a.primal);
  S tangent{};
  if (a.has_tangent()) {
    TA_ASSERT(n != S{0} && "norm2 JVP is undefined at A = 0");
    tangent = detail::real_of(ad::inner_product(a.primal, *a.tangent)) / n;
  }
  return DualScalar<S>{n, tangent};
}

/// JVP of bilinear `dot`: `dot(dA, B) + dot(A, dB)`.
template <typename Array>
DualScalar<typename Array::numeric_type> dot(const Dual<Array>& a,
                                             const Dual<Array>& b) {
  using S = typename Array::numeric_type;
  S tangent{};
  if (a.has_tangent()) tangent += ad::dot(*a.tangent, b.primal);
  if (b.has_tangent()) tangent += ad::dot(a.primal, *b.tangent);
  return DualScalar<S>{ad::dot(a.primal, b.primal), tangent};
}

/// JVP of sesquilinear `inner_product`: `⟨dA, B⟩ + ⟨A, dB⟩`.
template <typename Array>
DualScalar<typename Array::numeric_type> inner_product(const Dual<Array>& a,
                                                       const Dual<Array>& b) {
  using S = typename Array::numeric_type;
  S tangent{};
  if (a.has_tangent()) tangent += ad::inner_product(*a.tangent, b.primal);
  if (b.has_tangent()) tangent += ad::inner_product(a.primal, *b.tangent);
  return DualScalar<S>{ad::inner_product(a.primal, b.primal), tangent};
}

}  // namespace TiledArray::ad

#endif  // TILEDARRAY_AD_DUAL_H__INCLUDED
