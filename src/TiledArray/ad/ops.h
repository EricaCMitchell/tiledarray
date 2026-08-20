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
 *  ops.h
 *  The functional primitive API ("narrow waist") for TiledArray's AD layer.
 */

#ifndef TILEDARRAY_AD_OPS_H__INCLUDED
#define TILEDARRAY_AD_OPS_H__INCLUDED

#include <cmath>
#include <string>

#include <TiledArray/conversions/foreach.h>
#include <TiledArray/dist_array.h>

/// \file ops.h
///
/// The functional primitive surface of the AD layer. Each function is
/// side-effect-free and returns a value. It wraps the lazy-expression engine
/// (`Expr::eval_to`) behind a stable signature. That stable signature is what
/// makes a tape node or a custom derivative rule well-defined.
///
/// These are the only operations that the AD machinery knows about. Everything
/// below them (SUMMA, MADWorld tasks, BLAS, MPI collectives) is opaque to it on
/// purpose.
///
/// The set is `contract`, `add`, `subt`, `scale`, `permute`, `mult`
/// (Hadamard), `conj`, `elementwise`, and the reductions `trace`, `sum`,
/// `squared_norm`, `norm2`, `dot`, and `inner_product`. The forward layer
/// (`dual.h`) and the reverse layer (`tape.h`) attach their JVP and VJP rules
/// to these functions.

namespace TiledArray::ad {

namespace detail {

/// Build a canonical comma-separated annotation `"i0,i1,...,i{rank-1}"`.
///
/// The rank-agnostic elementwise primitives (`add`, `scale`) use this because
/// the index labels do not matter. Only one property matters: every operand
/// and the result share the same labels. Contraction takes its annotations
/// explicitly, because there the index pattern is the operation.
inline std::string canonical_annotation(std::size_t rank) {
  std::string annot;
  for (std::size_t d = 0; d < rank; ++d) {
    if (d != 0) annot += ',';
    annot += 'i';
    annot += std::to_string(d);
  }
  return annot;
}

}  // namespace detail

/// Tensor contraction `C = factor * (A * B)` under explicit Einstein
/// annotations.
///
/// This is the primary primitive: one binary contraction in TA's annotation
/// form. The annotation triple fully determines the operation. It gives the
/// Hadamard, contracted, and external index groups, and any implicit
/// permutation of the output. The expression engine resolves all of that.
///
/// \param a left operand
/// \param b right operand
/// \param a_annot annotation for `a`, for example `"i,k"`
/// \param b_annot annotation for `b`, for example `"k,j"`
/// \param c_annot annotation for the result, for example `"i,j"`
/// \param factor scalar prefactor folded into the contraction (the fused-gemm
///        `factor`). Its VJP carries `conj(factor)`, which is `factor` for
///        real arrays.
/// \return a new result array on `a.world()`
///
/// \note This surface always uses `beta = 0`, thus the result is new. The AD
///       layer models the in-place `beta * C` accumulation of
///       `ContractReduce` and `Tensor::gemm` as an additive input whose VJP is
///       a pass-through. That accumulation is not part of this
///       side-effect-free surface.
template <typename Array>
Array contract(const Array& a, const Array& b, const std::string& a_annot,
               const std::string& b_annot, const std::string& c_annot,
               typename Array::numeric_type factor = 1) {
  Array c;
  // set_world binds the evaluator to the world of the left operand. A fresh
  // `c` falls back to get_default_world() and breaks multi-World use.
  c(c_annot) = (factor * (a(a_annot) * b(b_annot))).set_world(a.world());
  return c;
}

/// Elementwise sum `C = A + B`.
///
/// Both operands must have the same tiled range. The index labels do not
/// matter, thus the code generates them canonically.
template <typename Array>
Array add(const Array& a, const Array& b) {
  const std::string annot = detail::canonical_annotation(a.trange().rank());
  Array c;
  c(annot) = (a(annot) + b(annot)).set_world(a.world());
  return c;
}

/// Scalar multiply `C = alpha * A`.
template <typename Array>
Array scale(const Array& a, typename Array::numeric_type alpha) {
  const std::string annot = detail::canonical_annotation(a.trange().rank());
  Array c;
  c(annot) = (alpha * a(annot)).set_world(a.world());
  return c;
}

/// Elementwise difference `C = A - B`.
template <typename Array>
Array subt(const Array& a, const Array& b) {
  const std::string annot = detail::canonical_annotation(a.trange().rank());
  Array c;
  c(annot) = (a(annot) - b(annot)).set_world(a.world());
  return c;
}

/// Index permutation `C = P(A)`.
///
/// A change of annotation gives the permutation. The code reads `a` under
/// `in_annot` and writes it under `out_annot`. For example, `in_annot="i,j"`
/// with `out_annot="j,i"` is a transpose. The VJP is the inverse permutation:
/// it reads the cotangent under `out_annot` and writes it under `in_annot`.
template <typename Array>
Array permute(const Array& a, const std::string& in_annot,
              const std::string& out_annot) {
  Array c;
  c(out_annot) = a(in_annot).set_world(a.world());
  return c;
}

/// Hadamard (elementwise) product `C = A ∘ B`.
///
/// Both operands and the result share the same index labels. That is what
/// makes `a(annot) * b(annot)` an elementwise product in the expression
/// engine, and not a contraction.
template <typename Array>
Array mult(const Array& a, const Array& b) {
  const std::string annot = detail::canonical_annotation(a.trange().rank());
  Array c;
  c(annot) = (a(annot) * b(annot)).set_world(a.world());
  return c;
}

/// Complex conjugation `C = conj(A)`.
///
/// `conj` is antilinear, thus its JVP is `conj(dA)` and its VJP is `conj(C̄)`.
/// For real arrays it is the identity. The primitive set needs `conj` to be
/// closed under differentiation, because the complex VJPs use it.
template <typename Array>
Array conj(const Array& a) {
  const std::string annot = detail::canonical_annotation(a.trange().rank());
  Array c;
  c(annot) = a(annot).conj().set_world(a.world());
  return c;
}

/// Apply a unary scalar function elementwise: `C(e) = f(A(e))`.
///
/// `df = f'` is a required argument. A bare `f` is opaque to the tape and to a
/// custom derivative rule, thus the rule needs the derivative from the caller.
/// Forward evaluation uses only `f`. The JVP and VJP layers use `df`. `f` must
/// be holomorphic or real to make the elementwise VJP `Ā += conj(f'(A)) ∘ C̄`
/// valid.
///
/// \tparam F   callable `numeric_type -> numeric_type` (the function)
/// \tparam DF  callable `numeric_type -> numeric_type` (its derivative)
template <typename Array, typename F, typename DF>
Array elementwise(const Array& a, F f, DF /*df*/) {
  // foreach reuses the policy-correct sparse or dense path and the tiling of
  // the primal. The void-returning op makes a sparse result inherit the
  // sparsity pattern of `a`, which is the lower bound on the structure.
  return TiledArray::foreach (
      a, [f](typename Array::value_type& out,
              const typename Array::value_type& in) { out = in.unary(f); });
}

/// Trace `s = Σ_i A(i,i)` (square operands only). Linear → scalar.
template <typename Array>
typename Array::numeric_type trace(const Array& a) {
  const std::string annot = detail::canonical_annotation(a.trange().rank());
  return a(annot).trace(a.world()).get();
}

/// Full reduction `s = Σ_e A(e)`. Linear → scalar.
template <typename Array>
typename Array::numeric_type sum(const Array& a) {
  const std::string annot = detail::canonical_annotation(a.trange().rank());
  return a(annot).sum(a.world()).get();
}

/// Squared Frobenius norm `s = Σ_e |A(e)|²` (real-valued, even for complex `A`).
template <typename Array>
typename Array::scalar_type squared_norm(const Array& a) {
  const std::string annot = detail::canonical_annotation(a.trange().rank());
  return a(annot).squared_norm(a.world()).get();
}

/// Frobenius norm `s = ‖A‖ = √squared_norm(A)`.
///
/// \note The VJP `Ā += (s̄/‖A‖)·A` is undefined at `A = 0`. The derivative
///       layers hold that guard, not this function.
template <typename Array>
typename Array::scalar_type norm2(const Array& a) {
  using std::sqrt;
  return sqrt(squared_norm(a));
}

/// Bilinear (unconjugated) inner product `s = Σ_e A(e)·B(e)`.
///
/// This is TA's `dot`. It does not conjugate `A`. Its sesquilinear twin is
/// `inner_product` below. The two have different VJPs, thus each one needs its
/// own rule.
template <typename Array>
typename Array::numeric_type dot(const Array& a, const Array& b) {
  const std::string annot = detail::canonical_annotation(a.trange().rank());
  return a(annot).dot(b(annot), a.world()).get();
}

/// Sesquilinear inner product `s = Σ_e conj(A(e))·B(e)`.
template <typename Array>
typename Array::numeric_type inner_product(const Array& a, const Array& b) {
  const std::string annot = detail::canonical_annotation(a.trange().rank());
  return a(annot).inner_product(b(annot), a.world()).get();
}

}  // namespace TiledArray::ad

#endif  // TILEDARRAY_AD_OPS_H__INCLUDED
