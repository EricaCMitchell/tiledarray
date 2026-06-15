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
/// This header defines the *functional* primitive surface (autodiff plan B1)
/// that both the native C++ AD layer (B4) and the Enzyme custom rules (B5) hang
/// derivative rules on. Each function is side-effect-free and value-returning:
/// it wraps the existing lazy-expression engine (`Expr::eval_to`) but presents a
/// stable function signature, which is what makes a tape node or an Enzyme
/// custom rule well-defined. These are the *only* operations the AD machinery
/// knows about; everything underneath (SUMMA, MADWorld tasks, BLAS, MPI
/// collectives) is opaque to it on purpose — see the autodiff plan's "narrow
/// waist" rationale.
///
/// The full Part A primitive set is implemented here: `contract`, `add`,
/// `subt`, `scale`, `permute`, `mult` (Hadamard), `conj`, `elementwise`, and
/// the reductions `trace`/`sum`/`squared_norm`/`norm2`/`dot`/`inner_product`.
/// The forward (`dual.h`) and reverse (`tape.h`) AD layers hang their JVP/VJP
/// rules on exactly these functions.

namespace TiledArray::ad {

namespace detail {

/// Build a canonical comma-separated annotation `"i0,i1,...,i{rank-1}"`.
///
/// Used by the rank-agnostic elementwise primitives (`add`, `scale`) where the
/// index labels are immaterial — only that every operand and the result share
/// the same labels. Contraction, by contrast, takes its annotations explicitly
/// because the index pattern *is* the operation.
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
/// This is the workhorse primitive: a single binary contraction expressed in
/// TA's annotation form. The annotation triple fully determines the operation,
/// including Hadamard, contracted, and external index groups, and any implicit
/// permutation of the output — the expression engine resolves all of that.
///
/// \param a left operand
/// \param b right operand
/// \param a_annot annotation for `a`, e.g. `"i,k"`
/// \param b_annot annotation for `b`, e.g. `"k,j"`
/// \param c_annot annotation for the result, e.g. `"i,j"`
/// \param factor scalar prefactor folded into the contraction (the fused-gemm
///        `factor`; its VJP carries `conj(factor)` per the scale rule — for
///        real arrays simply `factor`)
/// \return a freshly allocated result array on `a.world()`
///
/// \note The functional waist is deliberately `beta = 0` (a fresh result). The
///       in-place `beta * C` accumulation that `ContractReduce`/`Tensor::gemm`
///       also support is modeled at the AD layer as an additive input whose VJP
///       is a pass-through (Part A "Fused-gemm subtlety"); it is not part of
///       this side-effect-free surface.
template <typename Array>
Array contract(const Array& a, const Array& b, const std::string& a_annot,
               const std::string& b_annot, const std::string& c_annot,
               typename Array::numeric_type factor = 1) {
  Array c;
  // set_world binds the evaluator to the left operand's world; a fresh `c`
  // would otherwise fall back to get_default_world() and break multi-World use
  // (CLAUDE.md, Expr::eval_to world-selection priority).
  c(c_annot) = (factor * (a(a_annot) * b(b_annot))).set_world(a.world());
  return c;
}

/// Elementwise sum `C = A + B`.
///
/// Both operands must share the same tiled range; the (immaterial) index labels
/// are generated canonically.
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
/// The permutation is expressed as a change of annotation: `a` is read under
/// `in_annot` and written under `out_annot` (e.g. `in_annot="i,j"`,
/// `out_annot="j,i"` is a transpose). Its VJP is the inverse permutation —
/// reading the cotangent under `out_annot` and writing under `in_annot`.
template <typename Array>
Array permute(const Array& a, const std::string& in_annot,
              const std::string& out_annot) {
  Array c;
  c(out_annot) = a(in_annot).set_world(a.world());
  return c;
}

/// Hadamard (elementwise) product `C = A ∘ B`.
///
/// Both operands and the result share the same index labels, which is exactly
/// what makes `a(annot) * b(annot)` an elementwise (rather than contracting)
/// product in the expression engine.
template <typename Array>
Array mult(const Array& a, const Array& b) {
  const std::string annot = detail::canonical_annotation(a.trange().rank());
  Array c;
  c(annot) = (a(annot) * b(annot)).set_world(a.world());
  return c;
}

/// Complex conjugation `C = conj(A)`.
///
/// A genuine primitive of the set (not a generic linear op): `conj` is
/// *antilinear*, so its JVP is `conj(dA)` and its VJP is `conj(C̄)` (Part A).
/// For real arrays it is the identity. Closing the primitive set under
/// differentiation requires `conj` because the complex VJPs reference it.
template <typename Array>
Array conj(const Array& a) {
  const std::string annot = detail::canonical_annotation(a.trange().rank());
  Array c;
  c(annot) = a(annot).conj().set_world(a.world());
  return c;
}

/// Apply a unary scalar function elementwise: `C(e) = f(A(e))`.
///
/// `df = f'` is a *required* argument, not optional: a bare `f` is opaque to
/// both the tape and Enzyme-as-atomic-primitive, so the derivative rule needs
/// the caller-supplied derivative (Part A / B1). Forward evaluation uses only
/// `f`; `df` is consumed by the JVP/VJP layers. `f` must be holomorphic (or
/// real) for the elementwise VJP `Ā += conj(f'(A)) ∘ C̄` to be valid.
///
/// \tparam F   callable `numeric_type -> numeric_type` (the function)
/// \tparam DF  callable `numeric_type -> numeric_type` (its derivative)
template <typename Array, typename F, typename DF>
Array elementwise(const Array& a, F f, DF /*df*/) {
  // Build the result via foreach so the policy-correct sparse/dense path and
  // the primal's tiling are reused. The void-returning op makes the sparse
  // result inherit `a`'s sparsity pattern (the B3 lower-bound on structure).
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
/// \note The VJP `Ā += (s̄/‖A‖)·A` is undefined at `A = 0`; that guard lives in
///       the derivative layers, not here.
template <typename Array>
typename Array::scalar_type norm2(const Array& a) {
  using std::sqrt;
  return sqrt(squared_norm(a));
}

/// Bilinear (unconjugated) inner product `s = Σ_e A(e)·B(e)`.
///
/// This is TA's `dot`; it does **not** conjugate `A`. Its sesquilinear twin is
/// `inner_product` below, and the two have *different* VJPs (Part A) — bind
/// each to its own rule.
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
