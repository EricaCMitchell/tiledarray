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

#include <string>

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
/// Phase 0 (de-risk) implements only the subset needed to stand up and verify
/// the `contract` VJP: `contract`, `add`, and `scale`. The remaining primitives
/// from the Part A table (subt/permute/mult/trace/dot/norm/sum/elementwise/conj)
/// are Phase 1 work.

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

}  // namespace TiledArray::ad

#endif  // TILEDARRAY_AD_OPS_H__INCLUDED
