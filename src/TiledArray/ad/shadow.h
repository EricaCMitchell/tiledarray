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
 *  shadow.h
 *  Tangent / adjoint ("shadow") array infrastructure for TiledArray's AD layer.
 */

#ifndef TILEDARRAY_AD_SHADOW_H__INCLUDED
#define TILEDARRAY_AD_SHADOW_H__INCLUDED

#include <optional>
#include <string>
#include <utility>

#include <TiledArray/ad/ops.h>
#include <TiledArray/dist_array.h>

/// \file shadow.h
///
/// Adjoint and tangent ("shadow") array infrastructure.
///
/// A derivative needs an array that mirrors the distributed and sparse
/// structure of a primal. `DistArray` holds `shape_` and `pmap_` as
/// `shared_ptr<const>` members, thus a shadow shares them at no copy cost.
///
/// An adjoint slot stays null until the first write into it (the symbolic
/// zero), as in the lazy `.grad` of PyTorch. A zero-seeded sparse array is
/// empty, because the shape rejects below-threshold tiles. The symbolic zero
/// prevents that, and it gives a zero gradient for an unconnected input at no
/// memory cost.
///
/// Cotangent sparsity for `SparsePolicy` follows two rules:
///   1. The shape of the primal operand is the lower bound. A materialized
///      mirror inherits it from `make_shadow`, thus every tile that can carry
///      gradient has a slot.
///   2. The ordinary `SparseShape` screening then drops below-threshold tiles.
///      It narrows the structure, but never widens it past rule 1.
///
/// The result is a deterministic function of the operand shapes, and not of
/// the accumulation order. `tests/ad_sparse.cpp` tests it on block-sparse
/// inputs.

namespace TiledArray::ad {

/// Construct an empty mirror of `arg` with the same structure.
///
/// The result shares the tiled range, shape, and pmap of `arg`. The shape and
/// the pmap come from `shared_ptr<const>` members, thus the copy is cheap. No
/// tiles are set. Use this function when the code needs a real zero array. The
/// default adjoint-accumulation path uses the symbolic zero of `Cotangent` and
/// never calls it.
///
/// \note This function does not `fill(0)`. A `fill(0)` empties a
///       `SparsePolicy` array.
template <typename Array>
Array make_shadow(const Array& arg) {
  return Array(arg.world(), arg.trange(), arg.shape(), arg.pmap());
}

/// A cotangent (adjoint) accumulator with a symbolic-zero state.
///
/// It holds nothing (the symbolic zero) or a materialized `Array`.
/// `accumulate` does the reverse-mode fan-in `Ā += …`. The first contribution
/// moves or copies the delta in. Later contributions sum in place with the
/// array `+=` (the `add_to` family). These summed contributions encode the
/// fan-out of the forward program, where several ops consume one array.
template <typename Array>
class Cotangent {
 public:
  Cotangent() = default;

  /// \return true if this is still the symbolic zero (nothing accumulated yet)
  bool is_zero() const noexcept { return !value_.has_value(); }

  /// Fan-in one cotangent contribution.
  ///
  /// The first call moves the delta in with no arithmetic. Later calls sum it
  /// in with `ad::add`. The functional `add` keeps `Cotangent` agnostic of the
  /// element type, thus `Cotangent` also works when `Array` is a `Dual`
  /// (forward-over-reverse). This function consumes `delta`.
  void accumulate(Array delta) {
    if (!value_) {
      value_ = std::move(delta);
    } else {
      value_ = ad::add(*value_, delta);  // Ā += delta
    }
  }

  /// Access the materialized cotangent. Precondition: `!is_zero()`.
  const Array& value() const {
    TA_ASSERT(value_.has_value());
    return *value_;
  }

  /// Lower the symbolic zero to a concrete value.
  ///
  /// \return the accumulated array, or an explicit structural zero with the
  ///         shape of `like`.
  ///
  /// Most callers must branch on `is_zero()` and propagate the symbolic zero.
  /// This function exists for the few places, such as tests, that need a
  /// concrete array in all cases.
  Array value_or_zero(const Array& like) const {
    if (value_) return *value_;
    // 0 * like routes through the engine and gives the policy-correct zero:
    // explicit zero tiles for DensePolicy, an empty array for SparsePolicy.
    return ad::scale(like, typename Array::numeric_type{0});
  }

 private:
  std::optional<Array> value_;
};

}  // namespace TiledArray::ad

#endif  // TILEDARRAY_AD_SHADOW_H__INCLUDED
