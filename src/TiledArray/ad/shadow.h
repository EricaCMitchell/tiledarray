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
/// Adjoint/tangent "shadow" infrastructure (autodiff plan B2).
///
/// A derivative needs an array that mirrors a primal's distributed/sparse
/// structure. Because `DistArray`'s `shape_`/`pmap_` are `shared_ptr<const>`
/// (`tensor_impl.h`), a shadow can share them with the primal at no copy cost.
///
/// The central design choice is the **symbolic zero**: an adjoint slot is left
/// *null* until something is first written into it, exactly like PyTorch's
/// lazily-allocated `.grad` and JAX's symbolic zeros. This:
///   * sidesteps the `SparsePolicy` `fill(0)` footgun (zero-seeding a sparse
///     array yields an *empty* array because the shape rejects below-threshold
///     tiles — CLAUDE.md), and
///   * gives "unconnected input ⇒ zero gradient" for free without spending
///     memory on accumulators that are never touched.
///
/// **B3 — cotangent-sparsity policy (decision, pinned in Phase 0).** For
/// `SparsePolicy`, an adjoint's sparsity is governed by *its own* per-tile
/// Frobenius norms vs the global threshold, which need not match the primal's
/// pattern. To keep gradients reproducible in *structure* (not merely in value)
/// the policy is:
///   1. **Lower bound = primal operand's shape.** A materialized adjoint mirror
///      inherits the corresponding primal operand's shape (`make_shadow` builds
///      it from `arg.shape()`), so every tile that could carry gradient has a
///      slot. This is the deterministic part — it does not depend on the
///      (data-dependent) norms of the accumulated cotangent.
///   2. **Re-screen by norm after accumulation.** Once contributions are summed
///      in, below-threshold tiles may be dropped by the ordinary `SparseShape`
///      screening, exactly as for any sparse result. Re-screening *narrows* the
///      structure but never widens it past the step-1 lower bound, so the result
///      is a deterministic function of the operand shapes, not of accumulation
///      order.
/// Phase 0 is dense, so only step 1 (already implemented by `make_shadow`) is
/// exercised here; the symbolic-zero `Cotangent` path defers materialization
/// until first write and so trivially satisfies the lower-bound rule. Step 2's
/// explicit re-screen is Phase 1 work, to be tested on block-sparse inputs.

namespace TiledArray::ad {

/// Construct a structurally-identical, *empty* mirror of `arg`.
///
/// The result shares `arg`'s tiled range, shape, and pmap (the latter two via
/// the primal's `shared_ptr<const>` members, so this is cheap) but has no tiles
/// set. It is the materialized accumulator one would use when a real zero array
/// is genuinely required; the default adjoint-accumulation path uses the
/// symbolic zero of `Cotangent` instead and never calls this.
///
/// \note This deliberately does **not** `fill(0)`: doing so would empty a
///       `SparsePolicy` array (the CLAUDE.md sparse-fill footgun).
template <typename Array>
Array make_shadow(const Array& arg) {
  return Array(arg.world(), arg.trange(), arg.shape(), arg.pmap());
}

/// A cotangent (adjoint) accumulator with a symbolic-zero state.
///
/// Holds either nothing (the symbolic zero) or a materialized `Array`. Reverse-
/// mode fan-in (`Ā += …`) is `accumulate`: the first contribution moves/copies
/// the delta in; subsequent contributions sum in place via the existing array
/// `+=` (the `add_to` family). Fan-out in the forward program — one array
/// consumed by several ops — is precisely what these summed contributions
/// encode (Part A table's `+=`).
template <typename Array>
class Cotangent {
 public:
  Cotangent() = default;

  /// \return true if this is still the symbolic zero (nothing accumulated yet)
  bool is_zero() const noexcept { return !value_.has_value(); }

  /// Fan-in one cotangent contribution.
  ///
  /// First call moves the delta in (no arithmetic); later calls accumulate in
  /// place. `delta` is consumed.
  void accumulate(Array delta) {
    if (!value_) {
      value_ = std::move(delta);
    } else {
      const std::string annot =
          ad::detail::canonical_annotation(value_->trange().rank());
      // in-place add_to via the expression engine: Ā += delta
      (*value_)(annot) += delta(annot);
    }
  }

  /// Access the materialized cotangent. Precondition: `!is_zero()`.
  const Array& value() const {
    TA_ASSERT(value_.has_value());
    return *value_;
  }

  /// Lower the symbolic zero to a concrete value, returning the accumulated
  /// array if non-zero or else an explicit structural zero shaped like `like`.
  ///
  /// Most callers should branch on `is_zero()` and propagate the symbolic zero
  /// instead; this exists for the few places (e.g. tests) that want a concrete
  /// array unconditionally.
  Array value_or_zero(const Array& like) const {
    if (value_) return *value_;
    // 0 * like routes through the engine and yields the policy-correct zero:
    // explicit zero tiles for DensePolicy, an empty array for SparsePolicy.
    return ad::scale(like, typename Array::numeric_type{0});
  }

 private:
  std::optional<Array> value_;
};

}  // namespace TiledArray::ad

#endif  // TILEDARRAY_AD_SHADOW_H__INCLUDED
