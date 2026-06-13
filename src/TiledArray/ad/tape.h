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
 *  tape.h
 *  A minimal define-by-run reverse-mode tape for TiledArray's AD layer.
 */

#ifndef TILEDARRAY_AD_TAPE_H__INCLUDED
#define TILEDARRAY_AD_TAPE_H__INCLUDED

#include <complex>
#include <functional>
#include <string>
#include <vector>

#include <TiledArray/ad/ops.h>
#include <TiledArray/ad/shadow.h>

/// \file tape.h
///
/// Reverse-mode (VJP) tape over the B1 primitives (autodiff plan B4, reverse
/// component). Phase 0 (de-risk) records only `contract` so its hand-written
/// VJP can be verified end-to-end; the remaining primitives' nodes are Phase 1.
///
/// This is *define-by-run*: every primitive call appends a node carrying a
/// closure for its VJP, so host control flow (`if`/loops) is handled for free —
/// the tape records the branch actually taken — and reverse *issue order* is a
/// valid topological order, so `backward` simply walks the nodes in reverse.
///
/// Residual policy (Part A / B4): each VJP closure saves shallow-copy handles to
/// the operands it needs (`contract` saves `A` and `B`). Because `DistArray` is
/// a shallow-copy handle, a saved residual aliases live user data — the V1 rule
/// is that **operands of recorded ops are frozen until `backward()`**. Mutating
/// an operand in between silently corrupts the gradient. The cheap version of
/// PyTorch's version counters (a per-variable write guard) is Phase 1 work.

namespace TiledArray::ad {

namespace detail {

template <typename>
struct is_std_complex : std::false_type {};
template <typename T>
struct is_std_complex<std::complex<T>> : std::true_type {};

/// Scalar conjugation that is a no-op for real types and `std::conj` for
/// `std::complex`. The VJPs in the Part A table carry `conj(factor)`, which for
/// real arrays reduces to `factor`.
template <typename S>
S conj_scalar(const S& s) {
  if constexpr (is_std_complex<S>::value) {
    return std::conj(s);
  } else {
    return s;
  }
}

}  // namespace detail

/// Reverse-mode tape: a flat list of adjoint slots plus VJP closures.
///
/// Variables are referred to by integer id (an index into `adjoints_`). Adjoint
/// slots are `Cotangent`s, i.e. symbolic zero until first written.
template <typename Array>
class Tape {
 public:
  using array_type = Array;
  using numeric_type = typename Array::numeric_type;
  using cotangent_type = Cotangent<Array>;
  using node_type = std::function<void(std::vector<cotangent_type>&)>;

  /// Register a new variable (leaf or intermediate) and return its id. Its
  /// adjoint starts as the symbolic zero.
  std::size_t make_variable() {
    const std::size_t id = adjoints_.size();
    adjoints_.emplace_back();
    return id;
  }

  /// Record `C = factor * A * B` and return the result variable's id.
  ///
  /// Saves shallow-copy residuals `a`, `b` (see the residual/freeze policy in
  /// the file header) and registers the two-contraction VJP:
  ///   Ā(aA) += conj(factor) * C̄(aC) * B(aB)     [contract C̄, B → aA]
  ///   B̄(aB) += conj(factor) * A(aA) * C̄(aC)     [contract A, C̄ → aB]
  /// matching Giles's real `Ā = C̄ Bᵀ`, `B̄ = Aᵀ C̄` with the fused-gemm
  /// `factor` carried into both adjoints as `conj(factor)`.
  std::size_t record_contract(std::size_t a_id, Array a, std::size_t b_id,
                              Array b, std::string a_annot,
                              std::string b_annot, std::string c_annot,
                              numeric_type factor) {
    const std::size_t c_id = make_variable();
    const numeric_type cfactor = detail::conj_scalar(factor);
    nodes_.emplace_back([=](std::vector<cotangent_type>& adj) {
      // No upstream cotangent reached this op ⇒ it contributes nothing
      // (symbolic-zero short circuit; "unconnected ⇒ zero gradient").
      if (adj[c_id].is_zero()) return;
      const Array& c_bar = adj[c_id].value();
      adj[a_id].accumulate(
          ad::contract(c_bar, b, c_annot, b_annot, a_annot, cfactor));
      adj[b_id].accumulate(
          ad::contract(a, c_bar, a_annot, c_annot, b_annot, cfactor));
    });
    return c_id;
  }

  /// Seed the output adjoint and walk the tape in reverse, accumulating into
  /// operand adjoints. Each node is dropped after it runs, releasing its
  /// residual handles so tile storage can free eagerly mid-pass.
  void backward(std::size_t seed_id, Array seed) {
    adjoints_[seed_id].accumulate(std::move(seed));
    for (auto it = nodes_.rbegin(); it != nodes_.rend(); ++it) {
      (*it)(adjoints_);
    }
    nodes_.clear();
  }

  /// Access a variable's accumulated cotangent (possibly the symbolic zero).
  cotangent_type& adjoint(std::size_t id) { return adjoints_[id]; }
  const cotangent_type& adjoint(std::size_t id) const { return adjoints_[id]; }

 private:
  std::vector<cotangent_type> adjoints_;
  std::vector<node_type> nodes_;
};

/// A reverse-mode variable: a primal array plus its identity on a tape.
///
/// Shallow-copy of the primal `DistArray` is intentional — the `value` aliases
/// the live array (the residual); see the freeze policy in the file header.
template <typename Array>
struct Var {
  Tape<Array>* tape = nullptr;
  std::size_t id = 0;
  Array value;
};

/// Register an existing array as a leaf (input) variable on `tape`.
template <typename Array>
Var<Array> make_leaf(Tape<Array>& tape, Array value) {
  const std::size_t id = tape.make_variable();
  return Var<Array>{&tape, id, std::move(value)};
}

/// Reverse-mode `contract` over `Var`s: computes the primal and records the
/// node. The only differentiable entry point Phase 0 exposes on the tape.
template <typename Array>
Var<Array> contract(const Var<Array>& a, const Var<Array>& b,
                    const std::string& a_annot, const std::string& b_annot,
                    const std::string& c_annot,
                    typename Array::numeric_type factor = 1) {
  Array c = ad::contract(a.value, b.value, a_annot, b_annot, c_annot, factor);
  const std::size_t c_id =
      a.tape->record_contract(a.id, a.value, b.id, b.value, a_annot, b_annot,
                              c_annot, factor);
  return Var<Array>{a.tape, c_id, std::move(c)};
}

}  // namespace TiledArray::ad

#endif  // TILEDARRAY_AD_TAPE_H__INCLUDED
