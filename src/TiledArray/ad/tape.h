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
 *  A define-by-run reverse-mode tape for TiledArray's AD layer.
 */

#ifndef TILEDARRAY_AD_TAPE_H__INCLUDED
#define TILEDARRAY_AD_TAPE_H__INCLUDED

#include <cmath>
#include <complex>
#include <functional>
#include <limits>
#include <optional>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include <TiledArray/ad/ops.h>
#include <TiledArray/ad/shadow.h>
#include <TiledArray/error.h>

/// \file tape.h
///
/// Reverse-mode (VJP) tape over the B1 primitives (autodiff plan B4, reverse
/// component), covering the full Part A table.
///
/// This is *define-by-run*: every primitive call appends a node carrying a
/// closure for its VJP, so host control flow (`if`/loops) is handled for free —
/// the tape records the branch actually taken — and reverse *issue order* is a
/// valid topological order, so `backward` simply walks the nodes in reverse.
///
/// Reverse-mode semantics designed in (each a documented failure mode in
/// PyTorch's autograd / TF's tape):
///
///  * **Activity tracking.** A `Var`/`ScalarVar` carries an `active` flag
///    (PyTorch's `requires_grad`). An op whose operands are all inactive
///    records nothing and yields an inactive result; only active operands get
///    a cotangent accumulated. This keeps constants (integrals, fixed
///    intermediates) from each acquiring a shadow array — the ~2× memory tax
///    the project exists to avoid.
///  * **Symbolic-zero shadows (B2).** Adjoint slots are `Cotangent`s (array)
///    or `std::optional` scalars: null until first written, so an unconnected
///    input has a zero gradient with nothing allocated.
///  * **Residual freeing.** Each node closure is destroyed as `backward`
///    consumes it, releasing its saved operand handles so tile storage frees
///    eagerly mid-pass.
///  * **Frozen-operand guard.** TA arrays are shallow-copy handles, so a saved
///    residual aliases live user data; mutating an operand in place between
///    record and `backward()` would silently corrupt the gradient. The V1 rule
///    is *operands of recorded ops are frozen until `backward()`*; it is
///    enforced cheaply by fingerprinting each saved residual's squared norm at
///    record time and re-checking at `backward` (the lightweight analogue of
///    PyTorch's version counters). Disable with `set_freeze_guard(false)`.
///
/// Scalars (reduction outputs) live in a separate id space from arrays; their
/// cotangents are stored as `numeric_type` (a real reduction's cotangent is a
/// real value carried in that type).

namespace TiledArray::ad {

namespace detail {

template <typename>
struct is_std_complex : std::false_type {};
template <typename T>
struct is_std_complex<std::complex<T>> : std::true_type {};

/// Scalar conjugation: `std::conj` for complex, identity for real.
template <typename S>
S conj_scalar(const S& s) {
  if constexpr (is_std_complex<S>::value) {
    return std::conj(s);
  } else {
    return s;
  }
}

}  // namespace detail

/// Reverse-mode tape.
///
/// Non-copyable / non-movable: `Var`s reference it by pointer and node closures
/// capture it, so its address must be stable for its whole lifetime.
template <typename Array>
class Tape {
 public:
  using array_type = Array;
  using numeric_type = typename Array::numeric_type;
  using scalar_type = typename Array::scalar_type;
  using cotangent_type = Cotangent<Array>;
  using node_type = std::function<void()>;

  /// Whether the frozen-operand guard applies. It fingerprints a residual by
  /// its squared norm, which must be a plain `scalar_type`. For a leaf `Array`
  /// (a `DistArray`) it is; for a `Dual<...>` element (forward-over-reverse)
  /// `squared_norm` returns a `DualScalar`, so the guard compiles out there.
  static constexpr bool guardable =
      std::is_same_v<decltype(ad::squared_norm(std::declval<const Array&>())),
                     scalar_type>;

  Tape() = default;
  Tape(const Tape&) = delete;
  Tape& operator=(const Tape&) = delete;
  Tape(Tape&&) = delete;
  Tape& operator=(Tape&&) = delete;

  /// Enable/disable the frozen-operand guard (on by default).
  void set_freeze_guard(bool on) noexcept { freeze_guard_ = on; }

  /// Register a new array variable; its adjoint starts as the symbolic zero.
  std::size_t make_variable() {
    const std::size_t id = adjoints_.size();
    adjoints_.emplace_back();
    return id;
  }

  /// Register a new scalar variable; its adjoint starts as the symbolic zero.
  std::size_t make_scalar_variable() {
    const std::size_t id = scalar_adjoints_.size();
    scalar_adjoints_.emplace_back();
    return id;
  }

  /// Access an array variable's accumulated cotangent (possibly symbolic zero).
  cotangent_type& adjoint(std::size_t id) { return adjoints_[id]; }
  const cotangent_type& adjoint(std::size_t id) const { return adjoints_[id]; }

  /// Access a scalar variable's accumulated cotangent (nullopt = symbolic zero).
  const std::optional<numeric_type>& scalar_adjoint(std::size_t id) const {
    return scalar_adjoints_[id];
  }

  /// Fan-in one array cotangent contribution into variable `id`.
  void accumulate_array(std::size_t id, Array delta) {
    adjoints_[id].accumulate(std::move(delta));
  }

  /// Fan-in one scalar cotangent contribution into scalar variable `id`.
  void accumulate_scalar(std::size_t id, numeric_type delta) {
    auto& slot = scalar_adjoints_[id];
    slot = slot ? (*slot + delta) : delta;
  }

  /// Record a VJP node (a closure run in reverse during `backward`).
  void push_node(node_type node) { nodes_.push_back(std::move(node)); }

  /// Register a saved residual for the frozen-operand guard.
  void register_guard(const Array& residual) {
    if constexpr (guardable) {
      if (freeze_guard_)
        guards_.emplace_back(residual, ad::squared_norm(residual));
    }
  }

  /// Seed an array output's adjoint and run the reverse pass.
  void backward(std::size_t seed_id, Array seed) {
    verify_guards();
    accumulate_array(seed_id, std::move(seed));
    run_reverse();
  }

  /// Seed a scalar output's adjoint (default `s̄ = 1`) and run the reverse pass.
  void backward(std::size_t seed_id, numeric_type seed = numeric_type{1}) {
    verify_guards();
    accumulate_scalar(seed_id, seed);
    run_reverse();
  }

 private:
  /// Re-check every saved residual's fingerprint; throw if any operand was
  /// mutated in place since it was recorded.
  void verify_guards() {
    if constexpr (guardable) {
      for (const auto& [residual, norm0] : guards_) {
        const scalar_type norm1 = ad::squared_norm(residual);
        const scalar_type scale = std::max<scalar_type>(
            {scalar_type{1}, std::abs(norm0), std::abs(norm1)});
        if (std::abs(norm1 - norm0) >
            scale * std::numeric_limits<scalar_type>::epsilon() * 64) {
          TA_EXCEPTION(
              "TiledArray::ad: a recorded operand was mutated before "
              "backward() (frozen-operand guard). Operands of recorded ops "
              "must not be modified in place until backward() completes.");
        }
      }
    }
  }

  /// Walk nodes in reverse, freeing each closure as it is consumed.
  void run_reverse() {
    for (auto it = nodes_.rbegin(); it != nodes_.rend(); ++it) {
      (*it)();
      *it = nullptr;  // release residual handles eagerly
    }
    nodes_.clear();
    guards_.clear();
  }

  std::vector<cotangent_type> adjoints_;
  std::vector<std::optional<numeric_type>> scalar_adjoints_;
  std::vector<node_type> nodes_;
  std::vector<std::pair<Array, scalar_type>> guards_;
  bool freeze_guard_ = true;
};

/// A reverse-mode array variable: a primal array plus its identity on a tape.
///
/// Shallow-copy of the primal is intentional — `value` aliases the live array
/// (the residual); see the freeze policy in the file header.
template <typename Array>
struct Var {
  Tape<Array>* tape = nullptr;
  std::size_t id = 0;
  Array value;
  bool active = true;
};

/// A reverse-mode scalar variable (a reduction output).
template <typename Array>
struct ScalarVar {
  Tape<Array>* tape = nullptr;
  std::size_t id = 0;
  typename Array::numeric_type value{};
  bool active = true;
};

/// Register an existing array as a leaf (input) variable on `tape`.
///
/// \param requires_grad if false the leaf is a constant: it gets no shadow and
///        ops consuming only constants record nothing (activity tracking).
template <typename Array>
Var<Array> make_leaf(Tape<Array>& tape, Array value, bool requires_grad = true) {
  const std::size_t id = requires_grad ? tape.make_variable() : 0;
  return Var<Array>{&tape, id, std::move(value), requires_grad};
}

// ---------------------------------------------------------------------------
// Array -> array primitives
// ---------------------------------------------------------------------------

/// Reverse-mode `contract` `C = factor*A*B`.
///
/// VJP (Part A, fused-`factor` carried as `conj(factor)`):
///   Ā += conj(factor)·C̄·conj(B);  B̄ += conj(factor)·conj(A)·C̄.
template <typename Array>
Var<Array> contract(const Var<Array>& a, const Var<Array>& b,
                    const std::string& a_annot, const std::string& b_annot,
                    const std::string& c_annot,
                    typename Array::numeric_type factor = 1) {
  Array c = ad::contract(a.value, b.value, a_annot, b_annot, c_annot, factor);
  const bool active = a.active || b.active;
  if (!active) return Var<Array>{a.tape, 0, std::move(c), false};

  Tape<Array>* tape = a.tape;
  const std::size_t c_id = tape->make_variable();
  const auto cfactor = detail::conj_scalar(factor);
  const std::size_t a_id = a.id, b_id = b.id;
  const bool aa = a.active, ba = b.active;
  if (aa) tape->register_guard(b.value);
  if (ba) tape->register_guard(a.value);
  Array A = a.value, B = b.value;
  tape->push_node([=]() {
    auto& cbar = tape->adjoint(c_id);
    if (cbar.is_zero()) return;
    const Array& C = cbar.value();
    // conj(B), conj(A) implement the complex VJP; no-ops for real arrays.
    if (aa)
      tape->accumulate_array(
          a_id, ad::contract(C, ad::conj(B), c_annot, b_annot, a_annot, cfactor));
    if (ba)
      tape->accumulate_array(
          b_id, ad::contract(ad::conj(A), C, a_annot, c_annot, b_annot, cfactor));
  });
  return Var<Array>{tape, c_id, std::move(c), true};
}

namespace detail {

/// Common scaffold for a binary array->array op: compute primal, handle
/// activity, register the node. `vjp` receives the result cotangent `C` and
/// must accumulate into the active operands.
template <typename Array, typename VJP>
Var<Array> record_binary(const Var<Array>& a, const Var<Array>& b, Array primal,
                         VJP vjp) {
  const bool active = a.active || b.active;
  if (!active) return Var<Array>{a.tape, 0, std::move(primal), false};
  Tape<Array>* tape = a.tape;
  const std::size_t c_id = tape->make_variable();
  tape->push_node([tape, c_id, vjp = std::move(vjp)]() {
    auto& cbar = tape->adjoint(c_id);
    if (cbar.is_zero()) return;
    vjp(cbar.value());
  });
  return Var<Array>{tape, c_id, std::move(primal), true};
}

/// Common scaffold for a unary array->array op.
template <typename Array, typename VJP>
Var<Array> record_unary(const Var<Array>& a, Array primal, VJP vjp) {
  if (!a.active) return Var<Array>{a.tape, 0, std::move(primal), false};
  Tape<Array>* tape = a.tape;
  const std::size_t c_id = tape->make_variable();
  tape->push_node([tape, c_id, vjp = std::move(vjp)]() {
    auto& cbar = tape->adjoint(c_id);
    if (cbar.is_zero()) return;
    vjp(cbar.value());
  });
  return Var<Array>{tape, c_id, std::move(primal), true};
}

}  // namespace detail

/// Reverse-mode `add`: Ā += C̄; B̄ += C̄.
template <typename Array>
Var<Array> add(const Var<Array>& a, const Var<Array>& b) {
  Tape<Array>* tape = a.tape;
  const std::size_t a_id = a.id, b_id = b.id;
  const bool aa = a.active, ba = b.active;
  return detail::record_binary(a, b, ad::add(a.value, b.value),
                               [=](const Array& C) {
                                 if (aa) tape->accumulate_array(a_id, C);
                                 if (ba) tape->accumulate_array(b_id, C);
                               });
}

/// Reverse-mode `subt`: Ā += C̄; B̄ += -C̄.
template <typename Array>
Var<Array> subt(const Var<Array>& a, const Var<Array>& b) {
  Tape<Array>* tape = a.tape;
  const std::size_t a_id = a.id, b_id = b.id;
  const bool aa = a.active, ba = b.active;
  return detail::record_binary(
      a, b, ad::subt(a.value, b.value), [=](const Array& C) {
        if (aa) tape->accumulate_array(a_id, C);
        if (ba)
          tape->accumulate_array(
              b_id, ad::scale(C, typename Array::numeric_type{-1}));
      });
}

/// Reverse-mode Hadamard `mult`: Ā += C̄∘conj(B); B̄ += conj(A)∘C̄.
template <typename Array>
Var<Array> mult(const Var<Array>& a, const Var<Array>& b) {
  Tape<Array>* tape = a.tape;
  const std::size_t a_id = a.id, b_id = b.id;
  const bool aa = a.active, ba = b.active;
  if (aa) tape->register_guard(b.value);
  if (ba) tape->register_guard(a.value);
  Array A = a.value, B = b.value;
  return detail::record_binary(a, b, ad::mult(a.value, b.value),
                               [=](const Array& C) {
                                 if (aa)
                                   tape->accumulate_array(
                                       a_id, ad::mult(C, ad::conj(B)));
                                 if (ba)
                                   tape->accumulate_array(
                                       b_id, ad::mult(ad::conj(A), C));
                               });
}

/// Reverse-mode `scale` by constant α: Ā += conj(α)·C̄.
template <typename Array>
Var<Array> scale(const Var<Array>& a, typename Array::numeric_type alpha) {
  Tape<Array>* tape = a.tape;
  const std::size_t a_id = a.id;
  const auto calpha = detail::conj_scalar(alpha);
  return detail::record_unary(a, ad::scale(a.value, alpha),
                              [=](const Array& C) {
                                tape->accumulate_array(a_id, ad::scale(C, calpha));
                              });
}

/// Reverse-mode `permute`: Ā += P⁻¹(C̄).
template <typename Array>
Var<Array> permute(const Var<Array>& a, const std::string& in_annot,
                   const std::string& out_annot) {
  Tape<Array>* tape = a.tape;
  const std::size_t a_id = a.id;
  return detail::record_unary(
      a, ad::permute(a.value, in_annot, out_annot), [=](const Array& C) {
        // inverse permutation: read C under out_annot, write under in_annot
        tape->accumulate_array(a_id, ad::permute(C, out_annot, in_annot));
      });
}

/// Reverse-mode `conj` (antilinear): Ā += conj(C̄).
template <typename Array>
Var<Array> conj(const Var<Array>& a) {
  Tape<Array>* tape = a.tape;
  const std::size_t a_id = a.id;
  return detail::record_unary(a, ad::conj(a.value), [=](const Array& C) {
    tape->accumulate_array(a_id, ad::conj(C));
  });
}

/// Reverse-mode unary `elementwise` f: Ā += conj(f'(A))∘C̄.
template <typename Array, typename F, typename DF>
Var<Array> elementwise(const Var<Array>& a, F f, DF df) {
  Tape<Array>* tape = a.tape;
  const std::size_t a_id = a.id;
  if (a.active) tape->register_guard(a.value);
  Array A = a.value;
  return detail::record_unary(
      a, ad::elementwise(a.value, f, df), [=](const Array& C) {
        Array dfA = ad::elementwise(A, df, df);  // f'(A) elementwise
        tape->accumulate_array(a_id, ad::mult(ad::conj(dfA), C));
      });
}

// ---------------------------------------------------------------------------
// Array -> scalar reductions
// ---------------------------------------------------------------------------

namespace detail {

/// Scaffold for a reduction over a single operand returning a scalar var.
template <typename Array, typename VJP>
ScalarVar<Array> record_reduction(const Var<Array>& a,
                                  typename Array::numeric_type primal,
                                  VJP vjp) {
  if (!a.active)
    return ScalarVar<Array>{a.tape, 0, primal, false};
  Tape<Array>* tape = a.tape;
  const std::size_t s_id = tape->make_scalar_variable();
  tape->push_node([tape, s_id, vjp = std::move(vjp)]() {
    const auto& sbar = tape->scalar_adjoint(s_id);
    if (!sbar) return;
    vjp(*sbar);
  });
  return ScalarVar<Array>{tape, s_id, primal, true};
}

}  // namespace detail

/// Reverse-mode `sum`: Ā += s̄ broadcast over A's structure.
template <typename Array>
ScalarVar<Array> sum(const Var<Array>& a) {
  using S = typename Array::numeric_type;
  Tape<Array>* tape = a.tape;
  const std::size_t a_id = a.id;
  if (a.active) tape->register_guard(a.value);
  Array A = a.value;
  return detail::record_reduction(a, ad::sum(a.value), [=](S sbar) {
    tape->accumulate_array(
        a_id, ad::elementwise(
                  A, [sbar](S) { return sbar; }, [](S) { return S{0}; }));
  });
}

/// Reverse-mode `trace`: Ā += s̄ scattered onto the diagonal.
template <typename Array>
ScalarVar<Array> trace(const Var<Array>& a) {
  using S = typename Array::numeric_type;
  Tape<Array>* tape = a.tape;
  const std::size_t a_id = a.id;
  if (a.active) tape->register_guard(a.value);
  Array A = a.value;
  return detail::record_reduction(a, ad::trace(a.value), [=](S sbar) {
    Array g(A.world(), A.trange(), A.shape(), A.pmap());
    g.init_elements([sbar](const auto& idx) {
      return idx[0] == idx[1] ? sbar : S{0};
    });
    tape->accumulate_array(a_id, g);
  });
}

/// Reverse-mode `squared_norm`: Ā += 2·s̄·A (s̄ real).
template <typename Array>
ScalarVar<Array> squared_norm(const Var<Array>& a) {
  using S = typename Array::numeric_type;
  Tape<Array>* tape = a.tape;
  const std::size_t a_id = a.id;
  if (a.active) tape->register_guard(a.value);
  Array A = a.value;
  return detail::record_reduction(
      a, S{ad::squared_norm(a.value)}, [=](S sbar) {
        tape->accumulate_array(a_id, ad::scale(A, S{2} * sbar));
      });
}

/// Reverse-mode `norm2`: Ā += (s̄/‖A‖)·A (undefined at A = 0).
template <typename Array>
ScalarVar<Array> norm2(const Var<Array>& a) {
  using S = typename Array::numeric_type;
  const typename Array::scalar_type n = ad::norm2(a.value);
  Tape<Array>* tape = a.tape;
  const std::size_t a_id = a.id;
  if (a.active) tape->register_guard(a.value);
  Array A = a.value;
  return detail::record_reduction(a, S{n}, [=](S sbar) {
    if (n == typename Array::scalar_type{0})
      TA_EXCEPTION("TiledArray::ad: norm2 VJP is undefined at A = 0");
    tape->accumulate_array(a_id, ad::scale(A, sbar / S{n}));
  });
}

// ---------------------------------------------------------------------------
// (Array, array) -> scalar reductions
// ---------------------------------------------------------------------------

namespace detail {

/// Scaffold for a reduction over two operands returning a scalar var.
template <typename Array, typename VJP>
ScalarVar<Array> record_binary_reduction(const Var<Array>& a,
                                         const Var<Array>& b,
                                         typename Array::numeric_type primal,
                                         VJP vjp) {
  const bool active = a.active || b.active;
  if (!active) return ScalarVar<Array>{a.tape, 0, primal, false};
  Tape<Array>* tape = a.tape;
  const std::size_t s_id = tape->make_scalar_variable();
  tape->push_node([tape, s_id, vjp = std::move(vjp)]() {
    const auto& sbar = tape->scalar_adjoint(s_id);
    if (!sbar) return;
    vjp(*sbar);
  });
  return ScalarVar<Array>{tape, s_id, primal, true};
}

}  // namespace detail

/// Reverse-mode bilinear `dot`: Ā += s̄·conj(B); B̄ += s̄·conj(A).
template <typename Array>
ScalarVar<Array> dot(const Var<Array>& a, const Var<Array>& b) {
  using S = typename Array::numeric_type;
  Tape<Array>* tape = a.tape;
  const std::size_t a_id = a.id, b_id = b.id;
  const bool aa = a.active, ba = b.active;
  if (aa) tape->register_guard(b.value);
  if (ba) tape->register_guard(a.value);
  Array A = a.value, B = b.value;
  return detail::record_binary_reduction(
      a, b, ad::dot(a.value, b.value), [=](S sbar) {
        if (aa) tape->accumulate_array(a_id, ad::scale(ad::conj(B), sbar));
        if (ba) tape->accumulate_array(b_id, ad::scale(ad::conj(A), sbar));
      });
}

/// Reverse-mode sesquilinear `inner_product`: Ā += conj(s̄)·B; B̄ += s̄·A.
template <typename Array>
ScalarVar<Array> inner_product(const Var<Array>& a, const Var<Array>& b) {
  using S = typename Array::numeric_type;
  Tape<Array>* tape = a.tape;
  const std::size_t a_id = a.id, b_id = b.id;
  const bool aa = a.active, ba = b.active;
  if (aa) tape->register_guard(b.value);
  if (ba) tape->register_guard(a.value);
  Array A = a.value, B = b.value;
  return detail::record_binary_reduction(
      a, b, ad::inner_product(a.value, b.value), [=](S sbar) {
        if (aa)
          tape->accumulate_array(a_id, ad::scale(B, detail::conj_scalar(sbar)));
        if (ba) tape->accumulate_array(b_id, ad::scale(A, sbar));
      });
}

}  // namespace TiledArray::ad

#endif  // TILEDARRAY_AD_TAPE_H__INCLUDED
