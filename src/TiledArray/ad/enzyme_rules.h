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
 *  enzyme_rules.h
 *  Enzyme custom-derivative rules for TiledArray's AD primitives (definitions).
 */

#ifndef TILEDARRAY_AD_ENZYME_RULES_H__INCLUDED
#define TILEDARRAY_AD_ENZYME_RULES_H__INCLUDED

#include <TiledArray/config.h>

#ifdef TILEDARRAY_HAS_ENZYME

#include <complex>
#include <functional>
#include <type_traits>
#include <utility>

// The umbrella header must precede the ad/ headers: ops.h pulls in foreach.h and
// the expression DSL (TsrExpr), whose full definitions live behind tiledarray.h.
#include <tiledarray.h>

#include <TiledArray/ad/ops.h>

/// \file enzyme_rules.h
///
/// This header *defines* the Enzyme custom rules for TiledArray's B1 functional
/// primitives (autodiff plan B5) — both the reverse (augmented-forward +
/// gradient) and forward (JVP derivative) rules — together with the
/// `__enzyme_register_gradient_*` / `__enzyme_register_derivative_*` globals
/// that bind them.
///
/// **Why a header, not a single library TU.** Enzyme reads its type metadata
/// from the *bodies* of all custom-rule functions at the point it transforms a
/// differentiated module — a declaration-only shim makes the plugin abort in
/// type analysis (`isa<>` on a null `ConstantAsMetadata`). So every TU that
/// differentiates host C++ over these primitives must compile the rule bodies
/// itself. They are `static` (internal linkage) so several differentiated TUs
/// can each include this header without colliding at link, and `used` so `-O`
/// cannot strip the registration before the Enzyme pass runs.
///
/// **The derivative math is identical to the native reverse-mode tape (tape.h)**
/// and forward dual (dual.h): both implement the Part-A VJP/JVP table over the
/// same B1 ops, so the Enzyme path and the native path are two registrations of
/// one set of rules. The rule *bodies* here are hand-written derivatives that
/// Enzyme never differentiates — it only emits calls to them — so they may use
/// templated helpers and ordinary `DistArray` locals freely. (That freedom does
/// **not** extend to the *differentiated routine*: see below.)
///
/// **No annotation/scalar tax (P0.3).** Annotation strings and scale factors are
/// *baked into the registered symbol* via the `TA_AD_*_RULE` macros rather than
/// passed as runtime arguments. This matters for two reasons: (1) Enzyme's
/// `getDefaultFunctionTypeForGradient` makes every non-FP parameter `DUP_ARG`,
/// so a runtime `const char*` annotation would force a phantom shadow slot in
/// every aug/reverse/forward signature (a hand-sync footgun: a mismatch yields
/// "Bad function type of custom reverse pass"); (2) a by-value floating-point
/// parameter is treated as `OUT_DIFF` — its derivative is *appended to the
/// return* — which would break the fixed reverse signature. Baking the spec in
/// removes both: each shim is a fixed-arity function of only array (and scalar
/// I/O pointer) arguments, which is exactly what Enzyme custom rules want.
///
/// A differentiated TU includes this header, instantiates the spec-bearing rules
/// it needs (`TA_AD_CONTRACT_RULE`, `TA_AD_SCALE_RULE`, `TA_AD_PERMUTE_RULE`,
/// `TA_AD_ELEMENTWISE_RULE`), and writes its routine as a chain of these shims
/// with all intermediates and shadows *caller-allocated* — no local `DistArray`
/// lifetime in the differentiated body, because Enzyme cannot reverse a
/// `DistArray` constructor/destructor. See tests/ad_enzyme/energy.cpp.

// ---------------------------------------------------------------------------
// Attribute macros for the shims.
//
// TA_AD_PRIMAL marks the *primal* shims `noinline` (P0.1): a primal shim is the
// call site the Enzyme pass must still see to match it to a registered rule. At
// -O2+ (Release/RelWithDebInfo) the inliner would otherwise fold a shim into the
// host routine *before* the Enzyme pass runs, the matchable call site would
// vanish, and Enzyme would descend into ad::contract -> the MADWorld runtime /
// world.gop and abort. `noinline` makes the integration opt-level-independent.
// `used` keeps every rule/registration symbol from being stripped before the
// pass. The augmented/reverse/forward rule bodies are never call sites that must
// survive, so they take only `used`.
// ---------------------------------------------------------------------------
#define TA_AD_PRIMAL __attribute__((noinline, used))
#define TA_AD_RULE __attribute__((used))

namespace TiledArray::ad::edetail {

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

/// Real part: `std::real` for complex, identity for real. Returns the scalar
/// (non-complex) type either way.
template <typename S>
auto real_part(const S& s) {
  if constexpr (is_std_complex<S>::value) {
    return std::real(s);
  } else {
    return s;
  }
}

/// Fan-in one cotangent contribution into a shadow array, honoring the
/// symbolic-zero convention (B2): an *uninitialized* shadow is the zero adjoint,
/// so the first contribution is moved in and later ones are summed via the
/// policy-aware `ad::add`. Both caller-seeded input shadows (passed
/// default-constructed) and Enzyme-managed shadows arrive uninitialized, so this
/// one guard covers both.
template <typename Array>
void accumulate_into(Array* dst, Array delta) {
  if (!dst->is_initialized()) {
    *dst = std::move(delta);
  } else {
    *dst = TiledArray::ad::add(*dst, delta);
  }
}

/// Combine two forward (JVP) tangent contributions, each possibly the symbolic
/// zero (an uninitialized `Array`). Returns the symbolic zero if both are.
template <typename Array>
Array jvp_sum(Array x, Array y) {
  if (!x.is_initialized()) return y;
  if (!y.is_initialized()) return x;
  return TiledArray::ad::add(x, y);
}

/// One operand a reverse rule needs at backward time, held *either* materialized
/// (the default) *or* as a recompute recipe (the checkpoint/recompute seam,
/// P2.1).
///
/// **Why this exists.** Each augmented forward saves the operands its reverse
/// pass will read; those saved handles pin the operands' `ArrayImpl`s alive from
/// the augmented forward until the reverse runs — and `DistArray` destruction is
/// already deferred to the next fence (`array_impl.h` `lazy_deleter`). For a deep
/// contraction chain that pinned-operand set is the reverse-mode peak-memory
/// term. Enzyme's own min-cut cache-vs-recompute heuristic cannot see inside an
/// atomic custom rule, so *we* own the cache-vs-recompute decision: the
/// augmented-forward residual is the only place to make it. This type reserves
/// that seam now (plan B4) without changing today's behavior.
///
/// **Default path is bit-identical to holding the operand.** Constructing from
/// an `Array` stores the materialized handle and `get()` returns it directly —
/// the same shallow handle the old `BinaryResidual{a, b}` held, so the gradient
/// is unchanged. The recompute constructor stores a thunk instead; `get()`
/// materializes (and memoizes) it on first read, trading the pinned-operand
/// memory term for recomputation. The thunk must reproduce the operand the
/// reverse pass expects (the V1 frozen-operand rule still applies — see tape.h).
template <typename Array>
class Checkpoint {
 public:
  Checkpoint() = default;
  /// Hold the operand materialized (default, zero-overhead seam).
  Checkpoint(Array value) : stored_(std::move(value)) {}
  /// Hold a recipe to recompute the operand instead of the operand itself.
  explicit Checkpoint(std::function<Array()> recompute)
      : recompute_(std::move(recompute)) {}

  /// The operand, materializing and memoizing the recompute thunk on first read.
  const Array& get() const {
    if (!stored_.is_initialized() && recompute_) stored_ = recompute_();
    return stored_;
  }

 private:
  mutable Array stored_;
  std::function<Array()> recompute_;
};

/// Residual saved by an augmented forward that holds *both* operands (cheap
/// shallow `DistArray` handles that merely keep the impl alive, unless a
/// recompute recipe was stored instead — see `Checkpoint`). The operands must
/// not be mutated in place before the reverse pass runs (the V1 frozen-operand
/// rule, as in tape.h).
template <typename Array>
struct BinaryResidual {
  Checkpoint<Array> a, b;
};

/// Residual saved by an augmented forward that holds a single operand.
template <typename Array>
struct UnaryResidual {
  Checkpoint<Array> a;
};

// ===========================================================================
// Templated derivative engines (one per primitive). These are NOT registered;
// they carry the math so the per-type/per-spec shims stay one-liners. The VJPs
// mirror tape.h verbatim; the JVPs are the standard tangent-propagation forms.
// `ad::conj` is the identity on real arrays, so the same body is correct for
// real and complex.
// ===========================================================================

// ---- contract: C(cc) = A(aa) * B(bb) --------------------------------------
template <typename A>
void contract_primal(const A* a, const A* b, A* c, const char* aa,
                     const char* bb, const char* cc) {
  *c = TiledArray::ad::contract(*a, *b, aa, bb, cc);
}
template <typename A>
void* contract_aug(const A* a, const A* b, A* c, const char* aa, const char* bb,
                   const char* cc) {
  *c = TiledArray::ad::contract(*a, *b, aa, bb, cc);
  return new BinaryResidual<A>{*a, *b};
}
template <typename A>
void contract_rev(A* da, A* db, const A* dc, const char* aa, const char* bb,
                  const char* cc, void* tape) {
  namespace ad = TiledArray::ad;
  auto* t = static_cast<BinaryResidual<A>*>(tape);
  if (dc->is_initialized()) {  // zero output cotangent => no contribution
    // Ā += C̄·conj(B);  B̄ += conj(A)·C̄  (conj is a no-op for real arrays).
    accumulate_into(da, ad::contract(*dc, ad::conj(t->b.get()), cc, bb, aa));
    accumulate_into(db, ad::contract(ad::conj(t->a.get()), *dc, aa, cc, bb));
  }
  delete t;  // free this node's residual as it is consumed (P2.1): releases the
             // pinned operands now instead of at function exit, keeping the
             // reverse-pass peak-memory term to the live (not yet consumed) tail.
}
template <typename A>
void contract_fwd(const A* a, const A* da, const A* b, const A* db, A* c, A* dc,
                  const char* aa, const char* bb, const char* cc) {
  namespace ad = TiledArray::ad;
  *c = ad::contract(*a, *b, aa, bb, cc);
  // JVP: dC = dA·B + A·dB.
  A t1 = da->is_initialized() ? ad::contract(*da, *b, aa, bb, cc) : A();
  A t2 = db->is_initialized() ? ad::contract(*a, *db, aa, bb, cc) : A();
  *dc = jvp_sum(std::move(t1), std::move(t2));
}

// ---- add: C = A + B (linear, no residual) ---------------------------------
template <typename A>
void add_primal(const A* a, const A* b, A* c) {
  *c = TiledArray::ad::add(*a, *b);
}
template <typename A>
void* add_aug(const A* a, const A* b, A* c) {
  *c = TiledArray::ad::add(*a, *b);
  return nullptr;
}
template <typename A>
void add_rev(A* da, A* db, const A* dc, void*) {
  if (dc->is_initialized()) {
    accumulate_into(da, *dc);
    accumulate_into(db, *dc);
  }
}
template <typename A>
void add_fwd(const A* a, const A* da, const A* b, const A* db, A* c, A* dc) {
  *c = TiledArray::ad::add(*a, *b);
  A t1 = da->is_initialized() ? *da : A();
  A t2 = db->is_initialized() ? *db : A();
  *dc = jvp_sum(std::move(t1), std::move(t2));
}

// ---- subt: C = A - B (linear) ---------------------------------------------
template <typename A>
void subt_primal(const A* a, const A* b, A* c) {
  *c = TiledArray::ad::subt(*a, *b);
}
template <typename A>
void* subt_aug(const A* a, const A* b, A* c) {
  *c = TiledArray::ad::subt(*a, *b);
  return nullptr;
}
template <typename A>
void subt_rev(A* da, A* db, const A* dc, void*) {
  if (dc->is_initialized()) {
    accumulate_into(da, *dc);
    accumulate_into(db,
                    TiledArray::ad::scale(*dc, typename A::numeric_type{-1}));
  }
}
template <typename A>
void subt_fwd(const A* a, const A* da, const A* b, const A* db, A* c, A* dc) {
  namespace ad = TiledArray::ad;
  *c = ad::subt(*a, *b);
  A t1 = da->is_initialized() ? *da : A();
  A t2 = db->is_initialized() ? ad::scale(*db, typename A::numeric_type{-1})
                              : A();
  *dc = jvp_sum(std::move(t1), std::move(t2));
}

// ---- mult: C = A ∘ B (Hadamard) -------------------------------------------
template <typename A>
void mult_primal(const A* a, const A* b, A* c) {
  *c = TiledArray::ad::mult(*a, *b);
}
template <typename A>
void* mult_aug(const A* a, const A* b, A* c) {
  *c = TiledArray::ad::mult(*a, *b);
  return new BinaryResidual<A>{*a, *b};
}
template <typename A>
void mult_rev(A* da, A* db, const A* dc, void* tape) {
  namespace ad = TiledArray::ad;
  auto* t = static_cast<BinaryResidual<A>*>(tape);
  if (dc->is_initialized()) {
    accumulate_into(da, ad::mult(*dc, ad::conj(t->b.get())));
    accumulate_into(db, ad::mult(ad::conj(t->a.get()), *dc));
  }
  delete t;  // eager residual free as consumed (P2.1)
}
template <typename A>
void mult_fwd(const A* a, const A* da, const A* b, const A* db, A* c, A* dc) {
  namespace ad = TiledArray::ad;
  *c = ad::mult(*a, *b);
  A t1 = da->is_initialized() ? ad::mult(*da, *b) : A();
  A t2 = db->is_initialized() ? ad::mult(*a, *db) : A();
  *dc = jvp_sum(std::move(t1), std::move(t2));
}

// ---- conj: C = conj(A) (antilinear) ---------------------------------------
template <typename A>
void conj_primal(const A* a, A* c) {
  *c = TiledArray::ad::conj(*a);
}
template <typename A>
void* conj_aug(const A* a, A* c) {
  *c = TiledArray::ad::conj(*a);
  return nullptr;
}
template <typename A>
void conj_rev(A* da, const A* dc, void*) {
  if (dc->is_initialized())
    accumulate_into(da, TiledArray::ad::conj(*dc));  // Ā += conj(C̄)
}
template <typename A>
void conj_fwd(const A* a, const A* da, A* c, A* dc) {
  namespace ad = TiledArray::ad;
  *c = ad::conj(*a);
  *dc = da->is_initialized() ? ad::conj(*da) : A();  // dC = conj(dA)
}

// ---- scale: C = α·A (α baked into the symbol) -----------------------------
template <typename A>
void scale_primal(const A* a, A* c, typename A::numeric_type alpha) {
  *c = TiledArray::ad::scale(*a, alpha);
}
template <typename A>
void scale_rev(A* da, const A* dc, typename A::numeric_type alpha, void*) {
  if (dc->is_initialized())  // Ā += conj(α)·C̄
    accumulate_into(da, TiledArray::ad::scale(*dc, conj_scalar(alpha)));
}
template <typename A>
void scale_fwd(const A* da, A* dc, typename A::numeric_type alpha) {
  *dc = da->is_initialized() ? TiledArray::ad::scale(*da, alpha) : A();
}

// ---- permute: C(out) = A(in) (annotations baked) --------------------------
template <typename A>
void permute_primal(const A* a, A* c, const char* in, const char* out) {
  *c = TiledArray::ad::permute(*a, in, out);
}
template <typename A>
void* permute_aug(const A* a, A* c, const char* in, const char* out) {
  *c = TiledArray::ad::permute(*a, in, out);
  return nullptr;
}
template <typename A>
void permute_rev(A* da, const A* dc, const char* in, const char* out, void*) {
  if (dc->is_initialized())  // inverse permutation: read C̄ under out, write in
    accumulate_into(da, TiledArray::ad::permute(*dc, out, in));
}
template <typename A>
void permute_fwd(const A* a, const A* da, A* c, A* dc, const char* in,
                 const char* out) {
  namespace ad = TiledArray::ad;
  *c = ad::permute(*a, in, out);
  *dc = da->is_initialized() ? ad::permute(*da, in, out) : A();
}

// ---- elementwise: C(e) = f(A(e)) (f, f' baked) ----------------------------
template <typename A, typename F, typename DF>
void elementwise_primal(const A* a, A* c, F f, DF df) {
  *c = TiledArray::ad::elementwise(*a, f, df);
}
template <typename A, typename F, typename DF>
void* elementwise_aug(const A* a, A* c, F f, DF df) {
  *c = TiledArray::ad::elementwise(*a, f, df);
  return new UnaryResidual<A>{*a};
}
template <typename A, typename F, typename DF>
void elementwise_rev(A* da, const A* dc, F /*f*/, DF df, void* tape) {
  namespace ad = TiledArray::ad;
  auto* t = static_cast<UnaryResidual<A>*>(tape);
  if (dc->is_initialized()) {  // Ā += conj(f'(A)) ∘ C̄
    A dfA = ad::elementwise(t->a.get(), df, df);
    accumulate_into(da, ad::mult(ad::conj(dfA), *dc));
  }
  delete t;  // eager residual free as consumed (P2.1)
}
template <typename A, typename F, typename DF>
void elementwise_fwd(const A* a, const A* da, A* c, A* dc, F f, DF df) {
  namespace ad = TiledArray::ad;
  *c = ad::elementwise(*a, f, df);
  *dc = da->is_initialized() ? ad::mult(ad::elementwise(*a, df, df), *da) : A();
}

// ---- reductions: A -> scalar ----------------------------------------------
// `out` is the reduction's value; its derivative travels through `*dout` (a
// caller-owned scalar shadow, the DUP_ARG slot Enzyme makes for the pointer).

// sum: s = Σ A(e);  Ā += s̄ broadcast over A's structure.
template <typename A>
void sum_primal(const A* a, typename A::numeric_type* out) {
  *out = TiledArray::ad::sum(*a);
}
template <typename A>
void* sum_aug(const A* a, typename A::numeric_type* out) {
  *out = TiledArray::ad::sum(*a);
  return new UnaryResidual<A>{*a};
}
template <typename A>
void sum_rev(A* da, const typename A::numeric_type* dout, void* tape) {
  using S = typename A::numeric_type;
  auto* t = static_cast<UnaryResidual<A>*>(tape);
  const S sb = *dout;
  accumulate_into(da, TiledArray::ad::elementwise(
                          t->a.get(), [sb](S) { return sb; },
                          [](S) { return S{0}; }));
  delete t;  // eager residual free as consumed (P2.1)
}
template <typename A>
void sum_fwd(const A* a, const A* da, typename A::numeric_type* out,
             typename A::numeric_type* dout) {
  using S = typename A::numeric_type;
  *out = TiledArray::ad::sum(*a);
  *dout = da->is_initialized() ? TiledArray::ad::sum(*da) : S{0};
}

// trace: s = Σ A(i,i);  Ā += s̄ scattered onto the diagonal.
template <typename A>
void trace_primal(const A* a, typename A::numeric_type* out) {
  *out = TiledArray::ad::trace(*a);
}
template <typename A>
void* trace_aug(const A* a, typename A::numeric_type* out) {
  *out = TiledArray::ad::trace(*a);
  return new UnaryResidual<A>{*a};
}
template <typename A>
void trace_rev(A* da, const typename A::numeric_type* dout, void* tape) {
  using S = typename A::numeric_type;
  auto* t = static_cast<UnaryResidual<A>*>(tape);
  const S sb = *dout;
  const A& A0 = t->a.get();
  A g(A0.world(), A0.trange(), A0.shape(), A0.pmap());
  g.init_elements(
      [sb](const auto& idx) { return idx[0] == idx[1] ? sb : S{0}; });
  accumulate_into(da, g);
  delete t;  // eager residual free as consumed (P2.1)
}
template <typename A>
void trace_fwd(const A* a, const A* da, typename A::numeric_type* out,
               typename A::numeric_type* dout) {
  using S = typename A::numeric_type;
  *out = TiledArray::ad::trace(*a);
  *dout = da->is_initialized() ? TiledArray::ad::trace(*da) : S{0};
}

// squared_norm: s = Σ |A(e)|² (real);  Ā += 2·s̄·A.
template <typename A>
void sqnorm_primal(const A* a, typename A::scalar_type* out) {
  *out = TiledArray::ad::squared_norm(*a);
}
template <typename A>
void* sqnorm_aug(const A* a, typename A::scalar_type* out) {
  *out = TiledArray::ad::squared_norm(*a);
  return new UnaryResidual<A>{*a};
}
template <typename A>
void sqnorm_rev(A* da, const typename A::scalar_type* dout, void* tape) {
  using S = typename A::numeric_type;
  auto* t = static_cast<UnaryResidual<A>*>(tape);
  accumulate_into(da,
                  TiledArray::ad::scale(t->a.get(), static_cast<S>(2 * (*dout))));
  delete t;  // eager residual free as consumed (P2.1)
}
template <typename A>
void sqnorm_fwd(const A* a, const A* da, typename A::scalar_type* out,
                typename A::scalar_type* dout) {
  using R = typename A::scalar_type;
  *out = TiledArray::ad::squared_norm(*a);
  // ds = 2·Re<A, dA>  (inner_product conjugates the first argument).
  *dout = da->is_initialized()
              ? R{2} * real_part(TiledArray::ad::inner_product(*a, *da))
              : R{0};
}

// norm2: s = ‖A‖;  Ā += (s̄/‖A‖)·A  (undefined at A = 0).
template <typename A>
void norm2_primal(const A* a, typename A::scalar_type* out) {
  *out = TiledArray::ad::norm2(*a);
}
template <typename A>
void* norm2_aug(const A* a, typename A::scalar_type* out) {
  *out = TiledArray::ad::norm2(*a);
  return new UnaryResidual<A>{*a};
}
template <typename A>
void norm2_rev(A* da, const typename A::scalar_type* dout, void* tape) {
  using S = typename A::numeric_type;
  using R = typename A::scalar_type;
  auto* t = static_cast<UnaryResidual<A>*>(tape);
  const A& A0 = t->a.get();
  const R n = TiledArray::ad::norm2(A0);
  if (n == R{0})
    TA_EXCEPTION("TiledArray::ad (Enzyme): norm2 VJP is undefined at A = 0");
  accumulate_into(da, TiledArray::ad::scale(A0, static_cast<S>(*dout / n)));
  delete t;  // eager residual free as consumed (P2.1)
}
template <typename A>
void norm2_fwd(const A* a, const A* da, typename A::scalar_type* out,
               typename A::scalar_type* dout) {
  using R = typename A::scalar_type;
  const R n = TiledArray::ad::norm2(*a);
  *out = n;
  *dout = da->is_initialized()
              ? real_part(TiledArray::ad::inner_product(*a, *da)) / n
              : R{0};
}

// dot: s = Σ A(e)·B(e) (bilinear, no conj);
//   Ā += s̄·conj(B);  B̄ += s̄·conj(A).
template <typename A>
void dot_primal(const A* a, const A* b, typename A::numeric_type* out) {
  *out = TiledArray::ad::dot(*a, *b);
}
template <typename A>
void* dot_aug(const A* a, const A* b, typename A::numeric_type* out) {
  *out = TiledArray::ad::dot(*a, *b);
  return new BinaryResidual<A>{*a, *b};
}
template <typename A>
void dot_rev(A* da, A* db, const typename A::numeric_type* dout, void* tape) {
  namespace ad = TiledArray::ad;
  using S = typename A::numeric_type;
  auto* t = static_cast<BinaryResidual<A>*>(tape);
  const S sb = *dout;
  accumulate_into(da, ad::scale(ad::conj(t->b.get()), sb));
  accumulate_into(db, ad::scale(ad::conj(t->a.get()), sb));
  delete t;  // eager residual free as consumed (P2.1)
}
template <typename A>
void dot_fwd(const A* a, const A* da, const A* b, const A* db,
             typename A::numeric_type* out, typename A::numeric_type* dout) {
  namespace ad = TiledArray::ad;
  using S = typename A::numeric_type;
  *out = ad::dot(*a, *b);
  const S t1 = da->is_initialized() ? ad::dot(*da, *b) : S{0};
  const S t2 = db->is_initialized() ? ad::dot(*a, *db) : S{0};
  *dout = t1 + t2;
}

// inner_product: s = Σ conj(A(e))·B(e) (sesquilinear);
//   Ā += conj(s̄)·B;  B̄ += s̄·A.
template <typename A>
void inner_primal(const A* a, const A* b, typename A::numeric_type* out) {
  *out = TiledArray::ad::inner_product(*a, *b);
}
template <typename A>
void* inner_aug(const A* a, const A* b, typename A::numeric_type* out) {
  *out = TiledArray::ad::inner_product(*a, *b);
  return new BinaryResidual<A>{*a, *b};
}
template <typename A>
void inner_rev(A* da, A* db, const typename A::numeric_type* dout, void* tape) {
  namespace ad = TiledArray::ad;
  using S = typename A::numeric_type;
  auto* t = static_cast<BinaryResidual<A>*>(tape);
  const S sb = *dout;
  accumulate_into(da, ad::scale(t->b.get(), conj_scalar(sb)));
  accumulate_into(db, ad::scale(t->a.get(), sb));
  delete t;  // eager residual free as consumed (P2.1)
}
template <typename A>
void inner_fwd(const A* a, const A* da, const A* b, const A* db,
               typename A::numeric_type* out, typename A::numeric_type* dout) {
  namespace ad = TiledArray::ad;
  using S = typename A::numeric_type;
  *out = ad::inner_product(*a, *b);
  const S t1 = da->is_initialized() ? ad::inner_product(*da, *b) : S{0};
  const S t2 = db->is_initialized() ? ad::inner_product(*a, *db) : S{0};
  *dout = t1 + t2;
}

}  // namespace TiledArray::ad::edetail

// The shims and rules carry internal linkage but C-style flat names; the
// registration globals reference them by address. Matching is by pointer (the
// global's first element), so the global's name suffix is arbitrary, and C++
// linkage is fine (no `extern "C"`, which would force external linkage and
// collide across differentiated TUs).

/// Bind a reverse-mode rule triple {primal, augmented-forward, reverse}.
#define TA_AD_REGISTER_GRADIENT(name, primal, aug, rev)                      \
  static void* __enzyme_register_gradient_##name[] __attribute__((used)) = { \
      (void*)(primal), (void*)(aug), (void*)(rev)}

/// Bind a forward-mode rule pair {primal, derivative} (no tape; section 1).
#define TA_AD_REGISTER_DERIVATIVE(name, primal, fwd)                           \
  static void* __enzyme_register_derivative_##name[] __attribute__((used)) = { \
      (void*)(primal), (void*)(fwd)}

// ---------------------------------------------------------------------------
// Spec-bearing rule macros (P0.3): the annotation triple / permutation / scale
// factor / elementwise function is baked into the registered symbol, so the
// shims are fixed-arity functions of array (and scalar I/O) pointers only. A
// differentiated TU instantiates the patterns it uses, e.g.
//   TA_AD_CONTRACT_RULE(TArrayD, ik_kj_ij, "i,k", "k,j", "i,j");
// ---------------------------------------------------------------------------

#define TA_AD_CONTRACT_RULE(ARRAY, SUFFIX, AA, BB, CC)                         \
  static TA_AD_PRIMAL void ta_ad_contract_##SUFFIX(const ARRAY* a,             \
                                                   const ARRAY* b, ARRAY* c) { \
    TiledArray::ad::edetail::contract_primal(a, b, c, AA, BB, CC);              \
  }                                                                            \
  static TA_AD_RULE void* ta_ad_contract_##SUFFIX##_aug(                       \
      const ARRAY* a, const ARRAY*, const ARRAY* b, const ARRAY*, ARRAY* c,    \
      ARRAY*) {                                                                \
    return TiledArray::ad::edetail::contract_aug(a, b, c, AA, BB, CC);          \
  }                                                                            \
  static TA_AD_RULE void ta_ad_contract_##SUFFIX##_rev(                        \
      const ARRAY*, ARRAY* da, const ARRAY*, ARRAY* db, const ARRAY*,          \
      const ARRAY* dc, void* tape) {                                           \
    TiledArray::ad::edetail::contract_rev(da, db, dc, AA, BB, CC, tape);        \
  }                                                                            \
  static TA_AD_RULE void ta_ad_contract_##SUFFIX##_fwd(                        \
      const ARRAY* a, const ARRAY* da, const ARRAY* b, const ARRAY* db,        \
      ARRAY* c, ARRAY* dc) {                                                   \
    TiledArray::ad::edetail::contract_fwd(a, da, b, db, c, dc, AA, BB, CC);     \
  }                                                                            \
  TA_AD_REGISTER_GRADIENT(ta_ad_contract_##SUFFIX, ta_ad_contract_##SUFFIX,    \
                          ta_ad_contract_##SUFFIX##_aug,                       \
                          ta_ad_contract_##SUFFIX##_rev);                      \
  TA_AD_REGISTER_DERIVATIVE(ta_ad_contract_##SUFFIX, ta_ad_contract_##SUFFIX,  \
                            ta_ad_contract_##SUFFIX##_fwd)

#define TA_AD_PERMUTE_RULE(ARRAY, SUFFIX, IN, OUT)                            \
  static TA_AD_PRIMAL void ta_ad_permute_##SUFFIX(const ARRAY* a, ARRAY* c) { \
    TiledArray::ad::edetail::permute_primal(a, c, IN, OUT);                    \
  }                                                                          \
  static TA_AD_RULE void* ta_ad_permute_##SUFFIX##_aug(                       \
      const ARRAY* a, const ARRAY*, ARRAY* c, ARRAY*) {                       \
    return TiledArray::ad::edetail::permute_aug(a, c, IN, OUT);                \
  }                                                                          \
  static TA_AD_RULE void ta_ad_permute_##SUFFIX##_rev(                        \
      const ARRAY*, ARRAY* da, const ARRAY*, const ARRAY* dc, void*) {        \
    TiledArray::ad::edetail::permute_rev(da, dc, IN, OUT, nullptr);            \
  }                                                                          \
  static TA_AD_RULE void ta_ad_permute_##SUFFIX##_fwd(                        \
      const ARRAY* a, const ARRAY* da, ARRAY* c, ARRAY* dc) {                 \
    TiledArray::ad::edetail::permute_fwd(a, da, c, dc, IN, OUT);               \
  }                                                                          \
  TA_AD_REGISTER_GRADIENT(ta_ad_permute_##SUFFIX, ta_ad_permute_##SUFFIX,     \
                          ta_ad_permute_##SUFFIX##_aug,                       \
                          ta_ad_permute_##SUFFIX##_rev);                      \
  TA_AD_REGISTER_DERIVATIVE(ta_ad_permute_##SUFFIX, ta_ad_permute_##SUFFIX,   \
                            ta_ad_permute_##SUFFIX##_fwd)

#define TA_AD_SCALE_RULE(ARRAY, SUFFIX, ALPHA)                              \
  static TA_AD_PRIMAL void ta_ad_scale_##SUFFIX(const ARRAY* a, ARRAY* c) { \
    TiledArray::ad::edetail::scale_primal(                                   \
        a, c, static_cast<typename ARRAY::numeric_type>(ALPHA));            \
  }                                                                        \
  static TA_AD_RULE void* ta_ad_scale_##SUFFIX##_aug(                       \
      const ARRAY* a, const ARRAY*, ARRAY* c, ARRAY*) {                     \
    TiledArray::ad::edetail::scale_primal(                                   \
        a, c, static_cast<typename ARRAY::numeric_type>(ALPHA));            \
    return nullptr;                                                         \
  }                                                                        \
  static TA_AD_RULE void ta_ad_scale_##SUFFIX##_rev(                        \
      const ARRAY*, ARRAY* da, const ARRAY*, const ARRAY* dc, void*) {      \
    TiledArray::ad::edetail::scale_rev(                                      \
        da, dc, static_cast<typename ARRAY::numeric_type>(ALPHA), nullptr); \
  }                                                                        \
  static TA_AD_RULE void ta_ad_scale_##SUFFIX##_fwd(                        \
      const ARRAY* a, const ARRAY* da, ARRAY* c, ARRAY* dc) {               \
    (void)a;                                                                \
    *c = TiledArray::ad::scale(                                            \
        *a, static_cast<typename ARRAY::numeric_type>(ALPHA));             \
    TiledArray::ad::edetail::scale_fwd(                                      \
        da, dc, static_cast<typename ARRAY::numeric_type>(ALPHA));          \
  }                                                                        \
  TA_AD_REGISTER_GRADIENT(ta_ad_scale_##SUFFIX, ta_ad_scale_##SUFFIX,       \
                          ta_ad_scale_##SUFFIX##_aug,                       \
                          ta_ad_scale_##SUFFIX##_rev);                      \
  TA_AD_REGISTER_DERIVATIVE(ta_ad_scale_##SUFFIX, ta_ad_scale_##SUFFIX,     \
                            ta_ad_scale_##SUFFIX##_fwd)

// F and DF are callables baked into the symbol (e.g. lambdas). f must be
// holomorphic (or real) for the elementwise VJP to be valid.
#define TA_AD_ELEMENTWISE_RULE(ARRAY, SUFFIX, F, DF)                        \
  static TA_AD_PRIMAL void ta_ad_ew_##SUFFIX(const ARRAY* a, ARRAY* c) {    \
    TiledArray::ad::edetail::elementwise_primal(a, c, F, DF);               \
  }                                                                        \
  static TA_AD_RULE void* ta_ad_ew_##SUFFIX##_aug(const ARRAY* a,           \
                                                  const ARRAY*, ARRAY* c,   \
                                                  ARRAY*) {                 \
    return TiledArray::ad::edetail::elementwise_aug(a, c, F, DF);           \
  }                                                                        \
  static TA_AD_RULE void ta_ad_ew_##SUFFIX##_rev(                          \
      const ARRAY*, ARRAY* da, const ARRAY*, const ARRAY* dc, void* tape) { \
    TiledArray::ad::edetail::elementwise_rev(da, dc, F, DF, tape);          \
  }                                                                        \
  static TA_AD_RULE void ta_ad_ew_##SUFFIX##_fwd(                          \
      const ARRAY* a, const ARRAY* da, ARRAY* c, ARRAY* dc) {              \
    TiledArray::ad::edetail::elementwise_fwd(a, da, c, dc, F, DF);          \
  }                                                                        \
  TA_AD_REGISTER_GRADIENT(ta_ad_ew_##SUFFIX, ta_ad_ew_##SUFFIX,             \
                          ta_ad_ew_##SUFFIX##_aug, ta_ad_ew_##SUFFIX##_rev); \
  TA_AD_REGISTER_DERIVATIVE(ta_ad_ew_##SUFFIX, ta_ad_ew_##SUFFIX,           \
                            ta_ad_ew_##SUFFIX##_fwd)

// ---------------------------------------------------------------------------
// Spec-free rule macros: array->array and array->scalar primitives whose only
// arguments are arrays (and a scalar I/O pointer for reductions). One TAG per
// element/policy keeps the symbols unique. Stamped for every supported type by
// TA_AD_REGISTER_SPECFREE_OPS below.
// ---------------------------------------------------------------------------

// Binary array->array, no residual (add, subt).
#define TA_AD_BINARY_LINEAR_RULE(ARRAY, TAG, OP)                              \
  static TA_AD_PRIMAL void ta_ad_##OP##_##TAG(const ARRAY* a, const ARRAY* b, \
                                              ARRAY* c) {                     \
    TiledArray::ad::edetail::OP##_primal(a, b, c);                            \
  }                                                                          \
  static TA_AD_RULE void* ta_ad_##OP##_##TAG##_aug(                          \
      const ARRAY* a, const ARRAY*, const ARRAY* b, const ARRAY*, ARRAY* c,  \
      ARRAY*) {                                                              \
    return TiledArray::ad::edetail::OP##_aug(a, b, c);                        \
  }                                                                          \
  static TA_AD_RULE void ta_ad_##OP##_##TAG##_rev(                          \
      const ARRAY*, ARRAY* da, const ARRAY*, ARRAY* db, const ARRAY*,        \
      const ARRAY* dc, void* tape) {                                         \
    TiledArray::ad::edetail::OP##_rev(da, db, dc, tape);                      \
  }                                                                          \
  static TA_AD_RULE void ta_ad_##OP##_##TAG##_fwd(                          \
      const ARRAY* a, const ARRAY* da, const ARRAY* b, const ARRAY* db,      \
      ARRAY* c, ARRAY* dc) {                                                 \
    TiledArray::ad::edetail::OP##_fwd(a, da, b, db, c, dc);                   \
  }                                                                          \
  TA_AD_REGISTER_GRADIENT(ta_ad_##OP##_##TAG, ta_ad_##OP##_##TAG,            \
                          ta_ad_##OP##_##TAG##_aug,                          \
                          ta_ad_##OP##_##TAG##_rev);                         \
  TA_AD_REGISTER_DERIVATIVE(ta_ad_##OP##_##TAG, ta_ad_##OP##_##TAG,          \
                            ta_ad_##OP##_##TAG##_fwd)

// Binary array->array with a saved residual (mult).
#define TA_AD_BINARY_RESIDUAL_RULE(ARRAY, TAG, OP)                           \
  static TA_AD_PRIMAL void ta_ad_##OP##_##TAG(const ARRAY* a, const ARRAY* b, \
                                              ARRAY* c) {                     \
    TiledArray::ad::edetail::OP##_primal(a, b, c);                            \
  }                                                                          \
  static TA_AD_RULE void* ta_ad_##OP##_##TAG##_aug(                          \
      const ARRAY* a, const ARRAY*, const ARRAY* b, const ARRAY*, ARRAY* c,  \
      ARRAY*) {                                                              \
    return TiledArray::ad::edetail::OP##_aug(a, b, c);                        \
  }                                                                          \
  static TA_AD_RULE void ta_ad_##OP##_##TAG##_rev(                          \
      const ARRAY*, ARRAY* da, const ARRAY*, ARRAY* db, const ARRAY*,        \
      const ARRAY* dc, void* tape) {                                         \
    TiledArray::ad::edetail::OP##_rev(da, db, dc, tape);                      \
  }                                                                          \
  static TA_AD_RULE void ta_ad_##OP##_##TAG##_fwd(                          \
      const ARRAY* a, const ARRAY* da, const ARRAY* b, const ARRAY* db,      \
      ARRAY* c, ARRAY* dc) {                                                 \
    TiledArray::ad::edetail::OP##_fwd(a, da, b, db, c, dc);                   \
  }                                                                          \
  TA_AD_REGISTER_GRADIENT(ta_ad_##OP##_##TAG, ta_ad_##OP##_##TAG,            \
                          ta_ad_##OP##_##TAG##_aug,                          \
                          ta_ad_##OP##_##TAG##_rev);                         \
  TA_AD_REGISTER_DERIVATIVE(ta_ad_##OP##_##TAG, ta_ad_##OP##_##TAG,          \
                            ta_ad_##OP##_##TAG##_fwd)

// Unary array->array (conj).
#define TA_AD_UNARY_RULE(ARRAY, TAG, OP)                                     \
  static TA_AD_PRIMAL void ta_ad_##OP##_##TAG(const ARRAY* a, ARRAY* c) {    \
    TiledArray::ad::edetail::OP##_primal(a, c);                              \
  }                                                                          \
  static TA_AD_RULE void* ta_ad_##OP##_##TAG##_aug(const ARRAY* a,           \
                                                   const ARRAY*, ARRAY* c,   \
                                                   ARRAY*) {                 \
    return TiledArray::ad::edetail::OP##_aug(a, c);                          \
  }                                                                          \
  static TA_AD_RULE void ta_ad_##OP##_##TAG##_rev(                          \
      const ARRAY*, ARRAY* da, const ARRAY*, const ARRAY* dc, void*) {       \
    TiledArray::ad::edetail::OP##_rev(da, dc, nullptr);                      \
  }                                                                          \
  static TA_AD_RULE void ta_ad_##OP##_##TAG##_fwd(                          \
      const ARRAY* a, const ARRAY* da, ARRAY* c, ARRAY* dc) {               \
    TiledArray::ad::edetail::OP##_fwd(a, da, c, dc);                         \
  }                                                                          \
  TA_AD_REGISTER_GRADIENT(ta_ad_##OP##_##TAG, ta_ad_##OP##_##TAG,            \
                          ta_ad_##OP##_##TAG##_aug,                          \
                          ta_ad_##OP##_##TAG##_rev);                         \
  TA_AD_REGISTER_DERIVATIVE(ta_ad_##OP##_##TAG, ta_ad_##OP##_##TAG,          \
                            ta_ad_##OP##_##TAG##_fwd)

// Unary reduction A->scalar with a saved residual. OUTT is the scalar pointer
// element type (numeric_type for sum/trace, scalar_type for sqnorm/norm2).
#define TA_AD_REDUCTION_RULE(ARRAY, TAG, OP, OUTT)                            \
  static TA_AD_PRIMAL void ta_ad_##OP##_##TAG(const ARRAY* a, OUTT* out) {    \
    TiledArray::ad::edetail::OP##_primal(a, out);                            \
  }                                                                          \
  static TA_AD_RULE void* ta_ad_##OP##_##TAG##_aug(const ARRAY* a,           \
                                                   const ARRAY*, OUTT* out,   \
                                                   OUTT*) {                   \
    return TiledArray::ad::edetail::OP##_aug(a, out);                        \
  }                                                                          \
  static TA_AD_RULE void ta_ad_##OP##_##TAG##_rev(                          \
      const ARRAY*, ARRAY* da, const OUTT*, const OUTT* dout, void* tape) {  \
    TiledArray::ad::edetail::OP##_rev(da, dout, tape);                       \
  }                                                                          \
  static TA_AD_RULE void ta_ad_##OP##_##TAG##_fwd(                          \
      const ARRAY* a, const ARRAY* da, OUTT* out, OUTT* dout) {             \
    TiledArray::ad::edetail::OP##_fwd(a, da, out, dout);                     \
  }                                                                          \
  TA_AD_REGISTER_GRADIENT(ta_ad_##OP##_##TAG, ta_ad_##OP##_##TAG,            \
                          ta_ad_##OP##_##TAG##_aug,                          \
                          ta_ad_##OP##_##TAG##_rev);                         \
  TA_AD_REGISTER_DERIVATIVE(ta_ad_##OP##_##TAG, ta_ad_##OP##_##TAG,          \
                            ta_ad_##OP##_##TAG##_fwd)

// Binary reduction (A,B)->scalar with a saved residual (dot, inner).
#define TA_AD_BINARY_REDUCTION_RULE(ARRAY, TAG, OP)                          \
  static TA_AD_PRIMAL void ta_ad_##OP##_##TAG(                               \
      const ARRAY* a, const ARRAY* b, typename ARRAY::numeric_type* out) {   \
    TiledArray::ad::edetail::OP##_primal(a, b, out);                         \
  }                                                                          \
  static TA_AD_RULE void* ta_ad_##OP##_##TAG##_aug(                          \
      const ARRAY* a, const ARRAY*, const ARRAY* b, const ARRAY*,            \
      typename ARRAY::numeric_type* out, typename ARRAY::numeric_type*) {    \
    return TiledArray::ad::edetail::OP##_aug(a, b, out);                     \
  }                                                                          \
  static TA_AD_RULE void ta_ad_##OP##_##TAG##_rev(                          \
      const ARRAY*, ARRAY* da, const ARRAY*, ARRAY* db,                      \
      const typename ARRAY::numeric_type*,                                   \
      const typename ARRAY::numeric_type* dout, void* tape) {                \
    TiledArray::ad::edetail::OP##_rev(da, db, dout, tape);                   \
  }                                                                          \
  static TA_AD_RULE void ta_ad_##OP##_##TAG##_fwd(                          \
      const ARRAY* a, const ARRAY* da, const ARRAY* b, const ARRAY* db,      \
      typename ARRAY::numeric_type* out, typename ARRAY::numeric_type* dout) { \
    TiledArray::ad::edetail::OP##_fwd(a, da, b, db, out, dout);              \
  }                                                                          \
  TA_AD_REGISTER_GRADIENT(ta_ad_##OP##_##TAG, ta_ad_##OP##_##TAG,            \
                          ta_ad_##OP##_##TAG##_aug,                          \
                          ta_ad_##OP##_##TAG##_rev);                         \
  TA_AD_REGISTER_DERIVATIVE(ta_ad_##OP##_##TAG, ta_ad_##OP##_##TAG,          \
                            ta_ad_##OP##_##TAG##_fwd)

/// Stamp the full spec-free primitive set for an array type with a unique TAG.
#define TA_AD_REGISTER_SPECFREE_OPS(ARRAY, TAG)                              \
  TA_AD_BINARY_LINEAR_RULE(ARRAY, TAG, add);                                 \
  TA_AD_BINARY_LINEAR_RULE(ARRAY, TAG, subt);                                \
  TA_AD_BINARY_RESIDUAL_RULE(ARRAY, TAG, mult);                              \
  TA_AD_UNARY_RULE(ARRAY, TAG, conj);                                        \
  TA_AD_REDUCTION_RULE(ARRAY, TAG, sum, typename ARRAY::numeric_type);       \
  TA_AD_REDUCTION_RULE(ARRAY, TAG, trace, typename ARRAY::numeric_type);     \
  TA_AD_REDUCTION_RULE(ARRAY, TAG, sqnorm, typename ARRAY::scalar_type);     \
  TA_AD_REDUCTION_RULE(ARRAY, TAG, norm2, typename ARRAY::scalar_type);      \
  TA_AD_BINARY_REDUCTION_RULE(ARRAY, TAG, dot);                              \
  TA_AD_BINARY_REDUCTION_RULE(ARRAY, TAG, inner)

// Pre-register the spec-free ops for the supported element/policy types
// (P1.1 reverse + P1.2 forward; P1.3 complex + sparse). `d` = real dense,
// `z` = complex dense, `sd` = real sparse.
TA_AD_REGISTER_SPECFREE_OPS(TiledArray::TArrayD, d);
TA_AD_REGISTER_SPECFREE_OPS(TiledArray::TArrayZ, z);
TA_AD_REGISTER_SPECFREE_OPS(TiledArray::TSpArrayD, sd);

#endif  // TILEDARRAY_HAS_ENZYME

#endif  // TILEDARRAY_AD_ENZYME_RULES_H__INCLUDED
