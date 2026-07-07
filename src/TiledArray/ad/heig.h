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
 *  heig.h
 *  Differentiable symmetric/Hermitian eigendecomposition for TiledArray's
 *  AD layer.
 */

#ifndef TILEDARRAY_AD_HEIG_H__INCLUDED
#define TILEDARRAY_AD_HEIG_H__INCLUDED

#include <algorithm>
#include <cmath>
#include <limits>
#include <optional>
#include <utility>
#include <vector>

#include <TiledArray/ad/dual.h>
#include <TiledArray/ad/ops.h>
#include <TiledArray/ad/shadow.h>
#include <TiledArray/ad/tape.h>
#include <TiledArray/error.h>
#include <TiledArray/math/linalg/heig.h>

/// \file heig.h
///
/// `ad::heig` — the symmetric/Hermitian standard eigenproblem `A = U Λ Uᴴ`
/// as an *atomic* AD primitive (autodiff plan Phase 4). The LAPACK solve
/// underneath is opaque to AD (same rationale as `contract`), but the JVP
/// and VJP bodies are written entirely in the B1 primitive set, so
/// forward-over-reverse (HVP through `heig`) composes with no extra rules.
///
/// Derivative rules (AUTODIFF_BACKGROUND.md §7; Giles §3.1 / Liao eq. 3),
/// with ascending eigenvalues and eigenvector columns `U`:
///
///   JVP:  M = Uᴴ·dA·U;  dλ = diag(M);  dU = U·(F ∘ M),
///         F_ij = 1/(λ_j − λ_i) (i≠j), F_ii = 0.
///   VJP:  Ā = U [ diag(λ̄) + F ∘ (UᴴŪ − ŪᴴU)/2 ] Uᴴ, Hermitian-projected.
///
/// Semantics designed in (each a documented eig-AD failure mode):
///
///  * **Degeneracy policy.** `F_ij` diverges as eigenvalues coalesce and the
///    eigenvector derivative is *genuinely undefined* within a degenerate
///    subspace. `HeigDiffPolicy` picks between `error` (throw
///    `TiledArray::Exception` — a production check, so `TA_EXCEPTION`, never
///    `TA_ASSERT`) and `broaden` (Lorentzian `1/x → x/(x²+ε)`, Liao
///    §III.A.1). QC-side escapes (Lagrangians, gauge-invariant
///    occupied/virtual pairings) are the caller's knowledge — only the
///    general policy lives here.
///  * **Lazy F in reverse mode.** When only the eigenvalue cotangent `λ̄` is
///    nonzero the VJP is `U·diag(λ̄)·Uᴴ` (Hellmann–Feynman) — well-defined
///    even at exact degeneracy. `heig_vjp` builds `F` only if an eigenvector
///    cotangent actually flows, so that path never trips the policy. Forward
///    mode is *eager* (`dU` is always computed alongside `dλ`), so the
///    policy check fires at the call site instead.
///  * **Gauge.** Eigenvector signs (real) / phases (complex) are arbitrary;
///    the rules adopt the standard gauge (`F_ii = 0` drops the diagonal of
///    `UᴴŪ`). Derivatives are meaningful only when downstream is
///    gauge-invariant; tests use gauge-invariant functionals.
///
/// Preconditions/scope (mirroring `math::linalg::heig`): `a` is rank-2 and
/// Hermitian (not checked — a non-Hermitian input yields garbage the same
/// way `linalg::heig` does); dense-policy arrays (the linalg solve
/// replicates through Eigen anyway) — sparse callers convert first. The
/// generalized problem `heig(A, B)` deliberately has **no** AD rule yet:
/// differentiating it adds `B`-cotangent terms and a `B`-metric gauge; the
/// seam is this same header once needed. SVD/Cholesky/QR are future work.

namespace TiledArray::ad {

/// Policy knobs for differentiating through `heig`.
///
/// Applies to any path that must build the gap matrix
/// `F_ij = 1/(λ_j − λ_i)`: forward mode always, reverse mode only when an
/// eigenvector cotangent flows (see file header).
struct HeigDiffPolicy {
  enum class Degeneracy {
    error,   ///< throw TiledArray::Exception on a (near-)degenerate spectrum
    broaden  ///< Lorentzian broadening φ_ε(x) = x/(x²+ε)  (Liao §III.A.1)
  };
  Degeneracy on_degeneracy = Degeneracy::error;
  double degeneracy_tol = 1e-10;  ///< relative min-gap threshold for `error`
  double broadening = 1e-12;      ///< ε in φ_ε(x) = x/(x²+ε)
};

/// Result of `ad::heig`: eigenvalues as a rank-1 array (so the single-typed
/// `Var`/`Dual` machinery applies), eigenvector columns, and the spectrum's
/// minimum relative gap (host-side, replicated — the degeneracy diagnostic).
template <typename Array>
struct HeigResult {
  Array evals;     ///< rank-1, ascending, real values in Array::numeric_type
  Array evecs;     ///< rank-2, column j = eigenvector j (A X = X E)
  double min_gap;  ///< min consecutive |Δλ| / spectral scale
};

namespace detail {

/// Minimum consecutive eigenvalue gap relative to the spectral scale.
/// \param evals eigenvalues in ascending order
template <typename S>
double min_relative_gap(const std::vector<S>& evals) {
  if (evals.size() < 2) return std::numeric_limits<double>::infinity();
  const double scale = std::max<double>(
      {1.0, std::abs(double(evals.front())), std::abs(double(evals.back()))});
  double gap = std::numeric_limits<double>::infinity();
  for (std::size_t i = 1; i < evals.size(); ++i)
    gap = std::min(gap, double(evals[i] - evals[i - 1]) / scale);
  return gap;
}

/// Enforce the degeneracy policy for a path that needs the F matrix.
/// Throws regardless of build type (production check → `TA_EXCEPTION`).
inline void enforce_degeneracy_policy(const HeigDiffPolicy& p,
                                      double min_gap) {
  if (p.on_degeneracy == HeigDiffPolicy::Degeneracy::error &&
      min_gap < p.degeneracy_tol)
    TA_EXCEPTION(
        "TiledArray::ad::heig: (near-)degenerate eigenvalues; the "
        "eigenvector derivative is ill-defined (F_ij = 1/(lambda_j - "
        "lambda_i)). Options: HeigDiffPolicy::Degeneracy::broaden, or "
        "differentiate eigenvalues only (the Hellmann-Feynman path never "
        "builds F).");
}

/// Constant rank-1 array of ones with `v`'s structure (inactive constant;
/// lift per AD level with `lifter`).
template <typename Array>
Array ones_like(const Array& v) {
  Array c(v.world(), v.trange());
  c.init_elements([](const auto&) {
    return typename Array::numeric_type(1);
  });
  return c;
}

/// Constant identity matrix with `m`'s (square, rank-2) structure.
template <typename Array>
Array identity_like(const Array& m) {
  Array c(m.world(), m.trange());
  c.init_elements([](const auto& idx) {
    return typename Array::numeric_type(idx[0] == idx[1] ? 1 : 0);
  });
  return c;
}

/// Peel `Dual<>` nesting: the plain `DistArray` type underneath.
template <typename T>
struct plain_array {
  using type = T;
};
template <typename T>
struct plain_array<Dual<T>> {
  using type = typename plain_array<T>::type;
};
template <typename T>
using plain_array_t = typename plain_array<T>::type;

/// The primal-most (plain-array) value of a possibly-nested `Dual`.
template <typename T>
const T& primal_of(const T& v) {
  return v;
}
template <typename T>
const plain_array_t<Dual<T>>& primal_of(const Dual<T>& v) {
  return primal_of(v.primal);
}

/// Lift a plain constant array to the AD level of `T` (constant duals at
/// every `Dual<>` layer; identity for a plain array).
template <typename T>
struct lifter {
  static T lift(T c) { return c; }
};
template <typename T>
struct lifter<Dual<T>> {
  static Dual<T> lift(plain_array_t<Dual<T>> c) {
    return make_constant(lifter<T>::lift(std::move(c)));
  }
};

/// Build the gap matrix `F = φ(Δ)`, `Δ_ij = λ_j − λ_i`, in the B1 primitive
/// set (so `T` may be `Array` or any `Dual<>` nesting — the route to
/// forward-over-reverse).
///
/// `error` policy: `φ(x) = x==0 ? 0 : 1/x` — once `enforce_degeneracy_policy`
/// has passed, the diagonal is the only exact zero. `broaden`: Lorentzian
/// `φ_ε(x) = x/(x²+ε)` — smooth, and `φ_ε(0) = 0` handles the diagonal for
/// free.
template <typename T>
T make_f_matrix(const T& evals, const HeigDiffPolicy& policy) {
  using Plain = plain_array_t<T>;
  using S = typename Plain::numeric_type;
  T ones = lifter<T>::lift(ones_like(primal_of(evals)));
  T lam_j = ad::contract(ones, evals, "i", "j", "i,j");  // Δ_ij = λ_j − λ_i
  T lam_i = ad::contract(evals, ones, "i", "j", "i,j");
  T delta = ad::subt(lam_j, lam_i);
  if (policy.on_degeneracy == HeigDiffPolicy::Degeneracy::broaden) {
    const auto eps = static_cast<decltype(std::abs(S{}))>(policy.broadening);
    return ad::elementwise(
        delta, [eps](S x) { return x / (x * x + eps); },
        [eps](S x) {
          auto d = x * x + eps;
          return (eps - x * x) / (d * d);
        });
  }
  return ad::elementwise(
      delta, [](S x) { return x == S{0} ? S{0} : S{1} / x; },
      [](S x) { return x == S{0} ? S{0} : S{-1} / (x * x); });
}

/// JVP kernel: `(dλ, dU)` from `(λ, U, dA)` —
/// `M = Uᴴ dA U`, `dλ = diag(M)`, `dU = U (F ∘ M)`.
///
/// Forward mode is eager (`dU` always computed), so the degeneracy policy is
/// enforced unconditionally here.
template <typename T>
std::pair<T, T> heig_jvp(const T& evals, const T& evecs, const T& da,
                         const HeigDiffPolicy& policy, double min_gap) {
  enforce_degeneracy_policy(policy, min_gap);  // dU always needs F
  T Uc = ad::conj(evecs);
  T M = ad::contract(ad::contract(Uc, da, "k,i", "k,l", "i,l"), evecs, "i,l",
                     "l,j", "i,j");  // Uᴴ dA U
  T I = lifter<T>::lift(identity_like(primal_of(evecs)));
  T ones = lifter<T>::lift(ones_like(primal_of(evals)));
  T dlam = ad::contract(ad::mult(M, I), ones, "i,j", "j", "i");  // diag(M)
  T F = make_f_matrix(evals, policy);
  T dU = ad::contract(evecs, ad::mult(F, M), "i,k", "k,j", "i,j");
  return {std::move(dlam), std::move(dU)};
}

/// VJP kernel: Hermitian-projected `Ā` from `(λ, U, λ̄?, Ū?)`; a null
/// cotangent pointer is the symbolic zero.
///
/// `Ā = U [ diag(λ̄) + F ∘ (UᴴŪ − ŪᴴU)/2 ] Uᴴ`, then `(Ā + Āᴴ)/2`. `F` is
/// built **only** when `ubar` is nonnull, so the eigenvalue-only
/// (Hellmann–Feynman) path never trips the degeneracy policy.
template <typename T>
T heig_vjp(const T& evals, const T& evecs, const T* lbar, const T* ubar,
           const HeigDiffPolicy& policy, double min_gap) {
  TA_ASSERT(lbar || ubar);
  using S = typename plain_array_t<T>::numeric_type;
  T Uc = ad::conj(evecs);
  std::optional<T> G;
  if (lbar) {  // diag(λ̄) = (λ̄ ⊗ 1) ∘ I
    T ones = lifter<T>::lift(ones_like(primal_of(evals)));
    T I = lifter<T>::lift(identity_like(primal_of(evecs)));
    G = ad::mult(ad::contract(*lbar, ones, "i", "j", "i,j"), I);
  }
  if (ubar) {  // F needed → the policy fires lazily, only on this branch
    enforce_degeneracy_policy(policy, min_gap);
    T F = make_f_matrix(evals, policy);
    T P = ad::contract(Uc, *ubar, "k,i", "k,j", "i,j");  // UᴴŪ
    T skew = ad::scale(ad::subt(P, ad::permute(ad::conj(P), "i,j", "j,i")),
                       S{0.5});  // (P − Pᴴ)/2
    T FS = ad::mult(F, skew);
    G = G ? std::optional<T>(ad::add(*G, FS)) : std::optional<T>(std::move(FS));
  }
  T A1 = ad::contract(ad::contract(evecs, *G, "i,k", "k,j", "i,j"), Uc, "i,k",
                      "j,k", "i,j");  // U G Uᴴ
  // project onto the Hermitian tangent space (the input is constrained
  // Hermitian; this also folds Im(λ̄) away — λ is real, so only Re(λ̄) can
  // matter under the Re⟨·,·⟩ pairing): Ā = (A1 + A1ᴴ)/2
  return ad::scale(ad::add(A1, ad::permute(ad::conj(A1), "i,j", "j,i")),
                   S{0.5});
}

}  // namespace detail

/// Functional (primal-only) differentiable-eigendecomposition entry point.
///
/// Wraps `math::linalg::heig` and additionally (a) returns the eigenvalues
/// as a rank-1 array whose tiling matches the eigenvector *column* tiling —
/// so every F/outer contraction in the derivative kernels is conformant —
/// and (b) reports the spectrum's minimum relative gap for the degeneracy
/// policy.
///
/// \param a rank-2 Hermitian array (precondition, not checked)
/// \param policy unused on the plain-array overload; kept so the uniform
///        `heig(x, policy, evec_trange)` signature compiles for generic code
///        across the `Array`/`Dual`/`Var` overloads
/// \param evec_trange forwarded to `math::linalg::heig`
template <typename Array>
HeigResult<Array> heig(const Array& a, const HeigDiffPolicy& /*policy*/ = {},
                       TiledRange evec_trange = TiledRange()) {
  auto [evals_vec, evecs] = TiledArray::math::linalg::heig(a, evec_trange);
  const double min_gap = detail::min_relative_gap(evals_vec);
  TiledRange evals_trange({evecs.trange().dim(1)});
  Array evals(a.world(), evals_trange);
  evals.init_elements([v = evals_vec](const auto& idx) {  // capture BY VALUE:
    return typename Array::numeric_type(v[idx[0]]);       // tasks may outlive
  });                                                     // this frame
  return {std::move(evals), std::move(evecs), min_gap};
}

/// Forward-mode (JVP) rule for `heig`.
///
/// Forward mode is *eager*: both `dλ` and `dU` are computed whenever the
/// input carries a tangent, so the degeneracy policy fires at this call
/// (unlike reverse mode's lazy F — see the file header). A constant input
/// yields constant outputs (symbolic-zero tangents) and never builds F.
///
/// The JVP kernel is invoked with `T = Array`, which may itself be a
/// `Dual<...>` — higher-order forward nesting needs nothing extra.
template <typename Array>
HeigResult<Dual<Array>> heig(const Dual<Array>& a,
                             const HeigDiffPolicy& policy = {},
                             TiledRange evec_trange = TiledRange()) {
  auto p = ad::heig(a.primal, policy, std::move(evec_trange));
  if (!a.has_tangent())
    return {make_constant(std::move(p.evals)),
            make_constant(std::move(p.evecs)), p.min_gap};
  auto [dlam, dU] =
      detail::heig_jvp(p.evals, p.evecs, *a.tangent, policy, p.min_gap);
  return {make_dual(std::move(p.evals), std::move(dlam)),
          make_dual(std::move(p.evecs), std::move(dU)), p.min_gap};
}

/// Result of reverse-mode `heig`: both outputs as tape variables.
template <typename T>
struct HeigVars {
  Var<T> evals;    ///< rank-1 eigenvalues variable
  Var<T> evecs;    ///< rank-2 eigenvector-columns variable
  double min_gap;  ///< spectrum's minimum relative gap (diagnostic)
};

/// Reverse-mode (VJP) rule for `heig`.
///
/// Records one node adjoining *both* outputs into `Ā` via `heig_vjp`. The
/// cotangents are read lazily at `backward()`: if only the eigenvalue
/// cotangent is nonzero the Hellmann–Feynman path runs and F is never built,
/// so the degeneracy policy cannot fire (see the file header). The saved
/// residuals `λ`, `U` alias the returned outputs and are freeze-guarded like
/// `contract`'s operands.
///
/// \tparam T `Array`, or `Dual<Array>` for forward-over-reverse (HVP)
template <typename T>
HeigVars<T> heig(const Var<T>& a, const HeigDiffPolicy& policy = {},
                 TiledRange evec_trange = TiledRange()) {
  auto p = ad::heig(a.value, policy, std::move(evec_trange));
  Tape<T>* tape = a.tape;
  if (!a.active)
    return {Var<T>{tape, 0, std::move(p.evals), false},
            Var<T>{tape, 0, std::move(p.evecs), false}, p.min_gap};
  const std::size_t l_id = tape->make_variable();
  const std::size_t u_id = tape->make_variable();
  const std::size_t a_id = a.id;
  tape->register_guard(p.evals);  // saved residuals alias the returned outputs
  tape->register_guard(p.evecs);
  T lam = p.evals, U = p.evecs;
  const double min_gap = p.min_gap;
  tape->push_node([=]() {
    auto& lbar = tape->adjoint(l_id);
    auto& ubar = tape->adjoint(u_id);
    if (lbar.is_zero() && ubar.is_zero()) return;
    tape->accumulate_array(
        a_id,
        detail::heig_vjp(lam, U, lbar.is_zero() ? nullptr : &lbar.value(),
                         ubar.is_zero() ? nullptr : &ubar.value(), policy,
                         min_gap));
  });
  return {Var<T>{tape, l_id, std::move(p.evals), true},
          Var<T>{tape, u_id, std::move(p.evecs), true}, p.min_gap};
}

}  // namespace TiledArray::ad

#endif  // TILEDARRAY_AD_HEIG_H__INCLUDED
