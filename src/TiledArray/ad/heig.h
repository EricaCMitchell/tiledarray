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

}  // namespace TiledArray::ad

#endif  // TILEDARRAY_AD_HEIG_H__INCLUDED
