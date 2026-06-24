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
 *  enzyme.h
 *  Public surface for differentiating host C++ with Enzyme over TiledArray.
 */

#ifndef TILEDARRAY_AD_ENZYME_H__INCLUDED
#define TILEDARRAY_AD_ENZYME_H__INCLUDED

#include <TiledArray/config.h>

#ifdef TILEDARRAY_HAS_ENZYME

/// \file enzyme.h
///
/// Driver surface for the Enzyme path of TiledArray's AD layer (autodiff plan
/// B5). It declares the Enzyme intrinsics a host program calls to differentiate
/// a routine; the differentiable *primitive* shims and their custom rules are in
/// `enzyme_rules.h`, which a differentiated TU must also include (Enzyme needs
/// the rule bodies in the module it transforms — see that header).
///
/// This is the **narrow waist** (autodiff plan, "why custom rules"): the host
/// routine is a chain of registered shims, so the Enzyme plugin treats each
/// TiledArray primitive as atomic and never descends into MADNESS tasks /
/// `Future<Tile>`, BLAS, or MPI collectives, which it cannot trace.
///
/// Usage (full example in tests/ad_enzyme/energy.cpp):
///
/// ```cpp
/// #include <TiledArray/ad/enzyme_rules.h>   // the shims + custom rules
/// #include <TiledArray/ad/enzyme.h>         // the driver intrinsics below
/// using TiledArray::TArrayD;
///
/// // All intermediates and shadows are CALLER-allocated: Enzyme cannot reverse
/// // a DistArray constructor/destructor, so the differentiated body holds no
/// // DistArray lifetime — only calls to the atomic custom rules.
/// extern "C" void ta_energy(const TArrayD* A, const TArrayD* B,
///                           const TArrayD* H, TArrayD* c1, TArrayD* c2,
///                           double* e) {
///   ta_ad_contract_d(A, B, c1, "i,k", "k,j", "i,j");
///   ta_ad_add_d(c1, H, c2);
///   ta_ad_sqnorm_d(c2, e);   // *e = || A·B + H ||²
/// }
///
/// TArrayD c1, c2, dA, dB, dH, dc1, dc2;   // shadows default-constructed = 0
/// double e = 0, de = 1;                   // de seeds the output cotangent
/// __enzyme_autodiff((void*)ta_energy, enzyme_dup, &A, &dA, enzyme_dup, &B, &dB,
///                   enzyme_dup, &H, &dH, enzyme_dup, &c1, &dc1,
///                   enzyme_dup, &c2, &dc2, enzyme_dup, &e, &de);
/// // dA now holds dE/dA.
/// ```
///
/// Mark each argument's activity with a sentinel before it: `enzyme_dup` for an
/// active value (followed by its shadow), `enzyme_const` for one to hold fixed.
/// All operands of a custom-rule call must be active so the call site matches
/// the all-shadow signature its registered rule expects (`enzyme_rules.h`).
///
/// Forward mode (JVP) uses the same activity markers but a separate driver,
/// `__enzyme_fwddiff`, and a separate registration global
/// (`__enzyme_register_derivative_*`, a 2-element {primal, derivative} pair with
/// no tape — *not* a variant of the reverse gradient triple). A forward-over-
/// reverse nesting (`__enzyme_fwddiff` of `__enzyme_autodiff`) gives a
/// Hessian-vector product.
///
/// \warning **Opt-level stance (P0.2).** Differentiated TUs must be built at the
/// optimization level the rules were validated at. The primal shims are
/// `noinline` (see `enzyme_rules.h`, `TA_AD_PRIMAL`) so that at -O2+ the inliner
/// cannot fold a shim into the host routine before the Enzyme pass runs — which
/// would erase the call site Enzyme matches against a registered rule and send
/// it descending into the MADWorld runtime it cannot trace. If you hit an
/// Enzyme type-analysis or pre-optimization crash, the FAQ's first workaround is
/// `-mllvm -enzyme-preopt=0`. `ta_enable_enzyme_on` attaches `-fpass-plugin`
/// per-target and inherits the build type's `-O`; keep all differentiated TUs on
/// one level.

extern "C" double __enzyme_autodiff(void*, ...);
extern "C" void __enzyme_fwddiff(void*, ...);
extern int enzyme_dup;
extern int enzyme_const;
extern int enzyme_out;
/// `enzyme_dupnoneed` (P2.2): like `enzyme_dup` (active value + shadow) but tells
/// Enzyme the *primal* output is dead — not read after the differentiated call —
/// so it need not keep that primal live. Use it for caller-allocated
/// intermediates whose value is only an input to the next rule and is never read
/// back once the forward pass is captured in the tapes (e.g. a contraction's
/// output that feeds a *linear* op saving no residual). It is an optimization,
/// not a correctness knob: the gradient is identical to `enzyme_dup`.
extern int enzyme_dupnoneed;

// ---------------------------------------------------------------------------
// Differentiable-surface helpers (P2.4).
//
// `__enzyme_autodiff` / `__enzyme_fwddiff` take *positional* activity-marker /
// pointer triples, hand-written and easy to desync as a routine grows (the
// energy demo is already six paired arguments). These macros keep each
// {marker, primal, shadow} group atomic so a primal can never drift from its
// shadow. Spell an autodiff call as:
//
//   __enzyme_autodiff((void*)f, TA_AD_DUP(A, dA), TA_AD_DUP(B, dB),
//                     TA_AD_DUP(e, de));
//
// and a forward call the same way (the markers mean the same thing in both).
// ---------------------------------------------------------------------------

/// An active {primal, shadow} pair: `enzyme_dup, &primal, &shadow`.
#define TA_AD_DUP(primal, shadow) enzyme_dup, &(primal), &(shadow)
/// An active pair whose primal output is dead (P2.2): `enzyme_dupnoneed, …`.
#define TA_AD_DUPNONEED(primal, shadow) enzyme_dupnoneed, &(primal), &(shadow)
/// An inactive (held-constant) argument: `enzyme_const, &value`.
#define TA_AD_CONST(value) enzyme_const, &(value)

/// \name Differentiable-surface checklist (P2.4)
/// A routine handed to `__enzyme_autodiff`/`__enzyme_fwddiff` must satisfy all of
/// the following, none of which the compiler enforces — a violation surfaces as a
/// deep plugin abort, not a normal C++ diagnostic:
///   1. **Every primal and every shadow is caller-allocated** and passed in;
///      the body owns no `DistArray` lifetime. Enzyme cannot reverse a
///      `DistArray` constructor/destructor (`reset_pimpl`) — one stray local
///      `RArray tmp = …` makes it try to build a shadow it cannot and aborts.
///      Scalar locals (the `double e, de` seeds) are fine.
///   2. **The body is only calls to the registered rule shims** (`enzyme_rules.h`)
///      — no direct `ad::*` op or expression-DSL call, which would descend into
///      the MADWorld runtime / `world.gop` Enzyme cannot trace.
///   3. **No exceptions cross the differentiated boundary** and **no virtual
///      dispatch** sits in the differentiated call chain.
///   4. **All operands of a rule call are active** (`enzyme_dup`/`dupnoneed`) so
///      the call site matches the all-shadow signature its registered rule
///      expects; hold a true constant with `enzyme_const` instead.
/// Incidental non-differentiable runtime calls that slip in can be softened with
/// `enzyme_inactive` (P2.6, below) — but that does **not** rescue a leaked
/// `DistArray` constructor, which still aborts.

// ---------------------------------------------------------------------------
// Defense-in-depth: enzyme_inactive (P2.6).
//
// Constraint #1 above fails *loudly* (plugin abort) the moment Enzyme reaches IR
// it cannot reverse. Marking a function `enzyme_inactive` makes Enzyme treat it
// as constant and not trace into it, turning a hard failure into a benign skip —
// a cheap guard for the handful of MADWorld / world.gop / fence-like symbols that
// might be inlined into a differentiated TU. It does NOT replace the
// no-DistArray-lifetime rule (a leaked *constructor* still aborts); it only
// hardens the boundary against incidental runtime calls.
//
// **Spelling matters for how the plugin is loaded.** The bare attribute
// `__attribute__((enzyme_inactive))` / `[[enzyme::inactive]]` is parsed only by
// Enzyme's *clang frontend* plugin (`-fplugin=ClangEnzyme-*.so`). This
// integration loads Enzyme as an *LLVM pass* plugin only (`-fpass-plugin=`, see
// FindOrFetchEnzyme.cmake), so that spelling is silently dropped
// ("unknown attribute 'enzyme_inactive' ignored"). The pass-plugin-compatible
// form is the GNU `annotate` attribute: clang lowers it into
// `llvm.global.annotations`, and Enzyme's `PreserveNVVM` pass lifts
// `annotate("enzyme_inactive")` back into the real `enzyme_inactive` function
// attribute — no frontend plugin needed. That is what `TA_AD_INACTIVE` uses.
//
//   TA_AD_INACTIVE void some_runtime_probe(const TArrayD* x);  // at the decl
//
// For symbols you cannot annotate at their definition (e.g. MADNESS internals),
// register them by pointer via an `__enzyme_inactivefn` global instead — also
// honored by the LLVM pass (PreserveNVVM), so it works under `-fpass-plugin`:
//
//   void* __enzyme_inactivefn[] __attribute__((used)) = {(void*)&some_world_op};
// ---------------------------------------------------------------------------
#define TA_AD_INACTIVE __attribute__((annotate("enzyme_inactive")))

#endif  // TILEDARRAY_HAS_ENZYME

#endif  // TILEDARRAY_AD_ENZYME_H__INCLUDED
