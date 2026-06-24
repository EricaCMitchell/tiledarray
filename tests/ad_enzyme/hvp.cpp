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
 *  hvp.cpp
 *  Phase-2 higher-order seam (enzyme_integration_suggestions.md P2.3).
 *
 *  FINDING — naive Enzyme-over-Enzyme nesting does NOT work here.
 *  ----------------------------------------------------------------
 *  The obvious route to a Hessian-vector product is to nest the drivers:
 *  `__enzyme_fwddiff` applied to a routine that itself calls
 *  `__enzyme_autodiff`. That was tried and does NOT compile in practical time:
 *  the outer forward pass differentiates Enzyme's *generated* reverse pass,
 *  whose calls land in our custom reverse/augmented rule SHIMS. To forward-
 *  differentiate those, Enzyme needs the *forward derivative of each reverse
 *  rule* — i.e. SECOND-ORDER custom rules — which we do not register. Lacking
 *  them, Enzyme inlines the reverse-rule bodies and tries to differentiate their
 *  raw `ad::contract`/`ad::add`/… calls, which expand into TiledArray's
 *  template-heavy MADWorld/BLAS runtime IR — exactly the IR the atomic-rule
 *  design exists to keep Enzyme out of. The result is a compile-time blow-up
 *  (observed: a single TU still in the Enzyme pass after >20 min at full CPU,
 *  versus ~2 min for the first-order TUs), not a clean diagnostic. First-order
 *  forward rules (P1.2) are necessary but NOT sufficient for nesting; the
 *  missing piece is the derivative-of-the-VJP, a Phase-3 undertaking.
 *
 *  SUPPORTED higher-order path (what this test exercises).
 *  -------------------------------------------------------
 *  The native AD layer composes forward-over-reverse directly: a reverse
 *  `Tape<Dual<Array>>` carries a forward tangent through the backward pass
 *  (dual.h, tests/ad_hvp.cpp), giving the gradient as the dual's primal and the
 *  Hessian-vector product as its tangent. This test pairs that with the Enzyme
 *  path: the *first-order gradient* is taken via Enzyme (`__enzyme_autodiff`,
 *  fast, the same rules energy.cpp/primitives.cpp validate) and the *curvature*
 *  Hv via the native dual tape, then cross-checks
 *    (1) Enzyme gradient  == native dual-gradient primal, and
 *    (2) native Hv         == central finite differences of the gradient.
 *  So the higher-order seam is proven end-to-end with Enzyme owning first order;
 *  closing it to Enzyme-only would require the second-order rules above.
 */

#include <cmath>
#include <cstdio>
#include <utility>

#include "tiledarray.h"

#include "TiledArray/ad/dual.h"
#include "TiledArray/ad/enzyme_rules.h"
#include "TiledArray/ad/enzyme.h"
#include "TiledArray/ad/ops.h"
#include "TiledArray/ad/tape.h"

using namespace TiledArray;
using RArray = TA::TArrayD;
using DualR = ad::Dual<RArray>;

int enzyme_dup;
int enzyme_const;
int enzyme_out;

// Square contraction X·X -> C, with annotations baked into the symbol (P0.3).
TA_AD_CONTRACT_RULE(RArray, sq_ik_kj_ij, "i,k", "k,j", "i,j");

namespace {

bool g_fail = false;
void check(bool ok, const char* what, double err, double tol) {
  if (TA::get_default_world().rank() == 0)
    std::printf("  %-40s err=%.3e  tol=%.1e  %s\n", what, err, tol,
                ok ? "ok" : "FAIL");
  if (!ok) g_fail = true;
}

RArray make_dense(World& world, const TiledRange& tr, double seed) {
  RArray a(world, tr);
  double n = seed;
  for (auto tidx : a.trange().tiles_range()) {
    auto range = a.trange().make_tile_range(tidx);
    RArray::value_type tile(range);
    for (std::size_t i = 0; i < tile.size(); ++i) {
      n = std::fmod(n * 1103515245.0 + 12345.0, 2147483648.0);
      tile[i] = (n / 2147483648.0) * 2.0 - 1.0;
    }
    a.set(tidx, tile);
  }
  world.gop.fence();
  return a;
}

double rel_diff(const RArray& x, const RArray& y) {
  const double dn = ad::norm2(ad::subt(x, y));
  const double xn = ad::norm2(x);
  return dn / (xn > 0 ? xn : 1.0);
}

// Native forward-over-reverse: ∇E as the primal of a Tape<Dual> gradient, the
// Hessian-vector product Hv as its tangent (mirrors tests/ad_hvp.cpp). A scalar
// reduction can't be pushed through a Tape<Dual> (it yields a DualScalar the
// scalar-seed backward() doesn't take), so — as in ad_hvp.cpp — we stop at
// C = X·X and seed the squared-norm cotangent C̄ = 2C directly.
//   E(X) = ‖X·X‖²
std::pair<RArray, RArray> grad_and_hvp_native(const RArray& X, const RArray& v) {
  ad::Tape<DualR> tape;
  auto vx = ad::make_leaf(tape, ad::make_dual(X, v));
  auto vc = ad::contract(vx, vx, "i,k", "k,j", "i,j");
  DualR seed = ad::scale(vc.value, 2.0);  // C̄ = d‖C‖²/dC = 2C (dual: tangent too)
  tape.backward(vc.id, seed);
  const DualR& g = tape.adjoint(vx.id).value();
  return {g.primal, g.has_tangent() ? *g.tangent : RArray()};
}

}  // namespace

// First-order gradient ∇E(X) via Enzyme reverse mode over the atomic custom
// rules — the same path validated by energy.cpp/primitives.cpp. Constraint #3:
// every primal/shadow is caller-allocated; the body holds no DistArray lifetime.
extern "C" void ta_E(const RArray* X, RArray* c1, double* e) {
  ta_ad_contract_sq_ik_kj_ij(X, X, c1);
  ta_ad_sqnorm_d(c1, e);
}

int main(int argc, char** argv) {
  World& world = TA::initialize(argc, argv);

  const TiledRange tr{{0, 2, 4}, {0, 2, 4}};  // square, for X·X
  RArray X = make_dense(world, tr, 3.0);
  RArray v = make_dense(world, tr, 41.0);

  if (world.rank() == 0)
    std::printf("[HVP] higher-order seam: Enzyme gradient + native curvature (P2.3)\n");

  // Enzyme first-order gradient g = ∇E(X).
  RArray c1, dc1, dX;          // dX = symbolic-zero shadow seeded by the reverse pass
  double e = 0.0, de = 1.0;
  __enzyme_autodiff((void*)ta_E, TA_AD_DUP(X, dX), TA_AD_DUP(c1, dc1),
                    TA_AD_DUP(e, de));
  world.gop.fence();

  // Native forward-over-reverse: gradient primal + Hessian-vector tangent.
  auto [g_native, hv_native] = grad_and_hvp_native(X, v);

  check(rel_diff(dX, g_native) < 1e-9, "gradient: enzyme vs native dual primal",
        rel_diff(dX, g_native), 1e-9);

  // Hv vs central finite differences of the (native) gradient along v.
  const double eps = 1e-5;
  auto [gp, hp] = grad_and_hvp_native(ad::add(X, ad::scale(v, eps)), v);
  auto [gm, hm] = grad_and_hvp_native(ad::add(X, ad::scale(v, -eps)), v);
  RArray fd = ad::scale(ad::subt(gp, gm), 1.0 / (2 * eps));
  check(rel_diff(hv_native, fd) < 1e-5, "HVP: native tangent vs fd-of-gradient",
        rel_diff(hv_native, fd), 1e-5);

  if (world.rank() == 0) std::printf("RESULT: %s\n", g_fail ? "FAIL" : "PASS");
  TA::finalize();
  return g_fail ? 1 : 0;
}
