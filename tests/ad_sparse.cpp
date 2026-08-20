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
 *  ad_sparse.cpp
 *  Block-sparse cotangent sparsity: deterministic structure, and values that
 *  agree with a dense recompute.
 */

#include "tiledarray.h"
#include "unit_test_config.h"

#include "TiledArray/conversions/sparse_to_dense.h"

#include "TiledArray/ad/ops.h"
#include "TiledArray/ad/tape.h"

using namespace TiledArray;

namespace {

using SpArray = TA::TSpArray<double>;   // DistArray<Tensor<double>, SparsePolicy>
using DnArray = TA::TArray<double>;

// Build a block-sparse array whose nonzero-tile pattern is given by `nonzero`
// (a predicate over the tile multi-index). Nonzero tiles are filled with random
// values; zero tiles are absent.
template <typename Pred>
SpArray rand_sparse(const TiledRange& tr, Pred nonzero) {
  Tensor<float> shape_tensor(tr.tiles_range(), 0.0f);
  for (auto it = tr.tiles_range().begin(); it != tr.tiles_range().end(); ++it)
    if (nonzero(*it)) shape_tensor[*it] = 1.0f;
  SpArray a(*GlobalFixture::world, tr,
            TiledArray::SparseShape<float>(shape_tensor, tr));
  for (auto idx : *a.pmap()) {
    if (a.is_zero(idx)) continue;
    auto range = a.trange().make_tile_range(idx);
    Tensor<double> tile(range);
    for (std::size_t i = 0; i < tile.size(); ++i)
      tile[i] = (int(GlobalFixture::world->rand() % 2001) - 1000) / 1000.0;
    a.set(idx, tile);
  }
  GlobalFixture::world->gop.fence();
  return a;
}

// squared 2-norm of (x - y) where both are densified first.
double dense_diff_sqnorm(const SpArray& x, const SpArray& y) {
  DnArray xd = TA::to_dense(x), yd = TA::to_dense(y);
  DnArray d;
  d("i,j") = xd("i,j") - yd("i,j");
  return d("i,j").squared_norm().get();
}

struct SparseFixture {
  // 3 tiles in each mode so a single zero tile leaves a nontrivial pattern.
  TiledRange trA{{0, 2, 4, 6}, {0, 2, 4, 6}};  // (i, k)
  TiledRange trB{{0, 2, 4, 6}, {0, 2, 4, 6}};  // (k, j)
  TiledRange trC{{0, 2, 4, 6}, {0, 2, 4, 6}};  // (i, j)
  std::string aA = "i,k", aB = "k,j", aC = "i,j";

  float saved_threshold;
  SparseFixture()
      : saved_threshold(TiledArray::SparseShape<float>::threshold()) {
    // binary sparsity: any nonzero norm clears the bar
    TiledArray::SparseShape<float>::threshold(
        std::numeric_limits<float>::min());
  }
  ~SparseFixture() {
    TiledArray::SparseShape<float>::threshold(saved_threshold);
  }
};

}  // namespace

BOOST_FIXTURE_TEST_SUITE(ad_sparse_suite, SparseFixture)

// The contract VJP on block-sparse operands must agree with the dense
// recompute. The sparsity structure of the gradient must not change between
// runs.
BOOST_AUTO_TEST_CASE(contract_vjp_block_sparse) {
  auto patA = [](const auto& idx) { return !(idx[0] == 0 && idx[1] == 1); };
  auto patB = [](const auto& idx) { return !(idx[0] == 2 && idx[1] == 0); };
  auto patC = [](const auto&) { return true; };

  SpArray A = rand_sparse(trA, patA);
  SpArray B = rand_sparse(trB, patB);
  SpArray Cbar = rand_sparse(trC, patC);

  auto run = [&](const SpArray& a, const SpArray& b, const SpArray& cbar) {
    ad::Tape<SpArray> tape;
    auto va = ad::make_leaf(tape, a), vb = ad::make_leaf(tape, b);
    auto vc = ad::contract(va, vb, aA, aB, aC);
    tape.backward(vc.id, cbar);
    return std::make_pair(tape.adjoint(va.id).value(),
                          tape.adjoint(vb.id).value());
  };

  auto [Abar, Bbar] = run(A, B, Cbar);

  // (1) Determinism: a second independent reverse pass yields identical
  //     structure (same nonzero-tile set) AND identical values.
  auto [Abar2, Bbar2] = run(A, B, Cbar);
  auto same_structure = [](const SpArray& x, const SpArray& y) {
    for (auto it = x.tiles_range().begin(); it != x.tiles_range().end(); ++it)
      if (x.is_zero(*it) != y.is_zero(*it)) return false;
    return true;
  };
  BOOST_CHECK(same_structure(Abar, Abar2));
  BOOST_CHECK(same_structure(Bbar, Bbar2));
  BOOST_CHECK_SMALL(dense_diff_sqnorm(Abar, Abar2), 1e-30);
  BOOST_CHECK_SMALL(dense_diff_sqnorm(Bbar, Bbar2), 1e-30);

  // (2) Value-correctness vs a dense recompute of the same VJP.
  DnArray Ad = TA::to_dense(A), Bd = TA::to_dense(B),
          Cbard = TA::to_dense(Cbar);
  ad::Tape<DnArray> dtape;
  auto dva = ad::make_leaf(dtape, Ad), dvb = ad::make_leaf(dtape, Bd);
  auto dvc = ad::contract(dva, dvb, aA, aB, aC);
  dtape.backward(dvc.id, Cbard);
  const DnArray& Abar_d = dtape.adjoint(dva.id).value();
  const DnArray& Bbar_d = dtape.adjoint(dvb.id).value();

  DnArray dA, dB;
  dA("i,j") = TA::to_dense(Abar)("i,j") - Abar_d("i,j");
  dB("i,j") = TA::to_dense(Bbar)("i,j") - Bbar_d("i,j");
  BOOST_CHECK_SMALL(dA("i,j").squared_norm().get(), 1e-18);
  BOOST_CHECK_SMALL(dB("i,j").squared_norm().get(), 1e-18);
}

BOOST_AUTO_TEST_SUITE_END()
