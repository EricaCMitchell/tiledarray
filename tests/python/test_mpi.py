#  This file is a part of TiledArray.
#  Copyright (C) 2020  Virginia Tech
#
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program.  If not, see <http://www.gnu.org/licenses/>.

"""Multi-rank tests.  Run via: mpirun -n 2 python -m pytest test_mpi.py"""

import numpy as np
import pytest
import tiledarray as ta

ARRAY_TYPES = [ta.TArray, ta.TSpArray]
ARRAY_TYPE_IDS = ["TArray", "TSpArray"]


@pytest.mark.parametrize("Array", ARRAY_TYPES, ids=ARRAY_TYPE_IDS)
class TestMPI:

    def test_world_size_at_least_one(self, Array, world):
        assert world.size >= 1

    def test_distributed_fill_gather(self, Array, world):
        a = Array([8, 8], block=2, world=world)
        a.fill(1.0, False)
        world.fence()
        buf = np.array(a)
        assert buf.shape == (8, 8)
        assert np.allclose(buf, 1.0)
        world.fence()

    def test_remote_tile_index_access(self, Array, world):
        # Every rank must be able to read every tile by index, including tiles
        # owned by another rank.  find()+get() drives the point-to-point
        # transfer; this is the per-tile path the numpy bridge relies on.
        a = Array([8, 8], block=4, world=world)
        a.fill(3.0, False)
        world.fence()
        for i in range(2):
            for j in range(2):
                tile = np.array(a[i, j])
                assert tile.shape == (4, 4)
                assert np.allclose(tile, 3.0)
        world.fence()

    def test_numpy_diagonal_is_replicated(self, Array, world):
        # Regression for the multi-rank failure: numpy.array(dist).diagonal()
        # must yield the full diagonal identically on every rank, not an empty
        # 0-d array on non-owning ranks.
        a = Array([6, 6], block=3, world=world)
        a.fill(0.0, True)  # skip_set: leave structure, fill explicit below
        world.fence()
        eye = np.eye(6)
        b = Array.from_array(eye, [[0, 3, 6], [0, 3, 6]], world)
        world.fence()
        diag = np.array(b).diagonal()
        assert diag.shape == (6,)
        assert np.allclose(diag, 1.0)
        world.fence()

    def test_sparse_zero_tile_index_raises(self, Array, world):
        # Indexing an absent tile must raise, not hang.  fill(0.0) on a sparse
        # array stores no tiles at all (below-threshold tiles are dropped), so
        # find()+get() on any index would otherwise block forever on a future
        # nothing sets.  Dense arrays store every tile, so the guard is a no-op
        # there and the index simply returns zeros.
        a = Array([6, 6], block=3, world=world)
        a.fill(0.0, False)
        world.fence()
        if Array is ta.TSpArray:
            assert a.is_zero([0, 0])
            with pytest.raises(RuntimeError):
                _ = a[0, 0]
        else:
            assert not a.is_zero([0, 0])
            assert np.allclose(np.array(a[0, 0]), 0.0)
        world.fence()

    def test_distributed_contraction(self, Array, world):
        a = Array([8, 8], block=4, world=world)
        b = Array([8, 8], block=4, world=world)
        c = Array([8, 8], block=4, world=world)
        a.fill(1.0, False)
        b.fill(1.0, False)
        world.fence()
        c["i,j"] = a["i,k"] * b["k,j"]
        world.fence()
        buf = np.array(c)
        assert np.allclose(buf, 8.0)
        world.fence()

    def test_from_array_distributed(self, Array, world):
        data = np.ones((8, 8), dtype=np.float64)
        a = Array.from_array(data, [[0, 4, 8], [0, 4, 8]], world)
        world.fence()
        buf = np.array(a)
        assert np.allclose(buf, 1.0)
        world.fence()

    def test_rank_range(self, Array, world):
        assert 0 <= world.rank < world.size
