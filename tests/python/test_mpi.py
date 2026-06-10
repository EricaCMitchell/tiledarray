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
