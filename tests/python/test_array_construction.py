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

import numpy as np
import pytest
import tiledarray as ta

ARRAY_TYPES = [ta.TArray, ta.TSpArray]
ARRAY_TYPE_IDS = ["TArray", "TSpArray"]


@pytest.mark.parametrize("Array", ARRAY_TYPES, ids=ARRAY_TYPE_IDS)
class TestArrayConstruction:

    def test_default_ctor(self, Array, world):
        a = Array()

    def test_shape_block_ctor(self, Array, world):
        a = Array([4, 4], 2, world)
        assert a.shape == (4, 4)
        world.fence()

    def test_shape_block_fill(self, Array, world):
        a = Array([4, 4], 2, world)
        a.fill(1, False)
        world.fence()

    def test_trange_list_ctor(self, Array, world):
        a = Array([[0, 2, 4], [0, 3, 6]])
        assert a.shape == (4, 6)
        world.fence()

    def test_trange_object_ctor(self, Array, world):
        tr = ta.TiledRange([[0, 2, 4], [0, 3, 6]])
        a = Array(tr, world=world)
        assert a.shape == (4, 6)
        world.fence()

    def test_ctor_with_op(self, Array, world):
        op = lambda r: np.ones(r.shape)
        a = Array([4, 4], 2, world=world, op=op)
        world.fence()
        buf = np.array(a)
        assert np.allclose(buf, 1.0)
        world.fence()

    def test_init(self, Array, world):
        a = Array([4, 8], block=2, world=world)
        a.init(lambda r: np.random.rand(*r.shape))
        world.fence()

    def test_shape_property(self, Array, world):
        a = Array([5, 5], 3, world)
        assert a.shape == (5, 5)
        world.fence()

    def test_trange_property(self, Array, world):
        a = Array([[0, 2, 4], [0, 3, 6]])
        tr = a.trange
        assert len(tr) == 2
        world.fence()

    def test_world_property(self, Array, world):
        a = Array([4, 4], 2, world)
        w = a.world
        assert w is not None
        world.fence()
