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
class TestArrayIndexing:

    def test_tile_setitem_getitem(self, Array, world):
        a = Array([4, 8], block=2, world=world)
        if world.rank == 0:
            a[0, 0] = np.ones([2, 2])
        world.fence()
        assert (a[0, 0] == np.ones([2, 2])).all()
        world.fence()

    def test_buffer_protocol(self, Array, world):
        a = Array([4, 8], block=8, world=world)
        a.fill(1)
        world.fence()
        b = np.array(a)
        assert b.shape == a.shape
        world.fence()

    def test_buffer_values(self, Array, world):
        a = Array([4, 4], block=2, world=world)
        a.fill(3.0)
        world.fence()
        buf = np.array(a)
        assert np.allclose(buf, 3.0)
        world.fence()

    def test_reference_index(self, Array, world):
        a = Array([4, 8], block=2, world=world)
        a.fill(1.0)
        world.fence()
        for tile in a:
            assert hasattr(tile, "index")
            break
        world.fence()

    def test_reference_range(self, Array, world):
        a = Array([4, 8], block=2, world=world)
        a.fill(1.0)
        world.fence()
        for tile in a:
            r = tile.range
            assert r.ndim == 2
            break
        world.fence()

    def test_reference_data_get(self, Array, world):
        a = Array([4, 8], block=2, world=world)
        a.fill(2.0)
        world.fence()
        for tile in a:
            data = tile.data
            assert isinstance(data, np.ndarray)
            assert np.allclose(data, 2.0)
            break
        world.fence()
