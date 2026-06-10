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
class TestNumPyInterop:

    def test_from_array(self, Array, world):
        data = np.ones((4, 6), dtype=np.float64)
        a = Array.from_array(data, [[0, 2, 4], [0, 3, 6]], world)
        world.fence()
        buf = np.array(a)
        assert buf.shape == (4, 6)
        assert np.allclose(buf, 1.0)
        world.fence()

    def test_from_array_values(self, Array, world):
        data = np.arange(12, dtype=np.float64).reshape(3, 4)
        a = Array.from_array(data, [[0, 3], [0, 4]], world)
        world.fence()
        buf = np.array(a)
        assert np.allclose(buf, data)
        world.fence()

    def test_to_array_via_buffer_protocol(self, Array, world):
        a = Array([4, 8], block=8, world=world)
        a.fill(1.0)
        world.fence()
        buf = np.array(a)
        assert buf.shape == a.shape
        assert np.allclose(buf, 1.0)
        world.fence()

    def test_init_elements(self, Array, world):
        a = Array([4, 4], 2, world)
        a.init_elements(lambda idx: float(idx[0] + idx[1]))
        world.fence()
        buf = np.array(a)
        assert buf.shape == (4, 4)
        for i in range(4):
            for j in range(4):
                assert abs(buf[i, j] - (i + j)) < 1e-10
        world.fence()

    def test_from_array_default_world(self, Array, world):
        data = np.ones((4, 4), dtype=np.float64)
        a = Array.from_array(data, [[0, 2, 4], [0, 2, 4]])
        world.fence()
        buf = np.array(a)
        assert np.allclose(buf, 1.0)
        world.fence()
