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
class TestLifetime:

    def test_clone_is_deep_copy(self, Array, world):
        # DistArray assignment is a shallow handle; clone() is a deep copy.
        a = Array([4, 4], 2, world)
        a.fill(1.0, False)
        world.fence()
        b = a.clone()
        b.fill(2.0, False)
        world.fence()
        a_buf = np.array(a)
        b_buf = np.array(b)
        assert np.allclose(a_buf, 1.0)
        assert np.allclose(b_buf, 2.0)
        world.fence()

    def test_truncate(self, Array, world):
        a = Array([4, 4], 2, world)
        a.fill(1.0, False)
        world.fence()
        a.truncate()
        world.fence()

    def test_delete_original_after_clone(self, Array, world):
        a = Array([4, 4], 2, world)
        a.fill(3.0, False)
        world.fence()
        b = a.clone()
        del a
        world.fence()
        buf = np.array(b)
        assert np.allclose(buf, 3.0)
        world.fence()
