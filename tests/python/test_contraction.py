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
class TestContraction:

    def test_matmul(self, Array, world):
        a = Array([4, 8], block=2, world=world)
        b = Array([8, 4], block=2, world=world)
        c = Array([4, 4], block=2, world=world)
        a.fill(1.0, False)
        b.fill(1.0, False)
        world.fence()
        c["i,j"] = a["i,k"] * b["k,j"]
        world.fence()
        buf = np.array(c)
        assert np.allclose(buf, 8.0)
        world.fence()

    def test_einsum_binary(self, Array, world):
        a = Array([4, 8], block=2)
        b = Array([8, 12], block=2)
        c = Array([4, 12], block=2)
        a.fill(1.0, False)
        b.fill(1.0, False)
        ta.einsum("ik,kj->ij", a, b, c)
        world.fence()
        buf = np.array(c)
        assert np.allclose(buf, 8.0)
        world.fence()

    def test_einsum_ternary(self, Array, world):
        a = Array([4, 8], block=2)
        b = Array([8, 12], block=2)
        c = Array([4, 12], block=2)
        d = Array()
        a.fill(1.0, False)
        b.fill(1.0, False)
        c.fill(1.0, False)
        ta.einsum("ik,kj,ab->ijab", a, b, c, d)
        world.fence()
