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
class TestExpression:

    def test_scalar_mul(self, Array, world):
        a = Array([4, 4], 2, world)
        a.fill(1.0, False)
        b = Array([4, 4], 2, world)
        b["i,j"] = 2.0 * a["i,j"]
        world.fence()
        buf = np.array(b)
        assert np.allclose(buf, 2.0)
        world.fence()

    def test_scalar_div(self, Array, world):
        a = Array([4, 4], 2, world)
        a.fill(4.0, False)
        b = Array([4, 4], 2, world)
        b["i,j"] = a["i,j"] / 2.0
        world.fence()
        buf = np.array(b)
        assert np.allclose(buf, 2.0)
        world.fence()

    def test_add(self, Array, world):
        a = Array([4, 4], 2, world)
        a.fill(1.0, False)
        b = Array([4, 4], 2, world)
        b["i,j"] = a["i,j"] + a["i,j"]
        world.fence()
        buf = np.array(b)
        assert np.allclose(buf, 2.0)
        world.fence()

    def test_sub(self, Array, world):
        a = Array([4, 4], 2, world)
        a.fill(3.0, False)
        c = Array([4, 4], 2, world)
        c["i,j"] = a["i,j"] - a["i,j"]
        world.fence()

    def test_neg(self, Array, world):
        a = Array([4, 4], 2, world)
        a.fill(1.0, False)
        b = Array([4, 4], 2, world)
        b["i,j"] = -a["i,j"]
        world.fence()
        buf = np.array(b)
        assert np.allclose(buf, -1.0)
        world.fence()

    def test_permutation(self, Array, world):
        a = Array([4, 8], block=2, world=world)
        a.fill(1.0, False)
        b = Array([8, 4], block=2, world=world)
        b["i,j"] = a["j,i"]
        world.fence()
        buf = np.array(b)
        assert np.allclose(buf, 1.0)
        world.fence()

    def test_combined_scalar_ops(self, Array, world):
        a = Array([4, 8], block=2, world=world)
        a.fill(1.0, False)
        b = Array([8, 4], block=2, world=world)
        b["i,j"] = a["j,i"] - a["j,i"] / 2 + 6 * a["j,i"] * 2
        world.fence()
