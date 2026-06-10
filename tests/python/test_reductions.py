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

import pytest
import tiledarray as ta

ARRAY_TYPES = [ta.TArray, ta.TSpArray]
ARRAY_TYPE_IDS = ["TArray", "TSpArray"]


@pytest.mark.parametrize("Array", ARRAY_TYPES, ids=ARRAY_TYPE_IDS)
class TestReductions:

    def test_norm(self, Array, world):
        c = Array([8, 8], block=2, world=world)
        c.fill(5.0, False)
        world.fence()
        # norm of an 8x8 array filled with 5: sqrt(64 * 25) = 40
        assert abs(c["i,j"].norm() - 40.0) < 1e-10
        world.fence()

    def test_squared_norm(self, Array, world):
        a = Array([4, 4], 2, world)
        a.fill(1.0, False)
        world.fence()
        sn = a["i,j"].squared_norm()
        assert abs(sn - 16.0) < 1e-10
        world.fence()

    def test_dot(self, Array, world):
        c = Array([8, 8], block=2, world=world)
        c.fill(5.0, False)
        world.fence()
        assert abs(c["i,j"].dot(c["j,i"]) - (5**2) * 64) < 1e-6
        world.fence()

    def test_min(self, Array, world):
        c = Array([8, 8], block=2, world=world)
        c.fill(5.0, False)
        world.fence()
        # c["i,j"] + c["j,i"] = 10 everywhere for a square array
        assert abs((c["i,j"] + c["j,i"]).min() - 10.0) < 1e-10
        world.fence()

    def test_max(self, Array, world):
        c = Array([8, 8], block=2, world=world)
        c.fill(5.0, False)
        world.fence()
        assert abs((c["i,j"] - c["j,i"]).max() - 0.0) < 1e-10
        world.fence()

    def test_abs_min(self, Array, world):
        c = Array([8, 8], block=2, world=world)
        c.fill(5.0, False)
        world.fence()
        assert abs(c["i,j"].abs_min() - 5.0) < 1e-10
        world.fence()

    def test_abs_max(self, Array, world):
        c = Array([8, 8], block=2, world=world)
        c.fill(5.0, False)
        world.fence()
        assert abs(c["i,j"].abs_max() - 5.0) < 1e-10
        world.fence()

    def test_sum(self, Array, world):
        a = Array([4, 4], 2, world)
        a.fill(1.0, False)
        world.fence()
        s = a["i,j"].sum()
        assert abs(s - 16.0) < 1e-10
        world.fence()

    def test_trace(self, Array, world):
        a = Array([4, 4], 2, world)
        a.fill(1.0, False)
        world.fence()
        t = a["i,i"].trace()
        assert abs(t - 4.0) < 1e-10
        world.fence()
