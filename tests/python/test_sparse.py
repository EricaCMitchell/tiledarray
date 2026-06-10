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
import tiledarray as ta


class TestSparseThreshold:

    def test_get_threshold(self):
        t = ta.get_sparse_threshold()
        assert isinstance(t, float)
        assert t >= 0.0

    def test_set_threshold(self):
        original = ta.get_sparse_threshold()
        ta.set_sparse_threshold(1e-5)
        assert abs(ta.get_sparse_threshold() - 1e-5) < 1e-12
        ta.set_sparse_threshold(original)

    def test_threshold_round_trip(self):
        original = ta.get_sparse_threshold()
        ta.set_sparse_threshold(1e-3)
        ta.set_sparse_threshold(original)
        assert abs(ta.get_sparse_threshold() - original) < 1e-12


class TestSparseDenseSymmetry:

    def test_is_dense_true(self, world):
        a = ta.TArray([4, 4], 2, world)
        assert a.is_dense

    def test_is_dense_false(self, world):
        a = ta.TSpArray([4, 4], 2, world)
        assert not a.is_dense

    def test_is_zero_dense(self, world):
        a = ta.TArray([4, 4], 2, world)
        a.fill(1.0, False)
        world.fence()
        result = a.is_zero([0, 0])
        assert isinstance(result, bool)
        world.fence()

    def test_is_zero_sparse_nonzero(self, world):
        a = ta.TSpArray([4, 4], 2, world)
        a.fill(1.0, False)
        world.fence()
        result = a.is_zero([0, 0])
        assert isinstance(result, bool)
        world.fence()

    def test_sparse_fill_zero_yields_empty(self, world):
        # fill(0) on a TSpArray yields an empty array because no tiles exceed threshold.
        # This is a documented sparse footgun — see CLAUDE.md.
        a = ta.TSpArray([4, 4], 2, world)
        a.fill(0.0)
        world.fence()
        buf = np.array(a)
        assert np.allclose(buf, 0.0)
        world.fence()

    def test_dense_fill_zero_yields_zeros(self, world):
        a = ta.TArray([4, 4], 2, world)
        a.fill(0.0)
        world.fence()
        buf = np.array(a)
        assert np.allclose(buf, 0.0)
        world.fence()
